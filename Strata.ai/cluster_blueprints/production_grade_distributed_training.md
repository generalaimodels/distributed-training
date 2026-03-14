# Technical Report: Hardware-Specific LLM Training Deployment, Megatron-Core/DeepSpeed/FSDP Interoperability, and Long-Context + MoE Distributed Architecture

## 1. Design Rules

> **Rule 1:** “Put the most communication-intensive parallel dimension on the fastest fabric.”

> **Rule 2:** “One authority per sharding dimension. Never stack two parameter-sharding systems over the same tensor set.”

> **Rule 3:** “Checkpoint conversion must operate on logical tensors, not rank-local shards.”

> **Rule 4:** “A model fits only if the smallest-HBM rank fits.”

> **Rule 5:** “No throughput claim is valid without step-time decomposition, overlap accounting, and loss-parity validation.”

This report covers three tightly coupled deliverables:

- **A. Hardware-specific deployment blueprint** for $A100$, $H100$, $B200$-class, $MI300X$, and $MI350$-class clusters.
- **B. Megatron-Core + DeepSpeed + FSDP interoperability** across training, resume, resharding, and conversion.
- **C. Long-context and MoE training architecture** with exact parallel-group factorization, communication reasoning, and memory formulas.

The report is written from the perspective of a Principal Distributed Training Engineer and is implementation-focused throughout.

---

## 2. System Model and Notation

## 2.1 Parallel Dimensions

For dense models, the primary world-size factorization is:

$$
W = D \times T \times P \times C
$$

Where:

- $W$ = total accelerator count
- $D$ = data parallel or FSDP replica count
- $T$ = tensor parallel degree
- $P$ = pipeline stage count
- $C$ = context parallel degree

For MoE models with separate dense and expert tensor sharding:

$$
W = D \times P \times C \times T_d \times E \times T_e
$$

Where:

- $T_d$ = dense-path tensor parallel degree
- $E$ = expert parallel degree
- $T_e$ = expert tensor parallel degree

### Important clarification

- **Sequence Parallelism** is usually **coupled to $T$**, not an independent multiplicative axis.
- **Context Parallelism** is an independent axis over sequence partitioning.
- **FSDP / ZeRO** typically shard over the $D$ dimension.
- **Expert Parallelism** typically applies only to expert parameters, not the dense trunk.

---

## 2.2 Global Batch and Token Throughput

If each data-parallel replica processes $N_\mu$ microbatches of size $B_\mu$ per optimizer step, then:

$$
B_{\text{global}} = D \times N_\mu \times B_\mu
$$

For sequence length $S$:

$$
N_{\text{tokens/step}} = B_{\text{global}} \times S
$$

Throughput is:

$$
\text{tokens/s} = \frac{N_{\text{tokens/step}}}{T_{\text{step}}}
$$

For pipeline parallelism with non-interleaved $1F1B$, approximate bubble efficiency is:

$$
\eta_{\text{PP}} \approx \frac{N_\mu}{N_\mu + P - 1}
$$

This formula must be used during planning. If $N_\mu$ is too small, pipeline bubbles dominate and exposed communication becomes much harder to hide.

---

## 2.3 Collective Cost Model

For a message of size $n$ bytes:

$$
T_m(n) = \alpha + \beta n
$$

For a ring collective over group size $g$:

$$
T_{\text{AG}} \approx (g-1)\alpha + \frac{g-1}{g} n \beta
$$

$$
T_{\text{RS}} \approx (g-1)\alpha + \frac{g-1}{g} n \beta
$$

$$
T_{\text{AR}} \approx 2(g-1)\alpha + 2\frac{g-1}{g} n \beta
$$

And:

$$
\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}
$$

This identity is operationally central for FSDP, ZeRO-2, ZeRO-3, tensor parallel gradients, and sequence-parallel activation exchange.

---

# Part A. Hardware-Specific Deployment Blueprint

## 3. Hardware-Class Characteristics

## 3.1 Platform-Class View

| Platform class | Local accelerator fabric | Inter-node fabric | Default training precision | Best-use characteristics | Primary risks |
|---|---|---:|---|---|---|
| $A100$ HGX class | NVLink / NVSwitch | IB / RoCE | BF16 | mature kernels, strong BF16, stable compiler/runtime stack | lower low-precision headroom than newer platforms, comm becomes visible sooner |
| $H100$ HGX class | NVLink / NVSwitch | NDR-class IB / RoCE | BF16, FP8 where qualified | excellent local collectives, strong Transformer kernel stack, production FP8 viable | over-sharding small models can reduce MFU |
| $B200$-class | next-gen NVLink domain | high-bandwidth IB / RoCE | BF16, FP8/MXFP8 where production-qualified | more aggressive TP/CP/EP viable, faster low-precision kernels | toolchain immaturity risk during early adoption |
| $MI300X$ class | xGMI | IB / RoCE | BF16 | very large HBM, strong local memory capacity, can reduce PP pressure | RCCL topology tuning and HIP kernel coverage must be validated |
| $MI350$-class | next-gen xGMI-class domain | high-bandwidth IB / RoCE | BF16, FP8 where production-qualified | larger local-group opportunities, stronger low-precision potential | compiler/kernel maturity must be re-qualified per ROCm release |

### Engineering implication

- **Use local high-bandwidth fabrics for $T$, $C$, $E$, and $T_e$.**
- **Use inter-node fabrics primarily for $D$ and sometimes $P$.**
- **Do not place latency-sensitive fine-grained collectives across the slowest links unless the model size makes that unavoidable.**

---

## 3.2 Topology-Aware Placement Rules

### Preferred locality by parallel dimension

| Parallel axis | Communication frequency | Message size pattern | Placement priority |
|---|---|---|---|
| $T$ | every layer | medium to large | strict local fast fabric |
| Sequence parallel | every layer | activation-sized | same group as $T$ |
| $C$ | every attention layer | activation/KV-sized | local fast fabric first |
| $E$ | every MoE layer | token dispatch all-to-all | local fast fabric first |
| $T_e$ | every expert MLP | expert-weight and activation sized | local fast fabric |
| $P$ | every stage boundary | activation-sized | cross-node acceptable if stage balance is good |
| $D$ / FSDP / ZeRO | every wrapped unit or step | parameter/gradient sized | widest remaining domain |

### Practical mapping rule

On $8$-GPU nodes:

- Keep $T$, $C$, and $E$ entirely inside the node if possible.
- If using both $E$ and $T_e$, pack $E \times T_e$ inside a single node or within one fully connected local island.
- Stretch $P$ across nodes before stretching $T$ or $C$ across nodes.
- Use $D$ or FSDP across the full cluster.

---

## 3.3 Hardware-Specific Deployment Starting Points

The table below gives **starting configurations**, not absolute prescriptions. Final factorization must be derived from model size, sequence length, optimizer state, precision, and actual topology discovery.

### 3.3.1 Dense Pretraining Starting Points

| Platform | Recommended first dense layout | Notes |
|---|---|---|
| $A100$ 80 GB class | $T=2$ to $4$, $P=2$ to $8$, $C=1$ to $2$, FSDP or ZeRO over $D$ | use BF16, FlashAttention, sequence parallel with $T>1$, aggressive activation checkpointing |
| $H100$ class | $T=4$ to $8$, $P=2$ to $8$, $C=1$ to $4$, FSDP or distributed optimizer over $D$ | FP8 feasible after parity; larger intra-node $T$ practical |
| $B200$-class | $T=4$ to $8$, $C=2$ to $8$, $P=1$ to $6$ | higher local bandwidth reduces need for deep $P$; use stronger overlap and FP8/MXFP8 only after validation |
| $MI300X$ | $T=2$ to $4$, $P=1$ to $4$, $C=1$ to $4$, FSDP over $D$ | large HBM often allows fewer pipeline stages |
| $MI350$-class | $T=2$ to $8$, $P=1$ to $4$, $C=2$ to $8$ where stack is qualified | validate FP8 path and RCCL collectives before scaling |

### 3.3.2 Long-Context Starting Points

| Platform | Long-context recommendation | Notes |
|---|---|---|
| $A100$ | FlashAttention + full checkpointing + $C=2$ to $4$ | avoid oversized $T$; communication can dominate |
| $H100$ | FlashAttention + $C=2$ to $8$ + Ulysses or Ring Attention where required | best platform for aggressive long-context training |
| $B200$-class | $C=4$ to $8$ with low-precision attention kernels | validate numerical stability on long sequences |
| $MI300X$ | leverage HBM first, then $C=2$ to $4$ | use BF16 baseline first; validate fused attention coverage |
| $MI350$-class | $C=2$ to $8$ where kernels exist | same strategy as $MI300X$, with stronger local-group potential |

### 3.3.3 MoE Starting Points

| Platform | MoE recommendation | Notes |
|---|---|---|
| $A100$ | $E=4$ to $8$ local, $T_e=1$ to $2$ if needed | keep token all-to-all on-node if possible |
| $H100$ | $E=8$ local, $T_e=1$ to $2$, grouped GEMM, top-$2$ routing | best fit for aggressive EP |
| $B200$-class | $E=8$ local, $T_e=1$ to $4$ | exploit stronger local bandwidth and low-precision |
| $MI300X$ | $E=4$ to $8$ local with BF16, $T_e$ conservative | large HBM makes expert residency easier |
| $MI350$-class | $E=4$ to $8$, $T_e=1$ to $4$ if kernels qualify | validate HIP grouped GEMM and dispatcher efficiency |

---

## 4. Hardware Bring-Up Blueprint

## 4.1 Preflight Topology Discovery

Before any launch, discover:

- GPU adjacency matrix
- local-switch island topology
- GPU-to-NIC affinity
- CPU NUMA domains
- PCIe root complex layout
- NVLink/NVSwitch or xGMI connectivity
- IB / RoCE rail topology
- GPUDirect RDMA capability
- storage mount locality and bandwidth

### Mandatory outputs

- rank-to-GPU map
- rank-to-NIC map
- communicator groups by axis: $T$, $P$, $C$, $E$, $D$
- expected collective path classification:
  - local only
  - local-switch
  - cross-socket
  - cross-node single rail
  - cross-node dual rail

---

## 4.2 Pseudocode: Topology-Aware Parallelism Synthesis

```text
Input:
  cluster_topology
  model_profile
  target_workload
  candidate_parallel_tuples
  per-rank HBM budgets

Output:
  selected tuple (D, T, P, C, E, T_e)
  rank mapping
  communication placement plan

Procedure:
  1. Build topology graph:
       - vertices: GPUs
       - edges: local links, switch links, NIC-linked paths
       - weights: inverse bandwidth + latency penalty

  2. Enumerate feasible tuples:
       - require W divisibility
       - require layer-count divisibility for P
       - require attention and expert kernels for requested T, C, E, T_e
       - reject tuples violating minimum microbatch constraints

  3. For each tuple:
       - estimate persistent memory per rank
       - estimate transient activation and communication-buffer memory
       - reject if any rank exceeds its local HBM budget
       - score axis placement cost:
           cost(T) + cost(C) + cost(E) weighted highest
           cost(P) weighted medium
           cost(D) weighted lower
       - estimate exposed communication after overlap windows

  4. Select tuple with:
       - all ranks fitting in HBM
       - minimal weighted communication cost
       - acceptable PP bubble efficiency
       - maximal estimated tokens/s under stability constraints

  5. Emit deterministic rank ordering:
       - contiguous local ranks for T/C/E where possible
       - stage-local contiguous rank blocks for P
       - D/FSDP across nodes last

  6. Persist topology fingerprint and selected tuple into run metadata.
```

---

## 4.3 Communication Tuning Priorities by Platform

## 4.3.1 NVIDIA Clusters

Primary concerns:

- NCCL ring/tree path selection
- NVLink/NVSwitch saturation
- IB rail utilization
- GPUDirect RDMA enablement
- GPU/NIC NUMA mismatch
- collective launch serialization

High-value controls:

- gradient bucketing and tensor fusion
- hierarchical collectives when cross-node latency dominates
- overlap of all-gather and reduce-scatter with compute
- CUDA Graph capture for stable launch overhead reduction
- fused kernels to reduce HBM pressure

## 4.3.2 AMD Clusters

Primary concerns:

- RCCL ring construction and xGMI locality
- HIP kernel parity with CUDA baselines
- graph capture maturity by ROCm version
- dispatch overhead and grouped GEMM efficiency
- NIC and xGMI locality correctness

High-value controls:

- RCCL topology validation before blame assignment to the model
- BF16-first baseline before introducing FP8
- selective fallback to portable kernels if Triton/HIP coverage is incomplete
- xGMI-local placement for $T$, $C$, $E$

---

## 4.4 Bucket Sizing Strategy

Bucket sizing is not a static constant. It must satisfy both latency-hiding and memory-fit constraints.

For a bucket of size $n$ bytes in group size $g$:

$$
T_{\text{bucket}} \approx (g-1)\alpha + \frac{g-1}{g} n \beta
$$

A bucket is acceptable only if:

$$
T_{\text{bucket}} \le T_{\text{available overlap}}
$$

and:

$$
M_{\text{current peak}} + M_{\text{bucket}} \le M_{\max}
$$

### Engineering rule

- Increase bucket size until launch overhead is amortized.
- Stop increasing when:
  - overlap disappears,
  - HBM peaks rise,
  - copy-out cost becomes visible,
  - or collective tails widen.

This applies to:

- FSDP all-gather buckets
- FSDP reduce-scatter buckets
- gradient buckets
- tensor-parallel activation fusion buffers

---

## 4.5 Heterogeneous HBM Constraint

A cluster with mixed HBM capacities fits only if:

$$
M_{\text{peak},r} \le M_{\text{HBM},r} \quad \forall r
$$

### Consequence

- In pure FSDP or pure DP layouts, the **smallest-HBM rank** is the true capacity limiter.
- In pipeline-parallel layouts, larger-HBM nodes should host:
  - embeddings,
  - LM head,
  - MoE-heavy stages,
  - stages with larger transient gathers.

---

# Part B. Megatron-Core + DeepSpeed + FSDP Interoperability Report

## 5. Ownership Model

Interoperability becomes tractable only when each subsystem owns a clearly defined concern.

| Subsystem | Best ownership scope |
|---|---|
| Megatron-Core | $T$, sequence parallel, $P$, $C$, $E$, expert dispatch schedules, fused kernels |
| DeepSpeed | ZeRO optimizer partitioning, optimizer runtime, offload, some activation and comm orchestration |
| PyTorch FSDP / DTensor | parameter, gradient, optimizer-state sharding with logical mesh control |
| Compiler-native FSDP path | graph-visible communication bucketing/reordering for FSDP-style execution |

### Non-negotiable rule

> **Do not allow both ZeRO-3 and FSDP to shard the same parameter set simultaneously.**

That creates ambiguous ownership over:

- parameter materialization,
- optimizer-state locality,
- checkpoint metadata,
- resume semantics.

---

## 6. Recommended Interoperability Patterns

## 6.1 Recommended Patterns

| Pattern | Recommended | Why |
|---|---|---|
| Megatron-Core only + distributed optimizer | Yes | simplest ownership, strongest TP/PP/CP/EP support |
| Megatron-Core + DeepSpeed ZeRO-1/2 | Yes | optimizer/grad partitioning without duplicating parameter-shard ownership |
| Megatron-Core + stage-local FSDP on non-TP modules | Conditional | acceptable if parameter ownership boundaries are explicit |
| PyTorch FSDP/DTensor + torch.compile + vendor-portable kernels | Yes | strongest portability and simpler stack for non-frontier multidimensional training |
| DeepSpeed ZeRO-3 as sole sharding owner | Yes | valid if not mixing with outer FSDP on same tensors |

## 6.2 Discouraged or Unsafe Patterns

| Pattern | Status | Failure mode |
|---|---|---|
| Megatron TP/PP + DeepSpeed ZeRO-3 + outer full-model FSDP | Avoid | double or triple ownership of parameter shards |
| Changing $T$ or $E$ on resume without canonical resharding | Unsafe | incompatible shard geometry |
| Rank-local checkpoint copy between incompatible layouts | Unsafe | silent parameter misplacement |
| MoE expert remap without corresponding optimizer-state remap | Unsafe | optimizer corruption |
| Different compiler graph partitioning across ranks | Unsafe | collective desynchronization and deadlocks |

---

## 7. Checkpoint Semantics and Canonical Format

## 7.1 Canonical Checkpoint Metadata

Every logical tensor must carry:

- logical tensor name
- logical shape
- dtype
- tensor class:
  - dense trunk
  - embedding
  - router
  - expert
  - optimizer moment
  - master weight
- source sharding axes
- shard offsets and extents
- owning mesh dimensions
- optimizer-state mapping
- step and scheduler state
- RNG states
- data-loader cursor state
- tokenizer and dataset lineage fingerprints

### Required invariant

Checkpoint validity depends on reconstructability of the **logical tensor**, not on preservation of the original rank count.

---

## 7.2 What Can Change on Resume

| Axis / attribute | Can change without resharding? | Notes |
|---|---|---|
| $D$ / FSDP degree | Yes, with canonical or reshard-aware checkpoint | standard case |
| ZeRO stage | Yes, if optimizer states are canonicalized | conversion required |
| $C$ | Usually yes | $C$ partitions activations, not parameters |
| sequence parallel | Usually yes | runtime scheduling feature tied to $T$ |
| $T$ | No, not without tensor-axis resharding | dense tensor partition geometry changes |
| $P$ | No, not without stage repartitioning | stage ownership changes |
| $E$ | No, not without expert ownership remap | expert tensor locality changes |
| $T_e$ | No, not without expert-weight resharding | expert GEMM partition changes |
| dtype | Conditional | requires explicit cast policy and optimizer-state handling |

---

## 7.3 Pseudocode: Canonical Checkpoint Resharding

```text
Input:
  source_checkpoint
  source_mesh
  target_mesh
  conversion_policy

Output:
  target_checkpoint

Procedure:
  1. Read checkpoint metadata and validate integrity:
       - tensor names
       - shapes
       - dtypes
       - shard offsets
       - optimizer-state presence
       - step counters and RNG states

  2. For each logical tensor:
       a. Reconstruct the logical tensor view from source shards.
       b. If dtype conversion is requested, apply policy:
            - parameters to target training dtype
            - optimizer moments preserved or converted explicitly
       c. Partition the logical tensor according to target mesh:
            - TP split along declared tensor axis
            - PP assign by stage ownership
            - EP assign by expert ownership
            - FSDP/ZeRO shard over target D
       d. Emit target shards with new metadata.

  3. Repartition optimizer states with the same logical-to-physical mapping.

  4. Recompute and store:
       - tensor checksums
       - per-tensor statistics
       - mapping manifest
       - conversion provenance

  5. Validate target checkpoint:
       - complete tensor coverage
       - no overlapping shard ranges
       - no missing optimizer states
       - step and RNG continuity preserved
```

---

## 8. FSDP and ZeRO Interoperability Guidance

## 8.1 State Sharding Ownership

| Strategy | Parameters | Gradients | Optimizer states |
|---|---|---|---|
| ZeRO-1 | replicated | replicated | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 | sharded | sharded | sharded |
| FSDP full shard | sharded | sharded | sharded |

### Practical guidance

- Use **ZeRO-1/2** when Megatron-Core already owns model-parallel axes and you want lower integration complexity.
- Use **FSDP / ZeRO-3** when parameter memory is the hard limit.
- Do not combine **outer full-model FSDP** and **ZeRO-3** on the same tensors.

---

## 8.2 Compiler-Native FSDP Path

A compiler-native FSDP path is appropriate when priorities are:

- graph-visible communication,
- all-gather / reduce-scatter bucketing,
- communication-computation reordering,
- fewer intrusive model changes,
- improved portability over highly custom runtime stacks.

### Runtime semantics

For each wrapped unit:

- forward:
  - materialize parameters by all-gather,
  - compute locally,
  - free gathered parameters.
- backward:
  - rematerialize parameters by all-gather,
  - compute local gradients,
  - reduce-scatter gradients,
  - free gathered parameters.
- optimizer:
  - update only local shard.

That is the correct ZeRO-3 / FSDP execution model.

---

## 9. Launch Orchestration and Determinism

## 9.1 Deterministic Rank Construction

Launch orchestration must deterministically derive:

- global rank ordering
- local rank ordering
- mesh coordinates
- process groups for $T$, $P$, $C$, $E$, $D$
- NIC rail assignment
- CPU affinity
- RNG seed derivation

### Required invariant

Every rank must derive identical group membership from the same input manifest. Any inconsistency here leads directly to:

- collective mismatch,
- deadlock,
- or silent corruption.

---

## 9.2 Pseudocode: Deterministic Launcher Synthesis

```text
Input:
  cluster inventory
  selected tuple (D, T, P, C, E, T_e)
  topology fingerprint
  runtime policy

Output:
  launch manifest

Procedure:
  1. Sort nodes and GPUs by stable hardware identifiers.
  2. Construct local fast-fabric islands.
  3. Assign T/C/E/T_e groups inside local islands first.
  4. Assign P stage blocks next, keeping stage boundaries topology-aware.
  5. Assign D/FSDP groups over remaining dimensions.
  6. Bind each rank to:
       - one GPU
       - preferred NIC
       - CPU NUMA domain
  7. Emit:
       - rank-to-mesh map
       - communicator manifests
       - environment fingerprint
       - checkpoint format version
       - compile/cache keys
```

---

## 10. Version-Stable Configuration Strategy

The stable control surface must be **intent-level**, not framework-key-level.

### Intent-level configuration groups

- model
  - hidden size
  - attention heads
  - layer count
  - MoE experts and top-$k$
- parallelism
  - $D$, $T$, $P$, $C$, $E$, $T_e$
- precision
  - BF16 / FP16 / FP8
  - reduction dtype
  - optimizer-state dtype
- memory
  - checkpointing policy
  - offload policy
  - bucket limits
- communication
  - collective hierarchy
  - overlap policy
  - bucket strategy
- checkpoint
  - canonical format version
  - reshard rules
- data
  - tokenizer fingerprint
  - dataset lineage
  - packing mode

A renderer then maps this intent-level configuration to:

- Megatron-Core launch parameters
- DeepSpeed policy
- FSDP/DTensor mesh definition
- cluster runtime

This is how version-stable training systems survive library churn.

---

# Part C. Long-Context and MoE Training Architecture Report

## 11. Dense Long-Context Architecture

## 11.1 Core Constraint

For long-context training, activation memory becomes dominant before parameter memory on many workloads. The first system response should be:

1. FlashAttention-class kernels
2. activation checkpointing
3. sequence parallel
4. context parallel
5. only then deeper pipeline or offload

### Reason

FSDP solves persistent parameter-state pressure. It does **not** by itself solve quadratic or large linear activation growth with sequence length.

---

## 11.2 Activation Memory Formulas

Assume:

- microbatch size $B_\mu$
- sequence length $S$
- hidden size $H$
- context parallel degree $C$
- tensor parallel degree $T$
- activation dtype bytes $b_a$
- gated MLP expansion factor $f$

### Residual stream storage per block

$$
M_{\text{residual}} \approx \frac{B_\mu S H b_a}{C}
$$

### Projected attention activations with sequence/tensor partitioning

$$
M_{\text{qkv}} \approx \frac{3 B_\mu S H b_a}{C T}
$$

### MLP intermediate storage

$$
M_{\text{mlp,int}} \approx \frac{f B_\mu S H b_a}{C T}
$$

### Total first-order activation footprint per block

$$
M_{\text{act,block}} \approx M_{\text{residual}} + M_{\text{qkv}} + M_{\text{mlp,int}} + M_{\text{workspace}}
$$

### Key systems conclusion

- Increasing $C$ reduces sequence-local activation terms approximately linearly.
- Increasing $T$ reduces many projected intermediate terms, but not all residual-path storage.
- FlashAttention removes the need to materialize full attention-score matrices, changing memory scaling from naive $O(S^2)$ score storage to tiled streaming workspace.

---

## 11.3 Parameter and Optimizer Memory with FSDP

Let $N_{\theta,j}$ denote parameter class $j$ and let $s_j$ be the product of all sharding axes that apply to that class. Then persistent model-state memory per rank is:

$$
M_{\text{persist}} \approx \sum_j \frac{N_{\theta,j}}{s_j} \left(b_{w,j} + b_{g,j} + b_{o,j}\right)
$$

For Adam-like optimizers:

$$
b_{o,j} \approx 8 + 4\delta_{\text{master}}
$$

where:

- $8$ bytes = two FP32 moments
- $\delta_{\text{master}} = 1$ if FP32 master weights are retained

### FSDP transient gather memory

For wrapped bucket $\mathcal{B}$:

$$
M_{\text{gather,bucket}} \approx \sum_{l \in \mathcal{B}} \frac{N_{\theta,l} b_w}{T_l E_l T_{e,l}}
$$

The $D$-sharding is removed transiently during all-gather, but tensor or expert sharding may still remain.

---

## 11.4 Context Parallel Communication

Let local token count per rank be:

$$
N_t = \frac{B_\mu S}{C}
$$

For ring-style attention where each rank circulates $K$ and $V$ blocks across $C$ ranks, first-order communication per layer is:

$$
V_{\text{ring-attn}} \approx 2(C-1) \times N_t \times \frac{H}{T} \times b_a
$$

Where the factor $2$ accounts for $K$ and $V$.

For Ulysses-style redistribution, the communication is more all-to-all-like; first-order bytes per redistribution are of the same order as a full activation transpose of the local sequence slice.

### Decision rule

- Use **Ulysses-style** redistribution when:
  - local high-bandwidth collectives are strong,
  - all-to-all is efficient,
  - and kernel support is mature.
- Use **Ring Attention** when:
  - the sequence is extreme,
  - streaming KV exchange is preferable,
  - and avoiding large all-to-all bursts is more important.

---

## 12. MoE Architecture

## 12.1 Parallel Factorization

For MoE, use:

$$
W = D \times P \times C \times T_d \times E \times T_e
$$

Where:

- dense trunk uses $T_d$
- experts are partitioned across $E$
- very large experts can further use $T_e$

### Recommended locality

- place $E \times T_e$ inside a local fast-fabric island
- if that does not fit, reduce $E$ before stretching expert traffic across nodes
- keep router computation aligned with the dense trunk’s sharding, not with arbitrary expert layout

---

## 12.2 Token Dispatch Communication

Let:

- local token count be $N_t$
- hidden size be $H$
- top-$k$ routing be used
- EP group size be $E$
- activation bytes be $b_a$

Under balanced routing, one MoE all-to-all has approximate bytes:

$$
V_{\text{A2A}} \approx k N_t H b_a \left(1 - \frac{1}{E}\right)
$$

An MoE layer performs dispatch and combine, so first-order total is:

$$
V_{\text{MoE,layer}} \approx 2 k N_t H b_a \left(1 - \frac{1}{E}\right)
$$

### Interpretation

- communication scales linearly with:
  - token count
  - hidden size
  - top-$k$
- communication does **not** disappear with bigger experts
- therefore, expert all-to-all is a primary scaling bottleneck

---

## 12.3 Expert Capacity and Token Dropping

If the global token count entering an MoE layer is $N_{\text{global}}$ and the total number of experts is $N_{\text{experts}}$, expected balanced load per expert is:

$$
\mathbb{E}[n_e] = \frac{k N_{\text{global}}}{N_{\text{experts}}}
$$

With capacity factor $c_f$:

$$
\text{capacity} = \left\lceil c_f \times \frac{k N_{\text{global}}}{N_{\text{experts}}} \right\rceil
$$

Define imbalance ratio:

$$
\rho = \frac{\max_e n_e}{\mathbb{E}[n_e]}
$$

Then:

- if $\rho \le c_f$, no drops are required
- if $\rho > c_f$, overflow handling is required:
  - token dropping
  - rerouting
  - padding to max capacity
  - or dropless mode with larger buffers and more imbalance cost

### Production guidance

- **Pretraining:** dropless or near-dropless is preferred if bandwidth and memory allow.
- **Fine-tuning:** avoid aggressive token dropping because it can destabilize expert specialization and loss.

---

## 12.4 Expert Load Balancing Controls

Use:

- auxiliary load-balancing loss
- router z-loss or logit regularization
- deterministic top-$k$ tie-breaking
- token permutation order stabilization
- expert-capacity annealing only if justified by metrics

### Determinism requirement

For exact distributed determinism, equal router logits must break ties using a fixed expert-index order. Otherwise cross-rank routing can diverge between runs.

---

## 12.5 Expert Tensor Parallelism

Use $T_e > 1$ only when individual experts are too large for local GEMM efficiency or memory.

### Benefits

- reduces per-rank expert weight residency
- increases expert GEMM size regularity
- may improve arithmetic intensity

### Costs

- adds another per-expert communication axis
- complicates checkpoint conversion
- increases kernel and dispatcher complexity
- can reduce effective overlap if expert batches are small

### Rule

Use $T_e$ only after proving that:
- local expert GEMMs are too large to fit efficiently,
- or expert activation memory forces it,
- and expert batches remain large enough for grouped GEMM efficiency.

---

## 13. Combined Long-Context + MoE Design

## 13.1 Practical Ordering of Design Decisions

1. Make dense attention fit with:
   - FlashAttention
   - checkpointing
   - $C$
2. Place expert groups locally:
   - $E$
   - possibly $T_e$
3. Choose $T_d$ to match hidden-size and GEMM efficiency
4. Use $P$ only when memory or model depth requires
5. Use FSDP / ZeRO over $D$ to control persistent state

### Why this order matters

If $E$ is chosen before long-context attention is stabilized, MoE all-to-all and CP attention traffic will interfere and produce topology collapse.

---

## 13.2 Worked Factorization Example 1: Dense Long-Context on $512$ Accelerators

Assume:

- $W = 512$
- choose $T = 4$
- choose $P = 8$
- choose $C = 4$

Then:

$$
D = \frac{512}{4 \times 8 \times 4} = 4
$$

If:

- $B_\mu = 1$
- $N_\mu = 64$

then:

$$
B_{\text{global}} = 4 \times 64 \times 1 = 256
$$

And pipeline efficiency is:

$$
\eta_{\text{PP}} \approx \frac{64}{64 + 8 - 1} = \frac{64}{71} \approx 0.901
$$

This is operationally acceptable because bubble cost is below $10\%$ before overlap gains.

---

## 13.3 Worked Factorization Example 2: MoE on $256$ Accelerators

Assume:

- $W = 256$
- $T_d = 2$
- $P = 4$
- $C = 2$
- $E = 4$
- $T_e = 2$

Then:

$$
D = \frac{256}{2 \times 4 \times 2 \times 4 \times 2} = 2
$$

This layout is attractive on $8$-GPU nodes because:

$$
E \times T_e = 8
$$

So each expert island fits inside one node. That keeps MoE token dispatch on local fast fabric.

---

## 13.4 Combined Memory-Fit Condition

A long-context MoE run fits only if:

$$
M_{\text{persist}} + M_{\text{gather,bucket}} + M_{\text{act}} + M_{\text{A2A buffers}} + M_{\text{workspace}} + M_{\text{fragmentation}} \le M_{\text{HBM}}
$$

The most common hidden offenders are:

- token dispatch buffers
- MoE expert output staging
- large prefetched FSDP all-gather buckets
- attention workspace spikes at larger $S$
- fragmentation from variable token packing

---

## 13.5 Pseudocode: MoE Capacity-Safe Routing

```text
Input:
  local tokens
  router logits
  top-k value
  expert count
  capacity factor
  deterministic tie-break policy

Output:
  dispatch plan
  overflow accounting

Procedure:
  1. Compute top-k experts for each token using deterministic tie-breaking.
  2. Count assigned tokens per expert.
  3. Compute expert capacity from global-token estimate and capacity factor.
  4. For each expert:
       - if assigned tokens <= capacity:
           accept all tokens
       - else:
           apply configured overflow policy:
             a. drop excess tokens
             b. reroute to backup experts
             c. pad and process in dropless mode if memory allows
  5. Stable-sort tokens by target expert and local position.
  6. Build all-to-all dispatch metadata.
  7. Persist overflow statistics for regression gates.
```

---

# End-to-End Data Pipeline Architecture

## 14. Data Pipeline as a Distributed System

The data path must satisfy both:

- **throughput constraints**
- **statistical correctness constraints**

A high-throughput but statistically corrupted pipeline is not acceptable.

---

## 14.1 Pipeline Stages

A production pretraining pipeline should include:

1. ingestion and normalization
2. exact deduplication
3. near-duplicate detection
4. language/domain/quality filtering
5. tokenizer selection or training
6. tokenization
7. sequence packing
8. deterministic sharding
9. streaming or mmap serving
10. prefetch and async workers
11. resume-safe sample state
12. immutable lineage manifests

---

## 14.2 Deduplication Strategy

### Exact dedup

Use content hashing on normalized documents.

### Near dedup

Use MinHash/LSH or similar locality-sensitive fingerprinting on normalized n-gram shingles.

### Why this matters

- reduces memorization
- improves effective data diversity
- changes the number of unique useful tokens consumed
- directly affects convergence and final quality

---

## 14.3 Pseudocode: Distributed Near-Dedup

```text
Input:
  normalized document stream
  shingle policy
  quality score function
  distributed worker pool

Output:
  deduplicated manifest

Procedure:
  1. For each document:
       - normalize text
       - compute exact hash
       - discard exact duplicates
       - compute MinHash signature over shingles
       - assign signature to LSH buckets

  2. Within each LSH bucket:
       - compare candidate pairs
       - cluster near-duplicates
       - retain representative with highest quality score
       - record lineage from removed docs to representative

  3. Emit immutable manifest:
       - kept document IDs
       - source references
       - dedup cluster IDs
       - normalization version
       - hash algorithm version
```

---

## 14.4 Tokenizer Strategy

Tokenizer choice changes both throughput and optimization dynamics.

### It affects

- bytes-per-token compression
- morphological fragmentation
- multilingual fairness
- average sequence packing efficiency
- embedding table size
- downstream checkpoint compatibility

### Policy

- For **pretraining**, choose tokenizer after evaluating compression and fragmentation on representative corpus slices.
- For **continued pretraining / SFT / DPO / PPO**, keep the tokenizer frozen unless there is a hard incompatibility reason.
- A tokenizer change invalidates many assumptions in:
  - token accounting
  - checkpoint reuse
  - curriculum ratios
  - loss comparisons

---

## 14.5 Sequence Packing

Packing increases useful token ratio by reducing padding.

Define:

- nominal tokens per step: $N_{\text{tok/step}}$
- padding fraction: $p_{\text{pad}}$
- masked or unusable fraction: $p_{\text{mask}}$

Then useful training tokens per step are:

$$
N_{\text{useful}} = N_{\text{tok/step}} \times \left(1 - p_{\text{pad}} - p_{\text{mask}}\right)
$$

### Key caution

Packing is correct only if:

- document boundaries are preserved,
- attention masks are correct,
- loss is masked at non-target positions,
- positional encoding semantics remain valid.

Incorrect packing silently changes the training objective.

---

## 14.6 Pseudocode: Resume-Safe Packed Sampling

```text
Input:
  tokenized shard manifest
  rank r
  world size W
  global seed
  saved cursor state
  target sequence length S

Output:
  deterministic packed sequences

Procedure:
  1. Restore shard order and intra-shard cursor from saved state.
  2. Deterministically assign samples or token blocks to rank r.
  3. Accumulate into a pack buffer until length S is reached.
  4. Insert boundary metadata and exact loss masks.
  5. Emit packed sequence.
  6. Persist:
       - shard index
       - sample index
       - token offset
       - pack-buffer remainder
       - RNG state
```

---

## 14.7 Streaming and Storage

### Preferred data representations

| Representation | Best use case | Tradeoff |
|---|---|---|
| mmap token bins + index | highest-throughput tokenized text | less flexible for heterogeneous examples |
| Parquet | structured metadata + analytics | higher CPU overhead |
| WebDataset | sharded object streaming | convenient for heterogeneous corpora, but cache and decode path matter |

### Production rules

- preprocess once, serve many times
- use immutable shard manifests
- checksum every output shard
- keep preprocessing idempotent
- cache hot shards locally when remote storage jitter is nontrivial

---

## 14.8 How Data Design Changes Convergence

| Data design choice | Throughput effect | Convergence/quality effect |
|---|---|---|
| deduplication | smaller corpus, less repeated compute | reduces memorization and duplicate gradients |
| tokenizer compression | more useful tokens/s | changes subword distribution and loss scale |
| sequence packing | higher useful token ratio | must preserve exact causal masking |
| curriculum/domain mixing | no direct speed effect | changes gradient noise and specialization |
| deterministic resume | stable measurements | essential for scientific ablations |
| streaming cache design | eliminates GPU starvation | prevents dataloader-induced throughput collapse |

---

# Kernel, Numeric, and Communication Optimization

## 15. Kernel Stack Priorities

## 15.1 Training-Critical Kernels

Highest-value kernels for large-scale training:

- FlashAttention-class kernels
- fused MLP
- fused RMSNorm
- fused softmax
- fused RoPE application
- grouped GEMM for MoE experts
- fused optimizer updates
- persistent kernels where shape stability allows
- graph capture for stable inner loops

### Clarification on PagedAttention

PagedAttention is primarily an inference-serving optimization. It is relevant for RLHF rollout sidecars and serving stacks, but it is not the first-order kernel priority for dense pretraining throughput.

---

## 15.2 Precision Policy

| Precision mode | Recommended use |
|---|---|
| BF16 | default baseline on modern NVIDIA and AMD platforms |
| FP16 | only when hardware or model constraints require it; use careful loss scaling |
| FP8 | enable only after parity validation and reduction-precision review |
| MXFP8 / MXFP6 / MXFP4 | research or platform-specific production paths only after exhaustive validation |

### Mandatory safety rules

- keep optimizer moments in stable precision
- keep reductions in BF16 or FP32 where needed
- validate stable softmax and norm kernels
- verify gradient clipping across distributed shards
- compare fused and unfused parity

---

## 15.3 Loss-Parity Validation

For any kernel or precision change, validate against a trusted baseline over fixed data and seed.

### Minimum checks

- loss curve over first $N$ steps
- gradient norm trajectory
- activation summary statistics
- parameter delta norms
- no overflow / underflow anomalies
- equivalent or explainable convergence behavior at fixed token counts

---

## 15.4 Global Gradient Norm in Sharded Training

For logical gradient vector $g$ partitioned across ranks:

$$
\|g\|_2 = \sqrt{\sum_r \sum_{i \in \text{shard}_r} g_i^2}
$$

This must be computed globally over all contributing shards. Local-only clipping is incorrect.

---

# Scientific Scaling and Profiling

## 16. Step-Time Decomposition

A valid scaling report decomposes:

$$
T_{\text{step}} =
T_{\text{data}} +
T_{\text{forward}} +
T_{\text{backward}} +
T_{\text{optimizer}} +
T_{\text{checkpoint}} +
T_{\text{idle}}
$$

With forward and backward further broken into:

- exposed FSDP all-gather time
- hidden FSDP all-gather time
- exposed reduce-scatter time
- hidden reduce-scatter time
- TP collective time
- CP or attention-exchange time
- EP all-to-all time
- compute time
- copy-out or readout time
- pipeline bubble time

---

## 16.1 MFU and HFU

$$
\text{MFU} = \frac{\text{useful model FLOPs/s}}{\text{peak hardware FLOPs/s}}
$$

$$
\text{HFU} = \frac{\text{executed FLOPs/s}}{\text{peak hardware FLOPs/s}}
$$

### Interpretation

- MFU measures end-to-end training usefulness.
- HFU can rise from rematerialization or inefficient extra FLOPs.
- Report both.

---

## 16.2 Profiling Stack

### NVIDIA

- Nsight Systems
- Nsight Compute
- NCCL traces
- NIC counters
- GPU link counters
- PyTorch Profiler

### AMD

- rocprof
- RCCL traces
- xGMI counters
- NIC counters
- PyTorch Profiler on ROCm

### Required profiler questions

- Is the bottleneck kernel-side, collective-side, dataloader-side, or checkpoint-side?
- Are waits too early?
- Is the network on the expected path?
- Is overlap real or only assumed?
- Is a single rank straggling?

---

## 16.3 Regression Gates

Every optimization must have automated pass/fail gates:

- tokens/s
- p50 and p99 step time
- exposed communication time
- MFU / HFU
- max HBM use
- fragmentation growth
- loss at fixed token counts
- gradient norm drift
- dataloader stall time
- checkpoint save/restore latency

---

# Failure-Resilient Automation

## 17. Production Bootstrap Requirements

A robust training launcher must perform:

- topology discovery
- container/runtime validation
- driver/runtime version verification
- NCCL/RCCL transport verification
- local and cross-node bandwidth tests
- storage-health verification
- checkpoint integrity validation
- environment fingerprint emission
- deterministic rank-map generation
- retry and auto-resume policy installation

---

## 17.1 Pseudocode: Production Bootstrap

```text
Input:
  cluster allocation
  training intent config
  checkpoint location
  data manifest

Output:
  validated launch or explicit rejection

Procedure:
  1. Validate runtime:
       - driver versions
       - CUDA/ROCm versions
       - compiler stack
       - kernel support
       - container digest

  2. Discover topology and health:
       - GPU/NIC inventory
       - local-fabric adjacency
       - bandwidth microbenchmarks
       - storage latency and throughput
       - GPU ECC and health signals

  3. Synthesize parallel layout from topology and model constraints.

  4. Validate checkpoint:
       - metadata completeness
       - tensor-count correctness
       - checksum verification
       - world-size conversion requirements

  5. Validate data:
       - tokenizer fingerprint
       - manifest version
       - shard reachability
       - resume cursor consistency

  6. Emit launch manifest and begin training.

  7. During runtime:
       - collect logs
       - monitor stragglers
       - trigger auto-resume on recoverable failure
       - refuse unsafe partial restart without valid reshard plan
```

---

## 17.2 Partial World-Size Restart

### Safe cases

- changing only $D$ or FSDP degree with canonical checkpoint support
- restarting from last completed optimizer step
- identical tokenizer, data manifest, and model graph

### Unsafe without conversion

- changing $T$
- changing $P$
- changing $E$
- changing $T_e$
- changing expert ownership
- changing optimizer-state layout without conversion

---

## 17.3 Deadlock and Straggler Diagnosis

### Deadlock root causes

- mismatched collective order
- divergent control flow across ranks
- inconsistent auto-wrap or compile results across ranks
- pre-launch group-construction mismatch
- one rank faulting before a collective

### Straggler root causes

- slow NIC or GPU
- NUMA mismatch
- GPUDirect not active
- storage jitter
- dataloader worker starvation
- imbalance in expert routing

### Required evidence

- per-rank last collective trace
- stream timeline
- network counters
- dataloader queue depth
- rank-local exception logs
- per-rank step-time histogram

---

# Framework Selection Guidance

## 18. When to Use Which Stack

| Requirement | Recommended stack |
|---|---|
| maximum multidimensional scaling on homogeneous NVIDIA clusters | Megatron-Core + distributed optimizer |
| existing DeepSpeed estate with moderate model-parallel complexity | Megatron-Core + DeepSpeed ZeRO-1/2 |
| parameter-memory-limited training with simpler portability goals | PyTorch FSDP / DTensor |
| compiler-visible FSDP communication optimization | compiler-native FSDP path |
| mixed-vendor maintainability and simpler control plane | PyTorch-native distributed stack + selective custom kernels |
| MoE at large scale with aggressive EP and grouped GEMM | Megatron-Core or equivalent custom runtime |

---

## 18.1 Recommended Decision Rules

- If $T$, $P$, $C$, and $E$ are all large and performance is the top priority, prefer **Megatron-Core-class ownership**.
- If portability and operational simplicity dominate, prefer **FSDP/DTensor-class ownership**.
- If optimizer memory is the only problem, **ZeRO-1/2** is often enough.
- If parameter memory is the bottleneck, use **FSDP or ZeRO-3**, but only one of them as the owner.
- For MoE, prioritize **expert dispatch locality** over elegant but remote factorization.

---

# Final Deployment Recommendations

## 19. Platform-Specific Summary

## 19.1 $A100$

- Default to BF16.
- Keep $T$ conservative, usually $2$ to $4$.
- Use full activation checkpointing for long context.
- Prefer local $E$ only.
- Use FSDP or ZeRO when model-state memory is the limiter.
- Expect communication to become visible sooner than on newer platforms.

## 19.2 $H100$

- Best current platform for combining:
  - FP8,
  - high local $T$,
  - $C$ for long context,
  - local MoE EP.
- Use FlashAttention aggressively.
- Keep $T$, $C$, and $E$ local.
- Validate FP8 parity before scale-out.

## 19.3 $B200$-Class

- Treat as a high-performance local-fabric platform.
- Favor larger intra-node $T$, $C$, and $E$.
- Revisit bucket sizes; previous-generation settings will often underutilize the platform.
- Do not assume early software maturity; build stronger regression gates.

## 19.4 $MI300X$

- Exploit HBM to reduce unnecessary PP depth.
- Use BF16 baseline first.
- Keep $T$, $C$, and $E$ inside xGMI islands.
- Validate RCCL topology and HIP kernel coverage before scaling.

## 19.5 $MI350$-Class

- Same architectural logic as $MI300X$, with more aggressive low-precision and local-group opportunities where qualified.
- Re-qualify graph capture, fused kernels, and collective behavior per ROCm release.
- Avoid importing NVIDIA-centric assumptions into RCCL tuning.

---

# 20. Key Conclusions

1. **Topology-aware factorization is the primary systems decision.**  
   Most scaling failures come from putting the wrong communication pattern on the wrong fabric.

2. **Megatron-Core, DeepSpeed, and FSDP are interoperable only with explicit ownership boundaries.**  
   The failure mode is almost always ambiguous tensor ownership, not “bad luck.”

3. **Long-context training is activation- and attention-communication-dominated.**  
   The correct progression is FlashAttention, checkpointing, SP, then CP.

4. **MoE training is token-dispatch dominated.**  
   Keep expert all-to-all local before optimizing anything else.

5. **Checkpoint portability requires canonical logical-tensor metadata.**  
   Without that, train/resume/convert workflows become brittle and unsafe.

6. **Data pipeline correctness is part of distributed training correctness.**  
   Deduplication, tokenization, packing, and deterministic resume directly change both throughput and convergence.

7. **Every optimization must be scientifically validated.**  
   Measure overlap, memory, communication exposure, and loss parity together.

---

# 21. Reference Links

## Core Distributed Training

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- PyTorch DTensor: https://pytorch.org/docs/stable/distributed.tensor.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Megatron-LM / Megatron-Core: https://github.com/NVIDIA/Megatron-LM

## Communication and Profiling

- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/
- nccl-tests: https://github.com/NVIDIA/nccl-tests
- rccl-tests: https://github.com/ROCm/rccl-tests
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- ROCm profiling tools: https://rocm.docs.amd.com/

## Kernel and Attention

- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Triton: https://github.com/triton-lang/triton
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine

If required, the next deliverable can be expanded into one of the following with full production detail:

- **A detailed per-platform launch blueprint** with node-level rank maps and recommended parallel tuples for specific model sizes such as $7B$, $34B$, $70B$, and MoE variants.
- **A checkpoint interoperability specification** defining a canonical schema across Megatron-Core, DeepSpeed ZeRO, FSDP, and vendor changes.
- **A long-context + MoE performance workbook** containing layerwise communication and memory budgets for target sequence lengths such as $16k$, $32k$, $64k$, and $128k$.