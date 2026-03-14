# Technical Report: Hardware-Specific Deployment Blueprint for $A100$, $H100$, $B200$-Class, $MI300X$, and $MI350$-Class LLM Training Clusters

## 1. Objective

This report provides an end-to-end deployment blueprint for large-scale LLM training on the following accelerator classes:

- NVIDIA $A100$
- NVIDIA $H100$
- NVIDIA $B200$-class
- AMD $MI300X$
- AMD $MI350$-class

The report covers the complete production training stack required to operationalize these clusters for dense pretraining, continued pretraining, SFT, long-context training, and MoE workloads. It includes:

- hardware-aware parallelism design,
- communication topology placement,
- memory-fit derivation,
- kernel and numeric strategy,
- data pipeline sizing,
- checkpoint and resharding policy,
- launch automation,
- health validation,
- profiling and regression methodology,
- platform-specific failure analysis.

The central design constraint is simple:

> **The optimal distributed training configuration is determined by the joint minimization of exposed communication, peak memory, and pipeline bubble under the actual hardware topology and software maturity of the target platform.**

---

## 2. Non-Negotiable Deployment Principles

## 2.1 Topology Principle

> **Place the most communication-intensive axes on the fastest fabric.**

Ordering by communication sensitivity:

1. $TP$ and sequence parallel
2. $CP$
3. $EP$ and $ETP$
4. $PP$
5. $DP$ / FSDP / ZeRO sharding

This ordering should drive all group placement decisions.

---

## 2.2 Single Ownership Principle

> **Each tensor set must have one authoritative sharding owner.**

Examples:

- use FSDP or ZeRO-3 for parameter-state sharding, not both on the same tensors,
- use Megatron-Core to own $TP$/$PP$/$CP$/$EP$ if that is the chosen model-parallel runtime,
- use canonical checkpoints to separate logical tensor identity from physical shard placement.

---

## 2.3 Fit Principle

> **A model fits only if the smallest-HBM rank fits.**

For heterogeneous clusters:

$$
M_{\text{peak},r} \le M_{\text{HBM},r} \quad \forall r
$$

The deployment is invalid if even one rank violates this inequality.

---

## 2.4 Measurement Principle

> **No optimization is accepted without step-time decomposition, overlap evidence, and loss-parity validation.**

Throughput alone is insufficient.

---

## 2.5 Portability Principle

> **The control plane must express intent, not library-specific flags.**

Stable deployment systems define:

- model intent,
- parallelism intent,
- precision intent,
- memory policy,
- checkpoint policy,
- data lineage,

and then synthesize framework-specific launch artifacts per platform.

---

# 3. Hardware Taxonomy and Architectural Consequences

## 3.1 Platform Summary

| Platform | Intra-node fabric | Typical deployment character | Primary strength | Primary operational risk |
|---|---|---|---|---|
| $A100$ | NVLink / NVSwitch | mature BF16 platform | software maturity and broad kernel coverage | older compute-to-communication ratio exposes collectives sooner |
| $H100$ | NVLink / NVSwitch | high-throughput Hopper platform | strong FP8 path, high local collective efficiency | over-fragmented parallelism can reduce MFU |
| $B200$-class | Blackwell-generation local high-bandwidth fabric | next-generation high-throughput platform | larger intra-node communication budget and low-precision potential | early-stack instability, kernel maturity variance |
| $MI300X$ | xGMI | large-HBM AMD platform | substantial HBM capacity, good BF16 posture | RCCL topology quality and HIP kernel coverage must be qualified |
| $MI350$-class | next-generation xGMI-class fabric | newer AMD high-throughput platform | stronger local bandwidth and improving low-precision support | software/runtime maturity must be requalified per ROCm release |

---

## 3.2 Architectural Implications

### $A100$
- Strong choice for:
  - BF16 dense pretraining,
  - mature FSDP/ZeRO deployments,
  - conservative long-context training.
- Less forgiving of:
  - poor overlap,
  - cross-node tensor parallelism,
  - excessive small collectives.

### $H100$
- Best general-purpose platform for:
  - dense frontier-scale training,
  - long-context with $CP$,
  - local MoE expert islands,
  - production FP8 after parity qualification.
- Supports more aggressive intra-node groupings than $A100$.

### $B200$-class
- Designed for:
  - higher local-group bandwidth utilization,
  - more aggressive low-precision operating points,
  - larger local $TP$/$CP$/$EP$ domains.
- Deployment policy must assume:
  - software and compiler behavior require platform-specific validation,
  - bucket sizes and overlap schedules from prior generations will not be optimal.

### $MI300X$
- Large HBM changes the optimization frontier:
  - fewer pipeline stages are often required,
  - offload should be avoided except as last resort,
  - FSDP becomes easier to justify for dense persistent-state control.
- Requires strong RCCL and ROCm validation.

### $MI350$-class
- Similar placement philosophy to $MI300X$ with:
  - greater local-group opportunities,
  - potentially more viable low-precision training paths,
  - renewed need for kernel and collective requalification.

---

# 4. Parallelism Design from First Principles

## 4.1 World-Size Factorization

For dense models:

$$
W = D \times T \times P \times C
$$

Where:

- $W$ = total accelerators
- $D$ = data-parallel or FSDP group count
- $T$ = tensor-parallel degree
- $P$ = pipeline stage count
- $C$ = context-parallel degree

For MoE models:

$$
W = D \times P \times C \times T_d \times E \times T_e
$$

Where:

- $T_d$ = dense-path tensor-parallel degree
- $E$ = expert-parallel degree
- $T_e$ = expert tensor-parallel degree

### Engineering constraints

A valid factorization must satisfy:

- divisibility by total rank count,
- layer divisibility for $PP$,
- locality feasibility for $TP$, $CP$, and $EP$,
- sufficient microbatching to amortize pipeline bubbles,
- HBM fit on every rank,
- kernel availability for the chosen precision and sharding pattern.

---

## 4.2 Global Batch and Microbatching

If each data replica processes $m$ microbatches per optimizer step, each with microbatch size $B_\mu$, then:

$$
B_{\text{global}} = D \times m \times B_\mu
$$

Tokens per optimizer step:

$$
N_{\text{tokens/step}} = B_{\text{global}} \times S
$$

Where $S$ is sequence length.

### Deployment guidance

- Increase $m$ to improve $PP$ efficiency and overlap windows.
- Do not increase $m$ so far that:
  - optimizer step latency becomes too large,
  - activation rematerialization exceeds budget,
  - scheduling overhead becomes significant,
  - or convergence degrades from excessively large global batch.

---

## 4.3 Pipeline Bubble Constraint

For non-interleaved $1F1B$ pipeline schedules, first-order bubble efficiency is:

$$
\eta_{\text{PP}} \approx \frac{m}{m + P - 1}
$$

### Implication

- Deep $PP$ is justified only when:
  - model depth or HBM requires it,
  - or cross-node activation traffic is cheaper than cross-node $TP$/$CP$ traffic.
- On large-HBM platforms such as $MI300X$, deep $PP$ is often avoidable.
- On older $A100$ clusters, $PP$ may still be necessary to fit larger models.

---

## 4.4 Placement Priority by Axis

| Axis | Communication frequency | Preferred location |
|---|---|---|
| $TP$ | every layer | inside the fastest local domain |
| sequence parallel | every layer | same ranks as $TP$ |
| $CP$ | every attention layer | inside fast local domain first |
| $EP$ | every MoE layer | local if possible |
| $ETP$ | every expert MLP | local if possible |
| $PP$ | stage boundaries | can extend across nodes |
| $DP$/FSDP/ZeRO | step or wrapped-unit frequency | widest remaining domain |

### Correct placement rule

- Place $TP$, sequence parallel, and $CP$ on NVLink/NVSwitch or xGMI first.
- Stretch $PP$ across nodes before stretching $TP$ or $CP$ across nodes.
- Place $DP$/FSDP across the widest domain after model-parallel groups are fixed.

---

# 5. Memory Model for Deployment Feasibility

## 5.1 Persistent State Memory

Let:

- $N_\theta$ = total model parameters
- $b_w$ = parameter bytes
- $b_g$ = gradient bytes
- $b_o$ = optimizer-state bytes
- $s_\theta$ = effective sharding factor over parameter ownership

Then persistent per-rank model-state memory is approximately:

$$
M_{\text{persistent}} \approx \frac{N_\theta}{s_\theta} \left(b_w + b_g + b_o\right)
$$

For Adam-like optimizers:

- moments contribute approximately $8$ bytes per parameter if stored in FP32,
- master weights add approximately $4$ bytes per parameter if retained.

### Engineering implication

- replicated Adam state is often the first hard memory wall,
- ZeRO-1/2 or FSDP/ZeRO-3 are often introduced primarily to control optimizer-state residency before parameter residency becomes the dominant limit.

---

## 5.2 Activation Memory

For a transformer block, activation memory scales primarily with:

- microbatch size $B_\mu$,
- sequence length $S$,
- hidden size $H$,
- checkpointing policy,
- $TP$,
- $CP$.

A first-order block-level activation estimate is:

$$
M_{\text{act,block}} \approx \frac{B_\mu S H b_a}{C} \times \left(k_{\text{res}} + \frac{k_{\text{proj}}}{T}\right) + M_{\text{workspace}}
$$

Where:

- $b_a$ = activation bytes,
- $k_{\text{res}}$ and $k_{\text{proj}}$ summarize residual and projected intermediate terms,
- $M_{\text{workspace}}$ depends on attention kernel choice, fused kernels, and temporary buffers.

### Consequences

- $CP$ is often the most effective long-context activation control beyond FlashAttention and checkpointing.
- $TP$ reduces many projected intermediates but does not solve all residual-path storage.
- FlashAttention-class kernels materially reduce attention-score materialization pressure.

---

## 5.3 FSDP / ZeRO-3 Transient Memory

If full parameter materialization is used per wrapped bucket $\mathcal{B}$, transient gather memory per rank is roughly:

$$
M_{\text{gather}} \approx \sum_{l \in \mathcal{B}} \frac{N_{\theta,l} b_w}{T_l \times E_l \times T_{e,l}}
$$

This term is transient but often decisive in fit failures.

### Why fits fail unexpectedly

A configuration may fit persistent state but still fail due to:

- oversized all-gather buckets,
- activation checkpoint window overlap with gathered parameters,
- MoE dispatch buffers,
- fragmented allocator state,
- FP8 or mixed-precision staging buffers,
- sequence-length-specific workspace spikes.

---

## 5.4 Peak Memory Condition

Deployment is feasible only if:

$$
M_{\text{persistent}} + M_{\text{activations}} + M_{\text{transient comm}} + M_{\text{workspace}} + M_{\text{fragmentation}} \le M_{\text{HBM}}
$$

for every rank.

---

# 6. Communication Model and Topology-Aware Cost Reasoning

## 6.1 First-Order Collective Cost

For message size $n$ bytes:

$$
T_m(n) = \alpha + \beta n
$$

For ring-based collectives over group size $g$:

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

### Deployment consequence

- small collectives are latency-dominated,
- large collectives are bandwidth-dominated,
- bucket size must be tuned to hide startup cost without exceeding overlap and memory limits.

---

## 6.2 Fabric-Specific Collective Placement

### NVLink / NVSwitch
Use for:

- $TP$
- sequence parallel
- $CP$
- local $EP$
- local $ETP$

### xGMI
Use analogously for AMD clusters.

### InfiniBand / RoCE
Use primarily for:

- $DP$
- FSDP
- ZeRO communication
- stage-boundary $PP$ when required

### PCIe-only paths
Treat as emergency paths, not preferred steady-state transport for frequent collectives.

---

## 6.3 Hierarchical Collective Strategy

When local fabric is much faster than inter-node fabric, hierarchical collectives are often superior:

1. local reduction or gather,
2. inter-node exchange,
3. local redistribution.

This is especially relevant for:

- gradient synchronization,
- FSDP parameter gather across many nodes,
- MoE metadata exchange if cross-node dispatch cannot be avoided.

---

## 6.4 GPUDirect and RDMA Policy

### Mandatory validation items

- direct GPU-to-NIC transport is active,
- no silent host bounce buffering,
- GPU/NIC affinity is NUMA-consistent,
- RoCE congestion control is correct if used,
- multi-rail usage is explicit and reproducible.

### Symptoms of GPUDirect failure

- bandwidth collapse,
- elevated host CPU utilization,
- high per-rank latency variance,
- flattened overlap,
- collective tails unrelated to kernel timelines.

---

# 7. Topology Discovery and Rank Mapping Blueprint

## 7.1 Required Discovery Inputs

Before planning any run, discover:

- GPU identifiers and locality
- switch-domain membership
- GPU-to-GPU link matrix
- GPU-to-NIC affinity
- CPU NUMA affinity
- storage locality
- local NVMe presence
- per-link health status

---

## 7.2 Rank Mapping Strategy

### Correct mapping order

1. pack $TP$ ranks into the fastest local island,
2. co-locate sequence parallel with $TP$,
3. place $CP$ within remaining fast local capacity,
4. place $EP$ and $ETP$ locally if possible,
5. build $PP$ stage blocks next,
6. spread $DP$/FSDP across the residual domain.

### Incorrect mapping pattern

- cross-node $TP$ while local NVLink/xGMI capacity remains unused,
- cross-node $CP$ for moderate context lengths if local partitioning was possible,
- interleaving unrelated $DP$ ranks inside local islands and breaking $TP$ locality.

---

## 7.3 Pseudocode: Topology-Aware Group Construction

```text
Input:
  accelerator topology graph
  desired tuple (D, T, P, C, E, T_e)
  hardware health report

Output:
  rank map and process-group plan

Procedure:
  1. Build weighted topology graph:
       - lowest edge cost for same-switch or same-xGMI island
       - higher cost for cross-socket or cross-node paths

  2. Reserve local islands for latency-sensitive axes:
       - assign T groups first
       - bind sequence-parallel ranks to T groups
       - assign C groups next
       - assign E and T_e next if MoE is enabled

  3. Partition remaining ranks into P stage blocks:
       - minimize stage-boundary path cost
       - keep stage balance compatible with layer counts

  4. Form D/FSDP groups over the remaining dimension.

  5. Bind each rank to:
       - one GPU
       - one preferred NIC
       - one CPU NUMA domain

  6. Validate:
       - no axis crosses a slower fabric when a faster valid mapping exists
       - all ranks fit HBM budgets
       - all group memberships are deterministic
```

---

# 8. Software Stack Blueprint by Hardware Family

## 8.1 NVIDIA Stack

### Baseline software layers

- CUDA runtime and driver pinned to one validated release family
- NCCL version pinned to driver family and platform generation
- PyTorch release matched to CUDA and compiler stack
- Megatron-Core or PyTorch FSDP/DTensor depending system design
- FlashAttention-class kernels
- Transformer Engine or equivalent production low-precision runtime where qualified
- Nsight Systems and Nsight Compute
- container digest pinned by platform generation

### Operating rule

Do not mix:
- arbitrary CUDA minor versions,
- arbitrary NCCL minor versions,
- independently upgraded FlashAttention or compiler stacks,

inside the same training estate without requalification.

---

## 8.2 AMD Stack

### Baseline software layers

- ROCm pinned to a tested release family
- RCCL pinned to the ROCm runtime
- PyTorch ROCm build pinned to exact ROCm version
- HIP/Triton/Composable Kernel path qualified for target kernels
- rocprof and associated tracing tools
- container digest pinned by platform and ROCm release

### Operating rule

Feature parity assumptions from CUDA must never be imported blindly into ROCm. Revalidate:

- graph capture behavior,
- fused attention availability,
- grouped GEMM performance,
- FP8 maturity,
- collective scheduling stability.

---

## 8.3 Cross-Vendor Control Plane

The training control plane should define only:

- model shape
- workload type
- sequence length
- precision policy
- parallel tuple
- memory policy
- checkpoint schema
- data lineage

A renderer then emits:

- Megatron-Core launch config,
- DeepSpeed config,
- FSDP/DTensor mesh layout,
- environment manifest,
- profiler hooks,
- health-check directives.

This is the only maintainable path for mixed-vendor operations.

---

# 9. Platform-Specific Deployment Blueprints

# 9.1 $A100$ Blueprint

## 9.1.1 Best-Fit Use Cases

- mature BF16 dense training,
- medium-to-large dense pretraining,
- conservative long-context workloads,
- established ZeRO/FSDP training stacks,
- mature production environments where software stability is prioritized over peak accelerator capability.

---

## 9.1.2 Parallelism Recommendations

### Dense pretraining
- Prefer:
  - $T = 2$ to $4$
  - $P = 2$ to $8$
  - $C = 1$ to $2$
  - FSDP or ZeRO over $D$
- Sequence parallel is strongly recommended whenever $T > 1$.

### Long-context
- Use:
  - FlashAttention-class kernels,
  - full or selective activation checkpointing,
  - $C = 2$ to $4$ before increasing $TP$ across nodes.
- Avoid cross-node $TP$ for long-context unless absolutely necessary.

### MoE
- Keep $EP$ local.
- Use conservative $ETP$.
- Top-$2$ routing is viable only if expert all-to-all remains inside fast local fabric.

### Fine-tuning
- Use smaller $PP$ or eliminate $PP$ if model fits.
- Prefer FSDP full-shard or ZeRO-2/3 over heavy model-parallel complexity.

---

## 9.1.3 Precision Strategy

- Default: BF16
- FP16: only when required by stack constraints
- FP8: generally not the primary production path for $A100$

### Reason

$A100$ remains highly productive in BF16, but it does not provide the same low-precision operating envelope as newer platforms.

---

## 9.1.4 Communication Priorities

- Keep $TP$ strictly intra-node when possible.
- Use hierarchical gradient synchronization beyond a few nodes.
- Bucket aggressively enough to amortize latency.
- Reordering and overlap are mandatory; otherwise step time becomes communication-exposed quickly.

### Failure signature

If $A100$ throughput scales poorly, the likely cause is not raw compute. It is usually:

- exposed collectives,
- small-bucket launch overhead,
- dataloader starvation,
- or cross-node $TP$.

---

## 9.1.5 Memory Policy

- Use activation checkpointing early.
- Use FSDP/ZeRO when optimizer state dominates HBM.
- CPU offload should be treated as last resort.
- Keep bucket sizes smaller than on $H100$ and $B200$-class if overlap windows are narrow.

---

## 9.1.6 What Not to Do on $A100$

- large cross-node $TP$ for modest hidden sizes,
- aggressive unvalidated FP8 paths,
- deep MoE expert layouts with off-node dispatch if avoidable,
- overly fine-grained FSDP wrapping causing many small all-gathers.

---

# 9.2 $H100$ Blueprint

## 9.2.1 Best-Fit Use Cases

- large dense pretraining,
- long-context training with substantial $CP$,
- local MoE expert islands,
- aggressive fused-kernel and overlap strategies,
- production FP8 after parity validation.

---

## 9.2.2 Parallelism Recommendations

### Dense pretraining
- Strong starting region:
  - $T = 4$ to $8$
  - $P = 2$ to $8$
  - $C = 1$ to $4$
  - distributed optimizer or FSDP over $D$
- Sequence parallel should always accompany nontrivial $TP$.

### Long-context
- Prefer:
  - FlashAttention,
  - $CP = 2$ to $8$,
  - less reliance on deep $PP$ than $A100$,
  - checkpointing targeted to the actual activation bottlenecks.

### MoE
- $EP$ local to node or switch island.
- $ETP$ is feasible if expert size requires it.
- Grouped GEMM and expert-batching efficiency should be explicitly profiled.

### Fine-tuning and continued pretraining
- Often possible with:
  - reduced $PP$,
  - stronger use of FSDP,
  - lower system complexity than on $A100$ for the same model family.

---

## 9.2.3 Precision Strategy

- Baseline: BF16
- Advanced: FP8 after:
  - loss parity,
  - gradient norm parity,
  - fused/unfused path comparison,
  - multi-node equivalence validation.

### Precision rule

Use FP8 only where the stack preserves:
- stable reductions,
- stable optimizer states,
- correct clipping,
- and reproducible checkpoint resume.

---

## 9.2.4 Communication Priorities

- Intra-node collectives are efficient enough to support higher local $TP$ and $CP$.
- Use cross-node traffic primarily for $DP$/FSDP and stage boundaries.
- Re-evaluate bucket sizes upward relative to $A100$; legacy settings often underutilize $H100$.

---

## 9.2.5 Memory Policy

- Checkpointing remains important for long context.
- FP8 can reduce activation and weight bandwidth pressure, but does not replace memory-fit analysis.
- FSDP transient gather windows can still dominate peak memory if wrapping is too coarse.

---

## 9.2.6 What Not to Do on $H100$

- assume FP8 is safe without parity testing,
- copy $A100$ bucket sizes directly,
- oversplit the model into excessive $PP$ just because it worked on prior hardware,
- neglect dataloader and storage provisioning; $H100$ can outpace weak input systems quickly.

---

# 9.3 $B200$-Class Blueprint

## 9.3.1 Best-Fit Use Cases

- high-throughput dense training with large local model-parallel groups,
- long-context workloads where large $CP$ is valuable,
- aggressive low-precision experimentation after production qualification,
- local MoE expert islands with larger communication budgets.

---

## 9.3.2 Platform Assumption Model

Treat $B200$-class systems as:

- high-throughput local-switch platforms,
- favorable to larger intra-node groupings,
- capable of stronger low-precision utilization than $H100$ once software is fully qualified.

### Critical caution

Do not assume first-generation software stacks are mature simply because the hardware is fast.

---

## 9.3.3 Parallelism Recommendations

### Dense pretraining
- Often start with:
  - larger local $TP$ than on $H100$ if kernels remain efficient,
  - shallower $PP$,
  - higher reliance on local collectives,
  - FSDP or distributed optimizer across $D$.

### Long-context
- Push $CP$ higher than on previous generations if:
  - attention kernels remain stable,
  - overlap remains effective,
  - and local attention exchange stays inside the fast domain.

### MoE
- Larger local $EP$ becomes attractive.
- $ETP$ can be used for very large experts if expert batches stay sufficiently large.

---

## 9.3.4 Precision Strategy

- Baseline: BF16
- Production advanced path: FP8 or platform-qualified mixed-precision low-bit formats
- Research-only path: lower-than-FP8 formats only with strict gating

### Rule

Do not deploy new low-bit formats before validating:
- optimizer correctness,
- checkpoint resumption,
- loss parity,
- cross-node collective numerical behavior.

---

## 9.3.5 Communication Priorities

- Recompute bucket sizes from scratch.
- Recompute overlap windows from scratch.
- Validate collective algorithm selection and channel count anew.

### Common mistake

Teams inherit tuning from $H100$ and underutilize $B200$-class platforms because:
- buckets are too small,
- local groups are artificially limited,
- or stale guardrails cap overlap aggressively.

---

## 9.3.6 What Not to Do on $B200$-Class

- blindly import prior-generation launch defaults,
- deploy aggressive low-precision globally before platform-level parity certification,
- ignore compiler and graph-capture maturity,
- mix multiple experimental kernel stacks in one production estate.

---

# 9.4 $MI300X$ Blueprint

## 9.4.1 Best-Fit Use Cases

- large dense models whose parameter and optimizer-state pressure benefited from larger HBM,
- BF16 training with fewer pipeline stages,
- moderate-to-large long-context workloads with carefully qualified ROCm kernels,
- FSDP/DTensor deployments prioritizing portability and memory efficiency.

---

## 9.4.2 Parallelism Recommendations

### Dense pretraining
- Use:
  - moderate $TP$
  - conservative to moderate $PP$
  - more reliance on HBM than on deep $PP$
  - FSDP over $D$ when persistent state dominates

### Long-context
- First exploit:
  - large HBM,
  - FlashAttention availability,
  - checkpointing
- Then add:
  - $CP = 2$ to $4$ or higher if local xGMI grouping is efficient.

### MoE
- Keep $EP$ within xGMI islands.
- Use conservative $ETP$ unless expert size clearly justifies it.
- Profile grouped GEMM and token dispatch carefully.

---

## 9.4.3 Precision Strategy

- Baseline: BF16
- Advanced low precision: only after ROCm stack, kernels, and optimizer path are fully qualified

### Rule

$MI300X$ should be treated as a BF16-first production platform unless the exact software release has certified lower-precision training for the target workload.

---

## 9.4.4 Communication Priorities

- xGMI-local placement is mandatory for latency-sensitive axes.
- RCCL topology validation must occur before blaming model code.
- Cross-node collectives should be reserved for:
  - $DP$/FSDP,
  - some $PP$ boundaries,
  - unavoidable cross-node MoE cases.

### Failure signature

If performance on $MI300X$ is unexpectedly poor, the highest-probability causes are:

- xGMI locality misuse,
- suboptimal RCCL rings,
- kernel coverage regressions,
- or host/NUMA misbinding.

---

## 9.4.5 Memory Policy

- Large HBM reduces the need for offload and deep $PP$.
- Use that headroom deliberately:
  - larger checkpoint windows,
  - larger but overlap-safe buckets,
  - fewer pipeline partitions,
  - larger local expert residency.
- Do not waste HBM on fragmented allocator patterns or oversized communication buffers.

---

## 9.4.6 What Not to Do on $MI300X$

- assume CUDA kernel maturity equivalence,
- stretch $TP$ or $CP$ off-island before exhausting xGMI-local layouts,
- enable low precision without ROCm-specific parity evidence,
- ignore NUMA and CPU affinity.

---

# 9.5 $MI350$-Class Blueprint

## 9.5.1 Best-Fit Use Cases

- next-generation AMD training clusters with improved local-group capacity,
- BF16-first training with possible expansion into lower precision,
- larger local $CP$ and $EP$ layouts than earlier AMD generations where software supports them.

---

## 9.5.2 Parallelism Recommendations

### Dense pretraining
- Start from the $MI300X$ playbook, then test:
  - higher local $TP$,
  - shallower $PP$,
  - larger FSDP buckets if overlap remains strong.

### Long-context
- Increase $CP$ only after validating:
  - attention kernel efficiency,
  - stability of context-exchange collectives,
  - graph capture behavior if used.

### MoE
- Prefer local $EP$ within the next-generation xGMI domain.
- Introduce $ETP$ only if expert size or memory makes it necessary.

---

## 9.5.3 Precision Strategy

- Default: BF16
- Advanced: platform-qualified FP8 path if and only if:
  - ROCm stack supports it reliably,
  - kernel stack is stable,
  - checkpoint semantics are preserved.

### Rule

Treat lower precision as an optimization layer, not as a deployment baseline.

---

## 9.5.4 Communication Priorities

- Repeat full RCCL qualification; do not assume $MI300X$ tuning carries over.
- Validate all hierarchical collective heuristics on the new topology.
- Build rank maps around actual xGMI connectivity, not assumed symmetry.

---

## 9.5.5 What Not to Do on $MI350$-Class

- import stale RCCL tuning without retesting,
- assume equivalent graph capture and fused kernel behavior across ROCm releases,
- overexpand local groups before checking actual kernel occupancy and memory traffic.

---

# 10. Workload-Specific Hardware Deployment Patterns

## 10.1 Dense Pretraining

### Preferred ordering of controls

1. choose local $TP$ that matches GEMM efficiency,
2. add sequence parallel,
3. use $PP$ only as needed for fit or stage balance,
4. add FSDP/ZeRO when persistent state becomes dominant,
5. keep $DP$ across the widest domain.

### Hardware bias

- $A100$: more likely to need deeper $PP$ and tighter bucket control.
- $H100$: stronger local $TP$ and $CP$.
- $B200$-class: larger local groups and higher low-precision potential.
- $MI300X$: fewer stages due to HBM.
- $MI350$-class: same, with stronger local-group potential if stack is mature.

---

## 10.2 Continued Pretraining and SFT

Characteristics:
- often lower global batch,
- shorter sequences than frontier pretraining,
- smaller overlap windows,
- optimizer-state load may still be large.

### Hardware-specific effects

- $A100$: FSDP often preferable to complicated model parallelism for moderate scales.
- $H100$ and $B200$-class: simpler layouts may already fit and run efficiently.
- $MI300X$ and $MI350$-class: large HBM can simplify the training plan substantially.

---

## 10.3 Long-Context Training

### Correct strategy order

1. FlashAttention-class kernels
2. activation checkpointing
3. sequence parallel
4. context parallel
5. only then deeper $PP$ or offload

### Hardware observations

- $H100$ and $B200$-class are the strongest candidates for high-$CP$ long-context deployments.
- $A100$ can support long context, but exposed communication and memory pressure appear sooner.
- $MI300X$ and $MI350$-class benefit from HBM, but kernel qualification is decisive.

---

## 10.4 MoE Training

### Correct expert placement rule

- Keep $EP \times ETP$ local whenever possible.
- Stretch dense $DP$ across nodes before stretching expert all-to-all across nodes.

### Hardware observations

- $H100$ and $B200$-class are particularly strong for local MoE islands.
- $MI300X$ benefits from expert residency due to HBM but requires careful dispatch validation.
- $A100$ supports MoE well when expert groups remain local and not overly fragmented.

---

# 11. Kernel and Numerical Blueprint by Platform

## 11.1 Priority Kernel Stack

For training, the critical kernels are:

- FlashAttention
- fused MLP
- fused RMSNorm
- fused softmax
- fused RoPE
- grouped GEMM for MoE
- fused optimizer update where validated
- persistent kernels where shape stability allows
- graph capture where the runtime is stable

### Important clarification

PagedAttention is primarily an inference-serving optimization. It is relevant for serving and RL rollout systems, but it is not a first-order pretraining throughput lever.

---

## 11.2 Precision Recommendations by Platform

| Platform | Baseline | Advanced production path | Notes |
|---|---|---|---|
| $A100$ | BF16 | none preferred beyond BF16 | prioritize stability and overlap |
| $H100$ | BF16 | FP8 after full validation | strongest mature FP8 candidate |
| $B200$-class | BF16 | FP8 / qualified newer formats | strict gating required |
| $MI300X$ | BF16 | release-qualified low precision only | ROCm maturity decides |
| $MI350$-class | BF16 | release-qualified FP8 or equivalent | requalify per stack version |

---

## 11.3 Numerical Robustness Checklist

For every platform:

- global gradient norm clipping must be logically global,
- stable softmax and norm kernels must be verified,
- loss scaling must be correct where FP16 is used,
- overflow/underflow checks must run in regression,
- fused and unfused parity must be compared,
- single-node and multi-node parity must be compared,
- CUDA and ROCm behavior must be compared if portability is a goal.

For logical gradient vector $g$ sharded across ranks:

$$
\|g\|_2 = \sqrt{\sum_r \sum_{i \in \text{shard}_r} g_i^2}
$$

Local-only clipping is incorrect.

---

# 12. Data Pipeline Blueprint Matched to Accelerator Throughput

## 12.1 Core Requirement

The data pipeline must not bottleneck the cluster. This is a hard system requirement, not an optimization preference.

If GPU-side useful token consumption target is $\tau_{\text{gpu}}$ and packing efficiency is $\eta_{\text{pack}}$, then the input system must sustain:

$$
\tau_{\text{input}} \ge \frac{\tau_{\text{gpu}}}{\eta_{\text{pack}}}
$$

As accelerator throughput rises from $A100$ to $H100$ to $B200$-class, the data system becomes a first-class scaling bottleneck.

---

## 12.2 Mandatory Data Pipeline Components

- immutable corpus manifest
- exact deduplication
- near-duplicate removal
- tokenizer fingerprinting
- deterministic tokenization
- sequence packing with correct masks
- deterministic sharding
- resume-safe cursoring
- async prefetch
- local cache or mmap where possible
- lineage tracking for every processed shard

---

## 12.3 Hardware-Specific Data Recommendations

### $A100$
- pretokenized mmap or compact binary format strongly recommended,
- moderate prefetch depth may suffice,
- local NVMe cache still preferred.

### $H100$
- object-store-at-step-time is usually unacceptable,
- larger CPU worker pools and prefetch queues are needed,
- pretokenized and locally cached data are effectively mandatory at scale.

### $B200$-class
- fully preprocessed, locally cacheable token shards,
- deterministic sampling state,
- high-throughput storage and NIC isolation for data traffic,
- pipeline regressions will surface quickly if not tuned.

### $MI300X$ / $MI350$-class
- same logical requirements,
- additionally validate:
  - pinned host memory behavior,
  - CPU NUMA binding,
  - ROCm dataloader overheads,
  - storage locality relative to rank placement.

---

## 12.4 Pseudocode: Resume-Safe Distributed Data Serving

```text
Input:
  token shard manifest
  global seed
  rank r
  world size W
  saved cursor state
  sequence length S

Output:
  deterministic packed samples

Procedure:
  1. Restore:
       - shard index
       - sample index
       - token offset
       - pack-buffer remainder
       - RNG state

  2. Deterministically assign token ranges or samples to rank r.

  3. Fill pack buffer until length S is reached.

  4. Emit:
       - packed sequence
       - exact attention and loss masks
       - boundary metadata

  5. Persist cursor state periodically and at checkpoint boundaries.

Guarantee:
  resume reproduces the same token stream and same packing behavior.
```

---

# 13. Communication Tuning Blueprint

## 13.1 Bucket Sizing Policy

Bucket size must satisfy both:

$$
T_{\text{bucket}} \le T_{\text{available overlap}}
$$

and:

$$
M_{\text{peak current}} + M_{\text{bucket}} \le M_{\max}
$$

### Platform bias

- $A100$: smaller overlap windows; be more conservative.
- $H100$: larger efficient buckets usually viable.
- $B200$-class: retune from scratch; prior-generation buckets are usually suboptimal.
- $MI300X$ / $MI350$-class: exploit HBM but respect RCCL overlap behavior and kernel scheduling.

---

## 13.2 Overlap Strategy

Prioritize overlap of:

- FSDP all-gather with current compute,
- FSDP reduce-scatter with later compute,
- gradient bucket reduction with backward kernels,
- expert all-to-all with non-dependent dense-path work where possible.

### Wrong pattern

Launching collectives asynchronously but placing waits immediately before or after launch. That destroys effective overlap.

---

## 13.3 Deadlock Prevention

Deadlocks most often arise from:

- inconsistent process-group construction,
- divergent control flow across ranks,
- compiler graph divergence across ranks,
- mismatched collectives after partial failure,
- dynamic shape or data-dependent path divergence.

### Preventive controls

- deterministic launcher synthesis,
- identical model graph and wrapping policy across all ranks,
- static communicator manifests,
- rank-local exception capture before collective failure.

---

# 14. Health Checks, Preflight, and Automation Blueprint

## 14.1 Required Preflight Checks

Every launch must validate:

- GPU health
- ECC status
- fabric connectivity
- GPU-to-NIC affinity
- driver/runtime compatibility
- container digest
- NCCL/RCCL version compatibility
- storage health
- local and cross-node bandwidth
- checkpoint integrity
- tokenizer and data manifest identity

---

## 14.2 Pseudocode: Cluster Bring-Up

```text
Input:
  cluster allocation
  platform type
  model intent
  data manifest
  checkpoint path

Output:
  validated launch or hard rejection

Procedure:
  1. Validate runtime image:
       - driver
       - CUDA or ROCm
       - NCCL or RCCL
       - compiler stack
       - profiler stack

  2. Discover topology:
       - local GPU graph
       - GPU-to-NIC affinity
       - switch or xGMI islands
       - CPU NUMA map

  3. Run health and bandwidth tests:
       - local GPU-to-GPU
       - local GPU-to-NIC
       - cross-node collective microbenchmarks
       - storage throughput and latency

  4. Synthesize rank map and communicator plan.

  5. Validate memory-fit using target tuple and smallest-HBM rank.

  6. Validate checkpoint:
       - schema
       - shard completeness
       - checksum
       - reshard requirements

  7. Validate data lineage:
       - tokenizer fingerprint
       - manifest version
       - cursor compatibility

  8. Emit launch manifest and lock the environment fingerprint.
```

---

## 14.3 Auto-Resume Policy

Safe auto-resume requires:

- last known-good optimizer step,
- valid checkpoint integrity,
- identical model definition,
- compatible or explicitly resharded parallelism,
- preserved data cursor state,
- restored RNG states.

### Safe changes on resume

Usually safe with canonical checkpoints:
- $D$ change
- FSDP shard count change

Not safe without explicit conversion:
- $T$ change
- $P$ change
- $E$ change
- $ETP$ change

---

# 15. Checkpointing and Storage-Aware Blueprint

## 15.1 Canonical Checkpoint Requirements

Each logical tensor must store:

- name
- shape
- dtype
- tensor class
- shard metadata
- owning mesh axes
- optimizer-state mapping
- RNG state
- scheduler state
- data cursor metadata

### Why this is mandatory

Without canonical logical metadata:
- world-size changes are brittle,
- cross-framework conversion becomes unsafe,
- mixed-vendor portability becomes fragile.

---

## 15.2 Storage-Aware Checkpoint Policy

Checkpoint frequency must be chosen against:

- failure rate,
- checkpoint write bandwidth,
- training step time,
- restart cost.

### Engineering rule

Checkpointing must not create synchronized I/O stalls across the cluster. Use:

- asynchronous staging where supported,
- shard-local writes with manifest commit,
- post-write validation,
- bounded retention.

---

# 16. Profiling and Scientific Scaling Blueprint

## 16.1 Step-Time Decomposition

Every platform deployment must track:

$$
T_{\text{step}} =
T_{\text{data}} +
T_{\text{forward}} +
T_{\text{backward}} +
T_{\text{optimizer}} +
T_{\text{checkpoint}} +
T_{\text{idle}}
$$

And forward/backward must be further decomposed into:

- exposed all-gather,
- hidden all-gather,
- exposed reduce-scatter,
- hidden reduce-scatter,
- $TP$ collective time,
- $CP$ collective time,
- $EP$ all-to-all,
- compute,
- wait and copy-out,
- pipeline bubble.

---

## 16.2 MFU and HFU

$$
\text{MFU} = \frac{\text{useful model FLOPs/s}}{\text{peak hardware FLOPs/s}}
$$

$$
\text{HFU} = \frac{\text{executed FLOPs/s}}{\text{peak hardware FLOPs/s}}
$$

### Reporting rule

Report both. Rematerialization can raise $HFU$ while harming true useful efficiency.

---

## 16.3 Strong and Weak Scaling Requirements

### Strong scaling
- fixed global batch,
- fixed sequence length,
- increase rank count,
- quantify efficiency loss and communication exposure.

### Weak scaling
- fixed work per rank,
- increase rank count proportionally,
- measure throughput stability and tail behavior.

### Hardware interpretation

- $A100$ strong scaling usually exposes communication first.
- $H100$ and $B200$-class often expose kernel balance and dataloader limits next.
- $MI300X$ and $MI350$-class frequently expose topology and kernel maturity issues first.

---

## 16.4 Mandatory Tooling

### NVIDIA
- Nsight Systems
- Nsight Compute
- NCCL traces
- network counters
- PyTorch Profiler

### AMD
- rocprof
- RCCL traces
- xGMI counters
- network counters
- PyTorch Profiler on ROCm

### Minimum evidence standard

No regression or tuning decision should be accepted without:
- timeline evidence,
- collective timing breakdown,
- memory trace,
- and loss comparison at fixed token count.

---

# 17. Platform-Specific Failure Signatures and Remediation

## 17.1 Common Failure Table

| Symptom | $A100$ likely cause | $H100$ likely cause | $B200$-class likely cause | $MI300X$ likely cause | $MI350$ likely cause |
|---|---|---|---|---|---|
| poor scaling | exposed collectives | suboptimal bucket sizes | stale prior-gen tuning | xGMI misuse or RCCL topology | unqualified runtime or stale RCCL tuning |
| OOM despite sharding | coarse FSDP bucket | transient gather + activations | compiler staging buffers | fragmented HBM or staging buffers | same plus new-kernel workspace behavior |
| deadlock | mismatched communicator or control flow | compile divergence or collective ordering | experimental stack mismatch | RCCL group mismatch or control divergence | same plus immature kernel path |
| low MFU | dataloader starvation or comm | small kernels, insufficient local grouping | underutilized local fabric | kernel coverage or dispatch overhead | kernel maturity / launch overhead |
| unstable loss after precision change | unsupported low precision | insufficient FP8 validation | premature low-bit deployment | ROCm precision-path immaturity | same |

---

## 17.2 Root-Cause Procedure

### If bandwidth collapses
1. validate topology path,
2. validate GPUDirect or equivalent direct transport,
3. inspect per-rank bandwidth asymmetry,
4. inspect NUMA and NIC binding,
5. inspect bucket size and collective algorithm.

### If step time variance spikes
1. inspect dataloader queue depth,
2. inspect one-rank stragglers,
3. inspect storage jitter,
4. inspect expert load imbalance if MoE,
5. inspect checkpoint or logging interference.

### If loss diverges after platform migration
1. compare fused vs unfused path,
2. compare reduction dtype,
3. compare clipping behavior,
4. compare checkpoint load conversions,
5. compare tokenizer and data-order identity.

---

# 18. Framework Selection by Platform

## 18.1 Recommended Abstractions

| Platform posture | Preferred training abstraction |
|---|---|
| homogeneous NVIDIA frontier scale | Megatron-Core first, FSDP where state sharding is required |
| mature mixed-workload NVIDIA estate | DeepSpeed ZeRO-1/2 or FSDP depending ownership model |
| mixed-vendor portability priority | PyTorch FSDP / DTensor with selective custom kernels |
| AMD-first estate emphasizing portability | PyTorch FSDP / DTensor, Megatron where feature-complete and qualified |
| aggressive MoE with large local expert groups | Megatron-Core or custom expert runtime |

### Hardware-specific interpretation

- $A100$: Megatron-Core or FSDP both viable depending scale and maturity goals.
- $H100$: Megatron-Core is usually the highest-performance choice for large multidimensional training.
- $B200$-class: same as $H100$, with more cautious qualification.
- $MI300X$ and $MI350$-class: FSDP/DTensor often provides the least risky portability path unless Megatron feature coverage is fully qualified.

---

# 19. Example Starting Configurations by Platform

## 19.1 Dense Training Starting Points

| Platform | Initial dense layout bias |
|---|---|
| $A100$ | moderate $TP$, moderate/deeper $PP$, FSDP/ZeRO over $D$ |
| $H100$ | higher local $TP$, moderate $PP$, optional $CP$, FSDP or distributed optimizer over $D$ |
| $B200$-class | larger local $TP$ and $CP$, shallower $PP$, stronger low-precision potential |
| $MI300X$ | moderate $TP$, shallower $PP$, FSDP attractive due to HBM |
| $MI350$-class | same as $MI300X$ with more room for local groups if software allows |

---

## 19.2 Long-Context Starting Points

| Platform | Initial long-context bias |
|---|---|
| $A100$ | FlashAttention + checkpointing + moderate $CP$ |
| $H100$ | FlashAttention + larger $CP$ + strong local overlap |
| $B200$-class | larger $CP$ and low-precision attention where qualified |
| $MI300X$ | exploit HBM first, then add $CP$ |
| $MI350$-class | same as $MI300X$ with more aggressive local grouping if stable |

---

## 19.3 MoE Starting Points

| Platform | Initial MoE bias |
|---|---|
| $A100$ | local $EP$, conservative $ETP$ |
| $H100$ | local $EP$, grouped GEMM, optional $ETP$ for large experts |
| $B200$-class | larger local $EP$ islands, optional larger $ETP$ |
| $MI300X$ | local $EP$, BF16-first, conservative $ETP$ |
| $MI350$-class | local $EP$, qualified $ETP$ if kernels and dispatch scale |

---

# 20. Pseudocode: Full Deployment Planner

```text
Input:
  platform type
  cluster topology
  model specification
  workload type
  sequence length
  target precision
  checkpoint state
  data manifest

Output:
  complete deployment plan

Procedure:
  1. Discover and fingerprint topology.

  2. Select platform-specific software image:
       - driver/runtime
       - communication library
       - compiler stack
       - kernel stack
       - profiler stack

  3. Enumerate candidate tuples:
       - dense: (D, T, P, C)
       - MoE: (D, P, C, T_d, E, T_e)

  4. For each tuple:
       a. verify divisibility and kernel support
       b. estimate persistent state memory
       c. estimate activation memory
       d. estimate transient communication buffers
       e. reject if any rank exceeds HBM
       f. score topology placement quality
       g. estimate exposed communication
       h. estimate pipeline bubble
       i. estimate data-path demand

  5. Select tuple with:
       - valid fit on all ranks
       - best topology score
       - acceptable bubble efficiency
       - sustainable data feed
       - acceptable numerical risk

  6. Build deterministic rank map.

  7. Validate:
       - checkpoint compatibility
       - tokenizer identity
       - data cursor identity
       - auto-resume policy

  8. Install profiler hooks and regression gates.

  9. Launch only if all preflight checks pass.
```

---

# 21. Production Readiness Checklist

## 21.1 Hardware

- topology graph recorded
- failed links and unhealthy GPUs excluded
- GPU/NIC affinity validated
- local fabric health validated
- storage path and bandwidth validated

## 21.2 Software

- exact container digest pinned
- driver/runtime matrix pinned
- NCCL/RCCL version pinned
- compiler and kernel stack pinned
- profiler stack installed

## 21.3 Parallelism

- $TP$ local
- sequence parallel aligned with $TP$
- $CP$ local when possible
- $EP$ local when possible
- $PP$ justified by fit or balance
- $DP$/FSDP across widest remaining domain

## 21.4 Memory

- persistent state modeled
- activation memory modeled
- transient gather buffers modeled
- fragmentation headroom reserved
- smallest-HBM rank validated

## 21.5 Numerics

- BF16 baseline validated
- lower precision validated against baseline
- clipping verified globally
- fused/unfused parity checked
- multi-node parity checked

## 21.6 Data

- dedup manifest fixed
- tokenizer fingerprint fixed
- packing correctness validated
- resume-safe cursoring enabled
- local cache or mmap path validated

## 21.7 Operations

- checkpoint schema canonicalized
- auto-resume tested
- step-time decomposition enabled
- strong and weak scaling baselines recorded
- alerting and log collation enabled

---

# 22. Final Technical Conclusions

1. **$A100$ clusters reward conservative, well-overlapped BF16 deployments with carefully bounded $TP$ and earlier use of checkpointing and FSDP.**

2. **$H100$ clusters support materially larger local communication groups and make FP8 production-feasible, but only after rigorous parity validation.**

3. **$B200$-class clusters must be treated as a new optimization regime, not as a faster $H100$. Bucket sizes, group sizes, and low-precision policies must be recalibrated from first principles.**

4. **$MI300X$ changes the memory trade space through large HBM, often reducing the need for deep pipeline parallelism, but only if xGMI locality and ROCm kernel maturity are treated as first-class constraints.**

5. **$MI350$-class clusters extend the AMD local-group opportunity set, but they require a fresh qualification cycle for collectives, fused kernels, graph capture, and lower-precision execution paths.**

6. **Across all hardware families, topology-aware placement of $TP$, $CP$, and $EP$ on the fastest local fabric is the dominant determinant of scalable performance.**

7. **The correct deployment blueprint is never hardware-only. It is the joint solution across topology, memory, numerics, kernels, data throughput, checkpoint semantics, and operational resilience.**

---

# 23. Reference Links

## Distributed Training and Frameworks
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- PyTorch DTensor: https://pytorch.org/docs/stable/distributed.tensor.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Megatron-LM / Megatron-Core: https://github.com/NVIDIA/Megatron-LM

## Communication Libraries
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/
- nccl-tests: https://github.com/NVIDIA/nccl-tests
- rccl-tests: https://github.com/ROCm/rccl-tests

## Profiling
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- ROCm profiling tools: https://rocm.docs.amd.com/

## Kernel Optimization
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Triton: https://github.com/triton-lang/triton
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine

```coming_soon
the next deliverable should be a platform-by-platform concrete deployment workbook for specific model classes such as:

- 7B
- 34B
- 70B
- 175B
- long-context dense variants
- $8 \times 7B$ and $16 \times 7B$ MoE variants

including exact recommended tuples, stage boundaries, memory budgets, and per-platform rank maps.
```