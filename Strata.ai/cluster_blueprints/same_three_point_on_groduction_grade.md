# Technical Report: Hardware-Specific LLM Training Deployment, Runtime Interoperability, and Long-Context/MoE Parallel Architecture

---

## 1. Scope, Assumptions, and Engineering Invariants

This report defines three production-grade deliverables:

- **A. Hardware-specific deployment blueprint** for $A100$, $H100$, $B200$/$Blackwell$-class, $MI300X$, and $MI350$-class clusters.
- **B. Megatron-Core + DeepSpeed + FSDP interoperability report** covering runtime ownership boundaries, checkpoint interoperability, resume semantics, and cross-vendor portability.
- **C. Long-context and MoE training architecture report** with exact parallel-group factorization, memory formulas, topology placement rules, and deterministic training constraints.

### 1.1 Non-Negotiable Engineering Rules

> **Place the highest-frequency collectives on the fastest fabric.**  
> In practice, this means $TP$, $EP$, and often $CP$ must remain inside $NVLink$/$NVSwitch$ or $xGMI$ islands whenever possible. $PP$ may cross slower domains. $DP$ should be outermost.

> **Use one primary owner per state dimension.**  
> Do not simultaneously allow Megatron model-parallel sharding and ZeRO/FSDP full-shard logic to own the same parameter partitioning semantics without an explicit interoperability layer.

> **A model “fits” only if the worst rank fits.**  
> HBM headroom, allocator fragmentation, communication buffers, graph-capture pools, and checkpoint staging buffers must all be budgeted against the smallest usable memory rank in the critical parallel group.

> **Numerical compression is never adopted on faith.**  
> $FP8$, $MXFP8$, $MXFP6$, and $MXFP4$ require single-node parity, multi-node parity, and resumed-run parity against a trusted $BF16$ baseline.

---

## 2. Notation and Exact Parallelism Model

### 2.1 Symbols

| Symbol | Meaning |
|---|---|
| $W$ | Total accelerator count |
| $D$ | Data parallel degree |
| $T$ | Dense tensor parallel degree |
| $P$ | Pipeline parallel stage count |
| $C$ | Context parallel degree |
| $E$ | Expert parallel degree |
| $T_e$ | Expert tensor parallel degree |
| $SP$ | Sequence parallelism; activation sharding inside the $TP$ group |
| $B_\mu$ | Microbatch size per pipeline replica |
| $G$ | Gradient accumulation steps |
| $m$ | Number of in-flight microbatches in pipeline schedule |
| $S$ | Sequence length |
| $L$ | Number of transformer layers |
| $H$ | Hidden size |
| $N$ | Total parameter count |
| $N_{\text{dense}}$ | Dense parameter count |
| $N_{\text{expert}}$ | Expert parameter count |
| $b_w$ | Bytes per stored weight |
| $b_g$ | Bytes per stored gradient |
| $b_o$ | Bytes per optimizer state, including master weights if present |
| $b_a$ | Bytes per saved activation |

### 2.2 Exact World-Size Factorizations

For dense training with $DP + TP + PP + CP$:

$$
W = D \cdot T \cdot P \cdot C
$$

with the important constraint:

- $SP$ **does not introduce an additional multiplicative world-size dimension**.
- $SP$ shards activation-related tensors **inside** the existing $TP$ group.

For dense + MoE with orthogonal dense and expert tensor parallelism:

$$
W = D \cdot P \cdot C \cdot T \cdot E \cdot T_e
$$

For dense + MoE where expert tensor parallelism is aliased to dense tensor parallelism:

$$
W = D \cdot P \cdot C \cdot T \cdot E
$$

### 2.3 Exact Group Definitions

For dense training:

- $DP$ group: vary $d$, fix $(p,t,c)$
- $TP$ group: vary $t$, fix $(d,p,c)$
- $PP$ group: vary $p$, fix $(d,t,c)$
- $CP$ group: vary $c$, fix $(d,p,t)$
- $SP$: same ranks as the $TP$ group

For dense + MoE with orthogonal $T_e$:

- $EP$ group: vary $e$, fix $(d,p,c,t,t_e)$
- expert-$TP$ group: vary $t_e$, fix $(d,p,c,t,e)$
- MoE compute group: vary $(e,t_e)$, fix $(d,p,c,t)$

### 2.4 Deterministic Rank Linearization

A deterministic linearization for orthogonal $DP/PP/CP/TP/EP/T_e$ is:

$$
r = (((((d \cdot P + p)\cdot C + c)\cdot T + t)\cdot E + e)\cdot T_e + t_e)
$$

This is not the only valid layout. The correct production choice is the one that maps the innermost varying dimensions to the fastest physical fabric.

---

## 3. Parallelism Design from First Principles

### 3.1 Global Batch and Token Accounting

The exact global sample batch is:

$$
B_{\text{global}} = B_\mu \cdot G \cdot D
$$

The effective tokens per optimizer step are:

$$
\text{tokens/step} = B_\mu \cdot G \cdot D \cdot S \cdot \rho_{\text{pack}}
$$

where $\rho_{\text{pack}}$ is the sequence-packing utilization ratio.

### 3.2 Pipeline Bubble

For a standard $1F1B$ schedule:

$$
\beta_{\text{bubble}} = \frac{P - 1}{m + P - 1}
$$

For interleaved virtual pipeline stages with $v$ virtual chunks per physical stage, the bubble is reduced approximately in proportion to the shorter effective pipeline depth. In practice, interleaving is most valuable when:

- $P$ is large,
- $B_\mu$ is memory-constrained,
- and raising $m$ directly would violate HBM limits.

### 3.3 Collective Cost Models

For large-message ring collectives over $p$ ranks with payload $n$, effective bandwidth $B_{\text{eff}}$, and startup latency $\alpha$:

**All-reduce**

$$
t_{\text{AR}} \approx 2 \cdot \frac{p-1}{p} \cdot \frac{n}{B_{\text{eff}}} + 2(p-1)\alpha
$$

**Reduce-scatter**

$$
t_{\text{RS}} \approx \frac{p-1}{p} \cdot \frac{n}{B_{\text{eff}}} + (p-1)\alpha
$$

**All-gather**

$$
t_{\text{AG}} \approx \frac{p-1}{p} \cdot \frac{n}{B_{\text{eff}}} + (p-1)\alpha
$$

These equations directly explain why:

- ZeRO-2/FSDP-style $RS+AG$ frequently outperforms replicated gradient all-reduce when overlap is good.
- ZeRO-3/FSDP full shard can become parameter-gather bound on slower inter-node fabrics.
- MoE $A2A$ patterns are often more fragile than dense $AR/RS/AG$ because imbalance penalties compound the network cost.

---

## 4. A. Hardware-Specific Deployment Blueprint

## 4.1 Hardware-Class Summary

| Accelerator class | Practical memory posture | Fastest collective domain | Best default numeric path | Primary scaling risk | Recommended dominant inner dimensions |
|---|---|---|---|---|---|
| $A100$ $80GB$ | HBM-constrained for large dense and long-context runs | 8-GPU $NVSwitch$/$NVLink$ node | $BF16$ | activation memory and PP pressure | $TP$, then limited $CP$ or $EP$ inside node |
| $H100$ $80GB/94GB$ | balanced compute, memory, and interconnect | 8-GPU $NVSwitch$ domain | $BF16 \rightarrow FP8$ after parity | cross-node imbalance at high $CP/EP$ | $TP$, $CP$, $EP$ inside node or rack |
| $B200$/$Blackwell$-class | high compute and larger HBM budget | node or rack-scale $NVLink$ domain | $BF16 \rightarrow FP8/MXFP8$ | comm topology misuse more than raw memory | keep $TP \times CP \times EP$ inside rack-scale domain |
| $MI300X$ | very high HBM; good for long-context and large experts | 8-GPU $xGMI$ node | $BF16$ baseline; vendor $FP8$ only after certification | ROCm kernel maturity asymmetry and off-node collective sensitivity | $TP$ and $EP$ inside node; $PP$ or outer $DP$ across nodes |
| $MI350$-class | larger HBM and stronger intra-node links than $MI300X$ class | $xGMI$-class node domain | $BF16 \rightarrow class-native low precision after qualification | backend drift across ROCm/compiler stacks | reduce $PP$, increase $CP$ or expert capacity locally |

### 4.1.1 Placement Rule by Fabric

- **$TP$**: always on the fastest, lowest-latency fully connected domain.
- **$EP$**: same requirement as $TP$ unless expert capacity is small and routing is mostly local.
- **$CP$**: inside node if possible; across nodes only when long-context memory savings justify the K/V exchange.
- **$PP$**: acceptable across nodes, because the primitive is point-to-point send/recv, not a high-frequency all-reduce.
- **$DP$**: outermost; span racks and slower fabric domains.

---

## 4.2 A100 Cluster Blueprint

### 4.2.1 Where A100 Still Makes Sense

$A100$ remains a credible platform for:

- dense pretraining in the $7B$–$70B$ class with disciplined $TP/PP$,
- continued pretraining and SFT workloads,
- moderate-context MoE if expert dispatch is kept local,
- cost-sensitive clusters where software maturity is more important than absolute throughput.

### 4.2.2 Deployment Priorities

- Use $BF16$ as the default numeric path.
- Enable $SP$ whenever $T > 1$.
- Favor **higher $PP$** than on newer GPUs because $80GB$ HBM becomes the limiting factor first.
- Keep $TP \in \{2,4,8\}$ inside the 8-GPU node.
- Use $CP$ only when long context cannot fit with checkpointing plus $PP$.
- If inter-node bandwidth is modest, avoid off-node $EP$.

### 4.2.3 Recommended Factorization Envelope

**Dense training**

- Small/medium model: prefer $T=4$, $P=2$ or $4$, $D$ outer.
- Large dense model: prefer $T=8$, $P=4$ or $8$, $SP$ enabled, activation checkpointing mandatory.

**Long-context dense**

- First raise $P$ and checkpointing depth.
- Add $C=2$ only when $S$ forces it.
- Avoid large off-node $C$ because attention exchange can dominate.

**MoE**

- Keep $E$ within a node whenever possible.
- If experts exceed one node, route entire $EP$ groups over homogeneous node sets and validate all-to-all bandwidth separately from dense training.

### 4.2.4 Reference Layout

For a $64$-GPU $A100$ cluster running a memory-constrained dense pretraining job:

$$
W = 64 = D \cdot T \cdot P \cdot C = 2 \cdot 8 \cdot 4 \cdot 1
$$

Interpretation:

- $T=8$: one full node per tensor group
- $P=4$: four pipeline stages
- $D=2$: two data replicas
- $SP$: enabled inside each $TP$ group

This layout keeps the heaviest collective path inside the node and allows $PP$ and $DP$ to absorb inter-node traffic.

### 4.2.5 Operational Risks on A100

- Over-aggressive $CP$ destroys throughput before it saves enough memory.
- Missing $SP$ causes avoidable activation blow-up.
- $FP16$ may be numerically inferior to $BF16$ for large-scale runs.
- Small buckets and poor overlap lead to visible NCCL startup-latency domination.

---

## 4.3 H100 Cluster Blueprint

### 4.3.1 Why H100 Is the Current General-Purpose Training Sweet Spot

$H100$ is the most balanced production choice when the objective is:

- dense pretraining at scale,
- long-context training with $CP$,
- production MoE with moderate or large $EP$,
- $FP8$ training after $BF16$ parity certification.

### 4.3.2 Deployment Priorities

- Start with $BF16$; promote to $FP8$ only after parity gates.
- Use $TP=4$ or $8$ depending hidden size and GEMM efficiency.
- Keep $CP$ inside node or rack-scale fast domain when possible.
- Prefer $RS/AG$-based distributed optimizers over monolithic gradient all-reduce.
- Use interleaved $PP$ when $P \ge 8$ and microbatch count is limited.

### 4.3.3 Recommended Factorization Envelope

**Dense pretraining**

- $T=4$, $P=4$ or $8$, $D$ outermost
- $SP$ always on for $T>1$

**Long-context dense**

- $T=4$, $C=2$ or $4$
- $P=4$ or $8$ depending layer count and HBM budget

**MoE**

- $E=4$ or $8$ inside a fast domain
- expert tensor parallelism only when expert MLPs are themselves too large or kernel shape efficiency suffers without sharding

### 4.3.4 Reference Layout

For a $128$-GPU $H100$ cluster running dense long-context training:

$$
W = 128 = D \cdot T \cdot P \cdot C = 2 \cdot 4 \cdot 8 \cdot 2
$$

This is an operationally clean layout:

- each pipeline stage uses $T \cdot C = 8$ GPUs,
- a full stage can be packed inside one 8-GPU node if the node-local topology permits the intended $TP/CP$ map,
- $D=2$ creates two complete pipeline replicas.

### 4.3.5 H100-Specific Kernel Policy

- FlashAttention is mandatory for long context.
- Fused MLP, fused RMSNorm, fused RoPE, and graph capture materially affect step-time variance.
- $FP8$ should be enabled first on matmuls and only later extended to broader operator coverage.
- Validate resumed-run parity, not just fresh-start parity.

---

## 4.4 B200 / Blackwell-Class Blueprint

### 4.4.1 Distinguish the Deployment Shape Before Choosing the Plan

There are two materially different Blackwell-class deployments:

- **8-GPU node-centric systems**
- **rack-scale NVLink domains** such as large NVLink fabrics where dozens of GPUs form one high-bandwidth island

The GPU SKU alone is not the topology. The topology determines the correct parallel plan.

### 4.4.2 Deployment Priorities

- Keep $TP \times CP \times EP$ inside the largest fully connected NVLink domain available.
- Reduce $PP$ relative to $A100/H100$ when larger HBM permits it.
- Use $FP8$ or $MXFP8$ only after $BF16$ parity.
- Use $MXFP6/MXFP4$ only in tightly controlled ablations with optimizer-state precision validation.

### 4.4.3 Recommended Factorization Strategy

**If the deployment is node-centric**

- same logic as $H100$, but use larger local memory to reduce $PP$ and increase $C$ or local expert capacity.

**If the deployment is rack-scale NVLink**

- place all latency-sensitive parallelism inside the rack:
  - $TP$
  - $CP$
  - $EP$
  - expert-$TP$ if required
- place $DP$ across racks
- use $PP$ across racks only when model size forces it

### 4.4.4 Reference Rack-Scale Principle

> **For Blackwell-class racks, the rack is the new “node” for collective planning.**  
> If a rack-scale NVLink domain exists, treat it as the inner placement boundary for all high-frequency collectives.

### 4.4.5 When B200 Changes the Architecture

Compared with $H100$:

- more runs become **communication-bound before memory-bound**
- MoE dispatch locality matters even more because the platform can expose network design mistakes quickly
- larger local memory often makes lower $PP$ the correct answer
- very long context becomes practical with fewer compromises

---

## 4.5 MI300X Cluster Blueprint

### 4.5.1 Where MI300X Is Structurally Strong

$MI300X$ is especially strong for:

- long-context training due to large HBM,
- large expert capacity per rank,
- model classes where reducing or eliminating $PP$ yields major simplification,
- cross-vendor programs that require ROCm viability.

### 4.5.2 Deployment Priorities

- Use $BF16$ as the certification baseline.
- Keep $TP$ inside the 8-GPU $xGMI$ node.
- Prefer fewer $PP$ stages than on $A100$ or $H100$ for the same model.
- Avoid off-node $EP$ unless all-to-all bandwidth is characterized explicitly.
- Validate every fused kernel path on ROCm separately; do not assume CUDA parity by inspection.

### 4.5.3 Reference Layout

For a $64$-GPU $MI300X$ cluster running dense training:

$$
W = 64 = D \cdot T \cdot P \cdot C = 2 \cdot 8 \cdot 4 \cdot 1
$$

This matches the node-centric nature of the hardware:

- $T=8$: one full $xGMI$ node
- $P=4$: four stages
- $D=2$: two replicas

For long context, a practical alternative is:

$$
W = 64 = 1 \cdot 8 \cdot 4 \cdot 2
$$

which removes $DP$ and uses $C=2$ when sequence length is the real bottleneck.

### 4.5.4 ROCm-Specific Cautions

- Keep RCCL and ROCm driver/compiler versions pinned as a tested tuple.
- Some fused kernels may regress across compiler releases even when correctness remains intact.
- Graph capture and persistent-kernel stability must be revalidated after every stack upgrade.
- Numerical parity must be checked across CUDA and ROCm on identical seeds and sample order.

---

## 4.6 MI350-Class Blueprint

### 4.6.1 Planning Rule

Treat $MI350$-class systems as:

- a higher-HBM, higher-bandwidth successor class to $MI300X$,
- still requiring topology-first placement,
- still requiring ROCm stack pinning and backend-specific kernel certification.

### 4.6.2 Expected Architectural Changes Relative to MI300X

Assuming the common class improvements hold in the deployed SKU:

- fewer $PP$ stages required for the same model
- more practical $CP$ at larger $S$
- larger local expert counts before off-node $EP$ is required
- reduced pressure to use ZeRO-3/FSDP full-shard purely for memory fit

### 4.6.3 Reference Layout

For a $128$-GPU $MI350$-class cluster running long-context MoE with orthogonal expert tensor parallelism:

$$
W = 128 = D \cdot P \cdot C \cdot T \cdot E \cdot T_e = 2 \cdot 4 \cdot 4 \cdot 2 \cdot 2 \cdot 2
$$

This is not universal, but it is structurally valid:

- dense $TP=2$
- $CP=4$
- $EP=2$
- expert-$TP=2$
- $P=4$
- $D=2$

Use this style of factorization only if the physical topology can place the combined $T \times E \times T_e$ groups on the fastest local fabric.

---

## 5. Hardware-Aware World-Size Factorization Rules

## 5.1 Topology-First Heuristic

1. Measure the fabric graph.
2. Identify the largest low-latency collective cliques.
3. Assign:
   - $TP$ first
   - then $EP$
   - then $CP$
4. Assign $PP$ across remaining homogeneous domains.
5. Assign $DP$ outermost.

### 5.1.1 Practical Ordering

A reliable default ordering is:

- **intra-node or intra-rack**: $TP$, $EP$, optionally $CP$
- **cross-node**: $PP$
- **cross-rack**: $DP$

### 5.1.2 When to Break the Default

Break the default only when one of the following is provably true:

- HBM fit requires $CP$ before additional $PP$
- expert routing locality strongly favors a different $EP$ placement
- rack-scale NVLink makes cross-node collective cost effectively local
- the network is strong enough that keeping $PP$ local and moving $CP$ outward improves total step time

---

## 6. B. Megatron-Core + DeepSpeed + FSDP Interoperability Report

## 6.1 Runtime Ownership Boundaries

### 6.1.1 Megatron-Core

Megatron-Core is the preferred owner for:

- tensor parallelism
- sequence parallelism
- pipeline parallelism
- context parallelism
- expert parallelism
- interleaved schedules
- fused transformer kernels on NVIDIA-centric large-scale training

### 6.1.2 DeepSpeed

DeepSpeed is strongest when the primary requirement is:

- ZeRO optimizer sharding
- offload in memory-constrained environments
- legacy DeepSpeed operational estates
- some hybrid MoE and pipeline scenarios where organizational tooling already centers on DeepSpeed

### 6.1.3 FSDP / FSDP2 / DTensor

FSDP is strongest when the primary requirement is:

- portability
- standardized sharded state dicts
- PyTorch-native mesh semantics
- moderate-to-large scale without dependency on Megatron-specific fast paths
- cross-vendor maintainability

---

## 6.2 Interoperability Matrix

| Composition | Production recommendation | Best use case | Primary risk |
|---|---|---|---|
| Megatron-Core model parallel + Megatron distributed optimizer | **Preferred** on NVIDIA at largest scale | dense and MoE pretraining with $TP/PP/CP/EP$ | lower portability than PyTorch-native stacks |
| Megatron-Core + DeepSpeed ZeRO-1/2 | **Viable** | hybrid estates needing DeepSpeed optimizer/runtime features | duplicated state ownership if not carefully separated |
| Megatron-Core + DeepSpeed ZeRO-3 | **Conditional only** | memory emergency cases where full sharding is unavoidable | parameter all-gather pressure, checkpoint complexity, interaction fragility with heavy model parallelism |
| Pure PyTorch FSDP2 + DTensor TP | **Preferred** for portability | mixed-vendor or maintainability-first programs | may trail Megatron fast paths at extreme scale |
| FSDP wrapping already TP-sharded Megatron modules | **Generally avoid** | only under explicitly validated narrow designs | flat-param semantics and shard ownership conflicts |
| DeepSpeed pipeline + Megatron TP/EP/CP | **Conditional** | legacy DS pipeline estates | harder debugging, duplicated scheduler semantics |

### 6.2.1 Key Rule

> **Only one system should define how a tensor is partitioned for optimization and checkpointing.**  
> If Megatron owns model-parallel layout, ZeRO/FSDP should own only the $DP$-axis sharding semantics, never silently reinterpret model-parallel shards.

---

## 6.3 Recommended Interoperability Patterns

### 6.3.1 Pattern 1: NVIDIA Large-Scale Pretraining

- Megatron-Core for $TP/PP/SP/CP/EP$
- Megatron distributed optimizer or ZeRO-1/2 equivalent semantics on the $DP$ axis
- distributed checkpointing in logical-tensor form

**Use when**

- the goal is maximum throughput
- the deployment is mostly NVIDIA
- the model includes long-context or MoE features

### 6.3.2 Pattern 2: Portability-First Multi-Vendor Training

- PyTorch FSDP2 + DTensor
- backend-specific kernel substitution layer
- canonical sharded state dict and logical mesh schema

**Use when**

- portability and maintenance outrank absolute peak throughput
- the same model must train on CUDA and ROCm
- custom kernel ownership is acceptable

### 6.3.3 Pattern 3: Memory-Emergency Full Shard

- Megatron model parallel retained
- ZeRO-3/FSDP full shard only if static-state fit is impossible otherwise

**Use only when**

- $N$, $S$, or optimizer-state footprint cannot fit with $PP + TP + SP + checkpointing + ZeRO-2$

**Do not use by default** because the parameter all-gather schedule often becomes the bottleneck.

---

## 6.4 Checkpoint Interoperability Requirements

A production checkpoint must capture:

- logical tensor metadata
- physical shard metadata
- optimizer states
- scheduler state
- global step and consumed tokens
- RNG streams per rank and per parallel dimension
- sampler/dataloader position
- tokenizer identity and normalization spec
- dataset lineage manifest
- software stack manifest
- topology manifest

### 6.4.1 Canonical Logical Tensor Metadata

Each saved tensor should have:

- global tensor name
- global shape
- axis semantics
- source partition axes
- target-agnostic logical layout
- precision
- checksum
- shard map

This metadata is the foundation of resharding across:

- changed $D/T/P/C/E/T_e$
- changed runtime stack
- changed hardware cluster size
- training-to-inference conversion

---

## 6.5 Checkpoint Resharding Algorithm

### 6.5.1 Pseudocode: Logical-Tensor Resharding

```text
ALGORITHM ReshardCheckpoint
INPUT:
  source_manifest
  target_mesh
  target_partition_spec
  storage_backend

1. Load global checkpoint manifest.
2. For each logical tensor:
   a. Read its global shape and source shard metadata.
   b. Construct source coordinate map.
   c. Construct target coordinate map from target_mesh and target_partition_spec.
   d. For each target shard:
      i. Determine intersection regions with source shards.
      ii. Stream only overlapping slices from storage.
      iii. Reassemble target shard in canonical dtype.
      iv. Apply required transpose or axis remap if layout conventions differ.
      v. Write target shard and checksum.
3. Reshard optimizer states using the same logical parameter ids.
4. Rewrite scheduler, RNG, and dataloader state against the new world layout.
5. Validate:
   a. all logical tensors covered exactly once,
   b. checksums match,
   c. shard shapes match target spec.
6. Emit a target-side manifest with versioned schema.
OUTPUT:
  target_checkpoint
```

### 6.5.2 Critical Interoperability Detail

- **Do not key optimizer states by local parameter ordering alone.**
- Use stable global logical parameter identifiers.
- This is mandatory when changing:
  - $PP$ boundaries,
  - $TP$ degree,
  - MoE expert placement,
  - or runtime framework.

---

## 6.6 Resume Semantics Across World-Size Changes

A correct elastic resume must preserve:

- identical consumed-sample count
- deterministic next-sample selection
- optimizer-state identity
- RNG stream continuity per logical training event
- exact parameter reconstruction

### 6.6.1 What Can Safely Change

- $D$ can change most easily.
- $P$ can change if layer-to-stage mapping is resharded and optimizer states follow the new stage ownership.
- $T$, $C$, $E$, and $T_e$ can change only if the checkpoint stores logical tensors rather than framework-local shards.

### 6.6.2 What Commonly Breaks

- position in packed sample streams
- expert state naming when expert indices are remapped
- tied embedding/head tensors
- optimizer moments when parameter flattening order changed
- RNG-dependent dropout and routing parity

---

## 7. C. Long-Context and MoE Training Architecture

## 7.1 Exact Parallel-Group Factorization

### 7.1.1 Dense Long-Context

Exact factorization:

$$
W = D \cdot P \cdot T \cdot C
$$

with:

- $SP$ enabled within the $TP$ group
- no extra multiplicative term for $SP$

### 7.1.2 Long-Context + MoE with Orthogonal Expert Tensor Parallelism

Exact factorization:

$$
W = D \cdot P \cdot C \cdot T \cdot E \cdot T_e
$$

This form is required when:

- dense layers use one tensor mesh
- experts use an additional tensor split not identical to dense $TP$

### 7.1.3 Long-Context + MoE with Aliased Expert Tensor Parallelism

Exact factorization:

$$
W = D \cdot P \cdot C \cdot T \cdot E
$$

Use this when expert MLPs reuse the dense $TP$ axis and no extra orthogonal expert-$TP$ mesh exists.

---

## 7.2 Exact Static Memory Formulas

### 7.2.1 Dense Model State per Rank

For dense parameters under $TP$ and $PP$:

**ZeRO-0 / replicated optimizer**

$$
M_{\text{dense}}^{(0)} = \frac{N_{\text{dense}}}{P \cdot T} \left(b_w + b_g + b_o \right)
$$

**ZeRO-1**

$$
M_{\text{dense}}^{(1)} = \frac{N_{\text{dense}}}{P \cdot T} \left(b_w + b_g + \frac{b_o}{D} \right)
$$

**ZeRO-2**

$$
M_{\text{dense}}^{(2)} = \frac{N_{\text{dense}}}{P \cdot T} \left(b_w + \frac{b_g + b_o}{D} \right)
$$

**ZeRO-3 / FSDP full shard**

$$
M_{\text{dense}}^{(3)} = \frac{N_{\text{dense}}}{P \cdot T \cdot D} \left(b_w + b_g + b_o \right)
$$

### 7.2.2 Expert Model State per Rank

For expert parameters sharded by $EP$ and optional expert-$TP$:

**Replicated over $DP$**

$$
M_{\text{expert}}^{(0)} = \frac{N_{\text{expert}}}{P \cdot E \cdot T_e} \left(b_w + b_g + b_o \right)
$$

**ZeRO-1**

$$
M_{\text{expert}}^{(1)} = \frac{N_{\text{expert}}}{P \cdot E \cdot T_e} \left(b_w + b_g + \frac{b_o}{D} \right)
$$

**ZeRO-2**

$$
M_{\text{expert}}^{(2)} = \frac{N_{\text{expert}}}{P \cdot E \cdot T_e} \left(b_w + \frac{b_g + b_o}{D} \right)
$$

**ZeRO-3 / full shard**

$$
M_{\text{expert}}^{(3)} = \frac{N_{\text{expert}}}{P \cdot E \cdot T_e \cdot D} \left(b_w + b_g + b_o \right)
$$

### 7.2.3 Total Static State

$$
M_{\text{static}} = M_{\text{dense}} + M_{\text{expert}} + M_{\text{buffers}} + M_{\text{comm}} + M_{\text{fragmentation}}
$$

### 7.2.4 Fit Condition

The true fit condition is:

$$
M_{\text{static}} + M_{\text{act}} + M_{\text{runtime\_reserve}} \leq HBM_{\text{usable,min}}
$$

where $HBM_{\text{usable,min}}$ is the smallest usable HBM across the ranks in the critical model-parallel group.

---

## 7.3 Activation Memory for Long Context

Static state is exact. Activation memory depends on kernel save-for-backward behavior and must be modeled with calibrated coefficients.

A production-quality activation estimate for a decoder-only block is:

$$
M_{\text{act}} \approx \frac{L}{P} \cdot B_\mu \cdot \frac{S}{C} \cdot b_a \cdot
\left(
c_{\text{res}} H +
c_{\text{attn}} \frac{H}{T} +
c_{\text{ffn}} \frac{f_{\text{ffn}} H}{T}
\right)
\cdot \gamma_{\text{ckpt}}
$$

where:

- $f_{\text{ffn}}$ is the MLP expansion ratio
- $c_{\text{res}}, c_{\text{attn}}, c_{\text{ffn}}$ are kernel-calibrated coefficients
- $\gamma_{\text{ckpt}}$ is the activation checkpointing retention factor

### 7.3.1 Engineering Interpretation

- Activation memory scales linearly with local sequence length $S/C$.
- $CP$ is therefore the cleanest direct lever for long-context fit.
- $SP$ reduces the $TP$-shardable activation components but not every residual tensor.
- FlashAttention removes the quadratic attention-matrix storage term but not all attention activations.

---

## 7.4 Long-Context Technique Selection

| Technique | Primary benefit | Communication pattern | Best use case | Main risk |
|---|---|---|---|---|
| FlashAttention | removes explicit $S^2$ attention state | local kernel optimization | all modern long-context training | kernel/backend maturity on non-CUDA paths |
| $CP$ | reduces local sequence length | K/V exchange across context ranks | full-context exact training | off-node attention traffic |
| Ulysses-style sequence/head repartition | improves long-context scaling through reshaped collectives | all-to-all style redistribution | very long contexts where head partitioning is favorable | A2A sensitivity and backend support |
| Ring Attention | streams K/V through a ring | repeated send/recv or ring collective | very long exact attention with bounded local memory | latency accumulation and overlap quality |
| Sliding-window or blockwise attention | reduces compute and communication | local or structured | specialized long-context objectives | objective mismatch with full-context pretraining |

### 7.4.1 Selection Rule

- Use **FlashAttention** by default.
- Add **$CP$** when memory fit requires sequence partitioning.
- Use **Ring Attention** or **Ulysses** only when the context length is high enough that plain $CP$ plus FlashAttention is insufficient or inefficient.

---

## 7.5 Communication Volume for Long Context and MoE

### 7.5.1 Context Parallel Attention Exchange

A simplified per-layer K/V exchange estimate is:

$$
V_{\text{CP}} \propto B_\mu \cdot S \cdot H \cdot b_a \cdot \frac{C-1}{C}
$$

The exact constant depends on the attention implementation and whether K and V are exchanged once or streamed in chunks.

### 7.5.2 MoE Dispatch Volume

For top-$k$ routing with local token count:

$$
N_t = B_\mu \cdot \frac{S}{C}
$$

the dispatch-and-return activation traffic is approximately:

$$
V_{\text{EP}} \approx 2 \cdot k \cdot N_t \cdot H \cdot b_a \cdot (1 - f_{\text{local}})
$$

where $f_{\text{local}}$ is the fraction of tokens served by local experts.

### 7.5.3 Engineering Consequence

- Reducing $f_{\text{local}}$ is often more important than tuning a single NCCL bucket.
- Expert locality is a first-order systems problem, not a minor optimization.

---

## 7.6 MoE Load Balance, Capacity, and Token Dropping

### 7.6.1 Ideal Expert Load

For $E$ experts and top-$k$ routing:

$$
\bar{n}_e = \frac{k \cdot N_t}{E}
$$

### 7.6.2 Expert Capacity

With capacity factor $u$:

$$
\text{capacity} = \left\lceil u \cdot \bar{n}_e \right\rceil
$$

### 7.6.3 Load Imbalance Metric

$$
\lambda = \frac{\max_e n_e}{\bar{n}_e}
$$

### 7.6.4 Dropped Tokens

$$
n_{\text{drop}} = \sum_e \max(0, n_e - \text{capacity})
$$

### 7.6.5 Stability Rules

- Keep $\lambda$ near $1$.
- Scale router auxiliary losses consistently across $DP$ replicas.
- Apply deterministic tie-breaking in routing.
- Exclude dropped tokens consistently from loss normalization.
- Ensure all ranks observe the same global routing semantics, even when dispatch is local.

---

## 7.7 Deterministic MoE Routing Algorithm

```text
ALGORITHM DeterministicTopKMoERouting
INPUT:
  token_activations
  router_logits
  top_k
  expert_count E
  capacity_factor u
  deterministic_seed

1. Compute router probabilities from router_logits using the configured stable softmax path.
2. For each token:
   a. Select top_k experts with deterministic tie-breaking.
3. Count assigned tokens per expert.
4. Compute ideal load:
      ideal = top_k * token_count / E
5. Compute capacity:
      capacity = ceil(u * ideal)
6. For each expert:
   a. If assigned_count <= capacity:
      keep all assigned tokens
   b. Else:
      keep the highest-priority tokens under deterministic ordering
      mark the remainder as dropped or rerouted per policy
7. Build dispatch tables grouped by destination expert rank.
8. Exchange token payloads across EP ranks.
9. Run expert forward and backward.
10. Return outputs to source token order.
11. Log:
    a. imbalance metric,
    b. dropped-token ratio,
    c. local vs remote dispatch fraction.
OUTPUT:
  expert_outputs, routing_statistics
```

---

## 7.8 Combined Long-Context + MoE Placement Rules

### 7.8.1 Correct Ordering of Dimensions

For combined long-context MoE, the safest default ordering is:

1. $TP$
2. $EP$
3. $CP$
4. $PP$
5. $DP$

This changes only if:

- $CP$ must remain fully local while $EP$ can tolerate some network traffic,
- or the expert MLP kernels require $T_e$ alignment inside local fabric that would otherwise be broken.

### 7.8.2 Strong Production Rule

> **Do not allow both attention exchange and expert dispatch to cross the same weak fabric unless measured evidence shows the fabric is under-utilized.**  
> Otherwise the step becomes bimodally unstable and difficult to tune.

---

## 8. Data Pipeline as a Distributed System

## 8.1 Data-Lineage Requirements

A production LLM data pipeline must version:

- raw source manifest
- filtering rules
- dedup algorithm and thresholds
- tokenizer model and normalization rules
- packed-sequence construction policy
- shard boundaries and checksums
- sampling mixture weights
- curriculum schedule
- resume offsets

Without this lineage, convergence regressions are not scientifically attributable.

---

## 8.2 Curation, Deduplication, and Convergence

### 8.2.1 Deduplication Effects

Deduplication changes:

- sample diversity
- memorization pressure
- evaluation leakage risk
- effective gradient noise scale
- token-frequency distribution

### 8.2.2 Practical Rule

- perform exact and near-duplicate removal before final split assignment
- preserve source provenance at record granularity
- avoid over-deduplication that removes legitimate repeated structured content

> **Dedup is not a storage optimization. It is an optimization of gradient signal quality.**

---

## 8.3 Tokenizer Strategy

Tokenizer choice changes:

- tokens-per-byte
- average effective context occupancy
- multilingual fragmentation
- embedding table size
- cross-checkpoint compatibility

### 8.3.1 Production Guidance

- train tokenizer on the final intended corpus mix
- freeze it before large-scale training
- use byte fallback or equivalent robustness for out-of-vocabulary behavior
- never silently swap tokenizer versions mid-program unless checkpoint conversion includes embedding remapping

---

## 8.4 Sequence Packing

The effective packing utilization is:

$$
\rho_{\text{pack}} = \frac{\text{non-pad tokens}}{\text{allocated tokens}}
$$

Packing improves throughput but can silently corrupt training if:

- cross-document attention masking is incorrect
- loss masking leaks labels across boundaries
- resume accounting loses original sample identities

> **Packing is only a win if sample accounting remains exact. Otherwise it is silent label corruption.**

---

## 8.5 Deterministic Streaming Sampler

```text
ALGORITHM DeterministicStreamingSampler
INPUT:
  shard_manifest
  global_seed
  epoch_id
  world_layout
  resume_state

1. Build a stable ordered list of immutable token shards.
2. Derive a deterministic shard permutation from global_seed and epoch_id.
3. For each shard:
   a. derive a deterministic per-shard sample order
   b. partition samples among DP replicas without overlap
4. Maintain per-rank cursor:
   a. current shard id
   b. sample offset
   c. packed-sequence residual buffer
5. On checkpoint:
   a. persist all cursors and residual packing state
6. On resume:
   a. restore cursors exactly
   b. verify shard checksums and tokenizer hash
OUTPUT:
  deterministic sample stream
```

---

## 9. Communication Internals and Failure Diagnosis

## 9.1 Primitive-to-Use-Case Map

| Primitive | Typical owner | Main use |
|---|---|---|
| all-reduce | $TP$, replicated $DP$ gradients | dense synchronization |
| reduce-scatter | distributed optimizer, FSDP/ZeRO | shard gradients |
| all-gather | ZeRO/FSDP, tensor reconstruction | parameter or grad materialization |
| all-to-all | MoE dispatch, Ulysses-style repartition | token or head exchange |
| send/recv | $PP$, ring attention | stage transfer |
| broadcast | metadata, parameters, scalers | one-to-many sync |

---

## 9.2 Bandwidth Collapse: Root Causes

If collective bandwidth collapses, the likely causes are:

- topology mismatch between rank map and physical fabric
- PCIe fallback instead of NVLink/$xGMI$
- NIC/GPU NUMA misbinding
- undersized buckets causing latency domination
- too many concurrent communicators
- mixed traffic patterns from $CP$ and $EP$ colliding on the same links
- CPU-side launch starvation
- storage jitter causing apparent communication stalls

### 9.2.1 Required Evidence Before Root-Causing Model Code

Collect:

- NCCL/RCCL traces
- Nsight Systems or rocprof timeline
- IB/RDMA counters or Ethernet congestion counters
- topology export
- per-rank step-time histogram
- data-loader wait time
- expert imbalance statistics

If these are absent, the diagnosis is incomplete.

---

## 9.3 Deadlocks and Desynchronization

Typical root causes:

- divergent control flow between ranks
- mismatched collective ordering across $PP$ stages
- asynchronous error on a subset of ranks
- communicator teardown or recreation in inconsistent order
- shape drift under variable-length batches without deterministic padding contract

### 9.3.1 Deadlock Rule

> **If a distributed run deadlocks at scale but not on one node, suspect collective ordering and topology before suspecting the model.**

---

## 10. Kernel and Numerical Optimization

## 10.1 Kernel Priorities by Impact

1. FlashAttention
2. fused MLP
3. fused RMSNorm
4. fused RoPE
5. fused softmax where FlashAttention is not in the path
6. persistent kernels where supported
7. graph capture for stable-shape regions
8. Triton/CUDA/HIP custom kernels only for proven gaps

### 10.1.1 Occupancy vs Register Pressure

A faster kernel is not the one with the highest theoretical occupancy. The correct kernel maximizes end-to-end layer throughput after accounting for:

- register pressure
- shared memory pressure
- HBM traffic
- launch count
- graph-capture compatibility
- numerical stability in reduced precision

---

## 10.2 Precision Policy

### 10.2.1 Recommended Progression

- certify in $BF16$
- validate fused vs unfused parity
- validate single-node vs multi-node parity
- then enable $FP8$ or vendor low-precision path
- only then test microscaling variants

### 10.2.2 Stability Controls

- gradient clipping
- overflow/underflow guards
- stable softmax
- optimizer master state precision policy
- stochastic rounding where supported and beneficial
- parity gates on first-step, short-run, and resumed-run loss curves

---

## 11. Measurement, Scaling Science, and Regression Gates

## 11.1 Step-Time Decomposition

A production step-time model should report:

$$
t_{\text{step}} =
t_{\text{fwd}} +
t_{\text{bwd}} +
t_{\text{optimizer}} +
t_{\text{visible\_comm}} +
t_{\text{data\_stall}} +
t_{\text{pipeline\_idle}} +
t_{\text{checkpoint}}
$$

The value of overlap is determined by how much communication remains in $t_{\text{visible\_comm}}$ after hiding.

---

## 11.2 MFU and HFU

Model FLOP utilization:

$$
MFU = \frac{F_{\text{model step}}}{W \cdot F_{\text{peak}} \cdot t_{\text{step}}}
$$

Hardware FLOP utilization uses executed FLOPs, which may include recomputation and overhead. Both should be tracked. $MFU$ is the better metric for training efficiency; $HFU$ can be misleadingly high when recomputation grows.

---

## 11.3 Scaling Efficiency

Strong-scaling efficiency:

$$
\eta_{\text{strong}}(n) = \frac{X(n)}{n \cdot X(1)}
$$

where $X(n)$ is throughput at $n$ devices for fixed global workload.

Weak-scaling efficiency:

$$
\eta_{\text{weak}}(n) = \frac{X(n)}{X(1)}
$$

for constant per-device workload.

### 11.3.1 Required Ablations

Every optimization should have:

- controlled seed
- fixed data order
- fixed tokenizer and data lineage
- step-time decomposition
- memory breakdown
- communication breakdown
- loss parity versus trusted baseline

---

## 12. Failure-Resilient Automation

## 12.1 Production Bootstrap Sequence

```text
ALGORITHM ClusterBootstrapAndPreflight
INPUT:
  cluster_inventory
  container_manifest
  training_spec

1. Discover:
   a. GPU inventory
   b. local and global topology graph
   c. NIC-to-GPU affinity
   d. driver/runtime/library versions
2. Validate:
   a. container/runtime compatibility
   b. NCCL/RCCL health
   c. GPUDirect/RDMA availability
   d. filesystem/object-store throughput
3. Benchmark:
   a. all-reduce
   b. reduce-scatter
   c. all-gather
   d. all-to-all
   at multiple message sizes
4. Synthesize candidate parallel plans from topology and training_spec.
5. Reject any plan failing:
   a. HBM fit,
   b. bandwidth floor,
   c. stage balance,
   d. checkpoint compatibility.
6. Launch a short deterministic smoke test.
7. Validate:
   a. first-step loss,
   b. multi-step parity,
   c. checkpoint save/load,
   d. auto-resume.
8. Promote the plan to full training.
OUTPUT:
  validated launch plan
```

---

## 12.2 Auto-Resume and Partial World-Size Restart

A resilient production workflow must support:

- node loss detection
- rank eviction or restart
- checkpoint integrity verification
- resharded restart to a new $D/T/P/C/E/T_e$ plan
- deterministic continuation of sample consumption

### 12.2.1 Storage-Aware Checkpointing

Use:

- asynchronous checkpoint writes
- chunked tensor serialization
- checksum validation
- staggered flush scheduling
- metadata-first commit protocol
- immutable completed checkpoints

Do not allow partially written checkpoints to appear valid.

---

## 13. Recommended Architecture Choices by Objective

## 13.1 If the Objective Is Maximum NVIDIA Throughput

Use:

- Megatron-Core
- native $TP/PP/SP/CP/EP$
- distributed optimizer or ZeRO-1/2 style sharding on $DP$
- logical distributed checkpointing
- FlashAttention + fused transformer kernels
- $BF16 \rightarrow FP8$ after parity

## 13.2 If the Objective Is Portability Across NVIDIA and AMD

Use:

- PyTorch FSDP2 + DTensor as canonical mesh abstraction
- backend-specific kernel registry
- logical checkpoint schema
- $BF16$ baseline on both CUDA and ROCm
- vendor low precision only after separate certification

## 13.3 If the Objective Is Long-Context Dense Training

Use:

- FlashAttention first
- then $CP$
- then interleaved $PP$ if needed
- keep $TP$ inside the fastest local fabric
- avoid off-node $CP$ unless memory fit requires it and measured bandwidth is adequate

## 13.4 If the Objective Is Large-Scale MoE

Use:

- local $EP$ first
- expert locality-aware placement
- deterministic routing and capacity enforcement
- all-to-all profiling before full model launch
- expert-$TP$ only when expert MLPs are too large or shape efficiency demands it

---

## 14. Final Engineering Conclusions

### 14.1 Hardware-Specific Blueprint

- **$A100$**: memory-first planning; use higher $PP$, strict $SP$, cautious $CP$.
- **$H100$**: best balanced general-purpose platform; $TP + CP + EP$ inside fast local domains; adopt $FP8$ after parity.
- **$B200$/$Blackwell$-class**: topology dominates; exploit rack-scale fast domains; lower $PP$, higher local $CP/EP$.
- **$MI300X$**: HBM-rich and strong for long context; keep $TP/EP$ local; certify ROCm kernels independently.
- **$MI350$-class**: extend the $MI300X$ playbook with lower $PP$ and larger local memory headroom, but maintain strict ROCm qualification discipline.

### 14.2 Runtime Interoperability

- Megatron-Core should own model-parallel semantics on NVIDIA large-scale runs.
- DeepSpeed should be used selectively for optimizer/offload capabilities, not as a second owner of model-parallel state.
- FSDP/DTensor should be the canonical portability layer when maintainability and cross-vendor operation matter most.
- Checkpoint interoperability requires logical tensor metadata, not framework-local assumptions.

### 14.3 Long-Context and MoE

- Exact factorization must be explicit:
  - dense: $$W = D \cdot P \cdot T \cdot C$$
  - dense + MoE orthogonal: $$W = D \cdot P \cdot C \cdot T \cdot E \cdot T_e$$
- $CP$ is the cleanest long-context memory lever.
- $EP$ is the most topology-sensitive MoE lever.
- Static state memory is exactly derivable; activation memory must be kernel-calibrated but structurally modeled.
- A run is production-ready only when topology placement, checkpoint semantics, numerical parity, and resume determinism are all validated together.

---

## 15. Primary References

- Megatron-Core: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed: https://www.deepspeed.ai/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- PyTorch DTensor / DeviceMesh: https://pytorch.org/docs/stable/distributed.tensor.html
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute
- AMD rocprof: https://rocm.docs.amd.com/
- FlashAttention: https://github.com/Dao-AILab/flash-attention

If required, the next deliverable can be a **cluster-by-cluster configuration matrix** with per-hardware recommended values for $D$, $T$, $P$, $C$, $E$, $T_e$, microbatch count, checkpointing depth, optimizer sharding stage, and expected failure signatures for dense, long-context dense, and MoE workloads.