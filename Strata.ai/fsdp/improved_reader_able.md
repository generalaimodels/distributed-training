# Technical Report: FSDP, ZeRO-3, and SimpleFSDP for Production-Scale LLM Training

## 1. Scope

This report explains the end-to-end systems design of fully sharded large-model training, centered on PyTorch FSDP/ZeRO-3 semantics and the compiler-driven SimpleFSDP approach described in the provided material. The report is written for researchers, distributed systems engineers, and training-infrastructure owners who need exact reasoning across:

- parameter, gradient, and optimizer-state sharding,
- collective communication behavior,
- communication/computation overlap,
- compiler-visible graph optimization,
- memory-fit analysis,
- topology-aware parallelism composition,
- data-pipeline interactions,
- numerical robustness,
- performance instrumentation,
- production resilience and recovery.

The primary technical objective is to make clear **why** FSDP-style training fits larger models, **how** its communication schedule behaves at runtime, and **which optimizations** materially improve throughput without changing training semantics.

---

## 2. Core Thesis

> **All-Reduce = Reduce-Scatter + All-Gather**

That identity is not a slogan; it is the core systems decomposition behind efficient sharded training.

In classic data parallelism, each rank holds a full model replica and synchronizes gradients with all-reduce. In FSDP/ZeRO-3, each rank instead holds only a shard of model parameters, gradients, and optimizer states. The full parameters are materialized only when needed, and gradients are synchronized as sharded results rather than replicated results.

This changes the training system along three critical axes:

- **Memory footprint**
  - persistent replicated model state is eliminated,
  - transient replicated weights exist only at wrapped-module execution boundaries.

- **Communication structure**
  - forward and backward require parameter **all-gather**,
  - backward gradient synchronization is **reduce-scatter** rather than full all-reduce.

- **Optimization opportunity**
  - because collectives are asynchronous, they can be:
    - bucketed,
    - reordered,
    - prefetched,
    - overlapped with compute,
    - exposed to the compiler for graph-level scheduling.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $N_{\theta}$ | total parameter elements in the model |
| $N_{\theta,l}$ | parameter elements in wrapped module or layer $l$ |
| $D$ | data-parallel or FSDP group size |
| $T$ | tensor-parallel degree |
| $P$ | pipeline-parallel stage count |
| $C$ | context-parallel degree |
| $E$ | expert-parallel degree |
| $B_{\mu}$ | per-replica microbatch size |
| $G_{acc}$ | gradient accumulation steps or number of microbatches per optimizer step |
| $S$ | sequence length |
| $b_{\text{param}}$ | bytes per parameter element |
| $b_{\text{grad}}$ | bytes per gradient element |
| $b_{\text{opt}}$ | bytes per optimizer-state element |
| $\alpha$ | collective startup latency |
| $\beta$ | per-byte transfer cost |
| $T_c$ | computation time in a scheduling window |
| $M_c$ | peak memory in a scheduling window |
| $M_{\max}$ | memory budget limit for auto-wrapping decisions |

---

## 4. Collective Communication Foundations

## 4.1 Semantic Roles of the Main Collectives

| Collective | Input distribution | Output distribution | FSDP role |
|---|---|---|---|
| All-gather | each rank holds a shard | every rank obtains the full tensor | materialize full weights for local compute |
| Reduce-scatter | each rank holds a full or partial contribution | reduced result is sharded across ranks | shard synchronized gradients |
| All-reduce | each rank holds a full contribution | every rank gets full reduced tensor | classic DP gradient sync |
| Broadcast | one rank has full tensor | all ranks receive tensor | initialization, control |
| Send/recv | point-to-point | point-to-point | pipeline activations and gradients |

The exact equivalence is:

$$
\text{AllReduce}(x) = \text{AllGather}(\text{ReduceScatter}(x))
$$

This matters because FSDP intentionally stops after the reduce-scatter during backward. It does **not** immediately rebuild full gradients everywhere. It keeps gradients sharded, which is exactly what a sharded optimizer wants.

---

## 4.2 Cost Models

For large messages, ring algorithms are usually bandwidth-optimal; for small messages, tree or hierarchical algorithms may reduce latency. The standard first-order communication model is:

$$
T_m(n) = \alpha + \beta n
$$

For a ring collective across $D$ ranks, the approximate costs are:

$$
T_{\text{AG}}^{\text{ring}} \approx (D-1)\alpha + \frac{D-1}{D} n \beta
$$

$$
T_{\text{RS}}^{\text{ring}} \approx (D-1)\alpha + \frac{D-1}{D} n \beta
$$

Thus:

$$
T_{\text{AR}}^{\text{ring}} \approx 2(D-1)\alpha + 2\frac{D-1}{D} n \beta
$$

This explains the performance value of **bucketing**:

- if buckets are too small, the $\alpha$ term dominates,
- if buckets are too large, memory rises and overlap opportunities shrink,
- the best bucket is the largest one whose latency can still be hidden by available compute and memory headroom.

---

## 4.3 Topology Implications

Collective performance is dominated by the slowest fabric traversed:

- **NVLink/NVSwitch** on NVIDIA systems:
  - best location for TP, CP, EP, and fine-grained layerwise collectives.
- **xGMI** on AMD systems:
  - same placement logic; keep latency-sensitive sharding groups inside xGMI-connected islands.
- **InfiniBand/RDMA or RoCE** across nodes:
  - acceptable for DP/FSDP and stage-level PP traffic,
  - much more dangerous for fine-grained TP or CP if sequence lengths are not large enough.

If GPUDirect RDMA is disabled or misconfigured, collectives may silently stage through host memory. That often manifests as:

- bandwidth collapse,
- high CPU utilization,
- widened collective tails,
- step-time variance,
- degraded overlap.

---

## 5. FSDP and ZeRO-3 Execution Semantics

## 5.1 State Sharding by ZeRO Stage

| Strategy | Parameters | Gradients | Optimizer states |
|---|---|---|---|
| DDP / ZeRO-0 | replicated | replicated | replicated |
| ZeRO-1 | replicated | replicated | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 / FSDP full shard | sharded | sharded | sharded |

The provided material correctly states that in FSDP or ZeRO-3, **all three major state classes are sharded**.

---

## 5.2 Runtime Workflow per Wrapped Module

The extracted workflow is:

- **Forward**
  - load shard from CPU if offloaded,
  - all-gather parameters,
  - local forward compute,
  - free full weights.

- **Backward**
  - all-gather parameters again,
  - local backward compute,
  - reduce-scatter gradients,
  - free full weights,
  - optionally offload gradients to CPU.

- **Optimizer step**
  - update local weight shard only.

The crucial point is the **second all-gather** in backward. It exists because the full gathered parameters were already released after forward to save memory.

### Why this is correct

Forward only needs the full parameters long enough to execute the local wrapped module. After that, retaining full weights would inflate peak HBM. Therefore FSDP frees them immediately. During backward, those parameters are needed again to compute correct local gradients, so they are re-gathered.

That is a direct trade:

- **more communication**
- in exchange for
- **substantially lower persistent and peak memory**

---

## 5.3 Pseudocode: FSDP Unit Execution

**Algorithm 1. FSDP full-shard runtime for one wrapped unit**

```text
Input:
  activation x
  local parameter shard W_shard
  local optimizer shard O_shard
  DP/FSDP group G
  flags: cpu_param_offload, cpu_grad_offload

Forward:
  1. If cpu_param_offload is enabled, DMA W_shard from host to device.
  2. Launch asynchronous all-gather over G to materialize W_full.
  3. Wait for W_full readiness.
  4. Execute local forward computation y = f(x, W_full).
  5. Free W_full and retain only W_shard.
  6. Store only required activations/checkpoint metadata.

Backward:
  7. Launch asynchronous all-gather over G to rematerialize W_full.
  8. Wait for W_full readiness.
  9. Execute local backward on the wrapped unit.
 10. Produce local gradient contributions for W_full.
 11. Launch asynchronous reduce-scatter over G to obtain grad_shard.
 12. Free W_full.
 13. If cpu_grad_offload is enabled, DMA grad_shard to host.

Optimizer:
 14. Update O_shard and W_shard locally using grad_shard.
 15. If cpu_param_offload is enabled, optionally evict W_shard back to host.
Output:
  updated local shard W_shard
```

---

## 5.4 Memory Model

The persistent per-rank memory for full-shard training is approximately:

$$
M_{\text{persistent}} \approx \frac{N_{\theta}}{D}\left(b_{\text{param}} + b_{\text{grad}} + b_{\text{opt}}\right)
$$

The transient per-rank peak is dominated by the largest wrapped unit that must be materialized for compute:

$$
M_{\text{transient}} \approx \max_l \left(N_{\theta,l} b_{\text{param}} + M_{\text{act},l} + M_{\text{workspace},l}\right)
$$

Overall peak memory is:

$$
M_{\text{peak}} \approx M_{\text{persistent}} + M_{\text{transient}} + M_{\text{fragmentation}}
$$

This is why wrapping granularity matters:

- too coarse:
  - fewer collectives,
  - lower latency overhead,
  - but higher transient HBM.

- too fine:
  - smaller transient memory,
  - but too many collectives,
  - poor bandwidth utilization,
  - exposed communication.

---

## 6. Why FSDP Uses All-Gather and Reduce-Scatter Instead of All-Reduce

Per step, a rank in FSDP communicates roughly:

- one all-gather for forward,
- one all-gather for backward recomputation,
- one reduce-scatter for gradients.

Ignoring minor metadata and assuming dense sharded parameters, the communication volume per rank is approximately:

$$
V_{\text{step}} \approx \frac{D-1}{D} \left(2N_{\theta} b_{\text{param}} + N_{\theta} b_{\text{grad}}\right)
$$

An all-reduce of full gradients would replicate the full reduced gradient back to every rank, which is unnecessary for sharded optimizers. Reduce-scatter avoids that replication and directly produces the correct local gradient shard.

This is the central efficiency mechanism behind FSDP/ZeRO-3.

---

## 7. SimpleFSDP Architecture

## 7.1 Conceptual Design

The provided material describes a compiler-integrated approach with the following ingredients:

- parameters are initialized as **DTensor shards**,
- replication for compute is expressed through **differentiable redistribution**,
- the redistribution is wrapped using **parametrization**,
- release/re-gather semantics are modeled using **selective activation checkpointing**,
- the entire communication-plus-computation graph is exposed to **TorchInductor** through **torch.compile**.

This is a significant architectural simplification relative to hand-managed eager-mode communication logic.

> “The implementation benefits distributed training from two aspects: (i) Simplicity ... (ii) Debuggability ...”

That claim is technically credible because:

- communication becomes part of the compiler-visible graph,
- scheduling can be optimized globally,
- eager-mode user code remains structurally close to the original model code,
- debugging does not require rewriting the model into a bespoke distributed runtime.

---

## 7.2 Why DTensor and Parametrization Matter

Treating the sharded parameter as the canonical stored object and the replicated parameter as a derived runtime view yields two advantages:

1. **Differentiable sharding transitions**
   - forward redistribution materializes replicated weights,
   - backward naturally induces the corresponding gradient reduction/sharding behavior.

2. **Compiler visibility**
   - the all-gather, wait, compute, reduce-scatter, and free operations exist as explicit graph/IR nodes rather than opaque Python-side side effects.

That visibility is what enables later **bucketing** and **reordering**.

---

## 7.3 Activation Checkpointing Interpretation

The report material makes an important observation: the gathered full parameter can be treated like an activation.

That is correct from a systems perspective.

- It is **produced** before compute.
- It is **consumed** by compute.
- It can be **released** after compute.
- It can be **recomputed** before backward.

Therefore, FSDP parameter rematerialization is structurally analogous to activation checkpointing.

This is a clean formal model because it allows the system to:

- localize recomputation to FSDP-related communication,
- reduce memory without redefining gradient semantics,
- compose naturally with user-defined activation checkpoint policies.

---

## 7.4 Full-Graph vs Partial-Graph Compilation

Compiler-driven scheduling works best when the communication and computation appear in one graph.

- **Full graph**
  - maximal scheduling and reordering freedom,
  - best overlap potential.

- **Split graph**
  - necessary when data-dependent control flow or non-traceable content exists,
  - but limits cross-boundary optimization.

In practice, if performance is critical, the model should be refactored toward:

- static control flow,
- stable tensor shapes where possible,
- traceable wrapper boundaries,
- deterministic module ordering.

---

## 8. SimpleFSDP Optimizations

The material identifies two core optimizations:

1. **Bucketing**
2. **Reordering**

Both are necessary. Bucketing reduces per-collective overhead; reordering hides remaining collective latency behind compute.

---

## 8.1 Terminology from the Extracted Schedules

| Symbol | Meaning |
|---|---|
| AG | asynchronous all-gather launch |
| Wa | all-gather wait plus copy-out/readiness handling |
| C | local computation |
| RS | asynchronous reduce-scatter launch |
| Wr | reduce-scatter wait plus gradient readout |

The distinction between launch and wait is crucial. Without it, overlap reasoning becomes incorrect.

---

## 8.2 Bucketing

## 8.2.1 Motivation

If the compiler lowers each per-parameter communication event independently, the schedule looks like:

- AG1, Wa1, C1,
- AG2, Wa2, C2,
- AG3, Wa3, C3,
- ...

This is pathological at scale because:

- many small collectives incur many startup costs,
- the network never reaches efficient large-message bandwidth,
- communication remains highly exposed.

The communication-time model justifying bucketing is:

$$
T_m(n) = \alpha + \beta n
$$

Merging two messages of sizes $n_1$ and $n_2$ replaces:

$$
(\alpha + \beta n_1) + (\alpha + \beta n_2)
$$

with:

$$
\alpha + \beta(n_1 + n_2)
$$

which saves one startup cost $\alpha$.

---

## 8.2.2 All-Gather Bucketing Mechanics

For all-gather bucketing:

- flatten multiple tensors into a larger buffer,
- launch one collective on the larger buffer,
- wait once,
- copy/slice the gathered buffer into the original tensor views.

Operationally:

- AG1 and AG2 become AG12,
- Wa1 and Wa2 become Wa12,
- the larger buffer is gathered once,
- the copy-out stage reconstructs the original parameter boundaries.

This reduces launch overhead but introduces:

- temporary bucket buffer memory,
- copy-out memory traffic,
- dependency management complexity.

---

## 8.2.3 Reduce-Scatter Bucketing Mechanics

For reduce-scatter bucketing:

- each original gradient is logically chunked by world size,
- corresponding chunks from multiple gradients are concatenated into one larger buffer,
- one reduce-scatter is issued,
- the returned local shard is read out into the original gradient-shard destinations.

This is correct because reduce-scatter is linear over concatenation as long as partitioning is consistent.

---

## 8.2.4 Practical Bucket Constraints

Bucketing must be constrained by:

- available HBM,
- current compute overlap window,
- launch overhead,
- interconnect bandwidth,
- copy-out cost,
- stream concurrency limits.

A bucket that is too large can be worse than multiple smaller buckets if it:

- exceeds overlapable compute time,
- pushes peak memory beyond budget,
- delays readiness of the first parameter needed by the next compute region.

---

## 8.3 Reordering

## 8.3.1 Objective

Collectives are asynchronous and typically execute on a dedicated communication stream. If their launch is delayed until immediately before consumption, the entire collective latency becomes exposed.

Reordering moves future collectives earlier while preserving dependency correctness.

---

## 8.3.2 Forward Reordering

The material gives the manual-wrapping example:

- bucket AG1 and AG2 into AG12,
- bucket AG3 and AG4 into AG34,
- then move AG34 earlier so it overlaps with the current computation.

This works because:

- AG34 does not depend on the completion of compute for C1 and C2,
- only its **wait** must be placed before the corresponding C3/C4 consumption.

Therefore, the scheduler can:

- launch AG34 early,
- let AG34 progress while C1/C2 execute,
- reduce the exposed latency before C3/C4.

---

## 8.3.3 Backward Reordering

Backward is more complex because both directions are active:

- parameter all-gathers for rematerialization,
- gradient reduce-scatters for synchronization.

The reported scheduling principle is:

- pull future all-gathers forward when safe,
- place reduce-scatter launches as early as possible after their local gradient production,
- delay waits until the result is actually needed,
- overlap earlier reduce-scatter with later computation.

The backward case is more sensitive because two communication classes compete for overlap windows:

- the current step’s rematerialization all-gather,
- the previous step’s gradient reduce-scatter.

A correct scheduler must preserve:

- gradient dependency,
- parameter dependency,
- memory bounds,
- stream-order/event-order correctness.

---

## 8.3.4 Hidden Cost Often Missed: Copy-Out Work

The provided material correctly notes that after a bucketed all-gather or reduce-scatter completes, there is still local copy/readout work.

That copy-out is not negligible when:

- buckets are large,
- parameters are fragmented,
- dtype conversions occur,
- memory bandwidth is already saturated.

An expert implementation overlaps not only the collective itself, but also:

- post-AG slice/copy into parameter views,
- post-RS slice/readout into local gradient shards.

This is one reason compiler-visible scheduling is materially better than naive eager sequencing.

---

## 9. Model Wrapping Strategies

## 9.1 Manual Wrapping

Manual wrapping groups communication nodes according to pre-defined module lists.

This is conceptually similar to classic FSDP wrapping policies:

- wrap each transformer block,
- or wrap attention and MLP separately,
- or wrap a larger stage-specific submodule.

### When manual wrapping is best

- architecture is stable,
- module boundaries are semantically meaningful,
- researchers want predictable memory and performance,
- debugging and checkpoint compatibility matter more than maximizing every last percent of overlap.

### When manual wrapping is suboptimal

- module sizes are uneven,
- compute-to-communication ratios vary across depth,
- irregular layers exist,
- nonuniform sequence-length effects alter overlap windows,
- the best bucket boundaries do not match module boundaries.

---

## 9.2 Auto Wrapping

The extracted material shows that auto wrapping can choose boundaries that manual intuition would not. In the example, it may bucket layers $2$ and $3$ together rather than $1$ and $2$.

That is a strong design choice because the true optimization target is not semantic neatness; it is:

- minimal exposed communication,
- within a memory limit.

This is the correct systems objective.

---

## 9.3 Profiling Inputs for Auto Wrapping

The auto-wrap algorithm profiles or estimates:

- compute time $T_c$,
- compute peak memory $M_c$,
- communication time for candidate AG buckets,
- communication time for corresponding RS buckets.

Communication is estimated as:

$$
T_m = \alpha + \beta n
$$

Computation time and peak memory are obtained from profiled kernels.

The algorithm is greedy, which is appropriate here because:

- the schedule space is large,
- compile-time must remain practical,
- local exposure reduction is often sufficient,
- a fully global optimum is not usually worth the added complexity.

---

## 9.4 Pseudocode: Auto-Wrapping Decision Rule

**Algorithm 2. Greedy bucket decision for SimpleFSDP**

```text
Input:
  current AG bucket time T_AG_bucket
  previous RS bucket time T_RS_prev
  current compute time T_c
  current peak memory M_c
  candidate AG time T_AG_i
  candidate RS time T_RS_i
  next-step compute memory M_c_i
  memory limit M_max
  mode in {forward, backward}

Decision:
  If mode = forward:
      bucket only if:
        1. new AG bucket time can be hidden by current compute
        2. next-step prefetched memory stays within M_max

      Condition:
        T_AG_bucket+candidate <= T_c
        M_c + M_c_i <= M_max

  If mode = backward:
      bucket only if:
        1. previous RS exposure plus new AG bucket can be hidden by current compute
        2. next-step prefetched memory stays within M_max

      Condition:
        T_RS_prev + T_AG_bucket+candidate <= T_c
        M_c + M_c_i <= M_max

  If all required conditions hold:
      return Bucket
  Else:
      return DoNotBucket
```

### Interpretation

Forward requires only hiding the prefetched all-gather. Backward requires hiding both:

- previously launched reduce-scatter work,
- and newly prefetched all-gather work.

That asymmetry is technically correct and important.

---

## 10. Why SimpleFSDP Improves Real Throughput

SimpleFSDP is not merely “simpler FSDP.” Its performance value comes from compiler-visible scheduling.

### It reduces communication exposure through four mechanisms

- merges small collectives into larger efficient buckets,
- launches future collectives earlier,
- overlaps collective latency with current compute,
- overlaps post-collective copy-out with compute.

### It improves maintainability

- no large custom eager-mode communication stack is required,
- wrapping remains lightweight,
- optimization logic lives in the compiler backend,
- users can still reason about original module structure.

### It improves debuggability

- the distributed semantics are explicit,
- module lineage in IR metadata supports analysis,
- compile-time schedule decisions can be inspected and profiled.

---

## 11. Composition with Other Parallelism Dimensions

FSDP alone is rarely the final architecture for frontier LLM training. It must compose with:

- tensor parallelism,
- pipeline parallelism,
- sequence/context parallelism,
- expert parallelism,
- activation checkpointing,
- mixed precision,
- sometimes offload.

---

## 11.1 Placement Principle

**Latency-sensitive, every-layer communication should remain inside the fastest topology domain.**

Therefore:

- place $T$, $C$, and often $E$ **within node** or within the highest-bandwidth island,
- place $P$ across nodes if stage boundaries are balanced,
- place $D$ or FSDP across the widest remaining topology domain.

This rule is valid on:

- A100/H100 HGX with NVSwitch,
- B200-class NVLink/NVSwitch fabrics,
- MI300X/MI350-class xGMI-connected systems.

---

## 11.2 Hardware-Oriented Guidance

| Platform class | Fastest local fabric | Recommended use of that fabric |
|---|---|---|
| A100 HGX | NVLink/NVSwitch | TP, SP, CP, EP inside node; FSDP/DP across nodes |
| H100 HGX | NVLink/NVSwitch with stronger FP8 capability | same as A100, but larger intra-node TP/CP is more viable |
| B200-class systems | next-generation NVLink domain | larger TP/CP/EP groups become practical; aggressive low precision is viable |
| MI300X | xGMI with large HBM capacity | keep TP/EP/CP inside xGMI island; RCCL tuning is mandatory |
| MI350-class | newer xGMI-class topology and stronger low-precision throughput | same placement logic; validate kernel portability and RCCL path selection |

---

## 11.3 Tensor Parallel Composition

The material notes that a parameter can be doubly sharded across:

- the DP/FSDP dimension,
- the TP dimension.

At runtime:

- it is all-gathered over the DP/FSDP submesh,
- while remaining appropriately sharded over the TP submesh.

This is the correct composition.

### Key consequence

With TP enabled, the transient local parameter materialization may be only the TP-local slice rather than the full global tensor. That lowers HBM pressure and can improve FSDP viability at larger hidden sizes.

---

## 11.4 Pipeline Parallel Composition

FSDP wraps submodules assigned to each pipeline stage.

This is compatible because:

- PP owns model partitioning across stages,
- FSDP owns intra-stage state sharding across data replicas.

The main engineering requirement is stage balance.

Pipeline efficiency for non-interleaved 1F1B is approximately:

$$
\eta_{pp} \approx \frac{m}{m + P - 1}
$$

where $m$ is the number of microbatches in flight per step.

If $m$ is too small:

- bubble dominates,
- overlap opportunities shrink,
- FSDP communication exposure becomes more visible.

---

## 11.5 Global Batch Derivation

Ignoring curriculum and variable-length packing complications, the effective global batch is:

$$
B_{\text{global}} = B_{\mu} \times G_{acc} \times D
$$

Tokens processed per optimizer step are:

$$
N_{\text{tokens/step}} = B_{\text{global}} \times S
$$

Tokens per second are:

$$
\text{TPS} = \frac{B_{\text{global}} \times S}{T_{\text{step}}}
$$

These are the correct system-level metrics for comparing wrapping policies, bucket sizes, and precision modes.

---

## 11.6 Sequence Parallel and Context Parallel

### Sequence Parallel

Sequence parallelism reduces activation replication across tensor-parallel ranks. It is often necessary when TP is increased, because otherwise activation memory can cancel the parameter-memory savings from FSDP.

### Context Parallel

For long context, context parallelism partitions the sequence dimension itself. That is essential when $S$ becomes the dominant memory term.

Engineering rule:

- keep context-parallel groups inside the fastest local domain whenever possible,
- because attention-layer communication is frequent and latency-sensitive.

For extreme long-context training:

- Ulysses-style sequence partitioning,
- Ring Attention variants,
- selective rematerialization,
- FSDP on parameters,
- TP/SP on model dimensions

must be jointly balanced. FSDP solves parameter memory; it does **not** solve long-context activation explosion by itself.

---

## 11.7 MoE Composition

For Mixture-of-Experts training:

- use expert parallelism to shard experts,
- optionally use expert tensor parallelism for oversized experts,
- use FSDP/ZeRO primarily across the data-replica dimension, not blindly across every expert grouping.

### Critical constraints

- router behavior must remain deterministic across ranks,
- token dropping and capacity limits must be globally consistent,
- optimizer-state ownership for experts must match the physical shard layout,
- expert all-to-all traffic should remain inside the fastest domain if possible.

Poor EP/FSDP composition can cause:

- excessive rematerialization traffic,
- expert load imbalance,
- optimizer ownership bugs,
- unstable throughput.

---

## 12. Memory Control as a First-Class Variable

## 12.1 Why a Model Fits or Fails

A model fits if and only if:

$$
M_{\text{persistent}} + M_{\text{transient}} + M_{\text{fragmentation}} \leq M_{\text{HBM}}
$$

Where the transient term includes:

- gathered parameter buckets,
- activations,
- temporary workspaces,
- communication buffers,
- kernel scratch,
- possible fused-kernel staging.

### Common failure pattern

A job may appear to “fit by parameter count” but still OOM because:

- auto-wrap prefetched too many future parameters,
- activation checkpointing boundaries are too coarse,
- fragmentation grew after repeated allocations,
- fused attention workspace spiked at larger sequence length,
- FP8 or mixed-precision conversion added temporary buffers,
- optimizer shards plus gathered params overlapped badly.

---

## 12.2 Memory Levers

| Lever | Saves | Cost |
|---|---|---|
| FSDP / ZeRO-3 | parameter, gradient, optimizer persistence | more collectives |
| Activation checkpointing | activations | recompute FLOPs |
| Sequence/context parallel | activations | more layerwise communication |
| TP | per-rank parameter slice | more frequent collectives |
| CPU offload | HBM | PCIe/CXL latency and bandwidth penalty |
| NVMe offload | HBM | large latency; usually last resort |
| Smaller buckets | transient HBM | more latency overhead |
| Larger buckets | startup latency | more HBM, less overlap |
| Fused kernels | workspace and bandwidth | portability/debug complexity |

---

## 12.3 Fragmentation Control

Fragmentation is frequently misdiagnosed as random OOM.

Mitigations:

- stabilize bucket sizes and allocation patterns,
- preallocate communication buffers,
- use static shapes when possible,
- reduce allocator churn from variable packing,
- avoid mixing many temporary dtypes,
- separate long-lived and short-lived buffers.

On multi-tenant or elastic systems, allocator behavior should be part of regression tracking.

---

## 13. Data Pipeline as a Distributed System

The training system is only as fast and as statistically sound as the data path feeding it. Sharded training amplifies every data bug.

---

## 13.1 Required Pipeline Stages

- ingestion and normalization,
- exact deduplication,
- near-duplicate detection,
- language/domain/quality filtering,
- tokenizer training or selection,
- tokenization,
- sequence packing,
- deterministic sharding,
- streaming or memory-mapped serving,
- asynchronous prefetch,
- resume-safe sampling state,
- data lineage tracking.

---

## 13.2 Why Data Design Changes Convergence and Throughput

| Design choice | Throughput effect | Convergence/quality effect |
|---|---|---|
| exact and near dedup | smaller dataset, less wasted compute | reduces memorization, improves effective data entropy |
| tokenizer choice | changes average tokens per document | changes compression ratio, morphology handling, and loss scaling |
| sequence packing | raises token utilization, less padding | requires correct masks or it corrupts supervision |
| curriculum or mixture weights | can stabilize early training | changes gradient noise and domain retention |
| sample-length balancing | reduces dataloader skew | changes effective domain exposure if not tracked |
| deterministic sharding | stable restart behavior | necessary for reproducible training statistics |

### Critical implementation point

Packing improves throughput only if attention masks, position handling, and end-of-document loss masking are exact. Incorrect packing silently changes the learning problem.

---

## 13.3 Deterministic Resume Requirements

A production dataloader must checkpoint:

- global sample or token cursor,
- shuffle seed state,
- shard assignment state,
- intra-shard offset,
- packing buffer state,
- mixture scheduler state.

Without these, resuming a large FSDP job changes the data order and can invalidate convergence comparisons.

---

## 13.4 Pseudocode: Deterministic Distributed Sampling

**Algorithm 3. Resume-safe sharded token stream**

```text
Input:
  dataset shard manifest
  global seed
  current epoch
  world size W
  rank r
  saved cursor state

Initialization:
  1. Build deterministic shard order from manifest and seed.
  2. Build deterministic sample order inside each shard.
  3. Restore global token cursor and packing-buffer state if resuming.

Iteration:
  4. Assign each sample or token block to rank r by deterministic partition rule.
  5. Accumulate samples into a packing buffer until target sequence length is met.
  6. Emit packed sequences with exact document-boundary masks.
  7. Advance cursor and persist cursor periodically.

Guarantee:
  Given the same manifest, seed, and saved cursor, every resume reproduces the same token stream.
```

---

## 14. Kernel and Numeric Optimization

Once FSDP communication is mostly hidden, kernel efficiency becomes the next bottleneck.

---

## 14.1 Primary Kernel Targets

For training, the highest-value kernels are usually:

- FlashAttention,
- fused MLP,
- fused RMSNorm,
- fused softmax,
- fused RoPE application,
- fused optimizer updates,
- persistent kernels where supported,
- graph capture for stable inner loops.

### Important distinction

PagedAttention is primarily an inference-serving optimization. It is not usually the first-order pretraining kernel priority. For training, FlashAttention-class kernels dominate.

---

## 14.2 Precision Strategy

### BF16
- preferred on A100, H100, MI300X-class systems for most LLM training,
- avoids many FP16 loss-scaling pathologies.

### FP16
- still usable,
- requires stronger overflow handling and often dynamic loss scaling.

### FP8 and emerging lower precisions
- highly attractive on H100/B200-class and future AMD generations,
- must keep reductions and critical accumulations in safer precision,
- require exact parity validation against a trusted BF16 baseline.

### Optimizer precision partitioning
Typical robust choice:

- compute in BF16 or FP8 where supported,
- gradient reductions in BF16 or FP32 depending stability target,
- optimizer moments in FP32,
- optional master weights in FP32 where required by optimizer path.

---

## 14.3 Numerical Robustness Checklist

- global gradient norm clipping across all logical shards,
- overflow and underflow checks,
- deterministic random seeds,
- fixed bucketization during parity runs,
- stable softmax kernels,
- identical masking semantics across fused and unfused paths,
- cross-vendor parity checks on CUDA and ROCm,
- single-node and multi-node curve comparison.

For sharded gradients, global norm is:

$$
\|\nabla\|_2 = \sqrt{\sum_{r}\sum_{i \in \text{shard }r} g_i^2}
$$

That norm must be aggregated across every rank that contributes to the logical parameter, not just the local FSDP shard.

---

## 15. Performance Engineering and Scaling Science

## 15.1 Step-Time Decomposition

A serious training report always decomposes:

$$
T_{\text{step}} =
T_{\text{data}}
+ T_{\text{forward}}
+ T_{\text{backward}}
+ T_{\text{optimizer}}
+ T_{\text{checkpoint}}
+ T_{\text{idle}}
$$

Where forward/backward should be further decomposed into:

- exposed all-gather time,
- hidden all-gather time,
- exposed reduce-scatter time,
- hidden reduce-scatter time,
- compute time,
- copy-out or readout time.

This is the only correct way to prove that bucketing and reordering helped.

---

## 15.2 MFU and HFU

Model FLOP Utilization is:

$$
\text{MFU} = \frac{\text{useful model FLOPs per second}}{\text{peak hardware FLOPs per second}}
$$

Hardware FLOP Utilization is:

$$
\text{HFU} = \frac{\text{executed FLOPs per second}}{\text{peak hardware FLOPs per second}}
$$

Because rematerialization adds FLOPs, HFU can rise while MFU falls. Any report that only shows one of these can be misleading.

---

## 15.3 Strong and Weak Scaling

### Strong scaling
- fixed global batch and sequence length,
- increase GPU count,
- measure step time and efficiency degradation.

### Weak scaling
- fixed work per GPU,
- increase GPU count proportionally with global batch or token rate,
- measure stability of throughput per GPU.

For FSDP/SimpleFSDP, strong scaling typically reveals:

- collective latency becoming dominant,
- bucket sizing sensitivity,
- topology mismatch,
- overlap quality limits.

---

## 15.4 Required Tooling

Use evidence, not intuition.

### NVIDIA stack
- Nsight Systems for end-to-end timelines and stream overlap,
- Nsight Compute for occupancy, register pressure, memory traffic,
- NCCL traces and nccl-tests,
- NIC counters and IB telemetry.

### AMD stack
- rocprof,
- RCCL traces and rccl-tests,
- xGMI and NIC counters,
- HIP kernel profiling.

### Framework-level
- PyTorch Profiler,
- compiler IR inspection,
- allocator snapshots,
- checkpoint I/O traces.

---

## 16. Failure Analysis

## 16.1 Diagnostic Table

| Symptom | Likely root cause | Evidence to inspect | Primary remediation |
|---|---|---|---|
| all-gather bandwidth collapse | host staging, bad ring choice, topology misdetection | NCCL/RCCL logs, NIC counters, topology dump | fix GPUDirect/RDMA path, pin rings to topology |
| long collective tail on one rank | straggler GPU, NUMA mismatch, dataloader skew | per-rank step histograms, CPU affinity, stream traces | fix binding, isolate slow rank, equalize input load |
| deadlock | mismatched collective order or rank divergence | last successful collective trace, per-rank stack states | enforce identical wrapping and control flow |
| unexpected OOM after wrap change | prefetched bucket too large, fragmentation, workspace growth | allocator timeline, peak-memory snapshots | shrink bucket, change wrap granularity, preallocate buffers |
| poor overlap despite async collectives | waits placed too early, compute window too small | stream timeline with launch/wait markers | reorder launches, adjust microbatching or wrap size |
| loss divergence after optimization | reduction precision changed, bad masking, nondeterministic kernels | loss curve parity, gradient norm traces | restore safer reduce dtype, validate masks and seeds |
| dataloader stalls | storage jitter, worker starvation, small prefetch depth | CPU profile, storage latency, queue depths | raise prefetch, cache locally, rebalance shard sizes |
| MoE throughput collapse | expert imbalance, token overflow, all-to-all cross slow links | token-per-expert stats, dropped-token ratio | rebalance routing, keep EP local, adjust capacity factor |

---

## 16.2 Deadlock Reasoning from First Principles

Never blame the model first.

A distributed deadlock almost always comes from one of these:

- ranks entered collectives in different orders,
- some ranks skipped a wrapped region due to conditional logic,
- auto-wrap or compile path diverged across ranks,
- one rank OOMed or faulted before issuing the next collective,
- communicator initialization does not match physical topology,
- mixed-version binaries changed collective behavior.

The correct workflow is:

1. inspect per-rank last collective issued,
2. confirm identical module wrapping and mesh shape,
3. verify deterministic graph capture or compile output,
4. check rank-local exceptions before the hang,
5. inspect transport path and communicator membership.

---

## 17. Production Automation and Resilience

## 17.1 Preflight Must Be Automated

Before launching a multi-node FSDP job, automation should verify:

- driver and runtime version compatibility,
- CUDA/ROCm and compiler consistency,
- NCCL/RCCL availability,
- GPUDirect/RDMA enablement,
- NIC and GPU health,
- topology discovery,
- local and cross-node bandwidth tests,
- clock synchronization,
- storage and checkpoint path health,
- deterministic environment export.

---

## 17.2 Pseudocode: Cluster Bring-Up for FSDP Training

**Algorithm 4. Production preflight and launch synthesis**

```text
Input:
  cluster inventory
  model configuration
  target parallelism search space
  checkpoint path
  data manifest

Preflight:
  1. Discover GPU, NIC, CPU NUMA, and fabric topology on every node.
  2. Run local and cross-node bandwidth microbenchmarks.
  3. Validate container, driver, compiler, NCCL/RCCL, and kernel support.
  4. Reject unhealthy ranks, links, or storage paths.

Planning:
  5. Enumerate candidate parallel tuples (D, T, P, C, E).
  6. Filter tuples that violate divisibility, topology locality, or HBM budget.
  7. Estimate step time and peak memory for each tuple.
  8. Select the best feasible tuple.

Launch:
  9. Materialize deterministic rank mapping and process placement.
 10. Export the exact runtime environment and topology metadata.
 11. Validate checkpoint compatibility or produce a reshard plan.
 12. Launch with retry and health monitoring hooks enabled.

Runtime:
 13. Continuously collect step time, communication bandwidth, memory, and loss.
 14. On failure, checkpoint integrity is validated before automatic resume.
```

---

## 17.3 Checkpointing Requirements

A production-grade sharded checkpoint must store:

- logical parameter names,
- shapes and dtypes,
- sharding metadata,
- optimizer-state shard ownership,
- mesh topology and parallel degrees,
- RNG states,
- scheduler state,
- dataloader cursor state,
- tokenizer version and dataset lineage hashes.

### Why this matters

Without logical-to-physical metadata, resharding across different world sizes or different parallel tuples becomes fragile.

Checkpoint interoperability across:

- FSDP,
- ZeRO,
- Megatron-style TP/PP,
- vendor changes,
- resumed runs with different world sizes

depends on a canonical logical representation.

---

## 18. Framework Selection Guidance

## 18.1 Decision Matrix

| Requirement | Best-fit abstraction |
|---|---|
| minimal model-code changes, compiler-visible FSDP optimization | SimpleFSDP / PyTorch DTensor path |
| maximum maturity for TP + PP + CP + EP at very large scale | Megatron-Core |
| ZeRO-centric optimizer sharding in existing DeepSpeed deployments | DeepSpeed ZeRO-2/3 |
| highest portability across CUDA and ROCm with manageable complexity | PyTorch FSDP/DTensor plus selective custom kernels |
| specialized kernels and nonstandard scheduling | custom runtime layered under framework APIs |

---

## 18.2 Practical Recommendation

Use **SimpleFSDP** when the priorities are:

- low intrusion into model code,
- compiler scheduling of communication,
- strong debuggability,
- direct integration with PyTorch graph compilation.

Use **Megatron-Core or DeepSpeed hybrid parallelism** when the priorities are:

- aggressive TP/PP/CP/EP scaling,
- established multi-dimensional parallel schedules,
- custom fused training kernels at very large scale.

In many production systems, the correct architecture is not one or the other, but:

- FSDP or ZeRO semantics for state sharding,
- TP/PP/CP/EP for model decomposition,
- compiler-visible kernels where possible,
- topology-aware placement and strict checkpoint discipline.

---

## 19. Key Engineering Conclusions

1. **FSDP/ZeRO-3 changes the optimization target from replicated-state efficiency to sharded-state scheduling efficiency.**
   - Memory is saved by sharding persistence.
   - Performance is recovered by hiding the extra all-gathers and reduce-scatters.

2. **All-gather and reduce-scatter are the right primitives.**
   - They directly match the needs of sharded parameter materialization and sharded optimizer updates.

3. **SimpleFSDP is valuable because it exposes communication to the compiler.**
   - That makes bucketing and reordering possible at IR level.

4. **Bucketing alone is insufficient.**
   - It reduces startup overhead but does not guarantee hidden latency.
   - Reordering is what converts asynchronous collectives into real overlap.

5. **Auto wrapping is the correct direction for performance-critical deployments.**
   - The optimal bucket boundary is not always the module boundary.
   - The right criterion is hidden latency under a memory cap.

6. **Topology determines parallelism placement.**
   - Put $T$, $C$, and $E$ on the fastest local fabric.
   - Use $D$/FSDP across the widest domain.

7. **Memory-fit analysis must include transient gathered weights and fragmentation.**
   - Persistent shard math alone is insufficient.

8. **Data, numerics, and automation are part of the same training system.**
   - A perfectly optimized FSDP runtime still fails if the token stream is nondeterministic, the masks are wrong, or the checkpoint is not reshard-safe.

---

## 20. Relevant Links

### Core distributed training and FSDP
- PyTorch FSDP documentation: https://pytorch.org/docs/stable/fsdp.html
- PyTorch DTensor documentation: https://pytorch.org/docs/stable/distributed.tensor.html
- PyTorch distributed overview: https://pytorch.org/docs/stable/distributed.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Megatron-Core: https://github.com/NVIDIA/Megatron-LM

### Communication libraries
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/
- nccl-tests: https://github.com/NVIDIA/nccl-tests
- rccl-tests: https://github.com/ROCm/rccl-tests

### Profiling and performance analysis
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- ROCm rocprof: https://rocm.docs.amd.com/

### Kernel optimization
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Triton: https://github.com/triton-lang/triton

If required, the next report can extend this into one of three directions:

- **A.** a hardware-specific deployment blueprint for A100/H100/B200/MI300X/MI350 clusters,
- **B.** a Megatron-Core + DeepSpeed + FSDP interoperability report,
- **C.** a long-context and MoE training architecture report with exact parallel-group factorization and memory formulas.