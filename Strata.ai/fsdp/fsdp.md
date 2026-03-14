

# SimpleFSDP: Compiler-Optimized Fully Sharded Data Parallelism with Communication–Computation Overlap

## A Production-Grade Technical Report on FSDP Internals, Collective Decomposition, and TorchInductor-Level Scheduling for Large-Scale Distributed Training

---

## Table of Contents

1. [FSDP: First-Principles Architecture and Execution Model](#1-fsdp-first-principles-architecture-and-execution-model)
2. [Collective Communication Decomposition](#2-collective-communication-decomposition)
3. [SimpleFSDP: Compiler-Native FSDP via DTensor and TorchInductor](#3-simplefsdp-compiler-native-fsdp-via-dtensor-and-torchinductor)
4. [Bucketing: Amortizing Base Communication Latency](#4-bucketing-amortizing-base-communication-latency)
5. [Reordering: Prefetching Communication for Overlap](#5-reordering-prefetching-communication-for-overlap)
6. [Model Wrapping Strategies: Manual vs. Auto](#6-model-wrapping-strategies-manual-vs-auto)
7. [Composability with Multi-Dimensional Parallelism](#7-composability-with-multi-dimensional-parallelism)
8. [Memory Analysis and Constraint Modeling](#8-memory-analysis-and-constraint-modeling)
9. [Production Deployment Considerations](#9-production-deployment-considerations)
10. [Cross-Vendor Portability: NVIDIA and AMD Clusters](#10-cross-vendor-portability-nvidia-and-amd-clusters)

---

## 1. FSDP: First-Principles Architecture and Execution Model

### 1.1 Foundational Concept

Fully Sharded Data Parallelism (FSDP), also known as ZeRO Stage-3, eliminates the memory redundancy inherent in standard Data Parallelism (DP) by **sharding all three categories of training state**—model parameters, gradients, and optimizer states—across all participating ranks. Each rank stores only a $\frac{1}{N}$ slice of the full model state, where $N$ is the data-parallel world size.

The per-rank memory footprint under FSDP for a model with $\Phi$ parameters in mixed-precision training (BF16 parameters, FP32 optimizer) is:

$$
M_{\text{FSDP}}^{\text{rank}} = \frac{1}{N}\bigl(2\Phi + 2\Phi + (4\Phi + 4\Phi + 4\Phi)\bigr) = \frac{16\Phi}{N} \text{ bytes}
$$

where the terms correspond to:
- $2\Phi$: BF16 model parameters (sharded)
- $2\Phi$: BF16 gradients (sharded)
- $4\Phi + 4\Phi + 4\Phi$: FP32 master weights, first moment, and second moment of Adam (all sharded)

In contrast, standard DP replicates the full model and optimizer on every rank:

$$
M_{\text{DP}}^{\text{rank}} = 2\Phi + 2\Phi + 12\Phi = 16\Phi \text{ bytes}
$$

FSDP therefore achieves a **linear memory reduction** of $N\times$, at the cost of introducing all-gather and reduce-scatter collectives on the critical path.

### 1.2 Per-FSDP-Unit Execution Lifecycle

Each FSDP unit wraps $L$ contiguous layers and executes the following pipeline per training step. The lifecycle is strictly sequenced per unit and pipelined across units.

#### 1.2.1 Forward Pass Execution (Per FSDP Unit)

| Step | Operation | Communication | Memory Effect |
|------|-----------|---------------|---------------|
| 1 | **LOAD-MODEL-SHARD** | None (local I/O or HBM read) | Load local shard into working buffer; if CPU-offloaded, initiate H2D transfer |
| 2 | **ALL-GATHER** | All-gather across DP group | Materialize full parameter tensor: $2\Phi_{\text{unit}}$ bytes |
| 3 | **FORWARD (LOCAL)** | None | Compute activations locally using full parameters |
| 4 | **FREE FULL WEIGHTS** | None | Deallocate the all-gathered full-parameter buffer, retaining only the local shard |

> **Key insight:** The full parameter tensor exists in HBM **only for the duration of the local forward computation** of that FSDP unit. This transient materialization is the core mechanism enabling memory savings.

#### 1.2.2 Backward Pass Execution (Per FSDP Unit)

| Step | Operation | Communication | Memory Effect |
|------|-----------|---------------|---------------|
| 1 | **ALL-GATHER** | All-gather across DP group | Re-materialize full parameters for gradient computation |
| 2 | **BACKWARD (LOCAL)** | None | Compute local gradients using full parameters and saved activations |
| 3 | **REDUCE-SCATTER** | Reduce-scatter across DP group | Produce averaged gradient shard; each rank receives its $\frac{1}{N}$ slice |
| 4 | **FREE FULL WEIGHTS** | None | Deallocate the re-gathered full-parameter buffer |

#### 1.2.3 Optimizer Step

| Step | Operation | Communication | Memory Effect |
|------|-----------|---------------|---------------|
| 1 | **UPDATE WEIGHTS (LOCAL)** | None | Apply optimizer (e.g., Adam) on the local gradient shard and local optimizer states |

If CPU offloading is enabled:
- Gradients are offloaded D2H after reduce-scatter.
- The optimizer step executes on CPU using the offloaded gradient shard.
- Updated parameter shards are transferred H2D before the next forward pass.

#### 1.2.4 CPU Offloading Flow

```
Forward Entry:
  IF cpu_offload_enabled:
    H2D_transfer(local_parameter_shard)   // PCIe or NVLink-C2C
  ALL_GATHER(local_shard → full_params)

Backward Exit:
  REDUCE_SCATTER(full_grads → local_grad_shard)
  IF cpu_offload_enabled:
    D2H_transfer(local_grad_shard)
    CPU_optimizer_step(local_grad_shard, optimizer_states)
```

### 1.3 Multi-Unit Pipelining and Synchronization

In a model with $K$ FSDP units, the forward pass sequentially processes units $1, 2, \ldots, K$, and the backward pass processes them in reverse order $K, K-1, \ldots, 1$. The collectives form a **chain of dependencies**:

$$
\text{AG}_i^{\text{fwd}} \rightarrow \text{Compute}_i^{\text{fwd}} \rightarrow \text{AG}_{i+1}^{\text{fwd}} \rightarrow \cdots \rightarrow \text{AG}_K^{\text{bwd}} \rightarrow \text{Compute}_K^{\text{bwd}} \rightarrow \text{RS}_K^{\text{bwd}} \rightarrow \text{AG}_{K-1}^{\text{bwd}} \rightarrow \cdots
$$

The critical observation: **adjacent FSDP units' collectives can be overlapped with the current unit's compute** if scheduled on separate CUDA streams. This is the foundational principle that SimpleFSDP's reordering optimization exploits.

### 1.4 FSDP Execution Pseudocode

```
ALGORITHM: FSDP_Training_Step

INPUT:
  model partitioned into K FSDP units: {Unit_1, ..., Unit_K}
  each Unit_k holds local_shard_k of size Φ_k / N
  input mini-batch x

// ============ FORWARD PASS ============
FOR k = 1 TO K:
    IF cpu_offload:
        H2D_ASYNC(local_shard_k)
        STREAM_SYNC(compute_stream, transfer_stream)
    
    full_params_k ← ALL_GATHER(local_shard_k, group=dp_group)
    // full_params_k now has Φ_k parameters
    
    activations_k ← FORWARD_LOCAL(Unit_k, full_params_k, activations_{k-1})
    
    FREE(full_params_k)   // release N-1/N of gathered memory
    SAVE_FOR_BACKWARD(activations_k)  // or discard if using activation checkpointing

// ============ BACKWARD PASS ============
FOR k = K DOWNTO 1:
    full_params_k ← ALL_GATHER(local_shard_k, group=dp_group)
    
    grad_full_k ← BACKWARD_LOCAL(Unit_k, full_params_k, saved_activations_k, grad_output_k)
    
    grad_shard_k ← REDUCE_SCATTER(grad_full_k, op=AVG, group=dp_group)
    // grad_shard_k has Φ_k / N elements
    
    FREE(full_params_k)
    
    IF cpu_offload:
        D2H_ASYNC(grad_shard_k)

// ============ OPTIMIZER STEP ============
FOR k = 1 TO K:
    IF cpu_offload:
        CPU_ADAM_UPDATE(local_shard_k, grad_shard_k, optimizer_state_k)
    ELSE:
        GPU_ADAM_UPDATE(local_shard_k, grad_shard_k, optimizer_state_k)
```

### 1.5 Peak Memory During Execution

The transient peak memory for a single FSDP unit $k$ during training occurs at the moment when both the full parameters and the full gradients coexist in HBM (during backward):

$$
M_{\text{peak}}^{k} = \underbrace{\frac{\Phi_k}{N} \cdot 2}_{\text{local shard (BF16)}} + \underbrace{\Phi_k \cdot 2}_{\text{all-gathered params (BF16)}} + \underbrace{\Phi_k \cdot 2}_{\text{full gradients (BF16)}} + \underbrace{A_k}_{\text{activations}}
$$

After reduce-scatter and free:

$$
M_{\text{steady}}^{k} = \frac{\Phi_k}{N} \cdot 2 + \frac{\Phi_k}{N} \cdot 2 + \frac{\Phi_k}{N} \cdot 12
$$

This transient spike is the **binding constraint** when fitting large models and determines the maximum FSDP-unit granularity.

---

## 2. Collective Communication Decomposition

### 2.1 The Fundamental Identity

The all-reduce collective, which produces a globally reduced result replicated on every rank, decomposes exactly into two primitive collectives:

$$
\boxed{\textbf{All-Reduce} = \textbf{Reduce-Scatter} + \textbf{All-Gather}}
$$

This decomposition is not merely algebraic—it is the **architectural foundation** of FSDP. Standard DP uses all-reduce for gradient synchronization; FSDP replaces this with the two constituent collectives executed at different points in the computation graph, enabling sharded storage between them.

### 2.2 All-Reduce

**Semantics:** Every rank $r \in \{0, 1, \ldots, N-1\}$ holds input tensor $X_r$. After all-reduce with summation:

$$
\forall r: \quad Y_r = \sum_{i=0}^{N-1} X_i
$$

Every rank receives an identical copy of the fully reduced result.

**Communication volume** (ring algorithm):

$$
V_{\text{all-reduce}} = 2 \cdot \frac{N-1}{N} \cdot S \approx 2S \quad \text{for large } N
$$

where $S$ is the tensor size in bytes.

| Rank | Before | After |
|------|--------|-------|
| GPU 0 | $A$ | $A + B + C + D$ |
| GPU 1 | $B$ | $A + B + C + D$ |
| GPU 2 | $C$ | $A + B + C + D$ |
| GPU 3 | $D$ | $A + B + C + D$ |

### 2.3 Reduce-Scatter

**Semantics:** Each rank $r$ holds input tensor $X_r$ logically partitioned into $N$ chunks: $X_r = [X_r^{(0)}, X_r^{(1)}, \ldots, X_r^{(N-1)}]$. After reduce-scatter:

$$
\text{Rank } r \text{ receives: } Y_r = \sum_{i=0}^{N-1} X_i^{(r)}
$$

Each rank receives only its **own shard** of the reduced result.

**Communication volume** (ring algorithm):

$$
V_{\text{reduce-scatter}} = \frac{N-1}{N} \cdot S \approx S \quad \text{for large } N
$$

| Rank | Before (4 chunks each) | After (1 chunk each) |
|------|----------------------|---------------------|
| GPU 0 | $[A_0, A_1, A_2, A_3]$ | $A_0 + B_0 + C_0 + D_0$ |
| GPU 1 | $[B_0, B_1, B_2, B_3]$ | $A_1 + B_1 + C_1 + D_1$ |
| GPU 2 | $[C_0, C_1, C_2, C_3]$ | $A_2 + B_2 + C_2 + D_2$ |
| GPU 3 | $[D_0, D_1, D_2, D_3]$ | $A_3 + B_3 + C_3 + D_3$ |

### 2.4 All-Gather

**Semantics:** Each rank $r$ holds a shard $X_r$. After all-gather:

$$
\forall r: \quad Y_r = [X_0, X_1, \ldots, X_{N-1}]
$$

Every rank receives the **concatenation** of all shards—a full replica of the data.

**Communication volume** (ring algorithm):

$$
V_{\text{all-gather}} = \frac{N-1}{N} \cdot S \approx S \quad \text{for large } N
$$

| Rank | Before (1 shard each) | After (all shards) |
|------|----------------------|-------------------|
| GPU 0 | $A$ | $[A, B, C, D]$ |
| GPU 1 | $B$ | $[A, B, C, D]$ |
| GPU 2 | $C$ | $[A, B, C, D]$ |
| GPU 3 | $D$ | $[A, B, C, D]$ |

### 2.5 Verification of the Decomposition

Consider the all-reduce of tensors $\{X_0, X_1, X_2, X_3\}$ on 4 GPUs:

1. **Reduce-Scatter phase:** Each GPU $r$ receives $\sum_{i} X_i^{(r)}$—the reduced $r$-th chunk.
2. **All-Gather phase:** The reduced chunks are gathered so every GPU holds the full $[\sum_i X_i^{(0)}, \sum_i X_i^{(1)}, \sum_i X_i^{(2)}, \sum_i X_i^{(3)}] = \sum_i X_i$.

The total volume is:

$$
V_{\text{RS}} + V_{\text{AG}} = \frac{N-1}{N}S + \frac{N-1}{N}S = 2\frac{N-1}{N}S = V_{\text{all-reduce}}
$$

This confirms that the decomposition is **bandwidth-optimal** and introduces no overhead.

### 2.6 FSDP's Use of the Decomposition

| DP Phase | Collective Used | When | Why |
|----------|----------------|------|-----|
| Standard DP gradient sync | All-Reduce | After backward | Every rank needs full gradient for full-copy optimizer |
| FSDP forward | **All-Gather** | Before forward compute | Reconstruct full parameters from shards |
| FSDP backward (params) | **All-Gather** | Before backward compute | Re-materialize full parameters for gradient computation |
| FSDP backward (grads) | **Reduce-Scatter** | After backward compute | Produce averaged gradient shard for local optimizer |

> **Critical insight for FSDP designers:** By splitting the all-reduce into its two halves and inserting computation between them, FSDP achieves sharded memory semantics. The reduce-scatter produces a shard; the all-gather (deferred to the next usage point) reconstructs from shards. The interval between these two operations is exactly where the memory saving materializes.

### 2.7 Latency Cost Model

For any collective over $N$ ranks with message size $S$ bytes, using a ring algorithm on a network with per-message latency $\alpha$ and inverse bandwidth $\beta$ (seconds per byte):

$$
T_{\text{all-gather}} = (N-1)\alpha + \frac{N-1}{N} S \beta
$$

$$
T_{\text{reduce-scatter}} = (N-1)\alpha + \frac{N-1}{N} S \beta
$$

$$
T_{\text{all-reduce}} = 2(N-1)\alpha + 2\frac{N-1}{N} S \beta
$$

For NCCL's tree algorithm (commonly used on NVSwitch topologies):

$$
T_{\text{all-reduce}}^{\text{tree}} = 2 \log_2(N) \cdot \alpha + 2S\beta
$$

The per-message latency $\alpha$ is the **base latency** that SimpleFSDP's bucketing optimization targets.

---

## 3. SimpleFSDP: Compiler-Native FSDP via DTensor and TorchInductor

### 3.1 Architectural Philosophy

SimpleFSDP reimagines FSDP as a **compiler-first** distributed training strategy, in contrast to the runtime-centric approach of PyTorch FSDP1/FSDP2. The key architectural decisions are:

| Design Axis | Traditional FSDP (FSDP1/FSDP2) | SimpleFSDP |
|-------------|-------------------------------|-----------|
| Parameter sharding | Runtime hooks on `nn.Module` | DTensor with `Shard` placement |
| All-gather dispatch | Forward pre-hooks | `parametrization` + DTensor `redistribute` |
| Reduce-scatter dispatch | Backward post-hooks | Automatic via autograd of `redistribute` |
| Overlap scheduling | Manual stream management | TorchInductor IR node reordering |
| Bucketing | Runtime gradient bucket manager | TorchInductor IR node fusion |
| Graph capture | Incompatible or partial | Full graph via `torch.compile(fullgraph=True)` |
| Debuggability | Opaque hook-based execution | Transparent eager-mode semantics |

### 3.2 Core Mechanism: Parametrization + DTensor Redistribute

SimpleFSDP represents each model parameter as a **DTensor** with `Shard(0)` placement on the data-parallel mesh. The `ReplicateComputation` parametrization intercepts parameter access and issues the appropriate redistribute:

```
ALGORITHM: ReplicateComputation_Parametrization

CLASS ReplicateComputation:
    ATTRIBUTES: param_dtype, reduce_dtype

    FUNCTION replicate_compute(self, x):
        // x is a DTensor with Shard(0) placement on DP mesh
        // Shape: [Φ_unit / N, ...]
        
        replicated = x.redistribute(
            placements=(Replicate(),),
            forward_dtype=self.param_dtype,    // e.g., BF16
            backward_dtype=self.reduce_dtype   // e.g., FP32
        )
        // replicated: DTensor with Replicate() placement
        // Shape: [Φ_unit, ...]
        // Forward: issues ALL_GATHER on DP group
        // Backward: autograd generates REDUCE_SCATTER on DP group
        
        local_tensor = replicated.to_local(
            grad_placements=(Partial(reduce_op="avg"),)
        )
        // Extract local tensor for computation
        // grad_placements defines backward reduce semantics
        
        RETURN local_tensor
```

**Why this is elegant:**
- The `redistribute` call from `Shard(0)` to `Replicate()` **is** the all-gather—expressed as a differentiable DTensor operation.
- The backward pass of `redistribute` from `Replicate()` to `Shard(0)` **is** the reduce-scatter—generated automatically by autograd.
- Wrapping this in `selective_activation_checkpointing` means the all-gathered parameters are treated as "activations" that can be freed after forward and recomputed (re-gathered) in backward.

### 3.3 Selective Activation Checkpointing for Weight Freeing

The "free full weights after use" behavior of FSDP is implemented as selective activation checkpointing applied specifically to the parametrization output:

$$
\text{Selective AC on } \texttt{replicate\_compute} \Rightarrow \begin{cases} \text{Forward:} & \text{all-gather} \rightarrow \text{compute} \rightarrow \text{free full params} \\ \text{Backward:} & \text{re-all-gather} \rightarrow \text{backward compute} \rightarrow \text{reduce-scatter} \rightarrow \text{free} \end{cases}
$$

This reuse of the activation checkpointing primitive avoids introducing any FSDP-specific memory management hooks.

### 3.4 Graph Tracing and TorchInductor Lowering

After `simple_fsdp(model)` wrapping and `torch.compile`, the entire forward and backward pass—including all communication operations—is traced into a **single FX graph**, which is then lowered to TorchInductor IR nodes:

```
ALGORITHM: SimpleFSDP_Compilation_Pipeline

INPUT: model, training data loader
OUTPUT: optimized compiled training step

Step 1: WRAP
    wrapped_model ← simple_fsdp(model)
    // Applies DTensor sharding + ReplicateComputation parametrization
    // Applies selective activation checkpointing on parametrization

Step 2: COMPILE
    compiled_model ← torch.compile(wrapped_model, fullgraph=True)
    // Dynamo traces full FX graph including:
    //   - ALL_GATHER nodes (from redistribute)
    //   - COMPUTE nodes (matmul, layernorm, etc.)
    //   - REDUCE_SCATTER nodes (from autograd of redistribute)

Step 3: LOWER TO INDUCTOR IR
    ir_graph ← TorchInductor.lower(fx_graph)
    // Each operation becomes an IR node with:
    //   - Module provenance metadata
    //   - Estimated execution time (profiled or modeled)
    //   - Memory footprint

Step 4: OPTIMIZE IR (SimpleFSDP passes)
    bucketed_graph ← BUCKET(ir_graph)        // Section 4
    reordered_graph ← REORDER(bucketed_graph) // Section 5

Step 5: CODEGEN
    executable ← TorchInductor.codegen(reordered_graph)
    // Generate CUDA/HIP kernels with optimized scheduling
```

The critical advantage: **the compiler has visibility into both communication and computation**, enabling global scheduling decisions that are impossible with hook-based runtimes.

### 3.5 Simplicity and Debuggability

Two fundamental properties distinguish SimpleFSDP:

1. **Simplicity:** Users write exactly three lines of configuration beyond the standard model definition:
   - Set bucketing mode (`auto` or manual module list)
   - Enable reordering
   - Wrap with `simple_fsdp()` and `torch.compile()`

2. **Debuggability:** The eager-mode execution semantics are **identical** to the compiled execution. Users can remove `torch.compile` and run the model with standard PyTorch semantics for debugging—the `redistribute` calls still produce correct all-gather/reduce-scatter behavior, just without the bucketing and reordering optimizations.

---

## 4. Bucketing: Amortizing Base Communication Latency

### 4.1 The Latency Problem

In the unoptimized graph, each parameter produces its own all-gather and reduce-scatter IR node. For a Transformer layer with $P$ distinct parameter tensors (e.g., $Q$, $K$, $V$, $O$ projections, MLP up/down/gate, LayerNorm weights and biases), the total communication overhead includes $P$ instances of the base latency $\alpha$:

$$
T_{\text{comm}}^{\text{unbucketed}} = P \cdot \alpha + P \cdot \beta \cdot s_i = P\alpha + \beta \sum_{i=1}^{P} s_i
$$

where $s_i$ is the size of parameter $i$.

After bucketing $P$ parameters into a single collective:

$$
T_{\text{comm}}^{\text{bucketed}} = \alpha + \beta \cdot \sum_{i=1}^{P} s_i
$$

The savings are $(P-1)\alpha$, which is significant when $\alpha$ is non-trivial (typically 5–15 $\mu$s for NCCL on InfiniBand, 3–8 $\mu$s on NVSwitch).

### 4.2 Bucketing Mechanism

#### 4.2.1 All-Gather Bucketing

```
ALGORITHM: Bucket_AllGather

INPUT: 
    AG_1, AG_2, ..., AG_k  // Individual all-gather IR nodes
    Each AG_i reads tensor T_i of size s_i

OUTPUT:
    AG_{1..k}  // Single bucketed all-gather IR node

Step 1: ALLOCATE
    S_total ← Σ s_i
    buffer_local ← ALLOCATE(S_total / N)  // Local shard buffer
    buffer_full ← ALLOCATE(S_total)        // Full gathered buffer

Step 2: FLATTEN AND CONCATENATE
    offset ← 0
    FOR i = 1 TO k:
        COPY(T_i.local_shard → buffer_local[offset : offset + s_i/N])
        offset ← offset + s_i/N

Step 3: ISSUE SINGLE ALL-GATHER
    AG_{1..k} ← ALL_GATHER_ASYNC(buffer_local → buffer_full, group=dp_group)
    Wa_{1..k} ← WAIT(AG_{1..k})  // All-gather-wait node

Step 4: COPY-OUT (after wait)
    offset ← 0
    FOR i = 1 TO k:
        T_i.full_params ← VIEW(buffer_full[offset : offset + s_i])
        offset ← offset + s_i
```

#### 4.2.2 Reduce-Scatter Bucketing

```
ALGORITHM: Bucket_ReduceScatter

INPUT:
    RS_1, RS_2, ..., RS_k  // Individual reduce-scatter IR nodes
    Each RS_i operates on gradient G_i of size s_i

OUTPUT:
    RS_{1..k}  // Single bucketed reduce-scatter IR node

Step 1: CHUNK AND CONCATENATE
    // Each gradient G_i is split into N chunks of size s_i/N
    FOR chunk_rank = 0 TO N-1:
        offset ← 0
        FOR i = 1 TO k:
            COPY(G_i.chunk[chunk_rank] → buffer_chunked[chunk_rank][offset : offset + s_i/N])
            offset ← offset + s_i/N

Step 2: ISSUE SINGLE REDUCE-SCATTER
    RS_{1..k} ← REDUCE_SCATTER_ASYNC(buffer_chunked, op=AVG, group=dp_group)
    Wr_{1..k} ← WAIT(RS_{1..k})  // Reduce-scatter-wait node

Step 3: READ-OUT (after wait)
    offset ← 0
    FOR i = 1 TO k:
        G_i.shard ← VIEW(output_buffer[offset : offset + s_i/N])
        offset ← offset + s_i/N
```

### 4.3 Bucketing Cost-Benefit Analysis

| Factor | Unbucketed ($P$ collectives) | Bucketed (1 collective) |
|--------|-------------------------------|------------------------|
| Base latency | $P \cdot \alpha$ | $\alpha$ |
| Bandwidth term | $\beta \sum s_i$ | $\beta \sum s_i$ |
| Extra buffer memory | 0 | $\sum s_i$ (temporary) |
| Copy-out overhead | 0 | $O(\sum s_i)$ memcpy |
| Overlap granularity | Fine-grained (per-param) | Coarse-grained (per-bucket) |

> **Engineering trade-off:** Aggressive bucketing amortizes latency but reduces overlap granularity. If a single bucket encompasses too many parameters, the all-gather for that bucket cannot be hidden behind the previous computation because its volume exceeds the computation time. This is exactly the tension that the auto-wrapping algorithm resolves.

---

## 5. Reordering: Prefetching Communication for Overlap

### 5.1 The Overlap Principle

NCCL all-gather and reduce-scatter are **asynchronous** operations. An all-gather issued on the communication stream can execute concurrently with compute kernels on the default compute stream. The goal of reordering is to **schedule the next unit's communication to overlap with the current unit's computation**.

The ideal scheduling achieves:

$$
T_{\text{step}} = \max\left(\sum_k T_{\text{compute}}^k, \sum_k T_{\text{comm}}^k\right) + T_{\text{exposed}}
$$

where $T_{\text{exposed}}$ is the communication time that **cannot** be hidden behind computation. Perfect overlap yields $T_{\text{exposed}} = 0$.

### 5.2 Forward Pass Reordering

**Before reordering** (sequential execution):

```
AG12 → Wa12 → C1 → C2 → AG34 → Wa34 → C3 → C4
```

Here $\text{AG}_{34}$ starts only after $C_2$ completes—its latency is fully exposed.

**After reordering** (prefetched execution):

```
AG12 → AG34 → Wa12 → C1 → C2 → Wa34 → C3 → C4
```

Now $\text{AG}_{34}$ is issued **before** $\text{Wa}_{12}$, meaning $\text{AG}_{34}$ executes on the communication stream while $C_1$ and $C_2$ execute on the compute stream. The communication for the **next** FSDP unit is prefetched during the **current** unit's computation.

**Timeline visualization:**

```
Compute stream: |----Wa12-copy-out----|---C1---|---C2---|----Wa34-copy-out----|---C3---|---C4---|
Comm stream:    |--AG12--|---AG34 (overlapped with C1, C2)---|
```

The copy-out after $\text{Wa}_{12}$ (extracting individual tensors from the bucketed buffer) is itself computation that $\text{AG}_{34}$ can overlap with.

### 5.3 Backward Pass Reordering

The backward pass is more complex because both all-gather (for parameters) and reduce-scatter (for gradients) must be scheduled.

**Before reordering:**

```
AG12 → Wa12 → C1 → C2 → RS12 → Wr12 → AG34 → Wa34 → C3 → C4 → RS34 → Wr34
```

**After reordering:**

```
AG12 → Wa12 → AG34 → C1 → C2 → RS12 → Wa34 → C3 → C4 → Wr12 → RS34 → Wr34
```

**Overlap achieved:**

| Communication | Overlapped with |
|--------------|-----------------|
| $\text{AG}_{34}$ | Computation $C_1$, $C_2$ |
| $\text{RS}_{12}$ | Computation $C_3$, $C_4$ (after $\text{Wa}_{34}$ copy-out) |

**Key scheduling rule for backward:** $\text{Wr}_{12}$ (reduce-scatter-wait for the previous bucket) is placed **before** $\text{RS}_{34}$, allowing $\text{RS}_{12}$ to overlap with later computation. The $\text{AG}$ is placed **after** the preceding $\text{Wa}$ because the copy-out compute from the all-gather provides additional overlap opportunity.

### 5.4 Reordering Constraints

The reordering must respect data dependencies:

1. $\text{Wa}_i$ must follow $\text{AG}_i$ (cannot use gathered data before gather completes).
2. $C_i$ must follow $\text{Wa}_i$ (computation requires materialized parameters).
3. $\text{RS}_i$ must follow $C_i$ (reduce-scatter requires computed gradients).
4. $\text{Wr}_i$ must follow $\text{RS}_i$ (cannot read reduced gradient before scatter completes).

Within these constraints, the reordering maximizes the temporal distance between each async launch and its corresponding wait.

### 5.5 Formal Overlap Condition

For communication of bucket $j$ to be fully hidden by computation of bucket $i$:

**Forward pass:**

$$
T_{\text{AG}_j} \leq T_{C_i} + T_{\text{copy-out}_i}
$$

**Backward pass:**

$$
T_{\text{RS}_{i-1}} + T_{\text{AG}_j} \leq T_{C_i} + T_{\text{copy-out}_i}
$$

If these inequalities are violated, the excess communication time appears as **exposed latency** on the critical path.

---

## 6. Model Wrapping Strategies: Manual vs. Auto

### 6.1 Manual Wrapping

#### 6.1.1 Mechanism

Users provide a list of module names (e.g., `["layers.0", "layers.1", ...]`). SimpleFSDP uses the TorchInductor IR node metadata—which preserves the original module provenance—to construct a mapping:

$$
\texttt{module\_name} \rightarrow \{IR_{\text{AG}}, IR_{\text{RS}}, IR_{\text{compute}}\}
$$

All communication IR nodes belonging to the same module are bucketed together, and the buckets are reordered according to the scheme in Section 5.

#### 6.1.2 Properties

- **Granularity:** One FSDP unit per user-specified module boundary.
- **Predictability:** Users control the bucket boundaries explicitly.
- **Compatibility:** Equivalent to FSDP2's `fully_shard()` per-module wrapping.
- **Risk:** Sub-optimal bucket sizes if modules have heterogeneous parameter counts.

#### 6.1.3 Example

For a 2-module model (Module 1 contains layers 1–2, Module 2 contains layers 3–4):

```
Forward (after bucket + reorder):
  Compute stream: |---C1---|---C2---|---C3---|---C4---|
  Comm stream:    |--AG12--|---AG34 (overlapped)---|

Backward (after bucket + reorder):
  Compute stream: |---C1---|---C2---|---C3---|---C4---|
  Comm stream:    |--AG12--|--AG34 (overlapped)--|--RS12 (overlapped)--|--RS34--|
```

### 6.2 Auto Wrapping

#### 6.2.1 Design Philosophy

Auto wrapping provides **optimal bucketing without user intervention**. Since SimpleFSDP shards per-parameter (not per-module), it employs a **greedy algorithm** that iteratively considers whether to merge the next parameter's communication into the current bucket, subject to time and memory constraints.

#### 6.2.2 Profiling Phase

Before wrapping decisions, SimpleFSDP profiles each IR node:

**Computation profiling:**
- Convert `FakeTensor` metadata to real tensors.
- Execute the computation kernel with real inputs.
- Record CUDA event time $T_c$ and peak memory $M_c$.

**Communication profiling:**
- Use the analytical model:

$$
T_m = \alpha + \beta \cdot n
$$

where $n$ is the transmitted word size (bytes), $\alpha$ is the base latency, and $\beta$ is the per-byte transmit time.

| Variable | Definition |
|----------|-----------|
| $T_m^{AG}$ | Current step's bucketed all-gather communication time |
| $T_c$ | Current step's computation time |
| $M_c$ | Next step's peak computation memory |
| $T_m^{RS}$ | Last step's bucketed reduce-scatter communication time |
| $T_{m_i}^{AG}$ | $i$-th all-gather's communication time |
| $T_{c_i}$ | Time to compute with parameters prefetched by $i$-th all-gather |
| $M_{c_i}$ | Peak memory to compute with parameters prefetched by $i$-th all-gather |
| $T_{m_i}^{RS}$ | Time to reduce-scatter the gradient for parameters in $i$-th all-gather |

#### 6.2.3 Auto-Wrapping Decision Algorithm

```
ALGORITHM: Auto_Wrapping_Decision

INPUT:
    T_m^AG    // Current bucketed AG time
    T_m^RS    // Previous bucketed RS time
    T_c       // Current computation time available for overlap
    M_c       // Current computation peak memory
    T_mi^AG   // Candidate i-th AG communication time
    T_ci      // Computation time for candidate's parameters
    M_ci      // Peak memory for candidate's computation
    M_max     // Maximum allowed HBM memory

OUTPUT: 
    decision ∈ {BUCKET, NO_BUCKET}

IF phase == FORWARD:
    // Time constraint: can the enlarged bucket AG still be hidden?
    time_ok ← ( T_{m+m_i}^{AG} ≤ T_c )
    
    // Memory constraint: does prefetching the next step's 
    // parameters stay within HBM budget?
    mem_ok ← ( M_c + M_{c_i} ≤ M_max )
    
    IF time_ok AND mem_ok:
        RETURN BUCKET
    ELSE:
        RETURN NO_BUCKET

ELSE IF phase == BACKWARD:
    // Time constraint: must account for BOTH the previous RS
    // and the enlarged AG being hidden by current compute
    time_ok ← ( T_m^{RS} + T_{m+m_i}^{AG} ≤ T_c )
    
    // Memory constraint: same as forward
    mem_ok ← ( M_c + M_{c_i} ≤ M_max )
    
    IF time_ok AND mem_ok:
        RETURN BUCKET
        // Also bucket the corresponding RS nodes
    ELSE:
        RETURN NO_BUCKET
```

#### 6.2.4 Why the Backward Constraint is Stricter

In the backward pass, the communication stream must execute **both** the reduce-scatter from the previous computation step and the all-gather for the current step. Both must complete within the window of the current computation:

$$
T_m^{RS} + T_{m+m_i}^{AG} \leq T_c
$$

This dual constraint means backward buckets tend to be **smaller** than forward buckets for the same model, which is the correct behavior: backward compute typically has higher arithmetic intensity (gradient computation + activation recomputation), providing more overlap budget.

#### 6.2.5 Greedy Bucketing Walk

```
ALGORITHM: Greedy_Auto_Bucket

INPUT:
    Ordered list of per-parameter AG/RS/Compute IR nodes
    M_max: memory limit

OUTPUT:
    List of bucketed AG/RS groups with reordering

current_bucket ← {}
current_T_AG ← 0
current_T_RS ← 0
current_T_c ← 0
current_M_c ← 0

FOR each parameter p_i in layer order:
    candidate_T_AG ← COMM_TIME(current_bucket ∪ {p_i})  // T_{m+m_i}^AG
    candidate_M ← current_M_c + M_{c_i}
    
    IF phase == FORWARD:
        can_bucket ← (candidate_T_AG ≤ current_T_c) AND (candidate_M ≤ M_max)
    ELSE:  // BACKWARD
        can_bucket ← (current_T_RS + candidate_T_AG ≤ current_T_c) AND (candidate_M ≤ M_max)
    
    IF can_bucket:
        current_bucket ← current_bucket ∪ {p_i}
        UPDATE current_T_AG, current_M_c
    ELSE:
        EMIT_BUCKET(current_bucket)
        current_bucket ← {p_i}
        current_T_RS ← RS_TIME(previous_bucket)  // for backward
        RESET current_T_AG, current_T_c, current_M_c for new bucket

EMIT_BUCKET(current_bucket)  // final bucket

APPLY_REORDER(all emitted buckets)  // Section 5 reordering
```

#### 6.2.6 Auto vs. Manual Wrapping Comparison

| Property | Manual Wrapping | Auto Wrapping |
|----------|----------------|---------------|
| User input required | Module boundary list | None (fully automatic) |
| Bucket granularity | Per-module | Per-parameter (greedy optimal) |
| Adapts to compute/comm ratio | No (fixed boundaries) | Yes (profile-driven) |
| Adapts to memory budget | No (user responsibility) | Yes ($M_{\max}$ constraint) |
| Heterogeneous layers | Sub-optimal | Optimal bucket sizing per layer |
| Implementation complexity | Lower (module metadata lookup) | Higher (profiling + greedy algorithm) |
| Forward/Backward asymmetry | Same boundaries for both | Different bucket sizes for forward/backward |

### 6.3 Auto Wrapping Example

Consider a 4-parameter model where:
- $T_{c_1} = 10\mu s$, $T_{c_2} = 8\mu s$, $T_{c_3} = 12\mu s$, $T_{c_4} = 6\mu s$
- $T_{m_1}^{AG} = 3\mu s$, $T_{m_2}^{AG} = 3\mu s$, $T_{m_3}^{AG} = 7\mu s$, $T_{m_4}^{AG} = 4\mu s$
- Memory per-param: $M_{c_i} = 200\text{MB}$ each, $M_{\max} = 800\text{MB}$

**Forward auto-bucketing:**
1. Start with $p_1$: bucket = $\{p_1\}$, $T_m^{AG} = 3\mu s$.
2. Consider $p_2$: $T_{m_{1+2}}^{AG} = 5\mu s \leq T_{c_1} = 10\mu s$ ✓, $M = 400\text{MB} \leq 800\text{MB}$ ✓ → bucket.
3. Consider $p_3$: $T_{m_{1+2+3}}^{AG} = 11\mu s > T_c = 10\mu s$ ✗ → emit bucket $\{p_1, p_2\}$, start new.
4. Start with $p_3$: bucket = $\{p_3\}$, $T_m^{AG} = 7\mu s$.
5. Consider $p_4$: $T_{m_{3+4}}^{AG} = 10\mu s \leq T_{c_3} = 12\mu s$ ✓, $M = 400\text{MB} \leq 800\text{MB}$ ✓ → bucket.
6. Emit bucket $\{p_3, p_4\}$.

Result: `AG_{12}`, `AG_{34}` — happens to match the manual example but was derived automatically.

---

## 7. Composability with Multi-Dimensional Parallelism

### 7.1 Composability Architecture

SimpleFSDP's DTensor-native design provides seamless composability with other parallelism dimensions because DTensor inherently supports **multi-dimensional meshes**.

#### 7.1.1 Meta Initialization

During model initialization on `torch.device("meta")`, SimpleFSDP disables the all-gather parametrization:

```
ALGORITHM: Meta_Init_Optimization

IF device == "meta":
    DISABLE parametrization.replicate_compute
    // Parameters exist as metadata-only tensors
    // No all-gather is issued during initialization
    // Reduces init time from O(N * Φ) to O(Φ/N) per rank
    
AFTER weight loading:
    ENABLE parametrization.replicate_compute
    // Subsequent forward passes issue all-gather as normal
```

This avoids the $O(N)$ memory and time cost of gathering parameters during initialization, which can be substantial for models with $\Phi > 100B$ parameters.

#### 7.1.2 Mixed Precision Training

Mixed precision is implemented through the DTensor `redistribute` call's dtype arguments:

| Precision Component | Configuration | Effect |
|--------------------|---------------|--------|
| `param_dtype` | BF16 or FP16 | Parameters cast to this dtype during all-gather in forward |
| `reduce_dtype` | FP32 | Gradients cast to this dtype during reduce-scatter in backward |
| Optimizer states | FP32 (always) | Adam moments and master weights maintained at full precision |

The forward all-gather transmits parameters in `param_dtype` (2 bytes per element), reducing communication volume by $2\times$ compared to FP32. The backward reduce-scatter uses `reduce_dtype` for numerical stability in gradient accumulation.

The memory for mixed precision FSDP per rank:

$$
M_{\text{mixed-FSDP}}^{\text{rank}} = \frac{1}{N}\bigl(\underbrace{2\Phi}_{\text{BF16 params}} + \underbrace{4\Phi}_{\text{FP32 grads (reduce dtype)}} + \underbrace{12\Phi}_{\text{FP32 optimizer}}\bigr) = \frac{18\Phi}{N}
$$

However, in practice, the gradient shard is kept in the `reduce_dtype` only for the reduce-scatter operation; the stored gradient shard can be in BF16 for the optimizer if the optimizer itself handles upcasting:

$$
M_{\text{practical}}^{\text{rank}} = \frac{1}{N}\bigl(2\Phi + 2\Phi + 12\Phi\bigr) = \frac{16\Phi}{N}
$$

#### 7.1.3 Tensor Parallel (TP) Composability

For 2D parallelism (FSDP + TP), each parameter is a **2D DTensor** sharded on both dimensions:

```
ALGORITHM: 2D_Parallelism_Redistribute

// Device mesh: (DP=dp_size, TP=tp_size), total world_size = dp_size × tp_size
// Parameter P initialized as DTensor with placement:
//   Shard(0) on DP dim, Shard(col_dim) on TP dim
// Shape per rank: [Φ / (dp_size × tp_size)]

FUNCTION replicate_compute_2d(self, x):
    // x: DTensor([Shard(0), Shard(tp_dim)])
    
    // Step 1: All-gather on DP sub-mesh only
    x_dp_replicated = x.redistribute(
        placements=(Replicate(), Shard(tp_dim)),
        mesh_dim=0  // DP dimension
    )
    // x_dp_replicated: DTensor([Replicate(), Shard(tp_dim)])
    // Now replicated across DP, still sharded across TP
    
    // Step 2: TP computation proceeds with standard column/row parallel
    local_tp_shard = x_dp_replicated.to_local()
    // Ready for TP matmul operations
    
    RETURN local_tp_shard
```

The DP all-gather gathers across the DP group (typically inter-node over InfiniBand), while the TP communication (all-reduce or reduce-scatter within TP forward/backward) occurs across the TP group (typically intra-node over NVLink/NVSwitch).

**Topology-aware placement:**

$$
\text{TP group:} \quad \text{GPUs within the same node (NVLink: 450–900 GB/s per H100)} 
$$

$$
\text{DP group:} \quad \text{GPUs across nodes (InfiniBand: 50–100 GB/s per link)}
$$

#### 7.1.4 Pipeline Parallel (PP) Composability

```
ALGORITHM: PP_SimpleFSDP_Composition

// Model partitioned into S pipeline stages
// Each stage assigned to a subset of ranks

FOR each stage s:
    submodel_s ← extract_submodule(model, stage_boundaries[s])
    
    // SimpleFSDP wraps the local stage submodule
    fsdp_submodel_s ← simple_fsdp(submodel_s)
    
    // Compile the wrapped submodule
    compiled_stage_s ← torch.compile(fsdp_submodel_s)

// Pipeline schedule (1F1B, interleaved, etc.) orchestrates
// micro-batch execution across stages
// SimpleFSDP is transparent to the pipeline scheduler
```

No additional code is required because SimpleFSDP operates at the submodule level—it does not need global model visibility.

#### 7.1.5 Activation Checkpointing Composability

```
ALGORITHM: AC_SimpleFSDP_Composition

// Step 1: Apply user-defined activation checkpointing
apply_activation_checkpointing(
    model, 
    check_fn=lambda m: isinstance(m, TransformerLayer),
    policy=SELECTIVE  // or FULL
)

// Step 2: SimpleFSDP wraps the model (adds its own selective AC for weights)
wrapped = simple_fsdp(model)

// Step 3: Compile
compiled = torch.compile(wrapped, fullgraph=True)

// The two levels of activation checkpointing compose:
//   - User's AC: recompute activations (attention outputs, MLP intermediates)
//   - SimpleFSDP's AC: recompute all-gathered parameters (weight freeing)
```

The user's activation checkpointing reduces activation memory; SimpleFSDP's internal selective checkpointing reduces parameter memory. They compose orthogonally.

### 7.2 Multi-Dimensional Parallelism Summary

For a full 4D parallelism configuration on an H100 cluster:

| Dimension | Degree | Communication | Network | SimpleFSDP Role |
|-----------|--------|---------------|---------|-----------------|
| TP | 8 (intra-node) | All-reduce, all-gather within TP group | NVSwitch (900 GB/s bisection) | DTensor 2D sharding |
| PP | 4–16 (inter-node) | Point-to-point send/recv | InfiniBand (400 Gbps) | Wraps per-stage submodule |
| DP/FSDP | $W / (TP \times PP)$ | All-gather, reduce-scatter | InfiniBand or NVLink | Core sharding + bucketing + reordering |
| CP | 2–8 | Ring all-gather of KV | NVLink or InfiniBand | Composes at attention level |

The world-size factorization must satisfy:

$$
W = DP \times TP \times PP \times CP
$$

and SimpleFSDP operates on the DP dimension of the resulting device mesh.

---

## 8. Memory Analysis and Constraint Modeling

### 8.1 Memory Budget Decomposition

For a Transformer model with $L$ layers, hidden dimension $H$, vocabulary size $V$, sequence length $S$, micro-batch size $B$, and FSDP world size $N$:

**Per-rank sharded state:**

$$
M_{\text{sharded}} = \frac{1}{N} \cdot \left( 2\Phi + 2\Phi + 12\Phi \right) = \frac{16\Phi}{N}
$$

**Transient all-gathered parameters** (peak during forward or backward of one FSDP unit wrapping $L_u$ layers):

$$
M_{\text{transient}} = 2 \cdot \Phi_{\text{unit}} = 2 \cdot L_u \cdot (12H^2 + 13H) \approx 24 L_u H^2 \text{ bytes (BF16)}
$$

**Activations** (with selective activation checkpointing, per Transformer layer):

$$
M_{\text{act}}^{\text{layer}} \approx 2BSH + 34BSH \cdot \frac{1}{\text{AC\_ratio}}
$$

where $\text{AC\_ratio}$ is the fraction of layers checkpointed.

**Total per-rank peak:**

$$
M_{\text{peak}} = M_{\text{sharded}} + M_{\text{transient}} + L \cdot M_{\text{act}}^{\text{layer}} + M_{\text{buffers}} + M_{\text{fragmentation}}
$$

### 8.2 Memory Constraint in Auto Wrapping

The auto-wrapping algorithm's $M_{\max}$ parameter represents the **maximum allowable transient memory for prefetched parameters**:

$$
M_{\max} = M_{\text{HBM}} - M_{\text{sharded}} - M_{\text{activations}} - M_{\text{overhead}}
$$

For an H100 with 80 GB HBM3:

$$
M_{\max} = 80\text{GB} - M_{\text{sharded}} - M_{\text{activations}} - 2\text{GB (CUDA context, buffers)}
$$

The auto-wrapping algorithm ensures that at any point during execution, the total prefetched parameter memory does not exceed this budget:

$$
\sum_{i \in \text{bucket}} M_{c_i} \leq M_{\max}
$$

### 8.3 Memory Impact of Bucketing

Bucketing introduces **temporary buffer memory** for the concatenated communication buffer:

$$
M_{\text{bucket\_buffer}} = 2 \times \text{bucket\_size\_bytes}
$$

The factor of 2 accounts for both the local shard buffer (input to all-gather) and the full gathered buffer (output of all-gather). For reduce-scatter, a similar buffer is needed for the chunked gradient concatenation.

In the worst case (single bucket for all parameters):

$$
M_{\text{bucket\_buffer}}^{\max} = 2 \cdot 2\Phi = 4\Phi \text{ bytes}
$$

This is why the memory constraint in auto wrapping is essential—it prevents a single giant bucket from causing OOM.

### 8.4 HBM Budget Table for Representative Configurations

| Model | $\Phi$ | GPU | HBM | FSDP $N$ | $M_{\text{sharded}}$ | $M_{\max}$ (approx.) |
|-------|--------|-----|-----|----------|---------------------|---------------------|
| 7B | 7×10⁹ | H100 80GB | 80 GB | 8 | 14 GB | ~50 GB |
| 70B | 70×10⁹ | H100 80GB | 80 GB | 64 | ~17.5 GB | ~40 GB |
| 405B | 405×10⁹ | H100 80GB | 80 GB | 512 | ~12.7 GB | ~45 GB |
| 70B | 70×10⁹ | B200 192GB | 192 GB | 32 | 35 GB | ~130 GB |

---

## 9. Production Deployment Considerations

### 9.1 Launch and Configuration

```
ALGORITHM: Production_SimpleFSDP_Launch

INPUT:
    model_config: architecture definition
    cluster_topology: {num_nodes, gpus_per_node, interconnect}
    memory_budget: per-GPU HBM limit
    target_throughput: tokens/sec goal

Step 1: PARALLELISM PLANNING
    tp_degree ← min(gpus_per_node, 8)  // saturate NVLink
    pp_degree ← ceil(model_layers / max_layers_per_gpu)
    dp_degree ← total_gpus / (tp_degree × pp_degree)
    
Step 2: FSDP CONFIGURATION
    torch._inductor.config.simplefsdp.bucket_mode ← "auto"
    torch._inductor.config.simplefsdp.enable_reorder ← True
    
Step 3: MODEL WRAPPING
    model ← initialize_on_meta_device(model_config)
    model ← apply_tensor_parallel(model, tp_mesh)
    model ← apply_activation_checkpointing(model, policy=SELECTIVE)
    model ← simple_fsdp(model)
    model ← torch.compile(model, fullgraph=True)
    
Step 4: VALIDATION
    // Verify no OOM: run single step with profiling
    // Verify numerical parity: compare loss curve against eager baseline
    // Verify overlap: inspect Nsight Systems trace for exposed comm
    
Step 5: SCALING
    // Run strong-scaling study: measure MFU at 1, 2, 4, 8 nodes
    // Target: > 40% MFU on H100, > 45% on B200
```

### 9.2 Graph Capture Considerations

| Scenario | `fullgraph` Setting | Rationale |
|----------|-------------------|-----------|
| Standard dense Transformer | `True` | Full graph enables maximum bucketing and reordering |
| MoE with top-k routing (data-dependent) | `False` | Control flow breaks graph; subgraphs optimized independently |
| Dynamic sequence lengths | `False` | Shape-dependent control flow requires graph breaks |
| Static shapes, no conditionals | `True` | Always preferred for maximum optimization |

When `fullgraph=False`, TorchInductor generates multiple subgraphs, each of which SimpleFSDP optimizes independently. The cross-subgraph communication cannot be bucketed, resulting in some exposed latency at graph boundaries.

### 9.3 Profiling and Diagnosis

```
ALGORITHM: SimpleFSDP_Performance_Diagnosis

Step 1: TRACE COLLECTION
    profile ← nsight_systems.trace(training_step, iterations=5)
    
Step 2: IDENTIFY EXPOSED COMMUNICATION
    FOR each comm_node in trace:
        overlap_ratio ← compute_overlap(comm_node, adjacent_compute)
        IF overlap_ratio < 0.9:
            REPORT("Exposed comm: {comm_node}, overlap={overlap_ratio}")
            
Step 3: ANALYZE BUCKET SIZES
    FOR each bucket in simplefsdp_buckets:
        comm_time ← bucket.all_gather_duration
        available_compute ← bucket.overlapping_compute_duration
        IF comm_time > available_compute:
            REPORT("Over-sized bucket: {bucket}, excess={comm_time - available_compute}")
            RECOMMEND("Reduce M_max or split bucket manually")

Step 4: MEMORY ANALYSIS
    peak_memory ← torch.cuda.max_memory_allocated()
    fragmentation ← torch.cuda.memory_stats()['active_bytes.all.peak'] - peak_memory
    IF fragmentation > 0.1 * HBM_CAPACITY:
        REPORT("Fragmentation exceeds 10%: {fragmentation}")
        RECOMMEND("Enable memory pool defragmentation or reduce bucket count")

Step 5: MFU CALCULATION
    mfu ← (6 × Φ × tokens_per_step) / (step_time × gpu_flops_peak)
    REPORT("MFU = {mfu}, target > 0.40")
```

### 9.4 Fault Tolerance and Checkpoint Compatibility

SimpleFSDP checkpoints are **DTensor-based**, meaning the saved state dict contains sharded tensors with placement metadata:

```
ALGORITHM: SimpleFSDP_Checkpoint_Resharding

// Save: each rank saves its local shard with DTensor metadata
state_dict ← model.state_dict()
// state_dict[key] = DTensor(data=local_shard, placements=[Shard(0)], mesh=dp_mesh)

// Resume with different world size N' ≠ N:
FOR each key in state_dict:
    old_placement ← state_dict[key].placements  // Shard(0) on old mesh
    new_placement ← Shard(0) on new_mesh(N')
    
    // DTensor redistribute handles resharding automatically
    state_dict[key] ← state_dict[key].redistribute(new_placement)

// This enables elastic training: survive node failures by
// resharding checkpoints to reduced world size
```

---

## 10. Cross-Vendor Portability: NVIDIA and AMD Clusters

### 10.1 NCCL vs. RCCL Considerations

| Aspect | NVIDIA (NCCL) | AMD (RCCL) |
|--------|--------------|------------|
| Transport | NVLink/NVSwitch, IB, PCIe | xGMI, IB, PCIe |
| Ring/tree selection | Auto-tuned via NCCL_ALGO, NCCL_PROTO | Manual tuning often required |
| Latency $\alpha$ | 3–8 μs (NVSwitch), 5–15 μs (IB) | 5–12 μs (xGMI), 8–20 μs (IB) |
| Bandwidth $\beta$ | ~0.9× link BW achieved | ~0.8× link BW (topology-dependent) |

SimpleFSDP's communication cost model ($T_m = \alpha + \beta n$) must be **re-profiled** when switching between NVIDIA and AMD clusters, as $\alpha$ and $\beta$ differ significantly.

### 10.2 Compiler Backend Portability

| Component | NVIDIA | AMD |
|-----------|--------|-----|
| Compute kernels | CUDA, cuDNN, Triton-CUDA | HIP, MIOpen, Triton-HIP |
| FlashAttention | flash-attn (CUDA native) | flash-attn (CK backend or Triton-HIP) |
| TorchInductor backend | Triton codegen (CUDA) | Triton codegen (HIP/ROCm) |
| Graph capture | CUDA Graphs | HIP Graphs (partial support) |

SimpleFSDP's optimizations (bucketing, reordering) operate at the **TorchInductor IR level**, which is **backend-agnostic**. The same IR transformations produce correct schedules for both CUDA and HIP backends. The per-vendor differences only affect:
1. Profiled $\alpha$, $\beta$ values (affects auto-wrapping bucket decisions)
2. Kernel codegen (Triton generates different PTX vs. AMDGPU assembly)
3. Graph capture capabilities (CUDA Graphs vs. HIP Graphs)

### 10.3 Cluster-Specific Auto-Wrapping Behavior

On MI300X (192 GB HBM3, xGMI interconnect):
- Higher $M_{\max}$ due to larger HBM → larger buckets are feasible.
- Higher $\alpha$ for xGMI → bucketing provides greater benefit.
- Different compute/comm ratio → auto-wrapping produces different bucket boundaries than H100.

On B200 (192 GB HBM3e, NVLink5):
- Higher $M_{\max}$ and higher bandwidth → can sustain very large buckets.
- Lower $\alpha$ with NVLink5 → marginal bucketing benefit per merge, but still worthwhile for large parameter counts.

---

## 11. Summary of Key Engineering Principles

### 11.1 Why SimpleFSDP Outperforms Runtime FSDP

| Principle | Runtime FSDP | SimpleFSDP |
|-----------|-------------|-----------|
| **Scheduling scope** | Per-module hooks (local decisions) | Full-graph IR (global decisions) |
| **Bucketing** | Fixed bucket size (bytes threshold) | Profile-driven, memory-aware, asymmetric forward/backward |
| **Reordering** | Manual prefetch heuristics | Compiler-driven IR node scheduling |
| **Overlap quality** | Best-effort stream concurrency | Provably optimal within profiling accuracy |
| **Code complexity** | Hook registration, state machine | 3 lines of user code |
| **Debug path** | Hooks alter execution semantics | Eager mode = compiled mode semantics |

### 11.2 Performance Impact Chain

$$
\text{Bucketing} \xrightarrow{\text{reduces } P\alpha \text{ to } \alpha} \text{Lower base latency} \xrightarrow{\text{enables}} \text{Fewer comm calls}
$$

$$
\text{Reordering} \xrightarrow{\text{prefetches AG/RS}} \text{Overlap with compute} \xrightarrow{\text{reduces}} T_{\text{exposed}} \rightarrow 0
$$

$$
\text{Auto Wrapping} \xrightarrow{\text{profile-driven}} \text{Optimal bucket sizes} \xrightarrow{\text{satisfies}} \text{Time + Memory constraints}
$$

Combined effect on step time:

$$
T_{\text{step}}^{\text{optimized}} \approx \max\left(\sum_k T_{\text{compute}}^k, \; \sum_b T_{\text{comm}}^b\right)
$$

versus unoptimized:

$$
T_{\text{step}}^{\text{naive}} = \sum_k T_{\text{compute}}^k + \sum_i (T_{\text{AG}_i} + T_{\text{RS}_i})
$$

The speedup ratio:

$$
\text{Speedup} = \frac{T_{\text{step}}^{\text{naive}}}{T_{\text{step}}^{\text{optimized}}} = \frac{\sum T_c + \sum T_{\text{comm}}}{\max(\sum T_c, \sum T_{\text{comm}})}
$$

For compute-bound training (typical for large models on fast interconnects), $\sum T_{\text{comm}} < \sum T_c$, and the speedup approaches:

$$
\text{Speedup} \approx 1 + \frac{\sum T_{\text{comm}}}{\sum T_c}
$$

which is typically 1.15–1.35× for well-configured clusters.

---

## 12. End-to-End Deployment Pseudocode

```
ALGORITHM: Complete_SimpleFSDP_Training

// ==================== ENVIRONMENT ====================
INITIALIZE torch.distributed(backend="nccl")  // or "rccl" for AMD
world_size ← GET_WORLD_SIZE()
rank ← GET_RANK()

// ==================== DEVICE MESH ====================
mesh_2d ← DeviceMesh("cuda", shape=(dp_size, tp_size))
dp_mesh ← mesh_2d[0]  // DP dimension
tp_mesh ← mesh_2d[1]  // TP dimension

// ==================== MODEL INIT ====================
WITH meta_device():
    model ← TransformerModel(config)
    // All parameters on meta device (no memory)

// ==================== PARALLELISM ====================
// Apply TP first (innermost dimension)
apply_column_parallel(model.layers.*.mlp.up_proj, tp_mesh)
apply_row_parallel(model.layers.*.mlp.down_proj, tp_mesh)
apply_column_parallel(model.layers.*.attn.qkv_proj, tp_mesh)
apply_row_parallel(model.layers.*.attn.o_proj, tp_mesh)

// Apply activation checkpointing
apply_selective_ac(model, policy=OP_SELECTIVE)

// Materialize parameters from checkpoint
load_sharded_checkpoint(model, checkpoint_path, mesh_2d)

// ==================== SIMPLEFSDP ====================
// Configure SimpleFSDP optimizations
SET torch._inductor.config.simplefsdp.bucket_mode = "auto"
SET torch._inductor.config.simplefsdp.enable_reorder = True

// Wrap model
model ← simple_fsdp(model)

// Compile with full graph
model ← torch.compile(model, fullgraph=True)

// ==================== OPTIMIZER ====================
optimizer ← AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
// Optimizer states are automatically sharded (local shard only)

// ==================== TRAINING LOOP ====================
FOR step = 1 TO max_steps:
    batch ← dataloader.next()
    // Deterministic sampling, resume-safe state
    
    loss ← model(batch.input_ids, batch.labels)
    loss.backward()
    
    // Gradient clipping (on sharded gradients)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    IF step % checkpoint_interval == 0:
        save_sharded_checkpoint(model, optimizer, step)
    
    IF step % log_interval == 0:
        tokens_per_sec ← batch_tokens / step_time
        mfu ← compute_mfu(model_flops, step_time, gpu_peak_flops)
        LOG(step, loss, tokens_per_sec, mfu, peak_memory)
```

---

## Appendix A: Quick Reference — FSDP Communication Pattern

| Training Phase | Collective | Direction | Data | Volume per Rank |
|---------------|-----------|-----------|------|-----------------|
| Forward entry | All-Gather | Shard → Full | Parameters | $\frac{N-1}{N} \cdot 2\Phi_{\text{unit}}$ |
| Forward exit | — (free) | — | Full params | 0 (deallocation) |
| Backward entry | All-Gather | Shard → Full | Parameters | $\frac{N-1}{N} \cdot 2\Phi_{\text{unit}}$ |
| Backward exit | Reduce-Scatter | Full → Shard | Gradients | $\frac{N-1}{N} \cdot 2\Phi_{\text{unit}}$ |
| Optimizer | — (local) | — | Sharded state | 0 (no communication) |

**Total communication volume per step per rank:**

$$
V_{\text{total}} = 3 \cdot \frac{N-1}{N} \cdot 2\Phi \approx 6\Phi \text{ bytes (BF16, large } N\text{)}
$$

---

*This report reflects production-grade understanding of SimpleFSDP's compiler-driven FSDP implementation, suitable for deployment on H100, B200, MI300X, and MI350-class clusters at scales from 8 to 16,384+ GPUs.*