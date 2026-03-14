

# Technical Report: End-to-End Distributed LLM Training — Parallelism Strategy, Memory Planning, Throughput Optimization, and Production Frameworks

**Scope:** This report provides a rigorous, end-to-end methodology for configuring distributed large language model training across GPU clusters. It covers memory fitting, global batch size engineering, throughput optimization, parallelism strategy selection, and production framework configuration (Megatron-Core, DeepSpeed, PyTorch/torchrun, Triton). Every concept is derived from first principles with pseudocode, formulae, cost models, and decision procedures.

---

## Table of Contents

1. [Foundational Memory Model](#1-foundational-memory-model)
2. [Step 1: Fit the Model into GPU Memory](#2-step-1-fit-model-into-memory)
3. [Step 2: Satisfy the Target Global Batch Size](#3-step-2-satisfy-target-global-batch-size)
4. [Step 3: Optimize Training Throughput](#4-step-3-optimize-training-throughput)
5. [Parallelization Strategy Reference — Full Comparative Analysis](#5-parallelization-strategy-reference)
6. [Communication Cost Models and Overlap Analysis](#6-communication-cost-models)
7. [Framework Implementation: Megatron-Core](#7-megatron-core)
8. [Framework Implementation: DeepSpeed](#8-deepspeed)
9. [Launch Orchestration: torchrun and Elastic Recovery](#9-torchrun)
10. [Kernel Optimization: Triton, FlashAttention, Fused Kernels](#10-kernel-optimization)
11. [Topology-Aware Placement and Hardware Mapping](#11-topology-aware-placement)
12. [Long-Context and MoE Special Cases](#12-long-context-moe)
13. [Glossary and Canonical Formulae](#13-glossary)
14. [Decision Flowchart — Pseudocode](#14-decision-flowchart)

---

## 1. Foundational Memory Model

Before any parallelism decision, the engineer must derive the **exact per-GPU memory footprint**. Every byte must be accounted for.

### 1.1 Component Decomposition

For a dense Transformer with `L` layers, hidden dimension `H`, vocabulary `V`, sequence length `S`, micro-batch size `mbs`, and mixed-precision BF16 training with an Adam optimizer:

$$
\text{num\_params} \approx L \times 12H^2 + V \times H
$$

> The factor 12H² per layer accounts for: QKV projection (3H²), output projection (H²), gate projection (H² × FFN\_mult), up projection (H² × FFN\_mult), down projection (H² × FFN\_mult), plus layer norm parameters. For standard 4× FFN (SwiGLU uses 8/3× with two gates), 12H² is the canonical approximation.

| Memory Component | Formula | Bytes per Parameter | Sharding Denominator |
|---|---|---|---|
| **Model weights (BF16)** | `2 × num_params` | 2 B | `tp × pp × dp_if_zero3` |
| **Model master copy (FP32)** | `4 × num_params` | 4 B | `tp × pp × dp_if_zero1` |
| **Gradients (FP32)** | `4 × num_params` | 4 B | `tp × pp × dp_if_zero2` |
| **Optimizer states (Adam FP32)** | `8 × num_params` | 8 B (momentum + variance) | `tp × pp × dp_if_zero1` |
| **Activations** | Complex (see §1.2) | Variable | `tp × cp` (partial for `pp`, `dp_if_zero3`) |
| **Temporary buffers** | ~5–15% overhead | Variable | Not sharded |
| **NCCL/RCCL workspace** | 0.5–2 GB typical | Fixed | Per-GPU |

**Canonical peak memory per GPU:**

$$
M_{\text{peak}} = \underbrace{\frac{2P}{\text{tp} \cdot \text{pp} \cdot d_{\text{z3}}}}_{\text{model\_bf16}} + \underbrace{\frac{4P}{\text{tp} \cdot \text{pp} \cdot d_{\text{z1}}}}_{\text{model\_fp32}} + \underbrace{\frac{4P}{\text{tp} \cdot \text{pp} \cdot d_{\text{z2}}}}_{\text{grads\_fp32}} + \underbrace{\frac{8P}{\text{tp} \cdot \text{pp} \cdot d_{\text{z1}}}}_{\text{optim\_states}} + A_{\text{act}}
$$

Where:
- `P` = total parameter count
- `d_z1`, `d_z2`, `d_z3` = effective DP degree for ZeRO stage 1, 2, 3 respectively (each equals `dp` if that ZeRO stage is enabled, else `1`)
- `A_act` = activation memory (derived below)

### 1.2 Activation Memory

For a single Transformer layer without recomputation, the dominant activation tensors are:

$$
A_{\text{layer}} = S \times \text{mbs} \times H \times \left( 34 + 5 \frac{n_{\text{heads}} \times S}{H} \right) \times \frac{2}{\text{tp}}
$$

> The 34H term aggregates: input to QKV (2 bytes × H), Q/K/V projections, attention output, post-attention residual, FFN gate/up/down intermediates, layer norms, and residual connections. The attention score matrix contributes the S-dependent term.

With **full activation recomputation** (selective recompute of the entire forward pass during backward):

$$
A_{\text{layer}}^{\text{recomp}} = S \times \text{mbs} \times H \times 2 \quad \text{(only input activation stored per layer)}
$$

With **selective activation recomputation** (recompute attention but store FFN):

$$
A_{\text{layer}}^{\text{selective}} \approx S \times \text{mbs} \times H \times \left(10 + 5\frac{n_{\text{heads}} \times S}{H}\right) \times \frac{2}{\text{tp}}
$$

Total activation memory:

$$
A_{\text{act}} = \frac{L}{\text{pp}} \times \frac{A_{\text{layer}}}{\text{cp}} \times \underbrace{\min(\text{gas}, \text{pp})}_{\text{in-flight microbatches}}
$$

> In pipeline parallelism with 1F1B schedule, at most `pp` microbatches are in flight simultaneously, each holding activations for its assigned stages.

### 1.3 Memory Budget Validation Pseudocode

```
PROCEDURE ValidateMemoryFit(model_config, parallel_config, hw_config):
    P = ComputeParamCount(model_config)
    tp, pp, dp, cp, ep = parallel_config
    zero_stage = parallel_config.zero_stage
    HBM_capacity = hw_config.hbm_bytes       // e.g., 80 GB for A100, 96 GB for H100

    // Determine ZeRO sharding denominators
    d_z1 = dp IF zero_stage >= 1 ELSE 1
    d_z2 = dp IF zero_stage >= 2 ELSE 1
    d_z3 = dp IF zero_stage >= 3 ELSE 1

    // Model state memory
    model_bf16   = (2 * P) / (tp * pp * d_z3)
    model_fp32   = (4 * P) / (tp * pp * d_z1)
    grads_fp32   = (4 * P) / (tp * pp * d_z2)
    optim_states = (8 * P) / (tp * pp * d_z1)

    // Activation memory
    IF full_recomputation:
        A_per_layer = S * mbs * H * 2
    ELSE IF selective_recomputation:
        A_per_layer = ComputeSelectiveActivationMem(model_config, tp)
    ELSE:
        A_per_layer = ComputeFullActivationMem(model_config, tp)

    layers_per_stage = L / pp
    inflight_mbs = MIN(gas, pp)    // 1F1B steady-state
    A_act = layers_per_stage * (A_per_layer / cp) * inflight_mbs

    // Buffers and workspace
    nccl_workspace = 1.5 GB
    fragmentation_overhead = 0.10 * HBM_capacity

    M_peak = model_bf16 + model_fp32 + grads_fp32 + optim_states
             + A_act + nccl_workspace + fragmentation_overhead

    IF M_peak > 0.95 * HBM_capacity:
        RETURN FAIL, M_peak, "Exceeds HBM budget"
    ELSE:
        RETURN PASS, M_peak, "Fits within HBM budget"
```

---

## 2. Step 1: Fit the Model into GPU Memory

The first and non-negotiable objective is ensuring the model's training state fits within the HBM of each GPU. The strategy differs dramatically based on model scale and GPU availability.

### 2.1 GPU-Rich Regime

> **Principle:** When sufficient GPUs are available, select the minimal parallelism that fits the model, then scale out for throughput.

#### 2.1.1 Small Models (< 10B Parameters)

For models under ~10B parameters (e.g., LLaMA-7B: ~28 GB in BF16 model states), a **single parallelism dimension** across 8 GPUs within one node is sufficient:

**Option A: Tensor Parallelism (TP=8)**
- Splits all weight matrices across 8 GPUs within a single NVLink/NVSwitch domain
- Reduces per-GPU model state by 8×
- Communication: 4 all-reduce (or all-gather + reduce-scatter) operations per layer, constrained to intra-node NVLink bandwidth (~900 GB/s bidirectional on H100 NVSwitch)
- **Best when:** Intra-node bandwidth is abundant (NVSwitch), sequence lengths are moderate

**Option B: ZeRO-3 / FSDP (DP=8)**
- Shards all model states (weights, gradients, optimizer) across 8 GPUs
- Each GPU holds `1/8` of all state, reconstructed via all-gather before each layer's forward/backward
- Communication: all-gather before forward, all-gather before backward, reduce-scatter after backward — **per layer**
- **Best when:** Simplicity is preferred, no PP schedule complexity, single-node

**Option C: Full Activation Recomputation Only**
- If the model fits in weight memory but activation memory overflows, enable full recomputation (also called "gradient checkpointing")
- Trades ~33% additional compute for ~90% activation memory reduction
- No additional communication

```
PROCEDURE FitSmallModel(num_params, num_gpus=8, HBM=80GB):
    base_state = 18 * num_params    // 2 + 4 + 4 + 8 bytes per param
    per_gpu_state = base_state / 8

    IF per_gpu_state + activation_estimate < 0.9 * HBM:
        SELECT TP=8 (if NVSwitch available) OR ZeRO-3 DP=8
    ELSE:
        ENABLE full_activation_recomputation
        SELECT TP=8 OR ZeRO-3 DP=8
```

#### 2.1.2 Large Models (10B–100B+ Parameters)

Models exceeding ~10B parameters require more than 8 GPUs for state alone. Multiple parallelism dimensions must be combined.

**Configuration A: TP=8 + PP (Pipeline Parallelism)**
- TP within each 8-GPU NVSwitch domain (one node)
- PP across nodes via InfiniBand/RoCE
- Each pipeline stage holds `L/pp` layers, sharded by TP within the node
- Communication pattern: TP uses NVLink (high bandwidth, low latency), PP uses IB (point-to-point send/recv of activation tensors only)
- **Advantage:** PP communication volume is minimal (single activation tensor per microbatch boundary), making it ideal for inter-node links

**Configuration B: TP=8 + ZeRO-3 DP**
- TP within the node, ZeRO-3 sharding across nodes
- All model states partitioned across DP ranks, reconstructed per layer
- Communication pattern: ZeRO-3 requires all-gather/reduce-scatter of full model shards across nodes
- **Advantage:** Simpler scheduling (no pipeline bubbles), all GPUs process the same layers
- **Disadvantage:** Higher inter-node communication volume than PP; at scale (>64 nodes), the all-gather latency across the full DP group becomes prohibitive

**Configuration C: Pure ZeRO-3**
- No TP, no PP — only data-parallel sharding
- Every GPU participates in all-gather/reduce-scatter for every layer
- **Advantage:** Simplest configuration, no TP communication within layers
- **Disadvantage:** At scale, the per-layer all-gather across hundreds of nodes introduces significant latency; also, each GPU must execute the full model graph (no compute reduction)

```
PROCEDURE FitLargeModel(num_params, total_gpus, gpus_per_node=8, HBM):
    // Phase 1: Set TP within node
    tp = 8    // Fill one NVSwitch domain
    remaining_gpus = total_gpus / tp

    // Phase 2: Choose inter-node strategy
    state_per_gpu_tp = 18 * num_params / tp
    IF state_per_gpu_tp + max_activations > 0.9 * HBM:
        // Still too large: need PP to split layers
        pp = CEIL(state_per_gpu_tp / (0.7 * HBM))
        dp = total_gpus / (tp * pp)
    ELSE:
        // Fits with TP alone: use DP for throughput
        pp = 1
        dp = remaining_gpus

    // Phase 3: Select ZeRO stage for DP
    optim_per_dp_gpu = (8 * num_params) / (tp * pp)
    IF optim_per_dp_gpu > 0.3 * HBM:
        zero_stage = 1    // Shard optimizer states across DP
    IF optim_per_dp_gpu + grads_per_gpu > 0.4 * HBM:
        zero_stage = 2    // Also shard gradients
    // ZeRO-3 only if still memory-constrained

    RETURN {tp, pp, dp, zero_stage}
```

#### 2.1.3 Scale-Specific Guidance

| GPU Scale | Recommended Configuration | Rationale |
|---|---|---|
| **≤ 8 GPUs** (1 node) | TP=8 or ZeRO-3 DP=8 | Single NVSwitch domain; TP communication is free relative to compute |
| **16–64 GPUs** (2–8 nodes) | TP=8, DP=2–8 (ZeRO-1 or ZeRO-2) | DP all-reduce across few nodes is manageable; pipeline bubbles not yet justified |
| **64–512 GPUs** | TP=8, PP=2–4, DP with ZeRO-1 | PP reduces inter-node DP all-reduce volume; pipeline bubble < 15% with adequate GAS |
| **512–1024 GPUs** | TP=8, PP=4–8, DP with ZeRO-2 | Pure DP/ZeRO-3 becomes communication-bound; PP keeps inter-node traffic minimal |
| **1024+ GPUs** | TP=8, PP=4–16, DP with ZeRO-2, optional CP | 4D parallelism mandatory; ZeRO-2 reduces gradient memory; PP minimizes cross-node traffic |
| **Long context** (≥64K seq) | Add CP=2–8 within TP domain or across nodes | Context parallelism splits sequence; Ring Attention overlaps communication with attention compute |
| **MoE architectures** | Add EP=num_experts/experts_per_gpu | Expert parallelism distributes experts; all-to-all routes tokens; place EP within high-bandwidth domain |

### 2.2 GPU-Poor Regime

When GPU count is severely limited (1–4 GPUs, or single node for a 70B+ model), aggressive memory reduction techniques are mandatory:

**Strategy 1: Full Activation Recomputation**
- Discard all intermediate activations during forward pass
- Recompute them layer-by-layer during backward pass
- Memory reduction: ~60–90% of activation memory eliminated
- Compute overhead: ~33% additional FLOPs (one extra forward pass)

**Strategy 2: Gradient Accumulation**
- Reduce micro-batch size to minimum (mbs=1), accumulate gradients over multiple forward-backward passes
- Activation memory scales linearly with mbs; reducing mbs directly reduces activation footprint
- No additional communication until the accumulation boundary

**Strategy 3: CPU/NVMe Offload (ZeRO-Infinity)**
- Offload optimizer states, gradients, and optionally parameters to CPU DRAM or NVMe
- Reconstructed to GPU on demand via PCIe/CXL
- Throughput reduction: significant (PCIe bandwidth ~64 GB/s vs. HBM ~3.35 TB/s on H100)
- **Use only** when model cannot fit even with full recomputation

**Strategy 4: Mixed-Precision Reduction**
- Use FP8 forward pass (Transformer Engine on H100/B200) to halve activation memory relative to BF16
- Requires FP8-aware kernels and careful loss scaling

```
PROCEDURE FitGPUPoor(num_params, num_gpus, HBM):
    // Start with most aggressive memory reduction
    ENABLE full_activation_recomputation
    SET mbs = 1
    SET gas = target_GBS / (mbs * num_gpus)

    // Try TP if multi-GPU within node
    IF num_gpus > 1 AND num_gpus <= 8:
        tp = num_gpus
    ELSE:
        tp = 1

    state_per_gpu = (18 * num_params) / tp
    act_per_gpu = layers_per_gpu * S * mbs * H * 2  // recomp: minimal

    IF state_per_gpu + act_per_gpu > 0.9 * HBM:
        ENABLE ZeRO-3 across available GPUs
        state_per_gpu = (18 * num_params) / (tp * num_gpus)

    IF STILL exceeds HBM:
        ENABLE cpu_offload for optimizer states
        ENABLE nvme_offload if cpu_ram insufficient

    RETURN configuration
```

---

## 3. Step 2: Satisfy the Target Global Batch Size

### 3.1 The GBS Constraint

Empirical scaling laws and hyperparameter sweeps establish an optimal **global batch size (GBS)** measured in tokens. For large language models, this is typically in the range:

$$
\text{GBS}_{\text{tokens}} \in [4\text{M}, 40\text{M}] \text{ tokens}
$$

> Chinchilla-optimal training and subsequent work (LLaMA, GPT-4 technical reports) establish that the ideal GBS depends on learning rate, model size, and training stage. Many practitioners use ~4M tokens for warmup, ramping to 8–16M for stable training, potentially reaching 32–40M for very large models.

The relationship between GBS in tokens and parallelism configuration:

$$
\text{GBS}_{\text{tokens}} = \text{mbs} \times S_{\text{effective}} \times \text{dp} \times \text{gas}
$$

Where:
- `mbs` = micro-batch size (sequences per GPU per microbatch)
- `S_effective` = effective sequence length per sample (after packing or CP splitting)
- `dp` = data parallelism degree (includes ZeRO-sharded DP)
- `gas` = gradient accumulation steps

> **Critical note:** Context parallelism (CP) does **not** multiply the GBS. CP splits a single sequence across CP ranks — it does not add independent samples. However, CP enables processing longer sequences that would not otherwise fit, which increases `S_effective` per sample.

### 3.2 Adjusting GBS Upward

If the configuration from Step 1 produces a GBS below the target:

**Method 1: Increase Gradient Accumulation Steps (GAS)**
- No additional GPUs required
- Each GPU performs `gas` sequential forward-backward passes before the optimizer step
- Increases step time linearly with `gas` but does not increase memory (gradients accumulated in-place)
- **Drawback:** Reduces GPU utilization (sequential microbatches cannot be overlapped with communication in pure DP mode)

**Method 2: Increase Data Parallelism (DP)**
- Requires additional GPUs
- Each additional DP rank processes an independent microbatch simultaneously
- Communication: gradient synchronization (all-reduce or reduce-scatter) at the DP boundary
- **Effect on GBS:** Linear increase — doubling DP doubles GBS

**Method 3: Increase Context Parallelism (CP)**
- Applicable when training with very long sequences
- CP allows processing a longer sequence per sample, which increases `S_effective` and thus tokens-per-sample
- **Effect on GBS:** If CP enables doubling the sequence length per sample, `GBS_tokens` doubles (assuming sample count stays constant)

### 3.3 Adjusting GBS Downward

If the configuration from Step 1 produces a GBS above the target:

**Method 1: Reduce Data Parallelism**
- Reallocate GPUs from DP to TP or PP
- Example: Convert from TP=8, DP=16 (GBS = 16 × mbs × S) to TP=8, PP=2, DP=8 (GBS = 8 × mbs × S)

**Method 2: Reduce Gradient Accumulation**
- Set `gas = 1` (minimum)

**Method 3: Reduce Micro-Batch Size**
- Reduces tokens per GPU per microbatch
- Minimum practical value: mbs=1
- **Caution:** Very small mbs degrades compute efficiency (kernel launch overhead dominates, poor GPU occupancy)

```
PROCEDURE SatisfyGBS(target_GBS_tokens, current_config, S):
    current_GBS = current_config.mbs * S * current_config.dp * current_config.gas

    IF current_GBS < target_GBS_tokens:
        // Need to increase GBS
        deficit_ratio = target_GBS_tokens / current_GBS

        // Priority 1: Increase GAS (cheapest, no hardware change)
        max_gas = target_GBS_tokens / (current_config.mbs * S * current_config.dp)
        IF max_gas <= MAX_ACCEPTABLE_GAS:   // e.g., 64
            SET gas = max_gas
        ELSE:
            // Priority 2: Increase DP (need more GPUs)
            required_dp = target_GBS_tokens / (current_config.mbs * S * gas)
            REALLOCATE GPUs: reduce PP or add nodes to increase DP
            SET dp = required_dp

    ELSE IF current_GBS > target_GBS_tokens:
        // Need to decrease GBS
        // Priority 1: Reduce GAS to 1
        SET gas = 1
        new_GBS = current_config.mbs * S * current_config.dp

        IF new_GBS > target_GBS_tokens:
            // Priority 2: Reduce DP, reallocate to TP or PP
            required_dp = target_GBS_tokens / (current_config.mbs * S * 1)
            freed_gpus = (current_config.dp - required_dp) * current_config.tp
            REALLOCATE freed_gpus to PP stages
            SET dp = required_dp

    VALIDATE: mbs * S * dp * gas == target_GBS_tokens
    RETURN updated_config
```

### 3.4 GBS Engineering Decision Table

| Situation | Action | Side Effect |
|---|---|---|
| GBS too low, have spare GPUs | Increase DP | More inter-node communication |
| GBS too low, no spare GPUs | Increase GAS | Longer step time, no memory impact |
| GBS too low, long-context scenario | Increase CP (enables longer S) | Requires Ring Attention / Ulysses, more communication |
| GBS too high | Reduce DP, reallocate to TP/PP | May improve MFU if TP reduces cross-node comm |
| GBS too high, already DP=1 | Reduce mbs | May underutilize GPU compute |

---

## 4. Step 3: Optimize Training Throughput

> **Principle:** After achieving memory fit and correct GBS, the remaining task is maximizing **tokens per second per GPU** (equivalently, maximizing MFU — Model FLOPs Utilization). There is no universal recipe; optimization is empirical, guided by cost models.

### 4.1 Throughput Optimization Methodology

```
PROCEDURE OptimizeThroughput(base_config, cluster_topology, target_MFU):
    best_config = base_config
    best_throughput = MeasureThroughput(base_config)

    // Experiment 1: Scale TP within node
    FOR tp IN [1, 2, 4, 8]:
        config = AdjustTP(base_config, tp)
        IF ValidateMemoryFit(config) AND SatisfyGBS(config):
            throughput = MeasureThroughput(config)
            IF throughput > best_throughput:
                best_config = config
                best_throughput = throughput

    // Experiment 2: Trade PP for DP
    FOR pp IN [1, 2, 4, 8, 16]:
        config = AdjustPP(best_config, pp)
        IF ValidateMemoryFit(config) AND SatisfyGBS(config):
            throughput = MeasureThroughput(config)
            IF throughput > best_throughput AND PipelineBubble(config) < 0.15:
                best_config = config
                best_throughput = throughput

    // Experiment 3: Vary micro-batch size
    FOR mbs IN [1, 2, 4, 8, 16]:
        config = AdjustMBS(best_config, mbs)
        IF ValidateMemoryFit(config) AND SatisfyGBS(config):
            throughput = MeasureThroughput(config)
            IF throughput > best_throughput:
                best_config = config
                best_throughput = throughput

    // Experiment 4: ZeRO stage selection
    FOR zero_stage IN [0, 1, 2, 3]:
        config = AdjustZeRO(best_config, zero_stage)
        IF ValidateMemoryFit(config):
            throughput = MeasureThroughput(config)
            // ZeRO-3 trades communication for memory; only use if needed
            IF throughput > best_throughput:
                best_config = config
                best_throughput = throughput

    // Experiment 5: Enable communication-computation overlap
    ENABLE async_allreduce, ENABLE overlap_p2p_with_compute
    throughput = MeasureThroughput(best_config)

    COMPUTE MFU = measured_FLOPS / (num_gpus * peak_FLOPS)
    IF MFU < target_MFU:
        PROFILE step_time_decomposition
        IDENTIFY bottleneck: {kernel, communication, dataloader, launch_overhead}
        ADDRESS bottleneck with targeted optimization

    RETURN best_config, MFU
```

### 4.2 Key Optimization Levers

#### 4.2.1 Tensor Parallelism Scaling

**When to increase TP:**
- Intra-node bandwidth is underutilized
- DP all-reduce across nodes is the bottleneck
- Increasing TP reduces the per-GPU compute, allowing smaller mbs or freeing memory for larger mbs

**When TP hurts:**
- TP=8 on GPUs connected only via PCIe (not NVLink): the all-reduce per layer will bottleneck on PCIe bandwidth (~64 GB/s vs. ~900 GB/s NVSwitch)
- Very small models where TP splits matrices below efficient GEMM tile sizes

**TP communication volume per layer (Megatron-style column/row parallel):**

$$
V_{\text{TP, per layer}} = 4 \times 2 \times S \times \text{mbs} \times H \times \frac{\text{tp} - 1}{\text{tp}} \quad \text{bytes}
$$

> 4 communication points per layer (2 in attention, 2 in FFN), each transferring activations of size `S × mbs × H` in BF16 (2 bytes). The `(tp-1)/tp` factor reflects the reduce-scatter/all-gather volume.

**TP compute-communication overlap condition:**

$$
\frac{(\text{tp} - 1) \times \text{peak\_FLOPS}}{24 \times H \times \text{BW}_{\text{NVLink}}} < 1.0
$$

If this ratio is ≪ 1, communication is fully hidden behind compute. For H100 SXM (989 TFLOPS BF16, 900 GB/s NVSwitch):

$$
\frac{7 \times 989 \times 10^{12}}{24 \times 8192 \times 900 \times 10^9} \approx 39.2
$$

This indicates that for H=8192, TP=8 on H100 NVSwitch, communication is **not** fully hidden — there is meaningful TP overhead. This is why TP=8 does not achieve 100% MFU; the irreducible TP communication cost is significant.

#### 4.2.2 Pipeline Parallelism Tuning

**Pipeline bubble fraction (1F1B schedule):**

$$
\text{Bubble}_{\text{1F1B}} = \frac{\text{pp} - 1}{\text{pp} - 1 + \text{gas}} = \frac{\text{pp} - 1}{\text{mbs\_total}}
$$

Where `mbs_total = gas` (number of microbatches per pipeline flush).

| PP Degree | GAS Required for ≤10% Bubble | GAS Required for ≤5% Bubble |
|---|---|---|
| 2 | 10 | 20 |
| 4 | 27 | 57 |
| 8 | 63 | 133 |
| 16 | 135 | 285 |

> **Interleaved schedules** (Megatron-LM virtual pipeline stages) reduce bubble fraction by a factor of `v` (number of virtual stages per physical stage): `bubble = (pp-1) / (v × gas + pp - 1)`. This is critical at PP≥8.

**PP communication volume per microbatch:**

$$
V_{\text{PP, per\_microbatch}} = 2 \times S \times \text{mbs} \times H \times 2 \quad \text{bytes (forward + backward)}
$$

This is a **point-to-point** transfer, not a collective. At H=8192, mbs=1, S=4096:

$$
V_{\text{PP}} = 2 \times 4096 \times 1 \times 8192 \times 2 = 128 \text{ MB per microbatch}
$$

On InfiniBand HDR (200 Gb/s = 25 GB/s unidirectional), this takes ~5.1 ms — typically overlapped with the next microbatch's computation.

#### 4.2.3 ZeRO Stage Selection

| ZeRO Stage | What is Sharded | Communication Volume (per step) | Overlap Potential |
|---|---|---|---|
| **ZeRO-0** (vanilla DP) | Nothing | `2P` bytes (all-reduce gradients in BF16) | Overlapped with backward via bucketed all-reduce |
| **ZeRO-1** | Optimizer states | `2P` bytes (all-reduce gradients) + `4P/dp` bytes (all-gather FP32 params at step end) | Gradient all-reduce overlaps backward; param all-gather is serial |
| **ZeRO-2** | Optimizer states + gradients | `P` bytes (reduce-scatter gradients) + `4P/dp` bytes (all-gather FP32 params) | Reduce-scatter volume halved vs. all-reduce; better overlap |
| **ZeRO-3** (FSDP) | Everything | Per layer: all-gather params (forward) + all-gather params (backward) + reduce-scatter grads = `3 × 2P_layer / dp` per layer | Overlapped with next layer's compute; high communication frequency |

**Recommendation hierarchy:**
1. **ZeRO-1** as default for multi-node DP (minimal overhead, significant optimizer memory savings)
2. **ZeRO-2** when gradient memory is the binding constraint and reduce-scatter can overlap
3. **ZeRO-3/FSDP** only when model states cannot fit even with ZeRO-2; communication overhead is substantial at scale

#### 4.2.4 Micro-Batch Size Optimization

The micro-batch size controls the tradeoff between:
- **Compute efficiency:** Larger mbs → larger GEMMs → higher arithmetic intensity → higher GPU utilization
- **Memory pressure:** Larger mbs → more activation memory → may require more recomputation
- **Pipeline bubble:** Larger mbs → fewer microbatches per step (for fixed GBS) → larger bubble fraction
- **GBS constraint:** `mbs × dp × gas = GBS / S`

```
PROCEDURE OptimizeMBS(config, HBM_budget):
    // Start with largest mbs that fits in memory
    mbs_max = BinarySearchMaxMBS(config, HBM_budget)

    // Ensure GBS constraint is satisfiable
    FOR mbs IN [mbs_max, mbs_max/2, mbs_max/4, ..., 1]:
        gas = GBS / (mbs * S * dp)
        IF gas IS integer AND gas >= 1:
            bubble = (pp - 1) / (gas + pp - 1)
            IF bubble < 0.10:
                throughput = ProfileThroughput(config, mbs, gas)
                RECORD (mbs, gas, throughput, bubble)

    RETURN config with highest throughput
```

### 4.3 Compute-Communication Overlap — The Throughput Multiplier

The single most impactful throughput optimization is ensuring that communication operations execute concurrently with compute kernels on separate hardware units (NVLink/IB NICs vs. SMs).

**Overlap opportunities by parallelism dimension:**

| Parallelism | Communication | Overlapped With | Overlap Condition |
|---|---|---|---|
| DP (vanilla) | All-reduce gradients | Next microbatch's backward pass | Multiple microbatches (GAS > 1) |
| DP + ZeRO-2 | Reduce-scatter gradients | Current microbatch's backward (bucketed) | Bucket size tuning; backward compute > comm per bucket |
| ZeRO-3/FSDP | All-gather params (fwd/bwd), reduce-scatter grads | Next layer's forward/backward | Per-layer granularity; compute per layer > comm per layer |
| TP | All-reduce / reduce-scatter+all-gather activations | Next TP region (attention ↔ FFN) | TP region compute > TP collective time |
| PP | Send/recv activations and gradients | Next microbatch's forward/backward | Microbatch compute time > P2P transfer time |
| CP (Ring Attention) | Send/recv KV blocks | Current attention chunk computation | Attention compute per chunk > KV transfer time |
| EP | All-to-all token dispatch/combine | MoE FFN computation | Expert compute > all-to-all latency |

---

## 5. Parallelization Strategy Reference — Full Comparative Analysis

### 5.1 Comprehensive Strategy Table

| Strategy | Batch Size Effect | Memory Reduction | Compute Reduction | Communication Pattern | Compute/Communication Overlap Condition |
|---|---|---|---|---|---|
| **Data Parallelism (DP)** | GBS scales linearly with DP | Can reduce mbs (thus activations) by increasing DP | Can reduce mbs by increasing DP | **Backward:** All-reduce `grads_bf16` (size: `2P` bytes) | Overlapped with microbatch backward. Ratio: `(DP−1) × peak_FLOPS / (2 × BW_inter × num_tokens_per_gpu × DP)` |
| **DP + ZeRO-1** | GBS scales with DP | `model_fp32 / dp`, `optim_states / dp` | Same as DP | **Backward:** All-reduce `grads_bf16`. **Step end:** All-gather `model_fp32` | Same overlap as DP; param all-gather adds serial latency at step boundary |
| **DP + ZeRO-2** | GBS scales with DP | `model_fp32 / dp`, `grads_fp32 / dp`, `optim_states / dp` | Same as DP | **Backward:** Reduce-scatter `grads_bf16`. **Step end:** All-gather `model_fp32` | Reduce-scatter volume = `P` bytes (half of all-reduce). Overlap ratio: `(DP−1) × peak_FLOPS / (4 × BW × num_tokens × DP)` |
| **DP + ZeRO-3 (FSDP)** | GBS scales with DP | `model_bf16/dp`, `model_fp32/dp`, `grads_fp32/dp`, `optim_states/dp` | Same as DP | **Per layer (×L):** Fwd all-gather `model_fp32`, Bwd all-gather `model_fp32`, Bwd reduce-scatter `grads_fp32` | Overlapped with next layer fwd/bwd. Ratio: `(DP−1) × peak_FLOPS / (2 × S × mbs × BW)` |
| **Tensor Parallelism (TP)** | No effect on GBS | `model_bf16/tp`, `model_fp32/tp`, `grads_fp32/tp`, `optim_states/tp`, `activations/tp` | `compute/tp` per GPU | **Per layer (×4 comm points):** Fwd/Bwd all-gather + reduce-scatter `activs_bf16` | Overlapped with next TP region. Ratio: `(TP−1) × peak_FLOPS / (24 × H × BW_NVLink)` |
| **Pipeline Parallelism (1F1B)** | Prefers large GAS to minimize bubble | `model_bf16/pp`, `model_fp32/pp`, `grads_fp32/pp`, `optim_states/pp` | `compute/pp` per GPU | **Per microbatch (×GAS):** Fwd send/recv `activs_bf16`, Bwd send/recv `grads_bf16` | Overlapped with next microbatch fwd/bwd. Ratio: `PP × peak_FLOPS / (32 × H × L × BW_IB)` |
| **Context Parallelism (CP)** | Prefers large S for overlap | `activations / cp` | `attention_compute / cp` | **Per layer (×CP−1):** Fwd/Bwd send/recv KV blocks `activs_bf16` | Overlapped with attention computation (Ring Attention). Volume: `(CP−1) × H × (L/CP) × H_kv × (num_k + num_v)` |
| **Expert Parallelism (EP)** | GBS scales with EP | `expert_params / ep` | `expert_compute / ep` | **Per MoE layer:** Fwd all-to-all `activs_bf16`, Bwd all-to-all `grads_bf16` | Overlapped with MoE FFN block. Ratio: `(EP−1) × peak_FLOPS / (12 × num_experts × H × BW)` |

### 5.2 Parallelism Interaction Constraints

The total GPU count (`world_size`) must factorize exactly:

$$
\text{world\_size} = \text{tp} \times \text{pp} \times \text{dp} \times \text{cp} \times \text{ep}
$$

> **Important constraint for MoE:** EP is typically orthogonal to DP. If the model has `E` total experts and each GPU holds `E/ep` experts, then the DP degree for non-expert parameters is `dp = world_size / (tp × pp × cp × ep)`. Expert parameters only replicate across DP (non-EP) ranks.

**Topology placement rules:**
1. **TP** must be within a single NVSwitch domain (≤8 GPUs on DGX H100/A100, ≤8 on MI300X with xGMI)
2. **PP** prefers inter-node links (InfiniBand, RoCE) — minimal volume, latency-tolerant with overlap
3. **DP** can span any topology; ZeRO-3 communication is volume-heavy and benefits from high-bandwidth interconnect
4. **CP** can be intra-node (high bandwidth) or inter-node (if Ring Attention overlap is sufficient)
5. **EP** benefits from high-bandwidth interconnect (all-to-all is bandwidth-intensive); prefer intra-node or NVSwitch-connected GPUs

---

## 6. Communication Cost Models and Overlap Analysis

### 6.1 Collective Operation Cost Models

For a message of size `M` bytes across `N` participants with per-link bandwidth `BW` and latency `α`:

| Collective | Algorithm | Time Model |
|---|---|---|
| **All-reduce** | Ring | `2α(N−1) + 2M(N−1)/(N×BW)` |
| **All-reduce** | Tree (hierarchical) | `2α⌈log₂N⌉ + 2M/BW` (for large M, amortized) |
| **Reduce-scatter** | Ring | `α(N−1) + M(N−1)/(N×BW)` |
| **All-gather** | Ring | `α(N−1) + M(N−1)/(N×BW)` |
| **All-to-all** | Direct | `α(N−1) + M(N−1)/(N×BW)` (bisection BW limited) |
| **Send/recv** (P2P) | Direct | `α + M/BW` |

### 6.2 Per-Step Communication Volume Summary

For a model with `P` parameters, `L` layers, sequence length `S`, micro-batch size `mbs`, hidden dimension `H`:

**DP (vanilla all-reduce):**

$$
V_{\text{DP}} = 2 \times 2P \times \frac{\text{dp} - 1}{\text{dp}} \quad \text{bytes (ring all-reduce)}
$$

**TP (per step, all layers):**

$$
V_{\text{TP}} = 4L \times 2 \times S \times \text{mbs} \times H \times 2 \times \frac{\text{tp} - 1}{\text{tp}} \times \text{gas} \quad \text{bytes}
$$

**PP (per step, all microbatches):**

$$
V_{\text{PP}} = 2 \times \text{gas} \times S \times \text{mbs} \times H \times 2 \quad \text{bytes (fwd + bwd activations)}
$$

**ZeRO-3 (per step):**

$$
V_{\text{ZeRO3}} = 3 \times L \times \frac{2P_{\text{layer}}}{\text{dp}} \times (\text{dp} - 1) \times \text{gas} \quad \text{bytes}
$$

> Note: ZeRO-3 communicates per layer, per microbatch, making it the highest-volume strategy at scale.

### 6.3 Bandwidth Collapse Diagnosis

```
PROCEDURE DiagnoseBandwidthCollapse(profile_data):
    // Step 1: Measure achieved bandwidth per collective
    FOR EACH collective IN profile_data.nccl_traces:
        achieved_bw = collective.bytes / collective.duration
        theoretical_bw = GetLinkBandwidth(collective.src, collective.dst)
        efficiency = achieved_bw / theoretical_bw

        IF efficiency < 0.70:
            FLAG "Low bandwidth efficiency" ON collective
            INVESTIGATE:
                - PCIe bottleneck (GPU not directly NVLink-connected)
                - NCCL topology detection failure (NCCL_TOPO_FILE)
                - InfiniBand port congestion or routing imbalance
                - Message size too small (latency-dominated)
                - Overlap conflict (compute and comm on same SMs)

    // Step 2: Check for stragglers
    step_times = COLLECT per_rank_step_time
    straggler_threshold = MEAN(step_times) + 2 * STDDEV(step_times)
    FOR EACH rank WHERE step_time > straggler_threshold:
        FLAG "Straggler detected" ON rank
        INVESTIGATE:
            - Thermal throttling (nvidia-smi -q -d PERFORMANCE)
            - ECC errors (nvidia-smi -q -d ECC)
            - Network link degradation (ibstat, perfquery)
            - Dataloader stall (CPU bottleneck, storage latency)
            - Imbalanced expert routing (MoE)

    // Step 3: Check for deadlocks
    IF any rank blocked > 10 × expected_collective_time:
        DUMP NCCL debug logs (NCCL_DEBUG=INFO)
        CHECK process group initialization order
        CHECK for mismatched collective calls across ranks
        CHECK for CUDA stream synchronization issues
```

---

## 7. Framework Implementation: Megatron-Core

### 7.1 Architecture Overview

Megatron-Core (the refactored core of Megatron-LM) provides production-grade implementations of:
- **Tensor Parallelism:** Column-parallel and row-parallel linear layers with async all-reduce
- **Pipeline Parallelism:** 1F1B and interleaved (virtual pipeline) schedules
- **Sequence Parallelism (SP):** Splits LayerNorm and Dropout along the sequence dimension within TP groups (reduces activation memory by `tp`)
- **Context Parallelism (CP):** Ring Attention and Ulysses-style sequence splitting
- **Expert Parallelism (EP):** GroupedMLP with token routing and all-to-all dispatch
- **Distributed Optimizer:** Shards optimizer states across DP ranks (equivalent to ZeRO-1)

### 7.2 Parallelism Configuration

```
PROCEDURE ConfigureMegatronCore(model_config, cluster):
    // World size factorization
    ASSERT world_size == tp * pp * dp * cp * ep

    // Tensor Parallelism
    SET --tensor-model-parallel-size = tp      // Must divide num_attention_heads
    SET --sequence-parallel                    // Enable SP (pairs with TP)

    // Pipeline Parallelism
    SET --pipeline-model-parallel-size = pp    // Must divide num_layers
    SET --num-layers-per-virtual-pipeline-stage = V  // Interleaved schedule
    // V = num_layers / (pp * num_virtual_stages)
    // Interleaved reduces bubble by factor of num_virtual_stages

    // Context Parallelism
    SET --context-parallel-size = cp
    // Requires: sequence_length divisible by cp
    // Implementation: Ring Attention (send/recv KV blocks)

    // Expert Parallelism (MoE)
    SET --expert-model-parallel-size = ep
    SET --num-experts = E
    SET --moe-grouped-gemm                    // Fused grouped GEMM for experts
    SET --moe-token-dispatcher-type = alltoall // or allgather

    // Data Parallelism (implicit: dp = world_size / (tp * pp * cp * ep))
    SET --use-distributed-optimizer           // ZeRO-1 equivalent
    SET --overlap-grad-reduce                 // Overlap gradient reduce-scatter with backward
    SET --overlap-param-gather               // Overlap param all-gather with forward

    // Activation Checkpointing
    SET --recompute-granularity = {full, selective}
    SET --recompute-method = {uniform, block}
    SET --recompute-num-layers = K            // Recompute K layers per stage

    // Mixed Precision
    SET --bf16                                // BF16 training
    SET --fp8-format = {e4m3, hybrid}         // FP8 on H100/B200
    SET --fp8-amax-history-len = 1024
    SET --fp8-amax-compute-algo = max

    // Micro-batch and GBS
    SET --micro-batch-size = mbs
    SET --global-batch-size = GBS
    // gas is computed: gas = GBS / (mbs * dp)

    // Checkpoint
    SET --save-interval = N
    SET --load = /path/to/checkpoint
    // Megatron-Core handles TP/PP resharding on load
```

### 7.3 Megatron-Core Distributed Optimizer

The distributed optimizer in Megatron-Core shards optimizer states across DP ranks:

```
PROCEDURE MegatronDistributedOptimizer:
    // During initialization:
    FOR EACH parameter_group:
        SHARD parameters across dp_ranks
        EACH rank owns a contiguous slice of the flattened parameter buffer
        EACH rank maintains FP32 master weights + Adam states for its slice only

    // During backward:
    FOR EACH gradient bucket (bucketed by size):
        REDUCE-SCATTER gradients across dp_ranks
        // Each rank receives its owned gradient slice in FP32

    // During optimizer step:
    EACH rank updates its owned FP32 parameter slice
    ALL-GATHER updated FP32 parameters → reconstruct full model
    CAST FP32 → BF16 for forward pass weights

    // Memory per rank:
    // model_bf16: full (for forward/backward compute)
    // model_fp32: P / dp (owned slice)
    // grads_fp32: P / dp (owned slice after reduce-scatter)
    // optim_states: 2P / dp (momentum + variance for owned slice)
    // Total state: 2P + (4P + 4P + 8P) / dp bytes per rank
```

### 7.4 Checkpoint Interoperability

Megatron-Core checkpoints encode the parallelism topology. Converting between topologies requires resharding:

```
PROCEDURE ReshardCheckpoint(src_ckpt, src_config, dst_config):
    // Load with source topology
    LOAD model state_dict from src_ckpt with tp=src_tp, pp=src_pp

    // TP resharding: merge or split weight matrices
    IF dst_tp != src_tp:
        FOR EACH tensor-parallel weight (column_parallel, row_parallel):
            IF dst_tp > src_tp:
                SPLIT weight along partition dimension by factor dst_tp/src_tp
            ELSE:
                CONCATENATE weight shards along partition dimension

    // PP resharding: reassign layers to stages
    IF dst_pp != src_pp:
        RECOMPUTE layer-to-stage mapping
        REDISTRIBUTE layer state_dicts to new stage assignments

    // DP/ZeRO resharding: optimizer state repartitioning
    IF dst_dp != src_dp:
        FLATTEN optimizer states
        REPARTITION into dst_dp equal slices
        REDISTRIBUTE to new DP ranks

    SAVE with destination topology metadata
```

---

## 8. Framework Implementation: DeepSpeed

### 8.1 ZeRO Stages — Implementation Detail

DeepSpeed's ZeRO (Zero Redundancy Optimizer) provides three progressive sharding stages:

```
PROCEDURE DeepSpeedZeRO:
    // ===== ZeRO Stage 1 =====
    // Partition: Optimizer states only
    // Each DP rank owns 1/dp of parameters for optimization
    DURING backward:
        ALL-REDUCE gradients (full 2P bytes, ring all-reduce)
    DURING optimizer_step:
        EACH rank updates its 1/dp parameter slice
        ALL-GATHER updated parameters across DP ranks
    MEMORY: model_bf16 + model_fp32/dp + grads_fp32 + optim/dp

    // ===== ZeRO Stage 2 =====
    // Partition: Optimizer states + Gradients
    DURING backward:
        REDUCE-SCATTER gradients (each rank gets its 1/dp slice)
        // Volume: P bytes (half of all-reduce)
    DURING optimizer_step:
        EACH rank updates its 1/dp parameter slice
        ALL-GATHER updated parameters
    MEMORY: model_bf16 + model_fp32/dp + grads_fp32/dp + optim/dp

    // ===== ZeRO Stage 3 (FSDP equivalent) =====
    // Partition: Optimizer states + Gradients + Parameters
    DURING forward (per layer):
        ALL-GATHER parameters for current layer from all DP ranks
        COMPUTE forward for current layer
        DISCARD gathered parameters (free memory)
    DURING backward (per layer):
        ALL-GATHER parameters for current layer (needed for gradient computation)
        COMPUTE backward for current layer
        REDUCE-SCATTER gradients for current layer
        DISCARD gathered parameters
    DURING optimizer_step:
        EACH rank updates its 1/dp parameter slice (already local)
    MEMORY: model_bf16/dp + model_fp32/dp + grads_fp32/dp + optim/dp
    // Minimum model state memory, maximum communication
```

### 8.2 DeepSpeed Configuration Structure

```
PROCEDURE GenerateDeepSpeedConfig(model_config, parallel_config):
    ds_config = {
        "train_batch_size": GBS,
        "train_micro_batch_size_per_gpu": mbs,
        "gradient_accumulation_steps": gas,   // auto-computed if omitted

        "bf16": {"enabled": true},

        "zero_optimization": {
            "stage": zero_stage,              // 0, 1, 2, or 3
            "overlap_comm": true,             // Overlap reduce-scatter with backward
            "contiguous_gradients": true,     // Reduce memory fragmentation
            "reduce_bucket_size": 5e8,        // 500M elements per bucket
            "allgather_bucket_size": 5e8,

            // ZeRO-3 specific
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,

            // Offload (GPU-poor)
            "offload_optimizer": {
                "device": "cpu",              // or "nvme"
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            }
        },

        "activation_checkpointing": {
            "partition_activations": true,     // Shard activations across DP
            "contiguous_memory_optimization": true,
            "number_checkpoints": num_layers,
            "synchronize_checkpoint_boundary": false
        },

        "gradient_clipping": 1.0,

        "zero_allow_untested_optimizer": false,

        "communication_data_type": "bf16",

        "wall_clock_breakdown": true          // Step-time decomposition
    }

    RETURN ds_config
```

### 8.3 DeepSpeed + Megatron-LM Integration

DeepSpeed can be combined with Megatron-LM's TP and PP while providing ZeRO for the DP dimension:

```
PROCEDURE MegatronDeepSpeedIntegration:
    // Megatron handles: TP, PP, SP (sequence parallelism)
    // DeepSpeed handles: ZeRO sharding on DP dimension, optimizer, gradient clipping

    // Launch configuration:
    // world_size = tp * pp * dp
    // DeepSpeed sees dp_world_size = world_size / (tp * pp)

    // Process group hierarchy:
    TP_GROUP:  ranks within same TP partition (size = tp)
    PP_GROUP:  ranks across pipeline stages (size = pp)
    DP_GROUP:  ranks with same TP rank and PP stage (size = dp)
    // DeepSpeed applies ZeRO within DP_GROUP

    // Gradient flow:
    // 1. Backward computes gradients (sharded by TP)
    // 2. TP all-reduce within TP_GROUP (if sequence-parallel, reduce-scatter)
    // 3. DeepSpeed ZeRO reduce-scatter within DP_GROUP
    // 4. Optimizer step on owned shard
    // 5. DeepSpeed all-gather within DP_GROUP
    // 6. Pipeline send/recv activations within PP_GROUP
```

### 8.4 DeepSpeed Bucket Sizing

Bucket sizing critically affects overlap efficiency:

$$
t_{\text{bucket\_comm}} = \frac{\text{bucket\_size\_bytes} \times (\text{dp} - 1)}{\text{dp} \times \text{BW}_{\text{inter\_node}}}
$$

$$
t_{\text{bucket\_compute}} = \frac{\text{params\_in\_bucket} \times 2 \times S \times \text{mbs} \times 6}{\text{peak\_FLOPS}}
$$

**Optimal bucket size:** Choose such that `t_bucket_compute ≈ t_bucket_comm` — this maximizes overlap.

```
PROCEDURE OptimizeBucketSize(model, dp, BW_inter, peak_FLOPS, S, mbs):
    // Solve: bucket_params * 2 / BW ≈ bucket_params * 6 * 2 * S * mbs / peak_FLOPS
    // → bucket_params ≈ (peak_FLOPS * 2) / (BW * 6 * 2 * S * mbs) [if dp >> 1]

    optimal_bucket_elements = (peak_FLOPS * 2) / (BW_inter * 12 * S * mbs)
    optimal_bucket_bytes = optimal_bucket_elements * 2   // BF16

    // Clamp to practical range
    bucket_size = CLAMP(optimal_bucket_bytes, 10MB, 500MB)
    RETURN bucket_size
```

---

## 9. Launch Orchestration: torchrun and Elastic Recovery

### 9.1 torchrun Architecture

`torchrun` (PyTorch Elastic) is the canonical launcher for distributed training. It replaces `torch.distributed.launch` with elastic fault tolerance.

```
PROCEDURE TorchrunLaunch(num_nodes, gpus_per_node, training_script):
    // torchrun sets environment variables per rank:
    // RANK:             global rank [0, world_size)
    // LOCAL_RANK:       rank within the node [0, gpus_per_node)
    // WORLD_SIZE:       total number of processes
    // MASTER_ADDR:      address of rank 0 node
    // MASTER_PORT:      port for rendezvous
    // GROUP_RANK:       node index [0, num_nodes)

    // Launch command per node:
    // torchrun \
    //   --nnodes=<num_nodes> \
    //   --nproc-per-node=<gpus_per_node> \
    //   --rdzv-backend=c10d \
    //   --rdzv-endpoint=<master_addr>:<master_port> \
    //   --rdzv-id=<job_id> \
    //   --max-restarts=3 \
    //   --monitor-interval=5 \
    //   training_script.py [args]

    // Elastic semantics:
    // - Workers join via rendezvous barrier
    // - If a worker fails, remaining workers can continue (with world_size reduction)
    //   OR the failed worker is restarted and re-joins
    // - Checkpoint-based recovery: training resumes from last saved state
```

### 9.2 Multi-Node Launch Orchestration

```
PROCEDURE ProductionLaunch(cluster_config):
    // ===== Phase 1: Preflight Health Checks =====
    FOR EACH node IN cluster_config.nodes:
        VERIFY GPU count (nvidia-smi / rocm-smi)
        VERIFY driver version (nvidia-smi --query-gpu=driver_version)
        VERIFY NCCL/RCCL version
        VERIFY InfiniBand link status (ibstat: all ports ACTIVE)
        VERIFY NVLink topology (nvidia-smi topo -m)
        VERIFY GPU memory (no residual allocations)
        VERIFY CUDA/ROCm runtime version
        VERIFY container image hash (ensure identical across nodes)
        RUN GPU burn test (60s) to detect faulty GPUs
        VERIFY storage mount (checkpoint directory accessible, write test)

    // ===== Phase 2: Bandwidth Tests =====
    RUN NCCL all-reduce bandwidth test (nccl-tests/all_reduce_perf):
        - Intra-node: expect ~400-900 GB/s (NVSwitch dependent)
        - Inter-node: expect ~80-100% of IB line rate
    RUN point-to-point bandwidth test across all node pairs
    FLAG any link with < 80% of expected bandwidth

    // ===== Phase 3: Topology Discovery =====
    DETECT NVLink/NVSwitch topology per node
    DETECT InfiniBand fabric topology (ibnetdiscover)
    GENERATE NCCL topology file (NCCL_TOPO_FILE)
    COMPUTE optimal rank-to-GPU mapping:
        - TP ranks on NVLink-connected GPUs within node
        - PP ranks across nodes on same IB rail
        - DP ranks across nodes
    EXPORT mapping as environment variables or hostfile

    // ===== Phase 4: Environment Setup =====
    SET NCCL_SOCKET_IFNAME = <IB interface>
    SET NCCL_IB_DISABLE = 0
    SET NCCL_IB_HCA = <HCA list>
    SET NCCL_NET_GDR_LEVEL = 5         // GPUDirect RDMA
    SET NCCL_IB_QPS_PER_CONNECTION = 4
    SET NCCL_CROSS_NIC = 1             // Enable cross-NIC for multi-rail
    SET CUDA_DEVICE_MAX_CONNECTIONS = 1 // For Megatron overlap
    SET NCCL_ALGO = Ring               // or Tree, depending on topology
    SET NCCL_PROTO = Simple            // or LL128 for small messages
    SET OMP_NUM_THREADS = 4            // For dataloader workers

    // For AMD MI300X:
    SET NCCL_LIBRARY = /path/to/rccl/librccl.so
    SET HSA_FORCE_FINE_GRAIN_PCIE = 1
    SET RCCL_MSCCL_ENABLE = 1

    // ===== Phase 5: Launch =====
    FOR EACH node IN cluster_config.nodes PARALLEL:
        torchrun \
            --nnodes={num_nodes} \
            --nproc-per-node={gpus_per_node} \
            --rdzv-backend=c10d \
            --rdzv-endpoint={master_addr}:{master_port} \
            --rdzv-id={unique_job_id} \
            --max-restarts=3 \
            train.py {training_args}

    // ===== Phase 6: Monitoring =====
    CONTINUOUSLY MONITOR:
        - Per-rank step time (detect stragglers)
        - GPU utilization and memory (nvidia-smi dmon)
        - Network counters (ibstat, perfquery)
        - Training loss (detect divergence)
        - Checkpoint save success
        - Dataloader throughput (tokens/sec consumed)
```

### 9.3 Elastic Recovery and Checkpoint Resilience

```
PROCEDURE ElasticRecovery(training_state, checkpoint_dir):
    // On failure detection:
    // 1. Surviving workers enter barrier
    // 2. New workers spawned (or same workers restarted)
    // 3. All workers rendezvous with potentially new world_size

    // Checkpoint-based recovery:
    latest_ckpt = FindLatestValidCheckpoint(checkpoint_dir)

    IF latest_ckpt.world_size != current_world_size:
        // Resharding required
        IF latest_ckpt.tp != current_tp OR latest_ckpt.pp != current_pp:
            ReshardCheckpoint(latest_ckpt, current_config)
        // ZeRO/DP resharding
        IF latest_ckpt.dp != current_dp:
            ReshardOptimizerStates(latest_ckpt, current_dp)

    LOAD model_state_dict
    LOAD optimizer_state_dict
    LOAD lr_scheduler_state
    LOAD dataloader_state (consumed samples, RNG state)

    // Validate checkpoint integrity
    VERIFY parameter checksums
    VERIFY optimizer state shapes match model parameters
    VERIFY training step counter

    RESUME training from loaded state
```

---

## 10. Kernel Optimization: Triton, FlashAttention, Fused Kernels

### 10.1 FlashAttention

FlashAttention eliminates the materialization of the `S×S` attention matrix by computing attention in tiles, maintaining running statistics (online softmax), and fusing the entire attention computation into a single kernel.

**Memory reduction:**

$$
\text{Standard attention activation:} \quad O(S^2 \times n_{\text{heads}} \times \text{mbs})
$$

$$
\text{FlashAttention activation:} \quad O(S \times H \times \text{mbs})
$$

> For S=128K, H=8192, n_heads=64: standard requires ~128K² × 64 × 2 = 2 TB per sample (infeasible). FlashAttention requires ~128K × 8192 × 2 = 2 GB per sample.

```
PROCEDURE FlashAttentionForward(Q, K, V, block_size_q, block_size_kv):
    // Q, K, V: [batch, seq, num_heads, head_dim]
    // Output O: [batch, seq, num_heads, head_dim]
    // L: [batch, num_heads, seq]  (log-sum-exp for backward)

    FOR EACH query_block IN PARTITION(Q, block_size_q):
        m_i = -infinity    // running max
        l_i = 0            // running sum of exp
        O_i = 0            // running output accumulator

        FOR EACH kv_block IN PARTITION(K, V, block_size_kv):
            // Load Q_block, K_block, V_block from HBM to SRAM
            S_ij = Q_block @ K_block^T / sqrt(d)      // In SRAM
            APPLY causal mask IF applicable

            m_new = MAX(m_i, ROWMAX(S_ij))
            P_ij = EXP(S_ij - m_new)                   // Numerically stable
            l_new = EXP(m_i - m_new) * l_i + ROWSUM(P_ij)

            // Rescale previous accumulator
            O_i = (EXP(m_i - m_new) * l_i / l_new) * O_i + (P_ij / l_new) @ V_block

            m_i = m_new
            l_i = l_new

        STORE O_i, L_i = m_i + LOG(l_i) to HBM

    // Backward: recompute S_ij from Q, K blocks (no S×S stored)
```

### 10.2 Fused Kernels for Transformer Layers

| Kernel | What It Fuses | Memory Benefit | Compute Benefit |
|---|---|---|---|
| **Fused RMSNorm** | Norm + scaling in one kernel | Eliminates intermediate variance tensor | Reduces kernel launch overhead, single HBM pass |
| **Fused SwiGLU / GeGLU** | Gate projection + activation + up-projection | Eliminates 2× intermediate tensors | Single kernel, better register utilization |
| **Fused Softmax** | Softmax with causal mask and scaling | Eliminates mask materialization | Online computation, numerical stability |
| **Fused RoPE** | Rotary position embedding application | In-place, no extra tensor | Fused with Q/K projection or attention |
| **Fused Cross-Entropy** | Log-softmax + NLL loss | Eliminates full logits materialization (vocab×seq) | Chunks computation, massive memory savings for large vocab |
| **Fused Adam** | Adam update + FP32→BF16 cast | In-place, single pass over parameters | Reduces optimizer step time by 2–3× |

### 10.3 Triton Custom Kernels

Triton enables writing custom GPU kernels in Python with auto-tuning:

```
PROCEDURE TritonFusedResidualRMSNorm(x, residual, weight, eps):
    // Fuses: residual addition + RMS normalization + scaling
    // Standard: 3 separate kernels, 3 HBM reads + writes
    // Fused: 1 kernel, 1 HBM read + 1 write

    TRITON_KERNEL(grid = (batch * seq,)):
        // Load entire hidden dimension row into SRAM
        row = LOAD x[row_idx, :]
        res = LOAD residual[row_idx, :]

        // Fused residual add
        hidden = row + res

        // RMS norm in registers
        variance = MEAN(hidden * hidden)
        hidden_norm = hidden * RSQRT(variance + eps)

        // Scale
        output = hidden_norm * LOAD weight[:]

        // Single store
        STORE output → output[row_idx, :]
        STORE hidden → residual[row_idx, :]  // Updated residual for next layer

    // Auto-tune: BLOCK_SIZE over {256, 512, 1024, 2048}
    // Constraint: BLOCK_SIZE must be >= hidden_dim for single-row processing
```

### 10.4 CUDA Graphs for Launch Overhead Elimination

```
PROCEDURE CUDAGraphCapture(model, sample_batch):
    // Warm-up: run model once to initialize lazy state
    model.forward(sample_batch)

    // Capture: record all CUDA operations into a graph
    graph = CUDA_GRAPH_BEGIN_CAPTURE(stream)
    output = model.forward(sample_batch)
    loss = compute_loss(output)
    loss.backward()
    CUDA_GRAPH_END_CAPTURE(graph)

    // Replay: execute entire captured graph with single launch
    // Eliminates per-kernel launch overhead (typically 5-15μs per kernel)
    // For Transformer with ~100 kernels per layer, 80 layers:
    // Savings: ~100 * 80 * 10μs = 80ms per step (significant for small mbs)

    PROCEDURE train_step(batch):
        COPY batch data into graph's input buffers
        GRAPH.replay()
        RETURN graph's output buffers

    // Limitations:
    // - Cannot capture dynamic shapes (sequence length must be fixed)
    // - Cannot capture CPU-side control flow
    // - NCCL collectives require special handling (NCCL_GRAPH_MIXING_SUPPORT)
```

### 10.5 FP8 Training (H100/B200/MI350)

```
PROCEDURE FP8ForwardLinear(input_bf16, weight_fp8, scale_input, scale_weight):
    // Quantize input to FP8 E4M3
    input_fp8 = QUANTIZE(input_bf16, scale_input, format=E4M3)

    // FP8 GEMM on Tensor Cores (H100: 2x throughput vs. BF16)
    output_fp32 = FP8_GEMM(input_fp8, weight_fp8)

    // Dequantize and apply output scale
    output_bf16 = output_fp32 * (scale_input * scale_weight)

    // Delayed scaling: update amax history
    UPDATE amax_history(input) WITH MAX(ABS(input_bf16))
    UPDATE amax_history(weight) WITH MAX(ABS(weight_fp8))

    // Compute new scale for next iteration
    // scale = (FP8_MAX / amax) with amax from history (delayed by 1+ steps)
    scale_input_next = FP8_E4M3_MAX / MAX(amax_history(input)[-window:])
    scale_weight_next = FP8_E4M3_MAX / MAX(amax_history(weight)[-window:])

    RETURN output_bf16

    // Backward: use FP8 E5M2 for gradients (wider range, less precision)
    // Gradient GEMM: FP8_E5M2 × FP8_E4M3 → FP32 accumulation → BF16 output
```

**Numerical Validation Protocol:**

```
PROCEDURE ValidateFP8Parity(model, dataset, num_steps=1000):
    // Baseline: BF16 training
    baseline_losses = TrainBF16(model, dataset, num_steps)
    baseline_grad_norms = RecordGradNorms(model, dataset, num_steps)

    // FP8 training: same seeds, same data order
    fp8_losses = TrainFP8(model, dataset, num_steps)
    fp8_grad_norms = RecordGradNorms(model, dataset, num_steps)

    // Validation criteria:
    // 1. Loss curves must converge to same neighborhood
    loss_divergence = MAX(ABS(baseline_losses - fp8_losses)) / MEAN(baseline_losses)
    ASSERT loss_divergence < 0.02  // < 2% relative deviation

    // 2. Gradient norms must track
    grad_correlation = PEARSON_R(baseline_grad_norms, fp8_grad_norms)
    ASSERT grad_correlation > 0.98

    // 3. No overflow/underflow events
    ASSERT fp8_overflow_count == 0
    ASSERT fp8_underflow_fraction < 0.001  // < 0.1% of tensor elements

    REPORT parity results
```

---

## 11. Topology-Aware Placement and Hardware Mapping

### 11.1 Hardware Interconnect Hierarchy

| Level | Interconnect | Bandwidth (Bidirectional) | Latency | Parallelism Mapping |
|---|---|---|---|---|
| **Intra-SM** | Registers, Shared Memory | ~20 TB/s | <1 ns | Kernel-internal |
| **Intra-GPU** | HBM3/HBM3e | 3.35 TB/s (H100), 8 TB/s (B200) | ~100 ns | Memory-bound kernels |
| **Intra-Node (NVLink)** | NVLink 4.0 / NVSwitch | 900 GB/s (H100 NVSwitch) | 1–5 μs | **TP, SP** |
| **Intra-Node (xGMI)** | AMD Infinity Fabric | 896 GB/s (MI300X, 8-way) | 1–5 μs | **TP, SP** |
| **Inter-Node (IB)** | InfiniBand NDR/XDR | 400–800 Gb/s = 50–100 GB/s | 1–5 μs | **PP, DP, EP** |
| **Inter-Node (RoCE)** | RoCE v2 | 200–400 Gb/s = 25–50 GB/s | 2–10 μs | **PP, DP** |
| **Storage** | NVMe/NFS/Lustre | 1–100 GB/s | 10 μs–10 ms | Checkpoint, dataloader |

### 11.2 Placement Algorithm

```
PROCEDURE TopologyAwarePlacement(world_size, tp, pp, dp, cp, ep, cluster):
    // Constraint: tp * pp * dp * cp * ep = world_size
    gpus_per_node = cluster.gpus_per_node  // 8 typical

    // Rule 1: TP group must be within single NVSwitch domain
    ASSERT tp <= gpus_per_node
    // Place TP ranks on GPUs [0..tp-1] within each node

    // Rule 2: SP (sequence parallelism) follows TP group
    // SP is always co-located with TP (same process group)

    // Rule 3: CP group — prefer intra-node if bandwidth permits
    IF cp <= gpus_per_node / tp:
        // CP within node: multiple TP groups within one node form CP group
        cp_placement = INTRA_NODE
    ELSE:
        // CP across nodes: Ring Attention must overlap with attention compute
        cp_placement = INTER_NODE
        VERIFY ring_attention_overlap_condition(cp, S, H, BW_IB)

    // Rule 4: EP group — place on high-bandwidth domain
    IF ep <= gpus_per_node / tp:
        ep_placement = INTRA_NODE
    ELSE:
        ep_placement = INTER_NODE
        // all-to-all is bandwidth-intensive; verify:
        VERIFY all2all_bandwidth_sufficient(ep, E, tokens, BW_IB)

    // Rule 5: PP stages across nodes (minimal communication volume)
    // Assign consecutive PP stages to consecutive nodes
    // Within each PP stage: tp GPUs within one node
    nodes_per_pp_stage = 1  // ideally
    IF tp * dp_per_stage > gpus_per_node:
        nodes_per_pp_stage = CEIL((tp * dp_per_stage) / gpus_per_node)

    // Rule 6: DP group — spans across nodes
    // DP ranks are GPUs with same TP rank and same PP stage
    // Communication: gradient all-reduce (or reduce-scatter for ZeRO-2)
    // Place DP ranks to maximize bisection bandwidth

    // Generate rank mapping
    rank = 0
    FOR pp_stage IN [0..pp-1]:
        FOR dp_rank IN [0..dp-1]:
            FOR cp_rank IN [0..cp-1]:
                FOR ep_rank IN [0..ep-1]:
                    FOR tp_rank IN [0..tp-1]:
                        node = ComputeNodeAssignment(pp_stage, dp_rank, cp_rank, ep_rank)
                        gpu = tp_rank  // within node
                        ASSIGN rank → (node, gpu)
                        rank += 1

    RETURN rank_mapping
```

### 11.3 Cluster-Specific Configurations

| Cluster | GPU | HBM | Intra-Node BW | Inter-Node BW | Recommended TP | Notes |
|---|---|---|---|---|---|---|
| **DGX A100** | A100 80GB | 80 GB HBM2e | 600 GB/s (NVSwitch) | 200 Gb/s (HDR IB) | TP≤8 | 6 NVSwitches per node |
| **DGX H100** | H100 SXM | 80 GB HBM3 | 900 GB/s (NVSwitch) | 400 Gb/s (NDR IB) | TP≤8 | 4th-gen NVSwitch, 3.35 TB/s HBM |
| **DGX B200** | B200 | 192 GB HBM3e | 1.8 TB/s (NVLink 5.0) | 800 Gb/s (XDR IB) | TP≤8 | 5th-gen NVSwitch, 8 TB/s HBM |
| **MI300X** | MI300X | 192 GB HBM3 | 896 GB/s (xGMI) | 400 Gb/s (RoCE/IB) | TP≤8 | 8 XCDs per GPU, RCCL |
| **MI350** | MI350 | 288 GB HBM3e | ~TBD | ~TBD | TP≤8 | CDNA4, FP8/FP4 |

---

## 12. Long-Context and MoE Special Cases

### 12.1 Context Parallelism (CP) for Long Sequences

When sequence lengths exceed ~32K tokens, activation memory for attention becomes prohibitive even with FlashAttention. Context parallelism partitions the sequence across CP ranks.

**Ring Attention implementation:**

```
PROCEDURE RingAttentionForward(Q_local, K_local, V_local, cp_group):
    // Each CP rank holds S/cp tokens
    // Q_local: [batch, S/cp, num_heads, head_dim] — stays local
    // K_local, V_local: [batch, S/cp, num_kv_heads, head_dim] — rotate in ring

    O_local = zeros_like(Q_local)
    lse_local = -infinity  // log-sum-exp accumulator

    kv_buffer = (K_local, V_local)

    FOR step IN [0..cp-1]:
        // Compute attention for current KV block
        O_block, lse_block = FlashAttention(Q_local, kv_buffer.K, kv_buffer.V)

        // Online log-sum-exp combination
        O_local, lse_local = CombineAttentionOutputs(
            O_local, lse_local, O_block, lse_block)

        // Async send/recv: overlap with next attention computation
        IF step < cp - 1:
            ASYNC_SEND kv_buffer TO next_rank_in_ring
            ASYNC_RECV new_kv_buffer FROM prev_rank_in_ring
            kv_buffer = new_kv_buffer

    RETURN O_local

    // Communication volume per layer:
    // (cp - 1) × 2 × (S/cp) × batch × num_kv_heads × head_dim × 2 bytes
    // For GQA (grouped query attention): much less than full head_dim × num_heads

    // Overlap condition:
    // Attention compute time for one block > KV transfer time
    // 2 × (S/cp)² × batch × num_heads × head_dim / peak_FLOPS > V_kv / BW
```

**Ulysses-style sequence parallelism (alternative to Ring Attention):**

```
PROCEDURE UlyssesSequenceParallel(Q, K, V, cp_group):
    // All-to-all: redistribute from sequence-split to head-split
    // Before: each rank has [batch, S/cp, ALL_heads, head_dim]
    // After:  each rank has [batch, S, heads/cp, head_dim]

    Q_head_split = ALL_TO_ALL(Q, split_dim=heads, gather_dim=seq)
    K_head_split = ALL_TO_ALL(K, split_dim=heads, gather_dim=seq)
    V_head_split = ALL_TO_ALL(V, split_dim=heads, gather_dim=seq)

    // Standard FlashAttention on full sequence, subset of heads
    O_head_split = FlashAttention(Q_head_split, K_head_split, V_head_split)

    // All-to-all: redistribute back to sequence-split
    O_seq_split = ALL_TO_ALL(O_head_split, split_dim=seq, gather_dim=heads)

    RETURN O_seq_split

    // Advantage: No ring iteration, full FlashAttention efficiency
    // Disadvantage: Two all-to-all per layer (bandwidth-intensive)
    // Best when: cp is small (2-4) and all-to-all bandwidth is high (intra-node)
```

### 12.2 Expert Parallelism (EP) for MoE

```
PROCEDURE MoEForwardWithEP(input, router, experts, ep_group):
    // input: [batch * seq, hidden_dim] — flattened token representations
    // router: learned gating network
    // experts: E total experts, each rank holds E/ep experts

    // Step 1: Route tokens to experts
    router_logits = router(input)                    // [tokens, E]
    top_k_indices, top_k_weights = TopK(router_logits, k=top_k)

    // Step 2: Compute auxiliary load balancing loss
    // Prevent expert collapse: encourage uniform routing
    expert_counts = HISTOGRAM(top_k_indices, bins=E)
    balance_loss = E * SUM((expert_counts / tokens) * SOFTMAX(router_logits).mean(0))

    // Step 3: All-to-all dispatch — send tokens to expert-owning ranks
    // Each rank sends tokens destined for remote experts
    // Each rank receives tokens destined for its local experts
    dispatched_input = ALL_TO_ALL(input, routing=top_k_indices, group=ep_group)

    // Step 4: Expert computation (local)
    // Grouped GEMM: batch all tokens for each local expert into a single GEMM
    expert_output = GroupedGEMM(dispatched_input, local_experts)

    // Step 5: All-to-all combine — return processed tokens to source ranks
    combined_output = ALL_TO_ALL(expert_output, routing=reverse_mapping, group=ep_group)

    // Step 6: Weighted combination
    output = SUM(combined_output * top_k_weights, dim=expert_dim)

    RETURN output, balance_loss

    // Communication volume per MoE layer:
    // 2 × tokens × hidden_dim × 2 bytes × (ep - 1) / ep  (dispatch + combine)

    // Token dropping (capacity factor):
    // IF expert receives > capacity_factor * (tokens * top_k / E) tokens:
    //     DROP excess tokens (or route to shared expert)
    // Prevents memory overflow but wastes compute
```

**Expert Tensor Parallelism (combining EP with TP within experts):**

```
PROCEDURE ExpertTensorParallel(experts, ep, etp):
    // etp: expert tensor parallelism degree
    // Each expert's FFN weight matrices are split across etp ranks
    // Total expert-parallel ranks = ep * etp

    // Enables very large individual expert sizes
    // Communication: all-reduce within etp group after expert FFN

    FOR EACH expert IN local_experts:
        // Column-parallel first linear
        intermediate = ColumnParallelLinear(input, expert.W1, tp_group=etp_group)
        intermediate = activation_fn(intermediate)
        // Row-parallel second linear
        output = RowParallelLinear(intermediate, expert.W2, tp_group=etp_group)
        // All-reduce within etp_group (implicit in row-parallel)

    RETURN output
```

### 12.3 MoE Memory and Batch Size Considerations

For an MoE model with `E` experts, `top_k` routing, and `d_expert` parameters per expert:

$$
P_{\text{total}} = P_{\text{dense}} + E \times d_{\text{expert}}
$$

$$
P_{\text{per\_gpu}} = \frac{P_{\text{dense}}}{\text{tp} \times \text{pp}} + \frac{E \times d_{\text{expert}}}{\text{tp} \times \text{pp} \times \text{ep}}
$$

> MoE models are parameter-rich but compute-sparse: each token activates only `top_k` experts. The FLOPs per token are approximately the same as a dense model with `P_dense + top_k × d_expert` parameters, but the memory footprint includes all `E` experts.

---

## 13. Glossary and Canonical Formulae

### 13.1 Parallelization Terms

| Symbol | Definition |
|---|---|
| `tp` | Tensor parallelism degree — number of GPUs sharing each weight matrix |
| `pp` | Pipeline parallelism degree — number of pipeline stages |
| `dp` | Data parallelism degree — number of independent data-parallel replicas |
| `cp` | Context parallelism degree — number of GPUs sharing one sequence |
| `ep` | Expert parallelism degree — number of GPUs sharing expert pool |
| `sp` | Sequence parallelism — LayerNorm/Dropout partitioned along sequence within TP group (co-located with TP) |
| `d_z1` | Effective DP for ZeRO-1 sharding: `dp` if ZeRO≥1 else `1` |
| `d_z2` | Effective DP for ZeRO-2 sharding: `dp` if ZeRO≥2 else `1` |
| `d_z3` | Effective DP for ZeRO-3 sharding: `dp` if ZeRO≥3 else `1` |

### 13.2 Batch Size Terms

| Symbol | Definition | Formula |
|---|---|---|
| `mbs` | Micro-batch size — sequences per GPU per forward pass | User-configured |
| `gas` | Gradient accumulation steps — sequential microbatches per optimizer step | `GBS / (mbs × dp × S)` in tokens, or `GBS / (mbs × dp)` in sequences |
| `S` | Sequence length per sample | Fixed or variable (packed) |
| `S_gpu` | Effective sequence length per GPU after CP | `S / cp` |
| `GBS` | Global batch size (in sequences) | `mbs × dp × gas` |
| `GBS_tokens` | Global batch size in tokens | `mbs × S × dp × gas` |

### 13.3 Memory Terms and Formulae

| Symbol | Formula | Description |
|---|---|---|
| `model_bf16` | `2 × num_params / (tp × pp × d_z3)` | BF16 model weights per GPU |
| `model_fp32` | `4 × num_params / (tp × pp × d_z1)` | FP32 master weights per GPU (for optimizer) |
| `grads_fp32` | `4 × num_params / (tp × pp × d_z2)` | FP32 gradients per GPU |
| `optim_states` | `8 × num_params / (tp × pp × d_z1)` | Adam momentum + variance (FP32) per GPU |
| `activations` | `f(model_config, S_gpu, mbs, tp, cp, pp, recomp)` | Activation memory (see §1.2) |
| `num_params` | `L × 12H² + V × H` (approx.) | Total parameter count |

**Peak memory per GPU:**

$$
M_{\text{peak}} = \text{model\_bf16} + \text{model\_fp32} + \text{grads\_fp32} + \text{optim\_states} + A_{\text{act}} + M_{\text{buffer}}
$$

### 13.4 Compute Formulae

**FLOPs per training step (forward + backward):**

$$
C_{\text{step}} = 6 \times P \times \text{mbs} \times S \times \text{gas}
$$

> Factor of 6 = 2 (forward multiply-add) × 3 (forward + two backward passes for weight and activation gradients). Some references use `6P × tokens_per_step` which is equivalent.

**FLOPs per GPU per step:**

$$
C_{\text{gpu}} = \frac{C_{\text{step}}}{\text{world\_size}} = \frac{6 \times P \times \text{mbs} \times S \times \text{gas}}{\text{tp} \times \text{pp} \times \text{dp}}
$$

**Model FLOPs Utilization (MFU):**

$$
\text{MFU} = \frac{C_{\text{gpu}}}{T_{\text{step}} \times \text{FLOPS}_{\text{peak}}}
$$

Where `T_step` is the measured wall-clock time per training step, and `FLOPS_peak` is the GPU's theoretical peak (e.g., 989 TFLOPS BF16 for H100 SXM).

> Good MFU targets: 40–55% at scale for dense models on H100 clusters. MoE models typically achieve lower MFU due to all-to-all communication and expert load imbalance.

**Hardware FLOPs Utilization (HFU):**

$$
\text{HFU} = \frac{C_{\text{gpu}} + C_{\text{recompute}}}{T_{\text{step}} \times \text{FLOPS}_{\text{peak}}}
$$

HFU includes recomputed FLOPs and better represents actual hardware utilization.

**Tokens per second per GPU:**

$$
\text{TPS}_{\text{gpu}} = \frac{\text{mbs} \times S \times \text{gas}}{T_{\text{step}}}
$$

**Pipeline bubble fraction (1F1B):**

$$
\beta_{\text{1F1B}} = \frac{\text{pp} - 1}{\text{gas} + \text{pp} - 1}
$$

**Pipeline bubble fraction (interleaved, `v` virtual stages):**

$$
\beta_{\text{interleaved}} = \frac{\text{pp} - 1}{v \times \text{gas} + \text{pp} - 1}
$$

---

## 14. Decision Flowchart — Complete Pseudocode

```
PROCEDURE DistributedTrainingConfigurationEngine(
    model_config,        // L, H, V, num_heads, num_kv_heads, ffn_mult, num_experts
    training_config,     // target_GBS_tokens, S, learning_rate, optimizer
    cluster_config,      // num_nodes, gpus_per_node, gpu_type, HBM, NVLink_BW, IB_BW
    constraints          // max_step_time, min_MFU, max_bubble_fraction
):

    // ====================================================================
    // PHASE 1: COMPUTE MODEL PROPERTIES
    // ====================================================================
    P = ComputeParamCount(model_config)
    world_size = cluster_config.num_nodes * cluster_config.gpus_per_node
    HBM = cluster_config.HBM_bytes
    gpus_per_node = cluster_config.gpus_per_node

    LOG "Model: {P/1e9:.1f}B parameters, {world_size} GPUs, {HBM/1e9:.0f}GB HBM each"

    // ====================================================================
    // PHASE 2: FIT MODEL INTO MEMORY (Step 1)
    // ====================================================================

    // --- Determine TP ---
    IF P < 10e9 AND world_size <= gpus_per_node:
        // Small model, single node
        tp = gpus_per_node
        pp = 1
        dp = 1
        zero_stage = 0
        recomp = SELECTIVE
    ELSE IF P < 10e9:
        // Small model, multi-node: use ZeRO-3 or TP
        tp = gpus_per_node
        pp = 1
        dp = world_size / tp
        zero_stage = 1
        recomp = SELECTIVE
    ELSE:
        // Large model: TP within node, need PP or ZeRO-3
        tp = gpus_per_node     // Fill NVSwitch domain

        // Compute per-GPU state with TP only
        state_tp = (18 * P) / tp   // 2+4+4+8 bytes per param, sharded by TP
        IF state_tp < 0.6 * HBM:
            // Fits with TP alone; use DP for remaining GPUs
            pp = 1
            dp = world_size / tp
            zero_stage = 1
        ELSE:
            // Need PP to further split model
            pp = CEIL(state_tp / (0.5 * HBM))
            pp = NextPowerOf2(pp)    // PP degree should divide L evenly
            ASSERT model_config.L MOD pp == 0, "Layers must divide evenly into PP stages"
            dp = world_size / (tp * pp)
            zero_stage = 2    // ZeRO-2 for DP dimension

    // --- Verify memory fit ---
    config = {tp, pp, dp, cp=1, ep=1, zero_stage, mbs=1, gas=1, recomp}
    mem_result = ValidateMemoryFit(model_config, config, cluster_config)

    IF mem_result == FAIL:
        // Escalate: try full recomputation
        config.recomp = FULL
        mem_result = ValidateMemoryFit(model_config, config, cluster_config)

    IF mem_result == FAIL:
        // Escalate: try ZeRO-3
        config.zero_stage = 3
        mem_result = ValidateMemoryFit(model_config, config, cluster_config)

    IF mem_result == FAIL:
        // Escalate: CPU offload (GPU-poor regime)
        config.offload_optimizer = CPU
        mem_result = ValidateMemoryFit(model_config, config, cluster_config)

    ASSERT mem_result == PASS, "Cannot fit model even with all memory optimizations"

    // --- MoE: add EP ---
    IF model_config.num_experts > 1:
        ep = model_config.num_experts    // Start with full EP
        // Adjust dp: dp = world_size / (tp * pp * ep)
        IF dp < 1:
            ep = world_size / (tp * pp)  // Reduce EP to fit
        config.ep = ep
        config.dp = world_size / (tp * pp * ep)
        VERIFY memory fit with EP

    // --- Long context: add CP ---
    IF training_config.S > 32768:
        // Activation memory for attention may be excessive
        cp = 2
        WHILE NOT ValidateMemoryFit(config WITH cp):
            cp = cp * 2
            ASSERT cp <= gpus_per_node, "CP exceeds node size"
        config.cp = cp
        config.dp = world_size / (tp * pp * cp * config.ep)

    // ====================================================================
    // PHASE 3: SATISFY TARGET GBS (Step 2)
    // ====================================================================

    target_GBS_seqs = training_config.target_GBS_tokens / training_config.S
    current_GBS_seqs = config.mbs * config.dp * config.gas

    IF current_GBS_seqs < target_GBS_seqs:
        // Increase GBS
        // First: increase GAS
        required_gas = target_GBS_seqs / (config.mbs * config.dp)
        IF required_gas <= 256 AND IS_INTEGER(required_gas):
            config.gas = required_gas
        ELSE:
            // Increase mbs
            FOR mbs_candidate IN [2, 4, 8, 16, 32]:
                config.mbs = mbs_candidate
                required_gas = target_GBS_seqs / (config.mbs * config.dp)
                IF IS_INTEGER(required_gas) AND ValidateMemoryFit(config):
                    config.gas = required_gas
                    BREAK

    ELSE IF current_GBS_seqs > target_GBS_seqs:
        // Decrease GBS: reduce dp, reallocate to PP
        config.gas = 1
        required_dp = target_GBS_seqs / (config.mbs * 1)
        freed_gpus = (config.dp - required_dp) * tp
        // Reallocate freed GPUs to PP
        additional_pp = freed_gpus / tp
        config.pp = config.pp + additional_pp
        config.dp = required_dp

    VALIDATE config.mbs * config.dp * config.gas == target_GBS_seqs

    // ====================================================================
    // PHASE 4: OPTIMIZE THROUGHPUT (Step 3)
    // ====================================================================

    best_config = config
    best_MFU = 0

    // Experiment space: vary mbs, tp, pp, zero_stage
    FOR tp_try IN [2, 4, 8]:
      FOR pp_try IN [1, 2, 4, 8]:
        FOR mbs_try IN [1, 2, 4, 8]:
          FOR zero_try IN [0, 1, 2]:
            candidate = BuildConfig(tp_try, pp_try, mbs_try, zero_try, ...)

            IF NOT ValidateMemoryFit(candidate): CONTINUE
            IF NOT SatisfyGBS(candidate):         CONTINUE

            // Estimate throughput via cost model
            bubble = (pp_try - 1) / (candidate.gas + pp_try - 1)
            IF bubble > constraints.max_bubble_fraction: CONTINUE

            tp_comm_time = EstimateTPComm(tp_try, cluster_config)
            dp_comm_time = EstimateDPComm(candidate.dp, P, zero_try, cluster_config)
            pp_comm_time = EstimatePPComm(pp_try, mbs_try, S, H, cluster_config)

            compute_time = (6 * P * mbs_try * S * candidate.gas) /
                           (world_size * cluster_config.peak_FLOPS)

            // Account for overlap
            exposed_comm = MAX(0, tp_comm_time - compute_time * 0.3)  // partial overlap
                         + MAX(0, dp_comm_time - compute_time * 0.7)  // good overlap
                         + pp_comm_time * (1 - 0.9)                    // mostly overlapped

            step_time = compute_time * (1 + bubble) + exposed_comm
            MFU = (6 * P * mbs_try * S * candidate.gas) /
                  (step_time * world_size * cluster_config.peak_FLOPS)

            IF MFU > best_MFU:
                best_MFU = MFU
                best_config = candidate

    // ====================================================================
    // PHASE 5: ENABLE OVERLAP AND ADVANCED FEATURES
    // ====================================================================

    // Communication-computation overlap
    ENABLE gradient_reduce_scatter_overlap_with_backward
    ENABLE param_allgather_overlap_with_forward      // ZeRO-3/FSDP
    ENABLE pipeline_recv_overlap_with_compute         // PP
    ENABLE ring_attention_kv_overlap_with_attention   // CP

    // Kernel optimizations
    ENABLE flash_attention
    ENABLE fused_rmsnorm
    ENABLE fused_swiglu
    ENABLE fused_rope
    ENABLE fused_cross_entropy
    IF cluster_config.gpu_type IN [H100, B200, MI350]:
        ENABLE fp8_training WITH delayed_scaling

    // CUDA Graphs (if sequence length is fixed)
    IF training_config.S IS fixed AND NOT using PP:
        ENABLE cuda_graph_capture

    // ====================================================================
    // PHASE 6: VALIDATE AND DEPLOY
    // ====================================================================

    // Run validation
    RunPreflightChecks(cluster_config)
    RunBandwidthTests(cluster_config)
    RunShortTraining(best_config, num_steps=100)
    VerifyMemoryUsage(best_config)
    VerifyStepTimeDecomposition(best_config)
    VerifyLossConvergence(best_config, baseline)

    LOG "Final configuration:"
    LOG "  TP={tp}, PP={pp}, DP={dp}, CP={cp}, EP={ep}"
    LOG "  ZeRO={zero_stage}, mbs={mbs}, gas={gas}, GBS={GBS}"
    LOG "  Recomputation={recomp}, FP8={fp8_enabled}"
    LOG "  Estimated MFU={best_MFU:.1%}"
    LOG "  Tokens/sec/GPU={tps:.0f}"

    RETURN best_config


// ====================================================================
// FRAMEWORK-SPECIFIC LAUNCH
// ====================================================================

PROCEDURE GenerateLaunchCommand(config, framework):
    IF framework == "megatron-core":
        cmd = "torchrun"
            + " --nnodes={num_nodes}"
            + " --nproc-per-node={gpus_per_node}"
            + " --rdzv-backend=c10d"
            + " --rdzv-endpoint={master}:{port}"
            + " pretrain_gpt.py"
            + " --tensor-model-parallel-size={config.tp}"
            + " --pipeline-model-parallel-size={config.pp}"
            + " --context-parallel-size={config.cp}"
            + " --expert-model-parallel-size={config.ep}"
            + " --num-experts={config.num_experts}"
            + " --sequence-parallel"
            + " --use-distributed-optimizer"
            + " --overlap-grad-reduce"
            + " --overlap-param-gather"
            + " --micro-batch-size={config.mbs}"
            + " --global-batch-size={config.GBS}"
            + " --recompute-granularity={config.recomp}"
            + " --bf16"
            + " --num-layers={L}"
            + " --hidden-size={H}"
            + " --num-attention-heads={heads}"
            + " --seq-length={S}"

    ELSE IF framework == "deepspeed":
        cmd = "torchrun"
            + " --nnodes={num_nodes}"
            + " --nproc-per-node={gpus_per_node}"
            + " --rdzv-backend=c10d"
            + " --rdzv-endpoint={master}:{port}"
            + " train.py"
            + " --deepspeed"
            + " --deepspeed_config ds_config.json"
        // ds_config.json generated by GenerateDeepSpeedConfig()

    ELSE IF framework == "fsdp":
        cmd = "torchrun"
            + " --nnodes={num_nodes}"
            + " --nproc-per-node={gpus_per_node}"
            + " --rdzv-backend=c10d"
            + " --rdzv-endpoint={master}:{port}"
            + " train.py"
            + " --fsdp-sharding-strategy=FULL_SHARD"  // or SHARD_GRAD_OP
            + " --fsdp-auto-wrap-policy=transformer_layer"

    RETURN cmd
```

---

## 15. Step-Time Decomposition and Profiling Protocol

### 15.1 Step-Time Breakdown

Every training step decomposes into measurable components:

$$
T_{\text{step}} = T_{\text{data}} + T_{\text{fwd}} + T_{\text{bwd}} + T_{\text{comm}} + T_{\text{optim}} + T_{\text{bubble}} + T_{\text{overhead}}
$$

| Component | Measurement Tool | Expected Fraction | Red Flag |
|---|---|---|---|
| `T_data` | PyTorch Profiler, custom timers | < 1% | > 5% → dataloader bottleneck |
| `T_fwd` | Nsight Systems, PyTorch Profiler | ~15–20% | Significantly > 20% → kernel inefficiency |
| `T_bwd` | Nsight Systems | ~30–40% | >> 40% → recomputation overhead or kernel issue |
| `T_comm` | NCCL/RCCL traces, Nsight Systems | 10–30% (exposed) | > 40% → overlap failure, topology mismatch |
| `T_optim` | PyTorch Profiler | 2–5% | > 10% → optimizer not fused |
| `T_bubble` | Pipeline schedule analysis | `β × T_compute` | > 15% → increase GAS or use interleaved schedule |
| `T_overhead` | CPU profiling | < 2% | > 5% → Python overhead, excessive logging |

### 15.2 Profiling Pseudocode

```
PROCEDURE ProfileTrainingStep(config, num_warmup=5, num_profile=10):
    // Warmup: stabilize CUDA caches and JIT
    FOR i IN [1..num_warmup]:
        RunTrainingStep()

    // Profile with Nsight Systems
    IF NVIDIA:
        nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas
            --cuda-memory-usage=true
            --gpu-metrics-device=all
            --duration=60
            --output=profile.nsys-rep
            training_command

    // Profile with PyTorch Profiler
    WITH torch.profiler.profile(
        activities=[CPU, CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
        record_shapes=True,
        with_stack=True,
        with_flops=True
    ) AS prof:
        FOR step IN [1..num_profile]:
            RunTrainingStep()
            prof.step()

    // Extract metrics
    kernel_times = prof.key_averages()
    IDENTIFY top-10 kernels by CUDA time
    COMPUTE MFU from total FLOPS / (wall_time * peak_FLOPS)
    COMPUTE communication fraction from NCCL kernel times
    COMPUTE bubble fraction from idle gaps in pipeline schedule

    // For AMD (MI300X):
    IF AMD:
        rocprof --hip-trace --hsa-trace --roctx-trace
            --output-directory=profile_dir
            training_command
        // Analyze with Perfetto or rocprof post-processing

    REPORT step_time_decomposition
```

---

## 16. Numerical Robustness Checklist

| Concern | Mitigation | Validation |
|---|---|---|
| BF16 overflow in logits | Fused cross-entropy with chunked log-softmax | Compare loss values against FP32 baseline |
| FP16 gradient underflow | Dynamic loss scaling (DeepSpeed/AMP) | Monitor loss scale: should stabilize, not continuously decrease |
| FP8 range mismatch | Delayed scaling with per-tensor amax history | Verify amax history convergence; check for saturated values |
| Gradient explosion | Gradient clipping (max_norm=1.0) | Monitor gradient norms; sudden spikes indicate instability |
| Softmax numerical instability | Subtract max before exp (stable softmax) | All fused softmax kernels implement this by default |
| Adam epsilon sensitivity | Use eps=1e-8 (default); increase to 1e-6 for FP8 | Monitor optimizer step magnitude |
| Stochastic rounding (FP8) | Enable in Transformer Engine | Compare convergence curves with/without |
| Cross-vendor divergence | Bit-exact seeding, deterministic ops | Compare CUDA vs. ROCm loss curves for 1000 steps |

```
PROCEDURE ValidateNumericalRobustness(config, baseline_config, num_steps=2000):
    // Run baseline (BF16, unfused kernels, single node)
    baseline_losses = Train(baseline_config, num_steps, seed=42)
    baseline_grad_norms = RecordGradNorms(baseline_config, num_steps, seed=42)

    // Run target (FP8, fused kernels, multi-node, full parallelism)
    target_losses = Train(config, num_steps, seed=42)
    target_grad_norms = RecordGradNorms(config, num_steps, seed=42)

    // Parity checks
    loss_rel_diff = ABS(target_losses[-100:].mean() - baseline_losses[-100:].mean()) /
                    baseline_losses[-100:].mean()
    ASSERT loss_rel_diff < 0.01, "Loss divergence > 1%"

    grad_norm_corr = PEARSON_CORRELATION(baseline_grad_norms, target_grad_norms)
    ASSERT grad_norm_corr > 0.95, "Gradient norm correlation too low"

    // Check for NaN/Inf
    ASSERT NO NaN in target_losses
    ASSERT NO Inf in target_grad_norms

    // Check loss scaling stability (FP16)
    IF using FP16:
        ASSERT loss_scale stabilizes within first 500 steps
        ASSERT loss_scale > 1.0 at end of training

    REPORT "Numerical parity: PASS" OR "FAIL with details"
```

---

## 17. Summary — The Three-Step Configuration Protocol

| Step | Objective | Primary Levers | Constraint |
|---|---|---|---|
| **Step 1: Fit into Memory** | `M_peak ≤ 0.95 × HBM` | TP, PP, ZeRO stage, activation recomputation, offload | Hard physical limit |
| **Step 2: Match GBS** | `mbs × dp × gas × S = GBS_tokens` | DP, GAS, mbs, CP | Convergence quality |
| **Step 3: Maximize Throughput** | Maximize MFU, minimize `T_step` | mbs tuning, overlap, kernel fusion, FP8, bucket sizing, pipeline schedule | Diminishing returns |

> **The ordering is strict and non-negotiable.** Memory fit is a hard constraint — violating it causes OOM crashes. GBS matching is a convergence constraint — wrong GBS degrades final model quality. Throughput optimization is an efficiency objective — suboptimal throughput wastes compute budget but does not prevent training.

This protocol, combined with the parallelism strategy table, communication cost models, and framework-specific configurations, provides a complete decision framework for configuring distributed LLM training at any scale, on any hardware, with any framework.

---

*End of Technical Report.*