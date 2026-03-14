

# Megatron-Core + DeepSpeed + FSDP Interoperability: Training, Resume, Resharding, and Conversion

## A Complete Technical Report for Production-Grade Distributed Training Infrastructure

---

## Table of Contents

1. [Framework Architecture and Sharding Semantics](#1-framework-architecture-and-sharding-semantics)
2. [Parallelism Representation Across Frameworks](#2-parallelism-representation-across-frameworks)
3. [Checkpoint Format Specifications](#3-checkpoint-format-specifications)
4. [Interoperability Patterns and Composition](#4-interoperability-patterns-and-composition)
5. [Checkpoint Conversion Pipelines](#5-checkpoint-conversion-pipelines)
6. [Checkpoint Resharding Across Topologies](#6-checkpoint-resharding-across-topologies)
7. [Resume Workflows and Deterministic Recovery](#7-resume-workflows-and-deterministic-recovery)
8. [Mixed-Framework Training Configurations](#8-mixed-framework-training-configurations)
9. [Optimizer State Interoperability](#9-optimizer-state-interoperability)
10. [Memory Budget Analysis Under Interoperability](#10-memory-budget-analysis-under-interoperability)
11. [Communication Group Management](#11-communication-group-management)
12. [Numerical Consistency and Validation](#12-numerical-consistency-and-validation)
13. [Production Automation for Multi-Framework Deployments](#13-production-automation-for-multi-framework-deployments)
14. [Failure Modes and Diagnostic Reference](#14-failure-modes-and-diagnostic-reference)

---

## 1. Framework Architecture and Sharding Semantics

### 1.1 Architectural Overview

Each framework implements distributed training with fundamentally different abstractions for parameter ownership, gradient synchronization, and optimizer state management. Understanding these differences at the tensor-storage level is the prerequisite for all interoperability work.

| Aspect | Megatron-Core | DeepSpeed (ZeRO) | PyTorch FSDP (FSDP2/DTensor) |
|---|---|---|---|
| **Sharding Unit** | Named parameter tensors, partitioned by explicit TP/PP mapping | Flat parameter groups, partitioned across DP ranks by contiguous offset | Per-parameter DTensor with mesh placement |
| **TP Implementation** | Column/Row parallel linear with explicit all-reduce/reduce-scatter hooks | Not native; relies on Megatron or external TP wrapper | DTensor `Shard` placement on TP mesh dimension |
| **PP Implementation** | Explicit stage assignment via layer-to-rank mapping; virtual pipeline interleaving | `PipelineModule` with layer partitioning and custom schedule | Manual model partitioning; external PP schedule |
| **DP Implementation** | Replicated parameters + all-reduce; or distributed optimizer (ZeRO-1 equivalent) | ZeRO-1/2/3: partitioned optimizer/gradient/parameter states | FSDP sharding across DP mesh dimension |
| **Optimizer Ownership** | Each rank owns full optimizer state for its TP-local parameters (standard); or 1/DP-th (distributed optimizer) | Each rank owns 1/DP-th of flat-partitioned optimizer state | Each rank owns optimizer state for its FSDP-sharded parameters |
| **Gradient Format** | BF16/FP32 gradients on TP-local parameters; all-reduced across DP | Gradients accumulated in FP32 or BF16; reduce-scattered in ZeRO-2/3 | Gradients reduce-scattered to owning shard; dtype controlled by `reduce_dtype` |
| **Checkpoint Content** | TP-sharded, PP-staged model + optimizer state per rank | Flat-partitioned model + optimizer state per DP rank + ZeRO metadata | DTensor-based per-rank shards + `ShardedTensor` / `DTensor` metadata |

### 1.2 Parameter Identity: The Core Interoperability Challenge

The fundamental interoperability challenge is that **each framework assigns different identities and storage layouts to the same logical parameter**:

**Example: A linear layer weight $W \in \mathbb{R}^{h \times 4h}$ (MLP gate projection) with $t=4$ TP degree, $d=8$ DP degree:**

| Framework | What Rank 0 Stores | Shape on Rank 0 | Key Name |
|---|---|---|---|
| Megatron-Core | Column-parallel shard | $(h, h)$ i.e., $(h, 4h/t)$ | `model.layers.0.mlp.gate_proj.weight` |
| DeepSpeed ZeRO-3 | Flat slice of full param | $(h \times 4h / d,)$ = flat | `module.layers.0.mlp.gate_proj.weight` (flat partition) |
| FSDP | DTensor shard on DP dim | $(h, 4h / d)$ or flat | `layers.0.mlp.gate_proj.weight` (with DTensor metadata) |
| Megatron + DeepSpeed ZeRO-1 | Column-parallel shard (full on each DP rank) | $(h, h)$ | `module.model.layers.0.mlp.gate_proj.weight` |

> **Critical observation:** Megatron-Core shards along a **semantically meaningful dimension** (columns or rows of the weight matrix) determined by the parallelism strategy. DeepSpeed ZeRO-3 shards along a **flat contiguous offset** with no awareness of tensor semantics. FSDP shards via DTensor placement which can be either flat or dimension-aware. These three sharding semantics are **mutually incompatible** at the storage level and require explicit conversion logic.

### 1.3 Formal Sharding Representation

For a parameter $W$ with shape $(d_0, d_1, \ldots, d_n)$:

**Megatron-Core TP Sharding:**

$$
W_{\text{rank}_i}^{\text{Mcore}} = W[\ldots, \; i \cdot \frac{d_k}{t} : (i+1) \cdot \frac{d_k}{t}, \; \ldots]
$$

where $k$ is the sharding dimension (0 for row-parallel, last for column-parallel), and $t$ is the TP degree.

**DeepSpeed ZeRO-3 Flat Partitioning:**

$$
W_{\text{rank}_i}^{\text{ZeRO3}} = \text{flatten}(W)\left[i \cdot \left\lceil\frac{|W|}{d}\right\rceil : (i+1) \cdot \left\lceil\frac{|W|}{d}\right\rceil\right]
$$

where $|W| = \prod_{j} d_j$ is the total element count and $d$ is the DP world size.

**FSDP DTensor Sharding:**

$$
W_{\text{rank}_i}^{\text{FSDP}} = W[\ldots, \; i \cdot \frac{d_k}{d_{\text{mesh}}} : (i+1) \cdot \frac{d_k}{d_{\text{mesh}}}, \; \ldots]
$$

where $d_{\text{mesh}}$ is the mesh dimension size for the shard placement, and $k$ is the shard dimension (typically 0 for FSDP flat sharding, or the TP dimension for 2D DTensor).

---

## 2. Parallelism Representation Across Frameworks

### 2.1 Process Group Topology

Each framework constructs its own set of process groups (communication groups) for collective operations. When composing frameworks, process group compatibility is essential.

**Table: Process Group Construction**

| Group Type | Megatron-Core | DeepSpeed | FSDP |
|---|---|---|---|
| **TP Group** | `mpu.get_tensor_model_parallel_group()` — ranks sharing NVLink within a node | Not natively created; uses Megatron's groups if composed | DTensor `DeviceMesh` TP sub-mesh |
| **PP Group** | `mpu.get_pipeline_model_parallel_group()` — ranks across pipeline stages | `PipelineModule` internal group; or Megatron's | Not natively managed; external schedule |
| **DP Group** | `mpu.get_data_parallel_group()` — ranks with same TP+PP position | `deepspeed.comm` default group; or Megatron's if composed | FSDP's sharding group from DeviceMesh |
| **CP Group** | `mpu.get_context_parallel_group()` — ranks sharing context partition | Not natively supported | External implementation |
| **EP Group** | `mpu.get_expert_model_parallel_group()` — ranks sharing expert assignments | `deepspeed.moe` expert group | External implementation |

**Pseudocode 1: Process Group Compatibility Verification**

```
PROCEDURE VERIFY_PROCESS_GROUP_COMPATIBILITY(megatron_groups, deepspeed_groups, fsdp_mesh):
    // Verify that DP groups are consistent across frameworks
    
    // Extract DP group membership from each framework
    mcore_dp_ranks ← GET_RANKS(megatron_groups.data_parallel_group)
    ds_dp_ranks ← GET_RANKS(deepspeed_groups.data_parallel_group)
    
    IF fsdp_mesh IS NOT NONE THEN
        fsdp_dp_ranks ← GET_RANKS(fsdp_mesh.get_group("dp"))
    END IF
    
    // All DP groups must contain identical rank sets
    IF mcore_dp_ranks ≠ ds_dp_ranks THEN
        ERROR "DP group mismatch: Megatron and DeepSpeed see different DP ranks"
        LOG "Megatron DP: " + mcore_dp_ranks
        LOG "DeepSpeed DP: " + ds_dp_ranks
        // Root cause: world-size factorization disagreement
        // Fix: ensure t × p × d is consistent in both configs
    END IF
    
    // Verify TP group is only managed by one framework
    IF megatron_groups.has_tp AND deepspeed_groups.has_tp THEN
        WARN "Both frameworks managing TP groups — potential collective conflict"
        // Resolution: let Megatron-Core own TP; DeepSpeed operates on DP only
    END IF
    
    // Verify no overlapping collective calls on same group
    // This is the primary source of deadlocks in mixed-framework setups
    FOR each group IN ALL_GROUPS:
        collective_owners ← FRAMEWORKS_USING(group)
        IF |collective_owners| > 1 THEN
            VERIFY_SERIALIZED_ACCESS(group, collective_owners)
        END IF
    END FOR
    
    RETURN COMPATIBLE
```

### 2.2 Parallelism Factorization Mapping

Given world size $W$ and the factorization:

$$
W = d \times t \times p \times c \times e
$$

Each framework must agree on the same factorization. The rank-to-role mapping must be identical:

**Pseudocode 2: Unified Rank Mapping**

```
PROCEDURE COMPUTE_RANK_ROLES(global_rank, W, t, p, d, c, e):
    // Megatron-Core ordering: TP is innermost, then PP, then DP
    // [TP0,TP1,...,TPt-1] form innermost group
    // PP stages stride by t
    // DP replicas stride by t × p
    
    tp_rank ← global_rank MOD t
    pp_rank ← (global_rank / t) MOD p
    
    // If CP is used, it nests between PP and DP
    cp_rank ← (global_rank / (t × p)) MOD c
    
    // If EP is used, it nests within DP
    ep_rank ← (global_rank / (t × p × c)) MOD e
    
    dp_rank ← global_rank / (t × p × c × e)
    
    // Verify consistency
    reconstructed ← dp_rank × (t × p × c × e) + ep_rank × (t × p × c) + 
                     cp_rank × (t × p) + pp_rank × t + tp_rank
    ASSERT reconstructed = global_rank
    
    RETURN {tp_rank, pp_rank, dp_rank, cp_rank, ep_rank}
```

> **Interoperability requirement:** When DeepSpeed is composed with Megatron-Core, DeepSpeed must be initialized **after** Megatron-Core's process groups are created, and must use the **same DP group** that Megatron-Core computed. Failure to do this results in gradient synchronization across wrong ranks, producing silent convergence failure.

---

## 3. Checkpoint Format Specifications

### 3.1 Megatron-Core Checkpoint Format

Megatron-Core saves one checkpoint file per rank, containing the TP-sharded, PP-staged model and optimizer state.

**Directory structure:**

```
iter_XXXXXXX/
├── mp_rank_TT_PPP/        # One directory per (TP_rank, PP_rank) pair
│   └── model_optim_rng.pt  # Contains model, optimizer, RNG states
├── mp_rank_00_000/
│   └── model_optim_rng.pt
├── mp_rank_01_000/
│   └── model_optim_rng.pt
├── ...
└── latest_checkpointed_iteration.txt
```

**Contents of each `model_optim_rng.pt`:**

| Key | Content | Shape Context |
|---|---|---|
| `model` | `OrderedDict` of TP-sharded parameter tensors | Each tensor is shape `(original_dim / t, ...)` on TP shard dimension |
| `optimizer` | Optimizer state dict; states keyed by parameter index | FP32 master weights, momentum $m_t$, variance $v_t$ — all TP-sharded |
| `rng_state` | CUDA, CPU, numpy RNG states | Per-rank deterministic state |
| `iteration` | Current training iteration | Scalar |
| `args` | Training arguments (TP, PP, DP, etc.) | Configuration metadata |
| `checkpoint_version` | Format version number | Scalar |

**Key naming convention (Megatron-Core):**

```
model.language_model.encoder.layers.{L}.self_attention.query_key_value.weight
model.language_model.encoder.layers.{L}.mlp.dense_h_to_4h.weight
model.language_model.encoder.layers.{L}.input_layernorm.weight
model.language_model.embedding.word_embeddings.weight
```

> **Megatron-Core Distributed Optimizer:** When using the Megatron-Core distributed optimizer (equivalent to ZeRO-1), the optimizer states are further sharded across DP ranks. Each file then contains only $1/d$-th of the optimizer state for the TP-local parameters. The file naming remains the same, but `model_optim_rng.pt` is saved per-rank (including DP rank), not just per (TP, PP) pair.

### 3.2 DeepSpeed Checkpoint Format

DeepSpeed uses a different directory structure depending on the ZeRO stage.

**ZeRO-1/2 Directory Structure:**

```
global_step_XXXXXXX/
├── mp_rank_TT_model_states.pt    # Model weights (TP-sharded, full per DP rank)
├── zero_pp_rank_DD_mp_rank_TT_optim_states.pt  # Optimizer shard per DP rank
├── ...
└── latest
```

**ZeRO-3 Directory Structure:**

```
global_step_XXXXXXX/
├── zero_pp_rank_DD_mp_rank_TT/
│   └── fp32_flat/                 # Flat FP32 optimizer state partition
│       ├── 000.pt                 # Partition file
│       └── ...
├── bf16_zero_pp_rank_DD_mp_rank_TT_model_states.pt  # BF16 model state shard
├── zero_to_fp32.py               # Conversion script
├── mp_rank_TT_model_states.pt    # (ZeRO-1/2 only) Full model per TP rank
└── latest
```

**ZeRO-3 Flat Partitioning Metadata:**

DeepSpeed ZeRO-3 stores a critical metadata structure that maps flat offsets back to named parameters:

| Metadata Key | Content | Purpose |
|---|---|---|
| `param_shapes` | Dict mapping param name → original shape | Reconstruct tensor shapes from flat buffer |
| `param_offsets` | Dict mapping param name → (start, end) in flat buffer | Locate parameter within flat partition |
| `partition_count` | Number of DP partitions | Needed for reassembly |
| `numel_per_partition` | Elements per partition (with padding) | Handles uneven division |
| `ds_version` | DeepSpeed version | Format compatibility |

### 3.3 FSDP Checkpoint Format

FSDP (particularly FSDP2 with DTensor) uses PyTorch's `torch.distributed.checkpoint` (DCP) system.

**DCP Directory Structure:**

```
checkpoint_XXXXXXX/
├── __0_0.distcp                  # Shard file for rank 0
├── __1_0.distcp                  # Shard file for rank 1
├── ...
├── .metadata                     # Global metadata (ShardedTensor placements)
└── .metadata.json                # Human-readable metadata
```

**`.metadata` Content:**

| Field | Content | Purpose |
|---|---|---|
| `state_dict_metadata` | Per-key metadata: shape, dtype, placement | Describes how each tensor is sharded across files |
| `planner_data` | Shard-to-file mapping | Maps each tensor shard to its storage location |
| `storage_data` | File offsets and sizes | Byte-level addressing within shard files |

**FSDP2/DTensor Placement Information:**

Each parameter's metadata includes its DTensor placement specification:

```
{
  "key": "model.layers.0.mlp.gate_proj.weight",
  "global_shape": [h, 4h],
  "placements": [Shard(0), Replicate()],  # Shard on DP dim, Replicate on TP dim (or vice versa)
  "mesh_shape": [dp_size, tp_size],
  "local_shape": [h / dp_size, 4h],       # Per-rank shard shape
  "dtype": "torch.bfloat16"
}
```

### 3.4 Checkpoint Format Comparison Summary

| Property | Megatron-Core | DeepSpeed | FSDP/DCP |
|---|---|---|---|
| **File format** | PyTorch `torch.save` (pickle + zip) | PyTorch `torch.save` (pickle + zip) | `torch.distributed.checkpoint` (custom binary + metadata) |
| **Naming** | `mp_rank_TT_PPP/model_optim_rng.pt` | `zero_pp_rank_DD_mp_rank_TT_*.pt` | `__R_0.distcp` + `.metadata` |
| **Model state key** | `model` | `module` (with `module.` prefix from DDP wrapper) | Direct parameter names |
| **Optimizer state** | Keyed by param index; TP-sharded | Flat-partitioned by DP rank; keyed by flat index | Keyed by FQN; DTensor-sharded |
| **Resharding** | Requires conversion script | Requires `zero_to_fp32.py` or custom | Native resharding via DCP planner |
| **TP awareness** | Yes (sharded by TP dim) | No (flat partitioning ignores TP) | Yes (DTensor placement encodes TP) |
| **PP awareness** | Yes (separate files per stage) | Yes (separate files per stage) | Depends on implementation |
| **Cross-framework loadable** | No (without conversion) | No (without conversion) | No (without conversion) |

---

## 4. Interoperability Patterns and Composition

### 4.1 Pattern A: Megatron-Core + DeepSpeed ZeRO (Most Common Production Pattern)

This is the most widely deployed interoperability pattern, used by many production LLM training runs. Megatron-Core handles TP and PP; DeepSpeed handles optimizer sharding (ZeRO-1/2) or full parameter sharding (ZeRO-3) across the DP dimension.

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                   Megatron-Core                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  TP Group    │  │  PP Group   │  │  CP Group   │ │
│  │ (NVLink)     │  │ (IB/NVLink) │  │ (NVLink/IB) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│         ↕                                            │
│  Process Group Creation & Rank Mapping               │
│         ↕                                            │
├─────────────────────────────────────────────────────┤
│                   DeepSpeed                          │
│  ┌─────────────┐  ┌─────────────┐                   │
│  │  ZeRO Engine │  │  DP Group   │                   │
│  │ (Optim Shard)│  │ (IB/RoCE)  │                   │
│  └─────────────┘  └─────────────┘                   │
│         ↕                                            │
│  Gradient Reduction / Parameter Sharding             │
└─────────────────────────────────────────────────────┘
```

**Initialization Sequence (Critical Order):**

**Pseudocode 3: Megatron-Core + DeepSpeed Initialization**

```
PROCEDURE INITIALIZE_MCORE_DEEPSPEED(config):
    // ═══════════════════════════════════════════
    // STEP 1: Initialize Megatron-Core parallelism FIRST
    // This creates TP, PP, DP, CP, EP process groups
    // ═══════════════════════════════════════════
    CALL megatron.initialize.initialize_megatron(config)
    // Internally calls:
    //   torch.distributed.init_process_group(backend="nccl")
    //   mpu.initialize_model_parallel(
    //       tensor_model_parallel_size = t,
    //       pipeline_model_parallel_size = p,
    //       context_parallel_size = c,
    //       expert_model_parallel_size = e
    //   )
    
    // ═══════════════════════════════════════════
    // STEP 2: Build model using Megatron-Core modules
    // Model uses TP-aware layers (ColumnParallelLinear, RowParallelLinear)
    // ═══════════════════════════════════════════
    model ← BUILD_MEGATRON_MODEL(config)
    // model is already TP-sharded: each rank holds (h, 4h/t) for column-parallel layers
    
    // ═══════════════════════════════════════════
    // STEP 3: Extract Megatron's DP group for DeepSpeed
    // CRITICAL: DeepSpeed must use the SAME DP group
    // ═══════════════════════════════════════════
    dp_group ← mpu.get_data_parallel_group()
    dp_world_size ← mpu.get_data_parallel_world_size()
    dp_rank ← mpu.get_data_parallel_rank()
    
    // ═══════════════════════════════════════════
    // STEP 4: Configure DeepSpeed engine with Megatron's DP group
    // ═══════════════════════════════════════════
    ds_config ← {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_acc_steps,
        "zero_optimization": {
            "stage": config.zero_stage,
            "reduce_bucket_size": 5e8,          // 500M elements
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True
        },
        "bf16": {"enabled": True},
        "steps_per_print": 100
    }
    
    // ═══════════════════════════════════════════
    // STEP 5: Initialize DeepSpeed engine
    // Pass Megatron's DP group explicitly
    // ═══════════════════════════════════════════
    model_engine, optimizer, _, scheduler ← deepspeed.initialize(
        model = model,
        config = ds_config,
        model_parameters = model.parameters(),
        mpu = mpu,                    // Pass Megatron's MPU for group discovery
        dist_init_required = False    // Already initialized by Megatron
    )
    
    // ═══════════════════════════════════════════
    // STEP 6: Verify group consistency
    // ═══════════════════════════════════════════
    ASSERT model_engine.dp_world_size = dp_world_size
    ASSERT model_engine.dp_process_group = dp_group
    
    // ═══════════════════════════════════════════
    // STEP 7: If using ZeRO-3, verify parameter partitioning
    // ═══════════════════════════════════════════
    IF config.zero_stage = 3 THEN
        FOR each param IN model.parameters():
            // ZeRO-3 replaces param.data with a flat partition
            ASSERT param.ds_numel = param.original_numel / dp_world_size
            // The TP-sharded parameter is FURTHER flat-partitioned across DP
            // Rank 0 (DP) holds flat_slice[0 : numel/dp] of the TP-shard
        END FOR
    END IF
    
    RETURN model_engine
```

> **Key insight for ZeRO Stage selection with Megatron-Core:**
> - **ZeRO-1:** Partitions optimizer states across DP ranks. Model parameters and gradients are replicated. Most compatible with Megatron-Core; minimal interaction complexity.
> - **ZeRO-2:** Adds gradient partitioning. Gradients are reduce-scattered (each DP rank gets 1/d-th of gradients). Compatible but requires careful bucket sizing to not conflict with Megatron's gradient reduce hooks.
> - **ZeRO-3:** Partitions parameters themselves. Each DP rank holds 1/d-th of each (already TP-sharded) parameter. Requires all-gather before every forward/backward use. **Most complex interaction:** Megatron's TP collectives operate on the all-gathered full TP-shard, but ZeRO-3's all-gather is per-parameter across DP. Ordering of all-gathers must be deterministic.

### 4.2 Pattern B: Megatron-Core + FSDP (Emerging Pattern)

This pattern uses Megatron-Core's TP and PP with FSDP replacing DDP for the DP dimension. This is increasingly relevant with PyTorch's native DTensor-based TP+FSDP composition.

**Pseudocode 4: Megatron-Core + FSDP Composition**

```
PROCEDURE INITIALIZE_MCORE_FSDP(config):
    // ═══════════════════════════════════════════
    // STEP 1: Initialize base distributed environment
    // ═══════════════════════════════════════════
    torch.distributed.init_process_group(backend="nccl")
    
    // ═══════════════════════════════════════════
    // STEP 2: Create 2D DeviceMesh for TP + DP(FSDP)
    // ═══════════════════════════════════════════
    // For W GPUs with TP=t:
    //   DP dimension = W / t
    //   TP dimension = t
    mesh_2d ← DeviceMesh("cuda", shape=[W/t, t], dim_names=["dp", "tp"])
    
    // ═══════════════════════════════════════════
    // STEP 3: Build model with Megatron-Core TP-aware layers
    // Each parameter is initialized as 2D DTensor:
    //   - Shard on "dp" dimension (FSDP sharding)
    //   - Shard on "tp" dimension (TP sharding)
    // ═══════════════════════════════════════════
    model ← BUILD_MEGATRON_MODEL(config)
    
    // Apply TP via DTensor placements
    FOR each layer IN model.layers:
        // Column-parallel: shard weight on dim=-1 across TP mesh
        layer.qkv.weight ← distribute_tensor(
            layer.qkv.weight, mesh_2d["tp"], placements=[Shard(-1)]
        )
        // Row-parallel: shard weight on dim=0 across TP mesh
        layer.out_proj.weight ← distribute_tensor(
            layer.out_proj.weight, mesh_2d["tp"], placements=[Shard(0)]
        )
    END FOR
    
    // ═══════════════════════════════════════════
    // STEP 4: Apply FSDP on the DP dimension
    // FSDP shards parameters across the "dp" sub-mesh
    // ═══════════════════════════════════════════
    fsdp_model ← FSDP(
        model,
        mesh = mesh_2d["dp"],
        sharding_strategy = FULL_SHARD,     // ZeRO-3 equivalent
        mixed_precision = MixedPrecision(
            param_dtype = torch.bfloat16,
            reduce_dtype = torch.float32,
            buffer_dtype = torch.bfloat16
        ),
        use_orig_params = True  // Required for compatibility with Megatron param naming
    )
    
    // ═══════════════════════════════════════════
    // STEP 5: Handle PP separately (if used)
    // FSDP wraps each PP stage independently
    // ═══════════════════════════════════════════
    IF config.pp > 1 THEN
        // Extend mesh to 3D: [dp, pp, tp]
        // Or handle PP with manual send/recv between stages
        // Each stage is independently FSDP-wrapped
        stages ← PARTITION_MODEL_INTO_STAGES(fsdp_model, config.pp)
        local_stage ← stages[pp_rank]
        local_stage ← FSDP(local_stage, mesh=mesh_2d["dp"], ...)
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 6: Verify parameter sharding
    // Each param is now a 2D DTensor with placements:
    //   [Shard(0) on DP mesh, Shard(k) on TP mesh]
    // ═══════════════════════════════════════════
    FOR each name, param IN model.named_parameters():
        ASSERT isinstance(param, DTensor)
        ASSERT param.placements[0] == Shard(0)  // FSDP shard on DP
        // TP placement varies by layer type
        LOG name + ": " + param.placements + " local_shape=" + param._local_tensor.shape
    END FOR
    
    RETURN fsdp_model, mesh_2d
```

**Key differences from Pattern A:**

| Aspect | Megatron + DeepSpeed | Megatron + FSDP |
|---|---|---|
| Parameter representation | Native tensors + DeepSpeed flat partitioning | DTensor with explicit placement metadata |
| Checkpoint format | DeepSpeed ZeRO format | DCP format with DTensor metadata |
| Resharding support | Manual (requires conversion scripts) | Native (DCP planner handles resharding) |
| `torch.compile` compatibility | Limited (DeepSpeed hooks may break tracing) | Full (DTensor is compiler-friendly) |
| Overlap optimization | DeepSpeed's built-in overlap scheduler | SimpleFSDP / compiler-driven bucketing and reordering |
| FP8 integration | Via Transformer Engine (separate) | Via TE + DTensor FP8 casting |

### 4.3 Pattern C: DeepSpeed ZeRO-3 + FSDP (Rare, Anti-Pattern)

Composing DeepSpeed ZeRO-3 and FSDP simultaneously is an **anti-pattern** because both attempt to shard parameters across the same DP dimension. This creates:

- **Double sharding:** Parameters would be flat-partitioned by ZeRO-3 and then further sharded by FSDP, resulting in 1/(d²)-th of each parameter per rank—far beyond necessary.
- **Conflicting all-gather semantics:** Both frameworks issue all-gather before forward computation, but neither is aware of the other's shard state.
- **Deadlocks:** ZeRO-3's parameter fetch hooks and FSDP's all-gather occur asynchronously and may interleave incorrectly.

> **Recommendation:** Never compose DeepSpeed ZeRO-3 with FSDP. Choose one framework for DP-dimension sharding. If both are present in the codebase, ensure DeepSpeed handles optimizer sharding (ZeRO-1) while FSDP handles parameter sharding, or vice versa—but not both at Stage 3.

### 4.4 Pattern D: Megatron-Core + DeepSpeed + FSDP Triple Composition

In rare cases, all three frameworks may coexist:

- **Megatron-Core:** Manages TP, PP, and provides model architecture
- **DeepSpeed:** Provides ZeRO-1 optimizer sharding and infrastructure (e.g., elastic training, profiling hooks)
- **FSDP:** Wraps individual pipeline stages for intra-stage parameter sharding

**This configuration is highly complex and requires:**

**Pseudocode 5: Triple Framework Initialization**

```
PROCEDURE INITIALIZE_TRIPLE_COMPOSITION(config):
    // ═══════════════════════════════════════════
    // STEP 1: Megatron-Core initializes all process groups
    // ═══════════════════════════════════════════
    CALL megatron.initialize.initialize_megatron(config)
    
    // ═══════════════════════════════════════════
    // STEP 2: Build model with Megatron-Core (TP + PP)
    // ═══════════════════════════════════════════
    model ← BUILD_MEGATRON_MODEL(config)
    local_stage ← GET_LOCAL_PP_STAGE(model, pp_rank)
    
    // ═══════════════════════════════════════════
    // STEP 3: Apply FSDP to local pipeline stage
    // FSDP handles parameter sharding within the stage
    // ═══════════════════════════════════════════
    dp_mesh ← DeviceMesh("cuda", 
        ranks=mpu.get_data_parallel_group_ranks(),
        dim_names=["dp"])
    
    fsdp_stage ← FSDP(
        local_stage,
        mesh = dp_mesh,
        sharding_strategy = SHARD_GRAD_OP,  // ZeRO-2 equivalent; avoid ZeRO-3 overlap with DS
        use_orig_params = True
    )
    
    // ═══════════════════════════════════════════
    // STEP 4: Initialize DeepSpeed with ZeRO-1 ONLY
    // ZeRO-1 for optimizer sharding; FSDP handles grad sharding (ZeRO-2)
    // ═══════════════════════════════════════════
    ds_config ← {
        "zero_optimization": {"stage": 1},
        // Do NOT enable ZeRO-2/3 — FSDP already handles gradient/param sharding
        "bf16": {"enabled": True}
    }
    
    model_engine, optimizer, _, _ ← deepspeed.initialize(
        model = fsdp_stage,
        config = ds_config,
        mpu = mpu,
        dist_init_required = False
    )
    
    // ═══════════════════════════════════════════
    // VERIFY: No double-sharding
    // ═══════════════════════════════════════════
    FOR each param IN model_engine.parameters():
        // FSDP has sharded the gradient (SHARD_GRAD_OP)
        // DeepSpeed ZeRO-1 has sharded the optimizer state
        // Parameter itself is NOT flat-partitioned by DeepSpeed (ZeRO-1 doesn't shard params)
        // This is the safe composition
        ASSERT NOT hasattr(param, 'ds_numel')  // No ZeRO-3 flat partitioning
    END FOR
    
    RETURN model_engine
```

---

## 5. Checkpoint Conversion Pipelines

### 5.1 Conversion Matrix

**Table: All Checkpoint Conversion Paths**

| Source → Target | Feasibility | Complexity | Key Challenge |
|---|---|---|---|
| Megatron-Core → DeepSpeed | ✅ Feasible | Medium | Key renaming + flat partitioning of optimizer |
| DeepSpeed → Megatron-Core | ✅ Feasible | Medium | Flat-to-shaped reconstruction + key renaming |
| Megatron-Core → FSDP/DCP | ✅ Feasible | Medium-High | TP shard → DTensor placement metadata generation |
| FSDP/DCP → Megatron-Core | ✅ Feasible | Medium | DTensor → TP-shard extraction + key renaming |
| DeepSpeed → FSDP/DCP | ✅ Feasible | High | Flat partition → DTensor; no semantic shard info in ZeRO-3 |
| FSDP/DCP → DeepSpeed | ✅ Feasible | Medium-High | DTensor → flat partition + metadata generation |
| Megatron-Core → HuggingFace | ✅ Feasible | Low-Medium | TP shard concatenation + key mapping |
| HuggingFace → Megatron-Core | ✅ Feasible | Low-Medium | TP shard splitting + key mapping |
| DeepSpeed → HuggingFace | ✅ Feasible | Medium | Flat reconstruction + key mapping |
| Any → Consolidated FP32 | ✅ Feasible | Varies | Universal target for evaluation/export |

### 5.2 Megatron-Core to DeepSpeed Conversion

**Pseudocode 6: Megatron-Core → DeepSpeed Checkpoint Conversion**

```
PROCEDURE CONVERT_MCORE_TO_DEEPSPEED(mcore_ckpt_dir, target_ds_dir, 
                                       source_tp, source_pp,
                                       target_tp, target_pp, target_dp,
                                       target_zero_stage):
    // ═══════════════════════════════════════════
    // PHASE 1: Load all Megatron-Core shards
    // ═══════════════════════════════════════════
    mcore_shards ← {}
    FOR tp_rank IN 0..source_tp-1:
        FOR pp_rank IN 0..source_pp-1:
            path ← mcore_ckpt_dir + "/mp_rank_{tp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt"
            mcore_shards[(tp_rank, pp_rank)] ← LOAD(path, map_location="cpu")
        END FOR
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 2: Reconstruct full (unsharded) model state
    // Concatenate TP shards; merge PP stages
    // ═══════════════════════════════════════════
    full_state_dict ← {}
    
    // Get all unique parameter names (removing TP-specific prefixes)
    all_param_names ← EXTRACT_PARAM_NAMES(mcore_shards[(0, 0)]["model"])
    
    FOR each param_name IN all_param_names:
        // Determine which PP stage owns this parameter
        pp_stage ← DETERMINE_PP_STAGE(param_name, source_pp)
        
        // Collect TP shards for this parameter
        tp_shards ← []
        FOR tp_rank IN 0..source_tp-1:
            shard ← mcore_shards[(tp_rank, pp_stage)]["model"][param_name]
            tp_shards.append(shard)
        END FOR
        
        // Determine TP shard dimension and concatenation strategy
        shard_dim, shard_type ← GET_SHARD_INFO(param_name)
        // shard_type: "column" → concat along last dim
        //             "row" → concat along first dim
        //             "none" → replicated (e.g., LayerNorm)
        
        IF shard_type = "column" THEN
            full_param ← CONCATENATE(tp_shards, dim=-1)
        ELSE IF shard_type = "row" THEN
            full_param ← CONCATENATE(tp_shards, dim=0)
        ELSE  // "none" — replicated parameter
            full_param ← tp_shards[0]
            // Verify all TP ranks have identical copies
            FOR i IN 1..source_tp-1:
                ASSERT ALLCLOSE(tp_shards[0], tp_shards[i])
            END FOR
        END IF
        
        // Map Megatron key names to DeepSpeed key names
        ds_key ← MAP_MCORE_KEY_TO_DS_KEY(param_name)
        // Example mappings:
        //   "language_model.encoder.layers.{L}.self_attention.query_key_value.weight"
        //   → "module.model.layers.{L}.self_attn.qkv_proj.weight"
        // (Exact mapping depends on model architecture)
        
        full_state_dict[ds_key] ← full_param
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 3: Re-shard for target TP degree
    // ═══════════════════════════════════════════
    IF target_tp ≠ 1 THEN
        FOR each ds_key IN full_state_dict:
            shard_dim, shard_type ← GET_SHARD_INFO_DS(ds_key)
            IF shard_type ≠ "none" THEN
                full_param ← full_state_dict[ds_key]
                new_shards ← SPLIT(full_param, target_tp, dim=shard_dim)
                // Store only the shard for this TP rank
                full_state_dict[ds_key] ← new_shards[target_tp_rank]
            END IF
        END FOR
    END IF
    
    // ═══════════════════════════════════════════
    // PHASE 4: Create DeepSpeed ZeRO checkpoint structure
    // ═══════════════════════════════════════════
    IF target_zero_stage ≤ 2 THEN
        // Model weights: full per DP rank (TP-sharded if target_tp > 1)
        FOR tp_rank IN 0..target_tp-1:
            SAVE model_state AS mp_rank_{tp_rank}_model_states.pt
        END FOR
        
        // Optimizer states: partitioned across DP ranks
        full_optimizer_state ← RECONSTRUCT_OPTIMIZER_STATE(mcore_shards)
        // Re-shard for target TP, then flat-partition for target DP
        FOR dp_rank IN 0..target_dp-1:
            optim_partition ← FLAT_PARTITION(full_optimizer_state, dp_rank, target_dp)
            SAVE optim_partition AS zero_pp_rank_{dp_rank}_mp_rank_{tp_rank}_optim_states.pt
        END FOR
        
    ELSE IF target_zero_stage = 3 THEN
        // Both model and optimizer are flat-partitioned
        FOR dp_rank IN 0..target_dp-1:
            FOR tp_rank IN 0..target_tp-1:
                // Flatten all TP-local parameters into a single buffer
                flat_buffer ← FLATTEN_ALL_PARAMS(full_state_dict, tp_rank)
                
                // Partition flat buffer across DP
                partition ← flat_buffer[dp_rank * partition_size : (dp_rank+1) * partition_size]
                
                // Save with ZeRO-3 metadata
                SAVE partition WITH METADATA(param_shapes, param_offsets, partition_count)
                    AS bf16_zero_pp_rank_{dp_rank}_mp_rank_{tp_rank}_model_states.pt
            END FOR
        END FOR
    END IF
    
    // ═══════════════════════════════════════════
    // PHASE 5: Save metadata and conversion log
    // ═══════════════════════════════════════════
    SAVE {
        "source_framework": "megatron-core",
        "target_framework": "deepspeed",
        "source_tp": source_tp, "source_pp": source_pp,
        "target_tp": target_tp, "target_pp": target_pp, "target_dp": target_dp,
        "target_zero_stage": target_zero_stage,
        "param_count": TOTAL_PARAMS(full_state_dict),
        "conversion_timestamp": NOW()
    } AS conversion_metadata.json
```

### 5.3 DeepSpeed to Megatron-Core Conversion

**Pseudocode 7: DeepSpeed → Megatron-Core Checkpoint Conversion**

```
PROCEDURE CONVERT_DS_TO_MCORE(ds_ckpt_dir, target_mcore_dir,
                                source_tp, source_dp, source_zero_stage,
                                target_tp, target_pp):
    // ═══════════════════════════════════════════
    // PHASE 1: Reconstruct full model state from DeepSpeed
    // ═══════════════════════════════════════════
    
    IF source_zero_stage ≤ 2 THEN
        // Model weights are full per TP rank
        // Load from any DP rank's model state (they're identical)
        model_state ← {}
        FOR tp_rank IN 0..source_tp-1:
            shard ← LOAD(ds_ckpt_dir + "/mp_rank_{tp_rank}_model_states.pt")
            // Strip "module." prefix added by DDP/DeepSpeed wrapper
            FOR key, value IN shard["module"].items():
                clean_key ← STRIP_PREFIX(key, "module.")
                IF source_tp > 1 THEN
                    model_state[(clean_key, tp_rank)] ← value
                ELSE
                    model_state[clean_key] ← value
                END IF
            END FOR
        END FOR
        
    ELSE IF source_zero_stage = 3 THEN
        // Parameters are flat-partitioned across DP ranks
        // Must collect ALL DP partitions to reconstruct full parameters
        model_state ← {}
        
        FOR tp_rank IN 0..source_tp-1:
            flat_partitions ← []
            param_metadata ← NONE
            
            FOR dp_rank IN 0..source_dp-1:
                partition_data ← LOAD(ds_ckpt_dir + 
                    "/bf16_zero_pp_rank_{dp_rank}_mp_rank_{tp_rank}_model_states.pt")
                flat_partitions.append(partition_data["flat_params"])
                IF param_metadata IS NONE THEN
                    param_metadata ← partition_data["param_metadata"]
                END IF
            END FOR
            
            // Concatenate flat partitions to get full flat buffer
            full_flat ← CONCATENATE(flat_partitions, dim=0)
            
            // Unflatten using metadata
            FOR param_name, meta IN param_metadata.items():
                start ← meta["offset"]
                numel ← meta["numel"]
                shape ← meta["shape"]
                
                // Extract parameter from flat buffer and reshape
                param_tensor ← full_flat[start : start + numel].reshape(shape)
                model_state[(param_name, tp_rank)] ← param_tensor
            END FOR
        END FOR
    END IF
    
    // ═══════════════════════════════════════════
    // PHASE 2: Concatenate TP shards to get full parameters
    // ═══════════════════════════════════════════
    full_params ← {}
    unique_names ← GET_UNIQUE_PARAM_NAMES(model_state)
    
    FOR each param_name IN unique_names:
        IF source_tp > 1 THEN
            tp_shards ← [model_state[(param_name, tp)] FOR tp IN 0..source_tp-1]
            shard_dim, shard_type ← GET_SHARD_INFO_DS(param_name)
            
            IF shard_type = "column" THEN
                full_params[param_name] ← CONCATENATE(tp_shards, dim=-1)
            ELSE IF shard_type = "row" THEN
                full_params[param_name] ← CONCATENATE(tp_shards, dim=0)
            ELSE
                full_params[param_name] ← tp_shards[0]
            END IF
        ELSE
            full_params[param_name] ← model_state[param_name]
        END IF
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 3: Re-shard for target Megatron-Core topology
    // ═══════════════════════════════════════════
    FOR target_tp_rank IN 0..target_tp-1:
        FOR target_pp_rank IN 0..target_pp-1:
            mcore_state ← {}
            
            // Determine which layers belong to this PP stage
            layers_for_stage ← ASSIGN_LAYERS_TO_STAGE(total_layers, target_pp, target_pp_rank)
            
            FOR each param_name IN full_params:
                // Check if this param belongs to this PP stage
                IF NOT PARAM_IN_STAGE(param_name, layers_for_stage) THEN
                    CONTINUE
                END IF
                
                // Map DS key to Megatron-Core key
                mcore_key ← MAP_DS_KEY_TO_MCORE_KEY(param_name)
                
                // Apply TP sharding for target TP degree
                shard_dim, shard_type ← GET_SHARD_INFO_MCORE(mcore_key)
                IF shard_type ≠ "none" THEN
                    shards ← SPLIT(full_params[param_name], target_tp, dim=shard_dim)
                    mcore_state[mcore_key] ← shards[target_tp_rank]
                ELSE
                    mcore_state[mcore_key] ← full_params[param_name]
                END IF
            END FOR
            
            // Save Megatron-Core format
            ckpt ← {
                "model": mcore_state,
                "checkpoint_version": 3.0,
                "iteration": LOAD_ITERATION(ds_ckpt_dir),
                "args": BUILD_MCORE_ARGS(target_tp, target_pp)
            }
            
            path ← target_mcore_dir + "/mp_rank_{target_tp_rank:02d}_{target_pp_rank:03d}/model_optim_rng.pt"
            MAKEDIRS(path)
            SAVE(ckpt, path)
        END FOR
    END FOR
```

### 5.4 Megatron-Core to FSDP/DCP Conversion

**Pseudocode 8: Megatron-Core → FSDP/DCP Conversion**

```
PROCEDURE CONVERT_MCORE_TO_DCP(mcore_ckpt_dir, target_dcp_dir,
                                 source_tp, source_pp,
                                 target_mesh_shape):
    // ═══════════════════════════════════════════
    // PHASE 1: Reconstruct full (unsharded) model state
    // (Same as Phase 1-2 of Pseudocode 6)
    // ═══════════════════════════════════════════
    full_state_dict ← RECONSTRUCT_FULL_STATE(mcore_ckpt_dir, source_tp, source_pp)
    
    // ═══════════════════════════════════════════
    // PHASE 2: Map Megatron-Core keys to FSDP-compatible keys
    // ═══════════════════════════════════════════
    fsdp_state_dict ← {}
    FOR each mcore_key, value IN full_state_dict.items():
        // FSDP uses the module's original parameter names
        // Megatron uses "language_model.encoder.layers.{L}...."
        // Target uses "model.layers.{L}...."
        fsdp_key ← MAP_MCORE_KEY_TO_FSDP_KEY(mcore_key)
        fsdp_state_dict[fsdp_key] ← value
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 3: Generate DTensor placement metadata
    // ═══════════════════════════════════════════
    target_dp ← target_mesh_shape[0]
    target_tp ← target_mesh_shape[1] IF len(target_mesh_shape) > 1 ELSE 1
    
    metadata ← {}
    FOR each key, full_tensor IN fsdp_state_dict.items():
        shard_info ← GET_SHARD_INFO_FSDP(key)
        
        IF target_tp > 1 AND shard_info.tp_shard_dim IS NOT NONE THEN
            // 2D DTensor: Shard on DP + Shard on TP
            placements ← [Shard(0), Shard(shard_info.tp_shard_dim)]
        ELSE
            // 1D DTensor: Shard on DP only
            placements ← [Shard(0)]
        END IF
        
        metadata[key] ← {
            "global_shape": full_tensor.shape,
            "placements": placements,
            "dtype": full_tensor.dtype
        }
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 4: Shard tensors and write DCP format
    // ═══════════════════════════════════════════
    FOR each rank IN 0..TOTAL_RANKS-1:
        dp_rank ← rank / target_tp
        tp_rank ← rank MOD target_tp
        
        rank_state ← {}
        FOR each key, full_tensor IN fsdp_state_dict.items():
            meta ← metadata[key]
            
            // Apply TP sharding first
            IF target_tp > 1 AND meta.tp_shard_dim IS NOT NONE THEN
                tp_shard ← SPLIT(full_tensor, target_tp, dim=meta.tp_shard_dim)[tp_rank]
            ELSE
                tp_shard ← full_tensor
            END IF
            
            // Apply DP (FSDP) sharding
            // FSDP typically flattens and shards along dim 0
            dp_shard ← SPLIT(tp_shard.flatten(), target_dp, dim=0)[dp_rank]
            
            rank_state[key] ← dp_shard
        END FOR
        
        WRITE_DCP_SHARD(rank_state, metadata, rank, target_dcp_dir)
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 5: Write global DCP metadata
    // ═══════════════════════════════════════════
    WRITE_DCP_METADATA(metadata, target_dcp_dir + "/.metadata")
```

### 5.5 DeepSpeed to FSDP/DCP Conversion

**Pseudocode 9: DeepSpeed ZeRO-3 → FSDP/DCP Conversion**

```
PROCEDURE CONVERT_DS_ZERO3_TO_DCP(ds_ckpt_dir, target_dcp_dir,
                                     source_tp, source_dp,
                                     target_mesh_shape):
    // ═══════════════════════════════════════════
    // PHASE 1: Reconstruct full parameters from ZeRO-3 flat partitions
    // ═══════════════════════════════════════════
    full_params ← {}
    
    FOR tp_rank IN 0..source_tp-1:
        // Collect all DP partitions for this TP rank
        flat_partitions ← []
        FOR dp_rank IN 0..source_dp-1:
            partition ← LOAD_DS_PARTITION(ds_ckpt_dir, dp_rank, tp_rank)
            flat_partitions.append(partition["flat_buffer"])
        END FOR
        
        // Concatenate flat partitions
        full_flat ← CONCATENATE(flat_partitions, dim=0)
        
        // Unflatten using metadata
        param_metadata ← LOAD_DS_METADATA(ds_ckpt_dir, tp_rank=tp_rank)
        FOR param_name, meta IN param_metadata.items():
            param ← full_flat[meta.offset : meta.offset + meta.numel].reshape(meta.shape)
            
            IF source_tp > 1 THEN
                // This is a TP shard; accumulate for concatenation
                IF param_name NOT IN full_params THEN
                    full_params[param_name] ← [NONE] * source_tp
                END IF
                full_params[param_name][tp_rank] ← param
            ELSE
                full_params[param_name] ← param
            END IF
        END FOR
    END FOR
    
    // Concatenate TP shards
    IF source_tp > 1 THEN
        FOR param_name IN full_params:
            IF IS_LIST(full_params[param_name]) THEN
                shard_dim ← GET_TP_SHARD_DIM(param_name)
                full_params[param_name] ← CONCATENATE(full_params[param_name], dim=shard_dim)
            END IF
        END FOR
    END IF
    
    // ═══════════════════════════════════════════
    // PHASE 2: Validate reconstruction
    // ═══════════════════════════════════════════
    FOR param_name, param IN full_params.items():
        original_shape ← param_metadata[param_name].original_shape
        ASSERT param.shape = original_shape, 
            "Shape mismatch: " + param_name + " expected " + original_shape + " got " + param.shape
        
        // Check for NaN/Inf introduced during reconstruction
        ASSERT NOT ANY_NAN(param), "NaN detected in " + param_name
        ASSERT NOT ANY_INF(param), "Inf detected in " + param_name
    END FOR
    
    // ═══════════════════════════════════════════
    // PHASE 3: Generate DCP checkpoint (same as Phase 3-5 of Pseudocode 8)
    // ═══════════════════════════════════════════
    CALL WRITE_DCP_FROM_FULL_STATE(full_params, target_dcp_dir, target_mesh_shape)
```

### 5.6 Universal Conversion via Consolidated Intermediate

For maximum flexibility, a two-stage conversion through a consolidated (unsharded) intermediate is recommended:

**Pseudocode 10: Universal Conversion Pipeline**

```
PROCEDURE UNIVERSAL_CONVERT(source_dir, target_dir, source_format, target_format,
                              source_config, target_config):
    // ═══════════════════════════════════════════
    // STAGE 1: Consolidate to single unsharded state dict
    // ═══════════════════════════════════════════
    IF source_format = "megatron-core" THEN
        full_state, full_optim ← CONSOLIDATE_MCORE(source_dir, source_config.tp, source_config.pp)
    ELSE IF source_format = "deepspeed" THEN
        full_state, full_optim ← CONSOLIDATE_DEEPSPEED(source_dir, source_config.tp, 
                                                          source_config.dp, source_config.zero_stage)
    ELSE IF source_format = "fsdp-dcp" THEN
        full_state, full_optim ← CONSOLIDATE_DCP(source_dir)
    ELSE IF source_format = "huggingface" THEN
        full_state ← LOAD_HF_MODEL(source_dir)
        full_optim ← NONE
    END IF
    
    // ═══════════════════════════════════════════
    // STAGE 1.5: Key name normalization
    // Map all keys to a canonical naming convention
    // ═══════════════════════════════════════════
    canonical_state ← {}
    FOR each key, value IN full_state.items():
        canonical_key ← NORMALIZE_KEY(key, source_format)
        canonical_state[canonical_key] ← value
    END FOR
    
    // ═══════════════════════════════════════════
    // STAGE 2: Shard and write to target format
    // ═══════════════════════════════════════════
    IF target_format = "megatron-core" THEN
        WRITE_MCORE_CHECKPOINT(canonical_state, full_optim, target_dir,
                                target_config.tp, target_config.pp)
    ELSE IF target_format = "deepspeed" THEN
        WRITE_DS_CHECKPOINT(canonical_state, full_optim, target_dir,
                             target_config.tp, target_config.dp, target_config.zero_stage)
    ELSE IF target_format = "fsdp-dcp" THEN
        WRITE_DCP_CHECKPOINT(canonical_state, full_optim, target_dir,
                              target_config.mesh_shape)
    ELSE IF target_format = "huggingface" THEN
        WRITE_HF_CHECKPOINT(canonical_state, target_dir)
    END IF
    
    // ═══════════════════════════════════════════
    // STAGE 3: Validate conversion integrity
    // ═══════════════════════════════════════════
    VALIDATE_CONVERSION(source_dir, target_dir, source_format, target_format)
```

**Canonical Key Mapping Table (Example for LLaMA-style architecture):**

| Canonical Key | Megatron-Core Key | DeepSpeed Key | FSDP/HF Key |
|---|---|---|---|
| `layers.{L}.attn.qkv.weight` | `language_model.encoder.layers.{L}.self_attention.query_key_value.weight` | `module.model.layers.{L}.self_attn.qkv_proj.weight` | `model.layers.{L}.self_attn.qkv_proj.weight` |
| `layers.{L}.attn.out.weight` | `language_model.encoder.layers.{L}.self_attention.dense.weight` | `module.model.layers.{L}.self_attn.o_proj.weight` | `model.layers.{L}.self_attn.o_proj.weight` |
| `layers.{L}.mlp.gate.weight` | `language_model.encoder.layers.{L}.mlp.dense_h_to_4h.weight` | `module.model.layers.{L}.mlp.gate_proj.weight` | `model.layers.{L}.mlp.gate_proj.weight` |
| `layers.{L}.mlp.up.weight` | (merged with gate in some versions) | `module.model.layers.{L}.mlp.up_proj.weight` | `model.layers.{L}.mlp.up_proj.weight` |
| `layers.{L}.mlp.down.weight` | `language_model.encoder.layers.{L}.mlp.dense_4h_to_h.weight` | `module.model.layers.{L}.mlp.down_proj.weight` | `model.layers.{L}.mlp.down_proj.weight` |
| `layers.{L}.norm1.weight` | `language_model.encoder.layers.{L}.input_layernorm.weight` | `module.model.layers.{L}.input_layernorm.weight` | `model.layers.{L}.input_layernorm.weight` |
| `layers.{L}.norm2.weight` | `language_model.encoder.layers.{L}.post_attention_layernorm.weight` | `module.model.layers.{L}.post_attention_layernorm.weight` | `model.layers.{L}.post_attention_layernorm.weight` |
| `embed.weight` | `language_model.embedding.word_embeddings.weight` | `module.model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `head.weight` | `language_model.output_layer.weight` | `module.lm_head.weight` | `lm_head.weight` |

### 5.7 TP Shard Dimension Reference

**Table: TP Sharding Dimensions for Key Parameter Types**

| Parameter Type | Megatron Shard Type | Shard Dimension | Concatenation for Consolidation |
|---|---|---|---|
| QKV Projection Weight | Column-parallel | dim = -1 (output) | `torch.cat(shards, dim=-1)` |
| QKV Projection Bias | Column-parallel | dim = 0 | `torch.cat(shards, dim=0)` |
| Attention Output Weight | Row-parallel | dim = 0 (input) | `torch.cat(shards, dim=0)` |
| Attention Output Bias | Replicated | N/A (no sharding) | Use any shard |
| MLP Gate/Up Weight | Column-parallel | dim = -1 (output) | `torch.cat(shards, dim=-1)` |
| MLP Down Weight | Row-parallel | dim = 0 (input) | `torch.cat(shards, dim=0)` |
| LayerNorm Weight/Bias | Replicated | N/A | Use any shard |
| Embedding Weight | Vocab-parallel (row) or replicated | dim = 0 (vocab) | `torch.cat(shards, dim=0)` or use any |
| Output (LM Head) Weight | Column-parallel | dim = 0 (vocab) | `torch.cat(shards, dim=0)` |

> **Critical note on QKV projection:** Megatron-Core often uses a **fused QKV** projection where Q, K, V are concatenated into a single weight matrix. Some other frameworks split them. During conversion:
>
> $$W_{\text{QKV}}^{\text{Megatron}} = [W_Q; W_K; W_V] \in \mathbb{R}^{h \times 3h}$$
>
> If the target expects separate $W_Q \in \mathbb{R}^{h \times h}$, $W_K \in \mathbb{R}^{h \times h}$, $W_V \in \mathbb{R}^{h \times h}$:
>
> $$W_Q = W_{\text{QKV}}[:, 0:h], \quad W_K = W_{\text{QKV}}[:, h:2h], \quad W_V = W_{\text{QKV}}[:, 2h:3h]$$
>
> For **GQA (Grouped Query Attention)** where $n_{kv} < n_h$:
>
> $$W_{\text{QKV}} \in \mathbb{R}^{h \times (n_h \cdot d_h + 2 \cdot n_{kv} \cdot d_h)}$$
>
> Splitting must account for the unequal Q vs K/V head dimensions.

---

## 6. Checkpoint Resharding Across Topologies

### 6.1 Resharding Taxonomy

Resharding is required when the training topology changes between runs (e.g., scaling up/down, hardware migration, or recovery from partial cluster failure).

| Source → Target | Description | Complexity |
|---|---|---|
| TP=4 → TP=8 | Split each TP shard in half | Low |
| TP=8 → TP=4 | Concatenate pairs of TP shards | Low |
| PP=4 → PP=8 | Reassign layers to finer-grained stages | Medium |
| PP=8 → PP=4 | Merge pairs of stages | Medium |
| DP=32 → DP=64 | ZeRO: re-partition flat buffers; FSDP: DCP handles natively | Low-Medium |
| TP=4,PP=2 → TP=8,PP=1 | Combined TP expansion + PP merge | High |
| Framework A → Framework B | Full conversion (see §5) | High |

### 6.2 TP Resharding Algorithm

**Pseudocode 11: TP Resharding (Arbitrary Source → Target)**

```
PROCEDURE RESHARD_TP(source_shards, source_tp, target_tp, param_name):
    // source_shards: list of source_tp tensors, each being one TP shard
    // Returns: list of target_tp tensors, each being one new TP shard
    
    shard_dim, shard_type ← GET_SHARD_INFO(param_name)
    
    IF shard_type = "none" THEN
        // Replicated parameter — no resharding needed
        RETURN [source_shards[0]] * target_tp
    END IF
    
    // Step 1: Reconstruct full parameter
    full_param ← CONCATENATE(source_shards, dim=shard_dim)
    
    // Step 2: Verify divisibility
    dim_size ← full_param.shape[shard_dim]
    ASSERT dim_size MOD target_tp = 0,
        param_name + ": dim " + shard_dim + " size " + dim_size + 
        " not divisible by target_tp=" + target_tp
    
    // Step 3: Split for target TP degree
    target_shards ← SPLIT(full_param, target_tp, dim=shard_dim)
    
    // Step 4: Validate
    FOR each shard IN target_shards:
        expected_shape ← COPY(full_param.shape)
        expected_shape[shard_dim] ← dim_size / target_tp
        ASSERT shard.shape = expected_shape
    END FOR
    
    RETURN target_shards
```

### 6.3 PP Resharding Algorithm

**Pseudocode 12: PP Resharding (Arbitrary Source → Target)**

```
PROCEDURE RESHARD_PP(source_stage_assignments, source_pp, target_pp, total_layers):
    // source_stage_assignments: dict mapping layer_idx → source_pp_rank
    // Returns: dict mapping layer_idx → target_pp_rank
    
    // Standard uniform assignment
    layers_per_target_stage ← total_layers / target_pp
    ASSERT total_layers MOD target_pp = 0,
        "total_layers=" + total_layers + " not divisible by target_pp=" + target_pp
    
    target_assignments ← {}
    FOR layer_idx IN 0..total_layers-1:
        target_assignments[layer_idx] ← layer_idx / layers_per_target_stage
    END FOR
    
    // Handle non-layer parameters (embedding, output layer, final norm)
    // Convention: embedding → stage 0; output layer → last stage; final norm → last stage
    target_assignments["embedding"] ← 0
    target_assignments["output_layer"] ← target_pp - 1
    target_assignments["final_norm"] ← target_pp - 1
    
    // For interleaved (virtual) pipeline, assignment is more complex:
    // Layer i with v virtual stages per physical stage goes to:
    //   physical_stage = (i / layers_per_virtual_stage) MOD target_pp
    
    RETURN target_assignments

PROCEDURE RESHARD_PP_CHECKPOINT(source_ckpt_dir, target_ckpt_dir,
                                  source_tp, source_pp, target_pp, total_layers):
    // Step 1: Load all source stages
    all_params ← {}
    FOR pp_rank IN 0..source_pp-1:
        FOR tp_rank IN 0..source_tp-1:
            stage_state ← LOAD(source_ckpt_dir + "/mp_rank_{tp_rank}_{pp_rank}/...")
            FOR key, value IN stage_state["model"].items():
                // Reconstruct global key from stage-local key
                global_key ← MAKE_GLOBAL_KEY(key, pp_rank, source_pp, total_layers)
                all_params[(global_key, tp_rank)] ← value
            END FOR
        END FOR
    END FOR
    
    // Step 2: Compute new assignments
    new_assignments ← RESHARD_PP(NONE, source_pp, target_pp, total_layers)
    
    // Step 3: Write new checkpoint
    FOR target_pp_rank IN 0..target_pp-1:
        FOR tp_rank IN 0..source_tp-1:
            stage_state ← {}
            FOR (global_key, tp_r), value IN all_params.items():
                IF tp_r ≠ tp_rank THEN CONTINUE END IF
                
                layer_idx ← EXTRACT_LAYER_IDX(global_key)
                IF new_assignments[layer_idx] = target_pp_rank THEN
                    local_key ← MAKE_LOCAL_KEY(global_key, target_pp_rank, target_pp)
                    stage_state[local_key] ← value
                END IF
            END FOR
            
            SAVE stage_state TO target_ckpt_dir + "/mp_rank_{tp_rank}_{target_pp_rank}/..."
        END FOR
    END FOR
```

### 6.4 ZeRO/FSDP DP Resharding

**DeepSpeed ZeRO Resharding:**

For ZeRO-1 (optimizer states only), resharding requires re-partitioning the flat optimizer buffer:

$$
\text{partition\_size}_{\text{new}} = \left\lceil \frac{|\mathcal{O}|}{d_{\text{new}}} \right\rceil
$$

where $|\mathcal{O}|$ is the total optimizer state size (in elements) for one TP-sharded parameter set.

For ZeRO-3 (parameters + gradients + optimizer states), all three buffers must be re-partitioned:

$$
\text{param\_partition}_{\text{new}} = \left\lceil \frac{|\Phi_{\text{TP-local}}|}{d_{\text{new}}} \right\rceil
$$

**FSDP/DCP Native Resharding:**

PyTorch DCP supports native resharding through its `LoadPlanner` and `SavePlanner` abstraction:

**Pseudocode 13: DCP Native Resharding**

```
PROCEDURE DCP_RESHARD_LOAD(source_dcp_dir, target_mesh, model):
    // DCP's load_state_dict automatically handles resharding
    // It reads the .metadata file, compares source placements with target,
    // and generates a load plan that re-slices and redistributes tensors
    
    // Step 1: Create state dict with DTensor specifications for target mesh
    target_state_dict ← {}
    FOR each name, param IN model.named_parameters():
        // param is a DTensor with target mesh's placements
        target_state_dict[name] ← param
    END FOR
    
    // Step 2: Load with resharding
    // DCP compares:
    //   source: .metadata contains {key: (global_shape, source_placements, source_mesh)}
    //   target: target_state_dict contains {key: DTensor with target_placements on target_mesh}
    // If they differ, DCP generates a redistribution plan
    
    CALL torch.distributed.checkpoint.load_state_dict(
        state_dict = target_state_dict,
        storage_reader = FileSystemReader(source_dcp_dir),
        planner = DefaultLoadPlanner()  // Handles resharding automatically
    )
    
    // Step 3: Verify loaded state
    FOR each name, param IN model.named_parameters():
        ASSERT param.placements = TARGET_PLACEMENTS
        ASSERT param.device_mesh = target_mesh
        ASSERT NOT ANY_NAN(param._local_tensor)
    END FOR
    
    RETURN target_state_dict
```

> **DCP resharding advantage:** DCP is the only framework that supports **native resharding without a consolidation step**. It can directly load a checkpoint saved with $t=4, d=32$ into a model configured with $t=8, d=16$, computing the necessary tensor redistributions on-the-fly. This avoids the $O(\Phi)$ memory overhead of full consolidation, which is critical for models exceeding single-node memory.

---

## 7. Resume Workflows and Deterministic Recovery

### 7.1 Resume Taxonomy

| Resume Type | Description | Framework Compatibility |
|---|---|---|
| **Identical topology resume** | Same $t, p, d, c, e$ | All frameworks (native) |
| **Same-framework resharded resume** | Different $t, p, d$ but same framework | Megatron: via conversion script; DeepSpeed: via `zero_to_fp32` + reinit; FSDP/DCP: native |
| **Cross-framework resume** | Different framework entirely | Requires full conversion (§5) |
| **Elastic resume** | Different world size (node failure recovery) | DeepSpeed Elastic: built-in; FSDP/DCP: native; Megatron: requires conversion |

### 7.2 Identical Topology Resume

**Pseudocode 14: Identical Topology Resume (Megatron-Core + DeepSpeed)**

```
PROCEDURE RESUME_IDENTICAL_TOPOLOGY(ckpt_dir, model_engine, config):
    // ═══════════════════════════════════════════
    // STEP 1: Verify topology consistency
    // ═══════════════════════════════════════════
    saved_args ← LOAD(ckpt_dir + "/mp_rank_00_000/model_optim_rng.pt")["args"]
    
    ASSERT saved_args.tensor_model_parallel_size = config.tp
    ASSERT saved_args.pipeline_model_parallel_size = config.pp
    ASSERT saved_args.data_parallel_size = config.dp
    
    // ═══════════════════════════════════════════
    // STEP 2: Load model state (TP-sharded, PP-staged)
    // ═══════════════════════════════════════════
    tp_rank ← mpu.get_tensor_model_parallel_rank()
    pp_rank ← mpu.get_pipeline_model_parallel_rank()
    
    ckpt ← LOAD(ckpt_dir + "/mp_rank_{tp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt",
                 map_location="cuda:" + LOCAL_RANK)
    
    // ═══════════════════════════════════════════
    // STEP 3: Load model weights
    // ═══════════════════════════════════════════
    model_engine.module.load_state_dict(ckpt["model"], strict=True)
    
    // ═══════════════════════════════════════════
    // STEP 4: Load optimizer state
    // ═══════════════════════════════════════════
    IF config.uses_deepspeed THEN
        // DeepSpeed manages optimizer loading internally
        model_engine.load_checkpoint(ckpt_dir)
        // This loads:
        //   - ZeRO optimizer partitions (per DP rank)
        //   - Learning rate scheduler state
        //   - Training iteration counter
        //   - Random number generator states
    ELSE
        // Direct Megatron optimizer loading
        optimizer.load_state_dict(ckpt["optimizer"])
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 5: Restore RNG states for deterministic replay
    // ═══════════════════════════════════════════
    rng_state ← ckpt["rng_state"]
    torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
    torch.set_rng_state(rng_state["random_rng_state"])
    numpy.random.set_state(rng_state["np_rng_state"])
    
    // Megatron-Core tracker RNG (for dropout reproducibility across TP ranks)
    IF "tracker_rng_state" IN rng_state THEN
        mpu.get_cuda_rng_tracker().set_states(rng_state["tracker_rng_state"])
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 6: Restore dataloader state
    // ═══════════════════════════════════════════
    iteration ← ckpt["iteration"]
    consumed_samples ← iteration × config.global_batch_size
    
    dataloader.set_consumed_samples(consumed_samples)
    // This advances the sampler to skip already-seen data
    // Critical for reproducibility: same data ordering as original run
    
    // ═══════════════════════════════════════════
    // STEP 7: Synchronize all ranks before proceeding
    // ═══════════════════════════════════════════
    torch.distributed.barrier()
    
    LOG "Resumed from iteration " + iteration + " with " + consumed_samples + " consumed samples"
    
    RETURN iteration
```

### 7.3 Cross-Topology Resume (Resharded)

**Pseudocode 15: Cross-Topology Resume**

```
PROCEDURE RESUME_CROSS_TOPOLOGY(source_ckpt_dir, source_config, target_config, model_engine):
    // ═══════════════════════════════════════════
    // STEP 1: Determine what changed
    // ═══════════════════════════════════════════
    tp_changed ← source_config.tp ≠ target_config.tp
    pp_changed ← source_config.pp ≠ target_config.pp
    dp_changed ← source_config.dp ≠ target_config.dp
    framework_changed ← source_config.framework ≠ target_config.framework
    
    LOG "Topology change: TP " + source_config.tp + "→" + target_config.tp +
        ", PP " + source_config.pp + "→" + target_config.pp +
        ", DP " + source_config.dp + "→" + target_config.dp
    
    // ═══════════════════════════════════════════
    // STEP 2: Decide conversion strategy
    // ═══════════════════════════════════════════
    IF framework_changed THEN
        // Full cross-framework conversion required
        temp_dir ← TEMP_DIRECTORY()
        CALL UNIVERSAL_CONVERT(source_ckpt_dir, temp_dir,
                                source_config.framework, target_config.framework,
                                source_config, target_config)
        reshaped_ckpt_dir ← temp_dir
        
    ELSE IF tp_changed OR pp_changed THEN
        // Same framework, but TP/PP changed — need resharding
        temp_dir ← TEMP_DIRECTORY()
        
        IF target_config.framework = "megatron-core" THEN
            CALL RESHARD_MCORE_CHECKPOINT(source_ckpt_dir, temp_dir,
                                            source_config.tp, source_config.pp,
                                            target_config.tp, target_config.pp)
        ELSE IF target_config.framework = "deepspeed" THEN
            // DeepSpeed doesn't natively support TP resharding
            // Must go through consolidation
            consolidated ← CONSOLIDATE_DEEPSPEED(source_ckpt_dir, source_config)
            WRITE_DS_CHECKPOINT(consolidated, temp_dir, target_config)
        END IF
        
        reshaped_ckpt_dir ← temp_dir
        
    ELSE IF dp_changed THEN
        // Only DP changed — framework-specific handling
        IF target_config.framework = "fsdp-dcp" THEN
            // DCP handles DP resharding natively — no conversion needed
            reshaped_ckpt_dir ← source_ckpt_dir
        ELSE IF target_config.framework = "deepspeed" THEN
            // ZeRO partitions must be re-sliced
            temp_dir ← TEMP_DIRECTORY()
            RESHARD_ZERO_PARTITIONS(source_ckpt_dir, temp_dir,
                                      source_config.dp, target_config.dp)
            reshaped_ckpt_dir ← temp_dir
        END IF
    ELSE
        reshaped_ckpt_dir ← source_ckpt_dir
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 3: Load reshaped checkpoint
    // ═══════════════════════════════════════════
    CALL RESUME_IDENTICAL_TOPOLOGY(reshaped_ckpt_dir, model_engine, target_config)
    
    // ═══════════════════════════════════════════
    // STEP 4: Handle optimizer state
    // ═══════════════════════════════════════════
    IF tp_changed OR pp_changed OR framework_changed THEN
        // Optimizer state topology has changed
        // Option A: Re-initialize optimizer (lose momentum/variance — convergence hit)
        // Option B: Reshard optimizer state (complex but preserves training dynamics)
        
        IF OPTIMIZER_RESHARDING_SUPPORTED THEN
            RESHARD_OPTIMIZER_STATE(source_ckpt_dir, model_engine.optimizer,
                                     source_config, target_config)
            LOG "Optimizer state resharded — training dynamics preserved"
        ELSE
            REINITIALIZE_OPTIMIZER(model_engine.optimizer)
            LOG "WARNING: Optimizer state re-initialized — expect temporary convergence perturbation"
            // Impact: ~100-500 steps of slightly higher loss before optimizer re-adapts
        END IF
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 5: Verify numerical consistency
    // ═══════════════════════════════════════════
    VALIDATE_LOADED_CHECKPOINT(model_engine, source_ckpt_dir, source_config)
    
    // Cleanup temporary files
    IF temp_dir EXISTS THEN
        CLEANUP(temp_dir)
    END IF
```

### 7.4 Deterministic Resume Requirements

For bit-exact reproducibility across resume boundaries:

| State Component | Must Be Restored | Framework Support |
|---|---|---|
| Model parameters | ✅ Yes | All frameworks |
| Optimizer states ($m_t$, $v_t$, step count) | ✅ Yes | All frameworks |
| Learning rate scheduler state | ✅ Yes | All frameworks |
| CUDA RNG state (per GPU) | ✅ Yes | All frameworks |
| CPU RNG state | ✅ Yes | All frameworks |
| Numpy RNG state | ✅ Yes | All frameworks |
| Megatron CudaRNGStatesTracker | ✅ Yes (Megatron-Core) | Megatron-Core only |
| Dataloader consumed samples | ✅ Yes | Manual state tracking |
| Dataloader shuffle seed | ✅ Yes | Manual state tracking |
| Gradient accumulation step counter | ✅ Yes | Framework-specific |
| Loss scaler state (FP16/FP8) | ✅ Yes | Framework-specific |
| NCCL/RCCL internal state | ❌ No (stateless) | N/A |
| CUDA memory allocator state | ❌ No (non-deterministic) | N/A |

> **Important:** CUDA's `cublasLt` GEMM algorithms may produce different rounding in different execution orders. Even with identical RNG states, multi-GPU training is **not guaranteed to be bit-exact** across runs due to non-deterministic floating-point reduction order in all-reduce. Setting `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` improves but does not guarantee determinism at scale.

---

## 8. Mixed-Framework Training Configurations

### 8.1 Common Production Patterns

**Table: Real-World Mixed-Framework Configurations**

| Configuration | Use Case | Megatron-Core Role | DeepSpeed Role | FSDP Role |
|---|---|---|---|---|
| **Megatron + ZeRO-1** | Large-scale pretraining (most common) | TP, PP, model arch, data pipeline | Optimizer state sharding, lr scheduler | Not used |
| **Megatron + ZeRO-2** | Pretraining with gradient memory savings | TP, PP, model arch | Optimizer + gradient sharding | Not used |
| **Megatron + ZeRO-3** | Memory-constrained pretraining | TP, PP, model arch | Full parameter/grad/optim sharding | Not used |
| **Megatron TP + FSDP** | Compiler-friendly pretraining | TP via DTensor, model arch | Not used | DP sharding, mixed precision |
| **FSDP + DeepSpeed ZeRO-1** | Fine-tuning with efficient optimizer | Not used | Optimizer sharding only | Parameter + gradient sharding |
| **Megatron + DeepSpeed MoE** | MoE pretraining | TP, PP for dense layers | Expert parallelism, expert routing | Not used |

### 8.2 Configuration Compatibility Matrix

**Table: Feature Compatibility Across Compositions**

| Feature | Megatron + ZeRO-1 | Megatron + ZeRO-3 | Megatron + FSDP | FSDP + ZeRO-1 |
|---|---|---|---|---|
| Activation checkpointing | Megatron AC | Megatron AC | Composable AC | PyTorch AC |
| Mixed precision (BF16) | ✅ | ✅ | ✅ | ✅ |
| FP8 (Transformer Engine) | ✅ | ⚠️ (requires TE + ZeRO-3 compat) | ✅ (TE + DTensor) | ❌ (no TE in FSDP-only) |
| `torch.compile` | ❌ (DS hooks break graph) | ❌ | ✅ (native) | ⚠️ (partial) |
| CUDA Graphs | ⚠️ (static shapes only) | ❌ (dynamic allgather) | ⚠️ (with restrictions) | ❌ |
| Interleaved PP schedule | ✅ | ✅ | ❌ (not natively in FSDP) | N/A |
| Context Parallelism | ✅ | ⚠️ (extra allgather interaction) | ⚠️ (manual) | ❌ |
| Expert Parallelism | ✅ (Megatron-Core MoE) | ✅ (DS MoE) | ❌ (manual) | ❌ |
| Elastic training | ❌ | ✅ (DS Elastic) | ⚠️ (DCP resharding) | ✅ (DS Elastic) |
| Checkpoint resharding | Via scripts | Via `zero_to_fp32` | Native (DCP) | Mixed |

### 8.3 Mixed-Framework Checkpoint Save Strategy

When multiple frameworks are composed, checkpoint saving must coordinate all framework states:

**Pseudocode 16: Mixed-Framework Checkpoint Save**

```
PROCEDURE SAVE_MIXED_CHECKPOINT(model_engine, iteration, config, ckpt_dir):
    // Determine which framework manages which state
    
    // ═══════════════════════════════════════════
    // Component 1: Model State (Megatron-Core format)
    // ═══════════════════════════════════════════
    tp_rank ← mpu.get_tensor_model_parallel_rank()
    pp_rank ← mpu.get_pipeline_model_parallel_rank()
    dp_rank ← mpu.get_data_parallel_rank()
    
    model_state ← {}
    FOR name, param IN model_engine.module.named_parameters():
        IF config.uses_deepspeed AND config.zero_stage = 3 THEN
            // ZeRO-3: param.data is a flat partition, not the full TP shard
            // Must gather the full TP shard first
            full_tp_shard ← DEEPSPEED_GATHER_PARAM(param)
            model_state[name] ← full_tp_shard
        ELSE
            model_state[name] ← param.data
        END IF
    END FOR
    
    // ═══════════════════════════════════════════
    // Component 2: Optimizer State
    // ═══════════════════════════════════════════
    IF config.uses_deepspeed THEN
        // DeepSpeed handles its own optimizer checkpoint
        model_engine.save_checkpoint(ckpt_dir, tag="iter_" + iteration)
        // This saves: zero_pp_rank_DD_mp_rank_TT_optim_states.pt
    END IF
    
    IF config.uses_fsdp THEN
        // FSDP uses DCP for optimizer state
        optim_state ← FSDP.optim_state_dict(model_engine, optimizer)
        DCP.save_state_dict(
            state_dict = {"optimizer": optim_state},
            storage_writer = FileSystemWriter(ckpt_dir + "/optim_dcp")
        )
    END IF
    
    IF NOT config.uses_deepspeed AND NOT config.uses_fsdp THEN
        // Pure Megatron optimizer
        optim_state ← optimizer.state_dict()
    END IF
    
    // ═══════════════════════════════════════════
    // Component 3: RNG States (always Megatron-managed)
    // ═══════════════════════════════════════════
    rng_state ← {
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "random_rng_state": torch.get_rng_state(),
        "np_rng_state": numpy.random.get_state(),
        "tracker_rng_state": mpu.get_cuda_rng_tracker().get_states()
    }
    
    // ═══════════════════════════════════════════
    // Component 4: Training metadata
    // ═══════════════════════════════════════════
    metadata ← {
        "iteration": iteration,
        "consumed_samples": iteration × config.global_batch_size,
        "config": config,
        "frameworks": {
            "megatron_core_version": MCORE_VERSION,
            "deepspeed_version": DS_VERSION IF config.uses_deepspeed ELSE NONE,
            "pytorch_version": TORCH_VERSION,
            "fsdp_version": "fsdp2" IF config.uses_fsdp ELSE NONE
        },
        "topology": {
            "tp": config.tp, "pp": config.pp, "dp": config.dp,
            "cp": config.cp, "ep": config.ep,
            "zero_stage": config.zero_stage
        }
    }
    
    // ═══════════════════════════════════════════
    // Component 5: Unified save (Megatron format)
    // ═══════════════════════════════════════════
    ckpt ← {
        "model": model_state,
        "optimizer": optim_state IF NOT config.uses_deepspeed ELSE NONE,
        "rng_state": rng_state,
        "iteration": iteration,
        "metadata": metadata
    }
    
    save_path ← ckpt_dir + "/iter_{iteration:07d}/mp_rank_{tp_rank:02d}_{pp_rank:03d}/"
    
    IF config.uses_megatron_dist_optim THEN
        // Distributed optimizer: also need dp_rank in filename
        save_path ← save_path + "dp_rank_{dp_rank:03d}/"
    END IF
    
    MAKEDIRS(save_path)
    SAVE(ckpt, save_path + "model_optim_rng.pt")
    
    // ═══════════════════════════════════════════
    // Component 6: Checkpoint validation (async)
    // ═══════════════════════════════════════════
    ASYNC VALIDATE_CHECKPOINT_INTEGRITY(save_path)
    
    // Update latest checkpoint marker
    IF dp_rank = 0 AND tp_rank = 0 AND pp_rank = 0 THEN
        WRITE(iteration, ckpt_dir + "/latest_checkpointed_iteration.txt")
    END IF
    
    torch.distributed.barrier()
```

---

## 9. Optimizer State Interoperability

### 9.1 Optimizer State Format Differences

The optimizer state presents the most complex interoperability challenge because each framework stores it differently:

**Adam Optimizer State Per Parameter $\theta_i$:**

$$
\text{State}(\theta_i) = \left\{m_t^{(i)}, \; v_t^{(i)}, \; \text{step}^{(i)}\right\}
$$

where $m_t^{(i)}$ is the first moment (momentum), $v_t^{(i)}$ is the second moment (variance), and $\text{step}^{(i)}$ is the update count.

| Framework | Storage Format | Indexing | Precision |
|---|---|---|---|
| **Megatron-Core (standard)** | Per-parameter dict keyed by param index; each entry has `exp_avg`, `exp_avg_sq` | Sequential integer index per rank | FP32 |
| **Megatron-Core (distributed optim)** | Same as above but only 1/d-th of entries per DP rank | Parameter index with DP offset | FP32 |
| **DeepSpeed ZeRO-1** | Same as standard Adam but partitioned: rank $i$ stores optimizer state for params $[i \cdot N/d, (i+1) \cdot N/d)$ | Flat index into param group | FP32 |
| **DeepSpeed ZeRO-2/3** | Flat FP32 buffer containing interleaved $[m, v, \theta_{\text{master}}]$ for its partition slice | Flat offset | FP32 |
| **FSDP** | Standard `optimizer.state_dict()` with DTensor metadata; states are sharded like parameters | FQN (fully qualified name) | FP32 |

### 9.2 Optimizer State Conversion

**Pseudocode 17: Optimizer State Conversion (DeepSpeed ZeRO → Megatron-Core)**

```
PROCEDURE CONVERT_OPTIMIZER_DS_TO_MCORE(ds_ckpt_dir, source_dp, source_tp, source_zero_stage,
                                          target_tp, target_pp, model_param_shapes):
    // ═══════════════════════════════════════════
    // STEP 1: Collect all optimizer partitions
    // ═══════════════════════════════════════════
    
    IF source_zero_stage = 1 THEN
        // ZeRO-1: optimizer states are partitioned by param index
        all_optim_partitions ← []
        FOR dp_rank IN 0..source_dp-1:
            FOR tp_rank IN 0..source_tp-1:
                partition ← LOAD(ds_ckpt_dir + 
                    "/zero_pp_rank_{dp_rank}_mp_rank_{tp_rank}_optim_states.pt")
                all_optim_partitions.append(partition)
            END FOR
        END FOR
        
        // Merge partitions to get full optimizer state
        full_optim_state ← MERGE_ZERO1_PARTITIONS(all_optim_partitions, source_dp)
        
    ELSE IF source_zero_stage IN {2, 3} THEN
        // ZeRO-2/3: optimizer stored as flat FP32 buffer
        FOR tp_rank IN 0..source_tp-1:
            flat_fp32_partitions ← []
            FOR dp_rank IN 0..source_dp-1:
                partition ← LOAD(ds_ckpt_dir + 
                    "/zero_pp_rank_{dp_rank}_mp_rank_{tp_rank}/fp32_flat/000.pt")
                flat_fp32_partitions.append(partition)
            END FOR
            
            // Concatenate flat partitions
            full_flat_fp32 ← CONCATENATE(flat_fp32_partitions, dim=0)
            
            // The flat buffer is organized as:
            // [fp32_master_weights | exp_avg (momentum) | exp_avg_sq (variance)]
            // Each section has size = total_params_for_this_tp_rank
            total_params ← SUM(p.numel() FOR p IN model_param_shapes[tp_rank])
            
            fp32_master ← full_flat_fp32[0 : total_params]
            exp_avg ← full_flat_fp32[total_params : 2 * total_params]
            exp_avg_sq ← full_flat_fp32[2 * total_params : 3 * total_params]
            
            // Unflatten into per-parameter optimizer states
            offset ← 0
            FOR param_idx, (param_name, param_shape) IN ENUMERATE(model_param_shapes[tp_rank]):
                numel ← PRODUCT(param_shape)
                
                full_optim_state[(param_name, tp_rank)] ← {
                    "fp32_master": fp32_master[offset:offset+numel].reshape(param_shape),
                    "exp_avg": exp_avg[offset:offset+numel].reshape(param_shape),
                    "exp_avg_sq": exp_avg_sq[offset:offset+numel].reshape(param_shape),
                    "step": partition["step"]  // Global step count
                }
                
                offset ← offset + numel
            END FOR
        END FOR
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 2: Reconstruct full (unsharded) optimizer state
    // ═══════════════════════════════════════════
    IF source_tp > 1 THEN
        full_optim ← {}
        FOR each param_name IN UNIQUE_PARAM_NAMES:
            shard_dim ← GET_TP_SHARD_DIM(param_name)
            
            // Concatenate TP shards of optimizer states
            master_shards ← [full_optim_state[(param_name, tp)]["fp32_master"] 
                             FOR tp IN 0..source_tp-1]
            exp_avg_shards ← [full_optim_state[(param_name, tp)]["exp_avg"]
                              FOR tp IN 0..source_tp-1]
            exp_avg_sq_shards ← [full_optim_state[(param_name, tp)]["exp_avg_sq"]
                                  FOR tp IN 0..source_tp-1]
            
            IF shard_dim IS NOT NONE THEN
                full_optim[param_name] ← {
                    "fp32_master": CONCATENATE(master_shards, dim=shard_dim),
                    "exp_avg": CONCATENATE(exp_avg_shards, dim=shard_dim),
                    "exp_avg_sq": CONCATENATE(exp_avg_sq_shards, dim=shard_dim),
                    "step": full_optim_state[(param_name, 0)]["step"]
                }
            ELSE
                full_optim[param_name] ← full_optim_state[(param_name, 0)]
            END IF
        END FOR
    ELSE
        full_optim ← full_optim_state
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 3: Re-shard for target Megatron-Core topology
    // ═══════════════════════════════════════════
    mcore_optim_states ← {}  // Keyed by (tp_rank, pp_rank)
    
    FOR target_tp_rank IN 0..target_tp-1:
        FOR target_pp_rank IN 0..target_pp-1:
            stage_optim ← {}
            param_idx ← 0
            
            FOR each param_name IN full_optim:
                IF NOT PARAM_IN_STAGE(param_name, target_pp_rank, target_pp) THEN
                    CONTINUE
                END IF
                
                shard_dim ← GET_TP_SHARD_DIM(param_name)
                
                IF shard_dim IS NOT NONE THEN
                    // Split optimizer states along TP dimension
                    FOR state_key IN ["fp32_master", "exp_avg", "exp_avg_sq"]:
                        full_state ← full_optim[param_name][state_key]
                        tp_shard ← SPLIT(full_state, target_tp, dim=shard_dim)[target_tp_rank]
                        stage_optim[param_idx] ← stage_optim.get(param_idx, {})
                        stage_optim[param_idx][state_key] ← tp_shard
                    END FOR
                ELSE
                    stage_optim[param_idx] ← {
                        "fp32_master": full_optim[param_name]["fp32_master"],
                        "exp_avg": full_optim[param_name]["exp_avg"],
                        "exp_avg_sq": full_optim[param_name]["exp_avg_sq"]
                    }
                END IF
                
                stage_optim[param_idx]["step"] ← full_optim[param_name]["step"]
                param_idx ← param_idx + 1
            END FOR
            
            mcore_optim_states[(target_tp_rank, target_pp_rank)] ← stage_optim
        END FOR
    END FOR
    
    RETURN mcore_optim_states
```

### 9.3 Optimizer State Size Formulas

The total optimizer state memory for Adam/AdamW is:

$$
M_{\text{optim}} = \Phi \times (4 + 4 + 4) = 12\Phi \text{ bytes}
$$

for FP32 master weights ($4\Phi$), first moment $m_t$ ($4\Phi$), and second moment $v_t$ ($4\Phi$).

Per-rank optimizer memory depends on the sharding strategy:

| Configuration | Per-Rank Optimizer Memory |
|---|---|
| No sharding (DDP) | $12\Phi / t$ bytes |
| ZeRO-1 / Megatron Dist. Optim | $12\Phi / (t \times d)$ bytes |
| ZeRO-2 | $12\Phi / (t \times d)$ bytes |
| ZeRO-3 / FSDP FULL_SHARD | $12\Phi / (t \times d)$ bytes |
| ZeRO-3 + FSDP (anti-pattern) | $12\Phi / (t \times d^2)$ — but causes double-sharding issues |

---

## 10. Memory Budget Analysis Under Interoperability

### 10.1 Memory During Checkpoint Save

Checkpoint saving temporarily increases memory usage because the model state must be gathered/consolidated:

$$
M_{\text{save}}^{\text{peak}} = M_{\text{training}} + M_{\text{gather}} + M_{\text{serialize}}
$$

| Framework | $M_{\text{gather}}$ During Save | Notes |
|---|---|---|
| Megatron-Core (standard) | $0$ (already full TP shard on each rank) | Direct save; no extra memory |
| Megatron-Core (dist. optim) | $0$ (saves per-rank partition) | Direct save |
| DeepSpeed ZeRO-1 | $0$ (model is full per DP rank) | Direct save |
| DeepSpeed ZeRO-3 | $2\Phi / t$ (must gather each param for save) | Can OOM on tight HBM budgets |
| FSDP FULL_SHARD | $0$ with `StateDictType.SHARDED_STATE_DICT` | DCP saves shards directly |
| FSDP FULL_SHARD | $2\Phi / t$ with `StateDictType.FULL_STATE_DICT` | Gathers full state to rank 0 |

> **Production recommendation:** Always use **sharded checkpoint saving** (no full consolidation on any single rank). For Megatron-Core, save per-(TP, PP, DP) rank files. For DeepSpeed ZeRO-3, use `stage3_gather_16bit_weights_on_model_save=True` cautiously (it adds $2\Phi/t$ peak memory) or use sharded saves. For FSDP, always use DCP with `SHARDED_STATE_DICT`.

### 10.2 Memory During Checkpoint Load (Resume)

**Pseudocode 18: Memory-Safe Checkpoint Loading**

```
PROCEDURE MEMORY_SAFE_LOAD(ckpt_path, model, optimizer, config):
    // ═══════════════════════════════════════════
    // Approach: Load to CPU first, then move to GPU in chunks
    // This avoids doubling GPU memory during load
    // ═══════════════════════════════════════════
    
    // Step 1: Load checkpoint to CPU (uses host RAM, not HBM)
    ckpt ← LOAD(ckpt_path, map_location="cpu")
    
    // Step 2: Load model parameters one-by-one
    FOR name, param IN model.named_parameters():
        IF name IN ckpt["model"] THEN
            // Copy CPU tensor to GPU, replacing existing param data
            param.data.copy_(ckpt["model"][name].to(param.device))
            // Free CPU tensor immediately
            DELETE ckpt["model"][name]
        ELSE
            WARN "Parameter " + name + " not found in checkpoint"
        END IF
    END FOR
    
    // Step 3: Load optimizer state chunk-by-chunk
    IF config.uses_deepspeed THEN
        // DeepSpeed manages its own loading
        // For ZeRO-3: parameters are loaded as flat partitions
        // Each rank loads ONLY its partition
        model_engine.load_checkpoint(ckpt_dir, load_optimizer_states=True)
    ELSE
        // Direct optimizer state loading
        // Load one param group at a time to limit peak memory
        FOR group_idx, group IN ENUMERATE(optimizer.param_groups):
            IF group_idx IN ckpt["optimizer"]["state"] THEN
                FOR key IN ["exp_avg", "exp_avg_sq"]:
                    state_tensor ← ckpt["optimizer"]["state"][group_idx][key]
                    optimizer.state[group["params"][0]][key].copy_(
                        state_tensor.to(optimizer.state[group["params"][0]][key].device)
                    )
                    DELETE state_tensor
                END FOR
            END IF
        END FOR
    END IF
    
    // Step 4: Force garbage collection
    DELETE ckpt
    GC.COLLECT()
    torch.cuda.empty_cache()
    
    LOG "Checkpoint loaded. GPU memory: " + torch.cuda.memory_allocated() + " / " + 
        torch.cuda.max_memory_allocated()
```

### 10.3 Memory During Conversion

Full checkpoint conversion (e.g., Megatron → DeepSpeed) requires materializing the **entire unsharded model** in memory:

$$
M_{\text{conversion}}^{\text{peak}} = \underbrace{2\Phi}_{\text{source shards (BF16)}} + \underbrace{2\Phi}_{\text{full model (BF16)}} + \underbrace{2\Phi}_{\text{target shards (BF16)}} + \underbrace{12\Phi}_{\text{optimizer (FP32, if converting)}}
$$

For a 70B model: $M_{\text{conversion}} \approx 4 \times 140\text{GB} + 840\text{GB} \approx 1.4\text{TB}$

> **This exceeds single-node memory for large models.** Solutions:
> 1. **Layer-by-layer streaming conversion:** Process one layer at a time, never materializing the full model.
> 2. **Distributed conversion:** Use multiple CPU nodes with shared filesystem.
> 3. **Omit optimizer state:** Convert model weights only; reinitialize optimizer on the target framework (acceptable for continued pretraining, not ideal for fine-tuning resumption).

**Pseudocode 19: Memory-Efficient Layer-by-Layer Conversion**

```
PROCEDURE STREAMING_CONVERT(source_dir, target_dir, source_format, target_format,
                              source_config, target_config, total_layers):
    // Convert one layer at a time to limit peak memory
    
    // Phase 1: Handle non-layer parameters (embedding, output head, norms)
    embed_state ← LOAD_SINGLE_PARAM(source_dir, "embedding", source_format, source_config)
    embed_converted ← CONVERT_AND_RESHARD(embed_state, "embedding", target_format, target_config)
    SAVE_SINGLE_PARAM(embed_converted, target_dir, "embedding", target_format, target_config)
    DELETE embed_state, embed_converted
    GC.COLLECT()
    
    // Phase 2: Process each transformer layer independently
    FOR layer_idx IN 0..total_layers-1:
        LOG "Converting layer " + layer_idx + " / " + total_layers
        
        // Load only this layer's parameters from source
        layer_state ← LOAD_LAYER_PARAMS(source_dir, layer_idx, source_format, source_config)
        // Peak memory: ~2 × (parameters_per_layer) × dtype_size
        // For 70B model: ~2 × (12h² / L) × 2 bytes ≈ 2 × 245MB ≈ 490MB per layer
        
        // Reconstruct full (un-TP-sharded) layer
        full_layer ← UNSHARD_TP(layer_state, source_config.tp)
        DELETE layer_state
        
        // Re-shard for target topology
        target_layer ← RESHARD_FOR_TARGET(full_layer, target_config)
        DELETE full_layer
        
        // Convert key names
        target_layer ← MAP_KEYS(target_layer, source_format, target_format)
        
        // Save target layer shards
        SAVE_LAYER_PARAMS(target_layer, target_dir, layer_idx, target_format, target_config)
        DELETE target_layer
        
        GC.COLLECT()
    END FOR
    
    // Phase 3: Handle output layer and final norm
    // (Similar to Phase 1)
    
    // Phase 4: Write metadata
    WRITE_TARGET_METADATA(target_dir, target_format, target_config)
    
    LOG "Streaming conversion complete. Peak CPU memory: " + GET_PEAK_CPU_MEMORY()
```

---

## 11. Communication Group Management

### 11.1 Group Lifecycle in Mixed Deployments

When Megatron-Core and DeepSpeed are composed, the process group lifecycle must be carefully managed:

**Pseudocode 20: Complete Group Lifecycle Management**

```
PROCEDURE MANAGE_PROCESS_GROUPS(config):
    // ═══════════════════════════════════════════
    // Phase 1: Base initialization (PyTorch distributed)
    // ═══════════════════════════════════════════
    torch.distributed.init_process_group(
        backend = "nccl",          // or "nccl:rccl" for AMD
        init_method = "env://",
        world_size = config.world_size,
        rank = config.global_rank,
        timeout = timedelta(minutes=30)  // Large timeout for checkpoint loading
    )
    
    // ═══════════════════════════════════════════
    // Phase 2: Megatron-Core group construction
    // Creates: TP, PP, DP, CP, EP groups
    // ═══════════════════════════════════════════
    mpu.initialize_model_parallel(
        tensor_model_parallel_size = config.tp,
        pipeline_model_parallel_size = config.pp,
        context_parallel_size = config.cp,
        expert_model_parallel_size = config.ep,
        virtual_pipeline_model_parallel_size = config.virtual_pp
    )
    
    // Record Megatron's group assignments
    megatron_groups ← {
        "dp": mpu.get_data_parallel_group(),
        "tp": mpu.get_tensor_model_parallel_group(),
        "pp": mpu.get_pipeline_model_parallel_group(),
        "dp_world_size": mpu.get_data_parallel_world_size(),
        "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
        "pp_world_size": mpu.get_pipeline_model_parallel_world_size()
    }
    
    // ═══════════════════════════════════════════
    // Phase 3: DeepSpeed group initialization
    // Must reuse Megatron's groups, NOT create new ones
    // ═══════════════════════════════════════════
    IF config.uses_deepspeed THEN
        // DeepSpeed's initialize() will create its own DP group
        // We must pass mpu to make it use Megatron's
        
        // Verify DeepSpeed will use correct DP group
        ASSERT config.ds_config["train_micro_batch_size_per_gpu"] IS SET
        ASSERT config.ds_config["gradient_accumulation_steps"] IS SET
        
        // Critical: gradient_accumulation_steps × micro_batch × dp_world_size = global_batch
        expected_global_batch ← (
            config.ds_config["train_micro_batch_size_per_gpu"] ×
            config.ds_config["gradient_accumulation_steps"] ×
            megatron_groups["dp_world_size"]
        )
        ASSERT expected_global_batch = config.global_batch_size,
            "Batch size mismatch: DS expects " + expected_global_batch +
            " but config specifies " + config.global_batch_size
    END IF
    
    // ═══════════════════════════════════════════
    // Phase 4: FSDP mesh construction (if used)
    // ═══════════════════════════════════════════
    IF config.uses_fsdp THEN
        // Extract rank lists from Megatron's DP group
        dp_ranks ← GET_ALL_RANKS(megatron_groups["dp"])
        
        // Create DeviceMesh that matches Megatron's DP group
        dp_mesh ← DeviceMesh("cuda", dp_ranks, dim_names=["dp"])
        
        // If also using TP with DTensor:
        IF config.fsdp_tp_compose THEN
            tp_ranks ← GET_ALL_RANKS(megatron_groups["tp"])
            mesh_2d ← DeviceMesh("cuda", 
                [[tp_ranks[j] + dp_ranks[i] * config.tp 
                  FOR j IN range(config.tp)] 
                 FOR i IN range(config.dp)],
                dim_names=["dp", "tp"])
        END IF
    END IF
    
    // ═══════════════════════════════════════════
    // Phase 5: Group consistency verification
    // ═══════════════════════════════════════════
    VERIFY_PROCESS_GROUP_COMPATIBILITY(megatron_groups, 
                                        deepspeed_groups IF config.uses_deepspeed ELSE NONE,
                                        fsdp_mesh IF config.uses_fsdp ELSE NONE)
    
    RETURN megatron_groups
```

### 11.2 Collective Operation Ordering Constraints

When multiple frameworks issue collectives on overlapping process groups, **strict ordering** must be maintained to prevent deadlocks:

| Scenario | Potential Deadlock | Resolution |
|---|---|---|
| Megatron TP all-reduce + DeepSpeed DP all-reduce on same GPU | No (different groups) | Safe if groups are disjoint |
| Megatron DP all-reduce + DeepSpeed ZeRO-2 reduce-scatter on same DP group | **Yes** — both issue on same group | Use only ONE framework for DP gradient sync |
| FSDP all-gather + Megatron TP all-reduce | No (different groups) | Safe |
| DeepSpeed ZeRO-3 all-gather + FSDP all-gather on same group | **Yes** — double all-gather | Never compose ZeRO-3 + FSDP FULL_SHARD |

> **Rule:** For any given process group, **exactly one framework** should issue collectives on it. Megatron-Core owns TP and PP groups. DeepSpeed OR FSDP (not both) owns the DP group for gradient/parameter synchronization.

---

## 12. Numerical Consistency and Validation

### 12.1 Conversion Validation Protocol

After any checkpoint conversion, numerical validation must confirm that the converted checkpoint produces identical outputs:

**Pseudocode 21: Checkpoint Conversion Validation**

```
PROCEDURE VALIDATE_CONVERSION(source_ckpt, target_ckpt, source_framework, target_framework,
                                model_config, validation_data):
    // ═══════════════════════════════════════════
    // TEST 1: Parameter checksum validation
    // ═══════════════════════════════════════════
    source_params ← CONSOLIDATE_TO_FULL(source_ckpt, source_framework)
    target_params ← CONSOLIDATE_TO_FULL(target_ckpt, target_framework)
    
    FOR each key IN CANONICAL_KEYS(model_config):
        source_key ← MAP_TO_FRAMEWORK_KEY(key, source_framework)
        target_key ← MAP_TO_FRAMEWORK_KEY(key, target_framework)
        
        source_tensor ← source_params[source_key]
        target_tensor ← target_params[target_key]
        
        // Exact match (no tolerance — conversion should be lossless for same dtype)
        IF source_tensor.dtype = target_tensor.dtype THEN
            ASSERT EXACT_EQUAL(source_tensor, target_tensor),
                "Conversion error in " + key + 
                ": max_diff=" + MAX_ABS_DIFF(source_tensor, target_tensor)
        ELSE
            // Different dtypes (e.g., BF16 → FP32): use appropriate tolerance
            ASSERT ALLCLOSE(source_tensor.float(), target_tensor.float(), 
                           atol=1e-6, rtol=1e-5),
                "Conversion error in " + key
        END IF
    END FOR
    LOG "TEST 1 PASSED: All parameters match exactly"
    
    // ═══════════════════════════════════════════
    // TEST 2: Forward pass output validation
    // ═══════════════════════════════════════════
    // Load both checkpoints into their respective frameworks
    source_model ← LOAD_MODEL(source_ckpt, source_framework, model_config)
    target_model ← LOAD_MODEL(target_ckpt, target_framework, model_config)
    
    // Set both to eval mode
    source_model.eval()
    target_model.eval()
    
    // Run inference on same input
    input_ids ← validation_data[0]["input_ids"].cuda()
    
    WITH torch.no_grad():
        source_logits ← source_model(input_ids).logits
        target_logits ← target_model(input_ids).logits
    
    // Check output similarity (allow small numerical differences from different operation ordering)
    max_diff ← MAX(ABS(source_logits - target_logits))
    mean_diff ← MEAN(ABS(source_logits - target_logits))
    cosine_sim ← COSINE_SIMILARITY(source_logits.flatten(), target_logits.flatten())
    
    LOG "Max logit diff: " + max_diff
    LOG "Mean logit diff: " + mean_diff
    LOG "Cosine similarity: " + cosine_sim
    
    // Thresholds (for BF16 inference)
    ASSERT max_diff < 0.01, "Logit divergence too large: " + max_diff
    ASSERT cosine_sim > 0.9999, "Cosine similarity too low: " + cosine_sim
    LOG "TEST 2 PASSED: Forward pass outputs match within tolerance"
    
    // ═══════════════════════════════════════════
    // TEST 3: Loss validation (single training step)
    // ═══════════════════════════════════════════
    source_model.train()
    target_model.train()
    
    source_loss ← COMPUTE_LOSS(source_model, validation_data[0])
    target_loss ← COMPUTE_LOSS(target_model, validation_data[0])
    
    loss_diff ← ABS(source_loss - target_loss) / ABS(source_loss)
    ASSERT loss_diff < 0.001, "Loss divergence: " + loss_diff
    LOG "TEST 3 PASSED: Training loss matches within 0.1%"
    
    // ═══════════════════════════════════════════
    // TEST 4: Gradient norm validation (if optimizer state converted)
    // ═══════════════════════════════════════════
    source_loss.backward()
    target_loss.backward()
    
    source_grad_norm ← COMPUTE_GRAD_NORM(source_model)
    target_grad_norm ← COMPUTE_GRAD_NORM(target_model)
    
    grad_norm_diff ← ABS(source_grad_norm - target_grad_norm) / ABS(source_grad_norm)
    ASSERT grad_norm_diff < 0.01, "Gradient norm divergence: " + grad_norm_diff
    LOG "TEST 4 PASSED: Gradient norms match within 1%"
    
    // ═══════════════════════════════════════════
    // TEST 5: Optimizer state validation (if converted)
    // ═══════════════════════════════════════════
    IF OPTIMIZER_STATE_CONVERTED THEN
        source_optim ← CONSOLIDATE_OPTIMIZER(source_ckpt, source_framework)
        target_optim ← CONSOLIDATE_OPTIMIZER(target_ckpt, target_framework)
        
        FOR each param_key IN source_optim:
            FOR state_key IN ["exp_avg", "exp_avg_sq"]:
                ASSERT ALLCLOSE(
                    source_optim[param_key][state_key],
                    target_optim[param_key][state_key],
                    atol=1e-6, rtol=1e-5
                ), "Optimizer state mismatch: " + param_key + "." + state_key
            END FOR
        END FOR
        LOG "TEST 5 PASSED: Optimizer states match"
    END IF
    
    RETURN VALIDATION_PASSED
```

### 12.2 Precision Handling During Conversion

| Source Precision | Target Precision | Conversion Strategy | Risk |
|---|---|---|---|
| BF16 → BF16 | Same dtype | Exact copy (lossless) | None |
| BF16 → FP32 | Upcast | `tensor.float()` (lossless for values in BF16 range) | None |
| FP32 → BF16 | Downcast | `tensor.bfloat16()` (lossy — rounding) | Accumulated rounding error over many parameters |
| FP8 → BF16 | Upcast + dequant | Apply scaling factor, cast | Scaling factor must be preserved |
| BF16 → FP8 | Quant + downcast | Compute amax, apply scaling, cast | Loss of precision; must validate loss parity |
| Mixed (FP32 master + BF16 params) | Target framework's convention | Keep FP32 masters for optimizer; cast params | Must preserve FP32 masters for optimizer continuity |

> **Critical:** When converting optimizer states, **always maintain FP32 precision** for momentum ($m_t$) and variance ($v_t$). Downcasting optimizer states to BF16 causes catastrophic training instability because the Adam update rule:
>
> $$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}$$
>
> requires the fine-grained precision of FP32 to correctly compute the ratio $m_t / \sqrt{v_t}$, especially when $v_t$ values span many orders of magnitude.

---

## 13. Production Automation for Multi-Framework Deployments

### 13.1 Automated Conversion Pipeline

**Pseudocode 22: End-to-End Automated Conversion and Validation**

```
PROCEDURE AUTOMATED_CONVERSION_PIPELINE(source_ckpt_dir, target_dir,
                                          source_format, target_format,
                                          source_config, target_config,
                                          validation_data_path):
    // ═══════════════════════════════════════════
    // Phase 1: Pre-flight checks
    // ═══════════════════════════════════════════
    VERIFY_SOURCE_CHECKPOINT_INTEGRITY(source_ckpt_dir, source_format, source_config)
    VERIFY_TARGET_DIRECTORY_WRITABLE(target_dir)
    VERIFY_SUFFICIENT_DISK_SPACE(target_dir, ESTIMATE_TARGET_SIZE(source_config, target_config))
    VERIFY_SUFFICIENT_CPU_MEMORY(ESTIMATE_PEAK_MEMORY(source_config, target_config))
    
    // ═══════════════════════════════════════════
    // Phase 2: Determine conversion strategy
    // ═══════════════════════════════════════════
    model_size ← ESTIMATE_MODEL_SIZE(source_config)
    available_memory ← GET_AVAILABLE_CPU_MEMORY()
    
    IF model_size × 4 < available_memory THEN
        // Full consolidation fits in memory
        strategy ← "full_consolidation"
    ELSE
        // Must use streaming conversion
        strategy ← "streaming_layer_by_layer"
    END IF
    
    LOG "Conversion strategy: " + strategy
    LOG "Model size: " + model_size / 1e9 + " GB"
    LOG "Available memory: " + available_memory / 1e9 + " GB"
    
    // ═══════════════════════════════════════════
    // Phase 3: Execute conversion
    // ═══════════════════════════════════════════
    timer_start ← NOW()
    
    IF strategy = "full_consolidation" THEN
        CALL UNIVERSAL_CONVERT(source_ckpt_dir, target_dir,
                                source_format, target_format,
                                source_config, target_config)
    ELSE
        CALL STREAMING_CONVERT(source_ckpt_dir, target_dir,
                                source_format, target_format,
                                source_config, target_config,
                                source_config.total_layers)
    END IF
    
    conversion_time ← NOW() - timer_start
    LOG "Conversion completed in " + conversion_time + " seconds"
    
    // ═══════════════════════════════════════════
    // Phase 4: Validate conversion
    // ═══════════════════════════════════════════
    validation_data ← LOAD_VALIDATION_DATA(validation_data_path)
    
    validation_result ← VALIDATE_CONVERSION(
        source_ckpt_dir, target_dir,
        source_format, target_format,
        source_config.model_config, validation_data
    )
    
    IF validation_result ≠ PASSED THEN
        ERROR "Conversion validation failed. Target checkpoint may be corrupt."
        // DO NOT delete source checkpoint
        RETURN FAILURE
    END IF
    
    // ═══════════════════════════════════════════
    // Phase 5: Generate conversion report
    // ═══════════════════════════════════════════
    report ← {
        "source": {
            "path": source_ckpt_dir,
            "format": source_format,
            "config": source_config,
            "checksum": COMPUTE_CHECKSUM(source_ckpt_dir)
        },
        "target": {
            "path": target_dir,
            "format": target_format,
            "config": target_config,
            "checksum": COMPUTE_CHECKSUM(target_dir)
        },
        "conversion": {
            "strategy": strategy,
            "duration_seconds": conversion_time,
            "peak_memory_gb": GET_PEAK_MEMORY_USAGE() / 1e9,
            "validation": validation_result
        }
    }
    
    SAVE(report, target_dir + "/conversion_report.json")
    
    RETURN SUCCESS
```

### 13.2 Resume Configuration Synthesis

**Pseudocode 23: Automatic Resume Configuration Generation**

```
PROCEDURE SYNTHESIZE_RESUME_CONFIG(ckpt_dir, target_cluster_spec):
    // ═══════════════════════════════════════════
    // STEP 1: Detect checkpoint format and topology
    // ═══════════════════════════════════════════
    ckpt_format ← DETECT_CHECKPOINT_FORMAT(ckpt_dir)
    // Detection heuristics:
    //   - If "mp_rank_*_*/model_optim_rng.pt" exists → Megatron-Core
    //   - If "zero_pp_rank_*_mp_rank_*_optim_states.pt" exists → DeepSpeed
    //   - If ".metadata" and "*.distcp" exist → FSDP/DCP
    //   - If "model.safetensors" or "pytorch_model*.bin" exist → HuggingFace
    
    source_config ← EXTRACT_CONFIG_FROM_CHECKPOINT(ckpt_dir, ckpt_format)
    // Extract: tp, pp, dp, zero_stage, iteration, model_config
    
    // ═══════════════════════════════════════════
    // STEP 2: Determine target topology
    // ═══════════════════════════════════════════
    target_gpus ← target_cluster_spec.total_gpus
    target_hardware ← target_cluster_spec.gpu_type  // A100, H100, MI300X, etc.
    target_hbm ← GET_HBM_CAPACITY(target_hardware)
    
    // ═══════════════════════════════════════════
    // STEP 3: Check if identical topology resume is possible
    // ═══════════════════════════════════════════
    source_total_gpus ← source_config.tp × source_config.pp × source_config.dp
    
    IF target_gpus = source_total_gpus THEN
        // Same world size — prefer identical topology if it fits
        fit ← MEMORY_CHECK(source_config, target_hbm)
        IF fit THEN
            LOG "Identical topology resume possible"
            RETURN {
                "resume_type": "identical",
                "config": source_config,
                "conversion_needed": False
            }
        END IF
    END IF
    
    // ═══════════════════════════════════════════
    // STEP 4: Compute optimal target parallelism
    // ═══════════════════════════════════════════
    target_tp ← OPTIMAL_TP(source_config.model, target_hardware)
    target_pp ← OPTIMAL_PP(source_config.model, target_hardware, target_tp, target_hbm)
    target_dp ← target_gpus / (target_tp × target_pp)
    
    ASSERT target_dp ≥ 1
    
    // ═══════════════════════════════════════════
    // STEP 5: Determine conversion requirements
    // ═══════════════════════════════════════════
    needs_tp_reshard ← target_tp ≠ source_config.tp
    needs_pp_reshard ← target_pp ≠ source_config.pp
    needs_dp_reshard ← target_dp ≠ source_config.dp
    needs_framework_convert ← target_cluster_spec.target_framework ≠ ckpt_format
    
    conversion_steps ← []
    IF needs_framework_convert THEN
        conversion_steps.append("framework_conversion")
    END IF
    IF needs_tp_reshard THEN
        conversion_steps.append("tp_resharding: " + source_config.tp + " → " + target_tp)
    END IF
    IF needs_pp_reshard THEN
        conversion_steps.append("pp_resharding: " + source_config.pp + " → " + target_pp)
    END IF
    IF needs_dp_reshard THEN
        conversion_steps.append("dp_resharding: " + source_config.dp + " → " + target_dp)
    END IF
    
    LOG "Resume plan:"
    LOG "  Source: " + ckpt_format + " (TP=" + source_config.tp + ", PP=" + source_config.pp + 
        ", DP=" + source_config.dp + ")"
    LOG "  Target: " + target_cluster_spec.target_framework + " (TP=" + target_tp + 
        ", PP=" + target_pp + ", DP=" + target_dp + ")"
    LOG "  Steps: " + conversion_steps
    
    RETURN {
        "resume_type": "cross_topology",
        "source_config": source_config,
        "target_config": {tp: target_tp, pp: target_pp, dp: target_dp,
                          framework: target_cluster_spec.target_framework},
        "conversion_steps": conversion_steps,
        "estimated_conversion_time": ESTIMATE_TIME(source_config.model_size, conversion_steps)
    }
```

---

## 14. Failure Modes and Diagnostic Reference

### 14.1 Common Interoperability Failures

**Table: Failure Mode Catalog**

| Failure Mode | Symptom | Root Cause | Diagnostic | Resolution |
|---|---|---|---|---|
| **DP group mismatch** | NaN loss immediately after resume | DeepSpeed created its own DP group instead of using Megatron's | Compare `model_engine.dp_world_size` with `mpu.get_data_parallel_world_size()` | Pass `mpu` to `deepspeed.initialize()`; set `dist_init_required=False` |
| **Key name mismatch** | `RuntimeError: unexpected key` on load | Framework prefixes differ (`module.` from DDP wrapper) | Print `state_dict.keys()` and compare | Use `strict=False` + key mapping function |
| **TP shard shape mismatch** | `RuntimeError: shape mismatch` on load | Source and target TP degrees differ without resharding | Check `saved_args.tensor_model_parallel_size` vs current config | Run TP resharding conversion before load |
| **ZeRO flat offset corruption** | NaN loss after resume; params look wrong | ZeRO-3 metadata misaligned with actual flat buffer | Validate `param_offsets` sum equals flat buffer length | Re-generate ZeRO checkpoint from consolidated state |
| **QKV split mismatch** | Correct loss but wrong attention pattern | Source uses fused QKV; target expects split Q, K, V | Check QKV weight shapes: $[h, 3h]$ vs 3× $[h, h]$ | Add QKV split/merge to conversion logic |
| **GQA head count mismatch** | Shape error in attention | Source GQA has $n_{kv} \neq n_h$; conversion assumes $n_{kv} = n_h$ | Verify Q vs K/V head count in model config | Use GQA-aware splitting logic |
| **Optimizer index shift** | Training diverges after resume (high loss for 100s of steps) | Optimizer param indices shifted due to key reordering | Compare param ordering in source vs target | Re-map optimizer state keys by parameter name, not index |
| **RNG state missing** | Non-deterministic behavior after resume | Framework-specific RNG tracker not saved | Check for `tracker_rng_state` in checkpoint | Save Megatron's `CudaRNGStatesTracker` state |
| **Deadlock on load** | Training hangs during checkpoint load | Some ranks load from different checkpoint iteration | Ensure all ranks read `latest_checkpointed_iteration.txt` | Barrier after iteration agreement |
| **OOM during conversion** | Python killed during checkpoint conversion | Full model exceeds CPU/GPU memory | Monitor `RSS` during conversion | Use streaming layer-by-layer conversion |
| **Double gradient sync** | 2× expected gradient all-reduce time | Both Megatron DDP and DeepSpeed issuing gradient sync | Profile with `nsys` / `rocprof`; count all-reduce calls per step | Disable Megatron's gradient sync when DeepSpeed manages DP |
| **PP stage assignment mismatch** | Some layers missing in converted checkpoint | Source and target assign different layers to stages | Log layer-to-stage mapping for both | Ensure consistent `layers_per_stage = L / p` |

### 14.2 Diagnostic Pseudocode

**Pseudocode 24: Checkpoint Integrity Diagnostic**

```
PROCEDURE DIAGNOSE_CHECKPOINT(ckpt_dir):
    format ← DETECT_FORMAT(ckpt_dir)
    LOG "Format: " + format
    
    // ═══════════════════════════════════════════
    // CHECK 1: File completeness
    // ═══════════════════════════════════════════
    expected_files ← COMPUTE_EXPECTED_FILES(format, ckpt_dir)
    actual_files ← LIST_FILES(ckpt_dir)
    missing ← expected_files - actual_files
    extra ← actual_files - expected_files
    
    IF |missing| > 0 THEN
        ERROR "Missing files: " + missing
    END IF
    
    // ═══════════════════════════════════════════
    // CHECK 2: File loadability
    // ═══════════════════════════════════════════
    FOR each file IN actual_files:
        TRY:
            data ← LOAD(file, map_location="cpu")
            LOG "OK: " + file + " (" + FILE_SIZE(file) + " bytes)"
        CATCH:
            ERROR "Corrupt file: " + file
        END TRY
    END FOR
    
    // ═══════════════════════════════════════════
    // CHECK 3: Parameter consistency across shards
    // ═══════════════════════════════════════════
    IF format = "megatron-core" THEN
        // Verify replicated params are identical across TP ranks
        FOR pp_rank IN 0..pp-1:
            replicated_params ← GET_REPLICATED_PARAMS(format)
            FOR each param_name IN replicated_params:
                values ← []
                FOR tp_rank IN 0..tp-1:
                    v ← LOAD_PARAM(ckpt_dir, tp_rank, pp_rank, param_name)
                    values.append(v)
                END FOR
                
                FOR i IN 1..tp-1:
                    IF NOT EXACT_EQUAL(values[0], values[i]) THEN
                        ERROR "Replicated param " + param_name + 
                              " differs between TP rank 0 and " + i
                        LOG "Max diff: " + MAX_ABS_DIFF(values[0], values[i])
                    END IF
                END FOR
            END FOR
        END FOR
        LOG "CHECK 3 PASSED: Replicated parameters are consistent"
    END IF
    
    // ═══════════════════════════════════════════
    // CHECK 4: NaN/Inf detection
    // ═══════════════════════════════════════════
    FOR each file IN actual_files:
        data ← LOAD(file, map_location="cpu")
        FOR key, tensor IN EXTRACT_TENSORS(data):
            nan_count ← COUNT_NAN(tensor)
            inf_count ← COUNT_INF(tensor)
            IF nan_count > 0 THEN
                ERROR "NaN detected: " + file + "/" + key + " (" + nan_count + " elements)"
            END IF
            IF inf_count > 0 THEN
                ERROR "Inf detected: " + file + "/" + key + " (" + inf_count + " elements)"
            END IF
        END FOR
    END FOR
    LOG "CHECK 4 PASSED: No NaN/Inf values detected"
    
    // ═══════════════════════════════════════════
    // CHECK 5: Shape consistency with expected model config
    // ═══════════════════════════════════════════
    model_config ← EXTRACT_MODEL_CONFIG(ckpt_dir)
    FOR each param_name, expected_shape IN EXPECTED_SHAPES(model_config, format):
        actual_shape ← GET_PARAM_SHAPE(ckpt_dir, param_name)
        IF actual_shape ≠ expected_shape THEN
            ERROR "Shape mismatch: " + param_name + 
                  " expected " + expected_shape + " got " + actual_shape
        END IF
    END FOR
    LOG "CHECK 5 PASSED: All parameter shapes match expected configuration"
    
    // ═══════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════
    total_params ← COMPUTE_TOTAL_PARAMS(ckpt_dir, format)
    LOG "Total parameters: " + total_params / 1e9 + "B"
    LOG "Checkpoint size on disk: " + TOTAL_DISK_SIZE(ckpt_dir) / 1e9 + " GB"
    LOG "Iteration: " + GET_ITERATION(ckpt_dir)
    LOG "All checks passed. Checkpoint is valid."
```

---

## 15. Summary: Interoperability Decision Framework

### 15.1 When to Use Each Pattern

| Scenario | Recommended Pattern | Rationale |
|---|---|---|
| Large-scale dense pretraining (>100B params) | Megatron-Core + DeepSpeed ZeRO-1 | Best MFU; proven at scale; ZeRO-1 has minimal overhead |
| Memory-constrained pretraining (smaller cluster) | Megatron-Core + DeepSpeed ZeRO-3 | Maximum memory savings; trades communication for memory |
| Compiler-optimized training (with `torch.compile`) | Megatron-Core TP + FSDP (DTensor) | Native compiler support; SimpleFSDP overlap optimization |
| Fine-tuning (SFT/DPO) | FSDP or DeepSpeed ZeRO-3 standalone | Simpler setup; model fits with fewer parallelism dimensions |
| Cross-vendor portability (NVIDIA + AMD) | PyTorch FSDP + Triton kernels | Vendor-neutral; avoids TE/NCCL-specific features |
| MoE pretraining | Megatron-Core + DeepSpeed MoE | Megatron TP/PP for dense; DeepSpeed EP for experts |
| Elastic training (cloud, spot instances) | DeepSpeed Elastic + DCP checkpoints | Built-in elastic recovery; DCP handles resharding |
| Checkpoint portability (export/import) | HuggingFace safetensors intermediate | Universal format; widest ecosystem compatibility |

### 15.2 Conversion Path Selection

```
IF source = target THEN
    USE identical_topology_resume (fastest)
ELSE IF only DP changed THEN
    IF using FSDP/DCP THEN
        USE native DCP resharding (no conversion needed)
    ELSE
        USE ZeRO partition re-slicing
    END IF
ELSE IF TP or PP changed (same framework) THEN
    USE framework-specific resharding scripts
ELSE IF framework changed THEN
    USE universal_convert via consolidated intermediate
    IF model > available_memory / 4 THEN
        USE streaming_layer_by_layer conversion
    END IF
END IF

ALWAYS: validate with VALIDATE_CONVERSION after any conversion
```

### 15.3 Critical Rules Summary

1. **Never compose ZeRO-3 with FSDP FULL_SHARD** — double parameter sharding causes correctness failures.
2. **Initialize Megatron-Core process groups BEFORE DeepSpeed** — Megatron owns TP/PP groups.
3. **Pass `mpu` to `deepspeed.initialize()`** — ensures DeepSpeed uses Megatron's DP group.
4. **Set `dist_init_required=False`** in DeepSpeed when Megatron already initialized `torch.distributed`.
5. **Always validate converted checkpoints** — parameter checksums, forward pass outputs, and loss parity.
6. **Keep FP32 precision for optimizer states during conversion** — downcasting causes training instability.
7. **Use layer-by-layer streaming conversion for models >100B** — avoids OOM during conversion.
8. **Save checkpoint metadata (framework versions, topology, config)** alongside checkpoint files — essential for future conversion.
9. **One framework per process group for collectives** — prevents deadlocks and double synchronization.
10. **QKV split/merge and GQA head handling** are the most common sources of silent conversion errors — always verify attention output shapes.

---

*This report provides a complete, production-grade reference for Megatron-Core, DeepSpeed, and FSDP interoperability across all aspects of training, resume, resharding, and conversion. Every pseudocode algorithm, format specification, and diagnostic procedure is derived from first principles and reflects real-world production deployment experience at scale.*