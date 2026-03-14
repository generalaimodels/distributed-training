

# SOTAFSDP2: Production-Grade Fully Sharded Data Parallelism Engine

## Comprehensive Technical Report on Architecture, Failure Modes, Device Affinity, Checkpoint Safety, and Cross-Vendor Distributed Training

---

## Table of Contents

1. [Architectural Overview and Design Philosophy](#1-architectural-overview-and-design-philosophy)
2. [FSDP2 Memory Sharding: First-Principles Analysis](#2-fsdp2-memory-sharding-first-principles-analysis)
3. [The Device Affinity Problem and Its Resolution](#3-the-device-affinity-problem-and-its-resolution)
4. [Checkpoint Pipeline: Modern DCP vs. Legacy FSDP1 APIs](#4-checkpoint-pipeline-modern-dcp-vs-legacy-fsdp1-apis)
5. [Memory Management: Pressure-Aware Pooling and Fragmentation Control](#5-memory-management-pressure-aware-pooling-and-fragmentation-control)
6. [Stream Management: Communication-Computation Overlap Architecture](#6-stream-management-communication-computation-overlap-architecture)
7. [Gradient Accumulation: FSDP-Safe no_sync Protocol](#7-gradient-accumulation-fsdp-safe-no_sync-protocol)
8. [Mixed Precision: Numerical Robustness Across Precision Policies](#8-mixed-precision-numerical-robustness-across-precision-policies)
9. [Hardware Detection and Cross-Vendor Portability](#9-hardware-detection-and-cross-vendor-portability)
10. [Triton Kernel Integration: Fused Collective Primitives](#10-triton-kernel-integration-fused-collective-primitives)
11. [Metrics Collection: Zero-Stall GPU Instrumentation](#11-metrics-collection-zero-stall-gpu-instrumentation)
12. [Activation Checkpointing: Selective Recomputation Strategies](#12-activation-checkpointing-selective-recomputation-strategies)
13. [Auto-Wrap Policy: Transformer-Aware Layer Discovery](#13-auto-wrap-policy-transformer-aware-layer-discovery)
14. [Failure Mode Taxonomy and Root-Cause Resolution Chain](#14-failure-mode-taxonomy-and-root-cause-resolution-chain)
15. [Production Deployment: End-to-End Integration Contract](#15-production-deployment-end-to-end-integration-contract)

---

## 1. Architectural Overview and Design Philosophy

### 1.1 System Architecture

The SOTAFSDP2 engine implements a **production-hardened FSDP orchestrator** that sits between the user's training loop (SOTATrainer) and PyTorch's `FullyShardedDataParallel` runtime. It is not a reimplementation of FSDP sharding logic; rather, it is an **integration layer** that manages the full lifecycle of sharded training—wrapping, forward/backward scheduling, gradient accumulation, checkpoint serialization, memory pressure response, and cross-vendor hardware adaptation.

The architecture follows a **layered composition** pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                       SOTATrainer                           │
│   (Training loop, data loading, logging, scheduling)        │
├─────────────────────────────────────────────────────────────┤
│                        SOTAFSDP2                            │
│   ┌───────────┐ ┌──────────────┐ ┌────────────────────┐    │
│   │ StreamMgr │ │ MixedPrecCtx │ │ GradientAccumulator │    │
│   └───────────┘ └──────────────┘ └────────────────────┘    │
│   ┌───────────┐ ┌──────────────┐ ┌────────────────────┐    │
│   │ MemoryPool│ │ MetricsCollr │ │ FSDPCheckpointMgr  │    │
│   └───────────┘ └──────────────┘ └────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│              PyTorch FSDP / Distributed Runtime             │
│   (FullyShardedDataParallel, ProcessGroup, NCCL/RCCL)      │
├─────────────────────────────────────────────────────────────┤
│              Hardware Layer (CUDA / ROCm / HIP)             │
│   (A100, H100, H200, B100, B200, MI300X, MI325X, MI350)    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Explicit error handling** | `Result[T] = Ok[T] | Err[T]` — no exceptions for control flow |
| **Device affinity as invariant** | Every operation wrapped in `_enforce_device_affinity()` context |
| **Vendor-agnostic primitives** | Hardware detection → adaptive stream priorities, timing, kernels |
| **Memory as first-class variable** | Pressure-aware pool, proactive GC, watermark-triggered eviction |
| **Non-blocking instrumentation** | CUDA event timing, no `synchronize()` in hot path |
| **Graceful API degradation** | Modern DCP (PyTorch ≥ 2.3) with fallback to legacy FSDP1 API |
| **Atomic checkpointing** | tmp-file + rename pattern, single-barrier coordination |

### 1.3 Integration Contract (SOTATrainer ↔ SOTAFSDP2)

The engine exposes a strict API boundary:

| Contract ID | Method | Responsibility |
|------------|--------|----------------|
| INT-001 | `wrap_model()` | FSDP wrapping with AC, precision, offload |
| INT-003 | `backward(loss)` | Gradient accumulation, `no_sync`, loss scaling |
| INT-004 | `step(optimizer)` | Unscale → clip → step → zero_grad → reset |
| INT-006 | `FSDPCheckpointManager.save/load_checkpoint()` | Device-safe, memory-aware checkpoint I/O |
| INT-007 | `forward_context()` | Autocast + non-blocking forward timing |
| INT-009 | `memory_summary()` | Human-readable VRAM usage for logging |
| INT-010 | `backward()` returns `bool` | `True` = sync step (call `step()`); `False` = accumulating |

---

## 2. FSDP2 Memory Sharding: First-Principles Analysis

### 2.1 Sharding Strategy Memory Formulas

For a model with $\Phi$ total parameters, $W$ GPUs in the data-parallel group, and mixed-precision training with BF16 parameters and FP32 optimizer (Adam with master weights, first moment $m$, second moment $v$):

**FULL_SHARD (ZeRO-3):**

$$
M_{\text{FULL\_SHARD}}^{\text{rank}} = \frac{1}{W}\bigl(\underbrace{2\Phi}_{\text{BF16 params}} + \underbrace{2\Phi}_{\text{BF16 grads}} + \underbrace{4\Phi}_{\text{FP32 master}} + \underbrace{4\Phi}_{m} + \underbrace{4\Phi}_{v}\bigr) = \frac{16\Phi}{W}
$$

**SHARD_GRAD_OP (ZeRO-2):**

Parameters remain replicated; only gradients and optimizer states are sharded:

$$
M_{\text{SHARD\_GRAD\_OP}}^{\text{rank}} = \underbrace{2\Phi}_{\text{BF16 params (replicated)}} + \frac{1}{W}\bigl(\underbrace{2\Phi}_{\text{grads}} + \underbrace{12\Phi}_{\text{optimizer}}\bigr) = 2\Phi + \frac{14\Phi}{W}
$$

**NO_SHARD (Standard DDP):**

$$
M_{\text{NO\_SHARD}}^{\text{rank}} = 2\Phi + 2\Phi + 12\Phi = 16\Phi
$$

**HYBRID_SHARD:**

Full sharding within a node ($W_{\text{intra}}$ GPUs), replication across nodes ($W_{\text{inter}}$ node groups):

$$
M_{\text{HYBRID}}^{\text{rank}} = \frac{16\Phi}{W_{\text{intra}}}
$$

Communication cost drops from inter-node all-gather to intra-node all-gather (NVLink bandwidth), at the expense of higher per-rank memory than FULL_SHARD across all nodes.

### 2.2 Memory Per Shard Computation

The implementation computes the per-rank memory footprint after wrapping:

$$
M_{\text{shard}}^{\text{GB}} = \frac{\Phi_{\text{rank}} \cdot b_{\text{elem}}}{2^{30}}
$$

where $\Phi_{\text{rank}} = \Phi / W$ is the per-rank parameter count and $b_{\text{elem}}$ is the bytes per element for the configured `param_dtype`.

```
ALGORITHM: Compute_Per_Shard_Memory

INPUT: wrapped_model, world_size W, param_dtype
OUTPUT: memory_per_shard_gb

param_count ← SUM(p.numel() FOR p IN wrapped_model.parameters())
bytes_per_elem ← ELEMENT_SIZE(param_dtype)
Φ_rank ← param_count / W
memory_per_shard_gb ← (Φ_rank × bytes_per_elem) / 2^30

RETURN memory_per_shard_gb
```

### 2.3 VRAM Pressure Watermark

The system defines a pressure watermark at 90% of total HBM capacity:

$$
\text{VRAM\_PRESSURE\_WATERMARK} = 0.90
$$

When allocated memory exceeds this threshold, proactive GC and cache eviction are triggered:

$$
\text{IF } M_{\text{allocated}} > 0.90 \times M_{\text{HBM\_total}} \text{ THEN trigger\_gc\_and\_eviction()}
$$

This is checked:
- After every optimizer step (in `step()`)
- Before checkpoint saves (in `_pre_save_memory_cleanup()`)
- During memory pool allocation (in `_check_pressure()`)

---

## 3. The Device Affinity Problem and Its Resolution

### 3.1 Root Cause Analysis (FIX-005)

This is the single most critical production failure mode in multi-GPU FSDP training. The failure chain is:

```
FAILURE CHAIN: Device Affinity Violation

Step 1: User calls checkpoint save with offload_to_cpu=True
Step 2: FSDP.state_dict_type() internally creates staging tensors
Step 3: Staging tensors are allocated on cuda:0 (PyTorch default device)
        NOT on cuda:{local_rank} (the rank's actual device)
Step 4: state_dict() moves shard tensors to CPU via D2H transfer
Step 5: dcp.save() performs NCCL collective for metadata coordination
Step 6: NCCL discovers tensor on cuda:0 but process group bound to cuda:3
Step 7: NCCL raises:
        "RuntimeError: Tensor on cuda:0 but expected cuda:3"
Step 8: All ranks enter NCCL timeout wait (default 1800s = 30 minutes)
Step 9: Training appears frozen with no visible error for 30 minutes
Step 10: Eventually NCCL timeout fires, all ranks crash simultaneously
```

**Why cuda:0 specifically:** PyTorch's `torch.cuda.default_device()` returns `cuda:0` unless explicitly overridden. Internal FSDP staging buffers are allocated using the default device, not the rank-local device.

### 3.2 The Device Affinity Guard

The resolution is a **defense-in-depth** context manager that enforces correct device binding:

```
ALGORITHM: Enforce_Device_Affinity

CONTEXT MANAGER: _enforce_device_affinity(self)

// Entry:
    torch.cuda.set_device(self._device)          // Pin default device
    WITH torch.cuda.device(self._device):         // Override context
        YIELD                                      // Execute guarded ops

// Exit:
    IF debug_mode:
        _verify_param_devices()                    // O(P) scan for drift
```

The guard has **two layers**:
1. `torch.cuda.set_device()` — sets the global default device for this thread.
2. `torch.cuda.device()` context — overrides the device context for all tensor allocations within the scope.

### 3.3 Post-Operation Device Verification

After any operation that may cause device drift (checkpoint save/load, state dict manipulation), a verification scan detects and corrects drift:

```
ALGORITHM: Verify_Param_Devices

INPUT: wrapped_model, expected_device
OUTPUT: drift_count (side-effect: migrate drifted params)

drift_count ← 0
FOR (name, param) IN wrapped_model.named_parameters():
    IF param.is_cuda AND param.device ≠ expected_device:
        drift_count ← drift_count + 1
        IF drift_count ≤ 5:
            LOG_WARNING("Device drift: {name} on {param.device}")
        param.data ← param.data.to(expected_device)

IF drift_count > 0:
    LOG_WARNING("Migrated {drift_count} params back to {expected_device}")

RETURN drift_count
```

**Complexity:** $O(P)$ where $P$ is the number of parameters. Runs only during checkpoint operations (not in the training hot path) and only in debug mode during normal training.

### 3.4 Device Re-Pinning Pattern

Throughout the checkpoint code, device re-pinning appears after every API call that may internally allocate tensors:

```
ALGORITHM: Device_Repin_Pattern

// Before any FSDP/DCP API call:
torch.cuda.set_device(self._device)

// After API call that may cause drift:
API_CALL(...)                               // e.g., get_state_dict()
torch.cuda.set_device(self._device)         // IMMEDIATE re-pin

// Pattern repetition is INTENTIONAL:
// Each API call may internally call torch.cuda.set_device(0)
// or allocate on default device. Re-pinning after EVERY call
// ensures no drift accumulates.
```

> **Engineering rationale:** Re-pinning may appear redundant, but each is necessary because any PyTorch internal call (especially FSDP state dict operations) can reset the default device as a side effect. The cost of an unnecessary `set_device()` call is ~100ns; the cost of missing one is a 30-minute hang followed by a crash.

---

## 4. Checkpoint Pipeline: Modern DCP vs. Legacy FSDP1 APIs

### 4.1 API Version Detection

The system detects the available checkpoint API at module import time:

```
ALGORITHM: Detect_Checkpoint_API

TRY:
    IMPORT torch.distributed.checkpoint.state_dict.get_state_dict
    IMPORT torch.distributed.checkpoint.state_dict.set_state_dict
    IMPORT torch.distributed.checkpoint.state_dict.StateDictOptions
    _HAS_MODERN_DCP ← True
    // PyTorch ≥ 2.3: DTensor-based, no ShardedTensor
EXCEPT ImportError:
    _HAS_MODERN_DCP ← False
    // PyTorch < 2.3: Legacy FSDP1 API with ShardedTensor
```

### 4.2 Modern DCP Pipeline (PyTorch ≥ 2.3)

The modern DCP path eliminates all known checkpoint failure modes:

| Property | Modern DCP | Legacy FSDP1 |
|----------|-----------|-------------|
| Internal representation | DTensor | ShardedTensor (deprecated) |
| API calls for model+optimizer | 1 (`get_state_dict` with optimizers) | 2 (separate `state_dict_type` scopes) |
| All-gather collectives per save | 1 | 2 (model + optimizer) |
| Barriers per save | 1 | 4 (2 scopes × 2 barriers each) |
| Device drift risk | Minimal (DTensor tracks device) | High (ShardedTensor uses cuda:0) |
| Deprecation warnings | None | Hundreds per save |
| Approximate time (70B, 78% VRAM) | ~15s | ~50–100s (FIX-007 root cause) |

#### 4.2.1 Modern Save Flow

```
ALGORITHM: Save_Sharded_Modern

INPUT: fsdp, optimizer, checkpoint_dir, epoch, step, extra_state
OUTPUT: Result[None]

// Phase 1: Collect state using DTensor API
options ← StateDictOptions(
    full_state_dict=False,    // Sharded, not gathered
    cpu_offload=True          // Offload to CPU for write
)

// SINGLE API call for BOTH model and optimizer
(model_state, optim_state) ← get_state_dict(
    fsdp.wrapped_model,
    optimizers=[optimizer],
    options=options
)

// [FIX-005] Re-pin device after get_state_dict
torch.cuda.set_device(fsdp._device)

// Phase 2: Combine into single state dict
combined_state ← {
    "model": model_state,
    "optimizer": optim_state
}

// Phase 3: Single DCP save (one coordinated write across all ranks)
dcp.save(
    state_dict=combined_state,
    storage_writer=FileSystemWriter(checkpoint_dir, overwrite=True)
)

// Phase 4: Metadata on rank 0 only (no collective required)
IF rank == 0:
    meta ← {epoch, step, api_version="dcp_modern_v2", config}
    torch.save(meta, checkpoint_dir / "meta.pt")

RETURN Ok(None)
```

> **Why single `get_state_dict` matters:** The legacy path calls `state_dict_type()` twice—once for model, once for optimizer. Each call triggers a full all-gather across all ranks. With 78% VRAM utilization, the second all-gather can cause OOM-retry loops and NCCL retransmission, leading to the 50–100 second stalls documented in FIX-007.

#### 4.2.2 Modern Load Flow

```
ALGORITHM: Load_Sharded_Modern

INPUT: fsdp, optimizer, checkpoint_dir
OUTPUT: Result[Dict metadata]

// Phase 1: Get current state structure (DTensor shapes/placements)
options ← StateDictOptions(full_state_dict=False, cpu_offload=False)
(model_state, optim_state) ← get_state_dict(
    fsdp.wrapped_model,
    optimizers=[optimizer],
    options=options
)

// [FIX-005] Re-pin device
torch.cuda.set_device(fsdp._device)

// Phase 2: Load checkpoint data into state structure
combined_state ← {"model": model_state, "optimizer": optim_state}
dcp.load(
    state_dict=combined_state,
    storage_reader=FileSystemReader(checkpoint_dir)
)

// [FIX-005] Re-pin device
torch.cuda.set_device(fsdp._device)

// Phase 3: Apply loaded state to model and optimizer
set_state_dict(
    fsdp.wrapped_model,
    optimizers=[optimizer],
    model_state_dict=combined_state["model"],
    optim_state_dict=combined_state["optimizer"],
    options=options
)

// Phase 4: Load metadata
meta ← torch.load(checkpoint_dir / "meta.pt", map_location="cpu")

RETURN Ok(meta)
```

### 4.3 Legacy FSDP1 Pipeline (PyTorch < 2.3)

When the modern API is unavailable, the system falls back to the legacy path with comprehensive warning suppression and additional safety measures.

#### 4.3.1 Legacy Save Flow

```
ALGORITHM: Save_Sharded_Legacy

INPUT: fsdp, optimizer, checkpoint_dir, epoch, step, extra_state
OUTPUT: Result[None]

// [FIX-008] Suppress ALL legacy deprecation warnings
WITH warnings.catch_warnings():
    SUPPRESS FutureWarning("ShardedTensor")
    SUPPRESS FutureWarning("FSDP.state_dict_type")
    SUPPRESS UserWarning("_get_pg_default_device")
    SUPPRESS FutureWarning("set_state_dict_type")
    SUPPRESS UserWarning("existing checkpoint")

    // ── Model State (First Scope) ──
    model_cfg ← ShardedStateDictConfig(offload_to_cpu=True)
    WITH FSDP.state_dict_type(model, SHARDED_STATE_DICT, model_cfg):
        model_state ← model.state_dict()
    
    // [FIX-005] Re-pin device after CPU offload
    torch.cuda.set_device(fsdp._device)
    
    dcp.save({"model": model_state}, FileSystemWriter(model_dir))
    
    // [FIX-007] Free model state BEFORE optimizer all-gather
    DELETE model_state
    gc.collect()
    torch.cuda.empty_cache()
    
    // [FIX-005] Re-pin device AGAIN
    torch.cuda.set_device(fsdp._device)
    
    // ── Optimizer State (Second Scope) ──
    optim_cfg ← ShardedOptimStateDictConfig(offload_to_cpu=True)
    WITH FSDP.state_dict_type(model, SHARDED_STATE_DICT, optim_cfg):
        optim_state ← FSDP.optim_state_dict(model, optimizer)
    
    // [FIX-005] Re-pin device after CPU offload
    torch.cuda.set_device(fsdp._device)
    
    dcp.save({"optimizer": optim_state}, FileSystemWriter(optim_dir))
    
    DELETE optim_state
    gc.collect()
    torch.cuda.empty_cache()

// [FIX-005] Final device re-pin
torch.cuda.set_device(fsdp._device)

// Metadata on rank 0
IF rank == 0:
    torch.save(meta, checkpoint_dir / "meta.pt")

RETURN Ok(None)
```

**Critical difference from modern path:** The legacy path requires **two separate `state_dict_type()` scopes** (model + optimizer), each triggering independent all-gather collectives. The explicit `gc.collect() + empty_cache()` between them (FIX-007) creates VRAM headroom for the second all-gather.

### 4.4 Full State Dict Save (Portable)

For cross-world-size portability, the full state dict is gathered to rank 0:

```
ALGORITHM: Save_Full_StateDict

INPUT: fsdp, optimizer, path, epoch, step
OUTPUT: Result[None]

// [FIX-005] Pin device
torch.cuda.set_device(fsdp._device)

IF _HAS_MODERN_DCP:
    options ← StateDictOptions(full_state_dict=True, cpu_offload=True)
    (model_state, optim_state) ← get_state_dict(model, [optimizer], options)
ELSE:
    // Legacy with FullStateDictConfig(rank0_only=True)
    ...

IF rank == 0:
    checkpoint ← {model_state, optim_state, epoch, step, extra}
    
    // [FIX-007] ATOMIC WRITE: tmp + rename
    tmp_path ← path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)            // Atomic on POSIX filesystems

DELETE model_state, optim_state
gc.collect()
torch.cuda.empty_cache()

RETURN Ok(None)
```

> **Atomicity guarantee:** The tmp-file + rename pattern ensures that a crash during `torch.save()` never corrupts the checkpoint. The `rename()` syscall is atomic on POSIX filesystems (ext4, XFS, Lustre). If the process crashes between `torch.save()` and `rename()`, only the `.tmp` file exists—the original checkpoint is untouched.

### 4.5 Unified Checkpoint Save Pipeline

The complete save pipeline orchestrates all phases with device safety and memory management:

```
ALGORITHM: Unified_Save_Pipeline

INPUT: fsdp, optimizer, path, epoch, step, extra_state, sharded
OUTPUT: Result[None]

save_start ← monotonic_time()

TRY:
    // Phase 1: Create VRAM headroom
    _pre_save_memory_cleanup(fsdp)
        // Release pool slabs → gc.collect() → empty_cache()
        // Log: "Pre-save cleanup: X.XX/Y.Y GB (Z.Z% VRAM)"
    
    // Phase 2: Synchronize all ranks
    _barrier_with_timeout(fsdp, label="pre-save", timeout=300s)
        // [FIX-007] 5-minute timeout vs. default 30-minute
    
    // Phase 3: Save (under device affinity guard)
    WITH fsdp._enforce_device_affinity():
        IF sharded AND _HAS_MODERN_DCP:
            result ← _save_sharded_modern(...)
        ELSE IF sharded:
            result ← _save_sharded_legacy(...)
        ELSE:
            result ← _save_full(...)
    
    IF result.is_err():
        RETURN result
    
    // Phase 4: Verify no device drift
    _post_save_device_check(fsdp)
    
    // Phase 5: Cleanup gathered params
    gc.collect()
    torch.cuda.empty_cache()
    
    // Phase 6: Post-save barrier
    _barrier_with_timeout(fsdp, label="post-save", timeout=300s)
    
    elapsed ← monotonic_time() - save_start
    LOG("Checkpoint saved in {elapsed:.1f}s: {path}")
    
    RETURN Ok(None)

EXCEPT Exception AS e:
    LOG_ERROR("Checkpoint save failed on rank {rank}: {e}")
    // CRITICAL: Restore device affinity even on failure
    torch.cuda.set_device(fsdp._device)
    RETURN Err("Checkpoint save failed: {e}", code=1)
```

### 4.6 Warning Suppression Strategy (FIX-008)

The warning suppression follows a **fail-open policy**: only known-harmless deprecation warnings from specific PyTorch modules are suppressed; novel or unexpected warnings pass through.

| Warning | Category | Source Module | Suppression Reason |
|---------|----------|--------------|-------------------|
| `_get_pg_default_device will be deprecated` | UserWarning | `torch.distributed.distributed_c10d` | Internal API change; no user action possible |
| `FSDP.state_dict_type() being deprecated` | FutureWarning | `torch.distributed.fsdp` | We use modern API when available |
| `Please use DTensor, deprecating ShardedTensor` | FutureWarning | `torch.distributed` | We use DTensor when available |
| `Detected existing checkpoint, overwriting` | UserWarning | `torch.distributed.checkpoint` | We pass `overwrite=True` explicitly |

```
ALGORITHM: Warning_Suppression_Policy

// Called ONCE at module import time
FUNCTION _suppress_known_deprecation_warnings():
    FOR each (message_pattern, category, module_pattern) IN KNOWN_WARNINGS:
        warnings.filterwarnings(
            action="ignore",
            message=message_pattern,    // Regex match
            category=category,          // Exact class match
            module=module_pattern       // Regex match on source module
        )
    // NO catch-all suppressions
    // Unknown warnings from unknown modules → visible to user
```

---

## 5. Memory Management: Pressure-Aware Pooling and Fragmentation Control

### 5.1 Memory Pool Architecture

The `MemoryPool` eliminates `cudaMalloc` jitter through **power-of-2 bucketed pre-allocation** with pressure-aware eviction.

#### 5.1.1 Bucket Size Computation

For a requested allocation of $R$ bytes:

$$
B(R) = \begin{cases} 512 & \text{if } R \leq 512 \\ R & \text{if } R \geq 256\text{MB} \\ 2^{\lceil \log_2 R \rceil} & \text{otherwise} \end{cases}
$$

This ensures:
- **Small allocations** (< 512B) use minimum bucket size to avoid waste.
- **Large allocations** (≥ 256MB) are sized exactly (no rounding to avoid 2× memory waste at large scales).
- **Mid-range allocations** use power-of-2 rounding for bucket reuse.

#### 5.1.2 Allocation Flow

```
ALGORITHM: Pool_Allocate

INPUT: shape (tuple), dtype (optional)
OUTPUT: Tensor on GPU

num_elements ← PRODUCT(shape)
size_bytes ← num_elements × ELEMENT_SIZE(dtype)
bucket ← BUCKET_SIZE(size_bytes)

WITH lock:
    allocation_count ← allocation_count + 1
    
    // Phase 1: Check pressure and evict if necessary
    _check_pressure()
    
    // Phase 2: Try to reuse existing slab
    IF bucket IN pools AND pools[bucket] NOT EMPTY:
        slab ← pools[bucket].pop()
        IF slab.numel() ≥ num_elements:
            RETURN slab[:num_elements].view(shape)
        ELSE:
            DELETE slab  // Size mismatch (shouldn't happen with correct bucketing)
    
// Phase 3: Fresh allocation (outside lock for concurrency)
tensor ← torch.empty(num_elements, dtype=dtype, device=self._device)

WITH lock:
    total_allocated ← total_allocated + size_bytes
    peak_allocated ← MAX(peak_allocated, total_allocated)

RETURN tensor.view(shape)
```

#### 5.1.3 Pressure-Aware Eviction

```
ALGORITHM: Check_Pressure

// Called within allocation lock
allocated ← torch.cuda.memory_allocated(device)
threshold ← device_total_bytes × 0.90  // VRAM_PRESSURE_WATERMARK

IF allocated < threshold:
    RETURN  // No pressure

// Evict largest buckets first (greedy approach)
FOR bucket_size IN SORTED(pools.keys(), DESCENDING):
    WHILE pools[bucket_size] NOT EMPTY:
        tensor ← pools[bucket_size].pop()
        freed ← tensor.numel() × tensor.element_size()
        total_allocated ← total_allocated - freed
        DELETE tensor  // Triggers cudaFree
        
        IF torch.cuda.memory_allocated(device) < threshold:
            RETURN  // Pressure relieved
    
    // Remove empty bucket entry
    IF pools[bucket_size] IS EMPTY:
        DELETE pools[bucket_size]
```

**Eviction ordering:** Largest buckets are evicted first because:
1. They free the most memory per eviction.
2. Large slabs have lower reuse probability (fewer matching allocations).
3. Small slabs are more likely to be reused (high churn in gradient buffers).

#### 5.1.4 Release (Return to Pool)

```
ALGORITHM: Pool_Release

INPUT: tensor
// Non-contiguous tensors cannot be returned to pool
// (view semantics break bucket reuse)
IF NOT tensor.is_contiguous():
    RETURN  // Let CUDA GC handle it

size_bytes ← tensor.numel() × tensor.element_size()
bucket ← BUCKET_SIZE(size_bytes)

WITH lock:
    IF bucket NOT IN pools:
        pools[bucket] ← []
    pools[bucket].append(tensor.detach().view(-1))
```

### 5.2 Memory Lifecycle During Training

The total VRAM budget during a training step with FSDP and the memory pool is:

$$
M_{\text{total}} = M_{\text{sharded\_state}} + M_{\text{transient\_AG}} + M_{\text{activations}} + M_{\text{pool\_slabs}} + M_{\text{CUDA\_context}} + M_{\text{fragmentation}}
$$

| Component | Size | Lifetime |
|-----------|------|----------|
| Sharded state (params + grads + optimizer) | $16\Phi / W$ | Permanent |
| Transient all-gathered params | $2\Phi_{\text{unit}}$ | Forward/backward of one FSDP unit |
| Activations | $O(BSH \cdot L / \text{AC\_ratio})$ | Forward → backward of that layer |
| Pool slabs (cached) | Variable | Until pressure eviction |
| CUDA context | ~1–2 GB | Permanent |
| Fragmentation | ~5–15% of allocated | Continuous |

---

## 6. Stream Management: Communication-Computation Overlap Architecture

### 6.1 Stream Topology

The `StreamManager` creates four dedicated CUDA streams per rank:

```
ALGORITHM: Initialize_Streams

INPUT: device, is_amd
OUTPUT: StreamManager instance

high_priority ← 0 IF is_amd ELSE -1
// AMD ROCm/HIP: priority 0 is highest schedulable
// NVIDIA CUDA: priority -1 is higher than default (0)

WITH torch.cuda.device(device):
    compute_stream      ← default_stream(device)    // Priority: default
    allgather_stream    ← Stream(device, priority=high_priority)
    reduce_scatter_stream ← Stream(device, priority=high_priority)
    transfer_stream     ← Stream(device, priority=0)   // H2D/D2H
```

| Stream | Purpose | Priority | Overlap Target |
|--------|---------|----------|----------------|
| `compute` | Forward/backward kernels | Default (0) | — |
| `allgather` | Parameter gathering (NCCL all-gather) | High (-1/0) | Compute of previous FSDP unit |
| `reduce_scatter` | Gradient scattering (NCCL reduce-scatter) | High (-1/0) | Compute of next FSDP unit |
| `transfer` | CPU↔GPU data transfers (offload) | Normal (0) | Everything else |

### 6.2 Stream Synchronization Protocol

Inter-stream synchronization uses **CUDA events** (not `synchronize()`) to avoid blocking:

```
ALGORITHM: Sync_Stream_To

INPUT: src_stream, dst_stream

event ← cuda.Event(enable_timing=False, blocking=False)
event.record(src_stream)           // Record completion marker on src
dst_stream.wait_event(event)       // dst waits for src's marker

// Cost: ~1μs (event record + wait) vs. ~100μs (synchronize)
// No CPU-GPU synchronization. No pipeline bubble.
```

Convenience methods:
- `sync_allgather_to_compute()`: After all-gather completes, compute can use the gathered parameters.
- `sync_compute_to_reduce_scatter()`: After backward compute completes, reduce-scatter can begin.

### 6.3 Overlap Timeline

For a single FSDP unit in backward pass:

```
Time →
Compute stream:    |-----backward_compute_k-----|-----backward_compute_{k-1}-----|
AllGather stream:      |--AG_{k-1} (prefetch)----|
ReduceScatter stream:                             |--RS_k (overlap with compute_{k-1})--|
Transfer stream:                                                                   |--D2H grad (if offload)--|
```

The high-priority assignment ensures that communication kernels are scheduled by the GPU scheduler before compute kernels when both are ready, maximizing overlap effectiveness.

### 6.4 AMD-Specific Stream Adaptations

| Aspect | NVIDIA (CUDA) | AMD (ROCm/HIP) |
|--------|--------------|-----------------|
| Highest priority | -1 | 0 |
| Event timing support | Full | Partial (some kernels) |
| GPU timing in MetricsCollector | Enabled | **Disabled** (`enable_gpu_timing=not is_amd`) |
| Stream concurrency | 128+ concurrent kernels | Varies by GCD count on MI300X |

---

## 7. Gradient Accumulation: FSDP-Safe no_sync Protocol

### 7.1 The Problem with Naive Gradient Accumulation Under FSDP

Standard gradient accumulation calls `loss.backward()` for $G$ micro-steps before an optimizer step. Under FSDP, **each `backward()` triggers a reduce-scatter** across all ranks. Without mitigation:

$$
V_{\text{comm}}^{\text{naive}} = G \times \frac{N-1}{N} \times 2\Phi \text{ bytes per cycle}
$$

With `no_sync`:

$$
V_{\text{comm}}^{\text{optimized}} = 1 \times \frac{N-1}{N} \times 2\Phi \text{ bytes per cycle}
$$

The communication reduction is:

$$
\text{Savings} = \frac{G-1}{G} \times 100\%
$$

For $G = 4$ gradient accumulation steps, this is **75% communication reduction**.

### 7.2 The FSDP no_sync Protocol

```
ALGORITHM: FSDP_Gradient_Accumulation

INPUT: loss, accumulation_steps G, current_step counter
OUTPUT: should_sync (bool)

accumulation_counter ← accumulation_counter + 1
scaled_loss ← loss / G

IF accumulation_counter ≥ G:
    // SYNC STEP: Allow FSDP reduce-scatter to execute
    is_sync_step ← True
    scaled_loss ← scale_loss(scaled_loss)  // FP16 loss scaling if needed
    scaled_loss.backward()
    
    RETURN True  // Caller should call step()

ELSE:
    // NON-SYNC STEP: Suppress reduce-scatter
    WITH model.no_sync():
        // Inside no_sync():
        //   - FSDP registers a NOOP for reduce-scatter
        //   - Gradients accumulate in local param.grad buffers
        //   - All-gather still executes (needed for backward compute)
        //   - But reduce-scatter is DEFERRED to sync step
        scaled_loss ← scale_loss(scaled_loss)
        scaled_loss.backward()
    
    RETURN False  // Caller should NOT call step()
```

### 7.3 Communication Pattern with no_sync

| Micro-step | All-Gather (params) | Backward Compute | Reduce-Scatter (grads) | Communication Volume |
|------------|-------------------|-------------------|----------------------|---------------------|
| 1 (non-sync) | ✅ Executes | ✅ Executes | ❌ Suppressed | $\frac{N-1}{N} \cdot 2\Phi$ (AG only) |
| 2 (non-sync) | ✅ Executes | ✅ Executes | ❌ Suppressed | $\frac{N-1}{N} \cdot 2\Phi$ (AG only) |
| 3 (non-sync) | ✅ Executes | ✅ Executes | ❌ Suppressed | $\frac{N-1}{N} \cdot 2\Phi$ (AG only) |
| 4 (sync) | ✅ Executes | ✅ Executes | ✅ Executes | $2 \cdot \frac{N-1}{N} \cdot 2\Phi$ (AG + RS) |

**Total per cycle:** $5 \times \frac{N-1}{N} \cdot 2\Phi$ vs. $8 \times \frac{N-1}{N} \cdot 2\Phi$ without `no_sync`.

### 7.4 Why Manual GradientAccumulator Is Not Used with FSDP

The code includes a `GradientAccumulator` class with named buffer management and optional Triton-fused accumulation. However, for FSDP-wrapped models, **it is intentionally not instantiated**:

```python
# Use native autograd accumulation across micro-steps with
# no_sync(). Manual param.grad injection is unsafe with sharded
# parameters (e.g., local empty shards under FSDP2).
self._gradient_accumulator = None
```

**Root cause:** FSDP shards parameters across ranks. Some ranks may have empty shards for certain parameters (when parameter count is not evenly divisible by world size). The `GradientAccumulator` indexes by parameter name, but the gradient tensor shapes, devices, and even existence differ across ranks. Injecting gradients via `param.grad = buffer` on a shard that expects a different shape causes silent data corruption or NCCL mismatches.

The `no_sync()` context is the **only safe** gradient accumulation mechanism for FSDP because it operates at the FSDP runtime level, deferring the reduce-scatter without modifying gradient tensors directly.

### 7.5 GradientAccumulator Design (For Non-FSDP Use Cases)

The `GradientAccumulator` remains available for non-FSDP training (DDP, single-GPU) and uses Triton-fused accumulation when available:

```
ALGORITHM: Triton_Fused_Gradient_Accumulate

INPUT: grad_ptr, accum_ptr, num_elements, inv_accum_steps
// inv_accum_steps = 1/G (precomputed at init)

KERNEL _fused_gradient_accumulate_kernel:
    pid ← program_id(0)
    offsets ← pid × BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask ← offsets < num_elements
    
    grad ← LOAD(grad_ptr + offsets, mask=mask, other=0.0)
    accum ← LOAD(accum_ptr + offsets, mask=mask, other=0.0)
    
    // Fused multiply-add: accum += grad × (1/G)
    accum ← accum + grad × inv_accum_steps
    
    STORE(accum_ptr + offsets, accum, mask=mask)
```

This fuses the scaling ($\times 1/G$) and accumulation ($+=$) into a single kernel, saving one global memory read-write pass compared to:

```python
buffer.add_(grad, alpha=1.0/G)  # PyTorch: 2 kernels (scale + add)
```

---

## 8. Mixed Precision: Numerical Robustness Across Precision Policies

### 8.1 Precision Policy Matrix

| Policy | `param_dtype` | `reduce_dtype` | Loss Scaling | Use Case |
|--------|-------------|---------------|-------------|----------|
| `FULL_BF16` | BF16 | BF16 | No | Default for A100/H100/B200 |
| `FULL_FP16` | FP16 | FP16 | **Yes** (GradScaler) | Legacy GPUs or specific convergence needs |
| `PARAM_FP32` | FP32 | BF16 | No | Debugging, baseline comparison |
| `PURE_FP32` | FP32 | FP32 | No | Numerical validation, no autocast |

### 8.2 BF16 vs. FP16 Numerical Properties

$$
\text{BF16 range:} \quad \pm 3.39 \times 10^{38}, \quad \text{precision:} \approx 3 \text{ decimal digits} \quad (1 + 7 \text{ mantissa bits})
$$

$$
\text{FP16 range:} \quad \pm 6.55 \times 10^{4}, \quad \text{precision:} \approx 3.3 \text{ decimal digits} \quad (1 + 10 \text{ mantissa bits})
$$

BF16 has the **same exponent range as FP32** (8 bits), making it robust against overflow/underflow without loss scaling. FP16's narrow range ($\pm 65504$) requires dynamic loss scaling to prevent gradient underflow.

### 8.3 Loss Scaling for FP16

```
ALGORITHM: Dynamic_Loss_Scaling

// GradScaler configuration
init_scale ← 2^16 = 65536
growth_factor ← 2.0
backoff_factor ← 0.5
growth_interval ← 2000 steps

// Forward pass:
scaled_loss ← loss × current_scale

// Backward pass:
scaled_loss.backward()
// Gradients are now scaled by current_scale

// Before optimizer step:
unscale_(optimizer)
// Divides gradients by current_scale
// Checks for inf/nan

// Optimizer step (only if no inf/nan):
IF no_overflow:
    optimizer.step()
    consecutive_good ← consecutive_good + 1
    IF consecutive_good ≥ growth_interval:
        current_scale ← current_scale × growth_factor
        consecutive_good ← 0
ELSE:
    // Skip optimizer step (gradients are garbage)
    current_scale ← current_scale × backoff_factor
    gradient_overflow_count ← gradient_overflow_count + 1
```

### 8.4 FSDP Mixed Precision Integration

PyTorch FSDP's `MixedPrecision` policy controls three dtype axes:

| Axis | Setting | Effect on FSDP Communication |
|------|---------|------------------------------|
| `param_dtype` | BF16 | All-gather transmits 2 bytes/param (vs. 4 for FP32) |
| `reduce_dtype` | BF16 | Reduce-scatter transmits 2 bytes/grad (vs. 4 for FP32) |
| `buffer_dtype` | BF16 | Non-parameter buffers (BatchNorm stats, etc.) cast to BF16 |

Communication volume per training step with BF16:

$$
V_{\text{BF16}} = 3 \times \frac{N-1}{N} \times 2\Phi \approx 6\Phi \text{ bytes}
$$

With FP32:

$$
V_{\text{FP32}} = 3 \times \frac{N-1}{N} \times 4\Phi \approx 12\Phi \text{ bytes}
$$

BF16 halves communication volume, directly improving scaling efficiency.

### 8.5 Autocast Context Integration

```
ALGORITHM: Forward_Context

CONTEXT MANAGER: forward_context(self)

// Combines autocast + non-blocking forward timing
WITH metrics.measure_forward():                // CUDA event timing
    WITH mp_context.autocast():                 // torch.amp.autocast
        // Inside autocast:
        //   - Linear layers compute in param_dtype (BF16)
        //   - Softmax, LayerNorm, loss in FP32 (auto-promoted)
        //   - Output dtype matches param_dtype
        YIELD
```

---

## 9. Hardware Detection and Cross-Vendor Portability

### 9.1 Hardware Capability Enumeration

The `detect_hardware()` function probes CUDA device properties and infers vendor-specific capabilities:

```
ALGORITHM: Detect_Hardware

INPUT: device_id
OUTPUT: Result[HardwareInfo]

IF NOT cuda.is_available():
    RETURN Err("CUDA runtime not available")

props ← cuda.get_device_properties(device_id)

// Vendor detection via device name pattern matching
vendor ← MATCH props.name AGAINST:
    "NVIDIA", "A100", "H100", "RTX" → NVIDIA
    "AMD", "MI", "INSTINCT"          → AMD
    "INTEL"                           → INTEL
    DEFAULT                           → UNKNOWN

// Compute capability extraction
cc ← ComputeCapability(props.major, props.minor)
// SM version = major × 10 + minor
// SM 80 = A100, SM 89 = L40S/4090, SM 90 = H100/H200, SM 100 = B100/B200

// NVLink detection
has_nvlink ← ANY of ("A100", "H100", "H200", "V100", "DGX") IN props.name

RETURN Ok(HardwareInfo(
    vendor, props.name, cc, props.total_memory,
    props.multi_processor_count,
    props.max_threads_per_multi_processor,
    l2_cache_size, has_nvlink, pcie_bandwidth
))
```

### 9.2 Compute Capability Feature Matrix

| SM Version | GPU | BF16 | FP8 | TMA | Notes |
|-----------|-----|------|-----|-----|-------|
| SM 70 | V100 | ❌ | ❌ | ❌ | FP16 only |
| SM 80 | A100 | ✅ | ❌ | ❌ | BF16 compute, no FP8 |
| SM 89 | L40S, RTX 4090 | ✅ | ✅ | ❌ | FP8 E4M3/E5M2 |
| SM 90 | H100, H200 | ✅ | ✅ | ✅ | TMA, thread block clusters |
| SM 100 | B100, B200 | ✅ | ✅ | ✅ | MXFP8, MXFP6, MXFP4 |

```
ALGORITHM: Feature_Detection

sm_version ← major × 10 + minor

supports_bf16 ← (sm_version ≥ 80)
supports_fp8  ← (sm_version ≥ 89)
supports_tma  ← (sm_version ≥ 90)

// AMD MI300X: gfx942, supports BF16 and FP8
// AMD MI350:  gfx950, supports BF16, FP8, MXFP formats
// Detected via vendor flag, not SM version
```

### 9.3 Vendor-Adaptive Behavior

| Behavior | NVIDIA | AMD (MI300X/MI350) |
|----------|--------|-------------------|
| Stream priority for comm | -1 (high) | 0 (default=highest on HIP) |
| GPU event timing | Enabled | **Disabled** (accuracy issues on ROCm) |
| Triton kernel availability | Full support | Triton-HIP (partial, check at import) |
| CUDA Graphs | Supported | HIP Graphs (partial) |
| NCCL/RCCL | NCCL 2.18+ | RCCL (ROCm 6.x) |
| PCIe bandwidth estimate | 32 GB/s (Gen5, SM≥80) | 32 GB/s (assumed) |

### 9.4 Datacenter GPU Detection

```
ALGORITHM: Is_Datacenter_GPU

patterns ← ("A100", "H100", "H200", "B100", "B200", "MI300", "MI250")
is_datacenter ← ANY(pattern IN device_name FOR pattern IN patterns)

// Used for:
// - Enabling NVLink-optimized collectives
// - Setting aggressive bucket sizes (larger for datacenter interconnects)
// - Enabling advanced features (CPU offload, CUDA graphs)
```

---

## 10. Triton Kernel Integration: Fused Collective Primitives

### 10.1 Kernel Inventory

The system includes four Triton JIT-compiled kernels, each eliminating redundant memory passes:

#### 10.1.1 Fused All-Gather + Scale

```
ALGORITHM: Fused_AllGather_Scale_Kernel

INPUT: src_ptr (local shard), dst_ptr (gathered buffer), scale, num_elements, rank_offset
// Combines the data copy into the all-gather staging buffer
// with a scaling operation (e.g., for mixed-precision cast)

PARALLEL FOR pid IN range(num_blocks):
    offsets ← pid × BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask ← offsets < num_elements
    
    data ← LOAD(src_ptr + offsets, mask)
    data ← data × scale                    // Fused scale
    STORE(dst_ptr + rank_offset + offsets, data, mask)

// Saves: 1 kernel launch + 1 global memory pass vs. separate scale + copy
```

#### 10.1.2 Fused Cast and Scale

```
ALGORITHM: Fused_Cast_Scale_Kernel

INPUT: src_ptr (input dtype), dst_ptr (output dtype), scale, num_elements
// Combines dtype cast with scaling in a single pass
// Used for: BF16→FP32 gradient upcasting with 1/G scaling

PARALLEL FOR pid IN range(num_blocks):
    offsets ← pid × BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask ← offsets < num_elements
    
    data ← LOAD(src_ptr + offsets, mask)    // Load in source dtype
    data ← data × scale                     // Scale (in compute dtype)
    STORE(dst_ptr + offsets, data, mask)     // Store in target dtype

// Triton handles dtype promotion automatically via tl.load/tl.store
```

#### 10.1.3 Fused Gradient Accumulation

```
ALGORITHM: Fused_Gradient_Accumulate_Kernel

INPUT: grad_ptr, accum_ptr, num_elements, inv_accum_steps
// accum += grad × (1/G) in a single kernel

PARALLEL FOR pid IN range(num_blocks):
    offsets ← pid × BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask ← offsets < num_elements
    
    grad ← LOAD(grad_ptr + offsets, mask, other=0.0)
    accum ← LOAD(accum_ptr + offsets, mask, other=0.0)
    accum ← accum + grad × inv_accum_steps
    STORE(accum_ptr + offsets, accum, mask)

// Saves: 2 kernel launches + 1 memory pass vs. PyTorch buffer.add_(grad, alpha=1/G)
// PyTorch decomposes to: mul kernel + add kernel (2 passes over accum)
```

#### 10.1.4 Fused Parameter Sharding

```
ALGORITHM: Fused_Param_Shard_Kernel

INPUT: full_param_ptr, shard_ptr, full_numel, shard_size, rank_offset
// Extracts rank-local shard from all-gathered full parameter
// Replaces: shard = full_param[rank_offset : rank_offset + shard_size]

PARALLEL FOR pid IN range(num_blocks):
    offsets ← pid × BLOCK_SIZE + arange(0, BLOCK_SIZE)
    mask ← offsets < shard_size
    src_offsets ← rank_offset + offsets
    full_mask ← mask AND (src_offsets < full_numel)
    
    data ← LOAD(full_param_ptr + src_offsets, mask=full_mask, other=0.0)
    STORE(shard_ptr + offsets, data, mask)

// Autotuned across BLOCK_SIZE ∈ {1024, 2048, 4096, 8192}
// Key: shard_size (autotuner selects best config per shard size)
```

### 10.2 Autotune Configuration

The parameter sharding kernel uses Triton's `@autotune` decorator:

| Config | BLOCK_SIZE | num_warps | Best For |
|--------|-----------|-----------|----------|
| 1 | 1024 | 4 | Small shards (< 1M elements) |
| 2 | 2048 | 8 | Medium shards (1M–10M elements) |
| 3 | 4096 | 8 | Large shards (10M–100M elements) |
| 4 | 8192 | 16 | Very large shards (> 100M elements) |

The autotune key is `shard_size`, ensuring the best configuration is cached per unique shard size encountered during training.

### 10.3 Fallback Path

When Triton is unavailable, all operations fall back to PyTorch-native operators:

```
IF NOT TRITON_AVAILABLE:
    // Gradient accumulation fallback:
    buffer.add_(grad, alpha=inv_accum_steps)
    // ~15% slower than fused Triton (two kernel launches + extra memory pass)
```

---

## 11. Metrics Collection: Zero-Stall GPU Instrumentation

### 11.1 The Synchronization Problem

Naive GPU timing requires `torch.cuda.synchronize()`, which:
- Blocks the CPU until all GPU work completes.
- Creates a pipeline bubble between CPU-side scheduling and GPU-side execution.
- Costs 50–200μs per synchronization on H100 (PCIe latency + drain).

At scale, dozens of timing calls per step create **milliseconds of stall** that directly reduce MFU.

### 11.2 Non-Blocking Event-Based Timing

The `MetricsCollector` uses CUDA events with **deferred resolution**:

```
ALGORITHM: NonBlocking_GPU_Timing

// Recording phase (in hot path — zero stall):
CONTEXT MANAGER measure_gpu(field_name):
    start_event ← cuda.Event(enable_timing=True)
    end_event ← cuda.Event(enable_timing=True)
    
    start_event.record()              // Record on current stream (~100ns)
    YIELD                             // Execute timed operation
    end_event.record()                // Record completion (~100ns)
    
    // DO NOT CALL start.elapsed_time(end) HERE
    // That would require synchronization
    
    WITH lock:
        pending_events.append((start_event, end_event, field_name))

// Resolution phase (outside hot path — called at record_step):
FUNCTION _flush_pending_events():
    remaining ← []
    FOR (start, end, field) IN pending_events:
        IF end.query():               // Non-blocking check: is event done?
            elapsed_ms ← start.elapsed_time(end)  // Safe: event completed
            elapsed_ns ← int(elapsed_ms × 1e6)
            current[field] ← current[field] + elapsed_ns
        ELSE:
            remaining.append((start, end, field))  // Not yet done, retry later
    pending_events ← remaining
```

**Key insight:** `event.query()` returns `True/False` without blocking. `elapsed_time()` is only called after `query()` confirms completion. This **never** blocks the CPU.

### 11.3 Metrics Schema

```
DATACLASS FSDPMetrics:
    // Communication timing (nanoseconds)
    allgather_time_ns: int
    reduce_scatter_time_ns: int
    
    // Computation timing (nanoseconds)
    forward_time_ns: int
    backward_time_ns: int
    optimizer_time_ns: int
    
    // Memory (bytes)
    peak_memory_bytes: int
    allocated_memory_bytes: int
    reserved_memory_bytes: int
    
    // Throughput
    tokens_processed: int
    samples_processed: int
    
    // Numerical health
    gradient_norm: float
    gradient_overflow_count: int
```

### 11.4 Rolling Average Computation

```
ALGORITHM: Compute_Rolling_Average

INPUT: last_n steps (default 100)
OUTPUT: Dict[str, float] averaged metrics

WITH lock:
    _flush_pending_events()           // Resolve any completed events
    history ← self._history[-last_n:]

IF history IS EMPTY:
    RETURN {}

n ← len(history)
RETURN {
    "avg_allgather_ms":       SUM(h.allgather_time_ns) / n / 1e6,
    "avg_reduce_scatter_ms":  SUM(h.reduce_scatter_time_ns) / n / 1e6,
    "avg_forward_ms":         SUM(h.forward_time_ns) / n / 1e6,
    "avg_backward_ms":        SUM(h.backward_time_ns) / n / 1e6,
    "avg_optimizer_ms":       SUM(h.optimizer_time_ns) / n / 1e6,
    "max_peak_memory_gb":     MAX(h.peak_memory_bytes) / 2^30,
}
```

### 11.5 AMD Timing Limitation

On AMD GPUs, CUDA event timing is **disabled** due to accuracy issues in ROCm's HIP event implementation:

```
self._metrics = MetricsCollector(enable_gpu_timing=not self._is_amd)
```

When disabled, all timing context managers become no-ops, and the metrics contain zeros for timing fields. Profiling on AMD must use `rocprof` or `roctracer` externally.

---

## 12. Activation Checkpointing: Selective Recomputation Strategies

### 12.1 Checkpointing Modes

| Mode | Behavior | Memory Savings | Recomputation Cost |
|------|----------|----------------|-------------------|
| `full` | Checkpoint every eligible layer | Maximum (~60% activation reduction) | ~33% compute overhead |
| `selective` | Checkpoint every $k$-th layer (default $k=2$) | Moderate (~30% activation reduction) | ~15% compute overhead |
| `offload` | Checkpoint + offload activations to CPU | Maximum + CPU memory | D2H/H2D transfer overhead |

### 12.2 Selective Checkpointing Decision

```
ALGORITHM: Selective_AC_Check

INPUT: module, layer_counter, ac_frequency k, ac_mode
OUTPUT: should_checkpoint (bool)

// Only checkpoint layer-like modules
name ← module.__class__.__name__.lower()
is_layer ← ANY("layer", "block", "decoder", "encoder") IN name

IF NOT is_layer:
    RETURN False

IF ac_mode == "full":
    RETURN True

IF ac_mode == "selective":
    idx ← layer_counter
    layer_counter ← layer_counter + 1
    RETURN (idx MOD k == 0)  // Every k-th layer

RETURN True  // Default: checkpoint
```

For a 32-layer model with `ac_frequency=2`:
- Layers 0, 2, 4, ..., 30 are checkpointed (16 layers).
- Layers 1, 3, 5, ..., 31 retain activations in memory.
- Activation memory is approximately halved compared to no checkpointing.

### 12.3 Ordering Constraint: AC Before FSDP

```
ALGORITHM: Model_Wrapping_Order

// CRITICAL: Activation checkpointing MUST precede FSDP wrapping

Step 1: apply_activation_checkpointing(model)
    // Wraps selected layers with checkpoint_wrapper
    // Uses NO_REENTRANT implementation (safe with FSDP)

Step 2: FSDP(model, ...)
    // FSDP wraps the already-checkpointed model
    // FSDP's own weight freeing (selective AC on parametrization)
    // composes with user's activation checkpointing
```

**Why this order:** FSDP's `auto_wrap_policy` partitions the model into FSDP units based on module structure. If FSDP wraps first, the checkpoint wrapper cannot see the original module boundaries. If AC wraps first, FSDP sees the `CheckpointWrapper` modules and wraps them as atomic units.

### 12.4 Memory Impact

For a Transformer layer with hidden dimension $H$, sequence length $S$, batch size $B$:

**Without checkpointing (per layer):**

$$
M_{\text{act}}^{\text{layer}} \approx 2BSH \cdot (1 + \frac{4n_{\text{heads}} \cdot S}{H} + 8) \approx 34BSH \text{ bytes (BF16)}
$$

**With full checkpointing (per layer):**

$$
M_{\text{act}}^{\text{ckpt}} \approx 2BSH \text{ bytes (only input activation saved)}
$$

**Memory reduction per layer:**

$$
\Delta M = 34BSH - 2BSH = 32BSH \text{ bytes}
$$

For a 70B model ($H = 8192$, $S = 4096$, $B = 1$, $L = 80$):

$$
\Delta M_{\text{total}} = 80 \times 32 \times 1 \times 4096 \times 8192 \approx 80 \text{ GB}
$$

With selective checkpointing ($k=2$, 40 layers checkpointed):

$$
\Delta M_{\text{selective}} \approx 40 \text{ GB saved}
$$

---

## 13. Auto-Wrap Policy: Transformer-Aware Layer Discovery

### 13.1 Layer Class Discovery

The auto-wrap policy searches for known Transformer layer classes across common model libraries:

```
ALGORITHM: Auto_Wrap_Policy_Resolution

INPUT: auto_wrap_policy (list of class names)
OUTPUT: PyTorch auto_wrap_policy callable

// Phase 1: Resolve class names to Python classes
layer_classes ← SET()
FOR cls_name IN auto_wrap_policy:
    cls ← _try_import_class(cls_name)
    IF cls IS NOT None:
        layer_classes.add(cls)

// Phase 2: Select wrapping strategy
IF layer_classes IS NOT EMPTY:
    // Transformer-specific wrapping: one FSDP unit per layer class
    RETURN partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes
    )
ELSE:
    // Fallback: size-based wrapping (one unit per min_num_params)
    RETURN partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000  // 100M params per FSDP unit
    )
```

### 13.2 Supported Layer Classes

| Class Name | Model Family | Import Path |
|------------|-------------|-------------|
| `LlamaDecoderLayer` | Llama 2/3 | `transformers.models.llama.modeling_llama` |
| `MistralDecoderLayer` | Mistral/Mixtral | `transformers.models.mistral.modeling_mistral` |
| `Qwen2DecoderLayer` | Qwen 2 | `transformers.models.qwen2.modeling_qwen2` |
| `GPT2Block` | GPT-2 | `transformers.models.gpt2.modeling_gpt2` |
| `FalconDecoderLayer` | Falcon | `transformers.models.falcon.modeling_falcon` |
| `GemmaDecoderLayer` | Gemma | `transformers.models.gemma.modeling_gemma` |
| `Phi3DecoderLayer` | Phi-3 | `transformers.models.phi3.modeling_phi3` |
| `TransformerEncoderLayer` | Generic | `torch.nn` |
| `TransformerDecoderLayer` | Generic | `torch.nn` |
| `TransformerSentenceEncoderLayer` | Fairseq | `fairseq.modules` |

### 13.3 Why Transformer-Aware Wrapping Matters

**Per-layer wrapping** (one FSDP unit per Transformer layer) is optimal because:

1. **Uniform memory footprint:** Each layer has approximately equal parameter count ($\approx 12H^2$), producing balanced shard sizes.
2. **Natural pipeline boundary:** Forward/backward execution is sequential by layer—wrapping per layer enables maximum overlap between adjacent units' communication and computation.
3. **Activation checkpointing alignment:** Checkpointing is naturally per-layer; matching FSDP boundaries to AC boundaries avoids cross-boundary recomputation.

**Size-based wrapping** (fallback) is suboptimal because it may split a single attention block across two FSDP units, creating unnecessary all-gather/reduce-scatter boundaries within a logically atomic computation.

---

## 14. Failure Mode Taxonomy and Root-Cause Resolution Chain

### 14.1 Complete Fix Registry

| Fix ID | Category | Symptom | Root Cause | Resolution | Version |
|--------|----------|---------|------------|------------|---------|
| FIX-001 | Memory | OOM during training | Unbounded allocation without pooling | `MemoryPool` with pressure-aware eviction | v2.0 |
| FIX-002 | Compute | Low GPU utilization, CPU stalls | `torch.cuda.synchronize()` in metrics | Non-blocking CUDA event timing | v2.0 |
| FIX-003 | Communication | $G\times$ excess reduce-scatter during accumulation | Missing `no_sync()` for non-sync micro-steps | `_no_sync_context()` wrapping non-sync backward | v2.0 |
| FIX-004 | Portability | Crashes on MI300X | CUDA-specific stream priorities, timing | AMD detection → adaptive priorities, disabled event timing | v2.0 |
| FIX-005 | Device Affinity | 30-minute hang → NCCL crash during checkpoint | `offload_to_cpu` creates staging tensors on `cuda:0` | `_enforce_device_affinity()` context, re-pinning after every API call | v2.1 |
| FIX-006 | API Deprecation | Hundreds of `FutureWarning` per save | Legacy `FSDP.state_dict_type()` uses `ShardedTensor` | Modern DCP API (`get_state_dict`/`set_state_dict`) with DTensor | v2.1 |
| FIX-007 | Performance | 50–100s checkpoint stall at 78% VRAM | Two separate all-gather scopes + no memory cleanup between | Single `get_state_dict` call, proactive GC, reduced NCCL timeout | v2.1 |
| FIX-008 | UX | Warning spam in logs | Four categories of known-harmless deprecation warnings | Targeted `warnings.filterwarnings()` at module import, fail-open | v2.1 |
| FIX-009 | Correctness | Overwrite warning noise | `FileSystemWriter` default `overwrite=False` | Explicit `overwrite=True` in constructor | v2.1 |

### 14.2 Failure Chain Visualization

```
USER ACTION: save_checkpoint() at 78% VRAM

    ┌──────────────────────────────────┐
    │ FIX-007: Pre-save memory cleanup │
    │ gc.collect() + empty_cache()     │
    │ Release pool slabs (FIX-001)     │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-005: Device affinity guard   │
    │ torch.cuda.set_device(rank_dev)  │
    │ torch.cuda.device(rank_dev)      │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-006: API selection           │
    │ IF modern DCP → single scope     │
    │ ELSE legacy → dual scope + GC    │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-008: Warning suppression     │
    │ (legacy path only)               │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-009: FileSystemWriter        │
    │ overwrite=True                   │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-005: Post-save verification  │
    │ O(P) device scan + migration     │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │ FIX-007: Post-save barrier       │
    │ 5-minute timeout (vs. 30-min)    │
    └──────────────────────────────────┘
```

### 14.3 Diagnostic Decision Tree

```
ALGORITHM: Diagnose_FSDP_Failure

SYMPTOM: Training hangs with no error message

    Q1: Is the hang during checkpoint save?
    ├── YES → Check FIX-005: device affinity
    │         Run: torch.cuda.current_device() on each rank
    │         If any rank shows cuda:0 AND rank ≠ 0 → DEVICE DRIFT
    │         Fix: _enforce_device_affinity() context
    │
    └── NO → Q2: Is the hang during backward?
            ├── YES → Check FIX-003: no_sync missing
            │         Monitor: NCCL traffic during non-sync micro-steps
            │         If reduce-scatter observed → no_sync not active
            │
            └── NO → Q3: Is the hang at barrier?
                    ├── YES → Check FIX-007: NCCL timeout
                    │         One rank may have failed silently
                    │         Set NCCL_TIMEOUT=300 for faster diagnosis
                    │
                    └── NO → Check NCCL logs for topology mismatch

SYMPTOM: OOM during checkpoint save

    Check FIX-007: Pre-save memory cleanup
    Verify: VRAM utilization before save (target < 85%)
    If > 85%: increase GC aggressiveness or reduce model size
    If using legacy API: ensure GC between model + optimizer scopes

SYMPTOM: Hundreds of FutureWarning in logs

    Check FIX-008: Warning suppression installed?
    Verify: _suppress_known_deprecation_warnings() called at import
    If yes but warnings persist: new PyTorch version with new warnings
    → Update suppression patterns (fail-open means new warnings visible)
```

---

## 15. Production Deployment: End-to-End Integration Contract

### 15.1 Complete Training Step Flow

```
ALGORITHM: Production_Training_Step

// ==================== INITIALIZATION ====================
config ← FSDP2Config(
    sharding_strategy=FULL_SHARD,
    mixed_precision=FULL_BF16,
    activation_checkpointing=True,
    ac_mode="selective",
    ac_frequency=2,
    gradient_accumulation_steps=4,
    gradient_clipping_norm=1.0,
    use_triton_kernels=True,
    use_memory_pool=True,
    bucket_size_mb=25,
    forward_prefetch=True,
    backward_prefetch=BACKWARD_PRE,
    limit_all_gathers=True
)

fsdp ← SOTAFSDP2(config)
model ← create_model(model_config)
wrapped_model ← fsdp.wrap_model(model)
optimizer ← AdamW(wrapped_model.parameters(), lr=lr)

// ==================== TRAINING LOOP ====================
FOR step = 1 TO max_steps:
    FOR micro_step = 1 TO gradient_accumulation_steps:
        batch ← dataloader.next()
        
        // Forward pass [INT-007]
        WITH fsdp.forward_context():
            loss ← wrapped_model(batch.input_ids, batch.labels)
        
        // Backward pass [INT-003, INT-010]
        should_sync ← fsdp.backward(loss)
        // micro_steps 1,2,3: should_sync=False (no_sync active)
        // micro_step 4: should_sync=True (reduce-scatter executes)
    
    // Optimizer step [INT-004] — only after sync step
    IF should_sync:
        fsdp.step(optimizer, scheduler)
        // Internally:
        //   1. unscale_grads(optimizer)     — FP16 only
        //   2. clip_grad_norm_(1.0)         — distributed FSDP clip
        //   3. optimizer.step()             — AdamW update
        //   4. scheduler.step()             — LR schedule
        //   5. optimizer.zero_grad(set_to_none=True)
        //   6. accumulation_counter ← 0
        //   7. metrics.record_step()
        //   8. IF VRAM > 90%: gc + empty_cache
    
    // Checkpoint [INT-006]
    IF step MOD checkpoint_interval == 0:
        result ← FSDPCheckpointManager.save_checkpoint(
            fsdp, optimizer, checkpoint_dir,
            epoch=epoch, step=step, sharded=True
        )
        IF result.is_err():
            LOG_ERROR(result.error)
    
    // Logging [INT-009]
    IF step MOD log_interval == 0:
        LOG(fsdp.memory_summary())
        LOG(fsdp.metrics.get_average(last_n=100))
```

### 15.2 Cluster Configuration Matrix

| Cluster | GPU | HBM | Interconnect | Recommended Config |
|---------|-----|-----|-------------|-------------------|
| 8× A100 80GB | A100 | 80 GB | NVLink 600 GB/s | FULL_SHARD, BF16, AC selective, bucket 25MB |
| 8× H100 80GB | H100 | 80 GB | NVSwitch 900 GB/s | FULL_SHARD, BF16, AC selective, bucket 25MB, forward_prefetch |
| 8× H200 141GB | H200 | 141 GB | NVSwitch 900 GB/s | FULL_SHARD, BF16, bucket 50MB, larger batches |
| 8× B200 192GB | B200 | 192 GB | NVLink5 1.8 TB/s | FULL_SHARD, BF16, bucket 100MB, limit_all_gathers=False |
| 8× MI300X 192GB | MI300X | 192 GB | xGMI 896 GB/s | FULL_SHARD, BF16, Triton off (HIP), event timing off |
| 4× MI350 288GB | MI350 | 288 GB | xGMI | FULL_SHARD, BF16, larger FSDP units |

### 15.3 Per-Rank Memory Budget Verification

Before launching training, verify the model fits within the per-rank HBM budget:

$$
M_{\text{required}}^{\text{rank}} = \underbrace{\frac{16\Phi}{W}}_{\text{sharded state}} + \underbrace{2\Phi_{\text{unit}}}_{\text{transient AG}} + \underbrace{L \cdot M_{\text{act}}^{\text{layer}} / \text{AC\_factor}}_{\text{activations}} + \underbrace{2 \text{ GB}}_{\text{CUDA context + buffers}}
$$

$$
M_{\text{required}}^{\text{rank}} \leq 0.90 \times M_{\text{HBM}} \quad \text{(VRAM\_PRESSURE\_WATERMARK)}
$$

For a 70B model on 64× H100 80GB ($W=64$, $H=8192$, $L=80$, $S=4096$, $B=1$):

$$
M_{\text{sharded}} = \frac{16 \times 70 \times 10^9}{64} = 17.5 \text{ GB}
$$

$$
M_{\text{transient}} = 2 \times \frac{12 \times 8192^2}{64} \approx 25 \text{ MB (per FSDP unit, 1 layer)}
$$

$$
M_{\text{act}} = 80 \times 2 \times 1 \times 4096 \times 8192 / 2 \approx 2.5 \text{ GB (selective AC, k=2)}
$$

$$
M_{\text{total}} \approx 17.5 + 0.025 + 2.5 + 2.0 = 22.0 \text{ GB} \ll 72 \text{ GB (90\% of 80GB)}
$$

**Verdict:** Fits with significant headroom. Can increase batch size or sequence length.

### 15.4 Result Type Usage Pattern

The `Result[T]` type eliminates exception-based control flow for recoverable errors:

```
ALGORITHM: Result_Type_Pattern

// Checkpoint save with graceful degradation
result ← FSDPCheckpointManager.save_checkpoint(fsdp, optimizer, path)

IF result.is_ok():
    // Checkpoint saved successfully
    LOG("Checkpoint saved")
    
ELSE:  // result.is_err()
    // Checkpoint failed, but training continues
    LOG_ERROR(f"Checkpoint failed: {result.error} (code={result.code})")
    // Training is NOT interrupted
    // Next checkpoint will be attempted at the next interval
    // This is INTENTIONAL: checkpoint failure should not crash training

// Map pattern for chaining
result = detect_hardware(device_id).map(lambda hw: hw.memory_gb)
// Ok(HardwareInfo) → Ok(80.0)
// Err("not available") → Err("not available")
```

---

## 16. Summary: Key Engineering Decisions and Their Rationale

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Device affinity guard on every checkpoint op | Prevents 30-minute NCCL hangs from cuda:0 drift | Eliminates the #1 production failure mode |
| Modern DCP API with legacy fallback | Zero deprecation warnings, 2× faster saves, correct device tracking | Forward-compatible, backward-compatible |
| Single `get_state_dict` call for model+optimizer | One all-gather instead of two; prevents OOM at 78% VRAM | 50–100s → ~15s checkpoint time |
| `no_sync()` for non-sync micro-steps | Reduces communication by $(G-1)/G$ during gradient accumulation | 75% communication reduction for $G=4$ |
| Non-blocking CUDA event timing | Eliminates CPU-GPU synchronization in hot path | ~0.5–2% MFU improvement |
| Power-of-2 memory pool with pressure eviction | Eliminates cudaMalloc jitter; prevents OOM via proactive eviction | Stable step times, no OOM surprises |
| AMD-specific stream priority and timing | ROCm HIP has different priority semantics and event accuracy | Training runs correctly on MI300X/MI350 |
| Atomic checkpoint writes (tmp + rename) | Crash during save never corrupts existing checkpoint | Zero checkpoint corruption in production |
| Triton-fused gradient accumulation | Single kernel for scale+accumulate vs. two PyTorch kernels | ~15% gradient accumulation speedup |
| Targeted warning suppression (fail-open) | Clean logs without hiding novel warnings | Developer experience + debuggability |
| Result type for error handling | Checkpoint failures don't crash training; errors are explicit | Production resilience |
| AC before FSDP wrapping order | Ensures FSDP sees checkpoint-wrapped modules as atomic units | Correct composition of memory optimizations |

---

*This report documents the SOTAFSDP2 v2.1 production system for FSDP-based distributed training, covering the full failure-mode taxonomy, checkpoint safety protocol, memory management strategy, and cross-vendor portability layer for deployment on A100, H100, H200, B100, B200, MI300X, MI325X, and MI350-class accelerator clusters at scales from single-node to multi-thousand GPU.*