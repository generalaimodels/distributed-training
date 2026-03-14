# Technical Report: Megatron-Core + DeepSpeed + FSDP Interoperability Across Training, Resume, Resharding, and Conversion

## 1. Executive Summary

This report defines the correct interoperability model for combining **Megatron-Core**, **DeepSpeed**, and **PyTorch FSDP** in production-scale LLM training systems. The goal is not merely to make these systems coexist; the goal is to make them **coexist without ambiguous ownership, silent state corruption, optimizer mismatch, checkpoint incompatibility, or distributed nondeterminism**.

The central interoperability rule is:

> **Megatron-Core should own model-parallel decomposition; DeepSpeed or FSDP should own data-parallel state sharding; checkpoint conversion must operate on canonical logical tensors rather than framework-local shards.**

The correct design therefore hinges on four invariants:

1. **Single ownership of each state dimension**
   - one owner for parameter sharding,
   - one owner for optimizer-state sharding,
   - one owner for model-parallel tensor partitioning.

2. **Canonical logical checkpointing**
   - portable across:
     - Megatron-Core native training,
     - DeepSpeed ZeRO-$1/2/3$,
     - FSDP full-shard,
     - changed $DP$ degree,
     - changed framework stack,
     - mixed-vendor restarts.

3. **Deterministic mesh construction**
   - all ranks must derive identical:
     - $TP$ groups,
     - $PP$ groups,
     - $CP$ groups,
     - $EP$ groups,
     - $DP$ / FSDP / ZeRO groups,
     - RNG streams,
     - virtual pipeline chunk mapping.

4. **Conversion through logical tensor IR**
   - never by copying rank-local files directly between incompatible layouts.

---

## 2. Scope

This report covers, end to end:

- runtime ownership and composition semantics,
- supported and unsupported combinations,
- parameter, gradient, optimizer-state, and expert-state handling,
- process-group factorization,
- train-time initialization order,
- checkpoint schema design,
- same-layout resume,
- changed-world-size resume,
- DP-only resharding,
- TP/PP/EP conversion,
- ZeRO-stage migration,
- FSDP conversion,
- mixed-precision portability,
- FP8 metadata handling,
- activation checkpointing interactions,
- long-context and MoE interoperability constraints,
- failure modes and validation gates,
- automation strategy for train, resume, and conversion workflows.

This report is intentionally focused on **interoperability**. Hardware-specific deployment tuning is not the primary topic here except where it affects correctness and format design.

---

## 3. Architectural Roles of the Three Systems

## 3.1 Megatron-Core

Megatron-Core is best understood as the **model-parallel runtime owner**. It is the correct abstraction when the training system needs:

- tensor parallelism $TP$,
- pipeline parallelism $PP$,
- virtual pipeline interleaving,
- sequence parallelism $SP$,
- context parallelism $CP$,
- expert parallelism $EP$,
- expert tensor parallelism $ETP$,
- fused Transformer kernels,
- large-scale schedule control.

Megatron-Core should therefore own:

- layer partitioning across pipeline stages,
- tensor-axis partitioning for dense layers,
- expert placement and dispatch topology,
- sequence-parallel activation partitioning,
- context-parallel attention partitioning,
- interleaved schedule semantics.

---

## 3.2 DeepSpeed

DeepSpeed is best understood as a **data-parallel state-sharding and optimizer runtime**. In interoperability contexts it most commonly owns:

- ZeRO-$1$ optimizer-state sharding,
- ZeRO-$2$ gradient + optimizer-state sharding,
- ZeRO-$3$ parameter + gradient + optimizer-state sharding,
- offload policy,
- optimizer stepping logic,
- certain communication overlap policies.

DeepSpeed is strongest when the training stack already has an established model-parallel runtime and needs:

- memory reduction on the data-parallel axis,
- optimizer offload,
- ZeRO checkpointing,
- operational integration in existing training estates.

---

## 3.3 PyTorch FSDP

FSDP is also a **data-parallel state-sharding owner**, but with different runtime semantics and integration properties from DeepSpeed ZeRO-$3$.

FSDP should own:

- parameter sharding,
- gradient sharding,
- optimizer-state sharding,
- all-gather of parameters before local compute,
- reduce-scatter of gradients after backward,
- optional parameter or gradient offload,
- sharded or canonical checkpoint exports.

FSDP is often preferable when priorities are:

- PyTorch-native portability,
- cleaner integration with DTensor or compiler-visible graphs,
- canonical state-dict workflows,
- mixed-vendor portability with less framework-specific control logic.

---

## 3.4 Interoperability Principle

> **Megatron-Core owns model decomposition; DeepSpeed or FSDP owns data-parallel state sharding; they must not both shard the same parameter set over the same logical axis.**

This is the core rule that eliminates most interoperability failures.

---

# 4. Parallel Mesh and Group Construction

## 4.1 World-Size Factorization

For dense models:

$$
W = D \times T \times P \times C
$$

Where:

- $W$ = total world size,
- $D$ = data-parallel or shard-replica multiplicity,
- $T$ = tensor-parallel degree,
- $P$ = pipeline stage count,
- $C$ = context-parallel degree.

For MoE models:

$$
W = D \times T_d \times P \times C \times E \times T_e
$$

Where:

- $T_d$ = dense tensor-parallel degree,
- $E$ = expert-parallel degree,
- $T_e$ = expert tensor-parallel degree.

### Important note

Sequence parallelism is generally a runtime property attached to $TP$ and is not treated as an independent multiplicative mesh axis.

---

## 4.2 Group Ownership

The interoperable runtime must construct the following groups deterministically:

| Group type | Fixed coordinates | Varying coordinate | Owner |
|---|---|---|---|
| $TP$ group | $d,p,c,e,t_e$ | $t$ | Megatron-Core |
| $PP$ group | $d,t,c,e,t_e$ | $p$ | Megatron-Core |
| $CP$ group | $d,t,p,e,t_e$ | $c$ | Megatron-Core |
| $EP$ group | $d,t,p,c,t_e$ | $e$ | Megatron-Core |
| $ETP$ group | $d,t,p,c,e$ | $t_e$ | Megatron-Core |
| $DP$/ZeRO/FSDP group | $t,p,c,e,t_e$ | $d$ | DeepSpeed or FSDP |

For a dense model without MoE, the FSDP or ZeRO group for fixed model-parallel coordinates is:

$$
G_{\text{shard}}(t,p,c) = \{(d,t,p,c)\mid d \in [0, D-1]\}
$$

For a pipeline-parallel model, FSDP or ZeRO must operate **stage-locally** over replicas of the same stage and same model-parallel coordinates.

That means, for a given physical stage $p$:

$$
G_{\text{shard}}^{(p)}(t,c,e,t_e) = \{(d,t,p,c,e,t_e)\mid d \in [0,D-1]\}
$$

This point is critical. FSDP or ZeRO groups must **not** span multiple pipeline stages.

---

## 4.3 Virtual Pipeline Stages

When using interleaved pipeline schedules, a rank may own multiple virtual model chunks indexed by $v$.

Interoperability implication:

- checkpoint metadata must store both:
  - physical stage index $p$,
  - virtual chunk index $v$.

Changing virtual pipeline chunking without conversion is generally unsafe for exact continuation, because:

- rank-local module ordering changes,
- RNG consumption order changes,
- activation-checkpoint replay order changes,
- optimizer-state grouping may change.

---

# 5. Supported and Unsupported Interoperability Patterns

## 5.1 Support Matrix

| Pattern | Status | Recommended usage |
|---|---|---|
| Megatron-Core + native distributed optimizer | Strongly recommended | large-scale NVIDIA-centric model-parallel training |
| Megatron-Core + DeepSpeed ZeRO-$1$ | Strongly recommended | optimizer-state reduction with minimal parameter-shard complexity |
| Megatron-Core + DeepSpeed ZeRO-$2$ | Strongly recommended | larger DP groups with gradient sharding |
| Megatron-Core + DeepSpeed ZeRO-$3$ | Conditionally recommended | only if DeepSpeed is sole DP parameter-shard owner |
| Megatron-Core + stage-local FSDP | Conditionally recommended | PyTorch-native portability, memory reduction over DP axis |
| Megatron-Core + full-model outer FSDP | Discouraged | usually conflicts with PP/TP semantics unless carefully isolated |
| DeepSpeed ZeRO-$3$ + FSDP on same tensors | Forbidden | double ownership of parameter sharding |
| FSDP wrapping TP-sharded leaf tensors without tensor-axis metadata | Unsafe | checkpoint and conversion ambiguity |
| Cross-framework rank-local checkpoint copy without canonical remap | Forbidden | silent corruption risk |

---

## 5.2 Recommended Pattern A: Megatron-Core + DeepSpeed ZeRO-$1/2$

### Use when

- Megatron-Core already owns:
  - $TP$,
  - $PP$,
  - $SP$,
  - $CP$,
  - $EP$,
- the dominant memory burden is:
  - optimizer state,
  - or optimizer + gradients,
- parameter all-gather overhead of ZeRO-$3$/FSDP is not justified.

### Advantages

- simplest ownership model,
- strong checkpoint stability,
- minimal interference with Megatron tensor-parallel kernels,
- good scaling when $D$ is not large enough to justify ZeRO-$3$.

### Reasoning

If $D$ is small because much of the world size is already consumed by $TP$, $PP$, $CP$, and $EP$, the incremental savings from parameter sharding are limited.

For parameter class $j$ with model-parallel sharding factor $s_j^{MP}$:

- ZeRO-$1$ persistent memory:

$$
M_j^{Z1} \approx \frac{N_j}{s_j^{MP}} b_w + \frac{N_j}{s_j^{MP}} b_g + \frac{N_j}{s_j^{MP}D} b_o
$$

- ZeRO-$2$ persistent memory:

$$
M_j^{Z2} \approx \frac{N_j}{s_j^{MP}} b_w + \frac{N_j}{s_j^{MP}D}(b_g + b_o)
$$

If $D=2$ or $D=4$, the parameter-memory savings of ZeRO-$3$ may not justify the added communication complexity.

---

## 5.3 Recommended Pattern B: Megatron-Core + DeepSpeed ZeRO-$3$

### Use when

- parameter memory is the binding constraint,
- $D$ is large enough that full parameter sharding materially reduces HBM,
- DeepSpeed is selected as the sole data-parallel sharding owner.

### Advantages

- full DP-axis sharding of:
  - parameters,
  - gradients,
  - optimizer states,
- good memory efficiency,
- existing DeepSpeed operational ecosystem.

### Risks

- more exposed communication than ZeRO-$1/2$,
- stricter interaction requirements with:
  - TP kernels,
  - PP scheduling,
  - expert dispatch,
  - offload,
  - checkpoint conversion.

### Interoperability rule

If using DeepSpeed ZeRO-$3$, do **not** also place FSDP over the same tensors.

---

## 5.4 Recommended Pattern C: Megatron-Core + Stage-Local FSDP

### Use when

- PyTorch-native checkpointing and portability are desired,
- the training system wants FSDP semantics on the DP axis,
- model-parallel decomposition remains under Megatron-Core.

### Correct wrapping rule

FSDP should wrap modules **inside each pipeline stage**, across replicas of that stage, while preserving Megatron tensor-axis ownership.

### Good wrapping granularity

- transformer block,
- a small block group,
- MoE layer group if expert boundaries are respected.

### Bad wrapping granularity

- per-linear layer,
- per-parameter wrappers,
- wrappers that split a single tensor-parallel unit into many FSDP units,
- wrappers crossing pipeline-stage boundaries.

---

## 5.5 Discouraged Pattern: Outer Full-Model FSDP Around a Megatron Model

This pattern is frequently proposed and frequently incorrect.

### Why it is problematic

- it obscures pipeline-stage boundaries,
- it may flatten parameters across different tensor-parallel semantics,
- it complicates optimizer-state ownership,
- it makes checkpoint conversion unnecessarily difficult,
- it can create collective ordering hazards with Megatron runtime communications.

### Exception

It can work in reduced-complexity settings such as:

- no pipeline parallelism,
- minimal or no tensor parallelism,
- carefully controlled wrapping boundaries,
- explicit logical-tensor manifests.

But at that point the system is effectively closer to a pure FSDP stack than a true Megatron-Core interoperability design.

---

# 6. State Taxonomy and Ownership Semantics

## 6.1 State Classes

A correct interoperability design must track all of the following state classes:

| State class | Must be checkpointed | Owner |
|---|---|---|
| dense parameters | Yes | Megatron layout + ZeRO/FSDP shard owner |
| expert parameters | Yes | Megatron EP layout + ZeRO/FSDP shard owner |
| gradients | Usually transient, optional for restart | shard owner |
| optimizer moments | Yes | optimizer / ZeRO / FSDP owner |
| master weights | Yes if used | optimizer / precision runtime |
| FP16 loss scaler | Yes if FP16 | optimizer runtime |
| FP8 scale and amax history | Yes if exact resume required | FP8 runtime |
| LR scheduler state | Yes | optimizer control plane |
| RNG states | Yes | global runtime |
| sampler/data cursor state | Yes | data runtime |
| consumed tokens / samples | Yes | global training state |
| pipeline virtual chunk metadata | Yes | Megatron-Core |
| tensor-layout metadata | Yes | checkpoint schema |

---

## 6.2 Parameter Classes Relevant to Conversion

| Tensor class | Typical Megatron layout | Interop concern |
|---|---|---|
| embedding weights | vocab-parallel or replicated | tied-weight handling |
| attention $QKV$ projection | often fused, TP-sharded | fused vs unfused conversion |
| attention output projection | row-parallel | input-axis reconstruction |
| MLP gate/up projection | often fused in gated MLPs | fused vs split conversion |
| MLP down projection | row-parallel | input-axis reconstruction |
| layernorm / RMSNorm | usually replicated over TP | may be FSDP-sharded over DP |
| router weights | replicated or lightly sharded | expert metadata compatibility |
| expert MLP weights | EP-local, maybe $ETP$-sharded | expert ownership remap |
| LM head | often tied to embeddings | tie preservation |

---

## 6.3 Fused vs Unfused Layout Normalization

This is one of the most important interoperability issues.

Different stacks may store mathematically equivalent layers in different physical layouts:

- fused $QKV$ versus separate $Q$, $K$, $V$,
- fused gate/up projection versus separate tensors,
- packed expert tensors versus per-expert tensors,
- flattened FSDP buffers versus original parameters.

### Recommended checkpoint policy

Use **two distinct checkpoint representations**:

1. **Fast local restart format**
   - framework-native,
   - fused,
   - shard-local,
   - optimized for restart speed.

2. **Canonical portable format**
   - logical tensor representation,
   - explicit layout metadata,
   - reconstructable across Megatron-Core, DeepSpeed, and FSDP.

### Best-practice canonical choice

Canonicalize to mathematically meaningful logical tensors, for example:

- $Q$, $K$, $V$ separately,
- gate and up separately,
- explicit expert dimension,
- original parameter names independent of flattening.

This reduces ambiguity during cross-framework conversion.

---

# 7. Training-Time Interoperability Architecture

## 7.1 Correct Initialization Order

The correct high-level order is:

1. determine parallel mesh,
2. initialize process groups,
3. construct Megatron-Core model partitioning,
4. assign pipeline stages and virtual chunks,
5. apply tensor-parallel and expert-parallel transforms,
6. apply activation-checkpoint policies,
7. apply one and only one DP state-sharding owner:
   - DeepSpeed ZeRO,
   - or FSDP,
8. initialize optimizer and precision runtime,
9. load or initialize states,
10. validate logical parameter coverage,
11. start training.

### Critical point

Applying FSDP or ZeRO before Megatron model-parallel layout is fully defined is architecturally incorrect.

---

## 7.2 Pseudocode: Deterministic Runtime Construction

```text
Input:
  model specification
  parallel tuple
  checkpoint metadata
  precision policy
  runtime stack choice

Output:
  training runtime ready for forward/backward/step

Procedure:
  1. Build deterministic mesh coordinates for all ranks.
  2. Create process groups for TP, PP, CP, EP, ETP, and DP-shard axes.
  3. Construct Megatron-Core modules with fixed stage and tensor-parallel ownership.
  4. Materialize virtual pipeline chunks if interleaving is enabled.
  5. Apply activation recompute policies at stable module boundaries.
  6. If DeepSpeed is selected:
       - attach ZeRO stage over DP groups only.
     Else if FSDP is selected:
       - wrap stage-local modules across DP groups only.
  7. Initialize precision runtime:
       - compute dtype
       - reduction dtype
       - optimizer-state dtype
       - optional master-weight or FP8 metadata state
  8. If resuming:
       - validate checkpoint schema compatibility
       - load logical tensors and optimizer states
       - restore RNG and sampler state
     Else:
       - initialize logical parameters deterministically
  9. Run structural validation:
       - parameter count
       - shard coverage
       - tied-weight consistency
       - expert ownership consistency
 10. Enter training loop.
```

---

## 7.3 Activation Checkpointing Interaction

Activation checkpointing must be composed carefully with ZeRO-$3$ or FSDP:

- recomputation must preserve identical parameter materialization boundaries,
- dropout RNG must be restored consistently,
- pipeline replay order must match saved configuration,
- if exact continuation is required, the checkpointing policy should not change mid-run.

Changing checkpointing policy after resume may still produce functionally correct training, but **bitwise reproducibility is lost**.

---

## 7.4 Communication Stream Ownership

When combining Megatron-Core with DeepSpeed or FSDP, communication overlap settings can fight each other.

### Typical communication classes

- $TP$ collectives,
- $CP$ collectives,
- $EP$ all-to-all,
- $DP$ gradient reduction,
- FSDP or ZeRO parameter gather,
- FSDP or ZeRO gradient reduce-scatter.

### Interoperability rule

> **Only one subsystem should schedule overlap for a given communication path.**

If Megatron overlap, DeepSpeed overlap, and FSDP prefetch all try to aggressively overlap on the same network path, the result can be:

- stream contention,
- collective reordering hazards,
- degraded overlap,
- widened tails,
- deadlocks in edge cases.

---

# 8. Checkpoint Architecture

## 8.1 Why Framework-Local Checkpoints Are Insufficient

Framework-local shard files are optimized for speed, not portability. They frequently depend on:

- rank order,
- process-group geometry,
- flattened parameter order,
- fused tensor layouts,
- engine-specific optimizer state partitioning.

That makes them unsuitable as the sole interchange format.

---

## 8.2 Required Checkpoint Layers

A robust interoperability design should use two layers:

| Layer | Purpose | Frequency |
|---|---|---|
| local fast checkpoint | fast same-layout restart | frequent |
| canonical portable checkpoint | resharding and cross-framework conversion | periodic |

### Fast checkpoint

Optimized for:
- same cluster,
- same world layout,
- same framework stack,
- minimum downtime.

### Canonical checkpoint

Optimized for:
- changed $DP$ degree,
- changed ZeRO or FSDP strategy,
- changed framework,
- offline conversion,
- cluster migration,
- mixed-vendor portability.

---

## 8.3 Canonical Logical Checkpoint Schema

Every logical tensor record should contain:

| Field | Required |
|---|---|
| logical tensor name | Yes |
| logical shape | Yes |
| dtype | Yes |
| tensor class | Yes |
| canonical layout descriptor | Yes |
| source framework layout descriptor | Yes |
| model-parallel axes applied | Yes |
| source shard offsets and extents | Yes |
| target-agnostic tensor axis semantics | Yes |
| optimizer-state references | Yes if optimizer saved |
| tied-weight group ID | Yes where applicable |
| expert ID and expert-local index | Yes for MoE |
| pipeline stage and virtual chunk ID | Yes |
| precision-runtime metadata reference | Yes for FP16/FP8 |
| checksum and statistics | Strongly recommended |

---

## 8.4 Optimizer-State Schema

Optimizer states must be attached to logical tensors, not to rank-local parameter arrays.

Per logical tensor, store:

- first moment,
- second moment,
- master weight if present,
- parameter-group hyperparameter identity,
- optimizer step counters,
- any blockwise or tensorwise scaling metadata.

### Important

If a tensor is split from fused form to canonical atomic form, the optimizer moments must be split identically. Failing to transform optimizer states with the same layout function is a classic silent corruption bug.

---

## 8.5 Precision Metadata

### FP16

Store:

- loss-scaler state,
- overflow counters,
- master weights if retained.

### BF16

Usually no loss scaler, but store:
- master weights if optimizer uses them,
- optimizer moment dtype metadata.

### FP8

Store, at minimum:

- scaling recipe identifier,
- per-tensor or per-channel scale tensors,
- inverse scales where used,
- amax histories,
- update schedule state.

### Interoperability guidance

If converting between frameworks that do not share identical FP8 runtime semantics, the safe portable representation is:

- logical weights in BF16 or FP32,
- optimizer state in stable precision,
- FP8 runtime metadata as optional auxiliary state.

---

## 8.6 MoE Metadata

For expert parameters, checkpoint metadata must include:

- global expert count,
- global expert ID,
- local expert ID on source rank,
- source $EP$ degree,
- source $ETP$ degree,
- expert tensor layout descriptor,
- any packed-expert representation descriptor.

Without this metadata, expert remapping during conversion is unsafe.

---

# 9. Resume Workflows

## 9.1 Resume with Identical Layout

This is the simplest case.

### Requirements

- same framework family,
- same world size,
- same $TP$ / $PP$ / $CP$ / $EP$ / $ETP$ / $DP$ layout,
- same precision runtime,
- same virtual pipeline configuration,
- same tokenizer and data cursor state if exact continuation required.

### Correct strategy

Use local fast checkpoint if available.

---

## 9.2 Resume with Changed $DP$ Degree Only

This is the most common elasticity case.

### Usually safe if

- logical model layout is unchanged,
- only data-parallel shard degree changes,
- canonical checkpoint or DP-reshard-aware checkpoint exists.

### Safe examples

- DeepSpeed ZeRO-$2$ with changed $D$,
- DeepSpeed ZeRO-$3$ with changed $D$,
- FSDP with changed shard world size.

### Why it is safe

Model-parallel tensor-axis ownership does not change; only the DP-axis partitioning of parameters, gradients, or optimizer states changes.

---

## 9.3 Resume with ZeRO-Stage Change

| Migration | Status | Requirements |
|---|---|---|
| ZeRO-$1 \leftrightarrow 2$ | usually safe | canonical optimizer-state repartition |
| ZeRO-$2 \leftrightarrow 3$ | conditionally safe | parameter-shard conversion + optimizer-state remap |
| ZeRO-$3 \leftrightarrow$ FSDP | conditionally safe | canonical logical tensors + DP-axis rematerialization |
| ZeRO-$1/2 \leftrightarrow$ FSDP | conditionally safe | canonical checkpoint with optimizer states |

### Key requirement

The checkpoint must encode optimizer states logically, not just as rank-local partitions.

---

## 9.4 Resume with Changed $TP$, $PP$, or $EP$

This is **not** a restart; it is a **conversion**.

| Change | Safe without conversion? | Reason |
|---|---|---|
| change $TP$ | No | tensor-axis partition changes |
| change $PP$ | No | stage ownership changes |
| change virtual pipeline $V$ | No | chunk mapping and RNG order change |
| change $EP$ | No | expert ownership changes |
| change $ETP$ | No | expert tensor-axis partition changes |

These cases require offline or pre-resume conversion via canonical logical checkpoints.

---

## 9.5 Resume with Changed $CP$ or $SP$

This is a subtle case.

- $SP$ and $CP$ typically do not change parameter ownership.
- Therefore, parameter checkpoint conversion is often not required.

However:

- exact deterministic continuation is usually lost,
- attention kernel execution order changes,
- dropout and recompute RNG consumption can change.

### Practical rule

- **Functionally safe** if model state is loaded correctly.
- **Not bitwise deterministic** unless the same runtime path is preserved.

---

## 9.6 Pseudocode: Resume Decision Engine

```text
Input:
  source checkpoint manifest
  source parallel layout
  target parallel layout
  source framework
  target framework

Output:
  resume mode

Procedure:
  1. Compare logical model definition.
  2. Compare tensor-axis ownership:
       - TP
       - PP
       - EP
       - ETP
  3. Compare data-parallel shard ownership:
       - ZeRO stage
       - FSDP shard degree
       - DP size
  4. Compare precision runtime metadata requirements.
  5. If only DP-axis ownership changed:
       - return DP-only reshard resume
  6. Else if TP, PP, EP, ETP, or virtual stage mapping changed:
       - return full conversion required
  7. Else if only SP or CP changed:
       - return functional resume with determinism caveat
  8. Else:
       - return direct same-layout restart
```

---

# 10. Resharding Algorithms

## 10.1 DP-Only Resharding

This is the most straightforward conversion.

Given a logical tensor $X$ and new DP shard count $D'$:

1. reconstruct the logical tensor from source DP shards,
2. repartition across $D'$,
3. repartition optimizer states identically,
4. write new local shards.

### Cost

For total logical tensor size $n$ bytes, DP-only resharding is essentially a full logical read and repartition of:

$$
n_{\text{param}} + n_{\text{optimizer}}
$$

No tensor-axis transforms are required if $TP$ and $EP$ ownership remain fixed.

---

## 10.2 Tensor-Parallel Resharding

For a tensor-parallel split, conversion depends on the tensor class:

- column-parallel weights split along output-feature axis,
- row-parallel weights split along input-feature axis,
- vocab-parallel embeddings split along vocabulary axis,
- fused $QKV$ often combine both axis semantics and fusion layout.

### Required invariant

The conversion system must know the **mathematical axis** of the tensor-parallel partition, not just the source shard shape.

---

## 10.3 Pipeline Repartitioning

Pipeline conversion requires:

- reconstructing logical layer ordering,
- reassigning layers to physical and virtual stages,
- re-emitting stage-local shards,
- preserving global layer indices,
- preserving optimizer-state mapping by global parameter identity.

Changing pipeline partitioning without global layer identity metadata is unsafe.

---

## 10.4 Expert-Parallel Repartitioning

Expert conversion requires:

- reconstructing logical expert IDs,
- mapping source experts to target expert owners,
- optionally redistributing expert-local tensor-parallel shards,
- converting packed expert tensors if source and target packing differ.

### Important

Expert remapping is not just a data copy. It is a **semantic reassignment** of global expert IDs to new ranks.

---

## 10.5 Pseudocode: Full Reshard Conversion

```text
Input:
  canonical checkpoint
  source mesh
  target mesh
  tensor layout registry
  optimizer schema

Output:
  target sharded checkpoint

Procedure:
  1. For each logical tensor:
       a. Read canonical tensor metadata.
       b. Reconstruct logical full tensor or stream logical chunks.
       c. Determine target tensor layout:
            - replicated
            - TP-column split
            - TP-row split
            - vocab split
            - EP-local
            - ETP split
            - DP/FSDP shard
       d. Apply required transforms:
            - fuse or unfuse
            - concatenate or split
            - reorder expert axes
            - repartition stage ownership
       e. Partition into target physical shards.
       f. Apply identical transform to optimizer states.
  2. Rebuild target checkpoint manifest.
  3. Validate coverage, checksums, and shard non-overlap.
```

---

# 11. Cross-Framework Conversion Paths

## 11.1 Megatron-Core Native to DeepSpeed ZeRO

### Easier cases

- same $TP$ / $PP$ / $EP$ layout,
- change only DP-state ownership to ZeRO-$1$ or ZeRO-$2$,
- or attach ZeRO-$3$ as sole DP sharder.

### Required conversions

- optimizer-state repartition from native or distributed optimizer format,
- canonical parameter-group mapping,
- scheduler and scaler state transfer.

### Hard cases

- different fused tensor layouts,
- expert packing changes,
- changed virtual pipeline chunking.

---

## 11.2 Megatron-Core Native to FSDP

### Recommended approach

- export canonical logical checkpoint,
- reconstruct stage-local logical modules,
- apply FSDP wrapping on target runtime,
- load logical tensors into FSDP-owned shards.

### Important detail

If source Megatron uses fused tensors and target FSDP stack expects unfused modules, conversion must normalize those tensors logically first.

---

## 11.3 DeepSpeed ZeRO to FSDP

This is feasible if and only if:

- checkpoint provides logical tensor metadata,
- optimizer states are not trapped in opaque local partition formats,
- tensor identity is independent of flattened ZeRO buffer layout.

### Common pitfall

Local ZeRO shard files are often insufficient by themselves for portable conversion.

---

## 11.4 FSDP to Megatron-Core

This is feasible only if FSDP checkpointing preserves:

- original logical parameter names,
- tensor layout metadata,
- unflatten mapping if flattening was used,
- fused/unfused transformation rules.

### Strong recommendation

For any FSDP model intended for Megatron conversion, either:

- disable flattening for portable checkpoints,
- or emit a complete unflatten manifest.

---

## 11.5 Tensor Transformation Matrix

| Tensor class | Native Megatron | DeepSpeed ZeRO | FSDP | Conversion requirement |
|---|---|---|---|---|
| dense TP column weight | TP-sharded | same logical TP sharding | same TP logical layout, DP sharded by FSDP | preserve TP axis, change only DP ownership |
| dense TP row weight | TP-sharded | same | same | preserve TP axis and optimizer mapping |
| fused $QKV$ | often fused | may remain fused | may be fused or split | canonicalize or descriptor-driven transform |
| fused gate/up | often fused | may remain fused | may be split | canonicalize or descriptor-driven transform |
| expert weights | EP-local, optional $ETP$ | same | stage-local FSDP over DP with expert ownership fixed | expert-ID-aware mapping |
| norm weights | replicated over TP | same | DP sharded in FSDP if wrapped | preserve logical identity |
| embeddings / LM head | vocab-parallel or tied | same | FSDP or ZeRO DP ownership change | tied-group preservation |

---

# 12. Memory and Communication Implications of Interoperability Choices

## 12.1 State Memory by ZeRO Stage or FSDP

For tensor class $j$ with model-parallel sharding factor $s_j^{MP}$:

- ZeRO-$1$:

$$
M_j^{Z1} \approx \frac{N_j}{s_j^{MP}}(b_w + b_g) + \frac{N_j}{s_j^{MP}D}b_o
$$

- ZeRO-$2$:

$$
M_j^{Z2} \approx \frac{N_j}{s_j^{MP}}b_w + \frac{N_j}{s_j^{MP}D}(b_g + b_o)
$$

- ZeRO-$3$ or FSDP:

$$
M_j^{Z3/F} \approx \frac{N_j}{s_j^{MP}D}(b_w + b_g + b_o) + M_{j,\text{transient}}
$$

Where $M_{j,\text{transient}}$ is the temporary parameter-gather footprint during forward/backward.

### Decision insight

If $D$ is small, ZeRO-$3$/FSDP savings may be modest relative to the added communication.

---

## 12.2 Communication Cost of Full Sharding

Ignoring bucketization detail, ZeRO-$3$/FSDP add DP-axis parameter materialization traffic approximately proportional to:

$$
V_{\text{param-gather}} \approx 2 \sum_j \frac{D-1}{D} N_j b_w
$$

The factor $2$ reflects:

- forward all-gather,
- backward rematerialization all-gather.

Gradient synchronization adds approximately:

$$
V_{\text{grad-rs}} \approx \sum_j \frac{D-1}{D} N_j b_g
$$

This is in addition to:

- Megatron $TP$ collectives,
- pipeline send/recv,
- context-parallel communication,
- expert all-to-all.

### Practical consequence

When the model already uses substantial $TP$, $CP$, and $EP$, introducing ZeRO-$3$/FSDP must be justified by hard memory necessity, not by default.

---

# 13. Mixed Precision, FP8, and Numerical Portability

## 13.1 Precision Layers That Must Interoperate

A complete portability design must account for:

- parameter storage dtype,
- compute dtype,
- reduction dtype,
- optimizer-state dtype,
- master-weight dtype,
- scaling metadata dtype.

---

## 13.2 BF16

BF16 is the easiest interoperable precision mode because it usually avoids dynamic loss scaling and is supported consistently across modern NVIDIA and AMD platforms.

### Checkpoint requirements

- logical parameters,
- optimizer moments,
- scheduler state,
- RNG and sampler state.

---

## 13.3 FP16

FP16 requires:

- loss-scaler state,
- overflow history,
- master weights if optimizer uses them.

### Conversion risk

Migrating FP16 checkpoints across frameworks without preserving scaler and master weights can produce immediate divergence or slow instability.

---

## 13.4 FP8

FP8 introduces the most portability complexity.

### Store at minimum

- amax histories,
- scale tensors,
- inverse scales if present,
- recipe metadata,
- update interval state.

### Recommended portability policy

If exact FP8 runtime equivalence is not guaranteed across source and target framework stacks:

- serialize logical weights in BF16 or FP32,
- serialize optimizer states in stable precision,
- treat FP8 runtime metadata as auxiliary and optionally reinitialize it under controlled recalibration.

---

## 13.5 Parity Validation

Any interoperability conversion involving precision changes must pass:

- same-input forward loss comparison,
- one-step optimizer delta comparison,
- gradient norm comparison,
- short fixed-token loss-curve comparison.

---

# 14. Activation Checkpointing, CUDA Graphs, and Compile Interactions

## 14.1 Activation Checkpointing

Checkpointing boundaries must be aligned with:

- Megatron block structure,
- FSDP wrapping boundaries if used,
- ZeRO-$3$ parameter materialization windows,
- MoE routing replay boundaries.

### Unsafe pattern

Changing recompute boundaries during resume without acknowledging that:

- dropout stream ordering,
- activation materialization timing,
- and bitwise exactness

will change.

---

## 14.2 CUDA Graphs and Graph Capture

Graph capture across interoperable stacks is safe only if:

- communicator topology is fixed,
- parameter materialization shapes are static,
- no dynamic resharding happens inside the captured region,
- MoE token dispatch shapes are sufficiently stabilized or excluded,
- all lazy initialization is completed before capture.

### Implication

Capture should occur only after:

- process groups are finalized,
- checkpoints are loaded,
- optimizer states are created,
- any FSDP or ZeRO lazy buffers are materialized.

---

# 15. Long-Context and MoE Interoperability Constraints

## 15.1 Context Parallelism

$CP$ typically affects activation partitioning, not parameter ownership.

Therefore:

- checkpoints usually do not require tensor conversion when only $CP$ changes,
- but exact continuation is not guaranteed due to changed execution order and RNG stream consumption.

### Recommendation

Treat $CP$ changes as:

- **portable for model state**,
- **non-identical for exact numerical replay**.

---

## 15.2 Expert Parallelism

$EP$ directly changes parameter ownership for expert weights.

Therefore, changing $EP$ degree requires:

- expert-ID-aware conversion,
- expert optimizer-state remap,
- expert-local tensor metadata remap.

Changing only dense $DP$ without changing expert ownership is a much simpler case.

---

## 15.3 Token Dropping and Routing Determinism

For exact MoE continuation, checkpoint and runtime must preserve:

- router parameters,
- routing tie-break policy,
- token-dropping or overflow policy,
- any capacity-factor schedule,
- any auxiliary load-balancing coefficients.

### Critical warning

If the routing policy changes between source and target stacks, training may remain functional but is **not a true continuation**.

---

# 16. Failure Modes and Root-Cause Analysis

## 16.1 Failure Matrix

| Symptom | Likely root cause | Correct diagnosis method |
|---|---|---|
| load succeeds but loss diverges immediately | optimizer states mismapped | compare per-tensor moment statistics and one-step deltas |
| missing tensor on resume | stage or tensor-layout metadata incomplete | validate canonical manifest coverage |
| checkpoint loads but wrong shapes on target ranks | TP or EP axis semantics lost | inspect tensor-class layout registry |
| deadlock during first step after conversion | process-group construction mismatch | compare rank-local mesh manifests |
| expert layers fail only after resume | expert-ID remap incorrect | verify expert ownership table |
| FP8 resume unstable | missing scale or amax metadata | inspect precision auxiliary state |
| tied embedding / LM head diverges | tie relationship lost | validate tied-group IDs and shared optimizer state |
| same-layout restart slower than before | wrong fast-checkpoint path | inspect local-vs-canonical load path |
| identical world size but different results | RNG or recompute boundaries changed | compare checkpointing policy and RNG restoration |

---

## 16.2 Structural Validation Required Before First Training Step

After load or conversion, validate:

- full logical parameter count,
- no duplicated logical tensors,
- no missing logical tensors,
- tied-weight equivalence,
- optimizer-state count equals parameter-state count where required,
- global layer index continuity,
- expert ID coverage,
- no overlapping shard ranges,
- no orphan precision metadata.

---

## 16.3 Pseudocode: Post-Conversion Validator

```text
Input:
  target checkpoint
  target runtime manifest
  validation policy

Output:
  pass or fail

Procedure:
  1. Validate logical tensor inventory against model definition.
  2. Validate shard coverage:
       - full coverage
       - no overlap
       - correct extents
  3. Validate tied tensors and alias groups.
  4. Validate optimizer state presence and shape.
  5. Validate expert ownership mapping.
  6. Validate precision auxiliary states.
  7. Run forward parity on a fixed batch if reference is available.
  8. Run single optimizer step parity if required.
  9. Emit failure if any structural or numerical invariant is violated.
```

---

# 17. Automation Blueprint for Interoperability

## 17.1 Required Control-Plane Objects

A production control plane should maintain:

- model manifest,
- tensor layout registry,
- mesh manifest,
- checkpoint schema version,
- conversion rule version,
- optimizer schema version,
- precision schema version,
- data-lineage and sampler state schema,
- environment fingerprint.

---

## 17.2 Dual-Checkpoint Policy

A mature system should emit:

1. **frequent restart checkpoints**
   - local,
   - framework-native,
   - minimal downtime.

2. **periodic portability checkpoints**
   - canonical,
   - validated,
   - suitable for reshard and conversion.

This design prevents operators from paying the portability cost at every checkpoint while still preserving recovery and migration paths.

---

## 17.3 Pseudocode: Train / Save / Resume / Convert Control Loop

```text
Input:
  training run state
  checkpoint cadence policy
  portability cadence policy

Output:
  consistent restart and conversion artifacts

Procedure:
  During training:
    1. Save local fast checkpoint at restart cadence.
    2. Save canonical portable checkpoint at portability cadence.
    3. Validate each checkpoint manifest and checksum before commit.
    4. Persist RNG, sampler, scheduler, and global token counters.

  On resume:
    5. If same-layout restart is requested and fast checkpoint is valid:
         load local fast checkpoint.
       Else:
         load canonical checkpoint and reshard if needed.

  On conversion:
    6. Load canonical checkpoint only.
    7. Apply target layout conversion.
    8. Run structural and numerical validation.
    9. Emit target checkpoint family.
```

---

# 18. Recommended Interoperability Decisions by Scenario

## 18.1 When to Prefer Megatron-Core + DeepSpeed ZeRO-$1/2$

Choose this when:

- $TP$, $PP$, $CP$, and $EP$ are already substantial,
- $D$ is relatively small,
- optimizer state is the main memory problem,
- you want lower complexity than full parameter sharding,
- you prioritize steady large-scale production training.

---

## 18.2 When to Prefer Megatron-Core + DeepSpeed ZeRO-$3$

Choose this when:

- parameter memory is the hard limiter,
- $D$ is large enough that parameter sharding materially helps,
- DeepSpeed is already the operational standard,
- you can tolerate more DP-axis gather traffic.

---

## 18.3 When to Prefer Megatron-Core + FSDP

Choose this when:

- PyTorch-native checkpointing and portability matter,
- cross-vendor portability matters,
- stage-local FSDP can be applied cleanly,
- you want a more canonical state-dict-based conversion path.

---

## 18.4 When Not to Mix Them

Do not mix FSDP and ZeRO-$3$ on the same parameter set.

Do not convert via local shard copies.

Do not change $TP$, $PP$, or $EP$ on resume without canonical conversion.

Do not rely on flattened FSDP buffers for portability unless a full unflatten manifest exists.

---

# 19. Best-Practice Interoperability Rules

## 19.1 Design Rules

> **Rule 1:** Megatron-Core owns model-parallel topology.

> **Rule 2:** DeepSpeed or FSDP owns DP-axis state sharding, never both for the same tensor set.

> **Rule 3:** Conversion works on canonical logical tensors, not on local shards.

> **Rule 4:** Fused tensor layouts are a performance optimization, not a portable storage abstraction.

> **Rule 5:** Optimizer states must undergo the same tensor transforms as parameters.

> **Rule 6:** Exact continuation requires restoring RNG, sampler, virtual stage mapping, and checkpointing policy.

> **Rule 7:** $CP$ and $SP$ changes are usually model-state portable but not bitwise identical.

> **Rule 8:** $TP$, $PP$, $EP$, and $ETP$ changes are conversion events, not restart events.

---

# 20. Final Conclusions

1. **Megatron-Core, DeepSpeed, and FSDP are interoperable only when ownership boundaries are explicit and enforced.**

2. **The most robust production architecture is hierarchical:**
   - Megatron-Core for model-parallel decomposition,
   - DeepSpeed or FSDP for DP-axis state sharding,
   - canonical checkpointing for portability.

3. **Local checkpoint formats are not portability formats.**
   They are restart artifacts only.

4. **Most failed conversions are not caused by tensor bytes; they are caused by lost semantics:**
   - fused layout semantics,
   - TP axis meaning,
   - stage ownership,
   - expert IDs,
   - optimizer-state mapping,
   - precision metadata.

5. **The only safe universal conversion path is:**
   - reconstruct logical tensors,
   - apply explicit layout transforms,
   - repartition onto target mesh,
   - validate structurally and numerically.

6. **Changing $DP$ is usually a reshard. Changing $TP$, $PP$, or $EP$ is a full conversion.**
   This distinction should be embedded into the control plane and checkpoint policy.

---

# 21. Reference Links

## Core Frameworks
- Megatron-LM / Megatron-Core: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- PyTorch DTensor: https://pytorch.org/docs/stable/distributed.tensor.html

## Checkpointing and Runtime
- PyTorch Distributed Checkpointing: https://pytorch.org/docs/stable/distributed.checkpoint.html
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/

## Precision and Kernels
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Triton: https://github.com/triton-lang/triton



 the next document should be a **canonical checkpoint specification** for Megatron-Core, DeepSpeed ZeRO, and FSDP, including:

- tensor registry schema,
- fused-to-canonical transform registry,
- expert remap schema,
- optimizer-state transform rules,
- same-layout restart schema,
- conversion validation protocol.