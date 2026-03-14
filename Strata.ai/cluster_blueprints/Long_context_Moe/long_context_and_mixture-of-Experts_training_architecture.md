# Technical Report: Long-Context and MoE Training Architecture with Exact Parallel-Group Factorization, Communication Reasoning, and Memory Formulas

## 1. Scope

This report defines the end-to-end distributed training architecture for **long-context** and **Mixture-of-Experts** LLM training. It covers:

- exact parallel-group factorization,
- topology-aware placement of $DP$, $TP$, $PP$, $SP$, $CP$, $EP$, and $ETP$,
- dense-path and expert-path communication reasoning,
- long-context attention variants including standard $CP$, Ulysses-style redistribution, and Ring Attention,
- parameter, activation, optimizer-state, and communication-buffer memory formulas,
- pipeline and microbatch derivation,
- routing balance, token dropping, and MoE stability,
- FSDP / ZeRO interaction with long-context and MoE state ownership,
- production constraints across NVIDIA and AMD clusters.

The goal is not a high-level overview. The goal is a precise systems blueprint that explains **why a given configuration fits, scales, or fails**.

---

## 2. Design Rules

> **Rule 1:** Put the highest-frequency collectives on the fastest local fabric.

> **Rule 2:** Long-context training is usually activation- and attention-communication-bound before it is parameter-memory-bound.

> **Rule 3:** MoE training is usually token-dispatch-bound before it is expert-parameter-bound.

> **Rule 4:** $EP$ should be carved from the cheapest possible communication domain, usually local NVLink/NVSwitch or xGMI.

> **Rule 5:** $CP$ changes activation partitioning; $EP$ changes parameter ownership.

> **Rule 6:** Changing $CP$ is often model-state portable; changing $EP$ or $ETP$ is a checkpoint conversion event.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $W$ | total accelerator count |
| $D$ | data-parallel degree |
| $D_r$ | replica count of expert sets in MoE layouts |
| $T$ | dense tensor-parallel degree |
| $P$ | pipeline stage count |
| $C$ | context-parallel degree |
| $E$ | expert-parallel degree |
| $T_e$ | expert tensor-parallel degree |
| $B_\mu$ | per-rank microbatch size |
| $m$ | number of microbatches per optimizer step |
| $S$ | global sequence length |
| $H$ | model hidden size |
| $H_{ff}$ | dense MLP hidden size |
| $H_{ff}^{exp}$ | expert MLP hidden size |
| $f_{mlp}$ | dense MLP expansion factor, where $H_{ff} = f_{mlp} H$ |
| $f_{exp}$ | expert MLP expansion factor, where $H_{ff}^{exp} = f_{exp} H$ |
| $n_h$ | number of attention heads |
| $k$ | MoE router top-$k$ |
| $N_e$ | total experts in one MoE layer |
| $b_w$ | bytes per parameter |
| $b_g$ | bytes per gradient |
| $b_o$ | bytes per optimizer-state element |
| $b_a$ | bytes per activation element |
| $M_{\max}$ | available HBM budget per rank |
| $\alpha$ | collective startup latency |
| $\beta$ | per-byte transfer coefficient |

---

## 4. Architectural Decomposition

A long-context MoE model has two fundamentally different distributed subsystems:

1. **Dense trunk**
   - embeddings,
   - attention,
   - dense MLP blocks,
   - norms,
   - LM head.

2. **Sparse expert subsystem**
   - router,
   - expert assignment,
   - token dispatch,
   - expert-local MLP execution,
   - token combine.

These subsystems have different scaling laws:

- dense long-context scaling is dominated by:
  - activation memory,
  - attention communication,
  - sequence-length-dependent workspace.

- MoE scaling is dominated by:
  - expert all-to-all,
  - local expert batch size,
  - routing imbalance,
  - expert optimizer-state ownership.

A correct system therefore uses **different active sharding axes for dense layers and expert layers**.

---

# 5. Exact Parallel-Group Factorization

## 5.1 Dense Long-Context Factorization

For dense long-context training:

$$
W = D \times T \times P \times C
$$

Where:

- $D$ owns data parallelism or FSDP / ZeRO state sharding,
- $T$ owns dense tensor parallelism,
- $P$ owns pipeline stage decomposition,
- $C$ owns context partitioning,
- $SP$ is typically coupled to $T$, not a separate multiplicative axis.

### Dense rank coordinate

Define each rank by:

$$
r = (d, t, p, c)
$$

with:

- $d \in [0, D-1]$,
- $t \in [0, T-1]$,
- $p \in [0, P-1]$,
- $c \in [0, C-1]$.

### Dense process groups

- Tensor-parallel group:

$$
G_{TP}(d,p,c) = \{(d,t,p,c)\mid t \in [0,T-1]\}
$$

- Pipeline group:

$$
G_{PP}(d,t,c) = \{(d,t,p,c)\mid p \in [0,P-1]\}
$$

- Context-parallel group:

$$
G_{CP}(d,t,p) = \{(d,t,p,c)\mid c \in [0,C-1]\}
$$

- Data-parallel or shard group:

$$
G_{DP}(t,p,c) = \{(d,t,p,c)\mid d \in [0,D-1]\}
$$

---

## 5.2 MoE Factorization: Practical Production Form

The most useful production factorization is not the naive orthogonal MoE mesh. The practical factorization is:

$$
W = D_r \times E \times T \times P \times C \times T_e
$$

Where:

- $D_r$ = number of replicas of the full expert set,
- $E$ = expert-parallel degree,
- $T$ = dense tensor-parallel degree,
- $P$ = pipeline stage count,
- $C$ = context-parallel degree,
- $T_e$ = expert tensor-parallel degree.

## 5.2.1 Why this factorization is correct

In practical MoE systems:

- dense layers are effectively replicated over what becomes the expert axis,
- expert layers use that same axis as model parallelism rather than data parallelism.

Therefore:

- **dense effective data-parallel degree** is:

$$
D_{\text{dense}} = D_r \times E
$$

- **expert replica degree** is:

$$
D_{\text{expert}} = D_r
$$

This distinction is crucial for:

- optimizer-state ownership,
- FSDP / ZeRO sharding degree,
- checkpoint metadata,
- load-balance reasoning.

### Interpretation

Ranks with different $e$ coordinates:

- behave like dense replicas for non-MoE layers,
- but own different expert subsets for MoE layers.

That is the correct production semantics.

---

## 5.3 MoE Rank Coordinate

Define rank coordinate as:

$$
r = (d_r, e, t, p, c, t_e)
$$

with:

- $d_r \in [0, D_r-1]$,
- $e \in [0, E-1]$,
- $t \in [0, T-1]$,
- $p \in [0, P-1]$,
- $c \in [0, C-1]$,
- $t_e \in [0, T_e-1]$.

---

## 5.4 MoE Process Groups

### Dense tensor-parallel group

$$
G_{TP}(d_r,e,p,c,t_e) = \{(d_r,e,t,p,c,t_e)\mid t \in [0,T-1]\}
$$

### Context-parallel group

$$
G_{CP}(d_r,e,t,p,t_e) = \{(d_r,e,t,p,c,t_e)\mid c \in [0,C-1]\}
$$

### Pipeline group

$$
G_{PP}(d_r,e,t,c,t_e) = \{(d_r,e,t,p,c,t_e)\mid p \in [0,P-1]\}
$$

### Expert-parallel group

$$
G_{EP}(d_r,t,p,c,t_e) = \{(d_r,e,t,p,c,t_e)\mid e \in [0,E-1]\}
$$

### Expert tensor-parallel group

$$
G_{ETP}(d_r,e,t,p,c) = \{(d_r,e,t,p,c,t_e)\mid t_e \in [0,T_e-1]\}
$$

### Dense DP / FSDP group

For dense parameters, the effective sharding or replica group is over both $d_r$ and $e$:

$$
G_{DP}^{dense}(t,p,c,t_e) = \{(d_r,e,t,p,c,t_e)\mid d_r,e\}
$$

with size:

$$
|G_{DP}^{dense}| = D_r E
$$

### Expert DP / FSDP group

For expert parameters, only ranks with the same expert ownership coordinate participate:

$$
G_{DP}^{expert}(e,t,p,c,t_e) = \{(d_r,e,t,p,c,t_e)\mid d_r\}
$$

with size:

$$
|G_{DP}^{expert}| = D_r
$$

This is the exact ownership distinction that many incorrect reports omit.

---

# 6. Global Batch, Microbatch, and Pipeline Constraints

## 6.1 Global Batch

For both dense and MoE models, the optimizer-step global batch is:

$$
B_{\text{global}} = B_\mu \times m \times D_{\text{dense}}
$$

where:

$$
D_{\text{dense}} = D \quad \text{for dense-only models}
$$

and

$$
D_{\text{dense}} = D_r E \quad \text{for the practical MoE factorization above}
$$

Tokens per optimizer step:

$$
N_{\text{tok/step}} = B_{\text{global}} \times S
$$

---

## 6.2 Pipeline Bubble Efficiency

For non-interleaved $1F1B$ schedules:

$$
\eta_{PP} \approx \frac{m}{m + P - 1}
$$

### Engineering consequence

If $P$ is deep and $m$ is too small:

- pipeline bubbles dominate,
- overlap windows shrink,
- $CP$ and $EP$ exposure becomes visible,
- long-context jobs lose scaling efficiency rapidly.

---

## 6.3 Stage Balance for MoE Pipelines

For stage $p$, the actual time is not determined by layer count. It is:

$$
T_p = \sum_{l \in \mathcal{L}_p^{dense}} T_l^{dense} + \sum_{l \in \mathcal{L}_p^{moe}} \left(T_l^{router} + T_l^{dispatch} + T_l^{expert} + T_l^{combine}\right)
$$

The pipeline partition objective is therefore:

$$
\min \max_p T_p
$$

not:

- equal number of layers per stage,
- equal parameter count per stage.

This is especially important when:

- some stages contain more MoE layers,
- long-context attention layers are significantly slower,
- embeddings or LM head are large.

---

# 7. Long-Context Dense-Path Architecture

## 7.1 Why Long Context Is Hard

For large $S$, dense training is constrained by:

- residual-path activation memory,
- attention workspace,
- synchronization on $TP$ and $CP$ groups,
- lower arithmetic intensity when local token slices become too small.

The naive score matrix is:

$$
M_{\text{scores,naive}} \approx B_\mu \times \frac{n_h}{T} \times \frac{S}{C} \times S \times b_a
$$

This is the primary reason long-context training requires:

- FlashAttention-class kernels,
- checkpointing,
- $SP$,
- $CP$.

Without FlashAttention-style tiling, the score term grows as:

$$
O\left(\frac{S^2}{C}\right)
$$

per rank.

---

## 7.2 Local Token Count

With context parallelism:

$$
S_c = \frac{S}{C}
$$

If sequence parallel is active over $T$ and the router / local ops consume sequence-sharded activations, then the per-rank local token count can become:

$$
S_{c,t} = \frac{S}{C T}
$$

Whether the exact local token count is $S/C$ or $S/(CT)$ depends on the specific kernel path and whether the operation runs on sequence-sharded or gathered activations. Reports that ignore this distinction are not precise enough for production planning.

---

## 7.3 Dense Activation Memory Per Block

A useful first-order formula for dense block activations is:

$$
M_{\text{act,dense-block}} \approx \frac{B_\mu S H b_a}{C} \left(\alpha_{\text{rep}} + \frac{\alpha_{\text{shard}}}{T}\right) + M_{\text{attn-ws}}
$$

Where:

- $\alpha_{\text{rep}}$ counts activation terms replicated across $TP$ ranks,
- $\alpha_{\text{shard}}$ counts activation terms sequence-sharded across $TP$ ranks,
- $M_{\text{attn-ws}}$ is attention kernel workspace,
- both coefficients depend on checkpointing policy.

### Interpretation

- Without $SP$, $\alpha_{\text{rep}}$ is larger.
- With $SP$, more activations move into $\alpha_{\text{shard}} / T$.
- With aggressive activation checkpointing, both coefficients fall, but recompute FLOPs increase.

---

## 7.4 FlashAttention Effect

FlashAttention removes explicit materialization of the full score matrix. Therefore the dominant attention memory term changes from:

$$
O\left(B_\mu \frac{n_h}{T} \frac{S}{C} S\right)
$$

to a tile workspace and streaming state:

$$
M_{\text{attn-ws}} = O\left(B_\mu \frac{n_h}{T} \frac{S}{C} \times \text{tile size}\right)
$$

This is the enabling condition for long-context training beyond moderate $S$.

---

# 8. Context Parallelism Architectures

## 8.1 Standard Context Parallelism

Standard $CP$ partitions tokens across $C$ ranks:

- each rank holds local query positions,
- attention still requires access to global or streamed $K$ and $V$,
- communication occurs per attention layer.

### Best use case

- moderate $C$,
- strong local fabric,
- minimal need for complex sequence-head transposes.

### Risk

If $CP$ extends across slow inter-node links, attention-layer latency dominates quickly.

---

## 8.2 Ulysses-Style Context Parallelism

Ulysses-style methods perform sequence-to-head redistributions so that attention can execute with a different local partitioning than the original hidden-state layout.

### First-order redistribution cost

For a tensor of size $B_\mu S H$ all-to-all redistributed across $C$ ranks, per-rank bytes are approximately:

$$
V_{\text{A2A}}(B_\mu,S,H,C) \approx B_\mu S H b_a \left(1-\frac{1}{C}\right)
$$

A Ulysses-style attention path typically performs multiple such redistributions per layer. First-order forward communication is approximately:

$$
V_{\text{Ulysses,fwd}} \approx 4 B_\mu S H b_a \left(1-\frac{1}{C}\right)
$$

The factor $4$ corresponds to the combined movement of $Q$, $K$, $V$, and output-equivalent hidden states under a common sequence-head transpose interpretation.

Backward is of the same order, so training communication per layer is approximately:

$$
V_{\text{Ulysses,train}} \approx 8 B_\mu S H b_a \left(1-\frac{1}{C}\right)
$$

### When Ulysses is attractive

- local all-to-all is fast,
- head count and kernel schedule make transpose-based attention efficient,
- sequence length is large enough that standard approaches overexpose attention communication.

### When it is not

- local all-to-all is weak,
- cross-node deployment is unavoidable,
- head count is too small to preserve good local GEMM shape after redistribution.

---

## 8.3 Ring Attention

Ring Attention streams $K$ and $V$ blocks around the $CP$ ring instead of executing a bulk all-to-all transpose.

For local query tokens of size $S/C$, each rank sends and receives its $K$ and $V$ blocks for $C-1$ hops.

Per-rank forward communication is approximately:

$$
V_{\text{Ring,fwd}} \approx 2(C-1) \times B_\mu \times \frac{S}{C} \times H \times b_a
$$

The factor $2$ accounts for both $K$ and $V$.

Backward is of the same order, so training communication per layer is approximately:

$$
V_{\text{Ring,train}} \approx 4(C-1) \times B_\mu \times \frac{S}{C} \times H \times b_a
$$

### When Ring Attention is attractive

- extreme sequence lengths,
- need to avoid bursty large all-to-all,
- desire to exploit pipeline-like streaming overlap.

### Risk

Latency grows with ring steps. If $C$ is too large and local compute per step is too small, ring step latency becomes visible.

---

## 8.4 Context-Parallel Method Selection

| Method | Communication pattern | Strength | Weakness | Best deployment zone |
|---|---|---|---|---|
| standard $CP$ | collectives around attention | simpler integration | can expose global sync | moderate $C$ inside fast local domain |
| Ulysses | all-to-all redistributions | efficient for large local groups and suitable head structure | bursty all-to-all and transpose cost | NVSwitch / xGMI-local |
| Ring Attention | streaming ring exchange | extreme-$S$ friendly, avoids large burst | ring latency and scheduling complexity | when sequence is very large and local ring bandwidth is strong |

---

# 9. MoE Architecture

## 9.1 Expert Ownership

Let one MoE layer contain $N_e$ experts globally.

If $E$ is the expert-parallel degree, then each rank owns:

$$
N_{e,\text{local}} = \frac{N_e}{E}
$$

experts, assuming no expert replication inside the same expert set.

If expert tensor parallelism is used, each local expert is further sharded by $T_e$.

---

## 9.2 Router Semantics

For top-$k$ routing, each token selects $k$ experts.

The router output must preserve:

- deterministic expert ordering under ties,
- consistent top-$k$ selection across ranks,
- stable token permutation order before dispatch.

Any nondeterminism here causes:

- different all-to-all payloads,
- inconsistent expert load,
- non-reproducible optimizer trajectories.

---

## 9.3 Tokens Entering One EP Group

Let the local token count presented to one rank before dispatch be:

$$
N_t
$$

Then one expert-parallel group of size $E$ sees total tokens:

$$
N_{\text{tok,EPgrp}} = E \times N_t
$$

If $SP$ is active over $T$, a common case is:

$$
N_t = \frac{B_\mu S}{C T}
$$

If router input is gathered before routing, then instead:

$$
N_t = \frac{B_\mu S}{C}
$$

The exact value depends on the runtime path. This must be checked against the actual dispatcher implementation.

---

## 9.4 Expected Expert Load

The expected token count per expert under balanced routing is:

$$
\mathbb{E}[n_{\text{expert}}] = \frac{k \, N_{\text{tok,EPgrp}}}{N_e} = \frac{k E N_t}{N_e}
$$

Since each rank owns $N_e / E$ experts, the expected total expert tokens processed by one rank is:

$$
\mathbb{E}[n_{\text{rank}}] = \frac{N_e}{E} \times \frac{k E N_t}{N_e} = k N_t
$$

This is a critical result:

> Under balanced routing, **per-rank total expert token load is approximately independent of $E$**.

What changes with $E$ is not total local token load but:

- local expert count,
- per-expert batch size,
- all-to-all communication fraction.

The expected token count per local expert is:

$$
\mathbb{E}[n_{\text{local expert}}] = \frac{k N_t}{N_e/E} = \frac{k E N_t}{N_e}
$$

If this quantity becomes too small, grouped GEMM efficiency collapses.

---

## 9.5 Capacity Factor and Token Dropping

With capacity factor $f_{\text{cap}}$, expert capacity is:

$$
\text{cap} = \left\lceil f_{\text{cap}} \times \frac{k E N_t}{N_e} \right\rceil
$$

The maximum local expert tokens on one rank are then:

$$
N_{\text{tok,local}}^{\max} = \frac{N_e}{E} \times \text{cap}
$$

Using the approximation above:

$$
N_{\text{tok,local}}^{\max} \approx f_{\text{cap}} \times k N_t
$$

### Engineering consequence

Local MoE activation and dispatch buffers scale mainly with:

- $k$,
- $N_t$,
- capacity factor,

not directly with $N_e$.

---

## 9.6 Load Imbalance Metrics

Define per-expert load imbalance ratio:

$$
\rho = \frac{\max_i n_i}{\frac{1}{N_e}\sum_{i=1}^{N_e} n_i}
$$

where $n_i$ is tokens assigned to expert $i$.

A second useful metric is coefficient of variation:

$$
CV = \frac{\sqrt{\frac{1}{N_e}\sum_{i=1}^{N_e}(n_i-\bar{n})^2}}{\bar{n}}
$$

Where:

$$
\bar{n} = \frac{1}{N_e}\sum_i n_i
$$

These two metrics should be tracked continuously in production.

---

# 10. MoE Communication Reasoning

## 10.1 Dispatch and Combine Bytes

Each dispatched token carries hidden-state payload of size $H$.

The average fraction of routed tokens leaving the local rank is approximately:

$$
1 - \frac{1}{E}
$$

So forward dispatch bytes per rank are:

$$
V_{\text{dispatch,fwd}} \approx k N_t H b_a \left(1-\frac{1}{E}\right)
$$

Forward combine bytes are the same order:

$$
V_{\text{combine,fwd}} \approx k N_t H b_a \left(1-\frac{1}{E}\right)
$$

Hence total forward MoE communication per layer is:

$$
V_{\text{MoE,fwd}} \approx 2kN_tH b_a \left(1-\frac{1}{E}\right)
$$

Backward is of the same order, so full training communication per MoE layer is:

$$
V_{\text{MoE,train}} \approx 4kN_tH b_a \left(1-\frac{1}{E}\right)
$$

This excludes:

- small routing metadata,
- load-balance loss reductions,
- padding metadata.

Those are second order relative to hidden-state payload.

---

## 10.2 Expert Compute per Rank

For an expert MLP with hidden size $H_{ff}^{exp} = f_{exp}H$, forward FLOPs per local expert token are approximately:

$$
4 H H_{ff}^{exp} = 4 f_{exp} H^2
$$

With $T_e$ expert tensor-parallel sharding, per-rank expert FLOPs are approximately:

$$
F_{\text{expert,fwd}} \approx \frac{4 f_{exp} H^2 \times kN_t}{T_e}
$$

This yields the main MoE compute/communication insight:

- compute scales as $O(H^2)$,
- dispatch communication scales as $O(H)$.

Therefore larger hidden sizes improve compute-to-communication ratio, while very small local expert batches hurt it.

---

## 10.3 When MoE Throughput Collapses

MoE throughput collapses when one or more of the following happen:

1. **All-to-all escapes the fast local fabric**
   - $EP$ stretched across IB / RoCE,
   - dispatch becomes network-bound.

2. **Per-expert batch size becomes too small**
   - grouped GEMM efficiency drops,
   - launch overhead and memory latency dominate.

3. **Routing imbalance widens**
   - some ranks exceed capacity,
   - others idle,
   - p99 step time rises.

4. **Token dropping becomes frequent**
   - effective model capacity changes,
   - training becomes statistically unstable.

---

# 11. Expert Tensor Parallelism

## 11.1 When to Use $T_e$

Use $T_e > 1$ only when:

- individual expert weights are too large,
- expert GEMM dimensions are too large for one rank,
- or expert activation memory is too high.

### Benefit

Persistent expert parameter memory becomes:

$$
M_{\text{expert,param}} \propto \frac{1}{E T_e}
$$

### Cost

Expert-local communication increases and grouped GEMM batching becomes more complex.

---

## 11.2 Expert Parameter and Activation Memory

Per-rank expert parameter memory for one MoE layer is approximately:

$$
M_{\text{expert,param,persist}} \approx \frac{N_{\theta,\text{expert}} b_w}{E T_e D_{\text{expert-shard}}}
$$

where:

- $D_{\text{expert-shard}} = D_r$ under FSDP / ZeRO-$3$ on expert parameters,
- $D_{\text{expert-shard}} = 1$ for replicated expert parameters.

Expert activation memory per rank is approximately:

$$
M_{\text{expert,act}} \approx \frac{N_{\text{tok,local}} H b_a}{T_e}\left(\beta_{\text{in}} + f_{exp}\beta_{\text{mid}}\right) + M_{\text{grouped-gemm-ws}}
$$

At peak, replace $N_{\text{tok,local}}$ with capacity-based upper bound:

$$
N_{\text{tok,local}}^{\max} \approx f_{\text{cap}} k N_t
$$

So:

$$
M_{\text{expert,act}}^{\max} \approx \frac{f_{\text{cap}} k N_t H b_a}{T_e}\left(\beta_{\text{in}} + f_{exp}\beta_{\text{mid}}\right) + M_{\text{grouped-gemm-ws}}
$$

---

# 12. Dense and Sparse Persistent Memory Formulas

## 12.1 Dense Parameters

For pipeline stage $p$, let $\mathcal{L}_p^{dense}$ be its dense layers.

Persistent dense parameter memory per rank is:

$$
M_{\text{dense,param}} = \sum_{l \in \mathcal{L}_p^{dense}} \frac{N_{\theta,l}^{dense} b_w}{T \, D_{\text{dense-shard}}}
$$

where:

- $D_{\text{dense-shard}} = D_{\text{dense}}$ under FSDP / ZeRO-$3$,
- $D_{\text{dense-shard}} = 1$ under replicated parameters.

Dense gradient memory:

$$
M_{\text{dense,grad}} = \sum_{l \in \mathcal{L}_p^{dense}} \frac{N_{\theta,l}^{dense} b_g}{T \, D_{\text{dense-grad-shard}}}
$$

Dense optimizer-state memory:

$$
M_{\text{dense,opt}} = \sum_{l \in \mathcal{L}_p^{dense}} \frac{N_{\theta,l}^{dense} b_o}{T \, D_{\text{dense-opt-shard}}}
$$

---

## 12.2 Expert Parameters

For stage $p$, let $\mathcal{L}_p^{moe}$ be its MoE layers.

Per-rank persistent expert parameter memory is:

$$
M_{\text{expert,param}} = \sum_{l \in \mathcal{L}_p^{moe}} \frac{N_{\theta,l}^{expert} b_w}{E T_e \, D_{\text{expert-shard}}}
$$

Expert gradient memory:

$$
M_{\text{expert,grad}} = \sum_{l \in \mathcal{L}_p^{moe}} \frac{N_{\theta,l}^{expert} b_g}{E T_e \, D_{\text{expert-grad-shard}}}
$$

Expert optimizer-state memory:

$$
M_{\text{expert,opt}} = \sum_{l \in \mathcal{L}_p^{moe}} \frac{N_{\theta,l}^{expert} b_o}{E T_e \, D_{\text{expert-opt-shard}}}
$$

### Important distinction

- For dense weights, the relevant DP sharding degree is often $D_r E$.
- For expert weights, the relevant DP sharding degree is only $D_r$.

This is one of the most important formulas in the entire architecture.

---

## 12.3 Router and Norm Parameters

Router and norm tensors frequently follow different sharding rules from expert weights.

A router is often:

- replicated across $TP$,
- not $EP$-sharded,
- optionally DP/FSDP sharded.

Its memory should be modeled separately:

$$
M_{\text{router}} = \sum_{l \in \mathcal{L}_p^{moe}} \frac{N_{\theta,l}^{router} (b_w+b_g+b_o)}{D_{\text{router-shard}}}
$$

The same applies to norms and other replica-class tensors.

---

# 13. Transient Memory and Peak HBM Condition

## 13.1 FSDP / ZeRO-$3$ Gather Memory

For an FSDP or ZeRO-$3$ bucket $\mathcal{B}$, transient parameter materialization per rank is:

$$
M_{\text{gather}} = \sum_{l \in \mathcal{B}^{dense}} \frac{N_{\theta,l}^{dense} b_w}{T} + \sum_{l \in \mathcal{B}^{expert}} \frac{N_{\theta,l}^{expert} b_w}{E T_e}
$$

This assumes the $DP$ shard dimension is rematerialized while model-parallel sharding remains.

---

## 13.2 Total Peak HBM

The full peak condition is:

$$
M_{\text{peak}} =
M_{\text{dense,param}} +
M_{\text{dense,grad}} +
M_{\text{dense,opt}} +
M_{\text{expert,param}} +
M_{\text{expert,grad}} +
M_{\text{expert,opt}} +
M_{\text{act,dense}} +
M_{\text{act,expert}} +
M_{\text{gather}} +
M_{\text{commbuf}} +
M_{\text{workspace}} +
M_{\text{frag}}
$$

A valid deployment requires:

$$
M_{\text{peak}} \le M_{\max}
$$

on every rank.

### Typical hidden causes of failure

- MoE dispatch buffers omitted from accounting,
- FSDP gather buckets too coarse,
- Ring Attention or Ulysses temporary buffers omitted,
- fragmentation under variable sequence packing,
- expert activation peaks estimated with average rather than capacity.

---

# 14. Communication Stack Composition

## 14.1 Total Layer Communication

A long-context MoE layer may include all of the following:

- dense $TP$ collectives,
- context-parallel communication,
- MoE dispatch and combine,
- optional FSDP parameter all-gather and gradient reduce-scatter,
- stage send/recv for $PP$.

A stage-local layer time should therefore be reasoned as:

$$
T_l = T_l^{compute} + T_l^{TP} + T_l^{CP} + T_l^{EP} + T_l^{FSDP} + T_l^{PP} - T_l^{overlap}
$$

### Key engineering point

The wrong diagnosis is to attribute slowdown to "the MoE layer" or "the long context."  
The correct diagnosis is to identify which communication class is exposed:

- $TP$,
- $CP$,
- $EP$,
- FSDP,
- or $PP$.

---

## 14.2 Collective Cost Model

For a message of size $n$ bytes on group size $g$:

$$
T_m(n) = \alpha + \beta n
$$

For ring-like collectives:

$$
T_{\text{AG}} \approx (g-1)\alpha + \frac{g-1}{g}n\beta
$$

$$
T_{\text{RS}} \approx (g-1)\alpha + \frac{g-1}{g}n\beta
$$

$$
T_{\text{AR}} \approx 2(g-1)\alpha + 2\frac{g-1}{g}n\beta
$$

### Consequence for long-context MoE

- $CP$ and $EP$ are often not pure all-reduce problems.
- All-to-all and streaming ring behavior are more latency-sensitive.
- Bucketing helps FSDP and gradient synchronization, but not all-to-all in the same way.

---

## 14.3 Overlap Priority

Preferred overlap order:

1. overlap FSDP dense parameter gather with independent dense compute,
2. overlap Ring Attention communication with current tile compute,
3. overlap $PP$ send/recv with local kernels,
4. avoid scheduling FSDP gather and remote $EP$ all-to-all on the same slow link simultaneously.

### Critical topology rule

> If $EP$ all-to-all and FSDP gather both cross the same inter-node fabric, throughput will usually collapse before kernels become the bottleneck.

That is why local $EP$ placement is a first-order design requirement.

---

# 15. Long-Context + MoE Architecture Selection Workflow

## 15.1 Planning Sequence

The correct order of decisions is:

1. make dense attention memory-feasible,
2. select $TP$ and $SP$,
3. add $CP$ if sequence still does not fit or attention MFU is poor,
4. place $EP$ locally,
5. add $T_e$ only if expert sizes require it,
6. add $PP$ if stage memory or depth requires it,
7. add FSDP / ZeRO if persistent state still exceeds HBM.

Most failed designs invert this order.

---

## 15.2 Pseudocode: Parallel-Tuple Search

```text
Input:
  model dimensions
  sequence length S
  hidden size H
  number of experts N_e
  target hardware topology
  HBM budget per rank
  candidate tuples (D_r, E, T, P, C, T_e)
  checkpointing policy
  precision policy

Output:
  feasible and ranked training plans

Procedure:
  1. For each candidate tuple:
       a. Verify world-size divisibility.
       b. Verify layer divisibility for P.
       c. Verify head and kernel constraints for T and C.
       d. Verify expert divisibility for E and T_e.

  2. Compute dense effective data parallel degree:
       D_dense = D_r * E

  3. Compute expert replica degree:
       D_expert = D_r

  4. Estimate persistent memory:
       - dense params / grads / optimizer
       - expert params / grads / optimizer
       - router and norm tensors

  5. Estimate activation memory:
       - dense long-context activations
       - attention workspace
       - expert activations using capacity upper bound

  6. Estimate communication:
       - TP collectives
       - CP method cost
       - EP all-to-all
       - FSDP/ZeRO gather and reduce-scatter
       - PP send/recv

  7. Reject any tuple where:
       - peak HBM exceeds budget
       - EP or CP must cross a forbidden topology boundary
       - per-expert batch is below efficiency threshold
       - pipeline bubble is excessive

  8. Rank remaining tuples by:
       - lowest exposed communication
       - highest estimated tokens/s
       - acceptable numerical risk
       - checkpoint portability
```

---

# 16. Data Pipeline Implications for Long Context and MoE

## 16.1 Long-Context Data Requirements

Long-context training changes the data system significantly.

### The pipeline must support

- near-full-length packed sequences,
- exact boundary masks,
- deterministic resume at token offset granularity,
- length-aware bucketing,
- optionally sequence-length curriculum.

If the packing efficiency is $\eta_{\text{pack}}$, effective useful tokens per step are:

$$
N_{\text{useful}} = B_{\text{global}} \times S \times \eta_{\text{pack}}
$$

If $\eta_{\text{pack}}$ is poor, scaling $C$ or adding FlashAttention will not recover the lost useful throughput.

---

## 16.2 MoE Data Effects

MoE routing statistics depend on data mixture.

Different domain mixtures can change:

- token entropy,
- router balance,
- expert specialization,
- overflow rate.

Therefore production MoE training must checkpoint and track:

- routing balance metrics per data slice,
- token-drop rate,
- expert occupancy histograms,
- curriculum state if domain or sequence-length schedules are used.

---

# 17. Fine-Tuning Stability

## 17.1 Long-Context Fine-Tuning

Key long-context fine-tuning risks:

- incorrect RoPE extrapolation configuration,
- insufficient activation checkpointing after increasing $S$,
- packing or masking bugs,
- reduced local tokens causing poor kernel efficiency.

### Recommended controls

- validate perplexity at both original and extended context,
- preserve exact positional-scaling metadata in checkpoint manifests,
- ramp context length gradually if changing $S$ significantly.

---

## 17.2 MoE Fine-Tuning

MoE fine-tuning is more fragile than dense pretraining.

### Main stability risks

- expert collapse,
- router overconfidence,
- aggressive token dropping,
- very small expert batches,
- optimizer mismatch across expert and router parameters.

### Recommended controls

- use deterministic top-$k$,
- keep capacity factor conservative,
- avoid aggressive dropping,
- consider lower router learning rate than expert MLPs,
- track per-expert gradient norms,
- clip globally across dense and expert parameters.

---

# 18. Checkpointing and Determinism

## 18.1 What Must Be Saved

For long-context and MoE training, checkpoint metadata must include:

- dense parameter tensors,
- expert parameter tensors,
- router parameters,
- optimizer states,
- RNG states,
- data cursor state,
- $CP$ algorithm selection,
- RoPE scaling or position-interpolation metadata,
- $EP$ and $ETP$ ownership,
- capacity factor and routing policy,
- expert IDs and local expert mapping.

---

## 18.2 Which Parallel Changes Are Safe

| Change | Usually safe for model-state load | Exact continuation |
|---|---|---|
| change $CP$ | often yes | no |
| change $SP$ | often yes | no |
| change $D$ / FSDP degree | yes with resharding | yes if done correctly |
| change $EP$ | no, requires expert remap | no without conversion |
| change $T_e$ | no, requires expert tensor reshard | no |
| change $P$ | no, requires stage repartition | no |

---

# 19. Platform-Specific Guidance

## 19.1 $A100$

- Prefer moderate $C$.
- Keep $EP$ strictly local.
- Avoid off-node Ring Attention or Ulysses unless unavoidable.
- Use checkpointing aggressively.
- Use conservative $T_e$.

## 19.2 $H100$

- Strongest current platform for long-context + local MoE.
- Larger local $C$ and $E$ are practical.
- FP8 is viable only after full parity qualification.
- Local all-to-all and attention kernels support more aggressive designs.

## 19.3 $B200$-Class

- Recompute optimal $C$, $E$, and bucket sizes from scratch.
- Larger local-group designs become more attractive.
- Do not inherit Hopper-era scheduling assumptions.

## 19.4 $MI300X$

- Exploit HBM before introducing excessive $PP$.
- Keep $CP$ and $EP$ inside xGMI islands.
- Use BF16 baseline first.
- Validate RCCL all-to-all quality before scaling MoE.

## 19.5 $MI350$-Class

- Same placement principles as $MI300X$.
- Requalify FP8 and graph-capture behavior.
- Validate expert grouped GEMM efficiency and xGMI-local dispatcher path.

---

# 20. Failure Modes Explained from First Principles

## 20.1 Throughput Collapse with Larger $S$

If throughput falls sharply as $S$ increases, root causes usually are:

- activation memory forcing smaller $B_\mu$,
- $CP$ communication becoming exposed,
- attention workspace no longer fitting cache/HBM well,
- local token shards too small to sustain kernel occupancy,
- dataloader failing to provide near-full packed sequences.

---

## 20.2 Throughput Collapse with Larger $E$

If throughput falls as $E$ grows, root causes usually are:

- off-node all-to-all,
- expert batch size per local expert becoming too small,
- token imbalance increasing,
- more time spent in dispatch than in expert GEMM.

---

## 20.3 OOM with “Enough HBM on Paper”

This usually happens because the memory model ignored one of:

- FSDP gather buckets,
- attention temporary buffers,
- expert dispatch buffers,
- capacity-based local expert activation peaks,
- fragmentation,
- pipeline activation overlap.

---

## 20.4 Loss Instability After Enabling MoE

Typical causes:

- router tie-breaking nondeterminism,
- aggressive token dropping,
- incorrect expert optimizer-state mapping,
- router LR too high,
- different clipping or precision path for expert parameters.

---

# 21. Pseudocode: Capacity-Safe MoE Routing

```text
Input:
  local token activations
  router logits
  top-k value k
  total experts N_e
  expert-parallel degree E
  capacity factor f_cap
  deterministic tie-break rule

Output:
  dispatch plan
  local expert token buffers
  overflow statistics

Procedure:
  1. Select top-k experts per token with deterministic tie-breaking.
  2. Count expected per-expert token load.
  3. Compute expert capacity:
       cap = ceil(f_cap * k * E * N_t / N_e)
  4. For each expert:
       - accept tokens up to cap
       - handle overflow using configured policy:
         a. drop
         b. reroute
         c. dropless with larger buffers
  5. Stable-sort accepted tokens by target expert and local position.
  6. Build all-to-all send and receive metadata.
  7. Record:
       - per-expert counts
       - overflow rate
       - imbalance metrics
```

---

# 22. Pseudocode: Long-Context Attention Method Selection

```text
Input:
  sequence length S
  hidden size H
  head count n_h
  context degree candidates C
  topology bandwidth model
  memory budget
  kernel availability

Output:
  chosen attention strategy

Procedure:
  1. Estimate dense activation memory with FlashAttention and checkpointing.
  2. If model fits without CP:
       choose standard attention without CP.
  3. Else evaluate CP variants:
       - standard CP
       - Ulysses
       - Ring Attention
  4. For each variant:
       - estimate per-layer communication
       - estimate local token shard size
       - estimate kernel occupancy risk
       - reject if required fabric crosses slow topology domain
  5. Choose the variant with:
       - feasible HBM
       - lowest exposed attention communication
       - acceptable kernel efficiency
```

---

# 23. Summary of Exact Core Formulas

## 23.1 World Size

Dense:

$$
W = D \times T \times P \times C
$$

MoE practical production form:

$$
W = D_r \times E \times T \times P \times C \times T_e
$$

---

## 23.2 Effective Replica Degrees

Dense effective data-parallel degree in MoE layout:

$$
D_{\text{dense}} = D_r E
$$

Expert replica degree:

$$
D_{\text{expert}} = D_r
$$

---

## 23.3 Global Batch

$$
B_{\text{global}} = B_\mu \times m \times D_{\text{dense}}
$$

---

## 23.4 Dense Activation Memory

$$
M_{\text{act,dense-block}} \approx \frac{B_\mu S H b_a}{C} \left(\alpha_{\text{rep}} + \frac{\alpha_{\text{shard}}}{T}\right) + M_{\text{attn-ws}}
$$

---

## 23.5 Naive Score Memory

$$
M_{\text{scores,naive}} \approx B_\mu \times \frac{n_h}{T} \times \frac{S}{C} \times S \times b_a
$$

---

## 23.6 Ulysses Communication

$$
V_{\text{Ulysses,train}} \approx 8 B_\mu S H b_a \left(1-\frac{1}{C}\right)
$$

---

## 23.7 Ring Attention Communication

$$
V_{\text{Ring,train}} \approx 4(C-1) \times B_\mu \times \frac{S}{C} \times H \times b_a
$$

---

## 23.8 Expected Expert Load

$$
\mathbb{E}[n_{\text{expert}}] = \frac{k E N_t}{N_e}
$$

$$
\mathbb{E}[n_{\text{rank}}] = kN_t
$$

---

## 23.9 Capacity

$$
\text{cap} = \left\lceil f_{\text{cap}} \times \frac{k E N_t}{N_e} \right\rceil
$$

$$
N_{\text{tok,local}}^{\max} \approx f_{\text{cap}} k N_t
$$

---

## 23.10 MoE Communication

$$
V_{\text{MoE,train}} \approx 4kN_tH b_a \left(1-\frac{1}{E}\right)
$$

---

## 23.11 Expert Activation Memory

$$
M_{\text{expert,act}}^{\max} \approx \frac{f_{\text{cap}} k N_t H b_a}{T_e}\left(\beta_{\text{in}} + f_{exp}\beta_{\text{mid}}\right) + M_{\text{grouped-gemm-ws}}
$$

---

## 23.12 Peak Memory Condition

$$
M_{\text{peak}} \le M_{\max}
$$

with:

$$
M_{\text{peak}} =
M_{\text{persistent}} +
M_{\text{act,dense}} +
M_{\text{act,expert}} +
M_{\text{gather}} +
M_{\text{commbuf}} +
M_{\text{workspace}} +
M_{\text{frag}}
$$

---

# 24. Final Engineering Conclusions

1. **The correct practical MoE factorization distinguishes dense replica degree from expert replica degree.**  
   This changes optimizer sharding, checkpoint semantics, and memory formulas.

2. **Long-context scaling is primarily an activation and attention-communication problem.**  
   The right progression is:
   - FlashAttention,
   - checkpointing,
   - $SP$,
   - $CP$,
   - then deeper $PP$ or offload only if still necessary.

3. **MoE scaling is primarily a token-dispatch and local expert batch-size problem.**  
   Total local expert tokens per rank remain roughly $kN_t$ under balanced routing; per-expert batch size is what collapses if expert count is poorly chosen.

4. **$EP$ must remain on the fastest local fabric whenever possible.**  
   Off-node expert all-to-all is one of the fastest ways to destroy throughput.

5. **$CP$ and $EP$ are not interchangeable.**  
   $CP$ partitions activations and attention work; $EP$ changes parameter ownership and routing semantics.

6. **Any credible long-context MoE deployment plan must explicitly account for:**
   - dense-path memory,
   - expert-path memory,
   - dispatch buffers,
   - FSDP gather windows,
   - topology-aware communication exposure,
   - pipeline imbalance,
   - routing imbalance.

---

# 25. Reference Links

## Core Training Frameworks
- Megatron-LM / Megatron-Core: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed: https://www.deepspeed.ai/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html

## Attention and Kernel Optimization
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Triton: https://github.com/triton-lang/triton
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine

## Communication and Profiling
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- RCCL: https://rocm.docs.amd.com/projects/rccl/en/latest/
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- ROCm profiling tools: https://rocm.docs.amd.com/

the next deliverable should be a **worked design workbook** with concrete parallel tuples and memory budgets for:

- dense long-context models at $16K$, $32K$, $64K$, and $128K$,
- MoE models such as $8 \times 7B$ and $16 \times 7B$,
- platform-specific deployments on $A100$, $H100$, $B200$-class, $MI300X$, and $MI350$ clusters.