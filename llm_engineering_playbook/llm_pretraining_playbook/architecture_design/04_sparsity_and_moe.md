# Sparse Mixture-of-Experts Architectures, Hybrid Attention Mechanisms, and Base Architecture Selection for Large Language Models: A Comprehensive Technical Report
![04 sparsity and moe](./assets/04_sparsity_and_moe.png)
---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Mixture-of-Experts: Architectural Foundations](#2-mixture-of-experts-architectural-foundations)
3. [Efficiency Leverage: Quantifying MoE Compute Advantage](#3-efficiency-leverage)
4. [Sparsity and Activation Ratio Analysis](#4-sparsity-and-activation-ratio)
5. [Expert Granularity](#5-expert-granularity)
6. [Shared Expert Mechanisms](#6-shared-expert-mechanisms)
7. [Load Balancing in MoE Training and Inference](#7-load-balancing)
8. [Hybrid Architectures: Integrating State Space Models and Linear Attention](#8-hybrid-architectures)
9. [Advanced Linear Attention: A Unified Gating Framework](#9-advanced-linear-attention)
10. [Base Architecture Selection: Dense vs. MoE vs. Hybrid](#10-base-architecture-selection)
11. [Open Research Directions](#11-open-research-directions)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction and Motivation

Modern large language models (LLMs) have demonstrated remarkable capabilities across natural language understanding, generation, translation, code synthesis, and multimodal reasoning. However, the dominant dense transformer paradigm—where every parameter is activated for every input token—imposes a strict coupling between **model capacity** (total parameter count) and **computational cost** (floating-point operations per forward pass). As frontier models scale into hundreds of billions or trillions of parameters, this coupling renders both training and inference prohibitively expensive under fixed compute budgets.

The **Mixture-of-Experts (MoE)** paradigm directly addresses this bottleneck by decoupling total model capacity from per-token computational expenditure. The foundational intuition mirrors a well-established neuroscientific principle: the human brain does not uniformly activate all cortical regions for every cognitive task. The visual cortex activates during visual processing; Broca's area engages during language production; the motor cortex governs movement planning. Analogously, in an LLM, the subnetworks that have learned coding syntax need not activate when the model performs a natural language translation task.

If this selective activation can be implemented effectively, the computational savings are substantial: only a fraction of the total model parameters execute during each forward pass, while the full parameter set contributes to the model's aggregate learning capacity across the training distribution.

This report provides a rigorous, end-to-end technical treatment of:

- **MoE architectural design**, including sparsity configuration, expert granularity, shared expert mechanisms, and load balancing strategies
- **Hybrid architectures** that augment transformer attention with state space models (SSMs) and linear attention mechanisms to address quadratic complexity in long-context processing
- **Architecture selection criteria** for choosing between dense, MoE, and hybrid configurations under practical deployment, expertise, and timeline constraints

Throughout, we ground the analysis in recent empirical findings from frontier model releases including DeepSeek-V3, Kimi K2, Qwen3, GLM-4.5, and the Ant Group scaling laws study (Tian et al., 2025).

---

## 2. Mixture-of-Experts: Architectural Foundations

### 2.1 From Dense to Sparse: The Core Transformation

A standard dense transformer layer consists of two primary submodules applied sequentially to each token representation:

1. **Multi-Head Self-Attention (MHSA):** Computes contextual token representations through scaled dot-product attention across multiple heads.
2. **Feed-Forward Network (FFN) / Multilayer Perceptron (MLP):** Applies a position-wise nonlinear transformation, typically structured as:

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

where $W_1 \in \mathbb{R}^{d_{\text{intermediate}} \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{intermediate}}}$, and $\sigma(\cdot)$ denotes a nonlinear activation function (e.g., SiLU, GELU).

The MoE transformation **replaces the single FFN with $N$ parallel FFN modules** (termed "experts") and introduces a **learnable routing mechanism** (the "router" or "gating network") that selects, for each token, a small subset of $k$ experts to execute. Formally:

$$
\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

where:
- $E_i(\cdot)$ denotes the $i$-th expert network (an independent FFN)
- $g_i(x)$ is the gating weight assigned to expert $i$ for input token $x$
- The gating function enforces **sparsity**: for each token, only $k \ll N$ experts receive non-zero gate values

The gating weights are typically computed via a learned linear projection followed by a top-$k$ selection and softmax normalization:

$$
g(x) = \text{TopK}\!\left(\text{Softmax}(W_g x), k\right)
$$

where $W_g \in \mathbb{R}^{N \times d_{\text{model}}}$ is the router's weight matrix. The $\text{TopK}$ operation zeros out all but the $k$ highest-scoring entries, and the surviving values are renormalized.

### 2.2 Total Parameters vs. Active Parameters

This architectural transformation introduces a critical distinction:

| Quantity | Definition | Impact |
|----------|-----------|--------|
| **Total Parameters** ($P_{\text{total}}$) | Sum of all parameters across all $N$ experts plus the shared layers (attention, embeddings, router) | Determines the model's aggregate **learning capacity** across the full training distribution |
| **Active Parameters** ($P_{\text{active}}$) | Parameters that execute for a given token, i.e., the $k$ selected experts plus shared layers | Determines the per-token **training cost** and **inference latency** |

The fundamental value proposition of MoE is:

$$
P_{\text{active}} \ll P_{\text{total}}
$$

This enables models with massive total capacity (e.g., Kimi K2: $\sim$1T total parameters) to operate at the computational cost of a much smaller dense model (e.g., $\sim$32B active parameters), achieving superior loss at equivalent FLOPs budgets.

### 2.3 Core Design Questions

Designing an effective MoE layer requires resolving several interdependent design axes:

1. **Expert Shape and Sparsity:** Should the architecture employ many small experts or fewer large ones? How many experts should be active per token ($k$), and how many total experts ($N$) are required? Should certain experts be universally active (shared experts)?

2. **Utilization and Specialization:** How should the router select experts to ensure balanced utilization (avoiding idle capacity and routing collapse) while simultaneously encouraging expert specialization? This constitutes a load-balancing problem with direct implications for training stability, parallelization efficiency, and model quality.

3. **Compute-Optimal Configuration:** Given a fixed compute budget (measured in FLOPs), what MoE configuration (sparsity, granularity, shared experts, load balancing) minimizes the training loss?

The following sections address each of these design axes systematically.

---

## 3. Efficiency Leverage: Quantifying MoE Compute Advantage

### 3.1 Definition

To rigorously compare MoE and dense architectures under equivalent computational budgets, we adopt the **Efficiency Leverage (EL)** metric introduced by Tian et al. (2025) in the Ant Group MoE scaling laws study.

**Efficiency Leverage** quantifies the ratio of dense compute required to match the loss achieved by an MoE configuration:

$$
\text{EL}\!\left(\mathcal{X}_{\text{MoE}} \mid \mathcal{X}_{\text{Dense}}\right) = \frac{C_{\text{Dense}}}{C_{\text{MoE}}}
$$

where:
- $C_{\text{MoE}}$ is the compute (in FLOPs) used to train the MoE model to loss $\mathcal{L}$
- $C_{\text{Dense}}$ is the compute a dense model would require to reach the same loss $\mathcal{L}$, as predicted by the dense scaling law

**Interpretation:**
- $\text{EL} > 1$: The MoE delivers more loss reduction per FLOP than the dense baseline
- Higher EL $\Rightarrow$ greater compute efficiency of the MoE configuration
- $\text{EL} = 1$: No advantage over dense training

### 3.2 Scaling Law Comparison

Empirical scaling law curves (e.g., from the Ling 1.5 paper, L. Team et al., 2025) demonstrate that MoE models consistently achieve lower loss than dense models at matched FLOPs budgets, with the gap widening as compute increases. Concretely, for a given FLOP budget $C$, the MoE loss curve lies below the dense loss curve:

$$
\mathcal{L}_{\text{MoE}}(C) < \mathcal{L}_{\text{Dense}}(C) \quad \forall \; C > C_{\text{threshold}}
$$

This empirical finding motivates the widespread adoption of MoE in frontier systems, including DeepSeek-V3, Kimi K2, Gemini, and Grok.

---

## 4. Sparsity and Activation Ratio Analysis

### 4.1 Formal Definitions

Two reciprocal quantities characterize the degree of expert selection:

**Activation Ratio:**

$$
\text{Activation Ratio} = \frac{\#\text{activated experts}}{\#\text{total experts}} = \frac{k}{N}
$$

**Sparsity:**

$$
\text{Sparsity} = \frac{\#\text{total experts}}{\#\text{activated experts}} = \frac{N}{k} = \frac{1}{\text{Activation Ratio}}
$$

A sparsity of 1 corresponds to the dense regime (all experts active); higher sparsity values indicate increasingly selective routing.

### 4.2 Compute Implications

From a computational perspective, the cost of the MoE FFN layer is governed exclusively by the **active parameters**. If the number and size of activated experts remain fixed while the total number of experts increases:

- **Training/inference FLOPs per token remain approximately constant** (only the router computation scales marginally with $N$)
- **Model capacity increases** due to the additional expert parameters available across the training distribution
- **Performance generally improves**, provided sufficient training data to utilize the expanded capacity

### 4.3 Asymptotic Behavior and the Sparsity Sweet Spot

The two extremes of the sparsity spectrum are both suboptimal:

- **Sparsity $= 1$ (all experts active):** Degenerates to the dense regime; no compute savings are realized, and the router adds unnecessary overhead.
- **Sparsity $\to \infty$ (vanishingly few active parameters):** The active capacity becomes insufficient to represent even narrow-domain computations; the model cannot learn meaningful representations.

Empirical evidence from multiple frontier laboratories confirms a characteristic pattern:

> **Holding the number and size of active experts fixed, increasing total experts (i.e., increasing sparsity) improves loss with diminishing returns at very high sparsity.**

Key empirical findings:

1. **Kimi K2 (K. Team et al., 2025):** Demonstrates both effects—higher sparsity improves performance, but gains taper off as sparsity grows.
2. **Ant Group (Tian et al., 2025):** Confirms the same conclusion and further shows that higher-sparsity MoE configurations benefit disproportionately from increased compute budgets—i.e., the efficiency leverage of high-sparsity MoE grows with scale.

### 4.4 Sparsity Configurations of Frontier MoE Models

The following table documents the sparsity configurations of representative frontier MoE systems, illustrating the clear trend toward higher sparsity:

| Model | Total Experts | Activated per Token (Incl. Shared) | Sparsity |
|-------|---------------|-------------------------------------|----------|
| Mixtral-8×7B | 8 | 2 | 4.0 |
| Grok-1 | 8 | 2 | 4.0 |
| Grok-2 | 8 | 2 | 4.0 |
| OLMoE-1B-7B-0924 | 64 | 8 | 8.0 |
| gpt-oss 20b | 32 | 4 | 8.0 |
| Step-3 | 48 routed + 1 shared = 49 | 3 routed + 1 shared = 4 | 12.25 |
| GLM-4.5-Air | 128 routed + 1 shared = 129 | 8 routed + 1 shared = 9 | 14.3 |
| Qwen3-30B-A3B | 128 | 8 | 16.0 |
| Qwen3-235B-A22B | 128 | 8 | 16.0 |
| GLM-4.5 | 160 routed + 1 shared = 161 | 8 routed + 1 shared = 9 | 17.8 |
| DeepSeek-V2 | 160 routed + 2 shared = 162 | 6 routed + 2 shared = 8 | 20.25 |
| DeepSeek-V3 | 256 routed + 1 shared = 257 | 8 routed + 1 shared = 9 | 28.6 |
| gpt-oss 120b | 128 | 4 | 32.0 |
| Kimi K2 | 384 routed + 1 shared = 385 | 8 routed + 1 shared = 9 | 42.8 |
| Qwen3-Next-80B-A3B | 512 routed + 1 shared = 513 | 10 active + 1 shared = 11 | 46.6 |

### 4.5 Key Observations

1. **Secular Trend:** The field is moving decisively toward higher sparsity—from $\sim$4× in early MoE models (Mixtral, Grok-1) to $\sim$42–47× in the most recent releases (Kimi K2, Qwen3-Next).
2. **Hardware-Aware Deviations:** The optimal sparsity for a given deployment is not purely a function of loss; it depends on hardware topology, memory bandwidth, and communication constraints. For instance:
   - **Step-3** intentionally limits sparsity to 12.25 to maximize end-to-end throughput on its target hardware, optimizing for inter-node bandwidth and expert placement efficiency.
   - **gpt-oss-20b** uses a low sparsity of 8.0 due to on-device memory constraints, since even inactive (passive) experts consume memory.
3. **Diminishing Returns:** While increasing sparsity consistently improves loss, the marginal gain diminishes at very high sparsity values, suggesting an asymptotic bound on the benefit of additional capacity without proportional active compute.

---

## 5. Expert Granularity

### 5.1 Definition

Beyond sparsity, a critical design parameter is the **size of each individual expert**, captured by the metric **granularity** ($G$), introduced by Tian et al. (2025):

$$
G = \frac{\alpha \cdot d_{\text{model}}}{d_{\text{expert}}}
$$

where:
- $d_{\text{model}}$ is the model hidden dimension
- $d_{\text{expert}}$ is the intermediate (hidden) dimension of each expert's FFN
- $\alpha$ is a normalization constant, typically $\alpha = 2$ or $\alpha = 4$

**Interpretation:** A higher granularity value corresponds to **more experts with smaller individual dimensions**, given a fixed total parameter budget. Conversely, $G = 1$ would indicate that each expert has the same width as the standard dense FFN.

### 5.2 Relationship to Dense FFN Scaling

In dense transformer architectures, a common design heuristic sets the intermediate FFN dimension to:

$$
d_{\text{intermediate}} = 4 \cdot d_{\text{model}}
$$

Under the convention $\alpha = 4$ (following Krajewski et al., 2024), the granularity can be interpreted as the number of experts required to collectively match the dense MLP width:

$$
4 \cdot d_{\text{model}} = d_{\text{intermediate}} = G \cdot d_{\text{expert}}
$$

This interpretation is heuristic: modern MoE designs typically allocate substantially more total capacity than a single dense MLP, so the one-to-one correspondence breaks down in practice. The Ant Group team employs $\alpha = 2$, which represents a different normalization convention. For consistency throughout this report, we adopt $\alpha = 2$.

### 5.3 Cross-Model Granularity Survey

Because $G$ scales with $d_{\text{model}}$, cross-model comparisons require caution when model widths differ. The following table documents granularity values for recent MoE releases:

| Model | $d_{\text{model}}$ | $d_{\text{expert}}$ | $G = 2d_{\text{model}}/d_{\text{expert}}$ | Year |
|-------|---------------------|----------------------|-------------------------------------------|------|
| Mixtral-8x7B | 4,096 | 14,336 | 0.571 | 2023 |
| gpt-oss-120b | 2,880 | 2,880 | 2.0 | 2025 |
| gpt-oss-20b | 2,880 | 2,880 | 2.0 | 2025 |
| Grok 2 | 8,192 | 16,384 | 1.0 | 2024 |
| Step-3 | 7,168 | 5,120 | 2.8 | 2025 |
| OLMoE-1B-7B | 2,048 | 1,024 | 4.0 | 2025 |
| Qwen3-30B-A3B | 2,048 | 768 | 5.3 | 2025 |
| Qwen3-235B-A22B | 4,096 | 1,536 | 5.3 | 2025 |
| GLM-4.5-Air | 4,096 | 1,408 | 5.8 | 2025 |
| DeepSeek-V2 | 5,120 | 1,536 | 6.6 | 2024 |
| GLM-4.5 | 5,120 | 1,536 | 6.6 | 2025 |
| Kimi K2 | 7,168 | 2,048 | 7.0 | 2025 |
| DeepSeek-V3 | 7,168 | 2,048 | 7.0 | 2024 |
| Qwen3-Next-80B-A3B | 2,048 | 512 | 8.0 | 2025 |

### 5.4 Empirical Findings on Granularity

Analysis from the Ant Group scaling study reveals the following:

1. **Granularity is not the primary driver of Efficiency Leverage.** While increasing $G$ improves EL—particularly for values $G > 2$—the effect is secondary compared to sparsity.
2. **Diminishing returns beyond a sweet spot.** Increasing granularity helps up to a point, after which gains flatten. Excessively fine-grained experts may lack sufficient individual capacity to perform meaningful computation.
3. **Clear secular trend.** Recent releases exhibit a monotonic increase in granularity, from $G \approx 0.57$ (Mixtral, 2023) to $G = 8.0$ (Qwen3-Next, 2025), reflecting the field's empirical convergence on finer-grained expert decomposition.
4. **Not to be optimized in isolation.** Granularity interacts with sparsity, shared expert configuration, and load balancing—joint optimization is necessary.

---

## 6. Shared Expert Mechanisms

### 6.1 Concept and Motivation

A **shared expert** is an expert network through which **every token is routed**, regardless of the router's top-$k$ selection. Formally, if $E_{\text{shared}}$ denotes the shared expert and $\{E_{i_1}, \ldots, E_{i_k}\}$ denote the top-$k$ routed experts selected for token $x$, the MoE output becomes:

$$
\text{MoE}(x) = E_{\text{shared}}(x) + \sum_{j=1}^{k} g_{i_j}(x) \cdot E_{i_j}(x)
$$

The rationale is functional decomposition:
- **Shared experts absorb ubiquitous, domain-agnostic patterns** (e.g., syntactic structure, common collocations, formatting conventions) that recur across all input domains
- **Routed experts specialize more aggressively** on domain-specific or task-specific representations, as the shared expert has already captured the baseline patterns

### 6.2 Practical Configuration

Empirical evidence and frontier model configurations converge on a simple heuristic:

> **Use one shared expert.** This choice maximizes efficiency without introducing unnecessary complexity.

This configuration is adopted by DeepSeek-V3 (1 shared), Kimi K2 (1 shared), Qwen3-Next (1 shared), GLM-4.5 (1 shared), and Step-3 (1 shared). DeepSeek-V2 uses 2 shared experts, but this remains an exception.

### 6.3 Interaction with Granularity

Analysis from Tian et al. (2025) reveals an important interaction effect: **shared experts become more beneficial as granularity increases.** As individual routed experts become smaller (higher $G$), their individual capacity decreases, making the always-on shared expert's role in capturing baseline representations more important.

### 6.4 Impact on Efficiency Leverage

The overall impact of shared experts on EL is **modest**—shared experts do not dramatically change the efficiency leverage curve. They provide a reliable, low-risk improvement that is easy to implement, but they are not a substitute for proper sparsity and granularity configuration.

---

## 7. Load Balancing in MoE Training and Inference

### 7.1 The Load Balancing Problem

Load balancing is the **most critical operational challenge** in MoE design. If poorly configured, it can undermine all other architectural decisions. The failure mode is **routing collapse**: the router learns to concentrate all or most tokens on a small subset of experts, leaving the remaining experts idle.

**Why routing collapse is catastrophic:**

Consider a distributed training setup with 4 GPUs, each hosting one expert. If the router collapses and routes all tokens to Expert 1:

1. **Compute utilization drops to 25%:** Three of four GPUs sit idle, wasting 75% of the available compute.
2. **Effective model capacity collapses:** The model degenerates to a single-expert (approximately dense) configuration, negating the entire purpose of the MoE architecture.
3. **Training throughput degrades:** The overloaded GPU becomes a bottleneck, serializing what should be parallel computation.
4. **Expert specialization fails:** Underutilized experts receive insufficient gradient signal to develop meaningful specializations.

### 7.2 Auxiliary Loss–Based Load Balancing

The standard approach to preventing routing collapse is to augment the primary language modeling loss with an **auxiliary load-balancing loss** that penalizes uneven expert utilization:

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i \cdot P_i
$$

where:
- $\alpha \in \mathbb{R}^+$ is the **balancing coefficient**, controlling the strength of the auxiliary loss relative to the primary language modeling loss
- $N_r$ is the number of routed experts
- $f_i$ is the **traffic fraction**: the proportion of tokens in the batch routed to expert $i$

$$
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\left[i \in \text{TopK}(g(x_t))\right]
$$

- $P_i$ is the **probability mass**: the average router probability assigned to expert $i$ across all tokens in the batch

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} p_i(x_t)
$$

where $p_i(x_t) = \text{Softmax}(W_g x_t)_i$ is the softmax probability of expert $i$ for token $x_t$.

**Why both $f_i$ and $P_i$ are necessary:**
- $f_i$ captures the **actual routing decisions** (discrete, non-differentiable due to the top-$k$ operation)
- $P_i$ provides a **smooth, differentiable proxy** that enables gradient flow through the router parameters
- Their product $f_i \cdot P_i$ is minimized when both are uniform, i.e., $f_i = P_i = 1/N_r$ for all $i$

**Sensitivity of $\alpha$:**
- $\alpha$ too small $\Rightarrow$ insufficient routing guidance; collapse remains possible
- $\alpha$ too large $\Rightarrow$ routing uniformity dominates the loss landscape, degrading language modeling performance as the router is forced into suboptimal uniform assignments regardless of input semantics

### 7.3 Loss-Free Load Balancing

DeepSeek-V3 (DeepSeek-AI et al., 2025) introduced a **loss-free** alternative that achieves load balancing without an explicit auxiliary loss term. The mechanism operates by adding an adaptive **bias term** $b_i$ to the affinity scores before the routing softmax:

$$
\tilde{s}_i(x) = s_i(x) + b_i
$$

where $s_i(x) = (W_g x)_i$ is the raw affinity score. The bias $b_i$ is updated via a simple adaptive rule:

- If expert $i$ is **overloaded** (receiving more than its fair share of tokens): decrease $b_i$ by a constant factor $\gamma$
- If expert $i$ is **underutilized**: increase $b_i$ by $\gamma$

This mechanism acts as a **negative feedback controller** that nudges the routing distribution toward uniformity without introducing any additional term into the training loss. The advantages are:

1. **No interference with the primary loss landscape:** The language modeling gradient is not contaminated by balancing gradients
2. **Simplicity:** A single hyperparameter $\gamma$ replaces the complex tuning of $\alpha$
3. **Adaptive:** The bias adjusts dynamically throughout training as expert utilization patterns evolve

### 7.4 Scope of Routing Statistics: Local vs. Global Aggregation

A critical implementation detail with substantial impact on both expert specialization and model quality is the **scope** at which routing statistics ($f_i$ and $P_i$) are computed.

**Local computation (per micro-batch per worker):**
- Each worker/device computes $f_i$ and $P_i$ over its own local mini-batch
- Computationally simple; no inter-device communication required

**Global computation (aggregated across all workers/devices):**
- $f_i$ and $P_i$ are computed over the full global batch by aggregating statistics across all workers
- Requires inter-device communication (e.g., all-reduce operations)

**Empirical findings from Qwen (Qiu et al., 2025):**

The Qwen team's analysis demonstrates that **local computation can significantly harm both expert specialization and overall model performance** when token diversity within each local batch is insufficient. The mechanism is as follows:

1. When local batches are narrow (e.g., dominated by a single domain or document), the routing statistics become **noisy and biased**
2. The resulting load-balancing signal does not reflect the global routing distribution, leading to **suboptimal balancing decisions**
3. Expert specialization—the phenomenon where specific experts are preferentially activated for specific domains—is degraded because the local signal cannot distinguish meaningful specialization from statistical noise

**Expert specialization** is defined as the degree to which individual experts are disproportionately activated for specific input domains (e.g., code, mathematics, multilingual text). It serves as a reliable proxy for routing health.

**Recommendation:** Use global statistics (or at minimum, cross-device aggregation) whenever the communication overhead is feasible. Notably, at the time of the Qwen analysis, many widely-used training frameworks—including Megatron-LM—computed these statistics locally by default, potentially degrading MoE training quality in an undiagnosed manner.

### 7.5 Interdependence with Other Design Choices

Load balancing interacts with multiple other MoE design axes:

- **Granularity:** Higher granularity (more, smaller experts) may require more aggressive load balancing, as the router has more choices and the risk of utilization imbalance increases
- **Shared experts:** The presence of shared experts can partially mitigate the impact of imbalanced routing, as baseline functionality is preserved regardless of routing decisions
- **Sparsity:** Higher sparsity (lower activation ratio) makes load balancing more critical, as each expert sees fewer tokens and is more sensitive to routing noise

This interdependence underscores the importance of **joint ablation studies** rather than isolated optimization of individual design parameters.

---

## 8. Hybrid Architectures: Integrating State Space Models and Linear Attention

### 8.1 Motivation: The Long-Context Bottleneck

Standard softmax self-attention computes pairwise interactions between all tokens, incurring computational and memory costs that scale quadratically with sequence length:

$$
\text{Attention Complexity} = O(n^2 d)
$$

where $n$ is the sequence length and $d$ is the head dimension. For very long contexts (e.g., $n > 100\text{K}$ tokens), this quadratic scaling becomes a dominant bottleneck for both training throughput and inference latency.

**Hybrid architectures** address this by combining standard transformer blocks with **state space models (SSMs)** or **linear attention mechanisms** (MiniMax et al., 2025; Zuo et al., 2025). These alternative mechanisms occupy a middle ground between:

- **Recurrent models:** Process arbitrarily long contexts with $O(n)$ scaling but may struggle to fully leverage long-range contextual dependencies
- **Transformers:** Excel at leveraging contextual patterns but incur $O(n^2)$ cost

By interleaving both types of layers, hybrid architectures achieve the **pattern-matching strengths of attention** where needed while delegating sequence-level processing to **computationally cheaper linear mechanisms**.

### 8.2 From Softmax Attention to Linear Attention: Mathematical Derivation

#### 8.2.1 Standard Softmax Attention

At inference, producing the output for token $t$ involves:

$$
o_t = \frac{\sum_{j=1}^{t} \exp(q_t^\top k_j) \cdot v_j}{\sum_{l=1}^{t} \exp(q_t^\top k_l)}
$$

where $q_t$, $k_j$, $v_j \in \mathbb{R}^d$ are the query, key, and value vectors, respectively. The softmax normalization ensures that the attention weights form a valid probability distribution over the context.

#### 8.2.2 Removing the Softmax

Dropping the softmax normalization yields the unnormalized linear attention:

$$
o_t = \sum_{j=1}^{t} (q_t^\top k_j) \cdot v_j
$$

#### 8.2.3 The Critical Reordering

The key algebraic insight that enables the computational speedup is the **associativity of matrix multiplication**, which permits reordering the summation:

$$
\sum_{j=1}^{t} (q_t^\top k_j) \cdot v_j = \left(\sum_{j=1}^{t} v_j k_j^\top \right) q_t
$$

**Left form (token-centric):** For each past token $j$, compute the scalar dot product $q_t^\top k_j$, use it to scale $v_j$, and accumulate $t$ scaled vectors. Cost per step: $O(td)$.

**Right form (state-centric):** Maintain a single running **state matrix** that summarizes all past key-value information, and multiply by the current query. Cost per step: $O(d^2)$.

#### 8.2.4 The Recurrent State Formulation

Define the running state:

$$
S_t \triangleq \sum_{j=1}^{t} k_j v_j^\top = K_{1:t}^\top V_{1:t} \in \mathbb{R}^{d \times d}
$$

with the recurrent update:

$$
S_t = S_{t-1} + k_t v_t^\top
$$

The output at step $t$ becomes:

$$
o_t = S_t q_t = S_{t-1} q_t + v_t (k_t^\top q_t)
$$

#### 8.2.5 Complexity Analysis

| Form | Per-Step Cost | Total Cost for $T$ Tokens |
|------|--------------|---------------------------|
| Left (token-centric, softmax attention) | $O(td)$ | $O(T^2 d)$ |
| Right (state-centric, linear attention) | $O(d^2)$ | $O(Td^2)$ |

The transition from $O(T^2 d)$ to $O(Td^2)$ **trades dependence on sequence length for dependence on model dimension**, which is a favorable exchange when $T \gg d$, precisely the long-context regime.

#### 8.2.6 Training-Time Efficiency

The reordering applies equally during training. For the full sequence computation:

$$
\underbrace{(QK^\top)}_{n \times n} V = Q \underbrace{(K^\top V)}_{d \times d}
$$

The left form requires materializing the $n \times n$ attention matrix; the right form computes a compact $d \times d$ matrix, reducing both memory and compute by a factor proportional to $n/d$.

### 8.3 Lightning Attention

While the mathematical reordering is elegant, the removal of softmax introduces **numerical instability**: the softmax in standard attention plays a crucial stabilizing role by bounding the attention weights and ensuring they form a valid distribution. Naive linear attention can exhibit unbounded output magnitudes and training instability.

**Lightning Attention** (building on the NormAttention framework of Qin et al., 2022) addresses this by replacing softmax normalization with **norm-based scaling** and incorporating architectural stabilization mechanisms:

**Step 1: QKV Projection with SiLU Activation**

$$
Q = \text{SiLU}(Q), \quad K = \text{SiLU}(K), \quad V = \text{SiLU}(V), \quad G = \sigma(G)
$$

The SiLU (Sigmoid Linear Unit) activation ensures non-negative, smoothly bounded projections. $G$ is a learned gating vector with sigmoid activation $\sigma(\cdot)$.

**Step 2: Recurrent State Update with Decay**

$$
KV_t = \lambda \cdot KV_{t-1} + k_t^\top v_t
$$

$$
o_t = q_t \cdot KV_t
$$

where $\lambda \in (0, 1)$ is a **decay factor** that allows the state to geometrically discount older information, preventing unbounded state growth and introducing an implicit forgetting mechanism analogous to gated recurrent architectures.

**Step 3: RMSNorm with Output Gating**

$$
Y = G \odot \text{RMSNorm}(O)
$$

RMSNorm stabilizes the output magnitude, while the element-wise gating $G$ provides a learned modulation of the output, enabling the layer to adaptively scale its contribution.

### 8.4 Empirical Performance of Hybrid Models

Empirical evaluation from MiniMax et al. (2025) at multiple model scales (410M, 1B, 3B, 7B parameters) reveals:

1. **Common-sense reasoning benchmarks (PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OBQA):** Hybrid-Lightning models match or closely approximate softmax attention performance across all scales.
2. **Retrieval tasks (Needle-In-A-Haystack / NIAH):** Hybrid models demonstrate **substantially superior performance** compared to pure softmax attention—an unexpected finding suggesting synergistic interaction between the softmax and linear attention components.
3. **SCR (Synthetic Context Reasoning):** Performance remains comparable across architectures at all scales.

The NIAH result is particularly noteworthy: it suggests that the linear attention component may provide complementary retrieval capabilities that enhance the system beyond what either mechanism achieves in isolation.

### 8.5 Production Considerations: The MiniMax-M2 Case Study

Despite promising research results, the recently released **MiniMax-M2** does not employ hybrid or linear attention, reverting to pure softmax attention. The MiniMax team's rationale provides critical insights into the gap between research and production:

1. **Scale-dependent failure modes:** While early experiments with Lightning Attention showed promising results at smaller scales on benchmarks such as MMLU, BBH, and MATH, the team observed **clear deficits in complex, multi-hop reasoning tasks** at larger scales.
2. **Numerical precision issues during RL:** Reinforcement learning training introduced numerical precision challenges specific to the linear attention components.
3. **Infrastructure maturity:** Production training infrastructure was more thoroughly optimized for standard attention.
4. **Sensitivity to co-factors:** Designing a new architecture at scale is a **multivariable optimization problem** that is expensive to ablate due to sensitivity to data distribution, optimizer configuration, learning rate scheduling, and other training hyperparameters.

However, the MiniMax team explicitly acknowledges the long-term trajectory:

> *"As GPU compute growth slows while data length keeps increasing, the benefits of linear and sparse attention will gradually emerge."*

This case study underscores that **architecture decisions at production scale involve engineering, infrastructure, and timeline constraints** that may override theoretical or small-scale empirical advantages.

---

## 9. Advanced Linear Attention: A Unified Gating Framework

### 9.1 The Gated State Update

A key insight from recurrent model design is the utility of allowing the hidden state to **selectively forget past information**. In the context of linear attention, this translates to introducing a **data-dependent gate** $G_t$ that modulates the previous state before the new key-value outer product is added:

$$
S_t = G_t \odot S_{t-1} + v_t k_t^\top
$$

where $G_t \in \mathbb{R}^{d_k \times d_v}$ (or a lower-dimensional parameterization thereof) controls how much of the previous state is retained. This gated update is the **unifying structural element** across virtually all recent linear attention and state space model variants.

### 9.2 Taxonomy of Gating Parameterizations

Different architectural families instantiate $G_t$ with distinct parameterizations, varying in expressiveness, computational cost, and the number of learnable parameters. The following taxonomy (adapted from Yang et al., 2024) provides a unified view:

| Model | Gate Parameterization | Learnable Parameters |
|-------|-----------------------|----------------------|
| **Mamba** (Gu & Dao, 2024) | $G_t = \exp\!\left(-(1^\top \alpha_t) \odot \exp(A)\right)$, $\alpha_t = \text{softplus}(x_t W_{\alpha_1} W_{\alpha_2})$ | $A \in \mathbb{R}^{d_k \times d_v}$, $W_{\alpha_1} \in \mathbb{R}^{d \times d/16}$, $W_{\alpha_2} \in \mathbb{R}^{d/16 \times d_v}$ |
| **Mamba-2** (Dao & Gu, 2024) | $G_t = \gamma_t \mathbf{1}^\top \mathbf{1}$, $\gamma_t = \exp(-\text{softplus}(x_t W_\gamma) \exp(a))$ | $W_\gamma \in \mathbb{R}^{d \times 1}$, $a \in \mathbb{R}$ |
| **mLSTM** (Beck et al., 2025) | $G_t = \gamma_t \mathbf{1}^\top \mathbf{1}$, $\gamma_t = \sigma(x_t W_\gamma)$ | $W_\gamma \in \mathbb{R}^{d \times 1}$ |
| **Gated Retention** (Sun et al., 2024) | $G_t = \gamma_t \mathbf{1}^\top \mathbf{1}$, $\gamma_t = \sigma(x_t W_\gamma)^{1/\tau}$ | $W_\gamma \in \mathbb{R}^{d \times 1}$ |
| **DFW** (Mao, 2022; Pramanik et al., 2023) | $G_t = \alpha_t^\top \beta_t$, $\alpha_t = \sigma(x_t W_\alpha)$, $\beta_t = \sigma(x_t W_\beta)$ | $W_\alpha \in \mathbb{R}^{d \times d_k}$, $W_\beta \in \mathbb{R}^{d \times d_v}$ |
| **GateLoop** (Katsch, 2024) | $G_t = \alpha_t^\top \mathbf{1}$, $\alpha_t = \sigma(x_t W_{\alpha_1}) \exp(x_t W_{\alpha_2} i)$ | $W_{\alpha_1}, W_{\alpha_2} \in \mathbb{R}^{d \times d_k}$ |
| **HGRN-2** (Qin et al., 2024) | $G_t = \alpha_t^\top \mathbf{1}$, $\alpha_t = \gamma + (1-\gamma)\sigma(x_t W_\alpha)$ | $W_\alpha \in \mathbb{R}^{d \times d_k}$, $\gamma \in (0,1)^{d_k}$ |
| **RWKV-6** (Peng et al., 2024) | $G_t = \alpha_t^\top \mathbf{1}$, $\alpha_t = \exp(-\exp(x_t W_\alpha))$ | $W_\alpha \in \mathbb{R}^{d \times d_k}$ |
| **GLA** (Yang et al., 2024) | $G_t = \alpha_t^\top \mathbf{1}$, $\alpha_t = \sigma(x_t W_{\alpha_1} W_{\alpha_2})^{1/\tau}$ | $W_{\alpha_1} \in \mathbb{R}^{d \times 16}$, $W_{\alpha_2} \in \mathbb{R}^{16 \times d_k}$ |

### 9.3 Key Observations Across Gating Variants

1. **Scalar vs. vector vs. matrix gates:** Mamba-2 and mLSTM use a scalar gate broadcast to all state dimensions, while Mamba, GLA, and DFW employ more expressive per-dimension or outer-product gating. Higher expressiveness increases capacity but also computational cost.
2. **Sigmoid vs. exponential parameterization:** Models differ in whether the gate is parameterized through sigmoid ($\sigma$), softplus, or double-exponential ($\exp(-\exp(\cdot))$) transformations, affecting the effective forgetting dynamics.
3. **Mamba-2 in production hybrid models:** Mamba-2 has been adopted as the SSM component in several frontier hybrid models, including Nemotron-H (NVIDIA et al., 2025), Falcon H1 (Zuo et al., 2025), and Granite-4.0-h (IBM Research, 2025).

### 9.4 Emerging Variants and Sparse Attention

Beyond the gated linear attention family, several complementary approaches address long-context scaling:

- **Gated DeltaNet** (adopted by Qwen3-Next, Qwen Team, 2025): Reports faster inference for long contexts, faster training, and improved benchmark performance compared to standard attention.
- **Kimi Delta Attention:** Anticipated in future Kimi releases.
- **Sparse Attention:** Reduces attention cost by computing interactions only for selected blocks or queries, rather than all pairs. Examples include:
  - **Native Sparse Attention** (Yuan et al., 2025)
  - **DeepSeek Sparse Attention** (DeepSeek-AI, 2025)
  - **InfLLM v2** (M. Team et al., 2025)

---

## 10. Base Architecture Selection: Dense vs. MoE vs. Hybrid

### 10.1 Architecture Comparison

| Architecture | Description | Pros | Cons |
|-------------|-------------|------|------|
| **Dense Transformer** | Standard decoder-only transformer; every parameter activates for every token | Widely supported; well-understood training dynamics; stable training; good per-parameter performance | Compute scales linearly with size; a 70B model costs $\sim$23× more than a 3B model |
| **Mixture-of-Experts (MoE)** | FFN layers replaced by multiple expert FFNs with a learned router; only $k$ experts active per token | Superior performance per FLOP for both training and inference; decouples capacity from compute | High memory demands (all experts must be loaded even if inactive); more complex training; framework support less mature; distributed training involves expert placement, load balancing, and all-to-all communication challenges |
| **Hybrid (Transformer + SSM/Linear Attention)** | Interleaves transformer attention layers with SSM or linear attention layers | Potentially superior long-context handling; more efficient for very long sequences ($O(Td^2)$ vs. $O(T^2d)$) | Least mature; fewest proven training recipes; limited framework support; scale-dependent failure modes observed in production |

### 10.2 Decision Framework

The architecture selection should be driven by three primary factors in the following order of priority:

**Factor 1: Deployment Environment**
- **Edge/mobile/memory-constrained:** MoE is typically infeasible because all experts must reside in memory simultaneously, even though only $k$ are active. Dense architectures are the default choice.
- **Server/cloud with ample memory:** MoE and hybrid architectures become viable.

**Factor 2: Team Expertise**
- **First LLM training or limited experience:** Dense transformers provide the most well-documented training recipes, debugging tools, and community support. The risk of under-diagnosed training failures is minimized.
- **Experienced with dense training:** MoE or MoE + Hybrid architectures offer superior performance per compute and warrant the additional engineering investment.
- **Highly experienced with frontier training:** Full design space (MoE + Hybrid, novel attention variants) is accessible.

**Factor 3: Timeline and Exploration Budget**
- **Tight timeline (proven path required):** Dense architectures minimize risk and time-to-result.
- **Flexible timeline (open to exploration):** MoE or MoE + Hybrid architectures can deliver significantly better performance per compute.

### 10.3 Decision Tree

```
Where will this model run?
├── Edge/mobile/memory-constrained
│   ├── Dense (most cases)
│   └── Hybrid (for experienced teams only)
└── Server/cloud (more memory available)
    └── What is the team's expertise?
        ├── First LLM training → Dense (focus on fundamentals)
        ├── Experienced (comfortable with dense)
        │   └── What is the timeline?
        │       ├── Tight → Dense
        │       └── Flexible → MoE or MoE + Hybrid
        └── Very experienced → MoE or MoE + Hybrid
```

### 10.4 Case Study: SmolLM3 Architecture Selection

The SmolLM3 project illustrates this decision framework in practice:

- **Deployment target:** On-device (edge/mobile) → memory-constrained
- **Timeline:** ~3 months → tight
- **Team expertise:** Primarily dense model training experience
- **Context length requirement:** 128K tokens → within dense model capabilities

**Decision:** Dense Llama-style architecture. MoE was ruled out by memory constraints; hybrid was ruled out by the tight timeline and limited exploration budget, given that dense architectures are capable of supporting the target context length.

---

## 11. Open Research Directions

The MoE and hybrid architecture design space remains actively evolving. The following non-exhaustive list identifies high-priority research directions:

### 11.1 MoE Training Dynamics and Stability

- **Zero-computation experts:** Experts that contribute to capacity without executing computation, as explored in the LongCat-Flash paper
- **MoE layer rescaling:** Techniques for adjusting the output scale of MoE layers to stabilize training
- **Training monitoring:** Diagnostic metrics and visualization tools for tracking expert utilization, specialization, and routing dynamics during training

### 11.2 Advanced Load Balancing

- **Orthogonal loss load balancing:** As proposed in ERNIE 4.5, using orthogonality constraints to encourage expert diversity
- **Scheduled balancing coefficients:** Varying $\alpha$ (or $\gamma$ in loss-free approaches) over the course of training—e.g., stronger balancing during early training, relaxed during fine-tuning

### 11.3 Architecture–Optimizer Interactions

- **Optimizer rankings under MoE:** Whether the relative performance of optimizers (e.g., Adam vs. AdaFactor vs. SOAP) changes when applied to MoE architectures
- **$\mu$P for MoE:** Extending the Maximal Update Parameterization ($\mu$P) framework to MoE architectures, where the per-expert learning dynamics differ from dense layers
- **Expert-aware learning rate adaptation:** Since each expert sees a different number of tokens per batch (depending on routing), the effective learning rate per expert varies; adaptive schemes may be beneficial

### 11.4 Structural Configuration

- **Number of initial dense layers:** Several models (e.g., DeepSeek-V3) begin with a fixed number of dense layers before transitioning to MoE layers; the optimal number of such layers remains an open question
- **Layer-wise sparsity scheduling:** Whether sparsity should vary across layers (e.g., denser early layers, sparser later layers)

### 11.5 Hybrid Architecture Maturation

- **Scale-dependent behavior:** Understanding why hybrid models sometimes underperform at large scale despite small-scale advantages
- **RL compatibility:** Addressing numerical precision issues that arise when applying reinforcement learning to hybrid architectures
- **Diffusion models for text:** Emerging exploratory direction, currently at an early stage insufficient for production recommendations

---

## 12. Conclusion

This report has provided a comprehensive technical treatment of sparse Mixture-of-Experts architectures, hybrid attention mechanisms, and the systematic decision framework for selecting base architectures in large language model development.

**Key findings and design principles:**

1. **MoE architectures decouple model capacity from per-token compute cost**, enabling frontier-scale capacity at a fraction of the dense inference cost. The Efficiency Leverage metric provides a rigorous framework for quantifying this advantage.

2. **Higher sparsity consistently improves loss** at matched FLOPs budgets, with diminishing returns at very high sparsity. The field has converged on a secular trend from sparsity $\sim$4× (2023) to $\sim$47× (2025), though optimal sparsity depends on hardware constraints and deployment requirements.

3. **Finer granularity** (more, smaller experts) provides a secondary but meaningful improvement in efficiency leverage, with recent models trending toward $G \geq 5$.

4. **A single shared expert** is a simple, low-risk configuration that improves specialization of routed experts without significant additional complexity. Its benefit increases with granularity.

5. **Load balancing is the most critical operational challenge** in MoE design. Both auxiliary loss-based and loss-free (bias-adjustment) approaches are viable, but the **scope of routing statistics** (local vs. global) has a pronounced and often under-appreciated impact on expert specialization and model quality. Global or cross-device aggregation is strongly recommended.

6. **Hybrid architectures** (transformer + SSM/linear attention) offer theoretically superior long-context scaling through the $O(Td^2)$ vs. $O(T^2d)$ complexity reduction. However, **production maturity remains limited**, with scale-dependent failure modes, RL compatibility issues, and infrastructure constraints reported by teams operating at frontier scale (MiniMax-M2).

7. **Architecture selection is fundamentally a multi-objective optimization** over deployment constraints, team expertise, timeline, and performance requirements. Dense transformers remain the lowest-risk default; MoE offers the best performance-per-compute for unconstrained settings; hybrid architectures represent the highest-potential but highest-risk frontier.

These design principles, grounded in empirical findings from frontier laboratories and rigorous scaling law analysis, provide a systematic foundation for architectural decisions in modern large language model development.

---

## 13. References

- Beck, M., et al. (2025). xLSTM: Extended Long Short-Term Memory.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Spaces (Mamba-2).
- DeepSeek-AI, et al. (2025). DeepSeek-V3 Technical Report.
- Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- IBM Research. (2025). Granite-4.0-h.
- K. Team, et al. (2025). Kimi K2 Technical Report.
- Katsch, N. (2024). GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling.
- Krajewski, J., et al. (2024). Scaling Laws for Fine-Grained Mixture of Experts.
- L. Team, et al. (2025). Ling 1.5 Technical Report.
- M. Team, et al. (2025). InfLLM v2.
- Mao, C. (2022). DFW: Data-dependent Forgetting and Writing.
- MiniMax, et al. (2025). MiniMax-01: Scaling Foundation Models with Lightning Attention.
- NVIDIA, et al. (2025). Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models.
- Peng, B., et al. (2024). RWKV-6.
- Peng, H., et al. (2021). Random Feature Attention.
- Pramanik, S., et al. (2023). Recurrent Linear Transformers.
- Qin, Z., et al. (2022). NormAttention: Cosine Normalized Attention without Softmax.
- Qin, Z., et al. (2024). HGRN-2: Gated Linear RNNs with State Expansion.
- Qiu, J., et al. (2025). Qwen Technical Report on MoE Load Balancing.
- Qwen Team. (2025). Qwen3-Next Technical Report.
- Sun, Y., et al. (2024). Retentive Network: A Successor to Transformer for Large Language Models.
- Tian, Y., et al. (2025). Scaling Laws for Mixture of Experts (Ant Group).
- Waleffe, R., et al. (2024). An Empirical Study of Mamba-based Language Models.
- Yang, S., et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training.
- Yuan, Z., et al. (2025). Native Sparse Attention.
- Zuo, S., et al. (2025). Falcon H1: A Family of Hybrid-Head Language Models.