# Optimizer Selection and Training Hyperparameter Configuration for Large Language Model Pretraining: A Comprehensive Technical Report
![06 optimizer and hyperparameters](./assets/06_optimizer_and_hyperparameters.png)
---

## Table of Contents

1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [Optimizer Selection for LLM Pretraining](#2-optimizer-selection-for-llm-pretraining)
   - 2.1 AdamW: Decoupled Weight Decay Regularization
   - 2.2 Muon: Second-Order Matrix Orthogonalization
   - 2.3 The Broader Optimizer Landscape
   - 2.4 Fair Comparison Methodology and Pitfalls
3. [Learning Rate Configuration](#3-learning-rate-configuration)
   - 3.1 The Role of Learning Rate in Gradient-Based Optimization
   - 3.2 Warmup Phase: Stabilizing Early Training Dynamics
   - 3.3 Learning Rate Schedules: Taxonomy and Analysis
   - 3.4 Ablation Study: WSD versus Cosine Decay
   - 3.5 Learning Rate Sweep Methodology
4. [Batch Size Configuration and Critical Batch Size Theory](#4-batch-size-configuration-and-critical-batch-size-theory)
   - 4.1 Gradient Noise and Batch Size Scaling
   - 4.2 Critical Batch Size: Theoretical Foundation
   - 4.3 Learning Rate–Batch Size Co-Scaling
   - 4.4 Dynamic Batch Size Scheduling
5. [Scaling Laws for Hyperparameter Prediction](#5-scaling-laws-for-hyperparameter-prediction)
   - 5.1 Compute Budget Estimation
   - 5.2 Power-Law Relationships for Optimal Hyperparameters
   - 5.3 Practical Application of Scaling Laws
6. [Case Study: SmolLM3 Hyperparameter Configuration](#6-case-study-smollm3-hyperparameter-configuration)
7. [Decision Framework: Balancing Exploration and Execution](#7-decision-framework-balancing-exploration-and-execution)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction and Problem Statement

### 1.1 Motivation

The pretraining of large language models (LLMs) constitutes a multi-million-dollar computational investment where the choice of optimizer, learning rate, learning rate schedule, and batch size exerts a first-order effect on final model quality. After architectural design decisions have been finalized—including layer count, hidden dimensionality, attention head configuration, and tokenizer vocabulary—the training hyperparameter configuration remains as the decisive bridge between a well-designed model specification and a successfully trained model checkpoint.

### 1.2 The Pitfall of Naive Literature Transplantation

A common but suboptimal practice is to directly adopt hyperparameter configurations from published model reports. While literature values provide reasonable initialization points, they carry implicit dependencies:

| **Implicit Dependency** | **Risk of Direct Adoption** |
|---|---|
| Architecture-specific coupling | Optimal $\eta$ for a given depth/width ratio may not transfer to alternative configurations |
| Data distribution specificity | Learning rate optima shift with data composition, quality, and domain distribution |
| Compute constraint artifacts | Published values may reflect hardware constraints rather than performance optimization |
| Historical inertia | Values may have been chosen early in development and never revisited |
| Schedule-length coupling | Cosine decay parameters are tightly coupled to total training duration |

Even when model developers conduct thorough hyperparameter sweeps, the resulting optima are inherently conditioned on their **exact combination** of architecture, data pipeline, training regime, and compute budget—none of which may match the practitioner's setup.

### 1.3 Scope and Objectives

This report provides a comprehensive technical treatment of:

1. **Optimizer selection**: Rigorous comparison of $\text{AdamW}$ and $\text{Muon}$, including mathematical formulations, memory footprints, and empirical behavior at scale.
2. **Learning rate scheduling**: Taxonomy and analysis of cosine decay, warmup-stable-decay (WSD), and multi-step schedules, with ablation-backed recommendations.
3. **Batch size optimization**: Theory of critical batch size, learning rate co-scaling, and dynamic batch size scheduling.
4. **Scaling laws for hyperparameters**: Methodology for predicting optimal learning rate and batch size as functions of compute budget.
5. **Empirical validation**: Ablation studies demonstrating the practical impact of each hyperparameter dimension.

---

## 2. Optimizer Selection for LLM Pretraining

### 2.1 AdamW: Decoupled Weight Decay Regularization

#### 2.1.1 Mathematical Formulation

Adam (Adaptive Moment Estimation) (Kingma & Ba, 2014) is a **first-order** optimization method that adapts the learning rate for each parameter using exponentially weighted moving averages of past gradients (first moment) and squared gradients (second moment).

Given parameters $\theta$, loss function $\mathcal{L}$, and gradient $g_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})$ at timestep $t$:

$$
m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t
$$

$$
v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \theta_{t-1} \right)
$$

where:
- $m_t$ is the first moment estimate (mean of gradients)
- $v_t$ is the second moment estimate (uncentered variance of gradients)
- $\hat{m}_t, \hat{v}_t$ are bias-corrected estimates
- $\eta$ is the learning rate
- $\epsilon$ is a numerical stability constant (typically $10^{-8}$)
- $\lambda$ is the weight decay coefficient
- $\beta_1, \beta_2$ are exponential decay rates for the moment estimates

#### 2.1.2 The Necessity of Decoupled Weight Decay (the "W")

In standard SGD, $L_2$ regularization is implemented by adding a penalty term $\frac{\lambda}{2} \|\theta\|^2$ to the loss, yielding an effective gradient contribution of $\lambda \theta$. The update becomes:

$$
\theta_t = \theta_{t-1} - \eta \left( g_t + \lambda \, \theta_{t-1} \right)
$$

In this formulation, $L_2$ regularization and weight decay are **mathematically equivalent** for SGD. However, this equivalence **breaks down** for adaptive optimizers like Adam. When $L_2$ regularization is applied within the Adam update rule, the adaptive scaling by $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ modulates the regularization term alongside the gradient. This means the **regularization strength becomes dependent on gradient magnitudes**, weakening its effect for parameters with large historical gradients and strengthening it for parameters with small gradients—precisely the opposite of the intended uniform shrinkage behavior.

AdamW (Loshchilov & Hutter, 2019) resolves this by **decoupling** weight decay from the adaptive gradient update:

$$
\theta_t = (1 - \eta \lambda) \, \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

This ensures that weight decay operates as a **uniform multiplicative shrinkage** on all parameters, independent of the adaptive learning rate state.

#### 2.1.3 Canonical Hyperparameter Configuration

A striking empirical observation across the LLM literature is the remarkable stability of AdamW hyperparameters across model families and scales:

| **Hyperparameter** | **Standard Value** | **Models Using This Configuration** |
|---|---|---|
| $\beta_1$ | $0.9$ | LLaMA 1/2/3, DeepSeek V1/V2/V3 |
| $\beta_2$ | $0.95$ | LLaMA 1/2/3, DeepSeek V1/V2/V3 |
| Gradient norm clipping | $1.0$ | Nearly universal |
| Weight decay $\lambda$ | $0.1$ | Standard (LLaMA 3 405B reduces to $0.01$) |

This consistency across years of development—from LLaMA 1 through DeepSeek-V3-671B—suggests either that these values occupy a genuinely broad optimum in the hyperparameter landscape, or that the community has collectively under-explored this space due to the prohibitive cost of large-scale ablations.

#### 2.1.4 Memory Footprint Analysis

AdamW maintains **two auxiliary state tensors** per parameter (the first and second moment estimates), resulting in a total optimizer state memory of:

$$
\text{Memory}_{\text{AdamW}} = 2 \times |\theta| \times \text{bytes per element}
$$

For a model with $N$ parameters in mixed precision (FP32 optimizer states), this amounts to $8N$ bytes of optimizer state, compared to $2N$ bytes for the parameters themselves in BF16. The optimizer state thus constitutes a **significant fraction** of total GPU memory during training.

---

### 2.2 Muon: Second-Order Matrix Orthogonalization

#### 2.2.1 Core Algorithm

Muon is a **second-order optimizer** that operates on the **matrix-level geometry** of parameter tensors, in contrast to AdamW's parameter-wise (diagonal) preconditioning. The update rule proceeds as follows:

$$
G_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})
$$

$$
B_t = \mu \, B_{t-1} + G_t
$$

$$
O_t = \text{NewtonSchulz}_5(B_t) \approx U V^\top \quad \text{where } B_t = U \Sigma V^\top \text{ (SVD)}
$$

$$
\theta_t = \theta_{t-1} - \eta \, O_t
$$

where:
- $G_t$ is the gradient matrix at step $t$
- $B_t$ is the momentum-accumulated gradient buffer
- $\mu$ is the momentum coefficient
- $\text{NewtonSchulz}_5(\cdot)$ is a 5-iteration Newton-Schulz approximation to the matrix sign function, yielding an approximate orthogonal polar factor
- $U, \Sigma, V$ are the singular value decomposition components of $B_t$

The "second-order" character of Muon is embedded within the **Newton-Schulz iteration**, which implicitly computes matrix inverses (specifically, the inverse square root of $B_t^\top B_t$) without explicitly forming Hessian matrices.

#### 2.2.2 Three Key Mechanistic Principles

**Principle 1: Matrix-Wise Geometry versus Parameter-Wise Updates**

AdamW preconditions each parameter independently using a diagonal approximation to the second moment:

$$
\text{AdamW: } \quad \Delta \theta_i \propto \frac{m_i}{\sqrt{v_i} + \epsilon}
$$

This treats each weight as an independent scalar, ignoring correlations between rows and columns of weight matrices. Muon, by contrast, treats each weight matrix $W \in \mathbb{R}^{m \times n}$ as a **single geometric object** and updates along $O = UV^\top$, which captures the **row–column subspace structure** of the gradient.

**Principle 2: Isotropic Steps via Orthogonalization**

The singular value decomposition $G = U \Sigma V^\top$ decomposes the gradient into:
- **Directions**: Left and right singular subspaces $U$ and $V$
- **Magnitudes**: Singular values $\Sigma$

By replacing $G$ with $UV^\top$ (discarding $\Sigma$), Muon produces updates that are **isotropic in the active subspaces**. While discarding singular values may appear to lose information, this operation:
- Eliminates **axis-aligned bias** inherent in diagonal preconditioning
- Encourages **exploration of directions** that would otherwise be suppressed by very small singular values
- Ensures each gradient direction receives **equal step magnitude**, regardless of curvature

> **Open Research Question**: Whether the isotropic exploration induced by singular value removal encodes qualitatively different capabilities into the model—effects that may not be apparent from loss alone—remains an active area of investigation.

**Principle 3: Empirical Tolerance to Larger Batch Sizes**

In practice, Muon exhibits greater tolerance to larger batch sizes compared to AdamW. This property is of significant practical importance, as it implies potential improvements in **training throughput** without proportional degradation in data efficiency—a critical consideration for distributed training at scale.

#### 2.2.3 Adoption Status

| **Model** | **Optimizer** |
|---|---|
| Kimi K2 | Muon |
| GLM-4.5 | Muon |
| Virtually all other frontier models | AdamW |

The limited adoption of Muon despite its theoretical appeal can be attributed to several factors:
1. **Switching cost**: Changing a core training component in production systems carries substantial risk
2. **Ablation expense**: Validating optimizer performance at scale requires full-length training runs, not just short ablations
3. **Hyperparameter non-transferability**: Optimal hyperparameters from AdamW cannot be directly reused for Muon (see Section 2.4)
4. **Implementation complexity**: The Newton-Schulz iteration introduces additional engineering complexity relative to AdamW's elementwise operations

---

### 2.3 The Broader Optimizer Landscape

The optimization research community has produced a diverse taxonomy of methods, each exploring different aspects of gradient information:

| **Category** | **Optimizers** | **Key Mechanism** |
|---|---|---|
| Adaptive first-order | AdamW, NAdamW, StableAdamW, Lion | Diagonal second-moment preconditioning (with variants) |
| Second-order / structured | Muon, Shampoo, SOAP, CASPR | Matrix-level or Kronecker-factored preconditioning |
| Curvature-aware | Sophia, PSGD | Approximate Hessian or Fisher information |
| Momentum variants | AdEMAMix, DION | Multiple timescale momentum or momentum decomposition |

Each of these methods proposes a different approximation to the ideal preconditioning matrix (the inverse Hessian in Newton's method), trading off between:
- **Computational cost** per step
- **Memory overhead** for auxiliary states
- **Quality of curvature approximation**
- **Stability at scale**

A comprehensive benchmark by Wen et al. (2025) from Stanford's Marin team demonstrated that the relative ranking of optimizers is **highly sensitive to hyperparameter tuning quality**, underscoring the importance of rigorous comparison methodology.

---

### 2.4 Fair Comparison Methodology and Pitfalls

Comparing optimizers is methodologically challenging for the following reasons:

**Problem 1: Scale-Dependent Dynamics**

Optimizer behavior at $10^{17}$ FLOPs may not predict behavior at $10^{21}$ FLOPs. Gradient distributions, loss landscape curvature, and numerical precision requirements all evolve with scale in ways that are difficult to simulate in small ablations.

**Problem 2: Hyperparameter Entanglement**

Each optimizer defines its own hyperparameter space. A fair comparison requires independent hyperparameter optimization for **each** optimizer, spanning at minimum:
- Learning rate $\eta$ (1D sweep)
- Momentum parameters $\beta_1, \beta_2$ (potentially 2D sweep)
- Weight decay $\lambda$ (additional dimension)

This implies a minimum of $O(k^d)$ training runs for a $d$-dimensional grid with $k$ points per dimension—an exponentially expensive proposition.

**Problem 3: Weak Baseline Bias**

A systematic issue identified by Wen et al. (2025) is that new optimizers are frequently compared against **poorly tuned AdamW baselines**. This inflates reported gains and creates misleading conclusions about relative optimizer merit. Specifically:

$$
\Delta_{\text{reported}} = \mathcal{L}(\text{AdamW}_{\text{poorly tuned}}) - \mathcal{L}(\text{NewOpt}_{\text{well tuned}}) \gg \Delta_{\text{true}}
$$

**Recommendation**: Any optimizer comparison must ensure that all candidates, including the baseline, receive **equal hyperparameter tuning budget**. Without this condition, no comparative conclusions are scientifically valid.

---

## 3. Learning Rate Configuration

### 3.1 The Role of Learning Rate in Gradient-Based Optimization

The learning rate $\eta$ governs the magnitude of parameter updates at each training step. For a generic gradient-based optimizer:

$$
\theta_{t} = \theta_{t-1} - \eta \cdot \Phi(g_t, \text{state}_{t-1})
$$

where $\Phi(\cdot)$ represents the optimizer-specific transformation of the raw gradient $g_t$ (e.g., momentum-corrected and variance-normalized for AdamW).

The learning rate creates a fundamental tension:

| **Regime** | **Behavior** | **Consequence** |
|---|---|---|
| $\eta \ll \eta^*$ | Updates too small | Slow convergence; risk of entrapment in suboptimal basins; wasted compute budget |
| $\eta \approx \eta^*$ | Updates appropriately scaled | Efficient convergence to good minima |
| $\eta \gg \eta^*$ | Updates too large | Overshooting optima; oscillatory behavior; potential divergence ($\mathcal{L} \to \infty$) |

where $\eta^*$ denotes the (time-varying) optimal learning rate.

Critically, the optimal learning rate is **not constant** throughout training:
- **Early training**: Large gradients and high loss suggest tolerance for larger $\eta$, enabling rapid traversal of the loss landscape
- **Late training**: As the model approaches a minimum, smaller $\eta$ is required for fine-grained convergence without overshooting

This non-stationarity of the optimal $\eta$ motivates the use of **learning rate schedules**.

---

### 3.2 Warmup Phase: Stabilizing Early Training Dynamics

#### 3.2.1 Rationale

At initialization, model parameters are drawn from random distributions (e.g., Xavier, Kaiming), and the optimizer's moment estimates ($m_0 = 0$, $v_0 = 0$ for AdamW) are uninitialized. Applying a large learning rate in this state leads to:
- **Biased moment estimates**: The bias correction terms $\frac{1}{1 - \beta_1^t}$ and $\frac{1}{1 - \beta_2^t}$ are large for small $t$, amplifying noise
- **Unstable gradient magnitudes**: Initial gradients may have high variance and inconsistent directions
- **Loss spikes and divergence**: Large updates to randomly initialized weights can push the model into degenerate regions of the loss landscape

#### 3.2.2 Implementation

The standard warmup protocol linearly increases the learning rate from $0$ (or a small $\eta_{\min}$) to the peak value $\eta_{\text{peak}}$ over $T_{\text{warmup}}$ steps:

$$
\eta_t = \eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}}, \quad \text{for } t \leq T_{\text{warmup}}
$$

#### 3.2.3 Empirical Guidelines

| **Guideline** | **Recommendation** |
|---|---|
| Fixed warmup | 2,000 steps (used by most modern LLMs regardless of model size) |
| Proportional warmup | 1–5% of total training steps (for very short training runs) |
| Sensitivity | Increasing warmup beyond standard values does not generally improve performance for long training runs |

---

### 3.3 Learning Rate Schedules: Taxonomy and Analysis

#### 3.3.1 Cosine Decay

**Formulation**: After warmup, the learning rate follows a cosine annealing curve from $\eta_{\text{peak}}$ to $\eta_{\min}$:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\min}) \left(1 + \cos\left(\frac{\pi \cdot (t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)
$$

**Advantages**:
- Simple to implement
- Smooth decay profile prevents abrupt optimization dynamics
- Extensively validated across years of neural network training

**Disadvantages**:
- **Inflexibility**: Requires $T_{\text{total}}$ to be specified a priori; the cosine cycle length must match the total training duration
- **Non-extendable**: If the model has not plateaued and additional compute becomes available, training cannot be extended without restarting from scratch
- **Scaling law interference**: This rigidity historically skewed scaling law research. Kaplan et al. (2020) used a fixed cosine schedule length when training models on varying token counts, which **underestimated the impact of data size**. The Chinchilla study (Hoffmann et al., 2022) corrected this by matching schedule length to each model's actual training duration

#### 3.3.2 Warmup–Stable–Decay (WSD)

**Formulation** (Hu et al., 2024): The schedule consists of three phases:

$$
\eta_t = \begin{cases}
\eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \quad \text{(warmup)} \\
\eta_{\text{peak}} & \text{if } T_{\text{warmup}} < t \leq T_{\text{stable}} \quad \text{(stable)} \\
\eta_{\text{peak}} \cdot f_{\text{decay}}\!\left(\frac{t - T_{\text{stable}}}{T_{\text{total}} - T_{\text{stable}}}\right) & \text{if } t > T_{\text{stable}} \quad \text{(decay)}
\end{cases}
$$

where $f_{\text{decay}}(\cdot)$ is typically a linear or cosine decay function, and $T_{\text{stable}}$ marks the transition from the constant-rate phase to the decay phase.

**Decay phase duration**: Recommended allocation of **10–20% of total tokens** to the decay phase (Hägele et al., 2024). The required cooldown duration to match cosine performance **decreases with longer training runs**.

**Advantages**:
- **Extendable mid-run**: Training can be continued by extending the stable phase without restarting
- **Early progress assessment**: Decay can be applied at any intermediate checkpoint to assess model quality
- **Scaling law compatibility**: Multiple token count experiments can be conducted within a single main training run by branching at different points during the stable phase

#### 3.3.3 Multi-Step Schedule

**Formulation** (DeepSeek-AI et al., 2024): The learning rate is maintained at $\eta_{\text{peak}}$ and then reduced through discrete multiplicative drops:

$$
\eta_t = \begin{cases}
\eta_{\text{peak}} & \text{if } t \leq T_1 \\
\alpha_1 \cdot \eta_{\text{peak}} & \text{if } T_1 < t \leq T_2 \\
\alpha_2 \cdot \eta_{\text{peak}} & \text{if } t > T_2
\end{cases}
$$

where $T_1, T_2$ are the step boundaries and $\alpha_1, \alpha_2 < 1$ are the multiplicative decay factors.

**Phase split configurations** explored by DeepSeek LLM:

| **Configuration** | **Stable / Step 1 / Step 2** | **Performance vs. Cosine** |
|---|---|---|
| Baseline | 80% / 10% / 10% | Matches cosine |
| Extended decay | 70% / 15% / 15% | Slightly outperforms cosine |
| Aggressive decay | 60% / 20% / 20% | Slightly outperforms cosine |

#### 3.3.4 Hybrid Schedules (DeepSeek V2/V3 Variants)

More recent DeepSeek models introduced hybrid schedule designs:

- **DeepSeek V2**: Adjusted multi-step proportions to 60/30/10, allocating more time to the first decay step
- **DeepSeek V3**: Introduced a hybrid approach: constant phase → cosine decay (from 67% to 97% of training) → brief constant phase → final sharp step

These hybrid schedules combine the flexibility of WSD-style constant phases with the smoothness of cosine decay, though **no published ablations validate these specific design choices** in the respective technical reports.

#### 3.3.5 Comparative Summary

| **Property** | **Cosine Decay** | **WSD** | **Multi-Step** |
|---|---|---|---|
| Requires $T_{\text{total}}$ a priori | Yes | No | No |
| Extendable mid-run | No | Yes | Yes |
| Number of additional hyperparameters | 0 | 1 (decay fraction) | 2+ (step positions) |
| Smoothness of decay | High | Medium | Low (discontinuous) |
| Empirical performance | Baseline | Matches baseline | Matches/exceeds baseline |
| Scaling law experiment compatibility | Poor | Excellent | Good |

---

### 3.4 Ablation Study: WSD versus Cosine Decay

#### 3.4.1 Experimental Setup

To empirically validate that WSD matches cosine decay performance, the following ablation was conducted:

- **Model**: 1B parameter architecture (finalized from prior architecture ablations)
- **Data**: 45B tokens from the curated pretraining corpus
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$)
- **Configurations compared**:
  - Cosine decay (baseline)
  - WSD with 10% decay window
  - WSD with 20% decay window
- **Evaluation**: Training loss + downstream benchmarks (HellaSwag, MMLU, ARC, PIQA, OpenBookQA, WinoGrande)

#### 3.4.2 Results and Analysis

**Training Loss**: All three configurations converge to **equivalent final loss values**. During the stable phase (before WSD's decay begins), cosine decay achieves lower instantaneous loss, as expected—since cosine has already begun decaying the learning rate, producing smaller update steps and thus lower noise in the loss signal.

**Downstream Benchmarks**: HellaSwag, ARC, PIQA, and other benchmarks show similar patterns: cosine leads during the stable phase, but WSD exhibits an **approximately linear improvement** during the decay phase, recovering to match cosine's final performance.

**Key Observation**: The convergence behavior during WSD's decay phase is notably steep and approximately linear, suggesting that the transition from high to low learning rate induces rapid refinement of model representations once the model has already explored the loss landscape during the stable phase.

#### 3.4.3 Conclusion

A 10–20% decay window in WSD is **sufficient to match cosine decay's final performance** while providing substantially greater flexibility for:
- Mid-run training extensions
- Intermediate checkpoint evaluation
- Scaling law experimentation within a single training run

---

### 3.5 Learning Rate Sweep Methodology

#### 3.5.1 Sweep Design

To identify the acceptable learning rate range for a given model and data configuration, a coarse sweep is conducted across orders of magnitude:

**Ablation Setup (illustrative)**:
- Model: 1B parameters, 45B tokens
- Learning rates tested: $\{10^{-4}, 5 \times 10^{-4}, 5 \times 10^{-3}, 5 \times 10^{-2}\}$

#### 3.5.2 Observed Regimes

| **Learning Rate** | **Behavior** | **Classification** |
|---|---|---|
| $\eta = 5 \times 10^{-2}$ | Loss spikes immediately and never recovers | Divergent (catastrophically high) |
| $\eta = 10^{-4}$ | Stable training but significantly slower convergence | Too conservative |
| $\eta = 5 \times 10^{-4}$ | Good convergence, strong downstream performance | Near-optimal range |
| $\eta = 5 \times 10^{-3}$ | Comparable to $5 \times 10^{-4}$ with slight instability risk | Near-optimal range |

#### 3.5.3 Limitations of Short Sweeps

Critically, the optimal learning rate is **dependent on training duration**. The learning rate that produces the fastest convergence in a short ablation (e.g., 45B tokens) may not be optimal for the full training run (e.g., 4T tokens). Short sweeps are effective for **ruling out clearly suboptimal values** but cannot guarantee optimality at the target scale.

For SmolLM3 at 3B scale on 100B tokens:
- $\eta = 2 \times 10^{-4}$ converged significantly faster than $\eta = 10^{-4}$
- $\eta = 3 \times 10^{-4}$ offered only marginal improvement over $\eta = 2 \times 10^{-4}$, with increased instability risk during long training
- **Selected value**: $\eta = 2 \times 10^{-4}$ as the optimal risk-adjusted choice

This underscores that learning rate selection involves a **risk–reward tradeoff**: marginally higher learning rates may yield slightly better convergence in ablations but introduce **non-trivial divergence risk** over extended training durations involving trillions of tokens.

---

## 4. Batch Size Configuration and Critical Batch Size Theory

### 4.1 Gradient Noise and Batch Size Scaling

#### 4.1.1 Mini-Batch Gradient Estimator

The mini-batch gradient computed over $B$ samples provides a stochastic estimate of the true (population) gradient:

$$
\tilde{g}_B = \frac{1}{B} \sum_{i=1}^{B} \tilde{g}^{(i)}
$$

where $\tilde{g}^{(i)} = \nabla_\theta \mathcal{L}(x_i, \theta)$ is the per-sample gradient.

**Statistical Properties**:

The expectation of the mini-batch gradient is the true gradient (unbiasedness):

$$
\mathbb{E}[\tilde{g}_B] = g
$$

The covariance shrinks inversely with $B$:

$$
\text{Cov}(\tilde{g}_B) = \frac{\Sigma}{B}
$$

where $\Sigma = \text{Cov}(\tilde{g}^{(i)})$ is the per-sample gradient covariance matrix.

#### 4.1.2 Update Variance

For a simple SGD update $\Delta w = -\eta \, \tilde{g}_B$, the variance of the parameter update is:

$$
\text{Var}(\Delta w) \propto \frac{\eta^2 \, \Sigma}{B}
$$

This relationship is foundational: it establishes that **batch size and learning rate interact through the update variance**, and any change to one must be compensated by the other to maintain training dynamics.

---

### 4.2 Critical Batch Size: Theoretical Foundation

#### 4.2.1 Definition

The **critical batch size** $B_{\text{crit}}$ (McCandlish et al., 2018) is the batch size beyond which increasing $B$ yields diminishing returns in data efficiency. Below $B_{\text{crit}}$, doubling the batch size approximately halves the number of required updates without increasing total token consumption. Above $B_{\text{crit}}$, each additional sample contributes redundant gradient information.

#### 4.2.2 Two Scaling Regimes

| **Regime** | **Condition** | **Behavior** |
|---|---|---|
| Below critical | $B < B_{\text{crit}}$ | After retuning $\eta$, same loss is reached with same total token count; no data wasted; throughput improves |
| Above critical | $B > B_{\text{crit}}$ | Same loss requires more total tokens; wall-clock time may decrease (more parallelism) but data efficiency degrades |

#### 4.2.3 Gradient Noise Scale

The critical batch size is related to the **gradient noise scale** $B_{\text{noise}}$, defined as:

$$
B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|g\|^2}
$$

where $\text{tr}(\Sigma)$ is the trace of the gradient covariance and $\|g\|^2$ is the squared norm of the true gradient. The simple estimate $B_{\text{simple}}$ derived from this quantity provides a lower bound on the actual critical batch size.

**Dynamic Nature**: $B_{\text{crit}}$ is **not fixed**; it increases as training progresses:
- **Early training**: Large gradient magnitudes $\|g\|^2$ → small $B_{\text{noise}}$ → small $B_{\text{crit}}$
- **Late training**: Gradient magnitudes decrease as the model converges → $B_{\text{noise}}$ increases → larger batches become efficient

---

### 4.3 Learning Rate–Batch Size Co-Scaling

#### 4.3.1 Square Root Scaling Rule

To maintain constant update variance when scaling the batch size by a factor $k$:

$$
B_{\text{new}} = k \cdot B_{\text{original}} \implies \eta_{\text{new}} = \sqrt{k} \cdot \eta_{\text{original}}
$$

This **square root scaling rule** follows directly from the update variance relationship:

$$
\text{Var}(\Delta w) \propto \frac{\eta^2}{B} \implies \frac{\eta_{\text{new}}^2}{k \cdot B} = \frac{\eta_{\text{original}}^2}{B} \implies \eta_{\text{new}} = \sqrt{k} \cdot \eta_{\text{original}}
$$

#### 4.3.2 Caveats for Adaptive Optimizers

The square root rule is derived under simplified assumptions (SGD-like updates). For adaptive optimizers like AdamW:
- Interactions between batch size and the moment estimates ($\beta_1$, $\beta_2$) introduce **nonlinear effects** that can deviate from the square root prediction
- The effective per-parameter learning rate $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ introduces additional complexity

#### 4.3.3 Empirical Validation via Branching (Merrill et al., 2025)

A pragmatic alternative to theoretical scaling rules is the **branching validation** protocol:

1. **Branch**: At a chosen training step, duplicate the training run
2. **Modify**: In the branch, increase the batch size to $k \cdot B$ and rescale $\eta$ to $\sqrt{k} \cdot \eta$
3. **Warm up**: Linearly warm up the new learning rate and reset the optimizer state
4. **Compare**: Monitor loss curves for both branches over a defined time window
5. **Accept/reject**: If the branched loss aligns with the original within a predetermined tolerance $\epsilon_{\text{tol}}$, adopt the larger batch size; otherwise, revert

This approach provides an **empirical safeguard** against theoretical scaling failures and has been shown to reveal that $B_{\text{simple}}$ estimates **systematically underestimate** the true critical batch size.

---

### 4.4 Dynamic Batch Size Scheduling

#### 4.4.1 Batch Size Warmup

Given that $B_{\text{crit}}$ increases during training, several frontier models employ **batch size warmup**—starting with a smaller batch size and increasing it as training progresses:

| **Model** | **Initial Batch Size** | **Final Batch Size** | **Transition Point** |
|---|---|---|---|
| DeepSeek-V3 | 12.6M tokens | 62.9M tokens | ~469B tokens |
| MiniMax-01 | — | 128M tokens | Final training stage |

The rationale mirrors learning rate warmup: it keeps training on the **efficient frontier** of the throughput–data-efficiency tradeoff as the gradient noise scale evolves.

#### 4.4.2 Batch Size Schedule as Implicit Learning Rate Decay

An alternative perspective, employed by MiniMax-01, treats increasing batch size (without proportionally increasing the learning rate) as an **implicit learning rate decay**. From the update variance equation:

$$
\text{Var}(\Delta w) \propto \frac{\eta^2}{B}
$$

Increasing $B$ while holding $\eta$ constant reduces the effective update magnitude, functionally equivalent to reducing $\eta$ while holding $B$ constant. This perspective unifies batch size scheduling and learning rate scheduling as two manifestations of the same underlying control mechanism: **modulation of the effective step size**.

---

### 4.5 Practical Batch Size Selection Protocol

The following protocol integrates the theoretical and empirical considerations discussed above:

1. **Initialize**: Select an initial batch size and learning rate based on scaling laws (Section 5) or literature values for similar configurations
2. **Evaluate throughput headroom**: Assess whether the current batch size fully utilizes available hardware parallelism
3. **Test larger batch sizes**: If throughput gains are available, increase $B$ to $kB$ with corresponding $\eta \to \sqrt{k} \cdot \eta$
4. **Validate via branching**: Use the branching protocol (Section 4.3.3) to confirm that the larger batch size preserves training dynamics
5. **Adopt or revert**: If throughput improvement is insignificant or data efficiency degrades beyond tolerance, retain the original values

---

## 5. Scaling Laws for Hyperparameter Prediction

### 5.1 Compute Budget Estimation

#### 5.1.1 FLOPs Approximation

The standard approximation for the total compute budget of a transformer training run is:

$$
C \approx 6 \times N \times D
$$

where:
- $C$ is measured in floating-point operations (FLOPs)
- $N$ is the number of model parameters
- $D$ is the number of training tokens
- The constant $6$ accounts for the approximately 6 FLOPs per parameter per token required for forward and backward passes through a transformer

#### 5.1.2 Interpretation

This formulation provides a **hardware-agnostic** measure of computational effort:
- Training a 1B-parameter model on 100B tokens: $C \approx 6 \times 10^{20}$ FLOPs
- Training a 2B-parameter model on 100B tokens: $C \approx 1.2 \times 10^{21}$ FLOPs (2× more compute)
- Training a 1B-parameter model on 200B tokens: $C \approx 1.2 \times 10^{21}$ FLOPs (also 2× more compute)

For more precise estimates accounting for Mixture-of-Experts (MoE) layers and hybrid architectures, the `num_floating_point_operations` function in Megatron-LM provides a detailed computation.

---

### 5.2 Power-Law Relationships for Optimal Hyperparameters

#### 5.2.1 Methodology (following DeepSeek LLM)

The scaling law derivation proceeds as follows:

1. **Select schedule**: Use WSD for its flexibility in extending training runs to different token counts without restarting

2. **Define compute grid**: Train models across a range of compute budgets, e.g., $C \in \{10^{17}, 5 \times 10^{17}, 10^{18}, 5 \times 10^{18}, 10^{19}, 2 \times 10^{19}\}$ FLOPs, corresponding to different combinations of model sizes and token counts

3. **Sweep hyperparameters**: For each compute budget, perform sweeps over learning rate $\eta$ and batch size $B$, identifying configurations yielding **near-optimal** validation loss (within a defined margin, e.g., 0.25% of the best loss)

4. **Extract data points**: Each near-optimal configuration yields a data point $\left(C, \eta^*\right)$ or $\left(C, B^*\right)$

5. **Fit power laws**: On a log–log scale, the relationships typically appear approximately linear, indicating power-law behavior:

$$
\eta^*(C) = a_\eta \cdot C^{b_\eta}
$$

$$
B^*(C) = a_B \cdot C^{b_B}
$$

where $a_\eta, b_\eta, a_B, b_B$ are fitted constants.

#### 5.2.2 Core Intuition

The empirical finding is that as compute budget $C$ increases:

$$
\frac{d \eta^*}{d C} < 0 \quad \text{(optimal learning rate decreases)}
$$

$$
\frac{d B^*}{d C} > 0 \quad \text{(optimal batch size increases)}
$$

**Physical interpretation**: As training becomes larger and longer, the optimization process benefits from:
- **More stable updates** (smaller learning rates) to avoid accumulating small instabilities over millions of steps
- **More efficient gradient estimation** (larger batch sizes) to ensure each update step is well-directed

#### 5.2.3 Robustness: Broad Optima

A practically important finding is that for a fixed model size and compute budget, performance remains stable across a **wide range of hyperparameters**. This implies:
- There exists a **broad sweet spot** rather than a narrow optimum
- Exact precision in learning rate and batch size is not required—approximate values within the optimal neighborhood suffice
- This considerably reduces the practical cost of hyperparameter tuning

---

### 5.3 Practical Application of Scaling Laws

#### 5.3.1 Predicting Hyperparameters for Target Scale

Given fitted scaling laws $\eta^*(C)$ and $B^*(C)$, the workflow for a new training run is:

1. **Estimate compute budget**: $C_{\text{target}} = 6 \times N_{\text{target}} \times D_{\text{target}}$
2. **Predict initial hyperparameters**: $\eta_{\text{init}} = \eta^*(C_{\text{target}})$, $B_{\text{init}} = B^*(C_{\text{target}})$
3. **Validate with short ablation**: Perform a brief training run to confirm stability and reasonable convergence behavior
4. **Adjust batch size for throughput**: If hardware permits, increase $B$ toward $B_{\text{crit}}$ with corresponding $\eta$ rescaling (Section 4.3)

#### 5.3.2 Distinction: Scaling Law Optima vs. Practical Optima

The objective function implicit in scaling laws is **optimal data efficiency** (minimum tokens to reach a target loss). The practitioner's actual objective is typically:

$$
\min_{\eta, B} \quad \mathcal{L}(\theta_T) \quad \text{subject to} \quad T_{\text{wall}} \leq T_{\text{budget}}, \quad \text{GPU}_{\text{count}} = G
$$

This means the practical optimal batch size may **exceed** the scaling-law optimal batch size if:
- Increasing $B$ substantially improves throughput (tokens per second)
- The data efficiency penalty remains within acceptable bounds
- The resulting higher total token requirement is within budget

---

## 6. Case Study: SmolLM3 Hyperparameter Configuration

### 6.1 Optimizer Selection Process

The SmolLM3 training campaign conducted systematic optimizer ablations:

| **Optimizer** | **Scale** | **Performance** | **Stability** | **Outcome** |
|---|---|---|---|---|
| AdamW | 1B/100B | Highest final loss | Most stable | Baseline |
| AdEMAMix | 1B/100B | Similar to Muon | Less sensitive than Muon | Competitive |
| Muon | 1B/100B | Lowest final loss (when tuned) | Sensitive to $\eta$; divergence-prone | Best at 1B |
| AdamW | 3B scale-up | — | Most stable | **Selected** |
| AdEMAMix | 3B scale-up | — | More frequent divergence | Rejected |
| Muon | 3B scale-up | — | More frequent divergence | Rejected |

At 3B scale, both Muon and AdEMAMix exhibited **more frequent divergence episodes**. While a parallelism bug discovered after the ablation phase (detailed in training infrastructure reports) may have contributed, this could not be confirmed retroactively. The decision to adopt AdamW was made on the basis of **risk minimization** for a multi-week, multi-million-token training campaign.

### 6.2 Final Hyperparameter Configuration

| **Hyperparameter** | **Value** | **Rationale** |
|---|---|---|
| Optimizer | AdamW | Maximum stability at 3B scale |
| $\beta_1$ | $0.9$ | Standard; no evidence of benefit from deviation |
| $\beta_2$ | $0.95$ | Standard; no evidence of benefit from deviation |
| Weight decay $\lambda$ | $0.1$ | Standard |
| Gradient norm clipping | $1.0$ | Standard |
| Learning rate schedule | WSD (10% decay) | Flexibility; proven in SmolLM2; mid-run decay capability |
| Peak learning rate $\eta$ | $2 \times 10^{-4}$ | LR sweep at 3B/100B: best risk-adjusted convergence |
| Global batch size | 2.36M tokens | Throughput-optimized; 2M–4M range showed minimal loss/benchmark impact |

### 6.3 Decision Rationale Summary

The SmolLM3 hyperparameter configuration prioritized:

1. **Stability over marginal performance**: AdamW was selected despite Muon's superior loss at 1B scale, because reliability over multi-week training runs is non-negotiable
2. **Flexibility over optimality**: WSD was chosen over cosine decay for its ability to extend training and conduct mid-run experiments
3. **Risk-adjusted learning rate**: $2 \times 10^{-4}$ was selected over $3 \times 10^{-4}$ because the marginal convergence improvement did not justify the increased divergence risk over trillions of tokens
4. **Throughput-driven batch size**: $2.36$M tokens was selected for hardware utilization rather than theoretical optimal data efficiency, as the 2M–4M range showed no meaningful performance differentiation

---

## 7. Decision Framework: Balancing Exploration and Execution

### 7.1 Principles

The hyperparameter optimization process for LLM pretraining must navigate a fundamental tension between **exploration** (discovering better configurations) and **execution** (completing the training run with available resources).

**Principle 1: Allocate compute budget proportionally to expected impact.**

From empirical experience across multiple training campaigns, the rank-ordering of impact on final model quality is typically:

$$
\text{Data curation} > \text{Architecture design} > \text{Hyperparameter tuning} > \text{Optimizer selection}
$$

Spending weeks perfecting a novel optimizer configuration yields less return than investing equivalent compute in data quality improvements.

**Principle 2: Prefer flexibility and stability over marginal performance.**

When two configurations yield comparable performance, select the one with:
- Greater **implementation maturity** and community validation
- More **flexibility** for mid-training adjustments
- Better **stability** guarantees over long training durations

**Principle 3: Establish exploration deadlines and enforce them.**

There will always exist one more hyperparameter to sweep, one more optimizer to evaluate, one more schedule variant to test. The model that completes training with good-enough hyperparameters will always outperform the theoretically optimal model that never starts training.

### 7.2 Checkpoint Comparison Caveat

When comparing intermediate checkpoints between models trained with different schedules (e.g., cosine vs. WSD):

> **Critical methodological requirement**: If comparing a cosine checkpoint against a WSD checkpoint during the stable phase, a decay must be applied to the WSD checkpoint to produce a fair comparison. Without this correction, the WSD checkpoint will appear worse than it is, as it has not yet benefited from the learning rate reduction that its schedule prescribes.

---

## 8. Conclusion

This report has provided a comprehensive technical treatment of optimizer selection and training hyperparameter configuration for LLM pretraining, covering the following key findings:

1. **AdamW remains the dominant optimizer** for LLM pretraining due to its stability, simplicity, and broad empirical validation, with canonical hyperparameters ($\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$) showing remarkable consistency across model families and scales.

2. **Muon offers theoretical and empirical advantages** (matrix-level geometry, isotropic updates, batch size tolerance) but faces adoption barriers related to stability at scale, hyperparameter sensitivity, and the cost of rigorous comparison.

3. **WSD and multi-step schedules match cosine decay performance** while providing substantially greater practical flexibility, making them preferable for real-world training campaigns where training duration may be adjusted mid-run.

4. **The learning rate must be tuned for the specific training configuration**, as optimal values depend on model size, data volume, and total compute budget. Scaling laws provide principled initialization, while short sweeps eliminate clearly suboptimal values.

5. **Batch size selection involves a throughput–efficiency tradeoff** governed by the critical batch size, which itself evolves during training. Square root learning rate scaling provides a first-order correction when adjusting batch size, with empirical branching validation as a robust safeguard.

6. **Scaling laws for hyperparameters** reveal power-law relationships between compute budget and optimal learning rate/batch size, enabling principled extrapolation to target training scales.

7. **Pragmatic decision-making**—favoring stability, flexibility, and completion over theoretical optimality—is essential for successful LLM pretraining at scale.

---

## 9. References

- DeepSeek-AI, :, et al. (2024). DeepSeek LLM: Scaling open-source language models with longtermism.
- DeepSeek-AI, Liu, et al. (2024). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model.
- DeepSeek-AI et al. (2025). DeepSeek-V3 Technical Report.
- 5 Team et al. (2025). GLM-4.5 Technical Report.
- Hägele, A., et al. (2024). Scaling data-constrained language models.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla).
- Hu, S., et al. (2024). MiniCPM: Unveiling the potential of small language models with scalable training strategies.
- Kaplan, J., et al. (2020). Scaling laws for neural language models.
- Kingma, D. P. & Ba, J. (2014). Adam: A method for stochastic optimization.
- Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts.
- Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization.
- McCandlish, S., et al. (2018). An empirical model of large-batch training.
- Merrill, W., et al. (2025). Scaling laws and batch size dynamics for large language model training.
- Smith, L. N. & Topin, N. (2018). Super-convergence: Very fast training of neural networks using large learning rates.
- Wen, K., et al. (2025). The urgency of optimizer benchmarking: How tuning matters more than optimizer choice.

---