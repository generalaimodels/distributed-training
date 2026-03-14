# Data Curation and Mixture Optimization for Large Language Model Pretraining: A Comprehensive Technical Report

---

## 1. Introduction and Motivation

### 1.1 The Primacy of Data in Language Model Performance

A persistent and empirically validated observation in large language model (LLM) development is that **data quality, composition, and proportional mixing exert a dominant influence on downstream model capability**—frequently surpassing the impact of architectural modifications, hyperparameter optimization, or raw compute scaling. Despite significant research investment in novel attention mechanisms, positional encodings, and optimizer design, deployed models that fail to meet performance expectations can, in the overwhelming majority of cases, trace their deficiencies to defects in the training data pipeline.

Consider the following failure taxonomy commonly observed when data curation is insufficiently rigorous:

| **Observed Failure Mode** | **Probable Data-Level Root Cause** |
|---|---|
| Incoherent or syntactically broken code generation | Insufficient or low-quality source code in the training corpus |
| Degraded arithmetic and mathematical reasoning | Underrepresentation of structured mathematical data |
| Uncontrolled language switching mid-sequence | Improperly balanced multilingual data proportions |
| Factual hallucination on common-knowledge queries | Dominance of noisy web-crawl data lacking factual grounding |
| Inability to follow multi-step instructions | Absence of procedurally structured training documents |

These failure modes underscore a foundational principle:

> **If model architecture defines *how* a model learns, then training data defines *what* it learns. No amount of compute budget or optimizer refinement can compensate for training on an inadequately curated corpus.**

### 1.2 Problem Statement

Given a target model with parameter count $N$, a total training token budget $T$, and a collection of $K$ heterogeneous data sources $\mathcal{D} = \{D_1, D_2, \ldots, D_K\}$, the **data curation problem** can be formally stated as:

$$\text{Find } \mathbf{w}^* = (w_1^*, w_2^*, \ldots, w_K^*) \quad \text{such that} \quad \sum_{i=1}^{K} w_i = 1, \quad w_i \geq 0 \;\; \forall i$$

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{j=1}^{M} \lambda_j \cdot \mathcal{L}_j\!\left(\theta^*(\mathbf{w})\right)$$

where:

- $w_i$ denotes the **sampling weight** (proportion) assigned to source $D_i$
- $\theta^*(\mathbf{w})$ denotes the model parameters obtained after training on the mixture defined by $\mathbf{w}$
- $\mathcal{L}_j$ denotes the evaluation loss (or negative performance metric) on the $j$-th downstream benchmark
- $\lambda_j$ denotes the importance weight for the $j$-th evaluation objective
- $M$ denotes the total number of evaluation benchmarks

This optimization is **non-convex, model-dependent, and computationally intractable** to solve analytically, necessitating empirical ablation-based methodologies.

### 1.3 Scope and Objectives

This report provides an end-to-end technical treatment of data curation for LLM pretraining, covering:

1. **Data source taxonomy and quality characterization**
2. **Mixture formulation and the mathematics of data proportioning**
3. **Trade-off analysis between competing training objectives**
4. **Multi-stage curriculum training strategies**
5. **Ablation experimental design and methodology**
6. **Automated mixture optimization approaches and their limitations**
7. **Data repetition effects and deduplication considerations**
8. **Evaluation framework design for mixture validation**

---

## 2. Data Source Taxonomy and Quality Stratification

### 2.1 Source Classification

Training corpora for modern LLMs are assembled from multiple heterogeneous sources, each contributing distinct knowledge domains and linguistic characteristics. A rigorous classification is presented below:

| **Source Category** | **Examples** | **Primary Contribution** | **Typical Volume** | **Quality Variance** |
|---|---|---|---|---|
| General web crawl | Common Crawl, FineWeb | Broad world knowledge, linguistic diversity | Very high ($> 10^{13}$ tokens) | High variance |
| Curated web data | FineWeb-Edu, FineWeb2 | Filtered educational and informational content | High ($10^{11}$–$10^{12}$ tokens) | Moderate variance |
| Source code repositories | The Stack, Stack-Edu | Programming languages, algorithmic reasoning | Moderate ($10^{11}$ tokens) | Moderate variance |
| Mathematical corpora | FineMath, proof-net datasets | Formal and informal mathematical reasoning | Low ($10^{9}$–$10^{10}$ tokens) | Low variance (curated) |
| Multilingual text | CC-100, OPUS, CulturaX | Cross-lingual transfer, language coverage | High ($10^{12}$ tokens) | High variance |
| Books and long-form text | Project Gutenberg, BookCorpus | Long-range coherence, narrative structure | Moderate | Low variance |
| Scientific literature | S2ORC, arXiv | Domain-specific technical knowledge | Moderate | Low variance |
| Conversational and instructional | ShareGPT, FLAN collections | Dialogue patterns, instruction following | Low–Moderate | Moderate variance |

### 2.2 Quality Characterization Framework

Within each source category, documents exhibit substantial quality heterogeneity. Quality characterization typically operates along the following orthogonal dimensions:

**Definition 1 (Document Quality Score).** For a document $d$ drawn from source $D_i$, define a composite quality score:

$$Q(d) = \sum_{k=1}^{P} \alpha_k \cdot q_k(d), \quad \text{where} \quad \sum_{k=1}^{P} \alpha_k = 1$$

where $q_k(d) \in [0, 1]$ denotes the score along the $k$-th quality axis and $\alpha_k$ denotes its importance weight. The quality axes include:

1. **Linguistic coherence** $q_{\text{ling}}(d)$: Grammatical correctness, fluency, and structural integrity
2. **Informational density** $q_{\text{info}}(d)$: Ratio of substantive content to boilerplate, advertisements, or navigation artifacts
3. **Factual accuracy** $q_{\text{fact}}(d)$: Alignment with verified knowledge (often assessed via proxy models)
4. **Educational value** $q_{\text{edu}}(d)$: Degree to which the document teaches concepts in a structured manner (as operationalized in FineWeb-Edu)
5. **Domain relevance** $q_{\text{dom}}(d)$: Alignment with target training domains
6. **Deduplication status** $q_{\text{dedup}}(d)$: Binary or graded indicator of near-duplicate or exact-duplicate content

### 2.3 Quality Filtering as a Constrained Optimization

Aggressive quality filtering—retaining only documents with $Q(d) > \tau$ for a high threshold $\tau$—is superficially attractive but introduces a critical constraint. Define the **effective unique token count** after filtering:

$$T_{\text{eff}}(\tau) = \sum_{i=1}^{K} \sum_{d \in D_i} |d| \cdot \mathbb{1}[Q(d) > \tau]$$

where $|d|$ denotes the token count of document $d$. For a model with training budget $T$:

- If $T_{\text{eff}}(\tau) \geq T$: No repetition is necessary; the filtered corpus suffices.
- If $T_{\text{eff}}(\tau) < T$: The training procedure must repeat data, with each document seen on average $T / T_{\text{eff}}(\tau)$ times.

This directly motivates the **quality-quantity trade-off** formalized in the next section.

---

## 3. The Mathematics of Data Mixing

### 3.1 Formal Mixture Definition

**Definition 2 (Data Mixture).** A data mixture $\mathcal{M}$ over $K$ sources is defined by a weight vector $\mathbf{w} = (w_1, w_2, \ldots, w_K)$ residing on the $(K-1)$-dimensional probability simplex:

$$\Delta^{K-1} = \left\{ \mathbf{w} \in \mathbb{R}^K \;\middle|\; \sum_{i=1}^{K} w_i = 1, \quad w_i \geq 0 \;\; \forall i \right\}$$

During training, at each gradient step, a mini-batch $\mathcal{B}$ of size $B$ is constructed by sampling $\lfloor w_i \cdot B \rfloor$ documents (or token sequences) from source $D_i$.

The **expected number of tokens** consumed from source $D_i$ over $T$ total training tokens is:

$$T_i = w_i \cdot T$$

### 3.2 The Implicit Downweighting Principle

A critical and often underappreciated property of mixture design is the **zero-sum nature of weight allocation**. Since $\sum_{i=1}^{K} w_i = 1$, any increase in $w_j$ for some source $D_j$ necessarily decreases the effective weight on the remaining sources:

$$\frac{\partial w_i}{\partial w_j} = -\frac{w_i}{1 - w_j} \quad \text{for } i \neq j \quad \text{(under proportional redistribution)}$$

**Consequence:** Upweighting a task-relevant source (e.g., increasing the proportion of source code to improve coding performance) implicitly downweights all other sources, potentially degrading performance on unrelated tasks such as general knowledge question answering or multilingual generation.

This gives rise to the **capability trade-off surface**, formally defined as:

**Definition 3 (Pareto Frontier of Capabilities).** The set of mixture weight vectors $\mathbf{w}^*$ such that no alternative mixture $\mathbf{w}'$ can improve performance on any benchmark $\mathcal{L}_j$ without degrading performance on at least one other benchmark $\mathcal{L}_{j'}$:

$$\mathcal{P} = \left\{ \mathbf{w} \in \Delta^{K-1} \;\middle|\; \nexists \; \mathbf{w}' \in \Delta^{K-1} \text{ s.t. } \mathcal{L}_j(\mathbf{w}') \leq \mathcal{L}_j(\mathbf{w}) \;\forall j \text{ and } \exists j' : \mathcal{L}_{j'}(\mathbf{w}') < \mathcal{L}_{j'}(\mathbf{w}) \right\}$$

Identifying points on $\mathcal{P}$ requires empirical exploration, as the mapping $\mathbf{w} \mapsto \{\mathcal{L}_j(\theta^*(\mathbf{w}))\}_{j=1}^{M}$ is non-linear and model-dependent.

### 3.3 Data Repetition and Its Deleterious Effects

When the training budget $T$ exceeds the available unique tokens $T_{\text{eff}}$ for a given mixture, data repetition becomes unavoidable. Define the **epoch count** for source $D_i$:

$$E_i = \frac{w_i \cdot T}{|D_i|}$$

where $|D_i|$ is the total unique token count in source $D_i$. When $E_i > 1$, documents in $D_i$ are revisited multiple times.

Empirical findings (Muennighoff et al., 2025) demonstrate that excessive repetition leads to:

1. **Diminishing returns in loss reduction:** The marginal loss improvement per repeated epoch decreases approximately as:

$$\Delta \mathcal{L}(E) \approx \Delta \mathcal{L}(1) \cdot E^{-\beta}, \quad \beta > 0$$

2. **Memorization and degraded generalization:** The model increasingly memorizes specific training sequences rather than learning generalizable patterns, measurable as a widening gap between training loss and validation loss:

$$\text{Generalization Gap} = \mathcal{L}_{\text{train}} - \mathcal{L}_{\text{val}} \propto E^{\gamma}, \quad \gamma > 0$$

3. **Potential for training instability:** At very high repetition rates, loss spikes and gradient norm anomalies have been observed.

**Practical implication:** For large-scale training runs (e.g., $T = 11 \times 10^{12}$ tokens for SmolLM3), the mixture must be designed such that high-quality but low-volume sources are not excessively repeated, necessitating the inclusion of lower-quality but higher-volume sources to maintain $E_i$ within acceptable bounds across all $i$.

### 3.4 The Quality–Quantity–Repetition Trilemma

The interplay among these factors can be expressed as a trilemma:

```
                    Quality
                    /      \
                   /        \
                  /   Data    \
                 /  Curation   \
                /   Trilemma    \
               /________________\
          Quantity            Low Repetition
```

- **Maximizing quality** reduces available volume, forcing repetition or reduced training budget utilization.
- **Maximizing quantity** admits lower-quality data, potentially degrading model capabilities.
- **Minimizing repetition** constrains mixture weights to respect source-level volume budgets, limiting quality-weighted optimization.

No single mixture simultaneously optimizes all three objectives. The mixture designer must select an operating point that reflects the priority ordering among these constraints.

---

## 4. Multi-Stage Curriculum Training

### 4.1 From Static to Dynamic Mixtures

#### 4.1.1 Static Mixture Training (Legacy Approach)

Early LLM training paradigms (e.g., GPT-3, LLaMA-1) employed a **fixed mixture** throughout the entire training run:

$$\mathbf{w}(t) = \mathbf{w}_0 \quad \forall \; t \in [0, T]$$

where $t$ denotes the training token index. While operationally simple, this approach fails to exploit the empirical observation that a model's sensitivity to data composition varies across the training trajectory.

#### 4.1.2 Multi-Stage Mixture Training (Current Best Practice)

Modern training pipelines partition the total training budget $T$ into $S$ stages, each with a distinct mixture:

$$\mathbf{w}(t) = \mathbf{w}_s \quad \text{for } t \in [T_{s-1}, T_s), \quad s = 1, 2, \ldots, S$$

where $T_0 = 0$ and $T_S = T$.

**Definition 4 (Training Curriculum).** A training curriculum $\mathcal{C}$ is a sequence of (mixture, token budget, learning rate schedule) triples:

$$\mathcal{C} = \left\{ \left( \mathbf{w}_s, \; \Delta T_s, \; \eta_s(\cdot) \right) \right\}_{s=1}^{S}$$

where $\Delta T_s = T_s - T_{s-1}$ and $\eta_s(\cdot)$ is the learning rate schedule for stage $s$.

### 4.2 Theoretical Justification: Recency Bias in Gradient-Based Learning

The empirical basis for multi-stage training rests on the observation that **a language model's final behavior is disproportionately influenced by data encountered during the later stages of training** (Chen et al., 2025b). This phenomenon can be understood through the lens of gradient dynamics.

Consider the parameter update at training step $t$:

$$\theta_{t+1} = \theta_t - \eta_t \cdot \nabla_\theta \mathcal{L}(\theta_t; x_t)$$

where $x_t$ is the training sample at step $t$ and $\eta_t$ is the learning rate. The contribution of step $t$ to the final parameters $\theta_T$ can be approximated (under linearization) as:

$$\theta_T \approx \theta_0 - \sum_{t=0}^{T-1} \eta_t \cdot \prod_{t'=t+1}^{T-1} \left(I - \eta_{t'} H_{t'}\right) \cdot \nabla_\theta \mathcal{L}(\theta_t; x_t)$$

where $H_{t'} = \nabla^2_\theta \mathcal{L}(\theta_{t'}; x_{t'})$ is the Hessian at step $t'$. The product of $(I - \eta_{t'} H_{t'})$ terms acts as a **decay operator** that attenuates the influence of earlier gradients. Combined with the learning rate schedule (where $\eta_t$ typically decreases), updates from later stages carry relatively higher influence on the final parameter configuration.

**Practical implication:** High-quality, task-critical data should be concentrated in the later stages of training where its marginal impact on final model behavior is maximized.

### 4.3 Canonical Multi-Stage Structure

A widely adopted curriculum structure consists of three stages:

#### Stage 1: Main Pretraining (Bulk Learning Phase)

- **Token budget:** $\Delta T_1 \approx 0.6T$ to $0.9T$
- **Mixture composition:** Dominated by high-volume general sources (web crawl, general text)
- **Learning rate:** Warm-up followed by constant or slow cosine decay
- **Objective:** Establish broad linguistic competence, world knowledge, and basic reasoning capabilities

#### Stage 2: Domain-Enriched Continuation (Targeted Upweighting Phase)

- **Token budget:** $\Delta T_2 \approx 0.05T$ to $0.2T$
- **Mixture composition:** Increased proportions of domain-specific data (code, mathematics, multilingual data)
- **Learning rate:** Continued decay from Stage 1 endpoint
- **Objective:** Strengthen specific capability areas identified as deficient via intermediate evaluations

#### Stage 3: Annealing Phase (Final Refinement)

- **Token budget:** $\Delta T_3 \approx 0.02T$ to $0.1T$
- **Mixture composition:** Highest-quality subsets across all domains; small, curated datasets introduced
- **Learning rate:** Aggressive decay toward zero (typically linear or cosine annealing to a minimum $\eta_{\min}$)

$$\eta_3(t) = \eta_{\text{start}} \cdot \left(1 - \frac{t - T_2}{T_3 - T_2}\right) + \eta_{\min}$$

- **Objective:** Final calibration of model capabilities, incorporation of highest-signal data with maximum influence on final parameters

### 4.4 Stage Transition Decision Criteria

The decision to transition between stages is governed by empirically monitored signals:

1. **Performance plateau detection:** If evaluation metrics on target benchmarks exhibit diminishing improvement rates:

$$\frac{\partial \text{Perf}_j}{\partial t} < \epsilon_j \quad \text{for benchmark } j$$

this signals that the current mixture has exhausted its marginal utility for that domain.

2. **Capability imbalance detection:** If performance on one domain improves steadily while another stagnates or degrades, the mixture should be adjusted to rebalance:

$$\left| \frac{\Delta \text{Perf}_j}{\Delta t} - \frac{\Delta \text{Perf}_{j'}}{\Delta t} \right| > \delta_{jj'}$$

3. **Data exhaustion monitoring:** If a high-priority source approaches its epoch limit ($E_i \to E_{\max}$), the mixture should shift to reduce sampling from that source.

4. **Learning rate schedule alignment:** Stage transitions are often synchronized with predefined learning rate milestones (e.g., onset of annealing decay).

---

## 5. Ablation Experimental Design and Methodology

### 5.1 Foundational Principles

Ablation experiments for data mixture optimization must satisfy the following methodological requirements:

| **Requirement** | **Rationale** |
|---|---|
| **Controlled variable isolation** | Only one mixture dimension should change per experiment to attribute causal effects |
| **Sufficient training duration** | Short runs may not reveal long-horizon effects of mixture changes |
| **Target-scale execution** | Mixture effects are model-capacity-dependent; small-scale proxies may mislead |
| **Comprehensive evaluation** | Metrics must span all target capability domains, not just the varied domain |
| **Reproducibility** | Fixed seeds, deterministic data ordering, and checkpointed states |

### 5.2 Ablation Architectures

Two complementary ablation architectures are employed in practice:

#### 5.2.1 From-Scratch Ablation

Train the full-scale model (e.g., 3B parameters) from random initialization with distinct mixture configurations for a shortened token budget (e.g., $T_{\text{abl}} = 50 \times 10^9$ to $100 \times 10^9$ tokens):

$$\theta_0 \sim \mathcal{N}(0, \sigma^2 I) \quad \xrightarrow{\text{Train with } \mathbf{w}_a, \; T_{\text{abl}}} \quad \theta^*_a$$

**Procedure:**

1. Define a set of candidate mixtures $\{\mathbf{w}_a\}_{a=1}^{A}$
2. For each $\mathbf{w}_a$, train from the same initialization $\theta_0$ for $T_{\text{abl}}$ tokens
3. Evaluate all resulting models $\{\theta^*_a\}$ on the full benchmark suite
4. Compare performance profiles to identify superior mixture configurations

**Advantages:** Captures mixture effects from the earliest training dynamics; avoids confounds from prior training history.

**Limitations:** High computational cost ($A \times$ per-experiment cost); short $T_{\text{abl}}$ may not predict long-horizon behavior accurately.

#### 5.2.2 Annealing Ablation (Checkpoint-Based)

Resume training from an intermediate checkpoint $\theta_{T_c}$ of the main training run, applying different mixture configurations during the annealing phase:

$$\theta_{T_c} \quad \xrightarrow{\text{Anneal with } \mathbf{w}_a, \; \Delta T_{\text{anneal}}} \quad \theta^*_a$$

**Procedure:**

1. Select an intermediate checkpoint $\theta_{T_c}$ (e.g., at $T_c = 7 \times 10^{12}$ tokens)
2. For each candidate annealing mixture $\mathbf{w}_a$, continue training from $\theta_{T_c}$ for $\Delta T_{\text{anneal}}$ tokens with learning rate decay
3. Evaluate and compare

**Advantages:** Directly tests annealing-phase mixture sensitivity; lower computational cost per experiment (only $\Delta T_{\text{anneal}}$ tokens per run); directly applicable to multi-stage curriculum design.

**Limitations:** Results are conditioned on the specific checkpoint $\theta_{T_c}$ and its training history; does not capture mixture effects on early training dynamics.

### 5.3 Scale Considerations in Ablation Design

**Theorem (Informal — Capacity-Mixture Interaction).** The optimal mixture $\mathbf{w}^*$ is a function of the model's parameter count $N$. For two models with $N_1 < N_2$:

$$\mathbf{w}^*(N_1) \neq \mathbf{w}^*(N_2) \quad \text{in general}$$

This arises because models with higher capacity $N$ can **absorb** a wider diversity of data without cross-domain interference. A small model ($N_1$) may experience severe performance degradation when trained on highly multilingual data, whereas a larger model ($N_2$) can accommodate the same multilingual ratio without sacrificing English performance.

**Practical consequence:** Data mixture ablations should be conducted at or near the **target model scale** to avoid misleading conclusions. For SmolLM3 (3B parameters), ablations were executed directly at 3B scale rather than on smaller proxy models.

### 5.4 Evaluation Framework for Ablation Analysis

The evaluation suite must be designed to capture the full trade-off surface across domains. For a model targeting $M$ capability domains, define the **mixture quality score**:

$$\text{MQS}(\mathbf{w}) = \sum_{j=1}^{M} \lambda_j \cdot \text{Perf}_j(\theta^*(\mathbf{w}))$$

where $\lambda_j$ are application-specific importance weights and $\text{Perf}_j$ is the performance on the $j$-th evaluation benchmark.

Additionally, the **worst-case regression metric** is critical for detecting catastrophic degradation:

$$\text{WCR}(\mathbf{w}) = \min_{j \in \{1, \ldots, M\}} \left[ \text{Perf}_j(\theta^*(\mathbf{w})) - \text{Perf}_j(\theta^*(\mathbf{w}_{\text{baseline}})) \right]$$

A mixture is unacceptable if $\text{WCR}(\mathbf{w}) < -\delta_{\text{max}}$ for some predetermined regression tolerance $\delta_{\text{max}}$.

Comprehensive evaluation coverage should include:

| **Domain** | **Representative Benchmarks** | **Metric Type** |
|---|---|---|
| General English NLU | MMLU, HellaSwag, ARC-Challenge | Accuracy |
| Mathematical Reasoning | GSM8K, MATH, Minerva | Exact Match / Accuracy |
| Code Generation | HumanEval, MBPP, MultiPL-E | Pass@$k$ |
| Multilingual Understanding | MGSM, XWinograd, multilingual MMLU | Accuracy (per-language and averaged) |
| Long-Context Comprehension | RULER, Needle-in-a-Haystack | Recall / Accuracy |
| Commonsense Reasoning | Winogrande, PIQA, OpenBookQA | Accuracy |

---

## 6. Automated Mixture Optimization Methods

### 6.1 DoReMi (Domain Reweighting with Minimax Optimization)

Xie et al. (2023) propose learning domain weights using a **distributionally robust optimization** framework.

**Formulation:** Train a small proxy model $\theta_{\text{proxy}}$ and a reference model $\theta_{\text{ref}}$. Define the **excess loss** for domain $i$:

$$\ell_i^{\text{excess}}(\theta) = \mathcal{L}_i(\theta) - \mathcal{L}_i(\theta_{\text{ref}})$$

where $\mathcal{L}_i(\theta) = \mathbb{E}_{x \sim D_i}[-\log p_\theta(x)]$. The domain weights are updated via exponentiated gradient ascent:

$$w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp\!\left(\eta_w \cdot \ell_i^{\text{excess}}(\theta^{(t)})\right)$$

The proxy model parameters $\theta$ are simultaneously updated via standard gradient descent on the reweighted loss:

$$\mathcal{L}_{\text{DoReMi}}(\theta) = \sum_{i=1}^{K} w_i \cdot \mathcal{L}_i(\theta)$$

**Objective:** The minimax formulation seeks weights that minimize the worst-case excess loss across domains:

$$\min_\theta \max_{\mathbf{w} \in \Delta^{K-1}} \sum_{i=1}^{K} w_i \cdot \ell_i^{\text{excess}}(\theta)$$

### 6.2 RHO-LOSS (Reducible Holdout Loss Selection)

Mindermann et al. (2022) propose a **sample-level** selection criterion rather than a domain-level reweighting scheme.

**Formulation:** For each candidate training sample $x$, compute:

$$\text{RHO}(x) = \mathcal{L}(x; \theta_{\text{train}}) - \mathcal{L}(x; \theta_{\text{ref}})$$

where $\theta_{\text{train}}$ is the current model and $\theta_{\text{ref}}$ is a reference model trained on high-quality data. Samples are selected for training if:

$$\text{RHO}(x) > \tau_{\text{select}}$$

**Interpretation:** Samples with high $\text{RHO}(x)$ are those where the current model's loss is high but the reference model's loss is low, indicating the sample is **learnable, task-relevant, and not yet learned**. Samples with low $\text{RHO}(x)$ are either already learned (low $\mathcal{L}(x; \theta_{\text{train}})$) or inherently noisy/unlearnable (high $\mathcal{L}(x; \theta_{\text{ref}})$).

### 6.3 RegMix (Regularized Regression for Mixture Optimization)

Liu et al. (2025) frame mixture optimization as a **regression problem**:

1. Train a set of small proxy models on diverse mixture configurations $\{\mathbf{w}_a\}_{a=1}^{A}$
2. Evaluate each on multiple benchmarks to obtain performance vectors $\{\mathbf{p}_a\}_{a=1}^{A}$
3. Fit a regularized regression model:

$$\hat{\mathbf{p}} = f(\mathbf{w}; \phi) + \lambda_{\text{reg}} \|\phi\|_2^2$$

4. Optimize the fitted model to find $\mathbf{w}^*$ that maximizes predicted aggregate performance:

$$\mathbf{w}^* = \arg\max_{\mathbf{w} \in \Delta^{K-1}} \sum_{j=1}^{M} \lambda_j \cdot \hat{p}_j(\mathbf{w})$$

### 6.4 Empirical Assessment of Automated Methods

Despite theoretical elegance, empirical evaluation of these automated methods reveals significant limitations in practice:

| **Method** | **Observed Behavior** | **Practical Limitation** |
|---|---|---|
| DoReMi | Converges toward weights that approximate the natural size distribution of sources | Marginal improvement over informed manual baselines; proxy model fidelity degrades at scale |
| RHO-LOSS | Effective for small-scale filtering but computationally expensive for trillion-token budgets | Requires maintaining a reference model; selection threshold sensitivity |
| RegMix | Promising interpolation within explored mixture space | Extrapolation beyond explored configurations is unreliable; regression model capacity limits |

**Key finding:** State-of-the-art production models (as of 2025) continue to rely predominantly on **manual mixture tuning through systematic ablation experiments and annealing studies**, supplemented by informed heuristics derived from prior training campaigns. The automated methods serve as useful initialization points but do not replace the need for empirical validation at target scale.

---

## 7. Data Quality Infrastructure

### 7.1 Deduplication

Deduplication is a prerequisite step before mixture design, as duplicate content inflates the apparent volume of a source while providing no additional learning signal. Standard approaches include:

- **Exact deduplication:** Hash-based removal of byte-identical documents
- **Near-deduplication:** MinHash + Locality-Sensitive Hashing (LSH) to identify documents with high Jaccard similarity:

$$J(d_1, d_2) = \frac{|S(d_1) \cap S(d_2)|}{|S(d_1) \cup S(d_2)|} > \tau_J$$

where $S(d)$ denotes the set of $n$-gram shingles extracted from document $d$. Typical thresholds range from $\tau_J = 0.7$ to $0.8$.

- **Cross-source deduplication:** Removing duplicates that appear across multiple data sources to prevent hidden repetition inflation in the final mixture.

### 7.2 Quality Classifier Pipelines

For web-crawl sources, quality classifiers are typically trained as binary or regression models to predict document quality scores:

$$\hat{Q}(d) = \sigma\!\left(\mathbf{v}^\top \cdot \text{Enc}(d) + b\right)$$

where $\text{Enc}(d)$ is a fixed encoder representation (e.g., from a fastText model or a small transformer), $\mathbf{v}$ and $b$ are learned parameters, and $\sigma$ is the sigmoid function.

Training data for quality classifiers is typically derived from:

- Positive examples: Documents from curated, high-quality sources (e.g., Wikipedia, textbooks, peer-reviewed publications)
- Negative examples: Random web-crawl samples filtered for low quality

### 7.3 Language Identification and Routing

For multilingual mixtures, accurate language identification (LID) is essential to ensure that per-language mixture weights are faithfully realized:

$$\hat{l}(d) = \arg\max_{l \in \mathcal{L}} \; p(l \mid d)$$

where $\mathcal{L}$ is the set of target languages. Misclassification can lead to contamination of language-specific sub-corpora, degrading both the contaminated language's quality and the purity of the intended source.

---

## 8. Practical Mixture Design Workflow

### 8.1 End-to-End Pipeline

The complete mixture design workflow, as operationalized for large-scale models such as SmolLM3, proceeds through the following phases:

```
┌─────────────────────────────────────────────────┐
│  Phase 1: Capability Specification               │
│  Define target tasks, priority ordering,         │
│  and acceptable performance envelopes            │
├─────────────────────────────────────────────────┤
│  Phase 2: Source Inventory & Quality Audit       │
│  Catalog available datasets, measure volume,     │
│  assess quality distributions, identify gaps     │
├─────────────────────────────────────────────────┤
│  Phase 3: Initial Mixture Hypothesis             │
│  Propose candidate mixtures informed by prior    │
│  experience, scaling law estimates, and source   │
│  volume constraints                              │
├─────────────────────────────────────────────────┤
│  Phase 4: From-Scratch Ablation Campaign         │
│  Execute controlled experiments at target        │
│  scale with shortened token budgets              │
├─────────────────────────────────────────────────┤
│  Phase 5: Main Training Execution                │
│  Deploy the best-performing mixture for the      │
│  main pretraining phase                          │
├─────────────────────────────────────────────────┤
│  Phase 6: Intermediate Evaluation & Diagnosis    │
│  Monitor metrics at intermediate checkpoints;    │
│  identify capability bottlenecks                 │
├─────────────────────────────────────────────────┤
│  Phase 7: Annealing Ablation Campaign            │
│  Test candidate annealing mixtures from          │
│  intermediate checkpoint                         │
├─────────────────────────────────────────────────┤
│  Phase 8: Final Annealing Execution              │
│  Apply the selected annealing mixture for the    │
│  final training stage                            │
├─────────────────────────────────────────────────┤
│  Phase 9: Final Evaluation & Reporting           │
│  Comprehensive benchmark evaluation; ablation    │
│  analysis; documentation of mixture rationale    │
└─────────────────────────────────────────────────┘
```

### 8.2 Key Decision Questions

At each phase, the following questions must be systematically addressed:

1. **What do we want the model to excel at?** — Defines the priority vector $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_M)$ over evaluation objectives.

2. **Which datasets are best suited for each target domain, and how should they be combined?** — Determines the source-to-domain mapping and initial weight hypotheses.

3. **Do we have sufficient high-quality data for the target training scale?** — Computes $T_{\text{eff}}(\tau)$ for various quality thresholds and checks against the training budget $T$, informing the quality-quantity trade-off.

4. **What are the known cross-domain interference patterns?** — Informs which capability pairs are likely to compete for capacity (e.g., English fluency vs. multilingual breadth).

5. **At what training stage should specialized data be introduced?** — Governs curriculum design and stage transition scheduling.

---

## 9. Summary of Principles and Recommendations

The following consolidated principles emerge from the analysis presented in this report:

| **Principle** | **Technical Rationale** |
|---|---|
| Data quality dominates architecture in determining downstream utility | The training loss landscape is shaped by data statistics; no optimizer can extract signal absent from the input distribution |
| Mixture weights reside on a simplex; all adjustments are zero-sum | Upweighting one source necessarily downweights others; every decision involves explicit or implicit trade-offs |
| Excessive data repetition degrades generalization | Repeated exposure to identical sequences promotes memorization over abstraction, with empirically observed power-law diminishing returns |
| Multi-stage curricula exploit the recency bias of gradient-based learning | Later training updates have disproportionate influence on final model behavior; high-quality data has maximal impact in annealing phases |
| Ablations must be conducted at target scale | Mixture–capacity interactions invalidate conclusions drawn from undersized proxy models |
| Automated mixture optimization methods serve as useful priors but do not replace empirical validation | DoReMi, RHO-LOSS, and RegMix provide principled starting points but converge to suboptimal solutions in complex, multi-objective settings |
| Comprehensive evaluation across all target domains is non-negotiable | Single-domain evaluation creates blind spots for cross-domain regression introduced by mixture changes |
| The mixture design problem has no universal solution | Optimal proportions are functions of model capacity, training budget, source quality distributions, and application-specific performance requirements |

---

## 10. Conclusion

Data curation and mixture optimization constitute the most consequential—and most empirically demanding—decisions in the LLM pretraining pipeline. The formal framework presented in this report characterizes mixture design as a constrained multi-objective optimization problem on the probability simplex $\Delta^{K-1}$, subject to quality–quantity–repetition trade-offs, capacity-dependent cross-domain interference, and recency-biased gradient dynamics. The current state of the art, as demonstrated by production-scale models including SmolLM3, relies on systematic ablation campaigns executed at target model scale, augmented by multi-stage curriculum strategies that concentrate the highest-quality data in the annealing phase where its impact on final model behavior is maximized. While automated approaches offer theoretically motivated initialization heuristics, the irreducible complexity of the mixture–performance mapping continues to necessitate rigorous empirical exploration guided by clearly articulated capability priorities and comprehensive evaluation frameworks.

---

## References

- Allal, L. B., et al. (2025). SmolLM2: Efficient small language models through multi-stage training.
- Chen, Y., et al. (2025b). Recency effects in large language model pretraining.
- Liu, Q., et al. (2025). RegMix: Optimizing data mixture through regularized regression.
- Mindermann, S., et al. (2022). Prioritized training on points that are learnable, worth learning, and not yet learned. *ICML*.
- Muennighoff, N., et al. (2025). Scaling data-constrained language models. *NeurIPS*.
- Xie, S. M., et al. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. *NeurIPS*.