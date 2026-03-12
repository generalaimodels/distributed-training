

# SmolLM3: Data Mixture Curation for Multi-Domain Pretraining

## A Scientific Technical Report on Principled Data Composition for a 3B-Parameter Language Model

---

## 1. Problem Formulation and Objective

Training a general-purpose large language model (LLM) requires solving a constrained multi-objective optimization problem over the data mixture space. Given a fixed compute budget $C$ (measured in FLOPs or equivalently in total training tokens $T_{\text{total}}$), the goal is to determine a distribution $\mathcal{D}_{\text{mix}}$ over $K$ constituent data sources $\{D_1, D_2, \ldots, D_K\}$ such that downstream performance across $M$ target capability domains is jointly maximized.

Formally, let $\alpha_i$ denote the sampling weight assigned to data source $D_i$, where $\sum_{i=1}^{K} \alpha_i = 1$ and $\alpha_i \geq 0$. The mixture distribution is:

$$
\mathcal{D}_{\text{mix}} = \sum_{i=1}^{K} \alpha_i \cdot D_i
$$

The optimization objective is:

$$
\boldsymbol{\alpha}^{*} = \arg\max_{\boldsymbol{\alpha} \in \Delta^{K-1}} \sum_{j=1}^{M} w_j \cdot \mathcal{P}_j\left(\theta^{*}(\boldsymbol{\alpha})\right)
$$

where:

- $\Delta^{K-1}$ denotes the $(K-1)$-dimensional probability simplex
- $\theta^{*}(\boldsymbol{\alpha})$ represents the model parameters obtained after training on $\mathcal{D}_{\text{mix}}$ for $T_{\text{total}}$ tokens
- $\mathcal{P}_j(\cdot)$ is the performance metric on evaluation benchmark $j$
- $w_j$ is the importance weight for domain $j$

For SmolLM3, the target capability domains were defined as:

| Domain Index $j$ | Capability | Representative Benchmarks |
|---|---|---|
| 1 | English language understanding | HellaSwag, ARC-Challenge, MMLU |
| 2 | Multilingual proficiency | Multilingual benchmarks across FR, ES, DE, IT, PT + secondary languages |
| 3 | Mathematical reasoning | Math-specific evaluation suites |
| 4 | Code generation | Code generation and understanding benchmarks |

**Key constraint:** The total compute budget was fixed, meaning the total number of training tokens $T_{\text{total}}$ was predetermined. Any increase in tokens allocated to one domain necessarily reduces allocation to others. This introduces an inherent **zero-sum tradeoff** across domains:

$$
\sum_{i=1}^{K} \alpha_i \cdot T_{\text{total}} = T_{\text{total}} \implies \text{domain allocations are mutually competing}
$$

An additional constraint arises from **finite source sizes**. If a source $D_i$ contains $|D_i|$ unique tokens, then the number of epochs $E_i$ a source undergoes during training is:

$$
E_i = \frac{\alpha_i \cdot T_{\text{total}}}{|D_i|}
$$

Excessive repetition (high $E_i$) leads to overfitting and diminishing returns, establishing an upper bound on each $\alpha_i$ as a function of source size and total training duration.

---

## 2. Methodological Framework: Ablation-Driven Mixture Design

### 2.1 General Methodology

The data mixture design for SmolLM3 followed a systematic, empirically-grounded pipeline consisting of the following stages:

1. **Candidate Source Identification** — Enumerate all available high-quality datasets per target domain.
2. **Intra-Domain Selection** — For domains with multiple candidate sources, run controlled ablations to select or rank datasets.
3. **Inter-Domain Ratio Optimization** — Sweep over domain-level proportions $(\alpha_{\text{web}}, \alpha_{\text{multilingual}}, \alpha_{\text{code}}, \alpha_{\text{math}})$ via ablation experiments on the target model architecture.
4. **Ablation Validation** — Evaluate each candidate mixture on the full benchmark suite, comparing against controlled baselines with statistical rigor.
5. **Stage-Wise Refinement** — Apply annealing ablation methodology for later training stages, testing incremental dataset additions against a stable checkpoint.

This approach is **architecture-agnostic** and **domain-agnostic**: the identical pipeline applies whether the target is a low-resource language, a vertical domain (e.g., finance, healthcare), or a specialized capability (e.g., long-context reasoning).

### 2.2 Leveraging Prior Work: The Transfer from SmolLM2

SmolLM3 did not start from a tabula rasa. The predecessor model, SmolLM2, had already established a validated data recipe at 1.7B parameters, covering:

- Identification of optimal English web data sources
- Ranking of available open-source math and code datasets
- Initial domain ratio calibration

The scientific hypothesis underlying the SmolLM3 mixture design was:

> **H₁:** The optimal intra-domain dataset rankings and approximate inter-domain ratios identified at 1.7B parameters transfer to 3B parameters, with fine-grained ratio recalibration achievable through targeted ablation sweeps at the new scale.

This hypothesis is grounded in empirical observations from scaling laws literature (Hoffmann et al., 2022; Kaplan et al., 2020), which suggest that data quality rankings tend to be scale-invariant, while optimal data-to-parameter ratios may shift as model capacity increases.

---

## 3. English Web Data: The Foundation Layer

### 3.1 Source Selection

Web-crawled text constitutes the foundational corpus for any general-purpose LLM, providing broad coverage of world knowledge, linguistic diversity, and reasoning patterns. However, raw web data is highly heterogeneous in quality; aggressive filtering and curation are prerequisites for effective pretraining.

Two open-source English web datasets were identified as state-of-the-art at the time of training:

| Dataset | Description | Token Count | Strengths |
|---|---|---|---|
| **FineWeb-Edu** | Educationally filtered subset of FineWeb, enriched for pedagogical and STEM content | Component of 5.1T combined | Educational benchmarks, STEM reasoning |
| **DCLM** | DataComp for Language Models; a curated, quality-filtered web corpus | Component of 5.1T combined | Common-sense reasoning, general knowledge |

Combined, these two sources yielded approximately **5.1 trillion tokens** ($5.1 \times 10^{12}$) of high-quality English web data.

### 3.2 Intra-Source Ratio Optimization

The two sources exhibit complementary strengths: FineWeb-Edu biases toward educational and STEM content, improving performance on knowledge-intensive benchmarks, while DCLM provides broader common-sense coverage. The optimal mixing ratio between them is not trivially 50/50 and must be determined empirically.

**Ablation Design:**

- **Model:** 3B-parameter SmolLM3 architecture
- **Training Horizon:** 100B tokens ($10^{11}$ tokens) per ablation run
- **Controlled Variable:** FineWeb-Edu to DCLM ratio
- **Tested Ratios:** $\{20/80,\ 40/60,\ 50/50,\ 60/40,\ 80/20\}$
- **Evaluation:** Full benchmark suite spanning educational/STEM, common-sense, and general knowledge tasks

**Results:**

Let $r = \frac{\alpha_{\text{FWE}}}{\alpha_{\text{FWE}} + \alpha_{\text{DCLM}}}$ denote the FineWeb-Edu fraction within the English web sub-mixture. The ablation sweep produced the following qualitative findings:

- At $r = 0.8$: strong STEM and educational performance; degraded common-sense reasoning
- At $r = 0.2$: strong common-sense reasoning; diminished educational benchmark performance
- At $r \in \{0.5, 0.6\}$: **optimal Pareto front** across all evaluated benchmarks

The selected ratio for Stage 1 pretraining was:

$$
r^{*} = 0.50 \quad \Longrightarrow \quad \alpha_{\text{FWE}} : \alpha_{\text{DCLM}} = 50 : 50
$$

This finding demonstrated **cross-scale consistency** with SmolLM2 results, confirming that the optimal intra-domain ratio is relatively invariant between 1.7B and 3B parameter scales.

### 3.3 Supplementary English Sources

In addition to the two primary sources, several supplementary datasets were incorporated:

- **peS2o** (Perez et al.) — Scientific papers and academic text
- **Wikipedia & Wikibooks** — Encyclopedic, well-structured factual content
- **StackExchange** — Technical question-answer pairs across diverse domains

**Observed impact:** These supplementary sources did **not** yield statistically significant improvements on the benchmark suite. However, they were retained in the mixture for **distributional diversity**, serving as a regularization mechanism against over-specialization on the two dominant web corpora. The underlying rationale is that distributional breadth in pretraining data reduces the risk of narrow failure modes and improves robustness to out-of-distribution inputs during downstream fine-tuning.

---

## 4. Multilingual Web Data Integration

### 4.1 Language Selection and Source Composition

Multilingual capability was a first-class objective for SmolLM3, extending beyond the English-only scope of SmolLM2. The language selection followed a tiered strategy:

**Tier 1 — Primary Non-English Languages (5 languages):**

| Language | ISO Code | Source |
|---|---|---|
| French | $\texttt{fr}$ | FineWeb2-HQ |
| Spanish | $\texttt{es}$ | FineWeb2-HQ |
| German | $\texttt{de}$ | FineWeb2-HQ |
| Italian | $\texttt{it}$ | FineWeb2-HQ |
| Portuguese | $\texttt{pt}$ | FineWeb2-HQ |

**Tier 2 — Secondary Languages (~10 languages):**

Languages including Chinese ($\texttt{zh}$), Arabic ($\texttt{ar}$), and Russian ($\texttt{ru}$), sourced from the standard (non-HQ) FineWeb2 pipeline. These were included at lower sampling ratios, **not** targeting state-of-the-art multilingual performance, but providing sufficient signal to enable efficient **continual pretraining** by downstream users who wish to specialize SmolLM3 for these languages.

**Total multilingual corpus size:** approximately **628 billion tokens** ($6.28 \times 10^{11}$).

### 4.2 The Multilingual-English Tradeoff

Introducing multilingual data into a fixed-budget training run creates a direct tension with English performance. Let $\beta$ denote the fraction of the total web data allocated to non-English content. The tradeoff can be characterized as:

$$
\mathcal{P}_{\text{en}}(\beta) \approx \mathcal{P}_{\text{en}}(0) - \lambda_{\text{en}} \cdot \beta \quad \text{(English degradation, approximately linear for small } \beta \text{)}
$$

$$
\mathcal{P}_{\text{multi}}(\beta) \approx \mathcal{P}_{\text{multi}}(0) + \lambda_{\text{multi}} \cdot \beta \quad \text{(Multilingual improvement)}
$$

where $\lambda_{\text{en}}$ and $\lambda_{\text{multi}}$ are empirical sensitivity coefficients. The goal is to find $\beta^{*}$ where multilingual gains are substantial without material English regression.

**Ablation Results:**

Through ablation experiments on the 3B-parameter architecture, the optimal multilingual content ratio was determined to be:

$$
\beta^{*} = 0.12 \quad \text{(12\% of web data is non-English)}
$$

This corresponds to approximately **14% of the overall web sub-mixture** (accounting for the domain-level allocation of web data within the total mixture). At this ratio:

- Multilingual benchmark performance improved significantly across all five Tier 1 languages
- English benchmark performance (HellaSwag, ARC-C, MMLU) exhibited **no statistically significant degradation**

### 4.3 Repetition Constraint Analysis

The asymmetry in corpus size between English ($5.1 \times 10^{12}$ tokens) and multilingual ($6.28 \times 10^{11}$ tokens) imposed a natural ceiling on $\beta$. At the selected ratio, the effective epoch count for the multilingual corpus during Stage 1 training (estimated at $T_{\text{stage1}} \approx 8\text{-}9 \times 10^{12}$ tokens) can be computed as:

$$
E_{\text{multi}} = \frac{\beta \cdot \alpha_{\text{web}} \cdot T_{\text{stage1}}}{|D_{\text{multi}}|}
$$

For representative values ($\beta = 0.12$, $\alpha_{\text{web}} \approx 0.75$, $T_{\text{stage1}} \approx 8.5 \times 10^{12}$, $|D_{\text{multi}}| = 6.28 \times 10^{11}$):

$$
E_{\text{multi}} \approx \frac{0.12 \times 0.75 \times 8.5 \times 10^{12}}{6.28 \times 10^{11}} \approx 1.22 \text{ epochs}
$$

This is well within acceptable repetition bounds. Increasing $\beta$ substantially beyond 0.12 would push $E_{\text{multi}}$ toward multi-epoch territory, introducing overfitting risk on the smaller multilingual corpus without commensurate gains.

---

## 5. Code Data Integration

### 5.1 Source Composition

Code data was curated from the StarCoder2 and The Stack v2 training ecosystems, comprising the following sub-sources:

| Sub-Source | Content Type | Rationale |
|---|---|---|
| **The Stack v2 (16 languages)**, filtered as StarCoder2Data | Production source code across 16 programming languages | Core code syntax and semantics |
| **StarCoder2 GitHub Pull Requests** | Code review diffs, comments, discussions | Real-world code review reasoning and iterative refinement |
| **Jupyter & Kaggle Notebooks** | Executable, step-by-step computational workflows | Procedural reasoning, data analysis patterns |
| **GitHub Issues** | Bug reports, feature requests, technical discussions | Contextual problem-solving around codebases |
| **StackExchange (code-related)** | Technical Q&A threads | Explanatory code reasoning and debugging |

### 5.2 Code Ratio Ablation

Empirical evidence from prior work (Aryabumi et al., 2024) establishes that code data improves LLM performance **beyond coding tasks**, enhancing natural language reasoning, structured thinking, and world knowledge. This motivated an initial ablation starting point of 25% code in the total mixture.

**Ablation Design and Results:**

Let $\gamma$ denote the fraction of the total training mixture allocated to code data.

| $\gamma$ | English Benchmarks (HellaSwag, ARC-C, MMLU) | Code Benchmarks | Observation |
|---|---|---|---|
| 0.25 | **Significant degradation** | Strong | Excessive code displaces English web data, harming general capabilities |
| 0.10 | No improvement over $\gamma = 0$ | Moderate | Code capability acquired without English benchmark regression |
| 0.00 | Baseline | None | No code capability |

**Selected ratio:**

$$
\gamma^{*} = 0.10 \quad \text{(10\% of total mixture)}
$$

**Interpretation:** At $\gamma = 0.25$, the displacement of English web data (which must decrease proportionally under the fixed token budget constraint) caused unacceptable degradation on core English benchmarks. At $\gamma = 0.10$, the model acquired code generation capabilities without measurable harm to English performance. Although code data at 10% did not provide the auxiliary reasoning benefits reported by Aryabumi et al. (which were observed at 25%), code generation was deemed a sufficiently critical capability to justify inclusion regardless.

### 5.3 Staged Introduction of High-Quality Code Data

A curated educationally-filtered subset of StarCoder2Data, designated **Stack-Edu**, was deliberately withheld from Stage 1 pretraining. This follows the **data staging principle**: reserving the highest-quality, most concentrated data sources for later training stages where they have maximal impact on final model performance. The rationale is grounded in the observation that:

$$
\frac{\partial \mathcal{P}}{\partial \text{data quality}} \bigg|_{t \to T_{\text{total}}} > \frac{\partial \mathcal{P}}{\partial \text{data quality}} \bigg|_{t \to 0}
$$

That is, the marginal performance gain from high-quality data is greater during late-stage training (when the model has already acquired broad foundational representations) than during early training (when the model is learning basic distributional statistics).

---

## 6. Math Data Integration

### 6.1 Source Composition and Staging Strategy

Mathematical data followed a progressive quality escalation strategy across training stages:

**Stage 1 — Broad Coverage, Moderate Quality:**

| Dataset | Quality Tier | Description |
|---|---|---|
| **FineMath3+** | $\geq 3$ educational score | General mathematical web content, lower quality threshold |
| **InfiWebMath3+** | $\geq 3$ educational score | Complementary math web corpus, lower quality threshold |

**Later Stages — Concentrated Quality and Specialized Sources:**

| Dataset | Quality Tier | Description |
|---|---|---|
| **FineMath4+** | $\geq 4$ educational score | Higher quality subset of FineMath |
| **InfiWebMath4+** | $\geq 4$ educational score | Higher quality subset of InfiWebMath |
| **MegaMath** (Zhou et al., 2025) | Curated | Large-scale curated mathematical corpus |
| **OpenMathInstruct** (Toshniwal et al., 2024) | Instruction-tuned | Mathematical instruction-following data |
| **OpenMathReasoning** (Moshkov et al., 2025) | Reasoning chains | Mathematical reasoning with step-by-step solutions |

### 6.2 Math Ratio Selection for Stage 1

Let $\delta$ denote the fraction of the total Stage 1 mixture allocated to math data. The available math corpus for Stage 1 (FineMath3+ combined with InfiWebMath3+) totaled approximately **54 billion tokens** ($5.4 \times 10^{10}$).

Given an estimated Stage 1 training horizon of $T_{\text{stage1}} \approx 8\text{-}9 \times 10^{12}$ tokens, the epoch count at any given $\delta$ is:

$$
E_{\text{math}} = \frac{\delta \cdot T_{\text{stage1}}}{|D_{\text{math}}|} = \frac{\delta \times 8.5 \times 10^{12}}{5.4 \times 10^{10}} \approx 157.4 \cdot \delta
$$

**Repetition analysis at candidate $\delta$ values:**

| $\delta$ | $E_{\text{math}}$ (approx.) | Assessment |
|---|---|---|
| 0.03 | ~4.7 epochs | Acceptable; moderate repetition |
| 0.05 | ~7.9 epochs | High repetition; overfitting risk |
| 0.10 | ~15.7 epochs | Excessive; significant overfitting risk |

**Selected ratio:**

$$
\delta^{*} = 0.03 \quad \text{(3\% of total mixture)}
$$

The math allocation was equally split between the two sources:

$$
\alpha_{\text{FineMath3+}} = \alpha_{\text{InfiWebMath3+}} = \frac{\delta^{*}}{2} = 0.015
$$

At $\delta = 0.03$, the effective epoch count of approximately 4.7 represents a pragmatic upper bound — sufficient to impart foundational mathematical reasoning without inducing overfitting artifacts from excessive data repetition. Note that this relatively low fraction does not preclude strong math performance; higher-quality math data is introduced in subsequent training stages via the staged data quality escalation strategy.

---

## 7. Stage 1 Mixture Summary

Synthesizing the ablation results across all four domains, the finalized Stage 1 pretraining mixture for SmolLM3 was:

$$
\boldsymbol{\alpha}_{\text{stage1}} = \left(\alpha_{\text{web}},\ \alpha_{\text{multilingual}},\ \alpha_{\text{code}},\ \alpha_{\text{math}}\right) = (0.75,\ 0.12,\ 0.10,\ 0.03)
$$

**Detailed domain decomposition:**

| Domain | Proportion | Token Budget (at $T_{\text{stage1}} = 8.5\text{T}$) | Primary Sources |
|---|---|---|---|
| English Web | 75% | ~6.375T | FineWeb-Edu (50%), DCLM (50%), peS2o, Wikipedia, StackExchange |
| Multilingual Web | 12% | ~1.020T | FineWeb2-HQ (Tier 1), FineWeb2 (Tier 2) |
| Code | 10% | ~0.850T | StarCoder2Data, PRs, Notebooks, Issues, StackExchange |
| Math | 3% | ~0.255T | FineMath3+ (50%), InfiWebMath3+ (50%) |

**Verification of the simplex constraint:**

$$
0.75 + 0.12 + 0.10 + 0.03 = 1.00 \quad \checkmark
$$

---

## 8. Annealing Ablation Methodology for Multi-Stage Training

### 8.1 Motivation

While Stage 1 mixture ratios were determined via **from-scratch ablation sweeps** (training ablation models from random initialization), this approach becomes prohibitively expensive for evaluating candidate datasets in later stages. Each from-scratch ablation requires $O(10^{11})$ tokens of training, and the number of candidate datasets grows as new sources become available.

**Annealing ablations** provide a computationally efficient alternative: they evaluate candidate datasets by fine-tuning a late-stage checkpoint for a relatively short horizon, dramatically reducing the cost per experiment while preserving the informational signal about dataset utility.

### 8.2 Protocol Specification

The annealing ablation protocol for SmolLM3 was defined as follows:

**Checkpoint Selection:**

- A checkpoint at approximately **7 trillion tokens** ($7 \times 10^{12}$) into Stage 1 training was selected as the base model. This checkpoint represents a near-converged Stage 1 model with stable representations.

**Annealing Mixture Composition:**

For each candidate dataset $D_{\text{candidate}}$, an annealing mixture was constructed as:

$$
\mathcal{D}_{\text{anneal}} = 0.40 \cdot \mathcal{D}_{\text{stage1}} + 0.60 \cdot D_{\text{candidate}}
$$

where:

- $\mathcal{D}_{\text{stage1}}$ is the **exact** Stage 1 mixture (preserving the 75/12/10/3 domain split), serving as a **stability anchor** to prevent catastrophic forgetting of previously acquired capabilities
- $D_{\text{candidate}}$ is the new dataset under evaluation
- The 40/60 split allocates the majority of the annealing budget to the candidate, maximizing the signal-to-noise ratio for detecting the candidate's impact

**Training Duration:**

Each annealing experiment was conducted over **50 billion tokens** ($5 \times 10^{10}$), which is:

- Large enough to produce measurable benchmark deltas
- Small enough to permit rapid iteration over multiple candidates
- Approximately 0.6% of the total Stage 1 training budget, ensuring computational efficiency

**Evaluation:**

After each annealing run, the resulting checkpoint was evaluated on the full benchmark suite. Performance deltas $\Delta \mathcal{P}_j$ relative to a baseline annealing run (using 100% $\mathcal{D}_{\text{stage1}}$) were computed for each benchmark $j$:

$$
\Delta \mathcal{P}_j = \mathcal{P}_j(\theta_{\text{anneal}}^{\text{candidate}}) - \mathcal{P}_j(\theta_{\text{anneal}}^{\text{baseline}})
$$

A candidate dataset was accepted for inclusion in later stages if:

$$
\sum_{j \in \mathcal{J}_{\text{target}}} w_j \cdot \Delta \mathcal{P}_j > 0 \quad \text{and} \quad \max_{j \in \mathcal{J}_{\text{non-target}}} |\Delta \mathcal{P}_j^{-}| < \epsilon
$$

where $\mathcal{J}_{\text{target}}$ is the set of benchmarks relevant to the candidate's domain, $\mathcal{J}_{\text{non-target}}$ captures all other benchmarks, $\Delta \mathcal{P}_j^{-}$ denotes negative deltas only, and $\epsilon$ is the acceptable regression threshold.

### 8.3 Exemplar: MegaMath Evaluation

As a concrete illustration, the evaluation of MegaMath (Zhou et al., 2025) for inclusion in later training stages followed this protocol:

1. **Checkpoint:** 7T-token Stage 1 checkpoint
2. **Annealing Mixture:** 40% Stage 1 mixture (75/12/10/3 split preserved) + 60% MegaMath
3. **Training:** 50B tokens
4. **Evaluation:** Full benchmark suite with focus on math-specific metrics

The positive $\Delta \mathcal{P}_j$ on math benchmarks, combined with negligible regression on English, multilingual, and code benchmarks, validated MegaMath for inclusion in subsequent stages.

---

## 9. Multi-Stage Training Architecture

### 9.1 Staging Philosophy

SmolLM3's training was decomposed into three sequential stages, each with a distinct data composition strategy. The underlying principle is a **curriculum learning** framework adapted to pretraining:

$$
\text{Stage } s: \quad \mathcal{D}_{\text{mix}}^{(s)} = \sum_{i=1}^{K_s} \alpha_i^{(s)} \cdot D_i^{(s)}
$$

where the number of sources $K_s$, their identities $D_i^{(s)}$, and their weights $\alpha_i^{(s)}$ vary across stages according to the following design principles:

| Stage | Duration (est.) | Data Philosophy | Quality Profile |
|---|---|---|---|
| **Stage 1** | ~8–9T tokens | Broad coverage, diverse distribution | Moderate quality threshold (e.g., score $\geq 3$) |
| **Stage 2** | Shorter | Increase domain-specific concentration | Higher quality threshold (e.g., score $\geq 4$); introduce validated new sources |
| **Stage 3** | Shortest (annealing) | Maximum quality concentration | Highest quality subsets; instruction/reasoning data |

This progression follows the principle that:

1. **Early training** benefits from **distributional breadth** — the model must learn the statistical structure of language, code, and mathematics at a foundational level.
2. **Mid-stage training** benefits from **domain concentration** — upsampling high-quality data in target domains refines capabilities.
3. **Late-stage training** (annealing) benefits from the **highest-quality, most curated data** — the model's representations are sufficiently mature to extract maximal signal from concentrated, high-fidelity sources.

### 9.2 Stage Transitions: Data Quality Escalation

The math domain provides a clear illustration of the quality escalation strategy:

$$
\text{Stage 1:} \quad \text{FineMath3+, InfiWebMath3+} \quad (\text{broader coverage, lower quality floor})
$$

$$
\text{Stage 2/3:} \quad \text{FineMath4+, InfiWebMath4+, MegaMath, OpenMathInstruct, OpenMathReasoning}
$$

Similarly for code:

$$
\text{Stage 1:} \quad \text{StarCoder2Data (standard)} \quad (\text{broad code coverage})
$$

$$
\text{Stage 2/3:} \quad \text{Stack-Edu (educationally filtered)} \quad (\text{high-quality, didactic code})
$$

The staged introduction ensures that the model does not "waste" high-quality data during early training when its representations are insufficiently developed to fully leverage the quality signal.

---

## 10. Theoretical Analysis of Key Design Decisions

### 10.1 The Compute-Allocation Tradeoff

Under a fixed compute budget, the data mixture optimization is subject to the **allocation constraint**. Let the total training FLOPs be $C$, and let the model have $N$ parameters. By the Chinchilla scaling law (Hoffmann et al., 2022), the optimal total token count satisfies:

$$
T_{\text{opt}} \approx 20 \cdot N
$$

For $N = 3 \times 10^{9}$ (3B parameters):

$$
T_{\text{opt}} \approx 6 \times 10^{10} \text{ tokens}
$$

However, SmolLM3 was trained for significantly more tokens than the Chinchilla-optimal estimate (estimated 8–9T tokens in Stage 1 alone), placing it in the **over-trained** regime. This is a deliberate choice for inference-efficient models: over-training a smaller model reduces per-query compute cost at deployment while approaching the performance of a larger, Chinchilla-optimal model. This decision amplifies the importance of data mixture quality, as the model has more capacity to memorize artifacts from poor-quality sources during extended training.

### 10.2 Data Repetition and Diminishing Returns

The relationship between performance and epoch count $E$ on a finite dataset follows a saturating curve:

$$
\mathcal{P}(E) \approx \mathcal{P}_{\infty} - (\mathcal{P}_{\infty} - \mathcal{P}_0) \cdot e^{-\kappa E}
$$

where $\mathcal{P}_{\infty}$ is the asymptotic performance, $\mathcal{P}_0$ is the initial performance, and $\kappa$ is the learning rate of the dataset. Beyond a critical epoch count $E_{\text{crit}}$, additional repetition yields negligible improvement and risks memorization-induced overfitting:

$$
E > E_{\text{crit}} \implies \frac{\partial \mathcal{P}}{\partial E} \approx 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_{\text{val}}}{\partial E} > 0
$$

This analysis directly informed the 3% math allocation and 12% multilingual allocation decisions, where corpus sizes constrained the maximum practical $\alpha$ values.

### 10.3 Cross-Domain Transfer Effects

Code data's beneficial effect on non-code tasks (Aryabumi et al., 2024) can be understood through the lens of **representational transfer**: code enforces precise logical structure, variable tracking, and sequential reasoning, producing internal representations that transfer to natural language reasoning tasks. Formally, if $\phi_{\text{code}}$ and $\phi_{\text{NL}}$ denote the feature spaces learned from code and natural language respectively, effective code pretraining produces:

$$
\text{dim}(\phi_{\text{code}} \cap \phi_{\text{NL}}) \gg 0
$$

implying substantial shared structure. However, the SmolLM3 ablations revealed that this transfer effect has a **non-monotonic dependency on $\gamma$**: at $\gamma = 0.25$, the displacement of English web data overwhelmed the transfer benefit, producing a net negative effect on English benchmarks.

---

## 11. Summary of Empirical Findings and Design Principles

### 11.1 Key Quantitative Results

| Design Decision | Ablation Range | Optimal Value | Key Observation |
|---|---|---|---|
| FineWeb-Edu / DCLM ratio | 20/80 to 80/20 | **50/50** | Consistent with SmolLM2 findings; scale-invariant |
| Multilingual fraction $\beta$ | Variable | **12%** of web data | No English degradation; constrained by corpus size |
| Code fraction $\gamma$ | 0%, 10%, 25% | **10%** | 25% causes English regression; 10% acquires code capability |
| Math fraction $\delta$ | Constrained by 54B tokens | **3%** | Epoch count ceiling; high-quality data deferred to later stages |

### 11.2 Generalizable Design Principles

The SmolLM3 data mixture curation process yields the following transferable principles for practitioners:

1. **Ablation-driven ratio selection is essential.** Theoretical predictions and prior heuristics (e.g., "use 25% code") can serve as starting points, but target-architecture-specific ablation sweeps must validate all domain ratios.

2. **Corpus size constrains sampling weight.** The maximum practical $\alpha_i$ for any source is bounded by the acceptable epoch count: $\alpha_i \leq \frac{E_{\text{max}} \cdot |D_i|}{T_{\text{total}}}$.

3. **Data quality staging amplifies late-training impact.** Reserving the highest-quality subsets for later stages produces greater marginal performance gains than uniform mixing throughout training.

4. **Annealing ablations enable efficient multi-stage design.** Evaluating candidate datasets via short annealing runs on a late-stage checkpoint dramatically reduces the compute cost of mixture optimization for subsequent stages.

5. **Supplementary sources serve as regularizers.** Datasets that do not individually improve benchmarks can still contribute positively through distributional diversity and implicit regularization effects.

6. **Cross-domain transfer is real but non-monotonic.** Benefits from code or math data on general NLP tasks exist but saturate and reverse at high domain fractions due to English web data displacement.

---

## 12. Conclusion

The data mixture curation for SmolLM3 demonstrates that principled, ablation-driven data composition is a critical determinant of LLM pretraining success — arguably as important as architectural choices or training hyperparameters. Through systematic sweeps over intra-domain ratios (FineWeb-Edu/DCLM), inter-domain proportions (English/multilingual/code/math), and staged quality escalation, the final mixture achieved balanced performance across English language understanding, multilingual proficiency, mathematical reasoning, and code generation within a fixed compute budget.

The methodology — identify candidates, ablate ratios, validate on benchmarks, stage quality — constitutes a general-purpose framework applicable to any domain-targeted pretraining scenario, from low-resource language specialization to vertical domain adaptation in finance, healthcare, or scientific research.

---

## References

- Aryabumi, V., et al. (2024). Code improves language model performance beyond coding tasks.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models. *NeurIPS*.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*.
- Moshkov, N., et al. (2025). OpenMathReasoning.
- Toshniwal, S., et al. (2024). OpenMathInstruct.
- Zhou, C., et al. (2025). MegaMath.