# Beyond Base Models: A Comprehensive Technical Report on Post-Training Methodologies for Large Language Models in 2025

## Abstract

This report presents an end-to-end technical exposition of post-training methodologies for large language models (LLMs), with a detailed case study on the development of SmolLM3-3B as a state-of-the-art hybrid reasoning model. Post-training encompasses the full spectrum of techniques applied after pretraining—supervised fine-tuning (SFT), continued pretraining (mid-training), preference optimization (PO), and reinforcement learning with verifiable rewards (RLVR)—that collectively transform a raw language model from a next-token predictor into a reliable, steerable, and deployable system. We formalize the post-training compass framework (Why → What → How), detail the construction of evaluation suites, present rigorous hyperparameter ablations across SFT, preference optimization, and GRPO-based reinforcement learning, and document engineering challenges including chat template design, GPU failure mitigation, data pipeline debugging, and reward hacking in hybrid reasoning models. The report consolidates empirical findings, mathematical formulations, and actionable engineering principles derived from training thousands of model checkpoints.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Post-Training Compass: Strategic Framework](#2-post-training-compass-strategic-framework)
3. [Hybrid Reasoning Model Definition](#3-hybrid-reasoning-model-definition)
4. [Evaluation Infrastructure](#4-evaluation-infrastructure)
5. [Post-Training Frameworks and Tooling](#5-post-training-frameworks-and-tooling)
6. [Supervised Fine-Tuning (SFT)](#6-supervised-fine-tuning-sft)
7. [Continued Pretraining (Mid-Training)](#7-continued-pretraining-mid-training)
8. [Preference Optimization](#8-preference-optimization)
9. [Reinforcement Learning with Verifiable Rewards](#9-reinforcement-learning-with-verifiable-rewards)
10. [On-Policy Alternatives to Full RL](#10-on-policy-alternatives-to-full-rl)
11. [Consolidated Engineering Principles](#11-consolidated-engineering-principles)
12. [Conclusion](#12-conclusion)

---

## 1. Introduction and Motivation

### 1.1 The Pretraining–Post-Training Dichotomy

Pretraining endows a language model with broad distributional knowledge by optimizing the standard autoregressive cross-entropy objective over trillions of tokens:

$$\mathcal{L}_{\text{pretrain}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

where $\theta$ denotes model parameters, $x_t$ is the token at position $t$, and $T$ is the sequence length. This objective forces the model to internalize syntactic, semantic, and world knowledge from a massive web-scale corpus. However, the resulting base model is fundamentally a text completion engine: it predicts statistically likely continuations rather than producing responses aligned with user intent, safety constraints, or structured reasoning protocols.

Post-training bridges this gap. It encompasses all optimization procedures applied after pretraining to sculpt the raw distributional capability into a system that is:

- **Steerable**: responsive to user instructions, system prompts, and reasoning mode directives.
- **Reliable**: consistent in output quality across diverse domains and conversation turns.
- **Capable**: equipped with enhanced reasoning, tool-calling, and multilingual competencies that exceed what pretraining alone provides.

If pretraining is characterized as brute-force knowledge acquisition via gradient descent over web-scale corpora, post-training is the precision engineering phase that converts that latent capacity into operational utility.

### 1.2 Scope and Contributions

This report provides a complete technical treatment of the post-training pipeline as applied to SmolLM3-3B, a 3-billion-parameter dense transformer trained by Hugging Face. The contributions are:

1. **A formalized post-training compass** (Why → What → How) that provides strategic clarity before committing compute resources.
2. **A comprehensive evaluation framework** spanning capability evals, integrated task evals, overfitting prevention evals, internal evals, and vibe evals, with quantified justifications for each benchmark selection.
3. **Rigorous SFT ablations** covering learning rate, packing, masking, epoch scaling, and chat template design for hybrid reasoning models.
4. **Continued pretraining (mid-training) analysis** demonstrating that domain-specific mid-training on reasoning corpora yields multiplicative gains when composed with SFT.
5. **Preference optimization methodology** comparing DPO, IPO, KTO, ORPO, APO-zero, APO-down, and DiscoPOP, with hyperparameter sensitivity analysis over learning rate, $\beta$, and dataset scale.
6. **RLVR experimentation with GRPO** on hybrid reasoning models, including the identification and mitigation of reward hacking via overlong completion penalties.
7. **Engineering principles** distilled from training thousands of checkpoints, addressing GPU failure resilience, data pipeline validation, numerical precision, and inference compatibility.

---

## 2. Post-Training Compass: Strategic Framework

### 2.1 The Why → What → How Framework

Before allocating compute to post-training, three sequential questions must be answered with precision:

#### 2.1.1 Why Post-Train?

Three canonical motivations exist, mirroring those of pretraining:

| Motivation | Description | Example |
|---|---|---|
| **Research** | Exploring whether a specific technique (e.g., RLVR) can unlock new capabilities in an existing model | Investigating if reinforcement learning can elicit emergent chain-of-thought behaviors |
| **Production** | Distilling a large model into a smaller one for latency, cost, or deployment constraints | Compressing a 70B model to 3B via knowledge distillation for edge deployment |
| **Strategic Open Source** | Filling a gap where no strong open-weight model exists for a specific use case | Creating the first fully open hybrid reasoning model recipe at the 3B scale |

**Critical pre-commitment checks:**

1. **Necessity assessment**: Many open-weight models now rival proprietary systems across diverse tasks. Quantized variants can execute locally with modest compute. If a generalist assistant suffices, an off-the-shelf model from the Hugging Face Hub may already satisfy requirements—rendering post-training unnecessary.
2. **Data availability**: Post-training yields maximal returns when targeting specific tasks or domains where generalist models underperform and high-quality domain-specific data is accessible.
3. **Measurability**: Without clearly defined evaluation criteria, it is impossible to determine whether post-training produces genuine improvement. Success metrics must be established before training commences.

#### 2.1.2 What Should Post-Training Achieve?

The target capability profile determines the entire downstream pipeline. Representative objectives include:

- A **crisp instruction follower** that rarely drifts off-topic.
- A **versatile assistant** capable of switching tones, roles, and personas on demand.
- A **reasoning engine** that solves mathematical, coding, and agentic problems with verifiable correctness.
- A **multilingual system** that maintains quality across diverse languages.
- A **hybrid reasoner** that dynamically switches between concise direct responses and extended step-by-step reasoning.

#### 2.1.3 How Will You Get There?

The "how" maps to specific training recipes:

| Method | Purpose |
|---|---|
| Supervised Fine-Tuning (SFT) | Instill core instruction-following and conversational capabilities |
| Continued Pretraining (Mid-Training) | Strengthen foundational skills (e.g., reasoning, coding) before SFT |
| Preference Optimization (PO) | Learn directly from human or AI preferences to refine quality |
| Reinforcement Learning (RL) | Optimize for verifiable objectives beyond supervised data |
| Data Curation | Balance diversity, quality, and coverage across training stages |
| Evaluation | Track progress, catch regressions, and prevent overfitting |

### 2.2 Application to SmolLM3

For SmolLM3-3B, the compass resolved as follows:

- **Why**: A pretrained base model required post-training before release. Simultaneously, hybrid reasoning models (e.g., Qwen3) were gaining adoption, yet fully open training recipes were scarce. SmolLM3 addressed both goals: preparing a production-ready model and contributing a transparent recipe to the Pareto front alongside Qwen3-1.7B and Qwen3-4B.
- **What**: A hybrid reasoning model tailored to SmolLM3's strengths—specifically, reasoning quality that generalizes across languages beyond English, with tool-calling and long-context workflow support as core requirements.
- **How**: Detailed in the subsequent sections of this report.

### 2.3 Ablation Strategy: Post-Training vs. Pretraining

A critical distinction between pretraining and post-training ablation methodologies exists:

| Dimension | Pretraining Ablation | Post-Training Ablation |
|---|---|---|
| **"Small" means** | Smaller models and datasets | Smaller datasets and simpler algorithms |
| **Base model variation** | Commonly varied (proxy models) | Almost never varied; behavior is too model-dependent |
| **Run duration** | Prohibitively expensive at scale | Short enough to iterate on the target model directly |

The primary exception to the "never vary the base model" rule arises when using off-the-shelf base models from public repositories, where a model trained on $1 \times 10^{12}$ tokens exhibits fundamentally different behavior from one trained on $10 \times 10^{12}$ tokens, even at identical parameter counts.

---

## 3. Hybrid Reasoning Model Definition

### 3.1 Formal Definition

A **hybrid reasoning model** operates in two distinct inference modes:

1. **Concise mode** (`/no_think`): The model produces direct, succinct responses without explicit intermediate reasoning steps.
2. **Extended reasoning mode** (`/think`): The model generates a step-by-step chain of thought (CoT) enclosed within designated tags (e.g., `<think>...</think>`) before producing a final answer.

The operating mode is set explicitly by the user via the system message. Following the convention established by Qwen3, lightweight commands control mode selection:

- `/think` invokes extended reasoning.
- `/no_think` enforces concise answers.

This design grants the user explicit control over the depth-speed trade-off at inference time.

### 3.2 Architectural Implications

Hybrid reasoning does not require architectural modifications to the transformer. Instead, the behavioral distinction is induced entirely through:

1. **Training data composition**: Paired examples across both modes teach the model when and how to switch behaviors.
2. **Chat template design**: Structural tokens (e.g., `<think>`, `</think>`) delineate reasoning content from final responses.
3. **System prompt conditioning**: Mode-specific instructions in the system message steer generation.

The fundamental challenge is that training data must be **paired across modes**: each example must unambiguously indicate whether extended reasoning or concise answering is expected. This pairing constraint introduces additional complexity compared to standard SFT, where datasets can be mixed freely.

---

## 4. Evaluation Infrastructure

### 4.1 Design Principles

Post-training evaluation inherits the core principles from pretraining evaluation:

- **Monotonicity**: Genuine improvements should produce monotonically increasing eval scores.
- **Low noise**: Evaluations must exhibit sufficiently low variance to support reliable decision-making.
- **Above-random signal**: Benchmarks must discriminate model capability above chance-level performance.
- **Ranking consistency**: Relative rankings between models should remain stable across evaluation runs.

Additionally, post-training evaluations must test **behavioral** properties—instruction following, alignment, tool use, multilingual competence—that are absent from pretraining evaluations.

### 4.2 Evaluation Taxonomy

#### 4.2.1 Capability Evals

These target fundamental cognitive and knowledge-based skills:

| Domain | Benchmark | Description | Saturation Status at 3B |
|---|---|---|---|
| **Knowledge** | GPQA Diamond (Rein et al., 2024) | Graduate-level multiple-choice scientific reasoning | Far from saturated |
| **Knowledge** | SimpleQA (Wei et al., 2024) | Factuality assessment | Extremely challenging for small models |
| **Mathematics** | AIME 2025 | Competition-level mathematics | Active benchmark |
| **Mathematics** | MATH-500 (Lightman et al., 2023) | Mathematical problem solving | Largely saturated by reasoning models |
| **Code** | LiveCodeBench (latest version) | Competitive programming problems | Improvements translate to better coding |
| **Code** | SWE-bench Verified | Software engineering tasks | Typically too difficult for 3B models |
| **Multilinguality** | Global MMLU (Singh et al., 2025) | Multilingual question answering | Primary multilingual benchmark |
| **Multilinguality** | MGSM (Shi et al., 2022) | Multilingual mathematical reasoning | Supplementary multilingual eval |

#### 4.2.2 Integrated Task Evals

These test composite capabilities representative of real-world deployment:

| Capability | Benchmark | Description | Limitations |
|---|---|---|---|
| **Long Context** | NIAH (Kamradt, 2023) | Needle-in-a-haystack retrieval | Too superficial for discrimination |
| **Long Context** | RULER (Hsieh et al., 2024), HELMET (Yen et al., 2025) | Comprehensive long-context understanding | More discriminative alternatives |
| **Long Context** | MRCR, GraphWalks (OpenAI) | Extended difficulty long-context evals | Recent additions |
| **Instruction Following** | IFEval (J. Zhou et al., 2023) | Verifiable instruction constraints | Subject to benchmaxxing |
| **Instruction Following** | IFBench (Pyatkin et al., 2025), Multi-IF (He et al., 2024) | Diverse and multi-turn constraints | Mitigates benchmaxxing |
| **Alignment** | AlpacaEval (Dubois et al., 2025), ArenaHard (T. Li et al., 2024), MixEval (Ni et al., 2024) | LLM-as-judge proxy for human preferences | MixEval has strongest correlation with human Elo |
| **Tool Calling** | BFCL v3 | Function calling comprehensiveness | Often saturated quickly |
| **Tool Calling** | TAU-Bench (Barres et al., 2025) | Simulated customer service tool use | More realistic tool-use assessment |

#### 4.2.3 Overfitting Prevention Evals

To detect overfitting to specific skills or benchmark distributions:

- **GSMPlus** (Q. Li et al., 2024): Perturbed variants of GSM8k problems that test whether models generalize to structurally similar but numerically different problems.

#### 4.2.4 Internal and Vibe Evals

- **Internal evals**: Custom benchmarks targeting specific capabilities absent from public suites. For SmolLM3, a multi-turn reasoning eval (variant of Multi-IF) was implemented to test whether the model maintained consistent reasoning mode switching across conversation turns.
- **Vibe evals**: Interactive testing of intermediate checkpoints by human evaluators to uncover subtle behavioral quirks not captured by automated metrics. As documented in Section 6.5, vibe testing uncovered a critical data processing bug in SmolLM3's training pipeline.

### 4.3 SmolLM3 Evaluation Suite

The following evaluation suite was selected for SmolLM3 development:

| Benchmark | Category | Number of Prompts | Metric |
|---|---|---|---|
| AIME25 | Competitive mathematics | 30 | avg@64 |
| LiveCodeBench v4 (v5 for final release) | Competitive programming | 100 (268) | avg@16 |
| GPQA Diamond | Graduate-level reasoning | 198 | avg@8 |
| IFEval | Instruction following | 541 | Accuracy |
| MixEval Hard | Alignment | 1,000 | Accuracy |
| BFCL v3 | Tool use | 4,441 | Mixed |
| Global MMLU (lite for validation) | Multilingual Q&A | 590,000 (6,400) | Accuracy |
| GSMPlus (mini for validation) | Robustness | 10,000 (2,400) | Accuracy |
| RULER | Long context | 6,500 | Accuracy |

For benchmarks with small problem counts (typically $< 2000$), $k$ samples are drawn per problem and the avg@$k$ metric is reported to mitigate sampling noise.

### 4.4 Evaluation Engineering Rules

The following operational rules govern evaluation throughout model development:

1. **Use small correlated subsets** during rapid iteration. For instance, LiveCodeBench v4 correlates highly with v5 at half the runtime. Methods from tinyBenchmarks (Polo et al., 2024) identify minimal prompt subsets that reliably approximate full evaluations.

2. **For reasoning models, strip the chain of thought from scored output.** This eliminates false positives and prevents benchmarks like IFEval from penalizing responses that violate constraints embedded within the reasoning trace (e.g., "write a poem in under 50 words" violated by a 500-token CoT).

3. **Pin the LLM judge model and version** for apples-to-apples comparisons. Preferably, use an open-weight judge model to ensure reproducibility even after provider deprecation.

4. **Decontaminate training data** against evaluation benchmarks using $n$-gram matching to prevent overfitting, particularly when synthetic data generation is employed.

5. **Treat ablation benchmarks as validation, not test.** Reserve a set of held-out benchmarks exclusively for final model reports.

6. **Always include vibe evals** on proprietary data and tasks to detect overfitting to public suites.

7. **Verify new eval implementations** by replicating published results of reference models within acceptable error margins before deploying the eval at scale.

8. **Inspect evaluation data directly** when results are ambiguous—examine what the model is actually being prompted with.

---

## 5. Post-Training Frameworks and Tooling

### 5.1 Framework Landscape

The following table summarizes the feature support of major post-training frameworks:

| Framework | SFT | PO | RL | Multi-modal | FullFT | LoRA | Distributed |
|---|---|---|---|---|---|---|---|
| TRL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Axolotl | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenInstruct | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Unsloth | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| veRL | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prime RL | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| PipelineRL | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| ART | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| TorchForge | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| NemoRL | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| OpenRLHF | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

**Key terminology:**
- **FullFT** (Full Fine-Tuning): All model parameters $\theta$ are updated during training.
- **LoRA** (Low-Rank Adaptation): Only small low-rank matrices $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$ are updated, while the base model parameters remain frozen.
- **Multi-modal**: Support for training on modalities beyond text (e.g., images, audio).
- **Distributed**: Support for training across multiple GPUs via data parallelism, FSDP, DeepSpeed ZeRO, or context parallelism.

### 5.2 The Case for Frameworks

A persistent argument in the research community advocates implementing all training infrastructure from scratch. This position is untenable for production-grade post-training:

1. **Algorithmic correctness**: RL algorithms like PPO and GRPO are notoriously sensitive to implementation details. Subtle errors in normalization, KL penalty computation, or advantage estimation can waste days of compute (Huang et al., 2024).

2. **Scalability**: A single-file implementation that works at 1B parameters will not scale to 100B+ parameters without distributed training primitives (sharding, gradient accumulation, mixed-precision casting).

3. **Reproducibility**: Battle-tested frameworks encode hard-won engineering knowledge that is costly to rediscover independently.

The optimal workflow is to select a mature framework, understand its internals, and maintain an internal fork for rapid experimentation. New features are developed on the fork and upstreamed to the main library after validation.

---

## 6. Supervised Fine-Tuning (SFT)

### 6.1 Why SFT Remains the Foundation

Despite the prominence of reinforcement learning in contemporary discourse, supervised fine-tuning remains the indispensable first stage of virtually every effective post-training pipeline. The justification is threefold:

1. **Computational efficiency**: SFT requires modest compute relative to RL. Meaningful gains are achievable without the massive rollout budgets that RL demands, typically completing in a fraction of the time.

2. **Training stability**: Unlike RL, which exhibits sensitivity to reward design, KL coefficients, and clipping parameters, SFT optimizes a well-understood cross-entropy objective that converges reliably.

3. **Baseline quality**: A strong SFT checkpoint captures the majority of target performance gains and makes subsequent methods (DPO, RLHF, GRPO) far more effective by providing a high-quality initialization.

The SFT objective for training on assistant responses is:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t \in \mathcal{A}} \log P_\theta(x_t \mid x_{<t})$$

where $\mathcal{A}$ denotes the set of token positions corresponding to assistant responses, and user/system tokens are masked from the loss computation.

**Exception—DeepSeek-R1-Zero**: At the frontier, where no stronger model exists for distillation and human annotations are too noisy for complex behaviors like long chain-of-thought reasoning, skipping SFT and proceeding directly to RL can discover reasoning behaviors that standard supervision cannot teach. This regime represents an extreme case that does not generalize to most practitioners.

### 6.2 Base Model Selection

When selecting a base model for post-training, the following dimensions govern the decision:

| Dimension | Considerations |
|---|---|
| **Model size** | Larger models generalize better with fewer samples. Select a size representative of deployment constraints. |
| **Architecture (MoE vs. Dense)** | Mixture-of-Experts (MoE) models activate a subset of parameters per token, offering higher capacity per FLOP. Dense models are simpler to fine-tune and often outperform MoEs at smaller scales ($< 10$B parameters). |
| **Post-training track record** | Models with a demonstrated ecosystem of strong fine-tuned variants (e.g., community-produced instruct models) provide evidence of training amenability. |

Empirically, the base model families from Qwen, Mistral, and DeepSeek have proven most amenable to post-training, with Qwen being particularly favorable due to its broad parameter range (0.6B to 235B), which simplifies scaling experiments.

### 6.3 Data Composition for Hybrid Reasoning Baselines

For SmolLM3, the initial SFT data mixture targeted reasoning, instruction following, and steerability across both reasoning modes:

| Dataset | Reasoning Mode | # Examples | % Examples | # Tokens (M) | % Tokens | Avg. Tokens/Example | Avg. Context Tokens | Avg. Response Tokens | Avg. Turns |
|---|---|---|---|---|---|---|---|---|---|
| Everyday Conversations | `/no_think` | 2,260 | 2.3 | 0.6 | 0.8 | 260.2 | 222.3 | 94.0 | 7.8 |
| SystemChats 30k | `/no_think` | 33,997 | 35.2 | 21.5 | 28.2 | 631.9 | 422.8 | 267.7 | 6.3 |
| Tulu 3 SFT Personas IF | `/no_think` | 29,970 | 31.0 | 13.3 | 17.5 | 444.5 | 119.8 | 380.7 | 2.0 |
| Everyday Conversations (Qwen3-32B) | `/think` | 2,057 | 2.1 | 3.1 | 4.1 | 1,522.4 | 376.8 | 1,385.6 | 4.0 |
| SystemChats 30k (Qwen3-32B) | `/think` | 27,436 | 28.4 | 29.4 | 38.6 | 1,070.8 | 84.6 | 1,042.7 | 2.0 |
| s1k-1.1 | `/think` | 835 | 0.9 | 8.2 | 10.8 | 8,859.3 | 370.9 | 9,728.5 | 2.0 |
| **Total** | — | **96,555** | **100.0** | **76.1** | **100.0** | **2,131.5** | **266.2** | **2,149.9** | **4.0** |

**Critical observation on data balancing**: Data mixtures for hybrid reasoning models must be balanced by **token count**, not by **example count**. The s1k-1.1 dataset constitutes only $\sim 1\%$ of total examples but accounts for $\sim 11\%$ of total tokens due to its long reasoning responses. Balancing by examples would severely underweight this dataset's contribution to gradient updates.

### 6.4 Chat Template Design

#### 6.4.1 Design Considerations

The chat template governs how multi-turn conversations, system prompts, reasoning traces, and tool calls are serialized into token sequences. The following requirements informed the SmolLM3 template design:

| Requirement | Description |
|---|---|
| **System role customization** | Users must be able to define arbitrary system prompts (e.g., persona definitions) |
| **Tool calling support** | Structured outputs for API calls and tool responses must be accommodated |
| **Reasoning mode control** | `<think>...</think>` tags must separate reasoning content from final answers |
| **Code agent support** | Execution of arbitrary Python code (not just JSON tool calls) must be supported |
| **Inference engine compatibility** | Parsers in vLLM and SGLang must correctly handle the template structure |

#### 6.4.2 Comparison of Existing Templates

| Chat Template | System Role | Tools | Reasoning | Inference Compat. | Notes |
|---|---|---|---|---|---|
| ChatML | ✅ | ✅ | ❌ | ✅ | Simple, broadly applicable |
| Qwen3 | ✅ | ✅ | ✅ | ✅ | Hybrid reasoning template |
| DeepSeek-R1 | ❌ | ❌ | ✅ | ✅ | Prefills reasoning with `<think>` |
| Llama 3 | ✅ | ✅ | ❌ | ✅ | Built-in Python interpreter tools |
| Gemma 3 | ✅ | ❌ | ❌ | ❌ | System role at first user turn |
| Command A Reasoning | ✅ | ✅ | ✅ | ❌ | Multiple templates per model |
| gpt-oss | ✅ | ✅ | ✅ | ✅ | Based on Harmony format; complex |

#### 6.4.3 SmolLM3 Template Design Decision

Qwen3's template was selected as the starting point due to its balance across all dimensions. However, one design limitation was identified: **Qwen3 discards reasoning content for all but the final turn** in a multi-turn conversation. While this is sensible for inference (to prevent context window exhaustion), for training it is essential to **retain reasoning tokens across all turns** to properly condition the model. The SmolLM3 template was therefore extended to:

- Include a structured system prompt with metadata (knowledge cutoff, date, reasoning mode).
- Support code agent execution via Python tools.
- Provide explicit reasoning mode control through the system message.
- Retain reasoning traces across all conversation turns during training.

### 6.5 Baby Baselines: Validation Before Optimization

#### 6.5.1 Experimental Setup

The initial baseline experiments served as sanity checks to validate that the chat template elicited hybrid reasoning correctly and that hyperparameters produced stable training. Three data mixture variants were compared:

| Variant | Data Source |
|---|---|
| **Instruct** | Non-reasoning (`/no_think`) examples only |
| **Thinking** | Reasoning (`/think`) examples only |
| **Hybrid** | All examples from both modes |

Training configuration:
- **Base model**: SmolLM3-3B-Base
- **Fine-tuning method**: Full fine-tuning (FullFT)
- **Learning rate**: $1 \times 10^{-5}$
- **Effective batch size**: 128
- **Epochs**: 1
- **Sequence length**: 8,192 tokens (Instruct), 32,768 tokens (Thinking, Hybrid)
- **Packing**: Disabled (small datasets)
- **Hardware**: 1 node × 8 H100 GPUs
- **Runtime**: 30–90 minutes per experiment

#### 6.5.2 Key Finding: Split-Brain Behavior

The baseline experiments revealed that hybrid reasoning models exhibit **split-brain behavior**: the data mixture for one reasoning mode has negligible effect on the other mode's performance. In most evaluations, scores remained similar across the Instruct, Thinking, and Hybrid subsets. Exceptions included LiveCodeBench v4 and IFEval, where hybrid data produced synergistic improvements, suggesting that certain benchmarks benefit from cross-mode training.

#### 6.5.3 Data Pipeline Bug Discovery via Vibe Testing

Despite acceptable evaluation scores, interactive vibe testing of the hybrid baseline uncovered a critical failure: the model consistently ignored system message content (e.g., persona instructions like "act like a pirate"). Root cause analysis revealed a bug in the data processing pipeline:

The `custom_instructions` field in the chat template kwargs was set to `None` for all training samples, which caused the template to substitute the default SmolLM3 system prompt instead of the intended persona-specific instructions. This was particularly damaging for the SystemChats subset, where all persona definitions are conveyed through `custom_instructions`.

**Impact**: The model learned to associate any conversation context with the default system prompt, producing random character switches mid-conversation. The bug had **zero impact on automated evaluation scores**, underscoring the indispensability of vibe testing.

**Principle**: Always vibe-test models even when evaluations appear satisfactory. Automated benchmarks cannot capture all behavioral failure modes.

### 6.6 Targeting Multi-Turn Reasoning

#### 6.6.1 Problem Identification

Training on single-turn reasoning data fails to generalize to multi-turn reasoning contexts. This is an expected distributional mismatch: absent multi-turn reasoning examples, the model is evaluated outside its training distribution.

To quantify this, a **ThinkFollow** evaluation was implemented (inspired by Qwen3's internal eval), which randomly inserts `/think` or `/no_think` tags across turns in a multi-turn conversation and checks whether the model generates appropriate (empty or non-empty) think blocks. The hybrid baseline failed catastrophically beyond the first turn:

| Turn | Qwen3-1.7B | Hybrid Baseline |
|---|---|---|
| Turn 1 | ~95% | ~85% |
| Turn 2 | ~90% | ~20% |
| Turn 3 | ~85% | ~10% |

#### 6.6.2 IFThink Dataset Construction

To address this capability gap, the **IFThink** dataset was constructed using the following pipeline:

1. **Source prompts**: Single-turn instructions from Tulu 3's instruction-following subset.
2. **Multi-turn expansion**: Qwen3-32B generated verifiable follow-up instructions across multiple turns.
3. **Reasoning trace generation**: Qwen3-32B generated reasoning traces for each turn.
4. **Mode annotation**: Each turn was annotated with the appropriate reasoning mode (`/think` or `/no_think`).

Including IFThink in the training mixture produced dramatic improvements:

| Turn | Qwen3-1.7B | Hybrid Baseline w/ IFThink |
|---|---|---|
| Turn 1 | ~95% | ~95% |
| Turn 2 | ~90% | ~90% |
| Turn 3 | ~85% | ~85% |

### 6.7 Hyperparameter Analysis

#### 6.7.1 Masking User Turns

In standard SFT, the loss can be computed over all tokens or restricted to assistant tokens only. Masking user turns prevents the model from learning to autocomplete user queries and focuses optimization exclusively on response quality.

Formally, let a training sequence consist of interleaved user tokens $\mathcal{U}$ and assistant tokens $\mathcal{A}$. The masked SFT loss is:

$$\mathcal{L}_{\text{masked-SFT}}(\theta) = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log P_\theta(x_t \mid x_{<t})$$

In TRL, masking is implemented via the `{% generation %}` keyword in the Jinja2 chat template, which returns an `assistant_masks` tensor indicating which tokens contribute to the loss:

```
assistant_masks[t] = \begin{cases} 1 & \text{if } t \in \mathcal{A} \\ 0 & \text{if } t \in \mathcal{U} \cup \mathcal{S} \end{cases}
```

where $\mathcal{S}$ denotes system tokens.

**Empirical impact for SmolLM3**: Masking provided modest improvements (a few percentage points) across most benchmarks, with the most significant effect on **IFEval** in `/no_think` mode, likely because masking reduces the model's tendency to restate prompts and thereby improves adherence to verifiable constraints.

#### 6.7.2 Sequence Packing

**Problem**: SFT datasets contain variable-length samples. Without packing, each batch includes substantial padding, wasting compute and slowing convergence.

**Solution**: Sequence packing concatenates multiple samples until a target maximum token length is reached. TRL employs a **best-fit decreasing** strategy (Ding et al., 2024), which orders sequences by length to minimize truncation at batch boundaries while reducing padding tokens.

**Throughput analysis**: Packing improved training throughput by a factor of $3\text{–}5\times$ compared to unpacked training. With packing enabled, the number of non-padding tokens per batch scales linearly with batch size, achieving up to $33\times$ more tokens per optimization step compared to unpacked training.

**Performance trade-off**: Packing alters training dynamics by reducing the number of gradient updates per epoch (more tokens per step, fewer steps per epoch). This trade-off is most pronounced on small datasets. Specifically:

- At effective batch sizes $\leq 32$: Packing provides equivalent or improved performance.
- At effective batch sizes $> 32$: An average performance drop is observed, with IFEval exhibiting the most significant degradation.

**Recommendation**: For large-scale SFT, packing is almost always beneficial. For small or domain-specific datasets, start with packing enabled, monitor downstream evaluations, and disable packing if performance degrades.

#### 6.7.3 Learning Rate

The learning rate is the single most impactful hyperparameter in SFT after the data mixture itself. The optimal SFT learning rate is typically **one or more orders of magnitude smaller** than the pretraining learning rate, because aggressive updates to a pretrained model with rich representations risk catastrophic forgetting.

**Scan protocol**: An initial scan over $\{1 \times 10^{-6}, 3 \times 10^{-6}, 1 \times 10^{-5}, 3 \times 10^{-5}, 1 \times 10^{-4}\}$ covers two orders of magnitude and allows identification of the optimal region for finer-grained tuning.

**SmolLM3 results**: Learning rates of $3 \times 10^{-6}$ and $1 \times 10^{-5}$ yielded the best average performance across both reasoning modes. Learning rates above $1 \times 10^{-5}$ produced dramatic degradation on individual benchmarks (e.g., AIME25 performance collapsed).

**Interaction with packing**: Higher learning rates become more destabilizing when packing is enabled, as each optimization step processes more tokens. A slight reduction in learning rate is recommended when using packing.

#### 6.7.4 Epoch Scaling

After identifying an optimal data mixture and learning rate, increasing the number of training epochs can extract additional performance:

- Training for 5 epochs on the SmolLM3 baseline mixture yielded several additional percentage points on average.
- The impact was benchmark-specific: LiveCodeBench v4 with extended thinking nearly **doubled** in performance between epoch 1 and epoch 5.
- Diminishing returns are expected beyond a certain point, and overfitting monitoring via held-out evaluations is essential.

### 6.8 Optimizer Selection

AdamW remains the default optimizer for post-training. An open question exists regarding optimizer consistency between pretraining and post-training: the Kimi team demonstrated that using the same optimizer (Muon) for both stages yielded optimal performance for their Moonlight model. The AdamW update rule is:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

where $g_t$ is the gradient, $\beta_1, \beta_2$ are exponential decay rates for moment estimates, $\eta$ is the learning rate, $\epsilon$ is a numerical stability constant, and $\lambda$ is the weight decay coefficient.

---

## 7. Continued Pretraining (Mid-Training)

### 7.1 Rationale

Continued pretraining (mid-training) extends a base model's training on large-scale domain-specific corpora **before** SFT. This approach shifts the model's internal distribution toward a domain (e.g., mathematical reasoning, code) that better supports downstream fine-tuning, enabling SFT to focus on task-specific behaviors rather than learning core skills from scratch.

The technique traces back to **ULMFiT** (Howard & Ruder, 2018), which pioneered the three-stage pipeline:

$$\text{General Pretraining} \rightarrow \text{Domain-Specific Mid-Training} \rightarrow \text{Task-Specific Post-Training}$$

This architecture has been adopted by modern systems including FAIR's Code World Model and Phi-4-Mini-Reasoning.

### 7.2 When Mid-Training is Beneficial

Mid-training is most effective when:

1. The target capabilities for SFT share a **common core skill** (e.g., mathematical reasoning, code understanding).
2. The base model has **not already received sufficient exposure** to the target domain during pretraining.
3. The volume of domain-specific data justifies the additional compute cost (typically billions of tokens).

Mid-training is **less useful** when:

1. The base model already possesses the target skill.
2. The desired behaviors are shallow (style, conversational chit-chat), which can be efficiently instilled via SFT alone.
3. Compute would be better allocated to preference optimization or RL.

### 7.3 Experimental Setup for SmolLM3

Three mid-training corpora were evaluated:

| Dataset | Source Model | Samples | Tokens | Description |
|---|---|---|---|---|
| Mixture of Thoughts | DeepSeek-R1 | 350K | — | Math, code, science reasoning (reserved for SFT stage) |
| Llama-Nemotron-Post-Training-Dataset | Multiple (filtered for DeepSeek-R1) | 3.64M | 18.7B | Large-scale distilled reasoning data |
| OpenThoughts3-1.2M | QwQ-32B | 1.2M | 16.5B | High-quality reasoning distillation |

**Training configuration**:
- Chat template: ChatML (to avoid premature commitment to the SmolLM3 template)
- Epochs: 5
- Learning rate: $2 \times 10^{-5}$
- Hardware: 8 nodes of H100 GPUs
- Effective batch size: 128

### 7.4 GPU Failure and Numerical Stability

#### 7.4.1 Hardware Failures

The mid-training runs encountered systematic GPU throttling on aging hardware, producing frequent hardware failures and forced restarts. The training loss curves exhibited discontinuities corresponding to each restart.

Initial debugging attributed failures to DeepSpeed's throughput optimizations. A switch to data parallelism reduced hardware failures but introduced a **dramatic loss divergence**.

#### 7.4.2 Numerical Precision Bug

Root cause analysis revealed that data parallelism in Hugging Face Accelerate stored weights and gradients in the model's native precision (BF16), leading to **numerical instability** during gradient accumulation and optimization. BF16 has only 8 bits of mantissa precision, which is insufficient for accurate gradient accumulation over large effective batch sizes.

The standard solution—employed by DeepSpeed ZeRO and FSDP—is to maintain **FP32 master weights and optimizer states**, casting to BF16 only for forward and backward passes:

$$\theta_{\text{master}}^{(t+1)} = \theta_{\text{master}}^{(t)} - \eta \cdot \text{OptimizerStep}\left(\text{Cast}_{\text{FP32}}\left(\nabla_\theta \mathcal{L}\right)\right)$$
$$\theta_{\text{BF16}}^{(t+1)} = \text{Cast}_{\text{BF16}}\left(\theta_{\text{master}}^{(t+1)}\right)$$

#### 7.4.3 Resolution

The final mitigation strategy combined:
1. Return to DeepSpeed ZeRO-3 for proper mixed-precision handling.
2. Aggressive checkpointing at frequent intervals.
3. Automatic restart on hardware failure.
4. Remote checkpoint storage (Hugging Face Hub) to prevent accidental overwrites.

**Engineering Principle**: Save model checkpoints frequently and push to remote storage. Ensure the training framework is robust to failures and supports automatic restarts. These strategies are essential for long-running jobs.

### 7.5 Results

#### 7.5.1 Mid-Training Comparison

Across the three corpora and their combination:

| Configuration | AIME25 | GPQA-D | LiveCodeBench v4 | IFEval |
|---|---|---|---|---|
| Llama-Nemotron | Best individual | Good | Good | Baseline |
| OpenThoughts3 | Below Nemotron | Below Nemotron | Below Nemotron | Baseline |
| Combined Mix | **Best overall** | **Best overall** | **Best overall** | Baseline |

The Llama-Nemotron dataset provided stronger individual performance, but the combination of both corpora yielded the best aggregate results.

#### 7.5.2 Impact on SFT

Applying the baseline SFT data mixture to the mid-trained checkpoint (vs. the pretrained checkpoint) produced transformative improvements:

| Benchmark | Mode | Pretrained → SFT | Mid-trained → SFT | Relative Gain |
|---|---|---|---|---|
| AIME25 | `/think` | Baseline | ~3× improvement | ~200% |
| LiveCodeBench v4 | `/think` | Baseline | ~3× improvement | ~200% |
| GPQA-D | `/think` | Baseline | +10 points | ~30% |
| Reasoning benchmarks | `/no_think` | Baseline | +4–6 points | ~15–25% |

**Key insight**: Mid-training on reasoning corpora produces multiplicative (not merely additive) gains when composed with SFT, and the reasoning core partially transfers to the concise (`/no_think`) mode even though the mid-training data consisted entirely of reasoning traces.

**Conclusion**: For reasoning models, mid-training on domain-specific reasoning corpora almost always makes sense when the base model has not already received extensive reasoning data during pretraining.

---

## 8. Preference Optimization

### 8.1 Motivation: Beyond Imitation Learning

SFT is fundamentally a form of **imitation learning**: the model learns to reproduce patterns present in the training data. This creates an inherent ceiling:

1. If the training data lacks examples of self-correction, the model cannot learn to fix its own errors.
2. Even when training data contains a mix of successful and unsuccessful traces, the model may learn that making initial errors is part of the desired pattern, rather than learning to produce correct solutions from the start.

Preference optimization overcomes this by providing **comparative feedback**: "response $y_c$ is better than response $y_r$." This signal directly optimizes for quality and enables performance to scale beyond the limits of imitation learning.

Additionally, preference optimization typically requires **far less data** than SFT because the starting model already follows instructions and possesses domain knowledge from prior training stages.

### 8.2 Preference Dataset Construction

#### 8.2.1 Strong vs. Weak Approach

Given a fixed set of prompts $\{x_i\}_{i=1}^N$:

1. Generate one response from a weaker/baseline model: $y_r^{(i)} \sim \pi_{\text{weak}}(\cdot \mid x_i)$
2. Generate one response from a stronger model: $y_c^{(i)} \sim \pi_{\text{strong}}(\cdot \mid x_i)$
3. Label the stronger model's output as chosen and the weaker one as rejected.

This produces a dataset $\mathcal{D} = \{(x_i, y_c^{(i)}, y_r^{(i)})\}_{i=1}^N$.

#### 8.2.2 On-Policy with Grading

1. Use the model being trained to generate $K$ candidate responses per prompt: $\{y_k^{(i)}\}_{k=1}^K \sim \pi_\theta(\cdot \mid x_i)$
2. An external grader (verifier or reward model) scores each response along quality axes.
3. Preference labels are assigned based on grader scores.

This method produces on-policy data that reflects the model's current distribution, enabling iterative bootstrapping as the model improves.

#### 8.2.3 SmolLM3 Preference Data

At the time of SmolLM3 development, no preference datasets with reasoning traces existed. A "strong vs. weak" dataset was constructed using:

- **Prompts**: Ai2's Tulu 3 preference mixture.
- **Strong model**: Qwen3-32B (in `/think` mode).
- **Weak model**: Qwen3-0.6B (in `/think` mode).
- **Result**: 250K+ LLM-generated preference pairs.

### 8.3 Algorithm Comparison

#### 8.3.1 Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2024) directly optimizes the policy to satisfy preference constraints without training a separate reward model. The DPO loss is:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right]$$

where $\sigma(\cdot)$ is the sigmoid function, $\pi_{\text{ref}}$ is the reference policy (typically the SFT checkpoint), and $\beta$ controls the trade-off between matching preferences and staying close to the reference.

#### 8.3.2 Alternative Algorithms

| Algorithm | Key Modification | Use Case |
|---|---|---|
| **IPO** (Identity Preference Optimization) | Replaces sigmoid with squared hinge loss | More robust to noisy preferences |
| **KTO** (Kahneman–Tversky Optimization, Ethayarajh et al., 2024) | Models individual response desirability without pairs | Applicable when only thumbs-up/thumbs-down labels are available |
| **ORPO** (Odds Ratio PO, Hong et al., 2024) | Integrates PO into SFT via odds ratio augmentation of cross-entropy loss | Eliminates need for separate SFT stage and reference model |
| **APO** (Anchored PO, D'Oosterlinck et al., 2024) | Explicitly regularizes likelihood shifts for chosen vs. rejected | More controllable alignment dynamics; variants APO-zero and APO-down |
| **DiscoPOP** | Discovery of optimal PO loss | Learned loss function |

#### 8.3.3 SmolLM3 Algorithm Ablation

Using the Tulu 3 Preference Personas IF dataset, six algorithms were compared on IFEval:

| Algorithm | IFEval (` /no_think`) Gain over SFT |
|---|---|
| DPO | Moderate improvement |
| IPO | Moderate improvement |
| **APO-zero** | **+15–20 percentage points** |
| APO-down | Moderate improvement |
| DiscoPOP | Moderate improvement |

APO-zero also exhibited the best out-of-domain performance and was selected for all subsequent SmolLM3 preference optimization experiments.

**Key finding**: Preference optimization improves not only helpfulness and alignment but also **reasoning quality**. Generating "strong vs. weak" preferences and ablating loss functions can yield significant reasoning gains over vanilla DPO.

### 8.4 Hyperparameter Sensitivity Analysis

#### 8.4.1 Learning Rate

The optimal preference optimization learning rate is typically **$10\text{–}100\times$ smaller** than the SFT learning rate. For SmolLM3 (SFT learning rate $= 2 \times 10^{-5}$):

- Scan range: $\{1 \times 10^{-7}, 5 \times 10^{-7}, 1 \times 10^{-6}, 5 \times 10^{-6}, 1 \times 10^{-5}\}$
- **Optimal range**: $\sim 10\times$ smaller than SFT learning rate (approximately $1 \times 10^{-6}$ to $5 \times 10^{-6}$).
- Learning rates beyond this $10\times$ limit produced **worse performance than the SFT checkpoint** in extended thinking mode.
- The `/no_think` mode exhibited more stable behavior across learning rates.

**Recommendation**: Scan at $5\times$ to $20\times$ smaller than the SFT learning rate. The optimal value almost certainly lies within this range.

#### 8.4.2 $\beta$ Parameter

The $\beta$ parameter controls the margin between preference pairs. Lower $\beta$ encourages staying close to the reference policy; higher $\beta$ permits greater divergence toward the preference data.

- Scan range: $\{0.01, 0.05, 0.10, 0.50, 1.00\}$
- **Optimal value**: $\beta = 0.1$ yielded the highest performance for both reasoning modes and improved over the SFT checkpoint.
- $\beta < 0.1$ (e.g., $\beta = 0.01$) hurt performance, producing a model worse than the SFT checkpoint.
- Performance without extended thinking remained stable across multiple $\beta$ values.

**Interpretation**: Values greater than 0.1 are preferable, suggesting that aligning with preference data is more beneficial than staying close to the reference policy. However, very high $\beta$ values may erase SFT-acquired capabilities not captured by the evaluation suite.

**Recommendation**: Explore $\beta \in [0.01, 0.5]$.

#### 8.4.3 Dataset Scale

Experiments tested dataset sizes from 2K to 340K preference pairs:

- Performance remained **stable across the entire range**.
- Performance drops in extended thinking mode appeared beyond 100K pairs, but were less pronounced than those induced by learning rate mistuning.
- Smaller datasets (e.g., 2K–10K) already produced improvements over the SFT checkpoint.

**Implication**: During rapid iteration, smaller preference datasets suffice for identifying promising configurations. Full-scale datasets can be reserved for final training runs.

### 8.5 Preference Optimization Rules of Engagement

1. **Generate custom preference data.** With inference costs approaching negligible levels, constructing LLM preferences from multiple providers is straightforward and cost-effective.
2. **Start with DPO as the baseline** and iterate toward alternatives (ORPO, KTO, APO) based on data characteristics and empirical performance.
3. **Use a learning rate $\sim 10\times$ smaller than SFT.**
4. **Scan $\beta$ in the range $[0.01, 0.5]$.**
5. **Most preference algorithms overfit after one epoch.** Partition data and train iteratively for best results.
6. **Preference optimization is only as good as the offline data.** When static datasets exhaust their signal, transition to on-policy methods.

---

## 9. Reinforcement Learning with Verifiable Rewards

### 9.1 When RL is Justified

Reinforcement learning becomes the appropriate optimization strategy when:

1. **Automatic correctness verification** is available (unit tests, mathematical proofs, API calls, or high-quality verifiers/reward models).
2. **The task requires multi-step reasoning or planning**, where local preferences may not capture long-horizon success.
3. **Optimization objectives extend beyond preference labels** (e.g., passing all unit tests, maximizing a measurable objective function).

### 9.2 RLHF vs. RLVR

| Property | RLHF | RLVR |
|---|---|---|
| **Reward Source** | Learned reward model trained on human preferences | Deterministic verifier checking correctness criteria |
| **Failure Mode** | Reward hacking via out-of-distribution sequences | Reward hacking via length exploitation |
| **Scalability** | Limited by annotation cost and reward model generalization | Scales with availability of verifiable problems |
| **Popularized By** | InstructGPT (Ouyang et al., 2022) | DeepSeek-R1 (DeepSeek-AI, 2025) |

### 9.3 GRPO: Group Relative Policy Optimization

GRPO is an on-policy optimization algorithm where the model (policy) that generates completions is the same model being optimized. For each prompt $x$, $G$ completions $\{y_1, \ldots, y_G\}$ are sampled from the current policy $\pi_\theta$, and a reward $R(x, y_g)$ is computed for each. The advantage is estimated relative to the group:

$$\hat{A}_g = \frac{R(x, y_g) - \text{mean}(\{R(x, y_j)\}_{j=1}^G)}{\text{std}(\{R(x, y_j)\}_{j=1}^G) + \epsilon}$$

The GRPO objective maximizes:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_g\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{g=1}^{G} \min \left( r_g(\theta) \hat{A}_g, \; \text{clip}(r_g(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) \hat{A}_g \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

where $r_g(\theta) = \frac{\pi_\theta(y_g \mid x)}{\pi_{\theta_{\text{old}}}(y_g \mid x)}$ is the importance sampling ratio, $\epsilon_{\text{clip}}$ is the clipping parameter, and $\beta$ controls the KL penalty against a reference policy $\pi_{\text{ref}}$.

**On-policy caveat**: While GRPO is broadly on-policy, in practice multiple batches of generations may be sampled and $k$ updates applied to the model. The first batch is strictly on-policy; subsequent batches are slightly off-policy. Importance sampling and clipping compensate for this policy lag.

### 9.4 Challenges with Hybrid Reasoning Models

Hybrid reasoning models pose unique challenges for RLVR due to **bimodal generation length distributions**:

| Mode | Median Token Length (AIME25) | Distribution Shape |
|---|---|---|
| `/no_think` | ~2,000 tokens | Skewed right, moderate variance |
| `/think` | ~16,700 tokens | Fat-tailed, high variance |

Ideally, RLVR should improve performance in both modes without radically altering their respective length distributions.

### 9.5 Reward Hacking via Length Exploitation

#### 9.5.1 Problem Identification

Naively applying GRPO to optimize the `/no_think` mode (using Big-Math prompts with verified answers) produced a form of **reward hacking**: despite never being prompted to emit a long chain of thought, the model learned to exploit its latent reasoning capabilities to increase reward. Both reward and completion length increased simultaneously:

- By training step 333, reward reached ~75.6% with mean completion length of ~7,244 tokens.
- The model generated extended CoTs including cognitive behaviors (e.g., "Wait, ...") characteristic of reasoning models.
- RLVR effectively **converted the `/no_think` mode into a `/think`-like mode**, defeating the purpose of the hybrid architecture.

#### 9.5.2 Overlong Completion Penalty

To mitigate this, an **overlong completion penalty** was applied, as proposed in the DAPO paper (Yu et al., 2025). The penalty is parameterized by maximum completion length $L_{\max}$ and soft punishment cache $L_{\text{cache}}$:

$$R_{\text{length}}(y) = \begin{cases} 0, & |y| \leq L_{\max} - L_{\text{cache}} \\ \displaystyle\frac{L_{\max} - L_{\text{cache}} - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \leq L_{\max} \\ -1, & |y| > L_{\max} \end{cases}$$

This penalty creates three regimes:
1. **No penalty**: Completions shorter than $L_{\max} - L_{\text{cache}}$ receive zero penalty.
2. **Linear penalty**: Completions between $L_{\max} - L_{\text{cache}}$ and $L_{\max}$ receive a linearly increasing penalty.
3. **Maximum penalty**: Completions exceeding $L_{\max}$ receive a full $-1$ penalty.

The total reward becomes:

$$R_{\text{total}}(x, y) = R_{\text{task}}(x, y) + R_{\text{length}}(y)$$

#### 9.5.3 Ablation Results

Overlong penalties were varied from 1.5K–2K to 4K–4.5K ($(L_{\text{cache}}, L_{\max})$ ranges) in steps of 512 tokens:

| Penalty Range $(L_{\text{cache}}\text{–}L_{\max})$ | Step 245 Reward | AIME25 Improvement |
|---|---|---|
| 1.5K–2K | 79.0% | Minimal |
| 2K–2.5K | 79.9% | Moderate |
| 2.5K–3K | 80.5% | **Significant** |
| 3K–3.5K | 80.8% | Significant |
| 3.5K–4K | 81.9% | Significant |
| 4K–4.5K | 83.0% | Largest but with length inflation |

**Optimal trade-off**: Penalties in the 2.5K–3K range provided the best balance between downstream performance improvement and controlled output length distribution.

### 9.6 RLVR Results

Applying GRPO with a 2.5K–3K overlong penalty nearly **doubled** AIME 2025 performance compared to offline preference optimization:

| Method | AIME 2025 Score |
|---|---|
| SFT | Baseline |
| APO (Preference Optimization) | ~1.5× SFT |
| **GRPO (RLVR)** | **~2× APO** |

The output token distributions at step 400 confirmed that the overlong penalty successfully constrained completion lengths to align with the original `/no_think` distribution while still capturing the performance gains from RLVR.

### 9.7 Open Challenges: Joint Mode Training

Jointly training both reasoning modes (`/think` and `/no_think`) with RLVR remains an unsolved challenge:

1. Each mode requires its own length penalty parameterization.
2. The interplay between mode-specific penalties produces **unstable training dynamics**.
3. This difficulty is reflected in a trend among model developers (e.g., Qwen) to release instruct and reasoning variants as **separate models** rather than unified hybrid systems.

---

## 10. On-Policy Alternatives to Full RL

### 10.1 Overview

Several methods extend preference optimization and distillation into iterative loops that refresh the training signal as the model evolves, occupying a middle ground between static preference optimization and full RL:

#### 10.1.1 Online DPO

Rather than training once on a fixed preference dataset, the model continually:
1. Samples new responses from its current policy.
2. Collects fresh preference labels (from reward models or LLM graders).
3. Updates itself on the new data.

This keeps optimization on-policy, reducing drift between training data and the model's current distribution (Guo et al., 2024). The training loop synchronization frequency $s$ defines a spectrum:

| Variant | Sync Frequency | Characterization |
|---|---|---|
| Offline DPO | $s = \infty$ | Never sync; fully off-policy |
| Semi-online DPO | $s = k$ (periodic) | Sync every $k$ steps |
| Online DPO | $s = 1$ | Sync every step; fully on-policy |

Results from FAIR (Lanchantin et al., 2025) demonstrate that even semi-on-policy DPO ($s = 10\text{–}100$) can match GRPO performance with far less compute:

| Method | Math500 | NuminaMath | AMC23 |
|---|---|---|---|
| Seed (Llama-3.1-8B-Instruct) | 47.4 | 33.9 | 23.7 |
| Offline DPO ($s = \infty$) | 53.7 | 36.4 | 28.8 |
| Semi-online DPO ($s = 100$) | 58.9 | 39.3 | 35.1 |
| Semi-online DPO ($s = 10$) | 57.2 | 39.4 | 31.4 |
| Online DPO ($s = 1$) | 58.7 | 39.6 | 32.9 |
| GRPO | 58.1 | 38.8 | 33.6 |

#### 10.1.2 On-Policy Distillation

Instead of preference labels, the learning signal derives from a stronger teacher model. The student samples responses at each training step, and the KL divergence between student and teacher logit distributions provides the optimization objective:

$$\mathcal{L}_{\text{distill}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ D_{\text{KL}} \left( \pi_{\text{teacher}}(\cdot \mid x, y) \| \pi_\theta(\cdot \mid x, y) \right) \right]$$

This approach transfers teacher capabilities without requiring explicit preference labels or verifiers.

**Qwen3 results** demonstrate substantial advantages over GRPO for models under 32B parameters:

| Method | AIME24 | AIME25 | MATH500 | LiveCodeBench v5 | GPU Hours |
|---|---|---|---|---|---|
| Off-policy distillation | 55.0 | 42.8 | 92.4 | 42.0 | — |
| + Reinforcement learning | 67.6 | 55.5 | 94.8 | 52.9 | 17,920 |
| + On-policy distillation | **74.4** | **65.5** | **97.0** | **60.3** | **1,800** |

On-policy distillation achieved **superior performance at 10× lower compute cost** compared to RL.

**Limitation**: Traditional on-policy distillation requires the teacher and student to share the same tokenizer. The **General On-Policy Logit Distillation (GOLD)** method addresses this constraint by enabling distillation between any teacher-student pair regardless of tokenizer compatibility.

**Catastrophic forgetting mitigation**: Thinking Machines demonstrated that on-policy distillation effectively restores capabilities lost during domain-specific fine-tuning (e.g., IFEval regression after fine-tuning on internal data).

### 10.2 Method Selection Guide

| Algorithm | When to Use | Compute Cost | Model Size Sweet Spot |
|---|---|---|---|
| **Online DPO** | Cheap preference labels; evolving distribution alignment | Moderate | Any size |
| **On-policy distillation** | Strong teacher available; capability transfer | Low | $< 30$B parameters |
| **Reinforcement learning (GRPO)** | Verifiable rewards; multi-step reasoning/planning | High | $> 20$B parameters |

---

## 11. Consolidated Engineering Principles

### 11.1 Evaluation Principles

1. **Prioritize evaluation infrastructure before training begins.** Discuss with the pretraining team which core evaluations must be preserved across post-training. Implement them well before the base model finishes training.
2. **Use small correlated subsets** for rapid iteration; reserve full benchmarks for final reports.
3. **Strip CoT from scored output** for reasoning models.
4. **Pin LLM judge versions** for consistent comparisons.
5. **Decontaminate training data** against evaluation benchmarks via $n$-gram matching.
6. **Report avg@$k$ accuracy** for small benchmarks ($< 2000$ prompts) to mitigate sampling noise.
7. **Replicate published results** on reference models before deploying a new evaluation.

### 11.2 Training Stability Principles

1. **Save checkpoints frequently** and push to remote storage.
2. **Ensure automatic restart** capability in the training framework.
3. **Use FP32 master weights and optimizer states** during mixed-precision training to prevent numerical instability.
4. **Monitor for hardware failures** (GPU throttling, "falling off the bus") and implement aggressive retry logic.

### 11.3 Data Pipeline Principles

1. **Always vibe-test models**, even when evaluations look acceptable. Automated metrics cannot capture all behavioral failure modes.
2. **Verify chat template rendering** for every dataset variant. A single `None` value in a template argument can silently corrupt the entire training corpus.
3. **Balance hybrid reasoning mixtures by token count**, not example count.
4. **Pair data across reasoning modes**: each example must unambiguously indicate whether extended reasoning or concise answering is expected.

### 11.4 Hyperparameter Principles

| Stage | Learning Rate | Key Parameters | Epochs |
|---|---|---|---|
| **SFT** | $1\text{–}10\times 10^{-5}$ (scan two orders of magnitude) | Packing (enable/disable), masking (assistant-only), sequence length | 1 for ablation; 3–5 for final |
| **PO** | $10\text{–}100\times$ smaller than SFT LR | $\beta \in [0.01, 0.5]$, algorithm selection | Typically 1 (overfits quickly) |
| **RL (GRPO)** | Task-dependent | Overlong penalty $(L_{\text{cache}}, L_{\max})$, KL coefficient $\beta$, group size $G$ | Continuous online training |

### 11.5 Framework and Infrastructure Principles

1. **Fork your framework** for rapid experimentation; upstream validated features to the main library.
2. **Use battle-tested accelerators** (DeepSpeed ZeRO, FSDP2) for long-running jobs.
3. **Enable compute-efficient kernels** (FlashAttention, Liger) when hardware supports them.
4. **Select parallelism strategy** based on model and context size:
   - Data parallelism for small models or LoRA training.
   - FSDP2 or DeepSpeed ZeRO-3 for large models requiring weight/optimizer sharding.
   - Context parallelism for long-context training.

---

## 12. Conclusion

This report has presented a comprehensive technical account of post-training methodologies for large language models, grounded in the development of SmolLM3-3B as a state-of-the-art hybrid reasoning model at the 3-billion-parameter scale. The key conclusions are:

1. **The post-training compass (Why → What → How) prevents wasted compute.** Strategic clarity before committing resources ensures that every training run serves a defined objective with measurable success criteria.

2. **Evaluation infrastructure must precede training.** The failure to prioritize evaluations introduces blind spots that compound throughout the training pipeline. Vibe testing is a non-negotiable complement to automated benchmarks.

3. **SFT remains the foundational post-training stage.** Its computational efficiency, training stability, and role as a high-quality initialization for subsequent methods make it indispensable for all but the most extreme frontier research settings.

4. **Mid-training on domain-specific reasoning corpora produces multiplicative gains** when composed with SFT. For SmolLM3, mid-training nearly tripled performance on competitive mathematics and programming benchmarks.

5. **Preference optimization extends performance beyond imitation learning.** APO-zero yielded 15–20 percentage point improvements on instruction following, and preference optimization improved reasoning quality—not just helpfulness.

6. **RLVR with GRPO can further double reasoning performance**, but requires careful reward shaping. Hybrid reasoning models are susceptible to length-based reward hacking, which must be mitigated with overlong completion penalties.

7. **On-policy distillation offers a computationally efficient alternative to RL** for models under 32B parameters, achieving superior performance at $10\times$ lower compute cost.

8. **Engineering discipline—checkpoint management, numerical precision, data pipeline validation, and vibe testing—is as important as algorithmic innovation.** The most sophisticated training algorithm is rendered useless by a `None` value in a template argument or BF16 gradient accumulation errors.

The complete post-training pipeline for SmolLM3 integrated all of these components into a coherent recipe:

$$\text{Base Model} \xrightarrow{\text{Mid-Training}} \text{Reasoning-Enhanced Base} \xrightarrow{\text{SFT}} \text{Hybrid Reasoner} \xrightarrow{\text{APO}} \text{SmolLM3-3B}$$

The resulting model achieved state-of-the-art performance at the 3B scale, sitting on the Pareto front alongside Qwen3-1.7B and Qwen3-4B across eight popular LLM benchmarks.

---

## References

- Agarwal, R., et al. (2024). On-policy distillation of language models.
- Barres, A., et al. (2025). TAU-Bench: A benchmark for tool-augmented user simulation.
- Chu, T., et al. (2025). On the role of RL in post-training.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems (GSM8k).
- DeepSeek-AI, Guo, D., et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.
- Ding, Y., et al. (2024). Efficient sequence packing strategies for SFT.
- D'Oosterlinck, K., et al. (2024). Anchored preference optimization.
- Dubois, Y., et al. (2025). AlpacaEval: Length-controlled automatic evaluation of instruction-following models.
- Ethayarajh, K., et al. (2024). KTO: Model alignment as prospect theoretic optimization.
- Gandhi, K., et al. (2025). Cognitive behaviors in language model reasoning.
- Guo, S., et al. (2024). Online DPO: Direct preference optimization with online data.
- He, J., et al. (2024). Multi-IF: Multi-turn instruction following evaluation.
- Hong, J., et al. (2024). ORPO: Monolithic preference optimization without reference model.
- Howard, J. & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification (ULMFiT).
- Hsieh, C., et al. (2024). RULER: What's the real context size of your long-context language models?
- Huang, S., et al. (2024). The 37 implementation details of proximal policy optimization.
- Kamradt, G. (2023). Needle in a haystack: Measuring LLM context retrieval.
- Khatri, A., et al. (2025). Scaling RL effectively.
- Lambert, N., et al. (2022). Illustrating reinforcement learning from human feedback.
- Lambert, N., et al. (2025). Tulu 3: Pushing frontiers in open language model post-training.
- Lanchantin, J., et al. (2025). On-policy vs. off-policy DPO for reasoning.
- Li, Q., et al. (2024). GSMPlus: A comprehensive benchmark for evaluating the robustness of LLMs as mathematical problem solvers.
- Li, T., et al. (2024). ArenaHard: An LLM benchmark from crowdsourced pairwise preferences.
- Lightman, H., et al. (2023). Let's verify step by step (MATH-500).
- Ni, J., et al. (2024). MixEval: Deriving wisdom of the crowd from LLM benchmark mixtures.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT).
- Polo, F., et al. (2024). tinyBenchmarks: Evaluating LLMs with fewer examples.
- Pyatkin, V., et al. (2025). IFBench: Evaluating instruction following with diverse constraints.
- Rafailov, R., et al. (2024). Direct preference optimization: Your language model is secretly a reward model.
- Rein, D., et al. (2024). GPQA: A graduate-level Google-proof Q&A benchmark.
- Shi, F., et al. (2022). Language models are multilingual chain-of-thought reasoners (MGSM).
- Singh, A., et al. (2025). Global MMLU: Understanding and addressing cultural and linguistic biases.
- Sirdeshmukh, A., et al. (2025). MultiChallenge: A benchmark for multi-turn instruction following.
- team, FAIR, et al. (2025). Code World Model.
- Wei, J., et al. (2024). SimpleQA: Measuring short-form factuality in large language models.
- Xu, Y., et al. (2025). Phi-4-Mini-Reasoning.
- Yang, A., Li, B., et al. (2025). Qwen3 technical report.
- Yen, H., et al. (2025). HELMET: How to evaluate long-context language models effectively and thoroughly.
- Yu, Q., et al. (2025). DAPO: An open-source LLM reinforcement learning system.
- Yue, Y., et al. (2025). Does RL elicit new capabilities?
- Zhou, J., et al. (2023). Instruction-following evaluation for large language models (IFEval).