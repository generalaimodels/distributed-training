# GPT-5.4: Advanced Technical Report — Architecture, Capabilities, and Systems-Level Analysis

**Release Date:** March 5, 2026
**Classification:** Frontier Reasoning Model — General-Purpose with Native Computer-Use
**Variants:** `gpt-5.4`, `gpt-5.4-pro`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architectural Evolution and Design Philosophy](#2-architectural-evolution-and-design-philosophy)
3. [Knowledge Work and Professional Task Performance](#3-knowledge-work-and-professional-task-performance)
4. [Computer Use and Visual Perception](#4-computer-use-and-visual-perception)
5. [Coding Capabilities and Latency-Accuracy Tradeoffs](#5-coding-capabilities-and-latency-accuracy-tradeoffs)
6. [Tool Use, Tool Search, and Agentic Orchestration](#6-tool-use-tool-search-and-agentic-orchestration)
7. [Long-Context Processing and Memory Fidelity](#7-long-context-processing-and-memory-fidelity)
8. [Reasoning Efficiency and Token Economics](#8-reasoning-efficiency-and-token-economics)
9. [Steerability and Interactive Chain-of-Thought](#9-steerability-and-interactive-chain-of-thought)
10. [Factuality and Hallucination Reduction](#10-factuality-and-hallucination-reduction)
11. [Safety, Monitoring, and CoT Controllability](#11-safety-monitoring-and-cot-controllability)
12. [Comprehensive Benchmark Analysis](#12-comprehensive-benchmark-analysis)
13. [Pricing, Deployment, and API Specification](#13-pricing-deployment-and-api-specification)
14. [Critical Analysis and Open Research Questions](#14-critical-analysis-and-open-research-questions)

---

## 1. Executive Summary

GPT-5.4 represents a **unified frontier model** that consolidates advances across reasoning, coding, agentic tool use, computer operation, and professional knowledge work into a single architecture. It is the first general-purpose model from OpenAI with **native, state-of-the-art computer-use capabilities** and supports up to **1M tokens of context**.

### 1.1 Key Technical Contributions

| Dimension | Advance |
|---|---|
| **Knowledge Work** | $83.0\%$ win/tie rate on GDPval (vs. $70.9\%$ for GPT-5.2), spanning 44 occupations |
| **Computer Use** | $75.0\%$ on OSWorld-Verified, surpassing human performance ($72.4\%$) |
| **Coding** | $57.7\%$ on SWE-Bench Pro (Public) with lower latency across all reasoning efforts |
| **Tool Efficiency** | Tool search reduces token usage by $47\%$ while preserving accuracy |
| **Reasoning Efficiency** | Significantly fewer output tokens per problem vs. GPT-5.2 |
| **Factuality** | $33\%$ reduction in false individual claims; $18\%$ reduction in error-containing responses |
| **Web Search** | $82.7\%$ on BrowseComp ($+16.9$ pts abs. over GPT-5.2) |
| **Abstract Reasoning** | $73.3\%$ on ARC-AGI-2 Verified ($+20.4$ pts abs. over GPT-5.2) |
| **Context Window** | Experimental 1M token support in Codex |

### 1.2 Model Lineage

$$
\text{GPT-5.2} \xrightarrow{\text{codex specialization}} \text{GPT-5.3-Codex} \xrightarrow{\text{unified integration}} \text{GPT-5.4}
$$

GPT-5.4 subsumes GPT-5.3-Codex's coding capabilities while integrating improvements in knowledge work, vision, computer use, and tool orchestration — making it the first mainline model to **unify** these capabilities.

---

## 2. Architectural Evolution and Design Philosophy

### 2.1 Unification Principle

GPT-5.4 follows a **convergent architecture design** philosophy: rather than maintaining separate specialized models (e.g., GPT-5.3-Codex for coding), capabilities are consolidated into a single model that can be steered via reasoning effort levels and developer messages.

The design objectives can be formalized as a multi-objective optimization:

$$
\theta^{*} = \arg\min_{\theta} \sum_{d \in \mathcal{D}} \lambda_d \cdot \mathcal{L}_d(\theta) + \alpha \cdot \mathcal{C}(\theta)
$$

where:
- $\mathcal{D} = \{\text{reasoning}, \text{coding}, \text{vision}, \text{tool-use}, \text{computer-use}, \text{knowledge-work}\}$ is the set of capability domains
- $\mathcal{L}_d(\theta)$ is the domain-specific loss
- $\lambda_d$ are domain-specific weighting coefficients
- $\mathcal{C}(\theta)$ represents a token-efficiency regularizer penalizing verbose reasoning chains

### 2.2 Key Architectural Properties

1. **Reasoning-native:** Chain-of-thought reasoning is embedded as a first-class modality, not a prompting artifact
2. **Tool-native:** The model natively understands when to invoke, search for, and compose external tools
3. **Vision-native:** Screenshots, documents, and UI elements are processed with coordinate-level spatial understanding
4. **Context-scalable:** The architecture supports variable context windows from the standard $272\text{K}$ up to experimental $1\text{M}$ tokens
5. **Effort-adjustable:** Reasoning effort levels (from `none` to `xhigh`) provide a continuous latency-accuracy tradeoff

### 2.3 Training Signal Integration

The model integrates training signals from at minimum the following sources:

- **Reinforcement learning from human feedback (RLHF):** Preference alignment for knowledge work quality
- **Code execution feedback:** Verified correctness signals from code interpreters and test suites
- **Computer-use trajectories:** Action-observation pairs from desktop/browser environments
- **Tool-use traces:** Multi-step tool invocation sequences with outcome verification
- **Factuality supervision:** Grounded verification against reference corpora
- **Safety alignment:** Red-teaming, adversarial probing, and Chain-of-Thought controllability constraints

---

## 3. Knowledge Work and Professional Task Performance

### 3.1 GDPval Benchmark

GDPval evaluates agents' abilities to produce **well-specified knowledge work** across 44 occupations from the top 9 industries contributing to U.S. GDP. Tasks require real work products: sales presentations, accounting spreadsheets, urgent care schedules, manufacturing diagrams, and short videos.

| Model | Win/Tie Rate vs. Industry Professional |
|---|---|
| GPT-5.4 | $83.0\%$ |
| GPT-5.4 Pro | $82.0\%$ |
| GPT-5.2 Pro | $74.1\%$ |
| GPT-5.2 | $70.9\%$ |

**Interpretation:** The $+12.1$ percentage point absolute improvement from GPT-5.2 to GPT-5.4 indicates a qualitative shift — the model now matches or exceeds industry professionals in $> 4$ out of $5$ comparisons.

The improvement can be decomposed into:

$$
\Delta_{\text{GDPval}} = \Delta_{\text{reasoning}} + \Delta_{\text{artifact-quality}} + \Delta_{\text{factuality}} + \Delta_{\text{tool-integration}}
$$

### 3.2 Spreadsheet, Document, and Presentation Quality

GPT-5.4 received focused improvements on structured professional artifacts:

| Task Type | GPT-5.4 | GPT-5.2 | Improvement |
|---|---|---|---|
| Investment Banking Modeling Tasks (Internal) | $87.3\%$ | $68.4\%$ | $+18.9$ pts |
| OfficeQA | $68.1\%$ | $63.1\%$ | $+5.0$ pts |
| Presentation Preference (human raters) | $68.0\%$ preferred | — | — |

**Spreadsheet modeling:** The jump from $68.4\%$ to $87.3\%$ on investment banking modeling tasks is particularly significant — this represents the delta between an unreliable tool and one approaching professional competency for junior analyst-level financial modeling.

**Presentation quality:** Human raters preferred GPT-5.4 presentations $68.0\%$ of the time over GPT-5.2, citing:
- Stronger aesthetics
- Greater visual variety
- More effective use of image generation

### 3.3 FinanceAgent Performance

| Model | FinanceAgent v1.1 |
|---|---|
| GPT-5.4 Pro | $61.5\%$ |
| GPT-5.2 | $59.5\%$ |
| GPT-5.4 | $56.0\%$ |
| GPT-5.3-Codex | $54.0\%$ |

**Note:** GPT-5.4 Pro outperforms GPT-5.4 base on FinanceAgent, suggesting that complex financial reasoning benefits from extended compute budgets. The inversion between GPT-5.4 ($56.0\%$) and GPT-5.2 ($59.5\%$) on this benchmark warrants investigation — it may reflect a capability tradeoff in the unified training or benchmark-specific sensitivity to reasoning style.

---

## 4. Computer Use and Visual Perception

### 4.1 Native Computer-Use Capabilities

GPT-5.4 is the **first general-purpose model** with native computer-use capabilities. It operates computers through two complementary modalities:

1. **Code-driven automation:** Writing code to operate computers via libraries such as Playwright
2. **Visual-motor interaction:** Issuing mouse and keyboard commands in response to screenshots (coordinate-based clicking)

### 4.2 Desktop Environment Performance

#### OSWorld-Verified

| Model | Accuracy |
|---|---|
| GPT-5.4 | $75.0\%$ |
| GPT-5.3-Codex | $74.0\%$ |
| **Human Performance** | $72.4\%$ |
| GPT-5.2 | $47.3\%$ |

**Critical observation:** GPT-5.4 **surpasses human performance** on OSWorld-Verified by $+2.6$ pts. The jump from GPT-5.2 ($47.3\%$) to GPT-5.4 ($75.0\%$) represents a $+27.7$ pt absolute improvement — one of the largest single-generation capability gains across all benchmarks.

The efficiency profile is also notable: GPT-5.4 achieves higher accuracy with fewer tool yields (iterations requiring tool responses), indicating improved planning and parallel tool execution:

$$
\text{Efficiency Ratio} = \frac{\text{Accuracy}}{\text{Mean Tool Yields}}
$$

GPT-5.4 demonstrates a superior efficiency ratio across the entire tool-yield spectrum compared to GPT-5.2.

### 4.3 Browser-Based Computer Use

| Benchmark | GPT-5.4 | Comparator | Delta |
|---|---|---|---|
| WebArena-Verified (DOM + screenshot) | $67.3\%$ | GPT-5.2: $65.4\%$ | $+1.9$ pts |
| Online-Mind2Web (screenshot only) | $92.8\%$ | Atlas Agent Mode: $70.9\%$ | $+21.9$ pts |

The $92.8\%$ on Online-Mind2Web using **screenshot-based observations alone** is particularly noteworthy — the model achieves near-ceiling performance purely from visual perception of browser states, without requiring DOM access.

### 4.4 Visual Perception Foundations

Computer-use capabilities are built on improved **general visual understanding**:

| Benchmark | GPT-5.4 | GPT-5.2 | Metric |
|---|---|---|---|
| MMMU-Pro (no tools) | $81.2\%$ | $79.5\%$ | Accuracy |
| MMMU-Pro (with tools) | $82.1\%$ | $80.4\%$ | Accuracy |
| OmniDocBench | $0.109$ | $0.140$ | Norm. edit distance (lower = better) |

**OmniDocBench analysis:** The reduction from $0.140$ to $0.109$ in normalized edit distance represents a $22.1\%$ relative improvement in document parsing fidelity, measured without any reasoning effort — indicating stronger **base perceptual capabilities**.

### 4.5 High-Resolution Image Processing

GPT-5.4 introduces a new `original` image input detail level:

| Detail Level | Maximum Resolution | Maximum Total Pixels |
|---|---|---|
| `original` (new) | $6000 \times \text{dim}$ | $10.24\text{M}$ pixels |
| `high` (updated) | $2048 \times \text{dim}$ | $2.56\text{M}$ pixels |

This represents a $4\times$ increase in supported pixel count at the `original` level, directly improving:
- **Localization ability** (bounding box / coordinate accuracy)
- **Image understanding** (fine-grained detail perception)
- **Click accuracy** (for computer-use tasks)

---

## 5. Coding Capabilities and Latency-Accuracy Tradeoffs

### 5.1 SWE-Bench Pro Performance

SWE-Bench Pro evaluates a model's ability to resolve real-world software engineering tasks from open-source repositories.

| Model | Accuracy | Relative Latency Profile |
|---|---|---|
| GPT-5.4 | $57.7\%$ | Lower latency across all reasoning efforts |
| GPT-5.3-Codex | $56.8\%$ | Higher latency at equivalent accuracy |
| GPT-5.2 | $55.6\%$ | Highest latency at given accuracy |

**Pareto optimality:** GPT-5.4 dominates the latency-accuracy Pareto frontier across all reasoning effort levels. At every target accuracy threshold, GPT-5.4 achieves the required accuracy at lower estimated latency than both GPT-5.3-Codex and GPT-5.2.

The latency estimate accounts for:

$$
\hat{L} = T_{\text{tool}} + T_{\text{sampled}} + T_{\text{input}}
$$

where $T_{\text{tool}}$ is tool call execution duration, $T_{\text{sampled}}$ is sampled token generation time, and $T_{\text{input}}$ is input token processing time.

### 5.2 Terminal and Systems-Level Coding

| Benchmark | GPT-5.4 | GPT-5.3-Codex | GPT-5.2 |
|---|---|---|---|
| Terminal-Bench 2.0 | $75.1\%$ | $77.3\%$ | $62.2\%$ |

**Note:** GPT-5.3-Codex retains a $+2.2$ pt advantage on Terminal-Bench 2.0 over GPT-5.4, likely reflecting specialized terminal interaction training in the Codex variant. However, GPT-5.4's $+12.9$ pt improvement over GPT-5.2 represents substantial progress in systems-level coding.

### 5.3 Fast Mode and Priority Processing

Codex `/fast` mode delivers up to $1.5\times$ faster token velocity with GPT-5.4. This is the **same model at the same intelligence level** — the acceleration comes from infrastructure-level priority processing, not model distillation or capability reduction.

For API users, equivalent acceleration is available via `priority processing` at $2\times$ the standard API rate.

### 5.4 Frontend Development and Interactive Testing

GPT-5.4 excels at **complex frontend tasks** with notably improved aesthetic quality and functional correctness. The release includes an experimental Codex skill `Playwright (Interactive)` that enables:

- Visual debugging of web and Electron applications
- Automated browser playtesting during development
- Self-testing: the model tests applications it is building, as it builds them

This creates a **closed-loop development cycle**:

$$
\text{Code} \xrightarrow{\text{build}} \text{Application} \xrightarrow{\text{Playwright}} \text{Visual Test} \xrightarrow{\text{observation}} \text{Bug Detection} \xrightarrow{\text{fix}} \text{Code}
$$

---

## 6. Tool Use, Tool Search, and Agentic Orchestration

### 6.1 The Tool Scaling Problem

Prior to GPT-5.4, tool definitions were included **in full** in every prompt. For systems with large tool ecosystems:

- **Cost scaling:** Token count grows as $\mathcal{O}(|\mathcal{T}| \cdot \bar{d})$ where $|\mathcal{T}|$ is the number of tools and $\bar{d}$ is the mean definition length
- **Context contamination:** Thousands of tokens of tool definitions crowd the context window
- **Cache invalidation:** Prompt perturbation from tool definitions reduces cache hit rates

### 6.2 Tool Search Architecture

GPT-5.4 introduces **tool search**, a fundamentally different approach:

**Without tool search (prior models):**

$$
\text{Prompt} = [\text{System}] \oplus [\text{Tool}_1, \text{Tool}_2, \ldots, \text{Tool}_N] \oplus [\text{User Query}]
$$

**With tool search (GPT-5.4):**

$$
\text{Prompt} = [\text{System}] \oplus [\text{Tool Index}] \oplus [\text{User Query}]
$$

$$
\text{At invocation time: Model} \xrightarrow{\text{search}} \text{Tool}_k \text{ definition} \xrightarrow{\text{append}} \text{Context}
$$

The model receives a lightweight index of available tools and dynamically retrieves full definitions only when needed.

### 6.3 Efficiency Gains: MCP Atlas Evaluation

Evaluated on 250 tasks from Scale's MCP Atlas benchmark with all 36 MCP servers enabled:

| Configuration | Total Tokens (Avg.) | Relative Reduction |
|---|---|---|
| Without tool search | $123{,}139$ | — |
| With tool search | $65{,}320$ | $-47\%$ |

**Accuracy:** Identical between configurations.

Token decomposition:

| Token Category | Without Tool Search | With Tool Search |
|---|---|---|
| Upfront Input Tokens | Dominant (tool definitions) | Minimal (tool index) |
| Output Tokens | Comparable | Comparable |
| Input Tokens from Tool Outputs | Comparable | Comparable |

### 6.4 Agentic Tool Calling: Toolathlon

Toolathlon tests how well AI agents use real-world tools and APIs to complete multi-step tasks (e.g., read emails → extract attachments → upload → grade → record in spreadsheet).

| Model | Accuracy | Efficiency |
|---|---|---|
| GPT-5.4 | $54.6\%$ | Higher accuracy in fewer tool yields |
| GPT-5.3-Codex | $51.9\%$ | — |
| GPT-5.2 | $45.7\%$ | More tool yields for lower accuracy |

**Tool yields** (number of times the assistant yields to await tool responses) are a better proxy for latency than raw tool call counts because they account for parallelization:

$$
\text{Tool Yields} \leq \text{Tool Calls}
$$

$$
\text{Effective Latency} \propto \text{Tool Yields} \cdot \bar{T}_{\text{response}}
$$

### 6.5 Low-Latency Tool Use (Reasoning Effort: None)

For latency-sensitive applications where reasoning overhead must be minimized:

| Model | $\tau^2$-bench Telecom (no reasoning) |
|---|---|
| GPT-5.4 | $64.3\%$ |
| GPT-5.2 | $57.2\%$ |
| GPT-5.1 | $45.2\%$ |
| GPT-4.1 | $43.6\%$ |

**Trend:** Each model generation improves non-reasoning tool use performance by approximately $+7$–$12$ pts, suggesting that **base model capabilities** (not just reasoning chains) are advancing significantly.

With reasoning enabled at `xhigh`, $\tau^2$-bench Telecom reaches $98.9\%$ for GPT-5.4 (vs. $98.7\%$ for GPT-5.2), approaching saturation.

### 6.6 Agentic Web Search: BrowseComp

BrowseComp measures persistent, multi-round web browsing to locate hard-to-find information ("needle-in-a-haystack" questions).

| Model | Accuracy |
|---|---|
| GPT-5.4 Pro | $89.3\%$ (new SOTA) |
| GPT-5.4 | $82.7\%$ |
| GPT-5.2 Pro | $77.9\%$ |
| GPT-5.2 | $65.8\%$ |

**Methodological note:** A search blocklist excluding websites containing benchmark answers was applied to prevent contamination. GPT-5.4 was tested with a longer, updated blocklist and on a later date, so scores reflect changes in the model, the search system, and the state of the internet.

The $+16.9$ pt jump from GPT-5.2 to GPT-5.4 reflects improved **search persistence** (more rounds of search before giving up) and **source synthesis** (integrating information across multiple sources).

---

## 7. Long-Context Processing and Memory Fidelity

### 7.1 Context Window Specifications

| Surface | Standard Context | Extended Context |
|---|---|---|
| API | $272\text{K}$ tokens | — |
| Codex | $272\text{K}$ tokens | $1\text{M}$ tokens (experimental) |

Requests exceeding the standard $272\text{K}$ window in Codex count against usage limits at $2\times$ the normal rate.

### 7.2 Long-Context Benchmark Performance

#### Graphwalks (Graph Traversal Tasks)

| Context Range | BFS (GPT-5.4) | BFS (GPT-5.2) | Parents (GPT-5.4) | Parents (GPT-5.2) |
|---|---|---|---|---|
| $0$–$128\text{K}$ | $93.0\%$ | $94.0\%$ | $89.8\%$ | $89.0\%$ |
| $256\text{K}$–$1\text{M}$ | $21.4\%$ | N/A | $32.4\%$ | N/A |

**Critical analysis:** Performance within the standard context window ($0$–$128\text{K}$) is essentially **flat** between GPT-5.4 and GPT-5.2, with GPT-5.2 marginally better on BFS ($94.0\%$ vs. $93.0\%$). At extended context ranges ($256\text{K}$–$1\text{M}$), performance drops dramatically to $21.4\%$ (BFS) and $32.4\%$ (parents), indicating that the $1\text{M}$ context is functional but not yet reliable for tasks requiring precise information retrieval across the full context.

#### OpenAI MRCR v2 (8-needle Multi-Range Context Retrieval)

| Context Range | GPT-5.4 | GPT-5.2 |
|---|---|---|
| $4\text{K}$–$8\text{K}$ | $97.3\%$ | $98.2\%$ |
| $8\text{K}$–$16\text{K}$ | $91.4\%$ | $89.3\%$ |
| $16\text{K}$–$32\text{K}$ | $97.2\%$ | $95.3\%$ |
| $32\text{K}$–$64\text{K}$ | $90.5\%$ | $92.0\%$ |
| $64\text{K}$–$128\text{K}$ | $86.0\%$ | $85.6\%$ |
| $128\text{K}$–$256\text{K}$ | $79.3\%$ | $77.0\%$ |
| $256\text{K}$–$512\text{K}$ | $57.5\%$ | N/A |
| $512\text{K}$–$1\text{M}$ | $36.6\%$ | N/A |

### 7.3 Attention Degradation Model

The observed performance decay across context lengths follows an approximately log-linear degradation pattern for the $8$-needle retrieval task:

$$
\text{Accuracy}(L) \approx a - b \cdot \log_2\left(\frac{L}{L_0}\right)
$$

where $L$ is the context length, $L_0$ is a reference length, and $b$ captures the per-octave degradation rate. For MRCR v2 8-needle:

- Within $4\text{K}$–$128\text{K}$: degradation is moderate ($\sim 97\% \to 86\%$)
- Within $128\text{K}$–$1\text{M}$: degradation accelerates significantly ($\sim 79\% \to 37\%$)

This suggests a **phase transition** in attention effectiveness around the $128\text{K}$–$256\text{K}$ boundary, potentially corresponding to the standard context window training distribution.

---

## 8. Reasoning Efficiency and Token Economics

### 8.1 Token Efficiency Improvement

GPT-5.4 is described as the **most token-efficient reasoning model** to date, using significantly fewer tokens to solve problems compared to GPT-5.2. This efficiency manifests across:

1. **Reasoning tokens:** More concise chain-of-thought for equivalent problem difficulty
2. **Tool interaction tokens:** Fewer tool yields (parallel tool calls reduce sequential steps)
3. **Tool definition tokens:** Tool search eliminates upfront tool definition overhead ($47\%$ reduction)

### 8.2 Economic Analysis

| Model | Input Price | Cached Input | Output Price |
|---|---|---|---|
| `gpt-5.2` | $\$1.75/\text{M}$ | $\$0.175/\text{M}$ | $\$14/\text{M}$ |
| `gpt-5.4` | $\$2.50/\text{M}$ | $\$0.25/\text{M}$ | $\$15/\text{M}$ |
| `gpt-5.2-pro` | $\$21/\text{M}$ | — | $\$168/\text{M}$ |
| `gpt-5.4-pro` | $\$30/\text{M}$ | — | $\$180/\text{M}$ |

**Per-token cost increase:**
- Input: $+42.9\%$ ($\$1.75 \to \$2.50$)
- Output: $+7.1\%$ ($\$14 \to \$15$)

**Effective cost analysis:** Despite higher per-token pricing, the reduced total token count per task (from reasoning efficiency + tool search) can result in **lower total cost** for many workloads. The net cost impact for a given task is:

$$
\Delta\text{Cost}_{\text{task}} = \underbrace{(p_{\text{5.4}} - p_{\text{5.2}})}_{\text{price increase}} \cdot N_{\text{5.4}} + \underbrace{p_{\text{5.2}} \cdot (N_{\text{5.4}} - N_{\text{5.2}})}_{\text{token savings}}
$$

where $p$ denotes per-token price and $N$ denotes total tokens consumed. The sign of $\Delta\text{Cost}_{\text{task}}$ depends on the magnitude of token reduction relative to the price increase.

### 8.3 Batch, Flex, and Priority Pricing

| Processing Tier | Rate Multiplier |
|---|---|
| Batch / Flex | $0.5\times$ standard |
| Standard | $1.0\times$ |
| Priority | $2.0\times$ standard |

This tiered pricing enables developers to select their position on the **cost-latency Pareto curve** per workload.

---

## 9. Steerability and Interactive Chain-of-Thought

### 9.1 Preamble-Based Plan Exposure

GPT-5.4 Thinking introduces a **preamble mechanism** for complex queries: the model outlines its planned approach before executing, allowing users to:

1. **Review** the proposed approach
2. **Redirect** the model mid-response with additional instructions
3. **Converge** on the desired output with fewer total turns

This represents a shift from **open-loop reasoning** (model reasons fully, then presents result) to **interactive closed-loop reasoning**:

$$
\text{Open-loop:} \quad x \xrightarrow{\text{reason}} y
$$

$$
\text{Closed-loop:} \quad x \xrightarrow{\text{plan}} p \xrightarrow[\text{user feedback}]{\text{adjust}} p' \xrightarrow{\text{execute}} y
$$

### 9.2 Extended Coherent Reasoning

GPT-5.4 can **think longer** on difficult tasks while maintaining stronger awareness of earlier steps. This addresses a known failure mode in prior models where extended reasoning chains lead to **context amnesia** — the model loses track of constraints or intermediate results established earlier in the reasoning process.

### 9.3 Developer-Level Steerability

For API and Codex usage, GPT-5.4's behavior is steerable via:

- **Developer messages:** Adjusting behavior for particular use cases
- **Reasoning effort levels:** `none`, `low`, `medium`, `high`, `xhigh`
- **Custom confirmation policies:** Configuring safety behavior for different risk tolerance levels
- **Model context window parameters:** `model_context_window` and `model_auto_compact_token_limit`

---

## 10. Factuality and Hallucination Reduction

### 10.1 Quantified Improvements

Evaluated on de-identified prompts where users flagged factual errors:

| Metric | Improvement (GPT-5.4 vs. GPT-5.2) |
|---|---|
| Individual claims false rate | $-33\%$ (relative) |
| Full responses with any error | $-18\%$ (relative) |

### 10.2 Factuality as a Training Objective

Let $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$ denote the set of individual claims in a response. The response-level error rate $P(\text{error})$ is:

$$
P(\text{error}) = 1 - \prod_{i=1}^{n} (1 - p_i)
$$

where $p_i = P(c_i \text{ is false})$. A $33\%$ reduction in per-claim error probability yields:

$$
p_i^{\text{5.4}} = 0.67 \cdot p_i^{\text{5.2}}
$$

For a response with $n$ claims:

$$
P(\text{error})^{\text{5.4}} = 1 - (1 - 0.67 \cdot p_i^{\text{5.2}})^n
$$

The observed $18\%$ reduction in response-level errors is **consistent** with the $33\%$ claim-level reduction for typical response lengths, confirming internal consistency of the reported metrics.

---

## 11. Safety, Monitoring, and CoT Controllability

### 11.1 Preparedness Framework Classification

GPT-5.4 is classified as **High cyber capability** under the OpenAI Preparedness Framework, with corresponding protections:

- **Expanded cyber safety stack:** Monitoring systems, trusted access controls
- **Asynchronous blocking:** Higher-risk requests for Zero Data Retention (ZDR) surface customers
- **Request-level blocking:** Part of the cyber risk mitigation stack for certain ZDR customers
- **Dual-use awareness:** Cybersecurity capabilities are inherently dual-use; precautionary deployment maintained

### 11.2 Chain-of-Thought Controllability

A novel open-source evaluation — **CoT controllability** — measures whether models can deliberately **obfuscate their reasoning** to evade monitoring.

**Key finding:** GPT-5.4 Thinking's ability to control its CoT is **low**.

This is a **positive safety property**: if the model cannot deliberately hide or obfuscate its reasoning process, then CoT monitoring remains an effective tool for detecting potential misbehavior.

Formally, let $\mathcal{M}$ denote a model and $\text{CoT}(\mathcal{M}, x)$ denote its chain-of-thought for input $x$. CoT controllability measures:

$$
\kappa = \sup_{\text{adversarial instructions}} \left| \text{Intent}(\text{CoT}(\mathcal{M}, x)) - \text{True Intent}(\mathcal{M}, x) \right|
$$

A low $\kappa$ implies the model's CoT is a **faithful representation** of its actual reasoning, making it monitorable.

### 11.3 Safety Refinements

- Reduced unnecessary refusals and overly caveated responses
- Maintained strong protections against misuse
- Classifiers continue improving; some false positives in request-level blocking may occur

---

## 12. Comprehensive Benchmark Analysis

### 12.1 Academic Reasoning Benchmarks

| Benchmark | GPT-5.4 | GPT-5.4 Pro | GPT-5.2 | GPT-5.2 Pro | $\Delta$ (5.4 vs 5.2) |
|---|---|---|---|---|---|
| Frontier Science Research | $33.0\%$ | $36.7\%$ | $25.2\%$ | — | $+7.8$ pts |
| FrontierMath T1–3 | $47.6\%$ | $50.0\%$ | $40.7\%$ | — | $+6.9$ pts |
| FrontierMath T4 | $27.1\%$ | $38.0\%$ | $18.8\%$ | $31.3\%$ | $+8.3$ pts |
| GPQA Diamond | $92.8\%$ | $94.4\%$ | $92.4\%$ | $93.2\%$ | $+0.4$ pts |
| HLE (no tools) | $39.8\%$ | $42.7\%$ | $34.5\%$ | $36.6\%$ | $+5.3$ pts |
| HLE (with tools) | $52.1\%$ | $58.7\%$ | $45.5\%$ | $50.0\%$ | $+6.6$ pts |

**GPQA Diamond:** Near saturation at $92.8\%$; only $+0.4$ pts improvement suggests this benchmark is approaching its discriminative ceiling for frontier models.

**FrontierMath Tier 4:** The $+8.3$ pt improvement ($18.8\% \to 27.1\%$) is significant, but GPT-5.4 Pro achieves $38.0\%$, indicating that extended compute yields $+10.9$ pts of additional performance on the hardest mathematical problems.

**Humanity's Last Exam:** Tool use provides $+12.3$ pts for GPT-5.4 ($39.8\% \to 52.1\%$) and $+16.0$ pts for GPT-5.4 Pro ($42.7\% \to 58.7\%$), demonstrating the importance of tool-augmented reasoning for the most challenging academic tasks.

### 12.2 Abstract Reasoning

| Benchmark | GPT-5.4 | GPT-5.4 Pro | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|
| ARC-AGI-1 (Verified) | $93.7\%$ | $94.5\%$ | $86.2\%$ | $90.5\%$ |
| ARC-AGI-2 (Verified) | $73.3\%$ | $83.3\%$ | $52.9\%$ | $54.2\%$ (high) |

**ARC-AGI-2:** The $+20.4$ pt jump from GPT-5.2 ($52.9\%$) to GPT-5.4 ($73.3\%$) on ARC-AGI-2 is one of the most dramatic improvements across all benchmarks. GPT-5.4 Pro pushes this to $83.3\%$, a $+30.4$ pt improvement over GPT-5.2 base.

This suggests significant improvements in the model's ability to perform **novel pattern recognition and abstraction** — capabilities that are considered among the most robust indicators of general intelligence.

### 12.3 Pro Model Compute Scaling Analysis

The performance gap between base and Pro variants reveals the **marginal value of extended compute**:

| Benchmark | Base | Pro | $\Delta_{\text{Pro}}$ |
|---|---|---|---|
| ARC-AGI-2 | $73.3\%$ | $83.3\%$ | $+10.0$ pts |
| FrontierMath T4 | $27.1\%$ | $38.0\%$ | $+10.9$ pts |
| HLE (with tools) | $52.1\%$ | $58.7\%$ | $+6.6$ pts |
| BrowseComp | $82.7\%$ | $89.3\%$ | $+6.6$ pts |
| Frontier Science | $33.0\%$ | $36.7\%$ | $+3.7$ pts |
| GPQA Diamond | $92.8\%$ | $94.4\%$ | $+1.6$ pts |

**Pattern:** Pro models yield the largest gains on the **hardest tasks** (ARC-AGI-2, FrontierMath T4) and diminishing returns on tasks approaching saturation (GPQA Diamond). This is consistent with a **compute-optimal scaling** relationship where additional reasoning tokens have highest marginal value on problems at the frontier of the model's capability.

---

## 13. Pricing, Deployment, and API Specification

### 13.1 Availability Matrix

| Surface | Model | Availability |
|---|---|---|
| ChatGPT (Plus, Team, Pro) | GPT-5.4 Thinking | Rolling out March 5, 2026 |
| ChatGPT (Enterprise, Edu) | GPT-5.4 Thinking | Via admin early access settings |
| ChatGPT (Pro, Enterprise) | GPT-5.4 Pro | Available |
| Codex | GPT-5.4 | Available (1M context experimental) |
| API | `gpt-5.4` | Available now |
| API | `gpt-5.4-pro` | Available now |

### 13.2 Model Transition Timeline

| Event | Date |
|---|---|
| GPT-5.4 release | March 5, 2026 |
| GPT-5.2 Thinking → Legacy Models | March 5, 2026 |
| GPT-5.2 Thinking retirement | June 5, 2026 |

### 13.3 API Pricing Summary

| Model | Input | Cached Input | Output |
|---|---|---|---|
| `gpt-5.4` | $\$2.50/\text{M}$ | $\$0.25/\text{M}$ | $\$15/\text{M}$ |
| `gpt-5.4-pro` | $\$30/\text{M}$ | — | $\$180/\text{M}$ |

**Cache discount:** $10\times$ reduction for cached inputs ($\$2.50 \to \$0.25$)

**Pro premium:** $12\times$ input premium, $12\times$ output premium over base model

---

## 14. Critical Analysis and Open Research Questions

### 14.1 Capability Consolidation vs. Specialization

GPT-5.4 represents a bet on **capability consolidation** — a single model excelling across all modalities. However, several data points suggest that specialization still has value:

- GPT-5.3-Codex outperforms GPT-5.4 on Terminal-Bench 2.0 ($77.3\%$ vs. $75.1\%$)
- GPT-5.2 outperforms GPT-5.4 on FinanceAgent ($59.5\%$ vs. $56.0\%$)
- GPT-5.2 marginally outperforms GPT-5.4 on Graphwalks BFS $0$–$128\text{K}$ ($94.0\%$ vs. $93.0\%$)

These inversions suggest a **multi-task interference** phenomenon where consolidating training across all capability domains introduces small regressions on specific narrow benchmarks. The research question is:

$$
\exists \theta^* : \forall d \in \mathcal{D}, \quad \mathcal{L}_d(\theta^*) \leq \mathcal{L}_d(\theta_d^*) \quad ?
$$

i.e., does there exist a single parameter vector that is Pareto-optimal across all individual domain-specific optima? The empirical evidence suggests **not perfectly**, though the gaps are small.

### 14.2 Long-Context Reliability Gap

The sharp performance degradation beyond $128\text{K}$ tokens (Graphwalks BFS: $93.0\% \to 21.4\%$; MRCR: $86.0\% \to 36.6\%$) raises questions about the practical utility of the $1\text{M}$ context window for tasks requiring precise information retrieval. Key research directions:

1. **Attention mechanism scaling:** How does attention sparsity or approximation interact with retrieval accuracy at extreme context lengths?
2. **Training distribution:** Was the model trained on sufficient data at $256\text{K}$–$1\text{M}$ context lengths to generalize reliably?
3. **Position encoding:** Do current positional encoding schemes (e.g., RoPE extensions) maintain sufficient positional discriminability at $>256\text{K}$ positions?

### 14.3 CoT Controllability as a Safety Measure

The finding that GPT-5.4's CoT controllability is "low" is presented as a safety positive. However, this metric requires ongoing monitoring because:

- Future capability improvements may inadvertently increase CoT controllability
- The evaluation methodology must evolve alongside model capabilities
- The relationship between CoT faithfulness and actual model intent remains an open philosophical and empirical question

### 14.4 Compute Scaling Laws for Pro Models

The relationship between Pro model pricing ($12\times$ premium) and performance gains ($+1.6$ to $+10.9$ pts depending on task difficulty) suggests a **sublinear return on compute investment** that is highly task-dependent. Optimal compute allocation strategies should consider:

$$
\text{ROI}_{\text{Pro}} = \frac{\Delta \text{Performance}_{\text{Pro}}}{\Delta \text{Cost}_{\text{Pro}}} = f(\text{task difficulty})
$$

Pro models yield the best ROI on tasks in the **capability frontier zone** — tasks that the base model partially solves but where additional reasoning depth yields substantial accuracy gains.

### 14.5 Naming Convention and Model Versioning

The jump from GPT-5.2 to GPT-5.4 (skipping GPT-5.3 as a general release, with GPT-5.3-Codex being a specialized model) reflects a **semantic versioning signal** that GPT-5.4 subsumes GPT-5.3-Codex capabilities. The stated intent is to "simplify the choice between models," but the coexistence of Instant models and Thinking models evolving at different speeds introduces complexity that will require clear developer communication.

### 14.6 Superhuman Computer Use

GPT-5.4 surpassing human performance on OSWorld-Verified ($75.0\%$ vs. $72.4\%$) is a milestone for AI-driven computer operation. However, the nature of "superhuman" here must be carefully qualified:

- The benchmark measures **task completion rate**, not speed, robustness, or generalization
- Human participants may not represent expert computer users
- Real-world computer use involves edge cases, error recovery, and adaptation not fully captured by the benchmark

### 14.7 The Tool Search Paradigm

Tool search represents a potentially **foundational shift** in how agentic systems interact with tools. By decoupling tool awareness (knowing what tools exist) from tool specification (having full API definitions), GPT-5.4 mirrors how human experts operate: maintaining an index of available resources and looking up specifics on demand. The $47\%$ token reduction with no accuracy loss validates this approach, but open questions remain:

- How does tool search scale to $100$+ tool ecosystems? $1000$+?
- What is the latency overhead of tool definition retrieval?
- Does the tool index itself require curation, or can it be auto-generated?

---

## Appendix A: Complete Evaluation Summary Table

### A.1 Professional Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.3-Codex | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| GDPval | $83.0\%$ | $82.0\%$ | — | $70.9\%$ | $74.1\%$ |
| FinanceAgent v1.1 | $56.0\%$ | $61.5\%$ | $54.0\%$ | $59.5\%$ | — |
| IB Modeling (Internal) | $87.3\%$ | $83.6\%$ | $79.3\%$ | $68.4\%$ | $71.7\%$ |
| OfficeQA | $68.1\%$ | — | $65.1\%$ | $63.1\%$ | — |

### A.2 Coding Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.3-Codex | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| SWE-Bench Pro (Public) | $57.7\%$ | — | $56.8\%$ | $55.6\%$ | — |
| Terminal-Bench 2.0 | $75.1\%$ | — | $77.3\%$ | $62.2\%$ | — |

### A.3 Computer Use & Vision Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.3-Codex | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| OSWorld-Verified | $75.0\%$ | — | $74.0\%$ | $47.3\%$ | — |
| MMMU Pro (no tools) | $81.2\%$ | — | — | $79.5\%$ | — |
| MMMU Pro (with tools) | $82.1\%$ | — | — | $80.4\%$ | — |

### A.4 Tool Use Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.3-Codex | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| BrowseComp | $82.7\%$ | $89.3\%$ | $77.3\%$ | $65.8\%$ | $77.9\%$ |
| MCP Atlas | $67.2\%$ | — | — | $60.6\%$ | — |
| Toolathlon | $54.6\%$ | — | $51.9\%$ | $45.7\%$ | — |
| $\tau^2$-bench Telecom | $98.9\%$ | — | — | $98.7\%$ | — |

### A.5 Academic Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|
| Frontier Science Research | $33.0\%$ | $36.7\%$ | $25.2\%$ | — |
| FrontierMath T1–3 | $47.6\%$ | $50.0\%$ | $40.7\%$ | — |
| FrontierMath T4 | $27.1\%$ | $38.0\%$ | $18.8\%$ | $31.3\%$ |
| GPQA Diamond | $92.8\%$ | $94.4\%$ | $92.4\%$ | $93.2\%$ |
| HLE (no tools) | $39.8\%$ | $42.7\%$ | $34.5\%$ | $36.6\%$ |
| HLE (with tools) | $52.1\%$ | $58.7\%$ | $45.5\%$ | $50.0\%$ |

### A.6 Abstract Reasoning Benchmarks

| Eval | GPT-5.4 | GPT-5.4 Pro | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|
| ARC-AGI-1 (Verified) | $93.7\%$ | $94.5\%$ | $86.2\%$ | $90.5\%$ |
| ARC-AGI-2 (Verified) | $73.3\%$ | $83.3\%$ | $52.9\%$ | $54.2\%$ |

### A.7 Non-Reasoning Benchmarks

| Eval | GPT-5.4 (none) | GPT-5.2 (none) | GPT-4.1 |
|---|---|---|---|
| OmniDocBench (edit dist.) | $0.109$ | $0.140$ | — |
| $\tau^2$-bench Telecom | $64.3\%$ | $57.2\%$ | $43.6\%$ |

---

*All benchmarks conducted with reasoning effort set to* `xhigh` *unless otherwise specified. Results obtained in research environments may differ slightly from production ChatGPT outputs.*
