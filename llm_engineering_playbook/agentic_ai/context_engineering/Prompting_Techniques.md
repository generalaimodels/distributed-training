

# Prompting Techniques: A Principal-Level Technical Report

## Formal Methods, Mathematical Foundations, and SOTA Algorithmic Formulations for Agentic and RAG-Integrated Prompt Engineering

---

## 1. Formal Foundations and Scope

This report provides a rigorous, production-grade treatment of prompting techniques for Large Language Models (LLMs), formalized through probabilistic frameworks, typed pseudo-algorithms, and architectural integration patterns. Every technique is presented at the state-of-the-art (SOTA) level, with mathematical grounding sufficient for reproduction by researchers and implementation by systems engineers.

**Notation Convention:**

| Symbol | Definition |
|---|---|
| $\mathcal{V}$ | Vocabulary set of the tokenizer |
| $x = (x_1, \ldots, x_n)$ | Input token sequence (prompt) |
| $y = (y_1, \ldots, y_m)$ | Output token sequence (completion) |
| $z = (z_1, \ldots, z_l)$ | Latent reasoning trace (chain of thought) |
| $\theta$ | Model parameters |
| $\mathcal{T}$ | Tool registry |
| $\mathcal{M}$ | Memory state (working, session, episodic, semantic) |
| $\mathcal{R}(\cdot)$ | Retrieval function |
| $B$ | Token budget (hard context-window bound) |
| $\pi$ | Policy (prompt-compiled runtime artifact) |
| $\mathcal{D}_k$ | Demonstration set of $k$ exemplars |

**Core Premise:**

Prompting, at principal-level, is not string concatenation. It is the compilation of a *runtime execution artifact* $\pi$ from structured components—role policy, task objective, protocol bindings, tool affordances, retrieval payloads, memory summaries, and execution state—into a deterministic, token-budgeted prefill that maximizes $P(y^* \mid \pi; \theta)$ for desired output $y^*$.

$$
\pi = \texttt{Compile}\bigl(\text{Role}, \text{Task}, \text{Protocol}, \text{Tools}, \mathcal{R}(q), \mathcal{M}, \text{State}\bigr) \quad \text{s.t.} \quad |\pi| \leq B
$$

---

## 2. Chain-of-Thought (CoT) Prompting: Formal Treatment

### 2.1 Standard Autoregressive Baseline

In a standard autoregressive LLM, the joint probability of a response $y$ conditioned on prompt $x$ decomposes as:

$$
P_\theta(y \mid x) = \prod_{t=1}^{m} P_\theta(y_t \mid y_{<t}, x)
$$

This formulation generates $y$ directly from $x$ without any intermediate reasoning structure, resulting in *opaque single-hop inference*. For compositional, multi-step, or evidence-dense tasks (e.g., multi-document RAG synthesis), this produces systematically degraded accuracy due to the absence of explicit intermediate variable binding.

### 2.2 CoT as Latent Variable Marginalization

Chain-of-Thought prompting introduces a latent reasoning trace $z \in \mathcal{Z}$ such that the model first generates $z$ conditioned on $x$, then generates $y$ conditioned on both $x$ and $z$:

$$
P_\theta(y \mid x) = \sum_{z \in \mathcal{Z}} P_\theta(y \mid z, x) \cdot P_\theta(z \mid x)
$$

In practice, we do not enumerate over $\mathcal{Z}$. We use a *point estimate*: the model samples a single trace $\hat{z} \sim P_\theta(z \mid x)$, and conditions on it:

$$
P_\theta(y \mid x) \approx P_\theta(y \mid \hat{z}, x) \quad \text{where} \quad \hat{z} = \arg\max_z P_\theta(z \mid x)
$$

**Key Insight (Wei et al., 2022; Kojima et al., 2022):** The trace $z$ externalizes intermediate variable bindings that the model would otherwise need to represent implicitly in hidden-state activations. This shifts multi-step composition from implicit attention-head computation to explicit token-space reasoning, where each step is auto-regressively conditioned on prior steps.

### 2.3 SOTA CoT: Compressed, Task-Specific Reasoning Traces

Naive CoT ("Let's think step by step") wastes tokens on verbose natural-language narration. SOTA practice compiles the reasoning specification into a *compressed directive* that constrains the trace structure:

**Definition (Compressed CoT Directive):**

$$
\text{CoT}_{\text{compressed}} = \bigl\langle \text{ReasoningSchema}, \text{MaxSteps}, \text{StepFormat}, \text{TerminationCriterion} \bigr\rangle
$$

**Example Instantiation:**

```
Reason in draft form. Each step: ≤8 words.
Schema: [Identify entities] → [Extract relations] → [Resolve conflicts] → [Synthesize answer]
Max steps: 6. Terminate when answer is unambiguous.
```

This achieves two objectives simultaneously:
1. **Token efficiency:** Output token count for $z$ is bounded by $|z| \leq \text{MaxSteps} \times \text{StepTokenBound}$.
2. **Reasoning faithfulness:** The schema enforces task-relevant cognitive operations rather than unconstrained generation.

### 2.4 Pseudo-Algorithm: CoT-Augmented Generation with Token Budget Enforcement

```
Algorithm 1: CoT-Augmented Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x        : user query (token sequence)
  S        : reasoning schema (ordered list of cognitive operations)
  K_max    : maximum reasoning steps
  B_z      : token budget for reasoning trace
  B_y      : token budget for final answer

Output:
  y        : final answer

Procedure:
  1. z ← ∅                                          // Initialize empty trace
  2. FOR i = 1 TO K_max:
       a. s_i ← S[i mod |S|]                        // Select schema operation
       b. z_i ← LLM.generate(
            prompt  = [x; z; "Step {i} ({s_i}):"],
            max_tokens = B_z / K_max,
            stop    = ["\n"]
          )
       c. z ← z ∥ z_i                               // Append step to trace
       d. IF TerminationDetected(z_i):               // Check convergence
            BREAK
       e. IF |z| ≥ B_z:                              // Hard budget enforcement
            BREAK
  3. y ← LLM.generate(
       prompt     = [x; z; "Final Answer:"],
       max_tokens = B_y
     )
  4. RETURN y
```

### 2.5 Mathematical Analysis: When CoT Helps

CoT provides measurable gains when the task requires **compositional generalization**—i.e., when the correct answer $y^*$ is a function of multiple intermediate variables that must be bound sequentially:

$$
y^* = f\bigl(g_1(x), g_2(g_1(x)), \ldots, g_d(\cdots)\bigr)
$$

where $d$ is the **reasoning depth**. Empirically (Feng et al., 2023), standard transformers fail at $d \geq 2$ for out-of-distribution compositions, while CoT maintains accuracy up to the chain length that fits within the context window, because each $g_i$ output is materialized in token space and available for attention at subsequent steps.

**Failure Mode:** CoT degrades when:
- The reasoning schema is misspecified (model follows an incorrect decomposition).
- The trace exceeds the effective attention span, causing early-step information to be lost.
- The model hallucinates intermediate facts that propagate forward.

**Mitigation:** Combine with verification loops (Section 5) and provenance-tagged retrieval (Section 7).

---

## 3. Few-Shot Prompting: Formal Treatment as In-Context Learning

### 3.1 Bayesian Formulation of In-Context Learning

Few-shot prompting provides a demonstration set $\mathcal{D}_k = \{(x_1, y_1), \ldots, (x_k, y_k)\}$ in the context window. The model generates $y$ for a new query $x_{k+1}$ conditioned on both:

$$
P_\theta(y \mid x_{k+1}, \mathcal{D}_k)
$$

**Theoretical Interpretation (Xie et al., 2022; Akyürek et al., 2023):** In-context learning can be understood as *implicit Bayesian inference* over a latent concept variable $c$:

$$
P_\theta(y \mid x_{k+1}, \mathcal{D}_k) = \int P(y \mid x_{k+1}, c) \cdot P(c \mid \mathcal{D}_k) \, dc
$$

The demonstrations $\mathcal{D}_k$ serve as evidence that updates the model's posterior $P(c \mid \mathcal{D}_k)$ over the latent task concept $c$. The model does not update weights; it updates its *effective prior* through attention over the demonstration tokens.

### 3.2 SOTA Exemplar Selection: Optimization over Demonstration Utility

Naive few-shot uses arbitrary or manually curated examples. SOTA practice treats exemplar selection as a combinatorial optimization problem:

**Definition (Exemplar Selection Problem):**

Given a candidate pool $\mathcal{P} = \{(x_i, y_i)\}_{i=1}^{N}$, a query $x_q$, and a quality metric $Q$, select:

$$
\mathcal{D}_k^* = \arg\max_{\mathcal{D}_k \subset \mathcal{P}, |\mathcal{D}_k| = k} Q(x_q, \mathcal{D}_k)
$$

subject to $|\text{Tokenize}(\mathcal{D}_k)| \leq B_{\text{demo}}$ (token budget for demonstrations).

**SOTA Quality Metric (Multi-Signal Ranking):**

$$
Q(x_q, \mathcal{D}_k) = \sum_{i=1}^{k} \Bigl[ \alpha \cdot \text{sim}_{\text{semantic}}(x_q, x_i) + \beta \cdot \text{div}(x_i, \mathcal{D}_k \setminus \{x_i\}) + \gamma \cdot \text{complexity}(y_i) + \delta \cdot \text{recency}(x_i) \Bigr]
$$

where:
- $\text{sim}_{\text{semantic}}$: embedding cosine similarity between the query and exemplar input (ensures relevance).
- $\text{div}$: diversity penalty—maximum marginal relevance (MMR) or DPP-based diversity to avoid redundant demonstrations.
- $\text{complexity}$: reasoning complexity of the exemplar output (prioritize demonstrations that showcase the hardest reasoning the model must replicate).
- $\text{recency}$: temporal freshness for domains with concept drift.

### 3.3 Pseudo-Algorithm: Optimized Few-Shot Construction

```
Algorithm 2: SOTA Few-Shot Exemplar Selection and Prompt Assembly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x_q        : query
  P          : candidate exemplar pool [(x_i, y_i)]
  k          : number of demonstrations
  B_demo     : token budget for demonstrations
  α, β, γ, δ : ranking weights
  embed(·)   : embedding function

Output:
  D_k        : selected demonstration set
  π_fewshot  : compiled few-shot prompt

Procedure:
  1. // Phase 1: Semantic pre-filter (reduce candidate set)
     e_q ← embed(x_q)
     FOR EACH (x_i, y_i) ∈ P:
       scores[i] ← cosine(e_q, embed(x_i))
     P_filtered ← top-50 by scores[i]       // Pre-filter to top-50

  2. // Phase 2: MMR-based diverse selection
     D_k ← ∅
     FOR j = 1 TO k:
       FOR EACH candidate (x_i, y_i) ∈ P_filtered \ D_k:
         relevance_i ← α · scores[i]
         diversity_i ← β · min_{(x_d,·) ∈ D_k} (1 - cosine(embed(x_i), embed(x_d)))
         complexity_i ← γ · normalized_step_count(y_i)
         recency_i   ← δ · decay(timestamp(x_i))
         combined[i] ← relevance_i + diversity_i + complexity_i + recency_i
       best ← argmax_i combined[i]
       IF |Tokenize(D_k ∪ {best})| > B_demo:
         BREAK                               // Hard token budget enforcement
       D_k ← D_k ∪ {best}

  3. // Phase 3: Order demonstrations by increasing complexity
     D_k ← sort(D_k, key=complexity, order=ascending)

  4. // Phase 4: Compile prompt
     π_fewshot ← Compile(
       role_policy,
       task_objective,
       FORMAT_EACH(D_k as "Input: {x_i}\nOutput: {y_i}"),
       "Input: {x_q}\nOutput:"
     )
  5. RETURN D_k, π_fewshot
```

### 3.4 Demonstration Ordering Effects

Ordering is non-trivial. Lu et al. (2022) demonstrated that permutation of demonstrations can cause accuracy variance of up to 30+ percentage points. SOTA ordering strategies:

| Strategy | Formulation | When to Use |
|---|---|---|
| **Ascending complexity** | Sort $\mathcal{D}_k$ by $\text{complexity}(y_i)$ ascending | General reasoning tasks |
| **Recency-last** | Most recent exemplar placed last (closest to query) | Temporal domains |
| **Similarity-last** | Most similar exemplar placed last | Maximizes recency bias of attention |
| **Curriculum ordering** | Interleave easy and hard examples | Domain adaptation |

---

## 4. Combined CoT + Few-Shot: Formal Synthesis

### 4.1 Mathematical Formulation

The combined technique constructs demonstrations $\mathcal{D}_k$ where each exemplar includes both the input and a *full reasoning trace*:

$$
\mathcal{D}_k^{\text{CoT}} = \bigl\{(x_i, z_i, y_i)\bigr\}_{i=1}^{k}
$$

The model then generates:

$$
P_\theta(z, y \mid x_{k+1}, \mathcal{D}_k^{\text{CoT}}) = P_\theta(z \mid x_{k+1}, \mathcal{D}_k^{\text{CoT}}) \cdot P_\theta(y \mid z, x_{k+1}, \mathcal{D}_k^{\text{CoT}})
$$

This simultaneously conditions on:
1. **Task concept** (from the input-output pairs).
2. **Reasoning structure** (from the trace exemplars).
3. **Output format** (from the final answer format in demonstrations).

### 4.2 Token Budget Trade-Off Analysis

Including traces in demonstrations is token-expensive. The budget constraint becomes:

$$
|\text{Tokenize}(\text{SystemPrompt})| + \sum_{i=1}^{k} \bigl(|x_i| + |z_i| + |y_i|\bigr) + |x_q| + B_{\text{output}} \leq B
$$

**Optimization Strategy:** Compress demonstration traces $z_i$ to their *minimal sufficient* form:

$$
z_i^{\text{compressed}} = \arg\min_{z'} |z'| \quad \text{s.t.} \quad \text{InformationContent}(z', z_i) \geq \tau
$$

In practice, this is achieved by rewriting demonstration traces to use abbreviated notation, dropping filler words, and retaining only the variable bindings and logical transitions.

### 4.3 Pseudo-Algorithm: CoT + Few-Shot Compilation

```
Algorithm 3: CoT + Few-Shot Prompt Compilation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x_q           : query
  D_k_cot       : selected demonstrations with traces [(x_i, z_i, y_i)]
  role_policy   : system role and constraints
  cot_schema    : reasoning schema directive
  B             : total context window budget
  B_output      : reserved output budget

Output:
  π             : compiled prompt

Procedure:
  1. B_available ← B - B_output - |Tokenize(role_policy)| - |Tokenize(cot_schema)| - |x_q|

  2. // Compress traces to fit budget
     FOR i = 1 TO |D_k_cot|:
       z_i_compressed ← CompressTrace(z_i, max_tokens = B_available / (2 * |D_k_cot|))
       demo_i ← Format("Q: {x_i}\nReasoning: {z_i_compressed}\nA: {y_i}")
       IF Σ|demo_j| + |demo_i| > B_available:
         BREAK
       ELSE:
         demos ← demos ∥ demo_i

  3. π ← Concatenate(
       role_policy,
       cot_schema,
       "--- Examples ---",
       demos,
       "--- Now solve ---",
       "Q: {x_q}",
       "Reasoning:"
     )

  4. ASSERT |Tokenize(π)| + B_output ≤ B    // Hard budget invariant
  5. RETURN π
```

---

## 5. Tree of Thoughts (ToT): Formal Search-Theoretic Treatment

### 5.1 Formalization as State-Space Search

Tree of Thoughts (Yao et al., 2023) generalizes CoT from a *single sequential trace* to a *search over a tree of partial reasoning states*. The formalization:

**Definition (ToT State Space):**

$$
\mathcal{G} = (S, A, T, V, s_0)
$$

where:
- $S$: set of **thought states** (partial reasoning traces). Each state $s \in S$ is a sequence of thoughts $s = [z_1, z_2, \ldots, z_j]$.
- $A$: set of **thought generation actions**. Each action $a$ proposes a next thought $z_{j+1}$.
- $T: S \times A \to S$: **transition function** $T(s, a) = s \| z_{j+1}$.
- $V: S \to [0, 1]$: **state value function** (LLM-evaluated heuristic estimating probability that state $s$ leads to correct answer).
- $s_0$: **initial state** (the original query $x$).

### 5.2 Search Algorithms

**Breadth-First Search (BFS-ToT):**

At each depth level $d$, maintain a beam of $b$ states. For each state, generate $k$ candidate thoughts. Evaluate all $b \times k$ candidates with $V(\cdot)$, retain top-$b$.

$$
S_{d+1} = \text{top-}b \Bigl\{ T(s, a) \;\Big|\; s \in S_d, \; a \in \text{Generate}(s, k) \Bigr\} \quad \text{ranked by } V(\cdot)
$$

**Depth-First Search with Backtracking (DFS-ToT):**

Explore one branch fully. If $V(s) < V_{\text{threshold}}$, backtrack to the parent state and try the next candidate thought. This trades breadth for depth and is more token-efficient.

### 5.3 Value Function Design

The value function $V(s)$ is itself an LLM call with a structured evaluation prompt:

$$
V(s) = \text{LLM}\bigl(\text{"Evaluate the following partial reasoning. Rate from 0-1 whether it is on track to solve: "} \| s\bigr)
$$

**SOTA Enhancement: Multi-Criteria Value Function**

$$
V(s) = \omega_1 \cdot \text{Coherence}(s) + \omega_2 \cdot \text{Progress}(s) + \omega_3 \cdot \text{Consistency}(s, \mathcal{R}(x))
$$

where:
- $\text{Coherence}(s)$: logical consistency of the reasoning chain (no contradictions).
- $\text{Progress}(s)$: fraction of sub-questions addressed.
- $\text{Consistency}(s, \mathcal{R}(x))$: agreement with retrieved evidence.

### 5.4 Pseudo-Algorithm: BFS-ToT with Provenance

```
Algorithm 4: BFS Tree-of-Thoughts with Retrieval Consistency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x           : query
  R           : retrieved evidence set with provenance tags
  b           : beam width
  k           : branching factor (candidates per state)
  D_max       : maximum depth
  V_threshold : minimum value to retain a state
  ω₁, ω₂, ω₃ : value function weights

Output:
  y*          : best answer

Procedure:
  1. S_0 ← {s_0 = (x, R)}                    // Initial state includes query + evidence
  2. FOR d = 0 TO D_max - 1:
       candidates ← ∅
       FOR EACH s ∈ S_d:
         FOR j = 1 TO k:
           // Generate candidate thought conditioned on state + evidence
           z_j ← LLM.generate(
             prompt = [s; "Generate reasoning step {d+1}, candidate {j}:"],
             temperature = 0.7 + 0.1*j     // Increasing diversity across candidates
           )
           s_new ← T(s, z_j)
           candidates ← candidates ∪ {s_new}

       // Evaluate all candidates
       FOR EACH s_c ∈ candidates:
         v_c ← ω₁·Coherence(s_c) + ω₂·Progress(s_c) + ω₃·Consistency(s_c, R)

       // Prune and select beam
       candidates ← {s_c ∈ candidates | v_c ≥ V_threshold}
       S_{d+1} ← top-b(candidates, key=v_c)

       IF any state in S_{d+1} is terminal:     // Terminal = answer is fully derived
         BREAK

  3. s* ← argmax_{s ∈ S_{D_max}} V(s)
  4. y* ← ExtractAnswer(s*)
  5. RETURN y*

Complexity:
  LLM calls per depth level: b·k (generation) + b·k (evaluation) = 2·b·k
  Total LLM calls: O(D_max · b · k)
  Token cost: O(D_max · b · k · avg_state_length)
```

### 5.5 Complexity and Cost Analysis

| Parameter | Typical Value | Impact |
|---|---|---|
| Beam width $b$ | 3–5 | Higher $b$ → better coverage, linear cost increase |
| Branching factor $k$ | 2–5 | Higher $k$ → more diversity, linear cost increase |
| Max depth $D_{\max}$ | 3–6 | Task-dependent; deeper = more expensive |
| Total LLM calls | $2 \cdot b \cdot k \cdot D_{\max}$ | For $b=3, k=3, D_{\max}=4$: **72 calls** |

**Trade-off:** ToT provides substantially higher accuracy on multi-evidence synthesis tasks (10–30% improvement over single-chain CoT on GSM8K, Creative Writing benchmarks from Yao et al., 2023), but at **$O(b \cdot k)$× cost** per depth level. Use ToT selectively for high-stakes, multi-evidence tasks; fall back to single-chain CoT for routine queries.

---

## 6. ReAct Prompting: Formal Agent-Loop Treatment

### 6.1 Formalization as Interleaved Reasoning-Action Traces

ReAct (Yao et al., 2023b) defines agent execution as an alternating sequence of *thoughts* $t_i$, *actions* $a_i$, and *observations* $o_i$:

$$
\tau = (t_1, a_1, o_1, t_2, a_2, o_2, \ldots, t_n, a_n, o_n, y)
$$

where:
- $t_i \in \mathcal{L}$: a natural-language reasoning step (produced by the LLM).
- $a_i \in \mathcal{A}$: an action from the action space (tool call, retrieval query, API invocation).
- $o_i \in \mathcal{O}$: an observation (result returned by the environment).
- $y$: the final synthesized answer.

The probability factorizes as:

$$
P_\theta(\tau \mid x) = \prod_{i=1}^{n} P_\theta(t_i \mid x, \tau_{<i}) \cdot P_\theta(a_i \mid x, \tau_{<i}, t_i) \cdot \underbrace{P_{\text{env}}(o_i \mid a_i)}_{\text{deterministic}} \cdot P_\theta(y \mid x, \tau)
$$

The critical distinction from CoT: the observations $o_i$ are **not generated by the LLM**. They are returned by the external environment (tool execution, retrieval engine, API). This grounds the reasoning chain in real-world state and provides a *self-correction mechanism*—if an action returns unexpected results, the model can reason about the discrepancy and adjust.

### 6.2 SOTA ReAct: Bounded Control Loop with Verification

Production ReAct implementations must enforce the agent loop invariants: **bounded recursion, rollback conditions, failure-state persistence, and verification gates**.

### 6.3 Pseudo-Algorithm: Production-Grade ReAct Loop

```
Algorithm 5: Bounded ReAct Agent Loop with Verification and Failure Recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x           : user query
  T           : tool registry with typed schemas {tool_id → Schema}
  R           : retrieval engine
  N_max       : maximum reasoning-action cycles
  B_ctx       : context window budget
  retry_budget: per-action retry limit
  V_gate      : verification function (returns PASS/FAIL/UNCERTAIN)

Output:
  y           : final answer
  trace       : full execution trace for observability

Procedure:
  1. trace ← []
  2. working_ctx ← CompileInitialContext(x, T.schemas(), R.summary())
  3. FOR i = 1 TO N_max:

       // ── THINK ──
       t_i ← LLM.generate(
         prompt     = [working_ctx; "Thought {i}:"],
         max_tokens = 100,
         stop       = ["Action:"]
       )
       trace.append(("thought", t_i, timestamp()))

       // ── DECIDE ACTION ──
       a_i ← LLM.generate(
         prompt     = [working_ctx; t_i; "Action:"],
         max_tokens = 80,
         stop       = ["Observation:"],
         constrained_grammar = ActionGrammar(T)    // Typed action parsing
       )
       parsed_action ← ParseAction(a_i, T)        // Validate against tool schemas

       IF parsed_action.type == "FINISH":
         y ← parsed_action.answer
         BREAK

       IF parsed_action == PARSE_ERROR:
         trace.append(("error", "Invalid action syntax", timestamp()))
         working_ctx ← AppendRepairDirective(working_ctx, a_i)
         CONTINUE

       // ── EXECUTE ACTION ──
       o_i ← NULL
       FOR attempt = 1 TO retry_budget:
         TRY:
           o_i ← ExecuteTool(
             tool_id    = parsed_action.tool_id,
             params     = parsed_action.params,
             timeout    = T[parsed_action.tool_id].timeout_class,
             auth_scope = CallerScope(x)            // Least privilege
           )
           BREAK
         CATCH ToolError AS e:
           trace.append(("retry", e, attempt, timestamp()))
           IF attempt == retry_budget:
             o_i ← ErrorObservation(e)
           ELSE:
             WAIT(exponential_backoff(attempt) + jitter())

       trace.append(("action", parsed_action, timestamp()))
       trace.append(("observation", o_i, timestamp()))

       // ── VERIFY ──
       v_result ← V_gate(t_i, a_i, o_i, x)
       IF v_result == FAIL:
         trace.append(("verification_fail", v_result.reason, timestamp()))
         working_ctx ← AppendCorrectionDirective(working_ctx, v_result.reason)
         CONTINUE                                  // Re-reason without counting as progress

       // ── CONTEXT MANAGEMENT ──
       working_ctx ← UpdateContext(
         working_ctx,
         new_content = [t_i, a_i, o_i],
         budget      = B_ctx,
         strategy    = "compress_oldest_observations"  // Sliding window with summarization
       )

  4. // ── EXIT ──
     IF i == N_max AND y is undefined:
       y ← "Unable to resolve within {N_max} steps. Partial trace available."
       trace.append(("timeout", N_max, timestamp()))

  5. RETURN y, trace
```

### 6.4 ReAct + RAG Integration: Iterative Retrieval

In RAG contexts, ReAct enables *iterative retrieval refinement*—the model issues a retrieval action, inspects the results, reformulates the query based on what was missing, and retrieves again:

$$
\tau_{\text{RAG}} = (t_1, a_1^{\text{retrieve}}, o_1^{\text{docs}}, t_2^{\text{analyze}}, a_2^{\text{retrieve\_refined}}, o_2^{\text{docs}}, t_3^{\text{synthesize}}, a_3^{\text{FINISH}}, y)
$$

This is strictly superior to single-shot RAG because the model can:
1. Detect insufficient evidence and issue targeted follow-up queries.
2. Detect contradictory evidence and issue disambiguation queries.
3. Detect low-confidence retrievals and fall back to alternative sources.

---

## 7. Prompting for Tool Usage: Typed Contract Formalization

### 7.1 Tool as a Typed Interface

Every tool in a production agentic system is a *first-class typed contract*, not an informal description. The formal specification:

**Definition (Tool Contract):**

$$
\text{Tool} = \Bigl\langle \text{id}, \text{verb}, \text{description}, \Sigma_{\text{in}}, \Sigma_{\text{out}}, \mathcal{C}_{\text{pre}}, \mathcal{C}_{\text{post}}, \text{auth}, \text{timeout\_class}, \text{side\_effects} \Bigr\rangle
$$

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique tool identifier (e.g., `get_current_weather`) |
| `verb` | `string` | Active verb summarizing the action |
| `description` | `string` | One-paragraph functional description |
| $\Sigma_{\text{in}}$ | `JSON Schema` | Typed input schema with required/optional fields, formats, constraints |
| $\Sigma_{\text{out}}$ | `JSON Schema` | Typed output schema describing return structure |
| $\mathcal{C}_{\text{pre}}$ | `Predicate[]` | Preconditions that must hold before invocation |
| $\mathcal{C}_{\text{post}}$ | `Predicate[]` | Postconditions guaranteed after successful execution |
| `auth` | `AuthScope` | Required authorization level (caller-scoped, not agent-global) |
| `timeout_class` | `enum{fast,medium,slow}` | Latency tier classification |
| `side_effects` | `enum{none,read,write,delete}` | Mutation classification for approval gating |

### 7.2 SOTA Tool Description Engineering

The LLM's tool selection accuracy is a direct function of the description quality. Formal principles:

**Principle 1: Active Verb Naming**
```
✓  get_current_weather       (clear action)
✗  weather_data              (ambiguous noun)
```

**Principle 2: Schema-Complete Input Specification**
```json
{
  "name": "get_current_weather",
  "description": "Retrieves the current weather conditions for a specified city. Returns temperature (high/low in °C), humidity (%), and conditions (string). Limitation: Only supports cities with population > 100,000. Latency: ~200ms.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name in English (e.g., 'Paris', 'Tokyo')"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "celsius"
      }
    },
    "required": ["city"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "high": {"type": "number"},
      "low": {"type": "number"},
      "conditions": {"type": "string"},
      "humidity_pct": {"type": "number"}
    }
  }
}
```

**Principle 3: Explicit Boundary Conditions**

State what the tool *cannot* do. Models exploit negative constraints more reliably than positive-only descriptions:

```
"Limitation: Does not support historical weather queries. 
 For historical data, use get_historical_weather instead."
```

**Principle 4: Routing Guidance via Few-Shot**

Include canonical routing examples in the system prompt:

```
Routing Examples:
  "What's the weather in Paris?"        → get_current_weather(city="Paris")
  "Weather last Tuesday in London"      → get_historical_weather(city="London", date="2025-01-07")
  "Restaurant near Eiffel Tower"        → search_places(query="restaurant", near="Eiffel Tower")
  "Translate 'hello' to Japanese"       → DO NOT use weather tools. Use translate_text.
```

### 7.3 Pseudo-Algorithm: Tool Selection and Dispatch

```
Algorithm 6: Typed Tool Selection, Validation, and Dispatch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  x              : user query
  T              : tool registry [{id, description, Σ_in, Σ_out, C_pre, auth, timeout, side_effects}]
  auth_context   : caller's authorization scope
  approval_gate  : human-in-the-loop approval function (for write/delete tools)

Output:
  result         : tool execution result or error

Procedure:
  1. // ── TOOL SELECTION ──
     available_tools ← FilterByAuth(T, auth_context)
     tool_descriptions ← FormatDescriptions(available_tools)  // Lazy load only relevant schemas

     selection ← LLM.generate(
       prompt = [
         system_policy,
         tool_descriptions,
         routing_examples,
         "Query: {x}",
         "Select tool and parameters as JSON:"
       ],
       response_format = "json_object"
     )

  2. // ── SCHEMA VALIDATION ──
     parsed ← JSON.parse(selection)
     tool ← T[parsed.tool_id]
     IF tool is NULL:
       RETURN Error("Unknown tool: {parsed.tool_id}")

     validation ← JSONSchema.validate(parsed.params, tool.Σ_in)
     IF validation.errors:
       RETURN Error("Parameter validation failed: {validation.errors}")

  3. // ── PRECONDITION CHECK ──
     FOR EACH predicate ∈ tool.C_pre:
       IF NOT predicate.evaluate(parsed.params, auth_context):
         RETURN Error("Precondition failed: {predicate.description}")

  4. // ── APPROVAL GATE (for mutating operations) ──
     IF tool.side_effects ∈ {write, delete}:
       approval ← approval_gate(tool.id, parsed.params, auth_context)
       IF NOT approval.granted:
         RETURN Error("Human approval denied: {approval.reason}")

  5. // ── EXECUTION ──
     result ← ExecuteWithTimeout(
       tool.endpoint,
       parsed.params,
       timeout = tool.timeout_class.duration,
       idempotency_key = hash(x, parsed.tool_id, parsed.params)
     )

  6. // ── OUTPUT VALIDATION ──
     output_valid ← JSONSchema.validate(result, tool.Σ_out)
     IF NOT output_valid:
       RETURN Error("Tool returned invalid output schema")

  7. RETURN result
```

### 7.4 MCP Integration for Tool Discovery

In production agentic architectures, tools are not hardcoded. They are discovered at runtime via the **Model Context Protocol (MCP)**:

```
MCP Discovery Flow:
  1. Agent → MCP Server: ListTools()
     ← Response: [{name, description, inputSchema}]
  
  2. Agent → MCP Server: ListResources()
     ← Response: [{uri, name, mimeType}]
  
  3. Agent → MCP Server: CallTool(name, arguments)
     ← Response: {content: [{type, text}], isError}
```

Tools are loaded lazily—only their names and one-line descriptions are included in the initial context. Full schemas are loaded on-demand when the model signals intent to use a specific tool, minimizing baseline context cost.

---

## 8. Prompt Compilation: The Runtime Artifact Model

### 8.1 Formal Definition

A compiled prompt $\pi$ is a *deterministic, token-budgeted assembly* of structured components:

$$
\pi = \texttt{Compile}\bigl(P_{\text{role}}, P_{\text{task}}, P_{\text{proto}}, P_{\text{tools}}, P_{\text{retrieval}}, P_{\text{memory}}, P_{\text{state}}\bigr)
$$

with the invariant:

$$
\sum_{c \in \{role, task, proto, tools, retrieval, memory, state\}} |P_c| + B_{\text{output}} \leq B
$$

### 8.2 Component Priority Ordering

When the total exceeds $B$, components are trimmed in *reverse priority order*:

| Priority | Component | Trim Strategy |
|---|---|---|
| 1 (highest) | Role Policy $P_{\text{role}}$ | Never trimmed. Contains safety, format, and behavioral constraints. |
| 2 | Task Objective $P_{\text{task}}$ | Never trimmed. Contains the current query and desired outcome. |
| 3 | Protocol Bindings $P_{\text{proto}}$ | Summarize; remove verbose examples. |
| 4 | Tool Affordances $P_{\text{tools}}$ | Lazy-load only relevant tool schemas. |
| 5 | Retrieved Evidence $P_{\text{retrieval}}$ | Rank by provenance score; drop lowest-ranked chunks first. |
| 6 | Memory Summaries $P_{\text{memory}}$ | Compress to key-value summaries; drop episodic details. |
| 7 (lowest) | Execution State $P_{\text{state}}$ | Summarize older steps; retain only last $k$ reasoning steps. |

### 8.3 Pseudo-Algorithm: Prompt Compiler

```
Algorithm 7: Deterministic Prompt Compiler with Token Budget Enforcement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  components   : ordered list [(label, content, priority, compressor)]
  B            : total token budget
  B_output     : reserved output budget

Output:
  π            : compiled prompt (token sequence)

Procedure:
  1. B_available ← B - B_output
  2. allocated ← {}

  3. // Phase 1: Allocate mandatory components (priority 1-2)
     FOR EACH (label, content, priority, _) WHERE priority ≤ 2:
       tokens ← Tokenize(content)
       allocated[label] ← tokens
       B_available -= |tokens|
       IF B_available < 0:
         RAISE CompilationError("Mandatory components exceed budget")

  4. // Phase 2: Allocate remaining components in priority order
     FOR EACH (label, content, priority, compressor) WHERE priority > 2:
       tokens ← Tokenize(content)
       IF |tokens| ≤ B_available:
         allocated[label] ← tokens
         B_available -= |tokens|
       ELSE:
         // Compress to fit remaining budget
         compressed ← compressor(content, target_tokens=B_available * 0.8)
         IF |Tokenize(compressed)| ≤ B_available:
           allocated[label] ← Tokenize(compressed)
           B_available -= |Tokenize(compressed)|
         ELSE:
           // Drop entirely if cannot fit even after compression
           allocated[label] ← ∅
           LOG.warn("Component {label} dropped due to budget constraints")

  5. // Phase 3: Assemble in canonical order
     π ← Concatenate(
       allocated["role"],
       allocated["task"],
       allocated["protocol"],
       allocated["tools"],
       allocated["retrieval"],
       allocated["memory"],
       allocated["state"]
     )

  6. ASSERT |π| + B_output ≤ B
  7. RETURN π
```

---

## 9. Integration Architecture: Prompting Within the Agentic Stack

### 9.1 End-to-End Execution Flow

The following diagram illustrates how prompting techniques integrate with the full agentic execution stack:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        USER REQUEST (JSON-RPC boundary)                     │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  QUERY DECOMPOSITION & REWRITING                                            │
│  • Expand ambiguous queries into explicit sub-queries                       │
│  • Route sub-queries by schema, source, latency tier                        │
│  • Apply CoT decomposition schema for multi-step questions                  │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HYBRID RETRIEVAL ENGINE                                                    │
│  • Exact match (BM25/keyword)                                              │
│  • Semantic search (dense embedding similarity)                             │
│  • Metadata filters (source authority, freshness, lineage)                  │
│  • Provenance-tagged evidence ranking                                       │
│  • Latency-budgeted execution                                              │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PROMPT COMPILER (Algorithm 7)                                              │
│  • Assemble: role + task + protocol + tools + evidence + memory + state     │
│  • Enforce token budget B                                                   │
│  • Select prompting technique: CoT / Few-Shot / CoT+Few-Shot / ToT / ReAct │
│  • Compress and prioritize components                                       │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  AGENT EXECUTION LOOP (Algorithm 5: ReAct / Algorithm 4: ToT)              │
│  plan → decompose → retrieve → act → verify → critique → repair → commit   │
│  • Bounded recursion (N_max)                                               │
│  • Verification gates (V_gate)                                             │
│  • Tool dispatch with typed validation (Algorithm 6)                        │
│  • Context window management (sliding window + compression)                 │
│  • Failure state persistence                                                │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  RESPONSE SYNTHESIS & QUALITY GATE                                          │
│  • Final answer extraction                                                  │
│  • Hallucination check against provenance-tagged evidence                   │
│  • Format compliance verification                                           │
│  • Confidence scoring                                                       │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MEMORY WRITE-BACK                                                          │
│  • Promote non-obvious corrections to episodic memory                       │
│  • Deduplicate, validate provenance, set expiry                             │
│  • Update session memory with interaction context                           │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      RESPONSE (JSON-RPC boundary)                           │
│  + Full execution trace for observability                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Technique Selection Decision Function

The system must select the appropriate prompting technique at runtime based on query characteristics:

$$
\text{Technique}(x) = \begin{cases}
\text{Direct} & \text{if } \text{complexity}(x) < \tau_1 \text{ and } |\mathcal{R}(x)| \leq 1 \\
\text{CoT} & \text{if } \text{complexity}(x) \geq \tau_1 \text{ and } \text{depth}(x) \leq 3 \\
\text{Few-Shot} & \text{if } \text{domain\_specialization}(x) > \tau_2 \text{ and } \text{complexity}(x) < \tau_1 \\
\text{CoT+Few-Shot} & \text{if } \text{domain\_specialization}(x) > \tau_2 \text{ and } \text{complexity}(x) \geq \tau_1 \\
\text{ToT} & \text{if } \text{ambiguity}(x) > \tau_3 \text{ and } |\mathcal{R}(x)| \geq 3 \\
\text{ReAct} & \text{if } \text{tool\_required}(x) = \text{true} \text{ or } \text{iterative\_retrieval}(x) = \text{true}
\end{cases}
$$

where $\tau_1, \tau_2, \tau_3$ are calibrated thresholds determined by offline evaluation over task-specific benchmarks.

---

## 10. Hallucination Control Through Prompting

### 10.1 Provenance-Grounded Generation

Every retrieved chunk $c_i$ carries a provenance tag $\rho_i = (\text{source}, \text{timestamp}, \text{authority\_score}, \text{chunk\_id})$. The prompt directive enforces citation:

```
INSTRUCTION: Base your answer ONLY on the provided evidence. 
For each claim, cite the evidence chunk ID in brackets [chunk_id].
If no evidence supports a claim, state "Insufficient evidence" 
rather than generating unsupported content.
```

### 10.2 Self-Consistency Verification

For critical queries, generate $N$ independent CoT traces $\{z^{(1)}, \ldots, z^{(N)}\}$ (by sampling with temperature $T > 0$) and take the majority answer:

$$
y^* = \arg\max_{y} \sum_{j=1}^{N} \mathbb{1}\bigl[\text{ExtractAnswer}(z^{(j)}) = y\bigr]
$$

**Wang et al. (2023):** Self-consistency improves accuracy by 5–18% over single-chain CoT on arithmetic, commonsense, and symbolic reasoning benchmarks, with diminishing returns beyond $N \approx 10$.

### 10.3 Pseudo-Algorithm: Self-Consistency with Cost Cap

```
Algorithm 8: Self-Consistency Decoding with Cost-Bounded Sampling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  π          : compiled prompt
  N          : maximum sample count
  T          : sampling temperature
  C_max      : maximum token cost
  threshold  : agreement threshold (e.g., 0.7)

Output:
  y*         : majority answer
  confidence : agreement ratio

Procedure:
  1. answers ← []
  2. total_cost ← 0
  3. FOR j = 1 TO N:
       z_j, tokens_used ← LLM.generate(π, temperature=T, return_usage=true)
       total_cost += tokens_used
       y_j ← ExtractAnswer(z_j)
       answers.append(y_j)

       // Early exit if strong consensus already reached
       majority_count ← max(Counter(answers).values())
       IF majority_count / len(answers) ≥ threshold AND len(answers) ≥ 3:
         BREAK

       IF total_cost ≥ C_max:
         BREAK

  4. y* ← mode(answers)
  5. confidence ← count(y* in answers) / len(answers)
  6. RETURN y*, confidence
```

---

## 11. Programmatic Prompt Optimization Frameworks

### 11.1 DSPy: Declarative Prompt Programming

DSPy (Khattab et al., 2023) treats prompting as a *program synthesis* problem rather than manual engineering. Key abstractions:

| DSPy Concept | Formal Analogue | Role |
|---|---|---|
| **Signature** | Type signature $f: \text{Input} \to \text{Output}$ | Declares what the LLM module does |
| **Module** | Parameterized function with learnable prompt parameters | Encapsulates a prompting strategy (CoT, ReAct, etc.) |
| **Teleprompter** | Optimizer $\mathcal{O}$ over prompt parameters | Automatically tunes instructions, few-shot examples, and CoT structure |
| **Metric** | Evaluation function $Q: (y, y^*) \to \mathbb{R}$ | Defines correctness for optimization |

**DSPy Optimization Objective:**

$$
\pi^* = \arg\max_{\pi} \; \mathbb{E}_{(x, y^*) \sim \mathcal{D}_{\text{train}}} \bigl[ Q\bigl(\text{LLM}(x; \pi), y^*\bigr) \bigr]
$$

where $\pi$ includes instructions, demonstrations, and CoT structure—all optimized jointly.

**When to use DSPy:** When you have (a) a measurable quality metric, (b) a training/validation set of 50+ examples, and (c) a pipeline with multiple LLM calls that must be jointly optimized. DSPy eliminates manual prompt iteration and replaces it with automated search.

### 11.2 Comparative Analysis

| Criterion | Manual Prompting | DSPy | LlamaPromptOps | Synalinks |
|---|---|---|---|---|
| **Setup Cost** | Zero | Medium (define signatures, metrics) | Low | Low |
| **Optimization** | Human iteration | Automated (teleprompters) | Template-based | Graph-based |
| **Reproducibility** | Low (ad hoc) | High (deterministic optimization) | Medium | Medium |
| **Multi-stage pipelines** | Manual coordination | Native module composition | Partial | Native |
| **Evaluation integration** | Manual | Built-in metrics and assertions | External | External |
| **Recommended for** | Simple, one-off tasks | Complex multi-stage RAG/agent pipelines | Template standardization | Workflow orchestration |

---

## 12. Summary: Decision Matrix for Prompting Technique Selection

| Technique | Formal Mechanism | Cost Multiplier | Accuracy Gain (vs. Direct) | Best For |
|---|---|---|---|---|
| **Direct** | $P(y \mid x)$ | 1× | Baseline | Simple factual queries, low ambiguity |
| **CoT** | $P(y \mid z, x) \cdot P(z \mid x)$ | 1.3–2× (output tokens) | +10–25% on reasoning tasks | Multi-step reasoning, arithmetic, logic |
| **Compressed CoT** | CoT with $|z| \leq K$ constraint | 1.1–1.5× | +8–22% | Production systems with cost constraints |
| **Few-Shot** | $P(y \mid x, \mathcal{D}_k)$ | 1× (context tokens) | +5–30% on domain tasks | Format compliance, domain specialization |
| **CoT + Few-Shot** | $P(y, z \mid x, \mathcal{D}_k^{\text{CoT}})$ | 1.5–3× (context + output) | +15–35% | Complex domain-specific reasoning |
| **ToT** | BFS/DFS over thought tree | $O(b \cdot k \cdot D_{\max})$ × | +10–30% on multi-path tasks | Ambiguous queries, multi-evidence synthesis |
| **ReAct** | Interleaved $(t, a, o)$ traces | Variable (depends on steps) | Task-dependent | Tool use, iterative retrieval, dynamic tasks |
| **Self-Consistency** | Majority vote over $N$ samples | $N$× | +5–18% | High-stakes tasks requiring reliability |
| **DSPy-Optimized** | Automated $\pi^*$ search | Training cost + 1× inference | +10–40% over manual prompting | Production pipelines with quality metrics |

---

## 13. Production Deployment Considerations

### 13.1 Observability Requirements

Every prompting technique must emit structured traces:

```protobuf
message PromptTrace {
  string   trace_id         = 1;
  string   technique        = 2;   // "cot", "few_shot", "tot", "react"
  int32    total_tokens      = 3;
  int32    prompt_tokens     = 4;
  int32    completion_tokens = 5;
  float    latency_ms        = 6;
  float    confidence        = 7;
  repeated Step steps        = 8;
  string   selected_tool     = 9;
  bool     verification_pass = 10;
  string   failure_reason    = 11;
}
```

### 13.2 Cost Optimization Invariants

1. **Token budget enforcement:** Every compiled prompt $\pi$ must satisfy $|\pi| + B_{\text{output}} \leq B$. Violation is a compilation error, not a runtime surprise.
2. **Technique cost ceiling:** For each technique, define a maximum token cost $C_{\max}^{\text{technique}}$ and enforce it through early termination.
3. **Caching:** Few-shot demonstration blocks and tool schema blocks are *stable across queries*. Cache them as pre-tokenized segments and reuse across compilations.
4. **Prompt-prefix caching:** For models supporting it (e.g., Anthropic prompt caching), structure $\pi$ so that stable components occupy the prefix, minimizing re-processing cost.

### 13.3 Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation |
|---|---|---|
| CoT trace diverges | Step count exceeds $K_{\max}$; value function $V(s) < V_{\min}$ | Hard termination + fallback to direct generation |
| Few-shot exemplar mismatch | Low semantic similarity between selected demos and query | Dynamic re-selection with relaxed diversity constraint |
| ToT combinatorial explosion | Total LLM calls exceed cost ceiling | Reduce beam width $b$ or depth $D_{\max}$ dynamically |
| ReAct infinite loop | Same action repeated $>2$ times | Action deduplication check; force alternative action or FINISH |
| Tool schema validation failure | JSON Schema validation returns errors | Return structured error to LLM for self-repair; max 2 retries |
| Hallucination in final answer | Claim not traceable to any provenance-tagged chunk | Reject answer; re-generate with stricter grounding directive |

---

## 14. Conclusion

Prompting techniques, when formalized as typed, budgeted, and verifiable components of a prompt compilation pipeline, transform from ad hoc string manipulation into rigorous runtime artifacts. The principal-level approach demands:

1. **Mathematical grounding** of each technique's probabilistic mechanism.
2. **Algorithmic specification** with bounded complexity, explicit invariants, and failure recovery.
3. **Integration architecture** that positions prompting within the full agentic stack: retrieval, memory, tool dispatch, verification, and observability.
4. **Production discipline** enforcing token budgets, cost ceilings, schema validation, and structured traces at every boundary.

The techniques—CoT, Few-Shot, ToT, ReAct, and their compositions—are not alternatives to be chosen casually. They are *tools in a typed toolbox*, each with a formal cost-accuracy trade-off profile, to be selected dynamically by a calibrated decision function and executed within a bounded, observable, and recoverable control loop.

---

**References**

- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.*
- Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS 2022.*
- Yao, S. et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *NeurIPS 2023.*
- Yao, S. et al. (2023b). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023.*
- Wang, X. et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023.*
- Khattab, O. et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *arXiv:2310.03714.*
- Xie, S. M. et al. (2022). "An Explanation of In-Context Learning as Implicit Bayesian Inference." *ICLR 2022.*
- Akyürek, E. et al. (2023). "What learning algorithm is in-context learning? Investigations with linear models." *ICLR 2023.*
- Lu, Y. et al. (2022). "Fantastically Ordered Prompts and Where to Find Them." *ACL 2022.*
- Feng, G. et al. (2023). "Towards Revealing the Mystery behind Chain of Thought." *NeurIPS 2023.*
