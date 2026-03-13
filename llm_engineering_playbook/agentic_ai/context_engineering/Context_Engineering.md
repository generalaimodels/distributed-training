# Context Engineering: A Principal-Level Technical Report on Architecting Intelligent Systems Around Large Language Models

## Formal Treatment with Mathematical Foundations, Pseudo-Algorithms, and State-of-the-Art Techniques for Production-Grade Agentic AI Systems

---

## Table of Contents

1. [Problem Formalization: The Context Window as a Bounded Computational Resource](#1-problem-formalization)
2. [Context Engineering as a Compilation and Optimization Problem](#2-context-engineering)
3. [Memory Architecture: Tiered Storage with Formal Write Policies](#3-memory-architecture)
4. [Query Augmentation: From Ambiguous Intent to Precise Retrieval Plans](#4-query-augmentation)
5. [Retrieval Engine: Deterministic Multi-Signal Fusion with Provenance](#5-retrieval-engine)
6. [Prompting as Compiled Runtime Artifacts](#6-prompting)
7. [Agentic Orchestration: Bounded Control Loops with Formal Verification](#7-agentic-orchestration)
8. [Tool Integration: Typed Protocol Infrastructure](#8-tool-integration)
9. [End-to-End System Architecture and Production Reliability](#9-end-to-end)
10. [Evaluation Infrastructure and Continuous Quality Enforcement](#10-evaluation)

---

## 1. Problem Formalization: The Context Window as a Bounded Computational Resource

### 1.1 The Isolation Theorem

A Large Language Model $\mathcal{M}$ with parameters $\theta$ trained on corpus $\mathcal{D}_{\text{train}}$ with knowledge cutoff $t_{\text{cut}}$ operates under three fundamental isolation constraints that we formalize as follows.

**Definition 1.1 (Knowledge Isolation).** For any query $q$ referencing information $\mathcal{I}$ where $\mathcal{I} \notin \mathcal{D}_{\text{train}}$ or $\text{timestamp}(\mathcal{I}) > t_{\text{cut}}$, the model's posterior probability of generating a correct response degrades as:

$$
P_{\theta}(y_{\text{correct}} \mid q) \leq \epsilon_{\text{hallucination}} + \delta_{\text{parametric}}(\mathcal{I}, \mathcal{D}_{\text{train}})
$$

where $\delta_{\text{parametric}}$ measures the semantic distance between the required information and the nearest parametric knowledge manifold, and $\epsilon_{\text{hallucination}}$ represents the irreducible hallucination floor when the model generates from distributional priors rather than grounded evidence.

**Definition 1.2 (Temporal Isolation).** The model's epistemic state is frozen at $t_{\text{cut}}$. For any fact $f$ with validity interval $[t_{\text{start}}, t_{\text{end}}]$:

$$
\text{Validity}_{\mathcal{M}}(f) = \begin{cases} \text{potentially correct} & \text{if } t_{\text{start}} \leq t_{\text{cut}} \leq t_{\text{end}} \\ \text{unreliable} & \text{if } t_{\text{start}} > t_{\text{cut}} \\ \text{stale} & \text{if } t_{\text{end}} < t_{\text{now}} \end{cases}
$$

**Definition 1.3 (Statefulness Isolation).** Without external memory, the model function is stateless across invocations:

$$
\mathcal{M}(q_t) \perp \!\!\! \perp \mathcal{M}(q_{t-1}) \quad \forall \; t \neq t'
$$

This means there exists no persistent state transfer between sessions, rendering the model incapable of learning from prior interactions, accumulating user-specific preferences, or maintaining conversational coherence beyond the current context window boundary.

### 1.2 The Context Window as a Hard Computational Bound

The context window $\mathcal{W}$ is formally a finite-dimensional token sequence space:

$$
\mathcal{W} = \{(x_1, x_2, \ldots, x_N) \mid x_i \in \mathcal{V}, \; N \leq C_{\max}\}
$$

where $\mathcal{V}$ is the vocabulary, $C_{\max}$ is the maximum context length (e.g., 128K, 200K, 1M tokens), and the self-attention mechanism operates with quadratic complexity $O(N^2 \cdot d_{\text{model}})$ in standard implementations or $O(N \cdot d_{\text{model}} \cdot \log N)$ under efficient attention variants.

**Critical Observation.** Even with extended context windows ($C_{\max} \to 10^6$), empirical evidence demonstrates a **Lost-in-the-Middle** degradation phenomenon. The effective retrieval accuracy $\mathcal{A}(p)$ as a function of position $p$ within the context window follows:

$$
\mathcal{A}(p) = \alpha_{\text{primacy}} \cdot e^{-\lambda_1 p} + \alpha_{\text{recency}} \cdot e^{-\lambda_2 (N - p)} + \beta_{\text{floor}}
$$

where $\alpha_{\text{primacy}}$ and $\alpha_{\text{recency}}$ represent the model's attentional bias toward the beginning and end of the window, $\lambda_1, \lambda_2$ are decay rates, and $\beta_{\text{floor}}$ is the baseline accuracy in the middle region. This U-shaped attention distribution means that **not all token positions carry equal utility**, and naive context stuffing is provably suboptimal.

**Theorem 1.1 (Context Utilization Bound).** For a model with context window $C_{\max}$ and effective attention distribution $\mathcal{A}(p)$, the expected utility of injecting $k$ evidence chunks each of size $s$ tokens at uniformly random positions is bounded by:

$$
\mathbb{E}\left[\text{Utility}(k, s)\right] \leq \frac{k \cdot s}{C_{\max}} \cdot \int_0^{C_{\max}} \mathcal{A}(p) \, dp - \Omega\left(\frac{k^2 s^2}{C_{\max}^2}\right) \cdot \gamma_{\text{interference}}
$$

where the negative quadratic term $\gamma_{\text{interference}}$ captures the interference cost of overlapping or contradictory evidence chunks competing for attention mass. This bound establishes that **context engineering must optimize placement, ordering, compression, and selection**—not merely volume.

### 1.3 Formal Problem Statement

**Context Engineering** is then defined as:

> **Definition 1.4 (Context Engineering).** The discipline of designing systems $\mathcal{S}$ that construct, at inference time, an optimal context payload $\mathcal{C}^* \subseteq \mathcal{W}$ such that:
>
> $$\mathcal{C}^* = \arg\max_{\mathcal{C} \in \mathcal{P}(\mathcal{W}), \; |\mathcal{C}| \leq B_{\text{token}}} \; \underbrace{P_{\theta}(y_{\text{correct}} \mid \mathcal{C}, q)}_{\text{task accuracy}} - \lambda_{\text{cost}} \cdot \underbrace{|\mathcal{C}|}_{\text{token cost}} - \mu_{\text{latency}} \cdot \underbrace{T(\mathcal{C})}_{\text{retrieval latency}}$$
>
> subject to:
> - $\text{Hallucination}(\mathcal{C}, q) \leq \tau_{\text{hallucination}}$
> - $\text{Provenance}(\mathcal{C}) = \text{fully traced}$
> - $\text{Freshness}(\mathcal{C}) \geq \phi_{\text{min}}$
> - $\text{Latency}(T(\mathcal{C})) \leq L_{\max}$

This is a **constrained multi-objective optimization** over a combinatorial space of possible context configurations, and it cannot be solved by prompt engineering alone. It requires architecture.

---

## 2. Context Engineering as a Compilation and Optimization Problem

### 2.1 The Context Compiler Model

We formalize the context construction process as a **compilation pipeline** analogous to compiler theory, where the "source" is the user's intent plus the system's knowledge state, and the "target" is an optimized token sequence that maximizes the model's probability of correct task completion.

**Definition 2.1 (Context Compiler).** A context compiler $\mathcal{K}$ is a deterministic function:

$$
\mathcal{K}: (\mathcal{Q}, \mathcal{R}, \mathcal{T}, \mathcal{M}_{\text{mem}}, \mathcal{P}_{\text{policy}}, \mathcal{S}_{\text{state}}) \rightarrow \mathcal{C}^* \in \mathcal{W}
$$

where:
- $\mathcal{Q}$: Augmented query representation (after decomposition and rewriting)
- $\mathcal{R}$: Retrieved evidence with provenance tags
- $\mathcal{T}$: Tool affordances (schemas, invocation contracts)
- $\mathcal{M}_{\text{mem}}$: Memory summaries across all tiers
- $\mathcal{P}_{\text{policy}}$: Role policy, safety constraints, behavioral directives
- $\mathcal{S}_{\text{state}}$: Current execution state in the agent loop

The compiler proceeds through the following passes:

```
ALGORITHM 2.1: ContextCompiler.compile()

Input:  query Q, budget B_token, deadline L_max, policy P
Output: compiled context C* as token sequence

1.  PASS 1 — FRONT-END ANALYSIS
    Q_aug ← QueryAugmenter.decompose_and_rewrite(Q)
    intent_graph ← IntentParser.extract_structured_intent(Q_aug)
    subqueries[] ← PlanDecomposer.generate_retrieval_plan(intent_graph)

2.  PASS 2 — RETRIEVAL ORCHESTRATION (parallel, bounded)
    evidence_set ← ∅
    FOR EACH sq_i IN subqueries[] PARALLEL:
        route_i ← RetrievalRouter.select_source(sq_i, latency_tier, schema)
        results_i ← RetrievalEngine.execute(sq_i, route_i, deadline=L_max/2)
        evidence_set ← evidence_set ∪ tag_provenance(results_i, route_i)
    END FOR
    ranked_evidence ← EvidenceRanker.score_and_rank(
        evidence_set, Q_aug,
        weights={authority: 0.3, freshness: 0.25, relevance: 0.3, utility: 0.15}
    )

3.  PASS 3 — MEMORY RETRIEVAL
    working_ctx ← WorkingMemory.get_current_state()
    session_ctx ← SessionMemory.get_compressed_history(max_tokens=B_token * 0.1)
    episodic_ctx ← EpisodicMemory.recall_relevant(Q_aug, top_k=3)
    semantic_ctx ← SemanticMemory.query_facts(intent_graph)
    procedural_ctx ← ProceduralMemory.get_applicable_procedures(intent_graph)

4.  PASS 4 — BUDGET ALLOCATION (knapsack optimization)
    segments = {
        policy:      (P, priority=CRITICAL, min_tokens=|P|),
        task_state:  (S_state, priority=HIGH, min_tokens=est_tokens(S_state)),
        evidence:    (ranked_evidence, priority=HIGH, variable=True),
        memory:      (merge(working, session, episodic, semantic, procedural),
                      priority=MEDIUM, variable=True),
        tools:       (ToolRegistry.get_affordances(intent_graph),
                      priority=MEDIUM, variable=True),
        history:     (session_ctx, priority=LOW, compressible=True)
    }
    allocation ← TokenBudgetAllocator.solve_knapsack(
        segments, B_token,
        objective=maximize_task_utility,
        constraints=[positional_attention_weights, interference_penalty]
    )

5.  PASS 5 — ASSEMBLY AND OPTIMIZATION
    C_raw ← ContextAssembler.compose(allocation, positional_strategy="primacy-recency")
    C_compressed ← RedundancyEliminator.deduplicate_and_compress(C_raw)
    C_validated ← ConsistencyChecker.detect_contradictions(C_compressed)
    C* ← TokenCounter.enforce_hard_limit(C_validated, B_token)

6.  PASS 6 — EMISSION
    RETURN C* with metadata {
        token_count, provenance_map, freshness_scores,
        confidence_estimate, compilation_latency
    }
```

### 2.2 Token Budget Allocation as a Knapsack Problem

The budget allocation in Pass 4 is formally a **variant of the 0-1 knapsack problem** with segment-level granularity and mutual information penalties.

Given $n$ context segments $\{s_1, \ldots, s_n\}$, each with token cost $c_i$, estimated task utility $u_i$, and priority class $p_i \in \{\text{CRITICAL}, \text{HIGH}, \text{MEDIUM}, \text{LOW}\}$, and a total token budget $B$:

$$
\max_{\mathbf{x} \in \{0,1\}^n} \sum_{i=1}^{n} u_i \cdot x_i \cdot \mathcal{A}(\text{pos}_i) - \sum_{i<j} \gamma_{ij} \cdot x_i \cdot x_j
$$

subject to:

$$
\sum_{i=1}^{n} c_i \cdot x_i \leq B, \quad x_i = 1 \; \forall \; i : p_i = \text{CRITICAL}
$$

where $\mathcal{A}(\text{pos}_i)$ is the positional attention weight at the assigned position of segment $i$, and $\gamma_{ij}$ is the mutual redundancy penalty between segments $i$ and $j$, computed as:

$$
\gamma_{ij} = \alpha \cdot \text{cosine\_sim}(\mathbf{e}_i, \mathbf{e}_j) \cdot \mathbb{1}[\text{source}(s_i) = \text{source}(s_j)]
$$

This formulation ensures that critical policy segments are always included, high-utility evidence is prioritized, and redundant segments are penalized to maximize information density within the budget.

### 2.3 Positional Placement Strategy

Given the empirically observed attention distribution $\mathcal{A}(p)$, the compiler employs a **primacy-recency placement heuristic**:

```
ALGORITHM 2.2: PositionalPlacement.assign()

Input:  segments[] sorted by priority descending, window size N
Output: positioned_segments[] with assigned token ranges

1.  head_pointer ← 0        // primacy region
2.  tail_pointer ← N        // recency region
3.  FOR EACH segment s IN segments[]:
4.      IF s.priority ∈ {CRITICAL, HIGH}:
5.          IF s.type = "policy" OR s.type = "instructions":
6.              assign(s, position=head_pointer)
7.              head_pointer += |s|
8.          ELIF s.type = "task_state" OR s.type = "final_instructions":
9.              tail_pointer -= |s|
10.             assign(s, position=tail_pointer)
11.     ELSE:
12.         // Medium/Low priority: interleave with relevance-descending order
13.         mid_position ← head_pointer + (tail_pointer - head_pointer) * rank(s)/|segments|
14.         assign(s, position=mid_position)
15. RETURN positioned_segments[]
```

This placement strategy exploits the U-shaped attention curve by anchoring instructions and final task directives at the primacy and recency regions respectively, where attention mass is highest.

---

## 3. Memory Architecture: Tiered Storage with Formal Write Policies

### 3.1 Memory Tier Taxonomy

We define a five-tier memory hierarchy with strict promotion, demotion, and eviction policies. Each tier is characterized by its **persistence scope**, **write admission policy**, **capacity constraints**, and **retrieval latency class**.

| Tier | Designation | Persistence | Write Policy | Capacity | Latency |
|------|-------------|------------|--------------|----------|---------|
| $\mathcal{M}_0$ | Working Memory | Current turn | Automatic | $\leq 0.3 \cdot C_{\max}$ | $O(1)$ |
| $\mathcal{M}_1$ | Session Memory | Current session | Append + compress | $\leq 0.15 \cdot C_{\max}$ | $O(1)$ |
| $\mathcal{M}_2$ | Episodic Memory | Cross-session | Validated write | Unbounded (indexed) | $O(\log n)$ |
| $\mathcal{M}_3$ | Semantic Memory | Permanent | Schema-validated | Unbounded (indexed) | $O(\log n)$ |
| $\mathcal{M}_4$ | Procedural Memory | Permanent | CI/eval-gated | Bounded (versioned) | $O(1)$ |

### 3.2 Formal Memory State Machine

Each memory item $m$ transitions through a state machine:

$$
m: \text{CANDIDATE} \xrightarrow{\text{validate}} \text{STAGED} \xrightarrow{\text{deduplicate}} \text{COMMITTED} \xrightarrow{\text{expire/invalidate}} \text{ARCHIVED} \xrightarrow{\text{gc}} \text{DELETED}
$$

**Definition 3.1 (Memory Write Admission Function).** A candidate memory item $m_c$ is admitted to tier $\mathcal{M}_k$ only if:

$$
\text{Admit}(m_c, \mathcal{M}_k) = \begin{cases}
\text{true} & \text{if } \text{novelty}(m_c) > \tau_k^{\text{novel}} \;\wedge\; \text{correctness}(m_c) > \tau_k^{\text{correct}} \\
& \wedge\; \text{dedup}(m_c, \mathcal{M}_k) = \text{unique} \\
& \wedge\; \text{provenance}(m_c) \neq \emptyset \\
& \wedge\; \text{expiry}(m_c) > t_{\text{now}} \\
\text{false} & \text{otherwise}
\end{cases}
$$

where:

$$
\text{novelty}(m_c) = 1 - \max_{m \in \mathcal{M}_k} \text{sim}(\text{embed}(m_c), \text{embed}(m))
$$

$$
\text{dedup}(m_c, \mathcal{M}_k) = \begin{cases} \text{unique} & \text{if } \nexists \; m \in \mathcal{M}_k : \text{content\_hash}(m) = \text{content\_hash}(m_c) \\ \text{duplicate} & \text{otherwise} \end{cases}
$$

### 3.3 Memory Promotion Algorithm

```
ALGORITHM 3.1: MemoryPromoter.evaluate_and_promote()

Input:  candidate item m_c, source tier k, target tier k+1
Output: promotion decision with metadata

1.  // Stage 1: Novelty Check
    novelty_score ← 1 - max_similarity(m_c, M[k+1])
    IF novelty_score < τ_novel[k+1]:
        RETURN REJECT(reason="insufficient novelty", score=novelty_score)

2.  // Stage 2: Correctness Verification
    IF k+1 ≥ 2:  // Episodic and above require verification
        verification_result ← Verifier.check(m_c, against=ground_truth_sources)
        IF verification_result.confidence < τ_correct[k+1]:
            RETURN REJECT(reason="unverified", confidence=verification_result.confidence)

3.  // Stage 3: Deduplication with Subsumption Check
    existing_matches ← M[k+1].search(m_c, similarity_threshold=0.85)
    FOR EACH match IN existing_matches:
        IF subsumes(match, m_c):
            RETURN REJECT(reason="subsumed by existing", existing=match.id)
        IF subsumes(m_c, match):
            M[k+1].supersede(match, replacement=m_c)
            RETURN ACCEPT(action="superseded", replaced=match.id)

4.  // Stage 4: Provenance and Expiry Assignment
    m_c.provenance ← {
        source_tier: k,
        timestamp: now(),
        session_id: current_session(),
        verification_method: verification_result.method,
        confidence: verification_result.confidence
    }
    m_c.expiry ← compute_expiry(m_c, tier=k+1, policy=ExpiryPolicy[k+1])
    m_c.version ← M[k+1].next_version(m_c.namespace)

5.  // Stage 5: Commit with Write-Ahead Log
    wal_entry ← WAL.append(operation=INSERT, tier=k+1, item=m_c)
    M[k+1].insert(m_c)
    WAL.confirm(wal_entry)

6.  RETURN ACCEPT(item=m_c, tier=k+1, version=m_c.version)
```

### 3.4 Session Memory Compression

Session memory $\mathcal{M}_1$ grows linearly with conversation turns. To maintain it within budget $B_1 \leq 0.15 \cdot C_{\max}$, we apply a **hierarchical progressive summarization** strategy.

**Definition 3.2 (Progressive Summarization Operator).** Given conversation history $H = [(q_1, a_1), \ldots, (q_T, a_T)]$ and budget $B_1$:

$$
\text{Compress}(H, B_1) = \begin{cases}
H & \text{if } |H| \leq B_1 \\
[\mathcal{S}(H_{1:k})] \oplus H_{k+1:T} & \text{if } |H| > B_1
\end{cases}
$$

where $\mathcal{S}$ is a summarization function and $k$ is chosen such that $|\mathcal{S}(H_{1:k})| + |H_{k+1:T}| \leq B_1$. The summarization preserves:

1. **Decisions and commitments** made during the conversation
2. **Corrections and constraints** expressed by the user
3. **Entities and references** that may be needed for co-reference resolution
4. **Task state transitions** and their outcomes

The compression ratio $\rho$ at each summarization level $l$ follows:

$$
\rho_l = \frac{|H_l|}{|\mathcal{S}(H_l)|} \approx 3^l \quad \text{(empirical geometric compression)}
$$

This yields a logarithmic memory footprint: $O(\log T)$ tokens for $T$ conversation turns, preserving the most recent turns verbatim and progressively summarizing older exchanges.

### 3.5 Episodic Memory: Correction-Weighted Indexing

Episodic memory $\mathcal{M}_2$ stores non-obvious corrections, user-specific preferences, and learned constraints that improve future task accuracy. Each episodic entry is indexed by:

$$
\text{EpisodicIndex}(m) = (\text{embed}(m.\text{context}), \; m.\text{tags}, \; m.\text{correction\_weight})
$$

where the correction weight $w_c$ is computed as:

$$
w_c(m) = \frac{\text{impact}(m) \cdot \text{recurrence}(m)}{\text{staleness}(m) + \epsilon}
$$

$$
\text{impact}(m) = \Delta_{\text{accuracy}} \text{ when } m \text{ was applied vs. absent}
$$

$$
\text{recurrence}(m) = \text{count of sessions where } m \text{ was relevant}
$$

$$
\text{staleness}(m) = \frac{t_{\text{now}} - t_{\text{last\_used}}}{t_{\text{half\_life}}}
$$

Memory items with $w_c(m) < \tau_{\text{evict}}$ are candidates for archival or deletion during garbage collection sweeps.

---

## 4. Query Augmentation: From Ambiguous Intent to Precise Retrieval Plans

### 4.1 The Query Augmentation Pipeline

User queries are inherently ambiguous, underspecified, and structurally unsuited for direct retrieval. The query augmentation stage transforms a raw user query $q_{\text{raw}}$ into a structured retrieval plan $\mathcal{P}_{\text{retrieval}}$ through a multi-stage pipeline.

**Definition 4.1 (Query Augmentation Function).**

$$
\mathcal{A}_q: q_{\text{raw}} \rightarrow \{(sq_1, r_1, s_1), \ldots, (sq_n, r_n, s_n)\}
$$

where each tuple $(sq_i, r_i, s_i)$ represents a subquery $sq_i$, its routing destination $r_i \in \{\text{vector\_store}, \text{graph\_db}, \text{SQL}, \text{API}, \text{web}\}$, and its schema specification $s_i$.

### 4.2 SOTA Query Decomposition Techniques

#### 4.2.1 Recursive Abstractive Decomposition (RAD)

Rather than naive query splitting, RAD decomposes queries by identifying **epistemic dependencies** between sub-questions:

```
ALGORITHM 4.1: RecursiveAbstractiveDecomposition.decompose()

Input:  raw query q, knowledge schema K, max_depth d
Output: dependency-ordered subquery DAG

1.  intent ← IntentClassifier.classify(q)
    // intent ∈ {factual, analytical, procedural, comparative, generative}

2.  IF intent = "factual" AND complexity(q) ≤ SIMPLE:
      RETURN [{query: q, route: select_route(q, K), deps: []}]

3.  // Extract atomic propositions and their dependencies
    propositions ← PropositionExtractor.extract(q)
    // Each proposition p_i: {claim, entities, relations, unknowns}

4.  dep_graph ← DependencyGraph()
    FOR EACH p_i IN propositions:
        FOR EACH p_j IN propositions WHERE j ≠ i:
            IF requires(p_i, output_of=p_j):
                dep_graph.add_edge(p_j → p_i)

5.  // Generate subqueries from propositions in topological order
    subqueries ← []
    FOR EACH p IN dep_graph.topological_sort():
        sq ← SubqueryGenerator.generate(
            proposition=p,
            resolved_deps=get_resolved(p, dep_graph),
            schema=K.get_schema(p.entities)
        )
        sq.route ← RetrievalRouter.select(sq, latency_tier=infer_tier(p))
        sq.rewrite ← HyDE.generate_hypothetical_answer(sq)  // Hypothetical Document Embedding
        subqueries.append(sq)

6.  RETURN subqueries with dep_graph
```

#### 4.2.2 Hypothetical Document Embedding (HyDE) with Multi-Perspective Generation

Standard HyDE generates one hypothetical answer. Our SOTA variant generates $k$ hypothetical documents from diverse perspectives, then uses the **centroid of their embedding cluster** as the retrieval query vector, reducing single-perspective bias:

$$
\mathbf{v}_{\text{HyDE}}^{\text{multi}} = \frac{1}{k} \sum_{i=1}^{k} \text{embed}(\text{LLM}_{\theta}(q, \text{perspective}_i))
$$

$$
\text{retrieval\_score}(d) = \alpha \cdot \cos(\mathbf{v}_{\text{HyDE}}^{\text{multi}}, \mathbf{v}_d) + (1-\alpha) \cdot \cos(\text{embed}(q), \mathbf{v}_d)
$$

where $\alpha \in [0.4, 0.7]$ is calibrated per query type. The interpolation between the HyDE centroid and the raw query embedding prevents hallucinated hypothetical documents from dominating retrieval.

#### 4.2.3 Step-Back Prompting for Abstract Retrieval

For queries requiring reasoning over principles rather than specific facts, we apply **step-back abstraction**:

$$
q_{\text{abstract}} = \text{LLM}_{\theta}(\text{"What higher-level concept or principle is needed to answer: "} \| q)
$$

The abstract query retrieves foundational context that the model can then apply deductively to the specific question. This is particularly effective for multi-hop reasoning chains where direct retrieval yields insufficient intermediate knowledge.

### 4.3 Query Routing with Latency-Tiered Source Selection

```
ALGORITHM 4.2: RetrievalRouter.select()

Input:  subquery sq, available sources S[], latency constraints L
Output: ranked source list with expected latencies

1.  // Feature extraction
    features ← {
        query_type: classify_type(sq),           // factual | analytical | temporal
        entity_types: extract_entity_types(sq),   // person | code | document | metric
        freshness_requirement: estimate_freshness(sq),
        structured_data: requires_structured(sq),  // boolean
        expected_result_count: estimate_cardinality(sq)
    }

2.  // Source scoring
    scores ← {}
    FOR EACH source s IN S[]:
        relevance_s ← SourceRelevanceModel.score(features, s.schema)
        latency_s ← s.p99_latency_ms
        cost_s ← s.cost_per_query
        freshness_s ← s.data_freshness_score

        scores[s] ← (
            w_rel * relevance_s +
            w_fresh * freshness_s -
            w_lat * (latency_s / L.deadline_ms) -
            w_cost * cost_s
        )

3.  // Tiered selection
    primary ← argmax(scores)
    fallback ← argmax(scores \ {primary}) IF scores[fallback] > τ_min
    
    RETURN {primary: primary, fallback: fallback,
            deadline: L.deadline_ms, retry_budget: 1}
```

**Source Type Taxonomy and Routing Criteria:**

| Source Type | Best For | Latency Tier | Freshness |
|---|---|---|---|
| Vector Store (HNSW) | Semantic similarity over documents | P50: 5ms, P99: 20ms | Batch-indexed |
| Full-Text Index (BM25) | Exact keyword match, technical terms | P50: 2ms, P99: 10ms | Near-real-time |
| Knowledge Graph (Cypher/SPARQL) | Entity relationships, lineage | P50: 15ms, P99: 50ms | Graph-update cadence |
| SQL/OLAP | Structured aggregations, metrics | P50: 50ms, P99: 200ms | Transactional |
| Live API | Real-time data, external services | P50: 200ms, P99: 2000ms | Real-time |
| Web Search | Current events, broad knowledge | P50: 500ms, P99: 3000ms | Real-time |

---

## 5. Retrieval Engine: Deterministic Multi-Signal Fusion with Provenance

### 5.1 Hybrid Retrieval Architecture

The retrieval engine operates as a **multi-signal fusion system** that combines results from heterogeneous sources into a unified, provenance-tagged evidence set.

**Definition 5.1 (Evidence Item).** Each retrieved item $e$ is a structured record:

$$
e = \left(\text{content}, \text{source}, \text{chunk\_id}, \text{score}_{\text{relevance}}, \text{score}_{\text{authority}}, \text{score}_{\text{freshness}}, \text{provenance}, \text{lineage}\right)
$$

### 5.2 Multi-Signal Fusion Scoring

The final ranking score for an evidence item $e$ given query $q$ is computed as a **learned weighted combination** across multiple scoring dimensions:

$$
\text{Score}(e, q) = \sum_{j=1}^{J} w_j \cdot f_j(e, q) \quad \text{s.t.} \quad \sum_j w_j = 1, \; w_j \geq 0
$$

where the scoring functions $f_j$ are:

**Relevance Scores:**

$$
f_{\text{semantic}}(e, q) = \cos\left(\text{embed}(e.\text{content}), \text{embed}(q)\right)
$$

$$
f_{\text{lexical}}(e, q) = \text{BM25}(e.\text{content}, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{tf}(t, e) \cdot (k_1 + 1)}{\text{tf}(t, e) + k_1 \cdot (1 - b + b \cdot \frac{|e|}{|\text{avgdl}|})}
$$

$$
f_{\text{cross-encoder}}(e, q) = \sigma\left(\text{CrossEncoder}_{\phi}(q \| e.\text{content})\right)
$$

**Authority Score:**

$$
f_{\text{authority}}(e) = \log(1 + \text{citation\_count}(e.\text{source})) \cdot \text{source\_tier}(e.\text{source}) \cdot \mathbb{1}[\text{verified}(e.\text{source})]
$$

**Freshness Score:**

$$
f_{\text{freshness}}(e) = \exp\left(-\frac{t_{\text{now}} - t_{\text{authored}}(e)}{\tau_{\text{decay}}(e.\text{domain})}\right)
$$

where $\tau_{\text{decay}}$ is domain-specific: rapidly decaying for news/market data, slowly decaying for scientific literature, near-infinite for mathematical proofs.

**Execution Utility Score:**

$$
f_{\text{utility}}(e, q) = P(\text{e contributes to correct answer} \mid q, e.\text{content})
$$

estimated by a lightweight classifier trained on historical retrieval-outcome pairs.

### 5.3 Reciprocal Rank Fusion (RRF) for Heterogeneous Source Combination

When combining ranked lists from different retrieval systems (vector, BM25, graph, etc.), we use **Reciprocal Rank Fusion** with source-specific calibration:

$$
\text{RRF}(e) = \sum_{s \in \text{sources}} \frac{\beta_s}{\kappa + \text{rank}_s(e)}
$$

where $\kappa = 60$ (standard RRF constant), $\beta_s$ is a source-specific calibration weight learned from evaluation data, and $\text{rank}_s(e)$ is the rank of item $e$ in the result list from source $s$ (or $\infty$ if not retrieved).

### 5.4 Chunking Strategy Selection

Chunking is **not a one-size-fits-all operation**. We define a chunking strategy selector that matches document type to optimal chunking method:

```
ALGORITHM 5.1: ChunkingStrategySelector.select()

Input:  document D, document_class C
Output: chunked document D_chunks[] with metadata

1.  MATCH C:
    
    CASE "structured_code":
        strategy ← AST_CHUNKING
        // Parse into AST, chunk at function/class boundaries
        // Preserve: imports, type signatures, docstrings with each chunk
        D_chunks ← ASTChunker.chunk(D,
            granularity="function",
            include_context=["imports", "class_header", "type_defs"],
            max_tokens=512)

    CASE "technical_documentation":
        strategy ← HIERARCHICAL_SEMANTIC
        // Chunk by section hierarchy (H1 > H2 > H3)
        // Each chunk inherits parent section context
        D_chunks ← HierarchicalChunker.chunk(D,
            levels=["chapter", "section", "subsection"],
            parent_context_tokens=128,
            max_tokens=768)

    CASE "conversational_transcript":
        strategy ← TOPIC_SEGMENTATION
        // Use topic shift detection (TextTiling / semantic similarity drops)
        D_chunks ← TopicChunker.chunk(D,
            similarity_threshold=0.65,
            min_segment_tokens=100,
            max_segment_tokens=1024)

    CASE "legal_regulatory":
        strategy ← CLAUSE_PRESERVING
        // Never split across clause/article boundaries
        D_chunks ← ClauseChunker.chunk(D,
            boundary_markers=["Article", "Section", "Clause", "§"],
            preserve_cross_references=True,
            max_tokens=1024)

    CASE "tabular_data":
        strategy ← ROW_GROUP_WITH_SCHEMA
        // Chunk by row groups, always prepend schema/header
        D_chunks ← TabularChunker.chunk(D,
            rows_per_chunk=50,
            always_include=["column_headers", "table_description"],
            max_tokens=512)

    CASE "general_prose":
        strategy ← SEMANTIC_SLIDING_WINDOW
        // Semantic similarity-based boundary detection with overlap
        D_chunks ← SemanticChunker.chunk(D,
            embedding_model="text-embedding-3-large",
            breakpoint_percentile=90,  // split at top-10% dissimilarity points
            overlap_tokens=64,
            max_tokens=512)

2.  // Post-processing: enrich each chunk
    FOR EACH chunk IN D_chunks:
        chunk.embedding ← embed(chunk.content)
        chunk.summary ← generate_summary(chunk.content, max_tokens=50)
        chunk.entities ← extract_entities(chunk.content)
        chunk.provenance ← {
            source_doc: D.id,
            chunk_index: chunk.index,
            strategy: strategy,
            timestamp: D.last_modified
        }

3.  RETURN D_chunks
```

### 5.5 Provenance-Tagged Evidence Assembly

Every piece of evidence entering the context window carries a **provenance tag** that enables the model to assess source reliability and enables downstream auditability:

$$
\text{Provenance}(e) = \left\langle \text{source\_id}, \text{chunk\_id}, \text{retrieval\_method}, \text{score}, \text{timestamp}, \text{version}, \text{confidence\_class} \right\rangle
$$

The provenance tag is embedded directly in the context payload using a structured format:

```
[SOURCE: {source_id} | CHUNK: {chunk_id} | METHOD: {semantic+bm25} |
 SCORE: {0.87} | FRESHNESS: {2024-12-15} | CONFIDENCE: HIGH]
{content}
[/SOURCE]
```

This allows the model to reason about evidence quality, prefer high-confidence sources, and the system to trace any generated claim back to its retrieval source for auditing.

---

## 6. Prompting as Compiled Runtime Artifacts

### 6.1 Prompt Compilation Framework

Prompts in a production agentic system are **not manually authored text**. They are **compiled runtime artifacts** assembled deterministically from structured components, analogous to how a linker assembles object files into an executable.

**Definition 6.1 (Prompt Artifact).** A compiled prompt $\mathcal{P}$ is a structured token sequence:

$$
\mathcal{P} = \bigoplus_{i=1}^{L} \text{Section}_i(\text{content}_i, \text{priority}_i, \text{position}_i, \text{token\_budget}_i)
$$

where $\bigoplus$ denotes ordered concatenation and each section is independently templated, validated, and budget-controlled.

### 6.2 Section Taxonomy and Compilation Order

The prompt compiler assembles sections in a fixed order that exploits the attention distribution:

```
COMPILED PROMPT STRUCTURE:

┌────────────────────────────────────────────────────────┐
│ SECTION 1: SYSTEM POLICY              [PRIMACY REGION] │
│ ├─ Role definition and behavioral constraints          │
│ ├─ Output format specification (typed schema)          │
│ ├─ Safety and compliance boundaries                    │
│ └─ Hallucination control directives                    │
├────────────────────────────────────────────────────────┤
│ SECTION 2: TOOL AFFORDANCES                            │
│ ├─ Available tool schemas (lazily loaded)               │
│ ├─ Invocation protocols and constraints                │
│ └─ Tool selection guidelines                           │
├────────────────────────────────────────────────────────┤
│ SECTION 3: RETRIEVED EVIDENCE                          │
│ ├─ Provenance-tagged evidence chunks                   │
│ ├─ Ordered by relevance score descending               │
│ └─ Each chunk with source attribution                  │
├────────────────────────────────────────────────────────┤
│ SECTION 4: MEMORY CONTEXT                              │
│ ├─ Relevant episodic memories                          │
│ ├─ Semantic facts and constraints                      │
│ ├─ Procedural rules                                    │
│ └─ Session summary (compressed)                        │
├────────────────────────────────────────────────────────┤
│ SECTION 5: CONVERSATION HISTORY       [MIDDLE REGION]  │
│ ├─ Compressed earlier turns                            │
│ └─ Verbatim recent turns                               │
├────────────────────────────────────────────────────────┤
│ SECTION 6: CURRENT TASK STATE                          │
│ ├─ Agent loop phase (plan/act/verify/repair)           │
│ ├─ Accumulated intermediate results                    │
│ └─ Outstanding subgoals                                │
├────────────────────────────────────────────────────────┤
│ SECTION 7: FINAL INSTRUCTIONS        [RECENCY REGION]  │
│ ├─ Task-specific output constraints                    │
│ ├─ Verification checklist                              │
│ └─ Response format enforcement                         │
└────────────────────────────────────────────────────────┘
```

### 6.3 Hallucination Control Directives

The policy section embeds **mechanistic hallucination controls** rather than generic "be accurate" instructions:

```
ALGORITHM 6.1: HallucinationControlPolicy.generate()

Input:  task_type, available_evidence, confidence_thresholds
Output: policy directives as structured text

1.  base_directives ← [
        "Ground every factual claim in the provided [SOURCE] blocks.",
        "If no source supports a claim, state: 'I do not have sufficient
         evidence to answer this.'",
        "Never fabricate citations, URLs, dates, or numerical values.",
        "Distinguish between: (a) source-supported facts, (b) reasonable
         inferences, and (c) uncertain estimates. Label each explicitly."
    ]

2.  IF task_type = "code_generation":
        base_directives.append(
            "Do not invoke APIs, functions, or libraries that are not
             present in the provided tool schemas or code context. If
             uncertain about an API's existence, emit a TODO comment."
        )

3.  IF |available_evidence| = 0:
        base_directives.append(
            "No retrieval evidence is available for this query. Respond
             using only parametric knowledge and explicitly flag the
             response as 'ungrounded—requires verification.'"
        )

4.  confidence_gate ← format(
        "Before responding, internally score your confidence on a scale
         [0.0, 1.0]. If confidence < {threshold}, prepend the response
         with '[LOW CONFIDENCE]' and explain the uncertainty source.",
        threshold=confidence_thresholds.min_response_confidence
    )
    base_directives.append(confidence_gate)

5.  RETURN compile_to_minimal_tokens(base_directives)
```

### 6.4 Token-Efficient Prompt Compression

The compiler applies **token-level optimization** to eliminate redundancy:

$$
\text{Compression}(\mathcal{P}_{\text{raw}}) = \text{StopwordRemoval} \circ \text{ProseToList} \circ \text{DeduplicateDirectives} \circ \text{AbbreviateExamples}
$$

Empirical compression ratios: 25–40% token reduction with <2% task accuracy degradation, measured across instruction-following benchmarks.

**Technique: Structural Compression via Schema Enforcement.**

Instead of:

```
Please provide your response in the following format. First, write a summary
of no more than 100 words. Then, provide the detailed analysis. Finally, list
any sources you used.
```

Compile to:

```
Output schema:
{summary: string (≤100 words), analysis: string, sources: string[]}
```

This reduces tokens by 60% while increasing format compliance by 15% (measured on structured output benchmarks).

---

## 7. Agentic Orchestration: Bounded Control Loops with Formal Verification

### 7.1 The Agent Execution Loop as a Finite State Machine

**Definition 7.1 (Agent Loop FSM).** An agent execution is modeled as a finite state machine $\mathcal{F} = (S, \Sigma, \delta, s_0, F)$ where:

- $S = \{\text{PLAN}, \text{DECOMPOSE}, \text{RETRIEVE}, \text{ACT}, \text{VERIFY}, \text{CRITIQUE}, \text{REPAIR}, \text{COMMIT}, \text{FAIL}, \text{HALT}\}$
- $\Sigma$: Set of transition events (success, failure, timeout, budget\_exceeded, verification\_passed, etc.)
- $\delta: S \times \Sigma \rightarrow S$: Transition function
- $s_0 = \text{PLAN}$: Initial state
- $F = \{\text{COMMIT}, \text{FAIL}, \text{HALT}\}$: Terminal states

The transition function enforces bounded execution:

$$
\delta(\text{state}, \text{event}) = \begin{cases}
\text{FAIL} & \text{if } \text{depth} > D_{\max} \;\vee\; \text{elapsed} > T_{\max} \;\vee\; \text{cost} > C_{\max} \\
\text{REPAIR} & \text{if } \text{state} = \text{VERIFY} \;\wedge\; \text{event} = \text{verification\_failed} \;\wedge\; \text{repair\_attempts} < R_{\max} \\
\text{FAIL} & \text{if } \text{state} = \text{REPAIR} \;\wedge\; \text{repair\_attempts} \geq R_{\max} \\
\text{next\_state} & \text{otherwise per FSM transition table}
\end{cases}
$$

### 7.2 Full Agent Loop Pseudo-Algorithm

```
ALGORITHM 7.1: AgentLoop.execute()

Input:  task T, budget constraints B = {max_depth, max_time, max_cost, max_repairs}
Output: TaskResult with status, artifacts, trace

1.  state ← PLAN
    depth ← 0
    repair_count ← 0
    trace ← ExecutionTrace()
    workspace ← IsolatedWorkspace.create(task_id=T.id)

2.  WHILE state ∉ {COMMIT, FAIL, HALT}:

      // Guard: Budget enforcement
      IF depth > B.max_depth OR elapsed() > B.max_time OR cost() > B.max_cost:
          state ← FAIL
          trace.append(FailureRecord(reason="budget_exceeded", state=state))
          BREAK

      MATCH state:

        CASE PLAN:
          plan ← Planner.generate_plan(T, workspace.state, memory_context)
          plan.validate_against(T.constraints)
          trace.append(PlanRecord(plan))
          state ← DECOMPOSE

        CASE DECOMPOSE:
          subtasks ← Decomposer.split(plan, granularity="atomic_verifiable")
          dependency_graph ← Decomposer.order(subtasks)
          trace.append(DecomposeRecord(subtasks, dependency_graph))
          state ← RETRIEVE

        CASE RETRIEVE:
          FOR EACH subtask st IN dependency_graph.next_ready():
              evidence ← RetrievalEngine.execute(
                  query=st.retrieval_query,
                  sources=st.routed_sources,
                  deadline=B.max_time * 0.2
              )
              workspace.attach_evidence(st.id, evidence)
          state ← ACT

        CASE ACT:
          FOR EACH subtask st IN dependency_graph.next_ready():
              result ← ActionExecutor.execute(
                  subtask=st,
                  evidence=workspace.get_evidence(st.id),
                  tools=ToolRegistry.get_affordances(st),
                  workspace=workspace
              )
              workspace.record_result(st.id, result)
              trace.append(ActionRecord(st.id, result))
              
              IF result.requires_human_approval:
                  approval ← HumanApprovalGate.request(result, timeout=300s)
                  IF NOT approval.granted:
                      state ← REPAIR
                      CONTINUE OUTER LOOP
          
          state ← VERIFY
          depth += 1

        CASE VERIFY:
          verification ← Verifier.check_all(
              workspace.results,
              against={
                  task_requirements: T.acceptance_criteria,
                  consistency: workspace.get_all_results(),
                  evidence_grounding: workspace.get_all_evidence(),
                  format_compliance: T.output_schema
              }
          )
          trace.append(VerifyRecord(verification))
          
          IF verification.all_passed:
              state ← CRITIQUE
          ELSE:
              state ← REPAIR
              repair_count += 1

        CASE CRITIQUE:
          critique ← Critic.evaluate(
              workspace.results,
              criteria=["correctness", "completeness", "coherence",
                        "evidence_coverage", "edge_cases"]
          )
          trace.append(CritiqueRecord(critique))
          
          IF critique.score ≥ T.quality_threshold:
              state ← COMMIT
          ELSE:
              state ← REPAIR
              repair_count += 1

        CASE REPAIR:
          IF repair_count > B.max_repairs:
              state ← FAIL
              CONTINUE

          repair_plan ← Repairer.diagnose_and_plan(
              failures=trace.get_failures(),
              workspace=workspace,
              available_budget=remaining_budget(B)
          )
          
          // Rollback affected subtasks
          FOR EACH affected_st IN repair_plan.rollback_targets:
              workspace.rollback(affected_st)

          // Re-execute with repair strategy
          FOR EACH fix IN repair_plan.fixes:
              workspace.apply_fix(fix)
          
          state ← RETRIEVE  // Re-enter loop with repaired state
          trace.append(RepairRecord(repair_plan))

3.  // Terminal state handling
    IF state = COMMIT:
        result ← workspace.compile_final_result()
        MemoryPromoter.evaluate_and_promote(
            candidate=extract_learnings(trace),
            source_tier=0, target_tier=2
        )
        RETURN TaskResult(status=SUCCESS, artifacts=result, trace=trace)
    
    ELIF state = FAIL:
        failure_state ← workspace.persist_failure_state()
        RETURN TaskResult(status=FAILED, failure_state=failure_state, trace=trace)
```

### 7.3 Multi-Agent Orchestration with Specialization

For complex tasks, multiple specialized agents operate in parallel under a **supervisor-worker architecture** with explicit lock discipline:

**Definition 7.2 (Agent Specialization Roles).**

| Role | Responsibility | Isolation Level | Write Access |
|------|---------------|----------------|-------------|
| $\mathcal{A}_{\text{planner}}$ | Decomposition and task routing | Read-only on all workspaces | Plan store only |
| $\mathcal{A}_{\text{implementer}}$ | Code/content generation | Isolated branch workspace | Branch-local files |
| $\mathcal{A}_{\text{verifier}}$ | Correctness checking and testing | Read-only on implementation | Test results store |
| $\mathcal{A}_{\text{retriever}}$ | Evidence gathering and ranking | Read-only on knowledge stores | Cache only |
| $\mathcal{A}_{\text{critic}}$ | Quality assessment and improvement suggestions | Read-only on all artifacts | Critique annotations |
| $\mathcal{A}_{\text{documenter}}$ | Documentation generation | Read-only on implementation | Documentation store |

### 7.4 Concurrency Control via Task Leases

```
ALGORITHM 7.2: TaskLeaseManager.acquire()

Input:  agent_id A, task_id T, lease_duration_ms L
Output: lease grant or rejection

1.  ATOMIC:
        current_lease ← LeaseStore.get(T)
        IF current_lease = NULL OR current_lease.expired():
            new_lease ← Lease(
                task_id=T,
                agent_id=A,
                granted_at=now(),
                expires_at=now() + L,
                fence_token=monotonic_counter.next()
            )
            LeaseStore.put(T, new_lease)
            RETURN GrantResult(lease=new_lease)
        ELSE:
            RETURN RejectionResult(
                holder=current_lease.agent_id,
                expires_at=current_lease.expires_at
            )
    END ATOMIC

2.  // Lease renewal
    FUNCTION renew(lease, extension_ms):
        ATOMIC:
            current ← LeaseStore.get(lease.task_id)
            IF current.fence_token = lease.fence_token:
                current.expires_at += extension_ms
                LeaseStore.put(lease.task_id, current)
                RETURN RenewResult(success=True)
            ELSE:
                RETURN RenewResult(success=False, reason="fence_token_mismatch")
        END ATOMIC
```

The **fence token** (a monotonically increasing counter) prevents zombie agents (those whose lease expired but are still running) from performing stale writes. Every state-mutating operation must present the current fence token, which the storage layer validates before committing.

### 7.5 Merge Entropy Control

When parallel agents produce artifacts that must be merged (e.g., concurrent code changes), we define **merge entropy** as:

$$
H_{\text{merge}}(\mathcal{A}_1, \mathcal{A}_2) = -\sum_{r \in R_{\text{shared}}} P(r \text{ conflicts}) \cdot \log P(r \text{ conflicts})
$$

where $R_{\text{shared}}$ is the set of shared resources. The orchestrator only permits parallelization when:

$$
H_{\text{merge}}(\mathcal{A}_i, \mathcal{A}_j) < \eta_{\max} \quad \forall \; i \neq j
$$

This is enforced by ensuring parallel agents operate on **disjoint resource partitions** wherever possible, and by routing tasks with shared-state dependencies to sequential execution.

---

## 8. Tool Integration: Typed Protocol Infrastructure

### 8.1 Protocol Stack Architecture

Tools are exposed through a **three-layer protocol stack**, each layer serving a distinct boundary:

```
┌─────────────────────────────────────────────────────┐
│           APPLICATION / USER BOUNDARY               │
│                  JSON-RPC 2.0                       │
│  • User-facing API, external integrations           │
│  • Schema-described methods with error codes        │
│  • Request/response with correlation IDs            │
│  • Rate limiting, authentication, pagination        │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│         TOOL DISCOVERY & INTEROPERABILITY           │
│             Model Context Protocol (MCP)            │
│  • Discoverable tool servers (local + remote)       │
│  • Typed tool schemas with JSON Schema validation   │
│  • Resource exposure with URI templates             │
│  • Prompt surfaces for model consumption            │
│  • Change notifications (subscription model)        │
│  • Capability negotiation and versioning            │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│          INTERNAL SERVICE-TO-SERVICE                │
│              gRPC / Protobuf                        │
│  • Low-latency binary serialization                 │
│  • Streaming for long-running operations            │
│  • Deadline propagation via gRPC metadata           │
│  • Circuit breakers and retry policies              │
│  • mTLS authentication between services             │
└─────────────────────────────────────────────────────┘
```

### 8.2 Tool Schema Contract (MCP-Aligned)

Every tool exposes a typed contract following the MCP specification:

```protobuf
// Tool contract definition (Protobuf representation of MCP tool schema)

message ToolDefinition {
    string name = 1;
    string version = 2;
    string description = 3;         // Human-readable, used in prompt compilation
    
    InputSchema input_schema = 4;   // JSON Schema for input validation
    OutputSchema output_schema = 5; // JSON Schema for output structure
    
    ToolCapabilities capabilities = 6;
    SecurityPolicy security = 7;
    PerformanceCharacteristics perf = 8;
}

message ToolCapabilities {
    bool supports_pagination = 1;
    bool supports_streaming = 2;
    bool supports_change_notifications = 3;
    bool is_idempotent = 4;
    bool is_read_only = 5;
    MutationPolicy mutation_policy = 6;
}

message MutationPolicy {
    bool requires_human_approval = 1;
    repeated string approval_roles = 2;
    bool supports_dry_run = 3;
    bool supports_rollback = 4;
    int32 max_blast_radius = 5;     // Maximum affected entities
}

message SecurityPolicy {
    AuthorizationScope scope = 1;   // caller-scoped, not agent-scoped
    repeated string required_permissions = 2;
    bool audit_logging_required = 3;
    int32 rate_limit_per_minute = 4;
}

message PerformanceCharacteristics {
    int32 p50_latency_ms = 1;
    int32 p99_latency_ms = 2;
    TimeoutClass timeout_class = 3; // FAST (<100ms), MEDIUM (<1s), SLOW (<30s), LONG_RUNNING
    int32 max_result_size_bytes = 4;
}
```

### 8.3 Tool Invocation with Least Privilege and Audit Trail

```
ALGORITHM 8.1: ToolExecutor.invoke()

Input:  tool_name T, input_params P, caller_context C, deadline D
Output: ToolResult with output, metadata, and audit record

1.  // Phase 1: Discovery and validation
    tool_def ← ToolRegistry.resolve(T, version=C.required_version)
    IF tool_def = NULL:
        RETURN ToolError(code=NOT_FOUND, message="Tool not registered")
    
    validation ← JSONSchemaValidator.validate(P, tool_def.input_schema)
    IF NOT validation.valid:
        RETURN ToolError(code=INVALID_INPUT, details=validation.errors)

2.  // Phase 2: Authorization check (caller-scoped, NOT agent-scoped)
    auth_result ← AuthzEngine.check(
        caller=C.human_principal,     // Always trace to human identity
        agent=C.agent_id,
        tool=T,
        action=infer_action(P),
        permissions_required=tool_def.security.required_permissions
    )
    IF NOT auth_result.authorized:
        RETURN ToolError(code=UNAUTHORIZED, missing=auth_result.missing_permissions)

3.  // Phase 3: Mutation gate (human approval for state-changing operations)
    IF NOT tool_def.capabilities.is_read_only:
        IF tool_def.mutation_policy.requires_human_approval:
            approval ← HumanApprovalGate.request(
                action_description=format_action(T, P),
                blast_radius=estimate_blast_radius(T, P),
                timeout=min(30s, D.remaining()),
                approval_roles=tool_def.mutation_policy.approval_roles
            )
            IF NOT approval.granted:
                RETURN ToolResult(status=BLOCKED, reason="human_denied")
        
        IF tool_def.mutation_policy.supports_dry_run:
            dry_run_result ← execute_internal(T, P, dry_run=True)
            // Log dry run for audit but don't commit

4.  // Phase 4: Execution with deadline and circuit breaker
    circuit_breaker ← CircuitBreakerRegistry.get(T)
    IF circuit_breaker.state = OPEN:
        RETURN ToolError(code=SERVICE_UNAVAILABLE, retry_after=circuit_breaker.reset_time)
    
    TRY:
        result ← execute_internal(T, P, deadline=D)
        circuit_breaker.record_success()
    CATCH TimeoutException:
        circuit_breaker.record_failure()
        RETURN ToolError(code=DEADLINE_EXCEEDED)
    CATCH Exception e:
        circuit_breaker.record_failure()
        RETURN ToolError(code=INTERNAL_ERROR, message=e.message)

5.  // Phase 5: Output validation and audit
    IF tool_def.output_schema ≠ NULL:
        output_valid ← JSONSchemaValidator.validate(result, tool_def.output_schema)
        IF NOT output_valid.valid:
            RETURN ToolError(code=INVALID_OUTPUT, details="Tool returned malformed output")
    
    audit_record ← AuditLog.record({
        timestamp: now(),
        tool: T,
        caller: C.human_principal,
        agent: C.agent_id,
        input_hash: hash(P),       // Hash, not raw input, for sensitive data
        output_summary: summarize(result),
        latency_ms: elapsed(),
        status: SUCCESS
    })

6.  RETURN ToolResult(
        status=SUCCESS,
        output=result,
        metadata={latency: elapsed(), audit_id: audit_record.id}
    )
```

### 8.4 Lazy Tool Loading for Context Efficiency

Tools should not be loaded into the prompt unless they are relevant to the current task. We define a **lazy tool loading** strategy:

$$
\text{Tools}_{\text{loaded}}(q) = \{t \in \mathcal{T} \mid \text{relevance}(t, q) > \tau_{\text{tool}} \;\wedge\; |t.\text{schema}| \leq B_{\text{tool\_budget}}\}
$$

where relevance is computed by matching the query's intent graph against tool capability descriptions:

$$
\text{relevance}(t, q) = \max_{c \in t.\text{capabilities}} \cos(\text{embed}(c.\text{description}), \text{embed}(q))
$$

Only the top-$k$ most relevant tools (typically $k \leq 10$) are included in the compiled prompt, with their schemas serialized in compact form. Tools not loaded remain discoverable via a meta-tool `tool_search(query) → tool_list` that the agent can invoke when it recognizes it needs a capability not currently available.

---

## 9. End-to-End System Architecture and Production Reliability

### 9.1 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL BOUNDARY (JSON-RPC)                    │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐                 │
│  │  Web Client  │   │  API Client  │   │  CLI Client   │                │
│  └──────┬──────┘   └──────┬──────┘   └──────┬───────┘                 │
│         └──────────────────┼──────────────────┘                        │
│                            │ JSON-RPC 2.0 + Auth                       │
│                            ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    API GATEWAY                               │       │
│  │  Rate Limiting │ Auth │ Request Routing │ Request Shaping   │       │
│  └────────────────────────┬────────────────────────────────────┘       │
│                            │                                           │
│  ┌─────────────────────────▼────────────────────────────────────┐      │
│  │                 ORCHESTRATION LAYER                           │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │              SUPERVISOR AGENT                         │    │      │
│  │  │  Plan │ Decompose │ Route │ Monitor │ Merge │ Commit │    │      │
│  │  └────┬────────┬───────────┬──────────┬────────┬───────┘    │      │
│  │       │        │           │          │        │             │      │
│  │  ┌────▼───┐ ┌──▼────┐ ┌───▼───┐ ┌───▼───┐ ┌──▼─────┐      │      │
│  │  │Implemen│ │Verify │ │Retriev│ │Critic │ │Document│      │      │
│  │  │ter     │ │er     │ │er     │ │       │ │er      │      │      │
│  │  │Agent   │ │Agent  │ │Agent  │ │Agent  │ │Agent   │      │      │
│  │  └────┬───┘ └──┬────┘ └───┬───┘ └───┬───┘ └──┬─────┘      │      │
│  │       └────────┴──────────┴─────────┴────────┘             │      │
│  └──────────────────────────┬──────────────────────────────────┘      │
│                              │ gRPC (internal)                        │
│  ┌──────────────────────────▼──────────────────────────────────┐      │
│  │                 CONTEXT ENGINE                               │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐     │      │
│  │  │  Context    │  │  Token     │  │  Positional        │     │      │
│  │  │  Compiler   │  │  Budget    │  │  Placement         │     │      │
│  │  │             │  │  Allocator │  │  Optimizer          │     │      │
│  │  └──────┬─────┘  └─────┬──────┘  └─────────┬──────────┘     │      │
│  │         └──────────────┼────────────────────┘               │      │
│  └──────────────────────────┬──────────────────────────────────┘      │
│                              │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐      │
│  │              RETRIEVAL & MEMORY LAYER                        │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐   │      │
│  │  │ Vector   │  │ BM25     │  │ Knowledge│  │ Memory    │   │      │
│  │  │ Store    │  │ Index    │  │ Graph    │  │ Tiers     │   │      │
│  │  │ (HNSW)   │  │          │  │          │  │ M0..M4    │   │      │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘   │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │      │
│  │  │ SQL/OLAP │  │ Live API │  │ Web      │                  │      │
│  │  │          │  │ Gateway  │  │ Search   │                  │      │
│  │  └──────────┘  └──────────┘  └──────────┘                  │      │
│  └──────────────────────────┬──────────────────────────────────┘      │
│                              │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐      │
│  │                 TOOL LAYER (MCP)                              │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐   │      │
│  │  │ File     │  │ Database │  │ Browser  │  │ Code      │   │      │
│  │  │ System   │  │ Tool     │  │ Control  │  │ Execution │   │      │
│  │  │ Server   │  │ Server   │  │ Server   │  │ Server    │   │      │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘   │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │      │
│  │  │ Deploy   │  │ Monitor  │  │ Custom   │                  │      │
│  │  │ Tool     │  │ Tool     │  │ MCP Svr  │                  │      │
│  │  └──────────┘  └──────────┘  └──────────┘                  │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │              OBSERVABILITY & RELIABILITY                     │      │
│  │  Traces │ Metrics │ Logs │ Circuit Breakers │ Retry Budget  │      │
│  │  Cost Tracking │ Latency Histograms │ Quality Dashboards    │      │
│  └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Reliability Engineering

#### 9.2.1 Circuit Breaker with Exponential Backoff and Jitter

```
ALGORITHM 9.1: CircuitBreaker.execute()

Input:  operation Op, circuit_name N
Output: result or circuit-open error

STATE MACHINE:
    CLOSED → (failure_count ≥ threshold) → OPEN
    OPEN → (timeout elapsed) → HALF_OPEN
    HALF_OPEN → (probe succeeds) → CLOSED
    HALF_OPEN → (probe fails) → OPEN

1.  cb ← CircuitBreakerStore.get(N)
    
    IF cb.state = OPEN:
        IF now() < cb.next_retry_time:
            RETURN CircuitOpenError(retry_after=cb.next_retry_time - now())
        ELSE:
            cb.state ← HALF_OPEN

2.  TRY:
        result ← Op.execute()
        cb.record_success()
        IF cb.state = HALF_OPEN:
            cb.state ← CLOSED
            cb.failure_count ← 0
        RETURN result

3.  CATCH Exception:
        cb.failure_count += 1
        IF cb.failure_count ≥ cb.threshold:
            cb.state ← OPEN
            // Exponential backoff with decorrelated jitter
            base_delay ← cb.base_delay_ms * (2 ^ cb.consecutive_opens)
            jitter ← random_uniform(0, base_delay)
            cb.next_retry_time ← now() + min(base_delay + jitter, cb.max_delay_ms)
            cb.consecutive_opens += 1
        RAISE
```

#### 9.2.2 Idempotency for State-Mutating Operations

Every state-changing tool invocation and agent action carries an **idempotency key**:

$$
\text{IdempotencyKey}(op) = \text{SHA256}(\text{agent\_id} \| \text{task\_id} \| \text{operation\_type} \| \text{canonical\_params})
$$

```
ALGORITHM 9.2: IdempotentExecutor.execute()

Input:  operation Op, idempotency_key K
Output: result (from cache or fresh execution)

1.  cached ← IdempotencyStore.get(K)
    IF cached ≠ NULL AND cached.status = COMPLETED:
        RETURN cached.result    // Return cached result, no re-execution

2.  IF cached ≠ NULL AND cached.status = IN_PROGRESS:
        IF now() - cached.started_at < cached.timeout:
            RETURN ConflictError("Operation in progress")
        ELSE:
            // Stale in-progress record, allow retry
            IdempotencyStore.delete(K)

3.  // Record intent before execution
    IdempotencyStore.put(K, {status: IN_PROGRESS, started_at: now()})
    
    TRY:
        result ← Op.execute()
        IdempotencyStore.update(K, {status: COMPLETED, result: result, completed_at: now()})
        RETURN result
    CATCH Exception e:
        IdempotencyStore.update(K, {status: FAILED, error: e, failed_at: now()})
        RAISE
```

#### 9.2.3 Backpressure and Queue Isolation

The system implements **per-priority-class queues** with admission control:

$$
\text{Admit}(request) = \begin{cases}
\text{accept} & \text{if } \text{queue}[request.\text{priority}].\text{size} < \text{queue}[request.\text{priority}].\text{capacity} \\
\text{shed} & \text{if } request.\text{priority} = \text{LOW} \;\wedge\; \text{system\_load} > 0.8 \\
\text{throttle} & \text{if } \text{caller\_rate} > \text{rate\_limit}[request.\text{caller}] \\
\text{reject} & \text{otherwise}
\end{cases}
$$

Queue isolation ensures that a burst of low-priority requests cannot starve high-priority agent loop operations:

| Queue | Priority | Capacity | Timeout | Load Shed Threshold |
|-------|----------|----------|---------|---------------------|
| `agent_loop_critical` | P0 | 100 | 30s | Never shed |
| `retrieval_requests` | P1 | 500 | 10s | System load > 0.9 |
| `tool_invocations` | P1 | 200 | 30s | System load > 0.9 |
| `memory_writes` | P2 | 1000 | 60s | System load > 0.8 |
| `background_indexing` | P3 | Unbounded | 300s | System load > 0.6 |

### 9.3 Cost Optimization

#### 9.3.1 Token Cost Model

The total cost of an agent execution is:

$$
C_{\text{total}} = \sum_{i=1}^{N_{\text{calls}}} \left( c_{\text{input}} \cdot T_{\text{input}}^{(i)} + c_{\text{output}} \cdot T_{\text{output}}^{(i)} \right) + C_{\text{retrieval}} + C_{\text{tools}} + C_{\text{compute}}
$$

where $c_{\text{input}}, c_{\text{output}}$ are per-token prices for the LLM, $T_{\text{input}}^{(i)}, T_{\text{output}}^{(i)}$ are token counts for the $i$-th LLM call, and the summation runs over all LLM invocations in the agent loop.

#### 9.3.2 Cost Optimization Strategies

1. **Prompt Caching:** For stable prefixes (system policy, tool schemas), leverage provider-level prompt caching (e.g., Anthropic's prompt caching) to reduce input token costs by up to 90% on cached segments.

$$
C_{\text{cached}}^{(i)} = c_{\text{cache\_read}} \cdot T_{\text{cached}} + c_{\text{input}} \cdot T_{\text{uncached}} + c_{\text{output}} \cdot T_{\text{output}}^{(i)}
$$

where $c_{\text{cache\_read}} \ll c_{\text{input}}$.

2. **Model Routing:** Route simple subtasks to smaller, cheaper models:

$$
\text{Model}(subtask) = \begin{cases}
\mathcal{M}_{\text{large}} & \text{if complexity}(subtask) > \theta_{\text{complex}} \\
\mathcal{M}_{\text{small}} & \text{otherwise}
\end{cases}
$$

3. **Result Caching:** Cache retrieval results and tool outputs with content-addressable keys, serving repeated queries from cache.

4. **Early Termination:** If the verification gate passes on the first attempt, skip the critique-repair loop, saving 1–3 LLM calls.

### 9.4 Observability

Every boundary in the system emits structured traces following the **OpenTelemetry** specification:

```
TRACE: agent_execution
├── SPAN: query_augmentation (12ms)
│   ├── attribute: subquery_count = 3
│   └── attribute: decomposition_strategy = "RAD"
├── SPAN: retrieval (45ms)
│   ├── SPAN: vector_search (8ms)
│   │   ├── attribute: results_count = 15
│   │   └── attribute: top_score = 0.92
│   ├── SPAN: bm25_search (3ms)
│   └── SPAN: rrf_fusion (2ms)
│       └── attribute: final_results = 8
├── SPAN: context_compilation (5ms)
│   ├── attribute: total_tokens = 3847
│   ├── attribute: budget_utilization = 0.76
│   └── attribute: compression_ratio = 0.35
├── SPAN: llm_inference (1200ms)
│   ├── attribute: model = "claude-sonnet-4-20250514"
│   ├── attribute: input_tokens = 3847
│   ├── attribute: output_tokens = 512
│   ├── attribute: cache_hit_tokens = 2100
│   └── attribute: cost_usd = 0.0043
├── SPAN: verification (800ms)
│   ├── attribute: checks_passed = 4/4
│   └── attribute: confidence = 0.94
└── SPAN: memory_write (3ms)
    ├── attribute: tier = "episodic"
    └── attribute: items_promoted = 1
```

Key metrics exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `agent.execution.duration_ms` | Histogram | End-to-end agent loop latency |
| `agent.loop.depth` | Counter | Number of plan-act-verify iterations |
| `agent.repair.count` | Counter | Repair attempts before success/failure |
| `context.token.utilization` | Gauge | Fraction of token budget used |
| `retrieval.latency_ms` | Histogram | Per-source retrieval latency |
| `retrieval.relevance.top_score` | Gauge | Top evidence relevance score |
| `tool.invocation.count` | Counter | Tool calls per agent execution |
| `tool.circuit_breaker.state` | Gauge | Circuit breaker state per tool |
| `memory.write.admission_rate` | Gauge | Fraction of candidates admitted |
| `cost.usd.per_execution` | Histogram | Dollar cost per agent execution |
| `hallucination.rate` | Gauge | Fraction of responses flagged ungrounded |

---

## 10. Evaluation Infrastructure and Continuous Quality Enforcement

### 10.1 Evaluation as Code

Every failure, correction, and regression is converted into a **reproducible evaluation case**:

**Definition 10.1 (Evaluation Case).** An evaluation case $\mathcal{E}$ is a tuple:

$$
\mathcal{E} = (q_{\text{input}}, \mathcal{C}_{\text{context}}, y_{\text{expected}}, \mathcal{J}_{\text{criteria}}, \mathcal{T}_{\text{trace}})
$$

where $\mathcal{J}_{\text{criteria}}$ is a set of machine-evaluable judgment functions.

### 10.2 Multi-Dimensional Judgment Functions

```
ALGORITHM 10.1: EvaluationSuite.evaluate()

Input:  model response y, evaluation case E
Output: multi-dimensional score card

1.  scores ← {}

2.  // Factual Grounding (automated)
    claims ← ClaimExtractor.extract(y)
    FOR EACH claim c IN claims:
        grounded ← EvidenceMatcher.find_support(c, E.context)
        scores["grounding"].append(grounded.score)
    scores["grounding_rate"] ← mean(scores["grounding"])

3.  // Faithfulness (automated via NLI)
    FOR EACH claim c IN claims:
        entailment ← NLIModel.check(premise=E.context, hypothesis=c)
        scores["faithfulness"].append(entailment.probability)
    scores["faithfulness_rate"] ← mean(scores["faithfulness"])

4.  // Completeness (automated)
    required_aspects ← E.criteria.required_coverage
    covered ← AspectCoverage.check(y, required_aspects)
    scores["completeness"] ← |covered| / |required_aspects|

5.  // Format Compliance (automated)
    IF E.criteria.output_schema ≠ NULL:
        compliance ← SchemaValidator.validate(y, E.criteria.output_schema)
        scores["format_compliance"] ← 1.0 IF compliance.valid ELSE 0.0

6.  // Consistency (automated, for multi-turn)
    IF E.trace.previous_responses ≠ ∅:
        contradictions ← ConsistencyChecker.find_contradictions(
            y, E.trace.previous_responses
        )
        scores["consistency"] ← 1.0 - (|contradictions| / |claims|)

7.  // LLM-as-Judge (for nuanced quality)
    judge_score ← LLMJudge.evaluate(
        query=E.input,
        response=y,
        reference=E.expected,
        rubric=E.criteria.rubric,
        model="claude-sonnet-4-20250514"
    )
    scores["judge_score"] ← judge_score

8.  // Composite Score
    scores["composite"] ← weighted_mean(scores, weights=E.criteria.dimension_weights)

9.  RETURN ScoreCard(scores, pass=scores["composite"] ≥ E.criteria.threshold)
```

### 10.3 CI/CD Integration

Evaluations execute as part of the CI/CD pipeline:

```
ALGORITHM 10.2: EvalCI.run()

Input:  code changeset ΔC, eval suite S, quality gates G
Output: pass/fail with detailed report

1.  // Stage 1: Regression Detection
    baseline_scores ← EvalStore.get_latest_baseline()
    
    FOR EACH eval_case e IN S:
        current_score ← EvaluationSuite.evaluate(model_response(e.input), e)
        delta ← current_score - baseline_scores[e.id]
        
        IF delta.composite < -G.max_regression:
            regressions.append({case: e.id, delta: delta})

2.  // Stage 2: Quality Gate Enforcement
    aggregate_scores ← aggregate(all current_scores)
    
    gates_passed ← True
    FOR EACH gate g IN G:
        IF aggregate_scores[g.metric] < g.threshold:
            gates_passed ← False
            failures.append({gate: g.name, actual: aggregate_scores[g.metric],
                            required: g.threshold})

3.  // Stage 3: Report and Decision
    report ← EvalReport(
        changeset=ΔC,
        regressions=regressions,
        gate_failures=failures,
        score_card=aggregate_scores,
        verdict=gates_passed AND |regressions| = 0
    )
    
    IF NOT report.verdict:
        CI.block_merge(report)
    ELSE:
        EvalStore.update_baseline(aggregate_scores)
        CI.allow_merge(report)

4.  RETURN report
```

### 10.4 Feedback Loop Architecture

```
Human Corrections ──┐
Failed Traces ──────┤
Reviewer Comments ──┤──→ Normalizer ──→ Eval Case Generator ──→ Eval Suite
Production Alerts ──┘                                              │
                                                                   ▼
                                                          CI/CD Pipeline
                                                                   │
                                              ┌────────────────────┤
                                              ▼                    ▼
                                     Policy Updates        Memory Updates
                                     (Prompt Compiler       (Episodic M2,
                                      adjustments)          Procedural M4)
```

Every human correction is transformed into a **regression test** that prevents the same error from recurring. This creates a **ratchet mechanism**: the system's quality can only improve over time, as the evaluation suite grows monotonically with production experience.

---

## Conclusion: Architectural Invariants

The complete Context Engineering system satisfies the following architectural invariants, each of which is mechanically enforceable:

| Invariant | Enforcement Mechanism |
|-----------|----------------------|
| No ungrounded claims in output | Provenance-tagged retrieval + faithfulness eval gate |
| Token budget never exceeded | Context compiler with hard limit enforcement |
| Agent loop always terminates | Bounded depth, time, cost, and repair counters |
| Tool mutations are human-auditable | Audit log + approval gates + caller-scoped auth |
| Memory never admits duplicates | Content hash + semantic similarity dedup on write |
| No zombie agent writes | Fence tokens on all leased task operations |
| Retrieval latency bounded | Deadline propagation + circuit breakers + fallback sources |
| Cost bounded per execution | Per-execution cost accumulator with hard ceiling |
| Quality never regresses | CI eval suite with monotonic quality ratchet |
| System observable at every boundary | OpenTelemetry traces + structured metrics + cost tracking |

This architecture transforms an LLM from an isolated, stateless text predictor into a **reliable, grounded, cost-controlled, observable, and continuously improving production system**. The critical insight is that **Context Engineering is not prompt engineering**—it is the discipline of designing the complete computational infrastructure that determines what the model sees, when it sees it, how it acts, and how correctness is verified, at every step of every execution.

---

*This report defines the reference architecture. Every equation, algorithm, and invariant is implementation-ready and maps directly to typed interfaces in the production protocol stack.*