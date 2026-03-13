# Query Augmentation in Context Engineering: A Principal-Level Technical Report

## Deterministic Query Transformation, Expansion, Decomposition, and Agentic Query Orchestration for Production-Grade Retrieval Pipelines

---

## 1. Problem Statement and Formal Framing

In any agentic AI system, the **query interface** constitutes the singular point of maximum information loss. The raw user utterance $q_{\text{raw}}$ undergoes an entropy-increasing transformation as it passes through retrieval, ranking, and generation stages. If intent is misidentified at ingress, no downstream component—regardless of sophistication—can recover the signal. This is the **query bottleneck theorem** in context engineering:

$$
\mathcal{L}_{\text{system}} \geq \mathcal{L}_{\text{query}} + \mathcal{L}_{\text{retrieval}} + \mathcal{L}_{\text{generation}}
$$

where $\mathcal{L}_{\text{query}}$ represents the information loss at the query stage. Crucially:

$$
\frac{\partial \mathcal{L}_{\text{system}}}{\partial \mathcal{L}_{\text{query}}} \gg \frac{\partial \mathcal{L}_{\text{system}}}{\partial \mathcal{L}_{\text{retrieval}}} \gg \frac{\partial \mathcal{L}_{\text{system}}}{\partial \mathcal{L}_{\text{generation}}}
$$

This establishes that **marginal improvements at the query stage yield disproportionately higher system-level accuracy gains** than equivalent effort applied at retrieval or generation. The formal objective of query augmentation is therefore:

$$
q^{*} = \arg\max_{q' \in \mathcal{Q}'} \; \mathbb{E}\left[\text{Recall}@k\left(\mathcal{R}(q'), \mathcal{D}_{\text{relevant}}\right)\right] \cdot \mathbb{E}\left[\text{Precision}@k\left(\mathcal{R}(q'), \mathcal{D}_{\text{relevant}}\right)\right]
$$

subject to:

$$
\text{latency}(q') \leq \tau_{\text{budget}}, \quad \text{tokens}(q') \leq T_{\text{max}}, \quad \text{drift}(q', q_{\text{raw}}) \leq \delta_{\text{max}}
$$

where $\mathcal{R}$ is the retrieval function, $\mathcal{D}_{\text{relevant}}$ is the ground-truth relevant document set, $\tau_{\text{budget}}$ is the latency SLA, $T_{\text{max}}$ is the token budget for augmented queries, and $\delta_{\text{max}}$ is the maximum permissible semantic drift from the user's original intent.

---

## 2. Dual-Pipeline Incompatibility Problem

A critical architectural constraint that necessitates query augmentation is the **dual-pipeline incompatibility**: a query optimized for one downstream consumer is suboptimal—often catastrophically so—for another.

### 2.1 Formal Characterization

Let $f_{\text{vec}}: \mathcal{Q} \to \mathbb{R}^d$ be the embedding function for vector retrieval, and let $f_{\text{LLM}}: \mathcal{Q} \to \mathcal{A}$ be the LLM generation function. The optimal query forms diverge:

$$
q^{*}_{\text{retrieval}} = \arg\max_{q'} \; \text{sim}\left(f_{\text{vec}}(q'), f_{\text{vec}}(d^{*})\right), \quad d^{*} \in \mathcal{D}_{\text{relevant}}
$$

$$
q^{*}_{\text{generation}} = \arg\max_{q'} \; P_{\text{LLM}}\left(a^{*} \mid q', \mathcal{C}\right), \quad a^{*} \in \mathcal{A}_{\text{correct}}
$$

These two objectives produce **structurally incompatible** query forms:

| Property | Retrieval-Optimal Query | Generation-Optimal Query |
|---|---|---|
| **Lexical Density** | High keyword density, technical terminology | Natural language with contextual framing |
| **Disambiguation** | Explicit entity references, schema-aligned filters | Conversational co-reference resolution |
| **Structure** | Short, focused, mono-topical | Extended, multi-faceted, with constraints |
| **Token Profile** | Compact embedding-friendly surface | Rich instruction-following preamble |

This incompatibility mandates that query augmentation produce **multiple typed query representations** routed to appropriate consumers, not a single rewritten string.

### 2.2 Typed Query Envelope

Define the augmented query as a typed envelope:

```protobuf
message AugmentedQueryEnvelope {
  string request_id = 1;
  string raw_query = 2;
  QueryIntent intent = 3;
  repeated RetrievalQuery retrieval_queries = 4;
  GenerationQuery generation_query = 5;
  repeated FilterPredicate metadata_filters = 6;
  repeated string target_collections = 7;
  float drift_score = 8;
  int64 deadline_ms = 9;
  QueryProvenance provenance = 10;
}

message RetrievalQuery {
  string query_text = 1;
  QueryType type = 2;  // SEMANTIC, KEYWORD, HYBRID, GRAPH
  float weight = 3;
  repeated string target_fields = 4;
  int32 top_k = 5;
}

message GenerationQuery {
  string rewritten_query = 1;
  repeated string decomposed_subqueries = 2;
  string synthesized_intent = 3;
}

message QueryProvenance {
  string transformation_method = 1;
  string model_id = 2;
  float confidence = 3;
  int64 latency_ms = 4;
}
```

This protocol ensures every downstream consumer receives a query artifact in its optimal representation, with provenance tracking for observability and debugging.

---

## 3. Query Rewriting: Intent-Preserving Transformation

### 3.1 Formal Definition

Query rewriting is a **surjective mapping** $\phi: \mathcal{Q}_{\text{raw}} \to \mathcal{Q}_{\text{canonical}}$ that transforms an ill-formed, ambiguous, or suboptimally phrased user query into a canonical form that maximizes downstream retrieval precision while preserving semantic intent:

$$
\phi(q_{\text{raw}}) = q_{\text{rewritten}} \quad \text{s.t.} \quad \text{intent}(q_{\text{rewritten}}) \equiv \text{intent}(q_{\text{raw}}) \;\wedge\; \text{Recall}(\mathcal{R}(q_{\text{rewritten}})) \geq \text{Recall}(\mathcal{R}(q_{\text{raw}}))
$$

### 3.2 SOTA Rewriting Architecture: Step-Back Prompting with Chain-of-Density Compression

The state-of-the-art approach combines **Step-Back Prompting** (Zheng et al., 2023) with **Chain-of-Density** compression to produce maximally informative, retrieval-optimized rewrites.

**Step-Back Abstraction.** Instead of rewriting at the surface level, the model first generates a more abstract "step-back" question that captures the underlying principle, then produces a concrete retrieval query:

$$
q_{\text{abstract}} = \text{LLM}_{\text{step-back}}(q_{\text{raw}}), \quad q_{\text{rewritten}} = \text{LLM}_{\text{concretize}}(q_{\text{abstract}}, q_{\text{raw}})
$$

**Chain-of-Density Compression.** The rewritten query is iteratively densified: each pass increases entity density while maintaining fixed token length, ensuring maximum information per token:

$$
q^{(t+1)} = \text{Densify}\left(q^{(t)}\right) \quad \text{s.t.} \quad |q^{(t+1)}|_{\text{tokens}} \leq |q^{(t)}|_{\text{tokens}} \;\wedge\; \text{entities}(q^{(t+1)}) \geq \text{entities}(q^{(t)})
$$

### 3.3 Pseudo-Algorithm: Production Query Rewriter

```
ALGORITHM: SOTA_QUERY_REWRITE
──────────────────────────────────────────────────────────────

Input:
  q_raw         : string        — raw user utterance
  history       : ConvTurn[]    — session conversation history (last N turns)
  schema        : CollectionSchema — target data schema for filter extraction
  τ_max         : int           — latency budget in milliseconds
  T_budget      : int           — token budget for rewritten query

Output:
  envelope      : AugmentedQueryEnvelope

Procedure:

  1. CO-REFERENCE RESOLUTION
     ─────────────────────────
     q_resolved ← RESOLVE_COREFERENCES(q_raw, history)
     // Replace pronouns, ellipsis, implicit references
     // with explicit entity mentions from conversation history.
     //
     // Method: Sliding-window attention over last K turns.
     // For each anaphoric expression in q_raw, find the
     // most recent antecedent in history using entity salience scoring:
     //
     //   antecedent* = argmax_{e ∈ entities(history)}
     //                   [ salience(e) · recency_decay(e) · type_match(e, anaphor) ]
     //
     //   salience(e) = tf(e, history) · idf(e, corpus) · positional_weight(e)
     //   recency_decay(e) = exp(-λ · (current_turn - last_mention_turn(e)))

  2. INTENT CLASSIFICATION
     ──────────────────────
     intent ← CLASSIFY_INTENT(q_resolved)
     // Typed intent taxonomy:
     //   FACTUAL_LOOKUP | PROCEDURAL | COMPARATIVE | DIAGNOSTIC |
     //   AGGREGATION | EXPLORATORY | TRANSACTIONAL | CLARIFICATION
     //
     // Use a fine-tuned classifier or structured LLM extraction:
     //   intent = LLM(
     //     system="Classify the query intent into exactly one of: {taxonomy}",
     //     user=q_resolved,
     //     response_format={"type": "json", "schema": IntentSchema}
     //   )
     //
     // Confidence gating: if P(intent) < θ_intent (e.g., 0.7),
     // set intent = AMBIGUOUS and flag for clarification request.

  3. STEP-BACK ABSTRACTION
     ──────────────────────
     q_abstract ← STEP_BACK(q_resolved, intent)
     // Generate a higher-level conceptual question that captures
     // the underlying principle or domain area:
     //
     //   q_abstract = LLM(
     //     system="Given the user's specific question, generate a more
     //             general 'step-back' question that identifies the
     //             underlying concept, principle, or domain area.",
     //     user=q_resolved
     //   )
     //
     // Example:
     //   q_resolved:  "Why does my API call return 401 after token refresh?"
     //   q_abstract:  "How does OAuth2 token refresh interact with
     //                 authentication header validation in REST APIs?"

  4. ENTITY AND FILTER EXTRACTION
     ────────────────────────────
     entities ← EXTRACT_ENTITIES(q_resolved, schema)
     filters  ← DERIVE_FILTERS(entities, schema)
     // Extract named entities, technical terms, version numbers,
     // date ranges, and schema-aligned filter predicates.
     //
     // For each extracted entity e:
     //   IF e matches schema.field[f].type AND schema.field[f].filterable:
     //     filters.append(FilterPredicate(field=f, op=EQ|RANGE|IN, value=e))
     //
     // Example:
     //   q_resolved: "errors in payment service after v2.3 deployment last week"
     //   entities:   {service: "payment", version: "2.3", time: "last_week"}
     //   filters:    [{field: "service", op: EQ, val: "payment"},
     //                {field: "version", op: GTE, val: "2.3"},
     //                {field: "timestamp", op: GTE, val: NOW - 7d}]

  5. KEYWORD DENSIFICATION (Chain-of-Density)
     ─────────────────────────────────────────
     q_dense ← q_resolved
     FOR t = 1 TO N_density_passes (typically 2-3):
       q_dense ← DENSIFY(q_dense, entities, intent)
       // Each pass:
       //   - Identify missing salient entities from the domain
       //   - Replace vague terms with precise technical vocabulary
       //   - Inject synonyms and acronym expansions
       //   - Maintain or reduce token count
       //
       // Densification score:
       //   D(q) = |unique_entities(q)| / |tokens(q)|
       //
       // Terminate when D(q^(t+1)) - D(q^(t)) < ε_density

  6. MULTI-REPRESENTATION SYNTHESIS
     ──────────────────────────────
     // Produce distinct query forms for each retrieval modality:

     q_semantic  ← SYNTHESIZE_SEMANTIC(q_abstract, q_dense)
     // Natural-language query optimized for dense embedding similarity.
     // Emphasize conceptual coverage over keyword precision.

     q_keyword   ← EXTRACT_KEYWORDS(q_dense, top_k=10)
     // BM25-optimized keyword set with TF-IDF weighting.
     // Apply domain-specific stop-word removal and stemming.

     q_graph     ← CONSTRUCT_GRAPH_QUERY(entities, schema)
     // If graph/knowledge-base retrieval is available:
     // Generate SPARQL, Cypher, or structured traversal query
     // targeting entity relationships.

     q_generation ← SYNTHESIZE_GENERATION(q_resolved, q_abstract, intent)
     // Full natural-language question optimized for LLM comprehension.
     // Include explicit constraints, expected answer format,
     // and disambiguation context.

  7. DRIFT VALIDATION
     ─────────────────
     drift ← COMPUTE_DRIFT(q_raw, q_semantic, q_generation)
     // Drift metric: cosine distance in embedding space plus
     // entity overlap penalty:
     //
     //   drift(q, q') = α · (1 - cos(embed(q), embed(q')))
     //                + β · (1 - |entities(q) ∩ entities(q')| / |entities(q)|)
     //
     //   where α + β = 1, typically α=0.6, β=0.4
     //
     // IF drift > δ_max:
     //   FALL_BACK to q_resolved (minimally processed) for affected representations
     //   LOG drift_violation event with full provenance

  8. ENVELOPE ASSEMBLY
     ──────────────────
     envelope ← AugmentedQueryEnvelope(
       request_id   = generate_uuid(),
       raw_query    = q_raw,
       intent       = intent,
       retrieval_queries = [
         RetrievalQuery(q_semantic, SEMANTIC, weight=0.6, top_k=20),
         RetrievalQuery(q_keyword,  KEYWORD,  weight=0.3, top_k=30),
         RetrievalQuery(q_graph,    GRAPH,    weight=0.1, top_k=10)
       ],
       generation_query = GenerationQuery(q_generation, [], q_abstract),
       metadata_filters = filters,
       target_collections = ROUTE_COLLECTIONS(intent, entities, schema),
       drift_score  = drift,
       deadline_ms  = τ_max - elapsed_ms(),
       provenance   = QueryProvenance("SOTA_REWRITE_v2", model_id, confidence, elapsed)
     )

  RETURN envelope
```

### 3.4 Drift Control: Formal Guarantee

The drift constraint is non-negotiable in production. Define the **Intent Preservation Index (IPI)**:

$$
\text{IPI}(q_{\text{raw}}, q') = \frac{\sum_{e \in \mathcal{E}(q_{\text{raw}})} \mathbb{1}[e \in \mathcal{E}(q')]}{\left|\mathcal{E}(q_{\text{raw}})\right|} \cdot \cos\left(\mathbf{v}(q_{\text{raw}}), \mathbf{v}(q')\right)
$$

where $\mathcal{E}(\cdot)$ extracts salient entities and $\mathbf{v}(\cdot)$ is the dense embedding. A rewritten query is accepted only if:

$$
\text{IPI}(q_{\text{raw}}, q') \geq \theta_{\text{IPI}} \quad (\text{typically } \theta_{\text{IPI}} = 0.85)
$$

If this threshold is violated, the system falls back to the co-reference-resolved form $q_{\text{resolved}}$ and emits a structured observability event.

---

## 4. Query Expansion: Controlled Multi-Query Generation

### 4.1 Formal Definition

Query expansion generates a set of $n$ related queries $\{q_1', q_2', \ldots, q_n'\}$ from a single input $q$ to increase recall coverage across the retrieval corpus:

$$
\mathcal{Q}_{\text{expanded}} = \text{Expand}(q) = \{q_i' \mid i \in [1, n], \; \text{sim}_{\text{intent}}(q_i', q) \geq \theta_{\text{sim}}\}
$$

The aggregated retrieval result is:

$$
\mathcal{D}_{\text{expanded}} = \bigcup_{i=1}^{n} \mathcal{R}(q_i', k_i) \quad \text{with deduplication and re-ranking}
$$

### 4.2 SOTA Expansion: Hypothetical Document Embeddings (HyDE) + Reciprocal Rank Fusion

**HyDE (Gao et al., 2022).** Instead of embedding the query directly, generate a hypothetical answer document $\hat{d}$, then use $\hat{d}$'s embedding for retrieval:

$$
\hat{d} = \text{LLM}(q), \quad \mathbf{v}_{\text{retrieval}} = f_{\text{vec}}(\hat{d})
$$

This bridges the **query-document embedding gap**: the hypothetical document occupies the same embedding subspace as real documents, whereas queries occupy a different distributional region.

**Multi-Perspective Expansion.** Generate expansions along orthogonal semantic axes:

$$
q_i' = \text{LLM}\left(q, \text{perspective}_i\right), \quad \text{perspective}_i \in \{\text{synonym}, \text{hypernym}, \text{related\_concept}, \text{contrastive}, \text{procedural}\}
$$

**Reciprocal Rank Fusion (RRF).** Aggregate results from all expanded queries using RRF to avoid any single expansion dominating:

$$
\text{RRF\_score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}
$$

where $k$ is a smoothing constant (typically $k = 60$), and $\text{rank}_i(d)$ is the rank of document $d$ in the result set of query $q_i'$.

### 4.3 Pseudo-Algorithm: Production Query Expansion

```
ALGORITHM: CONTROLLED_QUERY_EXPANSION
──────────────────────────────────────────────────────────────

Input:
  q_rewritten   : string           — rewritten query from Stage 3
  intent        : QueryIntent      — classified intent
  n_max         : int              — maximum expansion count (default: 5)
  τ_budget      : int              — remaining latency budget (ms)
  retrieval_mode: enum             — {SEMANTIC, KEYWORD, HYBRID}

Output:
  expanded_set  : RetrievalQuery[] — weighted set of retrieval queries
  rrf_config    : RRFConfig        — fusion parameters

Procedure:

  1. EXPANSION STRATEGY SELECTION
     ────────────────────────────
     strategy ← SELECT_STRATEGY(intent, retrieval_mode)
     //
     // Strategy matrix (intent × retrieval_mode → expansion_type[]):
     //
     // ┌──────────────────┬──────────────────────────────────────────┐
     // │ Intent           │ Expansion Types                          │
     // ├──────────────────┼──────────────────────────────────────────┤
     // │ FACTUAL_LOOKUP   │ [SYNONYM, ACRONYM_EXPAND, HyDE]         │
     // │ PROCEDURAL       │ [STEP_ENUMERATE, HyDE, RELATED_CONCEPT] │
     // │ DIAGNOSTIC       │ [SYMPTOM_EXPAND, CAUSAL, CONTRASTIVE]   │
     // │ COMPARATIVE      │ [ENTITY_PAIR, ATTRIBUTE_AXIS, HyDE]     │
     // │ AGGREGATION      │ [FACET_ENUMERATE, TEMPORAL_WINDOW]      │
     // │ EXPLORATORY      │ [HYPERNYM, RELATED_CONCEPT, MULTI_ANGLE]│
     // └──────────────────┴──────────────────────────────────────────┘

  2. HYPOTHETICAL DOCUMENT GENERATION (HyDE)
     ────────────────────────────────────────
     IF HyDE ∈ strategy:
       d_hyp ← LLM(
         system="Given the following question, write a short paragraph that
                 would be found in a document that perfectly answers it.
                 Do not answer the question—write the SOURCE document passage.",
         user=q_rewritten,
         max_tokens=150,
         temperature=0.3
       )
       q_hyde ← RetrievalQuery(
         query_text=d_hyp, type=SEMANTIC, weight=0.4, top_k=15
       )

  3. MULTI-PERSPECTIVE EXPANSION
     ───────────────────────────
     perspectives ← strategy \ {HyDE}
     expanded_queries ← []

     FOR EACH perspective_type IN perspectives:
       q_exp ← LLM(
         system="Rewrite the following query from the perspective of
                 {perspective_type}. Produce a single, focused query
                 that would retrieve documents relevant to this aspect.
                 Output ONLY the rewritten query, nothing else.",
         user=q_rewritten,
         max_tokens=60,
         temperature=0.5
       )

       // Compute intent-preservation score:
       sim ← cosine(embed(q_rewritten), embed(q_exp))

       IF sim ≥ θ_expansion (typically 0.65):
         expanded_queries.append(
           RetrievalQuery(q_exp, SEMANTIC, weight=sim * 0.3, top_k=10)
         )

     // BUDGET ENFORCEMENT: if |expanded_queries| > n_max - 1:
     //   Sort by weight descending, truncate to n_max - 1.

  4. DIVERSITY-WEIGHTED DEDUPLICATION
     ────────────────────────────────
     // Remove near-duplicate expansions via MMR-style filtering:
     //
     //   MMR(q_i) = λ · sim(q_i, q_rewritten) - (1-λ) · max_{q_j ∈ S} sim(q_i, q_j)
     //
     //   where S is the already-selected set, λ = 0.7
     //
     // Greedily select queries maximizing MMR until |S| = n_max.

     expanded_set ← MMR_SELECT(
       candidates = [q_hyde] + expanded_queries + [original_as_retrieval(q_rewritten)],
       λ = 0.7,
       n = n_max
     )

  5. WEIGHT NORMALIZATION
     ────────────────────
     // Normalize weights to sum to 1.0 for RRF scoring:
     total_w ← SUM(q.weight for q in expanded_set)
     FOR EACH q IN expanded_set:
       q.weight ← q.weight / total_w

  6. RRF CONFIGURATION
     ──────────────────
     rrf_config ← RRFConfig(
       k = 60,
       score_function = "reciprocal_rank",
       dedup_field = "document_id",
       min_appearances = 1,     // document must appear in ≥1 result set
       max_results = top_k_final
     )

  RETURN expanded_set, rrf_config
```

### 4.4 Controlling Query Drift in Expansion

The three pathological modes of expansion failure—**drift**, **over-expansion**, and **latency blowup**—are controlled mechanically:

**Drift Control.** Each expanded query $q_i'$ is validated against the original via the pairwise similarity gate:

$$
q_i' \in \mathcal{Q}_{\text{accepted}} \iff \cos\left(\mathbf{v}(q_i'), \mathbf{v}(q_{\text{raw}})\right) \geq \theta_{\text{expansion}}
$$

**Over-Expansion Control.** Apply Maximal Marginal Relevance (MMR) selection to enforce diversity within the expansion set while bounding cardinality:

$$
\text{MMR}(q_i) = \lambda \cdot \text{sim}(q_i, q_{\text{raw}}) - (1 - \lambda) \cdot \max_{q_j \in \mathcal{S}} \text{sim}(q_i, q_j)
$$

where $\mathcal{S}$ is the already-selected expansion set and $\lambda \in [0.6, 0.8]$.

**Latency Control.** Execute expanded queries in parallel with a fan-out ceiling $n_{\text{max}}$ and a shared deadline. The RRF aggregation step has $O(n \cdot k \log k)$ complexity, which is negligible compared to retrieval latency:

$$
\text{latency}_{\text{expansion}} = \max_{i \in [1, n]} \text{latency}(\mathcal{R}(q_i')) + \text{latency}_{\text{RRF}} \approx \text{latency}_{\text{single\_retrieval}} + O(1)
$$

provided queries execute in parallel with proper fan-out and deadline propagation.

---

## 5. Query Decomposition: Divide-Retrieve-Synthesize

### 5.1 Formal Definition

For a complex multi-faceted query $q$ that requires information from $m$ distinct knowledge domains or document clusters, decomposition produces:

$$
\text{Decompose}(q) = \{s_1, s_2, \ldots, s_m\} \quad \text{s.t.} \quad \bigcup_{j=1}^{m} \text{scope}(s_j) \supseteq \text{scope}(q) \;\wedge\; \forall j \neq k: |\text{scope}(s_j) \cap \text{scope}(s_k)| \leq \epsilon
$$

where each sub-query $s_j$ targets a single information facet, the union of sub-query scopes covers the original query, and overlap between sub-queries is minimized.

### 5.2 SOTA Decomposition: Least-to-Most with Dependency DAG

The state-of-the-art decomposition approach constructs a **Directed Acyclic Graph (DAG)** of sub-queries with explicit dependency edges, enabling:

1. **Parallelization** of independent sub-queries
2. **Sequential chaining** where one sub-query's result informs another
3. **Early termination** if a critical sub-query yields no results

**Dependency DAG Construction.** For a decomposed set $\{s_1, \ldots, s_m\}$, define the dependency relation:

$$
s_i \to s_j \iff \text{answer}(s_i) \text{ is required to formulate or interpret } s_j
$$

The execution schedule is the topological sort of this DAG, with independent nodes executed in parallel:

$$
\text{Schedule} = \text{TopoSort}\left(\mathcal{G} = (\{s_1, \ldots, s_m\}, \{(s_i, s_j) \mid s_i \to s_j\})\right)
$$

### 5.3 Pseudo-Algorithm: Production Query Decomposition

```
ALGORITHM: DAG_QUERY_DECOMPOSITION
──────────────────────────────────────────────────────────────

Input:
  q_rewritten   : string          — rewritten query
  intent        : QueryIntent     — classified intent
  schema        : CollectionSchema
  max_depth     : int             — maximum decomposition depth (default: 3)
  max_subqueries: int             — maximum sub-query count (default: 6)

Output:
  dag           : QueryDAG        — dependency graph of sub-queries
  schedule      : ExecutionSchedule — topologically sorted execution plan

Procedure:

  1. COMPLEXITY ASSESSMENT
     ─────────────────────
     complexity ← ASSESS_COMPLEXITY(q_rewritten)
     //
     // Complexity scoring function:
     //   C(q) = w_1 · |entities(q)|
     //        + w_2 · |relation_types(q)|
     //        + w_3 · |temporal_references(q)|
     //        + w_4 · |comparison_axes(q)|
     //        + w_5 · conditional_depth(q)
     //
     //   where w = [0.25, 0.25, 0.15, 0.2, 0.15]
     //
     // IF C(q) < θ_decompose (e.g., 2.0):
     //   SKIP decomposition; return single-node DAG.
     //   Rationale: simple queries lose precision through decomposition.

     IF complexity < θ_decompose:
       dag ← SingleNodeDAG(q_rewritten)
       schedule ← [q_rewritten]
       RETURN dag, schedule

  2. FACET EXTRACTION
     ─────────────────
     facets ← EXTRACT_FACETS(q_rewritten, intent)
     //
     // Use structured extraction to identify distinct information needs:
     //
     //   facets = LLM(
     //     system="Analyze this complex question and identify the distinct
     //             information facets that need to be answered independently.
     //             For each facet, provide:
     //             - facet_id: unique identifier
     //             - sub_question: focused question for this facet
     //             - required_info_type: {factual, procedural, comparative, temporal}
     //             - depends_on: list of facet_ids whose answers are needed first
     //             - target_collection_hint: which data source likely contains this",
     //     user=q_rewritten,
     //     response_format={"type": "json", "schema": FacetListSchema}
     //   )
     //
     // CARDINALITY ENFORCEMENT:
     //   IF |facets| > max_subqueries:
     //     Merge the two most similar facets (by embedding distance) iteratively
     //     until |facets| ≤ max_subqueries.

  3. DEPENDENCY DAG CONSTRUCTION
     ───────────────────────────
     dag ← DirectedAcyclicGraph()
     FOR EACH facet IN facets:
       node ← SubQueryNode(
         id = facet.facet_id,
         query = facet.sub_question,
         info_type = facet.required_info_type,
         collection_hint = facet.target_collection_hint,
         status = PENDING
       )
       dag.add_node(node)

     FOR EACH facet IN facets:
       FOR EACH dep_id IN facet.depends_on:
         dag.add_edge(dep_id → facet.facet_id)

     // CYCLE DETECTION AND REPAIR:
     IF dag.has_cycle():
       // Break cycles by removing the edge with lowest confidence score:
       back_edges ← dag.find_back_edges()
       FOR EACH edge IN back_edges:
         dag.remove_edge(edge)
         IF NOT dag.has_cycle():
           BREAK

     // DEPTH ENFORCEMENT:
     IF dag.longest_path() > max_depth:
       // Flatten: promote deep nodes to have dependencies only on root nodes
       dag.flatten_to_depth(max_depth)

  4. EXECUTION SCHEDULE GENERATION
     ─────────────────────────────
     schedule ← TopologicalSort(dag)
     //
     // Group into execution waves (nodes at same depth execute in parallel):
     //
     //   Wave 0: all root nodes (no dependencies)
     //   Wave 1: nodes depending only on Wave 0 nodes
     //   ...
     //   Wave d: nodes depending on Wave d-1 or earlier
     //
     // Each wave has a timeout = τ_budget / (max_depth + 1)

     waves ← []
     FOR depth = 0 TO dag.max_depth():
       wave ← dag.nodes_at_depth(depth)
       waves.append(ExecutionWave(
         nodes = wave,
         timeout_ms = τ_budget / (dag.max_depth() + 1),
         parallel = TRUE
       ))

  5. SUB-QUERY REFINEMENT WITH CHAINED CONTEXT
     ──────────────────────────────────────────
     // After Wave i completes, inject retrieved results into
     // dependent sub-queries in Wave i+1:
     //
     // FOR EACH node IN wave[i+1]:
     //   dep_results ← GATHER(node.dependencies, results_cache)
     //   node.query ← REFINE_WITH_CONTEXT(node.query, dep_results)
     //
     // This enables progressive refinement:
     //   "What is the latency of service X?" (Wave 0)
     //   → result: "p99 latency is 450ms"
     //   "What caused the latency spike in service X beyond 450ms
     //    last Tuesday?" (Wave 1, refined with Wave 0 result)

  6. SYNTHESIS STRATEGY ASSIGNMENT
     ─────────────────────────────
     dag.synthesis_strategy ← SELECT_SYNTHESIS(intent, |facets|)
     //
     // Synthesis strategies:
     //   CONCATENATIVE:     Join all sub-results with section headers
     //   COMPARATIVE_TABLE: Format as comparison matrix
     //   NARRATIVE_MERGE:   LLM synthesizes a coherent narrative
     //   HIERARCHICAL:      Organize by DAG structure with nesting
     //
     // The synthesis strategy is passed to the generation phase
     // as part of the GenerationQuery in the AugmentedQueryEnvelope.

  RETURN dag, schedule
```

### 5.4 Synthesis Aggregation: Formal Fusion

After all sub-queries complete, results must be fused into a coherent response. Define the synthesis function:

$$
\text{Synthesize}(\{(s_j, \mathcal{D}_j)\}_{j=1}^m, q_{\text{original}}) = \text{LLM}\left(q_{\text{original}}, \bigoplus_{j=1}^{m} \text{Compress}(s_j, \mathcal{D}_j)\right)
$$

where $\bigoplus$ is the structured concatenation operator and $\text{Compress}$ extracts the minimal sufficient evidence from each sub-query's retrieval results:

$$
\text{Compress}(s_j, \mathcal{D}_j) = \arg\min_{c} |c|_{\text{tokens}} \quad \text{s.t.} \quad \text{NLI}(c \models s_j) \geq \theta_{\text{entailment}}
$$

This ensures each sub-result is compressed to its minimal entailing passage before injection into the synthesis context window.

---

## 6. Query Agents: Autonomous Query Orchestration

### 6.1 Formal Definition

A **Query Agent** is a bounded control loop $\mathcal{A}_q$ that autonomously selects and composes query augmentation strategies—rewriting, expansion, decomposition, filter construction, collection routing, and result evaluation—based on runtime analysis of the query, available data schema, and intermediate retrieval results:

$$
\mathcal{A}_q: (q_{\text{raw}}, \mathcal{S}, \mathcal{H}, \mathcal{C}_{\text{schema}}) \to \text{AugmentedQueryEnvelope}
$$

where $\mathcal{S}$ is the available tool/collection set, $\mathcal{H}$ is conversation history, and $\mathcal{C}_{\text{schema}}$ is the collection schema registry.

### 6.2 Agent Architecture: ReAct with Typed Tool Protocol

The query agent implements a **ReAct (Reason + Act)** loop augmented with typed tool contracts over MCP:

```
┌─────────────────────────────────────────────────────────┐
│                    QUERY AGENT LOOP                      │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ ANALYZE  │──▶│  PLAN    │──▶│  ROUTE   │             │
│  │ (intent, │   │ (select  │   │ (choose  │             │
│  │  schema, │   │  strategy│   │  collect-│             │
│  │  complex)│   │  stack)  │   │  ions)   │             │
│  └──────────┘   └──────────┘   └────┬─────┘             │
│                                      │                   │
│              ┌───────────────────────▼──────────┐        │
│              │         EXECUTE QUERIES           │        │
│              │  (parallel across collections,    │        │
│              │   mixed search + aggregation)     │        │
│              └───────────────────────┬──────────┘        │
│                                      │                   │
│  ┌──────────┐   ┌──────────┐   ┌────▼─────┐             │
│  │ RESPOND  │◀──│  MERGE   │◀──│ EVALUATE │             │
│  │ (synthe- │   │  (RRF +  │   │ (suffic- │◀──┐        │
│  │  size)   │   │  dedup)  │   │  iency?) │   │        │
│  └──────────┘   └──────────┘   └────┬─────┘   │        │
│                                      │ NO      │        │
│                                      ▼         │        │
│                               ┌──────────┐     │        │
│                               │ RE-QUERY │─────┘        │
│                               │ (adjust  │              │
│                               │  filters,│              │
│                               │  expand) │              │
│                               └──────────┘              │
│                                                          │
│  Bounded by: max_iterations, deadline_ms, token_budget   │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Pseudo-Algorithm: Production Query Agent

```
ALGORITHM: QUERY_AGENT_ORCHESTRATOR
──────────────────────────────────────────────────────────────

Input:
  q_raw           : string             — raw user utterance
  history         : ConvTurn[]         — conversation history
  collection_registry : CollectionMeta[] — schema, stats, capabilities per collection
  tool_registry   : ToolManifest[]     — available MCP tool servers
  config          : AgentConfig        — max_iterations, deadline_ms, token_budget,
                                          sufficiency_threshold, min_results

Output:
  response        : AgentResponse      — final answer + provenance + traces

Procedure:

  iteration ← 0
  results_cache ← {}
  action_trace ← []
  remaining_budget_ms ← config.deadline_ms

  1. INITIAL ANALYSIS
     ────────────────
     analysis ← ANALYZE(q_raw, history, collection_registry)
     //
     // Structured analysis output:
     //
     //   analysis = LLM(
     //     system="You are a query planning agent. Analyze the user query
     //             given the available data collections and conversation context.
     //             Determine:
     //             1. primary_intent: the user's core information need
     //             2. required_operations: [SEARCH | AGGREGATE | FILTER | COMPARE]
     //             3. target_collections: ranked list of relevant collections
     //                with justification (matched by schema fields, data types)
     //             4. complexity_class: SIMPLE | MULTI_FACET | TEMPORAL | COMPARATIVE
     //             5. required_augmentations: [REWRITE | EXPAND | DECOMPOSE | NONE]
     //             6. initial_filters: extracted filter predicates from the query
     //             7. confidence: float in [0,1]",
     //     user=FORMAT(q_raw, history, collection_registry.summaries),
     //     response_format=AnalysisSchema
     //   )
     //
     // Time cost: ~200-500ms (single LLM call)
     // Token cost: ~800-1200 input, ~200-400 output

     action_trace.append(TraceEntry("ANALYZE", analysis, elapsed_ms()))

  2. STRATEGY PLANNING
     ──────────────────
     plan ← PLAN_STRATEGY(analysis, config)
     //
     // Decision matrix for strategy selection:
     //
     //   IF analysis.complexity_class == SIMPLE:
     //     plan.augmentation = [REWRITE]
     //     plan.execution = SINGLE_COLLECTION_SEARCH
     //     plan.max_iterations = 2
     //
     //   ELIF analysis.complexity_class == MULTI_FACET:
     //     plan.augmentation = [REWRITE, DECOMPOSE]
     //     plan.execution = MULTI_COLLECTION_PARALLEL
     //     plan.max_iterations = 3
     //
     //   ELIF analysis.complexity_class == TEMPORAL:
     //     plan.augmentation = [REWRITE, FILTER_INJECT]
     //     plan.execution = TIME_WINDOWED_SEARCH
     //     plan.max_iterations = 2
     //
     //   ELIF analysis.complexity_class == COMPARATIVE:
     //     plan.augmentation = [REWRITE, DECOMPOSE, EXPAND]
     //     plan.execution = MULTI_ENTITY_PARALLEL
     //     plan.max_iterations = 3
     //
     // Each plan specifies:
     //   - ordered augmentation pipeline
     //   - target collection routing
     //   - search type per collection (vector / keyword / hybrid / aggregation)
     //   - merge strategy (RRF / weighted / cascading)
     //   - fallback strategy if primary retrieval fails

  3. QUERY AUGMENTATION EXECUTION
     ────────────────────────────
     envelope ← EMPTY_ENVELOPE(q_raw)

     FOR EACH augmentation IN plan.augmentation:
       SWITCH augmentation:
         CASE REWRITE:
           envelope ← SOTA_QUERY_REWRITE(q_raw, history, schema, ...)
           // (Algorithm from Section 3.3)

         CASE EXPAND:
           expanded, rrf ← CONTROLLED_QUERY_EXPANSION(
             envelope.generation_query.rewritten_query, ...)
           // (Algorithm from Section 4.3)
           envelope.retrieval_queries.extend(expanded)

         CASE DECOMPOSE:
           dag, schedule ← DAG_QUERY_DECOMPOSITION(
             envelope.generation_query.rewritten_query, ...)
           // (Algorithm from Section 5.3)
           envelope.generation_query.decomposed_subqueries ← dag.sub_queries()

         CASE FILTER_INJECT:
           filters ← EXTRACT_TEMPORAL_FILTERS(q_raw, analysis)
           envelope.metadata_filters.extend(filters)

  4. COLLECTION ROUTING
     ──────────────────
     routed_plan ← ROUTE_TO_COLLECTIONS(envelope, collection_registry)
     //
     // For each retrieval query in the envelope, determine:
     //   - Which collection(s) to target
     //   - What search type to use (based on collection capabilities)
     //   - What top_k to request (based on collection size and relevance estimate)
     //
     // Routing function:
     //   FOR EACH rq IN envelope.retrieval_queries:
     //     scores ← []
     //     FOR EACH coll IN collection_registry:
     //       field_overlap ← |fields(rq) ∩ fields(coll)| / |fields(rq)|
     //       schema_match  ← SCHEMA_SIM(rq.type, coll.capabilities)
     //       historical_hit← HISTORICAL_HIT_RATE(rq.intent, coll.id)
     //       score ← 0.4 * field_overlap + 0.3 * schema_match + 0.3 * historical_hit
     //       scores.append((coll, score))
     //
     //     rq.target_collections ← TOP_N(scores, n=3, threshold=0.3)

  5. RETRIEVAL EXECUTION LOOP
     ────────────────────────
     WHILE iteration < plan.max_iterations AND remaining_budget_ms > 0:

       5a. EXECUTE QUERIES (parallel)
           ────────────────────────
           results ← PARALLEL_EXECUTE(routed_plan, deadline=remaining_budget_ms * 0.6)
           //
           // For each (query, collection) pair:
           //   Use MCP tool invocation to call the appropriate search tool:
           //
           //   result = mcp.call_tool(
           //     server=collection.mcp_server,
           //     tool="search",
           //     arguments={
           //       "query": rq.query_text,
           //       "type": rq.type,
           //       "filters": envelope.metadata_filters,
           //       "limit": rq.top_k,
           //       "include_vectors": false,
           //       "include_metadata": true
           //     },
           //     timeout=per_query_timeout
           //   )
           //
           // Parallel execution with deadline propagation:
           //   All queries share a common deadline.
           //   Partial results are accepted if deadline expires.
           //   Failed queries emit structured error + continue.

       5b. RESULT FUSION
           ──────────────
           fused_results ← FUSE_RESULTS(results, rrf_config)
           //
           // Apply Reciprocal Rank Fusion across all result sets:
           //   RRF_score(d) = Σ_i [ w_i / (k + rank_i(d)) ]
           //
           // Then deduplicate by document_id,
           // apply freshness boost: score *= freshness_decay(d.timestamp),
           // apply authority boost: score *= authority_weight(d.source),
           // re-sort by final fused score,
           // truncate to top_k_final.

       5c. SUFFICIENCY EVALUATION
           ───────────────────────
           sufficiency ← EVALUATE_SUFFICIENCY(fused_results, envelope, q_raw)
           //
           // Sufficiency scoring function:
           //
           //   S(results, q) = w_coverage · COVERAGE(results, q)
           //                 + w_quality  · QUALITY(results)
           //                 + w_count    · min(|results| / min_results, 1.0)
           //
           //   COVERAGE(results, q) = fraction of sub-query facets
           //                          with ≥1 relevant result
           //                          (measured by NLI entailment score ≥ 0.7)
           //
           //   QUALITY(results)     = mean(relevance_score(d) for d in results)
           //
           //   w_coverage=0.5, w_quality=0.3, w_count=0.2
           //
           // IF S ≥ config.sufficiency_threshold (e.g., 0.75):
           //   EXIT loop → proceed to response generation
           //
           // ELSE:
           //   Identify the DEFICIT:
           //   - Which facets have insufficient coverage?
           //   - Which collections returned no results?
           //   - What filter might be too restrictive?

           IF sufficiency.score ≥ config.sufficiency_threshold:
             BREAK  // Sufficient results obtained

       5d. RE-QUERY ADAPTATION
           ────────────────────
           adaptation ← PLAN_REQUERY(sufficiency.deficit, envelope, results_cache)
           //
           // Adaptation strategies (selected based on deficit type):
           //
           //   DEFICIT: no_results_for_facet_X
           //   → Relax filters for facet X, expand query for facet X,
           //     try alternative collection
           //
           //   DEFICIT: low_relevance_scores
           //   → Rewrite query with different terminology,
           //     use HyDE for that specific facet
           //
           //   DEFICIT: overly_restrictive_filters
           //   → Remove lowest-confidence filter, widen date range
           //
           //   DEFICIT: wrong_collection
           //   → Route to next-best collection in ranking
           //
           // Apply adaptation to envelope and routed_plan:
           envelope ← APPLY_ADAPTATION(envelope, adaptation)
           routed_plan ← RE_ROUTE(envelope, collection_registry)

           action_trace.append(TraceEntry("RE_QUERY", adaptation, elapsed_ms()))
           iteration += 1
           remaining_budget_ms -= elapsed_since_last_iteration()

  6. RESPONSE SYNTHESIS
     ──────────────────
     IF plan.requires_generation:
       //
       // Assemble the synthesis prompt:
       //
       //   synthesis_context = COMPILE_PREFILL(
       //     role_policy    = "You are a precise technical assistant...",
       //     task_objective = envelope.generation_query.synthesized_intent,
       //     evidence       = FORMAT_EVIDENCE(fused_results, max_tokens=T_evidence),
       //     constraints    = [
       //       "Answer ONLY based on the provided evidence.",
       //       "Cite evidence passages by [source_id].",
       //       "If evidence is insufficient, state what is missing."
       //     ],
       //     format_spec    = dag.synthesis_strategy
       //   )
       //
       //   Token budget allocation:
       //     T_total    = context_window - T_reserved_output
       //     T_system   = min(500, 0.1 * T_total)
       //     T_evidence = min(0.7 * T_total, |fused_results_tokens|)
       //     T_history  = T_total - T_system - T_evidence

       response_text ← LLM(synthesis_context, max_tokens=T_output)
     ELSE:
       // Return structured results directly (e.g., for aggregation queries)
       response_text ← FORMAT_STRUCTURED(fused_results)

  7. PROVENANCE AND TRACE ASSEMBLY
     ─────────────────────────────
     response ← AgentResponse(
       answer         = response_text,
       sources        = fused_results.provenance_list(),
       query_trace    = action_trace,
       envelope       = envelope,
       iterations     = iteration,
       sufficiency    = sufficiency,
       total_latency  = elapsed_total_ms(),
       token_usage    = accumulate_tokens(action_trace),
       confidence     = sufficiency.score * analysis.confidence
     )

  RETURN response
```

### 6.4 Contextual Awareness: Memory-Augmented Query Resolution

The query agent maintains contextual awareness through a **tiered memory protocol**:

| Memory Layer | Purpose | Write Policy | Read Latency |
|---|---|---|---|
| **Working Memory** | Current query state, intermediate results, active sub-queries | Ephemeral, auto-expire on request completion | < 1ms (in-process) |
| **Session Memory** | Conversation history, resolved co-references, established filters | Validated per-turn, expires on session close | < 5ms (session store) |
| **Episodic Memory** | Past query patterns that led to successful retrievals for this user | Written after positive feedback, deduplicated | < 50ms (user store) |
| **Semantic Memory** | Organizational terminology mappings, query-to-collection routing priors | Admin-curated, version-controlled, validated | < 20ms (shared store) |

The agent reads from all layers during the ANALYZE phase:

$$
\text{context}_{\text{agent}} = \text{Working} \oplus \text{Session}_{[t-K:t]} \oplus \text{Episodic}_{\text{top-n}} \oplus \text{Semantic}_{\text{relevant}}
$$

Memory writes are governed by strict promotion policies:

$$
\text{Promote}(m) \iff \text{novelty}(m) > \theta_n \;\wedge\; \text{correctness\_signal}(m) = \text{TRUE} \;\wedge\; \neg\text{duplicate}(m, \mathcal{M}_{\text{existing}})
$$

---

## 7. Unified Quality Gates and Evaluation Infrastructure

### 7.1 Query Augmentation Evaluation Metrics

Every query augmentation component must be evaluated against measurable quality gates:

| Metric | Definition | Target | Measurement |
|---|---|---|---|
| **Intent Preservation Index (IPI)** | $\frac{\|E(q) \cap E(q')\|}{\|E(q)\|} \cdot \cos(\mathbf{v}(q), \mathbf{v}(q'))$ | $\geq 0.85$ | Computed per-rewrite |
| **Retrieval Recall Lift** | $\frac{\text{Recall}@k(q') - \text{Recall}@k(q)}{\text{Recall}@k(q)}$ | $\geq +15\%$ | Measured on eval set |
| **Expansion Diversity** | $1 - \frac{1}{\binom{n}{2}}\sum_{i<j} \cos(\mathbf{v}(q_i'), \mathbf{v}(q_j'))$ | $\geq 0.3$ | Per expansion set |
| **Decomposition Coverage** | $\frac{\|\text{facets\_covered}\|}{\|\text{facets\_required}\|}$ | $= 1.0$ | Per decomposition |
| **Agent Loop Efficiency** | $\frac{\text{sufficiency\_score}}{\text{iterations} \cdot \text{cost}}$ | Maximized | Per request |
| **End-to-End Latency** | Total wall-clock from $q_{\text{raw}}$ to $\text{response}$ | $\leq \tau_{\text{SLA}}$ | p50, p95, p99 |
| **Drift Violation Rate** | $\frac{\|\{q' \mid \text{IPI}(q, q') < \theta\}\|}{\|\text{total\_augmentations}\|}$ | $\leq 2\%$ | Continuous monitoring |

### 7.2 CI/CD Integration

```
PIPELINE: QUERY_AUGMENTATION_EVAL_CI
────────────────────────────────────

Trigger: on_commit to query_augmentation/* OR weekly_schedule

Stages:

  1. REPLAY_SET_EXECUTION
     ─────────────────────
     FOR EACH (q_raw, expected_results, ground_truth_intent) IN eval_corpus:
       envelope ← QUERY_AGENT_ORCHESTRATOR(q_raw, ...)
       metrics  ← COMPUTE_METRICS(envelope, expected_results, ground_truth_intent)
       store(metrics, run_id)

  2. REGRESSION_DETECTION
     ─────────────────────
     baseline ← load_metrics(baseline_run_id)
     current  ← load_metrics(current_run_id)
     FOR EACH metric IN [IPI, recall_lift, diversity, coverage, latency]:
       delta ← current[metric] - baseline[metric]
       IF delta < -regression_threshold[metric]:
         FAIL_BUILD("Regression detected: {metric} degraded by {delta}")

  3. DRIFT_AUDIT
     ────────────
     drift_violations ← current.filter(IPI < θ_IPI)
     IF |drift_violations| / |eval_corpus| > 0.02:
       FAIL_BUILD("Drift violation rate exceeds 2%")

  4. COST_BUDGET_CHECK
     ──────────────────
     avg_tokens ← mean(current.token_usage)
     avg_calls  ← mean(current.llm_calls)
     IF avg_tokens > token_budget_per_query OR avg_calls > max_calls_per_query:
       WARN("Cost budget exceeded: {avg_tokens} tokens, {avg_calls} calls")

  5. ARTIFACT_PUBLICATION
     ─────────────────────
     publish_metrics(current, dashboard)
     publish_replay_traces(current, trace_store)
     update_baseline_if_passing(current)
```

---

## 8. Production Reliability Engineering

### 8.1 Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation | Recovery |
|---|---|---|---|
| **LLM rewrite timeout** | Deadline exceeded | Fall back to co-reference-resolved $q_{\text{resolved}}$ | Continue with degraded quality; log event |
| **LLM generates hallucinated filters** | Filter predicate references non-existent schema field | Validate all filters against schema before execution | Drop invalid filters; emit structured warning |
| **Expansion drift** | IPI below threshold | Reject drifted expansions; use only validated subset | Minimum viable expansion set (original + 1 HyDE) |
| **Decomposition cycle** | Cycle detected in DAG | Break back-edges by confidence; re-validate DAG | Flatten to parallel independent sub-queries |
| **Agent loop divergence** | Iterations exceed $n_{\text{max}}$ with no sufficiency improvement | Force-terminate loop; return best available results | Return partial results with explicit insufficiency flag |
| **All collections return empty** | Zero results across all routed collections | Widen filters, remove metadata constraints, try broader collection | If still empty, return "no information found" with suggested reformulations |
| **Token budget exhaustion** | Accumulated tokens exceed budget | Terminate current expansion/decomposition step | Proceed with results obtained so far |

### 8.2 Idempotency and Determinism

Query augmentation pipelines must be **idempotent**: applying the same augmentation twice must produce equivalent results. This is enforced by:

1. **Temperature control**: All LLM calls in the augmentation pipeline use $T \leq 0.3$ for near-deterministic outputs.
2. **Seed pinning**: Where supported, fix the random seed per request ID for reproducibility.
3. **Cache-first execution**: Hash the (query, history, schema) tuple; return cached augmentation result if available within TTL.

$$
\text{cache\_key} = \text{SHA256}\left(q_{\text{raw}} \| \text{hash}(\mathcal{H}_{[-K:]}) \| \text{schema\_version}\right)
$$

### 8.3 Observability Contract

Every query augmentation invocation emits a structured trace conforming to:

```json
{
  "trace_id": "uuid",
  "span_name": "query_augmentation",
  "spans": [
    {"name": "coreference_resolution", "duration_ms": 45, "tokens_in": 320, "tokens_out": 80},
    {"name": "intent_classification",  "duration_ms": 120, "tokens_in": 450, "tokens_out": 50, "result": "DIAGNOSTIC"},
    {"name": "step_back_abstraction",  "duration_ms": 180, "tokens_in": 500, "tokens_out": 90},
    {"name": "entity_extraction",      "duration_ms": 95,  "tokens_in": 400, "tokens_out": 120},
    {"name": "keyword_densification",  "duration_ms": 150, "tokens_in": 380, "tokens_out": 60, "density_score": 0.42},
    {"name": "expansion_hyde",         "duration_ms": 200, "tokens_in": 450, "tokens_out": 150},
    {"name": "expansion_perspectives", "duration_ms": 350, "tokens_in": 900, "tokens_out": 240, "accepted": 3, "rejected": 1},
    {"name": "drift_validation",       "duration_ms": 15,  "ipi_score": 0.91, "passed": true},
    {"name": "collection_routing",     "duration_ms": 8,   "collections": ["docs_v2", "kb_internal"]}
  ],
  "total_duration_ms": 1163,
  "total_tokens": 3690,
  "augmentation_methods": ["REWRITE", "EXPAND_HYDE", "EXPAND_PERSPECTIVE"],
  "drift_score": 0.09,
  "final_retrieval_queries": 5
}
```

---

## 9. Architectural Integration: Where Query Augmentation Sits in the Agentic Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER / APPLICATION                        │
│                     (JSON-RPC boundary)                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ q_raw + session_id
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   QUERY AUGMENTATION LAYER                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Co-Ref   │→ │ Intent   │→ │ Rewrite  │→ │ Expand /         │ │
│  │ Resolve  │  │ Classify │  │ + Dense  │  │ Decompose /      │ │
│  │          │  │          │  │          │  │ Agent Orchestrate │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┬─────────┘ │
│                                                      │           │
│  Output: AugmentedQueryEnvelope (typed Protobuf)     │           │
└──────────────────────────────────────────────────────┼───────────┘
                                                       │
                             ┌─────────────────────────▼──────────┐
                             │      RETRIEVAL ENGINE               │
                             │  (hybrid: semantic + keyword +      │
                             │   graph + metadata filters)         │
                             │  Collection routing via MCP         │
                             │  RRF fusion + freshness/authority   │
                             └─────────────────────────┬──────────┘
                                                       │
                             ┌─────────────────────────▼──────────┐
                             │      CONTEXT COMPILER (PREFILL)     │
                             │  Assemble: role + evidence +        │
                             │  memory + constraints + format      │
                             │  Budget: explicit token allocation  │
                             └─────────────────────────┬──────────┘
                                                       │
                             ┌─────────────────────────▼──────────┐
                             │      GENERATION + VERIFICATION      │
                             │  LLM synthesis → critique →         │
                             │  repair → commit                    │
                             └─────────────────────────┬──────────┘
                                                       │
                             ┌─────────────────────────▼──────────┐
                             │      RESPONSE DELIVERY              │
                             │  + provenance + traces + feedback   │
                             └────────────────────────────────────┘
```

---

## 10. Summary of Key Invariants

| Invariant | Enforcement Mechanism |
|---|---|
| No raw query reaches the retrieval engine | Pipeline architecture: augmentation layer is mandatory |
| Every augmented query has provenance | `QueryProvenance` field is required in Protobuf schema |
| Drift never exceeds $\delta_{\text{max}}$ | IPI validation gate with automatic fallback |
| Expansion cardinality is bounded | MMR selection with hard cap $n_{\text{max}}$ |
| Decomposition depth is bounded | DAG depth enforcement with flatten-to-depth |
| Agent loop terminates | Iteration cap + deadline + token budget |
| All failures are observable | Structured trace emission on every span |
| Augmentation is idempotent | Cache-first with content-addressed keys |
| Cost is bounded per request | Token and LLM-call budget enforcement |
| Quality is continuously validated | CI pipeline with regression detection on eval corpus |

---

This report establishes query augmentation as a **typed, bounded, observable subsystem** within the agentic context engineering stack—not an ad hoc preprocessing step. Every transformation is governed by formal quality constraints, drift bounds, latency budgets, and typed protocol contracts, ensuring that the system operates predictably, safely, and cost-efficiently at production scale. The mathematical formulations, pseudo-algorithms, and architectural contracts provided herein constitute a complete implementation specification for a principal-level engineering team.