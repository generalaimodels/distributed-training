# Retrieval & Chunking Architecture for Agentic AI Systems

## A Principal-Level Technical Report on Deterministic Retrieval Engineering, Provenance-Aware Chunking, and Context-Optimal Document Decomposition

---

## Table of Contents

1. [Formal Problem Statement & Retrieval as a Control System](#1-formal-problem-statement)
2. [Chunking Theory: Information-Theoretic Foundations](#2-chunking-theory)
3. [The Chunking Strategy Matrix: Formal Optimization Framework](#3-chunking-strategy-matrix)
4. [Simple Chunking Techniques: Algorithmic Specifications](#4-simple-chunking-techniques)
5. [Advanced Chunking Techniques: SOTA Architectures](#5-advanced-chunking-techniques)
6. [Late Chunking: Contextual Embedding Inversion](#6-late-chunking)
7. [Pre-Chunking vs. Post-Chunking: Architectural Trade-Off Analysis](#7-pre-vs-post-chunking)
8. [Chunking Strategy Selection: Decision Framework](#8-strategy-selection)
9. [End-to-End Retrieval Pipeline: Production Architecture](#9-end-to-end-pipeline)
10. [Evaluation Infrastructure & Quality Gates](#10-evaluation-infrastructure)
11. [Operational Considerations: Reliability, Cost, Latency](#11-operational-considerations)

---

## 1. Formal Problem Statement & Retrieval as a Control System

### 1.1 The Retrieval Constraint

An LLM operates under a fixed context window $W$ measured in tokens. Given a corpus $\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$ with total token volume $|\mathcal{D}| \gg W$, the retrieval problem is to identify a subset $\mathcal{S} \subset \mathcal{D}$ such that:

$$\mathcal{S}^* = \arg\max_{\mathcal{S}} \; U(\mathcal{S}, q) \quad \text{subject to} \quad \sum_{s_i \in \mathcal{S}} |s_i| \leq W - B_{\text{reserved}}$$

where:
- $q$ is the user query (possibly decomposed into subqueries $\{q_1, \ldots, q_m\}$),
- $U(\mathcal{S}, q)$ is the **task utility function** measuring how well the retrieved set enables correct answer generation,
- $B_{\text{reserved}}$ is the token budget reserved for role policy, tool affordances, memory summaries, and generation capacity.

The critical insight is that $U$ is **not** simply relevance—it is a composite function:

$$U(\mathcal{S}, q) = \alpha \cdot \text{Relevance}(\mathcal{S}, q) + \beta \cdot \text{Completeness}(\mathcal{S}, q) + \gamma \cdot \text{Authority}(\mathcal{S}) + \delta \cdot \text{Freshness}(\mathcal{S}) - \lambda \cdot \text{Redundancy}(\mathcal{S})$$

where $\alpha + \beta + \gamma + \delta = 1$ and $\lambda$ is a penalty coefficient for information overlap.

### 1.2 Retrieval as a Deterministic Engine, Not Stochastic Search

In production agentic systems, retrieval must be treated as a **deterministic retrieval engine with provenance**, not as ad hoc RAG. Every retrieved chunk $s_i$ must carry:

```
RetrievedChunk {
    content: string,
    chunk_id: UUID,
    source_document_id: UUID,
    source_uri: URI,
    chunk_method: ChunkingStrategy,
    embedding_model_version: SemVer,
    retrieval_score: float,
    provenance: ProvenanceRecord {
        ingestion_timestamp: ISO8601,
        last_validated: ISO8601,
        authority_tier: enum {CANONICAL, REVIEWED, RAW},
        lineage_chain: List[TransformationStep]
    },
    freshness_epoch: uint64,
    token_count: uint32
}
```

This typed contract ensures that every piece of context entering the LLM's window is **auditable, traceable, and scoreable**.

### 1.3 Pseudo-Algorithm: Top-Level Retrieval Control Loop

```
ALGORITHM: RETRIEVAL_CONTROL_LOOP
────────────────────────────────────────────────────
Input:  q (raw user query), W (context window budget),
        B_reserved (tokens for system prompt, tools, memory),
        D (document corpus with index structures)
Output: S* (provenance-tagged, ranked chunk set)

1.  B_retrieval ← W - B_reserved
2.  Q ← QUERY_DECOMPOSE(q)                    // subquery expansion
3.  FOR EACH q_i IN Q:
4.      R_exact_i  ← EXACT_MATCH(q_i, D)      // BM25 / keyword
5.      R_semantic_i ← SEMANTIC_SEARCH(q_i, D) // dense vector ANN
6.      R_meta_i   ← METADATA_FILTER(q_i, D)  // structured filters
7.      R_graph_i  ← LINEAGE_TRAVERSE(q_i, D) // graph/knowledge base
8.      R_i ← RECIPROCAL_RANK_FUSION(R_exact_i, R_semantic_i,
                                       R_meta_i, R_graph_i)
9.  ENDFOR
10. R_merged ← CROSS_QUERY_DEDUPLICATE(∪ R_i)
11. R_ranked ← AUTHORITY_FRESHNESS_RERANK(R_merged)
12. S* ← BUDGET_PACK(R_ranked, B_retrieval)    // greedy knapsack
13. ATTACH_PROVENANCE(S*)
14. RETURN S*
────────────────────────────────────────────────────
```

---

## 2. Chunking Theory: Information-Theoretic Foundations

### 2.1 Why Chunking is the Dominant Performance Lever

Chunking transforms a continuous document $d$ into a discrete set of chunks $\mathcal{C}(d) = \{c_1, c_2, \ldots, c_k\}$. This transformation determines the **granularity of the retrieval unit**, which in turn determines the upper bound on retrieval precision.

**Theorem (Chunking-Retrieval Precision Bound):** For a retrieval system with embedding function $\phi$ and similarity function $\text{sim}$, the maximum achievable retrieval precision $P^*$ for a query $q$ is bounded by:

$$P^* \leq 1 - \frac{H_{\text{topic}}(c)}{H_{\text{total}}(c)}$$

where $H_{\text{topic}}(c)$ is the topical entropy within a single chunk (measuring how many distinct topics it covers) and $H_{\text{total}}(c)$ is the total information entropy of the chunk.

**Interpretation:** A chunk covering a single topic has $H_{\text{topic}} \approx 0$, yielding $P^* \approx 1$. A chunk covering many topics has high $H_{\text{topic}}$, degrading precision regardless of the embedding model's quality.

### 2.2 The Embedding Dilution Problem

When a chunk $c$ contains $T$ distinct semantic topics $\{t_1, \ldots, t_T\}$, the dense embedding $\phi(c)$ becomes an **average** over topic embeddings:

$$\phi(c) \approx \frac{1}{T} \sum_{j=1}^{T} \phi(t_j) + \epsilon$$

where $\epsilon$ represents interaction noise. For a query $q$ targeting topic $t_1$:

$$\text{sim}(\phi(q), \phi(c)) \approx \frac{1}{T} \text{sim}(\phi(q), \phi(t_1)) + \frac{T-1}{T} \cdot \bar{\text{sim}}_{\text{noise}}$$

As $T$ grows, the signal-to-noise ratio degrades as $O(1/T)$. This is the **embedding dilution effect**—the formal reason why oversized, multi-topic chunks are "rich but unfindable."

### 2.3 The Context Starvation Problem

Conversely, a chunk $c$ that is too small (e.g., a single sentence) may have a highly precise embedding but lacks **self-contained semantic closure**. Define the **contextual completeness** of a chunk as:

$$\text{CC}(c, q) = \frac{|I(c) \cap I_{\text{required}}(q)|}{|I_{\text{required}}(q)|}$$

where $I(c)$ is the information content of chunk $c$ and $I_{\text{required}}(q)$ is the information required to correctly answer query $q$.

A single-sentence chunk may have $\text{CC}(c, q) \ll 1$ even when perfectly retrieved, leading to **context starvation**: the LLM receives the correct fragment but cannot synthesize a correct answer.

### 2.4 Formal Objective: The Chunking Optimization Problem

The optimal chunking function $\mathcal{C}^*$ solves:

$$\mathcal{C}^* = \arg\max_{\mathcal{C}} \; \mathbb{E}_{q \sim Q} \left[ \sum_{c \in \text{Top-}k(\mathcal{C}(d), q)} \text{CC}(c, q) \cdot \text{RetrievalProbability}(c, q) \right]$$

subject to:

$$\forall c \in \mathcal{C}(d): \; |c| \in [L_{\min}, L_{\max}] \quad \text{and} \quad H_{\text{topic}}(c) \leq \tau$$

where $L_{\min}, L_{\max}$ are token bounds and $\tau$ is a topical entropy threshold.

This is a **joint optimization over retrieval precision and contextual richness**—the mathematical formalization of the "chunking sweet spot."

---

## 3. The Chunking Strategy Matrix: Formal Optimization Framework

### 3.1 Two-Dimensional Quality Space

Define two axes for any chunk $c$:

| Axis | Metric | Formal Definition |
|---|---|---|
| **Retrieval Precision** ($\mathcal{P}$) | Probability that $c$ is retrieved when it is the correct answer | $P(\text{rank}(c, q) \leq k \mid c \in \text{GoldSet}(q))$ |
| **Contextual Richness** ($\mathcal{R}$) | Fraction of answer-required information contained in $c$ | $\text{CC}(c, q)$ as defined above |

### 3.2 Quadrant Analysis

| Quadrant | $\mathcal{P}$ | $\mathcal{R}$ | Diagnosis | Failure Mode |
|---|---|---|---|---|
| **Precise but Incomplete** | High | Low | Chunks are atomic sentences; easy to match but insufficient for generation | LLM hallucinates missing context; requires multi-hop retrieval overhead |
| **The Failure Zone** | Low | Low | Random or poorly bounded chunks; neither findable nor useful | System returns irrelevant, incoherent fragments; complete retrieval failure |
| **The Sweet Spot** | High | High | Semantically coherent, topically focused, self-contained units | None—target operating point |
| **Rich but Unfindable** | Low | High | Large multi-topic chunks; contain the answer but embedding is diluted | Correct information exists but is never surfaced; silent failure mode |

### 3.3 Pseudo-Algorithm: Sweet-Spot Verification Gate

```
ALGORITHM: CHUNK_QUALITY_GATE
────────────────────────────────────────────────────
Input:  C (set of candidate chunks), Q_eval (evaluation query set),
        k (retrieval depth), τ_P (precision threshold),
        τ_R (richness threshold)
Output: PASS / FAIL with diagnostic report

1.  FOR EACH q IN Q_eval:
2.      gold ← GOLD_CHUNKS(q)
3.      retrieved ← TOP_K_RETRIEVE(C, q, k)
4.      P_q ← |retrieved ∩ gold| / |gold|        // recall@k
5.      R_q ← AVG_{c ∈ retrieved ∩ gold} CC(c, q) // avg completeness
6.  ENDFOR
7.  P_mean ← MEAN(P_q for all q)
8.  R_mean ← MEAN(R_q for all q)
9.  IF P_mean ≥ τ_P AND R_mean ≥ τ_R:
10.     RETURN PASS
11. ELIF P_mean < τ_P AND R_mean ≥ τ_R:
12.     RETURN FAIL("Rich but Unfindable: reduce chunk size")
13. ELIF P_mean ≥ τ_P AND R_mean < τ_R:
14.     RETURN FAIL("Precise but Incomplete: increase chunk size or add parent context")
15. ELSE:
16.     RETURN FAIL("Failure Zone: redesign chunking strategy")
────────────────────────────────────────────────────
```

---

## 4. Simple Chunking Techniques: Algorithmic Specifications

### 4.1 Fixed-Size Chunking

**Formal Definition:** Given document token sequence $\mathbf{d} = (t_1, t_2, \ldots, t_n)$, chunk size $s$, and overlap $o$ where $o < s$:

$$c_i = (t_{i(s-o)+1}, \; t_{i(s-o)+2}, \; \ldots, \; t_{i(s-o)+s}) \quad \text{for } i = 0, 1, \ldots, \left\lfloor \frac{n - s}{s - o} \right\rfloor$$

**Overlap Rationale:** Without overlap ($o=0$), a relevant sentence straddling a chunk boundary is split between two chunks, neither of which contains the complete semantic unit. Overlap $o > 0$ duplicates boundary tokens, ensuring that for any contiguous span of length $\leq o$, at least one chunk contains it entirely.

**Token Budget Impact of Overlap:**

$$\text{Total chunks} = \left\lceil \frac{n - o}{s - o} \right\rceil \quad \Rightarrow \quad \text{Total stored tokens} = s \cdot \left\lceil \frac{n - o}{s - o} \right\rceil$$

The storage overhead ratio is:

$$\text{Overhead} = \frac{s \cdot \lceil(n-o)/(s-o)\rceil}{n} \approx \frac{s}{s - o}$$

For $s=512, o=50$: overhead $\approx 1.108$ (10.8% storage increase).

```
ALGORITHM: FIXED_SIZE_CHUNK
────────────────────────────────────────────────────
Input:  tokens[1..n], chunk_size s, overlap o
Output: chunks[]

1.  stride ← s - o
2.  i ← 1
3.  WHILE i ≤ n:
4.      end ← MIN(i + s - 1, n)
5.      chunks.APPEND(tokens[i..end])
6.      i ← i + stride
7.  ENDWHILE
8.  RETURN chunks
────────────────────────────────────────────────────
```

**Trade-Off Analysis:**

| Property | Assessment |
|---|---|
| Complexity | $O(n)$, trivially parallelizable |
| Semantic coherence | None—cuts mid-sentence, mid-paragraph |
| Precision | Low: boundary artifacts create noisy embeddings |
| Use case | Baseline, speed-critical pipelines, homogeneous corpora |

### 4.2 Recursive Chunking

**Formal Definition:** Given a prioritized separator list $\mathbf{S} = (s_1, s_2, \ldots, s_m)$ ordered by structural significance (e.g., $s_1 = $ paragraph break, $s_2 = $ sentence boundary, $s_3 = $ word boundary), and a maximum chunk size $L_{\max}$:

```
ALGORITHM: RECURSIVE_CHUNK
────────────────────────────────────────────────────
Input:  text, separators S[1..m], L_max, depth=1
Output: chunks[]

1.  IF |text| ≤ L_max:
2.      RETURN [text]
3.  IF depth > m:
4.      RETURN FIXED_SIZE_CHUNK(TOKENIZE(text), L_max, overlap=0)
5.  segments ← SPLIT(text, S[depth])
6.  chunks ← []
7.  buffer ← ""
8.  FOR EACH seg IN segments:
9.      IF |buffer + seg| ≤ L_max:
10.         buffer ← buffer + S[depth] + seg   // preserve separator
11.     ELSE:
12.         IF buffer ≠ "":
13.             chunks.APPEND(buffer)
14.         IF |seg| > L_max:
15.             chunks.EXTEND(RECURSIVE_CHUNK(seg, S, L_max, depth+1))
16.         ELSE:
17.             buffer ← seg
18.     ENDIF
19. ENDFOR
20. IF buffer ≠ "":
21.     chunks.APPEND(buffer)
22. RETURN chunks
────────────────────────────────────────────────────
```

**Key Property:** The algorithm preserves the highest-level structural boundary possible. Paragraph breaks are respected first; only when a paragraph exceeds $L_{\max}$ does it fall back to sentence splitting, then word splitting. This yields chunks that are **structurally aligned** with the document's natural segmentation.

### 4.3 Document-Based (Structural) Chunking

**Formal Definition:** Given a document $d$ with a structural parse tree $T(d)$ (e.g., Markdown heading hierarchy, HTML DOM, AST for source code), chunks are defined as the content spans associated with leaf or intermediate nodes at a chosen depth $\ell$ of $T(d)$:

$$\mathcal{C}(d) = \{ \text{content}(v) \mid v \in T(d), \; \text{depth}(v) = \ell, \; |\text{content}(v)| \leq L_{\max} \}$$

Nodes exceeding $L_{\max}$ are recursively split at the next structural level.

```
ALGORITHM: STRUCTURAL_CHUNK
────────────────────────────────────────────────────
Input:  parse_tree T, L_max, target_depth ℓ
Output: chunks[]

1.  nodes ← COLLECT_NODES_AT_DEPTH(T, ℓ)
2.  chunks ← []
3.  FOR EACH node IN nodes:
4.      IF |node.content| ≤ L_max:
5.          chunk ← ChunkRecord {
6.              content: node.content,
7.              structural_path: node.ancestors(),  // e.g., "§3 > §3.2 > ¶1"
8.              heading_chain: node.heading_lineage()
9.          }
10.         chunks.APPEND(chunk)
11.     ELSE:
12.         children ← node.children()
13.         IF children ≠ ∅:
14.             chunks.EXTEND(STRUCTURAL_CHUNK(subtree(node), L_max, ℓ+1))
15.         ELSE:
16.             chunks.EXTEND(RECURSIVE_CHUNK(node.content, DEFAULT_SEPS, L_max))
17.     ENDIF
18. ENDFOR
19. RETURN chunks
────────────────────────────────────────────────────
```

**Critical Enhancement:** Each chunk retains its **structural path** (e.g., `"Chapter 3 → Section 3.2 → Paragraph 1"`). This metadata becomes a first-class retrieval signal—enabling **metadata-filtered retrieval** and providing the LLM with explicit document-positional context that prevents hallucinated attribution.

---

## 5. Advanced Chunking Techniques: SOTA Architectures

### 5.1 Semantic Chunking

**Core Principle:** Instead of using syntactic separators, semantic chunking detects **topic shift boundaries** in the text by measuring inter-sentence embedding similarity.

**Formal Algorithm:**

Given a document as a sequence of sentences $\mathbf{d} = (s_1, s_2, \ldots, s_n)$ and an embedding function $\phi$:

1. Compute pairwise adjacent cosine similarities:
$$\sigma_i = \text{cosine}(\phi(s_i), \phi(s_{i+1})) \quad \text{for } i = 1, \ldots, n-1$$

2. Compute the similarity gradient (rate of topic change):
$$\Delta_i = \sigma_{i-1} - \sigma_i \quad \text{for } i = 2, \ldots, n-1$$

3. Detect breakpoints where similarity drops below a threshold $\theta$ or where the gradient exceeds a shift magnitude $\delta$:
$$\text{Breakpoints} = \{i \mid \sigma_i < \theta\} \cup \{i \mid \Delta_i > \delta\}$$

4. Segment the sentence sequence at breakpoints.

```
ALGORITHM: SEMANTIC_CHUNK
────────────────────────────────────────────────────
Input:  sentences S[1..n], embedding_fn φ, threshold θ,
        gradient_threshold δ, L_max, L_min
Output: chunks[]

1.  embeddings ← [φ(S[i]) FOR i IN 1..n]
2.  similarities ← []
3.  FOR i ← 1 TO n-1:
4.      σ_i ← COSINE(embeddings[i], embeddings[i+1])
5.      similarities.APPEND(σ_i)
6.  ENDFOR

7.  // Adaptive threshold: percentile-based
8.  θ_adaptive ← PERCENTILE(similarities, 25)  // bottom quartile
9.  θ_effective ← MIN(θ, θ_adaptive)

10. breakpoints ← {0}  // always start a chunk at position 0
11. FOR i ← 1 TO n-2:
12.     Δ_i ← similarities[i-1] - similarities[i]
13.     IF similarities[i] < θ_effective OR Δ_i > δ:
14.         breakpoints.ADD(i+1)
15. ENDFOR
16. breakpoints.ADD(n)

17. // Construct chunks with size constraints
18. chunks ← []
19. sorted_bp ← SORT(breakpoints)
20. FOR j ← 0 TO |sorted_bp|-2:
21.     candidate ← JOIN(S[sorted_bp[j]..sorted_bp[j+1]-1])
22.     IF |candidate| > L_max:
23.         chunks.EXTEND(RECURSIVE_CHUNK(candidate, DEFAULT_SEPS, L_max))
24.     ELIF |candidate| < L_min AND j < |sorted_bp|-2:
25.         MERGE candidate with next segment  // avoid degenerate chunks
26.     ELSE:
27.         chunks.APPEND(candidate)
28. ENDFOR
29. RETURN chunks
────────────────────────────────────────────────────
```

**Complexity Analysis:** $O(n)$ embedding calls + $O(n)$ similarity computations. The embedding calls dominate; batching to the embedding model reduces wall-clock time to $O(\lceil n/B \rceil)$ where $B$ is the batch size.

**SOTA Enhancement — Sliding Window Similarity:** Instead of adjacent-only similarity, compute a windowed average:

$$\bar{\sigma}_i = \frac{1}{2w} \left( \sum_{j=\max(1,i-w)}^{i-1} \text{cosine}(\phi(s_j), \phi(s_i)) + \sum_{j=i+1}^{\min(n,i+w)} \text{cosine}(\phi(s_i), \phi(s_j)) \right)$$

This smooths noise from individual sentence pairs and detects macro-topic shifts more reliably.

### 5.2 LLM-Based Chunking

**Core Principle:** Delegate chunk boundary detection to an LLM that understands discourse structure, logical propositions, and argumentative flow.

**Architecture:**

```
ALGORITHM: LLM_BASED_CHUNK
────────────────────────────────────────────────────
Input:  document d, llm_fn LLM, L_max, instruction_template T
Output: chunks[] with proposition labels

1.  // Segment document into processable windows
2.  windows ← SLIDING_WINDOW(d, window_size=4096, stride=3072)

3.  all_propositions ← []
4.  FOR EACH w IN windows:
5.      prompt ← T.format(
6.          text=w,
7.          instructions="Decompose the following text into atomic,
8.              self-contained propositions. Each proposition must:
9.              (a) express exactly one fact or claim,
10.             (b) be understandable without surrounding text,
11.             (c) include necessary entity references (no dangling pronouns),
12.             (d) preserve numerical precision.
13.             Return as JSON array of {proposition, topic_label, confidence}."
14.     )
15.     result ← LLM(prompt, response_format=JSON)
16.     all_propositions.EXTEND(PARSE_JSON(result))
17. ENDFOR

18. // Deduplicate propositions from overlapping windows
19. deduped ← SEMANTIC_DEDUPLICATE(all_propositions, similarity_threshold=0.92)

20. // Group propositions by topic_label into chunks
21. groups ← GROUP_BY(deduped, key="topic_label")
22. chunks ← []
23. FOR EACH group IN groups:
24.     combined ← JOIN(group.propositions)
25.     IF |combined| > L_max:
26.         chunks.EXTEND(SEMANTIC_CHUNK(SPLIT_SENTENCES(combined), φ, θ, δ, L_max))
27.     ELSE:
28.         chunks.APPEND(combined)
29. ENDFOR
30. RETURN chunks
────────────────────────────────────────────────────
```

**Cost Model:** For a document of $n$ tokens with window size $w$ and stride $s$, the number of LLM calls is $\lceil (n - w) / s \rceil + 1$. Each call processes $w$ input tokens + generates $\approx w/2$ output tokens.

$$\text{Cost}_{\text{LLM-chunk}}(n) = \left\lceil \frac{n - w}{s} \right\rceil \cdot (C_{\text{input}} \cdot w + C_{\text{output}} \cdot \frac{w}{2})$$

For $n = 50{,}000$ tokens, $w = 4096$, $s = 3072$: $\approx 16$ calls. At current API pricing ($\sim$\$3/M input, $\sim$\$15/M output for frontier models), this costs $\approx \$0.70$ per document—acceptable for high-value corpora, prohibitive for bulk ingestion.

### 5.3 Agentic Chunking

**Core Principle:** An autonomous agent inspects a document's structure, content type, domain, and downstream use case, then **selects and composes** the optimal chunking strategy rather than applying a fixed method.

**This is the meta-chunking controller—the highest-level abstraction in the chunking hierarchy.**

```
ALGORITHM: AGENTIC_CHUNK
────────────────────────────────────────────────────
Input:  document d, agent A, strategy_registry SR, eval_fn E,
        quality_threshold τ
Output: chunks[] with strategy provenance

1.  // Phase 1: Document Analysis
2.  analysis ← A.analyze(d)
3.  // Returns: {
4.  //   doc_type: enum {STRUCTURED, SEMI_STRUCTURED, UNSTRUCTURED},
5.  //   format: enum {MARKDOWN, HTML, PDF, CODE, PLAINTEXT},
6.  //   domain: string,
7.  //   estimated_topic_count: int,
8.  //   has_tables: bool,
9.  //   has_code_blocks: bool,
10. //   avg_section_length: int,
11. //   structural_depth: int
12. // }

13. // Phase 2: Strategy Selection via Policy
14. strategy_plan ← A.select_strategy(analysis, SR)
15. // Example output:
16. // [
17. //   {region: "headers+prose", method: STRUCTURAL + SEMANTIC},
18. //   {region: "code_blocks",   method: AST_BASED},
19. //   {region: "tables",        method: ROW_GROUP_CHUNK},
20. //   {region: "references",    method: FIXED_SIZE(256)}
21. // ]

22. // Phase 3: Execute Composite Strategy
23. all_chunks ← []
24. FOR EACH plan_entry IN strategy_plan:
25.     region_text ← EXTRACT_REGION(d, plan_entry.region)
26.     strategy_fn ← SR.get(plan_entry.method)
27.     region_chunks ← strategy_fn(region_text)
28.     FOR EACH c IN region_chunks:
29.         c.strategy_provenance ← plan_entry
30.     all_chunks.EXTEND(region_chunks)
31. ENDFOR

32. // Phase 4: Quality Verification Loop
33. score ← E(all_chunks, sample_queries)
34. IF score < τ:
35.     // Agent self-critiques and adjusts
36.     feedback ← A.critique(all_chunks, score)
37.     revised_plan ← A.revise_strategy(strategy_plan, feedback)
38.     RETURN AGENTIC_CHUNK(d, A, SR, E, τ)  // bounded recursion
39. ENDIF

40. RETURN all_chunks
────────────────────────────────────────────────────
```

**Bounded Recursion:** The recursion depth is hard-capped (typically $\leq 3$). If the quality gate fails after maximum retries, the system falls back to the best-scoring attempt and flags the document for human review.

### 5.4 Hierarchical Chunking

**Core Principle:** Construct a **multi-resolution index** where the same document is represented at multiple granularity levels. Retrieval can traverse from coarse to fine.

**Formal Structure:**

Define a hierarchy of $L$ levels. At level $\ell \in \{1, \ldots, L\}$:

$$\mathcal{C}^{(\ell)}(d) = \text{chunks at granularity level } \ell$$

with the containment property:

$$\forall c^{(\ell)}_i \in \mathcal{C}^{(\ell)}: \exists \, c^{(\ell-1)}_j \in \mathcal{C}^{(\ell-1)} \text{ such that } c^{(\ell)}_i \subseteq c^{(\ell-1)}_j$$

Typical hierarchy:

| Level | Granularity | Token Range | Purpose |
|---|---|---|---|
| $\ell = 1$ | Document summary | 100–300 | Collection-level routing |
| $\ell = 2$ | Section summary | 200–500 | Topic identification |
| $\ell = 3$ | Paragraph | 100–400 | Detailed retrieval |
| $\ell = 4$ | Proposition | 30–100 | Fact-level precision |

```
ALGORITHM: HIERARCHICAL_CHUNK
────────────────────────────────────────────────────
Input:  document d, levels L, summarizer_fn Σ, chunk_fn C_fn
Output: hierarchy H = {level → chunks[]} with parent pointers

1.  // Level L (finest): generate leaf chunks
2.  H[L] ← C_fn(d, granularity=FINEST)
3.  FOR EACH c IN H[L]:
4.      c.parent_id ← NULL  // will be assigned

5.  // Levels L-1 down to 2: group and summarize
6.  FOR ℓ ← L-1 DOWNTO 2:
7.      groups ← GROUP_BY_SECTION(H[ℓ+1])
8.      H[ℓ] ← []
9.      FOR EACH group IN groups:
10.         content ← JOIN(group.chunks)
11.         summary ← Σ(content, target_tokens=TOKEN_BUDGET[ℓ])
12.         parent_chunk ← ChunkRecord {
13.             content: summary,
14.             level: ℓ,
15.             children: group.chunk_ids,
16.             section_path: group.section_path
17.         }
18.         H[ℓ].APPEND(parent_chunk)
19.         FOR EACH child IN group.chunks:
20.             child.parent_id ← parent_chunk.id
21.     ENDFOR
22. ENDFOR

23. // Level 1: document-level summary
24. H[1] ← [Σ(d, target_tokens=TOKEN_BUDGET[1])]
25. RETURN H
────────────────────────────────────────────────────
```

**Retrieval Strategy (Drill-Down):**

$$\text{retrieve}(q) = \text{Top-}k_1(\mathcal{C}^{(2)}, q) \rightarrow \text{expand children} \rightarrow \text{Top-}k_2(\mathcal{C}^{(3)}, q) \rightarrow \text{expand children} \rightarrow \text{Top-}k_3(\mathcal{C}^{(4)}, q)$$

This gives the LLM both the **summary context** (parent chunks) and the **precise evidence** (leaf chunks), solving the precision-richness trade-off structurally.

---

## 6. Late Chunking: Contextual Embedding Inversion

### 6.1 The Fundamental Problem Late Chunking Solves

In standard chunking, each chunk $c_i$ is embedded **independently**:

$$\phi_{\text{standard}}(c_i) = \text{Embed}(c_i)$$

The embedding of $c_i$ has **no awareness** of $c_{i-1}$ or $c_{i+1}$. Cross-chunk references, coreferences ("it," "this method," "the above equation"), and progressive arguments lose their semantic anchoring.

### 6.2 Architecture

Late chunking inverts the pipeline: **embed first, chunk second**.

**Step 1 — Full-Document Encoding:** Pass the entire document $d = (t_1, \ldots, t_n)$ through a long-context embedding model (e.g., a model supporting 8192+ tokens) to produce **token-level contextualized embeddings**:

$$\mathbf{E} = \text{LongContextEmbed}(t_1, \ldots, t_n) = (\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n)$$

Each $\mathbf{e}_i \in \mathbb{R}^d$ encodes token $t_i$ **in the context of the entire document** via self-attention over all $n$ tokens.

**Step 2 — Determine Chunk Boundaries:** Apply any chunking strategy (fixed, recursive, semantic, structural) to determine boundary indices $\mathcal{B} = \{b_0=1, b_1, b_2, \ldots, b_k=n\}$.

**Step 3 — Pool Token Embeddings Per Chunk:** For chunk $c_j$ spanning tokens $[b_{j-1}, b_j)$:

$$\phi_{\text{late}}(c_j) = \text{Pool}(\mathbf{e}_{b_{j-1}}, \mathbf{e}_{b_{j-1}+1}, \ldots, \mathbf{e}_{b_j - 1})$$

Common pooling strategies:
- **Mean pooling:** $\phi_{\text{late}}(c_j) = \frac{1}{b_j - b_{j-1}} \sum_{i=b_{j-1}}^{b_j - 1} \mathbf{e}_i$
- **Attention-weighted pooling:** $\phi_{\text{late}}(c_j) = \sum_i \alpha_i \mathbf{e}_i$ where $\alpha_i = \text{softmax}(\mathbf{w}^\top \mathbf{e}_i)$
- **CLS-like sentinel pooling:** Insert sentinel tokens at chunk boundaries during encoding

```
ALGORITHM: LATE_CHUNKING
────────────────────────────────────────────────────
Input:  document d, long_context_model M, chunking_fn C,
        pooling_fn POOL
Output: chunks[] with context-aware embeddings

1.  tokens ← TOKENIZE(d, M.tokenizer)
2.  IF |tokens| > M.max_context:
3.      // Segment with overlap for boundary continuity
4.      segments ← SLIDING_WINDOW(tokens, M.max_context, stride=M.max_context * 0.75)
5.      token_embeddings ← []
6.      FOR EACH seg IN segments:
7.          seg_embeddings ← M.encode(seg)  // returns per-token embeddings
8.          token_embeddings.EXTEND(seg_embeddings[unique portion])
9.      ENDFOR
10. ELSE:
11.     token_embeddings ← M.encode(tokens)
12. ENDIF

13. // Determine chunk boundaries on raw text
14. text_chunks ← C(d)
15. boundaries ← ALIGN_CHUNKS_TO_TOKEN_INDICES(text_chunks, tokens)

16. // Pool token embeddings per chunk
17. chunks_with_embeddings ← []
18. FOR j ← 0 TO |boundaries|-2:
19.     start ← boundaries[j]
20.     end ← boundaries[j+1]
21.     chunk_embedding ← POOL(token_embeddings[start..end-1])
22.     chunk_record ← ChunkRecord {
23.         content: text_chunks[j],
24.         embedding: chunk_embedding,
25.         embedding_method: "LATE_CHUNKING",
26.         context_window: |tokens|,
27.         token_span: (start, end)
28.     }
29.     chunks_with_embeddings.APPEND(chunk_record)
30. ENDFOR

31. RETURN chunks_with_embeddings
────────────────────────────────────────────────────
```

### 6.3 Why Late Chunking Produces Superior Embeddings

Consider a document containing:

> *"The BERT model was introduced in 2018. **It** achieved state-of-the-art results on eleven NLP benchmarks."*

With standard chunking at the sentence boundary:
- Chunk 1: "The BERT model was introduced in 2018." → $\phi$ captures {BERT, 2018, introduced}
- Chunk 2: "It achieved state-of-the-art results on eleven NLP benchmarks." → $\phi$ captures {it, SOTA, NLP} — **"it" is unresolved**

With late chunking:
- Token embedding for "It" in sentence 2 has attended to "BERT model" in sentence 1 during encoding
- $\phi_{\text{late}}(\text{chunk 2})$ inherently encodes the BERT reference

**Formal Property:** Let $\text{MutualInfo}(c_j; d \setminus c_j)$ be the mutual information between a chunk and the rest of the document. Late chunking satisfies:

$$I(\phi_{\text{late}}(c_j); d \setminus c_j) > I(\phi_{\text{standard}}(c_j); d \setminus c_j)$$

because the self-attention mechanism in the encoder propagates information from the full document into each token's representation.

### 6.4 Cost and Latency Trade-Off

| Property | Standard Embed-per-Chunk | Late Chunking |
|---|---|---|
| Embedding calls | $k$ (one per chunk) | 1 (or $\lceil n / W_{\text{model}} \rceil$) |
| Total tokens embedded | $\sum |c_i|$ | $n$ (document length) |
| Model requirement | Any embedding model | Long-context embedding model |
| Re-chunking cost | Re-embed all chunks | Re-pool only (embeddings cached) |
| Contextual quality | Independent per chunk | Full-document awareness |

**Key Advantage for Re-chunking:** If the chunking strategy changes (e.g., tuning chunk size), standard approaches require re-embedding the entire corpus. Late chunking only requires re-pooling from cached token embeddings—a CPU-bound operation orders of magnitude cheaper than re-encoding.

---

## 7. Pre-Chunking vs. Post-Chunking: Architectural Trade-Off Analysis

### 7.1 Pre-Chunking Architecture

```
┌──────────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│ Raw Documents│───→│ Chunker  │───→│ Embed & Index│───→│ Vector DB │
└──────────────┘    └──────────┘    └──────────────┘    └─────┬─────┘
                                                              │
                    ┌──────────┐    ┌──────────────┐          │
                    │  Query   │───→│ Embed Query  │───→ ANN Search
                    └──────────┘    └──────────────┘    ┌─────┴─────┐
                                                        │ Ranked    │
                                                        │ Chunks    │
                                                        └───────────┘
```

**Formal Latency Model:**

$$T_{\text{pre-chunk}}^{\text{query}} = T_{\text{embed}}(q) + T_{\text{ANN}}(k) + T_{\text{network}}$$

where:
- $T_{\text{embed}}(q) \approx 10\text{–}50\text{ms}$ (query embedding)
- $T_{\text{ANN}}(k) \approx 1\text{–}10\text{ms}$ (HNSW approximate nearest neighbor)
- $T_{\text{network}} \approx 1\text{–}5\text{ms}$

**Total query-time latency: 12–65ms** — dominated by embedding.

**Offline ingestion cost:**

$$T_{\text{pre-chunk}}^{\text{ingest}}(d) = T_{\text{chunk}}(d) + |\mathcal{C}(d)| \cdot T_{\text{embed}}(c) + |\mathcal{C}(d)| \cdot T_{\text{index}}(c)$$

### 7.2 Post-Chunking Architecture

```
┌──────────────┐    ┌──────────────┐
│ Raw Documents│───→│ Document     │  (store whole documents)
└──────────────┘    │ Store        │
                    └──────┬───────┘
                           │
┌──────────┐    ┌──────────┴───────┐    ┌──────────────┐
│  Query   │───→│ Document-Level   │───→│ Retrieve Top │
└──────────┘    │ Retrieval (BM25  │    │ Documents    │
                │ or summary embed)│    └──────┬───────┘
                └──────────────────┘           │
                    ┌──────────────────────────┴───────┐
                    │ REAL-TIME: Chunk Retrieved Docs   │
                    │ based on query context            │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────┴───────┐
                    │ Re-rank Chunks   │
                    │ against Query    │
                    └──────────────────┘
```

**Formal Latency Model:**

$$T_{\text{post-chunk}}^{\text{query}} = T_{\text{doc-retrieval}} + \sum_{d \in \text{Top-}K_{\text{docs}}} \left[ T_{\text{chunk}}(d) + |\mathcal{C}(d)| \cdot T_{\text{embed}}(c) \right] + T_{\text{rerank}}$$

This adds $100\text{–}2000\text{ms}$ depending on document count and chunking method.

### 7.3 Comparative Analysis

| Dimension | Pre-Chunking | Post-Chunking |
|---|---|---|
| **Query latency** | $O(1)$ — index lookup | $O(K \cdot n_{\text{doc}})$ — real-time processing |
| **Strategy flexibility** | Fixed at ingestion; change requires full re-index | Query-adaptive; different strategies per query type |
| **Index storage** | $O(|\mathcal{C}|)$ chunk embeddings | $O(N)$ document-level index (smaller) |
| **Freshness** | Requires re-ingestion pipeline | Documents can be updated atomically |
| **Infrastructure complexity** | Standard vector DB | Requires real-time chunking + embedding infra |
| **Optimal use case** | High-QPS, latency-sensitive applications | Low-QPS, high-precision, complex documents |

### 7.4 Hybrid Architecture (SOTA Recommendation)

Production systems should implement a **tiered approach**:

```
ALGORITHM: HYBRID_CHUNK_RETRIEVAL
────────────────────────────────────────────────────
Input:  query q, pre_chunked_index PI, document_store DS,
        latency_budget T_max

1.  // Tier 1: Fast pre-chunked retrieval (always executes)
2.  pre_results ← ANN_SEARCH(PI, φ(q), k=20)
3.  IF CONFIDENCE(pre_results[0]) > θ_high AND T_elapsed < T_max * 0.5:
4.      RETURN RERANK(pre_results, q, k=5)

5.  // Tier 2: If confidence insufficient, augment with post-chunking
6.  candidate_docs ← DOCUMENT_LEVEL_SEARCH(DS, q, k=3)
7.  dynamic_chunks ← []
8.  FOR EACH doc IN candidate_docs:
9.      strategy ← SELECT_STRATEGY(q, doc)  // query-adaptive
10.     dynamic_chunks.EXTEND(strategy.chunk(doc))
11. ENDFOR
12. dynamic_embeddings ← BATCH_EMBED(dynamic_chunks)

13. // Tier 3: Merge and re-rank
14. all_candidates ← pre_results ∪ SCORE(dynamic_chunks, dynamic_embeddings, q)
15. final ← CROSS_ENCODER_RERANK(all_candidates, q, k=5)
16. RETURN final
────────────────────────────────────────────────────
```

---

## 8. Chunking Strategy Selection: Decision Framework

### 8.1 Automated Strategy Selection Model

The strategy selection problem can be formalized as a classification task over document features:

$$\text{Strategy}^* = f(\mathbf{x}_d)$$

where the feature vector $\mathbf{x}_d$ includes:

| Feature | Type | Description |
|---|---|---|
| `structural_depth` | int | Maximum nesting depth of headings/sections |
| `avg_section_tokens` | float | Mean section length |
| `has_tables` | bool | Presence of tabular data |
| `has_code` | bool | Presence of source code |
| `format` | enum | Markdown, HTML, PDF, plaintext |
| `domain_complexity` | float | Estimated domain-specific terminology density |
| `topic_density` | float | Estimated topics per 1000 tokens |
| `document_length` | int | Total tokens |
| `query_type` | enum | Factoid, analytical, procedural, comparative |

### 8.2 Decision Matrix with Quantitative Boundaries

```
ALGORITHM: STRATEGY_SELECTOR
────────────────────────────────────────────────────
Input:  document_features x_d, query_distribution Q_dist
Output: ChunkingStrategy with parameters

1.  IF x_d.document_length < 500:
2.      RETURN WHOLE_DOCUMENT  // no chunking needed

3.  IF x_d.format IN {CODE}:
4.      RETURN AST_STRUCTURAL(language=x_d.language)

5.  IF x_d.structural_depth ≥ 3 AND x_d.document_length > 5000:
6.      RETURN HIERARCHICAL(
7.          levels=[DOCUMENT_SUMMARY, SECTION_SUMMARY, PARAGRAPH],
8.          summarizer=LLM_SUMMARIZE,
9.          leaf_strategy=SEMANTIC
10.     )

11. IF x_d.format IN {MARKDOWN, HTML} AND x_d.structural_depth ≥ 2:
12.     RETURN STRUCTURAL(
13.         split_on=HEADINGS,
14.         fallback=RECURSIVE(L_max=512)
15.     )

16. IF x_d.topic_density > 3.0:  // high topic variation
17.     RETURN SEMANTIC(
18.         θ=ADAPTIVE_PERCENTILE(25),
19.         L_max=512, L_min=100
20.     )

21. IF x_d.domain_complexity > 0.7 AND BUDGET_ALLOWS_LLM:
22.     RETURN LLM_BASED(
23.         proposition_mode=TRUE,
24.         window=4096, stride=3072
25.     )

26. // Default: robust general-purpose
27. RETURN RECURSIVE(
28.     separators=["\n\n", "\n", ". ", " "],
29.     L_max=512,
30.     overlap=50
31. )
────────────────────────────────────────────────────
```

### 8.3 Strategy Comparison: Quantitative Performance Characteristics

| Strategy | Retrieval Precision (relative) | Contextual Richness (relative) | Compute Cost | Latency | Robustness to Format Variation |
|---|---|---|---|---|---|
| Fixed-Size | 0.6 | 0.5 | $O(n)$ | ≤ 1ms/doc | High (format-agnostic) |
| Recursive | 0.7 | 0.7 | $O(n)$ | ≤ 2ms/doc | High |
| Structural | 0.8 | 0.8 | $O(n)$ | ≤ 5ms/doc | Medium (requires parseable structure) |
| Semantic | 0.85 | 0.85 | $O(n \cdot d_{\text{embed}})$ | 50–200ms/doc | High |
| LLM-Based | 0.9 | 0.9 | $O(n \cdot C_{\text{LLM}})$ | 2–10s/doc | High |
| Agentic | 0.92 | 0.92 | $O(n \cdot C_{\text{LLM}} \cdot r)$ | 5–30s/doc | Highest |
| Late Chunking | 0.88 | 0.9 | $O(n \cdot d_{\text{embed}})$ | 100–500ms/doc | High |
| Hierarchical | 0.85 | 0.93 | $O(n \cdot L)$ | Varies | Medium |

*Values are relative within a controlled benchmark; absolute numbers depend on corpus, query distribution, and embedding model.*

---

## 9. End-to-End Retrieval Pipeline: Production Architecture

### 9.1 Full System Architecture

```
                                 ┌─────────────────────────────┐
                                 │     APPLICATION BOUNDARY     │
                                 │       (JSON-RPC v2.0)        │
                                 └────────────┬────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         │          QUERY PROCESSING LAYER          │
                         │                                          │
                         │  ┌────────────┐  ┌───────────────────┐  │
                         │  │Query       │  │Subquery           │  │
                         │  │Rewriter    │→ │Decomposer &       │  │
                         │  │(expansion, │  │Router             │  │
                         │  │ NER, coref)│  │(schema-aware)     │  │
                         │  └────────────┘  └─────────┬─────────┘  │
                         └────────────────────────────┼────────────┘
                                                      │
                    ┌─────────────────────────────────┼──────────────────┐
                    │           RETRIEVAL LAYER (gRPC/Protobuf)          │
                    │                                                     │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
                    │  │BM25/     │ │Dense     │ │Metadata  │ │Graph  │ │
                    │  │Keyword   │ │Vector    │ │Filter    │ │Lineage│ │
                    │  │Index     │ │ANN Index │ │Index     │ │Store  │ │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬───┘ │
                    │       └──────┬─────┘────────────┘───────────┘     │
                    │              │                                      │
                    │       ┌──────┴──────────┐                          │
                    │       │ Reciprocal Rank  │                          │
                    │       │ Fusion (RRF)     │                          │
                    │       └──────┬──────────┘                          │
                    │              │                                      │
                    │       ┌──────┴──────────┐                          │
                    │       │ Cross-Encoder   │                          │
                    │       │ Re-Ranker       │                          │
                    │       └──────┬──────────┘                          │
                    │              │                                      │
                    │       ┌──────┴──────────┐                          │
                    │       │ Budget Packer   │                          │
                    │       │ (Token Knapsack)│                          │
                    │       └──────┬──────────┘                          │
                    └──────────────┼──────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   CONTEXT ASSEMBLY LAYER     │
                    │   (Prefill Compiler)          │
                    │                               │
                    │   [Role Policy]               │
                    │   [Task Objective]            │
                    │   [Retrieved Evidence]        │  → LLM
                    │   [Memory Summaries]          │
                    │   [Tool Affordances]          │
                    └──────────────────────────────┘
```

### 9.2 Hybrid Retrieval with Reciprocal Rank Fusion

Given $M$ retrieval subsystems, each returning a ranked list $R_m$ for query $q$, **Reciprocal Rank Fusion (RRF)** computes:

$$\text{RRF}(c) = \sum_{m=1}^{M} \frac{1}{k_{\text{rrf}} + \text{rank}_m(c)}$$

where $k_{\text{rrf}} = 60$ (standard constant that dampens the influence of high-rank positions).

```
ALGORITHM: RECIPROCAL_RANK_FUSION
────────────────────────────────────────────────────
Input:  ranked_lists R[1..M] (each a list of (chunk_id, score)),
        k_rrf = 60
Output: fused_ranking (list of (chunk_id, rrf_score))

1.  scores ← {}  // map: chunk_id → rrf_score
2.  FOR m ← 1 TO M:
3.      FOR rank, (chunk_id, _) IN ENUMERATE(R[m], start=1):
4.          scores[chunk_id] ← scores.GET(chunk_id, 0) + 1/(k_rrf + rank)
5.  ENDFOR
6.  fused_ranking ← SORT_DESC(scores.items(), key=rrf_score)
7.  RETURN fused_ranking
────────────────────────────────────────────────────
```

**Why RRF over learned fusion:** RRF is **model-free**, requires no training data, is robust to score distribution differences across retrieval methods (BM25 scores are unbounded; cosine similarity is in $[-1, 1]$), and adds zero latency.

### 9.3 Cross-Encoder Re-Ranking

After RRF, apply a cross-encoder (e.g., a fine-tuned BERT model that jointly encodes $(q, c)$) to re-score the top $n_{\text{rerank}}$ candidates:

$$\text{score}_{\text{CE}}(q, c) = \text{CrossEncoder}(\text{concat}(q, c))$$

This is computationally expensive ($O(n_{\text{rerank}} \cdot L^2)$ for transformer self-attention) but dramatically improves precision by allowing token-level interaction between query and chunk.

**Production Constraint:** $n_{\text{rerank}} \leq 50$ to stay within latency budgets. Beyond 50 candidates, the marginal precision gain is negligible.

### 9.4 Token Budget Packing (Greedy Knapsack)

After re-ranking, select chunks to fill the retrieval token budget $B_{\text{retrieval}}$:

$$\text{maximize} \sum_{i} \text{score}(c_i) \cdot x_i \quad \text{s.t.} \sum_{i} |c_i| \cdot x_i \leq B_{\text{retrieval}}, \; x_i \in \{0, 1\}$$

For ranked chunks (items already sorted by score), the **greedy algorithm** is optimal in practice:

```
ALGORITHM: BUDGET_PACK
────────────────────────────────────────────────────
Input:  ranked_chunks C (sorted by score desc), budget B
Output: selected_chunks S

1.  S ← []
2.  remaining ← B
3.  FOR EACH c IN C:
4.      IF |c.tokens| ≤ remaining:
5.          S.APPEND(c)
6.          remaining ← remaining - |c.tokens|
7.      IF remaining < MIN_CHUNK_SIZE:
8.          BREAK
9.  ENDFOR
10. RETURN S
────────────────────────────────────────────────────
```

---

## 10. Evaluation Infrastructure & Quality Gates

### 10.1 Retrieval Quality Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Recall@k** | $\frac{|\text{Retrieved}@k \cap \text{Gold}|}{|\text{Gold}|}$ | Are relevant chunks surfaced? |
| **Precision@k** | $\frac{|\text{Retrieved}@k \cap \text{Gold}|}{k}$ | Are irrelevant chunks excluded? |
| **MRR** | $\frac{1}{|Q|}\sum_{q} \frac{1}{\text{rank}_q(\text{first relevant})}$ | How early is the first relevant result? |
| **NDCG@k** | $\frac{\text{DCG}@k}{\text{IDCG}@k}$ where $\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i+1)}$ | Graded relevance quality |
| **Context Precision** | $\frac{\text{tokens of relevant info in context}}{\text{total tokens in context}}$ | Token efficiency of retrieved context |
| **Answer Correctness** | End-to-end accuracy of LLM response using retrieved context | True system-level metric |

### 10.2 Chunking-Specific Evaluation

```
ALGORITHM: CHUNKING_EVAL_SUITE
────────────────────────────────────────────────────
Input:  chunking_fn C, eval_corpus D_eval, eval_queries Q_eval,
        gold_annotations G
Output: quality_report

1.  // Metric 1: Chunk Boundary Alignment
2.  FOR EACH d IN D_eval:
3.      chunks ← C(d)
4.      gold_boundaries ← G.boundaries(d)
5.      boundary_f1 ← F1(predicted=BOUNDARIES(chunks),
6.                        gold=gold_boundaries, tolerance=2_sentences)
7.  ENDFOR

8.  // Metric 2: Topical Purity
9.  FOR EACH chunk IN ALL_CHUNKS:
10.     sentences ← SPLIT_SENTENCES(chunk)
11.     embeddings ← [φ(s) FOR s IN sentences]
12.     purity ← MEAN_PAIRWISE_COSINE(embeddings)
13. ENDFOR

14. // Metric 3: Retrieval Precision/Recall under this chunking
15. index ← BUILD_INDEX(ALL_CHUNKS)
16. FOR EACH q IN Q_eval:
17.     retrieved ← SEARCH(index, q, k=5)
18.     recall_q ← |retrieved ∩ G.gold_chunks(q)| / |G.gold_chunks(q)|
19.     precision_q ← |retrieved ∩ G.gold_chunks(q)| / 5
20. ENDFOR

21. // Metric 4: End-to-End Answer Quality
22. FOR EACH q IN Q_eval:
23.     context ← RETRIEVE_AND_PACK(index, q)
24.     answer ← LLM(q, context)
25.     correctness_q ← JUDGE(answer, G.gold_answer(q))  // LLM-as-judge or human
26. ENDFOR

27. RETURN QualityReport {
28.     boundary_f1: MEAN(boundary_f1),
29.     topical_purity: MEAN(purity),
30.     recall_at_5: MEAN(recall_q),
31.     precision_at_5: MEAN(precision_q),
32.     answer_correctness: MEAN(correctness_q)
33. }
────────────────────────────────────────────────────
```

### 10.3 CI/CD Integration

Every chunking strategy change triggers:

1. **Regression test:** Run `CHUNKING_EVAL_SUITE` against the golden evaluation set
2. **Quality gate:** `answer_correctness ≥ τ_correctness` AND `recall@5 ≥ τ_recall`
3. **Cost gate:** `ingestion_cost_per_doc ≤ budget_max`
4. **Latency gate:** `p99_query_latency ≤ T_max`

If any gate fails, the deployment is blocked and the change is flagged for review.

---

## 11. Operational Considerations: Reliability, Cost, Latency

### 11.1 Hallucination Control via Provenance

Every chunk in the LLM context carries provenance metadata. The system prompt instructs the LLM:

```
You MUST cite retrieved evidence by chunk_id when making factual claims.
If no retrieved chunk supports a claim, state "I don't have sufficient evidence."
Never synthesize facts not present in the provided evidence chunks.
```

The verification layer post-generation cross-references cited chunk IDs against the actual retrieved set, flagging any fabricated citations.

### 11.2 Fault Tolerance & Graceful Degradation

| Failure Mode | Detection | Recovery |
|---|---|---|
| Embedding service unavailable | Health check + circuit breaker | Fall back to BM25-only retrieval |
| Vector DB timeout | Deadline propagation (gRPC deadline) | Return cached results or degrade to keyword search |
| Cross-encoder timeout | Latency budget exhaustion | Skip re-ranking, return RRF results directly |
| Chunking pipeline failure | Schema validation on chunk output | Fall back to fixed-size chunking |
| LLM-based chunking cost spike | Cost monitoring per invocation | Circuit-break to semantic chunking |

### 11.3 Cost Optimization Model

$$\text{Cost}_{\text{total}} = \text{Cost}_{\text{ingest}} + \text{Cost}_{\text{storage}} + \text{Cost}_{\text{query}} \cdot Q_{\text{volume}}$$

where:

$$\text{Cost}_{\text{ingest}} = \sum_{d \in \mathcal{D}} \left[ C_{\text{chunk}}(d) + |\mathcal{C}(d)| \cdot C_{\text{embed}} + |\mathcal{C}(d)| \cdot C_{\text{index}} \right]$$

$$\text{Cost}_{\text{query}} = C_{\text{embed}}(q) + C_{\text{ANN}} + C_{\text{rerank}} \cdot n_{\text{rerank}}$$

**Optimization Levers:**

| Lever | Action | Impact |
|---|---|---|
| Embedding model selection | Use a smaller model (e.g., 384-dim vs 1024-dim) | 2–4x reduction in $C_{\text{embed}}$ and $C_{\text{storage}}$ |
| Chunk deduplication | Hash-based dedup across documents | 10–30% reduction in index size |
| Tiered storage | Hot (HNSW in memory) / Warm (disk-based) / Cold (archive) | 5–10x storage cost reduction |
| Lazy re-indexing | Only re-chunk/re-embed changed documents | Proportional to change rate |
| Cache query embeddings | LRU cache for frequent queries | Eliminates repeated embedding calls |

### 11.4 Observability Contract

Every retrieval request emits a structured trace:

```protobuf
message RetrievalTrace {
  string trace_id = 1;
  string query_text = 2;
  repeated SubqueryTrace subqueries = 3;
  repeated RetrievalResult candidates = 4;
  repeated RetrievalResult reranked = 5;
  repeated RetrievalResult selected = 6;
  int32 total_tokens_retrieved = 7;
  int32 budget_remaining = 8;
  Duration total_latency = 9;
  Duration embed_latency = 10;
  Duration search_latency = 11;
  Duration rerank_latency = 12;
  map<string, float> quality_scores = 13;
}
```

This enables:
- **Latency profiling** per sub-component
- **Retrieval quality dashboards** (daily recall/precision trends)
- **Drift detection** (embedding distribution shift over time)
- **Cost attribution** per query type

---

## Summary: The Two Fundamental Decisions

The effectiveness of a retrieval system reduces to two engineering decisions that are jointly optimized:

### Decision 1: The Chunking Strategy (The "How")

Determines the **granularity and semantic coherence** of retrieval units. Formally, it defines the mapping $\mathcal{C}: \mathcal{D} \rightarrow 2^{\mathcal{U}}$ from documents to chunk sets, optimizing the joint objective:

$$\max_{\mathcal{C}} \; \mathbb{E}_q \left[ \mathcal{P}(\mathcal{C}, q) \cdot \mathcal{R}(\mathcal{C}, q) \right]$$

### Decision 2: The Architectural Pattern (The "When")

Determines the **trade-off between query-time latency and strategy flexibility**. Pre-chunking minimizes $T_{\text{query}}$ at the cost of strategy rigidity; post-chunking maximizes adaptability at the cost of latency; hybrid tiered approaches achieve production-optimal Pareto frontiers.

Together, these decisions form the **retrieval contract** that governs the upper bound on any downstream LLM's ability to produce correct, grounded, and contextually complete responses. A retrieval system engineered with typed provenance, formal quality gates, hybrid multi-signal ranking, and continuous evaluation infrastructure transforms an LLM from a stochastic text generator into a **deterministic, evidence-grounded reasoning engine** operating within auditable bounds.

---

*This report specifies chunking and retrieval as first-class system-engineering concerns with typed contracts, formal optimization objectives, production-grade algorithms, and measurable quality gates—the foundation upon which reliable agentic AI systems are constructed.*