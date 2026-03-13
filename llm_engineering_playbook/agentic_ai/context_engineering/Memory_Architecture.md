# Memory Architecture for Agentic AI Systems: A Principal-Level Technical Report

## Formal Theory, Mathematical Foundations, Typed Protocols, and SOTA Algorithms for Production-Grade Agent Memory

---

## 1. Foundational Formalism: Memory as a Typed, Layered Control System

### 1.1 The Computational Model of Agent Memory

An agentic system operates under a fundamental constraint: the context window $W$ of size $|W| = N_{\max}$ tokens functions as a **bounded working register**, analogous to CPU registers plus L1 cache, not as unbounded RAM. Every token consumed by stale history, irrelevant retrieval, or uncompressed state is a token **denied** to reasoning, planning, and synthesis. Memory architecture is therefore a **resource-allocation optimization problem** under a hard token budget.

**Definition 1 (Agent Memory System).** An agent memory system is a tuple:

$$\mathcal{M} = \langle W, S, E, \Sigma, P, \phi_{\text{admit}}, \phi_{\text{evict}}, \phi_{\text{retrieve}}, \phi_{\text{promote}}, \phi_{\text{consolidate}} \rangle$$

where:

| Symbol | Type | Description |
|--------|------|-------------|
| $W$ | `BoundedBuffer<Token>` | Working memory (active context window) |
| $S$ | `SessionStore<MemoryItem>` | Session-scoped ephemeral memory |
| $E$ | `DurableStore<EpisodicRecord>` | Validated episodic memory (interaction traces) |
| $\Sigma$ | `DurableStore<SemanticFact>` | Canonical semantic/organizational knowledge |
| $P$ | `DurableStore<Procedure>` | Procedural memory (learned workflows) |
| $\phi_{\text{admit}}$ | `AdmissionPolicy` | Gate function controlling writes |
| $\phi_{\text{evict}}$ | `EvictionPolicy` | Eviction/pruning control |
| $\phi_{\text{retrieve}}$ | `RetrievalEngine` | Multi-signal ranked retrieval |
| $\phi_{\text{promote}}$ | `PromotionPolicy` | Layer-crossing promotion logic |
| $\phi_{\text{consolidate}}$ | `ConsolidationEngine` | Merge, deduplicate, compress |

**Definition 2 (Memory Item).** Every item stored in any layer is a provenance-tagged record:

$$m = \langle \text{id}, \text{content}, \text{embedding}, \text{source}, \text{timestamp}, \text{ttl}, \text{importance}, \text{access\_count}, \text{lineage}, \text{version}, \text{schema\_type} \rangle$$

```protobuf
message MemoryItem {
  string id = 1;
  string content = 2;
  repeated float embedding = 3;
  Provenance source = 4;
  google.protobuf.Timestamp created_at = 5;
  google.protobuf.Duration ttl = 6;
  float importance_score = 7;
  uint64 access_count = 8;
  repeated string lineage_ids = 9;
  uint32 version = 10;
  MemorySchemaType schema_type = 11;
  
  enum MemorySchemaType {
    EPISODIC = 0;
    SEMANTIC = 1;
    PROCEDURAL = 2;
    WORKING = 3;
    SESSION = 4;
  }
}

message Provenance {
  string origin_agent_id = 1;
  string task_id = 2;
  string tool_invocation_id = 3;
  string human_annotator_id = 4;
  float confidence = 5;
  ValidationStatus validation = 6;
}
```

### 1.2 The Token Budget Constraint

At every agent step $t$, the working memory $W_t$ must satisfy:

$$|W_t| = |\text{RolePolicy}| + |\text{TaskState}_t| + |\text{RetrievedEvidence}_t| + |\text{ToolAffordances}_t| + |\text{MemorySummary}_t| + |\text{History}_t| \leq N_{\max} - R_{\text{reserved}}$$

where $R_{\text{reserved}}$ is a **hard reservation** for generation output and chain-of-thought reasoning. Violating this constraint degrades generation quality non-linearly:

$$Q(\text{generation}) \propto \begin{cases} 1 & \text{if } |W_t| \leq N_{\max} - R_{\text{reserved}} \\ e^{-\lambda(|W_t| - (N_{\max} - R_{\text{reserved}}))} & \text{otherwise} \end{cases}$$

where $\lambda > 0$ is a model-specific degradation coefficient empirically measured via eval suites. The objective function for context construction at each step is:

$$\max_{C \subseteq \mathcal{C}_{\text{candidates}}} \sum_{c \in C} u(c, \text{task}_t) \quad \text{subject to} \quad \sum_{c \in C} \text{tokens}(c) \leq B_t$$

where $u(c, \text{task}_t)$ is the **task-conditioned utility** of context item $c$, and $B_t = N_{\max} - R_{\text{reserved}} - |\text{FixedPrefill}|$ is the available budget at step $t$.

> **This is a variant of the 0/1 Knapsack Problem** and is NP-hard in general. SOTA systems approximate it via greedy utility-density ranking or learned value functions.

---

## 2. Working Memory: The Active Reasoning Register

### 2.1 Formal Specification

Working memory $W$ is the **only memory layer visible to the model at inference time**. It is compiled—not appended—at each agent step by the **Prefill Compiler**.

**Definition 3 (Prefill Compiler).** A deterministic function:

$$\text{Compile}: (\text{RolePolicy}, \text{TaskState}_t, \text{Retrieved}_t, \text{Tools}_t, \text{MemSummary}_t, \text{History}_t) \rightarrow W_t$$

subject to $|W_t| \leq B_t$.

### 2.2 SOTA Context Compression: Hierarchical Summarization with Importance Weighting

Rather than naive truncation (dropping the oldest $k$ messages), SOTA systems apply **hierarchical importance-weighted summarization**.

**Definition 4 (Importance-Weighted Message Retention).** For a message history $H = [h_1, h_2, \ldots, h_n]$, assign each message an importance score:

$$I(h_i) = \alpha \cdot \text{Recency}(h_i) + \beta \cdot \text{TaskRelevance}(h_i, \text{task}_t) + \gamma \cdot \text{InformationDensity}(h_i) + \delta \cdot \text{ReferenceCount}(h_i)$$

where:

$$\text{Recency}(h_i) = e^{-\mu(t - t_i)}$$

$$\text{TaskRelevance}(h_i, \text{task}_t) = \cos(\mathbf{e}_{h_i}, \mathbf{e}_{\text{task}_t})$$

$$\text{InformationDensity}(h_i) = \frac{|\text{NamedEntities}(h_i)| + |\text{ToolCalls}(h_i)| + |\text{Decisions}(h_i)|}{|\text{tokens}(h_i)|}$$

$$\text{ReferenceCount}(h_i) = \sum_{j > i} \mathbb{1}[h_j \text{ references } h_i]$$

and $\alpha + \beta + \gamma + \delta = 1$ are tunable weights.

### 2.3 Algorithm: Adaptive Context Window Management

```
ALGORITHM 1: AdaptiveContextWindowManager
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  H[1..n]          — full message history
  task_t            — current task descriptor
  B_t               — available token budget
  R_reserved        — reserved reasoning tokens
  fixed_prefill     — compiled role policy + tool affordances

Output:
  W_t               — compiled working memory (context window payload)

PROCEDURE:
  1.  budget_remaining ← B_t - tokens(fixed_prefill)
  
  2.  // PHASE 1: Score all history items
      FOR each h_i IN H[1..n]:
          I(h_i) ← α·Recency(h_i) + β·TaskRelevance(h_i, task_t)
                    + γ·InformationDensity(h_i) + δ·ReferenceCount(h_i)
      END FOR
  
  3.  // PHASE 2: Partition into tiers
      tier_critical ← {h_i : I(h_i) ≥ θ_critical}      // Must retain verbatim
      tier_important ← {h_i : θ_important ≤ I(h_i) < θ_critical}  // Retain or summarize
      tier_background ← {h_i : I(h_i) < θ_important}    // Summarize or discard
  
  4.  // PHASE 3: Greedy knapsack packing
      W_t ← fixed_prefill
      
      // Pack critical items first (verbatim)
      FOR each h_i IN tier_critical SORTED BY I(h_i) DESC:
          IF tokens(h_i) ≤ budget_remaining:
              APPEND h_i TO W_t
              budget_remaining -= tokens(h_i)
          ELSE:
              summary_i ← CompressMessage(h_i, target_ratio=0.3)
              IF tokens(summary_i) ≤ budget_remaining:
                  APPEND summary_i TO W_t
                  budget_remaining -= tokens(summary_i)
      END FOR
      
      // Pack important items (summarize if needed)
      FOR each h_i IN tier_important SORTED BY I(h_i) DESC:
          IF tokens(h_i) ≤ budget_remaining:
              APPEND h_i TO W_t
              budget_remaining -= tokens(h_i)
          ELSE IF budget_remaining > MIN_SUMMARY_TOKENS:
              summary_i ← CompressMessage(h_i, target_ratio=0.2)
              APPEND summary_i TO W_t
              budget_remaining -= tokens(summary_i)
      END FOR
      
      // Pack background as batch summary
      IF |tier_background| > 0 AND budget_remaining > MIN_BATCH_SUMMARY:
          batch_summary ← BatchSummarize(tier_background, max_tokens=budget_remaining)
          APPEND batch_summary TO W_t
      END IF
  
  5.  // PHASE 4: Inject retrieved evidence and memory summaries
      retrieved ← φ_retrieve(task_t, budget=budget_remaining × 0.6)
      mem_summary ← CompileMemorySummary(S, E, Σ, budget=budget_remaining × 0.4)
      APPEND retrieved TO W_t
      APPEND mem_summary TO W_t
  
  6.  ASSERT tokens(W_t) ≤ N_max - R_reserved
      RETURN W_t
```

### 2.4 Compression Functions: SOTA Approaches

**Extractive Compression** selects salient sentences. **Abstractive Compression** generates condensed summaries. The SOTA approach is **LLM-as-Compressor with Fidelity Verification**:

$$\text{CompressMessage}(h, r) = \text{LLM}_{\text{compress}}\left(\text{prompt}_{\text{compress}}, h, \lfloor |h| \cdot r \rfloor\right)$$

with a fidelity verification step:

$$\text{Fidelity}(h, \hat{h}) = \frac{|\text{Facts}(h) \cap \text{Facts}(\hat{h})|}{|\text{Facts}(h)|}$$

Only accept $\hat{h}$ if $\text{Fidelity}(h, \hat{h}) \geq \tau_{\text{fidelity}}$ (typically $\tau_{\text{fidelity}} \geq 0.95$).

---

## 3. Session Memory: Ephemeral Task-Scoped State

### 3.1 Formal Definition

Session memory $S$ persists for the duration of a **single user session or task execution** and is destroyed on session termination. It serves as the **scratchpad** for multi-step task state.

$$S = \{s_k : s_k = \langle \text{task\_id}, \text{key}, \text{value}, \text{step\_created}, \text{step\_last\_accessed} \rangle\}$$

### 3.2 Task State Storage and Recall Protocol

```
ALGORITHM 2: TaskStateScratchpad
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE TaskScratchpad:
    store: HashMap<TaskID, HashMap<String, TypedValue>>
    max_entries_per_task: uint = 256
    ttl_per_entry: Duration = session_duration

PROCEDURE OffloadToScratchpad(task_id, key, value, step_t):
    // Offload from working memory to session scratchpad
    1. VALIDATE Schema(value) against expected_type(key)
    2. IF store[task_id].size ≥ max_entries_per_task:
           EVICT entry with MIN(step_last_accessed) from store[task_id]
    3. store[task_id][key] ← {value, step_created=step_t, step_last_accessed=step_t}
    4. EMIT Trace("scratchpad.write", task_id, key, tokens(value))

PROCEDURE RecallFromScratchpad(task_id, keys[], step_t):
    // Recall specific keys back into working memory
    1. results ← []
    2. FOR each k IN keys:
           IF k IN store[task_id]:
               store[task_id][k].step_last_accessed ← step_t
               APPEND store[task_id][k].value TO results
           ELSE:
               APPEND NULL TO results
               EMIT Warning("scratchpad.miss", task_id, k)
    3. RETURN results

PROCEDURE FlushOnSessionEnd(task_id):
    // Evaluate each entry for promotion to episodic memory
    1. FOR each (key, entry) IN store[task_id]:
           candidate ← ConstructMemoryItem(entry)
           IF φ_admit(candidate) = ACCEPT:
               φ_promote(candidate, target=E)
    2. DELETE store[task_id]
    3. EMIT Trace("scratchpad.flush", task_id)
```

### 3.3 Mathematical Model of Scratchpad Utility

The utility of offloading item $x$ from $W$ to $S$ at step $t$ is:

$$U_{\text{offload}}(x, t) = \underbrace{\text{tokens}(x)}_{\text{freed capacity}} \times \underbrace{(1 - P(\text{need } x \text{ at step } t+1))}_{\text{probability of non-use next step}}$$

Offload when $U_{\text{offload}}(x, t) > \theta_{\text{offload}}$. The probability of need is estimated via:

$$P(\text{need } x \text{ at step } t+1) = \sigma\left(w_1 \cdot \text{TaskRelevance}(x) + w_2 \cdot \text{StepProximity}(x) + w_3 \cdot \text{DependencyCount}(x)\right)$$

where $\sigma$ is the sigmoid function and $w_i$ are learned or heuristic weights.

---

## 4. Episodic Memory: Validated Interaction Traces

### 4.1 Formal Specification

Episodic memory $E$ stores **specific, validated interaction episodes** with full provenance. Each episode is a structured trace of a completed agent interaction.

$$e = \langle \text{episode\_id}, \text{task\_type}, \text{input\_summary}, \text{action\_trace}, \text{outcome}, \text{reward\_signal}, \text{timestamp}, \text{provenance}, \text{embeddings} \rangle$$

```protobuf
message EpisodicRecord {
  string episode_id = 1;
  string task_type = 2;
  string input_summary = 3;
  repeated ActionStep action_trace = 4;
  Outcome outcome = 5;
  float reward_signal = 6;
  google.protobuf.Timestamp timestamp = 7;
  Provenance provenance = 8;
  repeated float summary_embedding = 9;
  repeated float outcome_embedding = 10;
  uint64 retrieval_count = 11;
  float decayed_importance = 12;
}

message ActionStep {
  uint32 step_index = 1;
  string action_type = 2;       // "tool_call", "reasoning", "retrieval", "user_interaction"
  string action_detail = 3;
  string observation = 4;
  float step_confidence = 5;
}

message Outcome {
  bool success = 1;
  string result_summary = 2;
  repeated string error_codes = 3;
  string correction_applied = 4;
}
```

### 4.2 Episodic Importance Scoring with Temporal Decay

The importance of an episodic memory decays over time but is reinforced by retrieval:

$$I_E(e, t) = I_0(e) \cdot e^{-\lambda_{\text{decay}}(t - t_e)} + \eta \sum_{r \in \text{retrievals}(e)} e^{-\lambda_{\text{decay}}(t - t_r)}$$

where:
- $I_0(e)$ is the initial importance score assigned at admission
- $\lambda_{\text{decay}}$ is the temporal decay rate
- $\eta$ is the reinforcement coefficient per retrieval event
- $t_r$ are timestamps of past retrievals of this episode

This follows the **Ebbinghaus forgetting curve** augmented with **spaced-retrieval reinforcement**, a principle from cognitive science that SOTA memory systems operationalize.

### 4.3 Algorithm: Episodic Memory Admission Controller

```
ALGORITHM 3: EpisodicAdmissionController
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  candidate       — candidate memory item from completed interaction
  E               — current episodic store
  config          — admission thresholds and policies

Output:
  ACCEPT or REJECT decision with reason

PROCEDURE AdmitToEpisodicMemory(candidate, E, config):

  // GATE 1: Schema Validation
  1. IF NOT ValidateSchema(candidate, EpisodicRecord.schema):
         RETURN REJECT("schema_violation")

  // GATE 2: Novelty Check (prevent redundant storage)
  2. nearest_neighbors ← SemanticSearch(E, candidate.summary_embedding, k=5)
     max_similarity ← MAX({cos(candidate.summary_embedding, nn.summary_embedding) 
                           : nn ∈ nearest_neighbors})
     IF max_similarity > config.dedup_threshold:  // typically 0.92
         // Check if candidate has novel outcome information
         outcome_sim ← cos(candidate.outcome_embedding, 
                           nearest_neighbors[0].outcome_embedding)
         IF outcome_sim > config.outcome_dedup_threshold:  // typically 0.88
             RETURN REJECT("duplicate_episode")
         ELSE:
             // Merge: update existing episode with new outcome data
             MergeEpisode(nearest_neighbors[0], candidate)
             RETURN ACCEPT("merged_with_existing")

  // GATE 3: Importance Scoring via LLM Reflection
  3. importance_prompt ← ConstructReflectionPrompt(candidate)
     reflection ← LLM_evaluate(importance_prompt)
     I_0 ← ParseImportanceScore(reflection)  // ∈ [0.0, 1.0]
     
     IF I_0 < config.min_importance_threshold:  // typically 0.3
         RETURN REJECT("low_importance")

  // GATE 4: Factual Consistency Verification
  4. IF candidate.outcome.success = TRUE:
         consistency ← VerifyFactualConsistency(candidate.action_trace, 
                                                  candidate.outcome)
         IF consistency < config.consistency_threshold:  // typically 0.85
             FLAG candidate AS "requires_human_review"

  // GATE 5: Provenance Completeness
  5. IF candidate.provenance.confidence < config.min_provenance_confidence:
         RETURN REJECT("insufficient_provenance")

  // COMMIT
  6. candidate.decayed_importance ← I_0
     candidate.retrieval_count ← 0
     WRITE candidate TO E with IDEMPOTENCY_KEY = candidate.episode_id
     EMIT Trace("episodic.admit", candidate.episode_id, I_0)
     RETURN ACCEPT("committed")
```

### 4.4 LLM Reflection for Importance Scoring

The importance score $I_0(e)$ is computed via a **structured LLM reflection call**, not a heuristic:

$$I_0(e) = \text{LLM}_{\text{judge}}\left(\text{prompt}_{\text{reflect}}, e\right) \rightarrow [0, 1]$$

The reflection prompt enforces a multi-dimensional rubric:

$$I_0(e) = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{score}_d(e)$$

where $\mathcal{D} = \{\text{novelty}, \text{correctability}, \text{generalizability}, \text{failure\_informativeness}, \text{user\_preference\_signal}\}$.

Each dimension is scored on $[0, 1]$. The composite score determines admission. **Only non-obvious corrections, constraint discoveries, and preference signals that would alter future behavior pass the gate.** Routine successful completions with no novel information are rejected to prevent memory bloat.

---

## 5. Semantic Memory: Canonical Organizational Knowledge

### 5.1 Formal Specification

Semantic memory $\Sigma$ stores **validated, general-purpose facts, rules, and domain knowledge** independent of specific interaction episodes.

$$\sigma = \langle \text{fact\_id}, \text{assertion}, \text{domain}, \text{confidence}, \text{source\_episodes}[], \text{contradicts}[], \text{last\_validated}, \text{embedding} \rangle$$

### 5.2 Promotion from Episodic to Semantic Memory

A fact is promoted from episodic to semantic memory when it is **observed consistently across multiple independent episodes**:

$$P(\text{promote } f \text{ to } \Sigma) = \sigma\left(\sum_{e \in E_f} w_e \cdot \text{confidence}(f, e) - \theta_{\text{promote}}\right)$$

where $E_f = \{e \in E : f \text{ is attested in } e\}$ and $w_e = I_E(e, t)$.

```
ALGORITHM 4: SemanticPromotionEngine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  E              — episodic memory store
  Σ              — semantic memory store
  config         — promotion thresholds

PROCEDURE RunPromotionCycle():

  // STEP 1: Extract candidate facts from recent episodes
  1. recent_episodes ← E.query(created_after=now()-config.promotion_window)
     candidate_facts ← []
     FOR each e IN recent_episodes:
         facts ← ExtractFactualAssertions(e)  // NER + relation extraction
         FOR each f IN facts:
             candidate_facts.append((f, e))

  // STEP 2: Cluster identical/near-identical facts
  2. fact_clusters ← SemanticCluster(candidate_facts, 
                                       threshold=config.fact_similarity_threshold)

  // STEP 3: Evaluate promotion criteria per cluster
  3. FOR each cluster IN fact_clusters:
         attestation_count ← |cluster.source_episodes|
         mean_confidence ← MEAN({confidence(f, e) : (f, e) ∈ cluster})
         source_diversity ← |UNIQUE(e.provenance.origin_agent_id : e ∈ cluster)|
         
         promotion_score ← (
             config.w_attestation × min(attestation_count / config.required_attestations, 1.0)
           + config.w_confidence × mean_confidence
           + config.w_diversity × min(source_diversity / config.required_sources, 1.0)
         )
         
         IF promotion_score ≥ config.promotion_threshold:
             // Check for contradictions with existing semantic memory
             contradictions ← FindContradictions(cluster.canonical_fact, Σ)
             IF |contradictions| > 0:
                 ESCALATE_TO_HUMAN_REVIEW(cluster, contradictions)
             ELSE:
                 σ_new ← ConstructSemanticFact(cluster)
                 WRITE σ_new TO Σ
                 EMIT Trace("semantic.promote", σ_new.fact_id, promotion_score)
```

### 5.3 Contradiction Detection

When a candidate fact $f_{\text{new}}$ potentially contradicts an existing fact $f_{\text{existing}} \in \Sigma$:

$$\text{Contradiction}(f_{\text{new}}, f_{\text{existing}}) = \begin{cases} 1 & \text{if } \text{NLI}(f_{\text{new}}, f_{\text{existing}}) = \texttt{CONTRADICTION} \wedge \cos(\mathbf{e}_{f_{\text{new}}}, \mathbf{e}_{f_{\text{existing}}}) > \theta_{\text{topic}} \\ 0 & \text{otherwise} \end{cases}$$

where $\text{NLI}$ is a **Natural Language Inference** model (e.g., DeBERTa-v3-large fine-tuned on MNLI). The topic similarity gate prevents false positives from unrelated domains.

---

## 6. Procedural Memory: Learned Workflow Internalization

### 6.1 Formal Specification

Procedural memory $P$ stores **reusable action sequences** (workflows, strategies, routines) extracted from successful episodic traces.

$$p = \langle \text{procedure\_id}, \text{trigger\_condition}, \text{action\_sequence}[], \text{preconditions}, \text{postconditions}, \text{success\_rate}, \text{execution\_count}, \text{avg\_latency}, \text{source\_episodes}[] \rangle$$

```protobuf
message Procedure {
  string procedure_id = 1;
  string trigger_description = 2;
  repeated float trigger_embedding = 3;
  repeated ProcedureStep steps = 4;
  repeated string preconditions = 5;
  repeated string postconditions = 6;
  float success_rate = 7;
  uint64 execution_count = 8;
  float avg_latency_ms = 9;
  repeated string source_episode_ids = 10;
  uint32 version = 11;
}

message ProcedureStep {
  uint32 order = 1;
  string action_type = 2;
  string action_template = 3;
  repeated string required_inputs = 4;
  repeated string expected_outputs = 5;
  string fallback_action = 6;
}
```

### 6.2 Algorithm: Procedural Memory Extraction via Trace Alignment

```
ALGORITHM 5: ProceduralMemoryExtractor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  E                — episodic memory store
  P                — procedural memory store
  config           — extraction thresholds

PROCEDURE ExtractProcedures():

  // STEP 1: Identify recurring successful task patterns
  1. successful_episodes ← E.query(outcome.success=TRUE, 
                                     created_after=now()-config.extraction_window)
     task_type_groups ← GroupBy(successful_episodes, key=task_type)

  // STEP 2: For each task type, align action traces
  2. FOR each (task_type, episodes) IN task_type_groups:
         IF |episodes| < config.min_episodes_for_extraction:  // typically ≥ 3
             CONTINUE
         
         // Extract action trace sequences
         traces ← [e.action_trace FOR e IN episodes]
         
         // Multiple Sequence Alignment (MSA) on action types
         // Using Needleman-Wunsch adapted for action sequences
         aligned ← MultipleSequenceAlignment(traces, 
                       similarity_fn=ActionSimilarity,
                       gap_penalty=config.gap_penalty)
         
         // Identify conserved (consensus) steps
         consensus_steps ← []
         FOR each position IN aligned.columns:
             actions_at_pos ← aligned.column(position)
             conservation ← MostFrequent(actions_at_pos).frequency / |actions_at_pos|
             IF conservation ≥ config.conservation_threshold:  // typically ≥ 0.7
                 consensus_step ← Generalize(MostFrequent(actions_at_pos))
                 consensus_steps.append(consensus_step)

  // STEP 3: Construct procedure record
  3.     IF |consensus_steps| ≥ config.min_procedure_length:
             trigger ← SynthesizeTrigger(task_type, episodes)
             preconditions ← ExtractPreconditions(episodes)
             postconditions ← ExtractPostconditions(episodes)
             
             p_new ← Procedure {
                 procedure_id = GenerateID(),
                 trigger_description = trigger,
                 trigger_embedding = Embed(trigger),
                 steps = consensus_steps,
                 preconditions = preconditions,
                 postconditions = postconditions,
                 success_rate = MEAN({e.reward_signal : e ∈ episodes}),
                 execution_count = 0,
                 source_episode_ids = [e.episode_id FOR e IN episodes]
             }
             
             // Deduplicate against existing procedures
             existing_match ← P.query(
                 cos(trigger_embedding, p_new.trigger_embedding) > 0.90
             )
             IF existing_match:
                 MergeProcedure(existing_match, p_new)
             ELSE:
                 WRITE p_new TO P
                 EMIT Trace("procedural.extract", p_new.procedure_id)
```

### 6.3 Action Sequence Similarity (Needleman-Wunsch Adaptation)

The similarity between two action steps $a_i, a_j$ is:

$$\text{ActionSimilarity}(a_i, a_j) = \omega_1 \cdot \mathbb{1}[a_i.\text{type} = a_j.\text{type}] + \omega_2 \cdot \cos(\mathbf{e}_{a_i.\text{detail}}, \mathbf{e}_{a_j.\text{detail}}) + \omega_3 \cdot \text{IOOverlap}(a_i, a_j)$$

where:

$$\text{IOOverlap}(a_i, a_j) = \frac{|a_i.\text{inputs} \cap a_j.\text{inputs}| + |a_i.\text{outputs} \cap a_j.\text{outputs}|}{|a_i.\text{inputs} \cup a_j.\text{inputs}| + |a_i.\text{outputs} \cup a_j.\text{outputs}|}$$

---

## 7. Memory Retrieval Engine: Multi-Signal Ranked Retrieval

### 7.1 Architecture: Hybrid Retrieval Pipeline

Retrieval is the **critical path** that determines whether memory is useful. SOTA retrieval combines multiple signals in a unified ranking framework.

$$\phi_{\text{retrieve}}(\text{query}, \text{budget}) \rightarrow \text{RankedResults}[\text{MemoryItem}]$$

### 7.2 Multi-Signal Scoring Function

For each candidate memory item $m$ and query $q$:

$$\text{Score}(m, q) = \sum_{s \in \mathcal{S}} w_s \cdot f_s(m, q)$$

where $\mathcal{S}$ is the set of scoring signals:

| Signal $s$ | Function $f_s(m, q)$ | Description |
|---|---|---|
| Semantic | $\cos(\mathbf{e}_m, \mathbf{e}_q)$ | Dense vector similarity |
| Lexical | $\text{BM25}(m.\text{content}, q.\text{terms})$ | Exact term matching |
| Recency | $e^{-\lambda(t_{\text{now}} - m.\text{timestamp})}$ | Temporal decay |
| Authority | $m.\text{provenance.confidence} \times \text{SourceTrust}(m.\text{source})$ | Provenance-weighted trust |
| Freshness | $\mathbb{1}[m.\text{version} = \text{latest}(m.\text{fact\_id})]$ | Version currency |
| Task Affinity | $\cos(\mathbf{e}_m, \mathbf{e}_{\text{task\_type}})$ | Relevance to current task type |
| Access Pattern | $\log(1 + m.\text{access\_count}) \times \text{Recency}_{\text{access}}$ | Popularity with recency |
| Graph Proximity | $\text{PPR}(m, q, \mathcal{G}_{\text{knowledge}})$ | Personalized PageRank in knowledge graph |

### 7.3 Query Decomposition and Routing

Before retrieval, the query is **decomposed, expanded, and routed** per signal source:

```
ALGORITHM 6: QueryDecompositionAndRouting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  raw_query        — original user/agent query
  memory_layers    — {W, S, E, Σ, P}
  latency_budget   — maximum retrieval latency (ms)

Output:
  ranked_results   — provenance-tagged, ranked memory items

PROCEDURE RetrieveWithDecomposition(raw_query, memory_layers, latency_budget):

  // STEP 1: Query Analysis and Decomposition
  1. analysis ← LLM_analyze_query(raw_query)
     subqueries ← analysis.decomposed_subqueries     // e.g., ["user travel preferences", 
                                                      //        "airline booking workflow",
                                                      //        "visa requirements for Japan"]
     query_type ← analysis.type                       // factual | procedural | episodic | hybrid
     temporal_scope ← analysis.temporal_scope         // recent | historical | all

  // STEP 2: Route subqueries to appropriate memory layers and retrieval modes
  2. retrieval_plan ← []
     FOR each sq IN subqueries:
         targets ← RouteToLayers(sq, query_type, temporal_scope)
         // Example routing:
         //   "user travel preferences" → E (episodic) + Σ (semantic), mode=semantic_search
         //   "airline booking workflow" → P (procedural), mode=trigger_match
         //   "visa requirements" → Σ (semantic), mode=hybrid(BM25 + semantic)
         FOR each (layer, mode) IN targets:
             retrieval_plan.append({
                 subquery: sq,
                 expanded_query: ExpandQuery(sq),  // synonym expansion, HyDE
                 layer: layer,
                 mode: mode,
                 deadline: latency_budget / |retrieval_plan|
             })

  // STEP 3: Parallel Execution with Deadline Enforcement
  3. raw_results ← ParallelExecute(retrieval_plan, 
                                     timeout=latency_budget,
                                     strategy=RETURN_AVAILABLE_ON_TIMEOUT)

  // STEP 4: Cross-Layer Fusion and Reranking
  4. fused ← DeduplicateAndMerge(raw_results)
     
     // Multi-signal scoring
     FOR each m IN fused:
         m.final_score ← Σ_s (w_s × f_s(m, raw_query))
     
     // LLM-based reranking on top-K candidates
     top_k ← TopK(fused, k=config.rerank_pool_size)  // typically k=20
     reranked ← LLM_rerank(top_k, raw_query, 
                             criteria=["relevance", "actionability", "recency"])

  // STEP 5: Token-Budget-Aware Truncation
  5. selected ← []
     budget_remaining ← config.retrieval_token_budget
     FOR each m IN reranked:
         IF tokens(m.content) ≤ budget_remaining:
             selected.append(m)
             budget_remaining -= tokens(m.content)
         ELSE IF budget_remaining > MIN_ITEM_TOKENS:
             compressed ← CompressForContext(m, max_tokens=budget_remaining)
             selected.append(compressed)
             BREAK
     
     // Attach provenance tags
     FOR each m IN selected:
         m.provenance_tag ← FormatProvenance(m.source, m.confidence, m.timestamp)
     
     RETURN selected
```

### 7.4 Hypothetical Document Embedding (HyDE) for Query Expansion

Instead of embedding the raw query directly, generate a **hypothetical answer** and embed that:

$$\mathbf{e}_{\text{HyDE}} = \text{Embed}\left(\text{LLM}_{\text{generate}}(\text{"Answer this as if you knew: "} + q)\right)$$

$$\text{Retrieval}(q) = \text{TopK}\left(\{m \in \mathcal{M} : \cos(\mathbf{e}_m, \mathbf{e}_{\text{HyDE}}) > \theta\}\right)$$

This bridges the vocabulary gap between queries and stored documents. The SOTA enhancement is **multi-HyDE**: generate $k$ hypothetical documents from different perspectives and average their embeddings:

$$\mathbf{e}_{\text{multi-HyDE}} = \frac{1}{k}\sum_{i=1}^{k} \text{Embed}\left(\text{LLM}_{\text{generate}}(q, \text{perspective}_i)\right)$$

### 7.5 Iterative Retrieval with Self-Critique

```
ALGORITHM 7: IterativeRetrievalWithCritique
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  query            — original query
  max_iterations   — bounded iteration count (typically 3)
  quality_threshold — minimum retrieval quality score

Output:
  final_results    — high-quality retrieved items

PROCEDURE IterativeRetrieve(query, max_iterations, quality_threshold):
  
  current_query ← query
  accumulated_results ← []
  
  FOR iteration IN 1..max_iterations:
      // Retrieve
      results_i ← φ_retrieve(current_query, budget)
      accumulated_results ← Merge(accumulated_results, results_i)
      
      // Critique: does the retrieved set answer the query?
      critique ← LLM_critique(query, accumulated_results)
      // critique = {quality_score: float, gaps: [str], refinement_suggestion: str}
      
      IF critique.quality_score ≥ quality_threshold:
          BREAK  // Sufficient quality achieved
      
      IF |critique.gaps| = 0:
          BREAK  // No identifiable gaps; further iteration unlikely to help
      
      // Refine query based on identified gaps
      current_query ← LLM_refine_query(query, critique.gaps, 
                                          critique.refinement_suggestion,
                                          already_retrieved=accumulated_results)
  
  RETURN accumulated_results
```

---

## 8. Memory Consolidation and Pruning Engine

### 8.1 The Consolidation Problem

Over time, memory stores accumulate **redundant, stale, contradictory, and low-value entries**. Consolidation is the process of maintaining memory health, analogous to garbage collection in runtime systems.

### 8.2 Formal Pruning Criteria

A memory item $m$ is a **pruning candidate** if:

$$\text{PruneScore}(m, t) = \underbrace{w_{\text{age}} \cdot \frac{t - m.\text{timestamp}}{T_{\max}}}_{\text{staleness}} + \underbrace{w_{\text{access}} \cdot \left(1 - \frac{\log(1 + m.\text{access\_count})}{\log(1 + A_{\max})}\right)}_{\text{disuse}} + \underbrace{w_{\text{importance}} \cdot (1 - I_E(m, t))}_{\text{low importance}} + \underbrace{w_{\text{redundancy}} \cdot R(m, \mathcal{M})}_{\text{redundancy with peers}}$$

where the redundancy score is:

$$R(m, \mathcal{M}) = \max_{m' \in \mathcal{M} \setminus \{m\}} \cos(\mathbf{e}_m, \mathbf{e}_{m'}) \times \mathbb{1}[m'.\text{importance} \geq m.\text{importance}]$$

Prune $m$ if $\text{PruneScore}(m, t) > \theta_{\text{prune}}$.

### 8.3 Algorithm: Memory Consolidation Agent

```
ALGORITHM 8: MemoryConsolidationAgent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  memory_layers   — {E, Σ, P}
  config          — consolidation policies

// Runs as a scheduled background agent (e.g., every 6 hours)
PROCEDURE RunConsolidation(memory_layers, config):

  FOR each layer IN memory_layers:
      all_items ← layer.scan()
      
      // PHASE 1: Compute prune scores
      prune_candidates ← []
      FOR each m IN all_items:
          ps ← ComputePruneScore(m, now(), config)
          IF ps > config.prune_threshold:
              prune_candidates.append((m, ps))
      
      // PHASE 2: Sort by prune score descending (most pruneable first)
      SORT prune_candidates BY ps DESC
      
      // PHASE 3: Execute pruning with safety checks
      pruned_count ← 0
      FOR each (m, ps) IN prune_candidates:
          IF pruned_count ≥ config.max_prune_per_cycle:
              BREAK  // Rate limit to prevent mass deletion
          
          // Safety: check if any active procedure or task references this item
          IF IsReferencedByActiveProcedure(m, P) OR IsReferencedByActiveTask(m):
              CONTINUE  // Skip: item is in active use
          
          // For items above high-confidence prune threshold, auto-delete
          IF ps > config.auto_prune_threshold:
              ArchiveAndDelete(m, layer)  // Soft-delete with archive
              pruned_count += 1
          ELSE:
              // Items in uncertain zone: merge or summarize instead of delete
              merge_target ← FindMergeCandidate(m, layer)
              IF merge_target ≠ NULL:
                  MergeItems(merge_target, m, layer)
                  pruned_count += 1
      
      // PHASE 4: Deduplication pass
      embeddings ← [m.embedding FOR m IN layer.scan()]
      duplicate_pairs ← FindNearDuplicates(embeddings, 
                                              threshold=config.dedup_cos_threshold)
      FOR each (m_a, m_b) IN duplicate_pairs:
          survivor ← SelectSurvivor(m_a, m_b)  // Keep higher importance/newer
          MergeItems(survivor, Loser(m_a, m_b, survivor), layer)
      
      // PHASE 5: Contradiction resolution
      IF layer = Σ:
          contradictions ← DetectContradictions(layer)
          FOR each (σ_a, σ_b) IN contradictions:
              IF CanAutoResolve(σ_a, σ_b):
                  // Keep the one with more attestations and higher recency
                  ResolveContradiction(σ_a, σ_b, strategy="recency_and_attestation")
              ELSE:
                  ESCALATE_TO_HUMAN_REVIEW(σ_a, σ_b)
      
      EMIT Metrics("consolidation.pruned", pruned_count)
      EMIT Metrics("consolidation.merged", merged_count)
      EMIT Metrics("consolidation.layer_size", layer.count())
```

### 8.4 Merge Operation

When two items $m_a, m_b$ are merged:

$$m_{\text{merged}} = \left\langle \begin{array}{l} \text{content} = \text{LLM\_merge}(m_a.\text{content}, m_b.\text{content}) \\ \text{embedding} = \text{Embed}(m_{\text{merged}}.\text{content}) \\ \text{importance} = \max(m_a.\text{importance}, m_b.\text{importance}) \\ \text{access\_count} = m_a.\text{access\_count} + m_b.\text{access\_count} \\ \text{lineage} = m_a.\text{lineage} \cup m_b.\text{lineage} \\ \text{timestamp} = \max(m_a.\text{timestamp}, m_b.\text{timestamp}) \end{array} \right\rangle$$

---

## 9. Memory Write Policies and Admission Control

### 9.1 The Write Policy Framework

Every write to durable memory ($E$, $\Sigma$, $P$) must pass through a **policy gate**:

$$\phi_{\text{admit}}: \text{MemoryItem} \times \text{TargetLayer} \times \text{Context} \rightarrow \{\texttt{ACCEPT}, \texttt{REJECT}, \texttt{DEFER}, \texttt{MERGE}\}$$

### 9.2 Typed Write Policy Contract

```protobuf
message WritePolicy {
  string policy_id = 1;
  TargetLayer target = 2;
  
  // Admission gates (all must pass)
  float min_importance_score = 3;
  float max_similarity_to_existing = 4;    // dedup threshold
  float min_provenance_confidence = 5;
  bool require_factual_verification = 6;
  bool require_human_approval = 7;
  
  // Expiry
  google.protobuf.Duration default_ttl = 8;
  ExpiryStrategy expiry_strategy = 9;
  
  // Rate limiting
  uint32 max_writes_per_minute = 10;
  uint32 max_items_per_layer = 11;
  
  enum ExpiryStrategy {
    HARD_TTL = 0;           // Delete after TTL
    DECAY_SCORE = 1;        // Subject to pruning by importance decay
    NEVER_EXPIRE = 2;       // Permanent (semantic facts)
    REFRESH_ON_ACCESS = 3;  // TTL resets on retrieval
  }
}
```

### 9.3 Decision Matrix

| Condition | Working ($W$) | Session ($S$) | Episodic ($E$) | Semantic ($\Sigma$) | Procedural ($P$) |
|---|---|---|---|---|---|
| **Write Gate** | None (direct) | Schema check | Full admission | Promotion + contradiction | Trace alignment |
| **TTL** | Step-scoped | Session-scoped | Decay-based | Never / Annual review | Version-based |
| **Dedup** | N/A | Key-based | Embedding sim > 0.92 | NLI + embedding | Trigger sim > 0.90 |
| **Provenance Required** | No | Task ID | Full trace | Multi-episode attestation | Source episodes |
| **Human Approval** | No | No | On flag only | On contradiction | On low confidence |

---

## 10. End-to-End Memory Lifecycle: Integrated Agent Loop

### 10.1 Complete Memory Flow in One Agent Step

```
ALGORITHM 9: AgentStepWithMemory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  user_input      — current user message or task trigger
  agent_state     — persistent agent state across steps
  M               — memory system ⟨W, S, E, Σ, P⟩

Output:
  response        — agent response
  updated_state   — updated agent state

PROCEDURE ExecuteAgentStep(user_input, agent_state, M):

  // ═══════════════════════════════════════════════════
  // PHASE 1: PLAN — Understand intent, decompose task
  // ═══════════════════════════════════════════════════
  1. task_analysis ← AnalyzeIntent(user_input, agent_state.history)
     subtasks ← Decompose(task_analysis)

  // ═══════════════════════════════════════════════════
  // PHASE 2: RETRIEVE — Pull relevant memory
  // ═══════════════════════════════════════════════════
  2. // Recall task state from session scratchpad
     task_context ← RecallFromScratchpad(agent_state.task_id, 
                                           task_analysis.required_keys)
     
     // Retrieve from episodic, semantic, procedural memory
     retrieved_evidence ← IterativeRetrieve(
         query=task_analysis.retrieval_query,
         layers=[E, Σ],
         max_iterations=3,
         quality_threshold=0.8
     )
     
     // Retrieve applicable procedures
     applicable_procedures ← P.query(
         cos(trigger_embedding, Embed(task_analysis.description)) > 0.75
     )

  // ═══════════════════════════════════════════════════
  // PHASE 3: COMPILE — Build working memory (context)
  // ═══════════════════════════════════════════════════
  3. W_t ← PrefillCompiler.Compile(
         role_policy     = agent_state.role_policy,
         task_state      = task_context,
         retrieved       = retrieved_evidence,
         tools           = agent_state.available_tools,
         procedures      = applicable_procedures,
         history         = AdaptiveContextWindowManager(agent_state.history, 
                                                         task_analysis, B_t),
         token_budget    = B_t
     )
     ASSERT tokens(W_t) ≤ N_max - R_reserved

  // ═══════════════════════════════════════════════════
  // PHASE 4: ACT — Generate response / execute tools
  // ═══════════════════════════════════════════════════
  4. llm_output ← LLM_generate(W_t)
     IF llm_output.contains_tool_calls:
         tool_results ← ExecuteTools(llm_output.tool_calls, 
                                       authorization=agent_state.auth_scope)
         // Re-inject tool results and re-generate if needed
         W_t' ← InjectToolResults(W_t, tool_results)
         llm_output ← LLM_generate(W_t')

  // ═══════════════════════════════════════════════════
  // PHASE 5: VERIFY — Check output quality
  // ═══════════════════════════════════════════════════
  5. verification ← VerifyOutput(llm_output, task_analysis, retrieved_evidence)
     IF verification.hallucination_detected:
         // REPAIR: re-generate with explicit grounding constraint
         W_t_repair ← InjectGroundingConstraint(W_t, verification.flagged_claims)
         llm_output ← LLM_generate(W_t_repair)
         verification ← VerifyOutput(llm_output, task_analysis, retrieved_evidence)
     
     IF verification.quality_score < config.min_quality:
         ESCALATE("output_quality_below_threshold", llm_output, verification)

  // ═══════════════════════════════════════════════════
  // PHASE 6: COMMIT — Update memory layers
  // ═══════════════════════════════════════════════════
  6. response ← llm_output.final_response
     
     // Update session scratchpad with new task state
     new_task_state ← ExtractTaskState(llm_output, task_analysis)
     OffloadToScratchpad(agent_state.task_id, new_task_state)
     
     // Evaluate interaction for episodic storage
     episode_candidate ← ConstructEpisode(user_input, llm_output, 
                                            tool_results, verification)
     admission_result ← EpisodicAdmissionController(episode_candidate, E, config)
     
     // Extract and store new user preferences / corrections
     preference_updates ← ExtractPreferences(user_input, llm_output)
     FOR each pref IN preference_updates:
         IF φ_admit(pref, target=Σ) = ACCEPT:
             WRITE pref TO Σ
     
     // Update history
     agent_state.history.append({user: user_input, assistant: response})
     
     // Emit observability
     EMIT Trace("agent.step", {
         task_id: agent_state.task_id,
         tokens_used: tokens(W_t),
         retrieval_count: |retrieved_evidence|,
         admission_result: admission_result,
         verification_score: verification.quality_score,
         latency_ms: elapsed()
     })

  7. RETURN (response, agent_state)
```

### 10.2 State Transition Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MEMORY LIFECYCLE STATE MACHINE                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌──────────┐    context     ┌───────────┐                           │
│   │ EXTERNAL │──────load────→│  WORKING   │←───compile────┐          │
│   │  INPUT   │               │  MEMORY W  │               │          │
│   └──────────┘               └─────┬──────┘               │          │
│                                    │                       │          │
│                         ┌──────────┼──────────┐            │          │
│                         │ offload  │ retrieve │            │          │
│                         ▼          │          ▼            │          │
│                  ┌──────────┐      │   ┌──────────┐       │          │
│                  │ SESSION  │      │   │ EPISODIC │       │          │
│                  │    S     │      │   │    E     │       │          │
│                  └────┬─────┘      │   └────┬─────┘       │          │
│                       │            │        │              │          │
│            session    │            │        │ promote      │          │
│            end +      │            │        │ (≥N attested │          │
│            admit      │            │        │  episodes)   │          │
│                       ▼            │        ▼              │          │
│                  ┌──────────┐      │   ┌──────────┐       │          │
│                  │ EPISODIC │──────┘   │ SEMANTIC │───────┘          │
│                  │    E     │  recall  │    Σ     │  inject          │
│                  └────┬─────┘         └──────────┘                   │
│                       │                     ▲                         │
│            trace      │                     │ human validation       │
│            alignment  │                     │ or auto-resolve        │
│                       ▼                     │                         │
│                  ┌──────────┐               │                         │
│                  │PROCEDURAL│───────────────┘                         │
│                  │    P     │  successful procedure → fact            │
│                  └──────────┘                                         │
│                                                                        │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │  CONSOLIDATION AGENT (periodic background)                   │     │
│   │  • Prune stale/low-importance items                         │     │
│   │  • Merge near-duplicates                                     │     │
│   │  • Resolve contradictions                                    │     │
│   │  • Promote attested episodic → semantic                      │     │
│   │  • Extract recurring traces → procedural                     │     │
│   └─────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Production-Grade Reliability and Operational Concerns

### 11.1 Idempotent Memory Writes

Every write to durable memory must be **idempotent**. The idempotency key is derived from:

$$\text{IdempotencyKey}(m) = \text{SHA256}(m.\text{source.task\_id} \| m.\text{source.step\_index} \| m.\text{schema\_type})$$

Repeated writes with the same key are silently accepted (no-op) rather than creating duplicates.

### 11.2 Backpressure and Rate Limiting

Memory write throughput is bounded:

$$\text{WriteRate}_{\text{layer}} \leq \frac{R_{\max}^{\text{layer}}}{\Delta t} \quad \text{with jittered retry on rejection}$$

When the write queue exceeds capacity, apply **backpressure** by buffering in the session layer and deferring promotion.

### 11.3 Observability Contract

Every memory operation emits structured traces:

```protobuf
message MemoryTrace {
  string trace_id = 1;
  string operation = 2;    // "read", "write", "prune", "merge", "promote"
  string layer = 3;
  string item_id = 4;
  float latency_ms = 5;
  bool success = 6;
  string error_class = 7;
  uint32 tokens_involved = 8;
  map<string, string> metadata = 9;
}
```

Key metrics to monitor:

| Metric | Formula | Alert Threshold |
|--------|---------|-----------------|
| Memory Store Size | $|\mathcal{L}|$ per layer $\mathcal{L}$ | > 90% capacity |
| Retrieval Latency p99 | $P_{99}(\text{latency}_{\text{retrieve}})$ | > 500ms |
| Admission Rate | $\frac{|\text{accepted}|}{|\text{candidates}|}$ | < 5% or > 80% (miscalibrated) |
| Retrieval Hit Rate | $\frac{|\text{queries with } \geq 1 \text{ relevant result}|}{|\text{total queries}|}$ | < 60% |
| Contradiction Rate | $\frac{|\text{contradictions detected}|}{|\Sigma|}$ | > 2% |
| Context Budget Utilization | $\frac{|W_t|}{N_{\max} - R_{\text{reserved}}}$ | > 95% sustained |

### 11.4 Failure Recovery and Graceful Degradation

| Failure Mode | Detection | Recovery |
|---|---|---|
| Retrieval timeout | Deadline exceeded | Return partial results + flag `degraded_retrieval` |
| Memory store unavailable | Connection error / circuit breaker open | Fall back to working memory only; disable writes; queue for retry |
| Admission controller crash | Health check failure | Buffer candidates in session store; replay on recovery |
| Consolidation agent stall | Heartbeat timeout | Kill and restart; resume from last checkpoint |
| Embedding service down | gRPC error | Use cached embeddings; fall back to BM25-only retrieval |
| Token budget exceeded | Assert violation in compiler | Emergency truncation with priority-based eviction |

### 11.5 Cost Optimization

$$\text{Cost}_{\text{memory}} = \underbrace{C_{\text{embed}} \cdot N_{\text{embed}}}_{\text{embedding generation}} + \underbrace{C_{\text{store}} \cdot |\mathcal{M}|}_{\text{storage}} + \underbrace{C_{\text{retrieve}} \cdot N_{\text{queries}}}_{\text{retrieval ops}} + \underbrace{C_{\text{llm}} \cdot N_{\text{reflection}}}_{\text{LLM calls for scoring/reflection}}$$

Optimization strategies:

1. **Batch embedding generation**: Amortize embedding API calls by batching new items.
2. **Tiered storage**: Hot items in vector DB (HNSW index), warm in compressed store, cold in archive.
3. **Cache retrieval results**: LRU cache on $(query\_hash, layer, timestamp\_bucket) \rightarrow results$ with TTL.
4. **Lazy reflection**: Only invoke LLM-based importance scoring for items that pass cheap heuristic pre-filters.
5. **Shared embeddings**: Reuse task embeddings for both routing and retrieval scoring.

---

## 12. Evaluation Infrastructure for Memory Quality

### 12.1 Memory Quality Metrics

$$\text{MemoryQuality} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \left[ w_{\text{ret}} \cdot \text{RetrievalPrecision}(\tau) + w_{\text{util}} \cdot \text{UtilizationRate}(\tau) + w_{\text{fresh}} \cdot \text{FreshnessPenalty}(\tau) \right]$$

where:

$$\text{RetrievalPrecision}(\tau) = \frac{|\text{retrieved items judged relevant by evaluator}|}{|\text{total retrieved items}|}$$

$$\text{UtilizationRate}(\tau) = \frac{|\text{retrieved items actually cited in response}|}{|\text{total retrieved items}|}$$

$$\text{FreshnessPenalty}(\tau) = 1 - \frac{|\text{retrieved items that are stale/outdated}|}{|\text{total retrieved items}|}$$

### 12.2 Automated Regression Testing

```
ALGORITHM 10: MemoryRegressionSuite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROCEDURE RunMemoryRegressions():

  // Test 1: Admission Consistency
  FOR each (candidate, expected_decision) IN admission_test_set:
      actual ← φ_admit(candidate, target_layer, context)
      ASSERT actual.decision = expected_decision
  
  // Test 2: Retrieval Quality
  FOR each (query, expected_items, min_recall) IN retrieval_test_set:
      results ← φ_retrieve(query, budget)
      recall ← |results ∩ expected_items| / |expected_items|
      ASSERT recall ≥ min_recall
  
  // Test 3: Deduplication Effectiveness
  INJECT known_duplicate INTO E
  RunConsolidation()
  ASSERT NOT EXISTS(known_duplicate, E)  // Should have been merged
  
  // Test 4: Contradiction Detection
  INJECT contradictory_fact INTO Σ
  contradictions ← DetectContradictions(Σ)
  ASSERT contradictory_fact IN contradictions
  
  // Test 5: Context Budget Compliance
  FOR each scenario IN budget_stress_test_set:
      W_t ← PrefillCompiler.Compile(scenario)
      ASSERT tokens(W_t) ≤ N_max - R_reserved
  
  // Test 6: Procedural Memory Accuracy
  FOR each (task_type, expected_procedure_steps) IN procedural_test_set:
      procedures ← P.query(task_type)
      step_accuracy ← SequenceAlignment(procedures[0].steps, expected_procedure_steps)
      ASSERT step_accuracy ≥ 0.85
```

---

## 13. Summary: Design Invariants for Production Memory Systems

| Invariant | Enforcement Mechanism |
|---|---|
| **No anonymous context**: every retrieved item carries provenance | `MemoryItem.provenance` is a required field; retrieval rejects items without it |
| **Hard token budget**: context window never exceeds $N_{\max} - R_{\text{reserved}}$ | `ASSERT` in `PrefillCompiler.Compile`; emergency truncation on violation |
| **Idempotent writes**: duplicate writes are no-ops | SHA256 idempotency key on every durable write |
| **Validated admission only**: no item enters durable memory without passing gates | `φ_admit` pipeline with schema, novelty, importance, provenance gates |
| **Bounded consolidation**: pruning never deletes actively-referenced items | Reference check before delete; soft-delete with archive |
| **Contradiction-safe semantics**: contradictions are detected and escalated | NLI-based contradiction detection; human escalation on ambiguity |
| **Observable memory health**: all operations emit structured traces and metrics | `MemoryTrace` protocol; dashboard alerts on anomalous rates |
| **Graceful degradation**: retrieval failure does not crash the agent | Timeout + partial result + fallback flag |
| **Cost-bounded operations**: LLM calls for scoring are rate-limited and cached | Lazy reflection; LRU cache on reflection outputs |
| **Separation of layers**: working ≠ session ≠ episodic ≠ semantic ≠ procedural | Typed stores with distinct schemas, TTLs, and write policies |

---

**This architecture transforms memory from passive storage into an actively managed, provenance-tracked, quality-gated, cost-bounded subsystem** that enables agents to retain what matters, discard what does not, retrieve with precision under latency constraints, and improve continuously through consolidation, promotion, and evaluation—at production scale.