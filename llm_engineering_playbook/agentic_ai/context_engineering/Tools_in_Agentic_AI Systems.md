

# Tools in Agentic AI Systems: A Principal-Level Technical Report

## Typed Infrastructure for External Action, Retrieval, and Closed-Loop Execution

---

## 1. Formal Definition and Architectural Role of Tools

### 1.1 Tools as Typed External Capability Interfaces

A **tool** in the agentic AI stack is a **typed, schema-described, externally-bound callable** that extends the computational closure of an LLM agent beyond text generation into observable state mutation, information retrieval, and environment interaction. Formally:

**Definition 1.1 (Tool).**
A tool $\mathcal{T}$ is a 7-tuple:

$$
\mathcal{T} = \langle \, \texttt{id}, \; \Sigma_{\text{in}}, \; \Sigma_{\text{out}}, \; \pi, \; \delta, \; \kappa, \; \tau_{\max} \, \rangle
$$

where:

| Symbol | Semantics |
|---|---|
| $\texttt{id}$ | Globally unique, versioned tool identifier (e.g., `flights/search@v2.3`) |
| $\Sigma_{\text{in}}$ | Input schema: a typed JSON Schema or Protobuf message descriptor specifying required and optional parameters, constraints, and validation rules |
| $\Sigma_{\text{out}}$ | Output schema: typed return envelope including structured result, error classes, pagination cursors, and provenance metadata |
| $\pi$ | Permission descriptor: mutation class (`read-only`, `state-mutating`, `irreversible`), required authorization scopes, and human-approval gates |
| $\delta$ | Description embedding: a high-signal natural-language description compiled for LLM tool-selection inference |
| $\kappa$ | Cost model: expected latency tier ($L_0 < 100\text{ms}$, $L_1 < 1\text{s}$, $L_2 < 10\text{s}$, $L_3$ async), token cost of injecting the schema into context, and monetary cost per invocation |
| $\tau_{\max}$ | Deadline: hard timeout with cancellation semantics |

**Definition 1.2 (Tool Registry).**
A tool registry $\mathcal{R}$ is a discoverable, versioned catalog:

$$
\mathcal{R} = \{ \mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_N \}
$$

exposed via **Model Context Protocol (MCP)** capability negotiation, supporting:

- **`tools/list`**: paginated enumeration with cursor-based traversal
- **`tools/call`**: schema-validated invocation with deadline propagation
- **`notifications/tools/list_changed`**: push-based invalidation on registry mutation

The registry is the **single source of truth** for agent-accessible capabilities. No tool is callable unless registered, schema-validated, and authorization-scoped.

### 1.2 Why Tools Are Architecturally Non-Negotiable

An LLM without tools is a **closed-world reasoner** operating over a frozen parametric snapshot $\theta$ trained at time $t_{\text{cutoff}}$. The information-theoretic limitation is precise:

$$
I_{\text{agent}}(q, t) = \underbrace{I_{\theta}(q)}_{\text{parametric knowledge}} + \underbrace{I_{\mathcal{R}}(q, t)}_{\text{tool-retrieved, real-time}}
$$

For any query $q$ with temporal dependency $t > t_{\text{cutoff}}$ or requiring external state observation, $I_{\theta}(q) \approx 0$ and the agent **must** delegate to a tool. Tools thus transform the agent from a **static text generator** into a **dynamic actuator** capable of:

1. **Observation**: querying live databases, APIs, sensors, browser state, file systems
2. **Mutation**: writing records, sending messages, deploying code, modifying infrastructure
3. **Verification**: running tests, validating outputs against ground truth, inspecting execution traces

---

## 2. Protocol Stack: Typed Boundaries for Tool Exposure

### 2.1 Three-Tier Protocol Architecture

Tools are **not** exposed as ad hoc function names injected into a prompt string. They are served through a **typed protocol stack** with explicit schemas, error classes, versioned contracts, and deadline propagation:

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: JSON-RPC 2.0 — User / Application Boundary│
│  • Human-facing API surface                          │
│  • Request/response with structured error codes      │
│  • Pagination, capability discovery                  │
├─────────────────────────────────────────────────────┤
│  Layer 2: MCP (Model Context Protocol) — Tool Plane  │
│  • tools/list, tools/call, resources/read            │
│  • Schema-described inputs/outputs                   │
│  • Change notifications, capability negotiation      │
│  • Local (stdio) and remote (SSE/HTTP) transports    │
├─────────────────────────────────────────────────────┤
│  Layer 1: gRPC / Protobuf — Internal Execution Plane │
│  • Low-latency service-to-service calls              │
│  • Strongly typed message contracts                  │
│  • Streaming, deadline propagation, load balancing   │
│  • mTLS, per-RPC authorization                       │
└─────────────────────────────────────────────────────┘
```

**Design Rationale:**

| Layer | Why This Protocol | Trade-off |
|---|---|---|
| **JSON-RPC** | Human-readable, universal client support, natural fit for LLM-generated structured output | Higher serialization overhead vs. binary protocols |
| **MCP** | Purpose-built for LLM tool discovery; supports `tools`, `resources`, `prompts` as first-class primitives with schema negotiation | Emerging standard; requires MCP server implementation per tool provider |
| **gRPC** | Sub-millisecond internal dispatch, strongly typed Protobuf contracts, native streaming, built-in deadline/cancellation propagation | Requires code generation; not human-readable on wire |

### 2.2 Tool Contract Specification (Protobuf + MCP)

```protobuf
// Internal gRPC contract for tool execution
syntax = "proto3";
package agentic.tools.v2;

message ToolInvocationRequest {
  string tool_id = 1;                    // e.g., "flights/search@v2.3"
  string idempotency_key = 2;           // Client-generated UUID for retry safety
  google.protobuf.Struct arguments = 3; // Schema-validated input
  string caller_scope = 4;              // Authorization scope of invoking agent
  google.protobuf.Duration deadline = 5; // Hard timeout
  map<string, string> trace_context = 6; // OpenTelemetry W3C trace propagation
}

message ToolInvocationResponse {
  oneof result {
    google.protobuf.Struct content = 1;  // Structured output
    ToolError error = 2;                 // Typed error
  }
  ProvenanceMetadata provenance = 3;     // Source, timestamp, authority
  PaginationCursor next_cursor = 4;     // For paginated results
  CostReport cost = 5;                  // Tokens consumed, latency, monetary cost
}

message ToolError {
  enum ErrorClass {
    INVALID_ARGUMENTS = 0;
    PERMISSION_DENIED = 1;
    RATE_LIMITED = 2;
    UPSTREAM_FAILURE = 3;
    TIMEOUT = 4;
    HUMAN_APPROVAL_REQUIRED = 5;
  }
  ErrorClass class = 1;
  string message = 2;
  bool retryable = 3;
  google.protobuf.Duration retry_after = 4;
}
```

```json
// MCP tool declaration (served via tools/list)
{
  "name": "search_flights",
  "version": "2.3.0",
  "description": "Search real-time flight availability between airports. Returns ranked options by price, duration, and stops. Use when the user requests flight booking, comparison, or availability checking. Do NOT use for hotel or ground transport queries.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "origin": { "type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA airport code of departure" },
      "destination": { "type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA airport code of arrival" },
      "departure_date": { "type": "string", "format": "date", "description": "ISO 8601 date" },
      "passengers": { "type": "integer", "minimum": 1, "maximum": 9 },
      "cabin_class": { "type": "string", "enum": ["economy", "business", "first"], "default": "economy" }
    },
    "required": ["origin", "destination", "departure_date"]
  },
  "outputSchema": { "$ref": "#/definitions/FlightSearchResult" },
  "mutationClass": "read-only",
  "latencyTier": "L1",
  "costPerCall": { "usd": 0.002 },
  "requiresApproval": false
}
```

**Critical Observation**: The `description` field is the **highest-leverage context engineering artifact** for tool selection. It functions as a **compiled directive** to the LLM's tool-routing inference. It must contain:

1. **Positive specification**: what the tool does, precisely
2. **Negative specification**: what the tool does NOT do (disambiguation from sibling tools)
3. **Trigger conditions**: when to invoke (intent patterns)
4. **Input constraints**: type, format, valid ranges — redundant with schema but required for LLM grounding

---

## 3. Context Engineering for Tool Affordances

### 3.1 The Tool Context Budget Problem

Injecting all $N$ tool schemas into the LLM context window is **infeasible** at scale. Each tool schema consumes $c_i$ tokens. The total tool context cost is:

$$
C_{\text{tools}} = \sum_{i=1}^{N} c_i
$$

For an enterprise registry with $N = 500$ tools and average schema size $\bar{c} = 200$ tokens, $C_{\text{tools}} = 100{,}000$ tokens — exceeding most context windows before any task context, memory, or retrieval payload is loaded.

**Theorem 3.1 (Tool Budget Constraint).**
Given a context window of $W$ tokens and allocations for system policy ($B_{\text{sys}}$), task state ($B_{\text{task}}$), retrieval evidence ($B_{\text{ret}}$), memory ($B_{\text{mem}}$), and reasoning reserve ($B_{\text{res}}$), the tool budget is:

$$
B_{\text{tools}} = W - B_{\text{sys}} - B_{\text{task}} - B_{\text{ret}} - B_{\text{mem}} - B_{\text{res}}
$$

The set of tools injected into context, $\mathcal{T}_{\text{active}} \subseteq \mathcal{R}$, must satisfy:

$$
\sum_{\mathcal{T}_i \in \mathcal{T}_{\text{active}}} c_i \leq B_{\text{tools}}
$$

This is a **constrained selection optimization** problem.

### 3.2 Lazy Tool Loading via Relevance-Gated Selection

**SOTA Approach: Two-Phase Tool Resolution**

Tools are **never** bulk-loaded. Instead, a **tool router** selects the minimal relevant subset per turn.

**Phase 1 — Intent-Based Pre-Filter (Sub-Linear Scan)**

Given user query $q$, compute a task intent embedding $\mathbf{e}_q = \text{Embed}(q)$ and retrieve candidate tools by approximate nearest-neighbor search over pre-computed tool description embeddings:

$$
\mathcal{T}_{\text{candidates}} = \text{ANN}\!\left(\mathbf{e}_q, \; \{ \mathbf{e}_{\delta_i} \}_{i=1}^{N}, \; k \right)
$$

where $k$ is a retrieval budget (typically $k \in [5, 15]$).

**Phase 2 — Schema-Aware Re-Ranking**

Re-rank candidates using a scoring function that incorporates semantic relevance, historical invocation success, cost, and mutation safety:

$$
\text{score}(\mathcal{T}_i \mid q, \text{ctx}) = \underbrace{\alpha \cdot \text{sim}(\mathbf{e}_q, \mathbf{e}_{\delta_i})}_{\text{semantic relevance}} + \underbrace{\beta \cdot P(\text{success} \mid \mathcal{T}_i, q)}_{\text{historical success rate}} - \underbrace{\gamma \cdot \kappa_i}_{\text{cost penalty}} - \underbrace{\lambda \cdot \mathbb{1}[\pi_i = \text{mutating}]}_{\text{mutation risk}}
$$

Select $\mathcal{T}_{\text{active}}$ by solving:

$$
\mathcal{T}_{\text{active}}^{*} = \arg\max_{\mathcal{S} \subseteq \mathcal{T}_{\text{candidates}}} \sum_{\mathcal{T}_i \in \mathcal{S}} \text{score}(\mathcal{T}_i \mid q, \text{ctx}) \quad \text{s.t.} \quad \sum_{\mathcal{T}_i \in \mathcal{S}} c_i \leq B_{\text{tools}}
$$

This is a variant of the **0-1 knapsack problem** (NP-hard in general, but tractable for small $k$ via greedy approximation with guaranteed $1 - 1/e$ optimality ratio).

```
Algorithm 1: Lazy Tool Loading with Budget-Constrained Selection
─────────────────────────────────────────────────────────────────
Input: query q, registry R, token budget B_tools
Output: T_active (set of tool schemas to inject)

1.  e_q ← Embed(q)
2.  T_candidates ← ANN(e_q, {e_δ_i}_{i=1}^N, k=15)
3.  for each T_i in T_candidates:
4.      s_i ← α·sim(e_q, e_δ_i) + β·P_success(T_i, q) − γ·κ_i − λ·𝟙[mutating]
5.  Sort T_candidates by s_i descending
6.  T_active ← ∅; budget_used ← 0
7.  for each T_i in sorted T_candidates:
8.      if budget_used + c_i ≤ B_tools:
9.          T_active ← T_active ∪ {T_i}
10.         budget_used ← budget_used + c_i
11. return T_active
```

**Complexity**: $O(N)$ for embedding pre-computation (offline), $O(\log N)$ for ANN retrieval, $O(k \log k)$ for re-ranking. Total per-turn cost: **sub-millisecond**.

### 3.3 Tool Description as Compiled Directive

The tool description $\delta_i$ is the **primary signal** the LLM uses for tool selection. It must be engineered as a **compiled directive**, not prose documentation.

**SOTA Description Template:**

```
[WHAT] {Precise capability in one sentence.}
[WHEN] {Exact trigger conditions — user intents that SHOULD activate this tool.}
[WHEN NOT] {Explicit negative triggers — intents that SHOULD NOT activate this tool, with redirect to correct tool.}
[INPUTS] {Critical parameter semantics beyond what schema types convey.}
[OUTPUTS] {What the return payload represents; how to interpret absence/emptiness.}
[SIDE EFFECTS] {State mutations, if any. "None" for read-only tools.}
[CHAIN HINTS] {Which tools commonly precede or follow this one in multi-step workflows.}
```

**Example (SOTA-compiled):**

```
[WHAT] Search real-time flight availability between two airports on a specific date.
[WHEN] User asks to find, compare, or check availability of flights.
[WHEN NOT] Do NOT use for hotel search (→ search_hotels), car rental (→ search_cars), or flight booking/purchase (→ book_flight). Do NOT use if user already has flight details and wants to modify (→ modify_booking).
[INPUTS] origin/destination must be valid 3-letter IATA codes. If user provides city name, resolve to IATA code using resolve_airport_code FIRST.
[OUTPUTS] Returns ranked list of flight options with price, duration, stops, carrier. Empty list means no availability — inform user and suggest adjacent dates.
[SIDE EFFECTS] None (read-only).
[CHAIN HINTS] Often preceded by resolve_airport_code. Often followed by book_flight or search_hotels.
```

**Theorem 3.2 (Description Disambiguation Necessity).**
Given $n$ tools with overlapping semantic domains, the probability of tool misselection increases polynomially with domain overlap $\omega_{ij}$ between tool pairs $(i, j)$:

$$
P(\text{misselect}) \propto \sum_{i < j} \omega_{ij}^2
$$

Negative specifications (`[WHEN NOT]`) reduce $\omega_{ij}$ by providing **contrastive signal** that sharpens the decision boundary in the LLM's inference space. Empirically, adding negative specifications reduces tool misselection by **35–60%** on benchmarks like ToolBench and API-Bank.

---

## 4. The Evolution: From Prompt Hacking to Typed Function Calling

### 4.1 Generation Taxonomy

The evolution of tool invocation in LLM systems follows a strict capability ladder:

| Generation | Mechanism | Failure Mode | Reliability |
|---|---|---|---|
| **Gen-0: Regex Extraction** | Prompt instructs LLM to emit text matching a regex pattern (e.g., `ACTION: search("tokyo")`) | Fragile parsing; format drift; no type safety | ~60% parse success |
| **Gen-1: ReAct-Style Prompting** | Chain-of-thought + action/observation loop with text-based tool calls | Better reasoning but still text-parsed; hallucinated tool names/args | ~75% |
| **Gen-2: Native Function Calling** | LLM outputs structured JSON with function name and typed arguments as a **dedicated output mode** distinct from text generation | Mismatched schemas; argument type errors | ~90% |
| **Gen-3: Typed Protocol Binding (SOTA)** | Function calling + MCP discovery + schema validation + idempotency + approval gates + provenance | Requires infrastructure investment | ~98%+ with verification |

### 4.2 Native Function Calling: Formal Model

In Gen-2+ systems, the LLM operates as a **policy function** $\pi_\theta$ that, given context $\mathbf{c}$, produces either a **text response** $r$ or a **tool call** $f$:

$$
\pi_\theta(\mathbf{c}) \to \begin{cases} r \in \Sigma^* & \text{(text response)} \\ f = (\texttt{tool\_id}, \; \texttt{args} \in \Sigma_{\text{in}}) & \text{(tool invocation)} \end{cases}
$$

The training objective for function calling fine-tunes the model to emit valid JSON conforming to the declared schemas with high probability:

$$
\mathcal{L}_{\text{fc}} = -\mathbb{E}_{(q, \mathcal{T}_{\text{active}})}\left[\log P_\theta\!\left(f^* \mid q, \; \{\delta_i, \Sigma_{\text{in},i}\}_{i \in \mathcal{T}_{\text{active}}}\right)\right]
$$

where $f^*$ is the ground-truth tool call.

**Critical distinction from text generation**: The output mode switch is a **structured decode** — the model's logits are constrained (via guided/constrained decoding or fine-tuned output heads) to produce JSON tokens that conform to the declared input schema. This is fundamentally different from hoping the model "guesses" the right format.

### 4.3 Parallel and Chained Tool Calling

Modern SOTA models support **multiple tool calls per turn**, enabling:

**Parallel Invocation** (independent tools, no data dependency):

$$
\{f_1, f_2, \ldots, f_m\} = \pi_\theta(\mathbf{c}) \quad \text{where} \quad \forall i \neq j: \; \text{args}(f_j) \not\!\perp\!\!\!\perp \text{output}(f_i)
$$

All $m$ calls execute concurrently. Results are aggregated and re-injected into context.

**Sequential Chaining** (data-dependent tools):

$$
f_1 = \pi_\theta(\mathbf{c}_0), \quad \mathbf{c}_1 = \mathbf{c}_0 \oplus \text{result}(f_1), \quad f_2 = \pi_\theta(\mathbf{c}_1), \quad \ldots
$$

Each step's output enriches the context for the next decision. The chain length $L$ is **bounded**:

$$
L \leq L_{\max} \quad \text{(recursion depth bound to prevent runaway execution)}
$$

---

## 5. Tool Invocation as a Bounded Control Loop

### 5.1 The Complete Tool Execution Cycle

Tool invocation is **never** a fire-and-forget operation. It is embedded within a **closed-loop control system** with verification, error handling, and repair:

```
Algorithm 2: Tool Invocation Control Loop
──────────────────────────────────────────
Input: agent context c, tool registry R, max_iterations L_max
Output: final response r or failure state F

1.  iteration ← 0
2.  while iteration < L_max:
3.      // PLAN: Determine if a tool call is needed
4.      action ← π_θ(c)
5.      if action is TextResponse r:
6.          return r   // Terminal: agent has enough information
7.      
8.      // VALIDATE: Schema-check the proposed tool call
9.      f = (tool_id, args) ← action
10.     if tool_id ∉ R:
11.         c ← c ⊕ Error("Tool not found: {tool_id}. Available: {R.list()}")
12.         iteration += 1; continue
13.     if not validate(args, Σ_in[tool_id]):
14.         c ← c ⊕ Error("Invalid arguments: {validation_errors}")
15.         iteration += 1; continue
16.     
17.     // AUTHORIZE: Check permissions and approval gates
18.     if T[tool_id].π requires_approval and not human_approved(f):
19.         c ← c ⊕ PendingApproval(f)
20.         yield HumanApprovalRequest(f)  // Suspend execution
21.         // Resume on approval or rejection
22.     
23.     // EXECUTE: Invoke with deadline, idempotency, and circuit breaker
24.     idem_key ← generate_idempotency_key(f, c.session_id)
25.     result ← execute_with_retry(
26.         tool_id, args, idem_key,
27.         deadline=T[tool_id].τ_max,
28.         retry_budget=3,
29.         backoff=exponential_with_jitter
30.     )
31.     
32.     // VERIFY: Check result validity
33.     if result is ToolError:
34.         if result.retryable and retry_budget > 0:
35.             retry_budget -= 1; continue
36.         c ← c ⊕ ErrorContext(result)
37.         iteration += 1; continue
38.     
39.     // OBSERVE: Inject result with provenance into context
40.     c ← c ⊕ ToolResult(
41.         tool_id, result.content, result.provenance,
42.         timestamp=now(), latency=result.cost.latency
43.     )
44.     
45.     // CRITIQUE: Let the agent evaluate whether the result is sufficient
46.     iteration += 1
47.     // Loop continues — agent will decide to call another tool or respond
48.
49.  // Exceeded iteration bound
50.  return FailureState("Max iterations reached", context=c)
```

### 5.2 Formal Properties of the Control Loop

**Property 1: Termination Guarantee.**
The loop terminates in at most $L_{\max}$ iterations. This prevents unbounded tool-call chains that consume tokens, cost, and latency without convergence.

$$
\forall \text{execution}: \; |\text{iterations}| \leq L_{\max}
$$

**Property 2: Idempotency.**
Every state-mutating tool call carries an idempotency key $k = H(\texttt{tool\_id} \| \texttt{args} \| \texttt{session\_id})$. Retries with the same key produce identical effects:

$$
\text{execute}(f, k) = \text{execute}(f, k) \quad \forall \text{retries}
$$

**Property 3: Monotonic Context Enrichment.**
Each iteration either adds a tool result (information gain) or an error context (diagnostic gain) to $\mathbf{c}$. The information content of context is non-decreasing:

$$
H(\mathbf{c}_{t+1} \mid q) \geq H(\mathbf{c}_t \mid q)
$$

**Property 4: Human Interruptibility.**
Any state-mutating path with `requires_approval = true` **suspends** execution and yields control to a human operator. The agent cannot bypass this gate.

---

## 6. Tool Chaining: Multi-Step Workflow Orchestration

### 6.1 Dependency Graph Formalization

A complex user request decomposes into a **directed acyclic graph (DAG)** of tool invocations:

**Definition 6.1 (Tool Execution DAG).**
Given a task $T$, the execution plan is a DAG $G = (V, E)$ where:
- $V = \{v_1, \ldots, v_n\}$ are tool invocation nodes
- $E \subseteq V \times V$ are data-dependency edges: $(v_i, v_j) \in E$ iff $v_j$ requires output from $v_i$

The **critical path latency** of the plan is:

$$
\Lambda(G) = \max_{\text{path } p \in G} \sum_{v \in p} \tau(v)
$$

where $\tau(v)$ is the expected latency of tool $v$.

**Example: "Plan a weekend trip to San Francisco"**

```
                    ┌──────────────────┐
                    │  resolve_city    │
                    │  "San Francisco" │
                    │  → IATA: SFO     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌─────────────┐
     │search_flights│ │search_hotels│ │get_local    │
     │origin=USER  │  │city=SFO    │  │  _events    │
     │dest=SFO     │  │checkin=FRI │  │city=SFO     │
     │date=FRI     │  │checkout=SUN│  │dates=FRI-SUN│
     └──────┬─────┘  └──────┬─────┘  └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ synthesize_plan │
                   │ (LLM reasoning) │
                   └────────────────┘
```

**Parallelism Extraction**: `search_flights`, `search_hotels`, and `get_local_events` are **independent** given the resolved city. They execute **concurrently**, reducing wall-clock latency from $\sum \tau_i$ to $\max(\tau_i)$.

### 6.2 Automatic Plan Decomposition

```
Algorithm 3: Task Decomposition and Tool-Chain Planning
───────────────────────────────────────────────────────
Input: user query q, tool registry R, agent policy π_θ
Output: execution DAG G = (V, E)

1.  // DECOMPOSE: Break query into atomic sub-tasks
2.  subtasks ← π_θ(
3.      system="Decompose the user request into atomic sub-tasks. "
4.             "Each sub-task must map to exactly one tool or a synthesis step. "
5.             "Specify data dependencies between sub-tasks.",
6.      user=q,
7.      tools=T_active   // Lazy-loaded relevant subset
8.  )
9.  // Parse structured decomposition output
10. V ← {v_i : subtask_i ∈ subtasks}
11. E ← {(v_i, v_j) : v_j.depends_on contains v_i}
12.
13. // VALIDATE: Check DAG properties
14. assert is_acyclic(V, E)          // No circular dependencies
15. assert ∀ v ∈ V: v.tool_id ∈ R   // All tools exist
16. assert |V| ≤ V_max              // Bounded plan complexity
17.
18. // OPTIMIZE: Identify parallelizable groups
19. levels ← topological_sort_levels(V, E)
20. for each level L in levels:
21.     // Tools within the same level have no inter-dependencies
22.     // → Execute concurrently
23.     mark_parallel(L)
24.
25. return G = (V, E, levels)
```

### 6.3 Execution Engine with Concurrency Control

```
Algorithm 4: DAG Executor with Parallel Tool Dispatch
─────────────────────────────────────────────────────
Input: DAG G = (V, E, levels), context c
Output: aggregated results R_all, updated context c'

1.  R_all ← {}
2.  for each level L in topological order:
3.      // Dispatch all tools in this level concurrently
4.      futures ← {}
5.      for each v in L:
6.          // Resolve arguments: substitute outputs from predecessor nodes
7.          resolved_args ← resolve_references(v.args, R_all)
8.          fut ← async_execute(v.tool_id, resolved_args,
9.                              deadline=v.τ_max, idem_key=v.idem_key)
10.         futures[v.id] ← fut
11.     
12.     // Await all with deadline; handle partial failures
13.     results ← await_all_with_timeout(futures, deadline=max(v.τ_max for v in L))
14.     for each (v_id, result) in results:
15.         if result is Error:
16.             // Compensating action: retry, fallback, or skip with degraded quality
17.             result ← handle_tool_failure(v_id, result, compensation_policy)
18.         R_all[v_id] ← result
19.     
20.  // Inject all results into context with provenance
21.  c' ← c ⊕ {ToolResult(v_id, R_all[v_id]) for v_id in V}
22.  return R_all, c'
```

**Key Properties:**

- **Maximum parallelism**: Independent tools in the same topological level execute concurrently
- **Minimum latency**: Critical path latency $\Lambda(G)$ is achieved, not sequential sum
- **Fault isolation**: Failure of one tool in a level does not block siblings; compensating actions are applied per-node

---

## 7. Reliability Engineering for Tool Invocation

### 7.1 Retry Budget with Exponential Backoff and Jitter

For transient failures (network timeouts, rate limits), apply bounded retries:

$$
t_{\text{wait}}(n) = \min\!\left(\text{base} \cdot 2^n + \text{Uniform}(0, \text{jitter}), \; t_{\max}\right)
$$

where $n$ is the retry attempt number. The **retry budget** $R_b$ is the maximum number of retries per tool call. Total maximum latency per call:

$$
\tau_{\text{total}} \leq \tau_{\text{call}} + \sum_{n=1}^{R_b} t_{\text{wait}}(n) + \tau_{\text{call}}
$$

### 7.2 Circuit Breaker Pattern

For persistent failures (upstream service down), apply a **circuit breaker** to prevent cascading failures:

$$
\text{state}(\mathcal{T}_i) = \begin{cases}
\texttt{CLOSED} & \text{if } \text{fail\_rate}(\mathcal{T}_i, w) < \theta_{\text{open}} \\
\texttt{OPEN} & \text{if } \text{fail\_rate}(\mathcal{T}_i, w) \geq \theta_{\text{open}} \\
\texttt{HALF\_OPEN} & \text{after cooldown } \Delta t_{\text{cool}}
\end{cases}
$$

where $w$ is the observation window and $\theta_{\text{open}}$ is the failure-rate threshold (e.g., 50% over 60 seconds). In `OPEN` state, all calls to $\mathcal{T}_i$ are **immediately rejected** with a `CIRCUIT_OPEN` error, allowing the agent to select fallback tools or degrade gracefully.

### 7.3 Hallucination Control in Tool Usage

Tool-related hallucinations fall into three categories:

| Hallucination Type | Definition | Mitigation |
|---|---|---|
| **Phantom Tool** | LLM invokes a tool not in $\mathcal{R}$ | Schema-constrained decoding; validate `tool_id ∈ R` before execution |
| **Argument Fabrication** | LLM fills required arguments with plausible but invented values | JSON Schema validation; reject calls with `INVALID_ARGUMENTS`; re-prompt with explicit error |
| **Result Confabulation** | LLM ignores actual tool output and generates a plausible-sounding answer from parametric knowledge | Architectural enforcement: tool results injected as **system-role messages** with provenance tags; verification step compares response claims against tool output |

**Formal Mitigation: Grounded Response Verification**

After tool execution and response generation, apply a **grounding check**:

$$
\text{grounded}(r, \mathcal{O}) = \frac{|\text{claims}(r) \cap \text{facts}(\mathcal{O})|}{|\text{claims}(r)|}
$$

where $r$ is the agent's response and $\mathcal{O}$ is the set of tool outputs. If $\text{grounded}(r, \mathcal{O}) < \theta_{\text{ground}}$ (e.g., 0.9), the response is **rejected** and regenerated with a strengthened grounding instruction.

```
Algorithm 5: Grounded Response Verification
────────────────────────────────────────────
Input: response r, tool outputs O, threshold θ_ground
Output: verified response r' or rejection

1.  claims ← extract_factual_claims(r)  // NLI-based claim extraction
2.  for each claim c in claims:
3.      supported ← FALSE
4.      for each output o in O:
5.          if entails(o, c):   // NLI entailment check
6.              supported ← TRUE; break
7.      if not supported:
8.          mark_ungrounded(c)
9.  
10. grounding_score ← |supported_claims| / |claims|
11. if grounding_score < θ_ground:
12.     // Regenerate with explicit grounding instruction
13.     r' ← π_θ(c ⊕ "Your response contained ungrounded claims. "
14.                    "Use ONLY the tool outputs below. Do not add information "
15.                    "not present in these results." ⊕ O)
16.     return verify(r', O, θ_ground)  // Recursive, bounded
17. return r
```

---

## 8. Authorization, Least Privilege, and Human Governance

### 8.1 Scoped Authorization Model

Tools are bound to **caller-scoped authorization**, not broad agent credentials:

$$
\text{authorized}(f, \text{agent}, \text{user}) = \text{scope}(\text{user}) \cap \text{requires}(\mathcal{T}_{f.\text{id}}) \neq \emptyset
$$

The agent **inherits** the invoking user's permission set. It cannot escalate privileges.

### 8.2 Mutation Classification and Approval Gates

Every tool is classified on a **mutation severity spectrum**:

| Class | Example | Approval Policy |
|---|---|---|
| `read-only` | `search_flights`, `get_weather` | Auto-approved |
| `state-mutating-reversible` | `add_to_cart`, `create_draft` | Auto-approved with undo window |
| `state-mutating-irreversible` | `book_flight`, `send_payment` | **Human approval required** |
| `destructive` | `delete_account`, `cancel_subscription` | **Explicit double-confirmation** |

```
Algorithm 6: Mutation-Aware Tool Gating
───────────────────────────────────────
Input: tool call f, mutation_class π
Output: execution permission or suspension

1.  switch π:
2.      case READ_ONLY:
3.          return PERMIT
4.      case MUTATING_REVERSIBLE:
5.          log_with_undo_window(f, undo_ttl=300s)
6.          return PERMIT
7.      case MUTATING_IRREVERSIBLE:
8.          summary ← generate_human_summary(f)
9.          approval ← request_human_approval(summary, timeout=600s)
10.         if approval == APPROVED:
11.             return PERMIT
12.         elif approval == TIMEOUT:
13.             return DENY(reason="Approval timeout")
14.         else:
15.             return DENY(reason=approval.reason)
16.     case DESTRUCTIVE:
17.         require_double_confirmation(f)
18.         return conditional_permit()
```

---

## 9. Observability and Audit Infrastructure

### 9.1 Trace Structure for Tool Invocations

Every tool call produces a **structured trace span** compliant with OpenTelemetry:

$$
\text{Span}(\mathcal{T}_i) = \langle \texttt{trace\_id}, \; \texttt{span\_id}, \; \texttt{parent\_span}, \; \texttt{tool\_id}, \; \texttt{args\_hash}, \; \texttt{start}, \; \texttt{end}, \; \texttt{status}, \; \texttt{result\_hash}, \; \texttt{cost} \rangle
$$

**Metrics collected per tool:**

| Metric | Aggregation | Alert Threshold |
|---|---|---|
| `tool.invocation.count` | Counter per tool_id, per agent, per user | Anomaly detection on rate |
| `tool.latency.p50/p95/p99` | Histogram | $p_{99} > 2 \times \tau_{\max}$ |
| `tool.error.rate` | Rate per window | $> \theta_{\text{open}}$ triggers circuit breaker |
| `tool.misselection.rate` | Counter (detected via critique step) | $> 10\%$ triggers description rewrite |
| `tool.token.cost` | Sum of schema tokens injected | Budget overshoot alert |
| `tool.result.grounding` | Grounding score distribution | $\text{mean} < 0.85$ triggers pipeline review |

### 9.2 Audit Trail for Compliance

All tool invocations are persisted in an **append-only audit log**:

```
{
  "timestamp": "2025-01-15T14:23:07.831Z",
  "trace_id": "abc123...",
  "agent_id": "travel-planner-v3",
  "user_id": "user_9182",
  "tool_id": "flights/search@v2.3",
  "arguments": { "origin": "JFK", "destination": "SFO", "date": "2025-01-24" },
  "arguments_hash": "sha256:e3b0c442...",
  "result_summary": "5 flights found, cheapest $189",
  "result_hash": "sha256:d7a8fbb3...",
  "latency_ms": 342,
  "cost_usd": 0.002,
  "mutation_class": "read-only",
  "approval_required": false,
  "idempotency_key": "idem_8f3a...",
  "status": "SUCCESS"
}
```

---

## 10. Cost Optimization and Token Efficiency

### 10.1 Schema Compression

Tool schemas injected into context are **compressed** to minimize token consumption while preserving selection signal:

**Technique: Progressive Disclosure**

```
Level 0 (Discovery):     tool_name + one-line description       (~20 tokens)
Level 1 (Selection):     + required params + trigger conditions  (~80 tokens)
Level 2 (Invocation):    Full schema with all params, types,     (~200 tokens)
                          constraints, examples
```

At **Phase 1** (intent filtering), inject Level 0 descriptions for all candidates. At **Phase 2** (after selection), inject Level 2 only for the selected tool. Token savings:

$$
\Delta C = \sum_{i \in \mathcal{T}_{\text{candidates}} \setminus \mathcal{T}_{\text{active}}} (c_i^{L2} - c_i^{L0})
$$

For 15 candidates with 1–3 selected: savings of **~2,500 tokens per turn**.

### 10.2 Result Caching

Tool results are cached with content-addressed keys and TTL:

$$
\text{key} = H(\texttt{tool\_id} \| \texttt{canonical\_args})
$$

$$
\text{cache\_hit}(f) = \begin{cases}
\text{cached\_result} & \text{if } \text{age}(\text{cached\_result}) < \text{TTL}(\mathcal{T}_{f.\text{id}}) \\
\bot & \text{otherwise}
\end{cases}
$$

TTL is set per tool based on data volatility:

| Tool Category | TTL | Rationale |
|---|---|---|
| Static reference data | 24h | Airport codes, currency symbols |
| Semi-dynamic | 15min | Hotel availability, event listings |
| Real-time | 0 (no cache) | Stock prices, flight status |

---

## 11. End-to-End Prefill Compilation with Tool Context

The **prefill compiler** assembles the complete agent context as a deterministic artifact:

```
Algorithm 7: Prefill Compilation with Tool Affordances
──────────────────────────────────────────────────────
Input: query q, session s, memory M, retrieval R, policy P
Output: compiled context c (token-budgeted)

1.  // Allocate token budgets
2.  W ← model.context_window              // e.g., 128K
3.  B_sys ← 500                            // System policy
4.  B_res ← W * 0.25                       // Reasoning reserve (25%)
5.  B_remaining ← W - B_sys - B_res
6.
7.  // Phase 1: System policy (always included, highest priority)
8.  c ← compile_system_policy(P)           // Role, constraints, output format
9.
10. // Phase 2: Tool affordances (lazy-loaded)
11. T_active ← lazy_tool_load(q, R, B_tools=min(B_remaining*0.15, 3000))
12. c ← c ⊕ compile_tool_schemas(T_active, level=2)
13. B_remaining -= token_count(T_active)
14.
15. // Phase 3: Memory summaries
16. mem_summary ← compile_memory(M, budget=min(B_remaining*0.10, 1000))
17. c ← c ⊕ mem_summary
18. B_remaining -= token_count(mem_summary)
19.
20. // Phase 4: Retrieved evidence (provenance-tagged)
21. evidence ← retrieve_and_rank(q, budget=min(B_remaining*0.30, 4000))
22. c ← c ⊕ compile_evidence(evidence)
23. B_remaining -= token_count(evidence)
24.
25. // Phase 5: Conversation history (compressed, most recent first)
26. history ← compress_history(s.messages, budget=B_remaining)
27. c ← c ⊕ history
28.
29. // Phase 6: Current query
30. c ← c ⊕ format_query(q)
31.
32. assert token_count(c) ≤ W - B_res    // Hard invariant
33. return c
```

**Token Budget Allocation (128K window):**

```
┌──────────────────────────────────────────────────┐
│ System Policy          │    500 tokens  (0.4%)   │
│ Tool Schemas (active)  │  3,000 tokens  (2.3%)   │
│ Memory Summaries       │  1,000 tokens  (0.8%)   │
│ Retrieved Evidence     │  4,000 tokens  (3.1%)   │
│ Conversation History   │ 87,500 tokens  (68.4%)  │
│ Current Query          │  ~500 tokens   (0.4%)   │
│ ─── Reasoning Reserve ─│ 32,000 tokens  (25.0%)  │
└──────────────────────────────────────────────────┘
```

---

## 12. Production Deployment Architecture

```
                           ┌────────────────────┐
                           │   Client / UI      │
                           │  (JSON-RPC 2.0)    │
                           └────────┬───────────┘
                                    │
                           ┌────────▼───────────┐
                           │  API Gateway        │
                           │  Rate limit, AuthN  │
                           │  Trace injection    │
                           └────────┬───────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Agent Orchestrator           │
                    │   (Control loop: Algo 2)       │
                    │   Plan → Act → Verify → Commit │
                    └──┬────────┬────────┬──────────┘
                       │        │        │
              ┌────────▼──┐  ┌──▼─────┐  ┌▼──────────────┐
              │  Prefill   │  │ Tool   │  │  Memory       │
              │  Compiler  │  │ Router │  │  Manager      │
              │  (Algo 7)  │  │(Algo 1)│  │  (write/read) │
              └────────────┘  └──┬─────┘  └───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   MCP Tool Gateway       │
                    │   Schema validation      │
                    │   Auth scoping           │
                    │   Circuit breaker        │
                    │   Idempotency cache      │
                    └──┬──────┬──────┬────────┘
                       │      │      │
              ┌────────▼┐  ┌──▼───┐  ┌▼──────────┐
              │ Tool     │  │ Tool │  │ Tool      │
              │ Server A │  │ Srv B│  │ Server C  │
              │ (gRPC)   │  │(gRPC)│  │ (MCP/SSE) │
              └──────────┘  └──────┘  └───────────┘
```

### 12.1 Operational Constraints

| Constraint | Target | Mechanism |
|---|---|---|
| **Latency** | Tool selection < 50ms; total tool chain < 10s | ANN index, pre-computed embeddings, parallel dispatch |
| **Cost** | < $0.05 per agent turn (tool overhead) | Schema compression, caching, lazy loading |
| **Reliability** | 99.9% successful tool resolution | Circuit breakers, fallback tools, retry budgets |
| **Token Efficiency** | Tool context < 5% of window | Progressive disclosure, budget-constrained selection |
| **Safety** | Zero unauthorized mutations | Scoped auth, approval gates, audit log |
| **Observability** | 100% trace coverage | OpenTelemetry, structured audit log |

---

## 13. Summary: Tool Integration Quality Gates

Before any tool integration reaches production, it must pass the following **quality gates**:

| Gate | Criterion | Automated Check |
|---|---|---|
| **Schema Validity** | Input/output schemas parse without error; all required fields documented | JSON Schema meta-validation in CI |
| **Description Quality** | Contains [WHAT], [WHEN], [WHEN NOT], [INPUTS], [OUTPUTS], [SIDE EFFECTS] | Regex + LLM-judge scoring > 0.8 |
| **Disambiguation** | Pairwise similarity with sibling tools < 0.7 cosine | Embedding comparison in CI |
| **Latency Compliance** | p99 latency within declared tier | Load test with 100 synthetic queries |
| **Error Handling** | All error classes return typed ToolError; retryable flag accurate | Fault injection testing |
| **Idempotency** | Duplicate calls with same key produce identical state | Replay test suite |
| **Authorization** | Caller scope checked; escalation impossible | Penetration test + formal policy verification |
| **Grounding** | Agent responses using this tool achieve grounding > 0.9 | Eval suite with known-answer queries |
| **Cost Bound** | Per-call cost within declared budget | Cost monitoring in staging |

---

**End of Report.**

This document specifies tools not as informal API wrappers, but as **typed, governed, budget-constrained, observable infrastructure** within a closed-loop agentic control system. Every interface is schema-described, every invocation is traced, every mutation is gated, and every result is verified — the foundational requirements for production-grade agentic AI at enterprise scale.