# Agent-to-Agent (A2A) System Standards

**Domain:** AI / LLM Engineering
**Skill:** Multi-Agent Systems (A2A) Development
**Context:** Building scalable, reliable, and observable architectures where multiple specialized AI agents collaborate, communicate, and execute complex workflows.

---

## 1. Architecture & Topologies

*Defining how agents are organized to prevent chaotic, unpredictable interactions.*

### 1.1 Clear Role Definitions

* **Standard:** Agents must have strictly scoped, mutually exclusive responsibilities.
* **The "Agent Card" Standard (A2A Protocol):** Every agent must publish an "Agent Card" (metadata manifest) at a `.well-known` endpoint. This card defines:
* **Capabilities:** What the agent *can* do.
* **Input Schema:** The exact JSON structure it accepts.
* **Pricing/Quotas:** Cost per turn or rate limits.


* **Anti-Pattern:** "God Agents" with overlapping tools, leading to redundant API calls.

### 1.2 Routing and Topology Selection

* **Standard:** Choose an explicit interaction pattern:
* **Sequential (Pipeline):** Deterministic A  B  C flows.
* **Hierarchical (Supervisor/Worker):** A Router agent delegates to workers and synthesizes results.
* **Graph/State Machine:** Agents trigger based on shared state changes (e.g., LangGraph).


* **Anti-Pattern:** Unrestricted "Mesh" networking where any agent talks to any other without a defined protocol.

---

## 2. Inter-Agent Communication

*How agents pass data and negotiate tasks.*

### 2.1 Structured Data Handoffs

* **Standard:** Communication **must** use structured schemas (JSON/Pydantic), never raw natural language.
* **Requirement:** Inputs and Outputs must be validated against a schema *before* the receiving agent processes them. If validation fails, a "Validation Error" is returned to the sender immediately.

### 2.2 The Envelope Protocol

* **Standard:** Wrap messages in a standard envelope:
* `trace_id`: Global ID for the user request.
* `idempotency_key`: A unique hash for this specific step (prevents double-billing or duplicate actions on retries).
* `sender_id` / `receiver_id`: Identity verification.
* `payload`: The structured task data.



---

## 3. State & Context Management

*Handling memory efficiently to avoid token limits and hallucination loops.*

### 3.1 Global State vs. Local Memory

* **Standard:** Maintain a **Global State** (the "Project Board") separate from **Local Memory** (the agent's "Scratchpad").
* *Global:* "The code is written but not tested." (Shared truth)
* *Local:* "I tried method X and it failed, now trying Y." (Private reasoning)


* **Practice:** When handing off, Agent A summarizes its local reasoning into a clean status update for the Global State.

### 3.2 Context Pruning

* **Standard:** Implement "Context Compaction." Before a handoff, if the conversation history > N messages, an intermediate LLM call must summarize the history into a concise `previous_context` block.

---

## 4. Reliability, Control & Safety

*Preventing autonomous systems from spiraling out of control.*

### 4.1 Infinite Loop Prevention

* **Standard:** Hard circuit breakers are mandatory.
* **Max Turns:** Limit A  B exchanges to 3 turns.
* **Timeouts:** Strict execution time limits per agent.


* **Recovery:** If a limit is hit, the system must route to a **Deterministic Exit Node** (e.g., "Escalate to Human"), not crash silently.

### 4.2 Idempotency & Side Effects

* **Standard:** Agents performing side effects (sending emails, booking tickets, writing to DB) must check the `idempotency_key` before executing. If the key exists, return the cached result instead of re-executing.

### 4.3 Principle of Least Privilege

* **Standard:** Tool access is strictly role-bound. A "Researcher" agent must *never* have write access to the production database or execution access to shell terminals.

---

## 5. Observability & Evaluation

*Monitoring black-box interactions.*

### 5.1 Multi-Agent Tracing

* **Standard:** Use distributed tracing (e.g., LangSmith, Arize Phoenix). A single user request must be traceable across all agent hops using a unified `trace_id`.

### 5.2 "Model-as-Judge" Evaluation

* **Standard:** Do not trust agent outputs blindly. Implement a lightweight "Reviewer Node" (or use a cheaper model like GPT-4o-mini) to grade the output against a checklist before it is returned to the user.
* *Check:* "Does this answer actually address the user's prompt?"



---

## Checklist for Code Review

Before merging to `main`:

* [ ] **Discovery:** Does the agent publish an "Agent Card" / Schema?
* [ ] **Structure:** Are all handoffs typed with Pydantic models?
* [ ] **Safety:** Is `max_iterations` set for all loops?
* [ ] **Reliability:** Are side-effect tools protected by idempotency checks?
* [ ] **Context:** Is Global State separated from Local Scratchpads?
* [ ] **Observability:** Is the `trace_id` propagated through the entire chain?