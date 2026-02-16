# LangChain & LangGraph Development

**Domain:** AI / LLM Application Engineering  
**Skill:** Stateful Agentic Workflows  
**Context:** Building reasoning engines, RAG pipelines, and multi-step agents that require complex state management, cyclic logic (loops), and human intervention.

---

## 1. Orchestration & Flow Control (LangGraph)
*The standard for defining application logic. If it loops, it must be a Graph.*



### 1.1 The Graph Topology
* **Standard:** Use **`StateGraph`** for all agentic workflows.
    * **Nodes:** Python functions that receive the current state and return a state update (partial dictionary).
    * **Edges:** Define the transition logic. Use `add_conditional_edges` for dynamic routing (e.g., "If tool called -> go to ToolNode, else -> END").
* **Anti-Pattern:** Using legacy `AgentExecutor` or `SequentialChain`. These hide the control flow and are hard to debug.
* **Requirement:** Every graph must have a clearly defined `START` node and at least one path to `END`.

### 1.2 State Schema & Reducers
* **Standard:** Explicitly define your state using `TypedDict` or Pydantic.
* **Immutability vs. Reduction:**
    * LangGraph uses "reducers" (e.g., `operator.add` for lists of messages).
    * **Rule:** Your node functions should return *only* the keys they want to update, not the entire state object.
    * *Example:* `return {"messages": [new_message]}` (appends to list) vs. `return {"count": 5}` (overwrites value).

### 1.3 Persistence & "Time Travel"
* **Standard:** Use a **Checkpointer** (e.g., `MemorySaver` for dev, `PostgresSaver` for prod) to persist state between user turns.
    * **Thread ID:** Always pass a `thread_id` in the `config` to resume a conversation.
* **Human-in-the-Loop:** Use `interrupt_before=["action_node"]` to pause execution for user approval. This is the only safe way to implement "Ask for permission" workflows.

---

## 2. Composition Syntax (LangChain Core)
*How to construct the internal logic of a Node using LCEL.*



### 2.1 LCEL (LangChain Expression Language)
* **Standard:** Use the pipe `|` syntax for composing chains.
    * *Pattern:* `prompt | model | output_parser`.
* **Runnables:** Wrap custom logic in `RunnableLambda`. Use `RunnablePassthrough` to pass data through the chain unchanged (crucial for RAG inputs).
* **Anti-Pattern:** Subclassing `Chain` or using the deprecated `LLMChain` class.

### 2.2 Streaming & Async
* **Standard:** All chains inside a LangGraph node must be invoked asynchronously (`await chain.ainvoke(...)`).
* **Streaming:** Use `.astream_events()` (v2) to stream intermediate steps (tokens, tool calls) to the frontend. Avoid the older `CallbackHandler` based streaming where possible.

---

## 3. Retrieval Augmented Generation (RAG)
*Standards for feeding data to the LLM.*

### 3.1 Document Loading & Splitting
* **Standard:**
    * **Chunking:** Use `RecursiveCharacterTextSplitter`. Avoid hard character limits without overlap.
    * **Overlap:** Always set a `chunk_overlap` (e.g., 10-20%) to preserve context at boundaries.
* **Metadata:** Always preserve source metadata (page number, filename) in the `Document` object so the LLM can cite sources.

### 3.2 Retriever Construction
* **Standard:** Decouple retrieval from generation.
    * Use `MultiQueryRetriever` to generate alternative search terms.
    * Use `EnsembleRetriever` to combine **Keyword Search** (BM25) with **Semantic Search** (Vector) for better accuracy.
* **Anti-Pattern:** Stuffing 100 documents into the context window blindly. Use `ContextualCompressionRetriever` or reranking (Cohere/Cross-Encoder) to filter irrelevant chunks before they hit the LLM.

---

## 4. Tool Use & Structured Output
*Connecting the brain to the hands.*

### 4.1 Binding Tools
* **Standard:** Use `.bind_tools([tool_list])` on the model object.
    * Do not prompt engineer tool descriptions manually if the model supports native function calling (OpenAI/Anthropic/Gemini).
* **Execution:** Use the pre-built `ToolNode` from `langgraph.prebuilt` unless you need custom error handling logic.

### 4.2 Structured Output Parsing
* **Standard:** Use `.with_structured_output(PydanticModel)` instead of generic `JsonOutputParser`.
    * *Why:* This leverages the model's native JSON mode or function calling mode for higher reliability.
* **Validation:** If parsing fails, use a "Retry/Correction" node in your graph that feeds the error message back to the LLM.

---

## 5. Observability (LangSmith)
*If you can't trace it, you can't ship it.*



### 5.1 Tracing
* **Standard:** Set `LANGCHAIN_TRACING_V2=true` in all environments.
* **Tags & Metadata:** Add custom tags to your runs (e.g., `run_name="finance_agent"`, `metadata={"user_tier": "premium"}`) to filter traces easily in the dashboard.

### 5.2 Evaluation (Unit Tests for LLMs)
* **Standard:** Create a dataset in LangSmith (Inputs + Expected Outputs).
* **Metric:** Run automated evaluations on pull requests:
    * *Correctness:* "Does the answer match the ground truth?" (LLM-as-Judge).
    * *Hallucination:* "Is the answer supported by the retrieved context?" (RAGAS / TruLens).

---

## Checklist for Code Review
Before merging:

- [ ] **Graph:** Is `StateGraph` used instead of legacy Chains?
- [ ] **State:** Is the state schema defined with `TypedDict` and proper reducers?
- [ ] **Persistence:** Is a Checkpointer configured (memory or DB)?
- [ ] **Loop Safety:** Are there conditional edges to break infinite loops?
- [ ] **Syntax:** Is LCEL (`|`) used for chain composition?
- [ ] **RAG:** Is there a re-ranking or compression step for retrieved docs?
- [ ] **Observability:** Is LangSmith tracing enabled with descriptive tags?
- [ ] **Async:** Are all I/O bound calls using `await`?