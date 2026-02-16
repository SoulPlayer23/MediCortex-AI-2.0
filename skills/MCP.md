# Model Context Protocol (MCP) Standards

**Domain:** AI / Tool Engineering
**Skill:** MCP Server & Tool Development
**Context:** Building standardized, secure, and interoperable "tools" that allow any MCP-compliant agent (Claude, IDEs, Custom Agents) to interact with external data and systems.

---

## 1. Architecture & Components

*Understanding how agents "see" and "touch" the outside world via MCP.*

### 1.1 The Host-Client-Server Model

* **Standard:** Strictly adhere to the separation of concerns:
* **Host (The Brain):** The application running the LLM (e.g., Claude Desktop, Cursor, or a custom Agent Runtime).
* **Client (The Connector):** The internal component within the Host that speaks the MCP protocol.
* **Server (The Tool Provider):** The standalone process (local or remote) that exposes the actual capabilities.


* **Requirement:** Build **MCP Servers**, not custom API wrappers.
* *Why?* An MCP Server built once works instantly with Claude, IDEs, and any future MCP-compliant agent without code changes.



### 1.2 Transport Mechanisms

* **Standard:** Choose the transport based on deployment context:
* **STDIO (Local):** For agents running locally (e.g., on a developer's machine). The Host spawns the Server as a subprocess and communicates via `stdin/stdout`.
* **SSE (Server-Sent Events) over HTTP (Remote):** For production agents accessing remote services. Use SSE for server-to-client messages (events) and HTTP POST for client-to-server requests.



---

## 2. Tool Design & Schema Strategy

*Designing tools that LLMs can understand and use reliably.*

### 2.1 The "Description is Code" Principle

* **Standard:** The `description` field in your tool definition is **not** a comment; it is the primary prompt instruction for the LLM.
* **Practice:**
* **Weak:** `description: "Updates the user."`
* **Strong:** `description: "Updates a user's profile data. Requires 'user_id'. Only 'email' and 'phone' fields are mutable. Returns the updated user object."`


* **Validation:** Use `zod` (TypeScript) or Pydantic (Python) to generate the `inputSchema`. The MCP protocol requires strict JSON Schema drafts (v2020-12).

### 2.2 Tool Granularity

* **Standard:** Avoid "God Tools". Break complex operations into atomic, distinct tools.
* *Anti-Pattern:* `manage_database(action: "create" | "delete" | "query", ...)`
* *Standard:* `create_table(...)`, `delete_record(...)`, `execute_query(...)`.


* **Benefit:** Reduces hallucination by narrowing the LLM's action space per turn.

### 2.3 Error handling as Information

* **Standard:** When a tool fails, return a structured result with `isError: true` and a descriptive text message—**do not throw an exception** that crashes the server.
* **Practice:** Return the error *to the model* so it can self-correct.
* *Example:* "Error: File not found. The directory '/data' is empty." (The model effectively reads this and might try creating the file next).



---

## 3. Beyond Tools: Resources & Prompts

*MCP is not just for function calling. Use the full triad of primitives.*

### 3.1 Resources (Passive Data)

* **Standard:** Use **Resources** for reading data, not Tools.
* *Scenario:* Reading a log file or a database schema.
* *Why:* Resources (`resources/read`) allow the Client to subscribe to updates. If the log file changes, the Server sends a notification, and the Host (Agent) instantly sees the new data without polling.


* **URI Scheme:** Define custom URI schemes for your resources (e.g., `postgres://users/schema`, `logs://app/error.log`).

### 3.2 Prompts (Templates)

* **Standard:** Expose **Prompts** (`prompts/list`) to standardize how agents interacting with your server should behave.
* *Use Case:* A "Git MCP Server" should expose a `commit-message-generator` prompt that contains the team's specific commit style guide.
* *Benefit:* Ensures that any agent using your server follows your team's best practices automatically.



---

## 4. Security & Human-in-the-Loop

*Protecting the system from rogue agent actions.*

### 4.1 Sampling & Consent

* **Standard:** For high-stakes actions (e.g., `delete_database`, `transfer_funds`), the Server generally does not ask for confirmation—the **Host** (Client) is responsible for the "Human in the Loop" UI.
* **Server Responsibility:** Accurately mark strictly sensitive tools so the Host knows to prompt the user. (Note: MCP protocol is evolving to include specific "risk" capabilities).

### 4.2 Input Validation

* **Standard:** **Never trust the LLM.**
* The Server must validate all inputs against the schema *before* execution.
* Sanitize file paths (prevent `../../etc/passwd` directory traversal attacks).
* Sanitize SQL/Shell commands to prevent injection attacks.



---

## 5. Development & Observability

*Building and debugging MCP servers.*

### 5.1 The MCP Inspector

* **Standard:** Developers **must** use the `mcp-inspector` (or the Inspector in Claude Desktop) to test tools during development.
* **Workflow:** Do not test by "chatting" with an LLM immediately. Use the Inspector to manually invoke `tools/call` and verify the JSON-RPC response structure first.

### 5.2 Logging

* **Standard:** Use the MCP Logging primitive (`notifications/message`) to send logs from the Server to the Client.
* **Levels:**
* `debug`: Internal logic flow (for developers).
* `info`: "Tool 'search_users' called with query 'Alice'." (for audit trails).
* `error`: "Database connection failed."



---

## Checklist for Code Review

Before merging an MCP Server:

* [ ] **Schema:** Does every tool have a valid JSON Schema for `inputSchema`?
* [ ] **Descriptions:** Are tool descriptions prompt-engineered (clear, instructional)?
* [ ] **Primitives:** Are read-only operations exposed as **Resources**, not Tools?
* [ ] **Safety:** Are file path inputs sanitized to prevent directory traversal?
* [ ] **Error Handling:** Are errors returned as text content with `isError: true`?
* [ ] **Transport:** Does the server support `stdio` (for local dev) and `sse` (if remote)?
* [ ] **Testing:** Has the server been verified using the MCP Inspector?