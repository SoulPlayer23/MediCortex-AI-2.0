# MediCortex AI 2.0 — Technical Todo

Compliance gaps and improvements identified during A2A & MCP audit of all specialized agents.
These apply **system-wide** across all 5 agents.

---

## 🔴 High Priority

### ~~§5.2 (A2A) — Model-as-Judge Evaluation~~ ✅ (Done)
**Impact**: Catches hallucinated drug interactions, fabricated diagnoses, or off-topic responses before they reach the user — critical for a medical system.

**What to build**: A `node_reviewer` step in the LangGraph workflow inserted between `node_aggregator` and `node_restore_privacy`.

**Judge backend: Groq (free tier)** — `GROQ_API_KEY` already set in `.env` and verified ✅

**Groq free tier limits for `llama-3.3-70b-versatile`**:
- 30 RPM / 1,000 RPD / 12,000 TPM / **100,000 TPD**
- At ~1,000 tokens/judge call → **~100 judged requests/day** before hitting TPD
- Binding constraint is **tokens/day**, not requests/day

**Safeguards to implement** (prevents hitting limits):
1. **Input truncation** — truncate aggregated response to 500 tokens before sending to judge. Keeps each call to ~750 tokens → ~133 calls/day
2. **Sampling rate** — configurable `JUDGE_SAMPLE_RATE` (1.0 = always judge, 0.5 = 50% of requests). Use `1.0` for dev, lower for demos
3. **Fallback model** — if `llama-3.3-70b-versatile` hits TPD, fall back to `llama-3.1-8b-instant` (500,000 TPD free → ~500 calls/day)

**New config keys to add to `config.py`**:
```python
GROQ_API_KEY: str = ""
JUDGE_ENABLED: bool = True
JUDGE_SAMPLE_RATE: float = 1.0       # 0.0–1.0
JUDGE_MODEL: str = "llama-3.3-70b-versatile"
JUDGE_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
JUDGE_MAX_INPUT_TOKENS: int = 500    # truncate before sending to judge
```

**Judge scoring criteria** (in `node_reviewer` system prompt):
- Does the response address the user's query?
- Are all claims grounded in tool outputs (no fabricated facts)?
- Does the response contain any leaked PII placeholders?
- Score 1–5; if score < 3 → append a clinical disclaimer to the response

**Files to change**:
- `config.py` — add 6 new config keys above
- `orchestrator.py` — add `node_reviewer` graph node between `node_aggregator` and `node_restore_privacy`; use `langchain-groq` with `ChatGroq(model=settings.JUDGE_MODEL)`

---

## 🟡 Medium Priority

### ~~§4.2 (A2A) — Idempotency Cache~~ ✅ (Done)
**Impact**: Prevents duplicate execution if a request is retried (e.g., on network failure). Becomes critical if agents ever gain write capabilities (DB writes, notifications, external APIs).

**What to build**: An idempotency cache in `A2ABaseAgent.process()` in `base.py`.

**Files to change**:
- `specialized_agents/base.py` — add `self._response_cache: Dict[str, AgentResponse] = {}` and check `envelope.idempotency_key` before executing the ReAct loop. Store result after completion.
- Optionally back with Redis for distributed deployments.

---

### ~~§5.1 (MCP) — MCP Inspector Testing~~ ✅ (Resolved by Unit Tests)
**Impact**: Validates that all 13 MCP tools are correctly structured for any MCP-compliant external client (Claude Desktop, Cursor, etc.).

**Resolution**: A programmatic STDIO test script is redundant because `tests/mcp/test_mcp_server.py` already directly validates tool listing, schema conformance, and handler invocation natively. The core MCP protocol serialization is maintained by Anthropic's SDK.
Developers can manually test the Server's STDIO transport via Inspector:
`npx @modelcontextprotocol/inspector python tools/mcp_server.py`

---

## 🟢 Low Priority

### ~~§3.2 (MCP) — Prompts Primitive~~ ✅ (Done)
**Impact**: Standardizes how external MCP clients use our tools — useful for Claude Desktop or Cursor integrations.

**What to build**: Expose `prompts/list` and `prompts/get` in `mcp_server.py` with reusable workflow templates.

**Files to change**:
- `tools/mcp_server.py` — add `@server.list_prompts()` and `@server.get_prompt()` handlers with 3 prompt templates:

| Prompt Name | Workflow |
|---|---|
| `patient-full-review` | retrieve → vitals → medications → history |
| `drug-safety-check` | interactions → recommendations |
| `medical-report-analysis` | extract document/image → analyze report |

---

### ~~Dependency Upgrade — spaCy 3.7.4 → 3.8.11 (Deprecation Warnings)~~ ✅ (Done)
**Impact**: Clears 2 third-party deprecation warnings that currently appear in test output (cosmetic only, no functional breakage).

**Background**: Upgrading spaCy from `3.7.4` to `3.8.11` resolves:
1. **`BaseCommand` deprecated** (Click 8.3 → Typer 0.9.4) — Typer is upgraded to `0.24.1` as part of the spaCy 3.8 dependency chain.
2. **`split_arg_string` deprecated** (Click 8.3 → spaCy/weasel 0.3.4) — Fixed in spaCy 3.8.7+, weasel upgraded to `0.4.3`.

**Dry-run confirmed** — only 7 packages change inside `.venv`. All direct project dependencies (Pydantic, Presidio, LangChain, FastAPI, SQLAlchemy, etc.) are unaffected.

**What to do**:
```bash
source .venv/bin/activate
pip install "spacy==3.8.11"
python -m spacy download en_core_web_sm   # re-download model for 3.8 binary format
pytest tests/ -v                          # verify all 115 tests still pass
```

**Files to change**:
- `requirements.txt` — update `spacy==3.7.4` to `spacy==3.8.11`

**Note**: The 3rd warning (`datetime.utcnow()` in Pydantic internals) has **no fix available** — Pydantic 2.12.5 is already the latest version. It will self-resolve in a future Pydantic release.
