# PRD: ATT-1 — Attachment-Based Conversation Pipeline

**Status:** Ready for Implementation  
**Priority:** Medium  
**Author:** Research & Planning — 2026-04-23  
**Target:** Full end-to-end attachment conversation support across multi-turn sessions

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current State Audit](#2-current-state-audit)
3. [Technology Findings](#3-technology-findings)
4. [Gaps & Problems to Solve](#4-gaps--problems-to-solve)
5. [Agent Routing Design](#5-agent-routing-design)
6. [Implementation Phases](#6-implementation-phases)
7. [Data Model Changes](#7-data-model-changes)
8. [File Reference Map](#8-file-reference-map)
9. [Testing Plan (ATT-1)](#9-testing-plan-att-1)
10. [Non-Goals](#10-non-goals)
11. [Open Questions](#11-open-questions)

---

## 1. Overview

### Objective

Enable MediCortex AI to accept medical documents (PDF lab reports, discharge summaries, radiology reports) and medical images (X-rays, CT scans, histology) as attachments in a conversation, analyze them through the appropriate specialized agents, and maintain awareness of those attachments across multi-turn follow-up questions — without breaking any existing multi-hop, clarification, or re-retrieval features.

### Problem Statement

The attachment pipeline exists in skeletal form: files can be uploaded to MinIO, `report_analyzer` is always force-routed when `file_urls` is non-empty, and `extract_document_text` / `extract_image_findings` tools are implemented. However, three critical gaps prevent a production-quality experience:

1. **Cross-agent blindness**: `diagnosis` and `pharmacology` agents run in parallel with `report_analyzer` and never receive its extracted findings. They produce generic answers instead of document-grounded ones.
2. **No multi-turn file memory**: Attachments are stored in the DB but silently dropped when conversation history is re-injected on subsequent turns. A follow-up question loses all file context.
3. **PDF extraction quality**: Current extraction uses bare `pymupdf4llm.to_markdown()` with a 3000-char truncation and no layout awareness. Medical documents with multi-column layouts, tables, and embedded images are only partially extracted.

### Success Criteria

- Upload a PDF lab report → ask a clinical interpretation question → receive a grounded, document-specific answer
- Upload an X-ray image → ask "what do you see?" → MedGemma vision analyzes and responds
- Follow up with "what medications should I adjust?" in the same session → pharmacology agent sees the prior report findings without re-uploading
- All existing multi-turn, clarification, and re-retrieval behaviours continue to work as before

---

## 2. Current State Audit

### 2.1 What Is Already Built

| Component | File | Status |
|---|---|---|
| File upload endpoint (`POST /upload`) | `orchestrator.py:1318–1329` | ✅ Complete |
| MinIO presigned URL generation (7-day TTL) | `services/minio_service.py:38–55` | ✅ Complete |
| `AgentState.file_urls: List[str]` field | `orchestrator.py:161` | ✅ Complete |
| Attachments persisted in DB as JSONB | `database/models.py:26` | ✅ Complete |
| `report_analyzer` force-appended when files present | `orchestrator.py:912–913` | ✅ Complete |
| `extract_document_text` — PDF → Markdown via pymupdf4llm | `tools/document_extraction_tools.py:23–87` | ✅ Exists, needs improvement |
| `extract_image_findings` — image → MedGemma base64 vision | `tools/image_extraction_tools.py:27–104` | ✅ Exists, needs testing |
| `analyze_report` — clinical interpretation of extracted content | `tools/report_analysis_tools.py:21–98` | ✅ Exists |
| Frontend file input, attachments state, onSend callback | `frontend/src/components/InputArea.tsx:25–52` | ✅ Complete |
| `pymupdf==1.27.1`, `pymupdf4llm==0.2.9` installed | `.venv` | ✅ Present |

### 2.2 Current Data Flow (Single-Turn)

```
User uploads file (InputArea.tsx)
    → POST /upload → MinIO → presigned URL returned to frontend
    → User sends message with attachments array

POST /chat (orchestrator.py:1091)
    → file_urls extracted from request.attachments
    → user message + attachments saved to DB (chat_messages.attachments JSONB)
    → AgentState["file_urls"] = [presigned_url_1, ...]

node_analyze_privacy → node_retrieve_knowledge → node_router
    → route_decision() forces "report_analyzer" into routes (line 912–913)
    → report_analyzer runs (+ any LLM-routed agents in parallel)

make_agent_node("report_analyzer")
    → file_urls appended to enhanced_input string (line 562–563)
    → A2ABaseAgent.process() → _plan_and_synthesize()
    → Gemma 4 (planner) calls extract_document_text or extract_image_findings
    → findings text passed to MedGemmaLLM.invoke() for synthesis

node_aggregator → node_reviewer → node_restore_privacy → SSE response
```

### 2.3 Current Limitations by Component

#### `orchestrator.py` — Agent Injection (lines 562–563)
```python
# Current: only report_analyzer receives file_urls
if agent_key == "report_analyzer" and state.get("file_urls"):
    enhanced_input += "\n\nFiles to analyze:\n" + "\n".join(state["file_urls"])
```
`diagnosis` and `pharmacology` never see the files, and never see report_analyzer's extracted output.

#### `orchestrator.py` — History Re-Injection (line 1102)
```python
# Current: attachments silently dropped
history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]
```
File attachments from prior turns are not re-surfaced in subsequent turns.

#### `tools/document_extraction_tools.py`
```python
# Current extraction call — bare, no layout, low DPI, hard 3000-char truncation
md_text = pymupdf4llm.to_markdown(tmp_path)
return md_text[:3000]
```

#### `specialized_agents/medgemma_llm.py` — Synthesis path (line 45)
```python
# MedGemma synthesis always receives image_base64=None
payload = {"prompt": prompt, "image_base64": None, ...}
```
Images are only analyzed via the `extract_image_findings` tool, not in the synthesis call. This is intentional and correct — vision happens at tool-call time.

---

## 3. Technology Findings

### 3.1 MedGemma 1.5 4B (4-bit Quantized) — Vision Capabilities

#### Architecture
MedGemma 4B uses **MedSigLIP** — a 400M parameter SigLIP vision encoder pre-tuned on ~33 million de-identified medical image–text pairs. Resolution cap: **896×896 pixels**.

> **Note**: The 27B variant is text-only. Vision is exclusive to the 4B model. Confirm the locally-served model is `medgemma:4b` (not `medgemma:27b`) before running image tests.

#### Accepted Image Formats
The API layer (OpenAI-compatible `/v1/chat/completions` via Ollama) accepts:
- **JPEG, PNG, WebP** — via URL or base64 data URL
- **DICOM (`.dcm`)** — NOT accepted natively. Must pre-convert to JPEG/PNG via `pydicom` + `Pillow` before sending.
- **TIFF, BMP** — NOT accepted by the API layer directly.

Current `extract_image_findings` already uses base64 encoding — this is the correct approach for presigned URLs (avoids a second download by the model server).

#### Medical Modality Support

| Modality | Support Level | Notes |
|---|---|---|
| Chest X-ray (CXR) | **Strong** | MS-CXR-T: 66% macro accuracy; anatomical localization +35% IoU vs prior version |
| CT (computed tomography) | **Good** | 61% accuracy on CT finding classification |
| MRI | **Good** | 65% accuracy on MRI findings (+14% vs prior) |
| Histopathology / WSI | **Strong** | +47% macro F1 gain in WSI classification |
| Dermatology | **Moderate** | Part of 62% avg internal benchmark |
| Fundus / Ophthalmology | **Moderate** | Part of 62% avg internal benchmark |
| Ultrasound | **Unknown** | Not cited in published benchmarks |
| Pathology slides | **Supported** | See histopathology above |

#### 4-bit Quantization Impact
- **Classification tasks** (finding labels, disease detection): ~5% accuracy drop vs. full-precision — acceptable.
- **Spatial tasks** (lesion localization, anatomical boundary detection): ~10–15% accuracy drop — relevant for radiology.
- No published benchmark exists for quantized MedGemma specifically; these are extrapolated from general vision quantization literature.

#### Ollama API — Passing Images
```python
import base64

with open("scan.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "medgemma:4b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe any abnormalities in this chest X-ray."}
        ]
    }]
}
# POST to http://homeserver:11434/v1/chat/completions
```

The existing `extract_image_findings` tool sends to a custom `MEDGEMMA_API_URL` endpoint (not Ollama's `/v1`). Verify whether the local MedGemma server expects the same payload format.

#### Known Constraints
- **Single image per turn** — multi-image inputs are not evaluated/supported.
- **Single-turn optimized** — not specifically tuned for multi-turn dialogue.
- **Resolution cap**: 896×896. Large radiology images must be downsampled.
- **Flash Attention**: Already mitigated (`OLLAMA_FLASH_ATTENTION=0` on host). ✅
- **Not a diagnostic tool**: Output is non-deterministic. All responses must include clinical disclaimer.

---

### 3.2 PyMuPDF4LLM — PDF Extraction

#### Installed Version
`pymupdf4llm==0.2.9` — both packages already in `.venv`. No new installs required.

#### What pymupdf4llm Provides vs. Raw PyMuPDF

| Feature | Raw PyMuPDF (`fitz`) | pymupdf4llm |
|---|---|---|
| Markdown output | ❌ | ✅ with headings, bold, bullets |
| Table detection | ❌ | ✅ GitHub-flavored Markdown tables |
| Reading order correction | ❌ | ✅ |
| Multi-column layout | ❌ | ✅ via Layout mode (GNN-based) |
| OCR for scanned PDFs | Manual | ✅ selective (skips clean text) |
| Page chunking for RAG | ❌ | ✅ `page_chunks=True` |
| Image extraction | Manual | ✅ `write_images=True` (not in Layout mode) |

#### Layout Mode
"PyMuPDF Layout" is **not a separate package** — it is bundled inside `pymupdf4llm` as of recent versions and activated via `use_layout=True` (or automatically). Internally uses a CPU-only GNN that treats text boxes as nodes and spatial relationships as edges to classify elements (title, paragraph, table, figure). ~10x faster than vision-model-based alternatives.

> **Known conflict**: Image extraction (`write_images=True`) is **disabled when Layout mode is active**. For documents requiring both layout correction AND embedded image extraction, use a **two-pass approach** (see Phase 3 implementation).

#### Correct Usage Patterns

```python
import pymupdf4llm

# 1. Basic whole-document markdown
md_text = pymupdf4llm.to_markdown("report.pdf")

# 2. Layout-aware (multi-column, table correction)
md_text = pymupdf4llm.to_markdown("report.pdf", use_layout=True)

# 3. Page chunks for RAG (returns list of dicts per page)
chunks = pymupdf4llm.to_markdown("report.pdf", page_chunks=True)
# chunks[n] = {"metadata": {...}, "text": "...", "images": [...], "tables": [...]}

# 4. Image extraction alongside text (Layout must be OFF)
md_text = pymupdf4llm.to_markdown(
    "report.pdf",
    write_images=True,
    image_path="./extracted",
    image_format="png",
    dpi=200,          # 200–300 recommended for clinical scans
)

# 5. Specific pages only
md_text = pymupdf4llm.to_markdown("report.pdf", pages=[0, 1, 2])
```

#### Medical Document Quality

| Document Type | Quality | Notes |
|---|---|---|
| Digital lab reports | **Excellent** | Tables render cleanly, values/units preserved |
| Discharge summaries | **Good** | Section headers → Markdown headings, prose flows correctly |
| Radiology reports (digital) | **Good** | Free-text narrative preserved; impression/findings clean |
| Scanned radiology PDFs | **Moderate** | Depends on scan DPI; 300 DPI+ works well, <150 DPI degrades |
| Complex lab panels (merged-cell tables) | **Fair** | Merged cells may split; post-process with pandas if needed |

#### Key Gotchas
1. **Image extraction + Layout mode conflict** — two-pass approach required when both are needed.
2. **Merged table cells** — lab report panels with spanned headers may not parse correctly. Acceptable for MVP; log as known limitation.
3. **OCR requires Tesseract binary on PATH** — `pytesseract` is already in `requirements.txt` but Tesseract must be installed at OS level separately. RapidOCR is a Python-only fallback if Tesseract is unavailable.
4. **Default DPI is 150** — too low for clinical image quality. Use `dpi=200` minimum.
5. **3000-char truncation** is currently applied in `document_extraction_tools.py` — inappropriate for multi-page clinical documents. Replace with token-budget-aware chunking.

#### Comparison vs. Alternatives (for reference)

| Tool | Strengths | When to prefer over pymupdf4llm |
|---|---|---|
| `pdfplumber` | Best coordinate-based table extraction | When precise bounding-box table data is needed |
| `pypdf` | Lightweight, fast | Never for clinical documents (no layout awareness) |
| `marker` | Best markdown quality for complex academic PDFs | When GPU is available and quality > speed |
| `pymupdf4llm` | Best balance: speed, markdown, OCR, layout, CPU-only | **Default choice for this project** |

---

## 4. Gaps & Problems to Solve

### GAP-1 — Cross-Agent Blindness (High Impact)

**Problem**: When a file is uploaded alongside a clinical question, `report_analyzer` runs in **parallel** with `diagnosis`/`pharmacology`. These downstream agents never receive the extracted findings. They produce generic knowledge-based answers instead of document-specific analysis.

**Example failure**:
- User uploads a blood panel PDF and asks "Are my cholesterol levels concerning?"
- `report_analyzer` extracts: "LDL: 185 mg/dL, HDL: 38 mg/dL, Triglycerides: 240 mg/dL"
- `diagnosis` (running in parallel) has no access to these values — answers with generic hyperlipidemia information
- Aggregator combines two unrelated outputs → incoherent response

**Root cause**: `make_agent_node()` only injects file_urls into `report_analyzer`'s `enhanced_input`. No mechanism exists to share one agent's output as another agent's input within the same turn.

**Proposed fix**: Introduce a sequential execution model for attachment turns — `report_analyzer` runs first in a dedicated extraction node, its output is stored in `AgentState["extracted_document_context"]`, and downstream agents receive it injected into their `enhanced_input`. See Phase 1.

---

### GAP-2 — No Multi-Turn File Memory (High Impact)

**Problem**: The conversation history re-injection at `orchestrator.py:1102` drops all attachment data:
```python
history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]
```
The `chat_messages.attachments` JSONB column stores attachment metadata but it is never read back. A follow-up question in the same session has no awareness of previously uploaded files.

**Example failure**:
- Turn 1: Upload ECG report image, ask "Is this rhythm normal?"
- Turn 2: Ask "What medications would treat this?"
- `pharmacology` agent has no context about the ECG or prior findings → generic answer

**Proposed fix**: Two-layered approach:
1. **Text injection**: Store the extracted document text (from `report_analyzer`'s tool output) in the DB alongside the attachments. On subsequent turns, re-inject this extracted text into conversation history context.
2. **URL re-attachment** (secondary): For turns within the 7-day MinIO TTL, re-attach prior presigned URLs to `file_urls` so agents can re-analyze if needed.

See Phase 2.

---

### GAP-3 — PDF Extraction Quality (Medium Impact)

**Problem**: `document_extraction_tools.py` uses the simplest possible extraction call with a hard 3000-char truncation. For multi-page lab reports (4–10 pages), multi-column discharge summaries, or table-heavy blood panels, significant clinical data is lost or misformatted.

**Specific issues**:
- No `use_layout=True` → multi-column documents interleave columns incorrectly
- 3000-char truncation → typical lab report loses 60–70% of content
- No `page_chunks=True` → no way to process by section or page
- Default `dpi=150` → embedded images in scanned PDFs are low quality
- No scanned PDF detection → if OCR is needed, it silently fails to clean text

**Proposed fix**: Upgrade `extract_document_text` tool. See Phase 3.

---

### GAP-4 — Router Has No File Awareness for LLM Routing (Low Impact)

**Problem**: The LLM router (Gemma 4 in `node_router`) makes routing decisions from text content only. It has no knowledge that files were attached. In practice `report_analyzer` is always forced in regardless, but the LLM router's agent selection for secondary agents (diagnosis, pharmacology) could be improved if it knew file types were present.

**Example**: "Analyze this blood test" with a PDF attached → router may not select `diagnosis` because the query text alone is ambiguous. With file awareness in the prompt, it could confidently select `diagnosis` + `pharmacology`.

**Proposed fix**: Add a file-context summary line to the router prompt when `file_urls` is non-empty (e.g., "The user has attached 1 PDF document and 1 image."). Low risk, low effort. See Phase 1, Step 5.

---

### GAP-5 — No DICOM Support (Low Impact, Future)

**Problem**: Radiology images are commonly in DICOM format. MedGemma cannot receive raw `.dcm` files. No conversion utility exists in the current codebase.

**Proposed fix**: Not in scope for ATT-1 MVP. Log as ATT-2 backlog item. If a `.dcm` file is uploaded, return a user-facing error: "DICOM files are not supported. Please convert to JPEG or PNG before uploading."

---

### GAP-6 — Missing File Type Validation on Upload (Low Impact)

**Problem**: The `POST /upload` endpoint accepts any file. No server-side MIME type validation exists. A corrupted or unsupported file silently fails inside the agent tool.

**Proposed fix**: Add MIME type allow-list validation on the upload endpoint. Accepted: `application/pdf`, `image/jpeg`, `image/png`, `image/webp`. Reject all others with HTTP 415. See Phase 3.

---

## 5. Agent Routing Design

### Which Agents Handle Attachments?

| Agent | Receives Files? | Role in Attachment Flow | Condition |
|---|---|---|---|
| `report_analyzer` | ✅ Always (when files present) | **Primary intake**: downloads, extracts, and summarizes document/image findings | Any attachment present |
| `diagnosis` | ✅ After Phase 1 | **Consumer**: receives extracted findings as context; identifies conditions, differentials | When query has clinical/diagnostic intent |
| `pharmacology` | ✅ After Phase 1 | **Consumer**: receives extracted findings; analyzes medications, interactions, recommendations | When query references medications, treatments, or drug data in the document |
| `pubmed` | ⚠️ Conditional | **Enricher**: if extracted findings reference rare conditions; triggered by LLM router only | When query asks for research/evidence on a condition found in the document |
| `patient` | ⚠️ Edge case | Only if the uploaded document is a patient record and the query is patient-data-specific | When file context + query clearly targets patient record lookup |

### Why `report_analyzer` Must Always Run First (Sequential, Not Parallel)

The current parallel execution model is incompatible with cross-agent result sharing. The pipeline must shift to:

```
[Attachment turn]

report_analyzer (extraction node — runs first, standalone)
    ↓
extracted_document_context stored in AgentState
    ↓
diagnosis + pharmacology (receive extracted_document_context in enhanced_input)
    ↓
node_aggregator (combines all outputs)
```

This does not affect non-attachment turns — they continue to run all selected agents in parallel.

### Router Prompt Addition (for file awareness)

When `file_urls` is non-empty, prepend to the router system prompt:
```
The user has attached the following file(s): {file_count} file(s) ({file_types}).
The report_analyzer agent will always run to extract document content.
Select additional agents based on the user's clinical question.
```

---

## 6. Implementation Phases

---

### Phase 1 — Cross-Agent Result Sharing

**Priority:** High | **Effort:** Medium | **Risk:** Medium (changes orchestration flow)

**Goal**: `report_analyzer` runs first, its extracted findings are propagated to downstream agents in the same turn.

#### Step 1 — Add `extracted_document_context` to `AgentState`

**File**: `orchestrator.py`  
**Location**: `AgentState` TypedDict definition (~line 154)

```python
class AgentState(TypedDict):
    # ... existing fields ...
    extracted_document_context: str   # Extracted text from report_analyzer; empty string if no files
```

Initialize to `""` in both `ainvoke` call sites.

#### Step 2 — Add Dedicated Extraction Node

**File**: `orchestrator.py`  
**New node**: `node_extract_attachments`

This node runs `report_analyzer` synchronously (not as a graph branch) and stores the result:

```python
async def node_extract_attachments(state: AgentState) -> AgentState:
    if not state.get("file_urls"):
        return state   # no-op for non-attachment turns

    # Run report_analyzer agent directly
    envelope = build_envelope(state, agent_key="report_analyzer")
    result: AgentResponse = await report_agent_instance.process(envelope)

    # Store extracted findings text in state
    return {**state, "extracted_document_context": result.content}
```

Wire this node **after** `node_router` and **before** `make_agent_node` for all non-`report_analyzer` agents. `report_analyzer` is removed from the parallel agent batch when running in this mode.

#### Step 3 — Inject Extracted Context into Downstream Agent Inputs

**File**: `orchestrator.py`  
**Location**: `make_agent_node()` enhanced_input construction (~line 555)

```python
if state.get("extracted_document_context"):
    enhanced_input += (
        "\n\n---\n"
        "Document/Image Analysis (from uploaded file):\n"
        + state["extracted_document_context"]
        + "\n---\n"
        "Use the above extracted document content to ground your analysis."
    )
```

Apply to: `diagnosis`, `pharmacology`, `pubmed`. Do NOT re-apply to `report_analyzer` (it ran in the extraction node).

#### Step 4 — Conditional Graph Wiring

When `file_urls` is non-empty, the graph path becomes:
```
node_router → node_extract_attachments → [diagnosis | pharmacology | pubmed] (parallel) → node_aggregator
```

When no files:
```
node_router → [all selected agents] (parallel) → node_aggregator
```

Use a conditional edge from `node_router` that checks `state["file_urls"]`.

#### Step 5 — File Context Hint in Router Prompt

**File**: `orchestrator.py`  
**Location**: `node_router` system prompt construction (~line 433)

When `state.get("file_urls")`:
```python
file_hint = (
    f"[FILE CONTEXT] The user has uploaded {len(file_urls)} file(s). "
    "The report_analyzer agent will always be included. "
    "Select additional agents based on the clinical question."
)
# Prepend file_hint to the router system prompt
```

---

### Phase 2 — Multi-Turn File Memory

**Priority:** High | **Effort:** Medium | **Risk:** Low

**Goal**: Subsequent turns in the same session can reference previously uploaded files without re-uploading.

#### Step 1 — Store Extracted Text in DB

**File**: `database/models.py`, `services/chat_service.py`

Add `extracted_content: str | None` to the `ChatMessage` model (JSONB or separate text column). When `node_extract_attachments` runs, store the full extracted text alongside the assistant message.

Alternatively (simpler, no schema change): store extracted text inside `message_metadata["extracted_document_text"]` (already a JSONB field).

**Recommended approach**: Store inside `message_metadata` for zero schema migration:
```python
msg_metadata["extracted_document_text"] = state.get("extracted_document_context", "")
```

#### Step 2 — Re-Inject Extracted Text in History Context

**File**: `orchestrator.py`  
**Location**: history re-injection loop (~line 1102)

```python
history_lines = []
for m in past_turns[-10:]:
    history_lines.append(f"{m.role.capitalize()}: {m.content}")
    # Re-inject extracted document text from prior attachment turns
    if m.role == "user" and m.attachments:
        meta = prior_assistant_message_metadata  # fetch the assistant reply metadata
        extracted = meta.get("extracted_document_text", "")
        if extracted:
            history_lines.append(
                f"[Prior Document Context]: {extracted[:2000]}"  # cap at 2000 chars for history
            )
history_context = history_lines
```

**Implementation note**: The history is currently built from `past_turns` which returns `ChatMessage` objects. Fetching the associated assistant message's metadata requires either:
- Joining messages in pairs (user → assistant) in `chat_service.get_history()`, or
- Storing extracted text on the **user** message's metadata at upload time.

Storing on the user message (at POST /chat time) is simpler. When `file_urls` is present, mark `user_msg_metadata["has_attachments"] = True`. After the graph completes and `extracted_document_context` is populated, backfill the user message's metadata with the extracted text. This avoids a join.

#### Step 3 — Re-Attach Presigned URLs for Within-TTL Turns

If a user asks about an attachment within 7 days, the MinIO URL is still valid. Add URL re-attachment to the history injection:

```python
for m in past_turns[-10:]:
    if m.role == "user" and m.attachments:
        # Re-attach file URLs from prior turns if none uploaded this turn
        if not state.get("file_urls"):
            prior_urls = [a["url"] for a in m.attachments if "url" in a]
            state["file_urls"].extend(prior_urls)
            break  # Only re-attach the most recent attachment turn
```

This is a secondary mechanism — extracted text re-injection (Step 2) is more reliable since it does not depend on URL TTL.

---

### Phase 3 — PDF Extraction Quality Upgrade

**Priority:** Medium | **Effort:** Low | **Risk:** Low

**Goal**: Upgrade `extract_document_text` to produce higher-quality, structure-aware Markdown from medical PDFs.

#### Step 1 — Upgrade Extraction Call

**File**: `tools/document_extraction_tools.py`

Replace the current bare extraction:
```python
# BEFORE
md_text = pymupdf4llm.to_markdown(tmp_path)
return md_text[:3000]
```

With a two-pass layout-aware approach:
```python
import pymupdf4llm
import fitz  # raw PyMuPDF for pass 2

def extract_pdf_content(tmp_path: str, max_chars: int = 12000) -> str:
    """
    Pass 1: Layout-aware text extraction (tables, multi-column).
    Pass 2: Image extraction (separate pass — layout mode disables write_images).
    """
    # Pass 1: layout-aware markdown
    md_text = pymupdf4llm.to_markdown(
        tmp_path,
        use_layout=True,
        page_chunks=False,
        dpi=200,
    )

    # Truncate by char budget rather than hard cutoff
    if len(md_text) > max_chars:
        md_text = md_text[:max_chars] + "\n\n[Document truncated — showing first portion]"

    return md_text
```

#### Step 2 — Scanned PDF Detection

Before extraction, detect if the PDF is scanned (image-only, no selectable text):

```python
import fitz

def is_scanned_pdf(path: str) -> bool:
    doc = fitz.open(path)
    for page in doc:
        if page.get_text().strip():
            return False   # Has selectable text
    return True  # All pages are images
```

If `is_scanned_pdf()` returns `True`, log a warning and fall back to OCR mode:
```python
if is_scanned_pdf(tmp_path):
    md_text = pymupdf4llm.to_markdown(tmp_path, use_layout=False)  # OCR auto-triggers
```

#### Step 3 — Raise Token Budget

Replace 3000-char hard truncation with a 12000-char budget (covers ~8–10 pages of typical clinical text). Add a `[Document truncated]` marker so downstream agents know extraction was partial.

#### Step 4 — File Type Validation on Upload Endpoint

**File**: `orchestrator.py` — `POST /upload` handler (~line 1318)

```python
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}

@app.post("/upload")
async def upload_file(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: PDF, JPEG, PNG, WebP."
        )
    # ... rest of upload logic
```

Add matching validation in the frontend `InputArea.tsx` `accept` attribute:
```tsx
<input type="file" accept=".pdf,.jpg,.jpeg,.png,.webp" />
```

---

### Phase 4 — ATT-1 Integration Tests

**Priority:** Medium | **Effort:** Medium | **Risk:** Low

**Goal**: Write the integration tests specified in Todo.md `ATT-1` scope.

**Test file**: `tests/integration/test_attachment_pipeline.py`

#### Test AT1 — PDF Lab Report Upload → Structured Extraction

```
POST /upload (PDF)
→ Verify presigned URL returned
→ POST /chat with file attached + "Analyze my blood test results"
→ Assert: agents_used includes "report_analyzer"
→ Assert: response content references specific values from the PDF
→ Assert: message_metadata["extracted_document_text"] is non-empty
→ Assert: no pipeline errors
```

#### Test AT2 — Medical Image Upload → MedGemma Vision Analysis

```
POST /upload (JPEG chest X-ray)
→ POST /chat with image attached + "What do you see in this image?"
→ Assert: agents_used includes "report_analyzer"
→ Assert: response describes visual findings (not generic text)
→ Assert: is_clarification = false
→ Assert: no "extract_image_findings" tool call errors in thinking steps
```

#### Test AT3 — Multi-Turn Follow-Up After Attachment

```
Turn 1: POST /upload (PDF lab report) + "What does my blood panel show?"
→ Assert: report_analyzer ran, extracted_document_text stored in metadata

Turn 2 (same session): POST /chat (no file) + "What medications should I adjust?"
→ Assert: pharmacology agent ran
→ Assert: response references values from Turn 1 document (not generic)
→ Assert: extracted_document_context was re-injected from history
```

#### Test AT4 — route_decision Always Includes report_analyzer When file_urls Non-Empty

```python
# Unit test
state = AgentState(file_urls=["http://minio/test.pdf"], ...)
routes = route_decision(state)
assert "report_analyzer" in routes
```

#### Test AT5 — Presigned URL TTL Validation

```
Upload a file and record the presigned URL
Wait (or mock time) to confirm URL structure includes expiry
Assert: URL contains "X-Amz-Expires=604800" (7 days in seconds)
```

#### Test AT6 — Unsupported File Type Returns 415

```
POST /upload with a .docx file
Assert: HTTP 415 returned
Assert: Error message mentions accepted types
```

#### Test AT7 — PDF Extraction Quality Smoke Test

```python
# Unit test
from tools.document_extraction_tools import extract_document_text
result = extract_document_text.invoke({"file_url": SAMPLE_LAB_REPORT_PDF_URL})
assert len(result) > 500           # non-trivial content extracted
assert "mg/dL" in result or "mmol" in result  # clinical values preserved
assert "[Document truncated]" not in result or len(result) >= 12000
```

---

## 7. Data Model Changes

### 7.1 `message_metadata` JSONB — New Fields

No schema migration required. New keys added to existing `message_metadata` JSONB column:

| Key | Type | Added By | Description |
|---|---|---|---|
| `extracted_document_text` | `str` | Phase 2, streaming endpoint | Full extracted text from `report_analyzer`'s tool output for the current turn |
| `has_attachments` | `bool` | Phase 2, POST /chat | True if user message had file_urls |
| `file_types` | `List[str]` | Phase 3, POST /upload | MIME types of uploaded files (e.g. `["application/pdf"]`) |
| `extraction_method` | `str` | Phase 3, document_extraction_tools | `"layout"`, `"ocr"`, or `"basic"` — which extraction path was used |

### 7.2 `AgentState` — New Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `extracted_document_context` | `str` | `""` | Populated by `node_extract_attachments`; injected into downstream agent inputs |

---

## 8. File Reference Map

### Backend

| File | Phase | Changes |
|---|---|---|
| `orchestrator.py:154–187` | Phase 1 | Add `extracted_document_context` to `AgentState` |
| `orchestrator.py:433–524` | Phase 1 | Add file context hint to router prompt |
| `orchestrator.py:555–570` | Phase 1 | Inject `extracted_document_context` into downstream agent `enhanced_input` |
| `orchestrator.py:890–930` | Phase 1 | Add `node_extract_attachments` node; update graph wiring with conditional edge |
| `orchestrator.py:1091–1145` | Phase 2 | Re-attach prior file URLs and extracted text in history re-injection |
| `orchestrator.py:1318–1329` | Phase 3 | Add MIME type validation to `/upload` endpoint |
| `tools/document_extraction_tools.py:23–87` | Phase 3 | Two-pass layout extraction, scanned PDF detection, 12000-char budget |
| `specialized_agents/agents.py` | Phase 1 | Ensure `report_agent_instance` is importable for direct call in extraction node |
| `database/models.py` | Phase 2 | No schema change; document new `message_metadata` keys |
| `services/chat_service.py` | Phase 2 | Update `get_history()` to return `message_metadata` alongside messages |

### Frontend

| File | Phase | Changes |
|---|---|---|
| `frontend/src/components/InputArea.tsx:25–52` | Phase 3 | Add `accept=".pdf,.jpg,.jpeg,.png,.webp"` to file input |
| `frontend/src/components/InputArea.tsx` | Phase 3 | Show file type error if unsupported file selected |
| `frontend/src/components/MessageBubble.tsx` | Phase 4 | Show attachment thumbnail/badge on user messages that had files |

### Tests

| File | Phase | Description |
|---|---|---|
| `tests/integration/test_attachment_pipeline.py` | Phase 4 | Full ATT-1 test suite (AT1–AT7) |
| `tests/unit/test_document_extraction.py` | Phase 3 | Unit tests for upgraded extraction logic |
| `tests/unit/test_route_decision_files.py` | Phase 1 | Assert `report_analyzer` always in routes when `file_urls` non-empty |

---

## 9. Testing Plan (ATT-1)

### Pre-Test Checklist (Fresh Session)

1. Restart `python orchestrator.py` — confirm `Orchestrator Graph Compiled` and `Gemma 4 warmup complete`
2. Confirm MedGemma server running at `settings.MEDGEMMA_API_URL` and responding to health check
3. Confirm MinIO is running and accessible (`mc ls minio/medicortex-uploads` or equivalent)
4. Run `python3 -m knowledge_core.build_fast_assets` — verify ArangoDB populated
5. Open a new chat session in UI (do not reuse prior sessions)
6. Open orchestrator terminal to watch for: `[report_analyzer] extract_document_text`, `[report_analyzer] extract_image_findings`, `extracted_document_context populated`, `Injecting document context into diagnosis`

### Manual Browser Test Cases

| Test | Input | Expected |
|---|---|---|
| BT1 — PDF lab report | Upload PDF + "Analyze my blood test" | report_analyzer runs, response references specific values from PDF |
| BT2 — JPEG X-ray | Upload X-ray + "What abnormalities do you see?" | MedGemma vision runs, response describes visual findings |
| BT3 — PDF + clinical question | Upload discharge summary + "What are the diagnoses?" | report_analyzer + diagnosis both run; diagnosis response is document-specific |
| BT4 — PDF + drug question | Upload prescription PDF + "Are there any dangerous interactions?" | report_analyzer + pharmacology both run; pharmacology sees extracted drug list |
| BT5 — Multi-turn follow-up | Upload lab report → ask "what does this show?" → follow up "what diet changes?" | Second turn has no file but receives extracted context from first turn |
| BT6 — Unsupported file type | Try uploading a .docx | Error shown in UI; 415 from API |
| BT7 — Vague attachment query | Upload PDF + "what is this?" | Clarification fires OR report_analyzer extracts and summarizes |
| BT8 — HIPAA check | Upload PDF containing "Jane Doe DOB 1990-01-01..." + ask question | No "Jane Doe" in response; no `<PERSON_N>` visible |

### Known Limitations to Document

1. **DICOM not supported** — return HTTP 415 with message directing user to convert to JPEG/PNG
2. **Single image per turn** — multi-image analysis not supported by MedGemma 4B
3. **Scanned PDF OCR quality** — depends on scan DPI; 300 DPI minimum recommended
4. **Merged table cells** — complex lab panels may have misaligned values in extracted text
5. **4-bit quantization spatial accuracy** — lesion localization accuracy may be 10–15% lower than full-precision
6. **7-day URL TTL** — multi-turn file re-attachment relies on this; turns beyond 7 days fall back to extracted text only
7. **Max image resolution** — MedGemma caps at 896×896; large scans are downsampled automatically by the API layer

---

## 10. Non-Goals

- **DICOM support** — out of scope for ATT-1; log as ATT-2
- **Multi-image per turn** — MedGemma does not support this; out of scope
- **Real-time document streaming** — extraction runs synchronously inside the agent; no streaming of extraction progress
- **Document editing or annotation** — read-only analysis only
- **Optical character recognition quality tuning** — Tesseract config tuning is out of scope; default settings acceptable for MVP
- **Frontend attachment gallery or document viewer** — attachment badges only; no inline PDF rendering
- **Auth on MinIO uploads** — presigned URLs with 7-day expiry are sufficient for MVP

---

## 11. Open Questions — Resolved

| # | Question | Decision | Rationale |
|---|---|---|---|
| Q1 | Does the locally-served MedGemma endpoint use the OpenAI-compatible message format or a custom payload? | **Custom payload confirmed** — `{"prompt": ..., "image_base64": ...}`. No change needed to `extract_image_findings`. | Confirmed from codebase: `medgemma_llm.py` sends a custom payload to `MEDGEMMA_API_URL`. The Ollama OpenAI-compatible format (`image_url` content blocks) is only relevant if the server is replaced by Ollama-served MedGemma in future. |
| Q2 | What is the maximum file size the MinIO upload endpoint should accept? | **20 MB for images, 50 MB for PDFs.** Enforce at the upload endpoint (HTTP 413 if exceeded). Also enforce in frontend before upload attempt. | Prevents runaway MinIO storage costs and downstream tool timeouts. MedGemma vision scales image to 896×896 regardless of input size, so large files add no quality benefit. |
| Q3 | Should `extracted_document_text` be re-injected for all history turns or only the most recent? | **Session document memory store** — see resolution below. | Detailed in Q3 Resolution section. |
| Q4 | Should the aggregator label document-sourced vs. general-knowledge content? | **Yes — section header pattern, mandatory for HIPAA auditability.** | Detailed in Q4 Resolution section. |
| Q5 | If `report_analyzer` fails (e.g., expired URL), fail hard or fall back? | **Graceful failure with multi-hop re-upload prompt.** | Detailed in Q5 Resolution section. |

---

### Q3 Resolution — Session Document Memory Store

**Decision**: Introduce a **session-scoped document memory** stored in `message_metadata` at the session level, with a 6000-char total cap across all attachments in a session.

**Approach**:
- When `node_extract_attachments` completes, write extracted text into the **user message's** `message_metadata["extracted_document_text"]` (avoids needing a join with the assistant message later).
- On each new turn, the history re-injection loop scans all prior user messages for `extracted_document_text`. It concatenates them in order, most recent first, up to a **6000-char session budget**.
- Each prior document's contribution is prefixed with a timestamp + filename stub: `[Document from Turn N — {filename}]: {extracted_text}`.
- If the session budget is exhausted, older documents are truncated first (most recent wins).

**Why this approach**:
- **Quality**: 6000 chars covers ~3–4 full lab reports or ~6–8 pages of clinical notes — sufficient for most clinical sessions without overflowing the LLM context window.
- **No new infrastructure**: Stored in existing JSONB `message_metadata`; no Redis or separate table needed.
- **MinIO independence**: Text is stored after extraction, so subsequent turns do not depend on URL TTL (presigned URLs expire in 7 days; extracted text persists indefinitely in DB).
- **Auditability**: Every extraction event is timestamped and tied to a specific user message — full provenance chain in DB.

**Implementation location**: `orchestrator.py` — history re-injection loop (~line 1102) + streaming endpoint where `msg_metadata` is written.

---

### Q4 Resolution — Source Attribution Labeling

**Decision**: Aggregator responses that incorporate document-extracted content **must** include a `Based on uploaded document` section header above document-sourced content. General knowledge content is unmarked (industry default).

**Industry standard basis**:
- **ONC HTI-1 Final Rule (2024)**: Certified EHR CDS must display or make available its evidentiary basis. Section labeling satisfies this requirement.
- **FDA CDS Guidance (2022)**: Non-device CDS should make its basis "available to users" — a strong regulatory nudge toward source labeling.
- **CDS Hooks standard** (HL7): `source.label` is a required field on CDS response cards — the closest interoperability-level mandate for source attribution on AI output.
- **Production tools** (Epic, Azure Health Bot): Use inline section headers or `[Doc]`/`[Chart]` badges. **Section headers are the dominant pattern** in chatbot-style medical AI interfaces.

**Implementation**:
- Inject a directive into the `node_aggregator` system prompt:
  ```
  ATTRIBUTION RULE: When your response draws on content extracted from an uploaded
  document, attribute it naturally inline — e.g., "Based on your uploaded report, your
  LDL is 185 mg/dL..." or "According to the attached discharge summary, ...". Do not
  use section headers or badges. General medical knowledge requires no attribution.
  ```
- The aggregator already receives `extracted_document_context` in the agent outputs — this rule tells it to weave attribution naturally into the prose rather than marking it structurally.
- This serves dual purpose: (1) clinical transparency and readability for the user, (2) internal audit trail — since the attribution phrase appears in the stored `chat_messages.content`, the provenance is recoverable from the DB without any separate metadata field.

---

### Q5 Resolution — report_analyzer Failure Handling

**Decision**: **Graceful failure with multi-hop re-upload prompt** — do not fail the entire turn; instead, notify the user and continue with a no-document pipeline run if possible.

**Failure modes and handling**:

| Failure | Cause | Response |
|---|---|---|
| MinIO URL expired (>7 days) | Presigned URL TTL exceeded | Return clarification: *"I wasn't able to access your previously uploaded file — the link may have expired. Could you upload it again?"* Set `is_clarification=True`, `retrieval_ambiguous=False`. |
| MinIO URL unreachable (service down) | MinIO offline | Log error with `trace_id`. Return clarification: *"I'm having trouble accessing the attached file right now. You can continue with your question, or try uploading again."* Fall back to text-only pipeline run. |
| Extraction tool error (corrupt PDF, unsupported encoding) | File content issue | Log error. Return: *"I couldn't extract content from the attached file — it may be corrupted or in an unsupported format."* Fail the turn (do not attempt text-only run, as the user's intent was document-specific). |
| MedGemma vision timeout | Inference server unresponsive | Fall back to Gemma 4 (text-only) for synthesis. Log `"MedGemma vision timeout — falling back to text-only analysis"`. Response notes: *"Image analysis is currently unavailable; response based on general knowledge only."* |

**Auditability**: All failure events must be logged with `trace_id`, `session_id`, `file_url`, and failure reason. Store failure state in `message_metadata["attachment_failure"]` for dashboard observability (aligns with OBS-1).

**UX rationale**: Failing hard and returning an HTTP 500 loses the conversation context and forces the user to start over. The multi-hop re-upload pattern preserves session continuity, which is critical when a clinician is mid-workflow reviewing a patient record.

---

*PRD created: 2026-04-23 | Updated: 2026-04-23 (Q1–Q5 resolved) | Based on codebase audit + research into MedGemma 1.5 4B vision capabilities, pymupdf4llm 0.2.9, and medical AI source attribution standards (ONC HTI-1, FDA CDS Guidance 2022, HL7 CDS Hooks)*
