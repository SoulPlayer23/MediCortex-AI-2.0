"""
Patient Record Retriever Tool — HIPAA-compliant patient record lookup.

Accepts a redacted patient identifier (e.g., '<PERSON_1>') and a PII mapping
dictionary. Internally resolves the placeholder to the real value for database
lookup, then re-redacts the results before returning to the LLM agent.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information, input sanitization)
  - A2A (typed schemas via LangChain @tool)
"""

import asyncio
import json
import os
import re
from typing import Optional

import asyncpg
import structlog
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = structlog.get_logger("PatientRetrieverTool")

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/medicortex",
).replace("postgresql+asyncpg://", "postgresql://")


# ── Database helpers ─────────────────────────────────────────────────────────

_JSONB_FIELDS = {"address", "diagnoses", "medications", "allergies", "vitals_history"}


def _parse_row(row) -> dict:
    """Convert an asyncpg Record to a plain dict, decoding JSONB strings."""
    out = dict(row)
    for field in _JSONB_FIELDS:
        val = out.get(field)
        if isinstance(val, str):
            try:
                out[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    return out


async def _fetch_patient_async(real_identifier: str) -> Optional[dict]:
    """Query patients table by full_name (case-insensitive) or patient_id."""
    conn = await asyncpg.connect(_DATABASE_URL)
    try:
        row = await conn.fetchrow(
            """
            SELECT * FROM patients
            WHERE lower(full_name) = lower($1)
               OR patient_id = $1
            LIMIT 1
            """,
            real_identifier,
        )
        if row:
            return _parse_row(row)
        # Partial-name fallback
        row = await conn.fetchrow(
            "SELECT * FROM patients WHERE lower(full_name) LIKE lower($1) LIMIT 1",
            f"%{real_identifier}%",
        )
        return _parse_row(row) if row else None
    finally:
        await conn.close()


def _fetch_patient(real_identifier: str) -> Optional[dict]:
    """Sync wrapper safe for thread-pool contexts (LangGraph sync nodes)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_patient_async(real_identifier))
    finally:
        loop.close()


# ── (legacy simulated data removed — replaced by PostgreSQL patients table) ──

# (Simulated in-memory data removed — all patient lookups now query the PostgreSQL
#  `patients` table via _fetch_patient() above.)


def _resolve_identifier(redacted_identifier: str, pii_mapping_json: str) -> str:
    """Resolve a redacted placeholder to the real patient name/ID using PII mapping."""
    try:
        mapping = json.loads(pii_mapping_json) if pii_mapping_json else {}
    except json.JSONDecodeError:
        mapping = {}

    # If the identifier looks like a Presidio placeholder, resolve it
    if re.match(r"<\w+_\d+>", redacted_identifier.strip()):
        real_value = mapping.get(redacted_identifier.strip())
        if real_value:
            return real_value
        # Placeholder not found in mapping
        return redacted_identifier

    # Already a plain identifier (e.g., patient ID like "PT-10042")
    return redacted_identifier


def _format_record(record: dict, placeholder: str) -> str:
    """Format a patient record as human-readable Markdown with the placeholder."""
    latest_vitals = record["vitals_history"][0] if record.get("vitals_history") else {}

    lines = [
        f"## Patient Record [{placeholder}]",
        f"- **Patient ID:** {record['patient_id']}",
        f"- **Age:** {record['age']}",
        f"- **Sex:** {record['sex']}",
        f"- **Blood Type:** {record['blood_type']}",
        f"- **Allergies:** {', '.join(record['allergies']) if record['allergies'] else 'None known'}",
        f"- **Last Visit:** {record['last_visit']}",
        "",
        "### Vitals (Last Recorded)",
    ]
    for k, v in latest_vitals.items():
        if k == "date":
            continue
        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

    if len(record.get("vitals_history", [])) > 1:
        lines.append("")
        lines.append("### Vitals History")
        for visit in record["vitals_history"][1:]:
            lines.append(f"- **{visit.get('date', 'N/A')}**: BP {visit.get('blood_pressure', 'N/A')}, "
                         f"HR {visit.get('heart_rate', 'N/A')}, SpO2 {visit.get('spo2', 'N/A')}, "
                         f"Weight {visit.get('weight', 'N/A')}")

    lines.append("")
    lines.append("### Diagnoses")
    for d in record["diagnoses"]:
        lines.append(f"- {d['condition']} (Since: {d['diagnosed']}, Status: {d['status']})")

    lines.append("")
    lines.append("### Current Medications")
    for m in record["medications"]:
        lines.append(f"- {m['name']} {m['dosage']} — {m['frequency']}")

    return "\n".join(lines)


def _redact_output(text: str, real_name: str, placeholder: str) -> str:
    """Re-redact: replace any occurrence of the real name with the placeholder."""
    if real_name and placeholder:
        text = text.replace(real_name, placeholder)
        # Also handle case-insensitive replacement for partial matches
        text = re.sub(re.escape(real_name), placeholder, text, flags=re.IGNORECASE)
    return text


@tool
def retrieve_patient_records(redacted_identifier: str, pii_mapping_json: str = "{}") -> str:
    """Retrieve patient records from the database using a redacted patient identifier.

    This tool is HIPAA-compliant: it accepts a redacted identifier (e.g., '<PERSON_1>')
    and internally resolves it to the real patient name/ID for database lookup. The
    returned record is DE-IDENTIFIED — the real name is replaced with the redacted
    placeholder before being returned to the agent.

    The output includes both a structured JSON block (for downstream tool consumption)
    and a human-readable Markdown summary.

    Use this tool when the user asks about a specific patient's records, demographics,
    diagnoses, medications, or vitals.

    Args:
        redacted_identifier: The patient name placeholder (e.g., '<PERSON_1>') or patient ID.
        pii_mapping_json: JSON-encoded dict mapping placeholders to real values. Passed
                          automatically by the orchestrator.

    Returns:
        De-identified patient record with demographics, diagnoses, medications, vitals,
        and a structured JSON block for downstream tool use.
    """
    logger.info("patient_retrieval_start", redacted_identifier=redacted_identifier)

    # Step 1: Resolve the placeholder to the real name/ID
    real_identifier = _resolve_identifier(redacted_identifier, pii_mapping_json)
    logger.info("patient_identifier_resolved")  # Do NOT log real_identifier (HIPAA)

    # Step 2: Look up patient via PostgreSQL
    patient = _fetch_patient(real_identifier)

    if not patient:
        logger.info("patient_not_found", redacted_identifier=redacted_identifier)
        return (
            f"No patient records found for identifier '{redacted_identifier}'. "
            f"Please verify the patient name or ID and try again."
        )

    # Step 3: Build structured JSON (de-identified — uses placeholder, not real name)
    structured = {
        "patient_id": patient["patient_id"],
        "placeholder": redacted_identifier,
        "age": patient.get("age"),
        "sex": patient.get("sex"),
        "blood_type": patient.get("blood_type"),
        "allergies": patient.get("allergies") or [],
        "diagnoses": patient.get("diagnoses") or [],
        "medications": patient.get("medications") or [],
        "vitals_history": patient.get("vitals_history") or [],
        "last_visit": str(patient["last_visit"]) if patient.get("last_visit") else None,
    }

    # Step 4: Format human-readable markdown
    markdown = _format_record(patient, redacted_identifier)

    # Step 5: Combine structured + readable output
    result = (
        f"<!-- STRUCTURED_DATA_START -->\n"
        f"```json\n{json.dumps(structured, indent=2)}\n```\n"
        f"<!-- STRUCTURED_DATA_END -->\n\n"
        f"{markdown}"
    )

    # Step 6: Re-redact before returning
    result = _redact_output(result, real_identifier, redacted_identifier)
    logger.info("patient_retrieval_complete", redacted_identifier=redacted_identifier)
    return result
