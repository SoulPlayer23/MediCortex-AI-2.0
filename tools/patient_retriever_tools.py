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

import json
import re

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger("PatientRetrieverTool")

# ── Simulated Patient Database ──────────────────────────────────────
# In production, this would query a real EHR/patient records table.
# The keys are lowercase for case-insensitive matching.
_SIMULATED_PATIENTS = {
    "john smith": {
        "patient_id": "PT-10042",
        "age": 45,
        "sex": "Male",
        "blood_type": "O+",
        "diagnoses": [
            {"condition": "Type 2 Diabetes", "diagnosed": "2019-03-15", "status": "Active"},
            {"condition": "Hypertension", "diagnosed": "2018-07-20", "status": "Active"},
            {"condition": "Hyperlipidemia", "diagnosed": "2020-01-10", "status": "Managed"},
        ],
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily"},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "Once daily at bedtime"},
        ],
        "allergies": ["Penicillin", "Sulfa drugs"],
        "last_visit": "2026-01-28",
        "vitals_last_recorded": {
            "blood_pressure": "138/88 mmHg",
            "heart_rate": "76 bpm",
            "weight": "92 kg",
            "bmi": "28.4",
        },
    },
    "jane doe": {
        "patient_id": "PT-10078",
        "age": 32,
        "sex": "Female",
        "blood_type": "A+",
        "diagnoses": [
            {"condition": "Asthma", "diagnosed": "2010-06-01", "status": "Active"},
            {"condition": "Iron Deficiency Anemia", "diagnosed": "2024-11-15", "status": "Under Treatment"},
        ],
        "medications": [
            {"name": "Albuterol Inhaler", "dosage": "90mcg", "frequency": "As needed"},
            {"name": "Ferrous Sulfate", "dosage": "325mg", "frequency": "Once daily"},
        ],
        "allergies": ["Aspirin"],
        "last_visit": "2026-02-10",
        "vitals_last_recorded": {
            "blood_pressure": "118/74 mmHg",
            "heart_rate": "68 bpm",
            "weight": "58 kg",
            "bmi": "22.1",
        },
    },
    "raj patel": {
        "patient_id": "PT-10135",
        "age": 60,
        "sex": "Male",
        "blood_type": "B+",
        "diagnoses": [
            {"condition": "Coronary Artery Disease", "diagnosed": "2017-09-05", "status": "Stable"},
            {"condition": "Type 2 Diabetes", "diagnosed": "2015-04-12", "status": "Active"},
            {"condition": "Chronic Kidney Disease Stage 3", "diagnosed": "2022-08-20", "status": "Monitored"},
        ],
        "medications": [
            {"name": "Aspirin", "dosage": "81mg", "frequency": "Once daily"},
            {"name": "Metoprolol", "dosage": "50mg", "frequency": "Twice daily"},
            {"name": "Insulin Glargine", "dosage": "20 units", "frequency": "Once daily at bedtime"},
            {"name": "Losartan", "dosage": "50mg", "frequency": "Once daily"},
        ],
        "allergies": [],
        "last_visit": "2026-02-01",
        "vitals_last_recorded": {
            "blood_pressure": "142/90 mmHg",
            "heart_rate": "72 bpm",
            "weight": "85 kg",
            "bmi": "27.8",
        },
    },
}


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


def _redact_record(record: dict, real_name: str, placeholder: str) -> str:
    """Format a patient record and re-redact the real name with the placeholder."""
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
    for k, v in record["vitals_last_recorded"].items():
        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

    lines.append("")
    lines.append("### Diagnoses")
    for d in record["diagnoses"]:
        lines.append(f"- {d['condition']} (Since: {d['diagnosed']}, Status: {d['status']})")

    lines.append("")
    lines.append("### Current Medications")
    for m in record["medications"]:
        lines.append(f"- {m['name']} {m['dosage']} — {m['frequency']}")

    result = "\n".join(lines)

    # Re-redact: replace any occurrence of the real name with the placeholder
    if real_name and placeholder:
        result = result.replace(real_name, placeholder)

    return result


@tool
def retrieve_patient_records(redacted_identifier: str, pii_mapping_json: str = "{}") -> str:
    """Retrieve patient records from the database using a redacted patient identifier.

    This tool is HIPAA-compliant: it accepts a redacted identifier (e.g., '<PERSON_1>')
    and internally resolves it to the real patient name/ID for database lookup. The
    returned record is DE-IDENTIFIED — the real name is replaced with the redacted
    placeholder before being returned to the agent.

    Use this tool when the user asks about a specific patient's records, demographics,
    diagnoses, medications, or vitals.

    Args:
        redacted_identifier: The patient name placeholder (e.g., '<PERSON_1>') or patient ID.
        pii_mapping_json: JSON-encoded dict mapping placeholders to real values. Passed
                          automatically by the orchestrator.

    Returns:
        De-identified patient record with demographics, diagnoses, medications, and vitals.
    """
    logger.info("patient_retrieval_start", redacted_identifier=redacted_identifier)

    # Step 1: Resolve the placeholder to the real name/ID
    real_identifier = _resolve_identifier(redacted_identifier, pii_mapping_json)
    logger.info("patient_identifier_resolved")  # Do NOT log real_identifier (HIPAA)

    # Step 2: Look up patient (case-insensitive)
    patient = _SIMULATED_PATIENTS.get(real_identifier.lower())

    if not patient:
        # Also try matching by patient_id
        for _, record in _SIMULATED_PATIENTS.items():
            if record["patient_id"].lower() == real_identifier.lower():
                patient = record
                break

    if not patient:
        logger.info("patient_not_found", redacted_identifier=redacted_identifier)
        return (
            f"No patient records found for identifier '{redacted_identifier}'. "
            f"Please verify the patient name or ID and try again."
        )

    # Step 3: Format and re-redact before returning
    result = _redact_record(patient, real_identifier, redacted_identifier)
    logger.info("patient_retrieval_complete", redacted_identifier=redacted_identifier)
    return result
