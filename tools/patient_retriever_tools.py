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
        "vitals_history": [
            {
                "date": "2026-01-28",
                "blood_pressure": "138/88 mmHg",
                "heart_rate": "76 bpm",
                "temperature": "98.6°F",
                "respiratory_rate": "16/min",
                "spo2": "97%",
                "weight": "92 kg",
                "bmi": "28.4",
            },
            {
                "date": "2025-10-15",
                "blood_pressure": "142/92 mmHg",
                "heart_rate": "80 bpm",
                "temperature": "98.4°F",
                "respiratory_rate": "18/min",
                "spo2": "96%",
                "weight": "94 kg",
                "bmi": "29.0",
            },
        ],
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
        "vitals_history": [
            {
                "date": "2026-02-10",
                "blood_pressure": "118/74 mmHg",
                "heart_rate": "68 bpm",
                "temperature": "98.2°F",
                "respiratory_rate": "14/min",
                "spo2": "98%",
                "weight": "58 kg",
                "bmi": "22.1",
            },
        ],
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
        "vitals_history": [
            {
                "date": "2026-02-01",
                "blood_pressure": "142/90 mmHg",
                "heart_rate": "72 bpm",
                "temperature": "98.8°F",
                "respiratory_rate": "17/min",
                "spo2": "95%",
                "weight": "85 kg",
                "bmi": "27.8",
            },
            {
                "date": "2025-11-10",
                "blood_pressure": "148/94 mmHg",
                "heart_rate": "78 bpm",
                "temperature": "98.6°F",
                "respiratory_rate": "18/min",
                "spo2": "94%",
                "weight": "87 kg",
                "bmi": "28.5",
            },
            {
                "date": "2025-08-20",
                "blood_pressure": "150/96 mmHg",
                "heart_rate": "82 bpm",
                "temperature": "98.4°F",
                "respiratory_rate": "19/min",
                "spo2": "94%",
                "weight": "89 kg",
                "bmi": "29.1",
            },
        ],
    },
    "maria garcia": {
        "patient_id": "PT-10201",
        "age": 55,
        "sex": "Female",
        "blood_type": "AB+",
        "diagnoses": [
            {"condition": "Rheumatoid Arthritis", "diagnosed": "2016-02-18", "status": "Active"},
            {"condition": "Osteoporosis", "diagnosed": "2021-09-30", "status": "Under Treatment"},
            {"condition": "Hypothyroidism", "diagnosed": "2013-05-22", "status": "Managed"},
            {"condition": "Gastroesophageal Reflux Disease", "diagnosed": "2020-03-10", "status": "Active"},
        ],
        "medications": [
            {"name": "Methotrexate", "dosage": "15mg", "frequency": "Once weekly"},
            {"name": "Folic Acid", "dosage": "1mg", "frequency": "Once daily"},
            {"name": "Alendronate", "dosage": "70mg", "frequency": "Once weekly"},
            {"name": "Levothyroxine", "dosage": "75mcg", "frequency": "Once daily before breakfast"},
            {"name": "Omeprazole", "dosage": "20mg", "frequency": "Once daily"},
            {"name": "Calcium + Vitamin D", "dosage": "600mg/400IU", "frequency": "Twice daily"},
        ],
        "allergies": ["NSAIDs", "Codeine"],
        "last_visit": "2026-02-20",
        "vitals_history": [
            {
                "date": "2026-02-20",
                "blood_pressure": "128/82 mmHg",
                "heart_rate": "74 bpm",
                "temperature": "98.4°F",
                "respiratory_rate": "15/min",
                "spo2": "98%",
                "weight": "68 kg",
                "bmi": "25.6",
            },
        ],
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

    # Step 3: Build structured JSON (de-identified — uses placeholder, not real name)
    structured = {
        "patient_id": patient["patient_id"],
        "placeholder": redacted_identifier,
        "age": patient["age"],
        "sex": patient["sex"],
        "blood_type": patient["blood_type"],
        "allergies": patient["allergies"],
        "diagnoses": patient["diagnoses"],
        "medications": patient["medications"],
        "vitals_history": patient.get("vitals_history", []),
        "last_visit": patient["last_visit"],
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
