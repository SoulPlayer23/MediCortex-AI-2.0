from langchain_core.tools import tool
import logging

logger = logging.getLogger("SpecializedTools")

@tool
def search_patient_records(patient_id_or_name: str) -> str:
    """Retrieves patient history and electronic health records (EHR)."""
    logger.info(f"👤 [Patient] Retrieving records for: {patient_id_or_name}")
    return f"Patient Record [ID: {patient_id_or_name}]: 45YO Male, History of Hypertension, Metformin 500mg/day."
