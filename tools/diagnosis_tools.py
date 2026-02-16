from langchain_core.tools import tool
import logging

logger = logging.getLogger("SpecializedTools")

@tool
def consult_medical_guidelines(symptoms: str) -> str:
    """Consults standard medical guidelines for given set of symptoms."""
    logger.info(f"🩺 [Diagnosis] Checking guidelines for: {symptoms}")
    return f"Standard guidelines for {symptoms} suggest differential diagnosis of X, Y, and Z. Recommend ruling out infection."
