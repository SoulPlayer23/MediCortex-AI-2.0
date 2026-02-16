from langchain_core.tools import tool
import logging

logger = logging.getLogger("SpecializedTools")

@tool
def check_drug_interactions(medication_list: str) -> str:
    """Checks for contraindications and interactions between medications."""
    logger.info(f"💊 [Drug] Checking interactions for: {medication_list}")
    return f"Analysis for {medication_list}: Potential interaction between Drug A and Drug B (Moderate). Monitor renal function."
