from .pubmed_agent import pubmed_agent
from .diagnosis_agent import diagnosis_agent
from .report_agent import report_agent
from .patient_agent import patient_agent
from .drug_agent import drug_agent

AGENT_REGISTRY = {
    "pubmed": pubmed_agent,
    "diagnosis": diagnosis_agent,
    "report": report_agent,
    "patient": patient_agent,
    "drug": drug_agent
}
