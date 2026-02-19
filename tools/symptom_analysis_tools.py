"""
Symptom Analysis Tool — Parses user-provided symptoms and combines them with
context from the knowledge core to provide a structured symptom profile.

Compliant with:
  - FastAPI (async httpx, structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import json
import structlog
from langchain_core.tools import tool
from specialized_agents.medgemma_llm import MedGemmaLLM

logger = structlog.get_logger("SymptomAnalysisTool")

# Initialize LLM for tool use
llm = MedGemmaLLM()

@tool
def analyze_symptoms(query: str, knowledge_context: str = "") -> str:
    """Analyze the user's query for specific symptoms, clinical signs, and patient details,
    combining them with any retrieved knowledge context.
    
    This tool uses the MedGemma medical LLM to structure the input into a clinical
    profile to aid in differential diagnosis. Use this tool FIRST to understand the
    clinical picture before searching for diagnoses.
    
    Args:
        query: The user's natural language query describing symptoms.
        knowledge_context: context string retrieved from the knowledge core (optional).
    
    Returns:
        A structured string summarizing the clinical profile.
    """
    logger.info("symptom_analysis_start", query=query, has_context=bool(knowledge_context))

    system_prompt = (
        "You are an expert clinical symptom analyzer. Your task is to extract and structure "
        "clinical information from the user's query and the provided knowledge context from the medical graph.\n\n"
        "Output ONLY a markdown-formatted Clinical Symptom Profile with the following sections:\n"
        "1. **Analysis**\n"
        "   - **Detected Severity**: (Assess if mild, moderate, severe, or life-threatening)\n"
        "   - **Explicit Symptoms**: (List all symptoms found)\n"
        "   - **Patient Demographics**: (Age, sex, etc. if mentioned)\n"
        "   - **Relevant History**: (Any past conditions mentioned)\n"
        "2. **Knowledge Core Context**\n"
        "   - Summarize the provided knowledge graph context relevant to these symptoms.\n"
        "3. **Guidance for Diagnosis**\n"
        "   - Suggest key conditions to investigate based on this profile.\n\n"
        "Do not provide a final diagnosis, just the structured analysis."
    )
    
    user_input = f"User Query: {query}\n\nKnowledge Context: {knowledge_context if knowledge_context else 'None provided.'}"
    
    try:
        # Use MedGemma to analyze
        prompt = f"{system_prompt}\n\n{user_input}"
        response = llm.invoke(prompt)
        
        logger.info("symptom_analysis_llm_complete")
        return response
        
    except Exception as e:
        logger.error("symptom_analysis_failed", error=str(e))
        return f"Error analyzing symptoms with MedGemma: {str(e)}"
