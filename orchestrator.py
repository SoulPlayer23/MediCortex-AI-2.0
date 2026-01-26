import os
import sys
import logging
from typing import Dict, TypedDict, List, Optional, Tuple
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("Orchestrator")

# --- Third Party Imports ---
logger.info("Importing LangChain & Presidio dependencies...")
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# --- Local Imports ---
logger.info("Importing Local Engines...")
try:
    from knowledge_core.medical_engine import MedicalReasoningEngine
    logger.info("Initializing MedicalReasoningEngine...")
    medical_engine = MedicalReasoningEngine()
    logger.info("✅ MedicalReasoningEngine connected.")
except ImportError:
    logger.warning("Could not import MedicalKnowledgeEngine. Knowledge retrieval will fail.")
    medical_engine = None

# ==========================================
# 🛡️ PRIVACY LAYER (HIPAA COMPLIANCE)
# ==========================================
class PrivacyManager:
    def __init__(self):
        logger.info("🛡️  Initializing HIPAA Privacy Layer (Presidio)...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        logger.info("   ✅ Presidio Engines Loaded.")

    def redact_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}

        logger.debug(f"scanning text: {text[:20]}...")
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME"],
            language='en'
        )
        
        mapping = {}
        results = sorted(results, key=lambda x: x.start, reverse=True)
        redacted_text = text
        type_counts = {}

        for result in results:
            entity_type = result.entity_type
            start, end = result.start, result.end
            original_value = text[start:end]
            
            count = type_counts.get(entity_type, 0) + 1
            type_counts[entity_type] = count
            placeholder = f"<{entity_type}_{count}>"
            
            mapping[placeholder] = original_value
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]
            
        logger.info(f"   🔒 Redacted {len(mapping)} entities.")
        return redacted_text, mapping

    def restore_privacy(self, text: str, mapping: Dict[str, str]) -> str:
        restored_text = text
        for placeholder, original_value in mapping.items():
            restored_text = restored_text.replace(placeholder, original_value)
        return restored_text

logger.info("Instantiating PrivacyManager Singleton...")
privacy_manager = PrivacyManager()

# ==========================================
# 🧠 AGENT STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    input: str
    redacted_input: str
    pii_mapping: Dict[str, str]
    context: List[str]
    messages: List[BaseMessage]
    final_output: str
    error: Optional[str]

# ==========================================
# 🛠️ TOOLS
# ==========================================
@tool
def consult_medical_knowledge(query: str) -> str:
    """Consults the structured medical knowledge graph."""
    logger.info(f"🛠️ [Tool] consult_medical_knowledge invoked: '{query}'")
    if not medical_engine:
        return "Knowledge Engine Offline."
    results = medical_engine.search_and_reason(query)
    formatted = [f"- {r['name']} ({r['relation']}, Hop: {r.get('hop', '?')})" for r in results]
    return "\n".join(formatted) if formatted else "No specific knowledge found in graph."

tools = [consult_medical_knowledge]

# ==========================================
# 🤖 OPENAI LLM SETUP
# ==========================================
try:
    logger.info("Initializing OpenAI LLM Client...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    ).bind_tools(tools)
    logger.info("✅ OpenAI Client Ready.")
except Exception as e:
    logger.error(f"WARNING: OpenAI setup failed: {e}")
    llm = None

# ==========================================
# 🕸️ LANGGRAPH NODES
# ==========================================
def node_analyze_privacy(state: AgentState):
    logger.info("--- NODE: ANALYZE PRIVACY ---")
    logger.info(f"Input: '{state['input']}'")
    
    redacted, mapping = privacy_manager.redact_pii(state['input'])
    logger.info(f"Redacted Input: '{redacted}'")
    
    return {
        "redacted_input": redacted,
        "pii_mapping": mapping,
        "messages": [HumanMessage(content=redacted)]
    }

def node_retrieve_knowledge(state: AgentState):
    logger.info("--- NODE: RETRIEVE KNOWLEDGE ---")
    query = state['redacted_input']
    
    # 1. Graph
    logger.info("1. Querying Knowledge Graph...")
    graph_context = consult_medical_knowledge.invoke(query)
    
    sys_msg = SystemMessage(content=f"You are MediCortex Orchestrator. Use the following verified local knowledge graph context if relevant:\n{graph_context}")
    return {
        "context": [graph_context],
        "messages": [sys_msg] + state["messages"]
    }

def node_agent_reasoning(state: AgentState):
    logger.info("--- NODE: AGENT REASONING (OpenAI) ---")
    if not llm:
        return {"final_output": "System Offline."}
    
    try:
        logger.info("Sending request to OpenAI API...")
        response = llm.invoke(state['messages'])
        logger.info("✅ Received response from OpenAI.")
        return {"messages": [response], "final_output": response.content}
    except Exception as e:
        logger.error(f"❌ OpenAI Error: {e}")
        return {"final_output": "System Error.", "error": str(e)}

def node_restore_privacy(state: AgentState):
    logger.info("--- NODE: RESTORE PRIVACY ---")
    raw_output = state.get("final_output", "")
    mapping = state.get("pii_mapping", {})
    
    restored = privacy_manager.restore_privacy(raw_output, mapping)
    logger.info("✅ PII Restored.")
    return {"final_output": restored}

# ==========================================
# 🚀 GRAPH CONSTRUCTION
# ==========================================
logger.info("Compiling LangGraph State Machine...")
workflow = StateGraph(AgentState)
workflow.add_node("analyze_privacy", node_analyze_privacy)
workflow.add_node("retrieve_knowledge", node_retrieve_knowledge)
workflow.add_node("agent_reasoning", node_agent_reasoning)
workflow.add_node("restore_privacy", node_restore_privacy)

workflow.set_entry_point("analyze_privacy")
workflow.add_edge("analyze_privacy", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "agent_reasoning")
workflow.add_edge("agent_reasoning", "restore_privacy")
workflow.add_edge("restore_privacy", END)

app = workflow.compile()
logger.info("✅ Orchestrator Graph Compiled.")

if __name__ == "__main__":
    logger.info("🚀 Orchestrator Standalone Test")
    # Using a simple token-efficient query
    user_input = "Briefly define hypertension."
    result = app.invoke({"input": user_input, "messages": []})
    print(f"\nFinal Result: {result.get('final_output')}")
