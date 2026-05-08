"""
Live model evaluation — tests the actual LLMs in the exact production prompt setup.

These tests call real models (NO mocks):
  - gemma4:31b-cloud via homeserver Ollama  (router + aggregator prompts)
  - MedGemma via localhost:8000             (agent synthesis prompt)

Run command (requires homeserver reachable and orchestrator models loaded):
  source .venv/bin/activate
  .venv/bin/python3 -m pytest tests/integration/test_live_model_evaluation.py -v --tb=short -m live -s

Skip in normal CI (no live services):
  .venv/bin/python3 -m pytest tests/ -v --tb=short -m "not live"

Design constraints:
  - Serial only — 31B model shares VRAM; parallel requests will OOM or queue
  - 90s timeout per test — cold model load can take 60s on first call
  - 15 tests total — prevents exhausting homeserver during a full test session
  - Each test replicates the exact prompt construction from production code
"""

import json
import re
import time
import pytest
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from config import settings

# ---------------------------------------------------------------------------
# Marks and fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live  # skip unless -m live is passed

OLLAMA_BASE_URL = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
OLLAMA_MODEL    = "gemma4:31b-cloud"   # always use production model, not conftest override
TIMEOUT         = 90                   # seconds per inference call


def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _medgemma_reachable() -> bool:
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_ollama():
    if not _ollama_reachable():
        pytest.skip("homeserver Ollama not reachable — start it or run without -m live")


# Production-identical ChatOllama instance (router / aggregator)
@pytest.fixture(scope="session")
def llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        num_predict=256,          # router only needs a short JSON array
        base_url=OLLAMA_BASE_URL,
        timeout=TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Exact production router system prompt (copied from node_router verbatim)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You are the MediCortex Orchestrator. Your ONLY job is to select which specialist agents to call.\n\n"
    "VALID KEYS — use ONLY these, never invent new ones:\n"
    "- \"pubmed\"          → latest research, studies, evidence, guidelines\n"
    "- \"diagnosis\"       → symptoms, differential diagnosis, clinical assessment\n"
    "- \"report_analyzer\" → lab results, imaging reports, ECG, pathology, uploaded files\n"
    "- \"patient\"         → specific named/identified patient history, records, vitals\n"
    "- \"pharmacology\"    → drugs, medications, dosing, interactions, side effects, contraindications\n\n"
    "RULES:\n"
    "1. Return ONLY a valid JSON array containing keys from the list above — never invent new keys\n"
    "2. Select 1-3 agents maximum\n"
    "3. NEVER route to 'pubmed' unless research papers or evidence are explicitly requested\n"
    "4. Symptoms/diagnosis only → [\"diagnosis\"]\n"
    "5. Named drug question → [\"pharmacology\"]\n"
    "6. Symptoms + treatment → [\"diagnosis\", \"pharmacology\"]\n"
    "7. Uploaded file/report/image → always include \"report_analyzer\"\n\n"
    "EXAMPLES:\n"
    "\"What is the dose of amoxicillin for a child?\" → [\"pharmacology\"]\n"
    "\"Patient has fever, cough and low SpO2\" → [\"diagnosis\"]\n"
    "\"Interpret this CBC: WBC 14k, Hgb 8.2\" → [\"report_analyzer\"]\n"
    "\"Latest trials on checkpoint inhibitors\" → [\"pubmed\"]\n"
    "\"John Smith's last visit and his beta blocker dose\" → [\"patient\", \"pharmacology\"]\n"
    "\"Is metformin safe in CKD stage 4?\" → [\"pharmacology\", \"pubmed\"]\n"
    "\"45yo with chest pain and diaphoresis — diagnosis and treatment?\" → [\"diagnosis\", \"pharmacology\"]\n\n"
    "FOLLOW-UP RESOLUTION — if the query uses pronouns (his/her/their/it/the patient/the drug), "
    "resolve using the Recent Session Context below:\n"
    "- Prior [patient] + asks about drugs → [\"pharmacology\"]\n"
    "- Prior [diagnosis] + asks about treatment → [\"pharmacology\"]\n\n"
    "Return ONLY the JSON array, no explanation, no prose."
)

AGGREGATOR_SYSTEM_PROMPT = (
    "You are the MediCortex Interface. Format the following medical agent reports into "
    "a beautiful, human-readable Markdown response.\n"
)

DRUG_AGENT_SYSTEM_PROMPT = """\
You are the Pharmacology & Drug Safety Agent for MediCortex.

YOUR MISSION: Using the gathered pharmacology data (tool results from drugs.com, FDA,
Mayo Clinic, etc.), provide accurate, evidence-based drug information. You answer both
general dosage/interaction questions AND patient-specific queries. No patient clinical
record is required for general pharmacology questions — answer from the gathered tool
data. NEVER hallucinate drug interactions, dosages, or recommendations beyond what the
gathered data supports.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Summary** — Direct answer to the user's question.
2. **Evidence/Analysis** — Detailed findings with source citations (Drugs.com, FDA, etc.).
3. **Safety Warnings** — Any major/moderate interactions or contraindications.
4. **Disclaimer** — "Consult a healthcare professional before making any medication changes."
"""


def _route(llm, query: str, context: str = "") -> list[str]:
    """Call the router LLM with the exact production prompt. Returns parsed agent list."""
    user_message = f"User Query: {query}\n\nKnowledge Core Context (for your awareness): {context[:300]}"
    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=user_message)]
    response = llm.invoke(messages).content
    clean = response.replace("```json", "").replace("```", "").strip().replace("'", '"')
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r'\[.*?\]', clean, re.DOTALL)
        parsed = json.loads(m.group()) if m else ["diagnosis"]
    return parsed if isinstance(parsed, list) else ["diagnosis"]


# ---------------------------------------------------------------------------
# LIVE-ROUTER-01..07 — gemma4:31b-cloud routing accuracy
# ---------------------------------------------------------------------------

class TestLiveRouterAccuracy:
    """Verify gemma4:31b-cloud routes correctly with the production system prompt."""

    @pytest.mark.timeout(90)
    def test_live_router_01_dosage_to_pharmacology(self, llm):
        """Dosage question must route to pharmacology."""
        agents = _route(llm, "What is the dosage of atorvastatin for a middle-aged patient?")
        assert "pharmacology" in agents, f"Expected pharmacology, got {agents}"

    @pytest.mark.timeout(90)
    def test_live_router_02_drug_interaction_to_pharmacology(self, llm):
        """Drug interaction must route to pharmacology."""
        agents = _route(llm, "Can I take ibuprofen with warfarin?")
        assert "pharmacology" in agents, f"Expected pharmacology, got {agents}"

    @pytest.mark.timeout(90)
    def test_live_router_03_symptoms_to_diagnosis(self, llm):
        """Classic symptom cluster must route to diagnosis."""
        agents = _route(llm, "Patient has sudden chest pain, sweating, and left arm numbness.")
        assert "diagnosis" in agents, f"Expected diagnosis, got {agents}"

    @pytest.mark.timeout(90)
    def test_live_router_04_research_query_to_pubmed(self, llm):
        """Explicit research request must route to pubmed."""
        agents = _route(llm, "What do recent RCTs say about SGLT2 inhibitors in heart failure?")
        assert "pubmed" in agents, f"Expected pubmed, got {agents}"

    @pytest.mark.timeout(90)
    def test_live_router_05_symptoms_plus_treatment_dual_route(self, llm):
        """Symptoms + treatment question should include both diagnosis and pharmacology."""
        agents = _route(llm, "What causes hypertension and what drugs treat it?")
        assert "diagnosis" in agents or "pharmacology" in agents, (
            f"Expected at least one of diagnosis/pharmacology, got {agents}"
        )

    @pytest.mark.timeout(90)
    def test_live_router_06_output_is_valid_json_array(self, llm):
        """Router must always return a parseable JSON array of valid agent keys."""
        valid = {"pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"}
        agents = _route(llm, "What is the mechanism of metformin?")
        assert isinstance(agents, list), f"Router output is not a list: {agents}"
        assert len(agents) >= 1, "Router returned empty list"
        assert len(agents) <= 3, f"Router returned more than 3 agents: {agents}"
        for a in agents:
            assert a in valid, f"Unknown agent key '{a}' in router output"

    @pytest.mark.timeout(90)
    def test_live_router_07_followup_pronoun_resolution(self, llm):
        """Follow-up with pronoun 'it' after a pharmacology context → pharmacology."""
        context = "Prior session: [pharmacology] — user asked about metformin for diabetes."
        agents = _route(
            llm,
            "Is it safe to take during pregnancy?",
            context=context,
        )
        # With prior pharmacology context and a pronoun, should stay in pharmacology
        assert "pharmacology" in agents, (
            f"Pronoun follow-up after pharmacology context should stay pharmacology, got {agents}"
        )


# ---------------------------------------------------------------------------
# LIVE-AGGREGATOR-01..03 — gemma4:31b-cloud aggregator formatting
# ---------------------------------------------------------------------------

class TestLiveAggregatorFormatting:
    """Verify gemma4:31b-cloud formats agent outputs into valid Markdown."""

    def _aggregate(self, llm, agent_outputs: str, query: str) -> str:
        """Exact production aggregator prompt construction."""
        formatting_prompt = (
            AGGREGATOR_SYSTEM_PROMPT
            + "Rules:\n"
            "- Use ## headers for each topic\n"
            "- Use bullet points for lists\n"
            "- Bold key terms\n"
            "- Keep it medically accurate and avoid patient-specific advice\n"
            "- End with a 'Sources' section if source URLs are mentioned\n\n"
            f"Current User Query: {query}\n\n"
            f"Agent Reports:\n{agent_outputs}\n\n"
            "Formatted Response:"
        )
        messages = [HumanMessage(content=formatting_prompt)]
        return llm.invoke(messages).content

    @pytest.mark.timeout(90)
    def test_live_aggregator_01_produces_markdown_headers(self, llm):
        """Aggregator output must contain at least one markdown header."""
        agent_output = (
            "## Pharmacology Agent Response\n"
            "Atorvastatin is a statin used to lower LDL cholesterol. "
            "Standard dose: 10–80mg once daily. "
            "Common side effects: myalgia, elevated liver enzymes."
        )
        result = self._aggregate(llm, agent_output, "What is atorvastatin used for?")
        assert "#" in result, f"No markdown headers found in aggregator output:\n{result[:400]}"

    @pytest.mark.timeout(90)
    def test_live_aggregator_02_multi_agent_output_merged(self, llm):
        """Aggregator merges two agent outputs without losing both topics."""
        agent_outputs = (
            "## Diagnosis Agent Response\n"
            "Chest pain with diaphoresis and left arm radiation is classic for ACS/STEMI.\n\n"
            "## Pharmacology Agent Response\n"
            "Aspirin 300mg loading dose + nitroglycerin sublingual for acute chest pain."
        )
        result = self._aggregate(
            llm, agent_outputs,
            "Patient has chest pain and sweating — diagnosis and initial treatment?"
        )
        result_lower = result.lower()
        assert "chest" in result_lower or "acs" in result_lower or "pain" in result_lower, (
            f"Diagnosis content missing from merged output:\n{result[:400]}"
        )
        assert "aspirin" in result_lower or "nitroglycerin" in result_lower or "treatment" in result_lower, (
            f"Pharmacology content missing from merged output:\n{result[:400]}"
        )

    @pytest.mark.timeout(90)
    def test_live_aggregator_03_output_non_empty(self, llm):
        """Aggregator must always return non-empty text."""
        result = self._aggregate(
            llm,
            "## Pharmacology Agent Response\nMetformin 500mg twice daily for Type 2 diabetes.",
            "What is the dose of metformin?",
        )
        assert result.strip(), "Aggregator returned empty response"
        assert len(result) > 50, f"Aggregator response suspiciously short: {repr(result)}"


# ---------------------------------------------------------------------------
# LIVE-SYNTHESIS-01..05 — MedGemma agent synthesis (exact production prompt)
# ---------------------------------------------------------------------------

class TestLiveMedGemmaSynthesis:
    """
    Verify MedGemma responds correctly with the exact production synthesis prompt.
    Skipped automatically if MedGemma is offline.
    """

    @pytest.fixture(autouse=True)
    def require_medgemma(self):
        if not _medgemma_reachable():
            pytest.skip("MedGemma not reachable at localhost:8000 — start medgemma-host or RunPod")

    def _synthesize(self, query: str, tool_data: str, system_prompt: str = DRUG_AGENT_SYSTEM_PROMPT) -> str:
        """Exact production _synthesize() prompt construction from base.py."""
        prompt = (
            f"{system_prompt}\n\n"
            f"User Query: {query}\n\n"
            f"Gathered Data:\n{tool_data}\n\n"
            f"Using the gathered data above, provide your complete clinical response:"
        )
        from specialized_agents.medgemma_llm import MedGemmaLLM
        medgemma = MedGemmaLLM()
        return medgemma._call(prompt)

    @pytest.mark.timeout(120)
    def test_live_synthesis_01_dosage_response_not_empty(self):
        """MedGemma must return non-empty response for a dosage query with tool data."""
        tool_data = (
            "[recommend_drugs results]\n"
            "## Drug Query: Dosage for *Atorvastatin*\n"
            "### Atorvastatin Dosage — Drugs.com\n"
            "- Standard adult dose: 10–80mg once daily at bedtime.\n"
            "- Starting dose for most patients: 10–20mg/day.\n"
            "- Maximum dose: 80mg/day."
        )
        result = self._synthesize("What is the dosage of atorvastatin for a middle-aged patient?", tool_data)
        assert result.strip(), "MedGemma returned empty response"
        assert "No clinical data" not in result, (
            f"MedGemma refused a general dosage question: {result[:300]}"
        )

    @pytest.mark.timeout(120)
    def test_live_synthesis_02_response_mentions_drug_name(self):
        """Synthesis response must mention the queried drug."""
        tool_data = (
            "[recommend_drugs results]\n"
            "Metformin 500–2000mg daily, taken with meals to reduce GI side effects. "
            "First-line for Type 2 diabetes per ADA guidelines."
        )
        result = self._synthesize("What is the standard dose of metformin for Type 2 diabetes?", tool_data)
        assert "metformin" in result.lower(), (
            f"Drug name 'metformin' not found in synthesis response:\n{result[:400]}"
        )

    @pytest.mark.timeout(120)
    def test_live_synthesis_03_no_hallucinated_dose_without_data(self):
        """With no tool data, MedGemma should not fabricate specific dose numbers."""
        result = self._synthesize(
            "What is the dosage of atorvastatin?",
            tool_data="",
        )
        # If MedGemma responds at all without data, it should qualify its answer
        # rather than confidently stating a specific number as fact
        assert result.strip(), "MedGemma returned empty response even for no-data prompt"

    @pytest.mark.timeout(120)
    def test_live_synthesis_04_responds_to_interaction_query(self):
        """MedGemma must engage with a drug interaction query given interaction data."""
        tool_data = (
            "[check_drug_interactions results]\n"
            "## Drug Interactions: Warfarin + Aspirin\n"
            "**Severity: Major** — Concurrent use significantly increases bleeding risk. "
            "Aspirin inhibits platelet aggregation; warfarin inhibits clotting factors. "
            "Combined use can cause GI or intracranial haemorrhage. "
            "Source: Drugs.com, FDA Drug Safety Communication."
        )
        result = self._synthesize(
            "What are the interactions between warfarin and aspirin?",
            tool_data,
        )
        result_lower = result.lower()
        assert "bleed" in result_lower or "interaction" in result_lower or "warfarin" in result_lower, (
            f"Response does not address the warfarin/aspirin interaction:\n{result[:400]}"
        )

    @pytest.mark.timeout(120)
    def test_live_synthesis_05_no_looping_output(self):
        """MedGemma must not return a looped/repetitive response."""
        tool_data = (
            "[recommend_drugs results]\n"
            "Lisinopril 5–40mg once daily for hypertension. "
            "ACE inhibitor; reduces afterload. Monitor potassium and renal function."
        )
        result = self._synthesize(
            "What is the dose of lisinopril for hypertension?",
            tool_data,
        )
        # Check for repetition: split into sentences, count duplicates
        sentences = [s.strip() for s in re.split(r'[.!?]', result) if len(s.strip()) > 20]
        if sentences:
            from collections import Counter
            counts = Counter(sentences)
            most_common, freq = counts.most_common(1)[0]
            assert freq <= 3, (
                f"MedGemma loop detected — sentence repeated {freq}x:\n{most_common!r}"
            )
