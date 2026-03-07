import sys
from specialized_agents.medgemma_llm import MedGemmaLLM

def test_medgemma():
    print("Testing MedGemmaLLM...")
    llm = MedGemmaLLM()
    prompt = "Thought: Do I need to use a tool? Yes\nAction: analyze_symptoms\nAction Input: symptoms of Type 2 Diabetes"
    
    print(f"Prompt: {prompt}")
    print("\nInvoking MedGemma...")
    try:
        response = llm.invoke(prompt)
        print("\n--- RESPONSE ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_medgemma()
