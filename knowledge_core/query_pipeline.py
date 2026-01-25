
#!/usr/bin/env python3
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_core.medical_engine import MedicalReasoningEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 query_pipeline.py <medical_term>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    
    # Initialize Engine
    # Implicitly uses 'knowledge_core/assets' relative to medical_engine.py
    engine = MedicalReasoningEngine()
    
    # Execute Pipeline
    results = engine.search_and_reason(query)
    
    # Output formatting (matching request style)
    if results:
        print(f"   🏆 Context for Agent:")
        for f in results:
            print(f"      [{f['score']:.4f}] ... --[{f['relation']}]--> {f['name']} (Hop {f['hop']})")

if __name__ == "__main__":
    main()
