
import os
import pickle
import sys
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Constants
ARANGO_URL = "http://localhost:8529/_db/clinical_ontology/_api"
AUTH = ('root', 'arangopwd123')
DEFAULT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

class MedicalReasoningEngine:
    def __init__(self, asset_dir=None):
        print("⚙️  Initializing Medical Reasoning Engine (Optimized)...")
        self.asset_dir = asset_dir if asset_dir else DEFAULT_ASSET_DIR
        
        # A. Load Maps (Concepts & Synonyms)
        maps_path = os.path.join(self.asset_dir, "maps.pkl")
        print(f"   Loading Maps from {maps_path}...")
        try:
            with open(maps_path, "rb") as f:
                data = pickle.load(f)
                self.key_to_idx = data.get('key_to_idx', {})
                self.synonym_map = data.get('synonym_map', {})
            print(f"   ✅ Concepts Mapped: {len(self.key_to_idx)}")
            print(f"   ✅ Synonyms Mapped: {len(self.synonym_map)}")
        except Exception as e:
            print(f"❌ Error loading Maps: {e}")
            self.key_to_idx = {}
            self.synonym_map = {}

        # C. Load the "Brain" (RGCN Embeddings)
        vec_path = os.path.join(self.asset_dir, "vectors.npy")
        print(f"   Loading Embeddings from {vec_path}...")
        try:
            # Use mmap_mode='r' for instant loading
            self.embeddings = np.load(vec_path, mmap_mode='r')
            print(f"   ✅ Brain Loaded: {self.embeddings.shape} matrix.")
        except Exception as e:
            print(f"❌ Error loading Embeddings: {e}")
            self.embeddings = None

        print("✅ Engine Online.\n")

    def _aql(self, query, bind_vars=None):
        """Helper to execute AQL"""
        try:
            resp = requests.post(f"{ARANGO_URL}/cursor", json={"query": query, "bindVars": bind_vars or {}}, auth=AUTH)
            resp.raise_for_status()
            return resp.json().get('result', [])
        except Exception as e:
            print(f"   ⚠️ AQL Error: {e}")
            return []
            
    def fetch_node_by_id(self, node_id):
        # Helper to get node details
        aql = "RETURN DOCUMENT(CONCAT('concepts/', @id))"
        res = self._aql(aql, {"id": node_id})
        return res[0] if res else None

    def resolve_entity(self, user_query):
        """
        Smart Entity Linking: Synonyms -> Exact Match -> Fuzzy Match
        """
        query_lower = user_query.lower()
        
        # STRATEGY 1: Synonym Lookup
        if query_lower in self.synonym_map:
            concept_id = self.synonym_map[query_lower]
            node = self.fetch_node_by_id(concept_id)
            if node:
                print(f"   ✨ Synonym Hit: '{user_query}' mapped to '{node['name']}' (ID: {node['_key']})")
                return node

        # STRATEGY 2: Exact Match
        aql_exact = """
        FOR d IN concepts
          FILTER d.name == @q
          LIMIT 1
          RETURN d
        """
        res = self._aql(aql_exact, {"q": user_query})
        if res:
             print(f"   📍 Exact Match: '{res[0]['name']}' (ID: {res[0]['_key']})")
             return res[0]

        # STRATEGY 3: Fuzzy / Starts With
        aql_fuzzy = """
        FOR d IN concepts
          FILTER LIKE(d.name, CONCAT(@q, "%"), true)
          SORT LENGTH(d.name) ASC // Prefer shorter names
          LIMIT 1
          RETURN d
        """
        res = self._aql(aql_fuzzy, {"q": user_query})
        if res:
            print(f"   🔍 Fuzzy Match: '{res[0]['name']}' (ID: {res[0]['_key']})")
            return res[0]

        return None

    def search_and_reason(self, user_query, top_k=10):
        print(f"\n🔎 Query: '{user_query}'")
        
        # 1. Entity Linking
        anchor = self.resolve_entity(user_query)
        if not anchor:
            print("❌ Concept not found.")
            return []

        # Get Vector
        anchor_vec = None
        if anchor['_key'] in self.key_to_idx and self.embeddings is not None:
            idx = self.key_to_idx[anchor['_key']]
            anchor_vec = self.embeddings[idx].reshape(1, -1)
        else:
            print("   ⚠️ Warning: Anchor not in embedding matrix. Ranking disabled.")

        # 2. Graph Traversal
        aql_traverse = """
        FOR v, e, p IN 1..2 ANY @startId concept_relations
          LIMIT 20
          RETURN {
            key: v._key,
            name: v.name,
            relation: e.relation_type,
            hop: LENGTH(p.edges)
          }
        """
        start_id = f"concepts/{anchor['_key']}"
        facts = self._aql(aql_traverse, {"startId": start_id})
        
        if not facts:
            print("   ⚠️ No neighbors found. (Isolate Node)")
            return []

        # 3. Reference Ranking
        print(f"   🧠 Reasoning on {len(facts)} retrieved facts...")
        ranked_facts = []

        for fact in facts:
            score = 0.0
            if anchor_vec is not None and fact['key'] in self.key_to_idx:
                tgt_idx = self.key_to_idx[fact['key']]
                tgt_vec = self.embeddings[tgt_idx].reshape(1, -1)
                score = cosine_similarity(anchor_vec, tgt_vec)[0][0]
            
            ranked_facts.append({**fact, "score": float(score)})

        # Sort by Relevance
        ranked_facts.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_facts[:top_k]

if __name__ == "__main__":
    # Simple CLI test
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        engine = MedicalReasoningEngine()
        results = engine.search_and_reason(query)
        
        print(f"   🏆 Context for Agent:")
        for f in results:
             print(f"      [{f['score']:.4f}] ... --[{f['relation']}]--> {f['name']} (Hop {f['hop']})")
    else:
        print("Usage: python3 -m knowledge_core.medical_engine <query>")
