
import os
import pickle
import sys
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from config import settings

# Constants — read from config so .env controls the host
ARANGO_URL = f"{settings.ARANGODB_HOST.rstrip('/')}/_db/{settings.ARANGODB_DB_NAME}/_api"
AUTH = (settings.ARANGODB_USERNAME, settings.ARANGODB_PASSWORD)
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
        """Helper to execute AQL. 10s timeout prevents hanging when ArangoDB is unreachable."""
        try:
            resp = requests.post(
                f"{ARANGO_URL}/cursor",
                json={"query": query, "bindVars": bind_vars or {}},
                auth=AUTH,
                timeout=10,
            )
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

    def _resolve_candidates(self, user_query):
        """
        Return a list of candidate concept nodes in priority order:
        synonym hit → exact match → fuzzy match.
        Callers iterate until they find one with graph neighbors.
        """
        candidates = []
        query_lower = user_query.lower()

        # Candidate 1: synonym traversal
        if query_lower in self.synonym_map:
            syn_id = self.synonym_map[query_lower]
            aql_syn = """
            FOR r IN synonym_relations
              FILTER r._from == CONCAT('synonyms/', @syn_id)
              LIMIT 1
              RETURN DOCUMENT(r._to)
            """
            res = self._aql(aql_syn, {"syn_id": syn_id})
            if res and res[0]:
                candidates.append(("synonym", res[0]))

        # Candidate 2: exact name match
        aql_exact = "FOR d IN concepts FILTER d.name == @q LIMIT 1 RETURN d"
        res = self._aql(aql_exact, {"q": user_query})
        if res:
            candidates.append(("exact", res[0]))

        # Candidate 3: case-insensitive exact match
        aql_iexact = "FOR d IN concepts FILTER LOWER(d.name) == @q LIMIT 1 RETURN d"
        res = self._aql(aql_iexact, {"q": query_lower})
        if res:
            candidates.append(("iexact", res[0]))

        # Candidate 4: fuzzy / starts-with
        aql_fuzzy = """
        FOR d IN concepts
          FILTER LIKE(d.name, CONCAT(@q, "%"), true)
          SORT LENGTH(d.name) ASC
          LIMIT 1
          RETURN d
        """
        res = self._aql(aql_fuzzy, {"q": user_query})
        if res:
            candidates.append(("fuzzy", res[0]))

        return candidates

    def search_and_reason(self, user_query, top_k=10):
        print(f"\n🔎 Query: '{user_query}'")

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

        # 1. Entity Linking — try each candidate until one has graph neighbors
        anchor = None
        facts = []
        candidates = self._resolve_candidates(user_query)
        if not candidates:
            print("❌ Concept not found.")
            return []

        for strategy, node in candidates:
            start_id = f"concepts/{node['_key']}"
            candidate_facts = self._aql(aql_traverse, {"startId": start_id})
            if candidate_facts:
                anchor = node
                facts = candidate_facts
                print(f"   ✅ {strategy}: '{node['name']}' (ID: {node['_key']})")
                break
            else:
                print(f"   ⚠️ {strategy}: '{node['name']}' is an isolate — trying next candidate")

        if not anchor:
            print("   ⚠️ All candidates are isolate nodes — no graph facts available.")
            return []

        # 2b. Get embedding vector for cosine ranking
        anchor_vec = None
        if anchor['_key'] in self.key_to_idx and self.embeddings is not None:
            idx = self.key_to_idx[anchor['_key']]
            anchor_vec = self.embeddings[idx].reshape(1, -1)
        else:
            print("   ⚠️ Anchor not in embedding matrix. Ranking disabled.")

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
