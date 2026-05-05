
import asyncio
import os
import pickle
import sys
import numpy as np
import httpx
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
            self.embeddings = np.load(vec_path, mmap_mode='r')
            print(f"   ✅ Brain Loaded: {self.embeddings.shape} matrix.")
        except Exception as e:
            print(f"❌ Error loading Embeddings: {e}")
            self.embeddings = None

        print("✅ Engine Online.\n")

    async def _aql(self, client: httpx.AsyncClient, query, bind_vars=None):
        """Execute AQL query against ArangoDB. Uses shared httpx client for connection reuse."""
        try:
            resp = await client.post(
                f"{ARANGO_URL}/cursor",
                json={"query": query, "bindVars": bind_vars or {}},
                auth=AUTH,
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get('result', [])
        except Exception as e:
            print(f"   ⚠️ AQL Error: {e}")
            return []

    async def fetch_node_by_id(self, node_id):
        aql = "RETURN DOCUMENT(CONCAT('concepts/', @id))"
        async with httpx.AsyncClient() as client:
            res = await self._aql(client, aql, {"id": node_id})
        return res[0] if res else None

    async def _resolve_candidates(self, client: httpx.AsyncClient, user_query):
        """
        Return a list of candidate concept nodes in priority order:
        synonym hit → exact match → fuzzy match.
        All AQL queries fire in parallel; results are merged in priority order.
        """
        query_lower = user_query.lower()

        aql_exact = "FOR d IN concepts FILTER d.name == @q LIMIT 1 RETURN d"
        aql_iexact = "FOR d IN concepts FILTER LOWER(d.name) == @q LIMIT 1 RETURN d"
        aql_fuzzy = """
        FOR d IN concepts
          FILTER LIKE(d.name, CONCAT(@q, "%"), true)
          SORT LENGTH(d.name) ASC
          LIMIT 1
          RETURN d
        """

        # Build coroutines — synonym AQL only fires if the synonym map has a hit
        syn_coro = None
        syn_id = None
        if query_lower in self.synonym_map:
            syn_id = self.synonym_map[query_lower]
            aql_syn = """
            FOR r IN synonym_relations
              FILTER r._from == CONCAT('synonyms/', @syn_id)
              LIMIT 1
              RETURN DOCUMENT(r._to)
            """
            syn_coro = self._aql(client, aql_syn, {"syn_id": syn_id})

        coroutines = [
            self._aql(client, aql_exact, {"q": user_query}),
            self._aql(client, aql_iexact, {"q": query_lower}),
            self._aql(client, aql_fuzzy, {"q": user_query}),
        ]
        if syn_coro:
            coroutines.insert(0, syn_coro)

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        candidates = []
        idx = 0
        if syn_coro:
            res = results[idx] if not isinstance(results[idx], Exception) else []
            if res and res[0]:
                candidates.append(("synonym", res[0]))
            idx += 1

        res_exact = results[idx] if not isinstance(results[idx], Exception) else []
        if res_exact:
            candidates.append(("exact", res_exact[0]))

        res_iexact = results[idx + 1] if not isinstance(results[idx + 1], Exception) else []
        if res_iexact:
            candidates.append(("iexact", res_iexact[0]))

        res_fuzzy = results[idx + 2] if not isinstance(results[idx + 2], Exception) else []
        if res_fuzzy:
            candidates.append(("fuzzy", res_fuzzy[0]))

        return candidates

    async def search_and_reason(self, user_query, top_k=10):
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

        async with httpx.AsyncClient() as client:
            # 1. Resolve candidates — all lookup AQLs fire in parallel
            candidates = await self._resolve_candidates(client, user_query)
            if not candidates:
                print("❌ Concept not found.")
                return []

            # 2. Traverse graph — try candidates in priority order until one has neighbors
            anchor = None
            facts = []
            for strategy, node in candidates:
                start_id = f"concepts/{node['_key']}"
                candidate_facts = await self._aql(client, aql_traverse, {"startId": start_id})
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

        # 3. Embedding-based cosine ranking (CPU-bound, no IO)
        anchor_vec = None
        if anchor['_key'] in self.key_to_idx and self.embeddings is not None:
            idx = self.key_to_idx[anchor['_key']]
            anchor_vec = self.embeddings[idx].reshape(1, -1)
        else:
            print("   ⚠️ Anchor not in embedding matrix. Ranking disabled.")

        print(f"   🧠 Reasoning on {len(facts)} retrieved facts...")
        ranked_facts = []
        for fact in facts:
            score = 0.0
            if anchor_vec is not None and fact['key'] in self.key_to_idx:
                tgt_idx = self.key_to_idx[fact['key']]
                tgt_vec = self.embeddings[tgt_idx].reshape(1, -1)
                score = cosine_similarity(anchor_vec, tgt_vec)[0][0]
            ranked_facts.append({**fact, "score": float(score)})

        ranked_facts.sort(key=lambda x: x['score'], reverse=True)
        return ranked_facts[:top_k]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        engine = MedicalReasoningEngine()
        results = asyncio.run(engine.search_and_reason(query))
        print(f"   🏆 Context for Agent:")
        for f in results:
            print(f"      [{f['score']:.4f}] ... --[{f['relation']}]--> {f['name']} (Hop {f['hop']})")
    else:
        print("Usage: python3 -m knowledge_core.medical_engine <query>")
