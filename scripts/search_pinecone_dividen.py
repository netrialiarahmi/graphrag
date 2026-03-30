"""Search Pinecone for dividen interim content."""
import os, sys
os.environ['GRAPHRAG_STANDALONE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()
from utils import pinecone_client, llm_stance

# Search with embedding
emb = llm_stance.get_embedding("dividen interim perseroan terbatas")
print("Got embedding, searching Pinecone...")

# Global search
hits = pinecone_client.semantic_search(query_embedding=emb, top_k=10)
print(f"\n=== Top 10 global hits for 'dividen interim perseroan terbatas' ===")
for h in hits:
    print(f"  [{h['score']:.4f}] {h['doc_id']} / {h.get('article_id','?')} / {h.get('scope','')}:")
    print(f"    {h['content'][:200]}...")
    print()

# Search filtered to UU 40/2007
idx = pinecone_client.get_index()
r1 = idx.query(vector=emb, top_k=10, include_metadata=True, filter={"doc_id": "UU-NASIONAL-40-2007"})
print(f"\n=== UU-NASIONAL-40-2007 top hits ===")
for m in r1.get("matches", []):
    meta = m.get("metadata", {})
    print(f"  [{m['score']:.4f}] {meta.get('article_id','?')} / {meta.get('scope','')}:")
    print(f"    {meta.get('content','')[:200]}...")
    print()

# Search filtered to PERPPU 2/2022
r2 = idx.query(vector=emb, top_k=10, include_metadata=True, filter={"doc_id": "PERPPU-NASIONAL-2-2022"})
print(f"\n=== PERPPU-NASIONAL-2-2022 top hits ===")
for m in r2.get("matches", []):
    meta = m.get("metadata", {})
    print(f"  [{m['score']:.4f}] {meta.get('article_id','?')} / {meta.get('scope','')}:")
    print(f"    {meta.get('content','')[:200]}...")
    print()
