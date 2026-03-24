#!/usr/bin/env python3
"""Quick test: Dense vs BM25 vs Hybrid for multiple queries."""
import os, sys, time
os.environ.setdefault("GRAPHRAG_STANDALONE", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from shared import llm_stance, pinecone_client
from shared.bm25_index import bm25_search, hybrid_search

QUERIES = [
    "apa itu bangunan?",
    "sanksi bangunan tanpa IMB",
    "pengecualian UMKM dalam perizinan",
]

for q in QUERIES:
    print(f"\n{'='*70}")
    print(f"QUERY: {q}")
    print("="*70)
    emb = llm_stance.get_embedding(q)

    dense = pinecone_client.semantic_search(query_embedding=emb, top_k=10)
    hyb = hybrid_search(q, emb, top_k=10, alpha=0.4)

    print("\n  DENSE top-5:")
    for i, h in enumerate(dense[:5]):
        print(f'    {i+1}. [{h["score"]}] {h["doc_id"]}: {h["content"][:80]}')

    print("\n  HYBRID top-5:")
    for i, h in enumerate(hyb[:5]):
        ds = h.get("score") or "-"
        bs = h.get("bm25_score") or "-"
        print(f'    {i+1}. [rrf={h["rrf_score"]} d={ds} b={bs}] {h["doc_id"]}')
        print(f'       {h["content"][:90]}')
    print()
