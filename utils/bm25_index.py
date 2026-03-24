"""
In-memory BM25 index over Pinecone chunk corpus.

First run: downloads all chunk metadata from Pinecone → caches to disk.
Subsequent runs: loads from disk cache (~instant for 4.6k chunks).
Provides bm25_search(query, top_k) and hybrid_search() with RRF merging.
"""
import json
import os
import re
import time
import functools
from pathlib import Path

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Streamlit-agnostic caching
# ---------------------------------------------------------------------------
_IN_STREAMLIT = False
if not os.environ.get("GRAPHRAG_STANDALONE"):
    try:
        import streamlit as st
        _IN_STREAMLIT = True
    except ImportError:
        pass

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chatbot", "bm25_corpus.json"
)

# ── Tokenizer ────────────────────────────────────────────────────────────────
_STOP_WORDS = frozenset(
    "yang dan di dari untuk dengan pada dalam ini itu atau sebagai oleh"
    " sebagaimana dimaksud ayat pasal huruf angka"
    " apa apakah bagaimana berapa kapan mengapa kenapa siapa mana"
    " adalah ialah merupakan tersebut bahwa ketentuan peraturan"
    " undang pemerintah republik indonesia".split()
)

def _tokenize(text: str) -> list[str]:
    """Simple Indonesian tokenizer: lowercase, split on non-alnum, drop stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ── Corpus loading ───────────────────────────────────────────────────────────

def _load_corpus_from_pinecone() -> list[dict]:
    """Download all chunk metadata from Pinecone index."""
    from utils.pinecone_client import get_index

    idx = get_index()
    print("[BM25] Enumerating Pinecone IDs …")
    all_ids = []
    for ids in idx.list():
        all_ids.extend(ids)
    print(f"[BM25] Found {len(all_ids)} vectors. Fetching metadata …")

    corpus = []
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        result = idx.fetch(ids=batch)
        for vid, vdata in result.get("vectors", {}).items():
            meta = vdata.get("metadata", {})
            corpus.append({
                "id": vid,
                "doc_id": meta.get("doc_id", ""),
                "article_id": meta.get("article_id", ""),
                "content": meta.get("content", ""),
                "scope": meta.get("scope", ""),
            })
        if (i // batch_size) % 10 == 0:
            print(f"[BM25]   … fetched {min(i + batch_size, len(all_ids))}/{len(all_ids)}")

    # Save to disk cache
    cache_dir = os.path.dirname(os.path.abspath(_CACHE_PATH))
    os.makedirs(cache_dir, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"[BM25] Cached {len(corpus)} chunks to {_CACHE_PATH}")
    return corpus


def _load_corpus() -> list[dict]:
    """Load corpus from disk cache, or from Pinecone if cache missing."""
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        return corpus
    return _load_corpus_from_pinecone()


# ── BM25 Index singleton ────────────────────────────────────────────────────

class _BM25Index:
    """Lazy-initialized BM25 index over the Pinecone corpus."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []
        self._tokenized: list[list[str]] = []

    def _ensure_loaded(self):
        if self._bm25 is not None:
            return
        t0 = time.time()
        self._corpus = _load_corpus()
        self._tokenized = [_tokenize(doc.get("content", "")) for doc in self._corpus]
        self._bm25 = BM25Okapi(self._tokenized)
        elapsed = time.time() - t0
        print(f"[BM25] Index built: {len(self._corpus)} docs in {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """BM25 keyword search. Returns list of {id, doc_id, ..., bm25_score}."""
        self._ensure_loaded()
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices by score
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                break
            doc = self._corpus[idx]
            results.append({
                "id": doc["id"],
                "doc_id": doc["doc_id"],
                "article_id": doc.get("article_id", ""),
                "content": doc["content"],
                "scope": doc.get("scope", ""),
                "bm25_score": round(float(score), 4),
            })
        return results

    def refresh_cache(self):
        """Force re-download from Pinecone and rebuild index."""
        self._bm25 = None
        self._corpus = []
        self._tokenized = []
        if os.path.exists(_CACHE_PATH):
            os.remove(_CACHE_PATH)
        self._ensure_loaded()


_INDEX = _BM25Index()


def bm25_search(query: str, top_k: int = 50) -> list[dict]:
    """Run a BM25 keyword search over all Pinecone chunks."""
    return _INDEX.search(query, top_k)


def refresh_cache():
    """Force re-download corpus from Pinecone."""
    _INDEX.refresh_cache()


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int = 20,
    k: int = 60,
    alpha: float = 0.5,
) -> list[dict]:
    """Merge dense and BM25 results using weighted Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_hits : Pinecone results (must have 'id' and 'score')
    bm25_hits  : BM25 results (must have 'id' and 'bm25_score')
    top_k      : number of merged results to return
    k          : RRF constant (default 60)
    alpha      : weight for dense vs BM25 (0.5 = equal, >0.5 = more dense)
    """
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    # Dense contribution
    for rank, hit in enumerate(dense_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0) + alpha / (k + rank + 1)
        doc_data[hid] = hit

    # BM25 contribution
    for rank, hit in enumerate(bm25_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0) + (1 - alpha) / (k + rank + 1)
        if hid not in doc_data:
            doc_data[hid] = hit

    # Sort by fused score
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    results = []
    for hid in sorted_ids:
        entry = dict(doc_data[hid])
        entry["rrf_score"] = round(rrf_scores[hid], 6)
        # Keep original scores for debugging
        entry.setdefault("score", None)
        entry.setdefault("bm25_score", None)
        results.append(entry)
    return results


def hybrid_search(query: str, query_embedding: list[float], top_k: int = 20, alpha: float = 0.5) -> list[dict]:
    """Run hybrid dense + BM25 search with RRF merge.

    Parameters
    ----------
    query           : raw text query (for BM25)
    query_embedding : precomputed embedding vector (for Pinecone dense)
    top_k           : final number of results
    alpha           : dense weight (0.5 = balanced, 0.7 = mostly dense)
    """
    from utils import pinecone_client

    # Retrieve more candidates than needed for better fusion
    dense_k = max(top_k * 3, 50)
    bm25_k = max(top_k * 3, 50)

    dense_hits = pinecone_client.semantic_search(query_embedding=query_embedding, top_k=dense_k)
    bm25_hits = bm25_search(query, top_k=bm25_k)

    return reciprocal_rank_fusion(dense_hits, bm25_hits, top_k=top_k, alpha=alpha)
