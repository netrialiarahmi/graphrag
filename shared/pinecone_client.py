"""Pinecone vector database connector for legal document embeddings."""

import os
import math
import random
import functools
from pinecone import Pinecone
from dotenv import load_dotenv

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


def _cache_resource(func):
    if _IN_STREAMLIT:
        return st.cache_resource(func)
    return functools.lru_cache(maxsize=1)(func)


def _cache_data(**kwargs):
    def decorator(func):
        if _IN_STREAMLIT:
            return st.cache_data(**kwargs)(func)
        return func
    return decorator


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "lexport-trial")


@_cache_resource
def get_pinecone_client():
    """Create and cache a Pinecone client instance."""
    return Pinecone(api_key=PINECONE_API_KEY)


@_cache_resource
def get_index():
    """Get the Pinecone index."""
    pc = get_pinecone_client()
    return pc.Index(INDEX_NAME)


def test_connection() -> bool:
    """Test if Pinecone connection is alive."""
    try:
        index = get_index()
        stats = index.describe_index_stats()
        return stats is not None
    except Exception:
        return False


def get_index_stats() -> dict:
    """Get index statistics (dimension, vector count, etc.)."""
    try:
        index = get_index()
        stats = index.describe_index_stats()
        return {
            "dimension": stats.get("dimension", "?"),
            "total_vectors": stats.get("total_vector_count", 0),
            "namespaces": dict(stats.get("namespaces", {})),
        }
    except Exception as e:
        return {"error": str(e)}


@_cache_data(ttl=3600)
def semantic_search(query_embedding: list[float], top_k: int = 10, scope_filter: str = None) -> list[dict]:
    """
    Search Pinecone with a query embedding vector.
    Returns list of {id, doc_id, article_id, content, scope, score}.
    """
    index = get_index()

    filter_dict = {}
    if scope_filter and scope_filter != "all":
        filter_dict["scope"] = scope_filter

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    hits = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        hits.append({
            "id": match["id"],
            "score": round(match["score"], 4),
            "doc_id": meta.get("doc_id", ""),
            "article_id": meta.get("article_id", ""),
            "content": meta.get("content", ""),
            "scope": meta.get("scope", ""),
        })
    return hits


@_cache_data(ttl=3600)
def search_by_text(query: str, top_k: int = 10, scope_filter: str = None) -> list[dict]:
    """
    Search Pinecone using text query (requires embedding first via OpenRouter/OpenAI).
    This is a convenience wrapper — actual embedding is done in the caller.
    Falls back to metadata-based search if no embedding function available.
    """
    # This function is called after embedding the query externally.
    # The actual implementation is in the main app which embeds first, then calls semantic_search.
    raise NotImplementedError(
        "Use semantic_search() with a pre-computed embedding vector, "
        "or call the embedding function from llm_stance module first."
    )


@_cache_data(ttl=3600)
def fetch_by_doc_id(doc_id: str, top_k: int = 100) -> list[dict]:
    """Fetch all vectors belonging to a specific doc_id using metadata filter.

    Uses a random unit vector instead of zero vector — cosine similarity
    with a zero vector is undefined and causes Pinecone to return 0 matches.
    """
    index = get_index()

    stats = get_index_stats()
    dim = stats.get("dimension", 1024)

    # Random unit vector (deterministic seed per doc_id for cacheability)
    rng = random.Random(doc_id)
    rand_vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in rand_vec))
    if norm > 0:
        rand_vec = [x / norm for x in rand_vec]

    results = index.query(
        vector=rand_vec,
        top_k=top_k,
        include_metadata=True,
        filter={"doc_id": doc_id},
    )

    hits = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        hits.append({
            "id": match["id"],
            "doc_id": meta.get("doc_id", ""),
            "article_id": meta.get("article_id", ""),
            "content": meta.get("content", ""),
            "scope": meta.get("scope", ""),
        })
    return hits


@_cache_data(ttl=3600)
def fetch_by_ids(ids: list[str]) -> list[dict]:
    """Fetch specific vectors by their IDs."""
    if not ids:
        return []

    index = get_index()
    results = index.fetch(ids=ids)

    hits = []
    for vid, vec_data in results.get("vectors", {}).items():
        meta = vec_data.get("metadata", {})
        hits.append({
            "id": vid,
            "doc_id": meta.get("doc_id", ""),
            "article_id": meta.get("article_id", ""),
            "content": meta.get("content", ""),
            "scope": meta.get("scope", ""),
        })
    return hits
