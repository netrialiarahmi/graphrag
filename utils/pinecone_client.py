"""Pinecone vector database connector for legal document embeddings."""

import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "lexport-trial")


@st.cache_resource
def get_pinecone_client():
    """Create and cache a Pinecone client instance."""
    return Pinecone(api_key=PINECONE_API_KEY)


@st.cache_resource
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def fetch_by_doc_id(doc_id: str, top_k: int = 100) -> list[dict]:
    """Fetch all vectors belonging to a specific doc_id using metadata filter."""
    index = get_index()

    # Use a dummy zero vector to fetch by filter
    # We need to know the dimension first
    stats = get_index_stats()
    dim = stats.get("dimension", 1024)

    results = index.query(
        vector=[0.0] * dim,
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


@st.cache_data(ttl=3600)
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
