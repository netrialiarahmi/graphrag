"""LLM-based stance detection, causality judgement, and embedding via HuggingFace + OpenRouter."""

import os
import json
import functools
import requests
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Streamlit-agnostic caching: use st.cache_* when running inside Streamlit,
# otherwise fall back to functools.lru_cache / identity decorator.
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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1")

# HuggingFace embedding endpoint
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Govnetic/Indo-LegalBERT-V3")


@_cache_resource
def get_llm_client():
    """Create and cache an OpenAI-compatible client pointing to OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def test_connection() -> bool:
    """Test if OpenRouter is reachable."""
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Respond with OK"}],
            max_tokens=16,
        )
        return response.choices[0].message.content is not None
    except Exception:
        return False


def test_hf_connection() -> bool:
    """Test if HuggingFace embedding endpoint is reachable."""
    try:
        resp = requests.post(
            HF_ENDPOINT_URL,
            headers={"Authorization": f"Bearer {HF_AUTH_TOKEN}", "Content-Type": "application/json"},
            json={"inputs": "test"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_embedding(text: str, max_retries: int = 5) -> list[float]:
    """
    Get text embedding via HuggingFace Indo-LegalBERT endpoint.

    Retries up to *max_retries* times with exponential backoff on transient
    HTTP errors (429, 500, 502, 503, 504).
    """
    import time

    retryable_codes = {429, 500, 502, 503, 504}
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                HF_ENDPOINT_URL,
                headers={
                    "Authorization": f"Bearer {HF_AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text},
                timeout=60,
            )
            if resp.status_code in retryable_codes and attempt < max_retries:
                wait = 2 ** attempt          # 1, 2, 4, 8, 16 seconds
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            # HF endpoints may return {"embeddings": [...]}, [[...]], or [...] depending on config
            if isinstance(data, dict) and "embeddings" in data:
                return data["embeddings"]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    return data[0]
                return data
            raise ValueError(f"Unexpected embedding response format: {type(data)}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            raise
    if last_exc:
        raise last_exc


# ── LLM-powered query expansion ─────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """Use GPT to generate expanded search terms for an Indonesian legal query.

    Returns 3-5 alternative search phrases that capture the same legal concept
    using different terminology, specific UU/PP references, and synonyms.
    """
    client = get_llm_client()

    system_prompt = """Kamu adalah pakar hukum Indonesia. Tugasmu adalah menghasilkan variasi query pencarian untuk menemukan dokumen regulasi yang relevan di database vektor.

Untuk setiap pertanyaan hukum yang diberikan, hasilkan 3-5 variasi pencarian yang:
1. Menggunakan istilah hukum formal yang berbeda (sinonim hukum)
2. Menyebutkan UU/PP/Permen spesifik yang kemungkinan mengatur topik tersebut
3. Menggunakan frasa kunci dari pasal yang relevan
4. Mencakup istilah teknis yang mungkin muncul di dokumen regulasi

Format output: Satu variasi per baris, tanpa nomor atau bullet."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pertanyaan: {query}"},
            ],
            max_tokens=250,
            temperature=0.4,
        )
        raw = response.choices[0].message.content or ""
        # Parse lines, skip empty
        lines = [ln.strip().lstrip("0123456789.-) ") for ln in raw.strip().splitlines()]
        return [ln for ln in lines if len(ln) > 5][:5]
    except Exception:
        return []  # Graceful fallback — proceed with original query only


# ── LLM-powered document catalog search ──────────────────────────────────────

def smart_doc_lookup(query: str, all_docs: list[dict]) -> list[str]:
    """Use GPT to identify relevant documents from the full Neo4j catalog.

    The LLM examines document metadata (doc_id, judul, jenis) to determine
    which documents are likely relevant to the user's legal question,
    even if the embedding model fails to surface them.

    Parameters
    ----------
    query : str
        The user's legal question.
    all_docs : list[dict]
        Full list 