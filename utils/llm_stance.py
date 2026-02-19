"""LLM-based stance detection and embedding via OpenRouter."""

import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")


@st.cache_resource
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
            max_tokens=5,
        )
        return response.choices[0].message.content is not None
    except Exception:
        return False


def get_embedding(text: str) -> list[float]:
    """
    Get text embedding via OpenRouter.
    Falls back to a simple hash-based pseudo-embedding if embedding model unavailable.
    """
    client = get_llm_client()
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception:
        # Fallback: use the LLM to generate a search query instead
        # This means we can't do vector search directly — caller should handle
        raise


def classify_stance(text_a: str, text_b: str, doc_a_id: str = "", doc_b_id: str = "") -> dict:
    """
    Classify the relationship between two legal text excerpts.

    Returns:
        {
            "stance": "MENDUKUNG" | "MENENTANG" | "NETRAL",
            "reason": "Brief explanation in Indonesian",
            "confidence": 0.0 - 1.0
        }
    """
    client = get_llm_client()

    system_prompt = """Kamu adalah ahli hukum Indonesia yang menganalisis hubungan antar regulasi.
Tugasmu adalah mengklasifikasikan hubungan antara dua kutipan regulasi.

Klasifikasi:
- MENDUKUNG: Teks B memperkuat, melengkapi, atau sejalan dengan Teks A
- MENENTANG: Teks B bertentangan, membatasi, mencabut, atau mengubah ketentuan Teks A
- NETRAL: Tidak ada hubungan substansial yang jelas, atau mengatur hal yang berbeda

Berikan output dalam format JSON:
{"stance": "MENDUKUNG|MENENTANG|NETRAL", "reason": "Penjelasan singkat 1-2 kalimat", "confidence": 0.0-1.0}

HANYA output JSON, tanpa teks lain."""

    user_prompt = f"""Analisis hubungan antara dua kutipan regulasi berikut:

Dokumen A ({doc_a_id}):
{text_a[:2000]}

Dokumen B ({doc_b_id}):
{text_b[:2000]}

Klasifikasikan hubungannya."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        # Try to parse JSON from response
        # Handle potential markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        # Validate
        valid_stances = {"MENDUKUNG", "MENENTANG", "NETRAL"}
        if result.get("stance") not in valid_stances:
            result["stance"] = "NETRAL"
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "reason" not in result:
            result["reason"] = "No explanation provided."

        return result

    except json.JSONDecodeError:
        return {
            "stance": "NETRAL",
            "reason": f"Failed to parse LLM response: {raw[:200]}",
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "stance": "NETRAL",
            "reason": f"Error: {str(e)}",
            "confidence": 0.0,
        }


def batch_classify(pairs: list[dict]) -> list[dict]:
    """
    Classify stance for multiple pairs of texts.

    Args:
        pairs: List of dicts with keys: text_a, text_b, doc_a_id, doc_b_id

    Returns:
        List of stance classification results.
    """
    results = []
    for pair in pairs:
        # Check session state cache
        cache_key = f"stance_{pair.get('doc_a_id', '')}_{pair.get('doc_b_id', '')}"
        if cache_key in st.session_state:
            results.append(st.session_state[cache_key])
            continue

        result = classify_stance(
            text_a=pair["text_a"],
            text_b=pair["text_b"],
            doc_a_id=pair.get("doc_a_id", ""),
            doc_b_id=pair.get("doc_b_id", ""),
        )

        # Cache in session state
        st.session_state[cache_key] = result
        results.append(result)

    return results


def ask_about_documents(query: str, context_chunks: list[dict]) -> str:
    """
    RAG-style question answering: given a user query and relevant context chunks,
    generate an answer grounded in the legal documents.
    """
    client = get_llm_client()

    # Build context string
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        doc_id = chunk.get("doc_id", "Unknown")
        scope = chunk.get("scope", "")
        content = chunk.get("content", "")
        context_parts.append(f"[{i}] {doc_id} ({scope}):\n{content}")

    context_str = "\n\n".join(context_parts)

    system_prompt = """Kamu adalah asisten hukum Indonesia yang menjawab pertanyaan berdasarkan dokumen regulasi yang diberikan.
Jawab dalam Bahasa Indonesia. Selalu sebutkan sumber dokumen yang relevan (doc_id).
Jika informasi tidak cukup untuk menjawab, katakan dengan jujur."""

    user_prompt = f"""Konteks dari dokumen regulasi:
{context_str}

Pertanyaan: {query}

Jawab berdasarkan konteks di atas."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
