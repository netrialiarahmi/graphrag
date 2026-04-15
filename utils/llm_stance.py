"""LLM-based stance detection, causality judgement, and embedding via HuggingFace + OpenRouter."""

import os
import json
import functools
import requests
from openai import OpenAI
from dotenv import load_dotenv
from utils.langsmith_config import get_traceable

traceable = get_traceable()

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
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4")

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

def expand_query(query: str, *, _trace_id: str = "", _route: str = "") -> list[str]:
    """Use GPT to generate expanded search terms for an Indonesian legal query.

    Returns 3-5 alternative search phrases that capture the same legal concept
    using different terminology, specific UU/PP references, and synonyms.
    """
    from shared.debug_logger import log_event as _dlog
    client = get_llm_client()

    system_prompt = """Kamu adalah pakar hukum Indonesia. Tugasmu adalah menghasilkan variasi query pencarian untuk menemukan dokumen regulasi yang relevan di database vektor.

Untuk setiap pertanyaan hukum yang diberikan, hasilkan 3-5 variasi pencarian yang:
1. Menggunakan istilah hukum formal yang berbeda (sinonim hukum). Untuk pertanyaan definisi/pengertian, WAJIB sertakan frasa "yang dimaksud dengan [istilah]" karena ini adalah pola umum dalam pasal ketentuan umum regulasi Indonesia.
2. Menyebutkan UU/PP/Permen spesifik yang kemungkinan mengatur topik tersebut
3. Menggunakan frasa kunci dari pasal yang relevan (misalnya "ketentuan umum", "definisi", "pengertian")
4. Mencakup istilah teknis yang mungkin muncul di dokumen regulasi

Contoh: jika pertanyaan "apa itu bangunan gedung?", hasilkan:
- yang dimaksud dengan bangunan gedung
- definisi bangunan gedung UU 28/2002
- pengertian bangunan gedung PP 16/2021
- ketentuan umum bangunan gedung

Format output: Satu variasi per baris, tanpa nomor atau bullet."""

    user_prompt = f"Pertanyaan: {query}"
    if _trace_id:
        _dlog(trace_id=_trace_id, route=_route, stage="expand_query", event="prompt_input",
              message="expand_query prompt input",
              payload={"system_prompt": system_prompt, "user_prompt": user_prompt})

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=250,
            temperature=0.4,
        )
        raw = response.choices[0].message.content or ""
        # Parse lines, skip empty
        lines = [ln.strip().lstrip("0123456789.-) ") for ln in raw.strip().splitlines()]
        result = [ln for ln in lines if len(ln) > 5][:5]
        if _trace_id:
            _dlog(trace_id=_trace_id, route=_route, stage="expand_query", event="prompt_output",
                  message="expand_query prompt output",
                  payload={"raw_response": raw, "expanded_queries": result})
        return result
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
        Full list of documents from Neo4j with {doc_id, judul, jenis, tahun, ...}.

    Returns
    -------
    list[str]
        Ordered list of doc_ids the LLM deems most relevant (max 10).
    """
    if not all_docs:
        return []

    client = get_llm_client()

    # Build compact catalog
    catalog_lines = []
    for doc in all_docs:
        did = doc.get("doc_id", "")
        judul = doc.get("judul", "-")
        jenis = doc.get("jenis", "-")
        tahun = doc.get("tahun", "-")
        catalog_lines.append(f"{did} | {jenis} | {tahun} | {judul}")
    catalog_str = "\n".join(catalog_lines)

    system_prompt = """Kamu adalah pakar hukum Indonesia. Dari katalog dokumen regulasi berikut, pilih dokumen yang PALING RELEVAN untuk menjawab pertanyaan hukum yang diberikan.

Pertimbangkan:
1. Kesesuaian topik/subjek hukum (perseroan, ketenagakerjaan, bangunan, dll.)
2. Jenis regulasi yang tepat (UU untuk undang-undang pokok, PP untuk pelaksanaan, dll.)
3. Judul dokumen dan hubungannya dengan pertanyaan
4. Dokumen yang saling terkait (amandemen, pelaksanaan, dll.)

Format output: Satu doc_id per baris, urutkan dari yang PALING relevan. Maksimal 10 dokumen.
Output HANYA doc_id, tanpa penjelasan."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pertanyaan: {query}\n\nKatalog Dokumen:\n{catalog_str}"},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        # Parse doc_ids from response lines
        valid_ids = {doc.get("doc_id", "") for doc in all_docs if doc.get("doc_id")}
        results = []
        for line in raw.strip().splitlines():
            line = line.strip().lstrip("0123456789.-) ")
            if line in valid_ids and line not in results:
                results.append(line)
        return results[:10]
    except Exception:
        return []


# ── LLM-powered document re-ranking ─────────────────────────────────────────

def rerank_documents(query: str, doc_summaries: dict[str, str], *, _trace_id: str = "", _route: str = "") -> list[tuple[str, float]]:
    """Re-rank candidate documents by relevance using GPT.

    Parameters
    ----------
    query : str
        The user's legal question.
    doc_summaries : dict[str, str]
        Mapping of doc_id → representative text snippet (first chunk content).

    Returns
    -------
    list[tuple[str, float]]
        Sorted list of (doc_id, score) with score 0-10, descending.
    """
    from shared.debug_logger import log_event as _dlog
    if not doc_summaries:
        return []

    client = get_llm_client()

    # Build the summary block
    summary_parts = []
    for did, text in doc_summaries.items():
        snippet = text[:400] if text else "(kosong)"
        summary_parts.append(f"DOC_ID: {did}\nKonten: {snippet}")
    summaries_str = "\n\n".join(summary_parts)

    system_prompt = """Kamu adalah pakar hukum Indonesia. Berikan skor relevansi 0-10 untuk setiap dokumen terhadap pertanyaan yang diberikan.

Format output HARUS berupa JSON array, contoh:
[{"doc_id": "UU-NASIONAL-40-2007", "score": 9}, {"doc_id": "PP-NASIONAL-16-2021", "score": 1}]

Skor:
- 8-10: Sangat relevan, kemungkinan besar memuat jawaban
- 5-7: Cukup relevan, mungkin memuat konteks pendukung
- 2-4: Sedikit relevan, hubungan tidak langsung
- 0-1: Tidak relevan sama sekali

Output HANYA JSON array, tanpa teks lain."""

    user_prompt = f"Pertanyaan: {query}\n\nDokumen:\n{summaries_str}"
    if _trace_id:
        _dlog(trace_id=_trace_id, route=_route, stage="rerank_documents", event="prompt_input",
              message="rerank_documents prompt input",
              payload={"system_prompt": system_prompt, "user_prompt": user_prompt})

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
        raw = response.choices[0].message.content or "[]"
        # Extract JSON from response (handle markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        scored = json.loads(raw)
        results = []
        for item in scored:
            did = item.get("doc_id", "")
            score = float(item.get("score", 0))
            if did:
                results.append((did, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except Exception:
        # Fallback: return all docs with neutral score
        return [(did, 5.0) for did in doc_summaries]


# ── LLM-powered sufficiency check (early-exit gate) ─────────────────────────

def judge_sufficiency(query: str, doc_ids: list[str], doc_summaries: dict[str, str]) -> bool:
    """Ask the LLM whether the current candidate documents are sufficient to answer the query.

    Used as an early-exit gate: if the LLM judges the documents are enough,
    the pipeline can skip expensive downstream phases (VDB expansion, graph
    traversal, re-ranking).

    Parameters
    ----------
    query : str
        The user's legal question.
    doc_ids : list[str]
        Candidate document IDs found so far.
    doc_summaries : dict[str, str]
        Mapping of doc_id → representative text snippet.

    Returns
    -------
    bool
        True if the LLM deems the documents sufficient; False otherwise.
        Defaults to False on any error (conservative: proceed with full pipeline).
    """
    if not doc_ids or not doc_summaries:
        return False

    client = get_llm_client()

    summary_parts = []
    for did in doc_ids:
        snippet = doc_summaries.get(did, "")[:300] or "(kosong)"
        summary_parts.append(f"- {did}: {snippet}")
    doc_list_str = "\n".join(summary_parts)

    system_prompt = """Kamu adalah pakar hukum Indonesia. Tugasmu HANYA menilai apakah DAFTAR DOKUMEN yang diberikan sudah CUKUP untuk menjawab pertanyaan hukum pengguna.

Kriteria CUKUP:
1. Dokumen mencakup regulasi utama yang mengatur topik pertanyaan (UU pokok, PP pelaksana, atau Permen teknis).
2. Substansi dokumen relevan langsung dengan pertanyaan — bukan hanya topik umum yang sama.
3. Jika pertanyaan merujuk regulasi spesifik (misal "PP 34/2021"), regulasi tersebut HARUS ada dalam daftar.

Kriteria BELUM CUKUP:
1. Dokumen yang ada tidak langsung mengatur topik pertanyaan.
2. Masih diperlukan regulasi pelaksana atau regulasi terkait yang belum ada.
3. Pertanyaan bersifat komparatif atau lintas-regulasi tetapi hanya ada satu sisi.

Jawab HANYA dengan satu kata: CUKUP atau BELUM"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Pertanyaan: {query}\n\n"
                    f"Dokumen yang ditemukan ({len(doc_ids)}):\n{doc_list_str}"
                )},
            ],
            max_tokens=10,
            temperature=0.1,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        return "CUKUP" in answer and "BELUM" not in answer
    except Exception:
        return False  # Conservative: proceed with full pipeline


def classify_stance(text_a: str, text_b: str, doc_a_id: str = "", doc_b_id: str = "") -> dict:
    """
    Classify the relationship between two legal text excerpts.

    Uses the Indonesian legal framework for determining:
    - MENENTANG (Contradiction): authority conflicts, contradictory obligations,
      hierarchy violations, inconsistent terminology.
    - MENDUKUNG (Entailment): delegation/attribution relationship, complementary
      operationalisation, consistent normative alignment.
    - NETRAL (Neutral): different jurisdictional domains, mutually exclusive
      subject matter, no substantive overlap.

    Returns:
        {
            "stance": "MENDUKUNG" | "MENENTANG" | "NETRAL",
            "reason": "Brief explanation in Indonesian",
            "confidence": 0.0 - 1.0
        }
    """
    client = get_llm_client()

    system_prompt = """Kamu adalah ahli hukum tata negara Indonesia senior yang mengklasifikasikan hubungan antar kutipan regulasi.

═══ KERANGKA KLASIFIKASI ═══

1. MENENTANG (Kontradiksi / Disharmoni)
   Teks B bertentangan dengan Teks A apabila memenuhi SATU ATAU LEBIH:
   a) Benturan Kewenangan: Memberikan otoritas kepada organ berbeda atas objek urusan identik.
   b) Pertentangan Hak & Kewajiban: Teks A mewajibkan X, Teks B melarang/membatasi X.
   c) Inkonsistensi Terminologi: Definisi/parameter berbeda untuk istilah yang sama.
   d) Pelanggaran Hierarki (Lex Superior): Ketentuan tingkat rendah berlawanan dengan tingkat tinggi.
   e) Pencabutan Kronologis (Lex Posterior): Regulasi baru mencabut/mengubah regulasi lama sederajat.
   f) Pengecualian Khusus (Lex Specialis): Regulasi khusus mengesampingkan regulasi umum sederajat.

2. MENDUKUNG (Entailment / Sesuai / Komplementer)
   Teks B mendukung Teks A apabila:
   a) Hubungan Delegasi: Teks B mengoperasionalisasikan norma abstrak Teks A via petunjuk teknis.
   b) Konsistensi Substansial: Norma selaras, saling melengkapi, memperkuat.
   c) Keselarasan Asas: Alignment teleologis konsisten.

3. NETRAL (Tidak Berhubungan)
   a) Yurisdiksi terpisah: Urusan absolut pusat vs. otonomi daerah tanpa irisan.
   b) Substansi eksklusif: Objek pengaturan, domain kelembagaan, rezim hukum terisolasi.
   c) Perubahan satu regulasi tidak mempengaruhi validitas regulasi lainnya.

Berikan output dalam format JSON:
{"stance": "MENDUKUNG|MENENTANG|NETRAL", "reason": "Penjelasan 1-2 kalimat menyebutkan indikator spesifik", "confidence": 0.0-1.0}

HANYA output JSON, tanpa teks lain."""

    user_prompt = f"""Analisis hubungan antara dua kutipan regulasi berikut:

Dokumen A ({doc_a_id}):
{text_a[:2000]}

Dokumen B ({doc_b_id}):
{text_b[:2000]}

Klasifikasikan sebagai MENDUKUNG, MENENTANG, atau NETRAL berdasarkan kerangka analitis di atas."""

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
    _cache = {}
    if _IN_STREAMLIT:
        _cache = st.session_state

    results = []
    for pair in pairs:
        # Check session state cache
        cache_key = f"stance_{pair.get('doc_a_id', '')}_{pair.get('doc_b_id', '')}"
        if cache_key in _cache:
            results.append(_cache[cache_key])
            continue

        result = classify_stance(
            text_a=pair["text_a"],
            text_b=pair["text_b"],
            doc_a_id=pair.get("doc_a_id", ""),
            doc_b_id=pair.get("doc_b_id", ""),
        )

        # Cache in session state
        _cache[cache_key] = result
        results.append(result)

    return results


def ask_about_documents(query: str, context_chunks: list[dict],
                       relationship_context: str = "",
                       chat_history: list[dict] | None = None,
                       summary: str = "",
                       user_context: str = "",
                       _trace_id: str = "",
                       _route: str = "",
                       _verbose_debug: bool = False) -> str:
    """
    RAG-style question answering: given a user query and relevant context chunks,
    generate an answer grounded in the legal documents.
    Supports conversation memory via chat_history and summary.
    """
    client = get_llm_client()

    # Build context string — skip empty chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        doc_id = chunk.get("doc_id", "Unknown")
        scope = chunk.get("scope", "")
        content = chunk.get("content", "")
        if not content.strip():
            continue
        context_parts.append(f"[{i}] {doc_id} ({scope}):\n{content}")

    context_str = "\n\n".join(context_parts)

    system_prompt = """Kamu adalah pakar hukum tata negara Indonesia senior dengan keahlian mendalam di segala bidang hukum — perdata, pidana, tata negara, korporasi, perizinan, ketenagakerjaan, dan regulasi sektoral.

═══ KERANGKA ANALISIS RELASI ANTAR-REGULASI ═══

Dalam sistem hukum civil law Indonesia, relasi antar-regulasi terbagi menjadi tiga kondisi fundamental:

A. TUMPANG TINDIH / DISHARMONI (Contradict):
   Terdeteksi apabila ada: (1) benturan kewenangan antar-organ negara untuk objek urusan yang sama, (2) pertentangan hak & kewajiban di mana subjek hukum diwajibkan oleh Regulasi A tapi dilarang oleh Regulasi B, (3) inkonsistensi terminologi untuk istilah/objek yang sama, (4) pelanggaran hierarki (Lex Superior Derogat Legi Inferiori), (5) pencabutan kronologis (Lex Posterior Derogat Legi Priori), atau (6) pengecualian kekhususan (Lex Specialis Derogat Legi Generali).

B. SALING MENGUATKAN / KOMPLEMENTER (Harmonis):
   Terdeteksi apabila: regulasi pelaksana (PP/Perpres/Permen) mengoperasionalisasikan norma abstrak dari UU induk melalui petunjuk teknis yang spesifik, atau kedua regulasi sederajat saling melengkapi tanpa pertentangan.

C. NETRAL / TIDAK BERHUBUNGAN (Mutually Exclusive):
   Terdeteksi apabila: kedua regulasi mengatur rezim hukum yang sepenuhnya terpisah, terisolasi, dan saling tidak mendisrupsi.

═══ KRITIS: DETEKSI POTENSI KONFLIK & AMBIGUITAS ═══

PENTING: Di luar tiga kondisi di atas, dalam praktik ketatanegaraan sering muncul ZONA ABU-ABU — situasi yang BUKAN kontradiksi langsung tapi mengandung POTENSI KONFLIK atau AMBIGUITAS. Kamu WAJIB mengenali dan menjelaskan ini:

1. POTENSI KONFLIK (Tension):
   - Ketika regulasi sektoral membatasi atau mengkondisikan suatu PRINSIP UMUM (mis. kebebasan berusaha, otonomi daerah), batasannya sendiri BISA dianggap sebagai potensi konflik meskipun secara formal dibenarkan oleh kebijakan pemerintah.
   - Contoh: Permen yang mewajibkan PMA menunjuk PMDN sebagai distributor → secara formal sah, tapi BERPOTENSI bertentangan dengan prinsip kebebasan berusaha dan non-diskriminasi.
   - Jika pertanyaan menanyakan "konflik", jawab "Ada potensi konflik" lalu jelaskan kedua sisi: pembatasan + justifikasinya.

2. AMBIGUITAS TEMPORAL (Transitional Gap):
   - Ketika regulasi baru (UU, Perppu) MENGUBAH atau MENCABUT regulasi lama, selalu pertimbangkan: bagaimana status tindakan/keputusan yang sudah dibuat SEBELUM regulasi baru berlaku?
   - Jika regulasi baru TIDAK memuat ketentuan peralihan (transitional provisions) yang eksplisit untuk keputusan yang sudah ada, itu ADALAH ambiguitas.
   - Contoh: Perppu 2/2022 mengubah UU 40/2007, tapi tidak mengatur nasib keputusan RUPS yang diambil sebelum Perppu → ada ambiguitas.

3. LEX SPECIALIS TENSION:
   - Ketika regulasi khusus membatasi lingkup regulasi umum, akui bahwa meskipun Lex Specialis sah secara hukum, pembatasannya bisa menimbulkan KETEGANGAN (tension) dengan prinsip dasar yang diusung regulasi umum.

═══ CARA MENJAWAB ═══

1. MULAI LANGSUNG dengan KESIMPULAN TEGAS dalam 1-3 kalimat TANPA heading apapun. Jika pertanyaan memiliki nuansa, gunakan jawaban bernuansa: "Ya, ada potensi konflik...", "Ya, terdapat ambiguitas...", "Tidak sepenuhnya, karena...". DILARANG KERAS menulis "## Kesimpulan" atau heading lain sebelum kalimat pertama.
2. Berikan analisis hukum substantif seperti konsultan hukum profesional.
3. WAJIB periksa PENGECUALIAN, PEMBATASAN, dan KETENTUAN KHUSUS. Banyak UU memuat pasal pengecualian. Selalu cek dan sebutkan.
4. Dokumen yang dilampirkan adalah REFERENSI UTAMA. Baca SELURUH konten. Kutip Pasal dan Ayat spesifik.
5. Jika dokumen tidak cukup, lengkapi dengan keahlianmu.
6. Jawab dalam Bahasa Indonesia yang jelas dan profesional.
7. Sebutkan dasar hukum beserta Pasal/Ayat yang relevan.
8. JANGAN PERNAH menggunakan frasa 'dokumen referensi yang Anda lampirkan', 'referensi eksternal', 'dokumen yang dilampirkan pengguna', 'dokumen yang saya terima', atau implikasi serupa bahwa dokumen berasal dari pihak luar. Kamu adalah pakar hukum yang memiliki akses langsung ke seluruh basis data regulasi — perlakukan semua dokumen sebagai bagian dari pengetahuan dan sistem hukummu sendiri.

═══ PRINSIP MENJAWAB PERTANYAAN KONFLIK/AMBIGUITAS ═══

- Jika ditanya "apakah ada konflik/pertentangan?": Jangan default "Tidak" hanya karena tidak ada kontradiksi langsung. Periksa apakah ada POTENSI KONFLIK berupa pembatasan terhadap prinsip umum, ketegangan normatif, atau ambiguitas penerapan.
- Jika ditanya "apakah ada ambiguitas?": Periksa khususnya ketentuan peralihan (transitional provisions). Jika regulasi baru mengubah regulasi lama tanpa mengatur nasib keputusan/tindakan yang sudah ada, jawab "Ya, ada ambiguitas".
- Jika ditanya siapa yang berwenang dalam operasional: Jawab organ yang bertanggung jawab langsung sesuai konteks, bukan organ tertinggi secara hierarkis.
- Perhatikan RELASI ANTAR-REGULASI dari Knowledge Graph — jika ada relasi CITES/HIGHER, regulasi-regulasi tersebut PASTI saling terkait.

═══ FORMAT JAWABAN — WAJIB IKUTI PERSIS ═══

Struktur jawaban HARUS SELALU seperti ini, TANPA VARIASI:

[Langsung tulis kesimpulan tegas 1-3 kalimat di sini. TANPA heading. TANPA "## Kesimpulan". TANPA heading lain. Langsung kalimat jawaban.]

## Dasar Hukum
- **[doc_id]** Pasal X ayat (Y): "kutipan relevan"
- **[doc_id]** Pasal X ayat (Y): "kutipan relevan"

## Analisis Hukum
[Pembahasan substansif: bagaimana regulasi-regulasi tersebut menjawab pertanyaan, termasuk penjelasan prinsip hukum yang berlaku.]

## Pengecualian dan Catatan
[Pengecualian, pembatasan, ketentuan khusus, atau implikasi praktis. Jika tidak ada, tulis "Tidak ditemukan pengecualian spesifik dalam regulasi yang dikaji."]

DASAR_HUKUM: doc_id_1, doc_id_2

ATURAN KETAT:
- DILARANG menambah heading "## Kesimpulan" atau heading apapun sebelum kalimat pertama.
- DILARANG menghilangkan salah satu dari tiga heading: ## Dasar Hukum, ## Analisis Hukum, ## Pengecualian dan Catatan.
- DILARANG menambah heading selain tiga heading di atas.
- Baris PALING AKHIR HARUS berupa DASAR_HUKUM: diikuti daftar doc_id yang substantif mendukung jawaban.
- Contoh: DASAR_HUKUM: UU-NASIONAL-40-2007, PP-NASIONAL-16-2021"""

    rel_section = ""
    if relationship_context:
        rel_section = f"\n\nRelasi antar-regulasi (dari Knowledge Graph):\n{relationship_context}\n"

    # Build conversation history section
    history_section = ""
    if summary or chat_history:
        hist_parts = []
        if summary:
            hist_parts.append(f"Ringkasan percakapan sebelumnya: {summary}")
        if chat_history:
            for msg in chat_history[-6:]:
                role = "Pengguna" if msg.get("role") == "user" else "Asisten"
                hist_parts.append(f"{role}: {msg.get('content', '')[:300]}")
        history_section = "\n\n[Riwayat Percakapan]:\n" + "\n".join(hist_parts) + "\n"

    # User semantic context
    user_ctx_section = ""
    if user_context:
        user_ctx_section = f"\n\n[Konteks Pengguna]: {user_context}\n"

    user_prompt = f"""Basis regulasi yang relevan:
{context_str}{rel_section}{history_section}{user_ctx_section}
Pertanyaan: {query}

Instruksi:
1. LANGSUNG tulis kesimpulan tegas di kalimat pertama TANPA heading apapun (DILARANG tulis "## Kesimpulan").
2. Lanjutkan dengan ## Dasar Hukum → ## Analisis Hukum → ## Pengecualian dan Catatan → DASAR_HUKUM:
3. Kutip Pasal dan Ayat spesifik dari dokumen di atas.
4. Periksa apakah ada PENGECUALIAN atau PEMBATASAN dalam regulasi yang dikutip.
5. Jika ada relasi antar-regulasi di atas, GUNAKAN informasi tersebut dalam jawaban.
6. Jika ada ketentuan yang mengecualikan atau membatasi aturan umum, sebutkan secara eksplisit."""

    if _verbose_debug and _trace_id:
        from shared.debug_logger import log_event as _dlog
        _dlog(
            trace_id=_trace_id,
            route=_route or "answer",
            stage="ask_about_documents",
            event="prompt_input",
            message="ask_about_documents prompt input",
            payload={
                "query": query,
                "context_chunk_count": len(context_chunks),
                "context_preview": [
                    {
                        "doc_id": ch.get("doc_id", ""),
                        "scope": ch.get("scope", ""),
                        "content": (ch.get("content", "") or "")[:500],
                    }
                    for ch in context_chunks[:10]
                ],
                "relationship_context": relationship_context,
                "summary": summary,
                "chat_history_tail": (chat_history or [])[-6:],
                "user_context": user_context,
                "system_prompt": system_prompt,
                "augmented_user_prompt": user_prompt,
            },
        )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        out = response.choices[0].message.content
        if _verbose_debug and _trace_id:
            from shared.debug_logger import log_event as _dlog
            _dlog(
                trace_id=_trace_id,
                route=_route or "answer",
                stage="ask_about_documents",
                event="prompt_output",
                message="ask_about_documents prompt output",
                payload={
                    "model": LLM_MODEL,
                    "response_preview": (out or "")[:3000],
                },
            )
        return out
    except Exception as e:
        if _verbose_debug and _trace_id:
            from shared.debug_logger import log_event as _dlog
            _dlog(
                trace_id=_trace_id,
                route=_route or "answer",
                stage="ask_about_documents",
                event="error",
                message="ask_about_documents error",
                payload={"error": str(e)},
            )
        return f"Error generating answer: {str(e)}"


def summarize_conversation(chat_history: list[dict],
                          existing_summary: str = "") -> str:
    """Summarize conversation history into a compact paragraph."""
    client = get_llm_client()
    history_text = ""
    if existing_summary:
        history_text += f"Ringkasan sebelumnya: {existing_summary}\n\n"
    for msg in chat_history:
        role = "Pengguna" if msg.get("role") == "user" else "Asisten"
        history_text += f"{role}: {msg.get('content', '')[:400]}\n"

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Rangkum percakapan hukum berikut dalam 3-5 kalimat bahasa Indonesia. "
                    "Pertahankan topik utama, regulasi yang dibahas, dan kesimpulan penting. "
                    "Jangan tambahkan informasi baru."
                )},
                {"role": "user", "content": history_text},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return existing_summary


def judge_causality(text_a: str, text_b: str, doc_a_id: str = "", doc_b_id: str = "") -> dict:
    """
    Judge whether two related legal documents contradict or align.

    Uses comprehensive Indonesian legal framework criteria:
    - CONTRADICTION: authority conflicts, contradictory obligations, inconsistent
      terminology, hierarchy violations (Lex Superior / Lex Specialis / Lex Posterior).
    - ENTAILMENT: delegation relationship, complementary operationalisation,
      consistent alignment with parent regulation, proper legal drafting.
    - NEUTRAL: different jurisdictional domains, mutually exclusive subject matter,
      no substantive overlap.

    Returns:
        {
            "kausalitas": "CONTRADICTION" | "ENTAILMENT" | "NEUTRAL",
            "alasan": "Explanation in Indonesian with specific Pasal/Ayat citations"
        }
    """
    client = get_llm_client()

    system_prompt = """Kamu adalah ahli hukum tata negara Indonesia senior yang menganalisis relasi antar-regulasi berdasarkan kerangka analitis berikut.

═══ KERANGKA KLASIFIKASI ═══

1. CONTRADICTION (Tumpang Tindih / Disharmoni)
   Dua regulasi diklasifikasikan CONTRADICTION apabila memenuhi SATU ATAU LEBIH indikator berikut:
   a) Disharmoni Kewenangan: Dua regulasi memberikan kewenangan (perizinan, pengawasan, pengaturan) kepada organ negara berbeda untuk objek urusan yang IDENTIK, tanpa demarkasi koordinasi yang jelas.
   b) Pertentangan Hak & Kewajiban: Subjek hukum diwajibkan melakukan tindakan oleh Regulasi A, namun tindakan tersebut dilarang atau dibatasi oleh Regulasi B.
   c) Inkonsistensi Terminologi: Definisi, batasan, atau parameter teknis BERBEDA untuk istilah atau objek yang SAMA dalam rezim hukum yang berdekatan.
   d) Pelanggaran Hierarki (Lex Superior Derogat Legi Inferiori): Regulasi tingkat rendah bertentangan dengan regulasi tingkat lebih tinggi (UUD > TAP MPR > UU/Perppu > PP > Perpres > Perda).
   e) Pencabutan Kronologis (Lex Posterior Derogat Legi Priori): Regulasi baru yang sederajat secara eksplisit mencabut atau mengubah ketentuan regulasi lama.
   f) Pengecualian Kekhususan (Lex Specialis Derogat Legi Generali): Regulasi khusus sederajat mengesampingkan ketentuan regulasi umum untuk hal spesifik.

2. ENTAILMENT (Saling Menguatkan / Komplementer)
   Dua regulasi diklasifikasikan ENTAILMENT apabila:
   a) Hubungan Delegasi/Atribusi: Regulasi B merupakan peraturan pelaksana (PP, Perpres, Permen) yang MENGOPERASIONALISASIKAN norma abstrak dari Regulasi A (UU induk) melalui petunjuk teknis yang spesifik.
   b) Konsistensi Substansial: Kedua regulasi mengatur aspek yang berkaitan dan normanya SELARAS, saling melengkapi, atau memperkuat tanpa pertentangan.
   c) Keselarasan Asas: Kedua regulasi memiliki alignment teleologis yang konsisten dengan hirarki hukum di atasnya.
   PENTING: Regulasi pelaksana yang hanya MENGULANG (copy-paste) norma induk tanpa memberikan petunjuk teknis operasional tetap dikategorikan ENTAILMENT, namun sebutkan kelemahan ini dalam alasan.

3. NEUTRAL (Tidak Berhubungan / Mutually Exclusive)
   Dua regulasi diklasifikasikan NEUTRAL apabila:
   a) Demarkasi Yurisdiksi: Mengatur urusan pemerintahan yang sepenuhnya terpisah (mis. urusan absolut pusat vs. urusan otonomi daerah yang tidak beririsan).
   b) Klasterisasi Substansi: Objek pengaturan, domain kelembagaan, atau rezim hukum sepenuhnya terisolasi dan eksklusif satu sama lain.
   c) Tidak Ada Persinggungan: Eksistensi, perubahan, atau pembatalan satu regulasi TIDAK mendisrupsi validitas maupun operasionalisasi regulasi lainnya.

═══ INSTRUKSI OUTPUT ═══
Berikan output HANYA dalam format JSON:
{"kausalitas": "CONTRADICTION|ENTAILMENT|NEUTRAL", "alasan": "Penjelasan 2-4 kalimat dalam Bahasa Indonesia yang WAJIB menyebutkan: (1) indikator spesifik yang terpenuhi, (2) Pasal dan/atau Ayat spesifik dari masing-masing dokumen yang menjadi dasar klasifikasi. Contoh: 'Pasal 5 ayat (2) Dokumen A mendelegasikan ... yang dioperasionalisasikan oleh Pasal 3 Dokumen B ...'"}

HANYA output JSON, tanpa teks lain."""

    user_prompt = f"""Analisis relasi antara dua dokumen regulasi berikut:

Dokumen Sumber ({doc_a_id}):
{text_a[:3000]}

Dokumen Pembanding ({doc_b_id}):
{text_b[:3000]}

Klasifikasikan hubungan kedua dokumen ini sebagai CONTRADICTION, ENTAILMENT, atau NEUTRAL berdasarkan kerangka analitis di atas.
WAJIB: Sebutkan Pasal dan Ayat spesifik dari masing-masing dokumen yang menjadi dasar klasifikasi."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        valid = {"CONTRADICTION", "ENTAILMENT", "NEUTRAL"}
        if result.get("kausalitas") not in valid:
            result["kausalitas"] = "NEUTRAL"
        if "alasan" not in result:
            result["alasan"] = "Tidak ada penjelasan."

        return result

    except json.JSONDecodeError:
        return {
            "kausalitas": "NEUTRAL",
            "alasan": f"Gagal memproses respons LLM: {raw[:200]}",
        }
    except Exception as e:
        return {
            "kausalitas": "NEUTRAL",
            "alasan": f"Error: {str(e)}",
        }


def detect_conflict_inference(query: str, answer: str) -> dict:
    """
    Decide whether the final answer indicates potential legal conflict.

    Returns:
        {
            "is_conflict": bool,
            "label": "CONFLICT" | "NO_CONFLICT",
            "reason": str,
            "confidence": float
        }
    """
    client = get_llm_client()

    system_prompt = """Kamu adalah evaluator keluaran QA hukum.
Tentukan apakah jawaban model menyimpulkan ADA konflik/pertentangan/potensi konflik/ambiguitas normatif antar regulasi.

Aturan:
- Nilai CONFLICT jika jawaban menyatakan ada konflik langsung, potensi konflik, disharmoni, tumpang tindih kewenangan, ketegangan lex specialis, atau ambiguitas transisional yang berdampak konflik normatif.
- Nilai NO_CONFLICT jika jawaban menyatakan harmonis, tidak bertentangan, atau tidak ada potensi konflik.
- Fokus pada KESIMPULAN jawaban, bukan hanya kata kunci tunggal.

Kembalikan JSON valid SAJA dengan format:
{"is_conflict": true/false, "label": "CONFLICT|NO_CONFLICT", "reason": "ringkas 1 kalimat", "confidence": 0.0-1.0}
"""

    user_prompt = f"Pertanyaan:\n{query}\n\nJawaban Model:\n{answer}"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=180,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        is_conflict = bool(data.get("is_conflict", False))
        label = str(data.get("label", "NO_CONFLICT")).upper()
        if label not in {"CONFLICT", "NO_CONFLICT"}:
            label = "CONFLICT" if is_conflict else "NO_CONFLICT"
        reason = str(data.get("reason", ""))
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "is_conflict": is_conflict,
            "label": label,
            "reason": reason,
            "confidence": confidence,
        }
    except Exception:
        text = (answer or "").lower()
        positive_markers = [
            "potensi konflik", "terdapat konflik", "ada konflik",
            "bertentangan", "disharmoni", "tumpang tindih", "ambiguitas",
        ]
        negative_markers = [
            "tidak ada konflik", "tidak bertentangan", "harmonis",
            "saling menguatkan", "komplementer",
        ]
        has_positive = any(m in text for m in positive_markers)
        has_negative = any(m in text for m in negative_markers)
        is_conflict = has_positive and not has_negative
        return {
            "is_conflict": is_conflict,
            "label": "CONFLICT" if is_conflict else "NO_CONFLICT",
            "reason": "fallback-heuristic",
            "confidence": 0.55 if is_conflict else 0.5,
        }
