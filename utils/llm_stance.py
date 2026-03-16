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

def rerank_documents(query: str, doc_summaries: dict[str, str]) -> list[tuple[str, float]]:
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

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pertanyaan: {query}\n\nDokumen:\n{summaries_str}"},
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


def ask_about_documents(query: str, context_chunks: list[dict]) -> str:
    """
    RAG-style question answering: given a user query and relevant context chunks,
    generate an answer grounded in the legal documents.
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

    system_prompt = """Kamu adalah pakar hukum Indonesia senior dengan keahlian mendalam di segala bidang hukum — perdata, pidana, tata negara, korporasi, perizinan, ketenagakerjaan, dan sebagainya.

Cara menjawab:
1. Jawab pertanyaan secara LANGSUNG dan KOMPREHENSIF berdasarkan pengetahuan hukummu.
2. Berikan jawaban yang substantif, jelas, dan terstruktur — seperti seorang konsultan hukum profesional.
3. Dokumen regulasi yang dilampirkan adalah REFERENSI PENDUKUNG. Jika ada dokumen yang relevan, kutip Pasal dan Ayat spesifiknya untuk memperkuat jawabanmu.
4. Jangan pernah menolak menjawab hanya karena dokumen pendukung tidak memuat informasi yang ditanyakan. Gunakan keahlianmu.
5. Jika dokumen pendukung membahas topik yang berbeda dari pertanyaan, ABAIKAN dokumen tersebut dan jawab langsung dari pengetahuanmu.
6. Jawab dalam Bahasa Indonesia yang jelas dan profesional.
7. Sebutkan dasar hukum (UU, PP, Permen, dll.) beserta Pasal/Ayat yang relevan."""

    user_prompt = f"""Dokumen pendukung (referensi tambahan):
{context_str}

Pertanyaan: {query}

Jawab pertanyaan di atas secara langsung dan komprehensif. Kutip Pasal dan Ayat dari regulasi yang relevan."""

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
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


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
