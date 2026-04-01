"""
LangGraph Agentic RAG – Govnetic Chatbot
=========================================

6-node pipeline for Indonesian legal question answering.
Uses JSON-based Chain of Thought routing with user-facing narratives.

Architecture
------------
    ┌─────────────────────┐
    │ summarize_if_needed  │  Compress older chat history (>6 turns)
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │       router         │  LLM classifies query → direct | semantic | deep
    └──┬──────┬──────┬────┘
       │      │      │
  direct  semantic  deep
       │      │      │
       ▼      ▼      ▼
    ┌──────┐ ┌────────────┐ ┌──────────────┐
    │direct│ │  semantic   │ │    deep       │
    │lookup│ │  search     │ │  research     │
    └──┬───┘ └─────┬──────┘ └──────┬───────┘
       │           │               │
       │     may escalate          │
       │     to deep ──────────────┤
       ▼           ▼               ▼
    ┌─────────────────────────────────┐
    │        generate_answer           │
    └─────────────────────────────────┘

Routing Paths
-------------
1. **DIRECT** – Specific regulation lookup (e.g. "Pasal 5 UU 40 Tahun 2007")
   - Trigger : Regex detects explicit UU/PP/Perppu/Pergub reference in query
   - Flow    : router → direct_lookup → generate_answer
   - Method  : Exact doc_id match → Neo4j pasal/ayat fetch → VDB chunk fetch
   - Fallback: If doc not found → escalate to semantic

2. **SEMANTIC** – General legal concept (e.g. "syarat pendirian PT")
   - Trigger : LLM router classifies query as standard rule / concept question
   - Flow    : router → semantic_search → [generate_answer | deep_research]
   - Method  :
       a. Hybrid BM25 + Dense search (alpha=0.4, top_k=20)
       b. Select top-5 unique documents
       c. Gather all chunks per doc (VDB + Neo4j enrichment)
       d. **LLM Re-ranking** – each chunk scored 0-10 for query relevance,
          chunks below 4 filtered out (defeats embedding anisotropy)
       e. Sufficiency gate – LLM checks if retrieved context is complete
   - Escalate: If gate fails → auto-upgrade to deep

3. **DEEP** – Complex multi-law analysis (e.g. "konflik UU Cipta Kerja vs UU Ketenagakerjaan")
   - Trigger : LLM router identifies cross-regulation / exception / history query,
               OR semantic gate escalation
   - Flow    : router → deep_research → generate_answer
   - Method  :
       a. Multi-query expansion (query + 2 reformulations)
       b. Hybrid search across all variants (top_k=25 each)
       c. Neo4j graph traversal: smart_doc_lookup + citing docs (2-hop)
       d. LLM doc-level reranking (score >= 3.0)
       e. Assemble top-5 docs with full VDB + Neo4j chunks + relationships

State
-----
GraphState carries: query, route, primary_doc_ids, context_docs,
relationship_context, answer, logs, narratives, chat_history, summary,
user_context.
"""
import os
import re
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from utils import neo4j_client, pinecone_client, llm_stance
from utils.bm25_index import hybrid_search as _hybrid_search
from utils.benchmark_helpers import extract_doc_ids_from_question as _extract_doc_ids_from_question, get_unique_doc_ids as _get_unique_doc_ids

class GraphState(TypedDict):
    query: str
    route: str
    primary_doc_ids: List[str]
    context_docs: Dict[str, dict]
    relationship_context: str
    answer: str
    logs: List[str]
    narratives: List[str]  # User-facing legal explanations
    chat_history: List[dict]  # [{role: "user"|"assistant", content: str}]
    summary: str  # Condensed older conversation history
    user_context: str  # Injected semantic memory context

def _build_interleaved_context(primary_doc_ids, related_doc_ids, context_docs, max_chunks=30, max_chars=12000):
    result = []
    total_chars = 0
    doc_queues = {}
    for did in primary_doc_ids + related_doc_ids:
        info = context_docs.get(did)
        if not info: continue
        chunks = list(info["chunks"])
        scored = sorted([c for c in chunks if c.get("score") is not None], key=lambda c: c.get("score", 0), reverse=True)
        unscored = [c for c in chunks if c.get("score") is None]
        doc_queues[did] = scored + unscored
    if not doc_queues: return []
    seen_ids = set()
    doc_keys = list(doc_queues.keys())
    idx_map = {d: 0 for d in doc_keys}
    exhausted = set()
    while len(result) < max_chunks and total_chars < max_chars and len(exhausted) < len(doc_keys):
        for did in doc_keys:
            if did in exhausted: continue
            queue = doc_queues[did]
            idx = idx_map[did]
            if idx >= len(queue):
                exhausted.add(did)
                continue
            chunk = queue[idx]
            idx_map[did] = idx + 1
            cid = chunk.get("id", "")
            if cid and cid in seen_ids: continue
            if cid: seen_ids.add(cid)
            content = chunk.get("content", "")
            total_chars += len(content)
            result.append(chunk)
            if len(result) >= max_chunks or total_chars >= max_chars: break
    return result

def _keyword_score(text: str, keywords: list[str]) -> float:
    """Score text by keyword overlap — higher means more relevant."""
    text_lower = text.lower()
    return sum(1.0 for kw in keywords if kw in text_lower)

# ── Known sibling/successor laws for cross-validation ──────────────────────
# Laws that cover the same subject matter — when one is referenced,
# the sibling should be searched too for supplementary content.
_SIBLING_LAW_MAP: dict[str, list[str]] = {
    "PERPPU-NASIONAL-2-2022": ["UU-NASIONAL-11-2020"],
    "UU-NASIONAL-11-2020":    ["PERPPU-NASIONAL-2-2022"],
    "UU-NASIONAL-6-2023":     ["PERPPU-NASIONAL-2-2022", "UU-NASIONAL-11-2020"],
}

# Known inter-document relationships that may be missing from the graph.
_KNOWN_RELATIONSHIPS: list[tuple[str, str, str]] = [
    ("PERPPU-NASIONAL-2-2022", "CITES",  "UU-NASIONAL-40-2007"),
    ("PERPPU-NASIONAL-2-2022", "CITES",  "UU-NASIONAL-11-2020"),
    ("UU-NASIONAL-11-2020",    "CITES",  "UU-NASIONAL-40-2007"),
    ("UU-NASIONAL-6-2023",     "CITES",  "PERPPU-NASIONAL-2-2022"),
]

def _expand_with_siblings(doc_ids: list[str]) -> list[str]:
    """Expand doc list with known sibling/successor laws."""
    expanded = list(doc_ids)
    seen = set(expanded)
    for did in doc_ids:
        for sib in _SIBLING_LAW_MAP.get(did, []):
            if sib not in seen:
                expanded.append(sib)
                seen.add(sib)
    return expanded

def _inject_known_relationships(doc_ids: list[str], existing_rel: str) -> str:
    """Add known relationships that may be missing from the graph."""
    id_set = set(doc_ids)
    lines = []
    for src, rel, tgt in _KNOWN_RELATIONSHIPS:
        if src in id_set or tgt in id_set:
            line = f"- {src} --[{rel}]--> {tgt}"
            if line not in existing_rel:
                lines.append(line)
    if lines:
        return (existing_rel + "\n" + "\n".join(lines)).strip()
    return existing_rel

def _assemble_context_for_state(state: GraphState, raw_vdb_hits=None) -> GraphState:
    doc_ids = state.get("primary_doc_ids", [])

    # Cross-validate: expand with sibling laws
    doc_ids = _expand_with_siblings(doc_ids)
    state["primary_doc_ids"] = doc_ids
    context_docs = state.get("context_docs", {})
    seen_chunk_ids = set()
    for d, info in context_docs.items():
        for ch in info.get("chunks", []):
            if ch.get("id"): seen_chunk_ids.add(ch["id"])

    # Build keyword list from query for scoring Neo4j chunks
    _q_words = re.findall(r'[a-zA-Z\u00C0-\u024F]+', state.get("query", "").lower())
    _stopwords = {"yang", "dan", "di", "ke", "dari", "untuk", "dengan", "dalam",
                  "ini", "itu", "adalah", "pada", "atau", "bahwa", "sebagai",
                  "antara", "bagaimana", "apa", "apakah", "tidak", "ada",
                  "tentang", "hal", "tahun", "nomor", "pasal"}
    _keywords = [w for w in _q_words if w not in _stopwords and len(w) > 2]

    if raw_vdb_hits:
        for ch in raw_vdb_hits:
            did = ch.get("doc_id", "")
            if did in doc_ids:
                context_docs.setdefault(did, {"source": "VDB", "chunks": []})
                cid = ch.get("id")
                if cid and cid not in seen_chunk_ids:
                    context_docs[did]["chunks"].append(ch)
                    seen_chunk_ids.add(cid)

    # Track which docs already have VDB-scored chunks
    _docs_with_vdb = set()
    for did, info in context_docs.items():
        if any(ch.get("score") is not None for ch in info.get("chunks", [])):
            _docs_with_vdb.add(did)

    # For docs without VDB hits, do a targeted Pinecone search
    _docs_needing_vdb = [d for d in doc_ids if d not in _docs_with_vdb]
    if _docs_needing_vdb and _keywords:
        try:
            emb = llm_stance.get_embedding(state.get("query", ""))
            pc_index = pinecone_client.get_index()
            for did in _docs_needing_vdb[:5]:
                try:
                    res = pc_index.query(
                        vector=emb, top_k=10,
                        include_metadata=True,
                        filter={"doc_id": did},
                    )
                    context_docs.setdefault(did, {"source": "VDB-targeted", "chunks": []})
                    for m in res.get("matches", []):
                        cid = m["id"]
                        if cid not in seen_chunk_ids:
                            meta = m.get("metadata", {})
                            context_docs[did]["chunks"].append({
                                "id": cid, "doc_id": did,
                                "content": meta.get("content", ""),
                                "scope": meta.get("scope", ""),
                                "score": round(m["score"], 4),
                            })
                            seen_chunk_ids.add(cid)
                except Exception:
                    pass
        except Exception:
            pass

    if neo4j_client.test_connection():
        for did in doc_ids:
            try:
                detail = neo4j_client.get_document_detail(did)
                context_docs.setdefault(did, {"source": "Graph", "chunks": []})
                neo_chunks = []
                for p in detail.get("pasals", []) + detail.get("ayats", []):
                    content = p.get("content", "")
                    pid = str(p.get("name", ""))
                    nid = f"neo-{did}-{pid}"
                    if content and len(content) > 20 and nid not in seen_chunk_ids:
                        kscore = _keyword_score(content, _keywords)
                        neo_chunks.append({"id": nid, "doc_id": did, "content": content,
                                           "scope": "neo4j-pasal", "_kscore": kscore})
                        seen_chunk_ids.add(nid)
                # Sort Neo4j chunks by keyword relevance (desc) so topic-relevant
                # pasals surface first in the context window
                neo_chunks.sort(key=lambda c: c["_kscore"], reverse=True)
                for ch in neo_chunks:
                    ch.pop("_kscore", None)
                context_docs[did]["chunks"].extend(neo_chunks)
            except Exception: pass
    rel_context = state.get("relationship_context", "")
    if neo4j_client.test_connection():
        try:
            edges = neo4j_client.get_edges_between(list(context_docs.keys()))
            lines = [f'- {e["source_id"]} --[{e["type"]}]--> {e["target_id"]}' for e in edges.get("edges", [])]
            if lines:
                new_rels = "\\n".join(lines)
                if new_rels not in rel_context: rel_context += "\\n" + new_rels
        except Exception: pass
    # Inject known relationships that may be missing from the graph
    rel_context = _inject_known_relationships(doc_ids, rel_context)
    state["context_docs"] = context_docs
    state["relationship_context"] = rel_context.strip()
    # Debug: log chunk counts per doc
    _logs = state.get("logs", [])
    for did, info in context_docs.items():
        _logs.append(f"[Context] {did}: {len(info.get('chunks', []))} chunks (src={info.get('source', '?')})")
    if rel_context:
        _logs.append(f"[Context] Relationships: {rel_context[:600]}")
    state["logs"] = _logs
    return state

def _build_history_context(state: GraphState) -> str:
    """Build a compact conversation context string from summary + recent turns."""
    parts = []
    summary = state.get("summary", "")
    if summary:
        parts.append(f"[Ringkasan percakapan sebelumnya]: {summary}")
    history = state.get("chat_history", [])
    recent = history[-6:]  # last 3 exchanges
    for msg in recent:
        role = "Pengguna" if msg.get("role") == "user" else "Asisten"
        parts.append(f"{role}: {msg.get('content', '')[:300]}")
    return "\n".join(parts)


def router_node(state: GraphState) -> GraphState:
    query = state["query"]
    logs = state.get("logs", [])
    narratives = state.get("narratives", [])
    logs.append("[Router Node] Started analysis")
    
    regex_ids = _extract_doc_ids_from_question(query)

    # If the query references multiple regulations AND asks an analytical
    # question (relationship, conflict, comparison, history), route to deep
    # so the full graph traversal + reranking pipeline runs.
    _analytical_keywords = [
        "hubungan", "konflik", "bertentangan", "pertentangan", "perbandingan",
        "membandingkan", "tumpang tindih", "disharmoni", "lex specialis",
        "lex posterior", "lex superior", "harmonisasi", "keterkaitan",
        "mengubah", "mencabut", "menggantikan", "perubahan",
        "amandemen", "riwayat", "sejarah", "perbedaan",
    ]
    _q_lower = query.lower()
    _is_analytical = any(kw in _q_lower for kw in _analytical_keywords)

    if regex_ids and len(regex_ids) >= 2 and _is_analytical:
        # Multi-law analytical question → deep research for full context
        state["route"] = "deep"
        state["primary_doc_ids"] = list(regex_ids)
        logs.append(f"[Router Node] Regex hit ({len(regex_ids)} docs) + analytical query → deep")
        narratives.append(f"Teridentifikasi rujukan terhadap {len(regex_ids)} regulasi ({', '.join(list(regex_ids))}). Pertanyaan bersifat analitis, diperlukan penelusuran mendalam terhadap relasi antar-peraturan.")
        state["logs"] = logs
        state["narratives"] = narratives
        return state

    if regex_ids:
        state["route"] = "direct"
        state["primary_doc_ids"] = list(regex_ids)
        logs.append(f"[Router Node] Regex hit: {', '.join(list(regex_ids))}")
        narratives.append(f"Teridentifikasi rujukan langsung terhadap regulasi {', '.join(list(regex_ids))}. Melakukan penelusuran ketentuan hukum terkait.")
        state["logs"] = logs
        state["narratives"] = narratives
        return state

    # Build conversation context for follow-up awareness
    history_ctx = _build_history_context(state)
    history_block = ""
    if history_ctx:
        history_block = f"\n\n[Konteks Percakapan Sebelumnya]:\n{history_ctx}\n"

    system_prompt = """You are a Legal AI Planner indexing Indonesian law.
Analyze the user's query and return a strict JSON object with EXACTLY two keys:
1. "thought_process": A professional legal analysis in Indonesian, as written by a senior legal consultant explaining their research strategy. Use formal language. No emojis. Do NOT mention system architecture terms like "vector DB", "Deep Research node", "Pinecone", or "semantic search". Communicate like a practicing lawyer explaining reasoning to a client. (e.g., "Perlu dilakukan penelusuran terhadap ketentuan umum perseroan terbatas serta kemungkinan adanya pengecualian atau pembatasan yang diatur dalam regulasi sektoral...").
2. "route": Must strictly be ONE of these exact words:
   - "direct" (If asking standard specific pasal/UU)
   - "semantic" (If asking a general concept or standard rule)
   - "deep" (If asking a complex analytical question about conflicts, harmony, exceptions, or legal history between multiple laws)

Return ONLY valid JSON.
"""
    client = llm_stance.get_llm_client()
    try:
        router_model = os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL)
        user_content = query + history_block
        resp = client.chat.completions.create(
            model=router_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            max_tokens=250, temperature=0.1
        )
        content = (resp.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        route = data.get("route", "semantic").lower()
        thought = data.get("thought_process", "Merumuskan strategi penelusuran regulasi yang tepat berdasarkan substansi pertanyaan hukum yang diajukan.")
        
        state["route"] = route if route in ["direct", "semantic", "deep"] else "semantic"
        narratives.append(thought)
        logs.append(f"[Router Node] LLM chose route: {state['route']}")
    except Exception as e:
        state["route"] = "semantic"
        logs.append(f"[Router Node] Fallback to semantic. Err: {e}")
        narratives.append("Memulai penelusuran pada basis data regulasi untuk menemukan ketentuan hukum yang relevan.")
        
    state["logs"] = logs
    state["narratives"] = narratives
    return state

def direct_lookup_node(state: GraphState) -> GraphState:
    state["logs"].append("[Direct Node] Running direct meta lookup")
    query = state["query"]
    doc_ids = state.get("primary_doc_ids", [])
    
    if not doc_ids and neo4j_client.test_connection():
        all_docs = neo4j_client.get_all_documents()
        doc_ids = llm_stance.smart_doc_lookup(query, all_docs)[:3]
        
    if not doc_ids:
        state["logs"].append("[Direct Node] No docs found, fallback -> semantic")
        state["route"] = "semantic"
        return state

    # Graph expansion: follow CITES/HIGHER edges to discover related docs
    if neo4j_client.test_connection():
        known = set(doc_ids)
        for did in list(doc_ids):
            try:
                sub = neo4j_client.get_citing_documents(did, hops=1)
                for n in sub.get("nodes", []):
                    ndid = n.get("doc_id", "")
                    if ndid and ndid not in known:
                        doc_ids.append(ndid)
                        known.add(ndid)
            except Exception:
                pass
        if len(doc_ids) > len(state.get("primary_doc_ids", [])):
            state["logs"].append(f"[Direct Node] Graph expansion: {len(known)} docs total")

    state["primary_doc_ids"] = doc_ids
    state["logs"].append(f"[Direct Node] Dokumen ditemukan: {', '.join(doc_ids[:8])}")
    state["narratives"].append("Dokumen regulasi berhasil ditemukan. Melakukan analisis terhadap ketentuan pasal dan ayat yang berlaku.")
    return _assemble_context_for_state(state)

# ── Well-known topic → UU mapping for query expansion ──────────────────────
_TOPIC_LAW_MAP = {
    "perseroan terbatas": "Undang-Undang Nomor 40 Tahun 2007 tentang Perseroan Terbatas",
    "pt": "Undang-Undang Nomor 40 Tahun 2007 tentang Perseroan Terbatas",
    "ketenagakerjaan": "Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan",
    "cipta kerja": "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja",
    "bangunan gedung": "Undang-Undang Nomor 28 Tahun 2002 tentang Bangunan Gedung",
    "jasa konstruksi": "Undang-Undang Nomor 2 Tahun 2017 tentang Jasa Konstruksi",
    "penanaman modal": "Undang-Undang Nomor 25 Tahun 2007 tentang Penanaman Modal",
    "umkm": "Undang-Undang Nomor 20 Tahun 2008 tentang UMKM",
    "usaha mikro": "Undang-Undang Nomor 20 Tahun 2008 tentang UMKM",
    "perdagangan": "Undang-Undang Nomor 7 Tahun 2014 tentang Perdagangan",
    "arsitek": "Undang-Undang Nomor 6 Tahun 2017 tentang Arsitek",
    "pemerintahan daerah": "Undang-Undang Nomor 23 Tahun 2014 tentang Pemerintahan Daerah",
    "bantuan hukum": "Undang-Undang Nomor 16 Tahun 2011 tentang Bantuan Hukum",
    "perumahan": "Undang-Undang Nomor 1 Tahun 2011 tentang Perumahan dan Kawasan Permukiman",
    "rumah susun": "Undang-Undang Nomor 20 Tahun 2011 tentang Rumah Susun",
}

def _expand_for_definition(query: str) -> list[str]:
    """Generate definition-targeted query variants for general concept questions.
    Includes well-known topic → UU mapping for targeted BM25 matching.
    Rule-based (zero LLM cost, zero latency).
    """
    import re as _re
    q = query.lower().strip().rstrip("?!.")
    q = _re.sub(
        r"^(jelaskan\s+(tentang\s+)?|apa\s+(itu\s+|yang\s+dimaksud\s+(dengan\s+)?)?|"
        r"definisi\s+|pengertian\s+|ceritakan\s+(tentang\s+)?|"
        r"uraikan\s+(tentang\s+)?|deskripsikan\s+)",
        "", q
    ).strip()
    if not q or len(q) < 2:
        return []
    variants = [
        f"{q} adalah",
        f"definisi {q}",
        f"pengertian {q}",
        f"yang dimaksud dengan {q}",
    ]
    # Add well-known UU reference as explicit expansion
    for topic, uu_ref in _TOPIC_LAW_MAP.items():
        if topic in q:
            variants.append(uu_ref)
            break
    return variants


def _extract_doc_references(query: str) -> list[str]:
    """Extract explicit law references from query text and map to doc_id format.

    E.g. "UU 40 tahun 2007" → ["UU-NASIONAL-40-2007"]
         "PP No. 16/2021"   → ["PP-NASIONAL-16-2021"]
    Returns list of candidate doc_ids (may include aliases).
    """
    import re as _re
    results = []
    seen = set()
    # Patterns: "UU 40 tahun 2007", "UU No. 40/2007", "UU 40/2007", "PP 16 Tahun 2021"
    patterns = [
        (r'(?:Permen\s+PUPR|PERMEN\s+PUPR)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPUPR"),
        (r'(?:Permen\s+PPN|PERMEN\s+PPN)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPPN"),
        (r'(?:Perppu|PERPPU)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERPPU"),
        (r'(?:Pergub|PERGUB)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERGUB"),
        (r'\b(?:UU|Undang-Undang)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "UU"),
        (r'\b(?:PP|Peraturan\s+Pemerintah)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PP"),
    ]
    for pat, jenis in patterns:
        for m in _re.finditer(pat, query, _re.IGNORECASE):
            nomor = m.group(1)
            tahun = m.group(2)
            scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
            doc_id = f"{jenis}-{scope}-{nomor}-{tahun}"
            if doc_id not in seen:
                results.append(doc_id)
                seen.add(doc_id)
    return results


def _get_diverse_doc_ids(hits: list[dict], max_docs: int, max_per_doc: int = 4) -> list[str]:
    """Extract unique doc_ids using best-chunk-per-doc ranking."""
    best_score: dict[str, float] = {}
    doc_count: dict[str, int] = {}
    for h in hits:
        did = h.get("doc_id", "")
        if not did:
            continue
        score = h.get("rrf_score", 0)
        if did not in best_score or score > best_score[did]:
            best_score[did] = score
        doc_count[did] = doc_count.get(did, 0) + 1
    ranked_docs = sorted(best_score.keys(), key=lambda d: best_score[d], reverse=True)
    return ranked_docs[:max_docs]


def _rerank_chunks_llm(query: str, chunks: list[dict], batch_size: int = 10) -> list[dict]:
    """Score chunks for relevance using LLM (cross-encoder equivalent).

    Immune to embedding anisotropy — uses full query+document attention.
    Returns chunks with 'rerank_score', sorted descending.
    Only chunks with score >= 4 are returned.
    """
    import re as _re
    client = llm_stance.get_llm_client()
    scored = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        chunk_texts = []
        for j, ch in enumerate(batch):
            content = ch.get("content", "").strip()
            if len(content) > 500:
                content = content[:500] + "..."
            doc_id = ch.get("doc_id", "unknown")
            chunk_texts.append(f"[{j}] (doc: {doc_id})\n{content}")

        chunks_str = "\n\n".join(chunk_texts)

        prompt = f"""Kamu adalah evaluator relevansi dokumen hukum Indonesia.

PERTANYAAN PENGGUNA:
{query}

POTONGAN DOKUMEN:
{chunks_str}

Untuk setiap potongan dokumen di atas, berikan skor relevansi 0-10 terhadap pertanyaan pengguna:
- 0-2: Tidak relevan sama sekali (boilerplate, "Cukup Jelas", topik berbeda)
- 3-4: Sedikit relevan (topik terkait tapi tidak menjawab pertanyaan)
- 5-7: Cukup relevan (membahas topik yang ditanyakan)
- 8-10: Sangat relevan (langsung menjawab pertanyaan)

Balas HANYA dalam format JSON array, contoh: [7, 2, 9, 0, 5]
Jumlah elemen harus tepat {len(batch)}."""

        try:
            resp = client.chat.completions.create(
                model=llm_stance.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            raw_resp = resp.choices[0].message.content.strip()
            match = _re.search(r'\[[\d\s,]+\]', raw_resp)
            if match:
                scores = json.loads(match.group())
            else:
                scores = [5] * len(batch)
        except Exception:
            scores = [5] * len(batch)

        while len(scores) < len(batch):
            scores.append(5)
        scores = scores[:len(batch)]

        for ch, score in zip(batch, scores):
            ch_copy = dict(ch)
            ch_copy["rerank_score"] = int(score)
            if int(score) >= 4:
                scored.append(ch_copy)

    scored.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    return scored


def _follow_graph_edges(doc_ids: list[str], max_extra: int = 4) -> list[str]:
    """Follow Neo4j CITES/HIGHER edges from given doc_ids to discover related docs.
    Returns newly-discovered doc_ids not already in input list.
    """
    if not neo4j_client.test_connection():
        return []
    known = set(doc_ids)
    new_docs = []
    for did in doc_ids[:3]:  # limit to top-3 to control latency
        try:
            sub = neo4j_client.get_citing_documents(did, hops=1)
            for n in sub.get("nodes", []):
                ndid = n.get("doc_id", "")
                if ndid and ndid not in known:
                    new_docs.append(ndid)
                    known.add(ndid)
                    if len(new_docs) >= max_extra:
                        return new_docs
        except Exception:
            pass
    return new_docs

def semantic_search_node(state: GraphState) -> GraphState:
    """Hybrid BM25+Dense search with LLM re-ranking.

    Pipeline (V1 base + LLM rerank):
      1. Hybrid search with V1 params (alpha=0.4, top_k=20)
      2. Flat unique doc selection (top-5)
      3. Gather chunks per doc (VDB + Neo4j)
      4. LLM re-rank: score each chunk 0-10, filter low-relevance
      5. Sufficiency gate → answer or escalate to deep
    """
    state["logs"].append("[Semantic Node] Hybrid search (dense + BM25) + LLM rerank")
    query = state["query"]

    try:
        # Step 1: Hybrid BM25+Dense search (V1 params)
        emb = llm_stance.get_embedding(query)
        raw = _hybrid_search(query, emb, top_k=20, alpha=0.4)

        # Step 2: Select top docs (V1 flat unique logic)
        doc_ids = []
        for h in raw:
            did = h.get("doc_id", "")
            if did and did not in doc_ids:
                doc_ids.append(did)
            if len(doc_ids) >= 5:
                break
    except Exception as e:
        state["logs"].append(f"[Semantic Node] Hybrid Err: {str(e)}. Fallback -> deep")
        state["route"] = "deep"
        return state

    if not doc_ids:
        state["route"] = "deep"
        return state

    state["logs"].append(f"[Semantic Node] Dokumen ditemukan: {', '.join(doc_ids[:5])}. Reranking chunks...")

    # Step 3: Gather chunks per doc
    context_docs = {}
    seen_chunk_ids = set()
    for did in doc_ids:
        doc_chunks = [h for h in raw if h.get("doc_id") == did]
        for ch in doc_chunks:
            seen_chunk_ids.add(ch.get("id", ""))
        try:
            extra = pinecone_client.fetch_by_doc_id(did, top_k=80)
            for ch in extra:
                cid = ch.get("id", "")
                if cid not in seen_chunk_ids:
                    doc_chunks.append(ch)
                    seen_chunk_ids.add(cid)
        except Exception:
            pass
        context_docs[did] = {"source": "hybrid", "chunks": doc_chunks}

    # Neo4j enrichment
    if neo4j_client.test_connection():
        for did in doc_ids:
            try:
                detail = neo4j_client.get_document_detail(did)
                for p in detail.get("pasals", []) + detail.get("ayats", []):
                    content = p.get("content", "")
                    pid = str(p.get("name", ""))
                    nid = f"neo-{did}-{pid}"
                    if content and len(content) > 20 and nid not in seen_chunk_ids:
                        context_docs.setdefault(did, {"source": "hybrid", "chunks": []})[
                            "chunks"
                        ].append({"id": nid, "doc_id": did, "content": content, "scope": "neo4j-pasal"})
                        seen_chunk_ids.add(nid)
            except Exception:
                pass

    # Build relationship context
    rel_context = ""
    if neo4j_client.test_connection():
        try:
            edges = neo4j_client.get_edges_between(list(context_docs.keys()))
            lines = [f'- {e["source_id"]} --[{e["type"]}]--> {e["target_id"]}' for e in edges.get("edges", [])]
            if lines:
                rel_context = "\n".join(lines)
        except Exception:
            pass

    # Collect candidate chunks (interleaved across docs)
    all_candidates = []
    total_chars = 0
    doc_queues = {}
    for did in doc_ids:
        info = context_docs.get(did)
        if not info:
            continue
        chunks = list(info["chunks"])
        scored_chunks = sorted(
            [c for c in chunks if c.get("score") is not None or c.get("rrf_score") is not None],
            key=lambda c: c.get("rrf_score", 0) or c.get("score", 0), reverse=True
        )
        unscored = [c for c in chunks if c.get("score") is None and c.get("rrf_score") is None]
        doc_queues[did] = scored_chunks + unscored

    doc_keys = list(doc_queues.keys())
    idx_map = {d: 0 for d in doc_keys}
    exhausted = set()
    seen_build = set()
    while len(all_candidates) < 40 and total_chars < 16000 and len(exhausted) < len(doc_keys):
        for did in doc_keys:
            if did in exhausted:
                continue
            queue = doc_queues[did]
            idx = idx_map[did]
            if idx >= len(queue):
                exhausted.add(did)
                continue
            chunk = queue[idx]
            idx_map[did] = idx + 1
            cid = chunk.get("id", "")
            if cid and cid in seen_build:
                continue
            if cid:
                seen_build.add(cid)
            content = chunk.get("content", "")
            total_chars += len(content)
            all_candidates.append(chunk)
            if len(all_candidates) >= 40 or total_chars >= 16000:
                break

    # Step 4: LLM Re-ranking
    state["logs"].append(f"[Semantic Node] Reranking {len(all_candidates)} chunks via LLM...")
    reranked = _rerank_chunks_llm(query, all_candidates)
    state["logs"].append(f"[Semantic Node] {len(reranked)} chunks passed rerank filter (score >= 4)")

    # Build final context from reranked chunks
    llm_chunks = []
    char_count = 0
    for ch in reranked:
        content = ch.get("content", "")
        char_count += len(content)
        llm_chunks.append(ch)
        if len(llm_chunks) >= 30 or char_count >= 12000:
            break

    if not llm_chunks:
        llm_chunks = all_candidates[:20]  # fallback if reranker filtered everything

    # Recompute doc_ids from reranked chunks
    reranked_doc_ids = []
    for ch in llm_chunks:
        did = ch.get("doc_id", "")
        if did and did not in reranked_doc_ids:
            reranked_doc_ids.append(did)

    # Sufficiency gate
    summaries = {}
    for ch in llm_chunks:
        did = ch.get("doc_id", "")
        if not did:
            continue
        content = ch.get("content", "").strip()
        if len(content) < 30:
            continue
        prev = summaries.get(did, "")
        if len(content) > len(prev):
            summaries[did] = content[:500]

    sys_eval = """You are a senior lawyer assessing retrieval context.
The user asked a legal question. I have retrieved some document excerpts.
Determine if the provided excerpts comprehensively answer the question, or if we might be missing specific exceptions, definitions, or connected laws.
Return a strict JSON object with EXACTLY two keys:
1. "thought_process": string - As a senior legal consultant, explain whether the retrieved documents sufficiently address all aspects of the query. Use formal professional Indonesian. No emojis.
2. "is_sufficient": boolean (true or false). Return false if you suspect more context is needed.

ONLY output valid JSON. Example: {"thought_process": "...", "is_sufficient": true}
"""
    ctx_str = "\n\n".join([f"DOC {k}: {v}" for k, v in summaries.items()])
    client = llm_stance.get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL),
            messages=[
                {"role": "system", "content": sys_eval},
                {"role": "user", "content": f"Query: {query}\n\nRetrieved Context:\n{ctx_str}"}
            ],
            max_tokens=250, temperature=0.1
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            import re as _re
            _suf_match = _re.search(r'"is_sufficient"\s*:\s*(true|false)', content, _re.IGNORECASE)
            data = {"is_sufficient": _suf_match.group(1).lower() == "true"} if _suf_match else {"is_sufficient": False}

        is_sufficient = data.get("is_sufficient", True)
        thought = data.get("thought_process", "Memverifikasi apakah konteks hukum cukup...")
        state["narratives"].append(thought)

        if not is_sufficient:
            state["logs"].append("[Semantic Node] Gate Failed. Upgrading to deep.")
            state["route"] = "deep"
            return state

        state["logs"].append("[Semantic Node] Gate Passed.")
    except Exception as e:
        state["logs"].append(f"[Semantic Node] Gate Err: {e}")
        state["route"] = "deep"
        return state

    # Store reranked context in state
    cur_doc_ids = state.get("primary_doc_ids", [])
    for d in reranked_doc_ids:
        if d not in cur_doc_ids:
            cur_doc_ids.append(d)
    state["primary_doc_ids"] = cur_doc_ids

    # Build context_docs from reranked chunks
    reranked_context = {}
    for ch in llm_chunks:
        did = ch.get("doc_id", "")
        if not did:
            continue
        reranked_context.setdefault(did, {"source": "hybrid+rerank", "chunks": []})["chunks"].append(ch)
    state["context_docs"] = reranked_context
    state["relationship_context"] = rel_context.strip()
    return state

def deep_research_node(state: GraphState) -> GraphState:
    state["logs"].append("[Deep Node] Memulai penelusuran mendalam pada Graph dan VDB")
    state["narratives"].append("Diperlukan analisis hukum yang lebih komprehensif. Melakukan penelusuran mendalam terhadap regulasi terkait, termasuk riwayat perubahan dan relasi antar-peraturan.")
    query = state["query"]
    expanded = llm_stance.expand_query(query)
    
    raw = []
    seen = set()
    for trm in [query] + expanded[:2]:
        try:
            emb = llm_stance.get_embedding(trm)
            hits = _hybrid_search(trm, emb, top_k=25, alpha=0.4)
            for h in hits:
                if h["id"] not in seen:
                    raw.append(h)
                    seen.add(h["id"])
        except Exception: pass
            
    vdb_docs = _get_unique_doc_ids(raw, 10)
    merged_docs = list(vdb_docs)
    added = set(merged_docs)
    
    if neo4j_client.test_connection():
        all_docs = neo4j_client.get_all_documents()
        graph_docs = llm_stance.smart_doc_lookup(query, all_docs)
        for d in graph_docs:
            if d not in added:
                merged_docs.append(d)
                added.add(d)
                
        for did in list(merged_docs)[:3]:
            try:
                sub = neo4j_client.get_citing_documents(did, hops=2)
                for n in sub.get("nodes", []):
                    ndid = n.get("doc_id", "")
                    if ndid and ndid not in added:
                        merged_docs.append(ndid)
                        added.add(ndid)
            except Exception: pass
                
    # Also include sibling laws in the candidate pool
    for did in list(merged_docs):
        for sib in _SIBLING_LAW_MAP.get(did, []):
            if sib not in added:
                merged_docs.append(sib)
                added.add(sib)

    doc_summaries = {}
    for did in merged_docs[:15]:
        summary = ""
        chunks = [r for r in raw if r.get("doc_id") == did]
        if chunks: summary = chunks[0].get("content", "")
        if not summary and neo4j_client.test_connection():
            try:
                dtl = neo4j_client.get_document_detail(did)
                # Check pasals first, then ayats (pasals often have NULL content)
                for p in dtl.get("pasals", []):
                    if p.get("content") and len(p["content"]) > 30:
                        summary = p["content"]
                        break
                if not summary:
                    for a in dtl.get("ayats", []):
                        if a.get("content") and len(a["content"]) > 30:
                            summary = a["content"]
                            break
            except Exception: pass
        doc_summaries[did] = summary[:400]
        
    ranked = llm_stance.rerank_documents(query, doc_summaries)
    top_docs = [did for did, sc in ranked if sc >= 3.0][:5]
    if not top_docs: top_docs = merged_docs[:5]
        
    cur_doc_ids = state.get("primary_doc_ids", [])
    for d in top_docs:
        if d not in cur_doc_ids: cur_doc_ids.append(d)
            
    state["primary_doc_ids"] = cur_doc_ids
    state["logs"].append(f"[Deep Node] Reranked dokumen: {', '.join(cur_doc_ids[:5])}")
    doc_list = '; '.join(cur_doc_ids)
    state["narratives"].append(f"Ditemukan {len(cur_doc_ids)} dokumen: {doc_list}. Menyusun kesimpulan hukum.")
    return _assemble_context_for_state(state, raw_vdb_hits=raw)

def generate_answer_node(state: GraphState) -> GraphState:
    state["logs"].append("[Answer Node] Generating output")
    chunks = _build_interleaved_context(
        primary_doc_ids=state.get("primary_doc_ids", []),
        related_doc_ids=[],
        context_docs=state.get("context_docs", {}),
        max_chunks=40, max_chars=16000
    )
    ans = llm_stance.ask_about_documents(
        query=state["query"],
        context_chunks=chunks,
        relationship_context=state.get("relationship_context", ""),
        chat_history=state.get("chat_history", []),
        summary=state.get("summary", ""),
        user_context=state.get("user_context", ""),
    )
    state["answer"] = ans
    # Append current exchange to chat_history
    history = list(state.get("chat_history", []))
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": ans[:500]})
    state["chat_history"] = history
    return state

def route_after_direct(state: GraphState) -> str:
    if state.get("route") == "semantic": return "semantic_search"
    return "generate_answer"
    
def route_after_semantic(state: GraphState) -> str:
    if state.get("route") == "deep": return "deep_research"
    return "generate_answer"

def summarize_if_needed_node(state: GraphState) -> GraphState:
    """Summarize older conversation history when it exceeds 6 turns."""
    history = state.get("chat_history", [])
    if len(history) <= 6:
        return state
    # Summarize older turns, keep last 4
    older = history[:-4]
    kept = history[-4:]
    try:
        new_summary = llm_stance.summarize_conversation(
            older, existing_summary=state.get("summary", "")
        )
        state["summary"] = new_summary
        state["chat_history"] = kept
        state["logs"] = state.get("logs", []) + [f"[Summarize Node] Condensed {len(older)} msgs"]
    except Exception as e:
        state["logs"] = state.get("logs", []) + [f"[Summarize Node] Err: {e}"]
    return state


def create_agent(checkpointer=None):
    import sys
    from types import GeneratorType
    
    workflow = StateGraph(GraphState)
    workflow.add_node("summarize_if_needed", summarize_if_needed_node)
    workflow.add_node("router", router_node)
    workflow.add_node("direct_lookup", direct_lookup_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("deep_research", deep_research_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    workflow.set_entry_point("summarize_if_needed")
    workflow.add_edge("summarize_if_needed", "router")
    
    def router_condition(state: GraphState) -> str:
        r = state.get("route", "semantic")
        if r == "direct": return "direct_lookup"
        if r == "deep": return "deep_research"
        return "semantic_search"

    workflow.add_conditional_edges("router", router_condition)
    workflow.add_conditional_edges("direct_lookup", route_after_direct)
    workflow.add_conditional_edges("semantic_search", route_after_semantic)
    
    workflow.add_edge("deep_research", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    compile_kwargs = {}
    
    # Comprehensive checkpointer validation
    if checkpointer is not None:
        _type_name = type(checkpointer).__name__
        _type_str = str(type(checkpointer))
        
        # Red flags: context managers, generators, or invalid types
        if any(bad in _type_str for bad in ['GeneratorContextManager', 'contextmanager', '_GeneratorContextManager', 'contextlib']):
            print(f"[AGENT] ❌ Context manager detected: {_type_str}. Disabling checkpointer.", file=sys.stderr)
            checkpointer = None
        elif hasattr(checkpointer, '__enter__') and hasattr(checkpointer, '__exit__') and not hasattr(checkpointer, 'get_tuple'):
            print(f"[AGENT] ❌ Object is a context manager but not a valid saver. Disabling.", file=sys.stderr)
            checkpointer = None
        elif hasattr(checkpointer, 'get_tuple') and hasattr(checkpointer, 'put_writes'):
            print(f"[AGENT] ✅ Valid checkpointer: {_type_name}", file=sys.stderr)
            compile_kwargs["checkpointer"] = checkpointer
        else:
            print(f"[AGENT] ⚠️  Unknown checkpointer type: {_type_name}. Attempting to use it anyway.", file=sys.stderr)
            compile_kwargs["checkpointer"] = checkpointer
    
    return workflow.compile(**compile_kwargs)
