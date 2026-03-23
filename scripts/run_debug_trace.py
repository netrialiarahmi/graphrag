#!/usr/bin/env python3
"""
Manual debug trace for a single query through the GraphRAG pipeline.
Runs "apa itu bangunan?" and writes structured logs to chatbot/debug.log.

Usage:
    cd graphrag/
    GRAPHRAG_STANDALONE=1 python3 scripts/run_debug_trace.py
"""
import os, sys, json, uuid
from datetime import datetime, timezone

os.environ.setdefault("GRAPHRAG_STANDALONE", "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from shared import neo4j_client, pinecone_client, llm_stance
from shared.debug_logger import log_event, new_trace_id, get_log_path

# ── Config ────────────────────────────────────────────────────────────────────
QUERY = "apa itu bangunan?"
LOG_PATH = get_log_path()

# Clear previous log
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

trace_id = new_trace_id()
print(f"Trace ID: {trace_id}")
print(f"Query: {QUERY}")
print(f"Log: {LOG_PATH}")
print("=" * 60)

# ── Helper: extract doc ids from question (regex) ────────────────────────────
sys.path.insert(0, os.path.join(_ROOT, "chatbot"))
from utils.benchmark_helpers import extract_doc_ids_from_question, get_unique_doc_ids


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Router Node")

router_system = """You are a Legal AI Planner indexing Indonesian law.
Analyze the user's query and return a strict JSON object with EXACTLY two keys:
1. "thought_process": A user-friendly, non-technical explanation in Indonesian explaining what you are looking for (e.g., "Saya sedang menganalisis apakah pertanyaan ini meminta aturan umum, atau membutuhkan pencarian mendalam mengenai pengecualian hukum dan harmoni antar pasal..."). Do NOT mention system architecture terms like "vector DB" or "Deep Research node". Act like a human lawyer researching.
2. "route": Must strictly be ONE of these exact words:
   - "direct" (If asking standard specific pasal/UU)
   - "semantic" (If asking a general concept or standard rule)
   - "deep" (If asking a complex analytical question about conflicts, harmony, exceptions, or legal history between multiple laws)

Return ONLY valid JSON.
"""

log_event(trace_id=trace_id, route="router", stage="router", event="prompt_input",
          message="Router prompt input",
          payload={"system_prompt": router_system, "user_prompt": QUERY})

client = llm_stance.get_llm_client()
router_model = os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL)
resp = client.chat.completions.create(
    model=router_model,
    messages=[{"role": "system", "content": router_system}, {"role": "user", "content": QUERY}],
    max_tokens=250, temperature=0.1
)
router_raw = (resp.choices[0].message.content or "").strip()

log_event(trace_id=trace_id, route="router", stage="router", event="prompt_output",
          message="Router prompt output",
          payload={"raw_response": router_raw})

router_data = json.loads(router_raw.replace("```json", "").replace("```", "").strip())
route = router_data.get("route", "semantic")
print(f"  Route: {route}")
print(f"  Thought: {router_data.get('thought_process', '')[:120]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Semantic Search Node")

emb = llm_stance.get_embedding(QUERY)
raw_hits = pinecone_client.semantic_search(query_embedding=emb, top_k=20)
doc_ids = get_unique_doc_ids(raw_hits, 5)
print(f"  Doc IDs: {doc_ids}")

# Build summaries (first hit per doc — as current code does)
summaries = {}
for hit in raw_hits:
    did = hit.get("doc_id")
    if did and did not in summaries:
        summaries[did] = hit.get("content", "")[:500]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: SEMANTIC GATE (sufficiency check)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Semantic Gate")

gate_system = """You are a senior lawyer assessing retrieval context. 
The user asked a legal question. I have retrieved some document excerpts.
Determine if the provided excerpts comprehensively answer the question, or if we might be missing specific exceptions, definitions, or connected laws (e.g. anti-monopoly rules excluding UMKM).
Return a strict JSON object with:
1. "thought_process": Explain to the user whether the documents found are sufficient or missing nuances. Use friendly Indonesian. (e.g., "Konteks awal telah ditemukan, namun saya merasa perlu memverifikasi apakah ada pengecualian spesifik terkait UMKM...").
2. "is_sufficient": boolean (true or false). Return false if you suspect more context is needed.
ONLY output valid JSON.
"""
ctx_str = "\n\n".join([f"DOC {k}: {v}" for k, v in summaries.items()])
gate_user = f"Query: {QUERY}\\n\\nRetrieved Context:\\n{ctx_str}"

log_event(trace_id=trace_id, route="semantic", stage="semantic_gate", event="prompt_input",
          message="Semantic sufficiency prompt input",
          payload={"system_prompt": gate_system, "user_prompt": gate_user})

resp = client.chat.completions.create(
    model=router_model,
    messages=[
        {"role": "system", "content": gate_system},
        {"role": "user", "content": f"Query: {QUERY}\n\nRetrieved Context:\n{ctx_str}"}
    ],
    max_tokens=250, temperature=0.1
)
gate_raw = (resp.choices[0].message.content or "").strip()

log_event(trace_id=trace_id, route="semantic", stage="semantic_gate", event="prompt_output",
          message="Semantic sufficiency prompt output",
          payload={"raw_response": gate_raw})

gate_clean = gate_raw.replace("```json", "").replace("```", "").strip()
gate_data = json.loads(gate_clean)
is_sufficient = gate_data.get("is_sufficient", False)
print(f"  Sufficient: {is_sufficient}")
print(f"  Reasoning: {gate_data.get('thought_process', '')[:150]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: DEEP RESEARCH (if gate failed)
# ═══════════════════════════════════════════════════════════════════════════════
if not is_sufficient:
    print("\n[4] Deep Research Node")

    # expand_query (with trace)
    expanded = llm_stance.expand_query(QUERY, _trace_id=trace_id, _route="deep")
    print(f"  Expanded: {expanded}")

    log_event(trace_id=trace_id, route="deep", stage="deep_expand_query", event="prompt_result",
              message="Deep research expansion output",
              payload={"query": QUERY, "expanded_queries": expanded})

    # Merge VDB hits from expanded queries
    all_raw = list(raw_hits)
    seen = {h["id"] for h in raw_hits}
    for term in [QUERY] + expanded[:2]:
        try:
            e = llm_stance.get_embedding(term)
            hits = pinecone_client.semantic_search(query_embedding=e, top_k=25)
            for h in hits:
                if h["id"] not in seen:
                    all_raw.append(h)
                    seen.add(h["id"])
        except Exception:
            pass

    vdb_docs = get_unique_doc_ids(all_raw, 10)
    merged_docs = list(vdb_docs)
    added = set(merged_docs)

    # Graph lookup + citing docs
    if neo4j_client.test_connection():
        all_docs_list = neo4j_client.get_all_documents()
        graph_docs = llm_stance.smart_doc_lookup(QUERY, all_docs_list)
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
            except Exception:
                pass

    print(f"  Merged docs: {merged_docs[:10]}")

    # Build doc summaries for reranking
    doc_summaries = {}
    for did in merged_docs[:15]:
        summary = ""
        chunks = [r for r in all_raw if r.get("doc_id") == did]
        if chunks:
            summary = chunks[0].get("content", "")
        if not summary and neo4j_client.test_connection():
            try:
                dtl = neo4j_client.get_document_detail(did)
                ps = dtl.get("pasals", [])
                if ps:
                    summary = ps[0].get("content", "")
            except Exception:
                pass
        doc_summaries[did] = summary[:400]

    # Rerank (with trace)
    ranked = llm_stance.rerank_documents(QUERY, doc_summaries, _trace_id=trace_id, _route="deep")
    top_docs = [did for did, sc in ranked if sc >= 3.0][:5]
    if not top_docs:
        top_docs = merged_docs[:5]
    print(f"  Top docs after rerank: {top_docs}")
    doc_ids = top_docs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ASSEMBLE CONTEXT  (verbose retrieval payload)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] Assemble Context")

# Collect final retrieval items for the log
retrieval_items = []
context_docs = {}
seen_chunk_ids = set()

# VDB chunks
for ch in (all_raw if not is_sufficient else raw_hits):
    did = ch.get("doc_id", "")
    if did in doc_ids:
        context_docs.setdefault(did, {"source": "VDB", "chunks": []})
        cid = ch.get("id")
        if cid and cid not in seen_chunk_ids:
            context_docs[did]["chunks"].append(ch)
            seen_chunk_ids.add(cid)
            retrieval_items.append({
                "doc_id": did,
                "chunk_id": cid,
                "source": "VDB",
                "retrieval_method": ch.get("scope", "ayat"),
                "content": ch.get("content", "")[:500],
                "score": ch.get("score"),
            })

# Neo4j chunks
if neo4j_client.test_connection():
    for did in doc_ids:
        try:
            detail = neo4j_client.get_document_detail(did)
            context_docs.setdefault(did, {"source": "Graph", "chunks": []})
            for p in detail.get("pasals", []) + detail.get("ayats", []):
                content = p.get("content", "")
                pid = str(p.get("name", ""))
                nid = f"neo-{did}-{pid}"
                if content and len(content) > 20 and nid not in seen_chunk_ids:
                    context_docs[did]["chunks"].append({
                        "id": nid, "doc_id": did,
                        "content": content, "scope": "neo4j-pasal"
                    })
                    seen_chunk_ids.add(nid)
                    retrieval_items.append({
                        "doc_id": did,
                        "chunk_id": nid,
                        "source": "VDB",
                        "retrieval_method": "neo4j-pasal",
                        "content": content[:500],
                        "score": None,
                    })
        except Exception:
            pass

log_event(trace_id=trace_id, route="deep" if not is_sufficient else "semantic",
          stage="assemble_context", event="retrieval_items",
          message="Verbose retrieval payload",
          payload={"retrieval_items": retrieval_items})

print(f"  Total retrieval items: {len(retrieval_items)}")
for did in doc_ids:
    n = len(context_docs.get(did, {}).get("chunks", []))
    print(f"    {did}: {n} chunks")


print("\n" + "=" * 60)
print(f"Done. Log written to: {LOG_PATH}")
print(f"Total log entries: ", end="")
with open(LOG_PATH) as f:
    lines = f.readlines()
    print(len(lines))
