"""
LangGraph Agentic RAG implementation for GraphRAG.
Supports JSON-based Chain of Thought routing with user-friendly narratives.
"""
import os, sys
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Resolve project root so shared/ is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared import neo4j_client, pinecone_client, llm_stance
from utils.benchmark_helpers import extract_doc_ids_from_question as _extract_doc_ids_from_question, get_unique_doc_ids as _get_unique_doc_ids

class GraphState(TypedDict):
    query: str
    route: str
    primary_doc_ids: List[str]
    context_docs: Dict[str, dict]
    relationship_context: str
    answer: str
    logs: List[str]
    narratives: List[str] # User-facing legal explanations

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

def _assemble_context_for_state(state: GraphState, raw_vdb_hits=None) -> GraphState:
    doc_ids = state.get("primary_doc_ids", [])
    context_docs = state.get("context_docs", {})
    seen_chunk_ids = set()
    for d, info in context_docs.items():
        for ch in info.get("chunks", []):
            if ch.get("id"): seen_chunk_ids.add(ch["id"])
    if raw_vdb_hits:
        for ch in raw_vdb_hits:
            did = ch.get("doc_id", "")
            if did in doc_ids:
                context_docs.setdefault(did, {"source": "VDB", "chunks": []})
                cid = ch.get("id")
                if cid and cid not in seen_chunk_ids:
                    context_docs[did]["chunks"].append(ch)
                    seen_chunk_ids.add(cid)
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
                        context_docs[did]["chunks"].append({"id": nid, "doc_id": did, "content": content, "scope": "neo4j-pasal"})
                        seen_chunk_ids.add(nid)
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
    state["context_docs"] = context_docs
    state["relationship_context"] = rel_context.strip()
    return state

def router_node(state: GraphState) -> GraphState:
    query = state["query"]
    logs = state.get("logs", [])
    narratives = state.get("narratives", [])
    logs.append("[Router Node] Started analysis")
    
    regex_ids = _extract_doc_ids_from_question(query)
    if regex_ids:
        state["route"] = "direct"
        state["primary_doc_ids"] = list(regex_ids)
        logs.append(f"[Router Node] Regex hit: {list(regex_ids)[0]}")
        narratives.append(f"Saya mengenali bahwa Anda menanyakan regulasi spesifik ({list(regex_ids)[0]}). Saya akan langsung memeriksa isi dokumen tersebut.")
        state["logs"] = logs
        state["narratives"] = narratives
        return state

    system_prompt = """You are a Legal AI Planner indexing Indonesian law.
Analyze the user's query and return a strict JSON object with EXACTLY two keys:
1. "thought_process": A user-friendly, non-technical explanation in Indonesian explaining what you are looking for (e.g., "Saya sedang menganalisis apakah pertanyaan ini meminta aturan umum, atau membutuhkan pencarian mendalam mengenai pengecualian hukum dan harmoni antar pasal..."). Do NOT mention system architecture terms like "vector DB" or "Deep Research node". Act like a human lawyer researching.
2. "route": Must strictly be ONE of these exact words:
   - "direct" (If asking standard specific pasal/UU)
   - "semantic" (If asking a general concept or standard rule)
   - "deep" (If asking a complex analytical question about conflicts, harmony, exceptions, or legal history between multiple laws)

Return ONLY valid JSON.
"""
    client = llm_stance.get_llm_client()
    try:
        router_model = os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL)
        resp = client.chat.completions.create(
            model=router_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            max_tokens=250, temperature=0.1
        )
        content = (resp.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        route = data.get("route", "semantic").lower()
        thought = data.get("thought_process", "Saya sedang merumuskan strategi penelusuran regulasi yang tepat...")
        
        state["route"] = route if route in ["direct", "semantic", "deep"] else "semantic"
        narratives.append(thought)
        logs.append(f"[Router Node] LLM chose route: {state['route']}")
    except Exception as e:
        state["route"] = "semantic"
        logs.append(f"[Router Node] Fallback to semantic. Err: {e}")
        narratives.append("Mencari jawaban melalui penelusuran semantik pada pustaka hukum...")
        
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
        
    state["primary_doc_ids"] = doc_ids
    state["logs"].append(f"[Direct Node] Found {len(doc_ids)} docs.")
    state["narratives"].append("Saya telah menemukan dokumen yang tepat, dan sedang membaca ketentuannya...")
    return _assemble_context_for_state(state)

def semantic_search_node(state: GraphState) -> GraphState:
    state["logs"].append("[Semantic Node] Searching Pinecone")
    query = state["query"]
    
    try:
        emb = llm_stance.get_embedding(query)
        raw = pinecone_client.semantic_search(query_embedding=emb, top_k=20)
        doc_ids = _get_unique_doc_ids(raw, 5)
    except Exception as e:
        state["logs"].append(f"[Semantic Node] VDB Err: {str(e)}. Fallback -> deep")
        state["route"] = "deep"
        return state

    if not doc_ids:
        state["route"] = "deep"
        return state
        
    state["logs"].append(f"[Semantic Node] Found {len(doc_ids)} docs. Checking sufficiency...")
    summaries = {}
    for hit in raw:
        did = hit.get("doc_id")
        if did and did not in summaries:
            summaries[did] = hit.get("content", "")[:500]

    # JSON Sufficiency check
    sys_eval = """You are a senior lawyer assessing retrieval context. 
The user asked a legal question. I have retrieved some document excerpts.
Determine if the provided excerpts comprehensively answer the question, or if we might be missing specific exceptions, definitions, or connected laws (e.g. anti-monopoly rules excluding UMKM).
Return a strict JSON object with:
1. "thought_process": Explain to the user whether the documents found are sufficient or missing nuances. Use friendly Indonesian. (e.g., "Konteks awal telah ditemukan, namun saya merasa perlu memverifikasi apakah ada pengecualian spesifik terkait UMKM...").
2. "is_sufficient": boolean (true or false). Return false if you suspect more context is needed.
ONLY output valid JSON.
"""
    # Context format
    ctx_str = "\\n\\n".join([f"DOC {k}: {v}" for k,v in summaries.items()])
    client = llm_stance.get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL),
            messages=[
                {"role": "system", "content": sys_eval}, 
                {"role": "user", "content": f"Query: {query}\\n\\nRetrieved Context:\\n{ctx_str}"}
            ],
            max_tokens=250, temperature=0.1
        )
        content = (resp.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
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
        # Default to false if error to be safe
        state["route"] = "deep"
        return state
        
    cur_doc_ids = state.get("primary_doc_ids", [])
    for d in doc_ids:
        if d not in cur_doc_ids: cur_doc_ids.append(d)
    state["primary_doc_ids"] = cur_doc_ids
    return _assemble_context_for_state(state, raw_vdb_hits=raw)

def deep_research_node(state: GraphState) -> GraphState:
    state["logs"].append("[Deep Node] Traversing Graph and VDB heavily")
    state["narratives"].append("Sistem melakukan penelusuran hukum secara ekstensif (historis dan relasi regulasi) untuk memastikan keakuratan...")
    query = state["query"]
    expanded = llm_stance.expand_query(query)
    
    raw = []
    seen = set()
    for trm in [query] + expanded[:2]:
        try:
            emb = llm_stance.get_embedding(trm)
            hits = pinecone_client.semantic_search(query_embedding=emb, top_k=25)
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
                
    doc_summaries = {}
    for did in merged_docs[:15]:
        summary = ""
        chunks = [r for r in raw if r.get("doc_id") == did]
        if chunks: summary = chunks[0].get("content", "")
        if not summary and neo4j_client.test_connection():
            try:
                dtl = neo4j_client.get_document_detail(did)
                ps = dtl.get("pasals", [])
                if ps: summary = ps[0].get("content", "")
            except Exception: pass
        doc_summaries[did] = summary[:400]
        
    ranked = llm_stance.rerank_documents(query, doc_summaries)
    top_docs = [did for did, sc in ranked if sc >= 3.0][:5]
    if not top_docs: top_docs = merged_docs[:5]
        
    cur_doc_ids = state.get("primary_doc_ids", [])
    for d in top_docs:
        if d not in cur_doc_ids: cur_doc_ids.append(d)
            
    state["primary_doc_ids"] = cur_doc_ids
    state["logs"].append(f"[Deep Node] Reranked Top {len(cur_doc_ids)} docs.")
    state["narratives"].append(f"Berhasil merangkum {len(cur_doc_ids)} dokumen paling relevan beserta hubungannya. Sedang menyusun analisa...")
    return _assemble_context_for_state(state, raw_vdb_hits=raw)

def generate_answer_node(state: GraphState) -> GraphState:
    state["logs"].append("[Answer Node] Generating output")
    chunks = _build_interleaved_context(
        primary_doc_ids=state.get("primary_doc_ids", []),
        related_doc_ids=[],
        context_docs=state.get("context_docs", {}),
        max_chunks=40, max_chars=16000
    )
    ans = llm_stance.ask_about_documents(state["query"], chunks, state.get("relationship_context", ""))
    state["answer"] = ans
    return state

def route_after_direct(state: GraphState) -> str:
    if state.get("route") == "semantic": return "semantic_search"
    return "generate_answer"
    
def route_after_semantic(state: GraphState) -> str:
    if state.get("route") == "deep": return "deep_research"
    return "generate_answer"

def create_agent():
    workflow = StateGraph(GraphState)
    workflow.add_node("router", router_node)
    workflow.add_node("direct_lookup", direct_lookup_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("deep_research", deep_research_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    workflow.set_entry_point("router")
    
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
    
    return workflow.compile()
