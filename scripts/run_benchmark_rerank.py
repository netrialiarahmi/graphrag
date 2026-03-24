#!/usr/bin/env python3
"""
Benchmark: Semantic-only V5 — V1 base + LLM Re-ranking.
Implements the Cross-Encoder concept using LLM-as-judge
for chunk relevance scoring (full query+doc attention, immune to anisotropy).

Pipeline:
  1. Hybrid BM25+Dense search (V1 params: alpha=0.4, top_k=20)
  2. Select top-5 docs (V1 logic: flat unique doc IDs)
  3. Gather chunks per doc (VDB + Neo4j)
  4. **NEW** LLM Re-rank: score each chunk 0-10 for query relevance
  5. Filter chunks with score >= 4, reorder by relevance score
  6. Send reranked context to LLM for answer generation

Usage:
    python scripts/run_benchmark_rerank.py
    python scripts/run_benchmark_rerank.py --sample GOV_TC011 GOV_TC012
"""
import argparse, csv, json, os, sys, time, re

os.environ["GRAPHRAG_STANDALONE"] = "1"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from shared import llm_stance, neo4j_client, pinecone_client
from shared.bm25_index import hybrid_search

# ── Constants ────────────────────────────────────────────────────────────
BENCHMARK_DIR = os.path.join(_PROJECT_ROOT, "benchmark")
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "output")
DEFAULT_FILE  = os.path.join(BENCHMARK_DIR, "govnetic_qa_complete_50 (business).xlsx")
DELAY         = 2

# ── V1 parameters (proven baseline) ─────────────────────────────────────
ALPHA    = 0.4
TOP_K    = 20
MAX_DOCS = 5


# ── LLM Re-ranker ───────────────────────────────────────────────────────

def _rerank_chunks_llm(query: str, chunks: list[dict], batch_size: int = 10) -> list[dict]:
    """Score chunks for relevance using LLM (cross-encoder equivalent).

    Returns chunks with added 'rerank_score' field, sorted descending.
    Only chunks with score >= 4 are returned.
    """
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
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\[[\d\s,]+\]', raw)
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


# ── Semantic pipeline (V1 base + LLM re-ranking) ────────────────────────

def run_semantic_pipeline(query: str) -> tuple[str, list[str]]:
    """V1 hybrid search + LLM re-ranking of chunks before answer generation."""

    # Step 1: Hybrid BM25+Dense search (V1 params)
    emb = llm_stance.get_embedding(query)
    raw = hybrid_search(query, emb, top_k=TOP_K, alpha=ALPHA)

    # Step 2: Select top docs (V1 flat unique logic)
    seen_docs = []
    for h in raw:
        did = h.get("doc_id", "")
        if did and did not in seen_docs:
            seen_docs.append(did)
        if len(seen_docs) >= MAX_DOCS:
            break
    doc_ids = seen_docs

    if not doc_ids:
        return "(Tidak ditemukan dokumen relevan.)", []

    # Step 3: Gather chunks per doc (same as V1)
    context_docs = {}
    seen_chunk_ids = set()
    for did in doc_ids:
        doc_chunks = [h for h in raw if h.get("doc_id") == did]
        for ch in doc_chunks:
            seen_chunk_ids.add(ch.get("id", ""))
        extra = pinecone_client.fetch_by_doc_id(did, top_k=80)
        for ch in extra:
            cid = ch.get("id", "")
            if cid not in seen_chunk_ids:
                doc_chunks.append(ch)
                seen_chunk_ids.add(cid)
        context_docs[did] = {"source": "hybrid", "chunks": doc_chunks}

    # Neo4j enrichment
    neo4j_ok = False
    try:
        neo4j_client.get_all_documents()
        neo4j_ok = True
    except Exception:
        pass

    if neo4j_ok:
        for did in doc_ids:
            try:
                detail = neo4j_client.get_document_detail(did)
                for p in detail.get("pasals", []) + detail.get("ayats", []):
                    content = p.get("content", "")
                    pid = str(p.get("name", ""))
                    nid = f"neo-{did}-{pid}"
                    if content and len(content) > 20 and nid not in seen_chunk_ids:
                        context_docs[did]["chunks"].append({
                            "id": nid, "doc_id": did,
                            "content": content, "scope": "neo4j-pasal",
                        })
                        seen_chunk_ids.add(nid)
            except Exception:
                pass

    # Build relationship context
    relationship_context = ""
    if neo4j_ok:
        try:
            edges = neo4j_client.get_edges_between(list(context_docs.keys()))
            lines = [f'- {e["source_id"]} --[{e["type"]}]--> {e["target_id"]}' for e in edges.get("edges", [])]
            if lines:
                relationship_context = "\n".join(lines)
        except Exception:
            pass

    # Step 4: Collect all candidate chunks (interleaved, V1 style)
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

    # Step 5: LLM Re-ranking (the key addition)
    print(f"          ↳ reranking {len(all_candidates)} chunks...")
    reranked = _rerank_chunks_llm(query, all_candidates)
    print(f"          ↳ {len(reranked)} chunks passed filter (score >= 4)")

    # Use top reranked chunks (cap at 30 chunks / 12000 chars)
    llm_chunks = []
    char_count = 0
    for ch in reranked:
        content = ch.get("content", "")
        char_count += len(content)
        llm_chunks.append(ch)
        if len(llm_chunks) >= 30 or char_count >= 12000:
            break

    if not llm_chunks:
        llm_chunks = all_candidates[:20]

    # Step 6: Generate answer
    answer = llm_stance.ask_about_documents(
        query, llm_chunks,
        relationship_context=relationship_context,
    )

    final_doc_ids = []
    for ch in llm_chunks:
        did = ch.get("doc_id", "")
        if did and did not in final_doc_ids:
            final_doc_ids.append(did)

    return answer, final_doc_ids


# ── Judge ────────────────────────────────────────────────────────────────

def judge_answer(query: str, generated: str, expected: str) -> str:
    client = llm_stance.get_llm_client()
    prompt = f"""Kamu adalah evaluator jawaban hukum Indonesia. Bandingkan JAWABAN SISTEM dengan JAWABAN BENAR.

PERTANYAAN:
{query}

JAWABAN SISTEM:
{generated}

JAWABAN BENAR (referensi):
{expected}

Evaluasi apakah JAWABAN SISTEM sudah benar, parsial, atau salah dibandingkan JAWABAN BENAR.
- BENAR: jawaban sistem akurat dan mencakup poin utama dari jawaban benar
- PARSIAL: jawaban sistem benar sebagian namun ada informasi penting yang kurang atau sedikit keliru
- SALAH: jawaban sistem salah atau bertolak belakang dengan jawaban benar

Balas HANYA dengan satu kata: BENAR, PARSIAL, atau SALAH."""

    try:
        resp = client.chat.completions.create(
            model=llm_stance.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16, temperature=0.0,
        )
        v = resp.choices[0].message.content.strip().upper()
        if "BENAR" in v and "SALAH" not in v:
            return "BENAR"
        elif "PARSIAL" in v:
            return "PARSIAL"
        elif "SALAH" in v:
            return "SALAH"
        return v[:20]
    except Exception as e:
        return f"ERROR: {e}"


# ── IO ───────────────────────────────────────────────────────────────────

def _parse_questions(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        questions.append({
            "no": str(row[0]),
            "question": str(row[1]) if row[1] else "",
            "expected": str(row[2]) if len(row) > 2 and row[2] else "",
        })
    wb.close()
    return questions


def _save_results(results: list[dict], base_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, f"qa_benchmark_semantic_rerank_{base_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["No", "Pertanyaan", "Jawaban Semantic", "Jawaban Benar", "Docs", "Verdict"])
        writer.writeheader()
        writer.writerows(results)
    return out_csv


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic V1 + LLM Re-ranking Benchmark")
    parser.add_argument("xlsx", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("--sample", nargs="+", help="Only run these question IDs")
    args = parser.parse_args()

    xlsx_path = args.xlsx
    if not os.path.exists(xlsx_path):
        print(f"[ERROR] File not found: {xlsx_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(xlsx_path))[0].replace(" ", "_")
    print(f"\n{'='*60}")
    print(f"  Semantic V1 + LLM Re-ranking Benchmark")
    print(f"  File: {os.path.basename(xlsx_path)}")
    print(f"{'='*60}\n")

    questions = _parse_questions(xlsx_path)

    sample_set = set(args.sample) if args.sample else None
    if sample_set:
        questions = [q for q in questions if q["no"] in sample_set]
        print(f"[INFO] Sample mode: {len(questions)} questions selected\n")
    else:
        print(f"[INFO] Loaded {len(questions)} questions\n")

    results = []
    out_csv = os.path.join(OUTPUT_DIR, f"qa_benchmark_semantic_rerank_{base_name}.csv")
    done_nos = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["Verdict"] in ("BENAR", "PARSIAL", "SALAH"):
                    done_nos.add(r["No"])
                results.append(r)
        print(f"[INFO] Resuming — {len(done_nos)} already done\n")

    for i, q in enumerate(questions):
        if q["no"] in done_nos:
            print(f"[{i+1:02d}/{len(questions)}] {q['no']} — skipped")
            continue

        print(f"[{i+1:02d}/{len(questions)}] {q['no']} — {q['question'][:80]}…")

        try:
            answer, doc_ids = run_semantic_pipeline(q["question"])
            docs_str = ", ".join(doc_ids[:7])
            print(f"          ↳ docs: {docs_str}")
        except Exception as e:
            answer = f"(Pipeline error: {e})"
            docs_str = ""
            print(f"          ↳ ERROR: {e}")

        try:
            verdict = judge_answer(q["question"], answer, q["expected"])
            print(f"          ↳ verdict: {verdict}")
        except Exception as e:
            verdict = f"ERROR: {e}"

        results.append({
            "No": q["no"],
            "Pertanyaan": q["question"],
            "Jawaban Semantic": answer,
            "Jawaban Benar": q["expected"],
            "Docs": docs_str,
            "Verdict": verdict,
        })

        _save_results(results, base_name)

        if i < len(questions) - 1:
            time.sleep(DELAY)

    out_csv = _save_results(results, base_name)

    verdicts = [r["Verdict"] for r in results]
    benar   = verdicts.count("BENAR")
    parsial = verdicts.count("PARSIAL")
    salah   = verdicts.count("SALAH")
    total   = len(results)

    print(f"\n{'='*60}")
    print(f"  HASIL BENCHMARK — Semantic V1 + LLM Re-ranking")
    print(f"{'='*60}")
    print(f"  Total  : {total}")
    print(f"  BENAR  : {benar}  ({benar/total*100:.1f}%)")
    print(f"  PARSIAL: {parsial}  ({parsial/total*100:.1f}%)")
    print(f"  SALAH  : {salah}  ({salah/total*100:.1f}%)")
    print(f"\n  Output : {out_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
