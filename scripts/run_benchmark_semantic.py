#!/usr/bin/env python3
"""
Benchmark: Semantic-only (Hybrid BM25+Dense) with query expansion.
V2 — Incorporates 5-phase improvements:
  Phase 1: Doc-ID extraction + targeted Pinecone fetch
  Phase 2: Graph-assisted retrieval (CITES/HIGHER edges)
  Phase 3: Diversity-aware doc selection + per-doc chunk cap
  Phase 4: Tuned parameters (alpha=0.5, top_k=30, max_docs=7)
  Phase 5: Improved query expansion with topic→UU mapping

Usage:
    python scripts/run_benchmark_semantic.py
    python scripts/run_benchmark_semantic.py path/to/file.xlsx
"""
import argparse, csv, os, sys, time, re

os.environ["GRAPHRAG_STANDALONE"] = "1"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from shared import llm_stance, neo4j_client, pinecone_client
from shared.bm25_index import hybrid_search, bm25_search

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK_DIR = os.path.join(_PROJECT_ROOT, "benchmark")
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "output")
DEFAULT_FILE  = os.path.join(BENCHMARK_DIR, "govnetic_qa_complete_50 (business).xlsx")
DELAY         = 2

# ── Topic → UU mapping (same as chatbot) ─────────────────────────────────────
_TOPIC_LAW_MAP = {
    "perseroan terbatas": "Undang-Undang Nomor 40 Tahun 2007 tentang Perseroan Terbatas",
    "bangunan gedung": "Undang-Undang Nomor 28 Tahun 2002 tentang Bangunan Gedung",
    "cipta kerja": "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja",
    "jasa konstruksi": "Undang-Undang Nomor 2 Tahun 2017 tentang Jasa Konstruksi",
    "ketenagakerjaan": "Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan",
    "penanaman modal": "Undang-Undang Nomor 25 Tahun 2007 tentang Penanaman Modal",
    "perbankan": "Undang-Undang Nomor 10 Tahun 1998 tentang Perbankan",
    "kepailitan": "Undang-Undang Nomor 37 Tahun 2004 tentang Kepailitan",
    "hak tanggungan": "Undang-Undang Nomor 4 Tahun 1996 tentang Hak Tanggungan",
    "lingkungan hidup": "Undang-Undang Nomor 32 Tahun 2009 tentang Perlindungan dan Pengelolaan Lingkungan Hidup",
    "umkm": "Undang-Undang Nomor 20 Tahun 2008 tentang UMKM",
    "usaha mikro": "Undang-Undang Nomor 20 Tahun 2008 tentang UMKM",
    "perdagangan": "Undang-Undang Nomor 7 Tahun 2014 tentang Perdagangan",
    "arsitek": "Undang-Undang Nomor 6 Tahun 2017 tentang Arsitek",
    "pemerintahan daerah": "Undang-Undang Nomor 23 Tahun 2014 tentang Pemerintahan Daerah",
    "bantuan hukum": "Undang-Undang Nomor 16 Tahun 2011 tentang Bantuan Hukum",
    "perumahan": "Undang-Undang Nomor 1 Tahun 2011 tentang Perumahan dan Kawasan Permukiman",
    "rumah susun": "Undang-Undang Nomor 20 Tahun 2011 tentang Rumah Susun",
}


# ── Query expansion (same as chatbot, Phase 5) ──────────────────────────────

def _expand_for_definition(query: str) -> list[str]:
    q = query.lower().strip().rstrip("?!.")
    q = re.sub(
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
    for topic, uu_ref in _TOPIC_LAW_MAP.items():
        if topic in q:
            variants.append(uu_ref)
            break
    return variants


# ── Phase 1: Doc-ID extraction ──────────────────────────────────────────────

def _extract_doc_references(query: str) -> list[str]:
    results = []
    seen = set()
    patterns = [
        (r'(?:Permen\s+PUPR|PERMEN\s+PUPR)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPUPR"),
        (r'(?:Permen\s+PPN|PERMEN\s+PPN)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERMENPPN"),
        (r'(?:Perppu|PERPPU)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERPPU"),
        (r'(?:Pergub|PERGUB)\s+(?:No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PERGUB"),
        (r'\b(?:UU|Undang-Undang)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "UU"),
        (r'\b(?:PP|Peraturan\s+Pemerintah)\s+(?:Nomor\s+|No\.?\s*)?(\d+)(?:\s+[Tt]ahun\s+|/)(\d{4})', "PP"),
    ]
    for pat, jenis in patterns:
        for m in re.finditer(pat, query, re.IGNORECASE):
            nomor = m.group(1)
            tahun = m.group(2)
            scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
            doc_id = f"{jenis}-{scope}-{nomor}-{tahun}"
            if doc_id not in seen:
                results.append(doc_id)
                seen.add(doc_id)
    return results


# ── Phase 3: Diversity-aware doc selection ───────────────────────────────────

def _get_diverse_doc_ids(hits: list[dict], max_docs: int, max_per_doc: int = 4) -> list[str]:
    best_score: dict[str, float] = {}
    for h in hits:
        did = h.get("doc_id", "")
        if not did:
            continue
        score = h.get("rrf_score", 0)
        if did not in best_score or score > best_score[did]:
            best_score[did] = score
    ranked_docs = sorted(best_score.keys(), key=lambda d: best_score[d], reverse=True)
    return ranked_docs[:max_docs]


# ── Phase 2: Graph-assisted retrieval ────────────────────────────────────────

def _follow_graph_edges(doc_ids: list[str], max_extra: int = 4) -> list[str]:
    try:
        neo4j_client.get_all_documents()
    except Exception:
        return []
    known = set(doc_ids)
    new_docs = []
    for did in doc_ids[:3]:
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


# ── Semantic-only pipeline (V4 — reserved slots) ────────────────────────────

def run_semantic_pipeline(query: str) -> tuple[str, list[str]]:
    """Run hybrid BM25+dense search with reserved-slot approach.

    V4 = V3 base + Phase 1 with reserved slots:
    - Extract explicit law refs from query → fetch chunks → boost into pool
    - Diversity ranking → 5 doc slots (FIXED)
    - Mentioned docs get priority: if not already in top-5, they REPLACE
      the lowest-ranked doc (not append beyond 5)
    """

    # Phase 1a: Extract explicit doc references from query
    mentioned_doc_ids = _extract_doc_references(query)

    # Phase 5: Multi-query with improved expansion (topic→UU mapping)
    expansion = _expand_for_definition(query)
    all_queries = [query] + expansion

    raw_all = []
    seen_ids = set()
    for q in all_queries:
        emb = llm_stance.get_embedding(q)
        hits = hybrid_search(q, emb, top_k=30, alpha=0.5)
        for h in hits:
            hid = h["id"]
            if hid not in seen_ids:
                raw_all.append(h)
                seen_ids.add(hid)

    # Phase 1b: Targeted fetch for mentioned docs — inject with high boost
    for ref_did in mentioned_doc_ids:
        try:
            ref_chunks = pinecone_client.fetch_by_doc_id(ref_did, top_k=10)
            for ch in ref_chunks:
                if ch["id"] not in seen_ids:
                    ch["rrf_score"] = 0.06  # high boost (above typical hybrid ~0.01-0.03)
                    raw_all.append(ch)
                    seen_ids.add(ch["id"])
        except Exception:
            pass

    raw_all.sort(key=lambda h: h.get("rrf_score", 0), reverse=True)

    # Phase 3a: Cap per-doc chunks to reduce attractor dominance
    max_per_doc = 4
    doc_chunk_count: dict[str, int] = {}
    raw = []
    for h in raw_all:
        did = h.get("doc_id", "")
        cnt = doc_chunk_count.get(did, 0)
        if cnt < max_per_doc:
            raw.append(h)
            doc_chunk_count[did] = cnt + 1
        if len(raw) >= 30:
            break

    # Phase 3b: Diversity-aware doc selection — FIXED at 5
    doc_ids = _get_diverse_doc_ids(raw, max_docs=5)

    # Phase 1c: Reserved slots — ensure mentioned docs are in top-5
    for md in mentioned_doc_ids:
        if md not in doc_ids:
            if len(doc_ids) >= 5:
                doc_ids.pop()  # drop lowest-ranked
            doc_ids.insert(0, md)  # insert at top priority

    if not doc_ids:
        return "(Tidak ditemukan dokumen relevan.)", []

    # Assemble context from VDB chunks only
    context_docs = {}
    seen_chunk_ids = set()
    for did in doc_ids:
        doc_chunks = [h for h in raw if h.get("doc_id") == did]
        for ch in doc_chunks:
            seen_chunk_ids.add(ch.get("id", ""))
        # Fetch more chunks from same doc
        extra = pinecone_client.fetch_by_doc_id(did, top_k=80)
        for ch in extra:
            cid = ch.get("id", "")
            if cid not in seen_chunk_ids:
                doc_chunks.append(ch)
                seen_chunk_ids.add(cid)
        context_docs[did] = {"source": "hybrid", "chunks": doc_chunks}

    # Neo4j pasal/ayat content for found docs (enrich, but no graph traversal)
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

    # Build interleaved context
    llm_chunks = []
    total_chars = 0
    doc_queues = {}
    for did in doc_ids:
        info = context_docs.get(did)
        if not info:
            continue
        chunks = list(info["chunks"])
        scored = sorted([c for c in chunks if c.get("score") is not None or c.get("rrf_score") is not None],
                        key=lambda c: c.get("rrf_score", 0) or c.get("score", 0), reverse=True)
        unscored = [c for c in chunks if c.get("score") is None and c.get("rrf_score") is None]
        doc_queues[did] = scored + unscored

    doc_keys = list(doc_queues.keys())
    idx_map = {d: 0 for d in doc_keys}
    exhausted = set()
    seen_build = set()
    while len(llm_chunks) < 40 and total_chars < 16000 and len(exhausted) < len(doc_keys):
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
            llm_chunks.append(chunk)
            if len(llm_chunks) >= 40 or total_chars >= 16000:
                break

    answer = llm_stance.ask_about_documents(
        query, llm_chunks,
        relationship_context=relationship_context,
    )
    return answer, doc_ids


# ── Judge ────────────────────────────────────────────────────────────────────

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


# ── IO ───────────────────────────────────────────────────────────────────────

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
    out_csv = os.path.join(OUTPUT_DIR, f"qa_benchmark_semantic_v4_{base_name}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["No", "Pertanyaan", "Jawaban Semantic", "Jawaban Benar", "Docs", "Verdict"])
        writer.writeheader()
        writer.writerows(results)
    return out_csv


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic-only (Hybrid BM25+Dense) Benchmark")
    parser.add_argument("xlsx", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("--sample", nargs="+", help="Only run these question IDs (e.g. GOV_TC011 GOV_TC012)")
    args = parser.parse_args()

    xlsx_path = args.xlsx
    if not os.path.exists(xlsx_path):
        print(f"[ERROR] File not found: {xlsx_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(xlsx_path))[0].replace(" ", "_")
    print(f"\n{'='*60}")
    print(f"  Semantic-Only V4 (Reserved Slots) Benchmark")
    print(f"  File: {os.path.basename(xlsx_path)}")
    print(f"{'='*60}\n")

    questions = _parse_questions(xlsx_path)

    # Filter to sample if --sample provided
    sample_set = set(args.sample) if args.sample else None
    if sample_set:
        questions = [q for q in questions if q["no"] in sample_set]
        print(f"[INFO] Sample mode: {len(questions)} questions selected\n")
    else:
        print(f"[INFO] Loaded {len(questions)} questions\n")

    results = []
    # Resume support
    out_csv = os.path.join(OUTPUT_DIR, f"qa_benchmark_semantic_v4_{base_name}.csv")
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

        # Checkpoint
        _save_results(results, base_name)

        if i < len(questions) - 1:
            time.sleep(DELAY)

    out_csv = _save_results(results, base_name)

    # Summary
    verdicts = [r["Verdict"] for r in results]
    benar   = verdicts.count("BENAR")
    parsial = verdicts.count("PARSIAL")
    salah   = verdicts.count("SALAH")
    total   = len(results)

    print(f"\n{'='*60}")
    print(f"  HASIL BENCHMARK — Semantic Only V4 (Reserved Slots)")
    print(f"{'='*60}")
    print(f"  Total  : {total}")
    print(f"  BENAR  : {benar}  ({benar/total*100:.1f}%)")
    print(f"  PARSIAL: {parsial}  ({parsial/total*100:.1f}%)")
    print(f"  SALAH  : {salah}  ({salah/total*100:.1f}%)")
    print(f"\n  Output : {out_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
