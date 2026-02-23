#!/usr/bin/env python3
"""
Run IR Benchmark v4 — Full GraphRAG Pipeline (A–F).

Mirrors the production retrieval pipeline from app.py, including:
  A. Regex extraction + LLM query expansion
  B. LLM smart catalog lookup (scan full Neo4j catalog)
  C. VDB semantic search (query + expanded terms)
  D. Priority merge: Graph picks > Regex > VDB
  E. 2-hop graph traversal (CITES|HIGHER)
  F. LLM re-ranking (score 0–10, keep ≥ 3)

Also runs v3-style (VDB-only + VDB+Neo4j neighbours) for side-by-side comparison.

Scores all three methods with:
  Recall@K, Precision@K, F1@K, HitRate@K, AP@K, NDCG@K, MRR
for cutoffs K ∈ {5, 10, 20}.

Output:
  output/retrieval/detailed retrieval/{base}_v4.csv    — per-question results
  output/retrieval/metrics/{base}_v4_summary.csv       — aggregate metrics

Usage:
    python run_benchmark_v4.py                    # process ALL .xlsx in benchmark/
    python run_benchmark_v4.py path/to/file.xlsx  # single file
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

os.environ["GRAPHRAG_STANDALONE"] = "1"

import openpyxl
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from utils import llm_stance, neo4j_client, pinecone_client
from utils.benchmark_helpers import (
    extract_documents,
    get_correct_doc_id,
    get_unique_doc_ids,
    extract_doc_ids_from_question,
    build_doc_id_aliases,
    match_with_aliases,
)

# ── Tuning constants ──────────────────────────────────────────────────────────
VDB_TOP_K = 100            # Pinecone top_k per query
VDB_MAX_DOCS = 10          # Cap unique VDB docs
EXPANDED_TERM_TOP_K = 30   # Pinecone top_k per expanded term
MAX_EXPANDED_TERMS = 3     # How many expanded terms to search
GRAPH_TRAVERSE_HOPS = 2    # Hop depth for Phase E
RERANK_MIN_SCORE = 3.0     # Min LLM re-rank score to keep
RERANK_MAX_DOCS = 7        # Max primary docs after re-ranking
FINAL_MAX_DOCS = 20        # Overall cap on final doc list

CUTOFFS = [5, 10, 20]

# ── Output dirs ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DETAIL_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "detailed retrieval")
METRICS_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "metrics")


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _is_relevant(doc_id: str, gt: set[str], aliases: dict[str, set[str]]) -> bool:
    if doc_id in gt:
        return True
    doc_aliases = aliases.get(doc_id, {doc_id})
    return bool(doc_aliases & gt)


def _matched_gt_at_k(ranked: list[str], gt: set[str],
                      aliases: dict[str, set[str]], k: int) -> set[str]:
    matched = set()
    top_k = ranked[:k]
    for gt_id in gt:
        if gt_id in top_k:
            matched.add(gt_id)
            continue
        gt_aliases = aliases.get(gt_id, {gt_id})
        if set(top_k) & gt_aliases:
            matched.add(gt_id)
    return matched


def recall_at_k(ranked, gt, aliases, k):
    if not gt:
        return 0.0
    return len(_matched_gt_at_k(ranked, gt, aliases, k)) / len(gt)


def precision_at_k(ranked, gt, aliases, k):
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if _is_relevant(d, gt, aliases)) / len(top_k)


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate_at_k(ranked, gt, aliases, k):
    return 1.0 if _matched_gt_at_k(ranked, gt, aliases, k) else 0.0


def reciprocal_rank(ranked, gt, aliases):
    for i, doc in enumerate(ranked):
        if _is_relevant(doc, gt, aliases):
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(ranked, gt, aliases, k):
    if not gt:
        return 0.0
    top_k = ranked[:k]
    n_rel = 0
    sum_p = 0.0
    for i, doc in enumerate(top_k):
        if _is_relevant(doc, gt, aliases):
            n_rel += 1
            sum_p += n_rel / (i + 1)
    norm = min(k, len(gt))
    return sum_p / norm if norm > 0 else 0.0


def ndcg_at_k(ranked, gt, aliases, k):
    if not gt:
        return 0.0
    top_k = ranked[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top_k)
              if _is_relevant(d, gt, aliases))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / idcg if idcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# XLSX PARSING (same as v3)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_questions(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    questions: list[dict] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        q_no = str(row[0])
        q_text = str(row[1]) if row[1] else ""
        evidence = str(row[3]) if len(row) > 3 and row[3] else ""
        parsed_docs = extract_documents(evidence)
        gt_doc_ids: set[str] = set()
        all_candidates: set[str] = set()
        for d in parsed_docs:
            did = get_correct_doc_id(d)
            if did:
                gt_doc_ids.add(did)
                all_candidates.add(did)
            for c in d.get("candidates", []):
                all_candidates.add(c)
        q_doc_ids = extract_doc_ids_from_question(q_text)
        gt_doc_ids.update(q_doc_ids)
        all_candidates.update(q_doc_ids)
        if q_text:
            questions.append({
                "no": q_no,
                "question": q_text,
                "evidence_raw": evidence,
                "gt_doc_ids": gt_doc_ids,
                "all_candidates": all_candidates,
                "q_doc_ids": q_doc_ids,
            })
    wb.close()
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Phases A through F
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(question: str, neo4j_ok: bool) -> list[str]:
    """Run the full GraphRAG retrieval pipeline and return ranked doc_ids.

    Phases:
      A — Regex extraction + LLM query expansion
      B — LLM smart catalog lookup (scan full Neo4j document catalog)
      C — VDB semantic search (main query + expanded terms)
      D — Priority merge: Graph > Regex > VDB
      E — 2-hop graph traversal (CITES|HIGHER) from top merged docs
      F — LLM re-ranking of all candidates
    """

    # ── Phase A: Query Analysis ──────────────────────────────────────────
    regex_doc_ids = extract_doc_ids_from_question(question)

    try:
        expanded_terms = llm_stance.expand_query(question)
    except Exception:
        expanded_terms = []

    # ── Phase B: LLM Catalog Lookup ──────────────────────────────────────
    graph_doc_ids: list[str] = []
    if neo4j_ok:
        try:
            all_neo4j_docs = neo4j_client.get_all_documents()
            graph_doc_ids = llm_stance.smart_doc_lookup(question, all_neo4j_docs)
        except Exception:
            pass

    # ── Phase C: VDB Semantic Search ─────────────────────────────────────
    query_embedding = llm_stance.get_embedding(question)
    raw_results = pinecone_client.semantic_search(
        query_embedding=query_embedding, top_k=VDB_TOP_K,
    )

    # Search with expanded terms
    seen_ids: set[str] = {h.get("id", "") for h in raw_results}
    for term in expanded_terms[:MAX_EXPANDED_TERMS]:
        try:
            term_emb = llm_stance.get_embedding(term)
            term_results = pinecone_client.semantic_search(
                query_embedding=term_emb, top_k=EXPANDED_TERM_TOP_K,
            )
            for hit in term_results:
                hid = hit.get("id", "")
                if hid and hid not in seen_ids:
                    raw_results.append(hit)
                    seen_ids.add(hid)
        except Exception:
            pass

    vdb_doc_ids = get_unique_doc_ids(raw_results, VDB_MAX_DOCS)

    # ── Phase D: Priority Merge ──────────────────────────────────────────
    merged: list[str] = []
    added: set[str] = set()

    for did in graph_doc_ids:          # LLM catalog picks (highest priority)
        if did not in added:
            merged.append(did)
            added.add(did)
    for did in regex_doc_ids:          # Regex-extracted
        if did not in added:
            merged.append(did)
            added.add(did)
    for did in vdb_doc_ids:            # VDB semantic
        if did not in added:
            merged.append(did)
            added.add(did)

    if not merged:
        return []

    # ── Phase E: 2-hop Graph Traversal ───────────────────────────────────
    graph_expanded_ids: list[str] = []
    if neo4j_ok:
        for did in merged[:5]:
            try:
                subgraph = neo4j_client.get_citing_documents(
                    did, hops=GRAPH_TRAVERSE_HOPS,
                )
                for node in subgraph.get("nodes", []):
                    ndid = node.get("doc_id", "")
                    if ndid and ndid not in added:
                        graph_expanded_ids.append(ndid)
                        added.add(ndid)
            except Exception:
                pass

    # ── Phase F: LLM Re-ranking ──────────────────────────────────────────
    all_candidate_ids = merged + graph_expanded_ids

    # Build summaries for re-ranking
    semantic_chunks_by_doc: dict[str, list[dict]] = {}
    for hit in raw_results:
        did = hit.get("doc_id", "")
        if did:
            semantic_chunks_by_doc.setdefault(did, []).append(hit)

    doc_summaries: dict[str, str] = {}
    for did in all_candidate_ids:
        summary = ""
        if did in semantic_chunks_by_doc:
            summary = semantic_chunks_by_doc[did][0].get("content", "")
        if not summary:
            try:
                vdb_chunks = pinecone_client.fetch_by_doc_id(did, top_k=3)
                if vdb_chunks:
                    summary = vdb_chunks[0].get("content", "")
            except Exception:
                pass
        if not summary and neo4j_ok:
            try:
                detail = neo4j_client.get_document_detail(did)
                pasals = detail.get("pasals", [])
                if pasals:
                    texts = [p.get("content", "") or p.get("name", "")
                             for p in pasals[:3]]
                    summary = " ".join(t for t in texts if t)
            except Exception:
                pass
        doc_summaries[did] = summary[:500] if summary else ""

    try:
        ranked = llm_stance.rerank_documents(question, doc_summaries)
    except Exception:
        ranked = [(did, 5.0) for did in all_candidate_ids]

    # Keep docs with score >= threshold, max RERANK_MAX_DOCS
    primary_doc_ids = [did for did, score in ranked if score >= RERANK_MIN_SCORE]
    primary_doc_ids = primary_doc_ids[:RERANK_MAX_DOCS]

    # Always include graph + regex picks
    for did in list(graph_doc_ids[:5]) + list(regex_doc_ids):
        if did not in primary_doc_ids:
            primary_doc_ids.append(did)

    if not primary_doc_ids:
        primary_doc_ids = merged[:5]

    return primary_doc_ids[:FINAL_MAX_DOCS]


# ══════════════════════════════════════════════════════════════════════════════
# V3-STYLE PIPELINE (for comparison)
# ══════════════════════════════════════════════════════════════════════════════

def run_v3_pipeline(question: str, neo4j_ok: bool) -> tuple[list[str], list[str]]:
    """Run v3-style retrieval: VDB-only + VDB+Neo4j neighbours.

    Returns (vdb_doc_ids, graphrag_v3_doc_ids).
    """
    query_embedding = llm_stance.get_embedding(question)
    vdb_raw = pinecone_client.semantic_search(
        query_embedding=query_embedding, top_k=VDB_TOP_K,
    )
    vdb_doc_ids = get_unique_doc_ids(vdb_raw, VDB_MAX_DOCS)

    graphrag_set: dict[str, None] = {}
    for did in vdb_doc_ids:
        graphrag_set[did] = None

    if neo4j_ok:
        for did in vdb_doc_ids:
            if len(graphrag_set) >= 20:
                break
            try:
                related = neo4j_client.get_related_documents(did, limit=5)
                for rdoc in related:
                    rdid = rdoc.get("doc_id", "")
                    if rdid and rdid not in graphrag_set:
                        graphrag_set[rdid] = None
                        if len(graphrag_set) >= 20:
                            break
            except Exception:
                pass

    graphrag_v3_ids = list(graphrag_set.keys())[:20]
    return vdb_doc_ids, graphrag_v3_ids


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_ranked(ranked: list[str], gt: set[str], aliases: dict[str, set[str]]) -> dict:
    metrics = {}
    metrics["MRR"] = reciprocal_rank(ranked, gt, aliases)
    for k in CUTOFFS:
        metrics[f"Recall@{k}"] = recall_at_k(ranked, gt, aliases, k)
        metrics[f"Precision@{k}"] = precision_at_k(ranked, gt, aliases, k)
        metrics[f"F1@{k}"] = f1_score(
            metrics[f"Precision@{k}"], metrics[f"Recall@{k}"]
        )
        metrics[f"HitRate@{k}"] = hit_rate_at_k(ranked, gt, aliases, k)
        metrics[f"AP@{k}"] = average_precision_at_k(ranked, gt, aliases, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(ranked, gt, aliases, k)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ["MRR"]
for _k in CUTOFFS:
    for _m in ["Recall", "Precision", "F1", "HitRate", "AP", "NDCG"]:
        _METRIC_KEYS.append(f"{_m}@{_k}")

DETAIL_COLUMNS = (
    ["No", "Pertanyaan", "GT_Total", "GT_Doc_IDs", "GT_From_Question",
     "Dok_Full_GraphRAG", "Dok_VDB", "Dok_GraphRAG_v3",
     "Matched_GT_Full", "Matched_GT_VDB", "Matched_GT_GRv3"]
    + [f"Full_{mk}" for mk in _METRIC_KEYS]
    + [f"VDB_{mk}" for mk in _METRIC_KEYS]
    + [f"GRv3_{mk}" for mk in _METRIC_KEYS]
)


def process_xlsx(xlsx_path: str) -> tuple[list[dict], dict]:
    basename = os.path.basename(xlsx_path)
    print(f"\n{'='*78}")
    print(f"  Processing: {basename}")
    print(f"{'='*78}")

    questions = _parse_questions(xlsx_path)
    print(f"  Found {len(questions)} questions")

    try:
        neo4j_ok = neo4j_client.test_connection()
    except Exception:
        neo4j_ok = False
    print(f"  Neo4j: {'connected ✓' if neo4j_ok else 'OFFLINE ✗'}")

    result_rows: list[dict] = []
    t0 = time.time()

    for idx, q in enumerate(questions):
        pct = (idx + 1) / len(questions) * 100
        short_q = q["question"][:70]
        print(f"\n  [{idx+1}/{len(questions)}] ({pct:.0f}%) {short_q}…", flush=True)

        gt = q["gt_doc_ids"]
        q_injected = q["q_doc_ids"]

        if not gt:
            row = {c: "" for c in DETAIL_COLUMNS}
            row.update({"No": q["no"], "Pertanyaan": q["question"][:200], "GT_Total": 0})
            result_rows.append(row)
            print("    SKIP (no GT)")
            continue

        try:
            # ── Run full pipeline (A–F) ──────────────────────────────────
            full_ranked = run_full_pipeline(q["question"], neo4j_ok)
            print(f"    Full pipeline: {len(full_ranked)} docs → {full_ranked[:5]}")

            # ── Run v3 pipeline for comparison ───────────────────────────
            vdb_ranked, grv3_ranked = run_v3_pipeline(q["question"], neo4j_ok)
            print(f"    V3: VDB={len(vdb_ranked)}, GRv3={len(grv3_ranked)}")

            # ── Build aliases ────────────────────────────────────────────
            all_ids = (
                gt | set(full_ranked) | set(vdb_ranked)
                | set(grv3_ranked) | q["all_candidates"]
            )
            aliases = build_doc_id_aliases(all_ids)

            # ── Score all three methods ──────────────────────────────────
            full_metrics = score_ranked(full_ranked, gt, aliases)
            vdb_metrics = score_ranked(vdb_ranked, gt, aliases)
            grv3_metrics = score_ranked(grv3_ranked, gt, aliases)

            # Matched GT (for display)
            matched_full = _matched_gt_at_k(full_ranked, gt, aliases, FINAL_MAX_DOCS)
            matched_vdb = _matched_gt_at_k(vdb_ranked, gt, aliases, VDB_MAX_DOCS)
            matched_grv3 = _matched_gt_at_k(grv3_ranked, gt, aliases, 20)

            row = {
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_Full_GraphRAG": ", ".join(full_ranked),
                "Dok_VDB": ", ".join(vdb_ranked),
                "Dok_GraphRAG_v3": ", ".join(grv3_ranked),
                "Matched_GT_Full": ", ".join(sorted(matched_full)),
                "Matched_GT_VDB": ", ".join(sorted(matched_vdb)),
                "Matched_GT_GRv3": ", ".join(sorted(matched_grv3)),
            }
            for mk in _METRIC_KEYS:
                row[f"Full_{mk}"] = f"{full_metrics[mk]:.4f}"
                row[f"VDB_{mk}"] = f"{vdb_metrics[mk]:.4f}"
                row[f"GRv3_{mk}"] = f"{grv3_metrics[mk]:.4f}"

            result_rows.append(row)

            print(f"    Full  R@10={full_metrics['Recall@10']:.0%} "
                  f"MRR={full_metrics['MRR']:.3f} "
                  f"NDCG@10={full_metrics['NDCG@10']:.3f}")
            print(f"    VDB   R@10={vdb_metrics['Recall@10']:.0%} "
                  f"MRR={vdb_metrics['MRR']:.3f} "
                  f"NDCG@10={vdb_metrics['NDCG@10']:.3f}")
            print(f"    GRv3  R@10={grv3_metrics['Recall@10']:.0%} "
                  f"MRR={grv3_metrics['MRR']:.3f} "
                  f"NDCG@10={grv3_metrics['NDCG@10']:.3f}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback; traceback.print_exc()
            row = {c: "" for c in DETAIL_COLUMNS}
            row.update({
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_Full_GraphRAG": f"Error: {e}",
            })
            for prefix in ("Full_", "VDB_", "GRv3_"):
                for mk in _METRIC_KEYS:
                    row[f"{prefix}{mk}"] = "0.0000"
            result_rows.append(row)

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────
    scored_rows = [r for r in result_rows
                   if r.get("GT_Total")
                   and not str(r.get("Dok_Full_GraphRAG", "")).startswith("Error")]
    n_scored = len(scored_rows)

    summary: dict = {
        "Total_Questions": len(questions),
        "Scored_Questions": n_scored,
        "Elapsed_Seconds": f"{elapsed:.1f}",
    }
    for prefix in ("Full_", "VDB_", "GRv3_"):
        for mk in _METRIC_KEYS:
            col = f"{prefix}{mk}"
            vals = [float(r[col]) for r in scored_rows if r.get(col)]
            summary[f"Avg_{col}"] = sum(vals) / len(vals) if vals else 0.0

    return result_rows, summary


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def write_results(xlsx_path: str, rows: list[dict], summary: dict):
    os.makedirs(DETAIL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(xlsx_path))[0]

    # Detailed CSV
    detail_path = os.path.join(DETAIL_DIR, f"{base}_v4.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {detail_path}")

    # Summary CSV — side-by-side comparison table
    summary_path = os.path.join(METRICS_DIR, f"{base}_v4_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        cols = ["Metric", "Full_GraphRAG", "VDB", "GRv3",
                "Δ_Full_vs_VDB", "Δ_Full_vs_GRv3"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow({
            "Metric": "Scored_Questions",
            "Full_GraphRAG": summary["Scored_Questions"],
            "VDB": summary["Scored_Questions"],
            "GRv3": summary["Scored_Questions"],
            "Δ_Full_vs_VDB": "", "Δ_Full_vs_GRv3": "",
        })
        for mk in _METRIC_KEYS:
            fv = summary.get(f"Avg_Full_{mk}", 0)
            vv = summary.get(f"Avg_VDB_{mk}", 0)
            gv = summary.get(f"Avg_GRv3_{mk}", 0)
            writer.writerow({
                "Metric": mk,
                "Full_GraphRAG": f"{fv:.4f}",
                "VDB": f"{vv:.4f}",
                "GRv3": f"{gv:.4f}",
                "Δ_Full_vs_VDB": f"{fv - vv:+.4f}",
                "Δ_Full_vs_GRv3": f"{fv - gv:+.4f}",
            })
    print(f"  → {summary_path}")


def print_summary(name: str, summary: dict):
    n = summary["Scored_Questions"]
    print(f"\n{'='*90}")
    print(f"  {name}  ({n} scored questions)")
    print(f"{'='*90}")
    print(f"  {'Metric':<16} {'Full GR':>10} {'VDB':>10} {'GRv3':>10} "
          f"{'Δ F-VDB':>10} {'Δ F-GRv3':>10}")
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for mk in _METRIC_KEYS:
        fv = summary.get(f"Avg_Full_{mk}", 0)
        vv = summary.get(f"Avg_VDB_{mk}", 0)
        gv = summary.get(f"Avg_GRv3_{mk}", 0)
        print(f"  {mk:<16} {fv:>10.4f} {vv:>10.4f} {gv:>10.4f} "
              f"{fv - vv:>+10.4f} {fv - gv:>+10.4f}")
    print(f"{'='*90}")
    print(f"  Elapsed: {summary['Elapsed_Seconds']}s\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run IR benchmark v4 (full pipeline)")
    parser.add_argument("xlsx", nargs="?", default=None,
                        help="Path to a single .xlsx file")
    args = parser.parse_args()

    if args.xlsx:
        files = [args.xlsx]
    else:
        benchmark_dir = os.path.join(ROOT_DIR, "benchmark")
        files = sorted(glob.glob(os.path.join(benchmark_dir, "*.xlsx")))
        if not files:
            print("No .xlsx files found in benchmark/")
            sys.exit(1)

    print(f"╔{'═'*76}╗")
    print(f"║  IR Benchmark v4 — Full GraphRAG Pipeline (Phases A–F){' '*21}║")
    print(f"╚{'═'*76}╝")
    print(f"  Files: {len(files)}, Cutoffs: K={CUTOFFS}")
    print(f"  VDB_TOP_K={VDB_TOP_K}, VDB_MAX={VDB_MAX_DOCS}")
    print(f"  Expanded terms: max {MAX_EXPANDED_TERMS} @ top_k={EXPANDED_TERM_TOP_K}")
    print(f"  Graph hops: {GRAPH_TRAVERSE_HOPS}")
    print(f"  Re-rank: min_score={RERANK_MIN_SCORE}, max_docs={RERANK_MAX_DOCS}")
    print(f"  Final cap: {FINAL_MAX_DOCS} docs")

    for xlsx_path in files:
        rows, summary = process_xlsx(xlsx_path)
        write_results(xlsx_path, rows, summary)
        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        print_summary(base, summary)

    print("All done!")


if __name__ == "__main__":
    main()
