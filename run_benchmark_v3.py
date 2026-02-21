#!/usr/bin/env python3
"""
Run IR Benchmark v3 — improved scoring with alias matching & question injection.

Key improvements over v2:
  1. Fixed JENIS_MAP: PERMEN PUPR→PERMENPUPR, SK DIRJEN BK→SKDIRJENBK, etc.
  2. Question-text injection: parse explicit regulation refs from question text.
  3. Alias matching: cross-system doc_id equivalences (PERMENPUPR ↔ PERMEN).
  4. VDB top_k=100 for diversity (still capped at 10 unique docs).
  5. GraphRAG expanded: top_k=100, expand from ALL VDB docs (5 neighbours each), cap=20.
  6. Single canonical GT per doc (no 3-scope inflation).

Output:
  output/retrieval/detailed retrieval/{base}_v3.csv   — per-question results
  output/retrieval/metrics/{base}_v3_summary.csv      — aggregate metrics

Usage:
    python run_benchmark_v3.py                    # process ALL .xlsx in benchmark/
    python run_benchmark_v3.py path/to/file.xlsx  # single file
"""

import argparse
import csv
import glob
import os
import sys
import time

# Standalone mode — skip Streamlit imports in utils
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
VDB_TOP_K = 100          # Retrieve more from VDB for diversity
VDB_MAX_DOCS = 10        # Cap unique VDB docs
GRAPHRAG_MAX_DOCS = 20   # Cap unique GraphRAG docs (VDB + Neo4j neighbours)
NEO4J_NEIGHBOURS = 5     # Neighbours to fetch per VDB doc
EXPAND_FROM_ALL = True   # Expand from ALL VDB docs, not just top N

# ── Output dirs ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DETAIL_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "detailed retrieval")
METRICS_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "metrics")

# ── CSV column definitions ────────────────────────────────────────────────────
RESULT_COLUMNS = [
    "No",
    "Pertanyaan",
    "GT_Total",
    "GT_Doc_IDs",
    "GT_From_Question",
    "Dok_VDB",
    "Dok_GraphRAG",
    "Matched_GT_VDB",
    "Matched_GT_GraphRAG",
    "Recall_VDB",
    "Precision_VDB",
    "Recall_GraphRAG",
    "Precision_GraphRAG",
]

SUMMARY_COLUMNS = ["Metric", "Value"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_questions(xlsx_path: str) -> list[dict]:
    """Parse questions and canonical GT doc_ids from an XLSX file.

    Unlike v2, GT uses a single canonical doc_id per document (no 3-scope inflation).
    The candidates list is used only for alias-based matching at scoring time.
    """
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

        # Canonical GT doc_ids (one per document)
        gt_doc_ids: set[str] = set()
        # All candidate forms for alias building
        all_candidates: set[str] = set()

        for d in parsed_docs:
            did = get_correct_doc_id(d)
            if did:
                gt_doc_ids.add(did)
                all_candidates.add(did)
            for c in d.get("candidates", []):
                all_candidates.add(c)

        # Also extract doc_ids from question text
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


def process_xlsx(xlsx_path: str) -> tuple[list[dict], dict]:
    """Process one XLSX benchmark file with alias-aware scoring.

    Returns (result_rows, summary_dict).
    """
    basename = os.path.basename(xlsx_path)
    print(f"\n{'='*70}")
    print(f"  Processing: {basename}")
    print(f"{'='*70}")

    questions = _parse_questions(xlsx_path)
    print(f"  Found {len(questions)} questions")

    # Collect ALL unique GT + candidate doc_ids for building global alias map
    all_ids: set[str] = set()
    for q in questions:
        all_ids.update(q["gt_doc_ids"])
        all_ids.update(q["all_candidates"])

    print(f"  Total unique GT doc_ids (canonical): "
          f"{sum(len(q['gt_doc_ids']) for q in questions)} across all questions")

    # Check Neo4j connectivity for GraphRAG
    try:
        neo4j_ok = neo4j_client.test_connection()
    except Exception:
        neo4j_ok = False
    if not neo4j_ok:
        print("  [WARN] Neo4j not connected — GraphRAG = VDB-only")
    else:
        print("  Neo4j connected ✓")

    # ── Score each question ───────────────────────────────────────────────
    result_rows: list[dict] = []
    t0 = time.time()

    for idx, q in enumerate(questions):
        pct = (idx + 1) / len(questions) * 100
        short_q = q["question"][:80]
        print(f"  [{idx+1}/{len(questions)}] ({pct:.0f}%) {short_q}…", end="", flush=True)

        gt = q["gt_doc_ids"]
        q_injected = q["q_doc_ids"]

        if not gt:
            result_rows.append({
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": 0,
                "GT_Doc_IDs": "",
                "GT_From_Question": "",
                "Dok_VDB": "",
                "Dok_GraphRAG": "",
                "Matched_GT_VDB": "",
                "Matched_GT_GraphRAG": "",
                "Recall_VDB": "",
                "Precision_VDB": "",
                "Recall_GraphRAG": "",
                "Precision_GraphRAG": "",
            })
            print(" SKIP (no GT)")
            continue

        try:
            q_embedding = llm_stance.get_embedding(q["question"])

            # ── VDB retrieval (top_k=100, deduplicate to 10 unique docs) ──
            vdb_raw = pinecone_client.semantic_search(
                query_embedding=q_embedding, top_k=VDB_TOP_K
            )
            vdb_doc_ids = get_unique_doc_ids(vdb_raw, VDB_MAX_DOCS)

            # ── GraphRAG: VDB + Neo4j neighbours ─────────────────────────
            graphrag_set: dict[str, None] = {}  # ordered set via dict
            for did in vdb_doc_ids:
                graphrag_set[did] = None

            if neo4j_ok:
                expand_from = vdb_doc_ids if EXPAND_FROM_ALL else vdb_doc_ids[:5]
                for did in expand_from:
                    if len(graphrag_set) >= GRAPHRAG_MAX_DOCS:
                        break
                    try:
                        related = neo4j_client.get_related_documents(
                            did, limit=NEO4J_NEIGHBOURS
                        )
                        for rdoc in related:
                            rdid = rdoc.get("doc_id", "")
                            if rdid and rdid not in graphrag_set:
                                graphrag_set[rdid] = None
                                if len(graphrag_set) >= GRAPHRAG_MAX_DOCS:
                                    break
                    except Exception:
                        pass  # individual doc expansion failure is non-fatal

            graphrag_doc_ids = list(graphrag_set.keys())[:GRAPHRAG_MAX_DOCS]

            # ── Build aliases for this question's GT + retrieved docs ─────
            scoring_ids = gt | set(vdb_doc_ids) | set(graphrag_doc_ids) | q["all_candidates"]
            aliases = build_doc_id_aliases(scoring_ids)

            # ── Score with alias matching ─────────────────────────────────
            vdb_retrieved = set(vdb_doc_ids)
            gr_retrieved = set(graphrag_doc_ids)

            matched_gt_vdb = match_with_aliases(vdb_retrieved, gt, aliases)
            matched_gt_gr  = match_with_aliases(gr_retrieved, gt, aliases)

            vdb_recall     = len(matched_gt_vdb) / len(gt) if gt else 0
            vdb_precision  = len(matched_gt_vdb) / len(vdb_doc_ids) if vdb_doc_ids else 0
            gr_recall      = len(matched_gt_gr) / len(gt) if gt else 0
            gr_precision   = len(matched_gt_gr) / len(graphrag_doc_ids) if graphrag_doc_ids else 0

            result_rows.append({
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_VDB": ", ".join(vdb_doc_ids),
                "Dok_GraphRAG": ", ".join(graphrag_doc_ids),
                "Matched_GT_VDB": ", ".join(sorted(matched_gt_vdb)),
                "Matched_GT_GraphRAG": ", ".join(sorted(matched_gt_gr)),
                "Recall_VDB": f"{vdb_recall:.4f}",
                "Precision_VDB": f"{vdb_precision:.4f}",
                "Recall_GraphRAG": f"{gr_recall:.4f}",
                "Precision_GraphRAG": f"{gr_precision:.4f}",
            })
            print(f" ✓ R_GR={gr_recall:.2%} R_VDB={vdb_recall:.2%}  (GT={len(gt)}, GR={len(graphrag_doc_ids)})")

        except Exception as e:
            result_rows.append({
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_VDB": f"Error: {e}",
                "Dok_GraphRAG": "",
                "Matched_GT_VDB": "",
                "Matched_GT_GraphRAG": "",
                "Recall_VDB": "0",
                "Precision_VDB": "0",
                "Recall_GraphRAG": "0",
                "Precision_GraphRAG": "0",
            })
            print(f" ✗ {e}")

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────
    scored_rows = [r for r in result_rows if r["GT_Total"] and r["Recall_GraphRAG"] not in ("",)]
    n_scored = len(scored_rows)
    n_skipped = len(result_rows) - n_scored

    def _avg(key):
        vals = [float(r[key]) for r in scored_rows if r[key] not in ("", "—")]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "Total_Questions":        str(len(questions)),
        "Scored_Questions":       str(n_scored),
        "Skipped_Questions":      str(n_skipped),
        "VDB_TOP_K":              str(VDB_TOP_K),
        "VDB_MAX_DOCS":           str(VDB_MAX_DOCS),
        "GRAPHRAG_MAX_DOCS":      str(GRAPHRAG_MAX_DOCS),
        "NEO4J_NEIGHBOURS":       str(NEO4J_NEIGHBOURS),
        "Avg_Recall_VDB":         f"{_avg('Recall_VDB'):.4f}",
        "Avg_Precision_VDB":      f"{_avg('Precision_VDB'):.4f}",
        "Avg_Recall_GraphRAG":    f"{_avg('Recall_GraphRAG'):.4f}",
        "Avg_Precision_GraphRAG": f"{_avg('Precision_GraphRAG'):.4f}",
        "Elapsed_Seconds":        f"{elapsed:.1f}",
    }

    print(f"\n  Summary ({n_scored} scored, {n_skipped} skipped)")
    print(f"  Avg Recall  VDB      : {summary['Avg_Recall_VDB']}")
    print(f"  Avg Precision VDB    : {summary['Avg_Precision_VDB']}")
    print(f"  Avg Recall  GraphRAG : {summary['Avg_Recall_GraphRAG']}")
    print(f"  Avg Precision GraphRAG: {summary['Avg_Precision_GraphRAG']}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return result_rows, summary


def write_results(xlsx_path: str, rows: list[dict], summary: dict):
    """Write detailed results CSV and metrics summary CSV."""
    os.makedirs(DETAIL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(xlsx_path))[0]

    # ── Detailed per-question results ─────────────────────────────────────
    detail_path = os.path.join(DETAIL_DIR, f"{base}_v3.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {detail_path}")

    # ── Metrics summary ───────────────────────────────────────────────────
    metrics_path = os.path.join(METRICS_DIR, f"{base}_v3_summary.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for k, v in summary.items():
            writer.writerow({"Metric": k, "Value": v})
    print(f"  → {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Run IR benchmark v3 (alias-aware)")
    parser.add_argument("xlsx", nargs="?", default=None, help="Path to a single .xlsx file")
    args = parser.parse_args()

    if args.xlsx:
        files = [args.xlsx]
    else:
        benchmark_dir = os.path.join(ROOT_DIR, "benchmark")
        files = sorted(glob.glob(os.path.join(benchmark_dir, "*.xlsx")))
        if not files:
            print("No .xlsx files found in benchmark/")
            sys.exit(1)

    print(f"Will process {len(files)} file(s)")
    print(f"  VDB_TOP_K={VDB_TOP_K}, VDB_MAX={VDB_MAX_DOCS}, GRAPHRAG_MAX={GRAPHRAG_MAX_DOCS}")
    print(f"  NEO4J_NEIGHBOURS={NEO4J_NEIGHBOURS}, EXPAND_FROM_ALL={EXPAND_FROM_ALL}")

    for xlsx_path in files:
        rows, summary = process_xlsx(xlsx_path)
        write_results(xlsx_path, rows, summary)

    print(f"\n{'='*70}")
    print("All done!")
    print(f"  Detailed results: {DETAIL_DIR}")
    print(f"  Metrics:          {METRICS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
