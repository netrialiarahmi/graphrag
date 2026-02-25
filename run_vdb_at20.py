#!/usr/bin/env python3
"""
Re-run VDB retrieval with VDB_MAX_DOCS=20 for the QA 100 benchmark,
then recompute metrics @K so @20 has real data.

This reads the same XLSX, re-queries Pinecone for each question to get
20 unique VDB documents (instead of 10), and produces updated CSVs.
GraphRAG is NOT re-run — only VDB column is updated.
"""

import csv
import os
import sys
import time

os.environ["GRAPHRAG_STANDALONE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from utils import llm_stance, pinecone_client
from utils.benchmark_helpers import (
    extract_documents,
    get_correct_doc_id,
    get_unique_doc_ids,
    extract_doc_ids_from_question,
    build_doc_id_aliases,
    match_with_aliases,
)

# ── Configuration ─────────────────────────────────────────────────────────────
VDB_TOP_K = 100          # Raw vectors from Pinecone
VDB_MAX_DOCS = 20        # ← NEW: 20 unique docs instead of 10

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(ROOT_DIR, "benchmark", "QA 100 (test-all-sector).xlsx")
DETAIL_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "detailed retrieval")
METRICS_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "metrics")

DETAIL_CSV = os.path.join(DETAIL_DIR, "QA 100 (test-all-sector)_v3.csv")

RESULT_COLUMNS = [
    "No", "Pertanyaan", "GT_Total", "GT_Doc_IDs", "GT_From_Question",
    "Dok_VDB", "Matched_GT_VDB", "Recall_VDB", "Precision_VDB",
]


# ── Parse questions ───────────────────────────────────────────────────────────

def _parse_questions(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        q_no = str(row[0])
        q_text = str(row[1]) if row[1] else ""
        evidence = str(row[3]) if len(row) > 3 and row[3] else ""
        parsed_docs = extract_documents(evidence)
        gt_doc_ids = set()
        all_candidates = set()
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
                "gt_doc_ids": gt_doc_ids,
                "all_candidates": all_candidates,
                "q_doc_ids": q_doc_ids,
            })
    wb.close()
    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*70}")
    print(f"  VDB Retrieval @20 — QA 100 (test-all-sector)")
    print(f"  VDB_MAX_DOCS = {VDB_MAX_DOCS}")
    print(f"{'='*70}")

    # Test Pinecone
    if not pinecone_client.test_connection():
        print("ERROR: Cannot connect to Pinecone!")
        sys.exit(1)
    print("  Pinecone connected ✓")

    # Test HF embedding (may need wake-up time)
    print("  Testing HuggingFace embedding (may take a minute to wake up)...")
    hf_ok = False
    for attempt in range(5):
        if llm_stance.test_hf_connection():
            hf_ok = True
            break
        print(f"    Attempt {attempt+1}/5 failed, retrying in 15s...")
        time.sleep(15)
    if not hf_ok:
        print("ERROR: Cannot connect to HuggingFace embedding after 5 attempts!")
        sys.exit(1)
    print("  HuggingFace embedding connected ✓")

    # Load existing CSV to preserve GraphRAG columns
    existing_rows = {}
    if os.path.exists(DETAIL_CSV):
        with open(DETAIL_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows[row["No"]] = row
        print(f"  Loaded existing CSV: {len(existing_rows)} rows")

    questions = _parse_questions(XLSX_PATH)
    print(f"  Found {len(questions)} questions\n")

    result_rows = []
    t0 = time.time()

    for idx, q in enumerate(questions):
        pct = (idx + 1) / len(questions) * 100
        short_q = q["question"][:70]
        print(f"  [{idx+1}/{len(questions)}] ({pct:.0f}%) {short_q}…", end="", flush=True)

        gt = q["gt_doc_ids"]
        q_injected = q["q_doc_ids"]

        if not gt:
            print(" SKIP (no GT)")
            result_rows.append({
                "No": q["no"], "Pertanyaan": q["question"][:200],
                "GT_Total": 0, "GT_Doc_IDs": "", "GT_From_Question": "",
                "Dok_VDB": "", "Matched_GT_VDB": "",
                "Recall_VDB": "", "Precision_VDB": "",
            })
            continue

        try:
            q_embedding = llm_stance.get_embedding(q["question"])

            # VDB retrieval — deduplicate to 20 unique docs
            vdb_raw = pinecone_client.semantic_search(
                query_embedding=q_embedding, top_k=VDB_TOP_K
            )
            vdb_doc_ids = get_unique_doc_ids(vdb_raw, VDB_MAX_DOCS)

            # Score
            scoring_ids = gt | set(vdb_doc_ids) | q["all_candidates"]
            aliases = build_doc_id_aliases(scoring_ids)
            matched_gt_vdb = match_with_aliases(set(vdb_doc_ids), gt, aliases)

            vdb_recall = len(matched_gt_vdb) / len(gt) if gt else 0
            vdb_precision = len(matched_gt_vdb) / len(vdb_doc_ids) if vdb_doc_ids else 0

            # Get existing GraphRAG data
            old = existing_rows.get(q["no"], {})

            result_rows.append({
                "No": q["no"],
                "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_VDB": ", ".join(vdb_doc_ids),
                "Dok_GraphRAG": old.get("Dok_GraphRAG", ""),
                "Matched_GT_VDB": ", ".join(sorted(matched_gt_vdb)),
                "Matched_GT_GraphRAG": old.get("Matched_GT_GraphRAG", ""),
                "Recall_VDB": f"{vdb_recall:.4f}",
                "Precision_VDB": f"{vdb_precision:.4f}",
                "Recall_GraphRAG": old.get("Recall_GraphRAG", ""),
                "Precision_GraphRAG": old.get("Precision_GraphRAG", ""),
            })
            print(f" ✓ VDB={len(vdb_doc_ids)} docs, R={vdb_recall:.2%}")

        except Exception as e:
            old = existing_rows.get(q["no"], {})
            result_rows.append({
                "No": q["no"], "Pertanyaan": q["question"][:200],
                "GT_Total": len(gt),
                "GT_Doc_IDs": ", ".join(sorted(gt)),
                "GT_From_Question": ", ".join(sorted(q_injected)) if q_injected else "",
                "Dok_VDB": f"Error: {e}",
                "Dok_GraphRAG": old.get("Dok_GraphRAG", ""),
                "Matched_GT_VDB": "",
                "Matched_GT_GraphRAG": old.get("Matched_GT_GraphRAG", ""),
                "Recall_VDB": "0", "Precision_VDB": "0",
                "Recall_GraphRAG": old.get("Recall_GraphRAG", ""),
                "Precision_GraphRAG": old.get("Precision_GraphRAG", ""),
            })
            print(f" ✗ {e}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    # Write updated detailed CSV
    os.makedirs(DETAIL_DIR, exist_ok=True)
    output_columns = [
        "No", "Pertanyaan", "GT_Total", "GT_Doc_IDs", "GT_From_Question",
        "Dok_VDB", "Dok_GraphRAG", "Matched_GT_VDB", "Matched_GT_GraphRAG",
        "Recall_VDB", "Precision_VDB", "Recall_GraphRAG", "Precision_GraphRAG",
    ]
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"  → {DETAIL_CSV}")

    # Now recompute metrics
    print(f"\n  Recomputing metrics @K...")
    from run_metrics_at_k import process_file
    process_file(DETAIL_CSV)

    print("\nAll done!")


if __name__ == "__main__":
    main()
