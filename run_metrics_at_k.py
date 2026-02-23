#!/usr/bin/env python3
"""
Compute comprehensive retrieval metrics @K from existing benchmark CSV results.

Reads per-question CSVs produced by run_benchmark_v3.py and computes:
  • Recall@K, Precision@K, F1@K
  • Hit Rate@K (Success@K)
  • MRR (Mean Reciprocal Rank)
  • MAP@K (Mean Average Precision)
  • NDCG@K (Normalized Discounted Cumulative Gain)

For cutoffs K ∈ {5, 10, 20}, applied to both VDB and GraphRAG retrieval.

Output:
  output/retrieval/metrics/{base}_metrics_at_k_detail.csv   — per-question
  output/retrieval/metrics/{base}_metrics_at_k_summary.csv  — aggregated

Usage:
    python run_metrics_at_k.py                          # process all *_v3.csv
    python run_metrics_at_k.py path/to/detailed.csv     # single file
"""

import argparse
import csv
import glob
import math
import os
import sys
from typing import Optional

# Reuse alias logic from benchmark pipeline
os.environ["GRAPHRAG_STANDALONE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.benchmark_helpers import build_doc_id_aliases

# ── Configuration ─────────────────────────────────────────────────────────────
CUTOFFS = [5, 10, 20]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DETAIL_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "detailed retrieval")
METRICS_DIR = os.path.join(ROOT_DIR, "output", "retrieval", "metrics")


# ── Alias-aware relevance check ──────────────────────────────────────────────

def _is_relevant(doc_id: str, gt: set[str], aliases: dict[str, set[str]]) -> bool:
    """Check if a single doc_id matches any GT doc (via aliases)."""
    if doc_id in gt:
        return True
    # Check if doc_id is an alias for any GT doc
    doc_aliases = aliases.get(doc_id, {doc_id})
    return bool(doc_aliases & gt)


def _matched_gt_at_k(ranked: list[str], gt: set[str],
                      aliases: dict[str, set[str]], k: int) -> set[str]:
    """Return the set of GT doc_ids that have a match in ranked[:k]."""
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


# ── Metric functions ─────────────────────────────────────────────────────────

def recall_at_k(ranked: list[str], gt: set[str],
                aliases: dict[str, set[str]], k: int) -> float:
    """Fraction of GT documents found in top-K retrieved."""
    if not gt:
        return 0.0
    matched = _matched_gt_at_k(ranked, gt, aliases, k)
    return len(matched) / len(gt)


def precision_at_k(ranked: list[str], gt: set[str],
                   aliases: dict[str, set[str]], k: int) -> float:
    """Fraction of top-K retrieved documents that are relevant."""
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    n_relevant = sum(1 for d in top_k if _is_relevant(d, gt, aliases))
    return n_relevant / len(top_k)


def f1_at_k(p: float, r: float) -> float:
    """Harmonic mean of precision and recall."""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(ranked: list[str], gt: set[str],
                  aliases: dict[str, set[str]], k: int) -> float:
    """1.0 if at least one GT document is found in top-K, else 0.0."""
    matched = _matched_gt_at_k(ranked, gt, aliases, k)
    return 1.0 if matched else 0.0


def reciprocal_rank(ranked: list[str], gt: set[str],
                    aliases: dict[str, set[str]]) -> float:
    """1 / rank of the first relevant document. 0 if none found."""
    for i, doc in enumerate(ranked):
        if _is_relevant(doc, gt, aliases):
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(ranked: list[str], gt: set[str],
                           aliases: dict[str, set[str]], k: int) -> float:
    """Average Precision at K — TREC-style.

    AP@K = (1/min(K, |GT|)) * Σ_{i=1}^{K} P@i * rel(i)
    """
    if not gt:
        return 0.0
    top_k = ranked[:k]
    n_relevant_so_far = 0
    sum_precision = 0.0
    for i, doc in enumerate(top_k):
        if _is_relevant(doc, gt, aliases):
            n_relevant_so_far += 1
            sum_precision += n_relevant_so_far / (i + 1)
    # Normalise by min(K, |GT|) — the maximum possible relevant docs in top-K
    normaliser = min(k, len(gt))
    return sum_precision / normaliser if normaliser > 0 else 0.0


def ndcg_at_k(ranked: list[str], gt: set[str],
              aliases: dict[str, set[str]], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K (binary relevance).

    DCG@K  = Σ_{i=1}^{K} rel(i) / log₂(i+1)
    IDCG@K = Σ_{i=1}^{min(K,|GT|)} 1 / log₂(i+1)
    NDCG@K = DCG@K / IDCG@K
    """
    if not gt:
        return 0.0
    top_k = ranked[:k]

    # DCG
    dcg = 0.0
    for i, doc in enumerate(top_k):
        if _is_relevant(doc, gt, aliases):
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log₂(rank+1), rank=i+1

    # IDCG — ideal: all GT docs at the top
    ideal_count = min(k, len(gt))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


# ── CSV parsing ──────────────────────────────────────────────────────────────

def _parse_doc_list(cell: str) -> list[str]:
    """Parse comma-separated doc_ids from a CSV cell, preserving order."""
    if not cell or cell.startswith("Error"):
        return []
    return [d.strip() for d in cell.split(",") if d.strip()]


def _parse_doc_set(cell: str) -> set[str]:
    """Parse comma-separated doc_ids into a set."""
    return set(_parse_doc_list(cell))


def load_benchmark_csv(csv_path: str) -> list[dict]:
    """Load per-question benchmark CSV into list of dicts."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Core computation ─────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> tuple[list[dict], dict]:
    """Compute all metrics @K for each question and aggregate.

    Returns (detail_rows, summary_dict).
    """
    detail_rows: list[dict] = []
    # Accumulators for macro-averaging — keyed by metric name
    accum: dict[str, list[float]] = {}

    for row in rows:
        gt = _parse_doc_set(row.get("GT_Doc_IDs", ""))
        vdb_ranked = _parse_doc_list(row.get("Dok_VDB", ""))
        gr_ranked = _parse_doc_list(row.get("Dok_GraphRAG", ""))

        # Skip rows with no GT or error rows
        if not gt or row.get("Dok_VDB", "").startswith("Error"):
            continue

        # Build alias map from all IDs in play
        all_ids = gt | set(vdb_ranked) | set(gr_ranked)
        # Also include GT_From_Question if present
        q_ids = _parse_doc_set(row.get("GT_From_Question", ""))
        all_ids |= q_ids
        aliases = build_doc_id_aliases(all_ids)

        detail = {
            "No": row.get("No", ""),
            "Pertanyaan": row.get("Pertanyaan", "")[:150],
            "GT_Total": len(gt),
        }

        # MRR (no cutoff)
        mrr_vdb = reciprocal_rank(vdb_ranked, gt, aliases)
        mrr_gr = reciprocal_rank(gr_ranked, gt, aliases)
        detail["MRR_VDB"] = f"{mrr_vdb:.4f}"
        detail["MRR_GraphRAG"] = f"{mrr_gr:.4f}"
        accum.setdefault("MRR_VDB", []).append(mrr_vdb)
        accum.setdefault("MRR_GraphRAG", []).append(mrr_gr)

        # Metrics per cutoff
        for k in CUTOFFS:
            # VDB
            r_vdb = recall_at_k(vdb_ranked, gt, aliases, k)
            p_vdb = precision_at_k(vdb_ranked, gt, aliases, k)
            f_vdb = f1_at_k(p_vdb, r_vdb)
            h_vdb = hit_rate_at_k(vdb_ranked, gt, aliases, k)
            ap_vdb = average_precision_at_k(vdb_ranked, gt, aliases, k)
            ndcg_vdb = ndcg_at_k(vdb_ranked, gt, aliases, k)

            # GraphRAG
            r_gr = recall_at_k(gr_ranked, gt, aliases, k)
            p_gr = precision_at_k(gr_ranked, gt, aliases, k)
            f_gr = f1_at_k(p_gr, r_gr)
            h_gr = hit_rate_at_k(gr_ranked, gt, aliases, k)
            ap_gr = average_precision_at_k(gr_ranked, gt, aliases, k)
            ndcg_gr = ndcg_at_k(gr_ranked, gt, aliases, k)

            for metric, val_vdb, val_gr in [
                (f"Recall@{k}", r_vdb, r_gr),
                (f"Precision@{k}", p_vdb, p_gr),
                (f"F1@{k}", f_vdb, f_gr),
                (f"HitRate@{k}", h_vdb, h_gr),
                (f"AP@{k}", ap_vdb, ap_gr),
                (f"NDCG@{k}", ndcg_vdb, ndcg_gr),
            ]:
                detail[f"{metric}_VDB"] = f"{val_vdb:.4f}"
                detail[f"{metric}_GraphRAG"] = f"{val_gr:.4f}"
                accum.setdefault(f"{metric}_VDB", []).append(val_vdb)
                accum.setdefault(f"{metric}_GraphRAG", []).append(val_gr)

        detail_rows.append(detail)

    # ── Aggregate: macro averages ─────────────────────────────────────────
    n_scored = len(detail_rows)
    summary: dict[str, str | float] = {"Scored_Questions": n_scored}

    for key, vals in accum.items():
        avg = sum(vals) / len(vals) if vals else 0.0
        summary[f"Avg_{key}"] = avg

    return detail_rows, summary


# ── Output formatting ────────────────────────────────────────────────────────

def _detail_columns() -> list[str]:
    """Build ordered list of column names for the detailed CSV."""
    cols = ["No", "Pertanyaan", "GT_Total", "MRR_VDB", "MRR_GraphRAG"]
    for k in CUTOFFS:
        for metric in ["Recall", "Precision", "F1", "HitRate", "AP", "NDCG"]:
            cols.append(f"{metric}@{k}_VDB")
            cols.append(f"{metric}@{k}_GraphRAG")
    return cols


def write_detail_csv(path: str, rows: list[dict]):
    """Write per-question metrics CSV."""
    cols = _detail_columns()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, summary: dict):
    """Write summary CSV with VDB | GraphRAG | Delta columns, grouped by cutoff."""
    rows_out: list[dict] = []

    rows_out.append({
        "Metric": "Scored_Questions",
        "VDB": summary["Scored_Questions"],
        "GraphRAG": summary["Scored_Questions"],
        "Delta": "",
    })

    # MRR first (no cutoff)
    vdb_val = summary.get("Avg_MRR_VDB", 0)
    gr_val = summary.get("Avg_MRR_GraphRAG", 0)
    rows_out.append({
        "Metric": "MRR",
        "VDB": f"{vdb_val:.4f}",
        "GraphRAG": f"{gr_val:.4f}",
        "Delta": f"{gr_val - vdb_val:+.4f}",
    })

    # Metrics per cutoff
    for k in CUTOFFS:
        rows_out.append({"Metric": f"--- @{k} ---", "VDB": "", "GraphRAG": "", "Delta": ""})
        for metric in ["Recall", "Precision", "F1", "HitRate", "AP", "NDCG"]:
            vdb_key = f"Avg_{metric}@{k}_VDB"
            gr_key = f"Avg_{metric}@{k}_GraphRAG"
            v = summary.get(vdb_key, 0)
            g = summary.get(gr_key, 0)
            rows_out.append({
                "Metric": f"{metric}@{k}",
                "VDB": f"{v:.4f}",
                "GraphRAG": f"{g:.4f}",
                "Delta": f"{g - v:+.4f}",
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "VDB", "GraphRAG", "Delta"])
        writer.writeheader()
        writer.writerows(rows_out)


def print_summary_table(name: str, summary: dict):
    """Print a formatted summary table to the terminal."""
    n = summary["Scored_Questions"]
    print(f"\n{'=' * 78}")
    print(f"  {name}  ({n} scored questions)")
    print(f"{'=' * 78}")
    print(f"  {'Metric':<18} {'VDB':>10} {'GraphRAG':>10} {'Δ (GR−VDB)':>12}")
    print(f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 12}")

    # MRR
    v = summary.get("Avg_MRR_VDB", 0)
    g = summary.get("Avg_MRR_GraphRAG", 0)
    print(f"  {'MRR':<18} {v:>10.4f} {g:>10.4f} {g - v:>+12.4f}")

    for k in CUTOFFS:
        print(f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 12}")
        print(f"  @{k}")
        for metric in ["Recall", "Precision", "F1", "HitRate", "AP", "NDCG"]:
            vk = summary.get(f"Avg_{metric}@{k}_VDB", 0)
            gk = summary.get(f"Avg_{metric}@{k}_GraphRAG", 0)
            label = f"  {metric}@{k}"
            print(f"  {label:<18} {vk:>10.4f} {gk:>10.4f} {gk - vk:>+12.4f}")

    print(f"{'=' * 78}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def process_file(csv_path: str):
    """Process one detailed CSV and produce metrics output."""
    basename = os.path.basename(csv_path)
    name = os.path.splitext(basename)[0]
    print(f"\n  Loading: {basename}")

    rows = load_benchmark_csv(csv_path)
    print(f"  Rows loaded: {len(rows)}")

    detail_rows, summary = compute_metrics(rows)

    # Write outputs
    os.makedirs(METRICS_DIR, exist_ok=True)

    detail_path = os.path.join(METRICS_DIR, f"{name}_metrics_at_k_detail.csv")
    write_detail_csv(detail_path, detail_rows)
    print(f"  → {detail_path}")

    summary_path = os.path.join(METRICS_DIR, f"{name}_metrics_at_k_summary.csv")
    write_summary_csv(summary_path, summary)
    print(f"  → {summary_path}")

    print_summary_table(name, summary)


def main():
    parser = argparse.ArgumentParser(description="Compute retrieval metrics @K")
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to a single detailed CSV (default: all *_v3.csv)")
    args = parser.parse_args()

    if args.csv:
        files = [args.csv]
    else:
        files = sorted(glob.glob(os.path.join(DETAIL_DIR, "*_v3.csv")))
        if not files:
            print("No *_v3.csv files found in", DETAIL_DIR)
            sys.exit(1)

    print(f"Processing {len(files)} file(s), cutoffs K={CUTOFFS}")

    for csv_path in files:
        process_file(csv_path)

    print("Done!")


if __name__ == "__main__":
    main()
