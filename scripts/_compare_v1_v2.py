#!/usr/bin/env python3
"""Compare V1 vs V2 vs V3 benchmark results."""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
versions = {
    "V1": os.path.join(ROOT, "output", "qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv"),
    "V2": os.path.join(ROOT, "output", "qa_benchmark_semantic_v2_govnetic_qa_complete_50_(business).csv"),
    "V3": os.path.join(ROOT, "output", "qa_benchmark_semantic_v3_govnetic_qa_complete_50_(business).csv"),
}

data = {}
for ver, path in versions.items():
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            data[ver] = {r["No"]: r for r in csv.DictReader(f)}

rank = {"BENAR": 0, "PARSIAL": 1, "SALAH": 2}

# Summary table
print("=" * 60)
print(f"{'Version':<10} {'BENAR':>8} {'PARSIAL':>8} {'SALAH':>8} {'Score':>8}")
print("-" * 60)
for ver, rows in data.items():
    verdicts = [r["Verdict"] for r in rows.values()]
    b = verdicts.count("BENAR")
    p = verdicts.count("PARSIAL")
    s = verdicts.count("SALAH")
    total = len(verdicts)
    # Weighted score: BENAR=2, PARSIAL=1, SALAH=0
    score = (b * 2 + p * 1) / (total * 2) * 100
    print(f"{ver:<10} {b:>5} ({b/total*100:4.1f}%) {p:>3} ({p/total*100:4.1f}%) {s:>3} ({s/total*100:4.1f}%) {score:>6.1f}%")
print("=" * 60)

# V1 vs V3 comparison
if "V1" in data and "V3" in data:
    v1 = data["V1"]
    v3 = data["V3"]
    print("\n=== V1 -> V3 DEGRADED ===")
    for no in sorted(v1.keys()):
        v1v = v1[no]["Verdict"]
        v3v = v3[no]["Verdict"]
        if rank.get(v3v, 3) > rank.get(v1v, 3):
            q = v1[no]["Pertanyaan"][:80]
            d1 = v1[no].get("Docs", "")[:80]
            d3 = v3[no].get("Docs", "")[:80]
            print(f"  {no}: {v1v} -> {v3v}")
            print(f"    Q: {q}")
            print(f"    V1 docs: {d1}")
            print(f"    V3 docs: {d3}")
            print()

    print("=== V1 -> V3 IMPROVED ===")
    for no in sorted(v1.keys()):
        v1v = v1[no]["Verdict"]
        v3v = v3[no]["Verdict"]
        if rank.get(v3v, 3) < rank.get(v1v, 3):
            q = v1[no]["Pertanyaan"][:80]
            d1 = v1[no].get("Docs", "")[:80]
            d3 = v3[no].get("Docs", "")[:80]
            print(f"  {no}: {v1v} -> {v3v}")
            print(f"    Q: {q}")
            print(f"    V1 docs: {d1}")
            print(f"    V3 docs: {d3}")
            print()

    # Doc diversity
    v1_docs = [len([d for d in v1[no].get("Docs", "").split(", ") if d]) for no in v1]
    v3_docs = [len([d for d in v3[no].get("Docs", "").split(", ") if d]) for no in v3]
    print(f"V1 avg docs/query: {sum(v1_docs)/len(v1_docs):.1f}")
    print(f"V3 avg docs/query: {sum(v3_docs)/len(v3_docs):.1f}")
