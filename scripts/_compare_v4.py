#!/usr/bin/env python3
"""Compare V4 sample results against V1/V2/V3."""
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

versions = {
    "V1": os.path.join(ROOT, "output/qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv"),
    "V2": os.path.join(ROOT, "output/qa_benchmark_semantic_v2_govnetic_qa_complete_50_(business).csv"),
    "V3": os.path.join(ROOT, "output/qa_benchmark_semantic_v3_govnetic_qa_complete_50_(business).csv"),
    "V4": os.path.join(ROOT, "output/qa_benchmark_semantic_v4_govnetic_qa_complete_50_(business).csv"),
    "Rerank": os.path.join(ROOT, "output/qa_benchmark_semantic_rerank_govnetic_qa_complete_50_(business).csv"),
}

data = {}
for ver, path in versions.items():
    df = pd.read_csv(path)
    data[ver] = dict(zip(df["No"].astype(str), df["Verdict"].astype(str)))

v4_ids = sorted(data["V4"].keys())
score_map = {"BENAR": 2, "PARSIAL": 1, "SALAH": 0}

header = "{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {}".format("TC", "V1", "V2", "V3", "V4", "Rerank", "V1->Rerank")
print(header)
print("-" * 85)

for tc in v4_ids:
    v1 = data["V1"].get(tc, "?")
    v2 = data["V2"].get(tc, "?")
    v3 = data["V3"].get(tc, "?")
    v4 = data["V4"].get(tc, "?")
    rr = data["Rerank"].get(tc, "?")
    s1 = score_map.get(v1, -1)
    sr = score_map.get(rr, -1)
    if sr > s1:
        arrow = "UP IMPROVED"
    elif sr < s1:
        arrow = "DOWN DEGRADED"
    else:
        arrow = "= SAME"
    print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {}".format(tc, v1, v2, v3, v4, rr, arrow))

print()
print("Summary (12 sampled questions only):")
for ver in ["V1", "V2", "V3", "V4", "Rerank"]:
    vals = [data[ver].get(tc, "?") for tc in v4_ids]
    b = vals.count("BENAR")
    p = vals.count("PARSIAL")
    s = vals.count("SALAH")
    w = (b * 2 + p) / (len(vals) * 2) * 100
    print("  {}: BENAR={} PARSIAL={} SALAH={} weighted={:.1f}%".format(ver, b, p, s, w))
