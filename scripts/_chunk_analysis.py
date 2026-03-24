#!/usr/bin/env python3
"""
Chunk-level retrieval analysis.

For each benchmark question, shows:
  1. Query
  2. Chunks retrieved (top-5 with RRF score + source doc)
  3. Whether chunks are sufficient to answer the question
  4. Why / why not

Outputs to: output/chunk_analysis.csv  (machine-readable)
            stdout                      (human-readable report)
"""

import csv, json, os, re, sys, textwrap

os.environ["GRAPHRAG_STANDALONE"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from shared import llm_stance, pinecone_client
from shared.bm25_index import hybrid_search

# ── Paths ────────────────────────────────────────────────────────────────────
BENCHMARK = os.path.join(_ROOT, "benchmark", "govnetic_qa_complete_50 (business).xlsx")
RESULTS   = os.path.join(_ROOT, "output",
                         "qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv")
OUT_CSV   = os.path.join(_ROOT, "output", "chunk_analysis.csv")

# ── Doc alias map (Evidence text → doc_id) ───────────────────────────────────
DOC_ALIASES = {
    "UU 40/2007": "UU-NASIONAL-40-2007",
    "UU 5/1999":  "UU-NASIONAL-5-1999",
    "UU 12/2011": "UU-NASIONAL-12-2011",
    "UU 13/2022": None,          # not in corpus
    "UU 20/2008": "UU-NASIONAL-20-2008",
    "Perppu 2/2022": "PERPPU-NASIONAL-2-2022",
    "PP 29/2021":  None,         # not in corpus
    "PP 16/2021": "PP-NASIONAL-16-2021",
    "UU 23/2014": "UU-NASIONAL-23-2014",
    "UU 11/2020": "UU-NASIONAL-11-2020",
    "UU 28/2002": "UU-NASIONAL-28-2002",
    "UU 11/2014": "UU-NASIONAL-11-2014",
    "UU 31/2002": "UU-NASIONAL-31-2002",
    "Permen PPN 7/2023": "PERMENPPN-NASIONAL-7-2023",
    "Permen Perdagangan 24/2021": "PERMENDAG-NASIONAL-24-2021",
}

_REF_PAT = re.compile(
    r'(UU|PP|Perppu|Permen\s*(?:Perdagangan|PPN|PUPR)?)\s*(?:No\.\s*)?(\d+)/(\d{4})',
    re.IGNORECASE,
)


def _extract_expected(evidence: str) -> set[str]:
    if not evidence:
        return set()
    result = set()
    for alias, doc_id in sorted(DOC_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias.lower() in evidence.lower():
            if doc_id:
                result.add(doc_id)
    for m in _REF_PAT.finditer(evidence):
        j, n, y = m.group(1).strip().lower(), m.group(2), m.group(3)
        if j == 'uu':       pfx = 'UU-NASIONAL'
        elif j == 'pp':     pfx = 'PP-NASIONAL'
        elif j == 'perppu': pfx = 'PERPPU-NASIONAL'
        elif 'perdagangan' in j: pfx = 'PERMENDAG-NASIONAL'
        elif 'ppn' in j:    pfx = 'PERMENPPN-NASIONAL'
        elif 'pupr' in j:   pfx = 'PERMENPUPR-NASIONAL'
        else: pfx = None
        if pfx:
            result.add(f"{pfx}-{n}-{y}")
    return result


# ── Query expansion (same as chatbot) ────────────────────────────────────────
def _expand(query: str) -> list[str]:
    q = query.lower().strip().rstrip("?!.")
    q = re.sub(
        r"^(jelaskan\s+(tentang\s+)?|apa\s+(itu\s+|yang\s+dimaksud\s+(dengan\s+)?)?|"
        r"definisi\s+|pengertian\s+|ceritakan\s+(tentang\s+)?|"
        r"uraikan\s+(tentang\s+)?|deskripsikan\s+)",
        "", q,
    ).strip()
    if not q or len(q) < 2:
        return []
    return [
        f"{q} adalah",
        f"definisi {q}",
        f"pengertian {q}",
        f"yang dimaksud dengan {q}",
    ]


def _multi_hybrid(query: str, top_k: int = 20) -> list[dict]:
    """Multi-query hybrid search with expansion — same as chatbot."""
    variants = [query] + _expand(query)
    all_hits = []
    seen = set()
    for q in variants:
        emb = llm_stance.get_embedding(q)
        hits = hybrid_search(q, emb, top_k=top_k, alpha=0.4)
        for h in hits:
            hid = h["id"]
            if hid not in seen:
                all_hits.append(h)
                seen.add(hid)
    all_hits.sort(key=lambda h: h.get("rrf_score", 0), reverse=True)
    return all_hits[:top_k]


# ── Sufficiency keywords extraction from expected answer ─────────────────────
def _extract_key_phrases(expected_answer: str) -> list[str]:
    """Extract pasal/ayat references and key legal terms from expected answer."""
    phrases = set()
    # Pasal references: Pasal 72, Pasal 91, Ayat (1)
    for m in re.finditer(r'[Pp]asal\s+\d+(?:\s+[Aa]yat\s+(?:\(\d+\)|\d+))?', expected_answer):
        phrases.add(m.group().lower())
    # UU/PP references
    for m in re.finditer(r'(?:UU|PP|Perppu)\s+\d+/\d{4}', expected_answer, re.IGNORECASE):
        phrases.add(m.group().lower())
    return list(phrases)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    import time as _time

    # ── Load benchmark Excel ─────────────────────────────────────────────────
    wb = openpyxl.load_workbook(BENCHMARK, read_only=True)
    ws = wb.active
    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            break
        questions.append({
            "tc":       str(row[0]),
            "query":    str(row[1]),
            "expected": str(row[2]) if row[2] else "",
            "evidence": str(row[3]) if len(row) > 3 and row[3] else "",
            "category": str(row[5]) if len(row) > 5 and row[5] else "",
        })
    wb.close()

    # ── Load previous verdict from benchmark CSV ─────────────────────────────
    verdict_map = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                verdict_map[r.get("No", "")] = r.get("Verdict", "")

    # ── Resume support: load partial CSV if exists ───────────────────────────
    done_tcs = set()
    csv_rows = []
    PARTIAL_CSV = OUT_CSV + ".partial"
    if os.path.exists(PARTIAL_CSV):
        with open(PARTIAL_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done_tcs.add(r["TC"])
                csv_rows.append(r)
        print(f"[RESUME] Loaded {len(done_tcs)} previously completed TCs")

    print(f"\n{'='*90}")
    print(f"  CHUNK-LEVEL RETRIEVAL ANALYSIS")
    print(f"  Query → Top Chunks → Cukup? → Kenapa")
    print(f"{'='*90}\n")

    stats = {"CUKUP": 0, "PARSIAL": 0, "TIDAK": 0}
    for r in csv_rows:
        stats[r["Cukup?"]] = stats.get(r["Cukup?"], 0) + 1

    for i, q in enumerate(questions):
        tc      = q["tc"]
        if tc in done_tcs:
            continue
        query   = q["query"]
        expans  = q["expected"]
        evidence = q["evidence"]
        category = q["category"]
        verdict  = verdict_map.get(tc, "?")

        expected_docs = _extract_expected(evidence)
        key_phrases   = _extract_key_phrases(expans)

        # ── Run hybrid retrieval (with retry) ─────────────────────────────────
        print(f"  [{i+1}/{len(questions)}] Processing {tc}...", flush=True)
        for attempt in range(3):
            try:
                hits = _multi_hybrid(query, top_k=20)
                break
            except Exception as e:
                print(f"  [RETRY {attempt+1}/3] {tc}: {e}")
                _time.sleep(3)
        else:
            print(f"  [SKIP] {tc}: failed after 3 retries")
            continue
        top5 = hits[:5]

        # ── Determine doc overlap ────────────────────────────────────────────
        retrieved_docs = set()
        for h in hits:
            did = h.get("doc_id", "")
            if did:
                retrieved_docs.add(did)

        found_docs   = expected_docs & retrieved_docs
        missing_docs = expected_docs - retrieved_docs

        # ── Count key phrase hits in chunks ──────────────────────────────────
        all_text = " ".join(h.get("content", "") for h in hits).lower()
        phrases_found   = [p for p in key_phrases if p in all_text]
        phrases_missing = [p for p in key_phrases if p not in all_text]

        # ── Determine sufficiency ────────────────────────────────────────────
        doc_coverage = len(found_docs) / len(expected_docs) if expected_docs else 1.0
        phrase_coverage = len(phrases_found) / len(key_phrases) if key_phrases else 1.0

        if doc_coverage >= 1.0 and phrase_coverage >= 0.5:
            sufficiency = "CUKUP"
        elif doc_coverage > 0 or phrase_coverage > 0.3:
            sufficiency = "PARSIAL"
        else:
            sufficiency = "TIDAK"
        stats[sufficiency] += 1

        # ── Build "why" explanation ──────────────────────────────────────────
        reasons = []

        if not expected_docs:
            reasons.append("Tidak ada expected doc_id dari evidence")
        elif missing_docs:
            reasons.append(f"Dokumen tidak ditemukan: {', '.join(sorted(missing_docs))}")
        if found_docs:
            reasons.append(f"Dokumen ditemukan: {', '.join(sorted(found_docs))}")

        if phrases_missing:
            reasons.append(f"Pasal/referensi tidak ada di chunk: {', '.join(phrases_missing[:5])}")
        if phrases_found:
            reasons.append(f"Pasal/referensi ditemukan: {', '.join(phrases_found[:5])}")

        # Check if attractor docs dominate
        attractor_count = sum(1 for h in top5
                              if h.get("doc_id") in ("UU-NASIONAL-11-2020", "PERPPU-NASIONAL-2-2022"))
        if attractor_count >= 4:
            reasons.append(f"Top-5 didominasi attractor docs ({attractor_count}/5)")

        if not expected_docs & retrieved_docs and expected_docs:
            # All expected docs missed
            top_docs = []
            seen_d = set()
            for h in top5:
                d = h.get("doc_id", "")
                if d and d not in seen_d:
                    top_docs.append(d)
                    seen_d.add(d)
            reasons.append(f"Yang muncul malah: {', '.join(top_docs)}")

        why = "; ".join(reasons) if reasons else "Retrieval sesuai dengan kebutuhan"

        # ── Print ────────────────────────────────────────────────────────────
        print(f"┌─ [{tc}] {query[:80]}")
        print(f"│  Kategori: {category}  |  LLM Verdict: {verdict}")
        print(f"│  Expected docs: {', '.join(sorted(expected_docs)) if expected_docs else '(none)'}")
        print(f"│  Top-5 chunks retrieved:")
        for j, h in enumerate(top5):
            content_preview = h.get("content", "")[:100].replace("\n", " ")
            rrf = h.get("rrf_score", 0)
            did = h.get("doc_id", "?")
            print(f"│    {j+1}. [{did}] (rrf={rrf:.4f}) {content_preview}...")
        print(f"│")
        print(f"│  ➤ Cukup? {sufficiency}")
        print(f"│  ➤ Kenapa: {why}")
        print(f"└{'─'*89}")
        print()

        # ── CSV row ──────────────────────────────────────────────────────────
        chunk_summary = " | ".join(
            f"[{h.get('doc_id','?')}] {h.get('content','')[:80].replace(chr(10),' ')}"
            for h in top5
        )
        csv_rows.append({
            "TC":               tc,
            "Query":            query,
            "Expected Docs":    ", ".join(sorted(expected_docs)),
            "Retrieved Docs":   ", ".join(sorted(retrieved_docs)),
            "Missing Docs":     ", ".join(sorted(missing_docs)),
            "Top-5 Chunks":     chunk_summary,
            "Cukup?":           sufficiency,
            "Kenapa":           why,
            "LLM Verdict":      verdict,
        })

        # ── Checkpoint save ──────────────────────────────────────────────────
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        with open(PARTIAL_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(questions)
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"  CUKUP  : {stats['CUKUP']:3d}/{total} ({stats['CUKUP']/total*100:.1f}%) — chunk cukup untuk jawab")
    print(f"  PARSIAL: {stats['PARSIAL']:3d}/{total} ({stats['PARSIAL']/total*100:.1f}%) — sebagian ada, kurang lengkap")
    print(f"  TIDAK  : {stats['TIDAK']:3d}/{total} ({stats['TIDAK']/total*100:.1f}%) — chunk tidak bisa jawab")
    print()

    # Cross-tab with LLM verdict
    print("  Cross-tab: Chunk Sufficiency × LLM Verdict")
    print(f"  {'':10s} BENAR  PARSIAL  SALAH  OTHER")
    for suf in ["CUKUP", "PARSIAL", "TIDAK"]:
        row = f"  {suf:10s}"
        for v in ["BENAR", "PARSIAL", "SALAH"]:
            cnt = sum(1 for r in csv_rows if r["Cukup?"] == suf and r["LLM Verdict"] == v)
            row += f"  {cnt:5d}"
        other = sum(1 for r in csv_rows if r["Cukup?"] == suf and r["LLM Verdict"] not in ("BENAR","PARSIAL","SALAH"))
        row += f"  {other:5d}"
        print(row)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  Saved: {OUT_CSV}")

    # Clean up partial file
    if os.path.exists(PARTIAL_CSV):
        os.remove(PARTIAL_CSV)


if __name__ == "__main__":
    main()
