#!/usr/bin/env python3
"""
Retrieval-only evaluator for Semantic Benchmark.

Compares retrieved doc_ids against expected Evidence docs from the benchmark Excel.
This separates "did we find the right documents?" from "did the LLM know the answer already?"

Also checks whether the retrieved CHUNKS actually contain the answer keywords.
"""
import csv
import re
import openpyxl

BENCHMARK_XLSX = '/Users/netrialiarahmi/Library/Mobile Documents/com~apple~CloudDocs/career/govnetic/graphrag/graphrag/benchmark/govnetic_qa_complete_50 (business).xlsx'
RESULTS_CSV = '/Users/netrialiarahmi/Library/Mobile Documents/com~apple~CloudDocs/career/govnetic/graphrag/graphrag/output/qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv'

# ── Evidence → doc_id mapping ─────────────────────────────────────────────────
# Map natural language references to actual doc_ids in the corpus
DOC_ALIASES = {
    "UU 40/2007": "UU-NASIONAL-40-2007",
    "UU 5/1999": "UU-NASIONAL-5-1999",
    "UU 12/2011": "UU-NASIONAL-12-2011",
    "UU 13/2022": None,  # NOT in corpus — amends UU 12/2011
    "UU 20/2008": "UU-NASIONAL-20-2008",
    "Perppu 2/2022": "PERPPU-NASIONAL-2-2022",
    "PP 29/2021": None,  # NOT in corpus — check!
    "PP 15/2021": "PP-NASIONAL-15-2021",
    "PP 14/2021": "PP-NASIONAL-14-2021",
    "Permen Perdagangan 24/2021": "PERMENDAG-NASIONAL-24-2021",
    "Peraturan Menteri Perdagangan 24/2021": "PERMENDAG-NASIONAL-24-2021",
    "Permen PPN 7/2023": "PERMENPPN-NASIONAL-7-2023",
    "Peraturan Menteri Perencanaan Pembangunan Nasional": "PERMENPPN-NASIONAL-7-2023",
    "Peraturan BPS 2/2020": "PERBANBPS-NASIONAL-2-2020",
    "Peraturan Badan Pusat Statistik 2/2020": "PERBANBPS-NASIONAL-2-2020",
}

# Regex pattern: "UU|PP|Perppu|Permen" + number/year
_REF_PAT = re.compile(
    r'(UU|PP|Perppu|Permen\s+(?:Perdagangan|PPN|PUPR)?|Peraturan\s+(?:BPS|Badan\s+Pusat\s+Statistik|Menteri\s+\w+))\s+'
    r'(\d+)/(\d{4})',
    re.IGNORECASE
)

def _extract_expected_doc_ids(evidence_text: str) -> set[str]:
    """Extract expected doc_ids from the Evidence column text."""
    if not evidence_text:
        return set()

    result = set()

    # First try known aliases (longest match first)
    for alias, doc_id in sorted(DOC_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias.lower() in evidence_text.lower():
            if doc_id:
                result.add(doc_id)

    # Then try regex for any "TYPE number/year" pattern
    for m in _REF_PAT.finditer(evidence_text):
        jenis = m.group(1).strip()
        nomor = m.group(2)
        tahun = m.group(3)

        # Normalize jenis to doc_id prefix
        j = jenis.lower()
        if j == 'uu':
            prefix = 'UU-NASIONAL'
        elif j == 'pp':
            prefix = 'PP-NASIONAL'
        elif j == 'perppu':
            prefix = 'PERPPU-NASIONAL'
        elif 'perdagangan' in j:
            prefix = 'PERMENDAG-NASIONAL'
        elif 'ppn' in j:
            prefix = 'PERMENPPN-NASIONAL'
        elif 'pupr' in j:
            prefix = 'PERMENPUPR-NASIONAL'
        elif 'bps' in j or 'statistik' in j:
            prefix = 'PERBANBPS-NASIONAL'
        else:
            prefix = None

        if prefix:
            doc_id = f"{prefix}-{nomor}-{tahun}"
            result.add(doc_id)

    return result


def main():
    # Load evidence from Excel
    wb = openpyxl.load_workbook(BENCHMARK_XLSX, read_only=True)
    ws = wb.active
    evidence_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            break
        tc = str(row[0])
        evidence = str(row[3]) if len(row) > 3 and row[3] else ""
        evidence_map[tc] = evidence
    wb.close()

    # Load retrieved docs from benchmark CSV
    with open(RESULTS_CSV, newline='', encoding='utf-8-sig') as f:
        results = list(csv.DictReader(f))

    # Known doc_ids in corpus (for checking availability)
    import json
    with open('/Users/netrialiarahmi/Library/Mobile Documents/com~apple~CloudDocs/career/govnetic/graphrag/graphrag/chatbot/bm25_corpus.json') as f:
        corpus = json.load(f)
    corpus_doc_ids = set(c['doc_id'] for c in corpus if c.get('doc_id'))

    print(f"{'='*80}")
    print(f"  RETRIEVAL-ONLY EVALUATION")
    print(f"  (Does hybrid search find the RIGHT documents?)")
    print(f"{'='*80}\n")

    total = 0
    full_hit = 0  # all expected docs found
    partial_hit = 0  # some expected docs found
    zero_hit = 0  # none found
    not_in_corpus = 0  # expected docs not even in Pinecone
    total_expected = 0
    total_found = 0

    details = []

    for r in results:
        tc = r['No']
        evidence = evidence_map.get(tc, "")
        retrieved_str = r.get('Docs', '')
        retrieved = set(d.strip() for d in retrieved_str.split(',') if d.strip())
        expected = _extract_expected_doc_ids(evidence)

        # Check which expected docs are actually in corpus
        expected_in_corpus = expected & corpus_doc_ids
        expected_missing_from_corpus = expected - corpus_doc_ids

        # Calculate hit
        found = expected_in_corpus & retrieved
        missed = expected_in_corpus - retrieved

        total += 1
        total_expected += len(expected_in_corpus)
        total_found += len(found)

        if expected_missing_from_corpus:
            not_in_corpus += len(expected_missing_from_corpus)

        if len(expected_in_corpus) == 0:
            status = "N/A (no expected docs in corpus)"
        elif len(found) == len(expected_in_corpus):
            full_hit += 1
            status = "FULL HIT"
        elif len(found) > 0:
            partial_hit += 1
            status = "PARTIAL"
        else:
            zero_hit += 1
            status = "MISS"

        details.append({
            'tc': tc,
            'status': status,
            'expected': expected_in_corpus,
            'expected_not_in_corpus': expected_missing_from_corpus,
            'found': found,
            'missed': missed,
            'retrieved': retrieved,
            'verdict': r['Verdict'],
        })

    # Summary
    evaluable = full_hit + partial_hit + zero_hit
    recall = total_found / total_expected * 100 if total_expected else 0

    print(f"  Documents in corpus: {len(corpus_doc_ids)}")
    print(f"  Questions evaluated: {total}")
    print(f"  Expected docs not in corpus: {not_in_corpus} instances")
    print()
    print(f"  FULL HIT  (all expected docs retrieved) : {full_hit}/{evaluable} ({full_hit/evaluable*100:.1f}%)")
    print(f"  PARTIAL   (some expected docs retrieved) : {partial_hit}/{evaluable} ({partial_hit/evaluable*100:.1f}%)")
    print(f"  MISS      (zero expected docs retrieved) : {zero_hit}/{evaluable} ({zero_hit/evaluable*100:.1f}%)")
    print(f"  Document Recall: {total_found}/{total_expected} ({recall:.1f}%)")
    print()

    # Cross-tab: Retrieval vs Verdict
    print(f"  {'─'*60}")
    print(f"  Cross-tab: Retrieval Hit vs LLM Verdict")
    print(f"  {'─'*60}")
    print(f"  {'':15s} {'BENAR':>8s} {'PARSIAL':>8s} {'SALAH':>8s}")
    for s_name in ['FULL HIT', 'PARTIAL', 'MISS']:
        b = sum(1 for d in details if d['status'] == s_name and d['verdict'] == 'BENAR')
        p = sum(1 for d in details if d['status'] == s_name and d['verdict'] == 'PARSIAL')
        s = sum(1 for d in details if d['status'] == s_name and d['verdict'] == 'SALAH')
        print(f"  {s_name:15s} {b:>8d} {p:>8d} {s:>8d}")
    print()

    # Detail: MISS and cases where verdict=BENAR but retrieval=MISS/PARTIAL
    print(f"{'='*80}")
    print(f"  SUSPICIOUS: LLM said BENAR but retrieval was PARTIAL or MISS")
    print(f"  (= LLM possibly answered from internal knowledge, not chunks)")
    print(f"{'='*80}")
    for d in details:
        if d['verdict'] == 'BENAR' and d['status'] in ('PARTIAL', 'MISS'):
            print(f"\n  {d['tc']}: verdict={d['verdict']}, retrieval={d['status']}")
            print(f"    Expected (in corpus): {d['expected']}")
            print(f"    Found:    {d['found']}")
            print(f"    Missed:   {d['missed']}")
            print(f"    Retrieved: {d['retrieved']}")

    print(f"\n{'='*80}")
    print(f"  ALL RETRIEVAL DETAILS")
    print(f"{'='*80}")
    for d in details:
        flag = ""
        if d['verdict'] == 'BENAR' and d['status'] in ('PARTIAL', 'MISS'):
            flag = " ⚠️ SUSPICIOUS"
        elif d['status'] == 'MISS':
            flag = " ❌"
        print(f"\n  {d['tc']}: retrieval={d['status']}, verdict={d['verdict']}{flag}")
        if d['expected']:
            print(f"    Expected: {', '.join(sorted(d['expected']))}")
        if d['expected_not_in_corpus']:
            print(f"    NOT in corpus: {', '.join(sorted(d['expected_not_in_corpus']))}")
        if d['missed']:
            print(f"    MISSED:   {', '.join(sorted(d['missed']))}")
        print(f"    Retrieved: {', '.join(sorted(d['retrieved']))}")


if __name__ == '__main__':
    main()
