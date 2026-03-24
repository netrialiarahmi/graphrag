#!/usr/bin/env python3
"""Quick script to check benchmark results."""
import csv, sys

path = '/Users/netrialiarahmi/Library/Mobile Documents/com~apple~CloudDocs/career/govnetic/graphrag/graphrag/output/qa_benchmark_semantic_govnetic_qa_complete_50_(business).csv'
with open(path, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
benar = sum(1 for r in rows if r['Verdict'] == 'BENAR')
parsial = sum(1 for r in rows if r['Verdict'] == 'PARSIAL')
salah = sum(1 for r in rows if r['Verdict'] == 'SALAH')
other = total - benar - parsial - salah

print(f'Progress: {total}/50')
print(f'BENAR:   {benar} ({benar/total*100:.1f}%)')
print(f'PARSIAL: {parsial} ({parsial/total*100:.1f}%)')
print(f'SALAH:   {salah} ({salah/total*100:.1f}%)')
if other:
    print(f'OTHER:   {other}')

print('\n=== SALAH ===')
for r in rows:
    if r['Verdict'] == 'SALAH':
        print(f"  {r['No']}: {r['Pertanyaan'][:80]}")
        print(f"     docs: {r.get('Docs','')}")

print('\n=== BENAR ===')
for r in rows:
    if r['Verdict'] == 'BENAR':
        print(f"  {r['No']}: {r['Pertanyaan'][:60]}")
