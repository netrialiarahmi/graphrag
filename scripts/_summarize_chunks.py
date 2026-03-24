import csv
from collections import Counter

with open('output/chunk_analysis.csv', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

suf = Counter(r['Cukup?'] for r in rows)
print('=== CHUNK SUFFICIENCY ===')
for s in ['CUKUP','PARSIAL','TIDAK']:
    print(f'  {s:8s}: {suf[s]:2d}/40  ({suf[s]/40*100:.0f}%)')

print()
print('=== MISSING DOCS FREQUENCY ===')
miss = Counter()
for r in rows:
    for d in (d.strip() for d in r['Missing Docs'].split(',') if d.strip()):
        miss[d] += 1
for d,c in miss.most_common():
    print(f'  {c:2d}x missed: {d}')

print()
print('=== DETAIL PER TC ===')
for r in rows:
    s = r['Cukup?']
    icon = {'CUKUP':'V','PARSIAL':'~','TIDAK':'X'}[s]
    tc = r['TC']
    q = r['Query'][:65]
    missing = r['Missing Docs'] if r['Missing Docs'] else '-'
    verdict = r['LLM Verdict']
    print(f'  [{icon}] {tc} ({verdict:7s}) {q}')
    if missing != '-':
        print(f'      missing: {missing}')
