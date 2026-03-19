#!/usr/bin/env python3
import csv

for name in ['QA 100 (test-all-sector)_v3', 'govnetic_qa_complete_50 (business)_v3']:
    path = f'output/retrieval/detailed retrieval/{name}.csv'
    with open(path) as f:
        rows = list(csv.DictReader(f))
    errors = sum(1 for r in rows if 'Error' in str(r.get('Dok_VDB', '')))
    total = len(rows)
    scored = total - errors
    recalls_gr = [float(r['Recall_GraphRAG']) for r in rows if 'Error' not in str(r.get('Dok_VDB', '')) and r['Recall_GraphRAG']]
    recalls_vdb = [float(r['Recall_VDB']) for r in rows if 'Error' not in str(r.get('Dok_VDB', '')) and r['Recall_VDB']]
    perfect_gr = sum(1 for v in recalls_gr if v >= 1.0)
    perfect_vdb = sum(1 for v in recalls_vdb if v >= 1.0)
    zero_gr = sum(1 for v in recalls_gr if v == 0.0)
    zero_vdb = sum(1 for v in recalls_vdb if v == 0.0)
    avg_gr = sum(recalls_gr) / len(recalls_gr) if recalls_gr else 0
    avg_vdb = sum(recalls_vdb) / len(recalls_vdb) if recalls_vdb else 0
    print(f'=== {name} ===')
    print(f'  Total={total}, Errors={errors}, Scored={scored}')
    print(f'  Avg Recall GR={avg_gr:.4f}, Avg Recall VDB={avg_vdb:.4f}')
    print(f'  Perfect(1.0) GR={perfect_gr}, VDB={perfect_vdb}')
    print(f'  Zero(0.0) GR={zero_gr}, VDB={zero_vdb}')
    if errors > 0:
        clean_gr = [float(r['Recall_GraphRAG']) for r in rows if 'Error' not in str(r.get('Dok_VDB', ''))]
        clean_vdb = [float(r['Recall_VDB']) for r in rows if 'Error' not in str(r.get('Dok_VDB', ''))]
        cg = sum(clean_gr)/len(clean_gr)
        cv = sum(clean_vdb)/len(clean_vdb)
        print(f'  ** Excluding {errors} error rows: Avg Recall GR={cg:.4f}, VDB={cv:.4f}')
    print()
