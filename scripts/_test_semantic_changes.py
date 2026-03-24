"""Quick test for the new semantic search functions."""
import sys, os
os.environ['GRAPHRAG_STANDALONE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chatbot'))

from chatbot.utils.langgraph_agent import (
    _extract_doc_references, _get_diverse_doc_ids, _follow_graph_edges,
    _expand_for_definition, _TOPIC_LAW_MAP
)

print("=== _extract_doc_references ===")
tests = [
    ("Menurut UU 40 tahun 2007 tentang PT", ["UU-NASIONAL-40-2007"]),
    ("Berdasarkan PP No. 16/2021", ["PP-NASIONAL-16-2021"]),
    ("Sesuai Perppu 2 Tahun 2022", ["PERPPU-NASIONAL-2-2022"]),
    ("apa itu perseroan terbatas", []),  # no explicit ref
    ("UU Nomor 11 Tahun 2020", ["UU-NASIONAL-11-2020"]),
    ("Undang-Undang 28/2002", ["UU-NASIONAL-28-2002"]),
    ("PP 14 tahun 2021 dan UU 2/2017", ["UU-NASIONAL-2-2017", "PP-NASIONAL-14-2021"]),
]
all_ok = True
for query, expected in tests:
    result = _extract_doc_references(query)
    status = "OK" if result == expected else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  {status}: '{query}' -> {result} (expected {expected})")

print("\n=== _expand_for_definition ===")
exp_tests = [
    ("apa itu perseroan terbatas", True),    # should have UU ref
    ("jelaskan tentang bangunan gedung", True),  # should have UU ref
    ("apa itu arsitek", True),                   # should have UU ref
    ("bagaimana cara mendirikan PT", False),      # no topic match
]
for query, should_have_uu in exp_tests:
    result = _expand_for_definition(query)
    has_uu = any("Undang-Undang" in v for v in result)
    if should_have_uu:
        status = "OK" if has_uu else "FAIL"
    else:
        status = "OK"
    if status == "FAIL":
        all_ok = False
    print(f"  {status}: '{query}' -> {result}")

print("\n=== _get_diverse_doc_ids ===")
test_hits = [
    {'doc_id': 'A', 'rrf_score': 0.1},
    {'doc_id': 'A', 'rrf_score': 0.09},
    {'doc_id': 'A', 'rrf_score': 0.08},
    {'doc_id': 'B', 'rrf_score': 0.095},
    {'doc_id': 'C', 'rrf_score': 0.07},
    {'doc_id': 'A', 'rrf_score': 0.06},
    {'doc_id': 'D', 'rrf_score': 0.05},
]
diverse = _get_diverse_doc_ids(test_hits, max_docs=3)
expected_order = ['A', 'B', 'C']  # A(0.1) > B(0.095) > C(0.07)
status = "OK" if diverse == expected_order else "FAIL"
if status == "FAIL":
    all_ok = False
print(f"  {status}: {diverse} (expected {expected_order})")

print("\n=== TOPIC_LAW_MAP ===")
print(f"  {len(_TOPIC_LAW_MAP)} topic mappings loaded")

print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
