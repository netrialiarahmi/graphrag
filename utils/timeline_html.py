"""Helpers for building HTML timelines from relation CSVs."""
from __future__ import annotations

import pandas as pd
import re
from collections import defaultdict
from html import escape

VISUAL_LEVELS = [
    (1,  "Undang-Undang Dasar",     ["undang-undang dasar", "uud"]),
    (2,  "Ketetapan MPR",           ["ketetapan mpr", "tap mpr"]),
    (3,  "Undang-Undang / Perppu",  [
        "peraturan pemerintah pengganti undang-undang",
        " perpu", "undang-undang",
    ]),
    (4,  "Peraturan Pemerintah",    ["peraturan pemerintah", " pp"]),
    (5,  "Peraturan Presiden",      ["peraturan presiden", " perpres"]),
    (6,  "Keputusan Presiden",      ["keputusan presiden", " keppres"]),
    (7,  "Instruksi Presiden",      ["instruksi presiden", " inpres"]),
    (8,  "Peraturan Menteri",       ["peraturan menteri", " permen"]),
    (9,  "Peraturan Daerah",        [
        "peraturan daerah provinsi", "peraturan daerah kabupaten",
        "peraturan daerah kota", "peraturan daerah", " perda", "perda",
    ]),
    (10, "Peraturan Kepala Daerah", [
        "peraturan gubernur", " pergub", "peraturan bupati", " perbup",
        "peraturan wali kota", "peraturan walikota", " perwali",
        "wali kota", "walikota",
    ]),
    (11, "Regulasi Lainnya",        []),
]

COLOR_ENTAILMENT = "#059669"
COLOR_CONFLICT = "#dc2626"
COLOR_NEUTRAL = "#94a3b8"
COLOR_NODE = "#ffffff"
COLOR_BORDER = "#0f172a"
COLOR_BG = "#f8fafc"

YEAR_GAP = 240
LEVEL_GAP = 88
NODE_WIDTH = 196
NODE_HEIGHT = 52
LEFT_MARGIN = 320
TOP_MARGIN = 84
RIGHT_MARGIN = 90
BOTTOM_MARGIN = 116


def extract_year(doc_id: str | None) -> int | None:
    if not isinstance(doc_id, str):
        return None
    m = re.search(r"(19|20)\d{2}(?=$|[^0-9])", doc_id)
    if m:
        return int(m.group(0))
    m2 = re.search(r"-(\d{4})$", doc_id)
    return int(m2.group(1)) if m2 else None


def determine_level(doc_id: str | None) -> int:
    if not isinstance(doc_id, str):
        return 11
    ctx = doc_id.lower().strip()

    # Normalize common document-id prefixes before falling back to keywords.
    prefix_rules = [
        (r"^uud\b", 1),
        (r"^tap\s*mpr\b", 2),
        (r"^perppu\b", 3),
        (r"^uu\b", 3),
        (r"^pp\b", 4),
        (r"^perpres\b", 5),
        (r"^keppres\b", 6),
        (r"^inpres\b", 7),
        (r"^permen\b", 8),
        (r"^perda\b", 9),
        (r"^pergub\b", 10),
        (r"^perbup\b", 10),
        (r"^perwali\b", 10),
    ]
    for pattern, level in prefix_rules:
        if re.search(pattern, ctx):
            return level

    # Catch separators like UU-NASIONAL-12-2011, PP-NASIONAL-84-2001, etc.
    if re.search(r"\buu[-_ ]", ctx):
        return 3
    if re.search(r"\bpp[-_ ]", ctx):
        return 4
    if re.search(r"\bpermen[-_ ]", ctx):
        return 8
    if re.search(r"\bperda[-_ ]", ctx):
        return 9
    if re.search(r"\bpergub[-_ ]|\bperbup[-_ ]|\bperwali[-_ ]", ctx):
        return 10

    for level, _, keywords in VISUAL_LEVELS:
        for kw in keywords:
            needle = kw.strip().lower()
            if needle and needle in ctx:
                return level
    return 11


def _load_csv(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if not {"doc_1", "doc_2"}.issubset(df.columns):
        return None
    return df


def build_timeline_html(csv_path: str) -> str | None:
    df = _load_csv(csv_path)
    if df is None or df.empty:
        return None

    docs: dict[str, dict] = {}
    for col in ("doc_1", "doc_2"):
        for doc_id in df[col].dropna().astype(str).unique():
            year = extract_year(doc_id)
            level = determine_level(doc_id)
            docs[doc_id] = {"year": year, "level": level}

    if not docs:
        return None

    valid_years = sorted({info["year"] for info in docs.values() if info["year"] is not None})
    years = valid_years or [2020]
    x_map = {y: idx for idx, y in enumerate(years)}

    used_levels = sorted({info["level"] for info in docs.values()})
    level_to_row = {level: idx for idx, level in enumerate(used_levels)}

    positions: dict[str, dict] = {}
    slot_counts: dict[tuple[int, int], int] = defaultdict(int)
    row_count = len(used_levels)
    for doc_id, info in docs.items():
        year = info.get("year")
        idx = x_map.get(year, 0)
        level = info.get("level", 11)
        row = level_to_row.get(level, 0)
        slot = (idx, row)
        offset = slot_counts[slot]
        slot_counts[slot] += 1
        base_x = LEFT_MARGIN + idx * YEAR_GAP + offset * 10
        base_y = TOP_MARGIN + row * LEVEL_GAP + offset * 6
        positions[doc_id] = {
            "x": base_x,
            "y": base_y,
            "cx": base_x + NODE_WIDTH / 2,
            "cy": base_y + NODE_HEIGHT / 2,
            "year_idx": idx,
            "level": level,
        }

    edges = []
    seen_edges = set()
    for _, row in df.iterrows():
        a = str(row.get("doc_1", ""))
        b = str(row.get("doc_2", ""))
        if a not in positions or b not in positions:
            continue
        rel = str(row.get("relation_type", "")).lower()
        if "entail" in rel:
            color = COLOR_ENTAILMENT
        elif "conflict" in rel or "contrad" in rel or "conf" in rel:
            color = COLOR_CONFLICT
        else:
            color = COLOR_NEUTRAL
        edge_key = (a, b, color)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append({"source": a, "target": b, "color": color})

    svg_width = LEFT_MARGIN + max(x_map.values(), default=0) * YEAR_GAP + NODE_WIDTH + RIGHT_MARGIN
    svg_height = TOP_MARGIN + max(0, row_count - 1) * LEVEL_GAP + NODE_HEIGHT + BOTTOM_MARGIN

    level_labels = {level: label for level, label, _ in VISUAL_LEVELS}

    svg_lines = []
    for year, idx in x_map.items():
        x = LEFT_MARGIN + idx * YEAR_GAP
        svg_lines.append(
            f'<line x1="{x:.1f}" y1="{TOP_MARGIN - 20:.1f}" x2="{x:.1f}" y2="{svg_height - BOTTOM_MARGIN + 20:.1f}" '
            f'stroke="#cbd5f5" stroke-dasharray="4,6" stroke-width="1" />'
        )

    level_texts = []
    for level in used_levels:
        row = level_to_row[level]
        y = TOP_MARGIN + row * LEVEL_GAP + NODE_HEIGHT / 2
        label = level_labels.get(level, f"Level {level}")
        level_texts.append(
            f'<text x="{LEFT_MARGIN - 28}" y="{y + 6:.1f}" class="level-label">{escape(f"Level {level}: {label}")}</text>'
        )

    node_elements = []
    for doc_id, meta in positions.items():
        x = meta["x"]
        y = meta["y"]
        node_elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" '
            f'rx="12" ry="12" class="node-box" data-docid="{escape(doc_id)}" />'
        )
        node_elements.append(
            f'<text x="{meta["cx"]:.1f}" y="{y + NODE_HEIGHT / 2 + 4:.1f}" class="node-label">{escape(doc_id)}</text>'
        )

    edge_elements = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        pair_counts[(edge["source"], edge["target"])] += 1

    pair_seen: dict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        src = positions[edge["source"]]
        tgt = positions[edge["target"]]
        pair_key = (edge["source"], edge["target"])
        total = pair_counts[pair_key]
        idx = pair_seen[pair_key]
        pair_seen[pair_key] += 1
        # If there are multiple relations for the same pair, separate lines visually.
        offset = (idx - (total - 1) / 2) * 6.0
        y1 = src["cy"] + offset
        y2 = tgt["cy"] + offset
        edge_elements.append(
            f'<line x1="{src["cx"]:.1f}" y1="{y1:.1f}" x2="{tgt["cx"]:.1f}" y2="{y2:.1f}" '
            f'stroke="{edge["color"]}" stroke-width="3" stroke-linecap="round" opacity="0.9" />'
        )

    year_labels = []
    for year, idx in x_map.items():
        x = LEFT_MARGIN + idx * YEAR_GAP
        year_labels.append(
            f'<text x="{x + NODE_WIDTH / 2:.1f}" y="{svg_height - BOTTOM_MARGIN / 2:.1f}" class="year-label">{escape(str(year))}</text>'
        )

    legend_items = (
        f'<span class="legend-dot" style="background:{COLOR_ENTAILMENT};"></span>Entailment',
        f'<span class="legend-dot" style="background:{COLOR_CONFLICT};"></span>Contradiction',
    )

    html = f"""
<div class="timeline-card">
  <div class="timeline-header">Relasi Dokumen</div>
  <div class="timeline-legend">{' '.join(f'<span>{item}</span>' for item in legend_items)}</div>
  <div class="timeline-wrapper">
    <svg viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" width="100%" height="{svg_height:.0f}">
      <rect width="100%" height="100%" fill="{COLOR_BG}" rx="18"/>
      {''.join(svg_lines)}
      {''.join(level_texts)}
      {''.join(edge_elements)}
      {''.join(node_elements)}
      {''.join(year_labels)}
    </svg>
  </div>
</div>
"""

    styles = """
<style>
.timeline-card {
    border: 1px solid #dbe3f3;
  background: #ffffff;
  border-radius: 18px;
    padding: 18px 18px 14px;
  margin-bottom: 1.5rem;
    box-shadow: 0 10px 24px rgba(15,23,42,0.06);
}
.timeline-header {
    font-size: 1.05rem;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 8px;
}
.timeline-legend {
  display: flex;
    gap: 16px;
    font-size: 0.82rem;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
  color: #475569;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  margin-right: 6px;
}
.timeline-wrapper {
  width: 100%;
  overflow-x: auto;
  padding-bottom: 6px;
}
.timeline-wrapper svg {
  border-radius: 14px;
}
.node-box {
    fill: __COLOR_NODE__;
    stroke: __COLOR_BORDER__;
    stroke-width: 1.25;
    filter: drop-shadow(0 2px 5px rgba(15,23,42,0.08));
}
.node-label {
    font-size: 12px;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-weight: 700;
    fill: #0f172a;
  text-anchor: middle;
}
.level-label {
    font-size: 0.85rem;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    fill: #334155;
    text-anchor: end;
  font-weight: 600;
}
.year-label {
    font-size: 0.9rem;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    fill: #334155;
  text-anchor: middle;
  font-weight: 600;
}
</style>
"""

    styles = styles.replace("__COLOR_NODE__", COLOR_NODE).replace("__COLOR_BORDER__", COLOR_BORDER)
    return styles + html
