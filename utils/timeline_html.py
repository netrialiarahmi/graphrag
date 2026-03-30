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

BASE_YEAR_GAP = 320
H_SPACING = 224  # Horizontal offset when multiple boxes share the same year+level
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
    # NOTE: permen* must come BEFORE pp to avoid false matches on "permenppn".
    prefix_rules = [
        (r"^uud\b", 1),
        (r"^tap\s*mpr\b", 2),
        (r"^perppu\b", 3),
        (r"^uu\b", 3),
        (r"^perpres\b", 5),
        (r"^keppres\b", 6),
        (r"^inpres\b", 7),
        (r"^permen", 8),
        (r"^pp\b", 4),
        (r"^perda", 9),
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


def _canonical_doc_id(doc_id: str) -> str:
    """Return a canonical form for alias deduplication.

    E.g. PERMEN-NASIONAL-7-2023 and PERMENPPN-NASIONAL-7-2023 should resolve
    to the same document.  We keep the *more specific* form (PERMENPPN).
    """
    return doc_id  # identity; dedup handled via _build_alias_map


def _build_alias_map(doc_ids: list[str]) -> dict[str, str]:
    """Map generic PERMEN-* IDs to their specific PERMEN<X>-* alias when present.

    Example:  If both PERMENPPN-NASIONAL-7-2023 and PERMEN-NASIONAL-7-2023 exist,
    the generic one is mapped to the specific one.
    """
    alias_map: dict[str, str] = {}  # generic → specific
    specifics: dict[str, str] = {}  # (suffix) → specific_id

    for did in doc_ids:
        upper = did.upper()
        # Match PERMEN<QUALIFIER>-<rest>  e.g. PERMENPPN-NASIONAL-7-2023
        m = re.match(r"^(PERMEN[A-Z]+)-(.*)", upper)
        if m and m.group(1) != "PERMEN":
            suffix = m.group(2)
            specifics[suffix] = did

    for did in doc_ids:
        upper = did.upper()
        m = re.match(r"^PERMEN-(.*)", upper)
        if m:
            suffix = m.group(1)
            if suffix in specifics:
                alias_map[did] = specifics[suffix]

    return alias_map


def build_timeline_html(csv_path: str) -> tuple[str, int] | None:
    df = _load_csv(csv_path)
    if df is None or df.empty:
        return None

    # Collect all unique doc IDs first, then build alias map for dedup.
    all_ids: set[str] = set()
    for col in ("doc_1", "doc_2"):
        all_ids.update(df[col].dropna().astype(str).unique())

    alias_map = _build_alias_map(list(all_ids))

    # Replace aliased IDs in the dataframe.
    for col in ("doc_1", "doc_2"):
        df[col] = df[col].map(lambda x: alias_map.get(str(x), str(x)) if pd.notna(x) else x)

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

    # Compute per-year column widths based on max stacked nodes in any level for that year.
    per_year_max_slots: dict[int, int] = defaultdict(int)
    per_year_level_counts: dict[tuple[int, int], int] = defaultdict(int)
    for did, info in docs.items():
        year = info.get("year")
        level = info.get("level", 11)
        if year is None:
            continue
        per_year_level_counts[(year, level)] += 1
    for year in years:
        max_slots = 0
        for level in {info.get("level", 11) for info in docs.values()}:
            max_slots = max(max_slots, per_year_level_counts.get((year, level), 0))
        per_year_max_slots[year] = max_slots

    col_widths = []
    for year in years:
        slots = per_year_max_slots.get(year, 1)
        extra = max(0, slots - 1) * H_SPACING
        col_widths.append(BASE_YEAR_GAP + extra)

    # Map year → x center using cumulative widths, keeping the year label centered.
    x_map: dict[int, float] = {}
    cursor = LEFT_MARGIN
    for year, col_w in zip(years, col_widths):
        center = cursor + col_w / 2
        x_map[year] = center
        cursor += col_w

    used_levels = sorted({info["level"] for info in docs.values()})
    level_to_row = {level: idx for idx, level in enumerate(used_levels)}

    positions: dict[str, dict] = {}
    slot_counts: dict[tuple[int, int], int] = defaultdict(int)
    row_count = len(used_levels)
    max_x = 0
    for doc_id, info in docs.items():
        year = info.get("year")
        level = info.get("level", 11)
        row = level_to_row.get(level, 0)
        slot = (year, row)
        offset = slot_counts[slot]
        slot_counts[slot] += 1

        year_center = x_map.get(year, LEFT_MARGIN)
        cluster_count = per_year_level_counts.get((year, level), 1)
        base_x = year_center + (offset - (cluster_count - 1) / 2) * H_SPACING - NODE_WIDTH / 2
        base_y = TOP_MARGIN + row * LEVEL_GAP
        max_x = max(max_x, base_x)
        positions[doc_id] = {
            "x": base_x,
            "y": base_y,
            "cx": base_x + NODE_WIDTH / 2,
            "cy": base_y + NODE_HEIGHT / 2,
            "year": year,
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
        reason = str(row.get("reasoning", "")).strip()
        edge_key = (a, b, color, reason)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append({"source": a, "target": b, "color": color, "reason": reason})

    svg_width = max(cursor + RIGHT_MARGIN, max_x + NODE_WIDTH + RIGHT_MARGIN)
    svg_height = TOP_MARGIN + max(0, row_count - 1) * LEVEL_GAP + NODE_HEIGHT + BOTTOM_MARGIN

    level_labels = {level: label for level, label, _ in VISUAL_LEVELS}

    svg_lines = []
    
    # Draw a line at the border of each year compartment
    x_starts: list[float] = []
    current_x = LEFT_MARGIN
    for col_w in col_widths:
        x_starts.append(current_x)
        current_x += col_w
    
    # Left borders of each year
    for x_start in x_starts:
        svg_lines.append(
            f'<line x1="{x_start:.1f}" y1="{TOP_MARGIN - 20:.1f}" x2="{x_start:.1f}" y2="{svg_height - BOTTOM_MARGIN + 20:.1f}" '
            f'stroke="#9ca3af" stroke-dasharray="4,4" stroke-width="1.5" />'
        )
    # Right border of the final year
    if x_starts:
        svg_lines.append(
            f'<line x1="{current_x:.1f}" y1="{TOP_MARGIN - 20:.1f}" x2="{current_x:.1f}" y2="{svg_height - BOTTOM_MARGIN + 20:.1f}" '
            f'stroke="#9ca3af" stroke-dasharray="4,4" stroke-width="1.5" />'
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
            f'<line class="edge-line" x1="{src["cx"]:.1f}" y1="{y1:.1f}" x2="{tgt["cx"]:.1f}" y2="{y2:.1f}" '
            f'stroke="{edge["color"]}" stroke-width="3" stroke-linecap="round" opacity="0.9" '
            f'data-source="{escape(edge["source"])}" data-target="{escape(edge["target"])}" data-reason="{escape(edge.get("reason", ""))}" />'
        )

    year_labels = []
    for year, x_center in x_map.items():
        year_labels.append(
            f'<text x="{x_center:.1f}" y="{svg_height - BOTTOM_MARGIN / 2:.1f}" class="year-label">{escape(str(year))}</text>'
        )

    legend_items = (
            f'<span class="legend-dot" style="background:{COLOR_ENTAILMENT};"></span>Entailment',
            f'<span class="legend-dot" style="background:{COLOR_CONFLICT};"></span>Contradiction',
    )

    html = f"""
<div class="timeline-card">
    <div class="timeline-header">Relasi Dokumen</div>
    <div class="timeline-legend">{' '.join(f'<span>{item}</span>' for item in legend_items)}</div>
    <div id="selection-bar" class="selection-bar" style="display:none;">
        <span id="sel-text"></span>
        <button id="sel-clear">Clear</button>
    </div>
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
    <div id="explanation-box" class="explanation-box" style="display:none;"></div>
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
    position: relative;
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
  pointer-events: none;
}
.edge-line { transition: opacity 0.15s ease, stroke-width 0.15s ease; }
.node-box, .node-label { transition: opacity 0.15s ease; }
.dim { opacity: 0.18; }

.tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(15,23,42,0.92);
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    max-width: 260px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.25);
    opacity: 0;
    transition: opacity 0.1s ease;
    z-index: 5;
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
.selection-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f1f5f9;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 0.9rem;
    color: #1e293b;
    border: 1px solid #cbd5e1;
}
.selection-bar button {
    background: #ef4444;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
}
.selection-bar button:hover {
    background: #dc2626;
}
.explanation-box {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 14px 18px;
    margin-top: 14px;
    border-radius: 6px;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: 0.95rem;
    color: #334155;
    line-height: 1.5;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.node-box {
    cursor: pointer;
}
.node-box.selected {
    stroke: #3b82f6;
    stroke-width: 3;
    filter: drop-shadow(0 0 8px rgba(59,130,246,0.6));
}
.dim { opacity: 0.15; }
</style>
"""

    scripts = """
<script>
(function(){
    const wrapper = document.querySelector('.timeline-wrapper');
    const svg = document.querySelector('svg');
    const selBar = document.getElementById('selection-bar');
    const selText = document.getElementById('sel-text');
    const selClear = document.getElementById('sel-clear');
    const expBox = document.getElementById('explanation-box');
    
    if(!wrapper || !svg) return;
    
    const edges = svg.querySelectorAll('.edge-line');
    const boxes = svg.querySelectorAll('.node-box');
    const labels = svg.querySelectorAll('.node-label');
    
    let selectedNodes = [];

    function updateSelection() {
        // Clear highlights
        boxes.forEach(b => b.classList.remove('selected', 'dim'));
        edges.forEach(e => e.classList.remove('dim'));
        labels.forEach(l => l.classList.remove('dim'));
        expBox.style.display = 'none';
        
        if (selectedNodes.length === 0) {
            selBar.style.display = 'none';
            return;
        }

        selBar.style.display = 'flex';
        selText.innerHTML = `<strong>Terpilih:</strong> ${selectedNodes.join(' dan ')}`;

        // Highlight selected
        boxes.forEach(b => {
            const id = b.getAttribute('data-docid');
            if (selectedNodes.includes(id)) {
                b.classList.add('selected');
                b.classList.remove('dim');
            } else {
                b.classList.add('dim');
            }
        });
        
        labels.forEach(l => {
            const text = l.textContent.trim();
            if (!selectedNodes.includes(text)) {
                l.classList.add('dim');
            }
        });

        if (selectedNodes.length === 2) {
            // Find edge
            let foundEdge = null;
            edges.forEach(e => {
                const src = e.getAttribute('data-source');
                const tgt = e.getAttribute('data-target');
                if ((src === selectedNodes[0] && tgt === selectedNodes[1]) || 
                    (src === selectedNodes[1] && tgt === selectedNodes[0])) {
                    foundEdge = e;
                    e.classList.remove('dim');
                } else {
                    e.classList.add('dim');
                }
            });

            if (foundEdge) {
                const reason = foundEdge.getAttribute('data-reason') || 'Tidak ada penjelasan lebih lanjut.';
                const edgeColor = foundEdge.getAttribute('stroke');
                const type = edgeColor === '#dc2626' ? 'Kontradiksi' : 'Entailment (Mendukung)'; // #dc2626 is COLOR_CONFLICT
                
                expBox.style.display = 'block';
                expBox.innerHTML = `<strong>Relasi: ${type}</strong><br/>${reason}`;
            } else {
                expBox.style.display = 'block';
                expBox.innerHTML = `<em>Tidak ada relasi langsung ditemukan antara ${selectedNodes[0]} dan ${selectedNodes[1]}.</em>`;
            }
        } else {
            // Show all if not exactly 2 selected
            if (selectedNodes.length === 0) {
                clearSelection();
            }
        }
    }

    function clearSelection() {
        selectedNodes = [];
        selBar.style.display = 'none';
        expBox.style.display = 'none';
        boxes.forEach(b => b.classList.remove('selected', 'dim'));
        edges.forEach(e => e.classList.remove('dim'));
        labels.forEach(l => l.classList.remove('dim'));
    }

    boxes.forEach(box => {
        box.addEventListener('click', () => {
            const id = box.getAttribute('data-docid');
            if (selectedNodes.includes(id)) {
                selectedNodes = selectedNodes.filter(n => n !== id);
            } else {
                if (selectedNodes.length >= 2) {
                    selectedNodes = [id];
                } else {
                    selectedNodes.push(id);
                }
            }
            updateSelection();
        });
    });

    selClear.addEventListener('click', clearSelection);
})();
</script>
"""

    styles = styles.replace("__COLOR_NODE__", COLOR_NODE).replace("__COLOR_BORDER__", COLOR_BORDER)
    # Return HTML and the precise height to allow iframe resizing
    total_height = int(svg_height) + 120
    return styles + html + scripts, total_height
