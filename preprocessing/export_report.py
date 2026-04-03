"""
export_report.py — Generate static HTML report hasil chunking.

Jalankan:
    python preprocessing/export_report.py

Output:
    chunk_report.html (standalone, tidak butuh internet atau Python)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pdf_loader import load_all_pdfs
from preprocessing.chunker import chunk_documents, _detect_strategy

# ── Konstanta ──────────────────────────────────────────────────────────────────

DOCS_DIR  = Path(__file__).parent.parent / "docs"
OUTPUT    = Path(__file__).parent.parent / "chunk_report.html"

WORD_MIN_FLAG = 30
WORD_MAX_FLAG = 400

DOC_LEVEL_LABEL = {1: "L1 — Universitas", 2: "L2 — Dekan / PU", 3: "L3 — Panduan Teknis"}

METHOD_COLOR = {
    "pasal":      "#4C72B0",
    "subsection": "#55A868",
    "small_doc":  "#C44E52",
    "recursive":  "#DD8452",
}

STRATEGY_DESC = {
    "pasal":      "Split per Pasal — satu chunk = satu Pasal + semua ayatnya",
    "subsection": "Split per Sub-section — satu chunk = satu sub-judul bernomor (1.1, BAB)",
    "small_doc":  "Split per Section Letter — dokumen kecil, satu chunk per A/B/C",
    "recursive":  "Recursive Character Splitter — fallback untuk dokumen tanpa struktur jelas",
}

# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    print("Loading PDFs...")
    documents = load_all_pdfs(str(DOCS_DIR))
    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    return documents, chunks


# ── Chart builders ─────────────────────────────────────────────────────────────

def chart_chunks_per_doc(chunks) -> str:
    by_doc = defaultdict(int)
    by_doc_method = defaultdict(lambda: defaultdict(int))
    for c in chunks:
        label = c["source"][:45] + ("…" if len(c["source"]) > 45 else "")
        by_doc[label] += 1
        by_doc_method[label][c["chunk_method"]] += 1

    docs_sorted = sorted(by_doc.items(), key=lambda x: -x[1])
    labels = [d[0] for d in docs_sorted]

    methods = list(METHOD_COLOR.keys())
    fig = go.Figure()
    for method in methods:
        values = [by_doc_method[lbl].get(method, 0) for lbl in labels]
        if any(v > 0 for v in values):
            fig.add_trace(go.Bar(
                name=method,
                x=values,
                y=labels,
                orientation="h",
                marker_color=METHOD_COLOR[method],
                hovertemplate="%{y}<br>%{x} chunk<extra>" + method + "</extra>",
            ))

    fig.update_layout(
        barmode="stack",
        title="Jumlah Chunk per Dokumen",
        xaxis_title="Jumlah Chunk",
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_word_count_dist(chunks) -> str:
    wcs    = [c["word_count"] for c in chunks]
    labels = [c["chunk_method"] for c in chunks]

    fig = go.Figure()
    for method, color in METHOD_COLOR.items():
        method_wcs = [w for w, l in zip(wcs, labels) if l == method]
        if not method_wcs:
            continue
        fig.add_trace(go.Histogram(
            x=method_wcs,
            name=method,
            marker_color=color,
            opacity=0.8,
            xbins=dict(size=25),
            hovertemplate="Word count: %{x}<br>Jumlah: %{y}<extra>" + method + "</extra>",
        ))

    fig.add_vline(x=WORD_MIN_FLAG, line_dash="dash", line_color="red",
                  annotation_text=f"Min {WORD_MIN_FLAG}", annotation_position="top right")
    fig.add_vline(x=WORD_MAX_FLAG, line_dash="dash", line_color="orange",
                  annotation_text=f"Maks {WORD_MAX_FLAG}", annotation_position="top right")

    fig.update_layout(
        barmode="stack",
        title="Distribusi Word Count per Chunk",
        xaxis_title="Word Count",
        yaxis_title="Jumlah Chunk",
        height=320,
        margin=dict(l=10, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_method_pie(chunks) -> str:
    counts = Counter(c["chunk_method"] for c in chunks)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [METHOD_COLOR.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} chunk (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Distribusi Chunk Method",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── HTML builders ──────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", color: str = "#4C72B0") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>"""


def strategy_table(documents, chunks) -> str:
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["source"]].append(c)

    rows = ""
    for doc in documents:
        src = doc["source"]
        dc = by_doc.get(src, [])
        if not dc:
            continue
        strategy = _detect_strategy(doc["full_text"])
        method_counts = Counter(c["chunk_method"] for c in dc)
        dominant = method_counts.most_common(1)[0][0]
        avg_wc = sum(c["word_count"] for c in dc) / len(dc)
        badge_color = METHOD_COLOR.get(dominant, "#888")
        level_label = DOC_LEVEL_LABEL.get(doc["doc_level"], "?")
        level_badge = f'<span class="badge badge-level{doc["doc_level"]}">{level_label}</span>'
        method_badge = f'<span class="badge" style="background:{badge_color}">{dominant}</span>'
        desc = STRATEGY_DESC.get(strategy, "")
        rows += f"""
        <tr>
            <td class="doc-name">{src}</td>
            <td>{level_badge}</td>
            <td>{doc["tanggal_dokumen"]}</td>
            <td>{method_badge}<br><small style="color:#666">{desc}</small></td>
            <td class="num">{len(dc)}</td>
            <td class="num">{avg_wc:.0f}</td>
        </tr>"""

    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Dokumen</th>
                <th>Level</th>
                <th>Tanggal</th>
                <th>Strategi Chunking</th>
                <th>Total Chunk</th>
                <th>Avg Word Count</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def chunk_table(chunks) -> str:
    rows = ""
    for c in chunks:
        wc = c["word_count"]
        row_class = "row-short" if wc < WORD_MIN_FLAG else ("row-long" if wc > WORD_MAX_FLAG else "")
        badge_color = METHOD_COLOR.get(c["chunk_method"], "#888")
        method_badge = f'<span class="badge" style="background:{badge_color}">{c["chunk_method"]}</span>'
        level_badge  = f'<span class="badge badge-level{c["doc_level"]}">L{c["doc_level"]}</span>'
        preview = " ".join(c["content"].split()[:40]) + ("…" if len(c["content"].split()) > 40 else "")
        section = c.get("section", "")[:60]
        rows += f"""
        <tr class="{row_class}" data-content="{c['content'].replace('"', '&quot;').replace(chr(10), ' ')}"
            data-source="{c['source']}" data-section="{section}"
            data-halaman="{c['halaman']}" data-tanggal="{c['tanggal_dokumen']}"
            data-method="{c['chunk_method']}" data-level="{c['doc_level']}"
            data-wc="{wc}" onclick="showDetail(this)">
            <td class="num">{c["chunk_id"]}</td>
            <td>{level_badge}</td>
            <td class="doc-name">{c["source"][:40]}…</td>
            <td class="num">{c["halaman"]}</td>
            <td>{method_badge}</td>
            <td class="num">{wc}</td>
            <td class="preview">{preview}</td>
        </tr>"""

    return f"""
    <table class="data-table" id="chunk-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Level</th>
                <th>Dokumen</th>
                <th>Hal.</th>
                <th>Method</th>
                <th>WC</th>
                <th>Preview Konten</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


# ── Full HTML ──────────────────────────────────────────────────────────────────

def build_html(documents, chunks) -> str:
    all_wc = [c["word_count"] for c in chunks]
    n_short = sum(1 for w in all_wc if w < WORD_MIN_FLAG)
    n_long  = sum(1 for w in all_wc if w > WORD_MAX_FLAG)
    dupes   = sum(1 for v in Counter(c["content"] for c in chunks).values() if v > 1)
    avg_wc  = sum(all_wc) / len(all_wc)
    method_counts = Counter(c["chunk_method"] for c in chunks)

    # Charts
    chart_bar  = chart_chunks_per_doc(chunks)
    chart_hist = chart_word_count_dist(chunks)
    chart_pie  = chart_method_pie(chunks)

    # Tables
    strat_tbl = strategy_table(documents, chunks)
    c_tbl     = chunk_table(chunks)

    # Method metrics
    method_cards = ""
    for m, cnt in method_counts.most_common():
        method_cards += metric_card(
            f"Method: {m}", str(cnt),
            f"{cnt/len(chunks)*100:.0f}% dari total",
            METHOD_COLOR.get(m, "#888")
        )

    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chunk Inspector Report — Agentic RAG</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Inter", "Segoe UI", sans-serif; background: #f0f2f6; color: #1f2937; font-size: 14px; }}

  .topbar {{ background: #262730; color: white; padding: 14px 32px; display: flex; align-items: center; gap: 12px; }}
  .topbar h1 {{ font-size: 20px; font-weight: 600; }}
  .topbar .sub {{ font-size: 13px; color: #aaa; margin-left: auto; }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 28px; }}

  .section-title {{ font-size: 18px; font-weight: 600; margin: 28px 0 12px; padding-bottom: 6px;
                    border-bottom: 2px solid #e5e7eb; color: #111827; }}

  /* Metrics */
  .metrics-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 4px; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 16px 22px; flex: 1; min-width: 140px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-top: 3px solid #e5e7eb; }}
  .metric-card:first-child {{ border-top-color: #4C72B0; }}
  .metric-value {{ font-size: 28px; font-weight: 700; line-height: 1.1; }}
  .metric-label {{ font-size: 12px; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .metric-sub {{ font-size: 11px; color: #9ca3af; margin-top: 2px; }}

  /* Charts */
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .charts-row-3 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin: 16px 0; }}
  .chart-card {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}

  /* Tables */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ background: #f9fafb; padding: 10px 12px; text-align: left; font-weight: 600;
                    border-bottom: 2px solid #e5e7eb; color: #374151; white-space: nowrap; }}
  .data-table td {{ padding: 9px 12px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }}
  .data-table tr:hover {{ background: #f0f9ff; cursor: pointer; }}
  .data-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .doc-name {{ font-size: 12px; word-break: break-word; max-width: 300px; }}
  .preview {{ color: #4b5563; font-size: 12px; max-width: 380px; }}
  .row-short {{ background: #fff5f5; }}
  .row-long  {{ background: #fffbeb; }}
  .table-wrap {{ background: white; border-radius: 10px; overflow: auto; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: 11px;
            font-weight: 600; color: white; white-space: nowrap; }}
  .badge-level1 {{ background: #7c3aed; }}
  .badge-level2 {{ background: #0369a1; }}
  .badge-level3 {{ background: #047857; }}

  /* Search */
  .search-bar {{ display: flex; gap: 10px; margin-bottom: 12px; align-items: center; }}
  .search-bar input, .search-bar select {{ padding: 8px 12px; border: 1px solid #d1d5db;
    border-radius: 8px; font-size: 13px; outline: none; background: white; }}
  .search-bar input {{ flex: 1; }}
  .search-bar input:focus {{ border-color: #4C72B0; box-shadow: 0 0 0 3px rgba(76,114,176,0.15); }}

  /* Detail panel */
  .detail-panel {{ display: none; background: white; border-radius: 10px; padding: 20px;
                   box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-top: 16px; }}
  .detail-panel.active {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
  .detail-content {{ white-space: pre-wrap; font-size: 13px; background: #f9fafb; padding: 14px;
                     border-radius: 8px; border: 1px solid #e5e7eb; max-height: 360px; overflow-y: auto;
                     line-height: 1.6; color: #1f2937; }}
  .detail-meta {{ font-size: 13px; }}
  .detail-meta table {{ width: 100%; }}
  .detail-meta td {{ padding: 5px 0; vertical-align: top; }}
  .detail-meta td:first-child {{ color: #6b7280; font-weight: 500; padding-right: 10px; white-space: nowrap; }}
  .detail-meta td:last-child {{ color: #111827; word-break: break-word; }}
  .close-btn {{ float: right; cursor: pointer; color: #9ca3af; font-size: 18px; line-height: 1; }}
  .close-btn:hover {{ color: #374151; }}

  /* Footer */
  .footer {{ text-align: center; color: #9ca3af; font-size: 12px; padding: 24px 0; margin-top: 24px;
             border-top: 1px solid #e5e7eb; }}

  /* Highlight selected row */
  .selected-row {{ background: #dbeafe !important; }}

  /* Legend for table */
  .table-legend {{ display: flex; gap: 16px; font-size: 12px; color: #6b7280; margin-bottom: 8px; }}
  .legend-dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>

<div class="topbar">
  <span style="font-size:22px">🔍</span>
  <h1>Chunk Inspector Report — Agentic RAG</h1>
  <span class="sub">Sistem FAQ Akademik Telkom University &nbsp;|&nbsp; Generated: {generated_at}</span>
</div>

<div class="container">

  <!-- ── Strategi per Dokumen ── -->
  <div class="section-title">Strategi Chunking per Dokumen</div>
  <div class="table-wrap">{strat_tbl}</div>

  <!-- ── Global Metrics ── -->
  <div class="section-title">Ringkasan Global</div>
  <div class="metrics-row">
    {metric_card("Total Chunk", str(len(chunks)))}
    {metric_card("Avg Word Count", f"{avg_wc:.0f}", "kata per chunk")}
    {metric_card("Duplikat", str(dupes), "konten identik", "#6b7280")}
    {metric_card("Chunk &lt; 30 kata", str(n_short), "terlalu pendek", "#dc2626" if n_short > 0 else "#16a34a")}
    {metric_card("Chunk &gt; 400 kata", str(n_long), "perlu perhatian", "#d97706" if n_long > 0 else "#16a34a")}
    {method_cards}
  </div>

  <!-- ── Charts ── -->
  <div class="section-title">Visualisasi</div>
  <div class="charts-row-3">
    <div class="chart-card">{chart_bar}</div>
    <div class="chart-card">{chart_pie}</div>
  </div>
  <div class="chart-card">{chart_hist}</div>

  <!-- ── Chunk Table ── -->
  <div class="section-title">Daftar Chunk</div>

  <div class="search-bar">
    <input type="text" id="search-input" placeholder="Cari keyword dalam konten, dokumen, atau section..." oninput="filterTable()">
    <select id="filter-method" onchange="filterTable()">
      <option value="">Semua Method</option>
      <option value="pasal">pasal</option>
      <option value="subsection">subsection</option>
      <option value="small_doc">small_doc</option>
      <option value="recursive">recursive</option>
    </select>
    <select id="filter-level" onchange="filterTable()">
      <option value="">Semua Level</option>
      <option value="1">L1 — Universitas</option>
      <option value="2">L2 — Dekan/PU</option>
      <option value="3">L3 — Panduan Teknis</option>
    </select>
  </div>

  <div class="table-legend">
    <span><span class="legend-dot" style="background:#fff5f5;border:1px solid #fca5a5"></span> Chunk &lt; 30 kata (terlalu pendek)</span>
    <span><span class="legend-dot" style="background:#fffbeb;border:1px solid #fcd34d"></span> Chunk &gt; 400 kata (panjang)</span>
    <span style="color:#4b5563">Klik baris untuk melihat konten lengkap</span>
  </div>

  <div class="table-wrap" style="max-height:520px;overflow-y:auto" id="table-wrap">
    {c_tbl}
  </div>
  <div id="row-count" style="font-size:12px;color:#6b7280;margin-top:6px"></div>

  <!-- ── Detail Panel ── -->
  <div class="detail-panel" id="detail-panel">
    <div>
      <strong style="font-size:14px">Konten Chunk</strong>
      <pre class="detail-content" id="detail-content"></pre>
    </div>
    <div class="detail-meta">
      <span class="close-btn" onclick="closeDetail()">✕</span>
      <strong style="font-size:14px">Metadata</strong>
      <table id="detail-meta-table" style="margin-top:10px"></table>
    </div>
  </div>

</div>

<div class="footer">
  Agentic RAG — Student FAQ System &nbsp;|&nbsp; Fakultas Informatika, Telkom University<br>
  Laporan ini di-generate otomatis dari pipeline preprocessing
</div>

<script>
// ── Filter table ────────────────────────────────────────────────────────────
function filterTable() {{
  const kw     = document.getElementById("search-input").value.toLowerCase();
  const method = document.getElementById("filter-method").value;
  const level  = document.getElementById("filter-level").value;
  const rows   = document.querySelectorAll("#chunk-table tbody tr");
  let visible  = 0;

  rows.forEach(row => {{
    const content = (row.dataset.content || "").toLowerCase();
    const src     = (row.dataset.source  || "").toLowerCase();
    const section = (row.dataset.section || "").toLowerCase();
    const rowMethod = row.dataset.method || "";
    const rowLevel  = row.dataset.level  || "";

    const kwMatch  = !kw     || content.includes(kw) || src.includes(kw) || section.includes(kw);
    const mMatch   = !method || rowMethod === method;
    const lMatch   = !level  || rowLevel  === level;

    const show = kwMatch && mMatch && lMatch;
    row.style.display = show ? "" : "none";
    if (show) visible++;
  }});

  document.getElementById("row-count").textContent =
    `Menampilkan ${{visible}} dari ${{rows.length}} chunk`;
}}

// ── Detail viewer ───────────────────────────────────────────────────────────
let selectedRow = null;

function showDetail(row) {{
  if (selectedRow) selectedRow.classList.remove("selected-row");
  row.classList.add("selected-row");
  selectedRow = row;

  document.getElementById("detail-content").textContent = row.dataset.content;

  const wc = parseInt(row.dataset.wc);
  let wcNote = wc < {WORD_MIN_FLAG}
    ? `<span style="color:#dc2626">⚠️ Terlalu pendek (${{wc}} kata)</span>`
    : wc > {WORD_MAX_FLAG}
    ? `<span style="color:#d97706">⚠️ Cukup panjang (${{wc}} kata)</span>`
    : `<span style="color:#16a34a">✅ OK (${{wc}} kata)</span>`;

  const levelLabels = {{"1":"L1 — Universitas","2":"L2 — Dekan/PU","3":"L3 — Panduan Teknis"}};

  document.getElementById("detail-meta-table").innerHTML = `
    <tr><td>chunk_id</td><td>${{row.cells[0].textContent}}</td></tr>
    <tr><td>source</td><td>${{row.dataset.source}}</td></tr>
    <tr><td>halaman</td><td>${{row.dataset.halaman}}</td></tr>
    <tr><td>tanggal_dokumen</td><td>${{row.dataset.tanggal}}</td></tr>
    <tr><td>doc_level</td><td>${{levelLabels[row.dataset.level] || row.dataset.level}}</td></tr>
    <tr><td>chunk_method</td><td>${{row.dataset.method}}</td></tr>
    <tr><td>section</td><td>${{row.dataset.section || "—"}}</td></tr>
    <tr><td>word_count</td><td>${{wcNote}}</td></tr>
  `;

  const panel = document.getElementById("detail-panel");
  panel.classList.add("active");
  panel.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
}}

function closeDetail() {{
  document.getElementById("detail-panel").classList.remove("active");
  if (selectedRow) {{
    selectedRow.classList.remove("selected-row");
    selectedRow = null;
  }}
}}

// Init row count
window.addEventListener("load", () => {{
  const total = document.querySelectorAll("#chunk-table tbody tr").length;
  document.getElementById("row-count").textContent = `Menampilkan ${{total}} dari ${{total}} chunk`;
}});
</script>

</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    documents, chunks = load_data()
    print("Building HTML report...")
    html = build_html(documents, chunks)
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"✅ Report saved: {OUTPUT}")
    print(f"   Size: {size_kb:.0f} KB")
    print(f"   Chunks: {len(chunks)}")
    print(f"\nBuka di browser: open {OUTPUT}")
