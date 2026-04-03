import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pdf_loader import load_all_pdfs
from preprocessing.chunker import chunk_documents

# ── Konstanta ─────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent.parent / "docs"
WORD_MIN_FLAG = 30
WORD_MAX_FLAG = 400
PAGE_SIZE = 20

METHOD_COLORS = {
    "pasal": "#4C72B0",
    "subsection": "#55A868",
    "small_doc": "#C44E52",
    "recursive": "#DD8452",
}

DOC_LEVEL_LABEL = {1: "L1 — Universitas", 2: "L2 — Dekan/PU", 3: "L3 — Panduan Teknis"}


# ── Load & cache ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Membaca dan memproses semua PDF...")
def load_chunks() -> pd.DataFrame:
    documents = load_all_pdfs(str(DOCS_DIR))
    chunks = chunk_documents(documents)
    df = pd.DataFrame(chunks)
    df["preview"] = df["content"].apply(
        lambda x: " ".join(x.split()[:50]) + ("..." if len(x.split()) > 50 else "")
    )
    df["doc_level_label"] = df["doc_level"].map(DOC_LEVEL_LABEL)
    return df, documents


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chunk Inspector — Agentic RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Chunk Inspector")
st.caption("Verifikasi hasil chunking sebelum proses embedding ke ChromaDB.")

result = load_chunks()
df, documents = result

if df.empty:
    st.error(f"Tidak ada PDF di `{DOCS_DIR}`.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filter")

    sumber_list = ["Semua"] + sorted(df["source"].unique().tolist())
    pilih_sumber = st.selectbox("Dokumen", sumber_list)

    method_opts = ["Semua"] + sorted(df["chunk_method"].unique().tolist())
    pilih_method = st.selectbox("Chunk Method", method_opts)

    level_opts = ["Semua"] + sorted(df["doc_level_label"].unique().tolist())
    pilih_level = st.selectbox("Doc Level", level_opts)

    keyword = st.text_input("Cari keyword dalam konten", placeholder="SKS, IPK, sidang...")

    min_wc = int(df["word_count"].min())
    max_wc = int(df["word_count"].max())
    wc_range = st.slider("Range Word Count", min_value=min_wc, max_value=max_wc, value=(min_wc, max_wc))

    st.divider()
    st.caption(f"Total PDF: **{df['source'].nunique()}**")
    st.caption(f"Total chunk (before filter): **{len(df)}**")

# ── Terapkan filter ───────────────────────────────────────────────────────────

filtered = df.copy()
if pilih_sumber != "Semua":
    filtered = filtered[filtered["source"] == pilih_sumber]
if pilih_method != "Semua":
    filtered = filtered[filtered["chunk_method"] == pilih_method]
if pilih_level != "Semua":
    filtered = filtered[filtered["doc_level_label"] == pilih_level]
if keyword:
    filtered = filtered[filtered["content"].str.contains(keyword, case=False, na=False)]
filtered = filtered[
    (filtered["word_count"] >= wc_range[0]) & (filtered["word_count"] <= wc_range[1])
]

# ── Row 1 — Strategy summary per dokumen ─────────────────────────────────────

st.subheader("Strategi per Dokumen")

strategy_rows = []
for doc in documents:
    src = doc["source"]
    doc_chunks = df[df["source"] == src]
    if doc_chunks.empty:
        continue
    method = doc_chunks["chunk_method"].mode()[0]
    strategy_rows.append({
        "Dokumen": src[:65],
        "Level": DOC_LEVEL_LABEL.get(doc["doc_level"], "?"),
        "Tanggal": doc["tanggal_dokumen"],
        "Strategi": method.upper(),
        "Total Chunk": len(doc_chunks),
        "Avg Word Count": f"{doc_chunks['word_count'].mean():.0f}",
    })

st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Row 2 — Metrics ───────────────────────────────────────────────────────────

st.subheader("Ringkasan (Setelah Filter)")

total = len(filtered)
avg_wc = filtered["word_count"].mean() if total > 0 else 0
n_short = (filtered["word_count"] < WORD_MIN_FLAG).sum()
n_long = (filtered["word_count"] > WORD_MAX_FLAG).sum()

method_counts = filtered["chunk_method"].value_counts()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Chunk", total)
c2.metric("Avg Word Count", f"{avg_wc:.0f}")
c3.metric("⚠️ < 30 kata", n_short)
c4.metric("⚠️ > 400 kata", n_long)

method_cols = st.columns(len(method_counts) or 1)
for col, (method, count) in zip(method_cols, method_counts.items()):
    col.metric(f"Method: {method}", f"{count} ({count/total*100:.0f}%)" if total else "0")

st.divider()

# ── Row 3 — Charts ────────────────────────────────────────────────────────────

col_bar, col_hist = st.columns(2)

with col_bar:
    st.subheader("Chunk per Dokumen")
    if not filtered.empty:
        chart = (
            filtered.groupby("source")["chunk_id"]
            .count()
            .reset_index()
            .rename(columns={"chunk_id": "jumlah"})
            .sort_values("jumlah", ascending=False)
        )
        chart["source"] = chart["source"].str[:40]
        st.bar_chart(chart.set_index("source")["jumlah"])

with col_hist:
    st.subheader("Distribusi Word Count")
    if not filtered.empty:
        import altair as alt

        hist_df = filtered[["word_count", "chunk_method"]].copy()
        bins = list(range(0, max_wc + 50, 25))
        hist_df["bin"] = pd.cut(hist_df["word_count"], bins=bins, right=False)
        hist_df["bin_str"] = hist_df["bin"].apply(
            lambda x: f"{int(x.left)}-{int(x.right)}" if pd.notna(x) else "other"
        )
        hist_df["bin_left"] = hist_df["bin"].apply(
            lambda x: int(x.left) if pd.notna(x) else 0
        )
        hist_agg = (
            hist_df.groupby(["bin_str", "bin_left", "chunk_method"])
            .size()
            .reset_index(name="count")
            .sort_values("bin_left")
        )
        chart = (
            alt.Chart(hist_agg)
            .mark_bar()
            .encode(
                x=alt.X("bin_str:N", sort=alt.EncodingSortField(field="bin_left"), title="Word Count"),
                y=alt.Y("count:Q", title="Jumlah Chunk"),
                color=alt.Color("chunk_method:N", title="Method"),
                tooltip=["bin_str", "chunk_method", "count"],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"🔴 < {WORD_MIN_FLAG} kata = terlalu pendek  |  🟡 > {WORD_MAX_FLAG} kata = perlu perhatian")

st.divider()

# ── Row 4 — Tabel chunk ───────────────────────────────────────────────────────

st.subheader("Daftar Chunk")

if not filtered.empty:
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    pg_col, info_col = st.columns([1, 3])
    with pg_col:
        current_page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
    with info_col:
        st.caption(f"Halaman {current_page}/{total_pages} — {len(filtered)} chunk")

    start = (current_page - 1) * PAGE_SIZE
    page_df = filtered.iloc[start : start + PAGE_SIZE].reset_index(drop=True)

    def _highlight(row):
        if row["word_count"] < WORD_MIN_FLAG:
            return ["background-color:#ffe0e0"] * len(row)
        if row["word_count"] > WORD_MAX_FLAG:
            return ["background-color:#fff8d6"] * len(row)
        return [""] * len(row)

    display_cols = [
        "chunk_id", "source", "halaman", "doc_level_label",
        "chunk_method", "section", "word_count", "preview",
    ]
    # Hanya tampilkan kolom yang ada
    display_cols = [c for c in display_cols if c in page_df.columns]

    styled = page_df[display_cols].style.apply(_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada chunk yang cocok dengan filter.")

st.divider()

# ── Row 5 — Detail viewer ─────────────────────────────────────────────────────

st.subheader("Detail Chunk")

if not filtered.empty:
    selected_id = st.selectbox(
        "Pilih Chunk ID",
        options=filtered["chunk_id"].tolist(),
        format_func=lambda x: f"Chunk #{x}",
    )
    row = filtered[filtered["chunk_id"] == selected_id].iloc[0]

    left, right = st.columns([2, 1])

    with left:
        st.markdown("**Konten:**")
        st.text_area("", value=row["content"], height=320, disabled=True, label_visibility="collapsed")

    with right:
        st.markdown("**Metadata:**")
        meta_fields = {
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "halaman": row["halaman"],
            "tanggal_dokumen": row["tanggal_dokumen"],
            "doc_level": f"{row['doc_level']} — {row['doc_level_label']}",
            "chunk_method": row["chunk_method"],
            "section": row.get("section", ""),
            "word_count": row["word_count"],
        }
        for k, v in meta_fields.items():
            st.markdown(f"- **{k}:** `{v}`")

        wc = row["word_count"]
        if wc < WORD_MIN_FLAG:
            st.error(f"Terlalu pendek ({wc} kata)")
        elif wc > WORD_MAX_FLAG:
            st.warning(f"Cukup panjang ({wc} kata) — perhatikan konteks")
        else:
            st.success(f"Word count OK ({wc} kata)")
