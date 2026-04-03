import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Supaya bisa import dari root project
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pdf_loader import load_all_pdfs
from preprocessing.chunker import chunk_pages

# ── Konstanta ────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent.parent / "docs"
WORD_COUNT_MIN_FLAG = 30    # chunk terlalu pendek (merah)
WORD_COUNT_MAX_FLAG = 400   # chunk terlalu panjang (kuning)
PAGE_SIZE = 20              # jumlah chunk per halaman tabel


# ── Load & cache data ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Membaca dan memproses PDF...")
def load_chunks() -> pd.DataFrame:
    pages = load_all_pdfs(str(DOCS_DIR))
    chunks = chunk_pages(pages)
    df = pd.DataFrame(chunks)
    df["preview"] = df["content"].apply(
        lambda x: " ".join(x.split()[:50]) + ("..." if len(x.split()) > 50 else "")
    )
    return df


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chunk Inspector — Agentic RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Chunk Inspector")
st.caption("Verifikasi hasil chunking sebelum proses embedding ke ChromaDB.")

df = load_chunks()

if df.empty:
    st.error(f"Tidak ada PDF ditemukan di `{DOCS_DIR}`. Pastikan folder `docs/` berisi file PDF.")
    st.stop()

# ── Sidebar — Filter ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filter")

    # Filter dokumen
    sumber_list = ["Semua"] + sorted(df["source"].unique().tolist())
    pilih_sumber = st.selectbox("Dokumen", sumber_list)

    # Filter chunk method
    method_options = ["Semua", "pasal", "recursive"]
    pilih_method = st.selectbox("Chunk Method", method_options)

    # Search keyword
    keyword = st.text_input("Cari keyword dalam konten", placeholder="contoh: SKS, IPK, wisuda...")

    # Slider word count
    min_wc = int(df["word_count"].min())
    max_wc = int(df["word_count"].max())
    wc_range = st.slider(
        "Range Word Count",
        min_value=min_wc,
        max_value=max_wc,
        value=(min_wc, max_wc),
    )

    st.divider()
    st.caption(f"Total PDF: **{df['source'].nunique()}** file")
    st.caption(f"Total chunk (sebelum filter): **{len(df)}**")

# ── Terapkan filter ───────────────────────────────────────────────────────────

filtered = df.copy()

if pilih_sumber != "Semua":
    filtered = filtered[filtered["source"] == pilih_sumber]

if pilih_method != "Semua":
    filtered = filtered[filtered["chunk_method"] == pilih_method]

if keyword:
    filtered = filtered[
        filtered["content"].str.contains(keyword, case=False, na=False)
    ]

filtered = filtered[
    (filtered["word_count"] >= wc_range[0]) & (filtered["word_count"] <= wc_range[1])
]

# ── Row 1 — Metrics ───────────────────────────────────────────────────────────

st.subheader("Ringkasan")

total = len(filtered)
avg_wc = filtered["word_count"].mean() if total > 0 else 0
n_pasal = (filtered["chunk_method"] == "pasal").sum()
n_recursive = (filtered["chunk_method"] == "recursive").sum()
pct_pasal = (n_pasal / total * 100) if total > 0 else 0
pct_recursive = (n_recursive / total * 100) if total > 0 else 0
n_short = (filtered["word_count"] < WORD_COUNT_MIN_FLAG).sum()
n_long = (filtered["word_count"] > WORD_COUNT_MAX_FLAG).sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Chunk", total)
col2.metric("Avg Word Count", f"{avg_wc:.0f}")
col3.metric("Method: Pasal", f"{n_pasal} ({pct_pasal:.0f}%)")
col4.metric("Method: Recursive", f"{n_recursive} ({pct_recursive:.0f}%)")
col5.metric("⚠️ Terlalu Pendek (<30)", n_short)
col6.metric("⚠️ Terlalu Panjang (>400)", n_long)

st.divider()

# ── Row 2 — Bar chart chunk per dokumen ──────────────────────────────────────

st.subheader("Jumlah Chunk per Dokumen")

if not filtered.empty:
    chart_data = (
        filtered.groupby("source")["chunk_id"]
        .count()
        .reset_index()
        .rename(columns={"chunk_id": "jumlah_chunk"})
        .sort_values("jumlah_chunk", ascending=False)
    )
    st.bar_chart(chart_data.set_index("source")["jumlah_chunk"])
else:
    st.info("Tidak ada data untuk ditampilkan.")

st.divider()

# ── Row 3 — Histogram word count ─────────────────────────────────────────────

st.subheader("Distribusi Word Count")

if not filtered.empty:
    import altair as alt

    hist_df = filtered[["word_count", "chunk_method"]].copy()

    # Buat bins manual
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

    color_scale = alt.Scale(
        domain=["pasal", "recursive"],
        range=["#4C72B0", "#DD8452"],
    )

    chart = (
        alt.Chart(hist_agg)
        .mark_bar()
        .encode(
            x=alt.X("bin_str:N", sort=alt.EncodingSortField(field="bin_left"), title="Word Count Range"),
            y=alt.Y("count:Q", title="Jumlah Chunk"),
            color=alt.Color("chunk_method:N", scale=color_scale, title="Method"),
            tooltip=["bin_str", "chunk_method", "count"],
        )
        .properties(height=280)
    )

    # Tambahkan garis batas merah (< 30) dan kuning (> 400)
    rule_short = (
        alt.Chart(pd.DataFrame({"x": [str(WORD_COUNT_MIN_FLAG)]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.X("x:N"))
    )

    st.altair_chart(chart, use_container_width=True)

    col_a, col_b = st.columns(2)
    col_a.markdown(f"🔴 Batas minimum: **{WORD_COUNT_MIN_FLAG} kata** — chunk di bawah ini mungkin tidak informatif")
    col_b.markdown(f"🟡 Batas maksimum: **{WORD_COUNT_MAX_FLAG} kata** — chunk di atas ini mungkin terlalu panjang untuk embedding optimal")

else:
    st.info("Tidak ada data untuk ditampilkan.")

st.divider()

# ── Row 4 — Tabel chunk ───────────────────────────────────────────────────────

st.subheader("Daftar Chunk")

if not filtered.empty:
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    col_page, col_info = st.columns([1, 3])
    with col_page:
        current_page = st.number_input(
            "Halaman", min_value=1, max_value=total_pages, value=1, step=1
        )
    with col_info:
        st.caption(f"Menampilkan halaman {current_page} dari {total_pages} ({len(filtered)} chunk)")

    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = filtered.iloc[start_idx:end_idx].reset_index(drop=True)

    # Styling: highlight chunk pendek dan panjang
    def highlight_word_count(row):
        if row["word_count"] < WORD_COUNT_MIN_FLAG:
            return ["background-color: #ffe0e0"] * len(row)
        elif row["word_count"] > WORD_COUNT_MAX_FLAG:
            return ["background-color: #fff8d6"] * len(row)
        return [""] * len(row)

    display_cols = ["chunk_id", "source", "halaman", "tanggal_dokumen", "chunk_method", "word_count", "preview"]
    styled = page_df[display_cols].style.apply(highlight_word_count, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

else:
    st.info("Tidak ada chunk yang cocok dengan filter.")

st.divider()

# ── Row 5 — Detail viewer ─────────────────────────────────────────────────────

st.subheader("Detail Chunk")

if not filtered.empty:
    chunk_ids = filtered["chunk_id"].tolist()
    selected_id = st.selectbox(
        "Pilih Chunk ID untuk melihat konten lengkap",
        options=chunk_ids,
        format_func=lambda x: f"Chunk #{x}",
    )

    selected_row = filtered[filtered["chunk_id"] == selected_id].iloc[0]

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Konten:**")
        st.text_area(
            label="",
            value=selected_row["content"],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("**Metadata:**")
        meta_items = {
            "chunk_id": selected_row["chunk_id"],
            "source": selected_row["source"],
            "halaman": selected_row["halaman"],
            "tanggal_dokumen": selected_row["tanggal_dokumen"],
            "chunk_method": selected_row["chunk_method"],
            "word_count": selected_row["word_count"],
        }
        for k, v in meta_items.items():
            st.markdown(f"- **{k}:** `{v}`")

        wc = selected_row["word_count"]
        if wc < WORD_COUNT_MIN_FLAG:
            st.error(f"Chunk ini sangat pendek ({wc} kata) — pertimbangkan untuk merge atau filter.")
        elif wc > WORD_COUNT_MAX_FLAG:
            st.warning(f"Chunk ini cukup panjang ({wc} kata) — bisa kurangi chunk_size jika diperlukan.")
        else:
            st.success(f"Word count OK ({wc} kata).")
else:
    st.info("Pilih chunk dari tabel di atas.")
