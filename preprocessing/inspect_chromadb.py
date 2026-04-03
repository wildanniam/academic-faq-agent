"""
inspect_chromadb.py — Visual explorer untuk isi ChromaDB.

Jalankan:
    streamlit run preprocessing/inspect_chromadb.py --server.port 8502
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from preprocessing.embedder import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# ── Konstanta ─────────────────────────────────────────────────────────────────

DOC_LEVEL_LABEL = {1: "L1 — Universitas", 2: "L2 — Dekan/PU", 3: "L3 — Panduan Teknis"}
METHOD_COLOR = {
    "pasal":      "#4C72B0",
    "subsection": "#55A868",
    "small_doc":  "#C44E52",
    "recursive":  "#DD8452",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChromaDB Inspector — Agentic RAG",
    page_icon="🗄️",
    layout="wide",
)

st.title("🗄️ ChromaDB Inspector")
st.caption(f"Collection: `{COLLECTION_NAME}` — Path: `{CHROMA_DIR}`")

# ── Load ChromaDB ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None

collection = get_collection()

if collection is None:
    st.error(f"Collection `{COLLECTION_NAME}` tidak ditemukan di `{CHROMA_DIR}`.")
    st.info("Jalankan dulu: `python preprocessing/build_kb.py`")
    st.stop()

# ── Load semua data ───────────────────────────────────────────────────────────

@st.cache_data
def load_all_data() -> pd.DataFrame:
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for i, doc_id in enumerate(result["ids"]):
        meta = result["metadatas"][i]
        rows.append({
            "id":              doc_id,
            "content":         result["documents"][i],
            "source":          meta.get("source", ""),
            "halaman":         meta.get("halaman", 0),
            "tanggal_dokumen": meta.get("tanggal_dokumen", ""),
            "doc_level":       meta.get("doc_level", 3),
            "chunk_method":    meta.get("chunk_method", ""),
            "section":         meta.get("section", ""),
            "word_count":      meta.get("word_count", 0),
        })
    df = pd.DataFrame(rows)
    df["doc_level_label"] = df["doc_level"].map(DOC_LEVEL_LABEL)
    df["preview"] = df["content"].apply(
        lambda x: " ".join(x.split()[:50]) + ("..." if len(x.split()) > 50 else "")
    )
    return df

df = load_all_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Similarity Search", "📋 Browse Data"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Status Collection")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunk", collection.count())
    c2.metric("Total Dokumen", df["source"].nunique())
    c3.metric("Embedding Model", EMBED_MODEL)
    c4.metric("Dimensi Vektor", "1536")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Chunk per Dokumen")
        doc_counts = (
            df.groupby("source")["id"].count()
            .reset_index()
            .rename(columns={"id": "jumlah"})
            .sort_values("jumlah", ascending=False)
        )
        doc_counts["source"] = doc_counts["source"].str[:45]
        st.bar_chart(doc_counts.set_index("source")["jumlah"])

    with col_right:
        st.subheader("Distribusi Method & Level")

        method_df = df["chunk_method"].value_counts().reset_index()
        method_df.columns = ["method", "jumlah"]
        st.markdown("**Chunk Method:**")
        for _, row in method_df.iterrows():
            pct = row["jumlah"] / len(df) * 100
            color = METHOD_COLOR.get(row["method"], "#888")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:12px">{row["method"]}</span>'
                f'<span>{row["jumlah"]} chunk ({pct:.0f}%)</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Doc Level:**")
        level_df = df["doc_level_label"].value_counts().reset_index()
        level_df.columns = ["level", "jumlah"]
        for _, row in level_df.iterrows():
            pct = row["jumlah"] / len(df) * 100
            st.markdown(f"- **{row['level']}**: {row['jumlah']} chunk ({pct:.0f}%)")

    st.divider()
    st.subheader("Word Count Distribution")
    wc_data = df["word_count"].values
    bins = list(range(0, int(max(wc_data)) + 50, 25))
    hist_df = pd.cut(df["word_count"], bins=bins).value_counts().sort_index().reset_index()
    hist_df.columns = ["range", "count"]
    hist_df["range"] = hist_df["range"].astype(str)
    st.bar_chart(hist_df.set_index("range")["count"])

# ════════════════════════════════════════════════════════════
# TAB 2 — Similarity Search
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Test Similarity Search")
    st.caption("Simulasi query seperti yang akan dilakukan Agent 2. Masukkan pertanyaan, lihat chunk apa yang di-retrieve.")

    query = st.text_input(
        "Masukkan pertanyaan:",
        placeholder="contoh: berapa maksimal SKS per semester?",
    )

    col_k, col_level, col_method = st.columns(3)
    with col_k:
        k = st.slider("Top-K hasil", min_value=1, max_value=10, value=3)
    with col_level:
        filter_level = st.selectbox("Filter Doc Level", ["Semua", "L1 — Universitas", "L2 — Dekan/PU", "L3 — Panduan Teknis"])
    with col_method:
        filter_method = st.selectbox("Filter Method", ["Semua", "pasal", "subsection", "small_doc", "recursive"])

    if query:
        with st.spinner("Mencari chunk yang relevan..."):
            try:
                # Build where filter
                where = None
                conditions = []
                if filter_level != "Semua":
                    level_num = {"L1 — Universitas": 1, "L2 — Dekan/PU": 2, "L3 — Panduan Teknis": 3}
                    conditions.append({"doc_level": {"$eq": level_num[filter_level]}})
                if filter_method != "Semua":
                    conditions.append({"chunk_method": {"$eq": filter_method}})
                if len(conditions) == 1:
                    where = conditions[0]
                elif len(conditions) > 1:
                    where = {"$and": conditions}

                # Embed query pakai OpenAI (harus sama dengan model yang dipakai saat build_kb)
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                query_embedding = openai_client.embeddings.create(
                    input=query,
                    model=EMBED_MODEL,
                ).data[0].embedding

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                    where=where if where else None,
                )

                ids       = results["ids"][0]
                docs      = results["documents"][0]
                metas     = results["metadatas"][0]
                distances = results["distances"][0]

                st.markdown(f"**{len(ids)} chunk ditemukan untuk:** _{query}_")
                st.divider()

                for i in range(len(ids)):
                    similarity = 1 - distances[i]
                    meta = metas[i]
                    doc  = docs[i]

                    # Warna bar similarity
                    if similarity >= 0.7:
                        bar_color = "#16a34a"
                        verdict = "✅ Relevan"
                    elif similarity >= 0.5:
                        bar_color = "#d97706"
                        verdict = "⚠️ Cukup relevan"
                    else:
                        bar_color = "#dc2626"
                        verdict = "❌ Kurang relevan"

                    level_label = DOC_LEVEL_LABEL.get(meta.get("doc_level", 3), "")
                    method = meta.get("chunk_method", "")
                    method_color = METHOD_COLOR.get(method, "#888")

                    with st.expander(
                        f"#{i+1} | Similarity: {similarity:.4f} {verdict} | {meta.get('source','')[:45]}",
                        expanded=(i == 0),
                    ):
                        # Similarity bar
                        st.markdown(
                            f'<div style="background:#f3f4f6;border-radius:8px;overflow:hidden;height:8px;margin-bottom:12px">'
                            f'<div style="background:{bar_color};width:{similarity*100:.1f}%;height:100%"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        col_a, col_b = st.columns([2, 1])

                        with col_a:
                            st.markdown("**Konten:**")
                            st.text_area("", value=doc, height=220,
                                         disabled=True, label_visibility="collapsed",
                                         key=f"content_{i}")

                        with col_b:
                            st.markdown("**Metadata:**")
                            st.markdown(f"- **similarity:** `{similarity:.4f}`")
                            st.markdown(f"- **chunk_id:** `{ids[i]}`")
                            st.markdown(f"- **source:** `{meta.get('source','')}`")
                            st.markdown(f"- **halaman:** `{meta.get('halaman','')}`")
                            st.markdown(f"- **tanggal:** `{meta.get('tanggal_dokumen','')}`")
                            st.markdown(
                                f"- **doc_level:** "
                                f'<span style="background:#7c3aed;color:white;padding:1px 7px;'
                                f'border-radius:10px;font-size:12px">{level_label}</span>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"- **method:** "
                                f'<span style="background:{method_color};color:white;padding:1px 7px;'
                                f'border-radius:10px;font-size:12px">{method}</span>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f"- **section:** `{meta.get('section','—')}`")
                            st.markdown(f"- **word_count:** `{meta.get('word_count','')}`")

                            # Verdict box
                            if similarity >= 0.6:
                                st.success(f"Score ≥ 0.6 — lolos threshold agent")
                            else:
                                st.error(f"Score < 0.6 — agent akan reformulate query")

            except Exception as e:
                st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════
# TAB 3 — Browse Data
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Browse Semua Chunk di ChromaDB")

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sumber_opts = ["Semua"] + sorted(df["source"].unique().tolist())
        pilih_sumber = st.selectbox("Dokumen", sumber_opts)
    with col_f2:
        method_opts = ["Semua"] + sorted(df["chunk_method"].unique().tolist())
        pilih_method = st.selectbox("Method", method_opts)
    with col_f3:
        search_kw = st.text_input("Cari keyword", placeholder="SKS, sidang, IPK...")

    filtered = df.copy()
    if pilih_sumber != "Semua":
        filtered = filtered[filtered["source"] == pilih_sumber]
    if pilih_method != "Semua":
        filtered = filtered[filtered["chunk_method"] == pilih_method]
    if search_kw:
        filtered = filtered[
            filtered["content"].str.contains(search_kw, case=False, na=False) |
            filtered["section"].str.contains(search_kw, case=False, na=False)
        ]

    st.caption(f"Menampilkan {len(filtered)} dari {len(df)} chunk")

    if not filtered.empty:
        display_cols = ["id", "source", "halaman", "doc_level_label",
                        "chunk_method", "section", "word_count", "preview"]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # Detail viewer
        st.divider()
        st.subheader("Detail Chunk")
        selected_id = st.selectbox(
            "Pilih ID chunk",
            options=filtered["id"].tolist(),
            format_func=lambda x: f"Chunk #{x}",
        )
        row = filtered[filtered["id"] == selected_id].iloc[0]
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.text_area("Konten lengkap:", value=row["content"], height=300,
                         disabled=True)
        with col_r:
            st.markdown("**Metadata:**")
            for field in ["id", "source", "halaman", "tanggal_dokumen",
                          "doc_level_label", "chunk_method", "section", "word_count"]:
                st.markdown(f"- **{field}:** `{row.get(field, '')}`")
    else:
        st.info("Tidak ada chunk yang cocok.")
