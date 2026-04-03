"""
main.py — Entry point Streamlit UI untuk Agentic RAG FAQ Akademik.

Jalankan:
    streamlit run main.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from graph.pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FAQ Akademik — Telkom University",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp {
    max-width: 800px;
    margin: 0 auto;
}
.agent-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-router    { background: #7c3aed; color: white; }
.badge-retriever { background: #0369a1; color: white; }
.badge-responder { background: #15803d; color: white; }
.chunk-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}
.score-high   { color: #16a34a; font-weight: bold; }
.score-medium { color: #d97706; font-weight: bold; }
.score-low    { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions (didefinisikan sebelum digunakan) ────────────────────────

def _render_debug_panel(s: dict):
    """Tampilkan panel debug detail proses agent."""
    with st.expander("🔍 Detail Proses Agent", expanded=False):

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<span class="agent-badge badge-router">Agent 1 — Router</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Relevan:** {'✅' if s.get('is_relevant') else '❌'}")
            st.markdown(f"**Ambigu:** {'⚠️' if s.get('is_ambiguous') else '✅'}")

        with col2:
            st.markdown(
                '<span class="agent-badge badge-retriever">Agent 2 — Retriever</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Ditemukan:** {'✅' if s.get('is_found') else '❌'}")
            st.markdown(f"**Retry:** {s.get('retry_count', 0)}x")
            st.markdown(f"**Kontradiksi:** {'⚠️' if s.get('has_contradiction') else '✅'}")
            st.markdown(f"**Outdated:** {'⚠️' if s.get('is_outdated') else '✅'}")

        with col3:
            st.markdown(
                '<span class="agent-badge badge-responder">Agent 3 — Responder</span>',
                unsafe_allow_html=True,
            )
            refs  = s.get("references", [])
            flags = s.get("flags", [])
            st.markdown(f"**Referensi:** {len(refs)}")
            st.markdown(f"**Flags:** {len(flags)}")

        # Chunk cards
        chunks = s.get("retrieved_chunks", [])
        if chunks:
            st.markdown("---")
            st.markdown("**Chunk yang di-retrieve:**")
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("similarity_score", 0)
                if score >= 0.7:
                    score_cls = "score-high"
                    verdict   = "✅ relevan"
                elif score >= 0.5:
                    score_cls = "score-medium"
                    verdict   = "⚠️ cukup relevan"
                else:
                    score_cls = "score-low"
                    verdict   = "❌ kurang relevan"

                st.markdown(
                    f'<div class="chunk-card">'
                    f'<b>#{i}</b> — {chunk.get("source", "")[:50]} '
                    f'hal. {chunk.get("halaman", "")} | '
                    f'<span class="{score_cls}">score: {score:.4f} ({verdict})</span><br>'
                    f'<small><i>{chunk.get("section", "—")}</i></small><br>'
                    f'<small>{" ".join(chunk.get("content", "").split()[:30])}...</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Kontradiksi detail
        c_info = s.get("contradiction_info", {})
        if c_info.get("has_contradiction"):
            st.markdown("---")
            st.warning(f"⚠️ **Kontradiksi:** {c_info.get('reason', '')}")


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎓 FAQ Akademik Telkom University")
st.caption("Sistem multi-agent berbasis Agentic RAG — Fakultas Informatika")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {role, content, state}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Pengaturan")
    show_debug = st.toggle("Tampilkan detail proses agent", value=False)
    st.session_state["show_debug"] = show_debug

    st.divider()
    st.subheader("ℹ️ Info Sistem")
    st.markdown(
        "- **Framework:** LangGraph\n"
        "- **LLM:** GPT-4o-mini\n"
        "- **Embedding:** text-embedding-3-small\n"
        "- **Vector DB:** ChromaDB (local)\n"
        "- **Agent:** 3-agent pipeline\n"
    )

    st.divider()
    st.subheader("💬 Contoh Pertanyaan")
    example_questions = [
        "Berapa maksimal SKS per semester?",
        "Apa syarat untuk mendapat predikat cumlaude?",
        "Bagaimana prosedur pengajuan cuti akademik?",
        "Berapa IPK minimum untuk lulus?",
        "Apa syarat mengikuti sidang tugas akhir?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["prefill_query"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "state" in msg:
            if st.session_state.get("show_debug", False):
                _render_debug_panel(msg["state"])

# ── Handle prefill dari sidebar ───────────────────────────────────────────────

prefill = st.session_state.pop("prefill_query", None)

# ── Chat input ────────────────────────────────────────────────────────────────

user_query = st.chat_input("Ketik pertanyaan akademikmu di sini...")

if prefill and not user_query:
    user_query = prefill

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_query,
    })

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari informasi..."):
            start_t = time.time()
            try:
                final_state = run_pipeline(user_query)
                elapsed     = time.time() - start_t
                answer      = final_state.get("final_answer", "Maaf, terjadi kesalahan.")

                st.markdown(answer)
                st.caption(f"⏱️ Diproses dalam {elapsed:.1f} detik")

                if show_debug:
                    _render_debug_panel(final_state)

            except Exception as e:
                elapsed     = time.time() - start_t
                answer      = (
                    f"Maaf, terjadi kesalahan teknis: `{e}`\n\n"
                    "Silakan coba lagi atau hubungi administrator sistem."
                )
                final_state = {}
                st.error(answer)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "state":   final_state,
    })
