"""
pipeline.py — LangGraph graph definition & AgentState schema.

Graph:
    START → router → [retriever | END] → responder → END

Routing:
    - router → retriever  : is_relevant=True & is_ambiguous=False
    - router → END        : is_relevant=False atau is_ambiguous=True
                            (rejection_message sudah diisi Agent 1 via responder)
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from agents.router_agent import router_node
from agents.retriever_agent import retriever_node
from agents.responder_agent import responder_node


# ── State Schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Input awal
    query: str                   # Pertanyaan dari user

    # Hasil Agent 1
    is_relevant: bool            # Pertanyaan relevan dengan dokumen?
    is_ambiguous: bool           # Pertanyaan ambigu?
    rejection_message: str       # Pesan kalau ditolak atau minta klarifikasi

    # Hasil Agent 2
    retrieved_chunks: List       # Chunk hasil ChromaDB
    similarity_scores: List      # Score tiap chunk
    retry_count: int             # Sudah retry berapa kali (maks 2)
    has_contradiction: bool      # Ada kontradiksi antar dokumen?
    is_outdated: bool            # Dokumen outdated (> 1 tahun)?
    is_found: bool               # Info ditemukan di knowledge base?
    chunk_metadata: List         # Metadata lengkap per chunk
    contradiction_info: dict     # Detail kontradiksi dari detect_contradiction()

    # Hasil Agent 3
    final_answer: str            # Jawaban final ke user
    references: List             # Referensi pasal + halaman
    flags: List                  # Flag peringatan (outdated, kontradiksi)


# ── Routing condition ─────────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> str:
    """
    Tentukan langkah setelah Agent 1 (Router).

    → 'retriever' : Pertanyaan relevan dan tidak ambigu → lanjut retrieve
    → 'responder' : Tidak relevan atau ambigu → langsung ke responder
                    (responder akan pakai rejection_message dari router)
    """
    if state.get("is_relevant", True) and not state.get("is_ambiguous", False):
        return "retriever"
    return "responder"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Bangun dan compile LangGraph pipeline.

    Returns:
        Compiled LangGraph app yang siap dipanggil dengan .invoke()
    """
    graph = StateGraph(AgentState)

    # Daftarkan nodes
    graph.add_node("router",    router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)

    # Entry point
    graph.set_entry_point("router")

    # Edge: router → conditional
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever": "retriever",
            "responder": "responder",
        },
    )

    # Edge: retriever → responder (selalu)
    graph.add_edge("retriever", "responder")

    # Edge: responder → END
    graph.add_edge("responder", END)

    return graph.compile()


# ── Singleton pipeline ────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline():
    """Ambil compiled pipeline (singleton — di-compile sekali saja)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


# ── Public helper ─────────────────────────────────────────────────────────────

def run_pipeline(query: str) -> AgentState:
    """
    Jalankan pipeline untuk satu pertanyaan.

    Args:
        query : Pertanyaan dari user

    Returns:
        AgentState final — akses state["final_answer"] untuk jawaban
    """
    pipeline = get_pipeline()
    initial_state: AgentState = {
        "query":       query,
        "retry_count": 0,
    }
    return pipeline.invoke(initial_state)
