"""
retriever_agent.py — Agent 2: Query ChromaDB + evaluasi + reasoning.

Alur:
    1. Query ChromaDB pakai search_chromadb()
    2. Evaluasi similarity score — kalau rendah, reformulate & retry (maks 2x)
    3. Deteksi kontradiksi antar dokumen
    4. Cek apakah dokumen outdated
    5. Kumpulkan metadata untuk Agent 3

Tidak membuat jawaban — tugasnya retrieve dan evaluasi saja.
Dipanggil dari pipeline.py sebagai node 'retriever'.
"""

from tools.chromadb_tool import (
    search_chromadb,
    check_similarity_score,
    reformulate_query,
    detect_contradiction,
    check_document_date,
    get_document_metadata,
)

MAX_RETRY = 2


def retriever_node(state: dict) -> dict:
    """
    LangGraph node untuk Agent 2 — Retriever + Reasoner.

    Input state fields:
        query         : Pertanyaan dari user (bisa sudah direformulasi)
        retry_count   : Jumlah retry yang sudah dilakukan (default 0)

    Output state fields:
        retrieved_chunks    : List chunk dari ChromaDB
        similarity_scores   : List score per chunk
        retry_count         : Updated retry count
        has_contradiction   : Ada kontradiksi antar dokumen?
        is_outdated         : Setidaknya satu chunk dari dokumen outdated?
        is_found            : Info berhasil ditemukan?
        chunk_metadata      : List metadata lengkap per chunk (untuk Agent 3)
        contradiction_info  : Detail kontradiksi dari detect_contradiction()
    """
    query       = state.get("query", "")
    retry_count = state.get("retry_count", 0)

    current_query = query

    # ── Retrieve + self-correction loop ─────────────────────────────────────
    chunks = []
    found  = False

    for attempt in range(MAX_RETRY + 1):
        chunks = search_chromadb(current_query)
        passed = check_similarity_score(chunks)

        if passed:
            found = True
            break

        # Score rendah — reformulate kalau masih ada attempt tersisa
        if attempt < MAX_RETRY:
            retry_count += 1
            current_query = reformulate_query(query, attempt + 1)

    # ── Post-processing ──────────────────────────────────────────────────────

    # Kontradiksi
    contradiction_info = detect_contradiction(chunks) if chunks else {
        "has_contradiction": False,
        "chunks_involved":   [],
        "latest_chunk":      None,
        "reason":            "Tidak ada chunk.",
    }

    # Outdated check
    is_outdated = any(check_document_date(c) for c in chunks)

    # Metadata per chunk
    chunk_metadata = [get_document_metadata(c) for c in chunks]

    # Similarity scores
    similarity_scores = [c["similarity_score"] for c in chunks]

    return {
        **state,
        "query":              current_query,          # bisa sudah direformulasi
        "retrieved_chunks":   chunks,
        "similarity_scores":  similarity_scores,
        "retry_count":        retry_count,
        "has_contradiction":  contradiction_info["has_contradiction"],
        "is_outdated":        is_outdated,
        "is_found":           found,
        "chunk_metadata":     chunk_metadata,
        "contradiction_info": contradiction_info,
    }
