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

from datetime import datetime
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
    logs        = list(state.get("logs", []))

    current_query = query

    logs.append({
        "ts":    datetime.now().isoformat(),
        "agent": "Retriever (Agent 2)",
        "event": "START",
        "data":  {"query": current_query},
    })

    # ── Retrieve + self-correction loop ─────────────────────────────────────
    chunks = []
    found  = False

    for attempt in range(MAX_RETRY + 1):
        t_search = datetime.now()
        chunks   = search_chromadb(current_query)
        passed   = check_similarity_score(chunks)

        logs.append({
            "ts":    datetime.now().isoformat(),
            "agent": "Retriever (Agent 2)",
            "event": f"SEARCH (attempt {attempt + 1})",
            "data": {
                "query":      current_query,
                "n_chunks":   len(chunks),
                "scores":     [c["similarity_score"] for c in chunks],
                "chunk_ids":  [c.get("chunk_id", "") for c in chunks],
                "passed":     passed,
                "elapsed_ms": round((datetime.now() - t_search).total_seconds() * 1000),
            },
        })

        if passed:
            found = True
            break

        # Score rendah — reformulate kalau masih ada attempt tersisa
        if attempt < MAX_RETRY:
            retry_count  += 1
            old_query     = current_query
            current_query = reformulate_query(query, attempt + 1)
            logs.append({
                "ts":    datetime.now().isoformat(),
                "agent": "Retriever (Agent 2)",
                "event": "REFORMULATE",
                "data": {
                    "attempt":    attempt + 1,
                    "old_query":  old_query,
                    "new_query":  current_query,
                },
            })

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

    logs.append({
        "ts":    datetime.now().isoformat(),
        "agent": "Retriever (Agent 2)",
        "event": "RESULT",
        "data": {
            "is_found":        found,
            "retry_count":     retry_count,
            "has_contradiction": contradiction_info["has_contradiction"],
            "contradiction_reason": contradiction_info.get("reason", ""),
            "is_outdated":     is_outdated,
        },
    })

    return {
        **state,
        "query":              current_query,
        "retrieved_chunks":   chunks,
        "similarity_scores":  similarity_scores,
        "retry_count":        retry_count,
        "has_contradiction":  contradiction_info["has_contradiction"],
        "is_outdated":        is_outdated,
        "is_found":           found,
        "chunk_metadata":     chunk_metadata,
        "contradiction_info": contradiction_info,
        "logs":               logs,
    }
