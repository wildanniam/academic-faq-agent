"""
chromadb_tool.py — Tools yang digunakan Agent 2 untuk retrieve dan evaluasi chunk.

Semua fungsi di sini dipanggil oleh retriever_agent.py dalam pipeline LangGraph.
Tidak ada UI — ini murni logika bisnis RAG.

Fungsi:
    search_chromadb()        → Query ChromaDB, return top-k chunk
    check_similarity_score() → Evaluasi apakah hasil relevan
    reformulate_query()      → Reformulasi query via GPT-4o-mini
    detect_contradiction()   → Deteksi kontradiksi antar chunk
    check_document_date()    → Cek apakah dokumen outdated
    get_document_metadata()  → Ambil metadata untuk referensi output
"""

import os
from datetime import datetime, date
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

# ── Konfigurasi ───────────────────────────────────────────────────────────────

EMBED_MODEL      = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o-mini"
COLLECTION_NAME  = "academic_docs"
CHROMA_DIR       = os.getenv("CHROMA_PERSIST_DIR", "./vectordb")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
MAX_RETRY        = int(os.getenv("MAX_RETRY", "2"))
TOP_K            = int(os.getenv("TOP_K", "3"))
OUTDATED_YEARS   = 1   # dokumen dianggap outdated jika > 1 tahun dari hari ini

# ── Singleton clients ─────────────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None
_chroma_collection = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# ═════════════════════════════════════════════════════════════════════════════
# 1. search_chromadb
# ═════════════════════════════════════════════════════════════════════════════

def search_chromadb(query: str, k: int = TOP_K) -> list[dict]:
    """
    Similarity search ke ChromaDB.

    Query di-embed pakai OpenAI (harus model yang sama saat build_kb),
    lalu dibandingkan dengan semua vektor di collection pakai cosine similarity.

    Args:
        query : Pertanyaan atau query string
        k     : Jumlah chunk yang dikembalikan (default dari .env TOP_K)

    Returns:
        List of dict, diurutkan dari similarity tertinggi:
        [
            {
                "content"        : str,
                "source"         : str,
                "halaman"        : int,
                "tanggal_dokumen": str,
                "doc_level"      : int,
                "chunk_method"   : str,
                "section"        : str,
                "word_count"     : int,
                "similarity_score": float,  # 0.0 – 1.0
            },
            ...
        ]
    """
    openai = _get_openai()
    collection = _get_collection()

    # Embed query
    query_embedding = openai.embeddings.create(
        input=query,
        model=EMBED_MODEL,
    ).data[0].embedding

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        meta     = results["metadatas"][0][i]
        content  = results["documents"][0][i]

        chunks.append({
            "chunk_id":         results["ids"][0][i],
            "content":          content,
            "source":           meta.get("source", ""),
            "halaman":          meta.get("halaman", 0),
            "tanggal_dokumen":  meta.get("tanggal_dokumen", "unknown"),
            "doc_level":        meta.get("doc_level", 3),
            "chunk_method":     meta.get("chunk_method", ""),
            "section":          meta.get("section", ""),
            "word_count":       meta.get("word_count", 0),
            "similarity_score": round(1 - distance, 4),  # cosine distance → similarity
        })

    # Urutkan: doc_level rendah (lebih otoritatif) lebih diutamakan jika score sama
    chunks.sort(key=lambda x: (-x["similarity_score"], x["doc_level"]))
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# 2. check_similarity_score
# ═════════════════════════════════════════════════════════════════════════════

def check_similarity_score(
    results: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """
    Evaluasi apakah hasil retrieval cukup relevan.

    Args:
        results   : Output dari search_chromadb()
        threshold : Minimum similarity score (default 0.6 dari .env)

    Returns:
        True  → ada setidaknya satu chunk dengan score >= threshold (lanjut ke Agent 3)
        False → semua chunk di bawah threshold (perlu reformulate atau stop)
    """
    if not results:
        return False
    return any(chunk["similarity_score"] >= threshold for chunk in results)


# ═════════════════════════════════════════════════════════════════════════════
# 3. reformulate_query
# ═════════════════════════════════════════════════════════════════════════════

def reformulate_query(original_query: str, attempt: int) -> str:
    """
    Reformulasi query menggunakan GPT-4o-mini ketika similarity score rendah.

    Attempt menentukan strategi:
        1 → parafrase dengan sinonim akademik yang lebih spesifik
        2 → perluas ke istilah yang lebih umum / konteks lebih luas

    Args:
        original_query : Query asli dari user
        attempt        : Nomor percobaan (1 atau 2)

    Returns:
        Query baru yang sudah direformulasi (string)
    """
    openai = _get_openai()

    strategies = {
        1: (
            "Parafrase pertanyaan berikut menggunakan sinonim dan istilah akademik "
            "yang lebih spesifik dan formal, agar lebih mudah ditemukan dalam dokumen "
            "aturan dan pedoman akademik Telkom University. "
            "Ganti kata umum dengan istilah teknis akademik yang relevan."
        ),
        2: (
            "Perluas pertanyaan berikut dengan menggunakan istilah yang lebih umum "
            "dan konteks yang lebih luas, karena pencarian spesifik sebelumnya tidak "
            "menemukan hasil yang relevan. Fokus pada konsep utama yang ditanyakan."
        ),
    }

    strategy = strategies.get(attempt, strategies[2])

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten yang membantu memperbaiki query pencarian "
                    "untuk sistem RAG dokumen akademik. "
                    "Kembalikan HANYA query baru tanpa penjelasan apapun."
                ),
            },
            {
                "role": "user",
                "content": f"{strategy}\n\nPertanyaan asli: {original_query}",
            },
        ],
        temperature=0.3,
        max_tokens=100,
    )

    reformulated = response.choices[0].message.content.strip()
    # Hapus tanda kutip kalau GPT menambahkannya
    reformulated = reformulated.strip('"\'')
    return reformulated


# ═════════════════════════════════════════════════════════════════════════════
# 4. detect_contradiction
# ═════════════════════════════════════════════════════════════════════════════

def detect_contradiction(chunks: list[dict]) -> dict:
    """
    Deteksi apakah ada informasi yang bertentangan antar chunk dari dokumen berbeda.

    Kontradiksi dianggap ada jika:
    - Ada 2+ chunk dari source berbeda, DAN
    - Tanggal dokumen berbeda (kemungkinan versi lama vs baru)

    Dalam kasus kontradiksi, chunk dari dokumen terbaru diprioritaskan.
    Jika doc_level berbeda, chunk dengan level lebih rendah (lebih otoritatif) diprioritaskan.

    Args:
        chunks : Output dari search_chromadb()

    Returns:
        {
            "has_contradiction" : bool,
            "chunks_involved"   : list[dict],  # chunk yang berkontradiksi
            "latest_chunk"      : dict | None, # chunk yang diprioritaskan
            "reason"            : str,         # penjelasan singkat
        }
    """
    if len(chunks) <= 1:
        return {
            "has_contradiction": False,
            "chunks_involved":   [],
            "latest_chunk":      chunks[0] if chunks else None,
            "reason":            "Hanya satu chunk, tidak ada kontradiksi.",
        }

    # Kelompokkan per source
    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        src = chunk["source"]
        by_source.setdefault(src, []).append(chunk)

    unique_sources = list(by_source.keys())

    if len(unique_sources) <= 1:
        return {
            "has_contradiction": False,
            "chunks_involved":   [],
            "latest_chunk":      chunks[0],
            "reason":            "Semua chunk dari dokumen yang sama.",
        }

    # Cek apakah ada perbedaan tanggal antar dokumen
    dates_by_source = {}
    for src, src_chunks in by_source.items():
        tanggal = src_chunks[0]["tanggal_dokumen"]
        dates_by_source[src] = tanggal

    unique_dates = set(v for v in dates_by_source.values() if v != "unknown")
    has_contradiction = len(unique_dates) > 1

    if not has_contradiction:
        # Source berbeda tapi tanggal sama — mungkin dokumen pelengkap
        return {
            "has_contradiction": False,
            "chunks_involved":   [],
            "latest_chunk":      chunks[0],
            "reason":            "Chunk dari dokumen berbeda tapi tanggal sama — dianggap komplementer.",
        }

    # Ada kontradiksi: tentukan chunk mana yang diprioritaskan
    # Prioritas 1: doc_level terendah (paling otoritatif)
    # Prioritas 2: tanggal terbaru
    def priority_key(chunk: dict):
        tanggal = chunk["tanggal_dokumen"]
        try:
            parsed_date = datetime.strptime(tanggal, "%Y-%m-%d").date()
        except ValueError:
            parsed_date = date.min
        return (chunk["doc_level"], -parsed_date.toordinal())

    sorted_chunks = sorted(chunks, key=priority_key)
    latest_chunk  = sorted_chunks[0]

    return {
        "has_contradiction": True,
        "chunks_involved":   chunks,
        "latest_chunk":      latest_chunk,
        "reason": (
            f"Ditemukan informasi dari {len(unique_sources)} dokumen berbeda "
            f"dengan tanggal berbeda: {', '.join(sorted(unique_dates))}. "
            f"Diprioritaskan: '{latest_chunk['source']}' "
            f"(L{latest_chunk['doc_level']}, {latest_chunk['tanggal_dokumen']})."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. check_document_date
# ═════════════════════════════════════════════════════════════════════════════

def check_document_date(metadata: dict) -> bool:
    """
    Cek apakah dokumen sudah outdated (lebih dari OUTDATED_YEARS tahun dari hari ini).

    Args:
        metadata : Dict chunk yang memiliki field 'tanggal_dokumen' (YYYY-MM-DD)

    Returns:
        True  → dokumen outdated, perlu flag peringatan di output
        False → dokumen masih relevan / tanggal tidak diketahui
    """
    tanggal_str = metadata.get("tanggal_dokumen", "unknown")

    if tanggal_str == "unknown":
        return False  # tidak bisa menentukan, asumsikan tidak outdated

    try:
        doc_date  = datetime.strptime(tanggal_str, "%Y-%m-%d").date()
        today     = date.today()
        delta_years = (today - doc_date).days / 365.25
        return delta_years > OUTDATED_YEARS
    except ValueError:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 6. get_document_metadata
# ═════════════════════════════════════════════════════════════════════════════

def get_document_metadata(chunk: dict) -> dict:
    """
    Ambil metadata lengkap dari chunk untuk keperluan referensi output Agent 3.

    Args:
        chunk : Satu chunk dari output search_chromadb()

    Returns:
        {
            "source"         : str,   # nama file PDF
            "halaman"        : int,   # nomor halaman
            "tanggal_dokumen": str,   # YYYY-MM-DD
            "section"        : str,   # judul pasal / sub-section
            "doc_level"      : int,   # 1/2/3
            "similarity_score": float,
            "is_outdated"    : bool,
        }
    """
    return {
        "source":           chunk.get("source", ""),
        "halaman":          chunk.get("halaman", 0),
        "tanggal_dokumen":  chunk.get("tanggal_dokumen", "unknown"),
        "section":          chunk.get("section", ""),
        "doc_level":        chunk.get("doc_level", 3),
        "similarity_score": chunk.get("similarity_score", 0.0),
        "is_outdated":      check_document_date(chunk),
    }
