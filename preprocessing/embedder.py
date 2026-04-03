"""
embedder.py — Embed chunks dan simpan ke ChromaDB.

Menggunakan OpenAI text-embedding-3-small (1536 dimensi).
ChromaDB disimpan secara lokal dan persisten di folder vectordb/.
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

load_dotenv()

# ── Konfigurasi ───────────────────────────────────────────────────────────────

EMBED_MODEL      = "text-embedding-3-small"
COLLECTION_NAME  = "academic_docs"
CHROMA_DIR       = os.getenv("CHROMA_PERSIST_DIR", "./vectordb")
BATCH_SIZE       = 50    # jumlah chunk per request ke OpenAI
RETRY_DELAY      = 2     # detik tunggu kalau rate limit

# Metadata yang disimpan di ChromaDB (harus tipe str/int/float/bool)
_META_FIELDS = ["source", "halaman", "tanggal_dokumen", "doc_level",
                "chunk_method", "section", "word_count"]

# ── Client ─────────────────────────────────────────────────────────────────────

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan di environment / .env")
    return OpenAI(api_key=api_key)


def _get_chroma_collection(chroma_dir: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity untuk teks
    )
    return collection


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed satu batch teks, dengan retry kalau rate limit."""
    for attempt in range(3):
        try:
            response = openai_client.embeddings.create(
                input=texts,
                model=EMBED_MODEL,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < 2:
                print(f"  ⚠️  Retry {attempt + 1}/3 karena error: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def _build_metadata(chunk: dict) -> dict:
    """
    Ambil field metadata dari chunk dan pastikan semua nilainya
    kompatibel dengan ChromaDB (str/int/float/bool, tidak boleh None).
    """
    meta = {}
    for field in _META_FIELDS:
        val = chunk.get(field, "")
        if val is None:
            val = ""
        meta[field] = val
    return meta


# ── Main functions ─────────────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict], chroma_dir: str = CHROMA_DIR,
                    reset: bool = False) -> chromadb.Collection:
    """
    Embed semua chunk dan simpan ke ChromaDB.

    Args:
        chunks    : Output dari chunk_documents()
        chroma_dir: Path ke folder ChromaDB persistent storage
        reset     : Kalau True, hapus collection lama sebelum menyimpan

    Returns:
        ChromaDB Collection yang sudah terisi
    """
    openai_client = _get_openai_client()

    # Init ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    if reset and COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        print(f"  Menghapus collection lama '{COLLECTION_NAME}'...")
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Cek chunk yang sudah ada agar tidak double-embed
    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if str(c["chunk_id"]) not in existing_ids]

    if not new_chunks:
        print(f"  Semua {len(chunks)} chunk sudah ada di ChromaDB. Skip embedding.")
        return collection

    print(f"  Embedding {len(new_chunks)} chunk baru"
          f" ({len(chunks) - len(new_chunks)} sudah ada, di-skip)...")

    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(new_chunks))
        batch = new_chunks[start:end]

        texts     = [c["content"] for c in batch]
        ids       = [str(c["chunk_id"]) for c in batch]
        metadatas = [_build_metadata(c) for c in batch]

        embeddings = _embed_batch(openai_client, texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        pct = (batch_idx + 1) / total_batches * 100
        print(f"  [{pct:5.1f}%] Batch {batch_idx + 1}/{total_batches}"
              f" — chunk {start + 1}–{end} tersimpan")

    print(f"  ✅ Total tersimpan di ChromaDB: {collection.count()} chunk")
    return collection


def get_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """Ambil ChromaDB collection yang sudah ada (tanpa embedding ulang)."""
    return _get_chroma_collection(chroma_dir)
