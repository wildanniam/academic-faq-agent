"""
build_kb.py — Pipeline preprocessing lengkap (jalankan sekali).

Alur:
    PDF (docs/) → pdf_loader → chunker → embedder → ChromaDB (vectordb/)

Jalankan:
    python preprocessing/build_kb.py           # build normal
    python preprocessing/build_kb.py --reset   # hapus ChromaDB lama, build ulang
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pdf_loader import load_all_pdfs
from preprocessing.chunker import chunk_documents, _detect_strategy
from preprocessing.embedder import embed_and_store, CHROMA_DIR

DOCS_DIR = Path(__file__).parent.parent / "docs"


def main(reset: bool = False):
    start_time = time.time()

    print("=" * 60)
    print("  BUILD KNOWLEDGE BASE — Agentic RAG")
    print("  Sistem FAQ Akademik Telkom University")
    print("=" * 60)

    # ── Step 1: Load PDF ──────────────────────────────────────────
    print("\n[1/3] Loading PDF dokumen...")
    documents = load_all_pdfs(str(DOCS_DIR))

    if not documents:
        print(f"  ❌ Tidak ada PDF ditemukan di {DOCS_DIR}")
        sys.exit(1)

    for doc in documents:
        strategy = _detect_strategy(doc["full_text"])
        words = len(doc["full_text"].split())
        print(f"  ✓ [{strategy:11}] L{doc['doc_level']} | "
              f"{doc['tanggal_dokumen']} | "
              f"{words:,} kata | {doc['source']}")

    # ── Step 2: Chunking ──────────────────────────────────────────
    print(f"\n[2/3] Chunking {len(documents)} dokumen...")
    chunks = chunk_documents(documents)

    from collections import Counter
    method_counts = Counter(c["chunk_method"] for c in chunks)
    avg_wc = sum(c["word_count"] for c in chunks) / len(chunks)

    print(f"  Total chunk : {len(chunks)}")
    print(f"  Avg wc      : {avg_wc:.0f} kata")
    for method, count in method_counts.most_common():
        print(f"  {method:12}: {count} chunk ({count/len(chunks)*100:.0f}%)")

    # ── Step 3: Embed & simpan ke ChromaDB ───────────────────────
    print(f"\n[3/3] Embedding ke ChromaDB ({CHROMA_DIR})...")
    if reset:
        print("  Mode RESET: collection lama akan dihapus")

    collection = embed_and_store(chunks, chroma_dir=CHROMA_DIR, reset=reset)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  ✅ KNOWLEDGE BASE SIAP")
    print("=" * 60)
    print(f"  Dokumen diproses : {len(documents)}")
    print(f"  Total chunk      : collection.count() = {collection.count()}")
    print(f"  ChromaDB path    : {CHROMA_DIR}")
    print(f"  Waktu total      : {elapsed:.1f} detik")
    print("\n  Siap digunakan oleh agent. Jalankan:")
    print("  → streamlit run main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build knowledge base untuk Agentic RAG")
    parser.add_argument("--reset", action="store_true",
                        help="Hapus ChromaDB lama dan build ulang dari awal")
    args = parser.parse_args()
    main(reset=args.reset)
