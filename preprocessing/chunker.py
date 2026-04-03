import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Pola untuk deteksi header pasal/bab dalam teks dokumen akademik
PASAL_PATTERN = re.compile(r"((?:Pasal\s+\d+|BAB\s+[IVX]+)[^\n]*)", re.IGNORECASE)

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)


def _chunk_by_pasal(text: str) -> list[tuple[str, str]]:
    """
    Split teks berdasarkan pola Pasal/BAB.
    Return list of (header, content) tuples.
    Kalau tidak ada pola yang ditemukan, return list kosong.
    """
    parts = PASAL_PATTERN.split(text)

    if len(parts) <= 1:
        return []

    chunks = []
    i = 0

    # Kalau ada teks sebelum pasal pertama, ikutkan sebagai intro
    if parts[0].strip():
        chunks.append(("", parts[0].strip()))

    # parts[1], parts[2], parts[3], parts[4], ... -> header, content, header, content, ...
    while i + 1 < len(parts):
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if header or content:
            chunks.append((header, content))
        i += 2

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunking semua halaman PDF.

    Strategi:
    1. Coba split per Pasal/BAB via regex (chunk_method='pasal')
    2. Fallback ke RecursiveCharacterTextSplitter (chunk_method='recursive')

    Args:
        pages: Output dari pdf_loader.load_all_pdfs()

    Returns:
        List of dict chunk:
        {
            "content": str,
            "source": str,
            "halaman": int,
            "tanggal_dokumen": str,
            "chunk_id": int,
            "chunk_method": str,   # 'pasal' atau 'recursive'
            "word_count": int
        }
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        base_meta = {
            "source": page["source"],
            "halaman": page["halaman"],
            "tanggal_dokumen": page["tanggal_dokumen"],
        }

        pasal_chunks = _chunk_by_pasal(text)

        if pasal_chunks:
            for header, content in pasal_chunks:
                full_text = f"{header}\n{content}".strip() if header else content
                if not full_text:
                    continue

                all_chunks.append({
                    "content": full_text,
                    **base_meta,
                    "chunk_id": chunk_id,
                    "chunk_method": "pasal",
                    "word_count": len(full_text.split()),
                })
                chunk_id += 1
        else:
            fallback_texts = _recursive_splitter.split_text(text)
            for ft in fallback_texts:
                ft = ft.strip()
                if not ft:
                    continue

                all_chunks.append({
                    "content": ft,
                    **base_meta,
                    "chunk_id": chunk_id,
                    "chunk_method": "recursive",
                    "word_count": len(ft.split()),
                })
                chunk_id += 1

    return all_chunks
