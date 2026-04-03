import os
import re
import fitz  # PyMuPDF
from pathlib import Path


def _parse_date_from_filename(filename: str) -> str:
    """
    Parse tanggal dari nama file dengan format prefix YYYYMMDD_.
    Contoh: '20241112_Panduan KP 2024-signed.pdf' -> '2024-11-12'
    Return 'unknown' kalau tidak ada prefix tanggal.
    """
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def _parse_date_from_metadata(doc: fitz.Document) -> str:
    """
    Ambil tanggal dari metadata PDF (field CreationDate).
    Format PDF: 'D:20241112...' -> '2024-11-12'
    Return 'unknown' kalau tidak ada atau tidak bisa di-parse.
    """
    try:
        meta = doc.metadata
        creation = meta.get("creationDate", "") or meta.get("modDate", "")
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})", creation)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    except Exception:
        pass
    return "unknown"


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Ekstrak teks dari setiap halaman PDF.

    Args:
        pdf_path: Path ke file PDF

    Returns:
        List of dict, satu dict per halaman:
        {
            "text": str,
            "source": str,        # nama file PDF
            "halaman": int,       # nomor halaman (1-indexed)
            "tanggal_dokumen": str # YYYY-MM-DD atau 'unknown'
        }
    """
    path = Path(pdf_path)
    filename = path.name

    tanggal = _parse_date_from_filename(filename)

    pages = []
    doc = fitz.open(pdf_path)

    if tanggal is None:
        tanggal = _parse_date_from_metadata(doc)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        pages.append({
            "text": text,
            "source": filename,
            "halaman": page_num + 1,
            "tanggal_dokumen": tanggal,
        })

    doc.close()
    return pages


def load_all_pdfs(documents_dir: str) -> list[dict]:
    """
    Load semua file PDF dari satu direktori.

    Args:
        documents_dir: Path ke folder berisi PDF

    Returns:
        List of dict (gabungan semua halaman dari semua PDF)
    """
    docs_path = Path(documents_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    all_pages = []
    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)

    return all_pages
