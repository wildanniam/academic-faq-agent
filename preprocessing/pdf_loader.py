import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional


# ── Konstanta filter halaman ──────────────────────────────────────────────────

_SKIP_FIRST_N_PAGES = 2

_TOC_DOT_PATTERN = re.compile(r"\.{5,}")
_TOC_DOT_THRESHOLD = 5

# Keyword di baris pertama halaman yang menandakan noise
_NOISE_FIRST_LINE = {"LEMBAR PENGESAHAN", "KATA PENGANTAR", "TIM PENYUSUN"}

# ── Hierarki dokumen (doc_level) ──────────────────────────────────────────────
# Level 1 = paling otoritatif (peraturan universitas)
# Level 2 = peraturan dekan / turunan universitas
# Level 3 = panduan teknis pelaksanaan

_DOC_LEVEL_MAP = {
    "PERATURAN UNIVERSITAS TELKOM TENTANG PEDOMAN AKADEMIK.pdf": 1,
    "PU_PERSYARATAN_KELULUSAN_STUDI_DAN_STANDAR_LUARAN_TUGAS_AKHIR.pdf": 2,
    "PU_KRITERIA_TAMBAHAN_UNTUK_PREDIKAT_SUMMA_CUMLAUDE_DAN_CUMLAUDE.pdf": 2,
    "20250310_SK Dekan_Panduan TA FIF 2025_v4.pdf": 3,
    "20241119_SK Dekan_Panduan PENULISAN PROPOSAL TA.pdf": 3,
    "20241112_Panduan KP 2024-signed.pdf": 3,
    "Buku Panduan Penggunaan AI untuk Pembelajaran dan Pengajaran Versi 1.0.pdf": 3,
}

# Nomor halaman arab atau romawi di awal teks
_PAGE_NUM_PATTERN = re.compile(r"^\s*(?:\d+|[ivxlcdmIVXLCDM]+)\s*\n+")

# Baris yang merupakan entry TOC inline (misal "BAB 4 ...............38")
_INLINE_TOC_LINE = re.compile(r"^.+\.{4,}\s*\d+\s*$", re.MULTILINE)


# ── Helper tanggal ─────────────────────────────────────────────────────────────

def _parse_date_from_filename(filename: str) -> Optional[str]:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def _parse_date_from_metadata(doc: fitz.Document) -> str:
    try:
        meta = doc.metadata
        raw = meta.get("creationDate", "") or meta.get("modDate", "")
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    except Exception:
        pass
    return "unknown"


# ── Filter halaman ─────────────────────────────────────────────────────────────

def _is_noise_page(text: str, page_idx: int) -> bool:
    """Return True kalau halaman ini harus di-skip."""
    if page_idx < _SKIP_FIRST_N_PAGES:
        return True
    if len(_TOC_DOT_PATTERN.findall(text)) >= _TOC_DOT_THRESHOLD:
        return True
    # Cek beberapa baris pertama (bukan hanya baris pertama)
    # untuk menangkap halaman yang dimulai dengan nomor romawi lalu judul noise
    first_lines = [l.strip() for l in text.split("\n")[:5] if l.strip()]
    for line in first_lines:
        if line in _NOISE_FIRST_LINE:
            return True
    return False


# ── Core loader ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> dict:
    """
    Load satu PDF sebagai satu dokumen dengan full text + page_map.

    page_map dipakai oleh chunker untuk memetakan posisi karakter → nomor halaman,
    sehingga setiap chunk bisa menyimpan metadata halaman yang akurat meski teks
    di-chunk dari dokumen penuh (bukan per halaman).

    Returns:
        {
            "full_text"      : str,           # seluruh teks dokumen yang sudah bersih
            "page_map"       : list[tuple],   # [(start_char, end_char, page_num), ...]
            "source"         : str,           # nama file PDF
            "tanggal_dokumen": str,           # YYYY-MM-DD atau "unknown"
            "doc_level"      : int,           # 1=universitas, 2=dekan/PU, 3=panduan teknis
        }
    """
    path = Path(pdf_path)
    filename = path.name

    tanggal = _parse_date_from_filename(filename)
    doc_level = _DOC_LEVEL_MAP.get(filename, 3)

    doc = fitz.open(pdf_path)

    if tanggal is None:
        tanggal = _parse_date_from_metadata(doc)

    full_text = ""
    page_map: list[tuple[int, int, int]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text").strip()

        if not raw_text:
            continue
        if _is_noise_page(raw_text, page_idx):
            continue

        # Bersihkan nomor halaman di awal
        clean_text = _PAGE_NUM_PATTERN.sub("", raw_text).strip()
        # Bersihkan baris TOC inline yang lolos filter halaman
        clean_text = _INLINE_TOC_LINE.sub("", clean_text).strip()
        if not clean_text:
            continue

        start_idx = len(full_text)
        full_text += clean_text + "\n\n"
        end_idx = len(full_text)
        page_map.append((start_idx, end_idx, page_idx + 1))

    doc.close()

    return {
        "full_text": full_text,
        "page_map": page_map,
        "source": filename,
        "tanggal_dokumen": tanggal,
        "doc_level": doc_level,
    }


def load_all_pdfs(documents_dir: str) -> list[dict]:
    """
    Load semua PDF dari direktori sebagai list dokumen.

    Returns:
        List of document dicts (satu dict per file PDF).
    """
    docs_path = Path(documents_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    return [load_pdf(str(f)) for f in pdf_files]
