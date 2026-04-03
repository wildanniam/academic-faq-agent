"""
chunker.py — Smart chunking dengan 4 strategi per tipe dokumen.

Strategi dipilih otomatis berdasarkan struktur dokumen:
  1. pasal     → PERATURAN UNIVERSITAS: split per Pasal, gabungkan semua ayatnya
  2. subsection → PANDUAN TA/KP/PROPOSAL: split per sub-judul bernomor (1.1, 2.3, BAB)
  3. small_doc  → PU KELULUSAN/CUMLAUDE: split per section letter (A., B.) — dokumen kecil
  4. recursive  → fallback: RecursiveCharacterTextSplitter

Semua strategi bekerja di atas full_text dokumen (bukan per halaman),
sehingga satu Pasal atau sub-section yang span beberapa halaman tetap utuh.
"""

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Pasal X [judul opsional]
_PASAL_RE = re.compile(
    r"(Pasal\s+\d+(?:\s+[A-Za-z][^\n]{0,60})?)",
    re.IGNORECASE,
)

# Ayat dalam pasal: (1), (2), (3) dst — di awal baris
_AYAT_RE = re.compile(r"(\(\d+\))", re.MULTILINE)

# Sub-section bernomor: "1.1 JUDUL", "2.3.1 JUDUL", "BAB 1 JUDUL", "BAB I JUDUL"
# Hanya level 1 dan 2 (X.Y) yang dijadikan split point — level 3 (X.Y.Z) tetap dalam chunk
# Exclude baris yang mengandung "....." (TOC entry)
_SUBSECTION_RE = re.compile(
    r"(?<!\d)"
    r"((?:BAB\s+(?:\d+|[IVX]+)\.?\s*[^\n.]{0,60}|"  # BAB 1/BAB I/BAB I. (titik opsional)
    r"\d+\.\d+(?!\.\d)\s+[^\n.]{4,}))"               # 1.1 JUDUL (bukan 1.1.1)
    r"\s*(?=\n)",
    re.MULTILINE,
)

# Section letter: "A. Kriteria", "B. Syarat"
_SECTION_LETTER_RE = re.compile(
    r"\n([A-Z]\.\s+[^\n]{5,})\n",
    re.MULTILINE,
)

# ── Threshold ──────────────────────────────────────────────────────────────────

_MIN_WORDS = 30      # chunk lebih pendek dari ini dibuang
_MAX_WORDS = 500     # chunk lebih panjang dari ini dipecah lagi

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
)

_large_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _page_for_pos(pos: int, page_map: list[tuple]) -> int:
    """Kembalikan nomor halaman untuk posisi karakter `pos` dalam full_text."""
    for start, end, page_num in page_map:
        if start <= pos < end:
            return page_num
    return page_map[-1][2] if page_map else 1


def _detect_strategy(full_text: str) -> str:
    """Pilih strategi chunking berdasarkan struktur teks dokumen."""
    n_pasal      = len(re.findall(r"Pasal\s+\d+", full_text, re.IGNORECASE))
    n_subsection = len(re.findall(r"\d+\.\d+(?!\.\d)\s+[A-Z][^\n]{3,}", full_text))
    # BAB dengan angka arab atau romawi (I, II, III, IV, V...)
    n_bab        = len(re.findall(r"BAB\s+(?:\d+|[IVX]{1,5})\b", full_text, re.IGNORECASE))
    total_words  = len(full_text.split())

    if n_pasal >= 10:
        return "pasal"
    if n_subsection >= 5 or n_bab >= 3:
        return "subsection"
    if total_words < 3000:
        return "small_doc"
    return "recursive"


def _make_chunk(content: str, section: str, page: int, method: str, base: dict) -> dict:
    """Buat dict chunk standar."""
    return {
        "content": content,
        "section": section,
        "halaman": page,
        "chunk_method": method,
        **base,
    }


def _split_text_keeping_header(header: str, content: str, page: int,
                                method: str, base: dict) -> list[dict]:
    """
    Kalau satu section terlalu panjang (> _MAX_WORDS), pecah lagi dengan
    recursive splitter — tapi setiap sub-chunk tetap membawa header section-nya
    agar Agent 2 tahu konteks darimana chunk berasal.
    """
    full = f"{header}\n{content}".strip() if header else content.strip()

    if len(full.split()) < _MIN_WORDS:
        return []

    if len(content.split()) <= _MAX_WORDS:
        return [_make_chunk(full, header, page, method, base)]

    sub_texts = _recursive_splitter.split_text(content)
    result = []
    for i, sub in enumerate(sub_texts):
        sub = sub.strip()
        if len(sub.split()) < _MIN_WORDS:
            continue
        prefix = header if i == 0 else f"{header} (lanjutan)"
        result.append(_make_chunk(f"{prefix}\n{sub}", header, page, method, base))
    return result


# ── Strategi 1: Pasal + Ayat ───────────────────────────────────────────────────

def _split_pasal(full_text: str, page_map: list, base: dict) -> list[dict]:
    """
    Split per Pasal. Setiap chunk = satu Pasal lengkap dengan semua ayatnya.
    Kalau satu Pasal terlalu panjang, dipecah per kelompok ayat.
    """
    parts = _PASAL_RE.split(full_text)
    chunks = []

    # Teks sebelum Pasal pertama (Menimbang, Mengingat, dsb)
    intro = parts[0].strip()
    if intro and len(intro.split()) >= _MIN_WORDS:
        chunks.append(_make_chunk(intro, "intro", _page_for_pos(0, page_map), "pasal", base))

    i = 1
    while i < len(parts):
        header = parts[i].strip()                                  # "Pasal 7 Predikat Lulusan"
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        pos = full_text.find(header)
        page = _page_for_pos(pos if pos >= 0 else 0, page_map)

        full_chunk = f"{header}\n{content}".strip()
        word_count = len(full_chunk.split())

        if word_count < _MIN_WORDS:
            i += 2
            continue

        if word_count <= _MAX_WORDS:
            chunks.append(_make_chunk(full_chunk, header, page, "pasal", base))
        else:
            # Pecah per kelompok ayat agar tetap kohesif
            chunks.extend(_split_pasal_by_ayat(header, content, page, base))

        i += 2

    return chunks


def _split_pasal_by_ayat(header: str, content: str, page: int, base: dict) -> list[dict]:
    """Pecah satu pasal panjang menjadi beberapa chunk per kelompok ayat."""
    ayat_parts = _AYAT_RE.split(content)

    if len(ayat_parts) <= 1:
        # Tidak ada ayat bernomor → pakai recursive
        return _split_text_keeping_header(header, content, page, "pasal", base)

    chunks = []
    current_text = f"{header}\n{ayat_parts[0]}" if ayat_parts[0].strip() else header

    j = 1
    while j < len(ayat_parts):
        ayat_num = ayat_parts[j]                        # "(1)", "(2)", ...
        ayat_content = ayat_parts[j + 1] if j + 1 < len(ayat_parts) else ""
        candidate = current_text + "\n" + ayat_num + ayat_content

        if len(candidate.split()) > _MAX_WORDS and len(current_text.split()) >= _MIN_WORDS:
            chunks.append(_make_chunk(current_text.strip(), header, page, "pasal", base))
            current_text = f"{header} (lanjutan)\n{ayat_num}{ayat_content}"
        else:
            current_text = candidate
        j += 2

    if current_text.strip() and len(current_text.split()) >= _MIN_WORDS:
        chunks.append(_make_chunk(current_text.strip(), header, page, "pasal", base))

    return chunks


# ── Strategi 2: Sub-section bernomor ─────────────────────────────────────────

def _split_subsection(full_text: str, page_map: list, base: dict) -> list[dict]:
    """
    Split per sub-section (1.1, 2.3, BAB X).
    Setiap chunk = judul sub-section + seluruh kontennya (tabel/list ikut).
    """
    matches = list(_SUBSECTION_RE.finditer(full_text))
    chunks = []

    if not matches:
        return _split_recursive(full_text, page_map, base)

    # Teks sebelum section pertama
    intro = full_text[:matches[0].start()].strip()
    if intro and len(intro.split()) >= _MIN_WORDS:
        chunks.append(_make_chunk(intro, "intro", _page_for_pos(0, page_map), "subsection", base))

    for idx, match in enumerate(matches):
        header = match.group(0).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        page = _page_for_pos(match.start(), page_map)

        if not content and len(header.split()) < _MIN_WORDS:
            continue

        chunks.extend(_split_text_keeping_header(header, content, page, "subsection", base))

    return chunks


# ── Strategi 3: Section letter (dokumen kecil) ────────────────────────────────

def _split_small_doc(full_text: str, page_map: list, base: dict) -> list[dict]:
    """
    Split per section letter (A., B., C.) untuk dokumen kecil seperti
    PU Kelulusan dan PU Cumlaude. Kalau tidak ada section letter,
    gunakan large_splitter agar tidak terlalu banyak chunk.
    """
    matches = list(_SECTION_LETTER_RE.finditer(full_text))
    chunks = []

    if not matches:
        sub_texts = _large_splitter.split_text(full_text)
        for t in sub_texts:
            t = t.strip()
            if len(t.split()) >= _MIN_WORDS:
                chunks.append(_make_chunk(t, "", _page_for_pos(0, page_map), "small_doc", base))
        return chunks

    # Intro sebelum section pertama
    intro = full_text[:matches[0].start()].strip()
    if intro and len(intro.split()) >= _MIN_WORDS:
        chunks.append(_make_chunk(intro, "intro", _page_for_pos(0, page_map), "small_doc", base))

    for idx, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        full_chunk = f"{header}\n{content}".strip() if content else header

        page = _page_for_pos(match.start(), page_map)

        if len(full_chunk.split()) >= _MIN_WORDS:
            chunks.append(_make_chunk(full_chunk, header, page, "small_doc", base))

    return chunks


# ── Strategi 4: Recursive fallback ───────────────────────────────────────────

def _split_recursive(full_text: str, page_map: list, base: dict) -> list[dict]:
    """Fallback: RecursiveCharacterTextSplitter di atas full_text dokumen."""
    sub_texts = _recursive_splitter.split_text(full_text)
    chunks = []
    search_start = 0

    for text in sub_texts:
        text = text.strip()
        if len(text.split()) < _MIN_WORDS:
            continue

        pos = full_text.find(text[:60], search_start)
        if pos == -1:
            pos = search_start
        page = _page_for_pos(pos, page_map)
        search_start = max(0, pos)

        chunks.append(_make_chunk(text, "", page, "recursive", base))

    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk list of documents dari load_all_pdfs().

    Setiap dokumen dideteksi strateginya secara otomatis, lalu di-chunk.
    Semua chunk diberi chunk_id, word_count, dan field metadata lengkap.

    Metadata per chunk:
        chunk_id       : int   — ID unik global
        content        : str   — teks chunk
        section        : str   — judul section/pasal asal chunk
        source         : str   — nama file PDF
        halaman        : int   — nomor halaman awal chunk
        tanggal_dokumen: str   — YYYY-MM-DD atau 'unknown'
        doc_level      : int   — 1=univ, 2=dekan/PU, 3=panduan teknis
        chunk_method   : str   — 'pasal'|'subsection'|'small_doc'|'recursive'
        word_count     : int   — jumlah kata dalam chunk

    Returns:
        List of chunk dicts.
    """
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        full_text = doc["full_text"]
        page_map = doc["page_map"]

        if not full_text.strip():
            continue

        base = {
            "source": doc["source"],
            "tanggal_dokumen": doc["tanggal_dokumen"],
            "doc_level": doc["doc_level"],
        }

        strategy = _detect_strategy(full_text)

        if strategy == "pasal":
            chunks = _split_pasal(full_text, page_map, base)
        elif strategy == "subsection":
            chunks = _split_subsection(full_text, page_map, base)
        elif strategy == "small_doc":
            chunks = _split_small_doc(full_text, page_map, base)
        else:
            chunks = _split_recursive(full_text, page_map, base)

        for chunk in chunks:
            chunk["chunk_id"] = chunk_id
            chunk["word_count"] = len(chunk["content"].split())
            all_chunks.append(chunk)
            chunk_id += 1

    return all_chunks
