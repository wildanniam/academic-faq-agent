# Development Journal — Agentic RAG FAQ Akademik
## Sistem Multi-Agent untuk FAQ Aturan Akademik Telkom University (Fakultas Informatika)

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Fase Pengembangan](#3-fase-pengembangan)
   - [Fase 1 — PDF Loader](#fase-1--pdf-loader)
   - [Fase 2 — Chunking Strategi](#fase-2--chunking-strategi)
   - [Fase 3 — Embedding & Vector Database](#fase-3--embedding--vector-database)
   - [Fase 4 — Tools Layer](#fase-4--tools-layer)
   - [Fase 5 — Agents (Multi-Agent System)](#fase-5--agents-multi-agent-system)
   - [Fase 6 — LangGraph Pipeline](#fase-6--langgraph-pipeline)
   - [Fase 7 — Streamlit UI](#fase-7--streamlit-ui)
4. [Tabel Struggle & Solusi](#4-tabel-struggle--solusi)
5. [Evolusi Kualitas Chunk](#5-evolusi-kualitas-chunk)
6. [Keputusan Desain Penting](#6-keputusan-desain-penting)
7. [Hasil Akhir](#7-hasil-akhir)

---

## 1. Gambaran Umum Proyek

Proyek ini adalah sistem **Agentic RAG (Retrieval-Augmented Generation)** yang menjawab pertanyaan mahasiswa dan dosen seputar aturan dan pedoman akademik Telkom University (Fakultas Informatika).

Sistem menerima pertanyaan dalam bahasa natural, mencari jawaban dari 7 dokumen PDF resmi kampus, dan menghasilkan jawaban terstruktur lengkap dengan referensi pasal dan halaman.

### Dokumen Sumber (Knowledge Base)

| Dokumen | Level | Keterangan |
|---------|-------|------------|
| `PERATURAN UNIVERSITAS TELKOM TENTANG PEDOMAN AKADEMIK.pdf` | L1 | Peraturan tertinggi (universitas) |
| `PU_PERSYARATAN_KELULUSAN_STUDI_DAN_STANDAR_LUARAN_TUGAS_AKHIR.pdf` | L2 | Peraturan dekan |
| `PU_KRITERIA_TAMBAHAN_UNTUK_PREDIKAT_SUMMA_CUMLAUDE_DAN_CUMLAUDE.pdf` | L2 | Peraturan dekan |
| `20250310_SK Dekan_Panduan TA FIF 2025_v4.pdf` | L3 | Panduan teknis |
| `20241119_SK Dekan_Panduan PENULISAN PROPOSAL TA.pdf` | L3 | Panduan teknis |
| `20241112_Panduan KP 2024-signed.pdf` | L3 | Panduan teknis |
| `Buku Panduan Penggunaan AI untuk Pembelajaran dan Pengajaran Versi 1.0.pdf` | L3 | Panduan teknis |

### Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework Agentic | LangGraph |
| LLM | GPT-4o-mini (OpenAI) |
| Embedding | text-embedding-3-small (OpenAI, 1536 dimensi) |
| Vector Database | ChromaDB (local, persistent) |
| PDF Parser | PyMuPDF (fitz) |
| Web UI | Streamlit |
| Language | Python 3.9 |

---

## 2. Arsitektur Sistem

Sistem terdiri dari dua pipeline utama:

### Pipeline 1 — Preprocessing (dijalankan sekali)

```
PDF (docs/)
    ↓
pdf_loader.py       → Ekstrak teks + bangun page_map
    ↓
chunker.py          → Smart chunking (4 strategi adaptif)
    ↓
embedder.py         → Embed via OpenAI text-embedding-3-small
    ↓
ChromaDB            → Simpan vektor + metadata secara persisten
```

### Pipeline 2 — Agentic RAG (setiap ada pertanyaan)

```
User Input
    ↓
Agent 1 — Router
  → Relevan & tidak ambigu? → lanjut ke Agent 2
  → Tidak relevan?          → tolak sopan, stop
  → Ambigu?                 → minta klarifikasi, stop
    ↓
Agent 2 — Retriever + Reasoner
  → Query ChromaDB (top-3 chunk, cosine similarity)
  → Score < 0.6? → reformulate query, retry maks 2x
  → Tidak ditemukan setelah 2x? → flag, stop
  → Deteksi kontradiksi antar dokumen
  → Cek apakah dokumen outdated (> 1 tahun)
    ↓
Agent 3 — Responder
  → Generate jawaban via GPT-4o-mini
  → Format Level 3: jawaban + dasar hukum + disclaimer
    ↓
Output ke User
```

### State Schema (LangGraph)

```python
class AgentState(TypedDict):
    query: str                  # Pertanyaan user
    is_relevant: bool           # Hasil Agent 1
    is_ambiguous: bool
    rejection_message: str
    retrieved_chunks: list      # Hasil Agent 2
    similarity_scores: list
    retry_count: int
    has_contradiction: bool
    is_outdated: bool
    is_found: bool
    chunk_metadata: list
    contradiction_info: dict
    final_answer: str           # Hasil Agent 3
    references: list
    flags: list
    logs: list                  # Pipeline execution trace
```

---

## 3. Fase Pengembangan

---

### Fase 1 — PDF Loader

**File:** `preprocessing/pdf_loader.py`

**Tujuan:** Ekstrak teks dari 7 PDF secara bersih, dengan metadata yang akurat (halaman, tanggal dokumen, doc_level).

#### Pendekatan Awal (Gagal): Per-Page Extraction

Pendekatan pertama mengekstrak teks per halaman, menghasilkan list of pages. Masalahnya:
- Sebuah chunk bisa terpotong di batas halaman, memutus konteks
- Metadata halaman menjadi tidak akurat karena chunk bisa mencakup beberapa halaman

#### Solusi: Full-Document + Page Map

Beralih ke pendekatan **full-document extraction** — seluruh dokumen diekstrak sebagai satu string panjang, lalu dibangun `page_map` untuk memetakan posisi karakter ke nomor halaman.

```python
page_map = [(start_char, end_char, page_num), ...]
```

Saat sebuah chunk dibuat, halaman ditentukan dengan mencari `start_char` chunk di `page_map`. Ini memastikan nomor halaman selalu akurat meskipun chunk mencakup lebih dari satu halaman.

#### Struggle 1: Halaman Sampul & Daftar Isi Masuk sebagai Konten

**Masalah:** Halaman daftar isi, lembar pengesahan, dan kata pengantar ikut ter-extract dan menghasilkan chunk yang tidak berguna.

**Solusi:**
- Skip 2 halaman pertama secara otomatis
- Deteksi halaman TOC dengan menghitung pola titik-titik (`....`) — jika ≥ 5 baris mengandung pola ini, halaman tersebut dianggap daftar isi dan di-skip
- Filter baris pertama yang mengandung noise: `LEMBAR PENGESAHAN`, `KATA PENGANTAR`, `TIM PENYUSUN`

```python
_TOC_DOT_PATTERN = re.compile(r'\.{4,}\s*\d+')  # deteksi TOC
```

#### Struggle 2: Nomor Halaman Romawi Ikut Masuk ke Teks

**Masalah:** Beberapa PDF memiliki penomoran halaman romawi (i, ii, iii) di awal halaman. Teks "ii" atau "iii" ikut masuk ke konten chunk.

**Solusi:** Tambahkan regex untuk strip nomor halaman Arab dan Romawi di awal baris:

```python
_PAGE_NUM_PATTERN = re.compile(
    r'^\s*(?:\d+|[ivxlcdmIVXLCDM]{1,6})\s*$',
    re.MULTILINE
)
```

Kemudian dicek 5 baris pertama non-empty per halaman untuk noise tambahan seperti `TIM PENYUSUN`.

#### Struggle 3: Baris TOC Inline dalam Halaman Konten

**Masalah:** Beberapa halaman konten menyisipkan entri daftar isi di tengah-tengah teks (contoh: `Pasal 12 ........ 8`). Baris-baris ini lolos dari filter karena halamannya tidak sepenuhnya TOC.

**Solusi:** Tambahkan filter per-baris menggunakan regex:

```python
_INLINE_TOC_LINE = re.compile(r'^.+\.{4,}\s*\d+\s*$', re.MULTILINE)
```

Setiap baris yang cocok pola ini di-strip sebelum teks diproses lebih lanjut.

#### Struggle 4: Parsing Tanggal Dokumen

**Masalah:** Beberapa nama file memiliki prefix tanggal (`20250310_SK Dekan_...`), tapi dokumen lain tidak. Tanggal juga tersimpan di metadata PDF yang tidak selalu ada.

**Solusi:** Hierarki parsing:
1. Coba parse prefix `YYYYMMDD_` dari nama file
2. Fallback ke metadata PDF (`/CreationDate`)
3. Fallback ke `"unknown"` jika tidak ada

---

### Fase 2 — Chunking Strategi

**File:** `preprocessing/chunker.py`

**Tujuan:** Memotong dokumen menjadi chunk yang optimal untuk retrieval Agent 2 — tidak terlalu pendek (kehilangan konteks) dan tidak terlalu panjang (terlalu general).

#### Empat Strategi Adaptif

Sistem mendeteksi karakteristik setiap dokumen dan memilih strategi yang tepat secara otomatis:

| Strategi | Kondisi | Dokumen yang Cocok |
|----------|---------|-------------------|
| `pasal` | ≥ 10 pola "Pasal X" ditemukan | Peraturan Akademik (L1) |
| `subsection` | ≥ 5 pola BAB/subsection | Panduan TA, Panduan KP |
| `small_doc` | Total kata < 3.000 | Dokumen PU singkat (L2) |
| `recursive` | Fallback (tidak masuk kategori lain) | Panduan AI (L3) |

#### Bagaimana Strategi `pasal` Bekerja

Dokumen dipotong di setiap header pasal menggunakan regex:
```python
_PASAL_RE = re.compile(r'(?=Pasal\s+\d+)', re.IGNORECASE)
```

Setiap potongan disimpan dengan metadata `section` berisi judul pasalnya, sehingga Agent 2 bisa langsung tahu konteks chunk berasal dari pasal mana.

#### Bagaimana Strategi `subsection` Bekerja

Mendeteksi BAB dan sub-section (format `1.1`, `2.3`, dll):
```python
_SUBSECTION_RE = re.compile(
    r'(BAB\s+(?:\d+|[IVX]+)\.?\s*|'
    r'\d+\.\d+(?!\.\d)\s+[^\n.]{4,})',
    re.IGNORECASE
)
```

#### Struggle 1: Duplicate Chunks

**Masalah:** Setiap dokumen menghasilkan chunk duplikat — teks intro muncul dua kali.

**Penyebab:** Loop chunking per-pasal dimulai dari `i=0` (teks sebelum pasal pertama) lalu `i=1` (pasal pertama), tapi teks sebelum pasal pertama sudah ter-include dalam iterasi pertama.

**Solusi:** Ubah loop mulai dari `i=1`:

```python
# SEBELUM (bug):
for i in range(0, len(parts), 2):
    header = parts[i]
    body   = parts[i+1] if i+1 < len(parts) else ""

# SESUDAH (fix):
for i in range(1, len(parts), 2):
    header = parts[i]
    body   = parts[i+1] if i+1 < len(parts) else ""
```

#### Struggle 2: Dokumen PANDUAN_AI Tidak Terdeteksi sebagai Subsection

**Masalah:** Dokumen panduan AI menggunakan format BAB dengan titik: `"BAB I."` (ada titik setelah angka romawi). Regex awal tidak mengenali format ini.

**Solusi:** Tambahkan `\.?` (titik opsional) di regex:

```python
# SEBELUM:
r'BAB\s+(?:\d+|[IVX]+)\s*'
# SESUDAH:
r'BAB\s+(?:\d+|[IVX]+)\.?\s*'
```

#### Struggle 3: Filter Noise Tidak Efektif untuk Halaman dengan Nomor Romawi

**Masalah:** Halaman yang dimulai dengan "ii" (nomor halaman romawi) lalu "TIM PENYUSUN" lolos filter noise, karena filter hanya cek baris pertama literal bukan setelah strip nomor halaman.

**Solusi:** Update urutan pemrosesan — strip nomor halaman dulu, baru cek 5 baris pertama untuk noise.

#### Struggle 4: Chunk Terlalu Pendek

**Masalah:** Banyak chunk < 30 kata yang tidak berguna untuk retrieval (misalnya, pasal yang hanya berisi satu kalimat pendek).

**Solusi:** Tambahkan `_MIN_WORDS = 30` — chunk di bawah ambang ini di-merge ke chunk sebelumnya alih-alih disimpan sendiri.

#### Hasil Perbaikan Chunking

| Metrik | Sebelum | Sesudah |
|--------|---------|---------|
| Total chunk | 1.435 | 657 |
| Rata-rata word count | 58 kata | 117 kata |
| Chunk pendek (< 30 kata) | banyak | 0 |
| Chunk duplikat | 17 | 2 |

---

### Fase 3 — Embedding & Vector Database

**File:** `preprocessing/embedder.py`, `preprocessing/build_kb.py`

**Tujuan:** Mengubah 657 chunk teks menjadi vektor 1536 dimensi dan menyimpannya di ChromaDB untuk similarity search.

#### Konfigurasi ChromaDB

```python
collection = client.get_or_create_collection(
    name="academic_docs",
    metadata={"hnsw:space": "cosine"},  # cosine similarity untuk teks
)
```

Dipilih **cosine similarity** (bukan euclidean) karena mengukur sudut antar vektor — lebih stabil untuk teks yang panjangnya bervariasi.

#### Metadata yang Disimpan per Chunk

```python
metadata = {
    "source":          "nama_file.pdf",
    "halaman":         12,
    "tanggal_dokumen": "2024-07-01",
    "doc_level":       1,          # 1=Universitas, 2=Dekan, 3=Teknis
    "chunk_method":    "pasal",
    "section":         "Pasal 12 — Beban Studi",
    "word_count":      145,
}
```

`doc_level` sangat penting untuk prioritisasi — jika ada kontradiksi, dokumen L1 lebih otoritatif dari L3.

#### Struggle: Dimension Mismatch di ChromaDB Inspector

**Masalah:** Saat membuat visual inspector `inspect_chromadb.py`, query similarity search menghasilkan error dimension mismatch:
```
Expected 1536 dimensions, got 384
```

**Penyebab:** ChromaDB inspector menggunakan parameter `query_texts=["..."]` yang memicu ChromaDB menggunakan **built-in embedder default (384 dimensi)**, bukan OpenAI embedder yang dipakai saat build knowledge base (1536 dimensi).

**Solusi:** Embed query secara manual via OpenAI terlebih dahulu, lalu gunakan `query_embeddings=` (bukan `query_texts=`):

```python
# SEBELUM (salah):
results = collection.query(query_texts=[query])

# SESUDAH (benar):
query_embedding = openai_client.embeddings.create(
    input=query, model="text-embedding-3-small"
).data[0].embedding

results = collection.query(query_embeddings=[query_embedding])
```

**Lesson learned:** Embedding model saat build dan saat query **harus sama persis**. ChromaDB tidak memiliki mekanisme enforce ini secara otomatis.

---

### Fase 4 — Tools Layer

**File:** `tools/chromadb_tool.py`

**Tujuan:** Menyediakan 6 fungsi tools yang akan digunakan Agent 2 untuk retrieval dan evaluasi.

| Fungsi | Kegunaan |
|--------|----------|
| `search_chromadb()` | Similarity search, return top-k chunk |
| `check_similarity_score()` | Evaluasi apakah hasil cukup relevan (threshold 0.6) |
| `reformulate_query()` | Reformulasi query via GPT-4o-mini jika score rendah |
| `detect_contradiction()` | Deteksi kontradiksi antar chunk dari dokumen berbeda |
| `check_document_date()` | Cek apakah dokumen outdated (> 1 tahun) |
| `get_document_metadata()` | Ambil metadata lengkap untuk referensi output |

#### Struggle: Python 3.9 Incompatibility

**Masalah:** Saat file dijalankan, muncul error:
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

**Penyebab:** Menggunakan syntax type union modern (`X | None`) yang hanya tersedia di Python 3.10+, sedangkan environment menggunakan Python 3.9.

```python
# SEBELUM (Python 3.10+ only):
_openai_client: OpenAI | None = None

# SESUDAH (Python 3.9 compatible):
from typing import Optional
_openai_client: Optional[OpenAI] = None
```

Masalah yang sama juga ditemukan di `pdf_loader.py` untuk return type `str | None`.

#### Logika `detect_contradiction()`

Kontradiksi didefinisikan sebagai: chunk dari **2+ dokumen berbeda** dengan **tanggal dokumen berbeda**. Ketika kontradiksi terdeteksi, prioritas ditentukan oleh:
1. `doc_level` terendah (paling otoritatif)
2. Tanggal dokumen terbaru (jika doc_level sama)

---

### Fase 5 — Agents (Multi-Agent System)

**File:** `agents/router_agent.py`, `agents/retriever_agent.py`, `agents/responder_agent.py`

#### Agent 1 — Router

Menggunakan GPT-4o-mini dengan `response_format={"type": "json_object"}` untuk memastikan output selalu valid JSON:

```python
{
  "is_relevant": true/false,
  "is_ambiguous": true/false,
  "reason": "...",
  "message": "..."
}
```

#### Struggle: Router Terlalu Agresif Memfilter

**Masalah:** Pertanyaan seperti *"adakah aturan berpakaian di kampus?"* ditolak oleh Agent 1 dengan alasan "tidak ada di knowledge base" — padahal seharusnya diteruskan ke Agent 2 agar Agent 2 yang menentukan apakah informasinya ada atau tidak.

**Akar Masalah:** System prompt Agent 1 berisi daftar topik yang terlalu spesifik. Jika topik tidak ada dalam daftar, langsung ditolak. Ini menciptakan ambiguitas antara dua kondisi berbeda:

| Kondisi | Seharusnya |
|---------|-----------|
| Pertanyaan di luar topik kampus (cuaca, berita, dll) | Ditolak Agent 1 |
| Pertanyaan tentang kampus tapi dokumennya tidak ada | Lolos ke Agent 2, Agent 2 yang bilang "tidak ditemukan" |

**Solusi:** Ubah filosofi Agent 1 — hanya blokir yang **jelas-jelas bukan tentang kampus**, loloskan semua yang berkaitan dengan kehidupan akademik/kampus:

```
TOLAK hanya untuk:
- Pertanyaan non-kampus: cuaca, resep, berita politik
- Topik kampus lain (bukan Telkom University)

LOLOSKAN semua yang berkaitan kampus, termasuk:
- Aturan kampus (berpakaian, fasilitas, dll)
- Panduan penggunaan AI di kampus
- Organisasi kemahasiswaan, beasiswa
```

#### Agent 2 — Retriever + Self-Correction Loop

```
search_chromadb(query)
    ↓
check_similarity_score() → score < 0.6?
    ↓ Ya                         ↓ Tidak
reformulate_query()           → lanjut ke Agent 3
    ↓
search_chromadb(query_baru)   (maks 2x retry)
    ↓
jika masih gagal → is_found = False
```

Strategi reformulasi berbeda per attempt:
- **Attempt 1:** Parafrase dengan sinonim akademik lebih spesifik
- **Attempt 2:** Perluas ke istilah lebih umum / konteks lebih luas

#### Agent 3 — Responder

Menggunakan structured output format dengan section markers:

```
[JAWABAN]
...

[DASAR HUKUM]
→ ...

[CATATAN]
→ ...

[DISCLAIMER]
...
```

Kemudian di-parse per section untuk memisahkan referensi dari jawaban utama.

#### Struggle: Double Arrow di Output

**Masalah:** Output menampilkan `Catatan: → → Tidak ada catatan tambahan.`

**Penyebab:** GPT menghasilkan `→ Tidak ada catatan tambahan.` di section `[CATATAN]`, lalu fungsi `format_response()` menambahkan `→` lagi saat memformat.

**Solusi:** Strip prefix `→` dari notes sebelum diformat:

```python
notes = sections["[CATATAN]"].strip().lstrip("→").strip()
```

Dan skip tampilkan section Catatan jika isinya "Tidak ada catatan tambahan":

```python
if notes and "tidak ada catatan tambahan" not in notes.lower():
    parts.append(f"→ {notes}")
```

---

### Fase 6 — LangGraph Pipeline

**File:** `graph/pipeline.py`

**Tujuan:** Mendefinisikan graph LangGraph yang mengorkestrasi ketiga agent dengan routing kondisional.

```
START → router → [retriever → responder | responder] → END
```

Routing setelah Agent 1:
- `is_relevant=True` dan `is_ambiguous=False` → masuk retriever
- Selainnya → langsung ke responder (yang mengembalikan rejection message)

Keputusan ini dibuat agar Agent 3 tetap menjadi **single exit point** — semua respons ke user melewati Agent 3, memastikan format output konsisten.

---

### Fase 7 — Streamlit UI

**File:** `main.py`

**Tujuan:** Chat interface yang user-friendly dengan fitur debug untuk keperluan research.

#### Struggle: `NameError` — Fungsi Dipanggil Sebelum Didefinisikan

**Masalah:** Error `NameError: name '_render_debug_panel' is not defined` muncul saat ada riwayat chat.

**Penyebab:** Streamlit mengeksekusi file Python dari atas ke bawah secara linear. Fungsi `_render_debug_panel` dipanggil di loop riwayat chat (baris ~80), tapi didefinisikan setelah loop tersebut (baris ~84).

**Solusi:** Pindahkan definisi fungsi ke atas file, sebelum digunakan pertama kali.

#### Fitur Debug Panel

Setelah beberapa iterasi, debug panel berkembang menjadi:

1. **Summary Agent Status** — status tiap agent (relevan, ditemukan, retry, kontradiksi, outdated)
2. **Logs Timeline** — setiap event dicatat dengan timestamp dan data:
   - Router: keputusan relevansi + alasan
   - Retriever: query yang digunakan, scores per chunk, chunk_ids, waktu eksekusi
   - Retriever: query reformulasi (jika terjadi)
   - Responder: model yang digunakan, jumlah chunk yang diproses
3. **Chunk Cards** — tiap chunk ditampilkan dengan:
   - `chunk_id` untuk crosscheck di ChromaDB Inspector
   - Similarity score + verdict (relevan/cukup relevan/kurang relevan)
   - Metadata lengkap: source, halaman, section, doc_level, method, word_count
   - Preview 40 kata pertama konten
4. **Download Log (JSON)** — export seluruh pipeline trace per sesi query

---

## 4. Tabel Struggle & Solusi

| # | Fase | Masalah | Penyebab | Solusi |
|---|------|---------|---------|--------|
| 1 | PDF Loader | Halaman TOC masuk sebagai konten | Tidak ada filter halaman daftar isi | Hitung pola `....` per halaman, skip jika ≥ 5 |
| 2 | PDF Loader | Nomor halaman romawi ikut ke teks | Tidak ada strip nomor halaman | Regex strip angka Arab dan Romawi di awal baris |
| 3 | PDF Loader | Baris TOC inline lolos filter | Filter hanya untuk seluruh halaman TOC | Tambah regex per-baris: `^.+\.{4,}\s*\d+$` |
| 4 | PDF Loader | `TIM PENYUSUN` lolos noise filter | Filter tidak cek setelah strip nomor halaman | Update urutan: strip nomor dulu, baru cek noise |
| 5 | Chunking | Chunk duplikat | Loop mulai dari `i=0` bukan `i=1` | Ubah loop start ke `i=1` |
| 6 | Chunking | PANDUAN_AI tidak terdeteksi subsection | Format `BAB I.` (ada titik) tidak dikenali regex | Tambah `\.?` di regex: `BAB\s+(?:\d+\|[IVX]+)\.?` |
| 7 | Chunking | Chunk terlalu pendek (< 30 kata) | Tidak ada minimum word count | Tambah `_MIN_WORDS=30`, merge ke chunk sebelumnya |
| 8 | ChromaDB | Dimension mismatch (384 vs 1536) | `query_texts=` memicu embedder default ChromaDB | Embed manual via OpenAI, gunakan `query_embeddings=` |
| 9 | Tools | Python 3.9 TypeError pada type hint | Syntax `X \| None` hanya Python 3.10+ | Ganti ke `Optional[X]` dari `typing` |
| 10 | Agent 1 | Pertanyaan kampus valid ditolak | System prompt terlalu spesifik daftarkan topik | Ubah filosofi: blokir hanya yang jelas non-kampus |
| 11 | Agent 3 | Double arrow `→ →` di output | GPT sudah output `→`, format_response tambah lagi | `lstrip("→")` sebelum format |
| 12 | UI | `NameError: _render_debug_panel` | Fungsi dipanggil sebelum didefinisikan | Pindah definisi fungsi ke atas file |

---

## 5. Evolusi Kualitas Chunk

### Sebelum Refactor (Pendekatan Per-Page)

```
Total chunk     : 1.435
Avg word count  : 58 kata
Chunk < 30 kata : banyak (tidak diukur)
Chunk duplikat  : 17
Strategi        : 1 (recursive saja)
```

**Masalah utama:** Chunk pendek dan duplikat tidak berguna untuk retrieval. Agent 2 akan sering gagal menemukan jawaban karena chunk tidak mengandung cukup konteks.

### Sesudah Refactor (Full-Document + 4 Strategi)

```
Total chunk     : 657
Avg word count  : 117 kata
Chunk < 30 kata : 0
Chunk duplikat  : 2
Strategi        : 4 (pasal, subsection, small_doc, recursive)
```

**Distribusi strategi pada 7 dokumen:**
- `pasal` → Peraturan Akademik (struktur paling konsisten)
- `subsection` → Panduan TA, Panduan KP, Panduan AI
- `small_doc` → Dokumen PU singkat (L2)
- `recursive` → Dokumen dengan struktur tidak konsisten

---

## 6. Keputusan Desain Penting

### 1. Full-Document vs Per-Page Chunking

**Keputusan:** Full-document dengan page_map.

**Alasan:** Per-page chunking menyebabkan chunk terpotong di batas halaman, kehilangan konteks. Full-document memastikan chunker bisa membuat potongan yang semantically coherent, lalu page_map digunakan untuk lookup nomor halaman yang akurat.

### 2. Cosine Similarity, bukan Euclidean Distance

**Keputusan:** ChromaDB dikonfigurasi dengan `hnsw:space: cosine`.

**Alasan:** Cosine similarity mengukur sudut antar vektor, bukan jarak absolut. Untuk teks akademik yang panjangnya bervariasi, cosine lebih stabil karena tidak terpengaruh oleh panjang dokumen.

### 3. Doc Level Hierarchy (L1/L2/L3)

**Keputusan:** Setiap dokumen diberi level otoritas (1=paling otoritatif).

**Alasan:** Ketika dua dokumen memberikan informasi yang berbeda (kontradiksi), sistem perlu tahu mana yang harus diprioritaskan. Peraturan universitas (L1) lebih otoritatif dari panduan teknis (L3).

### 4. Self-Correction Loop di Agent 2

**Keputusan:** Agent 2 bisa reformulasi query dan retry maks 2x sebelum menyatakan "tidak ditemukan".

**Alasan:** Query user sering menggunakan bahasa sehari-hari yang tidak cocok dengan bahasa formal di dokumen. Reformulasi query meningkatkan recall tanpa mengorbankan precision.

### 5. Single Exit Point melalui Agent 3

**Keputusan:** Semua respons ke user (termasuk penolakan dan "tidak ditemukan") diproses melalui Agent 3.

**Alasan:** Memastikan format output konsisten. Agent 3 yang mengontrol tone dan format jawaban, bukan masing-masing agent secara terpisah.

### 6. Agent 1 Hanya Blokir yang Jelas Non-Kampus

**Keputusan:** Router Agent tidak mencoba menebak apakah dokumen tersedia. Itu tugas Agent 2.

**Alasan:** Memisahkan tanggung jawab dengan jelas:
- Agent 1: Apakah pertanyaan ini tentang kampus? (domain check)
- Agent 2: Apakah jawabannya ada di dokumen? (retrieval check)

---

## 7. Hasil Akhir

### Sistem yang Berhasil Dibangun

```
preprocessing/
├── pdf_loader.py       ✅ Full-document extraction + page_map + noise filter
├── chunker.py          ✅ 4 strategi adaptif, 657 chunk berkualitas
├── embedder.py         ✅ Batch embedding, skip existing, cosine similarity
├── build_kb.py         ✅ Pipeline preprocessing end-to-end
├── inspect_chunks.py   ✅ Visual inspector Streamlit untuk raw chunks
├── inspect_chromadb.py ✅ Visual inspector ChromaDB + similarity search test
└── export_report.py    ✅ Export laporan HTML standalone

agents/
├── router_agent.py     ✅ Klasifikasi relevansi + deteksi ambigu
├── retriever_agent.py  ✅ ChromaDB search + self-correction loop
└── responder_agent.py  ✅ Format Level 3 + referensi + disclaimer

graph/
└── pipeline.py         ✅ LangGraph graph + AgentState + routing kondisional

tools/
└── chromadb_tool.py    ✅ 6 tool functions + chunk_id tracking

main.py                 ✅ Streamlit chat UI + debug panel + download log
```

### Knowledge Base

```
Dokumen diproses : 7 PDF
Total chunk      : 657
Dimensi vektor   : 1536 (text-embedding-3-small)
Similarity metric: Cosine
Build time       : ~21 detik
```

### Kemampuan Sistem

- ✅ Menjawab pertanyaan akademik dengan referensi pasal dan halaman
- ✅ Menolak pertanyaan di luar topik akademik dengan sopan
- ✅ Meminta klarifikasi untuk pertanyaan ambigu
- ✅ Reformulasi query otomatis jika similarity score rendah (maks 2x)
- ✅ Deteksi kontradiksi antar dokumen + prioritaskan yang lebih otoritatif
- ✅ Flag peringatan untuk dokumen yang sudah outdated (> 1 tahun)
- ✅ Pipeline execution trace (logs) untuk setiap query
- ✅ Export log JSON per query untuk keperluan research

---

*Dokumen ini dibuat sebagai jurnal pengembangan untuk keperluan akademik.*
*Dikembangkan menggunakan Python 3.9, LangGraph, ChromaDB, dan OpenAI API.*
