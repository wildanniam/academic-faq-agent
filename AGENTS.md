# AGENTS.md — Student FAQ Agentic RAG
## Aturan & Pedoman Telkom University — Fakultas Informatika

---

## Deskripsi Proyek

Sistem multi-agent berbasis Agentic RAG yang menjawab pertanyaan seputar
aturan dan pedoman akademik Telkom University (khususnya Fakultas Informatika).
Sistem menerima pertanyaan dari mahasiswa atau dosen, lalu mencari jawaban
dari knowledge base yang dibangun dari dokumen PDF resmi kampus.

---

## Tech Stack

- Framework agentic : LangGraph
- LLM               : OpenAI GPT-4o-mini
- Embedding         : OpenAI text-embedding-3-small
- Vector DB         : ChromaDB (local, persistent)
- PDF Parser        : PyMuPDF (fitz)
- Chunking          : LangChain RecursiveCharacterTextSplitter
                      + custom regex per pasal (fallback)
- Web UI            : Streamlit
- Language          : Python 3.11+

---

## Struktur Proyek
```
student-faq-agent/
├── AGENTS.md
├── .env
├── requirements.txt
├── main.py                    # Entry point Streamlit UI
│
├── preprocessing/
│   ├── __init__.py
│   ├── pdf_loader.py          # Extract teks dari PDF (PyMuPDF)
│   ├── chunker.py             # Smart chunking (per pasal + fallback)
│   ├── embedder.py            # Embed + simpan ke ChromaDB
│   └── build_kb.py            # Script pipeline preprocessing (jalankan sekali)
│
├── agents/
│   ├── __init__.py
│   ├── router_agent.py        # Agent 1: klasifikasi & cek relevansi
│   ├── retriever_agent.py     # Agent 2: query ChromaDB + reasoning
│   └── responder_agent.py     # Agent 3: format output level 3
│
├── graph/
│   ├── __init__.py
│   └── pipeline.py            # LangGraph graph definition & state
│
├── tools/
│   ├── __init__.py
│   └── chromadb_tool.py       # Tool query ChromaDB untuk agent
│
├── vectordb/                  # ChromaDB persistent storage (auto-generated)
│
└── documents/                 # Letakkan PDF dokumen aturan di sini
    ├── aturan_akademik.pdf
    └── pedoman_fakultas_informatika.pdf
```

---

## Arsitektur Sistem

### Pipeline 1: Preprocessing (jalankan sekali)
```
PDF Dokumen (documents/)
        ↓
pdf_loader.py — PyMuPDF extract teks + metadata halaman
        ↓
chunker.py — deteksi pasal via regex → fallback RecursiveCharacterTextSplitter
             chunk_size=500, chunk_overlap=100
        ↓
embedder.py — OpenAI text-embedding-3-small
        ↓
ChromaDB — simpan permanen di vectordb/
```

Jalankan sekali dengan:
```bash
python preprocessing/build_kb.py
```

### Pipeline 2: Agentic RAG (setiap ada pertanyaan)
```
User input
        ↓
Agent 1 (Router)
  → Relevan? → lanjut ke Agent 2
  → Tidak relevan? → tolak sopan, stop
  → Ambigu? → minta klarifikasi, stop
        ↓
Agent 2 (Retriever + Reasoner)
  → Query ChromaDB (similarity search, k=3)
  → Evaluasi similarity score
  → Score rendah? → reformulate query, retry maks 2x
  → Tidak ditemukan? → flag, output langsung tanpa format
  → Kontradiksi? → flag kedua dokumen, prioritaskan terbaru
  → Outdated? → flag caution, arahkan verifikasi ke kampus
  → Berhasil? → lanjut ke Agent 3
        ↓
Agent 3 (Responder)
  → Format jawaban level 3
  → Susun referensi pasal + halaman
  → Tambah disclaimer
        ↓
Output ke user
```

---

## LangGraph State Schema
```python
from typing import TypedDict

class AgentState(TypedDict):
    # Input awal
    query: str                  # Pertanyaan dari user

    # Hasil Agent 1
    is_relevant: bool           # Pertanyaan relevan dengan dokumen?
    is_ambiguous: bool          # Pertanyaan ambigu?
    rejection_message: str      # Pesan kalau ditolak atau minta klarifikasi

    # Hasil Agent 2
    retrieved_chunks: list      # Chunk hasil ChromaDB
    similarity_scores: list     # Score tiap chunk
    retry_count: int            # Sudah retry berapa kali (maks 2)
    has_contradiction: bool     # Ada kontradiksi antar dokumen?
    is_outdated: bool           # Dokumen outdated (> 1 tahun)?
    is_found: bool              # Info ditemukan di knowledge base?

    # Hasil Agent 3
    final_answer: str           # Jawaban final ke user
    references: list            # Referensi pasal + halaman
    flags: list                 # Flag peringatan (outdated, kontradiksi)
```

---

## Tools (Custom — semua dibuat sendiri)

Semua tools adalah custom Python functions yang di-register ke LangGraph.
Tidak ada external API tools selain ChromaDB client.

### Agent 1 — Router

**`classify_question(query: str) -> dict`**
Klasifikasi pertanyaan user — apakah relevan dengan topik dokumen
yang tersedia di knowledge base, dan apakah ambigu.
```python
# Return:
{
    "is_relevant": bool,
    "is_ambiguous": bool,
    "reason": str
}
```

### Agent 2 — Retriever + Reasoner

**`search_chromadb(query: str, k: int = 3) -> list[dict]`**
Similarity search ke ChromaDB, return top-k chunk paling relevan.
```python
# Return:
[{
    "content": str,
    "source": str,
    "halaman": int,
    "tanggal_dokumen": str,
    "similarity_score": float
}]
```

**`check_similarity_score(results: list[dict], threshold: float = 0.6) -> bool`**
Evaluasi apakah hasil retrieval cukup relevan.
Return True kalau ada chunk dengan score >= threshold.

**`reformulate_query(original_query: str, attempt: int) -> str`**
Reformulasi query kalau score rendah sebelum retry.
Attempt menentukan strategi reformulasi (1 = sinonim, 2 = lebih general).

**`detect_contradiction(chunks: list[dict]) -> dict`**
Bandingkan antar chunk — deteksi apakah ada info yang bertentangan
dari dokumen atau tanggal yang berbeda.
```python
# Return:
{
    "has_contradiction": bool,
    "chunks_involved": list,
    "latest_chunk": dict
}
```

**`check_document_date(metadata: dict) -> bool`**
Cek apakah dokumen outdated (tanggal_dokumen > 1 tahun dari sekarang).
Return True kalau outdated.

**`get_document_metadata(chunk: dict) -> dict`**
Ambil metadata lengkap dari chunk untuk keperluan referensi output.
```python
# Return:
{
    "source": str,
    "halaman": int,
    "tanggal_dokumen": str,
    "pasal": str
}
```

### Agent 3 — Responder

**`format_response(answer: str, references: list, flags: list) -> str`**
Format jawaban final ke level 3 — susun jawaban, referensi pasal,
catatan flag, dan disclaimer. Return string siap tampil ke user.

---

## System Prompt per Agent

### Agent 1 — Router
```
Kamu adalah asisten akademik Telkom University yang ramah dan membantu.
Tugasmu adalah menentukan apakah pertanyaan user relevan dengan
topik aturan dan pedoman akademik Telkom University berdasarkan
dokumen yang tersedia di knowledge base.

Jika pertanyaan tidak relevan, tolak dengan sopan dan jelaskan
bahwa kamu hanya bisa menjawab seputar aturan dan pedoman akademik
Telkom University.

Jika pertanyaan ambigu, minta klarifikasi dengan ramah sebelum
melanjutkan proses.

Gunakan bahasa Indonesia yang semi-formal dan ramah.
```

### Agent 2 — Retriever + Reasoner
```
Tugasmu adalah mencari informasi yang relevan dari knowledge base
berdasarkan pertanyaan yang diberikan, lalu mengevaluasi hasil
retrieval secara objektif.

Evaluasi setiap chunk berdasarkan relevansi, deteksi kontradiksi
antar dokumen, dan tandai dokumen yang sudah outdated.

Berikan hasil evaluasi dalam format terstruktur untuk diteruskan
ke Agent 3. Jangan membuat jawaban sendiri — tugasmu hanya
retrieve dan evaluasi.
```

### Agent 3 — Responder
```
Kamu adalah asisten akademik Telkom University yang ramah dan membantu.
Tugasmu adalah menyusun jawaban final berdasarkan informasi yang
sudah ditemukan dari dokumen resmi kampus.

Selalu gunakan bahasa Indonesia yang semi-formal dan ramah.
Setiap jawaban wajib menyertakan referensi pasal dan halaman dokumen.
Selalu tambahkan disclaimer di akhir jawaban.

Jika ada flag kontradiksi, tampilkan kedua informasi dan
prioritaskan yang lebih baru.
Jika ada flag outdated, tambahkan peringatan verifikasi ke kampus.

Jangan pernah menjawab di luar informasi yang ditemukan dari
dokumen — tidak boleh hallucinate.
```

---

## Format Output (Level 3)
```
[Jawaban langsung dan jelas berdasarkan dokumen, bahasa semi-formal]

Dasar hukum:
→ [Nama dokumen], Pasal X ayat Y, hal. Z

Catatan:
→ [Jika ada kontradiksi atau info tambahan — opsional]

⚠️ Disclaimer:
Informasi ini berdasarkan dokumen resmi yang tersedia di sistem.
Untuk konfirmasi lebih lanjut, hubungi:
→ Bagian Akademik Fakultas Informatika
→ akademik.if@telkomuniversity.ac.id
```

---

## Kondisi Self-Correction Loop (Agent 2)

| Kondisi | Aksi |
|---|---|
| Similarity score < 0.6 | Reformulate query, retry maks 2x |
| Tidak ditemukan setelah 2x retry | Output langsung: "info tidak ditemukan, hubungi kampus" |
| Dua dokumen kontradiksi | Flag keduanya, prioritaskan tanggal terbaru |
| Dokumen outdated (> 1 tahun) | Tetap jawab + tambah flag peringatan verifikasi |
| Pertanyaan tidak relevan | Tolak sopan di Agent 1, tidak perlu retrieve |
| Pertanyaan ambigu | Minta klarifikasi di Agent 1 sebelum retrieve |

---

## Chunking Strategy
```python
# Prioritas 1: chunk per pasal (kalau struktur dokumen konsisten)
pattern = r'(Pasal\s+\d+|BAB\s+[IVX]+)'
chunks = re.split(pattern, text)

# Prioritas 2: fallback RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
```

Setiap chunk harus menyimpan metadata:
```python
metadata = {
    "source": "nama_file.pdf",
    "halaman": 12,
    "tanggal_dokumen": "2024-07-01",
    "chunk_id": 0
}
```

---

## Environment Variables

Buat file `.env` di root proyek:
```
OPENAI_API_KEY=sk-xxx
CHROMA_PERSIST_DIR=./vectordb
DOCUMENTS_DIR=./documents
SIMILARITY_THRESHOLD=0.6
MAX_RETRY=2
TOP_K=3
```

---

## Contoh Input & Output yang Diharapkan

### Input:
```
"Berapa maksimal SKS yang bisa diambil per semester?"
```

### Output yang diharapkan:
```
Halo! Berdasarkan aturan akademik yang berlaku, beban studi
maksimal per semester ditentukan berdasarkan IPK kamu
di semester sebelumnya:

→ IPK ≥ 3.00 : maksimal 24 SKS
→ IPK < 3.00 : maksimal 18 SKS

Dasar hukum:
→ Aturan Akademik Telkom University, Pasal 12, hal. 8

⚠️ Disclaimer:
Informasi ini berdasarkan dokumen resmi yang tersedia di sistem.
Untuk konfirmasi lebih lanjut, hubungi:
→ Bagian Akademik Fakultas Informatika
→ akademik.if@telkomuniversity.ac.id
```

---

## Catatan Penting untuk Codex

1. Jangan pernah hallucinate — kalau info tidak ada di ChromaDB,
   jawab jujur bahwa data tidak ditemukan.

2. Selalu sertakan referensi pasal dan halaman di setiap jawaban
   yang berhasil — ini krusial untuk metric Explainability (SAFE).

3. Kalau ada dua dokumen yang kontradiksi, tampilkan keduanya
   dan prioritaskan yang tanggal dokumennya lebih baru.

4. Agent 1 dan Agent 3 pakai persona asisten akademik yang ramah.
   Agent 2 tidak pakai persona — dia bekerja teknikal di balik layar.

5. Similarity threshold minimum 0.6 — di bawah itu harus
   reformulate query, bukan langsung jawab dengan hasil yang kurang relevan.

6. Format output selalu level 3 — jawaban + dasar hukum + disclaimer.
   Tidak boleh ada jawaban tanpa referensi dokumen.

7. State AgentState harus dipakai konsisten di semua agent —
   jangan buat variabel state baru di luar schema yang sudah didefinisikan.
