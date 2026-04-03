"""
responder_agent.py — Agent 3: Format dan susun jawaban final ke user.

Tugasnya:
    1. Terima chunk dari Agent 2
    2. Generate jawaban via GPT-4o-mini berdasarkan chunk yang ditemukan
    3. Format output Level 3: jawaban + dasar hukum + catatan + disclaimer
    4. Kalau kontradiksi → tampilkan keduanya, prioritaskan yang terbaru
    5. Kalau outdated → tambah flag peringatan

Dipanggil dari pipeline.py sebagai node 'responder'.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"

_openai_client = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


_SYSTEM_PROMPT = """\
Kamu adalah asisten akademik Telkom University yang ramah dan membantu.
Tugasmu adalah menyusun jawaban final berdasarkan informasi yang
sudah ditemukan dari dokumen resmi kampus.

Selalu gunakan bahasa Indonesia yang semi-formal dan ramah.
Setiap jawaban wajib menyertakan referensi pasal dan halaman dokumen.
Selalu tambahkan disclaimer di akhir jawaban.

Jika ada flag kontradiksi, tampilkan kedua informasi dan
prioritaskan yang lebih baru.
Jika ada flag outdated, tambahkan peringatan verifikasi ke kampus.

JANGAN pernah menjawab di luar informasi yang ditemukan dari
dokumen — tidak boleh hallucinate.

Format output WAJIB seperti ini (ikuti persis):
[JAWABAN]
<jawaban langsung dan jelas berdasarkan dokumen>

[DASAR HUKUM]
→ <Nama dokumen>, <pasal/bagian>, hal. <nomor>

[CATATAN]
→ <catatan kontradiksi atau info tambahan — isi hanya kalau ada, kalau tidak ada isi "Tidak ada catatan tambahan.">

[DISCLAIMER]
⚠️ Informasi ini berdasarkan dokumen resmi yang tersedia di sistem.
Untuk konfirmasi lebih lanjut, hubungi:
→ Bagian Akademik Fakultas Informatika
→ akademik.if@telkomuniversity.ac.id
"""


def _build_context(state: dict) -> str:
    """Bangun context string dari chunks untuk dikirim ke GPT."""
    chunks = state.get("retrieved_chunks", [])
    contradiction_info = state.get("contradiction_info", {})
    is_outdated = state.get("is_outdated", False)

    lines = []

    if is_outdated:
        lines.append("⚠️ PERHATIAN: Setidaknya satu dokumen mungkin sudah outdated (> 1 tahun).\n")

    if contradiction_info.get("has_contradiction"):
        lines.append(f"⚠️ KONTRADIKSI TERDETEKSI: {contradiction_info.get('reason', '')}\n")

    lines.append("=== CHUNK YANG DITEMUKAN ===\n")

    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Chunk {i}]\n"
            f"Source    : {chunk.get('source', '')}\n"
            f"Halaman   : {chunk.get('halaman', '')}\n"
            f"Tanggal   : {chunk.get('tanggal_dokumen', '')}\n"
            f"Section   : {chunk.get('section', '')}\n"
            f"Doc Level : L{chunk.get('doc_level', 3)}\n"
            f"Score     : {chunk.get('similarity_score', 0):.4f}\n"
            f"Konten    :\n{chunk.get('content', '')}\n"
        )

    return "\n".join(lines)


def _parse_output(raw: str) -> tuple:
    """
    Parse output GPT menjadi komponen terpisah.
    Returns: (answer, references_list, notes, flags_list)
    """
    answer     = ""
    references = []
    notes      = ""
    flags      = []

    sections = {
        "[JAWABAN]": "",
        "[DASAR HUKUM]": "",
        "[CATATAN]": "",
        "[DISCLAIMER]": "",
    }

    current = None
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped in sections:
            current = stripped
        elif current is not None:
            sections[current] += line + "\n"

    answer = sections["[JAWABAN]"].strip()

    # Parse referensi
    for line in sections["[DASAR HUKUM]"].strip().split("\n"):
        if line.strip().startswith("→"):
            references.append(line.strip()[1:].strip())

    notes = sections["[CATATAN]"].strip().lstrip("→").strip()

    # Ambil disclaimer sebagai flag kalau ada outdated/kontradiksi di notes
    if "outdated" in notes.lower() or "kontradiksi" in notes.lower():
        flags.append(notes)

    return answer, references, notes, flags


def format_response(
    answer: str,
    references: list,
    flags: list,
    notes: str = "",
) -> str:
    """
    Format jawaban final ke Level 3.

    Args:
        answer     : Jawaban utama
        references : List referensi pasal/halaman
        flags      : List peringatan (outdated, kontradiksi)
        notes      : Catatan tambahan (opsional)

    Returns:
        String siap tampil ke user
    """
    parts = [answer, ""]

    if references:
        parts.append("Dasar hukum:")
        for ref in references:
            parts.append(f"→ {ref}")
        parts.append("")

    _notes_clean = notes.strip().rstrip(".")
    if notes and _notes_clean.lower() not in ("tidak ada catatan tambahan", ""):
        parts.append("Catatan:")
        parts.append(f"→ {notes}")
        parts.append("")

    if flags:
        for flag in flags:
            parts.append(f"⚠️ {flag}")
        parts.append("")

    parts.append(
        "⚠️ Disclaimer:\n"
        "Informasi ini berdasarkan dokumen resmi yang tersedia di sistem.\n"
        "Untuk konfirmasi lebih lanjut, hubungi:\n"
        "→ Bagian Akademik Fakultas Informatika\n"
        "→ akademik.if@telkomuniversity.ac.id"
    )

    return "\n".join(parts)


def responder_node(state: dict) -> dict:
    """
    LangGraph node untuk Agent 3 — Responder.

    Input state fields:
        query               : Pertanyaan user (asli)
        retrieved_chunks    : Chunks dari Agent 2
        is_found            : Apakah info ditemukan
        has_contradiction   : Ada kontradiksi?
        is_outdated         : Dokumen outdated?
        contradiction_info  : Detail kontradiksi
        rejection_message   : Pesan dari Agent 1 (kalau tidak relevan)
        is_relevant         : Apakah relevan
        is_ambiguous        : Apakah ambigu

    Output state fields:
        final_answer : Jawaban final ke user
        references   : List referensi pasal + halaman
        flags        : List peringatan
    """
    logs = list(state.get("logs", []))
    t0   = datetime.now()

    logs.append({
        "ts":    t0.isoformat(),
        "agent": "Responder (Agent 3)",
        "event": "START",
        "data": {
            "is_relevant":  state.get("is_relevant"),
            "is_ambiguous": state.get("is_ambiguous"),
            "is_found":     state.get("is_found"),
        },
    })

    # ── Kasus: tidak relevan atau ambigu (dari Agent 1) ──────────────────────
    if not state.get("is_relevant", True):
        logs.append({"ts": datetime.now().isoformat(), "agent": "Responder (Agent 3)", "event": "STOP — not relevant", "data": {}})
        return {
            **state,
            "final_answer": state.get("rejection_message", "Maaf, pertanyaanmu di luar topik yang bisa aku bantu."),
            "references":   [],
            "flags":        [],
            "logs":         logs,
        }

    if state.get("is_ambiguous", False):
        logs.append({"ts": datetime.now().isoformat(), "agent": "Responder (Agent 3)", "event": "STOP — ambiguous", "data": {}})
        return {
            **state,
            "final_answer": state.get("rejection_message", "Bisa tolong perjelas pertanyaanmu?"),
            "references":   [],
            "flags":        [],
            "logs":         logs,
        }

    # ── Kasus: info tidak ditemukan (dari Agent 2) ───────────────────────────
    if not state.get("is_found", False):
        original_query = state.get("query", "pertanyaan tersebut")
        logs.append({"ts": datetime.now().isoformat(), "agent": "Responder (Agent 3)", "event": "STOP — not found", "data": {}})
        return {
            **state,
            "final_answer": (
                f"Maaf, aku tidak menemukan informasi yang relevan mengenai "
                f'"{original_query}" di knowledge base yang tersedia.\n\n'
                "Mungkin topik ini belum tercakup dalam dokumen yang ada, "
                "atau coba ulangi pertanyaan dengan kata-kata yang berbeda.\n\n"
                "Untuk informasi lebih lanjut, kamu bisa menghubungi:\n"
                "→ Bagian Akademik Fakultas Informatika\n"
                "→ akademik.if@telkomuniversity.ac.id"
            ),
            "references":   [],
            "flags":        ["Info tidak ditemukan di knowledge base."],
            "logs":         logs,
        }

    # ── Generate jawaban via GPT ─────────────────────────────────────────────
    openai  = _get_openai()
    context = _build_context(state)
    query   = state.get("query", "")

    logs.append({
        "ts":    datetime.now().isoformat(),
        "agent": "Responder (Agent 3)",
        "event": "GENERATE — calling GPT",
        "data":  {"model": CHAT_MODEL, "n_chunks": len(state.get("retrieved_chunks", []))},
    })

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Pertanyaan user: {query}\n\n"
                    f"Informasi dari dokumen:\n{context}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()
    answer, references, notes, flags = _parse_output(raw)

    # Tambah flag outdated kalau ada
    if state.get("is_outdated", False) and "outdated" not in " ".join(flags).lower():
        flags.append(
            "Beberapa dokumen yang digunakan mungkin sudah lebih dari 1 tahun. "
            "Harap verifikasi ke kampus untuk memastikan informasi masih berlaku."
        )

    # Kalau parse gagal, pakai raw output langsung
    if not answer:
        answer = raw
        references = []
        notes = ""

    final = format_response(answer, references, flags, notes)

    logs.append({
        "ts":    datetime.now().isoformat(),
        "agent": "Responder (Agent 3)",
        "event": "DONE",
        "data": {
            "n_references": len(references),
            "n_flags":      len(flags),
            "elapsed_ms":   round((datetime.now() - t0).total_seconds() * 1000),
        },
    })

    return {
        **state,
        "final_answer": final,
        "references":   references,
        "flags":        flags,
        "logs":         logs,
    }
