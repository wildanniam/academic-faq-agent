"""
router_agent.py — Agent 1: Klasifikasi & filter relevansi pertanyaan.

Tugasnya:
    1. Cek apakah pertanyaan relevan dengan topik akademik Telkom University
    2. Cek apakah pertanyaan ambigu (perlu klarifikasi)
    3. Kalau tidak relevan / ambigu → stop, kembalikan pesan ke user
    4. Kalau relevan & jelas → lanjut ke Agent 2

Dipanggil dari pipeline.py sebagai node 'router'.
"""

import os
import json
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
Tugasmu adalah menentukan apakah pertanyaan user relevan dengan
topik kampus dan akademik Telkom University.

Gunakan bahasa Indonesia yang semi-formal dan ramah.

TOLAK (is_relevant: false) HANYA untuk pertanyaan yang jelas-jelas
tidak ada hubungannya dengan kampus atau kehidupan akademik sama sekali:
- Pertanyaan umum non-kampus: cuaca, resep masakan, berita politik, olahraga, hiburan
- Topik kampus lain (bukan Telkom University)
- Permintaan yang tidak masuk akal atau berbahaya

LOLOSKAN (is_relevant: true) semua pertanyaan yang masih berkaitan
dengan kampus, akademik, atau kehidupan mahasiswa di Telkom University,
termasuk topik yang mungkin tidak ada dokumennya — biarkan sistem
pencarian yang menentukan apakah jawabannya tersedia atau tidak.
Contoh yang harus diloloskan:
- Aturan akademik, SKS, IPK, cuti, kelulusan, sidang, wisuda
- Aturan kampus (berpakaian, kedisiplinan, fasilitas, dll)
- Panduan teknis dan pedoman mahasiswa
- Penggunaan AI di kampus, etika akademik
- Organisasi kemahasiswaan, beasiswa, kegiatan kampus
- Pertanyaan seputar dosen, jurusan, program studi

AMBIGU (is_ambiguous: true) jika pertanyaan terlalu pendek atau tidak
jelas sehingga tidak bisa diproses sama sekali.
Contoh: "aturannya?", "gimana caranya?", "boleh ga?"

Kembalikan HANYA JSON dengan format ini:
{
  "is_relevant": true/false,
  "is_ambiguous": true/false,
  "reason": "alasan singkat keputusan",
  "message": "pesan ke user (hanya kalau not relevant atau ambigu, kosongkan kalau relevan)"
}
"""


def classify_question(query: str) -> dict:
    """
    Klasifikasi pertanyaan user.

    Args:
        query : Pertanyaan dari user

    Returns:
        {
            "is_relevant": bool,
            "is_ambiguous": bool,
            "reason": str,
            "message": str   # pesan ke user kalau perlu stop
        }
    """
    openai = _get_openai()

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback kalau JSON parse gagal — asumsikan relevan agar tidak block
        result = {
            "is_relevant": True,
            "is_ambiguous": False,
            "reason": "Parse gagal, diteruskan ke retriever.",
            "message": "",
        }

    # Pastikan semua field ada
    result.setdefault("is_relevant", True)
    result.setdefault("is_ambiguous", False)
    result.setdefault("reason", "")
    result.setdefault("message", "")

    return result


# ── Node function untuk LangGraph ────────────────────────────────────────────

def router_node(state: dict) -> dict:
    """
    LangGraph node untuk Agent 1 — Router.

    Input state fields  : query
    Output state fields : is_relevant, is_ambiguous, rejection_message
    """
    query = state.get("query", "")
    result = classify_question(query)

    return {
        **state,
        "is_relevant":        result["is_relevant"],
        "is_ambiguous":       result["is_ambiguous"],
        "rejection_message":  result.get("message", ""),
    }
