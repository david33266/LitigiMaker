import os
import re
import json
import gzip
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from openai import OpenAI
from pypdf import PdfReader


# ----------------------------
# Paths / Storage
# ----------------------------

APP_DATA_DIR = os.getenv("APP_DATA_DIR", "app_data")
KNOWLEDGE_DIR = os.path.join(APP_DATA_DIR, "knowledge")
STYLE_DIR = os.path.join(APP_DATA_DIR, "style")  # exams / model answers
BUNDLE_DIR = os.path.join(APP_DATA_DIR, "bundles")

ALLOWED_EXT = {".txt", ".pdf"}


def ensure_dirs() -> None:
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(STYLE_DIR, exist_ok=True)
    os.makedirs(BUNDLE_DIR, exist_ok=True)


# ----------------------------
# OpenAI client
# ----------------------------

def get_client(api_key: Optional[str] = None, timeout_seconds: int = 180) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY (set it in Streamlit Secrets).")
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


# ----------------------------
# File IO helpers
# ----------------------------

def safe_filename(name: str) -> str:
    # keep hebrew, english, numbers, basic punctuation
    name = name.strip().replace("\x00", "")
    name = re.sub(r"[^\w\-.()\[\]\s\u0590-\u05FF]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] if name else "file"


def list_files(folder: str) -> List[Dict[str, Any]]:
    ensure_dirs()
    out = []
    if not os.path.exists(folder):
        return out
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isfile(path):
            out.append(
                {
                    "name": fn,
                    "path": path,
                    "size": os.path.getsize(path),
                    "ext": os.path.splitext(fn)[1].lower(),
                }
            )
    return out


def delete_file(path: str) -> bool:
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    # Basic extraction (no OCR)
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if txt:
            parts.append(f"[PAGE {i+1}]\n{txt}")
    return "\n\n".join(parts).strip()


def load_text_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ----------------------------
# Chunking / retrieval (simple, fast)
# ----------------------------

def normalize_text(s: str) -> str:
    s = s.replace("\u200f", " ").replace("\u200e", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunk = text[i:end]
        chunks.append(chunk)
        i = end - overlap
        if i < 0:
            i = 0
        if end == n:
            break
    return chunks


def simple_score(query: str, chunk: str) -> float:
    # lightweight lexical score
    q = normalize_text(query).lower()
    c = normalize_text(chunk).lower()
    q_tokens = set(re.findall(r"[\w\u0590-\u05FF']{2,}", q))
    if not q_tokens:
        return 0.0
    hits = 0
    for t in q_tokens:
        if t in c:
            hits += 1
    return hits / max(1, len(q_tokens))


def retrieve_chunks(query: str, chunks: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    scored = []
    for ch in chunks:
        sc = simple_score(query, ch["text"])
        if sc > 0:
            scored.append((sc, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(item[1], score=item[0]) for item in scored[:k]]


# ----------------------------
# Bundle format
# ----------------------------

@dataclass
class BundlePaths:
    knowledge_dir: str = KNOWLEDGE_DIR
    style_dir: str = STYLE_DIR
    bundle_dir: str = BUNDLE_DIR


def bundle_path(course_id: str) -> str:
    ensure_dirs()
    safe = safe_filename(course_id).replace(" ", "_")
    return os.path.join(BUNDLE_DIR, f"{safe}.json.gz")


def save_gz_json(path: str, data: Dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_gz_json(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(course_id: str) -> Optional[Dict[str, Any]]:
    path = bundle_path(course_id)
    if not os.path.exists(path):
        return None
    return load_gz_json(path)


# ----------------------------
# LLM helpers
# ----------------------------

import json
import re

def llm_json(client, model: str, system: str, user: str):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return json.loads(content)

    except Exception:
        # fallback: בלי response_format ואז חילוץ JSON מתוך טקסט
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError(f"Model did not return JSON. Got: {text[:200]}")
        return json.loads(m.group(0))



# ----------------------------
# Build bundle
# ----------------------------

def build_bundle(course_id: str, model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads all knowledge + style files, chunks them, and builds an indexed bundle.
    """
    ensure_dirs()
    client = get_client(api_key=api_key)

    knowledge_files = list_files(KNOWLEDGE_DIR)
    style_files = list_files(STYLE_DIR)

    raw_docs: List[Dict[str, Any]] = []
    all_chunks: List[Dict[str, Any]] = []

    def ingest_one(file_meta: Dict[str, Any], doc_type: str) -> None:
        text = load_text_any(file_meta["path"])
        text = normalize_text(text)
        if not text:
            return
        doc_id = f"{doc_type}:{file_meta['name']}"
        raw_docs.append(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "file_name": file_meta["name"],
                "ext": file_meta["ext"],
                "size": file_meta["size"],
                "text": text,
            }
        )
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}#c{idx+1}",
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "file_name": file_meta["name"],
                    "text": ch,
                }
            )

    for fm in knowledge_files:
        ingest_one(fm, "knowledge")
    for fm in style_files:
        ingest_one(fm, "style")

    # Extract topics (lightweight) using LLM on a small sample of chunks
    sample = all_chunks[: min(18, len(all_chunks))]
    sample_text = "\n\n".join([f"- [{c['chunk_id']}] {c['text'][:500]}" for c in sample])

    system = (
        "אתה מסווג חומר לימודי של קורס במשפטים. "
        "החזר JSON בלבד. "
        "המטרה: לייצר אינדקס נושאים שימושי ללימוד וחיפוש."
    )
    user = (
        "קיבלתי דוגמאות מקטעים מהמחברת/חומר. "
        "הפק JSON עם:\n"
        "topics: מערך של נושאים (מחרוזות קצרות), עד 20.\n"
        "glossary: מערך של אובייקטים {term, meaning_short} עד 25.\n"
        "heuristics: 5 כללי מפתח לזיהוי שאלה/נושא.\n\n"
        f"קטעים:\n{sample_text}"
    )

    index = {"topics": [], "glossary": [], "heuristics": []}
    if sample:
        try:
            index = llm_json(client, model, system, user)
        except Exception:
            # If model errors, keep minimal
            index = {"topics": [], "glossary": [], "heuristics": []}

    bundle = {
        "meta": {
            "course_id": course_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "counts": {
                "knowledge_files": len(knowledge_files),
                "style_files": len(style_files),
                "docs": len(raw_docs),
                "chunks": len(all_chunks),
            },
        },
        "index": index,
        "raw": {
            # keep full texts (you asked: full text in JSON)
            "docs": raw_docs
        },
        "chunks": all_chunks,
    }

    save_gz_json(bundle_path(course_id), bundle)
    return bundle


# ----------------------------
# Teacher Assistant
# ----------------------------

def answer_question(bundle: Dict[str, Any], question: str, model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    client = get_client(api_key=api_key)
    chunks = bundle.get("chunks", [])
    top = retrieve_chunks(question, chunks, k=7)

    evidence = [
        {"chunk_id": c["chunk_id"], "file_name": c["file_name"], "quote": c["text"][:350]}
        for c in top
    ]

    system = (
        "את עוזרת הוראה לקורס משפטים. "
        "עני בעברית תקנית וברורה. "
        "חובה לעגן בתשובה לפחות 2 ציטוטים קצרים מהראיות שסופקו. "
        "אם אין מספיק מידע, תגידי מה חסר."
        "החזר JSON בלבד."
    )
    user = (
        f"שאלה: {question}\n\n"
        f"ראיות (מקטעים מהמחברת/חומר):\n"
        + "\n".join([f"- [{e['chunk_id']}] {e['quote']}" for e in evidence])
        + "\n\nהפק JSON עם:\n"
          "answer: תשובה.\n"
          "topic: שם נושא קצר שכדאי לחפש במחברת.\n"
          "citations: מערך של עד 5 פריטים {chunk_id, why_relevant}."
    )

    out = llm_json(client, model, system, user)
    out["evidence"] = evidence
    return out


# ----------------------------
# Grading
# ----------------------------

def grade_answer(
    bundle: Dict[str, Any],
    question_text: str,
    student_answer: str,
    model: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Numeric scoring 0-100 + diagnostics + suggested topic to review.
    IMPORTANT: style/exam answers may be imperfect; prefer course material as ground truth.
    """
    client = get_client(api_key=api_key)

    chunks = bundle.get("chunks", [])
    # retrieve from knowledge primarily, but include style too
    top = retrieve_chunks(question_text + " " + student_answer, chunks, k=10)

    evidence = [
        {"chunk_id": c["chunk_id"], "file_name": c["file_name"], "doc_type": c["doc_type"], "quote": c["text"][:420]}
        for c in top
    ]

    system = (
        "את בודקת תשובות במבחן משפטים. "
        "תני ציון 0-100. "
        "הסמכי בעיקר על חומר המחברת (knowledge). "
        "פתרונות/תשובות במבחנים (style) עלולים להכיל טעויות — אם הם סותרים את המחברת, המחברת גוברת. "
        "החזר JSON בלבד."
    )

    user = (
        f"שאלה/נושא: {question_text}\n\n"
        f"תשובת הסטודנט:\n{student_answer}\n\n"
        "ראיות רלוונטיות (מהמחברת + מהמבחנים):\n"
        + "\n".join([f"- ({e['doc_type']}) [{e['chunk_id']}] {e['quote']}" for e in evidence])
        + "\n\n"
        "הפק JSON עם המבנה הבא:\n"
        "{\n"
        '  "score": {"total": number, "breakdown": [{"criterion": str, "points": number, "why": str}]},\n'
        '  "diagnostics": [{"error_type": str, "severity": "low|med|high", "why_wrong": str, '
        '"fix": {"rewrite_suggestion": str}, "evidence": [{"chunk_id": str, "quote": str}]}],\n'
        '  "model_answer": str,\n'
        '  "review_topic": str\n'
        "}\n"
        "כל evidence בתוך diagnostics חייב להיות ציטוט קצר מתוך הראיות שסופקו."
    )

    out = llm_json(client, model, system, user)
    out["evidence_used"] = evidence
    return out
