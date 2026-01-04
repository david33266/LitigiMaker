import os
import json
import re
import gzip
import hashlib
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI


# =========================
# 0) CONFIG
# =========================

DEFAULT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus-latest")
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

MAX_SOLUTIONS = 30
MAX_TERMS = 200


# =========================
# 1) STORAGE (local files)
# =========================

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        if os.path.exists(path + ".gz"):
            with gzip.open(path + ".gz", "rb") as f:
                return json.load(f)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Dict[str, Any], *, gzip_it: bool = False) -> None:
    if gzip_it:
        with gzip.open(path + ".gz", "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# =========================
# 2) JSON extraction & parsing helpers
# =========================

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback: try to locate first JSON object
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(m.group(0))

def _clamp_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(float(x))
    except Exception:
        return default
    return max(lo, min(hi, v))

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _find_page_for_quote(doc_text: str, quote: str) -> Optional[int]:
    """
    If quote exists, try to infer the last preceding PAGE: N marker.
    Returns page number or None.
    """
    if not doc_text or not quote:
        return None
    doc_text_norm = doc_text
    q = quote.strip()
    idx = doc_text_norm.find(q)
    if idx < 0:
        # try relaxed match
        q2 = _normalize_whitespace(q)
        doc2 = _normalize_whitespace(doc_text_norm)
        idx2 = doc2.find(q2)
        if idx2 < 0:
            return None
        # can't map back reliably to PAGE; return None
        return None

    before = doc_text_norm[:idx]
    pages = list(re.finditer(r"PAGE:\s*(\d+)", before))
    if not pages:
        return None
    return int(pages[-1].group(1))

def _verify_evidence_quotes(course_profile: Dict[str, Any], doc_text_by_id: Dict[str, str]) -> None:
    """
    Walk through terminology + solutions + diagnostics and:
    - If evidence.quote not found in corresponding doc text: keep it but mark page null.
    - If page missing: infer from quote when possible.
    """
    def fix_evidence_list(evs: List[Dict[str, Any]]) -> None:
        for ev in evs:
            doc_id = ev.get("doc_id")
            quote = (ev.get("quote") or "").strip()
            if not doc_id or not quote:
                continue
            txt = doc_text_by_id.get(doc_id, "")
            # page inference
            if ev.get("page") is None:
                inferred = _find_page_for_quote(txt, quote)
                ev["page"] = inferred
            # existence check (best-effort)
            if quote not in txt:
                # relaxed check
                if _normalize_whitespace(quote) not in _normalize_whitespace(txt):
                    # still not found — don't delete, just reduce confidence
                    ev.setdefault("verified", False)
                else:
                    ev.setdefault("verified", True)
            else:
                ev.setdefault("verified", True)

    # terms
    terms = (((course_profile.get("terminology") or {}).get("canonical_terms")) or [])
    for t in terms:
        fix_evidence_list(t.get("sources") or [])

    # solutions
    sols = ((((course_profile.get("style_brain") or {}).get("solutions_bank") or {}).get("solutions")) or [])
    for s in sols:
        fix_evidence_list(s.get("sources") or [])

def _pack_docs_for_model(doc_registry: List[Dict[str, Any]], doc_text_by_id: Dict[str, str], only_style: bool = False) -> str:
    lines = ["DOC_REGISTRY:"]
    for d in doc_registry:
        if only_style and d.get("type") != "style":
            continue
        lines.append(f"- {{doc_id: \"{d['doc_id']}\", type: \"{d['type']}\", name: \"{d['name']}\"}}")

    lines.append("\nTEXT_BLOBS:")
    for d in doc_registry:
        if only_style and d.get("type") != "style":
            continue
        doc_id = d["doc_id"]
        blob = doc_text_by_id.get(doc_id, "")
        lines.append(f"\n[DOC {doc_id}]\n{blob}")

    return "\n".join(lines)


# =========================
# 3) CLIENT
# =========================

def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None, timeout_seconds: int = 240) -> OpenAI:
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_APIKEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY env var (or pass api_key=...).")
    return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL, timeout=timeout_seconds)


# =========================
# 4) PROMPTS (strong + exam-based scoring + citations)
# =========================

COURSE_PROFILE_META_PROMPT = """אתה אנליסט משפטי ופדגוגי. אתה עובד רק מתוך הטקסט שסופק.

מטרה: להחזיר JSON אחד בלבד בשם course_profile (בלי מעטפות).

חשוב:
- המערכת אגנוסטית לקורס. אל תכניס דוגמאות ספציפיות (כמו דיני חוזים/ירושה) אלא אם הופיעו במסמכים עצמם.
- אל תמציא ציטוטים. אם אין quote אמיתי מתוך הטקסט שסופק — אל תוסיף מקור.
- אל תסיק עובדות משפטיות מעבר לטקסט.

החזר בפורמט:
{
  "meta": {"course_id": "...", "generated_at": "YYYY-MM-DD", "version": "3.0"},
  "doc_registry": [...],
  "knowledge_brain": {
     "items": [
        {
          "item_id": "KITEM_###",
          "kind": "statute|doctrine|precedent|concept|procedure",
          "canonical": "שם קנוני כפי שמקובל בקורס (עדיף לפי STYLE אם מופיע)",
          "statement": "נוסחת כלל/מבחן/הלכה בקיצור (1-3 משפטים)",
          "importance_score": 0-100,
          "sources": [{"doc_id":"K1","page":50,"quote":"עד 25 מילים"}]
        }
     ],
     "topic_map": [
        {"topic":"נושא", "related_item_ids":["KITEM_001","..."]}
     ]
  },
  "style_brain": {
     "structure": {"detected_model":"IRAC|Issue-Rule-Application-Conclusion|אחר", "templates":[...]},
     "voice_signature": {
        "mandatory_phrasing": [],
        "preferred_terms": [],
        "avoid_terms": [],
        "must_write_exactly": []
     },
     "grading_rubric": {
        "weights": {"issue_spotting":25,"rule_statement":25,"application":30,"conclusion":15,"style_precision":5},
        "penalty_triggers": [],
        "bonus_triggers": []
     },
     "style_sources": [{"doc_id":"S1","page":1,"quote":"עד 25 מילים"}],
     "solutions_bank": {"enabled": true, "solutions": []}
  },
  "terminology": {
     "canonical_terms": [],
     "style_preferences": {
        "prefer_canonical_over_alias": true,
        "penalize_wrong_term_if_changes_meaning": true,
        "warn_on_noncanonical_if_equivalent": true
     }
  },
  "trainer_state": {"student_model":{"weak_topics":[],"repeat_misses":[],"confidence_by_topic":{}}}
}

כללים:
- items: אל תייצר יותר מ-250 items.
- importance_score: הערכת חשיבות יחסית מתוך החומר (0-100).
- אם אין PAGE במסמך — page=null.
- החזר JSON תקין בלבד.
"""

TERMINOLOGY_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ מונחים לקורס ספציפי. רק מתוך הטקסט שסופק.

מטרה: להחזיר רק JSON:
{ "terminology": { "canonical_terms": [ ... ] } }

לכל term:
{
  "term_id":"TERM_###",
  "canonical":"המונח הקנוני",
  "aliases":[...],
  "definition":"1-2 משפטים לפי ההקשר",
  "category":"doctrine|statute|procedural|concept|other",
  "usage_note":"הערת שימוש/ניסוח מקובל לפי STYLE אם אפשר",
  "sources":[{"doc_id":"K1|S1","page":50,"quote":"עד 25 מילים"}],
  "reliability":0.0-1.0
}

כללים:
- בלי ידע חיצוני. בלי המצאות.
- לכל מושג חייב sources עם quote אמיתי. אחרת אל תכלול.
- canonical נקבע לפי STYLE אם מופיע שם; אחרת KNOWLEDGE.
- max_terms: 200.
- אם אין PAGE, page=null.
- החזר JSON בלבד.
"""

SOLUTIONS_BANK_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ פתרונות מופת/מבחנים פתורים (STYLE בלבד). רק מתוך הטקסט שסופק.

מטרה: להחזיר רק JSON:
{
  "solutions_bank": {
    "solutions": [
      {
        "solution_id":"SOL_###",
        "label":"שם קצר (למשל: מבחן 2023 שאלה 2)",
        "question_hint":"רמז לזיהוי השאלה (לא חובה)",
        "answer_text":"טקסט הפתרון כפי שמופיע (מותר לקצר אם ארוך מאוד, בלי לשנות מהות)",
        "key_points":[
          "נקודת חובה 1 (במילים שלך, קצר)",
          "נקודת חובה 2",
          "... עד 12 נקודות"
        ],
        "sources":[{"doc_id":"S1","page":3,"quote":"עד 25 מילים שמראה שזה פתרון"}]
      }
    ]
  }
}

כללים:
- STYLE בלבד.
- לכל solution חייב מקור עם quote אמיתי.
- לא יותר מ-30 solutions.
- key_points: תמצת את "נקודות הליבה" של הפתרון כדי שנוכל לנקד לפי כיסוי.
- אל תדחוף ידע שלא הופיע בפתרון עצמו.
- החזר JSON בלבד.
"""

ANSWER_GRADING_META_PROMPT = """אתה בודק תשובות משפטי-פדגוגי.

קלט (JSON):
{
  "mode": "coach|examiner",
  "question_text": "...",
  "student_answer": "...",
  "course_profile": {...}
}

מטרה: להחזיר JSON אחד בשם trainer_result לפי הסכמה:

{
 "mode":"coach|examiner",
 "score":{
   "total":0-100,
   "breakdown":{"issue_spotting":0-25,"rule_statement":0-25,"application":0-30,"conclusion":0-15,"style_precision":0-5}
 },
 "diagnostics":[
   {
     "category":"terminology|doctrine|structure|application",
     "error_type":"wrong_term|missing_element|misapplied_rule|wrong_order|unsupported_claim|missing_exception",
     "symptom_in_answer":"...",
     "why_wrong":"מה הטעות",
     "correct_rule_or_term":"מה הנכון",
     "fix":{"rewrite_suggestion":"איך לתקן (משפט-שניים)","micro_steps":["צעד 1","צעד 2"]},
     "evidence":[{"doc_id":"K1|S1","page":50,"quote":"עד 25 מילים"}],
     "severity":"low|medium|high"
   }
 ],
 "improved_answer":{"full_text":"(רק אם coach)","delta":[...]},
 "sharpening_paragraph":{"title":"נקודת חידוד","explanation":"פסקת הסבר","memory_hook":"טריגר לזיכרון","one_check_question":"שאלת בדיקה קצרה"},
 "review_targets":[
    {"topic":"מה לחזור עליו","doc_id":"K1","page":50,"why":"למה זה החור הכי חשוב"}
 ]
}

כללים קשיחים:
- כל טענה מהותית חייב evidence מהחומר שסופק (knowledge/style). בלי מקור? תכתוב בזהירות, ושים severity נמוך.
- mode=examiner: אל תחזיר improved_answer.full_text מלא; רק תיקונים נקודתיים.
- החזר JSON בלבד.
"""

EXAM_RETRY_COMPARE_META_PROMPT = """אתה בודק "מבחן לחזרה" עם פתרונות מופת.

קלט (JSON):
{
 "mode":"exam_retry",
 "question_text":"...",
 "student_answer":"...",
 "course_profile": {...}  // כולל style_brain.solutions_bank.solutions
}

מטרה: להחזיר trainer_result במצב exam_retry עם ציון מספרי (0-100) שמבוסס על:
1) grading_rubric (מבנה כללי)
2) התאמה לפתרון מופת מתאים (solutions_bank)
3) כיסוי key_points של הפתרון

פלט:
{
 "mode":"exam_retry",
 "score": {...},
 "diagnostics":[...],
 "comparison_to_solution":{
   "solution_id":"SOL_###",
   "coverage_score":0-100,
   "missing_points":["נקודה חסרה 1",...],
   "extra_points":["נקודה טובה שהוספת",...],
   "style_gap_notes":["פער סגנוני",...]
 },
 "review_targets":[{"topic":"...","doc_id":"K1|S1","page":50,"why":"..."}]
}

כללים:
- אסור להעתיק את answer_text של פתרון המופת אל הפלט.
- מותר להשתמש רק ב-key_points כתיאור נקודות.
- כשאתה אומר "חסר X לפי הפתרון", תן evidence קצר מתוך sources של אותו solution (quote קצר).
- החזר JSON בלבד.
"""


# =========================
# 5) SCHEMAS / bundle skeleton
# =========================

def new_engine_bundle() -> Dict[str, Any]:
    return {
        "adaptive_learning_engine_bundle": {
            "meta": {
                "bundle_version": "3.0",
                "language": "he",
                "purpose": "Course-agnostic adaptive learning engine (knowledge + style + pedagogy).",
            },
            "instances": {
                "active_course_profile": None,
                "last_trainer_result": None,
                "raw_materials": None,  # optional: store full texts here too
            },
        }
    }


# =========================
# 6) BUILD: course bundle (profile + terms + solutions)
# =========================

def build_course_bundle(
    *,
    course_id: str,
    knowledge_docs: List[Dict[str, str]],
    style_docs: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 240,
    keep_full_texts_in_bundle: bool = True,
) -> Dict[str, Any]:
    """
    Input docs: [{"name": "...", "text": "..."}]
    TIP: for page references, insert PAGE: N in the text.
    """

    # 1) registry + texts
    doc_registry: List[Dict[str, Any]] = []
    doc_text_by_id: Dict[str, str] = {}

    def add_docs(docs: List[Dict[str, str]], doc_type: str, prefix: str) -> None:
        for i, d in enumerate(docs, start=1):
            doc_id = f"{prefix}{i}"
            txt = d.get("text", "") or ""
            doc_registry.append({
                "doc_id": doc_id,
                "type": doc_type,
                "name": d.get("name") or f"{doc_type}_{i}.txt",
                "sha256": sha256_text(txt),
                "pages": None,
            })
            doc_text_by_id[doc_id] = txt

    add_docs(knowledge_docs, "knowledge", "K")
    add_docs(style_docs, "style", "S")

    packed_all = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=False)
    packed_style_only = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=True)

    today = date.today().isoformat()
    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    # 2) course_profile
    user_payload_profile = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_all}\n"
    resp_profile = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COURSE_PROFILE_META_PROMPT},
            {"role": "user", "content": user_payload_profile},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    course_profile = _extract_json_object(resp_profile.choices[0].message.content)

    # harden
    course_profile.setdefault("meta", {})
    course_profile["meta"].setdefault("course_id", course_id)
    course_profile["meta"].setdefault("generated_at", today)
    course_profile["meta"].setdefault("version", "3.0")
    course_profile["doc_registry"] = doc_registry

    course_profile.setdefault("knowledge_brain", {})
    course_profile["knowledge_brain"].setdefault("items", [])
    course_profile["knowledge_brain"].setdefault("topic_map", [])

    course_profile.setdefault("style_brain", {})
    course_profile["style_brain"].setdefault("grading_rubric", {
        "weights": {"issue_spotting":25,"rule_statement":25,"application":30,"conclusion":15,"style_precision":5},
        "penalty_triggers": [],
        "bonus_triggers": [],
    })
    course_profile["style_brain"].setdefault("structure", {"detected_model":"IRAC", "templates":[]})
    course_profile["style_brain"].setdefault("voice_signature", {
        "mandatory_phrasing": [],
        "preferred_terms": [],
        "avoid_terms": [],
        "must_write_exactly": [],
    })
    course_profile["style_brain"].setdefault("solutions_bank", {"enabled": True, "solutions": []})
    course_profile["style_brain"].setdefault("style_sources", [])

    course_profile.setdefault("terminology", {})
    course_profile["terminology"].setdefault("canonical_terms", [])
    course_profile["terminology"].setdefault("style_preferences", {
        "prefer_canonical_over_alias": True,
        "penalize_wrong_term_if_changes_meaning": True,
        "warn_on_noncanonical_if_equivalent": True,
    })

    course_profile.setdefault("trainer_state", {"student_model": {"weak_topics": [], "repeat_misses": [], "confidence_by_topic": {}}})

    # 3) terminology
    user_payload_terms = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_all}\n"
    resp_terms = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TERMINOLOGY_EXTRACTION_META_PROMPT},
            {"role": "user", "content": user_payload_terms},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    terms_obj = _extract_json_object(resp_terms.choices[0].message.content)
    canonical_terms = (((terms_obj.get("terminology") or {}).get("canonical_terms")) or [])[:MAX_TERMS]
    course_profile["terminology"]["canonical_terms"] = canonical_terms

    # 4) solutions bank (with key_points)
    user_payload_solutions = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_style_only}\n"
    resp_solutions = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SOLUTIONS_BANK_EXTRACTION_META_PROMPT},
            {"role": "user", "content": user_payload_solutions},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    sol_obj = _extract_json_object(resp_solutions.choices[0].message.content)
    solutions = (((sol_obj.get("solutions_bank") or {}).get("solutions")) or [])[:MAX_SOLUTIONS]
    course_profile["style_brain"]["solutions_bank"] = {"enabled": True, "solutions": solutions}

    # 5) verify evidence pages / quotes best-effort
    _verify_evidence_quotes(course_profile, doc_text_by_id)

    # 6) compose bundle
    bundle = new_engine_bundle()
    bundle["adaptive_learning_engine_bundle"]["instances"]["active_course_profile"] = course_profile

    if keep_full_texts_in_bundle:
        bundle["adaptive_learning_engine_bundle"]["instances"]["raw_materials"] = {
            "doc_text_by_id": doc_text_by_id,   # full converted texts
        }

    return bundle


# =========================
# 7) GRADE: normal trainer
# =========================

def grade_answer(
    bundle: Dict[str, Any],
    *,
    student_answer: str,
    question_text: Optional[str] = None,
    mode: str = "coach",  # coach | examiner
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    if mode not in ("coach", "examiner"):
        raise ValueError("mode must be 'coach' or 'examiner'.")

    course_profile = (((bundle.get("adaptive_learning_engine_bundle") or {}).get("instances")) or {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    payload = {
        "mode": mode,
        "question_text": question_text or "",
        "student_answer": student_answer or "",
        "course_profile": course_profile,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_GRADING_META_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    trainer_result = _extract_json_object(resp.choices[0].message.content)

    # numeric hardening
    trainer_result.setdefault("mode", mode)
    trainer_result.setdefault("score", {})
    trainer_result["score"].setdefault("breakdown", {})
    br = trainer_result["score"]["breakdown"]
    br["issue_spotting"] = _clamp_int(br.get("issue_spotting"), 0, 25)
    br["rule_statement"] = _clamp_int(br.get("rule_statement"), 0, 25)
    br["application"] = _clamp_int(br.get("application"), 0, 30)
    br["conclusion"] = _clamp_int(br.get("conclusion"), 0, 15)
    br["style_precision"] = _clamp_int(br.get("style_precision"), 0, 5)
    trainer_result["score"]["total"] = _clamp_int(trainer_result["score"].get("total"), 0, 100, default=sum(br.values()))

    bundle["adaptive_learning_engine_bundle"]["instances"]["last_trainer_result"] = trainer_result
    return trainer_result


# =========================
# 8) GRADE: exam retry vs solved exams (numeric score + coverage)
# =========================

def grade_exam_retry(
    bundle: Dict[str, Any],
    *,
    student_answer: str,
    question_text: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 220,
) -> Dict[str, Any]:
    course_profile = (((bundle.get("adaptive_learning_engine_bundle") or {}).get("instances")) or {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    solutions_bank = ((course_profile.get("style_brain") or {}).get("solutions_bank") or {})
    solutions = solutions_bank.get("solutions") or []
    if not solutions:
        raise ValueError("No solutions found in style_brain.solutions_bank. Upload solved exams and rebuild bundle.")

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    payload = {
        "mode": "exam_retry",
        "question_text": question_text or "",
        "student_answer": student_answer or "",
        "course_profile": course_profile,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXAM_RETRY_COMPARE_META_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    trainer_result = _extract_json_object(resp.choices[0].message.content)

    # numeric hardening
    trainer_result.setdefault("mode", "exam_retry")
    trainer_result.setdefault("score", {})
    trainer_result["score"].setdefault("breakdown", {})
    br = trainer_result["score"]["breakdown"]
    br["issue_spotting"] = _clamp_int(br.get("issue_spotting"), 0, 25)
    br["rule_statement"] = _clamp_int(br.get("rule_statement"), 0, 25)
    br["application"] = _clamp_int(br.get("application"), 0, 30)
    br["conclusion"] = _clamp_int(br.get("conclusion"), 0, 15)
    br["style_precision"] = _clamp_int(br.get("style_precision"), 0, 5)
    trainer_result["score"]["total"] = _clamp_int(trainer_result["score"].get("total"), 0, 100, default=sum(br.values()))

    # coverage hardening
    comp = trainer_result.get("comparison_to_solution") or {}
    if comp:
        comp["coverage_score"] = _clamp_int(comp.get("coverage_score"), 0, 100)

    bundle["adaptive_learning_engine_bundle"]["instances"]["last_trainer_result"] = trainer_result
    return trainer_result
