import os
import json
import re
from datetime import date
from typing import List, Dict, Any, Optional

from openai import OpenAI


# =========================
# 0) CONFIG
# =========================

DEFAULT_MODEL = os.getenv("ALE_MODEL", "qwen-plus-latest")

def _default_base_url() -> str:
    # DashScope OpenAI-compatible endpoint (Qwen)
    return os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _get_client(api_key: Optional[str] = None,
                base_url: Optional[str] = None,
                timeout_seconds: int = 180) -> OpenAI:
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_APIKEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY env var (or pass api_key=...).")
    return OpenAI(api_key=api_key, base_url=base_url or _default_base_url(), timeout=timeout_seconds)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Best-effort: pull first JSON object
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(m.group(0))


# =========================
# 1) SCHEMAS (קבוע בקוד)
# =========================

ENGINE_BUNDLE_TEMPLATE: Dict[str, Any] = {
    "adaptive_learning_engine_bundle": {
        "meta": {
            "bundle_version": "1.3",
            "language": "he",
            "purpose": "Course-agnostic adaptive learning engine (knowledge + style + pedagogy).",
        },
        "schemas": {
            "course_profile_schema": {
                "meta": {"course_id": "string", "generated_at": "YYYY-MM-DD", "version": "2.3"},
                "doc_registry": [
                    {"doc_id": "string", "type": "knowledge|style", "name": "string"}
                ],
                "knowledge_brain": {
                    "doctrines": [],
                    "statutes": [],
                    "precedents": [],
                    "topic_map": []  # optional, model may fill
                },
                "style_brain": {
                    "structure": {"detected_model": "string", "templates": []},
                    "voice_signature": {
                        "mandatory_phrasing": [],
                        "preferred_terms": [],
                        "avoid_terms": [],
                        "must_write_exactly": []
                    },
                    "grading_rubric": {
                        "weights": {},
                        "penalty_triggers": [],
                        "bonus_triggers": []
                    },
                    "style_sources": [],
                    "solutions_bank": {"enabled": True, "solutions": []}
                },
                "terminology": {
                    "extraction_pipeline": {
                        "enabled": True,
                        "max_terms": 200,
                        "sources_priority": ["style", "knowledge"],
                        "candidate_rules": {
                            "ngram_range": [1, 4],
                            "min_frequency_knowledge": 3,
                            "min_frequency_style": 2,
                            "prefer_headings": True,
                            "prefer_definitional_patterns": [
                                "משמעותו", "משמעותה", "מוגדר כ", "הכוונה ל", "להלן",
                                "נדרש", "תנאי", "מבחן"
                            ],
                        },
                        "canonicalization_rules": {
                            "prefer_style_term": True,
                            "normalize_hyphens_and_quotes": True,
                            "strip_definite_article_variants": True,
                        },
                        "cluster_rules": {"merge_threshold": 0.82, "allow_aliases": True},
                        "quality_gates": {"must_have_source_quote": True, "min_reliability": 0.55},
                    },
                    "canonical_terms": [],
                    "style_preferences": {
                        "prefer_canonical_over_alias": True,
                        "penalize_wrong_term_if_changes_meaning": True,
                        "warn_on_noncanonical_if_equivalent": True,
                    },
                },
                "trainer_state": {
                    "student_model": {
                        "weak_topics": [],
                        "repeat_misses": [],
                        "confidence_by_topic": {}
                    }
                },
            },

            "trainer_result_schema": {
                "mode": "coach|examiner|exam_retry",
                "score": {
                    "total": "0-100",
                    "breakdown": {
                        "issue_spotting": "int",
                        "rule_statement": "int",
                        "application": "int",
                        "conclusion": "int",
                        "style_precision": "int"
                    },
                },
                "diagnostics": [
                    {
                        "category": "terminology|doctrine|structure|application|comparison",
                        "error_type": "wrong_term|missing_element|misapplied_rule|wrong_order|unsupported_claim|missing_precedent|missing_exception|missing_point_vs_solution",
                        "symptom_in_answer": "string",
                        "why_wrong": "string",
                        "correct_rule_or_term": "string",
                        "fix": {
                            "rewrite_suggestion": "string",
                            "micro_steps": ["string_optional"]
                        },
                        "evidence": [
                            {
                                "doc_id": "string",
                                "doc_name": "string_optional",
                                "quote": "string",
                                "reference": {
                                    "topic_label": "string_optional",
                                    "find_hint": "string_optional"
                                }
                            }
                        ],
                        "severity": "low|medium|high",
                    }
                ],
                "improved_answer": {
                    "full_text": "string_optional",
                    "delta": [
                        {
                            "action": "added|removed|reordered|rephrased|term_normalized",
                            "where": "string",
                            "why": "string"
                        }
                    ]
                },
                "sharpening_paragraph": {
                    "title": "string_optional",
                    "explanation": "string_optional",
                    "memory_hook": "string_optional",
                    "one_check_question": "string_optional"
                },
                "comparison_to_solution": {
                    "solution_id": "string_optional",
                    "coverage_score": "0-100_optional",
                    "missing_points": ["string_optional"],
                    "extra_points": ["string_optional"],
                    "style_gap_notes": ["string_optional"]
                },
                "next_drill": {"one_question": "string_optional", "expected_points": ["string_optional"]},
                "telemetry_updates": {
                    "topic_miss": ["string_optional"],
                    "repeat_miss": ["string_optional"],
                    "confidence_delta": {"topic": "delta_optional"}
                },
            },
        },
        "instances": {
            "active_course_profile": None,
            "last_trainer_result": None
        },
    }
}


# =========================
# 2) PROMPTS
# =========================

COURSE_PROFILE_META_PROMPT = """אתה אנליסט משפטי ופדגוגי.
מטרה: לייצר JSON אחד בשם course_profile (בלי מעטפות), על בסיס KNOWLEDGE + STYLE.

החזר בפורמט:
{
  "meta": {"course_id": "...", "generated_at": "YYYY-MM-DD", "version": "2.3"},
  "doc_registry": [...],
  "knowledge_brain": { "doctrines": [...], "statutes": [...], "precedents": [...], "topic_map": [...] },
  "style_brain": {
     "structure": {...},
     "voice_signature": {...},
     "grading_rubric": {...},
     "style_sources": [...],
     "solutions_bank": { "enabled": true, "solutions": [] }
  },
  "terminology": { "extraction_pipeline": {...}, "canonical_terms": [], "style_preferences": {...} },
  "trainer_state": {...}
}

כללים:
- החזר JSON תקין בלבד.
- רק לפי הטקסט שסופק, בלי ידע חיצוני.
- אל תמציא ציטוטים. אם אין quote, אל תכלול מקור.
- canonical_terms יישאר ריק בשלב זה (ימולא בשלב ייעודי).
- solutions_bank.solutions יישאר ריק בשלב זה (יימולא בשלב ייעודי).
"""

TERMINOLOGY_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ מונחים לקורס ספציפי.
מטרה: להחזיר רק JSON בפורמט:
{ "terminology": { "canonical_terms": [ ... ] } }

כללים:
- רק מהטקסט שסופק, בלי ידע חיצוני.
- לכל מושג חייב להיות sources עם quote (עד 25 מילים). אחרת אל תכלול.
- canonical נקבע לפי STYLE אם מופיע שם, אחרת לפי KNOWLEDGE.
- definition 1–2 משפטים מתוך ההקשר של המקור המצוטט.
- usage_note נלמד מ-STYLE אם יש.
- max_terms 200.
"""

SOLUTIONS_BANK_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ פתרונות מופת/מבחנים פתורים (STYLE בלבד).
קלט: טקסטים של מסמכי STYLE מסומנים [DOC S#].

מטרה: לחלץ מקטעים שנראים כמו "שאלה+פתרון" או "פתרון מופת" ולהחזיר רק JSON:
{
  "solutions_bank": {
    "solutions": [
      {
        "solution_id": "SOL_###",
        "label": "שם קצר (למשל: פתרון מופת 2023 שאלה 2)",
        "question_hint": "רמז קצר לזיהוי השאלה (לא חובה)",
        "answer_text": "טקסט הפתרון כפי שמופיע (אפשר לקצר אם ארוך מדי, אבל לשמור מהות)",
        "sources": [{ "doc_id": "S1", "doc_name":"...", "quote": "ציטוט קצר שמראה שזה הפתרון" }]
      }
    ]
  }
}

כללים:
- STYLE בלבד. אל תשתמש ב-KNOWLEDGE.
- אל תמציא. לכל solution חייב מקור עם quote.
- אל תוציא יותר מ-30 solutions (מעדיפים את הברורים/המלאים).
- החזר JSON בלבד.
"""

ANSWER_GRADING_META_PROMPT = """אתה בודק תשובות משפטי-פדגוגי.
קלט: (1) course_profile מלא (כולל terminology + style_brain + knowledge_brain)
(2) השאלה (אופציונלי)
(3) תשובת הסטודנט
(4) doc_id->doc_name map (כדי להציג שם מסמך)

מטרה: להחזיר JSON אחד בשם trainer_result לפי schema:
- score.total + breakdown
- diagnostics[] שמסביר: מה הטעות, למה, ומה הפתרון + rewrite_suggestion + micro_steps
- לכל diagnostics חייב evidence עם quote קצר (עד 25 מילים) מהחומר שסופק.

חשוב: אין עמודים. במקום זה, לכל evidence החזר reference:
- reference.topic_label: שם נושא קצר שאפשר לחפש במחברת (למשל "הדרישה הצורנית", "גילוי דעת", "חריגים לכלל...")
- reference.find_hint: 2–6 מילות מפתח לחיפוש בטקסט

מצבים:
- mode="coach": מותר להחזיר improved_answer.full_text (גרסה משודרגת).
- mode="examiner": אל תחזיר תשובה מלאה משודרגת; רק רמזים ותיקונים נקודתיים.

כללים:
- רק לפי החומר שסופק, בלי ידע חיצוני.
- אל תמציא מקורות. אם אין quote ברור — אל תכלול evidence לקביעה הזו (או רד בחומרה).
- החזר JSON בלבד.
"""

EXAM_RETRY_COMPARE_META_PROMPT = """אתה בודק "מבחן לחזרה".
קלט: course_profile (כולל solutions_bank.solutions), שאלה (אופציונלי), תשובת סטודנט, doc_id->doc_name map.

מטרה: להחזיר trainer_result במצב mode="exam_retry" שמכיל:
1) score (כרגיל) לפי rubric
2) diagnostics (כרגיל) + category="comparison" כשנקודה חסרה מול פתרון מופת
3) comparison_to_solution:
   - solution_id שנבחר (הכי מתאים לשאלה/רמז/דמיון)
   - coverage_score 0-100: עד כמה התשובה כיסתה את נקודות הליבה בפתרון
   - missing_points: רשימת נקודות מהותיות שחסרות (לא ציטוטי משפטים שלמים)
   - extra_points: נקודות טובות שהסטודנט הוסיף
   - style_gap_notes: פערי סגנון מול הפתרון (מבנה/טרמינולוגיה/סדר)

אין עמודים. במקום זה, לכל evidence החזר reference.topic_label + reference.find_hint, כמו בבדיקה רגילה.

כללים:
- אסור להעתיק את answer_text של פתרון המופת לתוך הפלט.
- מותר רק לתאר נקודות חסרות/עודפות ולתת ניסוחי תיקון קצרים.
- כשאתה טוען שחסר משהו "לפי פתרון", תן evidence קצר מ-sources של אותו solution (quote קצר).
- החזר JSON בלבד.
"""


# =========================
# 3) PACK DOCS
# =========================

def _pack_docs_for_model(doc_registry: List[Dict[str, Any]],
                         doc_text_by_id: Dict[str, str],
                         only_style: bool = False) -> str:
    lines = ["DOC_REGISTRY:"]
    for d in doc_registry:
        if only_style and d["type"] != "style":
            continue
        lines.append(f"- {{doc_id: \"{d['doc_id']}\", type: \"{d['type']}\", name: \"{d['name']}\"}}")

    lines.append("\nTEXT_BLOBS:")
    for d in doc_registry:
        if only_style and d["type"] != "style":
            continue
        doc_id = d["doc_id"]
        blob = doc_text_by_id.get(doc_id, "")
        lines.append(f"\n[DOC {doc_id}]\n{blob}")

    return "\n".join(lines)


# =========================
# 4) BUILD COURSE BUNDLE
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
) -> Dict[str, Any]:
    """
    knowledge_docs/style_docs: [{ "name": "...", "text": "..." }]
    No pages required. We rely on topic_label + hints instead of PAGE.
    """
    if not knowledge_docs and not style_docs:
        raise ValueError("No documents provided.")

    doc_registry: List[Dict[str, Any]] = []
    doc_text_by_id: Dict[str, str] = {}

    def _add_docs(docs: List[Dict[str, str]], doc_type: str, prefix: str) -> None:
        for i, d in enumerate(docs, start=1):
            doc_id = f"{prefix}{i}"
            name = d.get("name") or f"{doc_type}_{i}.txt"
            doc_registry.append({"doc_id": doc_id, "type": doc_type, "name": name})
            doc_text_by_id[doc_id] = d.get("text", "")

    _add_docs(knowledge_docs, "knowledge", "K")
    _add_docs(style_docs, "style", "S")

    packed_all = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=False)
    packed_style_only = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=True)

    today = date.today().isoformat()
    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    # A) Base course_profile
    user_payload_profile = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_all}\n"
    resp_profile = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COURSE_PROFILE_META_PROMPT},
            {"role": "user", "content": user_payload_profile}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    course_profile = _extract_json_object(resp_profile.choices[0].message.content)

    # Normalize required branches
    course_profile.setdefault("meta", {})
    course_profile["meta"].setdefault("course_id", course_id)
    course_profile["meta"].setdefault("generated_at", today)
    course_profile["meta"].setdefault("version", "2.3")
    course_profile["doc_registry"] = doc_registry

    course_profile.setdefault("knowledge_brain", {"doctrines": [], "statutes": [], "precedents": [], "topic_map": []})
    course_profile.setdefault("style_brain", {})
    course_profile["style_brain"].setdefault("structure", {"detected_model": "", "templates": []})
    course_profile["style_brain"].setdefault("voice_signature", {"mandatory_phrasing": [], "preferred_terms": [], "avoid_terms": [], "must_write_exactly": []})
    course_profile["style_brain"].setdefault("grading_rubric", {"weights": {}, "penalty_triggers": [], "bonus_triggers": []})
    course_profile["style_brain"].setdefault("style_sources", [])
    course_profile["style_brain"].setdefault("solutions_bank", {"enabled": True, "solutions": []})

    course_profile.setdefault("terminology", {})
    course_profile["terminology"].setdefault("canonical_terms", [])
    course_profile["terminology"].setdefault("style_preferences", {
        "prefer_canonical_over_alias": True,
        "penalize_wrong_term_if_changes_meaning": True,
        "warn_on_noncanonical_if_equivalent": True,
    })

    course_profile.setdefault("trainer_state", {"student_model": {"weak_topics": [], "repeat_misses": [], "confidence_by_topic": {}}})

    # B) Terminology extraction
    user_payload_terms = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_all}\n"
    resp_terms = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TERMINOLOGY_EXTRACTION_META_PROMPT},
            {"role": "user", "content": user_payload_terms}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    terms_obj = _extract_json_object(resp_terms.choices[0].message.content)
    course_profile["terminology"]["canonical_terms"] = (terms_obj.get("terminology", {}) or {}).get("canonical_terms", []) or []

    # C) Solutions bank extraction (style only)
    if style_docs:
        user_payload_solutions = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_style_only}\n"
        resp_solutions = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SOLUTIONS_BANK_EXTRACTION_META_PROMPT},
                {"role": "user", "content": user_payload_solutions}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        sol_obj = _extract_json_object(resp_solutions.choices[0].message.content)
        solutions = (sol_obj.get("solutions_bank", {}) or {}).get("solutions", []) or []
        course_profile["style_brain"]["solutions_bank"] = {"enabled": True, "solutions": solutions}

    # D) Compose bundle
    bundle = json.loads(json.dumps(ENGINE_BUNDLE_TEMPLATE))
    root = bundle["adaptive_learning_engine_bundle"]
    root["instances"]["active_course_profile"] = course_profile
    root["instances"]["last_trainer_result"] = None

    # Store raw texts (optional but useful)
    root["instances"]["_raw_texts"] = {
        "doc_text_by_id": doc_text_by_id,
        "doc_id_to_name": {d["doc_id"]: d["name"] for d in doc_registry},
    }
    return bundle


# =========================
# 5) GRADE: normal trainer
# =========================

def grade_answer(
    bundle: Dict[str, Any],
    *,
    student_answer: str,
    question_text: Optional[str] = None,
    mode: str = "coach",  # "coach" | "examiner"
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    if mode not in ("coach", "examiner"):
        raise ValueError("mode must be 'coach' or 'examiner'.")

    root = bundle.get("adaptive_learning_engine_bundle", {}) or {}
    course_profile = (root.get("instances", {}) or {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    doc_map = ((root.get("instances", {}) or {}).get("_raw_texts", {}) or {}).get("doc_id_to_name", {}) or {}

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    payload = {
        "mode": mode,
        "question_text": question_text or "",
        "student_answer": student_answer,
        "course_profile": course_profile,
        "doc_id_to_name": doc_map,
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

    # store
    root["instances"]["last_trainer_result"] = trainer_result
    return trainer_result


# =========================
# 6) GRADE: exam retry
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
    root = bundle.get("adaptive_learning_engine_bundle", {}) or {}
    course_profile = (root.get("instances", {}) or {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    solutions_bank = (course_profile.get("style_brain", {}) or {}).get("solutions_bank", {}) or {}
    if not solutions_bank.get("solutions"):
        raise ValueError("No solutions found. Upload style documents that include solved exams/model answers.")

    doc_map = ((root.get("instances", {}) or {}).get("_raw_texts", {}) or {}).get("doc_id_to_name", {}) or {}

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    payload = {
        "mode": "exam_retry",
        "question_text": question_text or "",
        "student_answer": student_answer,
        "course_profile": course_profile,
        "doc_id_to_name": doc_map,
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

    root["instances"]["last_trainer_result"] = trainer_result
    return trainer_result
