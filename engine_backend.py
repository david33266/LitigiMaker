import os
import json
import re
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

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
                    {"doc_id": "string", "type": "knowledge|style", "name": "string", "sha256": "string_optional", "pages": "int_optional"}
                ],
                "knowledge_brain": {"doctrines": [], "statutes": [], "precedents": [], "topic_map": [], "topic_index": []},
                "style_brain": {
                    "structure": {"detected_model": "string", "templates": []},
                    "voice_signature": {"mandatory_phrasing": [], "preferred_terms": [], "avoid_terms": [], "must_write_exactly": []},
                    "grading_rubric": {"weights": {}, "penalty_triggers": [], "bonus_triggers": []},
                    "style_sources": [],
                    "question_bank": {"enabled": True, "questions": []},
                    "solutions_bank": {"enabled": True, "solutions": []},
                },
                "terminology": {"canonical_terms": []},
                "trainer_state": {"student_model": {"weak_topics": [], "repeat_misses": [], "confidence_by_topic": {}}},
                "raw_docs": {"knowledge": {}, "style": {}},  # נשמור טקסטים שהומרו
            },

            "trainer_result_schema": {
                "mode": "coach|examiner|exam_retry|assistant",

                "score": {
                    "total": "0-100",
                    "breakdown": {"issue_spotting": "int", "rule_statement": "int", "application": "int", "conclusion": "int", "style_precision": "int"},
                },

                "diagnostics": [
                    {
                        "category": "terminology|doctrine|structure|application|comparison",
                        "error_type": "wrong_term|missing_element|misapplied_rule|wrong_order|unsupported_claim|missing_precedent|missing_exception|missing_point_vs_solution",
                        "symptom_in_answer": "string",
                        "why_wrong": "string",
                        "correct_rule_or_term": "string",
                        "fix": {"rewrite_suggestion": "string", "micro_steps": ["string_optional"]},
                        "evidence": [{"doc_id": "string", "page": "int|null", "quote": "string", "location": "string_optional"}],
                        "severity": "low|medium|high",
                    }
                ],

                "improved_answer": {"full_text": "string_optional", "delta": [{"action": "added|removed|reordered|rephrased|term_normalized", "where": "string", "why": "string"}]},

                "sharpening_paragraph": {"title": "string_optional", "explanation": "string_optional", "memory_hook": "string_optional", "one_check_question": "string_optional"},

                "comparison_to_solution": {
                    "solution_id": "string_optional",
                    "coverage_score": "0-100_optional",
                    "missing_points": ["string_optional"],
                    "extra_points": ["string_optional"],
                    "style_gap_notes": ["string_optional"]
                },

                "review_plan": {
                    "recommended_topics": ["string_optional"],
                    "search_hints": ["string_optional"],
                    "supporting_quotes": [{"doc_id": "string", "quote": "string", "location": "string_optional"}]
                },

                "assistant_answer": {"answer": "string_optional", "citations": [{"doc_id": "string", "quote": "string", "location": "string_optional"}]}
            },
        },
        "instances": {"active_course_profile": None, "last_trainer_result": None},
    }
}

# =========================
# 2) PROMPTS
# =========================

TOPIC_INDEX_PROMPT = """אתה מנוע אינדוקס למחברת קורס משפטי.
קלט: מסמכי KNOWLEDGE מסומנים [DOC K#]. בתוך הטקסט יש לעיתים PAGE: N.
מטרה: להחזיר JSON בלבד בפורמט:
{
  "topic_index": [
    {
      "topic": "שם נושא",
      "subtopics": ["..."],
      "keywords": ["..."],
      "summary": "2-4 משפטים תמצית",
      "sources": [{"doc_id":"K1","page":null,"quote":"עד 25 מילים"}]
    }
  ]
}

כללים:
- בלי ידע חיצוני.
- לכל item חייב לפחות מקור אחד עם quote שמופיע בטקסט.
- אם אין PAGE: אז page=null.
- עד 80 נושאים (מעדיפים נושאים מרכזיים).
"""

TERMINOLOGY_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ מונחים לקורס ספציפי.
מטרה: להחזיר רק JSON בפורמט:
{ "canonical_terms": [ ... ] }

מבנה מונח:
{
  "term_id": "TERM_###",
  "canonical": "המונח הקנוני",
  "aliases": ["..."],
  "definition": "1-2 משפטים",
  "category": "doctrine|statute|procedural|concept",
  "usage_note": "אופציונלי",
  "sources": [{"doc_id":"K1","page":null,"quote":"עד 25 מילים"}],
  "reliability": 0.0-1.0
}

כללים:
- רק מהטקסט שסופק.
- לכל מושג חייב sources עם quote (עד 25 מילים). אחרת אל תכלול.
- max_terms 200.
- אם אין PAGE מספרי, page=null.
"""

QUESTIONS_BANK_EXTRACTION_PROMPT = """אתה מנוע חילוץ שאלות ממבחנים (STYLE בלבד).
קלט: מסמכי STYLE מסומנים [DOC S#], ולעיתים PAGE: N.

מטרה: להחזיר JSON בלבד:
{
  "questions": [
    {
      "question_id": "Q_###",
      "label": "שם קצר (למשל: מבחן 2023 שאלה 2)",
      "question_text": "טקסט השאלה כפי שמופיע (אפשר לקצר מעט אם ארוך, לשמור מהות)",
      "legal_issue": "מה השאלה המשפטית המרכזית",
      "expected_points": ["נקודה 1", "נקודה 2", "..."],
      "sources": [{"doc_id":"S1","page":null,"quote":"ציטוט קצר שמראה שזו השאלה"}]
    }
  ]
}

כללים:
- STYLE בלבד.
- אל תמציא. לכל שאלה חייב מקור עם quote.
- עד 40 שאלות.
- אם אין PAGE, page=null.
"""

SOLUTIONS_BANK_EXTRACTION_META_PROMPT = """אתה מנוע חילוץ פתרונות מופת/מבחנים פתורים (STYLE בלבד).
קלט: טקסטים של מסמכי STYLE מסומנים [DOC S#] ובתוכם PAGE: N אם קיים.

מטרה: להחזיר JSON בלבד:
{
  "solutions": [
    {
      "solution_id": "SOL_###",
      "label": "שם קצר (למשל: פתרון מופת 2023 שאלה 2)",
      "question_hint": "רמז קצר לזיהוי השאלה (לא חובה)",
      "answer_text": "טקסט הפתרון כפי שמופיע (אפשר לקצר אם ארוך מדי, אבל לשמור מהות)",
      "sources": [{ "doc_id": "S1", "page": null, "quote": "ציטוט קצר שמראה שזה הפתרון" }]
    }
  ]
}

כללים:
- STYLE בלבד.
- אל תמציא. לכל solution חייב מקור עם quote.
- עד 30 solutions.
- אם אין PAGE, page=null.
"""

ANSWER_GRADING_META_PROMPT = """אתה בודק תשובות משפטי-פדגוגי.
קלט: (1) course_profile (כולל topic_index + terminology + style_brain + knowledge_brain)
(2) question_text (אופציונלי)
(3) student_answer
(4) retrieved_snippets: קטעים רלוונטיים מהמחברת/מבחנים שהמערכת שלפה מראש

מטרה: להחזיר JSON אחד (trainer_result):
- score.total + breakdown (0-100)
- diagnostics[] שמסביר: מה הטעות, למה, ומה הפתרון + rewrite_suggestion + micro_steps
- sharpening_paragraph: פסקת חידוד אחת
- review_plan: על מה לחזור (topics + search_hints + 1-3 ציטוטים קצרים)
כל תיקון מהותי חייב evidence (doc_id+quote) מתוך retrieved_snippets בלבד.
אל תמציא מקורות. אם אין מקור — הורד severity והימנע מקביעה נחרצת.

מצבים:
- mode="coach": מותר להחזיר improved_answer.full_text (גרסה משודרגת).
- mode="examiner": אל תחזיר תשובה מלאה משודרגת; רק תיקונים נקודתיים/רמזים.

החזר JSON בלבד.
"""

EXAM_RETRY_COMPARE_META_PROMPT = """אתה בודק "מבחן לחזרה".
קלט: course_profile (כולל solutions_bank.solutions), question_text (אופציונלי), student_answer,
retrieved_snippets: קטעים רלוונטיים.

מטרה: להחזיר trainer_result במצב mode="exam_retry" שמכיל:
1) score (0-100) + breakdown
2) diagnostics + category="comparison" כשנקודה חסרה מול פתרון מופת
3) comparison_to_solution:
   - solution_id שנבחר
   - coverage_score 0-100: כיסוי נקודות הליבה
   - missing_points / extra_points / style_gap_notes
4) review_plan: נושאים לחזרה + ציטוטים קצרים מתוך retrieved_snippets

כללים:
- אסור להעתיק את answer_text של פתרון המופת לתוך הפלט.
- evidence רק מתוך retrieved_snippets.
- החזר JSON בלבד.
"""

ASSISTANT_QA_PROMPT = """אתה "עוזר הוראה" לקורס משפטי.
קלט: question (שאלה של הסטודנט) + retrieved_snippets (קטעים רלוונטיים מהמחברת).
מטרה: להחזיר JSON בלבד:
{
  "assistant_answer": {
    "answer": "תשובה קצרה-ברורה בעברית תקנית",
    "citations": [
      {"doc_id":"K1","quote":"עד 25 מילים","location":"topic: ... / subtopic: ..."}
    ]
  }
}

כללים:
- בלי ידע חיצוני.
- תן 1-3 ציטוטים קצרים (quotes) מתוך retrieved_snippets בלבד.
"""

# =========================
# 3) HELPERS
# =========================

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback: first {...}
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(m.group(0))


def _get_client(
    api_key: Optional[str] = None,
    timeout_seconds: int = 180
) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY env var (or pass api_key=...).")
    return OpenAI(api_key=api_key, timeout=timeout_seconds)

def _pack_docs_for_model(doc_registry: List[Dict[str, Any]], doc_text_by_id: Dict[str, str], only_style: bool = False) -> str:
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


def _parse_page_from_near(text: str) -> Optional[int]:
    # אם יש PAGE: N בתוך ציטוט/סביבה – לפעמים יעזור, אבל אצלך רוב הזמן אין, אז נחזיר None.
    m = re.search(r"PAGE:\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _simple_retrieve_snippets(course_profile: Dict[str, Any], query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieval הכי פשוט: ניקוד לפי חפיפת מילות מפתח מול topic_index + fallback על raw knowledge text.
    מחזיר רשימת snippets עם doc_id + quote + location + full_text (לבדיקת quote).
    """
    q = _normalize_ws(query)
    q_tokens = set([t for t in re.split(r"[^\u0590-\u05FFA-Za-z0-9]+", q) if len(t) >= 3])

    topic_index = ((course_profile.get("knowledge_brain") or {}).get("topic_index") or [])
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for item in topic_index:
        kw = set([_normalize_ws(x) for x in (item.get("keywords") or []) if x])
        sub = set([_normalize_ws(x) for x in (item.get("subtopics") or []) if x])
        topic = _normalize_ws(item.get("topic") or "")
        bag = set()
        for x in list(kw) + list(sub) + ([topic] if topic else []):
            for t in re.split(r"[^\u0590-\u05FFA-Za-z0-9]+", x):
                if len(t) >= 3:
                    bag.add(t)

        overlap = len(q_tokens.intersection(bag))
        if overlap == 0:
            continue

        src0 = (item.get("sources") or [{}])[0] or {}
        scored.append((overlap, {
            "doc_id": src0.get("doc_id", "K?"),
            "page": src0.get("page", None),
            "quote": src0.get("quote", ""),
            "location": f"topic: {item.get('topic','')}" + (f" / subtopics: {', '.join(item.get('subtopics') or [])}" if item.get("subtopics") else ""),
            "full_text": item.get("summary","") + " " + " ".join(item.get("keywords") or []),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    snippets = [x[1] for x in scored[:k]]

    # fallback: אם אין – ננסה לקחת מהטקסט הגולמי של המחברת שורה-שתיים שמכילות טוקנים
    if not snippets:
        raw_knowledge = ((course_profile.get("raw_docs") or {}).get("knowledge") or {})
        for doc_id, txt in raw_knowledge.items():
            if not txt:
                continue
            # חיפוש “שורה” שמתאימה
            lines = txt.splitlines()
            best_line = ""
            best_score = 0
            for ln in lines:
                ln_n = _normalize_ws(ln)
                if len(ln_n) < 25:
                    continue
                toks = set([t for t in re.split(r"[^\u0590-\u05FFA-Za-z0-9]+", ln_n) if len(t) >= 3])
                ov = len(q_tokens.intersection(toks))
                if ov > best_score:
                    best_score = ov
                    best_line = ln_n
            if best_score > 0 and best_line:
                snippets.append({
                    "doc_id": doc_id,
                    "page": None,
                    "quote": best_line[:180],
                    "location": "raw_text_match",
                    "full_text": best_line
                })
                if len(snippets) >= k:
                    break

    return snippets


def _validate_quotes_in_result(trainer_result: Dict[str, Any], retrieved_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    אימות בסיסי: כל quote שמופיע ב-evidence/citations חייב להיות substring של אחד ה-snippets.full_text או של quote עצמו.
    אם לא — נוריד severity / נסמן invalid_quote=true.
    """
    haystacks = []
    for sn in retrieved_snippets:
        haystacks.append(_normalize_ws(sn.get("full_text", "")))
        haystacks.append(_normalize_ws(sn.get("quote", "")))

    def _quote_ok(q: str) -> bool:
        qn = _normalize_ws(q)
        if not qn:
            return True
        return any(qn in h for h in haystacks if h)

    # diagnostics evidence
    diags = trainer_result.get("diagnostics") or []
    for d in diags:
        evs = d.get("evidence") or []
        for ev in evs:
            qt = ev.get("quote", "")
            if qt and not _quote_ok(qt):
                ev["invalid_quote"] = True
                # downgrade severity a bit
                if d.get("severity") == "high":
                    d["severity"] = "medium"
                elif d.get("severity") == "medium":
                    d["severity"] = "low"

    # assistant citations
    aa = (trainer_result.get("assistant_answer") or {})
    cits = aa.get("citations") or []
    for c in cits:
        qt = c.get("quote", "")
        if qt and not _quote_ok(qt):
            c["invalid_quote"] = True

    # review_plan supporting_quotes
    rp = trainer_result.get("review_plan") or {}
    sq = rp.get("supporting_quotes") or []
    for c in sq:
        qt = c.get("quote", "")
        if qt and not _quote_ok(qt):
            c["invalid_quote"] = True

    return trainer_result


# =========================
# 4) BUILD: course bundle
# =========================

def build_course_bundle(
    *,
    course_id: str,
    knowledge_docs: List[Dict[str, str]],
    style_docs: List[Dict[str, str]],
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 240,
) -> Dict[str, Any]:
    """
    knowledge_docs/style_docs: [{ "name": "...", "text": "..." }]
    """

    doc_registry: List[Dict[str, Any]] = []
    doc_text_by_id: Dict[str, str] = {}
    raw_knowledge: Dict[str, str] = {}
    raw_style: Dict[str, str] = {}

    def _add_docs(docs: List[Dict[str, str]], doc_type: str, prefix: str) -> None:
        for i, d in enumerate(docs, start=1):
            doc_id = f"{prefix}{i}"
            name = d.get("name", f"{doc_type}_{i}")
            text = d.get("text", "") or ""
            doc_registry.append({"doc_id": doc_id, "type": doc_type, "name": name})
            doc_text_by_id[doc_id] = text
            if doc_type == "knowledge":
                raw_knowledge[doc_id] = text
            else:
                raw_style[doc_id] = text

    _add_docs(knowledge_docs, "knowledge", "K")
    _add_docs(style_docs, "style", "S")

    packed_all = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=False)
    packed_style_only = _pack_docs_for_model(doc_registry, doc_text_by_id, only_style=True)
    packed_knowledge_only = _pack_docs_for_model([d for d in doc_registry if d["type"] == "knowledge"], doc_text_by_id, only_style=False)

    today = date.today().isoformat()
    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    # A) topic_index from knowledge
    payload_topics = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_knowledge_only}\n"
    resp_topics = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": TOPIC_INDEX_PROMPT}, {"role": "user", "content": payload_topics}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    topics_obj = _extract_json_object(resp_topics.choices[0].message.content)
    topic_index = topics_obj.get("topic_index") or []

    # B) terminology from knowledge+style
    payload_terms = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_all}\n"
    resp_terms = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": TERMINOLOGY_EXTRACTION_META_PROMPT}, {"role": "user", "content": payload_terms}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    terms_obj = _extract_json_object(resp_terms.choices[0].message.content)
    canonical_terms = terms_obj.get("canonical_terms") or []

    # C) questions bank from style
    payload_questions = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_style_only}\n"
    resp_questions = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": QUESTIONS_BANK_EXTRACTION_PROMPT}, {"role": "user", "content": payload_questions}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    q_obj = _extract_json_object(resp_questions.choices[0].message.content)
    questions = q_obj.get("questions") or []

    # D) solutions bank from style
    payload_solutions = f"COURSE_ID: {course_id}\nGENERATED_AT: {today}\n\n{packed_style_only}\n"
    resp_solutions = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SOLUTIONS_BANK_EXTRACTION_META_PROMPT}, {"role": "user", "content": payload_solutions}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    sol_obj = _extract_json_object(resp_solutions.choices[0].message.content)
    solutions = sol_obj.get("solutions") or []

    # Compose course_profile
    course_profile = {
        "meta": {"course_id": course_id, "generated_at": today, "version": "2.3"},
        "doc_registry": doc_registry,
        "knowledge_brain": {"doctrines": [], "statutes": [], "precedents": [], "topic_map": [], "topic_index": topic_index},
        "style_brain": {
            "structure": {"detected_model": "generic", "templates": []},
            "voice_signature": {"mandatory_phrasing": [], "preferred_terms": [], "avoid_terms": [], "must_write_exactly": []},
            "grading_rubric": {"weights": {}, "penalty_triggers": [], "bonus_triggers": []},
            "style_sources": [],
            "question_bank": {"enabled": True, "questions": questions},
            "solutions_bank": {"enabled": True, "solutions": solutions},
        },
        "terminology": {"canonical_terms": canonical_terms},
        "trainer_state": {"student_model": {"weak_topics": [], "repeat_misses": [], "confidence_by_topic": {}}},
        "raw_docs": {"knowledge": raw_knowledge, "style": raw_style},
    }

    bundle = json.loads(json.dumps(ENGINE_BUNDLE_TEMPLATE))
    root = bundle["adaptive_learning_engine_bundle"]
    root["instances"]["active_course_profile"] = course_profile
    root["instances"]["last_trainer_result"] = None
    return bundle


# =========================
# 5) GRADE
# =========================

def grade_answer(
    bundle: Dict[str, Any],
    *,
    student_answer: str,
    question_text: Optional[str] = None,
    mode: str = "coach",  # coach|examiner
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    if mode not in ("coach", "examiner"):
        raise ValueError("mode must be 'coach' or 'examiner'.")

    course_profile = (bundle.get("adaptive_learning_engine_bundle", {}) or {}).get("instances", {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    query = (question_text or "") + "\n" + (student_answer or "")
    retrieved = _simple_retrieve_snippets(course_profile, query, k=6)

    payload = {
        "mode": mode,
        "question_text": question_text or "",
        "student_answer": student_answer,
        "course_profile": course_profile,
        "retrieved_snippets": retrieved,
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
    trainer_result = _validate_quotes_in_result(trainer_result, retrieved)

    bundle["adaptive_learning_engine_bundle"]["instances"]["last_trainer_result"] = trainer_result
    return trainer_result


def grade_exam_retry(
    bundle: Dict[str, Any],
    *,
    student_answer: str,
    question_text: Optional[str] = None,
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 220,
) -> Dict[str, Any]:
    course_profile = (bundle.get("adaptive_learning_engine_bundle", {}) or {}).get("instances", {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile. Build a course bundle first.")

    solutions_bank = ((course_profile.get("style_brain") or {}).get("solutions_bank") or {})
    if not (solutions_bank.get("solutions") or []):
        raise ValueError("No solutions found. Upload solved exams to STYLE first.")

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    query = (question_text or "") + "\n" + (student_answer or "")
    retrieved = _simple_retrieve_snippets(course_profile, query, k=7)

    payload = {
        "mode": "exam_retry",
        "question_text": question_text or "",
        "student_answer": student_answer,
        "course_profile": course_profile,
        "solutions_bank": solutions_bank,
        "retrieved_snippets": retrieved,
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
    trainer_result = _validate_quotes_in_result(trainer_result, retrieved)

    bundle["adaptive_learning_engine_bundle"]["instances"]["last_trainer_result"] = trainer_result
    return trainer_result


def assistant_answer(
    bundle: Dict[str, Any],
    *,
    question: str,
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    course_profile = (bundle.get("adaptive_learning_engine_bundle", {}) or {}).get("instances", {}).get("active_course_profile")
    if not course_profile:
        raise ValueError("Bundle has no active_course_profile.")

    client = _get_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    retrieved = _simple_retrieve_snippets(course_profile, question, k=6)

    payload = {"question": question, "retrieved_snippets": retrieved}

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ASSISTANT_QA_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    obj = _extract_json_object(resp.choices[0].message.content)
    # נארוז לתוך trainer_result schema-like
    trainer_result = {"mode": "assistant", "assistant_answer": obj.get("assistant_answer", {})}
    trainer_result = _validate_quotes_in_result(trainer_result, retrieved)
    bundle["adaptive_learning_engine_bundle"]["instances"]["last_trainer_result"] = trainer_result
    return trainer_result
