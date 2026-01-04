import os
import json
from typing import List, Dict

import streamlit as st

from engine_backend import build_course_bundle, grade_answer, grade_exam_retry


# =========================
# SETTINGS
# =========================

st.set_page_config(page_title="Adaptive Learning Engine", page_icon="🧠", layout="wide")

DATA_DIR = "data"
KNOW_DIR = os.path.join(DATA_DIR, "knowledge")
STYLE_DIR = os.path.join(DATA_DIR, "style")
BUNDLE_PATH = os.path.join(DATA_DIR, "course_bundle.json")

os.makedirs(KNOW_DIR, exist_ok=True)
os.makedirs(STYLE_DIR, exist_ok=True)


# =========================
# HELPERS: FILE OPS
# =========================

def list_files(folder: str) -> List[str]:
    try:
        return sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    except FileNotFoundError:
        return []

def save_uploaded_files(uploaded, folder: str) -> int:
    count = 0
    for f in uploaded:
        content = f.getvalue()
        path = os.path.join(folder, f.name)
        with open(path, "wb") as out:
            out.write(content)
        count += 1
    return count

def delete_file(folder: str, filename: str) -> None:
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        os.remove(path)

def read_text_files(folder: str) -> List[Dict[str, str]]:
    docs = []
    for name in list_files(folder):
        path = os.path.join(folder, name)
        # Only text files (simplest)
        if not name.lower().endswith((".txt", ".md")):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"name": name, "text": f.read()})
    return docs

def save_bundle(bundle: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

def load_bundle() -> Dict | None:
    if not os.path.exists(BUNDLE_PATH):
        return None
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# SIDEBAR: API + DOCS
# =========================

st.sidebar.title("⚙️ הגדרות")

api_key = st.sidebar.text_input("DASHSCOPE_API_KEY", type="password")
if api_key:
    os.environ["DASHSCOPE_API_KEY"] = api_key

course_id = st.sidebar.text_input("Course ID", value="course_001")

st.sidebar.markdown("---")
st.sidebar.subheader("📁 העלאת קבצים")

col_up1, col_up2 = st.sidebar.columns(2)

with col_up1:
    up_knowledge = st.file_uploader("Knowledge (TXT/MD)", type=["txt", "md"], accept_multiple_files=True, key="upk")
    if st.button("שמור Knowledge", use_container_width=True):
        if not up_knowledge:
            st.sidebar.warning("לא הועלו קבצים.")
        else:
            n = save_uploaded_files(up_knowledge, KNOW_DIR)
            st.sidebar.success(f"נשמרו {n} קבצים ל-knowledge.")

with col_up2:
    up_style = st.file_uploader("Style (TXT/MD)", type=["txt", "md"], accept_multiple_files=True, key="ups")
    if st.button("שמור Style", use_container_width=True):
        if not up_style:
            st.sidebar.warning("לא הועלו קבצים.")
        else:
            n = save_uploaded_files(up_style, STYLE_DIR)
            st.sidebar.success(f"נשמרו {n} קבצים ל-style.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 ניהול קבצים (מחיקה)")

k_files = list_files(KNOW_DIR)
s_files = list_files(STYLE_DIR)

if k_files:
    k_del = st.sidebar.selectbox("מחק מקובצי Knowledge", ["—"] + k_files)
    if st.sidebar.button("מחק Knowledge", use_container_width=True):
        if k_del != "—":
            delete_file(KNOW_DIR, k_del)
            st.sidebar.success(f"נמחק: {k_del}")
            st.rerun()

if s_files:
    s_del = st.sidebar.selectbox("מחק מקובצי Style", ["—"] + s_files)
    if st.sidebar.button("מחק Style", use_container_width=True):
        if s_del != "—":
            delete_file(STYLE_DIR, s_del)
            st.sidebar.success(f"נמחק: {s_del}")
            st.rerun()


# =========================
# MAIN UI
# =========================

st.title("🧠 Adaptive Learning Engine (Legal)")
st.caption("העלאת TXT → בניית Bundle → בדיקת תשובות עם ציון וריפרנס לפי *נושא* (בלי עמודים).")

tab_build, tab_train, tab_retry, tab_debug = st.tabs([
    "1) Build Bundle",
    "2) Trainer (Coach/Examiner)",
    "3) Exam Retry (Compare)",
    "4) Debug"
])


# ---------- TAB 1: BUILD ----------
with tab_build:
    st.subheader("Build Course Bundle")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**קבצים קיימים:**")
        st.write("Knowledge:", [f for f in list_files(KNOW_DIR) if f.lower().endswith((".txt", ".md"))] or "—")
        st.write("Style:", [f for f in list_files(STYLE_DIR) if f.lower().endswith((".txt", ".md"))] or "—")

    with c2:
        if st.button("🚀 Build / Rebuild Bundle", type="primary", use_container_width=True):
            if not api_key:
                st.error("חסר DASHSCOPE_API_KEY בסרגל הצד.")
            else:
                knowledge_docs = read_text_files(KNOW_DIR)
                style_docs = read_text_files(STYLE_DIR)

                if not knowledge_docs:
                    st.error("חסרים קבצי Knowledge (TXT/MD).")
                else:
                    with st.spinner("בונה Bundle (Course Profile + Terms + Solutions)…"):
                        try:
                            bundle = build_course_bundle(
                                course_id=course_id,
                                knowledge_docs=knowledge_docs,
                                style_docs=style_docs,
                                api_key=api_key,
                            )
                            save_bundle(bundle)
                            st.success("✅ Bundle נבנה ונשמר בהצלחה.")
                        except Exception as e:
                            st.error(f"שגיאה בבניית Bundle: {e}")

    st.info("טיפ: אם אין לך Style עדיין — אפשר לבנות Bundle רק מ-Knowledge, אבל Exam Retry לא יעבוד עד שתעלה פתרונות מופת.")


# ---------- LOAD BUNDLE ONCE ----------
bundle = load_bundle()
bundle_ready = bool(bundle and bundle.get("adaptive_learning_engine_bundle", {}).get("instances", {}).get("active_course_profile"))


# ---------- TAB 2: TRAIN ----------
with tab_train:
    st.subheader("Trainer (Coach / Examiner)")
    if not bundle_ready:
        st.warning("אין Bundle טעון. עבור לטאב Build ובנה Bundle.")
    else:
        mode = st.radio("Mode", ["coach", "examiner"], horizontal=True)

        q = st.text_input("שאלה / נושא (אופציונלי)")
        ans = st.text_area("התשובה שלך", height=260)

        if st.button("בדוק תשובה", type="primary"):
            if not api_key:
                st.error("חסר DASHSCOPE_API_KEY בסרגל הצד.")
            elif not ans.strip():
                st.error("הדבק תשובה.")
            else:
                with st.spinner("מנתח…"):
                    try:
                        res = grade_answer(
                            bundle,
                            question_text=q or None,
                            student_answer=ans,
                            mode=mode,
                            api_key=api_key,
                        )
                        st.session_state["last_res"] = res
                        save_bundle(bundle)  # persist last result
                    except Exception as e:
                        st.error(f"שגיאה: {e}")

        res = st.session_state.get("last_res")
        if res:
            _render_result(res, allow_improved=(mode == "coach"))


# ---------- TAB 3: EXAM RETRY ----------
with tab_retry:
    st.subheader("Exam Retry (השוואה לפתרונות מופת)")
    if not bundle_ready:
        st.warning("אין Bundle טעון. עבור לטאב Build ובנה Bundle.")
    else:
        q = st.text_input("שאלה / מזהה שאלה (ככל שתתאר יותר טוב, ההתאמה לפתרון תשתפר)", key="retry_q")
        ans = st.text_area("התשובה שלך", height=260, key="retry_a")

        if st.button("בדוק מול פתרון מופת", type="primary"):
            if not api_key:
                st.error("חסר DASHSCOPE_API_KEY בסרגל הצד.")
            elif not ans.strip():
                st.error("הדבק תשובה.")
            else:
                with st.spinner("משווה לפתרונות מופת ומחשב ציון…"):
                    try:
                        res = grade_exam_retry(
                            bundle,
                            question_text=q or None,
                            student_answer=ans,
                            api_key=api_key,
                        )
                        st.session_state["last_retry_res"] = res
                        save_bundle(bundle)
                    except Exception as e:
                        st.error(f"שגיאה: {e}")

        res = st.session_state.get("last_retry_res")
        if res:
            _render_result(res, allow_improved=False)


# ---------- TAB 4: DEBUG ----------
with tab_debug:
    st.subheader("Debug Bundle")
    if not bundle:
        st.info("אין Bundle על הדיסק עדיין.")
    else:
        st.code(json.dumps(bundle.get("adaptive_learning_engine_bundle", {}).get("instances", {}).get("active_course_profile", {}), ensure_ascii=False, indent=2))


# =========================
# UI RENDERER
# =========================

def _score_color(score: int) -> str:
    if score >= 85:
        return "green"
    if score >= 65:
        return "orange"
    return "red"

def _render_result(res: Dict, allow_improved: bool):
    score = int((res.get("score", {}) or {}).get("total", 0) or 0)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"<div style='border:2px solid {_score_color(score)}; border-radius:12px; padding:14px; text-align:center;'>"
            f"<div style='font-size:14px; opacity:0.8;'>ציון</div>"
            f"<div style='font-size:44px; font-weight:800; color:{_score_color(score)};'>{score}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        bd = (res.get("score", {}) or {}).get("breakdown", {}) or {}
        st.write("**Breakdown**")
        for k, v in bd.items():
            st.write(f"- {k}: {v}")

    with col2:
        sp = res.get("sharpening_paragraph", {}) or {}
        if sp.get("title") or sp.get("explanation"):
            st.markdown("### ✨ פסקת חידוד")
            if sp.get("title"):
                st.write(f"**{sp['title']}**")
            if sp.get("explanation"):
                st.write(sp["explanation"])
            if sp.get("memory_hook"):
                st.info(sp["memory_hook"])
            if sp.get("one_check_question"):
                st.write("✅ שאלת בדיקה:", sp["one_check_question"])

        comp = res.get("comparison_to_solution", {}) or {}
        if comp.get("solution_id") or comp.get("coverage_score") is not None:
            st.markdown("### 🧩 השוואה לפתרון מופת")
            st.write("Solution:", comp.get("solution_id", "—"))
            if comp.get("coverage_score") is not None:
                st.write("Coverage:", comp.get("coverage_score"))
            if comp.get("missing_points"):
                st.write("**Missing points:**")
                for p in comp["missing_points"]:
                    st.write("- ", p)
            if comp.get("extra_points"):
                st.write("**Extra points:**")
                for p in comp["extra_points"]:
                    st.write("- ", p)
            if comp.get("style_gap_notes"):
                st.write("**Style gaps:**")
                for p in comp["style_gap_notes"]:
                    st.write("- ", p)

    st.markdown("---")
    st.markdown("### 🛠️ Diagnostics (מה טעות ומה הפתרון)")

    diags = res.get("diagnostics", []) or []
    if not diags:
        st.success("לא נמצאו בעיות משמעותיות.")
    else:
        for i, d in enumerate(diags, start=1):
            title = f"{i}. {d.get('category','?')} • {d.get('error_type','?')} • {d.get('severity','?')}"
            with st.expander(title, expanded=False):
                st.write("**הבעיה בתשובה:**")
                st.code(d.get("symptom_in_answer", ""), language="text")

                st.write("**למה זה בעייתי:**")
                st.write(d.get("why_wrong", ""))

                st.write("**מה נכון / מה חסר:**")
                st.write(d.get("correct_rule_or_term", ""))

                fix = d.get("fix", {}) or {}
                if fix.get("rewrite_suggestion"):
                    st.write("**איך לכתוב במקום:**")
                    st.info(fix["rewrite_suggestion"])
                if fix.get("micro_steps"):
                    st.write("**מיקרו־צעדים:**")
                    for s in fix["micro_steps"]:
                        st.write("- ", s)

                ev = (d.get("evidence", []) or [])
                if ev:
                    st.write("**ריפרנס לחומר (בלי עמודים):**")
                    for e in ev[:2]:
                        ref = (e.get("reference", {}) or {})
                        topic = ref.get("topic_label")
                        hint = ref.get("find_hint")
                        doc_name = e.get("doc_name") or e.get("doc_id")
                        st.caption(f"מסמך: {doc_name}")
                        if topic:
                            st.write("📌 **נושא:**", topic)
                        if hint:
                            st.write("🔎 **מילות חיפוש:**", hint)
                        if e.get("quote"):
                            st.write("🧾 **ציטוט:**")
                            st.markdown(f"> {e['quote']}")

    if allow_improved:
        improved = res.get("improved_answer", {}) or {}
        if improved.get("full_text"):
            st.markdown("---")
            st.markdown("### ✅ תשובה משודרגת (Coach)")
            st.text_area("Improved Answer", improved["full_text"], height=320)
