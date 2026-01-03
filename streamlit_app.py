import os
import json
import gzip
import streamlit as st

from engine_backend import load_json, grade_answer, grade_exam_retry


BUNDLE_PATH = "course_bundle.json"

st.set_page_config(page_title="Adaptive Learning Engine", layout="wide", page_icon="🧠")

st.title("🧠 Adaptive Learning Engine (Hebrew)")
st.caption("קורס-אגנוסטי: טוען Bundle מוכן ומתקן תשובות עם משוב, ציון והשוואה לפתרונות מופת.")

# Health check (כדי שלא יהיה “עמוד ריק” בלי להבין)
st.write("✅ streamlit_app.py נטען")

with st.sidebar:
    st.subheader("הגדרות")
    api_key = st.text_input("DASHSCOPE_API_KEY", type="password")
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key

    st.divider()
    st.write("קובץ Bundle:")
    st.code(BUNDLE_PATH)

@st.cache_resource
def load_bundle():
    return load_json(BUNDLE_PATH)

bundle = None
try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"לא הצלחתי לטעון {BUNDLE_PATH}: {e}")

if not bundle:
    st.info("כדי לייצר Bundle:ץ: שים קבצי TXT בתיקיות knowledge/ ו-style/ ואז הרץ ingest_course_onefile.py")
    st.stop()

profile = bundle["adaptive_learning_engine_bundle"]["instances"]["active_course_profile"]
meta = profile.get("meta", {}) or {}
st.success(f"קורס נטען: {meta.get('course_id', 'unknown')} | נוצר בתאריך: {meta.get('generated_at', 'unknown')}")

tabs = st.tabs(["✍️ Coach", "🧪 Exam Retry", "🧩 Debug (פרופיל)"])

with tabs[0]:
    st.subheader("✍️ מצב Coach")
    col1, col2 = st.columns([1, 1])

    with col1:
        q = st.text_input("שאלה / נושא (לא חובה)", key="coach_q")
        a = st.text_area("התשובה שלך", height=280, key="coach_a")
        mode = st.selectbox("מצב בדיקה", ["coach", "examiner"], index=0)
        run = st.button("בדוק", type="primary", use_container_width=True)

    with col2:
        if run:
            if not os.getenv("DASHSCOPE_API_KEY"):
                st.error("חסר DASHSCOPE_API_KEY (שים בסיידבר).")
            elif not a.strip():
                st.error("חסרה תשובה.")
            else:
                with st.spinner("בודקת..."):
                    try:
                        res = grade_answer(bundle, student_answer=a, question_text=q, mode=mode)
                        st.session_state.coach_res = res
                    except Exception as e:
                        st.error(f"שגיאה בזמן בדיקה: {e}")

        res = st.session_state.get("coach_res")
        if res:
            score = (res.get("score", {}) or {}).get("total", 0)
            st.metric("ציון", score)

            st.markdown("### 🛠️ אבחונים (Diagnostics)")
            for i, d in enumerate(res.get("diagnostics", []) or [], start=1):
                title = f"{i}. {d.get('error_type')} | {d.get('severity')} | {d.get('category')}"
                with st.expander(title):
                    st.write("**מה הבעיה:**")
                    st.write(d.get("why_wrong"))
                    st.write("**מה נכון / פתרון:**")
                    st.info((d.get("fix", {}) or {}).get("rewrite_suggestion", ""))
                    ev = d.get("evidence") or []
                    if ev:
                        st.caption(f"📌 מקור: [{ev[0].get('doc_id')}] עמוד {ev[0].get('page')}: {ev[0].get('quote')}")

            sp = res.get("sharpening_paragraph") or {}
            if sp.get("explanation"):
                st.markdown("### 🎯 פסקת חידוד")
                st.write(f"**{sp.get('title','נקודת חידוד')}**")
                st.write(sp.get("explanation"))
                if sp.get("memory_hook"):
                    st.caption(f"Hook: {sp.get('memory_hook')}")
                if sp.get("one_check_question"):
                    st.warning(f"שאלת בדיקה: {sp.get('one_check_question')}")

            improved = (res.get("improved_answer") or {}).get("full_text")
            if improved:
                st.markdown("### ✅ תשובה משודרגת")
                st.text_area("גרסה משופרת", improved, height=280)

with tabs[1]:
    st.subheader("🧪 Exam Retry (השוואה לפתרון מופת)")
    q2 = st.text_input("שאלה / רמז לזיהוי", key="retry_q")
    a2 = st.text_area("התשובה שלך", height=260, key="retry_a")
    run2 = st.button("השווה לפתרון מופת", type="primary", use_container_width=True)

    if run2:
        if not os.getenv("DASHSCOPE_API_KEY"):
            st.error("חסר DASHSCOPE_API_KEY (שים בסיידבר).")
        elif not a2.strip():
            st.error("חסרה תשובה.")
        else:
            with st.spinner("משווה..."):
                try:
                    res2 = grade_exam_retry(bundle, student_answer=a2, question_text=q2)
                    st.session_state.retry_res = res2
                except Exception as e:
                    st.error(f"שגיאה בזמן השוואה: {e}")

    res2 = st.session_state.get("retry_res")
    if res2:
        score = (res2.get("score", {}) or {}).get("total", 0)
        comp = res2.get("comparison_to_solution") or {}

        st.metric("ציון", score)
        st.write(f"**Matched Solution:** {comp.get('solution_id','(לא נמצא)')}")
        st.write(f"**Coverage:** {comp.get('coverage_score','?')}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### חסר מול פתרון")
            for p in comp.get("missing_points") or []:
                st.write(f"• {p}")
        with c2:
            st.markdown("### תוספות טובות")
            for p in comp.get("extra_points") or []:
                st.write(f"• {p}")

        if comp.get("style_gap_notes"):
            st.markdown("### פערי סגנון")
            for n in comp.get("style_gap_notes") or []:
                st.write(f"• {n}")

        st.markdown("### אבחונים")
        for i, d in enumerate(res2.get("diagnostics", []) or [], start=1):
            title = f"{i}. {d.get('error_type')} | {d.get('severity')} | {d.get('category')}"
            with st.expander(title):
                st.write(d.get("why_wrong"))
                st.info((d.get("fix", {}) or {}).get("rewrite_suggestion", ""))
                ev = d.get("evidence") or []
                if ev:
                    st.caption(f"📌 מקור: [{ev[0].get('doc_id')}] עמוד {ev[0].get('page')}: {ev[0].get('quote')}")

with tabs[2]:
    st.subheader("🧩 Debug: Course Profile")
    st.json({
        "meta": profile.get("meta"),
        "docs": profile.get("doc_registry", [])[:8],
        "terms_count": len((profile.get("terminology", {}) or {}).get("canonical_terms", []) or []),
        "solutions_count": len((((profile.get("style_brain", {}) or {}).get("solutions_bank", {}) or {}).get("solutions", []) or [])),
        "has_raw_texts": bool((profile.get("raw_materials", {}) or {}).get("doc_text_by_id")),
    }, expanded=True)

    with st.expander("Show first 3 terms"):
        terms = (profile.get("terminology", {}) or {}).get("canonical_terms", []) or []
        st.json(terms[:3])

    with st.expander("Show first 2 solutions labels"):
        sols = (((profile.get("style_brain", {}) or {}).get("solutions_bank", {}) or {}).get("solutions", []) or []
        st.json([{"solution_id": s.get("solution_id"), "label": s.get("label")} for s in sols[:2]])
