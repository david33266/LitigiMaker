import os
import json
from datetime import datetime
import streamlit as st

from engine_backend import build_course_bundle, grade_answer, grade_exam_retry, assistant_answer

# =========================
# Paths
# =========================
DATA_DIR = "data"
KNOW_DIR = os.path.join(DATA_DIR, "knowledge")
STYLE_DIR = os.path.join(DATA_DIR, "style")
BUNDLE_DIR = os.path.join(DATA_DIR, "bundles")
BUNDLE_PATH = os.path.join(BUNDLE_DIR, "course_bundle.json")

def ensure_dirs():
    for p in [DATA_DIR, KNOW_DIR, STYLE_DIR, BUNDLE_DIR]:
        os.makedirs(p, exist_ok=True)

def list_files(folder: str):
    ensure_dirs()
    files = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            files.append(name)
    return files

def save_uploaded_files(uploaded, folder: str):
    ensure_dirs()
    saved = 0
    for uf in uploaded:
        name = uf.name
        full = os.path.join(folder, name)
        # שמירה בינארית כדי שיתמוך גם ב-txt עם קידוד לא מושלם
        with open(full, "wb") as f:
            f.write(uf.getbuffer())
        saved += 1
    return saved

def delete_file(folder: str, filename: str):
    full = os.path.join(folder, filename)
    if os.path.exists(full) and os.path.isfile(full):
        os.remove(full)
        return True
    return False

def read_text_file(path: str) -> str:
    # ננסה utf-8, ואם נשבר — fallback
    raw = open(path, "rb").read()
    for enc in ["utf-8", "utf-8-sig", "cp1255", "iso-8859-8"]:
        try:
            return raw.decode(enc)
        except Exception:
            pass
    # fallback: נוריד תווים בעייתיים
    return raw.decode("utf-8", errors="ignore")

def load_docs(folder: str):
    docs = []
    for fn in list_files(folder):
        p = os.path.join(folder, fn)
        txt = read_text_file(p)
        docs.append({"name": fn, "text": txt})
    return docs

def save_bundle(bundle: dict):
    ensure_dirs()
    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# UI
# =========================
st.set_page_config(page_title="LitigiMaker", layout="wide", page_icon="⚖️")
ensure_dirs()

st.title("⚖️ LitigiMaker")
st.caption("מנוע לימוד ובדיקה לקורסים משפטיים — מהמחברת + מבחנים/פתרונות. (גרסת MVP)")

with st.sidebar:
    st.subheader("הגדרות")
    api_key = st.text_input("DashScope API Key", type="password", help="מפתח DASHSCOPE_API_KEY")
    model = st.text_input("Model", value="qwen-plus-latest")
    course_id = st.text_input("Course ID", value="course_001")

    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key

    st.divider()
    st.subheader("ניהול קבצים")

    # Uploaders
    st.markdown("**מחברת/חומר קורס (Knowledge)**")
    up_k = st.file_uploader("העלה קבצי TXT למחברת", type=["txt"], accept_multiple_files=True, key="up_k")
    if st.button("שמור קבצי Knowledge"):
        if not up_k:
            st.warning("לא העלית קבצים.")
        else:
            n = save_uploaded_files(up_k, KNOW_DIR)
            st.success(f"נשמרו {n} קבצים ל-knowledge.")

    st.markdown("**מבחנים/פתרונות (Style)**")
    up_s = st.file_uploader("העלה קבצי TXT למבחנים/פתרונות", type=["txt"], accept_multiple_files=True, key="up_s")
    if st.button("שמור קבצי Style"):
        if not up_s:
            st.warning("לא העלית קבצים.")
        else:
            n = save_uploaded_files(up_s, STYLE_DIR)
            st.success(f"נשמרו {n} קבצים ל-style.")

    st.divider()
    st.markdown("**מחיקה**")
    k_files = list_files(KNOW_DIR)
    s_files = list_files(STYLE_DIR)

    del_k = st.selectbox("מחק קובץ Knowledge", [""] + k_files, index=0)
    if st.button("מחק Knowledge"):
        if del_k:
            ok = delete_file(KNOW_DIR, del_k)
            st.success("נמחק.") if ok else st.error("לא הצלחתי למחוק.")
        else:
            st.info("בחר קובץ למחיקה.")

    del_s = st.selectbox("מחק קובץ Style", [""] + s_files, index=0)
    if st.button("מחק Style"):
        if del_s:
            ok = delete_file(STYLE_DIR, del_s)
            st.success("נמחק.") if ok else st.error("לא הצלחתי למחוק.")
        else:
            st.info("בחר קובץ למחיקה.")

    st.divider()
    st.subheader("בניית המוח")
    if st.button("Build / Update Course Bundle", type="primary"):
        if not api_key:
            st.error("חסר API Key.")
        else:
            knowledge_docs = load_docs(KNOW_DIR)
            style_docs = load_docs(STYLE_DIR)
            if not knowledge_docs:
                st.error("חסר חומר במחברת (Knowledge).")
            elif not style_docs:
                st.error("חסרים מבחנים/פתרונות (Style).")
            else:
                with st.spinner("בונה מודל לקורס (topic_index + terms + questions + solutions)..."):
                    bundle = build_course_bundle(
                        course_id=course_id,
                        knowledge_docs=knowledge_docs,
                        style_docs=style_docs,
                        model=model,
                        api_key=api_key,
                    )
                save_bundle(bundle)
                st.success("ה-bundle נבנה ונשמר.")

bundle = load_bundle()

if not bundle:
    st.info("אין עדיין bundle. העלה קבצי TXT (Knowledge + Style) ואז לחץ Build.")
    st.stop()

profile = bundle["adaptive_learning_engine_bundle"]["instances"]["active_course_profile"]
qb = (((profile.get("style_brain") or {}).get("question_bank") or {}).get("questions") or [])
sb = (((profile.get("style_brain") or {}).get("solutions_bank") or {}).get("solutions") or [])
topics = ((profile.get("knowledge_brain") or {}).get("topic_index") or [])

st.success(f"קורס נטען: {profile.get('meta', {}).get('course_id')} | נבנה: {profile.get('meta', {}).get('generated_at')}")
colA, colB, colC = st.columns(3)
colA.metric("נושאים במחברת", len(topics))
colB.metric("שאלות שנחלצו", len(qb))
colC.metric("פתרונות שנחלצו", len(sb))

tab1, tab2, tab3, tab4 = st.tabs(["🎓 עוזר הוראה", "🧪 LitigiMaker (תרגול)", "🧾 Exam Retry (מבחן לחזרה)", "🧠 Debug (המוח)"])

with tab1:
    st.subheader("🎓 עוזר הוראה — שאל את המחברת")
    q = st.text_input("שאלה על החומר:")
    if st.button("ענה לי", key="ask_btn"):
        if not api_key:
            st.error("חסר API Key.")
        elif not q.strip():
            st.warning("כתוב שאלה.")
        else:
            with st.spinner("מחפש תשובה במחברת..."):
                res = assistant_answer(bundle, question=q, model=model, api_key=api_key)
            aa = res.get("assistant_answer", {}) or {}
            st.markdown("### תשובה")
            st.write(aa.get("answer", ""))

            cits = aa.get("citations") or []
            if cits:
                st.markdown("### ציטוטים (רפרנס)")
                for c in cits:
                    flag = " ⚠️" if c.get("invalid_quote") else ""
                    st.caption(f"{c.get('location','')} — [{c.get('doc_id','')}] {flag}")
                    st.markdown(f"> {c.get('quote','')}")
            else:
                st.info("לא נמצאו ציטוטים חזקים מספיק לשאלה הזו.")

with tab2:
    st.subheader("🧪 LitigiMaker — תרגול")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**בחירת שאלה מהמבחנים (אם נחלצה):**")
        opts = ["(בחירה ידנית)"] + [f"{x.get('question_id','Q?')} — {x.get('label','')}" for x in qb]
        choice = st.selectbox("שאלה", opts, index=0)

        if choice != "(בחירה ידנית)":
            idx = opts.index(choice) - 1
            chosen_q = qb[idx]
            question_text = chosen_q.get("question_text", "")
            st.text_area("טקסט שאלה", value=question_text, height=180)
        else:
            question_text = st.text_area("טקסט שאלה (ידני)", height=180)

        student_answer = st.text_area("התשובה שלך", height=260)
        mode = st.selectbox("מצב בדיקה", ["coach", "examiner"], index=0)

        if st.button("בדוק אותי", type="primary"):
            if not api_key:
                st.error("חסר API Key.")
            elif not student_answer.strip():
                st.warning("כתוב תשובה.")
            else:
                with st.spinner("בודק ומחשב ציון..."):
                    res = grade_answer(
                        bundle,
                        student_answer=student_answer,
                        question_text=question_text,
                        mode=mode,
                        model=model,
                        api_key=api_key,
                    )
                st.session_state["last_grade"] = res

    with right:
        res = st.session_state.get("last_grade")
        if res:
            score = (res.get("score") or {}).get("total", 0)
            st.markdown(f"## ציון: **{score}** / 100")

            st.markdown("### 🔍 הערות")
            diags = res.get("diagnostics") or []
            if not diags:
                st.success("לא נמצאו כשלים ברורים.")
            else:
                for d in diags[:12]:
                    title = f"{d.get('error_type','issue')} — {d.get('severity','')} ({d.get('category','')})"
                    with st.expander(title):
                        st.write(d.get("why_wrong",""))
                        fix = (d.get("fix") or {}).get("rewrite_suggestion","")
                        if fix:
                            st.info(f"תיקון מוצע: {fix}")
                        evs = d.get("evidence") or []
                        if evs:
                            ev = evs[0]
                            flag = " ⚠️" if ev.get("invalid_quote") else ""
                            st.caption(f"רפרנס: {ev.get('location','')} [{ev.get('doc_id','')}] {flag}")
                            st.markdown(f"> {ev.get('quote','')}")

            sp = res.get("sharpening_paragraph") or {}
            if any(sp.get(k) for k in ["title","explanation","memory_hook"]):
                st.markdown("### 🧠 נקודת חידוד")
                if sp.get("title"):
                    st.write(f"**{sp['title']}**")
                if sp.get("explanation"):
                    st.write(sp["explanation"])
                if sp.get("memory_hook"):
                    st.caption(f"זיכרון עזר: {sp['memory_hook']}")
                if sp.get("one_check_question"):
                    st.caption(f"שאלת בדיקה: {sp['one_check_question']}")

            rp = res.get("review_plan") or {}
            if rp.get("recommended_topics") or rp.get("supporting_quotes"):
                st.markdown("### 📌 על מה לחזור")
                if rp.get("recommended_topics"):
                    st.write("**נושאים:** " + ", ".join(rp["recommended_topics"]))
                if rp.get("search_hints"):
                    st.write("**מילות חיפוש:** " + ", ".join(rp["search_hints"]))
                for c in (rp.get("supporting_quotes") or [])[:3]:
                    flag = " ⚠️" if c.get("invalid_quote") else ""
                    st.caption(f"{c.get('location','')} [{c.get('doc_id','')}] {flag}")
                    st.markdown(f"> {c.get('quote','')}")

with tab3:
    st.subheader("🧾 Exam Retry — מבחן לחזרה מול פתרונות")
    st.caption("המערכת תמצא פתרון מופת רלוונטי, תחשב ציון, ותיתן 'coverage' + נקודות חסרות/עודפות.")

    q_text = st.text_area("שאלה (הדבק שאלה מהמבחן או תיאור קצר)", height=120)
    ans = st.text_area("תשובה שלך", height=260)

    if st.button("בדוק מול פתרון מופת", type="primary"):
        if not api_key:
            st.error("חסר API Key.")
        elif not ans.strip():
            st.warning("כתוב תשובה.")
        else:
            with st.spinner("משווה מול פתרון מופת ומחשב ציון..."):
                res = grade_exam_retry(bundle, question_text=q_text, student_answer=ans, model=model, api_key=api_key)
            st.session_state["last_retry"] = res

    res = st.session_state.get("last_retry")
    if res:
        score = (res.get("score") or {}).get("total", 0)
        comp = res.get("comparison_to_solution") or {}
        st.markdown(f"## ציון: **{score}** / 100")
        if comp:
            st.markdown(f"### כיסוי מול פתרון מופת: **{comp.get('coverage_score', 0)}%**")
            st.caption(f"Solution matched: {comp.get('solution_id','')}")

            if comp.get("missing_points"):
                st.markdown("**נקודות חסרות:**")
                for p in comp["missing_points"][:12]:
                    st.write("• " + p)

            if comp.get("extra_points"):
                st.markdown("**נקודות טובות שהוספת:**")
                for p in comp["extra_points"][:12]:
                    st.write("• " + p)

            if comp.get("style_gap_notes"):
                st.markdown("**פערי ניסוח/סגנון:**")
                for p in comp["style_gap_notes"][:12]:
                    st.write("• " + p)

        rp = res.get("review_plan") or {}
        if rp:
            st.markdown("### 📌 על מה לחזור")
            if rp.get("recommended_topics"):
                st.write("**נושאים:** " + ", ".join(rp["recommended_topics"]))
            if rp.get("search_hints"):
                st.write("**מילות חיפוש:** " + ", ".join(rp["search_hints"]))
            for c in (rp.get("supporting_quotes") or [])[:3]:
                flag = " ⚠️" if c.get("invalid_quote") else ""
                st.caption(f"{c.get('location','')} [{c.get('doc_id','')}] {flag}")
                st.markdown(f"> {c.get('quote','')}")

with tab4:
    st.subheader("🧠 Debug — מה המערכת למדה")
    with st.expander("Topic Index (נושאים במחברת)"):
        st.write(topics[:50])

    with st.expander("Question Bank (שאלות שנחלצו מה-STYLE)"):
        st.write(qb[:50])

    with st.expander("Solutions Bank (פתרונות שנחלצו מה-STYLE)"):
        st.write(sb[:30])

    with st.expander("Terminology (מונחים)"):
        st.write(((profile.get("terminology") or {}).get("canonical_terms") or [])[:100])
