import os
import streamlit as st

from engine_backend import (
    ensure_dirs,
    KNOWLEDGE_DIR,
    STYLE_DIR,
    list_files,
    delete_file,
    safe_filename,
    build_bundle,
    load_bundle,
    answer_question,
    grade_answer,
)

# ----------------------------
# Page config
# ----------------------------

st.set_page_config(page_title="LitigiMaker", layout="wide", page_icon="⚖️")
ensure_dirs()

# ----------------------------
# Helpers
# ----------------------------

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n = n / 1024
    return f"{n:.0f}TB"


def save_uploaded_files(uploaded, target_dir: str) -> int:
    saved = 0
    if not uploaded:
        return 0
    for f in uploaded:
        name = safe_filename(f.name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in [".txt", ".pdf"]:
            st.warning(f"מדלגת על {name}: רק TXT / PDF נתמכים כרגע.")
            continue
        path = os.path.join(target_dir, name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved += 1
    return saved


def render_file_table(title: str, folder: str, key_prefix: str):
    files = list_files(folder)
    st.subheader(title)
    if not files:
        st.info("אין קבצים עדיין.")
        return

    for f in files:
        cols = st.columns([6, 2, 2, 1])
        cols[0].write(f"📄 {f['name']}")
        cols[1].write(f"{human_size(f['size'])}")
        cols[2].write(f"{f['ext']}")
        if cols[3].button("🗑️", key=f"{key_prefix}_{f['name']}"):
            ok = delete_file(f["path"])
            if ok:
                st.success(f"נמחק: {f['name']}")
                st.rerun()
            else:
                st.error("מחיקה נכשלה.")


# ----------------------------
# Sidebar: Settings
# ----------------------------

st.sidebar.title("הגדרות")

# ✅ real key source: Streamlit secrets / env
has_key = bool(os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""))
st.sidebar.write("OPENAI_API_KEY ב־Secrets:", "✅" if ("OPENAI_API_KEY" in st.secrets) else "❌")

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
    index=0,
)

course_id = st.sidebar.text_input("Course ID", value="course_001")

st.sidebar.markdown("---")
st.sidebar.header("ניהול קבצים")

# Uploaders
up_k = st.sidebar.file_uploader(
    "מחברת/חומר (Knowledge) – העלה TXT/PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="up_knowledge",
)
up_s = st.sidebar.file_uploader(
    "מבחנים/פתרונות (Style) – העלה TXT/PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="up_style",
)

col_save1, col_save2 = st.sidebar.columns(2)
if col_save1.button("שמור Knowledge", use_container_width=True):
    n = save_uploaded_files(up_k, KNOWLEDGE_DIR)
    st.sidebar.success(f"נשמרו {n} קבצים.")
if col_save2.button("שמור Style", use_container_width=True):
    n = save_uploaded_files(up_s, STYLE_DIR)
    st.sidebar.success(f"נשמרו {n} קבצים.")

st.sidebar.markdown("---")
# Build / Load bundle
if st.sidebar.button("🧠 Build bundle", type="primary", use_container_width=True):
    if "OPENAI_API_KEY" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.error("חסר OPENAI_API_KEY ב־Secrets.")
    else:
        with st.sidebar:
            with st.spinner("בונה Bundle..."):
                b = build_bundle(course_id=course_id, model=model)
                st.sidebar.success("נבנה בהצלחה.")
                st.session_state["bundle"] = b
# Load existing bundle if present
if "bundle" not in st.session_state:
    existing = load_bundle(course_id)
    if existing:
        st.session_state["bundle"] = existing

bundle = st.session_state.get("bundle")

# ----------------------------
# Main UI
# ----------------------------

st.title("⚖️ LitigiMaker")
st.caption("מנוע לימוד ובדיקה לקורסים משפטיים — מחברת + מבחנים/פתרונות. (MVP)")

# Top status
if not bundle:
    st.info("אין bundle עדיין. העלה TXT/PDF (Knowledge + Style) ואז לחץ Build bundle.")
else:
    meta = bundle.get("meta", {})
    counts = meta.get("counts", {})
    st.success(
        f"Bundle נטען ✅ | course_id={meta.get('course_id')} | chunks={counts.get('chunks', 0)} | נוצר: {meta.get('generated_at')}"
    )

tabs = st.tabs(["📁 קבצים", "🧑‍🏫 עוזר הוראה", "✍️ בדיקת תשובה", "🧠 Debug / Brain"])

with tabs[0]:
    colA, colB = st.columns(2)
    with colA:
        render_file_table("Knowledge (מחברת/חומר)", KNOWLEDGE_DIR, "del_k")
    with colB:
        render_file_table("Style (מבחנים/פתרונות)", STYLE_DIR, "del_s")

with tabs[1]:
    st.subheader("🧑‍🏫 עוזר הוראה")
    if not bundle:
        st.warning("צריך לבנות bundle קודם.")
    else:
        q = st.text_area("שאלה על החומר", height=120, placeholder="למשל: מה ההבדל בין יסוד עובדתי ליסוד נפשי בעבירה X?")
        if st.button("ענה לי", type="primary"):
            if not q.strip():
                st.error("כתוב שאלה.")
            else:
                with st.spinner("חושבת..."):
                    res = answer_question(bundle=bundle, question=q, model=model)
                st.markdown("### תשובה")
                st.write(res.get("answer", ""))

                st.markdown("### נושא לחיפוש במחברת")
                st.info(res.get("topic", "—"))

                st.markdown("### מקורות (ציטוטים)")
                for c in res.get("citations", [])[:5]:
                    st.write(f"- `{c.get('chunk_id')}` — {c.get('why_relevant')}")

with tabs[2]:
    st.subheader("✍️ בדיקת תשובה (ציון + משוב)")
    if not bundle:
        st.warning("צריך לבנות bundle קודם.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            question_text = st.text_input("נושא/שאלה", placeholder="למשל: דיני ניירות ערך — חובת גילוי בתשקיף")
            student_answer = st.text_area("התשובה שלך", height=280)

            if st.button("בדוק אותי", type="primary", use_container_width=True):
                if not question_text.strip():
                    st.error("חסר נושא/שאלה.")
                elif not student_answer.strip():
                    st.error("חסרה תשובה.")
                else:
                    with st.spinner("מנתחת ומחשבת ציון..."):
                        res = grade_answer(
                            bundle=bundle,
                            question_text=question_text,
                            student_answer=student_answer,
                            model=model,
                        )
                    st.session_state["last_grade"] = res

        with col2:
            res = st.session_state.get("last_grade")
            if res:
                total = (res.get("score") or {}).get("total", 0)
                st.metric("ציון", f"{total}")

                st.markdown("### מה לשפר")
                for d in res.get("diagnostics", [])[:8]:
                    with st.expander(f"{d.get('error_type','בעיה')} ({d.get('severity','')})"):
                        st.write(d.get("why_wrong", ""))
                        st.info("תיקון מוצע:")
                        st.write((d.get("fix") or {}).get("rewrite_suggestion", ""))
                        ev = d.get("evidence") or []
                        if ev:
                            st.caption(f"מקור: {ev[0].get('chunk_id')}")
                            st.code(ev[0].get("quote", ""), language="text")

                st.markdown("### תשובה מומלצת (מבוססת על המחברת)")
                st.write(res.get("model_answer", ""))

                st.markdown("### נושא לחזרה")
                st.warning(res.get("review_topic", "—"))

with tabs[3]:
    st.subheader("🧠 Debug / Brain")
    if not bundle:
        st.warning("צריך לבנות bundle קודם.")
    else:
        idx = bundle.get("index", {})
        st.markdown("### Topics (אינדקס נושאים)")
        st.write(idx.get("topics", []))

        st.markdown("### Glossary")
        st.write(idx.get("glossary", []))

        st.markdown("### Heuristics")
        st.write(idx.get("heuristics", []))

        st.markdown("### Meta")
        st.json(bundle.get("meta", {}))
