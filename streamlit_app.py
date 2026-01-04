import streamlit as st
from engine_mock import grade_answer

st.set_page_config(
    page_title="Adaptive Learning Engine",
    layout="wide"
)

st.title("⚖️ Adaptive Learning Engine")
st.caption("אבטיפוס עם העלאת קבצים")

# -------- Sidebar --------
with st.sidebar:
    st.header("הגדרות")

    mode = st.selectbox(
        "מצב עבודה",
        ["אימון (Coach)", "בודק (Examiner)", "מבחן לחזרה"]
    )

    st.divider()

    st.subheader("📂 העלאת חומרי לימוד")

    knowledge_files = st.file_uploader(
        "מחברות / סיכומים",
        type=["txt"],
        accept_multiple_files=True
    )

    style_files = st.file_uploader(
        "מבחנים פתורים / פתרונות",
        type=["txt"],
        accept_multiple_files=True
    )

# -------- Main --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 שאלה / נושא")
    question = st.text_area(
        "הכנס שאלה או נושא:",
        height=150
    )

with col2:
    st.subheader("✍️ תשובת הסטודנט")
    answer = st.text_area(
        "כתוב את התשובה שלך:",
        height=150
    )

st.divider()

# -------- Button --------
if st.button("בדוק תשובה", type="primary", use_container_width=True):

    if not answer.strip():
        st.warning("לא הוזנה תשובה")
    else:
        # קריאת תוכן הקבצים
        knowledge_texts = [
            f.read().decode("utf-8") for f in knowledge_files
        ] if knowledge_files else []

        style_texts = [
            f.read().decode("utf-8") for f in style_files
        ] if style_files else []

        result = grade_answer(
            question=question,
            answer=answer,
            mode=mode,
            knowledge_docs=knowledge_texts,
            style_docs=style_texts
        )

        st.success(f"ציון: {result['score']}")
        st.write(result["feedback"])

        st.subheader("אבחנות")
        for d in result["diagnostics"]:
            st.write(f"• {d}")
