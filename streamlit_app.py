import streamlit as st
import os

from engine_mock import grade_answer

# ---------------- Config ----------------
st.set_page_config(
    page_title="Adaptive Learning Engine",
    layout="wide"
)

# ---------------- Utils ----------------
def save_uploaded_files(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    paths = []

    for f in files:
        path = os.path.join(target_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)

    return paths


def load_texts_from_dir(dir_path):
    texts = []
    if not os.path.exists(dir_path):
        return texts

    for fname in os.listdir(dir_path):
        if fname.endswith(".txt"):
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


# ---------------- UI ----------------
st.title("⚖️ Adaptive Learning Engine")
st.caption("אבטיפוס מלא – העלאה, שמירה, וחיבור למנוע")

# -------- Sidebar --------
with st.sidebar:
    st.header("הגדרות")

    mode = st.selectbox(
        "מצב עבודה",
        ["אימון (Coach)", "בודק (Examiner)", "מבחן לחזרה"]
    )

    st.divider()
    st.subheader("📂 העלאת קבצים")

    knowledge_files = st.file_uploader(
        "מחברות / סיכומים (TXT)",
        type=["txt"],
        accept_multiple_files=True
    )

    style_files = st.file_uploader(
        "מבחנים פתורים / פתרונות (TXT)",
        type=["txt"],
        accept_multiple_files=True
    )

    if st.button("💾 שמור קבצים"):
        saved = []

        if knowledge_files:
            saved += save_uploaded_files(
                knowledge_files, "data/knowledge"
            )

        if style_files:
            saved += save_uploaded_files(
                style_files, "data/style"
            )

        if saved:
            st.success(f"נשמרו {len(saved)} קבצים")
        else:
            st.warning("לא נבחרו קבצים לשמירה")

# -------- Main --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 שאלה / נושא")
    question = st.text_area(
        "הכנס שאלה או נושא:",
        height=180
    )

with col2:
    st.subheader("✍️ תשובת הסטודנט")
    answer = st.text_area(
        "כתוב את התשובה שלך:",
        height=180
    )

st.divider()

if st.button("בדוק תשובה", type="primary", use_container_width=True):

    if not answer.strip():
        st.warning("לא הוזנה תשובה")
    else:
        knowledge_texts = load_texts_from_dir("data/knowledge")
        style_texts = load_texts_from_dir("data/style")

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
