import streamlit as st
import os

from engine_mock import grade_answer

# ---------------- Config ----------------
st.set_page_config(
    page_title="Adaptive Learning Engine",
    layout="wide"
)

# ---------------- Helpers ----------------
def save_uploaded_files(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    saved_paths = []
    for f in files:
        path = os.path.join(target_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    return saved_paths

def load_texts_from_dir(dir_path):
    texts = []
    if not os.path.exists(dir_path):
        return texts

    for fname in os.listdir(dir_path):
        if fname.endswith(".txt"):
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as rf:
                texts.append(rf.read())
    return texts

# ---------------- UI ----------------
st.title("⚖️ Adaptive Learning Engine")
st.caption("העלאת קבצים + שמירה + בדיקה")

# -------- Sidebar --------
with st.sidebar:
    st.header("📂 העלאת קבצים ושמירה")

    mode = st.selectbox(
        "בחר מצב בדיקה:",
        ["אימון (Coach)", "בודק (Examiner)", "מבחן לחזרה"]
    )

    st.divider()

    knowledge_files = st.file_uploader(
        "📘 מחברות / סיכומים (TXT)",
        type=["txt"],
        accept_multiple_files=True
    )

    style_files = st.file_uploader(
        "🧾 מבחנים פתורים / פתרונות (TXT)",
        type=["txt"],
        accept_multiple_files=True
    )

    if st.button("💾 שמור קבצים לדיסק"):
        saved_list = []
        if knowledge_files:
            saved_list += save_uploaded_files(knowledge_files, "data/knowledge")
        if style_files:
            saved_list += save_uploaded_files(style_files, "data/style")

        if saved_list:
            st.success(f"נשמרו {len(saved_list)} קבצים")
        else:
            st.warning("לא נבחרו קבצים לשמירה")

# -------- Main --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 שאלה / נושא")
    question = st.text_area("הכנס שאלה או נושא:", height=150)

with col2:
    st.subheader("✍️ תשובת הסטודנט")
    answer = st.text_area("כתוב את התשובה שלך:", height=150)

st.divider()

if st.button("בדוק תשובה", type="primary", use_container_width=True):

    if not answer.strip():
        st.warning("חובה להזין תשובה")
    else:
        # load files from data folders
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

        st.subheader("🛠️ אבחנות")
        for item in result["diagnostics"]:
            st.write(f"• {item}")
