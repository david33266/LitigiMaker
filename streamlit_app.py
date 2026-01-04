import os
import streamlit as st

# =========================
# הגדרות בסיס
# =========================

BASE_DIR = "data"
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
STYLE_DIR = os.path.join(BASE_DIR, "style")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(STYLE_DIR, exist_ok=True)

st.set_page_config(
    page_title="LitiGiMaker – ניהול חומרים",
    layout="wide",
    page_icon="⚖️"
)

st.title("⚖️ LitiGiMaker – ניהול חומרים לקורס")

# =========================
# פונקציות עזר
# =========================

def save_uploaded_files(uploaded_files, target_dir):
    saved = []
    for file in uploaded_files:
        path = os.path.join(target_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(file.name)
    return saved


def list_files(dir_path):
    if not os.path.exists(dir_path):
        return []
    return sorted(os.listdir(dir_path))


def delete_file(dir_path, filename):
    path = os.path.join(dir_path, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# =========================
# UI – העלאת קבצים
# =========================

st.markdown("## 📥 העלאת קבצים")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Knowledge (חומר לימוד)")
    knowledge_files = st.file_uploader(
        "קבצי TXT למחברות / סיכומים",
        type=["txt"],
        accept_multiple_files=True,
        key="knowledge_uploader"
    )

with col2:
    st.subheader("📝 Style (פתרונות / מבחנים)")
    style_files = st.file_uploader(
        "קבצי TXT של פתרונות ומבחנים",
        type=["txt"],
        accept_multiple_files=True,
        key="style_uploader"
    )

if st.button("💾 שמור קבצים לדיסק", type="primary"):
    if knowledge_files:
        save_uploaded_files(knowledge_files, KNOWLEDGE_DIR)
        st.success("קבצי Knowledge נשמרו")
    if style_files:
        save_uploaded_files(style_files, STYLE_DIR)
        st.success("קבצי Style נשמרו")
    if not knowledge_files and not style_files:
        st.warning("לא נבחרו קבצים")

# =========================
# UI – ניהול / מחיקה
# =========================

st.markdown("## 📂 קבצים שנשמרו בשרת")

def render_file_list(title, dir_path, key_prefix):
    st.subheader(title)
    files = list_files(dir_path)

    if not files:
        st.caption("אין קבצים")
        return

    for fname in files:
        col_name, col_btn = st.columns([6, 1])
        col_name.write(f"📄 {fname}")

        if col_btn.button("❌", key=f"{key_prefix}_{fname}"):
            delete_file(dir_path, fname)
            st.experimental_rerun()


colA, colB = st.columns(2)

with colA:
    render_file_list("Knowledge", KNOWLEDGE_DIR, "del_k")

with colB:
    render_file_list("Style", STYLE_DIR, "del_s")
