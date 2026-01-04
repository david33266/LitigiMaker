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
    page_title="LitiGiMaker – העלאת חומרים",
    layout="wide",
    page_icon="⚖️"
)

st.title("⚖️ LitiGiMaker – העלאת חומרים לקורס")

# =========================
# פונקציות עזר
# =========================

def save_uploaded_files(uploaded_files, target_dir):
    saved = []
    os.makedirs(target_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(target_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(file.name)

    return saved


def list_saved_files():
    files = {}
    for section, path in {
        "knowledge": KNOWLEDGE_DIR,
        "style": STYLE_DIR
    }.items():
        if not os.path.exists(path):
            files[section] = []
        else:
            files[section] = sorted(os.listdir(path))
    return files


# =========================
# UI – העלאת קבצים
# =========================

st.markdown("## 📥 העלאת קבצים")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📘 Knowledge (חומר לימוד)")
    knowledge_files = st.file_uploader(
        "העלה קבצי TXT למחברת / סיכומים",
        type=["txt"],
        accept_multiple_files=True,
        key="knowledge_uploader"
    )

with col2:
    st.subheader("📝 Style (פתרונות / מבחנים)")
    style_files = st.file_uploader(
        "העלה קבצי TXT של פתרונות ומבחנים",
        type=["txt"],
        accept_multiple_files=True,
        key="style_uploader"
    )

# =========================
# כפתור שמירה
# =========================

if st.button("💾 שמור קבצים לדיסק", type="primary"):
    saved_any = False

    if knowledge_files:
        saved = save_uploaded_files(knowledge_files, KNOWLEDGE_DIR)
        st.success(f"נשמרו {len(saved)} קבצי Knowledge")
        saved_any = True

    if style_files:
        saved = save_uploaded_files(style_files, STYLE_DIR)
        st.success(f"נשמרו {len(saved)} קבצי Style")
        saved_any = True

    if not saved_any:
        st.warning("לא הועלו קבצים לשמירה")

# =========================
# הצגת קבצים שנשמרו
# =========================

st.markdown("## 📂 קבצים שנשמרו בשרת")

files = list_saved_files()

for section, items in files.items():
    st.subheader(section)
    if not items:
        st.caption("אין קבצים")
    else:
        for f in items:
            st.write("•", f)
