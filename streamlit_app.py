import streamlit as st

st.set_page_config(
    page_title="Adaptive Learning Engine",
    layout="wide"
)

st.title("⚖️ Adaptive Learning Engine")
st.caption("מערכת למידה אדפטיבית – שלב אבטיפוס")

# Sidebar
with st.sidebar:
    st.header("הגדרות")
    mode = st.selectbox(
        "מצב עבודה",
        ["אימון (Coach)", "בודק (Examiner)", "מבחן לחזרה"]
    )
    st.divider()
    st.info("המנוע עדיין לא מחובר")

# Main layout
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

if st.button("בדוק תשובה", type="primary", use_container_width=True):
    if not answer.strip():
        st.warning("לא הוזנה תשובה")
    else:
        st.success("UI עובד. כאן יחובר המנוע.")
