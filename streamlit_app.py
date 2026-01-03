import streamlit as st
import json
import os

st.set_page_config(
    page_title="Adaptive Learning Engine",
    layout="wide"
)

st.title("⚖️ Adaptive Learning Engine")
st.write("האפליקציה חיה. זה ה־UI.")

st.divider()

uploaded = st.file_uploader(
    "העלה קובץ JSON של הקורס",
    type=["json"]
)

if uploaded:
    data = json.load(uploaded)
    st.success("קובץ נטען בהצלחה")
    st.json(data, expanded=False)
else:
    st.info("טרם הועלה קובץ")
