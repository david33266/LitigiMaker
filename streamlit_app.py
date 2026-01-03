from engine_mock import grade_answer

if st.button("בדוק תשובה", type="primary", use_container_width=True):
    if not answer.strip():
        st.warning("לא הוזנה תשובה")
    else:
        result = grade_answer(
            question=question,
            answer=answer,
            mode=mode
        )

        st.success(f"ציון: {result['score']}")
        st.write(result["feedback"])

        st.subheader("אבחנות")
        for d in result["diagnostics"]:
            st.write(f"• {d}")
