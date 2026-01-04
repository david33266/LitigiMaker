def grade_answer(
    question: str,
    answer: str,
    mode: str,
    knowledge_docs: list,
    style_docs: list
):
    context_size = sum(len(t) for t in knowledge_docs + style_docs)

    score = min(100, 40 + len(answer) // 10)

    return {
        "score": score,
        "feedback": (
            f"המערכת קיבלה {len(knowledge_docs)} קבצי ידע ו־"
            f"{len(style_docs)} קבצי סגנון "
            f"(סה״כ {context_size} תווים)."
        ),
        "diagnostics": [
            "ה-engine מחובר לקבצים",
            "הקבצים זמינים ל-API",
            "ניתוח עומק יתווסף בשלב הבא"
        ]
    }
