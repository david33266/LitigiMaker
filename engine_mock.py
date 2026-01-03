def grade_answer(question: str, answer: str, mode: str):
    if not answer.strip():
        return {
            "score": 0,
            "feedback": "לא נכתבה תשובה."
        }

    score = min(100, 40 + len(answer) // 10)

    return {
        "score": score,
        "feedback": f"המערכת זיהתה תשובה באורך {len(answer)} תווים במצב {mode}.",
        "diagnostics": [
            "חסר ניתוח מפורט",
            "אין התייחסות לחריגים",
            "המבנה כללי מדי"
        ]
    }
