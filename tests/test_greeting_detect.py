from work_rag_orchestrator.nodes.retrieve import detect_greeting_kind


def test_compound_greetings():
    for q in (
        "سلام",
        "سلام روز بخیر",
        "سلام، روزتون بخیر",
        "روز بخیر",
        "سلام وقت بخیر",
        "درود بر شما",
    ):
        assert detect_greeting_kind(q) == "greeting", q


def test_questions_are_not_greetings():
    for q in (
        "امتیاز چک ۷۶",
        "گزارش اعتبارسنجی می‌خواهم",
        "سلام امتیاز من چند است",
    ):
        assert detect_greeting_kind(q) is None, q
