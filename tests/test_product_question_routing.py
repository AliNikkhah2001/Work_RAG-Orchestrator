"""Product leaf questions must not be treated as bare declarations / general defs."""

from __future__ import annotations

from work_rag_orchestrator.nodes.classify import _heuristic_classify
from work_rag_orchestrator.nodes.infer_specification import (
    is_cheque_for_legal_ask,
    is_definition_question,
    is_request_intent,
    _is_short_with_keyword,
)


def test_nardeban_chie_is_question_not_specification():
    assert _heuristic_classify("نردبان اعتباری چیه") == "question"
    assert _heuristic_classify("نردبان اعتباری چیست؟") == "question"


def test_kafil_chie_is_question():
    assert _heuristic_classify("طرح کفیل چیه؟") == "question"
    assert _heuristic_classify("کفیل چیست") == "question"


def test_definition_question_skips_product_leaves():
    assert is_definition_question("اعتبارسنجی چیست") is True
    assert is_definition_question("نردبان اعتباری چیه") is False
    assert is_definition_question("طرح کفیل چیست؟") is False
    assert is_definition_question("چک چیست") is False


def test_corporate_cheque_request_is_question_not_declaration():
    """«گزارش چک حقوقی میخوام» must retrieve DNE FAQ, not «حوزه چک انتخاب شد»."""
    for q in (
        "گزارش چک حقوقی میخوام",
        "گزارش چک شرکتی میخوام",
        "میخوام گزارش چک حقوقی بگیرم",
        "گزارش چک برای شرکت میخوام",
        "corporate cheque report میخوام",
    ):
        assert is_request_intent(q) or is_cheque_for_legal_ask(q), q
        assert _is_short_with_keyword(q) is False, q
        assert _heuristic_classify(q) == "question", q


def test_bare_cheque_still_specification_candidate():
    """Bare leaf «چک» remains a short keyword declaration."""
    assert is_request_intent("چک") is False
    assert is_cheque_for_legal_ask("چک") is False
    assert _is_short_with_keyword("چک") is True


def test_mikham_clitic_is_not_a_facilities_keyword():
    """«وام» ⊂ «میخوام»: the خواستن clitic must not count as a loan keyword.

    Without the guard, every «… میخوام» request was classified as a facilities
    query and the subspace label flipped to facilities_legal, bypassing the
    cheque FAQ that says the report is not defined for legal entities.
    """
    from work_rag_orchestrator.nodes.infer_specification import (
        has_facility_keyword,
        label_subspace_from_query,
    )

    # «وام» only as part of خواستن → not a facilities mention
    for q in ("میخوام", "نمیخوام", "میخواد", "بخوام", "میخواهم"):
        assert has_facility_keyword(q) is False, q
    # Real loan mentions still detected
    for q in ("وام", "وام مسکن", "تسهیلات", "قسط", "سقف وام"):
        assert has_facility_keyword(q) is True, q

    # The corporate cheque request must never label as facilities_*
    for q in (
        "گزارش چک حقوقی میخوام",
        "گزارش چک شرکتی میخوام",
        "گزارش چک برای شرکت میخوام",
    ):
        assert label_subspace_from_query(q) == "cheque", q
