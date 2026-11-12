"""Product leaf questions must not be treated as bare declarations / general defs."""

from __future__ import annotations

from work_rag_orchestrator.nodes.classify import _heuristic_classify
from work_rag_orchestrator.nodes.infer_specification import is_definition_question


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
