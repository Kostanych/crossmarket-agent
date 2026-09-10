"""Судьи: разбор вердикта, неразобранное и baseline в согласии, согласие по сплитам,
таблица расхождений, перенос ручных меток и проверка набора ответов. Живых вызовов LLM
нет.
"""

from __future__ import annotations

import pytest

from evals.answers import AnswerCase, merge, validate
from evals.judge import Verdict, agreement, parse_label
from evals.judges import JudgeCase, disagreements_table, split_agreements


def test_verdict_is_read_from_the_last_line() -> None:
    assert parse_label("Товары разные: цвет отличается\nbad") == "bad"
    assert parse_label("Вердикт верен\ngood") == "good"


def test_word_inside_the_reason_is_not_the_verdict() -> None:
    assert parse_label("Обоснование выглядит good, но вердикт неверен\nbad") == "bad"


def test_unparsed_answer_is_empty_not_good() -> None:
    assert parse_label("не могу решить") == ""
    assert parse_label("") == ""

    result = agreement([Verdict("q01", "")], {"q01": "good"}, positive="bad", split="test")
    assert result.unparsed == 1
    assert result.accuracy == 0.0


def test_baseline_is_the_share_of_the_majority_class() -> None:
    truth = {"q01": "good", "q02": "good", "q03": "good", "q04": "bad"}
    verdicts = [Verdict(case_id, "good") for case_id in truth]

    result = agreement(verdicts, truth, positive="bad", split="test")

    assert result.baseline == pytest.approx(0.75)
    assert result.accuracy == pytest.approx(0.75)
    assert result.recall == 0.0


def _judge_case(case_id: str, split: str, truth: str) -> JudgeCase:
    return JudgeCase(id=case_id, split=split, stratum="match", truth=truth, prompt="", note=case_id)


def test_agreement_is_counted_per_split() -> None:
    cases = [_judge_case("a", "calibration", "good"), _judge_case("b", "test", "bad")]
    verdicts = [Verdict("a", "good"), Verdict("b", "good")]

    result = split_agreements(cases, verdicts)

    assert result["calibration"].accuracy == 1.0
    assert result["test"].accuracy == 0.0
    assert not result["calibration"].reportable and result["test"].reportable


def test_disagreements_table_shows_the_reason() -> None:
    cases = [_judge_case("a", "test", "bad")]
    table = disagreements_table(cases, [Verdict("a", "good", "карточки совпадают")])

    assert table.rows == [["a", "test", "good", "bad", "карточки совпадают"]]


def _answer(case_id: str, verdict: str = "") -> AnswerCase:
    return AnswerCase(id=case_id, case_id=case_id, model="opus", stratum="found", split="test", verdict=verdict)


def test_resampling_keeps_manual_labels() -> None:
    merged = merge([_answer("q01"), _answer("q02")], [_answer("q01", "bad")])

    assert [case.verdict for case in merged] == ["bad", ""]


def test_unlabelled_answer_is_a_problem() -> None:
    problems = validate([_answer("q01"), _answer("q02", "good")])

    assert len(problems) == 1 and "q01" in problems[0]
