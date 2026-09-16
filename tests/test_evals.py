"""Харнесс: правило разбиения, разбор ответа и подсчёт согласия судьи.

Живых вызовов LLM здесь нет — проверяется то, что ломается молча: сплит, который
поехал после добавления кейсов, выдуманная цена, которую разбор не заметил, и
согласие судьи, посчитанное на калибровке вместо теста.
"""

from __future__ import annotations

import pytest

from crossmarket.agent import SEARCH_TOOL, SQL_TOOL, Answer
from crossmarket.models import Label
from evals.cases import Case, RoutingCase, SqlCase, assign_splits, validate, validate_routing_cases, validate_sql_cases
from evals.grading import grade, parse_tool_output
from evals.judge import Verdict, agreement
from evals.retrieval import first_hit_rank, recall_at_k
from evals.routing import grade as grade_routing

TOOL_OUTPUT = """id: 1000000001
Название: Точилка для ножей
Цена, ₽: 920
Категория: Кухня / Ножи
Ссылка: https://example.com/1
Скор: 0.842
Характеристики: Материал: сталь
Описание: Точит быстро.

Строка описания с пустой строкой выше.

id: 1000000002
Название: Мусат
Цена, ₽: 1450
Категория: Кухня / Ножи
Ссылка: https://example.com/2
Скор: 0.811"""


def _case(**overrides) -> Case:
    fields = {"id": "q01", "question": "чем наточить ножи", "relevant_ids": ["1000000001"], "expected": "found"}
    return Case(**(fields | overrides))


def test_tool_output_survives_blank_lines_in_description() -> None:
    """Описание карточки содержит пустые строки, и резать выдачу по ним нельзя."""
    shown = parse_tool_output([TOOL_OUTPUT])

    assert set(shown) == {"1000000001", "1000000002"}
    assert shown["1000000002"].price_rub == 1450
    assert shown["1000000001"].title == "Точилка для ножей"


def test_cited_card_counted_only_from_final_answer() -> None:
    answer = "Точилка для ножей (id: 1000000001), 920 ₽ — точит быстро."

    result = grade(_case(), answer, [TOOL_OUTPUT])

    assert result.cited_ids == ["1000000001"]
    assert result.cited_recall == 1.0
    assert result.outcome_correct and result.faithful and not result.refused


def test_card_shown_but_not_named_does_not_count() -> None:
    """Ровно то, ради чего метрика ужесточена: карточка была в выдаче, но в ответ не дошла."""
    result = grade(_case(), "Подходящих товаров не нашлось.", [TOOL_OUTPUT])

    assert result.cited_recall == 0.0
    assert result.refused and not result.outcome_correct


def test_invented_price_is_caught() -> None:
    answer = "Точилка (id: 1000000001) стоит 700 ₽."

    result = grade(_case(), answer, [TOOL_OUTPUT])

    assert result.bad_prices == [700]
    assert not result.faithful


def test_price_from_the_question_is_not_an_invention() -> None:
    """«Точилка до 1000 рублей» в ответе — повтор ограничения, а не выдуманная цена."""
    case = _case(question="точилка для ножей до 1000 рублей")

    result = grade(case, "Точилка (id: 1000000001) за 920 ₽ — как раз до 1000 ₽.", [TOOL_OUTPUT])

    assert result.faithful


def test_invented_card_is_caught() -> None:
    result = grade(_case(), "Есть мусат (id: 1000000009) за 920 ₽.", [TOOL_OUTPUT])

    assert result.unknown_ids == ["1000000009"]
    assert not result.faithful


def test_explicit_denial_is_the_correct_outcome_for_absent_case() -> None:
    case = Case(id="a01", question="зимние шины", expected="absent", kind="far")

    denied = grade(case, "Такого товара в базе нет.", [TOOL_OUTPUT])
    substituted = grade(case, "Есть точилка (id: 1000000001) за 920 ₽.", [TOOL_OUTPUT])

    assert denied.outcome_correct
    assert not substituted.outcome_correct


def test_denial_with_an_alternative_still_counts() -> None:
    """Близкий промах: искомого нет, сосед есть. Назвать соседа — верное поведение."""
    case = Case(id="a02", question="спрей против комаров", expected="absent", kind="near")

    result = grade(case, "Спрея в базе нет. Из близкого — мусат (id: 1000000002) за 1450 ₽.", [TOOL_OUTPUT])

    assert result.outcome_correct and result.denied
    assert not result.refused


def test_denial_is_recognised_by_meaning_not_by_wording() -> None:
    """Отрицание стоит то до источника, то после, и на разном расстоянии от него.
    Все формулировки взяты из настоящих прогонов."""
    case = Case(id="a03", question="чем построить дом", expected="absent", kind="far")
    answers = [
        "Дом в этой базе построить не из чего: стройматериалов в снапшоте нет вообще.",
        "В базе снапшота ничего для этого нет: настольных игр для взрослых не находится.",
        "В выдаче нет средств транспорта.",
        "Грипсов на мотоцикл в базе Wildberries не найдено.",
        "В базе Wildberries масляные краски для картин не найдены.",
        "В базе Wildberries специализированного боксёрского оборудования (груш, мешков, лап) не найдено.",
        "По запросу в базе найдены только аксессуары для путешествий, но не сам транспорт.",
    ]

    assert all(grade(case, answer, [TOOL_OUTPUT]).outcome_correct for answer in answers)


def test_clarifying_question_is_not_a_denial() -> None:
    """«Вопрос очень широкий, уточните» — агент не утверждал, что товара нет."""
    case = Case(id="a03", question="чем построить дом", expected="absent", kind="far")

    result = grade(case, "Вопрос очень широкий. Уточните, что именно вы ищете:\n\n- Конструктор\n- Инструменты", [])

    assert not result.denied and not result.outcome_correct


def test_scope_refusal_is_not_a_denial() -> None:
    """«Я отвечаю только на вопросы о товарах»: агент не смотрел в базу и ничего про
    отсутствие товара не утверждал."""
    case = Case(id="a04", question="чем прокопать туннель", expected="absent", kind="far")

    result = grade(case, "Я отвечаю только на вопросы о товарах Wildberries из снапшота базы.", [])

    assert not result.denied and not result.outcome_correct


def test_extra_cards_do_not_spoil_the_verdict() -> None:
    """Агент вправе предложить карточку, которой нет в разметке: это не ошибка ответа."""
    answer = "Точилка (id: 1000000001) и мусат (id: 1000000002)."

    result = grade(_case(), answer, [TOOL_OUTPUT])

    assert result.cited_extra == 1
    assert result.outcome_correct


def test_split_is_deterministic_and_stratified() -> None:
    cases = [_case(id=f"q{n:02d}") for n in range(10)]
    cases += [Case(id=f"a{n:02d}", question="нет в базе", expected="absent", kind="far") for n in range(10)]

    assign_splits(cases)
    again = [
        _case(id=case.id) if case.expected == "found" else Case(id=case.id, question="x", expected="absent", kind="far")
        for case in cases
    ]
    assign_splits(again)

    assert [case.split for case in cases] == [case.split for case in again]
    for stratum in ("found", "absent-far"):
        members = [case for case in cases if case.stratum == stratum]
        assert sum(case.split == "calibration" for case in members) == 6


def test_existing_split_is_never_reassigned() -> None:
    """Иначе кейс, на котором судья доводился, мог бы уехать в отчётный тест-сет."""
    old = [_case(id=f"q{n:02d}", split="test") for n in range(5)]
    fresh = [_case(id=f"q{n:02d}") for n in range(5, 20)]

    changed = assign_splits(old + fresh)

    assert all(case.split == "test" for case in old)
    assert {case.id for case in changed} == {case.id for case in fresh}


def test_validate_catches_silent_breakage() -> None:
    cases = [
        _case(id="q01", split="test"),
        _case(id="q01", split="test"),
        _case(id="q02", relevant_ids=[], split="test"),
        Case(id="a01", question="нет в базе", expected="absent", kind="", split="test"),
        _case(id="q03"),
    ]

    problems = "\n".join(validate(cases))

    assert "duplicate id" in problems
    assert "without relevant_ids" in problems
    assert "kind must be" in problems
    assert "split not assigned" in problems


def test_agreement_counts_and_lists_disagreements() -> None:
    verdicts = [Verdict("q01", "match"), Verdict("q02", "no_match", "цвет другой"), Verdict("q03", "match")]
    truth = {"q01": "match", "q02": "match", "q03": "match"}

    result = agreement(verdicts, truth, positive="match", split="test")

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.precision == 1.0
    assert result.disagreements().rows == [["q02", "no_match", "match", "цвет другой"]]
    assert result.for_readme() == pytest.approx(2 / 3)


def test_calibration_number_is_refused_for_the_readme() -> None:
    result = agreement([Verdict("q01", "match")], {"q01": "match"}, positive="match", split="calibration")

    assert not result.reportable
    with pytest.raises(ValueError, match="only test"):
        result.for_readme()


def test_missing_verdict_is_an_error_not_a_skip() -> None:
    with pytest.raises(ValueError, match="different cases"):
        agreement([Verdict("q01", "match")], {"q01": "match", "q02": "match"}, positive="match", split="test")


def test_recall_counts_share_of_relevant_not_a_hit() -> None:
    found = ["1", "9", "2"]

    assert recall_at_k(found, {"1", "2", "3"}, 5) == pytest.approx(2 / 3)
    assert recall_at_k(found, {"1", "2", "3"}, 1) == pytest.approx(1 / 3)
    assert first_hit_rank(found, {"2"}) == 3
    assert first_hit_rank(found, {"7"}) is None


def test_sql_case_round_trip_drops_defaults() -> None:
    case = SqlCase(id="c01", question="сколько всего?", sql="SELECT count() FROM products FINAL", kind="count")
    assert case.to_dict() == {
        "id": "c01",
        "question": "сколько всего?",
        "sql": "SELECT count() FROM products FINAL",
        "kind": "count",
    }
    ordered = SqlCase(id="c02", question="топ-3", sql="SELECT 1", kind="topn", ordered=True, split="test")
    assert SqlCase.from_dict(ordered.to_dict()) == ordered


def test_sql_validate_catches_silent_breakage() -> None:
    problems = validate_sql_cases(
        [
            SqlCase(id="c01", question="?", sql="SELECT 1", kind="count", split="test"),
            SqlCase(id="c01", question="?", sql="SELECT 1", kind="count", split="test"),
            SqlCase(id="c02", question="?", sql="   ", kind="count", split="test"),
            SqlCase(id="c03", question="?", sql="SELECT 1", kind="", split="test"),
            SqlCase(id="c04", question="?", sql="SELECT 1", kind="count"),
        ]
    )
    assert any("duplicate id" in p for p in problems)
    assert any("no reference SQL" in p for p in problems)
    assert any("no stratum" in p for p in problems)
    assert any("split not assigned" in p for p in problems)


def test_pairs_get_the_same_split_rule_as_questions() -> None:
    pairs = [Label(wb_id=f"{i:03d}", ozon_id=f"9{i:03d}", label="match") for i in range(10)]
    pairs += [Label(wb_id=f"1{i:02d}", ozon_id=f"8{i:03d}", label="no_match", negative_kind="hard") for i in range(10)]
    assign_splits(pairs)

    by_stratum: dict[str, list[str]] = {}
    for pair in pairs:
        by_stratum.setdefault(pair.stratum, []).append(pair.split or "—")
    assert set(by_stratum) == {"match", "no_match-hard"}
    for splits in by_stratum.values():
        assert splits.count("calibration") == 6

    again = [Label(wb_id=p.wb_id, ozon_id=p.ozon_id, label=p.label, negative_kind=p.negative_kind) for p in pairs]
    assign_splits(again)
    assert [p.split for p in again] == [p.split for p in pairs]


def _routing_case(**overrides) -> RoutingCase:
    fields = {"id": "r01", "question": "чем наточить ножи", "tool": "search_wb", "kind": "a", "split": "test"}
    return RoutingCase(**(fields | overrides))


def _answer(*names: str) -> Answer:
    return Answer(tool_calls=[{"name": name, "input": {}} for name in names])


def test_routing_verdict_is_decided_by_the_first_call() -> None:
    """Маршрут — это первое решение; что агент делал дальше, метрику не меняет."""
    result = grade_routing(_routing_case(), _answer(SEARCH_TOOL, SQL_TOOL))

    assert result.routed
    assert result.used_expected
    assert result.used_other


def test_wrong_first_call_is_a_miss_even_if_the_right_one_follows() -> None:
    result = grade_routing(_routing_case(), _answer(SQL_TOOL, SEARCH_TOOL))

    assert not result.routed
    assert result.used_expected


def test_skill_call_is_not_part_of_the_route() -> None:
    """`Skill` — служебный тул: попади он в последовательность, развилка A/C поехала бы."""
    result = grade_routing(_routing_case(tool="execute_sql"), _answer("Skill", SQL_TOOL))

    assert result.routed
    assert result.sequence == ["execute_sql"]
    assert result.skill_used


def test_answer_without_tools_is_a_miss() -> None:
    result = grade_routing(_routing_case(), _answer())

    assert not result.routed
    assert result.no_tool


def test_multihop_case_stays_out_of_accuracy() -> None:
    """У цепочки правильных вызовов два, и порядок между ними вопросом не задан."""
    case = _routing_case(id="r37", tool="", kind="multihop")
    result = grade_routing(case, _answer(SEARCH_TOOL, SQL_TOOL))

    assert not case.single_hop
    assert not result.routed
    assert result.sequence == ["search_wb", "execute_sql"]


def test_routing_validate_catches_silent_breakage() -> None:
    problems = validate_routing_cases(
        [
            _routing_case(id="r01"),
            _routing_case(id="r01"),
            _routing_case(id="r02", tool="search_ozon"),
            _routing_case(id="r03", split=None),
            _routing_case(id="r04", kind="multihop", tool="search_wb"),
        ]
    )

    assert any("duplicate id" in problem for problem in problems)
    assert any("r02" in problem and "label" in problem for problem in problems)
    assert any("r03" in problem and "split" in problem for problem in problems)
    assert any("r04" in problem and "multihop" in problem for problem in problems)
