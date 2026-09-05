"""Ограничители тула C и сравнение результата с эталоном.

Живых вызовов LLM нет, ClickHouse не нужен: `run_sql` получает поддельный клиент,
а сравнение работает с готовыми строками.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from crossmarket.sql import check_sql, format_result, run_sql
from evals.sql import answer_mentions, compare, normalise_rows


class FakeAnswer:
    def __init__(self, columns: list[str], rows: list[tuple]) -> None:
        self.column_names = columns
        self.result_rows = rows


class FakeClient:
    """Запоминает, дошёл ли запрос до драйвера: половина проверок — про то, что не дошёл."""

    def __init__(self, columns: list[str] | None = None, rows: list[tuple] | None = None, boom: str = "") -> None:
        self.columns = ["n"] if columns is None else columns
        self.rows = [(1,)] if rows is None else rows
        self.boom = boom
        self.seen: list[str] = []

    def query(self, sql: str, settings: dict | None = None) -> FakeAnswer:
        self.seen.append(sql)
        if self.boom:
            raise RuntimeError(self.boom)
        return FakeAnswer(self.columns, self.rows)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO products VALUES (1)",
        "DROP TABLE products",
        "TRUNCATE TABLE products",
        "ALTER TABLE products DELETE WHERE 1",
        "SELECT 1; DROP TABLE products",
        "   ",
    ],
)
def test_write_and_multi_statement_never_reach_the_driver(sql: str) -> None:
    client = FakeClient()
    result = run_sql(client, sql)
    assert not result.ok
    assert client.seen == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count() FROM products FINAL",
        "select 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT count() FROM products FINAL;",
        "-- считаем строки\nSELECT count() FROM products FINAL",
    ],
)
def test_read_queries_pass(sql: str) -> None:
    assert check_sql(sql) == ""
    client = FakeClient()
    assert run_sql(client, sql).ok


def test_trailing_semicolon_stripped_before_driver() -> None:
    client = FakeClient()
    run_sql(client, "SELECT 1;")
    assert client.seen == ["SELECT 1"]


def test_clickhouse_error_becomes_text_not_exception() -> None:
    result = run_sql(FakeClient(boom="Unknown expression identifier `nosuch`"), "SELECT nosuch FROM products")
    assert not result.ok
    assert "nosuch" in format_result(result)
    assert format_result(result).startswith("Ошибка запроса")


def test_empty_result_is_named_not_silent() -> None:
    result = run_sql(FakeClient(columns=["title"], rows=[]), "SELECT title FROM products FINAL WHERE 0")
    assert result.ok
    assert format_result(result) == "Запрос выполнен, строк нет."


def test_long_result_is_truncated_with_a_note() -> None:
    rows = [(i,) for i in range(10)]
    result = run_sql(FakeClient(columns=["n"], rows=rows), "SELECT n FROM t", shown_rows=3)
    assert result.truncated and result.total_rows == 10
    assert "Строк: 10, показаны первые 3" in format_result(result)


def test_rounding_and_row_order() -> None:
    assert normalise_rows([(1.005,)], ordered=True) == [(1.0,)]
    assert compare([(2,), (1,)], [(1,), (2,)], ordered=False) == (True, True)
    assert compare([(2,), (1,)], [(1,), (2,)], ordered=True)[0] is False


def test_decimal_and_int_compare_equal() -> None:
    assert compare([(Decimal("783.95"),)], [(783.95,)], ordered=False)[0]
    assert compare([(5,)], [(5.0,)], ordered=False)[0]


def test_extra_and_permuted_columns_count_as_match_but_not_exact() -> None:
    reference = [("Косметичка", 105), ("Кружка", 117)]
    permuted = [(105, "Косметичка"), (117, "Кружка")]
    assert compare(permuted, reference, ordered=True) == (True, False)

    with_extra = [("wb", "Косметичка", 105), ("wb", "Кружка", 117)]
    assert compare(with_extra, reference, ordered=True) == (True, False)
    assert compare(reference, reference, ordered=True) == (True, True)


def test_missing_column_is_not_a_match() -> None:
    assert compare([("Косметичка",)], [("Косметичка", 105)], ordered=True) == (False, False)


def test_wrong_value_is_not_a_match() -> None:
    assert compare([("Косметичка", 106)], [("Косметичка", 105)], ordered=True) == (False, False)


def test_both_empty_matches_and_one_sided_does_not() -> None:
    assert compare([], [], ordered=False) == (True, True)
    assert compare([(1,)], [], ordered=False) == (False, False)
    assert compare([], [(1,)], ordered=False) == (False, False)


def test_answer_must_name_the_reference_numbers() -> None:
    assert answer_mentions("Средняя цена — 1456.64 ₽.", [(1456.64,)], ordered=False)
    assert answer_mentions("Средняя цена — 1456,64 ₽.", [(1456.64,)], ordered=False)
    assert answer_mentions("Самый дорогой стоит 93 666 ₽.", [(93666,)], ordered=False)
    assert not answer_mentions("Средняя цена около полутора тысяч.", [(1456.64,)], ordered=False)


def test_empty_reference_requires_an_explicit_denial() -> None:
    assert answer_mentions("Таких товаров в базе нет.", [], ordered=False)
    assert not answer_mentions("Вот что нашлось: чайник за 500 ₽.", [], ordered=False)


def test_commas_inside_titles_survive_number_flattening() -> None:
    answer = "1. Аксессуары для сумки, рюкзака — 117 ₽"
    assert answer_mentions(answer, [("Аксессуары для сумки, рюкзака", 117)], ordered=True)


def test_marketplace_code_matches_its_human_name() -> None:
    answer = "Ozon — 210 карточек, Wildberries — 184."
    assert answer_mentions(answer, [("ozon", 210), ("wb", 184)], ordered=False)
