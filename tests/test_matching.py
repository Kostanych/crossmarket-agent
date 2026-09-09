"""Матчинг B: разбор вердикта, оба режима, исход «аналога нет».

Живых вызовов LLM здесь нет — модель подтверждения подменяется. Проверяется то, что
ломается молча: разбор вердикта, отказ по товару вне снапшота, пустой результат при
нуле подтверждений и оффлайн-расчёт «аналога нет» по вердиктам прогона.
"""

from __future__ import annotations

import asyncio

import pytest

from crossmarket import matching
from crossmarket.models import Label, Product
from evals import matching as suite


def _product(marketplace: str, product_id: str, title: str = "Корзина") -> Product:
    return Product(marketplace=marketplace, id=product_id, title=title, price_rub=100)


def _candidate(product_id: str, said: str | None, score: float = 0.9) -> matching.Candidate:
    return matching.Candidate(
        product=_product("ozon", product_id),
        score=score,
        verdict=matching.Verdict(said=said, reason="—"),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("тот же товар\nmatch", "match"),
        ("разный цвет\nno_match", "no_match"),
        ("разный размер\n**no_match**", "no_match"),
        ("другая комплектация\nNo-Match", "no_match"),
        ("это не match, а no_match\nno_match", "no_match"),
        ("Не могу решить", None),
        ("", None),
    ],
)
def test_parse_verdict(text: str, expected: str | None) -> None:
    """`match` — подстрока `no_match`, а «не match» — отрицание: поиск по всему тексту
    на обоих записывает модели противоположный ответ."""
    assert matching.parse_verdict(text) == expected


def test_verdict_comes_after_the_justification() -> None:
    """Вердикт читается из последней строки; тот же токен в первой ответом не считается."""
    text = "совершенно разные товары: шляпа против плюшевой куклы\nno_match"

    assert matching.parse_verdict(text) == "no_match"
    assert matching.parse_verdict("match\nсовершенно разные товары") is None


def test_unparsed_verdict_is_not_confirmation() -> None:
    assert matching.Verdict(said=None).matched is False


def test_pair_mode_skips_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """При заданном ozon_id кандидат один — заданная карточка, поиск не зовётся."""
    cards = {("wb", "1"): _product("wb", "1"), ("ozon", "2"): _product("ozon", "2")}
    monkeypatch.setattr(matching.clickhouse, "fetch_products", lambda client, keys: cards)
    monkeypatch.setattr(
        matching, "find_candidates", lambda *args, **kwargs: pytest.fail("в вырожденном режиме поиск не нужен")
    )

    async def fake_confirm(wb: Product, ozon: Product, options: object) -> matching.Verdict:
        return matching.Verdict(said="match", reason="то же изделие")

    monkeypatch.setattr(matching, "confirm", fake_confirm)
    result = asyncio.run(matching.match("1", None, None, ozon_id="2"))

    assert result.mode == "pair"
    assert [c.product.id for c in result.confirmed] == ["2"]


def test_product_outside_snapshot_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Товара вне снапшота для B не существует: отказ, а не поход по ссылке."""
    monkeypatch.setattr(matching.clickhouse, "fetch_products", lambda client, keys: {("wb", "1"): _product("wb", "1")})
    monkeypatch.setattr(matching, "confirm", lambda *args: pytest.fail("модель не должна зваться"))

    result = asyncio.run(matching.match("1", None, None, ozon_id="404"))

    assert result.confirmed == []
    assert "404" in (result.missing or "")
    assert "В базе нет" in matching.format_result(result)


def test_no_analog_when_nothing_confirmed() -> None:
    """Лучший из плохих не возвращается: подтверждений ноль — результат пуст."""
    result = matching.MatchResult(mode="full", candidates=[_candidate("2", "no_match"), _candidate("3", None)])

    assert result.confirmed == []
    assert "Аналога на Ozon не нашлось" in matching.format_result(result)


def test_full_mode_confirms_every_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Раннего останова на первом подтверждении нет — зовутся все кандидаты."""
    monkeypatch.setattr(matching.clickhouse, "fetch_products", lambda client, keys: {("wb", "1"): _product("wb", "1")})
    monkeypatch.setattr(
        matching,
        "find_candidates",
        lambda *args, **kwargs: [matching.Candidate(product=_product("ozon", str(n)), score=0.9) for n in (2, 3, 4)],
    )
    asked = []

    async def fake_confirm(wb: Product, ozon: Product, options: object) -> matching.Verdict:
        asked.append(ozon.id)
        return matching.Verdict(said="match" if ozon.id == "2" else "no_match")

    monkeypatch.setattr(matching, "confirm", fake_confirm)
    result = asyncio.run(matching.match("1", None, None))

    assert asked == ["2", "3", "4"]
    assert [c.product.id for c in result.confirmed] == ["2"]


def _full_row(partner_said: str, other_said: str) -> suite.FullRow:
    pair = Label(wb_id="1", ozon_id="2", label="match", split="test")
    result = matching.MatchResult(
        mode="full",
        wb=_product("wb", "1"),
        candidates=[_candidate("2", partner_said, 0.95), _candidate("3", other_said, 0.90)],
    )
    return suite.FullRow(pair=pair, result=result)


def test_no_analog_read_ignores_partner() -> None:
    """«Аналога нет» считается по кандидатам без размеченного партнёра: подтверждён
    только он — B вернул бы пусто, подтверждён кто-то ещё — нет."""
    assert _full_row("match", "no_match").no_analog_ok is True
    assert _full_row("match", "match").no_analog_ok is False


def test_full_metrics_count_partner_separately() -> None:
    rows = [_full_row("match", "no_match"), _full_row("no_match", "match")]
    metrics = suite.full_metrics(rows)

    assert metrics["found_top1"] == 0.5
    assert metrics["no_analog_ok"] == 0.5
    assert metrics["false_confirm_rate"] == 0.5


def test_sweep_restores_threshold_offline() -> None:
    """Отсечка выкидывает кандидата до модели, поэтому по сохранённым вердиктам она
    восстанавливается точно."""
    rows = [_full_row("match", "match")]
    table = suite.sweep(rows, thresholds=(0.0, 0.92))

    assert table.rows[0][-1] == 0.0  # 0.90 прошёл, лишнее подтверждение осталось
    assert table.rows[1][-1] == 1.0  # 0.90 отсечён, остался только партнёр


def test_pair_metrics_on_known_matrix() -> None:
    pairs = [
        (Label(wb_id="1", ozon_id="2", label="match"), "match"),
        (Label(wb_id="3", ozon_id="4", label="match"), "no_match"),
        (Label(wb_id="5", ozon_id="6", label="no_match", negative_kind="hard"), "no_match"),
        (Label(wb_id="7", ozon_id="8", label="no_match", negative_kind="hard"), "match"),
    ]
    rows = [suite.PairRow(pair=pair, verdict=matching.Verdict(said=said)) for pair, said in pairs]
    metrics = suite.pair_metrics(rows)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["negatives_correct"] == 0.5
    assert metrics["accuracy"] == 0.5
