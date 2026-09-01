"""Поиск карточек: порог, фильтры и выпадение дистракторов.

Обе базы и эмбеддер подменяются: живых сервисов у теста нет, а проверяется здесь
логика склейки, а не качество поиска — его меряет `python -m evals retrieval`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from crossmarket import retrieval
from crossmarket.models import Product
from crossmarket.storage import clickhouse


class FakeQdrant:
    """Отдаёт заданные точки, запоминая, с чем к нему пришли."""

    def __init__(self, points: list[dict[str, Any]]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []


class FakeClickHouse:
    def __init__(self, products: dict[tuple[str, str], Product]) -> None:
        self.products = products


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch):
    def fake_search(client, marketplace, vector, limit=10, category=None, price_max=None):
        client.calls.append({"limit": limit, "category": category, "price_max": price_max})
        return client.points

    def fake_fetch(client, keys):
        return {key: client.products[key] for key in keys if key in client.products}

    monkeypatch.setattr(retrieval.qdrant, "search", fake_search)
    monkeypatch.setattr(retrieval.clickhouse, "fetch_products", fake_fetch)
    monkeypatch.setattr(retrieval, "encode_queries", lambda texts: [[0.0]])


def _point(product_id: str, score: float) -> dict[str, Any]:
    return {"score": score, "id": product_id, "marketplace": "wb", "synthetic": False}


def _product(product_id: str) -> Product:
    return Product(marketplace="wb", id=product_id, title=f"Товар {product_id}", price_rub=100)


def test_threshold_drops_low_scores(fakes) -> None:
    qdrant_client = FakeQdrant([_point("1", 0.9), _point("2", 0.4)])
    clickhouse_client = FakeClickHouse({("wb", "1"): _product("1"), ("wb", "2"): _product("2")})

    hits = retrieval.search_products("вопрос", qdrant_client, clickhouse_client, min_score=0.5)

    assert [hit.product.id for hit in hits] == ["1"]


def test_everything_below_threshold_gives_empty(fakes) -> None:
    qdrant_client = FakeQdrant([_point("1", 0.2)])
    clickhouse_client = FakeClickHouse({("wb", "1"): _product("1")})

    assert retrieval.search_products("вопрос", qdrant_client, clickhouse_client, min_score=0.5) == []


def test_distractor_falls_out_without_snapshot_row(fakes) -> None:
    """Дистрактор живёт только в Qdrant, карточки в ClickHouse у него нет."""
    qdrant_client = FakeQdrant([_point("synthetic-1", 0.9), _point("1", 0.8)])
    clickhouse_client = FakeClickHouse({("wb", "1"): _product("1")})

    hits = retrieval.search_products("вопрос", qdrant_client, clickhouse_client, min_score=0.5)

    assert [hit.product.id for hit in hits] == ["1"]


def test_filters_reach_qdrant(fakes) -> None:
    qdrant_client = FakeQdrant([])
    retrieval.search_products(
        "вопрос",
        qdrant_client,
        FakeClickHouse({}),
        limit=3,
        category="Посуда",
        price_max=5000,
    )

    assert qdrant_client.calls == [{"limit": 3, "category": "Посуда", "price_max": 5000}]


def test_order_follows_score(fakes) -> None:
    qdrant_client = FakeQdrant([_point("1", 0.9), _point("2", 0.7)])
    clickhouse_client = FakeClickHouse({("wb", "1"): _product("1"), ("wb", "2"): _product("2")})

    hits = retrieval.search_products("вопрос", qdrant_client, clickhouse_client, min_score=0.5)

    assert [hit.score for hit in hits] == [0.9, 0.7]


def test_format_hits_reports_empty_result() -> None:
    assert retrieval.format_hits([]) == "Ничего не найдено."


def test_format_hits_keeps_id_and_price() -> None:
    hit = retrieval.Hit(product=_product("42"), score=0.812)
    text = retrieval.format_hits([hit])

    assert "id: 42" in text
    assert "Цена, ₽: 100" in text
    assert "0.812" in text


def test_clickhouse_row_becomes_product() -> None:
    """Имена полей в таблице и в модели расходятся в двух местах."""
    row = (
        "wb",
        "1",
        "https://example.test/1",
        "Товар",
        "Описание",
        100,
        "Посуда",
        {"Цвет": "белый"},
        7,
        datetime(2026, 8, 13, 19, 55, 39),
    )

    product = clickhouse.to_product(row)

    assert product.attributes == {"Цвет": "белый"}
    assert product.collected_at == "2026-08-13T19:55:39"
    assert product.key == ("wb", "1")
