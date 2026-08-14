"""Тесты чистых частей загрузчиков: сборка строки, точки и текста карточки.

Живых Qdrant и ClickHouse тут нет — клиенты импортируются внутри функций,
поэтому модули грузятся без баз и без torch. Проверяется то, что ломается
молча: порядок колонок, детерминированность идентификатора точки и то, что
карточка без цены не уедет в снапшот.
"""

from datetime import datetime

import pytest

from crossmarket.embedding import PASSAGE_PREFIX, QUERY_PREFIX, product_text
from crossmarket.models import Product
from crossmarket.storage import clickhouse, qdrant


def _product(**overrides) -> Product:
    fields = {
        "marketplace": "wb",
        "id": "1000000001",
        "url": "https://www.wildberries.ru/catalog/1000000001/detail.aspx",
        "title": "Лупа 10×",
        "description": "Оптическое стекло",
        "price_rub": 496,
        "category": "Канцтовары / Офисные принадлежности",
        "attributes": {"Диаметр": "60 мм"},
        "review_count": 12,
    }
    return Product(**{**fields, **overrides})


def test_product_text_is_title_alone_without_attributes() -> None:
    assert product_text(_product(attributes={}), "title+attrs") == "Лупа 10×"


def test_product_text_labels_each_part() -> None:
    assert product_text(_product(), "title+attrs") == "Лупа 10×\nХарактеристики: Диаметр: 60 мм"


def test_composition_controls_what_gets_indexed() -> None:
    """Состав текста подбирается замером, поэтому он параметр, а не константа."""
    short = product_text(_product(), "title+attrs")
    full = product_text(_product(), "title+attrs+cat+desc")

    assert "Канцтовары" not in short
    assert "Категория: Канцтовары / Офисные принадлежности" in full
    assert "Описание: Оптическое стекло" in full


def test_e5_prefixes_differ() -> None:
    """Документы и запросы у e5 кодируются разными префиксами."""
    assert QUERY_PREFIX != PASSAGE_PREFIX


def test_row_follows_column_order() -> None:
    row = clickhouse.to_row(_product())

    assert len(row) == len(clickhouse.COLUMNS)
    assert row[clickhouse.COLUMNS.index("price_rub")] == 496
    assert row[clickhouse.COLUMNS.index("id")] == "1000000001"
    assert row[clickhouse.COLUMNS.index("characteristics")] == {"Диаметр": "60 мм"}
    assert isinstance(row[clickhouse.COLUMNS.index("collected_at")], datetime)


def test_card_without_price_is_refused() -> None:
    with pytest.raises(ValueError, match="нет цены"):
        clickhouse.to_row(_product(price_rub=None))


def test_labels_never_reach_the_table() -> None:
    """Ground truth в таблице, которую читает агент, обнулил бы метрики B."""
    assert not {"label", "negative_kind", "wb_id", "ozon_id"} & set(clickhouse.COLUMNS)


def test_point_id_is_stable_and_marketplace_scoped() -> None:
    assert qdrant.point_id("wb", "123") == qdrant.point_id("wb", "123")
    assert qdrant.point_id("wb", "123") != qdrant.point_id("ozon", "123")


def test_payload_carries_only_filter_fields() -> None:
    """Цена нужна для фильтра в A, остальное берётся из ClickHouse по id."""
    payload = qdrant.payload_of(_product())

    assert payload["id"] == "1000000001"
    assert payload["price_rub"] == 496
    assert set(payload) == {"marketplace", "id", "category", "price_rub"}
