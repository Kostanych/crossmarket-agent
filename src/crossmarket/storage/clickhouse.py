"""Снапшот товаров в ClickHouse — источник данных для tool C.

Таблица одна и плоская: закрытый набор структурных полей, по которому tool C
генерирует SQL. Всё, чего в этом списке нет, — не его задача; узкая поверхность
схемы делает эталоны eval вычислимыми и убирает класс запросов по
несуществующим колонкам.

Характеристики лежат в `Map(String, String)`.

Описание есть, хотя по нему C не ищет (семантика из текста — задача tool A):
вырожденный режим B достаёт карточку из снапшота по id, и текстовой модели
подтверждения одного названия мало.

Меток пар здесь нет и быть не может: агент читает эту базу
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from crossmarket.config import (
    CLICKHOUSE_DB,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_TABLE,
    CLICKHOUSE_USER,
)
from crossmarket.models import Product

if TYPE_CHECKING:
    from clickhouse_connect.driver.client import Client

COLUMNS = (
    "marketplace",
    "id",
    "url",
    "title",
    "description",
    "price_rub",
    "category",
    "characteristics",
    "review_count",
    "collected_at",
)

DDL = f"""
CREATE TABLE IF NOT EXISTS {CLICKHOUSE_TABLE} (
    marketplace  LowCardinality(String),
    id           String,
    url          String,
    title        String,
    description  String,
    price_rub    UInt32,
    category     LowCardinality(String),
    characteristics Map(String, String),
    review_count Nullable(UInt32),
    collected_at DateTime
)
ENGINE = ReplacingMergeTree(collected_at)
ORDER BY (marketplace, id)
"""


def connect() -> Client:
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
    )


def create_table(client: Client) -> None:
    """Создать таблицу снапшота, если её ещё нет."""
    client.command(DDL)


def to_row(product: Product) -> list[object]:
    """Товар → строка таблицы в порядке `COLUMNS`.

    Цена обязательна: карточки без неё выбраковываются ещё на сборе датасета,
    потому что дырка в снапшоте молча превратилась бы в неверный агрегат.
    Время снапшота в модели строкой, а драйвер ждёт `datetime`.
    """
    if product.price_rub is None:
        raise ValueError(f"{product.marketplace}/{product.id}: нет цены, такие карточки в снапшот не идут.")
    return [
        product.marketplace,
        product.id,
        product.url,
        product.title,
        product.description,
        product.price_rub,
        product.category,
        product.attributes,
        product.review_count,
        datetime.fromisoformat(product.collected_at),
    ]


def insert_products(client: Client, products: list[Product]) -> int:
    """Залить товары. Повторная заливка того же id заменяет строку, а не двоит.

    За это отвечает ReplacingMergeTree по `collected_at`: до слияния кусков
    строки живут обе, поэтому читатели используют `FINAL`.
    """
    rows = [to_row(product) for product in products]
    client.insert(CLICKHOUSE_TABLE, rows, column_names=list(COLUMNS))
    return len(rows)
