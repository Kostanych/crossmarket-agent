"""Снапшот товаров в ClickHouse — источник данных для tool C.

Таблица одна и плоская: закрытый набор структурных полей, по которому tool C
генерирует SQL. Характеристики — в `Map(String, String)`.

Описание есть, хотя по нему C не ищет: вырожденный режим B достаёт карточку по id, и
текстовой модели подтверждения одного названия мало.

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
    """Клиент без сессии — `autogenerate_session_id=False` снимать нельзя.

    С сессией драйвер отбивает второй запрос по тому же клиенту, пока не ответил
    первый («Attempt to execute concurrent queries within the same session»). Клиент
    здесь один на процесс, а ходят в него параллельно: агент запускает тулы одного
    хода одновременно, tool B подтверждает кандидатов пачкой.
    """
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
        autogenerate_session_id=False,
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


def fetch_products(client: Client, keys: list[tuple[str, str]]) -> dict[tuple[str, str], Product]:
    """Карточки по ключам `(marketplace, id)`, в словаре по тому же ключу.

    Нужно тулам A и B: в payload Qdrant лежат только id, категория и цена, а отвечать
    надо по названию, описанию и характеристикам. `FINAL` обязателен — до слияния
    кусков ReplacingMergeTree держит обе версии строки, и без него на карточку
    придут дубли с разным `collected_at`.

    Ключи подставляются параметрами драйвера, а не форматированием строки.
    """
    if not keys:
        return {}
    conditions = " OR ".join(f"(marketplace = %(m{i})s AND id = %(i{i})s)" for i in range(len(keys)))
    params = {f"m{i}": key[0] for i, key in enumerate(keys)} | {f"i{i}": key[1] for i, key in enumerate(keys)}
    query = f"SELECT {', '.join(COLUMNS)} FROM {CLICKHOUSE_TABLE} FINAL WHERE {conditions}"
    rows = client.query(query, parameters=params).result_rows
    products = [to_product(row) for row in rows]
    return {product.key: product for product in products}


def to_product(row: tuple[object, ...]) -> Product:
    """Строка таблицы в порядке `COLUMNS` → товар.

    Два поля называются по-разному с обеих сторон: `characteristics` в таблице —
    это `attributes` в модели, а время драйвер отдаёт `datetime`, тогда как в
    модели оно строкой.
    """
    values = dict(zip(COLUMNS, row, strict=True))
    values["attributes"] = values.pop("characteristics")
    values["collected_at"] = values["collected_at"].isoformat()
    return Product(**values)


def insert_products(client: Client, products: list[Product]) -> int:
    """Залить товары. Повторная заливка того же id заменяет строку, а не двоит.

    За это отвечает ReplacingMergeTree по `collected_at`: до слияния кусков
    строки живут обе, поэтому читатели используют `FINAL`.
    """
    rows = [to_row(product) for product in products]
    client.insert(CLICKHOUSE_TABLE, rows, column_names=list(COLUMNS))
    return len(rows)
