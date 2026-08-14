"""Наполнение баз снапшотом из `data/products.jsonl`.

Читает товары, собранные `tools/dataset.py`, и раскладывает их по двум базам:
цены и структурные поля в ClickHouse (источник для tool C), тексты векторами
в две коллекции Qdrant (источник для A и B).

Запуск:
    docker compose --profile infra up -d
    poetry run python tools/load.py                # обе базы
    poetry run python tools/load.py --only qdrant  # только векторы

Идемпотентно: повторный прогон перезаписывает те же строки и точки, а не
двоит их. Первый запуск тянет веса эмбеддера с HuggingFace — это несколько
гигабайт и несколько минут.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from crossmarket.config import CLICKHOUSE_DB, CLICKHOUSE_TABLE, QDRANT_COLLECTIONS
from crossmarket.distractors import load_distractors
from crossmarket.embedding import COMPOSITIONS, DEFAULT_COMPOSITION
from crossmarket.indexing import index_products
from crossmarket.models import Marketplace, Product
from crossmarket.storage import clickhouse, qdrant
from crossmarket.storage.jsonl import load_products

MARKETPLACES: tuple[Marketplace, ...] = ("wb", "ozon")


def by_marketplace(products: list[Product]) -> dict[Marketplace, list[Product]]:
    grouped: dict[Marketplace, list[Product]] = defaultdict(list)
    for product in products:
        grouped[product.marketplace].append(product)
    return grouped


def load_clickhouse(products: list[Product]) -> None:
    client = clickhouse.connect()
    clickhouse.create_table(client)
    written = clickhouse.insert_products(client, products)
    total = client.query(f"SELECT count() FROM {CLICKHOUSE_TABLE} FINAL").result_rows[0][0]
    print(f"ClickHouse {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE}: залито {written}, всего в таблице {total}")


def load_qdrant(grouped: dict[Marketplace, list[Product]], composition: str, with_distractors: bool) -> None:
    client = qdrant.connect()
    if with_distractors:
        background = load_distractors()
        index_products(client, "wb", background, composition, synthetic=True)
        print(f"Qdrant {QDRANT_COLLECTIONS['wb']}: залито {len(background)} дистракторов")
    for marketplace in MARKETPLACES:
        products = grouped.get(marketplace, [])
        if not products:
            continue
        print(f"Qdrant {QDRANT_COLLECTIONS[marketplace]}: считаю векторы для {len(products)} карточек ({composition})…")
        written = index_products(client, marketplace, products, composition)
        total = client.count(QDRANT_COLLECTIONS[marketplace]).count
        print(f"Qdrant {QDRANT_COLLECTIONS[marketplace]}: залито {written}, всего в коллекции {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("clickhouse", "qdrant"), help="залить только одну базу")
    parser.add_argument(
        "--text",
        choices=sorted(COMPOSITIONS),
        default=DEFAULT_COMPOSITION,
        help="состав текста карточки в векторе",
    )
    parser.add_argument("--distractors", action="store_true", help="добить коллекцию ВБ придуманным фоном")
    args = parser.parse_args()

    products = list(load_products().values())
    if not products:
        raise SystemExit("data/products.jsonl пуст — сначала tools/dataset.py --write")
    print(f"Товаров в снапшоте: {len(products)}")

    if args.only != "qdrant":
        load_clickhouse(products)
    if args.only != "clickhouse":
        load_qdrant(by_marketplace(products), args.text, args.distractors)


main()
