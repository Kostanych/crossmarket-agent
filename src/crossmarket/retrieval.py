"""Поиск карточек ВБ по вопросу: вектор → Qdrant → порог → снапшот ClickHouse.

Отдельный модуль, а не часть агента, потому что зовут его двое: обёртка тула A и
eval-харнесс, а харнессу Claude Agent SDK не нужен.

Карточка собирается из двух баз: Qdrant отдаёт скор и идентификатор, ClickHouse —
название, описание и характеристики, которых в payload нет.

Дистракторы из выдачи выпадают сами: в ClickHouse их нет, потому что там считаются
настоящие цены. Фоном они при этом работают — занимают верхние места и вытесняют
настоящие карточки, ради чего и заливались.

Побочное следствие, которое надо помнить: **выдача короче запрошенного `limit`**.
На `limit=8` до модели доходит 4–6 карточек, остальное съел фон. Добирать карточки
после фильтра намеренно не стал: тогда дистракторы перестали бы что-либо стоить и
метрика снова стала бы слишком лёгкой. В проде эффекта нет вовсе — синтетика живёт
только в eval-корпусе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crossmarket.config import RETRIEVAL_MIN_SCORE
from crossmarket.embedding import encode_queries
from crossmarket.models import Marketplace, Product
from crossmarket.storage import clickhouse, qdrant

if TYPE_CHECKING:
    from clickhouse_connect.driver.client import Client
    from qdrant_client import QdrantClient


@dataclass
class Hit:
    """Найденная карточка вместе со скором, по которому её пропустил порог."""

    product: Product
    score: float


def search_products(
    question: str,
    qdrant_client: QdrantClient,
    clickhouse_client: Client,
    marketplace: Marketplace = "wb",
    limit: int = 5,
    category: str | None = None,
    price_max: int | None = None,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> list[Hit]:
    """Карточки, похожие на вопрос, от самой похожей к менее.

    `min_score` — часть тула, а не retrieval-метрик: в recall@k порог мог бы только
    выкинуть верную карточку, а тулу без него нечем ответить «нет в базе» — на любой
    вопрос вернулись бы ближайшие карточки, и модель ответила бы по мусору.

    Клиенты передаются снаружи: тул держит по одному на процесс, а тесты подменяют
    оба, не поднимая баз.
    """
    vector = encode_queries([question])[0]
    points = qdrant.search(
        qdrant_client,
        marketplace,
        vector,
        limit=limit,
        category=category,
        price_max=price_max,
    )
    passed = [point for point in points if point["score"] >= min_score]
    if not passed:
        return []

    products = clickhouse.fetch_products(clickhouse_client, [(marketplace, point["id"]) for point in passed])
    return [
        Hit(product=products[(marketplace, point["id"])], score=point["score"])
        for point in passed
        if (marketplace, point["id"]) in products
    ]


def format_hits(hits: list[Hit]) -> str:
    """Выдача текстом для модели: то, что ушло бы в ответ, и ничего сверх.

    Характеристики отдаются целиком — вопросы вида «а из чего он» иначе упираются
    в обрезанное поле, а карточек в выдаче единицы.
    """
    if not hits:
        return "Ничего не найдено."
    return "\n\n".join(_format_hit(hit) for hit in hits)


def _format_hit(hit: Hit) -> str:
    product = hit.product
    lines = [
        f"id: {product.id}",
        f"Название: {product.title}",
        f"Цена, ₽: {product.price_rub}",
        f"Категория: {product.category}",
        f"Ссылка: {product.url}",
        f"Скор: {hit.score:.3f}",
    ]
    if product.attributes:
        lines.append("Характеристики: " + "; ".join(f"{k}: {v}" for k, v in product.attributes.items()))
    if product.description:
        lines.append(f"Описание: {product.description}")
    return "\n".join(lines)
