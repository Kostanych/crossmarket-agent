"""Поиск карточек ВБ по вопросу: вектор → Qdrant → порог → снапшот ClickHouse.

Отдельный модуль, а не часть агента: зовут его двое — обёртка тула A и eval-харнесс,
которому Claude Agent SDK не нужен.

Карточка собирается из двух баз: Qdrant отдаёт скор и идентификатор, ClickHouse —
название, описание и характеристики, которых в payload нет.

Дистракторы выпадают сами: в ClickHouse их нет. Отсюда следствие, о котором надо
помнить: **выдача короче запрошенного `limit`** — на `limit=8` до модели доходит 4–6
карточек. Карточки после фильтра намеренно не добираются, иначе фон перестал бы
что-либо стоить. В проде эффекта нет: синтетика живёт только в eval-корпусе.
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

    `min_score` — часть тула, а не retrieval-метрик: в recall@k порог может только
    выкинуть верную карточку. Клиенты передаются снаружи — тул держит по одному на
    процесс, тесты подменяют оба.
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
    """Выдача текстом для модели.

    Характеристики отдаются целиком: карточек в выдаче единицы, а вопросы вида «а из
    чего он» упирались бы в обрезанное поле.
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
