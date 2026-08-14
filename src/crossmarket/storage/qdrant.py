"""Две коллекции Qdrant: карточки ВБ и карточки Озона.

Раздельные коллекции, а не одна с фильтром по площадке, — потому что этого
требуют оба сценария поиска: tool A ищет по ВБ, tool B ищет кандидатов на
Озоне для товара с ВБ. Смешанная коллекция в обоих случаях потребовала бы
фильтра, который ничего не даёт.

Идентификатор точки — UUID, выведенный из пары (площадка, id): Qdrant не берёт
произвольные строки, а повторная заливка того же товара должна перезаписывать
точку, а не плодить дубли.

В payload лежит идентификатор, по которому карточка достаётся из ClickHouse, и
два поля для фильтрации выдачи — категория и цена. Цена дублирует ClickHouse
осознанно: план требует, чтобы ценовое ограничение в tool A применялось фильтром
по payload, а не походом в tool C. Остальные поля берутся из снапшота по id.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from crossmarket.config import EMBEDDING_DIM, QDRANT_COLLECTIONS, QDRANT_URL
from crossmarket.models import Marketplace, Product

if TYPE_CHECKING:
    from numpy import ndarray
    from qdrant_client import QdrantClient

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(marketplace: str, product_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{marketplace}:{product_id}"))


def connect() -> QdrantClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, marketplace: Marketplace) -> str:
    """Создать коллекцию площадки, если её нет. Возвращает имя."""
    from qdrant_client.models import Distance, VectorParams

    name = QDRANT_COLLECTIONS[marketplace]
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return name


def payload_of(product: Product) -> dict[str, Any]:
    return {
        "marketplace": product.marketplace,
        "id": product.id,
        "category": product.category,
        "price_rub": product.price_rub,
    }


def upsert_products(
    client: QdrantClient,
    marketplace: Marketplace,
    products: list[Product],
    vectors: ndarray,
) -> int:
    """Залить карточки с готовыми векторами. Порядок `vectors` — как у `products`."""
    from qdrant_client.models import PointStruct

    name = ensure_collection(client, marketplace)
    points = [
        PointStruct(id=point_id(product.marketplace, product.id), vector=vector.tolist(), payload=payload_of(product))
        for product, vector in zip(products, vectors, strict=True)
    ]
    client.upsert(collection_name=name, points=points)
    return len(points)


def search(
    client: QdrantClient,
    marketplace: Marketplace,
    vector: ndarray,
    limit: int = 10,
    category: str | None = None,
    price_max: int | None = None,
) -> list[dict[str, Any]]:
    """Ближайшие карточки площадки, при желании — с фильтрами по payload."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    conditions = []
    if category is not None:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if price_max is not None:
        conditions.append(FieldCondition(key="price_rub", range=Range(lte=price_max)))
    query_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=QDRANT_COLLECTIONS[marketplace],
        query=vector.tolist(),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return [{"score": point.score, **(point.payload or {})} for point in response.points]
