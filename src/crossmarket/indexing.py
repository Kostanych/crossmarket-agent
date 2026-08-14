"""Наполнение коллекции Qdrant: текст карточки → вектор → точка.

Отдельный модуль, потому что вызывающих двое: `tools/load.py` заливает снапшот
один раз, а `tools/eval_retrieval.py` переиндексирует коллекцию под каждый
состав текста. Разъехавшиеся копии этой цепочки означали бы, что метрика
померена не на том, что лежит в базе.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crossmarket.embedding import DEFAULT_COMPOSITION, encode_passages, product_text
from crossmarket.models import Marketplace, Product
from crossmarket.storage import qdrant

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def index_products(
    client: QdrantClient,
    marketplace: Marketplace,
    products: list[Product],
    composition: str = DEFAULT_COMPOSITION,
    synthetic: bool = False,
) -> int:
    """Посчитать векторы для карточек площадки и залить их в коллекцию."""
    vectors = encode_passages([product_text(product, composition) for product in products])
    return qdrant.upsert_products(client, marketplace, products, vectors, synthetic)
