"""Локальный эмбеддер поверх sentence-transformers.

Модель по умолчанию — `intfloat/multilingual-e5-large`: тексты русские, нужен
мультиязычный энкодер. Веса тянутся с HuggingFace при первом вызове и дальше
живут в кеше `~/.cache/huggingface`.

Семейство e5 обучено с префиксами `query:` и `passage:` — без них качество
падает молча, поэтому голого `encode` наружу нет: документы кодируются через
`encode_passages`, вопросы через `encode_queries`. Модель грузится один раз на
процесс и переиспользуется.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from crossmarket.config import EMBEDDING_MODEL
from crossmarket.models import Product

if TYPE_CHECKING:
    from numpy import ndarray

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "title+attrs": ("title", "attributes"),
    "title+attrs+cat": ("title", "attributes", "category"),
    "title+attrs+desc": ("title", "attributes", "description"),
    "title+attrs+cat+desc": ("title", "attributes", "category", "description"),
}
DEFAULT_COMPOSITION = "title+attrs+desc"

_LABELS = {"attributes": "Характеристики", "category": "Категория", "description": "Описание"}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def product_text(product: Product, composition: str = DEFAULT_COMPOSITION) -> str:
    """Текст карточки для индексации.

    Состав меняется, потому что он влияет на retrieval сильнее всего остального
    и подбирается замером: `tools/eval_retrieval.py --text` гоняет варианты и
    кладёт метрики каждого отдельным прогоном MLflow. Название есть всегда,
    остальные части подписаны, чтобы модель видела, где кончается одно поле и
    начинается другое. Пустые части выпадают — карточек без характеристик и без
    описания в выгрузках хватает.
    """
    values = {
        "title": product.title,
        "attributes": "; ".join(f"{key}: {value}" for key, value in product.attributes.items()),
        "category": product.category,
        "description": product.description,
    }

    parts = []
    for name in COMPOSITIONS[composition]:
        value = values[name].strip()
        if not value:
            continue
        parts.append(f"{_LABELS[name]}: {value}" if name in _LABELS else value)
    return "\n".join(parts)


def encode_passages(texts: list[str], batch_size: int = 16) -> ndarray:
    """Векторы документов. Нормализованы, так что косинус равен скалярному."""
    prefixed = [PASSAGE_PREFIX + text for text in texts]
    return _model().encode(prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def encode_queries(texts: list[str], batch_size: int = 16) -> ndarray:
    """Векторы запросов. Тот же энкодер, другой префикс."""
    prefixed = [QUERY_PREFIX + text for text in texts]
    return _model().encode(prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
