"""Модели данных: товар из снапшота и метка пары ВБ↔Озон.

Живут в пакете, а не внутри разметчика: те же модели читают загрузчики
в Qdrant и ClickHouse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal

Marketplace = Literal["wb", "ozon"]
LabelValue = Literal["match", "no_match"]
NegativeKind = Literal["hard", "soft"]
Split = Literal["calibration", "test"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _known_fields(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


@dataclass
class Product:
    """Карточка товара на дату снапшота."""

    marketplace: Marketplace
    id: str
    url: str = ""
    title: str = ""
    description: str = ""
    price_rub: int | None = None
    category: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    review_count: int | None = None
    collected_at: str = field(default_factory=_now)

    @property
    def key(self) -> tuple[str, str]:
        return (self.marketplace, self.id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Product:
        """Незнакомые ключи игнорируются: в старых строках JSONL схема другая."""
        known = _known_fields(cls)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Label:
    """Метка пары. Ground truth для eval tool B.

    Хранится отдельно от товаров и никогда не попадает в ClickHouse, который читает
    агент: иначе он подсмотрит ответ.

    `negative_kind`: `hard` — похожий товар с отличием, `soft` — случайная карточка.
    Без доли hard в тест-сете метрика B надувается.

    `split` проставляется харнессом один раз и после этого не меняется: на парах
    калибровки доводится промпт модели подтверждения, отчётный F1 снимается с теста.
    Переразметка пары дописывает строку без сплита — его вернёт `evals.pairs`.
    """

    wb_id: str
    ozon_id: str
    label: LabelValue
    negative_kind: NegativeKind | None = None
    source: str = "manual"
    comment: str = ""
    labeled_at: str = field(default_factory=_now)
    split: Split | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.wb_id, self.ozon_id)

    @property
    def id(self) -> str:
        """Идентификатор пары для разбиения на сплиты и для отчётов."""
        return f"{self.wb_id}:{self.ozon_id}"

    @property
    def stratum(self) -> str:
        """`match`, `no_match-hard` или `no_match-soft` — разбиение стратифицируется по ней."""
        return self.label if self.label == "match" else f"{self.label}-{self.negative_kind or 'hard'}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Label:
        known = _known_fields(cls)
        return cls(**{k: v for k, v in raw.items() if k in known})
