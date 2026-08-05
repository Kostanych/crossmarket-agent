"""JSONL-хранилище этапа 1.

Инфраструктуры ещё нет, поэтому разметчик пишет в файлы; загрузчики в Qdrant
и ClickHouse читают их отдельно и позже.

Файлы append-only, при чтении побеждает последняя запись по ключу. Перезалить
товар или переставить метку — это дописать строку в конец: кода перезаписи
файла нет.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from crossmarket.models import Label, Marketplace, Product

DATA_DIR = Path("data")
PRODUCTS_FILE = "products.jsonl"
LABELS_FILE = "labels.jsonl"
RAW_DIR = "raw"

_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9_-]")


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_product(product: Product, data_dir: Path = DATA_DIR) -> None:
    _append(data_dir / PRODUCTS_FILE, product.to_dict())


def append_label(label: Label, data_dir: Path = DATA_DIR) -> None:
    _append(data_dir / LABELS_FILE, label.to_dict())


def load_products(data_dir: Path = DATA_DIR) -> dict[tuple[str, str], Product]:
    """Товары по ключу (marketplace, id). Последняя запись побеждает."""
    result: dict[tuple[str, str], Product] = {}
    for raw in _read(data_dir / PRODUCTS_FILE):
        product = Product.from_dict(raw)
        result[product.key] = product
    return result


def load_labels(data_dir: Path = DATA_DIR) -> dict[tuple[str, str], Label]:
    """Метки по ключу (wb_id, ozon_id). Последняя запись побеждает."""
    result: dict[tuple[str, str], Label] = {}
    for raw in _read(data_dir / LABELS_FILE):
        label = Label.from_dict(raw)
        result[label.key] = label
    return result


def save_raw(
    marketplace: Marketplace,
    product_id: str,
    content: str,
    data_dir: Path = DATA_DIR,
    ext: str = "csv",
) -> Path:
    """Сохранить исходную выгрузку рядом с извлечёнными полями.

    Если мэппинг окажется неполным, перечитать можно отсюда.

    Идентификатор попадает в имя файла, поэтому чистится: выход за каталог
    через `../` недопустим.
    """
    safe_id = _UNSAFE_IN_FILENAME.sub("_", product_id) or "unknown"
    path = data_dir / RAW_DIR / marketplace / f"{safe_id}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
