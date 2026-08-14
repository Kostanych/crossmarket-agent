"""Дистракторы — придуманные карточки-фон для коллекции ВБ.

Зачем: retrieval ищет десять карточек из 184, это 5.4% корпуса, и recall@10
упирается в потолок — на четырёх составах текста он принял всего два разных
значения, то есть почти ничего не различал. Дистракторы разбавляют корпус,
чтобы попадание в топ-10 снова что-то значило.

Карточки намеренно тупые, но в рамках своей категории: так они мешают поиску
как фон и при этом заведомо не являются ответом ни на один вопрос golden-set.
Проверять их глазами по одной поэтому не нужно.

Живут только в Qdrant и только в коллекции ВБ. В ClickHouse их нет — там tool C
считает настоящие средние цены. В коллекции Озона их нет — оттуда tool B берёт
кандидатов, и придуманная карточка создала бы ложный не-матч.
"""

from __future__ import annotations

import json
from pathlib import Path

from crossmarket.models import Product

DISTRACTORS_FILE = Path("evals/distractors.jsonl")
ID_PREFIX = "syn"


def load_distractors(path: Path = DISTRACTORS_FILE) -> list[Product]:
    """Прочитать дистракторы и выдать их как товары ВБ с искусственными id.

    Идентификатор `syn0001` не пересекается с артикулами площадки, поэтому
    дистрактор невозможно спутать с настоящей карточкой ни в выдаче, ни в
    лейблах.
    """
    products = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            products.append(
                Product(
                    marketplace="wb",
                    id=f"{ID_PREFIX}{number:04d}",
                    title=raw["title"],
                    description=raw.get("description", ""),
                    price_rub=raw["price_rub"],
                    category=raw["category"],
                    attributes=raw.get("characteristics", {}),
                )
            )
    return products
