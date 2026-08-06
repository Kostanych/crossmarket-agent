"""Извлечение полей карточки Wildberries из CSV браузерного скрапера.

Колонки те же, что у Озона, но содержимое ячеек различается.
Совпадает только идея «один файл на товар».
Отличия:

* значима **только первая строка** файла: скрапер сваливает в колонку `id`
  остатки характеристик, и в строках со второй по последнюю лежит мусор;
* в `price` две строки — цена с кошельком ВБ и без него; берётся первая,
  та, что показана на карточке крупно;
* разряды разделены неразрывным пробелом U+00A0, а не узким, как на Озоне;
* в `reviews` разделитель — `·` (U+00B7), а не `•`, и второй строкой идёт
  число вопросов, которое к отзывам отношения не имеет;
* строка с категориями начинаются с «Главная», это не категория;
* в характеристиках заголовки секций вперемешку с парами (см.
  `parse_characteristics`).

"""

from __future__ import annotations

import csv
import io
import re

from crossmarket.models import Product

DESCRIPTION_HEADER = "Описание"
BREADCRUMB_ROOT = "Главная"
CATEGORY_DEPTH = 2
CATEGORY_SEPARATOR = " / "
RATING_SEPARATOR = "·"
CHARACTERISTICS_COLUMNS = ("charasteristics", "characteristics")


def _first_row(text: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    for row in reader:
        return {(k or "").strip().lower(): (v or "") for k, v in row.items()}
    raise ValueError("CSV пуст: нет ни одной строки данных.")


def _cell(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name, "").strip():
            return row[name]
    return ""


def parse_price(raw: str) -> int | None:
    """`'871 ₽\\n889 ₽'` → `871`.

    Первая строка — цена с кошельком ВБ, вторая — без него. Берём первую:
    именно она показана на карточке как основная.
    """
    first_line = raw.strip().split("\n")[0]
    digits = re.sub(r"\D", "", first_line.split(",")[0])
    return int(digits) if digits else None


def parse_review_count(raw: str) -> int | None:
    """`'4,9 · 18 491 оценка\\n3 660 вопросов'` → `18491`.

    Первое число — рейтинг, отделён `·`. Вторая строка — вопросы, не отзывы.
    """
    first_line = raw.strip().split("\n")[0]
    _, separator, tail = first_line.partition(RATING_SEPARATOR)
    if not separator:
        tail = re.sub(r"^\s*\d+[.,]\d+", "", first_line)
    digits = re.sub(r"\D", "", tail)
    return int(digits) if digits else None


def parse_categories(raw: str) -> str:
    """Две верхние категории, без корня «Главная»."""
    levels = [line.strip() for line in raw.split("\n") if line.strip()]
    if levels and levels[0] == BREADCRUMB_ROOT:
        levels = levels[1:]
    return CATEGORY_SEPARATOR.join(levels[:CATEGORY_DEPTH])


def parse_description(raw: str) -> str:
    """Снять служебный заголовок «Описание» с начала блока."""
    lines = raw.split("\n")
    if lines and lines[0].strip() == DESCRIPTION_HEADER:
        lines = lines[1:]
    return "\n".join(lines).strip()


def parse_characteristics(raw: str) -> dict[str, str]:
    """Блок характеристик → словарь.

    Пустые строки здесь не мусор, а разделитель, и на нём держится разбор.
    После значения возможны ровно два продолжения:

    * две пустые строки, дальше следующий ключ;
    * сразу непустая строка — это заголовок секции («Материалы», «Габариты»),
      его надо пропустить, ключ идёт за ним.

    Первая строка блока — всегда заголовок секции. Значение забирается строкой
    следом за ключом вслепую, поэтому пустая строка внутри значения разбор не
    ломает — ломала бы только пустая строка на месте ключа, а такой в выгрузках
    не встречается.
    """
    lines = raw.split("\n")
    attributes: dict[str, str] = {}
    index = 0
    expect_section = True

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            expect_section = False
            continue
        if expect_section:
            index += 1
            expect_section = False
            continue

        key = lines[index].strip()
        index += 1
        value = lines[index].strip() if index < len(lines) else ""
        index += 1
        if key:
            attributes[key] = value
        expect_section = True

    return attributes


def extract_wb_csv(text: str, url: str = "") -> Product:
    """Собрать `Product` из выгрузки скрапера по одной карточке ВБ."""
    row = _first_row(text)

    return Product(
        marketplace="wb",
        id=row.get("id", "").strip(),
        url=url,
        title=row.get("title", "").strip(),
        description=parse_description(row.get("desc", "")),
        price_rub=parse_price(row.get("price", "")),
        category=parse_categories(row.get("cats", "")),
        attributes=parse_characteristics(_cell(row, *CHARACTERISTICS_COLUMNS)),
        review_count=parse_review_count(row.get("reviews", "")),
    )
