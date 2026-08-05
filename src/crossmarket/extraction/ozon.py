"""Извлечение полей карточки Озона из CSV браузерного скрапера.

Источник — расширение, которое снимает уже отрендеренную страницу, открытую
руками.

Формат: один файл на товар, колонки `title, price, cats, characteristics,
desc, reviews, id`. Переводы строк внутри ячеек сохраняются.

Файл может содержать несколько строк: значимая только первая, в остальных
заполнено одно `desc` — длинное описание разбивается по абзацам.
"""

from __future__ import annotations

import csv
import io
import re

from crossmarket.models import Product

ARTICLE_KEY = "Артикул"
BRAND_KEY = "Бренд"
DISCLAIMER_PREFIX = "Информация о технических характеристиках"
HEADER_LINES = frozenset({"Характеристики", "Добавить к сравнению"})
CATEGORY_DEPTH = 2
CATEGORY_SEPARATOR = " / "


def _rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    return [{(k or "").strip().lower(): (v or "") for k, v in row.items()} for row in reader]


def _first(rows: list[dict[str, str]], column: str) -> str:
    """Первое непустое значение колонки: скалярные поля лежат в первой строке."""
    return next((row[column].strip() for row in rows if row.get(column, "").strip()), "")


def _joined_description(rows: list[dict[str, str]]) -> str:
    parts = [row["desc"].strip() for row in rows if row.get("desc", "").strip()]
    return "\n\n".join(parts)


def parse_price(raw: str) -> int | None:
    """`'16 862 ₽'` → `16862`. Разделитель разрядов — узкий пробел U+2009."""
    whole = raw.split(",")[0]
    digits = re.sub(r"\D", "", whole)
    return int(digits) if digits else None


def parse_review_count(raw: str) -> int | None:
    """`'4.9 • 98 697 отзывов'` → `98697`.

    Первое число — рейтинг, он отбрасывается: разделяет их `•`.
    """
    after_bullet = re.search(r"•(.*)", raw, flags=re.DOTALL)
    tail = after_bullet.group(1) if after_bullet else re.sub(r"^\s*\d+[.,]\d+", "", raw)
    digits = re.sub(r"\D", "", tail)
    return int(digits) if digits else None


def parse_categories(raw: str) -> str:
    """Первые два элемента - две верхние категории.
    Последний элемент строки - бренд.
    """
    levels = [line.strip() for line in raw.split("\n") if line.strip()]
    return CATEGORY_SEPARATOR.join(levels[:CATEGORY_DEPTH])


def parse_brand(cats_raw: str, attributes: dict[str, str]) -> str:
    """Бренд карточки.

    В характеристиках ключ «Бренд» есть не всегда, зато последний уровень
    в строке с категориями - это он и есть.
    """
    explicit = attributes.get(BRAND_KEY, "").strip()
    if explicit:
        return explicit
    levels = [line.strip() for line in cats_raw.split("\n") if line.strip()]
    return levels[-1] if len(levels) > CATEGORY_DEPTH else ""


def parse_characteristics(raw: str) -> dict[str, str]:
    """Блок характеристик → словарь.

    Две формы записи в одном блоке:

    * пара строк — ключ, следом значение;
    * одна строка `Ключ: значение, значение` — так Озон отдаёт мультизначные
      характеристики.

    Строка проверяется на двоеточие только в позиции ключа: значение всегда
    забирается вслепую, иначе двоеточие внутри значения сбило бы чередование.
    """
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    lines = [line for line in lines if not line.startswith(DISCLAIMER_PREFIX)]

    if ARTICLE_KEY in lines:
        # Артикул — надёжный якорь: всё до него и он сам вместе со значением шапка.
        lines = lines[lines.index(ARTICLE_KEY) + 2 :]
    else:
        lines = [line for line in lines if line not in HEADER_LINES]

    attributes: dict[str, str] = {}
    index = 0
    while index < len(lines):
        key, separator, value = lines[index].partition(": ")
        if separator:
            attributes[key.strip()] = value.strip()
            index += 1
        elif index + 1 < len(lines):
            attributes[lines[index]] = lines[index + 1]
            index += 2
        else:
            # Ключ без значения в самом конце
            index += 1
    return attributes


def extract_ozon_csv(text: str, url: str = "") -> Product:
    """Собрать `Product` из выгрузки скрапера по одной карточке Озона."""
    rows = _rows(text)
    if not rows:
        raise ValueError("CSV пуст: нет ни одной строки данных.")

    cats_raw = _first(rows, "cats")
    attributes = parse_characteristics(_first(rows, "characteristics"))

    return Product(
        marketplace="ozon",
        id=_first(rows, "id"),
        url=url,
        title=_first(rows, "title"),
        description=_joined_description(rows),
        price_rub=parse_price(_first(rows, "price")),
        category=parse_categories(cats_raw),
        brand=parse_brand(cats_raw, attributes),
        attributes=attributes,
        review_count=parse_review_count(_first(rows, "reviews")),
    )
