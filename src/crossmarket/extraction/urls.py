"""Разбор ссылок на карточки товара.

Только регулярки по строке. В сеть эти функции не ходят и ходить не должны:
план запрещает и парсинг с обходом антибота, и live-fetch по ссылкам.
Ссылка нужна ради идентификатора и провенанса, не ради содержимого.
"""

from __future__ import annotations

import re

_WB_RE = re.compile(r"/catalog/(\d+)")
_OZON_RE = re.compile(r"/product/(?:[^/?#]*-)?(\d+)")

_DIGITS_RE = re.compile(r"^\d+$")


def _parse(pattern: re.Pattern[str], value: str) -> str | None:
    """Идентификатор из ссылки. Голые цифры — тоже валидный ввод, их и вернём."""
    value = value.strip()
    if not value:
        return None
    if _DIGITS_RE.match(value):
        return value
    match = pattern.search(value)
    return match.group(1) if match else None


def parse_wb_id(url: str) -> str | None:
    """Артикул ВБ из `wildberries.ru/catalog/123456789/detail.aspx`.

    None, если распознать не удалось.
    """
    return _parse(_WB_RE, url)


def parse_ozon_id(url: str) -> str | None:
    """SKU Озона из `ozon.ru/product/nothing-ear-2025-1785773893/`.

    Артикул — последняя группа цифр в слаге: `[^/?#]*-` жадный и отступает до
    последнего дефиса перед цифрами. None, если распознать не удалось.
    """
    return _parse(_OZON_RE, url)
