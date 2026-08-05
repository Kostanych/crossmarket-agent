"""Разбор ссылок на карточки товара.

Только регулярки по строке. В сеть эти функции не ходят и ходить не должны:
план запрещает и парсинг с обходом антибота, и live-fetch по ссылкам.
Ссылка нужна ради идентификатора и провенанса, не ради содержимого.
"""

from __future__ import annotations

import re

# https://www.wildberries.ru/catalog/123456789/detail.aspx
_WB_RE = re.compile(r"/catalog/(\d+)")

# https://www.ozon.ru/product/naushniki-nothing-ear-2025-1785773893/
# Артикул — последняя группа цифр в слаге: `[^/?#]*-` жадный и отступает
# до последнего дефиса перед цифрами.
_OZON_RE = re.compile(r"/product/(?:[^/?#]*-)?(\d+)")

_DIGITS_RE = re.compile(r"^\d+$")


def _parse(pattern: re.Pattern[str], value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if _DIGITS_RE.match(value):
        # Вставили голый артикул, а не ссылку — тоже валидный ввод.
        return value
    match = pattern.search(value)
    return match.group(1) if match else None


def parse_wb_id(url: str) -> str | None:
    """Артикул ВБ из ссылки, либо None, если распознать не удалось."""
    return _parse(_WB_RE, url)


def parse_ozon_id(url: str) -> str | None:
    """SKU Озона из ссылки, либо None, если распознать не удалось."""
    return _parse(_OZON_RE, url)
