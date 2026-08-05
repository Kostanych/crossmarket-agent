"""Поиск выгрузки скрапера по идентификатору товара.

Скрапер кладёт каждый товар отдельным файлом и имя даёт своё
(`blaze_scrapper (17).csv`), поэтому связать ссылку с файлом можно только по
содержимому: идентификатор лежит внутри — у ВБ колонкой `id`, у Озона
характеристикой «Артикул».

Раскладка `<root>/<marketplace>/*.csv` — на этапе 1 это `data/wb/` и
`data/ozon/`. Площадку определяет человек, когда раскладывает файлы: скрапится
сначала одна площадка, потом другая, так что это ничего не стоит.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from crossmarket.extraction.ozon import extract_ozon_csv
from crossmarket.extraction.wb import extract_wb_csv
from crossmarket.models import Marketplace, Product

EXTRACTORS = {"wb": extract_wb_csv, "ozon": extract_ozon_csv}

# Чужой файл в папке или обрезанная выгрузка не должны ронять поиск: файлов
# сотни, и один битый не повод останавливать разметку.
_BROKEN_DUMP = (OSError, UnicodeDecodeError, ValueError, KeyError, csv.Error)


def read_dump(marketplace: Marketplace, path: Path, url: str = "") -> Product:
    """Разобрать выгрузку скрапера. Кодировка с BOM — так её пишет расширение."""
    return EXTRACTORS[marketplace](path.read_text(encoding="utf-8-sig"), url=url)


def iter_dumps(marketplace: Marketplace, root: Path) -> Iterator[tuple[Path, Product]]:
    """Все читаемые выгрузки площадки. Нечитаемые молча пропускаются."""
    folder = root / marketplace
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.csv")):
        try:
            yield path, read_dump(marketplace, path)
        except _BROKEN_DUMP:
            continue


def find_dump(marketplace: Marketplace, product_id: str, root: Path) -> Path | None:
    """Файл с карточкой этого товара, либо None, если такого в папке нет."""
    for path, product in iter_dumps(marketplace, root):
        if product.id == product_id:
            return path
    return None
