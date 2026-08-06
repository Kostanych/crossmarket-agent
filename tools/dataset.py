"""Сборка датасета из выгрузок скрапера и сид-разметки.

Читает `data/wb/*.csv`, `data/ozon/*.csv` и `data/разметка.csv`, печатает три
отчёта — полнота полей, целостность выгрузок, покрытие разметки — и по флагу
`--write` складывает результат в `data/products.jsonl` и `data/labels.jsonl`.

Запуск:
    poetry run python tools/dataset.py            # только отчёт
    poetry run python tools/dataset.py --write    # отчёт и запись

Отчёт по умолчанию без записи: сначала смотрим, что собралось, потом решаем,
писать ли. Сид-источник разметки в репозиторий не попадает — он лежит в
`data/`, который в `.gitignore`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from crossmarket.extraction.scraped import iter_dumps
from crossmarket.models import Label, Marketplace, Product
from crossmarket.storage.jsonl import DATA_DIR, append_label, append_product

SEED_FILE = "разметка.csv"
SEED_COLUMNS = {"wb": "вб", "ozon": "озон", "result": "result", "comment": "comment"}
MARKETPLACES: tuple[Marketplace, ...] = ("wb", "ozon")

# Поля, отсутствие которых стоит видеть в отчёте. Из них обязательна только
# цена — без неё карточка выбраковывается, см. Dumps.
REQUIRED_FIELDS = ("title", "price_rub", "category", "attributes", "description")


class Dumps:
    """Разобранные выгрузки одной площадки плюс всё, что с ними не так."""

    def __init__(self, marketplace: Marketplace, root: Path) -> None:
        self.marketplace = marketplace
        self.files = sorted((root / marketplace).glob("*.csv"))
        self.products: dict[str, Product] = {}
        self.without_id: list[str] = []
        self.without_price: list[str] = []
        self.duplicates: list[str] = []

        for path, product in iter_dumps(marketplace, root):
            if not product.id:
                self.without_id.append(path.name)
                continue
            if product.price_rub is None:
                # Карточка без цены выбраковывается: сравнение цен между
                # площадками — половина задачи, а дырка в снапшоте молча
                # превратится в неверный агрегат tool C.
                self.without_price.append(product.id)
                continue
            if product.id in self.products:
                self.duplicates.append(product.id)
            self.products[product.id] = product

    @property
    def unreadable(self) -> int:
        accounted = len(self.products) + len(self.without_id) + len(self.without_price) + len(self.duplicates)
        return len(self.files) - accounted


def load_seed(path: Path) -> list[dict[str, str]]:
    """Сид-разметка: `вб;озон;result;comment`, где result 1 — матч, 0 — не матч."""
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(text.splitlines(), delimiter=";"))


def seed_to_label(row: dict[str, str]) -> Label:
    """Строка сида → метка.

    Все нули сида — hard negatives: список собирался из пар-кандидатов, которые
    похожи, иначе их не с чем было бы сравнивать. Easy negatives придётся
    добирать отдельно, случайными парами.
    """
    is_match = row[SEED_COLUMNS["result"]].strip() == "1"
    return Label(
        wb_id=row[SEED_COLUMNS["wb"]].strip(),
        ozon_id=row[SEED_COLUMNS["ozon"]].strip(),
        label="match" if is_match else "no_match",
        negative_kind=None if is_match else "hard",
        source="seed",
        comment=row.get(SEED_COLUMNS["comment"], "").strip(),
    )


def split_by_coverage(labels: list[Label], dumps: dict[str, Dumps]) -> dict[str, list[Label]]:
    """Разложить метки по тому, обе ли карточки пары есть на диске."""
    buckets: dict[str, list[Label]] = {"both": [], "wb_only": [], "ozon_only": [], "neither": []}
    for label in labels:
        has_wb = label.wb_id in dumps["wb"].products
        has_ozon = label.ozon_id in dumps["ozon"].products
        name = "both" if has_wb and has_ozon else "wb_only" if has_wb else "ozon_only" if has_ozon else "neither"
        buckets[name].append(label)
    return buckets


def report_fields(dumps: dict[str, Dumps]) -> None:
    print("\n== Полнота полей ==")
    for marketplace, side in dumps.items():
        total = len(side.products)
        holes = Counter()
        for product in side.products.values():
            for name in REQUIRED_FIELDS:
                if not getattr(product, name):
                    holes[name] += 1
        print(f"\n{marketplace}: карточек {total}")
        if not holes:
            print("  все поля на месте")
        for name in REQUIRED_FIELDS:
            missing = holes[name]
            if missing:
                mark = "ВСЕ" if missing == total else f"{missing}/{total}"
                print(f"  нет {name:12} {mark}")


def report_values(dumps: dict[str, Dumps]) -> None:
    """Что именно разобралось.

    Полнота полей ловит пустоту, но не мусор: когда селектор скрапера съезжает
    на соседний блок, поле остаётся непустым и выглядит здоровым. Так у ВБ в
    категорию попало «Артикул / 1128813622». Поэтому — глазами на значения.
    """
    print("\n== Как разобралось ==")
    for marketplace, side in dumps.items():
        if not side.products:
            continue
        categories = Counter(p.category for p in side.products.values() if p.category)
        print(f"\n{marketplace}: категорий {len(categories)} на {len(side.products)} карточек")
        for value, count in categories.most_common(5):
            print(f"  {count:3}  {value!r}")

        example = next(iter(side.products.values()))
        print(f"  пример карточки {example.id}:")
        for name in ("title", "price_rub", "category"):
            print(f"    {name:10} {getattr(example, name)!r}")
        first_attr = next(iter(example.attributes.items()), None)
        print(f"    {'attributes':10} {len(example.attributes)} шт, первая {first_attr!r}")


def report_dumps(dumps: dict[str, Dumps]) -> None:
    print("\n== Выгрузки ==")
    for marketplace, side in dumps.items():
        print(f"{marketplace}: файлов {len(side.files)}, карточек {len(side.products)}")
        if side.without_price:
            print(f"  выбраковано без цены: {len(side.without_price)} — {', '.join(side.without_price)}")
        if side.duplicates:
            print(f"  один товар в нескольких файлах: {', '.join(sorted(set(side.duplicates)))}")
        if side.without_id:
            print(f"  без идентификатора: {', '.join(side.without_id)}")
        if side.unreadable:
            print(f"  не разобрались: {side.unreadable}")


def report_coverage(buckets: dict[str, list[Label]], dumps: dict[str, Dumps], labels: list[Label]) -> None:
    titles = {
        "both": "обе карточки на диске",
        "wb_only": "есть только ВБ",
        "ozon_only": "есть только Озон",
        "neither": "нет ни одной",
    }
    print(f"\n== Покрытие разметки ({len(labels)} пар) ==")
    for name, title in titles.items():
        items = buckets[name]
        matches = sum(1 for label in items if label.label == "match")
        print(f"  {title:22} {len(items):3}  матчей {matches}, не-матчей {len(items) - matches}")

    print("\n== Скрапнуто мимо разметки ==")
    for marketplace, side in dumps.items():
        field = "wb_id" if marketplace == "wb" else "ozon_id"
        known = {getattr(label, field) for label in labels}
        extra = sorted(set(side.products) - known)
        print(f"  {marketplace}: {len(extra)}" + (f" — {', '.join(extra)}" if extra else ""))


def write_dataset(dumps: dict[str, Dumps], complete: list[Label], data_dir: Path) -> None:
    """Товары — все скрапнутые, метки — только для полных пар.

    Карточка без пары в корпусе всё равно нужна: для retrieval она кандидат
    и дистрактор. Метка без карточки не нужна никому — проверить её нечем.
    """
    for side in dumps.values():
        for product in side.products.values():
            append_product(product, data_dir)
    for label in complete:
        append_label(label, data_dir)
    print(f"\nЗаписано: карточек {sum(len(s.products) for s in dumps.values())}, меток {len(complete)}")


def main() -> None:
    # Консоль в cp1251, а данные русскоязычные.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--write", action="store_true", help="записать products.jsonl и labels.jsonl")
    args = parser.parse_args()

    dumps = {marketplace: Dumps(marketplace, args.data_dir) for marketplace in MARKETPLACES}
    labels = [seed_to_label(row) for row in load_seed(args.data_dir / SEED_FILE)]
    buckets = split_by_coverage(labels, dumps)

    report_dumps(dumps)
    report_fields(dumps)
    report_values(dumps)
    report_coverage(buckets, dumps, labels)

    if args.write:
        write_dataset(dumps, buckets["both"], args.data_dir)
    else:
        print(f"\nК записи готовы {len(buckets['both'])} пар. Запись: --write")


if __name__ == "__main__":
    main()
