"""Сборка датасета из выгрузок скрапера и сид-разметки.

Читает `data/wb/*.csv`, `data/ozon/*.csv` и сид-разметку из `data/`, печатает три
отчёта — полнота полей, целостность выгрузок, покрытие разметки — и по флагу
`--write` складывает результат в `data/products.jsonl` и `data/labels.jsonl`.

Запуск:
    poetry run python tools/dataset.py            # только отчёт
    poetry run python tools/dataset.py --write    # отчёт и запись

Отчёт по умолчанию без записи. Сид-источник разметки в репозиторий не попадает:
он лежит в `data/`, который в `.gitignore`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from crossmarket.extraction.scraped import iter_dumps
from crossmarket.models import Label, Marketplace, Product
from crossmarket.storage.jsonl import DATA_DIR, append_label, append_product

SEED_GLOB = "разметка_*.csv"
SEED_COLUMNS = {"wb": "wb", "ozon": "ozon", "result": "result", "comment": "comment"}
SEED_ALIASES = {"вб": "wb", "озон": "ozon"}
MARKETPLACES: tuple[Marketplace, ...] = ("wb", "ozon")

REQUIRED_FIELDS = ("title", "price_rub", "category", "attributes", "description")


class Dumps:
    """Разобранные выгрузки одной площадки плюс всё, что с ними не так.

    Карточка без цены выбраковывается: дырка в снапшоте молча превратится в неверный
    агрегат tool C. Остальные поля из `REQUIRED_FIELDS` только считаются в отчёте.
    """

    def __init__(self, marketplace: Marketplace, root: Path) -> None:
        self.marketplace = marketplace
        self.files = sorted((root / marketplace).rglob("*.csv"))
        self.products: dict[str, Product] = {}
        self.without_id: list[str] = []
        self.without_price: list[str] = []
        self.duplicates: list[str] = []

        for path, product in iter_dumps(marketplace, root):
            if not product.id:
                self.without_id.append(path.name)
                continue
            if product.price_rub is None:
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
    """Сид-разметка: `wb;ozon;result;comment`, где result 1 — матч, 0 — не матч.

    Кодировка у разных выгрузок сида разная (utf-8 с BOM или cp1251) и в теле
    файла кириллица есть, поэтому кодировка подбирается перебором. Имена колонок
    приводятся к нижнему регистру: они уже приезжали и строчными, и прописными.
    Первые две колонки приезжали и по-русски (`ВБ;ОЗОН`) — отсюда `SEED_ALIASES`.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path.name}: none of the encodings utf-8, cp1251 fit.")

    reader = csv.DictReader(text.splitlines(), delimiter=";")
    rows = []
    for row in reader:
        clean = {}
        for key, value in row.items():
            name = (key or "").strip().lower()
            clean[SEED_ALIASES.get(name, name)] = value or ""
        rows.append(clean)
    return rows


def seed_to_label(row: dict[str, str]) -> Label:
    """Строка сида → метка.

    Все нули сида — hard negatives: список собирался из пар-кандидатов, которые
    похожи, иначе их не с чем было бы сравнивать. Soft negatives добираются
    отдельно, случайными парами из выборки.
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
    print("\n== Field completeness ==")
    for marketplace, side in dumps.items():
        total = len(side.products)
        holes = Counter()
        for product in side.products.values():
            for name in REQUIRED_FIELDS:
                if not getattr(product, name):
                    holes[name] += 1
        print(f"\n{marketplace}: listings {total}")
        if not holes:
            print("  all fields present")
        for name in REQUIRED_FIELDS:
            missing = holes[name]
            if missing:
                mark = "ALL" if missing == total else f"{missing}/{total}"
                print(f"  missing {name:12} {mark}")


def report_values(dumps: dict[str, Dumps]) -> None:
    """Что именно разобралось.

    Полнота полей ловит пустоту, но не мусор: когда селектор скрапера съезжает
    на соседний блок, поле остаётся непустым и выглядит здоровым. Так у ВБ в
    категорию попало «Артикул / 1128813622». Поэтому — глазами на значения.
    """
    print("\n== How it parsed ==")
    for marketplace, side in dumps.items():
        if not side.products:
            continue
        categories = Counter(p.category for p in side.products.values() if p.category)
        print(f"\n{marketplace}: {len(categories)} categories over {len(side.products)} listings")
        for value, count in categories.most_common(5):
            print(f"  {count:3}  {value!r}")

        example = next(iter(side.products.values()))
        print(f"  sample listing {example.id}:")
        for name in ("title", "price_rub", "category"):
            print(f"    {name:10} {getattr(example, name)!r}")
        first_attr = next(iter(example.attributes.items()), None)
        print(f"    {'attributes':10} {len(example.attributes)} items, first {first_attr!r}")


def report_dumps(dumps: dict[str, Dumps]) -> None:
    print("\n== Dumps ==")
    for marketplace, side in dumps.items():
        print(f"{marketplace}: files {len(side.files)}, listings {len(side.products)}")
        if side.without_price:
            print(f"  rejected without price: {len(side.without_price)} — {', '.join(side.without_price)}")
        if side.duplicates:
            print(f"  one product in several files: {', '.join(sorted(set(side.duplicates)))}")
        if side.without_id:
            print(f"  without id: {', '.join(side.without_id)}")
        if side.unreadable:
            print(f"  unreadable: {side.unreadable}")


def report_coverage(buckets: dict[str, list[Label]], dumps: dict[str, Dumps], labels: list[Label]) -> None:
    titles = {
        "both": "both listings on disk",
        "wb_only": "WB only",
        "ozon_only": "Ozon only",
        "neither": "neither",
    }
    print(f"\n== Label coverage ({len(labels)} pairs) ==")
    for name, title in titles.items():
        items = buckets[name]
        matches = sum(1 for label in items if label.label == "match")
        print(f"  {title:22} {len(items):3}  matches {matches}, non-matches {len(items) - matches}")

    print("\n== Scraped outside the labels ==")
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
    print(f"\nWritten: listings {sum(len(s.products) for s in dumps.values())}, labels {len(complete)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--write", action="store_true", help="write products.jsonl and labels.jsonl")
    args = parser.parse_args()

    dumps = {marketplace: Dumps(marketplace, args.data_dir) for marketplace in MARKETPLACES}
    seed_files = sorted(args.data_dir.glob(SEED_GLOB))
    if not seed_files:
        raise SystemExit(f"no {SEED_GLOB} files in {args.data_dir}")
    print("Seed labels: " + ", ".join(path.name for path in seed_files))
    labels = [seed_to_label(row) for path in seed_files for row in load_seed(path)]
    buckets = split_by_coverage(labels, dumps)

    report_dumps(dumps)
    report_fields(dumps)
    report_values(dumps)
    report_coverage(buckets, dumps, labels)

    if args.write:
        write_dataset(dumps, buckets["both"], args.data_dir)
    else:
        print(f"\nReady to write {len(buckets['both'])} pairs. Write with --write")


if __name__ == "__main__":
    main()
