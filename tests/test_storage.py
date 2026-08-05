import json
from pathlib import Path

from crossmarket.models import Label, Product
from crossmarket.storage.jsonl import (
    append_label,
    append_product,
    load_labels,
    load_products,
    save_raw,
)


def _product(**overrides) -> Product:
    defaults = {"marketplace": "wb", "id": "123", "title": "Наушники"}
    return Product(**{**defaults, **overrides})


def test_product_roundtrip(tmp_path: Path) -> None:
    product = _product(price_rub=9990, attributes={"Бренд": "Nothing"})
    append_product(product, data_dir=tmp_path)

    loaded = load_products(data_dir=tmp_path)[("wb", "123")]
    assert loaded.title == "Наушники"
    assert loaded.price_rub == 9990
    assert loaded.attributes == {"Бренд": "Nothing"}


def test_missing_files_are_empty(tmp_path: Path) -> None:
    assert load_products(data_dir=tmp_path) == {}
    assert load_labels(data_dir=tmp_path) == {}


def test_product_last_record_wins(tmp_path: Path) -> None:
    append_product(_product(price_rub=9990), data_dir=tmp_path)
    append_product(_product(price_rub=8990), data_dir=tmp_path)

    products = load_products(data_dir=tmp_path)
    assert len(products) == 1
    assert products[("wb", "123")].price_rub == 8990
    # Файл append-only: обе строки на месте, история правок сохраняется.
    assert len((tmp_path / "products.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2


def test_products_of_different_marketplaces_do_not_collide(tmp_path: Path) -> None:
    append_product(_product(marketplace="wb", id="1"), data_dir=tmp_path)
    append_product(_product(marketplace="ozon", id="1"), data_dir=tmp_path)

    assert set(load_products(data_dir=tmp_path)) == {("wb", "1"), ("ozon", "1")}


def test_label_last_record_wins(tmp_path: Path) -> None:
    append_label(Label(wb_id="1", ozon_id="2", label="match"), data_dir=tmp_path)
    append_label(
        Label(wb_id="1", ozon_id="2", label="no_match", negative_kind="hard"),
        data_dir=tmp_path,
    )

    labels = load_labels(data_dir=tmp_path)
    assert len(labels) == 1
    assert labels[("1", "2")].label == "no_match"
    assert labels[("1", "2")].negative_kind == "hard"


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    """JSONL хранит историю: старые строки могут не совпадать с текущей схемой."""
    path = tmp_path / "products.jsonl"
    path.write_text(
        json.dumps({"marketplace": "wb", "id": "7", "legacy_field": "х"}) + "\n",
        encoding="utf-8",
    )

    assert load_products(data_dir=tmp_path)[("wb", "7")].id == "7"


def test_save_raw(tmp_path: Path) -> None:
    path = save_raw("ozon", "123", "title,price\nЯщик,900\n", data_dir=tmp_path)

    assert path == tmp_path / "raw" / "ozon" / "123.csv"
    assert path.read_text(encoding="utf-8") == "title,price\nЯщик,900\n"


def test_save_raw_sanitises_id(tmp_path: Path) -> None:
    """Идентификатор попадает в имя файла — выход за каталог недопустим."""
    path = save_raw("wb", "../../evil", "", data_dir=tmp_path)

    assert path.parent == tmp_path / "raw" / "wb"
    assert path.name == "______evil.csv"
