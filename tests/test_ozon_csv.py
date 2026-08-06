"""Тесты извлечения карточки Озона.

Фикстуры — настоящие выгрузки скрапера, не синтетика: смысл этих тестов
в том, чтобы ловить расхождение с реальным форматом, а придуманный CSV
проверял бы только сам себя.
"""

from pathlib import Path

import pytest

from crossmarket.extraction.ozon import (
    extract_ozon_csv,
    parse_categories,
    parse_characteristics,
    parse_price,
    parse_review_count,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("912 ₽", 912),  # разделитель разрядов — узкий пробел
        ("16 862 ₽", 16862),
        ("1 234 567 ₽", 1234567),
        ("", None),
        ("нет в наличии", None),
    ],
)
def test_parse_price(raw: str, expected: int | None) -> None:
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4.9 • 98 697 отзывов", 98697),  # 4.9 — рейтинг, он отбрасывается
        ("4.9 • 9 574 отзыва", 9574),
        ("5 • 1 отзыв", 1),
        ("", None),
    ],
)
def test_parse_review_count(raw: str, expected: int | None) -> None:
    assert parse_review_count(raw) == expected


def test_parse_categories_keeps_two_top_levels() -> None:
    raw = "Бытовая техника\nТехника для дома\nПылесосы\nРоботы-пылесосы\nDreame"
    assert parse_categories(raw) == "Бытовая техника / Техника для дома"


def test_parse_categories_tolerates_short_breadcrumbs() -> None:
    assert parse_categories("Электроника") == "Электроника"
    assert parse_categories("") == ""


def test_description_drops_only_its_own_header() -> None:
    """«Комплектация» и «Состав» — содержание, «Описание» — служебная строка."""
    header = "title,price,cats,characteristics,desc,reviews,id\n"
    # Переводы строк внутри ячейки, поэтому значение в кавычках — как в выгрузке.
    assert extract_ozon_csv(header + 't,1 ₽,Дом,,"Описание\nТекст",,1\n').description == "Текст"
    assert extract_ozon_csv(header + 't,1 ₽,Дом,,"Комплектация\n2 шт",,1\n').description == "Комплектация\n2 шт"


def test_characteristics_drop_header_and_article() -> None:
    raw = "Характеристики\nДобавить к сравнению\nАртикул\n123\nЦвет\nЖёлтый"
    assert parse_characteristics(raw) == {"Цвет": "Жёлтый"}


def test_characteristics_drop_disclaimer() -> None:
    raw = (
        "Артикул\n123\nЦвет\nЖёлтый\n"
        "Информация о технических характеристиках, комплекте поставки, стране "
        "изготовления, внешнем виде и цвете товара носит справочный характер"
    )
    assert parse_characteristics(raw) == {"Цвет": "Жёлтый"}


def test_characteristics_inline_multivalue_entries() -> None:
    """Мультизначные характеристики Озон отдаёт одной строкой через двоеточие."""
    raw = "Артикул\n1\nЦвет\nБелый\nДатчики: Датчик препятствий, Лазерный дальномер"
    assert parse_characteristics(raw) == {
        "Цвет": "Белый",
        "Датчики": "Датчик препятствий, Лазерный дальномер",
    }


def test_characteristics_colon_inside_value_does_not_desync() -> None:
    """Значение забирается вслепую, иначе двоеточие в нём сбило бы чередование."""
    raw = "Артикул\n1\nПримечание\nРежим: ночной\nЦвет\nБелый"
    assert parse_characteristics(raw) == {"Примечание": "Режим: ночной", "Цвет": "Белый"}


def test_single_row_card() -> None:
    product = extract_ozon_csv(_fixture("ozon_single.csv"), url="https://ozon.ru/product/x-3437056417/")

    assert product.marketplace == "ozon"
    assert product.id == "3437056417"
    assert product.url == "https://ozon.ru/product/x-3437056417/"
    assert product.title.startswith("Робот-пылесос Dreame F21")
    assert product.price_rub == 16862
    assert product.category == "Бытовая техника / Техника для дома"
    assert product.review_count == 9574
    assert product.attributes["Сила всасывания, Па"] == "20000"
    assert product.attributes["Датчики"] == ("Датчик препятствий, Лазерный дальномер, Датчик распознавания ковров")
    # Артикул из шапки в характеристики не попадает — он отдельное поле.
    assert "Артикул" not in product.attributes
    assert product.description


def test_multi_row_card_concatenates_description() -> None:
    product = extract_ozon_csv(_fixture("ozon_multirow.csv"))

    assert product.id == "1727694509"
    assert product.price_rub == 912
    assert product.review_count == 98697
    assert product.category == "Электроника / Аксессуары для электроники"
    assert product.attributes["Бренд"] == "GP"
    assert product.attributes["Количество в упаковке, шт"] == "20"

    # Описание собрано из строк 1..4 — в первой строке оно пустое.
    assert "Уникальные технологии" in product.description
    assert "Выгодная упаковка" in product.description


def test_empty_csv_is_rejected() -> None:
    with pytest.raises(ValueError, match="CSV пуст"):
        extract_ozon_csv("title,price,cats,characteristics,desc,reviews,id\n")
