"""Тесты извлечения карточки Wildberries.

Фикстуры — настоящие выгрузки скрапера: придуманный CSV проверял бы только
сам себя, а смысл в том, чтобы ловить расхождение с реальным форматом.
"""

from pathlib import Path

import pytest

from crossmarket.extraction.wb import (
    extract_wb_csv,
    parse_categories,
    parse_characteristics,
    parse_description,
    parse_price,
    parse_review_count,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Две строки: с кошельком ВБ и без. Берётся первая.
        ("871 ₽\n889 ₽", 871),
        ("1 668 ₽\n1 703 ₽", 1668),
        ("871 ₽", 871),
        ("", None),
    ],
)
def test_parse_price(raw: str, expected: int | None) -> None:
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Разделитель `·`, не `•`; вторая строка — вопросы, а не отзывы.
        ("4,9 · 18 491 оценка\n3 660 вопросов", 18491),
        ("5 · 1 оценка", 1),
        ("", None),
    ],
)
def test_parse_review_count(raw: str, expected: int | None) -> None:
    assert parse_review_count(raw) == expected


def test_parse_categories_drops_breadcrumb_root() -> None:
    raw = "Главная\nДом\nСпальня\nПостельные принадлежности\nНаматрасники\nStretch Jersy"
    assert parse_categories(raw) == "Дом / Спальня"


def test_parse_description_drops_header() -> None:
    assert parse_description("Описание\n\nТекст карточки") == "Текст карточки"
    assert parse_description("Текст без заголовка") == "Текст без заголовка"


def test_characteristics_skip_section_headers() -> None:
    """Заголовок секции идёт сразу после значения, без пустых строк между."""
    raw = "Основная информация\nЦвет\nбелый\nМатериалы\nМатериал изделия\nполиэстер"
    assert parse_characteristics(raw) == {"Цвет": "белый", "Материал изделия": "полиэстер"}


def test_characteristics_blank_lines_separate_pairs() -> None:
    """Две пустые строки означают, что дальше ключ, а не заголовок секции."""
    raw = "Основная информация\nЦвет\nбелый\n\n\nЖесткость\nмягкий"
    assert parse_characteristics(raw) == {"Цвет": "белый", "Жесткость": "мягкий"}


def test_matrasnik_card() -> None:
    product = extract_wb_csv(
        _fixture("wb_matrasnik.csv"), url="https://www.wildberries.ru/catalog/18326211/detail.aspx"
    )

    assert product.marketplace == "wb"
    assert product.id == "18326211"
    assert product.title == "Наматрасник непромокаемый на резинке 180х200 см"
    assert product.price_rub == 871
    assert product.category == "Дом / Спальня"
    assert product.brand == "Stretch Jersy"
    assert product.review_count == 18491
    assert product.description.startswith("Наматрасник 180×200 непромокаемый")

    # Заголовки секций в характеристики не попадают.
    assert "Основная информация" not in product.attributes
    assert "Материалы" not in product.attributes
    assert "Габариты" not in product.attributes
    assert product.attributes["Цвет"] == "белый"
    assert product.attributes["Материал чехла"] == "мулетон; полиэстер; мембрана"
    assert product.attributes["Страна производства"] == "Россия"


def test_shkatulka_card() -> None:
    product = extract_wb_csv(_fixture("wb_shkatulka.csv"))

    assert product.id == "681857332"
    assert product.title == "Металлическая шкатулка ящик для денег МВ4"
    assert product.price_rub == 1668
    assert product.category == "Дом / Хранение вещей"
    assert product.brand == "KlestO"
    assert product.review_count == 8
    assert product.description.startswith("Кэшбокс Klesto")
    assert product.attributes["Цвет"] == "черный матовый"
    assert product.attributes["Количество отделений"] == "1 шт."
    assert product.attributes["Вид замка"] == "ключевой"
    assert "Особенности корпуса" not in product.attributes


def test_only_first_row_is_used() -> None:
    """Со второй строки скрапер сваливает в колонку `id` остатки характеристик."""
    product = extract_wb_csv(_fixture("wb_matrasnik.csv"))

    assert product.id.isdigit()
    assert "мембрана" not in product.id


def test_empty_csv_is_rejected() -> None:
    with pytest.raises(ValueError, match="CSV пуст"):
        extract_wb_csv("title,price,cats,charasteristics,desc,reviews,id,brand\n")
