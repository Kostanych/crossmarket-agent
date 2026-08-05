"""Смоук-тесты разметчика через AppTest.

Ловят то, что не видят ни линтер, ни импорты: ошибки рендера и несовпадение
с API стримлита. Поиск выгрузки на диске проверяется отдельно в
`test_scraped.py`, разбор CSV — в `test_ozon_csv.py` и `test_wb_csv.py`,
здесь только каркас и реакция на ввод.

Ни в одном тесте сохранение не доходит до записи — пара либо без
идентификаторов, либо отбита сверкой, — поэтому в `data/` ничего не пишется.
"""

from streamlit.testing.v1 import AppTest

APP = "tools/labeler/app.py"

WB_URL = "https://www.wildberries.ru/catalog/123456789/detail.aspx"

# Разряды цены разделены узким пробелом, как на самих площадках. Через chr(),
# чтобы в исходнике теста не лежал невидимый символ.
NARROW_SPACE = chr(0x2009)


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_renders_without_exception() -> None:
    at = _run()
    assert not at.exception


def test_both_marketplaces_are_present() -> None:
    at = _run()
    assert [element.value for element in at.subheader] == ["Wildberries", "Ozon"]


def test_url_fills_id_without_extra_click() -> None:
    """Идентификатор подтягивается прямо при вставке ссылки."""
    at = _run()
    at.text_input(key="wb__url").set_value(WB_URL).run()

    assert not at.exception
    assert at.text_input(key="wb__id").value == "123456789"


def test_missing_dump_is_reported() -> None:
    """Ссылка есть, выгрузки в папке нет — это надо сказать, а не молчать."""
    at = _run()
    at.text_input(key="wb__url").set_value(WB_URL).run()

    assert not at.exception
    assert any("выгрузки 123456789 нет" in warning.value for warning in at.warning)


def test_warns_when_url_is_not_a_product_link() -> None:
    at = _run()
    at.text_input(key="ozon__url").set_value("https://www.ozon.ru/brands/dreame/").run()

    assert not at.exception
    assert any("не разобрал идентификатор" in warning.value for warning in at.warning)


def test_preview_shows_title_and_price() -> None:
    """Название и цена под ссылкой — та самая проверка глазами перед меткой."""
    at = _run()
    at.text_input(key="wb__title").set_value("Наматрасник непромокаемый").run()
    at.number_input(key="wb__price").set_value(16862).run()

    assert not at.exception
    assert any("Наматрасник непромокаемый" in md.value for md in at.markdown)
    assert any(f"16{NARROW_SPACE}862" in caption.value for caption in at.caption)


def test_save_without_ids_is_refused() -> None:
    at = _run()
    at.button(key="save_pair").click().run()

    assert not at.exception
    assert any("Пара не сохранена" in error.value for error in at.error)


def test_save_with_hand_edited_id_is_refused() -> None:
    """Единственный способ разойтись со ссылкой — поправить поле руками."""
    at = _run()
    at.text_input(key="wb__url").set_value(WB_URL).run()
    at.text_input(key="wb__id").set_value("18326211").run()
    at.text_input(key="ozon__id").set_value("1727694509").run()
    at.button(key="save_pair").click().run()

    assert not at.exception
    assert any("не совпал со ссылкой" in error.value for error in at.error)


def test_negative_kind_appears_only_for_no_match() -> None:
    """Тип негатива нужен для стратификации тест-сета и спрашивается по делу."""
    at = _run()
    assert len(at.radio) == 1

    at.radio(key="label").set_value("no_match").run()
    assert not at.exception
    assert len(at.radio) == 2
