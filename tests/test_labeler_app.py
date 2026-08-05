"""Смоук-тесты разметчика через AppTest.

Ловят то, что не видят ни линтер, ни импорты: ошибки рендера и несовпадение
с API стримлита. Загрузку файлов AppTest симулировать не умеет, поэтому разбор
CSV проверяется отдельно в `test_ozon_csv.py`, а здесь — только каркас.

Ни в одном тесте сохранение не доходит до записи — пара либо без
идентификаторов, либо отбита сверкой, — поэтому в `data/` ничего не пишется.
"""

from streamlit.testing.v1 import AppTest

APP = "tools/labeler/app.py"


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_renders_without_exception() -> None:
    at = _run()
    assert not at.exception


def test_both_marketplaces_are_present() -> None:
    at = _run()
    assert [element.value for element in at.subheader] == ["Wildberries", "Ozon"]


def test_csv_uploader_rendered_for_each_side() -> None:
    at = _run()
    assert len(at.file_uploader) == 2


def test_extract_without_csv_falls_back_to_url() -> None:
    """Ссылка — запасной источник идентификатора, когда CSV не загружен."""
    at = _run()
    at.text_input(key="wb__url").set_value("https://www.wildberries.ru/catalog/123456789/detail.aspx")
    at.button(key="extract_wb").click().run()

    assert not at.exception
    assert at.text_input(key="wb__id").value == "123456789"


def test_extract_warns_when_nothing_to_parse() -> None:
    at = _run()
    at.text_input(key="ozon__url").set_value("https://www.ozon.ru/brands/dreame/")
    at.button(key="extract_ozon").click().run()

    assert not at.exception
    assert any("не разобрал идентификатор" in warning.value for warning in at.warning)


def test_extract_warns_when_csv_and_url_disagree() -> None:
    """CSV одного товара со ссылкой другого — самая дорогая ошибка копипасты."""
    at = _run()
    at.text_input(key="wb__id").set_value("18326211")
    at.text_input(key="wb__url").set_value("https://www.wildberries.ru/catalog/999999999/detail.aspx")
    at.button(key="extract_wb").click().run()

    assert not at.exception
    assert any("не совпал со ссылкой" in warning.value for warning in at.warning)
    # Значение из CSV остаётся, правку выбирает человек.
    assert at.text_input(key="wb__id").value == "18326211"


def test_save_with_mismatched_id_is_refused() -> None:
    at = _run()
    at.text_input(key="wb__id").set_value("18326211")
    at.text_input(key="wb__url").set_value("https://www.wildberries.ru/catalog/999999999/detail.aspx")
    at.text_input(key="ozon__id").set_value("1727694509")
    at.button(key="save_pair").click().run()

    assert not at.exception
    assert any("Пара не сохранена" in error.value for error in at.error)


def test_save_without_ids_is_refused() -> None:
    at = _run()
    at.button(key="save_pair").click().run()

    assert not at.exception
    assert any("Пара не сохранена" in error.value for error in at.error)


def test_negative_kind_appears_only_for_no_match() -> None:
    """Тип негатива нужен для стратификации тест-сета и спрашивается по делу."""
    at = _run()
    assert len(at.radio) == 1

    at.radio(key="label").set_value("no_match").run()
    assert not at.exception
    assert len(at.radio) == 2
