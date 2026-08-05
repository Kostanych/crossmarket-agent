"""Стримлит-разметчик пар ВБ↔Озон.

Инструмент сбора данных, не часть агента: в docker-compose не входит, в витрину
не идёт. Таймбокс из плана — 1–2 вечера.

Запуск:  poetry run streamlit run tools/labeler/app.py

Цикл работы: снять обе карточки браузерным скрапером → загрузить сюда два CSV
→ «Извлечь поля» → проверить и поправить → поставить метку → «Сохранить пару».
"""

from __future__ import annotations

import streamlit as st

from crossmarket.extraction.ozon import extract_ozon_csv
from crossmarket.extraction.urls import parse_ozon_id, parse_wb_id
from crossmarket.extraction.wb import extract_wb_csv
from crossmarket.models import Label, Product
from crossmarket.storage.jsonl import (
    DATA_DIR,
    append_label,
    append_product,
    load_labels,
    load_products,
    save_raw,
)

SIDES = {"wb": "Wildberries", "ozon": "Ozon"}
EXTRACTORS = {"wb": extract_wb_csv, "ozon": extract_ozon_csv}
ID_PARSERS = {"wb": parse_wb_id, "ozon": parse_ozon_id}

FIELD_DEFAULTS: dict[str, object] = {
    "url": "",
    "id": "",
    "title": "",
    "description": "",
    "price": 0,
    "category": "",
    "brand": "",
    "attrs": "",
    "reviews": 0,
}


def _key(side: str, field: str) -> str:
    return f"{side}__{field}"


def _init_state() -> None:
    for side in SIDES:
        for field, default in FIELD_DEFAULTS.items():
            st.session_state.setdefault(_key(side, field), default)
        st.session_state.setdefault(f"{side}__raw", "")
    st.session_state.setdefault("label", "match")
    st.session_state.setdefault("negative_kind", "hard")


def _clear_form() -> None:
    for side in SIDES:
        for field, default in FIELD_DEFAULTS.items():
            st.session_state[_key(side, field)] = default
        st.session_state[f"{side}__raw"] = ""


def _attrs_to_text(attrs: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in attrs.items())


def _text_to_attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        name = name.strip()
        if name:
            attrs[name] = value.strip()
    return attrs


def _uploaded_text(side: str) -> str:
    """Содержимое загруженного CSV. Пустая строка, если файла нет."""
    uploaded = st.session_state.get(f"{side}__csv")
    if uploaded is None:
        return ""
    return uploaded.getvalue().decode("utf-8-sig")


def _apply_extracted(side: str, product: Product) -> None:
    st.session_state[_key(side, "id")] = product.id
    st.session_state[_key(side, "title")] = product.title
    st.session_state[_key(side, "description")] = product.description
    st.session_state[_key(side, "price")] = product.price_rub or 0
    st.session_state[_key(side, "category")] = product.category
    st.session_state[_key(side, "brand")] = product.brand
    st.session_state[_key(side, "attrs")] = _attrs_to_text(product.attributes)
    st.session_state[_key(side, "reviews")] = product.review_count or 0


def _id_mismatch(side: str) -> tuple[str, str] | None:
    """Идентификаторы из CSV и из ссылки, если они разошлись.

    Ссылка вставляется ради провенанса, но заодно это второй, независимый
    источник идентификатора. Расхождение означает, что CSV одного товара
    приехал вместе со ссылкой другого — за двести пар копипасты случай
    неизбежный, а постфактум в датасете почти неразличимый.
    """
    from_csv = st.session_state[_key(side, "id")].strip()
    from_url = ID_PARSERS[side](st.session_state[_key(side, "url")])
    if from_csv and from_url and from_csv != from_url:
        return from_csv, from_url
    return None


def _handle_extract(side: str) -> None:
    """Кнопка «Извлечь поля». Идентификатор берётся из CSV, а из ссылки — как запасной."""
    raw = _uploaded_text(side)
    st.session_state[f"{side}__raw"] = raw

    if raw:
        try:
            _apply_extracted(side, EXTRACTORS[side](raw, url=st.session_state[_key(side, "url")]))
        except NotImplementedError as exc:
            st.info(f"{SIDES[side]}: {exc} Заполни поля руками — форма редактируемая.")
        except (ValueError, KeyError) as exc:
            st.error(f"{SIDES[side]}: не разобрал CSV — {exc}")

    # Ссылка не обязательна, но если идентификатор из CSV не пришёл, выручит она.
    if not st.session_state[_key(side, "id")]:
        url = st.session_state[_key(side, "url")]
        product_id = ID_PARSERS[side](url)
        if product_id:
            st.session_state[_key(side, "id")] = product_id
        elif url.strip():
            st.warning(f"{SIDES[side]}: не разобрал идентификатор ни из CSV, ни из ссылки.")

    mismatch = _id_mismatch(side)
    if mismatch:
        st.warning(
            f"{SIDES[side]}: идентификатор из CSV ({mismatch[0]}) не совпал со ссылкой ({mismatch[1]}). "
            "Не перепутаны ли файлы?"
        )


def _render_side(side: str) -> None:
    st.subheader(SIDES[side])
    st.file_uploader("CSV скрапера", type="csv", key=f"{side}__csv")
    st.text_input(
        "Ссылка на карточку",
        key=_key(side, "url"),
        help="Нужна для провенанса. Идентификатор берётся из CSV, ссылка — запасной источник.",
    )
    st.button("Извлечь поля", key=f"extract_{side}", on_click=_handle_extract, args=(side,), width="stretch")

    st.divider()
    st.text_input("Идентификатор", key=_key(side, "id"))
    st.text_input("Название", key=_key(side, "title"))
    st.text_area("Описание", key=_key(side, "description"), height=100)
    st.number_input("Цена, ₽", min_value=0, step=1, key=_key(side, "price"))
    st.text_input("Категория", key=_key(side, "category"))
    st.text_input("Бренд", key=_key(side, "brand"))
    st.text_area(
        "Характеристики",
        key=_key(side, "attrs"),
        height=140,
        help="По одной в строке, в формате «ключ: значение».",
    )
    st.number_input("Отзывов", min_value=0, step=1, key=_key(side, "reviews"), help="0 — данных нет. Поле опционально.")


def _collect_product(side: str) -> Product:
    return Product(
        marketplace=side,  # type: ignore[arg-type]
        id=st.session_state[_key(side, "id")].strip(),
        url=st.session_state[_key(side, "url")].strip(),
        title=st.session_state[_key(side, "title")].strip(),
        description=st.session_state[_key(side, "description")].strip(),
        price_rub=st.session_state[_key(side, "price")] or None,
        category=st.session_state[_key(side, "category")].strip(),
        brand=st.session_state[_key(side, "brand")].strip(),
        attributes=_text_to_attrs(st.session_state[_key(side, "attrs")]),
        review_count=st.session_state[_key(side, "reviews")] or None,
    )


def _save_pair() -> None:
    wb, ozon = _collect_product("wb"), _collect_product("ozon")

    missing = [SIDES[s] for s, p in (("wb", wb), ("ozon", ozon)) if not p.id]
    if missing:
        st.error(f"Нет идентификатора: {', '.join(missing)}. Пара не сохранена.")
        return

    # Сверка, которая не предупреждает, а пропускает, датасет не защищает.
    for side in SIDES:
        mismatch = _id_mismatch(side)
        if mismatch:
            st.error(
                f"{SIDES[side]}: идентификатор из CSV ({mismatch[0]}) не совпал со ссылкой ({mismatch[1]}). "
                "Пара не сохранена — проверь, тот ли CSV загружен."
            )
            return

    label_value = st.session_state["label"]
    label = Label(
        wb_id=wb.id,
        ozon_id=ozon.id,
        label=label_value,  # type: ignore[arg-type]
        negative_kind=(st.session_state["negative_kind"] if label_value == "no_match" else None),  # type: ignore[arg-type]
    )

    for side, product in (("wb", wb), ("ozon", ozon)):
        raw = st.session_state.get(f"{side}__raw", "")
        if raw:
            save_raw(side, product.id, raw)  # type: ignore[arg-type]
        append_product(product)
    append_label(label)

    st.success(f"Сохранено: {wb.id} ↔ {ozon.id} — {label_value}")
    _clear_form()


def _render_sidebar() -> None:
    labels = load_labels()
    matches = sum(1 for lb in labels.values() if lb.label == "match")
    st.sidebar.metric("Пар размечено", len(labels))
    st.sidebar.metric("Матчей", matches)
    st.sidebar.metric("Не-матчей", len(labels) - matches)
    st.sidebar.metric("Карточек собрано", len(load_products()))
    st.sidebar.caption(f"Данные: `{DATA_DIR.resolve()}`")


def main() -> None:
    st.set_page_config(page_title="Разметчик ВБ↔Озон", layout="wide")
    _init_state()

    st.title("Разметчик пар ВБ↔Озон")
    st.caption("Сбор полуручной: карточки снимаются скрапером с открытых страниц, в сеть разметчик не ходит.")

    _render_sidebar()

    left, right = st.columns(2)
    with left:
        _render_side("wb")
    with right:
        _render_side("ozon")

    st.divider()
    st.radio(
        "Метка",
        options=["match", "no_match"],
        format_func=lambda v: "Матч" if v == "match" else "Не матч",
        key="label",
        horizontal=True,
    )
    if st.session_state["label"] == "no_match":
        st.radio(
            "Тип негатива",
            options=["hard", "easy"],
            format_func=lambda v: "Hard — близкий товар" if v == "hard" else "Easy — далёкий товар",
            key="negative_kind",
            horizontal=True,
            help="Нужен для стратификации тест-сета пар: на случайных негативах метрика B надувается.",
        )

    save, clear = st.columns([3, 1])
    save.button("Сохранить пару", key="save_pair", type="primary", on_click=_save_pair, width="stretch")
    clear.button("Очистить", key="clear_form", on_click=_clear_form, width="stretch")


main()
