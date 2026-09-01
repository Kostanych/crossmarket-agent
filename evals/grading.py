"""Разбор итогового ответа агента: что он назвал, откуда взял цены, отказал ли.

Метрика считается по итоговому ответу, а не по выдаче тула: агент видит до тридцати
карточек, и `shown_recall` по ней упирается в 1.00.

Карточка опознаётся по идентификатору — системный промпт агента требует его
указывать. Всё детерминированно, судьи нет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from evals.cases import Case

_ID = re.compile(r"(?:id[:\s]*)?\b(\d{9,12})\b")
"""Идентификаторы карточек — 9–12 цифр. Цены короче, так что не пересекаются."""

_PRICE = re.compile(r"(\d[\d\s  ]*)\s*(?:₽|руб)", re.IGNORECASE)
"""Ценой считается число рядом с ₽ или «руб»: иначе в проверку попали бы габариты и
количества. Разряды разделены обычным, неразрывным или узким пробелом."""

_ANY_NUMBER = re.compile(r"(\d[\d\s  ]*)")
"""Числа из вопроса исключаются из проверки цен: повтор ценового ограничения —
не выдумка."""

_SOURCE = r"баз[аеиуы]|снапшот[ае]?|выдач[аеиу]"
_ABSENT = r"\bнет\b|\bно не\b|не найден|не нашл|не обнаруж|отсутству"

_DENIAL = re.compile(
    rf"(?:{_SOURCE})[^.!?]*?(?:{_ABSENT})"
    rf"|(?:{_ABSENT})[^.!?]*?(?:{_SOURCE})"
    r"|нет (?:ни одного|подходящ|такого товара)"
    r"|ничего(?: подходящего)? не (?:наш|найден)",
    re.IGNORECASE,
)
"""Явный отказ: агент сказал, что искомого в базе нет.

Ищется связка «источник + отрицание» внутри предложения, в любом порядке: «в базе
нет спреев», «масляные краски не найдены», «в выдаче нет средств транспорта».
Расстояние между ними не ограничено — между источником и отрицанием вклинивается
перечисление: «в базе Wildberries боксёрского оборудования (груш, мешков, лап) не
найдено».

Отказом не считаются отказ по домену («я отвечаю только на вопросы о товарах»),
уточняющий вопрос вместо ответа и пустой ответ: во всех трёх агент не утверждал,
что товара нет, а в тул мог и не сходить.

Проверка по словам теряет перефразированный отказ, поэтому ответы на отрицательные
кейсы печатаются в отчёт целиком. Оценка свободного текста — судья-QA этапа 7;
история поломок этой проверки — `docs/failures.md`."""

_TOOL_ID = re.compile(r"^id: (\S+)$")
_TOOL_PRICE = re.compile(r"^Цена, ₽: (\d+)$")
_TOOL_TITLE = re.compile(r"^Название: (.*)$")


@dataclass
class Shown:
    """Карточка в том виде, в каком её увидела модель."""

    id: str
    title: str = ""
    price_rub: int | None = None


def parse_tool_output(texts: list[str]) -> dict[str, Shown]:
    """Карточки из выдачи тула — то, что модель могла назвать.

    Разбирается тот же текст, что ушёл в модель. Блок начинается строкой `id:`;
    по пустым строкам не режем — описание карточки их содержит.
    """
    shown: dict[str, Shown] = {}
    current: Shown | None = None
    for text in texts:
        for line in text.splitlines():
            if match := _TOOL_ID.match(line):
                current = shown.setdefault(match.group(1), Shown(id=match.group(1)))
            elif current is None:
                continue
            elif match := _TOOL_PRICE.match(line):
                current.price_rub = int(match.group(1))
            elif match := _TOOL_TITLE.match(line):
                current.title = match.group(1)
    return shown


def _numbers(pattern: re.Pattern[str], text: str) -> list[int]:
    return [int(re.sub(r"[\s  ]", "", match)) for match in pattern.findall(text)]


@dataclass
class Grade:
    """Разбор одного ответа."""

    cited_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    bad_prices: list[int] = field(default_factory=list)
    cited_recall: float = 0.0
    cited_extra: int = 0
    refused: bool = True
    denied: bool = False
    outcome_correct: bool = False

    @property
    def faithful(self) -> bool:
        return not self.unknown_ids and not self.bad_prices


def grade(case: Case, answer_text: str, tool_results: list[str]) -> Grade:
    """Оценить ответ против истины кейса.

    Верный исход на отрицательном кейсе — явный отказ: предлагать альтернативу
    можно, выдавать её за искомое нельзя.

    `cited_extra` в вердикт не идёт: агент может назвать разумную карточку, не
    попавшую в разметку, и штраф за это мерил бы полноту golden-set.
    """
    shown = parse_tool_output(tool_results)
    mentioned = list(dict.fromkeys(_ID.findall(answer_text)))
    relevant = set(case.relevant_ids)

    cited = [product_id for product_id in mentioned if product_id in shown]
    unknown = [product_id for product_id in mentioned if product_id not in shown]

    known_prices = {card.price_rub for card in shown.values() if card.price_rub is not None}
    asked = set(_numbers(_ANY_NUMBER, case.question))
    bad = [price for price in _numbers(_PRICE, answer_text) if price not in known_prices and price not in asked]

    hit = set(cited) & relevant
    result = Grade(
        cited_ids=cited,
        unknown_ids=unknown,
        bad_prices=bad,
        cited_recall=len(hit) / len(relevant) if relevant else 0.0,
        cited_extra=len(cited) - len(hit),
        refused=not cited,
        denied=bool(_DENIAL.search(answer_text)),
    )
    result.outcome_correct = result.denied if case.expected == "absent" else bool(hit)
    return result
