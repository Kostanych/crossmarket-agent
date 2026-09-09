"""Матчинг ВБ↔Озон: карточка ВБ → кандидаты из озон-коллекции → модель подтверждения.

Два режима различаются только тем, откуда берётся список кандидатов; подтверждение,
сборка результата и формат общие:

- **полный** — кандидаты ищутся в озон-коллекции по тексту карточки ВБ;
- **вырожденный** — озон-карточка задана, список кандидатов состоит из неё одной.
  Обе карточки берутся из снапшота по идентификатору; товара вне снапшота для B не
  существует, и по ссылке он не выкачивается — вместо этого честный отказ.

Подтверждается каждый кандидат, раннего останова на первом матче нет. Ни один не
подтверждён — `confirmed` пуст.

Транспорт — Claude Agent SDK без единого тула. Пакета `anthropic` в зависимостях нет,
аутентификация идёт через логин CLI, как у оркестратора.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from claude_agent_sdk import ClaudeAgentOptions

from crossmarket.config import MATCH_CANDIDATE_LIMIT, MATCH_MIN_SCORE, MATCH_MODEL
from crossmarket.embedding import product_text
from crossmarket.models import Product
from crossmarket.retrieval import search_products
from crossmarket.storage import clickhouse

if TYPE_CHECKING:
    from clickhouse_connect.driver.client import Client
    from qdrant_client import QdrantClient

DESCRIPTION_LIMIT = 1200
"""До скольких символов режется описание карточки в промпте."""

CONFIRM_PROMPT = """Ты решаешь, один и тот же ли товар продаётся на двух карточках с разных
маркетплейсов — Wildberries и Ozon.

Матч — это АБСОЛЮТНО ИДЕНТИЧНЫЙ товар: то же изделие того же производителя, в том же
исполнении и той же комплектации.

Не матч — всё остальное. В том числе очень похожий товар и даже тот же самый товар, у
которого отличается хоть одна существенная характеристика.

Список отличий НЕ ЗАКРЫТ, их много. Любое из них делает пару не-матчем:
- другой цвет;
- другой размер;
- другая комплектация или другое количество штук в упаковке;
- другой производитель;
- другой состав, в том числе заметно другие проценты в составе.
Это примеры, а не полный перечень: любое содержательное отличие товара — не матч.

Отличие может быть спрятано где угодно: в названии, в описании или в характеристиках.

При этом названия, описания и характеристики площадки пишут по-разному, и полнота
карточек разная. Отличаться должен сам товар, а не формулировка и не то, что одна
площадка какое-то поле просто не заполнила.

Цена товаром не является. Между площадками она расходится сама по себе: своя наценка,
своя скидка, а ещё продавцы намеренно задирают цену, когда товар кончился, чтобы
карточка просела в выдаче до пополнения запасов. Расхождение цены, даже в разы, — не
основание для no_match; совпадение цены — не основание для match.

Ответ ровно в два поля, без преамбул и без разметки:
первая строка — одна фраза, чем обосновано
вторая строка — match или no_match, одним словом и больше ничего
"""
"""Порядок полей менять нельзя: обоснование первым, вердикт последней строкой.

С вердиктом в первой строке модель ставит его до рассуждения и на очевидно чужих
карточках опровергает себя следующей же фразой. Замер — эксперимент 12 в
`evals/results.md`."""

VERDICT = re.compile(r"^\W*(no[_\s-]?match|match)\b", re.IGNORECASE)
"""Вердикт ищется только с начала строки.

Поиск по всему тексту ловит `match` внутри обоснования («это не match, а no_match») и
записывает модели противоположный ответ."""


@dataclass
class Verdict:
    """Решение модели подтверждения по одной паре карточек."""

    said: str | None
    reason: str = ""
    cost_usd: float = 0.0

    @property
    def matched(self) -> bool:
        """Неразобранный ответ (`said is None`) подтверждением не считается."""
        return self.said == "match"


@dataclass
class Candidate:
    """Карточка Озона вместе со скором похожести и вердиктом модели.

    `score` пуст в вырожденном режиме — карточка там задана, а не найдена поиском.
    """

    product: Product
    score: float | None = None
    verdict: Verdict | None = None

    @property
    def matched(self) -> bool:
        return self.verdict is not None and self.verdict.matched


@dataclass
class MatchResult:
    """Итог матчинга: что подтверждено, что отвергнуто, чего не нашлось в снапшоте."""

    mode: str
    wb: Product | None = None
    candidates: list[Candidate] = field(default_factory=list)
    missing: str | None = None

    @property
    def confirmed(self) -> list[Candidate]:
        return [candidate for candidate in self.candidates if candidate.matched]

    @property
    def rejected(self) -> list[Candidate]:
        return [candidate for candidate in self.candidates if not candidate.matched]

    @property
    def cost_usd(self) -> float:
        return sum(c.verdict.cost_usd for c in self.candidates if c.verdict is not None)


def format_card(product: Product, marketplace: str) -> str:
    """Карточка текстом для модели подтверждения. Пустые поля выпадают сами."""
    lines = [
        f"=== {marketplace} ===",
        f"Название: {product.title}",
        f"Цена, ₽: {product.price_rub}",
        f"Категория: {product.category}",
    ]
    if product.attributes:
        lines.append("Характеристики: " + "; ".join(f"{k}: {v}" for k, v in product.attributes.items()))
    if product.description:
        lines.append(f"Описание: {product.description[:DESCRIPTION_LIMIT]}")
    return "\n".join(lines)


def parse_verdict(text: str) -> str | None:
    """Вердикт из последней непустой строки. Не разобралось — `None`, а не «match».

    Строка последняя, потому что промпт ставит вердикт после обоснования (`CONFIRM_PROMPT`).
    `match` — подстрока `no_match`, поэтому не `in`; и не поиск по всему тексту (`VERDICT`).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    found = VERDICT.match(lines[-1])
    if not found:
        return None
    normalized = found.group(1).lower().replace(" ", "_").replace("-", "_")
    return "no_match" if normalized == "no_match" else "match"


def confirm_options(model: str = MATCH_MODEL) -> ClaudeAgentOptions:
    """Опции модели подтверждения: тулов нет, настройки проекта не читаются.

    `max_turns=2`, а не 1: ход тратится и на сам ответ, при 1 прогон рвётся по лимиту.
    """
    return ClaudeAgentOptions(
        model=model,
        system_prompt=CONFIRM_PROMPT,
        tools=[],
        mcp_servers={},
        allowed_tools=[],
        permission_mode="dontAsk",
        setting_sources=[],
        max_turns=2,
        max_budget_usd=0.10,
    )


async def confirm(wb: Product, ozon: Product, options: ClaudeAgentOptions | None = None) -> Verdict:
    """Один вызов модели подтверждения по паре карточек.

    Опции передаются готовыми — строятся раз на прогон, а `confirm` зовётся на каждого
    кандидата.
    """
    from crossmarket.agent import ask

    prompt = "\n\n".join(
        [
            format_card(wb, "Wildberries"),
            format_card(ozon, "Ozon"),
            "Один и тот же это товар?",
        ]
    )
    answer = await ask(prompt, options or confirm_options())
    lines = [line.strip() for line in answer.text.splitlines() if line.strip()]
    return Verdict(
        said=parse_verdict(answer.text),
        reason=lines[0] if len(lines) > 1 else "",
        cost_usd=answer.cost_usd,
    )


def find_candidates(
    wb: Product,
    qdrant_client: QdrantClient,
    clickhouse_client: Client,
    limit: int = MATCH_CANDIDATE_LIMIT,
    min_score: float = MATCH_MIN_SCORE,
) -> list[Candidate]:
    """Похожие карточки Озона по тексту карточки ВБ.

    Тот же `search_products`, что у тула A: площадка другая, запрос — карточка целиком.
    Замер на 118 матч-парах: recall@5 = 1.000, recall@1 = 0.975.
    """
    hits = search_products(
        product_text(wb),
        qdrant_client,
        clickhouse_client,
        marketplace="ozon",
        limit=limit,
        min_score=min_score,
    )
    return [Candidate(product=hit.product, score=hit.score) for hit in hits]


async def match(
    wb_id: str,
    qdrant_client: QdrantClient,
    clickhouse_client: Client,
    ozon_id: str | None = None,
    options: ClaudeAgentOptions | None = None,
    limit: int = MATCH_CANDIDATE_LIMIT,
    min_score: float = MATCH_MIN_SCORE,
) -> MatchResult:
    """Найти карточку Озона для товара ВБ либо проверить заданную пару.

    Карточки берутся из снапшота по идентификатору. Отсутствующей в снапшоте карточке
    соответствует заполненный `missing` и пустой список кандидатов.

    Поиск кандидатов синхронный (видеокарта и сеть) и уезжает в поток; подтверждения
    идут параллельно, их до `limit` на запрос.
    """
    keys = [("wb", wb_id)] + ([("ozon", ozon_id)] if ozon_id else [])
    cards = await asyncio.to_thread(clickhouse.fetch_products, clickhouse_client, keys)

    mode = "pair" if ozon_id else "full"
    wb = cards.get(("wb", wb_id))
    if wb is None:
        return MatchResult(mode=mode, missing=f"товара Wildberries {wb_id} нет в снапшоте")

    if ozon_id:
        ozon = cards.get(("ozon", ozon_id))
        if ozon is None:
            return MatchResult(mode=mode, wb=wb, missing=f"товара Ozon {ozon_id} нет в снапшоте")
        candidates = [Candidate(product=ozon)]
    else:
        candidates = await asyncio.to_thread(find_candidates, wb, qdrant_client, clickhouse_client, limit, min_score)

    options = options or confirm_options()
    verdicts = await asyncio.gather(*(confirm(wb, candidate.product, options) for candidate in candidates))
    for candidate, verdict in zip(candidates, verdicts, strict=True):
        candidate.verdict = verdict
    return MatchResult(mode=mode, wb=wb, candidates=candidates)


def format_result(result: MatchResult) -> str:
    """Итог матчинга текстом для агента. Отвергнутые кандидаты в выдачу не попадают."""
    if result.missing:
        return f"В базе нет: {result.missing}."
    if not result.confirmed:
        checked = len(result.candidates)
        return f"Аналога на Ozon не нашлось: проверено кандидатов — {checked}, ни один не подтверждён."

    lines = []
    for candidate in result.confirmed:
        product = candidate.product
        lines.append(
            "\n".join(
                [
                    f"id: {product.id}",
                    f"Название: {product.title}",
                    f"Цена, ₽: {product.price_rub}",
                    f"Почему это тот же товар: {candidate.verdict.reason if candidate.verdict else ''}",
                ]
            )
        )
    return "\n\n".join(lines)
