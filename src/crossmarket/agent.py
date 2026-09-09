"""Агент на Claude Agent SDK и его тулы: поиск по ВБ (A), матчинг с Озоном (B) и SQL (C).

Запрет live-fetch исполнен конфигурацией: `tools` не содержит `WebFetch`, `WebSearch`,
`Bash` и `Read`, `permission_mode="dontAsk"` не даёт исполниться ничему вне
`allowed_tools`. Единственное, что в `tools` попадает, — встроенный `Skill` у
конфигураций со скиллом: без него скилл обнаружен, но вызвать его нечем. В сеть он не
ходит, так что запрет держится.

Остановка детерминированная — `max_turns` и `max_budget_usd`. При исчерпании SDK и
отдаёт `ResultMessage` с `subtype=error_max_turns` / `error_max_budget_usd`, и
бросает исключение; обрабатывается и то и другое.

Тулы регистрируются в одном MCP-сервере, а разводятся по `allowed_tools`: у сюиты A
виден только `search_wb`, у C — только `execute_sql`, у роутера все три.

Знание о схеме БД подключается скиллом плагина, а не читается в промпт.
`setting_sources=[]` тут несёт смысл ограничения: с `["project"]` вместе со скиллами
приехал бы `CLAUDE.md` репозитория — 22 тысячи токенов с разбором eval, которым агент
же и оценивается (замер в CLAUDE.md). Скиллы при этом берутся из `plugins`, для
которого setting-sources не нужны.
"""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

from crossmarket.config import AGENT_MAX_BUDGET_USD, AGENT_MAX_TURNS, AGENT_MODEL, RETRIEVAL_MIN_SCORE
from crossmarket.matching import format_result as format_match
from crossmarket.matching import match
from crossmarket.retrieval import format_hits, search_products
from crossmarket.sql import format_result, run_sql
from crossmarket.storage import clickhouse, qdrant

_langfuse: Any = None
"""Клиент Langfuse, если трейсинг включён. Нужен, чтобы класть в трейс рассуждения:
инструментор их не переносит, в спанах остаются только вызовы тулов."""

_sdk_query = importlib.import_module("claude_agent_sdk.query")
"""Модуль, а не функция: инструментор OpenInference подменяет атрибут
`claude_agent_sdk.query:query`, и `from ... import query` заморозил бы ссылку на
неподменённую версию — трейсы в Langfuse оказались бы пустыми."""

SERVER_NAME = "crossmarket"
SEARCH_TOOL = f"mcp__{SERVER_NAME}__search_wb"
SQL_TOOL = f"mcp__{SERVER_NAME}__execute_sql"
MATCH_TOOL = f"mcp__{SERVER_NAME}__match_ozon"

PLUGIN_DIR = Path(__file__).resolve().parents[2]
"""Корень репозитория объявлен плагином: манифест лежит в `.claude-plugin/plugin.json`,
скиллы — в `skills/`, ровно там, где их ждёт формат. Следствие, о котором надо помнить:
`commands/`, `agents/`, `hooks/` и `.mcp.json`, заведённые в корне, тоже станут частью
плагина."""

PLUGIN_NAME = "crossmarket"
SQL_SKILL = f"{PLUGIN_NAME}:clickhouse-sql"
"""Имя из манифеста, а не из имени каталога. Без манифеста SDK взял бы имя папки, и
переименование репозитория выключило бы скилл из allowlist без единой ошибки."""


SYSTEM_PROMPT = """Ты отвечаешь на вопросы о товарах Wildberries по снапшоту базы.

Единственный источник фактов — тул search_wb. Товаров, цен и характеристик, которых
нет в его выдаче, не существует: не додумывай их и не вспоминай из общих знаний.

Если выдача пуста или найденные карточки не отвечают на вопрос — так и скажи:
такого товара в базе нет. Это нормальный ответ, а не неудача.

Ценовое ограничение из вопроса передавай параметром price_max, а не фильтруй
результат в уме.

Называя товар, ставь рядом его идентификатор из выдачи в скобках: (id: 1234567).
Цену указывай ровно ту, что в карточке, без округления.

Отвечай по-русски и сразу по делу, без преамбул вроде «сейчас поищу». Что нашлось,
цена, чем подходит. Цены — на дату снапшота, а не сегодняшние.
"""

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "Что ищем, своими словами: назначение, свойства, материал.",
        },
        "limit": {"type": "integer", "description": "Сколько карточек вернуть, по умолчанию 5."},
        "price_max": {"type": "integer", "description": "Верхняя граница цены в рублях."},
        "category": {"type": "string", "description": "Точное имя категории, если оно известно."},
    },
    "required": ["question"],
}

_clients: dict[str, Any] = {}


def _client(name: str) -> Any:
    """Соединения на процесс: агент отвечает на вопросы подряд, переподключаться незачем."""
    if name not in _clients:
        _clients[name] = qdrant.connect() if name == "qdrant" else clickhouse.connect()
    return _clients[name]


@tool(
    "search_wb",
    "Поиск товаров Wildberries по смыслу запроса в снапшоте базы. "
    "Возвращает карточки с названием, ценой, категорией и характеристиками.",
    SEARCH_SCHEMA,
)
async def search_wb(args: dict[str, Any]) -> dict[str, Any]:
    """Обёртка над `crossmarket.retrieval`: тул не знает ни про Qdrant, ни про ClickHouse.

    Поиск синхронный и упирается в видеокарту с сетью, поэтому уезжает в поток —
    иначе он заблокировал бы цикл событий, в котором SDK разговаривает с CLI.
    """
    hits = await asyncio.to_thread(
        search_products,
        args["question"],
        _client("qdrant"),
        _client("clickhouse"),
        limit=args.get("limit") or 5,
        category=args.get("category"),
        price_max=args.get("price_max"),
        min_score=RETRIEVAL_MIN_SCORE,
    )
    return {"content": [{"type": "text", "text": format_hits(hits)}]}


SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "description": "Один SELECT к таблице products. Не забудь FINAL."},
    },
    "required": ["sql"],
}


@tool(
    "execute_sql",
    "Выполнить SELECT по снапшоту товаров в ClickHouse и вернуть строки результата. "
    "Считает агрегаты, количества, сортировки и сравнения цен по обеим площадкам.",
    SQL_SCHEMA,
)
async def execute_sql(args: dict[str, Any]) -> dict[str, Any]:
    """Обёртка над `crossmarket.sql`: ограничения и разбор ошибки живут там.

    Драйвер синхронный и ходит по сети, поэтому запрос уезжает в поток — иначе он
    заблокировал бы цикл событий, в котором SDK разговаривает с CLI.
    """
    result = await asyncio.to_thread(run_sql, _client("clickhouse"), args["sql"])
    return {"content": [{"type": "text", "text": format_result(result)}]}


MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "wb_id": {"type": "string", "description": "Идентификатор товара Wildberries из выдачи search_wb."},
        "ozon_id": {
            "type": "string",
            "description": "Идентификатор карточки Ozon, если она уже названа в вопросе. "
            "Без него кандидаты ищутся по снапшоту Ozon.",
        },
    },
    "required": ["wb_id"],
}


@tool(
    "match_ozon",
    "Найти на Ozon тот же самый товар, что и карточка Wildberries, либо проверить, один ли товар в заданной паре. "
    "Возвращает только подтверждённые карточки; ничего не подтвердилось — так и говорит.",
    MATCH_SCHEMA,
)
async def match_ozon(args: dict[str, Any]) -> dict[str, Any]:
    """Обёртка над `crossmarket.matching`: поиск кандидатов и модель подтверждения там.

    Асинхронный, а не через `to_thread`: внутри параллельные вызовы модели, синхронные
    куски уезжают в потоки на своём уровне.
    """
    result = await match(
        args["wb_id"],
        _client("qdrant"),
        _client("clickhouse"),
        ozon_id=args.get("ozon_id") or None,
    )
    return {"content": [{"type": "text", "text": format_match(result)}]}


SERVER = create_sdk_mcp_server(SERVER_NAME, "1.0.0", [search_wb, execute_sql, match_ozon])

SQL_RULES = f"""Схему таблицы, ловушки этих данных и особенности диалекта ClickHouse
знает скилл {SQL_SKILL}. Вызови его перед первым SQL-запросом — сразу и молча, не объявляя
об этом в ответе: без него ты не знаешь ни про обязательный FINAL, ни про то, что lower()
не берёт кириллицу.

Единственный источник фактов — то, что вернул тул. Числа, которого нет в его выдаче, не
существует: не прикидывай и не вспоминай из общих знаний. Считать в уме поверх выдачи
тоже нельзя — если нужен ещё один агрегат, сделай ещё один запрос.

Первым запросом отвечай ровно на заданный вопрос и тем разрезом, который в нём спрошен:
спросили одно число — верни одно число, а не разбивку по площадкам. Дополнительную
детализацию, если она к месту, бери отдельным запросом уже после ответа.

Если запрос не выполнился, прочитай сообщение об ошибке, почини SQL и позови тул снова.

Пустой результат — законный ответ: так и скажи, что под условие ничего не подходит.

В ответе назови полученное число или строки явно, теми же значениями, что вернул тул,
без округления сверх того, что уже сделал SQL."""
"""Часть промпта про SQL, общая у конфигурации C и роутера: у обоих есть execute_sql и
скилл, и расхождение между двумя копиями этих правил меряло бы разницу промптов, а не
разницу конфигураций."""

SQL_SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы о товарах Wildberries и Ozon, переводя их
в SQL к снапшоту базы и вызывая тул execute_sql.

{SQL_RULES}

Отвечай по-русски и сразу по делу, без преамбул. Цены — на дату снапшота.
"""

ROUTER_SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы о товарах Wildberries и Ozon по
снапшоту базы. У тебя три тула, и первое решение по каждому вопросу — какой из них взять.

search_wb ищет карточки по смыслу запроса в коллекции Wildberries. Бери его, когда
ответ нужно доставать из текста карточки: что это за товар, для чего он, из чего сделан,
чем один подходит лучше другого. Ценовое ограничение из вопроса при этом остаётся в
search_wb параметром price_max — это сужение поиска, а не повод идти в SQL.

execute_sql считает по снапшоту обеих площадок. Бери его, когда спрошено число или
упорядоченный список по полям карточки: среднее, минимум, максимум, медиана, количество,
топ-N по цене, сравнение площадок или категорий между собой.

match_ozon ищет на Ozon тот же самый товар, что и заданная карточка Wildberries, и
проверяет пары. Бери его, когда спрошено про один конкретный товар и вторую площадку:
есть ли он на Ozon, продаётся ли то же самое там, один ли товар в названной паре.
Тулу нужен идентификатор карточки Wildberries. Если он уже назван в вопросе, передавай
его в match_ozon сразу — искать эту карточку отдельно не нужно. Если id в вопросе нет,
сначала найди товар через search_wb, потом передай найденный id в match_ozon. Названный
в вопросе идентификатор карточки Ozon передавай вторым параметром.

Развилка одной фразой: спрашивают «какой товар подойдёт» — это search_wb, спрашивают
«сколько, средняя, самый, больше ли» — это execute_sql, спрашивают «есть ли этот товар
на другой площадке, тот же ли это товар» — это match_ozon.

Считать и сравнивать по площадкам в целом — работа execute_sql, а не match_ozon: тот
отвечает про один товар, а не про выборку.

Если match_ozon не подтвердил ни одного кандидата, так и отвечай: такого товара на Ozon
в базе нет. Похожий за найденный не выдавай, даже если он очень близок.

Если вопрос честно требует нескольких тулов — сначала найди товар через search_wb, потом
зови следующий. Но не зови лишний тул просто на всякий случай.

{SQL_RULES}

По выдаче search_wb: товаров, цен и характеристик, которых в ней нет, не существует.
Если выдача пуста или найденные карточки не отвечают на вопрос — так и скажи, что такого
товара в базе нет. Это нормальный ответ, а не неудача. Называя товар, ставь рядом его
идентификатор из выдачи в скобках: (id: 1234567), а цену — ровно ту, что в карточке, без
округления.

Отвечай по-русски и сразу по делу. Не объявляй, что собираешься сделать, — ни
по-русски, ни по-английски: тул зови молча, в ответ пиши результат. Цены — на дату
снапшота, а не сегодняшние.
"""


def build_options(
    max_turns: int = AGENT_MAX_TURNS,
    max_budget_usd: float = AGENT_MAX_BUDGET_USD,
    system_prompt: str = SYSTEM_PROMPT,
    allowed_tools: list[str] | None = None,
    skills: list[str] | None = None,
) -> ClaudeAgentOptions:
    """Опции агента. `tools`, `allowed_tools` и `permission_mode` — см. докстринг модуля.

    По умолчанию — конфигурация тула A: сюиты этапов 2 и 3 зовут `build_options()` без
    аргументов. Конфигурации C и роутера передают свой промпт, свой список тулов и свой
    скилл.

    Скиллы приезжают из `plugins`, а не из `setting_sources`: второй путь притащил бы
    `CLAUDE.md` репозитория. Вместе со скиллом в `tools` кладётся `Skill` — без него
    скилл виден в сессии, но вызвать его нечем.

    `display="summarized"` возвращает текст промежуточных рассуждений: по умолчанию у
    Opus 4.7+ он `omitted`, то есть блок приходит с подписью и пустым содержимым.
    Рассуждение оплачивается в обоих режимах, разница только в видимости.
    """
    return ClaudeAgentOptions(
        thinking={"type": "adaptive", "display": "summarized"},
        model=AGENT_MODEL,
        system_prompt=system_prompt,
        tools=["Skill"] if skills else [],
        mcp_servers={SERVER_NAME: SERVER},
        allowed_tools=allowed_tools or [SEARCH_TOOL],
        permission_mode="dontAsk",
        setting_sources=[],
        skills=skills,
        plugins=[{"type": "local", "path": str(PLUGIN_DIR)}] if skills else [],
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )


def sql_options(max_turns: int = AGENT_MAX_TURNS, max_budget_usd: float = AGENT_MAX_BUDGET_USD) -> ClaudeAgentOptions:
    """Конфигурация тула C: виден только `execute_sql`, схема базы — скиллом."""
    return build_options(
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        system_prompt=SQL_SYSTEM_PROMPT,
        allowed_tools=[SQL_TOOL],
        skills=[SQL_SKILL],
    )


def router_options(
    max_turns: int = AGENT_MAX_TURNS, max_budget_usd: float = AGENT_MAX_BUDGET_USD
) -> ClaudeAgentOptions:
    """Конфигурация роутера: видны все три тула, выбор между ними делает модель."""
    return build_options(
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        system_prompt=ROUTER_SYSTEM_PROMPT,
        allowed_tools=[SEARCH_TOOL, SQL_TOOL, MATCH_TOOL],
        skills=[SQL_SKILL],
    )


@dataclass
class Answer:
    """Ответ агента вместе с тем, что нужно eval-харнессу и трейсу."""

    text: str = ""
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    subtype: str = ""
    terminal_reason: str | None = None
    num_turns: int = 0
    cost_usd: float = 0.0
    limit_hit: str | None = None
    error: str | None = None

    @property
    def searched(self) -> bool:
        return bool(self.tool_calls)


LIMIT_SUBTYPES = {"error_max_turns": "max_turns", "error_max_budget_usd": "max_budget_usd"}


def _result_text(block: ToolResultBlock) -> str:
    """Текст из результата тула.

    MCP отдаёт content списком блоков, и наивный `str()` превратил бы переносы строк
    в литералы `\\n` — выдача перестала бы разбираться построчно.
    """
    if isinstance(block.content, str):
        return block.content
    if isinstance(block.content, list):
        return "\n".join(part.get("text", "") for part in block.content if isinstance(part, dict))
    return ""


def collect(messages: list[Any], result: ResultMessage | None, error: Exception | None) -> Answer:
    """Сообщения прогона → плоский ответ.

    Текст собирается из всех блоков ассистента, а не только из `result.result`: при
    обрыве по лимиту `result` приходит пустым. Выдача тула и рассуждения забираются
    отдельно — харнессу нужно знать, что агент видел, а не только что он сказал.
    """
    answer = Answer()
    for message in messages:
        if isinstance(message, UserMessage) and isinstance(message.content, list):
            answer.tool_results += [
                _result_text(block) for block in message.content if isinstance(block, ToolResultBlock)
            ]
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking.strip():
                    answer.thinking.append(block.thinking.strip())
            elif isinstance(block, TextBlock):
                answer.text = f"{answer.text}\n{block.text}".strip()
            elif isinstance(block, ToolUseBlock):
                answer.tool_calls.append({"name": block.name, "input": block.input})

    if result is not None:
        answer.subtype = result.subtype
        answer.terminal_reason = result.terminal_reason
        answer.num_turns = result.num_turns
        answer.cost_usd = result.total_cost_usd or 0.0
        answer.limit_hit = LIMIT_SUBTYPES.get(result.subtype)
        if result.is_error:
            answer.error = "; ".join(result.errors or []) or result.subtype
    elif error is not None:
        answer.subtype = "error"
        answer.error = f"{type(error).__name__}: {error}"
    return answer


def instrument_langfuse() -> bool:
    """Включить трейсинг агента, если ключи Langfuse заданы. Повторный вызов бесплатен.

    Недоступность облака не поднимает исключение, а печатает причину и возвращает
    `False`: платный прогон от неё не должен сниматься, но и молчать нельзя — случай 1
    в `docs/failures.md`.

    Инструментор OpenInference, а не `@observe`: агент гоняет CLI дочерним процессом,
    и декоратор на наших функциях показал бы только их, оставив внутренний цикл SDK
    чёрным ящиком.
    """
    from langfuse import get_client
    from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

    global _langfuse
    try:
        client = get_client()
        if not client.auth_check():
            return False
    except Exception as exc:  # noqa: BLE001 — Langfuse в облаке, сеть отваливается
        print(f"Langfuse недоступен, прогон идёт без трейсов: {type(exc).__name__}: {exc}")
        return False
    instrumentor = ClaudeAgentSDKInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()
    _langfuse = client
    return True


def _log_thinking(text: str) -> None:
    """Резюме рассуждения — спаном внутри активного спана прогона.

    Зовётся по ходу разбора сообщений: контекст инструментора ещё активен, поэтому
    спан становится ребёнком `ClaudeAgentSDK.query`, а не отдельным трейсом.
    """
    if _langfuse is None:
        return
    _langfuse.start_span(name="thinking", output=text).end()


def flush_langfuse() -> None:
    """Дослать спаны перед выходом: экспорт батчами, и короткий скрипт успел бы завершиться раньше."""
    from langfuse import get_client

    get_client().flush()


async def ask(question: str, options: ClaudeAgentOptions | None = None) -> Answer:
    """Задать вопрос агенту и дождаться ответа.

    Исчерпание лимита SDK сообщает дважды — результатом и исключением, — поэтому
    исключение здесь не ошибка вызова, а один из штатных исходов.
    """
    options = options or build_options()
    messages: list[Any] = []
    result: ResultMessage | None = None
    error: Exception | None = None
    try:
        async for message in _sdk_query.query(prompt=question, options=options):
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ThinkingBlock) and block.thinking.strip():
                        _log_thinking(block.thinking.strip())
            if isinstance(message, ResultMessage):
                result = message
    except Exception as exc:  # noqa: BLE001 — обрыв по лимиту прилетает исключением
        error = exc
    return collect(messages, result, error)


def ask_sync(question: str, options: ClaudeAgentOptions | None = None) -> Answer:
    """Синхронный вход для скриптов: у харнесса своего цикла событий нет."""
    return asyncio.run(ask(question, options))
