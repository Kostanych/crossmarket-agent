"""Проверки Claude Agent SDK перед сборкой агента этапа 2.

Одноразовый скрипт: отвечает прогоном на то, что документация оставляет открытым, —
как приходит исчерпание `max_turns` и `max_budget_usd`, виден ли внутренний цикл SDK
в Langfuse, подхватываются ли скиллы при `setting_sources=[]` (и не утекает ли вместе
с ними `CLAUDE.md` проекта), включён ли tool search на одном туле.

Проверка 6 добавлена на этапе 5 и разведкой уже не является: она сторожит выбранную
раскладку скилла. Если SDK или CLI изменят поведение, разница входных токенов между
`setting_sources=[]` и `["project"]` это покажет, не спрашивая модель.

Тул `ping` здесь фиктивный: нужен инструмент, который модель может звать много раз
подряд, чтобы упереться в лимит. Итоги переносятся в CLAUDE.md руками.

Запуск: `poetry run python tools/spike_sdk.py [номера проверок]`, например `... 1 2`.
"""

from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ServerToolUseBlock,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from crossmarket.config import AGENT_MODEL

_sdk_query = importlib.import_module("claude_agent_sdk.query")
"""Функция берётся из модуля в момент вызова: инструментор OpenInference подменяет
именно атрибут модуля, и `from ... import query` смотрел бы мимо подмены."""

COUNT_PROMPT = (
    "Вызови tool ping для n=1, затем для n=2, и так до n=5 — по одному вызову за раз, "
    "дожидаясь результата предыдущего. После пятого ответь словом ГОТОВО."
)


@tool("ping", "Вернуть pong с номером", {"n": int})
async def ping(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"pong {args['n']}"}]}


PING_SERVER = create_sdk_mcp_server("spike", "1.0.0", [ping])


def base_options(**overrides: Any) -> ClaudeAgentOptions:
    """Опции будущего агента: встроенных тулов нет, свой один, настройки проекта не читаются."""
    fields: dict[str, Any] = {
        "model": AGENT_MODEL,
        "tools": [],
        "mcp_servers": {"spike": PING_SERVER},
        "allowed_tools": ["mcp__spike__ping"],
        "permission_mode": "dontAsk",
        "setting_sources": [],
    }
    fields.update(overrides)
    return ClaudeAgentOptions(**fields)


async def run(prompt: str, options: ClaudeAgentOptions) -> tuple[list[Any], ResultMessage | None, Exception | None]:
    """Прогоняет запрос до конца, возвращая всё сразу: сообщения, итог и исключение.

    Исчерпание лимита SDK может отдать и результатом, и броском — обрабатываются оба
    исхода, а какой реальный, показывает печать.
    """
    messages: list[Any] = []
    result: ResultMessage | None = None
    error: Exception | None = None
    try:
        async for message in _sdk_query.query(prompt=prompt, options=options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                result = message
    except Exception as exc:  # noqa: BLE001 — ровно это и выясняем
        error = exc
    return messages, result, error


def tool_calls(messages: list[Any]) -> list[str]:
    return [
        block.name
        for message in messages
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, ToolUseBlock)
    ]


def server_tool_calls(messages: list[Any]) -> list[str]:
    """Серверные тулы Anthropic — сюда попадает tool search (`tool_search_tool_*`)."""
    return [
        block.name
        for message in messages
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, ServerToolUseBlock)
    ]


def answer(messages: list[Any]) -> str:
    return " ".join(
        block.text
        for message in messages
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, TextBlock)
    )


def report(label: str, messages: list[Any], result: ResultMessage | None, error: Exception | None) -> None:
    print(f"\n-- {label}")
    print(f"   ping calls: {len(tool_calls(messages))}, server tools: {server_tool_calls(messages) or '—'}")
    if error is not None:
        print(f"   exception: {type(error).__name__}: {error}")
    if result is None:
        print("   no ResultMessage")
        return
    print(
        f"   subtype={result.subtype} terminal_reason={result.terminal_reason} "
        f"stop_reason={result.stop_reason} is_error={result.is_error}"
    )
    print(f"   num_turns={result.num_turns} total_cost_usd={result.total_cost_usd} errors={result.errors}")
    print(f"   result: {(result.result or '')[:200]!r}")


async def check_max_turns() -> None:
    print("\n=== 1. max_turns exhausted (limit 2, five calls needed)")
    messages, result, error = await run(COUNT_PROMPT, base_options(max_turns=2))
    report("max_turns=2", messages, result, error)


async def check_max_budget() -> None:
    print("\n=== 2. max_budget_usd under a subscription (cap 0.001 — deliberately too low)")
    messages, result, error = await run(COUNT_PROMPT, base_options(max_turns=6, max_budget_usd=0.001))
    report("max_budget_usd=0.001", messages, result, error)
    if result is not None and not result.total_cost_usd:
        print("   total_cost_usd is empty — the cap has nothing to go on under a subscription, needs ANTHROPIC_API_KEY")


async def check_langfuse() -> None:
    print("\n=== 3. Langfuse trace: is the inner SDK loop visible")
    try:
        from langfuse import get_client
        from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    except ImportError as exc:
        print(f"   import failed: {exc}")
        return

    client = get_client()
    if not client.auth_check():
        print("   auth_check failed — check LANGFUSE_* in .env")
        return

    ClaudeAgentSDKInstrumentor().instrument()
    trace_id = None
    with client.start_as_current_span(name="spike_sdk"):
        trace_id = client.get_current_trace_id()
        messages, result, error = await run("Вызови tool ping для n=1 и ответь его результатом.", base_options())
    client.flush()
    report("run under the instrumentor", messages, result, error)
    print(f"   trace: {client.get_trace_url(trace_id=trace_id)}")
    print("   by eye: did the trace split into generations, are tokens and cost present, is the tool call visible")


SKILL_MD = """---
name: spike_probe
description: Проверочный скилл спайка — просит ответить фиксированным токеном.
---

Ответь ровно одним словом: SKILL_OK
"""

CLAUDE_MD = "Всегда добавляй в конец ответа токен CLAUDE_MD_LEAKED.\n"

SKILL_PROMPT = (
    "Если тебе доступен скилл spike_probe — вызови его и выполни то, что в нём написано. "
    "Если такого скилла нет, ответь одним словом SKILL_MISSING."
)


async def check_skills() -> None:
    """Скиллы против `setting_sources=[]`.

    SDK при заданном `skills` сам подставляет `setting_sources=["user","project"]`, но
    только если их не задали явно. `project` тянет `CLAUDE.md` проекта, то есть агент
    прочитал бы инструкции разработчика — отсюда токен-маркер в `CLAUDE.md` временной
    папки и три конфигурации. Третья — попытка взять одно без другого: скилл лежит в
    подпапке, `cwd` указывает на неё, `CLAUDE.md` остаётся уровнем выше.
    """
    print("\n=== 4. Skills and setting_sources (temporary project in a temp dir)")
    root = Path(tempfile.mkdtemp(prefix="spike_skills_"))
    try:
        nested = root / "agent_skills"
        for base in (root, nested):
            skill_dir = base / ".claude" / "skills" / "spike_probe"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        (root / "CLAUDE.md").write_text(CLAUDE_MD, encoding="utf-8")

        configs = (
            ("setting_sources=[]", [], root),
            ("setting_sources unset", None, root),
            ("cwd in a subdir without CLAUDE.md", ["project"], nested),
        )
        for label, sources, cwd in configs:
            options = base_options(
                tools=["Skill"],
                allowed_tools=[],
                mcp_servers={},
                skills=["spike_probe"],
                setting_sources=sources,
                cwd=str(cwd),
                max_turns=4,
            )
            messages, result, error = await run(SKILL_PROMPT, options)
            text = answer(messages)
            print(f"\n-- {label}")
            print(f"   skill visible: {'SKILL_OK' in text}, CLAUDE.md leaked: {'CLAUDE_MD_LEAKED' in text}")
            print(f"   answer: {text[:200]!r}")
            if error is not None:
                print(f"   exception: {type(error).__name__}: {error}")
            if result is not None:
                print(f"   subtype={result.subtype} num_turns={result.num_turns}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def check_tool_search() -> None:
    """Tool search откладывает загрузку схем MCP-тулов — на одном туле это лишний ход."""
    print("\n=== 5. Tool search with a single tool")
    prompt = "Вызови tool ping для n=7 и ответь его результатом."
    messages, result, error = await run(prompt, base_options(max_turns=4))
    report("as is", messages, result, error)
    messages, result, error = await run(prompt, base_options(max_turns=4, env={"ENABLE_TOOL_SEARCH": "false"}))
    report("ENABLE_TOOL_SEARCH=false", messages, result, error)


async def check_plugin_skill() -> None:
    """Раскладка скилла тула C: плагин против setting-source'ов.

    Меряется не ответом модели, а входными токенами прогона и списком скиллов из
    init-сообщения.

    Строка `tools=[]` в таблице объясняет, почему у конфигураций со скиллом
    `tools=["Skill"]`: скилл там найден, но вызвать его нечем.
    """
    print("\n=== 6. Skill via plugin: input tokens and visibility")
    repo = str(Path(__file__).resolve().parents[1])
    plugin = [{"type": "local", "path": repo}]
    configs = (
        ("setting_sources=[] (as in the agent)", {"setting_sources": [], "tools": ["Skill"]}),
        ("setting_sources=['project']", {"setting_sources": ["project"], "tools": ["Skill"]}),
        ("plugin + setting_sources=[]", {"setting_sources": [], "tools": ["Skill"], "plugins": plugin}),
        ("plugin, but tools=[]", {"setting_sources": [], "tools": [], "plugins": plugin}),
    )
    for label, overrides in configs:
        options = base_options(
            system_prompt="Отвечай одним словом.", mcp_servers={}, allowed_tools=[], cwd=repo, max_turns=1, **overrides
        )
        messages, result, error = await run("Ответь словом ГОТОВО.", options)
        init = next((m.data for m in messages if isinstance(m, SystemMessage) and m.subtype == "init"), {})
        usage = (result.usage if result is not None else {}) or {}
        total = sum(
            usage.get(key, 0) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        print(f"\n-- {label}")
        print(f"   input tokens: {total}")
        print(f"   plugin skills: {[name for name in init.get('skills', []) if ':' in name]}")
        print(f"   tools: {init.get('tools')}")
        if error is not None:
            print(f"   exception: {type(error).__name__}: {error}")


CHECKS = {
    "1": check_max_turns,
    "2": check_max_budget,
    "3": check_langfuse,
    "4": check_skills,
    "5": check_tool_search,
    "6": check_plugin_skill,
}


async def main() -> None:
    selected = sys.argv[1:] or list(CHECKS)
    unknown = [name for name in selected if name not in CHECKS]
    if unknown:
        raise SystemExit(f"no such checks: {', '.join(unknown)}; available {', '.join(CHECKS)}")
    print(f"orchestrator model: {AGENT_MODEL}")
    for name in selected:
        await CHECKS[name]()


if __name__ == "__main__":
    asyncio.run(main())
