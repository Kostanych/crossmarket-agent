"""Агент: форма результата тула, разбор исходов прогона и конфигурация-запрет.

Живых вызовов LLM здесь нет — они платные и недетерминированные. Проверяется то,
что ломается молча: формат, которого ждёт SDK, разбор обрыва по лимиту и опции,
которыми исполнен запрет live-fetch.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from crossmarket import agent
from crossmarket.matching import Candidate, MatchResult, Verdict
from crossmarket.models import Product
from crossmarket.retrieval import Hit


def _result(subtype: str, **overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": subtype,
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": subtype != "success",
        "num_turns": 3,
        "session_id": "s",
        "total_cost_usd": 0.02,
        "result": "",
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def test_tool_returns_shape_sdk_expects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Расхождение здесь ломает агента без внятной ошибки."""
    hit = Hit(product=Product(marketplace="wb", id="1", title="Точилка", price_rub=920), score=0.9)
    monkeypatch.setattr(agent, "search_products", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(agent, "_client", lambda name: None)

    result = asyncio.run(agent.search_wb.handler({"question": "чем наточить нож"}))

    assert list(result) == ["content"]
    assert result["content"][0]["type"] == "text"
    assert "Точилка" in result["content"][0]["text"]


def test_tool_reports_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent, "_client", lambda name: None)

    result = asyncio.run(agent.search_wb.handler({"question": "ноутбук"}))

    assert result["content"][0]["text"] == "Ничего не найдено."


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [("error_max_turns", "max_turns"), ("error_max_budget_usd", "max_budget_usd"), ("success", None)],
)
def test_limit_outcome_is_named(subtype: str, expected: str | None) -> None:
    answer = agent.collect([], _result(subtype, errors=["предел"]), None)

    assert answer.limit_hit == expected
    assert answer.subtype == subtype


def test_exception_without_result_becomes_error() -> None:
    """Второй исход исчерпания лимита: SDK бросает после того, как отдал результат."""
    answer = agent.collect([], None, RuntimeError("оборвано"))

    assert answer.limit_hit is None
    assert answer.error == "RuntimeError: оборвано"


def test_result_and_exception_together_keep_the_limit() -> None:
    answer = agent.collect([], _result("error_max_turns", errors=["предел"]), RuntimeError("оборвано"))

    assert answer.limit_hit == "max_turns"
    assert answer.error == "предел"


def test_text_and_calls_survive_the_break() -> None:
    """При обрыве `result.result` пуст, и сказанное до обрыва берётся из сообщений."""
    messages = [
        AssistantMessage(
            content=[TextBlock(text="Ищу"), ToolUseBlock(id="1", name=agent.SEARCH_TOOL, input={"question": "нож"})],
            model="claude-opus-5",
        ),
        AssistantMessage(content=[TextBlock(text="Нашёл точилку")], model="claude-opus-5"),
    ]

    answer = agent.collect(messages, _result("error_max_turns"), None)

    assert answer.text == "Ищу\nНашёл точилку"
    assert answer.searched
    assert answer.tool_calls[0]["input"] == {"question": "нож"}


def test_tool_output_keeps_line_breaks() -> None:
    """MCP отдаёт content списком блоков; наивный str() съел бы переносы строк."""
    messages = [
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="1",
                    content=[{"type": "text", "text": "id: 42\nНазвание: Точилка"}],
                )
            ],
            parent_tool_use_id=None,
        )
    ]

    answer = agent.collect(messages, _result("success"), None)

    assert answer.tool_results == ["id: 42\nНазвание: Точилка"]


def test_options_forbid_builtin_tools() -> None:
    """Запрет live-fetch держится этими полями, а не промптом."""
    options = agent.build_options()

    assert options.tools == []
    assert options.allowed_tools == [agent.SEARCH_TOOL]
    assert options.permission_mode == "dontAsk"
    assert options.setting_sources == []
    assert options.max_turns and options.max_budget_usd


def test_thinking_is_collected_when_not_empty() -> None:
    """С `display="omitted"` блок приходит с подписью и пустым текстом — такой не нужен."""
    messages = [
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="Поищу насосы и баллончики.", signature="sig"),
                ThinkingBlock(thinking="", signature="sig"),
            ],
            model="claude-opus-5",
        )
    ]

    answer = agent.collect(messages, _result("success"), None)

    assert answer.thinking == ["Поищу насосы и баллончики."]


def test_options_ask_for_summarized_thinking() -> None:
    assert agent.build_options().thinking == {"type": "adaptive", "display": "summarized"}


def test_thinking_span_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без Langfuse запись рассуждения молчит, с ним — заводит спан с текстом."""
    agent._log_thinking("проверю насосы")  # клиента нет — не должно падать

    calls: list[dict[str, Any]] = []

    class FakeSpan:
        def end(self) -> None:
            calls.append({"ended": True})

    class FakeClient:
        def start_span(self, name: str, output: str) -> FakeSpan:
            calls.append({"name": name, "output": output})
            return FakeSpan()

    monkeypatch.setattr(agent, "_langfuse", FakeClient())
    agent._log_thinking("проверю насосы")

    assert calls == [{"name": "thinking", "output": "проверю насосы"}, {"ended": True}]


def test_sql_and_router_configs_take_the_skill_from_a_plugin() -> None:
    """Скилл приезжает плагином, а не setting-source'ами: те притащили бы `CLAUDE.md`.

    `tools=["Skill"]` — узкое исключение из запрета встроенных тулов: без него скилл
    виден в сессии, но вызвать его нечем (замер в `tools/spike_sdk.py`, проверка 6).
    """
    for options in (agent.sql_options(), agent.router_options()):
        assert options.skills == [agent.SQL_SKILL]
        assert options.plugins == [{"type": "local", "path": str(agent.PLUGIN_DIR)}]
        assert options.setting_sources == []
        assert options.tools == ["Skill"]
        assert options.permission_mode == "dontAsk"


def test_match_tool_returns_shape_sdk_expects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тул B отдаёт тот же формат, что и остальные: расхождение ломает агента молча."""

    async def fake_match(wb_id: str, *args: Any, **kwargs: Any) -> Any:
        return MatchResult(
            mode="full",
            wb=Product(marketplace="wb", id=wb_id),
            candidates=[
                Candidate(
                    product=Product(marketplace="ozon", id="42", title="Корзина", price_rub=100),
                    score=0.9,
                    verdict=Verdict(said="match", reason="то же изделие"),
                )
            ],
        )

    monkeypatch.setattr(agent, "match", fake_match)
    monkeypatch.setattr(agent, "_client", lambda name: None)

    result = asyncio.run(agent.match_ozon.handler({"wb_id": "1"}))

    assert list(result) == ["content"]
    assert result["content"][0]["type"] == "text"
    assert "42" in result["content"][0]["text"]


def test_plugin_declares_the_skill_it_promises() -> None:
    """Имя скилла собрано из манифеста: разъедься они, скилл молча выпал бы из allowlist."""
    manifest = json.loads((agent.PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    plugin_name, _, skill_name = agent.SQL_SKILL.partition(":")

    assert manifest["name"] == plugin_name
    assert (agent.PLUGIN_DIR / "skills" / skill_name / "SKILL.md").is_file()


def test_router_sees_all_tools() -> None:
    assert agent.router_options().allowed_tools == [agent.SEARCH_TOOL, agent.SQL_TOOL, agent.MATCH_TOOL]
    assert agent.build_options().allowed_tools == [agent.SEARCH_TOOL]
    assert agent.sql_options().allowed_tools == [agent.SQL_TOOL]


def test_sql_rules_are_shared_by_both_configs_that_write_sql() -> None:
    """Одни правила на C и роутер: разойдись копии, сюиты мерили бы разницу промптов."""
    assert agent.SQL_RULES in agent.SQL_SYSTEM_PROMPT
    assert agent.SQL_RULES in agent.ROUTER_SYSTEM_PROMPT
    assert agent.SQL_SKILL in agent.SQL_RULES
