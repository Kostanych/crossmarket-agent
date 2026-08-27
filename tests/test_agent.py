"""Агент: форма результата тула, разбор исходов прогона и конфигурация-запрет.

Живых вызовов LLM здесь нет — они платные и недетерминированные. Проверяется то,
что ломается молча: формат, которого ждёт SDK, разбор обрыва по лимиту и опции,
которыми исполнен запрет live-fetch.
"""

from __future__ import annotations

import asyncio
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
