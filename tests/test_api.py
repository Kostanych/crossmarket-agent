"""HTTP-вход: ответ /chat, коды и счётчики метрик по исходам. Агент подменён, живых вызовов LLM нет."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from crossmarket import agent, api


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _answer(**fields: Any) -> agent.Answer:
    base: dict[str, Any] = {
        "text": "Корзина (id: 1), 500 ₽",
        "tool_calls": [{"name": agent.SEARCH_TOOL, "input": {}}, {"name": agent.MATCH_TOOL, "input": {}}],
        "num_turns": 3,
        "cost_usd": 0.07,
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 900},
    }
    return agent.Answer(**(base | fields))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Тест-клиент с подменённым `ask`. Без `with`: lifespan не запускается — ни Langfuse, ни прогрева эмбеддера."""

    def use(answer: agent.Answer) -> TestClient:
        async def fake_ask(question: str, options: Any) -> agent.Answer:
            return answer

        monkeypatch.setattr(agent, "ask", fake_ask)
        return TestClient(api.app)

    return use


def test_chat_returns_answer_and_counts_it(client: Any) -> None:
    before_ok = _sample("chat_requests_total", outcome="ok")
    before_match = _sample("chat_tool_calls_total", tool="match_ozon")
    before_cost = _sample("chat_cost_usd_sum")
    before_cache = _sample("chat_tokens_total", kind="cache_read")

    response = client(_answer()).post("/chat", json={"question": "есть ли корзина на Озоне"})

    assert response.status_code == 200
    assert response.json()["tools"] == ["search_wb", "match_ozon"]
    assert _sample("chat_requests_total", outcome="ok") == before_ok + 1
    assert _sample("chat_tool_calls_total", tool="match_ozon") == before_match + 1
    assert _sample("chat_cost_usd_sum") == pytest.approx(before_cost + 0.07)
    assert _sample("chat_tokens_total", kind="cache_read") == before_cache + 900


def test_limit_is_not_an_error(client: Any) -> None:
    """Обрыв по лимиту — исход `limit` с кодом 200, хотя приходит с заполненным `error`."""
    before = _sample("chat_requests_total", outcome="limit")

    response = client(_answer(limit_hit="max_turns", error="error_max_turns")).post("/chat", json={"question": "?"})

    assert response.status_code == 200
    assert _sample("chat_requests_total", outcome="limit") == before + 1


def test_sdk_failure_is_502(client: Any) -> None:
    before = _sample("chat_requests_total", outcome="error")

    response = client(_answer(error="403 Failed to authenticate")).post("/chat", json={"question": "?"})

    assert response.status_code == 502
    assert _sample("chat_requests_total", outcome="error") == before + 1


def test_metrics_endpoint_speaks_prometheus(client: Any) -> None:
    response = client(_answer()).get("/metrics")

    assert response.status_code == 200
    assert "chat_requests_total" in response.text
