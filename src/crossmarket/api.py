"""HTTP-вход агента: `/chat` — роутер с тремя тулами, `/health` — доступность баз, `/metrics` — Prometheus.

Запуск: `uvicorn crossmarket.api:app`. Метрики хранятся в памяти процесса, воркер один; с несколькими воркерами
нужен multiprocess-режим `prometheus_client`.

Стоимость в метриках — оценка SDK вместе с вложенными вызовами модели подтверждения B;
токены — только внешнего цикла.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from crossmarket import agent
from crossmarket.embedding import encode_queries

REQUESTS = Counter("chat_requests", "/chat requests by outcome: ok, limit, error.", ["outcome"])
DURATION = Histogram(
    "chat_duration_seconds",
    "Full /chat response time.",
    buckets=(2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180),
)
COST = Histogram(
    "chat_cost_usd",
    "/chat request cost, SDK estimate including nested calls.",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
)
TURNS = Histogram("chat_turns", "Agent turns per request.", buckets=(1, 2, 3, 4, 5, 6, 7, 8))
TOOL_CALLS = Counter("chat_tool_calls", "Tool calls made by the agent.", ["tool"])
TOKENS = Counter("chat_tokens", "Outer-loop tokens by kind.", ["kind"])
IN_PROGRESS = Gauge("chat_in_progress", "/chat requests in progress.")

TOKEN_KINDS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_input_tokens": "cache_read",
    "cache_creation_input_tokens": "cache_write",
}

OPTIONS = agent.router_options()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    tools: list[str]
    num_turns: int
    cost_usd: float
    limit_hit: str | None = None
    error: str | None = None


def outcome_of(answer: agent.Answer) -> str:
    """Исход запроса: ok, limit или error. Лимит проверяется первым — обрыв по лимиту тоже приходит с `error`."""
    if answer.limit_hit:
        return "limit"
    return "error" if answer.error else "ok"


def tool_name(full: str) -> str:
    return full.removeprefix(f"mcp__{agent.SERVER_NAME}__")


def record(answer: agent.Answer, seconds: float) -> None:
    REQUESTS.labels(outcome_of(answer)).inc()
    DURATION.observe(seconds)
    COST.observe(answer.cost_usd)
    TURNS.observe(answer.num_turns)
    for call in answer.tool_calls:
        TOOL_CALLS.labels(tool_name(call["name"])).inc()
    for key, kind in TOKEN_KINDS.items():
        TOKENS.labels(kind).inc(answer.usage.get(key) or 0)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Старт: трейсинг Langfuse и прогрев эмбеддера — без него загрузку весов e5 платит первый
    запрос. Выход: досылка спанов.
    """
    tracing = agent.instrument_langfuse()
    print(f"трейсинг Langfuse: {tracing}")
    encode_queries(["прогрев"])
    yield
    if tracing == "включён":
        agent.flush_langfuse()


app = FastAPI(title="crossmarket-agent", lifespan=lifespan)


@app.post("/chat")
async def chat(request: ChatRequest, response: Response) -> ChatResponse:
    start = time.perf_counter()
    with IN_PROGRESS.track_inprogress():
        answer = await agent.ask(request.question, OPTIONS)
    record(answer, time.perf_counter() - start)
    if outcome_of(answer) == "error":
        response.status_code = 502
    return ChatResponse(
        answer=answer.text,
        tools=[tool_name(call["name"]) for call in answer.tool_calls],
        num_turns=answer.num_turns,
        cost_usd=answer.cost_usd,
        limit_hit=answer.limit_hit,
        error=answer.error,
    )


@app.get("/health")
def health(response: Response) -> dict[str, str]:
    """Доступность Qdrant и ClickHouse; 503, если хоть одна база не отвечает."""
    checks = {}
    for name, probe in (
        ("qdrant", lambda: agent._client("qdrant").get_collections()),
        ("clickhouse", lambda: agent._client("clickhouse").command("SELECT 1")),
    ):
        try:
            probe()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001 — любая ошибка базы значит «недоступна»
            checks[name] = f"{type(exc).__name__}: {exc}"
    if any(state != "ok" for state in checks.values()):
        response.status_code = 503
    return checks


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
