"""Адреса инфраструктуры, имена коллекций и параметры агента.

Значения по умолчанию совпадают с `docker-compose.yml`; переменные окружения их
перекрывают. При импорте подхватывается `.env` из корня — ключи Langfuse лежат
только там.
"""

from __future__ import annotations

import os
from pathlib import Path

from crossmarket.models import Marketplace


def _load_dotenv() -> None:
    """Читает `.env` в окружение процесса, не перекрывая уже заданное.

    Пустые значения пропускаются: в `.env` они означают «ключ не заполнен» и не
    должны затирать дефолт.
    """
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTIONS: dict[Marketplace, str] = {"wb": "wb_products", "ozon": "ozon_products"}

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "crossmarket")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "crossmarket")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "crossmarket")
CLICKHOUSE_TABLE = "products"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_DIM = 1024

RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0"))
"""Порог отсечки в tool A. 0.0 — порога нет: распределения скоров у настоящих ответов
и у вопросов вне корпуса пересекаются, замер в `evals/results.md`. Параметр оставлен,
на большем корпусе разделяющая точка может появиться."""

AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-5")
AGENT_MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "6"))
AGENT_MAX_BUDGET_USD = float(os.getenv("AGENT_MAX_BUDGET_USD", "0.5"))
