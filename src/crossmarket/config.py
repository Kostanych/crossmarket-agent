"""Адреса инфраструктуры и имена коллекций.

Значения по умолчанию совпадают с `docker-compose.yml`, поэтому при локальном
запуске задавать ничего не нужно. Переменные окружения перекрывают их — они
понадобятся, когда агент поедет в контейнере и `localhost` перестанет быть
адресом баз.
"""

from __future__ import annotations

import os

from crossmarket.models import Marketplace

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
