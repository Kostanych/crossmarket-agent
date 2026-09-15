# Цели поверх `python -m evals`, docker compose и uvicorn.

POETRY ?= poetry run

.PHONY: up observe serve eval eval-retrieval eval-qa eval-sql eval-routing eval-matching judges check test lint

up:                       ## Qdrant и ClickHouse
	docker compose --profile infra up -d

observe:                  ## + Prometheus (:9090) и Grafana (:3000)
	docker compose --profile infra --profile observability up -d

serve:                    ## /chat, /health, /metrics на :8000; один воркер — метрики в памяти процесса
	$(POETRY) uvicorn crossmarket.api:app --host 0.0.0.0 --port 8000

check:                    ## валидация golden-set, без LLM и без денег
	$(POETRY) python -m evals check

eval: check               ## полный прогон: retrieval + агент + text2sql + роутинг + матчинг
	$(POETRY) python -m evals all

eval-retrieval:           ## только recall@k и MRR
	$(POETRY) python -m evals retrieval --distractors

eval-qa:                  ## только end-to-end прогон агента (платный)
	$(POETRY) python -m evals qa

eval-sql:                 ## только text2sql к ClickHouse (платный)
	$(POETRY) python -m evals sql

eval-routing:             ## только выбор тула на однохоповых вопросах (платный)
	$(POETRY) python -m evals routing

eval-matching:            ## только матчинг ВБ↔Озон на размеченных парах (платный)
	$(POETRY) python -m evals matching

judges:                   ## согласие трёх судей с истиной на тест-сплитах (платный)
	$(POETRY) python -m evals judge --of qa
	$(POETRY) python -m evals judge --of b
	$(POETRY) python -m evals judge --of c

test:
	$(POETRY) pytest

lint:
	$(POETRY) ruff check . && $(POETRY) ruff format --check .
