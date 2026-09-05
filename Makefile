# Тонкая обёртка над `python -m evals`: плану нужен `make eval` как одна команда,
# из прогона которой снимается gif для витрины.

POETRY ?= poetry run

.PHONY: up eval eval-retrieval eval-qa eval-sql check test lint

up:                       ## Qdrant и ClickHouse
	docker compose --profile infra up -d

check:                    ## валидация golden-set, без LLM и без денег
	$(POETRY) python -m evals check

eval: check               ## полный прогон: retrieval + агент + text2sql
	$(POETRY) python -m evals all

eval-retrieval:           ## только recall@k и MRR
	$(POETRY) python -m evals retrieval --distractors

eval-qa:                  ## только end-to-end прогон агента (платный)
	$(POETRY) python -m evals qa

eval-sql:                 ## только text2sql к ClickHouse (платный)
	$(POETRY) python -m evals sql

test:
	$(POETRY) pytest

lint:
	$(POETRY) ruff check . && $(POETRY) ruff format --check .
