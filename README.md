# crossmarket-agent

Мультитул-агент для сравнения цен между маркетплейсами (Wildberries ↔ Озон):
поиск товара по описанию, матчинг карточек между площадками, аналитика цен —
плюс eval-харнесс и observability вокруг всего этого.

## Что уже работает

- **Снапшот корпуса**, собранный полуручно браузерным скрапером: цены и
  атрибуты в ClickHouse, тексты в Qdrant, эмбеддер
  `intfloat/multilingual-e5-large` локально.
- **Tool A: Retrieval-QA** — агент на Claude Agent SDK с одним тулом поверх
  Qdrant + ClickHouse. Отвечает по снапшоту или честно говорит, что товара в
  базе нет.
- **Eval**: recall@k и MRR на golden-set, end-to-end прогон агента, метрики в
  MLflow, трейсы в Langfuse.

## Стек

Claude Agent SDK (оркестратор), Qdrant, ClickHouse, MLflow, Langfuse,
sentence-transformers, FastAPI (позже), Grafana + Prometheus (позже).

## Этапы

- [x] Фундамент: сбор данных, Qdrant и ClickHouse, голый retrieval
- [x] Tool A: Retrieval-QA и агент на Claude Agent SDK
- [ ] Eval-харнесс и каркас судей
- [ ] Tool C: text2sql к ClickHouse
- [ ] Роутинг A + C
- [ ] Tool B: матчинг ВБ ↔ Озон
- [ ] Судьи и калибровка на отложенных тест-сетах
- [ ] Observability и cost
- [ ] README и артефакты витрины

MIT.
