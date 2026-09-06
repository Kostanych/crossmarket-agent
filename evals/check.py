"""Проверка golden-set без вызовов LLM.

Нужна там, где вопросы дописываются руками: отрицательный кейс годится, только если
в корпусе действительно нет ответа.
"""

from __future__ import annotations

from pathlib import Path

from crossmarket.embedding import encode_queries
from crossmarket.models import Product
from crossmarket.sql import run_sql
from crossmarket.storage import clickhouse, qdrant
from crossmarket.storage.jsonl import load_products
from evals import pairs as pairs_module
from evals.cases import (
    assign_splits,
    load_cases,
    load_routing_cases,
    load_sql_cases,
    save_cases,
    save_routing_cases,
    save_sql_cases,
    split_sizes,
    validate,
    validate_routing_cases,
    validate_sql_cases,
)

NEIGHBOURS = 5
"""Сколько настоящих карточек показать по отрицательному вопросу."""

NEIGHBOUR_DEPTH = 40
"""Глубина выдачи, из которой они выбираются: дистракторов вчетверо больше настоящих
карточек, и настоящий сосед легко оказывается ниже топ-10."""


def _corpus() -> dict[str, Product]:
    """Карточки ВБ снапшота по идентификатору: источник для проверки `relevant_ids`."""
    return {product.id: product for product in load_products().values() if product.marketplace == "wb"}


def _categories(corpus: dict[str, Product]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for product in corpus.values():
        counts[product.category] = counts.get(product.category, 0) + 1
    return dict(sorted(counts.items()))


def check_pairs(assign: bool = False) -> list[str]:
    """Сплит пар ВБ↔Озон: раздать, если просят, и напечатать страты."""
    pairs = pairs_module.load_pairs()
    if assign and (changed := pairs_module.assign(pairs)):
        print(f"Парам проставлен сплит: {len(changed)}")
    print(f"\n{pairs_module.summary(pairs)}")
    for stratum, sizes in pairs_module.strata(pairs).items():
        print(f"  {stratum:<16} {sizes}")
    return pairs_module.validate(pairs)


def check_sql_cases(assign: bool = False) -> list[str]:
    """Golden-set тула C: схема кейсов плюс исполнимость каждого эталонного SQL.

    Эталон хранится запросом, а не строками, поэтому проверять его надо прогоном:
    переименованная категория превратила бы кейс в вечный промах молча. Пустой
    результат допустим только в страте `empty` — и обязателен в ней.
    """
    cases = load_sql_cases()
    if assign and (changed := assign_splits(cases)):
        save_sql_cases(cases)
        print(f"SQL-кейсам проставлен сплит: {', '.join(f'{c.id}→{c.split}' for c in changed)}")

    problems = validate_sql_cases(cases)
    client = clickhouse.connect()
    strata: dict[str, int] = {}
    for case in cases:
        strata[case.kind] = strata.get(case.kind, 0) + 1
        result = run_sql(client, case.sql)
        if not result.ok:
            problems.append(f"{case.id}: эталонный SQL не исполняется — {result.error[:120]}")
        elif case.kind == "empty" and result.total_rows:
            problems.append(f"{case.id}: страта empty, а строк {result.total_rows}")
        elif case.kind != "empty" and not result.total_rows:
            problems.append(f"{case.id}: пустой результат вне страты empty")

    print(f"\nSQL-кейсов {len(cases)}: {split_sizes(cases)}")
    print(f"Страты: {dict(sorted(strata.items()))}")
    return problems


def check_routing_cases(assign: bool = False) -> list[str]:
    """Golden-set роутинга: схема кейсов и страты. Базы тут не нужны — лейбл проверяется
    глазами при написании вопроса, а не запросом."""
    cases = load_routing_cases()
    if assign and (changed := assign_splits(cases)):
        save_routing_cases(cases)
        print(f"Кейсам роутинга проставлен сплит: {', '.join(f'{c.id}→{c.split}' for c in changed)}")

    strata: dict[str, int] = {}
    for case in cases:
        strata[case.kind] = strata.get(case.kind, 0) + 1
    single = sum(1 for case in cases if case.single_hop)
    print(f"\nКейсов роутинга {len(cases)}, из них в accuracy {single}: {split_sizes(cases)}")
    print(f"Страты: {dict(sorted(strata.items()))}")
    return validate_routing_cases(cases)


def run(path: Path, assign: bool = False, categories: bool = False, pairs: bool = False) -> int:
    """Ноль — файл в порядке. Ненулевой код возврата ломает `make eval` до трат."""
    cases = load_cases(path)
    print(f"Кейсов {len(cases)}, из них с ответом в корпусе {sum(c.expected == 'found' for c in cases)}")

    if assign and (changed := assign_splits(cases)):
        save_cases(cases, path)
        print(f"Проставлен сплит: {', '.join(f'{case.id}→{case.split}' for case in changed)}")

    problems = validate(cases)
    problems += check_sql_cases(assign)
    problems += check_routing_cases(assign)
    if pairs:
        problems += check_pairs(assign)
    corpus = _corpus()
    for case in cases:
        for product_id in case.relevant_ids:
            if product_id not in corpus:
                problems.append(f"{case.id}: карточки {product_id} нет в снапшоте")

    strata: dict[str, int] = {}
    for case in cases:
        strata[case.stratum] = strata.get(case.stratum, 0) + 1
    print(f"Сплиты: {split_sizes(cases)}")
    print(f"Страты: {dict(sorted(strata.items()))}")

    if categories:
        print("\nКатегории корпуса ВБ (в них ответ есть — отрицательный вопрос должен быть про другое):")
        for name, count in _categories(corpus).items():
            print(f"  {count:>3}  {name}")

    absent = [case for case in cases if case.expected == "absent"]
    if absent:
        print("\nОтрицательные кейсы: ближайшие настоящие карточки — проверить глазами, что ответа нет")
        print("Дистракторы пропущены: они придуманы и ответом быть не могут, а места в выдаче занимают.")
        client = qdrant.connect()
        vectors = encode_queries([case.question for case in absent])
        for case, vector in zip(absent, vectors, strict=True):
            print(f"\n  {case.id} [{case.kind}] {case.question}")
            hits = qdrant.search(client, "wb", vector, limit=NEIGHBOUR_DEPTH)
            real = [hit for hit in hits if hit["id"] in corpus][:NEIGHBOURS]
            if not real:
                print(f"      ничего настоящего в топ-{NEIGHBOUR_DEPTH}")
            for hit in real:
                card = corpus[hit["id"]]
                print(f"      {hit['score']:.3f}  {card.title[:60]:<60} [{card.category}]")

    if problems:
        print("\nПроблемы:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nGolden-set в порядке.")
    return 0
