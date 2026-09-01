"""Проверка golden-set без вызовов LLM.

Нужна там, где вопросы дописываются руками: отрицательный кейс годится, только если
в корпусе действительно нет ответа.
"""

from __future__ import annotations

from pathlib import Path

from crossmarket.embedding import encode_queries
from crossmarket.models import Product
from crossmarket.storage import qdrant
from crossmarket.storage.jsonl import load_products
from evals.cases import assign_splits, load_cases, save_cases, split_sizes, validate

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


def run(path: Path, assign: bool = False, categories: bool = False) -> int:
    """Ноль — файл в порядке. Ненулевой код возврата ломает `make eval` до трат."""
    cases = load_cases(path)
    print(f"Кейсов {len(cases)}, из них с ответом в корпусе {sum(c.expected == 'found' for c in cases)}")

    if assign and (changed := assign_splits(cases)):
        save_cases(cases, path)
        print(f"Проставлен сплит: {', '.join(f'{case.id}→{case.split}' for case in changed)}")

    problems = validate(cases)
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
