"""Голый retrieval: recall@k и MRR на golden-set вопросов к коллекции ВБ.

Критерий «сделано» для этапа 1. Метрика прямая, судьи здесь нет и не нужно:
у каждого вопроса есть размеченные релевантные карточки.

Запуск:
    docker compose --profile infra up -d
    poetry run python tools/load.py
    poetry run python tools/eval_retrieval.py

Печатает построчный разбор и итоговую таблицу в stdout, метрики пишет в
MLflow (локальный бэкенд в `mlruns/`). Вывод в stdout — требование плана: из
него потом делается gif для витрины, а молчаливый прогон показывать нечего.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossmarket.config import EMBEDDING_MODEL, QDRANT_COLLECTIONS
from crossmarket.distractors import load_distractors
from crossmarket.embedding import COMPOSITIONS, DEFAULT_COMPOSITION, encode_queries
from crossmarket.indexing import index_products
from crossmarket.storage import qdrant
from crossmarket.storage.jsonl import load_products

GOLDEN_FILE = Path("evals/golden/retrieval_wb.jsonl")
K_VALUES = (1, 5, 10)


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """Доля релевантных карточек, попавших в топ-k.

    Именно доля, а не факт попадания: у половины вопросов релевантных карточек
    несколько (три точилки для ножей, два измерителя почвы), и hit@k прятал бы
    разницу между «нашлась одна из трёх» и «нашлись все».
    """
    return len(set(found[:k]) & relevant) / len(relevant)


def first_hit_rank(found: list[str], relevant: set[str]) -> int | None:
    for position, product_id in enumerate(found, start=1):
        if product_id in relevant:
            return position
    return None


def evaluate(cases: list[dict], limit: int) -> list[dict]:
    client = qdrant.connect()
    vectors = encode_queries([case["question"] for case in cases])

    results = []
    for case, vector in zip(cases, vectors, strict=True):
        hits = qdrant.search(client, "wb", vector, limit=limit)
        found = [hit["id"] for hit in hits]
        relevant = set(case["relevant_ids"])
        results.append(
            {
                "question": case["question"],
                "relevant": relevant,
                "found": found,
                "rank": first_hit_rank(found, relevant),
                "recall": {k: recall_at_k(found, relevant, k) for k in K_VALUES},
            }
        )
    return results


def print_cases(results: list[dict]) -> None:
    print(f"\n{'вопрос':<52} {'релев.':>6} {'ранг':>5} {'recall@10':>10}")
    print("-" * 76)
    for row in results:
        rank = row["rank"] if row["rank"] is not None else "—"
        print(f"{row['question'][:52]:<52} {len(row['relevant']):>6} {str(rank):>5} {row['recall'][10]:>10.2f}")


def print_summary(results: list[dict], corpus_size: int) -> dict[str, float]:
    # MLflow не берёт `@` в имени метрики, поэтому recall_at_k, а не recall@k.
    metrics = {f"recall_at_{k}": sum(row["recall"][k] for row in results) / len(results) for k in K_VALUES}
    metrics["mrr"] = sum(1 / row["rank"] if row["rank"] else 0.0 for row in results) / len(results)
    metrics["not_found"] = sum(1 for row in results if row["rank"] is None)

    print(f"\n{'итог':<20} {'значение':>10}")
    print("-" * 31)
    for name, value in metrics.items():
        print(f"{name:<20} {value:>10.3f}")
    print(f"\nвопросов {len(results)}, корпус {corpus_size} карточек, модель {EMBEDDING_MODEL}")
    return metrics


def log_to_mlflow(metrics: dict[str, float], composition: str, cases: int, corpus_size: int, distractors: int) -> None:
    import mlflow

    run_name = f"{composition}+{distractors}bg" if distractors else composition
    mlflow.set_experiment("retrieval_wb")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "embedding_model": EMBEDDING_MODEL,
                "document_text": composition,
                "collection": QDRANT_COLLECTIONS["wb"],
                "corpus_size": corpus_size,
                "distractors": distractors,
                "questions": cases,
            }
        )
        mlflow.log_metrics(metrics)


def reindex(composition: str, with_distractors: bool) -> tuple[int, int]:
    """Перезалить коллекцию ВБ под этот состав текста.

    Коллекция каждый раз пересоздаётся: иначе дистракторы от прошлого прогона
    остались бы в корпусе и прогон «без фона» мерил бы не то, что заявлено.
    Без переиндексации метрика мерила бы состав текста прошлого прогона —
    состав живёт в векторах, а не в запросе.
    """
    real = [product for product in load_products().values() if product.marketplace == "wb"]
    client = qdrant.connect()
    qdrant.drop_collection(client, "wb")

    print(f"\n=== {composition}{', с фоном' if with_distractors else ''}: индексирую {len(real)} карточек ВБ…")
    index_products(client, "wb", real, composition)

    background = load_distractors() if with_distractors else []
    if background:
        print(f"    плюс {len(background)} дистракторов")
        index_products(client, "wb", background, composition, synthetic=True)
    return len(real) + len(background), len(background)


def print_comparison(summary: dict[str, dict[str, float]]) -> None:
    names = sorted(summary)
    metric_names = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "not_found"]
    width = max(len(name) for name in names) + 2

    print(f"\n{'состав текста':<{width}}" + "".join(f"{name:>13}" for name in metric_names))
    print("-" * (width + 13 * len(metric_names)))
    for name in names:
        print(f"{name:<{width}}" + "".join(f"{summary[name][metric]:>13.3f}" for metric in metric_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    parser.add_argument("--limit", type=int, default=max(K_VALUES), help="глубина выдачи")
    parser.add_argument(
        "--text",
        nargs="+",
        choices=sorted(COMPOSITIONS),
        default=[DEFAULT_COMPOSITION],
        help="составы текста; каждый переиндексируется и логируется отдельным прогоном",
    )
    parser.add_argument("--distractors", action="store_true", help="добить корпус придуманным фоном")
    parser.add_argument("--no-mlflow", action="store_true", help="только stdout")
    args = parser.parse_args()

    cases = load_golden(args.golden)
    summary: dict[str, dict[str, float]] = {}

    for composition in args.text:
        corpus_size, background = reindex(composition, args.distractors)
        results = evaluate(cases, args.limit)
        print_cases(results)
        metrics = print_summary(results, corpus_size)
        summary[f"{composition}+bg" if background else composition] = metrics
        if not args.no_mlflow:
            log_to_mlflow(metrics, composition, len(cases), corpus_size, background)

    if len(summary) > 1:
        print_comparison(summary)
    if not args.no_mlflow:
        print("\nМетрики записаны в MLflow: mlruns/")


main()
