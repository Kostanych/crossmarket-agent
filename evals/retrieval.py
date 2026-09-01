"""recall@k и MRR на golden-set вопросов к коллекции ВБ.

В recall идут только кейсы `expected=found`. Отрицательные всё равно ищутся: их
скоры нужны для таблицы распределения.

Порога по скору нет: в recall@k он может только выкинуть верную карточку.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from crossmarket.config import EMBEDDING_MODEL, QDRANT_COLLECTIONS
from crossmarket.distractors import load_distractors
from crossmarket.embedding import encode_queries
from crossmarket.indexing import index_products
from crossmarket.storage import qdrant
from crossmarket.storage.jsonl import load_products
from evals.cases import Case, split_summary
from evals.report import Table, print_tables, write_report

K_VALUES = (1, 5, 10)


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """Доля релевантных карточек в топ-k.

    Доля, а не факт попадания: у половины вопросов релевантных карточек несколько, и
    hit@k прятал бы разницу между «нашлась одна из трёх» и «нашлись все».
    """
    return len(set(found[:k]) & relevant) / len(relevant)


def first_hit_rank(found: list[str], relevant: set[str]) -> int | None:
    for position, product_id in enumerate(found, start=1):
        if product_id in relevant:
            return position
    return None


def evaluate(cases: list[Case], limit: int) -> list[dict[str, Any]]:
    """Прогнать все кейсы через поиск. Отрицательные — тоже, ради скоров."""
    client = qdrant.connect()
    vectors = encode_queries([case.question for case in cases])

    results = []
    for case, vector in zip(cases, vectors, strict=True):
        hits = qdrant.search(client, "wb", vector, limit=limit)
        found = [hit["id"] for hit in hits]
        relevant = set(case.relevant_ids)
        results.append(
            {
                "case": case,
                "found": found,
                "top_score": hits[0]["score"] if hits else None,
                "rank": first_hit_rank(found, relevant) if relevant else None,
                "recall": {k: recall_at_k(found, relevant, k) for k in K_VALUES} if relevant else {},
            }
        )
    return results


def metrics_of(rows: list[dict[str, Any]]) -> dict[str, float]:
    """recall@k, MRR и число ненайденных по набору кейсов с разметкой."""
    if not rows:
        return {}
    # MLflow не берёт `@` в имени метрики, поэтому recall_at_k, а не recall@k.
    metrics = {f"recall_at_{k}": sum(row["recall"][k] for row in rows) / len(rows) for k in K_VALUES}
    metrics["mrr"] = sum(1 / row["rank"] if row["rank"] else 0.0 for row in rows) / len(rows)
    metrics["not_found"] = sum(1 for row in rows if row["rank"] is None)
    return metrics


def cases_table(rows: list[dict[str, Any]]) -> Table:
    table = Table("Кейсы", ["вопрос", "сплит", "релев.", "ранг", "recall@10"])
    for row in rows:
        case = row["case"]
        table.rows.append([case.question, case.split, len(case.relevant_ids), row["rank"], row["recall"][10]])
    return table


def summary_table(rows: list[dict[str, Any]]) -> Table:
    """Метрики по всему набору и по тест-сплиту. В README идёт строка `test`:
    состав вектора подбирался на всём наборе, и цифра по нему смещена вверх."""
    table = Table("Метрики", ["набор", "кейсов", "recall@1", "recall@5", "recall@10", "MRR", "не найдено"])
    groups = [("весь набор", rows), ("test (отчётный)", [row for row in rows if row["case"].split == "test"])]
    for name, group in groups:
        metrics = metrics_of(group)
        if not metrics:
            continue
        table.rows.append(
            [
                name,
                len(group),
                metrics["recall_at_1"],
                metrics["recall_at_5"],
                metrics["recall_at_10"],
                metrics["mrr"],
                int(metrics["not_found"]),
            ]
        )
    return table


def scores_table(rows: list[dict[str, Any]]) -> Table:
    """Распределение скора топ-1 по типам кейсов: видно, разделяются ли «есть ответ»
    и «нет ответа» порогом."""
    table = Table("Скор топ-1 по типам кейсов", ["тип", "кейсов", "min", "медиана", "max"])
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row["top_score"] is not None:
            groups.setdefault(row["case"].stratum, []).append(row["top_score"])
    for name, scores in sorted(groups.items()):
        table.rows.append([name, len(scores), min(scores), median(scores), max(scores)])
    return table


def reindex(composition: str, with_distractors: bool) -> tuple[int, int]:
    """Перезалить коллекцию ВБ под этот состав текста.

    Коллекция пересоздаётся: состав живёт в векторах, а не в запросе, и без
    переиндексации метрика померила бы прошлый прогон вместе с его фоном.
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


def log_to_mlflow(rows: list[dict[str, Any]], composition: str, corpus_size: int, distractors: int) -> None:
    import mlflow

    labelled = [row for row in rows if row["recall"]]
    metrics = metrics_of(labelled)
    metrics |= {
        f"{name}_test": value
        for name, value in metrics_of([row for row in labelled if row["case"].split == "test"]).items()
    }

    mlflow.set_experiment("retrieval_wb")
    with mlflow.start_run(run_name=f"{composition}+{distractors}bg" if distractors else composition):
        mlflow.log_params(
            {
                "embedding_model": EMBEDDING_MODEL,
                "document_text": composition,
                "collection": QDRANT_COLLECTIONS["wb"],
                "corpus_size": corpus_size,
                "distractors": distractors,
                "questions": len(labelled),
            }
        )
        mlflow.log_metrics(metrics)


def comparison_table(summary: dict[str, dict[str, float]]) -> Table:
    table = Table("Сравнение составов", ["состав текста", "recall@1", "recall@5", "recall@10", "MRR", "не найдено"])
    for name in sorted(summary):
        metrics = summary[name]
        table.rows.append(
            [
                name,
                metrics["recall_at_1"],
                metrics["recall_at_5"],
                metrics["recall_at_10"],
                metrics["mrr"],
                int(metrics["not_found"]),
            ]
        )
    return table


def run(
    cases: list[Case],
    limit: int,
    compositions: list[str],
    with_distractors: bool,
    use_mlflow: bool,
) -> None:
    summary: dict[str, dict[str, float]] = {}
    tables: list[Table] = []
    corpus_size = background = 0

    for composition in compositions:
        corpus_size, background = reindex(composition, with_distractors)
        rows = evaluate(cases, limit)
        labelled = [row for row in rows if row["recall"]]

        tables = [cases_table(labelled), summary_table(labelled), scores_table(rows)]
        print_tables(tables)
        print(f"\nвопросов с разметкой {len(labelled)}, корпус {corpus_size} карточек, модель {EMBEDDING_MODEL}")

        summary[f"{composition}+bg" if background else composition] = metrics_of(labelled)
        if use_mlflow:
            log_to_mlflow(rows, composition, corpus_size, background)

    if len(summary) > 1:
        tables.append(comparison_table(summary))
        print_tables(tables[-1:])

    path = write_report(
        "retrieval_wb",
        "Retrieval: recall@k и MRR на golden-set ВБ",
        "Голый поиск без агента: вопрос уходит в эмбеддер как есть. Порога по скору нет — "
        "в recall@k он может только выкинуть верную карточку.",
        {
            "эмбеддер": EMBEDDING_MODEL,
            "состав текста": ", ".join(compositions),
            "корпус": f"{corpus_size} карточек, из них фона {background}",
            "кейсов": len(cases),
            "сплиты": split_summary(cases),
        },
        tables,
    )
    print(f"\nОтчёт: {path}" + ("" if not use_mlflow else "\nМетрики записаны в MLflow: эксперимент retrieval_wb"))
