"""End-to-end прогон golden-set через агента — критерий «сделано» для этапа 2.

Метрики прямые, судьи здесь нет и не нужно: у каждого вопроса размечены
релевантные карточки, а свободный ответ на этом этапе оценивается только на
«непустой» — faithfulness и судья-QA идут этапами 3 и 7.

Отличие от `tools/eval_retrieval.py`: там вопрос уходил в эмбеддер как есть, здесь
запрос к retrieval формулирует оркестратор, и выдача проходит через порог. Поэтому
цифры двух харнессов сравнивать напрямую нельзя, и общий у них только golden-set.

Запуск:
    docker compose --profile infra up -d
    poetry run python tools/load.py --distractors
    poetry run python tools/eval_qa.py

Печатает построчный разбор и итоговую таблицу в stdout, метрики пишет в MLflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from crossmarket.agent import Answer, ask, build_options, flush_langfuse, instrument_langfuse
from crossmarket.config import AGENT_MAX_BUDGET_USD, AGENT_MAX_TURNS, AGENT_MODEL, RETRIEVAL_MIN_SCORE

GOLDEN_FILE = Path("evals/golden/retrieval_wb.jsonl")
ID_LINE = re.compile(r"^id: (\S+)$", re.MULTILINE)


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def shown_ids(answer: Answer) -> list[str]:
    """Идентификаторы карточек, которые агент увидел в выдаче тула.

    Разбор текста, а не отдельный канал: тул отдаёт модели ровно эту строку, и мерить
    надо то же, что видела модель.
    """
    return [product_id for text in answer.tool_results for product_id in ID_LINE.findall(text)]


def evaluate_case(case: dict, answer: Answer) -> dict:
    relevant = set(case["relevant_ids"])
    shown = shown_ids(answer)
    return {
        "question": case["question"],
        "relevant": relevant,
        "shown": shown,
        "recall": len(set(shown) & relevant) / len(relevant),
        "searched": answer.searched,
        "answered": bool(answer.text.strip()),
        "subtype": answer.subtype,
        "limit_hit": answer.limit_hit,
        "queries": [call["input"].get("question") for call in answer.tool_calls],
        "thinking": answer.thinking,
        "turns": answer.num_turns,
        "cost": answer.cost_usd,
        "text": answer.text,
        "error": answer.error,
    }


async def run(cases: list[dict], limit_turns: int, budget: float, verbose: bool) -> list[dict]:
    options = build_options(max_turns=limit_turns, max_budget_usd=budget)
    results = []
    for number, case in enumerate(cases, start=1):
        answer = await ask(case["question"], options)
        row = evaluate_case(case, answer)
        results.append(row)
        mark = "+" if row["recall"] else ("тул не звал" if not row["searched"] else "—")
        print(f"{number:>3}/{len(cases)} {mark:<12} {row['question'][:60]}")
        if row["limit_hit"]:
            print(f"        оборвано лимитом: {row['limit_hit']}")
        if row["error"]:
            print(f"        ошибка: {row['error'][:160]}")
        if verbose:
            for step in row["thinking"]:
                print(f"        думает: {step[:160]}")
            for query in row["queries"]:
                print(f"        ищет:   {query}")
            print(f"        ответ:  {row['text'][:300]}")
    return results


def print_cases(results: list[dict]) -> None:
    print(f"\n{'вопрос':<52} {'релев.':>6} {'recall':>7} {'ходов':>6} {'$':>7}")
    print("-" * 82)
    for row in results:
        print(
            f"{row['question'][:52]:<52} {len(row['relevant']):>6} "
            f"{row['recall']:>7.2f} {row['turns']:>6} {row['cost']:>7.4f}"
        )


def print_summary(results: list[dict]) -> dict[str, float]:
    count = len(results)
    metrics = {
        "tool_called": sum(row["searched"] for row in results) / count,
        "hit_rate": sum(1 for row in results if row["recall"]) / count,
        "recall": sum(row["recall"] for row in results) / count,
        "answered": sum(row["answered"] for row in results) / count,
        "limit_hit": sum(1 for row in results if row["limit_hit"]) / count,
        "errors": sum(1 for row in results if row["error"]) / count,
        "avg_turns": sum(row["turns"] for row in results) / count,
        "avg_cost_usd": sum(row["cost"] for row in results) / count,
        "total_cost_usd": sum(row["cost"] for row in results),
    }

    print(f"\n{'итог':<20} {'значение':>10}")
    print("-" * 31)
    for name, value in metrics.items():
        print(f"{name:<20} {value:>10.4f}")

    subtypes: dict[str, int] = {}
    for row in results:
        subtypes[row["subtype"] or "—"] = subtypes.get(row["subtype"] or "—", 0) + 1
    print(f"\nисходы: {subtypes}")
    print(f"вопросов {count}, модель {AGENT_MODEL}, порог {RETRIEVAL_MIN_SCORE}")
    return metrics


def log_to_mlflow(metrics: dict[str, float], cases: int, limit_turns: int, budget: float) -> None:
    import mlflow

    mlflow.set_experiment("qa_wb")
    with mlflow.start_run(run_name=AGENT_MODEL):
        mlflow.log_params(
            {
                "model": AGENT_MODEL,
                "min_score": RETRIEVAL_MIN_SCORE,
                "max_turns": limit_turns,
                "max_budget_usd": budget,
                "questions": cases,
            }
        )
        mlflow.log_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    parser.add_argument("--first", type=int, help="прогнать только первые N вопросов")
    parser.add_argument("--max-turns", type=int, default=AGENT_MAX_TURNS)
    parser.add_argument("--budget", type=float, default=AGENT_MAX_BUDGET_USD)
    parser.add_argument("--verbose", action="store_true", help="печатать ответы целиком")
    parser.add_argument("--no-mlflow", action="store_true", help="только stdout")
    args = parser.parse_args()

    cases = load_golden(args.golden)[: args.first]
    print(f"вопросов {len(cases)}, трейсинг Langfuse: {'включён' if instrument_langfuse() else 'нет ключей'}\n")

    results = asyncio.run(run(cases, args.max_turns, args.budget, args.verbose))
    flush_langfuse()
    print_cases(results)
    metrics = print_summary(results)

    if not args.no_mlflow:
        log_to_mlflow(metrics, len(cases), args.max_turns, args.budget)
        print("\nМетрики записаны в MLflow: mlruns/")


main()
