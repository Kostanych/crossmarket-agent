"""End-to-end прогон golden-set через агента: что дошло до итогового ответа.

Запрос к тулу формулирует оркестратор и зовёт его по нескольку раз, поэтому цифры
несравнимы с `evals/retrieval.py` — общий у сюит только golden-set.

`shown_recall` (по выдаче тула) остаётся диагностикой: в паре с `cited_recall` видно,
где потерялась карточка — в поиске или в ответе.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from crossmarket.agent import LIMIT_SUBTYPES, Answer, ask, build_options, flush_langfuse
from crossmarket.config import AGENT_MODEL, RETRIEVAL_MIN_SCORE
from evals.cases import Case, split_summary
from evals.grading import Grade, grade
from evals.report import Table, print_tables, write_report

RUNS_DIR = Path("evals/runs")

ID_LINE = re.compile(r"^id: (\S+)$", re.MULTILINE)


def shown_ids(answer: Answer) -> list[str]:
    """Идентификаторы карточек, которые агент увидел в выдаче тула."""
    return [product_id for text in answer.tool_results for product_id in ID_LINE.findall(text)]


def evaluate_case(case: Case, answer: Answer) -> dict[str, Any]:
    shown = set(shown_ids(answer))
    relevant = set(case.relevant_ids)
    return {
        "case": case,
        "answer": answer,
        "grade": grade(case, answer.text, answer.tool_results),
        "shown_recall": len(shown & relevant) / len(relevant) if relevant else None,
    }


def _mark(row: dict[str, Any]) -> str:
    result: Grade = row["grade"]
    if not result.outcome_correct:
        return "мимо" if not row["answer"].searched else "—"
    return "отказ+" if row["case"].expected == "absent" else "+"


async def run_cases(cases: list[Case], limit_turns: int, budget: float, verbose: bool) -> list[dict[str, Any]]:
    options = build_options(max_turns=limit_turns, max_budget_usd=budget)
    results = []
    for number, case in enumerate(cases, start=1):
        answer = await ask(case.question, options)
        row = evaluate_case(case, answer)
        results.append(row)

        print(f"{number:>3}/{len(cases)} {_mark(row):<7} {case.id} {case.question[:56]}")
        if answer.limit_hit:
            print(f"           оборвано лимитом: {answer.limit_hit}")
        if answer.error:
            print(f"           ошибка: {answer.error[:160]}")
        if row["grade"].unknown_ids or row["grade"].bad_prices:
            print(f"           не из выдачи: {row['grade'].unknown_ids} {row['grade'].bad_prices}")
        if verbose:
            for step in answer.thinking:
                print(f"           думает: {step[:160]}")
            for call in answer.tool_calls:
                print(f"           ищет:   {call['input'].get('question')}")
            print(f"           ответ:  {answer.text[:300]}")
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metrics_of(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Метрики набора. `cited_recall` и `shown_recall` — только по кейсам с разметкой."""
    if not rows:
        return {}
    labelled = [row for row in rows if row["case"].expected == "found"]
    absent = [row for row in rows if row["case"].expected == "absent"]
    metrics = {
        "outcome_correct": _mean([float(row["grade"].outcome_correct) for row in rows]),
        "cited_recall": _mean([row["grade"].cited_recall for row in labelled]),
        "shown_recall": _mean([row["shown_recall"] for row in labelled]),
        "faithful": _mean([float(row["grade"].faithful) for row in rows]),
        "tool_called": _mean([float(row["answer"].searched) for row in rows]),
        "answered": _mean([float(bool(row["answer"].text.strip())) for row in rows]),
        "refused": _mean([float(row["grade"].refused) for row in rows]),
        "limit_hit": _mean([float(bool(row["answer"].limit_hit)) for row in rows]),
        "errors": _mean([float(bool(row["answer"].error)) for row in rows]),
        "avg_turns": _mean([float(row["answer"].num_turns) for row in rows]),
        "avg_cost_usd": _mean([row["answer"].cost_usd for row in rows]),
        "total_cost_usd": sum(row["answer"].cost_usd for row in rows),
    }
    if absent:
        metrics["denied"] = _mean([float(row["grade"].denied) for row in absent])
    return metrics


def cases_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(
        "Кейсы",
        ["вопрос", "тип", "сплит", "cited", "shown", "faith", "исход", "ходов", "$"],
    )
    for row in rows:
        case, answer, result = row["case"], row["answer"], row["grade"]
        table.rows.append(
            [
                case.question,
                case.stratum,
                case.split,
                result.cited_recall if case.expected == "found" else "—",
                row["shown_recall"] if row["shown_recall"] is not None else "—",
                "да" if result.faithful else "НЕТ",
                "+" if result.outcome_correct else "—",
                answer.num_turns,
                f"{answer.cost_usd:.4f}",
            ]
        )
    return table


def summary_table(rows: list[dict[str, Any]]) -> Table:
    """Метрики по всему набору и по тест-сплиту. В README идёт строка `test`."""
    table = Table(
        "Метрики",
        ["набор", "кейсов", "исход", "cited_recall", "shown_recall", "faithful", "тул звал", "ходов", "$/вопрос"],
    )
    groups = [("весь набор", rows), ("test (отчётный)", [row for row in rows if row["case"].split == "test"])]
    for name, group in groups:
        metrics = metrics_of(group)
        if not metrics:
            continue
        table.rows.append(
            [
                name,
                len(group),
                metrics["outcome_correct"],
                metrics["cited_recall"],
                metrics["shown_recall"],
                metrics["faithful"],
                metrics["tool_called"],
                metrics["avg_turns"],
                f"{metrics['avg_cost_usd']:.4f}",
            ]
        )
    return table


def strata_table(rows: list[dict[str, Any]]) -> Table:
    """Исход по типам кейсов: с ответом в корпусе, близкий промах, далёкий."""
    table = Table("Исход по типам кейсов", ["тип", "кейсов", "верный исход", "явный отказ", "карточек не назвал"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["case"].stratum, []).append(row)
    for name, group in sorted(groups.items()):
        metrics = metrics_of(group)
        table.rows.append(
            [name, len(group), metrics["outcome_correct"], metrics.get("denied", "—"), metrics["refused"]]
        )
    return table


def absent_answers_table(rows: list[dict[str, Any]]) -> Table | None:
    """Ответы на отрицательные кейсы целиком.

    Проверка отказа по словам теряет перефразировки, и до судьи-QA единственная
    защита — прочитать ответы глазами, поэтому они лежат в отчёте.
    """
    table = Table("Ответы на отрицательные кейсы", ["кейс", "тип", "отказ", "ответ"])
    for row in rows:
        if row["case"].expected != "absent":
            continue
        table.rows.append(
            [row["case"].id, row["case"].kind, "да" if row["grade"].denied else "НЕТ", row["answer"].text[:200]]
        )
    return table if table.rows else None


def violations_table(rows: list[dict[str, Any]]) -> Table | None:
    """Что в ответе не подтвердилось выдачей тула. С самим числом-нарушителем:
    проверка цен эвристическая, ложное срабатывание должно быть видно."""
    table = Table("Не подтверждено выдачей тула", ["кейс", "id не из выдачи", "цены не из выдачи", "ответ"])
    for row in rows:
        result: Grade = row["grade"]
        if result.unknown_ids or result.bad_prices:
            table.rows.append(
                [
                    row["case"].id,
                    ", ".join(result.unknown_ids) or "—",
                    ", ".join(map(str, result.bad_prices)) or "—",
                    row["answer"].text[:80],
                ]
            )
    return table if table.rows else None


def dump_run(rows: list[dict[str, Any]], name: str = "qa_wb", limits: str = "") -> Path:
    """Сложить ответы и выдачу тула на диск для `regrade`.

    В гит не идёт — там названия и цены настоящих карточек.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            case, answer = row["case"], row["answer"]
            fh.write(
                json.dumps(
                    {
                        **case.to_dict(),
                        "answer": answer.text,
                        "tool_results": answer.tool_results,
                        "queries": [call["input"].get("question") for call in answer.tool_calls],
                        "num_turns": answer.num_turns,
                        "cost_usd": answer.cost_usd,
                        "subtype": answer.subtype,
                        "limits": limits,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _limits_of(name: str) -> str:
    """Лимиты того прогона, а не сегодняшние: перегрейд не должен выдавать дефолт за факт."""
    first = json.loads((RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return first.get("limits") or "не записаны в дампе"


def load_run(name: str = "qa_wb") -> list[dict[str, Any]]:
    """Восстановить прогон с диска и оценить заново.

    Разбор ответа меняется чаще самого прогона; перегрейд бесплатен и оценивает те
    самые ответы, на которых промах увидели, — новый прогон дал бы другие.
    """
    path = RUNS_DIR / f"{name}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        answer = Answer(
            text=raw["answer"],
            tool_results=raw["tool_results"],
            tool_calls=[{"name": "search_wb", "input": {"question": query}} for query in raw["queries"]],
            subtype=raw["subtype"],
            num_turns=raw["num_turns"],
            cost_usd=raw["cost_usd"],
            limit_hit=LIMIT_SUBTYPES.get(raw["subtype"]),
        )
        rows.append(evaluate_case(Case.from_dict(raw), answer))
    return rows


def log_to_mlflow(rows: list[dict[str, Any]], limits: str) -> None:
    import mlflow

    metrics = metrics_of(rows)
    metrics |= {
        f"{name}_test": value
        for name, value in metrics_of([row for row in rows if row["case"].split == "test"]).items()
    }

    mlflow.set_experiment("qa_wb")
    with mlflow.start_run(run_name=AGENT_MODEL):
        mlflow.log_params(
            {
                "model": AGENT_MODEL,
                "min_score": RETRIEVAL_MIN_SCORE,
                "limits": limits,
                "questions": len(rows),
            }
        )
        mlflow.log_metrics(metrics)


def run(
    cases: list[Case],
    limit_turns: int,
    budget: float,
    verbose: bool,
    use_mlflow: bool,
    name: str = "qa_wb",
) -> None:
    rows = asyncio.run(run_cases(cases, limit_turns, budget, verbose))
    flush_langfuse()
    limits = f"max_turns={limit_turns}, max_budget_usd={budget}"
    report(rows, limits, use_mlflow, dump_run(rows, name, limits), name)


def regrade(use_mlflow: bool = False, name: str = "qa_wb") -> None:
    """Пересчитать метрики сохранённого прогона новым разбором. LLM не зовётся."""
    rows = load_run(name)
    print(f"перегрейд {len(rows)} ответов из {RUNS_DIR / f'{name}.jsonl'}\n")
    report(rows, _limits_of(name), use_mlflow, RUNS_DIR / f"{name}.jsonl", name)


def report(rows: list[dict[str, Any]], limits: str, use_mlflow: bool, dump: Path, name: str = "qa_wb") -> None:
    """Таблицы в stdout, markdown-отчёт, метрики в MLflow. Общее для прогона и перегрейда.

    В файл идут только агрегаты: построчные таблицы и ответы содержат названия и цены
    настоящих карточек, а данных в репозитории нет. Смотреть их — в stdout и в дампе.
    """
    cases = [row["case"] for row in rows]
    aggregates = [summary_table(rows), strata_table(rows)]
    screen = [cases_table(rows), *aggregates]
    if answers := absent_answers_table(rows):
        screen.append(answers)
    if violations := violations_table(rows):
        screen.append(violations)
    print_tables(screen)

    subtypes: dict[str, int] = {}
    for row in rows:
        subtypes[row["answer"].subtype or "—"] = subtypes.get(row["answer"].subtype or "—", 0) + 1
    total = sum(row["answer"].cost_usd for row in rows)
    print(f"\nисходы SDK: {subtypes}")
    print(f"вопросов {len(rows)}, модель {AGENT_MODEL}, порог {RETRIEVAL_MIN_SCORE}, прогон ${total:.2f}")

    path = write_report(
        name,
        "Агент над Retrieval-QA: end-to-end на golden-set",
        "Метрика по итоговому ответу: карточка засчитана, если агент сослался на её id. "
        "`shown_recall` (метрика этапа 2, по выдаче тула) оставлена диагностикой.",
        {
            "оркестратор": AGENT_MODEL,
            "порог по скору": RETRIEVAL_MIN_SCORE,
            "лимиты": limits,
            "кейсов": len(cases),
            "сплиты": split_summary(cases),
            "стоимость прогона": f"${total:.2f}",
        },
        aggregates,
    )
    print(
        f"Отчёт: {path}, сырой прогон: {dump}"
        + ("" if not use_mlflow else "\nМетрики записаны в MLflow: эксперимент qa_wb")
    )
    if use_mlflow:
        log_to_mlflow(rows, limits)
