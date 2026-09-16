"""Сюита тула C: вопрос → SQL агента → сравнение результата с эталонным SQL.

Сравниваются строки результата, а не текст запроса: нормализуются округление чисел, порядок строк,
перестановка и лишние колонки.

Засчитывается любой исполнившийся запрос агента; совпал ли последний — диагностика `matched_last`. Запросы
переисполняются харнессом по базе, форматированная выдача тула не разбирается.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

from crossmarket.agent import LIMIT_SUBTYPES, Answer, ask, flush_langfuse, sql_options
from crossmarket.config import AGENT_MODEL
from crossmarket.sql import MAX_RESULT_ROWS, SqlResult, run_sql
from crossmarket.storage import clickhouse
from evals.cases import SqlCase, split_summary
from evals.grading import DENIAL
from evals.report import Table, print_tables, write_report

RUNS_DIR = Path("evals/runs")

ROUND = 2
"""До скольких знаков округляются числа перед сравнением."""

MAX_PERMUTATIONS = 5000
"""Предохранитель перебора колонок: `SELECT *` даёт десять колонок, и при эталоне из четырёх это 5040 сочетаний."""

ANSWER_ROWS = 5
"""Сколько первых строк эталона проверяются в тексте ответа."""

Row = tuple[Any, ...]


def normalise_value(value: Any) -> Any:
    """Значение в сравнимый вид: числа округляются, строки чистятся, NULL остаётся NULL."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        return round(float(value), ROUND)
    return str(value).strip()


def normalise_rows(rows: list[Row], ordered: bool) -> list[Row]:
    """Строки в сравнимый вид. Без `ordered` порядок снимается сортировкой.

    Сортировка по строковому представлению, а не по значениям: колонки разнотипные,
    и `None` рядом с числом иначе не сравнивается.
    """
    normalised = [tuple(normalise_value(value) for value in row) for row in rows]
    return normalised if ordered else sorted(normalised, key=lambda row: tuple(str(v) for v in row))


def _select(rows: list[Row], indexes: tuple[int, ...]) -> list[Row]:
    return [tuple(row[index] for index in indexes) for row in rows]


def compare(actual: list[Row], expected: list[Row], ordered: bool) -> tuple[bool, bool]:
    """`(совпало, совпало точно)`.

    Совпадением считается наличие в ответе подмножества колонок, дающего эталон: `SELECT *` на вопросе «топ-5
    самых дешёвых» даёт те же строки в том же порядке плюс лишние поля и засчитывается. Точное совпадение по
    форме — отдельная цифра `exact_match`.

    Перебор идёт по колонкам, а не по строкам: строки уже упорядочены запросом.
    """
    target = normalise_rows(expected, ordered)
    if not actual and not expected:
        return True, True
    if not actual or not expected:
        return False, False
    width, needed = len(actual[0]), len(expected[0])
    if width < needed:
        return False, False

    exact = normalise_rows(actual, ordered) == target if width == needed else False
    if exact:
        return True, True

    checked = 0
    for indexes in permutations(range(width), needed):
        checked += 1
        if checked > MAX_PERMUTATIONS:
            break
        if normalise_rows(_select(actual, indexes), ordered) == target:
            return True, False
    return False, False


_DIGIT_GROUP = re.compile(r"(?<=\d)[\s  ](?=\d)")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")

MARKETPLACE_NAMES = {
    "wb": ("wb", "wildberries", "вайлдберриз"),
    "ozon": ("ozon", "озон"),
}
"""Код площадки в строках эталона → названия, которыми её называет ответ."""


def _flatten_numbers(text: str) -> str:
    """Свести написание чисел к виду эталона: «93 666» → «93666», «1456,64» → «1456.64». Обе
    замены — только между цифрами.
    """
    return _DECIMAL_COMMA.sub(".", _DIGIT_GROUP.sub("", text))


def mentions(text: str, value: Any) -> bool:
    """Есть ли значение эталона в тексте ответа."""
    if value is None:
        return True
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        number = round(float(value), ROUND)
        forms = {f"{number:.{ROUND}f}".rstrip("0").rstrip(".")}
        if number == int(number):
            forms.add(str(int(number)))
        return any(form in text for form in forms)

    needle = str(value).strip().lower()
    lowered = text.lower()
    if needle in MARKETPLACE_NAMES:
        return any(name in lowered for name in MARKETPLACE_NAMES[needle])
    return needle[:40] in lowered


def answer_mentions(text: str, expected: list[Row], ordered: bool) -> bool:
    """Назвал ли агент значения эталона в самом ответе. Вторичная проверка к совпадению строк."""
    flat = _flatten_numbers(text)
    if not expected:
        return bool(DENIAL.search(text))
    for row in normalise_rows(expected, ordered)[:ANSWER_ROWS]:
        if not all(mentions(flat, value) for value in row):
            return False
    return True


@dataclass
class SqlGrade:
    """Разбор одного ответа сюиты C."""

    attempts: int = 0
    failures: int = 0
    sql: str = ""
    executed: bool = False
    result_match: bool = False
    exact_match: bool = False
    matched_last: bool = False
    answer_match: bool = False
    rows_returned: int = 0
    error: str = ""

    @property
    def first_try(self) -> bool:
        """Ни одного неисполнившегося запроса по дороге."""
        return self.executed and self.failures == 0


def agent_sqls(answer: Answer) -> list[str]:
    """Запросы, которые агент отправил в тул, по порядку."""
    return [
        call["input"]["sql"]
        for call in answer.tool_calls
        if call["name"].endswith("execute_sql") and isinstance(call["input"].get("sql"), str)
    ]


def grade(case: SqlCase, answer: Answer, client: Any) -> SqlGrade:
    """Прогнать запросы агента и эталон по базе, сравнить результаты."""
    reference = run_sql(client, case.sql, shown_rows=MAX_RESULT_ROWS)
    if not reference.ok:
        return SqlGrade(error=f"reference SQL fails: {reference.error}")

    result = SqlGrade(answer_match=answer_mentions(answer.text, reference.rows, case.ordered))
    for sql in agent_sqls(answer):
        result.attempts += 1
        got: SqlResult = run_sql(client, sql, shown_rows=MAX_RESULT_ROWS)
        if not got.ok:
            result.failures += 1
            continue
        result.executed = True
        matched, exact = compare(got.rows, reference.rows, case.ordered)
        result.matched_last = matched
        if matched and not result.result_match:
            # Совпавший запрос запоминается вместе со своими строками: в отчёт идёт он,
            # а не последний по счёту.
            result.sql, result.rows_returned = sql, got.total_rows
        result.result_match = result.result_match or matched
        result.exact_match = result.exact_match or exact
        if not result.result_match:
            result.sql, result.rows_returned = sql, got.total_rows
    if not result.executed and result.attempts:
        result.error = "no query executed"
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metrics_of(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "result_match": _mean([float(row["grade"].result_match) for row in rows]),
        "exact_match": _mean([float(row["grade"].exact_match) for row in rows]),
        "answer_match": _mean([float(row["grade"].answer_match) for row in rows]),
        "executed": _mean([float(row["grade"].executed) for row in rows]),
        "first_try": _mean([float(row["grade"].first_try) for row in rows]),
        "matched_last": _mean([float(row["grade"].matched_last) for row in rows]),
        "avg_sql_calls": _mean([float(row["grade"].attempts) for row in rows]),
        "avg_turns": _mean([float(row["answer"].num_turns) for row in rows]),
        "limit_hit": _mean([float(bool(row["answer"].limit_hit)) for row in rows]),
        "errors": _mean([float(bool(row["answer"].error)) for row in rows]),
        "avg_cost_usd": _mean([row["answer"].cost_usd for row in rows]),
        "total_cost_usd": sum(row["answer"].cost_usd for row in rows),
    }


async def run_cases(cases: list[SqlCase], limit_turns: int, budget: float, verbose: bool) -> list[dict[str, Any]]:
    options = sql_options(max_turns=limit_turns, max_budget_usd=budget)
    client = clickhouse.connect()
    results = []
    for number, case in enumerate(cases, start=1):
        answer = await ask(case.question, options)
        result = grade(case, answer, client)
        results.append({"case": case, "answer": answer, "grade": result})

        mark = "+" if result.result_match else ("~" if result.executed else "miss")
        print(f"{number:>3}/{len(cases)} {mark:<5} {case.id} [{case.kind}] {case.question[:52]}")
        if answer.limit_hit:
            print(f"           limit hit: {answer.limit_hit}")
        if answer.error:
            print(f"           error: {answer.error[:160]}")
        if result.failures:
            print(f"           failed queries: {result.failures} of {result.attempts}")
        if verbose:
            for sql in agent_sqls(answer):
                print(f"           sql:    {sql[:160]}")
            print(f"           answer: {answer.text[:300]}")
    return results


def cases_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(
        "Cases",
        ["question", "stratum", "split", "result", "exact", "in answer", "queries", "turns", "$"],
    )
    for row in rows:
        case, answer, result = row["case"], row["answer"], row["grade"]
        table.rows.append(
            [
                case.question,
                case.kind,
                case.split,
                "+" if result.result_match else "—",
                "+" if result.exact_match else "—",
                "+" if result.answer_match else "—",
                result.attempts,
                answer.num_turns,
                f"{answer.cost_usd:.4f}",
            ]
        )
    return table


def summary_table(rows: list[dict[str, Any]]) -> Table:
    """Метрики по всему набору и по тест-сплиту. В README идёт строка `test`."""
    table = Table(
        "Metrics",
        ["set", "cases", "result_match", "exact_match", "answer_match", "SQL executed", "first try", "$"],
    )
    groups = [("full set", rows), ("test (reported)", [row for row in rows if row["case"].split == "test"])]
    for name, group in groups:
        metrics = metrics_of(group)
        if not metrics:
            continue
        table.rows.append(
            [
                name,
                len(group),
                metrics["result_match"],
                metrics["exact_match"],
                metrics["answer_match"],
                metrics["executed"],
                metrics["first_try"],
                f"{metrics['avg_cost_usd']:.4f}",
            ]
        )
    return table


def strata_table(rows: list[dict[str, Any]]) -> Table:
    table = Table("By question type", ["stratum", "cases", "result_match", "first try", "queries per question"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["case"].kind, []).append(row)
    for name, group in sorted(groups.items()):
        metrics = metrics_of(group)
        table.rows.append([name, len(group), metrics["result_match"], metrics["first_try"], metrics["avg_sql_calls"]])
    return table


def misses_table(rows: list[dict[str, Any]]) -> Table | None:
    """Где не совпало: вопрос, эталон и запрос агента рядом. Только в stdout и в дампе:
    эталонный SQL и результаты несут данные карточек.
    """
    table = Table("Mismatches", ["case", "reference SQL", "agent SQL", "rows"])
    for row in rows:
        result = row["grade"]
        if result.result_match:
            continue
        table.rows.append([row["case"].id, row["case"].sql, result.sql or result.error or "—", result.rows_returned])
    return table if table.rows else None


def dump_run(rows: list[dict[str, Any]], name: str, limits: str = "") -> Path:
    """Сложить ответы и запросы агента на диск для `regrade`. В гит не идёт."""
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
                        "sqls": agent_sqls(answer),
                        "num_turns": answer.num_turns,
                        "cost_usd": answer.cost_usd,
                        "duration_ms": answer.duration_ms,
                        "usage": answer.usage,
                        "model_usage": answer.model_usage,
                        "subtype": answer.subtype,
                        "limits": limits,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _limits_of(name: str) -> str:
    """Лимиты из дампа прогона, а не текущие из конфига."""
    first = json.loads((RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return first.get("limits") or "not recorded in the dump"


def load_run(name: str) -> list[dict[str, Any]]:
    """Восстановить прогон с диска и оценить заново. LLM не зовётся, база — зовётся."""
    client = clickhouse.connect()
    rows = []
    for line in (RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        answer = Answer(
            text=raw["answer"],
            tool_calls=[{"name": "execute_sql", "input": {"sql": sql}} for sql in raw["sqls"]],
            subtype=raw["subtype"],
            num_turns=raw["num_turns"],
            cost_usd=raw["cost_usd"],
            limit_hit=LIMIT_SUBTYPES.get(raw["subtype"]),
        )
        case = SqlCase.from_dict(raw)
        rows.append({"case": case, "answer": answer, "grade": grade(case, answer, client)})
    return rows


def log_to_mlflow(rows: list[dict[str, Any]], limits: str) -> None:
    import mlflow

    metrics = metrics_of(rows)
    metrics |= {
        f"{name}_test": value
        for name, value in metrics_of([row for row in rows if row["case"].split == "test"]).items()
    }

    mlflow.set_experiment("text2sql")
    with mlflow.start_run(run_name=AGENT_MODEL):
        mlflow.log_params({"model": AGENT_MODEL, "limits": limits, "questions": len(rows)})
        mlflow.log_metrics(metrics)


def run(
    cases: list[SqlCase],
    limit_turns: int,
    budget: float,
    verbose: bool,
    use_mlflow: bool,
    name: str = "sql_c",
) -> None:
    rows = asyncio.run(run_cases(cases, limit_turns, budget, verbose))
    flush_langfuse()
    limits = f"max_turns={limit_turns}, max_budget_usd={budget}"
    report(rows, limits, use_mlflow, dump_run(rows, name, limits), name)


def regrade(use_mlflow: bool = False, name: str = "sql_c") -> None:
    """Пересчитать метрики сохранённого прогона новым сравнением. LLM не зовётся."""
    rows = load_run(name)
    print(f"regrading {len(rows)} answers from {RUNS_DIR / f'{name}.jsonl'}\n")
    report(rows, _limits_of(name), use_mlflow, RUNS_DIR / f"{name}.jsonl", name)


def report(rows: list[dict[str, Any]], limits: str, use_mlflow: bool, dump: Path, name: str = "sql_c") -> None:
    """Таблицы в stdout, markdown-отчёт, метрики в MLflow. Общее для прогона и перегрейда."""
    cases = [row["case"] for row in rows]
    aggregates = [summary_table(rows), strata_table(rows)]
    screen = [cases_table(rows), *aggregates]
    if misses := misses_table(rows):
        screen.append(misses)
    print_tables(screen)

    total = sum(row["answer"].cost_usd for row in rows)
    print(f"\nquestions {len(rows)}, model {AGENT_MODEL}, run ${total:.2f}")

    path = write_report(
        name,
        "Tool C: text2sql over the ClickHouse snapshot",
        "An answer counts if the rows of at least one executed agent query match the rows of the case's "
        "reference SQL after normalization: rounding of numbers, row order for unordered questions, "
        "column permutations and extra columns. `exact_match` is a match with no leniency on shape, "
        "`answer_match` is whether the reference values are named in the answer text itself.",
        {
            "orchestrator": AGENT_MODEL,
            "limits": limits,
            "cases": len(cases),
            "splits": split_summary(cases),
            "run cost": f"${total:.2f}",
        },
        aggregates,
    )
    print(
        f"Report: {path}, raw run: {dump}"
        + ("" if not use_mlflow else "\nMetrics logged to MLflow: experiment text2sql")
    )
    if use_mlflow:
        log_to_mlflow(rows, limits)
