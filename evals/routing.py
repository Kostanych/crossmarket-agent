"""Сюита роутинга: вопрос → агент с обоими тулами → какой тул он выбрал.

Метрика считается по вызовам тулов, а не по тексту ответа: качество самих ответов
меряют сюиты A и C, здесь предмет замера — выбор.

Вызовы скилла из последовательности вычищаются: `Skill` — служебный тул, к развилке
A/C отношения не имеющий. Что скилл вообще звался, остаётся диагностикой `skill_used`.

Мультихоповые кейсы в accuracy не входят: правильных вызовов там два, и порядок между
ними вопросом не задан. Они печатаются целиком и читаются глазами.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossmarket.agent import LIMIT_SUBTYPES, SEARCH_TOOL, SQL_TOOL, Answer, ask, flush_langfuse, router_options
from crossmarket.config import AGENT_MODEL
from evals.cases import ROUTING_TOOLS, RoutingCase, split_summary
from evals.report import Table, print_tables, write_report

RUNS_DIR = Path("evals/runs")

FULL_NAMES = {SEARCH_TOOL.rsplit("__", 1)[-1]: SEARCH_TOOL, SQL_TOOL.rsplit("__", 1)[-1]: SQL_TOOL}
"""Короткое имя лейбла → полное имя MCP-тула. Нужно на перегрейде: в дампе лежат
короткие имена, а `Answer` собирается с полными."""

SKILL_TOOL = "Skill"


def tool_sequence(answer: Answer) -> list[str]:
    """Короткие имена наших тулов в порядке вызова, без служебных."""
    names = []
    for call in answer.tool_calls:
        short = call["name"].rsplit("__", 1)[-1]
        if short in ROUTING_TOOLS:
            names.append(short)
    return names


@dataclass
class RoutingGrade:
    """Разбор одного кейса роутинга."""

    sequence: list[str] = field(default_factory=list)
    skill_used: bool = False
    routed: bool = False
    used_expected: bool = False
    used_other: bool = False

    @property
    def no_tool(self) -> bool:
        return not self.sequence


def grade(case: RoutingCase, answer: Answer) -> RoutingGrade:
    """Оценить выбор тула.

    `routed` решается первым вызовом: он и есть решение о маршруте. Дальнейшие вызовы
    ловятся отдельно — агент, сходивший в оба тула, ответил, возможно, и верно, но
    выбора не сделал.
    """
    sequence = tool_sequence(answer)
    result = RoutingGrade(
        sequence=sequence,
        skill_used=any(call["name"] == SKILL_TOOL for call in answer.tool_calls),
    )
    if not case.single_hop:
        return result
    result.routed = bool(sequence) and sequence[0] == case.tool
    result.used_expected = case.tool in sequence
    result.used_other = any(name != case.tool for name in sequence)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metrics_of(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Метрики набора. Accuracy считается только по однохоповым кейсам."""
    single = [row for row in rows if row["case"].single_hop]
    if not rows:
        return {}
    metrics = {
        "avg_turns": _mean([float(row["answer"].num_turns) for row in rows]),
        "avg_cost_usd": _mean([row["answer"].cost_usd for row in rows]),
        "total_cost_usd": sum(row["answer"].cost_usd for row in rows),
        "skill_used": _mean([float(row["grade"].skill_used) for row in rows]),
        "limit_hit": _mean([float(bool(row["answer"].limit_hit)) for row in rows]),
        "errors": _mean([float(bool(row["answer"].error)) for row in rows]),
    }
    if single:
        metrics |= {
            "routed": _mean([float(row["grade"].routed) for row in single]),
            "used_expected": _mean([float(row["grade"].used_expected) for row in single]),
            "used_other": _mean([float(row["grade"].used_other) for row in single]),
            "no_tool": _mean([float(row["grade"].no_tool) for row in single]),
        }
    return metrics


async def run_cases(cases: list[RoutingCase], limit_turns: int, budget: float, verbose: bool) -> list[dict[str, Any]]:
    options = router_options(max_turns=limit_turns, max_budget_usd=budget)
    results = []
    for number, case in enumerate(cases, start=1):
        answer = await ask(case.question, options)
        result = grade(case, answer)
        results.append({"case": case, "answer": answer, "grade": result})

        mark = "—" if case.single_hop and not result.routed else "+"
        print(f"{number:>3}/{len(cases)} {mark:<5} {case.id} [{case.kind}] {case.question[:50]}")
        print(f"           звал: {' → '.join(result.sequence) or 'ничего'}, ожидался {case.tool or '—'}")
        if answer.limit_hit:
            print(f"           оборвано лимитом: {answer.limit_hit}")
        if answer.error:
            print(f"           ошибка: {answer.error[:160]}")
        if verbose:
            print(f"           ответ:  {answer.text[:300]}")
    return results


def cases_table(rows: list[dict[str, Any]]) -> Table:
    table = Table("Кейсы", ["вопрос", "страта", "сплит", "ожидался", "звал", "верно", "ходов", "$"])
    for row in rows:
        case, answer, result = row["case"], row["answer"], row["grade"]
        table.rows.append(
            [
                case.question,
                case.kind,
                case.split,
                case.tool or "—",
                " → ".join(result.sequence) or "—",
                ("+" if result.routed else "—") if case.single_hop else "·",
                answer.num_turns,
                f"{answer.cost_usd:.4f}",
            ]
        )
    return table


def summary_table(rows: list[dict[str, Any]]) -> Table:
    """Метрики по всему набору и по тест-сплиту. В README идёт строка `test`."""
    table = Table(
        "Метрики роутинга",
        ["набор", "однохоповых", "routed", "нужный тул звал", "звал и лишний", "не звал ничего", "ходов", "$/вопрос"],
    )
    groups = [("весь набор", rows), ("test (отчётный)", [row for row in rows if row["case"].split == "test"])]
    for name, group in groups:
        metrics = metrics_of(group)
        if "routed" not in metrics:
            continue
        table.rows.append(
            [
                name,
                sum(1 for row in group if row["case"].single_hop),
                metrics["routed"],
                metrics["used_expected"],
                metrics["used_other"],
                metrics["no_tool"],
                metrics["avg_turns"],
                f"{metrics['avg_cost_usd']:.4f}",
            ]
        )
    return table


def strata_table(rows: list[dict[str, Any]]) -> Table:
    """Разрез по типам вопросов: прямые против пограничных."""
    table = Table("По типам вопросов", ["страта", "кейсов", "routed", "звал и лишний", "скилл звал", "ходов"])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["case"].kind, []).append(row)
    for name, group in sorted(groups.items()):
        metrics = metrics_of(group)
        table.rows.append(
            [
                name,
                len(group),
                metrics.get("routed", "—"),
                metrics.get("used_other", "—"),
                metrics["skill_used"],
                metrics["avg_turns"],
            ]
        )
    return table


def multihop_table(rows: list[dict[str, Any]]) -> Table | None:
    """Цепочки целиком: какие тулы и в каком порядке, и что вышло в ответе.

    Метрики здесь нет намеренно — это качественная проверка, что цепочка вообще
    складывается. Только в stdout: в ответах названия и цены настоящих карточек.
    """
    table = Table("Мультихоповые кейсы (вне метрики)", ["кейс", "тулы", "ходов", "ответ"])
    for row in rows:
        if row["case"].single_hop:
            continue
        table.rows.append(
            [
                row["case"].id,
                " → ".join(row["grade"].sequence) or "—",
                row["answer"].num_turns,
                row["answer"].text[:200],
            ]
        )
    return table if table.rows else None


def misses_table(rows: list[dict[str, Any]]) -> Table | None:
    """Кейсы, где маршрут выбран не тот. Вопрос рядом с решением — чтобы было видно,
    спорная это формулировка или промах агента."""
    table = Table("Промахи роутинга", ["кейс", "страта", "ожидался", "звал", "вопрос"])
    for row in rows:
        case, result = row["case"], row["grade"]
        if not case.single_hop or result.routed:
            continue
        table.rows.append([case.id, case.kind, case.tool, " → ".join(result.sequence) or "—", case.question])
    return table if table.rows else None


def dump_run(rows: list[dict[str, Any]], name: str, limits: str = "") -> Path:
    """Сложить прогон на диск для `regrade`. В гит не идёт: в ответах данные карточек."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            case, answer, result = row["case"], row["answer"], row["grade"]
            fh.write(
                json.dumps(
                    {
                        **case.to_dict(),
                        "answer": answer.text,
                        "sequence": result.sequence,
                        "skill_used": result.skill_used,
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


def load_run(name: str) -> list[dict[str, Any]]:
    """Восстановить прогон с диска и оценить заново. LLM не зовётся."""
    rows = []
    for line in (RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        calls = [{"name": FULL_NAMES[short], "input": {}} for short in raw["sequence"]]
        if raw.get("skill_used"):
            calls.insert(0, {"name": SKILL_TOOL, "input": {}})
        answer = Answer(
            text=raw["answer"],
            tool_calls=calls,
            subtype=raw["subtype"],
            num_turns=raw["num_turns"],
            cost_usd=raw["cost_usd"],
            limit_hit=LIMIT_SUBTYPES.get(raw["subtype"]),
        )
        case = RoutingCase.from_dict(raw)
        rows.append({"case": case, "answer": answer, "grade": grade(case, answer)})
    return rows


def log_to_mlflow(rows: list[dict[str, Any]], limits: str) -> None:
    import mlflow

    metrics = metrics_of(rows)
    metrics |= {
        f"{name}_test": value
        for name, value in metrics_of([row for row in rows if row["case"].split == "test"]).items()
    }

    mlflow.set_experiment("routing")
    with mlflow.start_run(run_name=AGENT_MODEL):
        mlflow.log_params({"model": AGENT_MODEL, "limits": limits, "questions": len(rows)})
        mlflow.log_metrics(metrics)


def run(
    cases: list[RoutingCase],
    limit_turns: int,
    budget: float,
    verbose: bool,
    use_mlflow: bool,
    name: str = "routing",
) -> None:
    rows = asyncio.run(run_cases(cases, limit_turns, budget, verbose))
    flush_langfuse()
    limits = f"max_turns={limit_turns}, max_budget_usd={budget}"
    report(rows, limits, use_mlflow, dump_run(rows, name, limits), name)


def regrade(use_mlflow: bool = False, name: str = "routing") -> None:
    """Пересчитать метрики сохранённого прогона новым разбором. LLM не зовётся."""
    rows = load_run(name)
    print(f"перегрейд {len(rows)} прогонов из {RUNS_DIR / f'{name}.jsonl'}\n")
    report(rows, _limits_of(name), use_mlflow, RUNS_DIR / f"{name}.jsonl", name)


def report(rows: list[dict[str, Any]], limits: str, use_mlflow: bool, dump: Path, name: str = "routing") -> None:
    """Таблицы в stdout, markdown-отчёт, метрики в MLflow. Общее для прогона и перегрейда."""
    cases = [row["case"] for row in rows]
    aggregates = [summary_table(rows), strata_table(rows)]
    screen = [cases_table(rows), *aggregates]
    if misses := misses_table(rows):
        screen.append(misses)
    if multihop := multihop_table(rows):
        screen.append(multihop)
    print_tables(screen)

    total = sum(row["answer"].cost_usd for row in rows)
    single = sum(1 for case in cases if case.single_hop)
    print(f"\nвопросов {len(rows)} (в accuracy {single}), модель {AGENT_MODEL}, прогон ${total:.2f}")

    path = write_report(
        name,
        "Роутинг A+C: какой тул выбирает оркестратор",
        "Accuracy роутинга — доля однохоповых вопросов, где первым вызванным тулом оказался "
        "размеченный. Вызовы скилла из последовательности вычищены. Мультихоповые кейсы в метрику "
        "не входят: правильных вызовов там два, и порядок между ними вопросом не задан.",
        {
            "оркестратор": AGENT_MODEL,
            "лимиты": limits,
            "кейсов": f"{len(cases)}, из них в accuracy {single}",
            "сплиты": split_summary(cases),
            "стоимость прогона": f"${total:.2f}",
        },
        aggregates,
    )
    print(
        f"Отчёт: {path}, сырой прогон: {dump}"
        + ("" if not use_mlflow else "\nМетрики записаны в MLflow: эксперимент routing")
    )
    if use_mlflow:
        log_to_mlflow(rows, limits)
