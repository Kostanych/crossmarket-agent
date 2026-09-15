"""Стоимость сохранённых прогонов из `evals/runs/*.jsonl`: `python -m evals cost`, без LLM.

Стоимость — `total_cost_usd` SDK, оценка по прайсу вшитой CLI. В дампах до этапа 8 нет
токенов, длительности и `nested_cost_usd` — в таких колонках прочерк.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.report import Table, write_report

RUNS_DIR = Path("evals/runs")

CACHE_KINDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _percentile(values: list[float], share: float) -> float | None:
    """Процентиль по ближайшему рангу, без интерполяции."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(share * len(ordered)))
    return ordered[index]


def _cost_of(raw: dict[str, Any]) -> float | None:
    """Стоимость записи; у полного режима B она лежит по кандидатам."""
    if isinstance(raw.get("cost_usd"), int | float):
        return float(raw["cost_usd"])
    candidates = raw.get("candidates")
    if isinstance(candidates, list):
        return sum(float(c.get("cost_usd") or 0.0) for c in candidates)
    return None


def summarize(name: str, rows: list[dict[str, Any]]) -> list[Any]:
    """Строка отчёта по прогону: записи, стоимость, доля вложенных вызовов, доля кэша, p50 и p95 длительности."""
    costs = [cost for raw in rows if (cost := _cost_of(raw)) is not None]
    durations = [raw["duration_ms"] / 1000 for raw in rows if raw.get("duration_ms")]
    usages = [raw["usage"] for raw in rows if isinstance(raw.get("usage"), dict)]
    nested = [float(raw["nested_cost_usd"]) for raw in rows if raw.get("nested_cost_usd")]

    tokens = {kind: sum(usage.get(kind) or 0 for usage in usages) for kind in CACHE_KINDS}
    total_input = sum(tokens.values())
    return [
        name,
        len(rows),
        round(sum(costs), 2) if costs else None,
        round(sum(costs) / len(costs), 4) if costs else None,
        round(sum(nested) / sum(costs), 3) if nested and costs else None,
        round(tokens["cache_read_input_tokens"] / total_input, 3) if total_input else None,
        round(_percentile(durations, 0.5) or 0, 1) if durations else None,
        round(_percentile(durations, 0.95) or 0, 1) if durations else None,
    ]


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run(runs_dir: Path = RUNS_DIR) -> None:
    table = Table(
        "Стоимость сохранённых прогонов",
        ["прогон", "записей", "всего $", "$/запись", "доля B", "из кэша", "p50 с", "p95 с"],
    )
    for path in sorted(runs_dir.glob("*.jsonl")):
        table.rows.append(summarize(path.stem, load(path)))

    print(table.to_text())
    print("\nОценка SDK по прайсу вшитой CLI. Прочерк — поля нет в дампе.")
    write_report(
        "cost",
        "Стоимость прогонов",
        intro=(
            "Стоимость — оценка Claude Agent SDK по прайсу вшитой CLI, не счёт.\n\n"
            "«доля B» — доля стоимости вложенных вызовов модели подтверждения B.\n\n"
            "Прочерк — поля нет в дампе: токены и длительность пишутся в дампы с этапа 8."
        ),
        config={"источник": "evals/runs/*.jsonl", "прогонов": len(table.rows)},
        tables=[table],
    )
