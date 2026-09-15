"""Сводка стоимости по дампам прогонов."""

from __future__ import annotations

from evals.cost import summarize


def test_old_runs_keep_cost_and_admit_missing_fields() -> None:
    """Дамп только со стоимостью: остальные колонки пустые."""
    row = summarize("qa_wb", [{"cost_usd": 0.06}, {"cost_usd": 0.08}])

    assert row[:4] == ["qa_wb", 2, 0.14, 0.07]
    assert row[4:] == [None, None, None, None]


def test_cache_share_and_nested_cost() -> None:
    rows = [
        {
            "cost_usd": 0.10,
            "nested_cost_usd": 0.09,
            "duration_ms": 10_000,
            "usage": {"input_tokens": 100, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 100},
        },
        {"cost_usd": 0.10, "duration_ms": 20_000, "usage": {"cache_read_input_tokens": 1000}},
    ]

    name, count, total, per_row, nested_share, cache_share, p50, p95 = summarize("routing", rows)

    assert (count, total, per_row) == (2, 0.2, 0.1)
    assert nested_share == 0.45
    assert cache_share == 0.9
    assert (p50, p95) == (20.0, 20.0)


def test_full_mode_cost_lives_in_candidates() -> None:
    """У полного режима B стоимость суммируется по кандидатам."""
    row = summarize("matching_b_full", [{"candidates": [{"cost_usd": 0.02}, {"cost_usd": 0.03}]}])

    assert row[2] == 0.05
