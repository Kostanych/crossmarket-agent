"""Сюита B: матчинг ВБ↔Озон против размеченных пар. Два режима, метрики раздельные.

**Вырожденный** (`--mode pair`) — precision / recall / F1 по размеченным парам, один
вызов модели подтверждения на пару. Отчётная цифра снимается с тест-сплита и
усредняется несколькими прогонами: у метрики есть разброс при неизменных промпте и
данных.

**Полный** (`--mode full`) — карточка ВБ → кандидаты с Озона → подтверждение по
каждому. Отсюда же считается исход «аналога нет»: у матч-пары партнёр известен, поэтому
остальные кандидаты — заведомо не-матчи, и по ним видно, вернул бы B пустой результат.
Отдельного прогона это не требует.

Подтверждённые не-партнёры печатаются целиком и читаются глазами: истины по ним нет.

Свип порога считается по сохранённому прогону — скор кандидата лежит рядом с вердиктом.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossmarket.config import MATCH_CANDIDATE_LIMIT, MATCH_MIN_SCORE, MATCH_MODEL
from crossmarket.matching import Candidate, MatchResult, Verdict, confirm, confirm_options, match
from crossmarket.models import Label, Product
from crossmarket.storage import clickhouse, qdrant
from evals.report import Table, print_tables, write_report

RUNS_DIR = Path("evals/runs")

SWEEP_THRESHOLDS = (0.0, 0.80, 0.83, 0.85, 0.87, 0.90)
"""Отсечки для свипа. Диапазон по замеру: скор партнёра 0.831–0.950, топ-1 у товаров
без размеченной пары 0.800–0.925."""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --- вырожденный режим -------------------------------------------------------


@dataclass
class PairRow:
    """Пара, её метка и вердикт модели."""

    pair: Label
    verdict: Verdict
    run: int = 1

    @property
    def correct(self) -> bool:
        return self.verdict.said == self.pair.label


def confusion(rows: list[PairRow]) -> dict[str, int]:
    """Матрица ошибок, положительный класс — `match`."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unparsed": 0}
    for row in rows:
        said = row.verdict.said
        if said is None:
            counts["unparsed"] += 1
        elif said == "match":
            counts["tp" if row.pair.label == "match" else "fp"] += 1
        else:
            counts["tn" if row.pair.label == "no_match" else "fn"] += 1
    return counts


def pair_metrics(rows: list[PairRow]) -> dict[str, float]:
    counts = confusion(rows)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": _mean([float(row.correct) for row in rows]),
        "negatives_correct": tn / (tn + fp) if tn + fp else 0.0,
        "unparsed": counts["unparsed"] / len(rows) if rows else 0.0,
        "total_cost_usd": sum(row.verdict.cost_usd for row in rows),
    }


async def run_pairs(pairs: list[Label], options: Any, run: int, concurrency: int = 4) -> list[PairRow]:
    client = clickhouse.connect()
    keys = [("wb", pair.wb_id) for pair in pairs] + [("ozon", pair.ozon_id) for pair in pairs]
    cards = clickhouse.fetch_products(client, keys)

    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def one(pair: Label) -> PairRow:
        nonlocal done
        async with gate:
            verdict = await confirm(cards[("wb", pair.wb_id)], cards[("ozon", pair.ozon_id)], options)
        row = PairRow(pair=pair, verdict=verdict, run=run)
        done += 1
        print(f"{done:>3}/{len(pairs)} {'+' if row.correct else 'MISS':<5} {pair.id} truth={pair.label}")
        return row

    return list(await asyncio.gather(*(one(pair) for pair in pairs)))


# --- полный режим ------------------------------------------------------------


@dataclass
class FullRow:
    """Матч-пара, прогнанная полным режимом, и что B по ней вернул."""

    pair: Label
    result: MatchResult

    @property
    def partner(self) -> Candidate | None:
        """Размеченный партнёр среди кандидатов. `None` — retrieval его не достал."""
        return next((c for c in self.result.candidates if c.product.id == self.pair.ozon_id), None)

    @property
    def others(self) -> list[Candidate]:
        """Кандидаты кроме размеченного партнёра — заведомо не-матчи."""
        return [c for c in self.result.candidates if c.product.id != self.pair.ozon_id]

    @property
    def found_top1(self) -> bool:
        confirmed = self.result.confirmed
        return bool(confirmed) and confirmed[0].product.id == self.pair.ozon_id

    @property
    def found_any(self) -> bool:
        return any(c.product.id == self.pair.ozon_id for c in self.result.confirmed)

    @property
    def no_analog_ok(self) -> bool:
        """Вернул бы B пустой результат, если убрать из кандидатов партнёра."""
        return not any(c.matched for c in self.others)


def full_metrics(rows: list[FullRow]) -> dict[str, float]:
    others = [c for row in rows for c in row.others]
    return {
        "partner_retrieved": _mean([float(row.partner is not None) for row in rows]),
        "found_top1": _mean([float(row.found_top1) for row in rows]),
        "found_any": _mean([float(row.found_any) for row in rows]),
        "confirmed_avg": _mean([float(len(row.result.confirmed)) for row in rows]),
        "no_analog_ok": _mean([float(row.no_analog_ok) for row in rows]),
        "false_confirm_rate": _mean([float(c.matched) for c in others]),
        "candidates_avg": _mean([float(len(row.result.candidates)) for row in rows]),
        "total_cost_usd": sum(row.result.cost_usd for row in rows),
    }


async def run_full(
    pairs: list[Label],
    options: Any,
    limit: int = MATCH_CANDIDATE_LIMIT,
    min_score: float = MATCH_MIN_SCORE,
    concurrency: int = 2,
) -> list[FullRow]:
    qdrant_client = qdrant.connect()
    clickhouse_client = clickhouse.connect()
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def one(pair: Label) -> FullRow:
        nonlocal done
        async with gate:
            result = await match(
                pair.wb_id,
                qdrant_client,
                clickhouse_client,
                options=options,
                limit=limit,
                min_score=min_score,
            )
        row = FullRow(pair=pair, result=result)
        done += 1
        found = "+" if row.found_top1 else ("~" if row.found_any else "MISS")
        print(
            f"{done:>3}/{len(pairs)} {found:<5} {pair.id} "
            f"candidates {len(result.candidates)}, confirmed {len(result.confirmed)}"
        )
        return row

    return list(await asyncio.gather(*(one(pair) for pair in pairs)))


def sweep(rows: list[FullRow], thresholds: tuple[float, ...] = SWEEP_THRESHOLDS) -> Table:
    """Что было бы при отсечке кандидатов по скору, по готовым вердиктам прогона.

    Отсечка выкидывает кандидата до модели, поэтому вердикты остальных от неё не
    зависят и пересчёт точен.
    """
    table = Table(
        "Candidate threshold sweep (over saved verdicts)",
        ["threshold", "candidates per product", "partner reached", "found@1", '"no counterpart" correct'],
    )
    for threshold in thresholds:
        kept = [(row, [c for c in row.result.candidates if (c.score or 0.0) >= threshold]) for row in rows]
        confirmed = [[c for c in cands if c.matched] for _, cands in kept]
        table.rows.append(
            [
                f"{threshold:.2f}",
                _mean([float(len(cands)) for _, cands in kept]),
                _mean([float(any(c.product.id == row.pair.ozon_id for c in cands)) for row, cands in kept]),
                _mean(
                    [
                        float(bool(hits) and hits[0].product.id == row.pair.ozon_id)
                        for (row, _), hits in zip(kept, confirmed, strict=True)
                    ]
                ),
                _mean(
                    [
                        float(all(c.product.id == row.pair.ozon_id for c in hits))
                        for (row, _), hits in zip(kept, confirmed, strict=True)
                    ]
                ),
            ]
        )
    return table


# --- таблицы -----------------------------------------------------------------


def matrix_table(rows: list[PairRow]) -> Table:
    counts = confusion(rows)
    return Table(
        "Confusion matrix (positive class — match)",
        ["", "model: match", "model: no_match", "unparsed"],
        [
            ["truth: match", counts["tp"], counts["fn"], "—"],
            ["truth: no_match", counts["fp"], counts["tn"], "—"],
            ["total", counts["tp"] + counts["fp"], counts["fn"] + counts["tn"], counts["unparsed"]],
        ],
    )


def pair_summary_table(rows: list[PairRow], model: str) -> Table:
    """Метрики по каждому прогону плюс строки среднего и разброса."""
    table = Table(
        f"Pair mode, confirmation model {model}",
        ["run", "pairs", "precision", "recall", "F1", "accuracy", "negatives correct", "$"],
    )
    runs = sorted({row.run for row in rows})
    per_run = []
    for run in runs:
        group = [row for row in rows if row.run == run]
        metrics = pair_metrics(group)
        per_run.append(metrics)
        table.rows.append(
            [
                f"#{run}",
                len(group),
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["accuracy"],
                metrics["negatives_correct"],
                f"{metrics['total_cost_usd']:.2f}",
            ]
        )
    if len(runs) > 1:
        names = ("precision", "recall", "f1", "accuracy", "negatives_correct")
        table.rows.append(
            ["mean", len(rows) // len(runs)]
            + [_mean([m[name] for m in per_run]) for name in names]
            + [f"{sum(m['total_cost_usd'] for m in per_run):.2f}"]
        )
        table.rows.append(
            ["range", "—"]
            + [f"{min(m[name] for m in per_run):.3f}–{max(m[name] for m in per_run):.3f}" for name in names]
            + ["—"]
        )
    return table


def misses_table(rows: list[PairRow]) -> Table | None:
    """Расхождения с разметкой. Только в stdout — в обосновании данные карточек."""
    table = Table(
        "Disagreements with labels",
        ["pair", "run", "truth", "model", "model reason"],
        [
            [row.pair.id, f"#{row.run}", row.pair.label, row.verdict.said or "—", row.verdict.reason]
            for row in rows
            if not row.correct
        ],
    )
    return table if table.rows else None


def full_summary_table(rows: list[FullRow], model: str) -> Table:
    metrics = full_metrics(rows)
    return Table(
        f"Full mode, confirmation model {model}",
        ["products", "candidates", "partner reached", "found@1", "found_any", "confirmed", "$"],
        [
            [
                len(rows),
                metrics["candidates_avg"],
                metrics["partner_retrieved"],
                metrics["found_top1"],
                metrics["found_any"],
                metrics["confirmed_avg"],
                f"{metrics['total_cost_usd']:.2f}",
            ]
        ],
    )


def no_analog_table(rows: list[FullRow]) -> Table:
    """Исход «аналога нет» по кандидатам кроме размеченного партнёра."""
    metrics = full_metrics(rows)
    others = [c for row in rows for c in row.others]
    return Table(
        '"No counterpart" outcome (partner removed from candidates)',
        ["products", "non-partner candidates", "would return empty", "false confirmations per candidate"],
        [[len(rows), len(others), metrics["no_analog_ok"], metrics["false_confirm_rate"]]],
    )


def false_confirms_table(rows: list[FullRow]) -> Table | None:
    """Подтверждённые не-партнёры, для проверки глазами: истины по ним нет.

    Только в stdout — здесь названия настоящих карточек.
    """
    table = Table(
        "Confirmed non-partners (check by eye)",
        ["pair", "WB product", "Ozon candidate", "score", "model reason"],
    )
    for row in rows:
        for candidate in row.others:
            if not candidate.matched:
                continue
            table.rows.append(
                [
                    row.pair.id,
                    row.result.wb.title if row.result.wb else "—",
                    f"{candidate.product.id} {candidate.product.title}",
                    f"{candidate.score:.3f}" if candidate.score is not None else "—",
                    candidate.verdict.reason if candidate.verdict else "",
                ]
            )
    return table if table.rows else None


# --- дамп и перегрейд --------------------------------------------------------


def dump_run(pair_rows: list[PairRow], full_rows: list[FullRow], name: str, meta: dict[str, Any]) -> Path:
    """Прогон на диск для `regrade` и свипа. В гит не идёт — данные карточек."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in pair_rows:
            fh.write(
                json.dumps(
                    {
                        "kind": "pair",
                        "run": row.run,
                        "wb_id": row.pair.wb_id,
                        "ozon_id": row.pair.ozon_id,
                        "label": row.pair.label,
                        "negative_kind": row.pair.negative_kind,
                        "split": row.pair.split,
                        "said": row.verdict.said,
                        "reason": row.verdict.reason,
                        "cost_usd": row.verdict.cost_usd,
                        **meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for full in full_rows:
            fh.write(
                json.dumps(
                    {
                        "kind": "full",
                        "wb_id": full.pair.wb_id,
                        "ozon_id": full.pair.ozon_id,
                        "label": full.pair.label,
                        "split": full.pair.split,
                        "wb_title": full.result.wb.title if full.result.wb else "",
                        "candidates": [
                            {
                                "id": c.product.id,
                                "title": c.product.title,
                                "score": c.score,
                                "said": c.verdict.said if c.verdict else None,
                                "reason": c.verdict.reason if c.verdict else "",
                                "cost_usd": c.verdict.cost_usd if c.verdict else 0.0,
                            }
                            for c in full.result.candidates
                        ],
                        **meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_run(name: str) -> tuple[list[PairRow], list[FullRow], dict[str, Any]]:
    """Восстановить прогон с диска. LLM не зовётся.

    Карточки восстанавливаются урезанными, только id и название: остальных полей в
    дампе нет.
    """
    pair_rows: list[PairRow] = []
    full_rows: list[FullRow] = []
    meta: dict[str, Any] = {}
    for line in (RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        meta = {key: raw[key] for key in ("model", "limits", "split") if key in raw}
        label = Label(
            wb_id=raw["wb_id"],
            ozon_id=raw["ozon_id"],
            label=raw["label"],
            negative_kind=raw.get("negative_kind"),
            split=raw.get("split"),
        )
        if raw["kind"] == "pair":
            pair_rows.append(
                PairRow(
                    pair=label,
                    verdict=Verdict(said=raw["said"], reason=raw["reason"], cost_usd=raw["cost_usd"]),
                    run=raw.get("run", 1),
                )
            )
        else:
            candidates = [
                Candidate(
                    product=Product(marketplace="ozon", id=c["id"], title=c["title"]),
                    score=c["score"],
                    verdict=Verdict(said=c["said"], reason=c["reason"], cost_usd=c["cost_usd"]),
                )
                for c in raw["candidates"]
            ]
            result = MatchResult(
                mode="full",
                wb=Product(marketplace="wb", id=raw["wb_id"], title=raw.get("wb_title", "")),
                candidates=candidates,
            )
            full_rows.append(FullRow(pair=label, result=result))
    return pair_rows, full_rows, meta


# --- прогон и отчёт ----------------------------------------------------------


def log_to_mlflow(pair_rows: list[PairRow], full_rows: list[FullRow], meta: dict[str, Any]) -> None:
    import mlflow

    metrics: dict[str, float] = {}
    if pair_rows:
        runs = sorted({row.run for row in pair_rows})
        per_run = [pair_metrics([r for r in pair_rows if r.run == run]) for run in runs]
        metrics |= {name: _mean([m[name] for m in per_run]) for name in per_run[0] if name != "total_cost_usd"}
        metrics["runs"] = float(len(runs))
        if len(runs) > 1:
            metrics["f1_min"] = min(m["f1"] for m in per_run)
            metrics["f1_max"] = max(m["f1"] for m in per_run)
    if full_rows:
        metrics |= {f"full_{name}": value for name, value in full_metrics(full_rows).items()}

    mlflow.set_experiment("matching_b")
    with mlflow.start_run(run_name=str(meta.get("model", MATCH_MODEL))):
        mlflow.log_params(meta | {"pairs": len(pair_rows), "full_products": len(full_rows)})
        mlflow.log_metrics(metrics)


UNPARSED_ALARM = 0.1
"""Доля неразобранных вердиктов, выше которой прогон помечается недействительным.

Неразобранный вердикт подтверждением не считается, то есть пустой ответ модели попадает
в метрику как «не матч». Упёршийся лимит провайдера так рисует осмысленную на вид
таблицу: прогон haiku с 89 пустыми ответами из 92 дал precision 1.000."""


def unparsed_warning(pair_rows: list[PairRow], full_rows: list[FullRow]) -> str:
    """Текст предупреждения, если неразобранных вердиктов больше `UNPARSED_ALARM`."""
    verdicts = [row.verdict for row in pair_rows]
    verdicts += [c.verdict for row in full_rows for c in row.result.candidates if c.verdict is not None]
    if not verdicts:
        return ""
    unparsed = sum(1 for verdict in verdicts if verdict.said is None)
    if unparsed <= UNPARSED_ALARM * len(verdicts):
        return ""
    return (
        f"WARNING: {unparsed} of {len(verdicts)} verdicts unparsed — the run's numbers are "
        "invalid. An empty model answer counts as no_match; the usual cause is a hit "
        "provider limit, not the answer format."
    )


def report(
    pair_rows: list[PairRow],
    full_rows: list[FullRow],
    meta: dict[str, Any],
    use_mlflow: bool,
    dump: Path,
    name: str,
) -> None:
    """Таблицы в stdout, markdown-отчёт, метрики в MLflow. Общее для прогона и перегрейда."""
    model = str(meta.get("model", MATCH_MODEL))
    aggregates: list[Table] = []
    screen: list[Table] = []

    if pair_rows:
        aggregates += [pair_summary_table(pair_rows, model), matrix_table(pair_rows)]
        screen += aggregates[-2:]
        if misses := misses_table(pair_rows):
            screen.append(misses)
    if full_rows:
        full_tables = [full_summary_table(full_rows, model), no_analog_table(full_rows), sweep(full_rows)]
        aggregates += full_tables
        screen += full_tables
        if false_confirms := false_confirms_table(full_rows):
            screen.append(false_confirms)

    print_tables(screen)
    total = sum(row.verdict.cost_usd for row in pair_rows) + sum(row.result.cost_usd for row in full_rows)
    print(f"\nconfirmation model {model}, run ${total:.2f}")
    if warning := unparsed_warning(pair_rows, full_rows):
        print(warning)

    path = write_report(
        name,
        "WB↔Ozon matching: confirmation model on labeled pairs",
        "Pair mode is the direct metric: precision / recall / F1 against pair labels. "
        "The reported number is the mean row on the test split: the metric varies from run "
        "to run with the same prompt and data. Full mode measures the whole path "
        'WB listing → candidates → confirmation; the "no counterpart" outcome is computed from the same '
        "verdicts by removing the labeled partner from the candidates. Confirmed non-partners "
        "are not counted as errors automatically — there is no ground truth for them, they are read by eye.",
        {
            "confirmation model": model,
            "split": meta.get("split", "—"),
            "limits": meta.get("limits", "—"),
            "pairs in pair mode": len(pair_rows),
            "products in full mode": len(full_rows),
            "run cost": f"${total:.2f}",
        },
        aggregates,
    )
    print(
        f"Report: {path}, raw run: {dump}"
        + ("" if not use_mlflow else "\nMetrics logged to MLflow: experiment matching_b")
    )
    if use_mlflow:
        log_to_mlflow(pair_rows, full_rows, meta)


def run(
    pairs: list[Label],
    mode: str,
    model: str,
    repeat: int,
    use_mlflow: bool,
    split: str = "test",
    limit: int = MATCH_CANDIDATE_LIMIT,
    min_score: float = MATCH_MIN_SCORE,
    name: str = "matching_b",
) -> None:
    options = confirm_options(model)
    pair_rows: list[PairRow] = []
    full_rows: list[FullRow] = []

    if mode in ("pair", "both"):
        for number in range(1, repeat + 1):
            print(f"\npair mode, run {number}/{repeat}, pairs {len(pairs)}\n")
            pair_rows += asyncio.run(run_pairs(pairs, options, number))
    if mode in ("full", "both"):
        matches = [pair for pair in pairs if pair.label == "match"]
        print(f"\nfull mode, products {len(matches)}, up to {limit} candidates per product\n")
        full_rows += asyncio.run(run_full(matches, options, limit, min_score))

    meta = {
        "model": model,
        "split": split,
        "limits": f"candidates={limit}, min_score={min_score}, repeat={repeat}",
    }
    report(pair_rows, full_rows, meta, use_mlflow, dump_run(pair_rows, full_rows, name, meta), name)


def regrade(use_mlflow: bool = False, name: str = "matching_b") -> None:
    """Пересчитать метрики сохранённого прогона новым разбором. LLM не зовётся."""
    pair_rows, full_rows, meta = load_run(name)
    print(f"regrading {len(pair_rows)} pairs and {len(full_rows)} products from {RUNS_DIR / f'{name}.jsonl'}\n")
    report(pair_rows, full_rows, meta, use_mlflow, RUNS_DIR / f"{name}.jsonl", name)
