"""CLI харнесса: `python -m evals <сюита>`.

    python -m evals check [--assign] [--categories]   валидация golden-set, без LLM
    python -m evals retrieval [--text СОСТАВ...]      recall@k и MRR
    python -m evals qa [--first N] [--verbose]        end-to-end прогон агента, платный
    python -m evals all                              обе сюиты подряд — цель `make eval`

Прогресс и таблицы идут в stdout, отчёт — в `evals/reports/`, метрики — в MLflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crossmarket.embedding import COMPOSITIONS, DEFAULT_COMPOSITION
from evals import check, qa, retrieval
from evals.cases import GOLDEN_FILE, load_cases


def _run_name(tag: str | None) -> str:
    """Имя отчёта и дампа. Без тега — основной прогон, с тегом — сравнительный рядом."""
    return f"qa_wb_{tag}" if tag else "qa_wb"


def _cases(path: Path, first: int | None = None, expected: str | None = None) -> list:
    cases = load_cases(path, expected=expected)
    return cases[:first] if first else cases


def main() -> None:
    # Построчная выдача: под конвейером (`make eval | tee`, запись gif) Python буферизует
    # stdout блоками, и прогресс появляется рывками в конце вместо хода прогона.
    sys.stdout.reconfigure(line_buffering=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    common.add_argument("--no-mlflow", action="store_true", help="только stdout и отчёт")

    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__, parents=[common])
    suites = parser.add_subparsers(dest="suite", required=True)

    checker = suites.add_parser("check", parents=[common], help="проверить golden-set, не тратя денег")
    checker.add_argument("--assign", action="store_true", help="проставить сплит новым кейсам")
    checker.add_argument("--categories", action="store_true", help="напечатать категории корпуса")

    finder = suites.add_parser("retrieval", parents=[common], help="recall@k и MRR")
    finder.add_argument("--limit", type=int, default=max(retrieval.K_VALUES), help="глубина выдачи")
    finder.add_argument(
        "--text",
        nargs="+",
        choices=sorted(COMPOSITIONS),
        default=[DEFAULT_COMPOSITION],
        help="составы текста; каждый переиндексируется и логируется отдельным прогоном",
    )
    finder.add_argument("--distractors", action="store_true", help="добить корпус придуманным фоном")

    asker = suites.add_parser("qa", parents=[common], help="end-to-end прогон агента (платный)")
    asker.add_argument("--first", type=int, help="прогнать только первые N вопросов")
    asker.add_argument("--max-turns", type=int, default=None)
    asker.add_argument("--budget", type=float, default=None)
    asker.add_argument("--verbose", action="store_true", help="печатать ответы, рассуждения и запросы")
    asker.add_argument("--tag", help="суффикс отчёта и дампа: сравнительный прогон не затрёт основной")

    regrader = suites.add_parser("regrade", parents=[common], help="пересчитать метрики сохранённого прогона, без LLM")
    regrader.add_argument("--tag", help="какой прогон пересчитать")
    suites.add_parser("all", parents=[common], help="retrieval + qa, корпус с фоном")

    args = parser.parse_args()
    use_mlflow = not args.no_mlflow

    if args.suite == "check":
        raise SystemExit(check.run(args.golden, assign=args.assign, categories=args.categories))

    if args.suite == "regrade":
        qa.regrade(use_mlflow, name=_run_name(args.tag))
        return

    if args.suite in ("retrieval", "all"):
        compositions = getattr(args, "text", [DEFAULT_COMPOSITION])
        retrieval.run(
            _cases(args.golden),
            limit=getattr(args, "limit", max(retrieval.K_VALUES)),
            compositions=compositions,
            with_distractors=getattr(args, "distractors", True),
            use_mlflow=use_mlflow,
        )

    if args.suite in ("qa", "all"):
        from crossmarket.agent import instrument_langfuse
        from crossmarket.config import AGENT_MAX_BUDGET_USD, AGENT_MAX_TURNS

        cases = _cases(args.golden, first=getattr(args, "first", None))
        print(f"\nвопросов {len(cases)}, трейсинг Langfuse: {'включён' if instrument_langfuse() else 'нет ключей'}\n")
        qa.run(
            cases,
            limit_turns=getattr(args, "max_turns", None) or AGENT_MAX_TURNS,
            budget=getattr(args, "budget", None) or AGENT_MAX_BUDGET_USD,
            verbose=getattr(args, "verbose", False),
            use_mlflow=use_mlflow,
            name=_run_name(getattr(args, "tag", None)),
        )


main()
