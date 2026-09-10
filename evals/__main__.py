"""CLI харнесса: `python -m evals <сюита>`.

    python -m evals check [--assign] [--categories] [--pairs] [--qa-sample|--qa-answers]  валидация, без LLM
    python -m evals retrieval [--text СОСТАВ...]      recall@k и MRR
    python -m evals qa [--first N] [--verbose]        end-to-end прогон агента, платный
    python -m evals sql [--first N] [--verbose]      text2sql к ClickHouse, платный
    python -m evals routing [--first N] [--verbose]  выбор тула на однохоповых вопросах, платный
    python -m evals matching [--mode] [--repeat N]   матчинг ВБ↔Озон на размеченных парах, платный
    python -m evals judge --of qa|b|c [--split] [--apply]  согласие судьи с истиной, платный
    python -m evals all                              все пять сюит — цель `make eval`

Прогресс и таблицы идут в stdout, отчёт — в `evals/reports/`, метрики — в MLflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crossmarket.config import JUDGE_MODEL, MATCH_MODEL
from crossmarket.embedding import COMPOSITIONS, DEFAULT_COMPOSITION
from evals import check, judges, matching, qa, retrieval, routing
from evals import sql as sql_suite
from evals.cases import GOLDEN_FILE, load_cases


def _run_name(base: str, tag: str | None) -> str:
    """Имя отчёта и дампа. Без тега — основной прогон, с тегом — сравнительный рядом."""
    return f"{base}_{tag}" if tag else base


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
    checker.add_argument("--pairs", action="store_true", help="сплиты размеченных пар ВБ↔Озон")
    checker.add_argument("--qa-answers", action="store_true", help="проверить ручную разметку ответов под судью-QA")
    checker.add_argument("--qa-sample", action="store_true", help="собрать заготовку разметки ответов, не теряя меток")

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

    sqler = suites.add_parser("sql", parents=[common], help="text2sql к ClickHouse (платный)")
    sqler.add_argument("--first", type=int, help="прогнать только первые N вопросов")
    sqler.add_argument("--max-turns", type=int, default=None)
    sqler.add_argument("--budget", type=float, default=None)
    sqler.add_argument("--verbose", action="store_true", help="печатать SQL агента и ответы")
    sqler.add_argument("--tag", help="суффикс отчёта и дампа")

    matcher = suites.add_parser("matching", parents=[common], help="матчинг ВБ↔Озон на размеченных парах (платный)")
    matcher.add_argument("--mode", choices=("pair", "full", "both"), default="pair", help="какой режим B мерить")
    matcher.add_argument("--split", choices=("test", "calibration", "all"), default="test")
    matcher.add_argument("--repeat", type=int, default=1, help="прогонов вырожденного режима для усреднения")
    matcher.add_argument("--model", default=MATCH_MODEL, help="модель подтверждения")
    matcher.add_argument("--first", type=int, help="прогнать только первые N пар")
    matcher.add_argument("--tag", help="суффикс отчёта и дампа")

    judger = suites.add_parser("judge", parents=[common], help="согласие судьи с истиной (платный)")
    judger.add_argument("--of", dest="judge", choices=tuple(judges.JUDGES), required=True, help="какой судья")
    judger.add_argument("--split", choices=("test", "calibration", "all"), default="test")
    judger.add_argument("--model", default=JUDGE_MODEL, help="модель судьи")
    judger.add_argument("--first", type=int, help="прогнать только первые N кейсов")
    judger.add_argument("--tag", help="суффикс отчёта и дампа")
    judger.add_argument(
        "--apply",
        action="store_true",
        help="судья-B на карточках вне разметки: истины нет, цифр согласия нет",
    )

    router = suites.add_parser("routing", parents=[common], help="выбор тула на однохоповых вопросах (платный)")
    router.add_argument("--first", type=int, help="прогнать только первые N вопросов")
    router.add_argument("--max-turns", type=int, default=None)
    router.add_argument("--budget", type=float, default=None)
    router.add_argument("--verbose", action="store_true", help="печатать ответы целиком")
    router.add_argument("--tag", help="суффикс отчёта и дампа")

    regrader = suites.add_parser("regrade", parents=[common], help="пересчитать метрики сохранённого прогона, без LLM")
    regrader.add_argument(
        "--of",
        dest="regrade_suite",
        choices=("qa", "sql", "routing", "matching", "judge-qa", "judge-b", "judge-c"),
        default="qa",
        help="какую сюиту пересчитать",
    )
    regrader.add_argument("--tag", help="какой прогон пересчитать")
    suites.add_parser("all", parents=[common], help="retrieval + qa + sql + routing + matching, корпус с фоном")

    args = parser.parse_args()
    use_mlflow = not args.no_mlflow

    if args.suite == "check":
        raise SystemExit(
            check.run(
                args.golden,
                assign=args.assign,
                categories=args.categories,
                pairs=args.pairs,
                qa_answers=args.qa_answers,
                qa_sample=args.qa_sample,
            )
        )

    if args.suite == "regrade":
        modules = {
            "sql": (sql_suite, "sql_c"),
            "routing": (routing, "routing"),
            "qa": (qa, "qa_wb"),
            "matching": (matching, "matching_b"),
            "judge-qa": (judges, "judge_qa"),
            "judge-b": (judges, "judge_b"),
            "judge-c": (judges, "judge_c"),
        }
        module, base = modules[args.regrade_suite]
        module.regrade(use_mlflow, name=_run_name(base, args.tag))
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
            name=_run_name("qa_wb", getattr(args, "tag", None)),
        )

    if args.suite in ("sql", "all"):
        from crossmarket.agent import instrument_langfuse
        from crossmarket.config import AGENT_MAX_BUDGET_USD, AGENT_MAX_TURNS
        from evals.cases import SQL_GOLDEN_FILE, load_sql_cases

        first = getattr(args, "first", None)
        cases = load_sql_cases(SQL_GOLDEN_FILE)
        cases = cases[:first] if first else cases
        traced = "включён" if instrument_langfuse() else "нет ключей"
        print(f"\nSQL-вопросов {len(cases)}, трейсинг Langfuse: {traced}\n")
        sql_suite.run(
            cases,
            limit_turns=getattr(args, "max_turns", None) or AGENT_MAX_TURNS,
            budget=getattr(args, "budget", None) or AGENT_MAX_BUDGET_USD,
            verbose=getattr(args, "verbose", False),
            use_mlflow=use_mlflow,
            name=_run_name("sql_c", getattr(args, "tag", None)),
        )

    if args.suite in ("matching", "all"):
        from crossmarket.agent import instrument_langfuse
        from evals.pairs import load_pairs

        split = getattr(args, "split", "test")
        pairs = [pair for pair in load_pairs() if split == "all" or pair.split == split]
        first = getattr(args, "first", None)
        pairs = pairs[:first] if first else pairs
        mode = getattr(args, "mode", "pair")
        model = getattr(args, "model", MATCH_MODEL)
        traced = "включён" if instrument_langfuse() else "нет ключей"
        print(f"\nПар {len(pairs)} (сплит {split}), режим {mode}, трейсинг Langfuse: {traced}\n")
        matching.run(
            pairs,
            mode=mode,
            model=model,
            repeat=getattr(args, "repeat", 1),
            use_mlflow=use_mlflow,
            split=split,
            name=_run_name("matching_b", getattr(args, "tag", None)),
        )

    if args.suite == "judge":
        from crossmarket.agent import instrument_langfuse

        print(f"\nтрейсинг Langfuse: {'включён' if instrument_langfuse() else 'нет ключей'}")
        if args.apply:
            judges.apply_run(model=args.model, first=args.first, name=_run_name("judge_b_apply", args.tag))
            return
        judges.run(
            args.judge,
            split=args.split,
            model=args.model,
            first=args.first,
            use_mlflow=use_mlflow,
            name=_run_name(f"judge_{args.judge}", args.tag),
        )
        return

    if args.suite in ("routing", "all"):
        from crossmarket.agent import instrument_langfuse
        from crossmarket.config import AGENT_MAX_BUDGET_USD, AGENT_MAX_TURNS
        from evals.cases import ROUTING_GOLDEN_FILE, load_routing_cases

        first = getattr(args, "first", None)
        cases = load_routing_cases(ROUTING_GOLDEN_FILE)
        cases = cases[:first] if first else cases
        traced = "включён" if instrument_langfuse() else "нет ключей"
        print(f"\nВопросов роутинга {len(cases)}, трейсинг Langfuse: {traced}\n")
        routing.run(
            cases,
            limit_turns=getattr(args, "max_turns", None) or AGENT_MAX_TURNS,
            budget=getattr(args, "budget", None) or AGENT_MAX_BUDGET_USD,
            verbose=getattr(args, "verbose", False),
            use_mlflow=use_mlflow,
            name=_run_name("routing", getattr(args, "tag", None)),
        )


main()
