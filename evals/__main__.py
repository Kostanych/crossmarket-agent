"""CLI харнесса: `python -m evals <сюита>`.

    python -m evals check [--assign] [--categories] [--pairs] [--qa-sample|--qa-answers]  валидация, без LLM
    python -m evals retrieval [--text СОСТАВ...]      recall@k и MRR
    python -m evals qa [--first N] [--verbose]        end-to-end прогон агента, платный
    python -m evals sql [--first N] [--verbose]      text2sql к ClickHouse, платный
    python -m evals routing [--first N] [--verbose]  выбор тула на однохоповых вопросах, платный
    python -m evals matching [--mode] [--repeat N]   матчинг ВБ↔Озон на размеченных парах, платный
    python -m evals judge --of qa|b|c [--split] [--apply]  согласие судьи с истиной, платный
    python -m evals all                              все пять сюит — цель `make eval`
    python -m evals cost                             стоимость сохранённых прогонов, без LLM

Прогресс и таблицы идут в stdout, отчёт — в `evals/reports/`, метрики — в MLflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crossmarket.config import JUDGE_MODEL, MATCH_MODEL
from crossmarket.embedding import COMPOSITIONS, DEFAULT_COMPOSITION
from evals import check, cost, judges, matching, qa, retrieval, routing
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
    common.add_argument("--no-mlflow", action="store_true", help="stdout and report only")

    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__, parents=[common])
    suites = parser.add_subparsers(dest="suite", required=True)

    checker = suites.add_parser("check", parents=[common], help="validate the golden set, free")
    checker.add_argument("--assign", action="store_true", help="assign a split to new cases")
    checker.add_argument("--categories", action="store_true", help="print corpus categories")
    checker.add_argument("--pairs", action="store_true", help="splits of labeled WB↔Ozon pairs")
    checker.add_argument("--qa-answers", action="store_true", help="validate manual answer labels for the QA judge")
    checker.add_argument("--qa-sample", action="store_true", help="build an answer labeling draft, keep labels")

    finder = suites.add_parser("retrieval", parents=[common], help="recall@k and MRR")
    finder.add_argument("--limit", type=int, default=max(retrieval.K_VALUES), help="search depth")
    finder.add_argument(
        "--text",
        nargs="+",
        choices=sorted(COMPOSITIONS),
        default=[DEFAULT_COMPOSITION],
        help="text compositions; each is reindexed and logged as a separate run",
    )
    finder.add_argument("--distractors", action="store_true", help="add made-up background listings to the corpus")

    asker = suites.add_parser("qa", parents=[common], help="end-to-end agent run (paid)")
    asker.add_argument("--first", type=int, help="run only the first N questions")
    asker.add_argument("--max-turns", type=int, default=None)
    asker.add_argument("--budget", type=float, default=None)
    asker.add_argument("--verbose", action="store_true", help="print answers, reasoning and queries")
    asker.add_argument("--tag", help="report and dump suffix: a comparison run will not overwrite the main one")

    sqler = suites.add_parser("sql", parents=[common], help="text2sql over ClickHouse (paid)")
    sqler.add_argument("--first", type=int, help="run only the first N questions")
    sqler.add_argument("--max-turns", type=int, default=None)
    sqler.add_argument("--budget", type=float, default=None)
    sqler.add_argument("--verbose", action="store_true", help="print agent SQL and answers")
    sqler.add_argument("--tag", help="report and dump suffix")

    matcher = suites.add_parser("matching", parents=[common], help="WB↔Ozon matching on labeled pairs (paid)")
    matcher.add_argument("--mode", choices=("pair", "full", "both"), default="pair", help="which B mode to measure")
    matcher.add_argument("--split", choices=("test", "calibration", "all"), default="test")
    matcher.add_argument("--repeat", type=int, default=1, help="pair mode runs to average")
    matcher.add_argument("--model", default=MATCH_MODEL, help="confirmation model")
    matcher.add_argument("--first", type=int, help="run only the first N pairs")
    matcher.add_argument("--tag", help="report and dump suffix")

    judger = suites.add_parser("judge", parents=[common], help="judge agreement with ground truth (paid)")
    judger.add_argument("--of", dest="judge", choices=tuple(judges.JUDGES), required=True, help="which judge")
    judger.add_argument("--split", choices=("test", "calibration", "all"), default="test")
    judger.add_argument("--model", default=JUDGE_MODEL, help="judge model")
    judger.add_argument("--first", type=int, help="run only the first N cases")
    judger.add_argument("--tag", help="report and dump suffix")
    judger.add_argument(
        "--apply",
        action="store_true",
        help="B judge on unlabeled listings: no ground truth, no agreement numbers",
    )

    router = suites.add_parser("routing", parents=[common], help="tool choice on single-hop questions (paid)")
    router.add_argument("--first", type=int, help="run only the first N questions")
    router.add_argument("--max-turns", type=int, default=None)
    router.add_argument("--budget", type=float, default=None)
    router.add_argument("--verbose", action="store_true", help="print full answers")
    router.add_argument("--tag", help="report and dump suffix")

    regrader = suites.add_parser("regrade", parents=[common], help="recompute metrics of a saved run, no LLM")
    regrader.add_argument(
        "--of",
        dest="regrade_suite",
        choices=("qa", "sql", "routing", "matching", "judge-qa", "judge-b", "judge-c"),
        default="qa",
        help="which suite to regrade",
    )
    regrader.add_argument("--tag", help="which run to regrade")
    suites.add_parser("cost", parents=[common], help="cost of saved runs, no LLM")
    everything = suites.add_parser(
        "all", parents=[common], help="retrieval + qa + sql + routing + matching, corpus with background"
    )
    everything.add_argument("--first", type=int, help="run only the first N cases of each paid suite")
    everything.add_argument("--tag", help="report and dump suffix for the paid suites; retrieval has none")

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

    if args.suite == "cost":
        cost.run()
        return

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
        print(f"\nquestions {len(cases)}, Langfuse tracing: {instrument_langfuse()}\n")
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
        traced = instrument_langfuse()
        print(f"\nSQL questions {len(cases)}, Langfuse tracing: {traced}\n")
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
        traced = instrument_langfuse()
        print(f"\nPairs {len(pairs)} (split {split}), mode {mode}, Langfuse tracing: {traced}\n")
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

        print(f"\nLangfuse tracing: {instrument_langfuse()}")
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
        traced = instrument_langfuse()
        print(f"\nRouting questions {len(cases)}, Langfuse tracing: {traced}\n")
        routing.run(
            cases,
            limit_turns=getattr(args, "max_turns", None) or AGENT_MAX_TURNS,
            budget=getattr(args, "budget", None) or AGENT_MAX_BUDGET_USD,
            verbose=getattr(args, "verbose", False),
            use_mlflow=use_mlflow,
            name=_run_name("routing", getattr(args, "tag", None)),
        )


main()
