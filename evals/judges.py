"""Три судьи: QA, B и C. Один пайплайн, разные источники истины и промпты.

Истина у каждого судьи своя:

- **QA** — ручная разметка ответов (`evals/answers.py`);
- **B** — совпал ли вердикт модели подтверждения с меткой пары. Судья оценивает
  вердикт вместе с обоснованием, а не решает пару сам;
- **C** — совпал ли результат SQL агента с эталонным запросом (`evals/sql.py`).

Согласие считается по сплитам, рядом печатается `baseline`. `apply_run` — судья-B на
карточках ВБ вне разметки, без истины.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossmarket.config import JUDGE_MODEL
from crossmarket.matching import MATCH_RULES, format_card
from crossmarket.models import Product
from evals.judge import ANSWER_FORMAT, POSITIVE, Agreement, Verdict, agreement, ask_all, judge_options
from evals.report import Table, print_tables, write_report

RUNS_DIR = Path("evals/runs")
SKILL_FILE = Path("skills/clickhouse-sql/SKILL.md")

MATCHING_DUMPS = {"calibration": "matching_b_calibration", "test": "matching_b"}
"""Прогоны тула B, из которых берутся вердикты: калибровочные пары и тестовые."""

SQL_DUMPS = ("sql_c", "sql_c_skill1", "sql_c_48cases_ties", "sql_c_haiku")
"""Прогоны тула C, из которых собирается набор судьи. Записи одних и тех же
кейсов из разных прогонов зависимы."""


QA_PROMPT = f"""Ты проверяешь ответ поискового агента по каталогу товаров маркетплейса.

Тебе дают вопрос пользователя, карточки, которые агент нашёл поиском, и его ответ.
Реши, хороший ответ или плохой.

Ответ хороший, если он делает одно из трёх:
- называет подходящий под вопрос товар из выдачи и не перевирает его свойства и цену;
- посмотрев выдачу, честно говорит, что такого товара в базе нет. Предложить рядом
  лежащее похожее при этом можно и нужно — главное не выдавать его за искомое;
- на вопрос, за которым товара маркетплейса стоять не может — фантастика или
  промышленная техника («на чём полететь в космос», «чем прокопать туннель в горе»), —
  не идёт в поиск, а уточняет, что имелось в виду, или говорит, что такого тут не
  продают. Холостой вызов поиска на таком вопросе — трата времени и денег.

Ответ плохой, если он:
- не пошёл в поиск, хотя за вопросом стоят настоящие товары. Товарным вопрос бывает и
  неочевидно: «чем разнообразить семейную жизнь» — это товары для взрослых, они на
  маркетплейсе есть, и посмотреть надо было;
- пошёл в поиск на заведомо невозможном запросе и предлагает смежные карточки вместо
  прямого ответа, что такого не бывает;
- утверждает, что товара в базе нет, не обратившись к поиску: угадать — не то же
  самое, что проверить, даже если угадано верно;
- выдаёт похожий товар за искомый, хотя искомого в выдаче нет;
- приписывает товару свойства или цену, которых в карточке нет.

Товара вне базы для агента не существует: ходить в интернет он не может, и «нет в
базе» — это нормальный правильный ответ, а не отказ.

В блоке карточек написано, обращался ли агент к поиску вообще: «агент не обращался к
поиску» и «поиск ничего не вернул» — разные вещи, и первое решает два пункта выше.

{ANSWER_FORMAT}
"""

B_PROMPT = f"""Ты проверяешь работу автоматического матчера карточек между маркетплейсами.

{MATCH_RULES}
Тебе дают две карточки и вердикт матчера с его обоснованием. Ты решаешь не про товар, а
про вердикт: верен он или нет.

good — вердикт матчера верный.
bad — вердикт матчера неверный: он назвал матчем разные товары или объявил разными один
и тот же товар.

Порядок работы: сначала реши по самим карточкам, один и тот же это товар, и только
потом посмотри на вердикт матчера. Совпало с твоим решением — good, не совпало — bad.
Вердикт матчера подсказкой не является: он неверен примерно в каждом шестом случае, и
обоснование к неверному вердикту звучит так же уверенно, как к верному.

Оценивается вердикт, а не формулировка. Слабое, неполное или неуклюжее обоснование при
верном вердикте — это good. Убедительное обоснование при неверном вердикте — bad.

Отличием товара НЕ являются:
- незаполненное поле у одной из площадок: отсутствие данных не отличие, а неполнота
  карточки;
- поле, противоречащее названию и описанию, — продавцы заполняют характеристики
  небрежно (бренд «LG» у карточки с моделью Mijia в названии);
- габариты и вес упаковки: площадки меряют и округляют их по-разному;
- разная детализация описания и состава, разные формулировки одного и того же;
- цена.

{ANSWER_FORMAT}
"""


def _sql_prompt() -> str:
    """Промпт судьи-C. Схема и ловушки — текст `SKILL_FILE` целиком: правка скилла
    меняет и этот промпт.
    """
    return f"""Ты проверяешь ответ text2sql-агента по снапшоту товаров в ClickHouse.

Тебе дают вопрос пользователя, SQL-запросы, которые агент выполнил, и его итоговый
ответ. Эталонного запроса у тебя нет — решай по схеме и смыслу вопроса.

good — запрос отвечает ровно на заданный вопрос, а текст ответа согласуется с запросом.
bad — запрос считает не то, что спрошено (не та площадка, не та категория, не тот
агрегат, потерянный фильтр, лишняя или недостающая группировка), либо ответ
противоречит собственному запросу.

Ошибкой НЕ являются:
- лишние колонки в выдаче и лишние пояснения в тексте;
- отсутствие разреза по площадке, если вопрос про площадку не спрашивал;
- расхождение чисел в ответе с числами из описания схемы ниже: примеры там
  иллюстрируют ловушки, эталоном для сверки не служат — данные меняются с каждой
  партией. Считать и пересчитывать сам ты не можешь — данных у тебя нет; суди по
  тому, то ли считает запрос, что спрошено;
- более грубый способ добиться того же результата, если результат тот же.

Ниже — описание схемы и ловушек этих данных, то же самое, что было у агента.

{SKILL_FILE.read_text(encoding="utf-8")}

{ANSWER_FORMAT}
"""


@dataclass
class JudgeCase:
    """Кейс судьи: что он видит, какая по нему истина и в каком он сплите."""

    id: str
    split: str
    stratum: str
    truth: str
    prompt: str
    note: str = ""


@dataclass
class Judge:
    """Судья: имя, промпт, загрузчик кейсов и подпись отчёта."""

    name: str
    title: str
    prompt: str
    load: Callable[[], list[JudgeCase]]
    intro: str
    extra: Callable[[list[JudgeCase], dict[str, Agreement]], list[Table]] | None = None


def _tool_outputs() -> dict[str, list[str]]:
    """Выдача тула целиком по идентификатору ответа (`<кейс>@<модель>`), из дампов QA.

    Компактный `AnswerCase.shown` судье не годится: без характеристик и описания любое
    свойство из карточки выглядит выдумкой.
    """
    from evals.answers import SOURCES

    outputs = {}
    for model, dump in SOURCES:
        for line in (RUNS_DIR / f"{dump}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                outputs[f"{raw['id']}@{model}"] = raw["tool_results"]
    return outputs


def load_qa_cases() -> list[JudgeCase]:
    """Размеченные вручную ответы агента. Неразмеченные пропускаются — истины по ним нет."""
    from evals.answers import VERDICTS, load_answers

    outputs = _tool_outputs()
    cases = []
    for case in load_answers():
        if case.verdict not in VERDICTS:
            continue
        shown = "\n\n".join(outputs.get(case.id, [])) or "— агент не обращался к поиску —"
        cases.append(
            JudgeCase(
                id=case.id,
                split=case.split,
                stratum=case.stratum,
                truth=case.verdict,
                prompt=(
                    f"Вопрос пользователя: {case.question}\n\n"
                    f"Выдача поиска, как её видел агент:\n{shown}\n\n"
                    f"Ответ агента:\n{case.answer}\n\n"
                    "Хороший это ответ или плохой?"
                ),
                note=f"{case.model}, {case.question[:40]}",
            )
        )
    return cases


def _pair_records(dump: str) -> list[dict[str, Any]]:
    """Вердикты вырожденного режима из дампа тула B, только первый прогон."""
    path = RUNS_DIR / f"{dump}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [record for record in records if record.get("kind") == "pair" and record.get("run", 1) == 1]


def load_b_cases(dumps: dict[str, str] = MATCHING_DUMPS) -> list[JudgeCase]:
    """Вердикты модели подтверждения по размеченным парам вместе с карточками.

    Карточки берутся из снапшота: в дампе прогона лежат только идентификаторы.
    Неразобранный вердикт тула пропускается — судить нечего.
    """
    from crossmarket.storage import clickhouse

    records = [record for dump in dumps.values() for record in _pair_records(dump) if record.get("said")]
    client = clickhouse.connect()
    keys = {("wb", record["wb_id"]) for record in records} | {("ozon", record["ozon_id"]) for record in records}
    cards = clickhouse.fetch_products(client, sorted(keys))

    cases = []
    for record in records:
        wb, ozon = cards.get(("wb", record["wb_id"])), cards.get(("ozon", record["ozon_id"]))
        if wb is None or ozon is None:
            continue
        cases.append(
            JudgeCase(
                id=f"{record['wb_id']}:{record['ozon_id']}",
                split=record["split"],
                stratum=record["label"],
                truth="good" if record["said"] == record["label"] else "bad",
                prompt=pair_prompt(wb, ozon, record["said"], record.get("reason", "")),
                note=f"tool: {record['said']}, label: {record['label']}",
            )
        )
    return cases


def pair_prompt(wb: Product, ozon: Product, said: str, reason: str) -> str:
    """Что видит судья-B: обе карточки и вердикт тула с обоснованием."""
    return "\n\n".join(
        [
            format_card(wb, "Wildberries"),
            format_card(ozon, "Ozon"),
            f"Вердикт матчера: {said}\nОбоснование матчера: {reason}",
            "Верен ли вердикт?",
        ]
    )


def load_c_cases(dumps: tuple[str, ...] = SQL_DUMPS) -> list[JudgeCase]:
    """Ответы тула C из сохранённых прогонов, истина — совпадение с эталонным SQL.

    Прогоны переисполняются по базе (`evals.sql.load_run`), LLM не зовётся. Отсутствующий
    дамп пропускается.
    """
    from evals.sql import load_run

    cases = []
    for dump in dumps:
        if not (RUNS_DIR / f"{dump}.jsonl").is_file():
            continue
        for row in load_run(dump):
            case, answer, result = row["case"], row["answer"], row["grade"]
            queries = "\n".join(call["input"]["sql"] for call in answer.tool_calls)
            sqls = queries or "— агент не сделал ни одного запроса —"
            cases.append(
                JudgeCase(
                    id=f"{case.id}@{dump}",
                    split=case.split or "",
                    stratum=case.kind,
                    truth="good" if result.result_match else "bad",
                    prompt=(
                        f"Вопрос пользователя: {case.question}\n\n"
                        f"Запросы агента:\n{sqls}\n\n"
                        f"Ответ агента:\n{answer.text}\n\n"
                        "Верный это ответ или нет?"
                    ),
                    note=f"{dump}, {case.question[:40]}",
                )
            )
    return cases


def qa_checker_table(cases: list[JudgeCase], _: dict[str, Agreement]) -> list[Table]:
    """Детерминированный чекер `evals/grading.py` против той же ручной истины —
    baseline судьи-QA.
    """
    from evals.answers import load_answers

    checker = {case.id: case.checker for case in load_answers()}
    table = Table("grading.py checker against the same manual ground truth", ["split", "cases", "agreement"])
    for split in ("calibration", "test"):
        group = [case for case in cases if case.split == split]
        if not group:
            continue
        agreed = sum(1 for case in group if checker.get(case.id) == case.truth)
        table.rows.append([split, len(group), agreed / len(group)])
    return [table]


JUDGES = {
    "qa": Judge(
        name="qa",
        title="QA judge: agreement with manual answer labels",
        prompt=QA_PROMPT,
        load=load_qa_cases,
        intro="Ground truth is manual labels of agent answers (opus and haiku on the same golden set). "
        "Next to the judge, the deterministic checker `evals/grading.py` is scored on the same ground truth: "
        "a judge that does not beat it is not needed.",
        extra=qa_checker_table,
    ),
    "b": Judge(
        name="b",
        title="B judge: agreement with ground truth on matcher verdicts",
        prompt=B_PROMPT,
        load=load_b_cases,
        intro="The judge grades the confirmation model's verdict together with its reason "
        "instead of deciding the pair itself. Ground truth is whether the tool's verdict matches the labeler's label.",
    ),
    "c": Judge(
        name="c",
        title="C judge: agreement with reference SQL",
        prompt=_sql_prompt(),
        load=load_c_cases,
        intro="Ground truth is `result_match`: whether the agent query result matches the case's reference query. "
        "The set is built from several tool C runs, haiku included: a single opus run has only a handful of misses.",
    ),
}


def split_agreements(cases: list[JudgeCase], verdicts: list[Verdict]) -> dict[str, Agreement]:
    """Согласие по сплитам. Кейс без сплита в цифры не попадает."""
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    result = {}
    for split in ("calibration", "test"):
        group = [case for case in cases if case.split == split]
        if not group:
            continue
        result[split] = agreement(
            [by_case[case.id] for case in group],
            {case.id: case.truth for case in group},
            positive=POSITIVE,
            split=split,
        )
    return result


def summary_table(agreements: dict[str, Agreement], model: str) -> Table:
    table = Table(
        f"Judge agreement with ground truth ({model})",
        ["split", "cases", "agreement", "baseline", "precision", "recall", "F1", "unparsed", "$"],
    )
    for split, result in agreements.items():
        table.rows.append(
            [
                split + (" (reported)" if split == "test" else ""),
                result.total,
                result.accuracy,
                result.baseline,
                result.precision,
                result.recall,
                result.f1,
                result.unparsed,
                round(result.cost_usd, 2),
            ]
        )
    return table


def strata_table(cases: list[JudgeCase], verdicts: list[Verdict]) -> Table:
    """Где судья ошибается — по стратам источника, на всём наборе."""
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    table = Table("By stratum (full set)", ["stratum", "cases", "bad by truth", "agreement"])
    for stratum in sorted({case.stratum for case in cases}):
        group = [case for case in cases if case.stratum == stratum]
        agreed = sum(1 for case in group if by_case[case.id].label == case.truth)
        bad = sum(1 for case in group if case.truth == POSITIVE)
        table.rows.append([stratum, len(group), bad, agreed / len(group)])
    return table


def disagreements_table(cases: list[JudgeCase], verdicts: list[Verdict], with_reason: bool = True) -> Table:
    """Кейсы, где судья разошёлся с истиной, с его обоснованием.

    `with_reason=False` — для markdown-отчёта: обоснования цитируют карточки, а отчёты
    лежат в гите.
    """
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    table = Table(
        "Judge disagreements with ground truth",
        ["case", "split", "judge", "truth"] + (["judge reason"] if with_reason else []),
    )
    for case in cases:
        verdict = by_case[case.id]
        if verdict.label != case.truth:
            table.rows.append(
                [case.note or case.id, case.split, verdict.label or "—", case.truth]
                + ([verdict.reason] if with_reason else [])
            )
    return table


def dump_run(cases: list[JudgeCase], verdicts: list[Verdict], name: str, meta: dict[str, Any]) -> Path:
    """Вердикты судьи вместе с промптами — в `evals/runs/` (вне гита), для перегрейда."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.jsonl"
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            verdict = by_case[case.id]
            fh.write(
                json.dumps(
                    {
                        "id": case.id,
                        "split": case.split,
                        "stratum": case.stratum,
                        "truth": case.truth,
                        "note": case.note,
                        "said": verdict.label,
                        "reason": verdict.reason,
                        "cost_usd": verdict.cost_usd,
                        "prompt": case.prompt,
                        **meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_dump(name: str) -> tuple[list[JudgeCase], list[Verdict], dict[str, Any]]:
    """Восстановить прогон судьи с диска."""
    path = RUNS_DIR / f"{name}.jsonl"
    cases, verdicts, meta = [], [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        meta = {"model": raw.get("model", "—"), "judge": raw.get("judge", "—")}
        cases.append(
            JudgeCase(
                id=raw["id"],
                split=raw["split"],
                stratum=raw["stratum"],
                truth=raw["truth"],
                prompt=raw.get("prompt", ""),
                note=raw.get("note", ""),
            )
        )
        verdicts.append(
            Verdict(case_id=raw["id"], label=raw["said"], reason=raw["reason"], cost_usd=raw.get("cost_usd", 0.0))
        )
    return cases, verdicts, meta


def log_to_mlflow(judge: Judge, agreements: dict[str, Agreement], model: str) -> None:
    import mlflow

    mlflow.set_experiment("judges")
    with mlflow.start_run(run_name=f"{judge.name}_{model}"):
        mlflow.log_params({"judge": judge.name, "model": model, "cases": sum(a.total for a in agreements.values())})
        for split, result in agreements.items():
            suffix = "_test" if split == "test" else "_calibration"
            mlflow.log_metrics(
                {
                    f"accuracy{suffix}": result.accuracy,
                    f"baseline{suffix}": result.baseline,
                    f"precision{suffix}": result.precision,
                    f"recall{suffix}": result.recall,
                    f"f1{suffix}": result.f1,
                    f"unparsed{suffix}": result.unparsed,
                    f"cost_usd{suffix}": result.cost_usd,
                }
            )


def report(
    judge: Judge,
    cases: list[JudgeCase],
    verdicts: list[Verdict],
    model: str,
    use_mlflow: bool,
    dump: Path,
    name: str,
) -> None:
    """Таблицы в stdout, markdown-отчёт, метрики в MLflow. Общее для прогона и перегрейда.

    В markdown расхождения идут без обоснований (`disagreements_table`), полные — в stdout
    и в дампе.
    """
    agreements = split_agreements(cases, verdicts)
    aggregates = [summary_table(agreements, model), strata_table(cases, verdicts)]
    if judge.extra:
        aggregates += judge.extra(cases, agreements)
    disagreements = disagreements_table(cases, verdicts)
    print_tables([*aggregates, disagreements])

    unparsed = sum(1 for verdict in verdicts if not verdict.label)
    if unparsed > len(verdicts) / 10:
        print(f"\nWARNING: {unparsed} of {len(verdicts)} judge answers unparsed — the numbers cannot be trusted.")
    total = sum(verdict.cost_usd for verdict in verdicts)
    print(f"\ncases {len(cases)}, judge {model}, run ${total:.2f}")

    path = write_report(
        name,
        judge.title,
        judge.intro,
        {
            "judge": model,
            "cases": len(cases),
            "splits": ", ".join(f"{split} {result.total}" for split, result in agreements.items()),
            "positive class": POSITIVE,
            "run cost": f"${total:.2f}",
        },
        [*aggregates, disagreements_table(cases, verdicts, with_reason=False)],
    )
    print(f"Report: {path}, raw run: {dump}")
    if use_mlflow:
        log_to_mlflow(judge, agreements, model)
        print("Metrics logged to MLflow: experiment judges")


def run(
    which: str,
    split: str = "test",
    model: str = JUDGE_MODEL,
    first: int | None = None,
    use_mlflow: bool = True,
    name: str | None = None,
) -> None:
    """Прогнать судью и отчитаться. Платный ход."""
    import asyncio

    judge = JUDGES[which]
    cases = [case for case in judge.load() if split == "all" or case.split == split]
    cases = cases[:first] if first else cases
    if not cases:
        raise SystemExit(f"no {which} judge cases on split {split}")

    print(f"\nJudge {which}: cases {len(cases)} (split {split}), model {model}\n")
    verdicts = asyncio.run(ask_all({case.id: case.prompt for case in cases}, judge_options(judge.prompt, model)))
    for case, verdict in zip(cases, verdicts, strict=True):
        mark = "+" if verdict.label == case.truth else "—"
        print(f"{mark} {case.id[:28]:<28} judge {verdict.label or '?':<4} truth {case.truth:<4} {case.note[:40]}")

    from crossmarket.agent import flush_langfuse

    flush_langfuse()
    name = name or f"judge_{which}"
    meta = {"model": model, "judge": which}
    report(judge, cases, verdicts, model, use_mlflow, dump_run(cases, verdicts, name, meta), name)


def regrade(use_mlflow: bool = False, name: str = "judge_b") -> None:
    """Пересчитать метрики сохранённого прогона судьи. LLM не зовётся."""
    cases, verdicts, meta = load_dump(name)
    judge = JUDGES[meta["judge"]] if meta["judge"] in JUDGES else JUDGES["b"]
    print(f"regrading {len(cases)} verdicts from {RUNS_DIR / f'{name}.jsonl'}\n")
    report(judge, cases, verdicts, meta["model"], use_mlflow, RUNS_DIR / f"{name}.jsonl", name)


def unlabelled_wb_ids() -> list[str]:
    """Карточки ВБ, которых нет ни в одной размеченной паре."""
    from crossmarket.storage.jsonl import load_labels, load_products

    labelled = {label.wb_id for label in load_labels().values()}
    products = load_products()
    return sorted(key[1] for key in products if key[0] == "wb" and key[1] not in labelled)


async def run_apply(wb_ids: list[str], model: str) -> list[dict[str, Any]]:
    """Полный режим B по неразмеченным карточкам плюс судья по каждому вердикту."""
    import asyncio

    from crossmarket.matching import match
    from crossmarket.storage import clickhouse, qdrant

    qdrant_client, clickhouse_client = qdrant.connect(), clickhouse.connect()
    options = judge_options(JUDGES["b"].prompt, model)

    rows = []
    for number, wb_id in enumerate(wb_ids, start=1):
        result = await match(wb_id, qdrant_client, clickhouse_client)
        if result.wb is None:
            continue
        prompts = {
            f"{wb_id}:{candidate.product.id}": pair_prompt(
                result.wb,
                candidate.product,
                candidate.verdict.said or "не разобран",
                candidate.verdict.reason if candidate.verdict else "",
            )
            for candidate in result.candidates
            if candidate.verdict is not None
        }
        verdicts = await ask_all(prompts, options)
        for candidate, verdict in zip(result.candidates, verdicts, strict=True):
            rows.append(
                {
                    "wb_id": wb_id,
                    "wb_title": result.wb.title,
                    "ozon_id": candidate.product.id,
                    "ozon_title": candidate.product.title,
                    "score": candidate.score,
                    "said": candidate.verdict.said if candidate.verdict else None,
                    "tool_reason": candidate.verdict.reason if candidate.verdict else "",
                    "judge": verdict.label,
                    "judge_reason": verdict.reason,
                    "cost_usd": (candidate.verdict.cost_usd if candidate.verdict else 0.0) + verdict.cost_usd,
                }
            )
        confirmed = sum(1 for candidate in result.candidates if candidate.matched)
        flagged = sum(1 for verdict in verdicts if verdict.label == POSITIVE)
        print(
            f"{number:>3}/{len(wb_ids)} {wb_id} confirmed {confirmed}/{len(result.candidates)}, "
            f"judge disputes {flagged} — {result.wb.title[:44]}"
        )
        await asyncio.sleep(0)
    return rows


def apply_tables(rows: list[dict[str, Any]]) -> list[Table]:
    """Итог применения: счёт вердиктов и несогласия судьи с тулом."""
    items = len({row["wb_id"] for row in rows})
    confirmed = [row for row in rows if row["said"] == "match"]
    disputed = [row for row in rows if row["judge"] == POSITIVE]
    summary = Table(
        "B judge applied to unlabeled listings",
        ["WB listings", "candidates", "confirmed by tool", "judge disputes", "disputes confirmations", "$"],
        [
            [
                items,
                len(rows),
                len(confirmed),
                len(disputed),
                sum(1 for row in disputed if row["said"] == "match"),
                round(sum(row["cost_usd"] for row in rows), 2),
            ]
        ],
    )
    table = Table(
        "Judge disagreements with the tool (no ground truth, read by eye)",
        ["WB", "Ozon", "tool", "judge reason"],
    )
    for row in disputed:
        table.rows.append([row["wb_title"][:44], row["ozon_title"][:44], row["said"], row["judge_reason"]])
    return [summary, table]


def apply_run(model: str = JUDGE_MODEL, first: int | None = None, name: str = "judge_b_apply") -> None:
    """Прогнать полный режим B по неразмеченным карточкам и судью по его вердиктам."""
    import asyncio

    from crossmarket.agent import flush_langfuse

    wb_ids = unlabelled_wb_ids()
    wb_ids = wb_ids[:first] if first else wb_ids
    print(f"\nUnlabeled WB listings: {len(wb_ids)}, judge {model}\n")
    rows = asyncio.run(run_apply(wb_ids, model))
    flush_langfuse()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tables = apply_tables(rows)
    print_tables(tables)
    report_path = write_report(
        name,
        "B judge on unlabeled listings",
        "The only place where the B judge does work instead of being checked: there is no ground truth here. "
        "No agreement numbers — only verdict counts and judge disagreements with the tool, read by eye.",
        {"judge": model, "WB listings": len({row["wb_id"] for row in rows}), "candidates": len(rows)},
        [tables[0]],
    )
    print(f"\nReport: {report_path}, raw run: {path}")
