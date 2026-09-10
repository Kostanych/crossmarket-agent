"""Набор ответов агента под ручную разметку — истина для судьи-QA.

Набор собирается из сохранённых прогонов сюиты QA: ответы opus и haiku на один и тот же
golden-set. Файлы лежат в `data/` (вне гита): в ответах настоящие названия и цены.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ANSWERS_FILE = Path("data/qa_answers.jsonl")
READABLE_FILE = Path("data/qa_answers.md")
"""Тот же набор в markdown — для чтения при разметке. Перезаписывается
пересборкой; метки ставятся в `ANSWERS_FILE`."""

SOURCES = (("opus", "qa_wb"), ("haiku", "qa_wb_haiku"))
"""Прогоны, из которых собирается набор: имя модели и имя дампа в `evals/runs/`."""

SAMPLE_SIZE = 40
VERDICTS = ("good", "bad")


@dataclass
class AnswerCase:
    """Один ответ агента вместе с ручной меткой.

    `shown` — компактная выдача тула: id, название, цена. `verdict` пуст до разметки.
    `checker` — вердикт детерминированного чекера (`evals.grading`): в истину не идёт, в
    отчёте судьи-QA служит baseline.

    `searched` — звал ли агент тул. Пустой `shown` его не заменяет: тул мог ответить
    «Ничего не найдено» — карточек нет, а поиск был.
    """

    id: str
    case_id: str
    model: str
    stratum: str
    split: str
    verdict: str = ""
    comment: str = ""
    question: str = ""
    answer: str = ""
    checker: str = ""
    searched: bool = True
    shown: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AnswerCase:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def load_answers(path: Path = ANSWERS_FILE) -> list[AnswerCase]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [AnswerCase.from_dict(json.loads(line)) for line in fh if line.strip()]


def save_answers(answers: list[AnswerCase], path: Path = ANSWERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for answer in answers:
            fh.write(json.dumps(answer.to_dict(), ensure_ascii=False) + "\n")


def _empty_output(case: AnswerCase) -> str:
    return "— поиск вызывался, карточек не вернул —" if case.searched else "— агент не обращался к поиску —"


def write_readable(answers: list[AnswerCase], path: Path = READABLE_FILE) -> Path:
    """Записать набор в markdown для чтения."""
    parts = [
        "# Ответы агента под ручную разметку",
        "",
        f"Читать здесь, метки ставить в `{ANSWERS_FILE}` (поля `verdict`: good/bad и `comment`).",
        "Файл перезаписывается пересборкой заготовки.",
        "",
    ]
    for case in answers:
        parts += [
            f"## {case.id} — {case.stratum}, {case.split}",
            "",
            f"**Вопрос:** {case.question}",
            "",
            "**Выдача тула:**",
            *[f"- {line}" for line in case.shown or [_empty_output(case)]],
            "",
            "**Ответ:**",
            "",
            case.answer.strip() or "— пусто —",
            "",
            f"*Чекер: {case.checker}*",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _order_key(case_id: str) -> str:
    """Хеш идентификатора, как в `evals.cases.order_key`: выборка не зависит от
    порядка записей в дампах.
    """
    return hashlib.blake2s(case_id.encode("utf-8"), digest_size=8).hexdigest()


def _collect(sources: tuple[tuple[str, str], ...] = SOURCES) -> list[AnswerCase]:
    """Все ответы обоих прогонов, оценённые чекером. LLM не зовётся."""
    from evals.grading import parse_tool_output
    from evals.qa import load_run

    pool = []
    for model, dump in sources:
        for row in load_run(dump):
            case, answer, checker = row["case"], row["answer"], row["grade"]
            shown = parse_tool_output(answer.tool_results)
            pool.append(
                AnswerCase(
                    id=f"{case.id}@{model}",
                    case_id=case.id,
                    model=model,
                    stratum=case.stratum,
                    split=case.split or "",
                    question=case.question,
                    answer=answer.text,
                    checker="good" if checker.outcome_correct else "bad",
                    searched=answer.searched,
                    shown=[f"{card.id} — {card.title} — {card.price_rub} ₽" for card in shown.values()],
                )
            )
    return pool


def sample(size: int = SAMPLE_SIZE, sources: tuple[tuple[str, str], ...] = SOURCES) -> list[AnswerCase]:
    """Выборка под разметку: все промахи чекера плюс добор до `size` по хешу
    идентификатора с сохранением долей страт пула.
    """
    pool = _collect(sources)
    chosen = {case.id: case for case in pool if case.checker == "bad"}
    rest = [case for case in pool if case.id not in chosen]
    remaining = max(size - len(chosen), 0)

    strata = sorted({case.stratum for case in pool})
    quotas, allocated = {}, 0
    for stratum in strata[:-1]:
        share = sum(1 for case in pool if case.stratum == stratum) / len(pool)
        quotas[stratum] = round(share * remaining)
        allocated += quotas[stratum]
    # Последней страте достаётся остаток: округление долей иначе теряет кейс.
    quotas[strata[-1]] = max(remaining - allocated, 0)

    for stratum, quota in quotas.items():
        ordered = sorted((case for case in rest if case.stratum == stratum), key=lambda case: _order_key(case.id))
        for case in ordered[:quota]:
            chosen[case.id] = case

    return sorted(chosen.values(), key=lambda case: (case.case_id, case.model))


def merge(fresh: list[AnswerCase], existing: list[AnswerCase]) -> list[AnswerCase]:
    """Обновить набор, не потеряв проставленные руками вердикты."""
    labelled = {case.id: case for case in existing}
    for case in fresh:
        if previous := labelled.get(case.id):
            case.verdict, case.comment = previous.verdict, previous.comment
    return fresh


def validate(answers: list[AnswerCase]) -> list[str]:
    """Что мешает считать согласие. Пустой список — набор готов."""
    problems = []
    seen: set[str] = set()
    for case in answers:
        if case.id in seen:
            problems.append(f"{case.id}: идентификатор повторяется")
        seen.add(case.id)
        if case.verdict not in VERDICTS:
            problems.append(f"{case.id}: вердикт {case.verdict!r}, а нужен один из {VERDICTS}")
        if case.split not in ("calibration", "test"):
            problems.append(f"{case.id}: сплит не проставлен")
    return problems


def summary(answers: list[AnswerCase]) -> str:
    sizes: dict[str, int] = {}
    for case in answers:
        sizes[case.split] = sizes.get(case.split, 0) + 1
    labelled = sum(1 for case in answers if case.verdict in VERDICTS)
    bad = sum(1 for case in answers if case.verdict == "bad")
    return (
        f"ответов {len(answers)}: калибровка {sizes.get('calibration', 0)}, тест {sizes.get('test', 0)}; "
        f"размечено {labelled}, из них плохих {bad}"
    )
