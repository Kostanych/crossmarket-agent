"""Каркас судей: вызов судьи, согласие с истиной, таблица расхождений.

Сами судьи (QA, B, C) — в `evals/judges.py`. Сплит калибровка/тест берётся из данных:
golden-set (`evals/cases.py`), пары ВБ↔Озон (`evals/pairs.py`), набор ответов
(`evals/answers.py`). `for_readme` отказывается отдавать согласие не с тест-сплита.

Транспорт — Claude Agent SDK без тулов (`crossmarket.agent.ask`), как у модели
подтверждения. Положительный класс у всех судей — `bad`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions

from crossmarket.config import JUDGE_MODEL
from evals.report import Table

LABELS = ("good", "bad")
POSITIVE = "bad"
"""Положительный класс для precision и recall."""

ANSWER_FORMAT = """Ответ ровно в два поля, без преамбул и без разметки:
первая строка — одна фраза, чем обосновано
вторая строка — good или bad, одним словом и больше ничего"""
"""Общий хвост промптов судей. Порядок полей менять нельзя: с вердиктом в
первой строке модель ставит его до рассуждения (`docs/failures.md`, случай 4)."""

VERDICT = re.compile(rf"^\W*({'|'.join(LABELS)})\b", re.IGNORECASE)
"""Вердикт ищется только с начала строки: `good` внутри обоснования не в счёт."""


def judge_options(prompt: str, model: str = JUDGE_MODEL) -> ClaudeAgentOptions:
    """Опции судьи: тулов нет, настройки проекта не читаются.

    `max_turns=2`, а не 1: ход тратится и на сам ответ, при 1 прогон рвётся по лимиту.
    """
    return ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        tools=[],
        mcp_servers={},
        allowed_tools=[],
        permission_mode="dontAsk",
        setting_sources=[],
        max_turns=2,
        max_budget_usd=0.10,
    )


def parse_label(text: str) -> str:
    """Вердикт из последней непустой строки. Не разобралось — пустая строка: в
    согласие не засчитывается и попадает в `Agreement.unparsed`.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    found = VERDICT.match(lines[-1])
    return found.group(1).lower() if found else ""


async def ask_judge(case_id: str, prompt: str, options: ClaudeAgentOptions) -> Verdict:
    """Один вызов судьи. Обоснование — первая строка ответа, вердикт — последняя."""
    from crossmarket.agent import ask

    answer = await ask(prompt, options)
    lines = [line.strip() for line in answer.text.splitlines() if line.strip()]
    return Verdict(
        case_id=case_id,
        label=parse_label(answer.text),
        reason=lines[0] if len(lines) > 1 else "",
        cost_usd=answer.cost_usd,
    )


async def ask_all(prompts: dict[str, str], options: ClaudeAgentOptions, concurrency: int = 4) -> list[Verdict]:
    """Судить пачку кейсов параллельно. Порядок выдачи — порядок `prompts`."""
    gate = asyncio.Semaphore(concurrency)

    async def one(case_id: str, prompt: str) -> Verdict:
        async with gate:
            return await ask_judge(case_id, prompt, options)

    return list(await asyncio.gather(*(one(case_id, prompt) for case_id, prompt in prompts.items())))


@dataclass
class Verdict:
    """Решение судьи по кейсу: метка, обоснование, стоимость вызова."""

    case_id: str
    label: str
    reason: str = ""
    cost_usd: float = 0.0


@dataclass
class Agreement:
    """Согласие судьи с истиной на одном сплите. Precision и recall считаются
    относительно `positive`.
    """

    split: str
    positive: str
    truth_labels: dict[str, str] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.truth_labels)

    @property
    def agreed(self) -> int:
        return sum(1 for case_id, label in self.truth_labels.items() if self.verdicts[case_id].label == label)

    @property
    def accuracy(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    def _counts(self) -> tuple[int, int, int]:
        true_positive = false_positive = false_negative = 0
        for case_id, truth in self.truth_labels.items():
            said = self.verdicts[case_id].label
            true_positive += said == self.positive and truth == self.positive
            false_positive += said == self.positive and truth != self.positive
            false_negative += said != self.positive and truth == self.positive
        return true_positive, false_positive, false_negative

    @property
    def precision(self) -> float:
        true_positive, false_positive, _ = self._counts()
        return true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0

    @property
    def recall(self) -> float:
        true_positive, _, false_negative = self._counts()
        return true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def unparsed(self) -> int:
        """Сколько ответов судьи не разобрались. Считаются несогласием."""
        return sum(1 for verdict in self.verdicts.values() if not verdict.label)

    @property
    def baseline(self) -> float:
        """Доля мажоритарного класса в истине — согласие судьи, отвечающего
        одинаково на всё.
        """
        if not self.truth_labels:
            return 0.0
        counts: dict[str, int] = {}
        for label in self.truth_labels.values():
            counts[label] = counts.get(label, 0) + 1
        return max(counts.values()) / self.total

    @property
    def cost_usd(self) -> float:
        return sum(verdict.cost_usd for verdict in self.verdicts.values())

    @property
    def reportable(self) -> bool:
        return self.split == "test"

    def for_readme(self) -> float:
        """Отчётная цифра согласия. Не с тест-сплита — `ValueError`."""
        if not self.reportable:
            raise ValueError(f"agreement computed on split {self.split!r}, only test goes into the report")
        return self.accuracy

    def disagreements(self) -> Table:
        """Кейсы, где судья разошёлся с истиной."""
        rows = [
            [case_id, self.verdicts[case_id].label, truth, self.verdicts[case_id].reason]
            for case_id, truth in self.truth_labels.items()
            if self.verdicts[case_id].label != truth
        ]
        headers = ["case", "judge", "truth", "reason"]
        return Table(f"Judge disagreements with ground truth ({self.split})", headers, rows)


def agreement(verdicts: list[Verdict], truth: dict[str, str], positive: str, split: str) -> Agreement:
    """Согласие судьи с истиной. Вердикт без истины или истина без вердикта — `ValueError`."""
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    if by_case.keys() != truth.keys():
        missing = truth.keys() ^ by_case.keys()
        raise ValueError(f"verdicts and ground truth cover different cases: {sorted(missing)}")
    return Agreement(split=split, positive=positive, truth_labels=truth, verdicts=by_case)
