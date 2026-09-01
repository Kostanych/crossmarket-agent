"""Каркас судей: вердикт, согласие с истиной, таблица расхождений.

Судей здесь нет — они появятся на этапе 7 (QA, B, C). Каркас заведён раньше, потому
что правило разбиения калибровка/тест фиксируется до того, как судья увидит данные;
сплит лежит в `evals/cases.py` и в самом golden-set.

`for_readme` отказывается отдавать согласие, посчитанное не на тест-сплите.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals.report import Table


@dataclass
class Verdict:
    """Решение судьи по кейсу. Без `reason` таблица расхождений вырождается в
    список идентификаторов."""

    case_id: str
    label: str
    reason: str = ""


@dataclass
class Agreement:
    """Согласие судьи с истиной на одном сплите.

    Precision/recall считаются относительно `positive` — метки, которую судья ставит
    положительным исходом. Одна доля согласия на перекошенном наборе выглядит
    прилично и при бесполезном судье.
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
    def reportable(self) -> bool:
        return self.split == "test"

    def for_readme(self) -> float:
        """Отчётная цифра согласия. На калибровке отказывает: там судья доводился,
        и цифра смещена вверх по построению."""
        if not self.reportable:
            raise ValueError(f"согласие посчитано на сплите {self.split!r}, в отчёт идёт только test")
        return self.accuracy

    def disagreements(self) -> Table:
        """На чём судья разошёлся с истиной — тот самый артефакт витрины."""
        rows = [
            [case_id, self.verdicts[case_id].label, truth, self.verdicts[case_id].reason]
            for case_id, truth in self.truth_labels.items()
            if self.verdicts[case_id].label != truth
        ]
        return Table(f"Расхождения судьи с истиной ({self.split})", ["кейс", "судья", "истина", "обоснование"], rows)


def agreement(verdicts: list[Verdict], truth: dict[str, str], positive: str, split: str) -> Agreement:
    """Согласие судьи с истиной. Вердикт без истины и истина без вердикта — ошибка:
    молча выпавший кейс завысил бы согласие."""
    by_case = {verdict.case_id: verdict for verdict in verdicts}
    if by_case.keys() != truth.keys():
        missing = truth.keys() ^ by_case.keys()
        raise ValueError(f"вердикты и истина не совпадают по кейсам: {sorted(missing)}")
    return Agreement(split=split, positive=positive, truth_labels=truth, verdicts=by_case)
