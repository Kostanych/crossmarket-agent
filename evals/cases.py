"""Golden-set вопросов к корпусу ВБ: схема кейса, чтение файла, правило разбиения.

Файл один на обе сюиты: retrieval берёт только `expected=found`, QA — все кейсы.

Сплит хранится в файле, а не вычисляется при чтении: иначе добавление кейсов
двигало бы границу, и кейс, отработанный в калибровке, мог уехать в тест.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

GOLDEN_FILE = Path("evals/golden/questions_wb.jsonl")

Expected = Literal["found", "absent"]
Split = Literal["calibration", "test"]

CALIBRATION_SHARE = 0.6


@dataclass
class Case:
    """Вопрос golden-set вместе с истиной и сплитом.

    `kind` заполняется только у отрицательных кейсов: `near` — товар из категории
    корпуса, `far` — под товар нет и категории.
    """

    id: str
    question: str
    relevant_ids: list[str] = field(default_factory=list)
    expected: Expected = "found"
    kind: str = ""
    split: Split | None = None

    @property
    def stratum(self) -> str:
        """`found`, `absent-near` или `absent-far` — разбиение стратифицируется по ней."""
        return "found" if self.expected == "found" else f"absent-{self.kind or 'far'}"

    def to_dict(self) -> dict[str, Any]:
        """Пустые поля выпадают: `kind` есть только у absent, `split` до раздачи нет."""
        return {key: value for key, value in asdict(self).items() if value not in ("", None)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Case:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def load_cases(path: Path = GOLDEN_FILE, expected: Expected | None = None) -> list[Case]:
    with path.open(encoding="utf-8") as fh:
        cases = [Case.from_dict(json.loads(line)) for line in fh if line.strip()]
    return [case for case in cases if expected is None or case.expected == expected]


def save_cases(cases: list[Case], path: Path = GOLDEN_FILE) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def _order_key(case: Case) -> str:
    """Хеш идентификатора: раздача не зависит от порядка, в котором дописывали вопросы."""
    return hashlib.blake2s(case.id.encode("utf-8"), digest_size=8).hexdigest()


def assign_splits(cases: list[Case], share: float = CALIBRATION_SHARE) -> list[Case]:
    """Проставить сплит кейсам, у которых его нет. Возвращает изменённые.

    Уже проставленное не трогается никогда. Новые кейсы внутри страты сортируются
    по хешу и добираются в калибровку до `share` с учётом уже размеченных.
    """
    changed = []
    strata: dict[str, list[Case]] = {}
    for case in cases:
        strata.setdefault(case.stratum, []).append(case)

    for members in strata.values():
        target = round(share * len(members))
        calibration = sum(1 for case in members if case.split == "calibration")
        for case in sorted((case for case in members if case.split is None), key=_order_key):
            if calibration < target:
                case.split = "calibration"
                calibration += 1
            else:
                case.split = "test"
            changed.append(case)
    return changed


def split_sizes(cases: list[Case]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for case in cases:
        sizes[case.split or "—"] = sizes.get(case.split or "—", 0) + 1
    return sizes


def split_summary(cases: list[Case]) -> str:
    sizes = split_sizes(cases)
    return f"калибровка {sizes.get('calibration', 0)}, тест {sizes.get('test', 0)}"


def validate(cases: list[Case]) -> list[str]:
    """Что не так с golden-set. Пустой список — файл в порядке.

    Ловится то, что ломает метрику молча: дубль id перетрёт кейс в отчёте, кейс без
    сплита выпадет из отчётной цифры, absent с `relevant_ids` испортит recall.
    """
    problems = []
    seen: set[str] = set()
    for case in cases:
        if not case.id:
            problems.append(f"кейс без идентификатора: {case.question[:50]}")
        elif case.id in seen:
            problems.append(f"{case.id}: идентификатор повторяется")
        seen.add(case.id)

        if case.expected == "found" and not case.relevant_ids:
            problems.append(f"{case.id}: expected=found без relevant_ids")
        if case.expected == "absent" and case.relevant_ids:
            problems.append(f"{case.id}: expected=absent с relevant_ids")
        if case.expected == "absent" and case.kind not in ("near", "far"):
            problems.append(f"{case.id}: kind должен быть near или far, а не {case.kind!r}")
        if case.expected not in ("found", "absent"):
            problems.append(f"{case.id}: expected={case.expected!r}")
        if case.split not in ("calibration", "test"):
            problems.append(f"{case.id}: сплит не проставлен")
    return problems
