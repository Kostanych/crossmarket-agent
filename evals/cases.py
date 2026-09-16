"""Golden-set вопросов: схемы кейсов трёх сюит, чтение файлов, правило разбиения.

`Case` — вопросы к корпусу ВБ, файл один на сюиты retrieval и QA: retrieval берёт
только `expected=found`, QA — все кейсы. `SqlCase` — вопрос и эталонный SQL тула C.
`RoutingCase` — вопрос и лейбл «правильный тул».

Сплит хранится в файле, а не вычисляется при чтении: иначе добавление кейсов
двигало бы границу, и кейс, отработанный в калибровке, мог уехать в тест. То же
правило раздаёт сплит парам ВБ↔Озон (`evals/pairs.py`), поэтому `assign_splits`
работает с протоколом, а не с `Case`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Protocol

GOLDEN_FILE = Path("evals/golden/questions_wb.jsonl")
SQL_GOLDEN_FILE = Path("evals/golden/questions_sql.jsonl")
ROUTING_GOLDEN_FILE = Path("evals/golden/questions_routing.jsonl")

Expected = Literal["found", "absent"]
Split = Literal["calibration", "test"]

CALIBRATION_SHARE = 0.6


class Splittable(Protocol):
    """Что нужно правилу разбиения: идентификатор, страта и место под сплит.

    Сплит раздаётся и вопросам golden-set, и парам ВБ↔Озон
    (`crossmarket.models.Label`).
    """

    id: str
    stratum: str
    split: Split | None


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


@dataclass
class SqlCase:
    """Эталонный кейс тула C: вопрос и SQL, которым считается правильный ответ.

    Эталон хранится запросом, а не строками результата: репозиторий публичный, данных
    в нём нет, а цены и названия — это данные. Эталонный SQL исполняется в момент
    прогона по тому же снапшоту, что видит агент.

    `ordered` — значим ли порядок строк: у топ-N значим, у группировок нет.
    """

    id: str
    question: str
    sql: str
    kind: str = ""
    ordered: bool = False
    split: Split | None = None
    note: str = ""

    @property
    def stratum(self) -> str:
        return self.kind or "—"

    def to_dict(self) -> dict[str, Any]:
        """Значения по умолчанию не пишутся: файл читается глазами при добавлении кейсов."""
        return {key: value for key, value in asdict(self).items() if value not in ("", None, False)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SqlCase:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def load_sql_cases(path: Path = SQL_GOLDEN_FILE) -> list[SqlCase]:
    with path.open(encoding="utf-8") as fh:
        return [SqlCase.from_dict(json.loads(line)) for line in fh if line.strip()]


def save_sql_cases(cases: list[SqlCase], path: Path = SQL_GOLDEN_FILE) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def validate_sql_cases(cases: list[SqlCase]) -> list[str]:
    """Что не так с golden-set тула C. Исполнимость SQL проверяется отдельно, в `check`."""
    problems = []
    seen: set[str] = set()
    for case in cases:
        if not case.id:
            problems.append(f"case without an id: {case.question[:50]}")
        elif case.id in seen:
            problems.append(f"{case.id}: duplicate id")
        seen.add(case.id)

        if not case.sql.strip():
            problems.append(f"{case.id}: no reference SQL")
        if not case.kind:
            problems.append(f"{case.id}: no stratum")
        if case.split not in ("calibration", "test"):
            problems.append(f"{case.id}: split not assigned")
    return problems


ROUTING_TOOLS = ("search_wb", "execute_sql", "match_ozon")
"""Лейблы роутинга — короткие имена тулов. Полное имя из вызова
(`mcp__crossmarket__search_wb`) сравнивается с ними по суффиксу."""

MULTIHOP = "multihop"
"""Страта цепочек между тулами. В accuracy не входит: правильных вызовов там несколько,
а порядок между ними вопросом задан не всегда. Такие кейсы читаются глазами."""


@dataclass
class RoutingCase:
    """Вопрос и тул, который на него отвечает.

    У мультихоповых кейсов лейбла нет: правильных вызовов там два и порядок между ними
    вопросу не задан.
    """

    id: str
    question: str
    tool: str = ""
    kind: str = ""
    split: Split | None = None

    @property
    def stratum(self) -> str:
        return self.kind or "—"

    @property
    def single_hop(self) -> bool:
        """Входит ли кейс в accuracy роутинга."""
        return self.kind != MULTIHOP

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in ("", None)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RoutingCase:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def load_routing_cases(path: Path = ROUTING_GOLDEN_FILE) -> list[RoutingCase]:
    with path.open(encoding="utf-8") as fh:
        return [RoutingCase.from_dict(json.loads(line)) for line in fh if line.strip()]


def save_routing_cases(cases: list[RoutingCase], path: Path = ROUTING_GOLDEN_FILE) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def validate_routing_cases(cases: list[RoutingCase]) -> list[str]:
    """Что не так с golden-set роутинга. Пустой список — файл в порядке."""
    problems = []
    seen: set[str] = set()
    for case in cases:
        if not case.id:
            problems.append(f"case without an id: {case.question[:50]}")
        elif case.id in seen:
            problems.append(f"{case.id}: duplicate id")
        seen.add(case.id)

        if not case.kind:
            problems.append(f"{case.id}: no stratum")
        if case.single_hop and case.tool not in ROUTING_TOOLS:
            problems.append(f"{case.id}: label must be one of {ROUTING_TOOLS}, got {case.tool!r}")
        if not case.single_hop and case.tool:
            problems.append(f"{case.id}: a multihop case must have no label, got {case.tool!r}")
        if case.split not in ("calibration", "test"):
            problems.append(f"{case.id}: split not assigned")
    return problems


def load_cases(path: Path = GOLDEN_FILE, expected: Expected | None = None) -> list[Case]:
    with path.open(encoding="utf-8") as fh:
        cases = [Case.from_dict(json.loads(line)) for line in fh if line.strip()]
    return [case for case in cases if expected is None or case.expected == expected]


def save_cases(cases: list[Case], path: Path = GOLDEN_FILE) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def order_key(case: Splittable) -> str:
    """Хеш идентификатора: раздача не зависит от порядка, в котором дописывали вопросы."""
    return hashlib.blake2s(case.id.encode("utf-8"), digest_size=8).hexdigest()


def assign_splits(cases: list[Splittable], share: float = CALIBRATION_SHARE) -> list[Splittable]:
    """Проставить сплит кейсам, у которых его нет. Возвращает изменённые.

    Уже проставленное не трогается никогда. Новые кейсы внутри страты сортируются
    по хешу и добираются в калибровку до `share` с учётом уже размеченных.
    """
    changed = []
    strata: dict[str, list[Splittable]] = {}
    for case in cases:
        strata.setdefault(case.stratum, []).append(case)

    for members in strata.values():
        target = round(share * len(members))
        calibration = sum(1 for case in members if case.split == "calibration")
        for case in sorted((case for case in members if case.split is None), key=order_key):
            if calibration < target:
                case.split = "calibration"
                calibration += 1
            else:
                case.split = "test"
            changed.append(case)
    return changed


def split_sizes(cases: list[Splittable]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for case in cases:
        sizes[case.split or "—"] = sizes.get(case.split or "—", 0) + 1
    return sizes


def split_summary(cases: list[Splittable]) -> str:
    sizes = split_sizes(cases)
    return f"calibration {sizes.get('calibration', 0)}, test {sizes.get('test', 0)}"


def validate(cases: list[Case]) -> list[str]:
    """Что не так с golden-set. Пустой список — файл в порядке.

    Ловится то, что ломает метрику молча: дубль id перетрёт кейс в отчёте, кейс без
    сплита выпадет из отчётной цифры, absent с `relevant_ids` испортит recall.
    """
    problems = []
    seen: set[str] = set()
    for case in cases:
        if not case.id:
            problems.append(f"case without an id: {case.question[:50]}")
        elif case.id in seen:
            problems.append(f"{case.id}: duplicate id")
        seen.add(case.id)

        if case.expected == "found" and not case.relevant_ids:
            problems.append(f"{case.id}: expected=found without relevant_ids")
        if case.expected == "absent" and case.relevant_ids:
            problems.append(f"{case.id}: expected=absent with relevant_ids")
        if case.expected == "absent" and case.kind not in ("near", "far"):
            problems.append(f"{case.id}: kind must be near or far, got {case.kind!r}")
        if case.expected not in ("found", "absent"):
            problems.append(f"{case.id}: expected={case.expected!r}")
        if case.split not in ("calibration", "test"):
            problems.append(f"{case.id}: split not assigned")
    return problems
