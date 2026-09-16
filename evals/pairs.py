"""Размеченные пары ВБ↔Озон: чтение, сплит калибровка/тест, сводка.

Сплит парам раздаётся тем же правилом, что и вопросам golden-set
(`evals.cases.assign_splits`): детерминированно по хешу идентификатора,
стратифицированно, уже проставленное не трогается.

Хранилище append-only, поэтому проставленный сплит дописывается новой строкой:
при чтении побеждает последняя запись по ключу.
"""

from __future__ import annotations

from pathlib import Path

from crossmarket.models import Label
from crossmarket.storage.jsonl import DATA_DIR, append_label, load_labels, load_products
from evals.cases import assign_splits, split_sizes


def load_pairs(data_dir: Path = DATA_DIR) -> list[Label]:
    """Пары, у которых обе карточки лежат в снапшоте, в стабильном порядке.

    Неполные пары отбрасываются: карточку без цены выбраковывает сбор датасета, и
    подтверждать по половине пары нечего.
    """
    products = load_products(data_dir)
    pairs = [
        label
        for label in load_labels(data_dir).values()
        if ("wb", label.wb_id) in products and ("ozon", label.ozon_id) in products
    ]
    return sorted(pairs, key=lambda label: label.id)


def assign(pairs: list[Label], data_dir: Path = DATA_DIR) -> list[Label]:
    """Проставить сплит парам, у которых его нет, и дописать изменённые в хранилище."""
    changed = assign_splits(pairs)
    for label in changed:
        append_label(label, data_dir)
    return changed


def strata(pairs: list[Label]) -> dict[str, dict[str, int]]:
    """Сколько пар в каждой страте по сплитам — таблица, по которой видно перекос."""
    counts: dict[str, dict[str, int]] = {}
    for label in pairs:
        counts.setdefault(label.stratum, {})
        key = label.split or "—"
        counts[label.stratum][key] = counts[label.stratum].get(key, 0) + 1
    return dict(sorted(counts.items()))


def summary(pairs: list[Label]) -> str:
    sizes = split_sizes(pairs)
    return f"pairs {len(pairs)}: calibration {sizes.get('calibration', 0)}, test {sizes.get('test', 0)}"


def validate(pairs: list[Label]) -> list[str]:
    """Что не так с набором пар. Пустой список — всё в порядке."""
    problems = []
    for label in pairs:
        if label.split not in ("calibration", "test"):
            problems.append(f"{label.id}: split not assigned")
        if label.label == "no_match" and label.negative_kind is None:
            problems.append(f"{label.id}: no_match without negative_kind, stratum unknown")
    return problems
