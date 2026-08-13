"""Тесты сборки датасета из сид-разметки.

`tools/dataset.py` — скрипт, а не модуль пакета, поэтому грузится по пути.
Проверяется то, что определяет ground truth: как строка сида превращается
в метку и какие пары считаются полными.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "dataset.py"
FIXTURES = Path(__file__).parent / "fixtures"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dataset_tool", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _seed_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "разметка.csv"
    path.write_text("вб;озон;result;comment\n" + body, encoding="utf-8-sig")
    return path


def test_seed_match_row(tool: ModuleType, tmp_path: Path) -> None:
    (row,) = tool.load_seed(_seed_file(tmp_path, "1000000001;4000000001;1;\n"))
    label = tool.seed_to_label(row)

    assert (label.wb_id, label.ozon_id) == ("1000000001", "4000000001")
    assert label.label == "match"
    assert label.negative_kind is None
    assert label.source == "seed"


def test_seed_zero_is_a_hard_negative(tool: ModuleType, tmp_path: Path) -> None:
    """Сид собран из похожих пар-кандидатов, случайных негативов в нём нет."""
    (row,) = tool.load_seed(_seed_file(tmp_path, "1000000002;4000000002;0;\n"))
    label = tool.seed_to_label(row)

    assert label.label == "no_match"
    assert label.negative_kind == "hard"


def test_comment_survives(tool: ModuleType, tmp_path: Path) -> None:
    """Оговорка из сида — материал для калибровки судьи, терять её нельзя."""
    (row,) = tool.load_seed(_seed_file(tmp_path, "1000000003;4000000003;1;оговорка\n"))

    assert tool.seed_to_label(row).comment == "оговорка"


def test_split_by_coverage(tool: ModuleType, tmp_path: Path) -> None:
    """Пара полна только тогда, когда на диске обе карточки."""
    for marketplace, fixture in (("wb", "wb_matrasnik.csv"), ("ozon", "ozon_single.csv")):
        folder = tmp_path / marketplace
        folder.mkdir()
        (folder / "dump.csv").write_text((FIXTURES / fixture).read_text(encoding="utf-8"), encoding="utf-8")

    dumps = {mp: tool.Dumps(mp, tmp_path) for mp in ("wb", "ozon")}
    rows = tool.load_seed(
        _seed_file(
            tmp_path,
            "18326211;3437056417;1;\n"  # обе на диске
            "18326211;9999999999;1;\n"  # нет озона
            "9999999999;3437056417;0;\n"  # нет вб
            "9999999999;8888888888;1;\n",  # нет ни одной
        )
    )
    buckets = tool.split_by_coverage([tool.seed_to_label(row) for row in rows], dumps)

    assert [len(buckets[name]) for name in ("both", "wb_only", "ozon_only", "neither")] == [1, 1, 1, 1]
