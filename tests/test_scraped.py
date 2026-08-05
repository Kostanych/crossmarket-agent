"""Тесты поиска выгрузки скрапера по идентификатору."""

from pathlib import Path

from crossmarket.extraction.scraped import find_dump, iter_dumps, read_dump

FIXTURES = Path(__file__).parent / "fixtures"


def _folder(tmp_path: Path, marketplace: str, *fixtures: str) -> Path:
    """Разложить фикстуры под именами, какие даёт скрапер: связи с id в них нет."""
    folder = tmp_path / marketplace
    folder.mkdir(parents=True)
    for number, name in enumerate(fixtures):
        target = folder / f"blaze_scrapper ({number}).csv"
        target.write_text((FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_finds_wb_dump_among_others(tmp_path: Path) -> None:
    root = _folder(tmp_path, "wb", "wb_matrasnik.csv", "wb_shkatulka.csv")

    path = find_dump("wb", "681857332", root)

    assert path is not None
    assert read_dump("wb", path).title == "Металлическая шкатулка ящик для денег МВ4"


def test_finds_ozon_dump(tmp_path: Path) -> None:
    root = _folder(tmp_path, "ozon", "ozon_single.csv", "ozon_multirow.csv")

    path = find_dump("ozon", "3437056417", root)

    assert path is not None
    assert read_dump("ozon", path).id == "3437056417"


def test_unknown_id_returns_none(tmp_path: Path) -> None:
    root = _folder(tmp_path, "wb", "wb_matrasnik.csv")

    assert find_dump("wb", "999999999", root) is None


def test_missing_folder_returns_none(tmp_path: Path) -> None:
    assert find_dump("ozon", "3437056417", tmp_path) is None


def test_broken_file_does_not_stop_the_scan(tmp_path: Path) -> None:
    """Одна битая выгрузка из сотен не повод останавливать разметку."""
    root = _folder(tmp_path, "wb", "wb_matrasnik.csv")
    (root / "wb" / "broken.csv").write_text("не csv вовсе", encoding="utf-8")

    assert find_dump("wb", "18326211", root) is not None
    assert len(list(iter_dumps("wb", root))) == 1
