"""Таблицы харнесса: одна модель данных на stdout и на markdown-отчёт.

Нужны оба: из stdout снимается gif прогона, файл показывает цифру без запуска.
Общий рендер — чтобы отчёт не разъезжался с экраном.

Отчёт перезаписывается каждым прогоном, история метрик живёт в git. Данных в нём
нет — только цифры и вопросы golden-set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("evals/reports")


def cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "—"
    return str(value)


@dataclass
class Table:
    """Заголовки и строки. Ширины считаются по содержимому, а не задаются руками."""

    caption: str
    headers: list[str]
    rows: list[list[Any]] = field(default_factory=list)

    def _cells(self) -> list[list[str]]:
        return [[cell(value) for value in row] for row in self.rows]

    def to_text(self, width_limit: int = 52) -> str:
        """Для stdout: первая колонка левым краем, остальные правым."""
        rows = [[value[:width_limit] for value in row] for row in self._cells()]
        widths = [
            max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
            for index, header in enumerate(self.headers)
        ]

        def line(values: list[str]) -> str:
            head = f"{values[0]:<{widths[0]}}"
            return " ".join([head] + [f"{value:>{widths[index]}}" for index, value in enumerate(values[1:], start=1)])

        body = [line(self.headers), "-" * (sum(widths) + len(widths) - 1)]
        body += [line(row) for row in rows]
        return f"\n{self.caption}\n" + "\n".join(body) if self.caption else "\n".join(body)

    def to_markdown(self) -> str:
        head = "| " + " | ".join(self.headers) + " |"
        rule = "|" + "|".join("---" for _ in self.headers) + "|"
        body = ["| " + " | ".join(row) + " |" for row in self._cells()]
        return (
            "\n".join([f"### {self.caption}", "", head, rule, *body])
            if self.caption
            else "\n".join([head, rule, *body])
        )


def print_tables(tables: list[Table]) -> None:
    for table in tables:
        print(table.to_text())


def write_report(name: str, title: str, intro: str, config: dict[str, Any], tables: list[Table]) -> Path:
    """Собрать markdown-отчёт прогона и положить в `evals/reports/<name>.md`."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.md"

    settings = Table("", ["параметр", "значение"], [[key, value] for key, value in config.items()])
    parts = [
        f"# {title}",
        "",
        f"Сгенерировано `python -m evals {name.split('_')[0]}` {date.today().isoformat()}. "
        "Файл перезаписывается каждым прогоном — история в git.",
        "",
        intro.strip(),
        "",
        "## Конфигурация",
        "",
        settings.to_markdown(),
        "",
    ]
    for table in tables:
        parts += [table.to_markdown(), ""]

    path.write_text("\n".join(parts), encoding="utf-8")
    return path
