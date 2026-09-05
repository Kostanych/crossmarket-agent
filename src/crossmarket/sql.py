"""Исполнение SQL по снапшоту ClickHouse — начинка тула C.

Зовут модуль двое: обёртка тула в агенте и eval-харнесс.

Запрос модель пишет сама и целиком; шаблонов с подстановкой параметров здесь нет.

**Ошибка возвращается текстом, а не исключением.** На ней держится починка запроса
агентом: он видит сообщение ClickHouse, правит SQL и зовёт тул снова.

Ограничения двойные — клиентская проверка формы запроса и серверный `readonly`:
клиентская смотрит только на начало строки.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clickhouse_connect.driver.client import Client

MAX_RESULT_ROWS = 500
"""Серверный предохранитель от раздувшегося джойна: в таблице 394 строки, столько
не вернёт ни один осмысленный запрос."""

SHOWN_ROWS = 60
"""Сколько строк уходит в контекст модели. Остальное обрезается с пометкой."""

TIMEOUT_S = 10

SETTINGS: dict[str, Any] = {
    "readonly": 1,
    "max_execution_time": TIMEOUT_S,
    "max_result_rows": MAX_RESULT_ROWS,
    "result_overflow_mode": "throw",
}
"""`readonly=1` проверен прогоном: INSERT, DROP и TRUNCATE отбиваются самим сервером
с кодом 164. Мульти-запрос режет HTTP-интерфейс ClickHouse, до нас он не доходит."""

ALLOWED_START = ("select", "with", "describe", "desc", "show", "explain")

_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def check_sql(sql: str) -> str:
    """Что не так с запросом. Пустая строка — можно исполнять.

    Проверяется форма, а не смысл: одно выражение и чтение, а не запись. Точка с
    запятой в конце снимается, внутри — отклоняется: `readonly` защитил бы от записи,
    но не от второго запроса в той же строке.
    """
    body = _COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not body:
        return "Пустой запрос."
    if ";" in body:
        return "Только одно выражение за раз: точка с запятой внутри запроса не допускается."
    first = body.split(None, 1)[0].lower().lstrip("(")
    if first not in ALLOWED_START:
        allowed = ", ".join(ALLOWED_START).upper()
        return f"Разрешено только чтение: запрос должен начинаться с {allowed}, а не с {first.upper()}."
    return ""


@dataclass
class SqlResult:
    """Что вернул запрос — или почему не вернул."""

    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str = ""
    elapsed_s: float = 0.0
    total_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def truncated(self) -> bool:
        return self.total_rows > len(self.rows)


def run_sql(client: Client, sql: str, shown_rows: int = SHOWN_ROWS) -> SqlResult:
    """Исполнить запрос модели. Ошибка ClickHouse ложится в `error`, а не летит выше."""
    if problem := check_sql(sql):
        return SqlResult(sql=sql, error=problem)

    started = time.perf_counter()
    try:
        answer = client.query(sql.strip().rstrip(";"), settings=SETTINGS)
    except Exception as exc:  # noqa: BLE001 — сообщение сервера нужно модели целиком
        return SqlResult(sql=sql, error=str(exc), elapsed_s=time.perf_counter() - started)

    rows = list(answer.result_rows)
    return SqlResult(
        sql=sql,
        columns=list(answer.column_names),
        rows=rows[:shown_rows],
        elapsed_s=time.perf_counter() - started,
        total_rows=len(rows),
    )


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_result(result: SqlResult) -> str:
    """Результат текстом для модели.

    Колонки идут заголовком, строки — через `|`. Пустая выдача называется словами:
    «строк нет» — законный ответ, и модель должна отличать его от ошибки.
    """
    if not result.ok:
        return f"Ошибка запроса: {result.error}"
    if not result.rows:
        return "Запрос выполнен, строк нет."

    lines = [" | ".join(result.columns)]
    lines += [" | ".join(_cell(value) for value in row) for row in result.rows]
    header = f"Строк: {result.total_rows}"
    if result.truncated:
        header += f", показаны первые {len(result.rows)}"
    return f"{header}\n" + "\n".join(lines)
