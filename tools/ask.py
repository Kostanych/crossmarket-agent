"""Вопрос агенту через запущенный `/chat` — без ручного JSON.

Запуск:
    make serve                                        # в отдельном терминале
    poetry run python tools/ask.py "вопрос"           # один вопрос
    poetry run python tools/ask.py                    # вопросы подряд, выход — пустая строка или Ctrl+D

Ходит в HTTP-сервис, а не в агента напрямую: запрос попадает в `/metrics` и на дашборд Grafana.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

TIMEOUT_S = 300


@contextmanager
def waiting() -> Iterator[None]:
    """Анимированные точки и секунды ожидания на время запроса. Вне терминала молчит: в файл или
    конвейер `\\r` попал бы мусором.
    """
    if not sys.stdout.isatty():
        yield
        return
    done = threading.Event()

    def spin() -> None:
        start = time.monotonic()
        dots = 0
        while not done.wait(0.4):
            dots = dots % 3 + 1
            print(f"\rthinking{'.' * dots:<3} {time.monotonic() - start:.0f}s", end="", flush=True)
        print("\r" + " " * 20 + "\r", end="", flush=True)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        yield
    finally:
        done.set()
        spinner.join()


def ask(url: str, question: str) -> None:
    request = urllib.request.Request(
        f"{url}/chat",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with waiting(), urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        body = json.load(exc)
    except urllib.error.URLError as exc:
        print(f"{url} is not reachable ({exc.reason}) — is `make serve` running?")
        return

    print(f"\n{body['answer']}\n")
    tools = " → ".join(body["tools"]) or "none"
    print(f"tools: {tools} | turns: {body['num_turns']} | ${body['cost_usd']:.4f}")
    if body.get("limit_hit"):
        print(f"limit hit: {body['limit_hit']}")
    elif body.get("error"):
        print(f"error: {body['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="question; without it, asks interactively")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.question:
        ask(args.url, " ".join(args.question))
        return
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        ask(args.url, question)


if __name__ == "__main__":
    main()
