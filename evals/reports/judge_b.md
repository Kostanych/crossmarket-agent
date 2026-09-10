# Судья-B: согласие с истиной на вердиктах матчера

Сгенерировано `python -m evals judge` 2026-09-10. Файл перезаписывается каждым прогоном — история в git.

Судья оценивает вердикт модели подтверждения вместе с обоснованием, а не решает пару сам. Истина — совпал ли вердикт тула с меткой разметчика.

## Конфигурация

| параметр | значение |
|---|---|
| судья | claude-sonnet-5 |
| кейсов | 61 |
| сплиты | test 61 |
| положительный класс | bad |
| стоимость прогона | $0.94 |

### Согласие судьи с истиной (claude-sonnet-5)

| сплит | кейсов | согласие | baseline | precision | recall | F1 | неразобрано | $ |
|---|---|---|---|---|---|---|---|---|
| test (отчётный) | 61 | 0.770 | 0.656 | 0.769 | 0.476 | 0.588 | 0 | 0.940 |

### По стратам (весь набор)

| страта | кейсов | плохих по истине | согласие |
|---|---|---|---|
| match | 47 | 16 | 0.872 |
| no_match | 14 | 5 | 0.429 |

### Расхождения судьи с истиной

| кейс | сплит | судья | истина |
|---|---|---|---|
| тул: match, метка: no_match | test | good | bad |
| тул: match, метка: no_match | test | good | bad |
| тул: no_match, метка: no_match | test | bad | good |
| тул: match, метка: no_match | test | good | bad |
| тул: no_match, метка: no_match | test | bad | good |
| тул: match, метка: no_match | test | good | bad |
| тул: no_match, метка: no_match | test | bad | good |
| тул: no_match, метка: match | test | good | bad |
| тул: match, метка: no_match | test | good | bad |
| тул: no_match, метка: match | test | good | bad |
| тул: no_match, метка: match | test | good | bad |
| тул: no_match, метка: match | test | good | bad |
| тул: no_match, метка: match | test | good | bad |
| тул: no_match, метка: match | test | good | bad |
