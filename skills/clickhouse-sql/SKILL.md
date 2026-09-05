---
name: clickhouse-sql
description: Схема снапшота товаров в ClickHouse, особенности кликового диалекта и ловушки этих данных. Нужен всякий раз, когда вопрос про агрегат, счёт, сортировку или сравнение цен переводится в SQL.
---

# Снапшот товаров в ClickHouse

Одна плоская таблица `products`: карточки Wildberries и Ozon на момент одного снимка.
Цены и атрибуты берутся только отсюда — ходить в интернет нельзя, товара нет в
таблице значит его нет.

```sql
CREATE TABLE products (
    marketplace     LowCardinality(String),  -- 'wb' | 'ozon'
    id              String,                  -- идентификатор внутри площадки
    url             String,
    title           String,
    description     String,
    price_rub       UInt32,                  -- целое число рублей
    category        LowCardinality(String),  -- два уровня через ' / '
    characteristics Map(String, String),
    review_count    Nullable(UInt32),
    collected_at    DateTime
)
ENGINE = ReplacingMergeTree(collected_at)
ORDER BY (marketplace, id)
```

Ключ — пара `(marketplace, id)`: один и тот же `id` может встретиться на обеих
площадках, поэтому `id` сам по себе не уникален.

## Ловушки

**`FINAL` обязателен в каждом запросе.** Движок `ReplacingMergeTree` до слияния
кусков хранит все версии строки. Без `FINAL` повторно залитая карточка попадёт в
`count()` дважды, а в `avg(price_rub)` — со старой ценой.

```sql
SELECT count() FROM products FINAL     -- 394, верно
SELECT count() FROM products           -- может быть больше
```

**Значения характеристик регистрозависимы, а `lower()` кириллицу не берёт.**
`characteristics['Цвет'] = 'черный'` даёт 0 строк, `'Черный'` — 11. Привести регистр
можно только `lowerUTF8()`: `lower()` в ClickHouse работает по ASCII и возвращает
`'Китай'` как есть, поэтому `lower(...) = 'китай'` молча даёт 0 вместо 191.
Для подстроки — `ILIKE`, он с кириллицей работает.

**Ключи характеристик у площадок почти не пересекаются.** Один и тот же смысл назван
по-разному: страна — это `Страна-изготовитель` у Ozon (199 карточек) и
`Страна производства` у Wildberries (60). `Цвет` есть у 158 карточек Ozon и только у
20 Wildberries, `Тип` и `Материал` — практически только Ozon; у Wildberries из
частых ключей в основном габариты упаковки и вес. Агрегат по ключу без
`marketplace`-разреза почти всегда описывает Ozon, а не обе площадки.

**Категории двух площадок — разные таксономии.** «Дом / Кухня» у Wildberries против
«Дом и сад / Посуда и кухонные принадлежности» у Ozon. Сравнивать категории между
площадками нельзя: одноимённых просто нет. Сравнение цен между площадками имеет
смысл либо целиком, либо по конкретной паре товаров.

**У 14 карточек Wildberries категория пустая** (`category = ''`). В группировке по
категориям это отдельная строка, а не пропуск.

**`review_count` заполнен у 4 карточек из 394.** Это не пробел в данных: товары
непопулярные, отзывов у них действительно нет. Считать по нему средние бессмысленно,
а «сколько товаров без отзывов» — законный вопрос.

**Описание есть не у всех:** пустое у 49 карточек Ozon и у 14 Wildberries.

**Снапшот один.** Все строки с одним `collected_at`, вопросов про динамику цен нет.

## Диалект

- `LIMIT n`, а не `TOP n`.
- `count()` без аргумента вместо `count(*)`.
- Доступ к характеристике — `characteristics['Ключ']`; отсутствующий ключ даёт
  пустую строку, а не NULL.
- Есть ли ключ вообще — `has(mapKeys(characteristics), 'Ключ')`.
- Развернуть Map в строки — `arrayJoin(mapKeys(characteristics))`.
- Медиана — `quantile(0.5)(price_rub)` или `median(price_rub)`.
- Условный агрегат короче через комбинатор `-If`: `countIf(price_rub < 5000)`,
  `avgIf(price_rub, marketplace = 'wb')`.
- Округление — `round(x, 2)`.
- Регистр текста — `lowerUTF8()` / `upperUTF8()`, ASCII-версии кириллицу не трогают.

## Примеры

Средняя цена в категории:

```sql
SELECT round(avg(price_rub), 2)
FROM products FINAL
WHERE marketplace = 'wb' AND category = 'Дом / Кухня'
```

Сколько товаров дешевле пяти тысяч:

```sql
SELECT count() FROM products FINAL WHERE price_rub < 5000
```

Топ-5 самых дешёвых в категории:

```sql
SELECT title, price_rub
FROM products FINAL
WHERE marketplace = 'ozon' AND category = 'Детские товары / Игрушки и игры'
ORDER BY price_rub ASC
LIMIT 5
```

Сравнение площадок:

```sql
SELECT marketplace, round(avg(price_rub)) AS avg_price, count() AS n
FROM products FINAL
GROUP BY marketplace
ORDER BY marketplace
```

Самые дорогие категории:

```sql
SELECT category, round(avg(price_rub)) AS avg_price
FROM products FINAL
WHERE marketplace = 'ozon'
GROUP BY category
ORDER BY avg_price DESC
LIMIT 5
```

Частотные значения характеристики:

```sql
SELECT characteristics['Цвет'] AS colour, count() AS n
FROM products FINAL
WHERE has(mapKeys(characteristics), 'Цвет')
GROUP BY colour
ORDER BY n DESC
LIMIT 10
```

Фильтр по характеристике, без оглядки на регистр:

```sql
SELECT count()
FROM products FINAL
WHERE lowerUTF8(characteristics['Страна-изготовитель']) = 'китай'
```
