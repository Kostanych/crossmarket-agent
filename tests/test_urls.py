import pytest

from crossmarket.extraction.urls import parse_ozon_id, parse_wb_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.wildberries.ru/catalog/123456789/detail.aspx", "123456789"),
        ("https://www.wildberries.ru/catalog/123456789/detail.aspx?targetUrl=GP", "123456789"),
        ("wildberries.ru/catalog/1/detail.aspx", "1"),
        ("123456789", "123456789"),
        ("  123456789  ", "123456789"),
        ("", None),
        ("https://www.wildberries.ru/brands/nothing", None),
    ],
)
def test_parse_wb_id(url: str, expected: str | None) -> None:
    assert parse_wb_id(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.ozon.ru/product/naushniki-nothing-ear-1785773893/", "1785773893"),
        ("https://www.ozon.ru/product/naushniki-nothing-ear-1785773893/?asb=abc", "1785773893"),
        # В слаге есть цифры — берём последнюю группу, а не первую.
        ("https://www.ozon.ru/product/nothing-ear-2025-1785773893/", "1785773893"),
        ("https://www.ozon.ru/product/1785773893/", "1785773893"),
        ("1785773893", "1785773893"),
        ("", None),
        ("https://www.ozon.ru/category/naushniki-15548/", None),
    ],
)
def test_parse_ozon_id(url: str, expected: str | None) -> None:
    assert parse_ozon_id(url) == expected
