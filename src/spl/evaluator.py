from collections.abc import Iterable

from spl.data_model import Product, TSet, TestSuite


def product_covers_tset(product: Product, tset: TSet) -> bool:
    return tset.values.issubset(set(product.assignments))


def suite_covers_tset(test_suite: TestSuite, tset: TSet) -> bool:
    return any(product_covers_tset(product, tset) for product in test_suite.products)


def covered_tsets(test_suite: TestSuite, valid_tsets: Iterable[TSet]) -> set[TSet]:
    return {tset for tset in valid_tsets if suite_covers_tset(test_suite, tset)}


def coverage_count(test_suite: TestSuite, valid_tsets: Iterable[TSet]) -> int:
    valid_tsets = tuple(valid_tsets)
    _require_valid_tsets(valid_tsets)
    return len(covered_tsets(test_suite, valid_tsets))


def coverage_percent(test_suite: TestSuite, valid_tsets: Iterable[TSet]) -> float:
    valid_tsets = tuple(valid_tsets)
    _require_valid_tsets(valid_tsets)
    return coverage_count(test_suite, valid_tsets) / len(valid_tsets) * 100.0


def fitness_coverage(test_suite: TestSuite, valid_tsets: Iterable[TSet]) -> float:
    return coverage_percent(test_suite, valid_tsets)


def _require_valid_tsets(valid_tsets: tuple[TSet, ...]) -> None:
    if not valid_tsets:
        raise ValueError("valid_tsets must not be empty")
