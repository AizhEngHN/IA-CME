from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Product:
    # Keep input order to reproduce Java output files. Coverage and membership
    # checks should still use set semantics, matching Java Product extends HashSet<Integer>.
    assignments: tuple[int, ...]

    def __init__(self, assignments: Iterable[int]):
        normalized = tuple(int(value) for value in assignments)
        if any(value == 0 for value in normalized):
            raise ValueError("feature assignments must be non-zero signed integers")

        object.__setattr__(self, "assignments", normalized)

    def selected_features(self) -> tuple[int, ...]:
        return tuple(value for value in self.assignments if value > 0)


@dataclass(frozen=True)
class TSet:
    values: frozenset[int]

    def __init__(self, values: Iterable[int]):
        normalized = frozenset(int(value) for value in values)
        if any(value == 0 for value in normalized):
            raise ValueError("t-set values must be non-zero signed integers")

        object.__setattr__(self, "values", normalized)

    def size(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class TestSuite:
    products: tuple[Product, ...]

    def __init__(self, products: Iterable[Product]):
        object.__setattr__(self, "products", tuple(products))

    def size(self) -> int:
        return len(self.products)
