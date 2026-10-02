from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spl.data_model import Product


@dataclass(frozen=True)
class DimacsCNF:
    num_variables: int
    declared_num_clauses: int
    clauses: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ProductValidation:
    valid: bool
    complete_assignment: bool
    unsatisfied_clause_index: int | None
    message: str


def parse_dimacs_cnf(path: str | Path) -> DimacsCNF:
    num_variables = None
    declared_num_clauses = None
    clauses = []
    current_clause = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("c") or line.startswith("%"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) != 4 or parts[1].lower() != "cnf":
                    raise ValueError(f"invalid DIMACS header on line {line_number}")
                num_variables = int(parts[2])
                declared_num_clauses = int(parts[3])
                continue
            if num_variables is None:
                raise ValueError("DIMACS clause found before header")

            for token in line.split():
                literal = int(token)
                if literal == 0:
                    if not current_clause:
                        raise ValueError(f"empty DIMACS clause ending on line {line_number}")
                    clauses.append(tuple(current_clause))
                    current_clause = []
                else:
                    if abs(literal) > num_variables:
                        raise ValueError(f"literal out of range on line {line_number}: {literal}")
                    current_clause.append(literal)

    if num_variables is None or declared_num_clauses is None:
        raise ValueError("DIMACS header not found")
    if current_clause:
        raise ValueError("unterminated DIMACS clause")
    if len(clauses) != declared_num_clauses:
        raise ValueError(
            f"DIMACS clause count mismatch: declared {declared_num_clauses}, parsed {len(clauses)}"
        )
    return DimacsCNF(num_variables, declared_num_clauses, tuple(clauses))


def validate_product_against_cnf(product: Product, cnf: DimacsCNF) -> ProductValidation:
    assignments = product.assignments
    assignment_set = set(assignments)
    complete = (
        len(assignments) == cnf.num_variables
        and len(assignment_set) == cnf.num_variables
        and all(
            variable in assignment_set or -variable in assignment_set
            for variable in range(1, cnf.num_variables + 1)
        )
        and not any(
            variable in assignment_set and -variable in assignment_set
            for variable in range(1, cnf.num_variables + 1)
        )
    )
    if not complete:
        return ProductValidation(False, False, None, "product is not a complete, consistent assignment")

    for index, clause in enumerate(cnf.clauses):
        if not any(literal in assignment_set for literal in clause):
            return ProductValidation(False, True, index, f"DIMACS clause {index} is not satisfied")
    return ProductValidation(True, True, None, "valid")


def validate_product_pool(
    products: Iterable[Product],
    cnf: DimacsCNF,
) -> dict:
    products = tuple(products)
    validations = tuple(validate_product_against_cnf(product, cnf) for product in products)
    unique_products = len({frozenset(product.assignments) for product in products})
    invalid = [index for index, validation in enumerate(validations) if not validation.valid]
    return {
        "num_products": len(products),
        "num_unique_products": unique_products,
        "duplicate_products": len(products) - unique_products,
        "num_valid_products": len(products) - len(invalid),
        "num_invalid_products": len(invalid),
        "invalid_product_indices": invalid,
        "all_products_valid": not invalid,
        "all_products_unique": unique_products == len(products),
        "num_variables": cnf.num_variables,
        "num_clauses": len(cnf.clauses),
    }
