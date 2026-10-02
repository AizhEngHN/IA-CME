from pathlib import Path

from spl.data_model import Product, TSet, TestSuite


def parse_ts_file(path: str | Path) -> TestSuite:
    products = []
    for line in _read_non_empty_lines(path):
        assignments = [int(value.strip()) for value in line.split(";") if value.strip()]
        products.append(Product(assignments))

    return TestSuite(products)


def parse_tsets_file(path: str | Path) -> set[TSet]:
    tsets = set()
    for line in _read_non_empty_lines(path):
        values = [int(value.strip()) for value in line.split(";") if value.strip()]
        tsets.add(TSet(values))

    return tsets


def parse_fitness_file(path: str | Path) -> list[float]:
    return [float(line) for line in _read_non_empty_lines(path)]


def parse_qd_score_file(path: str | Path) -> float:
    lines = _read_non_empty_lines(path)
    if len(lines) != 1:
        raise ValueError("QDscoreFitness file must contain exactly one value")

    return float(lines[0])


def parse_run_outputs(directory: str | Path, run: int, suite_index: int = 0) -> dict:
    directory = Path(directory)
    return {
        "test_suite": parse_ts_file(directory / f"TS.{run}_{suite_index}"),
        "fitness": parse_fitness_file(directory / f"Fitness.{run}"),
        "qd_score": parse_qd_score_file(directory / f"QDscoreFitness.{run}"),
    }


def _read_non_empty_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]
