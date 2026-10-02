from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable, Sequence

from iacme.sampler import parse_key_value_output, resolve_sat4j_jar
from spl.data_model import Product
from spl.dimacs_validation import DimacsCNF, validate_product_against_cnf


@dataclass(frozen=True)
class TargetSnapshot:
    snapshot_id: int
    generation: int
    interaction_indices: tuple[int, ...]
    interactions: tuple[tuple[int, ...], ...]
    created_at: float


@dataclass(frozen=True)
class TargetedProductBatch:
    snapshot: TargetSnapshot
    products: tuple[Product, ...]
    product_target_indices: tuple[int, ...]
    requested: int
    generated: int
    valid: int
    target_hits: int
    duplicate_with_snapshot_pool: int
    java_report: dict[str, str]
    started_at: float
    finished_at: float


def build_targeted_generator_commands(
    repo_root: str | Path,
    model_path: str | Path,
    targets_path: str | Path,
    output_path: str | Path,
    seed: int,
    random_assumption_count: int,
    max_attempts_per_target: int,
    randomize_solver_order: bool = False,
    javac_executable: str = "javac",
    java_executable: str = "java",
    sat4j_jar: str | Path | None = None,
    classes_dir: str | Path | None = None,
) -> tuple[list[str], list[str], Path]:
    repo_root = Path(repo_root).resolve()
    classes_dir = Path(classes_dir or repo_root / "tmp" / "targeted_sat4j").resolve()
    source = repo_root / "tools" / "java_runner" / "TargetedSAT4JProductGenerator.java"
    classpath = str(resolve_sat4j_jar(repo_root, sat4j_jar))
    compile_command = [javac_executable, "-cp", classpath, "-d", str(classes_dir), str(source)]
    run_command = [
        java_executable,
        "-cp",
        os.pathsep.join([str(classes_dir), classpath]),
        "TargetedSAT4JProductGenerator",
        str(Path(model_path).resolve()),
        str(Path(targets_path).resolve()),
        str(Path(output_path).resolve()),
        str(seed),
        str(random_assumption_count),
        str(max_attempts_per_target),
        str(bool(randomize_solver_order)).lower(),
    ]
    return compile_command, run_command, classes_dir


def write_target_snapshot(snapshot: TargetSnapshot, path: str | Path) -> None:
    lines = [";".join(str(value) for value in interaction) for interaction in snapshot.interactions]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_targeted_products(path: str | Path) -> list[tuple[int, Product]]:
    products = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            target_text, separator, assignments_text = line.strip().partition("\t")
            if not separator:
                raise ValueError("invalid targeted product output line")
            assignments = [int(value) for value in assignments_text.split(";") if value]
            products.append((int(target_text), Product(assignments)))
    return products


def validate_targeted_products(
    parsed_products: Iterable[tuple[int, Product]],
    snapshot: TargetSnapshot,
    cnf: DimacsCNF,
    existing_fingerprints: frozenset[frozenset[int]],
    java_report: dict[str, str],
    started_at: float,
    finished_at: float,
) -> TargetedProductBatch:
    accepted = []
    targets = []
    valid = 0
    target_hits = 0
    duplicate_with_pool = 0
    seen = set(existing_fingerprints)
    parsed_products = list(parsed_products)
    for target_index, product in parsed_products:
        if target_index < 0 or target_index >= len(snapshot.interactions):
            continue
        validation = validate_product_against_cnf(product, cnf)
        if not validation.valid:
            continue
        valid += 1
        assignments = frozenset(product.assignments)
        target_hit = set(snapshot.interactions[target_index]).issubset(assignments)
        if not target_hit:
            continue
        target_hits += 1
        if assignments in seen:
            duplicate_with_pool += 1
            continue
        seen.add(assignments)
        accepted.append(product)
        targets.append(target_index)
    return TargetedProductBatch(
        snapshot=snapshot,
        products=tuple(accepted),
        product_target_indices=tuple(targets),
        requested=len(snapshot.interactions),
        generated=len(parsed_products),
        valid=valid,
        target_hits=target_hits,
        duplicate_with_snapshot_pool=duplicate_with_pool,
        java_report=java_report,
        started_at=started_at,
        finished_at=finished_at,
    )


class ConcurrentSATReplenisher:
    def __init__(
        self,
        compile_command: Sequence[str],
        run_command_template: Sequence[str],
        classes_dir: str | Path,
        work_dir: str | Path,
        cnf: DimacsCNF,
        compile_on_prepare: bool = True,
    ) -> None:
        self.compile_command = list(compile_command)
        self.run_command_template = list(run_command_template)
        self.classes_dir = Path(classes_dir)
        self.work_dir = Path(work_dir)
        self.cnf = cnf
        self.compile_on_prepare = compile_on_prepare
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="targeted-sat4j")
        self.future: Future[TargetedProductBatch] | None = None

    def prepare(self) -> None:
        self.classes_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.compile_on_prepare:
            _run_checked(self.compile_command, "targeted SAT4J compilation")

    def submit(
        self,
        snapshot: TargetSnapshot,
        existing_products: Iterable[Product],
    ) -> bool:
        if self.future is not None:
            return False
        existing = frozenset(frozenset(product.assignments) for product in existing_products)
        self.future = self.executor.submit(self._generate, snapshot, existing)
        return True

    def poll(self) -> TargetedProductBatch | None:
        if self.future is None or not self.future.done():
            return None
        result = self.future.result()
        self.future = None
        return result

    def finish(self) -> TargetedProductBatch | None:
        if self.future is None:
            return None
        result = self.future.result()
        self.future = None
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def _generate(
        self,
        snapshot: TargetSnapshot,
        existing: frozenset[frozenset[int]],
    ) -> TargetedProductBatch:
        targets_path = self.work_dir / f"targets_{snapshot.snapshot_id}.txt"
        output_path = self.work_dir / f"products_{snapshot.snapshot_id}.txt"
        write_target_snapshot(snapshot, targets_path)
        command = list(self.run_command_template)
        command[-6] = str(targets_path.resolve())
        command[-5] = str(output_path.resolve())
        command[-4] = str(int(command[-4]) + snapshot.snapshot_id)
        started_at = time.perf_counter()
        completed = _run_checked(command, "targeted SAT4J generation")
        result = validate_targeted_products(
            parse_targeted_products(output_path),
            snapshot,
            self.cnf,
            existing,
            parse_key_value_output(completed.stdout),
            started_at,
            started_at,
        )
        return replace(result, finished_at=time.perf_counter())


def _run_checked(command: Sequence[str], operation: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(command), text=True, capture_output=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "no process output").strip()
        raise RuntimeError(f"{operation} failed with exit code {completed.returncode}:\n{details}")
    return completed
