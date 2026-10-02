"""Two-stage IA-CME runner over user-supplied DIMACS and interaction files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import torch

from iacme.pool_construction import build_support_deficit_pool
from iacme.sampler import build_sampler_commands, resolve_sat4j_jar
from iacme.sat import ConcurrentSATReplenisher, build_targeted_generator_commands
from spl.data_model import TestSuite
from spl.dimacs_validation import parse_dimacs_cnf, validate_product_pool
from spl.evaluator import fitness_coverage
from spl.evox_paper_aligned import (
    PaperAlignedEvoXConfig,
    run_paper_aligned_evox_map_elites_tensor_data,
)
from spl.interaction_guidance import interaction_support_counts, product_interaction_matrix
from spl.output_parser import parse_ts_file, parse_tsets_file
from spl.tensor_coverage import tensorize_spl_data


@dataclass(frozen=True)
class IACMEConfig:
    initial_pool_size: int = 100
    maximum_pool_size: int = 3000
    support_threshold: int = 1
    targets_per_batch: int = 32
    initial_assumption_count: int = 16
    random_assumption_count: int = 8
    max_attempts_per_product: int = 10000
    max_attempts_per_target: int = 1000
    lower_bound: int = 2
    upper_bound: int = 10
    search_evaluations: int = 4096
    batch_size: int = 64
    guidance_eta: float = 0.5
    candidate_sample_size: int = 64
    interaction_chunk_size: int = 64
    initial_seed: int = 0
    target_seed: int = 0
    sat_seed: int = 0
    search_seed: int = 0
    device: str = "auto"
    pool_device: str = "cpu"

    def validate(self) -> None:
        positive_fields = (
            "initial_pool_size", "maximum_pool_size", "support_threshold",
            "targets_per_batch", "max_attempts_per_product", "max_attempts_per_target",
            "lower_bound", "upper_bound", "batch_size", "candidate_sample_size",
            "interaction_chunk_size",
        )
        if any(getattr(self, name) <= 0 for name in positive_fields):
            raise ValueError("Pool sizes, size bounds, support and batch settings must be positive.")
        if self.maximum_pool_size < max(self.initial_pool_size, self.upper_bound):
            raise ValueError("Maximum pool size must accommodate the initial pool and suite sizes.")
        if self.upper_bound < self.lower_bound:
            raise ValueError("Upper suite-size bound must be at least the lower bound.")
        if self.search_evaluations < 0 or min(self.initial_assumption_count, self.random_assumption_count) < 0:
            raise ValueError("Evaluation and assumption counts must be non-negative.")
        if not 0.0 <= self.guidance_eta <= 1.0:
            raise ValueError("Guidance probability must be in [0, 1].")


def _run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_iacme(
    model_path: str | Path,
    interactions_path: str | Path,
    sat4j_jar: str | Path,
    config: IACMEConfig | None = None,
    java: str = "java",
    javac: str = "javac",
) -> dict[str, Any]:
    """Construct a qualified pool, freeze it, search, and validate every returned elite.

    Interactions must be valid for the supplied model and are read as signed literals,
    one semicolon-separated interaction per line. No dataset is bundled with this code.
    SAT calls occur only during Stage I; Java compilation is outside reported timings.
    """
    config = config or IACMEConfig()
    config.validate()
    package_root = Path(__file__).resolve().parents[2]
    jar = resolve_sat4j_jar(package_root, sat4j_jar)
    cnf = parse_dimacs_cnf(model_path)
    interactions = tuple(sorted(parse_tsets_file(interactions_path), key=lambda item: tuple(sorted(item.values))))
    if cnf.num_variables < 1 or not interactions:
        raise ValueError("The feature model and supplied interaction set must be non-empty.")
    strengths = {len(item.values) for item in interactions}
    if len(strengths) != 1 or 0 in strengths:
        raise ValueError("All supplied interactions must have the same positive strength.")
    for interaction in interactions:
        if any(abs(value) > cnf.num_variables or -value in interaction.values for value in interaction.values):
            raise ValueError("Interactions must contain consistent, in-range signed literals.")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if config.device == "auto" else config.device)
    pool_device = device if config.pool_device == "auto" else torch.device(config.pool_device)

    with tempfile.TemporaryDirectory(prefix="iacme_") as temporary:
        work = Path(temporary)
        canonical_model = work / "model.cnf"
        canonical_model.write_text(
            f"p cnf {cnf.num_variables} {len(cnf.clauses)}\n"
            + "".join(" ".join(map(str, clause)) + " 0\n" for clause in cnf.clauses),
            encoding="utf-8",
        )
        classes = work / "classes"
        classes.mkdir()
        initial_path = work / "initial_products.txt"
        initial_compile, initial_run = build_sampler_commands(
            package_root, canonical_model, initial_path, config.initial_pool_size,
            config.initial_seed, config.initial_assumption_count, config.max_attempts_per_product,
            java_executable=java, javac_executable=javac, sat4j_jar=jar, classes_dir=classes,
        )
        targeted_compile, targeted_run, _ = build_targeted_generator_commands(
            package_root, canonical_model, work / "targets.txt", work / "products.txt",
            config.sat_seed, config.random_assumption_count, config.max_attempts_per_target,
            True, java_executable=java, javac_executable=javac, sat4j_jar=jar, classes_dir=classes,
        )
        _run_checked(initial_compile)
        _run_checked(targeted_compile)

        pool_started = time.perf_counter()
        _run_checked(initial_run)
        initial_products = tuple(parse_ts_file(initial_path).products)
        validation = validate_product_pool(initial_products, cnf)
        if (len(initial_products) != config.initial_pool_size
                or not validation["all_products_valid"] or not validation["all_products_unique"]):
            raise RuntimeError("Initial products failed independent validity/uniqueness checks.")
        replenisher = ConcurrentSATReplenisher(
            targeted_compile, targeted_run, classes, work / "sat_batches", cnf,
            compile_on_prepare=False,
        )
        pool = build_support_deficit_pool(
            initial_products=initial_products,
            valid_tsets=interactions,
            target_size=config.maximum_pool_size,
            targets_per_batch=config.targets_per_batch,
            seed=config.target_seed,
            device=pool_device,
            replenisher=replenisher,
            support_thresholds=(config.support_threshold,),
            stop_when_support_thresholds_reached=True,
        )
        if config.support_threshold not in pool.support_checkpoints:
            raise RuntimeError("Pool size limit reached before support qualification; search was not started.")
        products = pool.support_checkpoints[config.support_threshold]
        validation = validate_product_pool(products, cnf)
        if not validation["all_products_valid"] or not validation["all_products_unique"]:
            raise RuntimeError("Qualified pool failed independent validity/uniqueness checks.")
        _synchronize(pool_device)
        pool_seconds = time.perf_counter() - pool_started

    if len(products) < config.upper_bound:
        raise ValueError("The qualified pool is smaller than the largest requested suite size.")
    _synchronize(device)
    matrix_started = time.perf_counter()
    data = tensorize_spl_data(products, interactions, device=device, include_tset_presence=False)
    matrix = product_interaction_matrix(data, product_chunk_size=config.interaction_chunk_size)
    minimum_support = int(interaction_support_counts(matrix).min().item())
    if minimum_support < config.support_threshold:
        raise RuntimeError("Cross-stage coverage representation failed support qualification.")
    _synchronize(device)
    matrix_seconds = time.perf_counter() - matrix_started
    search_config = PaperAlignedEvoXConfig(
        lower_bound=config.lower_bound, upper_bound=config.upper_bound,
        generations=math.ceil(config.search_evaluations / config.batch_size),
        batch_size=config.batch_size, seed=config.search_seed, device=device,
        guidance_eta=config.guidance_eta, guidance_candidate_sample_size=config.candidate_sample_size,
        interaction_chunk_size=config.interaction_chunk_size, search_evaluations=config.search_evaluations,
    )
    result = run_paper_aligned_evox_map_elites_tensor_data(
        data, len(products), search_config, coverage_matrix=matrix,
    )
    archive = {}
    for cell in result.archive.occupied_cells():
        indices = [int(value) for value in result.archive.elite_candidates[cell].cpu().tolist() if value >= 0]
        suite = TestSuite(products[index] for index in indices)
        fitness = float(result.archive.elite_fitness[cell].item())
        size = config.lower_bound + cell
        if len(indices) != size or len(set(indices)) != size:
            raise RuntimeError("Returned elite has invalid cardinality or duplicate products.")
        if not math.isclose(fitness, fitness_coverage(suite, interactions), abs_tol=1e-9, rel_tol=0.0):
            raise RuntimeError("Returned elite failed independent exact-coverage reevaluation.")
        archive[str(size)] = {
            "fitness": fitness,
            "product_indices": indices,
            "products": [list(product.assignments) for product in suite.products],
        }
    return {
        "method": "Interaction-Aware Coupled MAP-Elites (IA-CME)",
        "config": asdict(config),
        "device": str(device),
        "interaction_strength": next(iter(strengths)),
        "num_interactions": len(interactions),
        "pool_size": len(products),
        "minimum_support": minimum_support,
        "pool_ceiling_percent": pool.final_ceiling_percent,
        "best_fitness": result.best_fitness,
        "qd_score": result.qd_score,
        "initial_evaluations": result.initial_evaluations,
        "search_evaluations": result.search_evaluations,
        "guided_mutations": result.guided_mutations,
        "timings_seconds": {
            "pool_construction": pool_seconds,
            "representation": matrix_seconds,
            "archive_initialization": result.initialization_elapsed_seconds,
            "search": result.search_elapsed_seconds,
        },
        "archive": archive,
        "history": result.history,
    }
