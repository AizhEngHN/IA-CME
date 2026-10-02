from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from evox.core import Algorithm, Mutable, Problem
from evox.workflows import EvalMonitor, StdWorkflow

from spl.data_model import Product, TSet
from spl.tensor_archive import TensorMAPElitesArchive
from spl.tensor_coverage import TensorSPLData, fitness_coverage_tensor, tensorize_spl_data
from spl.interaction_guidance import (
    archive_interaction_hardness,
    interaction_aware_mutate_candidate_batch,
    matrix_coverage_percent,
    product_interaction_matrix,
)
from spl.tensor_variation import (
    candidate_sizes,
    paper_aligned_initial_candidate_batch,
    paper_aligned_mutate_candidate_batch,
)


class EvoXSPLCoverageProblem(Problem):
    """Exact t-wise coverage problem backed by tensors on the workflow device."""

    def __init__(self, data: TensorSPLData):
        super().__init__()
        if data.tset_presence is None:
            raise ValueError("dense tset presence is required without a coverage matrix")
        self.product_presence = Mutable(data.product_presence)
        self.tset_presence = Mutable(data.tset_presence)
        self.tset_indices = Mutable(data.tset_indices)
        self.tset_mask = Mutable(data.tset_mask)

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        return fitness_coverage_tensor(
            population,
            self.product_presence,
            self.tset_presence,
            self.tset_indices,
            self.tset_mask,
        )

    def append_product_presence(self, rows: torch.Tensor) -> None:
        if rows.ndim != 2 or rows.shape[1] != self.product_presence.shape[1]:
            raise ValueError("new product-presence rows have incompatible shape")
        self.product_presence = torch.cat(
            [self.product_presence, rows.to(self.product_presence.device, dtype=torch.bool)],
            dim=0,
        )


class EvoXSPLMatrixCoverageProblem(Problem):
    """Exact coverage evaluator backed only by the product-interaction matrix."""

    def __init__(self, coverage_matrix: torch.Tensor):
        super().__init__()
        self.coverage_matrix = Mutable(coverage_matrix)

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        return matrix_coverage_percent(population, self.coverage_matrix)


class PaperAlignedMAPElites(Algorithm):
    """Static-pool MAP-Elites following the original initialization and mutation rules."""

    def __init__(
        self,
        num_products: int,
        lower_bound: int,
        upper_bound: int,
        batch_size: int,
        seed: int,
        device: str | torch.device,
        coverage_matrix: torch.Tensor | None = None,
        guidance_eta: float = 0.0,
        guidance_candidate_sample_size: int = 64,
    ):
        super().__init__()
        if num_products < upper_bound:
            raise ValueError("num_products must be at least upper_bound for unique suites")
        if lower_bound <= 0 or upper_bound < lower_bound:
            raise ValueError("invalid suite-size bounds")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= guidance_eta <= 1.0:
            raise ValueError("guidance_eta must be in [0, 1]")
        if guidance_candidate_sample_size <= 0:
            raise ValueError("guidance_candidate_sample_size must be positive")
        if guidance_eta > 0.0 and coverage_matrix is None:
            raise ValueError("coverage_matrix is required when guidance is enabled")

        run_device = torch.device(device)
        self.num_products = num_products
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.batch_size = batch_size
        self.num_cells = upper_bound - lower_bound + 1
        self.guidance_eta = guidance_eta
        self.guidance_candidate_sample_size = guidance_candidate_sample_size
        self.generator = torch.Generator(device=run_device).manual_seed(seed)
        self.coverage_matrix = Mutable(
            torch.empty((0, 0), dtype=torch.bool, device=run_device)
            if coverage_matrix is None
            else coverage_matrix.to(device=run_device, dtype=torch.bool)
        )
        self.archive_candidates = Mutable(
            torch.full((self.num_cells, upper_bound), -1, dtype=torch.long, device=run_device)
        )
        self.archive_fitness = Mutable(
            torch.full((self.num_cells,), -torch.inf, dtype=torch.float64, device=run_device)
        )
        self.occupied = Mutable(torch.zeros((self.num_cells,), dtype=torch.bool, device=run_device))
        self.last_candidates = Mutable(torch.empty((0, upper_bound), dtype=torch.long, device=run_device))
        self.last_quality = Mutable(torch.empty((0,), dtype=torch.float64, device=run_device))
        self.initial_evaluations = 0
        self.search_evaluations = 0
        self.guided_mutations = Mutable(torch.zeros((), dtype=torch.int64, device=run_device))

    def init_step(self) -> None:
        candidates = paper_aligned_initial_candidate_batch(
            self.num_products,
            self.lower_bound,
            self.upper_bound,
            device=self.archive_candidates.device,
            generator=self.generator,
        )
        quality = -self.evaluate(candidates)
        self._update_archive(candidates, quality)
        self.last_candidates = candidates
        self.last_quality = quality
        self.initial_evaluations += self.num_cells

    def step(self) -> None:
        occupied_indices = self.occupied.nonzero(as_tuple=False).flatten()
        if occupied_indices.numel() == 0:
            raise RuntimeError("init_step must populate the archive before step")
        choices = torch.randint(
            0,
            int(occupied_indices.numel()),
            (self.batch_size,),
            device=self.archive_candidates.device,
            generator=self.generator,
        )
        parents = self.archive_candidates[occupied_indices[choices]].clone()
        if self.guidance_eta > 0.0:
            hardness = archive_interaction_hardness(
                self.archive_candidates,
                self.occupied,
                self.coverage_matrix,
            )
            candidates, guided_mask = interaction_aware_mutate_candidate_batch(
                parents,
                self.coverage_matrix,
                hardness,
                self.lower_bound,
                self.upper_bound,
                self.guidance_eta,
                self.guidance_candidate_sample_size,
                generator=self.generator,
            )
            self.guided_mutations += guided_mask.sum()
        else:
            candidates = paper_aligned_mutate_candidate_batch(
                parents,
                self.num_products,
                self.lower_bound,
                self.upper_bound,
                generator=self.generator,
            )
        quality = -self.evaluate(candidates)
        self._update_archive(candidates, quality)
        self.last_candidates = candidates
        self.last_quality = quality
        self.search_evaluations += self.batch_size

    def record_step(self) -> dict[str, torch.Tensor]:
        """Expose auxiliary state using the EvoX 1.3 monitoring interface."""
        return {"pop": self.last_candidates, "fit": -self.last_quality}

    def _update_archive(self, candidates: torch.Tensor, quality: torch.Tensor) -> None:
        descriptors = candidate_sizes(candidates)
        cells = descriptors - self.lower_bound
        batch_best = torch.full_like(self.archive_fitness, -torch.inf)
        batch_best.scatter_reduce_(0, cells, quality, reduce="amax", include_self=True)

        row_ids = torch.arange(candidates.shape[0], device=candidates.device)
        winner_rows = torch.full(
            (self.num_cells,), candidates.shape[0], dtype=torch.long, device=candidates.device
        )
        is_cell_best = quality == batch_best[cells]
        winner_rows.scatter_reduce_(
            0,
            cells,
            torch.where(is_cell_best, row_ids, candidates.shape[0]),
            reduce="amin",
            include_self=True,
        )
        has_candidate = winner_rows < candidates.shape[0]
        improves = has_candidate & (batch_best > self.archive_fitness)
        safe_rows = winner_rows.clamp_max(candidates.shape[0] - 1)
        selected = candidates[safe_rows]
        self.archive_candidates[improves] = selected[improves]
        self.archive_fitness[improves] = batch_best[improves]
        self.occupied[improves] = True

    def archive_snapshot(self) -> TensorMAPElitesArchive:
        archive = TensorMAPElitesArchive(
            self.lower_bound,
            self.upper_bound,
            self.upper_bound,
            device=self.archive_candidates.device,
        )
        archive.elite_candidates.copy_(self.archive_candidates)
        archive.elite_fitness.copy_(self.archive_fitness)
        archive.occupied.copy_(self.occupied)
        return archive

    def append_product_coverage(self, rows: torch.Tensor) -> None:
        if self.guidance_eta <= 0.0:
            raise ValueError("product coverage can only be appended when guidance is enabled")
        if rows.ndim != 2 or rows.shape[1] != self.coverage_matrix.shape[1]:
            raise ValueError("new product-coverage rows have incompatible shape")
        self.coverage_matrix = torch.cat(
            [self.coverage_matrix, rows.to(self.coverage_matrix.device, dtype=torch.bool)],
            dim=0,
        )
        self.num_products = int(self.coverage_matrix.shape[0])


@dataclass(frozen=True)
class PaperAlignedEvoXConfig:
    lower_bound: int
    upper_bound: int
    generations: int
    batch_size: int
    seed: int = 0
    device: str | torch.device | None = None
    guidance_eta: float = 0.0
    guidance_candidate_sample_size: int = 64
    interaction_chunk_size: int = 64
    search_evaluations: int | None = None
    guidance_eta_final: float | None = None
    guidance_switch_evaluations: int | None = None


@dataclass(frozen=True)
class PaperAlignedEvoXResult:
    archive: TensorMAPElitesArchive
    initialization: dict[str, Any]
    history: list[dict[str, Any]]
    best_candidate: torch.Tensor | None
    best_fitness: float
    qd_score: float
    initial_evaluations: int
    search_evaluations: int
    initialization_elapsed_seconds: float
    search_elapsed_seconds: float
    guided_mutations: int
    guidance_preparation_seconds: float
    device: torch.device


def run_paper_aligned_evox_map_elites(
    valid_products: list[Product],
    valid_tsets: list[TSet] | set[TSet] | tuple[TSet, ...],
    config: PaperAlignedEvoXConfig,
) -> PaperAlignedEvoXResult:
    _validate_inputs(valid_products, valid_tsets, config)
    run_device = _resolve_device(config.device)
    data = tensorize_spl_data(valid_products, valid_tsets, device=run_device)
    return run_paper_aligned_evox_map_elites_tensor_data(data, len(valid_products), config)


def run_paper_aligned_evox_map_elites_tensor_data(
    data: TensorSPLData,
    num_products: int,
    config: PaperAlignedEvoXConfig,
    record_history: bool = True,
    coverage_matrix: torch.Tensor | None = None,
) -> PaperAlignedEvoXResult:
    run_device = data.device
    if config.device is not None:
        requested_device = torch.device(config.device)
        same_type = requested_device.type == run_device.type
        same_index = requested_device.index is None or requested_device.index == run_device.index
        if not (same_type and same_index):
            raise ValueError("config.device must match tensor data device")
    if num_products != data.product_presence.shape[0]:
        raise ValueError("num_products must match tensor data")
    if config.upper_bound > num_products:
        raise ValueError("upper_bound must not exceed the static product-pool size")
    if config.guidance_eta_final is not None and not 0.0 <= config.guidance_eta_final <= 1.0:
        raise ValueError("guidance_eta_final must be in [0, 1]")
    if config.guidance_switch_evaluations is not None and config.guidance_switch_evaluations < 0:
        raise ValueError("guidance_switch_evaluations must be non-negative")

    guidance_preparation_seconds = 0.0
    if config.guidance_eta > 0.0 and coverage_matrix is None:
        _synchronize(run_device)
        guidance_start = time.perf_counter()
        coverage_matrix = product_interaction_matrix(
            data,
            product_chunk_size=config.interaction_chunk_size,
        )
        _synchronize(run_device)
        guidance_preparation_seconds = time.perf_counter() - guidance_start
    if coverage_matrix is not None:
        expected_shape = (num_products, data.tset_indices.shape[0])
        if tuple(coverage_matrix.shape) != expected_shape:
            raise ValueError(f"coverage_matrix must have shape {expected_shape}")
        if coverage_matrix.device != run_device:
            raise ValueError("coverage_matrix must be on the tensor data device")

    problem = (
        EvoXSPLCoverageProblem(data)
        if coverage_matrix is None
        else EvoXSPLMatrixCoverageProblem(coverage_matrix)
    )
    algorithm = PaperAlignedMAPElites(
        num_products,
        config.lower_bound,
        config.upper_bound,
        config.batch_size,
        config.seed,
        run_device,
        coverage_matrix=coverage_matrix,
        guidance_eta=config.guidance_eta,
        guidance_candidate_sample_size=config.guidance_candidate_sample_size,
    )
    monitor = EvalMonitor(full_fit_history=False, full_sol_history=False)
    workflow = StdWorkflow(
        algorithm=algorithm, problem=problem, monitor=monitor,
        opt_direction="max", device=run_device,
    )
    # EvoX 1.3 wraps the algorithm; read counters and archives from that instance.
    algorithm = workflow.algorithm
    eager_workflow = workflow

    _synchronize(run_device)
    initialization_start = time.perf_counter()
    eager_workflow.init_step()
    _synchronize(run_device)
    initialization_elapsed = time.perf_counter() - initialization_start
    initialization = _history_entry(-1, algorithm)
    history = []
    search_start = time.perf_counter()
    target_search_evaluations = (
        config.generations * config.batch_size
        if config.search_evaluations is None
        else config.search_evaluations
    )
    generation = 0
    while algorithm.search_evaluations < target_search_evaluations:
        if (
            config.guidance_switch_evaluations is not None
            and algorithm.search_evaluations >= config.guidance_switch_evaluations
        ):
            algorithm.guidance_eta = 0.0 if config.guidance_eta_final is None else config.guidance_eta_final
        elif config.guidance_eta_final is not None:
            progress = algorithm.search_evaluations / max(target_search_evaluations, 1)
            algorithm.guidance_eta = config.guidance_eta + (
                config.guidance_eta_final - config.guidance_eta
            ) * progress
        algorithm.batch_size = min(
            config.batch_size,
            target_search_evaluations - algorithm.search_evaluations,
        )
        eager_workflow.step()
        if record_history:
            history.append(_history_entry(generation, algorithm))
        generation += 1
    _synchronize(run_device)
    search_elapsed = time.perf_counter() - search_start

    archive = algorithm.archive_snapshot()
    best_fitness = archive.best_fitness()
    return PaperAlignedEvoXResult(
        archive=archive,
        initialization=initialization,
        history=history,
        best_candidate=archive.best_candidate(),
        best_fitness=0.0 if best_fitness is None else best_fitness,
        qd_score=archive.qd_score(),
        initial_evaluations=algorithm.initial_evaluations,
        search_evaluations=algorithm.search_evaluations,
        initialization_elapsed_seconds=initialization_elapsed,
        search_elapsed_seconds=search_elapsed,
        guided_mutations=int(algorithm.guided_mutations.item()),
        guidance_preparation_seconds=guidance_preparation_seconds,
        device=run_device,
    )


def _history_entry(generation: int, algorithm: PaperAlignedMAPElites) -> dict[str, Any]:
    archive = algorithm.archive_snapshot()
    return {
        "generation": generation,
        "initial_evaluations": algorithm.initial_evaluations,
        "search_evaluations": algorithm.search_evaluations,
        "total_evaluations": algorithm.initial_evaluations + algorithm.search_evaluations,
        "coverage": archive.coverage(),
        "qd_score": archive.qd_score(),
        "best_fitness": archive.best_fitness(),
        "occupied_cells": archive.occupied_cells(),
    }


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_inputs(
    valid_products: list[Product],
    valid_tsets: list[TSet] | set[TSet] | tuple[TSet, ...],
    config: PaperAlignedEvoXConfig,
) -> None:
    if not valid_products:
        raise ValueError("valid_products must not be empty")
    if not valid_tsets:
        raise ValueError("valid_tsets must not be empty")
    if config.lower_bound <= 0 or config.upper_bound < config.lower_bound:
        raise ValueError("invalid suite-size bounds")
    if config.upper_bound > len(valid_products):
        raise ValueError("upper_bound must not exceed the static product-pool size")
    if config.generations < 0:
        raise ValueError("generations must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.search_evaluations is not None and config.search_evaluations < 0:
        raise ValueError("search_evaluations must be non-negative")
