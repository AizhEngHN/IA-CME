from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("evox")

from spl.data_model import Product, TSet, TestSuite as SPLTestSuite
from spl.evaluator import fitness_coverage
from spl.evox_paper_aligned import (
    EvoXSPLCoverageProblem,
    PaperAlignedEvoXConfig,
    PaperAlignedMAPElites,
    run_paper_aligned_evox_map_elites,
    run_paper_aligned_evox_map_elites_tensor_data,
)
from spl.interaction_guidance import product_interaction_matrix
from spl.tensor_coverage import tensorize_spl_data
from spl.tensor_variation import (
    PAPER_ADD,
    PAPER_REMOVE,
    PAPER_REPLACE,
    candidate_sizes,
    has_duplicate_products,
    paper_aligned_initial_candidate_batch,
    paper_aligned_mutation_operations,
    paper_aligned_mutate_candidate_batch,
)


def test_paper_initialization_creates_exactly_one_suite_per_size_cell():
    generator = torch.Generator().manual_seed(2)
    candidates = paper_aligned_initial_candidate_batch(8, 2, 5, generator=generator)

    assert candidates.shape == (4, 5)
    assert candidate_sizes(candidates).tolist() == [2, 3, 4, 5]
    assert not torch.any(has_duplicate_products(candidates))


def test_paper_mutation_operation_boundaries_match_java_probabilities():
    sizes = torch.tensor([2, 2, 3, 3, 3, 4, 4])
    random_values = torch.tensor([0.65, 0.90, 0.20, 0.50, 0.90, 0.65, 0.90])

    operations = paper_aligned_mutation_operations(sizes, 2, 4, random_values)

    assert operations.tolist() == [
        PAPER_REPLACE,
        PAPER_ADD,
        PAPER_REPLACE,
        PAPER_ADD,
        PAPER_REMOVE,
        PAPER_REPLACE,
        PAPER_REMOVE,
    ]


def test_paper_mutation_preserves_bounds_and_unique_products():
    generator = torch.Generator().manual_seed(4)
    parents = paper_aligned_initial_candidate_batch(10, 2, 5, generator=generator)

    children = paper_aligned_mutate_candidate_batch(parents, 10, 2, 5, generator=generator)

    assert torch.all((2 <= candidate_sizes(children)) & (candidate_sizes(children) <= 5))
    assert not torch.any(has_duplicate_products(children))


def test_evox_problem_matches_python_exact_coverage():
    products = _products()
    tsets = _tsets()
    problem = EvoXSPLCoverageProblem(tensorize_spl_data(products, tsets))
    candidate = torch.tensor([[0, 1, -1, -1]])

    tensor_fitness = float(problem.evaluate(candidate)[0].item())
    python_fitness = fitness_coverage(SPLTestSuite([products[0], products[1]]), tsets)

    assert tensor_fitness == python_fitness


def test_evox_workflow_populates_every_cell_and_tracks_budgets():
    result = run_paper_aligned_evox_map_elites(_products(), _tsets(), _config(generations=3, batch_size=5))

    assert result.archive.occupied_cells() == [0, 1, 2]
    assert result.initial_evaluations == 3
    assert result.search_evaluations == 15
    assert result.initialization["search_evaluations"] == 0
    assert result.history[-1]["total_evaluations"] == 18
    assert result.best_candidate is not None


def test_exact_search_budget_uses_a_final_partial_batch():
    config = PaperAlignedEvoXConfig(
        2,
        4,
        generations=0,
        batch_size=5,
        seed=0,
        device="cpu",
        search_evaluations=12,
    )

    result = run_paper_aligned_evox_map_elites(_products(), _tsets(), config)

    assert result.search_evaluations == 12
    assert [row["search_evaluations"] for row in result.history] == [5, 10, 12]


def test_linear_guidance_schedule_reduces_guided_mutations():
    fixed = PaperAlignedEvoXConfig(
        2,
        4,
        generations=0,
        batch_size=4,
        seed=2,
        device="cpu",
        guidance_eta=0.5,
        search_evaluations=40,
    )
    scheduled = PaperAlignedEvoXConfig(
        2,
        4,
        generations=0,
        batch_size=4,
        seed=2,
        device="cpu",
        guidance_eta=0.5,
        guidance_eta_final=0.0,
        search_evaluations=40,
    )

    fixed_result = run_paper_aligned_evox_map_elites(_products(), _tsets(), fixed)
    scheduled_result = run_paper_aligned_evox_map_elites(_products(), _tsets(), scheduled)

    assert 0 < scheduled_result.guided_mutations < fixed_result.guided_mutations


def test_two_phase_guidance_switches_off_at_requested_budget():
    config = PaperAlignedEvoXConfig(
        2,
        4,
        generations=0,
        batch_size=4,
        seed=2,
        device="cpu",
        guidance_eta=1.0,
        guidance_eta_final=0.0,
        guidance_switch_evaluations=8,
        search_evaluations=20,
    )

    result = run_paper_aligned_evox_map_elites(_products(), _tsets(), config)

    assert result.guided_mutations == 8


def test_formal_components_are_evox_problem_and_algorithm_subclasses():
    from evox.core import Algorithm, Problem

    problem = EvoXSPLCoverageProblem(tensorize_spl_data(_products(), _tsets()))
    algorithm = PaperAlignedMAPElites(4, 2, 4, 3, 0, "cpu")

    assert isinstance(problem, Problem)
    assert isinstance(algorithm, Algorithm)


def test_paper_aligned_evox_seed_is_reproducible():
    config = _config(generations=6, batch_size=5, seed=9)

    first = run_paper_aligned_evox_map_elites(_products(), _tsets(), config)
    second = run_paper_aligned_evox_map_elites(_products(), _tsets(), config)

    assert first.initialization == second.initialization
    assert first.history == second.history
    assert torch.equal(first.archive.elite_candidates, second.archive.elite_candidates)
    assert torch.equal(first.archive.elite_fitness, second.archive.elite_fitness)


def test_matrix_problem_matches_dense_problem_without_tset_presence():
    config = _config(generations=3, batch_size=4, seed=5)
    dense = run_paper_aligned_evox_map_elites(_products(), _tsets(), config)
    indexed_data = tensorize_spl_data(
        _products(), _tsets(), include_tset_presence=False
    )
    matrix = product_interaction_matrix(indexed_data)

    indexed = run_paper_aligned_evox_map_elites_tensor_data(
        indexed_data,
        len(_products()),
        config,
        coverage_matrix=matrix,
    )

    assert torch.equal(dense.archive.elite_candidates, indexed.archive.elite_candidates)
    assert torch.equal(dense.archive.elite_fitness, indexed.archive.elite_fitness)


def test_paper_aligned_evox_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = run_paper_aligned_evox_map_elites(
        _products(), _tsets(), _config(generations=2, batch_size=4, device="cuda")
    )

    assert result.device.type == "cuda"
    assert result.archive.elite_candidates.device.type == "cuda"


def _config(
    generations: int,
    batch_size: int,
    seed: int = 0,
    device: str = "cpu",
) -> PaperAlignedEvoXConfig:
    return PaperAlignedEvoXConfig(2, 4, generations, batch_size, seed, device)


def _products() -> list[Product]:
    return [
        Product([1, -2, 3]),
        Product([1, 2, -3]),
        Product([-1, 2, 3]),
        Product([1, 2, 3]),
    ]


def _tsets() -> list[TSet]:
    return [TSet([1, -2]), TSet([1, 2]), TSet([-1, 2]), TSet([2, 3])]
