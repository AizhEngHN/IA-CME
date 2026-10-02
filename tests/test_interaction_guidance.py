from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spl.data_model import Product, TSet
from spl.interaction_guidance import (
    addition_gain_scores,
    archive_interaction_hardness,
    interaction_aware_mutate_candidate_batch,
    interaction_support_counts,
    matrix_coverage_percent,
    product_interaction_matrix,
    removal_loss_scores,
    suite_interaction_counts,
)
from spl.tensor_coverage import tensorize_spl_data
from spl.tensor_variation import candidate_sizes, has_duplicate_products


def test_product_interaction_matrix_matches_known_coverage():
    data = tensorize_spl_data(
        [Product([1, 2]), Product([1, -2]), Product([-1, 2])],
        [TSet([1, 2]), TSet([1, -2]), TSet([-1, 2]), TSet([-1, -2])],
    )

    matrix = product_interaction_matrix(data, product_chunk_size=2)

    assert matrix.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
    ]


def test_matrix_coverage_percent_matches_known_suite_coverage():
    matrix = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    candidates = torch.tensor([[0, -1], [0, 1], [1, 2]])

    fitness = matrix_coverage_percent(candidates, matrix, interaction_chunk_size=2)

    assert fitness.tolist() == [50.0, 75.0, 75.0]


def test_interaction_support_counts_matches_dense_sum_across_chunks():
    matrix = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ]
    )

    counts = interaction_support_counts(matrix, product_chunk_size=1)

    assert counts.tolist() == [2, 2, 2]


def test_hardness_ignores_unreachable_interactions_and_prioritizes_rare_misses():
    coverage = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [True, True, False, False],
        ]
    )
    archive = torch.tensor([[0, -1], [1, -1]])

    hardness = archive_interaction_hardness(archive, torch.tensor([True, True]), coverage)

    assert hardness[0] > 0
    assert hardness[1] == 0
    assert hardness[2] > hardness[0]
    assert hardness[3] == 0


def test_gain_and_loss_scores_follow_missing_and_unique_coverage():
    coverage = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
            [True, False, False, True],
        ]
    )
    candidate = torch.tensor([[0, 1, -1]])
    hardness = torch.tensor([1.0, 2.0, 3.0, 4.0])

    counts = suite_interaction_counts(candidate, coverage)
    losses = removal_loss_scores(candidate, coverage, hardness)
    gains = addition_gain_scores(candidate, coverage, hardness, torch.tensor([2, 3]), counts)

    assert counts.tolist() == [[1, 2, 1, 0]]
    assert losses[0, :2].tolist() == [1.0, 3.0]
    assert torch.isinf(losses[0, 2])
    assert gains.tolist() == [[4.0, 4.0]]


def test_guided_mutation_preserves_suite_invariants_and_counts_guided_rows():
    coverage = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
            [True, False, False, True],
            [True, True, True, False],
            [False, True, True, True],
        ]
    )
    parents = torch.tensor([[0, 1, -1, -1], [1, 2, 3, -1], [0, 1, 2, 3]])
    hardness = torch.ones(4)
    generator = torch.Generator().manual_seed(7)

    children, guided = interaction_aware_mutate_candidate_batch(
        parents,
        coverage,
        hardness,
        lower_bound=2,
        upper_bound=4,
        eta=1.0,
        candidate_sample_size=4,
        generator=generator,
    )

    assert guided.tolist() == [True, True, True]
    assert torch.all((2 <= candidate_sizes(children)) & (candidate_sizes(children) <= 4))
    assert not torch.any(has_duplicate_products(children))


def test_eta_zero_returns_only_random_rows():
    coverage = torch.eye(6, dtype=torch.bool)
    parents = torch.tensor([[0, 1, -1], [2, 3, 4]])

    _, guided = interaction_aware_mutate_candidate_batch(
        parents,
        coverage,
        torch.ones(6),
        2,
        3,
        eta=0.0,
        candidate_sample_size=3,
        generator=torch.Generator().manual_seed(3),
    )

    assert not torch.any(guided)
