from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spl.tensor_variation import (
    PADDING_INDEX,
    candidate_sizes,
    crossover_candidate_batch,
    has_duplicate_products,
    mutate_candidate_batch,
    random_candidate_batch,
    repair_candidate_batch,
)


def test_random_candidate_batch_respects_shape_bounds_padding_and_uniqueness():
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)

    candidates = random_candidate_batch(
        num_products=20,
        batch_size=8,
        lower_bound=2,
        upper_bound=5,
        device=device,
        generator=generator,
    )

    assert candidates.shape == (8, 5)
    assert torch.all((candidates == PADDING_INDEX) | ((0 <= candidates) & (candidates < 20)))
    assert torch.all((2 <= candidate_sizes(candidates)) & (candidate_sizes(candidates) <= 5))
    assert not torch.any(has_duplicate_products(candidates))


def test_mutate_candidate_batch_keeps_invariants():
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(1)
    parents = random_candidate_batch(12, 10, 2, 5, device=device, generator=generator)

    children = mutate_candidate_batch(parents, 12, 2, 5, generator=generator)

    assert children.shape == parents.shape
    assert torch.all((children == PADDING_INDEX) | ((0 <= children) & (children < 12)))
    assert torch.all((2 <= candidate_sizes(children)) & (candidate_sizes(children) <= 5))
    assert not torch.any(has_duplicate_products(children))


def test_mutation_can_add_remove_or_replace_without_breaking_bounds():
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(2)
    parents = torch.tensor(
        [
            [0, 1, PADDING_INDEX, PADDING_INDEX],
            [2, 3, 4, 5],
            [6, 7, 8, PADDING_INDEX],
        ],
        dtype=torch.long,
        device=device,
    )

    children = mutate_candidate_batch(parents, 10, 2, 4, generator=generator)

    assert torch.all((2 <= candidate_sizes(children)) & (candidate_sizes(children) <= 4))
    assert not torch.any(has_duplicate_products(children))


def test_crossover_candidate_batch_combines_two_parent_batches_and_keeps_invariants():
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(3)
    parent_a = torch.tensor([[0, 1, -1, -1], [2, 3, 4, -1]], dtype=torch.long, device=device)
    parent_b = torch.tensor([[2, 3, -1, -1], [4, 5, 6, -1]], dtype=torch.long, device=device)

    children = crossover_candidate_batch(parent_a, parent_b, 10, 2, 4, generator=generator)

    assert children.shape == (2, 4)
    assert torch.all((children == PADDING_INDEX) | ((0 <= children) & (children < 10)))
    assert torch.all((2 <= candidate_sizes(children)) & (candidate_sizes(children) <= 4))
    assert not torch.any(has_duplicate_products(children))


def test_repair_candidate_batch_removes_duplicates_and_fills_lower_bound():
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(4)
    candidates = torch.tensor(
        [
            [1, 1, 2, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    repaired = repair_candidate_batch(candidates, 8, 3, 4, generator=generator)

    assert torch.all((3 <= candidate_sizes(repaired)) & (candidate_sizes(repaired) <= 4))
    assert not torch.any(has_duplicate_products(repaired))


def test_tensor_variation_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(5)

    parents = random_candidate_batch(32, 16, 2, 6, device=device, generator=generator)
    children = mutate_candidate_batch(parents, 32, 2, 6, generator=generator)
    crossed = crossover_candidate_batch(parents, children, 32, 2, 6, generator=generator)

    assert parents.device.type == "cuda"
    assert children.device.type == "cuda"
    assert crossed.device.type == "cuda"
    assert not torch.any(has_duplicate_products(crossed))
