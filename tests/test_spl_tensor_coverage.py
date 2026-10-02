from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spl.data_model import Product, TSet, TestSuite as SplTestSuite
from spl.evaluator import coverage_count, fitness_coverage
from spl.tensor_coverage import (
    coverage_count_tensor,
    evaluate_candidate_batch_tensor,
    fitness_coverage_tensor,
    products_to_presence_tensor,
    suite_indices_to_tensor,
    tensorize_spl_data,
)


def test_tensorized_coverage_count_matches_python_evaluator():
    products = _valid_products()
    valid_tsets = _valid_tsets()
    data = tensorize_spl_data(products, valid_tsets)
    candidates = suite_indices_to_tensor([[0, 1]], max_suite_size=3)

    tensor_counts = coverage_count_tensor(candidates, data.product_presence, data.tset_presence)
    python_count = coverage_count(SplTestSuite([products[0], products[1]]), valid_tsets)

    assert tensor_counts.tolist() == [python_count]


def test_tensorized_fitness_matches_python_evaluator_for_batch():
    products = _valid_products()
    valid_tsets = _valid_tsets()
    data = tensorize_spl_data(products, valid_tsets)
    candidates = suite_indices_to_tensor([[0], [1, 2], [3]], max_suite_size=3)

    tensor_fitness = fitness_coverage_tensor(candidates, data.product_presence, data.tset_presence)
    python_fitness = [
        fitness_coverage(SplTestSuite([products[0]]), valid_tsets),
        fitness_coverage(SplTestSuite([products[1], products[2]]), valid_tsets),
        fitness_coverage(SplTestSuite([products[3]]), valid_tsets),
    ]

    assert tensor_fitness.tolist() == python_fitness


def test_tensorized_suite_coverage_uses_any_product_not_union_of_products():
    products = [Product([1]), Product([-2])]
    valid_tsets = [TSet([1, -2])]
    data = tensorize_spl_data(products, valid_tsets)
    candidates = suite_indices_to_tensor([[0, 1]], max_suite_size=2)

    tensor_fitness = fitness_coverage_tensor(candidates, data.product_presence, data.tset_presence)

    assert tensor_fitness.tolist() == [0.0]


def test_evaluate_candidate_batch_tensor_returns_quality_and_behavior():
    products = _valid_products()
    valid_tsets = _valid_tsets()
    data = tensorize_spl_data(products, valid_tsets)
    candidates = suite_indices_to_tensor([[0, 1], [2]], max_suite_size=3)

    qualities, behaviors = evaluate_candidate_batch_tensor(candidates, data)

    assert qualities.shape == (2,)
    assert behaviors.tolist() == [2.0, 1.0]


def test_tensorized_coverage_rejects_empty_valid_tsets():
    with pytest.raises(ValueError):
        tensorize_spl_data(_valid_products(), [])


def test_indexed_tensorization_omits_dense_tset_presence():
    data = tensorize_spl_data(
        _valid_products(), _valid_tsets(), include_tset_presence=False
    )

    assert data.tset_presence is None
    assert data.tset_indices.shape == (len(_valid_tsets()), 2)


def test_incremental_presence_can_project_out_values_not_used_by_interactions():
    mapping = {-1: 0, 1: 1}

    presence = products_to_presence_tensor(
        [Product([1, -2, 3])],
        mapping,
        ignore_unknown=True,
    )

    assert presence.tolist() == [[False, True]]
    with pytest.raises(KeyError):
        products_to_presence_tensor([Product([1, -2, 3])], mapping)


def test_presence_tensor_keeps_requested_device_after_batched_construction():
    mapping = {-1: 0, 1: 1, -2: 2, 2: 3, -3: 4, 3: 5}

    presence = products_to_presence_tensor(_valid_products(), mapping)

    assert presence.device.type == "cpu"
    assert presence.dtype == torch.bool


def test_tensorized_coverage_matches_python_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    products = _valid_products()
    valid_tsets = _valid_tsets()
    data = tensorize_spl_data(products, valid_tsets, device="cuda")
    candidates = suite_indices_to_tensor([[0, 1], [2, 3]], max_suite_size=3, device="cuda")

    tensor_fitness = fitness_coverage_tensor(candidates, data.product_presence, data.tset_presence)
    python_fitness = [
        fitness_coverage(SplTestSuite([products[0], products[1]]), valid_tsets),
        fitness_coverage(SplTestSuite([products[2], products[3]]), valid_tsets),
    ]

    assert tensor_fitness.device.type == "cuda"
    assert tensor_fitness.detach().cpu().tolist() == python_fitness


def _valid_products() -> list[Product]:
    return [
        Product([1, -2, 3]),
        Product([1, 2, -3]),
        Product([-1, 2, 3]),
        Product([1, 2, 3]),
    ]


def _valid_tsets() -> list[TSet]:
    return [
        TSet([1, -2]),
        TSet([1, 2]),
        TSet([-1, 2]),
        TSet([2, 3]),
    ]
