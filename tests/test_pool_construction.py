from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iacme.pool_construction import (
    build_support_deficit_pool,
    select_random_targets,
    select_support_deficit_targets,
)
from iacme.sat import TargetSnapshot, TargetedProductBatch
from spl.data_model import Product, TSet


def test_support_deficit_selection_takes_lowest_support_levels_first():
    counts = torch.tensor([3, 0, 1, 0, 2, 1], dtype=torch.int64)
    selected = select_support_deficit_targets(counts, 4, torch.Generator().manual_seed(7))

    selected_counts = counts[selected]
    assert sorted(selected_counts.tolist()) == [0, 0, 1, 1]


def test_support_deficit_selection_is_seed_reproducible_for_ties():
    counts = torch.zeros(20, dtype=torch.int64)
    first = select_support_deficit_targets(counts, 5, torch.Generator().manual_seed(11))
    second = select_support_deficit_targets(counts, 5, torch.Generator().manual_seed(11))

    assert torch.equal(first, second)
    assert len(set(first.tolist())) == 5


def test_random_target_selection_is_seeded_and_unique():
    counts = torch.arange(20)
    first = select_random_targets(counts, 7, torch.Generator().manual_seed(19))
    second = select_random_targets(counts, 7, torch.Generator().manual_seed(19))

    assert torch.equal(first, second)
    assert len(set(first.tolist())) == 7


def test_invalid_target_policy_is_rejected():
    try:
        build_support_deficit_pool(
            initial_products=[Product([1])],
            valid_tsets=[TSet([1])],
            target_size=1,
            targets_per_batch=1,
            seed=0,
            device="cpu",
            replenisher=_FakeReplenisher([]),
            target_policy="unsupported",
        )
    except ValueError as error:
        assert "target_policy" in str(error)
    else:
        raise AssertionError("invalid target policy was accepted")


class _FakeReplenisher:
    def __init__(self, products):
        self.products = iter(products)
        self.snapshot = None

    def prepare(self):
        pass

    def submit(self, snapshot: TargetSnapshot, existing_products):
        self.snapshot = snapshot
        return True

    def finish(self):
        product = next(self.products)
        batch_products = product if isinstance(product, tuple) else (product,)
        return TargetedProductBatch(
            self.snapshot,
            batch_products,
            tuple(range(len(batch_products))),
            len(batch_products),
            len(batch_products),
            len(batch_products),
            len(batch_products),
            0,
            {},
            0.0,
            0.1,
        )

    def close(self):
        pass


def test_support_threshold_stops_before_maximum_size_and_records_checkpoint():
    result = build_support_deficit_pool(
        initial_products=[Product([1, 2])],
        valid_tsets=[TSet([1]), TSet([-1])],
        target_size=4,
        targets_per_batch=1,
        seed=3,
        device="cpu",
        replenisher=_FakeReplenisher([Product([-1, 2]), Product([1, -2])]),
        support_thresholds=(1,),
        stop_when_support_thresholds_reached=True,
    )

    assert len(result.products) == 2
    assert len(result.support_checkpoints[1]) == 2
    assert result.final_minimum_support == 1
    assert result.stop_reason == "support_threshold_reached"


def test_empty_support_thresholds_retain_fixed_target_size_behavior():
    result = build_support_deficit_pool(
        initial_products=[Product([1, 2])],
        valid_tsets=[TSet([1]), TSet([-1])],
        target_size=3,
        targets_per_batch=1,
        seed=3,
        device="cpu",
        replenisher=_FakeReplenisher([Product([-1, 2]), Product([1, -2])]),
    )

    assert len(result.products) == 3
    assert result.stop_reason == "target_size"


def test_support_checkpoint_uses_exact_prefix_inside_generated_batch():
    result = build_support_deficit_pool(
        initial_products=[Product([1, 2, 3])],
        valid_tsets=[TSet([1]), TSet([-1])],
        target_size=5,
        targets_per_batch=3,
        seed=3,
        device="cpu",
        replenisher=_FakeReplenisher(
            [
                (
                    Product([-1, 2, 3]),
                    Product([1, -2, 3]),
                    Product([1, 2, -3]),
                )
            ]
        ),
        support_thresholds=(1,),
        stop_when_support_thresholds_reached=True,
    )

    assert len(result.products) == 2
    assert len(result.support_checkpoints[1]) == 2
    assert result.batch_records[0]["accepted"] == 3
    assert result.batch_records[0]["activated"] == 1
