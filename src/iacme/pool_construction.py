from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import torch

from iacme.sat import ConcurrentSATReplenisher, TargetSnapshot
from spl.data_model import Product, TSet
from spl.interaction_guidance import interaction_support_counts, product_interaction_matrix
from spl.tensor_coverage import TensorSPLData, products_to_presence_tensor, tensorize_spl_data


@dataclass(frozen=True)
class SupportDeficitPoolResult:
    products: tuple[Product, ...]
    checkpoints: dict[int, tuple[Product, ...]]
    initial_ceiling_percent: float
    final_ceiling_percent: float
    batch_records: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    sat_compilation_seconds: float
    support_preparation_seconds: float
    support_checkpoints: dict[int, tuple[Product, ...]]
    final_minimum_support: int
    stop_reason: str


def select_support_deficit_targets(
    support_counts: torch.Tensor,
    target_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Select least-supported interactions, using seeded noise only to break ties."""
    if support_counts.ndim != 1 or support_counts.numel() == 0:
        raise ValueError("support_counts must be a non-empty 1D tensor")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    count = min(target_count, support_counts.numel())
    noise = torch.rand(support_counts.shape, dtype=torch.float64, device=support_counts.device, generator=generator)
    priorities = support_counts.to(torch.float64) + noise * 0.25
    return torch.topk(priorities, k=count, largest=False).indices


def select_random_targets(
    support_counts: torch.Tensor,
    target_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if support_counts.ndim != 1 or support_counts.numel() == 0:
        raise ValueError("support_counts must be a non-empty 1D tensor")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    count = min(target_count, support_counts.numel())
    return torch.randperm(
        support_counts.numel(), device=support_counts.device, generator=generator
    )[:count]


def build_support_deficit_pool(
    initial_products: Sequence[Product],
    valid_tsets: Sequence[TSet],
    target_size: int,
    targets_per_batch: int,
    seed: int,
    device: str | torch.device,
    replenisher: ConcurrentSATReplenisher,
    checkpoint_sizes: Sequence[int] = (),
    support_thresholds: Sequence[int] = (),
    stop_when_support_thresholds_reached: bool = True,
    target_policy: str = "support_deficit",
) -> SupportDeficitPoolResult:
    preparation_started = time.perf_counter()
    products = list(initial_products)
    valid_tsets = tuple(valid_tsets)
    if not products or not valid_tsets:
        raise ValueError("initial_products and valid_tsets must not be empty")
    if target_size < len(products) or targets_per_batch <= 0:
        raise ValueError("invalid target_size or targets_per_batch")
    if target_policy not in {"support_deficit", "random"}:
        raise ValueError("target_policy must be 'support_deficit' or 'random'")
    checkpoints_requested = sorted({int(size) for size in checkpoint_sizes if len(products) <= size <= target_size})
    thresholds_requested = sorted({int(value) for value in support_thresholds})
    if any(value <= 0 for value in thresholds_requested):
        raise ValueError("support thresholds must be positive")
    run_device = torch.device(device)
    data = tensorize_spl_data(
        products, valid_tsets, device=run_device, include_tset_presence=False
    )
    coverage = product_interaction_matrix(data)
    initial_ceiling = _ceiling(coverage)
    initial_support = interaction_support_counts(coverage)
    generator = torch.Generator(device=run_device).manual_seed(seed)
    checkpoints: dict[int, tuple[Product, ...]] = {}
    support_checkpoints: dict[int, tuple[Product, ...]] = {
        threshold: tuple(products)
        for threshold in thresholds_requested
        if int(initial_support.min().item()) >= threshold
    }
    batch_records = []
    snapshot_id = 0

    compilation_started = time.perf_counter()
    replenisher.prepare()
    compilation_seconds = time.perf_counter() - compilation_started
    preparation_seconds = time.perf_counter() - preparation_started
    started = time.perf_counter()
    try:
        while len(products) < target_size and not (
            stop_when_support_thresholds_reached
            and thresholds_requested
            and len(support_checkpoints) == len(thresholds_requested)
        ):
            remaining = target_size - len(products)
            request_count = min(targets_per_batch, remaining)
            support_before = interaction_support_counts(coverage)
            if target_policy == "support_deficit":
                targets = select_support_deficit_targets(support_before, request_count, generator)
            else:
                targets = select_random_targets(support_before, request_count, generator)
            cpu_targets = tuple(int(index) for index in targets.cpu().tolist())
            snapshot = TargetSnapshot(
                snapshot_id=snapshot_id,
                generation=snapshot_id,
                interaction_indices=cpu_targets,
                interactions=tuple(
                    tuple(sorted(valid_tsets[index].values, key=lambda value: (abs(value), value)))
                    for index in cpu_targets
                ),
                created_at=time.perf_counter(),
            )
            if not replenisher.submit(snapshot, products):
                raise RuntimeError("support-deficit replenisher rejected a synchronous batch")
            batch = replenisher.finish()
            if batch is None or not batch.products:
                raise RuntimeError("support-deficit replenishment produced no accepted products")
            additions = list(batch.products[:remaining])
            products_before = len(products)
            presence = products_to_presence_tensor(
                additions,
                data.signed_value_to_column,
                device=run_device,
                ignore_unknown=True,
            )
            addition_data = TensorSPLData(
                product_presence=presence,
                tset_presence=data.tset_presence,
                tset_indices=data.tset_indices,
                tset_mask=data.tset_mask,
                signed_value_to_column=data.signed_value_to_column,
            )
            new_rows = product_interaction_matrix(addition_data)
            cumulative_support = support_before.unsqueeze(0) + new_rows.to(torch.int64).cumsum(dim=0)
            for threshold in thresholds_requested:
                if threshold in support_checkpoints:
                    continue
                reached_rows = (cumulative_support.min(dim=1).values >= threshold).nonzero(
                    as_tuple=False
                )
                if reached_rows.numel() > 0:
                    prefix_size = int(reached_rows[0].item()) + 1
                    support_checkpoints[threshold] = tuple(products + additions[:prefix_size])
            if (
                stop_when_support_thresholds_reached
                and thresholds_requested
                and len(support_checkpoints) == len(thresholds_requested)
            ):
                activated_size = max(
                    len(checkpoint) - products_before for checkpoint in support_checkpoints.values()
                )
                additions = additions[:activated_size]
                presence = presence[:activated_size]
                new_rows = new_rows[:activated_size]
            coverage = torch.cat([coverage, new_rows], dim=0)
            data = TensorSPLData(
                product_presence=torch.cat([data.product_presence, presence], dim=0),
                tset_presence=data.tset_presence,
                tset_indices=data.tset_indices,
                tset_mask=data.tset_mask,
                signed_value_to_column=data.signed_value_to_column,
            )
            products.extend(additions)
            support_after = interaction_support_counts(coverage)
            elapsed_after_seconds = time.perf_counter() - started
            batch_records.append(
                {
                    "snapshot_id": snapshot_id,
                    "products_before": products_before,
                    "products_after": len(products),
                    "requested": batch.requested,
                    "generated": batch.generated,
                    "valid": batch.valid,
                    "target_hits": batch.target_hits,
                    "accepted": len(batch.products),
                    "activated": len(additions),
                    "minimum_support_before": int(support_before.min().item()),
                    "minimum_support_after": int(support_after.min().item()),
                    "unsupported_before": int((support_before == 0).sum().item()),
                    "unsupported_after": int((support_after == 0).sum().item()),
                    "ceiling_after_percent": _ceiling(coverage),
                    "sat_seconds": batch.finished_at - batch.started_at,
                    "elapsed_after_seconds": elapsed_after_seconds,
                    "duplicate_with_snapshot_pool": batch.duplicate_with_snapshot_pool,
                    "java_report": batch.java_report,
                }
            )
            for checkpoint in checkpoints_requested:
                if checkpoint not in checkpoints and len(products) >= checkpoint:
                    checkpoints[checkpoint] = tuple(products[:checkpoint])
            snapshot_id += 1
    finally:
        replenisher.close()

    final_support = interaction_support_counts(coverage)
    all_thresholds_reached = len(support_checkpoints) == len(thresholds_requested)
    return SupportDeficitPoolResult(
        products=tuple(products),
        checkpoints=checkpoints,
        initial_ceiling_percent=initial_ceiling,
        final_ceiling_percent=_ceiling(coverage),
        batch_records=tuple(batch_records),
        elapsed_seconds=time.perf_counter() - started,
        sat_compilation_seconds=compilation_seconds,
        support_preparation_seconds=preparation_seconds,
        support_checkpoints=support_checkpoints,
        final_minimum_support=int(final_support.min().item()),
        stop_reason=(
            "support_threshold_reached"
            if thresholds_requested and all_thresholds_reached and stop_when_support_thresholds_reached
            else "target_size"
        ),
    )


def _ceiling(coverage: torch.Tensor) -> float:
    return float(coverage.any(dim=0).to(torch.float64).mean().item() * 100.0)
