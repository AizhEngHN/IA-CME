from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import torch

from spl.data_model import Product, TSet


PADDING_INDEX = -1


@dataclass(frozen=True)
class TensorSPLData:
    product_presence: torch.Tensor
    tset_presence: torch.Tensor | None
    tset_indices: torch.Tensor
    tset_mask: torch.Tensor
    signed_value_to_column: dict[int, int]

    @property
    def device(self) -> torch.device:
        return self.product_presence.device


def tensorize_spl_data(
    products: Iterable[Product],
    valid_tsets: Iterable[TSet],
    device: str | torch.device | None = None,
    include_tset_presence: bool = True,
) -> TensorSPLData:
    products = tuple(products)
    valid_tsets = tuple(valid_tsets)
    _require_products(products)
    _require_valid_tsets(valid_tsets)

    run_device = torch.device(device) if device is not None else torch.device("cpu")
    signed_value_to_column = build_signed_value_index(products, valid_tsets)
    tset_indices, tset_mask = tsets_to_index_tensor(valid_tsets, signed_value_to_column, run_device)
    return TensorSPLData(
        product_presence=products_to_presence_tensor(products, signed_value_to_column, run_device),
        tset_presence=(
            tsets_to_presence_tensor(valid_tsets, signed_value_to_column, run_device)
            if include_tset_presence
            else None
        ),
        tset_indices=tset_indices,
        tset_mask=tset_mask,
        signed_value_to_column=signed_value_to_column,
    )


def build_signed_value_index(
    products: Iterable[Product],
    valid_tsets: Iterable[TSet],
) -> dict[int, int]:
    values = set()
    for product in products:
        values.update(product.assignments)
    for tset in valid_tsets:
        values.update(tset.values)
    return {value: column for column, value in enumerate(sorted(values, key=lambda item: (abs(item), item)))}


def products_to_presence_tensor(
    products: Iterable[Product],
    signed_value_to_column: dict[int, int],
    device: str | torch.device | None = None,
    ignore_unknown: bool = False,
) -> torch.Tensor:
    products = tuple(products)
    _require_products(products)
    run_device = torch.device(device) if device is not None else torch.device("cpu")
    presence = torch.zeros((len(products), len(signed_value_to_column)), dtype=torch.bool)
    for row, product in enumerate(products):
        columns = []
        for value in set(product.assignments):
            column = signed_value_to_column.get(value)
            if column is None:
                if ignore_unknown:
                    continue
                raise KeyError(value)
            columns.append(column)
        if columns:
            presence[row, columns] = True
    return presence.to(run_device)


def tsets_to_presence_tensor(
    valid_tsets: Iterable[TSet],
    signed_value_to_column: dict[int, int],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    valid_tsets = tuple(valid_tsets)
    _require_valid_tsets(valid_tsets)
    run_device = torch.device(device) if device is not None else torch.device("cpu")
    presence = torch.zeros((len(valid_tsets), len(signed_value_to_column)), dtype=torch.bool, device=run_device)
    for row, tset in enumerate(valid_tsets):
        for value in tset.values:
            presence[row, signed_value_to_column[value]] = True
    return presence


def tsets_to_index_tensor(
    valid_tsets: Iterable[TSet],
    signed_value_to_column: dict[int, int],
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_tsets = tuple(valid_tsets)
    _require_valid_tsets(valid_tsets)
    run_device = torch.device(device) if device is not None else torch.device("cpu")
    width = max(tset.size() for tset in valid_tsets)
    indices = torch.zeros((len(valid_tsets), width), dtype=torch.long, device=run_device)
    mask = torch.zeros((len(valid_tsets), width), dtype=torch.bool, device=run_device)
    for row, tset in enumerate(valid_tsets):
        for column, value in enumerate(tset.values):
            indices[row, column] = signed_value_to_column[value]
            mask[row, column] = True
    return indices, mask


def suite_indices_to_tensor(
    suite_indices: Iterable[Iterable[int]],
    max_suite_size: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    suites = [tuple(int(index) for index in suite) for suite in suite_indices]
    width = max_suite_size if max_suite_size is not None else max((len(suite) for suite in suites), default=0)
    if width <= 0:
        raise ValueError("max_suite_size must be positive")

    run_device = torch.device(device) if device is not None else torch.device("cpu")
    candidates = torch.full((len(suites), width), PADDING_INDEX, dtype=torch.long, device=run_device)
    for row, suite in enumerate(suites):
        if len(suite) > width:
            raise ValueError("suite length exceeds max_suite_size")
        if any(index < 0 for index in suite):
            raise ValueError("product indices must be non-negative")
        if suite:
            candidates[row, : len(suite)] = torch.tensor(suite, dtype=torch.long, device=run_device)
    return candidates


def coverage_count_tensor(
    candidates: torch.Tensor,
    product_presence: torch.Tensor,
    tset_presence: torch.Tensor,
    tset_indices: torch.Tensor | None = None,
    tset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    candidates = _normalize_candidates(candidates, product_presence.device)
    _validate_presence_tensors(product_presence, tset_presence)
    _validate_candidate_indices(candidates, product_presence.shape[0])
    if tset_indices is None or tset_mask is None:
        tset_indices, tset_mask = _tset_indices_from_presence(tset_presence)
    tset_indices = tset_indices.to(device=product_presence.device, dtype=torch.long)
    tset_mask = tset_mask.to(device=product_presence.device, dtype=torch.bool)

    mask = candidates != PADDING_INDEX
    clamped = candidates.clamp_min(0)
    selected_products = product_presence[clamped] & mask.unsqueeze(-1)

    gather_indices = tset_indices.view(1, 1, tset_indices.shape[0], tset_indices.shape[1]).expand(
        selected_products.shape[0],
        selected_products.shape[1],
        -1,
        -1,
    )
    selected_for_tsets = selected_products.unsqueeze(2).expand(-1, -1, tset_indices.shape[0], -1)
    gathered_values = torch.gather(selected_for_tsets, dim=-1, index=gather_indices)
    product_covers_tset = (gathered_values | ~tset_mask.view(1, 1, tset_mask.shape[0], tset_mask.shape[1])).all(dim=-1)
    suite_covers_tset = (product_covers_tset & mask.unsqueeze(-1)).any(dim=1)
    return suite_covers_tset.sum(dim=1)


def coverage_percent_tensor(
    candidates: torch.Tensor,
    product_presence: torch.Tensor,
    tset_presence: torch.Tensor,
    tset_indices: torch.Tensor | None = None,
    tset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_presence_tensors(product_presence, tset_presence)
    return (
        coverage_count_tensor(candidates, product_presence, tset_presence, tset_indices, tset_mask).to(torch.float64)
        / tset_presence.shape[0]
        * 100.0
    )


def fitness_coverage_tensor(
    candidates: torch.Tensor,
    product_presence: torch.Tensor,
    tset_presence: torch.Tensor,
    tset_indices: torch.Tensor | None = None,
    tset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return coverage_percent_tensor(candidates, product_presence, tset_presence, tset_indices, tset_mask)


def evaluate_candidate_batch_tensor(candidates: torch.Tensor, data: TensorSPLData) -> tuple[torch.Tensor, torch.Tensor]:
    if data.tset_presence is None:
        raise ValueError("dense tset presence is required for direct candidate evaluation")
    qualities = fitness_coverage_tensor(
        candidates,
        data.product_presence,
        data.tset_presence,
        data.tset_indices,
        data.tset_mask,
    )
    behaviors = (candidates.to(data.device) != PADDING_INDEX).sum(dim=1).to(torch.float64)
    return qualities, behaviors


def _normalize_candidates(candidates: torch.Tensor, device: torch.device) -> torch.Tensor:
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 1D or 2D tensor")
    return candidates.to(device=device, dtype=torch.long)


def _validate_presence_tensors(product_presence: torch.Tensor, tset_presence: torch.Tensor) -> None:
    if product_presence.ndim != 2:
        raise ValueError("product_presence must be a 2D tensor")
    if tset_presence.ndim != 2:
        raise ValueError("tset_presence must be a 2D tensor")
    if product_presence.shape[1] != tset_presence.shape[1]:
        raise ValueError("product_presence and tset_presence must have the same signed-value width")
    if product_presence.shape[0] == 0:
        raise ValueError("product_presence must not be empty")
    if tset_presence.shape[0] == 0:
        raise ValueError("tset_presence must not be empty")


def _validate_candidate_indices(candidates: torch.Tensor, num_products: int) -> None:
    invalid = (candidates < PADDING_INDEX) | (candidates >= num_products)
    if bool(invalid.any().item()):
        raise ValueError("candidate product indices must be valid or padding index -1")


def _tset_indices_from_presence(tset_presence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = tset_presence.sum(dim=1)
    width = int(counts.max().item())
    indices = torch.zeros((tset_presence.shape[0], width), dtype=torch.long, device=tset_presence.device)
    mask = torch.zeros((tset_presence.shape[0], width), dtype=torch.bool, device=tset_presence.device)
    for row in range(tset_presence.shape[0]):
        row_indices = tset_presence[row].nonzero(as_tuple=False).flatten()
        indices[row, : row_indices.numel()] = row_indices
        mask[row, : row_indices.numel()] = True
    return indices, mask


def _require_products(products: tuple[Product, ...]) -> None:
    if not products:
        raise ValueError("products must not be empty")


def _require_valid_tsets(valid_tsets: tuple[TSet, ...]) -> None:
    if not valid_tsets:
        raise ValueError("valid_tsets must not be empty")
