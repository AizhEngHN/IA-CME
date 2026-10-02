from __future__ import annotations

import torch

from spl.tensor_coverage import TensorSPLData
from spl.tensor_variation import (
    PAPER_ADD,
    PAPER_REMOVE,
    candidate_sizes,
    paper_aligned_mutate_candidate_batch,
    paper_aligned_mutation_operations,
    repair_candidate_batch,
)


PADDING_INDEX = -1


def product_interaction_matrix(
    data: TensorSPLData,
    product_chunk_size: int = 64,
) -> torch.Tensor:
    """Build C[p, j], where product p covers interaction j."""
    if product_chunk_size <= 0:
        raise ValueError("product_chunk_size must be positive")
    num_products = data.product_presence.shape[0]
    num_tsets = data.tset_indices.shape[0]
    coverage = torch.empty((num_products, num_tsets), dtype=torch.bool, device=data.device)
    indices = data.tset_indices.unsqueeze(0)
    mask = data.tset_mask.unsqueeze(0)

    for start in range(0, num_products, product_chunk_size):
        products = data.product_presence[start : start + product_chunk_size]
        chunk_size = products.shape[0]
        gathered = torch.gather(
            products.unsqueeze(1).expand(-1, num_tsets, -1),
            dim=2,
            index=indices.expand(chunk_size, -1, -1),
        )
        coverage[start : start + chunk_size] = (gathered | ~mask).all(dim=2)
    return coverage


def matrix_coverage_percent(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
    interaction_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Evaluate suites from C[p, j] without materializing dense t-set features."""
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 1D or 2D tensor")
    if coverage_matrix.ndim != 2 or coverage_matrix.dtype != torch.bool:
        raise ValueError("coverage_matrix must be a two-dimensional boolean tensor")
    if interaction_chunk_size <= 0:
        raise ValueError("interaction_chunk_size must be positive")
    candidates = candidates.to(device=coverage_matrix.device, dtype=torch.long)
    invalid = (candidates < PADDING_INDEX) | (candidates >= coverage_matrix.shape[0])
    if bool(invalid.any().item()):
        raise ValueError("candidate product indices must be valid or padding index -1")
    mask = candidates != PADDING_INDEX
    indices = candidates.clamp_min(0)
    counts = torch.zeros(candidates.shape[0], dtype=torch.int64, device=coverage_matrix.device)
    for start in range(0, coverage_matrix.shape[1], interaction_chunk_size):
        stop = min(start + interaction_chunk_size, coverage_matrix.shape[1])
        selected = coverage_matrix[:, start:stop][indices]
        covered = (selected & mask.unsqueeze(-1)).any(dim=1)
        counts += covered.sum(dim=1, dtype=torch.int64)
    return counts.to(torch.float64) * (100.0 / coverage_matrix.shape[1])


def interaction_support_counts(
    coverage_matrix: torch.Tensor,
    product_chunk_size: int = 64,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Sum product support per interaction without casting the full matrix."""
    if coverage_matrix.ndim != 2 or coverage_matrix.dtype != torch.bool:
        raise ValueError("coverage_matrix must be a two-dimensional boolean tensor")
    if product_chunk_size <= 0:
        raise ValueError("product_chunk_size must be positive")
    counts = torch.zeros(
        coverage_matrix.shape[1], dtype=dtype, device=coverage_matrix.device
    )
    for start in range(0, coverage_matrix.shape[0], product_chunk_size):
        stop = min(start + product_chunk_size, coverage_matrix.shape[0])
        counts += coverage_matrix[start:stop].sum(dim=0, dtype=dtype)
    return counts


def archive_interaction_hardness(
    archive_candidates: torch.Tensor,
    occupied: torch.Tensor,
    coverage_matrix: torch.Tensor,
) -> torch.Tensor:
    """Weight archive-missing interactions inversely by static-pool availability."""
    occupied_candidates = archive_candidates[occupied]
    if occupied_candidates.shape[0] == 0:
        raise ValueError("archive must contain at least one occupied cell")
    selected, mask = _selected_product_coverage(occupied_candidates, coverage_matrix)
    archive_covered = (selected & mask.unsqueeze(-1)).any(dim=1)
    archive_miss = (~archive_covered).to(torch.float32).mean(dim=0)
    availability = interaction_support_counts(coverage_matrix, dtype=torch.float32)
    hardness = torch.where(
        availability > 0,
        archive_miss * coverage_matrix.shape[0] / availability.clamp_min(1.0),
        torch.zeros_like(archive_miss),
    )
    positive = hardness > 0
    if bool(positive.any().item()):
        hardness = hardness / hardness[positive].mean()
    return hardness


def sat_replenishment_weights(
    archive_candidates: torch.Tensor,
    occupied: torch.Tensor,
    coverage_matrix: torch.Tensor,
) -> torch.Tensor:
    """Prioritize archive misses, giving unreachable-in-pool interactions first claim."""
    occupied_candidates = archive_candidates[occupied]
    if occupied_candidates.shape[0] == 0:
        raise ValueError("archive must contain at least one occupied cell")
    selected, mask = _selected_product_coverage(occupied_candidates, coverage_matrix)
    archive_covered = (selected & mask.unsqueeze(-1)).any(dim=1)
    archive_miss = (~archive_covered).to(torch.float64).mean(dim=0)
    availability = interaction_support_counts(coverage_matrix, dtype=torch.float64)
    denominator = torch.where(availability > 0, availability, torch.full_like(availability, 0.5))
    weights = archive_miss / denominator
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def sample_target_interactions(
    weights: torch.Tensor,
    target_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty 1D tensor")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    positive_count = int((weights > 0).sum().item())
    if positive_count == 0:
        raise ValueError("weights must contain a positive value")
    return torch.multinomial(
        weights,
        num_samples=min(target_count, positive_count),
        replacement=False,
        generator=generator,
    )


def suite_interaction_counts(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
) -> torch.Tensor:
    selected, mask = _selected_product_coverage(candidates, coverage_matrix)
    return (selected & mask.unsqueeze(-1)).sum(dim=1, dtype=torch.int16)


def removal_loss_scores(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
    hardness: torch.Tensor,
) -> torch.Tensor:
    selected, mask = _selected_product_coverage(candidates, coverage_matrix)
    counts = (selected & mask.unsqueeze(-1)).sum(dim=1, dtype=torch.int16)
    unique_coverage = selected & mask.unsqueeze(-1) & (counts.unsqueeze(1) == 1)
    losses = torch.einsum("bst,t->bs", unique_coverage.to(torch.float32), hardness.to(torch.float32))
    return losses.masked_fill(~mask, torch.inf)


def addition_gain_scores(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
    hardness: torch.Tensor,
    product_indices: torch.Tensor,
    interaction_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if product_indices.ndim != 1 or product_indices.numel() == 0:
        raise ValueError("product_indices must be a non-empty 1D tensor")
    counts = (
        suite_interaction_counts(candidates, coverage_matrix)
        if interaction_counts is None
        else interaction_counts
    )
    weighted_missing = (counts == 0).to(torch.float32) * hardness.to(torch.float32).unsqueeze(0)
    sampled_coverage = coverage_matrix[product_indices].to(torch.float32)
    scores = weighted_missing @ sampled_coverage.T
    used = (candidates.unsqueeze(-1) == product_indices.view(1, 1, -1)).any(dim=1)
    return scores.masked_fill(used, -torch.inf)


def interaction_aware_mutate_candidate_batch(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
    hardness: torch.Tensor,
    lower_bound: int,
    upper_bound: int,
    eta: float,
    candidate_sample_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix paper-aligned random mutation with interaction-aware mutation."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if candidate_sample_size <= 0:
        raise ValueError("candidate_sample_size must be positive")
    num_products = coverage_matrix.shape[0]
    parents = repair_candidate_batch(candidates, num_products, lower_bound, upper_bound, generator)
    if eta == 0.0:
        return (
            paper_aligned_mutate_candidate_batch(
                parents,
                num_products,
                lower_bound,
                upper_bound,
                generator=generator,
            ),
            torch.zeros((parents.shape[0],), dtype=torch.bool, device=parents.device),
        )

    guided_mask = (
        torch.ones((parents.shape[0],), dtype=torch.bool, device=parents.device)
        if eta == 1.0
        else torch.rand((parents.shape[0],), device=parents.device, generator=generator) < eta
    )
    children = parents.clone()
    random_rows = (~guided_mask).nonzero(as_tuple=False).flatten()
    if random_rows.numel():
        children[random_rows] = paper_aligned_mutate_candidate_batch(
            parents[random_rows],
            num_products,
            lower_bound,
            upper_bound,
            generator=generator,
        )

    guided_rows = guided_mask.nonzero(as_tuple=False).flatten()
    if guided_rows.numel() == 0:
        return children, guided_mask
    guided_parents = parents[guided_rows]
    sizes = candidate_sizes(guided_parents)
    operation_values = torch.rand((guided_rows.numel(),), device=parents.device, generator=generator)
    operations = paper_aligned_mutation_operations(sizes, lower_bound, upper_bound, operation_values)
    losses = removal_loss_scores(guided_parents, coverage_matrix, hardness)
    remove_positions = losses.argmin(dim=1)
    counts = suite_interaction_counts(guided_parents, coverage_matrix)
    add_rows = (operations == PAPER_ADD).nonzero(as_tuple=False).flatten()
    remove_rows = (operations == PAPER_REMOVE).nonzero(as_tuple=False).flatten()
    replace_rows = ((operations != PAPER_ADD) & (operations != PAPER_REMOVE)).nonzero(as_tuple=False).flatten()

    sample_size = min(candidate_sample_size, num_products)
    sampled_products = torch.randperm(num_products, device=parents.device, generator=generator)[:sample_size]
    add_products = _guided_products_for_rows(
        add_rows,
        guided_parents,
        counts,
        coverage_matrix,
        hardness,
        sampled_products,
        num_products,
    )

    row_ids = torch.arange(guided_rows.numel(), device=parents.device)
    removed_coverage = coverage_matrix[guided_parents[row_ids, remove_positions]]
    replacement_counts = counts - removed_coverage.to(dtype=counts.dtype)
    replacement_products = _guided_products_for_rows(
        replace_rows,
        guided_parents,
        replacement_counts,
        coverage_matrix,
        hardness,
        sampled_products,
        num_products,
    )

    guided_children = guided_parents.clone()
    guided_children[add_rows, sizes[add_rows]] = add_products[add_rows]
    guided_children[replace_rows, remove_positions[replace_rows]] = replacement_products[replace_rows]

    if remove_rows.numel():
        width = guided_children.shape[1]
        columns = torch.arange(width, device=parents.device).unsqueeze(0).expand(remove_rows.numel(), -1)
        positions = remove_positions[remove_rows].unsqueeze(1)
        sources = torch.where(columns >= positions, columns + 1, columns).clamp_max(width - 1)
        shifted = guided_children[remove_rows].gather(1, sources)
        shifted[columns >= (sizes[remove_rows] - 1).unsqueeze(1)] = PADDING_INDEX
        guided_children[remove_rows] = shifted

    children[guided_rows] = guided_children

    return repair_candidate_batch(children, num_products, lower_bound, upper_bound, generator), guided_mask


def _selected_product_coverage(
    candidates: torch.Tensor,
    coverage_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2D tensor")
    mask = candidates != PADDING_INDEX
    selected = coverage_matrix[candidates.clamp_min(0)]
    return selected, mask


def _best_available_products(
    scores: torch.Tensor,
    sampled_products: torch.Tensor,
    candidates: torch.Tensor,
    num_products: int,
) -> torch.Tensor:
    best_positions = scores.argmax(dim=1)
    products = sampled_products[best_positions].clone()
    no_sampled_choice = ~torch.isfinite(scores.max(dim=1).values)
    for row in no_sampled_choice.nonzero(as_tuple=False).flatten().tolist():
        used = torch.zeros((num_products,), dtype=torch.bool, device=candidates.device)
        values = candidates[row][candidates[row] != PADDING_INDEX]
        used[values] = True
        available = (~used).nonzero(as_tuple=False).flatten()
        if available.numel() == 0:
            products[row] = candidates[row, 0]
        else:
            products[row] = available[0]
    return products


def _guided_products_for_rows(
    rows: torch.Tensor,
    candidates: torch.Tensor,
    interaction_counts: torch.Tensor,
    coverage_matrix: torch.Tensor,
    hardness: torch.Tensor,
    sampled_products: torch.Tensor,
    num_products: int,
) -> torch.Tensor:
    products = torch.zeros((candidates.shape[0],), dtype=torch.long, device=candidates.device)
    if rows.numel() == 0:
        return products
    scores = addition_gain_scores(
        candidates[rows],
        coverage_matrix,
        hardness,
        sampled_products,
        interaction_counts=interaction_counts[rows],
    )
    products[rows] = _best_available_products(
        scores,
        sampled_products,
        candidates[rows],
        num_products,
    )
    return products
