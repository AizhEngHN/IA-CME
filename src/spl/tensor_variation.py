from __future__ import annotations

import torch


PADDING_INDEX = -1
PAPER_REPLACE = 0
PAPER_ADD = 1
PAPER_REMOVE = 2


def candidate_sizes(candidates: torch.Tensor) -> torch.Tensor:
    candidates = _normalize_candidates(candidates)
    return (candidates != PADDING_INDEX).sum(dim=1)


def random_candidate_batch(
    num_products: int,
    batch_size: int,
    lower_bound: int,
    upper_bound: int,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    _validate_config(num_products, batch_size, lower_bound, upper_bound)
    run_device = torch.device(device) if device is not None else torch.device("cpu")
    candidates = torch.full((batch_size, upper_bound), PADDING_INDEX, dtype=torch.long, device=run_device)

    for row in range(batch_size):
        size = _randint(lower_bound, upper_bound + 1, run_device, generator)
        candidates[row, :size] = _sample_product_indices(num_products, size, run_device, generator)
    return candidates


def paper_aligned_initial_candidate_batch(
    num_products: int,
    lower_bound: int,
    upper_bound: int,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create one random test suite for every MAP-Elites size cell."""
    num_cells = upper_bound - lower_bound + 1
    _validate_config(num_products, num_cells, lower_bound, upper_bound)
    run_device = torch.device(device) if device is not None else torch.device("cpu")
    candidates = torch.full((num_cells, upper_bound), PADDING_INDEX, dtype=torch.long, device=run_device)

    for row, size in enumerate(range(lower_bound, upper_bound + 1)):
        candidates[row, :size] = _sample_product_indices(num_products, size, run_device, generator)
    return candidates


def paper_aligned_mutation_operations(
    sizes: torch.Tensor,
    lower_bound: int,
    upper_bound: int,
    random_values: torch.Tensor,
) -> torch.Tensor:
    """Map random values to the mutation probabilities used by the Java method."""
    sizes = sizes.to(dtype=torch.long)
    random_values = random_values.to(device=sizes.device)
    if sizes.shape != random_values.shape:
        raise ValueError("sizes and random_values must have the same shape")
    if bool(((random_values < 0.0) | (random_values >= 1.0)).any().item()):
        raise ValueError("random_values must be in [0, 1)")

    operations = torch.full_like(sizes, PAPER_REPLACE)
    if lower_bound == upper_bound:
        return operations

    at_lower = sizes == lower_bound
    at_upper = sizes == upper_bound
    interior = ~(at_lower | at_upper)
    operations[at_lower & (random_values >= 2.0 / 3.0)] = PAPER_ADD
    operations[at_upper & (random_values >= 2.0 / 3.0)] = PAPER_REMOVE
    operations[interior & (random_values >= 1.0 / 3.0) & (random_values < 2.0 / 3.0)] = PAPER_ADD
    operations[interior & (random_values >= 2.0 / 3.0)] = PAPER_REMOVE
    return operations


def paper_aligned_mutate_candidate_batch(
    candidates: torch.Tensor,
    num_products: int,
    lower_bound: int,
    upper_bound: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one no-crossover mutation per suite using the paper's probabilities."""
    _validate_config(num_products, candidates.shape[0], lower_bound, upper_bound)
    children = repair_candidate_batch(candidates, num_products, lower_bound, upper_bound, generator)
    device = children.device
    sizes = candidate_sizes(children)
    random_values = torch.rand((children.shape[0],), device=device, generator=generator)
    operations = paper_aligned_mutation_operations(sizes, lower_bound, upper_bound, random_values)

    for row in range(children.shape[0]):
        size = int(sizes[row].item())
        operation = int(operations[row].item())
        if operation == PAPER_ADD:
            children[row, size] = _choose_product_not_in_candidate(children[row], num_products, generator)
        elif operation == PAPER_REMOVE:
            remove_index = _randint(0, size, device, generator)
            _remove_at(children[row], remove_index, size)
        else:
            replace_index = _randint(0, size, device, generator)
            current = int(children[row, replace_index].item())
            children[row, replace_index] = _choose_product_not_in_candidate(
                children[row],
                num_products,
                generator,
                allow_current=current,
            )

    return repair_candidate_batch(children, num_products, lower_bound, upper_bound, generator)


def mutate_candidate_batch(
    candidates: torch.Tensor,
    num_products: int,
    lower_bound: int,
    upper_bound: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    _validate_config(num_products, candidates.shape[0], lower_bound, upper_bound)
    candidates = repair_candidate_batch(candidates, num_products, lower_bound, upper_bound, generator)
    children = candidates.clone()
    device = children.device

    for row in range(children.shape[0]):
        size = int((children[row] != PADDING_INDEX).sum().item())
        operations = ["replace"]
        if size < upper_bound:
            operations.append("add")
        if size > lower_bound:
            operations.append("remove")

        operation = operations[_randint(0, len(operations), device, generator)]
        if operation == "add":
            children[row, size] = _choose_product_not_in_candidate(children[row], num_products, generator)
        elif operation == "remove":
            remove_index = _randint(0, size, device, generator)
            _remove_at(children[row], remove_index, size)
        else:
            replace_index = _randint(0, size, device, generator)
            current = int(children[row, replace_index].item())
            children[row, replace_index] = _choose_product_not_in_candidate(
                children[row],
                num_products,
                generator,
                allow_current=current,
            )

    return repair_candidate_batch(children, num_products, lower_bound, upper_bound, generator)


def crossover_candidate_batch(
    parent_a: torch.Tensor,
    parent_b: torch.Tensor,
    num_products: int,
    lower_bound: int,
    upper_bound: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    parent_a = repair_candidate_batch(parent_a, num_products, lower_bound, upper_bound, generator)
    parent_b = repair_candidate_batch(parent_b, num_products, lower_bound, upper_bound, generator)
    if parent_a.shape != parent_b.shape:
        raise ValueError("parent_a and parent_b must have the same shape")

    batch_size = parent_a.shape[0]
    device = parent_a.device
    children = torch.full((batch_size, upper_bound), PADDING_INDEX, dtype=torch.long, device=device)

    for row in range(batch_size):
        values = _unique_non_padding(torch.cat([parent_a[row], parent_b[row]]))
        if values.numel() < lower_bound:
            values = _append_missing_products(values, num_products, lower_bound - values.numel(), generator)

        max_size = min(upper_bound, int(values.numel()))
        target_size = _randint(lower_bound, max_size + 1, device, generator)
        order = torch.randperm(values.numel(), device=device, generator=generator)
        selected = values[order[:target_size]]
        children[row, :target_size] = selected

    return repair_candidate_batch(children, num_products, lower_bound, upper_bound, generator)


def repair_candidate_batch(
    candidates: torch.Tensor,
    num_products: int,
    lower_bound: int,
    upper_bound: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    _validate_config(num_products, max(int(candidates.shape[0]), 1), lower_bound, upper_bound)
    candidates = _normalize_candidates(candidates)
    if candidates.shape[1] > upper_bound:
        candidates = candidates[:, :upper_bound]

    repaired = torch.full((candidates.shape[0], upper_bound), PADDING_INDEX, dtype=torch.long, device=candidates.device)
    for row in range(candidates.shape[0]):
        values = _unique_non_padding(candidates[row])
        values = values[(0 <= values) & (values < num_products)]
        if values.numel() > upper_bound:
            order = torch.randperm(values.numel(), device=candidates.device, generator=generator)
            values = values[order[:upper_bound]]
        if values.numel() < lower_bound:
            values = _append_missing_products(values, num_products, lower_bound - values.numel(), generator)

        size = min(int(values.numel()), upper_bound)
        repaired[row, :size] = values[:size]
    return repaired


def has_duplicate_products(candidates: torch.Tensor) -> torch.Tensor:
    candidates = _normalize_candidates(candidates)
    duplicates = []
    for row in range(candidates.shape[0]):
        values = candidates[row][candidates[row] != PADDING_INDEX]
        duplicates.append(values.unique().numel() != values.numel())
    return torch.tensor(duplicates, dtype=torch.bool, device=candidates.device)


def _normalize_candidates(candidates: torch.Tensor) -> torch.Tensor:
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 1D or 2D tensor")
    return candidates.to(dtype=torch.long)


def _validate_config(num_products: int, batch_size: int, lower_bound: int, upper_bound: int) -> None:
    if num_products <= 0:
        raise ValueError("num_products must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lower_bound <= 0:
        raise ValueError("lower_bound must be positive")
    if upper_bound < lower_bound:
        raise ValueError("upper_bound must be greater than or equal to lower_bound")
    if lower_bound > num_products:
        raise ValueError("lower_bound must not exceed num_products")


def _sample_product_indices(
    num_products: int,
    size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if size <= num_products:
        return torch.randperm(num_products, device=device, generator=generator)[:size]
    return torch.randint(0, num_products, (size,), device=device, generator=generator)


def _choose_product_not_in_candidate(
    candidate: torch.Tensor,
    num_products: int,
    generator: torch.Generator | None,
    allow_current: int | None = None,
) -> int:
    device = candidate.device
    used_mask_source = candidate != PADDING_INDEX
    if allow_current is not None:
        used_mask_source = used_mask_source & (candidate != allow_current)
    used = candidate[used_mask_source]
    used_mask = torch.zeros(num_products, dtype=torch.bool, device=device)
    valid_used = used[(0 <= used) & (used < num_products)]
    if valid_used.numel():
        used_mask[valid_used] = True
    choices = (~used_mask).nonzero(as_tuple=False).flatten()
    if choices.numel() == 0:
        return _randint(0, num_products, device, generator)
    choice_index = _randint(0, int(choices.numel()), device, generator)
    return int(choices[choice_index].item())


def _append_missing_products(
    values: torch.Tensor,
    num_products: int,
    count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if count <= 0:
        return values
    device = values.device
    used_mask = torch.zeros(num_products, dtype=torch.bool, device=device)
    valid_values = values[(0 <= values) & (values < num_products)]
    if valid_values.numel():
        used_mask[valid_values] = True
    choices = (~used_mask).nonzero(as_tuple=False).flatten()
    if choices.numel() == 0:
        return values
    order = torch.randperm(choices.numel(), device=device, generator=generator)
    additions = choices[order[: min(count, int(choices.numel()))]]
    return torch.cat([values, additions])


def _unique_non_padding(candidate: torch.Tensor) -> torch.Tensor:
    values = candidate[candidate != PADDING_INDEX]
    if values.numel() == 0:
        return values
    return values.unique(sorted=False)


def _remove_at(candidate: torch.Tensor, index: int, size: int) -> None:
    if index < size - 1:
        candidate[index : size - 1] = candidate[index + 1 : size].clone()
    candidate[size - 1] = PADDING_INDEX


def _randint(
    low: int,
    high: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    return int(torch.randint(low, high, (1,), device=device, generator=generator).item())
