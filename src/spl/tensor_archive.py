from __future__ import annotations

from dataclasses import dataclass

import torch


PADDING_INDEX = -1


@dataclass
class TensorMAPElitesArchive:
    lower_bound: int
    upper_bound: int
    max_suite_size: int
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.lower_bound <= 0:
            raise ValueError("lower_bound must be positive")
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be greater than or equal to lower_bound")
        if self.max_suite_size < self.upper_bound:
            raise ValueError("max_suite_size must be at least upper_bound")

        self.device = torch.device(self.device)
        self.num_cells = self.upper_bound - self.lower_bound + 1
        self.elite_candidates = torch.full(
            (self.num_cells, self.max_suite_size),
            PADDING_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        self.elite_fitness = torch.full(
            (self.num_cells,),
            -torch.inf,
            dtype=torch.float64,
            device=self.device,
        )
        self.occupied = torch.zeros((self.num_cells,), dtype=torch.bool, device=self.device)

    def add_batch(
        self,
        candidates: torch.Tensor,
        fitness: torch.Tensor,
        descriptors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidates = self._normalize_candidates(candidates)
        fitness = fitness.to(device=self.device, dtype=torch.float64).flatten()
        if candidates.shape[0] != fitness.shape[0]:
            raise ValueError("candidates and fitness must have the same batch size")

        descriptors = self._candidate_sizes(candidates) if descriptors is None else descriptors.to(self.device).long()
        descriptors = descriptors.flatten()
        if descriptors.shape[0] != candidates.shape[0]:
            raise ValueError("descriptors and candidates must have the same batch size")

        inserted = torch.zeros((candidates.shape[0],), dtype=torch.bool, device=self.device)
        for row in range(candidates.shape[0]):
            descriptor = int(descriptors[row].item())
            if descriptor < self.lower_bound or descriptor > self.upper_bound:
                continue

            cell = descriptor - self.lower_bound
            if (not bool(self.occupied[cell].item())) or fitness[row] > self.elite_fitness[cell]:
                self.elite_candidates[cell] = candidates[row]
                self.elite_fitness[cell] = fitness[row]
                self.occupied[cell] = True
                inserted[row] = True

        return inserted

    def add(
        self,
        candidate: torch.Tensor,
        fitness: float | torch.Tensor,
        descriptor: int | torch.Tensor | None = None,
    ) -> bool:
        fitness_tensor = torch.as_tensor([fitness], dtype=torch.float64, device=self.device)
        descriptor_tensor = None if descriptor is None else torch.as_tensor([descriptor], dtype=torch.long, device=self.device)
        inserted = self.add_batch(candidate.reshape(1, -1), fitness_tensor, descriptor_tensor)
        return bool(inserted[0].item())

    def coverage(self) -> float:
        return int(self.occupied.sum().item()) / self.num_cells

    def qd_score(self) -> float:
        if not bool(self.occupied.any().item()):
            return 0.0
        return float(self.elite_fitness[self.occupied].sum().item())

    def occupied_cells(self) -> list[int]:
        return self.occupied.nonzero(as_tuple=False).flatten().detach().cpu().tolist()

    def best_fitness(self) -> float | None:
        if not bool(self.occupied.any().item()):
            return None
        return float(self.elite_fitness[self.occupied].max().item())

    def best_candidate(self) -> torch.Tensor | None:
        if not bool(self.occupied.any().item()):
            return None
        occupied_fitness = self.elite_fitness.clone()
        occupied_fitness[~self.occupied] = -torch.inf
        index = int(torch.argmax(occupied_fitness).item())
        return self.elite_candidates[index].clone()

    def elite_for_cell(self, cell: int) -> tuple[torch.Tensor, float] | None:
        if cell < 0 or cell >= self.num_cells:
            raise ValueError("cell index out of range")
        if not bool(self.occupied[cell].item()):
            return None
        return self.elite_candidates[cell].clone(), float(self.elite_fitness[cell].item())

    def sample_elites(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        occupied_indices = self.occupied.nonzero(as_tuple=False).flatten()
        if occupied_indices.numel() == 0:
            raise ValueError("cannot sample from an empty archive")

        choice_indices = torch.randint(
            low=0,
            high=int(occupied_indices.numel()),
            size=(batch_size,),
            device=self.device,
            generator=generator,
        )
        return self.elite_candidates[occupied_indices[choice_indices]].clone()

    def _normalize_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0)
        if candidates.ndim != 2:
            raise ValueError("candidates must be a 1D or 2D tensor")
        if candidates.shape[1] > self.max_suite_size:
            raise ValueError("candidate width exceeds max_suite_size")

        normalized = torch.full(
            (candidates.shape[0], self.max_suite_size),
            PADDING_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        normalized[:, : candidates.shape[1]] = candidates.to(device=self.device, dtype=torch.long)
        return normalized

    def _candidate_sizes(self, candidates: torch.Tensor) -> torch.Tensor:
        return (candidates != PADDING_INDEX).sum(dim=1).long()
