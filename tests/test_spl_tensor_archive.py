from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spl.tensor_archive import TensorMAPElitesArchive


def test_empty_tensor_archive_coverage_and_qd_score_are_zero():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)

    assert archive.coverage() == 0.0
    assert archive.qd_score() == 0.0
    assert archive.occupied_cells() == []
    assert archive.best_fitness() is None
    assert archive.best_candidate() is None


def test_tensor_archive_inserts_new_candidate_into_empty_cell():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    candidate = torch.tensor([0, 1, -1, -1])

    inserted = archive.add(candidate, 10.0)

    assert inserted is True
    assert archive.occupied_cells() == [0]
    assert archive.best_fitness() == 10.0


def test_tensor_archive_replaces_same_cell_only_when_fitness_is_higher():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    old_candidate = torch.tensor([0, 1, -1, -1])
    worse_candidate = torch.tensor([2, 3, -1, -1])
    better_candidate = torch.tensor([4, 5, -1, -1])

    assert archive.add(old_candidate, 10.0) is True
    assert archive.add(worse_candidate, 9.0) is False
    assert archive.add(better_candidate, 11.0) is True

    elite, fitness = archive.elite_for_cell(0)
    assert fitness == 11.0
    assert elite.tolist() == [4, 5, -1, -1]


def test_tensor_archive_add_batch_keeps_best_candidate_per_cell():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    candidates = torch.tensor(
        [
            [0, 1, -1, -1],
            [2, 3, -1, -1],
            [4, 5, 6, -1],
            [7, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    fitness = torch.tensor([1.0, 3.0, 2.0, 4.0])

    inserted = archive.add_batch(candidates, fitness)

    assert inserted.tolist() == [True, True, True, True]
    assert archive.occupied_cells() == [0, 1, 2]
    assert archive.qd_score() == 9.0
    assert archive.best_fitness() == 4.0
    cell0_elite, cell0_fitness = archive.elite_for_cell(0)
    assert cell0_fitness == 3.0
    assert cell0_elite.tolist() == [2, 3, -1, -1]


def test_tensor_archive_uses_explicit_descriptors_when_provided():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    candidates = torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=torch.long)
    fitness = torch.tensor([1.0, 2.0])
    descriptors = torch.tensor([2, 4])

    inserted = archive.add_batch(candidates, fitness, descriptors)

    assert inserted.tolist() == [True, True]
    assert archive.occupied_cells() == [0, 2]


def test_tensor_archive_ignores_out_of_range_descriptors():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    candidates = torch.tensor([[0, -1, -1, -1], [1, 2, 3, 4]], dtype=torch.long)
    fitness = torch.tensor([1.0, 2.0])
    descriptors = torch.tensor([1, 5])

    inserted = archive.add_batch(candidates, fitness, descriptors)

    assert inserted.tolist() == [False, False]
    assert archive.occupied_cells() == []


def test_tensor_archive_samples_only_occupied_elites():
    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4)
    archive.add(torch.tensor([0, 1, -1, -1]), 1.0)
    archive.add(torch.tensor([2, 3, 4, -1]), 2.0)
    generator = torch.Generator(device=archive.device).manual_seed(0)

    samples = archive.sample_elites(5, generator=generator)

    assert samples.shape == (5, 4)
    assert all(sample.tolist() in ([0, 1, -1, -1], [2, 3, 4, -1]) for sample in samples)


def test_tensor_archive_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    archive = TensorMAPElitesArchive(lower_bound=2, upper_bound=4, max_suite_size=4, device="cuda")
    candidates = torch.tensor([[0, 1, -1, -1], [2, 3, 4, -1]], dtype=torch.long, device="cuda")
    fitness = torch.tensor([1.0, 2.0], device="cuda")

    inserted = archive.add_batch(candidates, fitness)

    assert inserted.device.type == "cuda"
    assert archive.elite_candidates.device.type == "cuda"
    assert archive.occupied_cells() == [0, 1]
    assert archive.best_candidate().device.type == "cuda"
