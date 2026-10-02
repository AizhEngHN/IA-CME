"""Input and dependency-interface checks without external benchmark data."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iacme.pipeline import IACMEConfig
from iacme.sampler import build_sampler_commands
from iacme.sat import build_targeted_generator_commands


def test_default_configuration_is_valid():
    IACMEConfig().validate()


@pytest.mark.parametrize("changes", [
    {"maximum_pool_size": 10},
    {"search_evaluations": -1},
    {"lower_bound": 4, "upper_bound": 3},
    {"guidance_eta": 1.5},
    {"targets_per_batch": 0},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        IACMEConfig(**changes).validate()


def test_missing_sat4j_dependency_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="sat4j-jar"):
        build_sampler_commands(ROOT, "model.cnf", "out.txt", 2, 0, 1, 10,
                               sat4j_jar=tmp_path / "missing.jar")


def test_sat4j_commands_accept_external_dependency_and_work_paths(tmp_path):
    jar = tmp_path / "core.jar"
    jar.touch()
    classes = tmp_path / "classes"
    compile_command, run_command = build_sampler_commands(
        ROOT, "model.cnf", "out.txt", 2, 0, 1, 10,
        sat4j_jar=jar, classes_dir=classes,
    )
    assert compile_command[2] == str(jar.resolve())
    assert compile_command[4] == str(classes.resolve())
    assert str(jar.resolve()) in run_command[2]
    _, targeted_run, target_classes = build_targeted_generator_commands(
        ROOT, "model.cnf", "targets.txt", "out.txt", 0, 1, 10, True,
        sat4j_jar=jar, classes_dir=classes,
    )
    assert target_classes == classes.resolve()
    assert targeted_run[-1] == "true"
    assert "SPLTestingMAP" not in str(run_command + targeted_run)
