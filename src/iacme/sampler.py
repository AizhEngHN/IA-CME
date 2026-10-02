"""Portable commands for the randomized SAT4J product sampler."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_sat4j_jar(repo_root: Path, sat4j_jar: str | Path | None) -> Path:
    jar = Path(sat4j_jar or repo_root / "dependencies" / "sat4j.jar").resolve()
    if not jar.is_file():
        raise FileNotFoundError("Supply an existing SAT4J core JAR using --sat4j-jar.")
    return jar


def build_sampler_commands(
    repo_root: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    target_products: int,
    seed: int,
    assumption_count: int,
    max_attempts_per_product: int,
    javac_executable: str = "javac",
    java_executable: str = "java",
    sat4j_jar: str | Path | None = None,
    classes_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    repo_root = Path(repo_root).resolve()
    classes_dir = Path(classes_dir or repo_root / "tmp" / "sat4j_sampler").resolve()
    source = repo_root / "tools" / "java_runner" / "DeterministicSAT4JProductSampler.java"
    classpath = str(resolve_sat4j_jar(repo_root, sat4j_jar))
    compile_command = [javac_executable, "-cp", classpath, "-d", str(classes_dir), str(source)]
    run_command = [
        java_executable,
        "-cp",
        os.pathsep.join([str(classes_dir), classpath]),
        "DeterministicSAT4JProductSampler",
        str(Path(model_path).resolve()),
        str(Path(output_path).resolve()),
        str(target_products),
        str(seed),
        str(assumption_count),
        str(max_attempts_per_product),
    ]
    return compile_command, run_command


def parse_key_value_output(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip():
            values[key.strip()] = value.strip()
    return values
