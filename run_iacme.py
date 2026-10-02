"""Command-line entry point for Interaction-Aware Coupled MAP-Elites."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-stage IA-CME algorithm on externally supplied inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python run_iacme.py --model model.cnf --interactions valid_tsets.txt "
            "--sat4j-jar /path/to/sat4j-core.jar --output result.json --device cuda\n"
            "Interaction input: one semicolon-separated signed-literal tuple per line. "
            "Requires a JDK with java and javac. No benchmark data is included."
        ),
    )
    parser.add_argument("--model", required=True, type=Path, help="Feature model in DIMACS CNF format")
    parser.add_argument("--interactions", required=True, type=Path, help="Supplied valid t-wise interaction set")
    parser.add_argument("--sat4j-jar", required=True, type=Path, help="SAT4J core JAR dependency")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    parser.add_argument("--java", default="java", help="Java executable")
    parser.add_argument("--javac", default="javac", help="Java compiler executable")
    parser.add_argument("--device", default="auto", help="Search device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--pool-device", default="cpu", help="Pool-support computation device")
    parser.add_argument("--initial-pool-size", type=int, default=100)
    parser.add_argument("--maximum-pool-size", type=int, default=3000)
    parser.add_argument("--support-threshold", type=int, default=1)
    parser.add_argument("--targets-per-batch", type=int, default=32)
    parser.add_argument("--initial-assumption-count", type=int, default=16)
    parser.add_argument("--random-assumption-count", type=int, default=8)
    parser.add_argument("--max-attempts-per-product", type=int, default=10000)
    parser.add_argument("--max-attempts-per-target", type=int, default=1000)
    parser.add_argument("--lower-bound", type=int, default=2)
    parser.add_argument("--upper-bound", type=int, default=10)
    parser.add_argument("--search-evaluations", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--guidance-eta", type=float, default=0.5)
    parser.add_argument("--candidate-sample-size", type=int, default=64)
    parser.add_argument("--interaction-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="Default seed for all components")
    for component in ("initial", "target", "sat", "search"):
        parser.add_argument(f"--{component}-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from iacme.pipeline import IACMEConfig, run_iacme

    fields = {name: getattr(args, name) for name in IACMEConfig.__dataclass_fields__}
    for component in ("initial", "target", "sat", "search"):
        name = f"{component}_seed"
        fields[name] = args.seed if fields[name] is None else fields[name]
    payload = run_iacme(
        args.model, args.interactions, args.sat4j_jar, IACMEConfig(**fields),
        java=args.java, javac=args.javac,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"IA-CME: best={payload['best_fitness']:.6f}, QD-score={payload['qd_score']:.6f}")
    print(f"Saved result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
