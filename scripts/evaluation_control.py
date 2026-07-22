from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from combat_ai.evaluation import (  # noqa: E402
    build_manifest, final_human_gate, policy_promotion_gate, summarize_baseline,
    validate_manifest,
)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and score reproducible held-out MCAI evaluation runs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--kind", choices=("baseline", "final_human"), required=True)
    generate.add_argument("--root-seed", type=int, default=20260718)
    generate.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    promotion = subparsers.add_parser("promotion-gate")
    promotion.add_argument("--results", type=Path, required=True)
    promotion.add_argument("--reference-win-rates", type=Path)
    promotion.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "generate":
        manifest = build_manifest(arguments.kind, arguments.root_seed)
        validate_manifest(manifest)
        write_json_atomic(arguments.output, manifest)
        print(json.dumps({"output": str(arguments.output), "matches": manifest["total_matches"]}))
        return

    if arguments.command == "promotion-gate":
        raw_results = json.loads(arguments.results.read_text(encoding="utf-8"))
        results = raw_results.get("results") if isinstance(raw_results, dict) else raw_results
        reference = (
            json.loads(arguments.reference_win_rates.read_text(encoding="utf-8"))
            if arguments.reference_win_rates else None
        )
        report = policy_promotion_gate(results, reference)
        write_json_atomic(arguments.output, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    raw_results = json.loads(arguments.results.read_text(encoding="utf-8"))
    results = raw_results.get("results") if isinstance(raw_results, dict) else raw_results
    if not isinstance(results, list):
        raise ValueError("results file must be a list or an object with a results list")
    report = (
        summarize_baseline(manifest, results)
        if manifest.get("kind") == "baseline"
        else final_human_gate(manifest, results)
    )
    write_json_atomic(arguments.output, report)
    print(json.dumps(report, indent=2))
    if manifest.get("kind") == "final_human":
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
