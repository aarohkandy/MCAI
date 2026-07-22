from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))

from combat_ai.league import LeagueManager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and promote an exploiter after exactly 100 held-out matches"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--results", type=Path, required=True,
        help="JSON list of booleans/outcomes, or an object containing results",
    )
    parser.add_argument("--promote", action="store_true")
    arguments = parser.parse_args()
    payload = json.loads(arguments.results.read_text(encoding="utf-8"))
    entries = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("results must be a JSON list")
    results = [_candidate_won(entry) for entry in entries]
    manager = LeagueManager(arguments.checkpoint_dir, torch.device("cpu"))
    report = manager.import_exploiter_results(arguments.candidate, results)
    output = {"candidate": str(arguments.candidate), **report}
    if arguments.promote:
        output["promoted_checkpoint"] = str(manager.add_frozen_checkpoint(arguments.candidate))
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if report["promotable"] else 1)


def _candidate_won(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("win", "candidate_win"):
            return True
        if normalized in ("loss", "draw", "main_win"):
            return False
    if isinstance(value, dict):
        return _candidate_won(value.get("outcome"))
    raise ValueError(f"invalid promotion result: {value!r}")


if __name__ == "__main__":
    main()
