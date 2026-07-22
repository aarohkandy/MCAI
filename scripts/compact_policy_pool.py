from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


KEEP = (
    "format_version",
    "architecture_version",
    "feature_contract_version",
    "policy",
    "policy_version",
    "total_agent_ticks",
    "config",
    "metrics",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove recovery-only state from numbered league policies."
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=20_000_000)
    arguments = parser.parse_args()

    total_before = 0
    total_after = 0
    compacted = 0
    for path in sorted(arguments.checkpoint_dir.glob("policy-*.pt")):
        before = path.stat().st_size
        total_before += before
        if before <= arguments.minimum_bytes:
            total_after += before
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("policy"), dict):
            raise ValueError(f"{path} is not a policy checkpoint")
        compact = {key: payload[key] for key in KEEP if key in payload}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(compact, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if set(verified.get("policy", {})) != set(payload["policy"]):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"policy verification failed for {path}")
        os.replace(temporary, path)
        after = path.stat().st_size
        total_after += after
        compacted += 1
        print(f"{path.name}: {before:,} -> {after:,}")
    print(
        f"compacted={compacted} total={total_before:,}->{total_after:,} "
        f"saved={total_before - total_after:,}"
    )


if __name__ == "__main__":
    main()
