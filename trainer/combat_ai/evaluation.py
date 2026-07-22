from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable

from .league import CRAZY_STYLES


EVALUATION_MODES = ("sword", "crystal", "combined", "terrain")
BASELINE_MATCHES_PER_MODE = 125
FINAL_MATCHES_PER_MODE = 25


def held_out_partition(seed: int) -> bool:
    """Stable 20% split shared by layouts, delays, kits, and arena seeds."""
    digest = hashlib.sha256(f"mcai-evaluation-split:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def build_manifest(kind: str, root_seed: int = 20260718) -> dict[str, Any]:
    if kind not in ("baseline", "final_human"):
        raise ValueError("kind must be 'baseline' or 'final_human'")
    count = BASELINE_MATCHES_PER_MODE if kind == "baseline" else FINAL_MATCHES_PER_MODE
    randomizer = random.Random(f"{kind}:{root_seed}")
    matches: list[dict[str, Any]] = []
    used: set[int] = set()
    for mode_index, mode in enumerate(EVALUATION_MODES):
        for index in range(count):
            while True:
                seed = randomizer.randrange(1, 2**31)
                if seed not in used and held_out_partition(seed):
                    used.add(seed)
                    break
            style = CRAZY_STYLES[(index + mode_index * 7) % len(CRAZY_STYLES)]
            matches.append({
                "match_id": f"{kind}-{mode}-{index + 1:03d}",
                "sequence": len(matches),
                "mode": mode,
                "opponent_style": style,
                "arena_seed": seed,
                "layout_seed": _derive(seed, "layout"),
                "kit_seed": _derive(seed, "kit"),
                "radius": 5 + _derive(seed, "radius") % 4,
                "action_delay_ticks": _derive(seed, "action-delay") % 4,
                "observation_delay_ticks": _derive(seed, "observation-delay") % 4,
                "held_out": True,
                "teachers_enabled": False,
                "mid_match_adaptation": False,
                "equalized_kit": True,
            })
    return {
        "format_version": 2,
        "kind": kind,
        "root_seed": root_seed,
        "split": {"algorithm": "sha256-mod-5", "held_out_bucket": 0, "fraction": 0.20},
        "frozen_policy_required": True,
        "matches_per_mode": count,
        "total_matches": len(matches),
        "matches": matches,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    kind = manifest.get("kind")
    expected = BASELINE_MATCHES_PER_MODE if kind == "baseline" else (
        FINAL_MATCHES_PER_MODE if kind == "final_human" else None
    )
    if manifest.get("format_version") != 2 or expected is None:
        raise ValueError("unsupported evaluation manifest")
    matches = manifest.get("matches")
    if not isinstance(matches, list) or len(matches) != expected * len(EVALUATION_MODES):
        raise ValueError("manifest has the wrong match count")
    ids: set[str] = set()
    counts = defaultdict(int)
    for sequence, match in enumerate(matches):
        if not isinstance(match, dict) or match.get("sequence") != sequence:
            raise ValueError("manifest sequence is not contiguous")
        match_id = str(match.get("match_id", ""))
        if not match_id or match_id in ids:
            raise ValueError("manifest match ids must be distinct")
        ids.add(match_id)
        mode = match.get("mode")
        if mode not in EVALUATION_MODES:
            raise ValueError("manifest contains an invalid mode")
        counts[mode] += 1
        if not held_out_partition(int(match.get("arena_seed", -1))):
            raise ValueError("manifest includes a training-partition seed")
        for flag, required in (
            ("held_out", True), ("teachers_enabled", False),
            ("mid_match_adaptation", False), ("equalized_kit", True),
        ):
            if match.get(flag) is not required:
                raise ValueError(f"manifest has invalid {flag}")
    if any(counts[mode] != expected for mode in EVALUATION_MODES):
        raise ValueError("manifest must balance all four modes")


def summarize_baseline(manifest: dict[str, Any], results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validate_manifest(manifest)
    joined = _join_results(manifest, results)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match, result in joined:
        by_mode[match["mode"]].append(result)
        by_style[match["opponent_style"]].append(result)
    return {
        "format_version": 1,
        "kind": "baseline_report",
        "complete": len(joined) == len(manifest["matches"]),
        "qualified": False,
        "qualification_reason": "baseline is descriptive; promotion gates run separately",
        "overall": _combat_metrics([result for _, result in joined]),
        "modes": {mode: _combat_metrics(by_mode[mode]) for mode in EVALUATION_MODES},
        "opponent_styles": {
            style: _combat_metrics(by_style[style]) for style in CRAZY_STYLES
        },
    }


def final_human_gate(manifest: dict[str, Any], results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validate_manifest(manifest)
    if manifest["kind"] != "final_human":
        raise ValueError("final gate requires a final_human manifest")
    joined = _join_results(manifest, results)
    ordered = [result for _, result in joined]
    wins = sum(str(result.get("outcome", "")).lower() == "win" for result in ordered)
    longest_losses = 0
    current_losses = 0
    for result in ordered:
        if str(result.get("outcome", "")).lower() == "loss":
            current_losses += 1
            longest_losses = max(longest_losses, current_losses)
        else:
            current_losses = 0
    complete = len(joined) == len(manifest["matches"])
    frozen = all(result.get("policy_frozen") is True for result in ordered)
    clean = all(
        result.get("teachers_enabled") is False
        and result.get("mid_match_adaptation") is False
        for result in ordered
    )
    return {
        "format_version": 1,
        "kind": "final_human_gate",
        "complete": complete,
        "wins": wins,
        "losses": sum(str(result.get("outcome", "")).lower() == "loss" for result in ordered),
        "draws": sum(str(result.get("outcome", "")).lower() == "draw" for result in ordered),
        "longest_losing_streak": longest_losses,
        "frozen_policy_verified": frozen,
        "clean_evaluation_verified": clean,
        "passed": complete and frozen and clean and wins >= 95 and longest_losses <= 2,
    }


def policy_promotion_gate(
    results: Iterable[dict[str, Any]],
    reference_win_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the frozen synthetic-league acceptance gates to audited matches."""
    entries = list(results)
    scripted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exploiters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    historical: list[dict[str, Any]] = []
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in entries:
        kind = str(result.get("opponent_kind", ""))
        if kind == "expert_script":
            scripted[str(result.get("opponent_id", ""))].append(result)
        elif kind == "exploiter":
            exploiters[str(result.get("opponent_id", ""))].append(result)
        elif kind == "historical":
            historical.append(result)
        categories[str(result.get("mode", ""))].append(result)
    script_rates = {key: _win_rate(values) for key, values in scripted.items()}
    exploiter_rates = {key: _win_rate(values) for key, values in exploiters.items()}
    category_rates = {key: _win_rate(values) for key, values in categories.items() if key}
    reference = reference_win_rates or {}
    regression_ok = all(
        category_rates.get(mode, 0.0) >= float(rate) - 0.03
        for mode, rate in reference.items()
    )
    total = len(entries)
    self_kill_rate = (
        sum(bool(entry.get("avoidable_self_kill")) for entry in entries) / total
        if total else 1.0
    )
    attempts = sum(int(_number(entry.get("legal_crystal_chains"))) for entry in entries)
    damaging = sum(int(_number(entry.get("damaging_crystal_chains"))) for entry in entries)
    retotem_attempts = sum(int(_number(entry.get("retotem_attempts"))) for entry in entries)
    fast_retotems = sum(int(_number(entry.get("retotem_within_two_ticks"))) for entry in entries)
    script_ok = (
        set(scripted) == set(CRAZY_STYLES)
        and all(len(values) >= 100 and script_rates[style] >= 0.90 for style, values in scripted.items())
    )
    exploiter_ok = bool(exploiters) and all(
        len(values) >= 100 and exploiter_rates[key] >= 0.75
        for key, values in exploiters.items()
    )
    historical_rate = _win_rate(historical)
    clean = bool(entries) and all(
        entry.get("policy_frozen") is True
        and entry.get("teachers_enabled") is False
        for entry in entries
    )
    checks = {
        "scripted_styles": script_ok,
        "retained_exploiters": exploiter_ok,
        "historical_mixture": len(historical) >= 100 and historical_rate >= 0.70,
        "avoidable_self_kills": self_kill_rate < 0.05,
        "crystal_conversion": attempts > 0 and damaging / attempts >= 0.80,
        "retotem": retotem_attempts > 0 and fast_retotems / retotem_attempts >= 0.95,
        "category_regression": regression_ok,
        "clean_frozen_evaluation": clean,
    }
    return {
        "format_version": 1,
        "kind": "policy_promotion_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "scripted_style_win_rates": script_rates,
        "exploiter_win_rates": exploiter_rates,
        "historical_win_rate": historical_rate,
        "category_win_rates": category_rates,
        "avoidable_self_kill_rate": self_kill_rate,
        "crystal_conversion": damaging / attempts if attempts else 0.0,
        "retotem_within_two_ticks": fast_retotems / retotem_attempts if retotem_attempts else 0.0,
    }


def _join_results(
    manifest: dict[str, Any], results: Iterable[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        match_id = str(result.get("match_id", ""))
        if not match_id or match_id in by_id:
            raise ValueError("results require distinct manifest match_id values")
        by_id[match_id] = result
    known = {match["match_id"] for match in manifest["matches"]}
    if not set(by_id).issubset(known):
        raise ValueError("results include a match outside the manifest")
    return [
        (match, by_id[match["match_id"]])
        for match in manifest["matches"] if match["match_id"] in by_id
    ]


def _combat_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    dealt = sum(_number(result.get("opponent_damage")) for result in results)
    taken = sum(_number(result.get("self_damage")) for result in results)
    first_hits = [
        _number(result["first_hit_seconds"]) for result in results
        if result.get("first_hit_seconds") is not None
    ]
    attempts = sum(int(_number(result.get("legal_crystal_chains"))) for result in results)
    damaging = sum(int(_number(result.get("damaging_crystal_chains"))) for result in results)
    retotems = sum(int(_number(result.get("retotem_attempts"))) for result in results)
    fast_retotems = sum(int(_number(result.get("retotem_within_two_ticks"))) for result in results)
    return {
        "matches": len(results),
        "wins": sum(str(result.get("outcome", "")).lower() == "win" for result in results),
        "deaths": sum(bool(result.get("died")) for result in results),
        "timeouts": sum(bool(result.get("timeout")) for result in results),
        "opponent_damage": dealt,
        "self_damage": taken,
        "damage_efficiency": dealt / max(1e-9, dealt + taken),
        "avoidable_self_kills": sum(bool(result.get("avoidable_self_kill")) for result in results),
        "median_first_hit_seconds": statistics.median(first_hits) if first_hits else None,
        "crystal_conversion": damaging / attempts if attempts else 0.0,
        "retotem_within_two_ticks": fast_retotems / retotems if retotems else 0.0,
        "blocks_placed": sum(int(_number(result.get("blocks_placed"))) for result in results),
        "blocks_mined": sum(int(_number(result.get("blocks_mined"))) for result in results),
    }


def _win_rate(results: list[dict[str, Any]]) -> float:
    return (
        sum(str(result.get("outcome", "")).lower() == "win" for result in results) / len(results)
        if results else 0.0
    )


def _derive(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0
