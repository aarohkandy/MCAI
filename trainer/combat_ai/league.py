from __future__ import annotations

import hashlib
import json
import math
import random
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .distribution import sample_actions
from .export import load_policy
from .features import batch_observations
from .model import CombatPolicy


SCRIPTED_STYLE_FAMILIES = {
    "sword": (
        "rush", "spacing", "sprint_reset", "strafe", "jump_critical",
        "retreat", "defensive", "bait_and_counter", "counterattack",
    ),
    "crystal": (
        "crystal_rush", "safe_crystal", "orbit_crystal", "crystal_bait",
        "crystal_kamikaze", "crystal_defense", "crystal_escape",
    ),
    "sustain": ("totem_pressure", "heal_pressure"),
    "terrain": (
        "obsidian_builder", "tactical_miner", "cover_builder",
        "high_ground", "terrain_trap",
    ),
    "chaos": ("erratic",),
}
CRAZY_STYLES = tuple(
    style for family in SCRIPTED_STYLE_FAMILIES.values() for style in family
)
CRAZY_RESULT_WINDOW = 16
CRAZY_MIN_SAMPLES_PER_STYLE = 4
CRAZY_MIN_COVERED_STYLES = 10
CRAZY_MIN_PASSING_STYLES = 8
CRAZY_STYLE_PASS_SCORE = 0.55
CRAZY_OVERALL_PASS_SCORE = 0.60
MATCHMAKING_SHARES = {
    "historical": 0.35,
    "exploiter": 0.30,
    "expert_script": 0.20,
    "mirror": 0.15,
}
RECOVERY_MATCHMAKING_SHARES = {
    "historical": 0.08,
    "exploiter": 0.02,
    "expert_script": 0.30,
    "mirror": 0.60,
}
BOOTSTRAP_MATCHMAKING_SHARES = {
    "historical": 0.15,
    "exploiter": 0.05,
    "expert_script": 0.35,
    "mirror": 0.45,
}
DEVELOPING_MATCHMAKING_SHARES = {
    "historical": 0.25,
    "exploiter": 0.15,
    "expert_script": 0.30,
    "mirror": 0.30,
}
RECOVERY_SCORE_CUTOFF = 0.15
BOOTSTRAP_SCORE_CUTOFF = 0.25
DEVELOPING_SCORE_CUTOFF = 0.50
PAYOFF_WINDOW = 100
EXPLOITER_PROMOTION_MATCHES = 100
EXPLOITER_PROMOTION_WIN_RATE = 0.60


@dataclass
class EpisodeAssignment:
    episode_id: str
    mode: str
    opponent_agent: str | None = None
    checkpoint: str | None = None
    style: str | None = None


class LeagueManager:
    """PFSP league with historical, exploiter, expert, and mirror populations."""

    def __init__(self, checkpoint_dir: Path, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.metadata_path = checkpoint_dir / "league.json"
        self.assignments: dict[str, EpisodeAssignment] = {}
        self.policy_cache: dict[str, CombatPolicy] = {}
        self.hidden: dict[tuple[str, str], torch.Tensor] = {}
        self.main_elo = 1000.0
        self.ratings: dict[str, float] = {}
        self.payoff_matrix: dict[str, dict[str, list[float]]] = {"main": {}}
        # Compatibility alias for the main-policy row used by older tooling.
        self.payoffs: dict[str, list[float]] = self.payoff_matrix["main"]
        self.exploiter_trials: dict[str, list[float]] = {}
        self.evaluations: list[float] = []
        self.crazy_results: dict[str, list[float]] = {
            style: [] for style in CRAZY_STYLES
        }
        self.matchmaking_audit = _empty_matchmaking_audit()
        self.forced_opponent: Path | None = None
        self._load()

    def assign_batch(self, steps: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[str]] = {}
        for step in steps:
            episode = str(step["observation"]["match"]["episode_id"])
            grouped.setdefault(episode, []).append(str(step["agent_id"]))
        shared_checkpoints: dict[str, str] = {}
        for episode, agents in grouped.items():
            if episode in self.assignments or episode == "waiting" or len(agents) < 2:
                continue
            assignment = self._new_assignment(episode, sorted(set(agents)))
            if assignment.mode in ("historical", "exploiter") and assignment.checkpoint:
                # Preserve PFSP sampling across batches while sharing a frozen
                # actor inside each simultaneous four-lane batch. This lets all
                # opponent observations use one batched forward pass per family.
                assignment.checkpoint = shared_checkpoints.setdefault(
                    assignment.mode, assignment.checkpoint,
                )
            self.assignments[episode] = assignment
            self._record_assignment(assignment)

    def assignment_for(self, episode_id: str) -> EpisodeAssignment | None:
        return self.assignments.get(episode_id)

    def set_device(self, device: torch.device) -> None:
        if device == self.device:
            return
        self.device = device
        self.policy_cache.clear()
        self.hidden.clear()

    def force_frozen_opponent(self, checkpoint: Path | None) -> None:
        self.forced_opponent = checkpoint

    def add_frozen_checkpoint(self, source: Path) -> Path:
        report = self.exploiter_promotion_report(source)
        if not report["promotable"]:
            raise ValueError(
                "exploiter promotion requires at least 100 held-out matches "
                "and a 60% candidate win rate"
            )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if "policy" not in payload:
            raise ValueError("exploiter checkpoint does not contain a policy")
        ticks = int(payload.get("total_agent_ticks", 0))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
        destination = self.checkpoint_dir / f"policy-exploiter-{ticks:012d}-{digest}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        self.ratings[destination.name] = self.main_elo
        self._save(exploiter_requested=False)
        return destination

    def note_exploiter_result(self, source: Path, candidate_won: bool) -> dict[str, Any]:
        """Record one held-out candidate-vs-frozen-main promotion match."""
        key = self._candidate_key(source)
        results = self.exploiter_trials.setdefault(key, [])
        results.append(1.0 if candidate_won else 0.0)
        self.exploiter_trials[key] = results[-EXPLOITER_PROMOTION_MATCHES:]
        self._save()
        return self.exploiter_promotion_report(source)

    def import_exploiter_results(self, source: Path, results: list[bool]) -> dict[str, Any]:
        """Atomically replace a candidate's promotion trial with an audited run."""
        if len(results) != EXPLOITER_PROMOTION_MATCHES:
            raise ValueError("promotion result import must contain exactly 100 matches")
        self.exploiter_trials[self._candidate_key(source)] = [
            1.0 if result else 0.0 for result in results
        ]
        self._save()
        return self.exploiter_promotion_report(source)

    def exploiter_promotion_report(self, source: Path) -> dict[str, Any]:
        results = self.exploiter_trials.get(self._candidate_key(source), [])
        wins = int(sum(results))
        matches = len(results)
        rate = wins / matches if matches else 0.0
        return {
            "matches": matches,
            "wins": wins,
            "win_rate": rate,
            "required_matches": EXPLOITER_PROMOTION_MATCHES,
            "required_win_rate": EXPLOITER_PROMOTION_WIN_RATE,
            "promotable": matches >= EXPLOITER_PROMOTION_MATCHES
            and rate >= EXPLOITER_PROMOTION_WIN_RATE,
        }

    def opponent_action(self, assignment: EpisodeAssignment, agent_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        return self.opponent_actions([(0, assignment, agent_id, observation)])[0]

    def opponent_actions(
        self,
        requests: list[tuple[int, EpisodeAssignment, str, dict[str, Any]]],
    ) -> dict[int, dict[str, Any]]:
        """Batch league inference by frozen checkpoint for laptop throughput."""
        results: dict[int, dict[str, Any]] = {}
        grouped: dict[str, list[tuple[int, EpisodeAssignment, str, dict[str, Any]]]] = {}
        for request in requests:
            index, assignment, _agent_id, observation = request
            if assignment.mode in ("historical", "exploiter") and assignment.checkpoint:
                grouped.setdefault(assignment.checkpoint, []).append(request)
            else:
                results[index] = scripted_action(observation, assignment.style or "erratic")
        for checkpoint, group in grouped.items():
            path = Path(checkpoint)
            policy = self.policy_cache.get(str(path))
            if policy is None:
                policy = load_policy(path, self.device)
                self.policy_cache[str(path)] = policy
            observations = [request[3] for request in group]
            features = batch_observations(observations, self.device)
            hidden = torch.cat([
                self.hidden.get(
                    (assignment.episode_id, agent_id),
                    policy.initial_hidden(1, self.device),
                )
                for _index, assignment, agent_id, _observation in group
            ], dim=1)
            with torch.inference_mode():
                output = policy(features, hidden)
                actions, _, _, _ = sample_actions(
                    output, features, deterministic=False, compute_entropy=False,
                )
            for offset, (index, assignment, agent_id, _observation) in enumerate(group):
                self.hidden[(assignment.episode_id, agent_id)] = output.hidden[:, offset:offset + 1].detach()
                results[index] = actions[offset]
        return results

    def record_result(
        self,
        episode_id: str,
        main_agent: str,
        reward: float,
        outcome: str | None = None,
        terminal_info: dict[str, Any] | None = None,
    ) -> None:
        assignment = self.assignments.get(episode_id)
        if assignment is None or assignment.opponent_agent == main_agent:
            return
        score = ({"win": 1.0, "draw": 0.5, "loss": 0.0}.get(str(outcome).lower())
                 if outcome is not None else None)
        if score is None:
            score = 1.0 if reward > 0 else (0.5 if abs(reward + 0.05) < 1e-6 else 0.0)
        info = terminal_info if isinstance(terminal_info, dict) else {}
        reason = str(info.get("reason", "unknown")).lower()
        policy_owned_terminal = info.get("policy_owned_kill") is True
        main_policy_kill = policy_owned_terminal and (
            assignment.mode == "mirror" or score == 1.0
        )
        self._record_matchmaking_result(
            assignment, score, reason, main_policy_kill,
        )
        if assignment.mode in ("historical", "exploiter") and assignment.checkpoint:
            key = Path(assignment.checkpoint).name
            opponent_elo = self.ratings.get(key, 1000.0)
            expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.main_elo) / 400.0))
            change = 24.0 * (score - expected)
            self.main_elo += change
            self.ratings[key] = opponent_elo - change
            self.record_payoff("main", key, score)
        elif assignment.mode == "expert_script" and assignment.style in CRAZY_STYLES:
            history = self.crazy_results.setdefault(assignment.style, [])
            # A script self-kill, teacher kill, or safety-delivered terminal is
            # visible gameplay but not evidence that main learned the style.
            progression_score = (
                0.0
                if score == 1.0 and terminal_info is not None and not main_policy_kill
                else score
            )
            history.append(progression_score)
            self.crazy_results[assignment.style] = history[-CRAZY_RESULT_WINDOW:]
        self.assignments.pop(episode_id, None)
        for key in [key for key in self.hidden if key[0] == episode_id]:
            self.hidden.pop(key, None)
        self._save()

    def note_evaluation(self, held_out_elo: float) -> bool:
        self.evaluations.append(float(held_out_elo))
        self.evaluations = self.evaluations[-5:]
        plateau = len(self.evaluations) == 5 and max(self.evaluations) - min(self.evaluations) < 5.0
        requested = plateau and self.can_specialize()
        self._save(exploiter_requested=requested)
        return requested

    def crazy_report(self) -> dict[str, Any]:
        scores = {
            style: sum(results) / len(results) if results else 0.0
            for style, results in self.crazy_results.items()
        }
        covered = [
            style for style, results in self.crazy_results.items()
            if len(results) >= CRAZY_MIN_SAMPLES_PER_STYLE
        ]
        passing = [style for style in covered if scores[style] >= CRAZY_STYLE_PASS_SCORE]
        samples = sum(len(self.crazy_results[style]) for style in covered)
        overall = (
            sum(sum(self.crazy_results[style]) for style in covered) / samples
            if samples else 0.0
        )
        ready = (
            len(covered) >= CRAZY_MIN_COVERED_STYLES
            and len(passing) >= CRAZY_MIN_PASSING_STYLES
            and overall >= CRAZY_OVERALL_PASS_SCORE
        )
        return {
            "ready": ready, "overall_score": overall,
            "covered_styles": len(covered), "passing_styles": len(passing),
            "required_covered_styles": CRAZY_MIN_COVERED_STYLES,
            "required_passing_styles": CRAZY_MIN_PASSING_STYLES,
            "style_scores": scores,
            "style_samples": {style: len(values) for style, values in self.crazy_results.items()},
        }

    def can_specialize(self) -> bool:
        return bool(self.crazy_report()["ready"])

    def competence_stage(self) -> str:
        """Return the matchmaking phase derived from persisted held-out results."""
        score = float(self.crazy_report()["overall_score"])
        if score < BOOTSTRAP_SCORE_CUTOFF:
            return "bootstrap"
        if score < DEVELOPING_SCORE_CUTOFF:
            return "developing"
        return "qualified"

    def recovery_active(self) -> bool:
        """Use extra self-play only while the broad script score is severe."""
        return float(self.crazy_report()["overall_score"]) < RECOVERY_SCORE_CUTOFF

    def matchmaking_shares(self) -> dict[str, float]:
        """Expose the active population mix without persisting redundant state."""
        stage = self.competence_stage()
        if stage == "bootstrap" and self.recovery_active():
            return dict(RECOVERY_MATCHMAKING_SHARES)
        if stage == "bootstrap":
            return dict(BOOTSTRAP_MATCHMAKING_SHARES)
        if stage == "developing":
            return dict(DEVELOPING_MATCHMAKING_SHARES)
        return dict(MATCHMAKING_SHARES)

    def matchmaking_report(self) -> dict[str, Any]:
        """Return persisted actual assignments and honest terminal outcomes."""
        assigned = self.matchmaking_audit["assigned"]
        completed = self.matchmaking_audit["completed"]
        total_assigned = max(0, int(assigned.get("total", 0)))
        actual_shares = {
            mode: int(assigned["by_mode"].get(mode, 0)) / total_assigned
            if total_assigned else 0.0
            for mode in MATCHMAKING_SHARES
        }
        return {
            "competence_stage": self.competence_stage(),
            "recovery_active": self.recovery_active(),
            "configured_shares": self.matchmaking_shares(),
            "assigned_total": total_assigned,
            "actual_shares": actual_shares,
            "assigned_by_mode": dict(assigned["by_mode"]),
            "assigned_by_style": dict(assigned["by_style"]),
            "assigned_by_opponent": dict(assigned["by_opponent"]),
            "completed_total": int(completed.get("total", 0)),
            "results_by_mode": _result_reports(completed["by_mode"]),
            "results_by_style": _result_reports(completed["by_style"]),
            "results_by_opponent": _result_reports(completed["by_opponent"]),
        }

    def _record_assignment(self, assignment: EpisodeAssignment) -> None:
        assigned = self.matchmaking_audit["assigned"]
        assigned["total"] += 1
        _increment_count(assigned["by_mode"], assignment.mode)
        if assignment.style:
            _increment_count(assigned["by_style"], assignment.style)
        _increment_count(
            assigned["by_opponent"], _assignment_opponent_key(assignment),
        )

    def _record_matchmaking_result(
        self,
        assignment: EpisodeAssignment,
        score: float,
        reason: str,
        policy_owned_kill: bool,
    ) -> None:
        completed = self.matchmaking_audit["completed"]
        completed["total"] += 1
        keys = [("by_mode", assignment.mode)]
        if assignment.style:
            keys.append(("by_style", assignment.style))
        keys.append(("by_opponent", _assignment_opponent_key(assignment)))
        for group, key in keys:
            counters = completed[group].setdefault(key, _empty_result_counts())
            counters["matches"] += 1
            if score >= 0.75:
                counters["wins"] += 1
            elif score >= 0.25:
                counters["draws"] += 1
            else:
                counters["losses"] += 1
            if reason == "timeout":
                counters["timeouts"] += 1
            elif reason == "disengaged":
                counters["disengaged"] += 1
            elif reason == "death":
                counters["deaths"] += 1
            if policy_owned_kill:
                counters["policy_owned_kills"] += 1

    def record_payoff(self, policy: str, opponent: str, score: float) -> None:
        """Update both sides of the bounded empirical population payoff matrix."""
        bounded = max(0.0, min(1.0, float(score)))
        for row, column, value in (
            (policy, opponent, bounded), (opponent, policy, 1.0 - bounded)
        ):
            entries = self.payoff_matrix.setdefault(row, {}).setdefault(column, [])
            entries.append(value)
            self.payoff_matrix[row][column] = entries[-PAYOFF_WINDOW:]
        self.payoffs = self.payoff_matrix.setdefault("main", {})

    def prune_pool(self) -> None:
        snapshots = sorted(self.checkpoint_dir.glob("policy-*.pt"), key=lambda path: path.stat().st_mtime)
        if len(snapshots) <= 30:
            return
        latest = set(snapshots[-10:])
        best = set(sorted(snapshots, key=lambda path: self.ratings.get(path.name, 1000.0), reverse=True)[:20])
        for path in snapshots:
            if path not in latest and path not in best:
                path.unlink(missing_ok=True)
                self.ratings.pop(path.name, None)
                self.policy_cache.pop(str(path), None)
        self._save()

    def _new_assignment(self, episode_id: str, agents: list[str]) -> EpisodeAssignment:
        randomizer = random.Random(_stable_seed(episode_id))
        if self.forced_opponent is not None:
            opponent = agents[randomizer.randrange(len(agents))]
            return EpisodeAssignment(episode_id, "historical", opponent, str(self.forced_opponent))
        draw = randomizer.random()
        opponent = agents[randomizer.randrange(len(agents))]
        shares = self.matchmaking_shares()
        historical_cutoff = shares["historical"]
        exploiter_cutoff = historical_cutoff + shares["exploiter"]
        script_cutoff = exploiter_cutoff + shares["expert_script"]
        if draw < historical_cutoff:
            checkpoint = self._pfsp_checkpoint(randomizer, exploiters=False)
            if checkpoint is not None:
                return EpisodeAssignment(episode_id, "historical", opponent, str(checkpoint))
            # Severe recovery deliberately stays on the safest self-play
            # fallback. Once broad script competence clears that floor, a
            # missing population bucket is more useful at the attainable
            # expert-script frontier than as extra timeout-heavy mirror play.
            if self.recovery_active():
                return EpisodeAssignment(episode_id, "mirror")
            return self._expert_script_assignment(episode_id, opponent, randomizer)
        elif draw < exploiter_cutoff:
            checkpoint = self._pfsp_checkpoint(randomizer, exploiters=True)
            if checkpoint is not None:
                return EpisodeAssignment(episode_id, "exploiter", opponent, str(checkpoint))
            if self.recovery_active():
                # Below the severe-recovery cutoff, retain the conservative
                # historical-then-mirror behavior until a real exploiter is
                # promoted into the pool.
                checkpoint = self._pfsp_checkpoint(randomizer, exploiters=False)
                if checkpoint is not None:
                    return EpisodeAssignment(
                        episode_id, "historical", opponent, str(checkpoint)
                    )
                return EpisodeAssignment(episode_id, "mirror")
            return self._expert_script_assignment(episode_id, opponent, randomizer)
        elif draw < script_cutoff:
            return self._expert_script_assignment(episode_id, opponent, randomizer)
        else:
            return EpisodeAssignment(episode_id, "mirror")

    def _expert_script_assignment(
        self,
        episode_id: str,
        opponent: str,
        randomizer: random.Random,
    ) -> EpisodeAssignment:
        return EpisodeAssignment(
            episode_id,
            "expert_script",
            opponent,
            style=self._crazy_style(randomizer),
        )

    def _crazy_style(self, randomizer: random.Random) -> str:
        # Early training needs opponents near the attainable frontier, not a
        # stream consisting almost entirely of the scripts that crush main.
        # Hard and under-covered styles retain explicit probability mass so
        # this curriculum cannot hide a weakness indefinitely.
        stage = self.competence_stage()
        known_scores = [
            sum(results) / len(results)
            for results in self.crazy_results.values() if results
        ]
        if stage == "bootstrap" and self.recovery_active():
            best_known = max(known_scores, default=0.40)
            frontier_target = max(0.25, min(0.40, best_known))
            frontier_mix, hard_mix, coverage_mix = 0.85, 0.08, 0.07
        elif stage == "bootstrap":
            best_known = max(known_scores, default=0.45)
            frontier_target = max(0.25, min(0.45, best_known))
            frontier_mix, hard_mix, coverage_mix = 0.75, 0.15, 0.10
        elif stage == "developing":
            frontier_target = 0.50
            frontier_mix, hard_mix, coverage_mix = 0.55, 0.35, 0.10
        else:
            frontier_target = 0.50
            frontier_mix, hard_mix, coverage_mix = 0.0, 1.0, 0.0
        weights: list[float] = []
        for style in CRAZY_STYLES:
            results = self.crazy_results.get(style, [])
            score = sum(results) / len(results) if results else 0.5
            frontier = math.exp(-0.5 * ((score - frontier_target) / 0.18) ** 2)
            hard = 0.05 + (1.0 - score) ** 2
            missing = max(0, CRAZY_MIN_SAMPLES_PER_STYLE - len(results))
            coverage = (missing + 1.0) / (CRAZY_MIN_SAMPLES_PER_STYLE + 1.0)
            weights.append(
                0.01
                + frontier_mix * frontier
                + hard_mix * hard
                + coverage_mix * coverage
            )
        return randomizer.choices(CRAZY_STYLES, weights=weights, k=1)[0]

    def _near_even_checkpoint(self) -> Path | None:
        snapshots = list(self.checkpoint_dir.glob("policy-*.pt"))
        if not snapshots:
            return None
        return min(snapshots, key=lambda path: abs(_expected(self.main_elo, self.ratings.get(path.name, 1000.0)) - 0.5))

    def _pfsp_checkpoint(self, randomizer: random.Random, *, exploiters: bool) -> Path | None:
        snapshots = sorted([
            path for path in self.checkpoint_dir.glob("policy-*.pt")
            if ("exploiter" in path.name) is exploiters
        ], key=lambda path: path.name)
        if not snapshots:
            return None
        stage = self.competence_stage()
        weights: list[float] = []
        for path in snapshots:
            results = self.payoffs.get(path.name, [])
            main_win_rate = sum(results) / len(results) if results else 0.5
            # PFSP prioritizes opponents that beat the main, while the
            # near-even term prevents forgetting close historical matchups.
            weakness = (1.0 - main_win_rate) ** 2
            near_even = max(0.0, 1.0 - abs(main_win_rate - 0.5) * 2.0)
            coverage = 1.0 / math.sqrt(1.0 + len(results))
            hard_pfsp = 0.05 + weakness + 0.25 * near_even + coverage
            if stage == "qualified":
                weights.append(hard_pfsp)
                continue
            # A 45% empirical main win rate is difficult enough to improve
            # against while still producing successful trajectories for PPO.
            frontier = math.exp(-0.5 * ((main_win_rate - 0.45) / 0.22) ** 2)
            if stage == "bootstrap" and self.recovery_active():
                weights.append(0.90 * frontier + 0.07 * hard_pfsp + 0.03 * coverage)
            elif stage == "bootstrap":
                weights.append(0.82 * frontier + 0.13 * hard_pfsp + 0.05 * coverage)
            else:
                weights.append(0.55 * frontier + 0.40 * hard_pfsp + 0.05 * coverage)
        return randomizer.choices(snapshots, weights=weights, k=1)[0]

    @staticmethod
    def _candidate_key(source: Path) -> str:
        return hashlib.sha256(source.read_bytes()).hexdigest()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.main_elo = float(value.get("main_elo", 1000.0))
            self.ratings = {str(key): float(rating) for key, rating in value.get("ratings", {}).items()}
            stored_matrix = value.get("payoff_matrix")
            if isinstance(stored_matrix, dict):
                self.payoff_matrix = {
                    str(row): {
                        str(column): [max(0.0, min(1.0, float(score))) for score in scores][-PAYOFF_WINDOW:]
                        for column, scores in columns.items() if isinstance(scores, list)
                    }
                    for row, columns in stored_matrix.items() if isinstance(columns, dict)
                }
            else:
                legacy = value.get("payoffs", {})
                self.payoff_matrix = {"main": {
                    str(key): [max(0.0, min(1.0, float(score))) for score in scores][-PAYOFF_WINDOW:]
                    for key, scores in legacy.items() if isinstance(scores, list)
                }}
            self.payoffs = self.payoff_matrix.setdefault("main", {})
            self.exploiter_trials = {
                str(key): [1.0 if float(score) >= 0.5 else 0.0 for score in scores][-EXPLOITER_PROMOTION_MATCHES:]
                for key, scores in value.get("exploiter_trials", {}).items() if isinstance(scores, list)
            }
            self.evaluations = [float(entry) for entry in value.get("evaluations", [])][-5:]
            stored_results = value.get("crazy_results", {})
            if isinstance(stored_results, dict):
                for style in CRAZY_STYLES:
                    raw = stored_results.get(style, [])
                    if isinstance(raw, list):
                        self.crazy_results[style] = [
                            max(0.0, min(1.0, float(entry))) for entry in raw
                        ][-CRAZY_RESULT_WINDOW:]
            stored_audit = value.get("matchmaking_audit")
            if isinstance(stored_audit, dict):
                self.matchmaking_audit = _sanitize_matchmaking_audit(stored_audit)
        except (ValueError, OSError):
            pass

    def _save(self, exploiter_requested: bool = False) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "format_version": 4, "main_elo": self.main_elo, "ratings": self.ratings,
            "payoffs": self.payoffs, "payoff_matrix": self.payoff_matrix,
            "exploiter_trials": self.exploiter_trials,
            "evaluations": self.evaluations, "exploiter_requested": exploiter_requested,
            "crazy_results": self.crazy_results,
            "crazy_report": self.crazy_report(),
            "competence_stage": self.competence_stage(),
            "matchmaking_shares": self.matchmaking_shares(),
            "matchmaking_audit": self.matchmaking_audit,
            "matchmaking_report": self.matchmaking_report(),
        }, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)


def scripted_action(observation: dict[str, Any], style: str) -> dict[str, Any]:
    opponent = observation.get("opponent")
    if not opponent:
        return _wire_action()
    # ObservationV1 keeps the original wrong-sign coordinate transform for
    # checkpoint compatibility. New workers also emit a physically correct
    # Mineflayer body frame; scripted league opponents must prefer it or they
    # turn away from the fighter at non-zero headings.
    relative = opponent.get("body_relative_position") or opponent["relative_position"]
    computed_distance = math.sqrt(sum(float(relative[axis]) ** 2 for axis in ("x", "y", "z")))
    distance = float(opponent.get("distance", computed_distance))
    yaw_error = float(opponent.get(
        "bearing_error", math.atan2(-float(relative["x"]), -float(relative["z"]))
    ))
    fallback_pitch_error = math.atan2(
        float(relative["y"]),
        max(1e-6, math.hypot(float(relative["x"]), float(relative["z"]))),
    ) - float(observation["self"].get("pitch", 0.0))
    pitch_error = float(opponent.get("pitch_error", fallback_pitch_error))
    tick = int(observation["match"]["tick"])
    mask = observation.get("action_mask") or {}
    forward = 1 if distance > (3.0 if style != "retreat" else 5.0) else (-1 if style in ("retreat", "defensive", "spacing") else 0)
    strafe = 0 if style == "rush" else (1 if (tick // 17) % 2 else -1)
    if style == "erratic":
        randomizer = random.Random(_stable_seed(f"{observation['match']['episode_id']}:{tick}"))
        strafe = randomizer.choice((-1, 0, 1))
    primary = "attack" if distance <= 3.0 and mask.get("attack") else "none"
    hotbar = -1
    jump = style in ("jump_critical", "high_ground") and tick % 13 == 0

    if style == "spacing":
        forward = 1 if distance > 3.15 else (-1 if distance < 2.7 else 0)
    elif style == "sprint_reset":
        # Release forward for one tick after each attack cadence so the next
        # approach creates a fresh sprint knockback window.
        forward = 0 if tick % 7 == 0 else (1 if distance > 2.6 else 0)
    elif style == "counterattack":
        self_hurt = int((observation.get("self") or {}).get("hurt_time", 0)) > 0
        forward = 1 if self_hurt else (-1 if distance < 4.5 else 0)
        primary = "attack" if self_hurt and distance <= 3.2 and mask.get("attack") else "none"

    if style in (
        "crystal_rush", "crystal_kamikaze", "orbit_crystal", "safe_crystal",
        "crystal_bait", "crystal_defense", "crystal_escape",
    ):
        crystal_slot = _hotbar_slot(observation, ("end_crystal",))
        if mask.get("crystal_attack_ready") and mask.get("attack"):
            primary = "attack"
        elif mask.get("crystal_place_ready") and mask.get("use_main") and crystal_slot >= 0:
            primary, hotbar = "use_main", crystal_slot
        if style == "crystal_kamikaze":
            # Intentionally reckless close pressure teaches the main policy to
            # punish and survive unusual high-variance crystal sequences.
            forward, strafe, jump = 1, 0, tick % 11 == 0
        elif style == "orbit_crystal":
            forward = 1 if distance > 5.0 else (-1 if distance < 3.5 else 0)
            strafe = 1 if (tick // 23) % 2 else -1
        elif style == "safe_crystal":
            forward = -1 if distance < 5.0 else 0
            strafe = 1 if (tick // 19) % 2 else -1
        elif style == "crystal_bait":
            forward = -1 if tick % 40 < 24 else 1
            strafe = 1 if (tick // 13) % 2 else -1
        elif style == "crystal_defense":
            forward = -1 if distance < 6.0 else 0
            strafe = 1 if (tick // 9) % 2 else -1
        elif style == "crystal_escape":
            forward = -1 if mask.get("crystal_attack_ready") or distance < 5.5 else 1
            strafe = 1 if (tick // 7) % 2 else -1
    elif style == "obsidian_builder":
        obsidian_slot = _hotbar_slot(observation, ("obsidian",))
        if _tactical_place_ready(observation) and mask.get("use_main") and obsidian_slot >= 0:
            primary, hotbar = "use_main", obsidian_slot
        forward = 1 if distance > 4.0 else -1
        strafe = 1 if (tick // 31) % 2 else -1
    elif style == "tactical_miner":
        pickaxe_slot = _hotbar_slot(observation, ("diamond_pickaxe", "netherite_pickaxe", "iron_pickaxe"))
        if mask.get("tactical_block_break_ready") and mask.get("attack") and pickaxe_slot >= 0:
            primary, hotbar = "attack", pickaxe_slot
        forward = 1 if distance > 4.0 else 0
        strafe = 1 if (tick // 29) % 2 else -1
    elif style == "bait_and_counter":
        opponent_hurt = int(opponent.get("hurt_time", 0)) > 0
        self_hurt = int((observation.get("self") or {}).get("hurt_time", 0)) > 0
        forward = 1 if opponent_hurt or self_hurt else (-1 if distance < 5.5 else 0)
        strafe = 1 if (tick // 11) % 2 else -1
        primary = "attack" if forward > 0 and distance <= 3.2 and mask.get("attack") else "none"
    elif style in ("cover_builder", "terrain_trap"):
        obsidian_slot = _hotbar_slot(observation, ("obsidian",))
        if _tactical_place_ready(observation) and mask.get("use_main") and obsidian_slot >= 0:
            primary, hotbar = "use_main", obsidian_slot
        forward = -1 if distance < (4.5 if style == "cover_builder" else 3.5) else 0
        strafe = 1 if (tick // 11) % 2 else -1
    elif style == "high_ground":
        forward = 1 if distance > 3.0 else 0
        jump = tick % 5 == 0
    elif style in ("totem_pressure", "heal_pressure"):
        self_state = observation.get("self") or {}
        if style == "totem_pressure":
            totem_slot = _hotbar_slot(observation, ("totem_of_undying",))
            offhand = self_state.get("offhand") or {}
            if str(offhand.get("name", "")) != "totem_of_undying" and totem_slot >= 0 and mask.get("swap_offhand"):
                hotbar, primary = totem_slot, "none"
                return _wire_action(
                    forward=-1, strafe=strafe, yaw_delta=max(-0.45, min(0.45, yaw_error)),
                    pitch_delta=max(-0.3, min(0.3, pitch_error)), hotbar=hotbar,
                    swap_offhand=True,
                )
        else:
            apple_slot = _hotbar_slot(observation, ("enchanted_golden_apple", "golden_apple"))
            if float(self_state.get("health", 20.0)) <= 12.0 and apple_slot >= 0 and mask.get("use_main"):
                primary, hotbar, forward = "use_main", apple_slot, -1

    return _wire_action(
        forward=forward, strafe=strafe, sprint=forward > 0,
        jump=jump,
        yaw_delta=max(-0.45, min(0.45, yaw_error)), pitch_delta=max(-0.3, min(0.3, pitch_error)),
        primary=primary, hotbar=hotbar,
    )


def _hotbar_slot(observation: dict[str, Any], names: tuple[str, ...]) -> int:
    hotbar = (observation.get("self") or {}).get("hotbar") or []
    legal = (observation.get("action_mask") or {}).get("hotbar") or []
    for index, item in enumerate(hotbar[:9]):
        if index < len(legal) and not legal[index]:
            continue
        if isinstance(item, dict) and str(item.get("name", "")) in names and int(item.get("count", 0)) > 0:
            return index
    return -1


def _empty_result_counts() -> dict[str, int]:
    return {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "timeouts": 0,
        "disengaged": 0,
        "deaths": 0,
        "policy_owned_kills": 0,
    }


def _empty_matchmaking_audit() -> dict[str, Any]:
    return {
        "assigned": {
            "total": 0,
            "by_mode": {},
            "by_style": {},
            "by_opponent": {},
        },
        "completed": {
            "total": 0,
            "by_mode": {},
            "by_style": {},
            "by_opponent": {},
        },
    }


def _assignment_opponent_key(assignment: EpisodeAssignment) -> str:
    if assignment.style:
        return f"script:{assignment.style}"
    if assignment.checkpoint:
        return Path(assignment.checkpoint).name
    return assignment.mode


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _sanitize_matchmaking_audit(value: dict[str, Any]) -> dict[str, Any]:
    result = _empty_matchmaking_audit()
    assigned = value.get("assigned")
    if isinstance(assigned, dict):
        result["assigned"]["total"] = _nonnegative_int(assigned.get("total"))
        for group in ("by_mode", "by_style", "by_opponent"):
            raw = assigned.get(group)
            if isinstance(raw, dict):
                result["assigned"][group] = {
                    str(key): _nonnegative_int(count)
                    for key, count in raw.items()
                }
    completed = value.get("completed")
    if isinstance(completed, dict):
        result["completed"]["total"] = _nonnegative_int(completed.get("total"))
        for group in ("by_mode", "by_style", "by_opponent"):
            raw = completed.get(group)
            if not isinstance(raw, dict):
                continue
            result["completed"][group] = {
                str(key): {
                    field: _nonnegative_int(
                        counters.get(field) if isinstance(counters, dict) else 0
                    )
                    for field in _empty_result_counts()
                }
                for key, counters in raw.items()
            }
    return result


def _result_reports(values: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for key in sorted(values):
        counters = values[key]
        matches = max(0, int(counters.get("matches", 0)))
        reports[key] = {
            **counters,
            "win_rate": counters.get("wins", 0) / matches if matches else 0.0,
            "timeout_rate": counters.get("timeouts", 0) / matches if matches else 0.0,
            "non_timeout_rate": 1.0 - counters.get("timeouts", 0) / matches
            if matches else 0.0,
            "policy_owned_kill_rate": counters.get("policy_owned_kills", 0) / matches
            if matches else 0.0,
        }
    return reports


def _tactical_place_ready(observation: dict[str, Any]) -> bool:
    """Infer the omitted V1 placement mask from its declared candidate slot.

    Older workers compute this legality internally but do not serialize the
    optional mask member. A marked, reachable, raycastable block together with
    legal use_main is the equivalent wire-level signal.
    """
    mask = observation.get("action_mask") or {}
    if mask.get("tactical_block_place_ready"):
        return True
    if not mask.get("use_main"):
        return False
    return any(
        isinstance(block, dict)
        and block.get("tactical_placement_target") is True
        and block.get("within_reach") is not False
        and block.get("raycastable") is not False
        for block in (observation.get("blocks") or [])
    )


def _wire_action(**updates: Any) -> dict[str, Any]:
    value = {"schema_version": 1, "forward": 0, "strafe": 0, "jump": False, "sprint": False,
             "sneak": False, "yaw_delta": 0.0, "pitch_delta": 0.0, "primary": "none",
             "release_use": False, "hotbar": -1, "swap_offhand": False}
    value.update(updates)
    return value


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _expected(main: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - main) / 400.0))
