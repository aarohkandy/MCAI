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


@dataclass
class EpisodeAssignment:
    episode_id: str
    mode: str
    opponent_agent: str | None = None
    checkpoint: str | None = None
    style: str | None = None


class LeagueManager:
    """Deterministic 40/40/20 episode assignment with an Elo-rated frozen pool."""

    def __init__(self, checkpoint_dir: Path, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.metadata_path = checkpoint_dir / "league.json"
        self.assignments: dict[str, EpisodeAssignment] = {}
        self.policy_cache: dict[str, CombatPolicy] = {}
        self.hidden: dict[tuple[str, str], torch.Tensor] = {}
        self.main_elo = 1000.0
        self.ratings: dict[str, float] = {}
        self.evaluations: list[float] = []
        self.forced_opponent: Path | None = None
        self._load()

    def assign_batch(self, steps: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[str]] = {}
        for step in steps:
            episode = str(step["observation"]["match"]["episode_id"])
            grouped.setdefault(episode, []).append(str(step["agent_id"]))
        for episode, agents in grouped.items():
            if episode in self.assignments or episode == "waiting" or len(agents) < 2:
                continue
            self.assignments[episode] = self._new_assignment(episode, sorted(set(agents)))

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

    def opponent_action(self, assignment: EpisodeAssignment, agent_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        if assignment.mode == "historical" and assignment.checkpoint:
            path = Path(assignment.checkpoint)
            policy = self.policy_cache.get(str(path))
            if policy is None:
                policy = load_policy(path, self.device)
                self.policy_cache[str(path)] = policy
            key = (assignment.episode_id, agent_id)
            hidden = self.hidden.get(key, policy.initial_hidden(1, self.device))
            features = batch_observations([observation], self.device)
            with torch.no_grad():
                output = policy(features, hidden)
                actions, _, _, _ = sample_actions(output, features, deterministic=False)
            self.hidden[key] = output.hidden
            return actions[0]
        return scripted_action(observation, assignment.style or "erratic")

    def record_result(self, episode_id: str, main_agent: str, reward: float, outcome: str | None = None) -> None:
        assignment = self.assignments.get(episode_id)
        if assignment is None or assignment.opponent_agent == main_agent:
            return
        if assignment.mode == "historical" and assignment.checkpoint:
            key = Path(assignment.checkpoint).name
            opponent_elo = self.ratings.get(key, 1000.0)
            expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.main_elo) / 400.0))
            score = ({"win": 1.0, "draw": 0.5, "loss": 0.0}.get(str(outcome).lower())
                     if outcome is not None else None)
            if score is None:
                score = 1.0 if reward > 0 else (0.5 if abs(reward + 0.05) < 1e-6 else 0.0)
            change = 24.0 * (score - expected)
            self.main_elo += change
            self.ratings[key] = opponent_elo - change
        self.assignments.pop(episode_id, None)
        for key in [key for key in self.hidden if key[0] == episode_id]:
            self.hidden.pop(key, None)
        self._save()

    def note_evaluation(self, held_out_elo: float) -> bool:
        self.evaluations.append(float(held_out_elo))
        self.evaluations = self.evaluations[-5:]
        plateau = len(self.evaluations) == 5 and max(self.evaluations) - min(self.evaluations) < 5.0
        self._save(exploiter_requested=plateau)
        return plateau

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
        if draw < 0.4:
            return EpisodeAssignment(episode_id, "mirror")
        opponent = agents[randomizer.randrange(len(agents))]
        if draw < 0.8:
            checkpoint = self._near_even_checkpoint()
            if checkpoint is not None:
                return EpisodeAssignment(episode_id, "historical", opponent, str(checkpoint))
        styles = ("rush", "strafe", "retreat", "jump_critical", "defensive", "erratic")
        return EpisodeAssignment(episode_id, "scripted", opponent, style=randomizer.choice(styles))

    def _near_even_checkpoint(self) -> Path | None:
        snapshots = list(self.checkpoint_dir.glob("policy-*.pt"))
        if not snapshots:
            return None
        return min(snapshots, key=lambda path: abs(_expected(self.main_elo, self.ratings.get(path.name, 1000.0)) - 0.5))

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.main_elo = float(value.get("main_elo", 1000.0))
            self.ratings = {str(key): float(rating) for key, rating in value.get("ratings", {}).items()}
            self.evaluations = [float(entry) for entry in value.get("evaluations", [])][-5:]
        except (ValueError, OSError):
            pass

    def _save(self, exploiter_requested: bool = False) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "format_version": 1, "main_elo": self.main_elo, "ratings": self.ratings,
            "evaluations": self.evaluations, "exploiter_requested": exploiter_requested,
        }, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)


def scripted_action(observation: dict[str, Any], style: str) -> dict[str, Any]:
    opponent = observation.get("opponent")
    if not opponent:
        return _wire_action()
    relative = opponent["relative_position"]
    distance = math.sqrt(sum(float(relative[axis]) ** 2 for axis in ("x", "y", "z")))
    yaw_error = math.atan2(-float(relative["x"]), -float(relative["z"]))
    pitch_error = math.atan2(
        float(relative["y"]), max(1e-6, math.hypot(float(relative["x"]), float(relative["z"])))
    ) - float(observation["self"].get("pitch", 0.0))
    tick = int(observation["match"]["tick"])
    forward = 1 if distance > (3.0 if style != "retreat" else 5.0) else (-1 if style in ("retreat", "defensive") else 0)
    strafe = 0 if style == "rush" else (1 if (tick // 17) % 2 else -1)
    if style == "erratic":
        randomizer = random.Random(_stable_seed(f"{observation['match']['episode_id']}:{tick}"))
        strafe = randomizer.choice((-1, 0, 1))
    return _wire_action(
        forward=forward, strafe=strafe, sprint=forward > 0,
        jump=style == "jump_critical" and tick % 13 == 0,
        yaw_delta=max(-0.45, min(0.45, yaw_error)), pitch_delta=max(-0.3, min(0.3, pitch_error)),
        primary="attack" if distance <= 3.0 and observation["action_mask"].get("attack") else "none",
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
