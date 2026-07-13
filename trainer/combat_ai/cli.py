from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch

from .config import PPOConfig, ServiceConfig
from .browser_bridge import serve_browser_bridge
from .curriculum import CurriculumState
from .export import export_flat_weights, export_onnx, load_policy
from .imitation import behavior_clone
from .league import LeagueManager
from .model import CombatPolicy
from .ppo import choose_device
from .service import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="mcai-trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("serve", help="run inference and online recurrent PPO")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8766)
    server.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    server.add_argument("--rollout-steps", type=int, default=8192)
    server.add_argument("--deterministic", action="store_true")
    server.add_argument("--imitation-data", type=Path,
                        help="whole-match JSONL replayed at 10%%, decaying over five million ticks")
    server.add_argument("--cpu-threads", type=int, default=0,
                        help="CPU threads for Windows training; 0 reserves roughly half for Paper")
    server.add_argument("--initialize-from", type=Path, action="append", default=[],
                        help="initialize an empty run from one or more sword/crystal checkpoints (averaged)")
    server.add_argument("--exploiter-target", type=Path,
                        help="train only against this frozen main checkpoint in an isolated checkpoint directory")
    cloning = subparsers.add_parser("clone", help="behavior-clone whole-match JSONL demonstrations")
    cloning.add_argument("demonstrations", type=Path)
    cloning.add_argument("--output", type=Path, default=Path("checkpoints/imitation.pt"))
    cloning.add_argument("--epochs", type=int, default=30)
    exporting = subparsers.add_parser("export", help="export ONNX and browser flat weights")
    exporting.add_argument("checkpoint", type=Path)
    exporting.add_argument("--directory", type=Path, default=Path("exports"))
    browser = subparsers.add_parser("browser", help="serve frozen JSON inference to the Eagler adapter")
    browser.add_argument("--checkpoint", type=Path)
    browser.add_argument("--host", default="127.0.0.1")
    browser.add_argument("--port", type=int, default=8767)
    browser.add_argument("--stochastic", action="store_true")
    curriculum = subparsers.add_parser("curriculum", help="check and persist a curriculum evaluation gate")
    curriculum.add_argument("--state", type=Path, default=Path("checkpoints/curriculum.json"))
    curriculum.add_argument("--results", type=Path, help="JSON containing stage and named held-out metrics")
    league = subparsers.add_parser("league-eval", help="record held-out Elo and detect a five-evaluation plateau")
    league.add_argument("held_out_elo", type=float)
    league.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    promote = subparsers.add_parser("promote-exploiter", help="atomically add an exploiter to the main frozen pool")
    promote.add_argument("checkpoint", type=Path)
    promote.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    arguments = parser.parse_args()
    if arguments.command == "serve":
        ppo = PPOConfig(rollout_agent_ticks=arguments.rollout_steps)
        service = ServiceConfig(
            host=arguments.host, port=arguments.port, checkpoint_dir=arguments.checkpoints,
            deterministic_inference=arguments.deterministic, cpu_threads=arguments.cpu_threads,
        )
        asyncio.run(serve(
            ppo, service, arguments.imitation_data, arguments.initialize_from, arguments.exploiter_target
        ))
    elif arguments.command == "clone":
        metrics = behavior_clone(
            CombatPolicy(), arguments.demonstrations, arguments.output, choose_device(), epochs=arguments.epochs
        )
        print(json.dumps(metrics))
    elif arguments.command == "export":
        policy = load_policy(arguments.checkpoint, torch.device("cpu"))
        arguments.directory.mkdir(parents=True, exist_ok=True)
        export_onnx(policy, arguments.directory / "policy.onnx")
        export_flat_weights(
            policy, arguments.directory / "policy.manifest.json", arguments.directory / "policy.weights.bin"
        )
    elif arguments.command == "browser":
        asyncio.run(serve_browser_bridge(
            arguments.checkpoint, arguments.host, arguments.port, deterministic=not arguments.stochastic
        ))
    elif arguments.command == "curriculum":
        state = CurriculumState.load(arguments.state)
        result = {"current_stage": state.current_stage, "completed": state.completed}
        if arguments.results:
            evaluation = json.loads(arguments.results.read_text(encoding="utf-8"))
            result.update(state.evaluate(evaluation))
            state.save(arguments.state)
        print(json.dumps(result, indent=2))
    elif arguments.command == "league-eval":
        league_state = LeagueManager(arguments.checkpoints, torch.device("cpu"))
        requested = league_state.note_evaluation(arguments.held_out_elo)
        print(json.dumps({"held_out_elo": arguments.held_out_elo, "exploiter_requested": requested}))
    elif arguments.command == "promote-exploiter":
        league_state = LeagueManager(arguments.checkpoints, torch.device("cpu"))
        destination = league_state.add_frozen_checkpoint(arguments.checkpoint)
        print(json.dumps({"promoted": str(destination)}))


if __name__ == "__main__":
    main()
