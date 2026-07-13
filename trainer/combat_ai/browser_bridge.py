from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import torch
import websockets

from .distribution import sample_actions
from .export import load_policy
from .features import batch_observations
from .model import CombatPolicy
from .ppo import choose_device


class BrowserBridge:
    """Frozen JSON inference endpoint used while the browser adapter is being debugged."""

    def __init__(self, checkpoint: Path | None, deterministic: bool = True):
        self.device = choose_device()
        self.policy = load_policy(checkpoint, self.device) if checkpoint else CombatPolicy().to(self.device)
        self.policy.eval()
        self.deterministic = deterministic

    async def handle(self, websocket: Any) -> None:
        hidden = self.policy.initial_hidden(1, self.device)
        async for data in websocket:
            message = json.loads(data)
            message_type = message.get("type")
            if message_type == "browser_hello":
                await websocket.send(json.dumps({
                    "type": "browser_ready", "schema_version": 1,
                    "policy_version": 0, "device": str(self.device), "frozen": True,
                }))
                continue
            if message_type == "emergency_stop":
                hidden.zero_()
                continue
            if message_type != "browser_step":
                continue
            observation = message["observation"]
            features = batch_observations([observation], self.device)
            with torch.no_grad():
                output = self.policy(features, hidden)
                actions, _, _, _ = sample_actions(output, features, self.deterministic)
            hidden = output.hidden
            if message.get("terminated") or message.get("truncated"):
                hidden = self.policy.initial_hidden(1, self.device)
            target = observation.get("opponent")
            await websocket.send(json.dumps({
                "type": "browser_action", "schema_version": 1,
                "sequence": message.get("sequence", 0), "policy_version": 0,
                "action": actions[0], "value": float(output.value[0]),
                "target": None if target is None else target.get("relative_position"),
            }))


async def serve_browser_bridge(
    checkpoint: Path | None, host: str = "127.0.0.1", port: int = 8767, deterministic: bool = True
) -> None:
    bridge = BrowserBridge(checkpoint, deterministic)
    async with websockets.serve(bridge.handle, host, port, max_size=32 * 1024 * 1024):
        print(json.dumps({"event": "browser_bridge_ready", "host": host, "port": port,
                          "device": str(bridge.device), "frozen": True}), flush=True)
        await asyncio.Future()
