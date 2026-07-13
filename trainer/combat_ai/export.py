from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import BLOCK_SIZE, ENTITY_SIZE, LEGAL_SIZE, MAX_BLOCKS, MAX_ENTITIES, OPPONENT_SIZE, SELF_SIZE, FeatureBatch
from .model import CATEGORICAL_SIZES, HIDDEN_SIZE, CombatPolicy


class ExportWrapper(nn.Module):
    def __init__(self, policy: CombatPolicy):
        super().__init__()
        self.policy = policy

    def forward(
        self, self_state: torch.Tensor, opponent: torch.Tensor, opponent_mask: torch.Tensor,
        entities: torch.Tensor, entity_mask: torch.Tensor, blocks: torch.Tensor,
        block_mask: torch.Tensor, legal: torch.Tensor, hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output = self.policy(FeatureBatch(
            self_state=self_state, opponent=opponent, opponent_mask=opponent_mask,
            entities=entities, entity_mask=entity_mask, blocks=blocks, block_mask=block_mask, legal=legal,
        ), hidden)
        logits = tuple(output.logits[name] for name in CATEGORICAL_SIZES)
        return (*logits, output.camera_mean, output.camera_log_std, output.value, output.hidden)


def load_policy(checkpoint: Path, device: torch.device | str = "cpu") -> CombatPolicy:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = CombatPolicy().to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy


def export_onnx(policy: CombatPolicy, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = next(policy.parameters()).device
    inputs = (
        torch.zeros(1, SELF_SIZE, device=device), torch.zeros(1, OPPONENT_SIZE, device=device),
        torch.zeros(1, 1, device=device), torch.zeros(1, MAX_ENTITIES, ENTITY_SIZE, device=device),
        torch.zeros(1, MAX_ENTITIES, device=device), torch.zeros(1, MAX_BLOCKS, BLOCK_SIZE, device=device),
        torch.zeros(1, MAX_BLOCKS, device=device), torch.zeros(1, LEGAL_SIZE, device=device),
        torch.zeros(1, 1, HIDDEN_SIZE, device=device),
    )
    input_names = ["self_state", "opponent", "opponent_mask", "entities", "entity_mask",
                   "blocks", "block_mask", "legal", "hidden"]
    output_names = [*CATEGORICAL_SIZES, "camera_mean", "camera_log_std", "value", "next_hidden"]
    dynamic_axes = {name: {0: "batch"} for name in input_names[:-1]}
    dynamic_axes["hidden"] = {1: "batch"}
    dynamic_axes.update({name: {0: "batch"} for name in output_names[:-1]})
    dynamic_axes["next_hidden"] = {1: "batch"}
    torch.onnx.export(
        ExportWrapper(policy), inputs, destination, input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17, do_constant_folding=True,
    )


def export_flat_weights(policy: CombatPolicy, manifest_path: Path, weights_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    arrays = []
    offset = 0
    for name, value in policy.state_dict().items():
        array = np.ascontiguousarray(value.detach().cpu().numpy().astype("<f4"))
        entries.append({"name": name, "shape": list(array.shape), "offset_f32": offset, "length": array.size})
        arrays.append(array.reshape(-1))
        offset += array.size
    packed = np.concatenate(arrays).astype("<f4", copy=False)
    weights_path.write_bytes(packed.tobytes())
    manifest_path.write_text(json.dumps({
        "format": "mcai-flat-f32", "format_version": 1, "little_endian": True,
        "parameter_count": int(packed.size), "architecture": "structured-mlp-gru128-v1",
        "categorical_heads": CATEGORICAL_SIZES, "tensors": entries,
    }, indent=2) + "\n", encoding="utf-8")
