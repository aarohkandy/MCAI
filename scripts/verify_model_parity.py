from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer"))
sys.path.insert(0, str(ROOT / "trainer" / "tests"))

from combat_ai.export import ExportWrapper, export_flat_weights, export_onnx  # noqa: E402
from combat_ai.features import batch_observations  # noqa: E402
from combat_ai.model import CATEGORICAL_SIZES, CombatPolicy  # noqa: E402
from fixtures import observation  # noqa: E402


def main() -> None:
    torch.manual_seed(73)
    policy = CombatPolicy().eval()
    value = observation()
    value["match"].update({
        "mode": "terrain", "lane": "terrain", "action_delay_ticks": 2,
        "observation_delay_ticks": 1, "arena_radius": 5, "curriculum_stage": 1,
    })
    value["self"].update({
        "yaw": 0.7,
        "mainhand": {
            "name": "diamond_sword", "count": 1, "durability": 12,
            "max_durability": 1561, "enchant_hash": 73,
        },
    })
    value["self"]["hotbar"][3] = {
        "name": "end_crystal", "count": 64, "durability": 0,
        "max_durability": 0, "enchant_hash": 0,
    }
    value["opponent"].update({
        "body_relative_position": {"x": 1.0, "y": 0.2, "z": -3.0},
        "body_relative_velocity": {"x": -0.1, "y": 0.0, "z": 0.2},
        "distance": 3.17, "horizontal_distance": 3.16,
        "bearing_error": -0.32, "pitch_error": 0.08, "closing_speed": 0.22,
        "within_melee_reach": True, "aim_alignment": 0.91,
        "facing_toward_self": 0.74, "head_yaw": -0.4,
    })
    value["entities"] = [{
        "kind": "end_crystal",
        "relative_position": {"x": 0.5, "y": 0.0, "z": -2.5},
        "relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "body_relative_position": {"x": -0.25, "y": 0.0, "z": -2.54},
        "body_relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "age_ticks": 5, "distance": 2.55, "raycastable": False,
    }]
    value["blocks"] = [{
        "name": "obsidian",
        "relative_position": {"x": 1.5, "y": -1.0, "z": -2.5},
        "body_relative_position": {"x": -0.46, "y": -1.0, "z": -2.88},
        "body_relative_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "collision": "solid", "hardness": 50, "replaceable": False,
        "break_progress": 0.0, "crystal_clearance": True, "exposed_faces": 5,
        "distance": 2.92, "within_reach": True, "raycastable": False,
        "sample_age_ticks": 2,
    }]
    value["action_mask"].update({
        "crystal_place_ready": True, "crystal_attack_ready": True,
        "tactical_block_break_ready": True,
    })
    features = batch_observations([value])
    hidden = policy.initial_hidden(1, "cpu")
    with torch.no_grad():
        output = policy(features, hidden)
    expected = {
        "value": float(output.value[0]),
        "camera_mean": output.camera_mean[0].tolist(),
        "hidden": output.hidden[0, 0].tolist(),
        "logits": {name: output.logits[name][0].tolist() for name in CATEGORICAL_SIZES},
        "features": {
            name: tensor[0].detach().reshape(-1).tolist()
            for name, tensor in vars(features).items()
        },
    }
    with tempfile.TemporaryDirectory(prefix="mcai-parity-") as temporary:
        directory = Path(temporary)
        manifest = directory / "policy.manifest.json"
        weights = directory / "policy.weights.bin"
        onnx = directory / "policy.onnx"
        export_flat_weights(policy, manifest, weights)
        export_onnx(policy, onnx)
        observation_file = directory / "observation.json"
        expected_file = directory / "expected.json"
        observation_file.write_text(json.dumps(value), encoding="utf-8")
        expected_file.write_text(json.dumps(expected), encoding="utf-8")
        try:
            completed = subprocess.run([
                "node", str(ROOT / "eagler-mod" / "test" / "flat-parity.mjs"),
                str(ROOT / "eagler-mod" / "flat-policy.js"), str(manifest), str(weights),
                str(observation_file), str(expected_file),
            ], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            # Surface the browser-runtime assertion instead of hiding it behind
            # a generic subprocess failure.
            if error.stdout:
                print(error.stdout, end="")
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
            raise
        result = {"pytorch_vs_browser": json.loads(completed.stdout)}
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            inputs = {
                name: tensor.detach().numpy()
                for name, tensor in vars(features).items()
            }
            inputs["hidden"] = hidden.detach().numpy()
            onnx_outputs = session.run(None, inputs)
            torch_outputs = ExportWrapper(policy)(*features.as_tuple(), hidden)
            maximum = max(float((torch_value.detach() - torch.from_numpy(onnx_value)).abs().max())
                          for torch_value, onnx_value in zip(torch_outputs, onnx_outputs))
            if maximum > 1e-5:
                raise RuntimeError(f"ONNX parity exceeded 1e-5: {maximum}")
            result["pytorch_vs_onnx_maximum_difference"] = maximum
        except ImportError:
            result["pytorch_vs_onnx"] = "onnxruntime not installed"
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
