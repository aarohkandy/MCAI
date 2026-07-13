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
    features = batch_observations([value])
    hidden = policy.initial_hidden(1, "cpu")
    with torch.no_grad():
        output = policy(features, hidden)
    expected = {
        "value": float(output.value[0]),
        "camera_mean": output.camera_mean[0].tolist(),
        "hidden": output.hidden[0, 0].tolist(),
        "logits": {name: output.logits[name][0].tolist() for name in CATEGORICAL_SIZES},
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
        completed = subprocess.run([
            "node", str(ROOT / "eagler-mod" / "test" / "flat-parity.mjs"),
            str(ROOT / "eagler-mod" / "flat-policy.js"), str(manifest), str(weights),
            str(observation_file), str(expected_file),
        ], check=True, text=True, capture_output=True)
        result = {"pytorch_vs_browser": json.loads(completed.stdout)}
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            inputs = {name: tensor.detach().numpy() for name, tensor in zip(
                ["self_state", "opponent", "opponent_mask", "entities", "entity_mask",
                 "blocks", "block_mask", "legal", "hidden"], (*features.as_tuple(), hidden)
            )}
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
