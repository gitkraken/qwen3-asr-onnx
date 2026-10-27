import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import awq_smooth
import quantize_static
from src.onnx_export import export_onnx
from tests.test_artifact_contract import _ort_result, _write_external_identity
from tests.test_tokenizer_transport import _source
from tests.tokenizer_fixtures import save_tokenizer_assets


def test_static_legacy_cache_is_preserved_and_only_matching_cache_is_reused(tmp_path, monkeypatch):
    source = _source(tmp_path / "source", "jinja")
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / "calib_n1_s1.npz"
    previous = np.array([{"x": np.full((1, 4), 99, dtype=np.float32)}], dtype=object)
    np.savez(legacy, init_inputs=previous, step_inputs=previous)
    legacy_bytes = legacy.read_bytes()
    collected = []

    def collect(*args, **kwargs):
        collected.append(kwargs)
        samples = [{"x": np.ones((1, 4), dtype=np.float32)}]
        return samples, samples

    monkeypatch.setattr(quantize_static, "collect_calibration_data", collect)

    def run(name):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "quantize_static.py",
                "--input",
                str(source),
                "--output",
                str(tmp_path / name),
                "--cache-dir",
                str(cache),
                "--n-samples",
                "1",
                "--n-steps",
                "1",
                "--no-share-weights",
            ],
        )
        quantize_static.main()

    run("first")
    assert len(collected) == 1
    run("second")
    assert len(collected) == 1
    (source / "preprocessor_config.json").write_text('{"changed": true}')
    run("third")
    assert len(collected) == 2
    assert legacy.read_bytes() == legacy_bytes
    assert len(list(cache.glob("*.npz"))) == 3


def test_awq_legacy_cache_is_preserved_and_checkpoint_identity_controls_reuse(tmp_path, monkeypatch):
    source, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(source)
    output.mkdir()
    legacy = output / "calibration_activations.npz"
    np.savez(legacy, norm=np.full(4, 99, dtype=np.float32))
    legacy_bytes = legacy.read_bytes()
    data = output / "encoder.onnx.data"
    np.array([42], dtype=np.float32).tofile(data)
    _write_external_identity(output, "aux.onnx", data.name)
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.model = torch.nn.Module()
    model.thinker.model.embed_tokens = torch.nn.Embedding(8, 4)
    model.config = SimpleNamespace(to_dict=lambda: {})
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: copy.deepcopy(model))
    collected = []
    scales = []

    def collect(*args, **kwargs):
        collected.append(kwargs)
        return {"norm": torch.arange(1, 5, dtype=torch.float32)}

    def compute(activations, **kwargs):
        scales.append(activations["norm"].clone())
        return activations

    def tiny_export(model, path, **kwargs):
        export_onnx(
            torch.nn.Identity(), (torch.ones(1, 4),), path, input_names=["x"], output_names=["y"], opset_version=17
        )

    monkeypatch.setattr(awq_smooth, "collect_activations", collect)
    monkeypatch.setattr(awq_smooth, "compute_scales", compute)
    monkeypatch.setattr(awq_smooth, "apply_smoothing", lambda *args: None)
    for name in ("export_encoder", "export_decoder_init", "export_decoder_step"):
        monkeypatch.setattr(awq_smooth, name, tiny_export)
    monkeypatch.setattr(
        awq_smooth, "write_config", lambda model, directory: (Path(directory) / "config.json").write_text("{}")
    )
    monkeypatch.setattr(
        sys, "argv", ["awq_smooth.py", "--model", str(source), "--output", str(output), "--n-samples", "1"]
    )
    awq_smooth.main()
    awq_smooth.main()
    assert len(collected) == 1
    for value in scales:
        torch.testing.assert_close(value, torch.arange(1, 5, dtype=torch.float32))
    with torch.no_grad():
        model.thinker.model.embed_tokens.weight[0, 0] += 1
    awq_smooth.main()
    assert len(collected) == 2
    assert legacy.read_bytes() == legacy_bytes
    assert data.read_bytes() == np.array([42], dtype=np.float32).tobytes()
    np.testing.assert_array_equal(_ort_result(output / "aux.onnx"), [42])
