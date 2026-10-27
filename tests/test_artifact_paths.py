import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import quantize as dynamic
import quantize_static as static
from src import onnx_artifacts
from tests.test_artifact_contract import _ort_result, _write_external_identity, _write_model
from tests.test_artifact_mutations import SCRIPTS, bytes_under, command, input_for


@pytest.mark.parametrize("script", SCRIPTS)
def test_internal_directory_alias_cannot_redirect_dependency_copy(tmp_path, script):
    source = input_for(tmp_path, script)
    (source / "alias").mkdir()
    np.array([99], dtype=np.float32).tofile(source / "alias/weights.data")
    _write_external_identity(source, "encoder.onnx", "alias/weights.data")
    target = tmp_path / "target"
    (target / "nested").mkdir(parents=True)
    np.array([42], dtype=np.float32).tofile(target / "nested/weights.data")
    (target / "alias").symlink_to("nested", target_is_directory=True)
    _write_external_identity(target, "aux.onnx", "nested/weights.data")
    before = bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "linked parent" in result.stdout + result.stderr
    assert bytes_under(target) == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


@pytest.mark.parametrize("operation", ["copy", "json", "generated"])
def test_direct_writers_refuse_internal_linked_parents(tmp_path, operation):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "alias").mkdir(parents=True)
    (source / "alias/value").write_bytes(b"incoming")
    (target / "nested").mkdir(parents=True)
    (target / "nested/value").write_bytes(b"retained")
    (target / "alias").symlink_to("nested", target_is_directory=True)
    with pytest.raises(ValueError, match="linked parent"):
        if operation == "copy":
            onnx_artifacts.copy_artifact_entry(source, target, "alias/value")
        elif operation == "json":
            onnx_artifacts.write_artifact_json(target, "alias/value", {})
        else:
            onnx_artifacts.validate_generated_destinations(target, ["alias/value"])
    assert (target / "nested/value").read_bytes() == b"retained"


@pytest.mark.parametrize("script", SCRIPTS)
def test_nested_output_cannot_overwrite_selected_source_storage(tmp_path, script):
    source = input_for(tmp_path, script)
    target = source / "out"
    target.mkdir()
    stem = "encoder" if script == "quantize.py" else "decoder_init"
    generated = f"{stem}.int8.onnx.data" if script == "quantize.py" else f"{stem}.onnx"
    if script == "quantize_nbits.py":
        generated = f"{stem}.int4.onnx"
    weight = np.ones((64, 64) if script == "quantize_nbits.py" else (4, 3), dtype=np.float32)
    _write_model(source, f"{stem}.onnx", external=True, data_name=f"out/{generated}", weight=weight)
    before = bytes_under(source)
    expected = _ort_result(source / f"{stem}.onnx")
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "source graph destination conflict" in result.stdout + result.stderr
    assert bytes_under(source) == before
    np.testing.assert_array_equal(_ort_result(source / f"{stem}.onnx"), expected)


@pytest.mark.parametrize("script", SCRIPTS)
def test_noncolliding_nested_output_can_be_rebuilt(tmp_path, script):
    source = input_for(tmp_path, script)
    target = source / "out"
    before = bytes_under(source)
    args = command(tmp_path, script, source, target)
    for _ in range(2):
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert all((source / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize("kind", ["dynamic", "static"])
@pytest.mark.parametrize("storage_alias", [False, True])
def test_ort_inference_never_writes_in_source_namespace(tmp_path, kind, storage_alias):
    source = input_for(tmp_path, "quantize.py")
    stem = "encoder" if kind == "dynamic" else "decoder_init"
    inferred = f"{stem}-inferred.onnx"
    _write_model(
        source,
        f"{stem}.onnx",
        external=storage_alias,
        data_name=inferred if storage_alias else None,
        weight=np.ones((4, 3), dtype=np.float32),
    )
    if not storage_alias:
        _write_model(source, inferred, weight=np.ones((4, 3), dtype=np.float32))
    before = bytes_under(source)
    expected = _ort_result(source / f"{stem}.onnx")
    output = tmp_path / "quantized.onnx"
    if kind == "dynamic":
        dynamic.quantize_onnx_file(str(source / f"{stem}.onnx"), str(output))
    else:
        static.quantize_decoder(str(source / f"{stem}.onnx"), str(output), [{"x": np.ones((1, 4), dtype=np.float32)}])
    assert bytes_under(source) == before
    np.testing.assert_array_equal(_ort_result(source / f"{stem}.onnx"), expected)
    np.testing.assert_allclose(_ort_result(output), expected, atol=1e-4, rtol=1e-4)


def test_dynamic_retry_keeps_both_attempts_in_owned_input_stage(tmp_path, monkeypatch):
    source = input_for(tmp_path, "quantize.py")
    _write_model(
        source,
        "encoder.onnx",
        external=True,
        data_name="encoder-inferred.onnx",
        weight=np.ones((4, 3), dtype=np.float32),
    )
    before = bytes_under(source)
    expected = _ort_result(source / "encoder.onnx")
    quantize = dynamic.quantize_dynamic
    seen = []

    def fail_once(input_path, *args, **kwargs):
        seen.append(Path(input_path))
        if len(seen) == 1:
            raise ValueError("injected direct quantization failure")
        return quantize(input_path, *args, **kwargs)

    monkeypatch.setattr(dynamic, "quantize_dynamic", fail_once)
    output = tmp_path / "quantized.onnx"
    dynamic.quantize_onnx_file(str(source / "encoder.onnx"), str(output))
    assert len(seen) == 2
    assert all(path.parent.name.startswith("qwen-onnx-") for path in seen)
    assert all(not path.parent.exists() for path in seen)
    assert bytes_under(source) == before
    np.testing.assert_allclose(_ort_result(output), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("retained", [False, True])
def test_dynamic_metadata_only_labels_generated_graphs(tmp_path, retained):
    source = input_for(tmp_path, "quantize.py")
    for name in ("encoder.onnx", "decoder_step.onnx"):
        (source / name).unlink()
    target = tmp_path / "target"
    target.mkdir()
    previous = {}
    if retained:
        for stem in ("encoder", "decoder_step"):
            _write_model(target, f"{stem}.int8.onnx", external=True, weight=np.ones((4, 3), dtype=np.float32))
        _write_external_identity(target, "aux.onnx", "encoder.int8.onnx.data")
        config = json.loads((source / "config.json").read_text())
        config["quantization"] = {"encoder": "int8_static", "decoder_step": "int8_static"}
        (target / "config.json").write_text(json.dumps(config))
        previous = {name: value for name, value in bytes_under(target).items() if name != Path("config.json")}
    result = subprocess.run(
        command(tmp_path, "quantize.py", source, target), capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((target / name).read_bytes() == value for name, value in previous.items())
    expected = {"decoder_init": "int8_dynamic"}
    if retained:
        expected.update(encoder="int8_static", decoder_step="int8_static")
        np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [1])
    assert json.loads((target / "config.json").read_text())["quantization"] == expected


def test_dynamic_late_asset_failure_keeps_retained_provenance(tmp_path, monkeypatch):
    source = input_for(tmp_path, "quantize.py")
    for name in ("encoder.onnx", "decoder_step.onnx"):
        (source / name).unlink()
    target = tmp_path / "target"
    target.mkdir()
    _write_model(target, "decoder_step.int8.onnx", weight=np.ones((4, 3), dtype=np.float32))
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"decoder_step": "int8_static"}
    (target / "config.json").write_text(json.dumps(config))
    previous = (target / "config.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])
    with monkeypatch.context() as patch:

        def fail_assets(*_args):
            raise ValueError("injected late asset failure")

        patch.setattr(dynamic, "copy_tokenizer_assets", fail_assets)
        with pytest.raises(ValueError, match="late asset failure"):
            dynamic.main()
    assert (target / "config.json").read_bytes() == previous
    dynamic.main()
    assert json.loads((target / "config.json").read_text())["quantization"] == {
        "decoder_init": "int8_dynamic",
        "decoder_step": "int8_static",
    }
