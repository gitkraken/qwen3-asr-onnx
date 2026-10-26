import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]


def source(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(44)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"], name="projection1"),
            helper.make_node("MatMul", ["h", "w2"], ["y"], name="projection2"),
        ],
        "precision",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 32])],
        [
            numpy_helper.from_array(rng.normal(size=(64, 64)).astype(np.float32), "w1"),
            numpy_helper.from_array(rng.normal(size=(64, 32)).astype(np.float32), "w2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    for name in ["decoder_init", "decoder_step"]:
        onnx.save(model, inputs / f"{name}.onnx")
    (inputs / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "decoder": {"vocab_size": 4, "hidden_size": 64},
                "embed_tokens_shape": [4, 64],
                "embed_tokens_dtype": "float32",
            }
        )
    )
    calib = tmp_path / "calib.npz"
    np.savez(calib, _n_samples=40, **{f"{i}_x": rng.normal(size=(2, 64)).astype(np.float32) for i in range(40)})
    return inputs, calib


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_eight_bits_rejected_before_output(tmp_path, algo):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "8",
            "--block-size",
            "32",
            "--accuracy-level",
            "4",
            "--algo",
            algo,
            "--calib-data",
            str(calib),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert not output.exists(), result.stdout


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_four_bit_cli_actual_nodes_and_metadata(tmp_path, algo):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "32",
            "--accuracy-level",
            "4",
            "--algo",
            algo,
            "--calib-data",
            str(calib),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    for name in ["decoder_init", "decoder_step"]:
        model = onnx.load(output / f"{name}.int4.onnx")
        nodes = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
        assert len(nodes) == 2
        assert all({a.name: helper.get_attribute_value(a) for a in n.attribute}["bits"] == 4 for n in nodes)
        assert all({a.name: helper.get_attribute_value(a) for a in n.attribute}["block_size"] == 32 for n in nodes)
        session = ort.InferenceSession(str(output / f"{name}.int4.onnx"), providers=["CPUExecutionProvider"])
        got = session.run(None, {"x": np.ones((2, 64), dtype=np.float32)})[0]
        assert got.shape == (2, 32)
        assert np.isfinite(got).all()
    cfg = json.loads((output / "config.json").read_text())
    assert set(cfg["quantization"].values()) == {f"int4_{algo}_block32_al4"}


def cli(inputs, output, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "32",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_unsupported_request_preserves_existing_output(tmp_path, algo):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "config.json"
    sentinel.write_text("previous usable revision")
    result = cli(inputs, output, "--bits", "8", "--algo", algo)
    assert result.returncode != 0
    assert "only 4-bit" in result.stderr
    assert sentinel.read_text() == "previous usable revision"
    assert list(output.iterdir()) == [sentinel]


def test_rtn_excludes_one_node_without_lying_about_others(tmp_path):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--nodes-to-exclude", "projection1")
    assert result.returncode == 0, result.stderr
    model = onnx.load(output / "decoder_init.int4.onnx")
    assert any(n.name == "projection1" and n.op_type == "MatMul" for n in model.graph.node)
    assert len([n for n in model.graph.node if n.op_type == "MatMulNBits"]) == 1


def test_gptq_exclusions_are_rejected_explicitly(tmp_path):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--algo", "gptq", "--calib-data", str(calib), "--nodes-to-exclude", "projection1")
    assert result.returncode != 0
    assert "GPTQ node exclusions are unsupported" in result.stderr
    assert not output.exists()


def test_excluding_every_node_does_not_label_fp32_as_int4(tmp_path):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--nodes-to-exclude", "projection1", "projection2")
    assert result.returncode != 0
    assert "No MatMulNBits" in result.stderr
    assert not list(output.glob("*.int4.onnx"))
    assert not (output / "config.json").exists()


@pytest.mark.parametrize("block_size", ["0", "8", "24"])
def test_bad_block_size_fails_before_output(tmp_path, block_size):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--block-size", block_size)
    assert result.returncode != 0
    assert not output.exists()


def test_missing_decoder_fails_before_output(tmp_path):
    inputs, _ = source(tmp_path)
    (inputs / "decoder_step.onnx").unlink()
    output = tmp_path / "out"
    result = cli(inputs, output)
    assert result.returncode != 0
    assert not output.exists()


def test_verifies_every_effective_operator_before_saving(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import quantize_nbits

    inputs, _ = source(tmp_path)
    model = onnx.load(inputs / "decoder_init.onnx")
    del model.graph.node[:]
    model.graph.node.extend(
        [
            helper.make_node(
                "MatMulNBits", ["x", "w1"], ["h"], domain="com.microsoft", name="first", bits=4, block_size=32
            ),
            helper.make_node(
                "MatMulNBits", ["h", "w2"], ["y"], domain="com.microsoft", name="second", bits=8, block_size=32
            ),
        ]
    )
    saved = []

    class Quantizer:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(model=model, save_model_to_file=lambda *a, **k: saved.append(a))

        def process(self):
            pass

    monkeypatch.setattr(quantize_nbits, "MatMulNBitsQuantizer", Quantizer)
    with pytest.raises(ValueError, match="second"):
        quantize_nbits.quantize_decoder(
            str(inputs / "decoder_init.onnx"), str(tmp_path / "out.int4.onnx"), 4, 32, 4, "rtn", None
        )
    assert not saved
    assert not (tmp_path / "out.int4.onnx").exists()


def test_gptq_external_input_isolated_without_losing_unquantized_data(tmp_path):
    import quantize_nbits

    inputs, calib = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    model = onnx.load(path)
    model.graph.node[-1].output[0] = "linear"
    model.graph.initializer.append(numpy_helper.from_array(np.full((32,), 0.25, dtype=np.float32), "bias"))
    model.graph.node.append(helper.make_node("Add", ["linear", "bias"], ["y"], name="bias_add"))
    (inputs / "weights").mkdir()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights/decoder.data",
        size_threshold=0,
    )
    original = {p.relative_to(inputs).as_posix(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    x = np.ones((2, 64), dtype=np.float32)
    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    output = tmp_path / "quantized" / "decoder.int4.onnx"
    quantize_nbits.quantize_decoder(str(path), str(output), 4, 32, 4, "gptq", str(calib))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": x})[0]
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 0.3
    assert {p.relative_to(inputs).as_posix(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()} == original
    assert not any(
        "qwen-gptq" in e.value
        for t in onnx.load(output, load_external_data=False).graph.initializer
        for e in t.external_data
    )


def test_gptq_failure_preserves_input_config_and_external_files(tmp_path, monkeypatch):
    import quantize_nbits

    inputs, calib = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    onnx.save_model(
        onnx.load(path),
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )
    original = {p.name: p.read_bytes() for p in inputs.iterdir() if p.is_file()}
    staged = []

    class Quantizer:
        def __init__(self, **kwargs):
            model_path = Path(kwargs["model"])
            assert model_path != path
            assert not (model_path.parent / "config.json").exists()
            assert model_path.exists()
            staged.append(model_path.parent)

        def process(self):
            raise RuntimeError("quantizer fault")

    monkeypatch.setattr(quantize_nbits, "MatMulNBitsQuantizer", Quantizer)
    with pytest.raises(RuntimeError, match="quantizer fault"):
        quantize_nbits.quantize_decoder(str(path), str(tmp_path / "result.int4.onnx"), 4, 32, 4, "gptq", str(calib))
    assert staged and all(not p.exists() for p in staged)
    assert {p.name: p.read_bytes() for p in inputs.iterdir() if p.is_file()} == original
    assert not (tmp_path / "result.int4.onnx").exists()


def test_quantization_replaces_stale_output_template(tmp_path):
    from src.prompt import build_prompt_ids, load_prompt_tokenizer
    from tests.tokenizer_fixtures import save_tokenizer_assets

    inputs, _ = source(tmp_path)
    save_tokenizer_assets(inputs, template_format="embedded")
    output = tmp_path / "out"
    output.mkdir()
    (output / "chat_template.jinja").write_text("stale template")
    result = cli(inputs, output)
    assert result.returncode == 0, result.stderr + result.stdout
    assert not (output / "chat_template.jinja").exists()
    assert build_prompt_ids(2, load_prompt_tokenizer(str(output), local_files_only=True)) == build_prompt_ids(
        2, load_prompt_tokenizer(str(inputs), local_files_only=True)
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("name", ["encoder", "custom_projection"])
def test_quantize_safe_model_leaf_real_command(tmp_path, name, inplace):
    from tests.test_artifact_contract import _write_model

    source = tmp_path / "input"
    source.mkdir()
    _write_model(source, name + ".onnx", weight=np.arange(2048, dtype=np.float32).reshape(128, 16) / 2048)
    (source / "config.json").write_text("{}")
    _write_model(source, name + ".int4.onnx")
    if inplace:
        (source / (name + ".int4.onnx")).rename(source / (name + ".int4.onnx.bak"))
    output = source if inplace else tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "64",
            "--accuracy-level",
            "4",
            "--decoders",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    graph = onnx.load(output / (name + ".int4.onnx"))
    nodes = [n for n in graph.graph.node if n.op_type == "MatMulNBits"]
    assert nodes
    assert json.loads((output / "config.json").read_text())["quantization"][name] == "int4_rtn_block64_al4"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside/encoder", "/tmp/encoder", "sub/encoder", "sub\\encoder", "C:encoder"]
)
def test_unsafe_model_name_is_rejected_before_output(tmp_path, name):
    inputs, _ = source(tmp_path)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "64",
            "--accuracy-level",
            "4",
            "--decoders",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Unsafe model name" in result.stderr
    assert not output.exists()
