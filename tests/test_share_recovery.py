import builtins
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import share_weights as share
from tests.test_artifact_contract import _ort_result, _write_model
from tests.test_native_decoder import (
    assert_decoder_matches,
    prefill_inputs,
)
from tests.test_native_decoder import decoder_exports as decoder_exports
from tests.test_native_decoder import decoder_threads as decoder_threads
from tests.test_native_decoder import text_model as text_model


def _pair(root, inline=False):
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(root, name, external=not inline)
    return {name: _ort_result(root / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}


def _assert_pair(root, expected):
    for name, values in expected.items():
        np.testing.assert_array_equal(_ort_result(root / name), values)


def _assert_clean(root):
    assert {path.name for path in root.iterdir()} == {"decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"}


def test_interruption_after_data_publication_is_recoverable(tmp_path, monkeypatch):
    expected = _pair(tmp_path)
    replace, rename = share.os.replace, share.os.rename

    def interrupt(operation):
        def run(source, target, *args, **kwargs):
            result = operation(source, target, *args, **kwargs)
            if Path(target).name == "decoder_weights.data":
                raise OSError("injected interruption after publishing data")
            return result

        return run

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "replace", interrupt(replace))
        patch.setattr(share.os, "rename", interrupt(rename))
        with pytest.raises(OSError, match="injected"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_clean(tmp_path)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("graph_name", ["decoder_init.onnx", "decoder_step.onnx"])
def test_partial_proto_write_keeps_original_pair_usable(tmp_path, monkeypatch, inline, graph_name):
    expected = _pair(tmp_path, inline=inline)
    original_open = builtins.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.stream, key)

        def write(self, data):
            self.stream.write(data[: len(data) // 2])
            raise OSError("injected partial proto write")

    def injected(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if Path(path).name == graph_name and mode == "wb":
            return PartialWrite(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", injected)
        with pytest.raises(OSError, match="partial proto"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_clean(tmp_path)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("graph_index", [0, 1])
def test_interruption_after_graph_replace_can_retry(tmp_path, monkeypatch, inline, graph_index):
    expected = _pair(tmp_path, inline=inline)
    original_replace = share.os.replace
    graph_name = ("decoder_init.onnx", "decoder_step.onnx")[graph_index]

    def injected(source, target):
        original_replace(source, target)
        if Path(target).name == graph_name:
            raise OSError("injected interruption after graph replace")

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "replace", injected)
        with pytest.raises(OSError, match="after graph"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_clean(tmp_path)


def test_cleanup_failure_keeps_models_usable_and_retry_cleans(tmp_path, monkeypatch):
    expected = _pair(tmp_path)
    original_unlink = share.os.unlink

    def injected(path, *args, **kwargs):
        if str(path).endswith("decoder_step.onnx.data"):
            raise OSError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "unlink", injected)
        with pytest.raises(OSError, match="cleanup"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_clean(tmp_path)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_sharing_preserves_unmatched_tensors_and_storage_contract(tmp_path, dtype):
    init_weight = np.arange(512, dtype=dtype).reshape(32, 16)
    step_weight = init_weight + 2
    _write_model(tmp_path, "decoder_init.onnx", external=True, weight=init_weight)
    _write_model(tmp_path, "decoder_step.onnx", external=True, weight=step_weight)
    expected = {name: _ort_result(tmp_path / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}
    assert share.share_external_models(tmp_path)
    for name, weight in (("decoder_init.onnx", init_weight), ("decoder_step.onnx", step_weight)):
        tensor = onnx.load(tmp_path / name).graph.initializer[0]
        actual = numpy_helper.to_array(tensor)
        assert actual.dtype == weight.dtype
        assert actual.shape == weight.shape
        assert actual.tobytes() == weight.tobytes()
    _assert_pair(tmp_path, expected)
    _assert_clean(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert share.share_external_models(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_shared_existence_does_not_hide_invalid_references(tmp_path):
    _pair(tmp_path)
    (tmp_path / "decoder_weights.data").write_bytes(b"unrelated")
    (tmp_path / "decoder_step.onnx.data").unlink()
    with pytest.raises((ValueError, onnx.checker.ValidationError, FileNotFoundError)):
        share.share_external_models(tmp_path)


def test_live_incompatible_canonical_data_is_not_overwritten(tmp_path):
    _pair(tmp_path)
    assert share.share_external_models(tmp_path)
    canonical = tmp_path / "decoder_weights.data"
    original = canonical.read_bytes()
    _write_model(
        tmp_path,
        "decoder_step.onnx",
        external=True,
        data_name="replacement.bin",
        weight=np.ones((32, 16), dtype=np.float32),
    )
    expected = {name: _ort_result(tmp_path / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}
    with pytest.raises(ValueError, match="rebuild"):
        share.share_external_models(tmp_path)
    assert canonical.read_bytes() == original
    _assert_pair(tmp_path, expected)


@pytest.mark.parametrize("data_name", ["embed_tokens.bin", "weights.bin", "sub/weights.data"])
def test_sharing_retains_independently_owned_dependencies(tmp_path, data_name):
    (tmp_path / data_name).parent.mkdir(parents=True, exist_ok=True)
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(tmp_path, name, external=True, data_name=data_name)
    expected = {name: _ort_result(tmp_path / name) for name in share._decoder_filenames()}
    data = tmp_path / data_name
    original = data.read_bytes()
    auxiliary = None
    if data_name.startswith("sub/"):
        auxiliary = data.parent / "aux.onnx"
        proto = onnx.load(tmp_path / "decoder_init.onnx", load_external_data=False)
        for entry in proto.graph.initializer[0].external_data:
            if entry.key == "location":
                entry.value = data.name
        onnx.save(proto, auxiliary)
        auxiliary_before = _ort_result(auxiliary)
    if data_name == "embed_tokens.bin":
        import json

        from src.embeddings import load_embed_tokens

        config = {"decoder": {"vocab_size": 64, "hidden_size": 16}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        embeddings_before = load_embed_tokens(tmp_path)
    assert share.share_external_models(tmp_path)
    assert data.read_bytes() == original
    _assert_pair(tmp_path, expected)
    if auxiliary is not None:
        np.testing.assert_array_equal(_ort_result(auxiliary), auxiliary_before)
    if data_name == "embed_tokens.bin":
        np.testing.assert_array_equal(load_embed_tokens(tmp_path), embeddings_before)


def _run_model(path, feed):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, feed)


def test_shared_native_decoder_preserves_prefill_and_step(tmp_path, decoder_exports, text_model):
    for path in decoder_exports["init"].parent.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    with torch.no_grad():
        _, prefill_feed = prefill_inputs(text_model, 1, 7, 1, 2)
        before = _run_model(tmp_path / "decoder_init.onnx", prefill_feed)
        token = torch.tensor([[3]], dtype=torch.int64)
        step_feed = {
            "input_embeds": text_model.thinker.model.embed_tokens(token).numpy(),
            "position_ids": np.array([[7]], dtype=np.int64),
            "past_keys": before[1],
            "past_values": before[2],
        }
        before_step = _run_model(tmp_path / "decoder_step.onnx", step_feed)
    assert share.share_external_models(tmp_path)
    assert_decoder_matches(_run_model(tmp_path / "decoder_init.onnx", prefill_feed), before)
    assert_decoder_matches(_run_model(tmp_path / "decoder_step.onnx", step_feed), before_step)
    _assert_clean(tmp_path)


@pytest.mark.parametrize("kind", ["Constant", "ConstantOfShape", "Split"])
def test_shape_and_attribute_constants_stay_executable(tmp_path, kind):
    shape = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="shape")
    if kind == "Constant":
        nodes = [
            helper.make_node("Constant", [], ["shape"], value=shape),
            helper.make_node("Reshape", ["x", "shape"], ["y"]),
        ]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])]
        initializers = []
        output_shape = [2, 2]
        feed = {"x": np.arange(4, dtype=np.float32)}
    elif kind == "ConstantOfShape":
        nodes = [
            helper.make_node(
                "ConstantOfShape", ["shape"], ["y"], value=numpy_helper.from_array(np.array([1.0], dtype=np.float32))
            )
        ]
        inputs = []
        initializers = [shape]
        output_shape = [2, 2]
        feed = {}
    else:
        initializers = [numpy_helper.from_array(np.ones(129, dtype=np.int64), name="splits")]
        outputs = ["y", *[f"unused{i}" for i in range(128)]]
        nodes = [helper.make_node("Split", ["x", "splits"], outputs, axis=0)]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [129])]
        output_shape = [1]
        feed = {"x": np.arange(129, dtype=np.float32)}
    graph = helper.make_graph(
        nodes, kind, inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    for name in share._decoder_filenames():
        onnx.save(model, tmp_path / name)
    before = _run_model(tmp_path / "decoder_init.onnx", feed)
    assert share.share_external_models(tmp_path)
    for name in share._decoder_filenames():
        np.testing.assert_array_equal(_run_model(tmp_path / name, feed)[0], before[0])
        proto = onnx.load(tmp_path / name, load_external_data=False)
        assert all(t.data_location != TensorProto.EXTERNAL for t in share.model_tensors(proto))


@pytest.mark.parametrize("inline", [False, True])
def test_staged_runtime_failure_preserves_source_pair(tmp_path, monkeypatch, inline):
    expected = _pair(tmp_path, inline=inline)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original = ort.InferenceSession

    def fail_staged(path, *args, **kwargs):
        if Path(path).parent.name.startswith(".share-"):
            raise RuntimeError("injected staged runtime failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ort, "InferenceSession", fail_staged)
        with pytest.raises(RuntimeError, match="staged runtime"):
            share.share_external_models(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    _assert_pair(tmp_path, expected)


@pytest.mark.parametrize("dtype,suffix", [(np.float32, "q4"), (np.float16, "q4f16")])
def test_shared_quantized_variants_pass_real_cpu_sessions(tmp_path, dtype, suffix):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    weight = numpy_helper.from_array(np.arange(64 * 32, dtype=dtype).reshape(64, 32) / 2048, name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="matmul")],
        "quantized",
        [helper.make_tensor_value_info("x", tensor_type, [1, 64])],
        [helper.make_tensor_value_info("y", tensor_type, [1, 32])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=False, accuracy_level=4)
    quantizer.process()
    for name in share._decoder_filenames(suffix):
        onnx.save(quantizer.model.model, tmp_path / name)
    feed = {"x": np.ones((1, 64), dtype=dtype)}
    before = _run_model(tmp_path / share._decoder_filenames(suffix)[0], feed)
    assert share.share_external_models(tmp_path, suffix=suffix)
    for name in share._decoder_filenames(suffix):
        np.testing.assert_array_equal(_run_model(tmp_path / name, feed)[0], before[0])


@pytest.mark.parametrize("inline", [False, True])
def test_conflicting_canonical_data_is_preserved(tmp_path, inline):
    expected = _pair(tmp_path, inline=inline)
    path = tmp_path / "decoder_weights.data"
    path.write_bytes(b"independently retained canonical bytes")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="incompatible"):
        share.share_external_models(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    _assert_pair(tmp_path, expected)


def test_identical_canonical_symlink_rejected_before_graph_publication(tmp_path):
    expected = _pair(tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    _pair(reference)
    assert share.share_external_models(reference)
    retained = tmp_path / "retained.bin"
    retained.write_bytes((reference / "decoder_weights.data").read_bytes())
    shutil.rmtree(reference)
    (tmp_path / "decoder_weights.data").symlink_to(retained)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        share.share_external_models(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    _assert_pair(tmp_path, expected)
