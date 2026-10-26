import builtins
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

import share_weights as share
from tests.test_artifact_contract import _ort_result, _write_model


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
