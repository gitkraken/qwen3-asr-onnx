import json
import sys
import tarfile

import numpy as np
import pytest

import create_tar
from src.release_validation import graph_names, validate_release
from tests.test_artifact_contract import _release_input, _write_model


def _archive(source):
    return {"name": "model", "source_dir": str(source), "repo": "unused/local", "files": graph_names()}


@pytest.mark.parametrize(
    "problem",
    [
        "short-embedding",
        "nonfinite-embedding",
        "config-shape",
        "missing-preprocessor",
        "bad-preprocessor",
        "incomplete-decoder",
        "invalid-graph",
        "runtime-graph",
    ],
)
def test_invalid_release_cannot_replace_archive_or_upload(tmp_path, monkeypatch, problem):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    previous = output / "model.tar.gz"
    previous.write_bytes(b"previous archive")
    if problem == "short-embedding":
        (source / "embed_tokens.bin").write_bytes(b"x")
    elif problem == "nonfinite-embedding":
        values = np.arange(32, dtype=np.float32)
        values[3] = np.nan
        values.tofile(source / "embed_tokens.bin")
    elif problem == "config-shape":
        config = json.loads((source / "config.json").read_text())
        config["embed_tokens_shape"] = [4, 8]
        (source / "config.json").write_text(json.dumps(config))
    elif problem == "missing-preprocessor":
        (source / "preprocessor_config.json").unlink()
    elif problem == "bad-preprocessor":
        (source / "preprocessor_config.json").write_text("[]")
    elif problem == "incomplete-decoder":
        (source / "decoder_step.onnx").unlink()
    elif problem == "invalid-graph":
        (source / "decoder_step.onnx").write_bytes(b"invalid ONNX")
    else:
        import onnx

        graph = onnx.load(source / "decoder_step.onnx")
        graph.graph.node[0].domain = "unavailable.backend"
        graph.opset_import.add(domain="unavailable.backend", version=1)
        onnx.save(graph, source / "decoder_step.onnx")
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    archive = _archive(source)
    assert create_tar.create_tar(archive, output) is None
    monkeypatch.setattr(create_tar, "ARCHIVES", [archive])
    uploads = []
    monkeypatch.setattr(create_tar, "upload_tar", lambda *args: uploads.append(args))
    monkeypatch.setattr(sys, "argv", ["create_tar.py", "--output-dir", str(output)])
    assert create_tar.main() == 1
    assert uploads == []
    assert previous.read_bytes() == b"previous archive"
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("layout", ["inline", "separate", "shared"])
def test_archive_uses_actual_graph_dependencies(tmp_path, layout):
    source = _release_input(tmp_path / "source")
    for name in graph_names():
        _write_model(source, name, external=layout != "inline", data_name="shared.data" if layout == "shared" else None)
    output = tmp_path / "archives"
    output.mkdir()
    path = create_tar.create_tar(_archive(source), output)
    assert path is not None
    with tarfile.open(path) as archive:
        actual = {member.name.removeprefix("model/") for member in archive.getmembers()}
    expected = set(graph_names()) | {
        "config.json",
        "preprocessor_config.json",
        "embed_tokens.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    }
    if layout == "shared":
        expected.add("shared.data")
    elif layout == "separate":
        expected.update(f"{name}.data" for name in graph_names())
    assert actual == expected
    assert actual == validate_release(source, graph_names())


def test_archive_refuses_incomplete_selected_graph_family(tmp_path):
    source = _release_input(tmp_path / "source")
    with pytest.raises(ValueError, match="complete encoder"):
        validate_release(source, ["encoder.onnx", "decoder_init.onnx"])


def test_archive_destination_cannot_replace_graph_storage(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "encoder.onnx", external=True, data_name="model.tar.gz")
    original = (source / "model.tar.gz").read_bytes()
    assert create_tar.create_tar(_archive(source), source) is None
    assert (source / "model.tar.gz").read_bytes() == original
    validate_release(source, graph_names())
