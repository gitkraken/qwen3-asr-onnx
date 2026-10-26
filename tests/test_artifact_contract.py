import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

import package as pack


def _write_model(root, name, *, external=False, data_name=None, weight=None):
    weight = np.arange(512, dtype=np.float32).reshape(32, 16) if weight is None else weight
    tensor = numpy_helper.from_array(weight, name="weight")
    dtype = helper.np_dtype_to_tensor_dtype(weight.dtype)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", dtype, [1, weight.shape[0]])],
        [helper.make_tensor_value_info("y", dtype, [1, weight.shape[1]])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save_model(
        model,
        root / name,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=data_name or f"{name}.data",
        size_threshold=0,
    )


def _release_input(root, dtype="float32", *, external=False):
    root.mkdir()
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        _write_model(root, name, external=external)
    config = {
        "decoder": {"vocab_size": 8, "hidden_size": 4},
        "embed_tokens_shape": [8, 4],
        "embed_tokens_dtype": dtype,
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (root / name).write_text("{}")
    (root / "config.json").write_text(json.dumps(config))
    np.arange(32, dtype=dtype).reshape(8, 4).tofile(root / "embed_tokens.bin")
    return root


def _ort_result(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    dtype = np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32
    return session.run(None, {"x": np.ones((1, 32), dtype=dtype)})[0]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_packaging_preserves_embedding_values_and_can_repeat(tmp_path, dtype):
    source = _release_input(tmp_path / "source", dtype)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    values = np.fromfile(output / "embed_tokens.bin", dtype=np.float16).reshape(8, 4)
    np.testing.assert_array_equal(values, np.arange(32).reshape(8, 4))
    assert pack.package(str(output), str(tmp_path / "repeated"))
    assert (tmp_path / "repeated" / "embed_tokens.bin").read_bytes() == (output / "embed_tokens.bin").read_bytes()


@pytest.mark.parametrize("missing", ["decoder_step.onnx", "encoder.onnx", "tokenizer.json", "embed_tokens.bin"])
def test_incomplete_release_preserves_existing_output(tmp_path, missing):
    source = _release_input(tmp_path / "source")
    (source / missing).unlink()
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_bytes(b"previous release")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_bytes() == b"previous release"


def test_partial_int4_variant_is_rejected(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "decoder_init.int4.onnx")
    assert not pack.package(str(source), str(tmp_path / "release"))


@pytest.mark.parametrize("layout", ["inline", "per-decoder", "shared", "custom"])
def test_archives_include_actual_graph_dependencies(tmp_path, layout):
    source = _release_input(tmp_path / "source", external=layout == "per-decoder")
    if layout == "shared":
        from share_weights import share_external_models

        share_external_models(source)
    if layout == "custom":
        for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, external=True, data_name=f"custom-{name}.bin")
    for name in ("encoder.int4.onnx", "decoder_init.int4.onnx", "decoder_step.int4.onnx"):
        _write_model(source, name, external=True)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    for suffix in ("", "-int4"):
        archive_path = tmp_path / f"release{suffix}.tar.gz"
        extracted = tmp_path / f"extracted{suffix}"
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted, filter="data")
        for model in extracted.rglob("*.onnx"):
            np.testing.assert_array_equal(_ort_result(model), _ort_result(source / model.name))
        assert (extracted / f"release{suffix}" / "embed_tokens.bin").stat().st_size == 64
    assert not list(output.glob("decoder_*.onnx.data"))
    assert not list(output.glob("custom-decoder*"))


def test_truncated_external_dependency_preserves_output(tmp_path):
    source = _release_input(tmp_path / "source", external=True)
    (source / "encoder.onnx.data").write_bytes(b"short")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"


def test_archive_failure_preserves_previous_release(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"old archive")

    def fail(*args, **kwargs):
        raise OSError("injected archive failure")

    monkeypatch.setattr(tarfile, "open", fail)
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"
    assert archive.read_bytes() == b"old archive"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_converter_cli_honors_declared_storage_dtype(tmp_path, dtype):
    source = _release_input(tmp_path / "source", dtype)
    result = subprocess.run(
        [sys.executable, str(Path(pack.__file__).with_name("convert_embed_fp16.py")), "--model-dir", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = np.fromfile(source / "embed_tokens.fp16.bin", dtype=np.float16)
    np.testing.assert_array_equal(values, np.arange(32, dtype=np.float16))


@pytest.mark.parametrize("problem", ["shape", "dtype", "truncated", "nan", "overflow"])
def test_bad_embeddings_are_rejected_without_replacing_output(tmp_path, problem):
    source = _release_input(tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    if problem == "shape":
        config["embed_tokens_shape"] = [4, 8]
    elif problem == "dtype":
        config["embed_tokens_dtype"] = "bfloat16"
    elif problem == "truncated":
        (source / "embed_tokens.bin").write_bytes(b"short")
    else:
        np.full((8, 4), np.nan if problem == "nan" else 1e20, dtype=np.float32).tofile(source / "embed_tokens.bin")
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"


def test_legacy_metadata_requires_unambiguous_fp32_bytes(tmp_path):
    from src.embeddings import load_embed_tokens

    source = _release_input(tmp_path / "source")
    (source / "config.json").write_text(json.dumps({"decoder": {"vocab_size": 8, "hidden_size": 4}}))
    np.testing.assert_array_equal(load_embed_tokens(source), np.arange(32).reshape(8, 4))
    np.arange(32, dtype=np.float16).tofile(source / "embed_tokens.bin")
    with pytest.raises(ValueError, match="byte count"):
        load_embed_tokens(source)


def test_publication_failure_restores_release_and_archives(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    previous_archive = tmp_path / "release.tar.gz"
    previous_archive.write_bytes(b"old archive")
    original_replace = pack.os.replace

    def injected(source, target):
        if Path(target) == previous_archive and ".package-" in str(source):
            raise OSError("injected archive publication failure")
        original_replace(source, target)

    monkeypatch.setattr(pack.os, "replace", injected)
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"
    assert previous_archive.read_bytes() == b"old archive"


def test_converter_rejects_overflow_and_preserves_existing_sidecar(tmp_path):
    source = _release_input(tmp_path / "source")
    np.full((8, 4), 1e20, dtype=np.float32).tofile(source / "embed_tokens.bin")
    destination = source / "embed_tokens.fp16.bin"
    destination.write_bytes(b"previous validated conversion")
    result = subprocess.run(
        [sys.executable, str(Path(pack.__file__).with_name("convert_embed_fp16.py")), "--model-dir", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert destination.read_bytes() == b"previous validated conversion"


def test_hardlinked_metadata_does_not_mutate_source(tmp_path):
    source = _release_input(tmp_path / "source")
    before = (source / "config.json").read_bytes()
    assert pack.package(str(source), str(tmp_path / "release"), hardlink=True)
    assert (source / "config.json").read_bytes() == before
    assert json.loads((tmp_path / "release" / "config.json").read_text())["embed_tokens_dtype"] == "float16"


def test_complete_int4_only_release_can_be_repackaged(tmp_path):
    source = _release_input(tmp_path / "source", "float16")
    for role in ("encoder", "decoder_init", "decoder_step"):
        (source / f"{role}.onnx").unlink()
        _write_model(source, f"{role}.int4.onnx", external=True)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    assert not (tmp_path / "release.tar.gz").exists()
    with tarfile.open(tmp_path / "release-int4.tar.gz") as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    extracted = tmp_path / "extracted" / "release-int4"
    assert pack.package(str(extracted), str(tmp_path / "repeated"))
    for model in (tmp_path / "repeated").glob("*.onnx"):
        np.testing.assert_array_equal(_ort_result(model), _ort_result(source / model.name))
    assert (tmp_path / "repeated" / "embed_tokens.bin").read_bytes() == (source / "embed_tokens.bin").read_bytes()


@pytest.mark.parametrize("name", ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"])
@pytest.mark.parametrize("contents", ['{"truncated":', "[]"])
def test_required_metadata_must_be_valid_json_objects(tmp_path, name, contents):
    source = _release_input(tmp_path / "source")
    (source / name).write_text(contents)
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("previous release")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "previous release"
    assert not (tmp_path / "release.tar.gz").exists()


@pytest.mark.parametrize("template_file", ["chat_template.jinja", "chat_template.json"])
def test_packaged_chat_template_reloads_offline(tmp_path, template_file, monkeypatch):
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from src.prompt import load_prompt_tokenizer
    from tests.test_prompt_contract import OFFICIAL_CHAT_TEMPLATE

    source = _release_input(tmp_path / "source")
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = OFFICIAL_CHAT_TEMPLATE
    tokenizer.save_pretrained(source)
    if template_file == "chat_template.json":
        (source / "chat_template.jinja").unlink()
        (source / template_file).write_text(json.dumps({"chat_template": OFFICIAL_CHAT_TEMPLATE}))
    template_bytes = (source / template_file).read_bytes()
    messages = [
        {"role": "system", "content": "Kepler"},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    expected = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    with tarfile.open(tmp_path / "release.tar.gz") as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    packaged = tmp_path / "extracted" / "release"
    assert (packaged / template_file).read_bytes() == template_bytes
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    offline = AutoTokenizer.from_pretrained(packaged, local_files_only=True)
    if not offline.chat_template:
        offline = load_prompt_tokenizer(str(packaged))
    assert offline.chat_template == OFFICIAL_CHAT_TEMPLATE
    assert offline.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) == expected
