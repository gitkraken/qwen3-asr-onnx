import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from src.prompt import build_prompt_ids, get_audio_pad_range
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    copy_tokenizer_assets,
    tokenizer_asset_names,
    validate_tokenizer_assets,
)
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE, save_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]


def _source(root, template_format):
    save_tokenizer_assets(root, template_format=template_format)
    config = {
        "decoder": {"vocab_size": 8, "hidden_size": 4},
        "embed_tokens_shape": [8, 4],
        "embed_tokens_dtype": "float32",
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text("{}")
    np.arange(32, dtype=np.float32).tofile(root / "embed_tokens.bin")
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 16
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            "toy",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(weight, "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save_model(model, root / name)
    return root


def _assert_source_assets(source, target):
    assert tokenizer_asset_names(target) == tokenizer_asset_names(source)
    for name in tokenizer_asset_names(source):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    tokenizer = validate_tokenizer_assets(target)
    ids = build_prompt_ids(3, tokenizer, context="Kepler", language="English")
    start, end = get_audio_pad_range(ids, expected_count=3)
    assert end - start == 3
    assert json.loads((target / "tokenizer_config.json").read_text())["origin_marker"] == "tiny-local-fixture"


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_asset_copy_replaces_stale_templates_and_preserves_origin(tmp_path, template_format):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, template_format=template_format)
    save_tokenizer_assets(target, template_format="jinja")
    (target / "chat_template.jinja").write_text("obsolete template")
    (target / "chat_template.json").write_text("{}")
    copy_tokenizer_assets(source, target)
    _assert_source_assets(source, target)


@pytest.mark.parametrize("problem", ["missing-template", "bad-template", "missing-tokenizer", "invalid-json"])
def test_invalid_source_is_rejected_before_target_changes(tmp_path, problem):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, include_template=problem != "missing-template")
    if problem == "bad-template":
        (source / "chat_template.jinja").write_text(OFFICIAL_CHAT_TEMPLATE.replace("system\\n", "broken\\n"))
    if problem == "missing-tokenizer":
        (source / "tokenizer.json").unlink()
    if problem == "invalid-json":
        (source / "tokenizer.json").write_text("not-json")
    target.mkdir()
    (target / "chat_template.jinja").write_text("previous")
    with pytest.raises(ValueError):
        copy_tokenizer_assets(source, target)
    assert (target / "chat_template.jinja").read_text() == "previous"


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_validation_is_offline(tmp_path, template_format, monkeypatch):
    import transformers.utils.hub

    source = tmp_path / "source"
    save_tokenizer_assets(source, template_format=template_format)

    def no_download(*args, **kwargs):
        raise AssertionError("Tokenizer validation must not download replacement assets")

    monkeypatch.setattr(transformers.utils.hub, "hf_hub_download", no_download)
    validate_tokenizer_assets(source)


def _converter_command(tmp_path, script, source, target):
    command = [
        sys.executable,
        str(ROOT / script),
        "--input",
        str(source),
        "--output",
        str(target),
        "--no-share-weights",
    ]
    if script == "quantize_static.py":
        cache = tmp_path / "cache"
        cache.mkdir()
        inputs = np.array([{"x": np.ones((1, 4), dtype=np.float32)}], dtype=object)
        np.savez(cache / "calib_n1_s1.npz", init_inputs=inputs, step_inputs=inputs)
        command += ["--cache-dir", str(cache), "--n-samples", "1", "--n-steps", "1"]
    return command


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py"])
def test_converter_cli_transports_complete_tokenizer(tmp_path, script, template_format):
    source = _source(tmp_path / "source", template_format)
    target = tmp_path / "target"
    save_tokenizer_assets(target, template_format="jinja")
    (target / "chat_template.jinja").write_text("obsolete template")
    command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_source_assets(source, target)
    for name in ("encoder.onnx", "embed_tokens.bin"):
        assert (target / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_archive_retains_complete_renderable_tokenizer(tmp_path, template_format):
    from create_tar import create_tar

    source = _source(tmp_path / "source", template_format)
    names = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]
    archive_path = create_tar({"name": "bundle", "source_dir": str(source), "files": names}, str(tmp_path))
    assert archive_path is not None
    extracted = tmp_path / "extracted"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted, filter="data")
    target = extracted / "bundle"
    _assert_source_assets(source, target)
    for name in names + ["embed_tokens.bin", "config.json", "preprocessor_config.json"]:
        assert (target / name).read_bytes() == (source / name).read_bytes()


def test_archive_rejects_missing_template_before_replacing_archive(tmp_path):
    from create_tar import create_tar

    source = _source(tmp_path / "source", "jinja")
    (source / "chat_template.jinja").unlink()
    destination = tmp_path / "bundle.tar.gz"
    destination.write_bytes(b"previous archive")
    assert create_tar({"name": "bundle", "source_dir": str(source), "files": []}, str(tmp_path)) is None
    assert destination.read_bytes() == b"previous archive"


def test_comparison_wrapper_uses_template_bearing_local_tokenizer(tmp_path, monkeypatch):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    import validate
    from compare import run_wrapper_pytorch

    tokenizer = save_tokenizer_assets(tmp_path / "tokenizer")
    config = Qwen3ASRAudioEncoderConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_layers=1,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        output_dim=4,
        n_window=50,
        n_window_infer=800,
    )
    config._attn_implementation = "eager"
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=Qwen3ASRAudioEncoder(config).eval()))
    prompts = []

    def decode(model, features, prompt_ids, **kwargs):
        prompts.append(prompt_ids)
        return [151645]

    monkeypatch.setattr(validate, "greedy_decode_pytorch", decode)
    result = run_wrapper_pytorch(model, np.zeros(1600, dtype=np.float32), tokenizer)
    assert prompts == [build_prompt_ids(result["audio_token_count"], tokenizer)]


@pytest.mark.parametrize("missing_template", [False, True])
def test_archive_cli_preserves_tokenizer_or_fails_without_overwriting(tmp_path, missing_template):
    from share_weights import share_external_models

    source = _source(tmp_path / "release" / "qwen3-asr-0.6b", "jinja")
    assert share_external_models(source)
    archive_path = tmp_path / "qwen3-asr-0.6b.tar.gz"
    if missing_template:
        (source / "chat_template.jinja").unlink()
        archive_path.write_bytes(b"previous archive")
    command = [
        sys.executable,
        str(ROOT / "create_tar.py"),
        "--dry-run",
        "--only",
        "qwen3-asr-0.6b",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if missing_template:
        assert result.returncode != 0
        assert archive_path.read_bytes() == b"previous archive"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp_path / "extracted", filter="data")
        _assert_source_assets(source, tmp_path / "extracted" / "qwen3-asr-0.6b")


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py"])
def test_graph_only_conversion_does_not_invent_tokenizer_assets(tmp_path, script):
    source = _source(tmp_path / "source", "jinja")
    for name in TOKENIZER_FILES:
        (source / name).unlink(missing_ok=True)
    target = tmp_path / "target"
    command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not tokenizer_asset_names(target)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_asset_copy_does_not_mutate_linked_retained_metadata(tmp_path, link_kind):
    source = tmp_path / "source"
    save_tokenizer_assets(source)
    target = tmp_path / "target"
    target.mkdir()
    retained = tmp_path / "retained.json"
    retained.write_bytes(b"retained metadata")
    path = target / "tokenizer_config.json"
    if link_kind == "symlink":
        path.symlink_to(retained)
    else:
        os.link(retained, path)
    copy_tokenizer_assets(source, target)
    assert retained.read_bytes() == b"retained metadata"
    assert not path.is_symlink()
    _assert_source_assets(source, target)


def test_archive_materializes_linked_template_bytes(tmp_path):
    from create_tar import create_tar

    source = _source(tmp_path / "source", "jinja")
    retained_template = tmp_path / "template.jinja"
    (source / "chat_template.jinja").rename(retained_template)
    (source / "chat_template.jinja").symlink_to(retained_template)
    path = create_tar({"name": "bundle", "source_dir": str(source), "files": []}, str(tmp_path))
    with tarfile.open(path) as archive:
        assert archive.getmember("bundle/chat_template.jinja").isfile()
        archive.extractall(tmp_path / "extracted", filter="data")
    _assert_source_assets(source, tmp_path / "extracted" / "bundle")
