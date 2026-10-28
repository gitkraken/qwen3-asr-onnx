import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.prompt import build_prompt_ids, load_prompt_tokenizer
from tests.converter_fixtures import _source
from tests.tokenizer_fixtures import save_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("representation", ["json", "embedded"])
@pytest.mark.parametrize(
    "script", ["convert_fp16.py", "quantize_static.py", "package.py", "quantize.py", "quantize_nbits.py"]
)
def test_conflicting_prompt_representation_fails_before_output(tmp_path, representation, script):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source, "jinja")
    if representation == "json":
        (source / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    else:
        path = source / "tokenizer_config.json"
        config = json.loads(path.read_text())
        config["chat_template"] = "WRONG PROMPT"
        path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--input", str(source), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "Invalid ASR tokenizer assets" in result.stdout + result.stderr
    assert not target.exists()


def test_export_removes_stale_processor_template_and_both_readers_agree(tmp_path):
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    import export

    checkpoint, target = tmp_path / "checkpoint", tmp_path / "target"
    save_tokenizer_assets(checkpoint)
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "Qwen2TokenizerFast"
    config["extra_special_tokens"] = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    config_path.write_text(json.dumps(config))
    target.mkdir()
    (target / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    export.copy_tokenizer(str(checkpoint), str(target))
    export.write_preprocessor_config(str(target))
    assert not (target / "chat_template.json").exists()
    tokenizer = load_prompt_tokenizer(str(target), local_files_only=True)
    processor = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    assert processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) == tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
def test_conflicting_reused_target_is_repaired_by_family_publication(tmp_path, script):
    from src.tokenizer_assets import validate_tokenizer_assets
    from tests.artifact_fixtures import _write_model
    from tests.converter_fixtures import _converter_command

    source, target = tmp_path / "source", tmp_path / "target"
    _source(source, "jinja")
    expected = load_prompt_tokenizer(str(source), local_files_only=True)
    save_tokenizer_assets(target, "jinja")
    (target / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
        command = [
            sys.executable,
            str(ROOT / script),
            "--input",
            str(source),
            "--output",
            str(target),
            "--bits",
            "4",
            "--block-size",
            "32",
        ]
    else:
        command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / "chat_template.json").exists()
    actual = validate_tokenizer_assets(target)
    assert build_prompt_ids(3, actual) == build_prompt_ids(3, expected)


def test_export_normalizes_selected_asr_template(tmp_path):
    import export
    from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, "embedded")
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = {"default": OFFICIAL_CHAT_TEMPLATE, "other": "unused alternate"}
    config_path.write_text(json.dumps(config))
    export.copy_tokenizer(str(source), str(target))
    assert (target / "chat_template.jinja").read_text() == OFFICIAL_CHAT_TEMPLATE
    assert not (target / "chat_templates").exists()
    assert load_prompt_tokenizer(str(target), local_files_only=True).chat_template == OFFICIAL_CHAT_TEMPLATE
