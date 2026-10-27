import json
import shutil
import tempfile
from pathlib import Path

from src import prompt
from src.artifact_publication import publish_artifacts

TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
    }
)


def tokenizer_asset_names(root) -> set[str]:
    root = Path(root)
    return {name for name in TOKENIZER_FILES if (root / name).is_file()}


def validate_tokenizer_assets(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {root}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Required tokenizer asset missing: {name}")
    try:
        tokenizer = prompt.load_prompt_tokenizer(str(root), local_files_only=True)
        _validate_prompt_tokenizer(tokenizer)
        _validate_shipped_templates(root, tokenizer)
    except Exception as error:
        raise ValueError(f"Invalid ASR tokenizer assets in {root}: {error}") from error
    return tokenizer


def _validate_prompt_tokenizer(tokenizer) -> None:
    special = {
        "<|endoftext|>": prompt.ENDOFTEXT_TOKEN_ID,
        "<|im_start|>": prompt.IM_START_TOKEN_ID,
        "<|im_end|>": prompt.IM_END_TOKEN_ID,
        "<|audio_start|>": prompt.AUDIO_START_TOKEN_ID,
        "<|audio_end|>": prompt.AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": prompt.AUDIO_PAD_TOKEN_ID,
        "<asr_text>": prompt.ASR_TEXT_TOKEN_ID,
    }
    for token, expected in special.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Tokenizer has an invalid ASR special token: {token}")
    for role in ("system", "user", "assistant"):
        ids = tokenizer.encode(role, add_special_tokens=False)
        if not ids or tokenizer.unk_token_id in ids:
            raise ValueError(f"Tokenizer cannot encode the {role} role")
    for context, language in (("", None), ("Kepler", None), ("", "English"), ("Kepler", "English")):
        actual = prompt.build_prompt_ids(2, tokenizer, context=context, language=language)
        expected_text = (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        if language is not None:
            expected_text += f"language {language}<asr_text>"
        if actual != tokenizer.encode(expected_text, add_special_tokens=False):
            raise ValueError("Tokenizer chat template does not implement the Qwen3-ASR prompt contract")


def _template_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict) and value and all(isinstance(item, str) for item in value.values()):
        return list(value.values())
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("template"), str)
            for item in value
        )
    ):
        return [item["template"] for item in value]
    raise ValueError("Chat template must be a string or a nonempty named template collection")


def _validate_shipped_templates(root, tokenizer):
    config = json.loads((root / "tokenizer_config.json").read_text())
    templates = []
    if "chat_template" in config:
        templates.extend(_template_values(config["chat_template"]))
    legacy = root / "chat_template.json"
    if legacy.is_file():
        document = json.loads(legacy.read_text())
        if not isinstance(document, dict) or "chat_template" not in document:
            raise ValueError("chat_template.json must contain chat_template")
        templates.extend(_template_values(document["chat_template"]))
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        templates.append(jinja.read_text())
    selected = tokenizer.chat_template
    try:
        for template in templates:
            tokenizer.chat_template = template
            _validate_prompt_tokenizer(tokenizer)
    finally:
        tokenizer.chat_template = selected


def copy_tokenizer_assets(source, target) -> None:
    source, target = Path(source), Path(target)
    validate_tokenizer_assets(source)
    if source.resolve() == target.resolve():
        return
    names = tokenizer_asset_names(source)
    target.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if (target / name).is_dir():
            raise ValueError(f"Tokenizer asset destination is a directory: {target / name}")
    # Existing metadata may link to a retained release; never write through those links.
    with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=target) as temp:
        stage = Path(temp)
        for name in sorted(names):
            shutil.copy2(source / name, stage / name)
        validate_tokenizer_assets(stage)
        publish_artifacts(
            [(stage / name, target / name) for name in sorted(names)],
            [target / name for name in sorted(TOKENIZER_FILES - names)],
        )
