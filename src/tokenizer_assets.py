import os
import shutil
import tempfile
from pathlib import Path

from src import prompt

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


def copy_tokenizer_assets(source, target) -> None:
    source, target = Path(source), Path(target)
    validate_tokenizer_assets(source)
    if source.resolve() == target.resolve():
        return
    names = tokenizer_asset_names(source)
    target.mkdir(parents=True, exist_ok=True)
    # Existing metadata may link to a retained release; never write through those links.
    with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=target) as temp:
        stage = Path(temp)
        for name in sorted(names):
            shutil.copy2(source / name, stage / name)
        for name in sorted(TOKENIZER_FILES - names):
            (target / name).unlink(missing_ok=True)
        for name in sorted(names):
            os.replace(stage / name, target / name)
    validate_tokenizer_assets(target)
