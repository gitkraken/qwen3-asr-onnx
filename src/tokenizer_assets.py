import json
import shutil
from pathlib import Path

from src import prompt
from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.artifact_sources import validate_publication_sources
from src.prompt import validate_prompt_tokenizer

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


def validate_tokenizer_destination(root):
    directory = Path(root) / "additional_chat_templates"
    if directory.is_dir():
        raise ValueError(f"Unsupported tokenizer template directory: {directory}; use a clean output directory")


def _validate_tokenizer_bindings(root, tokenizer=None):
    from transformers import PreTrainedTokenizerFast
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    from transformers.tokenization_utils_base import get_fast_tokenizer_file

    path = Path(root) / "tokenizer_config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError("Invalid tokenizer binding configuration")
    tokenizer_class = type(tokenizer) if tokenizer is not None else None
    if tokenizer_class is None and isinstance(config.get("tokenizer_class"), str):
        tokenizer_class = tokenizer_class_from_name(config["tokenizer_class"])
    file_slots = set(PreTrainedTokenizerFast.vocab_files_names)
    if tokenizer_class is not None:
        file_slots.update(tokenizer_class.vocab_files_names)
    for slot in file_slots:
        if slot in config:
            raise ValueError(f"Unsupported tokenizer file binding: {slot}")
    if config.get("from_slow") or any(
        config.get(slot) is not None for slot in ("tokenizer_object", "__slow_tokenizer", "gguf_file")
    ):
        raise ValueError("Unsupported tokenizer construction binding")
    if get_fast_tokenizer_file(config.get("fast_tokenizer_files", [])) != "tokenizer.json":
        raise ValueError("Unsupported tokenizer backend binding; expected tokenizer.json")


def _tokenizer_source_reads(root, tokenizer):
    root = Path(root)
    names = TOKENIZER_FILES | {"config.json"} | set(tokenizer.vocab_files_names.values())
    reads = {root / name for name in names if (root / name).is_file()}
    templates = root / "additional_chat_templates"
    if templates.is_dir():
        reads.update(path for path in templates.glob("*.jinja") if path.is_file())
    return reads


def tokenizer_checkpoint_reads(model_id, tokenizer):
    from transformers.utils import cached_file

    root = Path(model_id)
    if not root.is_dir():
        for name in ("tokenizer_config.json", "config.json", "tokenizer.json"):
            resolved = cached_file(model_id, name, local_files_only=True, _raise_exceptions_for_missing_entries=False)
            if resolved is not None:
                root = Path(resolved).parent
                break
        else:
            raise ValueError(f"Cannot establish tokenizer checkpoint source: {model_id}")
    _validate_tokenizer_bindings(root, tokenizer)
    return _tokenizer_source_reads(root, tokenizer)


def validate_tokenizer_assets(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {root}")
    validate_tokenizer_destination(root)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Required tokenizer asset missing: {name}")
    try:
        _validate_tokenizer_bindings(root)
        tokenizer = prompt.load_prompt_tokenizer(str(root), local_files_only=True)
        _validate_tokenizer_bindings(root, tokenizer)
        if not tokenizer.is_fast:
            raise ValueError("Unsupported tokenizer binding; expected a fast tokenizer")
        validate_prompt_tokenizer(tokenizer)
        _validate_shipped_templates(root, tokenizer)
    except Exception as error:
        raise ValueError(f"Invalid ASR tokenizer assets in {root}: {error}") from error
    return tokenizer


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
            validate_prompt_tokenizer(tokenizer)
    finally:
        tokenizer.chat_template = selected


def copy_tokenizer_assets(source, target) -> None:
    source, target = Path(source), Path(target)
    tokenizer = validate_tokenizer_assets(source)
    validate_tokenizer_destination(target)
    if source.resolve() == target.resolve():
        return
    names = tokenizer_asset_names(source)
    validate_publication_sources(
        _tokenizer_source_reads(source, tokenizer),
        files=(target / name for name in TOKENIZER_FILES),
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if (target / name).is_dir():
            raise ValueError(f"Tokenizer asset destination is a directory: {target / name}")
    # Existing metadata may link to a retained release; never write through those links.
    with ArtifactStaging(prefix=".tokenizer-", dir=target) as staging:
        stage = staging.path
        for name in sorted(names):
            shutil.copy2(source / name, stage / name)
        validate_tokenizer_assets(stage)
        publish_artifacts(
            [(stage / name, target / name) for name in sorted(names)],
            [target / name for name in sorted(TOKENIZER_FILES - names)],
        )
        staging.mark_complete()
