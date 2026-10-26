"""Prompt fixture from Qwen/Qwen3-ASR-0.6B revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 chat_template.json."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src import prompt
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "system": 31, "user": 42, "assistant": 53, "language": 64, "English": 75, "Kepler": 86}
    special = {
        "<|endoftext|>": prompt.ENDOFTEXT_TOKEN_ID,
        "<|im_start|>": prompt.IM_START_TOKEN_ID,
        "<|im_end|>": prompt.IM_END_TOKEN_ID,
        "<|audio_start|>": prompt.AUDIO_START_TOKEN_ID,
        "<|audio_end|>": prompt.AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": prompt.AUDIO_PAD_TOKEN_ID,
        "<asr_text>": prompt.ASR_TEXT_TOKEN_ID,
    }
    raw = Tokenizer(models.WordLevel(vocab | special, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    result = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", additional_special_tokens=list(special))
    result.chat_template = OFFICIAL_CHAT_TEMPLATE
    return result


@pytest.mark.parametrize("context,language", [("", None), ("Kepler", None), ("", "English"), ("Kepler", "English")])
@pytest.mark.parametrize("audio_count", [1, 3, 101])
def test_prompt_matches_official_template(tokenizer, context, language, audio_count):
    messages = [{"role": "system", "content": context}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    rendered = rendered.replace("<|audio_pad|>", "<|audio_pad|>" * audio_count)
    if language:
        rendered += f"language {language}<asr_text>"
    expected = tokenizer.encode(rendered, add_special_tokens=False)
    actual = prompt.build_prompt_ids(audio_count, tokenizer, context=context, language=language)
    assert actual == expected
    start, end = prompt.get_audio_pad_range(actual, expected_count=audio_count)
    assert end - start == audio_count
    assert actual[start - 1] == prompt.AUDIO_START_TOKEN_ID
    assert actual[end] == prompt.AUDIO_END_TOKEN_ID
    assert tokenizer.encode("system", add_special_tokens=False) == [31]
    assert tokenizer.encode("user", add_special_tokens=False) == [42]
    assert tokenizer.encode("assistant", add_special_tokens=False) == [53]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_rejects_invalid_audio_counts(tokenizer, count):
    with pytest.raises(ValueError, match="positive integer"):
        prompt.build_prompt_ids(count, tokenizer)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [prompt.AUDIO_PAD_TOKEN_ID],
        [
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_PAD_TOKEN_ID,
            123,
            prompt.AUDIO_PAD_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
        ],
        [
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_PAD_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
        ],
    ],
)
def test_rejects_malformed_audio_ranges(ids):
    with pytest.raises(ValueError):
        prompt.get_audio_pad_range(ids)


def test_rejects_wrong_audio_count(tokenizer):
    ids = prompt.build_prompt_ids(3, tokenizer)
    with pytest.raises(ValueError, match="count"):
        prompt.get_audio_pad_range(ids, expected_count=2)


def test_rejects_control_tokens_in_context(tokenizer):
    with pytest.raises(ValueError):
        prompt.build_prompt_ids(2, tokenizer, context="<|audio_pad|>")


def test_export_verifies_template_role_names(tokenizer, monkeypatch):
    import export

    monkeypatch.setattr(export, "load_prompt_tokenizer", lambda model: tokenizer)
    export.verify_special_tokens("fixture")
    tokenizer.chat_template = tokenizer.chat_template.replace("system\\n", "bogus\\n")
    with pytest.raises(ValueError, match="role names"):
        export.verify_special_tokens("fixture")


def test_export_config_records_real_embedding_shape_and_cache_dtype(tmp_path):
    import json
    from types import SimpleNamespace

    import torch
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    import export

    config = Qwen3ASRConfig()
    config.thinker_config.text_config.rope_scaling = {}
    config.thinker_config.text_config.vocab_size = 13
    config.thinker_config.text_config.hidden_size = 7
    embedding = torch.nn.Embedding(13, 7, dtype=torch.float16)
    model = SimpleNamespace(config=config, thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)))
    export.write_config(model, str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["embed_tokens_shape"] == [13, 7]
    assert saved["embed_tokens_dtype"] == "float32"


def test_loads_official_processor_template_when_tokenizer_has_none(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import transformers
    import transformers.utils

    path = tmp_path / "chat_template.json"
    path.write_text(json.dumps({"chat_template": OFFICIAL_CHAT_TEMPLATE}))
    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    requests = []

    def cached_file(model, filename, **kwargs):
        requests.append((model, filename))
        return str(path)

    monkeypatch.setattr(transformers.utils, "cached_file", cached_file)
    assert prompt.load_prompt_tokenizer("fixture").chat_template == OFFICIAL_CHAT_TEMPLATE
    assert requests == [("fixture", "chat_template.json")]


def test_exported_tokenizer_retains_official_template(tmp_path, monkeypatch):
    import export

    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = OFFICIAL_CHAT_TEMPLATE
    monkeypatch.setattr(export, "load_prompt_tokenizer", lambda model: tokenizer)
    export.copy_tokenizer("fixture", str(tmp_path))
    assert (tmp_path / "chat_template.jinja").is_file()
    assert prompt.load_prompt_tokenizer(str(tmp_path)).chat_template == OFFICIAL_CHAT_TEMPLATE


def test_missing_official_template_fails_with_actionable_error(monkeypatch):
    from types import SimpleNamespace

    import transformers
    import transformers.utils

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(chat_template=None)
    )
    monkeypatch.setattr(transformers.utils, "cached_file", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Missing official chat_template.json"):
        prompt.load_prompt_tokenizer("fixture")
