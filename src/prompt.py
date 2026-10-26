import json
from numbers import Integral

ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ASR_TEXT_TOKEN_ID = 151704
EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]
NEWLINE_TOKEN_ID = 198


def load_prompt_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    from transformers.utils import cached_file

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not tokenizer.chat_template:
        template_path = cached_file(model_id, "chat_template.json")
        if template_path is None:
            raise ValueError(f"Missing official chat_template.json for {model_id}")
        with open(template_path) as stream:
            tokenizer.chat_template = json.load(stream)["chat_template"]
    return tokenizer


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    from src.encoder_wrapper import _get_feat_extract_output_lengths

    return int(_get_feat_extract_output_lengths(input_lengths))


def build_prompt_ids(audio_token_count: int, tokenizer, language: str | None = None, context: str = "") -> list[int]:
    if isinstance(audio_token_count, bool) or not isinstance(audio_token_count, Integral) or audio_token_count <= 0:
        raise ValueError("audio_token_count must be a positive integer")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("language must be a nonempty string")
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if language is not None:
        text += f"language {language}<asr_text>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    start, end = get_audio_pad_range(ids)
    if end - start != 1:
        raise ValueError("Chat template must contain exactly one audio placeholder")
    ids[start:end] = [AUDIO_PAD_TOKEN_ID] * int(audio_token_count)
    get_audio_pad_range(ids, expected_count=int(audio_token_count))
    return ids


def get_audio_pad_range(prompt_ids: list[int], expected_count: int | None = None) -> tuple[int, int]:
    pads = [i for i, token in enumerate(prompt_ids) if token == AUDIO_PAD_TOKEN_ID]
    if not pads:
        raise ValueError("No <|audio_pad|> tokens found in prompt")
    start, end = pads[0], pads[-1] + 1
    if len(pads) != end - start:
        raise ValueError("Audio placeholders must form one contiguous range")
    if (
        start == 0
        or end == len(prompt_ids)
        or prompt_ids[start - 1] != AUDIO_START_TOKEN_ID
        or prompt_ids[end] != AUDIO_END_TOKEN_ID
    ):
        raise ValueError("Audio placeholders must be enclosed by audio delimiters")
    if prompt_ids.count(AUDIO_START_TOKEN_ID) != 1 or prompt_ids.count(AUDIO_END_TOKEN_ID) != 1:
        raise ValueError("Prompt must contain exactly one audio segment")
    if expected_count is not None and len(pads) != expected_count:
        raise ValueError(f"Audio placeholder count {len(pads)} != expected {expected_count}")
    return start, end
