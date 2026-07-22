"""Chat-template behavior of the shared HF runtime (real-adapter-repair §2, §8).

These tests inject a fake pipeline + tokenizer (no model download, no transformers
required) to verify: the chat template is applied when present, the fallback is
used when absent, system/user roles are preserved, deterministic mode does not
sample, and the full prompt is not echoed (``return_full_text=False``).
"""

from __future__ import annotations

from typing import Any

import pytest

from egrag.hf_runtime import HFTextPipeline


class _FakeTokenizer:
    def __init__(self, *, chat_template: str | None) -> None:
        self.chat_template = chat_template
        self.pad_token_id = 7
        self.eos_token_id = 7
        self.applied: list[dict[str, str]] | None = None

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.applied = messages
        return "<CHAT_TEMPLATE>" + "|".join(f"{m['role']}={m['content']}" for m in messages)


class _FakePipeline:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.last_text: str | None = None
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, str]]:
        self.last_text = text
        self.last_kwargs = kwargs
        return [{"generated_text": "OUTPUT"}]


def _runtime(*, chat_template: str | None) -> tuple[HFTextPipeline, _FakePipeline, _FakeTokenizer]:
    rt = HFTextPipeline("fake/model")
    tok = _FakeTokenizer(chat_template=chat_template)
    pipe = _FakePipeline(tok)
    rt._pipeline = pipe  # bypass real load
    rt._tokenizer = tok
    # Neutralize seeding so these fake-only tests never import transformers (which
    # would pollute sys.modules and break the lazy-loading assertions elsewhere).
    rt._seed = lambda seed=None: None  # type: ignore[method-assign]
    return rt, pipe, tok


@pytest.mark.unit
def test_chat_template_applied_when_available() -> None:
    rt, pipe, tok = _runtime(chat_template="{{ template }}")
    out = rt.generate(
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ],
        max_new_tokens=16,
    )
    assert out == "OUTPUT"
    assert tok.applied == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert pipe.last_text is not None and pipe.last_text.startswith("<CHAT_TEMPLATE>")


@pytest.mark.unit
def test_fallback_when_no_chat_template() -> None:
    rt, pipe, tok = _runtime(chat_template=None)
    rt.generate(
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ],
        max_new_tokens=16,
    )
    assert tok.applied is None  # template not used
    assert pipe.last_text is not None
    # roles preserved and separated in the fallback prompt
    assert "SYSTEM:" in pipe.last_text and "SYS" in pipe.last_text
    assert "USER:" in pipe.last_text and "USR" in pipe.last_text
    assert pipe.last_text.rstrip().endswith("ASSISTANT:")


@pytest.mark.unit
def test_deterministic_mode_does_not_sample() -> None:
    rt, pipe, _ = _runtime(chat_template="t")
    rt.generate(messages=[{"role": "user", "content": "x"}], deterministic=True)
    assert pipe.last_kwargs["do_sample"] is False
    assert "temperature" not in pipe.last_kwargs
    assert pipe.last_kwargs["return_full_text"] is False


@pytest.mark.unit
def test_sampling_mode_sets_temperature() -> None:
    rt, pipe, _ = _runtime(chat_template="t")
    rt.generate(
        messages=[{"role": "user", "content": "x"}],
        deterministic=False,
        temperature=0.7,
        top_p=0.9,
    )
    assert pipe.last_kwargs["do_sample"] is True
    assert pipe.last_kwargs["temperature"] == 0.7
    assert pipe.last_kwargs["top_p"] == 0.9


@pytest.mark.unit
def test_raw_prompt_path_passes_prompt_through() -> None:
    rt, pipe, _ = _runtime(chat_template="t")
    rt.generate(prompt="RAW PROMPT", max_new_tokens=8)
    assert pipe.last_text == "RAW PROMPT"
    assert pipe.last_kwargs["return_full_text"] is False


@pytest.mark.unit
def test_pad_token_falls_back_when_present() -> None:
    rt, pipe, _ = _runtime(chat_template="t")
    rt.generate(prompt="x")
    assert pipe.last_kwargs["pad_token_id"] == 7
