"""Persistent-cache wrapper tests (real-adapter-repair §6, §8).

Cold/warm equality, hit on identical input, miss on model-revision / prompt-version
/ threshold change, corruption quarantine, and disabled-cache no-write — all with
deterministic fakes and a real :class:`DiskCacheBackend` on a tmp dir.
"""

from __future__ import annotations

import pytest

from egrag.adapters.extraction.caching import CachedStructuredModel
from egrag.caching import DiskCacheBackend, NullCacheBackend
from egrag.domain.models import (
    AtomicClaim,
    ChatMessage,
    ClaimProvenance,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import CachedTextGenerator, GenerationConfig, GeneratorCapabilities
from egrag.graph.caching import CachedPairClassifier
from egrag.graph.types import ClaimPair, RelationProbabilities

# --- fakes ------------------------------------------------------------------


class CountingStructuredModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str:
        self.calls += 1
        return f'{{"claims": [], "echo": "{prompt[:8]}"}}'


class CountingClassifier:
    classifier_id = "counting"
    classifier_version = "1.0"
    model_revision = None

    def __init__(self) -> None:
        self.seen = 0

    def classify(self, pairs):
        self.seen += len(list(pairs))
        return [
            RelationProbabilities(entailment=0.9, contradiction=0.05, neutral=0.05) for _ in pairs
        ]


class CountingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(chat_template=True, has_tokenizer=True)

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        self.calls += 1
        return '{"answer": "x", "citations": [], "uncertainty": ""}'

    def complete_chat(self, messages, config: GenerationConfig) -> str:
        self.calls += 1
        return '{"answer": "chat", "citations": [], "uncertainty": ""}'


def _claim(cid: str, text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id="s"),
            spans=(SourceSpan(source_id="s", start=0, end=len(text), text=text),),
        ),
        extraction_confidence=1.0,
    )


def _pair() -> ClaimPair:
    return ClaimPair(
        source=_claim("a", "Paris is in France."), target=_claim("b", "Paris, France.")
    )


# --- structured model -------------------------------------------------------


@pytest.mark.unit
def test_structured_cold_warm_equal_and_single_call(tmp_path) -> None:
    inner = CountingStructuredModel()
    cache = DiskCacheBackend(tmp_path / "c")
    cached = CachedStructuredModel(inner, cache, model="m", model_revision="r1")
    cold = cached.complete("hello world prompt")
    warm = cached.complete("hello world prompt")
    assert cold == warm
    assert inner.calls == 1  # warm was a cache hit
    assert cache.metrics.hits == 1 and cache.metrics.writes == 1


@pytest.mark.unit
def test_structured_miss_on_revision_and_prompt_version(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingStructuredModel()
    CachedStructuredModel(inner, cache, model="m", model_revision="r1").complete("p")
    # different revision -> miss
    CachedStructuredModel(inner, cache, model="m", model_revision="r2").complete("p")
    # different prompt version -> miss
    CachedStructuredModel(
        inner, cache, model="m", model_revision="r1", prompt_version="v2"
    ).complete("p")
    assert inner.calls == 3


@pytest.mark.unit
def test_structured_disabled_cache_no_writes(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c", enabled=False)
    inner = CountingStructuredModel()
    cached = CachedStructuredModel(inner, cache, model="m")
    cached.complete("p")
    cached.complete("p")
    assert inner.calls == 2  # nothing cached
    assert cache.metrics.writes == 0


# --- NLI classifier ---------------------------------------------------------


@pytest.mark.unit
def test_nli_cold_warm_equal_and_cached(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingClassifier()
    clf = CachedPairClassifier(
        inner,
        cache,
        model="roberta-large-mnli",
        entailment_threshold=0.4,
        contradiction_threshold=0.7,
        duplicate_threshold=0.8,
    )
    cold = clf.classify([_pair()])
    warm = clf.classify([_pair()])
    assert cold == warm
    assert inner.seen == 1  # warm served from cache


@pytest.mark.unit
def test_nli_miss_on_threshold_change(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingClassifier()
    common = {
        "model": "roberta-large-mnli",
        "contradiction_threshold": 0.7,
        "duplicate_threshold": 0.8,
    }
    CachedPairClassifier(inner, cache, entailment_threshold=0.4, **common).classify([_pair()])
    CachedPairClassifier(inner, cache, entailment_threshold=0.5, **common).classify([_pair()])
    assert inner.seen == 2  # threshold change -> miss


# --- generator --------------------------------------------------------------


@pytest.mark.unit
def test_generator_cold_warm_and_revision_miss(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingGenerator()
    cfg = GenerationConfig()
    CachedTextGenerator(inner, cache, model="g", model_revision="r1").complete("p", cfg)
    CachedTextGenerator(inner, cache, model="g", model_revision="r1").complete("p", cfg)
    assert inner.calls == 1  # warm hit
    CachedTextGenerator(inner, cache, model="g", model_revision="r2").complete("p", cfg)
    assert inner.calls == 2  # revision change -> miss


@pytest.mark.unit
def test_generator_chat_cached(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingGenerator()
    cfg = GenerationConfig()
    msgs = (ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u"))
    g = CachedTextGenerator(inner, cache, model="g")
    a = g.complete_chat(msgs, cfg)
    b = g.complete_chat(msgs, cfg)
    assert a == b
    assert inner.calls == 1


@pytest.mark.unit
def test_corrupt_entry_becomes_miss(tmp_path) -> None:
    cache = DiskCacheBackend(tmp_path / "c")
    inner = CountingStructuredModel()
    cached = CachedStructuredModel(inner, cache, model="m")
    cached.complete("p")  # writes one entry
    # Corrupt every cache file, then a re-read must miss and recompute.
    for f in (tmp_path / "c").glob("*.json"):
        f.write_text("{ not valid json", encoding="utf-8")
    cached.complete("p")
    assert inner.calls == 2
    assert cache.metrics.corruptions >= 1


@pytest.mark.unit
def test_null_cache_no_writes() -> None:
    inner = CountingStructuredModel()
    cached = CachedStructuredModel(inner, NullCacheBackend(), model="m")
    cached.complete("p")
    cached.complete("p")
    assert inner.calls == 2
