# Extending EG-RAG

Every model-facing capability is a typed protocol. Implement the protocol, keep
provider imports lazy, and wire it in the composition root.

## A custom generator (TextGenerator)

```python
from egrag.generation import GeneratorCapabilities
from egrag.generation.config import GenerationConfig


class EchoGenerator:
    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(context_limit=8192, deterministic_decoding=True)

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        # Must return the JSON output contract the parser expects.
        return '{"answer": "stub", "citations": [], "uncertainty": ""}'
```

Use it via the generation service:

```python
from egrag.generation import GenerationService
answer = GenerationService().generate(package, EchoGenerator())
```

## A custom retriever (Retriever)

Subclass `egrag.adapters.retrieval.base.BaseRetriever` and implement `_rank`,
returning all candidates as `ScoredPassage`s in deterministic order. The base
class handles `top_k`, slicing, and stats; you get `retrieve()` (the
`egrag.domain.ports.Retriever` protocol) for free.

## A custom pair classifier (PairClassifier)

Implement `classify(self, pairs) -> list[RelationProbabilities]` plus
`classifier_id`, `classifier_version`, `model_revision`. Load any model lazily.

## A custom source-reliability strategy

Implement `score(self, source) -> float` (`SourceReliabilityScorer`). Remember:
reliability is a configurable prior, never inferred from recency, repetition,
ranking, or popularity.

## Wiring

Add construction logic in `egrag.composition.build_generator` (or the relevant
builder) keyed off the config, so there is exactly one composition root. Add a
deterministic fake and tests, and document any new config fields.
