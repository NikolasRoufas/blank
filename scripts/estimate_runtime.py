#!/usr/bin/env python
"""Runtime estimator: times model load + one NLI batch + one generation.

Offline, CPU. Writes a small JSON to stdout and projects FEVER-pilot runtime.
Requires the cached `roberta-large-mnli` and `Qwen2.5-0.5B-Instruct`. On a
near-full disk the model load may wedge — run only with comfortable free disk.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
     .venv/bin/python scripts/estimate_runtime.py
"""

from __future__ import annotations

import json
import time

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation.adapters import HuggingFaceGenerator
from egrag.generation.config import GenerationConfig
from egrag.graph import HuggingFaceNLIClassifier
from egrag.graph.types import ClaimPair

_NLI_REV = "2a8f12d27941090092df78e4ba6f0928eb5eac98"


def _claim(cid: str, text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
    )


def main() -> None:
    nli = HuggingFaceNLIClassifier(
        "roberta-large-mnli", model_revision=_NLI_REV, batch_size=16, max_length=256
    )
    pairs = [
        ClaimPair(
            source=_claim(f"a{i}", f"Entity{i} reported revenue of {i} million in 2023."),
            target=_claim(f"b{i}", f"Entity{i} did not report revenue in 2023."),
        )
        for i in range(16)
    ]
    t = time.perf_counter()
    nli.classify(pairs[:1])  # triggers load
    nli_load = time.perf_counter() - t
    t = time.perf_counter()
    nli.classify(pairs)
    nli_batch = time.perf_counter() - t

    gen = HuggingFaceGenerator("Qwen/Qwen2.5-0.5B-Instruct", context_limit=4096)
    cfg = GenerationConfig(max_new_tokens=48, deterministic=True, seed=0)
    prompt = (
        "Answer only from the evidence. Evidence: [c1] Paris is the capital of "
        "France. Question: What is the capital of France?"
    )
    t = time.perf_counter()
    gen.complete(prompt, cfg)  # load + first generation
    gen_cold = time.perf_counter() - t
    t = time.perf_counter()
    gen.complete(prompt, cfg)
    gen_warm = time.perf_counter() - t

    nli_per_pair = nli_batch / len(pairs)
    per_example_warm = 4 * nli_per_pair + gen_warm  # ~4 NLI pairs + 1 generation
    out = {
        "nli_load_s": round(nli_load, 2),
        "nli_per_pair_s": round(nli_per_pair, 3),
        "gen_cold_s": round(gen_cold, 2),
        "gen_warm_s": round(gen_warm, 2),
        "est_per_example_warm_s": round(per_example_warm, 2),
        "proj_fever_25x8_warm_min": round(per_example_warm * 25 * 8 / 60, 1),
    }
    print("RUNTIME " + json.dumps(out))


if __name__ == "__main__":
    main()
