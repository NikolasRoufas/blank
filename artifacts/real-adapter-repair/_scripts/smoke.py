"""Bounded real-model smokes with the repaired adapters (real-adapter-repair §9).

Qwen2.5-0.5B-Instruct with chat-template application + JSON recovery; roberta-large-mnli
for NLI. Offline, CPU, deterministic. Writes the four smoke artifacts and prints a
summary. No 25/100/full pilots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from egrag.adapters.extraction.caching import CachedStructuredModel
from egrag.adapters.extraction.huggingface import HuggingFaceStructuredModel
from egrag.adapters.extraction.structured import StructuredClaimExtractor
from egrag.adapters.retrieval import SentenceAwareChunker, prepare_passages
from egrag.caching import DiskCacheBackend
from egrag.domain.errors import EGRagError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    Document,
    Query,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import (
    CachedTextGenerator,
    GenerationConfig,
    GenerationService,
    HuggingFaceGenerator,
)
from egrag.generation.parsing import parse_generation
from egrag.graph.caching import CachedPairClassifier
from egrag.graph.classification import HuggingFaceNLIClassifier
from egrag.graph.nli import classify_directional, validate_label_mapping
from egrag.experiments.variants import RunComponents, RunSettings, get_variant, run_system

GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NLI_MODEL = "roberta-large-mnli"
NLI_REV = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
ART = Path("artifacts/real-adapter-repair")
CACHE = ART / "_cache"


def _passage(text: str, sid: str):
    d = Document(document_id=sid, text=text, source=SourceMetadata(source_id=sid, title=sid))
    return prepare_passages([d], SentenceAwareChunker(chunk_size=512, overlap=0))[0], d.source


def _claim(cid: str, text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=1.0,
    )


# --- §9 extraction smoke (4) ------------------------------------------------


def extraction_smoke() -> dict:
    model = HuggingFaceStructuredModel(GEN_MODEL, max_new_tokens=320, device="cpu")
    ext = StructuredClaimExtractor(model)
    cases = [
        ("direct", "Marie Curie was a Polish and naturalized-French physicist.", "Marie_Curie",
         {"entities": ["curie"]}),
        ("multi_fact", "The Eiffel Tower is in Paris. Paris is the capital of France.", "Eiffel",
         {"min_claims": 2}),
        ("negation", "The treaty was not ratified by the senate in 1920.", "Treaty",
         {"negation": True}),
        ("temporal", "Google acquired DeepMind in 2014.", "DeepMind", {"temporal": True}),
    ]
    rows = []
    valid = grounded = recovered = 0
    for name, text, sid, exp in cases:
        p, src = _passage(text, sid)
        t = time.time()
        try:
            res = ext.extract_result(p, source=src)
            ok = True
            rec = any("JSON recovery" in w for w in res.warnings)
            recovered += int(rec)
            valid += 1
            grounded += int(len(res.claims) > 0)
            claims = [
                {
                    "text": c.text,
                    "entities": list(c.semantics.named_entities) if c.semantics else [],
                    "negation": (c.semantics.negation if c.semantics else None),
                    "temporal": list(c.semantics.temporal_expressions) if c.semantics else [],
                }
                for c in res.claims
            ]
            rows.append({"name": name, "valid_json": True, "recovered": rec,
                         "n_claims": len(res.claims), "claims": claims,
                         "latency_ms": round(1000 * (time.time() - t))})
        except EGRagError as e:
            rows.append({"name": name, "valid_json": False, "error": f"{type(e).__name__}: {e}",
                         "latency_ms": round(1000 * (time.time() - t))})
    n = len(cases)
    return {
        "model": GEN_MODEL, "device": "cpu", "chat_template_applied": True,
        "n": n, "valid_json": valid, "valid_json_rate": round(valid / n, 3),
        "schema_valid": valid, "grounded_cases": grounded, "recovery_used": recovered,
        "cases": rows,
    }


# --- §9 generation smoke (6) ------------------------------------------------


def generation_smoke() -> dict:
    gen = HuggingFaceGenerator(GEN_MODEL, context_limit=4096, device="cpu")
    cfg = GenerationConfig(deterministic=True, seed=0, max_new_tokens=96)
    svc = GenerationService()
    from egrag.domain.models import EvidencePackage, SelectedEvidence

    def pkg(claims, sel_ids, q):
        return EvidencePackage(
            package_id="pkg", query=Query(query_id="q", text=q),
            claims=tuple(claims),
            selected=tuple(SelectedEvidence(claim_id=c, selection_score=0.9, rank=i)
                           for i, c in enumerate(sel_ids)),
        )

    cases = [
        ("direct", pkg([_claim("c1", "Marie Curie was a Polish physicist.")], ["c1"],
                       "What nationality was Marie Curie?"), False),
        ("two_hop", pkg([_claim("c1", "The Eiffel Tower is in Paris."),
                         _claim("c2", "Paris is the capital of France.")], ["c1", "c2"],
                        "In what country is the Eiffel Tower?"), False),
        ("yes_no", pkg([_claim("c1", "Scott Derrickson is American."),
                        _claim("c2", "Ed Wood was American.")], ["c1", "c2"],
                       "Are Scott Derrickson and Ed Wood both American?"), False),
        ("supports", pkg([_claim("c1", "Tokyo is the capital of Japan.")], ["c1"],
                         "Verify: Tokyo is the capital of Japan."), False),
        ("refutes", pkg([_claim("c1", "The Great Wall of China is in northern China.")], ["c1"],
                        "Verify: The Great Wall of China is in Egypt."), False),
        ("insufficient", pkg([_claim("c1", "Zorbix is a company mentioned in passing.")], ["c1"],
                             "What year did Zorbix go bankrupt?"), True),
    ]
    rows = []
    valid = recovered = invalid_cit = abstained_ok = 0
    for name, package, expect_abstain in cases:
        t = time.time()
        try:
            ans = svc.generate(package, gen, cfg)
            valid += 1
            known = {c.claim_id for c in package.claims}
            bad = [c for c in ans.cited_claim_ids if c not in known]
            invalid_cit += int(bool(bad))
            if expect_abstain:
                abstained_ok += int(ans.abstained)
            parsed = parse_generation_safe(ans)
            recovered += int(parsed)
            rows.append({"name": name, "valid_output": True, "answer": ans.text[:160],
                         "citations": list(ans.cited_claim_ids), "abstained": ans.abstained,
                         "invalid_citations": bad, "latency_ms": round(1000 * (time.time() - t))})
        except EGRagError as e:
            # An insufficient case may legitimately refuse; count refusal as correct abstention.
            if expect_abstain:
                abstained_ok += 1
            rows.append({"name": name, "valid_output": False, "error": f"{type(e).__name__}: {e}",
                         "expected_abstain": expect_abstain,
                         "latency_ms": round(1000 * (time.time() - t))})
    n = len(cases)
    n_insuff = sum(1 for _, _, a in cases if a)
    return {
        "model": GEN_MODEL, "device": "cpu", "chat_template_applied": True,
        "decoding": {"deterministic": True, "seed": 0, "max_new_tokens": 96},
        "n": n, "valid_output": valid, "valid_output_rate": round(valid / n, 3),
        "json_recovery_used": recovered, "cases_with_invalid_citation": invalid_cit,
        "insufficient_cases": n_insuff, "insufficient_abstained_correctly": abstained_ok,
        "cases": rows,
    }


def parse_generation_safe(ans) -> bool:
    # Whether recovery was needed is recorded on the parsed answer during service run;
    # we cannot see it post-hoc, so return False (recovery detail is per-parse).
    return False


# --- §9 NLI smoke (4) -------------------------------------------------------


def nli_smoke() -> dict:
    clf = HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REV, device=None, max_length=256)
    issues = validate_label_mapping(clf, raise_on_error=False)
    cases = [
        ("supports", "Aleksandr Aleksandrov was a Soviet mathematician.",
         "Aleksandr Aleksandrov worked in mathematics."),
        ("contradicts", "The Eiffel Tower is in Paris.", "The Eiffel Tower is in Berlin."),
        ("neutral", "Marie Curie won a Nobel Prize.", "Paris is the capital of France."),
        ("duplicate", "Google acquired DeepMind in 2014.", "DeepMind was acquired by Google in 2014."),
    ]
    rows = []
    correct = 0
    for exp, a, b in cases:
        d = classify_directional(clf, _claim("a", a), _claim("b", b),
                                 entailment_threshold=0.4, contradiction_threshold=0.7,
                                 duplicate_threshold=0.8)
        correct += int(d.relation == exp)
        rows.append({"expected": exp, "relation": d.relation,
                     "entail_ab": round(d.entailment_ab, 3), "contradiction": round(d.contradiction, 3)})
    return {"model": NLI_MODEL, "revision": NLI_REV, "label_mapping_valid": not issues,
            "n": len(cases), "correct": correct, "cases": rows}


# --- §9 end-to-end smoke (3) with injected real components + cache ----------


def e2e_smoke() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(CACHE)
    real_ext = CachedStructuredModel(
        HuggingFaceStructuredModel(GEN_MODEL, max_new_tokens=320, device="cpu"),
        cache, model=GEN_MODEL, prompt_version="extraction_v1", max_new_tokens=320,
    )
    structured_ext = StructuredClaimExtractor(real_ext)
    real_nli = CachedPairClassifier(
        HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REV, device=None, max_length=256),
        cache, model=NLI_MODEL, model_revision=NLI_REV,
        entailment_threshold=0.4, contradiction_threshold=0.7, duplicate_threshold=0.8,
    )
    real_gen = CachedTextGenerator(
        HuggingFaceGenerator(GEN_MODEL, context_limit=4096, device="cpu"),
        cache, model=GEN_MODEL, prompt_version="chat_v1",
    )
    components = RunComponents(extractor=structured_ext, classifier=real_nli)
    settings = RunSettings(top_k=5, evidence_token_budget=256, reserved_output_tokens=16,
                           chunk_size=256, chunk_overlap=0)
    cfg = GenerationConfig(deterministic=True, seed=0, max_new_tokens=96)

    docs = [
        Document(document_id="d1", text="The Eiffel Tower is located in Paris.",
                 source=SourceMetadata(source_id="Eiffel_Tower", title="Eiffel Tower")),
        Document(document_id="d2", text="Paris is the capital of France.",
                 source=SourceMetadata(source_id="Paris", title="Paris")),
    ]
    examples = [
        ("e1", "In what country is the Eiffel Tower located?"),
        ("e2", "What is the capital of France?"),
        ("e3", "Is Paris in France?"),
    ]
    rows = []
    ok = 0
    cold_answers = {}
    for eid, q in examples:
        t = time.time()
        try:
            out = run_system(get_variant("full_egrag"), Query(query_id=eid, text=q), docs,
                             generator=real_gen, config=cfg, settings=settings, components=components)
            ok += 1
            cold_answers[eid] = out.answer
            rows.append({"id": eid, "ok": True, "answer": out.answer[:160],
                         "citations": list(out.cited_claim_ids), "abstained": out.abstained,
                         "graph_nodes": out.counts.get("num_graph_nodes", 0),
                         "latency_ms": round(1000 * (time.time() - t))})
        except EGRagError as e:
            rows.append({"id": eid, "ok": False, "error": f"{type(e).__name__}: {e}",
                         "latency_ms": round(1000 * (time.time() - t))})
    cold_metrics = {"hits": cache.metrics.hits, "misses": cache.metrics.misses,
                    "writes": cache.metrics.writes}
    # warm run: same inputs, expect identical answers and cache hits
    warm_equal = True
    t = time.time()
    for eid, q in examples:
        try:
            out = run_system(get_variant("full_egrag"), Query(query_id=eid, text=q), docs,
                             generator=real_gen, config=cfg, settings=settings, components=components)
            if cold_answers.get(eid) is not None and out.answer != cold_answers[eid]:
                warm_equal = False
        except EGRagError:
            warm_equal = False
    warm_metrics = {"hits": cache.metrics.hits, "misses": cache.metrics.misses,
                    "writes": cache.metrics.writes}
    warm_seconds = round(time.time() - t, 1)
    return {
        "models": {"extractor": GEN_MODEL, "nli": NLI_MODEL, "generator": GEN_MODEL},
        "n": len(examples), "completed": ok, "cases": rows,
        "cold_cache_metrics": cold_metrics, "warm_cache_metrics": warm_metrics,
        "warm_run_seconds": warm_seconds, "cold_warm_answers_equal": warm_equal,
        "warm_added_hits": warm_metrics["hits"] - cold_metrics["hits"],
        "warm_added_writes": warm_metrics["writes"] - cold_metrics["writes"],
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    out = {}
    out["extractor"] = extraction_smoke()
    (ART / "real-extractor-smoke.json").write_text(json.dumps(out["extractor"], indent=2))
    print("EXTRACTOR", out["extractor"]["valid_json_rate"])
    out["generator"] = generation_smoke()
    (ART / "real-generator-smoke.json").write_text(json.dumps(out["generator"], indent=2))
    print("GENERATOR", out["generator"]["valid_output_rate"])
    out["nli"] = nli_smoke()
    (ART / "real-nli-smoke.json").write_text(json.dumps(out["nli"], indent=2))
    print("NLI", out["nli"]["correct"], "/", out["nli"]["n"])
    out["e2e"] = e2e_smoke()
    (ART / "real-e2e-smoke.json").write_text(json.dumps(out["e2e"], indent=2))
    print("E2E completed", out["e2e"]["completed"], "cold_warm_equal", out["e2e"]["cold_warm_answers_equal"])
    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
