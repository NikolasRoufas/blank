import json, time
from egrag.domain.models import Document, Query, SourceMetadata
from egrag.generation import GenerationConfig
from egrag.generation.adapters import HuggingFaceGenerator
from egrag.experiments.variants import RunSettings, get_variant, run_system

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
def doc(did, text, sid):
    return Document(document_id=did, text=text, source=SourceMetadata(source_id=sid, title=sid.replace("_"," ")))

class Recording(HuggingFaceGenerator):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.last_raw=None
    def complete(self, prompt, config):
        r=super().complete(prompt, config); self.last_raw=r; return r

CASES=[
 ("direct","claim_only_rag","What nationality was Marie Curie?",
   [doc("d1","Marie Curie was a Polish and naturalized-French physicist and chemist.","Marie_Curie")]),
 ("two_hop","full_egrag","What is the capital of the country where the Eiffel Tower is located?",
   [doc("d1","The Eiffel Tower is located in Paris.","Eiffel_Tower"),
    doc("d2","Paris is the capital of France.","Paris")]),
 ("yes_no","full_egrag","Are Scott Derrickson and Ed Wood both American?",
   [doc("d1","Scott Derrickson is an American director.","Scott_Derrickson"),
    doc("d2","Edward Davis Wood Jr. was an American filmmaker.","Ed_Wood")]),
 ("fever_supports","claim_only_rag","Verify: Tokyo is the capital of Japan.",
   [doc("d1","Tokyo is the capital city of Japan.","Tokyo")]),
 ("fever_refutes","full_egrag","Verify: The Great Wall of China is in Egypt.",
   [doc("d1","The Great Wall of China is located in northern China.","Great_Wall")]),
 ("insufficient","claim_only_rag","What year did the fictional company Zorbix go bankrupt?",
   [doc("d1","Zorbix is a company mentioned in passing with no further detail.","Zorbix")]),
]
settings=RunSettings(top_k=5, evidence_token_budget=256, reserved_output_tokens=64, chunk_size=256, chunk_overlap=32)
cfg=GenerationConfig(deterministic=True, temperature=0.0, seed=0, max_new_tokens=64)
gen=Recording(MODEL, context_limit=4096)

def run_one(name, variant, q, docs):
    gen.last_raw=None
    t=time.time()
    try:
        out=run_system(get_variant(variant), Query(query_id=name,text=q), docs, generator=gen, config=cfg, settings=settings)
        rec={"parse_ok":True,"error":None,"answer":out.answer,"abstained":out.abstained,
             "cited":list(out.cited_claim_ids),
             "invalid_citations":[c for c in out.cited_claim_ids if c not in out.known_claim_ids]}
    except Exception as e:
        rec={"parse_ok":False,"error":f"{type(e).__name__}: {e}","answer":None}
    rec.update({"name":name,"variant":variant,"latency_ms":round(1000*(time.time()-t)),
                "raw_output":(gen.last_raw[:400] if gen.last_raw else None)})
    return rec

t0=time.time()
results=[run_one(*c) for c in CASES]
# determinism on first runnable
det=None
for c in CASES:
    r1=run_one(*c)
    if r1["raw_output"] is not None:
        r2=run_one(*c)
        det={"case":c[0],"raw1":r1["raw_output"][:120],"identical_raw":r1["raw_output"]==r2["raw_output"]}
        break
total=round(time.time()-t0,1)
n=len(results); ok=sum(1 for r in results if r["parse_ok"])
import torch, transformers
print(json.dumps({
 "model":MODEL,"backend":"transformers.pipeline(text-generation), no chat template","device":"cpu",
 "torch":torch.__version__,"transformers":transformers.__version__,
 "decoding":{"deterministic":True,"temperature":0.0,"seed":0,"max_new_tokens":64},
 "context_limit":4096,"evidence_budget":256,"prompt_renderer":"PlainTextEvidenceRenderer",
 "n_cases":n,"valid_structured_output":ok,"valid_structured_output_rate":round(ok/n,3),
 "malformed_output":n-ok,
 "gen_latency_ms_median":sorted(r["latency_ms"] for r in results)[n//2],
 "determinism":det,"total_seconds_incl_load":total,"cases":results,
}, indent=2))
