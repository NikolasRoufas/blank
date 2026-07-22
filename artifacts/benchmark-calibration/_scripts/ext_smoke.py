import json, time
from egrag.domain.models import Document, Query, SourceMetadata
from egrag.adapters.retrieval import SentenceAwareChunker, prepare_passages
from egrag.adapters.extraction.huggingface import HuggingFaceStructuredModel
from egrag.adapters.extraction.structured import StructuredClaimExtractor
from egrag.adapters.extraction.interfaces import ExtractionConfig

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
def passage(text, sid):
    d=Document(document_id=sid, text=text, source=SourceMetadata(source_id=sid, title=sid.replace("_"," ")))
    ps=prepare_passages([d], SentenceAwareChunker(chunk_size=512, overlap=0))
    return ps[0], d.source

CASES=[
 ("direct","Marie Curie was a Polish and naturalized-French physicist.","Marie_Curie",
   {"keywords":["curie","polish","physicist"]}),
 ("multi_fact","The Eiffel Tower is located in Paris. Paris is the capital of France.","Eiffel",
   {"keywords":["eiffel","paris","france"],"min_claims":2}),
 ("negation","The treaty was not ratified by the senate in 1920.","Treaty",
   {"keywords":["treaty","ratified"],"negation":True}),
 ("temporal","Google acquired DeepMind in 2014.","DeepMind",
   {"keywords":["google","deepmind","2014"],"temporal":True}),
]
model=HuggingFaceStructuredModel(MODEL, max_new_tokens=256)
ext=StructuredClaimExtractor(model, prompt_version="extraction_v1", seed=0, deterministic=True)
cfg=ExtractionConfig()

t0=time.time(); rows=[]
for name,text,sid,exp in CASES:
    p,src=passage(text,sid)
    t=time.time()
    try:
        res=ext.extract_result(p, query=None, config=cfg, source=src)
        claims=[{"text":c.text,"conf":round(c.extraction_confidence,3),
                 "entities":list(c.semantics.named_entities) if c.semantics else [],
                 "negation":(c.semantics.negation if c.semantics else None),
                 "temporal":list(c.semantics.temporal_expressions) if c.semantics else [],
                 "span_text":c.provenance.spans[0].text[:80]} for c in res.claims]
        rec={"parse_ok":True,"error":None,"n_claims":len(claims),"claims":claims,
             "warnings":list(res.warnings)[:3]}
    except Exception as e:
        rec={"parse_ok":False,"error":f"{type(e).__name__}: {e}","n_claims":0,"claims":[]}
    rec.update({"name":name,"latency_ms":round(1000*(time.time()-t)),"expectation":exp})
    rows.append(rec)
total=round(time.time()-t0,1)
n=len(rows); ok=sum(1 for r in rows if r["parse_ok"])
grounded=sum(1 for r in rows if r["parse_ok"] and r["n_claims"]>0)
import torch, transformers
print(json.dumps({
 "model":MODEL,"extractor":"StructuredClaimExtractor","prompt_version":"extraction_v1",
 "backend":"transformers.pipeline(text-generation), no chat template","device":"cpu",
 "torch":torch.__version__,"transformers":transformers.__version__,
 "decoding":{"deterministic":True,"seed":0,"max_new_tokens":256},
 "n_cases":n,"valid_json_output":ok,"valid_json_output_rate":round(ok/n,3),
 "cases_with_grounded_claims":grounded,"malformed_output":n-ok,
 "total_seconds_incl_load":total,"cases":rows,
}, indent=2))
