import json, time
from egrag.domain.models import AtomicClaim, ClaimProvenance, SourceMetadata, SourceSpan
from egrag.graph.classification import HuggingFaceNLIClassifier
from egrag.graph.types import ClaimPair
from egrag.graph.nli import (validate_label_mapping, classify_directional,
                             LABEL_VALIDATION_CASES, LABEL_MAPPING_VERSION)

MODEL="roberta-large-mnli"; REV="2a8f12d27941090092df78e4ba6f0928eb5eac98"
def claim(t,cid):
    return AtomicClaim(claim_id=cid,text=t,
        provenance=ClaimProvenance(source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid,start=0,end=len(t),text=t),)),
        extraction_confidence=1.0)

t0=time.time()
clf=HuggingFaceNLIClassifier(MODEL,model_revision=REV,device=None,max_length=256,batch_size=16)
# 1) label-mapping validation (hard guard)
issues=validate_label_mapping(clf, raise_on_error=False)
load_and_validate_s=round(time.time()-t0,1)

# 2) controlled directional cases (entailment/contradiction/neutral) + a few FEVER-style pairs
cases=[
 ("supports","Aleksandr Danilovich Aleksandrov was a Soviet mathematician.","Aleksandr Aleksandrov worked in mathematics."),
 ("contradicts","The Eiffel Tower is located in Paris.","The Eiffel Tower is located in Berlin."),
 ("neutral","Marie Curie won a Nobel Prize.","Paris is the capital of France."),
 ("duplicate","Google acquired DeepMind in 2014.","DeepMind was acquired by Google in 2014."),
]
t1=time.time(); rows=[]
for exp,a,b in cases:
    d=classify_directional(clf, claim(a,"a"), claim(b,"b"),
        entailment_threshold=0.4, contradiction_threshold=0.7, duplicate_threshold=0.8)
    rows.append({"expected_family":exp,"relation":d.relation,
                 "entail_ab":round(d.entailment_ab,3),"entail_ba":round(d.entailment_ba,3),
                 "contradiction":round(d.contradiction,3),"source_first":d.source_first})
infer_s=round(time.time()-t1,2)
per_pair_ms=round(1000*infer_s/(len(cases)*2),1)  # 2 directions each

import torch, transformers
print(json.dumps({
 "model":MODEL,"revision":REV,"device":"cpu",
 "torch":torch.__version__,"transformers":transformers.__version__,
 "label_mapping_version":LABEL_MAPPING_VERSION,
 "label_mapping_issues":issues,"label_mapping_valid":not issues,
 "load_plus_validate_seconds":load_and_validate_s,
 "directional_infer_seconds":infer_s,"approx_ms_per_single_direction":per_pair_ms,
 "batch_size":16,"max_length":256,"truncation":True,
 "cases":rows,
}, indent=2))
