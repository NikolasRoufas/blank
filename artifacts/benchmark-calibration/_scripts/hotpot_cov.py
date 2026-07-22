import json
from collections import Counter
from egrag.experiments.benchmarks import HotpotQADataset
exs = HotpotQADataset(split="validation").load()
full=part=none=0
cov_by_type={"bridge":Counter(),"comparison":Counter()}
for e in exs:
    doc_sources={d.source.source_id for d in e.documents}
    gold=set(e.gold_evidence.source_ids)
    present=len(gold & doc_sources)
    if present==len(gold): full+=1; k="full"
    elif present==0: none+=1; k="none"
    else: part+=1; k="partial"
    cov_by_type[e.metadata["type"]][k]+=1
n=len(exs)
print(json.dumps({
 "n":n,
 "all_gold_pages_present": full, "all_gold_pct": round(100*full/n,1),
 "partial_coverage": part, "no_gold_page_present": none,
 "by_type": {t:dict(c) for t,c in cov_by_type.items()},
 "note":"fullwiki split: context is retrieved paragraphs; gold supporting paragraphs not guaranteed present"
}, indent=2))
