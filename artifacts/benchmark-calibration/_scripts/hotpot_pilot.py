import json, time
from pathlib import Path
from statistics import mean
from egrag.domain.models import Query
from egrag.generation import FakeTextGenerator, GenerationConfig
from egrag.experiments.benchmarks import HotpotQADataset
from egrag.experiments.variants import RunSettings, get_variant, run_system

sample=json.loads(Path("artifacts/benchmark-calibration/samples/hotpot-smoke-25.json").read_text())
ids=sample["example_ids"]; fullcov=set(sample["full_coverage_subset_ids"])
allex=HotpotQADataset(split="validation").load()
byid={e.example_id:e for e in allex}
exs=[byid[i] for i in ids if i in byid]

VARIANTS=["passage_rag","claim_only_rag","graph_no_propagation","graph_with_propagation",
          "graph_top_claim","graph_coherent_subgraph","graph_no_contradiction","full_egrag"]
gen=FakeTextGenerator(); cfg=GenerationConfig(deterministic=True, seed=0, max_new_tokens=64)

def prf(pred, gold):
    pred=set(pred); gold=set(gold)
    if not gold: return None
    tp=len(pred&gold)
    p=tp/len(pred) if pred else 0.0; r=tp/len(gold)
    return (p,r, 2*p*r/(p+r) if (p+r) else 0.0)

def run(top_k):
    out={}
    for vname in VARIANTS:
        v=get_variant(vname)
        s=RunSettings(top_k=top_k, evidence_token_budget=256, reserved_output_tokens=16, chunk_size=256, chunk_overlap=0)
        crec=[];erec=[];eprec=[];nodes=[];edges=[];nsel=[];npass=[];lat=[];gold_in_topk=[]
        for e in exs:
            gold=e.gold_evidence.source_ids
            full=e.example_id in fullcov
            t=time.time()
            r=run_system(v, Query(query_id=e.example_id, text=e.question), list(e.documents),
                         generator=gen, config=cfg, settings=s)
            lat.append(1000*(time.time()-t))
            nodes.append(r.counts.get("num_graph_nodes",0)); edges.append(r.counts.get("num_graph_edges",0))
            nsel.append(r.counts.get("num_selected",0)); npass.append(r.counts.get("num_passages",0))
            # retrieval gold-page coverage among selected sources (proxy; full-coverage subset only)
            if full and gold:
                pe=prf(r.selected_source_ids, gold); ce=prf(r.cited_source_ids, gold)
                if pe: eprec.append(pe[0]); erec.append(pe[1])
                if ce: crec.append(ce[1])
        out[vname]={
          "evidence_precision_fullcov":round(mean(eprec),3) if eprec else None,
          "evidence_recall_fullcov":round(mean(erec),3) if erec else None,
          "citation_recall_fullcov":round(mean(crec),3) if crec else None,
          "avg_graph_nodes":round(mean(nodes),1),"avg_graph_edges":round(mean(edges),1),
          "avg_passages":round(mean(npass),1),"avg_selected":round(mean(nsel),1),
          "median_latency_ms":round(sorted(lat)[len(lat)//2],1),
          "n_fullcov_scored":len(erec)}
    return out

t0=time.time()
res={f"top_k_{k}":run(k) for k in (3,5,8)}
res["meta"]={"sample":"hotpot-smoke-25","n":len(exs),"n_full_coverage":len(fullcov),
  "generator":"FakeTextGenerator (deterministic)","budget":256,
  "note":"HotpotQA fullwiki has distractors (avg 42.5 docs). Evidence/citation metrics on FULL-COVERAGE subset only (gold chain present), clearly a PROXY. Measures evidence-graph structure + selection behavior, NOT answer EM/F1 (needs usable real generator).",
  "wall_seconds":round(time.time()-t0,1)}
print(json.dumps(res, indent=2))
