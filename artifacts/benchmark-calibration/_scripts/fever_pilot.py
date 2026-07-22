import json, time
from pathlib import Path
from statistics import mean
from egrag.domain.models import Query
from egrag.generation import FakeTextGenerator, GenerationConfig
from egrag.experiments.benchmarks import FeverGoldEvidenceDataset
from egrag.experiments.variants import RunSettings, get_variant, run_system

sample=json.loads(Path("artifacts/benchmark-calibration/samples/fever-smoke-25.json").read_text())
want=set(sample["example_ids"])
allex=FeverGoldEvidenceDataset(split="valid").load()
byid={e.example_id:e for e in allex}
exs=[byid[i] for i in sample["example_ids"] if i in byid]

VARIANTS=["passage_rag","reranked_passage_rag","claim_only_rag","graph_no_propagation",
          "graph_with_propagation","graph_top_claim","graph_coherent_subgraph",
          "graph_no_temporal","graph_no_contradiction","full_egrag"]
gen=FakeTextGenerator()
cfg=GenerationConfig(deterministic=True, seed=0, max_new_tokens=64)

def prf(pred, gold):
    pred=set(pred); gold=set(gold)
    if not gold: return None,None,None
    tp=len(pred&gold)
    p=tp/len(pred) if pred else (1.0 if not gold else 0.0)
    r=tp/len(gold)
    f=2*p*r/(p+r) if (p+r) else 0.0
    return p,r,f

def run(budget):
    out={}
    for vname in VARIANTS:
        v=get_variant(vname)
        settings=RunSettings(top_k=5, evidence_token_budget=budget, reserved_output_tokens=16,
                             chunk_size=256, chunk_overlap=0)
        cp=cr=ep=er=[];  # placeholder
        cprec=[];crec=[];eprec=[];erec=[];nodes=[];edges=[];contra=[];nsel=[];lat=[];valid=0;n=0
        for e in exs:
            if not e.gold_evidence.has_evidence:  # skip NEI (no gold pages)
                continue
            n+=1
            gold=e.gold_evidence.source_ids
            t=time.time()
            try:
                r=run_system(v, Query(query_id=e.example_id, text=e.question), list(e.documents),
                             generator=gen, config=cfg, settings=settings)
                valid+= (0 if r.abstained and not r.cited_source_ids else 1)
                _,cr_,cf=prf(r.cited_source_ids, gold); cprec.append(prf(r.cited_source_ids,gold)[0] or 0.0); crec.append(cr_ or 0.0)
                ep_,er_,ef=prf(r.selected_source_ids, gold); eprec.append(ep_ or 0.0); erec.append(er_ or 0.0)
                nodes.append(r.counts.get("num_graph_nodes",0)); edges.append(r.counts.get("num_graph_edges",0))
                contra.append(r.counts.get("contradiction_edges", r.counts.get("num_graph_edges",0) if False else 0))
                nsel.append(r.counts.get("num_selected",0))
            except Exception as ex:
                cprec.append(0.0);crec.append(0.0);eprec.append(0.0);erec.append(0.0)
            lat.append(round(1000*(time.time()-t),1))
        out[vname]={"n_scored":n,
            "citation_precision":round(mean(cprec),3) if cprec else None,
            "citation_recall":round(mean(crec),3) if crec else None,
            "evidence_precision":round(mean(eprec),3) if eprec else None,
            "evidence_recall":round(mean(erec),3) if erec else None,
            "avg_graph_nodes":round(mean(nodes),1) if nodes else 0,
            "avg_graph_edges":round(mean(edges),1) if edges else 0,
            "avg_selected":round(mean(nsel),1) if nsel else 0,
            "median_latency_ms":sorted(lat)[len(lat)//2] if lat else None}
    return out

t0=time.time()
res={"budget_256":run(256),"budget_512":run(512)}
res["meta"]={"sample":"fever-smoke-25","n_examples_total":len(exs),
  "n_with_gold_pages":sum(1 for e in exs if e.gold_evidence.has_evidence),
  "generator":"FakeTextGenerator (deterministic; valid structured output)",
  "note":"FEVER gold-evidence setting: documents ARE the gold sentences (no distractors). NEI examples have no gold pages and are excluded from evidence metrics. This measures evidence selection + graph structure, NOT answer/label accuracy (which needs a usable real generator).",
  "wall_seconds":round(time.time()-t0,1)}
print(json.dumps(res, indent=2))
