import json, random
from pathlib import Path
from egrag.experiments.benchmarks import HotpotQADataset, dataset_fingerprint

SEED=20260629
exs = HotpotQADataset(split="validation").load()
full_fp = dataset_fingerprint(exs)

def coverage(e):
    docs={d.source.source_id for d in e.documents}
    gold=set(e.gold_evidence.source_ids)
    p=len(gold&docs)
    return "full" if p==len(gold) and gold else ("none" if p==0 else "partial")

def ctx_len_bucket(e):
    n=len(e.documents)
    return "short" if n<30 else ("mid" if n<50 else "long")

rec={e.example_id:e for e in exs}

def stratified(n, seed):
    rng=random.Random(seed)
    # strata: (type, yesno) with proportional-ish balance; ensure comparison+yesno+bridge represented
    bridge=[e.example_id for e in exs if e.metadata["type"]=="bridge"]
    comp=[e.example_id for e in exs if e.metadata["type"]=="comparison"]
    yesno=[e.example_id for e in exs if e.gold_answers[0].lower() in ("yes","no")]
    for L in (bridge,comp,yesno): rng.shuffle(L)
    # target: ~60% bridge, ~25% comparison, ~15% yes/no (dedup), ensure full-coverage present
    take_b=int(round(n*0.55)); take_c=int(round(n*0.25)); take_y=n-take_b-take_c
    chosen=[]
    seen=set()
    def add(ids,k):
        c=0
        for i in ids:
            if c>=k: break
            if i in seen: continue
            seen.add(i); chosen.append(i); c+=1
    add(bridge,take_b); add(comp,take_c); add(yesno,take_y)
    # top up if dedup short
    pool=[e.example_id for e in exs]; rng.shuffle(pool)
    add(pool, n-len(chosen)+len(chosen))  # fill remaining
    chosen=chosen[:n]
    return chosen

def manifest(ids, name, seed):
    items=[]
    cov=ctype=yn={}
    from collections import Counter
    covc=Counter(); typec=Counter(); ync=0; ctxc=Counter()
    for i in ids:
        e=rec[i]; c=coverage(e)
        covc[c]+=1; typec[e.metadata["type"]]+=1; ctxc[ctx_len_bucket(e)]+=1
        is_yn=e.gold_answers[0].lower() in ("yes","no"); ync+=int(is_yn)
        items.append({"example_id":i,"type":e.metadata["type"],"answer":e.gold_answers[0],
                      "yes_no":is_yn,"gold_pages":list(e.gold_evidence.source_ids),
                      "gold_coverage":c,"num_documents":len(e.documents),"ctx_bucket":ctx_len_bucket(e)})
    full_cov=[it["example_id"] for it in items if it["gold_coverage"]=="full"]
    out={
      "benchmark":"hotpotqa","split":"validation (development)","name":name,
      "size":len(ids),"seed":seed,
      "dataset_fingerprint_sha256":full_fp,
      "selection":"deterministic stratified by type(bridge~55%/comparison~25%)+yes/no~20%; seeded; NEVER by model success",
      "type_distribution":dict(typec),"yes_no":ync,
      "gold_coverage_distribution":dict(covc),
      "context_length_buckets":dict(ctxc),
      "full_coverage_subset_ids":full_cov,
      "full_coverage_subset_note":"Use ONLY this subset for supporting-fact/evidence/bridge-connectivity PROXY metrics; headline answer EM/F1 use all ids.",
      "example_ids":list(ids),
      "examples":items,
    }
    return out

outdir=Path("artifacts/benchmark-calibration/samples")
for n,fname,seed in [(25,"hotpot-smoke-25.json",SEED),(100,"hotpot-dev-100.json",SEED)]:
    ids=stratified(n,seed)
    m=manifest(ids,fname.replace(".json",""),seed)
    (outdir/fname).write_text(json.dumps(m,indent=2))
    print(fname, "size",m["size"],"types",m["type_distribution"],"yesno",m["yes_no"],"cov",m["gold_coverage_distribution"],"fullsub",len(m["full_coverage_subset_ids"]))
