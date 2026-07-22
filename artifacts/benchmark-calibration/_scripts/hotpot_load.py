import glob, hashlib, json, time
from pathlib import Path
from egrag.experiments.benchmarks import HotpotQADataset, dataset_fingerprint, validate_benchmark

t0=time.time()
ds = HotpotQADataset(split="validation")
ppath = ds._parquet_path()
exs = ds.load()
load_s = round(time.time()-t0,2)

# file hash
h = hashlib.sha256(Path(ppath).read_bytes()).hexdigest()
size = Path(ppath).stat().st_size

# integrity
ids = [e.example_id for e in exs]
dup = len(ids)-len(set(ids))
issues = validate_benchmark(exs)

# supporting-fact reference validity: every gold page must appear as a document source_id
bad_sf = 0
empty_sf = 0
for e in exs:
    doc_sources = {d.source.source_id for d in e.documents}
    sp = json.loads(e.metadata["supporting_facts"])
    if not sp:
        empty_sf += 1
    for title, sidx in sp:
        if title not in doc_sources:
            bad_sf += 1
            break

# type/level distribution
from collections import Counter
types = Counter(e.metadata["type"] for e in exs)
levels = Counter(e.metadata["level"] for e in exs)
yesno = sum(1 for e in exs if e.gold_answers[0].lower() in ("yes","no"))
avg_docs = round(sum(len(e.documents) for e in exs)/len(exs),1)
avg_goldpages = round(sum(len(e.gold_evidence.source_ids) for e in exs)/len(exs),2)

fp = dataset_fingerprint(exs)
print(json.dumps({
  "parquet_path": str(ppath),
  "file_size_bytes": size,
  "file_sha256": h,
  "rows": len(exs),
  "load_seconds": load_s,
  "duplicate_ids": dup,
  "validate_issues": len(issues),
  "validate_issues_sample": issues[:3],
  "supporting_fact_unresolved": bad_sf,
  "empty_supporting_facts": empty_sf,
  "type_distribution": dict(types),
  "level_distribution": dict(levels),
  "yes_no_answers": yesno,
  "avg_documents_per_example": avg_docs,
  "avg_gold_pages_per_example": avg_goldpages,
  "dataset_fingerprint_sha256": fp[:24],
  "adapter_version": exs[0].metadata["adapter_version"],
}, indent=2))
