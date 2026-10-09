"""Build the pointing-model input for the point benchmark from the finders runs.

python3 scripts/combine_finders.py  ->  results/run-finders-combined.jsonl

- LocateAnything: run finders-2 (protocol v2: plural wording).
- MolmoPoint, Rex-Omni: run finders-2b (v2; full GPU, SAM 3.1 helper paused).
- Florence-2 in BOTH modes, as two entrants, because no documentation says
  which of its text modes means "every object of this kind", and picking one
  after seeing both scores would be choosing by result:
    florence2-large-grounding  caption phrase grounding (finders-1, v1)
    florence2-large-ovd        open-vocabulary detection (finders-2b, v2)
Records are copied unchanged except "model" (Florence) and a "derived_from"
note naming the source file and line. Raw runs stay untouched.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PICK = [("run-finders-2.jsonl", {"locateanything-3b": "locateanything-3b"}),
        ("run-finders-2b.jsonl", {"molmopoint-8b": "molmopoint-8b", "rexomni-3b": "rexomni-3b",
                                  "florence2-large": "florence2-large-ovd"}),
        ("run-finders-1.jsonl", {"florence2-large": "florence2-large-grounding"})]
out = []
for name, keep in PICK:
    for i, line in enumerate((ROOT / "results" / name).read_text().splitlines(), 1):
        r = json.loads(line)
        if r["model"] in keep:
            r["derived_from"] = f"results/{name}:{i}"
            r["model"] = keep[r["model"]]
            out.append(r)
(ROOT / "results" / "run-finders-combined.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), "records")
