"""Count repetition loops ("doom loops") in every Perceptron Mk1.5 benchmark call.

Usage: uv run python scripts/mk15_loops.py [-v]

A call counts as looping when its thinking or its answer shows any of:
  A. one line written over and over: the same line (numbers masked) more than
     max(20, 3 x the number of real objects) times;
  B. the same coordinate pair written 20+ times;
  C. a "grid march": 12+ positions in a row with one axis fixed and the other
     moving by the same step (points laid on a ruler, not on objects; boxes are
     checked corner by corner);
  D. an answer with more than max(20, 3 x the real objects) points;
  E. going round in circles: 100+ lines without numbers in which fewer than 40%
     are new.
"Ran out of room" = 7,500+ output tokens (the cap is 8,192; on 2026-10-15
Perceptron cut the thinking near 7,700 tokens and forced a short answer).
"""
import collections, json, re, sys

RUNS = [
    ("4 photo tasks", "v2", "medium", "results/run-points-v2.jsonl"),
    ("4 photo tasks", "v2", "min", "results/run-points-v2.jsonl"),
    ("4 photo tasks", "v1-hd", "medium", "results/run-points-v1hd.jsonl"),
    ("4 photo tasks", "tags", "medium", "results/run-points-tags.jsonl"),
    ("7 new photo tasks", "v2", "medium", "results/run-points-v2-b2.jsonl"),
    ("7 new photo tasks", "v1-hd", "medium", "results/run-points-v1hd-b2.jsonl"),
    ("screens round 2", "screen-v1", "medium", "results/run-screens-2.jsonl"),
    ("screens round 2", "screen-tags", "medium", "results/run-screens-2-tags.jsonl"),
]
# reviewed reference points per photo task (results/point-benchmark-v3/per-task.csv)
OBJECTS = {"logs": 83, "cows": 11, "fig": 33, "dishes": 26, "ants": 13, "beetles": 22, "birds": 16,
           "mushrooms": 14, "jars": 6, "pinkleaves": 37, "whitecow": 1}
PAIR = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)|\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]|\"x\"\s*:\s*(-?\d+)\s*,\s*\"y\"\s*:\s*(-?\d+)")


def lines(text):
    return [x.strip().lower() for x in re.split(r"[\n.!?]+|\}\s*,\s*\{|</point>|</point_box>", text) if len(x.strip()) > 3]


def pairs(text):
    return [tuple(int(v) for v in m if v) for m in PAIR.findall(text)]


def march(ps):
    best = 0
    for stride in (1, 2):
        seq = ps[::stride] if stride == 1 else ps[0::2]
        run = 0
        for i in range(2, len(seq)):
            (x0, y0), (x1, y1), (x2, y2) = seq[i - 2], seq[i - 1], seq[i]
            ok = (y0 == y1 == y2 and x1 != x0 and abs((x2 - x1) - (x1 - x0)) <= 1) or \
                 (x0 == x1 == x2 and y1 != y0 and abs((y2 - y1) - (y1 - y0)) <= 1)
            run = run + 1 if ok else 0
            best = max(best, run + 2 if run else 0)
    return best


def check(text, limit):
    ls = lines(text)
    masked = collections.Counter(re.sub(r"-?\d+", "#", x) for x in ls)
    plain = [x for x in ls if not re.search(r"\d", x)]
    ps = pairs(text)
    why = []
    if masked and masked.most_common(1)[0][1] > limit:
        line, n = masked.most_common(1)[0]
        why.append(f"A: {n}x '{line[:60]}'")
    same = collections.Counter(ps).most_common(1)
    if same and same[0][1] >= 20:
        why.append(f"B: {same[0][1]}x {same[0][0]}")
    m = march(ps)
    if m >= 12:
        why.append(f"C: march of {m}")
    if len(plain) >= 100 and len(set(plain)) / len(plain) < 0.4:
        why.append(f"E: {len(set(plain))}/{len(plain)} lines new")
    return why


def classify(r):
    parts = ((r.get("response") or {}).get("message") or {}).get("parts") or []
    think = "".join(p.get("text", "") for p in parts if p.get("type") == "thinking")
    answer = r.get("text") or ""
    real = 0 if r["track"] == "trick" else OBJECTS.get(r["task"], 10)
    limit = max(20, 3 * real)
    why = [f"thinking {w}" for w in check(think, limit)] + [f"answer {w}" for w in check(answer, limit)]
    got = r.get("answer", {}).get("items") if isinstance(r.get("answer"), dict) else r.get("clicks")
    if got and len(got) > limit:
        why.append(f"D: {len(got)} points for {real} objects")
    out = (r.get("usage") or {}).get("output_tokens") or 0
    return why, out >= 7500, out


def main(verbose=False):
    print(f"{'set':18} {'prompt':11} {'thinking':8} {'calls':>5} {'loops':>5} {'out of room':>11} {'stopped itself':>14} {'out of room, no loop':>20}")
    for group, version, level, path in RUNS:
        calls = [r for r in map(json.loads, open(path))
                 if r.get("model") == "mk1.5" and r.get("prompt_version") == version and r.get("level") == level]
        t = collections.Counter()
        for r in calls:
            why, full, out = classify(r)
            t["calls"] += 1
            t["loop"] += bool(why)
            t["loop_full"] += bool(why) and full
            t["full_only"] += full and not why
            if verbose and (why or full):
                print(f"  {path.split('/')[-1]:26} {version:11} {level:6} {r['task']:16} {r['track']:6} try{r['repeat']} "
                      f"out={out:5} {'FULL' if full else '    '} {'; '.join(why) or 'NO LOOP FOUND'}")
        print(f"{group:18} {version:11} {level:8} {t['calls']:5} {t['loop']:5} {t['loop_full']:11} "
              f"{t['loop'] - t['loop_full']:14} {t['full_only']:20}")


if __name__ == "__main__":
    main("-v" in sys.argv)
