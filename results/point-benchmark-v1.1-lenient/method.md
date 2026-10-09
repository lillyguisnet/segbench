# Point-finding pilot v1

A temporary point-localisation benchmark while object masks are unfinished.
Do not call these segmentation scores or point-inside-object scores. No
model consensus participates in judging answers after the human reference
export has been approved. Human-reviewed consensus remains susceptible to
anchoring and shared omissions; a separate annotation audit is still needed.

## Reference and scope

Reference: `annotations/reviews/segbench-review-6187f623-approved.json`.
Only objects with status `accepted` are targets: 83 logs, 11 cows, 33 fig
leaves, 26 dishes (19 clean, 7 dirty). Rejected and hidden pending proposals
are not targets. Image hashes and dimensions must match the exported source.
Duplicate reference coordinates are an error, not merged automatically.

Use only original `find` calls. SAM 3 and YOLOE's mask-derived interior
points count as their finder output; SAM 1/2 pairs do not enter because they
reuse an existing finder's answer. Red foliage and road are region tasks and
are excluded. There is no hinted-outlining track in this pilot.

## Versioned metric, set before running this scorer

Compute distances in original image pixels, retaining the photo's aspect
ratio. A prediction's [x,y] coordinates are scaled from 0..1000; no automatic
swaps, clipping, snapping or correction from the reference. Out-of-photo
predictions remain extras, never matches.

For reference object j, let d_j be its distance to the nearest other approved
point, and D the image diagonal. Use three tolerances:

- tight: r_j = min(0.10 d_j, 0.02 D)
- medium: r_j = min(0.25 d_j, 0.02 D)
- loose: r_j = min(0.50 d_j, 0.02 D)

For a task with one reference point, use d_j = D. All thresholds are
engineering choices, not measured object sizes. Local spacing limits
confusion in crowded scenes; the 2% diagonal cap prevents an isolated object
from accepting a point arbitrarily far away. Large-object points can still
be unfairly rejected, and background points can be accepted. This is the
explicit cost of doing point-only evaluation before masks exist.

At each tolerance, a predicted point may match a reference if their distance
is <= r_j. Choose the maximum-cardinality one-to-one matching. A greedy
nearest-first assignment is not sufficient: it can reduce the number of
valid matches and depend on answer order. Each reference and prediction
counts at most once. For dishes, edges also require the same dirty/clean
label. A wrong label therefore causes both a false positive and a false
negative in labelled detection. Report unlabelled location F1 separately
as a diagnostic, not as the dishes task's primary score.

TP = matched objects, FP = unmatched predictions, FN = unmatched references.
Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN).

Task score = arithmetic mean of F1 at the three tolerances. Four-task score
= equal mean of task scores (not pooled objects, which would overweight logs).
Also publish each tolerance's separate score. The mean is NOT a probability,
not an integral over all tolerances, and not proof of robust ranking outside
these chosen tolerances. Report maximum rank movement across the three
settings for complete entrants. Tolerance failures may reflect within-object
point placement, not object-finding failures.

Count error is prediction count minus reference count, reported separately,
with per-label counts for dishes. A correct count cannot cancel incorrect
locations. A malformed or failed response scores zero; its predicted count
is unknown, not zero. For parsing, accept a JSON object with an optional code
fence/prose wrapper, reject duplicate keys, non-finite numbers, booleans as
coordinates, missing point/label fields, or non-list object arrays. Existing
out-of-range finite points remain readable but cannot match.

## Coverage, time, cost and uncertainty

Only complete four-task entrants get a headline overall score or appear in
the overall scatterplot. RF-DETR's cow-only result is shown with coverage
1/4 and no comparable overall mean. Missing tasks are not silently averaged
away. This differs from the broad-suite zero-for-unsupported convention in
the older chart plan: here coverage is separated from measured ability.

Aggregate quality by task, then across tasks. Report median call latency and
mean estimated cost per call for these four find tasks only (not trick,
whole, or warm-up calls). Include failed calls' known latency/cost; missing
cost/time makes the aggregate unknown, not free/instantaneous. Specialists
reuse one inference result for find/whole; don't count both as finder work.

Specialist times are flagged for shared GPU occupancy, and their costs
extrapolate load factors from earlier smoke tests. They may be displayed
with an explicit provisional marker, but cannot establish a definitive
quality/cost/speed frontier. API costs are list-price equivalents, including
subscription-backed calls; not all were cash charges. This pilot is one
medium-thinking answer per language-model/task, so it provides no measured
repeat uncertainty. No confidence intervals are invented.

JSON/CSV records preserve source file + line, source hashes, reference hash,
metric constants and score diagnostics. Changing the reference or metric
produces a new run folder. Do not overwrite raw replies or reference files.
This six-photo project (four object tasks here) does not support broad claims
of universally superior object finding.
