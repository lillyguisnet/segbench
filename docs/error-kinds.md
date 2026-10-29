# Kinds of mistakes on the eleven photos (2026-10-19)

Asked by Maxime: sort every mistake behind the photo charts (point benchmark
v3, box benchmark v1) into kinds. Diagnostic only: no model calls, no score
changes.

    (cd specialists/sam && uv run key_masks.py)  # outlines of the key objects
    uv run scripts/find_lookalikes.py            (SAM 3 outlines of look-alikes)
    uv run --with numpy scripts/classify_errors.py                     -> results/error-kinds/v1/
    uv run --with numpy --with pillow scripts/draw_error_kinds.py --examples results/error-kinds/v1
    uv run scripts/plot_error_kinds.py           -> charts/error-kinds-v1/ (overview, families, families-points, families-points-correct, families-boxes, families-boxes-correct, by-model; index.html with the examples)

The kinds and their order are in the header of `scripts/classify_errors.py`.
Outputs: `summary.md` (counts, and the score each kind costs each model),
`points/` and `boxes/` (`items.csv`: every point or box and every unfound
object with its kind; `replies.csv`; `by-model.csv`), `examples/` (pictures).

## Decisions and why

- **Very far points are false positives, sorted by what is under them, not
  by distance.** First pass (2026-10-15, distance only) showed that "far"
  mixes four things: points on another thing (the black cows for "the white
  cow"), points on the right object but far from its dot (big jars), whole
  answers shifted (Terra's cows along the treeline), and points on nothing.
  Distance cut-offs had no natural gap, so they were dropped.
- **"On the object" is decided by an outline**, not the box: SAM 2.1 Large,
  prompted with each reviewed answer-key box (and the object's dot), cut to
  the box (`results/error-kinds/support/key-masks-sam2.1.npz`). A box alone
  would call the sky between a bird's wings "on the bird". SAM 2.1 because
  it is not an entrant. Not reviewed by a person.
- **Look-alikes** ("wrong object") come from SAM 3 with phrases fixed before
  the run (`scripts/find_lookalikes.py`: the questions' own exclusions and the
  broader kind of thing), minus anything mostly inside a key box; plus spots
  where 3+ other models made an off-object mistake ("shared": something there
  fools models, e.g. the plant by the window on the fig photo, which SAM 3's
  "leaf" missed). SAM 3 is an entrant; here it only names other things on the
  photo and never decides what is right.
- **Mix-ups and shifts** are reported only when the other reading raises F1
  (widest ring / 50% overlap) by 0.25 or more and at least doubles it; the
  shift-and-stretch fit (four numbers) only for 6+ answers mostly off every
  outline, so an ordinary answer cannot be "fitted" into a better score.
- **Group boxes** (one box half-covering 2+ objects) are a box kind; the
  objects they swallowed count as "group box", not "missed".
- **Score each kind costs** follows TIDE (Bolya et al. 2020): fix that one
  kind in each reply, rescore exactly as charted, average like the chart.

- **Square box charts (2026-10-21, asked by Maxime: the same as the point
  ones):** `families-boxes-correct` and `families-boxes`, same four colours,
  models sorted by share of correct boxes. "Correct" is the loosest level
  (50% overlap); the score averages 50/75/90/95, so the green share is far
  more lenient than the score for boxes than for points (GPT-6 Astra 86%
  green, score 57; 35% of its correct boxes also pass 90%). Objects swallowed
  by a group box are orange with the box, not blue (as in `families`).

## Limits

- Outlines and look-alikes are machine-made and unreviewed; a few points
  near an object's edge can land on the wrong side of a rough outline.
- The order of the rules decides mixed cases (a point both on a look-alike
  and near an unfound object is "near miss" only if it is not on the
  look-alike).
- The logs' answer key was checked over the whole photo (Maxime confirmed,
  2026-10-19; the review file's `full_photo_checked` flag was never set and
  is left as it is, since score manifests hold that file's hash). SAM 3
  "cut log end" outlines outside the key are therefore look-alikes: points on
  them are "wrong object". "Maybe missing from key" is left for points on the
  one fig leaf drawn by hand in the box review, which the point key lacks.
- Kinds are tuned on these eleven photos.
