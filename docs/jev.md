# Jev (TypeSafe's decision model) on pictures, 2026-10-15/16

Asked by Maxime on 2026-10-15: look into Jev's vision release and how it
could join the benchmarks; can it give points, can it return several
objects? Then (2026-10-16): try finding objects by halving the picture.
Everything below is on synthetic pictures only (the seven red ovals and four
blue squares of `scripts/check_boxes.py`); no benchmark photo and no answer
key has been shown to Jev.

## What Jev 1.14 offers (docs.typesafe.ai, read 2026-10-15)

- Released 2026-10-14 as `jev-1.14.0` (alias `jev-preview`; `jev-latest` is
  still 1.13, which reads pictures but has no boxes). Pictures and boxes are
  TypeSafe's beta ("dreamer-ready"): format, answers and limits may change.
- Questions are judgments: choice, yes/no (noul), score, and since 1.14 a
  **box**: one rectangle per question, edges as fractions of the picture,
  given in steps of 0.01, plus a confidence. No points. No "not found": a box
  is always returned (ask a yes/no beside it). One box per question.
- Pictures: PNG or JPEG, never resized by the API. The docs say 256 to 1568 px
  a side; **the API rejects sides under 280** (tried 2026-10-16: 279x500 and
  457x256 rejected, 280x500 and 280x280 accepted).
- Price: $0.042 per million input tokens, output free; a 1568x883 picture is
  about 3,900 tokens (~$0.00016). About 0.15-0.2 s per request.
- Model page still says "Input: Text" for both versions; the Images page
  says both read pictures. The Images page is right (checked).

## Calling it

Through `segbench.call.call` since 2026-10-16 (lm15 1.5.0, model key `jev`,
route `typesafe:jev-1.14.0`): pass `response_format=lm15.judgments(...)` and
`prompt=None`; the record gets `data` (answers) and `probabilities`; a box's
confidence is in `response.provider_data.typesafe.answers.<name>.confidence`.
Jev is in `JUDGMENT_ONLY` (segbench/models.py): the ordinary runs, which ask
for text, never call it. `vision=None` because the circle test asks for
text. TypeSafe has no logo or colour in `segbench/brand.py` yet.

The lm15 upgrade (1.2.1 -> 1.5.0) was checked before use:
`results/lm15-upgrade-1.5.0/` (identical requests for all 124 model x level x
prompt combinations; one live call per service read into the same records;
Gemini not checked live, its key was at its monthly spending cap).

## First probe (`scripts/probe_jev.py`, direct HTTP)

| how | ovals found (of 7) |
|---|---|
| one oval alone, one box | 7 of 7 boxed (IoU 0.55-0.87), middle of the box always on it |
| all seven, one box | one oval, confidence falls to 0.22 |
| none / one / several | "several", certain |
| one box per oval described in words | 4 of 6 (needs to know the objects first: not allowed) |
| find, white out, ask again | 3 |
| find, frame it, ask for an unframed one | 1 (same oval 12 times) |
| quarters, recursively | 4, 53 requests |
| 8x5 grid of yes/no | every cell right, but 7 ovals joined into 2 blobs |

## Halving (`scripts/probe_jev_split.py`)

Ask none / one / several (cut-off ovals count) about a piece, starting with
the whole picture; several -> cut across the longer side and ask both halves;
one -> the box asked in the same request is the object; none -> stop. Every
piece is cut from the full-size picture (halving also zooms in), fitted to
Jev's limits (grey bars for pieces too thin), all pieces of one depth asked
at once. Afterwards, parts of one oval split by a cut are joined (each touches
the same cut line, their spans along it overlap by half or more), and each
joined box is asked again; "several" there halves it once more. All these
choices were tuned on this one synthetic picture.

| run | wide 4000x2252 | tall 2252x4000 | square 1015 | small 743x1319 |
|---|---|---|---|---|
| 2026-10-15 20:34, direct HTTP, `photo` wording | 6 / 83 req | 6 / 80 | 6 / 83 | 6 / 76 |
| 2026-10-15 21:02, through lm15, "the image" | 6 / 85 | 7 / 81 | 7 / 84 | 7 / 59 |

(box middles on a different oval, of 7 / requests.) About 1-2 s of waiting
and $0.002-0.005 per picture. The miss is the pair of overlapping ovals,
boxed as one. Between the two rows both the wording and the run changed;
Jev's answers on identical pieces vary between runs (the same piece "one" at
0.72 and at 0.63), so one run each cannot tell the two apart. Pictures of
the newest run: `results/probe-jev-split-view/` (`scripts/view_jev_split.py`).

## Known limits, before any benchmark photo

- Touching look-alikes (the log ends: ~70 touching circles) are where the
  counting fails; a synthetic picture of touching circles should come first.
- Box edges are in steps of 1% of the piece: on a whole 4000 px photo that is
  40 px, so halving is needed for small objects at strict overlap levels.
- As an entrant it would be "Jev + halving", a search we wrote around the
  model, with every request counted in its cost and time; not "Jev" alone.

## On the benchmark photos (2026-10-16): `jev-halving-1`

Maxime: run Jev with halving on the photos only (no screenshots, no other
model). The search (`segbench/jev_halving.py`, `scripts/run_jev_halving.py`)
was committed (d3c79f7) before any photo was sent; nothing was changed after.
Eleven object tasks of point benchmark v3, 3 tries each, the task's own words
(dishes by location only). 7,285 requests, none failed, $0.27 for all 33
searches. Scored as v3 (`scripts/score_points.py`, lenient, both approved
references, cost and seconds from the first four tasks):
`results/point-benchmark-v3-jev-halving`; pictures (first try):
`results/point-benchmark-v3-jev-halving-view` (`scripts/view_jev_halving.py`).

**23.4%**, 27th of 29 entrants (above YOLOE-26x and RF-DETR Seg, below
Florence-2 Large at 25.8%); $0.018 and 3.9 s per photo on the first four
tasks.

| task | objects | Jev's boxes (3 tries) | score |
|---|---|---|---|
| logs | 83 | 256, 215, 227 | 22.7 |
| cows | 11 | 7, 7, 9 | 40.2 |
| fig | 33 | 53, 44, 47 | 33.6 |
| dishes | 26 | 31, 30, 30 | 12.6 |
| ants | 13 | 17, 16, 12 | 36.5 |
| beetles | 22 | 26, 24, 23 | 33.6 |
| birds | 16 | 31, 26, 22 | 16.7 |
| mushrooms | 14 | 34, 40, 29 | 23.7 |
| jars | 6 | 8, 6, 6 | 16.1 |
| pinkleaves | 37 | 15, 16, 16 | 21.5 |
| whitecow | 1 | 1, 1, 1 | 0.0 |

What the pictures show (first try):
- **Splitting does not stop on crowded photos.** "Cut-off ones count" means
  almost every small piece of the log pile holds part of some log end, so Jev
  keeps answering "several" down to the deepest level, which is then boxed
  anyway: 211 such forced boxes on the first logs try, in clusters on fungus
  and bark. The rule that worked on seven separate ovals is the cause; the
  joining step cannot merge them (they are not parts of one object cut by a line).
- **"One" for pieces that hold several.** Mushrooms: large boxes over two or
  three caps, and boxes on the dead leaves around them.
- **Box edges still too coarse for tiny targets.** White cow: Jev said "one"
  for the whole photo (right), and boxed a spot about 40 px left of the cow
  (its edges come in 1% steps of the 1439 px photo); the middle misses.

Not done: box scores. The box answer key is still waiting for human review
(docs/box-benchmark.md); the boxes are in `answer.items` of each record for
`scripts/score_boxes.py` once it is.
