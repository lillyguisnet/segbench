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

### Does merging change it? (2026-10-16, `scripts/jev_merge_variants.py`, no new requests)

Five ways to merge the same run's boxes, written down before scoring any
(`results/point-benchmark-v3-jev-merge-*`):

| variant | score | logs boxes (83) | mushroom boxes (14) | cost / photo |
|---|---|---|---|---|
| parts (nothing joined) | 22.7 | 112 | 43-47 | $0.006 |
| joined across cuts, no recheck | **23.5** | 44 | 16-19 | $0.006 |
| final (as run: joined + recheck) | 23.4 | 215-256 | 29-40 | $0.018 |
| parts, overlapping boxes merged | 22.7 | 112 | 43-47 | $0.006 |
| joined, then overlapping merged | 21.0 | 20-28 | 4-13 | $0.006 |

- The recheck is what multiplied the boxes (logs 44 -> 256): it tripled the
  cost and changed the score by -0.1.
- Merging moves the score by at most ~1 point either way. On the logs, joining
  trades recall for precision (it merges touching neighbours: recall at the
  loose tolerance 0.59 -> 0.35, precision 0.44 -> 0.66).
- The limit is before any merging: with nothing joined, only 4 of ~43
  mushroom boxes and 5 of 36 bird boxes have their middle near a real one.
  Jev's per-piece boxes land on hands, leaves and between objects.
- "overlap" equals "parts": boxes from different pieces stay inside their own
  pieces, so they never overlap by half.

### Merging boxes whose sides touch (2026-10-16, `scripts/jev_touch_merge.py`)

Maxime: the pieces never overlap, so merge when a share of a box's side
touches another box. Eight rules, written before scoring: sides facing each
other within 0.5% of the photo, shared length >= 25/50/75/90% of the shorter
side ("either") or of the longer one ("both"), grouped; on the halving's boxes
(no recheck). Scores 21.1 to 22.5 (`results/point-benchmark-v3-jev-touch-*`),
all below the run's own join across cut lines (23.5) and around "nothing
joined" (22.7). Loose rules chain touching neighbours (27 boxes for 83 logs);
strict ones leave most parts apart (65 for 83).

### Why the left plate on the dishes photo kept two boxes (2026-10-16)

Maxime spotted it. The halving cut that plate into four parts (cut lines at
x = 250 and 500, y = 1126). Three were joined. The part between x = 250 and
500 ends at x = 492, 8 px short of the cut; the run's join allows 2% of the
piece, 5 px for a 250 px piece, so it stayed apart from the part right of
x = 500. Picture: `results/point-benchmark-v3-jev-halving-view/dishes-left-plate.jpg`.
The touch merges join it ("either" at 25-90%, "both" up to 75%). The run's
join with a tolerance of 0.5% of the photo (20 px) also joins it, and its
middle then lands on the plate's answer-key dot, but it chains other dishes
into large boxes too: 20.3 overall (dishes 7.8) against 23.5 (dishes 16.5)
(`scripts/jev_join_photo_tol.py`, `results/point-benchmark-v3-jev-join-cut-photo-tol`;
exploratory, found by looking at a test photo).

### Pens on a floor (2026-10-16, Maxime's photo from the chat)

960x540, objects well apart: three pens (blue, grey-copper, purple), a
yellow highlighter, a white remote with a stylus, an eraser, chewing gum, a
cream tube. `segbench/jev_halving` unchanged, "each pen", 3 tries
(`results/probe-jev-pens/`): all three pens boxed tightly, nothing else, the
same in every try; 10 requests, about 1 s, $0.0004 per try. The purple pen's
tip crossed a cut line: the piece above answered "one" for the tip and the
join put the two parts back together.

### Beetles with up to 15 halvings (2026-10-16, `jev-halving-beetles-depth15`)

Maxime: run only the butterflies-and-beetles photo, allowing 15 halvings
instead of 7. `scripts/run_jev_halving.py --max-depth 15` (option added in
a65da77; the search is otherwise unchanged), 3 tries, scored like v3
(`results/point-benchmark-v3-jev-beetles-depth15`); pictures with every
cut: `scripts/view_jev_halving_cuts.py` ->
`results/point-benchmark-v3-jev-beetles-depth15-view/beetles-cuts.jpg`
(the depth-7 run: `results/point-benchmark-v3-jev-halving-view/beetles-cuts.jpg`).

| | depth 7 (jev-halving-1) | depth 15 |
|---|---|---|
| score (3 tries) | 33.6 (32, 32, 37) | **56.7** (61, 55, 54) |
| beetles found, medium tolerance (of 22) | 8, 8, 9 | 21, 19, 18 |
| boxes | 26, 24, 23 | 36, 37, 35 |
| deepest cut reached | 7 (6 pieces forced) | 10, 11, 12 (none forced) |
| requests, seconds, cost per try | 52-69, 1.6 s, $0.002 | 115-126, 3.4-5.4 s, $0.004 |

- The limit of 15 was never reached; 7 was too low for this photo. About half
  the parts (24-27 of 46-50) come from pieces cut 8-12 times.
- Part of the gain is chance: the piece left of centre in the top half
  ([0, 329, 371, 659]) is near 50/50 between "one" and "several" (0.53 "one"
  in the depth-7 try 0; 0.37-0.58 here); when "one", its beetles are never
  looked for. All three new tries still beat all three old ones.
- Still wrong: the top-right quarter answers "one" (0.70-0.75) in every try,
  so its corner beetle is never found; boxes on the light switch and on
  butterflies (15-18 extra boxes per try).
- 56.7 on beetles would be 24th of 29 on this task in v3 (Gemini 3.1 Pro
  66.7, DeepSeek Flash 39.5; the leaders 98-100). Only this photo was run:
  the eleven-photo score at depth 15 is unknown, and the logs photo, which
  never stopped splitting at depth 7, could cost far more requests.

### Feedback loop on the boxes (2026-10-16, `jev-halving-beetles-depth15-verify`)

Maxime: once Jev gives a box, ask back whether it is on a beetle; no -> find
again; no a second time -> reject; yes -> keep. `segbench/jev_verify.py`
(7b59813), `--verify` in `scripts/run_jev_halving.py`, on the final boxes of
the halving at depth 15:
- Check: the box plus 25% margin on each side, cut from the photo; yes / no
  "Is there a beetle (not the butterflies) in the middle of the image (whole,
  or most of it)?"; yes at p >= 0.5.
- Find again: an identical request gives (almost) the same box (three
  identical requests, boxes within 0.03), so the halving piece holding the
  box's middle is sent again with the rejected box painted grey, with the
  halving's questions; "none" -> reject, else the new box is checked once.
  A new box whose middle is inside a kept box is dropped.
- Tried on the synthetic ovals first. One correct oval was rejected (p 0.37-0.44
  under every margin and wording tried): in the square picture it is a circle.
  Margin 25% kept the blue square out (at 100% the plain wordings said yes, 0.8).
  Nothing was changed after that. The loop was designed after looking at this
  photo, so the numbers below are exploratory.

Same halving boxes scored before and after the check (the run saves both:
`results/run-jev-halving-beetles-depth15-verify-before-check.jsonl`):

| | before the check | after |
|---|---|---|
| score (3 tries) | 52.7 (55, 54, 50) | **65.4** (71, 65, 60) |
| beetles found, medium tolerance (of 22) | 18, 16, 16 | 17, 14, 14 |
| extra points, medium tolerance | 21, 14, 17 | 6, 4, 5 |
| boxes | 39, 30, 33 | 23, 18, 19 |
| requests per try | ~115-150 | 150-204 ($0.005-0.006, ~4 s) |

- What the check did (a box counted "on a beetle" if an answer-key point is
  inside it): 41 boxes on no beetle rejected over the three tries, 12 kept;
  one box on a beetle rejected (p 0.41).
- Finding again added nothing: 3 replacements in all, none on a beetle (one
  dropped as a duplicate). The gain is all from rejecting.
- Recall drops more than "one box on a beetle rejected" suggests: some boxes
  rejected were near a beetle (counted at the medium tolerance) but not on it.
- This halving run (52.7 before the check) scored below the earlier depth-15
  run (56.7): run-to-run variation in Jev's answers.
- Pictures: `scripts/view_jev_verify.py` ->
  `results/point-benchmark-v3-jev-beetles-depth15-verify-view/beetles-verify.jpg`.
