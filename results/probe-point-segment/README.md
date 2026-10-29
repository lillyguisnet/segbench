# Outline what a red dot is on (probe, 2026-10-16)

Maxime asked for this. A 4 x 4 grid of dots on one photo
(`images/butterfly_display_case.jpg`, 743 x 1319). Each dot was sent in its
own request, with the dot drawn on the photo and its position given in words,
and the model was asked to outline the thing the dot is on.
Script: `scripts/probe_point_segment.py` (the five ways of asking are in its
`PROMPTS`). No answer key and no score. Results were judged by eye, plus one
check that needs no key: does the outline contain its own dot?

Every model ran at medium thinking. Cost is the provider's list price per token
(`prices.json`). GPT-6.1 Sol ran on our ChatGPT plan, so it was not billed.
Gemini 3.8 Flash ran through OpenRouter, pinned to Google AI Studio, because
Google's own key had hit its monthly spending cap. OpenRouter billed $0.652
against our $0.659. That route cannot send Google's highest image-detail setting.

| run | model, prompt | cost for 16 dots | seconds per dot (mean / max) | dot inside its outline |
|---|---|---|---|---|
| gemini-flash-20261016-190813 | Gemini 3.8 Flash, A | $0.659 | 78 / 178 | 15 of 16 (2 outlines scrambled) |
| sol-A-20261016-191822 | GPT-6.1 Sol, A | $0.089 | 7.9 / 16.5 | 13 of 16 |
| sol-B-20261016-191857 | GPT-6.1 Sol, B | $0.097 | 8.0 / 14.5 | 14 of 16 |
| sol-C-20261016-191935 | GPT-6.1 Sol, C | $0.105 | 8.7 / 14.0 | 13 of 16 |
| sol-D-20261016-192013 | GPT-6.1 Sol, D | $0.100 | 8.4 / 16.7 | 16 of 16 |
| sol-E-20261016-192054 | GPT-6.1 Sol, E | $0.133 | 13.7 / 33.9 | 14 of 16 (blocky, some masks misplaced) |

Two single-dot runs were also kept. `gemini-flash-20261016-190628` was the
first check of the route. `sol-E-20261016-191807` was the first check of the plan.

Each way of asking was tried once, on one photo, and the ways of asking were
written and compared on that same photo. Dots 5, 6 and 8 sit right on the
edge of an insect, so what they are "on" is open to argument.
`compare-prompts.jpg` shows the dots where the runs differ.

## Jev as a judge of the outlines (2026-10-18)

`scripts/jev_judge_segments.py`. Each of the 96 outlines above was sent to
Jev 1.14 as drawn, both as the whole photo and as a zoom on the outline. Jev
was asked yes/no "is the outline correct" and picked one of four verdicts.
192 requests, $0.018, 2.5 s. Results: `jev-judge-20261018-091513*`; the
sheet shows Jev's answers under each zoom.

My labels, by eye, on the clear cases: 6 clear mistakes (Gemini dots 4, 6, 8, 11;
Sol A and Sol C dot 4) and 16 correct outlines (Sol D). On those 22, Jev
ranks a mistake below a correct outline 90 % of the time. With "yes" at
p >= 0.5, it still passes 2 or 3 of the 6 mistakes. The mistakes it passes
are the ones that matter most: the neighbouring butterfly outlined when the
dot is on the board next to it (p 0.47-0.68). It also turns down correct
outlines: Sol D dot 15 at 0.27 on the whole photo. The whole-photo and zoom
answers sometimes disagree (Sol C dot 6: 0.67 against 0.19). Treat these as
exploratory: one try, a question I wrote, and labels that are my own.

## Claude Opus 5.5 as a second judge (2026-10-18)

`scripts/opus_judge_segments.py`. Opus got the same pictures as Jev and the
same two questions, written as text, one separate call per picture. It ran on
our Claude plan at medium thinking: 192 calls, $1.97 at API price (not billed),
4.2 s median, 5 minutes in all. Sheet: `jev-vs-opus-20261018-092103-sheet.jpg`.

- Opus and Jev give the same yes/no on 140 of 192 pictures.
- On the 5 clear mistakes (Gemini dots 4, 6, 11; Sol A and Sol C dot 4), Opus
  says no to 5 of 5 on the whole photo and 4 of 5 on the zoom. Jev says no to
  3 and 4. On the 13 clear correct outlines of prompt D, Opus says yes to 12
  and 13, Jev to 12 and 13.
- Sol D dots 5, 6 and 8 sit right on an insect's edge. There, Opus reads the
  dot as being on the insect at dot 5 (and so turns down the board outline),
  and as being on the board at dot 8 (and turns down the butterfly outline),
  explaining its reasoning both times. Earlier, I counted all of prompt D as
  correct; for these edge dots that was generous.
- Opus is not much steadier than Jev between the two views: the same yes/no
  on 81 of 96 outlines (Jev: 83 of 96).
