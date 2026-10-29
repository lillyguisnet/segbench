# Thinking levels on the eleven photos (2026-10-20): preview, not published

Asked by Maxime on 2026-10-20: rerun every Claude and GPT model, and Muse Wonder on our
Muse Code plan (not the API), with no thinking where the model allows it and with maximum
thinking, to see whether thinking changes pointing scores and the kinds of mistakes.
Nothing published was changed.

## What was run

- **Models**: Claude Sonnet 5.5, Opus 5.5, Haiku 5.5 (Claude plan); GPT-5.6 Terra, GPT-6 Luna,
  GPT-6 Astra, GPT-6.1 Sol (ChatGPT plan); Muse Wonder (Muse Code plan, the plan's own key).
  Muse Wonder Full was not run.
- **Levels** (`segbench/models.py` THINKING, checked on the wire 2026-10-08 to 10-14):

  | model | least ("min") | medium | max |
  |---|---|---|---|
  | Sonnet 5.5 | off (`thinking: between_tools`) | adaptive, effort medium | adaptive, effort max |
  | Haiku 5.5 | off (`thinking: disabled`) | adaptive, effort medium | adaptive, effort max |
  | Opus 5.5 | effort low (cannot be turned off) | effort medium | effort max |
  | Terra, Luna | off | medium | max |
  | Astra, Sol | low (cannot be turned off) | medium | max |
  | Muse Wonder | minimal (cannot be turned off) | medium | max |

  Astra's max was dropped on 2026-10-09 for its per-token price; restored for this run, which is on
  the plan.
- **Questions**: the eleven photo tasks of point v3, find track only (no trick questions), 3 tries.
- **Prompt**: each model's prompt on the v3 chart at every level (Luna and Sol v1-hd, the others
  v2; Muse Wonder v2, as its API copy on v3). The prompt was chosen at medium; it was not
  re-chosen per level.
- **Medium** answers are the very records of point v3 (identical scores), except Muse Wonder:
  its plan copy (first four photos from 2026-10-12, `run-points-v2.jsonl`; the other seven run
  2026-10-20, `run-thinking-levels-v2-muse-medium.jsonl`), not the API copy charted in v3.
- min and max: `results/run-thinking-levels-v2.jsonl` (Claude; the first 55 answers came from a
  start that queued every provider behind Claude and was restarted), `-v2-gpt.jsonl` (Terra, Astra),
  `-v2-muse.jsonl`, `-v1hd.jsonl` (Luna, Sol)
  (2026-10-20). Medium answers are 6 to 11 days older: a provider may have changed a model under
  the same name in between.

## Scoring and mistakes

- Scores exactly as point v3 (`scripts/build_thinking_levels.py` -> `results/thinking-levels-v1/`).
- Kinds of mistakes: `scripts/classify_errors.py --points-dir results/thinking-levels-v1
  --skip-boxes`, the same rules as the published error charts. The "wrong object" rule
  (3 other models erred on the same spot) still counts against the published v3 answers only,
  and never counts the same model at another level or route as another model. Checked: sorting
  the published answers with this version gives yesterday's files byte for byte.

## Missing and unanswered

- GPT-5.6 Terra at max, fig photo, try 0: never answered. Six attempts (2026-10-20), each cut by
  the connection at 15 minutes while Terra was still thinking. Left out (a call never answered),
  not scored 0: Terra at max has 32 answers, its fig score is the mean of 2 tries. Built with
  `scripts/build_thinking_levels.py --allow-short` for this one gap.
- Answers that came back with nothing to read are scored as given (0), as always: Opus at max 1
  (dishes) and Haiku at max 3 (all three log answers) spent the whole 32,000-token Claude allowance
  thinking. The ChatGPT plan set no such limit (Terra at max thought up to 47,404 tokens).

## Results (`charts/thinking-levels-v1/`, `per-level.csv`)

Score (point v3 scoring, out of 100) at least / medium / most thinking; in brackets, max minus
medium with a 95% interval from resampling the eleven photos (paired):

| model | least | medium | max | max - medium |
|---|---:|---:|---:|---|
| GPT-6 Astra | 76.7 (low) | 77.6 | 70.2 | -7.5 (-17.5 to -1.4) |
| GPT-6.1 Sol | 75.4 (low) | 76.7 | 71.7 | -5.0 (-11.1 to -0.9) |
| Muse Wonder (plan) | 62.9 (minimal) | 69.6 | 70.7 | +1.1 (-2.1 to +4.9) |
| Claude Opus 5.5 | 64.5 (low) | 65.4 | 73.1 | +7.7 (+0.8 to +15.0) |
| Claude Haiku 5.5 | 11.4 (off) | 55.8 | 51.1 | -4.7 (-18.0 to +5.3) |
| Claude Sonnet 5.5 | 42.2 (off) | 55.0 | 72.8 | +17.7 (+9.3 to +28.8) |
| GPT-6 Luna | 22.6 (off) | 36.3 | 45.0 | +8.7 (+2.1 to +15.6) |
| GPT-5.6 Terra | 28.4 (off) | 34.2 | 34.7 | +0.5 (-3.8 to +5.6) |

- Astra's and Sol's loss at max is mostly the jars photo (-54 and -31): the right six jars every
  time, points further from the key's dots.
- Haiku with thinking off: 8 of 33 answers in the wrong coordinate scale or x/y swapped, 3 shifted
  or stretched, 3 empty.
- Sonnet thought 0 tokens both "off" and at "medium" yet scored 42.2 and 55.0. The medium answers
  are from 2026-10-09/15, the off answers from 2026-10-20; the effort word may change the answer
  itself, or the model behind the name changed. Not separated here.
