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
