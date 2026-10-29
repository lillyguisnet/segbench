# Kinds of mistakes on the eleven photos

Made by scripts/classify_errors.py (rules in its header, reasons in docs/error-kinds.md). Scores unchanged. Outlines of the key objects (SAM 2.1, from the reviewed boxes) and of look-alikes (SAM 3) are measuring aids, not reviewed by a person.

## Points (results/thinking-levels-v1, 791 replies)

| kind | wrong points | share | objects left unfound | share |
|---|---:|---:|---:|---:|
| no answer: stuck |  |  | 275 | 5.4% |
| no answer: empty |  |  | 10 | 0.2% |
| coordinate mix-up | 290 | 4.0% | 249 | 4.9% |
| shifted or stretched | 207 | 2.9% | 189 | 3.7% |
| doom loop | 262 | 3.7% |  |  |
| off image | 180 | 2.5% |  |  |
| on the object, off-centre | 1584 | 22.1% |  |  |
| duplicate | 1000 | 14.0% |  |  |
| wrong object | 284 | 4.0% |  |  |
| near miss | 825 | 11.5% |  |  |
| in a gap | 1032 | 14.4% |  |  |
| maybe missing from key | 5 | 0.1% |  |  |
| background | 1494 | 20.9% |  |  |
| missed |  |  | 4370 | 85.8% |
| total | 7163 | | 5093 | |

Correct points: 11329 (full credit at the strictest level: 6709). Objects left unfound: the reply-level kind when there is one, the group box that swallowed them, or missed.

### Score each kind costs, per model (points out of 100, as charted)

| model | score | no answer: stuck | no answer: unreadable | no answer: empty | coordinate mix-up | shifted or stretched | doom loop | off image | on the object, off-centre | duplicate | wrong object | near miss | in a gap | maybe missing from key | background | missed | partial credit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra | 78 |  |  |  |  |  |  |  | 3.9 | 0.6 | 0.6 | 0.8 | 0.4 |  | 1.3 | 4.5 | 9.1 |
| GPT-6 Astra | 77 |  |  |  |  |  |  |  | 4.4 | 0.8 | 0.5 | 0.6 | 0.6 |  | 0.9 | 4.7 | 10.1 |
| GPT-6.1 Sol | 77 |  |  |  |  |  |  |  | 4.7 | 0.7 | 0.4 | 1.1 | 0.3 |  | 0.8 | 5.6 | 8.8 |
| GPT-6.1 Sol | 75 |  |  |  |  |  |  |  | 4.5 | 0.7 | 0.5 | 0.4 | 0.3 |  | 0.9 | 6.8 | 10.2 |
| Claude Opus 5.5 | 73 | 3.0 |  |  |  |  |  |  | 8.5 |  | 0.3 | 0.5 |  |  |  | 6.7 | 8.9 |
| Claude Sonnet 5.5 | 73 |  |  |  |  |  |  |  | 10.1 | 0.1 | 0.9 |  |  |  | 0.3 | 8.3 | 8.3 |
| GPT-6.1 Sol | 72 |  |  |  |  |  |  |  | 9.8 | 0.4 | 0.5 | 0.3 | 0.2 |  | 1.1 | 5.0 | 10.2 |
| Muse Wonder | 71 |  |  |  |  |  |  |  | 5.3 | 0.7 | 0.2 | 1.3 | 0.3 |  | 0.3 | 8.2 | 14.0 |
| GPT-6 Astra | 70 |  |  |  |  |  |  |  | 10.9 | 0.8 | 0.4 | 0.4 | 0.4 |  | 1.4 | 4.7 | 9.6 |
| Muse Wonder | 70 |  |  |  |  |  |  |  | 4.9 | 0.8 | 0.4 | 2.2 | 0.2 |  | 0.6 | 9.1 | 13.2 |
| Claude Opus 5.5 | 65 |  |  |  |  |  |  |  | 10.7 | 0.4 | 0.2 | 2.7 | 0.3 |  | 0.1 | 11.6 | 10.3 |
| Claude Opus 5.5 | 64 |  |  |  |  |  |  |  | 11.0 | 0.5 | 0.3 | 1.1 | 0.2 |  | 0.1 | 12.5 | 11.8 |
| Muse Wonder | 63 |  |  | 3.0 |  | 2.2 |  |  | 6.1 | 1.0 | 0.3 | 2.2 | 0.3 |  | 0.5 | 10.6 | 11.3 |
| Claude Haiku 5.5 | 56 |  |  | 9.1 |  |  |  |  | 10.6 | 0.5 | 0.3 | 1.2 | 0.2 | 0.1 |  | 15.8 | 8.5 |
| Claude Sonnet 5.5 | 55 |  |  |  |  |  |  | 0.2 | 11.3 | 2.0 | 0.1 | 3.7 | 0.6 |  | 0.4 | 15.5 | 8.7 |
| Claude Haiku 5.5 | 51 | 9.1 |  | 9.1 |  |  |  |  | 12.6 | 0.1 | 0.5 | 1.0 |  |  | 0.1 | 10.3 | 7.4 |
| GPT-6 Luna | 45 |  |  |  |  | 2.6 |  |  | 13.1 | 0.4 | 0.2 | 4.2 | 0.2 |  | 0.1 | 19.3 | 10.0 |
| Claude Sonnet 5.5 | 42 |  |  |  |  |  | 1.1 | 0.4 | 10.6 | 2.1 |  | 7.1 | 0.6 |  | 0.8 | 20.3 | 7.1 |
| GPT-6 Luna | 36 |  |  |  | 0.5 | 1.0 | 1.3 |  | 14.8 | 0.6 | 0.4 | 5.9 | 0.4 |  | 1.5 | 22.9 | 9.3 |
| GPT-5.6 Terra | 35 |  |  |  |  | 1.3 |  |  | 16.3 | 0.1 | 0.2 | 4.9 | 0.2 |  | 0.7 | 26.6 | 7.3 |
| GPT-5.6 Terra | 34 |  |  |  |  | 0.6 |  |  | 13.8 | 0.6 | 0.2 | 5.1 | 0.5 |  | 0.6 | 28.8 | 6.4 |
| GPT-5.6 Terra | 28 |  |  |  |  | 1.6 |  |  | 11.4 | 1.0 | 0.2 | 7.8 | 0.6 |  | 1.1 | 28.4 | 7.9 |
| GPT-6 Luna | 23 |  |  |  |  | 1.8 | 0.4 |  | 13.4 | 0.8 | 0.2 | 6.8 | 0.7 |  | 1.8 | 29.4 | 6.7 |
| Claude Haiku 5.5 | 11 |  |  | 9.1 | 8.8 | 2.6 |  | 0.1 | 11.1 | 0.2 | 0.4 | 7.8 | 0.1 |  | 0.6 | 21.5 | 4.0 |

Each figure: how much higher that model's charted score would be if that one kind alone were fixed. Fixes interact; the figures do not add up.
