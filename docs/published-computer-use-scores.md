# Published computer-use scores for our models (read 2026-10-11)

Our screen-click test is small (five screenshots), but every model gets the
same pictures, the same untuned prompt and the same scoring. Readers will want
to know how its results compare with the public computer-use benchmarks. This
page lists what has been published for the exact model versions we ran. **We
did not rerun or check any of these numbers.** The rows we used, with their
links, are in `evidence/computer-use-scores-2026-10-11.json`.

## The three public tests

| Test | What the model must do | Closest to our test? |
|---|---|---|
| [ScreenSpot-Pro](https://gui-agent.github.io/grounding-leaderboard/) | One click on the requested button or text in a screenshot of professional software (1,581 questions) | Yes: one screenshot, one click |
| [OSWorld-Verified](https://os-world.github.io/) | Finish a whole task on a real computer, step by step, up to 100 actions | No: also tests planning and recovering from mistakes |
| OSWorld 2.0 (paper 2026-06-28) | 108 long workflows (about 1.6 hours each for a person), pass or fail | No: a much harder, longer version of the task above |

## Official tables (one setup for everyone)

Neither official table lists any of the model versions we tested.

- **OSWorld-Verified**, run by the OSWorld team: the nearest entries are
  earlier versions. Claude Opus 5 scores 83.4% (2026-08-01), Kimi K2.6 73.1%,
  Qwen 3.7 Plus 73.3%. Gemini 3.1 Pro appears only inside another company's
  agent software (OpenAPA, 78.3%), which also lets it run code, so that is
  not the model alone.
- **ScreenSpot-Pro**, run by its authors: 99 entries, mostly small
  specialist models. The only general models are old ones (GPT-5 at 18.5%,
  an early Claude computer-use model at 17.1%).

## Numbers the makers report, as collected by llm-stats.com and benchlm.ai

| Our model | Listed as | ScreenSpot-Pro | OSWorld-Verified | OSWorld 2.0 |
|---|---|---|---|---|
| astra | GPT-6 Astra | 92.7% | – | 72.6% |
| gemini-pro | Gemini 3.1 Pro | 84.4% ¹ | – | – |
| qwen-27b | Qwen3.8-27B | – | 84.3% | – |
| gemini-flash-lite | Gemini 3.5 Flash-Lite | – | 74.0% | – |
| sol | GPT-6.1 Sol | – | – | 71.4% ² |
| gemini-flash | Gemini 3.8 Flash | – | – | 59.0% |
| luna | GPT-6 Luna | – | – | 52.7% ² |
| terra | GPT-5.6 Terra | – | – | 50.2% |
| opus | Claude Opus 5.5 | – | – | conflicting ³ |
| qwen-flash | Qwen3.8 Flash | – | – | 19.4% ⁴ |
| sonnet, kimi-k3, glm-5.3-flash, deepseek-flash | | nothing found | nothing found | nothing found |

Earlier versions of our models that the ScreenSpot-Pro tables list (makers'
own numbers; not the versions we ran): Gemini 3 Pro 72.7%, Gemini 3 Flash
69.1%, Qwen3.5-27B 70.3%, Claude Opus 4.8 87.9%. llm-stats.com's own
"verified" tab for ScreenSpot-Pro holds a single run (GPT-5.4 nano, shown as
1.14 on a 0-1 scale), so none of these numbers has been checked
independently there either.

1. On benchlm.ai only (llm-stats.com lists Gemini 3 Pro, not 3.1 Pro).
2. On llm-stats.com only. For Sol, OpenAI's launch post says it comes within
   2.1 points of Astra at maximum reasoning effort.
3. llm-stats.com says 81.8%, benchlm.ai says 48.7%. Anthropic's own page
   shows 81.8% as a *partial-credit* score on OSWorld **2.1**, with maximum
   thinking. Other makers report pass-or-fail on 2.0, so the two numbers do
   not measure the same thing.
4. Doubtful: benchlm.ai gives Qwen3.8 Max exactly the same 19.4%, which
   suggests one number was copied to both.

## Why these numbers cannot be lined up against each other, or against ours

- **Each maker ran its own setup:** how much thinking, which helper tools,
  how many tries, how many steps, which screen size. The collector sites
  label almost every row "self-reported", and so do we.
- **Scoring differs.** Partial credit and pass-or-fail can differ by a
  factor of two for the same model (note 3).
- **The tests measure different things.** OSWorld rewards planning over many
  steps. ScreenSpot-Pro and our test measure where a single click lands, but
  ScreenSpot-Pro uses large professional-software screens, while ours uses
  everyday web apps and asks for several right clicks per question.
- **No public score exists for four of our fourteen models.**

The fair comparison is our own small test, where every model was treated the
same way. The scores above are context, not a check on our ranking.
