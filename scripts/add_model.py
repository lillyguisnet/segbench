"""Add a model to the benchmark, one checked step at a time.

    uv run scripts/add_model.py lint                  # every model: is anything half-registered?
    uv run scripts/add_model.py status KEY            # what is done for KEY, and the next step
    uv run scripts/add_model.py check KEY             # the pre-benchmark checks that are still missing
    uv run scripts/add_model.py run KEY               # the standard runs: shows the plan, sends nothing
    uv run scripts/add_model.py run KEY --go          # ... and sends them
    uv run scripts/add_model.py preview KEY           # its scores and rank, nothing published

The whole procedure, and the decisions this script does not make for you
(which host, which thinking level, which price): docs/adding-a-model.md.
The standard runs are segbench/protocol.py, the same for every model.

Nothing here edits the registry, the price list, a published result, a chart
or the site. `check` and `run` call the existing scripts (check_models.py,
probe_thinking.py, probe_detail.py, check_points_v2.py, run.py,
run_screens.py) with the model's key, so a new model gets exactly what the
others got. They read the API keys from .env themselves.

`preview` scores the model with the published scorers and settings into
results/preview/<key>/ (not committed) and ranks it against what is
published; it is how the numbers in a "should we publish it?" message are
made. Publishing stays a separate, deliberate step (the guide, "Publish").
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from segbench import brand, cost, muse_plan, protocol  # noqa: E402
from segbench.call import SUBSCRIPTION_CREDENTIALS, SUBSCRIPTION_PROVIDERS, _router  # noqa: E402
from segbench.local_models import BY_KEY as LOCAL, LOCAL_MODELS  # noqa: E402
from segbench.models import BY_KEY as REMOTE, LEVELS, MODELS, THINKING  # noqa: E402
from segbench.point_score import read_points  # noqa: E402
from segbench.screens import PROMPT_VERSION as SCREEN_PROMPT, ROUNDS  # noqa: E402
from segbench.tasks import TASKS, applies  # noqa: E402

RESULTS = ROOT / "results"
FINDERS = Path("/home/maxime/Projects/finders")
PREVIEW = RESULTS / "preview"
KNOWN_CREDENTIALS = {muse_plan.CREDENTIAL}


# ----------------------------------------------------------------------------- lint

def lint_remote(m) -> tuple[list[str], list[str]]:
    """(problems, warnings) for one model of segbench/models.py."""
    bad, warn = [], []
    prefix = m.route.split(":", 1)[0] if ":" in m.route else None
    if prefix is None:
        bad.append(f"route {m.route!r} has no provider prefix: lm15 may pick another provider (or the paid API)")
    else:
        try:
            provider = _router.resolve(m.route).provider
        except Exception as error:  # an unknown prefix
            provider = None
            bad.append(f"lm15 cannot resolve route {m.route!r}: {error}")
        if provider and provider != prefix:
            bad.append(f"route {m.route!r} resolves to provider {provider!r}, not {prefix!r}")
    if m.billing not in ("subscription", "per-token"):
        bad.append(f"billing {m.billing!r}: must be 'subscription' or 'per-token'")
    if m.credential and m.credential not in KNOWN_CREDENTIALS:
        bad.append(f"credential {m.credential!r} is unknown to segbench/call.py (only {sorted(KNOWN_CREDENTIALS)})")
    on_plan = prefix in SUBSCRIPTION_PROVIDERS or m.credential in SUBSCRIPTION_CREDENTIALS
    if (m.billing == "subscription") != on_plan:
        bad.append(f"billing {m.billing!r} does not match its route/credential: every call would be refused")
    if prefix == "openrouter":
        pin = (m.extensions or {}).get("provider") or {}
        if not pin.get("order") or pin.get("allow_fallbacks") is not False:
            bad.append("OpenRouter route without a pinned host: set extensions={'provider': {'order': [HOST], "
                       "'allow_fallbacks': False}} (the same model scored 3/3 on one host and 0/3 on another)")
    if m.vision is False:
        return bad, warn  # documented as not seeing pictures: never benchmarked
    if m.vision is None:
        warn.append("vision not checked yet (vision=None): `add_model.py check` runs the circle test")
    levels = THINKING.get(m.key)
    if levels is None:
        bad.append("no THINKING entry in segbench/models.py (probe_thinking.py shows what the model accepts)")
    else:
        if protocol.LEVEL not in levels:
            bad.append(f"THINKING has no {protocol.LEVEL!r} level, which every model is benchmarked at")
        for level, settings in levels.items():
            if level not in LEVELS:
                bad.append(f"THINKING level {level!r} is not one of {LEVELS}")
            if set(settings) - {"effort", "extensions"}:
                bad.append(f"THINKING[{level!r}] has keys other than effort/extensions: {sorted(settings)}")
    price = prices().get(m.key)
    if price is None:
        bad.append(f"no price in prices.json (snapshot {cost.PRICES_DATE}): its cost would be missing from the chart")
    else:
        for field in ("input", "output"):
            if not isinstance(price.get(field), (int, float)):
                bad.append(f"prices.json {field!r} is not a number")
        if not price.get("listed_as"):
            warn.append("prices.json entry has no 'listed_as' (where the price was read)")
    warn += lint_maker(m.maker)
    if on_plan:
        from run import PER_PROVIDER  # scripts/run.py
        pool = m.credential or prefix
        if pool not in PER_PROVIDER:
            warn.append(f"subscription pool {pool!r} has no concurrency limit in scripts/run.py PER_PROVIDER "
                        f"(default {PER_PROVIDER.default_factory()} calls at once; plans are shared with the family)")
    return bad, warn


def lint_local(m) -> tuple[list[str], list[str]]:
    bad, warn = [], []
    if m.photo_rounds is not None and not (m.finders and m.task):
        bad.append("photo_rounds set but no finders model/task: scripts/time_finders.py cannot run it")
    if m.screen_seeds is not None:
        if not (m.finders and m.task):
            bad.append("screen_seeds set but no finders model/task: scripts/run_screens_local.py cannot run it")
        if m.load_factor is None:
            bad.append("screen_seeds set but no load_factor: time it on the photos first (scripts/time_finders.py)")
    if m.finders and not (FINDERS / "models" / m.finders).is_dir():
        warn.append(f"~/Projects/finders/models/{m.finders} does not exist (add the model to finders first)")
    warn += lint_maker(m.maker)
    return bad, warn


def lint_maker(maker: str) -> list[str]:
    warn = []
    if maker not in brand.LOGOS:
        warn.append(f"maker {maker!r} has no logo in segbench/brand.py LOGOS: charts show initials "
                    f"{brand.initials(maker)!r} (add the logo + its source and licence in assets/logos/SOURCES.md)")
    if maker not in brand.BRAND:
        warn.append(f"maker {maker!r} has no colour in segbench/brand.py BRAND: grey on the site, overlays and video")
    return warn


def lint_global() -> list[str]:
    bad = []
    keys = Counter([m.key for m in MODELS] + [m.key for m in LOCAL_MODELS])
    bad += [f"key {k!r} is used {n} times (models.py and local_models.py share one name space)"
            for k, n in keys.items() if n > 1]
    bad += [f"THINKING has {k!r}, which is not in MODELS" for k in THINKING if k not in REMOTE]
    bad += [f"prices.json has {k!r}, which is not in MODELS" for k in prices() if k not in REMOTE]
    bad += [f"logo file assets/logos/{f} for {maker!r} is missing"
            for maker, f in brand.LOGOS.items() if not (brand.LOGO_DIR / f).is_file()]
    return bad


def prices() -> dict:
    return json.loads(cost.PRICES_FILE.read_text())["snapshots"][cost.PRICES_DATE]["models"]


def lint_one(key: str) -> tuple[list[str], list[str]]:
    if key in REMOTE:
        return lint_remote(REMOTE[key])
    if key in LOCAL:
        return lint_local(LOCAL[key])
    return [f"unknown key {key!r}: add it to segbench/models.py (remote) or segbench/local_models.py (our GPUs)"], []


def cmd_lint(keys: list[str]) -> int:
    keys = keys or [m.key for m in MODELS] + [m.key for m in LOCAL_MODELS]
    n_bad = 0
    for problem in lint_global():
        print(f"ERROR  {problem}")
        n_bad += 1
    for key in keys:
        bad, warn = lint_one(key)
        n_bad += len(bad)
        for line in bad:
            print(f"ERROR  {key:24} {line}")
        for line in warn:
            print(f"warn   {key:24} {line}")
    print(f"\n{len(keys)} models, {n_bad} errors" + ("" if n_bad else ": registry complete"))
    return 1 if n_bad else 0


# ----------------------------------------------------------------------------- evidence on disk

def latest(pattern: str, key: str, where=lambda r: True) -> list[dict]:
    """The model's records in the newest results/<pattern> file that has any."""
    for path in sorted(RESULTS.glob(pattern), reverse=True):
        rows = [r for r in map(json.loads, path.read_text().splitlines()) if r.get("model") == key and where(r)]
        if rows:
            for r in rows:
                r["_file"] = path.name
            return rows
    return []


def circle_test(key: str) -> tuple[bool | None, str]:
    rows = latest("checks-*.jsonl", key, lambda r: r.get("check") == "red-circles-v1")
    if not rows:
        return None, "not run"
    r = rows[-1]
    if "error" in r:
        return False, f"error: {r['error'][:100]} ({r['_file']})"
    return bool(r.get("passed")), f"{r.get('detail')} ({r['_file']})"


def points_check(key: str) -> tuple[bool | None, str]:
    rows = latest("check-points-v2-*.jsonl", key)
    if not rows:
        return None, "not run"
    parts, ok = [], True
    for r in sorted(rows, key=lambda r: r.get("shape", "")):
        if "error" in r or "unreadable" in r:
            ok = False
            parts.append(f"{r.get('shape')}: {'error' if 'error' in r else 'unreadable'}")
        else:
            passed = r.get("hit") == 6 and r.get("extra") == 0
            ok &= passed
            parts.append(f"{r.get('shape')}: {r.get('hit')}/6, {r.get('extra')} extra"
                         + ("" if r.get("named") else ", x/y not named"))
    return ok, "; ".join(parts) + f" ({rows[0]['_file']})"


def probe(pattern: str, key: str) -> str | None:
    rows = latest(pattern, key)
    return rows[0]["_file"] if rows else None


def expected_runs(key: str) -> list[tuple[str, int, Path, callable]]:
    """(label, calls expected, file, record filter) for the standard runs of a remote model."""
    out = []
    for run in protocol.PHOTO_RUNS:
        n = sum(applies(track, t) for t in TASKS for track in run.tracks) * protocol.REPEATS
        out.append((f"photos {run.prompt_version}", n, RESULTS / f"run-{run.name}.jsonl",
                    lambda r, v=run.prompt_version: r.get("prompt_version") == v))
    n = len(ROUNDS[protocol.SCREEN_ROUND]) * protocol.REPEATS
    round_tasks = {t.key for t in ROUNDS[protocol.SCREEN_ROUND]}
    out.append((f"screens round {protocol.SCREEN_ROUND}", n, RESULTS / f"run-{protocol.SCREEN_RUN}.jsonl",
                lambda r: r.get("task") in round_tasks and r.get("prompt_version") == SCREEN_PROMPT))
    return out


def run_progress(key: str):
    """Per standard run: (label, expected, answered, final errors, retryable errors, unreadable)."""
    rows = []
    for label, n, path, keep in expected_runs(key):
        recs = [r for r in read_jsonl(path) if r.get("model") == key and r.get("level") == protocol.LEVEL and keep(r)]
        answered = sum("error" not in r for r in recs)
        final_err = sum("error" in r and not r.get("retryable") for r in recs)
        retry = sum("error" in r and r.get("retryable") for r in recs)
        unreadable = sum("error" not in r and r.get("readable") is False for r in recs)
        rows.append((label, n, answered, final_err, retry, unreadable))
    return rows


def appended_only(path: Path) -> bool | None:
    """Did the shared run file only grow since the last commit (no earlier line changed)? None: not tracked."""
    rel = path.relative_to(ROOT).as_posix()
    head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
    if head.returncode:
        return None
    return path.read_text().startswith(head.stdout)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ----------------------------------------------------------------------------- status

def cmd_status(key: str) -> int:
    if key in LOCAL:
        return status_local(key)
    if key not in REMOTE:
        print(lint_one(key)[0][0])
        return 1
    m = REMOTE[key]
    if m.vision is False:
        print(f"{m.label} ({key}): does not see pictures (segbench/models.py), never benchmarked: {m.note}")
        return 0
    bad, warn = lint_remote(m)
    print(f"{m.label} ({key})  route {m.route}  billing {m.billing}" + (f"  credential {m.credential}" if m.credential else ""))
    steps: list[tuple[str, bool | None, str, str]] = []  # (step, ok, detail, what to do; "" = not blocking)

    steps.append(("registry", not bad, "; ".join(bad) or "models.py, THINKING and prices.json complete",
                  "fix the errors above (docs/adding-a-model.md, step 1)"))
    ok, detail = circle_test(key)
    steps.append(("circle test", ok, detail, f"uv run scripts/add_model.py check {key}"))
    tp = probe("thinking-probe-*.jsonl", key)
    decided = key in THINKING
    steps.append(("thinking probe", True if tp else None,
                  tp or ("none on file; THINKING written from another measurement" if decided else "not run"),
                  "" if decided else f"uv run scripts/add_model.py check {key}  (then write its THINKING entry "
                                     "from the table)"))
    steps.append(("sees pictures", m.vision is True, f"vision={m.vision}",
                  "set vision=True in segbench/models.py once the circle test passed"))
    dp = probe("detail-probe-*.jsonl", key)
    steps.append(("image-detail probe", True if dp else None, dp or "not run",
                  f"uv run scripts/add_model.py check {key}  (a setting that raises input tokens -> scripts/run.py "
                  "IMAGE_SETTINGS_V2)"))
    ok, detail = points_check(key)
    steps.append(("answer-format check", ok, detail, f"uv run scripts/add_model.py check {key}"))
    # Unreadable answers are left to `preview`: the scorer reads more than run.py's first reading does.
    for label, n, answered, final_err, retry, _ in run_progress(key):
        done = answered + final_err
        extra = ", ".join(s for s in (f"{final_err} failed" if final_err else "",
                                      f"{retry} to resend" if retry else "") if s)
        steps.append((f"run: {label}", None if done == 0 else done >= n, f"{done}/{n}" + (f" ({extra})" if extra else ""),
                      f"uv run scripts/add_model.py run {key} --go"))

    next_step = None
    for step, ok, detail, todo in steps:
        mark = {True: "ok", None: "--", False: "NO"}[ok if ok is None else bool(ok)]
        print(f"  [{mark}] {step:22} {detail}")
        if not ok and next_step is None and todo and not (step == "answer-format check" and ok is False):
            next_step = todo  # a failed format check is information, not a stop
    for line in warn:
        print(f"  warn: {line}")
    for path in {p for _, _, p, _ in expected_runs(key)}:
        grown = appended_only(path)
        if grown is False:
            print(f"  warn: {path.name} changed earlier lines since the last commit, not only appended")
    print(f"\nnext: {next_step or f'uv run scripts/add_model.py preview {key}, then publish (docs/adding-a-model.md)'}")
    return 0


def status_local(key: str) -> int:
    m = LOCAL[key]
    bad, warn = lint_local(m)
    print(f"{m.label} ({key})  local" + (f", finders {m.finders} task {m.task}" + (f" variant {m.variant}" if m.variant else "")
                                         if m.finders else ", own environment in specialists/"))
    for line in bad:
        print(f"  ERROR {line}")
    for line in warn:
        print(f"  warn: {line}")
    photo_files = sorted({r["_file"] for r in local_records(key, "find")})
    screen_files = sorted({r["_file"] for r in local_records(key, "screen")})
    in_selection = [f for f in photo_files if f in protocol.LOCAL_PHOTO_RUNS]
    print(f"  photos timed (time_finders): {'yes' if m.photo_rounds else 'not set up'}; answers in {photo_files or 'no file'}"
          + (f"; chosen for the chart from {in_selection}" if in_selection else
             "; not in protocol.LOCAL_PHOTO_RUNS yet" if photo_files else ""))
    print(f"  screenshots: {'set up' if m.screen_seeds else 'not set up'}; answers in {screen_files or 'no file'}")
    print(f"  load factor: {m.load_factor}")
    print("\nLocal runs need the GPU to be free (they time the model): see docs/adding-a-model.md, 'A model on our GPUs'.")
    return 1 if bad else 0


def local_records(key: str, track: str) -> list[dict]:
    files = [RESULTS / f for f in protocol.LOCAL_PHOTO_RUNS] if track == "find" else []
    files += sorted(RESULTS.glob("run-finders-*.jsonl")) + sorted(RESULTS.glob("run-screens-local-*.jsonl"))
    out = []
    for path in dict.fromkeys(files):
        if ".raw-" in path.name or not path.exists():
            continue
        for r in read_jsonl(path):
            if r.get("model") == key and r.get("track") == track:
                r["_file"] = path.name
                out.append(r)
    return out


# ----------------------------------------------------------------------------- check and run

def env_with_keys() -> dict:
    """os.environ plus the keys in .env (what `set -a; source .env` gives the scripts)."""
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.removeprefix("export ").split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if value:
                env.setdefault(name.strip(), value)
    return env


def sh(cmd: list[str], log: Path | None = None) -> int:
    print("$ " + " ".join(cmd) + (f"  > {log.relative_to(ROOT)}" if log else ""), flush=True)
    if log is None:
        return subprocess.run(cmd, cwd=ROOT, env=env_with_keys()).returncode
    with log.open("a") as f:
        return subprocess.run(cmd, cwd=ROOT, env=env_with_keys(), stdout=f, stderr=subprocess.STDOUT).returncode


def cmd_check(keys: list[str], again: bool) -> int:
    for key in keys:
        if key not in REMOTE:
            sys.exit(f"{key}: `check` is for remote models; local models are checked by `finders smoke` "
                     "(docs/adding-a-model.md)")
    first = [k for k in keys if again or circle_test(k)[0] is None]
    if first:
        sh(["uv", "run", "scripts/check_models.py", *first])
    first = [k for k in keys if again or not probe("thinking-probe-*.jsonl", k)]
    if first:
        sh(["uv", "run", "scripts/probe_thinking.py", *first])
    # These two send at the model's own levels, so they need vision=True and a THINKING entry.
    ready = [k for k in keys if REMOTE[k].vision is True and protocol.LEVEL in THINKING.get(k, {})]
    waiting = [k for k in keys if k not in ready]
    todo = [k for k in ready if again or not probe("detail-probe-*.jsonl", k)]
    if todo:
        sh(["uv", "run", "--with", "pillow", "scripts/probe_detail.py", *todo])
    todo = [k for k in ready if again or points_check(k)[0] is None]
    if todo:
        sh(["uv", "run", "--with", "pillow", "scripts/check_points_v2.py", *todo])
    if waiting:
        print(f"\nNot yet sent the image-detail and answer-format checks to {waiting}: they need vision=True (circle "
              f"test passed) and a THINKING entry with {protocol.LEVEL!r} (from the thinking table above). "
              "Then run `check` again.")
    print()
    for key in keys:
        cmd_status(key)
    return 0


def run_commands(key: str) -> list[tuple[list[str], str]]:
    """The standard runs of segbench/protocol.py for one remote model, as (command, run name)."""
    cmds = []
    for run in protocol.PHOTO_RUNS:
        cmds.append((["uv", "run", "scripts/run.py", "--name", run.name, "--models", key, "--levels", protocol.LEVEL,
                      "--tracks", *run.tracks, "--repeats", str(protocol.REPEATS),
                      "--prompt-version", run.prompt_version], run.name))
    cmds.append((["uv", "run", "scripts/run_screens.py", "--name", protocol.SCREEN_RUN, "--models", key,
                  "--round", str(protocol.SCREEN_ROUND), "--repeats", str(protocol.REPEATS)], protocol.SCREEN_RUN))
    return cmds


def cmd_run(keys: list[str], go: bool, force: bool) -> int:
    blocked = False
    for key in keys:
        if key in LOCAL:
            sys.exit(f"{key} runs on our GPUs: those runs need the GPU free and are started by hand "
                     "(docs/adding-a-model.md, 'A model on our GPUs')")
        bad, _ = lint_one(key)
        reasons = list(bad)
        if REMOTE.get(key) and REMOTE[key].vision is not True:
            reasons.append("vision is not True")
        if circle_test(key)[0] is not True:
            reasons.append("the circle test has not passed")
        fmt, detail = points_check(key)
        if fmt is None:
            reasons.append("the answer-format check has not been run")
        elif fmt is False:
            print(f"note {key}: the answer-format check did not pass ({detail}). That does not stop a model "
                  "(Gemini Flash Lite failed it too), but say so when reporting its scores.")
        if reasons:
            blocked = True
            print(f"{key}: not ready ({'; '.join(reasons)})" + (", sending anyway (--force)" if force else ""))
    if blocked and not force:
        print("\n`uv run scripts/add_model.py status KEY` shows what is missing.")
        return 1

    for key in keys:
        print(f"\n== {REMOTE[key].label} ({key}): the standard runs (segbench/protocol.py)")
        if not go:
            for cmd, _ in run_commands(key):
                sh(cmd + ["--dry-run"])
            continue
        for cmd, name in run_commands(key):
            code = sh(cmd, RESULTS / f"run-{name}.{key}.log")
            if code:
                print(f"{name} stopped with code {code}; see its log. Running the same command again only sends "
                      "what is missing.")
                return code
        if key in REMOTE and REMOTE[key].credential == muse_plan.CREDENTIAL:
            print("(Muse Code plan: `uv run python -c 'from segbench import muse_plan; print(muse_plan.usage())'` "
                  "shows how much of the plan's window this used.)")
    if not go:
        print("\nNothing sent. Add --go to send these (each run skips what it already has).")
    else:
        print("\nDone. Next: " + " ".join(f"uv run scripts/add_model.py preview {k}" for k in keys))
    return 0


# ----------------------------------------------------------------------------- preview

def photo_candidates(key: str, sources: list[Path] | None) -> dict[str, list[dict]]:
    """prompt version -> the model's find and trick records, from one file per version."""
    if sources is None:
        sources = [RESULTS / f"run-{r.name}.jsonl" for r in protocol.PHOTO_RUNS]
        sources += [RESULTS / f for f in protocol.LOCAL_PHOTO_RUNS]
    by_version: dict[str, list[dict]] = defaultdict(list)
    files: dict[str, set] = defaultdict(set)
    for path in sources:
        for r in read_jsonl(path):
            if r.get("model") != key or r.get("track") not in ("find", "trick"):
                continue
            if key in REMOTE and r.get("level") != protocol.LEVEL:
                continue
            if "error" in r and r.get("retryable"):
                continue  # never answered: run.py sends it again
            by_version[r["prompt_version"]].append(r)
            files[r["prompt_version"]].add(path.name)
    for version, names in files.items():
        if len(names) > 1:
            sys.exit(f"{key}: prompt {version} answers are in several files {sorted(names)}; choose with --from")
    return dict(by_version)


def score_photos(key: str, version: str, records: list[dict], out: Path) -> dict | None:
    import score_points  # scripts/score_points.py: the published scorer

    find = [r for r in records if r["track"] == "find"]
    if not find:
        return None
    src = out.parent / f"run-{key}-{version}.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in find))
    with contextlib.redirect_stdout(io.StringIO()):
        score_points.build(ROOT / protocol.PHOTO_REFERENCE, [src], out, protocol.PHOTO_SCORING["parser"],
                           protocol.PHOTO_SCORING["method"], protocol.PHOTO_SCORING["dish_labels"])
    rows = list(csv.DictReader((out / "summary.csv").open()))
    return rows[0] if rows else None


def screen_scores(key: str, sources: list[Path] | None) -> dict | None:
    """Scored as the published chart scores them (scripts/plot_screens.py)."""
    from plot_screens import f1

    if sources is None:
        sources = [RESULTS / f"run-{protocol.SCREEN_RUN}.jsonl"]
        sources += sorted(p for p in RESULTS.glob("run-screens-local-*.jsonl") if ".raw-" not in p.name)
    answer_key = json.loads((ROOT / protocol.SCREEN_KEY).read_text())
    per_task, secs, costs, errors, unreadable = defaultdict(list), [], [], 0, 0
    for path in sources:
        for r in read_jsonl(path):
            if r.get("model") != key or r.get("track") != "screen" or r.get("task") not in answer_key["tasks"]:
                continue
            if key in REMOTE and r.get("level") != protocol.LEVEL:
                continue
            if "error" in r:
                errors += 1  # left out, as the chart does (scripts/plot_screens.py)
                continue
            W, H = r["image_size"]
            clicks = [(c["x"] * W / 1000, c["y"] * H / 1000) for c in r.get("clicks", [])] if r.get("readable") else []
            unreadable += not r.get("readable")
            per_task[r["task"]].append(f1(clicks, answer_key["tasks"][r["task"]]["targets"]))
            secs.append(r.get("seconds"))
            costs.append(r.get("cost_usd"))
    if not per_task:
        return None
    task_means = {t: statistics.mean(v) for t, v in per_task.items()}
    return {"score": statistics.mean(task_means.values()), "tasks": task_means,
            "missing_tasks": sorted(set(answer_key["tasks"]) - set(per_task)), "answers": len(secs),
            "seconds": statistics.median(s for s in secs if s is not None),
            "cost_per_1000": 1000 * statistics.mean(costs) if costs and None not in costs else None,
            "errors": errors, "unreadable": unreadable}


def rank(score: float, table: list[tuple[str, float]], exclude: str) -> tuple[int, int, list[str]]:
    others = [(label, s) for entrant, label, s in table if entrant.split("@")[0] != exclude]
    above = sorted([(s, label) for label, s in others if s > score + 1e-12], reverse=True)
    return len(above) + 1, len(others) + 1, [f"{label} {s * 100:.1f}" for s, label in above[-2:]]


def published_photo_table() -> list[tuple[str, str, float]]:
    path = RESULTS / protocol.PUBLISHED_PHOTOS / "summary.csv"
    return [(r["entrant"], r["label"], float(r["score"])) for r in csv.DictReader(path.open()) if r["score"]]


def published_screen_table() -> list[tuple[str, str, float]]:
    path = ROOT / "charts" / protocol.PUBLISHED_SCREENS / "with-specialists" / "points.csv"
    return [(r["entrant"], r["label"], float(r["quality"])) for r in csv.DictReader(path.open())]


def cmd_preview(keys: list[str], sources: list[Path] | None) -> int:
    for key in keys:
        if key not in REMOTE and key not in LOCAL:
            sys.exit(lint_one(key)[0][0])
        label = REMOTE[key].label if key in REMOTE else LOCAL[key].label
        out = PREVIEW / key
        if out.exists():
            shutil.rmtree(out)  # scratch: rebuilt every time
        out.mkdir(parents=True)
        print(f"\n== {label} ({key}): preview, scored as {protocol.PUBLISHED_PHOTOS} and {protocol.PUBLISHED_SCREENS} "
              f"are; nothing published\n")

        candidates = photo_candidates(key, sources)
        scored = {v: score_photos(key, v, recs, out / f"points-{v}") for v, recs in sorted(candidates.items())}
        scored = {v: s for v, s in scored.items() if s}
        if scored:
            complete = {v: s for v, s in scored.items() if s["score"]}
            for v, s in sorted(scored.items()):
                tasks = "  ".join(f"{t} {float(s[t]) * 100:.0f}" for t in ("logs", "cows", "fig", "dishes") if s[t])
                total = f"{float(s['score']) * 100:.1f}" if s["score"] else f"incomplete ({s['coverage']}/4 tasks)"
                print(f"photos, prompt {v:8} {total:>6}   {tasks}   unreadable {s['unreadable']}, "
                      f"recovered {s['recovered']}")
            if complete:
                tie = protocol.PROMPT_TIE_GOES_TO
                best = max(complete, key=lambda v: (float(complete[v]["score"]), v == tie))
                s = complete[best]
                score = float(s["score"])
                n, total, above = rank(score, published_photo_table(), key)
                cost = float(s["cost_usd"]) * 1000 if s["cost_usd"] else None
                print(f"  -> keeps prompt {best}: {score * 100:.1f}, {n} of {total} on {protocol.PUBLISHED_PHOTOS}"
                      + (f" (just below {', '.join(above)})" if above else " (first)"))
                print(f"     {float(s['seconds']):.1f} s per photo (median)"
                      + (f", ${cost:.2f} per 1,000 photos" if cost is not None else ", cost unknown"))
            trick = [r for recs in candidates.values() for r in recs if r["track"] == "trick" and "error" not in r]
            if trick:
                invented = unread = 0
                for r in trick:
                    try:
                        invented += bool(read_points(r.get("text", ""), parser="lenient")[0])
                    except Exception:
                        unread += 1
                print(f"  trick questions (things not in the photo): invented objects in {invented} of {len(trick)}"
                      + (f", {unread} unreadable" if unread else ""))
        else:
            print("photos: no answers found" + ("" if sources else " (looked in the standard run files)"))

        sc = screen_scores(key, sources)
        if sc:
            n, total, above = rank(sc["score"], published_screen_table(), key)
            print(f"\nscreens round {protocol.SCREEN_ROUND}: {sc['score'] * 100:.1f}, {n} of {total} on "
                  f"{protocol.PUBLISHED_SCREENS}" + (f" (just below {', '.join(above)})" if above else " (first)"))
            print("  " + "  ".join(f"{t} {v * 100:.0f}" for t, v in sc["tasks"].items())
                  + f"   ({sc['answers']} answers, {sc['unreadable']} unreadable)")
            print(f"  {sc['seconds']:.1f} s per screenshot (median)"
                  + (f", ${sc['cost_per_1000']:.2f} per 1,000 screenshots" if sc["cost_per_1000"] is not None else ""))
            if sc["missing_tasks"]:
                print(f"  INCOMPLETE: no answers for {sc['missing_tasks']}")
            if sc["errors"]:
                print(f"  {sc['errors']} answers ended in an error and are left out, as the published chart does")
        else:
            print("\nscreens: no answers found")

        if key in REMOTE:
            ours = billed = 0.0
            n_billed = 0
            files = [RESULTS / f"run-{r.name}.jsonl" for r in protocol.PHOTO_RUNS]
            files.append(RESULTS / f"run-{protocol.SCREEN_RUN}.jsonl")
            for path in files:
                for r in read_jsonl(path):
                    if r.get("model") == key:
                        ours += r.get("cost_usd") or 0
                        if r.get("provider_cost_usd") is not None:
                            billed += r["provider_cost_usd"]
                            n_billed += 1
            print(f"\nthese runs at the charted price: ${ours:.2f}"
                  + (f"; the provider billed ${billed:.2f} for {n_billed} of them" if n_billed else "")
                  + (" (a subscription: charted at an API price, nothing billed per call)"
                     if REMOTE[key].billing == "subscription" else ""))
        (out / "README.md").write_text(
            f"Preview of {label} ({key}), {date.today()}: scratch, rebuilt by "
            f"`uv run scripts/add_model.py preview {key}`. Not published, not committed.\n")
        print(f"\nreport: {out.relative_to(ROOT)}/ (points-*/index.html shows every answer)")
    return 0


# ----------------------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lint", help="registry completeness, no calls (default: every model)")
    p.add_argument("keys", nargs="*")
    p = sub.add_parser("status", help="what is done for a model, and the next step")
    p.add_argument("keys", nargs="+")
    p = sub.add_parser("check", help="the pre-benchmark checks still missing (remote models)")
    p.add_argument("keys", nargs="+")
    p.add_argument("--again", action="store_true", help="send every check again, even those on file")
    p = sub.add_parser("run", help="the standard runs (segbench/protocol.py); --go to send")
    p.add_argument("keys", nargs="+")
    p.add_argument("--go", action="store_true")
    p.add_argument("--force", action="store_true", help="send even if a check is missing (say so when reporting)")
    p = sub.add_parser("preview", help="score and rank, nothing published")
    p.add_argument("keys", nargs="+")
    p.add_argument("--from", dest="sources", nargs="+", type=Path,
                   help="answer files to read instead of the standard ones (local models, trials)")
    a = ap.parse_args(argv)
    if a.cmd == "lint":
        return cmd_lint(a.keys)
    if a.cmd == "status":
        return max(cmd_status(k) for k in a.keys)
    if a.cmd == "check":
        return cmd_check(a.keys, a.again)
    if a.cmd == "run":
        return cmd_run(a.keys, a.go, a.force)
    return cmd_preview(a.keys, [s.resolve() for s in a.sources] if a.sources else None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
