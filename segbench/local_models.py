"""The models we run on our own GPUs, in one place.

Before 2026-10-13 each local model was written down four times (chart name in
segbench/chart_data.py, photo timing setup in scripts/time_finders.py, screen
setup and load factor in scripts/run_screens_local.py, wording in
scripts/run_finders.py). Those tables are now read from here;
tests/test_registry.py checks they still hold exactly what they held.

Adding a pointing model from ~/Projects/finders (docs/adding-a-model.md):
add it to finders first (its README, "Adding a model"), then one LocalModel
here with `photo_rounds` set, time it on the photos (scripts/time_finders.py),
copy the load factor it measured into `load_factor`, then set `screen_seeds`
and run it on the screenshots (scripts/run_screens_local.py).

Fields:
- finders, task, variant: the ~/Projects/finders model, its task and variant
  (None for models with their own environment in specialists/).
- plural: asked with plurals ("the cows"), as its own documentation asks for
  all instances; else singular phrases (scripts/run_finders.py PLURAL).
- photo_copies, photo_rounds: scripts/time_finders.py, copies tried at once
  under load and rounds of answers. photo_rounds None: not timed by that script.
- screen_seeds: scripts/run_screens_local.py, tries per screenshot (3 for a
  model that samples, 1 for a deterministic one). None: not run there.
- screen_task: the finders task used on the screenshots, when the model has
  one made for them (GroundingPI's `gui`); None: the same task as on the photos.
- service: the finders model answers from a separate service (vLLM). Under
  load, scripts/time_finders.py then starts one service that answers up to
  max(photo_copies) requests at once, and "copies" are that many processes
  sending requests to it (how a vLLM model is used), not copies of the model.
- warmup_photos: one untimed question per photo before timing starts, in the
  single-copy pass and in every copy of the load test (whose window then starts
  after each copy's warm-up). For models that compile on their first question
  (Falcon: ~20-50 s once per process), which is not answering time. Default
  off: the other models keep the single warm-up question they were timed with.
- load_factor: measured by scripts/time_finders.py on the photos
  (results/run-<name>.timing.json, "load_factor"); cost = $/GPU-hour x
  single-copy seconds x this factor. Copied by hand on purpose: a published
  cost must not change because a later timing run overwrote a file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModel:
    key: str  # used in results files and on the chart
    label: str  # shown on the chart
    maker: str  # segbench/brand.py
    finders: str | None = None
    task: str | None = None
    variant: str | None = None
    plural: bool = False
    photo_copies: tuple[int, ...] = ()
    photo_rounds: int | None = None
    screen_seeds: int | None = None
    load_factor: float | None = None
    note: str = ""
    screen_task: str | None = None
    service: bool = False
    warmup_photos: bool = False


LOCAL_MODELS: tuple[LocalModel, ...] = (
    # Own environments in specialists/ (their README): segmentation models.
    LocalModel("sam1-vit-h", "SAM 1", "Meta"),
    LocalModel("sam2.1-large", "SAM 2.1", "Meta"),
    LocalModel("sam3", "SAM 3", "Meta", "sam3", "segment", screen_seeds=1, load_factor=1.022,
               note="photos: specialists/sam; screenshots: through finders"),
    LocalModel("dinov3", "DINOv3", "Meta"),
    LocalModel("yoloe-26x", "YOLOE-26x", "Ultralytics"),
    LocalModel("rfdetr-seg-2xl", "RF-DETR Seg", "Roboflow"),
    LocalModel("gen2seg-sd", "gen2seg", "UC Davis"),
    # Pointing models from ~/Projects/finders.
    LocalModel("locateanything-3b", "LocateAnything 3B", "NVIDIA", "locateanything", "point", plural=True,
               photo_copies=(2,), photo_rounds=3, screen_seeds=3, load_factor=1.0,
               note="samples (temperature 0.7): 3 seeds on the screenshots"),
    LocalModel("molmopoint-8b", "MolmoPoint 8B", "Ai2", "molmopoint", "point", plural=True,
               photo_copies=(), photo_rounds=2, screen_seeds=1, load_factor=1.0,
               note="~19 GB: one copy only on 24 GB"),
    LocalModel("rexomni-3b", "Rex-Omni 3B", "IDEA", "rexomni", "point",
               photo_copies=(2,), photo_rounds=3, screen_seeds=1, load_factor=0.6254),
    LocalModel("florence2-large", "Florence-2 Large", "Microsoft", "florence2", "box", "large",
               note="first runs only (scripts/run_finders.py, protocol v1/v2); superseded by the two modes below"),
    LocalModel("florence2-large-grounding", "Florence-2 Large", "Microsoft", "florence2", "box", "large",
               photo_copies=(2, 4), photo_rounds=3, screen_seeds=1, load_factor=0.4041,
               note="its better text mode on the photos (phrase grounding)"),
    LocalModel("florence2-large-ovd", "Florence-2 · detection", "Microsoft", "florence2", "ovd", "large",
               photo_copies=(2, 4), photo_rounds=3, screen_seeds=1, load_factor=0.528),
    # Added 2026-10-11, timed on the whole of GPU 1 (--gpu 1): same model of card
    # (RTX 3090), its own programs stopped for the run. Singular phrases, like
    # Rex-Omni: Moondream's documentation points at "face", detects "person".
    # Photon reserves ~17 GB of the card (weights + its answer memory): one copy only.
    LocalModel("moondream-3.1", "Moondream 3.1", "Moondream", "moondream", "point",
               photo_copies=(), photo_rounds=3, screen_seeds=1, load_factor=1.0,
               note="results/run-finders-4.timing.json"),
    # Added 2026-10-14 (Maxime asked the same day; weights out that day).
    # GroundingPI 4B, paper arXiv:2609.39601, through its authors' vLLM service
    # (~/Projects/finders/models/groundingpi). Deterministic (temperature 0):
    # one answer per screenshot. Photos: its category-point template
    # ("Point to: cow."), singular like its own examples ("car</c>person").
    # Screenshots: its GUI template ("Point to the UI element to click for the
    # following instruction: ..."), made for exactly that.
    # Timed 2026-10-14 on the whole of GPU 0 (its one service, the paper
    # rewriter, stopped): load factor 0.3148 with 16 requests at once (2: 0.682,
    # 4: 0.496, 8: 0.371), results/run-finders-5.timing.json. vLLM's reuse of
    # earlier requests off (finders serve_fresh.py): with it on, repeats differed
    # and ran faster than any new picture would (results/superseded/run-finders-5.prefix-cache-on.*).
    LocalModel("groundingpi-4b", "GroundingPI 4B", "GroundingPI", "groundingpi", "point",
               photo_copies=(2, 4, 8, 16), photo_rounds=3, screen_seeds=1, screen_task="gui", service=True,
               load_factor=0.3148, note="vLLM service; copies = requests at once to one service"),
    # Added 2026-10-23 at the user's request; photos benchmarked 2026-10-23 on the
    # whole of GPU 1 (its six services stopped, user's go-ahead). Native masks and
    # boxes; its point is the deepest point inside its mask, not a native point.
    # Singular phrases, like its own examples ("cat"). Deterministic (greedy).
    # ~10.7 GB per copy: two copies fit on 24 GB. Compiles on its first question
    # in each process: warmup_photos. Screenshots not run (screen_seeds unset).
    LocalModel("falcon-perception-hd", "Falcon Perception-HD", "TII", "falcon", "segment", "hd",
               photo_copies=(2,), photo_rounds=3, warmup_photos=True, load_factor=1.0,
               note="RL release 19-08-2026; points: 2 copies ran out of memory (90 full-size masks on "
                    "the logs photo), so 1 copy, factor 1.0, results/run-finders-6.timing.json; "
                    "boxes: 2 copies 1.004, so 1.0 (run-finders-boxes-4-falcon.timing.json); docs/falcon.md"),
)

BY_KEY = {m.key: m for m in LOCAL_MODELS}


def photo_setups() -> dict[str, tuple]:
    """scripts/time_finders.py SETUPS: key -> (finders model, task, variant, copies, rounds)."""
    return {m.key: (m.finders, m.task, m.variant, m.photo_copies, m.photo_rounds)
            for m in LOCAL_MODELS if m.photo_rounds is not None}


def screen_setups() -> dict[str, tuple]:
    """scripts/run_screens_local.py MODELS: key -> (finders model, task, variant, seeds, load factor)."""
    return {m.key: (m.finders, m.screen_task or m.task, m.variant, m.screen_seeds, m.load_factor)
            for m in LOCAL_MODELS if m.screen_seeds is not None}
