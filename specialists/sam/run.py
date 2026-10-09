"""Smoke test + latency + throughput for SAM 1, SAM 2.1 and SAM 3 (image model).

uv run run.py MODE      (MODE: see MODES below, or "all")

Modes:
  *-auto  "segment everything": the automatic mask generator (a grid of
          points over the whole image). No prompt needed; one image at a time.
  *-box   one box per image (here: the middle half of the picture, a stand-in
          for a box another model would supply). Image encoder runs batched.
  sam3-text  a short text phrase per image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import harness  # noqa: E402  (pins the GPU before torch loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

WEIGHTS = harness.ROOT / "weights"
BOX_BATCHES = (1, 2, 4, 8, 16, 32)



def middle_box(w: int, h: int) -> list[float]:
    return [w * 0.25, h * 0.25, w * 0.75, h * 0.75]


def area_frac(mask: np.ndarray) -> dict:
    return {"mask_area_frac": round(float(mask.mean()), 3)}


# --- SAM 1 ------------------------------------------------------------------
def sam1_load(variant: str = "vit_h"):
    from segment_anything import sam_model_registry

    ckpt = {"vit_h": "sam_vit_h_4b8939.pth", "vit_b": "sam_vit_b_01ec64.pth"}[variant]
    return sam_model_registry[variant](checkpoint=str(WEIGHTS / "sam1" / ckpt)).cuda().eval()


def sam1_auto():
    from segment_anything import SamAutomaticMaskGenerator

    def load():
        return SamAutomaticMaskGenerator(sam1_load())

    def predict(gen, ims):
        with torch.inference_mode():
            return [gen.generate(np.asarray(im)) for im in ims]

    harness.run(
        model_id="sam1-vit-h-auto",
        config={"weights": "sam_vit_h_4b8939.pth (md5 4b8939a8…)", "precision": "fp32 (official)",
                "generator": "SamAutomaticMaskGenerator defaults (32×32 points)"},
        load=load, predict=predict, summarize=lambda r: {"masks": len(r)}, batch_sizes=(1,),
        notes="Automatic mode runs one image at a time; post-processing is on the CPU.",
    )


def sam1_box(precision: str):
    from segment_anything.utils.transforms import ResizeLongestSide

    def load():
        sam = sam1_load()
        return sam, ResizeLongestSide(sam.image_encoder.img_size)

    def predict(state, ims):
        sam, tr = state
        batch = []
        for im in ims:
            a = np.asarray(im)
            h, w = a.shape[:2]
            x = torch.as_tensor(tr.apply_image(a), device="cuda").permute(2, 0, 1).contiguous()
            box = torch.tensor([middle_box(w, h)], device="cuda", dtype=torch.float)
            batch.append({"image": x, "original_size": (h, w), "boxes": tr.apply_boxes_torch(box, (h, w))})
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            out = sam(batch, multimask_output=False)
        return [o["masks"][0, 0].cpu().numpy() for o in out]

    harness.run(
        model_id=f"sam1-vit-h-box-{precision}",
        config={"weights": "sam_vit_h_4b8939.pth", "precision": precision,
                "prompt": "one box: middle half of the image"},
        load=load, predict=predict, summarize=area_frac, batch_sizes=BOX_BATCHES,
    )


# --- SAM 2.1 ------------------------------------------------------------------
SAM2_REPO = "facebook/sam2.1-hiera-large"


def sam2_auto():
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    def load():
        return SAM2AutomaticMaskGenerator.from_pretrained(SAM2_REPO)

    def predict(gen, ims):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return [gen.generate(np.asarray(im)) for im in ims]

    harness.run(
        model_id="sam2.1-large-auto",
        config={"weights": SAM2_REPO, "precision": "bf16 autocast (official examples)",
                "generator": "SAM2AutomaticMaskGenerator defaults"},
        load=load, predict=predict, summarize=lambda r: {"masks": len(r)}, batch_sizes=(1,),
        notes="Automatic mode runs one image at a time; post-processing is on the CPU.",
    )


def sam2_box():
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    def load():
        return SAM2ImagePredictor.from_pretrained(SAM2_REPO)

    def predict(pred, ims):
        arrays = [np.asarray(im) for im in ims]
        boxes = [np.array([middle_box(a.shape[1], a.shape[0])]) for a in arrays]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred.set_image_batch(arrays)
            masks, _, _ = pred.predict_batch(box_batch=boxes, multimask_output=False)
        return [np.asarray(m).reshape(-1, *m.shape[-2:])[0] > 0 for m in masks]

    harness.run(
        model_id="sam2.1-large-box",
        config={"weights": SAM2_REPO, "precision": "bf16 autocast (official examples)",
                "prompt": "one box: middle half of the image"},
        load=load, predict=predict, summarize=area_frac, batch_sizes=BOX_BATCHES,
    )


# --- SAM 3 ------------------------------------------------------------------
def sam3_text():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    def load():
        return Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)

    def predict(proc, ims):
        out = []
        for im in ims:
            prompt = harness.prompt_for(im)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = proc.set_image(im)
                st = proc.set_text_prompt(prompt, st)
            out.append({
                "prompt": prompt,
                "masks": st["masks"].cpu().numpy(),
                "scores": st["scores"].float().cpu().numpy(),
            })
        return out

    harness.run(
        model_id="sam3-text",
        config={"weights": "facebook/sam3 (sam3.pt)", "precision": "bf16 autocast (official examples)",
                "confidence_threshold": 0.5, "prompt": "short phrase per image"},
        load=load, predict=predict,
        summarize=lambda r: {"prompt": r["prompt"], "masks": int(len(r["masks"])),
                             "top_score": round(float(r["scores"].max()), 3) if len(r["scores"]) else None},
        batch_sizes=(1,),
        notes="The official image processor grounds one image per call; batch > 1 not supported by it.",
    )


MODES = {
    "sam1-auto": sam1_auto,
    "sam1-box-fp32": lambda: sam1_box("fp32"),
    "sam1-box-bf16": lambda: sam1_box("bf16"),
    "sam2-auto": sam2_auto,
    "sam2-box": sam2_box,
    "sam3-text": sam3_text,
}



# --- automatic modes with several workers sharing the GPU ----------------------
def _make_sam1_auto():
    from segment_anything import SamAutomaticMaskGenerator

    gen = SamAutomaticMaskGenerator(sam1_load())

    def predict(ims):
        with torch.inference_mode():
            return [gen.generate(np.asarray(im)) for im in ims]
    return predict


def _make_sam2_auto():
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    gen = SAM2AutomaticMaskGenerator.from_pretrained(SAM2_REPO)

    def predict(ims):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return [gen.generate(np.asarray(im)) for im in ims]
    return predict


def auto_concurrent():
    # 3 workers do not fit for SAM 1 ViT-H in 24 GB (measured 2026-10-08); 2 bring the GPU to ~91 % busy.
    for workers in (2,):
        harness.run_concurrent(model_id="sam1-vit-h-auto", workers=workers, make_predict=_make_sam1_auto,
                               config={"weights": "sam_vit_h_4b8939.pth", "precision": "fp32"})
        harness.run_concurrent(model_id="sam2.1-large-auto", workers=workers, make_predict=_make_sam2_auto,
                               config={"weights": SAM2_REPO, "precision": "bf16 autocast"})


MODES["auto-concurrent"] = auto_concurrent

if __name__ == "__main__":
    mode = sys.argv[1]
    for name in ([m for m in MODES if m != "auto-concurrent"] if mode == "all" else [mode]):
        print(f"== {name}", flush=True)
        MODES[name]()
