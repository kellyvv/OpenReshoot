"""Validate the Core ML model output matches PyTorch on a real image."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlpackage", type=Path, default=ROOT / "out" / "SHARP.mlpackage")
    parser.add_argument("--image", type=Path, default=ROOT / "out" / "koala.png")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    return parser.parse_args()


args = parse_args()
W = args.width

# --- preprocess a real image exactly like predict_image ---
img, _, f_px = sio.load_rgb(args.image)
H0, W0 = img.shape[:2]
x = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
x = F.interpolate(x[None], size=(W, W), mode="bilinear", align_corners=True)
disp = torch.tensor([f_px / W0]).float()

# --- PyTorch reference ---
model = create_predictor(PredictorParams())
model.load_state_dict(torch.load(ROOT / "model" / "sharp_2572gikvuh.pt", weights_only=True))
model.eval()
with torch.no_grad():
    ref = model(x, disp)
ref_t = [ref.mean_vectors, ref.singular_values, ref.quaternions, ref.colors, ref.opacities]
names = ["mean", "sv", "quat", "color", "opacity"]

# --- Core ML ---
ml = ct.models.MLModel(str(args.mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
out = ml.predict(
    {"image": x.numpy().astype(np.float32), "disparity_factor": disp.numpy().astype(np.float32)}
)
spec_names = [o.name for o in ml._spec.description.output]
print("coreml output order:", spec_names)
print("coreml output shapes:", {k: np.array(v).shape for k, v in out.items()})
print()

ok = True
for i, nm in enumerate(names):
    cm = np.array(out[spec_names[i]]).astype(np.float32)
    rt = ref_t[i].numpy().astype(np.float32)
    if cm.shape != rt.shape:
        print(f"  {nm:8s} SHAPE MISMATCH coreml{cm.shape} vs torch{rt.shape}")
        ok = False
        continue
    d = np.abs(cm - rt)
    denom = np.abs(rt).mean() + 1e-8
    rel = d.mean() / denom
    flag = "" if rel < 0.02 else "  <-- CHECK"
    print(f"  {nm:8s} {str(cm.shape):18s} max|Δ|={d.max():.4g}  mean|Δ|={d.mean():.4g}  rel={rel:.4g}{flag}")
    if rel >= 0.05:
        ok = False
print()
print("PARITY:", "OK (fp16 close to fp32)" if ok else "MISMATCH — needs fixing")
