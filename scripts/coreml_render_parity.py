"""Render PyTorch vs Core ML gaussians and compare — the decisive parity test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from PIL import Image

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io as sio
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
from sharp.utils.mps_renderer import MPSGaussianRenderer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 1536
dev = torch.device("mps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlpackage", type=Path, default=ROOT / "out" / "SHARP-sdpa.mlpackage")
    parser.add_argument("--image", type=Path, default=ROOT / "out" / "koala.png")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "coreml_parity.png")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    return parser.parse_args()


args = parse_args()
W = args.width

img, _, f_px = sio.load_rgb(args.image)
H0, W0 = img.shape[:2]
x = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
x = F.interpolate(x[None], size=(W, W), mode="bilinear", align_corners=True)
disp = torch.tensor([f_px / W0]).float()

# intrinsics for unprojection (same as predict_image)
intr = torch.tensor([[f_px, 0, W0 / 2, 0], [0, f_px, H0 / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
intr_r = intr.clone()
intr_r[0] *= W / W0
intr_r[1] *= W / H0


def unproj(g_ndc):
    return unproject_gaussians(g_ndc, torch.eye(4), intr_r, (W, W))


# --- PyTorch ---
model = create_predictor(PredictorParams())
model.load_state_dict(torch.load(ROOT / "model" / "sharp_2572gikvuh.pt", weights_only=True))
model.eval()
with torch.no_grad():
    g_pt = model(x, disp)
g_pt = unproj(g_pt)

# --- Core ML -> assemble Gaussians3D ---
ml = ct.models.MLModel(str(args.mlpackage))
out = ml.predict({"image": x.numpy().astype(np.float32), "disparity_factor": disp.numpy().astype(np.float32)})
sn = [o.name for o in ml._spec.description.output]
t = lambda i: torch.from_numpy(np.array(out[sn[i]]).astype(np.float32))
g_cm = Gaussians3D(mean_vectors=t(0), singular_values=t(1), quaternions=t(2), colors=t(3), opacities=t(4))
g_cm = unproj(g_cm)

# --- render both at identity ---
RW, RH = W0, H0
K = torch.tensor([[f_px, 0, RW / 2], [0, f_px, RH / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
E = torch.eye(4, device=dev)
r = MPSGaussianRenderer(color_space="linearRGB")


def render(g):
    o = r(g.to(dev), extrinsics=E, intrinsics=K, image_width=RW, image_height=RH)
    torch.mps.synchronize()
    return o.color[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


a = render(g_pt)
b = render(g_cm)
mse = float(((a - b) ** 2).mean())
psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
print(f"\nrender PSNR (PyTorch vs Core ML): {psnr:.2f} dB  (higher=identical; >35 = visually indistinguishable)")
sbs = (np.concatenate([a, b], axis=1) * 255).astype(np.uint8)
args.output.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(sbs).save(args.output)
print(f"saved {args.output} (left=PyTorch, right=Core ML)")
