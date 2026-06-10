"""Convert the OpenReshoot reconstruction network to a Core ML fp16 .mlpackage.

Only the neural forward goes into Core ML (depth=None makes depth_alignment a
no-op). The NDC->metric unprojection + covariance SVD + 3DGS rendering stay
outside (Swift/Accelerate/Metal). Run on the Mac to de-risk before Xcode.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

from sharp.models import PredictorParams, create_predictor

ROOT = Path(__file__).resolve().parents[1]
W = 1536
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def log(msg):
    print(msg, flush=True)


log(f"[1/5] loading model on {DEV} ...")
model = create_predictor(PredictorParams())
state = torch.load(ROOT / "model" / "sharp_2572gikvuh.pt", weights_only=True)
model.load_state_dict(state)
model.eval().to(DEV)

image = torch.rand(1, 3, W, W, device=DEV)
disp = torch.tensor([1246.6742 / W], device=DEV)

log("[2/5] forward sanity ...")
t = time.time()
with torch.no_grad():
    ref = model(image, disp)
log(
    "  ok in %.1fs | outputs: %s"
    % (
        time.time() - t,
        {
            k: tuple(getattr(ref, k).shape)
            for k in ("mean_vectors", "singular_values", "quaternions", "colors", "opacities")
        },
    )
)

log("[3/5] tracing (torch.jit.trace, depth=None) ...")
t = time.time()
with torch.no_grad():
    traced = torch.jit.trace(model, (image, disp), check_trace=False)
log("  traced in %.1fs" % (time.time() - t))
traced.save(str(ROOT / "out" / "sharp_traced.pt"))
log("  saved out/sharp_traced.pt")

log("[4/5] converting to Core ML fp16 (this can take a while) ...")
t = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="image", shape=tuple(image.shape), dtype=np.float32),
        ct.TensorType(name="disparity_factor", shape=tuple(disp.shape), dtype=np.float32),
    ],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
log("  converted in %.1fs" % (time.time() - t))

log("[5/5] saving out/SHARP.mlpackage ...")
mlmodel.save(str(ROOT / "out" / "SHARP.mlpackage"))
log("DONE -> out/SHARP.mlpackage")
