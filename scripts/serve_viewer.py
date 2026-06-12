"""OpenReshot local server: upload a photo -> reconstruct on MPS -> parallax viewer.

Serves the viewer at /splat/ and exposes POST /api/predict, which takes raw image
bytes, runs the reconstruction model, writes a clean single-element .ply + a downscaled photo under
/uploads/, and returns JSON config for the viewer. The ~2.8GB model is loaded once
at startup and reused for every upload.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/serve_viewer.py

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import io as pyio
import json
import logging
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from google import genai
from google.genai import types
from PIL import Image
from plyfile import PlyData, PlyElement

from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import color_space as cs
from sharp.utils import io as sio
from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("serve")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
UP = OUT / "uploads"
UP.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LOG.info("Loading reconstruction model once on %s ...", DEVICE)
_state = torch.load(ROOT / "model" / "sharp_2572gikvuh.pt", weights_only=True)
MODEL = create_predictor(PredictorParams())
MODEL.load_state_dict(_state)
MODEL.eval()
MODEL.to(DEVICE)
LOG.info("Model ready.")

# Serialize model inference: concurrent MPS forward passes can deadlock.
_PREDICT_LOCK = threading.Lock()

ENHANCE_PROMPT = (
    "This image is a novel-view render produced from a 3D Gaussian Splatting reconstruction. "
    "Because the camera viewpoint changed, some newly exposed edges, disoccluded regions, "
    "stretched areas, warped details, holes, and blurry splat artifacts may appear. "
    "Repair only those rendering artifacts. Fill missing or extended areas with realistic "
    "detail consistent with the surrounding scene and the original photo. Keep the original "
    "subject, layout, camera framing, lighting, colors, materials, and composition unchanged. "
    "Do not restyle, beautify, replace, or add new main objects. If people are visible, "
    "preserve them exactly as shown: same identity, appearance, hair, clothing, pose, "
    "expression, and framing. Do not alter people or create any new person. Output one single "
    "complete clear photo. 这是 3DGS 新视角渲染结果，只修复模糊、拉伸、露底、空洞和缺失区域；"
    "如果画面里有人，人物必须保持原样。"
)


def _gemini_enhance(image_bytes: bytes, key: str | None = None) -> bytes:
    """Send the current-angle frame to Gemini to fill blurry holes and sharpen.

    The key comes from the page (X-Gemini-Key header) or falls back to the env var.
    """
    key = (key or "").strip() or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key — set it in the page (top-right key button) or GEMINI_API_KEY env")
    client = genai.Client(api_key=key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=ENHANCE_PROMPT),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        ),
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        image_config=types.ImageConfig(image_size="1K"),
        response_modalities=["IMAGE", "TEXT"],
    )
    out = None
    for chunk in client.models.generate_content_stream(
        model="gemini-3.1-flash-image", contents=contents, config=config
    ):
        if not chunk.parts:
            continue
        part = chunk.parts[0]
        if part.inline_data and part.inline_data.data:
            out = part.inline_data.data
        elif getattr(chunk, "text", None):
            LOG.info("gemini: %s", chunk.text)
    if out is None:
        raise RuntimeError("Gemini returned no image (refused or safety-blocked?)")
    return out


def _clean_ply_bytes(g) -> bytes:
    """Encode gaussians as a standard single-element 3DGS .ply (web-viewer ready)."""
    xyz = g.mean_vectors[0].cpu().numpy()
    f_dc = convert_rgb_to_spherical_harmonics(cs.linearRGB2sRGB(g.colors[0])).cpu().numpy()
    op = torch.logit(g.opacities[0].clamp(1e-6, 1 - 1e-6)).cpu().numpy()[:, None]
    sc = torch.log(g.singular_values[0]).cpu().numpy()
    rot = g.quaternions[0].cpu().numpy()
    data = np.concatenate([xyz, f_dc, op, sc, rot], axis=1).astype(np.float32)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    el = np.empty(data.shape[0], dtype=[(n, "f4") for n in names])
    el[:] = list(map(tuple, data))
    buf = pyio.BytesIO()
    PlyData([PlyElement.describe(el, "vertex")]).write(buf)
    return buf.getvalue()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        # Never cache the html/js so edits always take effect (the .ply/.jpg still cache).
        p = self.path.split("?")[0]
        if p.endswith((".html", ".js")) or p.endswith("/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path.startswith("/api/enhance"):
            self._enhance()
            return
        if not self.path.startswith("/api/predict"):
            self.send_error(404)
            return
        try:
            q = parse_qs(urlparse(self.path).query)
            name = q.get("name", ["photo.jpg"])[0]
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            stem = "".join(c for c in Path(name).stem if c.isalnum()) or "photo"
            suf = Path(name).suffix.lower() or ".jpg"
            ipath = UP / (stem + suf)
            ipath.write_bytes(raw)
            LOG.info("predict %s (%d bytes)", ipath.name, n)

            img, _, f_px = sio.load_rgb(ipath)
            height, width = int(img.shape[0]), int(img.shape[1])
            with _PREDICT_LOCK:
                gaussians = predict_image(MODEL, img, f_px, DEVICE)

            (UP / f"{stem}.ply").write_bytes(_clean_ply_bytes(gaussians))
            pim = Image.fromarray(img)
            pim.thumbnail((1000, 1000))
            pim.convert("RGB").save(UP / f"{stem}.jpg", quality=85)

            med = float(gaussians.mean_vectors[0, :, 2].clamp_min(1e-3).median())
            resp = {
                "url": f"/uploads/{stem}.ply",
                "photo": f"/uploads/{stem}.jpg",
                "fx": round(float(f_px), 3),
                "fy": round(float(f_px), 3),
                "W": width,
                "H": height,
                "focus": round(med, 3),
            }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            LOG.info("done -> %s", resp)
        except Exception as e:  # noqa: BLE001
            LOG.exception("predict failed")
            self.send_error(500, str(e))

    def _enhance(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            page_key = self.headers.get("X-Gemini-Key")
            LOG.info("enhance request (%d bytes, key from %s)", n, "page" if page_key else "env")
            out = _gemini_enhance(raw, page_key)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            LOG.info("enhance done -> %d bytes", len(out))
        except Exception as e:  # noqa: BLE001
            LOG.exception("enhance failed")
            self.send_error(500, str(e))


if __name__ == "__main__":
    LOG.info("Serving on http://127.0.0.1:8765  (viewer at /splat/)")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
