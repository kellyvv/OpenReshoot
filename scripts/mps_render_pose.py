"""Render an OpenReshoot reconstruction scene from off-axis camera poses.

This utility renders the scene from one or more shifted camera positions and,
for each, writes:

  * <name>.color.png   - the new-angle photo
  * <name>.alpha.png   - accumulated opacity (white = observed, black = hole)
  * <name>.holes.png   - the color with disocclusion holes painted magenta

The holes are exactly the regions the single input photo never saw; they are
the inpainting mask to hand to a generative image model (GPT-Image / Gemini)
for a photorealistic, geometry-grounded "re-shoot" at the new angle.

Usage:
  python scripts/mps_render_pose.py -i out/teaser.ply -o out/poses \\
      --sweep-x -0.3 0.3 --steps 5

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image

from sharp.utils import camera
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import load_ply
from sharp.utils.mps_renderer import MPSGaussianRenderer

LOGGER = logging.getLogger(__name__)


def _pick_device(name: str) -> torch.device:
    if name == "default":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    return torch.device(name)


def _save_outputs(out_dir: Path, name: str, color: np.ndarray, alpha: np.ndarray) -> None:
    Image.fromarray((color * 255).astype(np.uint8)).save(out_dir / f"{name}.color.png")
    Image.fromarray((alpha * 255).astype(np.uint8)).save(out_dir / f"{name}.alpha.png")
    holes = color.copy()
    hole_mask = alpha < 0.5
    holes[hole_mask] = np.array([1.0, 0.0, 1.0])  # magenta = inpaint here
    Image.fromarray((holes * 255).astype(np.uint8)).save(out_dir / f"{name}.holes.png")


@click.command()
@click.option("-i", "--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-o", "--output-path", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--device", type=str, default="default", help="['default','mps','cpu','cuda']")
@click.option("--lookat", type=click.Choice(["point", "ahead"]), default="point")
@click.option(
    "--sweep-x",
    nargs=2,
    type=float,
    default=(-0.15, 0.15),
    help="Lateral camera sweep range in meters (min max).",
)
@click.option("--dy", type=float, default=0.0, help="Vertical offset in meters.")
@click.option("--dz", type=float, default=0.0, help="Forward/backward offset in meters.")
@click.option("--steps", type=int, default=5, help="Number of poses across the sweep.")
@click.option("-v", "--verbose", is_flag=True)
def main(input_path, output_path, device, lookat, sweep_x, dy, dz, steps, verbose):
    """Render off-axis views of a reconstruction scene, exporting hole masks."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    dev = _pick_device(device)
    output_path.mkdir(parents=True, exist_ok=True)

    gaussians, meta = load_ply(input_path)
    gaussians = gaussians.to(dev)
    width, height = int(meta.resolution_px[0]), int(meta.resolution_px[1])
    f_px = float(meta.focal_length_px)
    LOGGER.info(
        "Loaded %d gaussians | %dx%d | f=%.1f | device=%s",
        gaussians.mean_vectors.shape[1],
        width,
        height,
        f_px,
        dev,
    )

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=dev,
        dtype=torch.float32,
    )

    # Report the nearby-view baseline budget the reconstruction model is designed for.
    budget = camera.compute_max_offset(
        gaussians, camera.TrajectoryParams(), (width, height), f_px
    )
    LOGGER.info(
        "Suggested nearby-view budget: lateral +/-%.3fm, forward +/-%.3fm",
        float(budget[0]),
        float(budget[2]),
    )

    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=(width, height), lookat_mode=lookat
    )
    renderer = MPSGaussianRenderer(color_space=meta.color_space)

    panels = []
    xs = np.linspace(sweep_x[0], sweep_x[1], steps) if steps > 1 else [0.0]
    for i, dx in enumerate(xs):
        eye = torch.tensor([float(dx), float(dy), float(dz)], dtype=torch.float32)
        info = camera_model.compute(eye)
        out = renderer(
            gaussians,
            extrinsics=info.extrinsics[None].to(dev),
            intrinsics=info.intrinsics[None].to(dev),
            image_width=info.width,
            image_height=info.height,
        )
        if dev.type == "mps":
            torch.mps.synchronize()
        color = out.color[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        alpha = out.alpha[0, 0].cpu().numpy()
        name = f"x{dx:+.3f}"
        _save_outputs(output_path, name, color, alpha)
        hole_pct = float((alpha < 0.5).mean()) * 100
        LOGGER.info("[%d/%d] dx=%+.3fm -> holes %.2f%% (%s)", i + 1, steps, dx, hole_pct, name)
        panels.append((color, alpha))

    # Contact sheet: row 1 = color, row 2 = hole overlay.
    if panels:
        row_color = np.concatenate([c for c, _ in panels], axis=1)
        row_holes = np.concatenate(
            [np.where((a < 0.5)[..., None], np.array([1.0, 0.0, 1.0]), c) for c, a in panels],
            axis=1,
        )
        sheet = np.concatenate([row_color, row_holes], axis=0)
        Image.fromarray((sheet * 255).astype(np.uint8)).save(output_path / "contact_sheet.png")
        LOGGER.info("Saved contact sheet -> %s", output_path / "contact_sheet.png")


if __name__ == "__main__":
    main()
