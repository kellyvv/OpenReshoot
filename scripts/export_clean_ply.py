"""Export an OpenReshot reconstruction .ply as a clean single-element 3DGS .ply.

The reconstruction .ply carries extra PLY elements (extrinsic / intrinsic / image_size /
...). Some web viewers (e.g. antimatter15/splat) parse the header by summing
*every* `property` line across *all* elements, which corrupts the per-vertex
stride and renders garbage. This rewrites ONLY the `vertex` element, so any
standard 3DGS web viewer loads it correctly.

  --flip rotates 180 deg about X (OpenCV y-down/z-forward -> y-up), in case the
  target viewer expects a y-up world.

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import color_space as cs
from sharp.utils.gaussians import (
    apply_transform,
    convert_rgb_to_spherical_harmonics,
    load_ply,
)


@click.command()
@click.option("-i", "--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-o", "--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--flip/--no-flip", default=False, help="180deg about X (y-down -> y-up).")
def main(input_path: Path, output_path: Path, flip: bool) -> None:
    """Rewrite a reconstruction .ply with only the vertex element for web viewers."""
    g, _ = load_ply(input_path)
    if flip:
        g = apply_transform(g, torch.tensor([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0]]))

    # Recenter on the median point so a viewer's origin-looking default camera
    # frames the object (median is robust to far background outlier splats).
    center = g.mean_vectors[0].median(dim=0).values
    g = g._replace(mean_vectors=g.mean_vectors - center)

    xyz = g.mean_vectors[0].cpu().numpy()
    # Same vertex encoding as the model exporter: sRGB SH, logit opacity, log scale.
    f_dc = convert_rgb_to_spherical_harmonics(cs.linearRGB2sRGB(g.colors[0])).cpu().numpy()
    opacity = torch.logit(g.opacities[0].clamp(1e-6, 1 - 1e-6)).cpu().numpy()[:, None]
    scale = torch.log(g.singular_values[0]).cpu().numpy()
    rot = g.quaternions[0].cpu().numpy()

    data = np.concatenate([xyz, f_dc, opacity, scale, rot], axis=1).astype(np.float32)
    names = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    elements = np.empty(data.shape[0], dtype=[(n, "f4") for n in names])
    elements[:] = list(map(tuple, data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    print(f"wrote {output_path}  ({data.shape[0]} gaussians, single vertex element, flip={flip})")


if __name__ == "__main__":
    main()
