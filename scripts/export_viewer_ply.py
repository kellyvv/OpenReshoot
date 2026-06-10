"""Re-orient an OpenReshoot reconstruction .ply for standard y-up web viewers.

The model exports gaussians in OpenCV camera space (x right, y DOWN, z forward), so
public viewers (SuperSplat, antimatter15, ...) show them upside-down. This
rotates 180 deg about X to get y-up and recenters the scene at the origin,
producing a .ply that loads upright. Output is still a standard 3DGS .ply.

Usage:
  python scripts/export_viewer_ply.py -i out/koala_ply/koala.ply -o out/koala_viewer.ply

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

from pathlib import Path

import click
import torch

from sharp.utils.gaussians import apply_transform, load_ply, save_ply


@click.command()
@click.option("-i", "--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-o", "--output-path", type=click.Path(path_type=Path), required=True)
def main(input_path: Path, output_path: Path) -> None:
    """Rotate OpenCV-space gaussians to y-up and recenter for web viewers."""
    gaussians, meta = load_ply(input_path)

    # 180 deg about X: (x, y, z) -> (x, -y, -z). Proper rotation (no mirroring),
    # turns y-down/z-forward (OpenCV) into y-up/z-toward-viewer (OpenGL/viewers).
    flip = torch.tensor([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0]])
    gaussians = apply_transform(gaussians, flip)

    # Recenter on the median point so the viewer frames it nicely.
    center = gaussians.mean_vectors[0].median(dim=0).values
    gaussians = gaussians._replace(mean_vectors=gaussians.mean_vectors - center)

    width, height = int(meta.resolution_px[0]), int(meta.resolution_px[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, meta.focal_length_px, (height, width), output_path)
    print(
        f"wrote {output_path}  ({gaussians.mean_vectors.shape[1]} gaussians, y-up, recentered)"
    )


if __name__ == "__main__":
    main()
