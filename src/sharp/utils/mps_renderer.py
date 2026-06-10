"""Pure-PyTorch forward rasterizer for 3D Gaussians (runs on MPS / CPU / CUDA).

This is a dependency-free, forward-only drop-in for the CUDA-only ``gsplat``
renderer in :mod:`sharp.utils.gsplat`, so that view synthesis works on Apple
Silicon (MPS) without CUDA. It implements EWA splatting:

    project means -> 2D, project covariance via the perspective Jacobian,
    then tile the image and alpha-composite gaussians front-to-back.

It returns the same :class:`~sharp.utils.gsplat.RenderingOutputs` (color, depth,
alpha) as ``GSplatRenderer``. The ``alpha`` channel is the accumulated opacity:
pixels where ``alpha`` is low are disocclusions ("holes") that the single input
view never observed -- exactly the mask to hand to a generative inpainter.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging

import torch

from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import (
    BackgroundColor,
    Gaussians3D,
    compose_covariance_matrices,
)
from sharp.utils.gsplat import RenderingOutputs

LOGGER = logging.getLogger(__name__)


class MPSGaussianRenderer:
    """Forward EWA-splatting renderer in pure PyTorch (MPS-friendly).

    Drop-in for :class:`sharp.utils.gsplat.GSplatRenderer`: call it with
    ``(gaussians, extrinsics=, intrinsics=, image_width=, image_height=)``.
    """

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "linearRGB",
        background_color: BackgroundColor = "black",
        tile_size: int = 64,
        chunk_size: int = 512,
        near: float = 1e-2,
        low_pass_filter: float = 0.3,
        cull_sigma: float = 3.0,
        max_radius_ratio: float = 0.5,
    ) -> None:
        """Initialize the renderer.

        Args:
            color_space: Color space the gaussian colors live in. ``linearRGB``
                is blended then converted to sRGB on output (matching SHARP).
            background_color: Background to composite over (only ``black`` and
                ``white`` are supported here).
            tile_size: Image is processed in ``tile_size`` x ``tile_size`` tiles
                to bound memory.
            chunk_size: Number of depth-sorted gaussians composited at once per
                tile. Lower it if memory is tight.
            near: Gaussians with camera-space depth below this are culled.
            low_pass_filter: Dilation added to the 2D covariance diagonal for
                anti-aliasing / invertibility (gsplat "classic" uses 0.3).
            cull_sigma: Gaussian footprint radius in standard deviations.
            max_radius_ratio: Splats whose pixel radius exceeds this fraction of
                the image diagonal are dropped (guards against degenerate
                near-camera gaussians blowing up a tile).
        """
        self.color_space = color_space
        self.background_color = background_color
        self.tile_size = tile_size
        self.chunk_size = chunk_size
        self.near = near
        self.low_pass_filter = low_pass_filter
        self.cull_sigma = cull_sigma
        self.max_radius_ratio = max_radius_ratio

    def __call__(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        """Render gaussians from one or more cameras.

        Args:
            gaussians: Batched gaussians (leading batch dim of size B).
            extrinsics: World->camera matrices, shape ``(B, 4, 4)`` (OpenCV).
            intrinsics: Camera intrinsics, ``(B, 4, 4)`` or ``(B, 3, 3)``.
            image_width: Output width in pixels.
            image_height: Output height in pixels.
        """
        if extrinsics.ndim == 2:
            extrinsics = extrinsics[None]
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]

        batch_size = len(gaussians.mean_vectors)
        colors, depths, alphas = [], [], []
        for ib in range(batch_size):
            color, depth, alpha = self._render_one(
                means=gaussians.mean_vectors[ib],
                quaternions=gaussians.quaternions[ib],
                singular_values=gaussians.singular_values[ib],
                opacities=gaussians.opacities[ib],
                gaussian_colors=gaussians.colors[ib],
                viewmat=extrinsics[ib],
                intrinsics=intrinsics[ib],
                width=image_width,
                height=image_height,
            )
            colors.append(color)
            depths.append(depth)
            alphas.append(alpha)

        return RenderingOutputs(
            color=torch.stack(colors, dim=0).contiguous(),
            depth=torch.stack(depths, dim=0).contiguous(),
            alpha=torch.stack(alphas, dim=0).contiguous(),
        )

    @torch.no_grad()
    def _render_one(
        self,
        means: torch.Tensor,
        quaternions: torch.Tensor,
        singular_values: torch.Tensor,
        opacities: torch.Tensor,
        gaussian_colors: torch.Tensor,
        viewmat: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = means.device
        dtype = torch.float32

        rotation = viewmat[:3, :3].to(dtype)
        translation = viewmat[:3, 3].to(dtype)
        fx = intrinsics[0, 0].to(dtype)
        fy = intrinsics[1, 1].to(dtype)
        cx = intrinsics[0, 2].to(dtype)
        cy = intrinsics[1, 2].to(dtype)

        # --- Transform to camera space and project. -------------------------
        p_cam = means.to(dtype) @ rotation.T + translation  # (N, 3)
        z = p_cam[:, 2]
        inv_z = 1.0 / z.clamp_min(1e-6)
        u = fx * p_cam[:, 0] * inv_z + cx
        v = fy * p_cam[:, 1] * inv_z + cy

        # --- 3D covariance -> camera space -> 2D via the projection Jacobian.
        cov3d = compose_covariance_matrices(quaternions.to(dtype), singular_values.to(dtype))
        cov_cam = rotation @ cov3d @ rotation.T  # (N, 3, 3)

        jac = torch.zeros((means.shape[0], 2, 3), device=device, dtype=dtype)
        jac[:, 0, 0] = fx * inv_z
        jac[:, 0, 2] = -fx * p_cam[:, 0] * inv_z * inv_z
        jac[:, 1, 1] = fy * inv_z
        jac[:, 1, 2] = -fy * p_cam[:, 1] * inv_z * inv_z
        cov2d = jac @ cov_cam @ jac.transpose(1, 2)  # (N, 2, 2)

        cov_a = cov2d[:, 0, 0] + self.low_pass_filter
        cov_b = cov2d[:, 0, 1]
        cov_c = cov2d[:, 1, 1] + self.low_pass_filter
        det = cov_a * cov_c - cov_b * cov_b

        # Inverse 2D covariance (conic) and footprint radius (largest eigenvalue).
        inv_det = 1.0 / det.clamp_min(1e-12)
        conic_a = cov_c * inv_det
        conic_b = -cov_b * inv_det
        conic_c = cov_a * inv_det
        mid = 0.5 * (cov_a + cov_c)
        lambda_max = mid + (mid * mid - det).clamp_min(0.0).sqrt()
        radius = (self.cull_sigma * lambda_max.clamp_min(0.0).sqrt()).ceil()

        # --- Frustum / bounds / sanity culling. -----------------------------
        diag = float((width**2 + height**2) ** 0.5)
        max_radius = self.max_radius_ratio * diag
        x_min, x_max = u - radius, u + radius
        y_min, y_max = v - radius, v + radius
        visible = (
            (z > self.near)
            & (det > 1e-12)
            & (radius > 0)
            & (radius < max_radius)
            & (x_max >= 0)
            & (x_min <= width - 1)
            & (y_max >= 0)
            & (y_min <= height - 1)
        )

        color_buf = torch.zeros((height, width, 3), device=device, dtype=dtype)
        depth_buf = torch.zeros((height, width), device=device, dtype=dtype)
        trans_buf = torch.ones((height, width), device=device, dtype=dtype)  # transmittance

        idx = visible.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            self._rasterize(
                idx=idx,
                u=u,
                v=v,
                z=z,
                conic_a=conic_a,
                conic_b=conic_b,
                conic_c=conic_c,
                opacities=opacities.to(dtype),
                gaussian_colors=gaussian_colors.to(dtype),
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                width=width,
                height=height,
                color_buf=color_buf,
                depth_buf=depth_buf,
                trans_buf=trans_buf,
            )

        alpha = 1.0 - trans_buf  # (H, W)
        depth = depth_buf / alpha.clamp_min(1e-8)
        color = self._compose_background(color_buf, alpha)

        if self.color_space == "linearRGB":
            color = cs_utils.linearRGB2sRGB(color)

        color = color.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()  # (3, H, W)
        return color, depth[None], alpha[None]

    def _rasterize(
        self,
        idx: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        z: torch.Tensor,
        conic_a: torch.Tensor,
        conic_b: torch.Tensor,
        conic_c: torch.Tensor,
        opacities: torch.Tensor,
        gaussian_colors: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        width: int,
        height: int,
        color_buf: torch.Tensor,
        depth_buf: torch.Tensor,
        trans_buf: torch.Tensor,
    ) -> None:
        device = u.device
        ts = self.tile_size

        # Gather the visible subset once.
        gu, gv, gz = u[idx], v[idx], z[idx]
        ga, gb, gc = conic_a[idx], conic_b[idx], conic_c[idx]
        gop = opacities[idx]
        gcol = gaussian_colors[idx]

        # Integer tile span each gaussian touches.
        n_tx = (width + ts - 1) // ts
        n_ty = (height + ts - 1) // ts
        tx0 = (x_min[idx] / ts).floor().clamp(0, n_tx - 1).to(torch.int64)
        tx1 = (x_max[idx] / ts).floor().clamp(0, n_tx - 1).to(torch.int64)
        ty0 = (y_min[idx] / ts).floor().clamp(0, n_ty - 1).to(torch.int64)
        ty1 = (y_max[idx] / ts).floor().clamp(0, n_ty - 1).to(torch.int64)

        for tyi in range(n_ty):
            in_rows = (ty0 <= tyi) & (ty1 >= tyi)
            if not bool(in_rows.any()):
                continue
            y0 = tyi * ts
            y1 = min(y0 + ts, height)
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
            for txi in range(n_tx):
                member = in_rows & (tx0 <= txi) & (tx1 >= txi)
                m = member.nonzero(as_tuple=False).squeeze(1)
                if m.numel() == 0:
                    continue
                x0 = txi * ts
                x1 = min(x0 + ts, width)
                xs = torch.arange(x0, x1, device=device, dtype=torch.float32)

                # Sort this tile's gaussians front-to-back.
                order = torch.argsort(gz[m])
                m = m[order]
                self._composite_tile(
                    gu=gu[m],
                    gv=gv[m],
                    gz=gz[m],
                    ga=ga[m],
                    gb=gb[m],
                    gc=gc[m],
                    gop=gop[m],
                    gcol=gcol[m],
                    xs=xs,
                    ys=ys,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    color_buf=color_buf,
                    depth_buf=depth_buf,
                    trans_buf=trans_buf,
                )

    def _composite_tile(
        self,
        gu: torch.Tensor,
        gv: torch.Tensor,
        gz: torch.Tensor,
        ga: torch.Tensor,
        gb: torch.Tensor,
        gc: torch.Tensor,
        gop: torch.Tensor,
        gcol: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        color_buf: torch.Tensor,
        depth_buf: torch.Tensor,
        trans_buf: torch.Tensor,
    ) -> None:
        grid_x = xs[None, :]  # (1, tw)
        grid_y = ys[:, None]  # (th, 1)
        n = gu.shape[0]

        trans = torch.ones((y1 - y0, x1 - x0), device=gu.device, dtype=torch.float32)
        color = torch.zeros((y1 - y0, x1 - x0, 3), device=gu.device, dtype=torch.float32)
        depth = torch.zeros((y1 - y0, x1 - x0), device=gu.device, dtype=torch.float32)

        for s in range(0, n, self.chunk_size):
            e = min(n, s + self.chunk_size)
            du = grid_x[None] - gu[s:e, None, None]  # (g, th, tw)
            dv = grid_y[None] - gv[s:e, None, None]
            power = (
                -0.5 * (ga[s:e, None, None] * du * du + gc[s:e, None, None] * dv * dv)
                - gb[s:e, None, None] * du * dv
            )
            alpha = (gop[s:e, None, None] * torch.exp(power)).clamp(max=0.99)
            alpha = torch.where(power <= 0, alpha, torch.zeros_like(alpha))

            one_minus = 1.0 - alpha
            incl = torch.cumprod(one_minus, dim=0)
            excl = torch.cat([torch.ones_like(incl[:1]), incl[:-1]], dim=0)
            weight = trans[None] * alpha * excl  # (g, th, tw)

            color = color + (weight[..., None] * gcol[s:e, None, None, :]).sum(dim=0)
            depth = depth + (weight * gz[s:e, None, None]).sum(dim=0)
            trans = trans * incl[-1]

        color_buf[y0:y1, x0:x1] += color
        depth_buf[y0:y1, x0:x1] += depth
        trans_buf[y0:y1, x0:x1] *= trans

    def _compose_background(self, color: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.background_color == "black":
            return color
        if self.background_color == "white":
            return color + (1.0 - alpha)[..., None]
        raise ValueError(
            f"MPSGaussianRenderer only supports black/white background, got {self.background_color}"
        )
