"""
Upscale Module V2: Handcrafted Convex Upsampler (HCU)
Replaces the Guided Filter with a Triton-native tile-level bilateral upsampler.
Also provides block mean luma extraction.
"""
import torch
import torch.nn.functional as F
from kernels.hcu_kernel import run_hcu_upsample


def compute_block_mean_luma(luma_hr, block_size=8):
    """
    Compute the mean luma for each 8x8 coarse block.
    Uses avg_pool2d for speed — no edge aliasing from center-pixel sampling.

    Args:
        luma_hr: (B, H, W) float32, full-res luma [0, 1]
        block_size: int

    Returns:
        mean_luma: (B, GH, GW) float32
    """
    B, H, W = luma_hr.shape
    # avg_pool2d expects (B, 1, H, W)
    pooled = F.avg_pool2d(
        luma_hr.unsqueeze(1),
        kernel_size=block_size,
        stride=block_size,
        padding=0,
    )
    return pooled.squeeze(1)  # (B, GH, GW)


def hcu_upsample(dx_px, dy_px, cost_grid, luma_hr, block_size=8,
                 alpha=10.0, beta=5.0, gamma=2.0, eps=1e-3):
    """
    HCU edge-aware upsampling: block mean luma + Triton bilateral kernel.

    Args:
        dx_px, dy_px: (B, GH, GW) float32 — coarse flow in pixel units
        cost_grid: (B, GH, GW) float32/int32 — JFA cost
        luma_hr: (B, H, W) float32 — full-res luma [0, 1]
        block_size: int
        alpha, beta, gamma, eps: HCU tunables

    Returns:
        dx_hr, dy_hr: (B, H, W) float32 — full-res flow
    """
    mean_luma = compute_block_mean_luma(luma_hr, block_size)
    dx_hr, dy_hr = run_hcu_upsample(
        dx_px, dy_px, cost_grid.float(), mean_luma,
        luma_hr, block_size=block_size,
        alpha=alpha, beta=beta, gamma=gamma, eps=eps,
    )
    return dx_hr, dy_hr
