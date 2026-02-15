"""
Refinement Pipeline V3: Median → Sub-Pixel → Guided Upscale + Temporal Damping
Post-processes raw JFA block-grid flow into smooth full-res flow.
Includes temporal EMA to suppress frame-to-frame flicker.
"""
import torch
import torch.nn.functional as F
from kernels.refine_kernel import run_vector_median
from kernels.subpixel_kernel import run_subpixel_refine
from kernels.census_kernel import run_census
from core.upscale import guided_upsample


# Persistent state for temporal damping
_prev_dx_full = None
_prev_dy_full = None


def reset_temporal():
    """Reset temporal state (call at start of new video)."""
    global _prev_dx_full, _prev_dy_full
    _prev_dx_full = None
    _prev_dy_full = None


def _temporal_damp(dx_full, dy_full, cost_grid, block_size=8, jump_thresh=15.0, cost_bypass=2):
    """
    Adaptive EMA temporal damping.
    Alpha = clamp(1.0 - cost/16.0, 0.1, 0.9) per pixel.
    High confidence (low cost) → trust new data (high alpha).
    Low confidence (high cost) → trust history (low alpha).

    Args:
        dx_full, dy_full: (B, H, W) current frame flow
        cost_grid: (B, GH, GW) int32 JFA cost on block grid
        block_size, jump_thresh, cost_bypass: parameters
    Returns:
        dx_damped, dy_damped: (B, H, W)
    """
    global _prev_dx_full, _prev_dy_full

    if _prev_dx_full is None or _prev_dx_full.shape != dx_full.shape:
        _prev_dx_full = dx_full.detach().clone()
        _prev_dy_full = dy_full.detach().clone()
        return dx_full, dy_full

    # Compute per-pixel magnitude jump
    delta_dx = dx_full - _prev_dx_full
    delta_dy = dy_full - _prev_dy_full
    jump_mag = torch.sqrt(delta_dx ** 2 + delta_dy ** 2)

    # Upscale cost to full res for adaptive alpha
    B, H, W = dx_full.shape
    cost_full = F.interpolate(
        cost_grid.unsqueeze(1).float(), (H, W), mode='nearest'
    ).squeeze(1)

    # Adaptive alpha: high confidence → trust new, low confidence → trust history
    alpha = (1.0 - cost_full / 16.0).clamp(0.1, 0.9)

    # Jump detected AND cost is not extremely low → damp with adaptive alpha
    needs_damping = (jump_mag > jump_thresh) & (cost_full >= cost_bypass)

    # EMA blend: alpha * current + (1-alpha) * previous
    dx_damped = torch.where(needs_damping, alpha * dx_full + (1 - alpha) * _prev_dx_full, dx_full)
    dy_damped = torch.where(needs_damping, alpha * dy_full + (1 - alpha) * _prev_dy_full, dy_full)

    # Update state
    _prev_dx_full = dx_damped.detach().clone()
    _prev_dy_full = dy_damped.detach().clone()

    return dx_damped, dy_damped


def refine_flow(dx_grid, dy_grid, conf_grid, img1, img2, block_size=8, temporal=True):
    """
    Full refinement pipeline V3: median → sub-pixel → guided upscale → temporal damp.

    Args:
        dx_grid, dy_grid: (B, GH, GW) float32, pixel-scaled flow on block grid
        conf_grid: (B, GH, GW) uint8, JFA cost
        img1, img2: (B, C, H, W) float32 normalized input frames
        block_size: int
        temporal: bool, enable temporal EMA damping

    Returns:
        dx_full, dy_full: (B, H, W) float32 refined flow in pixels
    """
    B, C, H, W = img1.shape
    device = img1.device

    # --- Step 1: Vector Median Filter on grid-level vectors ---
    dx_int = (dx_grid / block_size).to(torch.int32)
    dy_int = (dy_grid / block_size).to(torch.int32)
    cost_int = conf_grid.to(torch.int32)

    dx_med, dy_med = run_vector_median(dx_int, dy_int, cost_int, cost_threshold=12)
    torch.cuda.synchronize()

    # --- Step 2: Compute block-grid Census (for sub-pixel) ---
    luma1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
    luma2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]

    census1_full, _ = run_census(luma1)
    census2_full, _ = run_census(luma2)
    torch.cuda.synchronize()

    census1_grid = census1_full[:, ::block_size, ::block_size].contiguous()
    census2_grid = census2_full[:, ::block_size, ::block_size].contiguous()

    # --- Step 3: Sub-Pixel Quadratic Refinement (with guardrails) ---
    dx_sub, dy_sub = run_subpixel_refine(
        dx_med, dy_med, cost_int, census1_grid, census2_grid, max_cost=8
    )
    torch.cuda.synchronize()

    # Scale to pixel units
    dx_px = dx_sub * block_size
    dy_px = dy_sub * block_size

    # --- Step 4: Guided Filter Upscale ---
    confidence = torch.exp(-cost_int.float() / 4.0)
    guide_hr = luma1

    dx_full = guided_upsample(dx_px, guide_hr, confidence_lr=confidence, radius=4, eps=0.01)
    dy_full = guided_upsample(dy_px, guide_hr, confidence_lr=confidence, radius=4, eps=0.01)
    torch.cuda.synchronize()

    # --- Step 5: Temporal Damping ---
    if temporal:
        dx_full, dy_full = _temporal_damp(dx_full, dy_full, cost_int, block_size=block_size)
        torch.cuda.synchronize()

    return dx_full, dy_full
