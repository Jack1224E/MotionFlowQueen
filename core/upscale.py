"""
Edge-Aware Guided Filter Upscale (PyTorch)
Upscales block-grid flow to full resolution using high-res luma as a guide.
Preserves sharp edges (character outlines) while smoothing flat regions.
"""
import torch
import torch.nn.functional as F


def _boxfilter2d(x, radius):
    """Fast box filter using avg_pool2d with replicate padding."""
    k = 2 * radius + 1
    # x: (B, 1, H, W)
    return F.avg_pool2d(
        F.pad(x, [radius] * 4, mode='replicate'),
        kernel_size=k, stride=1, padding=0
    )


def guided_upsample(flow_lr, guide_hr, confidence_lr=None, radius=4, eps=0.01):
    """
    Fast Guided Filter for edge-aware flow upsampling.

    Args:
        flow_lr: (B, GH, GW) float32 - low-res flow (one component, pixel units)
        guide_hr: (B, H, W) float32 - high-res luma guide image [0, 1]
        confidence_lr: (B, GH, GW) float32 or None - confidence weights (0-1)
        radius: int - box filter radius at low resolution
        eps: float - regularization (controls smoothness)

    Returns:
        flow_hr: (B, H, W) float32 - edge-aware upsampled flow
    """
    B, GH, GW = flow_lr.shape
    H, W = guide_hr.shape[1], guide_hr.shape[2]

    # Downsample guide to match flow resolution
    guide_lr = F.interpolate(
        guide_hr.unsqueeze(1), (GH, GW), mode='bilinear', align_corners=False
    ).squeeze(1)  # (B, GH, GW)

    # Apply confidence weighting: low-confidence regions get pulled toward 0
    if confidence_lr is not None:
        p = flow_lr * confidence_lr
    else:
        p = flow_lr

    I = guide_lr  # (B, GH, GW)

    # Add channel dim for pooling
    I4d = I.unsqueeze(1)     # (B, 1, GH, GW)
    p4d = p.unsqueeze(1)     # (B, 1, GH, GW)

    # Box-filtered statistics at low resolution
    mean_I = _boxfilter2d(I4d, radius).squeeze(1)
    mean_p = _boxfilter2d(p4d, radius).squeeze(1)
    mean_Ip = _boxfilter2d((I * p).unsqueeze(1), radius).squeeze(1)
    mean_II = _boxfilter2d((I * I).unsqueeze(1), radius).squeeze(1)

    if confidence_lr is not None:
        w4d = confidence_lr.unsqueeze(1)
        sum_w = _boxfilter2d(w4d, radius).squeeze(1).clamp(min=1e-6)
        mean_p = _boxfilter2d((flow_lr * confidence_lr).unsqueeze(1), radius).squeeze(1) / sum_w
        mean_Ip = _boxfilter2d((I * flow_lr * confidence_lr).unsqueeze(1), radius).squeeze(1) / sum_w

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    # Guided filter coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Average the coefficients (second box filter pass)
    mean_a = _boxfilter2d(a.unsqueeze(1), radius).squeeze(1)
    mean_b = _boxfilter2d(b.unsqueeze(1), radius).squeeze(1)

    # Upscale coefficients to full resolution
    a_hr = F.interpolate(
        mean_a.unsqueeze(1), (H, W), mode='bilinear', align_corners=False
    ).squeeze(1)
    b_hr = F.interpolate(
        mean_b.unsqueeze(1), (H, W), mode='bilinear', align_corners=False
    ).squeeze(1)

    # Apply: output = a * guide_fullres + b
    flow_hr = a_hr * guide_hr + b_hr
    return flow_hr
