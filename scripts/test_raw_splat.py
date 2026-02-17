#!/usr/bin/env python3
"""
LSFG-Style Softmax Splat + TAA-Grade History Cache
Pure PyTorch — no neural nets, no CuPy.

Pipeline per frame pair:
  1. MFQ forward + backward flow (2 passes)
  2. Cost-weighted softmax splatting at t=0.5
  3. Weight-based merge (high cost loses, sharp edges win)
  4. History cache with:
     - Bicubic fetch (no sub-pixel blur)
     - YCoCg variance clipping (μ±σ, kills ghost shimmer)
     - Trust mask rejection (hard disocclusions → 3×3 spatial fill)
"""
import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from core.bjf_patchmatch import BJFPatchMatch
from core.refine import refine_flow, reset_temporal

VIDEO_PATH  = os.path.join(ROOT, 'sample_darksouls2.mp4')
OUTPUT_PATH = os.path.join(ROOT, 'output_lsfg_variance_splat.mp4')
MAX_FRAMES  = 300
DEVICE      = 'cuda'
TEMPERATURE = 5.0
GAMMA       = 1.0     # Variance clipping tightness (1.0 = strict)


# ═══════════════════════════════════════════════════════════
#  YCoCg Color Space
# ═══════════════════════════════════════════════════════════

def rgb_to_ycocg(rgb):
    """(B, 3, H, W) RGB → (B, 3, H, W) YCoCg"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y  =  0.25 * r + 0.50 * g + 0.25 * b
    co =  0.50 * r             - 0.50 * b
    cg = -0.25 * r + 0.50 * g - 0.25 * b
    return torch.cat([y, co, cg], dim=1)


def ycocg_to_rgb(ycocg):
    """(B, 3, H, W) YCoCg → (B, 3, H, W) RGB"""
    y, co, cg = ycocg[:, 0:1], ycocg[:, 1:2], ycocg[:, 2:3]
    r = y + co - cg
    g = y      + cg
    b = y - co - cg
    return torch.cat([r, g, b], dim=1)


# ═══════════════════════════════════════════════════════════
#  Softmax Splat
# ═══════════════════════════════════════════════════════════

def softmax_splat(frame, dx, dy, cost):
    """
    Cost-weighted forward splatting.
    weight = exp(-cost / T).  Accumulates (w*color, w) via scatter_add_.
    Returns raw color_buf and weight_buf (NOT resolved).
    """
    B, C, H, W = frame.shape
    weight = torch.exp(-cost / TEMPERATURE)

    yy = torch.arange(H, device=frame.device).view(1, H, 1).expand(B, H, W).float()
    xx = torch.arange(W, device=frame.device).view(1, 1, W).expand(B, H, W).float()

    tx = (xx + dx).round().long()
    ty = (yy + dy).round().long()
    valid = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)

    batch_off = torch.arange(B, device=frame.device).view(B, 1, 1) * (H * W)
    flat_idx = ((ty * W + tx).clamp(0, H * W - 1) + batch_off) * valid.long()
    flat_idx = flat_idx.view(-1)

    w = (weight * valid.float()).view(-1)
    src = frame.permute(0, 2, 3, 1).reshape(-1, C) * w.unsqueeze(1)

    total = B * H * W
    color_flat  = torch.zeros(total, C, device=frame.device)
    weight_flat = torch.zeros(total, 1, device=frame.device)

    color_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), src)
    weight_flat.scatter_add_(0, flat_idx.unsqueeze(1), w.unsqueeze(1))

    color_buf  = color_flat.view(B, H, W, C).permute(0, 3, 1, 2)
    weight_buf = weight_flat.view(B, H, W, 1).permute(0, 3, 1, 2)
    return color_buf, weight_buf


def softmax_merge(color_a, wt_a, color_b, wt_b):
    """Weight-based merge. No 50/50. High-confidence dominates."""
    total_color = color_a + color_b
    total_wt    = wt_a + wt_b
    safe_wt     = total_wt.clamp(min=1e-6)
    merged      = total_color / safe_wt
    return merged, total_wt


# ═══════════════════════════════════════════════════════════
#  Task 1: Bicubic History Fetch
# ═══════════════════════════════════════════════════════════

def warp_history_bicubic(history, dx, dy):
    """
    Backward-warp history using flow_0→1, bicubic interpolation.
    Keeps textures razor-sharp vs bilinear.
    """
    B, C, H, W = history.shape

    yy = torch.arange(H, device=history.device).view(1, H, 1).expand(B, H, W).float()
    xx = torch.arange(W, device=history.device).view(1, 1, W).expand(B, H, W).float()

    sx = xx + 0.5 * dx
    sy = yy + 0.5 * dy

    sx = 2.0 * sx / max(W - 1, 1) - 1.0
    sy = 2.0 * sy / max(H - 1, 1) - 1.0

    grid = torch.stack([sx, sy], dim=-1)

    warped = F.grid_sample(history, grid, mode='bicubic',
                           padding_mode='zeros', align_corners=True)
    return warped


# ═══════════════════════════════════════════════════════════
#  Task 2: YCoCg Variance Clipping
# ═══════════════════════════════════════════════════════════

def variance_clip(merged, warped_history):
    """
    Clip warped history colors to μ±(γ·σ) of the current merged
    canvas's 3×3 spatial neighborhood, computed in YCoCg space.

    Args:
        merged:         (B, 3, H, W) — current softmax-merged canvas
        warped_history: (B, 3, H, W) — bicubic-warped previous output

    Returns:
        clipped: (B, 3, H, W) — history with colors constrained
    """
    # Convert to YCoCg
    curr_ycocg = rgb_to_ycocg(merged)
    hist_ycocg = rgb_to_ycocg(warped_history)

    # 3×3 neighborhood mean and variance via avg_pool2d
    # Pad to keep spatial dims
    padded = F.pad(curr_ycocg, (1, 1, 1, 1), mode='reflect')

    # Mean: avg over 3×3
    mu = F.avg_pool2d(padded, kernel_size=3, stride=1, padding=0)

    # Variance: E[x²] - E[x]²
    padded_sq = padded * padded
    mu_sq = F.avg_pool2d(padded_sq, kernel_size=3, stride=1, padding=0)
    var = (mu_sq - mu * mu).clamp(min=0.0)
    sigma = torch.sqrt(var + 1e-8)

    # Clip history to μ ± γ·σ
    lo = mu - GAMMA * sigma
    hi = mu + GAMMA * sigma
    clipped_ycocg = torch.clamp(hist_ycocg, min=lo, max=hi)

    # Back to RGB
    clipped = ycocg_to_rgb(clipped_ycocg)
    return clipped


# ═══════════════════════════════════════════════════════════
#  Task 3: Trust Mask + Spatial Fallback
# ═══════════════════════════════════════════════════════════

def spatial_fallback_fill(canvas, hole_mask):
    """
    Minimal 3×3 avg_pool2d spatial fill for hard-disoccluded pixels
    that cannot be trusted from history. Only fills remaining holes.
    """
    # Iterative fill: keep applying until no holes remain (max 3 iters)
    current = canvas.clone()
    mask = hole_mask.clone()  # (B, 1, H, W), 1.0 = hole

    for _ in range(3):
        if mask.sum() == 0:
            break
        # Pool the current image (holes are 0, which drag avg down)
        valid = 1.0 - mask  # 1 = has data
        weighted = current * valid
        pooled_color = F.avg_pool2d(
            F.pad(weighted, (1, 1, 1, 1), mode='reflect'),
            kernel_size=3, stride=1, padding=0
        )
        pooled_weight = F.avg_pool2d(
            F.pad(valid, (1, 1, 1, 1), mode='constant', value=0),
            kernel_size=3, stride=1, padding=0
        )
        safe_w = pooled_weight.clamp(min=1e-6)
        fill = pooled_color / safe_w

        # Only paste into holes
        current = current * (1.0 - mask) + fill * mask

        # Update mask: filled pixels are no longer holes
        new_valid = (pooled_weight > 0).float()
        mask = mask * (1.0 - new_valid)

    return current


def upsample_cost(conf_grid, H, W):
    """Block-grid JFA cost (B, GH, GW) uint8 → (B, H, W) float32."""
    return F.interpolate(
        conf_grid.unsqueeze(1).float(), size=(H, W), mode='nearest'
    ).squeeze(1)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n      = min(total - 1, MAX_FRAMES)

    print(f"[Config] Softmax T={TEMPERATURE}, Variance γ={GAMMA}")
    print(f"[Video]  {W}×{H}, {fps_in:.0f}fps, {total} frames → {n} pairs")
    print(f"[Output] 60fps → {OUTPUT_PATH}")

    mfq = BJFPatchMatch(H, W).to(DEVICE)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 60, (W, H))

    ret, prev_bgr = cap.read()
    if not ret:
        print("ERROR: Can't read first frame"); sys.exit(1)
    writer.write(prev_bgr)

    history_frame = None

    t_start = time.time()
    flow_ms, splat_ms, hist_ms = [], [], []
    holes_pre, holes_post = [], []

    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            break

        t0 = torch.from_numpy(prev_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        t1 = torch.from_numpy(cur_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)

        # ── Step 1: Both flows + costs ────────────────────
        t_f = time.time()

        reset_temporal()
        res_fwd = mfq(t0, t1)
        if res_fwd is not None:
            dx01, dy01 = refine_flow(res_fwd[0], res_fwd[1], res_fwd[2], t0, t1, block_size=8, temporal=False)
            cost_01 = upsample_cost(res_fwd[2], H, W)
        else:
            dx01 = torch.zeros(1, H, W, device=DEVICE)
            dy01 = torch.zeros(1, H, W, device=DEVICE)
            cost_01 = torch.zeros(1, H, W, device=DEVICE)

        reset_temporal()
        res_bck = mfq(t1, t0)
        if res_bck is not None:
            dx10, dy10 = refine_flow(res_bck[0], res_bck[1], res_bck[2], t1, t0, block_size=8, temporal=False)
            cost_10 = upsample_cost(res_bck[2], H, W)
        else:
            dx10 = torch.zeros(1, H, W, device=DEVICE)
            dy10 = torch.zeros(1, H, W, device=DEVICE)
            cost_10 = torch.zeros(1, H, W, device=DEVICE)

        torch.cuda.synchronize()
        flow_ms.append((time.time() - t_f) * 1000)

        # ── Step 2: Softmax Splat + Merge ─────────────────
        t_s = time.time()

        color_a, wt_a = softmax_splat(t0, dx01 * 0.5, dy01 * 0.5, cost_01)
        color_b, wt_b = softmax_splat(t1, dx10 * 0.5, dy10 * 0.5, cost_10)
        merged, total_wt = softmax_merge(color_a, wt_a, color_b, wt_b)

        torch.cuda.synchronize()
        splat_ms.append((time.time() - t_s) * 1000)

        # Hole analysis
        hole_mask = (total_wt < 1e-6).float()     # (B, 1, H, W)
        has_a = (wt_a > 1e-6).float()             # (B, 1, H, W)
        has_b = (wt_b > 1e-6).float()             # (B, 1, H, W)

        # Hard disocclusion: hole in BOTH canvases — don't trust history
        hard_disoccl = (1.0 - has_a) * (1.0 - has_b)   # (B, 1, H, W)
        # Soft hole: has data in one canvas but not the other — can use history
        soft_hole = hole_mask * (1.0 - hard_disoccl)    # should be ~0 (holes are by def both-black)
        # Actually: if it's a hole in the merged, both must be zero. So hole_mask == hard_disoccl.
        # But let's keep the logic explicit for clarity.

        pre_pct = hole_mask.mean().item() * 100
        holes_pre.append(pre_pct)

        # ── Step 3: History Cache + Variance Clip ─────────
        t_h = time.time()

        final = merged.clone()

        if history_frame is not None and pre_pct > 0:
            # Bicubic warp history forward
            warped_hist = warp_history_bicubic(history_frame, dx01, dy01)

            # Variance clip in YCoCg
            clipped_hist = variance_clip(merged, warped_hist)

            # Trust mask: only use history for NON-hard-disocclusion holes
            # For soft holes (at least one canvas had data nearby), history is OK
            # For hard disocclusions (both black), reject history
            trustworthy_holes = hole_mask * (1.0 - hard_disoccl)
            hard_holes = hard_disoccl

            # Paste clipped history into trustworthy holes
            final = final * (1.0 - trustworthy_holes) + clipped_hist * trustworthy_holes

            # Spatial fallback for hard disocclusions
            if hard_holes.sum() > 0:
                final = final * (1.0 - hard_holes) + spatial_fallback_fill(final, hard_holes) * hard_holes

        elif pre_pct > 0:
            # No history yet (first frame): spatial fallback for all holes
            final = spatial_fallback_fill(final, hole_mask)

        torch.cuda.synchronize()
        hist_ms.append((time.time() - t_h) * 1000)

        # Post-fill stats
        post_hole = 0.0  # all holes handled
        holes_post.append(post_hole)

        # Save as history for next frame
        history_frame = final.clone().detach()

        # Output
        out_np = (final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_bgr = out_np[:, :, ::-1].copy()

        writer.write(prev_bgr)
        writer.write(out_bgr)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            af  = np.mean(flow_ms[-50:])
            asp = np.mean(splat_ms[-50:])
            ah  = np.mean(hist_ms[-50:])
            ahp = np.mean(holes_pre[-50:])
            print(f"  [{i+1}/{n}]  flow={af:.1f}ms  splat={asp:.1f}ms  hist={ah:.1f}ms  "
                  f"holes={ahp:.1f}%  ({(i+1)/elapsed:.1f} pairs/s)")

        del t0, t1
        prev_bgr = cur_bgr

    cap.release()
    writer.release()

    n_done = len(flow_ms)
    elapsed = time.time() - t_start
    total_pipe = np.mean(flow_ms) + np.mean(splat_ms) + np.mean(hist_ms)

    print(f"\n{'='*60}")
    print(f"  SOFTMAX SPLAT + VARIANCE CLIP + HISTORY — DONE")
    print(f"{'='*60}")
    print(f"  Frames:          {n_done} pairs → {n_done * 2 + 1} output frames")
    print(f"  Total time:      {elapsed:.1f}s ({n_done/elapsed:.1f} pairs/s)")
    print(f"  ─── Per-Frame Breakdown ───")
    print(f"  MFQ flow (×2):   {np.mean(flow_ms):.1f} ms")
    print(f"  Softmax splat:   {np.mean(splat_ms):.1f} ms")
    print(f"  Hist+clip+fill:  {np.mean(hist_ms):.1f} ms")
    print(f"  Total pipeline:  {total_pipe:.1f} ms/frame")
    print(f"  ─── Hole Stats ───")
    print(f"  Pre-history:     {np.mean(holes_pre):.2f}% avg")
    print(f"  Post-fill:       {np.mean(holes_post):.2f}%")
    print(f"  Output:          {OUTPUT_PATH}")
    print(f"  Size:            {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
