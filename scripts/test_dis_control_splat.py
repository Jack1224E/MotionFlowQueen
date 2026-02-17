#!/usr/bin/env python3
"""
DIS Control Test — Splatter Diagnostic
OpenCV DIS (Medium) flow → our exact softmax + variance-clip + history compositor.
Tests: is the compositor (splatting pipeline) good when given high-quality flow?
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

VIDEO_PATH  = os.path.join(ROOT, 'sample_darksouls2.mp4')
OUTPUT_PATH = os.path.join(ROOT, 'output_dis_control_splat.mp4')
MAX_FRAMES  = 300
DEVICE      = 'cuda'
TEMPERATURE = 5.0
GAMMA       = 1.0


# ═══════════════════════════════════════════════════════════
#  DIS Flow (OpenCV, CPU)
# ═══════════════════════════════════════════════════════════

def compute_dis_flow(img0_bgr, img1_bgr, dis):
    """
    Compute bidirectional DIS flow + photometric pseudo-cost.

    Returns:
        dx_01, dy_01: (1, H, W) float32 GPU tensors — forward flow
        dx_10, dy_10: (1, H, W) float32 GPU tensors — backward flow
        cost_01:      (1, H, W) float32 GPU — photometric L1 cost for forward
        cost_10:      (1, H, W) float32 GPU — photometric L1 cost for backward
    """
    g0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

    # Forward: 0 → 1
    flow_01_np = dis.calc(g0, g1, None)  # (H, W, 2) float32
    # Backward: 1 → 0
    flow_10_np = dis.calc(g1, g0, None)

    # Convert to tensors
    dx_01 = torch.from_numpy(flow_01_np[:, :, 0]).unsqueeze(0).to(DEVICE)
    dy_01 = torch.from_numpy(flow_01_np[:, :, 1]).unsqueeze(0).to(DEVICE)
    dx_10 = torch.from_numpy(flow_10_np[:, :, 0]).unsqueeze(0).to(DEVICE)
    dy_10 = torch.from_numpy(flow_10_np[:, :, 1]).unsqueeze(0).to(DEVICE)

    # Photometric pseudo-cost: |I0 - warp(I1, flow_01)|
    H, W = g0.shape
    img0_t = torch.from_numpy(img0_bgr).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
    img1_t = torch.from_numpy(img1_bgr).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)

    # Build grids for backward warp
    yy = torch.arange(H, device=DEVICE).view(1, H, 1).expand(1, H, W).float()
    xx = torch.arange(W, device=DEVICE).view(1, 1, W).expand(1, H, W).float()

    # Warp I1 → I0 using flow_01
    sx_01 = 2.0 * (xx + dx_01) / max(W - 1, 1) - 1.0
    sy_01 = 2.0 * (yy + dy_01) / max(H - 1, 1) - 1.0
    grid_01 = torch.stack([sx_01, sy_01], dim=-1)
    warped_1to0 = F.grid_sample(img1_t, grid_01, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)
    cost_01 = torch.abs(img0_t - warped_1to0).mean(dim=1)  # (1, H, W)

    # Warp I0 → I1 using flow_10
    sx_10 = 2.0 * (xx + dx_10) / max(W - 1, 1) - 1.0
    sy_10 = 2.0 * (yy + dy_10) / max(H - 1, 1) - 1.0
    grid_10 = torch.stack([sx_10, sy_10], dim=-1)
    warped_0to1 = F.grid_sample(img0_t, grid_10, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)
    cost_10 = torch.abs(img1_t - warped_0to1).mean(dim=1)  # (1, H, W)

    # Scale cost to roughly match JFA cost range (0-63)
    # Photometric L1 is [0, 1], JFA cost is [0, 63]
    cost_01 = cost_01 * 63.0
    cost_10 = cost_10 * 63.0

    return dx_01, dy_01, dx_10, dy_10, cost_01, cost_10


# ═══════════════════════════════════════════════════════════
#  Softmax Splat (identical to test_raw_splat.py)
# ═══════════════════════════════════════════════════════════

def softmax_splat(frame, dx, dy, cost):
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

    return (color_flat.view(B, H, W, C).permute(0, 3, 1, 2),
            weight_flat.view(B, H, W, 1).permute(0, 3, 1, 2))


def softmax_merge(color_a, wt_a, color_b, wt_b):
    total_color = color_a + color_b
    total_wt    = wt_a + wt_b
    return total_color / total_wt.clamp(min=1e-6), total_wt


# ═══════════════════════════════════════════════════════════
#  YCoCg Variance Clipping (identical)
# ═══════════════════════════════════════════════════════════

def rgb_to_ycocg(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return torch.cat([0.25*r + 0.50*g + 0.25*b,
                      0.50*r - 0.50*b,
                     -0.25*r + 0.50*g - 0.25*b], dim=1)

def ycocg_to_rgb(ycocg):
    y, co, cg = ycocg[:, 0:1], ycocg[:, 1:2], ycocg[:, 2:3]
    return torch.cat([y + co - cg, y + cg, y - co - cg], dim=1)

def variance_clip(merged, warped_history):
    curr_ycocg = rgb_to_ycocg(merged)
    hist_ycocg = rgb_to_ycocg(warped_history)
    padded = F.pad(curr_ycocg, (1, 1, 1, 1), mode='reflect')
    mu = F.avg_pool2d(padded, kernel_size=3, stride=1, padding=0)
    mu_sq = F.avg_pool2d(padded * padded, kernel_size=3, stride=1, padding=0)
    sigma = torch.sqrt((mu_sq - mu * mu).clamp(min=0.0) + 1e-8)
    clipped = torch.clamp(hist_ycocg, min=mu - GAMMA * sigma, max=mu + GAMMA * sigma)
    return ycocg_to_rgb(clipped)


def warp_history_bicubic(history, dx, dy):
    B, C, H, W = history.shape
    yy = torch.arange(H, device=history.device).view(1, H, 1).expand(B, H, W).float()
    xx = torch.arange(W, device=history.device).view(1, 1, W).expand(B, H, W).float()
    sx = 2.0 * (xx + 0.5 * dx) / max(W - 1, 1) - 1.0
    sy = 2.0 * (yy + 0.5 * dy) / max(H - 1, 1) - 1.0
    grid = torch.stack([sx, sy], dim=-1)
    return F.grid_sample(history, grid, mode='bicubic',
                         padding_mode='zeros', align_corners=True)


def spatial_fallback_fill(canvas, hole_mask):
    current = canvas.clone()
    mask = hole_mask.clone()
    for _ in range(3):
        if mask.sum() == 0:
            break
        valid = 1.0 - mask
        pooled_c = F.avg_pool2d(F.pad(current * valid, (1,1,1,1), mode='reflect'),
                                kernel_size=3, stride=1, padding=0)
        pooled_w = F.avg_pool2d(F.pad(valid, (1,1,1,1), mode='constant', value=0),
                                kernel_size=3, stride=1, padding=0)
        fill = pooled_c / pooled_w.clamp(min=1e-6)
        current = current * (1.0 - mask) + fill * mask
        mask = mask * (1.0 - (pooled_w > 0).float())
    return current


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

    print(f"[DIS Control] Medium preset, photometric L1 pseudo-cost")
    print(f"[Video] {W}×{H}, {fps_in:.0f}fps, {total} frames → {n} pairs")
    print(f"[Output] 60fps → {OUTPUT_PATH}")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 60, (W, H))

    ret, prev_bgr = cap.read()
    if not ret:
        print("ERROR"); sys.exit(1)
    writer.write(prev_bgr)

    history_frame = None

    t_start = time.time()
    flow_ms, splat_ms, hist_ms = [], [], []
    holes_pre = []

    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            break

        t0 = torch.from_numpy(prev_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        t1 = torch.from_numpy(cur_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)

        # ── DIS Flow ──────────────────────────────────────
        t_f = time.time()
        dx01, dy01, dx10, dy10, cost_01, cost_10 = compute_dis_flow(prev_bgr, cur_bgr, dis)
        flow_ms.append((time.time() - t_f) * 1000)

        # ── Softmax Splat + Merge ─────────────────────────
        t_s = time.time()
        color_a, wt_a = softmax_splat(t0, dx01 * 0.5, dy01 * 0.5, cost_01)
        color_b, wt_b = softmax_splat(t1, dx10 * 0.5, dy10 * 0.5, cost_10)
        merged, total_wt = softmax_merge(color_a, wt_a, color_b, wt_b)
        torch.cuda.synchronize()
        splat_ms.append((time.time() - t_s) * 1000)

        hole_mask = (total_wt < 1e-6).float()
        hard_disoccl = hole_mask  # all holes are hard (both canvases black)
        pre_pct = hole_mask.mean().item() * 100
        holes_pre.append(pre_pct)

        # ── History Cache + Variance Clip ─────────────────
        t_h = time.time()
        final = merged.clone()

        if history_frame is not None and pre_pct > 0:
            warped_hist = warp_history_bicubic(history_frame, dx01, dy01)
            clipped_hist = variance_clip(merged, warped_hist)
            # Trust mask: reject history for hard disocclusions
            trustworthy = hole_mask * (1.0 - hard_disoccl)
            hard_only = hard_disoccl
            final = final * (1.0 - trustworthy) + clipped_hist * trustworthy
            if hard_only.sum() > 0:
                final = final * (1.0 - hard_only) + spatial_fallback_fill(final, hard_only) * hard_only
        elif pre_pct > 0:
            final = spatial_fallback_fill(final, hole_mask)

        torch.cuda.synchronize()
        hist_ms.append((time.time() - t_h) * 1000)

        history_frame = final.clone().detach()

        out_np = (final[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_bgr = out_np[:, :, ::-1].copy()

        writer.write(prev_bgr)
        writer.write(out_bgr)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{n}] flow={np.mean(flow_ms[-50:]):.1f}ms  "
                  f"splat={np.mean(splat_ms[-50:]):.1f}ms  hist={np.mean(hist_ms[-50:]):.1f}ms  "
                  f"holes={np.mean(holes_pre[-50:]):.1f}%  ({(i+1)/elapsed:.1f} pairs/s)")

        del t0, t1
        prev_bgr = cur_bgr

    cap.release()
    writer.release()

    n_done = len(flow_ms)
    elapsed = time.time() - t_start
    total_pipe = np.mean(flow_ms) + np.mean(splat_ms) + np.mean(hist_ms)

    print(f"\n{'='*60}")
    print(f"  DIS CONTROL SPLAT — DONE")
    print(f"{'='*60}")
    print(f"  Frames:       {n_done} pairs → {n_done * 2 + 1} output frames")
    print(f"  Total time:   {elapsed:.1f}s ({n_done/elapsed:.1f} pairs/s)")
    print(f"  DIS flow:     {np.mean(flow_ms):.1f} ms")
    print(f"  Splat+merge:  {np.mean(splat_ms):.1f} ms")
    print(f"  Hist+clip:    {np.mean(hist_ms):.1f} ms")
    print(f"  Pipeline:     {total_pipe:.1f} ms/frame")
    print(f"  Avg holes:    {np.mean(holes_pre):.2f}%")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Size:         {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
