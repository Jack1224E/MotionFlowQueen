#!/usr/bin/env python3
"""
DIS vs MFQ Flow Forensic Comparison
Runs BOTH flow engines on the same frames, then compares:
  1. Flow smoothness (spatial gradient of flow field)
  2. Forward-splat hole maps
  3. Flow magnitude statistics
  4. Sub-pixel coverage (what % of flow values are integers?)
  5. Block-edge discontinuity analysis

Outputs a side-by-side comparison video + terminal statistics.
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
OUTPUT_PATH = os.path.join(ROOT, 'output_flow_comparison.mp4')
MAX_FRAMES  = 50   # Enough for diagnosis
DEVICE      = 'cuda'


def flow_to_color(dx, dy, max_flow=30.0):
    """HSV flow visualization: hue=direction, saturation=1, value=magnitude."""
    mag = torch.sqrt(dx**2 + dy**2).cpu().numpy()
    ang = torch.atan2(dy, dx).cpu().numpy()

    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def forward_splat_holes(dx, dy, H, W):
    """Count holes from forward splatting (where no pixel lands)."""
    yy = torch.arange(H, device=dx.device).view(H, 1).expand(H, W).float()
    xx = torch.arange(W, device=dx.device).view(1, W).expand(H, W).float()

    tx = (xx + dx * 0.5).round().long()
    ty = (yy + dy * 0.5).round().long()
    valid = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)

    hit_map = torch.zeros(H, W, device=dx.device)
    flat_idx = (ty * W + tx).clamp(0, H * W - 1) * valid.long()
    hit_flat = hit_map.view(-1)
    hit_flat.scatter_add_(0, flat_idx.view(-1), valid.float().view(-1))
    hit_map = hit_flat.view(H, W)

    return (hit_map == 0).float()  # 1.0 = hole


def flow_smoothness(dx, dy):
    """Spatial gradient magnitude of flow field (lower = smoother)."""
    # Sobel-like gradient via finite differences
    grad_dx_x = torch.abs(dx[:, 1:] - dx[:, :-1])  # dF/dx
    grad_dx_y = torch.abs(dx[1:, :] - dx[:-1, :])  # dF/dy
    grad_dy_x = torch.abs(dy[:, 1:] - dy[:, :-1])
    grad_dy_y = torch.abs(dy[1:, :] - dy[:-1, :])

    smooth_x = (grad_dx_x.mean() + grad_dy_x.mean()) / 2
    smooth_y = (grad_dx_y.mean() + grad_dy_y.mean()) / 2
    return (smooth_x + smooth_y).item()


def block_discontinuity(dx, dy, block_size=8):
    """Measure flow discontinuity at block edges (MFQ artifact signature)."""
    H, W = dx.shape

    # Horizontal block edges (every `block_size` columns)
    h_edges = []
    for x in range(block_size, W - 1, block_size):
        jump_dx = torch.abs(dx[:, x] - dx[:, x-1]).mean().item()
        jump_dy = torch.abs(dy[:, x] - dy[:, x-1]).mean().item()
        h_edges.append(jump_dx + jump_dy)

    # Vertical block edges
    v_edges = []
    for y in range(block_size, H - 1, block_size):
        jump_dx = torch.abs(dx[y, :] - dx[y-1, :]).mean().item()
        jump_dy = torch.abs(dy[y, :] - dy[y-1, :]).mean().item()
        v_edges.append(jump_dx + jump_dy)

    # Non-edge average
    non_edges_h = []
    for x in range(1, W-1):
        if x % block_size != 0:
            jump = torch.abs(dx[:, x] - dx[:, x-1]).mean().item()
            non_edges_h.append(jump)
            if len(non_edges_h) > 200:
                break

    avg_edge = np.mean(h_edges + v_edges) if h_edges or v_edges else 0
    avg_non_edge = np.mean(non_edges_h) if non_edges_h else 0

    return avg_edge, avg_non_edge


def subpixel_fraction(dx, dy):
    """What % of flow values are NOT on integer grid? (Higher = better sub-pixel)."""
    frac_dx = (dx - dx.round()).abs()
    frac_dy = (dy - dy.round()).abs()
    has_subpixel = ((frac_dx > 0.01) | (frac_dy > 0.01)).float()
    return has_subpixel.mean().item() * 100


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total - 1, MAX_FRAMES)

    print(f"[Video] {W}×{H}, {n} pairs")
    print(f"[Output] {OUTPUT_PATH}\n")

    mfq = BJFPatchMatch(H, W).to(DEVICE)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # Side-by-side output: [MFQ Flow | DIS Flow | MFQ Holes | DIS Holes]
    panel_w = W * 4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 15, (panel_w, H))

    ret, prev_bgr = cap.read()

    # Accumulators
    stats = {k: [] for k in [
        'mfq_mag', 'dis_mag', 'mfq_smooth', 'dis_smooth',
        'mfq_holes', 'dis_holes', 'mfq_subpx', 'dis_subpx',
        'mfq_edge', 'mfq_nonedge', 'dis_edge', 'dis_nonedge'
    ]}

    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            break

        # Tensors
        t0 = torch.from_numpy(prev_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        t1 = torch.from_numpy(cur_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

        # ── MFQ Flow ──────────────────────────────────────
        reset_temporal()
        res = mfq(t0, t1)
        if res is not None:
            dx_m, dy_m = refine_flow(res[0], res[1], res[2], t0, t1, block_size=8, temporal=False)
        else:
            dx_m = torch.zeros(1, H, W, device=DEVICE)
            dy_m = torch.zeros(1, H, W, device=DEVICE)
        dx_m = dx_m[0]  # (H, W)
        dy_m = dy_m[0]

        # ── DIS Flow ──────────────────────────────────────
        flow_dis = dis.calc(g0, g1, None)  # (H, W, 2)
        dx_d = torch.from_numpy(flow_dis[:, :, 0]).to(DEVICE)
        dy_d = torch.from_numpy(flow_dis[:, :, 1]).to(DEVICE)

        # ── Statistics ────────────────────────────────────
        mag_m = torch.sqrt(dx_m**2 + dy_m**2)
        mag_d = torch.sqrt(dx_d**2 + dy_d**2)
        stats['mfq_mag'].append(mag_m.mean().item())
        stats['dis_mag'].append(mag_d.mean().item())

        stats['mfq_smooth'].append(flow_smoothness(dx_m, dy_m))
        stats['dis_smooth'].append(flow_smoothness(dx_d, dy_d))

        holes_m = forward_splat_holes(dx_m, dy_m, H, W)
        holes_d = forward_splat_holes(dx_d, dy_d, H, W)
        stats['mfq_holes'].append(holes_m.mean().item() * 100)
        stats['dis_holes'].append(holes_d.mean().item() * 100)

        stats['mfq_subpx'].append(subpixel_fraction(dx_m, dy_m))
        stats['dis_subpx'].append(subpixel_fraction(dx_d, dy_d))

        me, mne = block_discontinuity(dx_m, dy_m, 8)
        de, dne = block_discontinuity(dx_d, dy_d, 8)
        stats['mfq_edge'].append(me)
        stats['mfq_nonedge'].append(mne)
        stats['dis_edge'].append(de)
        stats['dis_nonedge'].append(dne)

        # ── Visualization ─────────────────────────────────
        mfq_vis = flow_to_color(dx_m, dy_m)
        dis_vis = flow_to_color(dx_d, dy_d)
        holes_m_vis = (holes_m.cpu().numpy() * 255).astype(np.uint8)
        holes_m_vis = cv2.cvtColor(holes_m_vis, cv2.COLOR_GRAY2BGR)
        # Color holes red
        holes_m_vis[:, :, 2] = holes_m_vis[:, :, 0]
        holes_m_vis[:, :, 0] = 0
        holes_m_vis[:, :, 1] = 0

        holes_d_vis = (holes_d.cpu().numpy() * 255).astype(np.uint8)
        holes_d_vis = cv2.cvtColor(holes_d_vis, cv2.COLOR_GRAY2BGR)
        holes_d_vis[:, :, 1] = holes_d_vis[:, :, 0]  # green
        holes_d_vis[:, :, 0] = 0
        holes_d_vis[:, :, 2] = 0

        # Labels
        cv2.putText(mfq_vis, 'MFQ Flow', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(dis_vis, 'DIS Flow', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(holes_m_vis, f'MFQ Holes: {stats["mfq_holes"][-1]:.1f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(holes_d_vis, f'DIS Holes: {stats["dis_holes"][-1]:.1f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        panel = np.hstack([mfq_vis, dis_vis, holes_m_vis, holes_d_vis])
        writer.write(panel)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] processed")

        prev_bgr = cur_bgr

    cap.release()
    writer.release()

    # ═══════════════════════════════════════════════════════
    #  Final Report
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DIS vs MFQ FLOW FORENSIC REPORT ({n} frames)")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Metric':<30} {'MFQ':>12} {'DIS':>12} {'Gap':>12}")
    print(f"  {'-'*66}")

    def row(label, mfq_val, dis_val, fmt='.2f', lower_better=True):
        gap = mfq_val - dis_val
        arrow = '↑ worse' if (gap > 0 and lower_better) or (gap < 0 and not lower_better) else '✓ ok'
        print(f"  {label:<30} {mfq_val:>12{fmt}} {dis_val:>12{fmt}} {gap:>+8{fmt}} {arrow}")

    row('Avg flow magnitude (px)',    np.mean(stats['mfq_mag']),    np.mean(stats['dis_mag']),  '.1f', False)
    row('Flow smoothness (grad)',     np.mean(stats['mfq_smooth']), np.mean(stats['dis_smooth']), '.3f')
    row('Splat holes (%)',            np.mean(stats['mfq_holes']),  np.mean(stats['dis_holes']),  '.2f')
    row('Sub-pixel values (%)',       np.mean(stats['mfq_subpx']),  np.mean(stats['dis_subpx']),  '.1f', False)
    row('Block-edge discontinuity',   np.mean(stats['mfq_edge']),   np.mean(stats['dis_edge']),   '.3f')
    row('Non-edge discontinuity',     np.mean(stats['mfq_nonedge']), np.mean(stats['dis_nonedge']), '.3f')

    edge_ratio_mfq = np.mean(stats['mfq_edge']) / max(np.mean(stats['mfq_nonedge']), 1e-6)
    edge_ratio_dis = np.mean(stats['dis_edge']) / max(np.mean(stats['dis_nonedge']), 1e-6)

    print(f"")
    print(f"  {'Block edge / non-edge ratio':<30} {edge_ratio_mfq:>12.2f}x {edge_ratio_dis:>12.2f}x")
    print(f"")
    print(f"  ─── DIAGNOSIS ───")
    if edge_ratio_mfq > 1.5:
        print(f"  ⚠  MFQ block edges are {edge_ratio_mfq:.1f}x more discontinuous than interior.")
        print(f"     This is the #1 cause of splatting holes.")
    if np.mean(stats['mfq_subpx']) < 50:
        print(f"  ⚠  MFQ has only {np.mean(stats['mfq_subpx']):.0f}% sub-pixel flow values.")
        print(f"     DIS has {np.mean(stats['dis_subpx']):.0f}%. Quantized flow → collision holes.")
    if np.mean(stats['mfq_smooth']) > 2 * np.mean(stats['dis_smooth']):
        print(f"  ⚠  MFQ flow is {np.mean(stats['mfq_smooth'])/np.mean(stats['dis_smooth']):.1f}x rougher than DIS.")
        print(f"     No spatial smoothness constraint → noisy flow field.")
    print(f"")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
