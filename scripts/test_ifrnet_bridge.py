#!/usr/bin/env python3
"""
MFQ × IFRNet True Bridge v3
Fixes: zero-motion bias, true bidirectional flow, correct IFRNet_S.pth, dual-scale injection.

Changes from v2:
  - Runs MFQ TWICE per pair: I0→I1 (forward) and I1→I0 (backward)
  - Uses real flow_1_to_0 instead of approximating as -flow_0_to_1
  - Packs [0.5 * flow_10, 0.5 * flow_01] matching FG2's FastFlowEngine output
  - Injects at both 1/8 and 1/4 decoder layers
  - Uses IFRNet_S.pth (not Vimeo90K)
"""
import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# ─── Path Setup ─────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'FG2', 'repo_ifrnet'))

# Stub imageio (repo_ifrnet/utils.py imports it, inference never calls it)
import types
if 'imageio' not in sys.modules:
    _stub = types.ModuleType('imageio')
    _stub.imread = None   # type: ignore[attr-defined]
    _stub.imwrite = None  # type: ignore[attr-defined]
    _stub.mimsave = None  # type: ignore[attr-defined]
    sys.modules['imageio'] = _stub

from core.bjf_patchmatch import BJFPatchMatch
from core.refine import refine_flow, reset_temporal

# ─── Config ─────────────────────────────────────────────────
VIDEO_PATH  = os.path.join(ROOT, 'sample_darksouls2.mp4')
OUTPUT_PATH = os.path.join(ROOT, 'output_true_bridge_60fps.mp4')
MAX_FRAMES  = 300
DEVICE      = 'cuda'


def load_ifrnet():
    """Load IFRNet_S with FG2's exact checkpoint."""
    from models.IFRNet_S import Model
    ckpt_primary  = os.path.join(ROOT, 'FG2', 'repo_ifrnet', 'checkpoints', 'IFRNet_S.pth')
    ckpt_fallback = os.path.join(ROOT, 'FG2', 'repo_ifrnet', 'checkpoints', 'IFRNet_S_Vimeo90K.pth')
    ckpt = ckpt_primary if os.path.exists(ckpt_primary) else ckpt_fallback

    model = Model().to(DEVICE).eval()
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"[IFRNet] Loaded IFRNet_S from {os.path.basename(ckpt)}")
    return model


def pad_to_32(img):
    """Pad (B,C,H,W) so H,W are multiples of 32."""
    _, _, h, w = img.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    if ph != h or pw != w:
        img = F.pad(img, (0, pw - w, 0, ph - h))
    return img, h, w


def build_flow_dict(dx_fwd, dy_fwd, dx_bck, dy_bck):
    """
    Pack TRUE bidirectional flow into IFRNet's injection format.

    Matches FG2/fast_flow_engine.py (line 163):
        flow_out = cat([0.5 * flow_bck, 0.5 * flow_fwd], dim=1)

    Then FG2/benchmark_hybrid.py (lines 102-106):
        for each scale sc in {1/8, 1/4}:
            scaled_flow = interpolate(raw_flow, scale_factor=sc) * sc

    Args:
        dx_fwd, dy_fwd: (B, H, W) true flow I0→I1 in pixel units
        dx_bck, dy_bck: (B, H, W) true flow I1→I0 in pixel units

    Returns:
        dict with '1/8' and '1/4' keys → (B, 4, H_sc, W_sc) tensors
    """
    # Stack each direction to (B, 2, H, W)
    flow_01 = torch.stack([dx_fwd, dy_fwd], dim=1)   # forward:  I0 → I1
    flow_10 = torch.stack([dx_bck, dy_bck], dim=1)   # backward: I1 → I0

    # Pack matching FG2 FastFlowEngine:
    #   channels [0:2] = 0.5 * flow_bck (flow_t→0 ≈ 0.5 * flow_10)
    #   channels [2:4] = 0.5 * flow_fwd (flow_t→1 ≈ 0.5 * flow_01)
    raw_flow = torch.cat([0.5 * flow_10, 0.5 * flow_01], dim=1)  # (B, 4, H, W)

    # Downscale to both injection points
    flow_dict = {}
    for key, sc in [('1/8', 1/8), ('1/4', 1/4)]:
        scaled = F.interpolate(raw_flow, scale_factor=sc, mode='bilinear', align_corners=False)
        scaled = scaled * sc   # adjust magnitude for the smaller resolution
        flow_dict[key] = scaled

    return flow_dict


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in  = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n       = min(total - 1, MAX_FRAMES)
    fps_out = fps_in * 2

    print(f"[Video] {W}×{H}, {fps_in:.0f}fps, {total} frames → processing {n} pairs")
    print(f"[Output] {fps_out:.0f}fps → {OUTPUT_PATH}")

    # ─── Initialize ─────────────────────────────────────────
    mfq = BJFPatchMatch(H, W).to(DEVICE)
    ifrnet = load_ifrnet()
    embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(DEVICE)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_out, (W, H))

    ret, prev_bgr = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame"); sys.exit(1)
    writer.write(prev_bgr)

    t_start = time.time()
    flow_times, synth_times = [], []

    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            print(f"[WARN] EOF at frame {i}"); break

        # BGR→RGB, HWC→CHW, [0,1], batch
        t0 = torch.from_numpy(prev_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        t1 = torch.from_numpy(cur_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)

        # ══════════════════════════════════════════════════════
        #  Task 2: TRUE Bidirectional Flow (run MFQ twice)
        # ══════════════════════════════════════════════════════
        t_flow = time.time()

        # Forward: I0 → I1
        reset_temporal()
        result_fwd = mfq(t0, t1)
        if result_fwd is not None:
            dx_g_fwd, dy_g_fwd, conf_fwd = result_fwd
            dx_fwd, dy_fwd = refine_flow(dx_g_fwd, dy_g_fwd, conf_fwd, t0, t1, block_size=8, temporal=False)
        else:
            dx_fwd = torch.zeros(1, H, W, device=DEVICE)
            dy_fwd = torch.zeros(1, H, W, device=DEVICE)

        # Backward: I1 → I0  (true backward flow, not -forward)
        result_bck = mfq(t1, t0)
        if result_bck is not None:
            dx_g_bck, dy_g_bck, conf_bck = result_bck
            dx_bck, dy_bck = refine_flow(dx_g_bck, dy_g_bck, conf_bck, t1, t0, block_size=8, temporal=False)
        else:
            dx_bck = torch.zeros(1, H, W, device=DEVICE)
            dy_bck = torch.zeros(1, H, W, device=DEVICE)

        torch.cuda.synchronize()
        flow_times.append(time.time() - t_flow)

        # ══════════════════════════════════════════════════════
        #  Task 3: IFRNet Synthesis with dual-scale injection
        # ══════════════════════════════════════════════════════
        t_synth = time.time()

        img0_pad, oh, ow = pad_to_32(t0)
        img1_pad, _, _    = pad_to_32(t1)
        _, _, ph, pw = img0_pad.shape

        # Pad flow to match padded dims
        if ph != H or pw != W:
            dx_fwd_p = F.pad(dx_fwd.unsqueeze(1), (0, pw - W, 0, ph - H)).squeeze(1)
            dy_fwd_p = F.pad(dy_fwd.unsqueeze(1), (0, pw - W, 0, ph - H)).squeeze(1)
            dx_bck_p = F.pad(dx_bck.unsqueeze(1), (0, pw - W, 0, ph - H)).squeeze(1)
            dy_bck_p = F.pad(dy_bck.unsqueeze(1), (0, pw - W, 0, ph - H)).squeeze(1)
        else:
            dx_fwd_p, dy_fwd_p = dx_fwd, dy_fwd
            dx_bck_p, dy_bck_p = dx_bck, dy_bck

        # Build injection dict: true bidir, dual scale
        ext_flow = build_flow_dict(dx_fwd_p, dy_fwd_p, dx_bck_p, dy_bck_p)

        with torch.no_grad():
            imgt = ifrnet.inference(img0_pad, img1_pad, embt, external_flow=ext_flow)

        imgt_np = (imgt[0, :, :oh, :ow].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        imgt_bgr = imgt_np[:, :, ::-1].copy()
        torch.cuda.synchronize()
        synth_times.append(time.time() - t_synth)

        # Write: interpolated, then original
        writer.write(imgt_bgr)
        writer.write(cur_bgr)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            avg_flow = np.mean(flow_times[-50:]) * 1000
            avg_synth = np.mean(synth_times[-50:]) * 1000
            total_fps = (i + 1) / elapsed
            print(f"  [{i+1}/{n}] flow={avg_flow:.1f}ms  synth={avg_synth:.1f}ms  "
                  f"total={total_fps:.1f} pairs/s  elapsed={elapsed:.0f}s")

        del t0, t1
        prev_bgr = cur_bgr

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    n_done = len(flow_times)
    print(f"\n{'='*60}")
    print(f"  MFQ × IFRNet TRUE Bridge v3 — DONE")
    print(f"{'='*60}")
    print(f"  Frames processed:  {n_done} pairs → {n_done * 2 + 1} output frames")
    print(f"  Total time:        {elapsed:.1f}s ({n_done/elapsed:.1f} pairs/s)")
    print(f"  Avg MFQ flow (2x): {np.mean(flow_times)*1000:.1f} ms/pair")
    print(f"  Avg IFRNet synth:  {np.mean(synth_times)*1000:.1f} ms/frame")
    print(f"  Output:            {OUTPUT_PATH}")
    print(f"  Output size:       {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
    print(f"{'='*60}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
