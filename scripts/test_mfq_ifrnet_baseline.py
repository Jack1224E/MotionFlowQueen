#!/usr/bin/env python3
"""
MFQ × Custom IFRNet Baseline
MFQ flow → IFRNet_S (custom checkpoint) synthesis.
This is the "best neural baseline" for visual comparison.
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
sys.path.insert(0, os.path.join(ROOT, 'FG2', 'repo_ifrnet'))

# Stub imageio
import types
if 'imageio' not in sys.modules:
    _stub = types.ModuleType('imageio')
    _stub.imread = None   # type: ignore[attr-defined]
    _stub.imwrite = None  # type: ignore[attr-defined]
    _stub.mimsave = None  # type: ignore[attr-defined]
    sys.modules['imageio'] = _stub

from core.bjf_patchmatch import BJFPatchMatch
from core.refine import refine_flow, reset_temporal

VIDEO_PATH  = os.path.join(ROOT, 'sample_darksouls2.mp4')
OUTPUT_PATH = os.path.join(ROOT, 'output_mfq_ifrnet_baseline.mp4')
MAX_FRAMES  = 300
DEVICE      = 'cuda'


def load_ifrnet():
    """Load IFRNet_S with the custom checkpoint (NOT Vimeo90K)."""
    from models.IFRNet_S import Model
    ckpt = os.path.join(ROOT, 'FG2', 'repo_ifrnet', 'checkpoints', 'IFRNet_S.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, 'FG2', 'repo_ifrnet', 'checkpoints', 'IFRNet_S_Vimeo90K.pth')
        print(f"[WARN] Custom IFRNet_S.pth not found, falling back to Vimeo90K")
    model = Model().to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    print(f"[IFRNet] Loaded from {os.path.basename(ckpt)}")
    return model


def pad_to_32(img):
    _, _, h, w = img.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    if ph != h or pw != w:
        img = F.pad(img, (0, pw - w, 0, ph - h))
    return img, h, w


def build_flow_dict(dx_fwd, dy_fwd, dx_bck, dy_bck):
    """
    Pack true bidirectional flow for IFRNet injection.
    Matches FG2/fast_flow_engine.py: [0.5*flow_bck, 0.5*flow_fwd]
    Then scale spatially and by magnitude for each target scale.
    """
    flow_01 = torch.stack([dx_fwd, dy_fwd], dim=1)
    flow_10 = torch.stack([dx_bck, dy_bck], dim=1)
    raw_flow = torch.cat([0.5 * flow_10, 0.5 * flow_01], dim=1)

    flow_dict = {}
    for key, sc in [('1/8', 1/8), ('1/4', 1/4)]:
        scaled = F.interpolate(raw_flow, scale_factor=sc, mode='bilinear', align_corners=False)
        scaled = scaled * sc
        flow_dict[key] = scaled
    return flow_dict


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n      = min(total - 1, MAX_FRAMES)

    print(f"[Video] {W}×{H}, {fps_in:.0f}fps, {total} frames → {n} pairs")
    print(f"[Output] 60fps → {OUTPUT_PATH}")

    mfq = BJFPatchMatch(H, W).to(DEVICE)
    ifrnet = load_ifrnet()
    embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(DEVICE)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in * 2, (W, H))

    ret, prev_bgr = cap.read()
    if not ret:
        print("ERROR"); sys.exit(1)
    writer.write(prev_bgr)

    t_start = time.time()
    flow_ms, synth_ms = [], []

    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            break

        t0 = torch.from_numpy(prev_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)
        t1 = torch.from_numpy(cur_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(DEVICE)

        # ── Bidirectional MFQ Flow ────────────────────────
        t_f = time.time()
        reset_temporal()
        res_fwd = mfq(t0, t1)
        if res_fwd is not None:
            dx01, dy01 = refine_flow(res_fwd[0], res_fwd[1], res_fwd[2], t0, t1, block_size=8, temporal=False)
        else:
            dx01 = torch.zeros(1, H, W, device=DEVICE)
            dy01 = torch.zeros(1, H, W, device=DEVICE)

        reset_temporal()
        res_bck = mfq(t1, t0)
        if res_bck is not None:
            dx10, dy10 = refine_flow(res_bck[0], res_bck[1], res_bck[2], t1, t0, block_size=8, temporal=False)
        else:
            dx10 = torch.zeros(1, H, W, device=DEVICE)
            dy10 = torch.zeros(1, H, W, device=DEVICE)

        torch.cuda.synchronize()
        flow_ms.append((time.time() - t_f) * 1000)

        # ── IFRNet Synthesis ──────────────────────────────
        t_s = time.time()
        img0_pad, oh, ow = pad_to_32(t0)
        img1_pad, _, _    = pad_to_32(t1)
        _, _, ph, pw = img0_pad.shape

        # Pad flow
        if ph != H or pw != W:
            dx01_p = F.pad(dx01.unsqueeze(1), (0, pw-W, 0, ph-H)).squeeze(1)
            dy01_p = F.pad(dy01.unsqueeze(1), (0, pw-W, 0, ph-H)).squeeze(1)
            dx10_p = F.pad(dx10.unsqueeze(1), (0, pw-W, 0, ph-H)).squeeze(1)
            dy10_p = F.pad(dy10.unsqueeze(1), (0, pw-W, 0, ph-H)).squeeze(1)
        else:
            dx01_p, dy01_p = dx01, dy01
            dx10_p, dy10_p = dx10, dy10

        ext_flow = build_flow_dict(dx01_p, dy01_p, dx10_p, dy10_p)

        with torch.no_grad():
            imgt = ifrnet.inference(img0_pad, img1_pad, embt, external_flow=ext_flow)

        imgt_np = (imgt[0, :, :oh, :ow].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        imgt_bgr = imgt_np[:, :, ::-1].copy()
        torch.cuda.synchronize()
        synth_ms.append((time.time() - t_s) * 1000)

        writer.write(imgt_bgr)
        writer.write(cur_bgr)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{n}] flow={np.mean(flow_ms[-50:]):.1f}ms  "
                  f"synth={np.mean(synth_ms[-50:]):.1f}ms  "
                  f"({(i+1)/elapsed:.1f} pairs/s)")

        del t0, t1
        prev_bgr = cur_bgr

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    n_done = len(flow_ms)
    total_pipe = np.mean(flow_ms) + np.mean(synth_ms)
    print(f"\n{'='*60}")
    print(f"  MFQ × IFRNet BASELINE — DONE")
    print(f"{'='*60}")
    print(f"  Frames:       {n_done} pairs → {n_done * 2 + 1} output frames")
    print(f"  Total time:   {elapsed:.1f}s ({n_done/elapsed:.1f} pairs/s)")
    print(f"  MFQ flow:     {np.mean(flow_ms):.1f} ms")
    print(f"  IFRNet synth: {np.mean(synth_ms):.1f} ms")
    print(f"  Pipeline:     {total_pipe:.1f} ms/frame")
    print(f"  Output:       {OUTPUT_PATH}")
    print(f"  Size:         {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
