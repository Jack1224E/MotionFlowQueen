#!/usr/bin/env python3
"""
500-Frame Endurance Stress Test V2 (with Nervous System)
Includes shock detection, adaptive EMA, quadratic vetoing.
"""
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.bjf_patchmatch import BJFPatchMatch
from core.refine import refine_flow, reset_temporal
from core.shock import ShockDetector

MAX_FRAMES = 500
SAVE_EVERY = 100
ANOMALY_VEC_THRESH = 64.0
ANOMALY_ERR_THRESH = 0.15
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'validation_output')


def flow_to_middlebury(dx, dy, max_mag=None):
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)
    if max_mag is None:
        max_mag = max(float(mag.max()), 1e-5)
    hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def warp_error_np(img1, img2, dx, dy):
    H, W = img1.shape[:2]
    gy, gx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')
    warped = cv2.remap(img1, (gx + dx).astype(np.float32),
                       (gy + dy).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return float(np.abs(warped.astype(np.float32) - img2.astype(np.float32)).mean() / 255.0)


def main():
    device = 'cuda'
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sample_darksouls2.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}"); sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total_frames - 1, MAX_FRAMES)
    print(f"[INFO] Video: {W}x{H}, {total_frames} total, processing {n} pairs")
    print(f"[INFO] Nervous System: shock_det + adaptive_ema + quadratic_veto")

    model = BJFPatchMatch(H, W).to(device)
    reset_temporal()
    shock_det = ShockDetector(mad_threshold=30.0, err_spike_ratio=1.5)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_errs, all_mags, all_max_vecs = [], [], []
    anomalies, shocks = [], []

    ret, prev_bgr = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame"); sys.exit(1)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    t_total = time.time()
    for i in range(n):
        ret, cur_bgr = cap.read()
        if not ret:
            print(f"[WARN] EOF at frame {i}"); break
        cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

        # --- Shock Detection (pre-processing) ---
        is_shock_pre, shock_reason = shock_det.check(prev_gray, cur_gray)
        if is_shock_pre:
            reset_temporal()
            msg = f"  ⚡ SHOCK Frame {i}: {shock_reason} (temporal reset)"
            print(msg)
            shocks.append(msg)

        # --- MFQ Finesse ---
        t1 = torch.from_numpy(prev_bgr).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(device)
        t2 = torch.from_numpy(cur_bgr).permute(2, 0, 1).float().div_(255).unsqueeze_(0).to(device)

        result = model(t1, t2)
        torch.cuda.synchronize()

        if result is None:
            fin_dx_np = np.zeros((H, W), np.float32)
            fin_dy_np = np.zeros((H, W), np.float32)
        else:
            dx_grid, dy_grid, conf = result
            fin_dx, fin_dy = refine_flow(dx_grid, dy_grid, conf, t1, t2, block_size=8, temporal=True)
            torch.cuda.synchronize()
            fin_dx_np = fin_dx[0].cpu().numpy()
            fin_dy_np = fin_dy[0].cpu().numpy()

        # --- Metrics ---
        err = warp_error_np(prev_bgr, cur_bgr, fin_dx_np, fin_dy_np)
        mag = float(np.sqrt(fin_dx_np**2 + fin_dy_np**2).mean())
        max_vec = float(np.sqrt(fin_dx_np**2 + fin_dy_np**2).max())

        # --- Post-hoc shock check (warp error spike) ---
        if not is_shock_pre:
            is_shock_post, shock_reason_post = shock_det.check(prev_gray, cur_gray, warp_error=err)
            if is_shock_post:
                reset_temporal()
                msg = f"  ⚡ SHOCK Frame {i}: {shock_reason_post} (post-error reset)"
                print(msg)
                shocks.append(msg)
        else:
            shock_det.check(prev_gray, cur_gray, warp_error=err)  # update history

        all_errs.append(err)
        all_mags.append(mag)
        all_max_vecs.append(max_vec)

        # --- Anomaly Detection ---
        if max_vec > ANOMALY_VEC_THRESH:
            msg = f"  ⚠️  ANOMALY Frame {i}: max_vec={max_vec:.1f}px"
            print(msg)
            anomalies.append(msg)
        if err > ANOMALY_ERR_THRESH:
            msg = f"  ⚠️  ANOMALY Frame {i}: warp_err={err:.4f}"
            print(msg)
            anomalies.append(msg)

        # --- Save comparison every SAVE_EVERY frames ---
        if (i + 1) % SAVE_EVERY == 0 or i == 0:
            dis_flow = dis.calc(prev_gray, cur_gray, None)
            dis_dx, dis_dy = dis_flow[:, :, 0], dis_flow[:, :, 1]
            dis_err = warp_error_np(prev_bgr, cur_bgr, dis_dx, dis_dy)
            shared_max = max(float(np.sqrt(fin_dx_np**2 + fin_dy_np**2).max()),
                             float(np.sqrt(dis_dx**2 + dis_dy**2).max()), 1e-5)
            p1 = cur_bgr.copy()
            cv2.putText(p1, f"F{i}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            p2 = flow_to_middlebury(dis_dx, dis_dy, shared_max)
            cv2.putText(p2, f"DIS {dis_err:.3f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            p3 = flow_to_middlebury(fin_dx_np, fin_dy_np, shared_max)
            cv2.putText(p3, f"MFQ {err:.3f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            combined = np.concatenate([p1, p2, p3], axis=1)
            cv2.imwrite(os.path.join(OUT_DIR, f'comparison_nervous_{i:04d}.png'), combined)
            print(f"  [SAVE] comparison_nervous_{i:04d}.png")

        if (i + 1) % 50 == 0:
            avg_err = np.mean(all_errs[-50:])
            avg_mag = np.mean(all_mags[-50:])
            avg_max = np.mean(all_max_vecs[-50:])
            elapsed = time.time() - t_total
            print(f"  [{i+1}/{n}] err={avg_err:.4f} mag={avg_mag:.1f} max={avg_max:.1f} | {(i+1)/elapsed:.1f} fps | {elapsed:.0f}s")

        del t1, t2
        prev_bgr = cur_bgr
        prev_gray = cur_gray

    cap.release()
    elapsed_total = time.time() - t_total
    n_processed = len(all_errs)

    report = [
        "=" * 60,
        "  NERVOUS SYSTEM STRESS TEST — FINAL REPORT",
        "=" * 60,
        f"  Frames processed:   {n_processed}",
        f"  Total time:         {elapsed_total:.1f}s ({n_processed/elapsed_total:.1f} fps)",
        "",
        f"  Avg Warp Error:     {np.mean(all_errs):.4f}",
        f"  Peak Warp Error:    {max(all_errs):.4f}",
        f"  Avg Flow Magnitude: {np.mean(all_mags):.1f} px",
        f"  Avg Max Vector:     {np.mean(all_max_vecs):.1f} px",
        f"  Peak Max Vector:    {max(all_max_vecs):.1f} px",
        "",
        f"  Shocks detected:    {len(shocks)}",
        f"  Anomalies detected: {len(anomalies)}",
    ]
    for s in shocks[:10]:
        report.append(f"    {s}")
    if shocks:
        report.append("")
    for a in anomalies[:10]:
        report.append(f"    {a}")
    report.append("=" * 60)

    print("\n".join(report))
    with open(os.path.join(OUT_DIR, 'stress_nervous_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    print(f"\n[INFO] Report: {OUT_DIR}/stress_nervous_report.txt")
    print("[INFO] Done.")
    del model
    torch.cuda.empty_cache()
    sys.exit(0)


if __name__ == '__main__':
    main()
