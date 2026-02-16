#!/usr/bin/env python3
"""Quick diagnostic: check raw coarse vector magnitudes."""
import torch, cv2, sys
sys.path.insert(0, '/home/jack/Documents/MotionFlowQueen')
from core.bjf_patchmatch import BJFPatchMatch

cap = cv2.VideoCapture('/home/jack/Documents/MotionFlowQueen/sample_darksouls2.mp4')
ret, f1 = cap.read()
ret, f2 = cap.read()
cap.release()

t1 = torch.from_numpy(f1[:,:,::-1].copy()).permute(2,0,1).float().div(255).unsqueeze(0).cuda()
t2 = torch.from_numpy(f2[:,:,::-1].copy()).permute(2,0,1).float().div(255).unsqueeze(0).cuda()

bjf = BJFPatchMatch(h=f1.shape[0], w=f1.shape[1])
dx, dy, conf = bjf(t1, t2)
dx_px = dx.float() * 8
dy_px = dy.float() * 8
mag = torch.sqrt(dx_px**2 + dy_px**2)
print(f"Max coarse mag (px): {mag.max():.1f}")
print(f"Mean coarse mag (px): {mag.mean():.1f}")
print(f"Blocks >64px: {(mag > 64).sum().item()}/{mag.numel()}")
print(f"Blocks >80px: {(mag > 80).sum().item()}/{mag.numel()}")
print(f"Max |dx_px|: {dx_px.abs().max():.1f}")
print(f"Max |dy_px|: {dy_px.abs().max():.1f}")
