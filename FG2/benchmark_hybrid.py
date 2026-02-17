#!/home/jack/Documents/FG2/venv/bin/python3
import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
sys.path.append(os.path.abspath('repo_ifrnet'))

# Import our Mission 2 modules
try:
    from nvofa_wrapper import NvOFA
    from disflow_wrapper import DISFlowWrapper # Mission 3
    from flow_utils import process_nvofa_flow
    from trust_mask import TrustMask
    print("Mission 2 modules imported successfully.")
except ImportError as e:
    print(f"Failed to import Mission 2 modules: {e}")
    sys.exit(1)

# IFRNet Import
try:
    from models.IFRNet_S import Model as IFRNetModel
    print("Successfully imported IFRNet_S")
except ImportError as e:
    print(f"Failed to import IFRNet: {e}")
    sys.exit(1)

# Force CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def load_ifrnet():
    print("Loading IFRNet model...")
    model = IFRNetModel()
    model.to(device)
    model.eval()
    # IFRNet weights path
    path = 'repo_ifrnet/checkpoints/IFRNet_S.pth'
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print("IFRNet weights loaded.")
        except Exception as e:
            print(f"Error loading IFRNet weights: {e}")
            return None
    else:
        print(f"Warning: IFRNet weights not found at {path}")
        return None
    return model

def run_hybrid_pipeline(model, nvofa, trust_mask_gen, img0, img1):
    # Prepare Inputs
    h, w, _ = img0.shape
    # Padding for IFRNet (divisible by 32)
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, 0, pw - w, ph - h)
    
    img0_torch = torch.tensor(img0).permute(2, 0, 1).float() / 255.0
    img1_torch = torch.tensor(img1).permute(2, 0, 1).float() / 255.0
    img0_torch = img0_torch.to(device).unsqueeze(0)
    img1_torch = img1_torch.to(device).unsqueeze(0)
    
    img0_pad = F.pad(img0_torch, padding)
    img1_pad = F.pad(img1_torch, padding)
    
    embt = torch.tensor(0.5).view(1, 1, 1, 1).to(device) # Middle frame
    
    # --- HYBRID PIPELINE START ---
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 1. FastFlow Hardware Tap
    start_fast_time = time.time()
    try:
        # compute_flow_and_trust returns (flow, trust)
        raw_flow, raw_trust = nvofa.compute_flow_and_trust(img0_pad, img1_pad)
    except RuntimeError as e:
        print(f"FastFlow Execute failed: {e}")
        return None, 0
    
    # 2. Process Flow & Scaling
    # We only want 1/8 and 1/4 scales as per Mission 3.5 instructions.
    # process_nvofa_flow might be reusable, but let's be explicit here or use it if flexible.
    # flow_utils.process_nvofa_flow takes raw_flow and scales.
    
    # Update: we need to scale the TRUST MASK too.
    # Let's adapt flow_utils or just do it inline for transparency.
    flow_dict = {}
    mask_dict = {}
    
    target_scales = {'1/8': 1/8, '1/4': 1/4}
    
    for key, sc in target_scales.items():
        # Flow scaling: resize AND multiply values
        scaled_flow = F.interpolate(raw_flow, scale_factor=sc, mode='bilinear', align_corners=False)
        scaled_flow = scaled_flow * sc
        flow_dict[key] = scaled_flow

        # Mask scaling: resize (area/bilinear ok for mask?) User said "area interpolation".
        scaled_mask = F.interpolate(raw_trust, scale_factor=sc, mode='area')
        mask_dict[key] = scaled_mask

    # 3. IFRNet Inference with Injection
    # We pass the dictionaries.
    # Note: external_flow and trust_mask args in IFRNet_S.py expect these dicts or tensors?
    # Our previous edit to IFRNet_S.py checks "if external_flow is not None and '1/8' in external_flow".
    # So it expects a dict.
    
    inference_result = model.inference(img0_pad, img1_pad, embt, external_flow=flow_dict, trust_mask=mask_dict)
    
    torch.cuda.synchronize()
    end_time = time.time()
    # --- HYBRID PIPELINE END ---

    # Standard Output Processing
    res = inference_result[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    res = res[:h, :w]
    res = res.astype('uint8')
    
    return res, end_time - start_time

def main():
    # Setup test data
    if not os.path.exists('input_frames/frame1.png'):
        print("Test frames not found. Creating dummy data...")
        os.makedirs('input_frames', exist_ok=True)
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite('input_frames/frame1.png', dummy)
        cv2.imwrite('input_frames/frame3.png', dummy)
        
    img0 = cv2.imread('input_frames/frame1.png')
    img1 = cv2.imread('input_frames/frame3.png')
    
    # Resizing
    max_w = 640
    if img0.shape[1] > max_w:
        scale = max_w / img0.shape[1]
        h = int(img0.shape[0] * scale)
        w = int(img0.shape[1] * scale)
        h = ((h - 1) // 32 + 1) * 32
        w = ((w - 1) // 32 + 1) * 32
        img0 = cv2.resize(img0, (w, h))
        img1 = cv2.resize(img1, (w, h))
        print(f"Resized to {w}x{h}")

    # Initialize Components
    ifrnet = load_ifrnet()
    if ifrnet is None:
        print("Failed to load IFRNet. Exiting.")
        return

    # MISSION 3.5 PIVOT: Use FastFlowEngine (Lazy Flow + Trust)
    # MISSION 4 NOTE: FastFlowEngine(use_sad=True) available but slower (143ms).
    try:
        from fast_flow_engine import FastFlowEngine
        nvofa = FastFlowEngine(use_sad=False)
        print(" Using FastFlowEngine (Lazy DIS + Trust Mask)")
    except ImportError as e:
        print(f"Failed to initialize FastFlowEngine: {e}")
        return
    except Exception as e:
        print(f"Failed to initialize FastFlowEngine: {e}")
        return

    trust_mask_gen = TrustMask()
    
    print("Starting Hybrid Pipeline Benchmark...")
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        res, t = run_hybrid_pipeline(ifrnet, nvofa, trust_mask_gen, img0, img1)
    
    # Benchmark
    print("Benchmarking (10 runs, skipping first for compilation/warmup)...")
    latencies = []
    for i in range(10):
        res, lat = run_hybrid_pipeline(ifrnet, nvofa, trust_mask_gen, img0, img1)
        if res is None:
            continue
        print(f"Run {i+1}: {lat:.4f}s")
        latencies.append(lat)
        
    if len(latencies) > 1:
        # Skip the first run (compilation overhead)
        avg_lat = sum(latencies[1:]) / (len(latencies) - 1)
        print(f"\nAverage Hybrid Inference Time (Runs 2-10): {avg_lat:.4f}s")
    else:
        print("Not enough successful runs.")
        os.makedirs('output_frames', exist_ok=True)
    output_path = 'output_frames/hybrid_dis_modified.png'
    cv2.imwrite(output_path, res)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
