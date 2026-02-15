import torch
import time
import argparse
from core.bjf_patchmatch import BJFPatchMatch

def benchmark():
    print("Initializing MotionFlowQueen Benchmark...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dummy data setup (1080p)
    H, W = 1080, 1920
    # Simulate Video Loop (10 frames)
    # 3 frames are duplicates (Zero Motion)
    frames = []
    
    # Base frames
    f0 = torch.rand(1, 3, H, W).to(device)
    f1 = torch.rand(1, 3, H, W).to(device)
    f2 = torch.rand(1, 3, H, W).to(device)
    f3 = torch.rand(1, 3, H, W).to(device)
    f4 = torch.rand(1, 3, H, W).to(device)
    f5 = torch.rand(1, 3, H, W).to(device)
    f6 = torch.rand(1, 3, H, W).to(device)
    
    # 0. Normal Motion
    frames.append((f0, f1))
    # 1. Normal Motion
    frames.append((f1, f2))
    # 2. Zero Motion (Duplicate)
    frames.append((f2, f2))
    # 3. Normal Motion
    frames.append((f2, f3))
    # 4. Zero Motion (Duplicate)
    frames.append((f3, f3))
    # 5. Normal Motion
    frames.append((f3, f4))
    # 6. Normal Motion
    frames.append((f4, f5))
    # 7. Zero Motion (Duplicate)
    frames.append((f5, f5))
    # 8. Normal Motion
    frames.append((f5, f6))
    # 9. Normal Motion
    frames.append((f6, f0))
    
    # Total 10 steps. 3 are duplicates (30%).
    
    # Initialize model
    model = BJFPatchMatch(H, W).to(device)
    
    # --- Verification Step ---
    print("\n--- Verification: Checking Zero-Motion Logic ---")
    for i, (f_curr, f_next) in enumerate(frames):
        is_duplicate = torch.allclose(f_curr, f_next)
        output = model(f_curr, f_next)
        
        status = "Zero Motion (Skipped)" if output is None else "Motion Detected (Processed)"
        # Check correctness
        correct = False
        if is_duplicate and output is None: correct = True
        elif not is_duplicate and output is not None: correct = True
        
        print(f"Frame Pair {i}: Duplicate={is_duplicate} -> Result: {status} [{'OK' if correct else 'FAIL'}]")
        
        if output is not None:
            # Print a snippet of the census transform
            # output is (B, H, W) int32
            snippet = output[0, H//2, W//2-4:W//2+4].cpu().numpy()
            print(f"    Census Sample (Center): {snippet}")

    torch.cuda.synchronize()
    
    # Timing
    print("\n--- Benchmark: Amortized Performance ---")
    print("Running amortized benchmark (50 loops x 10 frames = 500 frames)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        for img1, img2 in frames:
            _ = model(img1, img2)
    end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_per_frame = total_time_ms / 500.0
    
    print(f"Total Time: {total_time_ms:.2f} ms")
    print(f"Amortized Average Execution Time: {avg_per_frame:.4f} ms")

if __name__ == "__main__":
    benchmark()
