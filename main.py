import torch
from core.bjf_patchmatch import BJFPatchMatch

def benchmark():
    print("MotionFlowQueen Phase 2 Benchmark")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    H, W = 1080, 1920
    
    # Simulate Video Sequence (10 frames, 3 duplicates for Zero-Motion)
    f0 = torch.rand(1, 3, H, W, device=device)
    f1 = torch.rand(1, 3, H, W, device=device)
    f2 = torch.rand(1, 3, H, W, device=device)
    f3 = torch.rand(1, 3, H, W, device=device)
    f4 = torch.rand(1, 3, H, W, device=device)
    f5 = torch.rand(1, 3, H, W, device=device)
    f6 = torch.rand(1, 3, H, W, device=device)
    
    frames = [
        (f0, f1),  # Motion
        (f1, f2),  # Motion
        (f2, f2),  # Zero Motion
        (f2, f3),  # Motion
        (f3, f3),  # Zero Motion
        (f3, f4),  # Motion
        (f4, f5),  # Motion
        (f5, f5),  # Zero Motion
        (f5, f6),  # Motion
        (f6, f0),  # Motion
    ]
    
    model = BJFPatchMatch(H, W).to(device)
    
    # --- Verification ---
    print("\n--- Verification ---")
    for i, (img1, img2) in enumerate(frames):
        is_dup = torch.allclose(img1, img2)
        result = model(img1, img2)
        
        if result is None:
            status = "SKIPPED (Zero Motion)"
            check = "[OK]" if is_dup else "[FAIL]"
        else:
            dx, dy, conf = result
            status = f"PROCESSED -> Flow({dx.shape}), Conf min={conf.min().item()} max={conf.max().item()}"
            check = "[OK]" if not is_dup else "[FAIL]"
        
        print(f"  Pair {i}: Dup={is_dup} -> {status} {check}")
    
    torch.cuda.synchronize()
    
    # --- Benchmark ---
    print("\n--- Benchmark (50 loops x 10 frames = 500 frames) ---")
    
    # Warm-up
    for _ in range(3):
        for img1, img2 in frames:
            _ = model(img1, img2)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50):
        for img1, img2 in frames:
            _ = model(img1, img2)
    end.record()
    
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / 500.0
    
    print(f"  Total: {total_ms:.2f} ms")
    print(f"  Amortized per frame: {avg_ms:.4f} ms")
    print(f"  Full pipeline: Census + JFA (Phase 1 + Phase 2)")

if __name__ == "__main__":
    benchmark()
