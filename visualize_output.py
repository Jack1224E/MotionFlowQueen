import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from core.bjf_patchmatch import BJFPatchMatch

def visualize():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize Model
    # Need H, W from video first
    video_path = "sample.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Resolution: {W}x{H}")

    model = BJFPatchMatch(H, W).to(device)
    
    # Output dir
    out_dir = "vis_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Read first 10 frames
    frames_cv = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret: break
        frames_cv.append(frame)
    cap.release()
    
    print(f"Read {len(frames_cv)} frames.")
    
    test_pairs = []
    # Real Motion Sequence (Consecutive frames)
    for i in range(len(frames_cv) - 1):
        test_pairs.append((frames_cv[i], frames_cv[i+1], "Motion"))
        
    # Inject one artificial Duplicate at index 5
    if len(frames_cv) > 5:
        test_pairs.insert(5, (frames_cv[5], frames_cv[5], "Artificial-Duplicate"))
    
    prev_census_vis = None
    
    print("\n--- MAD Analysis ---")
    print(f"{'Pair':<5} | {'Type':<20} | {'MAD Value':<12} | {'Threshold':<10} | {'Decision':<10}")
    print("-" * 70)

    for i, (img1_cv, img2_cv, label) in enumerate(test_pairs):
        # Convert to Tensor
        t1 = torch.from_numpy(img1_cv).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(img2_cv).permute(2, 0, 1).float() / 255.0
        
        t1 = t1.unsqueeze(0).to(device)
        t2 = t2.unsqueeze(0).to(device)
        
        # Independent MAD Check
        luma1 = 0.299 * t1[:, 0] + 0.587 * t1[:, 1] + 0.114 * t1[:, 2]
        luma2 = 0.299 * t2[:, 0] + 0.587 * t2[:, 1] + 0.114 * t2[:, 2]
        d1 = F.avg_pool2d(luma1.unsqueeze(1), kernel_size=16, stride=16)
        d2 = F.avg_pool2d(luma2.unsqueeze(1), kernel_size=16, stride=16)
        mad = torch.mean(torch.abs(d1 - d2)).item()
        
        # Run Model
        output = model(t1, t2)
        
        is_skipped = (output is None)
        # Check against MODEL'S threshold (0.00001 now)
        decision_match = (mad < 0.00001) == is_skipped
        
        print(f"{i:<5} | {label:<20} | {mad:.6f}     | 0.00001    | {'SKIP' if is_skipped else 'PROCESS'} {'[OK]' if decision_match else '[MISMATCH]'}")
        
        # Visualize Input (Image 2 - Current)
        vis_input = img2_cv.copy()
        cv2.putText(vis_input, f"Frame {i} ({label})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_input, f"MAD: {mad:.6f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if is_skipped:
            status = "SKIPPED"
            color = (0, 0, 255)
            if prev_census_vis is not None:
                vis_census = prev_census_vis.copy()
                cv2.putText(vis_census, "REUSED PREVIOUS", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                vis_census = np.zeros_like(img1_cv)
            cv2.putText(vis_census, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            status = "PROCESSED"
            color = (255, 255, 0)
            cens = output[0].float().cpu().numpy()
            c_min, c_max = cens.min(), cens.max()
            cens_norm = (cens - c_min) / (c_max - c_min) * 255.0 if c_max > c_min else np.zeros_like(cens)
            vis_census = cv2.cvtColor(cens_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            prev_census_vis = vis_census.copy()
            cv2.putText(vis_census, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        combined = np.hstack((vis_input, vis_census))
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_{label}.jpg"), combined)

if __name__ == "__main__":
    visualize()
