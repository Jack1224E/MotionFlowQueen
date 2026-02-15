import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.census_kernel import run_census
# from kernels.jfa_kernel import jump_flood_algorithm

class BJFPatchMatch(nn.Module):
    def __init__(self, h=1080, w=1920):
        super().__init__()
        # Precompute Luma weights? No, usage is arithmetic.
        # Persistent buffer for Luma (2 frames, H, W)
        # Using B=1 for default allocation from main.py
        # Shape: (2, h, w). If B > 1, we resize in forward.
        self.luma_buffer = torch.empty((2, h, w), dtype=torch.float32, device='cuda')

    def check_zero_motion(self, luma1, luma2, threshold=0.00001):
        # 16x downsample for "virtually free" check
        # luma is (B, H, W) -> need (B, 1, H, W) for avg_pool
        d1 = F.avg_pool2d(luma1.unsqueeze(1), kernel_size=16, stride=16)
        d2 = F.avg_pool2d(luma2.unsqueeze(1), kernel_size=16, stride=16)
        
        mad = torch.mean(torch.abs(d1 - d2))
        return mad < threshold

    def forward(self, img1, img2):
        # 2. Census Transform
        # Baseline Optimized: Efficient Float Arithmetic + Stable Kernel + Persistent Buffer
        
        B, C, H, W = img1.shape
        
        # Check buffer size
        needed_items = 2 * B
        
        if (self.luma_buffer.shape[0] < needed_items) or \
           (self.luma_buffer.shape[1] != H) or \
           (self.luma_buffer.shape[2] != W):
             # Reallocate
             # Keep float32 as it is faster than casting overhead on 1650
            self.luma_buffer = torch.empty((needed_items, H, W), dtype=img1.dtype, device=img1.device)
            
        current_buffer = self.luma_buffer[:needed_items]
        
        # In-place write to buffer
        if C == 3:
            # Planar RGB to Luma (ITU-R BT.601) - In-place
            # img1 -> current_buffer[0:B]
            # img2 -> current_buffer[B:2*B]
            
            # Note: We compute directly into buffer to save memory.
            # But for check_zero_motion, we ideally want to check BEFORE full luma calc?
            # Or is luma calc cheap enough? Luma calc is ~0.1-0.2ms. 
            # Downsample is also cheap.
            # Let's compute luma first (needed anyway if not static).
            
            luma1_view = current_buffer[:B]
            luma2_view = current_buffer[B:]
            
            luma1_view[:] = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            luma2_view[:] = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            current_buffer[:B] = img1.squeeze(1)
            current_buffer[B:] = img2.squeeze(1)
            
        # Zero-Motion Check
        # Check if frames are effectively identical (dirty rectangle = null)
        if self.check_zero_motion(current_buffer[:B], current_buffer[B:]):
            return None # Signal to orchestrator that flow is zero
            
        # Run Census
        # Input (2B, H, W)
        ct_stack_full, ct_stack_half = run_census(current_buffer)
        
        # Split back (views, no copy)
        ct1 = ct_stack_full[:B]
        
        return ct1
