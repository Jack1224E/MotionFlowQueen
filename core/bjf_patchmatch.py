import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.census_kernel import run_census
from kernels.jfa_kernel import run_jfa

class BJFPatchMatch(nn.Module):
    def __init__(self, h=1080, w=1920):
        super().__init__()
        # --- Phase 1 Buffers ---
        # Persistent buffer for Luma (2 frames, H, W)
        self.luma_buffer = torch.empty((2, h, w), dtype=torch.float32, device='cuda')
        
        # --- Phase 2 Buffers (Pre-allocated, NO runtime allocs) ---
        # Block grid dimensions (8x8 downsampling)
        self.block_size = 8
        gh = h // self.block_size  # 135 for 1080p
        gw = w // self.block_size  # 240 for 1920p
        
        # Ping-Pong flow buffers (packed uint32 state)
        self.flow_a = torch.empty((1, gh, gw), dtype=torch.int32, device='cuda')
        self.flow_b = torch.empty((1, gh, gw), dtype=torch.int32, device='cuda')
        
        # Confidence mask (uint8)
        self.confidence = torch.empty((1, gh, gw), dtype=torch.uint8, device='cuda')
        
        # Cache grid dims
        self._gh = gh
        self._gw = gw

    def check_zero_motion(self, luma1, luma2, threshold=0.00001):
        # 16x downsample for "virtually free" check
        # luma is (B, H, W) -> need (B, 1, H, W) for avg_pool
        d1 = F.avg_pool2d(luma1.unsqueeze(1), kernel_size=16, stride=16)
        d2 = F.avg_pool2d(luma2.unsqueeze(1), kernel_size=16, stride=16)
        
        mad = torch.mean(torch.abs(d1 - d2))
        return mad < threshold

    def forward(self, img1, img2):
        """Full pipeline: Luma -> Zero-Motion Check -> Census -> Block-Grid JFA -> Flow Output"""
        
        B, C, H, W = img1.shape
        
        # --- Luma Computation ---
        needed_items = 2 * B
        
        if (self.luma_buffer.shape[0] < needed_items) or \
           (self.luma_buffer.shape[1] != H) or \
           (self.luma_buffer.shape[2] != W):
            self.luma_buffer = torch.empty((needed_items, H, W), dtype=img1.dtype, device=img1.device)
            
        current_buffer = self.luma_buffer[:needed_items]
        
        if C == 3:
            luma1_view = current_buffer[:B]
            luma2_view = current_buffer[B:]
            luma1_view[:] = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            luma2_view[:] = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            current_buffer[:B] = img1.squeeze(1)
            current_buffer[B:] = img2.squeeze(1)
            
        # --- Phase 1: Zero-Motion Check ---
        if self.check_zero_motion(current_buffer[:B], current_buffer[B:]):
            return None  # Signal to orchestrator that flow is zero
            
        # --- Phase 1: Census Transform ---
        ct_stack_full, ct_stack_half = run_census(current_buffer)
        
        # Split census maps: frame1 = first B, frame2 = last B
        census1_full = ct_stack_full[:B]  # (B, H, W) int32
        census2_full = ct_stack_full[B:]  # (B, H, W) int32
        
        # --- Phase 2: Block-Grid JFA ---
        # Downsample census to 8x8 block grid using avg_pool
        # Census is int32 bitmask. avg_pool on bitmask is nonsensical.
        # Instead: take the census value at the center of each 8x8 block.
        # Strided slice: census1_full[:, ::8, ::8] gives (B, GH, GW)
        census1_grid = census1_full[:, ::self.block_size, ::self.block_size].contiguous()
        census2_grid = census2_full[:, ::self.block_size, ::self.block_size].contiguous()
        
        gh, gw = census1_grid.shape[1], census1_grid.shape[2]
        
        # Ensure ping-pong buffers match batch size and grid dims
        if self.flow_a.shape[0] < B or self.flow_a.shape[1] != gh or self.flow_a.shape[2] != gw:
            self.flow_a = torch.empty((B, gh, gw), dtype=torch.int32, device=img1.device)
            self.flow_b = torch.empty((B, gh, gw), dtype=torch.int32, device=img1.device)
            self.confidence = torch.empty((B, gh, gw), dtype=torch.uint8, device=img1.device)
        
        # Run JFA
        dx, dy, conf = run_jfa(
            census1_grid, census2_grid,
            self.flow_a[:B], self.flow_b[:B],
            self.confidence[:B]
        )
        
        # Scale flow vectors back from block-grid to pixel space
        # Each grid cell = 8 pixels
        dx_px = dx.to(torch.float32) * self.block_size
        dy_px = dy.to(torch.float32) * self.block_size
        
        return dx_px, dy_px, conf
