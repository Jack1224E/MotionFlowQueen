import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniSADBlockMatcher(nn.Module):


    def _compute_one_way(self, img_ref, img_trg):
        """
        Computes optical flow from img_ref to img_trg using SAD Block Matching.
        Returns: flow (B, 2, H, W)
        """
        B, C, H, W = img_ref.shape
        
        # 1. Pad Target Image for Slicing
        # We need to slice img_trg at offsets -R to +R.
        # Pad by search_range.
        pad_r = self.search_range
        # Padding: (left, right, top, bottom)
        img_trg_pad = F.pad(img_trg, (pad_r, pad_r, pad_r, pad_r), mode='replicate')
        
        # 2. Search Grid
        # We scan dys, dxs.
        # We want to maintain min_cost and best_flow.
        
        # Initialize Cost Map with infinity
        min_cost = torch.full((B, 1, H, W), float('inf'), device=img_ref.device, dtype=img_ref.dtype)
        flow_x = torch.zeros((B, 1, H, W), device=img_ref.device, dtype=img_ref.dtype)
        flow_y = torch.zeros((B, 1, H, W), device=img_ref.device, dtype=img_ref.dtype)
        
        # Pre-calculate pooling kernel for avg_pool2d
        # We use functional interface, no init needed.
        
        # Loop
        # Range includes +R? usually -R to +R inclusive? 
        # range(start, stop, step). stop is exclusive. 
        # So range(-R, R+1, step).
        
        for dy in range(-self.search_range, self.search_range + 1, self.step):
            for dx in range(-self.search_range, self.search_range + 1, self.step):
                # Slice img_trg corresponding to shift (dx, dy)
                # If d = 0, we take center: pad_r : pad_r+H
                # If d = +k, we take pad_r+k : ...
                # img_trg_pad indices:
                y_start = pad_r + dy
                y_end = y_start + H
                x_start = pad_r + dx
                x_end = x_start + W
                
                img_trg_shifted = img_trg_pad[:, :, y_start:y_end, x_start:x_end]
                
                # SAD
                diff = torch.abs(img_ref - img_trg_shifted)
                # Sum over channels
                diff_sum = diff.sum(dim=1, keepdim=True)
                
                # Block Sum (Average Pooling)
                # kernel_size=self.block_size
                # stride=1
                # padding=self.pool_padding
                cost = F.avg_pool2d(diff_sum, kernel_size=self.block_size, stride=1, padding=self.pool_padding)
                
                # Handle padding size mismatch if odd/even issues
                # Logic: pool output might be slightly off. Trim to H, W.
                if cost.shape[2:] != (H, W):
                    cost = cost[:, :, :H, :W] # Naive trim
                
                # Update Best
                mask = cost < min_cost
                min_cost = torch.where(mask, cost, min_cost)
                flow_x = torch.where(mask, torch.tensor(dx, device=img_ref.device, dtype=img_ref.dtype), flow_x)
                flow_y = torch.where(mask, torch.tensor(dy, device=img_ref.device, dtype=img_ref.dtype), flow_y)
                
        return torch.cat([flow_x, flow_y], dim=1)

    # Compile the heavy loop?
    # Putting the loop in a sub-function helps compilation.
    
    def forward(self, img0, img1):
        """
        Args:
            img0, img1: (B, 3, H, W) float tensors.
        Returns:
            flow_out: (B, 4, H, W) stacked [flow_bck, flow_fwd]
                      flow_bck is img1->img0
                      flow_fwd is img0->img1
        """
        # img0->img1 (Forward)
        flow_fwd = self._compute_one_way(img0, img1)
        
        # img1->img0 (Backward)
        flow_bck = self._compute_one_way(img1, img0)
        
        # Stack for IFRNet: needs 4 channels.
        # User requested explicitly stack flow_fwd and flow_bck along dim=1.
        # Usually IFRNet helpers take (bck, fwd) order? Or (fwd, bck)?
        # fast_flow_engine used: cat([0.5*bck, 0.5*fwd]).
        # Benchmark script expects to pass this to trust mask gen AND IFRNet.
        # IFRNet inference: up_flow0_4 = external_flow...[:, 0:2].
        # In `fast_flow_engine`, we put `bck` first.
        # So we cat [bck, fwd].
        
        flow_out = torch.cat([flow_bck, flow_fwd], dim=1)
        return flow_out

        flow_out = torch.cat([flow_bck, flow_fwd], dim=1)
        return flow_out

    # Optimization: Compile the loop-heavy method
    def __init__(self, search_range=16, block_size=8, step=2):
        super().__init__()
        self.search_range = search_range
        self.block_size = block_size
        self.step = step
        self.pool_padding = block_size // 2
        
        # Compile if available
        if hasattr(torch, 'compile'):
            print("MiniSAD: Enabling torch.compile(mode='default')...")
            self._compute_one_way = torch.compile(self._compute_one_way, mode="default") 
