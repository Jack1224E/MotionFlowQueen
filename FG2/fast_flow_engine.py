import cv2
import torch
import numpy as np
import torch.nn.functional as F

class FastFlowEngine:
    def __init__(self, use_sad=False):
        """
        Initializes the Flow engine.
        Args:
            use_sad (bool): If True, use MiniSADBlockMatcher (GPU).
                            If False, use OpenCV DIS (CPU).
        """
        self.use_sad = use_sad
        
        if self.use_sad:
            from mini_sad_engine import MiniSADBlockMatcher
            self.sad = MiniSADBlockMatcher(search_range=16, block_size=8, step=2)
            print("Initialized FastFlowEngine (GPU Native SAD)")
        else:
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            # self.dis.setUseVariationalRefinement(False) # Not available in this OpenCV ver
            self.dis.setVariationalRefinementIterations(0) # Equivalent to disabling
            self.dis.setUseSpatialPropagation(False) # As requested for "lazy" behavior
            print("Initialized FastFlowEngine (Lazy DIS: No VarRef, No SpatialProp)")

    def compute_flow_and_trust(self, img0, img1):
        """
        Computes forward flow and a trust mask.
        
        Args:
            img0, img1: torch.Tensor (B, C, H, W), float normalized 0-1 or byte 0-255.
                        Assumes B=1 for this implementation.
        
        Returns:
            flow_fwd: torch.Tensor (1, 2, H, W)
            trust_mask: torch.Tensor (1, 1, H, W) -> 0.0 (untrustworthy) to 1.0 (trustworthy)
        """
        # 1. Convert to Numpy uint8 Gray
        def to_gray_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy()
                if arr.ndim == 4: arr = arr[0]
                if arr.shape[0] in [1, 3]: arr = arr.transpose(1, 2, 0)
            else:
                arr = tensor
            
            # Handle normalization if float 0-1
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 255).astype(np.uint8)
            
            if arr.shape[-1] == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            elif arr.shape[-1] == 1:
                gray = arr.squeeze(-1)
            else:
                gray = arr
            return gray

        # Determine device from input
        device = img0.device if isinstance(img0, torch.Tensor) else torch.device('cpu')

        if self.use_sad:
            # GPU Path (SAD)
            # Inputs are already tensors (usually on GPU if benchmarking).
            # fast_flow_engine inputs are img0, img1.
            # MiniSAD expects (B, 3, H, W).
            # Benchmark passes (1, 3, H, W).
            
            # Ensure tensors
            if not isinstance(img0, torch.Tensor):
                # Fallback for numpy input (unlikely in this pipeline now)
                pass 
                
            # Compute Flow (Bck, Fwd) stacked
            flow_out_sad = self.sad(img0, img1)
            
            # Split for trust mask calculation
            # flow_out_sad is (B, 4, H, W) -> [bck, fwd]
            flow_bck = flow_out_sad[:, 0:2]
            flow_fwd = flow_out_sad[:, 2:4]
            
            # For IFRNet, we need 0.5 * flow.
            # But we calculate trust mask on FULL flow first.
            
        else:
            # CPU Path (DIS)
            g0 = to_gray_numpy(img0)
            g1 = to_gray_numpy(img1)
            
            # 2. Compute Bidirectional Flow
            # Forward: 0 -> 1
            flow_fwd_np = self.dis.calc(g0, g1, None)
            # Backward: 1 -> 0
            flow_bck_np = self.dis.calc(g1, g0, None)
            
            
            # 3. Convert to Tensor (B=1, C=2, H, W)
            # device is already defined
            
            flow_fwd = torch.from_numpy(flow_fwd_np).permute(2, 0, 1).unsqueeze(0).to(device)
            flow_bck = torch.from_numpy(flow_bck_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 4. Forward-Backward Consistency Check
        # ... shared logic ...
        B, C, H, W = flow_fwd.shape
        # Create Grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(device)
        
        # Absolute coordinates to sample
        vgrid = grid + flow_fwd
        
        # Normalize to -1, 1 for grid_sample
        # 2x / (W-1) - 1
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1) # B, H, W, 2
        
        # Sample backward flow
        warped_flow_bck = F.grid_sample(flow_bck, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # FB Error: norm(flow_fwd + warped_flow_bck)
        # Ideally flow_fwd = - warpped_flow_bck, so sum should be 0
        fb_err = torch.norm(flow_fwd + warped_flow_bck, dim=1, keepdim=True)
        
        # 5. Photometric Error
        # Warp img1 to img0 using flow_fwd
        if isinstance(img1, torch.Tensor):
            img1_t = img1.clone().to(device)
            img0_t = img0.clone().to(device)
        else:
            # Assume already handled by to_gray if passed separately, but we need color tensors for photo error?
            # Actually, compute_flow_and_trust arg states they are tensors.
            # But earlier we handled "if tensor" to convert to numpy.
            # So they ARE tensors.
            img1_t = img1
            img0_t = img0

        # Sample img1 at vgrid
        warped_img1 = F.grid_sample(img1_t, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Photo Error: L1 dist per pixel (averaged over channels)
        photo_err = torch.abs(img0_t - warped_img1).mean(dim=1, keepdim=True)
        
        # 6. Generate Trust Mask
        # Heuristic from user: exp(-fb * 5) * exp(-photo * 10)
        mask_fb = torch.exp(-fb_err * 5.0)
        mask_photo = torch.exp(-photo_err * 10.0)
        
        trust_mask = mask_fb * mask_photo
        
        # Prepare Output Flow
        # IFRNet expects 4 channels: up_flow0 (t->0) and up_flow1 (t->1).
        # We assume t=0.5.
        # flow(t->0) approx 0.5 * flow(1->0) [flow_bck]
        # flow(t->1) approx 0.5 * flow(0->1) [flow_fwd]
        
        flow_out = torch.cat([0.5 * flow_bck, 0.5 * flow_fwd], dim=1)
        
        return flow_out, trust_mask
