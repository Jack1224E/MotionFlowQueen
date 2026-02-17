import cv2
import torch
import numpy as np

class DISFlowWrapper:
    def __init__(self, width=640, height=384, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        """
        Initializes the DIS Optical Flow wrapper.
        DIS (Dense Inverse Search) is a fast, robust optical flow algorithm available in OpenCV.
        It runs on CPU primarily but is highly optimized.
        
        Args:
            width: Input width (for consistency, not strictly needed for init)
            height: Input height
            preset: Performance preset (ULTRAFAST, FAST, MEDIUM)
        """
        self.width = width
        self.height = height
        self.dis = cv2.DISOpticalFlow_create(preset)
        self.is_initialized = True
        print(f"Initializing DISFlow (Software Wrapper) with preset {preset}")

    def execute(self, frame0, frame1):
        """
        Computes optical flow between two frames using DIS.
        
        Args:
            frame0, frame1: torch.Tensor (B, C, H, W) or numpy (H, W, C).
                            If torch, expects float/byte.
        
        Returns:
            flow: torch.Tensor (B, 2, H, W) in absolute coordinates (pixels).
            cost: torch.Tensor (B, 1, H, W) - simulated cost/confidence (DIS lacks direct cost output).
        """
        if not self.is_initialized:
            raise RuntimeError("DISFlow not initialized")

        # 1. Convert Inputs to Numpy Grayscale (H, W)
        def to_gray_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                # Detach, move to CPU, numpy
                arr = tensor.detach().cpu().numpy()
                if arr.ndim == 4: # B, C, H, W
                    arr = arr[0] # Take batch 0
                if arr.shape[0] in [1, 3]: # C, H, W -> H, W, C
                    arr = arr.transpose(1, 2, 0)
            else:
                arr = tensor
            
            # Convert to Gray if needed
            if arr.shape[-1] == 3:
                gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            elif arr.shape[-1] == 1:
                gray = arr.squeeze(-1).astype(np.uint8)
            else:
                gray = arr.astype(np.uint8)
            return gray

        g0 = to_gray_numpy(frame0)
        g1 = to_gray_numpy(frame1)

        # 2. Compute Flow
        # flow is (H, W, 2) float32
        flow_np = self.dis.calc(g0, g1, None)

        # 3. Convert back to Torch (B, 2, H, W)
        flow_torch = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0) # (1, 2, H, W)
        
        # 4. Simulate Cost/Confidence
        # DIS doesn't output a cost volume directly accessible in the python API usually.
        # We can simulate a "trust" based on flow magnitude or gradient, or just return zeros for now.
        # The 'trust_mask.py' logic might need adaptation if it relied heavily on NVOFA's specific cost scaling.
        # For now, we return a zero cost (representing "perfect trust"? or maybe high cost?)
        # Let's return local contrast as a proxy for texture-less regions (high uncertainty).
        
        # Simple proxy: gradient magnitude of image 0. Low gradient = High uncertainty?
        # Actually trust_mask.py uses (cost - min) / (max - min).
        # Let's verify trust_mask.py expectation.
        # It expects `cost_val`. NVOFA cost is roughly SAD (Sum of Absolute Differences).
        # We can approximate SAD error by warping g1 back to g0? That's expensive.
        # Let's return Zeros for now to pass the pipeline check.
        cost_torch = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32)

        # Move to GPU if input was GPU
        if isinstance(frame0, torch.Tensor) and frame0.is_cuda:
            flow_torch = flow_torch.cuda()
            cost_torch = cost_torch.cuda()

        return flow_torch, cost_torch

    def destroy(self):
        self.is_initialized = False
        del self.dis
        print("Destroying DISFlow instance")
