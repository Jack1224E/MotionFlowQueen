import torch
import torch.nn.functional as F

def process_nvofa_flow(raw_flow, scales=[1/8, 1/4, 1/2, 1.0]):
    """
    Process raw NVOFA flow for injection into IFRNet.
    
    Args:
        raw_flow: torch.Tensor (B, 2, H, W) - The flow from NVOFA (usually at full or 1/4 resolution)
        scales: list of float - Target scales for IFRNet (1/8, 1/4, 1/2, 1.0)
        
    Returns:
        flow_dict: dictionary mapping scale string (e.g. '1/8') to resized flow tensor
    """
    flow_dict = {}
    
    # Check if raw_flow is valid
    if raw_flow is None:
        return None

    # Get original dimensions
    B, C, H, W = raw_flow.shape
    
    for scale in scales:
        target_h = int(H * scale)
        target_w = int(W * scale)
        
        resize_mode = 'bilinear' if scale < 1 else 'bilinear' # consistent
        resized_flow = F.interpolate(raw_flow, size=(target_h, target_w), mode=resize_mode, align_corners=False)
        
        # Adjust flow magnitude for the scale
        magnitude_scale = scale
        resized_flow = resized_flow * magnitude_scale
        
        # Approximate intermediate flows for IFRNet (t=0.5)
        # IFRNet expects:
        # up_flow0: flow from t to 0 (approx -0.5 * flow_0->1)
        # up_flow1: flow from t to 1 (approx +0.5 * flow_0->1)
        # We start with flow_0->1 from NVOFA.
        
        # Note: raw_flow from NVOFA is 0->1.
        # flow_t->0 = -0.5 * flow_0->1
        # flow_t->1 = +0.5 * flow_0->1
        # However, IFRNet might refine these.
        
        flow_t0 = -0.5 * resized_flow
        flow_t1 = 0.5 * resized_flow
        
        # Concatenate to (B, 4, H, W)
        combined_flow = torch.cat([flow_t0, flow_t1], dim=1)
        
        key = f"{int(1/scale) if scale < 1 else 1}/{1 if scale >= 1 else int(1/scale)}"
        if scale == 1.0: key = "1/1"
        if scale == 0.125: key = "1/8" # specific catch for 1/8
        if scale == 0.25: key = "1/4"
        if scale == 0.5: key = "1/2"
        
        flow_dict[key] = combined_flow
        
    return flow_dict
