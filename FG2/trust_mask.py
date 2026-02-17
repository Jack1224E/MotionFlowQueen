import torch
import torch.nn.functional as F

class TrustMask:
    def __init__(self, use_heuristic=True):
        self.use_heuristic = use_heuristic

    def heuristic_trust(self, nvofa_cost, flow_consistency):
        """
        Zero-shot trust computation
        nvofa_cost: NVOFA cost buffer (0-255, lower=better). Shape (B, 1, H, W)
        flow_consistency: |F_fwd + F_bwd|. Shape (B, 2, H, W) -> magnitude (B, 1, H, W)
        """
        # Normalize cost to confidence (0 to 1, where 1 is trustworthy)
        # NVOFA cost is typically 0-255.
        cost_conf = 1.0 - (nvofa_cost / 255.0)
        cost_conf = torch.clamp(cost_conf, 0, 1)

        # Consistency confidence (exponential falloff)
        # If forward + backward flow is 0, consistency is perfect.
        consistency_mag = torch.norm(flow_consistency, dim=1, keepdim=True)
        consist_conf = torch.exp(-consistency_mag * 10) # 10 is a strictness factor
        
        # Combined trust (geometric mean)
        trust = torch.sqrt(cost_conf * consist_conf)
        return trust

    def compute(self, cost_fwd, cost_bwd, flow_fwd, flow_bwd):
        """
        Compute trust mask from forward and backward flow/cost.
        Note: requires warping backward flow to current frame coordinates 
        to properly check consistency, or just simple magnitude check if simplified.
        
        For a true forward-backward check: 
        P_t + V_fwd(P_t) = P_{t+1}
        P_{t+1} + V_bwd(P_{t+1}) = P_t
        So V_fwd(P_t) + V_bwd(P_t + V_fwd(P_t)) should be 0.
        
        Here we will use a simpler approximation if full warp is too expensive:
        Just checking raw cost + divergence magnitude if available.
        """
        # For this Alpha Build, we will trust the provided cost primarily 
        # as calculating full consistency requires warping which adds latency.
        # We will assume flow_consistency is provided or we can compute it if we have the warp function.
        
        # Placeholder for full consistency check:
        # warped_bwd = warp(flow_bwd, flow_fwd)
        # diff = flow_fwd + warped_bwd
        
        # Using just cost for speed in this initial step, combined with a dummy consistency = 0
        consistency_dummy = torch.zeros_like(cost_fwd) 
        
        # We process forward and backward trust separately or combined?
        # Usually we want trust for the specific pixel being interpolated.
        
        trust = self.heuristic_trust(cost_fwd, consistency_dummy)
        return trust
