"""
Handcrafted Convex Upsampler V2 (HCU) — Triton Kernel
Tile-level launch: one program per coarse block.
Exp-free bilateral softmax with inverse SQUARED distance weighting.
Sub-block pixel offset makes edge pixels snap to their nearest neighbor.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def hcu_upsample_kernel(
    # Coarse grid inputs (GH × GW)
    dx_coarse_ptr, dy_coarse_ptr, cost_coarse_ptr, mean_luma_coarse_ptr,
    # High-res luma input (H × W)
    luma_hr_ptr,
    # High-res output (H × W)
    dx_out_ptr, dy_out_ptr,
    # Strides
    stride_c_b, stride_c_h, stride_c_w,   # coarse grid strides
    stride_hr_b, stride_hr_h, stride_hr_w, # HR image strides
    # Dimensions
    GH: tl.constexpr, GW: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BS: tl.constexpr,  # block size (8)
    # Tunables
    ALPHA: tl.constexpr,  # luma weight
    BETA: tl.constexpr,   # cost weight
    GAMMA: tl.constexpr,  # spatial distance weight
    EPS: tl.constexpr,    # stability epsilon
):
    pid_b = tl.program_id(2)
    cx = tl.program_id(0)
    cy = tl.program_id(1)

    c_base = pid_b * stride_c_b
    hr_base = pid_b * stride_hr_b

    # --- SRAM Cache: Load 3x3 coarse neighborhood ---
    # Neighbor order: 0=(-1,-1) 1=(-1,0) 2=(-1,+1) 3=(0,-1) 4=(0,0) 5=(0,+1)
    #                 6=(+1,-1) 7=(+1,0) 8=(+1,+1)

    ny0 = tl.maximum(cy - 1, 0); nx0 = tl.maximum(cx - 1, 0)
    ny1 = tl.maximum(cy - 1, 0); nx1 = cx
    ny2 = tl.maximum(cy - 1, 0); nx2 = tl.minimum(cx + 1, GW - 1)
    ny3 = cy;                     nx3 = tl.maximum(cx - 1, 0)
    ny4 = cy;                     nx4 = cx
    ny5 = cy;                     nx5 = tl.minimum(cx + 1, GW - 1)
    ny6 = tl.minimum(cy + 1, GH - 1); nx6 = tl.maximum(cx - 1, 0)
    ny7 = tl.minimum(cy + 1, GH - 1); nx7 = cx
    ny8 = tl.minimum(cy + 1, GH - 1); nx8 = tl.minimum(cx + 1, GW - 1)

    p0 = c_base + ny0 * stride_c_h + nx0 * stride_c_w
    p1 = c_base + ny1 * stride_c_h + nx1 * stride_c_w
    p2 = c_base + ny2 * stride_c_h + nx2 * stride_c_w
    p3 = c_base + ny3 * stride_c_h + nx3 * stride_c_w
    p4 = c_base + ny4 * stride_c_h + nx4 * stride_c_w
    p5 = c_base + ny5 * stride_c_h + nx5 * stride_c_w
    p6 = c_base + ny6 * stride_c_h + nx6 * stride_c_w
    p7 = c_base + ny7 * stride_c_h + nx7 * stride_c_w
    p8 = c_base + ny8 * stride_c_h + nx8 * stride_c_w

    dx0 = tl.load(dx_coarse_ptr + p0); dy0 = tl.load(dy_coarse_ptr + p0)
    dx1 = tl.load(dx_coarse_ptr + p1); dy1 = tl.load(dy_coarse_ptr + p1)
    dx2 = tl.load(dx_coarse_ptr + p2); dy2 = tl.load(dy_coarse_ptr + p2)
    dx3 = tl.load(dx_coarse_ptr + p3); dy3 = tl.load(dy_coarse_ptr + p3)
    dx4 = tl.load(dx_coarse_ptr + p4); dy4 = tl.load(dy_coarse_ptr + p4)
    dx5 = tl.load(dx_coarse_ptr + p5); dy5 = tl.load(dy_coarse_ptr + p5)
    dx6 = tl.load(dx_coarse_ptr + p6); dy6 = tl.load(dy_coarse_ptr + p6)
    dx7 = tl.load(dx_coarse_ptr + p7); dy7 = tl.load(dy_coarse_ptr + p7)
    dx8 = tl.load(dx_coarse_ptr + p8); dy8 = tl.load(dy_coarse_ptr + p8)

    c0_ = tl.load(cost_coarse_ptr + p0); l0 = tl.load(mean_luma_coarse_ptr + p0)
    c1_ = tl.load(cost_coarse_ptr + p1); l1 = tl.load(mean_luma_coarse_ptr + p1)
    c2_ = tl.load(cost_coarse_ptr + p2); l2 = tl.load(mean_luma_coarse_ptr + p2)
    c3_ = tl.load(cost_coarse_ptr + p3); l3 = tl.load(mean_luma_coarse_ptr + p3)
    c4_ = tl.load(cost_coarse_ptr + p4); l4 = tl.load(mean_luma_coarse_ptr + p4)
    c5_ = tl.load(cost_coarse_ptr + p5); l5 = tl.load(mean_luma_coarse_ptr + p5)
    c6_ = tl.load(cost_coarse_ptr + p6); l6 = tl.load(mean_luma_coarse_ptr + p6)
    c7_ = tl.load(cost_coarse_ptr + p7); l7 = tl.load(mean_luma_coarse_ptr + p7)
    c8_ = tl.load(cost_coarse_ptr + p8); l8 = tl.load(mean_luma_coarse_ptr + p8)

    # Normalized cost (scalar per neighbor)
    C0 = c0_ / 32.0; C1 = c1_ / 32.0; C2 = c2_ / 32.0
    C3 = c3_ / 32.0; C4 = c4_ / 32.0; C5 = c5_ / 32.0
    C6 = c6_ / 32.0; C7 = c7_ / 32.0; C8 = c8_ / 32.0

    # Neighbor center positions in fractional block units relative to this block center
    # block center = (cx + 0.5, cy + 0.5) in block coords
    # n0 center = (cx-1 + 0.5, cy-1 + 0.5) → offset (-1, -1)
    # Stored as (off_x, off_y) for each neighbor:
    # off_x: -1, 0, 1, -1, 0, 1, -1, 0, 1
    # off_y: -1,-1,-1,  0, 0, 0,  1, 1, 1

    # --- Inner loop: BS x BS HR pixels ---
    local_x = tl.arange(0, BS)  # [0..7]
    hr_x = cx * BS + local_x
    x_valid = hr_x < W

    # Sub-block fractional position: (lx + 0.5) / BS - 0.5 → [-0.4375, 0.4375] for BS=8
    # This encodes how far each HR pixel is from the block center
    fx = (local_x.to(tl.float32) + 0.5) / BS - 0.5  # (BS,)

    for ly in tl.static_range(BS):
        py = cy * BS + ly
        if py < H:
            fy = (ly + 0.5) / BS - 0.5  # scalar: sub-block y offset

            # Load HR luma row
            hr_ptrs = hr_base + py * stride_hr_h + hr_x * stride_hr_w
            hr_luma = tl.load(luma_hr_ptr + hr_ptrs, mask=x_valid, other=0.0)

            # Per-pixel spatial distance to each neighbor center
            # neighbor i at offset (ox_i, oy_i):
            # dist_i = sqrt((fx - ox_i)^2 + (fy - oy_i)^2)
            # We use squared distance directly for the quadratic kernel

            # dist² for each neighbor
            dsq0 = (fx - (-1.0)) * (fx - (-1.0)) + (fy - (-1.0)) * (fy - (-1.0))
            dsq1 = (fx -   0.0) * (fx -   0.0) + (fy - (-1.0)) * (fy - (-1.0))
            dsq2 = (fx -   1.0) * (fx -   1.0) + (fy - (-1.0)) * (fy - (-1.0))
            dsq3 = (fx - (-1.0)) * (fx - (-1.0)) + (fy -   0.0) * (fy -   0.0)
            dsq4 = (fx -   0.0) * (fx -   0.0) + (fy -   0.0) * (fy -   0.0)
            dsq5 = (fx -   1.0) * (fx -   1.0) + (fy -   0.0) * (fy -   0.0)
            dsq6 = (fx - (-1.0)) * (fx - (-1.0)) + (fy -   1.0) * (fy -   1.0)
            dsq7 = (fx -   0.0) * (fx -   0.0) + (fy -   1.0) * (fy -   1.0)
            dsq8 = (fx -   1.0) * (fx -   1.0) + (fy -   1.0) * (fy -   1.0)

            # Luma difference (squared for sharper edge)
            Lsq0 = (hr_luma - l0) * (hr_luma - l0)
            Lsq1 = (hr_luma - l1) * (hr_luma - l1)
            Lsq2 = (hr_luma - l2) * (hr_luma - l2)
            Lsq3 = (hr_luma - l3) * (hr_luma - l3)
            Lsq4 = (hr_luma - l4) * (hr_luma - l4)
            Lsq5 = (hr_luma - l5) * (hr_luma - l5)
            Lsq6 = (hr_luma - l6) * (hr_luma - l6)
            Lsq7 = (hr_luma - l7) * (hr_luma - l7)
            Lsq8 = (hr_luma - l8) * (hr_luma - l8)

            # Weight = 1 / (eps + α*L² + β*C² + γ*D²)
            # Quadratic terms → much sharper falloff near edges
            w0 = 1.0 / (EPS + ALPHA * Lsq0 + BETA * C0 * C0 + GAMMA * dsq0)
            w1 = 1.0 / (EPS + ALPHA * Lsq1 + BETA * C1 * C1 + GAMMA * dsq1)
            w2 = 1.0 / (EPS + ALPHA * Lsq2 + BETA * C2 * C2 + GAMMA * dsq2)
            w3 = 1.0 / (EPS + ALPHA * Lsq3 + BETA * C3 * C3 + GAMMA * dsq3)
            w4 = 1.0 / (EPS + ALPHA * Lsq4 + BETA * C4 * C4 + GAMMA * dsq4)
            w5 = 1.0 / (EPS + ALPHA * Lsq5 + BETA * C5 * C5 + GAMMA * dsq5)
            w6 = 1.0 / (EPS + ALPHA * Lsq6 + BETA * C6 * C6 + GAMMA * dsq6)
            w7 = 1.0 / (EPS + ALPHA * Lsq7 + BETA * C7 * C7 + GAMMA * dsq7)
            w8 = 1.0 / (EPS + ALPHA * Lsq8 + BETA * C8 * C8 + GAMMA * dsq8)

            # Normalize (convex combination)
            w_sum = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8
            inv_sum = 1.0 / w_sum
            w0 *= inv_sum; w1 *= inv_sum; w2 *= inv_sum
            w3 *= inv_sum; w4 *= inv_sum; w5 *= inv_sum
            w6 *= inv_sum; w7 *= inv_sum; w8 *= inv_sum

            # Blend flow
            out_dx = (w0*dx0 + w1*dx1 + w2*dx2 + w3*dx3 + w4*dx4 +
                      w5*dx5 + w6*dx6 + w7*dx7 + w8*dx8)
            out_dy = (w0*dy0 + w1*dy1 + w2*dy2 + w3*dy3 + w4*dy4 +
                      w5*dy5 + w6*dy6 + w7*dy7 + w8*dy8)

            # Store
            out_ptrs = hr_base + py * stride_hr_h + hr_x * stride_hr_w
            tl.store(dx_out_ptr + out_ptrs, out_dx, mask=x_valid)
            tl.store(dy_out_ptr + out_ptrs, out_dy, mask=x_valid)


def run_hcu_upsample(dx_coarse, dy_coarse, cost_coarse, mean_luma_coarse,
                     luma_hr, block_size=8,
                     alpha=40.0, beta=3.0, gamma=8.0, eps=1e-3):
    """
    HCU V2: Handcrafted Convex Upsampler.
    Tile-level Triton launch with exp-free bilateral softmax.
    Uses SQUARED terms for sharper edge-snapping.

    Args:
        dx_coarse, dy_coarse: (B, GH, GW) float32 — coarse flow in pixel units
        cost_coarse: (B, GH, GW) float32 — JFA cost
        mean_luma_coarse: (B, GH, GW) float32 — block mean luma [0, 1]
        luma_hr: (B, H, W) float32 — high-res luma [0, 1]
        block_size: int
        alpha, beta, gamma, eps: tunables

    Returns:
        dx_hr, dy_hr: (B, H, W) float32 — full-resolution flow
    """
    B, GH, GW = dx_coarse.shape
    _, H, W = luma_hr.shape
    device = dx_coarse.device

    dx_out = torch.empty(B, H, W, dtype=torch.float32, device=device)
    dy_out = torch.empty(B, H, W, dtype=torch.float32, device=device)

    dx_c = dx_coarse.contiguous().float()
    dy_c = dy_coarse.contiguous().float()
    cost_c = cost_coarse.contiguous().float()
    ml_c = mean_luma_coarse.contiguous().float()
    luma = luma_hr.contiguous().float()

    s_c = dx_c.stride()
    s_hr = luma.stride()

    grid = (GW, GH, B)

    hcu_upsample_kernel[grid](
        dx_c, dy_c, cost_c, ml_c,
        luma,
        dx_out, dy_out,
        s_c[0], s_c[1], s_c[2],
        s_hr[0], s_hr[1], s_hr[2],
        GH=GH, GW=GW,
        H=H, W=W,
        BS=block_size,
        ALPHA=alpha, BETA=beta, GAMMA=gamma, EPS=eps,
    )
    return dx_out, dy_out
