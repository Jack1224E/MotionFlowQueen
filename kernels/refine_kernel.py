"""
Refinement Kernel: 3x3 Vector Median Filter (Triton)
Operates on the block grid (e.g. 80x45).
Replaces each vector with the vector median of its 3x3 neighborhood.
Cost gating: zeros out untrusted vectors (high cost or outlier magnitude).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def vector_median_kernel(
    dx_in_ptr, dy_in_ptr, cost_in_ptr,
    dx_out_ptr, dy_out_ptr,
    stride_b, stride_h, stride_w,
    GH: tl.constexpr, GW: tl.constexpr,
    COST_THRESH: tl.constexpr,
    BSX: tl.constexpr, BSY: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BSX + tl.arange(0, BSX)
    offs_y = pid_y * BSY + tl.arange(0, BSY)
    mask = (offs_x[None, :] < GW) & (offs_y[:, None] < GH)

    base = pid_b * stride_b

    # Neighbor offsets (3x3 including self)
    oy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    ox = [-1,  0,  1, -1, 0, 1, -1, 0, 1]

    # For each candidate, compute L1 distance sum to all 9 neighbors
    best_dist = tl.full((BSY, BSX), 999999, dtype=tl.int32)
    best_dx = tl.zeros((BSY, BSX), dtype=tl.int32)
    best_dy = tl.zeros((BSY, BSX), dtype=tl.int32)

    for ci in tl.static_range(9):
        # Load candidate vector
        cnx = tl.maximum(tl.minimum(offs_x[None, :] + ox[ci], GW - 1), 0)
        cny = tl.maximum(tl.minimum(offs_y[:, None] + oy[ci], GH - 1), 0)
        c_dx = tl.load(dx_in_ptr + base + cny * stride_h + cnx * stride_w, mask=mask, other=0)
        c_dy = tl.load(dy_in_ptr + base + cny * stride_h + cnx * stride_w, mask=mask, other=0)

        dist_sum = tl.zeros((BSY, BSX), dtype=tl.int32)

        for nj in tl.static_range(9):
            nnx = tl.maximum(tl.minimum(offs_x[None, :] + ox[nj], GW - 1), 0)
            nny = tl.maximum(tl.minimum(offs_y[:, None] + oy[nj], GH - 1), 0)
            n_dx = tl.load(dx_in_ptr + base + nny * stride_h + nnx * stride_w, mask=mask, other=0)
            n_dy = tl.load(dy_in_ptr + base + nny * stride_h + nnx * stride_w, mask=mask, other=0)
            dist_sum += tl.abs(c_dx - n_dx) + tl.abs(c_dy - n_dy)

        better = dist_sum < best_dist
        best_dx = tl.where(better, c_dx, best_dx)
        best_dy = tl.where(better, c_dy, best_dy)
        best_dist = tl.where(better, dist_sum, best_dist)

    # Cost gating: zero out vectors with high census cost
    cost_val = tl.load(cost_in_ptr + base + offs_y[:, None] * stride_h + offs_x[None, :] * stride_w,
                       mask=mask, other=0)
    untrusted = cost_val > COST_THRESH
    best_dx = tl.where(untrusted, 0, best_dx)
    best_dy = tl.where(untrusted, 0, best_dy)

    # Store
    out_ptrs = base + offs_y[:, None] * stride_h + offs_x[None, :] * stride_w
    tl.store(dx_out_ptr + out_ptrs, best_dx, mask=mask)
    tl.store(dy_out_ptr + out_ptrs, best_dy, mask=mask)


def run_vector_median(dx, dy, cost, cost_threshold=12):
    """
    Run 3x3 vector median filter on block-grid flow.
    Args: dx, dy: (B, GH, GW) int32; cost: (B, GH, GW) uint8/int32
    Returns: dx_out, dy_out: (B, GH, GW) int32
    """
    B, GH, GW = dx.shape
    dx_in = dx.to(torch.int32).contiguous()
    dy_in = dy.to(torch.int32).contiguous()
    cost_in = cost.to(torch.int32).contiguous()

    dx_out = torch.empty_like(dx_in)
    dy_out = torch.empty_like(dy_in)

    s_b, s_h, s_w = dx_in.stride()
    BSX, BSY = 16, 16
    grid = (triton.cdiv(GW, BSX), triton.cdiv(GH, BSY), B)

    vector_median_kernel[grid](
        dx_in, dy_in, cost_in, dx_out, dy_out,
        s_b, s_h, s_w,
        GH=GH, GW=GW,
        COST_THRESH=cost_threshold,
        BSX=BSX, BSY=BSY,
    )
    return dx_out, dy_out
