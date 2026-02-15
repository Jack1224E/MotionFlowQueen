"""
Sub-Pixel Quadratic Refinement (Triton)
For each block on the grid, evaluates Census cost at 9 neighbor displacements,
fits a 2D quadratic surface, and finds the sub-pixel minimum.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _popc(x):
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return (x * 0x01010101) >> 24


@triton.jit
def subpixel_refine_kernel(
    dx_in_ptr, dy_in_ptr, cost_in_ptr,
    census1_ptr, census2_ptr,
    dx_out_ptr, dy_out_ptr,
    stride_f_b, stride_f_h, stride_f_w,
    stride_c_b, stride_c_h, stride_c_w,
    GH: tl.constexpr, GW: tl.constexpr,
    MAX_COST: tl.constexpr,  # Only refine blocks with cost <= this
    BSX: tl.constexpr, BSY: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BSX + tl.arange(0, BSX)
    offs_y = pid_y * BSY + tl.arange(0, BSY)
    mask = (offs_x[None, :] < GW) & (offs_y[:, None] < GH)

    f_base = pid_b * stride_f_b
    c_base = pid_b * stride_c_b
    f_ptrs = f_base + offs_y[:, None] * stride_f_h + offs_x[None, :] * stride_f_w

    # Load current integer displacement and cost
    cur_dx = tl.load(dx_in_ptr + f_ptrs, mask=mask, other=0)
    cur_dy = tl.load(dy_in_ptr + f_ptrs, mask=mask, other=0)
    cur_cost = tl.load(cost_in_ptr + f_ptrs, mask=mask, other=99)

    # Load our census1 value
    c1_ptrs = c_base + offs_y[:, None] * stride_c_h + offs_x[None, :] * stride_c_w
    src_val = tl.load(census1_ptr + c1_ptrs, mask=mask, other=0)

    # Evaluate census cost at 9 displacement offsets around (cur_dx, cur_dy)
    # Offsets: (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    oy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    ox = [-1,  0,  1, -1, 0, 1, -1, 0, 1]

    # We store 5 key costs for the parabolic fit
    # c_m0 = cost(dx-1, dy), c_p0 = cost(dx+1, dy)
    # c_0m = cost(dx, dy-1), c_0p = cost(dx, dy+1)
    # c_00 = cost(dx, dy) -> center
    # Plus diagonals for cross-term
    c_mm = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_m0 = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_mp = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_0m = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_00 = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_0p = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_pm = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_p0 = tl.zeros((BSY, BSX), dtype=tl.int32)
    c_pp = tl.zeros((BSY, BSX), dtype=tl.int32)

    for ni in tl.static_range(9):
        ddx = ox[ni]
        ddy = oy[ni]

        # Target in census2
        tx = offs_x[None, :] + cur_dx + ddx
        ty = offs_y[:, None] + cur_dy + ddy
        valid = (tx >= 0) & (tx < GW) & (ty >= 0) & (ty < GH) & mask

        target_ptrs = c_base + ty * stride_c_h + tx * stride_c_w
        dst_val = tl.load(census2_ptr + target_ptrs, mask=valid, other=0)
        cost_val = _popc(src_val ^ dst_val)
        cost_val = tl.where(valid, cost_val, 63)

        # Store to the right slot
        c_mm = tl.where(ni == 0, cost_val, c_mm)
        c_m0 = tl.where(ni == 1, cost_val, c_m0)
        c_mp = tl.where(ni == 2, cost_val, c_mp)
        c_0m = tl.where(ni == 3, cost_val, c_0m)
        c_00 = tl.where(ni == 4, cost_val, c_00)
        c_0p = tl.where(ni == 5, cost_val, c_0p)
        c_pm = tl.where(ni == 6, cost_val, c_pm)
        c_p0 = tl.where(ni == 7, cost_val, c_p0)
        c_pp = tl.where(ni == 8, cost_val, c_pp)

    # --- Guardrail 1: Texture / Ambiguity Check ---
    # If the cost surface is flat (max - min < 2), don't refine
    cost_max = tl.maximum(tl.maximum(tl.maximum(tl.maximum(c_mm, c_m0), tl.maximum(c_mp, c_0m)),
                          tl.maximum(tl.maximum(c_00, c_0p), tl.maximum(c_pm, c_p0))), c_pp)
    cost_min = tl.minimum(tl.minimum(tl.minimum(tl.minimum(c_mm, c_m0), tl.minimum(c_mp, c_0m)),
                          tl.minimum(tl.minimum(c_00, c_0p), tl.minimum(c_pm, c_p0))), c_pp)
    has_texture = (cost_max - cost_min) >= 2

    # --- Guardrail 4: Local Minimum Veto ---
    # Only refine if center (c_00) is the minimum of all 9 costs.
    # If it's NOT the minimum, the parabola would slide to an adjacent minimum — skip.
    center_is_min = (c_00 <= c_mm) & (c_00 <= c_m0) & (c_00 <= c_mp) & \
                    (c_00 <= c_0m) & (c_00 <= c_0p) & \
                    (c_00 <= c_pm) & (c_00 <= c_p0) & (c_00 <= c_pp)

    # 2D Quadratic Surface Fit
    a_xx = (c_m0 - 2 * c_00 + c_p0).to(tl.float32)
    a_yy = (c_0m - 2 * c_00 + c_0p).to(tl.float32)
    a_xy = ((c_pp - c_pm - c_mp + c_mm).to(tl.float32)) * 0.25
    g_x = ((c_p0 - c_m0).to(tl.float32)) * 0.5
    g_y = ((c_0p - c_0m).to(tl.float32)) * 0.5

    det = a_xx * a_yy - a_xy * a_xy

    # --- Guardrail 2: Determinant Gating (Hessian singularity check) ---
    # Convex, non-singular, low cost, textured, AND center is the local minimum
    refinable = (det > 1e-6) & (a_xx > 1e-6) & (cur_cost <= MAX_COST) & \
                has_texture & center_is_min & mask

    # Newton step to minimum (safe: det > 1e-6 guaranteed by mask)
    safe_det = tl.where(refinable, det, 1.0)  # avoid div-by-zero
    delta_x = tl.where(refinable, -(a_yy * g_x - a_xy * g_y) / safe_det, 0.0)
    delta_y = tl.where(refinable, -(a_xx * g_y - a_xy * g_x) / safe_det, 0.0)

    # --- Guardrail 3: Strict Clamping to ±0.5 (no teleporting sub-pixels) ---
    delta_x = tl.maximum(tl.minimum(delta_x, 0.5), -0.5)
    delta_y = tl.maximum(tl.minimum(delta_y, 0.5), -0.5)

    # Output: float displacement with sub-pixel offset
    out_dx = cur_dx.to(tl.float32) + delta_x
    out_dy = cur_dy.to(tl.float32) + delta_y

    tl.store(dx_out_ptr + f_ptrs, out_dx, mask=mask)
    tl.store(dy_out_ptr + f_ptrs, out_dy, mask=mask)


def run_subpixel_refine(dx_int, dy_int, cost, census1_grid, census2_grid, max_cost=8):
    """
    Sub-pixel quadratic refinement on block grid.
    Args:
        dx_int, dy_int: (B, GH, GW) int32
        cost: (B, GH, GW) int32/uint8
        census1_grid, census2_grid: (B, GH, GW) int32
        max_cost: only refine blocks with cost <= max_cost
    Returns:
        dx_sub, dy_sub: (B, GH, GW) float32
    """
    B, GH, GW = dx_int.shape
    dx_in = dx_int.to(torch.int32).contiguous()
    dy_in = dy_int.to(torch.int32).contiguous()
    cost_in = cost.to(torch.int32).contiguous()

    dx_out = torch.empty(B, GH, GW, dtype=torch.float32, device=dx_in.device)
    dy_out = torch.empty(B, GH, GW, dtype=torch.float32, device=dx_in.device)

    s_f_b, s_f_h, s_f_w = dx_in.stride()
    s_c_b, s_c_h, s_c_w = census1_grid.stride()

    BSX, BSY = 16, 16
    grid = (triton.cdiv(GW, BSX), triton.cdiv(GH, BSY), B)

    subpixel_refine_kernel[grid](
        dx_in, dy_in, cost_in,
        census1_grid, census2_grid,
        dx_out, dy_out,
        s_f_b, s_f_h, s_f_w,
        s_c_b, s_c_h, s_c_w,
        GH=GH, GW=GW,
        MAX_COST=max_cost,
        BSX=BSX, BSY=BSY,
    )
    return dx_out, dy_out
