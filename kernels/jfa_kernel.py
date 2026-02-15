"""
Phase 2: PatchMatch-Accelerated Bounded JFA (V5.1 - Propagation Fix)

Root cause of V5 failure:
  - Init tested 8 compass neighbors at step=64 on a 45x80 grid -> ALL OOB
  - Every pixel stayed at seed (0,0), nothing to propagate
  
Fix:
  - Init: test ALL integer displacements in a small window (brute-force -R..+R)
  - JFA steps: cap step to max(GH, GW) // 2 automatically
  - PatchMatch random injection: larger random radius, every pass
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================
# Constants (Python-side only)
# ============================================================
MAX_JUMP = 64

# ============================================================
# Inline Helpers
# ============================================================

@triton.jit
def manual_popc(x):
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return (x * 0x01010101) >> 24

@triton.jit
def pack_state(dx, dy, cost):
    return (dx & 0x3FF) | ((dy & 0x3FF) << 10) | ((cost & 0x3F) << 20)

@triton.jit
def unpack_dx(packed):
    raw = packed & 0x3FF
    return raw - ((raw >> 9) & 1) * 1024

@triton.jit
def unpack_dy(packed):
    raw = (packed >> 10) & 0x3FF
    return raw - ((raw >> 9) & 1) * 1024

@triton.jit
def unpack_cost(packed):
    return (packed >> 20) & 0x3F


# ============================================================
# Kernel 1: Exhaustive Init
# Search all displacements in [-R, +R] x [-R, +R] window
# R = 8 -> searches 17x17 = 289 candidates per pixel
# On block grid: R=8 blocks = 64 pixel effective radius
# ============================================================

@triton.jit
def jfa_init_kernel(
    census1_ptr, census2_ptr, flow_out_ptr,
    stride_c_b, stride_c_h, stride_c_w,
    stride_f_b, stride_f_h, stride_f_w,
    GH: tl.constexpr, GW: tl.constexpr,
    SEARCH_R: tl.constexpr,  # Search radius in grid cells
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    mask = (offs_x[None, :] < GW) & (offs_y[:, None] < GH)

    # Load our census1 value
    src_ptrs = census1_ptr + pid_b * stride_c_b + offs_y[:, None] * stride_c_h + offs_x[None, :] * stride_c_w
    src_val = tl.load(src_ptrs, mask=mask, other=0)

    # Seed: (0, 0) displacement
    dst_ptrs = census2_ptr + pid_b * stride_c_b + offs_y[:, None] * stride_c_h + offs_x[None, :] * stride_c_w
    dst_val = tl.load(dst_ptrs, mask=mask, other=0)
    best_cost = manual_popc(src_val ^ dst_val)
    best_dx = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int32)
    best_dy = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int32)

    # Exhaustive search: test all (dx, dy) in [-R, +R]
    DIAM: tl.constexpr = 2 * SEARCH_R + 1
    for di in tl.static_range(DIAM * DIAM):
        cand_dy = (di // DIAM) - SEARCH_R
        cand_dx = (di % DIAM) - SEARCH_R

        # Skip (0,0) — already computed as seed
        # Triton doesn't support continue, so use conditional guard
        not_center = (cand_dx != 0) | (cand_dy != 0)

        tx = offs_x[None, :] + cand_dx
        ty = offs_y[:, None] + cand_dy
        valid = (tx >= 0) & (tx < GW) & (ty >= 0) & (ty < GH) & mask & not_center

        target_ptrs = census2_ptr + pid_b * stride_c_b + ty * stride_c_h + tx * stride_c_w
        target_val = tl.load(target_ptrs, mask=valid, other=0)
        cand_cost = manual_popc(src_val ^ target_val)
        cand_cost = tl.where(valid, cand_cost, 63)

        better = cand_cost < best_cost
        best_dx = tl.where(better, cand_dx, best_dx)
        best_dy = tl.where(better, cand_dy, best_dy)
        best_cost = tl.where(better, cand_cost, best_cost)

    packed = pack_state(best_dx, best_dy, best_cost)
    out_ptrs = flow_out_ptr + pid_b * stride_f_b + offs_y[:, None] * stride_f_h + offs_x[None, :] * stride_f_w
    tl.store(out_ptrs, packed, mask=mask)


# ============================================================
# Kernel 2: JFA Step (Propagation)
# ============================================================

@triton.jit
def jfa_step_kernel(
    census1_ptr, census2_ptr,
    flow_in_ptr, flow_out_ptr,
    stride_c_b, stride_c_h, stride_c_w,
    stride_f_b, stride_f_h, stride_f_w,
    GH: tl.constexpr, GW: tl.constexpr,
    STEP: tl.constexpr,
    PASS_IDX,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    mask = (offs_x[None, :] < GW) & (offs_y[:, None] < GH)

    # Load current best
    in_ptrs = flow_in_ptr + pid_b * stride_f_b + offs_y[:, None] * stride_f_h + offs_x[None, :] * stride_f_w
    cur_packed = tl.load(in_ptrs, mask=mask, other=0)
    best_dx = unpack_dx(cur_packed)
    best_dy = unpack_dy(cur_packed)
    best_cost = unpack_cost(cur_packed)

    # Load our census
    src_ptrs = census1_ptr + pid_b * stride_c_b + offs_y[:, None] * stride_c_h + offs_x[None, :] * stride_c_w
    src_val = tl.load(src_ptrs, mask=mask, other=0)

    # Predication: skip solved pixels
    needs_work = best_cost > 0

    # 9 neighbors: self + 8 compass at distance STEP
    # Including self re-evaluation ensures we don't lose quality
    for ni in tl.static_range(9):
        # Direction lookup (0=self, 1-8=compass)
        ndx = tl.where(ni == 0, 0,
              tl.where(ni == 1, 0,
              tl.where(ni == 2, STEP,
              tl.where(ni == 3, STEP,
              tl.where(ni == 4, STEP,
              tl.where(ni == 5, 0,
              tl.where(ni == 6, -STEP,
              tl.where(ni == 7, -STEP,
                       -STEP))))))))  # ni == 8

        ndy = tl.where(ni == 0, 0,
              tl.where(ni == 1, -STEP,
              tl.where(ni == 2, -STEP,
              tl.where(ni == 3, 0,
              tl.where(ni == 4, STEP,
              tl.where(ni == 5, STEP,
              tl.where(ni == 6, STEP,
              tl.where(ni == 7, 0,
                       -STEP))))))))  # ni == 8

        # Neighbor grid position
        nx = offs_x[None, :] + ndx
        ny = offs_y[:, None] + ndy
        nb_valid = (nx >= 0) & (nx < GW) & (ny >= 0) & (ny < GH) & mask & needs_work

        # Load neighbor's vector
        nb_ptrs = flow_in_ptr + pid_b * stride_f_b + ny * stride_f_h + nx * stride_f_w
        nb_packed = tl.load(nb_ptrs, mask=nb_valid, other=0)
        cand_dx = unpack_dx(nb_packed)
        cand_dy = unpack_dy(nb_packed)

        # Evaluate: what cost do WE get with neighbor's vector?
        tx = offs_x[None, :] + cand_dx
        ty = offs_y[:, None] + cand_dy
        target_valid = (tx >= 0) & (tx < GW) & (ty >= 0) & (ty < GH) & nb_valid

        target_ptrs = census2_ptr + pid_b * stride_c_b + ty * stride_c_h + tx * stride_c_w
        target_val = tl.load(target_ptrs, mask=target_valid, other=0)
        cand_cost = manual_popc(src_val ^ target_val)
        cand_cost = tl.where(target_valid, cand_cost, 63)

        better = cand_cost < best_cost
        best_dx = tl.where(better, cand_dx, best_dx)
        best_dy = tl.where(better, cand_dy, best_dy)
        best_cost = tl.where(better, cand_cost, best_cost)

    # Random injection (every pass, for unsolved pixels)
    needs_random = best_cost > 4
    seed = (offs_x[None, :] * 73856093) ^ (offs_y[:, None] * 19349669) ^ (PASS_IDX * 83492791)
    rand_dx = ((seed >> 0) & 0x1F) - 16  # Range [-16, 15]
    rand_dy = ((seed >> 5) & 0x1F) - 16

    rtx = offs_x[None, :] + rand_dx
    rty = offs_y[:, None] + rand_dy
    rand_valid = (rtx >= 0) & (rtx < GW) & (rty >= 0) & (rty < GH) & mask & needs_random

    rand_ptrs = census2_ptr + pid_b * stride_c_b + rty * stride_c_h + rtx * stride_c_w
    rand_val = tl.load(rand_ptrs, mask=rand_valid, other=0)
    rand_cost = manual_popc(src_val ^ rand_val)
    rand_cost = tl.where(rand_valid, rand_cost, 63)

    rand_better = rand_cost < best_cost
    best_dx = tl.where(rand_better, rand_dx, best_dx)
    best_dy = tl.where(rand_better, rand_dy, best_dy)
    best_cost = tl.where(rand_better, rand_cost, best_cost)

    # Store to pong buffer
    packed = pack_state(best_dx, best_dy, best_cost)
    out_ptrs = flow_out_ptr + pid_b * stride_f_b + offs_y[:, None] * stride_f_h + offs_x[None, :] * stride_f_w
    tl.store(out_ptrs, packed, mask=mask)


# ============================================================
# Python Orchestrator
# ============================================================

def run_jfa(census1, census2, flow_a, flow_b, confidence):
    B, GH, GW = census1.shape
    s_c_b, s_c_h, s_c_w = census1.stride()
    s_f_b, s_f_h, s_f_w = flow_a.stride()

    BSX, BSY = 16, 16
    grid = (triton.cdiv(GW, BSX), triton.cdiv(GH, BSY), B)

    # --- Init: Exhaustive search in [-8, +8] window ---
    # 8 blocks = 64px effective radius. 17x17 = 289 candidates.
    SEARCH_R = 8
    jfa_init_kernel[grid](
        census1, census2, flow_a,
        s_c_b, s_c_h, s_c_w,
        s_f_b, s_f_h, s_f_w,
        GH=GH, GW=GW,
        SEARCH_R=SEARCH_R,
        BLOCK_SIZE_X=BSX, BLOCK_SIZE_Y=BSY,
    )

    # --- JFA+1 Loop ---
    # Cap steps to grid dimensions
    max_step = max(GH, GW) // 2
    steps = []
    s = max_step
    while s >= 1:
        steps.append(s)
        s //= 2
    steps.append(1)  # JFA+1 correction pass

    src, dst = flow_a, flow_b
    for pass_idx, step in enumerate(steps):
        jfa_step_kernel[grid](
            census1, census2, src, dst,
            s_c_b, s_c_h, s_c_w,
            s_f_b, s_f_h, s_f_w,
            GH=GH, GW=GW,
            STEP=step,
            PASS_IDX=pass_idx,
            BLOCK_SIZE_X=BSX, BLOCK_SIZE_Y=BSY,
        )
        src, dst = dst, src

    final_flow = src

    # Unpack
    packed = final_flow
    raw_dx = (packed & 0x3FF).to(torch.int16)
    sign_dx = ((packed >> 9) & 1).to(torch.int16)
    dx = raw_dx - sign_dx * 1024

    raw_dy = ((packed >> 10) & 0x3FF).to(torch.int16)
    sign_dy = ((packed >> 19) & 1).to(torch.int16)
    dy = raw_dy - sign_dy * 1024

    cost = ((packed >> 20) & 0x3F).to(torch.uint8)
    confidence[:] = cost

    return dx, dy, confidence
