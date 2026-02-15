import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def census_kernel_v3(
    # Audited: All loads use mask and other=0.0 for safety.
    img_ptr,
    census_ptr,
    census_half_ptr,
    stride_img_b, stride_img_h, stride_img_w,
    stride_cens_b, stride_cens_h, stride_cens_w,
    stride_half_b, stride_half_h, stride_half_w,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
    HALO: tl.constexpr
):
    # Grid Z is Batch
    pid_b = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # 1. Coordinates & Pointers
    offs_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Base pointers
    img_base = img_ptr + pid_b * stride_img_b
    
    # Center pointers
    center_ptrs = img_base + (offs_y[:, None] * stride_img_h + offs_x[None, :] * stride_img_w)
    
    # 2. Masks (Strict Safety)
    # Global bounds check for Center
    mask_x_in = (offs_x >= 0) & (offs_x < W)
    mask_y_in = (offs_y >= 0) & (offs_y < H)
    mask_in = mask_x_in[None, :] & mask_y_in[:, None]
    
    # Halo Validity (Where 7x7 fits safely)
    mask_x_valid = (offs_x >= HALO) & (offs_x < W - HALO)
    mask_y_valid = (offs_y >= HALO) & (offs_y < H - HALO)
    mask_valid = mask_x_valid[None, :] & mask_y_valid[:, None]
    
    # Load Center (Float -> Int32 Quantization approx)
    # We essentially compare float values directly? NO.
    # The original kernel loaded floats and compared them.
    # Census is (val >= center). 
    # Float comparison is fine.
    
    center_val = tl.load(center_ptrs, mask=mask_in, other=0.0)
    
    # Accumulator
    census_sig = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int32)

    # 3. Full Resolution Census (32 Neighbors, Explicit Loads)
    offsets_y = [-1,-1,-1, 0, 0, 1, 1, 1,   -2,-2,-2,-2,-2, -1,-1, 0, 0, 1, 1, 2,2,2,2,2,   -3,-3,-3, 0, 0, 3, 3, 3]
    offsets_x = [-1, 0, 1,-1, 1,-1, 0, 1,   -2,-1, 0, 1, 2, -2, 2,-2, 2,-2, 2,-2,-1, 0, 1, 2,   -3, 0, 3,-3, 3,-3, 0, 3]
    
    for i in tl.static_range(32):
        dy = offsets_y[i]
        dx = offsets_x[i]
        
        # Shift Pointers
        nb_ptrs = center_ptrs + (dy * stride_img_h + dx * stride_img_w)
        
        # SAFETY: Mask with Valid Halo. If not valid, we don't load (avoids OOB).
        nb_val = tl.load(nb_ptrs, mask=mask_valid, other=0.0)
        
        bit = (nb_val >= center_val).to(tl.int32)
        census_sig = census_sig | (bit << i)

    # Sentinel for Borders
    SENTINEL = 0xFFFFFFFF
    final_census = tl.where(mask_valid, census_sig, SENTINEL)
    
    # Store Full Res
    census_out_base = census_ptr + pid_b * stride_cens_b
    out_ptrs = census_out_base + (offs_y[:, None] * stride_cens_h + offs_x[None, :] * stride_cens_w)
    
    # Store everywhere in block (masked by image bounds)
    tl.store(out_ptrs, final_census, mask=mask_in)

    # 4. Coarse Resolution Census (Half Res)
    is_even_x = (offs_x % 2 == 0)
    is_even_y = (offs_y % 2 == 0)
    is_active = is_even_x[None, :] & is_even_y[:, None]
    
    mask_x_valid_c = (offs_x >= 2) & (offs_x < W - 2)
    mask_y_valid_c = (offs_y >= 2) & (offs_y < H - 2)
    mask_valid_c = mask_x_valid_c[None, :] & mask_y_valid_c[:, None]
    
    mask_compute_c = is_active & mask_valid_c
    
    v01 = tl.load(center_ptrs + stride_img_w, mask=mask_compute_c, other=0.0)
    v10 = tl.load(center_ptrs + stride_img_h, mask=mask_compute_c, other=0.0)
    v11 = tl.load(center_ptrs + stride_img_h + stride_img_w, mask=mask_compute_c, other=0.0)
    
    avg_center = (center_val + v01 + v10 + v11) * 0.25
    
    coarse_sig = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int32)
    
    c_offsets_y = [-2,-2,-2, 0, 0, 2, 2, 2]
    c_offsets_x = [-2, 0, 2,-2, 2,-2, 0, 2]
    
    for k in tl.static_range(8):
        cdy = c_offsets_y[k]
        cdx = c_offsets_x[k]
        
        nb_ptr_c = center_ptrs + (cdy * stride_img_h + cdx * stride_img_w)
        nb_val = tl.load(nb_ptr_c, mask=mask_compute_c, other=0.0)
        
        bit_c = (nb_val >= avg_center).to(tl.int32)
        coarse_sig = coarse_sig | (bit_c << k)
        
    final_coarse = tl.where(mask_valid_c, coarse_sig, SENTINEL)
    
    half_base = census_half_ptr + pid_b * stride_half_b
    coarse_y = offs_y // 2
    coarse_x = offs_x // 2
    out_ptrs_h = half_base + (coarse_y[:, None] * stride_half_h + coarse_x[None, :] * stride_half_w)
    
    mask_h_bounds = (coarse_y[:, None] < (H // 2)) & (coarse_x[None, :] < (W // 2))
    mask_store_h = is_active & mask_h_bounds
    
    tl.store(out_ptrs_h, final_coarse, mask=mask_store_h)

def run_census(img_stack):
    B, H, W = img_stack.shape
    
    census_full = torch.empty((B, H, W), dtype=torch.int32, device=img_stack.device)
    census_half = torch.empty((B, H//2, W//2), dtype=torch.int32, device=img_stack.device)
    
    s_img_b, s_img_h, s_img_w = img_stack.stride()
    s_cens_b, s_cens_h, s_cens_w = census_full.stride()
    s_half_b, s_half_h, s_half_w = census_half.stride()
    
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32
    grid = (triton.cdiv(W, BLOCK_SIZE_X), triton.cdiv(H, BLOCK_SIZE_Y), B)
    
    census_kernel_v3[grid](
        img_stack,
        census_full,
        census_half,
        s_img_b, s_img_h, s_img_w,
        s_cens_b, s_cens_h, s_cens_w,
        s_half_b, s_half_h, s_half_w,
        H, W,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        HALO=3
    )
    
    return census_full, census_half
