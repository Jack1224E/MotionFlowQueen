# Phase 1: Technical Audit - Census Transform

## 1. Memory Layout & Tensor Shapes
The data flow from Python to Triton is managed as follows:

- **Input Tensor (`img_tensor`)**:
    - **Shape**: `(H, W)` (Single channel 2D tensor).
    - **Type**: `float32` (as per benchmark generation) or `uint8` (implicitly supported by Triton load, but benchmark uses float).
    - **Memory Layout**: Enforced as **contiguous** via `.contiguous()` in `core/bjf_patchmatch.py`.
    - **Strides**: Passed explicitly to the kernel (`stride_img_h`, `stride_img_w`).

- **Output Tensor (`census_tensor`)**:
    - **Shape**: `(H, W)`.
    - **Type**: `int32` (32-bit Census Signature).
    - **Memory Layout**: Contiguous.

**Note**: The current implementation loops over the batch dimension in Python (`for b in range(B)`), launching a separate grid for each image in the batch.

## 2. Spatial Sampling Pattern (32-bit Signature)
The Census Transform calculates a 32-bit signature by comparing the center pixel to 32 surrounding neighbors. The sampling pattern is a **Sparse 7x7 Grid**, constructed as concentric rings:

- **Radius 1 (Inner Ring)**: 8 pixels.
    - Offsets: `(-1,-1)` to `(1,1)` (full 3x3 box excluding center).
    - Bits: 0-7.
- **Radius 2 (Middle Ring)**: 16 pixels.
    - Offsets: `(-2,-2)` to `(2,2)` (full 5x5 box excluding inner 3x3).
    - Bits: 8-23.
- **Radius 3 (Outer Ring)**: 8 pixels (Sparse).
    - Offsets: `(-3,-3), (-3,0), (-3,3), (0,-3), (0,3), (3,-3), (3,0), (3,3)`.
    - Bits: 24-31.

This pattern captures local texture (Radius 1 & 2) while providing wider context (Radius 3) without the computational cost of a full 7x7 block (48 neighbors).

## 3. Triton Block Sizing & Grid Launch
- **Block Size**: `BLOCK_SIZE_X = 32`, `BLOCK_SIZE_Y = 32`.
    - Each thread block processes a 32x32 pixel tile (1024 threads/pixels).
- **Grid Dimensions**:
    - `grid_x = ceil(W / 32)`
    - `grid_y = ceil(H / 32)`
    - This ensures full coverage of the image.

## 4. Bitwise Accumulation Logic
The kernel uses unrolled bitwise operations to construct the `int32` signature. For each neighbor $i \in [0, 31]$:

```python
# Pseudo-code logic inside kernel
bit = (neighbor_val >= center_val)
signature |= (bit << i)
```
- **Initialization**: `census_sig` is initialized to 0.
- **Comparison**: Strictly `>=`.
- **Accumulation**: `OR` operation with left-shifted bit index.

## 5. Memory Alignment & Overhead
- **Contiguity Enforcement**: `luma1.contiguous()` in the Python wrapper ensures the tensor is packed in memory. This adds a small copy overhead if the input (e.g., from a sliced tensor) is not already contiguous.
- **Masked Loads**: All memory loads use `tl.load(..., mask=mask)`.
    - **Pros**: Perfectly safe against out-of-bounds access.
    - **Cons**: Slightly slower than vectorized block loads without masking, but required here due to the stencil nature and arbitrary image sizes.
- **Batch Overhead**: Python-side looping for batches (`for b in range(B)`) introduces CUDA launch overhead per image. For small batches (e.g., typical real-time inference of 1-2 frames), this is negligible.

## 6. Padding & Border Handling
**Strategy**: Virtual Padding (Masking).
- No physical padding is applied to the input image (zero-copy).
- **Mask Logic**:
  ```python
  mask_x = (offs_x >= 3) & (offs_x < W - 3)
  mask_y = (offs_y >= 3) & (offs_y < H - 3)
  mask = mask_x & mask_y
  ```
- **Effect**:
    - Pixels within the 3-pixel border of the image have their `mask` set to `False`.
    - `tl.store` is masked, so these border pixels in the output `census_tensor` will remain **0** (or uninitialized if `empty` was used, but `tl.store` simply skips them).
    - **Safety**: Prevents the stencil from reading invalid memory addresses outside the image buffer.

## Architecture Critique / Next Steps
- **Optimization**: The explicit 32 loads per pixel are memory bandwidth intensive. For a 32x32 block, many neighbors are shared. Using `tl.load` on a larger block (e.g., 38x38) into Shared Memory (SRAM) and then computing the census from SRAM could significantly reduce global memory traffic (L2 cache hits dependent).
- **Batching**: Future optimization should move the batch dimension (N) into the Triton grid (z-axis) to remove Python loop overhead.
