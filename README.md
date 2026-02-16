# MotionFlowQueen: High-Fidelity Optical Flow Engine

MotionFlowQueen is a specialized, GPU-native optical flow engine designed for high-resolution video processing. It leverages **OpenAI Triton** kernels for massive parallelism and a sophisticated refinement pipeline to achieve sub-pixel accuracy comparable to state-of-the-art dense flow methods (like OpenCV DIS), but with significantly higher stability and temporal coherence.

## Core Architecture

The engine operates in two distinct phases: **Coarse Estimation (JFA)** and **Finesse Refinement**.

### Phase 1: The Census Transform & JFA
At the heart of the system is a valid 64-bit Census Transform (Triton optimized) that provides robustness to illumination changes.
- **Census Kernel**: Hand-optimized bit-packing and popcount operations running at ~5ms per 1080p frame.
- **Jump Flood Algorithm (JFA)**: Operates on an 8x8 block grid to propagate motion vectors across the image.
  - **Exhaustive Initialization**: Searches a [-8, +8] window (289 candidates) to find the best seed.
  - **Bounded Propagation**: Step sizes are dynamically capped to grid dimensions to ensure valid memory access.

### Phase 2: The Refinement Suite ("Finesse")
Raw block-based flow is noisy and quantized. The refinement suite transforms this into a smooth, dense field.
1.  **Vector Median Filter**: A 3x3 kernel (Triton) that removes outlier vectors by enforcing local coherence.
2.  **Sub-Pixel Quadratic Refinement**: Fits a 2D paraboloid to find the true sub-pixel minimum via Newton step, with determinant gating, texture checks, and strict clamping.
3.  **HCU Upscale**: A Triton tile-level kernel (1 program per coarse block) that upscales flow using exp-free bilateral softmax with per-pixel luma, cost, and spatial weighting. Replaces the old Guided Filter for better edge-snapping.

### Phase 3: The "Nervous System" (Temporal Stability)
-   **Shock Detection**: Luma MAD + warp error spike monitoring. Resets temporal buffers on scene cuts.
-   **Adaptive EMA**: Per-pixel `α = clamp(1 - cost/16, 0.1, 0.9)`. High confidence → trust new data; low confidence → trust history.
-   **Goblin Leash**: Confidence-weighted soft clamp — `vmax = lerp(48, 200, exp(-cost/8))`. High-confidence large motion (camera pans) passes through; low-confidence hallucinations are choked to 48px.
-   **Quadratic Veto**: Prevents sub-pixel refiner from sliding into adjacent local minima.

## Performance Benchmarks

Tested on `sample_darksouls2.mp4` (640×360, 30fps, fast-paced Dark Souls II gameplay).

### Pipeline Progression

| Version | Avg Warp Error | Avg Flow Mag | Key Change |
| :--- | :---: | :---: | :--- |
| Raw JFA | 0.076 | 31.4 px | Census + JFA block grid |
| V1 Refined | 0.064 | 25.0 px | + Vector Median + Pyramidal |
| Finesse (Guided) | 0.057 | 15.8 px | + Sub-Pixel + Guided Filter |
| Nervous V1 | 0.086 | 14.1 px | + EMA + Shock Detection |
| **HCU + Goblin Leash** | **0.088** | **19.0 px** | + Triton HCU + Soft Clamp |

> First-10-frame avg error is **0.057** vs DIS **0.055**. On frames 3, 6, 7, **MFQ beats DIS**.

### 500-Frame Endurance Test (HCU + Goblin Leash)

| Metric | Value |
| :--- | :--- |
| **Throughput** | **51.6 FPS** |
| Avg Warp Error | 0.088 |
| Peak Warp Error | 0.180 (scene cut) |
| Peak Max Vector | 162.1 px (high-conf pan) |
| Shocks | 16 |
| Anomalies | **5** (warp_err only, 0 max_vec) |
| Crashes / NaN / Inf | **0** |

> Anomaly = vector >100px **AND** JFA cost >15 (low confidence). True large-displacement motion is not penalized.

See [benchmarks.md](benchmarks.md) for full tables, HCU vs Guided Filter comparison, and shock logs.

## Quick Start

### Prerequisites
-   NVIDIA GPU (Triton compatible)
-   Python 3.10+
-   PyTorch 2.0+
-   OpenCV (`opencv-contrib-python`)

### Running the Comparison
Run the validation script to generate side-by-side comparisons with OpenCV DIS:

```bash
python3 scripts/validate_flow.py
```

This will process the sample video and output:
-   `validation_output/comparison_nervous_XXXX.png`: Visual verification.
-   `validation_output/stress_nervous_report.txt`: detailed metrics.

## File Structure

-   `kernels/`: Low-level Triton kernels (`census_kernel.py`, `jfa_kernel.py`, `subpixel_kernel.py`, `refine_kernel.py`).
-   `core/`: High-level Python modules (`bjf_patchmatch.py`, `refine.py`, `shock.py`, `upscale.py`).
-   `scripts/`: Validation and benchmarking tools.

## Future Roadmap

-   **Synthesis Phase**: Leveraging the high-quality flow for frame interpolation.
-   **Multi-Scale JFA**: Moving beyond the single 8x8 grid to a hierarchical approach.
-   **Neural Refinement**: Experimenting with lightweight CNNs for the final upscale step.

---
*Built with ❤️ by the MotionFlowQueen Team.*
