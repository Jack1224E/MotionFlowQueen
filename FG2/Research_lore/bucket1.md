# Unified Bucket 1: The Zero-Training Alpha Build

This document outlines the primary components for the initial "Zero-Training" implementation strategy, focusing on speed and latency minimization.

## 1. The Core Engine: IFRNet (S)
> **Goal:** Primary Synthesis Engine

Since the benchmark proved IFRNet outperforms RIFE in both speed and visual clarity, we are locking this in as the primary synthesis engine. It’s a single-pass "fused" architecture, which means it’s built for the exact "external motion injection" we want to do.

* 
## 2. The Speed Cheat: NVOFA (Hardware Flow)
> **Goal:** Zero-Overhead Motion Estimation

We are completely ripping out the internal motion estimation from the neural net and injecting NVIDIA’s dedicated optical flow hardware (NVOFA).

*   **The Play:** Use the `HighFPSViewer-NvOFFRUC` architecture as the reference. This gives us 400+ FPS flow without touching CUDA cores or maxing out the 4GB VRAM.

## 3. The Visual Fix: Softmax Splatting (CUDA)
> **Goal:** Artifact Reduction

To fix that "Vaseline smear" look we saw in the RIFE benchmark, we are swapping the clunky backward warping for Softmax Splatting.
## 2. The Speed Cheat: NVOFA (Hardware Flow)
> **Goal:** Zero-Overhead Motion Estimation

We are completely ripping out the internal motion estimation from the neural net and injecting NVIDIA’s dedicated optical flow hardware (NVOFA).

*   **The Play:** Use the `HighFPSViewer-NvOFFRUC` architecture as the reference. This gives us 400+ FPS flow without touching CUDA cores or maxing out the 4GB VRAM.

*   **The Play:** This is a forward-warping primitive that handles occlusions and "many-to-one" pixel collisions intelligently. We need to write this as a fused CUDA kernel/TensorRT plugin to keep it inside the 8ms latency budget.

## 4. The "Trust Mask" (Heuristic Filter)
> **Goal:** Quality Control / Garbage Filter

Before the neural net even sees the frames, we run a dirt-cheap consistency check in the terminal to flag garbage motion.

*   **The Play:** We compute the photometric error and divergence between the forward and backward NVOFA flows. If the error is high, we flag that region as "unreliable" so the decoder knows to synthesize from features instead of just stretching broken pixels.

## 5. The Zero-Copy Pipeline (No-Lag Plumbing)
> **Goal:** Latency Elimination

This is the multiplier that kills the "Python overhead" death spiral.

*   **The Play:** We use DXGI Desktop Duplication to capture the game frames directly into a CUDA array. We then map that pointer directly to TensorRT inputs with zero copies between the CPU and GPU. This bypasses the OOM crashes and RAM bottlenecks we hit earlier.

## 6. The Safety Net: FastFlowNet Fallback
> **Goal:** Robustness for Hard Cases

For frames where the hardware flow (NVOFA) hits a wall—like thin fences or extreme transparency—we need a backup.

*   **The Play:** We keep FastFlowNet on standby as a lightweight (~3-5ms) learned fallback. If the "Trust Mask" confidence is low, we fire up the FastFlowNet sub-module to fix the holes.
