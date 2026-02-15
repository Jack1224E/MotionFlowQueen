Phase 1 Complete: The Census Transform is implemented and benchmarked.

Implementation Summary
Kernel: kernels/census_kernel.py uses a 32-bit signature with a sparse 7x7 sampling pattern.
Logic: 2D blocking (32x32), bitwise accumulation, and boundary masking.
Integration: core/bjf_patchmatch.py converts RGB -> Luma and calls the Triton kernel.
Performance:
Total Execution Time: ~3.67 ms (for two 1080p frames + overhead).
Kernel Estimate: ~0.85 ms per 1080p frame.
The system is now stable and ready for Phase 2: The Jump Flood Algorithm.