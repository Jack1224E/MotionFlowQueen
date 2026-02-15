1. The "Memory System Savior" (Shared Memory + Padding)

    The Refinement: We aren't just doing a halo load; we’re using the 32x33 Bank Conflict Trick.

    Why it's better: My previous plan didn't explicitly specify the 33rd column. On the GTX 1650, if 32 threads in a warp access a 32-wide block, they hit the same shared memory bank, causing a "serialized stall"—meaning the GPU waits for 32 separate cycles instead of 1.

    The Implementation:

        SRAM Tiling: Fetch a 38x38 "Halo" tile (32 block + 3px radius on each side) into a tl.zeros((38, 39), dtype=tl.float32) buffer.

        The Padding: That 39 is the magic number. It offsets every row by 1 bank, so the column-wise access is conflict-free.

2. Killing the "Python Tax" (Grid Z Dimension)

    The Refinement: Ditch the Python loop for Frame 1 and Frame 2.

    Why it's better: Right now, run_census launches two separate kernels. Each kernel launch from Python has a ~0.1ms overhead. If we want to stay under 2ms total, we can't waste 10% of our budget just talking to the CPU.

    The Implementation:

        Stack the frames as (2, H, W).

        Use pid_z = tl.program_id(2) to identify if we are processing the current or previous frame.

        This makes it a single hardware dispatch, cutting our "total time" immediately.

3. The "Border Sentinel" (Deterministic Behavior)

    The Refinement: Use other=0 and a BORDER_SENTINEL.

    Why it's better: My previous skeleton had "undefined-ish" borders. If the Jump-Flood algorithm in Phase 2 sees random noise at the screen edges, it will try to "match" it, creating flickering trash at the borders.

    The Implementation:

        Use tl.load(..., other=0) so out-of-bounds pixels are always black luma.

        For the census signature, if a pixel is in the 3px border, hard-code the output to 0xFFFFFFFF. This tells the matcher: "I am a border, do not try to find a motion vector for me."

4. Register Control (The "Don't Spill" Rule)

    The Refinement: Use tl.max_contiguous and tl.multiple_of hints.

    Why it's better: Triton is smart, but Turing's TU117 is register-starved. If we don't give the compiler "trust me bro" hints, it will generate slow code that checks every single pixel's alignment.

    The Implementation:

        Explicitly tell Triton our memory is a multiple_of(16).

        This allows the compiler to use LDG.E.128 instructions—fetching 4 pixels at a time into registers—which is the absolute peak speed for your 1650.

5. Multi-Scale Seeding (Half-Res Census)

    The Refinement: Compute Census at 1/2 resolution concurrently.

    Why it's better: 32-bit Census at 1080p only "sees" a 7x7 area. If a character in Elden Ring jumps 40 pixels, the full-res Census will be blind. A half-res map sees double the distance for the same compute cost.

    The Implementation:

        During the same Census kernel, while the luma tile is in shared memory, run a 2x2 box filter and compute a second, coarse Census map.