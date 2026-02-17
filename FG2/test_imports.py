import sys
print("Starting imports...", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print("Torch imported.", flush=True)
except Exception as e:
    print(f"Torch failed: {e}", flush=True)

try:
    print("Importing cv2...", flush=True)
    import cv2
    print("CV2 imported.", flush=True)
except Exception as e:
    print(f"CV2 failed: {e}", flush=True)

try:
    print("Importing nvofa_wrapper...", flush=True)
    from nvofa_wrapper import NvOFA
    print("NvOFA imported.", flush=True)
except Exception as e:
    print(f"NvOFA failed: {e}", flush=True)

try:
    print("Instantiating NvOFA...", flush=True)
    nvofa = NvOFA()
    print("NvOFA instantiated.", flush=True)
except Exception as e:
    print(f"NvOFA init failed: {e}", flush=True)
