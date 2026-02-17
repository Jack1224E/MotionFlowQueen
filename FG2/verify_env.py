import sys
import os
import time

print("Checking environment...")
print(f"Python: {sys.version}")

# Add repo paths explicitly
sys.path.append(os.path.abspath('repo_rife'))
sys.path.append(os.path.abspath('repo_ifrnet'))

try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Torch Error: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV Error: {e}")

try:
    from model.RIFE_HDv3 import Model as RIFEModel
    print("RIFE_HDv3 import: SUCCESS")
except ImportError as e:
    print(f"RIFE_HDv3 import Error: {e}")

try:
    from models.IFRNet_S import Model as IFRNetModel
    print("IFRNet_S import: SUCCESS")
except ImportError as e:
    print(f"IFRNet_S import Error: {e}")

print("Environment check complete.")
