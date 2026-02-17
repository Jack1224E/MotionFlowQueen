#!/home/jack/Documents/FG2/venv/bin/python3
import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
sys.path.append(os.path.abspath('repo_rife'))
sys.path.append(os.path.abspath('repo_ifrnet'))

# RIFE Import
# We use the specific HDv3 model file we copied/verified
try:
    from model.RIFE_HDv3 import Model as RIFEModel
    print("Successfully imported RIFE_HDv3")
except ImportError:
    try:
        from model.RIFE import Model as RIFEModel
        print("Imported standard RIFE Model")
    except ImportError as e:
        print(f"Failed to import RIFE: {e}")
        RIFEModel = None

# IFRNet Import
try:
    from models.IFRNet_S import Model as IFRNetModel
    print("Successfully imported IFRNet_S")
except ImportError as e:
    print(f"Failed to import IFRNet: {e}")
    IFRNetModel = None

# Force CPU to avoid issues on this machine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def load_rife():
    if RIFEModel is None: 
        return None
    
    print("Loading RIFE model...")
    try:
        model = RIFEModel()
        model.device()
        model.eval()
    except Exception as e:
        print(f"Error initializing RIFE model: {e}")
        return None

    path = 'repo_rife/train_log'
    # RIFE load_model expects the directory, not the file
    if os.path.exists(os.path.join(path, 'flownet.pkl')):
        try:
            model.load_model(path)
            print("RIFE weights loaded.")
        except Exception as e:
            print(f"Error loading RIFE weights: {e}")
            return None
    else:
        print(f"Warning: RIFE weights not found at {path}")
        return None
        
    return model

def load_ifrnet():
    if IFRNetModel is None: return None
    print("Loading IFRNet model...")
    model = IFRNetModel()
    model.to(device)
    model.eval()
    # IFRNet weights path
    path = 'repo_ifrnet/checkpoints/IFRNet_S.pth'
    if os.path.exists(path):
        try:
            # Check if loading needs strict=False or specific key adjustments
            model.load_state_dict(torch.load(path, map_location=device))
            print("IFRNet model loaded.")
        except Exception as e:
            print(f"Error loading IFRNet weights: {e}")
            return None
    else:
        print(f"Warning: IFRNet weights not found at {path}")
        return None
    return model

def run_rife(model, img0, img1):
    # Run RIFE inference
    # RIFE requires dimensions to be multiples of 32.
    # We pad the input images.
    h, w, _ = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, 0, pw - w, ph - h)
    
    img0_torch = torch.tensor(img0).permute(2, 0, 1).float() / 255.0
    img1_torch = torch.tensor(img1).permute(2, 0, 1).float() / 255.0
    img0_torch = img0_torch.to(device).unsqueeze(0)
    img1_torch = img1_torch.to(device).unsqueeze(0)
    
    # import torch.nn.functional as F # Already imported globally
    img0_pad = F.pad(img0_torch, padding)
    img1_pad = F.pad(img1_torch, padding)
    
    start = time.time()
    res = model.inference(img0_pad, img1_pad)
    end = time.time()
    
    # Postprocess (crop back)
    res = res[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    res = res[:h, :w]
    res = res.astype('uint8')
    return res, end - start

def run_ifrnet(model, img0, img1):
    # IFRNet inference
    # IFRNet also typically needs padding or specific resize.
    # We'll apply same padding just in case.
    h, w, _ = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, 0, pw - w, ph - h)
    
    img0_torch = torch.tensor(img0).permute(2, 0, 1).float() / 255.0
    img1_torch = torch.tensor(img1).permute(2, 0, 1).float() / 255.0
    img0_torch = img0_torch.to(device).unsqueeze(0)
    img1_torch = img1_torch.to(device).unsqueeze(0)
    
    # import torch.nn.functional as F # Already imported globally
    img0_pad = F.pad(img0_torch, padding)
    img1_pad = F.pad(img1_torch, padding)
    
    embt = torch.tensor(0.5).view(1, 1, 1, 1).to(device) # Middle frame
    
    start = time.time()
    # IFRNet S inference returns only imgt_pred
    res = model.inference(img0_pad, img1_pad, embt)
    end = time.time()
    
    res = res[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    res = res[:h, :w]
    res = res.astype('uint8')
    return res, end - start

def main():
    if not os.path.exists('input_frames/frame1.png') or not os.path.exists('input_frames/frame3.png'):
        print("Input frames not found in 'input_frames/'. Run setup_test_data.py first.")
        return

    img0 = cv2.imread('input_frames/frame1.png')
    img1 = cv2.imread('input_frames/frame3.png')
    
    # Resize if too large (to avoid VRAM OOM on 4GB GPU)
    max_w = 640
    if img0.shape[1] > max_w:
        scale = max_w / img0.shape[1]
        h = int(img0.shape[0] * scale)
        w = int(img0.shape[1] * scale)
        # Ensure divisible by 32
        h = ((h - 1) // 32 + 1) * 32
        w = ((w - 1) // 32 + 1) * 32
        img0 = cv2.resize(img0, (w, h))
        img1 = cv2.resize(img1, (w, h))
        print(f"Resized input to {w}x{h} for VRAM safety.")
    
    os.makedirs('output_frames', exist_ok=True)
    
    # RIFE
    rife = load_rife()
    if rife:
        print("Running RIFE inference...")
        try:
            rife_out, rife_time = run_rife(rife, img0, img1)
            cv2.imwrite('output_frames/rife_out.png', rife_out)
            print(f"RIFE execution time: {rife_time:.4f}s")
        except Exception as e:
            print(f"RIFE inference failed: {e}")
    
    # IFRNet
    ifrnet = load_ifrnet()
    if ifrnet:
        print("Running IFRNet inference...")
        try:
            ifrnet_out, ifrnet_time = run_ifrnet(ifrnet, img0, img1)
            cv2.imwrite('output_frames/ifrnet_out.png', ifrnet_out)
            print(f"IFRNet execution time: {ifrnet_time:.4f}s")
        except Exception as e:
            print(f"IFRNet inference failed: {e}")

if __name__ == "__main__":
    main()
