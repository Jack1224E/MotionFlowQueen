import torch
import subprocess
import sys
import os
import re

def check_cuda_capability():
    print("[-] Checking CUDA Capability...")
    if not torch.cuda.is_available():
        print("[!] CUDA not available in PyTorch.")
        return False
    
    cap = torch.cuda.get_device_capability(0)
    print(f"    Detected Compute Capability: {cap[0]}.{cap[1]}")
    if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
        print("[!] ERROR: NVOFA requires Turing (7.5) or newer.")
        return False
    print("    [OK] Capability >= 7.5")
    return True

def check_driver_version():
    print("[-] Checking NVIDIA Driver Version...")
    try:
        smi_out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], encoding='utf-8')
        version_str = smi_out.strip()
        print(f"    Detected Driver Version: {version_str}")
        
        major_ver = float(version_str.split('.')[0])
        if major_ver < 470:
            print("[!] ERROR: Driver version must be > 470.")
            return False
        print("    [OK] Driver Version > 470")
        return True
    except Exception as e:
        print(f"[!] ERROR: Failed to query nvidia-smi: {e}")
        return False

def check_library_existence():
    print("[-] Checking libnvidia-opticalflow.so.1...")
    paths = [
        "/usr/lib/libnvidia-opticalflow.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.1",
        "/usr/lib64/libnvidia-opticalflow.so.1"
    ]
    
    found = False
    for p in paths:
        if os.path.exists(p):
            print(f"    [OK] Found library at: {p}")
            found = True
            break
    
    if not found:
        # Try ldconfig to find it
        try:
            ld_out = subprocess.check_output(["ldconfig", "-p"], encoding='utf-8')
            if "libnvidia-opticalflow.so.1" in ld_out:
                print("    [OK] Found in ldconfig cache.")
                found = True
        except:
            pass
            
    if not found:
        print("[!] ERROR: libnvidia-opticalflow.so.1 not found.")
        return False
    return True

def main():
    print("=== NVOFA Pre-flight Check ===")
    checks = [
        check_cuda_capability(),
        check_driver_version(),
        check_library_existence()
    ]
    
    if all(checks):
        print("\n[SUCCESS] Environment is ready for NVOFA.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Environment check failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
