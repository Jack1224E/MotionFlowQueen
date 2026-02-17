import os
import gdown
import zipfile

def download_rife():
    # RIFE HD model (from README)
    file_id = "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"
    dest_dir = "repo_rife/train_log"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "rife_weights.zip")
    
    # Check if weights already exist
    if os.path.exists(os.path.join(dest_dir, "flownet.pkl")):
        print("RIFE weights (flownet.pkl) already exist. Skipping download.")
        return

    print(f"Downloading RIFE weights to {zip_path}...")
    
    try:
        gdown.download(id=file_id, output=zip_path, quiet=False)
        print("RIFE weights downloaded. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extracted RIFE weights.")
        # Optional: verify extraction
        if os.path.exists(os.path.join(dest_dir, "flownet.pkl")):
             print("Verification: flownet.pkl found.")
        else:
             print("Warning: flownet.pkl not found after extraction. Check zip content.")

    except Exception as e:
        print(f"Failed to download/extract RIFE weights: {e}")

def download_ifrnet():
    dest_dir = "repo_ifrnet/checkpoints"
    os.makedirs(dest_dir, exist_ok=True)
    output_pth = os.path.join(dest_dir, "IFRNet_S.pth")
    
    if os.path.exists(output_pth):
         print("IFRNet weights already exist.")
         return

    print(f"Downloading IFRNet weights from Dropbox...")
    # Dropbox folder link modified for direct download
    url = "https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=1"
    zip_path = os.path.join(dest_dir, "ifrnet_weights.zip")
    
    try:
        # Use simple os.system or subprocess to download with wget first, checking availability
        # fallback to python requests if needed (but we removed requests import, let's stick to system wget or curl)
        cmd = f"wget -O {zip_path} '{url}'"
        print(f"Running: {cmd}")
        ret = os.system(cmd)
        
        if ret != 0:
            print("wget failed. Trying curl...")
            cmd = f"curl -L -o {zip_path} '{url}'"
            print(f"Running: {cmd}")
            ret = os.system(cmd)
            
        if ret == 0 and os.path.exists(zip_path):
            print("Dropbox zip downloaded. Inspecting contents...")
            found = False
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"Files in zip: {file_list}")
                
                for file in file_list:
                    if file.endswith(".pth"):
                        print(f"Found model file: {file}. Extracting...")
                        zip_ref.extract(file, dest_dir)
                        
                        # Move to root of checkpoints if nested
                        extracted_path = os.path.join(dest_dir, file)
                        base_name = os.path.basename(file)
                        final_path = os.path.join(dest_dir, base_name)
                        
                        if extracted_path != final_path:
                             os.rename(extracted_path, final_path)
                             # Cleanup empty dirs
                             try:
                                 os.removedirs(os.path.dirname(extracted_path))
                             except:
                                 pass
                        
                        print(f"Extracted to {final_path}")
                        found = True
            
            if found:
                # Clean up zip only if successful
                os.remove(zip_path)
            else:
                print("No .pth files found in zip. Keeping zip for inspection.")
        else:
            print("Failed to download zip.")

    except Exception as e:
        print(f"Error downloading/extracting IFRNet weights: {e}")

if __name__ == "__main__":
    download_rife()
    download_ifrnet()
