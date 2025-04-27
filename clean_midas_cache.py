"""
clean_midas_cache.py

This script removes all locally cached files and directories
related to MiDaS models loaded via torch.hub.
"""

import os
import shutil
import torch

def remove_dir(path):
    if os.path.isdir(path):
        print(f"Removing directory: {path}")
        shutil.rmtree(path, ignore_errors=True)

def remove_file(path):
    if os.path.isfile(path):
        print(f"Removing file: {path}")
        os.remove(path)

def main():
    # Locate torch.hub cache directory
    hub_dir = torch.hub.get_dir()
    print(f"Torch hub directory: {hub_dir}")

    # 1) Remove the MiDaS repo directory
    midas_repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
    remove_dir(midas_repo)

    # 2) Remove MiDaS checkpoints under hub_dir/checkpoints
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for fname in os.listdir(ckpt_dir):
            # Common MiDaS checkpoint filename patterns
            if fname.lower().startswith("dpt_") or "midas" in fname.lower():
                remove_file(os.path.join(ckpt_dir, fname))

    # 3) (Optional) Remove related EfficientNet weights if desired
    eff_repo = os.path.join(hub_dir, "rwightman_gen-efficientnet-pytorch_master")
    remove_dir(eff_repo)

    print("MiDaS cache cleanup complete.")

if __name__ == "__main__":
    main()
