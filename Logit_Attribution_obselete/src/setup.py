import subprocess
import sys
import os

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
# 1. Install from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    # Step 1: Install from requirements.txt
    print("Installing from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Step 2: Install additional packages explicitly
    packages = [
        "pip",
        "bitsandbytes==0.41.1",
        "bs4",
        "captum==0.7.0",
        "datasets",
        "dotenv",
        "google",
        "huggingface_hub",
        "numpy==1.26.4",
        "requests",
        "torch==2.2.2",
        "tqdm",
        "transformers",
        "ipywidgets==8.1.1",
        "scipy"
    ]

    print("Installing additional packages...")
    for pkg in packages:
        pip_install(pkg)

    # Optional: Check installed versions
    import transformers
    import torch
    import numpy
    import tqdm

    print("Versions Installed:")
    print("Transformers:", transformers.__version__)
    print("Torch:", torch.__version__)
    print("NumPy:", numpy.__version__)
    print("TQDM:", tqdm.__version__)

if __name__ == "__main__":
    main()
