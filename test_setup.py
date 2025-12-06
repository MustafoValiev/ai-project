import torch
import pandas as pd
import numpy as np
import sklearn
import os

print("=" * 70)
print("SETUP VERIFICATION")
print("=" * 70)

print("\nPython packages:")
print(f"  - Pandas: {pd.__version__}")
print(f"  - NumPy: {np.__version__}")
print(f"  - Scikit-learn: {sklearn.__version__}")

print("\nPyTorch:")
print(f"  - Version: {torch.__version__}")
print(f"  - CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\nGPU READY")
else:
    print("\nGPU not available; using CPU")

print("\nDataset:")
if os.path.exists('./data/train.csv'):
    size = os.path.getsize('./data/train.csv') / 1e6
    print(f"  - train.csv: {size:.1f} MB")
else:
    print("  - train.csv: NOT FOUND")

print("=" * 70)
