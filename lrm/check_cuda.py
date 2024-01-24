import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available.")
