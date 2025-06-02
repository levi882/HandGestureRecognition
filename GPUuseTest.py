import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}") # 0是第一个GPU的索引
else:
    print("CUDA is not available. Please check your PyTorch installation and NVIDIA drivers.")
