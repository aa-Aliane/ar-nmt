import torch
from accelerate import Accelerator

import src


def verify():
    print("--- GPU Check ---")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPUs detected: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\n--- Accelerate Check ---")
    accelerator = Accelerator()
    print(f"Device: {accelerator.device}")

    print("\n--- Package Check ---")
    print(f"Local 'src' path: {src.__file__}")
    print("Verification Successful!")


if __name__ == "__main__":
    verify()
