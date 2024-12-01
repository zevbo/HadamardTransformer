import torch

import sys

sys.path.append("cuda_extension.so")
import hadamard_cuda

import numpy as np
from scipy.linalg import hadamard as scipy_hadamard


def test_hadamard():
    # Create test tensors
    x = torch.randn(1024, device="cuda", dtype=torch.float32)

    for i in range(0, 1024):
        break
        if i >= 32 != 0:
            x[i] = 0

    s = 0
    for i in range(32):
        s += x[i]
        print(f"{x[i]}, ", end="")

    print(f"\n{s = }")

    # Call your CUDA function
    print(f"{x = }")
    H = scipy_hadamard(1024)
    correct = torch.tensor(np.dot(H, x.cpu().numpy())).to(torch.float32)

    c = hadamard_cuda.hadamard_transform_f32_1024(x)

    assert torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.01)
    # Verify results (assuming hadamard is element-wise multiplication)
    # expected = a * b
    # assert torch.allclose(c, expected)
    print("Test passed!")


if __name__ == "__main__":
    test_hadamard()
