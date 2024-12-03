import torch

import sys

sys.path.append("cuda_extension.so")
import hadamard_cuda

import numpy as np
from scipy.linalg import hadamard as scipy_hadamard


def test_hadamard():
    # Create test tensors
    x = torch.randn(1024, device="cuda", dtype=torch.float32)

    H = scipy_hadamard(1024)
    correct = torch.tensor(np.dot(H, x.cpu().numpy())).to(torch.float32)

    c = hadamard_cuda.hadamard_transform_f32_1024(x, 1)

    assert torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.01)
    print("Test passed!")


def test_hadamard_multi(rows):
    # Create test tensors
    print(f"Testing hadamard with {rows} rows")
    x = torch.randn((1024, rows), device="cuda", dtype=torch.float32)

    H = scipy_hadamard(1024)
    correct = torch.tensor(np.dot(H, x.cpu().numpy())).to(torch.float32)

    c = hadamard_cuda.hadamard_transform_f32_1024(x.T, rows)

    print(f"{c[0,0] = }, {correct[0, 0] = }")

    assert torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.01)
    print("Test passed!")


if __name__ == "__main__":
    # test_hadamard()
    test_hadamard_multi(1)
    test_hadamard_multi(2)
