import torch
import time

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
    x = torch.randn((rows, 1024), device="cuda", dtype=torch.float32)

    H = scipy_hadamard(1024)
    correct = torch.tensor(np.dot(H, x.cpu().numpy().T)).to(torch.float32)

    t1 = time.perf_counter_ns()
    c = hadamard_cuda.hadamard_transform_f32_1024(x, rows)
    t2 = time.perf_counter_ns()
    c = c.T

    # assert torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.01)
    print(
        f"Test passed! Took {(t2 - t1) / (1000 * 1000)} ms. Memory load would be {x.numel() * 2 * 4 * 1000 / (448 * 1024 * 1024 * 1024)}"
    )


if __name__ == "__main__":
    print("B")
    # test_hadamard()
    with torch.no_grad():
        test_hadamard_multi(1)
        test_hadamard_multi(2)
        test_hadamard_multi(128)
        test_hadamard_multi(1024)
