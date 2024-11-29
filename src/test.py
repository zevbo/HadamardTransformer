import torch

import sys

sys.path.append("cuda_extension.so")
import hadamard_cuda


def test_hadamard():
    # Create test tensors
    x = torch.randn(1024, device="cuda", dtype=torch.float32)

    # Call your CUDA function
    c = hadamard_cuda.hadamard_transform_f32_1024(x)
    print(f"{c[0]}")

    # Verify results (assuming hadamard is element-wise multiplication)
    # expected = a * b
    # assert torch.allclose(c, expected)
    print("Test passed!")


if __name__ == "__main__":
    test_hadamard()
