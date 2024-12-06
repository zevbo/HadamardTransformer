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


size_to_f = {
    512: hadamard_cuda.hadamard_transform_f32_512,
    1024: hadamard_cuda.hadamard_transform_f32_1024,
}


def test_hadamard_multi(size, rows):
    # Create test tensors
    print(f"Testing hadamard with {rows} rows")
    x = torch.randn((rows, size), device="cuda", dtype=torch.float32)

    H = scipy_hadamard(size)
    correct = torch.tensor(np.dot(H, x.cpu().numpy().T)).to(torch.float32)

    t1 = time.perf_counter_ns()
    c = size_to_f[size](x)
    t2 = time.perf_counter_ns()
    c = c.T
    # print(f"{c.shape = }, {correct.shape = }")
    # for i in range(4):
    #    print(f"{i = }: {c[i,0] = }, {correct[i,0] = }")
    ideal_t = x.numel() * 2 * 4 * 1000 / (448 * 1024 * 1024 * 1024)
    total_time = (t2 - t1) / (1000 * 1000)
    slowdown = total_time / ideal_t

    passed = torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.01)
    if passed:
        print(
            f"Test passed! Took {total_time} ms, which is a slowdown of {round(slowdown, 2)}"
        )
    else:
        print(
            f"Test failed. Took {total_time} ms, which is a slowdown of {round(slowdown, 2)}"
        )


def test_hadamard_tensor_core(rows):
    size = 256
    print(f"Testing tensor core hadamard with {rows} rows")
    x = torch.randn((rows, size), device="cuda", dtype=torch.float16)

    print(f"{x.shape = }, {x.stride = }, {x.is_contiguous()}")

    H = scipy_hadamard(size)
    correct = torch.tensor(np.dot(H, x.cpu().numpy().T)).to(torch.float16)

    t1 = time.perf_counter_ns()
    c: torch.Tensor = hadamard_cuda.hadamard_transform_tensor_core_256(x)
    t2 = time.perf_counter_ns()
    print(f"{c.shape = }, {c.stride() = }, {c.is_contiguous() = }")
    c = c.T

    ideal_t = x.numel() * 2 * 4 * 1000 / (448 * 1024 * 1024 * 1024)
    total_time = (t2 - t1) / (1000 * 1000)
    slowdown = total_time / ideal_t

    passed = torch.allclose(c, torch.tensor(correct).to("cuda"), atol=0.05)
    if passed:
        print(
            f"TC Test passed! Took {total_time} ms, which is a slowdown of {round(slowdown, 2)}"
        )
    else:
        print(
            f"TC Test failed. Took {total_time} ms, which is a slowdown of {round(slowdown, 2)}"
        )


if __name__ == "__main__":
    print("C")
    # test_hadamard()
    size = 1024
    with torch.no_grad():
        test_hadamard_tensor_core(1)
        test_hadamard_tensor_core(128)
        test_hadamard_tensor_core(1024 * 16)
        test_hadamard_tensor_core(1024 * 128)
        test_hadamard_multi(size, 1)
        test_hadamard_multi(size, 2)
        test_hadamard_multi(size, 128)
        test_hadamard_multi(size, 1024)
        test_hadamard_multi(size, 1024 * 16)

        # test_hadamard_multi(size, 1024 * 32)
        # test_hadamard_multi(size, 1024 * 64)
        # test_hadamard_multi(size, 1024 * 128)
