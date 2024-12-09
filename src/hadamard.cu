#include "hadamard.cuh"

#ifndef __CUDACC__ // If we're not compiling with nvcc or CUDA isn't available
#else

#include <torch/extension.h>
#include <vector>
#endif

template <int nFullSize> torch::Tensor hadamard_transform_f32(torch::Tensor x) {
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat, "Must be f32");
  auto out = torch::empty_like(x);
  int32_t rows = x.size(0);
  printf("Rows, nFullSize: %d, %d\n", rows, nFullSize);
  fflush(stdout);

  auto t1 = std::chrono::high_resolution_clock::now();
  hadamard_transform_from_global<nFullSize, 32, float>
      <<<rows, 32, nFullSize * sizeof(float)>>>(x.data_ptr<float>(),
                                                out.data_ptr<float>());
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  long unsigned int expected_us = ((uint64_t)rows) * ((uint64_t)nFullSize) * 2 *
                                  sizeof(float) * 1000 * 1000 /
                                  (448 * 1024 * 1024);
  expected_us /= 1024;
  float slowdown = (float)us / expected_us;
  printf("Total us %d x %d: %lu. Ideal: %lu. Slowdown of %.2f\n", rows,
         nFullSize, us, expected_us,
         slowdown); //,
  //         (float)us / expected_us);
  return out;
}

torch::Tensor hadamard_transform_tensor_core_256(torch::Tensor x) {
  printf("Starting tensor core 256 run\n");
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "Must be f16");
  auto out = torch::empty_like(x, x.options().dtype(at::kHalf).memory_format(
                                      torch::MemoryFormat::Contiguous));
  int32_t rows = x.size(0);
  auto t1 = std::chrono::high_resolution_clock::now();
  tensor_core_hadamard_256<<<rows, 32, 256 * sizeof(half)>>>(
      reinterpret_cast<half *>(x.data_ptr<at::Half>()),
      reinterpret_cast<half *>(out.data_ptr<at::Half>()));
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  long unsigned int expected_us = ((uint64_t)rows) * ((uint64_t)256) * 2 *
                                  sizeof(half) * 1000 * 1000 /
                                  (448 * 1024 * 1024);
  expected_us /= 1024;
  float slowdown = (float)us / expected_us;
  printf("TC Total us %d: %lu. Ideal: %lu. Slowdown of %.2f\n", rows, us,
         expected_us,
         slowdown); //,
  return out;
}

torch::Tensor hadamard_transform_f32_512(torch::Tensor x) {
  return hadamard_transform_f32<512>(x);
}

torch::Tensor hadamard_transform_f32_1024(torch::Tensor x) {
  return hadamard_transform_f32<1024>(x);
}
torch::Tensor hadamard_transform_f32_2048(torch::Tensor x) {
  return hadamard_transform_f32<2048>(x);
}

torch::Tensor hadamard_transform_f32_32768(torch::Tensor x) {
  return hadamard_transform_f32<32768>(x);
}

int main() {
  printf("Hello World!\n");
  return 0;
}
