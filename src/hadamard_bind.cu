#include <chrono>
#include <torch/extension.h>

#include "hadamard.cuh"

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
                                  4 * 1000 * 1000 / (448 * 1024 * 1024);
  expected_us /= 1024;
  float slowdown = (float)us / expected_us;
  printf("Total us: %lu. Ideal: %lu. Slowdown of %.2f\n", us, expected_us,
         slowdown); //,
  //         (float)us / expected_us);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform_f32_512", &hadamard_transform_f32_512,
        "Test hadamard, f32, 512 long");
  m.def("hadamard_transform_f32_1024", &hadamard_transform_f32_1024,
        "Test hadamard, f32, 1024 long");
  m.def("hadamard_transform_f32_2048", &hadamard_transform_f32_2048,
        "Test hadamard, f32, 2048 long");
  m.def("hadamard_transform_f32_32768", &hadamard_transform_f32_32768,
        "Test hadamard, f32, 32768 long");
}
