#include <torch/extension.h>
#include <vector>

// Declaration of the kernel function (defined in extension_kernel.cu).
torch::Tensor hadamard_transform_tensor_core_256(torch::Tensor x);
torch::Tensor hadamard_transform_f32_512(torch::Tensor x);
torch::Tensor hadamard_transform_f32_1024(torch::Tensor x);
torch::Tensor hadamard_transform_f32_2048(torch::Tensor x);
torch::Tensor hadamard_transform_f32_32768(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform_tensor_core_256",
        &hadamard_transform_tensor_core_256, "Test hadamard, f16, 256 long");
  m.def("hadamard_transform_f32_512", &hadamard_transform_f32_512,
        "Test hadamard, f32, 512 long");
  m.def("hadamard_transform_f32_1024", &hadamard_transform_f32_1024,
        "Test hadamard, f32, 1024 long");
  m.def("hadamard_transform_f32_2048", &hadamard_transform_f32_2048,
        "Test hadamard, f32, 2048 long");
  m.def("hadamard_transform_f32_32768", &hadamard_transform_f32_32768,
        "Test hadamard, f32, 32768 long");
}
