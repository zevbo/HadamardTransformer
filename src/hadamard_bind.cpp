#include <torch/extension.h>
#include <vector>

// Declaration of the kernel function (defined in extension_kernel.cu).
torch::Tensor hadamard_transform_f32_1024(torch::Tensor x, int num_rows);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform_f32_1024", &hadamard_transform_f32_1024,
        "Test hadamard, f32, 1024 long");
}
