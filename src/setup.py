from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hadamard_cuda",
    ext_modules=[
        CUDAExtension(
            name="hadamard_cuda",  # This will be the name you import
            sources=["hadamard_bind.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
