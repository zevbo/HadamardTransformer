#ifndef __CUDACC__ // If we're not compiling with nvcc or CUDA isn't available
#define __shared__
// #include <thread>
#include <ctime>
#define __global__
#define __device__
#define __host__
#define __forceinline__
struct dims {
  int x;
  int y;
  int z;
};
dims threadIdx = {};
dims blockDim = {};
dims blockIdx = {};
dims gridDim = {};
#include "cuda_runtime.h"

// #include <immintrin.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#endif

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

typedef __half half;

template <int nSize, typename ty> __device__ void simple_hadamard(ty x[nSize]) {
#pragma unroll
  for (int32_t exchange = 1; exchange < nSize; exchange *= 2) {
    int32_t group_size = exchange * 2;
#pragma unroll
    for (int32_t group_i0 = 0; group_i0 < nSize; group_i0 += group_size) {
#pragma unroll
      for (int32_t i = 0; i < exchange; i++) {
        int32_t i0 = group_i0 + i;
        int32_t i1 = i0 + exchange;
        assert(i0 < nSize);
        assert(i1 < nSize);
        ty a = x[i0];
        ty b = x[i1];
        x[i0] = a + b;
        x[i1] = a - b;
      }
    }
  }
}

#define FULL_MASK 0xFFFFFFFF // uint32_t(-1)

template <int nSize, int nWarpSize, typename ty>
__device__ void warp_shuffle_hadamard(ty x[nSize]) {

  int32_t thread_idx = threadIdx.x % nWarpSize;
#pragma unroll
  for (int32_t exchange = 1; exchange < nWarpSize; exchange *= 2) {
    bool is_bottom = (exchange & thread_idx);
#pragma unroll
    for (int32_t i = 0; i < nSize; i++) {
      ty this_val = x[i];
      ty other_x = __shfl_xor_sync(FULL_MASK, this_val, exchange, nWarpSize);
      x[i] = other_x + (is_bottom ? -1 : 1) * this_val;
    }
  }
}

template <int nSize, int nThreads, int nWarpSize, typename ty>
__device__ void interwarp_transpose(ty x[nSize], ty *shmem) {
  constexpr int32_t nWarps = nThreads / nWarpSize;
  int32_t thread_idx = threadIdx.x % nThreads;
  int32_t thread_id = thread_idx % nWarpSize;
  int32_t warp_id = thread_idx / nWarpSize;
  int32_t transposed_thread_id = thread_idx / nWarps;
  int32_t transposed_warp_id = thread_idx % nWarps;
#define index_of(i, thread, warp) (i * nThreads + warp * nWarpSize + thread)

  for (int32_t i = 0; i < nSize; i++) {
    shmem[index_of(i, thread_id, warp_id)] = x[i];
  }
  __syncthreads();
  for (int32_t i = 0; i < nSize; i++) {
    x[i] = shmem[index_of(i, transposed_thread_id, transposed_warp_id)];
  }
}

template <int nSize, int nThreads, int nWarpSize, typename ty>
__device__ void hadamard_transform(ty x[nSize], ty *shmem) {
  constexpr int32_t nWarps = nThreads / nWarpSize;
  simple_hadamard<nSize, ty>(x);
  warp_shuffle_hadamard<nSize, nWarpSize, ty>(x);
  if (nWarps > 1) {
    assert(shmem != nullptr);
    interwarp_transpose<nSize, nThreads, nWarpSize, ty>(x, shmem);
    warp_shuffle_hadamard<nSize, nWarps, ty>(x);
    interwarp_transpose<nSize, nThreads, nWarpSize, ty>(x, shmem);
  }
}

template <int nFullSize, int nWarpSize, typename ty>
__device__ void hadamard_transform_from_shmem(ty *shmem_x) {
  static_assert(nFullSize % nWarpSize == 0,
                "nFullSize must be divisible by nWarpSize");
  constexpr int32_t nSize = nFullSize / nWarpSize;
  ty x[nSize];
  int32_t i0 = threadIdx.x * nSize;
#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    int32_t j = i; // ^ threadIdx.x;
    x[j] = shmem_x[i0 + j];
  }

  hadamard_transform<nSize, nWarpSize, nWarpSize, ty>(x, shmem_x);

#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    int32_t j = i; // ^ threadIdx.x;
    shmem_x[i0 + j] = x[j];
  }
}

__device__ char float16_to_int4(half val, float scale) {
  float f32 = __half2float(val);
  int scaled = __float2int_rn(f32 * scale);
  return (char)min(max(scaled, -8), 7);
}

__device__ char comb_int4s(char i41, char i42) { return (i41 << 4) + i42; }

template <int nFullSize, int nWarpSize>
__device__ void hadamard_transform_quantize(const half *input_x, char *output) {
  static_assert(nFullSize % nWarpSize == 0,
                "nFullSize must be divisible by nWarpSize");
  constexpr int32_t nSize = nFullSize / nWarpSize;
  static_assert(nSize % 2 == 0,
                "nSize must be a power of 2 (this just checks even though)");
  half x[nSize];
  int32_t thread_idx = threadIdx.x % nWarpSize;
  int32_t i0 = thread_idx * nSize;
#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    int32_t j = i; // ^ thread_idx;
    x[j] = input_x[i0 + j];
  }

  hadamard_transform<nSize, nWarpSize, nWarpSize, half>(x, nullptr);

  int32_t i0_out = i0 / 2;

#pragma unroll
  for (int32_t i = 0; i < nSize / 2; i++) {
    int32_t j = i; // ^ thread_idx);
    output[i0_out + j] =
        comb_int4s(float16_to_int4(x[j * 2]), float16_to_int4(x[j * 2 + 1]));
  }
}

template <int nFullSize, int nWarpSize, typename ty>
__global__ void hadamard_transform_from_global(const ty *x, ty *out) {
  if (blockIdx.x > 0) {
    return;
  }
  const ty *block_x = x + nFullSize * blockIdx.x;
  ty *block_out = out + nFullSize * blockIdx.x;
  extern __shared__ float shmem[];
  ty *shmem_x = (ty *)shmem;

  for (int32_t i = threadIdx.x; i < nFullSize; i += blockDim.x) {
    if (blockIdx.x == 1) {
      // assert(block_x[i] == 0);
    }
    shmem_x[i] = block_x[i];
  }

  hadamard_transform_from_shmem<nFullSize, nWarpSize, ty>(shmem_x);

  for (int32_t i = threadIdx.x; i < nFullSize; i += blockDim.x) {
    block_out[i] = shmem_x[i];
  }
}

torch::Tensor hadamard_transform_f32_1024(torch::Tensor x, int rows) {
  TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat, "Must be f32");
  auto out = torch::empty_like(x);
  int32_t rows_ = x.size(0);
  int32_t cols = x.size(1);
  printf("Rows, cols: %d, %d\n", rows_, cols);
  fflush(stdout);
  hadamard_transform_from_global<1024, 32, float>
      <<<rows, 32, 1024 * 48>>>(x.data_ptr<float>(), out.data_ptr<float>());
  return out;
}

int main() {
  printf("Hello World!\n");
  return 0;
}
