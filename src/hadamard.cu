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
#include <cuda_runtime.h>
#endif

#include <assert.h>

template <int nSize, typename ty> __device__ void simple_hadamard(ty x[nSize]) {
#pragma unroll
  for (int32_t exchange = 1; exchange < nSize; exchange *= 2) {
    int32_t group_size = exchange << 1;
#pragma unroll
    for (int32_t group = 0; group < nSize; group += group_size) {
      int32_t group_i0 = group * group_size;
#pragma unroll
      for (int32_t i = 0; i < exchange; i++) {
        int32_t i0 = group_i0 + i;
        int32_t i1 = i0 + exchange;
        ty a = x[i0];
        ty b = x[i1];
        x[i0] = a + b;
        x[i1] = a - b;
      }
    }
  }
}

#define FULL_MASK uint32_t(-1)

template <int nSize, int nWarpSize, typename ty>
__device__ void warp_shuffle_hadamard(ty x[nSize]) {

  int32_t thread_idx = threadIdx.x % nWarpSize;
#pragma unroll
  for (int32_t exchange = 1; exchange < nWarpSize; exchange *= 2) {
    int32_t group_size = exchange << 1;
    bool is_bottom = exchange & thread_idx;
#pragma unroll
    for (int32_t i = 0; i < nSize; i++) {
      int32_t this_val = x[i];
      int32_t other_x = __shfl_xor_sync(FULL_MASK, this_val, exchange);
      x[i] = other_x + (is_bottom ? -1 : 1) * x[i];
    }
  }
}

template <int nSize, int nWarpSize, typename ty>
__device__ void hadamard_transform(ty x[nSize]) {
  simple_hadamard<nSize, ty>(x);
  warp_shuffle_hadamard<nSize, ty>(x);
}

template <int nFullSize, int nWarpSize, typename ty>
__device__ void hadamard_transform_from_shmem(ty *shmem_x) {
  if (threadIdx.x >= nWarpSize) {
    // multi-warp not yet supported
    return;
  }
  static_assert(nFullSize % nWarpSize == 0,
                "nFullSize must be divisible by nWarpSize");
  constexpr int32_t nSize = nFullSize / nWarpSize;
  ty x[nSize];
  int32_t i0 = threadIdx.x * nSize;
#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    int32_t j = i ^ threadIdx.x;
    x[j] = shmem_x[i0 + j];
  }

  simple_hadamard<nSize, nWarpSize, ty>(x);

#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    int32_t j = i ^ threadIdx.x;
    shmem_x[i0 + j] = shmem_x[j];
  }
}

template <int nFullSize, int nWarpSize, typename ty>
__global__ void hadamard_transform_from_global(ty *x) {
  ty *block_x = x + nFullSize * blockIdx.x;
  extern __shared__ float shmem[];
  ty shmem_x[nFullSize] = (ty *)shmem;

  for (int32_t i = threadIdx.x; i < nFullSize; i += blockDim.x) {
    shmem_x[i] = block_x[i];
  }

  hadamard_transform_from_shmem<nFullSize, nWarpSize, ty>(shmem_x);
}
