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

#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <assert.h>
#include <stdint.h>

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
        // assert(i0 < nSize);
        // assert(i1 < nSize);
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
      x[i] = other_x + (is_bottom ? -this_val : this_val);
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

__forceinline__ __device__ uint8_t float16_to_int4(half val, half scale) {
  int scaled = __half2int_rn(__hdiv(val, scale));
  return (uint8_t)min(max(scaled, -8), 7);
}

__device__ uint8_t pack_int4s(uint8_t i41, uint8_t i42) {
  return (i42 << 4) | (i41 & 0xF);
}

template <int nFullSize, int nWarpSize = 32>
__device__ void hadamard_transform_group_quantize(const half *input_x,
                                                  uint8_t *output,
                                                  half *group_scale) {
  static_assert(nFullSize % nWarpSize == 0,
                "nFullSize must be divisible by nWarpSize");

  constexpr int32_t nSize = nFullSize / nWarpSize;
  static_assert((nSize & (nSize - 1)) == 0, "nSize must be a power of 2");

  half x[nSize];

  int32_t lane_idx = threadIdx.x % nWarpSize;
  int32_t i0 = lane_idx * nSize;

  // Use vectorized loads to reduce bank conflicts
  if constexpr (nSize == 2) {
    unsigned int raw = *reinterpret_cast<const unsigned int *>(input_x + i0);
    x[0] = __ushort_as_half(raw & 0xFFFF);
    x[1] = __ushort_as_half(raw >> 16);
  } else if constexpr (nSize == 4) {
    uint2 raw = *reinterpret_cast<const uint2 *>(input_x + i0);
    x[0] = __ushort_as_half(raw.x & 0xFFFF);
    x[1] = __ushort_as_half(raw.x >> 16);
    x[2] = __ushort_as_half(raw.y & 0xFFFF);
    x[3] = __ushort_as_half(raw.y >> 16);
  } else if constexpr (nSize == 8) {
    uint4 raw = *reinterpret_cast<const uint4 *>(input_x + i0);
    x[0] = __ushort_as_half(raw.x & 0xFFFF);
    x[1] = __ushort_as_half(raw.x >> 16);
    x[2] = __ushort_as_half(raw.y & 0xFFFF);
    x[3] = __ushort_as_half(raw.y >> 16);
    x[4] = __ushort_as_half(raw.z & 0xFFFF);
    x[5] = __ushort_as_half(raw.z >> 16);
    x[6] = __ushort_as_half(raw.w & 0xFFFF);
    x[7] = __ushort_as_half(raw.w >> 16);
  } else {
#pragma unroll
    for (int32_t i = 0; i < nSize; i++) {
      x[i] = input_x[i0 + i];
    }
  }

  hadamard_transform<nSize, nWarpSize, nWarpSize, half>(x, nullptr);

  half absmax = 0;
#pragma unroll
  for (int32_t i = 0; i < nSize; i++) {
    absmax = __hmax(absmax, __habs(x[i]));
  }

  // Get the absolute maximum value across the warp, via INTEGER warp reduction
  // operation. This is a bit of a hack, but doable because absmax are positive,
  // under which IEEE-754 floats can be compared bit-wise as integers.
  absmax =
      __short_as_half(__reduce_max_sync(0xFFFFFFFF, __half_as_short(absmax)));
  half scale = __hdiv(absmax, 7);

  if (lane_idx == 0) {
    *group_scale = scale;
  }

  int32_t i0_out = i0 / 2;
#pragma unroll
  for (int32_t i = 0; i < nSize / 2; i++) {
    uint8_t packed = pack_int4s(float16_to_int4(x[i * 2], scale),
                                float16_to_int4(x[i * 2 + 1], scale));
    output[i0_out + i] = packed;
  }
}

template <int nFullSize, int nWarpSize, typename ty>
__global__ void hadamard_transform_from_global(const ty *x, ty *out) {
  const ty *block_x = x + nFullSize * blockIdx.x;
  ty *block_out = out + nFullSize * blockIdx.x;
  extern __shared__ float shmem[];
  ty *shmem_x = (ty *)shmem;

  constexpr int32_t nPortion = nFullSize / nWarpSize;
  ty load_registers[nPortion];

#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    load_registers[i] = block_x[threadIdx.x + i * nWarpSize];
  }
#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    shmem_x[threadIdx.x + i * nWarpSize] = load_registers[i];
  }

  __syncthreads();

  hadamard_transform_from_shmem<nFullSize, nWarpSize, ty>(shmem_x);

  __syncthreads();

#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    load_registers[i] = shmem_x[threadIdx.x + i * nWarpSize];
  }
#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    block_out[threadIdx.x + i * nWarpSize] = load_registers[i];
  }
}
