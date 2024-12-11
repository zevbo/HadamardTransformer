#ifndef HADAMARD_CUH
#define HADAMARD_CUH
#ifndef __CUDACC__ // If we're not compiling with nvcc or CUDA isn't available
#define __shared__
// #include <thread>
#include <chrono>
#include <cstdio>
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
#include <cuda_runtime_api.h>
#include <vector>
#endif

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef __half half;
typedef __half2 packed_half;

template <int nFullSize, int nWarpSize, typename ty>
__device__ void load_to_shmem(const ty *x, ty *shmem_x) {
  const ty *block_x = x + nFullSize * blockIdx.x;
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
}

template <int nFullSize, int nWarpSize, typename ty>
__device__ void load_from_shmem(ty *out, const ty *shmem_x) {
  __syncthreads();
  ty *block_out = out + nFullSize * blockIdx.x;
  constexpr int32_t nPortion = nFullSize / nWarpSize;
  ty load_registers[nPortion];
#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    load_registers[i] = shmem_x[threadIdx.x + i * nWarpSize];
  }
#pragma unroll
  for (int32_t i = 0; i < nPortion; i++) {
    block_out[threadIdx.x + i * nWarpSize] = load_registers[i];
  }
}

struct HalfOp {
  static __device__ inline half add(half h1, half h2) { return __hadd(h1, h2); }
  static __device__ inline half sub(half h1, half h2) { return __hsub(h1, h2); }
};

template <int nSize, typename ty, typename op>
__device__ inline void simple_hadamard_tmpl(ty x[nSize], int32_t swizzler) {
#pragma unroll
  for (int32_t exchange = 1; exchange < nSize; exchange *= 2) {
    bool reverse_exchange = (swizzler & exchange) != 0;
    int32_t group_size = exchange * 2;
#pragma unroll
    for (int32_t group_i0 = 0; group_i0 < nSize; group_i0 += group_size) {
#pragma unroll
      for (int32_t i = 0; i < exchange; i++) {
        int32_t i0 = group_i0 + i;
        int32_t i1 = i0 ^ exchange;
        if (reverse_exchange) {
          ty a = x[i1];
          ty b = x[i0];
          x[i1] = op::add(a, b);
          x[i0] = op::sub(a, b);
        } else {
          ty a = x[i0];
          ty b = x[i1];
          x[i0] = op::add(a, b);
          x[i1] = op::sub(a, b);
        }
      }
    }
  }
}

template <int nSize, typename ty>
__device__ inline void simple_hadamard(ty x[nSize]) {
#pragma unroll
  for (int32_t exchange = 1; exchange < nSize; exchange *= 2) {
    int32_t group_size = exchange * 2;
#pragma unroll
    for (int32_t group_i0 = 0; group_i0 < nSize; group_i0 += group_size) {
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

inline __device__ uint32_t half2_to_uint(packed_half h2_val) {
  return *reinterpret_cast<uint32_t *>(&h2_val);
}

void __device__ tensor_core_hadamard_shmem_128(half *shmem_x) {
  constexpr int width = 8;
  constexpr int height = 16;
  int32_t r0 = threadIdx.x / 4;
  int32_t c0 = (threadIdx.x % 4) * 2;
#define is_neg_corn_no_mod_128(r, c, size) (r >= (size / 2) && c >= (size / 2))
#define is_neg_corn_128(r, c, size)                                            \
  ((r % size) >= (size / 2) && (c % size) >= (size / 2))
  bool is_neg_0 = is_neg_corn_no_mod_128(r0, c0, 8) ^
                  is_neg_corn_128(r0, c0, 4) ^ is_neg_corn_128(r0, c0, 2);
  float H_0_0 = is_neg_0 ? -1.0f : 1.0f;
  bool is_neg_1 = is_neg_0 ^ is_neg_corn_128(r0, c0 + 1, 2);
  float H_0_1 = is_neg_1 ? -1.0f : 1.0f;

  packed_half H_0 = __half2(__float2half(H_0_0), __float2half(H_0_1));
  packed_half H_1 = H_0;
  packed_half H_2 = H_0;
  packed_half H_3 = __half2(__float2half(-1 * H_0_0), __float2half(-1 * H_0_1));

  //  constexpr int size = side_size * side_size;
  int32_t row_0 = 2 * (threadIdx.x % 4);
  int32_t col_0 = threadIdx.x / 4;
#define get_shmem_x_128(row, col) shmem_x[(row) * width + (col)]
  packed_half t_0_1 =
      __half2(get_shmem_x_128(row_0, col_0), get_shmem_x_128(row_0 + 1, col_0));
  packed_half t_0_2 = __half2(get_shmem_x_128(row_0 + height / 2, col_0),
                              get_shmem_x_128(row_0 + height / 2 + 1, col_0));

  uint32_t output[2];
  packed_half *packed_half_output = reinterpret_cast<packed_half *>(output);

  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, "
      "{%2, %3, %4, %5}, "
      "{%6, %7}, "
      "{%8, %9};"
      : "=r"(output[0]), "=r"(output[1])
      : "r"(half2_to_uint(H_0)), "r"(half2_to_uint(H_1)),
        "r"(half2_to_uint(H_2)), "r"(half2_to_uint(H_3)),
        "r"(half2_to_uint(t_0_1)), "r"(half2_to_uint(t_0_2)), "r"(0), "r"(0));

  int32_t write_row_0 = threadIdx.x / 4;
  int32_t write_col_0 = (threadIdx.x % 4) * 2;

  __syncthreads();

  get_shmem_x_128(write_row_0, write_col_0) = __low2half(packed_half_output[0]);
  get_shmem_x_128(write_row_0, write_col_0 + 1) =
      __high2half(packed_half_output[0]);
  get_shmem_x_128(write_row_0 + 8, write_col_0) =
      __low2half(packed_half_output[1]);
  get_shmem_x_128(write_row_0 + 8, write_col_0 + 1) =
      __high2half(packed_half_output[1]);

  __syncthreads();

  half local_x[8];
  int32_t i0 = width * threadIdx.x;
  int32_t swizzler = threadIdx.x % 8;
#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    local_x[i] = shmem_x[i0 + i ^ swizzler];
  }

  simple_hadamard_tmpl<8, half, HalfOp>(local_x, swizzler);

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    shmem_x[i0 + i ^ swizzler] = local_x[i];
  }
}

void __device__ tensor_core_hadamard_shmem_256(half *shmem_x) {
  constexpr int side_size = 16;
  const int32_t lane_id = threadIdx.x % 32;

  int32_t r0 = lane_id / 4;
  int32_t c0 = (lane_id % 4) * 2;
#define is_neg_corn_no_mod(r, c, size) (r >= (size / 2) && c >= (size / 2))
#define is_neg_corn(r, c, size)                                                \
  ((r % size) >= (size / 2) && (c % size) >= (size / 2))
  bool is_neg_0 = is_neg_corn_no_mod(r0, c0, 8) ^ is_neg_corn(r0, c0, 4) ^
                  is_neg_corn(r0, c0, 2);
  float H_0_0 = is_neg_0 ? -1.0f : 1.0f;
  bool is_neg_1 = is_neg_0 ^ is_neg_corn(r0, c0 + 1, 2);
  float H_0_1 = is_neg_1 ? -1.0f : 1.0f;

  packed_half H_0 = __half2(__float2half(H_0_0), __float2half(H_0_1));
  packed_half H_1 = H_0;
  packed_half H_2 = H_0;
  packed_half H_3 = __half2(__float2half(-1 * H_0_0), __float2half(-1 * H_0_1));

  //  constexpr int size = side_size * side_size;
  for (int run = 0; run <= 1; run++) {
    for (int side = 0; side <= 1; side++) {
      int32_t row_0 = 2 * (lane_id % 4);
      int32_t col_0 = lane_id / 4;
#define get_shmem_x_under(row, col)                                            \
  shmem_x[(run == 0 ? (row) : (col)) +                                         \
          ((run == 0 ? (col) : (row)) * (side_size))]
#define get_shmem_x(row, col)                                                  \
  get_shmem_x_under(row, (col + side * (side_size / 2)))
      packed_half t_0_1 =
          __half2(get_shmem_x(row_0, col_0), get_shmem_x(row_0 + 1, col_0));
      packed_half t_0_2 =
          __half2(get_shmem_x(row_0 + side_size / 2, col_0),
                  get_shmem_x(row_0 + side_size / 2 + 1, col_0));

      uint32_t output[2];
      packed_half *packed_half_output = reinterpret_cast<packed_half *>(output);

      asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0, %1}, "
          "{%2, %3, %4, %5}, "
          "{%6, %7}, "
          "{%8, %9};"
          : "=r"(output[0]), "=r"(output[1])
          : "r"(half2_to_uint(H_0)), "r"(half2_to_uint(H_1)),
            "r"(half2_to_uint(H_2)), "r"(half2_to_uint(H_3)),
            "r"(half2_to_uint(t_0_1)), "r"(half2_to_uint(t_0_2)), "r"(0),
            "r"(0));

      __syncthreads();

      int32_t write_row_0 = lane_id / 4;
      int32_t write_col_0 = (lane_id % 4) * 2;
      get_shmem_x(write_row_0, write_col_0) = __low2half(packed_half_output[0]);
      get_shmem_x(write_row_0, write_col_0 + 1) =
          __high2half(packed_half_output[0]);
      get_shmem_x(write_row_0 + 8, write_col_0) =
          __low2half(packed_half_output[1]);
      get_shmem_x(write_row_0 + 8, write_col_0 + 1) =
          __high2half(packed_half_output[1]);
    }
  }
}

__global__ void tensor_core_hadamard_128(const half *x, half *out) {
  extern __shared__ float shmem[];
  half *shmem_x = (half *)shmem;
  load_to_shmem<128, 32, half>(x, shmem_x);
  tensor_core_hadamard_shmem_128(shmem_x);
  load_from_shmem<128, 32, half>(out, shmem_x);
}

__global__ void tensor_core_hadamard_256(const half *x, half *out) {
  extern __shared__ float shmem[];
  half *shmem_x = (half *)shmem;
  load_to_shmem<256, 32, half>(x, shmem_x);
  tensor_core_hadamard_shmem_256(shmem_x);
  load_from_shmem<256, 32, half>(out, shmem_x);
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

template <int nFullSize, int nWarpSize = 32, bool quantize = true>
__device__ void hadamard_transform_group_quantize(uint8_t *data,
                                                  half *group_scale) {
  static_assert(nFullSize % nWarpSize == 0,
                "nFullSize must be divisible by nWarpSize");

  constexpr int32_t nSize = nFullSize / nWarpSize;
  static_assert((nSize & (nSize - 1)) == 0, "nSize must be a power of 2");

  half x[nSize];

  int32_t lane_idx = threadIdx.x % nWarpSize;

  int32_t i0 = lane_idx * nSize;

  if constexpr (nFullSize == 256) {
    tensor_core_hadamard_shmem_256(reinterpret_cast<half *>(data));
  }

  // Use vectorized loads to reduce bank conflicts
  {
    const half *input_x = reinterpret_cast<const half *>(data);
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
  }

  if constexpr (nFullSize != 256) {
    hadamard_transform<nSize, nWarpSize, nWarpSize, half>(x, nullptr);
  }
  // hadamard_transform<nSize, nWarpSize, nWarpSize, half>(x, nullptr);

  if constexpr (quantize) {
    half absmax = __float2half(0.0);
#pragma unroll
    for (int32_t i = 0; i < nSize; i++) {
      absmax = __hmax(absmax, __habs(x[i]));
    }

    // Get the absolute maximum value across the warp, via INTEGER warp
    // reduction operation. This is a bit of a hack, but doable because absmax
    // are positive, under which IEEE-754 floats can be compared bit-wise as
    // integers.
    absmax =
        __short_as_half(__reduce_max_sync(0xFFFFFFFF, __half_as_short(absmax)));
    half scale = __hdiv(absmax, __float2half(7.0));

    if (lane_idx == 0) {
      *group_scale = scale;
    }
    __syncwarp();
    {
      int32_t i0_out = i0 / 2;
      uint8_t *output = data;
#pragma unroll
      for (int32_t i = 0; i < nSize / 2; i++) {
        uint8_t packed = pack_int4s(float16_to_int4(x[i * 2], scale),
                                    float16_to_int4(x[i * 2 + 1], scale));
        output[i0_out + i] = packed;
      }
    }
  }
}

template <int nFullSize, int nWarpSize, typename ty>
__global__ void hadamard_transform_from_global(const ty *x, ty *out) {
  extern __shared__ float shmem[];
  ty *shmem_x = (ty *)shmem;

  load_to_shmem<nFullSize, nWarpSize, ty>(x, shmem_x);
  hadamard_transform_from_shmem<nFullSize, nWarpSize, ty>(shmem_x);
  load_from_shmem<nFullSize, nWarpSize, ty>(out, shmem_x);
}

#endif
