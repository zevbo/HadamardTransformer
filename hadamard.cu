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

template <int nSize, typename ty> __device__ void simple_hadamard(ty x[]) {
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
