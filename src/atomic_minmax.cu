#include <cstdint>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "unified_tensor.hpp"

namespace fab {

#define DEFINE_LOW_HALF_OP(op) \
inline __device__ int op##_to_low_half(int val, int16_t x) { \
  int16_t low_half = op(static_cast<int16_t>(val & 0x0000FFFF), x); \
  return (val & 0xFFFF0000) | (static_cast<int>(low_half) & 0x0000FFFF); \
}

#define DEFINE_HIGH_HALF_OP(op) \
inline __device__ int op##_to_high_half(int val, int16_t x) { \
  int16_t high_half = op(static_cast<int16_t>(val >> 16), x); \
  return (val & 0x0000FFFF) | (static_cast<int>(high_half) << 16); \
}

DEFINE_LOW_HALF_OP(min)
DEFINE_LOW_HALF_OP(max)
DEFINE_HIGH_HALF_OP(min)
DEFINE_HIGH_HALF_OP(max)

#define DEFINE_ATOMIC_MINMAX(OpType, op, bypass_op)                \
  __device__ __forceinline__ int16_t CudaAtomic##OpType(int16_t* address,           \
                                                      const int16_t val) {        \
  if (*address bypass_op val) { \
    return *address; \
  } \
  int *address_as_ui = reinterpret_cast<int *>(   \
      reinterpret_cast<char *>(address) -   \
      (reinterpret_cast<uintptr_t>(address) & 0x02));   \
  int old = 0, assumed = 0; \
  if ((uintptr_t)address & 0x02) {   \
    old = *address_as_ui; \
    do {    \
      assumed = old;    \
      old = atomicCAS(address_as_ui, assumed, op##_to_high_half(assumed, val));   \
    } while (old != assumed);   \
    return static_cast<int16_t>(old >> 16); \
  } else {  \
    old = *address_as_ui; \
    do {    \
      assumed = old;    \
      old = atomicCAS(address_as_ui, assumed, op##_to_low_half(assumed, val)); \
    } while (old != assumed);   \
    return static_cast<int16_t>(old & 0x0000FFFF);     \
  } \
}

DEFINE_ATOMIC_MINMAX(Min, min, <=)
DEFINE_ATOMIC_MINMAX(Max, max, >=)

#undef DEFINE_ATOMIC_MINMAX
#undef DEFINE_LOW_HALF_OP
#undef DEFINE_HIGH_HALF_OP

// A dummy copy implemented by atomic assign, purpose for this kernel
// is to test whether my atomic primitives are good
template <typename T, bool is_min>
__global__ void TensorAtomicMinMax(
    const T* src,
    T* dst,
    int64_t reduce_size,
    int64_t numel
) {
    int64_t tid = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
    if (tid >= numel) return;
    int64_t dst_offset = tid / reduce_size;
    // reduce the last dim of the tensor: map every `reduce_size` elements 
    // to one elements, and see if anything goes wrong
    if constexpr (is_min) {
        CudaAtomicMin(dst + dst_offset, *(src + tid));
    } else {
        CudaAtomicMax(dst + dst_offset, *(src + tid));
    }
}

template <bool is_min_op>
static void minmax_last_dim(const DLManagedTensor* _src, DLManagedTensor* _dst) {
const DLTensor& src = _src->dl_tensor;
    DLTensor& inout = _dst->dl_tensor;
    if (src.device.device_type != kDLCUDA ||
        inout.device.device_type != kDLCUDA
    ) {
        THROW_IN_HOST("Check the input or indices tensor, at least one of them is not a CUDA Tensor.\n")
    } 

    if (src.dtype.code != inout.dtype.code ||
        src.dtype.bits != inout.dtype.bits
    ) {
        THROW_IN_HOST("Expect two tensors to have the same dtype but received different dtypes, please check.\n")
    }

    if (src.ndim <= 1) {
        THROW_IN_HOST("Expect source tensor to be at least 2D, but got ndim = %d.\n", src.ndim);
    }

    int64_t reduce_size = src.shape[src.ndim - 1];
    int64_t src_numel = src.shape[0] * src.strides[0];

    constexpr int64_t thread_num = 128;
    constexpr int64_t max_thread_num = 2147483647;      // INT_MAX, actually, this should be device-aware
    int64_t block_num = std::min(max_thread_num, src_numel + thread_num - 1) / thread_num;
    
#define LAUNCH_ATOMIC_KERNEL(dtype, is_min)                         \
    TensorAtomicMinMax<dtype, is_min><<< block_num, thread_num>>>(  \
        static_cast<const dtype*>(src.data),                        \
        static_cast<dtype*>(inout.data),                            \
        reduce_size,                                                \
        src_numel                                                   \
    )

    if (inout.dtype.code == kDLInt) {
        if (inout.dtype.bits == 16) {
            LAUNCH_ATOMIC_KERNEL(int16_t, is_min_op);
        } else {
            THROW_IN_HOST("Unsupported int type with bit: %d\n", inout.dtype.bits);
        }
    } else {
        THROW_IN_HOST("Unsupported data type.\n");
    }
#undef LAUNCH_ATOMIC_KERNEL
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void TensorProcessor::min_last_dim(const DLManagedTensor* _src, DLManagedTensor* _dst) const {
    minmax_last_dim<true>(_src, _dst);
}

void TensorProcessor::max_last_dim(const DLManagedTensor* _src, DLManagedTensor* _dst) const {
    minmax_last_dim<false>(_src, _dst);
}

}   // end namespace fab