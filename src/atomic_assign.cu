#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "unified_tensor.hpp"

namespace fab {

template <typename T>
__device__ T atomicAssignGeneric(T* address, T val) {
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(address) & ~3;
    uint32_t* aligned_addr = reinterpret_cast<uint32_t*>(base_addr);
    
    uint32_t offset_bytes = reinterpret_cast<uintptr_t>(address) - base_addr;
    uint32_t shift, mask;
    
    if constexpr (sizeof(T) == 1) {
        shift = offset_bytes * 8;
        mask = 0xFFU << shift;
    } else if constexpr (sizeof(T) == 2) {
        shift = (offset_bytes / 2) * 16;
        mask = 0xFFFFU << shift;
    } else {
        return val;
    }
    
    uint32_t old32, assumed32;
    uint32_t new_val32 = *(reinterpret_cast<uint32_t*>(&val)) << shift;
    
    do {
        old32 = *aligned_addr;
        assumed32 = old32;
        uint32_t new32 = (old32 & ~mask) | new_val32;
        old32 = atomicCAS(aligned_addr, assumed32, new32);
    } while (assumed32 != old32);
    uint32_t result = (old32 & mask) >> shift;
    return *reinterpret_cast<T*>(&result);
}

template <typename T>
__device__ __inline__ T atomicAssign(T* address, T val) {
    if constexpr (std::is_same_v<T, int>) {
        return atomicExch(address, val);
    } else if constexpr (std::is_same_v<T, float>) {
        return atomicExch(address, val);
    } else if constexpr (sizeof(T) == 8) {
        auto* dest_address = reinterpret_cast<unsigned long long*>(address);
        unsigned long long value_to_store = *reinterpret_cast<const unsigned long long*>(&val);
        unsigned long long ret = atomicExch(dest_address, value_to_store);
        return *reinterpret_cast<T*>(ret);
    } else {
        return atomicAssignGeneric(address, val);
    }
}

// A dummy copy implemented by atomic assign, purpose for this kernel
// is to test whether my atomic primitives are good
template <typename T>
__global__ void TensorAtomicCopy(
    const T* src,
    T* dst,
    int64_t reduce_size
) {
    int64_t tid = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
    int64_t dst_offset = tid / reduce_size;
    // reduce the last dim of the tensor: map every `reduce_size` elements 
    // to one elements, and see if anything goes wrong
    atomicAssign(dst + dst_offset, *(src + tid));
    
}

void TensorProcessor::assign_reduce(const DLManagedTensor* _src, DLManagedTensor* _dst) const {
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
    
#define LAUNCH_ATOMIC_KERNEL(dtype)                         \
    TensorAtomicCopy<dtype><<< block_num, thread_num>>>(    \
        static_cast<const dtype*>(src.data),                \
        static_cast<dtype*>(inout.data),                    \
        reduce_size                                         \
    )
    if (inout.dtype.code == kDLFloat) {
        if (inout.dtype.bits == 32) {
            LAUNCH_ATOMIC_KERNEL(float);
        } else if (inout.dtype.bits == 64) {
            LAUNCH_ATOMIC_KERNEL(double);
        } else if (inout.dtype.bits == 16) {
            LAUNCH_ATOMIC_KERNEL(half);
        } else {
            THROW_IN_HOST("Unsupported float type with bit: %d\n", inout.dtype.bits);
        }
    } else if (inout.dtype.code == kDLInt) {
        if (inout.dtype.bits == 32) {
            LAUNCH_ATOMIC_KERNEL(int);
        } else if (inout.dtype.bits == 64) {
            LAUNCH_ATOMIC_KERNEL(int64_t);
        } else if (inout.dtype.bits == 16) {
            LAUNCH_ATOMIC_KERNEL(int16_t);
        } else if (inout.dtype.bits == 8) {
            LAUNCH_ATOMIC_KERNEL(uint8_t);
        } else {
            THROW_IN_HOST("Unsupported int type with bit: %d\n", inout.dtype.bits);
        }
    } else if (inout.dtype.code == kDLUInt) {
        if (inout.dtype.bits == 8) {
            LAUNCH_ATOMIC_KERNEL(uint8_t);
        } else {
            THROW_IN_HOST("Unsupported unsigned type with bit: %d\n", inout.dtype.bits);
        }
    } else {
        THROW_IN_HOST("Unsupported data type.\n");
    }
#undef LAUNCH_ATOMIC_KERNEL
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}



};  // end namespace fab

