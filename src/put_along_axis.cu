/**
 * A simple test module for put along axis (isolated test)
*/
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "unified_tensor.hpp"

constexpr int MAX_DIMS = 9;

namespace fab {

struct TensorDesc {
    /**
     * Basic version, no fast div mod, just simple mod and div
     * We can optimize this later
     * 
     * We can use only 16 int64_t in total!
     * 1. dim to put does not need to be recorded here
     * 2. the last stride (well, maybe we need to record it, since transpose can happen, ask somebody who's good at this)
    */
    int64_t shape[MAX_DIMS];            // shape of the index tensor
    int64_t stride_in[MAX_DIMS];        // stride of the input tensor
};

TensorDesc* copy_desc_to_device(const TensorDesc& host_desc) {
    TensorDesc* d_desc;
    CUDA_CHECK_RETURN(cudaMalloc(&d_desc, sizeof(TensorDesc)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_desc, &host_desc, sizeof(TensorDesc), cudaMemcpyHostToDevice));
    return d_desc;
}

__device__ __forceinline__
int64_t compute_offset(
    const TensorDesc* desc, 
    int64_t tid, 
    const int ndim, 
    const int dim_to_put,
    const int64_t idx_on_dim
) {
    int64_t offset = 0;

    // TODO(heqianyue): can be optimized by faster div mod
    // TODO(heqianyue): (builtin assume) shape > 0, no valid check?
    for (int d = ndim - 1; d > dim_to_put; --d) {
        // before the put dim
        int64_t this_shape = desc->shape[d];
        offset += (tid % this_shape) * desc->stride_in[d];      
        tid /= this_shape;
    }
    offset += idx_on_dim * desc->stride_in[dim_to_put];
    tid /= desc->shape[dim_to_put];
    for (int d = dim_to_put - 1; d >= 0; --d) {
        // after the put dim
        int64_t this_shape = desc->shape[d];
        offset += (tid % this_shape) * desc->stride_in[d];      
        tid /= this_shape;
    }
    return offset;
}

template <typename T>
__global__ void ScatterKernel(
    T* inout_tensor,
    const int64_t* indices, 
    const TensorDesc* desc, 
    const int dim,
    const int ndim,
    const int64_t numel,
    T value_to_set
) {
    __shared__ TensorDesc smem_desc;
    if (threadIdx.x < (sizeof(TensorDesc) >> 3)) {
        *(reinterpret_cast<int64_t*>(&smem_desc) + threadIdx.x) = 
        *(reinterpret_cast<int64_t*>(&desc) + threadIdx.x);
    }
    const int64_t thread_num = gridDim.x * blockDim.x;
    __syncthreads();
    int64_t flattened_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // thread coarsening, which is necessary
    for (int64_t idx = flattened_idx; idx < numel; idx += thread_num) {
        int64_t idx_on_dim = indices[flattened_idx];
        int64_t inout_index = compute_offset(&smem_desc, flattened_idx, ndim, dim, idx_on_dim);
        // for actual put along axis, replace the following with src tensor
        atomicExch(inout_tensor + inout_index, value_to_set);
    }
}

void TensorProcessor::put_along_axis(DLManagedTensor* inout_m, const DLManagedTensor* indices_m, int dim) {
    DLTensor& inout = inout_m->dl_tensor;
    const DLTensor& indices = indices_m->dl_tensor;
    if (inout.device.device_type != kDLCUDA ||
        indices.device.device_type != kDLCUDA
    ) {
        THROW_IN_HOST("Check the input or indices tensor, at least one of them is not a CUDA Tensor.\n")
    } 

    int64_t numel_inds = indices.strides[0] * indices.shape[0];

    TensorDesc desc;
    for (int i = 0; i < indices.ndim; i++) {
        desc.shape[i] = inout.shape[i];
        desc.stride_in[i] = inout.strides[i];
    }
    
    auto dev_desc = copy_desc_to_device(desc);

    constexpr int64_t thread_num = 128;
    constexpr int64_t max_thread_num = 2147483647;      // INT_MAX, actually, this should be device-aware
    int64_t block_num = std::min(max_thread_num, numel_inds + thread_num - 1) / thread_num;
    
    if (inout.dtype.bits != 32) {
        THROW_IN_HOST("Unsupported data type with bits = %d.\n", inout.dtype.bits);
    }
#define LAUNCH_SCATTER_KERNEL(dtype)                                \
    ScatterKernel<dtype><<< block_num, thread_num >>>(              \
        static_cast<dtype*>(inout.data),                            \
        static_cast<const int64_t*>(indices.data),                  \
        dev_desc, dim, indices.ndim, numel_inds, 6                  \
    )
    if (inout.dtype.code == kDLFloat) {
        LAUNCH_SCATTER_KERNEL(float);
    } else if (inout.dtype.code == kDLInt) {
        LAUNCH_SCATTER_KERNEL(int);
    } else {
        THROW_IN_HOST("Unsupported data type.");
    }
#undef LAUNCH_SCATTER_KERNEL
    CUDA_CHECK_RETURN(cudaFree(dev_desc));
}


};  // end namespace fab