/**
 * A simple test module for put along axis (isolated test)
 * Note that, I don't want to optimize this any more, since
 * the implementation is migrated to paddle's PHI lib
 * 
 * I will optimize things there.
*/
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cuda_utils.cuh"
#include "unified_tensor.hpp"

namespace fab {

__device__ __forceinline__
int64_t compute_offset(
    const int64_t* index_shape,
    int64_t tid, 
    const int ndim, 
    const int dim_to_put,
    const int64_t idx_on_dim
) {
    int64_t offset = 0;
    // index shape tensor is reused, the first `ndim` elements are the shape of the index tensor
    // while the second `ndim` elements are the strides of the input tensor
    const int64_t* input_stride = index_shape + ndim;
    for (int d = ndim - 1; d > dim_to_put; --d) {
        // before the put dim
        int64_t this_shape = index_shape[d];
        offset += (tid % this_shape) * input_stride[d];      

        tid /= this_shape;
    }
    offset += idx_on_dim * input_stride[dim_to_put];
    tid /= index_shape[dim_to_put];
    for (int d = dim_to_put - 1; d >= 0; --d) {
        // after the put dim
        int64_t this_shape = index_shape[d];
        offset += (tid % this_shape) * input_stride[d];      
        tid /= this_shape;
    }
    return offset;
}

template <typename T>
__global__ void ScatterKernel(
    T* inout_tensor,
    const int64_t* indices, 
    const int64_t* desc, 
    const int dim,
    const int ndim,
    const int64_t numel,
    T value_to_set
) {
    extern __shared__ int64_t smem_desc[];
    if (threadIdx.x < (2 * ndim)) {
        *(smem_desc + threadIdx.x) = 
        *(desc + threadIdx.x);
    }
    const int64_t thread_num = gridDim.x * blockDim.x;
    __syncthreads();
    int64_t flattened_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // thread coarsening, which is necessary
    for (int64_t idx = flattened_idx; idx < numel; idx += thread_num) {
        int64_t idx_on_dim = indices[idx];
        int64_t inout_index = compute_offset(smem_desc, idx, ndim, dim, idx_on_dim);
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

   thrust::host_vector<int64_t> host_vec(indices.ndim * 2);
   thrust::device_vector<int64_t> devc_vec(indices.ndim * 2);

    for (int i = 0; i < indices.ndim; i++) {
        host_vec[i] = indices.shape[i];
        host_vec[i + indices.ndim] = inout.strides[i];
    }
    
    devc_vec = host_vec;
    const int64_t* dev_desc = thrust::raw_pointer_cast(devc_vec.data());

    constexpr int64_t thread_num = 128;
    constexpr int64_t max_thread_num = 2147483647;      // INT_MAX, actually, this should be device-aware
    int64_t block_num = std::min(max_thread_num, numel_inds + thread_num - 1) / thread_num;
    
    if (inout.dtype.bits != 32) {
        THROW_IN_HOST("Unsupported data type with bits = %d.\n", inout.dtype.bits);
    }
    const size_t dyn_smem_size = sizeof(int64_t) * indices.ndim * 2;       
#define LAUNCH_SCATTER_KERNEL(dtype)                                    \
    ScatterKernel<dtype><<< block_num, thread_num,  dyn_smem_size>>>(   \
        static_cast<dtype*>(inout.data),                                \
        static_cast<const int64_t*>(indices.data),                      \
        dev_desc, dim, indices.ndim, numel_inds, 6                      \
    )
    if (inout.dtype.code == kDLFloat) {
        LAUNCH_SCATTER_KERNEL(float);
    } else if (inout.dtype.code == kDLInt) {
        LAUNCH_SCATTER_KERNEL(int);
    } else {
        THROW_IN_HOST("Unsupported data type.");
    }
#undef LAUNCH_SCATTER_KERNEL
}


};  // end namespace fab