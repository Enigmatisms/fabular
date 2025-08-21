#include "unified_tensor.hpp"
#include "cuda_utils.cuh"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace fab {

void TensorProcessor::print(const DLManagedTensor* tensor) {
    const DLTensor* dlt = &tensor->dl_tensor;
    print_tensor_info(dlt);
    print_first_elements(dlt);
}

int TensorProcessor::process(const DLManagedTensor* tensor) {
    // Currently: a very simple function, just used for testing speed
    const DLTensor* dlt = &tensor->dl_tensor;
    return dlt->ndim;
}

void TensorProcessor::print_tensor_info(const DLTensor* dlt) {
    printf("=== Unified Tensor Info ===\n");
    
    std::string device_str;
    switch (dlt->device.device_type) {
        case kDLCPU: device_str = "CPU"; break;
        case kDLCUDA: device_str = "CUDA"; break;
        case kDLROCM: device_str = "ROCM"; break;
        default: device_str = "Other";
    }
    printf("Device: %s (%d)\n", device_str.c_str(), dlt->device.device_id);
    
    printf("Dimensions: %d [", dlt->ndim);
    for (int i = 0; i < dlt->ndim; ++i) {
        printf("%lld", (long long)dlt->shape[i]);
        if (i < dlt->ndim - 1) printf(", ");
    }
    printf("]\n");

    printf("Strides: %d [", dlt->ndim);
    for (int i = 0; i < dlt->ndim; ++i) {
        printf("%lld", (long long)dlt->strides[i]);
        if (i < dlt->ndim - 1) printf(", ");
    }
    printf("]\n");
    
    std::string dtype_str;
    if (dlt->dtype.code == kDLFloat) {
        dtype_str = "float";
    } else if (dlt->dtype.code == kDLInt) {
        dtype_str = "int";
    } else if (dlt->dtype.code == kDLUInt) {
        dtype_str = "uint";
    } else {
        dtype_str = "unknown";
    }
    printf("Data Type: %s%d\n", dtype_str.c_str(), dlt->dtype.bits);
}

void TensorProcessor::print_elements(void* data, DLDataType dtype, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        char* ptr = static_cast<char*>(data) + i * (dtype.bits / 8);
        if (dtype.code == kDLFloat && dtype.bits == 32) {
            if (dtype.bits == 32) {
                printf("%.2f ", *reinterpret_cast<float*>(ptr));
            } else if (dtype.bits == 64) {
                printf("%.4lf ", *reinterpret_cast<double*>(ptr));
            } else {
                THROW_IN_HOST("Not currently supported float type (supported bits: 32 & 64)")
            }
        } else if (dtype.code == kDLInt && dtype.bits == 64) {
            if (dtype.bits == 32) {
                printf("%d ", *reinterpret_cast<int*>(ptr));
            } else if (dtype.bits == 64) {
                printf("%lld ", *reinterpret_cast<int64_t*>(ptr));
            } else {
                THROW_IN_HOST("Not currently supported int type (supported bits: 32 & 64)")
            }
        } else {
                THROW_IN_HOST("Dtype Not currently supported.")
        }
    }
    printf("\n\n");
}

void TensorProcessor::print_first_elements(const DLTensor* dlt) {
    size_t num_elements = 1;
    for (int i = 0; i < dlt->ndim; ++i) {
        num_elements *= dlt->shape[i];
    }
    size_t elements_to_print = std::min(num_elements, static_cast<size_t>(5));
    
    if (dlt->device.device_type == kDLCUDA) {
        std::vector<char> host_buffer(elements_to_print * (dlt->dtype.bits / 8));
        
        CUDA_CHECK_RETURN(cudaMemcpy(
            host_buffer.data(), 
            static_cast<const char*>(dlt->data), 
            host_buffer.size(), 
            cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        printf("First %zu elements (GPU): ", elements_to_print);
        print_elements(host_buffer.data(), dlt->dtype, elements_to_print);
    } 
    else if (dlt->device.device_type == kDLCPU || dlt->device.device_type == kDLCUDAManaged) {
        printf("First %zu elements (CPU): ", elements_to_print);
        print_elements(dlt->data, dlt->dtype, elements_to_print);
    }
}

}   // end namespace fab