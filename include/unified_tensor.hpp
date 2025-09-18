#pragma once
#include "dlpack/dlpack.h"
#include <string>

namespace fab {

class TensorProcessor {
public:
    void print(const DLManagedTensor* tensor) const;
    int process(const DLManagedTensor* tensor) const;
    void put_along_axis(DLManagedTensor* inout_m, const DLManagedTensor* indices_m, int dim) const;
    void assign_reduce(const DLManagedTensor* src, DLManagedTensor* dst) const;
    void min_last_dim(const DLManagedTensor* _src, DLManagedTensor* _dst) const;
    void max_last_dim(const DLManagedTensor* _src, DLManagedTensor* _dst) const;
    void unordered_expand(const DLManagedTensor* in_tensor, DLManagedTensor* out_tensor) const;
private:
    void print_tensor_info(const DLTensor* dlt) const;
    void print_elements(void* data, DLDataType dtype, size_t count) const;
    void print_first_elements(const DLTensor* dlt) const;
};

}   // end namespace fab