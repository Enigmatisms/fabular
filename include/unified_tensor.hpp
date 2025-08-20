#pragma once
#include "dlpack/dlpack.h"
#include <string>

namespace fab {

class TensorProcessor {
public:
    void print(const DLManagedTensor* tensor);
    int process(const DLManagedTensor* tensor);
private:
    void print_tensor_info(const DLTensor* dlt);
    void print_elements(void* data, DLDataType dtype, size_t count);
    void print_first_elements(const DLTensor* dlt);
};

}   // end namespace fab