#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "unified_tensor.hpp"

namespace nb = nanobind;

void process_dlpack(nb::capsule dlpack_capsule) {
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(dlpack_capsule.data());
    
    fab::TensorProcessor processor;
    processor.process(managed);
}

NB_MODULE(fabular, m) {
    m.def("process_dlpack", &process_dlpack, 
          "Process DLPack tensor from any framework");
    
    m.attr("kDLCPU") = nb::int_(static_cast<int>(kDLCPU));
    m.attr("kDLCUDA") = nb::int_(static_cast<int>(kDLCUDA));
    // TODO: add more devices?
}