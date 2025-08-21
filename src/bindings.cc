#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "unified_tensor.hpp"

namespace nb = nanobind;

void print_dlpack(nb::capsule dlpack_capsule) {
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(dlpack_capsule.data());
    
    fab::TensorProcessor processor;
    processor.print(managed);
}

int process_dlpack(nb::capsule dlpack_capsule) {
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(dlpack_capsule.data());
    
    fab::TensorProcessor processor;
    return processor.process(managed);
}

void put_along_axis(nb::capsule inout_cap, nb::capsule indices_cap, int dim) {
    DLManagedTensor* managed_inout = static_cast<DLManagedTensor*>(inout_cap.data());
    DLManagedTensor* managed_inds = static_cast<DLManagedTensor*>(indices_cap.data());
    
    fab::TensorProcessor processor;
    return processor.put_along_axis(
        managed_inout, 
        managed_inds, 
        dim
    );
}

NB_MODULE(fabular, m) {
    m.doc() = "DLPack unified tensor processing with nanobind";

    m.def("process_dlpack", &process_dlpack, 
          "Process DLPack tensor from any framework");

    m.def("print_dlpack", &print_dlpack, 
          "Printing DLPack tensor from any framework");

    m.def("put_along_axis", &put_along_axis, 
          "Put along axis minimalist implementation for testing");
    
    m.attr("kDLCPU") = nb::int_(static_cast<int>(kDLCPU));
    m.attr("kDLCUDA") = nb::int_(static_cast<int>(kDLCUDA));
    m.attr("kDLCUDAManaged") = nb::int_(static_cast<int>(kDLCUDAManaged));
    m.attr("kDLOpenCL") = nb::int_(static_cast<int>(kDLOpenCL));
    m.attr("kDLVulkan") = nb::int_(static_cast<int>(kDLVulkan));
    m.attr("kDLMetal") = nb::int_(static_cast<int>(kDLMetal));
    m.attr("kDLVPI") = nb::int_(static_cast<int>(kDLVPI));
    m.attr("kDLROCM") = nb::int_(static_cast<int>(kDLROCM));
    m.attr("kDLExtDev") = nb::int_(static_cast<int>(kDLExtDev));
}