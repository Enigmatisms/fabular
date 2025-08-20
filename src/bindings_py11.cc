#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "unified_tensor.hpp"  // 包含 TensorProcessor 的声明

namespace py = pybind11;

void print_dlpack(py::capsule dlpack_capsule) {
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(dlpack_capsule);
    
    fab::TensorProcessor processor;
    processor.print(managed);
}

int process_dlpack(py::capsule dlpack_capsule) {
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(dlpack_capsule);
    
    fab::TensorProcessor processor;
    return processor.process(managed);
}

PYBIND11_MODULE(fabular_py11, m) {
    m.doc() = "DLPack unified tensor processing with pybind11";
    
    m.def("process_dlpack", &process_dlpack, 
          "Process DLPack tensor from any framework");

    m.def("print_dlpack", &print_dlpack, 
          "Printing DLPack tensor from any framework");
    
    m.attr("kDLCPU") = static_cast<int>(kDLCPU);
    m.attr("kDLCUDA") = static_cast<int>(kDLCUDA);
    m.attr("kDLCUDAManaged") = static_cast<int>(kDLCUDAManaged);
    m.attr("kDLOpenCL") = static_cast<int>(kDLOpenCL);
    m.attr("kDLVulkan") = static_cast<int>(kDLVulkan);
    m.attr("kDLMetal") = static_cast<int>(kDLMetal);
    m.attr("kDLVPI") = static_cast<int>(kDLVPI);
    m.attr("kDLROCM") = static_cast<int>(kDLROCM);
    m.attr("kDLExtDev") = static_cast<int>(kDLExtDev);
}