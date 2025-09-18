#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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

void atomic_assign(const nb::capsule& src, nb::capsule& dst) {
    const DLManagedTensor* managed_src = static_cast<const DLManagedTensor*>(src.data());
    DLManagedTensor* managed_dst = static_cast<DLManagedTensor*>(dst.data());
    
    fab::TensorProcessor processor;
    return processor.assign_reduce(
        managed_src, 
        managed_dst
    );
}

#define MINMAX_LAST_DIM_FUNC(op)                                                            \
void op##_last_dim(const nb::capsule& src, nb::capsule& dst) {                              \
    const DLManagedTensor* managed_src = static_cast<const DLManagedTensor*>(src.data());   \
    DLManagedTensor* managed_dst = static_cast<DLManagedTensor*>(dst.data());               \
    fab::TensorProcessor processor;                                                         \
    return processor.op##_last_dim(                                                         \
        managed_src,                                                                        \
        managed_dst                                                                         \
    );                                                                                      \
}

void unordered_expand(const nb::capsule& src, nb::capsule& dst) {
    const DLManagedTensor* managed_src = static_cast<const DLManagedTensor*>(src.data());
    DLManagedTensor* managed_dst = static_cast<DLManagedTensor*>(dst.data());
    
    fab::TensorProcessor processor;
    return processor.unordered_expand(
        managed_src, 
        managed_dst
    );
}

MINMAX_LAST_DIM_FUNC(min)
MINMAX_LAST_DIM_FUNC(max)


class DataHolder {
public:
    int int_val;
    float float_val;
    bool bool_val;

    DataHolder() : int_val(0), float_val(0), bool_val(false) {}
    DataHolder(int i, float f, bool b) : int_val(i), float_val(f), bool_val(b) {}

    std::string describe() const {
        return "DataHolder(int=" + std::to_string(int_val) + 
               ", float=" + std::to_string(float_val) + 
               ", bool=" + (bool_val ? "true" : "false") + ")";
    }

    void multiply(float factor) {
        int_val *= static_cast<int>(factor);
        float_val *= factor;
    }
};

class DataHolderDummy {
public:
    int int_val;
    float float_val;
    bool bool_val;

    DataHolderDummy() : int_val(0), float_val(0), bool_val(false) {}
    DataHolderDummy(int i, float f, bool b) : int_val(i), float_val(f), bool_val(b) {}

    std::string describe() const {
        return "DataHolderDummy(int=" + std::to_string(int_val) + 
               ", float=" + std::to_string(float_val) + 
               ", bool=" + (bool_val ? "true" : "false") + ")";
    }

    void multiply(float factor) {
        int_val *= static_cast<int>(factor);
        float_val *= factor;
    }
};


DataHolder dynamic_data_holder_getter(nb::object obj, bool forced = false) {
    // nb::ndarray<> arr;
    auto CastToDataHolder = [](nb::object& obj) {
        DataHolder res;
        auto data = obj.ptr();
        if (!nb::try_cast<DataHolder>(obj, res))
            throw std::runtime_error("Object must be a DataHolder");
        return res;
    };
    auto ForceCastToDataHolder = [](nb::object& obj) {
        return nb::cast<DataHolder>(obj, true);
    };
    if (forced) 
        return ForceCastToDataHolder(obj);
    return CastToDataHolder(obj);
}

NB_MODULE(fabular, m) {
    m.doc() = "DLPack unified tensor processing with nanobind";

    m.def("process_dlpack", &process_dlpack, 
          "Process DLPack tensor from any framework");

    m.def("print_dlpack", &print_dlpack, 
          "Printing DLPack tensor from any framework");

    m.def("put_along_axis", &put_along_axis, 
          "Put along axis minimalist implementation for testing");

    m.def("atomic_assign", &atomic_assign, 
          "Assign atomic primitives testing API");

    m.def("min_last_dim", &min_last_dim, 
          "Atomic min over the last dimenstion (for atomic min testing)");

    m.def("max_last_dim", &max_last_dim, 
          "Atomic max over the last dimenstion (for atomic max testing)");

    m.def("unordered_expand", &unordered_expand, 
          "Expand the tensor by 4 times, but with unordered copy");
    
    m.attr("kDLCPU") = nb::int_(static_cast<int>(kDLCPU));
    m.attr("kDLCUDA") = nb::int_(static_cast<int>(kDLCUDA));
    m.attr("kDLCUDAManaged") = nb::int_(static_cast<int>(kDLCUDAManaged));
    m.attr("kDLOpenCL") = nb::int_(static_cast<int>(kDLOpenCL));
    m.attr("kDLVulkan") = nb::int_(static_cast<int>(kDLVulkan));
    m.attr("kDLMetal") = nb::int_(static_cast<int>(kDLMetal));
    m.attr("kDLVPI") = nb::int_(static_cast<int>(kDLVPI));
    m.attr("kDLROCM") = nb::int_(static_cast<int>(kDLROCM));
    m.attr("kDLExtDev") = nb::int_(static_cast<int>(kDLExtDev));

    nb::class_<DataHolder>(m, "DataHolder")
        .def(nb::init<>())
        .def(nb::init<int, float, bool>())
        .def_rw("int_val", &DataHolder::int_val)
        .def_rw("float_val", &DataHolder::float_val)
        .def_rw("bool_val", &DataHolder::bool_val)
        .def("describe", &DataHolder::describe)
        .def("multiply", &DataHolder::multiply);

    nb::class_<DataHolderDummy>(m, "DataHolderDummy")
        .def(nb::init<>())
        .def(nb::init<int, float, bool>())
        .def_rw("int_val", &DataHolderDummy::int_val)
        .def_rw("float_val", &DataHolderDummy::float_val)
        .def_rw("bool_val", &DataHolderDummy::bool_val)
        .def("describe", &DataHolderDummy::describe)
        .def("multiply", &DataHolderDummy::multiply);

    m.def("get_data_holder_dyn", &dynamic_data_holder_getter, 
          "Process DLPack tensor from any framework");
}