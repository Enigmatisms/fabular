cmake_minimum_required(VERSION 3.18)
project(dlpack_unified LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Development.Module Interpreter)
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python executable: ${Python_EXECUTABLE}")


if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"
    OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug"
    OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    message(STATUS "Build Type: Debug")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
    message(STATUS "Build Type: Release")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/wjakob/nanobind.git external/nanobind
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(external/nanobind)
add_subdirectory(external/dlpack)
add_subdirectory(external/pybind11)
set(dlpack_INCLUDE_DIR external/dlpack/include)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
    ${nanobind_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIR}
    ${dlpack_INCLUDE_DIR}
)

message(STATUS ${dlpack_INCLUDE_DIR})

set(MODULE_NAME fabular)
set(SOURCES 
    src/unified_tensor.cu
    src/put_along_axis.cu
    src/bindings.cc
)

nanobind_add_module(${MODULE_NAME} ${SOURCES})

set(PYBIND_SOURCES
    src/unified_tensor.cu
    src/put_along_axis.cu
    src/bindings_py11.cc
)

pybind11_add_module(${MODULE_NAME}_py11 ${PYBIND_SOURCES})

if(MSVC)
    set(MODULE_SUFFIX ".pyd")
else()
    set(MODULE_SUFFIX ".so")
endif()

set_target_properties(${MODULE_NAME} PROPERTIES
    OUTPUT_NAME ${MODULE_NAME}
    PREFIX ""
    SUFFIX ${MODULE_SUFFIX}
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(${MODULE_NAME}_py11 PROPERTIES
    OUTPUT_NAME ${MODULE_NAME}_py11
    PREFIX ""
    SUFFIX ${MODULE_SUFFIX}
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_custom_target(run_test
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py
    DEPENDS ${MODULE_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Python tests"
)