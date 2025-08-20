#pragma once
#include <cstdio>
#include <stdexcept>

__host__ static void CheckCudaErrorAux(const char *, unsigned, const char *,
                                       cudaError_t);
#define CUDA_CHECK_RETURN(value)                                               \
    CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

__host__ static void CheckCudaErrorAux(const char *file, unsigned line,
                                       const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err),
           err, file, line);
    exit(1);
}

#define THROW_IN_GLOBAL(msg, ...)                                                       \
    if (threadIdx.x == 0 &&                                                             \
        threadIdx.y == 0 &&                                                             \
        threadIdx.z == 0 &&                                                             \
        blockIdx.x == 0 &&                                                              \
        blockIdx.y == 0 &&                                                              \
        blockIdx.z == 0                                                                 \
    ) {                                                                                 \
        std::fprintf(stderr, "[ERR] %s:%d: " msg, __FILE__, __LINE__, ##__VA_ARGS__);   \
    }                                                                                   \
    asm("trap;");

#define THROW_IN_HOST(msg, ...)                                                         \
    do {                                                                                \
        std::fprintf(stderr, "[ERR] %s:%d: " msg, __FILE__, __LINE__, ##__VA_ARGS__);   \
        throw std::runtime_error("Error occur in host");                                \
    } while (0);
