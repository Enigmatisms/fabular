#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // single node only needs `nvshmem_init`
    nvshmem_init();

    // processing element (PE) ID, means the `rank`
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int dev_id = my_pe;
    cudaSetDevice(dev_id);

    printf("Hello from PE %d of %d (using GPU %d)\n", my_pe, n_pes, dev_id);

    nvshmem_barrier_all();
    nvshmem_finalize();
    return 0;
}