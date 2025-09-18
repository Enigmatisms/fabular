#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "unified_tensor.hpp"

static __device__ int gmem_work_cnt[1];

namespace fab {

enum Barrier {
    SmemEmpty,
    SmemFull,
    ProducerSync
};

__host__ static int get_grid_size() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return prop.multiProcessorCount;
} 

__device__ __forceinline__
void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}


__device__ __forceinline__
void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id) {
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

template <int NumProducerThreads=96, int NumConsumerThreads=288>
class SimpleDynamicScheduler {
private:
    const int num_works;
    int* const work_cnt_ptr;
    int* const smem_ptr;

    static constexpr int NumThreads = NumProducerThreads + NumConsumerThreads;
public:
    __device__ SimpleDynamicScheduler(
        const int num_works_,
        int* const work_cnt_ptr_,
        int* const smem_ptr_
    ): num_works(num_works_), work_cnt_ptr(work_cnt_ptr_), smem_ptr(smem_ptr_) { }

    __device__ int get_initial_work() const {
        return int(blockIdx.x);
    }

    __device__ int is_valid(int work_id) const {
        return work_id < num_works;
    }

    __device__ int init_consumer() const {
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
    }

    __device__ void prefetch_next_work(int& current_work_id) {
        if (threadIdx.x == 96) {
            current_work_id = atomicAdd(work_cnt_ptr, 1);
        }
    }

    template <bool IsProducerWarp=false>
    __device__ int get_next_work(int current_work_id) const {
        // bar.sync: blocking until enough threads arrives at this barrier. Threads arrived directly will add to counter
        // bar.arrive: non-blocking, only increase the counter.
        if constexpr (IsProducerWarp) {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
            if (threadIdx.x == 96) {    // hard-coded, since n_block producer threads are in [32, 128)
                *smem_ptr = current_work_id;
            }
            // Sync all the producers
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
            named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(Barrier::ProducerSync) /*id*/);
            return *smem_ptr;
        } else {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
            int work_idx = *smem_ptr;
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
            return work_idx;
        }
    }
};

__global__ void SpecializedProducerConsumer(
    const int* const __restrict__ src, 
    int* const __restrict__ dst, 
    int array_len
) {
    __shared__ int shared_buffer[1];
    int tid = threadIdx.x;
    int warp_id = tid >> 5;         // div by 32
    int warp_group_id = tid >> 7;   // div by 128
    const int num_works = array_len;
    int* const work_cnt_ptr = gmem_work_cnt;
    int* const smem_ptr = shared_buffer;

    if (threadIdx.x == 0) {
        gmem_work_cnt[0] = gridDim.x;
        shared_buffer[0] = 0;
    }
    SimpleDynamicScheduler<96, 288> scheduler(num_works, work_cnt_ptr, smem_ptr);
    __syncthreads();

    if (warp_group_id == 0 && warp_id > 0) {
        // producer, with 96 threads
        for (int work_id = scheduler.get_initial_work(); scheduler.is_valid(work_id); ) {
            __nanosleep(100);
            scheduler.prefetch_next_work(work_id);
            if (threadIdx.x == 96) {
                printf("(%03d/%03d) Current job: %d / %d\n", blockIdx.x, gridDim.x, work_id, array_len);
            }
            work_id = scheduler.get_next_work<true>(work_id);
        }
        if (threadIdx.x == 32)
            printf("(%03d/%03d) Producer quitted\n", blockIdx.x, gridDim.x);
    } else {
        // consumer, with 288 threads
        scheduler.init_consumer();
        for (int work_id = scheduler.get_initial_work(); scheduler.is_valid(work_id); work_id = scheduler.get_next_work<false>(work_id)) {
            for (int i = 0; i < 4; i++) {
                dst[i + work_id * 4] = src[work_id]; 
            }
        }
        if (threadIdx.x == 128)
            printf("(%03d/%03d) Consumer quitted\n", blockIdx.x, gridDim.x);
    }
}

void TensorProcessor::unordered_expand(const DLManagedTensor* in_tensor, DLManagedTensor* out_tensor) const {
    const DLTensor& in_dl = in_tensor->dl_tensor;
    DLTensor& out_dl = out_tensor->dl_tensor;
    if (in_dl.device.device_type != kDLCUDA ||
        out_dl.device.device_type != kDLCUDA
    ) {
        THROW_IN_HOST("Check the input or indices tensor, at least one of them is not a CUDA Tensor.\n")
    } 

    int numel = in_dl.strides[0] * in_dl.shape[0];
    int num_blocks = get_grid_size();
    SpecializedProducerConsumer<<<num_blocks, 384>>>(
        static_cast<int*>(in_dl.data), static_cast<int*>(out_dl.data), numel);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

}   // end namespace fab