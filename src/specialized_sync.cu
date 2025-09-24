#include "cuda_utils.cuh"
#include "scheduler.cuh"
#include "unified_tensor.hpp"

static __device__ int gmem_work_cnt[1];

namespace fab {

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

template <int NT_producer, int NT_Consumer>
using PPTScheduler = SimpleDynamicScheduler<NT_producer, NT_Consumer>;

template <int NT_producer, int NT_Consumer>
using PPTXScheduler = DualPreemptivePersistentTileExecutionScheduler<NT_producer, NT_Consumer>;

static constexpr bool using_pptx = true;

__global__ void SpecializedProducerConsumer(
    const int* const __restrict__ src, 
    int* const __restrict__ dst, 
    int array_len
) {
    __shared__ int shared_buffer[2];
    int tid = threadIdx.x;
    int warp_id = tid >> 5;         // div by 32
    int warp_group_id = tid >> 7;   // div by 128
    const int num_works = array_len;
    int* const work_cnt_ptr = gmem_work_cnt;
    int* const smem_ptr = shared_buffer;

    
    if (threadIdx.x == 0) {
        if constexpr (using_pptx) {
            gmem_work_cnt[0] = gridDim.x * 2;
            shared_buffer[0] = -1;
            shared_buffer[1] = -1;
        } else {
            gmem_work_cnt[0] = gridDim.x;
            shared_buffer[0] = 0;
            shared_buffer[1] = 0;
        }
    }

    PPTXScheduler<96, 288> scheduler(num_works, work_cnt_ptr, smem_ptr);
    __syncthreads();

    if (warp_group_id == 0 && warp_id > 0) {
        // producer, with 96 threads
        for (int work_id = scheduler.get_initial_work<true>(); scheduler.is_valid(work_id); work_id = scheduler.get_next_work<true>(work_id)) {
            scheduler.prefetch_next_work(work_id);
            if (threadIdx.x == 96) {
                printf("(%03d/%03d) Current Producer job: %d / %d\n", blockIdx.x, gridDim.x, work_id, array_len);
            }
        }
        if (threadIdx.x == 32)
            printf("(%03d/%03d) Producer quitted\n", blockIdx.x, gridDim.x);
    } else {
        // consumer, with 288 threads
        scheduler.init_consumer();
        for (int work_id = scheduler.get_initial_work<false>(); scheduler.is_valid(work_id); work_id = scheduler.get_next_work<false>(work_id)) {
            __nanosleep(10000);
            for (int i = 0; i < 4; i++) {
                dst[i + work_id * 4] = src[work_id]; 
            }
            if (threadIdx.x == 128) {
                printf("(%03d/%03d) Current Consumer job: %d / %d\n", blockIdx.x, gridDim.x, work_id, array_len);
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

    int numel = in_dl.strides[0] * in_dl.shape[0];
    int num_blocks = get_grid_size();
    SpecializedProducerConsumer<<<num_blocks, 384>>>(
        static_cast<int*>(in_dl.data), static_cast<int*>(out_dl.data), numel);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

}   // end namespace fab