#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "logging.h"

namespace fab {

enum Barrier {
    SmemEmpty = 0,
    SmemFull = 1,
    ProducerSync = 2,
    SmemEmptyDual = 8,
    SmemFullDual = 9
};

__device__ __forceinline__
void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id) {
    if (threadIdx.x == 96) {
        ERROR_PRINT("Producer sync: %d threads, barrier id: %d\n", num_threads, barrier_id);
    } else if (threadIdx.x == 128) {
        ERROR_PRINT("Consumer sync: %d threads, barrier id: %d\n", num_threads, barrier_id);
    }
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

    template <bool IsProducerWarp=false>
    __device__ int get_initial_work() const {
        return int(blockIdx.x);
    }

    __device__ __forceinline__ int is_valid(int work_id) const {
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

    template<bool IsProducerWarp=false>
    __device__ uint32_t stage() const noexcept { return 0; }
};

template<int NumProducerThreads=96, int NumConsumerThreads=288>
class DualPreemptivePersistentTileExecutionScheduler {
    // **PPT** scheduler: performs correct synchronization for producer (generate_n_block) and consumer (KV load and computation pipeline)
    // This scheduler has the same coordinate computation logic as StaticPersistentTileSch, the difference is that
    // we employ a preemptive scheduling strategy based on a rough estimation of the workload for the consumer
    // In PPT, NumConsumerThreads is the total number of threads for (KV load and computation pipeline), and for FlashMask V2
    // it will be the #threads for (wg_id = 0, wp_id = 0) + (wg_id > 0, wp_id = *). The NumProducerThreads is simply 96 (hard-coded).
    static_assert(NumProducerThreads == 96, "DualPPTX Scheduler has incorrect producer thread num.");
    static constexpr int NumThreads = NumConsumerThreads + NumProducerThreads;
protected:
    const int num_works;
    int* const work_cnt_ptr;
    int* const smem_ptr;
    uint32_t sch_stage_;
public:
    __device__ DualPreemptivePersistentTileExecutionScheduler(
        const int num_works_,
        int* const work_cnt_ptr_,
        int* const smem_ptr_
    ): num_works(num_works_), work_cnt_ptr(work_cnt_ptr_), smem_ptr(smem_ptr_) { }

    template<bool IsProducerWarp=false>
    __device__ int get_initial_work() {
        // when all the blocks (SMs) done initializing and no SM has done the first task, tile_count_semaphore will be
        // at least `gridDim.x`, then, we just let prefetch_next_work and non-deterministic schedule (workload-related) take over 

        // For FlashMask V2, only generate_n_block pipeline is the big brother producer to be preemptively scheduled!
        // since the initial work is assigned deterministically via blockIdx.x, we need to ensure that the initial state of
        // tile_count_semaphore is gridDim.x. Can't use atomicAdd here, since if we do, for example, SM1 is really fast, it performs
        // prefetch_next_work even before SM2 calls get_initial_work, then SM1 will risk computing the same block as SM2.

        // for the initial work: assign deterministically
        if constexpr (IsProducerWarp) {
            sch_stage_ = 0;  // producer initial state is 0, since the first get_next, producer should sync full-1 (dual)
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
        } else {
            sch_stage_ = 1;  // consumer initial state is 1, since the first get_next, producer should sync empty-0 (non-dual)
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemFullDual) /*id*/);
        }
        if (threadIdx.x == 96 || threadIdx.x == 128) {
            DEBUG_PRINT("Block: %d, Current initial stage: %u, is_producer: %d\n", blockIdx.x, sch_stage_, int(IsProducerWarp));
        }
        return int(blockIdx.x);
    }

    __device__ __forceinline__ int is_valid(int work_id) const {
        return work_id < num_works;
    }

    __device__ void init_consumer() const { /* Init is done in get_initial work, therefore no need to repeat. */ }

    __device__ void prefetch_next_work(int& current_work_id) const {
        // PPTX prefetch is moved to consumer for more exact delay scheduling
    }

    template<bool IsProducerWarp=false>
    __device__ int get_next_work(int current_work_id) {
        // change state immediately, since we are to get next work
        // Note that for the return value: except from the initial work, PPT always dynamic schedules
        // Dual PPTX will have static schedule for only twice: get initial work and the first time get_next_work
        // This is intentional, since in the first get_next_work, smem is not fully ready.
        if constexpr (IsProducerWarp) {
            sch_stage_ = 0x1 ^ sch_stage_;
            
            if (threadIdx.x == 96) {
                WARN_PRINT("Block: %d, warp id: %d, Producer syncing, stage: %u\n", blockIdx.x, threadIdx.x / 32, sch_stage_);
            }
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) + (sch_stage_ << 3) /*id*/);
            int tile_idx = smem_ptr[sch_stage_];
            if (threadIdx.x == 96) {
                WARN_PRINT("Block: %d, warp id: %d, Producer arrives.\n", blockIdx.x, threadIdx.x / 32);
            }
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) + (sch_stage_ << 3) /*id*/);
            // Sync all the producers in case some of the producers return before the smem is updated
            return {tile_idx >= 0 ? tile_idx : int(blockIdx.x + gridDim.x)};
        } else {
            // for example: 
            // the 1st get_next_work of consumer: load from 1, and atomicAdd store to 0 
            //      load from 1 not initialized, use blockIdx.x + gridDim.x (static scheduling)
            // the 2nd get_next_work of consumer: load from 0, and atomicAdd store to 1
            //      load from 0 initialized: the 3rd consumer work ID is correctly set 
            int tile_idx = smem_ptr[sch_stage_];
            sch_stage_ = 0x1 ^ sch_stage_;
            if (threadIdx.x == 128) {
                WARN_PRINT("Block: %d, warp id: %d, Consumer syncing, stage: %u\n", blockIdx.x, threadIdx.x / 32, sch_stage_);
            }
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) + (sch_stage_ << 3) /*id*/);
            if (threadIdx.x == NumConsumerThreads) {    // thread 288 hard-coded, since n_block consumer threads are in [128, 384)
                smem_ptr[sch_stage_] = atomicAdd(work_cnt_ptr, 1);
            }
            if (threadIdx.x == 128) {
                WARN_PRINT("Block: %d, warp id: %d, Consumer arrives.\n", blockIdx.x, threadIdx.x / 32);
            }
            named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) + (sch_stage_ << 3) /*id*/);
            return {tile_idx >= 0 ? tile_idx : int(blockIdx.x + gridDim.x)};
        }
    }

    template<bool IsProducerWarp=false>
    __device__ uint32_t stage() const noexcept {
        // producer always returns the current stage, while consumer returns 1 - current stage
        // so that consumer can always have valid input
        if constexpr (IsProducerWarp)
            return sch_stage_;
        else
            return 0x1 ^ sch_stage_;
    }
};

}   // end namespace fab