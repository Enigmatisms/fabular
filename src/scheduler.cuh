#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "logging.h"

namespace fab {

enum Barrier {
    SmemEmpty = 6,
    SmemFull = 7,
    ProducerSync = 9,
    Consumer1Sync = 10,
    Consumer2Sync = 11,
    SmemEmptyDual = 14,
    SmemFullDual = 15
};

__device__ __forceinline__
void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id) {
    if (threadIdx.x == 0) {
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

template <int NumProducerThreads=96, int NumConsumerThreads=288, int Stride=1>
class SimpleDynamicScheduler {
private:
    const int num_works;
    int* const work_cnt_ptr;
    int* const smem_ptr;
    int inner_step;

    static constexpr int NumThreads = NumProducerThreads + NumConsumerThreads;
public:
    static constexpr int stride = Stride;
    __device__ SimpleDynamicScheduler(
        const int num_works_,
        int* const work_cnt_ptr_,
        int* const smem_ptr_
    ): num_works(num_works_), work_cnt_ptr(work_cnt_ptr_), smem_ptr(smem_ptr_), inner_step(0) { }

    template <bool IsProducerWarp=false>
    __device__ int get_initial_work() {
        if constexpr (Stride > 1) {
            inner_step = (inner_step + 1) % Stride;
        }
        return int(blockIdx.x) * stride;
    }

    __device__ __forceinline__ int is_valid(int work_id) const {
        return work_id < num_works;
    }

    __device__ int init_consumer() const {
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
    }

    __device__ void prefetch_next_work(int& current_work_id) {
        if (threadIdx.x == 96 && (Stride == 1 || inner_step == 0)) {
            current_work_id = atomicAdd(work_cnt_ptr, stride);
        }
    }

    __device__
    constexpr bool skippable() const {
        return (Stride > 1) && (inner_step == 0); 
    }

    template <bool IsProducerWarp=false>
    __device__ int get_next_work(int current_work_id) {
        // bar.sync: blocking until enough threads arrives at this barrier. Threads arrived directly will add to counter
        // bar.arrive: non-blocking, only increase the counter.
        if constexpr (Stride > 1) {
            inner_step = (inner_step + 1) % Stride;
            if (inner_step == 0) {
                return current_work_id + 1;
            }
        }
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

    __device__
    void
    producer_notify() const {}

    __device__
    void
    consumer_notify() const {}
};

template<int NumProducerThreads=64, int NumConsumerThreads=320>
class DualPreemptivePersistentTileExecutionScheduler {
    static constexpr int NumThreads = NumConsumerThreads + NumProducerThreads;
protected:
    const int num_works;
    int* const work_cnt_ptr;
    int* const smem_ptr;
    uint32_t sch_stage_;
public:
    static constexpr int stride = 1;
    __device__ DualPreemptivePersistentTileExecutionScheduler(
        const int num_works_,
        int* const work_cnt_ptr_,
        int* const smem_ptr_
    ): num_works(num_works_), work_cnt_ptr(work_cnt_ptr_), smem_ptr(smem_ptr_) {}

    template<bool IsProducerWarp=false>
    __device__ int get_initial_work() {
        // when all the blocks (SMs) done initializing and no SM has done the first task, tile_count_semaphore will be
        // at least `gridDim.x`, then, we just let prefetch_next_work and non-deterministic schedule (workload-related) take over 

        // For FlashMask V2, only generate_n_block pipeline is the big brother producer to be preemptively scheduled!
        // since the initial work is assigned deterministically via blockIdx.x, we need to ensure that the initial state of
        // tile_count_semaphore is gridDim.x. Can't use atomicAdd here, since if we do, for example, SM1 is really fast, it performs
        // prefetch_next_work even before SM2 calls get_initial_work, then SM1 will risk computing the same block as SM2.

        sch_stage_ = 0;
        if constexpr (IsProducerWarp) {
            if (threadIdx.x == NumProducerThreads) {
                smem_ptr[0] = atomicAdd(work_cnt_ptr, 1);
            }
            // make sure the smem update is visible to all warps
            named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(Barrier::ProducerSync));
            return {smem_ptr[0]};
        } else {
            // wait the notify of producer (wait full 0)
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
            return {smem_ptr[0]};
        }
    }

    __device__ __forceinline__ int is_valid(int work_id) const {
        return work_id < num_works;
    }

    __device__ void init_consumer() const {
        // notify the producer that pos 1 is ready to be filled (empty 1)
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmptyDual) /*id*/);
    }

    __device__ void prefetch_next_work(int& current_work_id) const {
        // PPTX prefetch is moved to consumer for more exact delay scheduling
    }

    __device__
    constexpr bool skippable() const {
        return false; 
    }

    template<bool IsProducerWarp=false>
    __device__ int get_next_work(int current_work_id) {
        // change state immediately, since we are to get next work
        // Note that for the return value: except from the initial work, PPT always dynamic schedules
        // Dual PPTX will have static schedule for only twice: get initial work and the first time get_next_work
        // This is intentional, since in the first get_next_work, smem is not fully ready.
        if constexpr (IsProducerWarp) {
            // for example: 
            // the 1st get_next_work of consumer: load from 1, and atomicAdd store to 0 
            //      load from 1 not initialized, use blockIdx.x + gridDim.x (static scheduling)
            // the 2nd get_next_work of consumer: load from 0, and atomicAdd store to 1
            //      load from 0 initialized: the 3rd consumer work ID is correctly set 
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) + (sch_stage_ << 3) /*id*/);
            if (threadIdx.x == NumProducerThreads) {    // thread 288 hard-coded, since n_block consumer threads are in [128, 384)
                smem_ptr[sch_stage_] = atomicAdd(work_cnt_ptr, 1);
            }
            named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(Barrier::ProducerSync));
            return smem_ptr[sch_stage_];
        } else {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) + (sch_stage_ << 3) /*id*/);
            // Sync all the producers in case some of the producers return before the smem is updated
            return smem_ptr[sch_stage_];
        }
    }

    template<bool IsProducerWarp=false>
    __device__ uint32_t stage() const noexcept {
        return sch_stage_;
    }

    __device__
    void
    producer_notify() {
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) + (sch_stage_ << 3) /*id*/);
        sch_stage_ = 1 - sch_stage_;
    }

    __device__
    void
    consumer_notify() {
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) + (sch_stage_ << 3) /*id*/);
        sch_stage_ = 1 - sch_stage_;
    }
};


template<int NumProducerThreads=128, int NumConsumerThreads = 256>
class BwdPreemptivePersistentTileScheduler {
    const int num_works;
    int* const work_cnt_ptr;
    int* const smem_ptr;

    static constexpr int NumThreads = NumProducerThreads + NumConsumerThreads;
public:
    static constexpr int stride = 1;
    __device__ BwdPreemptivePersistentTileScheduler(
        const int num_works_,
        int* const work_cnt_ptr_,
        int* const smem_ptr_
    ): num_works(num_works_), work_cnt_ptr(work_cnt_ptr_), smem_ptr(smem_ptr_) { }

    template<bool IsProducerWarp=false>
    __device__
    int get_initial_work() const {
        if constexpr (!IsProducerWarp) {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
        }
        return int(blockIdx.x);
    }

    __device__ void init_consumer() const {}

    __device__ void prefetch_next_work(int& ) const {}

    __device__ __forceinline__ int is_valid(int work_id) const {
        return work_id < num_works;
    }

    __device__
    constexpr bool skippable() const {
        return false; 
    }

    __device__
    void
    producer_notify() const {     // notify the consumer that we've written data into the buffer
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
    }

    __device__
    void
    consumer_notify() const {
        // sync to make sure (*tile_count_smem) modification is visible to consumers
        named_barrier_arrive(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
    }

    template<bool IsProducerWarp=false>
    __device__
    int get_next_work(int current_work) const {
        if constexpr (IsProducerWarp) {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemEmpty) /*id*/);
            if (threadIdx.x == 0) {    // hard-coded, since n_block producer threads are in [32, 128)
                // the next job we are going to process: number of currently blocks done
                *smem_ptr = atomicAdd(work_cnt_ptr, 1);
            }
            named_barrier_sync(NumProducerThreads, static_cast<uint32_t>(Barrier::ProducerSync) /*id*/);
        } else {
            named_barrier_sync(NumThreads, static_cast<uint32_t>(Barrier::SmemFull) /*id*/);
        }
        return *smem_ptr;
    }

    template<bool IsProducerWarp=false>
    __device__
    constexpr uint32_t stage() const noexcept { return 0; }
};


}   // end namespace fab