/**
 * @file MappedTransferServiceDevice.cuh
 * @brief Graph-bounded CUDA copy service with a device-owned resumable cursor.
 *
 * Native future-event consumers can occupy all queues used by independent copy
 * submissions. A root branch of the inference graph is already admitted and
 * can progress its mapped inbox in that situation. It must not outlive the
 * inference interval: CUDA host-memory reclamation may wait for active kernels.
 * Finite byte quanta, private graph terminal state, and one GPU claimant per
 * inbox provide that bounded lifetime without losing or duplicating a copy.
 */
#pragma once

#include "../../transfer/MappedTransferProgressABI.h"
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace llaminar2
{
    /** Read the device nanosecond timer; SM clocks are not a wall-time authority. */
    __device__ __forceinline__ std::uint64_t mappedTransferNanoseconds()
    {
        std::uint64_t value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    }

    /** GPU-only lifecycle publication ordered in the primary captured branch. */
    __global__ void mappedTransferIntervalKernel(
        std::uint32_t *interval, MappedTransferInterval value)
    {
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_device>(*interval)
            .store(static_cast<std::uint32_t>(value), ::cuda::memory_order_release);
    }

    /**
     * @brief Progress independent inboxes until their private interval closes.
     * @param commands Immutable host publications, one per physical inbox.
     * @param completions GPU release receipts acquired by the maintenance owner.
     * @param cursors Device-only locks and partial byte positions shared by workers.
     * @param capacity Bounded physical inbox count, unrelated to model layer count.
     * @param maximum_bytes Admission-time extent limit for every command.
     * @param interval Private captured graph state; absent only for a finite pass.
     * @param run Typed lifetime; neither mode waits for another worker's lock.
     *
     * Every copying thread system-fences before releasing its claim or publishing
     * completion. A fence only in thread zero does not publish other threads'
     * mapped writes. While retaining a claim, successive 64 KiB quanta need no
     * system fence or PCIe cursor writes: only retirement/completion publishes.
     */
    __global__ void mappedTransferServiceKernel(
        const MappedTransferProgressCommand *commands,
        MappedTransferProgressCompletion *completions,
        MappedTransferServiceCursor *cursors,
        std::size_t capacity, std::size_t maximum_bytes,
        const std::uint32_t *interval, MappedTransferServiceRun run)
    {
        constexpr std::uint64_t quantum = 64u * 1024u;
        __shared__ MappedTransferProgressCommand command;
        __shared__ std::uint64_t position;
        __shared__ std::uint64_t started_nanoseconds;
        __shared__ unsigned admitted;
        __shared__ unsigned closed;
        __shared__ unsigned error;
        do
        {
            for (std::size_t slot = blockIdx.x; slot < capacity; slot += gridDim.x)
            {
                if (threadIdx.x == 0u)
                {
                    closed = run == MappedTransferServiceRun::CapturedInterval &&
                        ::cuda::atomic_ref<const std::uint32_t,
                            ::cuda::thread_scope_device>(*interval)
                            .load(::cuda::memory_order_acquire) ==
                                static_cast<std::uint32_t>(MappedTransferInterval::Closed);
                    admitted = 0u;
                    if (!closed)
                    {
                        auto &cursor = cursors[slot];
                        unsigned expected = 0u;
                        auto lock = ::cuda::atomic_ref<std::uint32_t,
                            ::cuda::thread_scope_device>(cursor.claimed);
                        if (lock.compare_exchange_strong(expected, 1u,
                                ::cuda::memory_order_acquire, ::cuda::memory_order_relaxed))
                        {
                            const auto generation = ::cuda::atomic_ref<const std::uint64_t,
                                ::cuda::thread_scope_system>(commands[slot].generation)
                                .load(::cuda::memory_order_acquire);
                            // Never reread a completed command body: the host is
                            // allowed to publish its successor after the receipt.
                            if (generation != 0u &&
                                (cursor.generation != generation ||
                                 cursor.copied_bytes < cursor.command_bytes))
                            {
                                command = commands[slot];
                                command.generation = generation;
                                error = static_cast<unsigned>(MappedTransferProgressError::None);
                                if (command.generation_magic != (kMappedTransferProgressMagic ^ static_cast<std::uint32_t>(generation)) ||
                                    command.generation_version != (kMappedTransferProgressVersion ^ static_cast<std::uint32_t>(generation >> 32u)) ||
                                    command.source_complement != ~command.source_address ||
                                    command.destination_complement != ~command.destination_address ||
                                    command.bytes_complement != ~command.bytes)
                                    error = static_cast<unsigned>(MappedTransferProgressError::InvalidIdentity);
                                else if (!command.source_address || !command.destination_address)
                                    error = static_cast<unsigned>(MappedTransferProgressError::InvalidAddress);
                                else if (!command.bytes || command.bytes > maximum_bytes)
                                    error = static_cast<unsigned>(MappedTransferProgressError::InvalidByteCount);
                                if (cursor.generation != generation)
                                {
                                    cursor.generation = generation;
                                    cursor.copied_bytes = 0u;
                                    cursor.command_bytes = command.bytes;
                                    cursor.active_nanoseconds = 0u;
                                }
                                position = cursor.copied_bytes;
                                started_nanoseconds = mappedTransferNanoseconds();
                                admitted = 1u;
                            }
                            else
                                lock.store(0u, ::cuda::memory_order_release);
                        }
                    }
                }
                __syncthreads();
                if (closed)
                    return;
                if (!admitted)
                    continue;

                while (position < command.bytes && !error)
                {
                    const auto remaining = command.bytes - position;
                    const auto count = remaining < quantum ? remaining : quantum;
                    auto *dst = reinterpret_cast<unsigned char *>(command.destination_address) + position;
                    const auto *src = reinterpret_cast<const unsigned char *>(command.source_address) + position;
                    if (((command.destination_address | command.source_address) & 15u) == 0u)
                    {
                        const auto vectors = count / sizeof(uint4);
                        for (std::uint64_t i = threadIdx.x; i < vectors; i += blockDim.x)
                            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
                        for (std::uint64_t i = vectors * sizeof(uint4) + threadIdx.x; i < count; i += blockDim.x)
                            dst[i] = src[i];
                    }
                    else
                    {
                        for (std::uint64_t i = threadIdx.x; i < count; i += blockDim.x)
                            dst[i] = src[i];
                    }
                    __syncthreads();
                    if (threadIdx.x == 0u)
                    {
                        position += count;
                        closed = run == MappedTransferServiceRun::CapturedInterval &&
                            ::cuda::atomic_ref<const std::uint32_t,
                                ::cuda::thread_scope_device>(*interval)
                                .load(::cuda::memory_order_acquire) ==
                                    static_cast<std::uint32_t>(MappedTransferInterval::Closed);
                    }
                    __syncthreads();
                    if (closed)
                        break;
                }
                // Publish all participating threads' payload stores before the
                // next claimant or host consumer can observe the byte cursor.
                __threadfence_system();
                __syncthreads();
                if (threadIdx.x == 0u)
                {
                    auto &cursor = cursors[slot];
                    cursor.active_nanoseconds += mappedTransferNanoseconds() - started_nanoseconds;
                    cursor.copied_bytes = error ? command.bytes : position;
                    if (error || position == command.bytes)
                    {
                        auto &receipt = completions[slot];
                        receipt.error = error;
                        receipt.completed_bytes = error ? 0u : command.bytes;
                        receipt.device_active_nanoseconds = cursor.active_nanoseconds;
                        ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system>(
                            receipt.completed_generation)
                            .store(command.generation, ::cuda::memory_order_release);
                    }
                    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_device>(cursor.claimed)
                        .store(0u, ::cuda::memory_order_release);
                }
                __syncthreads();
                if (closed)
                    return;
            }
            // An empty inbox must not saturate PCIe polling or occupy issue
            // bandwidth. This does not delay the private terminal observation
            // by more than one short device sleep and one bounded inbox scan.
            if (run == MappedTransferServiceRun::CapturedInterval)
                __nanosleep(128u);
        } while (run == MappedTransferServiceRun::CapturedInterval);
    }
} // namespace llaminar2
