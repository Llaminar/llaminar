/**
 * @file CUDAPeerPushReductionKernels.cu
 * @brief Register-resident CUDA In-Band LL direct-P2P reductions and broadcasts with device-owned epochs.
 *
 * Uses NCCL-LL-style 64-bit in-band packets (`[epoch32 : data32]`) transferred via
 * `st.relaxed.sys.global.u64` and acquired via `ld.relaxed.sys.global.u64` directly
 * across PCIe/NVLink BAR without any `MEMBAR.SYS` global L2 drain or separate flag
 * roundtrips. Each block `b` owns a disjoint 512-packet (`uint64_t`) slice across all
 * five kernels (`FP32`, `FP16`, `FP32ViaFP16`, `ReduceFP32`, `BroadcastFP32`) and its
 * own per-block monotonically advancing epoch word in `own.flags`, making arbitrary
 * mixed sequences across arbitrary complete P2P cliques (`participants >= 2`) 100%
 * race-free and deadlock-free without host synchronization or replay-time allocations.
 */
#include "collective/PeerPushReduction.h"
#include "backends/cuda/CUDAHardwareMetadata.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kLLPacketsPerBlock = 2u * PeerPushLayout::threads; // 512 uint64_t packets/block

        /** @brief System-scope relaxed 64-bit store directly into peer BAR / L2 without MEMBAR.SYS. */
        __device__ __forceinline__ void storeRelaxedSysU64(std::uint64_t *addr, std::uint64_t val) noexcept
        {
            asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
        }

        /** @brief System-scope relaxed 64-bit load bypassing SM L1 without MEMBAR.SYS. */
        __device__ __forceinline__ std::uint64_t loadRelaxedSysU64(const std::uint64_t *addr) noexcept
        {
            std::uint64_t val = 0u;
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
            return val;
        }

        /** @brief Poll an in-band LL packet until its upper 32-bit epoch matches `epoch32`. */
        __device__ __forceinline__ bool pollLLPacket(
            const std::uint64_t *addr, std::uint32_t epoch32,
            const volatile std::uint32_t *abort_flag, std::uint32_t &payload_out) noexcept
        {
            unsigned spins = 0u;
            while (true)
            {
                const std::uint64_t pkt = loadRelaxedSysU64(addr);
                if (static_cast<std::uint32_t>(pkt >> 32) == epoch32)
                {
                    payload_out = static_cast<std::uint32_t>(pkt & 0xffffffffu);
                    return false;
                }
                if ((++spins & 63u) == 0u)
                {
                    if (*abort_flag != 0u)
                        return true;
#if __CUDA_ARCH__ >= 700
                    __nanosleep(32u);
#endif
                }
            }
        }

        /** @brief Poll a peer acknowledgment word in `own.flags` until `>= target_epoch`. */
        __device__ __forceinline__ bool pollAckEpoch(
            const std::uint64_t *addr, std::uint64_t target_epoch,
            const volatile std::uint32_t *abort_flag) noexcept
        {
            unsigned spins = 0u;
            while (loadRelaxedSysU64(addr) < target_epoch)
            {
                if ((++spins & 63u) == 0u)
                {
                    if (*abort_flag != 0u)
                        return true;
#if __CUDA_ARCH__ >= 700
                    __nanosleep(32u);
#endif
                }
            }
            return false;
        }

        /** @brief Advance per-block 64-bit epoch while guaranteeing non-zero lower 32-bit tag. */
        __device__ __forceinline__ std::uint64_t nextEpoch(std::uint64_t prev) noexcept
        {
            std::uint64_t ep = prev + 1u;
            if (static_cast<std::uint32_t>(ep) == 0u)
                ep += 2u; // Skip tag == 0 (initial zeroed inbox) while preserving (ep & 1) parity step if needed.
            return ep;
        }

        /** @brief Exact FP32 allreduce sum in ascending rank order via 64-bit In-Band LL packets. */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void cudaPeerPushFP32Kernel(PeerPushBinding binding, const float *input, float *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const int live = live_rows ? *live_rows : int(rows);
            if (live < 0 || std::size_t(live) > rows)
                asm volatile("trap;");

            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t block_base = std::size_t(blockIdx.x) * kLLPacketsPerBlock;
            if (block_base >= live_elements)
                return;

            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = nextEpoch(__ldcv(own.flags + flag_index));
            const std::uint32_t epoch32 = static_cast<std::uint32_t>(epoch);
            const std::size_t ll_stride = layout.bank_elements / 2u;
            const std::size_t slot_base = (epoch & 1u) * std::size_t(layout.participants) * ll_stride;
            const auto *own_inbox = reinterpret_cast<const std::uint64_t *>(own.inbox) + slot_base;

            bool aborted = false;
            #pragma unroll
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                if (idx < live_elements)
                {
                    const float own_val = input[idx];
                    const std::uint32_t fbits = __float_as_uint(own_val);
                    const std::uint64_t pkt = (static_cast<std::uint64_t>(epoch32) << 32) | static_cast<std::uint64_t>(fbits);
                    const std::size_t send_off = slot_base + std::size_t(binding.rank) * ll_stride + idx;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        if (peer != binding.rank)
                            storeRelaxedSysU64(reinterpret_cast<std::uint64_t *>(binding.endpoints[peer].inbox) + send_off, pkt);
                    }

                    float sum = 0.0f;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        float val = own_val;
                        if (peer != binding.rank)
                        {
                            std::uint32_t pbits = 0u;
                            if (pollLLPacket(own_inbox + std::size_t(peer) * ll_stride + idx, epoch32, binding.abort, pbits))
                            {
                                aborted = true;
                                break;
                            }
                            val = __uint_as_float(pbits);
                        }
                        sum = (peer == 0) ? val : __fadd_rn(sum, val);
                    }
                    if (!aborted)
                        output[idx] = sum;
                }
            }
            if (__syncthreads_or(aborted))
                return;
            if (threadIdx.x == 0u)
                __stwt(own.flags + flag_index, epoch);
            if (blockIdx.x == 0u && threadIdx.x == 0u)
            {
                if (payload_bytes)
                    atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(float) * (layout.participants - 1)));
                if (metadata_bytes)
                    atomicAdd(metadata_bytes, static_cast<unsigned long long>(live_elements * sizeof(std::uint32_t) * (layout.participants - 1)));
            }
        }

        /** @brief Rooted FP32 sum reduction in ascending rank order via 64-bit In-Band LL packets + ack. */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void cudaPeerPushReduceFP32Kernel(PeerPushBinding binding, const float *input,
            float *output, int root, std::size_t total_elements)
        {
            const auto layout = binding.layout;
            const std::size_t block_base = std::size_t(blockIdx.x) * kLLPacketsPerBlock;
            if (block_base >= total_elements)
                return;

            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = nextEpoch(__ldcv(own.flags + flag_index));
            const std::uint32_t epoch32 = static_cast<std::uint32_t>(epoch);
            const std::size_t ll_stride = layout.bank_elements / 2u;
            const std::size_t slot_base = (epoch & 1u) * std::size_t(layout.participants) * ll_stride;

            bool aborted = false;
            if (binding.rank != root)
            {
                auto *root_inbox = reinterpret_cast<std::uint64_t *>(binding.endpoints[root].inbox) +
                    slot_base + std::size_t(binding.rank) * ll_stride;
                #pragma unroll
                for (std::size_t pass = 0u; pass < 2u; ++pass)
                {
                    const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                    if (idx < total_elements)
                    {
                        const std::uint32_t fbits = __float_as_uint(input[idx]);
                        const std::uint64_t pkt = (static_cast<std::uint64_t>(epoch32) << 32) | static_cast<std::uint64_t>(fbits);
                        storeRelaxedSysU64(root_inbox + idx, pkt);
                    }
                }
                if (threadIdx.x == 0u)
                {
                    const auto root_ack_idx = (std::size_t(root) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    aborted = pollAckEpoch(own.flags + root_ack_idx, epoch, binding.abort);
                }
                if (__syncthreads_or(aborted))
                    return;
                if (threadIdx.x == 0u)
                    __stwt(own.flags + flag_index, epoch);
                return;
            }

            const auto *own_inbox = reinterpret_cast<const std::uint64_t *>(own.inbox) + slot_base;
            #pragma unroll
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                if (idx < total_elements)
                {
                    const float own_val = input[idx];
                    float sum = 0.0f;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        float val = own_val;
                        if (peer != root)
                        {
                            std::uint32_t pbits = 0u;
                            if (pollLLPacket(own_inbox + std::size_t(peer) * ll_stride + idx, epoch32, binding.abort, pbits))
                            {
                                aborted = true;
                                break;
                            }
                            val = __uint_as_float(pbits);
                        }
                        sum = (peer == 0) ? val : __fadd_rn(sum, val);
                    }
                    if (!aborted)
                        output[idx] = sum;
                }
            }
            if (__syncthreads_or(aborted))
                return;
            if (threadIdx.x == 0u)
                __stwt(own.flags + flag_index, epoch);
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
            {
                if (peer != root)
                    storeRelaxedSysU64(binding.endpoints[peer].flags + flag_index, epoch);
            }
        }

        /** @brief Rooted FP32 broadcast via 64-bit In-Band LL packets + ack. */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void cudaPeerPushBroadcastFP32Kernel(PeerPushBinding binding, const float *input,
            float *output, int root, std::size_t total_elements)
        {
            const auto layout = binding.layout;
            const std::size_t block_base = std::size_t(blockIdx.x) * kLLPacketsPerBlock;
            if (block_base >= total_elements)
                return;

            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = nextEpoch(__ldcv(own.flags + flag_index));
            const std::uint32_t epoch32 = static_cast<std::uint32_t>(epoch);
            const std::size_t ll_stride = layout.bank_elements / 2u;
            const std::size_t slot_base = (epoch & 1u) * std::size_t(layout.participants) * ll_stride;

            bool aborted = false;
            if (binding.rank == root)
            {
                const std::size_t send_off = slot_base + std::size_t(root) * ll_stride;
                #pragma unroll
                for (std::size_t pass = 0u; pass < 2u; ++pass)
                {
                    const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                    if (idx < total_elements)
                    {
                        const float val = input[idx];
                        const std::uint64_t pkt = (static_cast<std::uint64_t>(epoch32) << 32) |
                            static_cast<std::uint64_t>(__float_as_uint(val));
                        for (int peer = 0; peer < layout.participants; ++peer)
                        {
                            if (peer != root)
                                storeRelaxedSysU64(reinterpret_cast<std::uint64_t *>(binding.endpoints[peer].inbox) + send_off + idx, pkt);
                        }
                        if (output != input)
                            output[idx] = val;
                    }
                }
                for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                {
                    if (peer != root)
                    {
                        const auto ack_idx = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                        if (pollAckEpoch(own.flags + ack_idx, epoch, binding.abort))
                        {
                            aborted = true;
                            break;
                        }
                    }
                }
                if (__syncthreads_or(aborted))
                    return;
                if (threadIdx.x == 0u)
                    __stwt(own.flags + flag_index, epoch);
                return;
            }

            const auto *own_inbox = reinterpret_cast<const std::uint64_t *>(own.inbox) +
                slot_base + std::size_t(root) * ll_stride;
            #pragma unroll
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                if (idx < total_elements)
                {
                    std::uint32_t pbits = 0u;
                    if (pollLLPacket(own_inbox + idx, epoch32, binding.abort, pbits))
                    {
                        aborted = true;
                        break;
                    }
                    output[idx] = __uint_as_float(pbits);
                }
            }
            if (__syncthreads_or(aborted))
                return;
            if (threadIdx.x == 0u)
            {
                __stwt(own.flags + flag_index, epoch);
                storeRelaxedSysU64(binding.endpoints[root].flags + flag_index, epoch);
            }
        }

        /** @brief Exact FP16 allreduce sum in ascending rank order via packed half2 In-Band LL packets. */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void cudaPeerPushFP16Kernel(PeerPushBinding binding, const __half *input, __half *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const int live = live_rows ? *live_rows : int(rows);
            if (live < 0 || std::size_t(live) > rows)
                asm volatile("trap;");

            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t live_pairs = (live_elements + 1u) / 2u;
            const std::size_t block_base = std::size_t(blockIdx.x) * kLLPacketsPerBlock;
            if (block_base >= live_pairs)
                return;

            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = nextEpoch(__ldcv(own.flags + flag_index));
            const std::uint32_t epoch32 = static_cast<std::uint32_t>(epoch);
            const std::size_t ll_stride = layout.bank_elements / 2u;
            const std::size_t slot_base = (epoch & 1u) * std::size_t(layout.participants) * ll_stride;
            const auto *own_inbox = reinterpret_cast<const std::uint64_t *>(own.inbox) + slot_base;

            bool aborted = false;
            #pragma unroll
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                if (idx < live_pairs)
                {
                    const std::size_t elem0 = 2u * idx;
                    const bool has1 = (elem0 + 1u < live_elements);
                    const __half h0 = input[elem0];
                    const __half h1 = has1 ? input[elem0 + 1u] : __float2half(0.0f);
                    const __half2 own_h2 = __halves2half2(h0, h1);
                    const std::uint32_t hbits = *reinterpret_cast<const std::uint32_t *>(&own_h2);
                    const std::uint64_t pkt = (static_cast<std::uint64_t>(epoch32) << 32) | static_cast<std::uint64_t>(hbits);
                    const std::size_t send_off = slot_base + std::size_t(binding.rank) * ll_stride + idx;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        if (peer != binding.rank)
                            storeRelaxedSysU64(reinterpret_cast<std::uint64_t *>(binding.endpoints[peer].inbox) + send_off, pkt);
                    }

                    __half2 sum = own_h2;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        __half2 val = own_h2;
                        if (peer != binding.rank)
                        {
                            std::uint32_t pbits = 0u;
                            if (pollLLPacket(own_inbox + std::size_t(peer) * ll_stride + idx, epoch32, binding.abort, pbits))
                            {
                                aborted = true;
                                break;
                            }
                            val = *reinterpret_cast<const __half2 *>(&pbits);
                        }
                        sum = (peer == 0) ? val : __hadd2(sum, val);
                    }
                    if (!aborted)
                    {
                        output[elem0] = __low2half(sum);
                        if (has1)
                            output[elem0 + 1u] = __high2half(sum);
                    }
                }
            }
            if (__syncthreads_or(aborted))
                return;
            if (threadIdx.x == 0u)
                __stwt(own.flags + flag_index, epoch);
            if (blockIdx.x == 0u && threadIdx.x == 0u)
            {
                if (payload_bytes)
                    atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(__half) * (layout.participants - 1)));
                if (metadata_bytes)
                    atomicAdd(metadata_bytes, static_cast<unsigned long long>(live_pairs * sizeof(std::uint32_t) * (layout.participants - 1)));
            }
        }

        /** @brief Fused FP32->FP16->rank-ordered-sum->FP32 allreduce via packed half2 In-Band LL packets. */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void cudaPeerPushFP32ViaFP16Kernel(PeerPushBinding binding, const float *input, float *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const int live = live_rows ? *live_rows : int(rows);
            if (live < 0 || std::size_t(live) > rows)
                asm volatile("trap;");

            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t live_pairs = (live_elements + 1u) / 2u;
            const std::size_t block_base = std::size_t(blockIdx.x) * kLLPacketsPerBlock;
            if (block_base >= live_pairs)
                return;

            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = nextEpoch(__ldcv(own.flags + flag_index));
            const std::uint32_t epoch32 = static_cast<std::uint32_t>(epoch);
            const std::size_t ll_stride = layout.bank_elements / 2u;
            const std::size_t slot_base = (epoch & 1u) * std::size_t(layout.participants) * ll_stride;
            const auto *own_inbox = reinterpret_cast<const std::uint64_t *>(own.inbox) + slot_base;

            bool aborted = false;
            #pragma unroll
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t idx = block_base + pass * PeerPushLayout::threads + threadIdx.x;
                if (idx < live_pairs)
                {
                    const std::size_t elem0 = 2u * idx;
                    const bool has1 = (elem0 + 1u < live_elements);
                    const float f0 = input[elem0];
                    const float f1 = has1 ? input[elem0 + 1u] : 0.0f;
                    const __half2 own_h2 = __floats2half2_rn(f0, f1);
                    const std::uint32_t hbits = *reinterpret_cast<const std::uint32_t *>(&own_h2);
                    const std::uint64_t pkt = (static_cast<std::uint64_t>(epoch32) << 32) | static_cast<std::uint64_t>(hbits);
                    const std::size_t send_off = slot_base + std::size_t(binding.rank) * ll_stride + idx;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        if (peer != binding.rank)
                            storeRelaxedSysU64(reinterpret_cast<std::uint64_t *>(binding.endpoints[peer].inbox) + send_off, pkt);
                    }

                    __half2 sum = own_h2;
                    for (int peer = 0; peer < layout.participants; ++peer)
                    {
                        __half2 val = own_h2;
                        if (peer != binding.rank)
                        {
                            std::uint32_t pbits = 0u;
                            if (pollLLPacket(own_inbox + std::size_t(peer) * ll_stride + idx, epoch32, binding.abort, pbits))
                            {
                                aborted = true;
                                break;
                            }
                            val = *reinterpret_cast<const __half2 *>(&pbits);
                        }
                        sum = (peer == 0) ? val : __hadd2(sum, val);
                    }
                    if (!aborted)
                    {
                        const float2 out_f = __half22float2(sum);
                        output[elem0] = out_f.x;
                        if (has1)
                            output[elem0 + 1u] = out_f.y;
                    }
                }
            }
            if (__syncthreads_or(aborted))
                return;
            if (threadIdx.x == 0u)
                __stwt(own.flags + flag_index, epoch);
            if (blockIdx.x == 0u && threadIdx.x == 0u)
            {
                if (payload_bytes)
                    atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(__half) * (layout.participants - 1)));
                if (metadata_bytes)
                    atomicAdd(metadata_bytes, static_cast<unsigned long long>(live_pairs * sizeof(std::uint32_t) * (layout.participants - 1)));
            }
        }
    } // namespace

    bool cudaPeerPushResidentCapacity(std::size_t &capacity)
    {
        capacity = 0;
        int per_sm_fp32 = 0, per_sm_fp16 = 0, per_sm_fused = 0, per_sm_red = 0, per_sm_bcast = 0, device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_fp32, cudaPeerPushFP32Kernel, PeerPushLayout::threads, 0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_fp16, cudaPeerPushFP16Kernel, PeerPushLayout::threads, 0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_fused, cudaPeerPushFP32ViaFP16Kernel, PeerPushLayout::threads, 0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_red, cudaPeerPushReduceFP32Kernel, PeerPushLayout::threads, 0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_bcast, cudaPeerPushBroadcastFP32Kernel, PeerPushLayout::threads, 0) != cudaSuccess)
        {
            return false;
        }
        int effective = per_sm_fp32;
        if (per_sm_fp16 < effective) effective = per_sm_fp16;
        if (per_sm_fused < effective) effective = per_sm_fused;
        if (per_sm_red < effective) effective = per_sm_red;
        if (per_sm_bcast < effective) effective = per_sm_bcast;
        if (effective <= 0)
            return false;
        capacity = std::size_t(effective) * std::size_t(cudaHardwareMetadata().at(device).sm_count);
        return true;
    }

    bool launchCUDAPeerPushFP32(const PeerPushBinding &binding, const float *input, float *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        const std::size_t max_fp32 = binding.layout.bank_elements / 2u;
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > max_fp32 / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t) != 0u)
            return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(float) || b > UINTPTR_MAX - elements * sizeof(float) ||
            (a != b && a < b + elements * sizeof(float) && b < a + elements * sizeof(float)))
            return false;
        const std::size_t grid = (elements + kLLPacketsPerBlock - 1u) / kLLPacketsPerBlock;
        cudaPeerPushFP32Kernel<<<dim3(grid), dim3(PeerPushLayout::threads), 0, static_cast<cudaStream_t>(stream)>>>(
            binding, input, output, live_rows, rows, columns, payload_bytes, metadata_bytes);
        return cudaGetLastError() == cudaSuccess;
    }

    bool launchCUDAPeerPushReduceFP32(const PeerPushBinding &binding, const float *input,
        float *output, int root, std::size_t count, void *stream)
    {
        const std::size_t max_fp32 = binding.layout.bank_elements / 2u;
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            root < 0 || root >= binding.layout.participants ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
            !count || count > max_fp32 ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0u)
            return false;
        const std::size_t grid = (count + kLLPacketsPerBlock - 1u) / kLLPacketsPerBlock;
        cudaPeerPushReduceFP32Kernel<<<dim3(grid), dim3(PeerPushLayout::threads), 0, static_cast<cudaStream_t>(stream)>>>(
            binding, input, output, root, count);
        return cudaGetLastError() == cudaSuccess;
    }

    bool launchCUDAPeerPushBroadcastFP32(const PeerPushBinding &binding, const float *input,
        float *output, int root, std::size_t count, void *stream)
    {
        const std::size_t max_fp32 = binding.layout.bank_elements / 2u;
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            root < 0 || root >= binding.layout.participants ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
            !count || count > max_fp32 ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0u)
            return false;
        const std::size_t grid = (count + kLLPacketsPerBlock - 1u) / kLLPacketsPerBlock;
        cudaPeerPushBroadcastFP32Kernel<<<dim3(grid), dim3(PeerPushLayout::threads), 0, static_cast<cudaStream_t>(stream)>>>(
            binding, input, output, root, count);
        return cudaGetLastError() == cudaSuccess;
    }

    bool launchCUDAPeerPushFP16(const PeerPushBinding &binding, const void *input, void *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > binding.layout.bank_elements / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(__half) != 0u ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(__half) != 0u ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t) != 0u)
            return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(__half) || b > UINTPTR_MAX - elements * sizeof(__half) ||
            (a != b && a < b + elements * sizeof(__half) && b < a + elements * sizeof(__half)))
            return false;
        cudaPeerPushFP16Kernel<<<dim3(PeerPushLayout::blocksFor(elements)),
            dim3(PeerPushLayout::threads), 0, static_cast<cudaStream_t>(stream)>>>(
            binding, static_cast<const __half *>(input), static_cast<__half *>(output),
            live_rows, rows, columns, payload_bytes, metadata_bytes);
        return cudaGetLastError() == cudaSuccess;
    }

    bool launchCUDAPeerPushFP32ViaFP16(const PeerPushBinding &binding, const float *input, float *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > binding.layout.bank_elements / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0u ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t) != 0u)
            return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(float) || b > UINTPTR_MAX - elements * sizeof(float) ||
            (a != b && a < b + elements * sizeof(float) && b < a + elements * sizeof(float)))
            return false;
        cudaPeerPushFP32ViaFP16Kernel<<<dim3(PeerPushLayout::blocksFor(elements)),
            dim3(PeerPushLayout::threads), 0, static_cast<cudaStream_t>(stream)>>>(
            binding, input, output, live_rows, rows, columns, payload_bytes, metadata_bytes);
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace llaminar2
