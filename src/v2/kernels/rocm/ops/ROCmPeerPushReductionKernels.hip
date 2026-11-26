/**
 * @file ROCmPeerPushReductionKernels.hip
 * @brief Register-resident small FP32 reductions with device-owned double-buffer progress.
 *
 * Like Castagnia's small all-peer push, producers write contiguous original
 * values into receiver-local uncached inboxes, publish per-block flags and
 * then reduce local banks. Unlike a host token, self flags advance on replay.
 * Fixed block ranges make changing retained row geometries safe. A participant
 * cannot reuse a slot until its peer has finished reading the older occupant:
 * reaching epoch t+2 requires that peer's publication for t+1, whose preceding
 * stream operation completed every read for t. Ordinary RCCL uses disjoint
 * storage and remains ordered by the same original participant stream.
 */
#include "collective/PeerPushReduction.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace llaminar2
{
    namespace
    {
        using Vector = float __attribute__((ext_vector_type(4)));

        /** @brief Preserve all four original bits with a scalar tail for unaligned operands. */
        __device__ __forceinline__ Vector loadVector(const float *p)
        {
            if ((reinterpret_cast<std::uintptr_t>(p) & 15) == 0)
                return *reinterpret_cast<const Vector *>(p);
            return Vector{p[0], p[1], p[2], p[3]};
        }

        /** @brief Publish a vector without requiring tensor bases to have vector alignment. */
        __device__ __forceinline__ void storeVector(float *p, Vector value)
        {
            if ((reinterpret_cast<std::uintptr_t>(p) & 15) == 0)
                *reinterpret_cast<Vector *>(p) = value;
            else
                for (int i = 0; i < 4; ++i) p[i] = value[i];
        }

        /**
         * @brief Push one fixed block range, join original peer publications and fold ranks.
         *
         * No peer count is encoded in the program or grid. Threads first publish
         * every readiness flag with system-release ordering, then acquire peer
         * flags; large memberships cannot form a publish-after-wait cycle. An
         * explicit epoch-1 guard before writing a double-buffered inbox slot
         * guarantees safe arbitrary interleaving with rooted Reduce/Broadcast
         * operations sharing the same uncached inbox and epoch flags.
         */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void peerPushFP32Kernel(PeerPushBinding binding, const float *input, float *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) + 1;
            const int live = live_rows ? *live_rows : int(rows);
            if (!epoch || live < 0 || std::size_t(live) > rows)
            {
                __builtin_trap(); // Invalid device state is fatal, never wrapped into a stale epoch.
            }
            const std::size_t first = std::size_t(blockIdx.x) * layout.elements_per_block;
            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t count = first < live_elements ? min(layout.elements_per_block, live_elements - first) : 0;
            const std::size_t vectors = count / 4;
            const std::size_t slot = (epoch & 1) * std::size_t(layout.participants) * layout.bank_elements;
            const std::size_t send_offset = slot + std::size_t(binding.rank) * layout.bank_elements + first;

            // Ensure every peer has retired epoch-2 before reusing its slot.
            bool aborted = false;
            if (epoch >= 2)
            {
                for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                    if (peer != binding.rank)
                    {
                        const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                        while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch - 1)
                        {
                            if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                if (__syncthreads_or(aborted)) return;
            }

            if (threadIdx.x < vectors)
            {
                const Vector value = loadVector(input + first + 4 * threadIdx.x);
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank)
                        *reinterpret_cast<Vector *>(binding.endpoints[peer].inbox + send_offset + 4 * threadIdx.x) = value;
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = 4 * vectors + threadIdx.x;
                const float value = input[first + i];
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank) binding.endpoints[peer].inbox[send_offset + i] = value;
            }
            // __syncthreads() emits s_waitcnt vmcnt(0) lgkmcnt(0) before barrier;
            // system-release flag stores then expose uncached inbox writes.
            __syncthreads();
            if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                    __hip_atomic_store(binding.endpoints[peer].flags + flag_index, epoch,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                {
                    const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
                    {
                        if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            if (__syncthreads_or(aborted)) return;
            if (threadIdx.x < vectors)
            {
                const auto i = first + 4 * threadIdx.x;
                Vector sum = binding.rank == 0 ? loadVector(input + i) :
                    *reinterpret_cast<const volatile Vector *>(own.inbox + slot + i);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    const Vector value = peer == binding.rank ? loadVector(input + i) :
                        *reinterpret_cast<const volatile Vector *>(own.inbox + slot + std::size_t(peer) * layout.bank_elements + i);
                    #pragma unroll
                    for (int component = 0; component < 4; ++component)
                        sum[component] = __fadd_rn(sum[component], value[component]);
                }
                storeVector(output + i, sum);
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = first + 4 * vectors + threadIdx.x;
                float sum = binding.rank == 0 ? input[i] : own.inbox[slot + i];
                for (int peer = 1; peer < layout.participants; ++peer)
                    sum = __fadd_rn(sum, peer == binding.rank ? input[i] :
                        own.inbox[slot + std::size_t(peer) * layout.bank_elements + i]);
                output[i] = sum;
            }
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                if (payload_bytes) atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(float) * (layout.participants - 1)));
                if (metadata_bytes) atomicAdd(metadata_bytes, static_cast<unsigned long long>(gridDim.x * sizeof(std::uint64_t) * (layout.participants - 1)));
            }
        }

        /**
         * @brief Direct-push rooted FP32 sum reduction in exact ascending rank order.
         *
         * Non-root ranks push only their live slice to root's uncached inbox
         * and release-publish their flag. Root waits for all peers, sums in
         * exact rank order 0..participants-1, and acknowledges every non-root
         * so subsequent collectives know this inbox slot is free.
         */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void peerPushReduceFP32Kernel(PeerPushBinding binding, const float *input,
            float *output, int root, std::size_t total_elements)
        {
            const auto layout = binding.layout;
            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) + 1;
            if (!epoch) __builtin_trap();
            const std::size_t first = std::size_t(blockIdx.x) * layout.elements_per_block;
            const std::size_t count = first < total_elements ? min(layout.elements_per_block, total_elements - first) : 0;
            const std::size_t vectors = count / 4;
            const std::size_t slot = (epoch & 1) * std::size_t(layout.participants) * layout.bank_elements;

            bool aborted = false;
            if (binding.rank != root)
            {
                if (epoch >= 2)
                {
                    if (threadIdx.x == 0)
                    {
                        const auto root_idx = (std::size_t(root) * layout.blocks + blockIdx.x) * layout.flag_stride;
                        while (__hip_atomic_load(own.flags + root_idx, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch - 1)
                        {
                            if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                    if (__syncthreads_or(aborted)) return;
                }
                const std::size_t send_offset = slot + std::size_t(binding.rank) * layout.bank_elements + first;
                if (threadIdx.x < vectors)
                {
                    const Vector value = loadVector(input + first + 4 * threadIdx.x);
                    *reinterpret_cast<Vector *>(binding.endpoints[root].inbox + send_offset + 4 * threadIdx.x) = value;
                }
                if (4 * vectors + threadIdx.x < count)
                {
                    const auto i = 4 * vectors + threadIdx.x;
                    binding.endpoints[root].inbox[send_offset + i] = input[first + i];
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
                    __hip_atomic_store(binding.endpoints[root].flags + flag_index, epoch,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                }
                return;
            }

            // Root participant: acquire every non-root contribution, fold in rank order, and ack.
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != root)
                {
                    const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
                    {
                        if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            if (__syncthreads_or(aborted)) return;
            if (threadIdx.x < vectors)
            {
                const auto i = first + 4 * threadIdx.x;
                Vector sum = root == 0 ? loadVector(input + i) :
                    *reinterpret_cast<const volatile Vector *>(own.inbox + slot + i);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    const Vector value = peer == root ? loadVector(input + i) :
                        *reinterpret_cast<const volatile Vector *>(own.inbox + slot + std::size_t(peer) * layout.bank_elements + i);
                    #pragma unroll
                    for (int component = 0; component < 4; ++component)
                        sum[component] = __fadd_rn(sum[component], value[component]);
                }
                storeVector(output + i, sum);
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = first + 4 * vectors + threadIdx.x;
                float sum = root == 0 ? input[i] : own.inbox[slot + i];
                for (int peer = 1; peer < layout.participants; ++peer)
                    sum = __fadd_rn(sum, peer == root ? input[i] :
                        own.inbox[slot + std::size_t(peer) * layout.bank_elements + i]);
                output[i] = sum;
            }
            __syncthreads();
            if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != root)
                    __hip_atomic_store(binding.endpoints[peer].flags + flag_index, epoch,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        /**
         * @brief Direct-push rooted FP32 broadcast across an arbitrary P2P clique.
         *
         * Root waits for peers to retire epoch-2, pushes live vectors to every
         * peer's uncached inbox and release-publishes its flag. Each non-root
         * acquires root's flag, copies its slice to output, and acknowledges root.
         */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void peerPushBroadcastFP32Kernel(PeerPushBinding binding, const float *input,
            float *output, int root, std::size_t total_elements)
        {
            const auto layout = binding.layout;
            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) + 1;
            if (!epoch) __builtin_trap();
            const std::size_t first = std::size_t(blockIdx.x) * layout.elements_per_block;
            const std::size_t count = first < total_elements ? min(layout.elements_per_block, total_elements - first) : 0;
            const std::size_t vectors = count / 4;
            const std::size_t slot = (epoch & 1) * std::size_t(layout.participants) * layout.bank_elements;

            bool aborted = false;
            if (binding.rank == root)
            {
                if (epoch >= 2)
                {
                    for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                        if (peer != root)
                        {
                            const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                            while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch - 1)
                            {
                                if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                                __builtin_amdgcn_s_sleep(1);
                            }
                        }
                    if (__syncthreads_or(aborted)) return;
                }
                const std::size_t send_offset = slot + std::size_t(root) * layout.bank_elements + first;
                if (threadIdx.x < vectors)
                {
                    const Vector value = loadVector(input + first + 4 * threadIdx.x);
                    for (int peer = 0; peer < layout.participants; ++peer)
                        if (peer != root)
                            *reinterpret_cast<Vector *>(binding.endpoints[peer].inbox + send_offset + 4 * threadIdx.x) = value;
                    if (output != input) storeVector(output + first + 4 * threadIdx.x, value);
                }
                if (4 * vectors + threadIdx.x < count)
                {
                    const auto i = 4 * vectors + threadIdx.x;
                    const float value = input[first + i];
                    for (int peer = 0; peer < layout.participants; ++peer)
                        if (peer != root) binding.endpoints[peer].inbox[send_offset + i] = value;
                    if (output != input) output[first + i] = value;
                }
                __syncthreads();
                if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
                for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                    if (peer != root)
                        __hip_atomic_store(binding.endpoints[peer].flags + flag_index, epoch,
                            __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                return;
            }

            // Non-root participant: acquire root publication, copy to output, and ack root.
            if (threadIdx.x == 0)
            {
                const auto root_idx = (std::size_t(root) * layout.blocks + blockIdx.x) * layout.flag_stride;
                while (__hip_atomic_load(own.flags + root_idx, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
                {
                    if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            if (__syncthreads_or(aborted)) return;
            const std::size_t recv_offset = slot + std::size_t(root) * layout.bank_elements + first;
            if (threadIdx.x < vectors)
            {
                const Vector value = *reinterpret_cast<const volatile Vector *>(own.inbox + recv_offset + 4 * threadIdx.x);
                storeVector(output + first + 4 * threadIdx.x, value);
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = 4 * vectors + threadIdx.x;
                output[first + i] = own.inbox[recv_offset + i];
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
                __hip_atomic_store(binding.endpoints[root].flags + flag_index, epoch,
                    __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }

        using Half4Bits = std::uint32_t __attribute__((ext_vector_type(2)));

        union Half4Pack
        {
            __half h[4];
            Half4Bits u;
        };

        /** @brief Load four contiguous FP16 values with scalar fallback for unaligned bases. */
        __device__ __forceinline__ Half4Pack loadHalf4(const __half *p)
        {
            Half4Pack out;
            if ((reinterpret_cast<std::uintptr_t>(p) & 7) == 0)
                out.u = *reinterpret_cast<const Half4Bits *>(p);
            else
                for (int i = 0; i < 4; ++i) out.h[i] = p[i];
            return out;
        }

        /** @brief Store four contiguous FP16 values with scalar fallback for unaligned bases. */
        __device__ __forceinline__ void storeHalf4(__half *p, Half4Pack v)
        {
            if ((reinterpret_cast<std::uintptr_t>(p) & 7) == 0)
                *reinterpret_cast<Half4Bits *>(p) = v.u;
            else
                for (int i = 0; i < 4; ++i) p[i] = v.h[i];
        }

        /**
         * @brief Direct-push FP16 allreduce sum in exact ascending rank order.
         *
         * Uses the same fixed block ranges and double-buffered uncached inbox as
         * peerPushFP32Kernel (8-byte uint2 = 4 halfs per thread, halving PCIe bytes).
         */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void peerPushFP16Kernel(PeerPushBinding binding, const __half *input, __half *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) + 1;
            const int live = live_rows ? *live_rows : int(rows);
            if (!epoch || live < 0 || std::size_t(live) > rows)
            {
                __builtin_trap();
            }
            const std::size_t first = std::size_t(blockIdx.x) * layout.elements_per_block;
            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t count = first < live_elements ? min(layout.elements_per_block, live_elements - first) : 0;
            const std::size_t vectors = count / 4;
            const std::size_t slot = (epoch & 1) * std::size_t(layout.participants) * layout.bank_elements;
            const std::size_t send_offset = slot + std::size_t(binding.rank) * layout.bank_elements + first;

            bool aborted = false;
            if (epoch >= 2)
            {
                for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                    if (peer != binding.rank)
                    {
                        const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                        while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch - 1)
                        {
                            if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                if (__syncthreads_or(aborted)) return;
            }

            if (threadIdx.x < vectors)
            {
                const Half4Pack value = loadHalf4(input + first + 4 * threadIdx.x);
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank)
                        *reinterpret_cast<Half4Bits *>(reinterpret_cast<__half *>(binding.endpoints[peer].inbox) + send_offset + 4 * threadIdx.x) = value.u;
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = 4 * vectors + threadIdx.x;
                const __half value = input[first + i];
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank)
                        reinterpret_cast<__half *>(binding.endpoints[peer].inbox)[send_offset + i] = value;
            }
            __syncthreads();
            if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                    __hip_atomic_store(binding.endpoints[peer].flags + flag_index, epoch,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                {
                    const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
                    {
                        if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            if (__syncthreads_or(aborted)) return;
            if (threadIdx.x < vectors)
            {
                const auto i = first + 4 * threadIdx.x;
                Half4Pack sum;
                if (binding.rank == 0)
                    sum = loadHalf4(input + i);
                else
                    sum.u = *reinterpret_cast<const volatile Half4Bits *>(reinterpret_cast<const __half *>(own.inbox) + slot + i);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    Half4Pack value;
                    if (peer == binding.rank)
                        value = loadHalf4(input + i);
                    else
                        value.u = *reinterpret_cast<const volatile Half4Bits *>(reinterpret_cast<const __half *>(own.inbox) + slot + std::size_t(peer) * layout.bank_elements + i);
                    #pragma unroll
                    for (int component = 0; component < 4; ++component)
                        sum.h[component] = __hadd(sum.h[component], value.h[component]);
                }
                storeHalf4(output + i, sum);
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = first + 4 * vectors + threadIdx.x;
                const volatile __half *own_h = reinterpret_cast<const volatile __half *>(own.inbox);
                __half sum = binding.rank == 0 ? input[i] : static_cast<__half>(own_h[slot + i]);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    const __half value = peer == binding.rank ? input[i]
                        : static_cast<__half>(own_h[slot + std::size_t(peer) * layout.bank_elements + i]);
                    sum = __hadd(sum, value);
                }
                output[i] = sum;
            }
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                if (payload_bytes) atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(__half) * (layout.participants - 1)));
                if (metadata_bytes) atomicAdd(metadata_bytes, static_cast<unsigned long long>(gridDim.x * sizeof(std::uint64_t) * (layout.participants - 1)));
            }
        }

        /**
         * @brief Fused FP32->FP16->rank-ordered-sum->FP32 direct-push allreduce.
         *
         * Preserves bit-exact FP16 transport arithmetic (__float2half -> __hadd
         * in ascending rank order -> __half2float) while eliminating two separate
         * cast kernel launches and FP16 scratch reads/writes.
         */
        __global__ __launch_bounds__(PeerPushLayout::threads)
        void peerPushFP32ViaFP16Kernel(PeerPushBinding binding, const float *input, float *output,
            const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
            unsigned long long *payload_bytes, unsigned long long *metadata_bytes)
        {
            const auto layout = binding.layout;
            const auto own = binding.endpoints[binding.rank];
            const auto flag_index = (std::size_t(binding.rank) * layout.blocks + blockIdx.x) * layout.flag_stride;
            const std::uint64_t epoch = *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) + 1;
            const int live = live_rows ? *live_rows : int(rows);
            if (!epoch || live < 0 || std::size_t(live) > rows)
            {
                __builtin_trap();
            }
            const std::size_t first = std::size_t(blockIdx.x) * layout.elements_per_block;
            const std::size_t live_elements = std::size_t(live) * columns;
            const std::size_t count = first < live_elements ? min(layout.elements_per_block, live_elements - first) : 0;
            const std::size_t vectors = count / 4;
            const std::size_t slot = (epoch & 1) * std::size_t(layout.participants) * layout.bank_elements;
            const std::size_t send_offset = slot + std::size_t(binding.rank) * layout.bank_elements + first;

            bool aborted = false;
            if (epoch >= 2)
            {
                for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                    if (peer != binding.rank)
                    {
                        const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                        while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch - 1)
                        {
                            if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                            __builtin_amdgcn_s_sleep(1);
                        }
                    }
                if (__syncthreads_or(aborted)) return;
            }

            Half4Pack own_vec{};
            __half own_scalar{};
            if (threadIdx.x < vectors)
            {
                const Vector vf = loadVector(input + first + 4 * threadIdx.x);
                #pragma unroll
                for (int component = 0; component < 4; ++component)
                    own_vec.h[component] = __float2half(vf[component]);
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank)
                        *reinterpret_cast<Half4Bits *>(reinterpret_cast<__half *>(binding.endpoints[peer].inbox) + send_offset + 4 * threadIdx.x) = own_vec.u;
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = 4 * vectors + threadIdx.x;
                own_scalar = __float2half(input[first + i]);
                for (int peer = 0; peer < layout.participants; ++peer)
                    if (peer != binding.rank)
                        reinterpret_cast<__half *>(binding.endpoints[peer].inbox)[send_offset + i] = own_scalar;
            }
            __syncthreads();
            if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                    __hip_atomic_store(binding.endpoints[peer].flags + flag_index, epoch,
                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                {
                    const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    while (__hip_atomic_load(own.flags + index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch)
                    {
                        if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            if (__syncthreads_or(aborted)) return;
            if (threadIdx.x < vectors)
            {
                const auto i = first + 4 * threadIdx.x;
                Half4Pack sum;
                if (binding.rank == 0)
                    sum = own_vec;
                else
                    sum.u = *reinterpret_cast<const volatile Half4Bits *>(reinterpret_cast<const __half *>(own.inbox) + slot + i);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    Half4Pack value;
                    if (peer == binding.rank)
                        value = own_vec;
                    else
                        value.u = *reinterpret_cast<const volatile Half4Bits *>(reinterpret_cast<const __half *>(own.inbox) + slot + std::size_t(peer) * layout.bank_elements + i);
                    #pragma unroll
                    for (int component = 0; component < 4; ++component)
                        sum.h[component] = __hadd(sum.h[component], value.h[component]);
                }
                Vector out_f;
                #pragma unroll
                for (int component = 0; component < 4; ++component)
                    out_f[component] = __half2float(sum.h[component]);
                storeVector(output + i, out_f);
            }
            if (4 * vectors + threadIdx.x < count)
            {
                const auto i = first + 4 * vectors + threadIdx.x;
                const volatile __half *own_h = reinterpret_cast<const volatile __half *>(own.inbox);
                __half sum = binding.rank == 0 ? own_scalar : static_cast<__half>(own_h[slot + i]);
                for (int peer = 1; peer < layout.participants; ++peer)
                {
                    const __half value = peer == binding.rank ? own_scalar
                        : static_cast<__half>(own_h[slot + std::size_t(peer) * layout.bank_elements + i]);
                    sum = __hadd(sum, value);
                }
                output[i] = __half2float(sum);
            }
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                if (payload_bytes) atomicAdd(payload_bytes, static_cast<unsigned long long>(live_elements * sizeof(__half) * (layout.participants - 1)));
                if (metadata_bytes) atomicAdd(metadata_bytes, static_cast<unsigned long long>(gridDim.x * sizeof(std::uint64_t) * (layout.participants - 1)));
            }
        }
    }

    bool rocmPeerPushResidentCapacity(std::size_t &capacity)
    {
        capacity = 0;
        int per_cu = 0, per_cu_fp16 = 0, per_cu_fused = 0, device = -1;
        if (hipGetDevice(&device) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, peerPushFP32Kernel, PeerPushLayout::threads, 0) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu_fp16, peerPushFP16Kernel, PeerPushLayout::threads, 0) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu_fused, peerPushFP32ViaFP16Kernel, PeerPushLayout::threads, 0) != hipSuccess ||
            per_cu <= 0 || per_cu_fp16 <= 0 || per_cu_fused <= 0) return false;
        const int effective_per_cu = per_cu < per_cu_fp16 ? (per_cu < per_cu_fused ? per_cu : per_cu_fused)
                                                          : (per_cu_fp16 < per_cu_fused ? per_cu_fp16 : per_cu_fused);
        // Occupancy belongs to this code object. The processor count belongs to
        // the immutable device inventory captured before communicator admission.
        capacity = std::size_t(effective_per_cu) * rocmHardwareMetadata().at(device).compute_units;
        return true;
    }

    bool launchROCmPeerPushFP32(const PeerPushBinding &binding, const float *input, float *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > binding.layout.bank_elements / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t)) return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(float) || b > UINTPTR_MAX - elements * sizeof(float) ||
            (a != b && a < b + elements * sizeof(float) && b < a + elements * sizeof(float))) return false;
        hipLaunchKernelGGL(peerPushFP32Kernel, dim3(PeerPushLayout::blocksFor(elements)),
            dim3(PeerPushLayout::threads), 0, static_cast<hipStream_t>(stream),
            binding, input, output, live_rows, rows, columns, payload_bytes, metadata_bytes);
        return hipGetLastError() == hipSuccess;
    }

    bool launchROCmPeerPushReduceFP32(const PeerPushBinding &binding, const float *input,
        float *output, int root, std::size_t count, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            root < 0 || root >= binding.layout.participants ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !count || count > binding.layout.bank_elements ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float)) return false;
        hipLaunchKernelGGL(peerPushReduceFP32Kernel, dim3(PeerPushLayout::blocksFor(count)),
            dim3(PeerPushLayout::threads), 0, static_cast<hipStream_t>(stream),
            binding, input, output, root, count);
        return hipGetLastError() == hipSuccess;
    }

    bool launchROCmPeerPushBroadcastFP32(const PeerPushBinding &binding, const float *input,
        float *output, int root, std::size_t count, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            root < 0 || root >= binding.layout.participants ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !count || count > binding.layout.bank_elements ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float)) return false;
        hipLaunchKernelGGL(peerPushBroadcastFP32Kernel, dim3(PeerPushLayout::blocksFor(count)),
            dim3(PeerPushLayout::threads), 0, static_cast<hipStream_t>(stream),
            binding, input, output, root, count);
        return hipGetLastError() == hipSuccess;
    }

    bool launchROCmPeerPushFP16(const PeerPushBinding &binding, const void *input, void *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > binding.layout.bank_elements / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(__half) ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(__half) ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t)) return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(__half) || b > UINTPTR_MAX - elements * sizeof(__half) ||
            (a != b && a < b + elements * sizeof(__half) && b < a + elements * sizeof(__half))) return false;
        hipLaunchKernelGGL(peerPushFP16Kernel, dim3(PeerPushLayout::blocksFor(elements)),
            dim3(PeerPushLayout::threads), 0, static_cast<hipStream_t>(stream),
            binding, static_cast<const __half *>(input), static_cast<__half *>(output),
            live_rows, rows, columns, payload_bytes, metadata_bytes);
        return hipGetLastError() == hipSuccess;
    }

    bool launchROCmPeerPushFP32ViaFP16(const PeerPushBinding &binding, const float *input, float *output,
        const std::int32_t *live_rows, std::size_t rows, std::size_t columns,
        unsigned long long *payload_bytes, unsigned long long *metadata_bytes, void *stream)
    {
        if (!binding.endpoints || !binding.abort || binding.rank < 0 ||
            binding.rank >= binding.layout.participants || binding.layout.participants < 2 ||
            !input || !output || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !rows || rows > std::size_t(INT32_MAX) || !columns ||
            columns > binding.layout.bank_elements / rows ||
            binding.layout.bank_elements / PeerPushLayout::elements_per_block != binding.layout.blocks ||
            reinterpret_cast<std::uintptr_t>(input) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(output) % alignof(float) ||
            reinterpret_cast<std::uintptr_t>(live_rows) % alignof(std::int32_t)) return false;
        const auto elements = rows * columns;
        const auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(output);
        if (a > UINTPTR_MAX - elements * sizeof(float) || b > UINTPTR_MAX - elements * sizeof(float) ||
            (a != b && a < b + elements * sizeof(float) && b < a + elements * sizeof(float))) return false;
        hipLaunchKernelGGL(peerPushFP32ViaFP16Kernel, dim3(PeerPushLayout::blocksFor(elements)),
            dim3(PeerPushLayout::threads), 0, static_cast<hipStream_t>(stream),
            binding, input, output, live_rows, rows, columns, payload_bytes, metadata_bytes);
        return hipGetLastError() == hipSuccess;
    }
}
