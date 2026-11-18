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
         * every readiness flag, then poll their assigned peers; large memberships
         * cannot form a publish-after-wait cycle. The block's self flag is written
         * only after all threads read its old value. A partial/empty live prefix
         * still advances this captured progress lane but transmits no padding.
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
            __threadfence_system();
            __syncthreads();
            if (threadIdx.x == 0) *reinterpret_cast<volatile std::uint64_t *>(own.flags + flag_index) = epoch;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                    *reinterpret_cast<volatile std::uint64_t *>(binding.endpoints[peer].flags + flag_index) = epoch;
            // The receiver polls locally; outgoing flag writes need no extra
            // fence. Data's preceding system fence is the publication edge.
            bool aborted = false;
            for (int peer = threadIdx.x; peer < layout.participants; peer += blockDim.x)
                if (peer != binding.rank)
                {
                    const auto index = (std::size_t(peer) * layout.blocks + blockIdx.x) * layout.flag_stride;
                    while (*reinterpret_cast<volatile std::uint64_t *>(own.flags + index) < epoch)
                    {
                        if (__atomic_load_n(binding.abort, __ATOMIC_RELAXED)) { aborted = true; break; }
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            if (__syncthreads_or(aborted)) return;
            __threadfence_system();
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
    }

    bool rocmPeerPushResidentCapacity(std::size_t &capacity)
    {
        capacity = 0;
        int per_cu = 0, device = -1, compute_units = 0;
        if (hipGetDevice(&device) != hipSuccess ||
            hipDeviceGetAttribute(&compute_units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, peerPushFP32Kernel, PeerPushLayout::threads, 0) != hipSuccess ||
            per_cu <= 0 || compute_units <= 0) return false;
        capacity = std::size_t(per_cu) * compute_units;
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
}
