/**
 * @file MoEOverlayActivationPacketDevice.inl
 * @brief Shared CUDA/HIP kernels for device-owned sparse activation packets.
 *
 * This implementation is included by one CUDA and one HIP bridge translation
 * unit.  It contains no backend-role assumptions: every launch receives the
 * exact continuation/follower lane and logical participant chosen by planning.
 * Canonical CPU-ticket ingress separately overlaps two scalar HIP mapped reads
 * while retaining its exact publication lifetime and bit-preserving permutation.
 * Packet metadata uses a deterministic block scan that preserves original
 * row/router-slot order without serializing a whole prefill packet onto one GPU
 * thread. Bulk FP32 movement is parallel and remains ordered by the caller's
 * exact stream and TransferEngine timeline publication.
 */

#pragma once

#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "execution/moe/MoEOverlayActivationAdmissionPublication.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda/atomic>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

// Launch descriptors are immutable capture state. CUDA must address the
// original parameter storage when helpers borrow it; a per-thread copy would
// materialize the joint-membership words and inflate local-memory pressure.
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define LLAMINAR_MOE_PACKET_GRID_CONSTANT const __grid_constant__
#else
#define LLAMINAR_MOE_PACKET_GRID_CONSTANT const
#endif

#include "execution/moe/MoEOverlaySharedDispatchSelection.h"

namespace llaminar2::moe_activation_packet_device
{
    /** Threads in the portable CUDA/HIP deterministic metadata compactor. */
    inline constexpr std::uint32_t kDispatchMetadataBlockThreads = 256u;
    /**
     * One-row fused dispatch metadata+copy blocks, sharing one bound value with
     * every backend launcher. On sm_120/121 the metadata chain (grant
     * validation, snapshotting, route compaction) needs the 255-register
     * budget, which a 512-thread CTA cannot supply; 256 threads keep the fused
     * single-node launch while still giving eight warps of independent mapped
     * reads for the copy. Every other compiled target (older CUDA
     * architectures and all HIP devices) keeps the original 512-thread
     * schedule: those builds were spill-free and must not change. The host
     * launchers select the matching value against the actual device, so the
     * launch never exceeds the bound of the cubin the driver will run.
     */
#if defined(__CUDA_ARCH__)
    // 256 threads keeps per-CTA register demand <= 32K regs (<= 50% of an SM on
    // SM86/SM120) so 4-CTA single-row dispatch batch kernels co-schedule immediately
    // alongside 80-CTA MoE suffix kernels without SM-register starvation while
    // issuing 4 KB/iteration of coalesced float4 PCIe writes.
    inline constexpr std::uint32_t kSingleRowDispatchThreads = 256u;
#else
    inline constexpr std::uint32_t kSingleRowDispatchThreads = 512u;
#endif

    /**
     * @brief Load one peer-published scalar without accepting a cached line.
     *
     * Node-local heterogeneous participants do not share one GPU L2. The
     * timeline acquire orders peer writes, while these backend cache-volatile
     * loads force the payload read itself to observe system memory. Ordinary
     * volatile C++ loads are insufficient because they constrain compiler
     * motion but do not select the required GPU cache policy.
     */
    template <typename Value>
    __device__ __forceinline__ Value loadPeerPublished(
        const Value *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __ldcv(address);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __builtin_nontemporal_load(address);
#else
        return *address;
#endif
    }

    /**
     * @brief Load four contiguous peer-published floats as one 16-byte transaction.
     *
     * On CUDA this emits a 128-bit cache-volatile load (__ldcv); on HIP it
     * emits a 128-bit nontemporal vector load (u32x4) with bit-preserving
     * conversion, sustaining 3-4x higher mapped-memory PCIe bandwidth on gfx906
     * than scalar 32-bit loads without rounding or altering any FP32 bits.
     */
    __device__ __forceinline__ float4 loadPeerPublishedFloat4(
        const float4 *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __ldcv(address);
#elif defined(__HIP_DEVICE_COMPILE__)
        using u32x4 = std::uint32_t __attribute__((ext_vector_type(4)));
        const u32x4 raw_vec = __builtin_nontemporal_load(
            reinterpret_cast<const u32x4 *>(address));
        float4 out;
        out.x = __uint_as_float(raw_vec[0]);
        out.y = __uint_as_float(raw_vec[1]);
        out.z = __uint_as_float(raw_vec[2]);
        out.w = __uint_as_float(raw_vec[3]);
        return out;
#else
        return *address;
#endif
    }

    /**
     * @brief Acquire one peer-owned publication word at system scope.
     *
     * A retained timeline wait gates graph scheduling, but a captured copy-
     * engine node may sit between that wait and the packet validation kernel.
     * Re-acquiring the exact publication word here carries peer descriptor and
     * metadata visibility into this compute kernel across that engine boundary.
     */
    __device__ __forceinline__ std::uint64_t loadSystemAcquire64(
        const std::uint64_t *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *const_cast<std::uint64_t *>(address));
        return reference.load(cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_load(
            address,
            __ATOMIC_ACQUIRE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        return *address;
#endif
    }

    /** @return Fresh original absorbing admission state after an unsuccessful grant attempt. */
    __device__ __forceinline__ std::uint32_t loadSystemAcquire32(
        const std::uint32_t *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> reference(
            *const_cast<std::uint32_t *>(address));
        return reference.load(cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        return *address;
#endif
    }

    /** @brief Device-side unsigned-GEQ wait for a TransferEngine-bound edge. */
    __device__ __forceinline__ void waitSystemAcquire64(
        const MoEOverlayActivationTimelineWaitDeviceBinding &binding) noexcept
    {
        while (loadSystemAcquire64(binding.signal) < binding.value)
        {
#if defined(__CUDA_ARCH__)
            __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1u);
#endif
        }
    }

    /** @brief Publish a TransferEngine-bound edge with system-release ordering. */
    __device__ __forceinline__ void storeSystemRelease64(
        const MoEOverlayActivationTimelinePublishDeviceBinding &binding) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *binding.signal);
        reference.store(binding.value, cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
        __hip_atomic_store(
            binding.signal,
            binding.value,
            __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *binding.signal = binding.value;
#endif
    }

    /**
     * @brief Publish a TransferEngine-bound edge after an explicit `__threadfence_system()`.
     *
     * Avoids emitting redundant serialized per-lane system barriers when a batch
     * kernel has already fenced all payload and metadata writes once across the CTA.
     */
    __device__ __forceinline__ void storeSystemRelaxed64(
        const MoEOverlayActivationTimelinePublishDeviceBinding &binding) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *binding.signal);
        reference.store(binding.value, cuda::memory_order_relaxed);
#elif defined(__HIP_DEVICE_COMPILE__)
        __hip_atomic_store(
            binding.signal,
            binding.value,
            __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *binding.signal = binding.value;
#endif
    }

    /**
     * @brief Snapshot a naturally aligned protocol record with fresh loads.
     *
     * Protocol records are fixed-width, 64-byte-aligned ABI objects. Loading
     * each 64-bit lane through @ref loadPeerPublished avoids mixing fields from
     * stale and current cache lines before validation computes the digest.
     */
    template <typename Record>
    __device__ __forceinline__ Record snapshotPeerPublished(
        const Record *record) noexcept
    {
        static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0u);
        Record snapshot{};
        const auto *source = reinterpret_cast<const std::uint64_t *>(record);
        auto *destination = reinterpret_cast<std::uint64_t *>(&snapshot);
#pragma unroll
        for (std::size_t lane = 0u;
             lane < sizeof(Record) / sizeof(std::uint64_t);
             ++lane)
        {
            destination[lane] = loadPeerPublished(source + lane);
        }
        return snapshot;
    }

    /**
     * @brief Cooperatively snapshot an immutable, acquired protocol record.
     * @param record Peer record held immutable until the later acknowledgement.
     * @return The same fresh snapshot in every thread of the calling block.
     *
     * All block threads must enter this function. Adjacent lanes fetch adjacent
     * words in one coalesced system-memory request, then share the result. A
     * lane-zero loop instead makes a separate cache-volatile PCIe request for
     * every word while every other warp waits at the validation barrier.
     * The preceding acquire and following acknowledgement still own lifetime;
     * shared memory only amortizes reads inside this one kernel invocation.
     */
    /**
     * @brief Fetch one peer-published control/descriptor record into caller-owned
     *        kernel-local shared memory so `-rdc=true` never links a module-wide
     *        shared-memory slab across unrelated kernels.
     */
    template <typename Record>
    static __device__ __forceinline__ Record snapshotPeerPublishedBlockWithShared(
        const Record *record,
        std::uint64_t *shared_words) noexcept
    {
        static_assert(std::is_trivially_copyable_v<Record>);
        static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0u);
        constexpr std::size_t words = sizeof(Record) / sizeof(std::uint64_t);
        const auto *source = reinterpret_cast<const std::uint64_t *>(record);
        for (std::size_t word = threadIdx.x; word < words; word += blockDim.x)
            shared_words[word] = loadPeerPublished(source + word);
        __syncthreads();
        Record snapshot{};
        __builtin_memcpy(&snapshot, shared_words, sizeof(Record));
        // Also make repeated calls safe: no warp may replace shared words
        // while a sibling still copies its snapshot from this invocation.
        __syncthreads();
        return snapshot;
    }

    template <typename Record>
    static __device__ __forceinline__ Record snapshotPeerPublishedBlock(
        const Record *record) noexcept
    {
        static_assert(std::is_trivially_copyable_v<Record>);
        static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0u);
        constexpr std::size_t words = sizeof(Record) / sizeof(std::uint64_t);
        __shared__ std::uint64_t shared_words[words];
        return snapshotPeerPublishedBlockWithShared(record, shared_words);
    }

    /**
     * @brief Copy one acquired FP32 route row without changing any payload bits.
     * @param source Immutable mapped row owned by the peer publication lease.
     * @param destination Uniquely owned row in the continuation's device bank.
     * @param width Exact live hidden width, including non-vector-aligned tails.
     *
     * All block threads stripe the row. CUDA aligned rows use 16-byte fresh
     * loads to reduce dependent PCIe round trips; unaligned geometry uses scalar
     * fresh loads. HIP retains coalesced scalar wavefront loads, which sustain
     * higher measured mapped-memory throughput on gfx906 than per-lane vectors.
     * Neither backend permits cached payloads or rounds arithmetic.
     * The caller keeps row indices stable until every thread finishes the copy.
     */
    __device__ __forceinline__ void copyPeerPublishedRow(
        const float *__restrict__ source, float *__restrict__ destination, std::size_t width) noexcept
    {
        std::size_t copied_columns = 0u;
#if defined(__CUDA_ARCH__)
        using Vector = float4;
        constexpr std::size_t lanes = 4u;
        const bool aligned =
            ((reinterpret_cast<std::uintptr_t>(source) |
              reinterpret_cast<std::uintptr_t>(destination)) & (sizeof(Vector) - 1u)) == 0u;
        const std::size_t vectors = aligned ? width / lanes : 0u;
        const std::size_t stride = blockDim.x;
        for (std::size_t v0 = threadIdx.x; v0 < vectors; v0 += 4u * stride)
        {
            // Issue 4 independent 16-byte cache-volatile PCIe loads in flight
            // before storing so GDDR6X/PCIe round-trip latency overlaps fully.
            const std::size_t v1 = v0 + stride;
            const std::size_t v2 = v0 + 2u * stride;
            const std::size_t v3 = v0 + 3u * stride;
            const auto *src_v = reinterpret_cast<const Vector *>(source);
            auto *dst_v = reinterpret_cast<Vector *>(destination);
            const Vector r0 = loadPeerPublishedFloat4(src_v + v0);
            Vector r1{}, r2{}, r3{};
            if (v1 < vectors) r1 = loadPeerPublishedFloat4(src_v + v1);
            if (v2 < vectors) r2 = loadPeerPublishedFloat4(src_v + v2);
            if (v3 < vectors) r3 = loadPeerPublishedFloat4(src_v + v3);
            dst_v[v0] = r0;
            if (v1 < vectors) dst_v[v1] = r1;
            if (v2 < vectors) dst_v[v2] = r2;
            if (v3 < vectors) dst_v[v3] = r3;
        }
        copied_columns = vectors * lanes;
#elif defined(__HIP_DEVICE_COMPILE__)
        using Vector = float4;
        constexpr std::size_t lanes = 4u;
        const bool aligned =
            ((reinterpret_cast<std::uintptr_t>(source) |
              reinterpret_cast<std::uintptr_t>(destination)) & (sizeof(Vector) - 1u)) == 0u;
        const std::size_t vectors = aligned ? width / lanes : 0u;
        for (std::size_t vector = threadIdx.x; vector < vectors; vector += 2u * blockDim.x)
        {
            // Independent 16-byte loads are issued before dependent stores so
            // PCIe round-trip latency overlaps across both CUDA and HIP.
            const auto first = loadPeerPublishedFloat4(reinterpret_cast<const Vector *>(source) + vector);
            const auto next = vector + blockDim.x;
            Vector second{};
            if (next < vectors)
                second = loadPeerPublishedFloat4(reinterpret_cast<const Vector *>(source) + next);
            reinterpret_cast<Vector *>(destination)[vector] = first;
            if (next < vectors)
                reinterpret_cast<Vector *>(destination)[next] = second;
        }
        copied_columns = vectors * lanes;
#endif
        for (std::size_t column = copied_columns + threadIdx.x;
             column < width; column += blockDim.x)
            destination[column] = loadPeerPublished(source + column);
    }

    /** @return Whether this GPU-to-GPU MainPrefill packet uses placement-invariant fixed-point wire compaction. */
    __device__ __forceinline__ bool usePrefillFixedPointWire(
        MoEOverlayActivationPrefillWirePolicy policy,
        MoEOverlayInferenceGraphRole role) noexcept
    {
        return policy == MoEOverlayActivationPrefillWirePolicy::FixedPoint32Adaptive16 &&
               role == MoEOverlayInferenceGraphRole::MainPrefill;
    }

    /**
     * @brief Exact fixed-point parameters for commutative/associative GPU prefill return folding.
     *
     * Scale 2^12 (1 LSB = 2.44140625e-4, 4x finer than FP16 at 1.0) guarantees
     * that summing up to 32 clamped routes in int32_t cannot overflow (32 * 16000 * 4096
     * = 2.097e9 < INT32_MAX).  Because integer addition is strictly commutative and
     * associative, pre-folding per-follower routes produces bitwise-identical outputs
     * across all dynamic expert movement epochs and GPU topologies.
     */
    inline constexpr float kMoEPrefillFixedPointScale = 4096.0f;
    inline constexpr float kMoEPrefillFixedPointInvScale = 1.0f / 4096.0f;
    inline constexpr float kMoEPrefillFixedPointClamp = 16000.0f;

    /** @brief Quantize one FP32 route contribution to exact IEEE round-to-nearest-even int32 fixed-point. */
    __device__ __forceinline__ std::int32_t quantizePrefillFixed32(float v) noexcept
    {
        const float clamped = fmaxf(-kMoEPrefillFixedPointClamp, fminf(kMoEPrefillFixedPointClamp, v));
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __float2int_rn(clamped * kMoEPrefillFixedPointScale);
#else
        return static_cast<std::int32_t>(clamped * kMoEPrefillFixedPointScale);
#endif
    }

    /** @brief Load 16 bytes from peer-published mapped memory as uint4. */
    __device__ __forceinline__ uint4 loadPeerPublishedUint4(const uint4 *address) noexcept
    {
        const float4 vf = loadPeerPublishedFloat4(reinterpret_cast<const float4 *>(address));
        uint4 out{};
        __builtin_memcpy(&out, &vf, sizeof(out));
        return out;
    }

    /** @brief Pack 8 in-range int32 values into one 16-byte uint4 holding 8 signed int16 values. */
    __device__ __forceinline__ uint4 pack8Int32ToInt16Uint4(const std::int32_t v[8]) noexcept
    {
        auto pack2 = [](std::int32_t a, std::int32_t b) -> std::uint32_t {
            const auto la = static_cast<std::uint16_t>(static_cast<std::int16_t>(a));
            const auto lb = static_cast<std::uint16_t>(static_cast<std::int16_t>(b));
            return static_cast<std::uint32_t>(la) | (static_cast<std::uint32_t>(lb) << 16u);
        };
        return make_uint4(
            pack2(v[0], v[1]),
            pack2(v[2], v[3]),
            pack2(v[4], v[5]),
            pack2(v[6], v[7]));
    }

    /** @brief Unpack one 16-byte uint4 (8 signed int16 values) into two bit-cast int32 float4 vectors. */
    __device__ __forceinline__ void unpackInt16Uint4To2Int32Float4(
        uint4 packed, float4 &lo, float4 &hi) noexcept
    {
        auto toFloatBits = [](std::uint16_t raw16) -> float {
            const std::int32_t s = static_cast<std::int32_t>(static_cast<std::int16_t>(raw16));
            float out = 0.0f;
            __builtin_memcpy(&out, &s, sizeof(out));
            return out;
        };
        lo.x = toFloatBits(static_cast<std::uint16_t>(packed.x & 0xffffu));
        lo.y = toFloatBits(static_cast<std::uint16_t>(packed.x >> 16u));
        lo.z = toFloatBits(static_cast<std::uint16_t>(packed.y & 0xffffu));
        lo.w = toFloatBits(static_cast<std::uint16_t>(packed.y >> 16u));
        hi.x = toFloatBits(static_cast<std::uint16_t>(packed.z & 0xffffu));
        hi.y = toFloatBits(static_cast<std::uint16_t>(packed.z >> 16u));
        hi.z = toFloatBits(static_cast<std::uint16_t>(packed.w & 0xffffu));
        hi.w = toFloatBits(static_cast<std::uint16_t>(packed.w >> 16u));
    }

    /**
     * @brief Quantize one local FP32 row to exact fixed-point int32 and store
     *        adaptively as 16-bit (5 KB) when all elements fit in int16_t or
     *        lossless 32-bit (10 KB) when any outlier exceeds [-32768, 32767].
     * @return True if stored as 32-bit int32_t, false if stored as 16-bit int16_t.
     */
    __device__ __forceinline__ bool storeRowFP32ToAdaptiveFixed16Or32(
        const float *__restrict__ source,
        float *__restrict__ destination_slot,
        std::size_t width) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const bool fast_aligned =
            (width & 7u) == 0u &&
            width <= static_cast<std::size_t>(blockDim.x) * 16u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination_slot)) & 15u) == 0u);
        if (fast_aligned)
        {
            const std::size_t chunks = width >> 3u;
            const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(source);
            std::int32_t qbuf[2][8];
            int local_wide = 0;
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t c = threadIdx.x + pass * blockDim.x;
                if (c < chunks)
                {
                    const float4 lo = src4[c * 2u];
                    const float4 hi = src4[c * 2u + 1u];
                    qbuf[pass][0] = quantizePrefillFixed32(lo.x);
                    qbuf[pass][1] = quantizePrefillFixed32(lo.y);
                    qbuf[pass][2] = quantizePrefillFixed32(lo.z);
                    qbuf[pass][3] = quantizePrefillFixed32(lo.w);
                    qbuf[pass][4] = quantizePrefillFixed32(hi.x);
                    qbuf[pass][5] = quantizePrefillFixed32(hi.y);
                    qbuf[pass][6] = quantizePrefillFixed32(hi.z);
                    qbuf[pass][7] = quantizePrefillFixed32(hi.w);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (qbuf[pass][j] < -32768 || qbuf[pass][j] > 32767)
                            local_wide = 1;
                }
            }
            const int any_wide = __syncthreads_or(local_wide);
            if (!any_wide)
            {
                auto *__restrict__ dst_u4 = reinterpret_cast<uint4 *>(destination_slot);
                for (std::size_t pass = 0u; pass < 2u; ++pass)
                {
                    const std::size_t c = threadIdx.x + pass * blockDim.x;
                    if (c < chunks)
                        dst_u4[c] = pack8Int32ToInt16Uint4(qbuf[pass]);
                }
                return false;
            }
            auto *__restrict__ dst_i4 = reinterpret_cast<int4 *>(destination_slot);
            for (std::size_t pass = 0u; pass < 2u; ++pass)
            {
                const std::size_t c = threadIdx.x + pass * blockDim.x;
                if (c < chunks)
                {
                    dst_i4[c * 2u] = make_int4(qbuf[pass][0], qbuf[pass][1], qbuf[pass][2], qbuf[pass][3]);
                    dst_i4[c * 2u + 1u] = make_int4(qbuf[pass][4], qbuf[pass][5], qbuf[pass][6], qbuf[pass][7]);
                }
            }
            return true;
        }
#endif
        auto *__restrict__ dst_i32 = reinterpret_cast<std::int32_t *>(destination_slot);
        for (std::size_t col = threadIdx.x; col < width; col += blockDim.x)
            dst_i32[col] = quantizePrefillFixed32(source[col]);
        return true;
    }

    /**
     * @brief Caller-owned shared scratch for exact Q8_1 MainPrefill dispatch wire compaction.
     *
     * Placed in kernel-local `__shared__` (not inside device helpers) so CUDA `-rdc=true`
     * never merges module-wide shared slabs across unrelated kernels.
     */
    struct alignas(16) Q8DispatchWireScratch
    {
        static constexpr std::uint32_t kBatchBlocks = 32u;
        union
        {
            std::int8_t qs[kBatchBlocks * 32u];
            uint4 qs_u4[kBatchBlocks * 2u];
        };
        float scales[kBatchBlocks];
    };

    /** @return Wire byte size of one exact-Q8 MainPrefill dispatch row (`width` qs + `width/32` FP32 scales). */
    __device__ __forceinline__ std::size_t prefillQ8DispatchWireBytes(std::size_t width) noexcept
    {
        return width + (width >> 5u) * sizeof(float);
    }

    /**
     * @brief Quantize one FP32 hidden row into the exact GPUAlignedExpert Q8_1 wire format.
     *
     * Reduces MainPrefill dispatch PCIe traffic by 3.56x (2,880 B vs 10,240 B at H=2560)
     * while preserving exact Q8_1 scale and signed int8 codes across CUDA and ROCm.
     */
    __device__ __forceinline__ void packRowFP32ToExactQ8Wire(
        const float *__restrict__ src_row,
        float *__restrict__ dst_row,
        std::size_t width,
        Q8DispatchWireScratch &scratch) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const std::size_t num_blocks = width >> 5u;
        auto *__restrict__ dst_qs_u4 = reinterpret_cast<uint4 *>(dst_row);
        auto *__restrict__ dst_scales = reinterpret_cast<float *>(
            reinterpret_cast<std::uint8_t *>(dst_row) + width);
        const std::uint32_t sub_id = threadIdx.x >> 5u;
        const std::uint32_t lane = threadIdx.x & 31u;
        const std::uint32_t num_subwarps = blockDim.x >> 5u;

        for (std::size_t base_blk = 0u; base_blk < num_blocks; base_blk += Q8DispatchWireScratch::kBatchBlocks)
        {
            const std::uint32_t batch_blks = static_cast<std::uint32_t>(
                (num_blocks - base_blk) < Q8DispatchWireScratch::kBatchBlocks
                    ? (num_blocks - base_blk)
                    : Q8DispatchWireScratch::kBatchBlocks);
            for (std::uint32_t b = sub_id; b < batch_blks; b += num_subwarps)
            {
                const float v = src_row[(base_blk + b) * 32u + lane];
                float abs_v = device_q8_activation_contract::maximumAbsoluteValue(v, 0.0f);
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                {
#if defined(__CUDA_ARCH__)
                    const float peer = __shfl_xor_sync(0xffffffffu, abs_v, mask);
#else
                    const float peer = __shfl_xor(abs_v, mask, 32);
#endif
                    abs_v = device_q8_activation_contract::maximumAbsoluteValue(abs_v, peer);
                }
                const float s = device_q8_activation_contract::scale(abs_v);
                const float inv_s = device_fp32_contract::reciprocalPositive(s);
                const std::int32_t q = device_q8_activation_contract::quantizePreparedInverse(v, inv_s);
                scratch.qs[b * 32u + lane] = static_cast<std::int8_t>(q);
                if (lane == 0u)
                    scratch.scales[b] = s;
            }
            __syncthreads();
            const std::uint32_t u4_count = batch_blks * 2u;
            for (std::uint32_t idx = threadIdx.x; idx < u4_count; idx += blockDim.x)
                dst_qs_u4[base_blk * 2u + idx] = scratch.qs_u4[idx];
            if (threadIdx.x >= 64u && (threadIdx.x - 64u) < batch_blks)
            {
                const std::uint32_t s_idx = threadIdx.x - 64u;
                dst_scales[base_blk + s_idx] = scratch.scales[s_idx];
            }
            __syncthreads();
        }
#else
        (void)src_row; (void)dst_row; (void)width; (void)scratch;
#endif
    }

    /**
     * @brief Unpack one peer-published exact-Q8 MainPrefill dispatch row back to FP32.
     *
     * Issues coalesced 128-bit non-temporal/cache-volatile PCIe bursts for qs and
     * scales in parallel, then reconstructs exact FP32 multiples `q * scale` so
     * downstream GPUAlignedExpert quantization reproduces bit-identical Q8_1 blocks.
     */
    __device__ __forceinline__ void unpackRowExactQ8WireToFP32(
        const float *__restrict__ src_row,
        float *__restrict__ dst_row,
        std::size_t width,
        Q8DispatchWireScratch &scratch) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const std::size_t num_blocks = width >> 5u;
        const auto *__restrict__ src_qs_u4 = reinterpret_cast<const uint4 *>(src_row);
        const auto *__restrict__ src_scales = reinterpret_cast<const float *>(
            reinterpret_cast<const std::uint8_t *>(src_row) + width);
        auto *__restrict__ dst4 = reinterpret_cast<float4 *>(dst_row);

        for (std::size_t base_blk = 0u; base_blk < num_blocks; base_blk += Q8DispatchWireScratch::kBatchBlocks)
        {
            const std::uint32_t batch_blks = static_cast<std::uint32_t>(
                (num_blocks - base_blk) < Q8DispatchWireScratch::kBatchBlocks
                    ? (num_blocks - base_blk)
                    : Q8DispatchWireScratch::kBatchBlocks);
            const std::uint32_t u4_count = batch_blks * 2u;
            if (threadIdx.x < u4_count)
                scratch.qs_u4[threadIdx.x] = loadPeerPublishedUint4(src_qs_u4 + base_blk * 2u + threadIdx.x);
            else if (threadIdx.x >= 64u && (threadIdx.x - 64u) < batch_blks)
            {
                const std::uint32_t s_idx = threadIdx.x - 64u;
                scratch.scales[s_idx] = loadPeerPublished(src_scales + base_blk + s_idx);
            }
            __syncthreads();
            const std::uint32_t vecs = batch_blks * 8u;
            for (std::uint32_t v = threadIdx.x; v < vecs; v += blockDim.x)
            {
                const std::uint32_t b = v >> 3u;
                const float s = scratch.scales[b];
                const std::int8_t *q4 = scratch.qs + v * 4u;
                dst4[base_blk * 8u + v] = make_float4(
                    static_cast<float>(q4[0]) * s,
                    static_cast<float>(q4[1]) * s,
                    static_cast<float>(q4[2]) * s,
                    static_cast<float>(q4[3]) * s);
            }
            __syncthreads();
        }
#else
        (void)src_row; (void)dst_row; (void)width; (void)scratch;
#endif
    }

    /** @brief Load one peer-published signed int16 row into local VRAM as bit-cast int32 words. */
    __device__ __forceinline__ void copyPeerPublishedFixed16ToInt32Bits(
        const std::int16_t *__restrict__ source,
        float *__restrict__ destination,
        std::size_t width) noexcept
    {
        std::size_t copied = 0u;
#if defined(__CUDA_ARCH__)
        if ((width & 7u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t chunks = width >> 3u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t c0 = threadIdx.x; c0 < chunks; c0 += 4u * stride)
            {
                const std::size_t c1 = c0 + stride;
                const std::size_t c2 = c0 + 2u * stride;
                const std::size_t c3 = c0 + 3u * stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + c0);
                uint4 p1{}, p2{}, p3{};
                if (c1 < chunks) p1 = loadPeerPublishedUint4(src_u4 + c1);
                if (c2 < chunks) p2 = loadPeerPublishedUint4(src_u4 + c2);
                if (c3 < chunks) p3 = loadPeerPublishedUint4(src_u4 + c3);
                float4 lo0{}, hi0{}, lo1{}, hi1{}, lo2{}, hi2{}, lo3{}, hi3{};
                unpackInt16Uint4To2Int32Float4(p0, lo0, hi0);
                dst4[c0 * 2u] = lo0;
                dst4[c0 * 2u + 1u] = hi0;
                if (c1 < chunks) { unpackInt16Uint4To2Int32Float4(p1, lo1, hi1); dst4[c1 * 2u] = lo1; dst4[c1 * 2u + 1u] = hi1; }
                if (c2 < chunks) { unpackInt16Uint4To2Int32Float4(p2, lo2, hi2); dst4[c2 * 2u] = lo2; dst4[c2 * 2u + 1u] = hi2; }
                if (c3 < chunks) { unpackInt16Uint4To2Int32Float4(p3, lo3, hi3); dst4[c3 * 2u] = lo3; dst4[c3 * 2u + 1u] = hi3; }
            }
            copied = chunks << 3u;
        }
#elif defined(__HIP_DEVICE_COMPILE__)
        if ((width & 7u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t chunks = width >> 3u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t c0 = threadIdx.x; c0 < chunks; c0 += 2u * stride)
            {
                const std::size_t c1 = c0 + stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + c0);
                uint4 p1{};
                if (c1 < chunks) p1 = loadPeerPublishedUint4(src_u4 + c1);
                float4 lo0{}, hi0{}, lo1{}, hi1{};
                unpackInt16Uint4To2Int32Float4(p0, lo0, hi0);
                dst4[c0 * 2u] = lo0;
                dst4[c0 * 2u + 1u] = hi0;
                if (c1 < chunks) { unpackInt16Uint4To2Int32Float4(p1, lo1, hi1); dst4[c1 * 2u] = lo1; dst4[c1 * 2u + 1u] = hi1; }
            }
            copied = chunks << 3u;
        }
#endif
        auto *__restrict__ dst_i32 = reinterpret_cast<std::int32_t *>(destination);
        for (std::size_t col = copied + threadIdx.x; col < width; col += blockDim.x)
            dst_i32[col] = static_cast<std::int32_t>(loadPeerPublished(source + col));
    }

    /** @brief Zero one local FP32 route row using 16-byte stores when aligned. */
    __device__ __forceinline__ void zeroLocalFP32Row(
        float *__restrict__ destination,
        std::size_t width) noexcept
    {
        std::size_t cleared = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if ((width & 3u) == 0u &&
            (reinterpret_cast<std::uintptr_t>(destination) & 15u) == 0u)
        {
            const std::size_t vecs = width >> 2u;
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                dst4[v] = z4;
            cleared = vecs << 2u;
        }
#endif
        for (std::size_t col = cleared + threadIdx.x; col < width; col += blockDim.x)
            destination[col] = 0.0f;
    }

    /** @brief Unpack one peer-published signed int16 fixed-point row into local VRAM as FP32. */
    __device__ __forceinline__ void copyPeerPublishedFixed16ToFP32(
        const std::int16_t *__restrict__ source,
        float *__restrict__ destination,
        std::size_t width) noexcept
    {
        auto unpackToFP32 = [](uint4 packed, float4 &lo, float4 &hi) noexcept {
            auto s16 = [](std::uint16_t raw16) -> float {
                return static_cast<float>(static_cast<std::int16_t>(raw16)) *
                       kMoEPrefillFixedPointInvScale;
            };
            lo.x = s16(static_cast<std::uint16_t>(packed.x & 0xffffu));
            lo.y = s16(static_cast<std::uint16_t>(packed.x >> 16u));
            lo.z = s16(static_cast<std::uint16_t>(packed.y & 0xffffu));
            lo.w = s16(static_cast<std::uint16_t>(packed.y >> 16u));
            hi.x = s16(static_cast<std::uint16_t>(packed.z & 0xffffu));
            hi.y = s16(static_cast<std::uint16_t>(packed.z >> 16u));
            hi.z = s16(static_cast<std::uint16_t>(packed.w & 0xffffu));
            hi.w = s16(static_cast<std::uint16_t>(packed.w >> 16u));
        };
        std::size_t copied = 0u;
#if defined(__CUDA_ARCH__)
        if ((width & 7u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t chunks = width >> 3u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t c0 = threadIdx.x; c0 < chunks; c0 += 4u * stride)
            {
                const std::size_t c1 = c0 + stride;
                const std::size_t c2 = c0 + 2u * stride;
                const std::size_t c3 = c0 + 3u * stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + c0);
                uint4 p1{}, p2{}, p3{};
                if (c1 < chunks) p1 = loadPeerPublishedUint4(src_u4 + c1);
                if (c2 < chunks) p2 = loadPeerPublishedUint4(src_u4 + c2);
                if (c3 < chunks) p3 = loadPeerPublishedUint4(src_u4 + c3);
                float4 lo0{}, hi0{}, lo1{}, hi1{}, lo2{}, hi2{}, lo3{}, hi3{};
                unpackToFP32(p0, lo0, hi0);
                dst4[c0 * 2u] = lo0;
                dst4[c0 * 2u + 1u] = hi0;
                if (c1 < chunks) { unpackToFP32(p1, lo1, hi1); dst4[c1 * 2u] = lo1; dst4[c1 * 2u + 1u] = hi1; }
                if (c2 < chunks) { unpackToFP32(p2, lo2, hi2); dst4[c2 * 2u] = lo2; dst4[c2 * 2u + 1u] = hi2; }
                if (c3 < chunks) { unpackToFP32(p3, lo3, hi3); dst4[c3 * 2u] = lo3; dst4[c3 * 2u + 1u] = hi3; }
            }
            copied = chunks << 3u;
        }
#elif defined(__HIP_DEVICE_COMPILE__)
        if ((width & 7u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t chunks = width >> 3u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t c0 = threadIdx.x; c0 < chunks; c0 += 2u * stride)
            {
                const std::size_t c1 = c0 + stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + c0);
                uint4 p1{};
                if (c1 < chunks) p1 = loadPeerPublishedUint4(src_u4 + c1);
                float4 lo0{}, hi0{}, lo1{}, hi1{};
                unpackToFP32(p0, lo0, hi0);
                dst4[c0 * 2u] = lo0;
                dst4[c0 * 2u + 1u] = hi0;
                if (c1 < chunks) { unpackToFP32(p1, lo1, hi1); dst4[c1 * 2u] = lo1; dst4[c1 * 2u + 1u] = hi1; }
            }
            copied = chunks << 3u;
        }
#endif
        for (std::size_t col = copied + threadIdx.x; col < width; col += blockDim.x)
            destination[col] = static_cast<float>(loadPeerPublished(source + col)) *
                               kMoEPrefillFixedPointInvScale;
    }

    /** @brief Unpack one peer-published signed int32 fixed-point row into local VRAM as FP32. */
    __device__ __forceinline__ void copyPeerPublishedFixed32ToFP32(
        const std::int32_t *__restrict__ source,
        float *__restrict__ destination,
        std::size_t width) noexcept
    {
        auto convert4 = [](uint4 p) -> float4 {
            return make_float4(
                static_cast<float>(static_cast<std::int32_t>(p.x)) * kMoEPrefillFixedPointInvScale,
                static_cast<float>(static_cast<std::int32_t>(p.y)) * kMoEPrefillFixedPointInvScale,
                static_cast<float>(static_cast<std::int32_t>(p.z)) * kMoEPrefillFixedPointInvScale,
                static_cast<float>(static_cast<std::int32_t>(p.w)) * kMoEPrefillFixedPointInvScale);
        };
        std::size_t copied = 0u;
#if defined(__CUDA_ARCH__)
        if ((width & 3u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t vecs = width >> 2u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t v0 = threadIdx.x; v0 < vecs; v0 += 4u * stride)
            {
                const std::size_t v1 = v0 + stride;
                const std::size_t v2 = v0 + 2u * stride;
                const std::size_t v3 = v0 + 3u * stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + v0);
                uint4 p1{}, p2{}, p3{};
                if (v1 < vecs) p1 = loadPeerPublishedUint4(src_u4 + v1);
                if (v2 < vecs) p2 = loadPeerPublishedUint4(src_u4 + v2);
                if (v3 < vecs) p3 = loadPeerPublishedUint4(src_u4 + v3);
                dst4[v0] = convert4(p0);
                if (v1 < vecs) dst4[v1] = convert4(p1);
                if (v2 < vecs) dst4[v2] = convert4(p2);
                if (v3 < vecs) dst4[v3] = convert4(p3);
            }
            copied = vecs << 2u;
        }
#elif defined(__HIP_DEVICE_COMPILE__)
        if ((width & 3u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(source) |
               reinterpret_cast<std::uintptr_t>(destination)) & 15u) == 0u))
        {
            const std::size_t vecs = width >> 2u;
            const auto *__restrict__ src_u4 = reinterpret_cast<const uint4 *>(source);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
            const std::size_t stride = blockDim.x;
            for (std::size_t v0 = threadIdx.x; v0 < vecs; v0 += 2u * stride)
            {
                const std::size_t v1 = v0 + stride;
                const uint4 p0 = loadPeerPublishedUint4(src_u4 + v0);
                uint4 p1{};
                if (v1 < vecs) p1 = loadPeerPublishedUint4(src_u4 + v1);
                dst4[v0] = convert4(p0);
                if (v1 < vecs) dst4[v1] = convert4(p1);
            }
            copied = vecs << 2u;
        }
#endif
        for (std::size_t col = copied + threadIdx.x; col < width; col += blockDim.x)
            destination[col] = static_cast<float>(loadPeerPublished(source + col)) *
                               kMoEPrefillFixedPointInvScale;
    }

    /** @return Raw fixed-width value for one strongly typed protocol enum. */
    template <typename Enum>
    __device__ __forceinline__ std::uint32_t raw(Enum value) noexcept
    {
        return static_cast<std::uint32_t>(value);
    }

    /** @return Whether two digest lanes are exactly equal without host operators. */
    __device__ __forceinline__ bool sameDigest(
        const MoEOverlayActivationDigest &left,
        const MoEOverlayActivationDigest &right) noexcept
    {
        return left.low == right.low && left.high == right.high;
    }

    /**
     * @brief Pack the exact placement-acquire result into terminal evidence.
     *
     * Activation endpoint status already carries one 64-bit diagnostic word.
     * On a stage-zero placement failure it records the last completed epoch
     * operation, the request ticket visible to the failing packet, and the
     * epoch currently published in that ticket's selected placement bank. The
     * three 16-bit epoch lanes are diagnostic modulo counters; operation and
     * result retain their full ABI widths. This remains entirely device-owned
     * and lets the bounded host watchdog distinguish a missing acquire, a
     * prematurely reused placement bank, and an ordinary semantic rejection
     * without adding a synchronization or another mapped diagnostic field.
     */
    /** @brief Branch-select one of the two RCU placement banks without a stack frame. */
    __device__ __forceinline__ MoEOverlayRoutePlacementBankDeviceView selectPlacementBank(
        const MoEOverlayRoutePlacementDeviceBinding &placement,
        std::uint32_t bank) noexcept
    {
        static_assert(kDeviceMoEOverlayEpochBankCount == 2u);
        return bank == 0u ? placement.banks[0] : placement.banks[1];
    }

    __device__ __forceinline__ std::uint64_t placementAcquireEvidence(
        const MoEOverlayRoutePlacementDeviceBinding &placement) noexcept
    {
        if (!placement.status)
            return 0u;
        const DeviceMoEOverlayEpochStatus status = *placement.status;
        const std::uint64_t ticket_epoch =
            placement.ticket ? placement.ticket->epoch : 0u;
        const std::uint32_t ticket_bank =
            placement.ticket
                ? static_cast<std::uint32_t>(
                      placement.ticket->selector & 1u)
                : kDeviceMoEOverlayInvalidBank;
        const auto selected =
            ticket_bank < kDeviceMoEOverlayEpochBankCount
                ? selectPlacementBank(placement, ticket_bank)
                : MoEOverlayRoutePlacementBankDeviceView{};
        const std::uint64_t selected_bank_epoch =
            selected.epoch
                ? static_cast<std::uint64_t>(*selected.epoch)
                : 0u;
        return ((status.epoch & 0xffffull) << 48u) |
               ((ticket_epoch & 0xffffull) << 32u) |
               ((selected_bank_epoch & 0xffffull) << 16u) |
               (static_cast<std::uint64_t>(status.operation & 0xffu) << 8u) |
               static_cast<std::uint64_t>(status.code & 0xffu);
    }

    /** @return Expected physical rows encoded by the immutable admission ticket. */
    __device__ __forceinline__ std::uint64_t admittedPhysicalRows(
        const MoEOverlayActivationEpochIdentity &identity) noexcept
    {
        return static_cast<std::uint64_t>(identity.request_count) *
               static_cast<std::uint64_t>(
                   identity.physical_rows_per_request);
    }

    /** @return Status record exclusively owned by one protocol endpoint. */
    __device__ __forceinline__ MoEOverlayActivationEndpointStatus *statusFor(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint) noexcept
    {
        return endpoint == MoEOverlayActivationEndpoint::Continuation
                   ? &control->continuation_status
                   : &control->follower_status;
    }

    /** @brief Publish one endpoint-owned successful operation. */
    __device__ __forceinline__ void publishSuccess(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t timeline,
        bool consumed,
        bool published,
        std::uint64_t payload_bytes = 0u,
        std::uint64_t live_rows = 0u,
        std::uint64_t live_entries = 0u) noexcept
    {
        if (!control || !grant)
            return;
        grant->code = raw(MoEOverlayActivationStatusCode::Success);
        if (consumed)
            grant->last_consumed_stage =
                static_cast<std::int32_t>(stage_ordinal);
        if (published)
            grant->last_published_stage =
                static_cast<std::int32_t>(stage_ordinal);

        /*
         * Payload geometry is bounded by the authenticated graph family. Each
         * endpoint is the sole writer and retained stages are stream-ordered,
         * so the hot counters need neither atomics nor mapped-memory traffic.
         */
        if (published &&
            (operation == MoEOverlayActivationOperation::PublishDispatch ||
             operation == MoEOverlayActivationOperation::PublishReturn))
        {
            grant->published_payload_bytes += payload_bytes;
            grant->published_live_rows += live_rows;
            grant->published_live_entries += live_entries;
            ++grant->published_stage_count;
        }

        if (stage_ordinal + 1u != grant->stage_count || !consumed ||
            !published)
        {
            return;
        }

        /* Only the terminal stage exports device-local evidence. The
         * scheduler acquires this Complete state after the graph's terminal
         * event, so it can read the preceding counters without a host shadow. */
        grant->state = raw(MoEOverlayActivationEndpointState::Complete);
        auto *status = statusFor(control, endpoint);
        status->digest = grant->digest;
        status->generation = grant->generation;
        status->observed_timeline = timeline;
        status->operation = raw(operation);
        status->code = raw(MoEOverlayActivationStatusCode::Success);
        status->last_model_layer = model_layer_index;
        status->last_consumed_stage = grant->last_consumed_stage;
        status->last_published_stage = grant->last_published_stage;
        status->published_payload_bytes = grant->published_payload_bytes;
        status->published_shared_hidden_bytes = grant->published_shared_hidden_bytes;
        status->published_live_rows = grant->published_live_rows;
        status->published_live_entries = grant->published_live_entries;
        status->published_stage_count = grant->published_stage_count;
        __threadfence_system();
        status->state = raw(MoEOverlayActivationEndpointState::Complete);
    }

    /** @brief Publish a terminal semantic failure without trapping peer streams. */
    __device__ __forceinline__ void publishFailure(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        MoEOverlayActivationStatusCode code,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t observed_timeline,
        std::uint32_t failure_diagnostic = 0u,
        std::uint32_t failure_auxiliary = 0u) noexcept
    {
        /* A retained graph must drain after a terminal packet fault so its
         * unconditional timeline nodes can release the peer. Preserve the
         * first semantic failure while those later stages no-op: overwriting
         * it with the final cascading rejection hides the actual broken edge. */
        if (grant &&
            grant->state ==
                raw(MoEOverlayActivationEndpointState::Aborted))
        {
            return;
        }
        if (grant)
        {
            grant->state = raw(MoEOverlayActivationEndpointState::Aborted);
            grant->code = raw(code);
        }
        auto *status = statusFor(control, endpoint);
        const auto identity = snapshotPeerPublished(&control->identity);
        status->digest = identity.digest;
        status->generation = identity.epoch_generation;
        status->observed_timeline = observed_timeline;
        status->operation = raw(operation);
        status->code = raw(code);
        status->last_model_layer = model_layer_index;
        status->failure_diagnostic = failure_diagnostic;
        status->failure_auxiliary = failure_auxiliary;
        status->last_consumed_stage =
            grant ? grant->last_consumed_stage : -1;
        status->last_published_stage =
            grant ? grant->last_published_stage : -1;
        status->published_payload_bytes =
            grant ? grant->published_payload_bytes : 0u;
        status->published_shared_hidden_bytes =
            grant ? grant->published_shared_hidden_bytes : 0u;
        status->published_live_rows =
            grant ? grant->published_live_rows : 0u;
        status->published_live_entries =
            grant ? grant->published_live_entries : 0u;
        status->published_stage_count =
            grant ? grant->published_stage_count : 0u;
        __threadfence_system();
        status->state = raw(MoEOverlayActivationEndpointState::Aborted);
    }

    __device__ __forceinline__ bool validControl(
        const MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant) noexcept;

    /**
     * @brief Capture every stage-zero continuation admission predicate.
     *
     * A set bit means the named predicate succeeded.  The mask is emitted only
     * after admission rejects the retained packet, so these redundant reads do
     * not add traffic to successful inference.  Bits 0 through 16 cover the
     * complete placement/grant expression in @ref prepareDispatchMetadata;
     * bits 17 through 20 distinguish the mapped-control identity edge that is
     * otherwise hidden inside @ref acquireOrValidateGrant.
     */
    static __device__ __forceinline__ std::uint32_t dispatchAdmissionDiagnostic(
        const MoEOverlayActivationDispatchPackLaunch &launch,
        bool grant_acquired) noexcept
    {
        const DeviceMoEOverlayEpochTicket ticket = launch.placement.ticket
            ? *launch.placement.ticket
            : DeviceMoEOverlayEpochTicket{};
        const std::uint32_t bank =
            static_cast<std::uint32_t>(ticket.selector & 1u);
        const std::uint64_t selector_generation = ticket.selector >> 1u;
        const bool bank_valid =
            bank < kDeviceMoEOverlayEpochBankCount;
        const auto selected = bank_valid
            ? selectPlacementBank(launch.placement, bank)
            : MoEOverlayRoutePlacementBankDeviceView{};
        const auto identity = launch.control
            ? snapshotPeerPublished(&launch.control->identity)
            : MoEOverlayActivationEpochIdentity{};
        const auto endpoint_status = launch.control
            ? snapshotPeerPublished(
                  &launch.control->continuation_status)
            : MoEOverlayActivationEndpointStatus{};
        const auto endpoint_state =
            static_cast<MoEOverlayActivationEndpointState>(
                endpoint_status.state);

        std::uint32_t mask = 0u;
        const auto record = [&](std::uint32_t bit, bool passed)
        {
            if (passed)
                mask |= std::uint32_t{1} << bit;
        };
        record(0u, grant_acquired);
        record(1u, launch.placement.valid());
        record(2u, launch.grant && launch.grant->digest.valid());
        record(3u, launch.grant && launch.grant->generation != 0u);
        record(4u, launch.grant && launch.grant->placement_epoch != 0u);
        record(
            5u,
            launch.grant &&
                launch.grant->state ==
                    raw(MoEOverlayActivationEndpointState::Active));
        record(
            6u,
            launch.grant &&
                launch.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success));
        record(
            7u,
            launch.grant &&
                launch.grant->stage_count > launch.stage_ordinal);
        record(
            8u,
            launch.grant &&
                launch.grant->physical_rows == launch.physical_rows);
        record(
            9u,
            launch.grant &&
                launch.grant->endpoint ==
                    raw(MoEOverlayActivationEndpoint::Continuation));
        record(
            10u,
            launch.grant &&
                launch.grant->graph_role !=
                    MoEOverlayInferenceGraphRole::None);
        record(11u, ticket.epoch != 0u);
        record(12u, selector_generation != 0u);
        record(
            13u,
            launch.grant &&
                ticket.epoch == launch.grant->placement_epoch);
        record(14u, bank_valid);
        record(15u, selected.valid());
        record(
            16u,
            selected.valid() &&
                static_cast<std::uint64_t>(*selected.epoch) == ticket.epoch);
        record(
            17u,
            endpoint_state == MoEOverlayActivationEndpointState::Armed ||
                endpoint_state == MoEOverlayActivationEndpointState::Active);
        record(
            18u,
            sameDigest(endpoint_status.digest, identity.digest));
        record(
            19u,
            endpoint_status.generation == identity.epoch_generation);
        record(
            20u,
            validControl(
                launch.control,
                MoEOverlayActivationEndpoint::Continuation,
                launch.stage_ordinal,
                launch.physical_rows,
                launch.target_participant_id));
        return mask;
    }

    /** @return Whether immutable mapped control bytes authenticate this graph. */
    __device__ __forceinline__ bool validControl(
        const MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant = -1) noexcept
    {
        if (!control || physical_rows <= 0)
            return false;
        // The publication word owns visibility of every peer record below.
        // Snapshotting identity before this acquire cannot borrow its ordering.
        const std::uint64_t admission_timeline =
            loadSystemAcquire64(&control->admission.ready_signal);
        const auto channel = snapshotPeerPublished(&control->channel);
        const auto identity = snapshotPeerPublished(&control->identity);
        const auto admission = snapshotPeerPublished(&control->admission);
        const auto status = snapshotPeerPublished(
            endpoint == MoEOverlayActivationEndpoint::Continuation
                ? &control->continuation_status
                : &control->follower_status);
        // Revalidate generation BEFORE the final ready acquire. A cached prior
        // ready word must never lend publication authority to a native rearm.
        const bool current_publication = moeOverlayActivationAdmissionPublicationCurrent(
            identity.epoch_generation,
            [&]() { return loadSystemAcquire64(&control->identity.epoch_generation); },
            [&]() { return loadSystemAcquire64(&control->admission.ready_signal); });
        if (channel.magic != kMoEOverlayActivationMagic ||
            channel.abi_version != kMoEOverlayActivationABIVersion || channel.channel_nonce == 0u ||
            channel.buffer_count != kMoEOverlayActivationBufferCount || channel.stage_count == 0u ||
            stage_ordinal >= channel.stage_count ||
            admission_timeline != kMoEOverlayActivationAdmissionTimeline ||
            admission.ready_signal != kMoEOverlayActivationAdmissionTimeline ||
            !current_publication ||
            admission.state != raw(MoEOverlayActivationAdmissionState::Armed) ||
            identity.epoch_generation == 0u || identity.stage_count != channel.stage_count ||
            identity.lane_ordinal != channel.lane_ordinal ||
            identity.return_role != channel.return_role ||
            identity.cohort_target != channel.cohort_target ||
            !moeOverlayActivationReturnRoleValid(identity.return_role) ||
            identity.channel_nonce != channel.channel_nonce ||
            identity.workspace_generation != channel.workspace_generation ||
            identity.stage_manifest_digest != channel.stage_manifest_digest ||
            identity.source_world_rank != channel.source_world_rank ||
            identity.target_world_rank != channel.target_world_rank ||
            identity.source_participant_id != channel.source_participant_id ||
            identity.target_participant_id != channel.target_participant_id ||
            identity.source_tier_priority != channel.source_tier_priority ||
            identity.target_tier_priority != channel.target_tier_priority ||
            identity.source_domain_ordinal != channel.source_domain_ordinal ||
            identity.target_domain_ordinal != channel.target_domain_ordinal ||
            identity.graph_role >= 32u ||
            (channel.graph_role_mask & (std::uint32_t{1} << identity.graph_role)) == 0u ||
            admittedPhysicalRows(identity) != static_cast<std::uint64_t>(physical_rows) ||
            (expected_target_participant >= 0 &&
             channel.target_participant_id != expected_target_participant) ||
            !sameDigest(identity.digest, admission.digest) ||
            !sameDigest(identity.digest, moeOverlayActivationIdentityDigest(identity)))
        {
            return false;
        }

        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            status.state);
        return sameDigest(status.digest, identity.digest) &&
               status.generation == identity.epoch_generation &&
               (state == MoEOverlayActivationEndpointState::Armed ||
                state == MoEOverlayActivationEndpointState::Active);
    }

    /**
     * @brief Acquire mapped admission once, then validate only local epoch state.
     *
     * Stage zero is ordered after the scheduler's mapped admission signal and
     * authenticates the complete channel/identity/status record. Every later
     * stage is already part of that same immutable retained executable, so it
     * checks the endpoint-private grant instead of traversing the PCIe mapping.
     */
    __device__ __forceinline__ bool acquireOrValidateGrant(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant = -1,
        const MoEOverlayRoutePlacementDeviceBinding &placement = {}) noexcept
    {
        if (!control || !grant || physical_rows <= 0)
            return false;
        const bool begins_epoch =
            operation == MoEOverlayActivationOperation::PublishDispatch ||
            operation == MoEOverlayActivationOperation::ConsumeDispatch;
        if (stage_ordinal == 0u && begins_epoch)
        {
            /*
             * The capture-stable admission wait uses a reusable non-zero
             * threshold. It can therefore pass on transaction N's still-live
             * publication when transaction N+1 has already been submitted by
             * the continuation host. The endpoint-private grant is the device
             * anti-ABA authority: an epoch-opening operation must observe a
             * fully authenticated scheduler identity strictly newer than the
             * generation it last consumed. Later stage-zero operations reuse
             * the active grant and never enter this loop.
             *
             * This wait is normally one iteration. Under intentional host
             * submission overlap it suspends only this endpoint's graph block;
             * the peer device retires N and publishes N+1 independently. No
             * host fence, graph mutation, or mutable execution-state mirror is
             * introduced.
             */
            const std::uint64_t previous_generation = grant->generation;
            MoEOverlayActivationEpochIdentity identity{};
            for (;;)
            {
                if (validControl(
                        control,
                        endpoint,
                        stage_ordinal,
                        physical_rows,
                        expected_target_participant))
                {
                    identity = snapshotPeerPublished(&control->identity);
                    if (identity.epoch_generation > previous_generation)
                        break;
                }
                // An aborted channel can never acquire another generation.
                // Drain its retained graph so teardown does not leave a block
                // polling an identity that the scheduler will never publish.
                // Successful first-iteration admission performs no extra load.
                if (moeOverlayActivationAdmissionWaitAborted(
                        loadSystemAcquire64(&control->admission.ready_signal),
                        [&]() { return loadSystemAcquire32(&control->admission.state); }))
                {
                    return false;
                }
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            }
            if (!placement.valid())
                return false;

            /*
             * The host ticket carries only a lower bound because background
             * device maintenance may publish after scheduler admission. Read
             * the endpoint's request ticket on its exact graph stream and
             * authenticate the selected durable bank before making that exact
             * epoch part of the private grant. Continuation and follower use
             * the same device-owned admission epoch, while the mapped packet
             * descriptor proves cross-device equality without a host shadow.
             */
            const DeviceMoEOverlayEpochTicket placement_ticket{
                .epoch = placement.ticket->epoch,
                .selector = placement.ticket->selector,
            };
            const std::uint32_t placement_bank =
                static_cast<std::uint32_t>(
                    placement_ticket.selector & 1u);
            const std::uint64_t placement_generation =
                placement_ticket.selector >> 1u;
            const auto selected_placement_bank =
                placement_bank < kDeviceMoEOverlayEpochBankCount
                    ? selectPlacementBank(placement, placement_bank)
                    : MoEOverlayRoutePlacementBankDeviceView{};
            if (placement_ticket.epoch == 0u ||
                placement_ticket.epoch < identity.placement_epoch_floor ||
                placement_generation == 0u ||
                placement_bank >= kDeviceMoEOverlayEpochBankCount ||
                !selected_placement_bank.valid() ||
                static_cast<std::uint64_t>(
                    *selected_placement_bank.epoch) !=
                    placement_ticket.epoch)
            {
                return false;
            }
            *grant = MoEOverlayActivationDeviceEpochGrant{
                .digest = identity.digest,
                .generation = identity.epoch_generation,
                .placement_epoch = placement_ticket.epoch,
                .published_payload_bytes = 0u,
                .published_live_rows = 0u,
                .published_live_entries = 0u,
                .published_stage_count = 0u,
                .published_shared_hidden_bytes = 0u,
                .live_rows = 0u,
                .live_entries = 0u,
                .stage_count = identity.stage_count,
                .physical_rows = physical_rows,
                .last_published_stage = -1,
                .last_consumed_stage = -1,
                .single_row_id = -1,
                .endpoint = raw(endpoint),
                .state = raw(MoEOverlayActivationEndpointState::Active),
                .code = raw(MoEOverlayActivationStatusCode::Success),
                .graph_role = static_cast<MoEOverlayInferenceGraphRole>(
                    identity.graph_role),
                .original_slot_index = identity.original_slot_index,
                .channel_cohort_target = identity.cohort_target,
                .channel_return_role = identity.return_role,
            };
        }
        return grant->digest.valid() && grant->generation != 0u &&
               grant->placement_epoch != 0u &&
               grant->stage_count > stage_ordinal &&
               grant->physical_rows == physical_rows &&
               grant->endpoint == raw(endpoint) &&
               grant->graph_role != MoEOverlayInferenceGraphRole::None &&
               grant->state ==
                   raw(MoEOverlayActivationEndpointState::Active) &&
               grant->code == raw(MoEOverlayActivationStatusCode::Success);
    }

    /**
     * @brief Acquire this grant's publication after its anti-ABA admission loop.
     * @param control Same original immutable lane acquired by the grant.
     * @param bank Leased stage bank, already validated by the retained stage.
     * @param timeline Exact positive stage visit, never allocation capacity.
     * @return Fresh acquired signal or the original absorbing cancellation word.
     * Pre-grant stream waits remain scheduling edges, but cannot authorize the
     * next request's descriptor. This exact second acquire never reads payload
     * or metadata until the newly granted publication actually exists.
     */
    __device__ __forceinline__ std::uint64_t acquireGrantedDispatchPublication(
        const MoEOverlayActivationEpochControl* control,std::uint32_t bank,std::uint64_t timeline) noexcept
    {
        return awaitMoEOverlayGrantedDispatchPublication(timeline,
            [&]() { return loadSystemAcquire64(&control->buffers[bank].dispatch_signal.value); },
            [&]() { return moeOverlayActivationAdmissionWaitAborted(
                loadSystemAcquire64(&control->admission.ready_signal),
                [&]() { return loadSystemAcquire32(&control->admission.state); }); },
            [&]() {
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            });
    }

    /** @return Whether a descriptor names the exact retained stage and payload. */
    __device__ __forceinline__ bool validDescriptor(
        const MoEOverlayActivationPayloadDescriptor &descriptor,
        const MoEOverlayActivationDeviceEpochGrant &grant,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t timeline) noexcept
    {
        return sameDigest(descriptor.digest, grant.digest) &&
               descriptor.timeline == timeline &&
               descriptor.placement_epoch == grant.placement_epoch &&
               descriptor.stage_ordinal == stage_ordinal &&
               descriptor.model_layer_index == model_layer_index;
    }

    /** Device-local result of validating one dispatch metadata launch. */
    struct DispatchMetadataPreparation
    {
        const std::int32_t *route_participants = nullptr;
        std::int32_t active_rows = 0;

        /** @return Whether validation selected a live placement bank. */
        __device__ __forceinline__ bool valid() const noexcept
        {
            return route_participants != nullptr;
        }
    };

    /**
     * @brief Validate captured control state before any packet byte is written.
     *
     * Exactly one thread calls this helper. Failures are published immediately
     * so every peer retained graph drains through the ordinary protocol edge.
     */
    __device__ __forceinline__ DispatchMetadataPreparation
    prepareDispatchMetadata(const MoEOverlayActivationDispatchPackLaunch &launch)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const DeviceMoEOverlayEpochTicket placement_ticket{
            .epoch = launch.placement.ticket->epoch,
            .selector = launch.placement.ticket->selector,
        };
        const std::uint32_t placement_bank =
            static_cast<std::uint32_t>(placement_ticket.selector & 1u);
        const std::uint64_t placement_generation =
            placement_ticket.selector >> 1u;
        const auto selected_placement =
            selectPlacementBank(launch.placement, placement_bank);
        const bool grant_acquired = acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows,
                launch.target_participant_id,
                launch.placement);
        if (!grant_acquired || placement_ticket.epoch == 0u || placement_generation == 0u ||
            placement_ticket.epoch != launch.grant->placement_epoch ||
            launch.joint_route_target.cohortTarget() != launch.grant->channel_cohort_target ||
            launch.joint_route_target.enabled() !=
                (launch.grant->channel_return_role !=
                 static_cast<std::uint32_t>(MoEOverlayActivationReturnRole::Ordinary)) ||
            placement_bank >= kDeviceMoEOverlayEpochBankCount || !selected_placement.valid() ||
            static_cast<std::uint64_t>(*selected_placement.epoch) != placement_ticket.epoch)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                placementAcquireEvidence(launch.placement),
                dispatchAdmissionDiagnostic(launch, grant_acquired),
                launch.grant
                    ? (static_cast<std::uint32_t>(
                           launch.grant->generation & 0xffffu) << 16u) |
                          static_cast<std::uint32_t>(
                              launch.grant->placement_epoch & 0xffffu)
                    : 0u);
            return {};
        }
        const std::int32_t *const overlay_route_participants =
            selected_placement.route_participants;

        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_published_stage != expected_previous ||
            (launch.stage_ordinal >= kMoEOverlayActivationBufferCount &&
             launch.grant->last_consumed_stage <
                 static_cast<std::int32_t>(
                     launch.stage_ordinal -
                     kMoEOverlayActivationBufferCount)))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return {};
        }

        const std::int32_t active_rows =
            launch.active_row_count_device
                ? *launch.active_row_count_device
                : launch.physical_rows;
        if (active_rows < 0 || active_rows > launch.physical_rows)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return {};
        }
        return {
            .route_participants = overlay_route_participants,
            .active_rows = active_rows,
        };
    }

    /**
     * @brief Decode and validate one canonical route slot.
     * @return Whether the expert, weight, and placement entry are valid.
     */
    __device__ __forceinline__ bool decodeDispatchRoute(
        const MoEOverlayActivationDispatchPackLaunch &launch,
        const std::int32_t *route_participants,
        std::size_t route_slot,
        std::int32_t *expert,
        float *weight,
        std::int32_t *participant) noexcept
    {
        const float expert_fp32 = launch.route_expert_ids_fp32[route_slot];
        const float route_weight = launch.route_weights[route_slot];
        if (!isfinite(expert_fp32) || expert_fp32 < 0.0f ||
            expert_fp32 >= static_cast<float>(launch.placement.expert_count) ||
            truncf(expert_fp32) != expert_fp32 || !isfinite(route_weight))
        {
            return false;
        }
        const auto expert_id = static_cast<std::int32_t>(expert_fp32);
        const std::int32_t route_participant =
            route_participants[expert_id];
        *expert = expert_id;
        *weight = route_weight;
        // Resolve this exact lane's source cohort, without inventing a whole
        // owner or changing the original router-slot/weight publication order.
        // Metadata's existing complete-prefix validation also authenticates the
        // publisher's entire immutable union before any shared row is written.
        if (MoEOverlayRouteTarget::fromWord(route_participant).isCohort() &&
            moeOverlayJointRouteWeightIsZero(route_weight))
        {
            *participant = -1;
            return true;
        }
        return launch.joint_route_target.resolve(expert_id, route_participant, participant);
    }

    /** @brief Publish the authenticated descriptor for one compact packet. */
    __device__ __forceinline__ void finishDispatchMetadata(
        const MoEOverlayActivationDispatchPackLaunch &launch,
        std::uint64_t compact_rows,
        std::uint64_t compact_entries,
        std::int32_t single_row_id)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        if (timeline == 0u)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::StaleGeneration,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        launch.grant->live_rows = static_cast<std::uint32_t>(compact_rows);
        launch.grant->live_entries = static_cast<std::uint32_t>(compact_entries);
        launch.grant->single_row_id =
            compact_rows == 1u ? single_row_id : -1;
        buffer.dispatch_descriptor = {
            .digest = launch.grant->digest,
            .timeline = timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = compact_rows,
            .live_entries = compact_entries,
            .payload_bytes = moeOverlayDispatchPayloadBytes(
                compact_rows,
                compact_entries,
                static_cast<std::uint32_t>(launch.packet.d_model)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/false,
            /*published=*/true,
            buffer.dispatch_descriptor.payload_bytes,
            compact_rows,
            compact_entries);
    }

    /** @brief Deterministically compact one target's row-major route metadata. */
    static __device__ __forceinline__ void packDispatchMetadata(
        const MoEOverlayActivationDispatchPackLaunch &launch)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const auto preparation = prepareDispatchMetadata(launch);
        if (!preparation.valid())
            return;

        std::uint64_t compact_rows = 0u;
        std::uint64_t compact_entries = 0u;
        std::int32_t first_row_id = -1;
        launch.packet.entry_offsets[0] = 0;
        for (std::int32_t row = 0; row < preparation.active_rows; ++row)
        {
            const std::uint64_t row_begin = compact_entries;
            for (std::int32_t slot = 0; slot < launch.packet.top_k; ++slot)
            {
                const std::size_t route_slot =
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(launch.packet.top_k) +
                    static_cast<std::size_t>(slot);
                std::int32_t expert = -1;
                std::int32_t route_participant = -1;
                float weight = 0.0f;
                if (!decodeDispatchRoute(
                        launch,
                        preparation.route_participants,
                        route_slot,
                        &expert,
                        &weight,
                        &route_participant))
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                    return;
                }
                if (route_participant != launch.target_participant_id)
                    continue;
                if (compact_entries >= launch.packet.entry_capacity)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                    return;
                }
                launch.packet.expert_ids[compact_entries] = expert;
                launch.packet.route_weights[compact_entries] = weight;
                launch.packet.original_route_slots[compact_entries] =
                    static_cast<std::int32_t>(route_slot);
                launch.packet.compact_route_slots[compact_entries] =
                    static_cast<std::int32_t>(
                        compact_rows *
                            static_cast<std::uint64_t>(
                                launch.packet.top_k) +
                        (compact_entries - row_begin));
                ++compact_entries;
            }
            if (compact_entries == row_begin)
                continue;
            if (compact_rows >= launch.packet.row_capacity)
            {
                publishFailure(
                    launch.control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::PayloadMismatch,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    0u);
                return;
            }
            if (compact_rows == 0u)
                first_row_id = row;
            launch.packet.row_ids[compact_rows] = row;
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(row_begin);
            ++compact_rows;
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(compact_entries);
        }
        finishDispatchMetadata(launch, compact_rows, compact_entries, first_row_id);
    }

    /** Two-lane scan value: matching entries and non-empty rows. */
    struct DispatchMetadataScanValue
    {
        std::uint32_t entries;
        std::uint32_t rows;
    };

    /**
     * @brief Deterministically compact multi-row metadata with one GPU block.
     *
     * A thread owns one row in each fixed-size chunk. The inclusive block scan
     * assigns stable row and entry offsets; a second slot walk writes matches in
     * original router order. Chunk carries make the same algorithm total for
     * graph buckets larger than one block without atomics or reordered output.
     */
    // Both backend bridges launch this exact scan width, not 1024 work items.
    static __global__ __launch_bounds__(kDispatchMetadataBlockThreads) void packDispatchMetadataKernel(
        LLAMINAR_MOE_PACKET_GRID_CONSTANT MoEOverlayActivationDispatchPackLaunch launch)
    {
        constexpr std::uint32_t kChunkSmemEntries = 1024u;
        __shared__ DispatchMetadataScanValue
            scan[kDispatchMetadataBlockThreads];
        __shared__ std::int32_t s_row_ids[kDispatchMetadataBlockThreads];
        __shared__ std::int32_t s_entry_offsets[kDispatchMetadataBlockThreads];
        __shared__ std::int32_t s_expert_ids[kChunkSmemEntries];
        __shared__ float s_route_weights[kChunkSmemEntries];
        __shared__ std::int32_t s_original_slots[kChunkSmemEntries];
        __shared__ std::int32_t s_compact_slots[kChunkSmemEntries];
        __shared__ const std::int32_t *route_participants;
        __shared__ std::int32_t active_rows;
        __shared__ std::uint64_t compact_rows;
        __shared__ std::uint64_t compact_entries;
        __shared__ std::int32_t first_row_id;
        __shared__ std::uint32_t packet_valid;
        __shared__ std::uint32_t chunk_invalid;

        if (blockIdx.x != 0u ||
            blockDim.x != kDispatchMetadataBlockThreads)
        {
            return;
        }
        if (threadIdx.x == 0u)
        {
            const auto preparation = prepareDispatchMetadata(launch);
            route_participants = preparation.route_participants;
            active_rows = preparation.active_rows;
            compact_rows = 0u;
            compact_entries = 0u;
            first_row_id = -1;
            packet_valid = preparation.valid() ? 1u : 0u;
            launch.packet.entry_offsets[0] = 0;
        }
        __syncthreads();
        if (packet_valid == 0u)
            return;

        for (std::uint32_t chunk_begin = 0u;
             chunk_begin < static_cast<std::uint32_t>(active_rows);
             chunk_begin += blockDim.x)
        {
            if (threadIdx.x == 0u)
                chunk_invalid = 0u;
            __syncthreads();

            const std::uint32_t row = chunk_begin + threadIdx.x;
            std::uint32_t matching_entries = 0u;
            if (row < static_cast<std::uint32_t>(active_rows))
            {
                for (std::int32_t slot = 0;
                     slot < launch.packet.top_k;
                     ++slot)
                {
                    const std::size_t route_slot =
                        static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(launch.packet.top_k) +
                        static_cast<std::size_t>(slot);
                    std::int32_t expert = -1;
                    std::int32_t participant = -1;
                    float weight = 0.0f;
                    if (!decodeDispatchRoute(
                            launch,
                            route_participants,
                            route_slot,
                            &expert,
                            &weight,
                            &participant))
                    {
                        atomicExch(&chunk_invalid, 1u);
                        continue;
                    }
                    matching_entries +=
                        participant == launch.target_participant_id ? 1u : 0u;
                }
            }

            /* Fixed-order 32-lane sub-warp inclusive scan (identical integer
             * prefix sums on CUDA, gfx906 wave64, and gfx1100 wave32 with 2
             * barriers instead of 16). */
            std::uint32_t scan_entries = matching_entries;
            std::uint32_t scan_rows = matching_entries == 0u ? 0u : 1u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            constexpr std::uint32_t kSubWarp = 32u;
            constexpr std::uint32_t kNumSubWarps =
                kDispatchMetadataBlockThreads / kSubWarp;
            const std::uint32_t sub_lane = threadIdx.x & (kSubWarp - 1u);
            const std::uint32_t sub_id = threadIdx.x >> 5u;
#pragma unroll
            for (std::uint32_t offset = 1u; offset < kSubWarp; offset <<= 1u)
            {
#if defined(__CUDA_ARCH__)
                const std::uint32_t pe =
                    __shfl_up_sync(0xffffffffu, scan_entries, offset);
                const std::uint32_t pr =
                    __shfl_up_sync(0xffffffffu, scan_rows, offset);
#else
                const std::uint32_t pe =
                    __shfl_up(scan_entries, offset, kSubWarp);
                const std::uint32_t pr =
                    __shfl_up(scan_rows, offset, kSubWarp);
#endif
                if (sub_lane >= offset)
                {
                    scan_entries += pe;
                    scan_rows += pr;
                }
            }
            if (sub_lane == kSubWarp - 1u)
            {
                scan[sub_id] = {
                    .entries = scan_entries,
                    .rows = scan_rows,
                };
            }
            __syncthreads();
            if (chunk_invalid != 0u)
            {
                if (threadIdx.x == 0u)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        MoEOverlayActivationEndpoint::Continuation,
                        MoEOverlayActivationOperation::PublishDispatch,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                return;
            }
            if (sub_id == 0u)
            {
                std::uint32_t we =
                    sub_lane < kNumSubWarps ? scan[sub_lane].entries : 0u;
                std::uint32_t wr =
                    sub_lane < kNumSubWarps ? scan[sub_lane].rows : 0u;
#pragma unroll
                for (std::uint32_t offset = 1u; offset < kNumSubWarps; offset <<= 1u)
                {
#if defined(__CUDA_ARCH__)
                    const std::uint32_t pe =
                        __shfl_up_sync(0xffffffffu, we, offset);
                    const std::uint32_t pr =
                        __shfl_up_sync(0xffffffffu, wr, offset);
#else
                    const std::uint32_t pe = __shfl_up(we, offset, kSubWarp);
                    const std::uint32_t pr = __shfl_up(wr, offset, kSubWarp);
#endif
                    if (sub_lane >= offset)
                    {
                        we += pe;
                        wr += pr;
                    }
                }
                if (sub_lane < kNumSubWarps)
                {
                    scan[sub_lane] = {.entries = we, .rows = wr};
                }
            }
            __syncthreads();
            if (sub_id > 0u)
            {
                scan_entries += scan[sub_id - 1u].entries;
                scan_rows += scan[sub_id - 1u].rows;
            }
            const auto chunk_total = scan[kNumSubWarps - 1u];
#else
            scan[threadIdx.x] = {
                .entries = scan_entries,
                .rows = scan_rows,
            };
            __syncthreads();
            if (chunk_invalid != 0u)
            {
                if (threadIdx.x == 0u)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        MoEOverlayActivationEndpoint::Continuation,
                        MoEOverlayActivationOperation::PublishDispatch,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                return;
            }
            for (std::uint32_t offset = 1u;
                 offset < blockDim.x;
                 offset <<= 1u)
            {
                DispatchMetadataScanValue prior{};
                if (threadIdx.x >= offset)
                    prior = scan[threadIdx.x - offset];
                __syncthreads();
                scan[threadIdx.x].entries += prior.entries;
                scan[threadIdx.x].rows += prior.rows;
                __syncthreads();
            }
            scan_entries = scan[threadIdx.x].entries;
            scan_rows = scan[threadIdx.x].rows;
            const auto chunk_total = scan[blockDim.x - 1u];
#endif

            const std::uint64_t entry_base = compact_entries;
            const std::uint64_t row_base = compact_rows;
            if (entry_base + chunk_total.entries >
                    launch.packet.entry_capacity ||
                row_base + chunk_total.rows > launch.packet.row_capacity)
            {
                if (threadIdx.x == 0u)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        MoEOverlayActivationEndpoint::Continuation,
                        MoEOverlayActivationOperation::PublishDispatch,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                return;
            }

            const std::uint32_t local_row =
                scan_rows - (matching_entries == 0u ? 0u : 1u);
            const std::uint32_t local_entry = scan_entries - matching_entries;
            const std::uint64_t row_offset = row_base + local_row;
            const std::uint64_t entry_offset = entry_base + local_entry;
            const bool stage_entries = chunk_total.entries <= kChunkSmemEntries;
            if (matching_entries != 0u)
            {
                if (row_offset == 0u)
                    first_row_id = static_cast<std::int32_t>(row);
                s_row_ids[local_row] = static_cast<std::int32_t>(row);
                s_entry_offsets[local_row] =
                    static_cast<std::int32_t>(entry_offset);

                std::uint32_t row_match = 0u;
                for (std::int32_t slot = 0;
                     slot < launch.packet.top_k;
                     ++slot)
                {
                    const std::size_t route_slot =
                        static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(launch.packet.top_k) +
                        static_cast<std::size_t>(slot);
                    std::int32_t expert = -1;
                    std::int32_t participant = -1;
                    float weight = 0.0f;
                    if (!decodeDispatchRoute(
                            launch,
                            route_participants,
                            route_slot,
                            &expert,
                            &weight,
                            &participant) ||
                        participant != launch.target_participant_id)
                    {
                        continue;
                    }
                    const std::int32_t compact_slot = static_cast<std::int32_t>(
                        row_offset *
                            static_cast<std::uint64_t>(launch.packet.top_k) +
                        row_match);
                    if (stage_entries)
                    {
                        const std::uint32_t se = local_entry + row_match;
                        s_expert_ids[se] = expert;
                        s_route_weights[se] = weight;
                        s_original_slots[se] =
                            static_cast<std::int32_t>(route_slot);
                        s_compact_slots[se] = compact_slot;
                    }
                    else
                    {
                        const std::uint64_t output_entry =
                            entry_offset + row_match;
                        launch.packet.expert_ids[output_entry] = expert;
                        launch.packet.route_weights[output_entry] = weight;
                        launch.packet.original_route_slots[output_entry] =
                            static_cast<std::int32_t>(route_slot);
                        launch.packet.compact_route_slots[output_entry] =
                            compact_slot;
                    }
                    ++row_match;
                }
            }
            __syncthreads();
            /* Coalesced mapped PCIe flush for chunk rows and entries. */
            if (threadIdx.x < chunk_total.rows)
            {
                launch.packet.row_ids[row_base + threadIdx.x] =
                    s_row_ids[threadIdx.x];
                launch.packet.entry_offsets[row_base + threadIdx.x] =
                    s_entry_offsets[threadIdx.x];
            }
            if (stage_entries)
            {
                for (std::uint32_t idx = threadIdx.x;
                     idx < chunk_total.entries;
                     idx += blockDim.x)
                {
                    const std::uint64_t dst = entry_base + idx;
                    launch.packet.expert_ids[dst] = s_expert_ids[idx];
                    launch.packet.route_weights[dst] = s_route_weights[idx];
                    launch.packet.original_route_slots[dst] =
                        s_original_slots[idx];
                    launch.packet.compact_route_slots[dst] =
                        s_compact_slots[idx];
                }
            }
            if (threadIdx.x == 0u)
            {
                compact_entries += chunk_total.entries;
                compact_rows += chunk_total.rows;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u)
        {
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(compact_entries);
            finishDispatchMetadata(launch, compact_rows, compact_entries, first_row_id);
        }
    }

    /**
     * @brief Gather compact live hidden rows directly into the mapped packet.
     *
     * The mapped timeline publication that follows this kernel is the release
     * edge for every payload byte. One thread caches the compact-to-source row
     * for blocks wholly contained in one row; adversarial narrow widths retain
     * the exact general lookup. This path deliberately avoids a fixed-capacity
     * copy-engine transaction at every MoE layer, which is more expensive than
     * direct mapped access for decode-sized packets on the measured topology.
     * Shared-physical launches use the separate union writer, which publishes
     * each original row required by this channel group exactly once.
     */
    static __global__ void packDispatchHiddenKernel(
        LLAMINAR_MOE_PACKET_GRID_CONSTANT MoEOverlayActivationDispatchPackLaunch launch)
    {
        __shared__ std::uint32_t block_active;
        __shared__ std::int32_t block_source_row;
        __shared__ std::uint32_t use_q8_wire;
        __shared__ Q8DispatchWireScratch q8_scratch;
        const std::size_t compact_row = static_cast<std::size_t>(blockIdx.x);
        if (threadIdx.x == 0u)
        {
            block_active = 0u;
            block_source_row = 0;
            use_q8_wire = 0u;
            if (launch.grant->state ==
                    raw(MoEOverlayActivationEndpointState::Active) &&
                compact_row < launch.grant->live_rows)
            {
                block_source_row =
                    launch.grant->live_rows == 1u
                        ? launch.grant->single_row_id
                        : loadPeerPublished(launch.packet.row_ids + compact_row);
                use_q8_wire =
                    usePrefillFixedPointWire(launch.prefill_wire_policy, launch.grant->graph_role)
                        ? 1u
                        : 0u;
                block_active = 1u;
            }
        }
        __syncthreads();
        if (block_active == 0u)
            return;

        const std::size_t width = static_cast<std::size_t>(launch.packet.d_model);
        const float *__restrict__ src_row =
            launch.hidden_rows_fp32 +
            static_cast<std::size_t>(block_source_row) * width;
        float *__restrict__ dst_row =
            launch.packet.hidden_rows_fp32 + compact_row * width;
        if (use_q8_wire != 0u && (width & 31u) == 0u && blockDim.x >= 96u &&
            (((reinterpret_cast<std::uintptr_t>(src_row) |
               reinterpret_cast<std::uintptr_t>(dst_row)) &
              15u) == 0u))
        {
            packRowFP32ToExactQ8Wire(src_row, dst_row, width, q8_scratch);
            return;
        }
        if ((width & 3u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(src_row) |
               reinterpret_cast<std::uintptr_t>(dst_row)) &
              (sizeof(float4) - 1u)) == 0u))
        {
            const std::size_t vecs = width >> 2u;
            const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(src_row);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(dst_row);
            const std::size_t stride = blockDim.x;
            for (std::size_t v0 = threadIdx.x; v0 < vecs; v0 += 2u * stride)
            {
                const std::size_t v1 = v0 + stride;
                const float4 r0 = src4[v0];
                float4 r1{};
                if (v1 < vecs)
                    r1 = src4[v1];
                dst4[v0] = r0;
                if (v1 < vecs)
                    dst4[v1] = r1;
            }
        }
        else
        {
            for (std::size_t column = threadIdx.x; column < width; column += blockDim.x)
                dst_row[column] = src_row[column];
        }
    }

    /**
     * @brief Write each needed original row once into the original shared matrix.
     * Metadata on this same stream has already validated the complete route
     * prefix and immutable group. Each CTA owns one original row; inactive and
     * unrequested rows remain untouched. No capacity DMA or union scratch exists.
     */
    static __global__ void packSharedDispatchUnionKernel(
        LLAMINAR_MOE_PACKET_GRID_CONSTANT MoEOverlayActivationDispatchPackLaunch launch)
    {
        __shared__ std::uint32_t needed;
        __shared__ std::uint32_t use_q8_wire;
        __shared__ Q8DispatchWireScratch q8_scratch;
        const auto row=static_cast<std::int32_t>(blockIdx.x);
        if(threadIdx.x==0u) {
            needed=0u;
            use_q8_wire=0u;
            const auto active=launch.active_row_count_device ? *launch.active_row_count_device : launch.physical_rows;
            if(launch.grant->state==raw(MoEOverlayActivationEndpointState::Active) && row<active) {
                const auto bank=static_cast<std::uint32_t>(launch.placement.ticket->selector&1u);
                const auto* routes=selectPlacementBank(launch.placement,bank).route_participants;
                needed=selectMoEOverlaySharedRow(launch,routes,launch.grant->placement_epoch,row)==
                    MoEOverlaySharedRowDecision::Needed;
                if (needed)
                    use_q8_wire =
                        usePrefillFixedPointWire(launch.prefill_wire_policy, launch.grant->graph_role)
                            ? 1u
                            : 0u;
            }
        }
        __syncthreads();
        if(!needed)return;
        const std::size_t width = static_cast<std::size_t>(launch.packet.d_model);
        const std::size_t row_base = static_cast<std::size_t>(row) * width;
        const float *__restrict__ src_row = launch.hidden_rows_fp32 + row_base;
        float *__restrict__ dst_row = launch.packet.hidden_rows_fp32 + row_base;
        std::size_t wire_bytes = width * sizeof(float);
        if (use_q8_wire != 0u && (width & 31u) == 0u && blockDim.x >= 96u &&
            (((reinterpret_cast<std::uintptr_t>(src_row) |
               reinterpret_cast<std::uintptr_t>(dst_row)) &
              15u) == 0u))
        {
            packRowFP32ToExactQ8Wire(src_row, dst_row, width, q8_scratch);
            wire_bytes = prefillQ8DispatchWireBytes(width);
        }
        else if ((width & 3u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(src_row) |
               reinterpret_cast<std::uintptr_t>(dst_row)) &
              (sizeof(float4) - 1u)) == 0u))
        {
            const std::size_t vecs = width >> 2u;
            const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(src_row);
            auto *__restrict__ dst4 = reinterpret_cast<float4 *>(dst_row);
            const std::size_t stride = blockDim.x;
            for (std::size_t v0 = threadIdx.x; v0 < vecs; v0 += 2u * stride)
            {
                const std::size_t v1 = v0 + stride;
                const float4 r0 = src4[v0];
                float4 r1{};
                if (v1 < vecs)
                    r1 = src4[v1];
                dst4[v0] = r0;
                if (v1 < vecs)
                    dst4[v1] = r1;
            }
        }
        else
        {
            for(std::int32_t column=threadIdx.x;column<launch.packet.d_model;column+=blockDim.x)
                copyMoEOverlaySharedRowElement(launch,row,column);
        }
        __syncthreads();
        if(threadIdx.x==0u)
            atomicAdd(reinterpret_cast<unsigned long long*>(&launch.grant->published_shared_hidden_bytes),
                static_cast<unsigned long long>(wire_bytes));
    }

    /** @brief Validate the exact dispatch descriptor before follower reads. */
    __device__ __forceinline__ void validateDispatch(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        *launch.active_row_count_device = 0;
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeDispatch;
        if (!acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows,
                /*expected_target_participant=*/-1,
                launch.placement))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                placementAcquireEvidence(launch.placement));
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage != expected_previous)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }

        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const auto &buffer = control->buffers[
            moeOverlayActivationBufferIndex(launch.stage_ordinal)];
        const std::uint64_t dispatch_timeline = acquireGrantedDispatchPublication(
            control,moeOverlayActivationBufferIndex(launch.stage_ordinal),timeline);
        if(dispatch_timeline==kMoEOverlayActivationAbortTimeline) {
            publishFailure(control,launch.grant,endpoint,operation,
                MoEOverlayActivationStatusCode::PeerAborted,launch.stage_ordinal,
                launch.model_layer_index,dispatch_timeline);
            return;
        }
        // Keep this helper's peer snapshot inside its own acquire edge even
        // when its current fused caller already waited on the same signal.
        const auto descriptor = snapshotPeerPublished(
            &buffer.dispatch_descriptor);
        bool valid = dispatch_timeline == timeline &&
                     validDescriptor(
                         descriptor,
                         *launch.grant,
                         launch.stage_ordinal,
                         launch.model_layer_index,
                         timeline) &&
                     descriptor.live_rows <=
                         static_cast<std::uint64_t>(launch.physical_rows) &&
                     descriptor.live_rows <= launch.packet.row_capacity &&
                     descriptor.live_entries <= launch.packet.entry_capacity &&
                     descriptor.live_entries <=
                         descriptor.live_rows *
                             static_cast<std::uint64_t>(launch.packet.top_k) &&
                     descriptor.payload_bytes ==
                         moeOverlayDispatchPayloadBytes(
                             descriptor.live_rows,
                             descriptor.live_entries,
                             static_cast<std::uint32_t>(
                                 launch.packet.d_model));
        std::int32_t prior_row = -1;
        std::int32_t prior_offset = 0;
        if (valid &&
            loadPeerPublished(launch.packet.entry_offsets) != 0)
            valid = false;
        for (std::uint64_t row = 0u; valid && row < descriptor.live_rows; ++row)
        {
            const std::int32_t row_id =
                loadPeerPublished(launch.packet.row_ids + row);
            const std::int32_t begin =
                loadPeerPublished(launch.packet.entry_offsets + row);
            const std::int32_t end =
                loadPeerPublished(launch.packet.entry_offsets + row + 1u);
            if (row_id <= prior_row || row_id < 0 ||
                row_id >= launch.physical_rows || begin != prior_offset ||
                end <= begin || end - begin > launch.packet.top_k ||
                end > static_cast<std::int32_t>(descriptor.live_entries))
            {
                valid = false;
                break;
            }
            for (std::int32_t entry = begin; entry < end; ++entry)
            {
                const std::int32_t original_slot = loadPeerPublished(
                    launch.packet.original_route_slots + entry);
                const std::int32_t compact_slot = loadPeerPublished(
                    launch.packet.compact_route_slots + entry);
                const std::int32_t expected_compact_slot =
                    static_cast<std::int32_t>(row) * launch.packet.top_k +
                    (entry - begin);
                const std::int32_t original_row =
                    original_slot >= 0
                        ? original_slot / launch.packet.top_k
                        : -1;
                const std::int32_t original_route =
                    original_slot >= 0
                        ? original_slot % launch.packet.top_k
                        : -1;
                const std::int32_t prior_original_slot =
                    entry == begin
                        ? row_id * launch.packet.top_k - 1
                        : loadPeerPublished(
                              launch.packet.original_route_slots +
                              entry - 1);
                if (loadPeerPublished(
                        launch.packet.expert_ids + entry) < 0 ||
                    !isfinite(loadPeerPublished(
                        launch.packet.route_weights + entry)) ||
                    original_row != row_id || original_route < 0 ||
                    original_route >= launch.packet.top_k ||
                    original_slot <= prior_original_slot ||
                    compact_slot != expected_compact_slot)
                {
                    valid = false;
                    break;
                }
            }
            prior_row = row_id;
            prior_offset = end;
        }
        if (valid &&
            prior_offset != static_cast<std::int32_t>(
                                descriptor.live_entries))
        {
            valid = false;
        }
        if (!valid)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                dispatch_timeline);
            return;
        }

        *launch.active_row_count_device =
            static_cast<std::int32_t>(descriptor.live_rows);
        launch.grant->live_rows = static_cast<std::uint32_t>(descriptor.live_rows);
        launch.grant->live_entries = static_cast<std::uint32_t>(descriptor.live_entries);
        launch.grant->single_row_id =
            descriptor.live_rows == 1u
                ? loadPeerPublished(launch.packet.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/true,
            /*published=*/false);
    }

    /**
     * @brief Cooperatively validate one multi-row dispatch packet.
     *
     * The descriptor and endpoint lifecycle remain single-authority work for
     * thread zero. Row/entry evidence lives in node-local mapped memory,
     * however, so reading thousands of values serially turns PCIe latency into
     * a per-layer bubble. One block stripes those independent checks while a
     * shared validity word records the first failure. Thread zero publishes
     * the exact same endpoint transition only after every checker arrives at
     * the final barrier.
     */
    static __global__ __launch_bounds__(256) void validateDispatchKernel(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;

        __shared__ std::uint32_t block_header_ready;
        __shared__ std::uint32_t block_valid;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint64_t block_timeline;
        __shared__ std::uint64_t block_dispatch_timeline;

        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeDispatch;
        if (threadIdx.x == 0u)
        {
            *launch.active_row_count_device = 0;
            block_header_ready = 0u;
            block_valid = 0u;
            block_live_rows = 0u;
            block_live_entries = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
            block_dispatch_timeline = 0u;

            if (!acquireOrValidateGrant(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    launch.stage_ordinal,
                    launch.physical_rows,
                    /*expected_target_participant=*/-1,
                    launch.placement))
            {
                publishFailure(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    placementAcquireEvidence(launch.placement));
            }
            else
            {
                const std::int32_t expected_previous =
                    launch.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              launch.stage_ordinal - 1u);
                if (launch.grant->last_consumed_stage != expected_previous)
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else
                {
                    const auto &buffer = control->buffers[
                        moeOverlayActivationBufferIndex(
                            launch.stage_ordinal)];
                    block_dispatch_timeline = acquireGrantedDispatchPublication(control,
                        moeOverlayActivationBufferIndex(launch.stage_ordinal),block_timeline);
                    if(block_dispatch_timeline==kMoEOverlayActivationAbortTimeline) {
                        publishFailure(control,launch.grant,endpoint,operation,
                            MoEOverlayActivationStatusCode::PeerAborted,launch.stage_ordinal,
                            launch.model_layer_index,block_dispatch_timeline);
                    } else {
                    // Only this current-grant acquire authorizes the peer bytes.
                    const auto descriptor = snapshotPeerPublished(
                        &buffer.dispatch_descriptor);
                    block_live_rows = descriptor.live_rows;
                    block_live_entries = descriptor.live_entries;
                    block_header_ready = 1u;
                    block_valid =
                        block_dispatch_timeline == block_timeline &&
                                validDescriptor(
                                    descriptor,
                                    *launch.grant,
                                    launch.stage_ordinal,
                                    launch.model_layer_index,
                                    block_timeline) &&
                                descriptor.live_rows <=
                                    static_cast<std::uint64_t>(
                                        launch.physical_rows) &&
                                descriptor.live_rows <=
                                    launch.packet.row_capacity &&
                                descriptor.live_entries <=
                                    launch.packet.entry_capacity &&
                                descriptor.live_entries <=
                                    descriptor.live_rows *
                                        static_cast<std::uint64_t>(
                                            launch.packet.top_k) &&
                                descriptor.payload_bytes ==
                                    moeOverlayDispatchPayloadBytes(
                                        descriptor.live_rows,
                                        descriptor.live_entries,
                                        static_cast<std::uint32_t>(
                                            launch.packet.d_model))
                            ? 1u
                            : 0u;
                    }
                }
            }
        }
        __syncthreads();

        if (block_header_ready != 0u && block_valid != 0u)
        {
            for (std::uint64_t row = threadIdx.x;
                 row < block_live_rows;
                 row += blockDim.x)
            {
                const std::int32_t row_id =
                    loadPeerPublished(launch.packet.row_ids + row);
                const std::int32_t begin =
                    loadPeerPublished(launch.packet.entry_offsets + row);
                const std::int32_t end =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row + 1u);
                const std::int32_t prior_row =
                    row == 0u
                        ? -1
                        : loadPeerPublished(
                              launch.packet.row_ids + row - 1u);
                const bool row_valid =
                    row_id > prior_row && row_id >= 0 &&
                    row_id < launch.physical_rows && begin >= 0 &&
                    (row != 0u || begin == 0) && end > begin &&
                    end - begin <= launch.packet.top_k &&
                    end <= static_cast<std::int32_t>(block_live_entries) &&
                    (row + 1u != block_live_rows ||
                     end == static_cast<std::int32_t>(
                                block_live_entries));
                bool route_slots_valid = row_valid;
                std::int32_t prior_original_slot =
                    row_id * launch.packet.top_k - 1;
                for (std::int32_t entry = begin;
                     route_slots_valid && entry < end;
                     ++entry)
                {
                    const std::int32_t original_slot = loadPeerPublished(
                        launch.packet.original_route_slots + entry);
                    const std::int32_t compact_slot = loadPeerPublished(
                        launch.packet.compact_route_slots + entry);
                    route_slots_valid =
                        original_slot / launch.packet.top_k == row_id &&
                        original_slot % launch.packet.top_k >= 0 &&
                        original_slot > prior_original_slot &&
                        compact_slot ==
                            static_cast<std::int32_t>(row) *
                                launch.packet.top_k +
                            (entry - begin);
                    prior_original_slot = original_slot;
                }
                if (!route_slots_valid)
                    atomicExch(&block_valid, 0u);
            }
            for (std::uint64_t entry = threadIdx.x;
                 entry < block_live_entries;
                 entry += blockDim.x)
            {
                if (loadPeerPublished(
                        launch.packet.expert_ids + entry) < 0 ||
                    !isfinite(loadPeerPublished(
                        launch.packet.route_weights + entry)))
                {
                    atomicExch(&block_valid, 0u);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_header_ready == 0u)
            return;
        if (block_valid == 0u)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                block_dispatch_timeline);
            return;
        }

        *launch.active_row_count_device =
            static_cast<std::int32_t>(block_live_rows);
        launch.grant->live_rows = block_live_rows;
        launch.grant->live_entries = block_live_entries;
        launch.grant->single_row_id =
            block_live_rows == 1u
                ? loadPeerPublished(launch.packet.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/false);
    }

    /**
     * @brief Materialize one validated mapped dispatch into follower tensors.
     *
     * Active hidden rows and route metadata are read directly from the mapped
     * packet after its timeline acquire; inactive rows and slots are cleared.
     * Grant state is endpoint-private device memory, so one thread snapshots it
     * for the block without issuing redundant mapped control loads.
     */
    static __global__ void materializeMappedDispatchKernel(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        __shared__ std::int32_t block_live_rows;
        __shared__ std::uint64_t block_hidden_payload_row;
        __shared__ std::uint32_t use_q8_wire;
        __shared__ Q8DispatchWireScratch q8_scratch;
        const std::size_t row_block = static_cast<std::size_t>(blockIdx.x);
        if (threadIdx.x == 0u)
        {
            const std::uint32_t endpoint_state = launch.grant->state;
            block_live_rows =
                endpoint_state ==
                        raw(MoEOverlayActivationEndpointState::Active)
                    ? static_cast<std::int32_t>(launch.grant->live_rows)
                    : 0;
            use_q8_wire =
                (endpoint_state == raw(MoEOverlayActivationEndpointState::Active) &&
                 usePrefillFixedPointWire(launch.prefill_wire_policy, launch.grant->graph_role))
                    ? 1u
                    : 0u;
            block_hidden_payload_row = 0u;
            if (row_block < static_cast<std::size_t>(block_live_rows))
            {
                block_hidden_payload_row =
                    launch.hidden_payload_layout ==
                            MoEOverlayActivationHiddenPayloadLayout::
                                SharedPhysicalRows
                        ? static_cast<std::uint64_t>(loadPeerPublished(
                              launch.packet.row_ids + row_block))
                        : row_block;
            }
        }
        __syncthreads();

        if (row_block < static_cast<std::size_t>(launch.physical_rows))
        {
            const std::size_t width =
                static_cast<std::size_t>(launch.packet.d_model);
            float *dst_row = launch.hidden_rows_fp32 + row_block * width;
            if (row_block < static_cast<std::size_t>(block_live_rows))
            {
                const float *src_row =
                    launch.packet.hidden_rows_fp32 +
                    static_cast<std::size_t>(block_hidden_payload_row) * width;
                if (use_q8_wire != 0u && (width & 31u) == 0u && blockDim.x >= 96u &&
                    (((reinterpret_cast<std::uintptr_t>(src_row) |
                       reinterpret_cast<std::uintptr_t>(dst_row)) &
                      15u) == 0u))
                {
                    unpackRowExactQ8WireToFP32(src_row, dst_row, width, q8_scratch);
                }
                else
                {
                    copyPeerPublishedRow(src_row, dst_row, width);
                }
            }
            else if ((width & 3u) == 0u &&
                     (reinterpret_cast<std::uintptr_t>(dst_row) &
                      (sizeof(float4) - 1u)) == 0u)
            {
                const std::size_t vecs = width >> 2u;
                auto *dst4 = reinterpret_cast<float4 *>(dst_row);
                const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                    dst4[v] = zero4;
            }
            else
            {
                for (std::size_t col = threadIdx.x; col < width; col += blockDim.x)
                    dst_row[col] = 0.0f;
            }
        }

        const std::size_t element =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t route_elements =
            static_cast<std::size_t>(launch.physical_rows) *
            static_cast<std::size_t>(launch.packet.top_k);
        if (element < route_elements)
        {
            const std::size_t row =
                element / static_cast<std::size_t>(launch.packet.top_k);
            const std::int32_t slot = static_cast<std::int32_t>(
                element % static_cast<std::size_t>(launch.packet.top_k));
            std::int32_t expert = -1;
            float weight = 0.0f;
            if (row < static_cast<std::size_t>(block_live_rows))
            {
                const std::int32_t begin =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row);
                const std::int32_t end =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row + 1u);
                if (begin + slot < end)
                {
                    expert = loadPeerPublished(
                        launch.packet.expert_ids + begin + slot);
                    weight = loadPeerPublished(
                        launch.packet.route_weights + begin + slot);
                }
            }
            launch.routing_indices_fp32[element] =
                static_cast<float>(expert);
            launch.routing_weights_fp32[element] = weight;
        }
    }

    /** @brief Validate follower state and publish the compact return descriptor. */
    __device__ __forceinline__ void packReturnMetadata(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::PublishReturn;
        if (!acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal) ||
            launch.grant->last_published_stage != expected_previous)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }

        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        if (timeline == 0u ||
            launch.grant->live_entries >
                launch.returned.route_slot_capacity ||
            launch.grant->live_entries >
                launch.grant->live_rows *
                    static_cast<std::uint64_t>(launch.dispatch.top_k))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        const MoEOverlayActivationPayloadDescriptor descriptor{
            .digest = launch.grant->digest,
            .timeline = timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = launch.grant->live_rows,
            .live_entries = launch.grant->live_entries,
            .payload_bytes = moeOverlayReturnPayloadBytesWithPolicy(
                launch.grant->live_rows,
                launch.grant->live_entries,
                static_cast<std::uint32_t>(launch.returned.d_model),
                usePrefillFixedPointWire(
                    launch.prefill_wire_policy,
                    launch.grant->graph_role)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        buffer.return_descriptor = descriptor;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/true,
            /*published=*/true,
            descriptor.payload_bytes,
            descriptor.live_rows,
            descriptor.live_entries);
    }

    /**
     * @brief Publish multi-row return metadata with coalesced row-id stores.
     *
     * Lifecycle validation and descriptor publication stay on thread zero.
     * The independent row-id copy is striped across the block so a wide
     * prefill packet does not serialize mapped reads and writes on one lane.
     */
    static __global__ void packReturnMetadataKernel(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;

        __shared__ std::uint32_t block_ready;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_timeline;
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishReturn;
        if (threadIdx.x == 0u)
        {
            block_ready = 0u;
            block_live_rows = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
            if (!acquireOrValidateGrant(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    launch.stage_ordinal,
                    launch.physical_rows))
            {
                publishFailure(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    0u);
            }
            else
            {
                const std::int32_t expected_previous =
                    launch.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              launch.stage_ordinal - 1u);
                if (launch.grant->last_consumed_stage !=
                        static_cast<std::int32_t>(
                            launch.stage_ordinal) ||
                    launch.grant->last_published_stage !=
                        expected_previous)
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else if (block_timeline == 0u ||
                         launch.grant->live_entries >
                             launch.returned.route_slot_capacity ||
                         launch.grant->live_entries >
                             launch.grant->live_rows *
                                 static_cast<std::uint64_t>(
                                     launch.dispatch.top_k))
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else
                {
                    block_live_rows = launch.grant->live_rows;
                    block_ready = 1u;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_ready == 0u)
            return;
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        const MoEOverlayActivationPayloadDescriptor descriptor{
            .digest = launch.grant->digest,
            .timeline = block_timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = block_live_rows,
            .live_entries = launch.grant->live_entries,
            .payload_bytes = moeOverlayReturnPayloadBytesWithPolicy(
                block_live_rows,
                launch.grant->live_entries,
                static_cast<std::uint32_t>(
                    launch.returned.d_model),
                usePrefillFixedPointWire(
                    launch.prefill_wire_policy,
                    launch.grant->graph_role)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        buffer.return_descriptor = descriptor;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/true,
            descriptor.payload_bytes,
            descriptor.live_rows,
            descriptor.live_entries);
    }

    /**
     * @brief Publish follower-local expert output directly into mapped pages.
     *
     * Metadata runs first on the same stream and records the validated live-row
     * prefix in the endpoint-private grant. Each block reads the two mapped
     * route identities once and stripes that row's device-to-mapped stores.
     * The subsequent timeline publication is the only
     * peer-visible release edge, so a continuation cannot observe a partial
     * return even though metadata and payload use separate kernels.
     */
    static __global__ void packMappedReturnPayloadKernel(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::uint32_t block_status_code;
        __shared__ std::int32_t block_last_published_stage;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint32_t block_use_fp16_folded;
        __shared__ std::int32_t block_slots[2];
        __shared__ std::int32_t s_begin;
        __shared__ std::int32_t s_end;
        __shared__ std::int32_t s_orig0;
        __shared__ std::int32_t s_compact_slots[32];
        if (threadIdx.x == 0u)
        {
            block_endpoint_state = launch.grant->state;
            block_status_code = launch.grant->code;
            block_last_published_stage = launch.grant->last_published_stage;
            block_live_rows = launch.grant->live_rows;
            block_live_entries = launch.grant->live_entries;
            block_use_fp16_folded = usePrefillFixedPointWire(
                launch.prefill_wire_policy, launch.grant->graph_role) ? 1u : 0u;
        }
        __syncthreads();

        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            block_endpoint_state);
        if ((state != MoEOverlayActivationEndpointState::Active &&
             state != MoEOverlayActivationEndpointState::Complete) ||
            block_status_code !=
                raw(MoEOverlayActivationStatusCode::Success) ||
            block_last_published_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal) ||
            block_live_entries > launch.returned.route_slot_capacity)
        {
            return;
        }
        const std::size_t route_slot_limit =
            static_cast<std::size_t>(launch.physical_rows) *
            static_cast<std::size_t>(launch.dispatch.top_k);
        const auto width = static_cast<std::size_t>(launch.returned.d_model);
        if (block_use_fp16_folded != 0u)
        {
            /*
             * GPU-to-GPU MainPrefill placement-invariant wire compaction:
             * quantize each route contribution to exact fixed-point int32
             * (scale 2^12), sum all follower-owned routes for this row in
             * commutative/associative int32 arithmetic, and store adaptively
             * into slot s_orig0's exclusive 4*width-byte span as int16 (5 KB)
             * when all values fit in [-32768, 32767] or lossless int32 (10 KB)
             * otherwise.  `returned.row_wide_flags[row]` carries the exact
             * per-row width tag with zero cross-follower aliasing.
             */
            const auto *src_base = launch.local_canonical_route_contributions_fp32;
            const bool vec_aligned =
                (width & 7u) == 0u &&
                width <= static_cast<std::size_t>(blockDim.x) * 16u &&
                (((reinterpret_cast<std::uintptr_t>(src_base) |
                   reinterpret_cast<std::uintptr_t>(
                       launch.returned.canonical_route_contributions_fp32)) & 15u) == 0u);
            for (std::size_t row = blockIdx.x; row < block_live_rows; row += gridDim.x)
            {
                if (threadIdx.x == 0u)
                {
                    s_begin = loadPeerPublished(launch.dispatch.entry_offsets + row);
                    s_end = loadPeerPublished(launch.dispatch.entry_offsets + row + 1u);
                }
                __syncthreads();
                const std::int32_t count = s_end - s_begin;
                if (count > 0 && s_begin >= 0 &&
                    static_cast<std::uint64_t>(s_end) <= block_live_entries)
                {
                    if (threadIdx.x == 0u)
                        s_orig0 = loadPeerPublished(launch.dispatch.original_route_slots + s_begin);
                    if (count <= 32 && threadIdx.x < static_cast<std::uint32_t>(count))
                    {
                        s_compact_slots[threadIdx.x] = loadPeerPublished(
                            launch.dispatch.compact_route_slots + s_begin + threadIdx.x);
                    }
                    __syncthreads();
                    if (s_orig0 >= 0 && static_cast<std::size_t>(s_orig0) < route_slot_limit)
                    {
                        float *dst_slot =
                            launch.returned.canonical_route_contributions_fp32 +
                            static_cast<std::size_t>(s_orig0) * width;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                        if (vec_aligned && count <= 32 && launch.returned.row_wide_flags)
                        {
                            const std::size_t chunks = width >> 3u;
                            const std::size_t f4_stride = width >> 2u;
                            const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(src_base);
                            std::int32_t acc[2][8];
                            int local_wide = 0;
                            for (std::size_t pass = 0u; pass < 2u; ++pass)
                            {
                                const std::size_t c = threadIdx.x + pass * blockDim.x;
                                if (c < chunks)
                                {
#pragma unroll
                                    for (int j = 0; j < 8; ++j)
                                        acc[pass][j] = 0;
                                    for (std::int32_t k = 0; k < count; ++k)
                                    {
                                        const std::int32_t comp = s_compact_slots[k];
                                        if (comp >= 0 && static_cast<std::size_t>(comp) < route_slot_limit)
                                        {
                                            const std::size_t base4 =
                                                static_cast<std::size_t>(comp) * f4_stride + c * 2u;
                                            const float4 v0 = src4[base4];
                                            const float4 v1 = src4[base4 + 1u];
                                            acc[pass][0] += quantizePrefillFixed32(v0.x);
                                            acc[pass][1] += quantizePrefillFixed32(v0.y);
                                            acc[pass][2] += quantizePrefillFixed32(v0.z);
                                            acc[pass][3] += quantizePrefillFixed32(v0.w);
                                            acc[pass][4] += quantizePrefillFixed32(v1.x);
                                            acc[pass][5] += quantizePrefillFixed32(v1.y);
                                            acc[pass][6] += quantizePrefillFixed32(v1.z);
                                            acc[pass][7] += quantizePrefillFixed32(v1.w);
                                        }
                                    }
#pragma unroll
                                    for (int j = 0; j < 8; ++j)
                                        if (acc[pass][j] < -32768 || acc[pass][j] > 32767)
                                            local_wide = 1;
                                }
                            }
                            const int any_wide = __syncthreads_or(local_wide);
                            if (threadIdx.x == 0u)
                                launch.returned.row_wide_flags[row] = any_wide ? 1 : 0;
                            if (!any_wide)
                            {
                                auto *__restrict__ dst_u4 = reinterpret_cast<uint4 *>(dst_slot);
                                for (std::size_t pass = 0u; pass < 2u; ++pass)
                                {
                                    const std::size_t c = threadIdx.x + pass * blockDim.x;
                                    if (c < chunks)
                                        dst_u4[c] = pack8Int32ToInt16Uint4(acc[pass]);
                                }
                            }
                            else
                            {
                                auto *__restrict__ dst_i4 = reinterpret_cast<int4 *>(dst_slot);
                                for (std::size_t pass = 0u; pass < 2u; ++pass)
                                {
                                    const std::size_t c = threadIdx.x + pass * blockDim.x;
                                    if (c < chunks)
                                    {
                                        dst_i4[c * 2u] = make_int4(
                                            acc[pass][0], acc[pass][1], acc[pass][2], acc[pass][3]);
                                        dst_i4[c * 2u + 1u] = make_int4(
                                            acc[pass][4], acc[pass][5], acc[pass][6], acc[pass][7]);
                                    }
                                }
                            }
                        }
                        else
#endif
                        {
                            if (threadIdx.x == 0u && launch.returned.row_wide_flags)
                                launch.returned.row_wide_flags[row] = 1;
                            auto *__restrict__ dst_i32 = reinterpret_cast<std::int32_t *>(dst_slot);
                            for (std::size_t col = threadIdx.x; col < width; col += blockDim.x)
                            {
                                std::int32_t sum = 0;
                                for (std::int32_t k = 0; k < count; ++k)
                                {
                                    const std::int32_t comp = count <= 32
                                        ? s_compact_slots[k]
                                        : loadPeerPublished(launch.dispatch.compact_route_slots + s_begin + k);
                                    if (comp >= 0 && static_cast<std::size_t>(comp) < route_slot_limit)
                                        sum += quantizePrefillFixed32(
                                            src_base[static_cast<std::size_t>(comp) * width + col]);
                                }
                                dst_i32[col] = sum;
                            }
                        }
                    }
                }
                __syncthreads();
            }
            return;
        }
        for (std::size_t entry = blockIdx.x; entry < block_live_entries;
             entry += gridDim.x)
        {
            // Two independent system-memory reads replace two reads for every
            // output scalar. The dispatch lease pins these identities until
            // the stream-ordered return publication releases the complete row.
            if (threadIdx.x < 2u)
            {
                const auto *slots = threadIdx.x == 0u
                    ? launch.dispatch.original_route_slots
                    : launch.dispatch.compact_route_slots;
                block_slots[threadIdx.x] = loadPeerPublished(slots + entry);
            }
            __syncthreads();
            const auto original_slot = block_slots[0];
            const auto compact_slot = block_slots[1];
            if (original_slot >= 0 && compact_slot >= 0 &&
                static_cast<std::size_t>(original_slot) < route_slot_limit &&
                static_cast<std::size_t>(compact_slot) < route_slot_limit)
            {
                auto *destination = launch.returned.canonical_route_contributions_fp32 +
                    static_cast<std::size_t>(original_slot) * width;
                const auto *source = launch.local_canonical_route_contributions_fp32 +
                    static_cast<std::size_t>(compact_slot) * width;
                if ((width & 3u) == 0u &&
                    (((reinterpret_cast<std::uintptr_t>(destination) |
                       reinterpret_cast<std::uintptr_t>(source)) &
                      (sizeof(float4) - 1u)) == 0u))
                {
                    const std::size_t vecs = width >> 2u;
                    auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
                    const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(source);
                    for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                        dst4[v] = src4[v];
                }
                else
                {
                    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x)
                        destination[column] = source[column];
                }
            }
            // A grid-stride iteration may reuse shared identities only once
            // every warp has finished all stores for the preceding route.
            __syncthreads();
        }
    }

    /**
     * @brief Authenticate one acquired return using all threads in its block.
     * @param packet One participant's immutable lane binding and private grant.
     *
     * The caller's stream/acquire edge pins the peer descriptor and row maps.
     * Only lane zero transitions the grant. Descriptor words and route identity
     * checks run cooperatively, so one-row MTP and multi-row prefill execute one
     * validation protocol without serial per-field/per-route PCIe round trips.
     * Every thread must enter; callers that consume publication inside this
     * kernel must meet at a block barrier after the helper returns.
     */
    /** @brief Kernel-local shared scratch for `validateReturnBlock` (avoids CUDA `-rdc=true` global `.shared` merge). */
    struct ValidateReturnBlockShared
    {
        std::uint64_t descriptor_words[sizeof(MoEOverlayActivationPayloadDescriptor) / sizeof(std::uint64_t)];
        std::uint64_t block_live_rows;
        std::uint64_t block_live_entries;
        std::uint64_t block_timeline;
        std::uint64_t block_return_timeline;
        std::uint32_t block_header_ready;
        std::uint32_t block_valid;
        std::int32_t orig_chunk[257];
        std::int32_t comp_chunk[256];
    };

    static __device__ __forceinline__ void validateReturnBlock(
        MoEOverlayActivationReturnConsumeLaunch packet,
        ValidateReturnBlockShared &shared)
    {
        auto &block_header_ready = shared.block_header_ready;
        auto &block_valid = shared.block_valid;
        auto &block_live_rows = shared.block_live_rows;
        auto &block_live_entries = shared.block_live_entries;
        auto &block_timeline = shared.block_timeline;
        auto &block_return_timeline = shared.block_return_timeline;
        auto *const control =
            const_cast<MoEOverlayActivationEpochControl *>(packet.control);
        constexpr auto endpoint =
            MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeReturn;
        if (threadIdx.x == 0u)
        {
            block_header_ready = 0u;
            block_valid = 0u;
            block_live_rows = 0u;
            block_live_entries = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(packet.stage_ordinal));
            block_return_timeline = 0u;
            if (!acquireOrValidateGrant(
                    control,
                    packet.grant,
                    endpoint,
                    operation,
                    packet.stage_ordinal,
                    packet.physical_rows))
            {
                publishFailure(
                    control,
                    packet.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    packet.stage_ordinal,
                    packet.model_layer_index,
                    0u);
            }
            else
            {
                const std::int32_t expected_previous =
                    packet.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              packet.stage_ordinal - 1u);
                if (packet.grant->last_published_stage <
                        static_cast<std::int32_t>(packet.stage_ordinal) ||
                    packet.grant->last_consumed_stage != expected_previous)
                {
                    publishFailure(
                        control,
                        packet.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        packet.stage_ordinal,
                        packet.model_layer_index,
                        0u);
                }
                else
                {
                    const auto &buffer = control->buffers[
                        moeOverlayActivationBufferIndex(
                            packet.stage_ordinal)];
                    block_return_timeline = loadSystemAcquire64(
                        &buffer.return_signal.value);
                    block_header_ready = 1u;
                }
            }
        }
        __syncthreads();

        // The acquire above owns visibility and the live lane owns lifetime.
        // After the block barrier every lane can fetch a different descriptor
        // word, instead of lane zero issuing eight serial mapped-memory reads.
        // Invalid grants exit uniformly before entering the cooperative helper.
        if (block_header_ready == 0u)
            return;
        const auto &buffer = control->buffers[
            moeOverlayActivationBufferIndex(packet.stage_ordinal)];
        const auto descriptor = snapshotPeerPublishedBlockWithShared(
            &buffer.return_descriptor, shared.descriptor_words);
        if (threadIdx.x == 0u)
        {
            block_live_rows = descriptor.live_rows;
            block_live_entries = descriptor.live_entries;
            block_valid =
                block_return_timeline == block_timeline &&
                        validDescriptor(
                            descriptor,
                            *packet.grant,
                            packet.stage_ordinal,
                            packet.model_layer_index,
                            block_timeline) &&
                        moeOverlayActivationReturnCountsMatch(packet.grant->channel_return_role,
                            descriptor.live_rows,descriptor.live_entries,
                            packet.grant->live_rows,packet.grant->live_entries) &&
                        descriptor.live_entries <= packet.returned.route_slot_capacity &&
                        descriptor.live_entries <= descriptor.live_rows *
                            static_cast<std::uint64_t>(packet.dispatch.top_k) &&
                        descriptor.live_rows <=
                            static_cast<std::uint64_t>(packet.physical_rows) &&
                        descriptor.payload_bytes == moeOverlayReturnPayloadBytesWithPolicy(
                            descriptor.live_rows,
                            descriptor.live_entries,
                            static_cast<std::uint32_t>(packet.returned.d_model),
                            usePrefillFixedPointWire(
                                packet.prefill_wire_policy,
                                packet.grant->graph_role))
                    ? 1u : 0u;
        }
        __syncthreads();

        if (block_header_ready != 0u && block_valid != 0u)
        {
            const std::uint64_t route_slot_limit =
                static_cast<std::uint64_t>(packet.physical_rows) *
                static_cast<std::uint64_t>(packet.dispatch.top_k);
            // Stage each chunk of route slots once into shared memory using
            // coalesced 16-byte peer loads when aligned, eliminating the
            // redundant unaligned (entry - 1) PCIe read on every slot.
            const std::uint32_t chunk_cap = blockDim.x < 256u ? blockDim.x : 256u;
            const bool orig_aligned16 =
                (reinterpret_cast<std::uintptr_t>(packet.dispatch.original_route_slots) & 15u) == 0u;
            const bool comp_aligned16 =
                (reinterpret_cast<std::uintptr_t>(packet.dispatch.compact_route_slots) & 15u) == 0u;
            std::int32_t prev_chunk_last = -1;
            std::uint32_t local_invalid = 0u;
            for (std::uint64_t base = 0u; base < block_live_entries; base += chunk_cap)
            {
                const std::uint32_t chunk_len = static_cast<std::uint32_t>(
                    (block_live_entries - base) < chunk_cap ? (block_live_entries - base) : chunk_cap);
                if (threadIdx.x == 0u)
                    shared.orig_chunk[0] = prev_chunk_last;
                const std::uint32_t vec_count = chunk_len >> 2u;
                const std::uint32_t vec_tail = vec_count << 2u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                if (orig_aligned16 && ((base & 3u) == 0u) && vec_count > 0u)
                {
                    if (threadIdx.x < vec_count)
                    {
                        const float4 vf = loadPeerPublishedFloat4(
                            reinterpret_cast<const float4 *>(packet.dispatch.original_route_slots + base) + threadIdx.x);
                        std::int32_t tmp[4];
                        __builtin_memcpy(tmp, &vf, sizeof(tmp));
                        const std::uint32_t idx = threadIdx.x << 2u;
                        shared.orig_chunk[1u + idx] = tmp[0];
                        shared.orig_chunk[2u + idx] = tmp[1];
                        shared.orig_chunk[3u + idx] = tmp[2];
                        shared.orig_chunk[4u + idx] = tmp[3];
                    }
                    if (threadIdx.x >= vec_tail && threadIdx.x < chunk_len)
                    {
                        shared.orig_chunk[1u + threadIdx.x] = loadPeerPublished(
                            packet.dispatch.original_route_slots + base + threadIdx.x);
                    }
                }
                else
#endif
                {
                    if (threadIdx.x < chunk_len)
                    {
                        shared.orig_chunk[1u + threadIdx.x] = loadPeerPublished(
                            packet.dispatch.original_route_slots + base + threadIdx.x);
                    }
                }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                if (comp_aligned16 && ((base & 3u) == 0u) && vec_count > 0u)
                {
                    if (threadIdx.x < vec_count)
                    {
                        const float4 vf = loadPeerPublishedFloat4(
                            reinterpret_cast<const float4 *>(packet.dispatch.compact_route_slots + base) + threadIdx.x);
                        std::int32_t tmp[4];
                        __builtin_memcpy(tmp, &vf, sizeof(tmp));
                        const std::uint32_t idx = threadIdx.x << 2u;
                        shared.comp_chunk[idx] = tmp[0];
                        shared.comp_chunk[idx + 1u] = tmp[1];
                        shared.comp_chunk[idx + 2u] = tmp[2];
                        shared.comp_chunk[idx + 3u] = tmp[3];
                    }
                    if (threadIdx.x >= vec_tail && threadIdx.x < chunk_len)
                    {
                        shared.comp_chunk[threadIdx.x] = loadPeerPublished(
                            packet.dispatch.compact_route_slots + base + threadIdx.x);
                    }
                }
                else
#endif
                {
                    if (threadIdx.x < chunk_len)
                    {
                        shared.comp_chunk[threadIdx.x] = loadPeerPublished(
                            packet.dispatch.compact_route_slots + base + threadIdx.x);
                    }
                }
                __syncthreads();
                if (threadIdx.x < chunk_len)
                {
                    const std::int32_t original_slot = shared.orig_chunk[1u + threadIdx.x];
                    const std::int32_t prior_original_slot = shared.orig_chunk[threadIdx.x];
                    const std::int32_t compact_slot = shared.comp_chunk[threadIdx.x];
                    if (original_slot <= prior_original_slot ||
                        original_slot < 0 || compact_slot < 0 ||
                        static_cast<std::uint64_t>(original_slot) >= route_slot_limit ||
                        static_cast<std::uint64_t>(compact_slot) >= route_slot_limit)
                    {
                        local_invalid = 1u;
                    }
                }
                prev_chunk_last = shared.orig_chunk[chunk_len];
                __syncthreads();
            }
            if (local_invalid != 0u)
                atomicExch(&block_valid, 0u);
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_header_ready == 0u)
            return;
        if (block_valid == 0u)
        {
            publishFailure(
                control,
                packet.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                packet.stage_ordinal,
                packet.model_layer_index,
                block_return_timeline);
            return;
        }
        packet.grant->live_rows = block_live_rows;
        packet.grant->live_entries = block_live_entries;
        packet.grant->single_row_id =
            block_live_rows == 1u
                ? loadPeerPublished(packet.dispatch.row_ids)
                : -1;
        publishSuccess(
            control,
            packet.grant,
            endpoint,
            operation,
            packet.stage_ordinal,
            packet.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/true);
    }

    /**
     * @return Whether this exact stage published an authenticated return grant.
     * @param packet Lane whose device-local grant is already stream ordered.
     */
    __device__ __forceinline__ bool returnGrantPublished(
        const MoEOverlayActivationReturnConsumeLaunch &packet)
    {
        const auto state = static_cast<MoEOverlayActivationEndpointState>(packet.grant->state);
        return (state == MoEOverlayActivationEndpointState::Active ||
                state == MoEOverlayActivationEndpointState::Complete) &&
               packet.grant->code == raw(MoEOverlayActivationStatusCode::Success) &&
               packet.grant->last_consumed_stage == static_cast<std::int32_t>(packet.stage_ordinal);
    }

    /** @brief Single-lane graph entrypoint for the shared block validator. */
    static __global__ void validateReturnKernel(
        MoEOverlayActivationReturnConsumeLaunch launch)
    {
        __shared__ ValidateReturnBlockShared shared;
        if (blockIdx.x == 0u)
            validateReturnBlock(launch, shared);
    }

    /**
     * @brief Materialize one authenticated participant's original route rows.
     *
     * Each compact entry names one unique original route slot.  The follower
     * already wrote that slot into the rank-pair shared matrix; this kernel
     * copies it into continuation VRAM without performing an addition.  The
     * later canonical reducer consequently sees the same slot tensor for every
     * placement and owns the only floating-point fold.
     */
    static __global__ void materializeMappedCanonicalReturnKernel(
        MoEOverlayActivationReturnConsumeLaunch launch)
    {
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::uint32_t block_status_code;
        __shared__ std::int32_t block_last_consumed_stage;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint32_t block_use_fp16_folded;
        __shared__ std::int32_t s_begin;
        __shared__ std::int32_t s_end;
        __shared__ std::int32_t s_row_wide;
        __shared__ std::int32_t s_orig_slots[32];
        if (threadIdx.x == 0u)
        {
            block_endpoint_state = launch.grant->state;
            block_status_code = launch.grant->code;
            block_last_consumed_stage = launch.grant->last_consumed_stage;
            block_live_rows = launch.grant->live_rows;
            block_live_entries = launch.grant->live_entries;
            block_use_fp16_folded = usePrefillFixedPointWire(
                launch.prefill_wire_policy, launch.grant->graph_role) ? 1u : 0u;
        }
        __syncthreads();

        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            block_endpoint_state);
        if ((state != MoEOverlayActivationEndpointState::Active &&
             state != MoEOverlayActivationEndpointState::Complete) ||
            block_status_code !=
                raw(MoEOverlayActivationStatusCode::Success) ||
            block_last_consumed_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal))
        {
            return;
        }
        if (block_live_entries > launch.returned.route_slot_capacity)
        {
            return;
        }
        if (block_use_fp16_folded != 0u)
        {
            const std::size_t width = static_cast<std::size_t>(launch.returned.d_model);
            for (std::size_t row = blockIdx.x; row < block_live_rows; row += gridDim.x)
            {
                if (threadIdx.x == 0u)
                {
                    s_begin = launch.dispatch.entry_offsets[row];
                    s_end = launch.dispatch.entry_offsets[row + 1u];
                    s_row_wide = launch.returned.row_wide_flags
                        ? loadPeerPublished(launch.returned.row_wide_flags + row)
                        : 1;
                }
                __syncthreads();
                const std::int32_t count = s_end - s_begin;
                if (count > 0 && s_begin >= 0 &&
                    static_cast<std::uint64_t>(s_end) <= block_live_entries)
                {
                    if (count <= 32)
                    {
                        if (threadIdx.x < static_cast<std::uint32_t>(count))
                            s_orig_slots[threadIdx.x] = launch.dispatch.original_route_slots[s_begin + threadIdx.x];
                        __syncthreads();
                        const std::int32_t orig0 = s_orig_slots[0];
                        if (orig0 >= 0 &&
                            static_cast<std::size_t>(orig0) < launch.returned.route_slot_capacity)
                        {
                            const std::size_t offset = static_cast<std::size_t>(orig0) * width;
                            const float *slot_base =
                                launch.returned.canonical_route_contributions_fp32 + offset;
                            if (s_row_wide == 0)
                            {
                                copyPeerPublishedFixed16ToInt32Bits(
                                    reinterpret_cast<const std::int16_t *>(slot_base),
                                    launch.canonical_route_contributions_fp32 + offset,
                                    width);
                            }
                            else
                            {
                                copyPeerPublishedRow(
                                    slot_base,
                                    launch.canonical_route_contributions_fp32 + offset,
                                    width);
                            }
                        }
                        for (std::int32_t k = 1; k < count; ++k)
                        {
                            const std::int32_t orig_k = s_orig_slots[k];
                            if (orig_k >= 0 &&
                                static_cast<std::size_t>(orig_k) < launch.returned.route_slot_capacity)
                            {
                                zeroLocalFP32Row(
                                    launch.canonical_route_contributions_fp32 +
                                        static_cast<std::size_t>(orig_k) * width,
                                    width);
                            }
                        }
                    }
                    else
                    {
                        if (threadIdx.x == 0u)
                            s_orig_slots[0] = launch.dispatch.original_route_slots[s_begin];
                        __syncthreads();
                        const std::int32_t orig0 = s_orig_slots[0];
                        if (orig0 >= 0 &&
                            static_cast<std::size_t>(orig0) < launch.returned.route_slot_capacity)
                        {
                            const std::size_t offset = static_cast<std::size_t>(orig0) * width;
                            const float *slot_base =
                                launch.returned.canonical_route_contributions_fp32 + offset;
                            if (s_row_wide == 0)
                            {
                                copyPeerPublishedFixed16ToInt32Bits(
                                    reinterpret_cast<const std::int16_t *>(slot_base),
                                    launch.canonical_route_contributions_fp32 + offset,
                                    width);
                            }
                            else
                            {
                                copyPeerPublishedRow(
                                    slot_base,
                                    launch.canonical_route_contributions_fp32 + offset,
                                    width);
                            }
                        }
                        for (std::int32_t k = 1; k < count; ++k)
                        {
                            __syncthreads();
                            if (threadIdx.x == 0u)
                                s_orig_slots[0] = launch.dispatch.original_route_slots[s_begin + k];
                            __syncthreads();
                            const std::int32_t orig_k = s_orig_slots[0];
                            if (orig_k >= 0 &&
                                static_cast<std::size_t>(orig_k) < launch.returned.route_slot_capacity)
                            {
                                zeroLocalFP32Row(
                                    launch.canonical_route_contributions_fp32 +
                                        static_cast<std::size_t>(orig_k) * width,
                                    width);
                            }
                        }
                    }
                }
                __syncthreads();
            }
            return;
        }
        __shared__ std::int32_t original_slot;
        for (std::size_t compact_entry = blockIdx.x;
             compact_entry < block_live_entries; compact_entry += gridDim.x)
        {
            if (threadIdx.x == 0u)
                original_slot = loadPeerPublished(
                    launch.dispatch.original_route_slots + compact_entry);
            __syncthreads();
            if (original_slot >= 0 &&
                static_cast<std::size_t>(original_slot) < launch.returned.route_slot_capacity)
            {
                const std::size_t offset =
                    static_cast<std::size_t>(original_slot) * launch.returned.d_model;
                copyPeerPublishedRow(
                    launch.returned.canonical_route_contributions_fp32 + offset,
                    launch.canonical_route_contributions_fp32 + offset,
                    launch.returned.d_model);
            }
            // A later route must not replace the shared index under another warp.
            __syncthreads();
        }
    }

    /**
     * @brief Wait for one CPU publication without occupying a payload-sized grid.
     *
     * Only one thread waits. The following exact-stream materialization node
     * cannot begin until its acquire succeeds, leaving unrelated maintenance
     * work runnable while the CPU is still producing the expert contribution.
     * The producer retains the immutable ticket until the final acknowledgement.
     */
    static __global__ void awaitCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        do
        {
            const auto consumed = loadSystemAcquire64(&launch.control->consumed_sequence);
            const auto published = loadSystemAcquire64(&launch.control->published_sequence);
            if (published > consumed && published - consumed == 1u)
                return;
#if defined(__CUDA_ARCH__)
            __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1u);
#endif
        } while (true);
    }

    /**
     * @brief Materialize one acquired colocated CPU canonical-route ticket.
     *
     * The CPU has already authenticated the sparse packet, computed one raw
     * expert result per compact route, multiplied each row by its router
     * weight, and release-published a new sequence. The preceding one-thread
     * acquisition node has observed that sequence before any payload block runs.
     * The kernel performs only a route-slot permutation
     * into continuation VRAM; the following canonical reducer remains the sole
     * floating-point fold authority.
     */
    static __global__ void materializeCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint32_t block_ticket_valid;
        __shared__ std::uint32_t block_fixed_point;
        const auto control = snapshotPeerPublishedBlock(launch.control);
        if (threadIdx.x == 0u)
        {
            const auto publication_status =
                static_cast<MoEOverlayCanonicalRouteTicketStatus>(
                    control.publication_status);
            const bool success_payload =
                moeOverlayCanonicalRouteTicketStatusSucceeded(publication_status);
            const bool abort_payload =
                publication_status ==
                MoEOverlayCanonicalRouteTicketStatus::Aborted;
            block_fixed_point =
                publication_status ==
                        MoEOverlayCanonicalRouteTicketStatus::SuccessFixedPoint32Adaptive16
                    ? 1u
                    : 0u;
            block_ticket_valid =
                success_payload && control.valid() &&
                        control.publicationPending() &&
                        control.residency_epoch != 0u &&
                        control.route_capacity ==
                            static_cast<std::int32_t>(
                                launch.route_capacity) &&
                        control.d_model == launch.d_model &&
                        control.live_entry_count <= launch.route_capacity
                    ? 1u
                    : 0u;
            /* An authenticated abort deliberately materializes zero rows. The
             * following acknowledgement kernel still advances the sequence,
             * allowing every later wait in a failed retained parent to drain. */
            block_live_entries =
                abort_payload && control.valid() &&
                        control.publicationPending() &&
                        control.live_entry_count == 0u
                    ? 0u
                    : control.live_entry_count;
        }
        __syncthreads();
        if (block_ticket_valid == 0u)
            return;

        __shared__ std::int32_t original_slot;
        __shared__ std::int32_t compact_slot;
        // One block owns a complete route row. The mapped indices are read
        // once per row, rather than twice per output element across PCIe.
        // No arithmetic changes: every payload element still has one writer.
        for (std::size_t compact_entry = blockIdx.x;
             compact_entry < block_live_entries;
             compact_entry += gridDim.x)
        {
            if (threadIdx.x == 0u)
            {
                original_slot = loadPeerPublished(launch.original_route_slots + compact_entry);
                compact_slot = loadPeerPublished(launch.compact_route_slots + compact_entry);
            }
            __syncthreads();
            if (block_fixed_point != 0u)
            {
                if (original_slot >= 0 &&
                    static_cast<std::size_t>(original_slot) < launch.route_capacity)
                {
                    const std::size_t destination =
                        static_cast<std::size_t>(original_slot) * launch.d_model;
                    if (compact_slot == kMoEOverlayCanonicalTicketZeroSlot)
                    {
                        // Duplicate route for an already-folded token: zero locally with zero PCIe reads.
                        zeroLocalFP32Row(
                            launch.canonical_route_contributions_fp32 + destination,
                            launch.d_model);
                    }
                    else if (compact_slot >= 0)
                    {
                        const bool is_wide =
                            (compact_slot & kMoEOverlayCanonicalTicketCompactWideFlag) != 0;
                        const std::size_t src_row = static_cast<std::size_t>(
                            compact_slot & kMoEOverlayCanonicalTicketCompactSlotMask);
                        if (src_row < launch.route_capacity)
                        {
                            const std::size_t source = src_row * launch.d_model;
                            if (launch.output_fixed_point_int32_bits != 0u)
                            {
                                if (!is_wide)
                                {
                                    copyPeerPublishedFixed16ToInt32Bits(
                                        reinterpret_cast<const std::int16_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        launch.d_model);
                                }
                                else
                                {
                                    copyPeerPublishedRow(
                                        launch.compact_preweighted_contributions_fp32 + source,
                                        launch.canonical_route_contributions_fp32 + destination,
                                        launch.d_model);
                                }
                            }
                            else
                            {
                                if (!is_wide)
                                {
                                    copyPeerPublishedFixed16ToFP32(
                                        reinterpret_cast<const std::int16_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        launch.d_model);
                                }
                                else
                                {
                                    copyPeerPublishedFixed32ToFP32(
                                        reinterpret_cast<const std::int32_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        launch.d_model);
                                }
                            }
                        }
                    }
                }
            }
            else if (original_slot >= 0 && compact_slot >= 0 &&
                static_cast<std::size_t>(original_slot) < launch.route_capacity &&
                static_cast<std::size_t>(compact_slot) < launch.route_capacity)
            {
                const std::size_t source = static_cast<std::size_t>(compact_slot) * launch.d_model;
                const std::size_t destination = static_cast<std::size_t>(original_slot) * launch.d_model;
#if defined(__HIP_DEVICE_COMPILE__)
                // This acquired ticket keeps both source columns immutable until
                // the unchanged whole-grid acknowledgement. Issue two independent
                // SCALAR fresh reads before their stores: the original gfx906
                // loop waited/stored every scalar before issuing the next read.
                // Keep this scheduling local; the shared helper's other packet
                // consumers retain their qualified implementation/cache policy.
                for (std::size_t column = threadIdx.x;
                     column < static_cast<std::size_t>(launch.d_model);
                     column += 2u * blockDim.x)
                {
                    const float first = loadPeerPublished(
                        launch.compact_preweighted_contributions_fp32 + source + column);
                    const std::size_t next = column + blockDim.x;
                    float second{};
                    if (next < static_cast<std::size_t>(launch.d_model))
                        second = loadPeerPublished(
                            launch.compact_preweighted_contributions_fp32 + source + next);
                    launch.canonical_route_contributions_fp32[destination + column] = first;
                    if (next < static_cast<std::size_t>(launch.d_model))
                        launch.canonical_route_contributions_fp32[destination + next] = second;
                }
#else
                copyPeerPublishedRow(
                    launch.compact_preweighted_contributions_fp32 + source,
                    launch.canonical_route_contributions_fp32 + destination,
                    launch.d_model);
#endif
            }
            // Do not overwrite shared indices while another warp is still
            // using them; this also covers rejected rows and odd hidden widths.
            __syncthreads();
        }
    }

    /**
     * @brief Single-block fused materialization and release-acknowledgement for
     *        one-row canonical CPU route tickets (`route_capacity <= 32`).
     *
     * Snapshots the 64-byte mapped control record once across the block,
     * coalesces all live slot indices into shared memory, copies every live
     * route row across all threads, and after a block barrier publishes the
     * system-release acknowledgement without a second kernel launch or serial
     * control re-read.
     */
    static __global__ void materializeAndAcknowledgeSingleRowCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint64_t block_published_sequence;
        __shared__ std::uint32_t block_ticket_valid;
        __shared__ std::uint32_t block_ack_valid;
        __shared__ std::uint32_t block_fixed_point;
        __shared__ std::int32_t shared_original_slots[32];
        __shared__ std::int32_t shared_compact_slots[32];

        const auto control = snapshotPeerPublishedBlock(launch.control);
        if (threadIdx.x == 0u)
        {
            const auto publication_status =
                static_cast<MoEOverlayCanonicalRouteTicketStatus>(
                    control.publication_status);
            const bool success_payload =
                moeOverlayCanonicalRouteTicketStatusSucceeded(publication_status);
            const bool abort_payload =
                publication_status ==
                MoEOverlayCanonicalRouteTicketStatus::Aborted;
            block_fixed_point =
                publication_status ==
                        MoEOverlayCanonicalRouteTicketStatus::SuccessFixedPoint32Adaptive16
                    ? 1u
                    : 0u;
            const bool common_ok =
                control.valid() && control.publicationPending() &&
                control.route_capacity ==
                    static_cast<std::int32_t>(launch.route_capacity) &&
                control.d_model == launch.d_model;
            const bool valid_success =
                common_ok && success_payload &&
                control.residency_epoch != 0u &&
                control.live_entry_count <= launch.route_capacity;
            const bool valid_abort =
                common_ok && abort_payload &&
                control.residency_epoch == 0u &&
                control.live_entry_count == 0u;
            block_ticket_valid = valid_success ? 1u : 0u;
            block_ack_valid = (valid_success || valid_abort) ? 1u : 0u;
            block_live_entries = valid_success ? control.live_entry_count : 0u;
            block_published_sequence = control.published_sequence;
        }
        __syncthreads();

        if (block_ticket_valid != 0u && block_live_entries > 0u)
        {
            if (threadIdx.x < block_live_entries)
            {
                shared_original_slots[threadIdx.x] =
                    loadPeerPublished(launch.original_route_slots + threadIdx.x);
                shared_compact_slots[threadIdx.x] =
                    loadPeerPublished(launch.compact_route_slots + threadIdx.x);
            }
            __syncthreads();

            const std::size_t width = static_cast<std::size_t>(launch.d_model);
            if (block_fixed_point != 0u)
            {
                for (std::size_t entry = 0u; entry < block_live_entries; ++entry)
                {
                    const std::int32_t original_slot = shared_original_slots[entry];
                    const std::int32_t compact_slot = shared_compact_slots[entry];
                    if (original_slot < 0 ||
                        static_cast<std::size_t>(original_slot) >= launch.route_capacity)
                        continue;
                    const std::size_t destination =
                        static_cast<std::size_t>(original_slot) * width;
                    if (compact_slot == kMoEOverlayCanonicalTicketZeroSlot)
                    {
                        zeroLocalFP32Row(
                            launch.canonical_route_contributions_fp32 + destination, width);
                    }
                    else if (compact_slot >= 0)
                    {
                        const bool is_wide =
                            (compact_slot & kMoEOverlayCanonicalTicketCompactWideFlag) != 0;
                        const std::size_t src_row = static_cast<std::size_t>(
                            compact_slot & kMoEOverlayCanonicalTicketCompactSlotMask);
                        if (src_row < launch.route_capacity)
                        {
                            const std::size_t source = src_row * width;
                            if (launch.output_fixed_point_int32_bits != 0u)
                            {
                                if (!is_wide)
                                    copyPeerPublishedFixed16ToInt32Bits(
                                        reinterpret_cast<const std::int16_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        width);
                                else
                                    copyPeerPublishedRow(
                                        launch.compact_preweighted_contributions_fp32 + source,
                                        launch.canonical_route_contributions_fp32 + destination,
                                        width);
                            }
                            else
                            {
                                if (!is_wide)
                                    copyPeerPublishedFixed16ToFP32(
                                        reinterpret_cast<const std::int16_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        width);
                                else
                                    copyPeerPublishedFixed32ToFP32(
                                        reinterpret_cast<const std::int32_t *>(
                                            launch.compact_preweighted_contributions_fp32 + source),
                                        launch.canonical_route_contributions_fp32 + destination,
                                        width);
                            }
                        }
                    }
                }
            }
            else
            {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            const bool vec_aligned =
                (width % 4u == 0u) &&
                (((reinterpret_cast<std::uintptr_t>(launch.compact_preweighted_contributions_fp32) |
                   reinterpret_cast<std::uintptr_t>(launch.canonical_route_contributions_fp32)) &
                  (sizeof(float4) - 1u)) == 0u);
            if (vec_aligned)
            {
                // Flatten all live entries into one 4-way pipelined float4 loop so
                // independent PCIe reads across multiple route rows overlap in flight.
                const std::size_t vecs_per_row = width / 4u;
                const std::size_t total_vecs = block_live_entries * vecs_per_row;
                const auto *src_v = reinterpret_cast<const float4 *>(
                    launch.compact_preweighted_contributions_fp32);
                auto *dst_v = reinterpret_cast<float4 *>(
                    launch.canonical_route_contributions_fp32);
                const std::size_t stride = blockDim.x;
                for (std::size_t idx0 = threadIdx.x; idx0 < total_vecs; idx0 += 4u * stride)
                {
                    const std::size_t idx1 = idx0 + stride;
                    const std::size_t idx2 = idx0 + 2u * stride;
                    const std::size_t idx3 = idx0 + 3u * stride;
                    const auto resolve = [&](std::size_t idx, std::size_t &s, std::size_t &d) -> bool {
                        if (idx >= total_vecs)
                            return false;
                        const std::size_t e = idx / vecs_per_row;
                        const std::size_t v = idx - e * vecs_per_row;
                        const std::int32_t orig = shared_original_slots[e];
                        const std::int32_t comp = shared_compact_slots[e];
                        if (orig < 0 || comp < 0 ||
                            static_cast<std::size_t>(orig) >= launch.route_capacity ||
                            static_cast<std::size_t>(comp) >= launch.route_capacity)
                            return false;
                        s = static_cast<std::size_t>(comp) * vecs_per_row + v;
                        d = static_cast<std::size_t>(orig) * vecs_per_row + v;
                        return true;
                    };
                    std::size_t s0 = 0, d0 = 0, s1 = 0, d1 = 0, s2 = 0, d2 = 0, s3 = 0, d3 = 0;
                    const bool ok0 = resolve(idx0, s0, d0);
                    const bool ok1 = resolve(idx1, s1, d1);
                    const bool ok2 = resolve(idx2, s2, d2);
                    const bool ok3 = resolve(idx3, s3, d3);
                    float4 v0{}, v1{}, v2{}, v3{};
                    if (ok0) v0 = loadPeerPublishedFloat4(src_v + s0);
                    if (ok1) v1 = loadPeerPublishedFloat4(src_v + s1);
                    if (ok2) v2 = loadPeerPublishedFloat4(src_v + s2);
                    if (ok3) v3 = loadPeerPublishedFloat4(src_v + s3);
                    if (ok0) dst_v[d0] = v0;
                    if (ok1) dst_v[d1] = v1;
                    if (ok2) dst_v[d2] = v2;
                    if (ok3) dst_v[d3] = v3;
                }
            }
            else
#endif
            for (std::size_t entry = 0u; entry < block_live_entries; ++entry)
            {
                const std::int32_t original_slot = shared_original_slots[entry];
                const std::int32_t compact_slot = shared_compact_slots[entry];
                if (original_slot >= 0 && compact_slot >= 0 &&
                    static_cast<std::size_t>(original_slot) < launch.route_capacity &&
                    static_cast<std::size_t>(compact_slot) < launch.route_capacity)
                {
                    const std::size_t source =
                        static_cast<std::size_t>(compact_slot) * width;
                    const std::size_t destination =
                        static_cast<std::size_t>(original_slot) * width;
#if defined(__HIP_DEVICE_COMPILE__)
                    for (std::size_t column = threadIdx.x;
                         column < width;
                         column += 2u * blockDim.x)
                    {
                        const float first = loadPeerPublished(
                            launch.compact_preweighted_contributions_fp32 + source + column);
                        const std::size_t next = column + blockDim.x;
                        float second{};
                        if (next < width)
                            second = loadPeerPublished(
                                launch.compact_preweighted_contributions_fp32 + source + next);
                        launch.canonical_route_contributions_fp32[destination + column] = first;
                        if (next < width)
                            launch.canonical_route_contributions_fp32[destination + next] = second;
                    }
#else
                    copyPeerPublishedRow(
                        launch.compact_preweighted_contributions_fp32 + source,
                        launch.canonical_route_contributions_fp32 + destination,
                        width);
#endif
                }
            }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u && block_ack_valid != 0u)
        {
            storeSystemRelease64(
                MoEOverlayActivationTimelinePublishDeviceBinding{
                    .signal = &launch.control->consumed_sequence,
                    .value = block_published_sequence,
                });
        }
    }

    /**
     * @brief Release-acknowledge a completely materialized CPU route payload.
     *
     * Backends enqueue this warp-coalesced node immediately after every
     * multi-row or metadata-only @ref materializeCanonicalRouteTicketKernel
     * launch on the same exact stream. Using a block snapshot turns eight
     * serial uncached PCIe reads into one coalesced cache-line fetch.
     */
    static __global__ void acknowledgeCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        const auto control = snapshotPeerPublishedBlock(launch.control);
        if (threadIdx.x != 0u)
            return;
        const auto publication_status =
            static_cast<MoEOverlayCanonicalRouteTicketStatus>(
                control.publication_status);
        const bool success_payload =
            moeOverlayCanonicalRouteTicketStatusSucceeded(publication_status);
        const bool abort_payload =
            publication_status ==
            MoEOverlayCanonicalRouteTicketStatus::Aborted;
        if (!control.valid() || !control.publicationPending() ||
            control.route_capacity !=
                static_cast<std::int32_t>(launch.route_capacity) ||
            control.d_model != launch.d_model ||
            (launch.metadataOnly() ? control.live_entry_count != 0u :
                control.live_entry_count > launch.route_capacity) ||
            (success_payload && control.residency_epoch == 0u) ||
            (abort_payload &&
             (control.residency_epoch != 0u ||
              control.live_entry_count != 0u)) ||
            (!success_payload && !abort_payload))
        {
            return;
        }
        storeSystemRelease64(
            MoEOverlayActivationTimelinePublishDeviceBinding{
                .signal = &launch.control->consumed_sequence,
                .value = control.published_sequence,
            });
    }

    /**
     * @brief Authenticate independent prefill lanes using the common protocol.
     *
     * One block owns one lane. Its following materializer is stream ordered
     * after every block has either published success or a terminal failure.
     */
    static __global__ void validateMultiRowReturnBatchKernel(
        MoEOverlayActivationMultiRowReturnBatchLaunch launch)
    {
        __shared__ ValidateReturnBlockShared shared;
        const std::uint32_t lane = blockIdx.x;
        if (lane >= launch.lane_count)
            return;
        const auto packet = launch.lanes[lane];
        validateReturnBlock(packet, shared);
        if (threadIdx.x == 0u)
            launch.lane_valid[lane] = returnGrantPublished(packet) ? 1 : 0;
    }

    /**
     * @brief Copy authenticated lane contributions into original route slots.
     *
     * Grid Y owns a participant lane and grid X owns complete route rows. Each
     * block reads a mapped route index once, not once per FP32 element. Dynamic
     * placement guarantees one authoritative participant per original slot,
     * so stores are disjoint and require no
     * atomics or cross-lane ordering.
     */
    static __global__ void materializeMultiRowCanonicalReturnBatchKernel(
        MoEOverlayActivationMultiRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.y;
        if (lane >= launch.lane_count || launch.lane_valid[lane] == 0)
            return;
        const auto packet = launch.lanes[lane];
        __shared__ std::int32_t original_slot;
        __shared__ std::int32_t s_begin;
        __shared__ std::int32_t s_end;
        __shared__ std::int32_t s_row_wide;
        __shared__ std::int32_t s_orig_slots[32];
        if (usePrefillFixedPointWire(packet.prefill_wire_policy, packet.grant->graph_role))
        {
            const std::size_t width = static_cast<std::size_t>(launch.d_model);
            const std::uint64_t live_rows = packet.grant->live_rows;
            const std::uint64_t live_entries = packet.grant->live_entries;
            for (std::size_t row = blockIdx.x; row < live_rows; row += gridDim.x)
            {
                if (threadIdx.x == 0u)
                {
                    s_begin = packet.dispatch.entry_offsets[row];
                    s_end = packet.dispatch.entry_offsets[row + 1u];
                    s_row_wide = packet.returned.row_wide_flags
                        ? loadPeerPublished(packet.returned.row_wide_flags + row)
                        : 1;
                }
                __syncthreads();
                const std::int32_t count = s_end - s_begin;
                if (count > 0 && s_begin >= 0 &&
                    static_cast<std::uint64_t>(s_end) <= live_entries)
                {
                    if (count <= 32)
                    {
                        if (threadIdx.x < static_cast<std::uint32_t>(count))
                            s_orig_slots[threadIdx.x] = packet.dispatch.original_route_slots[s_begin + threadIdx.x];
                        __syncthreads();
                        const std::int32_t orig0 = s_orig_slots[0];
                        if (orig0 >= 0 &&
                            static_cast<std::size_t>(orig0) < packet.returned.route_slot_capacity)
                        {
                            const std::size_t dest = static_cast<std::size_t>(orig0) * width;
                            const float *slot_base =
                                packet.returned.canonical_route_contributions_fp32 + dest;
                            if (s_row_wide == 0)
                            {
                                copyPeerPublishedFixed16ToInt32Bits(
                                    reinterpret_cast<const std::int16_t *>(slot_base),
                                    launch.canonical_route_contributions_fp32 + dest,
                                    width);
                            }
                            else
                            {
                                copyPeerPublishedRow(
                                    slot_base,
                                    launch.canonical_route_contributions_fp32 + dest,
                                    width);
                            }
                        }
                        for (std::int32_t k = 1; k < count; ++k)
                        {
                            const std::int32_t orig_k = s_orig_slots[k];
                            if (orig_k >= 0 &&
                                static_cast<std::size_t>(orig_k) < packet.returned.route_slot_capacity)
                            {
                                zeroLocalFP32Row(
                                    launch.canonical_route_contributions_fp32 +
                                        static_cast<std::size_t>(orig_k) * width,
                                    width);
                            }
                        }
                    }
                    else
                    {
                        if (threadIdx.x == 0u)
                            s_orig_slots[0] = packet.dispatch.original_route_slots[s_begin];
                        __syncthreads();
                        const std::int32_t orig0 = s_orig_slots[0];
                        if (orig0 >= 0 &&
                            static_cast<std::size_t>(orig0) < packet.returned.route_slot_capacity)
                        {
                            const std::size_t dest = static_cast<std::size_t>(orig0) * width;
                            const float *slot_base =
                                packet.returned.canonical_route_contributions_fp32 + dest;
                            if (s_row_wide == 0)
                            {
                                copyPeerPublishedFixed16ToInt32Bits(
                                    reinterpret_cast<const std::int16_t *>(slot_base),
                                    launch.canonical_route_contributions_fp32 + dest,
                                    width);
                            }
                            else
                            {
                                copyPeerPublishedRow(
                                    slot_base,
                                    launch.canonical_route_contributions_fp32 + dest,
                                    width);
                            }
                        }
                        for (std::int32_t k = 1; k < count; ++k)
                        {
                            __syncthreads();
                            if (threadIdx.x == 0u)
                                s_orig_slots[0] = packet.dispatch.original_route_slots[s_begin + k];
                            __syncthreads();
                            const std::int32_t orig_k = s_orig_slots[0];
                            if (orig_k >= 0 &&
                                static_cast<std::size_t>(orig_k) < packet.returned.route_slot_capacity)
                            {
                                zeroLocalFP32Row(
                                    launch.canonical_route_contributions_fp32 +
                                        static_cast<std::size_t>(orig_k) * width,
                                    width);
                            }
                        }
                    }
                }
                __syncthreads();
            }
            return;
        }
        for (std::size_t compact_entry = blockIdx.x;
             compact_entry < packet.grant->live_entries;
             compact_entry += gridDim.x)
        {
            // This continuation published the immutable route map before
            // dispatch. Unlike the peer-produced return payload, its own map
            // may reuse the local cache after the stream-ordered validator.
            if (threadIdx.x == 0u)
                original_slot = packet.dispatch.original_route_slots[compact_entry];
            __syncthreads();
            if (original_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(launch.d_model);
                copyPeerPublishedRow(
                    packet.returned.canonical_route_contributions_fp32 + destination,
                    launch.canonical_route_contributions_fp32 + destination,
                    launch.d_model);
            }
            __syncthreads();
        }
    }

    /**
     * @brief Compact, copy, and release one direct-mapped dispatch in one node.
     *
     * One block is sufficient because physical row count is exactly one. Thread
     * zero retains the scalar protocol arithmetic; all threads stripe the hidden
     * width, meet at a block barrier, and only then publish the system timeline.
     * The release is unconditional so a semantic packet failure still drains the
     * peer graph instead of leaving it resident in a permanent wait.
     */
    /** @brief Kernel-local shared scratch for `packSingleRowDispatchBody` (avoids CUDA `-rdc=true` global `.shared` merge). */
    struct SingleRowDispatchShared
    {
        std::uint64_t live_rows;
        std::uint64_t live_entries;
        std::int32_t source_row;
        std::uint32_t packet_active;
        std::uint32_t staged_meta;
        std::int32_t s_expert_ids[32];
        float s_route_weights[32];
        std::int32_t s_orig_slots[32];
        std::int32_t s_compact_slots[32];
    };

    static __device__ __forceinline__ void packSingleRowDispatchBody(
        const MoEOverlayActivationSingleRowDispatchPackLaunch &launch,
        SingleRowDispatchShared &shared)
    {
        auto &packet = launch.packet;
        auto &live_rows = shared.live_rows;
        auto &live_entries = shared.live_entries;
        auto &source_row = shared.source_row;
        auto &packet_active = shared.packet_active;
        auto &staged_meta = shared.staged_meta;
        auto *s_expert_ids = shared.s_expert_ids;
        auto *s_route_weights = shared.s_route_weights;
        auto *s_orig_slots = shared.s_orig_slots;
        auto *s_compact_slots = shared.s_compact_slots;
        if (threadIdx.x == 0u)
        {
            if (launch.admission_wait.signal != nullptr)
                waitSystemAcquire64(launch.admission_wait);
            staged_meta = 0u;
            live_entries = 0u;
            if (packet.physical_rows == 1 && packet.packet.top_k > 0 && packet.packet.top_k <= 32)
            {
                const auto preparation = prepareDispatchMetadata(packet);
                if (preparation.valid())
                {
                    std::uint64_t compact_rows = 0u;
                    std::uint64_t compact_entries = 0u;
                    std::int32_t first_row_id = -1;
                    bool ok = true;
                    if (preparation.active_rows == 1)
                    {
                        for (std::int32_t slot = 0; slot < packet.packet.top_k; ++slot)
                        {
                            std::int32_t expert = -1;
                            std::int32_t route_participant = -1;
                            float weight = 0.0f;
                            if (!decodeDispatchRoute(
                                    packet, preparation.route_participants,
                                    static_cast<std::size_t>(slot),
                                    &expert, &weight, &route_participant))
                            {
                                publishFailure(
                                    packet.control, packet.grant,
                                    MoEOverlayActivationEndpoint::Continuation,
                                    MoEOverlayActivationOperation::PublishDispatch,
                                    MoEOverlayActivationStatusCode::PayloadMismatch,
                                    packet.stage_ordinal, packet.model_layer_index, 0u);
                                ok = false;
                                break;
                            }
                            if (route_participant != packet.target_participant_id)
                                continue;
                            if (compact_entries >= packet.packet.entry_capacity)
                            {
                                publishFailure(
                                    packet.control, packet.grant,
                                    MoEOverlayActivationEndpoint::Continuation,
                                    MoEOverlayActivationOperation::PublishDispatch,
                                    MoEOverlayActivationStatusCode::PayloadMismatch,
                                    packet.stage_ordinal, packet.model_layer_index, 0u);
                                ok = false;
                                break;
                            }
                            s_expert_ids[compact_entries] = expert;
                            s_route_weights[compact_entries] = weight;
                            s_orig_slots[compact_entries] = slot;
                            s_compact_slots[compact_entries] = static_cast<std::int32_t>(compact_entries);
                            ++compact_entries;
                        }
                        if (ok && compact_entries > 0u)
                        {
                            if (packet.packet.row_capacity == 0u)
                            {
                                publishFailure(
                                    packet.control, packet.grant,
                                    MoEOverlayActivationEndpoint::Continuation,
                                    MoEOverlayActivationOperation::PublishDispatch,
                                    MoEOverlayActivationStatusCode::PayloadMismatch,
                                    packet.stage_ordinal, packet.model_layer_index, 0u);
                                ok = false;
                            }
                            else
                            {
                                first_row_id = 0;
                                compact_rows = 1u;
                            }
                        }
                    }
                    if (ok)
                    {
                        finishDispatchMetadata(packet, compact_rows, compact_entries, first_row_id);
                        live_entries = compact_entries;
                        staged_meta = 1u;
                    }
                }
            }
            else
            {
                packDispatchMetadata(packet);
            }
            live_rows = packet.grant->live_rows;
            source_row = packet.grant->single_row_id;
            packet_active =
                packet.grant->state ==
                    raw(MoEOverlayActivationEndpointState::Active) &&
                packet.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success)
                    ? 1u
                    : 0u;
        }
        __syncthreads();

        if (packet_active != 0u && staged_meta != 0u)
        {
            if (threadIdx.x == 0u)
            {
                packet.packet.entry_offsets[0] = 0;
                if (live_rows == 1u)
                {
                    packet.packet.row_ids[0] = 0;
                    packet.packet.entry_offsets[1] = static_cast<std::int32_t>(live_entries);
                }
            }
            if (threadIdx.x < live_entries)
            {
                const std::uint32_t idx = threadIdx.x;
                packet.packet.expert_ids[idx] = s_expert_ids[idx];
                packet.packet.route_weights[idx] = s_route_weights[idx];
                packet.packet.original_route_slots[idx] = s_orig_slots[idx];
                packet.packet.compact_route_slots[idx] = s_compact_slots[idx];
            }
        }

        if (packet_active != 0u && live_rows == 1u && source_row >= 0)
        {
            const std::size_t width = static_cast<std::size_t>(packet.packet.d_model);
            const float *src = packet.hidden_rows_fp32 +
                static_cast<std::size_t>(source_row) * width;
            float *dst = packet.packet.hidden_rows_fp32;
            std::size_t copied = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if (((reinterpret_cast<std::uintptr_t>(src) |
                  reinterpret_cast<std::uintptr_t>(dst)) & (sizeof(float4) - 1u)) == 0u)
            {
                const std::size_t vecs = width / 4u;
                for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                    reinterpret_cast<float4 *>(dst)[v] =
                        reinterpret_cast<const float4 *>(src)[v];
                copied = vecs * 4u;
            }
#endif
            for (std::size_t column = copied + threadIdx.x;
                 column < width;
                 column += blockDim.x)
            {
                dst[column] = src[column];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            __threadfence_system();
            storeSystemRelaxed64(launch.publication);
        }
    }

    /** @brief Launch-compatible wrapper for one direct-mapped dispatch lane. */
    // The bound follows the shared arch-conditional constant above: 256 threads
    // on compute capability >= 12.0 (where the metadata chain needs the wider
    // register budget), 512 on every other CUDA target and all HIP devices.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1200)
#define LLAMINAR_SINGLE_ROW_DISPATCH_LAUNCH_BOUNDS __launch_bounds__(kSingleRowDispatchThreads, 2)
#else
#define LLAMINAR_SINGLE_ROW_DISPATCH_LAUNCH_BOUNDS __launch_bounds__(kSingleRowDispatchThreads)
#endif
    static __global__ LLAMINAR_SINGLE_ROW_DISPATCH_LAUNCH_BOUNDS void packSingleRowDispatchKernel(
        LLAMINAR_MOE_PACKET_GRID_CONSTANT MoEOverlayActivationSingleRowDispatchPackLaunch launch)
    {
        __shared__ SingleRowDispatchShared shared;
        if (blockIdx.x == 0u)
            packSingleRowDispatchBody(launch, shared);
    }

    /**
     * @brief Compact and publish every independent continuation lane in parallel.
     *
     * Lane descriptors are already ordered by the topology planner. Dispatch
     * has no cross-lane arithmetic, so assigning one block to each descriptor
     * removes serial launch and mapped-store latency without changing any lane's
     * packet bytes or timeline protocol.
     */
    static __global__ LLAMINAR_SINGLE_ROW_DISPATCH_LAUNCH_BOUNDS void packSingleRowDispatchBatchKernel(
        MoEOverlayActivationSingleRowDispatchBatchLaunch launch)
    {
        if (launch.shared_hidden_rows_fp32 != nullptr &&
            launch.lane_count > 1u && launch.lane_count <= 32u)
        {
            if (blockIdx.x != 0u)
                return;
            // Shared 1-block single-row dispatch: all lanes in this rank-pair
            // group share one 10 KB FP32 hidden vector written ONCE across PCIe
            // and released behind ONE system fence instead of N redundant copies
            // and N fences.
            __shared__ std::uint32_t s_any_active_row;
            if (threadIdx.x == 0u)
                s_any_active_row = 0u;
            __syncthreads();
            if (threadIdx.x < launch.lane_count)
            {
                const auto &lane_launch = launch.lanes[threadIdx.x];
                if (lane_launch.admission_wait.signal != nullptr)
                    waitSystemAcquire64(lane_launch.admission_wait);
                packDispatchMetadata(lane_launch.packet);
                if (lane_launch.packet.grant->state ==
                        raw(MoEOverlayActivationEndpointState::Active) &&
                    lane_launch.packet.grant->code ==
                        raw(MoEOverlayActivationStatusCode::Success) &&
                    lane_launch.packet.grant->live_rows == 1u &&
                    lane_launch.packet.grant->single_row_id >= 0)
                {
                    s_any_active_row = 1u;
                }
            }
            __syncthreads();
            if (s_any_active_row != 0u)
            {
                const auto &pkt0 = launch.lanes[0].packet;
                const std::size_t width = static_cast<std::size_t>(pkt0.packet.d_model);
                const float *__restrict__ src = pkt0.hidden_rows_fp32;
                float *__restrict__ dst = launch.shared_hidden_rows_fp32;
                std::size_t copied = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                if ((width & 3u) == 0u &&
                    (((reinterpret_cast<std::uintptr_t>(src) |
                       reinterpret_cast<std::uintptr_t>(dst)) &
                      (sizeof(float4) - 1u)) == 0u))
                {
                    const std::size_t vecs = width >> 2u;
                    const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(src);
                    auto *__restrict__ dst4 = reinterpret_cast<float4 *>(dst);
                    for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                        dst4[v] = src4[v];
                    copied = vecs << 2u;
                }
#endif
                for (std::size_t col = copied + threadIdx.x; col < width; col += blockDim.x)
                    dst[col] = src[col];
            }
            __syncthreads();
            // Issue ONE system fence for all metadata + shared-hidden writes,
            // then publish every lane's timeline signal in parallel with relaxed
            // system stores instead of N serialized membar.sys round trips.
            if (threadIdx.x == 0u)
                __threadfence_system();
            __syncthreads();
            if (threadIdx.x < launch.lane_count)
                storeSystemRelaxed64(launch.lanes[threadIdx.x].publication);
            return;
        }
        __shared__ SingleRowDispatchShared shared;
        const std::uint32_t lane = blockIdx.x;
        if (lane < launch.lane_count)
            packSingleRowDispatchBody(launch.lanes[lane], shared);
    }
#undef LLAMINAR_SINGLE_ROW_DISPATCH_LAUNCH_BOUNDS

    /**
     * @brief Acquire and materialize one direct-mapped dispatch in one node.
     *
     * Thread zero owns the potentially long peer wait and authenticated scalar
     * validation. Once it publishes the local live-row snapshot through shared
     * memory, the block copies hidden columns and initializes every route slot.
     * Thus downstream expert compute observes the same tensors as the ordinary
     * validator/materializer pair, with two graph nodes removed.
     */
    static __global__ __launch_bounds__(kSingleRowDispatchThreads) void consumeSingleRowDispatchKernel(
        MoEOverlayActivationSingleRowDispatchConsumeLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::int32_t live_rows;
        if (threadIdx.x == 0u)
        {
            waitSystemAcquire64(launch.acquire);
            validateDispatch(packet);
            live_rows =
                packet.grant->state ==
                        raw(MoEOverlayActivationEndpointState::Active) &&
                    packet.grant->code ==
                        raw(MoEOverlayActivationStatusCode::Success)
                    ? static_cast<std::int32_t>(packet.grant->live_rows)
                    : 0;
        }
        __syncthreads();

        const std::size_t d_model_cols =
            static_cast<std::size_t>(packet.packet.d_model);
        const float *src_hidden =
            launch.shared_hidden_rows_fp32 != nullptr
                ? launch.shared_hidden_rows_fp32
                : packet.packet.hidden_rows_fp32;
        if ((d_model_cols & 3u) == 0u &&
            (((reinterpret_cast<std::uintptr_t>(packet.hidden_rows_fp32) |
               reinterpret_cast<std::uintptr_t>(src_hidden)) &
              (sizeof(float4) - 1u)) == 0u))
        {
            const std::size_t vecs = d_model_cols >> 2u;
            auto *__restrict__ dst4 =
                reinterpret_cast<float4 *>(packet.hidden_rows_fp32);
            const auto *__restrict__ src4 =
                reinterpret_cast<const float4 *>(src_hidden);
            for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
            {
                dst4[v] = live_rows == 1
                              ? loadPeerPublishedFloat4(src4 + v)
                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            for (std::size_t column = threadIdx.x;
                 column < d_model_cols;
                 column += blockDim.x)
            {
                packet.hidden_rows_fp32[column] =
                    live_rows == 1
                        ? loadPeerPublished(src_hidden + column)
                        : 0.0f;
            }
        }
        for (std::int32_t slot = static_cast<std::int32_t>(threadIdx.x);
             slot < packet.packet.top_k;
             slot += static_cast<std::int32_t>(blockDim.x))
        {
            std::int32_t expert = -1;
            float weight = 0.0f;
            if (live_rows == 1)
            {
                const std::int32_t begin = loadPeerPublished(
                    packet.packet.entry_offsets);
                const std::int32_t end = loadPeerPublished(
                    packet.packet.entry_offsets + 1u);
                if (begin + slot < end)
                {
                    expert = loadPeerPublished(
                        packet.packet.expert_ids + begin + slot);
                    weight = loadPeerPublished(
                        packet.packet.route_weights + begin + slot);
                }
            }
            packet.routing_indices_fp32[slot] = static_cast<float>(expert);
            packet.routing_weights_fp32[slot] = weight;
        }
    }

    /**
     * @brief Pack and release one direct-mapped participant return in one node.
     *
     * Metadata is finalized first, the block stripes live output columns into
     * mapped pages, and a system fence plus release store exposes the complete
     * packet to the continuation endpoint. Empty routes still publish their
     * descriptor and timeline so graph topology is independent of routing skew.
     */
    static __global__ void packSingleRowReturnKernel(
        MoEOverlayActivationSingleRowReturnPackLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::uint64_t live_entries;
        __shared__ std::uint32_t packet_valid;
        __shared__ std::int32_t route_slots[2];
        if (threadIdx.x == 0u)
        {
            packReturnMetadata(packet);
            const auto state = static_cast<MoEOverlayActivationEndpointState>(
                packet.grant->state);
            packet_valid =
                (state == MoEOverlayActivationEndpointState::Active ||
                 state == MoEOverlayActivationEndpointState::Complete) &&
                    packet.grant->code ==
                        raw(MoEOverlayActivationStatusCode::Success) &&
                    packet.grant->last_published_stage ==
                        static_cast<std::int32_t>(packet.stage_ordinal)
                    ? 1u
                    : 0u;
            live_entries = packet_valid != 0u
                               ? packet.grant->live_entries
                               : 0u;
        }
        __syncthreads();

        const auto width = static_cast<std::size_t>(packet.returned.d_model);
        for (std::size_t entry = 0u; entry < live_entries; ++entry)
        {
            // The fused publication has one block, so finish each route before
            // reusing its two shared identities. Neither identity belongs to an
            // activation element: issuing these PCIe reads per scalar is waste.
            if (threadIdx.x < 2u)
                route_slots[threadIdx.x] = loadPeerPublished(
                    (threadIdx.x == 0u ? packet.dispatch.original_route_slots
                                      : packet.dispatch.compact_route_slots) + entry);
            __syncthreads();
            const auto original_slot = route_slots[0];
            const auto compact_slot = route_slots[1];
            if (original_slot >= 0 && compact_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                auto *destination = packet.returned.canonical_route_contributions_fp32 +
                    static_cast<std::size_t>(original_slot) * width;
                const auto *source = packet.local_canonical_route_contributions_fp32 +
                    static_cast<std::size_t>(compact_slot) * width;
                if ((width & 3u) == 0u &&
                    (((reinterpret_cast<std::uintptr_t>(destination) |
                       reinterpret_cast<std::uintptr_t>(source)) &
                      (sizeof(float4) - 1u)) == 0u))
                {
                    const std::size_t vecs = width >> 2u;
                    auto *__restrict__ dst4 = reinterpret_cast<float4 *>(destination);
                    const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(source);
                    for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                        dst4[v] = src4[v];
                }
                else
                {
                    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x)
                        destination[column] = source[column];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            __threadfence_system();
            storeSystemRelaxed64(launch.publication);
        }
    }

    /**
     * @brief Acquire, validate, and materialize one direct-mapped return.
     */
    static __global__ void consumeSingleRowReturnKernel(
        MoEOverlayActivationSingleRowReturnConsumeLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ ValidateReturnBlockShared validate_shared;
        __shared__ std::uint64_t live_entries;
        __shared__ std::int32_t original_slot;
        if (threadIdx.x == 0u)
            waitSystemAcquire64(launch.acquire);
        __syncthreads();
        validateReturnBlock(packet, validate_shared);
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            const auto state = static_cast<MoEOverlayActivationEndpointState>(
                packet.grant->state);
            const bool valid =
                (state == MoEOverlayActivationEndpointState::Active ||
                 state == MoEOverlayActivationEndpointState::Complete) &&
                packet.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success) &&
                packet.grant->last_consumed_stage ==
                    static_cast<std::int32_t>(packet.stage_ordinal);
            live_entries = valid ? packet.grant->live_entries : 0u;
        }
        __syncthreads();

        const auto width = static_cast<std::size_t>(packet.returned.d_model);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (live_entries > 0u && live_entries <= 32u && (width % 4u == 0u) &&
            (((reinterpret_cast<std::uintptr_t>(packet.returned.canonical_route_contributions_fp32) |
               reinterpret_cast<std::uintptr_t>(packet.canonical_route_contributions_fp32)) &
              (sizeof(float4) - 1u)) == 0u))
        {
            __shared__ std::int32_t orig_slots[32];
            if (threadIdx.x < live_entries)
                orig_slots[threadIdx.x] = loadPeerPublished(packet.dispatch.original_route_slots + threadIdx.x);
            __syncthreads();
            const std::size_t vecs_per_row = width / 4u;
            const std::size_t total_vecs = live_entries * vecs_per_row;
            const auto *src_v = reinterpret_cast<const float4 *>(packet.returned.canonical_route_contributions_fp32);
            auto *dst_v = reinterpret_cast<float4 *>(packet.canonical_route_contributions_fp32);
            const std::size_t stride = blockDim.x;
            for (std::size_t idx0 = threadIdx.x; idx0 < total_vecs; idx0 += 4u * stride)
            {
                const std::size_t idx1 = idx0 + stride;
                const std::size_t idx2 = idx0 + 2u * stride;
                const std::size_t idx3 = idx0 + 3u * stride;
                const auto resolve = [&](std::size_t idx, std::size_t &off) -> bool {
                    if (idx >= total_vecs) return false;
                    const std::size_t e = idx / vecs_per_row;
                    const std::size_t v = idx - e * vecs_per_row;
                    const std::int32_t orig = orig_slots[e];
                    if (orig < 0 || static_cast<std::size_t>(orig) >= packet.returned.route_slot_capacity)
                        return false;
                    off = static_cast<std::size_t>(orig) * vecs_per_row + v;
                    return true;
                };
                std::size_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
                const bool ok0 = resolve(idx0, o0);
                const bool ok1 = resolve(idx1, o1);
                const bool ok2 = resolve(idx2, o2);
                const bool ok3 = resolve(idx3, o3);
                float4 v0{}, v1{}, v2{}, v3{};
                if (ok0) v0 = loadPeerPublishedFloat4(src_v + o0);
                if (ok1) v1 = loadPeerPublishedFloat4(src_v + o1);
                if (ok2) v2 = loadPeerPublishedFloat4(src_v + o2);
                if (ok3) v3 = loadPeerPublishedFloat4(src_v + o3);
                if (ok0) dst_v[o0] = v0;
                if (ok1) dst_v[o1] = v1;
                if (ok2) dst_v[o2] = v2;
                if (ok3) dst_v[o3] = v3;
            }
            return;
        }
#endif
        for (std::size_t entry = 0u; entry < live_entries; ++entry)
        {
            if (threadIdx.x == 0u)
                original_slot = loadPeerPublished(packet.dispatch.original_route_slots + entry);
            __syncthreads();
            if (original_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) * width;
                copyPeerPublishedRow(
                    packet.returned.canonical_route_contributions_fp32 + destination,
                    packet.canonical_route_contributions_fp32 + destination, width);
            }
            // The wait and validation remain fused; all threads finish using
            // this identity before lane zero starts the next complete row.
            __syncthreads();
        }
    }

    /**
     * @brief Wait for, authenticate, and immediately materialize one-row returns concurrently.
     *
     * Each block owns one mapped lane, its private grant, and its disjoint
     * original route slots in `canonical_route_contributions_fp32`. Fusing
     * materialization into the wait/validate block lets faster followers begin
     * PCIe return copies immediately while slower followers are still computing,
     * and removes the second per-layer kernel launch completely.
     */
    static __global__ void validateSingleRowReturnBatchKernel(
        MoEOverlayActivationSingleRowReturnBatchLaunch launch)
    {
        __shared__ ValidateReturnBlockShared validate_shared;
        __shared__ std::uint64_t live_entries;
        __shared__ std::int32_t orig_slots[32];
        const std::uint32_t lane = blockIdx.x;
        if (lane >= launch.lane_count)
            return;

        const auto lane_launch = launch.lanes[lane];
        auto packet = lane_launch.packet;
        if (threadIdx.x == 0u)
            waitSystemAcquire64(lane_launch.acquire);
        __syncthreads();
        validateReturnBlock(packet, validate_shared);
        if (threadIdx.x == 0u)
        {
            const bool valid = returnGrantPublished(packet) &&
                packet.grant->live_rows <= 1u &&
                (packet.grant->live_rows == 0u ||
                 packet.grant->single_row_id == 0);
            launch.lane_valid[lane] = valid ? 1 : 0;
            live_entries = valid ? packet.grant->live_entries : 0u;
        }
        __syncthreads();

        const std::uint64_t entries = live_entries;
        if (entries == 0u)
            return;

        const std::size_t width = static_cast<std::size_t>(launch.d_model);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (entries <= 32u && (width % 4u == 0u) &&
            (((reinterpret_cast<std::uintptr_t>(packet.returned.canonical_route_contributions_fp32) |
               reinterpret_cast<std::uintptr_t>(launch.canonical_route_contributions_fp32)) &
              (sizeof(float4) - 1u)) == 0u))
        {
            if (threadIdx.x < entries)
                orig_slots[threadIdx.x] = packet.dispatch.original_route_slots[threadIdx.x];
            __syncthreads();
            const std::size_t vecs_per_row = width / 4u;
            const std::size_t total_vecs = static_cast<std::size_t>(entries) * vecs_per_row;
            const auto *src_v = reinterpret_cast<const float4 *>(packet.returned.canonical_route_contributions_fp32);
            auto *dst_v = reinterpret_cast<float4 *>(launch.canonical_route_contributions_fp32);
            const std::size_t stride = blockDim.x;
            for (std::size_t idx0 = threadIdx.x; idx0 < total_vecs; idx0 += 4u * stride)
            {
                const std::size_t idx1 = idx0 + stride;
                const std::size_t idx2 = idx0 + 2u * stride;
                const std::size_t idx3 = idx0 + 3u * stride;
                const auto resolve = [&](std::size_t idx, std::size_t &off) -> bool {
                    if (idx >= total_vecs) return false;
                    const std::size_t e = idx / vecs_per_row;
                    const std::size_t v = idx - e * vecs_per_row;
                    const std::int32_t orig = orig_slots[e];
                    if (orig < 0 || orig >= launch.top_k) return false;
                    off = static_cast<std::size_t>(orig) * vecs_per_row + v;
                    return true;
                };
                std::size_t o0 = 0, o1 = 0, o2 = 0, o3 = 0;
                const bool ok0 = resolve(idx0, o0);
                const bool ok1 = resolve(idx1, o1);
                const bool ok2 = resolve(idx2, o2);
                const bool ok3 = resolve(idx3, o3);
                float4 v0{}, v1{}, v2{}, v3{};
                if (ok0) v0 = loadPeerPublishedFloat4(src_v + o0);
                if (ok1) v1 = loadPeerPublishedFloat4(src_v + o1);
                if (ok2) v2 = loadPeerPublishedFloat4(src_v + o2);
                if (ok3) v3 = loadPeerPublishedFloat4(src_v + o3);
                if (ok0) dst_v[o0] = v0;
                if (ok1) dst_v[o1] = v1;
                if (ok2) dst_v[o2] = v2;
                if (ok3) dst_v[o3] = v3;
            }
            return;
        }
#endif
        for (std::size_t entry = 0u; entry < entries; ++entry)
        {
            if (threadIdx.x == 0u)
                orig_slots[0] = packet.dispatch.original_route_slots[entry];
            __syncthreads();
            const std::int32_t original_slot = orig_slots[0];
            if (original_slot >= 0 && original_slot < launch.top_k)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) * width;
                copyPeerPublishedRow(
                    packet.returned.canonical_route_contributions_fp32 + destination,
                    launch.canonical_route_contributions_fp32 + destination,
                    width);
            }
            __syncthreads();
        }
    }

    /**
     * @brief Materialize authenticated one-row lane contributions in parallel.
     *
     * Each block owns one original route row, just as the multi-row path does.
     * This avoids repeated peer metadata reads and keeps byte-identical copying
     * independent of both expert format and continuation backend.
     */
    static __global__ void materializeSingleRowCanonicalReturnBatchKernel(
        MoEOverlayActivationSingleRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.y;
        if (lane >= launch.lane_count || launch.lane_valid[lane] == 0)
            return;
        const auto packet = launch.lanes[lane].packet;
        __shared__ std::int32_t original_slot;
        for (std::size_t entry = blockIdx.x; entry < packet.grant->live_entries;
             entry += gridDim.x)
        {
            // The continuation owns this route map; the peer only writes its
            // separate return payload, which still requires fresh loads.
            if (threadIdx.x == 0u)
                original_slot = packet.dispatch.original_route_slots[entry];
            __syncthreads();
            if (original_slot >= 0 && original_slot < launch.top_k)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(launch.d_model);
                copyPeerPublishedRow(
                    packet.returned.canonical_route_contributions_fp32 + destination,
                    launch.canonical_route_contributions_fp32 + destination,
                    launch.d_model);
            }
            __syncthreads();
        }
    }
} // namespace llaminar2::moe_activation_packet_device

#undef LLAMINAR_MOE_PACKET_GRID_CONSTANT
