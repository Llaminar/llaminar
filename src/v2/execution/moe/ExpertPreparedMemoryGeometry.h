/**
 * @file ExpertPreparedMemoryGeometry.h
 * @brief Canonical CPU/GPU allocation geometry for one routed-expert projection.
 *
 * Capacity resolution and concrete slot materialization must agree byte for
 * byte. This header is their single pure arithmetic authority. It covers every
 * catalogued NativeVNNI source codebook plus FP16, BF16, and FP32, and delegates
 * final allocator rounding to the allocator-owned `AlignedVector` and GPU
 * weight-pool contracts instead of copying those policies into callers.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "loaders/GPUVramPreflight.h"
#include "tensors/AlignedVector.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace llaminar2
{
    /** @brief Exact shape/provenance identity of one reusable projection slot. */
    struct ExpertPreparedProjectionIdentity
    {
        int N = 0;
        int K = 0;
        ExpertWeightFormat format;

        /** @return Whether shape and exact arithmetic provenance are equal. */
        bool operator==(const ExpertPreparedProjectionIdentity &) const = default;

        /** @return Deterministic order used only to group compatible arenas. */
        bool operator<(
            const ExpertPreparedProjectionIdentity &other) const noexcept
        {
            return std::tie(
                       N,
                       K,
                       format.kind,
                       format.native_vnni.codebook_id,
                       format.native_vnni.is_superblock,
                       format.native_vnni.present) <
                   std::tie(
                       other.N,
                       other.K,
                       other.format.kind,
                       other.format.native_vnni.codebook_id,
                       other.format.native_vnni.is_superblock,
                       other.format.native_vnni.present);
        }
    };

    /**
     * @brief Complete gate/up/down identity for cross-layer slot recycling.
     *
     * Two layers may share physical arrival storage only when all three
     * projection identities compare equal. This key is the single grouping
     * contract consumed by memory admission and physical materialization.
     */
    struct ExpertPreparedTripletGeometryKey
    {
        std::array<ExpertPreparedProjectionIdentity, 3> projections;

        /** @return Whether all gate/up/down projection identities are equal. */
        bool operator==(const ExpertPreparedTripletGeometryKey &) const = default;

        /** @return Deterministic order used only to group compatible arenas. */
        bool operator<(
            const ExpertPreparedTripletGeometryKey &other) const noexcept
        {
            return projections < other.projections;
        }
    };

    /** Exact allocation extents for one prepared expert projection. */
    struct ExpertPreparedProjectionMemoryGeometry
    {
        /** CPU final execution allocation, page-owned for deterministic NUMA use. */
        std::size_t cpu_bytes = 0u;
        /** GPU live representation produced directly from the source codebook. */
        std::size_t gpu_live_bytes = 0u;
        /** GPU inactive arrival slot capable of accepting every source codebook. */
        std::size_t gpu_shadow_bytes = 0u;
    };

    namespace expert_prepared_memory_detail
    {
        /** @brief Multiply allocation geometry without permitting wraparound. */
        [[nodiscard]] inline std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string(description) + " byte product overflows size_t");
            }
            return left * right;
        }

        /** @brief Add allocation geometry without permitting wraparound. */
        [[nodiscard]] inline std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string(description) + " byte sum overflows size_t");
            }
            return left + right;
        }

        /** @brief Resolve final CPU storage for quantized or floating weights. */
        [[nodiscard]] inline std::size_t cpuProjectionBytes(
            int N,
            int K,
            const ExpertWeightFormat &format)
        {
            if (N <= 0 || K <= 0 || !format.valid())
            {
                throw std::invalid_argument(
                    "CPU expert projection memory requires positive geometry and a valid format");
            }

            constexpr std::size_t page_alignment = 4096u;
            if (format.isFloating())
            {
                const std::size_t elements = checkedMultiply(
                    static_cast<std::size_t>(N),
                    static_cast<std::size_t>(K),
                    "CPU floating expert elements");
                switch (format.kind)
                {
                case ExpertWeightFormatKind::FP16:
                case ExpertWeightFormatKind::BF16:
                    return AlignedVector<std::uint16_t>::requiredAllocationBytes(
                        elements, page_alignment);
                case ExpertWeightFormatKind::FP32:
                    return AlignedVector<float>::requiredAllocationBytes(
                        elements, page_alignment);
                default:
                    throw std::logic_error(
                        "Validated floating expert has no CPU storage precision");
                }
            }

            if (K % 32 != 0)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI expert projection K is not block aligned");
            }
            const auto *source = native_vnni_formats::forSourceIdentity(
                format.native_vnni.codebook_id,
                format.native_vnni.is_superblock);
            if (!source)
            {
                throw std::invalid_argument(
                    "CPU expert projection source codebook is not catalogued");
            }
            const auto encoding =
                cpu::native_vnni::preparedEncodingForCodebook(
                    source->codebook_id);
            const std::size_t stride =
                cpu::native_vnni::preparedInterleavedBlockStride(
                    encoding, source->is_asymmetric);
            const std::size_t padded_n =
                checkedAdd(static_cast<std::size_t>(N), 63u, "CPU padded N") &
                ~std::size_t{63u};
            const std::size_t units = checkedMultiply(
                padded_n / 64u,
                static_cast<std::size_t>(K / 32),
                "CPU NativeVNNI units");
            const std::size_t logical_bytes = checkedMultiply(
                units, stride, "CPU NativeVNNI projection");
            return AlignedVector<std::uint8_t>::requiredAllocationBytes(
                logical_bytes, page_alignment);
        }

        /**
         * @brief Resolve one separated GPU projection exactly as WeightVRAMPool.
         */
        [[nodiscard]] inline std::size_t gpuSeparatedProjectionBytes(
            int N,
            int K,
            int payload_bytes_per_block,
            bool is_asymmetric,
            bool has_emins)
        {
            if (N <= 0 || K <= 0 || K % 32 != 0 ||
                payload_bytes_per_block <= 0)
            {
                throw std::invalid_argument(
                    "GPU expert projection memory requires positive block-aligned geometry");
            }
            const std::size_t blocks = checkedMultiply(
                static_cast<std::size_t>(N),
                static_cast<std::size_t>(K / 32),
                "GPU expert blocks");
            std::size_t cursor = 0u;
            const auto append = [&](std::size_t bytes, const char *description)
            {
                if (bytes == 0u)
                    return;
                cursor = alignGPUWeightLoadAllocation(cursor);
                cursor = checkedAdd(cursor, bytes, description);
            };
            append(
                checkedMultiply(
                    blocks,
                    static_cast<std::size_t>(payload_bytes_per_block),
                    "GPU expert payload"),
                "GPU expert payload");
            append(
                checkedMultiply(blocks, sizeof(std::uint16_t), "GPU expert scales"),
                "GPU expert scales");
            if (is_asymmetric)
            {
                append(
                    checkedMultiply(blocks, sizeof(std::uint16_t), "GPU expert minima"),
                    "GPU expert minima");
            }
            if (has_emins)
            {
                append(
                    checkedMultiply(blocks, sizeof(std::uint32_t), "GPU expert effective minima"),
                    "GPU expert effective minima");
            }
            return alignGPUWeightLoadAllocation(cursor);
        }
    } // namespace expert_prepared_memory_detail

    /**
     * @brief Resolve all execution-tier allocation extents for one projection.
     * @param N Output-row count for the prepared GEMM weight.
     * @param K Reduction dimension for the prepared GEMM weight.
     * @param format Exact quantized codebook or floating precision provenance.
     * @return Byte-exact CPU live/shadow and GPU live/shadow geometry.
     */
    [[nodiscard]] inline ExpertPreparedProjectionMemoryGeometry
    resolveExpertPreparedProjectionMemoryGeometry(
        int N,
        int K,
        const ExpertWeightFormat &format)
    {
        using namespace expert_prepared_memory_detail;
        ExpertPreparedProjectionMemoryGeometry result;
        result.cpu_bytes = cpuProjectionBytes(N, K, format);

        if (format.isFloating())
        {
            const std::size_t elements = checkedMultiply(
                static_cast<std::size_t>(N),
                static_cast<std::size_t>(K),
                "GPU floating expert elements");
            const std::size_t logical_bytes = checkedMultiply(
                elements,
                format.floatingElementBytes(),
                "GPU floating expert projection");
            result.gpu_live_bytes =
                alignGPUWeightLoadAllocation(logical_bytes);
            result.gpu_shadow_bytes = result.gpu_live_bytes;
            return result;
        }

        const auto *source = native_vnni_formats::forSourceIdentity(
            format.native_vnni.codebook_id,
            format.native_vnni.is_superblock);
        if (!source)
        {
            throw std::invalid_argument(
                "GPU expert projection source codebook is not catalogued");
        }
        const auto live = reusableDeviceVnniAllocationFormat(*source);
        result.gpu_live_bytes = gpuSeparatedProjectionBytes(
            N,
            K,
            live.payload_bytes_per_block,
            live.has_mins,
            live.has_emins);
        result.gpu_shadow_bytes = gpuSeparatedProjectionBytes(
            N,
            K,
            /*payload_bytes_per_block=*/32,
            /*is_asymmetric=*/true,
            /*has_emins=*/true);
        return result;
    }
} // namespace llaminar2
