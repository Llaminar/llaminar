/**
 * @file ExpertTierWeightDeviceLayout.h
 * @brief Allocation-free device contract for streamed expert-weight repacking.
 *
 * The structures in this file deliberately contain only scalar geometry and
 * already-owned pointers. CUDA and HIP conversion launchers consume them
 * directly, allowing isolated correctness, timing, and compiler-resource
 * tests without coupling a kernel translation unit to model orchestration.
 */

#pragma once

#include "kernels/cpu/gemm/CPUNativeVNNIPreparedFootprint.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2
{
    /** Direction in which one streamed CPU-format chunk is converted. */
    enum class ExpertTierWeightConversionDirection : std::uint8_t
    {
        GpuToCpu = 1, ///< Source GPU decodes/reformats bytes before D2H transfer.
        CpuToGpu = 2, ///< Destination GPU reformats arriving CPU bytes after H2D.
    };

    /**
     * @brief Complete immutable geometry needed by a repack kernel launch.
     *
     * `unit_count` covers padded output columns because CPU NativeVNNI stores
     * complete 64-column units. GPU regions cover only the logical `N` rows.
     */
    struct ExpertTierWeightDeviceLayout
    {
        // Direction determines whether GPU metadata is a source or destination.
        ExpertTierWeightConversionDirection direction =
            ExpertTierWeightConversionDirection::CpuToGpu;
        // Logical and padded matrix geometry used to map a unit to (N, K block).
        std::int32_t N = 0;
        std::int32_t K = 0;
        std::int32_t N_padded = 0;
        std::int32_t blocks_per_row = 0;
        // Source provenance disambiguates shared execution codebooks.
        std::uint8_t cpu_codebook_id = 0;
        std::uint8_t gpu_source_codebook_id = 0;
        std::uint8_t gpu_source_is_superblock = 0;
        // Endpoint format metadata selects one compile-time backend kernel.
        std::uint8_t gpu_codebook_id = 0;
        std::uint8_t gpu_payload_bytes_per_block = 0;
        std::uint8_t cpu_is_asymmetric = 0;
        std::uint8_t cpu_is_superblock = 0;
        std::uint8_t gpu_is_asymmetric = 0;
        std::uint8_t gpu_has_emins = 0;
        cpu::native_vnni::CPUNativeVNNIEncoding cpu_encoding =
            cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;
        // CPU-unit and persistent staging capacity; no device allocation occurs.
        std::uint32_t cpu_data_stride = 0;
        std::uint32_t cpu_block_stride = 0;
        std::uint32_t unit_count = 0;
        std::uint32_t maximum_units_per_chunk = 0;

        /**
         * @brief Validate all scalar geometry and format relationships.
         * @return Whether the layout can safely drive a device kernel.
         */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            using Encoding = cpu::native_vnni::CPUNativeVNNIEncoding;
            const bool gpu_to_cpu =
                direction == ExpertTierWeightConversionDirection::GpuToCpu;
            const bool cpu_to_gpu =
                direction == ExpertTierWeightConversionDirection::CpuToGpu;
            // CPU kernels consume whole 32-K blocks and 64-column N units.
            if ((!gpu_to_cpu && !cpu_to_gpu) ||
                N <= 0 || K <= 0 || (K % 32) != 0 ||
                blocks_per_row != K / 32 || N_padded < N ||
                (N_padded % 64) != 0 || N_padded != ((N + 63) / 64) * 64 ||
                !cpu::native_vnni::isKnownPreparedEncoding(cpu_encoding))
            {
                return false;
            }
            // Redundant wire strides must agree with the canonical encoding.
            if (cpu_data_stride !=
                    cpu::native_vnni::preparedDataStride(cpu_encoding) ||
                cpu_block_stride !=
                    cpu::native_vnni::preparedInterleavedBlockStride(
                        cpu_encoding, cpu_is_asymmetric != 0))
            {
                return false;
            }
            // Padded N contributes real zero-filled CPU units to the stream.
            const std::uint64_t expected_units =
                static_cast<std::uint64_t>(N_padded / 64) *
                static_cast<std::uint64_t>(blocks_per_row);
            if (expected_units == 0 ||
                expected_units > std::numeric_limits<std::uint32_t>::max() ||
                unit_count != expected_units || maximum_units_per_chunk == 0 ||
                maximum_units_per_chunk > unit_count)
            {
                return false;
            }

            // Both directions retain source provenance even when CPU→GPU
            // promotion normalizes an expanded source into codebook 19 or 23.
            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    gpu_source_codebook_id,
                    gpu_source_is_superblock != 0);
            if (source_format == nullptr ||
                cpu_codebook_id != gpu_source_codebook_id ||
                cpu_is_superblock != gpu_source_is_superblock)
            {
                return false;
            }

            if (gpu_to_cpu)
            {
                // A demotion kernel must decode the actual accelerator source,
                // including Q2_K effective minima and dual-scale codebooks.
                // A previous CPU promotion may instead have normalized an
                // ExpandedInt8 source to physical codebook 19/23. That is a
                // first-class live representation, not a provenance change.
                const bool canonical_source =
                    gpu_codebook_id == canonicalDeviceVnniCodebookId(
                                           source_format->codebook_id) &&
                    gpu_payload_bytes_per_block ==
                        source_format->payload_bytes &&
                    gpu_is_asymmetric ==
                        static_cast<std::uint8_t>(
                            source_format->is_asymmetric) &&
                    gpu_has_emins ==
                        static_cast<std::uint8_t>(source_format->has_emins);
                const bool normalized_source =
                    cpu_encoding == Encoding::ExpandedInt8 &&
                    gpu_payload_bytes_per_block == 32 &&
                    gpu_has_emins == 0 &&
                    gpu_is_asymmetric == cpu_is_asymmetric &&
                    gpu_codebook_id ==
                        (cpu_is_asymmetric != 0
                             ? kNativeVnniExpandedInt8MinCodebook
                             : static_cast<std::uint8_t>(19));
                if ((!canonical_source && !normalized_source) ||
                    cpu_is_asymmetric !=
                        static_cast<std::uint8_t>(
                            source_format->is_asymmetric))
                {
                    return false;
                }

                const Encoding expected_encoding =
                    gpu_codebook_id == 0 || gpu_codebook_id == 4 ||
                            gpu_codebook_id == 5
                        ? Encoding::NibbleLUT
                        : (gpu_codebook_id == 8
                               ? Encoding::Q6KNativeDualScale
                               : Encoding::ExpandedInt8);
                return cpu_encoding == expected_encoding;
            }

            // Promotion consumes final CPU bytes. Expanded CPU encodings have
            // deliberately lost their compact source representation and must
            // therefore target the normalized signed-INT8 codebooks.
            if (gpu_has_emins != 0)
                return false;
            switch (cpu_encoding)
            {
            case Encoding::NibbleLUT:
                return gpu_payload_bytes_per_block == 16 &&
                       gpu_is_asymmetric == cpu_is_asymmetric &&
                       (((gpu_codebook_id == 0 || gpu_codebook_id == 4) &&
                         cpu_is_asymmetric == 0) ||
                        (gpu_codebook_id == 5 && cpu_is_asymmetric != 0));
            case Encoding::ExpandedInt8:
                return gpu_payload_bytes_per_block == 32 &&
                       gpu_codebook_id ==
                           (cpu_is_asymmetric != 0
                                ? kNativeVnniExpandedInt8MinCodebook
                                : static_cast<std::uint8_t>(19)) &&
                       gpu_is_asymmetric == cpu_is_asymmetric;
            case Encoding::Q6KNativeDualScale:
                return cpu_codebook_id == 8 &&
                       gpu_source_codebook_id == 8 &&
                       gpu_codebook_id == 8 &&
                       gpu_payload_bytes_per_block == 24 &&
                       cpu_is_asymmetric != 0 && gpu_is_asymmetric != 0;
            }
            return false;
        }

        /**
         * @brief Compute staging bytes for complete CPU units.
         * @param units Number of consecutive indivisible units.
         * @return Required bytes for the chunk.
         */
        [[nodiscard]] constexpr std::size_t chunkBytes(
            std::uint32_t units) const noexcept
        {
            return static_cast<std::size_t>(units) * cpu_block_stride;
        }
    };

    /** Read-only common CUDA/ROCm separated projection view. */
    struct ExpertTierGpuConstProjectionView
    {
        // Pointers name already-owned separated device allocations.
        const std::uint8_t *payload = nullptr;
        const std::uint16_t *scales = nullptr;
        const std::uint16_t *mins = nullptr;
        const std::uint32_t *emins = nullptr;
        // Capacities are exact so stale/undersized descriptors fail before launch.
        std::size_t payload_bytes = 0;
        std::size_t scales_bytes = 0;
        std::size_t mins_bytes = 0;
        std::size_t emins_bytes = 0;

        /**
         * @brief Validate read-only regions against a scalar layout.
         * @param layout Expected geometry, formats, and region sizes.
         * @return Whether pointers and capacities exactly satisfy `layout`.
         */
        [[nodiscard]] constexpr bool validFor(
            const ExpertTierWeightDeviceLayout &layout) const noexcept
        {
            // Validate scalar arithmetic before deriving any byte count.
            if (!layout.valid() || payload == nullptr || scales == nullptr)
                return false;
            // GPU arrays omit padded N and are block-major over logical rows.
            const std::size_t blocks =
                static_cast<std::size_t>(layout.N) *
                static_cast<std::size_t>(layout.blocks_per_row);
            return payload_bytes ==
                       blocks * layout.gpu_payload_bytes_per_block &&
                   scales_bytes == blocks * sizeof(std::uint16_t) &&
                   mins_bytes ==
                       (layout.gpu_is_asymmetric != 0
                            ? blocks * sizeof(std::uint16_t)
                            : 0u) &&
                   emins_bytes ==
                       (layout.gpu_has_emins != 0
                            ? blocks * sizeof(std::uint32_t)
                            : 0u) &&
                   (layout.gpu_is_asymmetric == 0 || mins != nullptr) &&
                   (layout.gpu_has_emins == 0 || emins != nullptr);
        }
    };

    /** Writable common CUDA/ROCm separated projection view. */
    struct ExpertTierGpuMutableProjectionView
    {
        // Mutable counterpart used only by CPU-to-GPU arrival conversion.
        std::uint8_t *payload = nullptr;
        std::uint16_t *scales = nullptr;
        std::uint16_t *mins = nullptr;
        std::uint32_t *emins = nullptr;
        // Capacities must describe the complete logical GPU projection.
        std::size_t payload_bytes = 0;
        std::size_t scales_bytes = 0;
        std::size_t mins_bytes = 0;
        std::size_t emins_bytes = 0;

        /**
         * @brief Convert mutable ownership to a read-only launch view.
         * @return A view with identical pointer identity and capacity.
         */
        [[nodiscard]] constexpr ExpertTierGpuConstProjectionView asConst()
            const noexcept
        {
            // Pointer values are preserved; only write permission is removed.
            return ExpertTierGpuConstProjectionView{
                .payload = payload,
                .scales = scales,
                .mins = mins,
                .emins = emins,
                .payload_bytes = payload_bytes,
                .scales_bytes = scales_bytes,
                .mins_bytes = mins_bytes,
                .emins_bytes = emins_bytes,
            };
        }

        /**
         * @brief Validate writable regions against a scalar layout.
         * @param layout Expected geometry, formats, and region sizes.
         * @return Whether pointers and capacities exactly satisfy `layout`.
         */
        [[nodiscard]] constexpr bool validFor(
            const ExpertTierWeightDeviceLayout &layout) const noexcept
        {
            return asConst().validFor(layout);
        }
    };
} // namespace llaminar2
