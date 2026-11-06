/**
 * @file CPUNativeVNNIWeightView.h
 * @brief Checked immutable read view of the canonical prepared CPU weight stream.
 *
 * ISA leaves borrow published bytes; they neither allocate weights nor retain
 * a second source-order representation. This header has no Tensor/NUMA/arena
 * dependency, so native kernels can compile independently of the owning
 * storage implementation. Shape/encoding/span admission is performed once;
 * block readers retain the existing 64-column, 32-K physical layout. The
 * owning prepared store must retain its immutable publication during the call.
 */
#pragma once

#include "CPUNativeVNNIEncoding.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/NativeVnniIQ2SScale.h"
#include "tensors/NativeVnniIQ3SScale.h"
#include "tensors/NativeVnniIQ4XSScale.h"
#include "tensors/NativeVnniQ6Scale.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::native_vnni
{
    /** @brief Cold admission inputs for one published, resident prepared matrix. */
    struct CPUNativeVNNIWeightDescriptor
    {
        int columns = 0; ///< Logical N before the canonical 64-column padding.
        int reduction_width = 0; ///< Logical K before final-block activation padding.
        NativeVnniSourceIdentity source; ///< Original source identity, not an execution alias.
        CPUNativeVNNIEncoding encoding = CPUNativeVNNIEncoding::ExpandedInt8;
        CPUProjectionNumericalPolicy numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
        bool is_asymmetric = false; ///< Existing prepared metadata/stride property.
    };

    /**
     * @brief Decode the primary metadata factor without narrowing source products.
     * @param encoding Physical prepared encoding.
     * @param codebook Original CPU source codebook.
     * @param primary Persisted half scale or original source half base.
     * @param secondary Raw source coefficient word when required by the encoding.
     * @return Exact existing FP32 source factor.
     * @throws std::invalid_argument For reserved bits in source coefficient metadata.
     */
    inline float nativeVNNIPrimaryScale(CPUNativeVNNIEncoding encoding,
        std::uint8_t codebook, std::uint16_t primary, std::uint16_t secondary)
    {
        // Native MXFP4 units carry the exact BF16 word of their E8M0 power of
        // two; it widens by placement alone (word << 16), never through FP16.
        const float base = preparedScaleEncoding(codebook, encoding) == CPUNativeVNNIScaleEncoding::PowerOfTwoBF16
            ? std::bit_cast<float>(std::uint32_t{primary} << 16)
            : fp16_to_fp32(primary);
        if (encoding == CPUNativeVNNIEncoding::IQ3SExpandedSourceScale)
        {
            if (!native_vnni_iq3s_scale::valid(secondary))
                throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(base, static_cast<float>(native_vnni_iq3s_scale::coefficient(secondary)));
        }
        if (encoding == CPUNativeVNNIEncoding::NibbleLUTSourceScale)
        {
            if (!native_vnni_iq4xs_scale::valid(secondary))
                throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
            return device_fp32_contract::multiply(base, static_cast<float>(native_vnni_iq4xs_scale::coefficient(secondary)));
        }
        if (encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale)
            return device_fp32_contract::multiply(base, static_cast<float>(native_vnni_q6_scale::low(secondary)));
        if (encoding == CPUNativeVNNIEncoding::CompactMultiScale && codebook == 13)
        {
            if (!native_vnni_iq2s_scale::valid(secondary))
                throw std::invalid_argument("IQ2_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(device_fp32_contract::multiply(base,
                0.5f + static_cast<float>(native_vnni_iq2s_scale::low(secondary))), 0.25f);
        }
        return base;
    }

    /**
     * @brief Decode the secondary source factor or additive minimum.
     * @param encoding Physical prepared encoding.
     * @param codebook Original CPU source codebook.
     * @param primary Persisted original half base where required.
     * @param secondary Original coefficient word or persisted half minimum/scale.
     * @return Exact second source factor, or zero for single-source-scale tuples.
     * @throws std::invalid_argument For reserved IQ2_S coefficient bits.
     */
    inline float nativeVNNISecondaryScale(CPUNativeVNNIEncoding encoding,
        std::uint8_t codebook, std::uint16_t primary, std::uint16_t secondary)
    {
        if (encoding == CPUNativeVNNIEncoding::NibbleLUTSourceScale ||
            encoding == CPUNativeVNNIEncoding::IQ3SExpandedSourceScale)
            return 0.0f;
        if (encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale)
            return device_fp32_contract::multiply(fp16_to_fp32(primary),
                static_cast<float>(native_vnni_q6_scale::high(secondary)));
        if (encoding == CPUNativeVNNIEncoding::CompactMultiScale && codebook == 13)
        {
            if (!native_vnni_iq2s_scale::valid(secondary))
                throw std::invalid_argument("IQ2_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(device_fp32_contract::multiply(fp16_to_fp32(primary),
                0.5f + static_cast<float>(native_vnni_iq2s_scale::high(secondary))), 0.25f);
        }
        return fp16_to_fp32(secondary);
    }

    /** @brief Validated immutable layout and borrowed storage for native CPU readers. */
    class CPUNativeVNNIWeightView final
    {
    public:
        const int N; ///< Live output columns.
        const int K; ///< Live reduction elements.
        const int N_padded; ///< Canonical 64-column physical capacity.
        const int blocks_per_row; ///< Canonical 32-value K blocks.
        const int payload_bytes; ///< Physical execution payload bytes per column/K block.
        const int source_payload_bytes; ///< Original codec payload width before execution expansion.
        const std::uint8_t codebook_id; ///< Original CPU source codebook.
        const bool is_asymmetric; ///< Published metadata/stride property.
        const bool is_superblock; ///< Original source block geometry.
        const CPUNativeVNNIEncoding encoding; ///< Validated physical encoding.
        const CPUProjectionNumericalPolicy numerical_policy; ///< Immutable arithmetic identity.
        const int data_stride; ///< Weight data bytes before inline metadata.
        const int interleaved_block_stride; ///< Total physical bytes per 64-column/K block.

        /**
         * @brief Admit a complete resident publication without allocating/repacking it.
         * @param descriptor Original shape/source/encoding/arithmetic publication.
         * @param bytes Exact canonical resident stream, with at least uint32 alignment.
         * @return Immutable synchronous read view; the external owner retains storage.
         * @throws std::invalid_argument For invalid source, encoding, arithmetic, shape or storage.
         * @throws std::overflow_error For padded geometry/extent overflow.
         */
        [[nodiscard]] static CPUNativeVNNIWeightView fromPublished(
            const CPUNativeVNNIWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes)
        {
            if (descriptor.columns <= 0 || descriptor.reduction_width <= 0 || !descriptor.source.present)
                throw std::invalid_argument("Native CPU weight view requires positive geometry and original source identity");
            const auto *source = native_vnni_formats::forSourceIdentity(
                descriptor.source.codebook_id, descriptor.source.is_superblock);
            if (!source || !isKnownPreparedEncoding(descriptor.encoding) ||
                (descriptor.numerical_policy != CPUProjectionNumericalPolicy::BackendNative &&
                 descriptor.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert))
                throw std::invalid_argument("Native CPU weight view requires a known source, encoding and numerical policy");
            // Explicit rotation may publish ExpandedInt8 for any original source.
            // Other encodings must retain the source-aware canonical preparation.
            if (descriptor.encoding != CPUNativeVNNIEncoding::ExpandedInt8 &&
                descriptor.encoding != preparedEncodingForSource(*source))
                throw std::invalid_argument("Native CPU weight view encoding disagrees with original source");
            if (descriptor.is_asymmetric != source->is_asymmetric &&
                !(descriptor.encoding == CPUNativeVNNIEncoding::ExpandedInt8 && !descriptor.is_asymmetric))
                throw std::invalid_argument("Native CPU weight view has inconsistent secondary metadata");
            const std::int64_t padded = ((std::int64_t{descriptor.columns} + 63) / 64) * 64;
            const std::int64_t blocks = (std::int64_t{descriptor.reduction_width} + 31) / 32;
            if (padded > std::numeric_limits<int>::max())
                throw std::overflow_error("Native CPU padded output width exceeds its ABI");
            const std::size_t stride = preparedInterleavedBlockStride(descriptor.encoding, descriptor.is_asymmetric);
            const std::size_t chunks = static_cast<std::size_t>(padded / 64);
            if (chunks > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(blocks) / stride)
                throw std::overflow_error("Native CPU weight stream extent overflow");
            const std::size_t extent = chunks * static_cast<std::size_t>(blocks) * stride;
            if (bytes.empty() || !bytes.data() || bytes.size() != extent ||
                reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(std::uint32_t) != 0)
                throw std::invalid_argument("Native CPU weight view requires the exact aligned resident byte extent");
            static_assert(std::endian::native == std::endian::little,
                          "Canonical CPU native weight metadata requires little-endian storage");
            return CPUNativeVNNIWeightView(descriptor, *source, static_cast<int>(padded), static_cast<int>(blocks), bytes);
        }

        /** @return Whether a four-bit payload is retained in the stream. */
        [[nodiscard]] bool usesNibbleLUT() const noexcept
        { return encoding == CPUNativeVNNIEncoding::NibbleLUT || usesNibbleLUTSourceScale(); }

        /** @brief Interpretation of the primary scale words, derived from codebook and encoding.
         * @return PowerOfTwoBF16 for native nibble MXFP4; FP16 otherwise (including rotated MXFP4).
         */
        [[nodiscard]] CPUNativeVNNIScaleEncoding scaleEncoding() const noexcept
        { return preparedScaleEncoding(codebook_id, encoding); }
        /** @return Whether the nibble payload retains IQ4_XS original source factors. */
        [[nodiscard]] bool usesNibbleLUTSourceScale() const noexcept
        { return encoding == CPUNativeVNNIEncoding::NibbleLUTSourceScale; }
        /** @return Whether signed INT8 payload bytes are stored directly. */
        [[nodiscard]] bool usesExpandedInt8() const noexcept
        { return encoding == CPUNativeVNNIEncoding::ExpandedInt8 || usesIQ3SExpandedSourceScale(); }
        /** @return Whether IQ3_S original half/nibble factors accompany expanded grid bytes. */
        [[nodiscard]] bool usesIQ3SExpandedSourceScale() const noexcept
        { return encoding == CPUNativeVNNIEncoding::IQ3SExpandedSourceScale; }
        /** @return Whether the six-bit source bitplanes and coefficients are retained. */
        [[nodiscard]] bool usesQ6KNativeDualScale() const noexcept
        { return encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale; }
        /** @return Whether a compact multi-scale source payload is retained. */
        [[nodiscard]] bool usesCompactMultiScale() const noexcept
        { return encoding == CPUNativeVNNIEncoding::CompactMultiScale; }
        /** @return Whether inline signed weight-sum compensation precedes primary scales. */
        [[nodiscard]] bool usesInlineCompensation() const noexcept
        { return !usesQ6KNativeDualScale() && !usesCompactMultiScale(); }
        /** @return Borrowed canonical byte root; no source-order mirror exists. */
        [[nodiscard]] const std::uint8_t *interleavedBase() const noexcept { return bytes_.data(); }
        /** @return Exact borrowed publication extent. */
        [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept { return bytes_; }

        /** @brief Borrow one complete primary metadata vector.
         * @param chunk Physical 64-column chunk.
         * @param block Logical 32-value K block.
         * @return Sixty-four primary half words in original column order.
         */
        [[nodiscard]] const std::uint16_t *chunkScales(int chunk, int block) const
        { return reinterpret_cast<const std::uint16_t *>(unit(chunk, block) + data_stride + (usesInlineCompensation() ? 128 : 0)); }

        /** @brief Borrow the second source coefficient/scale/minimum vector.
         * @param chunk Physical 64-column chunk.
         * @param block Logical K block.
         * @return Sixty-four opaque secondary words.
         * @throws std::logic_error If this encoding has no secondary words.
         */
        [[nodiscard]] const std::uint16_t *chunkMins(int chunk, int block) const
        {
            if (!is_asymmetric && !usesNibbleLUTSourceScale() && !usesIQ3SExpandedSourceScale() &&
                !usesQ6KNativeDualScale() && !usesCompactMultiScale())
                throw std::logic_error("Native CPU weight encoding has no secondary metadata");
            return reinterpret_cast<const std::uint16_t *>(unit(chunk, block) + data_stride + (usesInlineCompensation() ? 256 : 128));
        }

        /** @brief Borrow Q2_K's packed pair of effective minima.
         * @param chunk Physical output chunk.
         * @param block Logical K block.
         * @return Sixty-four raw low/high half-word pairs.
         * @throws std::logic_error If the encoding is not compact multi-scale.
         */
        [[nodiscard]] const std::uint32_t *chunkEffectiveMins(int chunk, int block) const
        {
            if (!usesCompactMultiScale())
                throw std::logic_error("Effective minima require compact CPU weights");
            return reinterpret_cast<const std::uint32_t *>(unit(chunk, block) + data_stride + 256);
        }

        /** @brief Borrow one 16-column/four-byte interleaving group.
         * @param chunk Physical output chunk.
         * @param block Logical K block.
         * @param group Four-byte payload group admitted by the data stride.
         * @param vector Sixteen-column vector number in [0,3].
         * @return Sixty-four live payload bytes.
         * @throws std::out_of_range For invalid payload group/vector indices.
         */
        [[nodiscard]] const std::uint8_t *interleavedB(int chunk, int block, int group, int vector) const
        {
            if (group < 0 || group >= data_stride / 256 || vector < 0 || vector >= 4)
                throw std::out_of_range("Native CPU interleaved payload group is outside its unit");
            return unit(chunk, block) + group * 256 + vector * 64;
        }

        /** @brief Reconstruct the first source factor from its published metadata.
         * @param chunk Physical output chunk.
         * @param block Logical K block.
         * @param column Column within the physical chunk in [0,63].
         * @return Exact original first factor, with no new half rounding.
         */
        [[nodiscard]] float blockScale(int chunk, int block, int column) const
        {
            validateColumn(column);
            const bool source_scale = usesNibbleLUTSourceScale() || usesIQ3SExpandedSourceScale() ||
                usesQ6KNativeDualScale() || (usesCompactMultiScale() && codebook_id == 13);
            const auto secondary = source_scale ? chunkMins(chunk, block)[column] : 0;
            return nativeVNNIPrimaryScale(encoding, codebook_id, chunkScales(chunk, block)[column], secondary);
        }

        /** @brief Reconstruct the second source factor/additive minimum.
         * @param chunk Physical output chunk.
         * @param block Logical K block.
         * @param column Column within [0,63].
         * @return Exact original second factor; zero for single-factor source tuples.
         */
        [[nodiscard]] float blockMin(int chunk, int block, int column) const
        {
            validateColumn(column);
            (void)unit(chunk, block);
            if (usesNibbleLUTSourceScale() || usesIQ3SExpandedSourceScale() ||
                (!is_asymmetric && !usesQ6KNativeDualScale() && !usesCompactMultiScale()))
                return 0.0f;
            return nativeVNNISecondaryScale(encoding, codebook_id,
                chunkScales(chunk, block)[column], chunkMins(chunk, block)[column]);
        }

    private:
        /** @brief Seal a completely validated immutable publication.
         * @param d Checked source/encoding/arithmetic descriptor.
         * @param source Canonical original source metadata.
         * @param padded Checked 64-column output capacity.
         * @param blocks Checked 32-value K block count.
         * @param bytes Exact externally owned resident stream.
         */
        CPUNativeVNNIWeightView(const CPUNativeVNNIWeightDescriptor &d, const NativeVnniFormatInfo &source,
            int padded, int blocks, std::span<const std::uint8_t> bytes)
            : N(d.columns), K(d.reduction_width), N_padded(padded), blocks_per_row(blocks),
              payload_bytes(d.encoding == CPUNativeVNNIEncoding::ExpandedInt8 ||
                  d.encoding == CPUNativeVNNIEncoding::IQ3SExpandedSourceScale ? 32 : source.payload_bytes),
              source_payload_bytes(source.payload_bytes), codebook_id(source.codebook_id),
              is_asymmetric(d.is_asymmetric), is_superblock(source.is_superblock), encoding(d.encoding),
              numerical_policy(d.numerical_policy), data_stride(preparedDataStride(d.encoding)),
              interleaved_block_stride(preparedInterleavedBlockStride(d.encoding, d.is_asymmetric)), bytes_(bytes) {}

        /** @brief Check one column coordinate without clamping it into a valid lane.
         * @param column Physical column within one 64-column unit.
         * @throws std::out_of_range For a coordinate outside [0,63].
         */
        static void validateColumn(int column)
        {
            if (column < 0 || column >= 64)
                throw std::out_of_range("Native CPU weight column is outside its physical chunk");
        }

        /** @brief Validate one physical unit coordinate and borrow its bounded byte span.
         * @param chunk Physical 64-column chunk in the admitted matrix.
         * @param block Logical K block within the original row.
         * @return Complete resident unit's first byte, retained by the external owner.
         * @throws std::out_of_range For any coordinate outside the admitted extent.
         */
        [[nodiscard]] const std::uint8_t *unit(int chunk, int block) const
        {
            if (chunk < 0 || chunk >= N_padded / 64 || block < 0 || block >= blocks_per_row)
                throw std::out_of_range("Native CPU weight unit is outside the admitted publication");
            return bytes_.data() + (std::size_t(chunk) * blocks_per_row + block) * interleaved_block_stride;
        }

        std::span<const std::uint8_t> bytes_; ///< Borrowed immutable stream; storage remains prepared-store owned.
    };
} // namespace llaminar2::cpu::native_vnni
