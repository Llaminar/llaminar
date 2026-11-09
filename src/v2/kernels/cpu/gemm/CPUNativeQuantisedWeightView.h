/**
 * @file CPUNativeQuantisedWeightView.h
 * @brief Checked immutable read view of the canonical prepared CPU weight stream.
 *
 * ISA leaves borrow published bytes; they neither allocate weights nor retain
 * a second source-order representation. This header has no Tensor/NUMA/arena
 * dependency, so native kernels can compile independently of the owning
 * storage implementation. Shape/encoding/span admission is performed once;
 * block readers retain the existing 64-column, 32-K physical layout. The
 * owning prepared store must retain its immutable publication during the call.
 *
 * Every view is backed by a CPUNativeQuantisedMetadataSeal: one complete pass over
 * the published metadata plane that rejects reserved source-coefficient bits.
 * Execution readers therefore never test metadata inside a worker team; the
 * owner of the bytes decides when a seal is current (see
 * CPUNativeQuantisedPublishedBytes).
 */
#pragma once

#include "CPUNativeQuantisedEncoding.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/NativeQuantisedIQ2SScale.h"
#include "tensors/NativeQuantisedIQ3SScale.h"
#include "tensors/NativeQuantisedIQ4XSScale.h"
#include "tensors/NativeQuantisedQ6Scale.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::cpu::native_quantised
{
    /** @brief Cold admission inputs for one published, resident prepared matrix. */
    struct CPUNativeQuantisedWeightDescriptor
    {
        int columns = 0; ///< Logical N before the canonical 64-column padding.
        int reduction_width = 0; ///< Logical K before final-block activation padding.
        NativeQuantisedSourceIdentity source; ///< Original source identity, not an execution alias.
        CPUNativeQuantisedEncoding encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
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
    inline float nativeQuantisedPrimaryScale(CPUNativeQuantisedEncoding encoding,
        std::uint8_t codebook, std::uint16_t primary, std::uint16_t secondary)
    {
        const float base = fp16_to_fp32(primary);
        if (encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale)
        {
            if (!native_quantised_iq3s_scale::valid(secondary))
                throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(base, static_cast<float>(native_quantised_iq3s_scale::coefficient(secondary)));
        }
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
        {
            if (!native_quantised_iq4xs_scale::valid(secondary))
                throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
            return device_fp32_contract::multiply(base, static_cast<float>(native_quantised_iq4xs_scale::coefficient(secondary)));
        }
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale)
            return device_fp32_contract::multiply(base, static_cast<float>(native_quantised_q6_scale::low(secondary)));
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale && codebook == 13)
        {
            if (!native_quantised_iq2s_scale::valid(secondary))
                throw std::invalid_argument("IQ2_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(device_fp32_contract::multiply(base,
                0.5f + static_cast<float>(native_quantised_iq2s_scale::low(secondary))), 0.25f);
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
    inline float nativeQuantisedSecondaryScale(CPUNativeQuantisedEncoding encoding,
        std::uint8_t codebook, std::uint16_t primary, std::uint16_t secondary)
    {
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale)
            return 0.0f;
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale)
            return device_fp32_contract::multiply(fp16_to_fp32(primary),
                static_cast<float>(native_quantised_q6_scale::high(secondary)));
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale && codebook == 13)
        {
            if (!native_quantised_iq2s_scale::valid(secondary))
                throw std::invalid_argument("IQ2_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(device_fp32_contract::multiply(fp16_to_fp32(primary),
                0.5f + static_cast<float>(native_quantised_iq2s_scale::high(secondary))), 0.25f);
        }
        return fp16_to_fp32(secondary);
    }

    class CPUNativeQuantisedWeightView;

    /**
     * @brief Proof that one exact byte range passed complete metadata admission.
     *
     * Only validate() constructs a seal, after checking the descriptor and the
     * byte extent and scanning every metadata word any reader can load,
     * including the padding lanes of partial 64-column chunks:
     *
     * | Encoding | Secondary word rule |
     * |---|---|
     * | ExpandedInt8 (symmetric or asymmetric) | none; the asymmetric word is an FP16 minimum |
     * | NibbleLUT (any admitted codebook) | none; an FP16 minimum where asymmetric |
     * | NibbleLUTSourceScale (IQ4_XS) | native_quantised_iq4xs_scale::valid, i.e. <= 63 |
     * | IQ3SExpandedSourceScale (IQ3_S) | native_quantised_iq3s_scale::valid, i.e. <= 15 |
     * | Q6KNativeDualScale | none; both bytes are signed coefficients |
     * | CompactMultiScale codebook 13 (IQ2_S) | native_quantised_iq2s_scale::valid, i.e. <= 255 |
     * | CompactMultiScale other codebooks | none; FP16 scales and minima |
     *
     * This is exactly the set of reserved-bit checks that execution performed
     * per K block. The seal binds the byte root, extent and every descriptor
     * field that changes how metadata is interpreted; arithmetic policy is not
     * part of metadata validity. Whether a seal is still current for mutable
     * storage is decided by its owner, not by this value.
     */
    class CPUNativeQuantisedMetadataSeal final
    {
    public:
        /**
         * @brief Admit the structure and scan the complete metadata plane once.
         * @param descriptor Shape/source/encoding publication.
         * @param bytes Exact canonical resident stream.
         * @return Seal of exactly these bytes under exactly this interpretation.
         * @throws std::invalid_argument For invalid structure or any reserved metadata bit.
         * @throws std::overflow_error For padded geometry/extent overflow.
         */
        [[nodiscard]] static CPUNativeQuantisedMetadataSeal validate(
            const CPUNativeQuantisedWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes);

        /**
         * @brief Test whether this seal covers a publication.
         * @param descriptor Interpretation the reader will apply.
         * @param bytes Byte range the reader will borrow.
         * @return Whether root, extent and metadata interpretation are identical.
         */
        [[nodiscard]] bool admits(const CPUNativeQuantisedWeightDescriptor &descriptor,
            std::span<const std::uint8_t> bytes) const noexcept
        {
            return bytes.data() == root_ && bytes.size() == extent_ &&
                descriptor.columns == identity_.columns &&
                descriptor.reduction_width == identity_.reduction_width &&
                descriptor.source.codebook_id == identity_.source.codebook_id &&
                descriptor.source.is_superblock == identity_.source.is_superblock &&
                descriptor.source.present == identity_.source.present &&
                descriptor.encoding == identity_.encoding &&
                descriptor.is_asymmetric == identity_.is_asymmetric;
        }

        /**
         * @brief The same proof for a content-identical copy at another root.
         * @param bytes Deep copy of the sealed range, written in full.
         * @return Seal of @p bytes; the copier vouches that the content is identical.
         * @throws std::invalid_argument If the extent differs.
         */
        [[nodiscard]] CPUNativeQuantisedMetadataSeal rebound(std::span<const std::uint8_t> bytes) const
        {
            if (bytes.size() != extent_ || (extent_ != 0 && !bytes.data()))
                throw std::invalid_argument("Native CPU metadata seal can only follow a complete identical copy");
            auto copy = *this;
            copy.root_ = bytes.data();
            return copy;
        }

    private:
        CPUNativeQuantisedMetadataSeal(const CPUNativeQuantisedWeightDescriptor &identity, std::span<const std::uint8_t> bytes) noexcept
            : identity_(identity), root_(bytes.data()), extent_(bytes.size()) {}

        CPUNativeQuantisedWeightDescriptor identity_; ///< Metadata interpretation; policy is ignored by admits().
        const std::uint8_t *root_ = nullptr;     ///< Exact sealed root; a move keeps it, a copy must rebind.
        std::size_t extent_ = 0;                 ///< Exact sealed extent.
    };

    /** @brief Validated immutable layout and borrowed storage for native CPU readers. */
    class CPUNativeQuantisedWeightView final
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
        const CPUNativeQuantisedEncoding encoding; ///< Validated physical encoding.
        const CPUProjectionNumericalPolicy numerical_policy; ///< Immutable arithmetic identity.
        const int data_stride; ///< Weight data bytes before inline metadata.
        const int interleaved_block_stride; ///< Total physical bytes per 64-column/K block.

        /// Every view is metadata-admitted: readers need no reserved-bit tests.
        static constexpr bool metadata_admitted = true;

        /**
         * @brief Admit a complete sealed resident publication without allocating/repacking it.
         * @param descriptor Original shape/source/encoding/arithmetic publication.
         * @param bytes Exact canonical resident stream, with at least uint32 alignment.
         * @param seal Metadata seal of exactly these bytes under this interpretation.
         * @return Immutable synchronous read view; the external owner retains storage.
         * @throws std::invalid_argument For invalid structure or a seal of other bytes.
         * @throws std::overflow_error For padded geometry/extent overflow.
         * @details O(1): the metadata plane was scanned when the seal was made.
         */
        [[nodiscard]] static CPUNativeQuantisedWeightView fromPublished(const CPUNativeQuantisedWeightDescriptor &descriptor,
            std::span<const std::uint8_t> bytes, const CPUNativeQuantisedMetadataSeal &seal)
        {
            if (!seal.admits(descriptor, bytes))
                throw std::invalid_argument("Native CPU weight view requires a metadata seal of exactly these bytes");
            return admitStructure(descriptor, bytes);
        }

        /** @return Whether a four-bit payload is retained in the stream. */
        [[nodiscard]] bool usesNibbleLUT() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::NibbleLUT || usesNibbleLUTSourceScale(); }
        /** @return Whether the nibble payload retains IQ4_XS original source factors. */
        [[nodiscard]] bool usesNibbleLUTSourceScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale; }
        /** @return Whether signed INT8 payload bytes are stored directly. */
        [[nodiscard]] bool usesExpandedInt8() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::ExpandedInt8 || usesIQ3SExpandedSourceScale(); }
        /** @return Whether IQ3_S original half/nibble factors accompany expanded grid bytes. */
        [[nodiscard]] bool usesIQ3SExpandedSourceScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale; }
        /** @return Whether the six-bit source bitplanes and coefficients are retained. */
        [[nodiscard]] bool usesQ6KNativeDualScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale; }
        /** @return Whether a compact multi-scale source payload is retained. */
        [[nodiscard]] bool usesCompactMultiScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::CompactMultiScale; }
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

        /**
         * @brief Primary words of an admitted unit, without coordinate checks.
         * @param chunk Chunk inside [0, N_padded/64), proven by the caller's binding.
         * @param block Block inside [0, blocks_per_row), proven by the caller's traversal.
         * @return Sixty-four primary half words.
         */
        [[nodiscard]] const std::uint16_t *admittedScales(int chunk, int block) const noexcept
        { return reinterpret_cast<const std::uint16_t *>(admittedUnit(chunk, block) + data_stride + (usesInlineCompensation() ? 128 : 0)); }

        /**
         * @brief Secondary words of an admitted unit, without coordinate or encoding checks.
         * @param chunk Chunk inside [0, N_padded/64), proven by the caller's binding.
         * @param block Block inside [0, blocks_per_row), proven by the caller's traversal.
         * @return Sixty-four secondary words; meaningful only for encodings that own them.
         */
        [[nodiscard]] const std::uint16_t *admittedMins(int chunk, int block) const noexcept
        { return reinterpret_cast<const std::uint16_t *>(admittedUnit(chunk, block) + data_stride + (usesInlineCompensation() ? 256 : 128)); }

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
            return nativeQuantisedPrimaryScale(encoding, codebook_id, chunkScales(chunk, block)[column], secondary);
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
            return nativeQuantisedSecondaryScale(encoding, codebook_id,
                chunkScales(chunk, block)[column], chunkMins(chunk, block)[column]);
        }

    private:
        friend class CPUNativeQuantisedMetadataSeal;

        /**
         * @brief Admit shape, source, encoding, policy and extent; metadata is not read.
         * @param descriptor Original shape/source/encoding/arithmetic publication.
         * @param bytes Exact canonical resident stream, with at least uint32 alignment.
         * @return Structurally checked view, used only by the metadata seal and sealed admission.
         * @throws std::invalid_argument For invalid source, encoding, arithmetic, shape or storage.
         * @throws std::overflow_error For padded geometry/extent overflow.
         */
        [[nodiscard]] static CPUNativeQuantisedWeightView admitStructure(
            const CPUNativeQuantisedWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes)
        {
            if (descriptor.columns <= 0 || descriptor.reduction_width <= 0 || !descriptor.source.present)
                throw std::invalid_argument("Native CPU weight view requires positive geometry and original source identity");
            const auto *source = native_quantised_formats::forSourceIdentity(
                descriptor.source.codebook_id, descriptor.source.is_superblock);
            if (!source || !isKnownPreparedEncoding(descriptor.encoding) ||
                (descriptor.numerical_policy != CPUProjectionNumericalPolicy::BackendNative &&
                 descriptor.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert))
                throw std::invalid_argument("Native CPU weight view requires a known source, encoding and numerical policy");
            // Explicit rotation may publish ExpandedInt8 for any original source.
            // Other encodings must retain the source-aware canonical preparation.
            if (descriptor.encoding != CPUNativeQuantisedEncoding::ExpandedInt8 &&
                descriptor.encoding != preparedEncodingForSource(*source))
                throw std::invalid_argument("Native CPU weight view encoding disagrees with original source");
            if (descriptor.is_asymmetric != source->is_asymmetric &&
                !(descriptor.encoding == CPUNativeQuantisedEncoding::ExpandedInt8 && !descriptor.is_asymmetric))
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
            return CPUNativeQuantisedWeightView(descriptor, *source, static_cast<int>(padded), static_cast<int>(blocks), bytes);
        }

        /** @brief Seal a completely validated immutable publication.
         * @param d Checked source/encoding/arithmetic descriptor.
         * @param source Canonical original source metadata.
         * @param padded Checked 64-column output capacity.
         * @param blocks Checked 32-value K block count.
         * @param bytes Exact externally owned resident stream.
         */
        CPUNativeQuantisedWeightView(const CPUNativeQuantisedWeightDescriptor &d, const NativeQuantisedFormatInfo &source,
            int padded, int blocks, std::span<const std::uint8_t> bytes)
            : N(d.columns), K(d.reduction_width), N_padded(padded), blocks_per_row(blocks),
              payload_bytes(d.encoding == CPUNativeQuantisedEncoding::ExpandedInt8 ||
                  d.encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ? 32 : source.payload_bytes),
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

        /** @brief Unit address for coordinates an admitted binding already proved in range. */
        [[nodiscard]] const std::uint8_t *admittedUnit(int chunk, int block) const noexcept
        { return bytes_.data() + (std::size_t(chunk) * blocks_per_row + block) * interleaved_block_stride; }

        std::span<const std::uint8_t> bytes_; ///< Borrowed immutable stream; storage remains prepared-store owned.
    };

    inline CPUNativeQuantisedMetadataSeal CPUNativeQuantisedMetadataSeal::validate(
        const CPUNativeQuantisedWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes)
    {
        const auto view = CPUNativeQuantisedWeightView::admitStructure(descriptor, bytes);
        bool (*rule)(std::uint16_t) noexcept = nullptr;
        const char *name = nullptr;
        if (view.usesIQ3SExpandedSourceScale())
        { rule = native_quantised_iq3s_scale::valid; name = "IQ3_S"; }
        else if (view.usesNibbleLUTSourceScale())
        { rule = native_quantised_iq4xs_scale::valid; name = "IQ4_XS"; }
        else if (view.usesCompactMultiScale() && view.codebook_id == 13)
        { rule = native_quantised_iq2s_scale::valid; name = "IQ2_S"; }
        if (rule)
        {
            // Every lane of every unit, padding columns included: wider native
            // loads read them, so the proof must not depend on a reader's width.
            for (int chunk = 0; chunk < view.N_padded / 64; ++chunk)
                for (int block = 0; block < view.blocks_per_row; ++block)
                {
                    const std::uint16_t *words = view.chunkMins(chunk, block);
                    std::uint16_t reserved = 0;
                    for (int column = 0; column < 64; ++column)
                        reserved |= rule(words[column]) ? 0 : 1;
                    if (reserved)
                        throw std::invalid_argument(std::string(name) + " coefficient contains reserved bits");
                }
        }
        return CPUNativeQuantisedMetadataSeal(descriptor, bytes);
    }
} // namespace llaminar2::cpu::native_quantised
