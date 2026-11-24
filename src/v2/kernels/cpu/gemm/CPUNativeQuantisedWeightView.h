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

#include "CPUNativeQuantisedCodedMetadata.h"
#include "CPUNativeQuantisedEncoding.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeQuantisedIQ2SScale.h"
#include "tensors/NativeQuantisedIQ3SScale.h"
#include "tensors/NativeQuantisedIQ4XSScale.h"
#include "tensors/NativeQuantisedQ6Scale.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
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
        CPUNativeQuantisedEncoding encoding = CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        CPUProjectionNumericalPolicy numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
        bool is_asymmetric = false; ///< Existing prepared metadata/stride property.
    };

    /**
     * @brief Reject retired prepared layouts at CPU receive and publication.
     * @param encoding Prepared encoding supplied by the sender or local owner.
     * @param codebook Original CPU source codebook, not a GPU execution alias.
     * @throws std::invalid_argument For encoding 5 or non-Q8 encoding 1.
     */
    inline void requireAdmittedCPUPreparedEncoding(CPUNativeQuantisedEncoding encoding, std::uint8_t codebook)
    {
        if (static_cast<std::uint8_t>(encoding) == kRetiredIQ3SExpandedSourceScaleEncoding)
            throw std::invalid_argument("CPU prepared encoding 5 (IQ3SExpandedSourceScale) is retired; IQ3_S requires CompactMultiScale");
        if (encoding == CPUNativeQuantisedEncoding::SignedInt8SingleScale &&
            !isSignedInt8SingleScaleSource(codebook))
            throw std::invalid_argument("CPU prepared encoding 1 (SignedInt8SingleScale) requires a Q8_0, Q8_1, or Q8_K source; received codebook " +
                std::to_string(codebook));
    }

    /**
     * @brief Identify an invalid Q8 byte extent matching the retired inline plane.
     * @param encoding Prepared encoding supplied by the sender or local owner.
     * @param codebook Original CPU source codebook.
     * @param bytes Observed extent of the units, including any legacy plane.
     * @param units Number of 64-column, 32-value units in the supplied extent.
     * @throws std::invalid_argument For Q8 encoding 1 with the legacy compensation plane.
     */
    inline void rejectLegacyQ8WeightSumExtent(CPUNativeQuantisedEncoding encoding,
        std::uint8_t codebook, std::uint64_t bytes, std::uint64_t units)
    {
        if (encoding == CPUNativeQuantisedEncoding::SignedInt8SingleScale && units != 0 &&
            bytes / units == preparedInterleavedBlockStride(encoding, false, codebook) + 128u && bytes % units == 0)
            throw std::invalid_argument("CPU prepared encoding 1 (SignedInt8SingleScale) does not admit a Q8 inline weight-sum compensation plane");
    }

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
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale && codebook == 11)
        {
            if (!native_quantised_iq3s_scale::valid(secondary))
                throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
            return device_fp32_contract::multiply(base,
                static_cast<float>(native_quantised_iq3s_scale::coefficient(secondary)));
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
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
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
        // IQ3_S is single-source-scale: it carries no additive minimum plane.
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale && codebook == 11)
            return 0.0f;
        return fp16_to_fp32(secondary);
    }

    class CPUNativeQuantisedWeightView;
    class CPUNativeQuantisedPublishedBytes;

    /**
     * @brief Proof that one exact byte range passed complete metadata admission.
     *
     * Only validate() constructs a seal, after checking the descriptor and the
     * byte extent and scanning every metadata word any reader can load,
     * including the padding lanes of a partial final 64-column chunk. Padding
     * is part of the common publication contract on every ISA (x86 loaders
     * test whole vectors): producers write a benign value there in the same
     * publication generation (see CPUNativeQuantisedPackedWeights::normalizePaddingMetadata),
     * and the seal rejects any garbage that remains:
     *
     * | Encoding | Secondary word rule |
     * |---|---|
     * | SignedInt8SingleScale | none; Q8 has only a primary FP16 scale |
     * | NibbleLUT (any admitted codebook) | none; an FP16 minimum where asymmetric |
     * | NibbleLUTSourceScale (IQ4_XS) | none; coded coefficients are six bits |
     * | Q6KNativeDualScale | none; both bytes are signed coefficients |
     * | CompactMultiScale | no per-block secondary word; every code and superblock word is in-domain |
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

    private:
        // Only the byte owner, while deep-copying bytes it holds, may move a
        // seal to a new root; nobody can attach a seal to arbitrary bytes.
        friend class CPUNativeQuantisedPublishedBytes;

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

        CPUNativeQuantisedMetadataSeal(const CPUNativeQuantisedWeightDescriptor &identity, std::span<const std::uint8_t> bytes) noexcept
            : identity_(identity), root_(bytes.data()), extent_(bytes.size()) {}

        CPUNativeQuantisedWeightDescriptor identity_; ///< Metadata interpretation; policy is ignored by admits().
        const std::uint8_t *root_ = nullptr;     ///< Exact sealed root; a move keeps it, a copy must rebind.
        std::size_t extent_ = 0;                 ///< Exact sealed extent.
    };

    /** @brief Canonical checked geometry of one stored publication; no readable view. */
    struct CPUNativeQuantisedPublishedGeometry
    {
        int columns = 0;                  ///< Live output columns N.
        int padded_columns = 0;           ///< Canonical 64-column capacity.
        int blocks_per_row = 0;           ///< ceil(K / 32).
        int data_stride = 0;              ///< Payload bytes before inline metadata.
        int interleaved_block_stride = 0; ///< Total bytes per 64-column/K-block unit.
        int secondary_offset = 0;         ///< Unit offset of the secondary metadata words.
        bool has_secondary = false;       ///< Whether the encoding stores secondary words.
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
        const int superblock_header_bytes; ///< Bytes after each eight-block coded unit (0 for per-block encodings).

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

        /**
         * @brief Admit structure and return canonical geometry without exposing a view.
         * @param descriptor Shape/source/encoding publication.
         * @param bytes Exact stored stream.
         * @return Checked geometry; nothing readable or executable is created.
         * @throws std::invalid_argument For an unknown source/encoding, inconsistent
         *         metadata or a stream that is not the exact canonical extent.
         * @throws std::overflow_error For padded geometry/extent overflow.
         * @details For producers that must write descriptor-owned planes (for
         * example padding normalization) before the publication is sealed.
         */
        [[nodiscard]] static CPUNativeQuantisedPublishedGeometry publishedGeometry(
            const CPUNativeQuantisedWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes)
        {
            const auto view = admitStructure(descriptor, bytes);
            const bool secondary = !view.superblockCoded() &&
                view.is_asymmetric;
            return {
                .columns = view.N, .padded_columns = view.N_padded, .blocks_per_row = view.blocks_per_row,
                .data_stride = view.data_stride, .interleaved_block_stride = view.interleaved_block_stride,
                .secondary_offset = view.data_stride + 128,
                .has_secondary = secondary,
            };
        }

        /** @return Whether a four-bit payload is retained in the stream. */
        [[nodiscard]] bool usesNibbleLUT() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::NibbleLUT || usesNibbleLUTSourceScale() ||
            encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded; }

        /** @brief Interpretation of the primary scale words, derived from codebook and encoding.
         * @return E8M0 for native nibble MXFP4; FP16 otherwise.
         */
        [[nodiscard]] CPUNativeQuantisedScaleEncoding scaleEncoding() const noexcept
        { return preparedScaleEncoding(codebook_id, encoding); }
        /** @return Whether the nibble payload retains IQ4_XS original source factors. */
        [[nodiscard]] bool usesNibbleLUTSourceScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale; }
        /** @return Whether signed INT8 payload bytes are stored directly. */
        [[nodiscard]] bool usesSignedInt8SingleScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::SignedInt8SingleScale; }
        [[nodiscard]] bool usesQ6KNativeDualScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale; }
        /** @return Whether a compact multi-scale source payload is retained. */
        [[nodiscard]] bool usesCompactMultiScale() const noexcept
        { return encoding == CPUNativeQuantisedEncoding::CompactMultiScale; }
        /** @return Borrowed canonical byte root; no source-order mirror exists. */
        [[nodiscard]] const std::uint8_t *interleavedBase() const noexcept { return bytes_.data(); }
        /** @return Exact borrowed publication extent. */
        [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept { return bytes_; }

        /** @return Whether metadata is superblock-coded (codes + per-superblock words). */
        [[nodiscard]] bool superblockCoded() const noexcept { return preparedSuperblockCoded(encoding); }
        /** @return Block/header addressing of this publication (superblock-aligned). */
        [[nodiscard]] CPUNativeQuantisedBlockGeometry geometry() const noexcept
        {
            return {.block_stride = std::size_t(interleaved_block_stride),
                    .header_bytes = std::size_t(superblock_header_bytes),
                    .blocks_per_row = std::size_t(blocks_per_row)};
        }

        /** @brief Borrow one complete primary metadata vector.
         * @param chunk Physical 64-column chunk.
         * @param block Logical 32-value K block.
         * @return Sixty-four primary half words in original column order.
         * @throws std::logic_error For superblock-coded units (no per-block words exist).
         */
        [[nodiscard]] const std::uint16_t *chunkScales(int chunk, int block) const
        {
            requirePerBlockMetadata();
            if (scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
                throw std::logic_error("Native MXFP4 weights have byte scales, not half words");
            return reinterpret_cast<const std::uint16_t *>(unit(chunk, block) + data_stride);
        }

        /** @return One admitted 64-column MXFP4 exponent plane (E8M0 with a zero marker). */
        [[nodiscard]] const std::uint8_t *admittedE8M0Scales(int chunk, int block) const noexcept
        { return admittedUnit(chunk, block) + preparedDataStride(CPUNativeQuantisedEncoding::NibbleLUT, 25); }

        /** @brief First byte of one 64-column block (payload planes, then code planes). */
        [[nodiscard]] const std::uint8_t *blockUnit(int chunk, int block) const { return unit(chunk, block); }
        /** @brief blockUnit() for coordinates an admitted binding proved in range. */
        [[nodiscard]] const std::uint8_t *admittedBlockUnit(int chunk, int block) const noexcept
        {
            if (chunk < 0 || chunk >= N_padded / 64 || block < 0 || block >= blocks_per_row) [[unlikely]]
                rejectOutsideAdmittedPublication();
            return bytes_.data() + geometry().blockOffset(std::size_t(chunk), std::size_t(block));
        }

        /**
         * @brief The 64 superblock words serving one coded block.
         * @param chunk Physical 64-column chunk.
         * @param block Logical K block (any block of the source superblock).
         * @return Header of the block's eight-block unit.
         * @throws std::logic_error For per-block encodings.
         */
        [[nodiscard]] const std::uint8_t *superblockHeader(int chunk, int block) const
        {
            if (!superblockCoded())
                throw std::logic_error("Native CPU weight encoding has no superblock words");
            (void)unit(chunk, block);
            return bytes_.data() + geometry().headerOffset(std::size_t(chunk), std::size_t(block));
        }
        /** @brief superblockHeader() for coordinates an admitted binding proved in range. */
        [[nodiscard]] const std::uint8_t *admittedSuperblockHeader(int chunk, int block) const noexcept
        {
            if (chunk < 0 || chunk >= N_padded / 64 || block < 0 || block >= blocks_per_row) [[unlikely]]
                rejectOutsideAdmittedPublication();
            return bytes_.data() + geometry().headerOffset(std::size_t(chunk), std::size_t(block));
        }

        /**
         * @brief Reconstruct a coded column's per-block staged words through the codec.
         * @return Exactly the words a per-block unit would have held (GPU contract).
         */
        [[nodiscard]] native_quantised_superblock::StagedBlockMetadata codedBlockMetadata(int chunk, int block, int column) const
        {
            validateColumn(column);
            const std::uint8_t *base = unit(chunk, block);
            const auto sb_codebook = superblockMetadataCodebook(encoding, codebook_id);
            uint8_t payload[16]{};
            if (native_quantised_superblock::superblockCodeInPayload(sb_codebook))
                compactColumnPayload(base, codebook_id, column, payload);
            const auto code = coded_metadata::columnCode(base, preparedCodedPlaneOffset(encoding, data_stride),
                                                         sb_codebook, column, payload);
            const auto word = coded_metadata::loadSuperblockWord(superblockHeader(chunk, block), sb_codebook, column);
            return native_quantised_superblock::decodeSuperblockMetadata(sb_codebook, code, word);
        }

        /** @brief Borrow the second source coefficient/scale/minimum vector.
         * @param chunk Physical 64-column chunk.
         * @param block Logical K block.
         * @return Sixty-four opaque secondary words.
         * @throws std::logic_error If this encoding has no secondary words.
         */
        [[nodiscard]] const std::uint16_t *chunkMins(int chunk, int block) const
        {
            requirePerBlockMetadata();
            if (!is_asymmetric && !usesNibbleLUTSourceScale() &&
                !usesQ6KNativeDualScale())
                throw std::logic_error("Native CPU weight encoding has no secondary metadata");
            return reinterpret_cast<const std::uint16_t *>(unit(chunk, block) + data_stride + 128);
        }

        /**
         * @brief Primary words of an admitted unit, without coordinate checks.
         * @param chunk Chunk inside [0, N_padded/64), proven by the caller's binding.
         * @param block Block inside [0, blocks_per_row), proven by the caller's traversal.
         * @return Sixty-four primary half words.
         */
        [[nodiscard]] const std::uint16_t *admittedScales(int chunk, int block) const noexcept
        { return reinterpret_cast<const std::uint16_t *>(admittedUnit(chunk, block) + data_stride); }

        /**
         * @brief Secondary words of an admitted unit, without coordinate or encoding checks.
         * @param chunk Chunk inside [0, N_padded/64), proven by the caller's binding.
         * @param block Block inside [0, blocks_per_row), proven by the caller's traversal.
         * @return Sixty-four secondary words; meaningful only for encodings that own them.
         */
        [[nodiscard]] const std::uint16_t *admittedMins(int chunk, int block) const noexcept
        { return reinterpret_cast<const std::uint16_t *>(admittedUnit(chunk, block) + data_stride + 128); }

        /** @return Primary words under an admitted compile-time per-block encoding. */
        template <CPUNativeQuantisedEncoding Physical, bool WithMinimum>
        [[nodiscard]] const std::uint16_t *admittedPerBlockPrimary(int chunk, int block) const noexcept
        {
            static_assert(Physical != CPUNativeQuantisedEncoding::SignedInt8SingleScale || !WithMinimum);
            const int offset = Physical == CPUNativeQuantisedEncoding::SignedInt8SingleScale
                ? interleaved_block_stride - 128 : preparedDataStride(Physical, 0);
            return reinterpret_cast<const std::uint16_t *>(admittedUnit(chunk, block) + offset);
        }

        /** @return Secondary words under an admitted compile-time per-block encoding. */
        template <CPUNativeQuantisedEncoding Physical>
        [[nodiscard]] const std::uint16_t *admittedPerBlockSecondary(int chunk, int block) const noexcept
        { return admittedPerBlockPrimary<Physical, true>(chunk, block) + 64; }

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
            if (group < 0 || group >= (usesCompactMultiScale() ? int(preparedCompactPayloadWords(codebook_id))
                                                                  : data_stride / 256) ||
                vector < 0 || vector >= 4)
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
            if (superblockCoded())
            {
                const auto words = codedBlockMetadata(chunk, block, column);
                return nativeQuantisedPrimaryScale(encoding, codebook_id, words.primary, words.secondary);
            }
            if (scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
            {
                const std::uint8_t e = unit(chunk, block)[data_stride + column];
                return std::bit_cast<float>(e ? mxfp4::normalScaleFP32Bits(e) : 0u);
            }
            const bool source_scale = usesNibbleLUTSourceScale() ||
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
            if (superblockCoded())
            {
                const auto words = codedBlockMetadata(chunk, block, column);
                return nativeQuantisedSecondaryScale(encoding, codebook_id, words.primary, words.secondary);
            }
            if (usesNibbleLUTSourceScale() ||
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
            requireAdmittedCPUPreparedEncoding(descriptor.encoding, descriptor.source.codebook_id);
            const auto *source = native_quantised_formats::forSourceIdentity(
                descriptor.source.codebook_id, descriptor.source.is_superblock);
            if (!source || !isKnownPreparedEncoding(descriptor.encoding) ||
                (descriptor.numerical_policy != CPUProjectionNumericalPolicy::BackendNative &&
                 descriptor.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert))
                throw std::invalid_argument("Native CPU weight view requires a known source, encoding and numerical policy");
            if (descriptor.encoding != preparedEncodingForSource(*source))
                throw std::invalid_argument("Native CPU weight view encoding disagrees with original source");
            if (descriptor.is_asymmetric != source->is_asymmetric)
                throw std::invalid_argument("Native CPU weight view has inconsistent secondary metadata");
            const std::int64_t padded = ((std::int64_t{descriptor.columns} + 63) / 64) * 64;
            const std::int64_t blocks = (std::int64_t{descriptor.reduction_width} + 31) / 32;
            if (padded > std::numeric_limits<int>::max())
                throw std::overflow_error("Native CPU padded output width exceeds its ABI");
            const std::size_t stride = preparedInterleavedBlockStride(descriptor.encoding, descriptor.is_asymmetric, descriptor.source.codebook_id);
            const std::size_t header = preparedSuperblockHeaderBytes(descriptor.encoding, descriptor.source.codebook_id);
            const std::size_t chunks = static_cast<std::size_t>(padded / 64);
            if (chunks > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(blocks) / (stride + header))
                throw std::overflow_error("Native CPU weight stream extent overflow");
            const std::size_t extent = CPUNativeQuantisedBlockGeometry{
                .block_stride = stride, .header_bytes = header, .blocks_per_row = static_cast<std::size_t>(blocks)}
                                           .matrixBytes(chunks);
            if (bytes.empty() || !bytes.data() || bytes.size() != extent ||
                reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(std::uint32_t) != 0)
            {
                rejectLegacyQ8WeightSumExtent(descriptor.encoding, descriptor.source.codebook_id,
                    bytes.size(), std::uint64_t(chunks) * blocks);
                throw std::invalid_argument("Native CPU weight view requires the exact aligned resident byte extent");
            }
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
              payload_bytes(source.payload_bytes),
              source_payload_bytes(source.payload_bytes), codebook_id(source.codebook_id),
              is_asymmetric(d.is_asymmetric), is_superblock(source.is_superblock), encoding(d.encoding),
              numerical_policy(d.numerical_policy), data_stride(preparedDataStride(d.encoding, d.source.codebook_id)),
              interleaved_block_stride(preparedInterleavedBlockStride(d.encoding, d.is_asymmetric, d.source.codebook_id)),
              superblock_header_bytes(static_cast<int>(preparedSuperblockHeaderBytes(d.encoding, d.source.codebook_id))),
              bytes_(bytes) {}

        /** @throws std::logic_error For superblock-coded units, which hold no per-block words. */
        void requirePerBlockMetadata() const
        {
            if (superblockCoded())
                throw std::logic_error("Superblock-coded CPU weights hold no per-block metadata words");
        }

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
            return bytes_.data() + geometry().blockOffset(std::size_t(chunk), std::size_t(block));
        }

        /**
         * @brief Unit address for coordinates an admitted binding already proved in range.
         * @details The bound is still tested, and a violation is fatal (never an
         * exception). The guarded branch is also a scheduling boundary: without
         * it compilers hoist later metadata/payload loads into the dot program
         * and spill vector registers (measured with gcc-14 and gcc-16).
         */
        [[nodiscard]] const std::uint8_t *admittedUnit(int chunk, int block) const noexcept
        {
            if (chunk < 0 || chunk >= N_padded / 64 || block < 0 || block >= blocks_per_row) [[unlikely]]
                rejectOutsideAdmittedPublication();
            // Per-block encodings only (their admitted metadata readers): the
            // original expression, with no superblock-header arithmetic in the
            // hot path. Coded readers use admittedBlockUnit().
            return bytes_.data() + (std::size_t(chunk) * blocks_per_row + block) * interleaved_block_stride;
        }

        /** @brief Fatal diagnostic for a coordinate outside an admitted publication. */
        [[noreturn, gnu::cold, gnu::noinline]] static void rejectOutsideAdmittedPublication() noexcept
        {
            std::fputs("ERROR: native CPU weight coordinate outside its admitted publication\n", stderr);
            std::abort();
        }

        std::span<const std::uint8_t> bytes_; ///< Borrowed immutable stream; storage remains prepared-store owned.
    };

    inline CPUNativeQuantisedMetadataSeal CPUNativeQuantisedMetadataSeal::validate(
        const CPUNativeQuantisedWeightDescriptor &descriptor, std::span<const std::uint8_t> bytes)
    {
        const auto view = CPUNativeQuantisedWeightView::admitStructure(descriptor, bytes);
        if (view.scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
            for (int chunk = 0; chunk < view.N_padded / 64; ++chunk)
                for (int block = 0; block < view.blocks_per_row; ++block)
                {
                    const auto *exponents = view.admittedE8M0Scales(chunk, block);
                    for (int column = 0; column < 64; ++column)
                        if (exponents[column] && !mxfp4::exponentFitsNativeE8M0(exponents[column]))
                            throw std::invalid_argument("Native MXFP4 prepared exponent is outside the admitted E8M0 range");
                }
        return CPUNativeQuantisedMetadataSeal(descriptor, bytes);
    }
} // namespace llaminar2::cpu::native_quantised
