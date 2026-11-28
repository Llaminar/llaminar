/**
 * @file CPUNativeQuantisedPackedWeights.h
 * @brief One retained CPU interleaved-weight owner, independent of tensor preparation.
 *
 * Aligned backings retain their physical-allocation leases through moves and
 * release. Resident execution borrows one immutable bounded read view; source
 * archives and bounded cold preparation are separate from the owning ABI.
 * ISA-specific accessors read the same canonical 64-column by 32-K stream.
 * No tensor factory, activation rotation, device discovery or weight repacking
 * is needed to inspect/use a published owner.
 */
#pragma once
#include "CPUNativeQuantisedPublishedBytes.h"
#include "CPUNativeQuantisedWeightView.h"
#include "CPUNativeQuantisedPreparedFootprint.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/VnniPackContext.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#include "CPUNativeQuantisedMultiScaleKernels.h"
#endif

namespace llaminar2::cpu::native_quantised
{
    /** @brief Sole retained backing and metadata for an admitted prepared CPU matrix. */
    struct CPUNativeQuantisedPackedWeights
    {
        /// Optional source-order archive compatibility section. Newly prepared
        /// engines leave it empty; publication retires imported legacy bytes.
        AlignedVector<uint8_t> payload;

        /// VNNI-interleaved weight data with inline metadata (64-byte aligned).
        ///
        /// Each K-block contains group data followed by scales and optionally minima:
        ///   [groups × 4 ZMMs × 64B | 128B scales_fp16 | 128B mins_fp16]
        ///
        /// For nibble-LUT formats: 4 groups (1024B data) + 128B metadata = 1152B (symmetric)
        /// For Q8: 8 groups (2048B data) + 128B metadata = 2176B (symmetric)
        /// Asymmetric formats add 128B for mins.
        ///
        /// When deferred packing is active (workspace_data_ is set), this may be empty.
        /// Accessor methods automatically use workspace_data_ when set.
        /// Readers have const access; every write opens a new publication
        /// generation that a producer must seal before native execution.
        CPUNativeQuantisedPublishedBytes native_interleaved;

        /// Flat INT8 buffer used as intermediate during packing (INT8 pre-decoded formats only).
        /// Layout: [N_chunks][blocks_per_row][64][32] int8_t
        /// Freed after interleaving; only retained when keepDecodedBuffer is true (scalar tests).
        AlignedVector<int8_t> int8_flat;

        /// Original dimensions
        int N = 0;
        int K = 0;

        /// Padded N (rounded up to multiple of 64)
        int N_padded = 0;

        /// K-blocks per row (ceil(K/32))
        int blocks_per_row = 0;

        /// Bytes per native payload block
        int payload_bytes = 0;

        /// Format codebook ID for kernel dispatch
        uint8_t codebook_id = 0;

        /// Whether format has non-zero min offsets
        bool is_asymmetric = false;

        /// Whether format uses 256-element superblocks
        bool is_superblock = false;

        /// Physical prepared encoding consumed by serial and grouped kernels.
        CPUNativeQuantisedEncoding encoding = CPUNativeQuantisedEncoding::SignedInt8SingleScale;

        /// Interpretation of the primary scale words, derived from codebook and encoding.
        [[nodiscard]] CPUNativeQuantisedScaleEncoding scaleEncoding() const noexcept
        {
            return preparedScaleEncoding(codebook_id, encoding);
        }

        /// Pack-context layout that writes this unit's primary scale words.
        [[nodiscard]] VnniPrimaryScaleLayout primaryScaleLayout() const noexcept
        {
            return scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0
                       ? VnniPrimaryScaleLayout::NativeE8M0Bytes
                       : VnniPrimaryScaleLayout::FP16Words;
        }

        /**
         * Arithmetic identity selected by the owner of this prepared matrix.
         * ExpertOverlay matrices use the cross-backend reduction and activation
         * contract; ordinary non-expert CPU matrices retain tuned native math.
         */
        CPUProjectionNumericalPolicy numerical_policy =
            CPUProjectionNumericalPolicy::BackendNative;

        /// Bytes of pure group data per K-block (before metadata).
        /// 1024 for nibble-LUT (4 groups × 4 ZMMs × 64 bytes).
        /// 2048 for INT8 pre-decoded (8 groups × 4 ZMMs × 64 bytes).
        int data_stride = 1024;

        /// Total bytes per K-block including inline metadata.
        /// SignedInt8SingleScale: signed Q8 payload plus primary FP16 scale.
        int interleaved_block_stride = 1280;

        /** @brief Return whether this matrix uses native four-bit interleaving. */
        [[nodiscard]] bool usesNibbleLUT() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::NibbleLUT || usesNibbleLUTSourceScale() ||
                encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded;
        }

        /** @brief Whether raw IQ4_XS source factors accompany the nibble payload. */
        [[nodiscard]] bool usesNibbleLUTSourceScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
        }

        /** @brief Return whether this matrix retains native signed Q8 bytes. */
        [[nodiscard]] bool usesSignedInt8SingleScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        }

        /** @brief Return whether this matrix retains native dual-scale Q6_K. */
        [[nodiscard]] bool usesQ6KNativeDualScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale;
        }

        /** @brief Whether native payload and all multi-scale metadata are retained. */
        [[nodiscard]] bool usesCompactMultiScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        }

        /**
         * @brief Describe the exact prepared bytes consumed by cache tiling.
         *
         * `payload_bytes` describes one source-format block and is deliberately
         * absent from this contract.  The execution stream is
         * `interleaved_block_stride` bytes for each 64-column by 32-K unit,
         * including all metadata laid out by the packer.
         *
         * @return Complete prepared-weight, activation, and output footprint.
         */
        [[nodiscard]] NativeQuantisedPreparedFootprint preparedFootprint() const noexcept
        {
            return NativeQuantisedPreparedFootprint{
                .encoding = encoding,
                .is_asymmetric = is_asymmetric,
                // A coded block also owns 1/8 of its superblock header (8 words of 64).
                .weight_bytes_per_n_chunk_k_block =
                    static_cast<std::uint64_t>(interleaved_block_stride) + superblockHeaderBytes() / 8u,
                .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
                .output_bytes_per_row_n_chunk = 64u * sizeof(float),
            };
        }

        /**
         * @brief Borrow the exact published resident stream for architecture-native kernels.
         * @return Checked immutable read view without a new allocation or source mirror.
         * @throws std::logic_error For a deferred/unpublished binding or inconsistent owner geometry.
         * @throws std::invalid_argument For invalid source/encoding/arithmetic/byte metadata.
         * @details Only resident publication is accepted. Legacy workspace pointers
         * do not carry a bounded lease and cannot become an ARM hot-path repack route.
         */
        [[nodiscard]] CPUNativeQuantisedWeightView readView() const
        {
            if (workspace_data_)
                throw std::logic_error("Native CPU read view requires resident prepared weights, not a deferred workspace binding");
            const auto view = native_interleaved.admittedView(descriptor());
            if (view.N_padded != N_padded || view.blocks_per_row != blocks_per_row ||
                payload_bytes != view.source_payload_bytes || view.data_stride != data_stride ||
                view.interleaved_block_stride != interleaved_block_stride)
                throw std::logic_error("Native CPU prepared owner and resident publication geometry disagree");
            return view;
        }

        /** @return Metadata interpretation of this owner's published bytes. */
        [[nodiscard]] CPUNativeQuantisedWeightDescriptor descriptor() const noexcept
        {
            return {
                .columns = N, .reduction_width = K,
                .source = {.codebook_id = codebook_id, .is_superblock = is_superblock, .present = true},
                .encoding = encoding, .numerical_policy = numerical_policy,
                .is_asymmetric = is_asymmetric,
            };
        }

        /**
         * @brief Seal the current byte generation after its producer's last write.
         * @throws std::invalid_argument For invalid structure or reserved metadata bits.
         * @details One pass over the metadata plane; see CPUNativeQuantisedMetadataSeal
         * for the per-encoding rules. Producers call this, readers never do.
         */
        void sealPublication() { native_interleaved.seal(descriptor()); }

        /**
         * @brief Normalize metadata that admission checks in a partial final chunk's padding.
         * @throws std::invalid_argument For an unknown source/encoding or a non-canonical stored extent.
         * @throws std::overflow_error For unrepresentable geometry.
         * @throws std::logic_error If owner fields disagree with the canonical geometry.
         * @details Producers call this after their last live write and before
         * sealPublication(), inside the generation they own. Movement transfers
         * live columns only; padding fields would otherwise retain arbitrary
         * bytes in the retired slot, while the seal checks whole SIMD vectors.
         * Inline secondary words accept zero, and the native MXFP4 exponent
         * uses zero as its prepared zero-scale marker. Payload, unchecked
         * metadata, and every live column remain untouched; nothing is allocated.
         */
        void normalizePaddingMetadata()
        {
            // Canonical checked admission first: source, encoding, positive
            // geometry, widened extents and the exact stored size. No mutable
            // span or pointer arithmetic exists until all of it holds.
            const auto geometry = CPUNativeQuantisedWeightView::publishedGeometry(
                descriptor(), {native_interleaved.data(), native_interleaved.size()});
            if (geometry.padded_columns != N_padded || geometry.blocks_per_row != blocks_per_row ||
                geometry.data_stride != data_stride || geometry.interleaved_block_stride != interleaved_block_stride)
                throw std::logic_error("Native CPU padding normalization requires the descriptor-owned geometry");
            const bool byte_scale = scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0;
            if ((!geometry.has_secondary && !byte_scale) || geometry.columns % 64 == 0)
                return;
            const std::size_t field_bytes = byte_scale ? 1u : sizeof(std::uint16_t);
            const std::size_t field_offset = byte_scale ? geometry.data_stride : geometry.secondary_offset;
            const std::size_t chunks = std::size_t(geometry.padded_columns / 64);
            const std::size_t blocks = std::size_t(geometry.blocks_per_row);
            const std::size_t stride = std::size_t(geometry.interleaved_block_stride);
            const std::size_t live = std::size_t(geometry.columns % 64);
            const auto bytes = native_interleaved.unsealedGenerationBytes();
            for (std::size_t block = 0; block < blocks; ++block)
            {
                auto *plane = bytes.data() + ((chunks - 1) * blocks + block) * stride + field_offset;
                std::memset(plane + live * field_bytes, 0, (64 - live) * field_bytes);
            }
        }

        /** @return Whether the current byte generation is sealed for this owner's interpretation. */
        [[nodiscard]] bool publicationSealed() const noexcept { return native_interleaved.sealedFor(descriptor()); }

        // -------------------------------------------------------------------
        // Deferred packing (workspace) support
        // -------------------------------------------------------------------

        /// When set, accessor methods use this pointer instead of native_interleaved.data().
        /// This enables deferred packing: native blocks are stored permanently, and
        /// interleaved data is repacked into a shared workspace on demand.
        mutable const uint8_t *workspace_data_ = nullptr;

        /// Set the workspace pointer for deferred packing.
        /// After calling this, all accessor methods (interleavedB, chunkScales, etc.)
        /// will read from the workspace buffer instead of native_interleaved.
        void setWorkspace(const uint8_t *data) const { workspace_data_ = data; }

        /// Clear the workspace pointer. Accessors will revert to native_interleaved.
        void clearWorkspace() const { workspace_data_ = nullptr; }

        /// Returns the active interleaved data base pointer.
        /// Uses workspace_data_ if set, otherwise native_interleaved.data().
        inline const uint8_t *interleavedBase() const
        {
            return workspace_data_ ? workspace_data_ : native_interleaved.data();
        }

        // -------------------------------------------------------------------
        // Superblock-coded geometry (CPUNativeQuantisedPreparedFootprint.h)
        // -------------------------------------------------------------------

        /** @return Whether metadata is superblock-coded (codes + per-superblock words). */
        [[nodiscard]] bool superblockCoded() const noexcept { return preparedSuperblockCoded(encoding); }
        /** @return Bytes after each eight-block coded unit (0 for per-block encodings). */
        [[nodiscard]] std::size_t superblockHeaderBytes() const noexcept
        { return preparedSuperblockHeaderBytes(encoding, codebook_id); }
        /** @return Block/header addressing of this matrix (superblock-aligned). */
        [[nodiscard]] CPUNativeQuantisedBlockGeometry geometry() const noexcept
        {
            return {.block_stride = std::size_t(interleaved_block_stride), .header_bytes = superblockHeaderBytes(),
                    .blocks_per_row = std::size_t(blocks_per_row)};
        }
        /** @return Byte offset of block (c, kb). */
        [[nodiscard]] std::size_t blockOffset(int c, int kb) const noexcept
        { return geometry().blockOffset(std::size_t(c), std::size_t(kb)); }
        /** @return Byte offset of the superblock header serving block (c, kb). */
        [[nodiscard]] std::size_t superblockHeaderOffset(int c, int kb) const noexcept
        { return geometry().headerOffset(std::size_t(c), std::size_t(kb)); }
        /** @return Exact bytes of the complete prepared matrix. */
        [[nodiscard]] std::size_t preparedBytes() const noexcept
        { return geometry().matrixBytes(std::size_t(N_padded / 64)); }
        /** @return First byte of block (c, kb): payload planes, then code planes. */
        [[nodiscard]] const uint8_t *blockUnit(int c, int kb) const { return interleavedBase() + blockOffset(c, kb); }
        /** @return blockUnit() under an admitted binding (same address). */
        [[nodiscard]] const uint8_t *admittedBlockUnit(int c, int kb) const noexcept
        { return interleavedBase() + blockOffset(c, kb); }
        /** @return The 64 superblock words serving block (c, kb). */
        [[nodiscard]] const uint8_t *superblockHeader(int c, int kb) const
        {
            if (!superblockCoded())
                throw std::logic_error("Native CPU weight encoding has no superblock words");
            return interleavedBase() + superblockHeaderOffset(c, kb);
        }
        /** @return superblockHeader() under an admitted binding (same address). */
        [[nodiscard]] const uint8_t *admittedSuperblockHeader(int c, int kb) const noexcept
        { return interleavedBase() + superblockHeaderOffset(c, kb); }
        /** @return A coded column's per-block staged words reconstructed through the codec. */
        [[nodiscard]] native_quantised_superblock::StagedBlockMetadata codedBlockMetadata(int c, int kb, int column) const
        {
            const uint8_t *base = blockUnit(c, kb);
            const auto sb_codebook = superblockMetadataCodebook(encoding, codebook_id);
            uint8_t payload[16]{};
            if (native_quantised_superblock::superblockCodeInPayload(sb_codebook))
                compactColumnPayload(base, codebook_id, column, payload);
            const auto code = coded_metadata::columnCode(base, preparedCodedPlaneOffset(encoding, data_stride),
                                                         sb_codebook, column, payload);
            const auto word = coded_metadata::loadSuperblockWord(superblockHeader(c, kb), sb_codebook, column);
            return native_quantised_superblock::decodeSuperblockMetadata(sb_codebook, code, word);
        }

        // -------------------------------------------------------------------
        // Accessors for the kernel inner loop
        // -------------------------------------------------------------------

#if defined(__AVX2__) || defined(__AVX512F__)
        /**
         * @brief Widen the six-bit IQ4_XS codes of one SIMD strip.
         * @tparam Columns Eight or sixteen output columns per SIMD strip.
         * @param planes The block's code planes: a nibble plane, then a two-bit plane at byte 32.
         * @param column First column, aligned to the strip width.
         * @return One code in [0,63] per column.
         * @details BMI2 deposits the two narrow planes into byte slots before
         * one widening move; no shuffle masks occupy vector registers and no
         * byte is assembled with scalar inserts.
         */
        template <int Columns>
        [[nodiscard, gnu::always_inline]] static inline auto iq4xsCodes(const std::uint8_t *planes, int column)
        {
            using I = typename multi_scale::SIMD<Columns>::I;
            if constexpr (Columns == 16)
            {
                std::uint64_t low;
                std::uint32_t high;
                std::memcpy(&low, planes + column / 2, sizeof(low));
                std::memcpy(&high, planes + 32 + column / 4, sizeof(high));
                const auto first = _pdep_u64(low & 0xffffffffull, 0x0f0f0f0f0f0f0f0full) |
                    _pdep_u64(high & 0xffffu, 0x3030303030303030ull);
                const auto second = _pdep_u64(low >> 32, 0x0f0f0f0f0f0f0f0full) |
                    _pdep_u64(high >> 16, 0x3030303030303030ull);
                return std::bit_cast<I>(_mm512_cvtepu8_epi32(
                    _mm_set_epi64x(static_cast<long long>(second), static_cast<long long>(first))));
            }
            else
            {
                std::uint32_t low;
                std::uint16_t high;
                std::memcpy(&low, planes + column / 2, sizeof(low));
                std::memcpy(&high, planes + 32 + column / 4, sizeof(high));
                const auto bytes = _pdep_u64(low, 0x0f0f0f0f0f0f0f0full) |
                    _pdep_u64(high, 0x3030303030303030ull);
                return std::bit_cast<I>(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(bytes)));
            }
        }

        /**
         * @brief Widen the original IQ4_XS superblock d of one SIMD strip.
         * @param header The superblock header: one binary16 word per column.
         * @param column First column, aligned to the strip width.
         */
        template <int Columns>
        [[nodiscard, gnu::always_inline]] static inline auto iq4xsBases(const std::uint8_t *header, int column)
        {
            const auto *bases = header + 2 * column;
            using F = typename multi_scale::SIMD<Columns>::F;
            if constexpr (Columns == 16)
                return std::bit_cast<F>(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(bases))));
            else
                return std::bit_cast<F>(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bases))));
        }

        /**
         * @brief Widen the twelve-bit Q4_K codes of one SIMD strip.
         * @param planes The block's code planes: a byte plane, then a nibble plane at byte 64.
         * @param column First column, aligned to the strip width.
         * @return code = byte | nibble << 8 per column; one VPMOVZXBD widens the bytes.
         */
        template <int Columns>
        [[nodiscard, gnu::always_inline]] static inline auto q4kCodes(const std::uint8_t *planes, int column)
        {
            using Ops = multi_scale::SIMD<Columns>;
            using L = coded_metadata::Lanes<Ops>;
            typename Ops::I low;
            if constexpr (Columns == 16)
                low = std::bit_cast<typename Ops::I>(_mm512_cvtepu8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(planes + column))));
            else
                low = std::bit_cast<typename Ops::I>(_mm256_cvtepu8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(planes + column))));
            return low | (L::nibbles(planes + 64, column) << 8);
        }

        /**
         * @brief Reconstruct an IQ4_XS source scale strip in registers.
         * @tparam Columns Eight or sixteen output columns per SIMD strip.
         * @param chunk 64-column chunk index.
         * @param block Logical 32-value K block.
         * @param column First column, aligned to the strip width.
         * @return Original widened d multiplied by the signed six-bit factor.
         * @details The caller has proved the physical encoding. Fixed strides
         * avoid re-dispatching the format in every iteration of the K loop.
         */
        template <int Columns>
        [[nodiscard, gnu::always_inline]] inline auto loadIQ4XSCodedScales(int chunk, int block, int column) const
        {
            constexpr auto codebook = kNativeQuantisedIQ4XSSourceScaleCodebook;
            constexpr auto physical = CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
            constexpr auto source = native_quantised_formats::IQ4_XS.codebook_id;
            using Ops = multi_scale::SIMD<Columns>;
            const CPUNativeQuantisedBlockGeometry geometry{preparedInterleavedBlockStride(physical, false, source),
                preparedSuperblockHeaderBytes(physical, source), static_cast<std::size_t>(blocks_per_row)};
            const auto *base = interleavedBase() + geometry.blockOffset(chunk, 0);
            const auto *planes = base + geometry.blockOffset(0, block) +
                preparedCodedPlaneOffset(physical, preparedDataStride(physical, source));
            return coded_metadata::sourceScaleFactors<codebook, Ops>(iq4xsCodes<Columns>(planes, column),
                iq4xsBases<Columns>(base + geometry.headerOffset(0, block), column)).primary;
        }

        /**
         * @brief Reconstruct a Q4_K coded scale/minimum strip in registers.
         * @tparam Columns Eight or sixteen output columns per SIMD strip.
         * @param chunk 64-column chunk index.
         * @param block Logical 32-value K block.
         * @param column First column, aligned to the strip width.
         * @return Exactly coded_metadata::reconstruct<5>() of the block's
         *         12-bit codes and its superblock d|dmin words.
         * @details The caller has proved the physical encoding. Fixed strides
         * address the unit and header without per-block layout branches.
         */
        template <int Columns>
        [[nodiscard, gnu::always_inline]] inline auto loadQ4KCodedFactors(int chunk, int block, int column) const
        {
            constexpr auto physical = CPUNativeQuantisedEncoding::Q4KSuperblockCoded;
            constexpr auto source = native_quantised_formats::Q4_K.codebook_id;
            using Ops = multi_scale::SIMD<Columns>;
            using L = coded_metadata::Lanes<Ops>;
            const CPUNativeQuantisedBlockGeometry geometry{preparedInterleavedBlockStride(physical, true, source),
                preparedSuperblockHeaderBytes(physical, source), static_cast<std::size_t>(blocks_per_row)};
            const auto *base = interleavedBase() + geometry.blockOffset(chunk, 0);
            const auto *planes = base + geometry.blockOffset(0, block) + preparedDataStride(physical, source);
            return coded_metadata::reconstruct<5, Ops>(q4kCodes<Columns>(planes, column),
                                                       L::template superblockWords<5>(base + geometry.headerOffset(0, block), column));
        }
#endif

#if defined(__AVX512F__)
        /**
         * @brief Widen sixteen ordinary inline FP16 scale words, without factors.
         * @param primary Thirty-two-byte-aligned half words from one inline scale plane.
         * @return The original sixteen half values widened to FP32.
         *
         * This is only for per-block FP16 units. Coded scale/coefficients
         * and MXFP4 exponent bytes have their own layout-aware readers.
         */
        [[nodiscard]] static inline __m512 loadPlainFP16Scales512(const uint16_t *primary)
        {
            return _mm512_cvtph_ps(_mm256_load_si256(
                reinterpret_cast<const __m256i *>(primary)));
        }

        /** @brief Load 16 effective scales from an inline per-block plane.
         * @param primary Aligned primary words of a per-block encoding.
         * @return Widened FP16 scale.
         */
        [[nodiscard]] inline __m512 loadScales512(const std::uint16_t *primary) const
        {
            return loadPlainFP16Scales512(primary);
        }

        /** @brief Load 16 effective scales from an admitted nibble block.
         * @param chunk 64-column chunk index.
         * @param block Logical 32-value K block.
         * @param column First column of the strip, a multiple of 16.
         * @return Exact FP32 source factors or widened ordinary scale words.
         */
        [[nodiscard, gnu::always_inline]] inline __m512 loadScales512(int chunk, int block, int column) const
        {
            if (usesNibbleLUTSourceScale())
                return loadIQ4XSCodedScales<16>(chunk, block, column);
            const auto *primary = interleavedB(chunk, block, 0, 0) + data_stride + 2 * column;
            const __m256i words = _mm256_load_si256(reinterpret_cast<const __m256i *>(primary));
            return _mm512_cvtph_ps(words);
        }
#endif
#if defined(__AVX2__)
        /** @brief Load 8 effective scales from an inline per-block plane.
         * @param primary Aligned primary words of a per-block encoding.
         * @return Widened FP16 scale.
         */
        [[nodiscard]] inline __m256 loadScales256(const std::uint16_t *primary) const
        {
            const __m128i words = _mm_load_si128(reinterpret_cast<const __m128i *>(primary));
            return _mm256_cvtph_ps(words);
        }

        /** @brief Load 8 effective scales from an admitted nibble block.
         * @param chunk 64-column chunk index.
         * @param block Logical 32-value K block.
         * @param column First column of the strip, a multiple of 8.
         * @return Exact FP32 source factors or widened ordinary scale words.
         */
        [[nodiscard, gnu::always_inline]] inline __m256 loadScales256(int chunk, int block, int column) const
        {
            if (usesNibbleLUTSourceScale())
                return loadIQ4XSCodedScales<8>(chunk, block, column);
            const auto *primary = interleavedB(chunk, block, 0, 0) + data_stride + 2 * column;
            const __m128i words = _mm_load_si128(reinterpret_cast<const __m128i *>(primary));
            return _mm256_cvtph_ps(words);
        }
#endif

        /**
         * @brief Resolve the first source factor without narrowing its product.
         * @param c Physical64-column chunk index.
         * @param kb Logical32-value K-block index.
         * @param n_local Column0..63 within the chunk.
         * @return Q6_K/IQ2_S/IQ4_XS original-source factor or the existing widened scale.
         */
        inline float blockScale(int c, int kb, int n_local) const
        {
            if (superblockCoded())
            {
                const auto words = codedBlockMetadata(c, kb, n_local);
                return nativeQuantisedPrimaryScale(encoding, codebook_id, words.primary, words.secondary);
            }
            if (scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
            {
                const std::uint8_t e = blockUnit(c, kb)[data_stride + n_local];
                return std::bit_cast<float>(e ? mxfp4::normalScaleFP32Bits(e) : 0u);
            }
            const bool source_scale = usesNibbleLUTSourceScale() ||
                usesQ6KNativeDualScale() || (usesCompactMultiScale() && codebook_id == 13);
            const auto secondary = source_scale ? chunkMins(c, kb)[n_local] : 0;
            return nativeQuantisedPrimaryScale(encoding, codebook_id, chunkScales(c, kb)[n_local], secondary);
        }

        /**
         * @brief Resolve a second source factor or the existing minimum scale.
         * @param c Physical64-column chunk index.
         * @param kb Logical32-value K-block index.
         * @param n_local Column0..63 within the chunk.
         * @return Q6_K/IQ2_S second source factor or the existing widened minimum.
         */
        inline float blockMin(int c, int kb, int n_local) const
        {
            if (superblockCoded())
            {
                const auto words = codedBlockMetadata(c, kb, n_local);
                return nativeQuantisedSecondaryScale(encoding, codebook_id, words.primary, words.secondary);
            }
            if (usesNibbleLUTSourceScale() ||
                (!is_asymmetric && !usesQ6KNativeDualScale() && !usesCompactMultiScale()))
                return 0.0f;
            return nativeQuantisedSecondaryScale(encoding, codebook_id,
                chunkScales(c, kb)[n_local], chunkMins(c, kb)[n_local]);
        }

        /// Scales pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkScales(int c, int kb) const
        {
            if (superblockCoded())
                throw std::logic_error("Superblock-coded CPU weights hold no per-block metadata words");
            if (scaleEncoding() == CPUNativeQuantisedScaleEncoding::E8M0)
                throw std::logic_error("Native MXFP4 weights have byte scales, not half words");
            size_t block_offset = blockOffset(c, kb);
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride);
        }

        /// Secondary words for one chunk: FP16 minimum/scale, or Q6_K/IQ2_S/IQ3_S/IQ4_XS raw coefficients.
        inline const uint16_t *chunkMins(int c, int kb) const
        {
            if (superblockCoded())
                throw std::logic_error("Superblock-coded CPU weights hold no per-block metadata words");
            size_t block_offset = blockOffset(c, kb);
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + 128u);
        }

        /// Native-interleaved B data for N-chunk c, K-block kb, group g, ZMM z (0..3)
        /// Returns pointer to 64 bytes: 16 columns × 4 values each.
        /// For nibble-LUT: group ∈ [0,3], each holds 4 native payload bytes.
        /// For INT8 pre-decoded: group ∈ [0,7], each holds 4 INT8 values.
        inline const uint8_t *interleavedB(int c, int kb, int group, int z) const
        {
            size_t block_offset = blockOffset(c, kb);
            return interleavedBase() + block_offset + group * 256 + z * 64;
        }

        /**
         * @brief Return Q6_K high-bit planes for one K group/ZMM.
         *
         * The native Q6_K data region stores the familiar 1024-byte nibble
         * transpose first, followed by eight 64-byte group regions. Each
         * group/ZMM region contains two little-endian 64-bit masks. Mask bit
         * `lane * 4 + k` is the low or high bit of the two-bit Q6 high part
         * for byte `lane * 4 + k` in the corresponding VNNI weight vector.
         * This retains the native two-bits-per-weight footprint while allowing
         * AVX-512 to expand all 64 bytes with two masked broadcasts.
         *
         * @param c Output-column chunk.
         * @param kb Logical 32-value K block.
         * @param group VNNI group in [0, 7].
         * @param z Sixteen-column vector within the chunk in [0, 3].
         * @return Pointer to the low-bit mask followed by the high-bit mask.
         */
        inline const uint8_t *q6KHighBitplanes(
            int c, int kb, int group, int z) const
        {
            if (!usesQ6KNativeDualScale() || group < 0 || group >= 8 ||
                z < 0 || z >= 4)
            {
                throw std::logic_error(
                    "q6KHighBitplanes requires native Q6_K encoding and valid group/ZMM indices");
            }
            const size_t block_offset = blockOffset(c, kb);
            return interleavedBase() + block_offset + 1024u +
                   static_cast<size_t>(group) * 64u +
                   static_cast<size_t>(z) * 16u;
        }

        /**
         * @brief Retire optional legacy source-order bytes at execution publication.
         *
         * Archive readers retain the optional wire section until import has
         * completed. CPU engines use only eager interleaved bytes, including
         * scalar diagnostics, so publication must not retain this second matrix.
         */
        void releaseSourcePayload()
        {
            AlignedVector<uint8_t> empty;
            payload.swap(empty);
            AlignedVector<int8_t> decoded;
            int8_flat.swap(decoded);
        }

        /// Release the permanent interleaved data to save memory.
        /// Used with deferred packing: after this, GEMM/GEMV must set workspace_data_
        /// before accessing interleaved accessors (interleavedB, chunkScales, etc.).
        /// Also releases the payload array since native blocks are the primary storage.
        void releaseInterleavedData()
        {
            native_interleaved.release();
            payload.clear();
            payload.shrink_to_fit();
        }

        /// Returns true if the interleaved data is available (either owned or via workspace).
        bool hasInterleavedData() const
        {
            return workspace_data_ != nullptr || !native_interleaved.empty();
        }
    };

    /// Return the extent of the complete interleaved matrix.
    inline size_t interleavedWorkspaceSize(const CPUNativeQuantisedPackedWeights &meta)
    {
        return meta.preparedBytes();
    }

} // namespace llaminar2::cpu::native_quantised
