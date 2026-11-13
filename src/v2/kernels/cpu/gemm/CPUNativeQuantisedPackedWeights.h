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
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Integer coefficient interpretation guaranteed by canonical preparation.
     *
     * This describes the unscaled integer payload, not its final FP32 weight.
     * Other asymmetric families may have negative decoded bytes and therefore
     * cannot infer unsigned operands merely from minimum-plane presence.
     */
    enum class CPUNativeQuantisedCoefficientDomain : uint8_t
    {
        SignedInt8,     ///< General decoded coefficients in [-128, 127].
        UnsignedFiveBit, ///< Canonical Q5_1/Q5_K coefficients in [0, 31].
    };

    /** @brief Sole retained backing and metadata for an admitted prepared CPU matrix. */
    struct CPUNativeQuantisedPackedWeights
    {
        /// Optional source-order archive compatibility section. Newly prepared
        /// engines leave it empty; publication retires imported legacy bytes.
        AlignedVector<uint8_t> payload;

        /// VNNI-interleaved weight data with inline metadata (64-byte aligned).
        ///
        /// Each K-block contains group data followed by comp, scales, and optionally mins:
        ///   [groups × 4 ZMMs × 64B | 128B comp_int16 | 128B scales_fp16 | 128B mins_fp16]
        ///
        /// For nibble-LUT formats: 4 groups (1024B data) + 256B metadata = 1280B (symmetric)
        /// For INT8 pre-decoded: 8 groups (2048B data) + 256B metadata = 2304B (symmetric)
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
        CPUNativeQuantisedEncoding encoding = CPUNativeQuantisedEncoding::ExpandedInt8;

        /// Physical unit layout of the prepared stream this owner publishes.
        /// The packer fixes it once at preparation time from kHostPreparedLayout;
        /// every reader and movement link below re-derives geometry from it.
        CPUNativeQuantisedLayout layout = CPUNativeQuantisedLayout::InterleavedCanonical;

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
        /// = data_stride + 128 (comp) + 128 (scales) [+ 128 (mins) if asymmetric]
        int interleaved_block_stride = 1280;

        /** @brief Return whether this matrix uses native four-bit interleaving. */
        [[nodiscard]] bool usesNibbleLUT() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::NibbleLUT || usesNibbleLUTSourceScale();
        }

        /** @brief Whether raw IQ4_XS source factors accompany the nibble payload. */
        [[nodiscard]] bool usesNibbleLUTSourceScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
        }

        /** @brief Return whether this matrix was expanded to signed INT8. */
        [[nodiscard]] bool usesExpandedInt8() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::ExpandedInt8 || usesIQ3SExpandedSourceScale();
        }

        /**
         * @brief Resolve signedness from the immutable canonical prepared tuple.
         * @return UnsignedFiveBit only for ordinary expanded Q5 coefficients.
         *
         * Both Q5_1 and Q5_K keep their original nonnegative five-bit codes;
         * scale/minimum terms stay separate. Signed Q5_0/Q8 and source-factor
         * encodings retain the general signed interpretation. No payload scan,
         * cache, capability probe, storage change or numerical-policy override
         * is needed to choose VNNI's unsigned operand.
         */
        [[nodiscard]] CPUNativeQuantisedCoefficientDomain coefficientDomain() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::ExpandedInt8 &&
                    codebook_id == native_quantised_formats::Q5_1.codebook_id &&
                    is_asymmetric
                ? CPUNativeQuantisedCoefficientDomain::UnsignedFiveBit
                : CPUNativeQuantisedCoefficientDomain::SignedInt8;
        }

        /** @brief Whether decoded IQ3_S grid bytes retain original source factors. */
        [[nodiscard]] bool usesIQ3SExpandedSourceScale() const noexcept
        {
            return encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale;
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
         * @brief Return the one section authority for this owner's stream.
         * @details Cold admission/copier queries derive which sections exist
         * from the layout tuple, never from the encoding family alone; the
         * result is meaningful only when non-zero `block_stride` proves the
         * layout is implemented.
         */
        [[nodiscard]] PreparedUnitSections preparedSections() const noexcept
        {
            const auto *const source = native_quantised_formats::forSourceIdentity(
                codebook_id, is_superblock);
            return source ? preparedUnitSections(layout, encoding, is_asymmetric, *source)
                          : PreparedUnitSections{};
        }

        /**
         * @brief Return whether each prepared block carries weight compensation.
         *
         * Nibble and expanded-INT8 kernels bias signed weights by shifting the
         * activation to unsigned bytes, and therefore need one per-column
         * weight sum. Q6_K reverses the VNNI operands: unsigned six-bit weights
         * multiply signed activations, so its centering correction depends only
         * on the activation half sums and no weight compensation is stored.
         */
        [[nodiscard]] bool usesInlineCompensation() const noexcept
        {
            return layout == CPUNativeQuantisedLayout::InterleavedCanonical &&
                !usesQ6KNativeDualScale() && !usesCompactMultiScale();
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
                .layout = layout,
                .encoding = encoding,
                .is_asymmetric = is_asymmetric,
                .weight_bytes_per_n_chunk_k_block =
                    static_cast<std::uint64_t>(interleaved_block_stride),
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
            // Direct preparation retains the input codec width; migrated/slot
            // ExpandedInt8 owners publish 32. The read view separates original
            // codec width from physical execution width without changing either
            // producer's existing serialization or the resident bytes.
            const bool payload_matches = payload_bytes == view.source_payload_bytes ||
                (view.usesExpandedInt8() && payload_bytes == view.payload_bytes);
            if (view.N_padded != N_padded || view.blocks_per_row != blocks_per_row ||
                !payload_matches || view.layout() != layout || view.data_stride != data_stride ||
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
                .encoding = encoding, .layout = layout, .numerical_policy = numerical_policy,
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
         * @brief Write the benign value into every padding metadata word of a partial final chunk.
         * @throws std::invalid_argument For an unknown source/encoding or a non-canonical stored extent.
         * @throws std::overflow_error For unrepresentable geometry.
         * @throws std::logic_error If owner fields disagree with the canonical geometry.
         * @details Producers call this after their last live write and before
         * sealPublication(), inside the generation they own. Movement transfers
         * live columns only; padding words of the final chunk would otherwise keep
         * whatever the storage held, while every ISA's readers load whole vectors.
         * Zero is in-domain for every secondary word (IQ3_S, IQ4_XS and IQ2_S
         * coefficients, Q6_K coefficients, FP16 minima). Payload bytes, primary
         * scales and live columns are untouched; nothing is allocated.
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
            if (!geometry.has_secondary || geometry.columns % 64 == 0)
                return;
            const std::size_t chunks = std::size_t(geometry.padded_columns / 64);
            const std::size_t blocks = std::size_t(geometry.blocks_per_row);
            const std::size_t stride = std::size_t(geometry.interleaved_block_stride);
            const std::size_t live = std::size_t(geometry.columns % 64);
            const auto bytes = native_interleaved.unsealedGenerationBytes();
            for (std::size_t block = 0; block < blocks; ++block)
            {
                auto *words = bytes.data() + ((chunks - 1) * blocks + block) * stride + std::size_t(geometry.secondary_offset);
                std::memset(words + live * sizeof(std::uint16_t), 0, (64 - live) * sizeof(std::uint16_t));
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
        /// After calling this, all accessor methods (interleavedB, chunkComp, etc.)
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
        // Accessors for the kernel inner loop
        // -------------------------------------------------------------------

#if defined(__AVX512F__)
        /**
         * @brief Widen sixteen ordinary inline FP16 scale words, without factors.
         * @param primary Thirty-two-byte-aligned half words from one inline scale plane.
         * @return The original sixteen half values widened to FP32.
         *
         * General source-factor formats still call loadScales512. Direct use
         * requires a prepared-type proof that the scale plane contains plain
         * halves: the unsigned-five-bit domain guarantees ordinary ExpandedInt8.
         * The original nibble dispatcher also proves plain NibbleLUT for its
         * Q4_0 linear and Q4_1/Q4_K identity decoder kinds. IQ4_XS remains LUT
         * with source factors; IQ3_S and rotated operands use expanded paths.
         * Neither asymmetry nor a model name supplies this layout proof.
         */
        [[nodiscard]] static inline __m512 loadPlainFP16Scales512(const uint16_t *primary)
        {
            return _mm512_cvtph_ps(_mm256_load_si256(
                reinterpret_cast<const __m256i *>(primary)));
        }

        /** @brief Load sixteen effective scales without rounding source products to half.
         * @param primary Aligned inline primary words, within one64-column unit.
         * @return Widened ordinary scale or exact IQ4_XS base times signed coefficient.
         */
        [[nodiscard]] inline __m512 loadScales512(const uint16_t *primary) const
        {
            const __m512 base = loadPlainFP16Scales512(primary);
            if (!usesNibbleLUTSourceScale() && !usesIQ3SExpandedSourceScale()) return base;
            const __m512i raw = _mm512_cvtepu16_epi32(_mm256_load_si256(
                reinterpret_cast<const __m256i *>(primary + 64)));
            if (usesIQ3SExpandedSourceScale())
            {
                if (_mm512_test_epi32_mask(raw, _mm512_set1_epi32(~15)) != 0)
                    throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
                return _mm512_mul_ps(base, _mm512_cvtepi32_ps(_mm512_add_epi32(
                    _mm512_slli_epi32(raw, 1), _mm512_set1_epi32(1))));
            }
            if (_mm512_test_epi32_mask(raw, _mm512_set1_epi32(~63)) != 0)
                throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
            return _mm512_mul_ps(base, _mm512_cvtepi32_ps(
                _mm512_sub_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(63)), _mm512_set1_epi32(32))));
        }
#endif
#if defined(__AVX2__)
        /** @brief Load eight effective scales using the same original source equation.
         * @param primary Aligned inline primary words, within one64-column unit.
         * @return Separately rounded FP32 scale vector, shared across grouped rows.
         */
        [[nodiscard]] inline __m256 loadScales256(const uint16_t *primary) const
        {
            const __m256 base = _mm256_cvtph_ps(_mm_load_si128(
                reinterpret_cast<const __m128i *>(primary)));
            if (!usesNibbleLUTSourceScale() && !usesIQ3SExpandedSourceScale()) return base;
            const __m256i raw = _mm256_cvtepu16_epi32(_mm_load_si128(
                reinterpret_cast<const __m128i *>(primary + 64)));
            if (usesIQ3SExpandedSourceScale())
            {
                if (!_mm256_testz_si256(raw, _mm256_set1_epi32(~15)))
                    throw std::invalid_argument("IQ3_S coefficient contains reserved bits");
                return _mm256_mul_ps(base, _mm256_cvtepi32_ps(_mm256_add_epi32(
                    _mm256_slli_epi32(raw, 1), _mm256_set1_epi32(1))));
            }
            if (!_mm256_testz_si256(raw, _mm256_set1_epi32(~63)))
                throw std::invalid_argument("IQ4_XS coefficient contains reserved bits");
            return _mm256_mul_ps(base, _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(63)), _mm256_set1_epi32(32))));
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
            const bool source_scale = usesNibbleLUTSourceScale() || usesIQ3SExpandedSourceScale() ||
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
            if (usesNibbleLUTSourceScale() || usesIQ3SExpandedSourceScale() ||
                (!is_asymmetric && !usesQ6KNativeDualScale() && !usesCompactMultiScale()))
                return 0.0f;
            return nativeQuantisedSecondaryScale(encoding, codebook_id,
                chunkScales(c, kb)[n_local], chunkMins(c, kb)[n_local]);
        }

        /// Scales pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkScales(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            const size_t compensation_bytes = usesInlineCompensation() ? 128u : 0u;
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + compensation_bytes);
        }

        /// Secondary words for one chunk: FP16 minimum/scale, or Q6_K/IQ2_S/IQ3_S/IQ4_XS raw coefficients.
        inline const uint16_t *chunkMins(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            const size_t metadata_offset =
                usesInlineCompensation() ? 256u : 128u;
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + metadata_offset);
        }

        /**
         * @brief Access the packed pair of Q2_K effective minima per output column.
         * @param c Output chunk of 64 columns.
         * @param kb Logical K block.
         * @return Sixty-four packed FP16 pairs.
         * @throws std::logic_error If this stream's section layout has no
         * effective-minimum section (presence derives from the layout tuple,
         * not the encoding family alone).
         */
        inline const uint32_t *chunkEffectiveMins(int c, int kb) const
        {
            const PreparedUnitSections sections = preparedSections();
            if (!sections.isValid() || !sections.has_emins)
                throw std::logic_error(
                    "Effective minima require a prepared layout with an effective-minimum section");
            const size_t offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const uint32_t *>(
                interleavedBase() + offset + sections.emins_offset);
        }

        /// Native-interleaved B data for N-chunk c, K-block kb, group g, ZMM z (0..3)
        /// Returns pointer to 64 bytes: 16 columns × 4 values each.
        /// For nibble-LUT: group ∈ [0,3], each holds 4 native payload bytes.
        /// For INT8 pre-decoded: group ∈ [0,7], each holds 4 INT8 values.
        inline const uint8_t *interleavedB(int c, int kb, int group, int z) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return interleavedBase() + block_offset + group * 256 + z * 64;
        }

        /// Pre-computed compensation for N-chunk c, K-block kb (contiguous 64 INT16, inline in native_interleaved)
        inline const int16_t *chunkComp(int c, int kb) const
        {
            if (!usesInlineCompensation())
                throw std::logic_error(
                    "Q6_K native dual-scale weights do not carry compensation metadata");
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const int16_t *>(
                interleavedBase() + block_offset + data_stride);
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
            const size_t block_offset =
                ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return interleavedBase() + block_offset + 1024u +
                   static_cast<size_t>(group) * 64u +
                   static_cast<size_t>(z) * 16u;
        }

        /// Flat INT8 values for N-chunk c, K-block kb, column n_local (32 INT8 values)
        /// Only valid for ExpandedInt8 preparation.
        /// Only available if keepDecodedBuffer was true during packing.
        inline const int8_t *blockInt8(int c, int kb, int n_local) const
        {
            size_t offset = ((size_t)c * blocks_per_row * 64 + (size_t)kb * 64 + n_local) * 32;
            return int8_flat.data() + offset;
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
        }

        /// Release the intermediate INT8 decode buffer to save memory.
        /// The AVX-512 GEMV hot path only uses native_interleaved, not int8_flat.
        /// After calling this, blockInt8() is invalid.
        void releaseDecodedBuffer()
        {
            int8_flat.clear();
            int8_flat.shrink_to_fit();
        }

        /// Release the permanent interleaved data to save memory.
        /// Used with deferred packing: after this, GEMM/GEMV must set workspace_data_
        /// before accessing interleaved accessors (interleavedB, chunkComp, etc.).
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

    /// Compute the workspace buffer size needed for repackNativeBlocksToInterleaved().
    inline size_t interleavedWorkspaceSize(const CPUNativeQuantisedPackedWeights &meta)
    {
        int N_chunks = meta.N_padded / 64;
        return (size_t)N_chunks * meta.blocks_per_row * meta.interleaved_block_stride;
    }

} // namespace llaminar2::cpu::native_quantised
