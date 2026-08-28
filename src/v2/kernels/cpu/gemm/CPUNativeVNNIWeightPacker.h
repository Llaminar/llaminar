/**
 * @file CPUNativeVNNIWeightPacker.h
 * @brief Repacks tensor weights from native quantized formats into a CPU-cache-friendly
 *        layout for NativeVNNI GEMV/GEMM with inline decode + vpdpbusd accumulation.
 *
 * ## Design
 *
 * Unlike the existing CPUQuantisedGemmKernel path (which pre-decodes ALL formats to INT8),
 * this packer preserves the native weight bytes (Q4_0 nibbles, IQ4_NL nibbles, etc.) and
 * reorganizes them for efficient streaming during GEMV/GEMM.
 *
 * The kernel decodes blocks inline at compute time, trading a small decode cost for
 * 2-4× less memory traffic on memory-bound GEMV (M=1).
 *
 * ## Packed Layout
 *
 * Weights are arranged in N-blocks of 64 columns for AVX-512 SIMD:
 *
 * ```
 * payload: [N_chunks][blocks_per_row][64][payload_bytes]
 *   - N_chunks = ceil(N/64)
 *   - blocks_per_row = ceil(K/32)
 *   - payload_bytes = format-specific (Q4_0=16, IQ4_NL=16, Q6_K=24, etc.)
 *
 * scales: [N_chunks][blocks_per_row][64] × FP32
 *   - One scale per 32-element block per column
 *
 * mins: [N_chunks][blocks_per_row][64] × FP32  (only for asymmetric formats)
 *   - One min offset per block per column
 * ```
 *
 * This layout ensures:
 * - Sequential K-block streaming within an N-chunk → good prefetch/cache line use
 * - 64 contiguous scales per K-block → aligned ZMM loads
 * - Native payload preserved → 2× less memory for Q4_0 vs INT8
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "CPUNativeVNNIDecode.h"
#include "CPUNativeVNNIPreparedFootprint.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorClasses.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_vnni
{
    /**
     * @brief Select the physical CPU execution encoding for a source codebook.
     *
     * @param codebook_id NativeVNNI execution codebook identifier.
     * @param rotation_enabled Whether preparation applies activation rotation;
     *        rotated values are requantized to the expanded INT8 encoding.
     * @return Physical encoding consumed by CPU GEMV/GEMM kernels.
     */
    [[nodiscard]] inline CPUNativeVNNIEncoding preparedEncodingForCodebook(
        uint8_t codebook_id,
        bool rotation_enabled = false) noexcept
    {
        if (rotation_enabled)
            return CPUNativeVNNIEncoding::ExpandedInt8;
        if (is_nibble_lut_format(codebook_id))
            return CPUNativeVNNIEncoding::NibbleLUT;
        if (codebook_id == 8)
            return CPUNativeVNNIEncoding::Q6KNativeDualScale;
        return CPUNativeVNNIEncoding::ExpandedInt8;
    }

    /**
     * @brief Build an exact cache footprint without materializing a matrix.
     *
     * This helper is used by planning/corpus tools that know format metadata
     * but intentionally do not allocate production-sized weights.
     */
    [[nodiscard]] inline NativeVNNIPreparedFootprint preparedFootprintForFormat(
        uint8_t codebook_id,
        bool is_asymmetric,
        bool rotation_enabled = false) noexcept
    {
        const CPUNativeVNNIEncoding encoding =
            preparedEncodingForCodebook(codebook_id, rotation_enabled);
        const bool prepared_is_asymmetric =
            rotation_enabled ? false : is_asymmetric;
        return NativeVNNIPreparedFootprint{
            .encoding = encoding,
            .is_asymmetric = prepared_is_asymmetric,
            .weight_bytes_per_n_chunk_k_block =
                static_cast<std::uint64_t>(preparedInterleavedBlockStride(
                    encoding, prepared_is_asymmetric)),
            .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
            .output_bytes_per_row_n_chunk = 64u * sizeof(float),
        };
    }

    /**
     * @brief CPU-packed native VNNI weights.
     *
     * All buffers are allocated and filled by packWeightsCPUNativeVNNI().
     *
     * ## Layout
     *
     * Weights are stored in two complementary layouts:
     *
     * 1. **payload**: Raw native bytes in [N_chunks][bpr][64][payload_bytes] layout
     *    Used by the scalar reference path.
     *
     * 2. **native_interleaved**: VNNI-interleaved bytes with inline metadata:
     *    `[N_chunks][bpr][ groups×4 ZMMs×64B | 128B comp | 128B scales | 128B mins ]`
     *    Each 64-byte ZMM holds 16 columns × 4 consecutive native payload bytes.
     *    Metadata (comp, scales, mins) is embedded at the end of each K-block
     *    so the entire working set is a single sequential memory stream per thread.
     *    This improves L3 cache utilization for large-K shapes like FFN_Down.
     */
    /**
     * @brief Return the raw source block size for a native-VNNI format.
     *
     * Several source formats intentionally share an execution codebook after
     * preparation, so codebook id alone is not sufficient.  The superblock bit
     * disambiguates IQ4_NL/IQ4_XS, Q4_1/Q4_K, and Q5_1/Q5_K.  This metadata is
     * used only by transferred/deferred prepared weights; eager kernels retain
     * their already interleaved representation.
     */
    inline size_t native_block_bytes_for_format(uint8_t codebook_id, bool is_superblock)
    {
        switch (codebook_id)
        {
        case 0:  return 18;  // Q4_0Block
        case 4:  return is_superblock ? 136 : 18; // IQ4_XSBlock / IQ4_NLBlock
        case 5:  return is_superblock ? 144 : 20; // Q4_KBlock / Q4_1Block
        case 6:  return 22;  // Q5_0Block
        case 7:  return is_superblock ? 176 : 24; // Q5_KBlock / Q5_1Block
        case 8:  return 210; // Q6_KBlock
        case 9:  return 110; // Q3_KBlock
        case 10: return 84;  // Q2_KBlock
        case 11: return 110; // IQ3_SBlock
        case 12: return 98;  // IQ3_XXSBlock
        case 13: return 82;  // IQ2_SBlock
        case 14: return 74;  // IQ2_XSBlock
        case 15: return 66;  // IQ2_XXSBlock
        case 16: return 50;  // IQ1_SBlock
        case 17: return 56;  // IQ1_MBlock
        case 19: return 34;  // Q8_0Block
        case 20: return 36;  // Q8_1Block
        case 21: return 288; // Q8_KBlock
        default: return 0;
        }
    }

    struct CPUNativeVNNIPackedWeights
    {
        /// Raw native payload bytes in [N_chunks][blocks_per_row][64][payload_bytes] layout
        /// Only populated for the legacy scalar nibble-format oracle.
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
        AlignedVector<uint8_t> native_interleaved;

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
        CPUNativeVNNIEncoding encoding = CPUNativeVNNIEncoding::ExpandedInt8;

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
            return encoding == CPUNativeVNNIEncoding::NibbleLUT;
        }

        /** @brief Return whether this matrix was expanded to signed INT8. */
        [[nodiscard]] bool usesExpandedInt8() const noexcept
        {
            return encoding == CPUNativeVNNIEncoding::ExpandedInt8;
        }

        /** @brief Return whether this matrix retains native dual-scale Q6_K. */
        [[nodiscard]] bool usesQ6KNativeDualScale() const noexcept
        {
            return encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale;
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
            return !usesQ6KNativeDualScale();
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
        [[nodiscard]] NativeVNNIPreparedFootprint preparedFootprint() const noexcept
        {
            return NativeVNNIPreparedFootprint{
                .encoding = encoding,
                .is_asymmetric = is_asymmetric,
                .weight_bytes_per_n_chunk_k_block =
                    static_cast<std::uint64_t>(interleaved_block_stride),
                .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
                .output_bytes_per_row_n_chunk = 64u * sizeof(float),
            };
        }

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

        /// Payload pointer for N-chunk c, K-block kb, column n_local (within chunk)
        inline const uint8_t *blockPayload(int c, int kb, int n_local) const
        {
            size_t offset = ((size_t)c * blocks_per_row * 64 + (size_t)kb * 64 + n_local) * payload_bytes;
            return payload.data() + offset;
        }

        /// Scale for N-chunk c, K-block kb, column n_local (returns FP32, converts from FP16)
        inline float blockScale(int c, int kb, int n_local) const
        {
            return fp16_to_fp32(chunkScales(c, kb)[n_local]);
        }

        /// Min for N-chunk c, K-block kb, column n_local (returns FP32, converts from FP16)
        inline float blockMin(int c, int kb, int n_local) const
        {
            return fp16_to_fp32(chunkMins(c, kb)[n_local]);
        }

        /// Scales pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkScales(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            const size_t compensation_bytes = usesInlineCompensation() ? 128u : 0u;
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + compensation_bytes);
        }

        /// Mins pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkMins(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            const size_t metadata_offset =
                usesInlineCompensation() ? 256u : 128u;
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + metadata_offset);
        }

        /// Payload pointer for N-chunk c, K-block kb (contiguous 64 × payload_bytes)
        inline const uint8_t *chunkPayload(int c, int kb) const
        {
            size_t offset = ((size_t)c * blocks_per_row + kb) * 64 * payload_bytes;
            return payload.data() + offset;
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
            { AlignedVector<uint8_t> empty; native_interleaved.swap(empty); }
            payload.clear();
            payload.shrink_to_fit();
        }

        /// Returns true if the interleaved data is available (either owned or via workspace).
        bool hasInterleavedData() const
        {
            return workspace_data_ != nullptr || !native_interleaved.empty();
        }
    };

    // =========================================================================
    // Deferred packing: repack native blocks → VNNI-interleaved workspace
    //
    // This function repacks native quantized blocks (Q4_0, IQ4_NL, Q8_0, etc.)
    // into the VNNI-interleaved format used by GEMM/GEMV. Unlike packWeightsCPUNativeVNNI()
    // which reads from a tensor, this operates on raw block bytes that were saved
    // from the original tensor during construction.
    //
    // For nibble-LUT formats (Q4_0, IQ4_NL, Q4_1): extracts payload nibbles + scales,
    // interleaves into 4-group VNNI layout with inline comp/scales.
    //
    // For INT8 pre-decoded formats (Q8_0, Q5_0, etc.): extracts INT8 values + scales,
    // interleaves into 8-group VNNI layout with inline comp/scales.
    //
    // Superblock formats (Q6_K, Q3_K, Q2_K) are NOT supported for deferred repacking
    // because they require the full superblock context for correct decode.
    // =========================================================================

    /**
     * @brief Repack native blocks into a pre-allocated VNNI-interleaved workspace.
     *
     * @param native_blocks  Raw native block bytes, row-major: [N × blocks_per_row × block_size]
     * @param block_size     Bytes per native block (e.g., 34 for Q8_0, 18 for Q4_0)
     * @param meta           Packed weights metadata (N, K, blocks_per_row, codebook_id, etc.)
     * @param workspace      Output buffer, must be at least N_chunks * blocks_per_row * interleaved_block_stride bytes
     */
    inline void repackNativeBlocksToInterleaved(
        const uint8_t *native_blocks,
        size_t block_size,
        const CPUNativeVNNIPackedWeights &meta,
        uint8_t *workspace)
    {
        if (meta.usesQ6KNativeDualScale())
        {
            throw std::invalid_argument(
                "Deferred native-block repacking does not own the Q6_K "
                "superblock context required by the native dual-scale layout");
        }
        const int N = meta.N;
        const int bpr = meta.blocks_per_row;
        const int N_chunks = (meta.N_padded) / 64;
        const uint8_t codebook_id = meta.codebook_id;

        // Row stride in native_blocks array (bytes between adjacent rows)
        const size_t native_row_stride = static_cast<size_t>(bpr) * block_size;

        if (meta.usesNibbleLUT())
        {
            // ---------------------------------------------------------------
            // NIBBLE-LUT PATH (Q4_0, IQ4_NL, Q4_1)
            // 4 groups, payload_bytes per block (16 for Q4_0/IQ4_NL)
            //
            // Vectorized interleave: AVX-512 gather from temp buffer
            // ---------------------------------------------------------------
            const int pb = meta.payload_bytes;
            // Payload offset within native block (after scale, optionally after min)
            const int payload_offset = (codebook_id == 5) ? 4 : 2; // Q4_1=4, others=2

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < bpr; ++kb)
                {
                    int n_cols = std::min(64, N - chunk * 64);

                    size_t block_offset = ((size_t)chunk * bpr + kb) * meta.interleaved_block_stride;
                    uint8_t *dst = workspace + block_offset;

                    // Extract payload, scales, mins from native blocks into contiguous temp arrays.
                    // Layout: payload_buf[col][pb] with col stride = pb.
                    alignas(64) uint8_t payload_buf[64 * 16]; // max pb=16
                    alignas(64) uint16_t scales_buf[64];
                    alignas(64) uint16_t mins_buf[64];

                    // Only zero tail if last chunk has partial columns
                    if (n_cols < 64)
                    {
                        std::memset(payload_buf, 0, 64 * pb);
                        std::memset(scales_buf, 0, sizeof(scales_buf));
                        std::memset(mins_buf, 0, sizeof(mins_buf));
                    }

                    for (int col = 0; col < n_cols; ++col)
                    {
                        int row = chunk * 64 + col;
                        const uint8_t *blk = native_blocks + row * native_row_stride + kb * block_size;

                        std::memcpy(&scales_buf[col], blk, 2);
                        if (codebook_id == 5)
                            std::memcpy(&mins_buf[col], blk + 2, 2);
                        std::memcpy(payload_buf + col * pb, blk + payload_offset, pb);
                    }

                    // --- Interleave payload into 4-group VNNI layout ---
                    for (int group = 0; group < 4; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    const uint8_t *p = payload_buf + col * pb;
                                    std::memcpy(zmm_dst + lane * 4, p + group * 4, 4);
                                }
                                else
                                {
                                    std::memset(zmm_dst + lane * 4, 0, 4);
                                }
                            }
                        }
                    }

                    // --- Compute comp (sum of decoded INT8 values per column) ---
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(dst + 1024);
                    for (int col = 0; col < n_cols; ++col)
                    {
                        int8_t decoded[32];
                        decode_native_block(codebook_id, payload_buf + col * pb, decoded);
#ifdef __AVX512F__
                        // Vectorized sum of 32 signed int8 → int16
                        __m256i v = _mm256_loadu_si256((const __m256i *)decoded);
                        __m256i ones = _mm256_set1_epi8(1);
                        __m256i pair_sums = _mm256_maddubs_epi16(ones, v); // 16 × int16
                        __m256i ones16 = _mm256_set1_epi16(1);
                        __m256i quad_sums = _mm256_madd_epi16(pair_sums, ones16); // 8 × int32
                        __m128i hi = _mm256_extracti128_si256(quad_sums, 1);
                        __m128i lo = _mm256_castsi256_si128(quad_sums);
                        __m128i sum128 = _mm_add_epi32(lo, hi);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        inline_comp[col] = static_cast<int16_t>(_mm_cvtsi128_si32(sum128));
#else
                        int32_t sum = 0;
                        for (int i = 0; i < 32; ++i)
                            sum += decoded[i];
                        inline_comp[col] = static_cast<int16_t>(sum);
#endif
                    }
                    for (int col = n_cols; col < 64; ++col)
                        inline_comp[col] = 0;

                    // Write scales inline
                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(dst + 1024 + 128);
                    std::memcpy(inline_scales, scales_buf, 64 * sizeof(uint16_t));

                    // Write mins inline (if asymmetric)
                    if (meta.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(dst + 1024 + 256);
                        std::memcpy(inline_mins, mins_buf, 64 * sizeof(uint16_t));
                    }
                }
            }
        }
        else
        {
            // ---------------------------------------------------------------
            // INT8 PRE-DECODED PATH (Q8_0, Q5_0, Q5_1)
            // 8 groups, 32 INT8 values per block
            // ---------------------------------------------------------------

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < bpr; ++kb)
                {
                    const int n_cols = std::min(64, N - chunk * 64);
                    const size_t block_offset_ws = ((size_t)chunk * bpr + kb) * meta.interleaved_block_stride;
                    uint8_t *dst = workspace + block_offset_ws;

                    // Extract INT8 values, scales, and mins from native blocks.
                    // int8_buf layout: [col][32] with stride 32 between columns.
                    alignas(64) int8_t int8_buf[64][32];
                    alignas(64) uint16_t scales_buf[64];
                    alignas(64) uint16_t mins_buf[64];
                    alignas(64) int16_t comp_buf[64];

                    // Only zero tail if last chunk has partial columns
                    if (n_cols < 64)
                    {
                        std::memset(int8_buf, 0, sizeof(int8_buf));
                        std::memset(scales_buf, 0, sizeof(scales_buf));
                        std::memset(mins_buf, 0, sizeof(mins_buf));
                        std::memset(comp_buf, 0, sizeof(comp_buf));
                    }

                    // Extract blocks + compute comp in a single pass
                    for (int col = 0; col < n_cols; ++col)
                    {
                        const int row = chunk * 64 + col;
                        const uint8_t *blk = native_blocks + row * native_row_stride + kb * block_size;

                        if (codebook_id == 19) // Q8_0: scale(2) + qs(32) = 34 bytes
                        {
                            std::memcpy(&scales_buf[col], blk, 2);
                            std::memcpy(int8_buf[col], blk + 2, 32);
                        }
                        else if (codebook_id == 6) // Q5_0
                        {
                            std::memcpy(&scales_buf[col], blk, 2);
                            Q5_0Block block;
                            block.d = 0;
                            std::memcpy(block.qh, blk + 2, 4);
                            std::memcpy(block.qs, blk + 6, 16);
                            simd::unpack_q5_0_to_int8(block, int8_buf[col]);
                        }
                        else if (codebook_id == 7) // Q5_1
                        {
                            std::memcpy(&scales_buf[col], blk, 2);
                            std::memcpy(&mins_buf[col], blk + 2, 2);
                            Q5_1Block block;
                            block.d = 0;
                            block.m = 0;
                            std::memcpy(block.qh, blk + 4, 4);
                            std::memcpy(block.qs, blk + 8, 16);
                            simd::unpack_q5_1_to_int8(block, int8_buf[col]);
                        }

                        // Comp: vectorized sum of 32 signed int8
#ifdef __AVX512F__
                        __m256i v = _mm256_loadu_si256((const __m256i *)int8_buf[col]);
                        __m256i ones = _mm256_set1_epi8(1);
                        __m256i pair_sums = _mm256_maddubs_epi16(ones, v);
                        __m256i ones16 = _mm256_set1_epi16(1);
                        __m256i quad_sums = _mm256_madd_epi16(pair_sums, ones16);
                        __m128i hi = _mm256_extracti128_si256(quad_sums, 1);
                        __m128i lo = _mm256_castsi256_si128(quad_sums);
                        __m128i sum128 = _mm_add_epi32(lo, hi);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        comp_buf[col] = static_cast<int16_t>(_mm_cvtsi128_si32(sum128));
#else
                        int32_t sum = 0;
                        for (int i = 0; i < 32; ++i)
                            sum += int8_buf[col][i];
                        comp_buf[col] = static_cast<int16_t>(sum);
#endif
                    }

                    // --- Interleave INT8 values into 8-group VNNI layout ---
                    for (int group = 0; group < 8; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                const int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    std::memcpy(zmm_dst + lane * 4,
                                                int8_buf[col] + group * 4, 4);
                                }
                                else
                                {
                                    std::memset(zmm_dst + lane * 4, 0, 4);
                                }
                            }
                        }
                    }

                    // Write comp + scales
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(dst + meta.data_stride);
                    std::memcpy(inline_comp, comp_buf, 64 * sizeof(int16_t));

                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(
                        dst + meta.data_stride + 128);
                    std::memcpy(inline_scales, scales_buf, 64 * sizeof(uint16_t));

                    if (meta.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(
                            dst + meta.data_stride + 256);
                        std::memcpy(inline_mins, mins_buf, 64 * sizeof(uint16_t));
                    }
                }
            }
        }
    }

    /// Compute the workspace buffer size needed for repackNativeBlocksToInterleaved().
    inline size_t interleavedWorkspaceSize(const CPUNativeVNNIPackedWeights &meta)
    {
        int N_chunks = meta.N_padded / 64;
        return (size_t)N_chunks * meta.blocks_per_row * meta.interleaved_block_stride;
    }

    /**
     * @brief Decode Expanded-INT8 source blocks directly into final VNNI bytes.
     *
     * The historical preparation path first materialized an `N x K` INT8
     * matrix and three matrix-sized metadata arrays, then reread those
     * temporaries to construct `native_interleaved`.  ExpertOverlay prepares
     * thousands of independently owned projections, so that extra traffic and
     * transient resident set materially increased both cold-start time and
     * allocator pressure.
     *
     * This pass owns one `(64 output columns, up to eight K blocks)` tile per
     * OpenMP iteration.  A tile reads every source block once and writes every
     * byte of its disjoint final destination.  Group bytes, compensation,
     * FP16 scales, and optional FP16 minima retain exactly the same arithmetic
     * and byte order as the former two-pass implementation.  Eight-block tiles
     * preserve the format-specific superblock decoder, so K-quant and IQuant
     * codebooks do not regress to eight redundant header decodes.
     *
     * @param unpackable Source-codebook decode authority.
     * @param out Fully initialized Expanded-INT8 layout metadata and final
     *        storage owner.
     * @param row_start First source row represented by destination row zero.
     * @param N Number of source rows to pack.
     * @param blocks_per_row Number of logical 32-value K blocks per row.
     * @param N_chunks Number of 64-column destination chunks.
     * @param use_superblock Whether one source decoder call produces eight
     *        consecutive logical K blocks.
     */
    inline void packExpandedInt8Direct(
        const IINT8Unpackable &unpackable,
        CPUNativeVNNIPackedWeights &out,
        int row_start,
        int N,
        int blocks_per_row,
        int N_chunks,
        bool use_superblock)
    {
        if (!out.usesExpandedInt8() || out.data_stride != 2048)
        {
            throw std::invalid_argument(
                "Direct Expanded-INT8 packing requires the complete 2048-byte data layout");
        }

        const size_t interleaved_total =
            static_cast<size_t>(N_chunks) * blocks_per_row *
            static_cast<size_t>(out.interleaved_block_stride);
        out.native_interleaved.resize_uninitialized(interleaved_total);

        constexpr int kBlocksPerTile = 8;
        constexpr int kColumnsPerChunk = 64;
        constexpr int kValuesPerBlock = 32;
        constexpr int kValuesPerGroup = 4;
        constexpr int kGroups = kValuesPerBlock / kValuesPerGroup;
        const int block_tiles =
            (blocks_per_row + kBlocksPerTile - 1) / kBlocksPerTile;

#pragma omp parallel for schedule(static) collapse(2)
        for (int chunk = 0; chunk < N_chunks; ++chunk)
        {
            for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
            {
                const int first_kb = block_tile * kBlocksPerTile;
                const int block_count = std::min(
                    kBlocksPerTile, blocks_per_row - first_kb);
                const int valid_columns = std::min(
                    kColumnsPerChunk, N - chunk * kColumnsPerChunk);

                // One stack tile is reused for each column.  It is small
                // enough to remain in L1 while the eight final blocks are hot.
                alignas(64) int8_t decoded[
                    kBlocksPerTile * kValuesPerBlock]{};
                alignas(64) float scales[kBlocksPerTile]{};
                alignas(64) float mins[kBlocksPerTile]{};

                for (int column = 0; column < kColumnsPerChunk; ++column)
                {
                    const bool source_column_valid = column < valid_columns;
                    if (source_column_valid)
                    {
                        const int source_row =
                            row_start + chunk * kColumnsPerChunk + column;
                        if (use_superblock &&
                            block_count == kBlocksPerTile)
                        {
                            // One call decodes the source header and all eight
                            // logical blocks.  Scale/min conversion remains
                            // below so it follows the historical FP32->FP16 edge.
                            unpackable.unpack_superblock_to_int8(
                                static_cast<size_t>(source_row),
                                static_cast<size_t>(block_tile),
                                decoded,
                                scales,
                                mins);
                        }
                        else
                        {
                            for (int local_kb = 0;
                                 local_kb < block_count;
                                 ++local_kb)
                            {
                                const int kb = first_kb + local_kb;
                                unpackable.unpack_block_to_int8(
                                    static_cast<size_t>(source_row),
                                    static_cast<size_t>(kb),
                                    decoded + local_kb * kValuesPerBlock);
                                scales[local_kb] =
                                    unpackable.get_block_scale(
                                        static_cast<size_t>(source_row),
                                        static_cast<size_t>(kb));
                                mins[local_kb] =
                                    unpackable.get_block_min(
                                        static_cast<size_t>(source_row),
                                        static_cast<size_t>(kb));
                            }
                        }
                    }

                    for (int local_kb = 0;
                         local_kb < block_count;
                         ++local_kb)
                    {
                        const int kb = first_kb + local_kb;
                        const size_t block_offset =
                            (static_cast<size_t>(chunk) * blocks_per_row +
                             static_cast<size_t>(kb)) *
                            static_cast<size_t>(out.interleaved_block_stride);
                        uint8_t *const destination =
                            out.native_interleaved.data() + block_offset;

                        const int zmm = column / 16;
                        const int lane = column % 16;
                        const int8_t *const values =
                            decoded + local_kb * kValuesPerBlock;
                        for (int group = 0; group < kGroups; ++group)
                        {
                            uint8_t *const group_destination =
                                destination + group * 256 + zmm * 64 +
                                lane * kValuesPerGroup;
                            if (source_column_valid)
                            {
                                std::memcpy(
                                    group_destination,
                                    values + group * kValuesPerGroup,
                                    kValuesPerGroup);
                            }
                            else
                            {
                                std::memset(
                                    group_destination,
                                    0,
                                    kValuesPerGroup);
                            }
                        }

                        auto *const compensation =
                            reinterpret_cast<int16_t *>(
                                destination + out.data_stride);
                        auto *const inline_scales =
                            reinterpret_cast<uint16_t *>(
                                destination + out.data_stride + 128);
                        if (source_column_valid)
                        {
                            int32_t sum = 0;
                            // Fixed scalar order is the byte-deterministic
                            // preparation contract shared by all codebooks.
                            for (int value = 0;
                                 value < kValuesPerBlock;
                                 ++value)
                            {
                                sum += values[value];
                            }
                            compensation[column] =
                                static_cast<int16_t>(sum);
                            inline_scales[column] =
                                fp32_to_fp16(scales[local_kb]);
                        }
                        else
                        {
                            compensation[column] = 0;
                            inline_scales[column] = 0;
                        }

                        if (out.is_asymmetric)
                        {
                            auto *const inline_mins =
                                reinterpret_cast<uint16_t *>(
                                    destination + out.data_stride + 256);
                            inline_mins[column] = source_column_valid
                                                      ? fp32_to_fp16(
                                                            mins[local_kb])
                                                      : 0;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Pack tensor weights into CPUNativeVNNIPackedWeights.
     *
     * Uses the IINT8Unpackable interface to extract native block data
     * from any supported quantized tensor format.
     *
     * @param weights     Source tensor (must implement IINT8Unpackable + vnniFormatInfo)
     * @param out         Output packed weights
     * @param row_start   First row to pack (for TP slicing), default 0
     * @param row_end     One-past-last row, default -1 (all rows)
     * @return true on success
     */
    inline bool packWeightsCPUNativeVNNI(const TensorBase *weights,
                                         CPUNativeVNNIPackedWeights &out,
                                         int row_start = 0, int row_end = -1,
                                         const ActivationRotation *rotation = nullptr)
    {
        // Validate
        int full_N = weights->shape()[0];
        int K = weights->shape()[1];
        if (row_end < 0)
            row_end = full_N;
        if (row_start < 0)
            row_start = 0;
        if (row_end > full_N)
            row_end = full_N;
        if (row_start >= row_end)
        {
            LOG_ERROR("[CPUNativeVNNI] Invalid row range: [" << row_start << ", " << row_end << ")");
            return false;
        }

        // Get IINT8Unpackable interface
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        if (!unpackable)
        {
            LOG_ERROR("[CPUNativeVNNI] Tensor does not implement IINT8Unpackable");
            return false;
        }

        // Get native format info
        const NativeVnniFormatInfo *fmt = unpackable->vnniFormatInfo();
        if (!fmt)
        {
            LOG_ERROR("[CPUNativeVNNI] Tensor does not provide vnniFormatInfo (8-bit formats not supported)");
            return false;
        }

        int N = row_end - row_start;
        int N_padded = (N + 63) / 64 * 64;
        int blocks_per_row = (K + 31) / 32;
        int N_chunks = N_padded / 64;

        out.N = N;
        out.K = K;
        out.N_padded = N_padded;
        out.blocks_per_row = blocks_per_row;
        out.payload_bytes = fmt->payload_bytes;
        out.codebook_id = fmt->codebook_id;
        out.is_asymmetric = fmt->is_asymmetric;
        out.is_superblock = fmt->is_superblock;

        // When rotation is active, rotation mixes values across the 32-element
        // quantization block boundaries, so we must dequant → rotate → requant.
        // The result is always INT8 pre-decoded format (8 groups), regardless of
        // the original tensor format. We still preserve the original codebook_id
        // for diagnostic purposes, but the packed layout uses the INT8 path.
        const bool use_rotated_path = (rotation != nullptr);
        if (use_rotated_path)
            out.is_asymmetric = false;
        out.encoding = preparedEncodingForCodebook(
            fmt->codebook_id, use_rotated_path);

        switch (out.encoding)
        {
        case CPUNativeVNNIEncoding::NibbleLUT:
            out.data_stride = 1024;
            break;
        case CPUNativeVNNIEncoding::ExpandedInt8:
            out.data_stride = 2048;
            break;
        case CPUNativeVNNIEncoding::Q6KNativeDualScale:
            out.data_stride = 1536;
            break;
        }
        out.interleaved_block_stride = preparedInterleavedBlockStride(
            out.encoding, out.is_asymmetric);

        // Temporary metadata is allocated only by layouts that still need a
        // distinct source-oriented pass.  Expanded-INT8 preparation writes
        // directly to final storage and therefore owns no matrix-sized shadow.
        size_t total_blocks = (size_t)N_chunks * blocks_per_row * 64;
        std::vector<uint16_t> temp_scales;
        std::vector<uint16_t> temp_mins;

        bool use_superblock = (unpackable->superblock_size() == 256);

        if (use_rotated_path)
        {
            // =================================================================
            // ROTATION PATH (any format → rotated INT8)
            //
            // Dequantize each row to FP32, apply FWHT rotation, requantize to
            // INT8 with per-32-element block scales. This fuses rotation into
            // the one-time packing step so the original tensor is never modified.
            //
            // The rotation mixes values across the K dimension, so we must work
            // on full rows (not individual blocks). After rotation, values are
            // requantized with symmetric per-block scales and stored in the
            // same INT8 pre-decoded layout used by Q5_0/Q6_K/etc.
            //
            // Asymmetric formats become symmetric after rotation (the rotation
            // distributes outliers evenly, centering the distribution).
            // =================================================================

            temp_scales.assign(total_blocks, 0);
            size_t int8_total = (size_t)N_chunks * blocks_per_row * 64 * 32;
            out.int8_flat.resize(int8_total, 0);

            LOG_DEBUG("[CPUNativeVNNI] Rotation packing: dequant→rotate→requant "
                      << N << "×" << K << " (block_dim=" << rotation->block_dim() << ")");

#pragma omp parallel
            {
                // Per-thread scratch for full-row FP32 dequantization + rotation
                std::vector<float> row_fp32(K);

#pragma omp for schedule(static)
                for (int n = 0; n < N; ++n)
                {
                    int src_row = row_start + n;
                    int chunk = n / 64;
                    int n_local = n % 64;

                    // Step 1: Dequantize full row to FP32
                    weights->to_fp32_row(src_row, row_fp32.data());

                    // Step 2: Apply FWHT rotation in-place
                    rotation->rotate_inplace(row_fp32.data(), K);

                    // Step 3: Requantize to INT8 with per-32-element block scales
                    for (int kb = 0; kb < blocks_per_row; ++kb)
                    {
                        const float *block_start = row_fp32.data() + kb * 32;
                        int block_len = std::min(32, K - kb * 32);

                        // Find absmax for symmetric quantization
                        float amax = 0.0f;
#if defined(__AVX512F__)
                        if (block_len == 32)
                        {
                            __m512 vmax = _mm512_setzero_ps();
                            __m512 v0 = _mm512_loadu_ps(block_start);
                            __m512 v1 = _mm512_loadu_ps(block_start + 16);
                            // abs via AND with sign-bit mask
                            const __m512 sign_mask = _mm512_castsi512_ps(
                                _mm512_set1_epi32(0x7FFFFFFF));
                            v0 = _mm512_and_ps(v0, sign_mask);
                            v1 = _mm512_and_ps(v1, sign_mask);
                            vmax = _mm512_max_ps(v0, v1);
                            amax = _mm512_reduce_max_ps(vmax);
                        }
                        else
#endif
                        {
                            for (int i = 0; i < block_len; ++i)
                            {
                                float a = std::fabs(block_start[i]);
                                if (a > amax)
                                    amax = a;
                            }
                        }

                        float scale = amax / 127.0f;
                        float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;

                        size_t idx = (size_t)chunk * blocks_per_row * 64 +
                                     (size_t)kb * 64 + n_local;
                        temp_scales[idx] = fp32_to_fp16(scale);

                        // Quantize to int8
                        size_t flat_offset = idx * 32;
                        int8_t *dst = out.int8_flat.data() + flat_offset;

#if defined(__AVX512F__)
                        if (block_len == 32)
                        {
                            __m512 vinv = _mm512_set1_ps(inv_scale);
                            __m512 v0 = _mm512_loadu_ps(block_start);
                            __m512 v1 = _mm512_loadu_ps(block_start + 16);
                            v0 = _mm512_mul_ps(v0, vinv);
                            v1 = _mm512_mul_ps(v1, vinv);
                            // Round to nearest integer (ROUNDSCALE_RND_MODE=0 = round to nearest even)
                            __m512i i0 = _mm512_cvtps_epi32(v0);
                            __m512i i1 = _mm512_cvtps_epi32(v1);
                            // Pack 32-bit → 16-bit → 8-bit
                            __m512i packed16 = _mm512_packs_epi32(i0, i1);
                            // packs interleaves: need to fix lane order
                            // After packs_epi32: [a0..a7,b0..b7 | a8..a15,b8..b15 | ...]
                            // We need sequential order
                            __m512i packed8 = _mm512_packs_epi16(packed16, _mm512_setzero_si512());
                            // Extract lower 32 bytes (the int8 values are in the lower half of each 128-bit lane)
                            // Use vpermq to gather them
                            // After packs_epi16 with zero: each 128-bit lane has 8 valid bytes + 8 zeros
                            // Lane 0: i0[0..3],i1[0..3], 0,0,0,0,0,0,0,0
                            // Lane 1: i0[4..7],i1[4..7], 0,0,0,0,0,0,0,0
                            // etc.
                            // Simpler: just use scalar store after cvt
                            // Actually, let me use the straightforward approach with _mm512_cvtsepi32_epi8
                            __m128i bytes0 = _mm512_cvtsepi32_epi8(i0);  // 16 int8 values from i0
                            __m128i bytes1 = _mm512_cvtsepi32_epi8(i1);  // 16 int8 values from i1
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), bytes0);
                            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), bytes1);
                        }
                        else
#endif
                        {
                            for (int i = 0; i < block_len; ++i)
                            {
                                int v = static_cast<int>(std::round(block_start[i] * inv_scale));
                                dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
                            }
                            // Zero-fill tail
                            for (int i = block_len; i < 32; ++i)
                                dst[i] = 0;
                        }
                    }
                }
            }

            // Fall through to the INT8 interleaving path below
        }
        else if (out.usesQ6KNativeDualScale())
        {
            // =================================================================
            // NATIVE Q6_K DUAL-SCALE PATH
            //
            // Preserve the exact 24-byte payload emitted for each logical
            // 32-value block.  The first sixteen bytes carry paired low
            // nibbles and the final eight bytes carry four two-bit high parts
            // each.  The two FP16 metadata arrays are independent scales for
            // values [0,16) and [16,32); they are not scale/min correction.
            //
            // Prepared bytes per 64-column/K block:
            //   1024 B low-nibble transpose
            //    512 B high-two-bit transpose
            //    128 B low-half scales
            //    128 B high-half scales
            // = 1792 B, versus 2304 B for the previous requantized INT8 path.
            // =================================================================
            std::vector<uint8_t> temp_payload(
                static_cast<size_t>(blocks_per_row) * N_padded *
                    fmt->payload_bytes,
                0);
            std::vector<uint16_t> temp_primary_scales(
                static_cast<size_t>(blocks_per_row) * N_padded,
                0);
            std::vector<uint16_t> temp_secondary_scales(
                static_cast<size_t>(blocks_per_row) * N_padded,
                0);

            VnniPackContext native_context;
            native_context.raw_bytes = nullptr;
            native_context.N = N_padded;
            native_context.K = K;
            native_context.blocks_per_row = blocks_per_row;
            native_context.payload_bytes = fmt->payload_bytes;
            native_context.payload_array = temp_payload.data();
            native_context.scales_array = temp_primary_scales.data();
            native_context.mins_array = temp_secondary_scales.data();
            native_context.emins_array = nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const int src_row = row_start + n;
                for (int kb = 0; kb < blocks_per_row; ++kb)
                    unpackable->packVnniBlock(native_context, src_row, n, kb);
            }

            const size_t interleaved_total =
                static_cast<size_t>(N_chunks) * blocks_per_row *
                out.interleaved_block_stride;
            out.native_interleaved.resize_uninitialized(interleaved_total);

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    const int n_cols = std::min(64, N - chunk * 64);
                    const size_t block_offset =
                        (static_cast<size_t>(chunk) * blocks_per_row + kb) *
                        out.interleaved_block_stride;
                    uint8_t *const destination =
                        out.native_interleaved.data() + block_offset;

                    // Low nibbles retain the established four-group/ZMM layout.
                    for (int group = 0; group < 4; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *const zmm_destination =
                                destination + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                const int local_column = z * 16 + lane;
                                if (local_column >= n_cols)
                                {
                                    std::memset(zmm_destination + lane * 4, 0, 4);
                                    continue;
                                }
                                const int column = chunk * 64 + local_column;
                                const size_t linear =
                                    static_cast<size_t>(kb) * N_padded + column;
                                const uint8_t *const source =
                                    temp_payload.data() +
                                    linear * fmt->payload_bytes + group * 4;
                                std::memcpy(zmm_destination + lane * 4, source, 4);
                            }
                        }
                    }

                    /*
                     * Transpose the native two-bit fields into two bitplanes
                     * for each group/ZMM. The low-nibble vector is ordered as
                     * sixteen columns of four adjacent K values, so mask bit
                     * `lane * 4 + index` directly addresses its destination
                     * byte. Invalid padded columns remain clear.
                     */
                    for (int group = 0; group < 8; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint64_t low_bitplane = 0;
                            uint64_t high_bitplane = 0;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                const int local_column = z * 16 + lane;
                                if (local_column >= n_cols)
                                    continue;

                                const int column =
                                    chunk * 64 + local_column;
                                const size_t linear =
                                    static_cast<size_t>(kb) * N_padded +
                                    column;
                                const uint8_t packed_high =
                                    temp_payload[
                                        linear * fmt->payload_bytes +
                                        16 + group];
                                for (int index = 0; index < 4; ++index)
                                {
                                    const uint64_t bit =
                                        static_cast<uint64_t>(lane * 4 + index);
                                    const uint8_t high_part =
                                        static_cast<uint8_t>(
                                            (packed_high >> (index * 2)) &
                                            0x03u);
                                    low_bitplane |=
                                        static_cast<uint64_t>(high_part & 1u)
                                        << bit;
                                    high_bitplane |=
                                        static_cast<uint64_t>(high_part >> 1)
                                        << bit;
                                }
                            }

                            uint8_t *const high_destination =
                                destination + 1024 + group * 64 + z * 16;
                            std::memcpy(
                                high_destination,
                                &low_bitplane,
                                sizeof(low_bitplane));
                            std::memcpy(
                                high_destination + sizeof(low_bitplane),
                                &high_bitplane,
                                sizeof(high_bitplane));
                        }
                    }

                    auto *const primary_scales =
                        reinterpret_cast<uint16_t *>(destination + 1536);
                    auto *const secondary_scales =
                        reinterpret_cast<uint16_t *>(destination + 1664);
                    for (int local_column = 0; local_column < 64;
                         ++local_column)
                    {
                        if (local_column >= n_cols)
                        {
                            primary_scales[local_column] = 0;
                            secondary_scales[local_column] = 0;
                            continue;
                        }
                        const int column = chunk * 64 + local_column;
                        const size_t linear =
                            static_cast<size_t>(kb) * N_padded + column;
                        primary_scales[local_column] =
                            temp_primary_scales[linear];
                        secondary_scales[local_column] =
                            temp_secondary_scales[linear];
                    }
                }
            }
        }
        else if (out.usesNibbleLUT())
        {
            // =================================================================
            // NIBBLE-LUT PATH (Q4_0, IQ4_NL, Q4_1, IQ4_XS)
            //
            // Store raw native payload bytes + VNNI-interleaved native bytes.
            // The GEMV kernel decodes nibbles at runtime via vpshufb LUT.
            // Memory: 0.5 byte/element (half of INT8).
            // =================================================================

            temp_scales.assign(total_blocks, 0);
            temp_mins.assign(total_blocks, 0);
            out.payload.resize(total_blocks * fmt->payload_bytes, 0);

// Pass 1: Extract scales/mins from superblocks
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                int src_row = row_start + n;
                int chunk = n / 64;
                int n_local = n % 64;
                int kb = 0;

                if (use_superblock)
                {
                    int K_superblocks = blocks_per_row / 8;
                    for (int sb = 0; sb < K_superblocks; ++sb)
                    {
                        int8_t sb_vals[256];
                        float sb_scales[8];
                        float sb_mins[8];
                        unpackable->unpack_superblock_to_int8(src_row, sb, sb_vals, sb_scales, sb_mins);

                        for (int i = 0; i < 8; ++i)
                        {
                            size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)(kb + i) * 64 + n_local;
                            temp_scales[idx] = fp32_to_fp16(sb_scales[i]);
                            temp_mins[idx] = fp32_to_fp16(sb_mins[i]);
                        }
                        kb += 8;
                    }
                }
                for (; kb < blocks_per_row; ++kb)
                {
                    size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                    temp_scales[idx] = fp32_to_fp16(unpackable->get_block_scale(src_row, kb));
                    temp_mins[idx] = fp32_to_fp16(unpackable->get_block_min(src_row, kb));
                }
            }

            // Pass 2: Pack native payload via packVnniBlock into temp GPU-interleaved buffer
            std::vector<uint8_t> temp_payload((size_t)blocks_per_row * N_padded * fmt->payload_bytes, 0);
            std::vector<uint16_t> temp_scales2((size_t)blocks_per_row * N_padded, 0);
            std::vector<uint16_t> temp_mins2((size_t)blocks_per_row * N_padded, 0);

            VnniPackContext gpu_ctx;
            gpu_ctx.raw_bytes = nullptr;
            gpu_ctx.N = N_padded;
            gpu_ctx.K = K;
            gpu_ctx.blocks_per_row = blocks_per_row;
            gpu_ctx.payload_bytes = fmt->payload_bytes;
            gpu_ctx.payload_array = temp_payload.data();
            gpu_ctx.scales_array = temp_scales2.data();
            gpu_ctx.mins_array = fmt->is_asymmetric ? temp_mins2.data() : nullptr;
            gpu_ctx.emins_array = nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const int src_row = row_start + n;
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    unpackable->packVnniBlock(gpu_ctx, src_row, n, kb);
                }
            }

            // Pass 3: Reorganize from GPU interleaved to CPU blocked layout
#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    for (int n_local = 0; n_local < 64; ++n_local)
                    {
                        int n = chunk * 64 + n_local;
                        if (n >= N) continue;
                        size_t gpu_idx = (size_t)kb * N_padded + n;
                        size_t cpu_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                        std::memcpy(out.payload.data() + cpu_idx * fmt->payload_bytes,
                                    temp_payload.data() + gpu_idx * fmt->payload_bytes,
                                    fmt->payload_bytes);
                    }
                }
            }

            // Pass 4: Build VNNI-interleaved B buffer (4 groups) + inline comp/scales/mins
            size_t interleaved_total = (size_t)N_chunks * blocks_per_row * out.interleaved_block_stride;
            // The following parallel pass writes every byte. Avoid a
            // single-threaded zero fill here because it first-touches the whole
            // permanent weight buffer on the allocator thread's NUMA node.
            out.native_interleaved.resize_uninitialized(interleaved_total);

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    const uint8_t *chunk_payload = out.chunkPayload(chunk, kb);
                    int n_cols = std::min(64, N - chunk * 64);

                    size_t block_offset = ((size_t)chunk * blocks_per_row + kb) * out.interleaved_block_stride;
                    uint8_t *interleaved_dst = out.native_interleaved.data() + block_offset;

                    for (int group = 0; group < 4; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = interleaved_dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    const uint8_t *src = chunk_payload + (size_t)col * fmt->payload_bytes;
                                    zmm_dst[lane * 4 + 0] = src[group * 4 + 0];
                                    zmm_dst[lane * 4 + 1] = src[group * 4 + 1];
                                    zmm_dst[lane * 4 + 2] = src[group * 4 + 2];
                                    zmm_dst[lane * 4 + 3] = src[group * 4 + 3];
                                }
                                else
                                {
                                    zmm_dst[lane * 4 + 0] = 0;
                                    zmm_dst[lane * 4 + 1] = 0;
                                    zmm_dst[lane * 4 + 2] = 0;
                                    zmm_dst[lane * 4 + 3] = 0;
                                }
                            }
                        }
                    }

                    // Write comp inline after group data
                    size_t meta_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64;
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(interleaved_dst + 1024);
                    for (int col = 0; col < 64; ++col)
                    {
                        if (col < n_cols)
                        {
                            int8_t decoded[32];
                            decode_native_block(out.codebook_id,
                                                chunk_payload + (size_t)col * fmt->payload_bytes,
                                                decoded);
                            int32_t sum = 0;
                            for (int i = 0; i < 32; ++i)
                                sum += decoded[i];
                            inline_comp[col] = static_cast<int16_t>(sum);
                        }
                        else
                        {
                            inline_comp[col] = 0;
                        }
                    }

                    // Write scales inline after comp
                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(interleaved_dst + 1024 + 128);
                    std::memcpy(inline_scales, &temp_scales[meta_idx], 64 * sizeof(uint16_t));

                    // Write mins inline after scales (if asymmetric)
                    if (out.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(interleaved_dst + 1024 + 256);
                        std::memcpy(inline_mins, &temp_mins[meta_idx], 64 * sizeof(uint16_t));
                    }
                }
            }
        }
        else
        {
            // =================================================================
            // INT8 PRE-DECODED PATH (Q5_0, Q5_1, Q6_K, Q3_K, Q2_K, IQ2/3/1*)
            //
            // Decode directly into the final eight-group VNNI layout.  The
            // GEMV kernel loads these pre-decoded INT8 values directly.
            // Memory: 1.0 byte/element, with no transient full INT8 matrix.
            // =================================================================
            packExpandedInt8Direct(
                *unpackable,
                out,
                row_start,
                N,
                blocks_per_row,
                N_chunks,
                use_superblock);
        }

        // =================================================================
        // INT8 INTERLEAVING (shared by rotation path and INT8 pre-decoded path)
        //
        // Build VNNI-interleaved INT8 buffer (8 groups) + inline comp/scales/mins.
        // Rotation still needs a complete row before its FWHT.  It is the only
        // Expanded-INT8 producer that retains a temporary decoded matrix.
        // =================================================================
        if (use_rotated_path)
        {
            size_t interleaved_total = (size_t)N_chunks * blocks_per_row * out.interleaved_block_stride;
            // The following parallel pass writes every byte. Avoid a
            // single-threaded zero fill here because it first-touches the whole
            // permanent weight buffer on the allocator thread's NUMA node.
            out.native_interleaved.resize_uninitialized(interleaved_total);

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    int n_cols = std::min(64, N - chunk * 64);

                    size_t block_offset = ((size_t)chunk * blocks_per_row + kb) * out.interleaved_block_stride;
                    uint8_t *interleaved_dst = out.native_interleaved.data() + block_offset;

                    // Interleave: [8 groups][4 ZMMs][64 bytes]
                    // Group g covers INT8[g*4..g*4+3] for each column
                    for (int group = 0; group < 8; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = interleaved_dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    size_t flat_idx = ((size_t)chunk * blocks_per_row * 64 +
                                                       (size_t)kb * 64 + col) * 32;
                                    const int8_t *vals = out.int8_flat.data() + flat_idx;
                                    // Cast to uint8_t for storage; vpdpbusd interprets B as signed
                                    zmm_dst[lane * 4 + 0] = static_cast<uint8_t>(vals[group * 4 + 0]);
                                    zmm_dst[lane * 4 + 1] = static_cast<uint8_t>(vals[group * 4 + 1]);
                                    zmm_dst[lane * 4 + 2] = static_cast<uint8_t>(vals[group * 4 + 2]);
                                    zmm_dst[lane * 4 + 3] = static_cast<uint8_t>(vals[group * 4 + 3]);
                                }
                                else
                                {
                                    zmm_dst[lane * 4 + 0] = 0;
                                    zmm_dst[lane * 4 + 1] = 0;
                                    zmm_dst[lane * 4 + 2] = 0;
                                    zmm_dst[lane * 4 + 3] = 0;
                                }
                            }
                        }
                    }

                    // Write comp inline after group data
                    size_t meta_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64;
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(interleaved_dst + 2048);
                    for (int col = 0; col < 64; ++col)
                    {
                        if (col < n_cols)
                        {
                            size_t flat_idx = ((size_t)chunk * blocks_per_row * 64 +
                                               (size_t)kb * 64 + col) * 32;
                            const int8_t *vals = out.int8_flat.data() + flat_idx;
                            int32_t sum = 0;
                            for (int i = 0; i < 32; ++i)
                                sum += vals[i];
                            inline_comp[col] = static_cast<int16_t>(sum);
                        }
                        else
                        {
                            inline_comp[col] = 0;
                        }
                    }

                    // Write scales inline after comp
                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(interleaved_dst + 2048 + 128);
                    std::memcpy(inline_scales, &temp_scales[meta_idx], 64 * sizeof(uint16_t));

                    // Write mins inline after scales (if asymmetric)
                    if (out.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(interleaved_dst + 2048 + 256);
                        std::memcpy(inline_mins, &temp_mins[meta_idx], 64 * sizeof(uint16_t));
                    }
                }
            }
        }

        // Release the intermediate INT8 decode buffer.
        // The AVX-512 GEMV uses only native_interleaved; int8_flat is dead weight.
        // For FFN_Down (7B): frees ~68 MB, reducing TLB and virtual memory pressure.
        out.releaseDecodedBuffer();

        return true;
    }

} // namespace llaminar2::cpu::native_vnni
