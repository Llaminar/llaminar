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
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "CPUNativeVNNIDecode.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "tensors/AlignedVector.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorClasses.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_vnni
{

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
    /// Return the native block byte size for a given codebook_id.
    /// For per-block formats (Q4_0, IQ4_NL, Q8_0, etc.), returns sizeof(BlockType).
    /// For superblock formats (Q6_K, Q3_K, etc.), returns the superblock byte size.
    /// Returns 0 for unknown formats.
    inline size_t native_block_bytes_for_codebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:  return 18;  // Q4_0Block
        case 4:  return 18;  // IQ4_NLBlock
        case 5:  return 20;  // Q4_1Block
        case 6:  return 22;  // Q5_0Block
        case 7:  return 24;  // Q5_1Block
        case 10: return 210; // Q6_KBlock (superblock, 256 elements)
        case 11: return 110; // Q3_KBlock (superblock, 256 elements)
        case 12: return 84;  // Q2_KBlock (superblock, 256 elements)
        case 13: return 144; // Q4_KBlock (superblock, 256 elements)
        case 14: return 176; // Q5_KBlock (superblock, 256 elements)
        case 19: return 34;  // Q8_0Block
        default: return 0;
        }
    }

    /// Return the number of quantized elements per native block for a given codebook_id.
    /// Per-block formats: 32 elements. Superblock formats: 256 elements.
    inline int native_block_elements_for_codebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 10: // Q6_K
        case 11: // Q3_K
        case 12: // Q2_K
        case 13: // Q4_K
        case 14: // Q5_K
            return 256;
        default:
            return 32;
        }
    }

    struct CPUNativeVNNIPackedWeights
    {
        /// Raw native payload bytes in [N_chunks][blocks_per_row][64][payload_bytes] layout
        /// Only populated for nibble-LUT formats (is_nibble_lut == true).
        std::vector<uint8_t> payload;

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
        std::vector<int8_t> int8_flat;

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

        /// Whether this format uses vpshufb nibble LUT decode in the GEMV inner loop.
        /// True for: Q4_0, IQ4_NL, Q4_1, IQ4_XS (4-bit nibble formats).
        /// False for: Q5_0, Q5_1, Q6_K, Q3_K, Q2_K, IQ2/3/1* (pre-decoded INT8).
        bool is_nibble_lut = false;

        /// Bytes of pure group data per K-block (before metadata).
        /// 1024 for nibble-LUT (4 groups × 4 ZMMs × 64 bytes).
        /// 2048 for INT8 pre-decoded (8 groups × 4 ZMMs × 64 bytes).
        int data_stride = 1024;

        /// Total bytes per K-block including inline metadata.
        /// = data_stride + 128 (comp) + 128 (scales) [+ 128 (mins) if asymmetric]
        int interleaved_block_stride = 1280;

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
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + 128);
        }

        /// Mins pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkMins(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const uint16_t *>(
                interleavedBase() + block_offset + data_stride + 256);
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
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const int16_t *>(
                interleavedBase() + block_offset + data_stride);
        }

        /// Flat INT8 values for N-chunk c, K-block kb, column n_local (32 INT8 values)
        /// Only valid for INT8 pre-decoded formats (is_nibble_lut == false).
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
        const int N = meta.N;
        const int bpr = meta.blocks_per_row;
        const int N_chunks = (meta.N_padded) / 64;
        const uint8_t codebook_id = meta.codebook_id;

        // Row stride in native_blocks array (bytes between adjacent rows)
        const size_t native_row_stride = static_cast<size_t>(bpr) * block_size;

        if (meta.is_nibble_lut)
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
        out.is_nibble_lut = is_nibble_lut_format(fmt->codebook_id);

        // When rotation is active, rotation mixes values across the 32-element
        // quantization block boundaries, so we must dequant → rotate → requant.
        // The result is always INT8 pre-decoded format (8 groups), regardless of
        // the original tensor format. We still preserve the original codebook_id
        // for diagnostic purposes, but the packed layout uses the INT8 path.
        const bool use_rotated_path = (rotation != nullptr);
        if (use_rotated_path)
            out.is_nibble_lut = false; // Force INT8 pre-decoded path

        out.data_stride = out.is_nibble_lut ? 1024 : 2048;
        out.interleaved_block_stride = out.data_stride + 256 + (out.is_asymmetric ? 128 : 0);

        // Temporary per-column metadata arrays used during packing.
        // These get copied inline into native_interleaved during the interleaving pass.
        size_t total_blocks = (size_t)N_chunks * blocks_per_row * 64;
        std::vector<uint16_t> temp_scales(total_blocks, 0);
        std::vector<uint16_t> temp_mins(total_blocks, 0);
        std::vector<int16_t> temp_comp(total_blocks, 0);

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

            out.is_asymmetric = false; // Rotated weights are always symmetric
            out.interleaved_block_stride = 2048 + 256; // Recompute without mins

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
        else if (out.is_nibble_lut)
        {
            // =================================================================
            // NIBBLE-LUT PATH (Q4_0, IQ4_NL, Q4_1, IQ4_XS)
            //
            // Store raw native payload bytes + VNNI-interleaved native bytes.
            // The GEMV kernel decodes nibbles at runtime via vpshufb LUT.
            // Memory: 0.5 byte/element (half of INT8).
            // =================================================================

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
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    unpackable->packVnniBlock(gpu_ctx, n, kb);
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
            // Decode to INT8 at pack time, store in VNNI-interleaved layout
            // with 8 groups. The GEMV kernel loads pre-decoded INT8 directly.
            // Memory: 1.0 byte/element.
            // =================================================================

            size_t int8_total = (size_t)N_chunks * blocks_per_row * 64 * 32;
            out.int8_flat.resize(int8_total, 0);

// Single pass: extract scales/mins AND decoded INT8 values
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

                            // Store decoded INT8 values
                            size_t flat_offset = idx * 32;
                            std::memcpy(out.int8_flat.data() + flat_offset, sb_vals + i * 32, 32);
                        }
                        kb += 8;
                    }
                }

                // Process remaining blocks individually
                for (; kb < blocks_per_row; ++kb)
                {
                    size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                    temp_scales[idx] = fp32_to_fp16(unpackable->get_block_scale(src_row, kb));
                    temp_mins[idx] = fp32_to_fp16(unpackable->get_block_min(src_row, kb));

                    // Decode to INT8
                    size_t flat_offset = idx * 32;
                    unpackable->unpack_block_to_int8(src_row, kb, out.int8_flat.data() + flat_offset);
                }
            }
        }

        // =================================================================
        // INT8 INTERLEAVING (shared by rotation path and INT8 pre-decoded path)
        //
        // Build VNNI-interleaved INT8 buffer (8 groups) + inline comp/scales/mins.
        // Both paths populate int8_flat + temp_scales before reaching here.
        // =================================================================
        if (!out.is_nibble_lut)
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
