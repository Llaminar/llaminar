/**
 * @file CPUNativeVNNIWeightPacker.h
 * IQ3_S keeps original half/nibble factors beside unchanged decoded grid bytes;
 * all effective-scale loads reconstruct in FP32 before existing contributions.
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
 * The sole retained execution root is [N_chunks][K_blocks][interleaved unit].
 * Native source payload/scale/min words are read through the same bounded
 * native-group append primitive used by streaming preparation. Its scratch is
 * capped by the common source policy and owned by WeightLoadStaging; no whole
 * source-order nibble/Q6 matrix is materialized. Legacy archive payload sections
 * remain import-only and retire at engine publication. Activation rotation
 * retains its separate existing complete-row arithmetic contract.
 *
 * This layout ensures:
 * - Sequential K-block streaming within an N-chunk → good prefetch/cache line use
 * - 64 contiguous scales per K-block → aligned ZMM loads
 * - Native payload preserved → 2× less memory for Q4_0 vs INT8
 */

#pragma once

#include "memory/CPUWeightStoragePlacement.h"
#include "kernels/common/NativeVnniScaleLoad.h"
#include "loaders/NativeWeightLoadStaging.h"

#include <cstdint>
#include <span>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "CPUNativeVNNIDecode.h"
#include "CPUNativeVNNIEncoding.h"
#include "CPUNativeVNNIWeightView.h"
#include "CPUNativeVNNIPackedWeights.h"
#include "CPUNativeVNNIPreparedFootprint.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorClasses.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/NativeVnniQ6Scale.h"
#include "tensors/NativeVnniIQ2SScale.h"
#include "tensors/NativeVnniIQ4XSScale.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeVnniIQ3SScale.h"
#include "kernels/common/DeviceFP32NumericalContract.h"
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
     * Newly prepared weights retain one execution representation. Source-order
     * nibble bytes are temporary setup storage; scalar diagnostics reconstruct
     * their bounded block from the same interleaved bytes as AVX2/AVX512.
     * The optional payload member remains solely for archive compatibility.
     *
     * **native_interleaved**: VNNI-interleaved bytes with inline metadata:
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
        case 21: return sizeof(Q8_KBlock); // Original FP32-scale Q8_K source block
        case kNativeVnniMXFP4Codebook: return sizeof(MXFP4Block); // E8M0 + 16 code bytes
        default: return 0;
        }
    }



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
        if (meta.usesQ6KNativeDualScale() || meta.usesCompactMultiScale() ||
            meta.usesNibbleLUTSourceScale() || meta.usesIQ3SExpandedSourceScale())
        {
            throw std::invalid_argument(
                "Deferred native-block repacking does not own the source "
                "superblock context required by native multi-scale layouts");
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
            // MXFP4 stores one E8M0 byte before its codes; Q4_1 a half scale and min.
            const bool mxfp4_source = codebook_id == kNativeVnniMXFP4Codebook;
            const int payload_offset = mxfp4_source ? 1 : ((codebook_id == 5) ? 4 : 2); // Q4_1=4, others=2

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

                        if (mxfp4_source)
                        {
                            // Native nibble MXFP4 units always carry the exact BF16
                            // power-of-two word of their E8M0 scale.
                            if (meta.scaleEncoding() != CPUNativeVNNIScaleEncoding::PowerOfTwoBF16)
                                throw std::logic_error("MXFP4 nibble units require the BF16 power-of-two scale encoding");
                            if (!mxfp4::blockFitsNativeE8M0(blk[0], blk + 1))
                                throw std::invalid_argument(
                                    "MXFP4 block exponent is outside the native E8M0 range [2, 252]");
                            scales_buf[col] = mxfp4::powerOfTwoBF16Word(mxfp4::preparedNativeE8M0Byte(blk[0]));
                        }
                        else
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
     * byte of its disjoint final destination. Single-scale native payloads
     * retain their exact integer grids and FP16 scale/minimum metadata; they
     * must not pass through the lossy Q8 normalization helpers. Other formats
     * use the source's complete superblock decoder. Both paths write the same
     * interleaved execution layout without matrix-sized temporary arrays.
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
     * @param placement Final destination first-touch policy, never a post-pack migration.
     */
    inline void packExpandedInt8Direct(
        const IINT8Unpackable &unpackable,
        CPUNativeVNNIPackedWeights &out,
        int row_start,
        int N,
        int blocks_per_row,
        int N_chunks,
        bool use_superblock,
        CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
    {
        if (!out.usesExpandedInt8() || out.data_stride != 2048)
        {
            throw std::invalid_argument(
                "Direct Expanded-INT8 packing requires the complete 2048-byte data layout");
        }

        const size_t interleaved_total =
            static_cast<size_t>(N_chunks) * blocks_per_row *
            static_cast<size_t>(out.interleaved_block_stride);
        out.native_interleaved = placement.allocate<uint8_t>(interleaved_total);

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
                uint16_t source_bases[kBlocksPerTile]{};
                uint16_t source_coefficients[kBlocksPerTile]{};

                for (int column = 0; column < kColumnsPerChunk; ++column)
                {
                    const bool source_column_valid = column < valid_columns;
                    if (source_column_valid)
                    {
                        const int source_row =
                            row_start + chunk * kColumnsPerChunk + column;
                        if (is_payload_decodable(out.codebook_id))
                        {
                            // Read the same native block contract as GPU
                            // preparation. IQuant grids already fit signed
                            // bytes; an absmax/Q8 pass would change the weights.
                            // One bounded stack tile keeps preparation streaming
                            // without a matrix-sized separated shadow.
                            alignas(64) uint8_t native_payload[kBlocksPerTile * 32]{};
                            uint16_t native_scales[kBlocksPerTile]{};
                            uint16_t native_mins[kBlocksPerTile]{};
                            VnniPackContext context{
                                .raw_bytes = nullptr,
                                .N = 1,
                                .K = out.K,
                                .blocks_per_row = blocks_per_row,
                                .payload_bytes = out.payload_bytes,
                                .payload_array = native_payload,
                                .scales_array = native_scales,
                                .mins_array = native_mins,
                                .emins_array = nullptr,
                                .destination_block_origin = first_kb,
                                .destination_block_count = block_count,
                                .primary_layout = out.primaryScaleLayout(),
                            };
                            for (int local_kb = 0; local_kb < block_count; ++local_kb)
                            {
                                unpackable.packVnniBlock(
                                    context, source_row, 0, first_kb + local_kb);
                                decode_native_block(
                                    out.codebook_id,
                                    native_payload + local_kb * out.payload_bytes,
                                    decoded + local_kb * kValuesPerBlock);
                                if (out.usesIQ3SExpandedSourceScale())
                                {
                                    source_bases[local_kb] = native_scales[local_kb];
                                    source_coefficients[local_kb] = native_mins[local_kb];
                                    if (!native_vnni_iq3s_scale::valid(source_coefficients[local_kb]))
                                        throw std::invalid_argument("IQ3_S source nibble is invalid");
                                }
                                else
                                {
                                    scales[local_kb] = fp16_to_fp32(native_scales[local_kb]);
                                    mins[local_kb] = fp16_to_fp32(native_mins[local_kb]);
                                }
                            }
                        }
                        else if (use_superblock &&
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
                            inline_scales[column] = out.usesIQ3SExpandedSourceScale()
                                ? source_bases[local_kb] : fp32_to_fp16(scales[local_kb]);
                        }
                        else
                        {
                            compensation[column] = 0;
                            inline_scales[column] = 0;
                        }

                        if (out.is_asymmetric || out.usesIQ3SExpandedSourceScale())
                        {
                            auto *const inline_mins =
                                reinterpret_cast<uint16_t *>(
                                    destination + out.data_stride + 256);
                            inline_mins[column] = source_column_valid
                                ? (out.usesIQ3SExpandedSourceScale()
                                    ? source_coefficients[local_kb] : fp32_to_fp16(mins[local_kb])) : 0;
                        }
                    }
                }
            }
        }
    }

#include "CPUNativeVNNIWindowPacking.inc"

    /**
     * @brief Exact capped workspace for the immediate consumer of already-resident native bytes.
     * @param format Same canonical append codec, with no second format table.
     * @param n Useful final rows. @param k Original source columns.
     * @param placement Same original allocator placement as final prepared storage.
     * @return Workspace for at most one 64-row chunk, further bounded by the common startup cap.
     */
    inline NativeWeightLoadStagingGeometry immediateNativePackScratchGeometry(
        const NativeVnniFormatInfo& format,int n,int k,CPUStoragePlacement placement)
    {
        if(n<=0 || k<=0 || k>std::numeric_limits<int>::max()-31)
            throw std::invalid_argument("Immediate native pack requires representable positive source geometry");
        const auto groups=(size_t(k)+31)/32;
        const auto native_groups=(groups+(format.is_superblock?7:0))/(format.is_superblock?8:1);
        return resolveNativeWeightLoadStaging({NativePackGroupScratch::checked(format).bytes,0,0,
            std::min(size_t(n),size_t{64})*native_groups},placement,configuredNativeWeightLoadStagingBudgetBytes());
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
     * @param rotation    Optional activation rotation to incorporate in packed weights.
     * @param placement   Final destination first-touch policy for every encoding.
     * @return true on success
     */

    inline bool packWeightsCPUNativeVNNI(const TensorBase *weights,
                                         CPUNativeVNNIPackedWeights &out,
                                         int row_start = 0, int row_end = -1,
                                         const ActivationRotation *rotation = nullptr,
                                         CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
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

        // Reusing an archive-imported container must not preserve its optional
        // source-order matrix when constructing a new canonical execution bank.
        out.releaseSourcePayload();
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
        out.encoding = preparedEncodingForSource(
            *fmt, use_rotated_path);

        out.data_stride = preparedDataStride(out.encoding);
        out.interleaved_block_stride = preparedInterleavedBlockStride(
            out.encoding, out.is_asymmetric);

        // Temporary metadata is allocated only by layouts that still need a
        // distinct source-oriented pass.  Expanded-INT8 preparation writes
        // directly to final storage and therefore owns no matrix-sized shadow.
        size_t total_blocks = (size_t)N_chunks * blocks_per_row * 64;
        std::vector<uint16_t> temp_scales;
        std::vector<uint16_t> temp_mins;

        bool use_superblock = (unpackable->superblock_size() == 256);

        if (!use_rotated_path)
        {
            // The source already owns native bytes. Only the same append
            // primitive's workspace is transient; never make a native copy.
            const auto original_memory=placement.physicalMemoryAuthority();
            std::optional<PhysicalMemoryCPUWeightSourceLease> source_scope;
            if(original_memory)
                source_scope.emplace(original_memory->acquireCPUWeightSourceTransaction(placement.allocatorDevice()));
            const auto geometry=immediateNativePackScratchGeometry(*fmt,N,K,placement.storagePlacement());
            auto scratch=original_memory
                ? CPUStorageRoot<uint8_t>(geometry.native_bytes,PhysicalMemoryOwner::WeightLoadStaging,
                    placement.allocatorDevice(),placement.storagePlacement(),CPUStorageInitialization::Uninitialized).allocate(original_memory)
                : placement.allocate<uint8_t>(geometry.native_bytes);
            initializeNativeWindowOutput(*fmt,N,K,placement,out);
            const int group_blocks=fmt->is_superblock?8:1;
            const size_t groups=(size_t(blocks_per_row)+group_blocks-1)/group_blocks;
            auto bytes=std::span<std::byte>(reinterpret_cast<std::byte*>(scratch.data()),scratch.size());
            if(geometry.maximum_units_per_window>=groups)
            {
                const int rows_per_window=static_cast<int>(std::min(size_t{64},geometry.maximum_units_per_window/groups));
                for(int row=0;row<N;row+=rows_per_window)
                    appendNativeWeightGroups(*unpackable,*fmt,K,row_start+row,0,std::min(rows_per_window,N-row),
                        blocks_per_row,row,0,bytes,out);
            }
            else
            {
                // Even one full source row may exceed the cap. Visit native
                // K groups in order, without rebuilding a row-sized shadow.
                const int blocks_per_window=static_cast<int>(geometry.maximum_units_per_window)*group_blocks;
                for(int row=0;row<N;++row)for(int kb=0;kb<blocks_per_row;kb+=blocks_per_window)
                    appendNativeWeightGroups(*unpackable,*fmt,K,row_start+row,kb,1,
                        std::min(blocks_per_window,blocks_per_row-kb),row,kb,bytes,out);
            }
            return true;
        }

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
            out.native_interleaved = placement.allocate<uint8_t>(interleaved_total);

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
                    if (out.is_asymmetric || out.usesNibbleLUTSourceScale())
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
