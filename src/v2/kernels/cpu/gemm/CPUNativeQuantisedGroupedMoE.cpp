/**
 * @file CPUNativeQuantisedGroupedMoE.cpp
 * @brief Single-OMP fused CPU grouped MoE FFN megakernel across all 18 codebooks and Scalar/AVX2/AVX-512.
 *
 * Eliminates both the expert-major Q8 activation copy and the full Gate/Up FP32
 * DRAM round-trip by indexing unique row-major router Q8 rows directly, running
 * LPT-sorted L1d-resident strips through fused Gate + Up -> SwiGLU -> Q8_1,
 * synchronizing once, and streaming weight-stationary Down directly into the
 * caller's destination route bank. Preserves exact ascending K-partition FP32
 * reduction order bitwise identical to serial M=1 decode and GPU-aligned MoE.
 */
#include "CPUNativeQuantisedGemmKernel.h"
#include "CPUNativeQuantisedGemv.h"
#include "CPUNativeQuantisedScalarReference.h"
#include "kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "utils/CPUFeatures.h"
#include "utils/Logger.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace llaminar2::cpu::native_quantised
{
namespace
{
    constexpr int kStripMaxRows = 64;
    constexpr int kChunkCols = 64;

    struct Phase1StripTask
    {
        int expert_idx;
        int chunk;
        int strip_begin;
        int strip_rows;
    };

    struct Phase2StripTask
    {
        int expert_idx;
        int chunk_begin;
        int chunk_count;
        int strip_begin;
        int strip_rows;
    };

    struct SinglePassExpertTask
    {
        int expert_idx;
        int strip_begin;
        int strip_rows;
    };

    struct TileConfigCacheKey
    {
        uint8_t codebook_id = 0;
        CPUNativeQuantisedEncoding encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
        bool is_asymmetric = false;
        CPUProjectionNumericalPolicy numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
        int n = 0;
        int k = 0;
        bool operator==(const TileConfigCacheKey &) const = default;
    };

    struct TileConfigCacheEntry
    {
        TileConfigCacheKey key{};
        int k_tiles = 1;
        int k_blocks_per_tile = 1;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i lut512 = _mm512_setzero_si512();
#endif
#if defined(__AVX2__)
        __m256i lut256 = _mm256_setzero_si256();
#endif
    };

    struct PreparedMatrixInfo
    {
        const CPUNativeQuantisedPackedWeights *packed = nullptr;
        int k_blocks = 0;
        int k_tiles = 1;
        int k_blocks_per_tile = 1;
        bool is_nibble = false;
        bool is_unsigned_nibble = false;
        bool is_unsigned_int8 = false;
        bool is_compact_ms = false;
        bool is_compact_five_bit = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i lut512 = _mm512_setzero_si512();
#endif
#if defined(__AVX2__)
        __m256i lut256 = _mm256_setzero_si256();
#endif
    };

    inline PreparedMatrixInfo prepareMatrixInfoCached(
        const CPUNativeQuantisedPackedWeights &packed,
        int n,
        int k,
        int workers,
        const CPUCacheGeometry &cache,
        std::array<TileConfigCacheEntry, 16> &tile_cache,
        size_t &tile_cache_count)
    {
        PreparedMatrixInfo info{};
        info.packed = &packed;
        info.k_blocks = packed.blocks_per_row;
        info.is_nibble = packed.usesNibbleLUT();
        info.is_unsigned_nibble = packed.usesUnsignedNibble();
        info.is_unsigned_int8 = packed.usesUnsignedInt8() && !packed.is_asymmetric;
        info.is_compact_ms = packed.usesCompactMultiScale();
        info.is_compact_five_bit = packed.usesCompactFiveBit();

        const TileConfigCacheKey key{
            packed.codebook_id,
            packed.encoding,
            packed.is_asymmetric,
            packed.numerical_policy,
            n,
            k};
        for (size_t i = 0; i < tile_cache_count; ++i)
        {
            if (tile_cache[i].key == key)
            {
                info.k_tiles = tile_cache[i].k_tiles;
                info.k_blocks_per_tile = tile_cache[i].k_blocks_per_tile;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                info.lut512 = tile_cache[i].lut512;
#endif
#if defined(__AVX2__)
                info.lut256 = tile_cache[i].lut256;
#endif
                return info;
            }
        }

        const auto cfg = serialTileConfigForPreparedProjection(
            packed.preparedFootprint(),
            packed.codebook_id,
            packed.numerical_policy,
            n,
            k,
            workers,
            cache);
        info.k_tiles = std::max(1, cfg.k_tiles);
        info.k_blocks_per_tile = (info.k_blocks + info.k_tiles - 1) / info.k_tiles;
        // Per MoEProjectionNumericalContract::oneNativeQuantisedBlockPerOrderedPartition,
        // when every ordered partition contains exactly one Q8 block (e.g., K <= 512),
        // accumulating partitions from +0 and folding left-to-right is bitwise identical
        // to one contiguous ascending K loop from +0, eliminating k_tiles x dispatch
        // and stack-fold overhead.
        if (info.k_tiles > 1 && info.k_tiles == info.k_blocks && info.k_blocks_per_tile == 1)
        {
            info.k_tiles = 1;
            info.k_blocks_per_tile = info.k_blocks;
        }
        const bool needs_lut = info.is_nibble && nibbleDecodeKind(packed.codebook_id) == NibbleDecodeKind::LUT;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (needs_lut)
            info.lut512 = build_decode_lut(packed.codebook_id);
#endif
#if defined(__AVX2__)
        if (needs_lut)
            info.lut256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);
#endif
        if (tile_cache_count < tile_cache.size())
        {
            auto &slot = tile_cache[tile_cache_count++];
            slot.key = key;
            slot.k_tiles = info.k_tiles;
            slot.k_blocks_per_tile = info.k_blocks_per_tile;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
            slot.lut512 = info.lut512;
#endif
#if defined(__AVX2__)
            slot.lut256 = info.lut256;
#endif
        }
        return info;
    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /**
     * @brief Six-row 2z UnsignedNibble AVX-512 kernel (`Q4_K`, `Q4_0`, `Q4_1`).
     *
     * Shares each 32-bit activation broadcast across two 16-column ZMM weight vectors (2:1
     * `vpdpbusd:vpbroadcastd` reuse) and halves weight passes compared to 1z kernels.
     * Supports both single-partition (`k_tiles <= 1`) and ordered multi-partition (`k_tiles > 1`)
     * reduction without register spills inside the inner K loop.
     */
    template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
    inline void gemm_6row_unsigned_nibble_2z_impl(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const Q8_1Block *const r0 = rows[0], *const r1 = rows[1], *const r2 = rows[2];
        const Q8_1Block *const r3 = rows[3], *const r4 = rows[4], *const r5 = rows[5];
        alignas(64) float red_buf[6][2][16];

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            for (int kt = 0; kt < k_tiles; ++kt)
            {
                const int kb_start = (k_tiles <= 1) ? 0 : kt * bpt;
                const int kb_end = (k_tiles <= 1) ? k_blocks : std::min(kb_start + bpt, k_blocks);
                __m512 fp00 = _mm512_setzero_ps(), fp01 = _mm512_setzero_ps();
                __m512 fp10 = _mm512_setzero_ps(), fp11 = _mm512_setzero_ps();
                __m512 fp20 = _mm512_setzero_ps(), fp21 = _mm512_setzero_ps();
                __m512 fp30 = _mm512_setzero_ps(), fp31 = _mm512_setzero_ps();
                __m512 fp40 = _mm512_setzero_ps(), fp41 = _mm512_setzero_ps();
                __m512 fp50 = _mm512_setzero_ps(), fp51 = _mm512_setzero_ps();

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    if (kb + 1 < k_blocks)
                    {
                        __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 0, zbase), 0, 3);
                        __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 0, zbase + 1), 0, 3);
                        __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 2, zbase), 0, 3);
                        __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 2, zbase + 1), 0, 3);
                        if (zbase == 0)
                            __builtin_prefetch(packed.chunkScales(chunk, kb + 1), 0, 3);
                    }
                    const Q8_1Block &a0 = r0[kb], &a1 = r1[kb], &a2 = r2[kb];
                    const Q8_1Block &a3 = r3[kb], &a4 = r4[kb], &a5 = r5[kb];

                    __m512i i00 = _mm512_setzero_si512(), i01 = _mm512_setzero_si512();
                    __m512i i10 = _mm512_setzero_si512(), i11 = _mm512_setzero_si512();
                    __m512i i20 = _mm512_setzero_si512(), i21 = _mm512_setzero_si512();
                    __m512i i30 = _mm512_setzero_si512(), i31 = _mm512_setzero_si512();
                    __m512i i40 = _mm512_setzero_si512(), i41 = _mm512_setzero_si512();
                    __m512i i50 = _mm512_setzero_si512(), i51 = _mm512_setzero_si512();

                    for (int group = 0; group < 4; ++group)
                    {
                        const __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                        const __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                        const __m512i lo0 = _mm512_and_si512(raw0, mask_0F);
                        const __m512i lo1 = _mm512_and_si512(raw1, mask_0F);
                        const int goff = group * 4;
                        int32_t w;
#define LLAMINAR_UNIB6X2_STEP(R, ABLK, V0, V1, OFF)                                 \
                        std::memcpy(&w, ABLK.qs + (OFF), sizeof(w));                \
                        {                                                           \
                            const __m512i av = _mm512_set1_epi32(w);                \
                            i##R##0 = _mm512_dpbusd_epi32(i##R##0, V0, av);         \
                            i##R##1 = _mm512_dpbusd_epi32(i##R##1, V1, av);         \
                        }
                        LLAMINAR_UNIB6X2_STEP(0, a0, lo0, lo1, goff);
                        LLAMINAR_UNIB6X2_STEP(1, a1, lo0, lo1, goff);
                        LLAMINAR_UNIB6X2_STEP(2, a2, lo0, lo1, goff);
                        LLAMINAR_UNIB6X2_STEP(3, a3, lo0, lo1, goff);
                        LLAMINAR_UNIB6X2_STEP(4, a4, lo0, lo1, goff);
                        LLAMINAR_UNIB6X2_STEP(5, a5, lo0, lo1, goff);

                        const __m512i hi0 = _mm512_and_si512(_mm512_srli_epi16(raw0, 4), mask_0F);
                        const __m512i hi1 = _mm512_and_si512(_mm512_srli_epi16(raw1, 4), mask_0F);
                        const int hioff = goff + 16;
                        LLAMINAR_UNIB6X2_STEP(0, a0, hi0, hi1, hioff);
                        LLAMINAR_UNIB6X2_STEP(1, a1, hi0, hi1, hioff);
                        LLAMINAR_UNIB6X2_STEP(2, a2, hi0, hi1, hioff);
                        LLAMINAR_UNIB6X2_STEP(3, a3, hi0, hi1, hioff);
                        LLAMINAR_UNIB6X2_STEP(4, a4, hi0, hi1, hioff);
                        LLAMINAR_UNIB6X2_STEP(5, a5, hi0, hi1, hioff);
#undef LLAMINAR_UNIB6X2_STEP
                    }
                    if (decode_kind == NibbleDecodeKind::Q4_0_LINEAR)
                    {
#define LLAMINAR_UNIB6X2_Q40(R, ABLK)                                               \
                        {                                                           \
                            const __m512i u = _mm512_set1_epi32(static_cast<int32_t>(ABLK.sum_qs) * 8); \
                            i##R##0 = _mm512_sub_epi32(i##R##0, u);                 \
                            i##R##1 = _mm512_sub_epi32(i##R##1, u);                 \
                        }
                        LLAMINAR_UNIB6X2_Q40(0, a0); LLAMINAR_UNIB6X2_Q40(1, a1);
                        LLAMINAR_UNIB6X2_Q40(2, a2); LLAMINAR_UNIB6X2_Q40(3, a3);
                        LLAMINAR_UNIB6X2_Q40(4, a4); LLAMINAR_UNIB6X2_Q40(5, a5);
#undef LLAMINAR_UNIB6X2_Q40
                    }

                    const uint16_t *const sc_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                    const __m512 sc0 = loadScales512(packed, sc_ptr);
                    const __m512 sc1 = loadScales512(packed, sc_ptr + 16);
                    if constexpr (Asymmetric)
                    {
                        const uint16_t *const mn_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                        const __m512 mn0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mn_ptr)));
                        const __m512 mn1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mn_ptr + 16)));
#define LLAMINAR_UNIB6X2_ASYM(R, ABLK)                                              \
                        {                                                           \
                            const float d = nativeQuantisedFP16ScaleToFP32(ABLK.d); \
                            fp##R##0 = accumulateCorrectedSingleScaleBlockAVX512(fp##R##0, i##R##0, sc0, ABLK.sum_qs, mn0, d, Policy); \
                            fp##R##1 = accumulateCorrectedSingleScaleBlockAVX512(fp##R##1, i##R##1, sc1, ABLK.sum_qs, mn1, d, Policy); \
                        }
                        LLAMINAR_UNIB6X2_ASYM(0, a0); LLAMINAR_UNIB6X2_ASYM(1, a1);
                        LLAMINAR_UNIB6X2_ASYM(2, a2); LLAMINAR_UNIB6X2_ASYM(3, a3);
                        LLAMINAR_UNIB6X2_ASYM(4, a4); LLAMINAR_UNIB6X2_ASYM(5, a5);
#undef LLAMINAR_UNIB6X2_ASYM
                    }
                    else
                    {
#define LLAMINAR_UNIB6X2_SYM(R, ABLK)                                               \
                        {                                                           \
                            const float d = nativeQuantisedFP16ScaleToFP32(ABLK.d); \
                            fp##R##0 = accumulateSingleScaleBlockAVX512(fp##R##0, i##R##0, sc0, d, Policy); \
                            fp##R##1 = accumulateSingleScaleBlockAVX512(fp##R##1, i##R##1, sc1, d, Policy); \
                        }
                        LLAMINAR_UNIB6X2_SYM(0, a0); LLAMINAR_UNIB6X2_SYM(1, a1);
                        LLAMINAR_UNIB6X2_SYM(2, a2); LLAMINAR_UNIB6X2_SYM(3, a3);
                        LLAMINAR_UNIB6X2_SYM(4, a4); LLAMINAR_UNIB6X2_SYM(5, a5);
#undef LLAMINAR_UNIB6X2_SYM
                    }
                }

                if (k_tiles <= 1)
                {
                    _mm512_store_ps(out_buf[0] + zbase * 16, fp00); _mm512_store_ps(out_buf[0] + (zbase + 1) * 16, fp01);
                    _mm512_store_ps(out_buf[1] + zbase * 16, fp10); _mm512_store_ps(out_buf[1] + (zbase + 1) * 16, fp11);
                    _mm512_store_ps(out_buf[2] + zbase * 16, fp20); _mm512_store_ps(out_buf[2] + (zbase + 1) * 16, fp21);
                    _mm512_store_ps(out_buf[3] + zbase * 16, fp30); _mm512_store_ps(out_buf[3] + (zbase + 1) * 16, fp31);
                    _mm512_store_ps(out_buf[4] + zbase * 16, fp40); _mm512_store_ps(out_buf[4] + (zbase + 1) * 16, fp41);
                    _mm512_store_ps(out_buf[5] + zbase * 16, fp50); _mm512_store_ps(out_buf[5] + (zbase + 1) * 16, fp51);
                }
                else if (kt == 0)
                {
                    _mm512_store_ps(red_buf[0][0], fp00); _mm512_store_ps(red_buf[0][1], fp01);
                    _mm512_store_ps(red_buf[1][0], fp10); _mm512_store_ps(red_buf[1][1], fp11);
                    _mm512_store_ps(red_buf[2][0], fp20); _mm512_store_ps(red_buf[2][1], fp21);
                    _mm512_store_ps(red_buf[3][0], fp30); _mm512_store_ps(red_buf[3][1], fp31);
                    _mm512_store_ps(red_buf[4][0], fp40); _mm512_store_ps(red_buf[4][1], fp41);
                    _mm512_store_ps(red_buf[5][0], fp50); _mm512_store_ps(red_buf[5][1], fp51);
                }
                else
                {
                    const bool last = (kt + 1 == k_tiles);
#define LLAMINAR_UNIB6X2_FOLD(R)                                                    \
                    {                                                               \
                        const __m512 s0 = addOrderedAVX512(_mm512_load_ps(red_buf[R][0]), fp##R##0); \
                        const __m512 s1 = addOrderedAVX512(_mm512_load_ps(red_buf[R][1]), fp##R##1); \
                        if (last) {                                                 \
                            _mm512_store_ps(out_buf[R] + zbase * 16, s0);           \
                            _mm512_store_ps(out_buf[R] + (zbase + 1) * 16, s1);     \
                        } else {                                                    \
                            _mm512_store_ps(red_buf[R][0], s0);                     \
                            _mm512_store_ps(red_buf[R][1], s1);                     \
                        }                                                           \
                    }
                    LLAMINAR_UNIB6X2_FOLD(0); LLAMINAR_UNIB6X2_FOLD(1);
                    LLAMINAR_UNIB6X2_FOLD(2); LLAMINAR_UNIB6X2_FOLD(3);
                    LLAMINAR_UNIB6X2_FOLD(4); LLAMINAR_UNIB6X2_FOLD(5);
#undef LLAMINAR_UNIB6X2_FOLD
                }
            }
        }
    }

    /**
     * @brief Three-row 4z single-pass UnsignedNibble AVX-512 kernel (`Q4_K`, `Q4_0`, `Q4_1`).
     *
     * Sweeps all 64 columns in one pass over weights and activations with 4:1
     * `vpdpbusd:vpbroadcastd` reuse, covering 3..5-row strips and remainders at peak VNNI rate.
     */
    template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
    inline void gemm_3row_unsigned_nibble_4z_impl(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const Q8_1Block *const r0 = rows[0], *const r1 = rows[1], *const r2 = rows[2];
        alignas(64) float red_buf[3][64];

        for (int kt = 0; kt < k_tiles; ++kt)
        {
            const int kb_start = (k_tiles <= 1) ? 0 : kt * bpt;
            const int kb_end = (k_tiles <= 1) ? k_blocks : std::min(kb_start + bpt, k_blocks);
            __m512 fp00 = _mm512_setzero_ps(), fp01 = _mm512_setzero_ps(), fp02 = _mm512_setzero_ps(), fp03 = _mm512_setzero_ps();
            __m512 fp10 = _mm512_setzero_ps(), fp11 = _mm512_setzero_ps(), fp12 = _mm512_setzero_ps(), fp13 = _mm512_setzero_ps();
            __m512 fp20 = _mm512_setzero_ps(), fp21 = _mm512_setzero_ps(), fp22 = _mm512_setzero_ps(), fp23 = _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                if (kb + 1 < k_blocks)
                {
                    __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 0, 0), 0, 3);
                    __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 0, 2), 0, 3);
                    __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 2, 0), 0, 3);
                    __builtin_prefetch(packed.interleavedB(chunk, kb + 1, 2, 2), 0, 3);
                    __builtin_prefetch(packed.chunkScales(chunk, kb + 1), 0, 3);
                }
                const Q8_1Block &a0 = r0[kb], &a1 = r1[kb], &a2 = r2[kb];
                __m512i i00 = _mm512_setzero_si512(), i01 = _mm512_setzero_si512(), i02 = _mm512_setzero_si512(), i03 = _mm512_setzero_si512();
                __m512i i10 = _mm512_setzero_si512(), i11 = _mm512_setzero_si512(), i12 = _mm512_setzero_si512(), i13 = _mm512_setzero_si512();
                __m512i i20 = _mm512_setzero_si512(), i21 = _mm512_setzero_si512(), i22 = _mm512_setzero_si512(), i23 = _mm512_setzero_si512();

                for (int group = 0; group < 4; ++group)
                {
                    const int goff = group * 4;
                    int32_t w0, w1, w2, h0, h1, h2;
                    std::memcpy(&w0, a0.qs + goff, 4); std::memcpy(&w1, a1.qs + goff, 4); std::memcpy(&w2, a2.qs + goff, 4);
                    std::memcpy(&h0, a0.qs + goff + 16, 4); std::memcpy(&h1, a1.qs + goff + 16, 4); std::memcpy(&h2, a2.qs + goff + 16, 4);
                    const __m512i al0 = _mm512_set1_epi32(w0), al1 = _mm512_set1_epi32(w1), al2 = _mm512_set1_epi32(w2);
                    const __m512i ah0 = _mm512_set1_epi32(h0), ah1 = _mm512_set1_epi32(h1), ah2 = _mm512_set1_epi32(h2);

#define LLAMINAR_UNIB3X4_Z(Z)                                                       \
                    {                                                               \
                        const __m512i raw = _mm512_load_si512(packed.interleavedB(chunk, kb, group, Z)); \
                        const __m512i lo = _mm512_and_si512(raw, mask_0F);          \
                        i0##Z = _mm512_dpbusd_epi32(i0##Z, lo, al0);                \
                        i1##Z = _mm512_dpbusd_epi32(i1##Z, lo, al1);                \
                        i2##Z = _mm512_dpbusd_epi32(i2##Z, lo, al2);                \
                        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask_0F); \
                        i0##Z = _mm512_dpbusd_epi32(i0##Z, hi, ah0);                \
                        i1##Z = _mm512_dpbusd_epi32(i1##Z, hi, ah1);                \
                        i2##Z = _mm512_dpbusd_epi32(i2##Z, hi, ah2);                \
                    }
                    LLAMINAR_UNIB3X4_Z(0); LLAMINAR_UNIB3X4_Z(1);
                    LLAMINAR_UNIB3X4_Z(2); LLAMINAR_UNIB3X4_Z(3);
#undef LLAMINAR_UNIB3X4_Z
                }
                if (decode_kind == NibbleDecodeKind::Q4_0_LINEAR)
                {
                    const __m512i u0 = _mm512_set1_epi32(static_cast<int32_t>(a0.sum_qs) * 8);
                    const __m512i u1 = _mm512_set1_epi32(static_cast<int32_t>(a1.sum_qs) * 8);
                    const __m512i u2 = _mm512_set1_epi32(static_cast<int32_t>(a2.sum_qs) * 8);
                    i00 = _mm512_sub_epi32(i00, u0); i01 = _mm512_sub_epi32(i01, u0); i02 = _mm512_sub_epi32(i02, u0); i03 = _mm512_sub_epi32(i03, u0);
                    i10 = _mm512_sub_epi32(i10, u1); i11 = _mm512_sub_epi32(i11, u1); i12 = _mm512_sub_epi32(i12, u1); i13 = _mm512_sub_epi32(i13, u1);
                    i20 = _mm512_sub_epi32(i20, u2); i21 = _mm512_sub_epi32(i21, u2); i22 = _mm512_sub_epi32(i22, u2); i23 = _mm512_sub_epi32(i23, u2);
                }

                const uint16_t *const sc_ptr = packed.chunkScales(chunk, kb);
                const float d0 = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float d1 = nativeQuantisedFP16ScaleToFP32(a1.d);
                const float d2 = nativeQuantisedFP16ScaleToFP32(a2.d);
                if constexpr (Asymmetric)
                {
                    const uint16_t *const mn_ptr = packed.chunkMins(chunk, kb);
#define LLAMINAR_UNIB3X4_ASYM(Z)                                                    \
                    {                                                               \
                        const __m512 sc = loadScales512(packed, sc_ptr + Z * 16);   \
                        const __m512 mn = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mn_ptr + Z * 16))); \
                        fp0##Z = accumulateCorrectedSingleScaleBlockAVX512(fp0##Z, i0##Z, sc, a0.sum_qs, mn, d0, Policy); \
                        fp1##Z = accumulateCorrectedSingleScaleBlockAVX512(fp1##Z, i1##Z, sc, a1.sum_qs, mn, d1, Policy); \
                        fp2##Z = accumulateCorrectedSingleScaleBlockAVX512(fp2##Z, i2##Z, sc, a2.sum_qs, mn, d2, Policy); \
                    }
                    LLAMINAR_UNIB3X4_ASYM(0); LLAMINAR_UNIB3X4_ASYM(1);
                    LLAMINAR_UNIB3X4_ASYM(2); LLAMINAR_UNIB3X4_ASYM(3);
#undef LLAMINAR_UNIB3X4_ASYM
                }
                else
                {
#define LLAMINAR_UNIB3X4_SYM(Z)                                                     \
                    {                                                               \
                        const __m512 sc = loadScales512(packed, sc_ptr + Z * 16);   \
                        fp0##Z = accumulateSingleScaleBlockAVX512(fp0##Z, i0##Z, sc, d0, Policy); \
                        fp1##Z = accumulateSingleScaleBlockAVX512(fp1##Z, i1##Z, sc, d1, Policy); \
                        fp2##Z = accumulateSingleScaleBlockAVX512(fp2##Z, i2##Z, sc, d2, Policy); \
                    }
                    LLAMINAR_UNIB3X4_SYM(0); LLAMINAR_UNIB3X4_SYM(1);
                    LLAMINAR_UNIB3X4_SYM(2); LLAMINAR_UNIB3X4_SYM(3);
#undef LLAMINAR_UNIB3X4_SYM
                }
            }

            if (k_tiles <= 1)
            {
                _mm512_store_ps(out_buf[0] + 0, fp00); _mm512_store_ps(out_buf[0] + 16, fp01);
                _mm512_store_ps(out_buf[0] + 32, fp02); _mm512_store_ps(out_buf[0] + 48, fp03);
                _mm512_store_ps(out_buf[1] + 0, fp10); _mm512_store_ps(out_buf[1] + 16, fp11);
                _mm512_store_ps(out_buf[1] + 32, fp12); _mm512_store_ps(out_buf[1] + 48, fp13);
                _mm512_store_ps(out_buf[2] + 0, fp20); _mm512_store_ps(out_buf[2] + 16, fp21);
                _mm512_store_ps(out_buf[2] + 32, fp22); _mm512_store_ps(out_buf[2] + 48, fp23);
            }
            else if (kt == 0)
            {
                _mm512_store_ps(red_buf[0] + 0, fp00); _mm512_store_ps(red_buf[0] + 16, fp01);
                _mm512_store_ps(red_buf[0] + 32, fp02); _mm512_store_ps(red_buf[0] + 48, fp03);
                _mm512_store_ps(red_buf[1] + 0, fp10); _mm512_store_ps(red_buf[1] + 16, fp11);
                _mm512_store_ps(red_buf[1] + 32, fp12); _mm512_store_ps(red_buf[1] + 48, fp13);
                _mm512_store_ps(red_buf[2] + 0, fp20); _mm512_store_ps(red_buf[2] + 16, fp21);
                _mm512_store_ps(red_buf[2] + 32, fp22); _mm512_store_ps(red_buf[2] + 48, fp23);
            }
            else
            {
                const bool last = (kt + 1 == k_tiles);
#define LLAMINAR_UNIB3X4_FOLD(R)                                                    \
                {                                                                   \
                    float *dst = last ? out_buf[R] : red_buf[R];                    \
                    _mm512_store_ps(dst + 0,  addOrderedAVX512(_mm512_load_ps(red_buf[R] + 0),  fp##R##0)); \
                    _mm512_store_ps(dst + 16, addOrderedAVX512(_mm512_load_ps(red_buf[R] + 16), fp##R##1)); \
                    _mm512_store_ps(dst + 32, addOrderedAVX512(_mm512_load_ps(red_buf[R] + 32), fp##R##2)); \
                    _mm512_store_ps(dst + 48, addOrderedAVX512(_mm512_load_ps(red_buf[R] + 48), fp##R##3)); \
                }
                LLAMINAR_UNIB3X4_FOLD(0); LLAMINAR_UNIB3X4_FOLD(1); LLAMINAR_UNIB3X4_FOLD(2);
#undef LLAMINAR_UNIB3X4_FOLD
            }
        }
    }

    inline void gemm_6row_unsigned_nibble_2z(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                gemm_6row_unsigned_nibble_2z_impl<true, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
            else
                gemm_6row_unsigned_nibble_2z_impl<true, CPUProjectionNumericalPolicy::BackendNative>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        }
        else if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            gemm_6row_unsigned_nibble_2z_impl<false, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        else
            gemm_6row_unsigned_nibble_2z_impl<false, CPUProjectionNumericalPolicy::BackendNative>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
    }

    inline void gemm_3row_unsigned_nibble_4z(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                gemm_3row_unsigned_nibble_4z_impl<true, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
            else
                gemm_3row_unsigned_nibble_4z_impl<true, CPUProjectionNumericalPolicy::BackendNative>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        }
        else if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            gemm_3row_unsigned_nibble_4z_impl<false, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        else
            gemm_3row_unsigned_nibble_4z_impl<false, CPUProjectionNumericalPolicy::BackendNative>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
    }

    /**
     * @brief Six-row 2z CompactFiveBit AVX-512 kernel (`Q5_0`, `Q5_1`, `Q5_K`).
     *
     * Halves activation broadcast overhead and weight passes versus 1z kernels across both
     * single-partition (`down_proj`) and multi-partition (`gate`/`up`) MoE geometries.
     */
    template <bool Asymmetric, CPUProjectionNumericalPolicy Policy>
    inline void sixRowsFiveBit2zAVX512(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i bit16 = _mm512_set1_epi8(0x10);
        const uint8_t *const base = packed.interleavedBase();
        const size_t stride = static_cast<size_t>(packed.interleaved_block_stride);
        const size_t chunk_origin = static_cast<size_t>(chunk) * packed.blocks_per_row;
        const Q8_1Block *const r0 = rows[0], *const r1 = rows[1], *const r2 = rows[2];
        const Q8_1Block *const r3 = rows[3], *const r4 = rows[4], *const r5 = rows[5];
        alignas(64) float red_buf[6][2][16];

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            for (int kt = 0; kt < k_tiles; ++kt)
            {
                const int kb_start = (k_tiles <= 1) ? 0 : kt * bpt;
                const int kb_end = (k_tiles <= 1) ? k_blocks : std::min(kb_start + bpt, k_blocks);
                __m512 fp00 = _mm512_setzero_ps(), fp01 = _mm512_setzero_ps();
                __m512 fp10 = _mm512_setzero_ps(), fp11 = _mm512_setzero_ps();
                __m512 fp20 = _mm512_setzero_ps(), fp21 = _mm512_setzero_ps();
                __m512 fp30 = _mm512_setzero_ps(), fp31 = _mm512_setzero_ps();
                __m512 fp40 = _mm512_setzero_ps(), fp41 = _mm512_setzero_ps();
                __m512 fp50 = _mm512_setzero_ps(), fp51 = _mm512_setzero_ps();

                for (int kb = kb_start; kb < kb_end; ++kb)
                {
                    if (zbase == 0 && kb + 2 < packed.blocks_per_row)
                        packed.prefetchInterleavedUnitT0(chunk, kb + 2);
                    const Q8_1Block &a0 = r0[kb], &a1 = r1[kb], &a2 = r2[kb];
                    const Q8_1Block &a3 = r3[kb], &a4 = r4[kb], &a5 = r5[kb];
                    const uint8_t *const unit = base + (chunk_origin + kb) * stride;

                    __m512i d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                    __m512i d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                    __m512i d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                    __m512i d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                    __m512i d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                    __m512i d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();

                    for (int group = 0; group < 4; ++group)
                    {
                        const __m512i raw0 = _mm512_load_si512(unit + static_cast<size_t>(group) * 256u + static_cast<size_t>(zbase) * 64u);
                        const __m512i raw1 = _mm512_load_si512(unit + static_cast<size_t>(group) * 256u + static_cast<size_t>(zbase + 1) * 64u);
                        const __m512i blo0 = fivebit_detail::mergeFiveBitMaskAVX512(
                            _mm512_and_si512(raw0, mask_0F),
                            unit + 1024u + static_cast<size_t>(group) * 32u + static_cast<size_t>(zbase) * 8u, bit16);
                        const __m512i blo1 = fivebit_detail::mergeFiveBitMaskAVX512(
                            _mm512_and_si512(raw1, mask_0F),
                            unit + 1024u + static_cast<size_t>(group) * 32u + static_cast<size_t>(zbase + 1) * 8u, bit16);
                        const int goff = group * 4;
#define LLAMINAR_FIVE6X2_STEP(R, ABLK, V0, V1, OFF)                                 \
                        {                                                           \
                            const __m512i av = _mm512_set1_epi32(fivebit_detail::packSignedActivationWord(ABLK, OFF)); \
                            d##R##0 = _mm512_dpbusd_epi32(d##R##0, V0, av);         \
                            d##R##1 = _mm512_dpbusd_epi32(d##R##1, V1, av);         \
                        }
                        LLAMINAR_FIVE6X2_STEP(0, a0, blo0, blo1, goff);
                        LLAMINAR_FIVE6X2_STEP(1, a1, blo0, blo1, goff);
                        LLAMINAR_FIVE6X2_STEP(2, a2, blo0, blo1, goff);
                        LLAMINAR_FIVE6X2_STEP(3, a3, blo0, blo1, goff);
                        LLAMINAR_FIVE6X2_STEP(4, a4, blo0, blo1, goff);
                        LLAMINAR_FIVE6X2_STEP(5, a5, blo0, blo1, goff);

                        const __m512i bhi0 = fivebit_detail::mergeFiveBitMaskAVX512(
                            _mm512_and_si512(_mm512_srli_epi16(raw0, 4), mask_0F),
                            unit + 1024u + static_cast<size_t>(group + 4) * 32u + static_cast<size_t>(zbase) * 8u, bit16);
                        const __m512i bhi1 = fivebit_detail::mergeFiveBitMaskAVX512(
                            _mm512_and_si512(_mm512_srli_epi16(raw1, 4), mask_0F),
                            unit + 1024u + static_cast<size_t>(group + 4) * 32u + static_cast<size_t>(zbase + 1) * 8u, bit16);
                        const int hioff = goff + 16;
                        LLAMINAR_FIVE6X2_STEP(0, a0, bhi0, bhi1, hioff);
                        LLAMINAR_FIVE6X2_STEP(1, a1, bhi0, bhi1, hioff);
                        LLAMINAR_FIVE6X2_STEP(2, a2, bhi0, bhi1, hioff);
                        LLAMINAR_FIVE6X2_STEP(3, a3, bhi0, bhi1, hioff);
                        LLAMINAR_FIVE6X2_STEP(4, a4, bhi0, bhi1, hioff);
                        LLAMINAR_FIVE6X2_STEP(5, a5, bhi0, bhi1, hioff);
#undef LLAMINAR_FIVE6X2_STEP
                    }

                    const uint16_t *const scales = reinterpret_cast<const uint16_t *>(unit + 1280u) + zbase * 16;
                    const __m512 sc0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales)));
                    const __m512 sc1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scales + 16)));
                    if constexpr (Asymmetric)
                    {
                        const uint16_t *const mins = reinterpret_cast<const uint16_t *>(unit + 1408u) + zbase * 16;
                        const __m512 mn0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins)));
                        const __m512 mn1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mins + 16)));
#define LLAMINAR_FIVE6X2_ASYM(R, ABLK)                                              \
                        {                                                           \
                            const float d = nativeQuantisedFP16ScaleToFP32(ABLK.d); \
                            fp##R##0 = fivebit_detail::applyFiveBitBlockAVX512<true, Policy>(fp##R##0, d##R##0, ABLK, d, sc0, mn0); \
                            fp##R##1 = fivebit_detail::applyFiveBitBlockAVX512<true, Policy>(fp##R##1, d##R##1, ABLK, d, sc1, mn1); \
                        }
                        LLAMINAR_FIVE6X2_ASYM(0, a0); LLAMINAR_FIVE6X2_ASYM(1, a1);
                        LLAMINAR_FIVE6X2_ASYM(2, a2); LLAMINAR_FIVE6X2_ASYM(3, a3);
                        LLAMINAR_FIVE6X2_ASYM(4, a4); LLAMINAR_FIVE6X2_ASYM(5, a5);
#undef LLAMINAR_FIVE6X2_ASYM
                    }
                    else
                    {
                        const __m512 zero = _mm512_setzero_ps();
#define LLAMINAR_FIVE6X2_SYM(R, ABLK)                                               \
                        {                                                           \
                            const float d = nativeQuantisedFP16ScaleToFP32(ABLK.d); \
                            fp##R##0 = fivebit_detail::applyFiveBitBlockAVX512<false, Policy>(fp##R##0, d##R##0, ABLK, d, sc0, zero); \
                            fp##R##1 = fivebit_detail::applyFiveBitBlockAVX512<false, Policy>(fp##R##1, d##R##1, ABLK, d, sc1, zero); \
                        }
                        LLAMINAR_FIVE6X2_SYM(0, a0); LLAMINAR_FIVE6X2_SYM(1, a1);
                        LLAMINAR_FIVE6X2_SYM(2, a2); LLAMINAR_FIVE6X2_SYM(3, a3);
                        LLAMINAR_FIVE6X2_SYM(4, a4); LLAMINAR_FIVE6X2_SYM(5, a5);
#undef LLAMINAR_FIVE6X2_SYM
                    }
                }

                if (k_tiles <= 1)
                {
                    _mm512_store_ps(out_buf[0] + zbase * 16, fp00); _mm512_store_ps(out_buf[0] + (zbase + 1) * 16, fp01);
                    _mm512_store_ps(out_buf[1] + zbase * 16, fp10); _mm512_store_ps(out_buf[1] + (zbase + 1) * 16, fp11);
                    _mm512_store_ps(out_buf[2] + zbase * 16, fp20); _mm512_store_ps(out_buf[2] + (zbase + 1) * 16, fp21);
                    _mm512_store_ps(out_buf[3] + zbase * 16, fp30); _mm512_store_ps(out_buf[3] + (zbase + 1) * 16, fp31);
                    _mm512_store_ps(out_buf[4] + zbase * 16, fp40); _mm512_store_ps(out_buf[4] + (zbase + 1) * 16, fp41);
                    _mm512_store_ps(out_buf[5] + zbase * 16, fp50); _mm512_store_ps(out_buf[5] + (zbase + 1) * 16, fp51);
                }
                else if (kt == 0)
                {
                    _mm512_store_ps(red_buf[0][0], fp00); _mm512_store_ps(red_buf[0][1], fp01);
                    _mm512_store_ps(red_buf[1][0], fp10); _mm512_store_ps(red_buf[1][1], fp11);
                    _mm512_store_ps(red_buf[2][0], fp20); _mm512_store_ps(red_buf[2][1], fp21);
                    _mm512_store_ps(red_buf[3][0], fp30); _mm512_store_ps(red_buf[3][1], fp31);
                    _mm512_store_ps(red_buf[4][0], fp40); _mm512_store_ps(red_buf[4][1], fp41);
                    _mm512_store_ps(red_buf[5][0], fp50); _mm512_store_ps(red_buf[5][1], fp51);
                }
                else
                {
                    const bool last = (kt + 1 == k_tiles);
#define LLAMINAR_FIVE6X2_FOLD(R)                                                    \
                    {                                                               \
                        const __m512 s0 = addOrderedAVX512(_mm512_load_ps(red_buf[R][0]), fp##R##0); \
                        const __m512 s1 = addOrderedAVX512(_mm512_load_ps(red_buf[R][1]), fp##R##1); \
                        if (last) {                                                 \
                            _mm512_store_ps(out_buf[R] + zbase * 16, s0);           \
                            _mm512_store_ps(out_buf[R] + (zbase + 1) * 16, s1);     \
                        } else {                                                    \
                            _mm512_store_ps(red_buf[R][0], s0);                     \
                            _mm512_store_ps(red_buf[R][1], s1);                     \
                        }                                                           \
                    }
                    LLAMINAR_FIVE6X2_FOLD(0); LLAMINAR_FIVE6X2_FOLD(1);
                    LLAMINAR_FIVE6X2_FOLD(2); LLAMINAR_FIVE6X2_FOLD(3);
                    LLAMINAR_FIVE6X2_FOLD(4); LLAMINAR_FIVE6X2_FOLD(5);
#undef LLAMINAR_FIVE6X2_FOLD
                }
            }
        }
    }

    inline void gemmCompactFiveBitSixRows2zAVX512(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        if (packed.is_asymmetric)
        {
            if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                sixRowsFiveBit2zAVX512<true, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
            else
                sixRowsFiveBit2zAVX512<true, CPUProjectionNumericalPolicy::BackendNative>(
                    packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        }
        else if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            sixRowsFiveBit2zAVX512<false, CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        else
            sixRowsFiveBit2zAVX512<false, CPUProjectionNumericalPolicy::BackendNative>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
    }

    /** @brief Three-row 4z single-pass UnsignedInt8 (`Q8_0`) AVX-512 kernel for grouped MoE strips. */
    template <CPUProjectionNumericalPolicy Policy>
    inline void gemm_3row_unsigned_int8_4z_impl(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        const uint8_t *const base = packed.interleavedBase();
        const size_t block_stride = static_cast<size_t>(packed.interleaved_block_stride);
        const size_t chunk_origin = static_cast<size_t>(chunk) * static_cast<size_t>(packed.blocks_per_row);
        const Q8_1Block *const r0 = rows[0], *const r1 = rows[1], *const r2 = rows[2];
        alignas(64) float red_buf[3][64];

        for (int kt = 0; kt < k_tiles; ++kt)
        {
            const int kb_start = (k_tiles <= 1) ? 0 : kt * bpt;
            const int kb_end = (k_tiles <= 1) ? k_blocks : std::min(kb_start + bpt, k_blocks);
            __m512 fp00 = _mm512_setzero_ps(), fp01 = _mm512_setzero_ps(), fp02 = _mm512_setzero_ps(), fp03 = _mm512_setzero_ps();
            __m512 fp10 = _mm512_setzero_ps(), fp11 = _mm512_setzero_ps(), fp12 = _mm512_setzero_ps(), fp13 = _mm512_setzero_ps();
            __m512 fp20 = _mm512_setzero_ps(), fp21 = _mm512_setzero_ps(), fp22 = _mm512_setzero_ps(), fp23 = _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = r0[kb], &a1 = r1[kb], &a2 = r2[kb];
                const uint8_t *const unit = base + (chunk_origin + static_cast<size_t>(kb)) * block_stride;
                __m512i i00 = _mm512_setzero_si512(), i01 = _mm512_setzero_si512(), i02 = _mm512_setzero_si512(), i03 = _mm512_setzero_si512();
                __m512i i10 = _mm512_setzero_si512(), i11 = _mm512_setzero_si512(), i12 = _mm512_setzero_si512(), i13 = _mm512_setzero_si512();
                __m512i i20 = _mm512_setzero_si512(), i21 = _mm512_setzero_si512(), i22 = _mm512_setzero_si512(), i23 = _mm512_setzero_si512();

                for (int g = 0; g < 8; ++g)
                {
                    int32_t w0, w1, w2;
                    std::memcpy(&w0, a0.qs + g * 4, 4); std::memcpy(&w1, a1.qs + g * 4, 4); std::memcpy(&w2, a2.qs + g * 4, 4);
                    const __m512i a0_bc = _mm512_set1_epi32(w0), a1_bc = _mm512_set1_epi32(w1), a2_bc = _mm512_set1_epi32(w2);
                    const uint8_t *const gp = unit + static_cast<size_t>(g) * 256u;
                    const __m512i b0 = _mm512_load_si512(gp + 0);
                    i00 = _mm512_dpbusd_epi32(i00, b0, a0_bc); i10 = _mm512_dpbusd_epi32(i10, b0, a1_bc); i20 = _mm512_dpbusd_epi32(i20, b0, a2_bc);
                    const __m512i b1 = _mm512_load_si512(gp + 64);
                    i01 = _mm512_dpbusd_epi32(i01, b1, a0_bc); i11 = _mm512_dpbusd_epi32(i11, b1, a1_bc); i21 = _mm512_dpbusd_epi32(i21, b1, a2_bc);
                    const __m512i b2 = _mm512_load_si512(gp + 128);
                    i02 = _mm512_dpbusd_epi32(i02, b2, a0_bc); i12 = _mm512_dpbusd_epi32(i12, b2, a1_bc); i22 = _mm512_dpbusd_epi32(i22, b2, a2_bc);
                    const __m512i b3 = _mm512_load_si512(gp + 192);
                    i03 = _mm512_dpbusd_epi32(i03, b3, a0_bc); i13 = _mm512_dpbusd_epi32(i13, b3, a1_bc); i23 = _mm512_dpbusd_epi32(i23, b3, a2_bc);
                }
                const __m512i ub0 = _mm512_set1_epi32(static_cast<int32_t>(a0.sum_qs) * 128);
                const __m512i ub1 = _mm512_set1_epi32(static_cast<int32_t>(a1.sum_qs) * 128);
                const __m512i ub2 = _mm512_set1_epi32(static_cast<int32_t>(a2.sum_qs) * 128);
                const uint16_t *const sc_ptr = packed.chunkScales(chunk, kb);
                const float d0 = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float d1 = nativeQuantisedFP16ScaleToFP32(a1.d);
                const float d2 = nativeQuantisedFP16ScaleToFP32(a2.d);
                const __m512 s0 = loadScales512(packed, sc_ptr + 0);
                fp00 = accumulateSingleScaleBlockAVX512(fp00, _mm512_sub_epi32(i00, ub0), s0, d0, Policy);
                fp10 = accumulateSingleScaleBlockAVX512(fp10, _mm512_sub_epi32(i10, ub1), s0, d1, Policy);
                fp20 = accumulateSingleScaleBlockAVX512(fp20, _mm512_sub_epi32(i20, ub2), s0, d2, Policy);
                const __m512 s1 = loadScales512(packed, sc_ptr + 16);
                fp01 = accumulateSingleScaleBlockAVX512(fp01, _mm512_sub_epi32(i01, ub0), s1, d0, Policy);
                fp11 = accumulateSingleScaleBlockAVX512(fp11, _mm512_sub_epi32(i11, ub1), s1, d1, Policy);
                fp21 = accumulateSingleScaleBlockAVX512(fp21, _mm512_sub_epi32(i21, ub2), s1, d2, Policy);
                const __m512 s2 = loadScales512(packed, sc_ptr + 32);
                fp02 = accumulateSingleScaleBlockAVX512(fp02, _mm512_sub_epi32(i02, ub0), s2, d0, Policy);
                fp12 = accumulateSingleScaleBlockAVX512(fp12, _mm512_sub_epi32(i12, ub1), s2, d1, Policy);
                fp22 = accumulateSingleScaleBlockAVX512(fp22, _mm512_sub_epi32(i22, ub2), s2, d2, Policy);
                const __m512 s3 = loadScales512(packed, sc_ptr + 48);
                fp03 = accumulateSingleScaleBlockAVX512(fp03, _mm512_sub_epi32(i03, ub0), s3, d0, Policy);
                fp13 = accumulateSingleScaleBlockAVX512(fp13, _mm512_sub_epi32(i13, ub1), s3, d1, Policy);
                fp23 = accumulateSingleScaleBlockAVX512(fp23, _mm512_sub_epi32(i23, ub2), s3, d2, Policy);
            }

            if (k_tiles <= 1)
            {
                _mm512_store_ps(out_buf[0] + 0, fp00); _mm512_store_ps(out_buf[0] + 16, fp01);
                _mm512_store_ps(out_buf[0] + 32, fp02); _mm512_store_ps(out_buf[0] + 48, fp03);
                _mm512_store_ps(out_buf[1] + 0, fp10); _mm512_store_ps(out_buf[1] + 16, fp11);
                _mm512_store_ps(out_buf[1] + 32, fp12); _mm512_store_ps(out_buf[1] + 48, fp13);
                _mm512_store_ps(out_buf[2] + 0, fp20); _mm512_store_ps(out_buf[2] + 16, fp21);
                _mm512_store_ps(out_buf[2] + 32, fp22); _mm512_store_ps(out_buf[2] + 48, fp23);
            }
            else if (kt == 0)
            {
                _mm512_store_ps(red_buf[0] + 0, fp00); _mm512_store_ps(red_buf[0] + 16, fp01);
                _mm512_store_ps(red_buf[0] + 32, fp02); _mm512_store_ps(red_buf[0] + 48, fp03);
                _mm512_store_ps(red_buf[1] + 0, fp10); _mm512_store_ps(red_buf[1] + 16, fp11);
                _mm512_store_ps(red_buf[1] + 32, fp12); _mm512_store_ps(red_buf[1] + 48, fp13);
                _mm512_store_ps(red_buf[2] + 0, fp20); _mm512_store_ps(red_buf[2] + 16, fp21);
                _mm512_store_ps(red_buf[2] + 32, fp22); _mm512_store_ps(red_buf[2] + 48, fp23);
            }
            else
            {
                const bool last = (kt + 1 == k_tiles);
                float *d0 = last ? out_buf[0] : red_buf[0];
                float *d1 = last ? out_buf[1] : red_buf[1];
                float *d2 = last ? out_buf[2] : red_buf[2];
                _mm512_store_ps(d0 + 0,  addOrderedAVX512(_mm512_load_ps(red_buf[0] + 0),  fp00));
                _mm512_store_ps(d0 + 16, addOrderedAVX512(_mm512_load_ps(red_buf[0] + 16), fp01));
                _mm512_store_ps(d0 + 32, addOrderedAVX512(_mm512_load_ps(red_buf[0] + 32), fp02));
                _mm512_store_ps(d0 + 48, addOrderedAVX512(_mm512_load_ps(red_buf[0] + 48), fp03));
                _mm512_store_ps(d1 + 0,  addOrderedAVX512(_mm512_load_ps(red_buf[1] + 0),  fp10));
                _mm512_store_ps(d1 + 16, addOrderedAVX512(_mm512_load_ps(red_buf[1] + 16), fp11));
                _mm512_store_ps(d1 + 32, addOrderedAVX512(_mm512_load_ps(red_buf[1] + 32), fp12));
                _mm512_store_ps(d1 + 48, addOrderedAVX512(_mm512_load_ps(red_buf[1] + 48), fp13));
                _mm512_store_ps(d2 + 0,  addOrderedAVX512(_mm512_load_ps(red_buf[2] + 0),  fp20));
                _mm512_store_ps(d2 + 16, addOrderedAVX512(_mm512_load_ps(red_buf[2] + 16), fp21));
                _mm512_store_ps(d2 + 32, addOrderedAVX512(_mm512_load_ps(red_buf[2] + 32), fp22));
                _mm512_store_ps(d2 + 48, addOrderedAVX512(_mm512_load_ps(red_buf[2] + 48), fp23));
            }
        }
    }

    inline void gemm_3row_unsigned_int8_4z(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *const *rows,
        float (*out_buf)[64],
        int chunk,
        int k_blocks,
        int k_tiles,
        int bpt) noexcept
    {
        if (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            gemm_3row_unsigned_int8_4z_impl<CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
        else
            gemm_3row_unsigned_int8_4z_impl<CPUProjectionNumericalPolicy::BackendNative>(
                packed, rows, out_buf, chunk, k_blocks, k_tiles, bpt);
    }
#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    /**
     * @brief Compute one 64-column chunk for a strip of `strip_rows <= 32` rows
     *        in exact ascending K-partition FP32 reduction order.
     *
     * Keeps each `(chunk, k_tile)` weight slice L1d-resident across the entire
     * row strip and dispatches across all 18 NativeQuantised codebooks and
     * runtime `AVX512 / AVX2 / Scalar` ISAs.
     */
    inline void runStripChunkOrdered(
        const PreparedMatrixInfo &m,
        const Q8_1Block *const *row_ptrs,
        int strip_rows,
        float (*out_buf)[kChunkCols],
        int chunk,
        int cols,
        ISALevel isa) noexcept
    {
        const auto &packed = *m.packed;
        const int k_blocks = m.k_blocks;
        const int k_tiles = m.k_tiles;
        const int bpt = m.k_blocks_per_tile;
        const bool bounded = (cols < kChunkCols);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (isa >= ISALevel::AVX512)
        {
            if (m.is_compact_ms)
            {
                int r = 0;
                if (k_tiles <= 1)
                {
                    for (; r + 4 <= strip_rows; r += 4)
                        multi_scale::dispatch<16>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2], row_ptrs[r + 3]},
                            std::array{out_buf[r], out_buf[r + 1], out_buf[r + 2], out_buf[r + 3]},
                            chunk, 0, k_blocks, false, cols);
                    if (r + 3 <= strip_rows)
                    {
                        multi_scale::dispatch<16>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2]},
                            std::array{out_buf[r], out_buf[r + 1], out_buf[r + 2]},
                            chunk, 0, k_blocks, false, cols);
                        r += 3;
                    }
                    if (r + 2 <= strip_rows)
                    {
                        multi_scale::dispatch<16>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1]},
                            std::array{out_buf[r], out_buf[r + 1]},
                            chunk, 0, k_blocks, false, cols);
                        r += 2;
                    }
                    for (; r < strip_rows; ++r)
                        multi_scale::dispatch<16>(packed,
                            std::array{row_ptrs[r]}, std::array{out_buf[r]},
                            chunk, 0, k_blocks, false, cols);
                }
                else
                {
                    for (; r + 4 <= strip_rows; r += 4)
                        multi_scale::dispatchOrdered<16, 4>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2], row_ptrs[r + 3]},
                            std::array{out_buf[r], out_buf[r + 1], out_buf[r + 2], out_buf[r + 3]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                    if (r + 3 <= strip_rows)
                    {
                        multi_scale::dispatchOrdered<16, 3>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2]},
                            std::array{out_buf[r], out_buf[r + 1], out_buf[r + 2]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                        r += 3;
                    }
                    if (r + 2 <= strip_rows)
                    {
                        multi_scale::dispatchOrdered<16, 2>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1]},
                            std::array{out_buf[r], out_buf[r + 1]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                        r += 2;
                    }
                    for (; r < strip_rows; ++r)
                        multi_scale::dispatchOrdered<16, 1>(packed,
                            std::array{row_ptrs[r]}, std::array{out_buf[r]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                }
                return;
            }

            // Register-resident 2z/4z wide-row fast paths for UnsignedNibble (Q4_K/Q4_0/Q4_1),
            // CompactFiveBit (Q5_0/Q5_1/Q5_K), and symmetric UnsignedInt8 (Q8_0): share each
            // 32-bit activation broadcast across 2 or 4 ZMM weight columns and handle both
            // single-partition and ordered multi-partition geometries without stack folds.
            int row_offset = 0;
            if (m.is_unsigned_nibble)
            {
                for (; row_offset + 6 <= strip_rows; row_offset += 6)
                    gemm_6row_unsigned_nibble_2z(
                        packed, row_ptrs + row_offset, out_buf + row_offset, chunk, k_blocks, k_tiles, bpt);
                if (row_offset + 3 <= strip_rows)
                {
                    gemm_3row_unsigned_nibble_4z(
                        packed, row_ptrs + row_offset, out_buf + row_offset, chunk, k_blocks, k_tiles, bpt);
                    row_offset += 3;
                }
                if (row_offset == strip_rows)
                {
                    (void)bounded;
                    return;
                }
            }
            else if (m.is_compact_five_bit)
            {
                for (; row_offset + 6 <= strip_rows; row_offset += 6)
                    gemmCompactFiveBitSixRows2zAVX512(
                        packed, row_ptrs + row_offset, out_buf + row_offset, chunk, k_blocks, k_tiles, bpt);
                if (row_offset == strip_rows)
                {
                    (void)bounded;
                    return;
                }
            }
            else if (m.is_unsigned_int8)
            {
                for (; row_offset + 3 <= strip_rows; row_offset += 3)
                    gemm_3row_unsigned_int8_4z(
                        packed, row_ptrs + row_offset, out_buf + row_offset, chunk, k_blocks, k_tiles, bpt);
                if (row_offset == strip_rows)
                {
                    (void)bounded;
                    return;
                }
            }

            // Remaining rows (< 8 or < 6 for fast paths, or full strip for Q8_0/Q6_K/ExpandedInt8):
            // K-tile outer keeps each weight slice L1d-hot across the row tail and folds in exact order.
            alignas(64) float p0[kChunkCols], p1[kChunkCols], p2[kChunkCols], p3[kChunkCols];
            for (int kt = 0; kt < k_tiles; ++kt)
            {
                const int kb_start = kt * bpt;
                const int kb_end = std::min(kb_start + bpt, k_blocks);
                const bool first = (kt == 0);
                int r = row_offset;
                if (m.is_nibble)
                {
                    for (; r + 4 <= strip_rows; r += 4)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        float *d2 = first ? out_buf[r + 2] : p2;
                        float *d3 = first ? out_buf[r + 3] : p3;
                        gemm_4row_native_1z_chunk<false>(
                            packed, row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2], row_ptrs[r + 3],
                            d0, d1, d2, d3, chunk, kb_start, kb_end, m.lut512, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 2], p2, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 3], p3, false, kChunkCols);
                        }
                    }
                    if (r + 3 <= strip_rows)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        float *d2 = first ? out_buf[r + 2] : p2;
                        gemm_3row_native_2z_chunk<false>(
                            packed, row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2],
                            d0, d1, d2, chunk, kb_start, kb_end, m.lut512, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 2], p2, false, kChunkCols);
                        }
                        r += 3;
                    }
                    if (r + 2 <= strip_rows)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        gemm_2row_native_chunk<false>(
                            packed, row_ptrs[r], row_ptrs[r + 1],
                            d0, d1, chunk, kb_start, kb_end, m.lut512, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                        }
                        r += 2;
                    }
                    for (; r < strip_rows; ++r)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        gemv_native_quantised_avx512_chunk_native<false>(
                            packed, row_ptrs[r], d0, chunk, kb_start, kb_end, m.lut512, false, kChunkCols);
                        if (!first)
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                    }
                }
                else
                {
                    for (; r + 4 <= strip_rows; r += 4)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        float *d2 = first ? out_buf[r + 2] : p2;
                        float *d3 = first ? out_buf[r + 3] : p3;
                        gemm_4row_non_nibble_2z_chunk<false, false>(
                            packed, row_ptrs[r], row_ptrs[r + 1], row_ptrs[r + 2], row_ptrs[r + 3],
                            d0, d1, d2, d3, chunk, kb_start, kb_end, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 2], p2, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 3], p3, false, kChunkCols);
                        }
                    }
                    if (r + 2 <= strip_rows)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        gemm_2row_non_nibble_chunk<false, false>(
                            packed, row_ptrs[r], row_ptrs[r + 1],
                            d0, d1, chunk, kb_start, kb_end, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX512Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                        }
                        r += 2;
                    }
                    for (; r < strip_rows; ++r)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        gemv_native_quantised_avx512_chunk_non_nibble<false>(
                            packed, row_ptrs[r], d0, chunk, kb_start, kb_end, false, kChunkCols);
                        if (!first)
                            foldOrderedAVX512Partition<false>(out_buf[r], p0, false, kChunkCols);
                    }
                }
            }
            (void)bounded;
            return;
        }
#endif

#if defined(__AVX2__)
        if (isa >= ISALevel::AVX2)
        {
            if (m.is_compact_ms)
            {
                int r = 0;
                if (k_tiles <= 1)
                {
                    for (; r + 2 <= strip_rows; r += 2)
                        multi_scale::dispatch<8>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1]},
                            std::array{out_buf[r], out_buf[r + 1]},
                            chunk, 0, k_blocks, false, cols);
                    for (; r < strip_rows; ++r)
                        multi_scale::dispatch<8>(packed,
                            std::array{row_ptrs[r]}, std::array{out_buf[r]},
                            chunk, 0, k_blocks, false, cols);
                }
                else
                {
                    for (; r + 2 <= strip_rows; r += 2)
                        multi_scale::dispatchOrdered<8, 2>(packed,
                            std::array{row_ptrs[r], row_ptrs[r + 1]},
                            std::array{out_buf[r], out_buf[r + 1]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                    for (; r < strip_rows; ++r)
                        multi_scale::dispatchOrdered<8, 1>(packed,
                            std::array{row_ptrs[r]}, std::array{out_buf[r]},
                            chunk, k_blocks, k_tiles, bpt, cols);
                }
                return;
            }

            alignas(32) float p0[kChunkCols], p1[kChunkCols];
            for (int kt = 0; kt < k_tiles; ++kt)
            {
                const int kb_start = kt * bpt;
                const int kb_end = std::min(kb_start + bpt, k_blocks);
                const bool first = (kt == 0);
                int r = 0;
                if (m.is_nibble)
                {
                    for (; r + 2 <= strip_rows; r += 2)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        gemm_2row_native_chunk_avx2<false>(
                            packed, row_ptrs[r], row_ptrs[r + 1],
                            d0, d1, chunk, kb_start, kb_end, m.lut256, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX2Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX2Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                        }
                    }
                    for (; r < strip_rows; ++r)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        gemv_avx2_chunk_native<false>(
                            packed, row_ptrs[r], d0, chunk, kb_start, kb_end, m.lut256, false, kChunkCols);
                        if (!first)
                            foldOrderedAVX2Partition<false>(out_buf[r], p0, false, kChunkCols);
                    }
                }
                else
                {
                    for (; r + 2 <= strip_rows; r += 2)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        float *d1 = first ? out_buf[r + 1] : p1;
                        gemm_2row_non_nibble_chunk_avx2<false, false>(
                            packed, row_ptrs[r], row_ptrs[r + 1],
                            d0, d1, chunk, kb_start, kb_end, false, kChunkCols);
                        if (!first)
                        {
                            foldOrderedAVX2Partition<false>(out_buf[r], p0, false, kChunkCols);
                            foldOrderedAVX2Partition<false>(out_buf[r + 1], p1, false, kChunkCols);
                        }
                    }
                    for (; r < strip_rows; ++r)
                    {
                        float *d0 = first ? out_buf[r] : p0;
                        gemv_avx2_chunk_non_nibble<false>(
                            packed, row_ptrs[r], d0, chunk, kb_start, kb_end, false, kChunkCols);
                        if (!first)
                            foldOrderedAVX2Partition<false>(out_buf[r], p0, false, kChunkCols);
                    }
                }
            }
            (void)bounded;
            return;
        }
#endif

        // Scalar ISA: exact device_fp32_contract K-partitioned reference.
        (void)isa;
        (void)bounded;
        alignas(64) float partial[kChunkCols];
        for (int kt = 0; kt < k_tiles; ++kt)
        {
            const int kb_start = kt * bpt;
            const int kb_end = std::min(kb_start + bpt, k_blocks);
            const bool first = (kt == 0);
            for (int r = 0; r < strip_rows; ++r)
            {
                float *dst = first ? out_buf[r] : partial;
                gemv_native_quantised_scalar_chunk(packed, row_ptrs[r], dst, chunk, kb_start, kb_end, cols);
                if (!first)
                {
                    for (int c = 0; c < cols; ++c)
                        out_buf[r][c] = device_fp32_contract::add(out_buf[r][c], partial[c]);
                }
            }
        }
    }

    inline void quantizeSwiGLUChunkToQ8(
        const float *gate_vals,
        const float *up_vals,
        Q8_1Block *dst_q8,
        int cols,
        CPUProjectionNumericalPolicy policy,
        ISALevel isa) noexcept
    {
        alignas(64) float swiglu_vals[kChunkCols];
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            primitives::compute_swiglu_gpu_aligned_expert_serial(gate_vals, up_vals, swiglu_vals, cols);
        else
            primitives::compute_swiglu_serial(gate_vals, up_vals, swiglu_vals, cols);

#if defined(__AVX512F__)
        if (cols == 64 && isa >= ISALevel::AVX512)
        {
            gpu_aligned_expert_q8::quantizeTwoBlocks(swiglu_vals, dst_q8[0], dst_q8[1]);
            return;
        }
#endif
        (void)isa;
        const int blocks = (cols + static_cast<int>(Q8_1Block::BLOCK_SIZE) - 1) /
                           static_cast<int>(Q8_1Block::BLOCK_SIZE);
        for (int b = 0; b < blocks; ++b)
        {
            const int off = b * static_cast<int>(Q8_1Block::BLOCK_SIZE);
            const int valid = std::min(static_cast<int>(Q8_1Block::BLOCK_SIZE), cols - off);
            gpu_aligned_expert_q8::quantizeBlock(swiglu_vals + off, dst_q8[b], valid);
        }
    }
} // namespace

bool CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn(
    std::span<const BatchedPrequantizedProjectionDesc> gate_up_descriptors,
    const Q8_1Block *unique_router_q8,
    int unique_router_rows,
    int router_q8_blocks_per_row,
    std::span<const int> router_row_indices,
    int hidden_width,
    Q8_1Block *activation_q8,
    int route_rows,
    int intermediate,
    int activation_blocks_per_row,
    std::span<const BatchedPrequantizedProjectionDesc> down_descriptors,
    std::optional<GroupedDownRouteOrder> down_route_order)
{
    const size_t num_experts = down_descriptors.size();
    if (num_experts == 0u || gate_up_descriptors.size() != 2u * num_experts ||
        !unique_router_q8 || unique_router_rows <= 0 || router_q8_blocks_per_row <= 0 ||
        !activation_q8 || route_rows <= 0 || hidden_width <= 0 || intermediate <= 0 ||
        activation_blocks_per_row <= 0 ||
        router_row_indices.size() != static_cast<size_t>(route_rows))
    {
        LOG_ERROR("[CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn] invalid descriptor or buffer contract");
        return false;
    }

    int summed_rows = 0;
    for (size_t e = 0; e < num_experts; ++e)
    {
        const auto &g = gate_up_descriptors[2u * e];
        const auto &u = gate_up_descriptors[2u * e + 1u];
        const auto &d = down_descriptors[e];
        if (!g.kernel || !g.kernel->valid_ || g.kernel->activation_rotation_ || g.bias ||
            !u.kernel || !u.kernel->valid_ || u.kernel->activation_rotation_ || u.bias ||
            !d.kernel || !d.kernel->valid_ || d.kernel->activation_rotation_ || d.bias ||
            g.rows <= 0 || g.rows != u.rows || g.rows != d.rows ||
            g.n != intermediate || u.n != intermediate || d.n != hidden_width ||
            g.kernel->packed_.K != hidden_width || g.kernel->packed_.N != intermediate ||
            u.kernel->packed_.K != hidden_width || u.kernel->packed_.N != intermediate ||
            d.kernel->packed_.K != intermediate || d.kernel->packed_.N != hidden_width ||
            !d.output || d.ldc < d.n ||
            !d.output_rows.matches(d.output, d.ldc, d.n, d.rows))
        {
            LOG_ERROR("[CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn] expert " << e
                      << " violates the eager pre-quantized fused contract");
            return false;
        }
        summed_rows += d.rows;
    }
    if (summed_rows != route_rows)
    {
        LOG_ERROR("[CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn] summed expert rows ("
                  << summed_rows << ") != route_rows (" << route_rows << ")");
        return false;
    }
    for (int idx : router_row_indices)
    {
        if (idx < 0 || idx >= unique_router_rows)
        {
            LOG_ERROR("[CPUNativeQuantisedGemmKernel::execute_moe_grouped_fused_ffn] router_row_index out of bounds");
            return false;
        }
    }

    if (down_route_order)
        requireGroupedDownRoutePartition(down_descriptors, route_rows, *down_route_order);
    else
        requireDisjointIndexedBatchedOutputs(down_descriptors);

    const ISALevel isa = activeISALevel();
    const int workers = std::max(1, nativeQuantisedInvocationWorkerCount());
    const auto &cache = CPUExecutionGeometry::local().cache;
    const int gate_n_chunks = (intermediate + kChunkCols - 1) / kChunkCols;
    const int down_n_chunks = (hidden_width + kChunkCols - 1) / kChunkCols;
    const CPUProjectionNumericalPolicy swiglu_policy = gate_up_descriptors[0].kernel->numerical_policy_;

    // Retained thread-local task buffers avoid any hot-path heap allocation after warmup.
    thread_local std::vector<PreparedMatrixInfo> gate_info;
    thread_local std::vector<PreparedMatrixInfo> up_info;
    thread_local std::vector<PreparedMatrixInfo> down_info;
    thread_local std::vector<int> expert_begins;
    thread_local std::vector<Phase1StripTask> phase1_tasks;
    thread_local std::vector<Phase2StripTask> phase2_tasks;
    thread_local std::vector<SinglePassExpertTask> single_pass_tasks;

    gate_info.resize(num_experts);
    up_info.resize(num_experts);
    down_info.resize(num_experts);
    expert_begins.resize(num_experts);

    // When active expert strips comfortably exceed worker count (e.g. 512 experts
    // in prefill or ~60+ experts in M=8 verifier on 28 cores), fuse Gate + Up ->
    // SwiGLU -> Q8_1 -> Down into a single barrier-free per-expert pass. Each core
    // keeps `in_rows` L1d-hot across Gate/Up chunks and keeps `activation_q8` L1d-hot
    // directly into sequential Down chunks, eliminating 24x dynamic task contention
    // and the inter-phase barrier.
    const bool use_single_pass_expert_fused =
        (num_experts >= static_cast<size_t>(std::max(2, workers * 2)));

    std::array<TileConfigCacheEntry, 16> tile_cache{};
    size_t tile_cache_count = 0;

    if (use_single_pass_expert_fused)
    {
        single_pass_tasks.clear();
        int row_cursor = 0;
        for (size_t e = 0; e < num_experts; ++e)
        {
            expert_begins[e] = row_cursor;
            gate_info[e] = prepareMatrixInfoCached(
                gate_up_descriptors[2u * e].kernel->packed_, intermediate, hidden_width, workers, cache, tile_cache, tile_cache_count);
            up_info[e] = prepareMatrixInfoCached(
                gate_up_descriptors[2u * e + 1u].kernel->packed_, intermediate, hidden_width, workers, cache, tile_cache, tile_cache_count);
            down_info[e] = prepareMatrixInfoCached(
                down_descriptors[e].kernel->packed_, hidden_width, intermediate, workers, cache, tile_cache, tile_cache_count);

            const int rows = down_descriptors[e].rows;
            for (int strip_begin = 0; strip_begin < rows; strip_begin += kStripMaxRows)
            {
                const int strip_rows = std::min(kStripMaxRows, rows - strip_begin);
                single_pass_tasks.push_back({static_cast<int>(e), strip_begin, strip_rows});
            }
            row_cursor += rows;
        }

        std::stable_sort(single_pass_tasks.begin(), single_pass_tasks.end(),
            [](const SinglePassExpertTask &a, const SinglePassExpertTask &b) { return a.strip_rows > b.strip_rows; });

        const PreparedMatrixInfo *const gate_info_ptr = gate_info.data();
        const PreparedMatrixInfo *const up_info_ptr = up_info.data();
        const PreparedMatrixInfo *const down_info_ptr = down_info.data();
        const int *const expert_begins_ptr = expert_begins.data();
        const BatchedPrequantizedProjectionDesc *const down_desc_ptr = down_descriptors.data();
        const int *const row_idx_ptr = router_row_indices.data();
        const size_t n_tasks = single_pass_tasks.size();
        const SinglePassExpertTask *const tasks_ptr = single_pass_tasks.data();

        auto run_single_pass = [=]()
        {
#pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < n_tasks; ++t)
            {
                const auto &task = tasks_ptr[t];
                const int e = task.expert_idx;
                const int sr = task.strip_rows;
                const int base_route = expert_begins_ptr[static_cast<size_t>(e)] + task.strip_begin;

                const Q8_1Block *in_rows[kStripMaxRows];
                for (int r = 0; r < sr; ++r)
                {
                    const int unique_row = row_idx_ptr[base_route + r];
                    in_rows[r] = unique_router_q8 +
                                 static_cast<size_t>(unique_row) * static_cast<size_t>(router_q8_blocks_per_row);
                }

                alignas(64) float gate_buf[kStripMaxRows][kChunkCols];
                alignas(64) float up_buf[kStripMaxRows][kChunkCols];
                const auto &g_info = gate_info_ptr[static_cast<size_t>(e)];
                const auto &u_info = up_info_ptr[static_cast<size_t>(e)];
                for (int chunk = 0; chunk < gate_n_chunks; ++chunk)
                {
                    const int cols = std::min(kChunkCols, intermediate - chunk * kChunkCols);
                    runStripChunkOrdered(g_info, in_rows, sr, gate_buf, chunk, cols, isa);
                    runStripChunkOrdered(u_info, in_rows, sr, up_buf, chunk, cols, isa);

                    const int q8_chunk_offset = chunk * 2;
                    for (int r = 0; r < sr; ++r)
                    {
                        Q8_1Block *dst_q8 = activation_q8 +
                            static_cast<size_t>(base_route + r) * static_cast<size_t>(activation_blocks_per_row) +
                            q8_chunk_offset;
                        quantizeSwiGLUChunkToQ8(gate_buf[r], up_buf[r], dst_q8, cols, swiglu_policy, isa);
                    }
                }

                // Down consumes the exact freshly published L1d-hot Q8_1 rows immediately.
                const auto &d_desc = down_desc_ptr[static_cast<size_t>(e)];
                const auto &d_info = down_info_ptr[static_cast<size_t>(e)];
                const Q8_1Block *const swiglu_base =
                    d_desc.input_q8 + static_cast<size_t>(task.strip_begin) * static_cast<size_t>(activation_blocks_per_row);
                const Q8_1Block *swiglu_rows[kStripMaxRows];
                for (int r = 0; r < sr; ++r)
                    swiglu_rows[r] = swiglu_base + static_cast<size_t>(r) * static_cast<size_t>(activation_blocks_per_row);

                alignas(64) float out_buf[kStripMaxRows][kChunkCols];
                for (int chunk = 0; chunk < down_n_chunks; ++chunk)
                {
                    const int cols = std::min(kChunkCols, hidden_width - chunk * kChunkCols);
                    runStripChunkOrdered(d_info, swiglu_rows, sr, out_buf, chunk, cols, isa);
                    const int col_offset = chunk * kChunkCols;
                    for (int r = 0; r < sr; ++r)
                    {
                        float *dst = d_desc.output_rows.rowAddress(
                                         d_desc.output,
                                         d_desc.ldc,
                                         static_cast<size_t>(task.strip_begin + r)) +
                                     col_offset;
                        std::memcpy(dst, out_buf[r], static_cast<size_t>(cols) * sizeof(float));
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(run_single_pass);
    }
    else
    {
        phase1_tasks.clear();
        phase2_tasks.clear();

        int row_cursor = 0;
        for (size_t e = 0; e < num_experts; ++e)
        {
            expert_begins[e] = row_cursor;
            gate_info[e] = prepareMatrixInfoCached(
                gate_up_descriptors[2u * e].kernel->packed_, intermediate, hidden_width, workers, cache, tile_cache, tile_cache_count);
            up_info[e] = prepareMatrixInfoCached(
                gate_up_descriptors[2u * e + 1u].kernel->packed_, intermediate, hidden_width, workers, cache, tile_cache, tile_cache_count);
            down_info[e] = prepareMatrixInfoCached(
                down_descriptors[e].kernel->packed_, hidden_width, intermediate, workers, cache, tile_cache, tile_cache_count);

            const int rows = down_descriptors[e].rows;
            for (int strip_begin = 0; strip_begin < rows; strip_begin += kStripMaxRows)
            {
                const int strip_rows = std::min(kStripMaxRows, rows - strip_begin);
                for (int c = 0; c < gate_n_chunks; ++c)
                    phase1_tasks.push_back({static_cast<int>(e), c, strip_begin, strip_rows});
                for (int c = 0; c < down_n_chunks; c += 2)
                    phase2_tasks.push_back({static_cast<int>(e), c, std::min(2, down_n_chunks - c), strip_begin, strip_rows});
            }
            row_cursor += rows;
        }

        // LPT ordering: schedule heavy strips first so dynamic worksharing finishes evenly.
        std::stable_sort(phase1_tasks.begin(), phase1_tasks.end(),
            [](const Phase1StripTask &a, const Phase1StripTask &b) { return a.strip_rows > b.strip_rows; });
        std::stable_sort(phase2_tasks.begin(), phase2_tasks.end(),
            [](const Phase2StripTask &a, const Phase2StripTask &b) { return a.strip_rows > b.strip_rows; });

        // Capture plain pointers to master thread's retained scratch vectors:
        // thread_local identifiers inside an OpenMP region resolve to each worker's
        // own empty vector rather than the master's prepared task arrays.
        const PreparedMatrixInfo *const gate_info_ptr = gate_info.data();
        const PreparedMatrixInfo *const up_info_ptr = up_info.data();
        const PreparedMatrixInfo *const down_info_ptr = down_info.data();
        const int *const expert_begins_ptr = expert_begins.data();
        const BatchedPrequantizedProjectionDesc *const down_desc_ptr = down_descriptors.data();
        const int *const row_idx_ptr = router_row_indices.data();
        const size_t n_phase1 = phase1_tasks.size();
        const size_t n_phase2 = phase2_tasks.size();
        const Phase1StripTask *const phase1_ptr = phase1_tasks.data();
        const Phase2StripTask *const phase2_ptr = phase2_tasks.data();

        auto run_fused_team = [=]()
        {
            // Phase 1: Fused Gate + Up -> SwiGLU -> Q8_1 (zero gate/up DRAM round-trip).
#pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < n_phase1; ++t)
            {
                const auto &task = phase1_ptr[t];
                const int e = task.expert_idx;
                const int chunk = task.chunk;
                const int cols = std::min(kChunkCols, intermediate - chunk * kChunkCols);
                const int base_route = expert_begins_ptr[static_cast<size_t>(e)] + task.strip_begin;
                const int sr = task.strip_rows;

                const Q8_1Block *in_rows[kStripMaxRows];
                for (int r = 0; r < sr; ++r)
                {
                    const int unique_row = row_idx_ptr[base_route + r];
                    in_rows[r] = unique_router_q8 +
                                 static_cast<size_t>(unique_row) * static_cast<size_t>(router_q8_blocks_per_row);
                }

                alignas(64) float gate_buf[kStripMaxRows][kChunkCols];
                alignas(64) float up_buf[kStripMaxRows][kChunkCols];
                runStripChunkOrdered(gate_info_ptr[static_cast<size_t>(e)], in_rows, sr, gate_buf, chunk, cols, isa);
                runStripChunkOrdered(up_info_ptr[static_cast<size_t>(e)], in_rows, sr, up_buf, chunk, cols, isa);

                const int q8_chunk_offset = chunk * 2;
                for (int r = 0; r < sr; ++r)
                {
                    Q8_1Block *dst_q8 = activation_q8 +
                        static_cast<size_t>(base_route + r) * static_cast<size_t>(activation_blocks_per_row) +
                        q8_chunk_offset;
                    quantizeSwiGLUChunkToQ8(gate_buf[r], up_buf[r], dst_q8, cols, swiglu_policy, isa);
                }
            }
            // Implicit OpenMP barrier at the end of Phase 1 publishes all activation_q8 rows.

            // Phase 2: Weight-stationary Down directly into destination route rows.
#pragma omp for schedule(dynamic, 1) nowait
            for (size_t t = 0; t < n_phase2; ++t)
            {
                const auto &task = phase2_ptr[t];
                const int e = task.expert_idx;
                const int sr = task.strip_rows;
                const auto &d_desc = down_desc_ptr[static_cast<size_t>(e)];
                const auto &d_info = down_info_ptr[static_cast<size_t>(e)];
                const Q8_1Block *const swiglu_base =
                    d_desc.input_q8 + static_cast<size_t>(task.strip_begin) * static_cast<size_t>(activation_blocks_per_row);

                const Q8_1Block *swiglu_rows[kStripMaxRows];
                for (int r = 0; r < sr; ++r)
                    swiglu_rows[r] = swiglu_base + static_cast<size_t>(r) * static_cast<size_t>(activation_blocks_per_row);

                alignas(64) float out_buf[kStripMaxRows][kChunkCols];
                for (int c_off = 0; c_off < task.chunk_count; ++c_off)
                {
                    const int chunk = task.chunk_begin + c_off;
                    const int cols = std::min(kChunkCols, hidden_width - chunk * kChunkCols);
                    runStripChunkOrdered(d_info, swiglu_rows, sr, out_buf, chunk, cols, isa);
                    const int col_offset = chunk * kChunkCols;
                    for (int r = 0; r < sr; ++r)
                    {
                        float *dst = d_desc.output_rows.rowAddress(
                                         d_desc.output,
                                         d_desc.ldc,
                                         static_cast<size_t>(task.strip_begin + r)) +
                                     col_offset;
                        std::memcpy(dst, out_buf[r], static_cast<size_t>(cols) * sizeof(float));
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(run_fused_team);
    }

    if ((!omp_in_parallel() || omp_get_thread_num() == 0) &&
        PerfStatsCollector::isDomainEnabled("kernel"))
    {
        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_native_quantised_grouped_moe_fused_ffn_calls",
            1.0,
            "gemm",
            "cpu",
            {{"active_experts", std::to_string(num_experts)},
             {"route_rows", std::to_string(route_rows)},
             {"unique_rows", std::to_string(unique_router_rows)},
             {"hidden_width", std::to_string(hidden_width)},
             {"intermediate", std::to_string(intermediate)},
             {"isa", isaLevelName(isa)},
             {"team_lifetime", "single_omp_fused_moe_ffn"}});
        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_native_quantised_router_q8_grouped_decode_equivalent_projection_calls",
            1.0,
            "gemm",
            "cpu",
            {{"m", std::to_string(route_rows)},
             {"k", std::to_string(hidden_width)},
             {"projections", std::to_string(gate_up_descriptors.size())},
             {"path", "grouped_fused_ffn"}});
        for (size_t e = 0; e < num_experts; ++e)
        {
            const int rows = down_descriptors[e].rows;
            const std::string m_str = std::to_string(rows);
            const std::string route = (rows > 1) ? "grouped_fused_ffn" : "decode_fused_ffn";
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_quantised_fused_verifier_rows_projection_launch",
                PerfStatsCollector::CounterObservationBatch{.observations = 3u, .value = 3.0},
                "gemm",
                "cpu",
                {{"m", m_str},
                 {"route", route},
                 {"isa", isaLevelName(isa)}});
        }
    }
    return true;
}

} // namespace llaminar2::cpu::native_quantised
