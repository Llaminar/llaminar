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
    constexpr int kStripMaxRows = 32;
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

    struct PreparedMatrixInfo
    {
        const CPUNativeQuantisedPackedWeights *packed = nullptr;
        int k_blocks = 0;
        int k_tiles = 1;
        int k_blocks_per_tile = 1;
        bool is_nibble = false;
        bool is_compact_ms = false;
        bool is_compact_five_bit = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i lut512 = _mm512_setzero_si512();
#endif
#if defined(__AVX2__)
        __m256i lut256 = _mm256_setzero_si256();
#endif
    };

    inline PreparedMatrixInfo prepareMatrixInfo(
        const CPUNativeQuantisedPackedWeights &packed,
        int n,
        int k,
        int workers,
        const CPUCacheGeometry &cache)
    {
        PreparedMatrixInfo info{};
        info.packed = &packed;
        info.k_blocks = packed.blocks_per_row;
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
        info.is_nibble = packed.usesNibbleLUT();
        info.is_compact_ms = packed.usesCompactMultiScale();
        info.is_compact_five_bit = packed.usesCompactFiveBit();
        const bool needs_lut = info.is_nibble && nibbleDecodeKind(packed.codebook_id) == NibbleDecodeKind::LUT;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (needs_lut)
            info.lut512 = build_decode_lut(packed.codebook_id);
#endif
#if defined(__AVX2__)
        if (needs_lut)
            info.lut256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);
#endif
        return info;
    }

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

            // K-tile outer: each 5-block weight slice (~5-10 KiB) stays L1d-hot
            // across all strip_rows, and partial buffers fold in exact ascending order.
            alignas(64) float p0[kChunkCols], p1[kChunkCols], p2[kChunkCols], p3[kChunkCols];
            for (int kt = 0; kt < k_tiles; ++kt)
            {
                const int kb_start = kt * bpt;
                const int kb_end = std::min(kb_start + bpt, k_blocks);
                const bool first = (kt == 0);
                int r = 0;
                if (m.is_nibble)
                {
                    for (; r + 3 <= strip_rows; r += 3)
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
            primitives::compute_swiglu(gate_vals, up_vals, swiglu_vals, cols);

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

    gate_info.resize(num_experts);
    up_info.resize(num_experts);
    down_info.resize(num_experts);
    expert_begins.resize(num_experts);
    phase1_tasks.clear();
    phase2_tasks.clear();

    int row_cursor = 0;
    for (size_t e = 0; e < num_experts; ++e)
    {
        expert_begins[e] = row_cursor;
        gate_info[e] = prepareMatrixInfo(gate_up_descriptors[2u * e].kernel->packed_, intermediate, hidden_width, workers, cache);
        up_info[e] = prepareMatrixInfo(gate_up_descriptors[2u * e + 1u].kernel->packed_, intermediate, hidden_width, workers, cache);
        down_info[e] = prepareMatrixInfo(down_descriptors[e].kernel->packed_, hidden_width, intermediate, workers, cache);

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
