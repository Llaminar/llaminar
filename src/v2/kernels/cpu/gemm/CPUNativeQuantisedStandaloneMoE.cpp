/**
 * @file CPUNativeQuantisedStandaloneMoE.cpp
 * @brief Execute native one-token MoE expert phases on one fixed physical worker team.
 *
 * Admission binds original ordered dot entries and persistent route storage for
 * any admitted `(hidden, intermediate)` geometry divisible by the 64-column
 * NativeQuantised tile width across all 18 codebooks and all three runtime CPU
 * ISA levels (`AVX-512`, `AVX2`, and `Scalar`) via `activeISALevel()`. Keeping
 * the specialized tile bodies in this compilation unit gives the grouped
 * prefill/verifier paths independent compiler inlining and register budgets.
 */
#include "CPUNativeQuantisedGemmKernel.h"
#include "CPUNativeQuantisedGemv.h"
#include "CPUNativeQuantisedScalarReference.h"
#include "execution/moe/MoERuntimeTable.h"

namespace llaminar2::cpu::native_quantised
{
namespace
{
    inline void computeSwiGLU64(const float *gate, const float *up, float *out, CPUProjectionNumericalPolicy policy) noexcept
    {
        if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            primitives::compute_swiglu_gpu_aligned_expert_serial(gate, up, out, 64);
        else
            primitives::compute_swiglu_serial(gate, up, out, 64);
    }

    inline void quantize64ValuesToTwoQ8Blocks(
        const float *values,
        Q8_1Block *dst,
        CPUProjectionNumericalPolicy down_policy,
        ISALevel isa) noexcept
    {
        if (down_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
        {
#if defined(__AVX512F__)
            if (isa >= ISALevel::AVX512)
            {
                gpu_aligned_expert_q8::quantizeTwoBlocks(values, dst[0], dst[1]);
                return;
            }
#endif
            (void)isa;
            gpu_aligned_expert_q8::quantizeBlock(values, dst[0], 32);
            gpu_aligned_expert_q8::quantizeBlock(values + 32, dst[1], 32);
            return;
        }
        (void)isa;
        quantizeTwoActivationBlocks(values, dst[0], dst[1], down_policy);
    }

    inline void addOrdered64Floats(float *dst, const float *src, ISALevel isa) noexcept
    {
#if defined(__AVX512F__)
        if (isa >= ISALevel::AVX512)
        {
            _mm512_storeu_ps(dst, addOrderedAVX512(_mm512_loadu_ps(dst), _mm512_load_ps(src)));
            _mm512_storeu_ps(dst + 16, addOrderedAVX512(_mm512_loadu_ps(dst + 16), _mm512_load_ps(src + 16)));
            _mm512_storeu_ps(dst + 32, addOrderedAVX512(_mm512_loadu_ps(dst + 32), _mm512_load_ps(src + 32)));
            _mm512_storeu_ps(dst + 48, addOrderedAVX512(_mm512_loadu_ps(dst + 48), _mm512_load_ps(src + 48)));
            return;
        }
#endif
#if defined(__AVX2__)
        if (isa >= ISALevel::AVX2)
        {
            for (int seg = 0; seg < 8; ++seg)
            {
                const int off = seg * 8;
                _mm256_storeu_ps(dst + off, addOrderedAVX2(_mm256_loadu_ps(dst + off), _mm256_load_ps(src + off)));
            }
            return;
        }
#endif
        (void)isa;
        for (int c = 0; c < 64; ++c)
            dst[c] = device_fp32_contract::add(dst[c], src[c]);
    }
} // namespace

int CPUNativeQuantisedGemmKernel::standaloneMoEOneRowWorkers(int hidden, int intermediate, size_t routes) noexcept
{
    if (hidden >= 64 && intermediate >= 64 && (hidden % 64) == 0 && (intermediate % 64) == 0 &&
        routes && routes <= kDeviceMoEMaxTopK &&
        !llaminar::v2::inLexicalWorkshare() &&
        !omp_get_dynamic() && cpuNativeQuantisedSerialOutputPartitionN() == 0)
    {
        const int gate_blocks = hidden / 32;
        const int down_blocks = intermediate / 32;
        const int gate_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(hidden);
        const int down_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(intermediate);
        if (gate_blocks > 0 && down_blocks > 0 && gate_parts > 0 && down_parts > 0)
        {
            const int workers = omp_get_max_threads();
            if (workers > 0 && workers <= omp_get_thread_limit())
                return workers;
        }
    }
    return 0;
}

/** @brief Original one-row operands and exact native tile entry, resolved on the invoking owner. */
struct CPUNativeQuantisedGemmKernel::PreparedMoERow
{
    const CPUNativeQuantisedPackedWeights *weights; ///< Borrowed prepared weight owner.
    const Q8_1Block *input; ///< Original router Q8 row or the ordered SwiGLU publication.
    float *output; ///< One phase's original scratch row with disjoint N64 owners.
    void (*tile)(const PreparedMoERow &, int); ///< Native entry bound before the workers start.
    void (*part_range)(const PreparedMoERow &, int, int, int, float *, bool); ///< Ordered partition range entry for split-K.
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i lut512; ///< Immutable AVX-512 codebook lookup for entries that consume nibbles.
#endif
#if defined(__AVX2__)
    __m256i lut256; ///< Immutable AVX2 codebook lookup for entries that consume nibbles.
#endif
    int blocks; ///< Exact 32-element Q8_1 K-block count (`K / 32`).
    int partitions; ///< Ordered K-partition count from MoEProjectionNumericalContract.
    ISALevel isa; ///< Active runtime ISA resolved on the invoking owner.
};

/**
 * @brief Run one complete N64 tile with the selected source, ISA, and original ordered partitions.
 * @tparam Nibble Whether the bound reader consumes a nibble lookup table.
 * @param row Prepared operands for one live projection.
 * @param chunk Disjoint output column tile owned by this worker.
 */
template<bool Nibble>
void CPUNativeQuantisedGemmKernel::preparedMoERowTile(const PreparedMoERow &row, int chunk)
{
    constexpr int columns = 64;
    const int blocks = row.blocks;
    const int partitions = row.partitions;
    const int bpt = (blocks + partitions - 1) / partitions;
    float *output = row.output + chunk * columns;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    if (row.isa >= ISALevel::AVX512)
    {
        if (partitions <= 1 || partitions == blocks)
        {
            if constexpr (Nibble)
                gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, output, chunk, 0, blocks,
                    row.lut512, false);
            else
                gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, blocks, false);
        }
        else if constexpr (Nibble)
            dispatch_native_avx512_ordered_partitions(*row.weights, row.input, output, chunk, blocks,
                partitions, bpt, row.lut512);
        else
            gemvNonNibbleAVX512OrderedPartitions(*row.weights, row.input, output, chunk, blocks,
                partitions, bpt);
        return;
    }
#endif
#if defined(__AVX2__)
    if (row.isa >= ISALevel::AVX2)
    {
        if (partitions <= 1)
        {
            if constexpr (Nibble)
                gemv_avx2_chunk_native(*row.weights, row.input, output, chunk, 0, blocks, row.lut256, false);
            else
                gemv_avx2_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, blocks, false);
        }
        else
            gemvAVX2OrderedPartitions(*row.weights, row.input, output, chunk, blocks,
                partitions, bpt, row.lut256);
        return;
    }
#endif
    gemvScalarOrderedPartitions(*row.weights, row.input, output, chunk, blocks,
        partitions, bpt, columns);
}

/**
 * @brief Run an ordered sub-range `[p_start, p_end)` of K-partitions for one N64 tile.
 *
 * When `fold_into_dst` is true (`p_start == 0`), partitions are left-folded directly
 * into the 64-float `dst` vector; when false, each partition `p` is materialized into
 * `dst + (p - p_start) * 64` so SwiGLU can left-fold partitions in exact
 * bitwise-identical FP32 serial order.
 */
template<bool Nibble>
void CPUNativeQuantisedGemmKernel::preparedMoERowPartitionRange(const PreparedMoERow &row, int chunk,
    int p_start, int p_end, float *dst, bool fold_into_dst)
{
    const int bpt = (row.blocks + row.partitions - 1) / row.partitions;
    auto run_single_partition = [&](int kb_start, int kb_end, float *out)
    {
        if (kb_start >= kb_end)
        {
            std::fill_n(out, 64, 0.0f);
            return;
        }
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (row.isa >= ISALevel::AVX512)
        {
            if constexpr (Nibble)
                gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, row.lut512, false);
            else
                gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, false);
            return;
        }
#endif
#if defined(__AVX2__)
        if (row.isa >= ISALevel::AVX2)
        {
            if constexpr (Nibble)
                gemv_avx2_chunk_native(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, row.lut256, false);
            else
                gemv_avx2_chunk_non_nibble(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, false);
            return;
        }
#endif
        gemv_native_quantised_scalar_chunk(*row.weights, row.input, out, chunk, kb_start, kb_end, 64);
    };

    if (fold_into_dst)
    {
        if (p_start == 0 && p_end == row.partitions)
        {
            preparedMoERowTile<Nibble>(row, chunk);
            if (dst != row.output + chunk * 64)
                std::memcpy(dst, row.output + chunk * 64, 64 * sizeof(float));
            return;
        }
        alignas(64) float part_buf[64];
        for (int p = p_start; p < p_end; ++p)
        {
            const int kb_start = p * bpt;
            const int kb_end = std::min(kb_start + bpt, row.blocks);
            float *out = (p == p_start) ? dst : part_buf;
            run_single_partition(kb_start, kb_end, out);
            if (p > p_start)
                addOrdered64Floats(dst, part_buf, row.isa);
        }
        return;
    }
    for (int p = p_start; p < p_end; ++p)
    {
        float *out = dst + static_cast<size_t>(p - p_start) * 64u;
        const int kb_start = p * bpt;
        const int kb_end = std::min(kb_start + bpt, row.blocks);
        run_single_partition(kb_start, kb_end, out);
    }
}

/**
 * @brief Bind admitted source operands and their immutable lookup before forking.
 */
bool CPUNativeQuantisedGemmKernel::prepareMoERows(std::span<const BatchedPrequantizedProjectionDesc> original,
    std::span<PreparedMoERow> prepared, int width, int blocks, int workers, ISALevel isa)
{
    const int k = blocks * 32;
    const int gpu_aligned_partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
    if (gpu_aligned_partitions <= 0 || original.empty())
        return false;
    const auto &cache = CPUExecutionGeometry::local().cache;
    for (size_t projection = 0; projection < original.size(); ++projection)
    {
        const auto &member = original[projection];
        if (!member.kernel || !member.kernel->valid_ || !member.input_q8 || !member.output)
            throw std::invalid_argument("Prepared standalone CPU MoE projection lacks its complete native operands");
        const auto &weights = member.kernel->packed_;
        if (member.rows != 1 || member.n != width || member.ldc < width || member.bias || member.kernel->activation_rotation_ ||
            weights.N != width || weights.K != k ||
            member.purpose != ProjectionRowsPurpose::Decode || member.verifier_schedule != VerifierRowsPolicy::Auto)
            return false;
        const int resolved_k_tiles = std::max(1, serialTileConfigForPreparedProjection(
            weights.preparedFootprint(), weights.codebook_id,
            weights.numerical_policy, width, k, workers, cache).k_tiles);
        if (weights.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert &&
            resolved_k_tiles != gpu_aligned_partitions)
            return false;
        const bool nibble = weights.usesNibbleLUT();
        const bool needs_lut = nibble && nibbleDecodeKind(weights.codebook_id) == NibbleDecodeKind::LUT;
        PreparedMoERow plan{};
        plan.weights = &weights;
        plan.input = member.input_q8;
        plan.output = member.output;
        plan.tile = nibble ? preparedMoERowTile<true> : preparedMoERowTile<false>;
        plan.part_range = nibble ? preparedMoERowPartitionRange<true> : preparedMoERowPartitionRange<false>;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        plan.lut512 = needs_lut ? build_decode_lut(weights.codebook_id) : _mm512_setzero_si512();
#endif
#if defined(__AVX2__)
        plan.lut256 = needs_lut ? build_decode_lut_avx2_for_codebook(weights.codebook_id) : _mm256_setzero_si256();
#endif
        plan.blocks = blocks;
        plan.partitions = resolved_k_tiles;
        plan.isa = isa;
        prepared[projection] = plan;
    }
    return true;
}

template<bool Join>
void CPUNativeQuantisedGemmKernel::executeMoERows(std::span<const PreparedMoERow> rows, int width)
{
    const int tiles = width / 64;
    const int tasks = static_cast<int>(rows.size()) * tiles;
#pragma omp for schedule(static) nowait
    for (int task = 0; task < tasks; ++task)
    {
        const auto &row = rows[task / tiles];
        row.tile(row, task % tiles);
    }
    if constexpr (Join)
    {
#pragma omp barrier
    }
}

inline int selectStandaloneGateUpKSplit(int gate_tiles, int gate_partitions, int workers, size_t max_splitk_floats) noexcept
{
    if (workers <= 1 || gate_tiles <= 0 || gate_partitions < 2 || gate_tiles >= workers * 2)
        return 1;
    const auto fits = [&](int k) noexcept -> bool {
        if ((gate_partitions % k) != 0)
            return false;
        const int tail_parts = gate_partitions - (gate_partitions / k);
        return static_cast<size_t>(gate_tiles) * static_cast<size_t>(tail_parts) * 64u <= max_splitk_floats;
    };
    const auto peak_cost = [&](int k) noexcept -> double {
        const int tasks = gate_tiles * k;
        const int max_per_worker = (tasks + workers - 1) / workers;
        return static_cast<double>(max_per_worker) / static_cast<double>(k);
    };
    const double c1 = peak_cost(1);
    const double c2 = peak_cost(2);
    const double c4 = peak_cost(4);
    if (fits(4) && c4 < c1 * 0.95 && c4 < c2 * 0.96)
        return 4;
    if (fits(2) && c2 < c1 * 0.95)
        return 2;
    return 1;
}

static constexpr size_t kStandaloneSplitKMaxFloats = 160u * 12u * 64u;

template<int KSplit>
__attribute__((noinline))
bool CPUNativeQuantisedGemmKernel::runPreparedFixedMoETilesSplitK(
    std::span<const PreparedMoERow> gate_plans,
    std::span<const PreparedMoERow> down_plans,
    std::span<Q8_1Block> activation,
    int hidden,
    int intermediate,
    int gate_partitions,
    int workers,
    ISALevel isa)
{
    static_assert(KSplit == 2 || KSplit == 4);
    constexpr int columns_per_task = 64;
    const int tasks_per_route = intermediate / columns_per_task;
    const size_t blocks_per_route = static_cast<size_t>(intermediate) / Q8_1Block::BLOCK_SIZE;
    const int parts_per_split = gate_partitions / KSplit;
    const int tail_parts = gate_partitions - parts_per_split;

    alignas(64) std::array<float, kStandaloneSplitKMaxFloats> partials;
    const int gate_tiles = static_cast<int>(gate_plans.size()) * tasks_per_route;
    const int gate_tasks = gate_tiles * KSplit;
    const int activation_tasks = static_cast<int>(down_plans.size()) * tasks_per_route;
    std::atomic<bool> valid{true};

    const auto fold_tile = [&](float *dst, const float *tile_partials) noexcept {
        for (int p = 0; p < tail_parts; ++p)
        {
            const float *src = tile_partials + static_cast<size_t>(p) * columns_per_task;
            addOrdered64Floats(dst, src, isa);
        }
    };

    const auto run = [&]() {
        if (omp_get_num_threads() != workers)
        {
            valid.store(false, std::memory_order_relaxed);
            return;
        }
#pragma omp for schedule(static)
        for (int task = 0; task < gate_tasks; ++task)
        {
            const int tile = task / KSplit;
            const int split = task % KSplit;
            const auto &row = gate_plans[tile / tasks_per_route];
            const int chunk = tile % tasks_per_route;
            const int p_start = split * parts_per_split;
            const int p_end = p_start + parts_per_split;
            if (split == 0)
            {
                row.part_range(row, chunk, 0, parts_per_split, row.output + chunk * columns_per_task, true);
            }
            else
            {
                float *dst = partials.data() +
                    (static_cast<size_t>(tile) * static_cast<size_t>(tail_parts) +
                     static_cast<size_t>(p_start - parts_per_split)) *
                    columns_per_task;
                row.part_range(row, chunk, p_start, p_end, dst, false);
            }
        }
#pragma omp for schedule(static)
        for (int task = 0; task < activation_tasks; ++task)
        {
            const int route = task / tasks_per_route;
            const int chunk = task % tasks_per_route;
            const int offset = chunk * columns_per_task;
            const size_t gate_tile = static_cast<size_t>(2 * route) * tasks_per_route + static_cast<size_t>(chunk);
            const size_t up_tile = static_cast<size_t>(2 * route + 1) * tasks_per_route + static_cast<size_t>(chunk);
            float *const gate_dst = gate_plans[2 * route].output + offset;
            float *const up_dst = gate_plans[2 * route + 1].output + offset;
            fold_tile(gate_dst, partials.data() + gate_tile * static_cast<size_t>(tail_parts) * columns_per_task);
            fold_tile(up_dst, partials.data() + up_tile * static_cast<size_t>(tail_parts) * columns_per_task);
            alignas(64) float values[columns_per_task];
            computeSwiGLU64(gate_dst, up_dst, values, down_plans[route].weights->numerical_policy);
            auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
            quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
        }
        executeMoERows<false>(down_plans, hidden);
    };
    OMP_WORKSHARE_REGION(run);
    return valid.load(std::memory_order_relaxed);
}

std::optional<bool> CPUNativeQuantisedGemmKernel::runPreparedFixedMoETiles(
    std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation,
    int hidden, int intermediate, int workers)
{
    constexpr int columns_per_task = 64;
    const int tasks_per_route = intermediate / columns_per_task;
    const size_t blocks_per_route = static_cast<size_t>(intermediate) / Q8_1Block::BLOCK_SIZE;
    if (activation.size() < down.size() * blocks_per_route)
        throw std::invalid_argument("Prepared standalone CPU MoE requires the admitted live SwiGLU bank");
    for (size_t route = 0; route < down.size(); ++route)
    {
        if (down[route].input_q8 != activation.data() + route * blocks_per_route)
            throw std::invalid_argument("Prepared standalone CPU MoE Down must consume its own SwiGLU row");
        const storage_proof::Bytes output{down[route].input_q8, blocks_per_route * sizeof(Q8_1Block)};
        storage_proof::requireDisjoint(output, {gate[2 * route].output, static_cast<size_t>(intermediate) * sizeof(float)});
        storage_proof::requireDisjoint(output, {gate[2 * route + 1].output, static_cast<size_t>(intermediate) * sizeof(float)});
    }
    const ISALevel isa = activeISALevel();
    std::array<PreparedMoERow, 2 * kDeviceMoEMaxTopK> gate_storage;
    std::array<PreparedMoERow, kDeviceMoEMaxTopK> down_storage;
    auto gate_plans = std::span(gate_storage).first(gate.size());
    auto down_plans = std::span(down_storage).first(down.size());
    const int gate_blocks = hidden / 32;
    const int down_blocks = intermediate / 32;
    if (!prepareMoERows(gate, gate_plans, intermediate, gate_blocks, workers, isa) ||
        !prepareMoERows(down, down_plans, hidden, down_blocks, workers, isa))
        return std::nullopt;

    const int gate_tiles = static_cast<int>(gate_plans.size()) * tasks_per_route;
    const int gate_partitions = gate_plans[0].partitions;
    const bool uniform_gate_partitions = std::all_of(gate_plans.begin(), gate_plans.end(),
        [gate_partitions](const PreparedMoERow &p) { return p.partitions == gate_partitions; });
    const int k_split = uniform_gate_partitions
        ? selectStandaloneGateUpKSplit(gate_tiles, gate_partitions, workers, kStandaloneSplitKMaxFloats)
        : 1;
    if (k_split == 4)
        return runPreparedFixedMoETilesSplitK<4>(gate_plans, down_plans, activation, hidden, intermediate, gate_partitions, workers, isa);
    if (k_split == 2)
        return runPreparedFixedMoETilesSplitK<2>(gate_plans, down_plans, activation, hidden, intermediate, gate_partitions, workers, isa);

    const int activation_tasks = static_cast<int>(down.size()) * tasks_per_route;
    const int fused_peak = 2 * ((activation_tasks + workers - 1) / workers);
    const int unfused_peak = (gate_tiles + workers - 1) / workers;
    const bool fuse_gate_up_swiglu = (fused_peak == unfused_peak);
    std::atomic<bool> valid{true};
    const auto run = [&]() {
        if (omp_get_num_threads() != workers)
        {
            valid.store(false, std::memory_order_relaxed);
            return;
        }
        if (fuse_gate_up_swiglu)
        {
#pragma omp for schedule(static)
            for (int task = 0; task < activation_tasks; ++task)
            {
                const int route = task / tasks_per_route;
                const int chunk = task % tasks_per_route;
                const int offset = chunk * columns_per_task;
                const auto &g_row = gate_plans[2 * route];
                const auto &u_row = gate_plans[2 * route + 1];
                g_row.tile(g_row, chunk);
                u_row.tile(u_row, chunk);
                alignas(64) float values[columns_per_task];
                computeSwiGLU64(g_row.output + offset, u_row.output + offset, values,
                                down_plans[route].weights->numerical_policy);
                auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
                quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
            }
        }
        else
        {
            executeMoERows<true>(gate_plans, intermediate);
#pragma omp for schedule(static)
            for (int task = 0; task < activation_tasks; ++task)
            {
                const int route = task / tasks_per_route;
                const int offset = task % tasks_per_route * columns_per_task;
                alignas(64) float values[columns_per_task];
                computeSwiGLU64(gate_plans[2 * route].output + offset,
                    gate_plans[2 * route + 1].output + offset, values,
                    down_plans[route].weights->numerical_policy);
                auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
                quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
            }
        }
        // The standalone team join retires all final Down stores.
        executeMoERows<false>(down_plans, hidden);
    };
    OMP_WORKSHARE_REGION(run);
    return valid.load(std::memory_order_relaxed);
}

std::optional<bool> CPUNativeQuantisedGemmKernel::tryExecuteStandaloneMoEOneRow(std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation,
    int hidden, int intermediate)
{
    if (gate.size() == 2 * down.size())
    {
        const int workers = standaloneMoEOneRowWorkers(hidden, intermediate, down.size());
        if (workers > 0)
        {
            requireDisjointIndexedBatchedOutputs(gate);
            requireDisjointIndexedBatchedOutputs(down);
            const auto result = runPreparedFixedMoETiles(gate, down, activation, hidden, intermediate, workers);
            if (result.value_or(false) && PerfStatsCollector::isDomainEnabled("kernel"))
                PerfStatsCollector::addCounter("kernel", "cpu_native_quantised_moe_prepared_phase_plans",
                    2.0, "gemm", "cpu");
            return result;
        }
    }
    return std::nullopt;
}

bool CPUNativeQuantisedGemmKernel::supportsStandaloneMoEOneRowInTeam(int hidden, int intermediate, size_t routes) noexcept
{
    if (hidden < 64 || intermediate < 64 || (hidden % 64) != 0 || (intermediate % 64) != 0 ||
        !routes || routes > kDeviceMoEMaxTopK ||
        omp_get_dynamic() || cpuNativeQuantisedSerialOutputPartitionN() != 0)
        return false;
    const int gate_blocks = hidden / 32;
    const int down_blocks = intermediate / 32;
    const int gate_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(hidden);
    const int down_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(intermediate);
    return gate_blocks > 0 && down_blocks > 0 && gate_parts > 0 && down_parts > 0;
}

namespace
{
    using PlanRowEntry = CPUNativeQuantisedGemmKernel::StandaloneMoEOneRowPlan::RowEntry;

    template<bool Nibble>
    void planRowEntryTile(const PlanRowEntry &row, int chunk)
    {
        constexpr int columns = 64;
        const int blocks = row.blocks;
        const int partitions = row.partitions;
        const int bpt = (blocks + partitions - 1) / partitions;
        float *output = row.output + chunk * columns;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (row.isa >= ISALevel::AVX512)
        {
            __m512i lut512 = _mm512_setzero_si512();
            if constexpr (Nibble)
                lut512 = _mm512_load_si512(reinterpret_cast<const void *>(row.lut.data()));
            if (partitions <= 1 || partitions == blocks)
            {
                if constexpr (Nibble)
                    gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, output, chunk, 0, blocks,
                        lut512, false);
                else
                    gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, blocks, false);
            }
            else if constexpr (Nibble)
                dispatch_native_avx512_ordered_partitions(*row.weights, row.input, output, chunk, blocks,
                    partitions, bpt, lut512);
            else
                gemvNonNibbleAVX512OrderedPartitions(*row.weights, row.input, output, chunk, blocks,
                    partitions, bpt);
            return;
        }
#endif
#if defined(__AVX2__)
        if (row.isa >= ISALevel::AVX2)
        {
            __m256i lut256 = _mm256_setzero_si256();
            if constexpr (Nibble)
                lut256 = _mm256_load_si256(reinterpret_cast<const __m256i *>(row.lut.data()));
            if (partitions <= 1)
            {
                if constexpr (Nibble)
                    gemv_avx2_chunk_native(*row.weights, row.input, output, chunk, 0, blocks, lut256, false);
                else
                    gemv_avx2_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, blocks, false);
            }
            else
                gemvAVX2OrderedPartitions(*row.weights, row.input, output, chunk, blocks,
                    partitions, bpt, lut256);
            return;
        }
#endif
        gemvScalarOrderedPartitions(*row.weights, row.input, output, chunk, blocks,
            partitions, bpt, columns);
    }

    template<bool Nibble>
    void planRowEntryPartitionRange(const PlanRowEntry &row, int chunk,
        int p_start, int p_end, float *dst, bool fold_into_dst)
    {
        const int bpt = (row.blocks + row.partitions - 1) / row.partitions;
        auto run_single = [&](int kb_start, int kb_end, float *out)
        {
            if (kb_start >= kb_end)
            {
                std::fill_n(out, 64, 0.0f);
                return;
            }
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
            if (row.isa >= ISALevel::AVX512)
            {
                if constexpr (Nibble)
                {
                    const __m512i lut512 = _mm512_load_si512(reinterpret_cast<const void *>(row.lut.data()));
                    gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, out, chunk,
                        kb_start, kb_end, lut512, false);
                }
                else
                {
                    gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, out, chunk,
                        kb_start, kb_end, false);
                }
                return;
            }
#endif
#if defined(__AVX2__)
            if (row.isa >= ISALevel::AVX2)
            {
                if constexpr (Nibble)
                {
                    const __m256i lut256 = _mm256_load_si256(reinterpret_cast<const __m256i *>(row.lut.data()));
                    gemv_avx2_chunk_native(*row.weights, row.input, out, chunk,
                        kb_start, kb_end, lut256, false);
                }
                else
                {
                    gemv_avx2_chunk_non_nibble(*row.weights, row.input, out, chunk,
                        kb_start, kb_end, false);
                }
                return;
            }
#endif
            gemv_native_quantised_scalar_chunk(*row.weights, row.input, out, chunk, kb_start, kb_end, 64);
        };

        if (fold_into_dst)
        {
            if (p_start == 0 && p_end == row.partitions)
            {
                planRowEntryTile<Nibble>(row, chunk);
                if (dst != row.output + chunk * 64)
                    std::memcpy(dst, row.output + chunk * 64, 64 * sizeof(float));
                return;
            }
            alignas(64) float part_buf[64];
            for (int p = p_start; p < p_end; ++p)
            {
                const int kb_start = p * bpt;
                const int kb_end = std::min(kb_start + bpt, row.blocks);
                float *out = (p == p_start) ? dst : part_buf;
                run_single(kb_start, kb_end, out);
                if (p > p_start)
                    addOrdered64Floats(dst, part_buf, row.isa);
            }
            return;
        }
        for (int p = p_start; p < p_end; ++p)
        {
            float *out = dst + static_cast<size_t>(p - p_start) * 64u;
            const int kb_start = p * bpt;
            const int kb_end = std::min(kb_start + bpt, row.blocks);
            run_single(kb_start, kb_end, out);
        }
    }

    bool fillPlanRowEntries(
        std::span<const CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc> original,
        std::span<PlanRowEntry> prepared,
        int width, int blocks, int workers, ISALevel isa)
    {
        const int k = blocks * 32;
        const int gpu_aligned_partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
        if (gpu_aligned_partitions <= 0 || original.empty())
            return false;
        const auto &cache = CPUExecutionGeometry::local().cache;
        for (size_t i = 0; i < original.size(); ++i)
        {
            const auto &member = original[i];
            if (!member.kernel || !member.input_q8 || !member.output)
                return false;
            const auto &weights = member.kernel->packedWeights();
            if (member.rows != 1 || member.n != width || member.ldc < width || member.bias ||
                weights.N != width || weights.K != k ||
                member.purpose != ProjectionRowsPurpose::Decode ||
                member.verifier_schedule != VerifierRowsPolicy::Auto)
                return false;
            const int resolved_k_tiles = std::max(1, serialTileConfigForPreparedProjection(
                weights.preparedFootprint(), weights.codebook_id,
                weights.numerical_policy, width, k, workers, cache).k_tiles);
            if (weights.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert &&
                resolved_k_tiles != gpu_aligned_partitions)
                return false;
            const bool nibble = weights.usesNibbleLUT();
            const bool needs_lut = nibble && nibbleDecodeKind(weights.codebook_id) == NibbleDecodeKind::LUT;
            PlanRowEntry entry{};
            entry.weights = &weights;
            entry.input = member.input_q8;
            entry.output = member.output;
            entry.tile = nibble ? planRowEntryTile<true> : planRowEntryTile<false>;
            entry.part_range = nibble ? planRowEntryPartitionRange<true> : planRowEntryPartitionRange<false>;
            if (needs_lut)
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                if (isa >= ISALevel::AVX512)
                {
                    const __m512i lut512 = build_decode_lut(weights.codebook_id);
                    _mm512_store_si512(reinterpret_cast<void *>(entry.lut.data()), lut512);
                }
                else
#endif
#if defined(__AVX2__)
                if (isa >= ISALevel::AVX2)
                {
                    const __m256i lut256 = build_decode_lut_avx2_for_codebook(weights.codebook_id);
                    _mm256_store_si256(reinterpret_cast<__m256i *>(entry.lut.data()), lut256);
                }
#endif
            }
            entry.blocks = blocks;
            entry.partitions = resolved_k_tiles;
            entry.isa = isa;
            prepared[i] = entry;
        }
        return true;
    }
} // namespace

bool CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowEntry(
    StandaloneMoEOneRowPlan::RowEntry &out,
    const CPUNativeQuantisedGemmKernel *kernel,
    int width, int k, int workers, ISALevel isa)
{
    out = {};
    if (!kernel || width <= 0 || k <= 0 || (k % 32) != 0)
        return false;
    const int blocks = k / 32;
    const int gpu_aligned_partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
    if (gpu_aligned_partitions <= 0)
        return false;
    const auto &weights = kernel->packedWeights();
    if (weights.N != width || weights.K != k)
        return false;
    const auto &cache = CPUExecutionGeometry::local().cache;
    const int eff_workers = std::max(1, workers);
    const int resolved_k_tiles = std::max(1, serialTileConfigForPreparedProjection(
        weights.preparedFootprint(), weights.codebook_id,
        weights.numerical_policy, width, k, eff_workers, cache).k_tiles);
    if (weights.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert &&
        resolved_k_tiles != gpu_aligned_partitions)
        return false;
    const bool nibble = weights.usesNibbleLUT();
    const bool needs_lut = nibble && nibbleDecodeKind(weights.codebook_id) == NibbleDecodeKind::LUT;
    out.weights = &weights;
    out.tile = nibble ? planRowEntryTile<true> : planRowEntryTile<false>;
    out.part_range = nibble ? planRowEntryPartitionRange<true> : planRowEntryPartitionRange<false>;
    if (needs_lut)
    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (isa >= ISALevel::AVX512)
        {
            const __m512i lut512 = build_decode_lut(weights.codebook_id);
            _mm512_store_si512(reinterpret_cast<void *>(out.lut.data()), lut512);
        }
        else
#endif
#if defined(__AVX2__)
        if (isa >= ISALevel::AVX2)
        {
            const __m256i lut256 = build_decode_lut_avx2_for_codebook(weights.codebook_id);
            _mm256_store_si256(reinterpret_cast<__m256i *>(out.lut.data()), lut256);
        }
#endif
    }
    out.blocks = blocks;
    out.partitions = resolved_k_tiles;
    out.isa = isa;
    return true;
}

bool CPUNativeQuantisedGemmKernel::finalizeStandaloneMoEOneRowPlan(
    StandaloneMoEOneRowPlan &plan,
    size_t route_count,
    int hidden, int intermediate,
    int workers, ISALevel isa,
    size_t splitk_partials_capacity) noexcept
{
    plan.valid = false;
    if (!supportsStandaloneMoEOneRowInTeam(hidden, intermediate, route_count))
        return false;
    const int eff_workers = std::max(1, workers);
    const int tasks_per_route = intermediate / 64;
    const int gate_tiles = static_cast<int>(2u * route_count) * tasks_per_route;
    const auto gate_span = std::span(plan.gate_plans).first(2u * route_count);
    const int gate_partitions = gate_span[0].partitions;
    const bool uniform_gate_partitions = std::all_of(gate_span.begin(), gate_span.end(),
        [gate_partitions](const PlanRowEntry &p) { return p.partitions == gate_partitions; });
    const size_t usable_splitk = std::min(splitk_partials_capacity, kStandaloneSplitKMaxFloats);
    plan.route_count = route_count;
    plan.hidden = hidden;
    plan.intermediate = intermediate;
    plan.workers = eff_workers;
    plan.gate_partitions = gate_partitions;
    plan.k_split = (uniform_gate_partitions && usable_splitk > 0)
        ? selectStandaloneGateUpKSplit(gate_tiles, gate_partitions, eff_workers, usable_splitk)
        : 1;
    plan.isa = isa;
    plan.valid = true;
    return true;
}

bool CPUNativeQuantisedGemmKernel::prepareStandaloneMoEOneRowPlan(
    StandaloneMoEOneRowPlan &plan,
    std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down,
    int hidden, int intermediate,
    int workers, ISALevel isa,
    size_t splitk_partials_capacity)
{
    static_assert(StandaloneMoEOneRowPlan::kMaxRoutes >= kDeviceMoEMaxTopK);
    plan.valid = false;
    if (gate.size() != 2 * down.size() || !supportsStandaloneMoEOneRowInTeam(hidden, intermediate, down.size()))
        return false;
    const int gate_blocks = hidden / 32;
    const int down_blocks = intermediate / 32;
    const int eff_workers = std::max(1, workers);
    auto gate_span = std::span(plan.gate_plans).first(gate.size());
    auto down_span = std::span(plan.down_plans).first(down.size());
    if (!fillPlanRowEntries(gate, gate_span, intermediate, gate_blocks, eff_workers, isa) ||
        !fillPlanRowEntries(down, down_span, hidden, down_blocks, eff_workers, isa))
        return false;

    const int tasks_per_route = intermediate / 64;
    const int gate_tiles = static_cast<int>(gate.size()) * tasks_per_route;
    const int gate_partitions = gate_span[0].partitions;
    const bool uniform_gate_partitions = std::all_of(gate_span.begin(), gate_span.end(),
        [gate_partitions](const PlanRowEntry &p) { return p.partitions == gate_partitions; });
    const size_t usable_splitk = std::min(splitk_partials_capacity, kStandaloneSplitKMaxFloats);
    plan.route_count = down.size();
    plan.hidden = hidden;
    plan.intermediate = intermediate;
    plan.workers = eff_workers;
    plan.gate_partitions = gate_partitions;
    plan.k_split = (uniform_gate_partitions && usable_splitk > 0)
        ? selectStandaloneGateUpKSplit(gate_tiles, gate_partitions, eff_workers, usable_splitk)
        : 1;
    plan.isa = isa;
    plan.valid = true;
    return true;
}

bool CPUNativeQuantisedGemmKernel::executePreparedStandaloneMoEOneRowInTeam(
    const StandaloneMoEOneRowPlan &plan,
    std::span<Q8_1Block> activation,
    float *splitk_partials, size_t splitk_partials_capacity)
{
    if (!plan.valid)
        return false;
    constexpr int columns_per_task = 64;
    const int tasks_per_route = plan.intermediate / columns_per_task;
    const size_t blocks_per_route = static_cast<size_t>(plan.intermediate) / Q8_1Block::BLOCK_SIZE;
    if (activation.size() < plan.route_count * blocks_per_route)
        return false;

    const auto gate_plans = std::span(plan.gate_plans).first(2 * plan.route_count);
    const auto down_plans = std::span(plan.down_plans).first(plan.route_count);
    const int gate_tiles = static_cast<int>(gate_plans.size()) * tasks_per_route;
    const int activation_tasks = static_cast<int>(plan.route_count) * tasks_per_route;
    const int down_tiles_per_route = plan.hidden / columns_per_task;
    const int down_tasks = static_cast<int>(plan.route_count) * down_tiles_per_route;
    const ISALevel isa = plan.isa;
    const int k_split = (splitk_partials && splitk_partials_capacity > 0) ? plan.k_split : 1;

    const int gate_blocks = plan.hidden / static_cast<int>(Q8_1Block::BLOCK_SIZE);
    const int cur_threads = omp_get_num_threads();
    const int cur_tid = omp_get_thread_num();

    // Stage compact shared Q8_1 inputs (<= 128 blocks = 4.6 KiB) into thread-private
    // L1d stack storage in staggered block order to eliminate multi-core HitM convoys.
    constexpr int kMaxLocalGateBlocks = 128;
    alignas(64) Q8_1Block local_gate_in0[kMaxLocalGateBlocks];
    alignas(64) Q8_1Block local_gate_in1[kMaxLocalGateBlocks];
    using PlanRow = StandaloneMoEOneRowPlan::RowEntry;
    std::array<PlanRow, 2 * StandaloneMoEOneRowPlan::kMaxRoutes> local_gate_plans{};
    std::span<const PlanRow> active_gate_plans = gate_plans;
    if (cur_threads > 1 && gate_blocks > 0 && gate_blocks <= kMaxLocalGateBlocks && !gate_plans.empty())
    {
        const Q8_1Block *src0 = gate_plans[0].input;
        const Q8_1Block *src1 = nullptr;
        for (const auto &gp : gate_plans)
        {
            if (gp.input != src0)
            {
                src1 = gp.input;
                break;
            }
        }
        const int start = (cur_tid * gate_blocks) / cur_threads;
        if (src0)
        {
            for (int step = 0; step < gate_blocks; ++step)
            {
                int idx = start + step;
                if (idx >= gate_blocks)
                    idx -= gate_blocks;
                local_gate_in0[idx] = src0[idx];
            }
        }
        if (src1)
        {
            for (int step = 0; step < gate_blocks; ++step)
            {
                int idx = start + step;
                if (idx >= gate_blocks)
                    idx -= gate_blocks;
                local_gate_in1[idx] = src1[idx];
            }
        }
        for (size_t i = 0; i < gate_plans.size(); ++i)
        {
            local_gate_plans[i] = gate_plans[i];
            if (gate_plans[i].input == src0 && src0)
                local_gate_plans[i].input = local_gate_in0;
            else if (gate_plans[i].input == src1 && src1)
                local_gate_plans[i].input = local_gate_in1;
        }
        active_gate_plans = std::span<const PlanRow>(local_gate_plans.data(), gate_plans.size());
    }

    auto run_down_plans = [&]() {
        constexpr int kMaxLocalActBlocks = 160;
        alignas(64) Q8_1Block local_act[kMaxLocalActBlocks];
        std::array<PlanRow, StandaloneMoEOneRowPlan::kMaxRoutes> local_down_plans{};
        std::span<const PlanRow> active_down_plans = down_plans;
        const int total_act_blocks = static_cast<int>(plan.route_count * blocks_per_route);
        if (cur_threads > 1 && total_act_blocks > 0 && total_act_blocks <= kMaxLocalActBlocks &&
            activation.size() >= static_cast<size_t>(total_act_blocks))
        {
            const int start = (cur_tid * total_act_blocks) / cur_threads;
            const Q8_1Block *act_src = activation.data();
            for (int step = 0; step < total_act_blocks; ++step)
            {
                int idx = start + step;
                if (idx >= total_act_blocks)
                    idx -= total_act_blocks;
                local_act[idx] = act_src[idx];
            }
            for (size_t r = 0; r < plan.route_count; ++r)
            {
                local_down_plans[r] = down_plans[r];
                if (down_plans[r].input == act_src + r * blocks_per_route)
                    local_down_plans[r].input = local_act + r * blocks_per_route;
            }
            active_down_plans = std::span<const PlanRow>(local_down_plans.data(), plan.route_count);
        }
        const int route_count_local = static_cast<int>(plan.route_count);
        if (down_tiles_per_route >= 4 && (down_tiles_per_route % 2) == 0)
        {
            const int chunk_pairs = down_tiles_per_route / 2;
            const int pair_tasks = chunk_pairs * route_count_local;
#pragma omp for schedule(static)
            for (int task = 0; task < pair_tasks; ++task)
            {
                const int pair = task / route_count_local;
                const int route = task % route_count_local;
                const auto &row = active_down_plans[static_cast<size_t>(route)];
                row.tile(row, pair * 2);
                row.tile(row, pair * 2 + 1);
            }
        }
        else
        {
#pragma omp for schedule(static)
            for (int task = 0; task < down_tasks; ++task)
            {
                const int chunk = task / route_count_local;
                const int route = task % route_count_local;
                const auto &row = active_down_plans[static_cast<size_t>(route)];
                row.tile(row, chunk);
            }
        }
    };

    if (k_split == 2 || k_split == 4)
    {
        const int parts_per_split = plan.gate_partitions / k_split;
        const int tail_parts = plan.gate_partitions - parts_per_split;
        const int gate_tasks = gate_tiles * k_split;

#pragma omp for schedule(static)
        for (int task = 0; task < gate_tasks; ++task)
        {
            const int tile = task / k_split;
            const int split = task % k_split;
            const auto &row = active_gate_plans[tile / tasks_per_route];
            const int chunk = tile % tasks_per_route;
            const int p_start = split * parts_per_split;
            const int p_end = p_start + parts_per_split;
            if (split == 0)
            {
                row.part_range(row, chunk, 0, parts_per_split, row.output + chunk * columns_per_task, true);
            }
            else
            {
                float *dst = splitk_partials +
                    (static_cast<size_t>(tile) * static_cast<size_t>(tail_parts) +
                     static_cast<size_t>(p_start - parts_per_split)) *
                    columns_per_task;
                row.part_range(row, chunk, p_start, p_end, dst, false);
            }
        }
#pragma omp for schedule(static)
        for (int task = 0; task < activation_tasks; ++task)
        {
            const int route = task / tasks_per_route;
            const int chunk = task % tasks_per_route;
            const int offset = chunk * columns_per_task;
            const size_t gate_tile = static_cast<size_t>(2 * route) * tasks_per_route + static_cast<size_t>(chunk);
            const size_t up_tile = static_cast<size_t>(2 * route + 1) * tasks_per_route + static_cast<size_t>(chunk);
            float *const gate_dst = active_gate_plans[2 * route].output + offset;
            float *const up_dst = active_gate_plans[2 * route + 1].output + offset;
            for (int p = 0; p < tail_parts; ++p)
            {
                addOrdered64Floats(gate_dst,
                    splitk_partials + (gate_tile * static_cast<size_t>(tail_parts) + static_cast<size_t>(p)) * columns_per_task, isa);
                addOrdered64Floats(up_dst,
                    splitk_partials + (up_tile * static_cast<size_t>(tail_parts) + static_cast<size_t>(p)) * columns_per_task, isa);
            }
            alignas(64) float values[columns_per_task];
            computeSwiGLU64(gate_dst, up_dst, values, down_plans[route].weights->numerical_policy);
            auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
            quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
        }
        run_down_plans();
        return true;
    }

    const int route_count = static_cast<int>(plan.route_count);
    const int gate_row_count = static_cast<int>(active_gate_plans.size());
    const int workers = plan.workers;
    const int fused_peak = 2 * ((activation_tasks + workers - 1) / workers);
    const int unfused_peak = (gate_tiles + workers - 1) / workers;
    const bool fuse_gate_up_swiglu = (fused_peak == unfused_peak);
    if (fuse_gate_up_swiglu)
    {
        // Chunk-major scheduling spreads heterogeneous codebooks (e.g., a Q8_0
        // shared expert fused alongside Q4_K/Q5_1 routed experts) evenly across
        // all workers so no single thread inherits a full heavy-route tail.
#pragma omp for schedule(static)
        for (int task = 0; task < activation_tasks; ++task)
        {
            const int chunk = task / route_count;
            const int route = task % route_count;
            const int offset = chunk * columns_per_task;
            const auto &g_row = active_gate_plans[2 * route];
            const auto &u_row = active_gate_plans[2 * route + 1];
            g_row.tile(g_row, chunk);
            u_row.tile(u_row, chunk);
            alignas(64) float values[columns_per_task];
            computeSwiGLU64(g_row.output + offset, u_row.output + offset, values,
                            down_plans[route].weights->numerical_policy);
            auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
            quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
        }
    }
    else
    {
#pragma omp for schedule(static)
        for (int task = 0; task < gate_tiles; ++task)
        {
            const int chunk = task / gate_row_count;
            const int row_idx = task % gate_row_count;
            const auto &row = active_gate_plans[static_cast<size_t>(row_idx)];
            row.tile(row, chunk);
        }
#pragma omp for schedule(static)
        for (int task = 0; task < activation_tasks; ++task)
        {
            const int chunk = task / route_count;
            const int route = task % route_count;
            const int offset = chunk * columns_per_task;
            alignas(64) float values[columns_per_task];
            computeSwiGLU64(active_gate_plans[2 * route].output + offset,
                active_gate_plans[2 * route + 1].output + offset, values,
                down_plans[route].weights->numerical_policy);
            auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
            quantize64ValuesToTwoQ8Blocks(values, q8, down_plans[route].weights->numerical_policy, isa);
        }
    }
    run_down_plans();
    return true;
}

bool CPUNativeQuantisedGemmKernel::executeStandaloneMoEOneRowInTeam(
    std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down,
    std::span<Q8_1Block> activation,
    int hidden, int intermediate,
    float *splitk_partials, size_t splitk_partials_capacity)
{
    StandaloneMoEOneRowPlan plan{};
    if (!prepareStandaloneMoEOneRowPlan(
            plan, gate, down, hidden, intermediate,
            std::max(1, omp_get_num_threads()), activeISALevel(),
            splitk_partials ? splitk_partials_capacity : 0u))
        return false;
    return executePreparedStandaloneMoEOneRowInTeam(
        plan, activation, splitk_partials, splitk_partials_capacity);
}
} // namespace llaminar2::cpu::native_quantised
