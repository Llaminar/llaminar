/**
 * @file CPUNativeQuantisedStandaloneMoE.cpp
 * @brief Execute native one-token MoE expert phases on one fixed physical worker team.
 *
 * Admission binds original ordered dot entries and persistent route storage for
 * any admitted `(hidden, intermediate)` geometry divisible by the 64-column
 * AVX-512 tile width. Keeping the specialized tile bodies in this compilation
 * unit gives the existing grouped verifier independent compiler inlining and
 * register budgets.
 */
#include "CPUNativeQuantisedGemmKernel.h"
#include "CPUNativeQuantisedGemv.h"
#include "execution/moe/MoERuntimeTable.h"

namespace llaminar2::cpu::native_quantised
{
int CPUNativeQuantisedGemmKernel::standaloneMoEOneRowWorkers(int hidden, int intermediate, size_t routes) noexcept
{
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    if (hidden >= 64 && intermediate >= 64 && (hidden % 64) == 0 && (intermediate % 64) == 0 &&
        routes && routes <= kDeviceMoEMaxTopK &&
        activeISALevel() >= ISALevel::AVX512 && !llaminar::v2::inLexicalWorkshare() &&
        !omp_get_dynamic() && cpuNativeQuantisedSerialOutputPartitionN() == 0)
    {
        const int gate_blocks = hidden / 32;
        const int down_blocks = intermediate / 32;
        const int gate_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(hidden);
        const int down_parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(intermediate);
        if (gate_parts > 0 && down_parts > 0 &&
            (gate_blocks % gate_parts) == 0 && (down_blocks % down_parts) == 0)
        {
            const int workers = omp_get_max_threads();
            if (workers > 0 && workers <= omp_get_thread_limit())
                return workers;
        }
    }
#else
    (void)hidden; (void)intermediate; (void)routes;
#endif
    return 0;
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
/** @brief Original one-row operands and exact native tile entry, resolved on the invoking owner. */
struct CPUNativeQuantisedGemmKernel::PreparedMoERow
{
    const CPUNativeQuantisedPackedWeights *weights; ///< Borrowed prepared weight owner.
    const Q8_1Block *input; ///< Original router Q8 row or the ordered SwiGLU publication.
    float *output; ///< One phase's original scratch row with disjoint N64 owners.
    void (*tile)(const PreparedMoERow &, int); ///< Native entry bound before the workers start.
    void (*part_range)(const PreparedMoERow &, int, int, int, float *, bool); ///< Ordered partition range entry for split-K.
    __m512i lut; ///< Immutable codebook lookup for entries that consume nibbles.
    int blocks; ///< Exact 32-element Q8_1 K-block count (`K / 32`).
    int partitions; ///< Ordered K-partition count from MoEProjectionNumericalContract.
};

/**
 * @brief Run one complete N64 tile with the selected source and original ordered partitions.
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
    float *output = row.output + chunk * columns;
    if (partitions == blocks)
    {
        if constexpr (Nibble)
            gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, output, chunk, 0, blocks,
                row.lut, false);
        else
            gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, blocks, false);
    }
    else if constexpr (Nibble)
        dispatch_native_avx512_ordered_partitions(*row.weights, row.input, output, chunk, blocks,
            partitions, blocks / partitions, row.lut);
    else
        gemvNonNibbleAVX512OrderedPartitions(*row.weights, row.input, output, chunk, blocks,
            partitions, blocks / partitions);
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
    const int bpt = row.blocks / row.partitions;
    if (fold_into_dst)
    {
        if (p_start == 0 && p_end == row.partitions)
        {
            if constexpr (Nibble)
                dispatch_native_avx512_ordered_partitions(*row.weights, row.input, dst, chunk,
                    row.blocks, row.partitions, bpt, row.lut);
            else
                gemvNonNibbleAVX512OrderedPartitions(*row.weights, row.input, dst, chunk,
                    row.blocks, row.partitions, bpt);
            return;
        }
        alignas(64) float part_buf[64];
        for (int p = p_start; p < p_end; ++p)
        {
            const int kb_start = p * bpt;
            const int kb_end = kb_start + bpt;
            float *out = (p == p_start) ? dst : part_buf;
            if constexpr (Nibble)
                gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, row.lut, false);
            else
                gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, out, chunk,
                    kb_start, kb_end, false);
            if (p > p_start)
            {
                _mm512_storeu_ps(dst, addOrderedAVX512(_mm512_loadu_ps(dst), _mm512_load_ps(part_buf)));
                _mm512_storeu_ps(dst + 16, addOrderedAVX512(_mm512_loadu_ps(dst + 16), _mm512_load_ps(part_buf + 16)));
                _mm512_storeu_ps(dst + 32, addOrderedAVX512(_mm512_loadu_ps(dst + 32), _mm512_load_ps(part_buf + 32)));
                _mm512_storeu_ps(dst + 48, addOrderedAVX512(_mm512_loadu_ps(dst + 48), _mm512_load_ps(part_buf + 48)));
            }
        }
        return;
    }
    for (int p = p_start; p < p_end; ++p)
    {
        float *out = dst + static_cast<size_t>(p - p_start) * 64u;
        const int kb_start = p * bpt;
        const int kb_end = kb_start + bpt;
        if constexpr (Nibble)
            gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, out, chunk,
                kb_start, kb_end, row.lut, false);
        else
            gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, out, chunk,
                kb_start, kb_end, false);
    }
}

/**
 * @brief Bind admitted source operands and their immutable lookup before forking.
 * @param original Borrowed live projection operands in original route order.
 * @param prepared Invocation-owned destination for that exact live prefix.
 * @param width Number of output columns of each projection (multiple of 64).
 * @param blocks Number of input Q8 blocks in each source row (`K / 32`).
 * @param workers Fixed standalone team width used by the original tile policy.
 * @return Whether all members admit the measured device-ordered schedule.
 */
bool CPUNativeQuantisedGemmKernel::prepareMoERows(std::span<const BatchedPrequantizedProjectionDesc> original,
    std::span<PreparedMoERow> prepared, int width, int blocks, int workers)
{
    const int k = blocks * 32;
    const int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
    if (partitions <= 0 || (blocks % partitions) != 0)
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
            member.purpose != ProjectionRowsPurpose::Decode || member.verifier_schedule != VerifierRowsPolicy::Auto ||
            weights.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert ||
            serialTileConfigForPreparedProjection(weights.preparedFootprint(), weights.codebook_id,
                weights.numerical_policy, width, k, workers, cache).k_tiles != partitions)
            return false;
        const bool nibble = weights.usesNibbleLUT();
        const bool needs_lut = nibble && nibbleDecodeKind(weights.codebook_id) == NibbleDecodeKind::LUT;
        prepared[projection] = {&weights, member.input_q8, member.output,
            nibble ? preparedMoERowTile<true> : preparedMoERowTile<false>,
            nibble ? preparedMoERowPartitionRange<true> : preparedMoERowPartitionRange<false>,
            needs_lut ? build_decode_lut(weights.codebook_id) : _mm512_setzero_si512(),
            blocks, partitions};
    }
    return true;
}

/**
 * @brief Each worker executes its exclusive output tiles in the already launched team.
 * @tparam Join Whether the following phase needs a separate completion edge.
 * @param rows Exact live projection prefix.
 * @param width Output columns in each prepared row (multiple of 64).
 */
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

/**
 * @brief Select the smallest Gate/Up K-split in `{1, 2, 4}` that meaningfully cuts worker tail imbalance.
 *
 * Each split multiplies task count by `K` while keeping exact left-to-right FP32 reduction order.
 */
inline int selectStandaloneGateUpKSplit(int gate_tiles, int gate_partitions, int workers, size_t max_splitk_floats) noexcept
{
    // Once each worker already owns >= 2 full N64 tiles, K-splitting adds stack
    // partial-buffer pressure, an extra barrier, and smaller K-block calls without
    // improving load balance. Reserve K-split for small route counts where cores
    // would otherwise sit idle.
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

/// Stack-bounded split-K scratch budget (floats): covers up to 160 gate tiles × 12 tail partitions × 64 cols (~490 KB).
static constexpr size_t kStandaloneSplitKMaxFloats = 160u * 12u * 64u;

/**
 * @brief Execute Gate/Up with exact bit-identical `KSplit ∈ {2, 4}` partition parallelism.
 *
 * Kept `noinline` so the `KSplit == 1` path never allocates or touches the partial scratch frame.
 */
template<int KSplit>
__attribute__((noinline))
bool CPUNativeQuantisedGemmKernel::runPreparedFixedMoETilesSplitK(
    std::span<const PreparedMoERow> gate_plans,
    std::span<const PreparedMoERow> down_plans,
    std::span<Q8_1Block> activation,
    int hidden,
    int intermediate,
    int gate_partitions,
    int workers)
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
        __m512 r0 = _mm512_loadu_ps(dst);
        __m512 r1 = _mm512_loadu_ps(dst + 16);
        __m512 r2 = _mm512_loadu_ps(dst + 32);
        __m512 r3 = _mm512_loadu_ps(dst + 48);
        for (int p = 0; p < tail_parts; ++p)
        {
            const float *src = tile_partials + static_cast<size_t>(p) * columns_per_task;
            r0 = addOrderedAVX512(r0, _mm512_load_ps(src));
            r1 = addOrderedAVX512(r1, _mm512_load_ps(src + 16));
            r2 = addOrderedAVX512(r2, _mm512_load_ps(src + 32));
            r3 = addOrderedAVX512(r3, _mm512_load_ps(src + 48));
        }
        _mm512_storeu_ps(dst, r0);
        _mm512_storeu_ps(dst + 16, r1);
        _mm512_storeu_ps(dst + 32, r2);
        _mm512_storeu_ps(dst + 48, r3);
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
            primitives::compute_swiglu_gpu_aligned_expert_serial(gate_dst, up_dst, values, columns_per_task);
            auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
            gpu_aligned_expert_q8::quantizeTwoBlocks(values, q8[0], q8[1]);
        }
        executeMoERows<false>(down_plans, hidden);
    };
    OMP_WORKSHARE_REGION(run);
    return valid.load(std::memory_order_relaxed);
}

/**
 * @brief Execute a standalone one-row FFN on its original arithmetic and publication order.
 * @param gate Original Gate/Up descriptors whose K is `hidden`.
 * @param down Original Down descriptors whose K is `intermediate`.
 * @param activation Exact canonical intermediate activation bank published between the projections.
 * @param hidden Hidden dimension (multiple of 64).
 * @param intermediate Local shard intermediate dimension (multiple of 64).
 * @param workers Fixed OpenMP team selected before either phase is bound.
 * @return Empty before writes for other supported schedules, or exact team completion.
 */
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
    std::array<PreparedMoERow, 2 * kDeviceMoEMaxTopK> gate_storage;
    std::array<PreparedMoERow, kDeviceMoEMaxTopK> down_storage;
    auto gate_plans = std::span(gate_storage).first(gate.size());
    auto down_plans = std::span(down_storage).first(down.size());
    const int gate_blocks = hidden / 32;
    const int down_blocks = intermediate / 32;
    if (!prepareMoERows(gate, gate_plans, intermediate, gate_blocks, workers) ||
        !prepareMoERows(down, down_plans, hidden, down_blocks, workers))
        return std::nullopt;

    const int gate_tiles = static_cast<int>(gate_plans.size()) * tasks_per_route;
    const int gate_partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(hidden);
    const int k_split = selectStandaloneGateUpKSplit(gate_tiles, gate_partitions, workers, kStandaloneSplitKMaxFloats);
    if (k_split == 4)
        return runPreparedFixedMoETilesSplitK<4>(gate_plans, down_plans, activation, hidden, intermediate, gate_partitions, workers);
    if (k_split == 2)
        return runPreparedFixedMoETilesSplitK<2>(gate_plans, down_plans, activation, hidden, intermediate, gate_partitions, workers);

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
                primitives::compute_swiglu_gpu_aligned_expert_serial(
                    g_row.output + offset, u_row.output + offset, values, columns_per_task);
                auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
                gpu_aligned_expert_q8::quantizeTwoBlocks(values, q8[0], q8[1]);
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
                primitives::compute_swiglu_gpu_aligned_expert_serial(gate_plans[2 * route].output + offset,
                    gate_plans[2 * route + 1].output + offset, values, columns_per_task);
                auto *q8 = activation.data() + route * blocks_per_route + offset / Q8_1Block::BLOCK_SIZE;
                gpu_aligned_expert_q8::quantizeTwoBlocks(values, q8[0], q8[1]);
            }
        }
        // The standalone team join retires all final Down stores.
        executeMoERows<false>(down_plans, hidden);
    };
    OMP_WORKSHARE_REGION(run);
    return valid.load(std::memory_order_relaxed);
}
#endif

std::optional<bool> CPUNativeQuantisedGemmKernel::tryExecuteStandaloneMoEOneRow(std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation,
    int hidden, int intermediate)
{
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    if (down.size() <= kDeviceMoEMaxTopK && gate.size() == 2 * down.size())
    {
        const int workers = standaloneMoEOneRowWorkers(hidden, intermediate, down.size());
        if (!workers) return std::nullopt;
        const auto result = runPreparedFixedMoETiles(gate, down, activation, hidden, intermediate, workers);
        if (result.value_or(false) && PerfStatsCollector::isDomainEnabled("kernel"))
        {
            PerfStatsCollector::addCounter("kernel", "cpu_native_quantised_moe_prepared_phase_plans",
                2.0, "gemm", "cpu");
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_quantised_router_q8_grouped_decode_equivalent_projection_calls",
                1.0,
                "gemm",
                "cpu",
                {{"m", "1"},
                 {"k", std::to_string(hidden)},
                 {"projections", std::to_string(gate.size())},
                 {"path", "decode_standalone"}});
        }
        return result;
    }
#else
    (void)gate; (void)down; (void)activation; (void)hidden; (void)intermediate;
#endif
    return std::nullopt;
}
} // namespace llaminar2::cpu::native_quantised
