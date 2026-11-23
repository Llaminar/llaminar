/**
 * @file CPUNativeQuantisedStandaloneMoE.cpp
 * @brief Execute native one-token QFN expert phases on one fixed physical worker team.
 *
 * Admission binds original ordered dot entries and persistent route storage.
 * Keeping the specialized tile bodies in this compilation unit gives the
 * existing grouped verifier independent compiler inlining and register budgets.
 */
#include "CPUNativeQuantisedGemmKernel.h"
#include "execution/moe/MoERuntimeTable.h"

namespace llaminar2::cpu::native_quantised
{
int CPUNativeQuantisedGemmKernel::standaloneMoEOneRowWorkers(int hidden, int intermediate, size_t routes) noexcept
{
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    constexpr size_t panels_per_route = 2 * (320 / 64);
    if (hidden == 2560 && intermediate == 320 && routes && routes <= kDeviceMoEMaxTopK &&
        activeISALevel() >= ISALevel::AVX512 && !llaminar::v2::inLexicalWorkshare() &&
        !omp_get_dynamic() && cpuNativeQuantisedSerialOutputPartitionN() == 0)
    {
        const int workers = omp_get_max_threads();
        if (workers > 0 && workers <= omp_get_thread_limit() &&
            routes >= (static_cast<size_t>(workers) - 1) / panels_per_route + 1)
            return workers;
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
    __m512i lut; ///< Immutable codebook lookup for entries that consume nibbles.
};

/**
 * @brief Run one complete N64 tile with the selected source and original ordered partitions.
 * @tparam Blocks Number of Q8 blocks consumed in each output tile.
 * @tparam Nibble Whether the bound reader consumes a nibble lookup table.
 * @param row Prepared operands for one live projection.
 * @param chunk Disjoint output column tile owned by this worker.
 */
template<int Blocks, bool Nibble>
void CPUNativeQuantisedGemmKernel::preparedMoERowTile(const PreparedMoERow &row, int chunk)
{
    constexpr int columns = 64;
    constexpr int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(Blocks * 32);
    float *output = row.output + chunk * columns;
    if constexpr (partitions == Blocks)
    {
        if constexpr (Nibble)
            gemv_native_quantised_avx512_chunk_native(*row.weights, row.input, output, chunk, 0, Blocks,
                row.lut, false);
        else
            gemv_native_quantised_avx512_chunk_non_nibble(*row.weights, row.input, output, chunk, 0, Blocks, false);
    }
    else if constexpr (Nibble)
        dispatch_native_avx512_ordered_partitions(*row.weights, row.input, output, chunk, Blocks,
            partitions, Blocks / partitions, row.lut);
    else
        gemvNonNibbleAVX512OrderedPartitions(*row.weights, row.input, output, chunk, Blocks,
            partitions, Blocks / partitions);
}

/**
 * @brief Bind fixed-shape source operands and their immutable lookup before forking.
 * @tparam Width Number of output columns of each projection.
 * @tparam Blocks Number of input Q8 blocks in each source row.
 * @param original Borrowed live projection operands in original route order.
 * @param prepared Invocation-owned destination for that exact live prefix.
 * @param workers Fixed standalone team width used by the original tile policy.
 * @return Whether all members admit the measured device-ordered schedule.
 */
template<int Width, int Blocks>
bool CPUNativeQuantisedGemmKernel::prepareMoERows(std::span<const BatchedPrequantizedProjectionDesc> original,
    std::span<PreparedMoERow> prepared, int workers)
{
    constexpr int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(Blocks * 32);
    for (size_t projection = 0; projection < original.size(); ++projection)
    {
        const auto &member = original[projection];
        if (!member.kernel || !member.kernel->valid_ || !member.input_q8 || !member.output)
            throw std::invalid_argument("Prepared standalone CPU MoE projection lacks its complete native operands");
        const auto &weights = member.kernel->packed_;
        if (member.rows != 1 || member.n != Width || member.ldc < Width || member.bias || member.kernel->activation_rotation_ ||
            weights.N != Width || weights.K != Blocks * 32 ||
            member.purpose != ProjectionRowsPurpose::Decode || member.verifier_schedule != VerifierRowsPolicy::Auto ||
            weights.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert ||
            serialTileConfigForPackedMatrix(weights, Width, Blocks * 32, workers).k_tiles != partitions)
            return false;
        const bool nibble = weights.usesNibbleLUT();
        prepared[projection] = {&weights, member.input_q8, member.output,
            nibble ? preparedMoERowTile<Blocks, true> : preparedMoERowTile<Blocks, false>,
            nibble ? build_decode_lut(weights.codebook_id) : _mm512_setzero_si512()};
    }
    return true;
}

/**
 * @brief Each worker executes its exclusive output tiles in the already launched team.
 * @tparam Width Output columns in each prepared row.
 * @tparam Join Whether the following phase needs a separate completion edge.
 * @param rows Exact live projection prefix.
 */
template<int Width, bool Join>
void CPUNativeQuantisedGemmKernel::executeMoERows(std::span<const PreparedMoERow> rows)
{
    constexpr int tiles = Width / 64;
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
 * @brief Execute a fixed-shape standalone FFN on its original arithmetic and publication order.
 * @param gate Original Gate/Up descriptors whose K is 2560.
 * @param down Original Down descriptors whose K is 320.
 * @param activation Exact canonical intermediate activation bank published between the projections.
 * @param workers Fixed OpenMP team selected before either phase is bound.
 * @return Empty before writes for other supported schedules, or exact team completion.
 */
std::optional<bool> CPUNativeQuantisedGemmKernel::runPreparedFixedMoETiles(
    std::span<const BatchedPrequantizedProjectionDesc> gate,
    std::span<const BatchedPrequantizedProjectionDesc> down, std::span<Q8_1Block> activation, int workers)
{
    constexpr int columns_per_task = 64, tasks_per_route = 320 / columns_per_task;
    constexpr size_t blocks_per_route = 320 / Q8_1Block::BLOCK_SIZE;
    if (activation.size() < down.size() * blocks_per_route)
        throw std::invalid_argument("Prepared standalone CPU MoE requires the admitted live SwiGLU bank");
    for (size_t route = 0; route < down.size(); ++route)
    {
        if (down[route].input_q8 != activation.data() + route * blocks_per_route)
            throw std::invalid_argument("Prepared standalone CPU MoE Down must consume its own SwiGLU row");
        const storage_proof::Bytes output{down[route].input_q8, blocks_per_route * sizeof(Q8_1Block)};
        storage_proof::requireDisjoint(output, {gate[2 * route].output, 320 * sizeof(float)});
        storage_proof::requireDisjoint(output, {gate[2 * route + 1].output, 320 * sizeof(float)});
    }
    std::array<PreparedMoERow, 2 * kDeviceMoEMaxTopK> gate_storage;
    std::array<PreparedMoERow, kDeviceMoEMaxTopK> down_storage;
    auto gate_plans = std::span(gate_storage).first(gate.size());
    auto down_plans = std::span(down_storage).first(down.size());
    if (!prepareMoERows<320, 80>(gate, gate_plans, workers) ||
        !prepareMoERows<2560, 10>(down, down_plans, workers))
        return std::nullopt;
    const int activation_tasks = static_cast<int>(down.size()) * tasks_per_route;
    std::atomic<bool> valid{true};
    const auto run = [&]() {
        if (omp_get_num_threads() != workers)
        {
            valid.store(false, std::memory_order_relaxed);
            return;
        }
        executeMoERows<320, true>(gate_plans);
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
        // The standalone team join retires all final Down stores.
        executeMoERows<2560, false>(down_plans);
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
        const auto result = runPreparedFixedMoETiles(gate, down, activation, workers);
        if (result.value_or(false) && PerfStatsCollector::isDomainEnabled("kernel"))
            PerfStatsCollector::addCounter("kernel", "cpu_native_quantised_moe_prepared_phase_plans",
                2.0, "gemm", "cpu");
        return result;
    }
#else
    (void)gate; (void)down; (void)activation; (void)hidden; (void)intermediate;
#endif
    return std::nullopt;
}
} // namespace llaminar2::cpu::native_quantised
