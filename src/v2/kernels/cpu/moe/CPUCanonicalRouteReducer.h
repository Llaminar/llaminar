/**
 * @file CPUCanonicalRouteReducer.h
 * @brief Stable-team, ordered-FMA reduction for canonical CPU MoE routes.
 *
 * CPU participant-assigned routed execution publishes one unweighted expert
 * row for every
 * live router slot. The fixed root folds those rows in original router order
 * before broadcasting one compact hidden row per token. This interface owns
 * the CPU launch geometry for that fold so graph stages describe the
 * arithmetic contract without constructing ad hoc OpenMP teams.
 *
 * Every output element visits route slots in increasing top-k order and uses
 * the same FP32 FMA primitive as serial decode. Parallelism is exclusively
 * across independent output-column tiles, making the grouped result byte
 * identical while exposing enough work to a socket-wide persistent team.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace llaminar2::cpu::moe
{
    /**
     * @brief Physical executor selected for one canonical route fold.
     */
    enum class CanonicalRouteFoldExecutor : uint8_t
    {
        CallerThread,
        ExistingOpenMPTeam,
        FullOpenMPTeam,
    };

    /**
     * @brief Immutable launch plan derived from runtime geometry and team size.
     */
    struct CanonicalRouteFoldPlan
    {
        CanonicalRouteFoldExecutor executor =
            CanonicalRouteFoldExecutor::CallerThread;
        int column_tile = 0;
        int column_tiles_per_row = 0;
        size_t parallel_tasks = 0;
        int team_threads = 1;
    };

    /**
     * @brief Inputs for an ordered canonical route reduction.
     *
     * `record_for_route_slot` selects the packed rooted-gather layout when
     * non-null. Each live flat route slot maps to one packed record index. A
     * null mapping selects dense original-route-slot storage, where record
     * `slot` starts at `route_rows + slot * columns`.
     */
    struct CanonicalRouteFoldArgs
    {
        const float *route_rows = nullptr;
        const float *route_weights = nullptr;
        const int32_t *record_for_route_slot = nullptr;
        float *output = nullptr;
        int rows = 0;
        int top_k = 0;
        int columns = 0;
    };

    /**
     * @brief Build a total launch plan for every positive fold geometry.
     *
     * Tiny folds remain on the caller so they neither fork nor resize an
     * OpenMP team. Larger folds use either the already-active team or the
     * process-wide configured team. The planner deliberately has no bounded
     * team state: alternating a small root-only team with full-socket GEMMs
     * causes libgomp to retire and recreate workers on every model layer.
     *
     * @param rows Number of independent output rows.
     * @param top_k Ordered router slots contributing to each row.
     * @param columns Hidden width of each row.
     * @return Complete executor and column-tile geometry.
     */
    CanonicalRouteFoldPlan planCanonicalRouteFold(
        int rows,
        int top_k,
        int columns);

    /**
     * @brief Fold canonical route rows with serial-decode-equivalent FP32 math.
     *
     * The function performs no allocation and never creates a sub-maximal
     * OpenMP team. Packed mappings are validated before parallel execution so
     * malformed publication state fails before any output is mutated.
     *
     * @param args Validated route tensors, dimensions, and optional packed map.
     * @param used_plan Optional destination for the exact selected launch plan.
     * @return true on complete reduction; false for an invalid contract.
     */
    bool reduceCanonicalRouteRowsOrderedFMA(
        const CanonicalRouteFoldArgs &args,
        CanonicalRouteFoldPlan *used_plan = nullptr);

    /**
     * @brief Original-route partial rows from one member of an admitted expert shard cohort.
     * The caller orders these immutable spans by the original cohort membership;
     * each holds only live unweighted [row,top_k,hidden] down-projection values.
     * This is a borrowed execution input, never placement or allocation authority.
     */
    struct CanonicalRouteDownPartialRows
    {
        std::span<const float> values;
        /// Optional original-route -> packed-record map; negative only for inactive routes.
        std::span<const int32_t> record_for_route_slot{};
    };

    /** @brief Fixed-order partial reduction followed by the unchanged canonical top-k fold. */
    struct CanonicalRouteDownPartialFoldArgs
    {
        std::span<const CanonicalRouteDownPartialRows> ordered_participants;
        std::span<float> combined_route_rows; ///< Caller-owned existing PMA-admitted scratch.
        const float *route_weights = nullptr;
        float *output = nullptr;
        int rows = 0; ///< Actual live rows, never allocation capacity; zero writes nothing.
        int top_k = 0;
        int columns = 0;
    };

    /** @brief Unweighted fixed-order combine, before any original top-k weighting. */
    struct CanonicalRouteDownPartialCombineArgs {
        std::span<const CanonicalRouteDownPartialRows> ordered_participants;
        std::span<float> combined_route_rows;
        const float *route_weights = nullptr; ///< Zero marks an absent original route.
        int rows = 0;
        int top_k = 0;
        int columns = 0;
    };
    /**
     * @brief Combine original cohort partials in FP32 participant order only.
     * @param args Actual original-route geometry, immutable banks and admitted output.
     * @return False before output mutation for incomplete/overlapping/malformed inputs.
     */
    bool combineCanonicalRouteDownPartialsOrdered(
        const CanonicalRouteDownPartialCombineArgs &args);

    /**
     * @brief Combine each expert/route in original participant order before weighting it.
     * @param args Complete dense original-route banks, live geometry and admitted scratch.
     * @param used_plan Optional observation of the existing canonical fold launch geometry.
     * @return False before any mutation for invalid extents or overlapping output/input banks.
     *
     * Every element adds participant0,1,... in FP32 order. Only then does the
     * ordinary serial-row-equivalent ordered-FMA reducer apply original top-k
     * weights. Folding local experts before adding shards is a different tree
     * and is deliberately not implemented. No storage or worker is allocated.
     */
    bool reduceCanonicalRouteDownPartialsOrderedFMA(
        const CanonicalRouteDownPartialFoldArgs &args,
        CanonicalRouteFoldPlan *used_plan = nullptr);

    /**
     * @brief Return a stable diagnostic name for a fold executor.
     */
    const char *canonicalRouteFoldExecutorName(
        CanonicalRouteFoldExecutor executor) noexcept;
} // namespace llaminar2::cpu::moe
