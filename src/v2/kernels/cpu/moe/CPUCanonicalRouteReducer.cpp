/**
 * @file CPUCanonicalRouteReducer.cpp
 * @brief Stable-team implementation of ordered canonical CPU MoE reduction.
 */

#include "CPUCanonicalRouteReducer.h"

#include "../../../execution/moe/CanonicalMoERouteRecord.h"
#include "../primitives/VectorPrimitives.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <omp.h>

namespace llaminar2::cpu::moe
{
    namespace
    {
        /**
         * Keep a tiny fold on the caller when its complete arithmetic fits in
         * roughly one decode-sized hidden-row transaction. Skipping a parallel
         * region leaves an existing libgomp pool intact; unlike a bounded team,
         * it cannot force the next full-socket GEMM to recreate workers.
         */
        constexpr size_t kCallerThreadFMAThreshold = 16u * 1024u;

        /**
         * Column boundaries are rounded to the widest compiled FP32 vector.
         * Consequently splitting a row never turns an element that was handled
         * by AVX-512/AVX2 in serial decode into a scalar-tail operation.
         */
        constexpr int kColumnAlignment = 32;

        /** Round a positive integer up to the next vector-aligned boundary. */
        int roundColumnsUp(int columns)
        {
            return ((columns + kColumnAlignment - 1) / kColumnAlignment) *
                   kColumnAlignment;
        }

        /** Return one packed or dense expert row for a flat route slot. */
        const float *routeRow(
            const CanonicalRouteFoldArgs &args,
            size_t flat_slot)
        {
            if (!args.record_for_route_slot)
            {
                return args.route_rows +
                       flat_slot * static_cast<size_t>(args.columns);
            }
            return canonical_moe_route_record::record(
                args.route_rows,
                static_cast<size_t>(
                    args.record_for_route_slot[flat_slot]),
                args.columns);
        }
    } // namespace

    CanonicalRouteFoldPlan planCanonicalRouteFold(
        int rows,
        int top_k,
        int columns)
    {
        if (rows <= 0 || top_k <= 0 || columns <= 0)
            return {};

        const bool existing_team = omp_in_parallel() != 0;
        const int team_threads = std::max(
            1,
            existing_team ? omp_get_num_threads() : omp_get_max_threads());
        const size_t fma_count =
            static_cast<size_t>(rows) * static_cast<size_t>(top_k) *
            static_cast<size_t>(columns);
        const bool caller_thread =
            !existing_team &&
            (team_threads == 1 || fma_count <= kCallerThreadFMAThreshold);

        /*
         * Aim for one independent task per worker. Each task owns a vector-
         * aligned column interval and applies every route in canonical order.
         * As M grows, the tile naturally expands until one whole row is a task;
         * no model width, socket width, or MTP depth is hard-coded.
         */
        const int target_tasks = caller_thread ? 1 : team_threads;
        const int target_tiles_per_row = std::max(
            1,
            (target_tasks + rows - 1) / rows);
        const int unaligned_tile = std::max(
            1,
            (columns + target_tiles_per_row - 1) /
                target_tiles_per_row);
        const int column_tile = std::min(
            columns,
            std::max(kColumnAlignment, roundColumnsUp(unaligned_tile)));
        const int column_tiles_per_row =
            (columns + column_tile - 1) / column_tile;

        return {
            .executor = caller_thread
                            ? CanonicalRouteFoldExecutor::CallerThread
                        : existing_team
                            ? CanonicalRouteFoldExecutor::ExistingOpenMPTeam
                            : CanonicalRouteFoldExecutor::FullOpenMPTeam,
            .column_tile = column_tile,
            .column_tiles_per_row = column_tiles_per_row,
            .parallel_tasks =
                static_cast<size_t>(rows) *
                static_cast<size_t>(column_tiles_per_row),
            .team_threads = team_threads,
        };
    }

    bool reduceCanonicalRouteRowsOrderedFMA(
        const CanonicalRouteFoldArgs &args,
        CanonicalRouteFoldPlan *used_plan)
    {
        if (!args.route_rows || !args.route_weights || !args.output ||
            args.rows <= 0 || args.top_k <= 0 || args.columns <= 0)
        {
            return false;
        }

        const size_t route_slot_count =
            static_cast<size_t>(args.rows) *
            static_cast<size_t>(args.top_k);
        if (args.record_for_route_slot)
        {
            for (size_t slot = 0; slot < route_slot_count; ++slot)
            {
                if (args.route_weights[slot] != 0.0f &&
                    args.record_for_route_slot[slot] < 0)
                {
                    return false;
                }
            }
        }

        const CanonicalRouteFoldPlan plan = planCanonicalRouteFold(
            args.rows,
            args.top_k,
            args.columns);
        if (plan.column_tile <= 0 ||
            plan.column_tiles_per_row <= 0 ||
            plan.parallel_tasks == 0u ||
            plan.parallel_tasks >
                static_cast<size_t>(std::numeric_limits<long long>::max()))
        {
            return false;
        }
        if (used_plan)
            *used_plan = plan;

        const auto reduce_task = [&](size_t task)
        {
            const int row = static_cast<int>(
                task / static_cast<size_t>(plan.column_tiles_per_row));
            const int column_tile = static_cast<int>(
                task % static_cast<size_t>(plan.column_tiles_per_row));
            const int column_begin = column_tile * plan.column_tile;
            const int column_count = std::min(
                plan.column_tile,
                args.columns - column_begin);
            float *const destination =
                args.output + static_cast<size_t>(row) * args.columns +
                column_begin;

            std::fill_n(destination, column_count, 0.0f);
            const size_t first_slot =
                static_cast<size_t>(row) *
                static_cast<size_t>(args.top_k);
            for (int route = 0; route < args.top_k; ++route)
            {
                const size_t slot = first_slot + static_cast<size_t>(route);
                const float weight = args.route_weights[slot];
                if (weight == 0.0f)
                    continue;
                primitives::vec_axpy(
                    destination,
                    routeRow(args, slot) + column_begin,
                    weight,
                    column_count);
            }
        };

        if (plan.executor == CanonicalRouteFoldExecutor::CallerThread)
        {
            for (size_t task = 0; task < plan.parallel_tasks; ++task)
                reduce_task(task);
            return true;
        }

        auto reduce_tasks = [&]()
        {
#pragma omp for schedule(static)
            for (long long task = 0;
                 task < static_cast<long long>(plan.parallel_tasks);
                 ++task)
            {
                reduce_task(static_cast<size_t>(task));
            }
        };
        OMP_WORKSHARE_REGION(reduce_tasks);
        return true;
    }

    const char *canonicalRouteFoldExecutorName(
        CanonicalRouteFoldExecutor executor) noexcept
    {
        switch (executor)
        {
        case CanonicalRouteFoldExecutor::CallerThread:
            return "caller_thread_column_tiles";
        case CanonicalRouteFoldExecutor::ExistingOpenMPTeam:
            return "existing_team_column_tiles";
        case CanonicalRouteFoldExecutor::FullOpenMPTeam:
            return "full_team_column_tiles";
        }
        return "unknown";
    }
} // namespace llaminar2::cpu::moe
