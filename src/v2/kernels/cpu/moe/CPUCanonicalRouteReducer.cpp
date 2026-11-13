/**
 * @file CPUCanonicalRouteReducer.cpp
 * @brief Stable-team implementation of ordered canonical CPU MoE reduction.
 *
 * Joint expert fragments retain independent prepared-engine execution. Their
 * unweighted down rows are combined in immutable cohort order before the
 * existing canonical top-k FMA fold. Callers own every input/scratch/output
 * allocation through the ordinary PMA; this kernel neither prices nor places
 * experts and never allocates or changes worker-team width.
 * A borrowed weighted sink is filled by the same worker immediately after the
 * original addition routine finishes a column tile. This retains that routine's
 * exact arithmetic while avoiding a second traversal of the complete raw bank.
 */

#include "CPUCanonicalRouteReducer.h"

#include "../../../execution/moe/CanonicalMoERouteRecord.h"
#include "../primitives/VectorPrimitives.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <cmath>
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

    /**
     * @brief One arithmetic implementation over immutable indexed or descriptor bank views.
     * @tparam BankAt Allocation-free original-participant lookup, inlinable into every tile.
     * @param args Actual original routes, output roots and optional fused weighting sink.
     * @param population Original cohort cardinality, independent of physical local hardware availability.
     * @param bankAt Borrow one original immutable bank by its existing participant index.
     * @return Whether the complete validation and original fixed-order combine succeeded.
     */
    template<class BankAt>
    static bool combineCanonicalRouteDownPartialsOrderedImpl(
        const CanonicalRouteDownPartialCombineArgs &args, size_t population, BankAt bankAt)
    {
        if (population < 2 || population > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            args.rows < 0 || args.top_k <= 0 || args.columns <= 0) return false;
        if (args.rows == 0) return true;
        const auto maximum = std::numeric_limits<size_t>::max();
        const auto rows = static_cast<size_t>(args.rows);
        const auto top_k = static_cast<size_t>(args.top_k);
        const auto columns = static_cast<size_t>(args.columns);
        if (rows > maximum / top_k || rows * top_k > maximum / columns) return false;
        const auto slots = rows * top_k;
        const auto elements = slots * columns;
        if (elements > maximum / sizeof(float) || !args.route_weights ||
            args.combined_route_rows.size() < elements || !args.combined_route_rows.data()) return false;
        const auto plan = planCanonicalRouteFold(args.rows,args.top_k,args.columns);
        if (!plan.parallel_tasks || plan.parallel_tasks > static_cast<size_t>(std::numeric_limits<long long>::max()))
            return false;
        // Validate the complete byte geometry before publishing even one scratch element.
        const auto overlap = [](const void *left,size_t leftBytes,const void *right,size_t rightBytes) {
            const auto a=reinterpret_cast<std::uintptr_t>(left),b=reinterpret_cast<std::uintptr_t>(right);
            if (leftBytes>std::numeric_limits<std::uintptr_t>::max()-a ||
                rightBytes>std::numeric_limits<std::uintptr_t>::max()-b) return true;
            return a<b+rightBytes && b<a+leftBytes;
        };
        const auto bytes=elements*sizeof(float),weightBytes=slots*sizeof(float);
        if (overlap(args.combined_route_rows.data(),bytes,args.route_weights,weightBytes)) return false;
        for (size_t participant=0;participant<population;++participant) {
            const auto &bank=bankAt(participant);
            const auto input=bank.values;
            const auto inputBytes=bank.record_for_route_slot.empty()?bytes:input.size_bytes();
            if ((!input.empty() && !input.data()) || input.size()>maximum/sizeof(float) ||
                overlap(input.data(),inputBytes,args.combined_route_rows.data(),bytes)) return false;
            if (bank.record_for_route_slot.empty()) {
                if (input.size()<elements) return false;
            } else {
                if (bank.record_for_route_slot.size()!=slots) return false;
                const auto width=columns+1;
                if (input.size()%width) return false;
                for (size_t slot=0;slot<slots;++slot) {
                    const auto record=bank.record_for_route_slot[slot];
                    if (args.route_weights[slot]==0.0f) {
                        if (record!=-1) return false;
                        continue;
                    }
                    if (record<0 || static_cast<size_t>(record)>=input.size()/width) return false;
                    if (canonical_moe_route_record::readFlatRouteSlot(
                            input.data()+static_cast<size_t>(record)*width,args.columns)!=slot) return false;
                }
            }
            for (size_t prior=0;prior<participant;++prior)
                if (overlap(input.data(),inputBytes,bankAt(prior).values.data(),
                            bankAt(prior).record_for_route_slot.empty()?bytes:
                                bankAt(prior).values.size_bytes())) return false;
            if (!bank.record_for_route_slot.empty() &&
                (overlap(bank.record_for_route_slot.data(),bank.record_for_route_slot.size_bytes(),
                         args.combined_route_rows.data(),bytes) ||
                 overlap(bank.record_for_route_slot.data(),bank.record_for_route_slot.size_bytes(),
                         args.route_weights,weightBytes))) return false;
        }
        // Validate the complete second output before either output is touched.
        // The compact index is derived from immutable original slots, never a
        // second route/ownership ledger or an inferred active capacity.
        if (args.preweighted_output) {
            const auto &w=*args.preweighted_output;
            const auto count=w.original_slots.size();
            if (slots>static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
                count!=w.weights.size() || count!=w.destination_slots.size() || count>slots || count>maximum/columns ||
                w.values.size()>maximum/sizeof(float) || w.values.size()%columns ||
                w.values.size()<count*columns || (count && (!w.values.data() ||
                !w.original_slots.data() || !w.destination_slots.data() || !w.weights.data()))) return false;
            const auto sinkBytes=w.values.size_bytes();
            if (overlap(w.values.data(),sinkBytes,args.combined_route_rows.data(),bytes) ||
                overlap(w.values.data(),sinkBytes,args.route_weights,weightBytes) ||
                overlap(w.values.data(),sinkBytes,w.original_slots.data(),w.original_slots.size_bytes()) ||
                overlap(w.values.data(),sinkBytes,w.weights.data(),w.weights.size_bytes()) ||
                overlap(w.values.data(),sinkBytes,w.destination_slots.data(),w.destination_slots.size_bytes()) ||
                overlap(args.combined_route_rows.data(),bytes,w.original_slots.data(),w.original_slots.size_bytes()) ||
                overlap(args.combined_route_rows.data(),bytes,w.destination_slots.data(),w.destination_slots.size_bytes()) ||
                overlap(args.combined_route_rows.data(),bytes,w.weights.data(),w.weights.size_bytes())) return false;
            for (size_t participant=0;participant<population;++participant) {
                const auto bank=bankAt(participant);
                if (overlap(w.values.data(),sinkBytes,bank.values.data(),
                    bank.record_for_route_slot.empty()?bytes:bank.values.size_bytes()) ||
                    overlap(w.values.data(),sinkBytes,bank.record_for_route_slot.data(),
                            bank.record_for_route_slot.size_bytes())) return false;
            }
            int32_t previous=-1,previous_destination=-1;
            for(size_t entry=0;entry<count;++entry) {
                const auto slot=w.original_slots[entry],destination=w.destination_slots[entry];
                if(slot<=previous || static_cast<size_t>(slot)>=slots ||
                   destination<=previous_destination || static_cast<size_t>(destination)>=w.values.size()/columns ||
                   args.route_weights[slot]==0.0f || !std::isfinite(w.weights[entry]) ||
                   w.weights[entry]==0.0f) return false;
                previous=slot;previous_destination=destination;
            }
            size_t live=0;
            for(size_t slot=0;slot<slots;++slot)live+=args.route_weights[slot]!=0.0f;
            if(live!=count)return false;
        }
        const auto partialRow=[&](size_t participant,size_t slot,size_t first) {
            const auto &bank=bankAt(participant);
            const auto offset=bank.record_for_route_slot.empty()?slot*columns:
                static_cast<size_t>(bank.record_for_route_slot[slot])*(columns+1);
            return bank.values.data()+offset+first;
        };
        // The optional sink already names every required original route once
        // in order. Schedule that live list directly instead of rediscovering
        // each route by binary search in every column tile. Retain the original
        // column partition so the called addition's arithmetic boundaries do
        // not change. Unweighted callers retain their existing row/task grid.
        const size_t task_count=args.preweighted_output?
            args.preweighted_output->original_slots.size()*static_cast<size_t>(plan.column_tiles_per_row):
            plan.parallel_tasks;
        const auto combine = [&](size_t task) {
            const auto row=task/static_cast<size_t>(plan.column_tiles_per_row);
            const auto first=static_cast<int>(task%static_cast<size_t>(plan.column_tiles_per_row))*plan.column_tile;
            const auto count=std::min(plan.column_tile,args.columns-first);
            const auto first_slot=args.preweighted_output?
                static_cast<size_t>(args.preweighted_output->original_slots[row]):row*top_k;
            const auto route_count=args.preweighted_output?size_t{1}:top_k;
            for (size_t route=0;route<route_count;++route) {
                const auto slot=first_slot+route;
                if (args.route_weights[slot]==0.0f) continue;
                const auto offset=slot*columns+static_cast<size_t>(first);
                float *const destination=args.combined_route_rows.data()+offset;
                // The original degree is at least two. Add the first two
                // immutable rows directly: copying participant 0 into scratch
                // first would write and reread the same live columns before
                // this identical vec_add. Both operand order and its FP32
                // rounding point are unchanged, including scalar tails.
                const auto last=population-1;
                if (!args.preweighted_output) {
                    primitives::vec_add(destination,
                        partialRow(0,slot,static_cast<size_t>(first)),
                        partialRow(1,slot,static_cast<size_t>(first)),count);
                    for (size_t participant=2;participant<=last;++participant)
                        primitives::vec_add(destination,destination,
                            partialRow(participant,slot,static_cast<size_t>(first)),count);
                } else {
                    const auto &w=*args.preweighted_output;
                    const auto entry=row;
                    // Keep the actual original add routine, including its
                    // compiled NaN operand selection and scalar-tail behavior.
                    // Consume its completed column tile immediately on this
                    // worker, before another task displaces the useful data.
                    // This fuses ownership/worksharing rather than arithmetic:
                    // there is no second full-bank pass or parallel region.
                    if(last>1) {
                        primitives::vec_add(destination,partialRow(0,slot,static_cast<size_t>(first)),
                            partialRow(1,slot,static_cast<size_t>(first)),count);
                        for(size_t participant=2;participant<last;++participant)
                            primitives::vec_add(destination,destination,
                                partialRow(participant,slot,static_cast<size_t>(first)),count);
                    }
                    primitives::vec_add(destination,
                        last==1?partialRow(0,slot,static_cast<size_t>(first)):destination,
                        partialRow(last,slot,static_cast<size_t>(first)),count);
                    float *const weighted=w.values.data()+
                        static_cast<size_t>(w.destination_slots[entry])*columns+static_cast<size_t>(first);
                    const float factor=w.weights[entry];
                    for(int column=0;column<count;++column)
                        weighted[column]=destination[column]*factor;
                }
            }
        };
        if (plan.executor==CanonicalRouteFoldExecutor::CallerThread) {
            for (size_t task=0;task<task_count;++task) combine(task);
        } else {
            auto work=[&]() {
#pragma omp for schedule(static)
                for (long long task=0;task<static_cast<long long>(task_count);++task)
                    combine(static_cast<size_t>(task));
            };
            OMP_WORKSHARE_REGION(work);
        }
        return true;
    }

    bool combineCanonicalRouteDownPartialsOrdered(
        const CanonicalRouteDownPartialCombineArgs &args)
    {
        return combineCanonicalRouteDownPartialsOrderedImpl(args,args.ordered_participants.size(),
            [&](size_t participant){return args.ordered_participants[participant];});
    }

    bool combineCanonicalRouteDownPartialsOrderedIndexed(
        const CanonicalRouteDownPartialCombineArgs &args,
        const CanonicalRouteDownIndexedPartialRows &indexed)
    {
        const auto maximum=std::numeric_limits<size_t>::max();
        if(!args.ordered_participants.empty() || args.rows<0 || args.top_k<=0 || args.columns<=0 ||
            indexed.source_ranges.size()<2 ||
            indexed.source_ranges.size()>static_cast<size_t>(std::numeric_limits<int>::max()) ||
            indexed.source_ranges.size()>maximum/sizeof(RootedFloatRecordRange) ||
            (!indexed.source_ranges.empty() && !indexed.source_ranges.data()) ||
            (!indexed.records.empty() && !indexed.records.data()) ||
            (!indexed.participant_route_maps.empty() && !indexed.participant_route_maps.data()) ||
            indexed.records.size()>maximum/sizeof(float)) return false;
        const auto rows=static_cast<size_t>(args.rows),k=static_cast<size_t>(args.top_k);
        const auto columns=static_cast<size_t>(args.columns),population=indexed.source_ranges.size();
        if(rows>maximum/k || rows*k>maximum/population || rows*k>maximum/columns ||
            rows*k*columns>maximum/sizeof(float)) return false;
        const auto slots=rows*k,width=columns+1;
        // The indexed source table remains live throughout numerical writes;
        // unlike copied descriptors, it must be disjoint from BOTH outputs.
        // Authenticate this metadata before any range traversal or arithmetic.
        const auto metadataOverlaps=[&](const float* output,size_t bytes) {
            if(!bytes) return false;
            const auto source=reinterpret_cast<uintptr_t>(indexed.source_ranges.data());
            const auto target=reinterpret_cast<uintptr_t>(output);
            const auto source_bytes=indexed.source_ranges.size_bytes();
            return !output || source_bytes>std::numeric_limits<uintptr_t>::max()-source ||
                bytes>std::numeric_limits<uintptr_t>::max()-target ||
                (source<target+bytes && target<source+source_bytes);
        };
        if(metadataOverlaps(args.combined_route_rows.data(),slots*columns*sizeof(float))) return false;
        if(args.preweighted_output) {
            const auto& sink=args.preweighted_output->values;
            if(sink.size()>maximum/sizeof(float) || metadataOverlaps(sink.data(),sink.size_bytes())) return false;
        }
        if(indexed.records.size()%width ||
            !validRootedFloatRecordRanges(indexed.source_ranges,indexed.records.size()/width) ||
            indexed.participant_route_maps.size()<slots*population ||
            indexed.participant_route_maps.size()>maximum/sizeof(int32_t)) return false;
        // All range arithmetic is validated before reconstructing the first
        // borrowed span. The common reducer retains its complete output/alias
        // checks and the exact original participant addition order.
        return combineCanonicalRouteDownPartialsOrderedImpl(args,population,[&](size_t participant){
            const auto range=indexed.source_ranges[participant];
            return CanonicalRouteDownPartialRows{
                indexed.records.subspan(range.first_record*width,range.record_count*width),
                indexed.participant_route_maps.subspan(participant*slots,slots)};
        });
    }

    bool reduceCanonicalRouteDownPartialsOrderedFMA(
        const CanonicalRouteDownPartialFoldArgs &args, CanonicalRouteFoldPlan *used_plan)
    {
        // Authenticate the complete destination before the unweighted combine writes scratch.
        if (args.rows<0 || args.top_k<=0 || args.columns<=0) return false;
        if (args.rows==0) return args.ordered_participants.size()>=2 &&
            args.ordered_participants.size()<=static_cast<size_t>(std::numeric_limits<int>::max());
        const size_t rows=static_cast<size_t>(args.rows),k=static_cast<size_t>(args.top_k),h=static_cast<size_t>(args.columns);
        const auto max=std::numeric_limits<size_t>::max();
        if (!args.output || rows>max/k || rows*k>max/h || rows*k*h>max/sizeof(float)) return false;
        const auto outputBytes=rows*h*sizeof(float);
        const auto overlap=[&](const void *pointer,size_t bytes) {
            const auto a=reinterpret_cast<std::uintptr_t>(pointer),b=reinterpret_cast<std::uintptr_t>(args.output);
            return bytes>std::numeric_limits<std::uintptr_t>::max()-a ||
                   outputBytes>std::numeric_limits<std::uintptr_t>::max()-b || (a<b+outputBytes && b<a+bytes);
        };
        if (overlap(args.combined_route_rows.data(),rows*k*h*sizeof(float)) ||
            overlap(args.route_weights,rows*k*sizeof(float))) return false;
        for (const auto &bank:args.ordered_participants)
            if (bank.values.size()>max/sizeof(float) || overlap(bank.values.data(),
                bank.record_for_route_slot.empty()?rows*k*h*sizeof(float):bank.values.size_bytes()) ||
                (!bank.record_for_route_slot.empty() &&
                 overlap(bank.record_for_route_slot.data(),bank.record_for_route_slot.size_bytes()))) return false;
        if (!combineCanonicalRouteDownPartialsOrdered({args.ordered_participants,args.combined_route_rows,
                                                       args.route_weights,args.rows,args.top_k,args.columns})) return false;
        return reduceCanonicalRouteRowsOrderedFMA({args.combined_route_rows.data(),args.route_weights,
            nullptr,args.output,args.rows,args.top_k,args.columns},used_plan);
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
