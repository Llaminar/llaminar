/**
 * @file MTPSpecStatePublisher.cpp
 * @brief Implements MTP verifier-state publication for host-plan and
 *        device-resident speculative decode paths.
 *
 * The GPU resident path is intentionally shaped around device row-index
 * pointers.  Host step plans may still describe CPU/legacy publication, but the
 * promoted resident path must not recover accepted rows by copying compact
 * metadata to the CPU first. CPU single-request publication validates immutable
 * compact GDN descriptors and every source/live span before constructing typed
 * copy, zero-fill, and phase-word tasks for the existing persistent OpenMP team.
 * Canonical states are written directly; no full mutable bank is staged first.
 */

#include "MTPSpecStatePublisher.h"

#include "../compute_stages/IComputeStage.h"
#include "../local_execution/graph/ComputeGraph.h"
#include "../../utils/OpenMPUtils.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Validate one compact accepted prefix without mutating any bank.
         * @param plan Canonical publication plan containing a typed GDN prefix.
         * @return True for complete geometry, generation, capacity, and head metadata.
         * Malformed public plans fail before construction of any scatter task.
         */
        bool validSharedGDNPrefix(const CPUVerifierStateRestorePlan &plan)
        {
            if (!plan.shared_gdn_prefix) return true;
            const auto &accepted=*plan.shared_gdn_prefix;
            const auto &t=accepted.transaction;
            if (!t.slots || plan.source!=t.slots || t.requests!=1 || t.heads<=0 ||
                t.dk<=0 || t.dk>512 || t.dv<=0 || t.dv>512 || t.width<=0 ||
                t.width>std::numeric_limits<int>::max()-8 || t.live_width<=0 || t.live_width>t.width ||
                accepted.row<0 || accepted.row>=t.live_width ||
                !t.generation_authority ||
                t.slot_floats>std::numeric_limits<size_t>::max()/sizeof(float)) return false;
            // Authenticate complete address domains before reading either
            // protocol authority or head metadata. The later generic plan-span
            // validation cannot protect these earlier compact metadata reads.
            const uintptr_t source_begin=reinterpret_cast<uintptr_t>(t.slots);
            const uintptr_t generation_begin=reinterpret_cast<uintptr_t>(t.generation_authority);
            const size_t source_bytes=t.slot_floats*sizeof(float);
            if (source_bytes>std::numeric_limits<uintptr_t>::max()-source_begin ||
                sizeof(*t.generation_authority)>std::numeric_limits<uintptr_t>::max()-generation_begin)
                return false;
            if (*t.generation_authority!=t.generation) return false;
            const CPUGDNStateLayout one_head(1,t.dk,t.dv);
            if (t.heads>std::numeric_limits<int>::max()/one_head.headFloats()) return false;
            const CPUGDNStateLayout layout(t.heads,t.dk,t.dv);
            const size_t span=CPUSharedGDNSnapshots::headFloats(layout,t.width);
            if (span>t.slot_floats/static_cast<size_t>(t.heads) ||
                plan.bytes!=static_cast<size_t>(layout.floats())*sizeof(float)) return false;
            for(int head=0;head<t.heads;++head)
                if (!CPUSharedGDNSnapshots(layout,t.slots+static_cast<size_t>(head)*span,t.width).validRow(accepted.row)) return false;
            return true;
        }

        /**
         * @brief Resolve the entire immutable source span consumed by a publication.
         * @param plan Validated contiguous or compact-GDN plan.
         * @return Source byte extent, including retained compact headers/history.
         */
        size_t publicationSourceBytes(const CPUVerifierStateRestorePlan &plan)
        {
            return plan.shared_gdn_prefix
                ? plan.shared_gdn_prefix->transaction.slot_floats*sizeof(float)
                : plan.bytes;
        }

        MTPSpecStatePublicationResult publicationFailure(
            const MTPSpecStepPlan &plan,
            std::string reason)
        {
            MTPSpecStatePublicationResult result;
            result.ok = false;
            result.error = std::move(reason);
            result.request_id = plan.request_id;
            result.accepted_count = plan.accepted_count;
            return result;
        }

        MTPSpecStatePublicationResult batchPublicationFailure(
            const MTPSpecStepPlanBatch &plans,
            std::string reason)
        {
            MTPSpecStatePublicationResult result;
            result.ok = false;
            result.error = std::move(reason);
            for (const MTPSpecStepPlan &step : plans.steps)
                result.accepted_count += step.accepted_count;
            return result;
        }

        MTPSpecStatePublicationResult devicePublicationFailure(
            const MTPDeviceVerifierStatePublicationShape &shape,
            std::string reason)
        {
            MTPSpecStatePublicationResult result;
            result.ok = false;
            result.error = std::move(reason);
            result.request_id = shape.request_id;
            result.accepted_count = 0;
            return result;
        }

        MTPDeviceVerifierStatePublicationShape deviceShapeFromPlan(
            const MTPSpecStepPlan &plan)
        {
            MTPDeviceVerifierStatePublicationShape shape;
            shape.request_count = 1;
            shape.target_rows = plan.target_rows;
            shape.request_id = plan.request_id;
            return shape;
        }

        MTPDeviceVerifierStatePublicationShape deviceShapeFromPlanBatch(
            const MTPSpecStepPlanBatch &plans)
        {
            MTPDeviceVerifierStatePublicationShape shape;
            shape.request_count = plans.request_count;
            shape.target_rows = plans.shape.max_draft_tokens;
            for (const MTPSpecStepPlan &step : plans.steps)
                shape.target_rows = std::max(shape.target_rows, step.target_rows);
            return shape;
        }

        void preserveLegacyPlanAccounting(
            MTPSpecStatePublicationResult &result,
            const MTPSpecStepPlan &plan)
        {
            result.request_id = plan.request_id;
            result.accepted_count = plan.accepted_count;
        }

        void preserveLegacyBatchAccounting(
            MTPSpecStatePublicationResult &result,
            const MTPSpecStepPlanBatch &plans)
        {
            result.accepted_count = 0;
            for (const MTPSpecStepPlan &step : plans.steps)
                result.accepted_count += step.accepted_count;
        }

    } // namespace

    MTPSpecStatePublicationResult publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromVerifierRow(
            plan,
            plan.accepted_count - 1,
            state_stages,
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRow(
        const MTPSpecStepPlan &plan,
        int verifier_restore_row,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!device.is_valid())
            return publicationFailure(plan, "cannot publish MTP spec state on invalid device");
        if (device.is_gpu() && stream == nullptr)
        {
            return publicationFailure(
                plan,
                "GPU MTP spec-state publication requires an explicit non-null stream");
        }
        if (plan.draft_count < 0 || plan.target_rows != plan.draft_count + 1)
        {
            return publicationFailure(
                plan,
                "MTP spec-step plan has invalid draft/target row shape");
        }
        if (plan.accepted_count < 0 || plan.accepted_count > plan.draft_count)
        {
            return publicationFailure(
                plan,
                "MTP spec-step accepted count is outside the draft prefix");
        }
        if (plan.accepted_count > 0 && verifier_restore_row < 0)
        {
            return publicationFailure(
                plan,
                "MTP spec-state publication received a negative verifier restore row");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_restore_row_requests",
            1.0,
            "decode",
            device.toString(),
            {{"request_id", std::to_string(plan.request_id)},
             {"accepted_count", std::to_string(plan.accepted_count)},
             {"draft_count", std::to_string(plan.draft_count)},
             {"restore_row", std::to_string(verifier_restore_row)},
             {"target_cached_tokens", std::to_string(plan.target_cached_tokens)}});

        if(device.is_cpu())
        {
            // This CPU-owned scalar is one request-local decision vector. Keep
            // legacy caller accounting while sharing all plan-before-write logic.
            auto step=plan;step.request_index=0;
            MTPSpecStepPlanBatch batch;batch.ok=true;batch.request_count=1;batch.steps.push_back(step);
            auto published=publishAcceptedMTPSpecStateFromVerifierRows(batch,&verifier_restore_row,
                state_stages,device,stream,require_captured_stage);
            preserveLegacyPlanAccounting(published,plan);return published;
        }

        MTPSpecStatePublicationResult result;
        result.ok = true;
        result.request_id = plan.request_id;
        result.accepted_count = plan.accepted_count;


        if (plan.accepted_count == 0)
        {
            result.skipped_stage_count = static_cast<int>(state_stages.size());
            return result;
        }

        const int restore_row = verifier_restore_row;
        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (stage == nullptr)
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication received null stage at index "
                    << i;
                return publicationFailure(plan, msg.str());
            }
            if (!stage->hasVerifierStateCapture())
            {
                if (require_captured_stage &&
                    stage->requiresVerifierStateCaptureForPublication())
                {
                    std::ostringstream msg;
                    msg << "MTP spec-state publication required verifier capture for stage "
                        << stage->name() << " at index " << i
                        << " but no capture was bound";
                    return publicationFailure(plan, msg.str());
                }
                ++result.skipped_stage_count;
                continue;
            }
            if (!stage->restoreVerifierStateCaptureRow(restore_row, stream))
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication failed restoring verifier row "
                    << restore_row << " for stage " << stage->name()
                    << " at index " << i;
                return publicationFailure(plan, msg.str());
            }
            ++result.restored_stage_count;
            stage->clearVerifierStateCaptureBindingAfterPublication();
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return publicationFailure(
                plan,
                "MTP spec-state publication required a verifier-captured state stage but restored none");
        }

        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *host_verifier_restore_rows,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!device.is_valid())
            return batchPublicationFailure(plans, "cannot publish batched MTP spec state on invalid device");
        if (device.is_gpu())
        {
            return batchPublicationFailure(
                plans,
                "host-indexed batched MTP spec-state publication is CPU-only");
        }
        if(stream)
            return batchPublicationFailure(plans,"CPU accepted-state publication cannot use a GPU stream");
        if (!plans.ok || plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return batchPublicationFailure(
                plans,
                plans.error.empty()
                    ? "host-indexed batched MTP spec-state publication received an invalid plan batch"
                    : plans.error);
        }
        if (!host_verifier_restore_rows)
        {
            return batchPublicationFailure(
                plans,
                "host-indexed batched MTP spec-state publication received null restore rows");
        }

        bool any_restore = false;
        for (const MTPSpecStepPlan &step : plans.steps)
        {
            if (step.request_index < 0 || step.request_index >= plans.request_count ||
                step.accepted_count < 0 || step.accepted_count > step.draft_count)
            {
                return batchPublicationFailure(
                    plans,
                    "host-indexed batched MTP spec-state publication received an invalid request step");
            }
            const int row = host_verifier_restore_rows[step.request_index];
            if (step.accepted_count > 0 && row < 0)
            {
                return batchPublicationFailure(
                    plans,
                    "host-indexed batched MTP spec-state publication is missing an accepted restore row");
            }
            any_restore = any_restore || row >= 0;
        }

        MTPSpecStatePublicationResult result;
        result.ok = true;
        preserveLegacyBatchAccounting(result, plans);
        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_host_batch_restore_requests",
            1.0,
            "decode",
            device.toString(),
            {{"request_count", std::to_string(plans.request_count)},
             {"publication_policy", "request_owned_grouped_restore"}});

        // An all-negative decision is legacy snapshot NoOp, but a typed native
        // prefix stage may need to restore original immutable prior bytes. Plan
        // every stage before deciding whether any physical write is necessary.

        {
            /*
             * Every request and stage contributes immutable plans against its
             * sole accepted-state owner before any live byte can be changed.
             * Planning every copy first makes bad rows, stale workspaces, and
             * overlapping destinations fatal before any byte becomes live.
             * Once validated, one persistent OpenMP team fans the independent
             * layer copies across physical cores. This replaces dozens of
             * serial memory-bandwidth operations without changing one byte or
             * replaying any recurrence arithmetic.
             */
            std::vector<CPUVerifierStateRestorePlan> restore_plans;
            restore_plans.reserve(state_stages.size());
            std::vector<CPUVerifierStateRestoreDestination> live_destinations;
            std::vector<IComputeStage *> native_prefix_stages;
            std::vector<IComputeStage *> changed_stages;
            enum class RestoreOperation : uint8_t { Copy, Zero, PhaseWord };
            struct RestoreCopyChunk
            {
                void *destination = nullptr;
                const void *source = nullptr;
                size_t bytes = 0;
                RestoreOperation operation = RestoreOperation::Copy;
                float phase_word = 0.0f;
            };
            constexpr size_t kRestoreCopyChunkBytes = 128u * 1024u;
            std::vector<RestoreCopyChunk> restore_chunks;
            size_t restore_bytes = 0;
            const int restore_row = host_verifier_restore_rows[0];

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "spec_state_cpu_parallel_restore_plan",
                    "decode");
                for (size_t i = 0; i < state_stages.size(); ++i)
                {
                    IComputeStage *stage = state_stages[i];
                    if (!stage)
                    {
                        std::ostringstream msg;
                        msg << "CPU request-owned MTP publication received null stage at index "
                            << i;
                        return batchPublicationFailure(plans, msg.str());
                    }
                    if (!stage->hasVerifierStateCapture())
                    {
                        if (require_captured_stage &&
                            stage->requiresVerifierStateCaptureForPublication())
                        {
                            std::ostringstream msg;
                            msg << "CPU request-owned MTP publication required verifier capture for stage "
                                << stage->name() << " at index " << i;
                            return batchPublicationFailure(plans, msg.str());
                        }
                        continue;
                    }

                    const auto first_plan=restore_plans.size();
                    if(!stage->planCPUVerifierStateRestoreRows(host_verifier_restore_rows,plans.request_count,restore_plans))
                    {
                        std::ostringstream msg;
                        msg << "CPU request-owned MTP publication could not plan a native restore for captured stage "
                            << stage->name() << " at index " << i;
                        return batchPublicationFailure(plans,msg.str());
                    }
                    bool native_commit=false,stage_work=false;
                    for(std::size_t plan_index=first_plan;plan_index<restore_plans.size();++plan_index)
                    {
                    const auto &restore=restore_plans[plan_index];
                    if(restore.status==CPUVerifierStateRestorePlanStatus::NoOp)continue;
                    if(!restore.ready() || !validSharedGDNPrefix(restore))return batchPublicationFailure(plans,
                        "CPU request-owned MTP publication returned an invalid immutable restore plan");

                    const auto admit_destination = [&](CPUVerifierStateRestoreDestination span) {
                        const uintptr_t begin = reinterpret_cast<uintptr_t>(span.data);
                        if (!span.data || !span.bytes ||
                            span.bytes > std::numeric_limits<uintptr_t>::max() - begin ||
                            span.bytes > std::numeric_limits<size_t>::max() - restore_bytes)
                            return false;
                        for (const auto &planned : live_destinations)
                        {
                            const uintptr_t planned_begin = reinterpret_cast<uintptr_t>(planned.data);
                            if (begin < planned_begin + planned.bytes && planned_begin < begin + span.bytes)
                                return false;
                        }
                        live_destinations.push_back(span);
                        restore_bytes += span.bytes;
                        return true;
                    };
                    if (restore.kind == CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit)
                    {
                        for (const auto &span : restore.native_destinations)
                            if (!admit_destination(span))
                                return batchPublicationFailure(plans,
                                    "CPU request-owned MTP publication has invalid or overlapping native-prefix live destinations");
                        // A Ready native operation is still a publication when its
                        // exact payload is empty; its own validated live predicate owns that case.
                        native_commit=stage_work=true;
                    }
                    else
                    {
                        const uintptr_t destination_begin = reinterpret_cast<uintptr_t>(restore.destination);
                        const uintptr_t source_begin = reinterpret_cast<uintptr_t>(restore.source);
                        const size_t source_bytes=publicationSourceBytes(restore);
                        if (source_bytes > std::numeric_limits<uintptr_t>::max() - source_begin ||
                            restore.bytes > std::numeric_limits<uintptr_t>::max() - destination_begin ||
                            (destination_begin < source_begin + source_bytes && source_begin < destination_begin + restore.bytes) ||
                            !admit_destination({restore.destination, restore.bytes}))
                            return batchPublicationFailure(plans,
                                "CPU request-owned MTP publication has invalid or overlapping snapshot live destinations");
                        stage_work=true;
                    }

                    }
                    if(native_commit)native_prefix_stages.push_back(stage);
                    if(stage_work)changed_stages.push_back(stage);
                }

                // All inputs remain immutable until the entire transaction is
                // copied. A live destination may alias neither another live bank
                // nor another plan's full input span, including compact history.
                for (const auto &destination : live_destinations)
                    for (const auto &source_plan : restore_plans)
                        if (source_plan.status == CPUVerifierStateRestorePlanStatus::Ready &&
                            CPUSharedGDNSnapshots::buffersOverlap(
                                destination.data,destination.bytes,
                                source_plan.source,publicationSourceBytes(source_plan)))
                            return batchPublicationFailure(plans,
                                "CPU request-owned MTP publication live destination aliases a retained input span");

                // Every task emits a nonempty disjoint piece of the already
                // checked canonical byte total, so this exact task count cannot
                // exceed restore_bytes or overflow its validated size domain.
                const auto chunks_for=[](size_t bytes) {
                    return bytes/kRestoreCopyChunkBytes+(bytes%kRestoreCopyChunkBytes!=0);
                };
                size_t restore_chunk_count=0;
                for(const auto &restore : restore_plans)
                {
                    if (restore.status != CPUVerifierStateRestorePlanStatus::Ready ||
                        restore.kind == CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit)
                        continue;
                    if (!restore.shared_gdn_prefix) {
                        restore_chunk_count+=chunks_for(restore.bytes);
                        continue;
                    }
                    const auto &accepted=*restore.shared_gdn_prefix;
                    const auto &t=accepted.transaction;
                    const CPUGDNStateLayout layout(t.heads,t.dk,t.dv);
                    const size_t span=CPUSharedGDNSnapshots::headFloats(layout,t.width);
                    for(int head=0;head<t.heads;++head) {
                        const auto candidate=CPUSharedGDNSnapshots(layout,
                            t.slots+static_cast<size_t>(head)*span,t.width).candidate(accepted.row);
                        restore_chunk_count+=1+chunks_for(static_cast<size_t>(layout.matrixFloats())*sizeof(float))
                            +chunks_for(static_cast<size_t>(candidate.phase)*layout.recordFloats()*sizeof(float))
                            +chunks_for(static_cast<size_t>(8-candidate.phase)*layout.recordFloats()*sizeof(float));
                    }
                }
                restore_chunks.reserve(restore_chunk_count);
                const auto append_chunks = [&](void *destination,const void *source,
                    size_t bytes,RestoreOperation operation)
                {
                    auto *target=static_cast<std::byte *>(destination);
                    const auto *input=static_cast<const std::byte *>(source);
                    for(size_t offset=0;offset<bytes;offset+=kRestoreCopyChunkBytes)
                        restore_chunks.push_back({.destination=target+offset,
                            .source=operation==RestoreOperation::Copy ? input+offset : nullptr,
                            .bytes=std::min(kRestoreCopyChunkBytes,bytes-offset),
                            .operation=operation});
                };
                for (const auto &restore : restore_plans)
                {
                    if (restore.status != CPUVerifierStateRestorePlanStatus::Ready ||
                        restore.kind == CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit)
                        continue;
                    if (!restore.shared_gdn_prefix)
                    {
                        append_chunks(restore.destination,restore.source,restore.bytes,RestoreOperation::Copy);
                        continue;
                    }
                    const auto &accepted=*restore.shared_gdn_prefix;
                    const auto &t=accepted.transaction;
                    const CPUGDNStateLayout layout(t.heads,t.dk,t.dv);
                    const size_t span=CPUSharedGDNSnapshots::headFloats(layout,t.width);
                    for(int head=0;head<t.heads;++head)
                    {
                        CPUSharedGDNSnapshots snapshot(layout,t.slots+static_cast<size_t>(head)*span,t.width);
                        const auto candidate=snapshot.candidate(accepted.row);
                        auto *target=static_cast<float *>(restore.destination)+static_cast<size_t>(head)*layout.headFloats();
                        restore_chunks.push_back({.destination=target,.source=nullptr,
                            .bytes=sizeof(float),.operation=RestoreOperation::PhaseWord,
                            .phase_word=static_cast<float>(candidate.phase)});
                        append_chunks(target+1,snapshot.checkpoint(candidate.checkpoint),
                            static_cast<size_t>(layout.matrixFloats())*sizeof(float),RestoreOperation::Copy);
                        append_chunks(layout.record(target,0),snapshot.record(candidate.prefix-candidate.phase),
                            static_cast<size_t>(candidate.phase)*layout.recordFloats()*sizeof(float),RestoreOperation::Copy);
                        append_chunks(layout.record(target,candidate.phase),nullptr,
                            static_cast<size_t>(CPUGDNStateLayout::kPeriod-candidate.phase)*layout.recordFloats()*sizeof(float),RestoreOperation::Zero);
                    }
                }
            }

            if(require_captured_stage && any_restore && changed_stages.empty())
                return batchPublicationFailure(plans,
                    "CPU request-owned MTP publication required a captured stage but planned no state write");

            if(!restore_chunks.empty())
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "spec_state_cpu_parallel_restore_copy",
                    "decode");
                auto copy_restore_plans = [&]()
                {
#pragma omp for schedule(static)
                for (std::ptrdiff_t i = 0;
                     i < static_cast<std::ptrdiff_t>(restore_chunks.size());
                     ++i)
                {
                    const RestoreCopyChunk &restore =
                        restore_chunks[static_cast<size_t>(i)];
                        switch(restore.operation)
                        {
                            case RestoreOperation::Copy:
                                std::memcpy(restore.destination,restore.source,restore.bytes);
                                break;
                            case RestoreOperation::Zero:
                                std::memset(restore.destination,0,restore.bytes);
                                break;
                            case RestoreOperation::PhaseWord:
                                std::memcpy(restore.destination,&restore.phase_word,sizeof(float));
                                break;
                        }
                    }
                };
                OMP_WORKSHARE_REGION(copy_restore_plans);
            }

            /* Native prefix kernels own their worksharing policy. Calling them outside
             * the copy team avoids entering OMP_SINGLE/collective worksharing from one
             * worker, while retaining parallel channel copies for the PLE history.
             * All candidate bounds and all live destinations were admitted above.
             */
            for (IComputeStage *stage : native_prefix_stages)
                if (!(plans.request_count==1 ? stage->restoreVerifierStateCaptureRow(restore_row,stream) :
                    stage->restoreVerifierStateCaptureRows(host_verifier_restore_rows,plans.request_count,stream)))
                    return batchPublicationFailure(plans,
                        "CPU request-owned MTP validated native-prefix commit failed");
            for (const auto &restore : restore_plans)
                if (restore.shared_gdn_prefix)
                {
                    PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materializations",1,"publication","cpu");
                    PerfStatsCollector::addCounter("kernel","cpu_gdn_shared_accepted_materialization_bytes",
                        static_cast<double>(restore.bytes),"publication","cpu");
                    PerfStatsCollector::addCounter("mtp","spec_state_cpu_compact_scatter_plans",1,"decode","cpu");
                }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "spec_state_cpu_parallel_restore_finalize",
                    "decode");
                for (size_t i = 0; i < state_stages.size(); ++i)
                {
                    IComputeStage *stage = state_stages[i];
                    if (std::find(changed_stages.begin(),changed_stages.end(),stage)!=changed_stages.end())
                    {
                        ++result.restored_stage_count;
                        stage->clearVerifierStateCaptureBindingAfterPublication();
                        continue;
                    }
                    ++result.skipped_stage_count;
                }
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "spec_state_cpu_parallel_restore_bytes",
                static_cast<double>(restore_bytes),
                "decode",
                device.toString(),
                {{"restore_spans", std::to_string(restore_plans.size())},
                 {"restore_chunks", std::to_string(restore_chunks.size())},
                 {"copy_chunk_bytes", std::to_string(kRestoreCopyChunkBytes)},
                 {"native_prefix_commits", std::to_string(native_prefix_stages.size())},
                 {"publication_policy", "validated_snapshot_and_native_prefix_publication"}});
            return result;
        }

    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPSpecStepPlan &plan,
        std::span<const int> device_verifier_restore_row,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!device.is_valid())
            return publicationFailure(plan, "cannot publish MTP spec state on invalid device");
        if (!device.is_gpu())
        {
            return publicationFailure(
                plan,
                "device-indexed MTP spec-state publication currently requires a GPU device");
        }
        if (stream == nullptr)
        {
            return publicationFailure(
                plan,
                "GPU device-indexed MTP spec-state publication requires an explicit non-null stream");
        }
        if (!deviceVerifierStateMetadataCovers(device_verifier_restore_row, 1, 1))
        {
            return publicationFailure(
                plan,
                "device-indexed MTP spec-state publication received a null row pointer");
        }
        if (plan.draft_count < 0 || plan.target_rows != plan.draft_count + 1)
        {
            return publicationFailure(
                plan,
                "MTP spec-step plan has invalid draft/target row shape");
        }

        MTPSpecStatePublicationResult result =
            publishAcceptedMTPSpecStateFromDeviceVerifierRow(
                deviceShapeFromPlan(plan),
                device_verifier_restore_row,
                state_stages,
                device,
                stream,
                require_captured_stage);
        preserveLegacyPlanAccounting(result, plan);
        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPDeviceVerifierStatePublicationShape &shape,
        std::span<const int> device_verifier_restore_row,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!device.is_valid())
            return devicePublicationFailure(shape, "cannot publish MTP spec state on invalid device");
        if (!device.is_gpu())
        {
            return devicePublicationFailure(
                shape,
                "device-indexed MTP spec-state publication currently requires a GPU device");
        }
        if (stream == nullptr)
        {
            return devicePublicationFailure(
                shape,
                "GPU device-indexed MTP spec-state publication requires an explicit non-null stream");
        }
        if (!deviceVerifierStateMetadataCovers(device_verifier_restore_row, 1, 1))
        {
            return devicePublicationFailure(
                shape,
                "device-indexed MTP spec-state publication received a null row pointer");
        }
        if (!shape.validScalar())
        {
            return devicePublicationFailure(
                shape,
                "device-indexed MTP spec-state publication received an invalid scalar request shape");
        }

        MTPSpecStatePublicationResult result;
        result.ok = true;
        result.request_id = shape.request_id;
        result.accepted_count = 0;

        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (stage == nullptr)
            {
                std::ostringstream msg;
                msg << "device-indexed MTP spec-state publication received null stage at index "
                    << i;
                return devicePublicationFailure(shape, msg.str());
            }
            if (!stage->hasVerifierStateCapture())
            {
                if (require_captured_stage &&
                    stage->requiresVerifierStateCaptureForPublication())
                {
                    std::ostringstream msg;
                    msg << "device-indexed MTP spec-state publication required verifier capture for stage "
                        << stage->name() << " at index " << i
                        << " but no capture was bound";
                    return devicePublicationFailure(shape, msg.str());
                }
                ++result.skipped_stage_count;
                continue;
            }
            if (!stage->restoreVerifierStateCaptureRowFromDeviceIndex(
                    device_verifier_restore_row,
                    stream))
            {
                std::ostringstream msg;
                msg << "device-indexed MTP spec-state publication failed restoring verifier row for stage "
                    << stage->name()
                    << " at index " << i;
                return devicePublicationFailure(shape, msg.str());
            }
            ++result.restored_stage_count;
            stage->clearVerifierStateCaptureBindingAfterPublication();
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return devicePublicationFailure(
                shape,
                "device-indexed MTP spec-state publication required a verifier-captured state stage but restored none");
        }

        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        std::span<const int> device_verifier_restore_rows,
        int row_index_stride,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!plans.ok || plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return batchPublicationFailure(
                plans,
                plans.error.empty()
                    ? "batched MTP spec-state publication received an invalid step plan batch"
                    : plans.error);
        }
        for (const MTPSpecStepPlan &step : plans.steps)
        {
            if (step.accepted_count < 0 || step.accepted_count > step.draft_count)
            {
                return batchPublicationFailure(
                    plans,
                    "batched MTP spec-state publication step accepted count is outside the draft prefix");
            }
        }
        MTPSpecStatePublicationResult result =
            publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            deviceShapeFromPlanBatch(plans),
            device_verifier_restore_rows,
            row_index_stride,
            state_stages,
            device,
            stream,
            require_captured_stage);
        preserveLegacyBatchAccounting(result, plans);
        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPDeviceVerifierStatePublicationShape &shape,
        std::span<const int> device_verifier_restore_rows,
        int row_index_stride,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        if (!device.is_valid())
            return devicePublicationFailure(shape, "cannot publish batched MTP spec state on invalid device");
        if (!device.is_gpu())
        {
            return devicePublicationFailure(
                shape,
                "batched device-indexed MTP spec-state publication currently requires a GPU device");
        }
        if (stream == nullptr)
        {
            return devicePublicationFailure(
                shape,
                "batched GPU device-indexed MTP spec-state publication requires an explicit non-null stream");
        }
        if (!deviceVerifierStateMetadataCovers(device_verifier_restore_rows, 1, 1))
        {
            return devicePublicationFailure(
                shape,
                "batched device-indexed MTP spec-state publication received a null row pointer");
        }
        if (row_index_stride <= 0)
        {
            return devicePublicationFailure(
                shape,
                "batched device-indexed MTP spec-state publication received an invalid row-index stride");
        }
        if (!shape.validBatch())
        {
            return devicePublicationFailure(
                shape,
                "batched device-indexed MTP spec-state publication received an invalid request shape");
        }

        if (!deviceVerifierStateMetadataCovers(device_verifier_restore_rows,
                shape.request_count, row_index_stride))
            return devicePublicationFailure(shape,
                "device-indexed MTP publication metadata owner span is too short");

        MTPSpecStatePublicationResult result;
        result.ok = true;
        result.accepted_count = 0;

        /*
         * Device-indexed batch publication must not infer "nothing to do" from
         * host-side accepted-count fields.  The authoritative row choices live
         * in device_verifier_restore_rows, and rejected requests are encoded as
         * negative row indices that request-aware stages leave unchanged.
         */
        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (stage == nullptr)
            {
                std::ostringstream msg;
                msg << "batched device-indexed MTP spec-state publication received null stage at index "
                    << i;
                return devicePublicationFailure(shape, msg.str());
            }
            if (!stage->hasVerifierStateCapture())
            {
                if (require_captured_stage &&
                    stage->requiresVerifierStateCaptureForPublication())
                {
                    std::ostringstream msg;
                    msg << "batched device-indexed MTP spec-state publication required verifier capture for stage "
                        << stage->name() << " at index " << i
                        << " but no capture was bound";
                    return devicePublicationFailure(shape, msg.str());
                }
                ++result.skipped_stage_count;
                continue;
            }
            if (!stage->restoreVerifierStateCaptureRowsFromDeviceIndices(
                    device_verifier_restore_rows,
                    shape.request_count,
                    row_index_stride,
                    stream))
            {
                std::ostringstream msg;
                msg << "batched device-indexed MTP spec-state publication failed restoring verifier rows for stage "
                    << stage->name()
                    << " at index " << i;
                return devicePublicationFailure(shape, msg.str());
            }
            ++result.restored_stage_count;
            stage->clearVerifierStateCaptureBindingAfterPublication();
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return devicePublicationFailure(
                shape,
                "batched device-indexed MTP spec-state publication required a verifier-captured state stage but restored none");
        }

        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromVerifierRow(
            plan,
            plan.accepted_count - 1,
            graph,
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRow(
        const MTPSpecStepPlan &plan,
        int verifier_restore_row,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromVerifierRow(
            plan,
            verifier_restore_row,
            graph.getExecutionStages(),
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *host_verifier_restore_rows,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromVerifierRows(
            plans,
            host_verifier_restore_rows,
            graph.getExecutionStages(),
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPSpecStepPlan &plan,
        std::span<const int> device_verifier_restore_row,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            plan,
            device_verifier_restore_row,
            graph.getExecutionStages(),
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPDeviceVerifierStatePublicationShape &shape,
        std::span<const int> device_verifier_restore_row,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            shape,
            device_verifier_restore_row,
            graph.getExecutionStages(),
            device,
            stream,
            require_captured_stage);
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        std::span<const int> device_verifier_restore_rows,
        int row_index_stride,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        MTPSpecStatePublicationResult result =
            publishAcceptedMTPSpecStateFromDeviceVerifierRows(
                plans,
                device_verifier_restore_rows,
                row_index_stride,
                graph.getExecutionStages(),
                device,
                stream,
                require_captured_stage);
        preserveLegacyBatchAccounting(result, plans);
        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPDeviceVerifierStatePublicationShape &shape,
        std::span<const int> device_verifier_restore_rows,
        int row_index_stride,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage)
    {
        return publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            shape,
            device_verifier_restore_rows,
            row_index_stride,
            graph.getExecutionStages(),
            device,
            stream,
            require_captured_stage);
    }

} // namespace llaminar2
