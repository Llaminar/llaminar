/**
 * @file MTPSpecStatePublisher.cpp
 * @brief Implements MTP verifier-state publication for host-plan and
 *        device-resident speculative decode paths.
 *
 * The GPU resident path is intentionally shaped around device row-index
 * pointers.  Host step plans may still describe CPU/legacy publication, but the
 * promoted resident path must not recover accepted rows by copying compact
 * metadata to the CPU first.
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

        std::string publishPostRestoreStage(
            IComputeStage *stage,
            size_t stage_index,
            DeviceId device,
            void *stream,
            const char *publication_kind,
            MTPSpecStatePublicationResult &result)
        {
            if (!stage->requiresPostVerifierStatePublication())
                return {};

            if (device.is_gpu() && stream == nullptr)
            {
                std::ostringstream msg;
                msg << publication_kind
                    << " post-restore publication for GPU stage "
                    << stage->name()
                    << " at index " << stage_index
                    << " requires an explicit non-null stream";
                return msg.str();
            }

            if (!stage->publishPostVerifierStateRestore(stream))
            {
                std::ostringstream msg;
                msg << publication_kind
                    << " failed post-restore publication for stage "
                    << stage->name()
                    << " at index " << stage_index;
                return msg.str();
            }

            ++result.post_restore_stage_count;
            return {};
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

        MTPSpecStatePublicationResult result;
        result.ok = true;
        result.request_id = plan.request_id;
        result.accepted_count = plan.accepted_count;
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
                const std::string post_error = publishPostRestoreStage(
                    stage,
                    i,
                    device,
                    stream,
                    "MTP spec-state publication",
                    result);
                if (!post_error.empty())
                    return publicationFailure(plan, post_error);
                if (!stage->requiresPostVerifierStatePublication())
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
            const std::string post_error = publishPostRestoreStage(
                stage,
                i,
                device,
                stream,
                "MTP spec-state publication",
                result);
            if (!post_error.empty())
                return publicationFailure(plan, post_error);
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

        if (!any_restore)
        {
            result.skipped_stage_count = static_cast<int>(state_stages.size());
            return result;
        }

        if (plans.request_count == 1)
        {
            /*
             * The overwhelmingly common server transaction owns one request
             * and one independent live-state allocation per recurrent stage.
             * Planning every copy first makes bad rows, stale workspaces, and
             * overlapping destinations fatal before any byte becomes live.
             * Once validated, one persistent OpenMP team fans the independent
             * layer copies across physical cores. This replaces dozens of
             * serial memory-bandwidth operations without changing one byte or
             * replaying any recurrence arithmetic.
             */
            std::vector<CPUVerifierStateRestorePlan> restore_plans;
            restore_plans.reserve(state_stages.size());
            struct RestoreCopyChunk
            {
                void *destination = nullptr;
                const void *source = nullptr;
                size_t bytes = 0;
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
                        msg << "single-request CPU MTP publication received null stage at index "
                            << i;
                        return batchPublicationFailure(plans, msg.str());
                    }
                    if (!stage->hasVerifierStateCapture())
                    {
                        if (require_captured_stage &&
                            stage->requiresVerifierStateCaptureForPublication())
                        {
                            std::ostringstream msg;
                            msg << "single-request CPU MTP publication required verifier capture for stage "
                                << stage->name() << " at index " << i;
                            return batchPublicationFailure(plans, msg.str());
                        }
                        continue;
                    }

                    CPUVerifierStateRestorePlan restore =
                        stage->planCPUVerifierStateRestoreRow(restore_row);
                    if (!restore.ready())
                    {
                        std::ostringstream msg;
                        msg << "single-request CPU MTP publication could not plan a native restore for captured stage "
                            << stage->name() << " at index " << i
                            << " status=" << static_cast<int>(restore.status);
                        return batchPublicationFailure(plans, msg.str());
                    }

                    const uintptr_t destination_begin =
                        reinterpret_cast<uintptr_t>(restore.destination);
                    const uintptr_t source_begin =
                        reinterpret_cast<uintptr_t>(restore.source);
                    if (restore.bytes >
                            std::numeric_limits<uintptr_t>::max() - destination_begin ||
                        restore.bytes >
                            std::numeric_limits<uintptr_t>::max() - source_begin ||
                        restore.bytes >
                            std::numeric_limits<size_t>::max() - restore_bytes)
                    {
                        return batchPublicationFailure(
                            plans,
                            "single-request CPU MTP publication restore span overflowed its address domain");
                    }
                    const uintptr_t destination_end =
                        destination_begin + restore.bytes;
                    const uintptr_t source_end = source_begin + restore.bytes;
                    if (destination_begin < source_end &&
                        source_begin < destination_end)
                    {
                        std::ostringstream msg;
                        msg << "single-request CPU MTP publication source aliases its live destination for stage "
                            << stage->name() << " at index " << i;
                        return batchPublicationFailure(plans, msg.str());
                    }
                    for (const CPUVerifierStateRestorePlan &planned : restore_plans)
                    {
                        const uintptr_t planned_begin =
                            reinterpret_cast<uintptr_t>(planned.destination);
                        const uintptr_t planned_end = planned_begin + planned.bytes;
                        if (destination_begin < planned_end &&
                            planned_begin < destination_end)
                        {
                            std::ostringstream msg;
                            msg << "single-request CPU MTP publication found overlapping live-state destinations at stage "
                                << stage->name() << " index " << i;
                            return batchPublicationFailure(plans, msg.str());
                        }
                    }

                    restore_bytes += restore.bytes;
                    restore_plans.push_back(restore);
                }

                restore_chunks.reserve(
                    restore_bytes / kRestoreCopyChunkBytes +
                    restore_plans.size());
                for (const CPUVerifierStateRestorePlan &restore : restore_plans)
                {
                    auto *destination = static_cast<std::byte *>(restore.destination);
                    const auto *source = static_cast<const std::byte *>(restore.source);
                    for (size_t offset = 0; offset < restore.bytes;
                         offset += kRestoreCopyChunkBytes)
                    {
                        const size_t bytes = std::min(
                            kRestoreCopyChunkBytes,
                            restore.bytes - offset);
                        restore_chunks.push_back({
                            .destination = destination + offset,
                            .source = source + offset,
                            .bytes = bytes,
                        });
                    }
                }
            }

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
                        std::memcpy(
                            restore.destination,
                            restore.source,
                            restore.bytes);
                    }
                };
                OMP_WORKSHARE_REGION(copy_restore_plans);
            }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "spec_state_cpu_parallel_restore_finalize",
                    "decode");
                for (size_t i = 0; i < state_stages.size(); ++i)
                {
                    IComputeStage *stage = state_stages[i];
                    if (stage->hasVerifierStateCapture())
                    {
                        ++result.restored_stage_count;
                        const std::string post_error = publishPostRestoreStage(
                            stage,
                            i,
                            device,
                            stream,
                            "single-request CPU MTP publication",
                            result);
                        if (!post_error.empty())
                            return batchPublicationFailure(plans, post_error);
                        stage->clearVerifierStateCaptureBindingAfterPublication();
                        continue;
                    }

                    const std::string post_error = publishPostRestoreStage(
                        stage,
                        i,
                        device,
                        stream,
                        "single-request CPU MTP publication",
                        result);
                    if (!post_error.empty())
                        return batchPublicationFailure(plans, post_error);
                    if (!stage->requiresPostVerifierStatePublication())
                        ++result.skipped_stage_count;
                }
            }

            if (require_captured_stage && result.restored_stage_count == 0)
            {
                return batchPublicationFailure(
                    plans,
                    "single-request CPU MTP publication required a captured stage but restored none");
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
                 {"publication_policy", "validated_parallel_layer_copies"}});
            return result;
        }

        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (!stage)
            {
                std::ostringstream msg;
                msg << "host-indexed batched MTP spec-state publication received null stage at index "
                    << i;
                return batchPublicationFailure(plans, msg.str());
            }
            if (!stage->hasVerifierStateCapture())
            {
                if (require_captured_stage &&
                    stage->requiresVerifierStateCaptureForPublication())
                {
                    std::ostringstream msg;
                    msg << "host-indexed batched MTP spec-state publication required verifier capture for stage "
                        << stage->name() << " at index " << i;
                    return batchPublicationFailure(plans, msg.str());
                }
                const std::string post_error = publishPostRestoreStage(
                    stage,
                    i,
                    device,
                    stream,
                    "host-indexed batched MTP spec-state publication",
                    result);
                if (!post_error.empty())
                    return batchPublicationFailure(plans, post_error);
                if (!stage->requiresPostVerifierStatePublication())
                    ++result.skipped_stage_count;
                continue;
            }
            if (!stage->restoreVerifierStateCaptureRows(
                    host_verifier_restore_rows,
                    plans.request_count,
                    stream))
            {
                std::ostringstream msg;
                msg << "host-indexed batched MTP spec-state publication failed grouped restore for stage "
                    << stage->name() << " at index " << i;
                return batchPublicationFailure(plans, msg.str());
            }
            ++result.restored_stage_count;
            const std::string post_error = publishPostRestoreStage(
                stage,
                i,
                device,
                stream,
                "host-indexed batched MTP spec-state publication",
                result);
            if (!post_error.empty())
                return batchPublicationFailure(plans, post_error);
            stage->clearVerifierStateCaptureBindingAfterPublication();
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return batchPublicationFailure(
                plans,
                "host-indexed batched MTP spec-state publication required a captured stage but restored none");
        }
        return result;
    }

    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPSpecStepPlan &plan,
        const int *device_verifier_restore_row,
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
        if (!device_verifier_restore_row)
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
        const int *device_verifier_restore_row,
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
        if (!device_verifier_restore_row)
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
                const std::string post_error = publishPostRestoreStage(
                    stage,
                    i,
                    device,
                    stream,
                    "device-indexed MTP spec-state publication",
                    result);
                if (!post_error.empty())
                    return devicePublicationFailure(shape, post_error);
                if (!stage->requiresPostVerifierStatePublication())
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
            const std::string post_error = publishPostRestoreStage(
                stage,
                i,
                device,
                stream,
                "device-indexed MTP spec-state publication",
                result);
            if (!post_error.empty())
                return devicePublicationFailure(shape, post_error);
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
        const int *device_verifier_restore_rows,
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
        const int *device_verifier_restore_rows,
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
        if (!device_verifier_restore_rows)
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
                const std::string post_error = publishPostRestoreStage(
                    stage,
                    i,
                    device,
                    stream,
                    "batched device-indexed MTP spec-state publication",
                    result);
                if (!post_error.empty())
                    return devicePublicationFailure(shape, post_error);
                if (!stage->requiresPostVerifierStatePublication())
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
            const std::string post_error = publishPostRestoreStage(
                stage,
                i,
                device,
                stream,
                "batched device-indexed MTP spec-state publication",
                result);
            if (!post_error.empty())
                return devicePublicationFailure(shape, post_error);
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
        const int *device_verifier_restore_row,
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
        const int *device_verifier_restore_row,
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
        const int *device_verifier_restore_rows,
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
        const int *device_verifier_restore_rows,
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
