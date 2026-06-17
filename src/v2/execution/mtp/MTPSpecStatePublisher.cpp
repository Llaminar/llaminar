#include "MTPSpecStatePublisher.h"

#include "../compute_stages/IComputeStage.h"
#include "../local_execution/graph/ComputeGraph.h"

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
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return publicationFailure(
                plan,
                "MTP spec-state publication required a verifier-captured state stage but restored none");
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

        MTPSpecStatePublicationResult result;
        result.ok = true;
        result.request_id = plan.request_id;
        result.accepted_count = plan.accepted_count;

        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (stage == nullptr)
            {
                std::ostringstream msg;
                msg << "device-indexed MTP spec-state publication received null stage at index "
                    << i;
                return publicationFailure(plan, msg.str());
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
                    return publicationFailure(plan, msg.str());
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
                return publicationFailure(plan, msg.str());
            }
            ++result.restored_stage_count;
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return publicationFailure(
                plan,
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
        if (!device.is_valid())
            return batchPublicationFailure(plans, "cannot publish batched MTP spec state on invalid device");
        if (!device.is_gpu())
        {
            return batchPublicationFailure(
                plans,
                "batched device-indexed MTP spec-state publication currently requires a GPU device");
        }
        if (stream == nullptr)
        {
            return batchPublicationFailure(
                plans,
                "batched GPU device-indexed MTP spec-state publication requires an explicit non-null stream");
        }
        if (!device_verifier_restore_rows)
        {
            return batchPublicationFailure(
                plans,
                "batched device-indexed MTP spec-state publication received a null row pointer");
        }
        if (row_index_stride <= 0)
        {
            return batchPublicationFailure(
                plans,
                "batched device-indexed MTP spec-state publication received an invalid row-index stride");
        }
        if (!plans.ok || plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return batchPublicationFailure(
                plans,
                plans.error.empty()
                    ? "batched MTP spec-state publication received an invalid step plan batch"
                    : plans.error);
        }

        MTPSpecStatePublicationResult result;
        result.ok = true;
        for (const MTPSpecStepPlan &step : plans.steps)
            result.accepted_count += step.accepted_count;

        bool any_accepted = false;
        for (const MTPSpecStepPlan &step : plans.steps)
        {
            if (step.accepted_count < 0 || step.accepted_count > step.draft_count)
            {
                return batchPublicationFailure(
                    plans,
                    "batched MTP spec-state publication step accepted count is outside the draft prefix");
            }
            any_accepted = any_accepted || step.accepted_count > 0;
        }
        if (!any_accepted)
        {
            result.skipped_stage_count = static_cast<int>(state_stages.size());
            return result;
        }

        for (size_t i = 0; i < state_stages.size(); ++i)
        {
            IComputeStage *stage = state_stages[i];
            if (stage == nullptr)
            {
                std::ostringstream msg;
                msg << "batched device-indexed MTP spec-state publication received null stage at index "
                    << i;
                return batchPublicationFailure(plans, msg.str());
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
                    return batchPublicationFailure(plans, msg.str());
                }
                ++result.skipped_stage_count;
                continue;
            }
            if (!stage->restoreVerifierStateCaptureRowsFromDeviceIndices(
                    device_verifier_restore_rows,
                    plans.request_count,
                    row_index_stride,
                    stream))
            {
                std::ostringstream msg;
                msg << "batched device-indexed MTP spec-state publication failed restoring verifier rows for stage "
                    << stage->name()
                    << " at index " << i;
                return batchPublicationFailure(plans, msg.str());
            }
            ++result.restored_stage_count;
        }

        if (require_captured_stage && result.restored_stage_count == 0)
        {
            return batchPublicationFailure(
                plans,
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
        std::vector<IComputeStage *> stages;
        const auto &order = graph.getExecutionOrder();
        stages.reserve(order.size());

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (node == nullptr)
            {
                return publicationFailure(
                    plan,
                    "MTP spec-state graph publication references missing node '" +
                        node_name + "'");
            }
            if (!node->stage)
            {
                return publicationFailure(
                    plan,
                    "MTP spec-state graph publication found node '" +
                        node_name + "' without a stage");
            }

            stages.push_back(node->stage.get());
        }

        return publishAcceptedMTPSpecStateFromVerifierRow(
            plan,
            verifier_restore_row,
            stages,
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
        std::vector<IComputeStage *> stages;
        const auto &order = graph.getExecutionOrder();
        stages.reserve(order.size());

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (node == nullptr)
            {
                return publicationFailure(
                    plan,
                    "device-indexed MTP spec-state graph publication references missing node '" +
                        node_name + "'");
            }
            if (!node->stage)
            {
                return publicationFailure(
                    plan,
                    "device-indexed MTP spec-state graph publication found node '" +
                        node_name + "' without a stage");
            }

            stages.push_back(node->stage.get());
        }

        return publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            plan,
            device_verifier_restore_row,
            stages,
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
        std::vector<IComputeStage *> stages;
        const auto &order = graph.getExecutionOrder();
        stages.reserve(order.size());

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (node == nullptr)
            {
                return batchPublicationFailure(
                    plans,
                    "batched device-indexed MTP spec-state graph publication references missing node '" +
                        node_name + "'");
            }
            if (!node->stage)
            {
                return batchPublicationFailure(
                    plans,
                    "batched device-indexed MTP spec-state graph publication found node '" +
                        node_name + "' without a stage");
            }

            stages.push_back(node->stage.get());
        }

        return publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            plans,
            device_verifier_restore_rows,
            row_index_stride,
            stages,
            device,
            stream,
            require_captured_stage);
    }

} // namespace llaminar2
