#include "MTPSpecStatePublisher.h"

#include "../compute_stages/IComputeStage.h"
#include "../local_execution/graph/ComputeGraph.h"
#include "../../utils/OpenMPUtils.h"

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
        if (device.is_cpu() && state_stages.size() > 1)
        {
            std::vector<int> status(state_stages.size(), 0);
            auto publish_work = [&]()
            {
#pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(state_stages.size()); ++i)
                {
                    IComputeStage *stage = state_stages[static_cast<size_t>(i)];
                    if (stage == nullptr)
                    {
                        status[static_cast<size_t>(i)] = -2;
                        continue;
                    }
                    if (!stage->hasVerifierStateCapture())
                    {
                        status[static_cast<size_t>(i)] = 0;
                        continue;
                    }
                    status[static_cast<size_t>(i)] =
                        stage->restoreVerifierStateCaptureRow(restore_row, stream) ? 1 : -1;
                }
            };
            OMP_WORKSHARE_REGION_IF(publish_work, state_stages.size() > 1);

            for (size_t i = 0; i < state_stages.size(); ++i)
            {
                if (status[i] == 0)
                {
                    ++result.skipped_stage_count;
                    continue;
                }
                if (status[i] == 1)
                {
                    ++result.restored_stage_count;
                    continue;
                }

                std::ostringstream msg;
                if (status[i] == -2)
                {
                    msg << "MTP spec-state publication received null stage at index "
                        << i;
                }
                else
                {
                    IComputeStage *stage = state_stages[i];
                    msg << "MTP spec-state publication failed restoring verifier row "
                        << restore_row << " for stage "
                        << (stage ? stage->name() : "<null>")
                        << " at index " << i;
                }
                return publicationFailure(plan, msg.str());
            }

            if (require_captured_stage && result.restored_stage_count == 0)
            {
                return publicationFailure(
                    plan,
                    "MTP spec-state publication required a verifier-captured state stage but restored none");
            }

            return result;
        }

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

} // namespace llaminar2
