#pragma once

#include "MTPSpecStateContract.h"
#include "../../backends/DeviceId.h"

#include <string>
#include <vector>

namespace llaminar2
{
    class ComputeGraph;
    class IComputeStage;

    struct MTPSpecStatePublicationResult
    {
        bool ok = false;
        std::string error;

        int request_id = -1;
        int accepted_count = 0;
        int restored_stage_count = 0;
        int skipped_stage_count = 0;
    };

    MTPSpecStatePublicationResult publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

    MTPSpecStatePublicationResult publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

} // namespace llaminar2
