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

    /**
     * @brief Publish accepted verifier state from an explicit graph row.
     *
     * Request-batched verifier graphs are padded: request-local accepted row
     * `accepted_count - 1` is not necessarily the same as the physical row in
     * the flattened graph. This overload lets the caller supply the already
     * materialized verifier graph row while reusing the same stage-restore
     * contract and validation as the single-request helper.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRow(
        const MTPSpecStepPlan &plan,
        int verifier_restore_row,
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

    /**
     * @brief Graph-order variant of publishAcceptedMTPSpecStateFromVerifierRow().
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRow(
        const MTPSpecStepPlan &plan,
        int verifier_restore_row,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

} // namespace llaminar2
