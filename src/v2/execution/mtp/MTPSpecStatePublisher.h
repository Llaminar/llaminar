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

    /**
     * @brief Publish verifier state from a row index stored in device memory.
     *
     * Phase 10 device-resident stochastic publication derives the accepted
     * verifier row from compact GPU metadata. This helper restores every
     * verifier-capturing stage by reading that row index on @p stream, avoiding
     * a host synchronization solely to materialize `accepted_count - 1`.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPSpecStepPlan &plan,
        const int *device_verifier_restore_row,
        const std::vector<IComputeStage *> &state_stages,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

    /**
     * @brief Publish request-batched verifier state from device row indices.
     *
     * The row-index buffer is device-resident and laid out with
     * `device_verifier_restore_rows[request * row_index_stride]`.  Capturing
     * stages are invoked once through their batch restore hook; implementations
     * must restore into request-owned live-state slots and must not emulate this
     * by looping the scalar restore over one shared layer state.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *device_verifier_restore_rows,
        int row_index_stride,
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

    /**
     * @brief Graph-order variant of publishAcceptedMTPSpecStateFromDeviceVerifierRow().
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPSpecStepPlan &plan,
        const int *device_verifier_restore_row,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

    /**
     * @brief Graph-order variant of publishAcceptedMTPSpecStateFromDeviceVerifierRows().
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *device_verifier_restore_rows,
        int row_index_stride,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

} // namespace llaminar2
