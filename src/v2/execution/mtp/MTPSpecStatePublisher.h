/**
 * @file MTPSpecStatePublisher.h
 * @brief Verifier-state publication helpers for MTP speculative decode.
 *
 * These helpers move accepted verifier-row state directly into each live state
 * owner after a speculative verifier pass. Host-plan overloads exist for CPU
 * and transaction tests. GPU resident publication uses device-indexed request
 * shapes so accepted rows remain device-owned through the exact stage restore
 * hook; there is no secondary derived-state publication lifecycle.
 */

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

    /**
     * @brief Host-known shape for device-indexed verifier-state publication.
     *
     * The compact GPU reducer owns the accepted-count and accepted-row values.
     * Publication still needs structural bounds that are known before graph
     * replay: how many request lanes are present and how many verifier rows were
     * padded into each lane.  Keeping this separate from MTPSpecStepPlan avoids
     * smuggling host accepted counts, cache positions, or row-replay plans into
     * the GPU resident publication path.
     */
    struct MTPDeviceVerifierStatePublicationShape
    {
        int request_count = 1;
        int target_rows = 0;
        int request_id = 0;

        bool validScalar() const
        {
            return request_count == 1 && target_rows > 0;
        }

        bool validBatch() const
        {
            return request_count > 0 && target_rows > 0;
        }
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
     * @brief Publish a CPU request batch from host-selected verifier rows.
     *
     * Capturing stages receive the complete row vector exactly once and copy
     * each accepted row into request-owned live state.  This replaces the
     * obsolete scalar loop that repeatedly overwrote one stage state and
     * cleared capture bindings after the first request.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *host_verifier_restore_rows,
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
     * @brief Publish a scalar request from a device-owned verifier row index.
     *
     * This is the preferred GPU resident entry point.  The function forwards the
     * row-index pointer directly to verifier-capturing stages; it never reads an
     * accepted count or restore row from a host MTPSpecStepPlan.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPDeviceVerifierStatePublicationShape &shape,
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

    /**
     * @brief Publish a request batch from device-owned verifier row indices.
     *
     * Request-aware stages receive the device row-index buffer once with the
     * request count and stride.  Rejected rows are represented by negative
     * device row indices produced by the reducer; the host does not decide which
     * lanes restore.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPDeviceVerifierStatePublicationShape &shape,
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
     * @brief Graph-order CPU batch variant of host verifier-row publication.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromVerifierRows(
        const MTPSpecStepPlanBatch &plans,
        const int *host_verifier_restore_rows,
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
     * @brief Graph-order variant of the scalar device-shape publisher.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRow(
        const MTPDeviceVerifierStatePublicationShape &shape,
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

    /**
     * @brief Graph-order variant of the batched device-shape publisher.
     */
    MTPSpecStatePublicationResult publishAcceptedMTPSpecStateFromDeviceVerifierRows(
        const MTPDeviceVerifierStatePublicationShape &shape,
        const int *device_verifier_restore_rows,
        int row_index_stride,
        ComputeGraph &graph,
        DeviceId device,
        void *stream,
        bool require_captured_stage = false);

} // namespace llaminar2
