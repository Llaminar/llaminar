/**
 * @file MoEOverlayDeviceControllerRuntimeBinding.h
 * @brief Typed bridge from a model-owned MoE runtime table to device control.
 *
 * The topology-wide ExpertOverlay authority uses global participant ids, while
 * each homogeneous execution domain stores owner masks in its own dense local
 * namespace.  This record carries both identities beside the one authoritative
 * device runtime pointer.  It is immutable model topology resolved before graph
 * capture; no histogram values or placement decisions cross the host boundary.
 */

#pragma once

#include "backends/DeviceId.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

namespace llaminar2
{
    struct DeviceMoELayerRuntime;
    struct DeviceMoEOverlayServiceTelemetryCell;
    struct DeviceMoEOverlayServiceTelemetrySample;
    struct DeviceMoEOverlayEpochControl;
    struct DeviceMoEOverlayEpochStatus;
    class DeviceMoERuntimeTable;
    class MappedTransferProgressEpoch;

    /**
     * @brief Lock-free lifecycle receipt for mapped follower inference terminals.
     *
     * A heterogeneous mapped follower necessarily observes one exact terminal
     * GPU event before it can retire the activation lease and accept the next
     * MPI command.  Re-recording an event later, when background maintenance
     * finally consumes that boundary, is incorrect: a newer graph may already
     * be queued on the shared participant stream.  This state machine instead
     * publishes a monotonically numbered submission and marks that exact
     * generation complete only after its terminal event has been observed.
     *
     * The receipt contains no epoch, policy, histogram, pointer, or request
     * payload.  Device state remains device-owned.  It is solely the immutable
     * host-side proof created by the explicit heterogeneous transaction
     * boundary.  A topology-wide decision epoch may consume a boundary only
     * when the latest submitted generation is complete.  Starting the global
     * decision from one fast participant while another participant is still in
     * the newer sparse transaction can form a cross-rank wait cycle.  The
     * maintenance submitter therefore defers; inference itself never waits.
     */
    class MoEOverlayInferenceBoundaryReceipt
    {
    public:
        /** One acquire-consistent view consumed by the maintenance submitter. */
        struct Snapshot
        {
            std::uint64_t submitted_generation = 0u;
            std::uint64_t completed_generation = 0u;
            bool fresh_completion = false;

            /** @return Whether at least one exact terminal was completed. */
            [[nodiscard]] constexpr bool hasCompletedBoundary()
                const noexcept
            {
                return completed_generation != 0u;
            }

            /** @return Whether a newer inference transaction is still live. */
            [[nodiscard]] constexpr bool newerSubmissionInFlight()
                const noexcept
            {
                return submitted_generation > completed_generation;
            }
        };

        /**
         * @brief Allocate the next serial follower submission generation.
         *
         * Call before submitting the retained graph.  The release publication
         * closes the maintenance-admission race: once this method returns, a
         * maintenance reader observes the upcoming transaction as in flight
         * even if the backend worker has not accepted the graph yet.  A zero
         * return means the monotonic namespace was exhausted and is fatal; no
         * generation is ever wrapped or reused.
         *
         * @return Non-zero immutable generation, or zero on exhaustion.
         */
        [[nodiscard]] std::uint64_t beginSubmission() noexcept
        {
            std::uint64_t current = submitted_generation_.load(
                std::memory_order_relaxed);
            while (current != std::numeric_limits<std::uint64_t>::max())
            {
                if (submitted_generation_.compare_exchange_weak(
                        current,
                        current + 1u,
                        std::memory_order_release,
                        std::memory_order_relaxed))
                {
                    return current + 1u;
                }
            }
            return 0u;
        }

        /**
         * @brief Roll back an admission whose graph was never submitted.
         *
         * Backend rejection can occur after the receipt has closed the race
         * window but before any device work exists.  Only that newest serial
         * generation may be removed, and only while the preceding generation
         * is already complete.  A caller must never cancel after a graph launch
         * succeeds; that graph owns the generation until its exact terminal is
         * observed through @ref completeSubmission.
         *
         * @param generation Value returned by @ref beginSubmission.
         * @return True only when the not-launched newest admission was removed.
         */
        [[nodiscard]] bool cancelUnsubmitted(
            std::uint64_t generation) noexcept
        {
            if (generation == 0u ||
                completed_generation_.load(std::memory_order_acquire) !=
                    generation - 1u)
            {
                return false;
            }
            std::uint64_t expected = generation;
            return submitted_generation_.compare_exchange_strong(
                expected,
                generation - 1u,
                std::memory_order_acq_rel,
                std::memory_order_relaxed);
        }

        /**
         * @brief Publish one exact terminal-event completion in serial order.
         *
         * Mapped follower transactions are serial for one participant.  The
         * compare/exchange makes out-of-order completion unrepresentable and
         * release-publishes every device write already proven by the terminal
         * event before maintenance can acquire this generation.
         *
         * @param generation Value returned by @ref beginSubmission.
         * @return True only for the next submitted, not-yet-completed value.
         */
        [[nodiscard]] bool completeSubmission(
            std::uint64_t generation) noexcept
        {
            if (generation == 0u ||
                generation > submitted_generation_.load(
                                 std::memory_order_acquire))
            {
                return false;
            }
            std::uint64_t expected = generation - 1u;
            return completed_generation_.compare_exchange_strong(
                expected,
                generation,
                std::memory_order_release,
                std::memory_order_relaxed);
        }

        /**
         * @brief Consume the newest immutable completed boundary.
         *
         * This method never waits for an in-flight generation.  It may return
         * an older completed generation beside a newer submitted generation so
         * the caller can defer a topology-wide epoch without blocking
         * inference.  Repeated notifications are coalescible and therefore
         * return the same valid generation with `fresh_completion == false`.
         *
         * @return Acquire-consistent submission/completion snapshot.
         */
        [[nodiscard]] Snapshot consumeLatestCompleted() noexcept
        {
            Snapshot result;
            result.completed_generation = completed_generation_.load(
                std::memory_order_acquire);
            result.submitted_generation = submitted_generation_.load(
                std::memory_order_acquire);

            std::uint64_t consumed = consumed_generation_.load(
                std::memory_order_relaxed);
            while (consumed < result.completed_generation)
            {
                if (consumed_generation_.compare_exchange_weak(
                        consumed,
                        result.completed_generation,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed))
                {
                    result.fresh_completion = true;
                    break;
                }
            }
            return result;
        }

        /** @return Latest submitted generation for diagnostics and tests. */
        [[nodiscard]] std::uint64_t submittedGeneration() const noexcept
        {
            return submitted_generation_.load(std::memory_order_acquire);
        }

        /** @return Latest terminal-event-certified generation. */
        [[nodiscard]] std::uint64_t completedGeneration() const noexcept
        {
            return completed_generation_.load(std::memory_order_acquire);
        }

        /** @return Latest generation acquired by a maintenance submitter. */
        [[nodiscard]] std::uint64_t consumedGeneration() const noexcept
        {
            return consumed_generation_.load(std::memory_order_acquire);
        }

    private:
        std::atomic<std::uint64_t> submitted_generation_{0u};
        std::atomic<std::uint64_t> completed_generation_{0u};
        std::atomic<std::uint64_t> consumed_generation_{0u};
    };

    /**
     * @brief Result of admitting a maintenance reader at an inference boundary.
     *
     * `Deferred` is ordinary overlap with an admitted inference reader or a
     * host-submitted publication whose ready event has not been recorded yet.
     * The maintenance authority may retry the complete boundary fan-in, but it
     * must not launch a partial controller epoch. `Failed` is reserved for a
     * malformed or rejected lifecycle edge.
     */
    enum class MoEOverlayInferenceBoundaryStatus : std::uint8_t
    {
        Submitted = 0,
        Deferred = 1,
        Failed = 2,
    };

    /**
     * @brief Typed reason a maintenance reader joins an inference terminal.
     *
     * Service telemetry is observational and is valid for either host- or
     * device-resident ExpertOverlay authority. The remaining purposes expose
     * placement state to the sole device controller and are therefore legal
     * only for a device-resident authority. Keeping that distinction in the
     * request type prevents a diagnostic string from accidentally granting a
     * host controller access to a device-policy transition.
     */
    enum class MoEOverlayInferenceBoundaryPurpose : std::uint8_t
    {
        ServiceTelemetrySnapshot = 0,
        HistogramRebase = 1,
        PlacementDecisionSnapshot = 2,
    };

    /** @brief Immutable semantic request for one inference-to-maintenance edge. */
    struct MoEOverlayInferenceBoundaryRequest
    {
        MoEOverlayInferenceBoundaryPurpose purpose =
            MoEOverlayInferenceBoundaryPurpose::ServiceTelemetrySnapshot;

        /** @return Stable PerfStats identity derived from the typed purpose. */
        [[nodiscard]] constexpr const char *name() const noexcept
        {
            switch (purpose)
            {
            case MoEOverlayInferenceBoundaryPurpose::
                ServiceTelemetrySnapshot:
                return "service_telemetry_snapshot";
            case MoEOverlayInferenceBoundaryPurpose::HistogramRebase:
                return "histogram_rebase";
            case MoEOverlayInferenceBoundaryPurpose::
                PlacementDecisionSnapshot:
                return "placement_decision_snapshot";
            }
            return "invalid";
        }
    };

    /**
     * @brief Exact inference-to-maintenance RCU boundary for one GPU participant.
     *
     * A topology-wide device controller must close the participant's currently
     * acquired ExpertOverlay reader before it may build or publish a successor
     * runtime bank.  The controller service owns a different stream from the
     * inference graph, so it cannot infer this edge from a backend default or
     * from a host-side epoch mirror.  This interface lets the model executor
     * join every live inference producer and enqueue the captured reader
     * release on the caller-supplied stream.  An explicitly segmented mapped
     * follower may instead consume a @ref MoEOverlayInferenceBoundaryReceipt:
     * its exact terminal event was already observed before the follower could
     * return the committed MPI command, so recording a second event on its
     * mutable stream would target the wrong transaction.
     *
     * The operation is submission-only: it may enqueue event waits and retained
     * graph launches, but it must never synchronize a stream or wait for the
     * device.  Implementations remain the sole owners of request-local reader
     * lifecycle state.
     */
    class IMoEOverlayDeviceInferenceBoundary
    {
    public:
        virtual ~IMoEOverlayDeviceInferenceBoundary() = default;

        /**
         * @brief Enqueue the committed inference boundary on an exact stream.
         * @param maintenance_stream Non-null stream owned by this participant.
         * @param request Typed observation or placement-policy boundary.
         * @return Submitted after all waits/releases were enqueued, Deferred
         *         while an inference reader/publication still owns the
         *         boundary, or Failed for an invalid lifecycle.
         */
        [[nodiscard]] virtual MoEOverlayInferenceBoundaryStatus
        enqueueMoEOverlayDeviceInferenceBoundary(
            void *maintenance_stream,
            MoEOverlayInferenceBoundaryRequest request) = 0;

        /**
         * @brief Install this GPU's topology-accounted relay progress epoch.
         * @param epoch Non-null epoch whose device matches this participant.
         * @return True after idempotent setup-time installation.
         *
         * Implementations submit the retained epoch before their native
         * inference transaction without making that transaction wait for it.
         */
        [[nodiscard]] virtual bool installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch) = 0;
    };

    /** Immutable snapshot source for one global overlay participant. */
    struct MoEOverlayDeviceControllerRuntimeBinding
    {
        DeviceId device = DeviceId::invalid();
        DeviceMoELayerRuntime *runtime_layers_device = nullptr;
        /**
         * Model-lifetime host recipe authority for the device runtime banks.
         *
         * This pointer does not mirror live GPU execution state.  A
         * host-resident heterogeneous controller uses it only to build one
         * complete inactive-bank recipe before bounded H2D publication and to
         * acknowledge the exact bank after the device selector has changed.
         * The graph builder retains the table for the full binding lifetime.
         */
        DeviceMoERuntimeTable *runtime_table_host = nullptr;
        /** Canonical device-local `[layer][phase]` service accumulators. */
        DeviceMoEOverlayServiceTelemetryCell *service_telemetry_device =
            nullptr;
        /** Canonical device-local per-layer timing cursors. */
        DeviceMoEOverlayServiceTelemetrySample *service_samples_device =
            nullptr;
        std::int32_t overlay_participant_id = -1;
        std::uint32_t domain_participant_id = 0u;
        std::uint32_t domain_participant_count = 0u;
        std::uint32_t layer_count = 0u;
        std::uint32_t expert_count = 0u;
        std::uint32_t top_k = 0u;
        /** Participant-local RCU publication authority. */
        DeviceMoEOverlayEpochControl *epoch_control = nullptr;
        /** Stable candidate scalar used by retained maintenance graphs. */
        std::uint64_t *maintenance_epoch = nullptr;
        /** Stable reserve/ready/publish/retire semantic status. */
        DeviceMoEOverlayEpochStatus *maintenance_status = nullptr;
        /** Participant-local owner of the live inference reader boundary. */
        IMoEOverlayDeviceInferenceBoundary *inference_boundary = nullptr;

        /**
         * @return Whether global identity, local runtime identity, and geometry
         *         describe one addressable GPU participant.
         */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return device.is_gpu() && runtime_layers_device != nullptr &&
                   overlay_participant_id >= 0 &&
                   domain_participant_count > 0u &&
                   domain_participant_id < domain_participant_count &&
                   layer_count > 0u && expert_count > 0u && top_k > 0u;
        }

        /**
         * @return Whether this participant can publish device-owned placement.
         *
         * Snapshot-only Static certification needs only @ref valid. Dynamic
         * graph construction requires this stronger contract so a host-side
         * bank recipe cannot be substituted accidentally.
         */
        [[nodiscard]] constexpr bool publicationValid() const noexcept
        {
            return valid() && epoch_control != nullptr &&
                   maintenance_epoch != nullptr &&
                   maintenance_status != nullptr;
        }

        /**
         * @return Whether Dynamic economy can snapshot real GPU service work.
         *
         * Static bindings intentionally return false. Dynamic graph-service
         * construction requires this in addition to @ref publicationValid so
         * missing telemetry cannot be disguised as an indefinitely calibrating
         * controller.
         */
        [[nodiscard]] constexpr bool serviceTelemetryValid() const noexcept
        {
            return valid() && service_telemetry_device != nullptr &&
                   service_samples_device != nullptr;
        }

        /**
         * @return Whether production background publication can close readers.
         *
         * Device-kernel fixtures may deliberately use @ref publicationValid
         * without a live inference executor.  Production Dynamic composition
         * requires this stronger contract and must never synthesize the missing
         * boundary with a stream synchronize.
         */
        [[nodiscard]] constexpr bool backgroundPublicationValid() const noexcept
        {
            return publicationValid() && inference_boundary != nullptr;
        }

        /**
         * @return Whether a host authority can prepare and publish GPU banks.
         *
         * Device-resident homogeneous controllers deliberately need no host
         * table pointer.  This stronger predicate belongs only to the explicit
         * host-authority heterogeneous publication path.
         */
        [[nodiscard]] constexpr bool hostPublicationValid() const noexcept
        {
            return backgroundPublicationValid() &&
                   runtime_table_host != nullptr;
        }
    };

    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerRuntimeBinding>);
} // namespace llaminar2
