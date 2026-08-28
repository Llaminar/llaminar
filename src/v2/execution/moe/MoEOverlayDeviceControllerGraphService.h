/**
 * @file MoEOverlayDeviceControllerGraphService.h
 * @brief Retained background graphs for the all-GPU ExpertOverlay authority.
 *
 * A multi-group all-GPU overlay cannot use a host placement controller, and
 * its controller traffic must not be placed on an inference stream. This
 * service materializes a model-lifetime family of finite CUDA/HIP graphs for
 * every local participant. Participant graphs pack model-owned runtime state;
 * group roots aggregate lifecycle edges; the topology leader owns policy and
 * epoch publication. A background worker observes only immutable transaction
 * tickets and monotonic lifecycle words to submit the next eligible retained
 * epoch. It never reads histograms or placement. Short topology fan-in waits
 * are legal only among graphs already pre-submitted for that same epoch; no
 * graph may remain resident for a later host receipt, transfer, reader drain,
 * or participant-graph submission.
 */

#pragma once

#include "MoEOverlayDeviceControllerABI.h"
#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "../InferenceMeasurementReadiness.h"
#include "MoEOptimizationStatus.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
    class MoEOverlayDeviceControllerTopology;
    class MoEOverlayNodeLocalDeviceControllerFabric;
    class MoEOverlayPhysicalResidencyFabric;
    class MoEOverlayEconomyCertificationController;

    /** Inference phase whose retired logical rows advance maintenance. */
    enum class MoEOverlayInferencePhase : std::uint8_t
    {
        Prefill = 0u, ///< One retained bucket contributes its real prompt rows.
        Decode = 1u, ///< One committed serial token contributes one row.
    };

    /**
     * @brief Thread-safe inference-token budget for one maintenance window.
     *
     * Inference threads only publish monotonic boundary notifications. The
     * background controller consumes a complete window at once; notifications
     * racing that consume either join the returned batch or remain for the
     * next one. The counter saturates instead of wrapping during a stalled
     * maintenance worker.
     */
    class MoEOverlayMaintenanceBoundaryGate final
    {
    public:
        /** One coalesced, phase-pure maintenance admission. */
        struct ReadyWindow
        {
            MoEOverlayInferencePhase phase =
                MoEOverlayInferencePhase::Prefill;
            std::uint64_t completed_tokens = 0u;

            /** @return Whether this record admits a controller transaction. */
            [[nodiscard]] explicit operator bool() const noexcept
            {
                return completed_tokens != 0u;
            }
        };

        /**
         * @brief Construct one exact token window.
         * @throws std::invalid_argument when @p required_tokens is zero.
         */
        explicit MoEOverlayMaintenanceBoundaryGate(
            std::uint64_t required_tokens);

        /**
         * @brief Publish completed inference tokens without blocking.
         * @param completed_tokens Exact logical tokens retired at the boundary.
         */
        void notify(
            MoEOverlayInferencePhase phase,
            std::uint64_t completed_tokens) noexcept;

        /** @return Whether at least one complete token window is available. */
        [[nodiscard]] bool ready() const noexcept;

        /**
         * @brief Test whether one exact inference phase has a complete window.
         *
         * Followers use this after observing the authority-authenticated
         * transaction ticket. They must never choose between simultaneously
         * ready prefill and decode windows themselves.
         *
         * @param phase Authority-selected inference phase.
         * @return Whether that phase alone has reached the current threshold.
         */
        [[nodiscard]] bool readyForPhase(
            MoEOverlayInferencePhase phase) const noexcept;

        /**
         * @brief Consume accumulated boundaries only after external admission.
         *
         * Economy certification uses @p admission_open to preserve every
         * retired token while placement is not yet economical. Closing this
         * gate never clears or rewrites either phase counter.
         *
         * @param admission_open Whether the controller may begin a transaction.
         * @return Coalesced boundary count, or zero while closed/partial.
         */
        [[nodiscard]] ReadyWindow consumeReady(
            bool admission_open = true) noexcept;

        /**
         * @brief Consume only the phase named by an authority transaction.
         *
         * A ready window for the other phase remains pending. This makes the
         * continuation authority the only phase scheduler while preserving
         * every phase-pure histogram boundary on follower ranks.
         *
         * @param phase Exact authority-selected inference phase.
         * @param admission_open Whether the already-authenticated transaction
         *        may consume its local sideband evidence.
         * @return Coalesced boundary count for @p phase, or an empty record.
         */
        [[nodiscard]] ReadyWindow consumeReadyForPhase(
            MoEOverlayInferencePhase phase,
            bool admission_open = true) noexcept;

        /**
         * @brief Increase the recurring admission window after one transaction.
         *
         * Notifications that arrived while a maintenance transaction was in
         * flight remain pending, but they are evaluated against the new window.
         * This bounds maintenance duty without discarding cumulative routing
         * evidence: device histograms retain every observation independently of
         * this host-side retained-graph submission cadence.
         *
         * @param maximum_tokens Inclusive adaptive ceiling; zero disables growth.
         * @param growth_factor Multiplicative growth; values at most one disable it.
         * @return Effective required-token window after the update.
         */
        [[nodiscard]] std::uint64_t growRequiredTokens(
            std::uint64_t maximum_tokens,
            double growth_factor) noexcept;

        /**
         * @brief Observe phase counters without consuming maintenance admission.
         *
         * Device service certification uses this passive snapshot to avoid
         * republishing unchanged cumulative GPU telemetry while economy is not
         * ready. Values are advisory wake identities only; placement continues
         * to consume through @ref consumeReady.
         */
        [[nodiscard]] std::array<std::uint64_t, 2> pendingTokens() const
            noexcept;

        /** @return Immutable minimum boundary count for one transaction. */
        [[nodiscard]] std::uint64_t requiredTokens() const noexcept
        {
            return required_tokens_.load(std::memory_order_acquire);
        }

    private:
        /** Worker-authored adaptive threshold read by inference notifiers. */
        std::atomic<std::uint64_t> required_tokens_;
        /** Independent counters prevent prefill and decode demand from mixing. */
        std::array<std::atomic<std::uint64_t>, 2> pending_tokens_{};
    };

    /**
     * @brief Own captured controller transactions on dedicated GPU streams.
     *
     * Construction performs setup-only allocation and native graph capture.
     * Replay allocates and copies nothing. A setup certification may wait for
     * one terminal event and copy one fixed controller record for validation;
     * ordinary inference and later background maintenance never wait on the
     * inference stream or expose policy state to the host.
     */
    class MoEOverlayDeviceControllerGraphService final
    {
    public:
        /** Public compatibility name for the topology-wide typed phase. */
        using InferencePhase = MoEOverlayInferencePhase;

        /** Complete retained transaction family selected at setup. */
        enum class ExecutionMode
        {
            Static,  ///< Captured zero-movement certification only.
            Dynamic, ///< Device-authored durable placement and physical RCU.
        };

        /** Immutable topology, control fabric, and rank identity. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx;
            std::shared_ptr<const MoEOverlayDeviceControllerTopology> topology;
            /** Empty only on a world rank with no participant in this cell. */
            std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric> fabric;
            /** Exact model-owned runtime table for every local participant. */
            std::vector<MoEOverlayDeviceControllerRuntimeBinding>
                runtime_bindings;
            /** Exact retained graph family; no runtime fallback is permitted. */
            ExecutionMode execution_mode = ExecutionMode::Static;
            /**
             * Decode-token boundaries required before one Dynamic observation.
             * This is distinct from the routed-activation evidence floor: one
             * token contributes top-k samples at every routed MoE layer.
             */
            std::uint64_t maintenance_window_tokens = 1u;
            /** Adaptive cadence ceiling; zero preserves the initial window. */
            std::uint64_t maintenance_max_window_tokens = 0u;
            /** Recurring cadence multiplier; values at most one disable growth. */
            double maintenance_window_growth_factor = 1.0;
            /** Required physical-only transport authority in Dynamic mode. */
            std::shared_ptr<MoEOverlayPhysicalResidencyFabric>
                physical_fabric;
            /**
             * Asynchronous evidence owner. GPU service timing is imported only
             * from finite mapped snapshots; it may publish immutable costs but
             * placement and epochs remain device-owned.
             */
            std::shared_ptr<MoEOverlayEconomyCertificationController>
                economy_certification;
            /** Stable topology label attached to movement evidence. */
            std::string perf_device;
        };

        /**
         * @brief Capture every local participant controller graph.
         * @throws std::invalid_argument for incomplete topology or ownership.
         * @throws std::runtime_error when a backend cannot materialize a graph.
         */
        explicit MoEOverlayDeviceControllerGraphService(Config config);

        /**
         * @brief Drain topology-wide Dynamic work and release GPU resources.
         *
         * This is the explicit collective terminal transition. Every inference
         * producer first leaves admission while controller workers remain live;
         * all admitted transactions then reach a terminal durable epoch before
         * any rank releases graphs, events, or policy storage. Completed totals
         * and the authoritative movement ledger remain readable afterward.
         * Repeated calls are no-ops.
         *
         * The coordinated runner must publish its worker-loop shutdown command
         * before entering this transition.
         */
        void stopAndDrain() noexcept;

        /** @brief Invoke @ref stopAndDrain when ownership was not sealed explicitly. */
        ~MoEOverlayDeviceControllerGraphService();

        MoEOverlayDeviceControllerGraphService(
            const MoEOverlayDeviceControllerGraphService &) = delete;
        MoEOverlayDeviceControllerGraphService &operator=(
            const MoEOverlayDeviceControllerGraphService &) = delete;

        /**
         * @brief Execute and validate one captured zero-movement transaction.
         *
         * All ranks enter a setup-only barrier before launch. Each process
         * submits all local root graphs before observing a terminal event, so
         * independent domains cannot be serialized by host launch order.
         * The copied terminal record is diagnostic evidence only; it is never
         * retained as a policy or epoch mirror.
         *
         * @param error Optional diagnostic populated on failure.
         * @return True only when the device authority completes StaticCheck
         *         without changing the durable epoch or publishing movement.
         */
        [[nodiscard]] bool certifyStaticNoMovement(
            std::string *error = nullptr);

        /**
         * @brief Wake the Dynamic device authority with committed token progress.
         *
         * This operation only increments a coalescing notification counter and
         * wakes the background transport follower. It performs no device call,
         * allocation, MPI operation, or wait on the inference thread.
         *
         * @param phase Typed source of the committed progress.
         * @param completed_tokens Exact logical tokens retired by this boundary.
         * @return False only when this service is not Dynamic or has failed.
         */
        [[nodiscard]] bool notifyInferenceProgress(
            InferencePhase phase,
            std::uint64_t completed_tokens) noexcept;

        /** @return Whether the background device/physical protocol is healthy. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return First fatal background diagnostic, or an empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /**
         * @brief Report whether immutable economics reached device policy.
         *
         * Dynamic calibration and publication remain background-owned.  This
         * passive snapshot is intended for setup/performance lifecycle gates,
         * never for inference admission or placement decisions.
         */
        [[nodiscard]] InferenceMeasurementReadiness
        measurementReadiness() const;

        /**
         * @brief Observe device-owned economy activation and durable movement.
         *
         * The mapped words are device-authored lifecycle publications; this
         * method does not download policy state or consult PerfStats.
         */
        [[nodiscard]] MoEOptimizationStatus optimizationStatus() const;

        /**
         * @brief Snapshot exact device-authored edges completed by the follower.
         * @return Typed ledger independent of PerfStats filtering or resets.
         */
        [[nodiscard]] MoEOptimizationMovementLedger movementLedger() const;

        /** @return Number of retained participant graphs owned by this rank. */
        [[nodiscard]] std::size_t localGraphCount() const noexcept;

        /** @return Whether this rank retained the sole leader graph. */
        [[nodiscard]] bool ownsLeaderGraph() const noexcept;

    private:
        struct Endpoint;
        struct DynamicWorker;

        /** One retained graph for each bounded device-owned Dynamic epoch. */
        enum class DynamicGraphEpoch
        {
            ServiceTelemetrySnapshot,
            RebaseHistograms,
            BeginPrefillDecision,
            BeginDecodeDecision,
            SnapshotPrefillDemand,
            SnapshotDecodeDemand,
            PublishGroupSnapshot,
            AuthorPrefillDecision,
            AuthorDecodeDecision,
            Publish,
            Retire,
            Complete,
        };

        /** Capture one immutable role-specific action chain. */
        void materializeEndpoint(Endpoint &endpoint);
        /** Submit every local graph before any terminal observation. */
        [[nodiscard]] bool launchAll(std::string *error);
        /** Wait for and validate each graph's fixed terminal record. */
        [[nodiscard]] bool validateAllStaticTerminals(std::string *error);
        /** Release one endpoint on the exact owning device worker. */
        void releaseEndpoint(Endpoint &endpoint) noexcept;

        /**
         * Submit one bounded epoch on every local participant.
         * Demand-snapshot and telemetry epochs first join their exact
         * inference-boundary event. Participant snapshots and group-root
         * publication are separate epochs: the host scheduler admits the
         * latter only after every participant receipt is acquire-visible.
         * Begin, group publication, and author epochs are therefore selected
         * by monotonic mapped lifecycle receipts and contain no live peer
         * dependency; every later epoch is selected by an authenticated
         * physical or device lifecycle receipt.
         */
        [[nodiscard]] bool launchDynamicEpoch(
            DynamicGraphEpoch epoch,
            std::string *error);
        /** Poll exact events for only the participants selected by one phase. */
        [[nodiscard]] bool dynamicTerminalsReady(
            bool *ready,
            std::string *error) noexcept;
        /** Mark participant inboxes reusable after one complete movement wave. */
        [[nodiscard]] bool finishDynamicInboxes(std::string *error) noexcept;
        /** Latch the first background failure and release mapped wait kernels. */
        void failDynamic(std::string message) noexcept;

        Config config_;
        std::vector<std::unique_ptr<Endpoint>> endpoints_;
        std::unique_ptr<DynamicWorker> dynamic_worker_;
        /** Successful terminal snapshot retained after the worker is released. */
        MoEOptimizationMovementTotals terminal_movement_totals_;
        MoEOptimizationMovementLedger terminal_movement_ledger_;
        std::uint64_t terminal_published_movement_waves_ = 0u;
        bool stopped_ = false;
        bool static_certified_ = false;
    };
} // namespace llaminar2
