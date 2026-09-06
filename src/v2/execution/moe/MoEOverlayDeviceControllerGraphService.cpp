/**
 * @file MoEOverlayDeviceControllerGraphService.cpp
 * @brief Finite captured phases for the all-GPU overlay controller.
 *
 * Dynamic transactions remain device-authored: GPUs own snapshot reduction,
 * placement policy, immutable commands, RCU selectors, admission, and durable
 * epochs. The host-side worker is only a node-local phase submitter and
 * physical byte-transport follower. It polls const mapped lifecycle records so
 * every retained CUDA/HIP graph is launched after its external prerequisites
 * are monotonic facts. A graph may perform a bounded mapped-memory fan-in among
 * participant graphs that were all pre-submitted for the same epoch; it never
 * stays resident awaiting a later transfer, reader drain, host publication, or
 * graph submission. This distinction is required on heterogeneous schedulers,
 * where an open-ended maintenance wait can starve independent inference.
 */

#include "MoEOverlayDeviceControllerGraphService.h"
#include "MoEOverlayWorkerDrainProtocol.h"

#include "MoEOverlayDeviceControllerKernels.h"
#include "MoEOverlayEconomyCertificationController.h"
#include "MoEOverlayDevicePhysicalMovement.h"
#include "MoEOverlayDevicePreparedArrivalInbox.h"
#include "MoEOverlayDeviceTransportProtocol.h"
#include "MoEOverlayDeviceControllerTopology.h"
#include "MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "DeviceMoERebalanceController.h"

#include "backends/BackendManager.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/GPUDeviceContextPool.h"
#include "interfaces/IMPIContext.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>
#include <optional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Append one naturally aligned typed array to a byte layout.
         *
         * Static certification copies several non-contiguous device records
         * into one pinned allocation. Explicit offsets keep the transfer batch
         * allocation-free after submission without relying on packed structs.
         */
        template <typename T>
        std::size_t appendAlignedArray(
            std::size_t &cursor,
            std::size_t count)
        {
            constexpr std::size_t alignment = alignof(T);
            static_assert(
                (alignment & (alignment - 1u)) == 0u,
                "C++ object alignment must be a power of two");
            if (cursor >
                std::numeric_limits<std::size_t>::max() - (alignment - 1u))
            {
                throw std::overflow_error(
                    "Static controller validation alignment overflowed size_t");
            }
            cursor = (cursor + alignment - 1u) & ~(alignment - 1u);
            if (count >
                (std::numeric_limits<std::size_t>::max() - cursor) /
                    sizeof(T))
            {
                throw std::overflow_error(
                    "Static controller validation geometry overflowed size_t");
            }
            const std::size_t offset = cursor;
            cursor += count * sizeof(T);
            return offset;
        }

        /** Byte offsets for one root's setup-only terminal snapshot. */
        struct StaticTerminalValidationLayout
        {
            explicit StaticTerminalValidationLayout(
                std::size_t participant_count)
            {
                controller = appendAlignedArray<
                    MoEOverlayDeviceControllerSharedHeader>(bytes, 1u);
                command = appendAlignedArray<
                    MoEOverlayDeviceControllerCommandHeader>(bytes, 1u);
                group = appendAlignedArray<
                    MoEOverlayDeviceControllerGroupRecord>(bytes, 1u);
                participants = appendAlignedArray<
                    MoEOverlayDeviceControllerParticipantRecord>(
                    bytes, participant_count);
            }

            std::size_t controller = 0u;
            std::size_t command = 0u;
            std::size_t group = 0u;
            std::size_t participants = 0u;
            std::size_t bytes = 0u;
        };

        /** Frees one setup-only pinned allocation on its exact backend. */
        struct PinnedValidationDeleter
        {
            IBackend *backend = nullptr;
            int ordinal = -1;

            void operator()(std::byte *allocation) const noexcept
            {
                if (backend && allocation)
                    backend->freePinned(allocation, ordinal);
            }
        };

        /** @return Stable diagnostic spelling of a controller-authored axis. */
        const char *movementAxisName(MoEOptimizationMovementAxis axis)
        {
            switch (axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                return "tier_residency";
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                return "participant_placement";
            case MoEOptimizationMovementAxis::Combined:
                return "combined";
            }
            return "unknown";
        }
    } // namespace

    MoEOverlayMaintenanceBoundaryGate::MoEOverlayMaintenanceBoundaryGate(
        std::uint64_t required_tokens)
        : required_tokens_(required_tokens)
    {
        if (required_tokens_ == 0u)
        {
            throw std::invalid_argument(
                "ExpertOverlay maintenance boundary window must be positive");
        }
    }

    void MoEOverlayMaintenanceBoundaryGate::notify(
        MoEOverlayInferencePhase phase,
        std::uint64_t completed_tokens) noexcept
    {
        const auto phase_index = static_cast<std::size_t>(phase);
        if (completed_tokens == 0u ||
            phase_index >= pending_tokens_.size())
            return;

        auto &pending = pending_tokens_[phase_index];
        std::uint64_t observed = pending.load(
            std::memory_order_relaxed);
        while (observed != std::numeric_limits<std::uint64_t>::max())
        {
            const std::uint64_t available =
                std::numeric_limits<std::uint64_t>::max() - observed;
            const std::uint64_t next =
                completed_tokens >= available
                    ? std::numeric_limits<std::uint64_t>::max()
                    : observed + completed_tokens;
            if (pending.compare_exchange_weak(
                   observed,
                   next,
                   std::memory_order_release,
                   std::memory_order_relaxed))
            {
                return;
            }
            // compare_exchange refreshes observed before the next attempt.
        }
    }

    bool MoEOverlayMaintenanceBoundaryGate::ready() const noexcept
    {
        const std::uint64_t required_tokens =
            required_tokens_.load(std::memory_order_acquire);
        return std::any_of(
            pending_tokens_.begin(),
            pending_tokens_.end(),
            [required_tokens](const auto &pending)
            {
                return pending.load(std::memory_order_acquire) >=
                    required_tokens;
            });
    }

    bool MoEOverlayMaintenanceBoundaryGate::readyForPhase(
        MoEOverlayInferencePhase phase) const noexcept
    {
        const auto phase_index = static_cast<std::size_t>(phase);
        if (phase_index >= pending_tokens_.size())
            return false;
        return pending_tokens_[phase_index].load(
                   std::memory_order_acquire) >=
            required_tokens_.load(std::memory_order_acquire);
    }

    MoEOverlayMaintenanceBoundaryGate::ReadyWindow
    MoEOverlayMaintenanceBoundaryGate::consumeReadyForPhase(
        MoEOverlayInferencePhase phase,
        bool admission_open) noexcept
    {
        const auto phase_index = static_cast<std::size_t>(phase);
        if (!admission_open || phase_index >= pending_tokens_.size())
            return {};

        const std::uint64_t required_tokens =
            required_tokens_.load(std::memory_order_acquire);
        auto &pending = pending_tokens_[phase_index];
        std::uint64_t observed = pending.load(std::memory_order_acquire);
        while (observed >= required_tokens)
        {
            if (pending.compare_exchange_weak(
                    observed,
                    0u,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                return {
                    .phase = phase,
                    .completed_tokens = observed,
                };
            }
        }
        return {};
    }

    MoEOverlayMaintenanceBoundaryGate::ReadyWindow
    MoEOverlayMaintenanceBoundaryGate::consumeReady(
        bool admission_open) noexcept
    {
        if (!admission_open)
            return {};
        /* The sole authority resolves simultaneous phases deterministically.
         * Followers consume through consumeReadyForPhase() only after the
         * resulting transaction ticket has named this choice. */
        constexpr std::array<MoEOverlayInferencePhase, 2> kPhaseOrder{
            MoEOverlayInferencePhase::Prefill,
            MoEOverlayInferencePhase::Decode};
        for (const auto phase : kPhaseOrder)
        {
            if (auto window = consumeReadyForPhase(phase))
                return window;
        }
        return {};
    }

    std::uint64_t MoEOverlayMaintenanceBoundaryGate::advanceAfterReceipt(
        std::uint64_t maximum_tokens,
        double growth_factor,
        MoEOverlayMaintenanceCadenceReceipt receipt) noexcept
    {
        const std::uint64_t current =
            required_tokens_.load(std::memory_order_acquire);
        if (!receipt.permitsCooldown() || maximum_tokens == 0u ||
            maximum_tokens <= current ||
            !std::isfinite(growth_factor) || growth_factor <= 1.0)
        {
            return current;
        }

        /* Long double keeps the multiplication monotonic across the complete
         * uint64 cadence range. Ceil guarantees progress even for a factor
         * only slightly above one; the configured ceiling prevents overflow. */
        const long double scaled =
            static_cast<long double>(current) *
            static_cast<long double>(growth_factor);
        const std::uint64_t grown =
            scaled >= static_cast<long double>(maximum_tokens)
                ? maximum_tokens
                : std::max<std::uint64_t>(
                      current + 1u,
                      static_cast<std::uint64_t>(std::ceil(scaled)));
        required_tokens_.store(grown, std::memory_order_release);
        return grown;
    }

    std::array<std::uint64_t, 2>
    MoEOverlayMaintenanceBoundaryGate::pendingTokens() const noexcept
    {
        return {
            pending_tokens_[0].load(std::memory_order_acquire),
            pending_tokens_[1].load(std::memory_order_acquire),
        };
    }

    /** Model-lifetime resources for one local participant transaction graph. */
    struct MoEOverlayDeviceControllerGraphService::Endpoint
    {
        MoEOverlayDeviceControllerParticipantBinding binding;
        MoEOverlayDeviceControllerRuntimeBinding runtime_binding;
        DeviceMoERebalanceConfig snapshot_config;
        IWorkerGPUContext *worker = nullptr;
        IBackend *backend = nullptr;
        std::unique_ptr<IMoEKernel> kernel;
        std::unique_ptr<MoEOverlayDevicePreparedArrivalInbox> arrival_inbox;
        std::unique_ptr<IGPUGraphCapture> static_graph;
        std::unique_ptr<IGPUGraphCapture>
            dynamic_service_telemetry_snapshot_graph;
        /** Discards calibration demand by advancing both device baselines. */
        std::unique_ptr<IGPUGraphCapture> dynamic_histogram_rebase_graph;
        /** Sole-leader finite transaction-open nodes, specialized by phase. */
        std::unique_ptr<IGPUGraphCapture> dynamic_begin_prefill_graph;
        std::unique_ptr<IGPUGraphCapture> dynamic_begin_decode_graph;
        /** Sole-leader terminal restoration transaction-open graph. */
        std::unique_ptr<IGPUGraphCapture>
            prepared_context_restore_begin_graph;
        /** Participant-local demand publication with no peer dependency. */
        /** One phase-complete snapshot graph shared by both scheduler boundaries. */
        std::unique_ptr<IGPUGraphCapture> dynamic_snapshot_graph;
        /** Participant snapshot of live ownership for terminal restoration. */
        std::unique_ptr<IGPUGraphCapture>
            prepared_context_restore_snapshot_graph;
        /** Group-root reduction admitted after every participant receipt. */
        std::unique_ptr<IGPUGraphCapture> dynamic_group_snapshot_graph;
        /** Sole-leader policy nodes launched only after every group receipt. */
        std::unique_ptr<IGPUGraphCapture> dynamic_author_prefill_graph;
        std::unique_ptr<IGPUGraphCapture> dynamic_author_decode_graph;
        /** Sole-leader comparison against the immutable prepared owner table. */
        std::unique_ptr<IGPUGraphCapture>
            prepared_context_restore_author_graph;
        /** Build the complete inactive E+1 bank on every participant. */
        std::unique_ptr<IGPUGraphCapture> dynamic_prepare_candidate_graph;
        /** Group-root receipt after every local E+1 bank is complete. */
        std::unique_ptr<IGPUGraphCapture> dynamic_acknowledge_prepared_graph;
        /** Sole-leader commit after every group preparation receipt. */
        std::unique_ptr<IGPUGraphCapture> dynamic_begin_commit_graph;
        /** Participant-local RCU selector publication after commit. */
        std::unique_ptr<IGPUGraphCapture> dynamic_publish_candidate_graph;
        /** Group-root receipt after every local selector publication. */
        std::unique_ptr<IGPUGraphCapture> dynamic_acknowledge_published_graph;
        /** Sole-leader E+1 admission and E retirement opening. */
        std::unique_ptr<IGPUGraphCapture> dynamic_publish_admission_graph;
        /** Bounded participant-local old-bank reader-drain probe. */
        std::unique_ptr<IGPUGraphCapture>
            dynamic_retirement_readiness_graph;
        /** Reader-ticket-selected participant-local bank reclamation. */
        std::unique_ptr<IGPUGraphCapture> dynamic_retire_graph;
        /** Physical-retirement-selected topology terminal publication. */
        std::unique_ptr<IGPUGraphCapture> dynamic_complete_graph;
        void *stream = nullptr;
        void *terminal_event = nullptr;
        MoEOverlayDeviceControllerPolicyResult *policy_result = nullptr;
        /** Phase-pure decode, prefill, and verifier cumulative baselines. */
        std::uint64_t *histogram_phase_baselines = nullptr;
        std::size_t histogram_baseline_words = 0u;
        bool in_flight = false;
        bool terminal_ready = false;
    };

    /** Process-local follower for one device-owned physical transaction. */
    struct MoEOverlayDeviceControllerGraphService::DynamicWorker
    {
        /** One complete retained transaction objective. */
        enum class TransactionObjective : std::uint8_t
        {
            DynamicPrefill,
            DynamicDecode,
            PreparedContextRestore,
        };

        /** Device-authored terminal identity returned by @ref runOne. */
        struct TransactionResult
        {
            MoEOverlayDeviceControllerTransactionKind kind =
                MoEOverlayDeviceControllerTransactionKind::Invalid;
            std::uint64_t transaction = 0u;
            std::uint64_t durable_epoch = 0u;
            std::uint32_t command_count = 0u;
            /** Device-authored demand consumed by this exact transaction. */
            std::uint64_t snapshot_observations = 0u;

            /** @return Whether this transaction moved durable expert bytes. */
            [[nodiscard]] bool movedWeights() const noexcept
            {
                return command_count != 0u;
            }

            /**
             * @return Typed cadence projection of this device-authored result.
             */
            [[nodiscard]] MoEOverlayMaintenanceCadenceReceipt cadenceReceipt()
                const noexcept
            {
                return MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        snapshot_observations, command_count);
            }
        };

        /** Bind the worker to its sole service and immutable token cadence. */
        DynamicWorker(
            MoEOverlayDeviceControllerGraphService *service,
            std::uint64_t maintenance_window_tokens,
            std::uint64_t maintenance_max_window_tokens,
            double maintenance_window_growth_factor)
            : owner(service),
              boundary_gate(maintenance_window_tokens),
              maximum_window_tokens(maintenance_max_window_tokens),
              window_growth_factor(maintenance_window_growth_factor)
        {
        }

        MoEOverlayDeviceControllerGraphService *owner = nullptr;
        std::vector<std::unique_ptr<MoEOverlayDeviceTransportProtocol>>
            protocols;
        std::jthread thread;
        std::mutex wake_mutex;
        std::condition_variable wake_cv;
        MoEOverlayMaintenanceBoundaryGate boundary_gate;
        /** Immutable adaptive ceiling selected by production configuration. */
        const std::uint64_t maximum_window_tokens = 0u;
        /** Immutable recurring growth multiplier selected at setup. */
        const double window_growth_factor = 1.0;
        std::atomic<bool> healthy{true};
        /** Passive public projection of this worker's exact background work. */
        std::atomic<MoEOptimizationActivityState> activity{
            MoEOptimizationActivityState::LearningEconomy};
        /** Monotonic owner/worker proof that shutdown has no admitted work. */
        MoEOverlayWorkerDrainProtocol drain;
        /** Owner-selected terminal objective, published before @ref drain. */
        std::atomic<MoEOverlayDeviceControllerDrainIntent> drain_intent{
            MoEOverlayDeviceControllerDrainIntent::ReleaseResources};
        /** Exact device-certified terminal durable epoch. */
        std::atomic<std::uint64_t> restored_durable_epoch{0u};
        /** Number of physical restoration waves, excluding the final proof. */
        std::atomic<std::uint64_t> restoration_movement_waves{0u};
        /** Release-published only after a zero-command restoration receipt. */
        std::atomic<bool> prepared_context_certified{false};
        mutable std::mutex failure_mutex;
        std::string failure;
        std::uint64_t last_transaction = 0u;
        /** Even outside publication; odd while the worker updates all totals. */
        std::atomic<std::uint64_t> movement_publication_sequence{0u};
        /** Host-observed totals for physically completed device decisions. */
        std::atomic<std::uint64_t> movement_transactions{0u};
        std::atomic<std::uint64_t> movement_commands{0u};
        std::atomic<std::uint64_t> movement_physical_bytes{0u};
        std::atomic<std::uint64_t> movement_promotions{0u};
        std::atomic<std::uint64_t> movement_demotions{0u};
        std::atomic<std::uint64_t> movement_same_priority{0u};
        /** Exact completed edges retained independently of optional telemetry. */
        mutable std::mutex movement_ledger_mutex;
        std::vector<MoEOptimizationMovementEdge> movement_ledger;
        /** Sole leader's immutable admitting economics for completed waves. */
        std::vector<MoEOptimizationMovementEconomy> movement_economy;
        bool economy_ready = false;
        /** True after the retained local rebase graph has been submitted. */
        bool histogram_rebase_started = false;
        /** True after every local participant rebase terminal is acquired. */
        bool local_histograms_rebased = false;
        /** Prevent repeated lifecycle summaries while one drain stays idle. */
        bool drain_idle_diagnostic_published = false;
        /** Last inference notification identity represented by an import. */
        std::array<std::uint64_t, 2> service_snapshot_tokens{};
        /**
         * First-observation latch for each local participant and service phase.
         *
         * Cumulative snapshots are imported after every maintenance boundary,
         * but production diagnostics need only one proof that decode, prefill,
         * and (when active) grouped-verifier work reached each participant.
         */
        std::vector<std::array<bool, kExpertHistogramProductionSourceCount>>
            service_phase_observed;

        /** @return One coherent snapshot of all completed movement totals. */
        [[nodiscard]] MoEOptimizationMovementTotals
        completedMovement() const noexcept
        {
            for (;;)
            {
                const auto before = movement_publication_sequence.load(
                    std::memory_order_acquire);
                if ((before & 1u) != 0u)
                    continue;
                const MoEOptimizationMovementTotals totals{
                    .transactions = movement_transactions.load(
                        std::memory_order_relaxed),
                    .commands = movement_commands.load(
                        std::memory_order_relaxed),
                    .physical_bytes = movement_physical_bytes.load(
                        std::memory_order_relaxed),
                    .promotions = movement_promotions.load(
                        std::memory_order_relaxed),
                    .demotions = movement_demotions.load(
                        std::memory_order_relaxed),
                    .same_priority_moves = movement_same_priority.load(
                        std::memory_order_relaxed),
                };
                const auto after = movement_publication_sequence.load(
                    std::memory_order_acquire);
                if (before == after)
                    return totals;
            }
        }

        /** @return Race-safe complete movement identities for this runner. */
        [[nodiscard]] MoEOptimizationMovementLedger
        completedMovementLedger() const
        {
            std::lock_guard<std::mutex> lock(movement_ledger_mutex);
            return {
                .edges = movement_ledger,
                .discarded_edges = 0u,
                .economy = movement_economy,
                .discarded_economy_records = 0u,
            };
        }

        /** Run coalesced complete device/physical transactions until stopped. */
        void run(std::stop_token stop_token) noexcept;

        /** Execute one complete device-authored transaction. */
        [[nodiscard]] bool runOne(
            TransactionObjective objective,
            TransactionResult *result,
            std::string *error) noexcept;

        /** Run bounded restoration waves through a final zero-command proof. */
        [[nodiscard]] bool restorePreparedContext(
            std::string *error) noexcept;

        /**
         * @brief Observe a newer authority-authenticated transaction ticket.
         *
         * Every process-local group root must see the same transaction and
         * phase. The result is scheduling metadata only; placement and policy
         * remain device-owned in the shared controller fabric.
         *
         * @param transaction Receives zero when no newer transaction exists.
         * @param phase Receives the typed snapshot plane for a newer ticket.
         * @param error Receives a divergent-ticket diagnostic.
         * @return False only when local group roots disagree.
         */
        [[nodiscard]] bool observeAuthorityTransaction(
            std::uint64_t *transaction,
            MoEOverlayDeviceControllerTransactionKind *kind,
            MoEOverlayDeviceDemandPhase *phase,
            std::string *error) const noexcept;

        /** Advance host-owned measurements and publish immutable device costs. */
        [[nodiscard]] bool progressEconomy(std::string *error) noexcept;

        /** Publish and import one coherent cumulative GPU service snapshot. */
        [[nodiscard]] bool snapshotServiceTelemetry(
            std::string *error) noexcept;

        /** Drain an incomplete calibration before releasing transfer owners. */
        void stopEconomy() noexcept;

        /** Abort every retained projection before destroying its lane owner. */
        void abortPrepared(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            MoEOverlayParticipantPreparedTransfers &prepared) noexcept;
    };

    namespace
    {
        /** Set an optional diagnostic without requiring callers to allocate it. */
        bool fail(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @return Stable auxiliary-stream identity for one controller root. */
        std::string streamName(
            const MoEOverlayDeviceControllerParticipantBinding &binding)
        {
            return "moe_overlay_device_controller_group_" +
                   std::to_string(binding.group_id) + "_participant_" +
                   std::to_string(binding.participant_id);
        }

        /** @return Whether two acquired group views name identical command bytes. */
        bool sameCommand(
            const MoEOverlayDeviceTransportCommandBatch &left,
            const MoEOverlayDeviceTransportCommandBatch &right) noexcept
        {
            if (left.participant_count != right.participant_count ||
                left.num_layers != right.num_layers ||
                left.num_experts != right.num_experts ||
                std::memcmp(
                    &left.header, &right.header, sizeof(left.header)) != 0 ||
                left.entries.size() != right.entries.size())
            {
                return false;
            }
            return left.entries.empty() ||
                std::memcmp(
                    left.entries.data(),
                    right.entries.data(),
                    left.entries.size() *
                        sizeof(MoEOverlayDeviceMovementCommand)) == 0;
        }

        /** @return Standard protocol deadline used for one bounded epoch phase. */
        std::chrono::steady_clock::time_point protocolDeadline()
        {
            return std::chrono::steady_clock::now() +
                std::chrono::seconds(30);
        }

        /** Yield briefly without turning event progress into busy host traffic. */
        void pollPause()
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

    } // namespace

    MoEOverlayDeviceControllerGraphService::
        MoEOverlayDeviceControllerGraphService(Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx || !config_.topology ||
            !config_.topology->valid())
        {
            throw std::invalid_argument(
                "Device controller graph service requires MPI and a frozen topology");
        }

        const int local_rank = config_.mpi_ctx->rank();
        const bool owns_participant = std::any_of(
            config_.topology->participants.begin(),
            config_.topology->participants.end(),
            [local_rank](const auto &participant)
            {
                return participant.world_rank == local_rank;
            });
        if (owns_participant && !config_.fabric)
        {
            throw std::invalid_argument(
                "A rank owning a device-controller participant has no local mapped fabric");
        }
        if (!owns_participant && !config_.runtime_bindings.empty())
        {
            throw std::invalid_argument(
                "A rank without device-controller participants supplied runtime bindings");
        }
        if (config_.execution_mode == ExecutionMode::Dynamic &&
            (config_.maintenance_window_tokens == 0u ||
             !std::isfinite(config_.maintenance_window_growth_factor) ||
             config_.maintenance_window_growth_factor <= 0.0 ||
             (owns_participant && !config_.physical_fabric) ||
             (owns_participant && !config_.economy_certification &&
              !config_.fabric->economyProfilesPublished())))
        {
            throw std::invalid_argument(
                "Dynamic device controller graph service requires a positive token window, one process-local physical fabric, and certified or asynchronously certifiable economy evidence");
        }

        std::sort(
            config_.runtime_bindings.begin(),
            config_.runtime_bindings.end(),
            [](const auto &left, const auto &right)
            {
                return left.overlay_participant_id <
                       right.overlay_participant_id;
            });
        const auto &local_ids = config_.fabric
                                    ? config_.fabric->localParticipantIds()
                                    : std::vector<int>{};
        if (owns_participant &&
            config_.runtime_bindings.size() != local_ids.size())
        {
            throw std::invalid_argument(
                "Device controller graph service requires one runtime binding for every local participant");
        }

        for (std::size_t local_index = 0u;
             local_index < config_.runtime_bindings.size();
             ++local_index)
        {
            const auto &runtime = config_.runtime_bindings[local_index];
            if (!runtime.valid() ||
                !runtime.initialRuntimePublicationValid() ||
                runtime.overlay_participant_id != local_ids[local_index] ||
                runtime.layer_count != config_.fabric->layout().header.num_layers ||
                runtime.expert_count != config_.fabric->layout().header.num_experts)
            {
                throw std::logic_error(
                    "Device controller runtime bindings diverged from the mapped fabric geometry or local participant order");
            }
            if (config_.execution_mode == ExecutionMode::Dynamic &&
                (!runtime.backgroundPublicationValid() ||
                 !runtime.serviceTelemetryValid()))
            {
                throw std::logic_error(
                    "Dynamic device controller requires runtime RCU publication, device-local service telemetry, and an exact inference-boundary owner for every participant");
            }
            const auto topology_it = std::find_if(
                config_.topology->participants.begin(),
                config_.topology->participants.end(),
                [&runtime](const auto &participant)
                {
                    return participant.participant_id ==
                           runtime.overlay_participant_id;
                });
            if (topology_it == config_.topology->participants.end() ||
                topology_it->world_rank != local_rank ||
                topology_it->device != runtime.device)
            {
                throw std::logic_error(
                    "Device controller runtime identity diverged from the frozen topology");
            }

            auto endpoint = std::make_unique<Endpoint>();
            endpoint->runtime_binding = runtime;
            endpoint->binding = config_.fabric->participantBinding(
                runtime.overlay_participant_id);
            if (!endpoint->binding.valid() ||
                endpoint->binding.participant_id !=
                    runtime.overlay_participant_id ||
                endpoint->binding.device != runtime.device)
            {
                throw std::logic_error(
                    "Device controller participant binding diverged from its model runtime source");
            }

            const auto group_it = std::find_if(
                config_.topology->groups.begin(),
                config_.topology->groups.end(),
                [&endpoint](const auto &group)
                {
                    return group.group_id == endpoint->binding.group_id;
                });
            if (group_it == config_.topology->groups.end())
            {
                throw std::logic_error(
                    "Device controller participant has no frozen topology group");
            }
            const bool expected_root =
                group_it->root_participant_id ==
                runtime.overlay_participant_id;
            if (endpoint->binding.group_root != expected_root ||
                endpoint->binding.authority_leader !=
                    (runtime.overlay_participant_id ==
                     config_.topology->leader_participant_id))
            {
                throw std::logic_error(
                    "Device controller graph resolved inconsistent participant roles");
            }

            endpoint->snapshot_config = DeviceMoERebalanceConfig{
                .num_layers = runtime.layer_count,
                .num_experts = runtime.expert_count,
                .top_k = runtime.top_k,
                .participant_id = runtime.domain_participant_id,
                .participant_count = runtime.domain_participant_count,
                .root_participant = 0u,
                .window_size_tokens = 1u,
                .layer_window_start = 0u,
                .layer_window_count = runtime.layer_count,
                .layer_wave_count = runtime.layer_count,
            };
            if (!validateDeviceMoERebalanceConfig(
                    endpoint->snapshot_config))
            {
                throw std::logic_error(
                    "Device controller could not derive a valid full-model snapshot configuration");
            }
            materializeEndpoint(*endpoint);
            endpoints_.push_back(std::move(endpoint));
        }

        for (const auto &participant : config_.topology->participants)
        {
            if (participant.world_rank != local_rank)
                continue;
            const bool found = std::any_of(
                endpoints_.begin(),
                endpoints_.end(),
                [&participant](const auto &endpoint)
                {
                    return endpoint &&
                           endpoint->binding.participant_id ==
                               participant.participant_id;
                });
            if (!found)
            {
                throw std::logic_error(
                    "Device controller graph omitted a local frozen participant");
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "captured_participant_graphs",
            static_cast<double>(endpoints_.size()),
            "model_setup",
            {},
            {{"world_rank", std::to_string(local_rank)},
             {"leader_graph", ownsLeaderGraph() ? "true" : "false"},
             {"stream", "dedicated_background"},
             {"execution_mode",
              config_.execution_mode == ExecutionMode::Dynamic
                  ? "dynamic"
                  : "static"},
             {"maintenance_window_tokens",
              std::to_string(config_.maintenance_window_tokens)},
             {"maintenance_max_window_tokens",
              std::to_string(config_.maintenance_max_window_tokens)},
             {"maintenance_window_growth_factor",
              std::to_string(
                  config_.maintenance_window_growth_factor)},
             {"imbalance_threshold_per_mille",
              std::to_string(
                  config_.fabric->layout().header
                      .dynamic_imbalance_threshold_per_mille)},
             {"minimum_improvement_per_mille",
              std::to_string(
                  config_.fabric->layout().header
                      .dynamic_minimum_improvement_per_mille)},
             {"maximum_cycles_per_layer",
              std::to_string(
                  config_.fabric->layout().header
                      .dynamic_maximum_cycles_per_layer)},
             {"maximum_commands_per_wave",
              std::to_string(
                  config_.fabric->layout().header
                      .dynamic_maximum_commands_per_wave)},
             {"host_policy_mirror", "false"}});

        if (config_.execution_mode == ExecutionMode::Dynamic &&
            !endpoints_.empty())
        {
            dynamic_worker_ = std::make_unique<DynamicWorker>(
                this,
                config_.maintenance_window_tokens,
                config_.maintenance_max_window_tokens,
                config_.maintenance_window_growth_factor);
            for (const auto &endpoint : endpoints_)
            {
                if (!endpoint || !endpoint->binding.group_root)
                    continue;
                dynamic_worker_->protocols.push_back(
                    std::make_unique<MoEOverlayDeviceTransportProtocol>(
                        config_.fabric->transportBinding(
                            endpoint->binding.group_id)));
            }
            if (dynamic_worker_->protocols.empty())
            {
                throw std::logic_error(
                    "Dynamic device controller rank owns participants but no group-root transport lane");
            }
            /*
             * Do not start the submitter here. The runner still has one lean
             * serving topology to capture after this controller family is
             * materialized. Any CUDA/HIP work submitted from this thread in
             * that interval can invalidate an unrelated stream capture.
             * start() is the sole, typed activation edge after all graph
             * families have crossed rank-wide setup consensus.
             */
        }
    }

    void MoEOverlayDeviceControllerGraphService::start()
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (activation_state_.load(std::memory_order_acquire) !=
            MoEOverlayDeviceControllerActivationState::Prepared)
        {
            throw std::logic_error(
                "Device controller service may start exactly once from Prepared");
        }
        if (config_.execution_mode == ExecutionMode::Static &&
            !static_certified_)
        {
            throw std::logic_error(
                "Static device controller service cannot start before its no-movement certification");
        }

        activation_state_.store(
            MoEOverlayDeviceControllerActivationState::Starting,
            std::memory_order_release);
        try
        {
            if (dynamic_worker_)
            {
                dynamic_worker_->thread = std::jthread(
                    [worker = dynamic_worker_.get()](std::stop_token token)
                    {
                        worker->run(token);
                    });

                /*
                 * The worker can report a fatal setup/protocol error as soon
                 * as it enters run(). Do not overwrite that terminal state
                 * with Running merely because thread construction returned.
                 */
                auto expected =
                    MoEOverlayDeviceControllerActivationState::Starting;
                if (!activation_state_.compare_exchange_strong(
                        expected,
                        MoEOverlayDeviceControllerActivationState::Running,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                {
                    dynamic_worker_->thread.request_stop();
                    dynamic_worker_->wake_cv.notify_all();
                    if (dynamic_worker_->thread.joinable())
                        dynamic_worker_->thread.join();
                    throw std::runtime_error(
                        "Device controller worker failed during activation: " +
                        failureMessage());
                }
            }
            else
            {
                /* Static and participant-empty ranks own no worker thread. */
                activation_state_.store(
                    MoEOverlayDeviceControllerActivationState::Running,
                    std::memory_order_release);
            }
        }
        catch (...)
        {
            activation_state_.store(
                MoEOverlayDeviceControllerActivationState::Failed,
                std::memory_order_release);
            throw;
        }
    }

    MoEOverlayDeviceControllerActivationState
    MoEOverlayDeviceControllerGraphService::state() const noexcept
    {
        return activation_state_.load(std::memory_order_acquire);
    }

    MoEOverlayDeviceControllerGraphService::
        ~MoEOverlayDeviceControllerGraphService()
    {
        (void)stopAndDrain();
    }

    MoEOverlayDeviceControllerDrainResult
    MoEOverlayDeviceControllerGraphService::stopAndDrain(
        MoEOverlayDeviceControllerDrainIntent intent) noexcept
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        const auto activation_state =
            activation_state_.load(std::memory_order_acquire);
        if (activation_state ==
            MoEOverlayDeviceControllerActivationState::Stopped)
            return terminal_drain_result_;

        terminal_drain_result_ = {
            .intent = intent,
            .succeeded = false,
        };

        /*
         * Setup rollback must never enter the distributed terminal protocol.
         * A Prepared service has submitted no background work, so releasing
         * its retained graphs is purely process-local. Prepared-context
         * restoration is intentionally not manufactured here: terminal model
         * reuse is admitted only after a successfully Running authority.
         */
        if (activation_state ==
                MoEOverlayDeviceControllerActivationState::Prepared ||
            (activation_state ==
                 MoEOverlayDeviceControllerActivationState::Failed &&
             dynamic_worker_ && !dynamic_worker_->thread.joinable()))
        {
            terminal_drain_result_.succeeded =
                intent == MoEOverlayDeviceControllerDrainIntent::
                              ReleaseResources;
            dynamic_worker_.reset();
            for (auto &endpoint : endpoints_)
            {
                if (endpoint)
                    releaseEndpoint(*endpoint);
            }
            endpoints_.clear();
            activation_state_.store(
                MoEOverlayDeviceControllerActivationState::Stopped,
                std::memory_order_release);
            return terminal_drain_result_;
        }

        if (activation_state !=
                MoEOverlayDeviceControllerActivationState::Running &&
            activation_state !=
                MoEOverlayDeviceControllerActivationState::Failed)
        {
            LOG_ERROR(
                "[MoEOverlayDeviceController] terminal drain reached an invalid activation state");
            std::terminate();
        }
        activation_state_.store(
            MoEOverlayDeviceControllerActivationState::Draining,
            std::memory_order_release);

        if (dynamic_worker_)
        {
            /*
             * Every rank first closes ordinary inference. Background workers
             * remain live across this barrier, allowing a lagging group to
             * join a transaction that a faster peer already admitted. A
             * local jthread cancellation is not a distributed protocol edge.
             */
            dynamic_worker_->drain_intent.store(
                intent, std::memory_order_release);
            config_.mpi_ctx->barrier();
            const auto drain_request = dynamic_worker_->drain.request();
            dynamic_worker_->wake_cv.notify_all();

            const auto wait_for_local_drain = [this, drain_request]()
            {
                const auto deadline = protocolDeadline();
                while (!dynamic_worker_->drain.acknowledged(drain_request) &&
                       dynamic_worker_->healthy.load(
                           std::memory_order_acquire) &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    std::unique_lock<std::mutex> lock(
                        dynamic_worker_->wake_mutex);
                    dynamic_worker_->wake_cv.wait_for(
                        lock, std::chrono::milliseconds(2));
                }
                return dynamic_worker_->drain.acknowledged(drain_request);
            };

            /*
             * Only the rank retaining the authority graph can close global
             * admission. Followers must remain runnable until that authority
             * has finished any ticket published just before the monotonic
             * drain request.
             * The first barrier publishes authority quiescence; only then may
             * followers certify that their last observed transaction drained.
             */
            const bool leader_rank = ownsLeaderGraph();
            bool local_drained = true;
            if (leader_rank)
                local_drained = wait_for_local_drain();
            config_.mpi_ctx->barrier();
            if (!leader_rank)
                local_drained = wait_for_local_drain();

            if (!local_drained)
            {
                LOG_ERROR(
                    "[MoEOverlayDeviceController] Dynamic shutdown could not "
                    "drain its admitted transaction: "
                    << failureMessage());
                std::terminate();
            }

            /* No rank may release its graphs while a peer is still draining. */
            config_.mpi_ctx->barrier();
            dynamic_worker_->thread.request_stop();
            dynamic_worker_->wake_cv.notify_all();
            if (dynamic_worker_->thread.joinable())
                dynamic_worker_->thread.join();

            /*
             * The worker is now immutable and every topology peer has crossed
             * the same terminal barrier. Retain its completed authority state
             * before releasing worker storage so runner-lifetime diagnostics
             * do not depend on an implementation object's lifetime.
             */
            terminal_movement_totals_ =
                dynamic_worker_->completedMovement();
            terminal_movement_ledger_ =
                dynamic_worker_->completedMovementLedger();
            const std::uint64_t restoration_waves =
                dynamic_worker_->restoration_movement_waves.load(
                    std::memory_order_acquire);
            const std::uint64_t all_durable_waves = config_.fabric
                ? config_.fabric->completedDurableMovementEpochs()
                : 0u;
            terminal_published_movement_waves_ =
                all_durable_waves >= restoration_waves
                    ? all_durable_waves - restoration_waves
                    : 0u;
            terminal_drain_result_ = {
                .intent = intent,
                .succeeded = dynamic_worker_->healthy.load(
                    std::memory_order_acquire),
                .prepared_context_certified =
                    dynamic_worker_->prepared_context_certified.load(
                        std::memory_order_acquire),
                .durable_epoch =
                    dynamic_worker_->restored_durable_epoch.load(
                        std::memory_order_acquire),
                .restoration_movement_waves = restoration_waves,
            };
            if (intent == MoEOverlayDeviceControllerDrainIntent::
                              RestorePreparedContext &&
                !terminal_drain_result_.valid())
            {
                LOG_ERROR(
                    "[MoEOverlayDeviceController] terminal drain omitted its device-authored prepared-context certification");
                std::terminate();
            }
            dynamic_worker_.reset();
        }
        else
        {
            terminal_drain_result_.succeeded =
                intent == MoEOverlayDeviceControllerDrainIntent::
                              ReleaseResources;
        }
        for (auto &endpoint : endpoints_)
        {
            if (endpoint)
                releaseEndpoint(*endpoint);
        }
        endpoints_.clear();
        activation_state_.store(
            MoEOverlayDeviceControllerActivationState::Stopped,
            std::memory_order_release);
        return terminal_drain_result_;
    }

    void MoEOverlayDeviceControllerGraphService::materializeEndpoint(
        Endpoint &endpoint)
    {
        endpoint.worker = &GPUDeviceContextPool::instance().getContext(
            endpoint.binding.device);
        endpoint.backend = getBackendFor(endpoint.binding.device);
        endpoint.kernel =
            llaminar::v2::kernels::KernelFactory::createMoEKernel(
                endpoint.binding.device);
        if (!endpoint.worker || !endpoint.backend || !endpoint.kernel)
        {
            throw std::runtime_error(
                "Device controller graph could not resolve its worker, backend, or MoE kernel");
        }

        endpoint.worker->submitAndWait(
            [this, &endpoint]
            {
                endpoint.stream = endpoint.worker->getOrCreateAuxiliaryStream(
                    streamName(endpoint.binding));
                endpoint.terminal_event = endpoint.worker->createEvent();
                if (!endpoint.stream ||
                    !endpoint.runtime_binding.initial_runtime_publisher ||
                    !endpoint.runtime_binding.initial_runtime_publisher
                         ->publishMoEOverlayDeviceInitialRuntime(
                             endpoint.stream))
                {
                    throw std::runtime_error(
                        "Device controller could not order a complete retained-layer runtime image for participant " +
                        std::to_string(endpoint.binding.participant_id));
                }
                if (endpoint.binding.authority_leader)
                {
                    endpoint.policy_result = static_cast<
                        MoEOverlayDeviceControllerPolicyResult *>(
                        endpoint.backend->allocate(
                            sizeof(MoEOverlayDeviceControllerPolicyResult),
                            endpoint.binding.device.gpu_ordinal()));
                }
                if (config_.execution_mode == ExecutionMode::Dynamic)
                {
                    endpoint.histogram_baseline_words =
                        static_cast<std::size_t>(
                            endpoint.snapshot_config.num_layers) *
                        static_cast<std::size_t>(
                            endpoint.snapshot_config.num_experts);
                    if (endpoint.histogram_baseline_words == 0u ||
                        endpoint.histogram_baseline_words >
                            std::numeric_limits<std::size_t>::max() /
                                (kMoEOverlayDeviceControllerDemandPhaseCount * sizeof(std::uint64_t)))
                    {
                        throw std::overflow_error(
                            "Device controller histogram baseline geometry overflowed size_t");
                    }
                    endpoint.histogram_phase_baselines = static_cast<
                        std::uint64_t *>(endpoint.backend->allocate(
                        kMoEOverlayDeviceControllerDemandPhaseCount * endpoint.histogram_baseline_words *
                            sizeof(std::uint64_t),
                        endpoint.binding.device.gpu_ordinal()));
                    endpoint.arrival_inbox = std::make_unique<
                        MoEOverlayDevicePreparedArrivalInbox>(
                        MoEOverlayDevicePreparedArrivalInbox::Config{
                            .backend = endpoint.backend,
                            .runtime_binding = endpoint.runtime_binding,
                            .command_capacity =
                                config_.fabric->layout().header.command_capacity,
                            .consumer_stream = endpoint.stream,
                            .perf_device = config_.perf_device,
                        });
                }
                if (!endpoint.stream || !endpoint.terminal_event ||
                    (endpoint.binding.authority_leader &&
                     !endpoint.policy_result) ||
                    (config_.execution_mode == ExecutionMode::Dynamic &&
                     (!endpoint.arrival_inbox ||
                      !endpoint.histogram_phase_baselines)))
                {
                    throw std::runtime_error(
                        "Device controller graph could not allocate its persistent stream, event, policy result, histogram baselines, or arrival inbox");
                }
                if (config_.execution_mode == ExecutionMode::Dynamic &&
                    (!endpoint.backend->memset(
                         endpoint.histogram_phase_baselines,
                         0,
                         kMoEOverlayDeviceControllerDemandPhaseCount * endpoint.histogram_baseline_words *
                             sizeof(std::uint64_t),
                         endpoint.binding.device.gpu_ordinal(),
                         endpoint.stream)))
                {
                    throw std::runtime_error(
                        "Device controller could not initialize its model-lifetime histogram baselines");
                }

                const MoEKernelLaunchContext launch{
                    .stream = endpoint.stream};
                const auto enqueue = [&endpoint, &launch](
                                         MoEOverlayDeviceControllerAction action,
                                         MoEOverlayDeviceControllerTransactionKind kind =
                                             MoEOverlayDeviceControllerTransactionKind::Invalid,
                                         MoEOverlayDeviceDemandPhase demand_phase =
                                             MoEOverlayDeviceDemandPhase::Invalid)
                {
                    const bool policy_action =
                        action == MoEOverlayDeviceControllerAction::
                                      AuthorStaticPolicy ||
                        action == MoEOverlayDeviceControllerAction::
                                      AuthorDynamicPolicy ||
                        action == MoEOverlayDeviceControllerAction::
                                      AuthorPreparedContextRestore ||
                        action == MoEOverlayDeviceControllerAction::
                                      PublishCommand;
                    const bool publication_action =
                        action == MoEOverlayDeviceControllerAction::
                                      ApplyRuntimeCandidate ||
                        action == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeCandidate ||
                        action == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeRetirement;
                    const bool readiness_action =
                        action == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeRetirementReadiness;
                    return endpoint.kernel
                        ->runMoEOverlayDeviceControllerAction(
                            launch,
                            {
                                .binding = endpoint.binding.deviceBinding(),
                                .action = action,
                                .transaction_kind = kind,
                                .demand_phase = demand_phase,
                                .policy_result = policy_action
                                                     ? endpoint.policy_result
                                                     : nullptr,
                                .runtime_publication =
                                    publication_action &&
                                            endpoint.arrival_inbox
                                        ? endpoint.arrival_inbox
                                              ->publicationBinding()
                                        : MoEOverlayDeviceRuntimePublicationBinding{},
                                .retirement_readiness = readiness_action
                                    ? MoEOverlayDeviceRetirementReadinessBinding{
                                          .epoch_control = endpoint
                                              .runtime_binding.epoch_control,
                                      }
                                    : MoEOverlayDeviceRetirementReadinessBinding{},
                            });
                };

                const auto pack_snapshot = [
                                               &endpoint,
                                               &launch](
                                               std::uint64_t *baseline)
                {
                    // Packing and publication share the participant's exact
                    // retained stream. The mapped participant record is the
                    // release edge consumed by the group root.
                    const auto plane_words = static_cast<std::size_t>(
                        endpoint.snapshot_config.num_layers) *
                        endpoint.snapshot_config.num_experts;
                    // Histogram ABI and economy use the same source order.
                    // All planes are packed before the single release edge;
                    // no host histogram or additional collective is involved.
                    static_assert(moe_runtime_abi::kHistogramSourceCount ==
                                  kMoEOverlayDeviceControllerDemandPhaseCount);
                    for (std::uint32_t phase = 0u;
                         phase < kMoEOverlayDeviceControllerDemandPhaseCount;
                         ++phase)
                    {
                        const auto offset = phase * plane_words;
                        if (!endpoint.kernel->packDeviceRebalanceHistograms(
                                launch,
                                endpoint.runtime_binding.runtime_layers_device,
                                endpoint.binding.participant_collected_state + offset,
                                endpoint.snapshot_config,
                                /*wave_state=*/nullptr,
                                /*controller_state=*/nullptr,
                                /*command_buffer_count=*/1u,
                                1u << phase,
                                baseline ? baseline + offset : nullptr))
                            return false;
                    }
                    return true;
                };

                const auto capture = [&](std::unique_ptr<IGPUGraphCapture> &graph,
                                         auto &&body,
                                         const char *name)
                {
                    graph = endpoint.worker->createGraphCapture(endpoint.stream);
                    if (!graph || !graph->beginCapture() || !body() ||
                        !graph->endCapture() || !graph->instantiate())
                    {
                        throw std::runtime_error(
                            std::string("Device controller could not capture ") +
                            name + " for participant " +
                            std::to_string(endpoint.binding.participant_id));
                    }
                };

                if (config_.execution_mode == ExecutionMode::Static)
                {
                    capture(
                        endpoint.static_graph,
                        [&]
                        {
                            bool captured = true;
                            if (endpoint.binding.authority_leader)
                            {
                                captured = enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        BeginTransaction,
                                    MoEOverlayDeviceControllerTransactionKind::
                                        StaticCheck);
                            }
                            captured = captured &&
                                pack_snapshot(
                                    /*baseline=*/nullptr) &&
                                enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        PublishParticipantSnapshot);
                            if (endpoint.binding.group_root)
                            {
                                captured = captured && enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        PublishGroupSnapshot);
                            }
                            if (endpoint.binding.authority_leader)
                            {
                                captured = captured &&
                                    enqueue(
                                        MoEOverlayDeviceControllerAction::
                                            AuthorStaticPolicy) &&
                                    enqueue(
                                        MoEOverlayDeviceControllerAction::
                                            PublishCommand);
                            }
                            if (endpoint.binding.group_root)
                            {
                                captured = captured &&
                                    enqueue(
                                        MoEOverlayDeviceControllerAction::
                                            AcknowledgePrepared);
                            }
                            if (endpoint.binding.authority_leader)
                            {
                                captured = captured && enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        BeginCommit);
                            }
                            if (endpoint.binding.group_root)
                            {
                                captured = captured && enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        AcknowledgePublished);
                            }
                            if (endpoint.binding.authority_leader)
                            {
                                captured = captured && enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        PublishAdmission);
                            }
                            if (!endpoint.binding.authority_leader)
                            {
                                captured = captured && enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        AwaitTransactionComplete);
                            }
                            return captured;
                        },
                        "Static transaction");
                    return;
                }

                /* Each graph below is one bounded device epoch. Physical
                 * preparation, inference grace periods, and physical source
                 * retirement happen between them and select the next graph
                 * through immutable authenticated receipts. */
                capture(
                    endpoint.dynamic_service_telemetry_snapshot_graph,
                    [&]
                    {
                        return endpoint.kernel
                            ->publishMoEOverlayServiceTelemetry(
                                launch,
                                endpoint.runtime_binding
                                    .service_telemetry_device,
                                endpoint.runtime_binding
                                    .service_samples_device,
                                endpoint.runtime_binding.layer_count,
                                endpoint.runtime_binding
                                    .overlay_participant_id,
                                endpoint.binding
                                    .service_telemetry_publication);
                    },
                    "Dynamic service-telemetry snapshot phase");

                const auto capture_decision_family = [&] (
                    std::unique_ptr<IGPUGraphCapture> &begin_graph,
                    std::unique_ptr<IGPUGraphCapture> &author_graph,
                    MoEOverlayDeviceDemandPhase demand_phase,
                    const char *phase_name)
                {
                    /*
                     * Do not capture the cross-rank decision as one resident
                     * wait chain. A faster controller group can otherwise
                     * enter AuthorDynamicPolicy while a slower group still has
                     * an inference graph occupying the GPU needed to publish
                     * its snapshot. The three finite graphs are admitted by
                     * durable mapped receipts in runOne(): open has no peer
                     * dependency, every participant snapshot has no peer
                     * dependency, and author launches only after every group
                     * snapshot is acquire-visible.
                     */
                    if (endpoint.binding.authority_leader)
                    {
                        capture(
                            begin_graph,
                            [&]
                            {
                                return enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        BeginTransaction,
                                    MoEOverlayDeviceControllerTransactionKind::
                                        DynamicPlacement,
                                    demand_phase);
                            },
                            phase_name);
                    }
                    if (endpoint.binding.authority_leader)
                    {
                        capture(
                            author_graph,
                            [&]
                            {
                                return enqueue(
                                           MoEOverlayDeviceControllerAction::
                                               AuthorDynamicPolicy,
                                           MoEOverlayDeviceControllerTransactionKind::
                                               Invalid,
                                           demand_phase) &&
                                       enqueue(
                                           MoEOverlayDeviceControllerAction::
                                               PublishCommand) &&
                                       enqueue(
                                           MoEOverlayDeviceControllerAction::
                                               CompleteEmptyDynamicDecision);
                            },
                            phase_name);
                    }
                };
                capture(
                    endpoint.dynamic_snapshot_graph,
                    [&]
                    {
                        return pack_snapshot(endpoint.histogram_phase_baselines) &&
                            enqueue(MoEOverlayDeviceControllerAction::PublishParticipantSnapshot);
                    },
                    "Dynamic phase-complete participant snapshot");
                capture(
                    endpoint.dynamic_histogram_rebase_graph,
                    [&]
                    {
                        /* Each pack advances its device-resident cumulative
                         * baseline. The packed scratch row is intentionally
                         * unpublished: certification traffic is valid economy
                         * evidence, but must not become placement demand. */
                        return pack_snapshot(endpoint.histogram_phase_baselines);
                    },
                    "Dynamic calibration-demand histogram rebase");
                capture_decision_family(
                    endpoint.dynamic_begin_prefill_graph,
                    endpoint.dynamic_author_prefill_graph,
                    MoEOverlayDeviceDemandPhase::Prefill,
                    "Dynamic bounded prefill decision phase");
                capture_decision_family(
                    endpoint.dynamic_begin_decode_graph,
                    endpoint.dynamic_author_decode_graph,
                    MoEOverlayDeviceDemandPhase::Decode,
                    "Dynamic bounded decode decision phase");
                if (endpoint.binding.authority_leader)
                {
                    capture(
                        endpoint.prepared_context_restore_begin_graph,
                        [&]
                        {
                            return enqueue(
                                MoEOverlayDeviceControllerAction::
                                    BeginTransaction,
                                MoEOverlayDeviceControllerTransactionKind::
                                    PreparedContextRestore,
                                MoEOverlayDeviceDemandPhase::Invalid);
                        },
                        "prepared-context restoration transaction-open phase");
                    capture(
                        endpoint.prepared_context_restore_author_graph,
                        [&]
                        {
                            return enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           AuthorPreparedContextRestore,
                                       MoEOverlayDeviceControllerTransactionKind::
                                           Invalid,
                                       MoEOverlayDeviceDemandPhase::Invalid) &&
                                   enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           PublishCommand) &&
                                   enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           CompleteEmptyDynamicDecision);
                        },
                        "prepared-context restoration policy-author phase");
                }
                capture(
                    endpoint.prepared_context_restore_snapshot_graph,
                    [&]
                    {
                        /* Restoration consumes ownership only. Cumulative
                         * demand is packed solely because the shared snapshot
                         * wire format carries both values; no phase baseline
                         * is advanced and the restore author ignores counts. */
                        return pack_snapshot(
                                   /*baseline=*/nullptr) &&
                               enqueue(
                                   MoEOverlayDeviceControllerAction::
                                       PublishParticipantSnapshot);
                    },
                    "prepared-context restoration participant-snapshot phase");
                if (endpoint.binding.group_root)
                {
                    /* This reduction graph is intentionally phase-agnostic.
                     * runOne() selects it only after acquire-visible local
                     * participant receipts identify the open transaction. */
                    capture(
                        endpoint.dynamic_group_snapshot_graph,
                        [&]
                        {
                            return enqueue(
                                MoEOverlayDeviceControllerAction::
                                    PublishGroupSnapshot);
                        },
                        "Dynamic bounded group-snapshot publication phase");
                }

                /*
                 * Publication uses the same finite receipt-selected shape as
                 * snapshot reduction.  The former monolithic graph let a fast
                 * ROCm participant enter AcknowledgePrepared while a sibling's
                 * graph was still queued behind grouped inference.  That
                 * device-resident peer wait could in turn prevent the grouped
                 * transaction from retiring.  Each graph below now consumes
                 * only prerequisites which runOne() has already observed as
                 * immutable mapped facts.
                 */
                capture(
                    endpoint.dynamic_prepare_candidate_graph,
                    [&]
                    {
                        return enqueue(
                            MoEOverlayDeviceControllerAction::
                                ApplyRuntimeCandidate);
                    },
                    "Dynamic runtime-candidate preparation phase");
                if (endpoint.binding.group_root)
                {
                    capture(
                        endpoint.dynamic_acknowledge_prepared_graph,
                        [&]
                        {
                            return enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgePrepared);
                        },
                        "Dynamic group preparation-receipt phase");
                }
                if (endpoint.binding.authority_leader)
                {
                    capture(
                        endpoint.dynamic_begin_commit_graph,
                        [&]
                        {
                            return enqueue(
                                MoEOverlayDeviceControllerAction::BeginCommit);
                        },
                        "Dynamic topology commit phase");
                }
                capture(
                    endpoint.dynamic_publish_candidate_graph,
                    [&]
                    {
                        return enqueue(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeCandidate);
                    },
                    "Dynamic runtime-candidate publication phase");
                if (endpoint.binding.group_root)
                {
                    capture(
                        endpoint.dynamic_acknowledge_published_graph,
                        [&]
                        {
                            return enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgePublished);
                        },
                        "Dynamic group publication-receipt phase");
                }
                if (endpoint.binding.authority_leader)
                {
                    capture(
                        endpoint.dynamic_publish_admission_graph,
                        [&]
                        {
                            return enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           PublishAdmission) &&
                                   enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           BeginDynamicRetirement);
                        },
                        "Dynamic topology admission phase");
                }
                capture(
                    endpoint.dynamic_retirement_readiness_graph,
                    [&]
                    {
                        /* Retirement is already open before this graph is
                         * selected. The probe is bounded: an outstanding old-
                         * bank reader publishes the same receipt from its
                         * captured inference-release epilogue instead. */
                        return enqueue(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirementReadiness);
                    },
                    "Dynamic runtime-retirement readiness phase");

                capture(
                    endpoint.dynamic_retire_graph,
                    [&]
                    {
                        return enqueue(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirement);
                    },
                    "Dynamic bounded local-retirement epoch");

                capture(
                    endpoint.dynamic_complete_graph,
                    [&]
                    {
                        bool captured = true;
                        if (endpoint.binding.group_root)
                        {
                            captured = enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgeRetired);
                        }
                        if (endpoint.binding.authority_leader)
                        {
                            captured = captured && enqueue(
                                MoEOverlayDeviceControllerAction::
                                    CompleteDynamicRetirement);
                        }
                        else
                        {
                            captured = captured && enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AwaitTransactionComplete);
                        }
                        return captured;
                    },
                    "Dynamic bounded completion epoch");
            });
    }

    bool MoEOverlayDeviceControllerGraphService::launchAll(
        std::string *error)
    {
        for (auto &owned : endpoints_)
        {
            Endpoint &endpoint = *owned;
            try
            {
                endpoint.worker->submitAndWait(
                    [&endpoint]
                    {
                        if (!endpoint.static_graph || !endpoint.stream ||
                            !endpoint.terminal_event ||
                            !endpoint.static_graph->launchOnStream(
                                endpoint.stream) ||
                            !endpoint.worker->recordEventChecked(
                                endpoint.terminal_event,
                                endpoint.stream))
                        {
                            throw std::runtime_error(
                                "controller graph launch or terminal event publication failed");
                        }
                        endpoint.in_flight = true;
                        endpoint.terminal_ready = false;
                    });
            }
            catch (const std::exception &exception)
            {
                return fail(
                    error,
                    "Could not submit device controller graph for participant " +
                        std::to_string(endpoint.binding.participant_id) +
                        ": " + exception.what());
            }
        }
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::
        validateAllStaticTerminals(std::string *error)
    {
        for (auto &owned : endpoints_)
        {
            Endpoint &endpoint = *owned;
            MoEOverlayDeviceControllerSharedHeader controller{};
            MoEOverlayDeviceControllerCommandHeader command{};
            MoEOverlayDeviceControllerGroupRecord group_record{};
            std::vector<MoEOverlayDeviceControllerParticipantRecord>
                participant_records;
            if (endpoint.binding.group_root)
            {
                const auto &group_layout = config_.fabric->layout().groups.at(
                    static_cast<std::size_t>(endpoint.binding.group_id));
                participant_records.resize(
                    static_cast<std::size_t>(
                        group_layout.participant_record_count));
            }
            try
            {
                endpoint.worker->submitAndWait(
                    [&]
                    {
                        // Every participant has its own exact terminal event.
                        // Shared mapped records are read once through the group
                        // root alias: some drivers attribute one host mapping
                        // to the first registering device and reject redundant
                        // diagnostic copies through sibling-device aliases.
                        std::unique_ptr<
                            std::byte,
                            PinnedValidationDeleter>
                            staging(
                                nullptr,
                                PinnedValidationDeleter{
                                    .backend = endpoint.backend,
                                    .ordinal = endpoint.binding.device
                                                   .gpu_ordinal(),
                                });
                        std::optional<StaticTerminalValidationLayout> layout;
                        bool copies_submitted = true;
                        bool terminal_recorded = true;
                        if (endpoint.binding.group_root)
                        {
                            layout.emplace(participant_records.size());
                            staging.reset(static_cast<std::byte *>(
                                endpoint.backend->allocatePinned(
                                    layout->bytes,
                                    endpoint.binding.device.gpu_ordinal())));
                            if (!staging)
                            {
                                throw std::runtime_error(
                                    "controller terminal diagnostic staging allocation failed");
                            }

                            // The four records are non-contiguous on device,
                            // but share one pinned destination and one stream
                            // terminal. Every submission is attempted so an
                            // accepted prefix is always covered by that event.
                            copies_submitted = endpoint.backend
                                ->deviceToHostOnStream(
                                    staging.get() + layout->controller,
                                    endpoint.binding.controller,
                                    sizeof(controller),
                                    endpoint.binding.device.gpu_ordinal(),
                                    endpoint.stream);
                            copies_submitted = endpoint.backend
                                ->deviceToHostOnStream(
                                    staging.get() + layout->command,
                                    endpoint.binding.command,
                                    sizeof(command),
                                    endpoint.binding.device.gpu_ordinal(),
                                    endpoint.stream) && copies_submitted;
                            copies_submitted = endpoint.backend
                                ->deviceToHostOnStream(
                                    staging.get() + layout->group,
                                    endpoint.binding.local_group,
                                    sizeof(group_record),
                                    endpoint.binding.device.gpu_ordinal(),
                                    endpoint.stream) && copies_submitted;
                            copies_submitted = endpoint.backend
                                ->deviceToHostOnStream(
                                    staging.get() + layout->participants,
                                    endpoint.binding.group_participant_records,
                                    participant_records.size() *
                                        sizeof(MoEOverlayDeviceControllerParticipantRecord),
                                    endpoint.binding.device.gpu_ordinal(),
                                    endpoint.stream) && copies_submitted;
                            terminal_recorded =
                                endpoint.worker->recordEventChecked(
                                    endpoint.terminal_event,
                                    endpoint.stream);
                        }

                        const bool terminal_observed = terminal_recorded &&
                            endpoint.worker->synchronizeEventChecked(
                                endpoint.terminal_event);
                        if (!terminal_observed)
                        {
                            /* A rejected terminal cannot prove that accepted
                             * async copies stopped using the pinned pages.
                             * Preserve them on the poisoned setup path rather
                             * than risk a use-after-free during unwinding. */
                            (void)staging.release();
                            throw std::runtime_error(
                                "controller terminal event reported an asynchronous backend failure");
                        }
                        endpoint.in_flight = false;
                        if (!copies_submitted)
                        {
                            throw std::runtime_error(
                                "controller terminal diagnostic copy failed");
                        }

                        if (endpoint.binding.group_root)
                        {
                            std::memcpy(
                                &controller,
                                staging.get() + layout->controller,
                                sizeof(controller));
                            std::memcpy(
                                &command,
                                staging.get() + layout->command,
                                sizeof(command));
                            std::memcpy(
                                &group_record,
                                staging.get() + layout->group,
                                sizeof(group_record));
                            std::memcpy(
                                participant_records.data(),
                                staging.get() + layout->participants,
                                participant_records.size() *
                                    sizeof(MoEOverlayDeviceControllerParticipantRecord));
                        }
                    });
            }
            catch (const std::exception &exception)
            {
                return fail(
                    error,
                    "Could not validate device controller graph for participant " +
                        std::to_string(endpoint.binding.participant_id) +
                    ": " + exception.what());
            }

            if (!endpoint.binding.group_root)
                continue;

            const auto expected_state = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete);
            const auto expected_kind = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::StaticCheck);
            const auto no_error = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None);
            if (controller.magic != kMoEOverlayDeviceControllerMagic ||
                controller.version !=
                    kMoEOverlayDeviceControllerVersion ||
                controller.topology_fingerprint !=
                    config_.topology->topology_fingerprint ||
                controller.state != expected_state ||
                controller.transaction_kind != expected_kind ||
                controller.error_code != no_error ||
                controller.transaction_id == 0u ||
                controller.current_durable_epoch == 0u ||
                controller.base_epoch !=
                    controller.current_durable_epoch ||
                controller.candidate_epoch !=
                    controller.current_durable_epoch ||
                controller.admission_epoch !=
                    controller.current_durable_epoch ||
                controller.admission_transaction !=
                    controller.transaction_id ||
                controller.completed_transaction !=
                    controller.transaction_id ||
                controller.active_llep_transaction != 0u ||
                command.kind != expected_kind ||
                command.transaction_id != controller.transaction_id ||
                command.command_count != 0u ||
                command.packed_weight_bytes != 0u ||
                command.parallel_command_count != 0u ||
                command.movement_round_count != 0u ||
                command.hazard_count != 0u ||
                group_record.magic != kMoEOverlayDeviceControllerMagic ||
                group_record.version != kMoEOverlayDeviceControllerVersion ||
                group_record.group_id !=
                    static_cast<std::uint32_t>(endpoint.binding.group_id) ||
                group_record.topology_fingerprint !=
                    config_.topology->topology_fingerprint ||
                group_record.snapshot_transaction !=
                    controller.transaction_id ||
                group_record.prepared_transaction !=
                    controller.transaction_id ||
                group_record.published_transaction !=
                    controller.transaction_id ||
                group_record.status_code != no_error)
            {
                std::ostringstream detail;
                detail
                    << "Captured device controller published an invalid "
                       "zero-movement StaticCheck terminal for participant "
                    << endpoint.binding.participant_id
                    << ": controller{magic=" << controller.magic
                    << ",version=" << controller.version
                    << ",fingerprint=" << controller.topology_fingerprint
                    << ",state=" << controller.state
                    << ",kind=" << controller.transaction_kind
                    << ",error=" << controller.error_code
                    << ",error_group=" << controller.error_group_id
                    << ",transaction=" << controller.transaction_id
                    << ",durable=" << controller.current_durable_epoch
                    << ",base=" << controller.base_epoch
                    << ",candidate=" << controller.candidate_epoch
                    << ",admission_epoch=" << controller.admission_epoch
                    << ",admission_transaction="
                    << controller.admission_transaction
                    << ",completed_transaction="
                    << controller.completed_transaction
                    << ",active_llep="
                    << controller.active_llep_transaction
                    << "} command{magic=" << command.magic
                    << ",version=" << command.version
                    << ",kind=" << command.kind
                    << ",transaction=" << command.transaction_id
                    << ",commands=" << command.command_count
                    << ",bytes=" << command.packed_weight_bytes
                    << ",parallel=" << command.parallel_command_count
                    << ",rounds=" << command.movement_round_count
                    << ",hazards=" << command.hazard_count
                    << "} group{magic=" << group_record.magic
                    << ",version=" << group_record.version
                    << ",group=" << group_record.group_id
                    << ",fingerprint="
                    << group_record.topology_fingerprint
                    << ",snapshot="
                    << group_record.snapshot_transaction
                    << ",prepared="
                    << group_record.prepared_transaction
                    << ",published="
                    << group_record.published_transaction
                    << ",status=" << group_record.status_code << '}';
                return fail(
                    error,
                    detail.str());
            }

            const auto &group_layout = config_.fabric->layout().groups.at(
                static_cast<std::size_t>(endpoint.binding.group_id));
            for (std::size_t member = 0u;
                 member < participant_records.size();
                 ++member)
            {
                const auto &record = participant_records[member];
                if (record.magic !=
                        kMoEOverlayDeviceControllerFabricMagic ||
                    record.version !=
                        kMoEOverlayDeviceControllerFabricVersion ||
                    record.participant_id !=
                        group_layout.participant_ids[member] ||
                    record.group_id != group_layout.group_id ||
                    record.topology_fingerprint !=
                        config_.topology->topology_fingerprint ||
                    record.snapshot_digest == 0u ||
                    record.snapshot_transaction !=
                        controller.transaction_id ||
                    record.status_code != no_error)
                {
                    std::ostringstream detail;
                    detail
                        << "Captured device controller observed an invalid "
                           "participant snapshot record for group "
                        << group_layout.group_id << " member " << member
                        << ": participant=" << record.participant_id
                        << ",record_group=" << record.group_id
                        << ",fingerprint="
                        << record.topology_fingerprint
                        << ",digest=" << record.snapshot_digest
                        << ",transaction="
                        << record.snapshot_transaction
                        << ",status=" << record.status_code;
                    return fail(error, detail.str());
                }
            }
        }
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::certifyStaticNoMovement(
        std::string *error)
    {
        if (config_.execution_mode != ExecutionMode::Static)
        {
            return fail(
                error,
                "Dynamic device controller service cannot run the Static certification graph");
        }
        if (static_certified_)
            return true;
        if (state() !=
            MoEOverlayDeviceControllerActivationState::Prepared)
        {
            return fail(
                error,
                "Static device controller certification is setup-only and requires Prepared state");
        }

        // Capture may finish at different times on CUDA and ROCm. Do not start
        // a system-scope wait kernel until every rank has retained its graph.
        config_.mpi_ctx->barrier();
        const bool local_launch = launchAll(error);
        int launch_ok = local_launch ? 1 : 0;
        int global_launch_ok = launch_ok;
        const MPI_Comm communicator = config_.mpi_ctx->communicator();
        if ((communicator != MPI_COMM_NULL &&
             MPI_Allreduce(
                 &launch_ok,
                 &global_launch_ok,
                 1,
                 MPI_INT,
                 MPI_MIN,
                 communicator) != MPI_SUCCESS) ||
            global_launch_ok == 0)
        {
            if (local_launch)
                (void)fail(
                    error,
                    "A peer rank could not submit its captured device-controller graph");
            return false;
        }

        const bool local_terminal = validateAllStaticTerminals(error);
        int terminal_ok = local_terminal ? 1 : 0;
        int global_terminal_ok = terminal_ok;
        if ((communicator != MPI_COMM_NULL &&
             MPI_Allreduce(
                 &terminal_ok,
                 &global_terminal_ok,
                 1,
                 MPI_INT,
                 MPI_MIN,
                 communicator) != MPI_SUCCESS) ||
            global_terminal_ok == 0)
        {
            if (local_terminal)
                (void)fail(
                    error,
                    "A peer rank observed an invalid device-controller terminal");
            return false;
        }

        static_certified_ = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "static_no_movement_transactions",
            1.0,
            "model_setup",
            {},
            {{"world_rank", std::to_string(config_.mpi_ctx->rank())},
             {"captured", "true"},
             {"movement_commands", "0"},
             {"packed_weight_bytes", "0"},
             {"inference_stream_waits", "0"},
             {"host_policy_mirror", "false"}});
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::notifyInferenceProgress(
        InferencePhase phase,
        std::uint64_t completed_tokens) noexcept
    {
        if (config_.execution_mode != ExecutionMode::Dynamic ||
            state() !=
                MoEOverlayDeviceControllerActivationState::Running ||
            completed_tokens == 0u ||
            (phase != InferencePhase::Prefill &&
             phase != InferencePhase::Decode))
        {
            return false;
        }
        /*
         * Continuation-authoritative prefill and decode progress arrives in
         * the next ordinary inference-command payload. A process with no
         * participant in this controller cell has no
         * local work and is a successful no-op; requiring a synthetic worker
         * would make otherwise valid sparse rank layouts unrepresentable.
         */
        if (endpoints_.empty())
            return true;
        if (!dynamic_worker_ ||
            !dynamic_worker_->healthy.load(std::memory_order_acquire))
        {
            return false;
        }
        dynamic_worker_->boundary_gate.notify(phase, completed_tokens);
        dynamic_worker_->wake_cv.notify_one();
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "inference_progress_tokens",
            static_cast<double>(completed_tokens),
            "maintenance",
            config_.perf_device,
            {{"blocking", "false"},
             {"coalescing", "true"},
             {"phase",
              phase == InferencePhase::Prefill ? "prefill" : "decode"},
             {"policy_owner", "device"},
             {"sideband",
              phase == InferencePhase::Prefill
                  ? "retired_prefill_progress"
                  : "retired_decode_progress"}});
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::healthy() const noexcept
    {
        if (state() ==
            MoEOverlayDeviceControllerActivationState::Failed)
        {
            return false;
        }
        return !dynamic_worker_ ||
               dynamic_worker_->healthy.load(std::memory_order_acquire);
    }

    std::string MoEOverlayDeviceControllerGraphService::failureMessage() const
    {
        if (!dynamic_worker_)
            return {};
        std::lock_guard<std::mutex> lock(dynamic_worker_->failure_mutex);
        return dynamic_worker_->failure;
    }

    InferenceMeasurementReadiness
    MoEOverlayDeviceControllerGraphService::measurementReadiness() const
    {
        if (!healthy())
        {
            const std::string failure = failureMessage();
            return {
                .state = InferenceMeasurementReadinessState::Failed,
                .owner = "expert_overlay_device_controller",
                .phase = "failed",
                .diagnostic = failure.empty()
                                  ? "ExpertOverlay device controller failed"
                                  : failure,
            };
        }
        if (config_.execution_mode == ExecutionMode::Static)
            return {};
        if (state() ==
                MoEOverlayDeviceControllerActivationState::Prepared ||
            state() ==
                MoEOverlayDeviceControllerActivationState::Starting)
        {
            return {
                .state = InferenceMeasurementReadinessState::Calibrating,
                .owner = "expert_overlay_device_controller",
                .phase = "prepared_not_started",
            };
        }
        if (config_.fabric && config_.fabric->economyProfilesPublished())
        {
            InferenceMeasurementReadiness ready{
                .state = config_.fabric->demandHistogramsRebased()
                             ? InferenceMeasurementReadinessState::Ready
                             : InferenceMeasurementReadinessState::Calibrating,
                .owner = "expert_overlay_economy",
                .phase = config_.fabric->demandHistogramsRebased()
                             ? "device_profile_published"
                             : "device_histogram_rebase",
            };
            if (config_.economy_certification)
            {
                const auto certification =
                    config_.economy_certification->measurementReadiness();
                ready.completed_work_units =
                    certification.completed_work_units;
                ready.required_work_units =
                    certification.required_work_units;
            }
            return ready;
        }
        if (!config_.economy_certification)
        {
            return {
                .state = InferenceMeasurementReadinessState::Failed,
                .owner = "expert_overlay_device_controller",
                .phase = "economy_unavailable",
                .diagnostic = "Dynamic device controller has no published economy profile or certification owner",
            };
        }

        auto readiness =
            config_.economy_certification->measurementReadiness();
        if (readiness.ready() &&
            config_.economy_certification->state() ==
                MoEOverlayEconomyCertificationState::Complete)
        {
            /*
             * A complete certificate must be published before device policy
             * may run.  Awaiting ordinary service evidence is different:
             * inference is already valid, and that traffic is what completes
             * the certificate while the movement authority remains dormant.
             */
            readiness.state =
                InferenceMeasurementReadinessState::Calibrating;
            readiness.phase = "publishing_device_profile";
        }
        return readiness;
    }

    MoEOptimizationStatus
    MoEOverlayDeviceControllerGraphService::optimizationStatus() const
    {
        const auto activation_state = state();
        if (activation_state ==
            MoEOverlayDeviceControllerActivationState::Stopped)
        {
            return {
                .authority = MoEOptimizationAuthority::Device,
                .state = MoEOptimizationLifecycleState::Drained,
                .activity = MoEOptimizationActivityState::Draining,
                .published_movement_waves =
                    terminal_published_movement_waves_,
                .completed_movement = terminal_movement_totals_,
            };
        }
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Device,
            .state = MoEOptimizationLifecycleState::MovementDisabled,
            .activity = MoEOptimizationActivityState::Dormant,
            .published_movement_waves =
                config_.fabric
                    ? config_.fabric->completedDurableMovementEpochs()
                    : 0u,
        };
        if (dynamic_worker_)
            status.completed_movement = dynamic_worker_->completedMovement();
        if (activation_state ==
                MoEOverlayDeviceControllerActivationState::Prepared ||
            activation_state ==
                MoEOverlayDeviceControllerActivationState::Starting)
        {
            if (config_.execution_mode == ExecutionMode::Dynamic)
                status.state = MoEOptimizationLifecycleState::LearningEconomy;
            return status;
        }
        if (!healthy())
        {
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic = failureMessage().empty()
                                    ? "ExpertOverlay device controller failed"
                                    : failureMessage();
            return status;
        }
        if (config_.execution_mode == ExecutionMode::Static)
            return status;
        if (!config_.fabric)
        {
            status.state = MoEOptimizationLifecycleState::Failed;
            status.diagnostic =
                "Dynamic ExpertOverlay device controller has no mapped control fabric";
            return status;
        }
        if (config_.fabric->economyProfilesPublished() &&
            config_.fabric->demandHistogramsRebased())
        {
            status.state = MoEOptimizationLifecycleState::Active;
            status.activity = dynamic_worker_
                                  ? dynamic_worker_->activity.load(
                                        std::memory_order_acquire)
                                  : MoEOptimizationActivityState::Failed;
            return status;
        }
        if (!config_.economy_certification)
        {
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic =
                "Dynamic ExpertOverlay device controller has no economy certification owner";
            return status;
        }

        switch (config_.economy_certification->state())
        {
        case MoEOverlayEconomyCertificationState::Failed:
        case MoEOverlayEconomyCertificationState::Stopped:
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic =
                config_.economy_certification->failureMessage();
            if (status.diagnostic.empty())
            {
                status.diagnostic =
                    "Dynamic ExpertOverlay device economy certification stopped before activation";
            }
            break;
        case MoEOverlayEconomyCertificationState::CalibratingMovement:
        case MoEOverlayEconomyCertificationState::AwaitingServiceEvidence:
        case MoEOverlayEconomyCertificationState::ExchangingServiceReadiness:
        case MoEOverlayEconomyCertificationState::ExchangingServiceEvidence:
        case MoEOverlayEconomyCertificationState::RebasingRoutingEvidence:
        case MoEOverlayEconomyCertificationState::Complete:
            /*
             * Complete certification still needs one device-profile publish
             * and demand-histogram rebase before policy may consume it.
             */
            status.state = MoEOptimizationLifecycleState::LearningEconomy;
            status.activity = MoEOptimizationActivityState::LearningEconomy;
            break;
        }
        return status;
    }

    MoEOptimizationMovementLedger
    MoEOverlayDeviceControllerGraphService::movementLedger() const
    {
        return dynamic_worker_
                   ? dynamic_worker_->completedMovementLedger()
                   : terminal_movement_ledger_;
    }

    bool MoEOverlayDeviceControllerGraphService::launchDynamicEpoch(
        DynamicGraphEpoch epoch,
        std::string *error)
    {
        const auto epoch_name = [epoch]() noexcept -> const char *
        {
            switch (epoch)
            {
            case DynamicGraphEpoch::ServiceTelemetrySnapshot:
                return "service-telemetry-snapshot";
            case DynamicGraphEpoch::RebaseHistograms:
                return "histogram-rebase";
            case DynamicGraphEpoch::BeginPrefillDecision:
                return "begin-prefill-decision";
            case DynamicGraphEpoch::BeginDecodeDecision:
                return "begin-decode-decision";
            case DynamicGraphEpoch::BeginPreparedContextRestore:
                return "begin-prepared-context-restore";
            case DynamicGraphEpoch::SnapshotPrefillDemand:
                return "snapshot-prefill-demand";
            case DynamicGraphEpoch::SnapshotDecodeDemand:
                return "snapshot-decode-demand";
            case DynamicGraphEpoch::SnapshotPreparedContextRestore:
                return "snapshot-prepared-context-restore";
            case DynamicGraphEpoch::PublishGroupSnapshot:
                return "publish-group-snapshot";
            case DynamicGraphEpoch::AuthorPrefillDecision:
                return "author-prefill-decision";
            case DynamicGraphEpoch::AuthorDecodeDecision:
                return "author-decode-decision";
            case DynamicGraphEpoch::AuthorPreparedContextRestore:
                return "author-prepared-context-restore";
            case DynamicGraphEpoch::PrepareRuntimeCandidate:
                return "prepare-runtime-candidate";
            case DynamicGraphEpoch::AcknowledgePrepared:
                return "acknowledge-prepared";
            case DynamicGraphEpoch::BeginCommit:
                return "begin-commit";
            case DynamicGraphEpoch::PublishRuntimeCandidate:
                return "publish-runtime-candidate";
            case DynamicGraphEpoch::AcknowledgePublished:
                return "acknowledge-published";
            case DynamicGraphEpoch::PublishAdmission:
                return "publish-admission";
            case DynamicGraphEpoch::PublishRetirementReadiness:
                return "publish-retirement-readiness";
            case DynamicGraphEpoch::Retire:
                return "retire";
            case DynamicGraphEpoch::Complete:
                return "complete";
            }
            return "invalid";
        }();

        struct SelectedEndpoint
        {
            Endpoint *endpoint = nullptr;
            IGPUGraphCapture *graph = nullptr;
        };
        std::vector<SelectedEndpoint> selected;
        selected.reserve(endpoints_.size());
        for (auto &owned : endpoints_)
        {
            Endpoint &endpoint = *owned;
            IGPUGraphCapture *graph = nullptr;
            switch (epoch)
            {
            case DynamicGraphEpoch::ServiceTelemetrySnapshot:
                graph = endpoint
                            .dynamic_service_telemetry_snapshot_graph
                            .get();
                break;
            case DynamicGraphEpoch::RebaseHistograms:
                graph = endpoint.dynamic_histogram_rebase_graph.get();
                break;
            case DynamicGraphEpoch::BeginPrefillDecision:
                graph = endpoint.dynamic_begin_prefill_graph.get();
                break;
            case DynamicGraphEpoch::BeginDecodeDecision:
                graph = endpoint.dynamic_begin_decode_graph.get();
                break;
            case DynamicGraphEpoch::BeginPreparedContextRestore:
                graph = endpoint.prepared_context_restore_begin_graph.get();
                break;
            case DynamicGraphEpoch::SnapshotPrefillDemand:
                graph = endpoint.dynamic_snapshot_graph.get();
                break;
            case DynamicGraphEpoch::SnapshotDecodeDemand:
                graph = endpoint.dynamic_snapshot_graph.get();
                break;
            case DynamicGraphEpoch::SnapshotPreparedContextRestore:
                graph = endpoint.prepared_context_restore_snapshot_graph.get();
                break;
            case DynamicGraphEpoch::PublishGroupSnapshot:
                graph = endpoint.dynamic_group_snapshot_graph.get();
                break;
            case DynamicGraphEpoch::AuthorPrefillDecision:
                graph = endpoint.dynamic_author_prefill_graph.get();
                break;
            case DynamicGraphEpoch::AuthorDecodeDecision:
                graph = endpoint.dynamic_author_decode_graph.get();
                break;
            case DynamicGraphEpoch::AuthorPreparedContextRestore:
                graph = endpoint.prepared_context_restore_author_graph.get();
                break;
            case DynamicGraphEpoch::PrepareRuntimeCandidate:
                graph = endpoint.dynamic_prepare_candidate_graph.get();
                break;
            case DynamicGraphEpoch::AcknowledgePrepared:
                graph = endpoint.dynamic_acknowledge_prepared_graph.get();
                break;
            case DynamicGraphEpoch::BeginCommit:
                graph = endpoint.dynamic_begin_commit_graph.get();
                break;
            case DynamicGraphEpoch::PublishRuntimeCandidate:
                graph = endpoint.dynamic_publish_candidate_graph.get();
                break;
            case DynamicGraphEpoch::AcknowledgePublished:
                graph = endpoint.dynamic_acknowledge_published_graph.get();
                break;
            case DynamicGraphEpoch::PublishAdmission:
                graph = endpoint.dynamic_publish_admission_graph.get();
                break;
            case DynamicGraphEpoch::PublishRetirementReadiness:
                graph = endpoint.dynamic_retirement_readiness_graph.get();
                break;
            case DynamicGraphEpoch::Retire:
                graph = endpoint.dynamic_retire_graph.get();
                break;
            case DynamicGraphEpoch::Complete:
                graph = endpoint.dynamic_complete_graph.get();
                break;
            }
            const bool role_specific_epoch =
                epoch == DynamicGraphEpoch::BeginPrefillDecision ||
                epoch == DynamicGraphEpoch::BeginDecodeDecision ||
                epoch == DynamicGraphEpoch::BeginPreparedContextRestore ||
                epoch == DynamicGraphEpoch::PublishGroupSnapshot ||
                epoch == DynamicGraphEpoch::AuthorPrefillDecision ||
                epoch == DynamicGraphEpoch::AuthorDecodeDecision ||
                epoch == DynamicGraphEpoch::AuthorPreparedContextRestore ||
                epoch == DynamicGraphEpoch::AcknowledgePrepared ||
                epoch == DynamicGraphEpoch::BeginCommit ||
                epoch == DynamicGraphEpoch::AcknowledgePublished ||
                epoch == DynamicGraphEpoch::PublishAdmission;
            if (!graph && role_specific_epoch)
                continue;
            if (!graph)
                return fail(error, "Dynamic bounded epoch omitted a participant graph");
            if (endpoint.in_flight)
            {
                return fail(
                    error,
                    std::string("Dynamic epoch ") + epoch_name +
                        " would overwrite an in-flight participant terminal");
            }
            selected.push_back({.endpoint = &endpoint, .graph = graph});
        }

        std::vector<std::future<void>> submissions;
        submissions.reserve(selected.size());
        try
        {
            const bool inference_snapshot_epoch =
                epoch == DynamicGraphEpoch::ServiceTelemetrySnapshot ||
                epoch == DynamicGraphEpoch::RebaseHistograms ||
                epoch == DynamicGraphEpoch::SnapshotPrefillDemand ||
                epoch == DynamicGraphEpoch::SnapshotDecodeDemand ||
                epoch == DynamicGraphEpoch::SnapshotPreparedContextRestore;
            if (inference_snapshot_epoch)
            {
                /*
                 * A device-resident MTP publication can have closed reader
                 * admission before its producer event is recorded.  That is a
                 * normal inference/maintenance overlap, not a broken edge.
                 * Preflight every participant before launching any controller
                 * graph so one deferred continuation can never leave sibling
                 * CUDA/ROCm participants in a partial epoch.  Retries execute
                 * only on the maintenance worker and do not synchronize or
                 * delay the inference producer.
                 */
                const auto deadline = protocolDeadline();
                std::vector<MoEOverlayInferenceBoundaryStatus>
                    boundary_statuses(
                        selected.size(),
                        MoEOverlayInferenceBoundaryStatus::Deferred);
                bool boundaries_submitted = false;
                while (!boundaries_submitted &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    std::vector<std::future<void>> boundary_submissions;
                    boundary_submissions.reserve(selected.size());
                    for (std::size_t index = 0u;
                         index < selected.size();
                         ++index)
                    {
                        /* A submitted event edge is immutable and reusable for
                         * this fan-in attempt. Retry only participants whose
                         * latest inference terminal was not ready; otherwise a
                         * fast CUDA endpoint can accumulate thousands of
                         * duplicate waits while one ROCm follower drains. */
                        if (boundary_statuses[index] ==
                            MoEOverlayInferenceBoundaryStatus::Submitted)
                        {
                            continue;
                        }
                        const auto &selection = selected[index];
                        Endpoint &endpoint = *selection.endpoint;
                        boundary_submissions.push_back(
                            endpoint.worker->submitAsync(
                                [epoch,
                                 &endpoint,
                                 &boundary_statuses,
                                 index]
                                {
                                    if (!endpoint.stream ||
                                        !endpoint.runtime_binding
                                             .inference_boundary)
                                    {
                                        boundary_statuses[index] =
                                            MoEOverlayInferenceBoundaryStatus::
                                                Failed;
                                        return;
                                    }
                                    boundary_statuses[index] =
                                        endpoint.runtime_binding
                                            .inference_boundary
                                            ->enqueueMoEOverlayDeviceInferenceBoundary(
                                                endpoint.stream,
                                                MoEOverlayInferenceBoundaryRequest{
                                                    .purpose =
                                                        epoch == DynamicGraphEpoch::
                                                                     ServiceTelemetrySnapshot
                                                            ? MoEOverlayInferenceBoundaryPurpose::
                                                                  ServiceTelemetrySnapshot
                                                            : epoch == DynamicGraphEpoch::
                                                                          RebaseHistograms
                                                                  ? MoEOverlayInferenceBoundaryPurpose::
                                                                        HistogramRebase
                                                                  : MoEOverlayInferenceBoundaryPurpose::
                                                                        PlacementDecisionSnapshot,
                                                });
                                }));
                    }

                    for (auto &submission : boundary_submissions)
                        submission.get();
                    boundaries_submitted = true;
                    for (const auto status : boundary_statuses)
                    {
                        if (status ==
                            MoEOverlayInferenceBoundaryStatus::Failed)
                        {
                            throw std::runtime_error(
                                std::string("Dynamic ") + epoch_name +
                                " inference-boundary preflight failed");
                        }
                        boundaries_submitted =
                            boundaries_submitted &&
                            status == MoEOverlayInferenceBoundaryStatus::
                                          Submitted;
                    }
                    if (!boundaries_submitted)
                        pollPause();
                }
                if (!boundaries_submitted)
                {
                    throw std::runtime_error(
                        std::string("Dynamic ") + epoch_name +
                        " inference-boundary preflight exceeded the standard protocol deadline");
                }
            }

            for (const auto &selection : selected)
            {
                Endpoint &endpoint = *selection.endpoint;
                IGPUGraphCapture *const graph = selection.graph;
                submissions.push_back(endpoint.worker->submitAsync(
                    [epoch_name, &endpoint, graph]
                    {
                        if (!endpoint.stream || !endpoint.terminal_event ||
                            !graph->launchOnStream(endpoint.stream) ||
                            !endpoint.worker->recordEventChecked(
                                endpoint.terminal_event,
                                endpoint.stream))
                        {
                            throw std::runtime_error(
                                std::string("Dynamic retained epoch ") +
                                epoch_name +
                                " or terminal publication failed");
                        }
                        endpoint.in_flight = true;
                        endpoint.terminal_ready = false;
                    }));
            }
            // Submit every role owner before observing one backend exception;
            // this preserves parallel fan-out across CUDA and ROCm workers.
            for (auto &submission : submissions)
                submission.get();
        }
        catch (const std::exception &exception)
        {
            return fail(
                error,
                std::string("Could not submit Dynamic ") + epoch_name +
                    " epoch: " + exception.what());
        }
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::dynamicTerminalsReady(
        bool *ready,
        std::string *error) noexcept
    {
        if (ready)
            *ready = false;
        if (!ready)
            return fail(error, "Dynamic terminal query requires an output");
        bool complete = true;
        for (auto &owned : endpoints_)
        {
            Endpoint &endpoint = *owned;
            if (!endpoint.in_flight)
                continue;
            if (!endpoint.terminal_ready)
            {
                bool endpoint_ready = false;
                if (!endpoint.backend || !endpoint.terminal_event ||
                    !endpoint.backend->queryEvent(
                        endpoint.terminal_event,
                        endpoint.binding.device.gpu_ordinal(),
                        &endpoint_ready))
                {
                    return fail(
                        error,
                        "Dynamic participant terminal event query failed");
                }
                endpoint.terminal_ready = endpoint_ready;
            }
            complete = complete && endpoint.terminal_ready;
        }
        if (complete)
        {
            for (auto &owned : endpoints_)
            {
                owned->in_flight = false;
                owned->terminal_ready = false;
            }
        }
        *ready = complete;
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::finishDynamicInboxes(
        const MoEOverlayDeviceTransportProtocol &protocol,
        const MoEOverlayDeviceTransportCommandBatch &command,
        std::string *error) noexcept
    {
        for (auto &endpoint : endpoints_)
        {
            if (!endpoint || !endpoint->arrival_inbox ||
                !endpoint->arrival_inbox->finishWave(protocol, command, error))
            {
                if (error && error->empty())
                    *error = "Dynamic participant arrival inbox could not close";
                return false;
            }
        }
        return true;
    }

    void MoEOverlayDeviceControllerGraphService::failDynamic(
        std::string message) noexcept
    {
        if (!dynamic_worker_)
            return;
        bool expected = true;
        if (dynamic_worker_->healthy.compare_exchange_strong(
                expected, false, std::memory_order_acq_rel))
        {
            dynamic_worker_->activity.store(
                MoEOptimizationActivityState::Failed,
                std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(
                    dynamic_worker_->failure_mutex);
                dynamic_worker_->failure = std::move(message);
            }
            activation_state_.store(
                MoEOverlayDeviceControllerActivationState::Failed,
                std::memory_order_release);
        }
        /* A shutdown waiter must observe failure without spending its timeout. */
        dynamic_worker_->wake_cv.notify_all();
        for (auto &protocol : dynamic_worker_->protocols)
        {
            if (protocol)
                protocol->fail();
        }
    }

    void MoEOverlayDeviceControllerGraphService::DynamicWorker::run(
        std::stop_token stop_token) noexcept
    {
        while (!stop_token.stop_requested() &&
               healthy.load(std::memory_order_acquire))
        {
            activity.store(
                economy_ready
                    ? MoEOptimizationActivityState::CollectingDemand
                    : MoEOptimizationActivityState::LearningEconomy,
                std::memory_order_release);
            {
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(2),
                    [&]
                    {
                        return stop_token.stop_requested() ||
                            (economy_ready && boundary_gate.ready()) ||
                            drain.acknowledgementPending() ||
                            !healthy.load(std::memory_order_acquire);
                    });
            }
            if (stop_token.stop_requested() ||
                !healthy.load(std::memory_order_acquire))
            {
                break;
            }

            /* Terminal intent is sampled only after any prior runOne() has
             * completed, so there is never a second lifecycle stacked over an
             * admitted transaction. Every rank publishes the intent before
             * requesting drain; followers can therefore remain active while
             * the sole leader authors restoration waves. */
            if (drain.acknowledgementPending())
            {
                activity.store(
                    MoEOptimizationActivityState::Draining,
                    std::memory_order_release);
                const auto intent = drain_intent.load(
                    std::memory_order_acquire);
                stopEconomy();
                if (intent == MoEOverlayDeviceControllerDrainIntent::
                                  RestorePreparedContext)
                {
                    std::string restore_error;
                    if (!restorePreparedContext(&restore_error))
                    {
                        owner->failDynamic(
                            restore_error.empty()
                                ? "device-owned prepared-context restoration failed"
                                : std::move(restore_error));
                        break;
                    }
                }
                drain.acknowledge(drain.currentRequest());
                wake_cv.notify_all();
                continue;
            }

            activity.store(
                economy_ready
                    ? MoEOptimizationActivityState::ReconcilingDemand
                    : MoEOptimizationActivityState::LearningEconomy,
                std::memory_order_release);

            std::string economy_error;
            if (!progressEconomy(&economy_error))
            {
                if (economy_error.empty())
                {
                    economy_error =
                        "device controller economy certification failed";
                }
                owner->failDynamic(std::move(economy_error));
                break;
            }

            /*
             * Ordinary GPU inference owns the timing counters. Once a real
             * inference window exists, publish one finite cumulative snapshot
             * on every participant and import it into certification. The
             * pending-token identity is read before launch: notifications that
             * race the graph therefore force a later snapshot instead of being
             * incorrectly claimed by this one.
             */
            const auto certification_state =
                owner->config_.economy_certification
                    ? owner->config_.economy_certification->state()
                    : MoEOverlayEconomyCertificationState::Complete;
            const bool service_evidence_mutable =
                certification_state ==
                    MoEOverlayEconomyCertificationState::
                        CalibratingMovement ||
                certification_state ==
                    MoEOverlayEconomyCertificationState::
                        AwaitingServiceEvidence;
            if (!economy_ready && service_evidence_mutable &&
                boundary_gate.ready())
            {
                const auto pending = boundary_gate.pendingTokens();
                if (pending != service_snapshot_tokens)
                {
                    if (!snapshotServiceTelemetry(&economy_error))
                    {
                        if (economy_error.empty())
                        {
                            economy_error =
                                "device service telemetry snapshot failed";
                        }
                        owner->failDynamic(std::move(economy_error));
                        break;
                    }
                    service_snapshot_tokens = pending;
                    if (!progressEconomy(&economy_error))
                    {
                        if (economy_error.empty())
                        {
                            economy_error =
                                "device economy certification failed after service import";
                        }
                        owner->failDynamic(std::move(economy_error));
                        break;
                    }
                }
            }

            /*
             * A device-authored placement transaction is not meaningful until
             * every migration and service coordinate has measured economics.
             * More importantly, launching an observation-only transaction here
             * would occupy the same device submission workers that live
             * inference uses while calibration is trying to obtain its paired
             * sample. Keep polling the host-owned evidence producer instead and
             * leave every retired-token notification in the saturating gate.
             * Once publication completes, a retained graph advances both
             * device histogram baselines at an exact inference boundary.
             * Pending token notifications may schedule a harmless empty first
             * observation, but only post-readiness demand can author movement.
             */
            if (!economy_ready)
            {
                /*
                 * No placement transaction can have been admitted while the
                 * immutable economy profile is closed.  Once every inference
                 * producer has crossed the shutdown fence, cancellation and
                 * final evidence exchange are therefore owned by stopEconomy;
                 * there is no device transaction for this worker to drain.
                 */
                if (drain.shutdownRequested())
                {
                    continue;
                }
                continue;
            }

            /*
             * A routed-activation count is not a decode-token cadence: every
             * token contributes top-k observations at every MoE layer. Keep
             * the explicit token budget here so a fast physical transaction
             * cannot immediately consume the next partial histogram window.
             * A long wave may accumulate more than one complete window; those
             * boundaries are coalesced into one fresh observation because the
             * histogram already contains all of them.
             */
            const bool owns_admission = owner->ownsLeaderGraph();
            const std::uint64_t admitted_window_tokens =
                boundary_gate.requiredTokens();
            /* The continuation rank is the sole phase scheduler. It consumes
             * whichever phase is ready and opens a device-authored ticket.
             * A follower must observe that immutable ticket first, then claim
             * only the matching local sideband window. Letting every rank call
             * consumeReady() independently made a stale prefill window race a
             * newer decode ticket and split the topology across two retained
             * graph branches. */
            MoEOverlayMaintenanceBoundaryGate::ReadyWindow window{};
            if (owns_admission)
            {
                window = boundary_gate.consumeReady(
                    economy_ready && !drain.shutdownRequested());
            }
            else
            {
                std::uint64_t transaction = 0u;
                MoEOverlayDeviceControllerTransactionKind authority_kind =
                    MoEOverlayDeviceControllerTransactionKind::Invalid;
                MoEOverlayDeviceDemandPhase authority_demand_phase =
                    MoEOverlayDeviceDemandPhase::Invalid;
                std::string ticket_error;
                if (!observeAuthorityTransaction(
                        &transaction,
                        &authority_kind,
                        &authority_demand_phase,
                        &ticket_error))
                {
                    owner->failDynamic(
                        ticket_error.empty()
                            ? "device controller follower observed a divergent authority ticket"
                            : std::move(ticket_error));
                    break;
                }
                if (transaction != 0u)
                {
                    if (authority_kind !=
                            MoEOverlayDeviceControllerTransactionKind::
                                DynamicPlacement ||
                        (authority_demand_phase !=
                             MoEOverlayDeviceDemandPhase::Prefill &&
                         authority_demand_phase !=
                             MoEOverlayDeviceDemandPhase::Decode))
                    {
                        owner->failDynamic(
                            "ordinary maintenance observed a non-Dynamic authority ticket");
                        break;
                    }
                    const auto authority_phase =
                        authority_demand_phase ==
                                MoEOverlayDeviceDemandPhase::Prefill
                            ? MoEOverlayInferencePhase::Prefill
                            : MoEOverlayInferencePhase::Decode;
                    /* The command sideband and mapped device ticket travel on
                     * independent ordered channels. Either can arrive first;
                     * wait only on this background scheduler until the exact
                     * phase is locally visible. The other phase remains
                     * pending for a later authority transaction. */
                    const auto sideband_deadline = protocolDeadline();
                    while (!stop_token.stop_requested() &&
                           healthy.load(std::memory_order_acquire) &&
                           !window &&
                           std::chrono::steady_clock::now() <
                               sideband_deadline)
                    {
                        window = boundary_gate.consumeReadyForPhase(
                            authority_phase,
                            /*admission_open=*/true);
                        if (!window)
                            pollPause();
                    }
                    if (stop_token.stop_requested() ||
                        !healthy.load(std::memory_order_acquire))
                    {
                        break;
                    }
                    if (!window)
                    {
                        owner->failDynamic(
                            "device controller follower did not receive the continuation sideband for the authority-selected phase before the standard protocol deadline");
                        break;
                    }
                }
            }
            if (!window)
            {
                if (drain.shutdownRequested())
                {
                    if (!drain_idle_diagnostic_published)
                    {
                        std::ostringstream diagnostic;
                        diagnostic
                            << "[ExpertOverlay][Controller] Drain observed no "
                               "new authority ticket rank="
                            << owner->config_.mpi_ctx->rank()
                            << " last_transaction=" << last_transaction
                            << " owns_admission="
                            << (owns_admission ? "true" : "false")
                            << " lifecycle=[";
                        for (std::size_t index = 0u;
                             index < protocols.size(); ++index)
                        {
                            if (index != 0u)
                                diagnostic << " | ";
                            diagnostic << protocols[index]->describeLifecycle();
                        }
                        diagnostic << ']';
                        LOG_INFO(diagnostic.str());
                        drain_idle_diagnostic_published = true;
                    }
                    continue;
                }
                continue;
            }
            std::string error;
            activity.store(
                MoEOptimizationActivityState::ReconcilingDemand,
                std::memory_order_release);
            TransactionResult transaction_result;
            const auto objective =
                window.phase == MoEOverlayInferencePhase::Prefill
                    ? TransactionObjective::DynamicPrefill
                    : TransactionObjective::DynamicDecode;
            if (!runOne(objective, &transaction_result, &error))
            {
                if (error.empty())
                    error = "device controller background transaction failed";
                std::ostringstream diagnostic;
                diagnostic << error << "; rank="
                           << owner->config_.mpi_ctx->rank()
                           << "; lifecycle=[";
                for (std::size_t index = 0u;
                     index < protocols.size();
                     ++index)
                {
                    if (index != 0u)
                        diagnostic << " | ";
                    diagnostic << protocols[index]->describeLifecycle();
                }
                diagnostic << ']';
                error = diagnostic.str();
                LOG_ERROR("[MoEOverlayDeviceController] " << error);
                owner->failDynamic(std::move(error));
                break;
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "background_notification_batches",
                1.0,
                "maintenance",
                owner->config_.perf_device,
                {{"coalesced_tokens",
                 std::to_string(window.completed_tokens)},
                 {"required_tokens",
                  std::to_string(admitted_window_tokens)},
                 {"phase",
                  window.phase == MoEOverlayInferencePhase::Prefill
                      ? "prefill"
                      : "decode"},
                 {"blocking_inference", "false"}});

            const std::uint64_t next_window_tokens =
                boundary_gate.advanceAfterReceipt(
                    maximum_window_tokens,
                    window_growth_factor,
                    transaction_result.cadenceReceipt());
            if (next_window_tokens != admitted_window_tokens)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "maintenance_window_growth",
                    1.0,
                    "maintenance",
                    owner->config_.perf_device,
                    {{"previous_tokens",
                      std::to_string(admitted_window_tokens)},
                     {"next_tokens",
                      std::to_string(next_window_tokens)},
                     {"maximum_tokens",
                      std::to_string(maximum_window_tokens)},
                     {"growth_factor",
                      std::to_string(window_growth_factor)},
                     {"policy_owner", "device"},
                     {"scheduler_role", "retained_graph_submission"}});
            }

            if (drain.shutdownRequested())
            {
                /* Pending cadence is not admitted work. Once runOne() closes,
                 * no ready-but-unconsumed window may delay shutdown or cause
                 * a new controller transaction. */
                continue;
            }
        }
        activity.store(
            healthy.load(std::memory_order_acquire)
                ? MoEOptimizationActivityState::Draining
                : MoEOptimizationActivityState::Failed,
            std::memory_order_release);
        stopEconomy();
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::
        snapshotServiceTelemetry(std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!owner || !owner->config_.fabric ||
            !owner->config_.economy_certification)
        {
            return fail(
                error,
                "device service snapshot lost its fabric or certification owner");
        }
        if (!owner->launchDynamicEpoch(
                DynamicGraphEpoch::ServiceTelemetrySnapshot, error))
        {
            return false;
        }

        const auto deadline = protocolDeadline();
        while (std::chrono::steady_clock::now() < deadline)
        {
            bool ready = false;
            std::string terminal_error;
            if (!owner->dynamicTerminalsReady(&ready, &terminal_error))
            {
                return fail(error, std::move(terminal_error));
            }
            if (ready)
                break;
            pollPause();
        }
        bool ready = false;
        if (!owner->dynamicTerminalsReady(&ready, error) || !ready)
        {
            return fail(
                error,
                error && !error->empty()
                    ? *error
                    : "device service snapshot exceeded the standard protocol deadline");
        }

        std::uint64_t imported_rows = 0u;
        std::uint64_t imported_samples = 0u;
        if (service_phase_observed.empty())
        {
            service_phase_observed.resize(owner->endpoints_.size());
        }
        if (service_phase_observed.size() != owner->endpoints_.size())
        {
            return fail(
                error,
                "device service snapshot observation geometry changed after worker construction");
        }
        std::size_t endpoint_index = 0u;
        for (const auto &endpoint : owner->endpoints_)
        {
            if (!endpoint)
                return fail(error, "device service snapshot lost an endpoint");
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            std::uint64_t generation = 0u;
            if (!owner->config_.fabric->trySnapshotServiceTelemetry(
                    endpoint->binding.participant_id,
                    &rows,
                    &generation))
            {
                return fail(
                    error,
                    "device service snapshot was not coherently acquire-visible for participant " +
                        std::to_string(endpoint->binding.participant_id));
            }
            std::string import_error;
            if (!owner->config_.economy_certification
                     ->importDeviceServiceMeasurements(
                         endpoint->binding.participant_id,
                         rows,
                         &import_error))
            {
                return fail(
                    error,
                    import_error.empty()
                        ? "device service snapshot import was rejected"
                        : std::move(import_error));
            }
            imported_rows += rows.size();
            std::array<std::uint64_t,
                       kExpertHistogramProductionSourceCount>
                participant_samples{};
            std::array<std::uint64_t,
                       kExpertHistogramProductionSourceCount>
                participant_activations{};
            for (const auto &row : rows)
            {
                for (std::size_t phase = 0u;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    imported_samples += row.sample_count[phase];
                    participant_samples[phase] +=
                        row.sample_count[phase];
                    participant_activations[phase] +=
                        row.activation_count[phase];
                }
            }
            constexpr std::array<const char *,
                                 kExpertHistogramProductionSourceCount>
                phase_names{"decode", "prefill", "grouped_verifier"};
            for (std::size_t phase = 0u;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (participant_samples[phase] == 0u ||
                    service_phase_observed[endpoint_index][phase])
                {
                    continue;
                }
                service_phase_observed[endpoint_index][phase] = true;
                std::ostringstream observed_layers;
                bool first_observed_layer = true;
                for (const auto &row : rows)
                {
                    if (row.sample_count[phase] == 0u)
                        continue;
                    if (!first_observed_layer)
                        observed_layers << ',';
                    observed_layers << row.layer;
                    first_observed_layer = false;
                }
                LOG_INFO(
                    "[ExpertOverlay][Economy] Device service phase observed participant="
                    << endpoint->binding.participant_id
                    << " source=" << phase_names[phase]
                    << " samples=" << participant_samples[phase]
                    << " activations="
                    << participant_activations[phase]
                    << " layers="
                    << (first_observed_layer
                            ? std::string("none")
                            : observed_layers.str()));
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "device_service_phase_observed",
                    1.0,
                    "maintenance",
                    owner->config_.perf_device,
                    {{"participant_id",
                      std::to_string(endpoint->binding.participant_id)},
                     {"source", phase_names[phase]},
                     {"samples", std::to_string(participant_samples[phase])},
                     {"activations",
                      std::to_string(participant_activations[phase])},
                     {"blocking_inference", "false"},
                     {"evidence_source", "device_local"}});
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "device_service_snapshot_publications",
                1.0,
                "maintenance",
                owner->config_.perf_device,
                {{"participant_id",
                  std::to_string(endpoint->binding.participant_id)},
                 {"generation", std::to_string(generation)},
                 {"blocking_inference", "false"},
                 {"evidence_source", "device_local"}});
            ++endpoint_index;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "device_service_snapshot_rows",
            static_cast<double>(imported_rows),
            "maintenance",
            owner->config_.perf_device,
            {{"samples", std::to_string(imported_samples)},
             {"blocking_inference", "false"},
             {"evidence_source", "device_local"}});
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::
        progressEconomy(std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!owner || !owner->config_.fabric)
            return fail(error, "device economy worker lost its mapped fabric");
        if (economy_ready)
            return true;

        /*
         * The mapped publication is the sole topology-wide execution gate.
         * Non-authority ranks may still be finishing process-local evidence
         * bookkeeping when the authority has already certified and release-
         * published the complete profile. They must follow the authority's
         * next transaction immediately; requiring their local controller to
         * independently reach Complete would create a second admission
         * authority and can strand the leader in snapshot fan-in.
         */
        if (owner->config_.fabric->economyProfilesPublished())
        {
            if (!histogram_rebase_started)
            {
                if (!owner->launchDynamicEpoch(
                        DynamicGraphEpoch::RebaseHistograms,
                        error))
                {
                    return false;
                }
                histogram_rebase_started = true;
                return true;
            }
            if (!local_histograms_rebased)
            {
                bool ready = false;
                if (!owner->dynamicTerminalsReady(&ready, error))
                    return false;
                if (!ready)
                    return true;
                try
                {
                    owner->config_.fabric
                        ->publishLocalDemandHistogramRebase();
                }
                catch (const std::exception &exception)
                {
                    return fail(error, exception.what());
                }
                local_histograms_rebased = true;
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "device_demand_histogram_rebased",
                    1.0,
                    "maintenance",
                    owner->config_.perf_device,
                    {{"policy_owner", "device"},
                     {"calibration_demand_discarded", "true"},
                     {"blocking_inference", "false"}});
            }
            if (!owner->config_.fabric->demandHistogramsRebased())
                return true;
            economy_ready = true;
            LOG_INFO(
                "[ExpertOverlay][Economy] Shared device policy profile acquired"
                << " rank=" << owner->config_.mpi_ctx->rank()
                << " authority_rank="
                << (owner->config_.fabric->ownsEconomyPublication()
                        ? "true"
                        : "false"));
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "device_economy_ready",
                1.0,
                "maintenance",
                owner->config_.perf_device,
                {{"policy_owner", "device"},
                 {"publication", "shared_authority_profile"},
                 {"blocking_inference", "false"}});
            return true;
        }

        const auto &certification =
            owner->config_.economy_certification;
        if (!certification)
        {
            economy_ready =
                owner->config_.fabric->economyProfilesPublished();
            return economy_ready || fail(
                error,
                "device economy worker has neither a complete publication nor a certification owner");
        }

        certification->poll();
        if (!certification->healthy())
        {
            return fail(
                error,
                certification->failureMessage().empty()
                    ? "device economy certification reported a fatal failure"
                    : certification->failureMessage());
        }
        if (certification->state() !=
            MoEOverlayEconomyCertificationState::Complete)
        {
            return true;
        }

        if (owner->config_.fabric->ownsEconomyPublication() &&
            !owner->config_.fabric->economyProfilesPublished())
        {
            try
            {
                const auto profiles = certification->detachedProfiles();
                if (!profiles || !profiles->valid())
                {
                    return fail(
                        error,
                        "completed device economy certification omitted detached profiles");
                }
                owner->config_.fabric->publishCertifiedEconomyProfiles(
                    *profiles);
            }
            catch (const std::exception &exception)
            {
                return fail(error, exception.what());
            }
        }
        /* The next poll launches the exact device histogram rebase. Economy
         * publication alone is deliberately insufficient for admission. */
        return true;
    }

    void MoEOverlayDeviceControllerGraphService::DynamicWorker::
        stopEconomy() noexcept
    {
        if (!owner || !owner->config_.economy_certification)
            return;
        const auto &certification =
            owner->config_.economy_certification;
        const auto state = certification->state();
        if (state == MoEOverlayEconomyCertificationState::Complete ||
            state == MoEOverlayEconomyCertificationState::Failed ||
            state == MoEOverlayEconomyCertificationState::Stopped)
        {
            return;
        }
        certification->requestStop();
        const auto deadline = protocolDeadline();
        while (std::chrono::steady_clock::now() < deadline)
        {
            certification->poll();
            const auto current = certification->state();
            if (current == MoEOverlayEconomyCertificationState::Complete ||
                current == MoEOverlayEconomyCertificationState::Failed ||
                current == MoEOverlayEconomyCertificationState::Stopped)
            {
                return;
            }
            pollPause();
        }
        LOG_ERROR(
            "[MoEOverlayDeviceController] economy calibration did not drain within the standard protocol deadline");
    }

    void MoEOverlayDeviceControllerGraphService::DynamicWorker::
        abortPrepared(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            MoEOverlayParticipantPreparedTransfers &prepared) noexcept
    {
        for (auto &migration : prepared.migrations)
        {
            for (auto &projection : migration.projections)
            {
                if (projection)
                    projection->abort();
            }
        }

        const auto deadline = protocolDeadline();
        while (std::chrono::steady_clock::now() < deadline)
        {
            /*
             * A stop may arrive after the last inference graph has replayed.
             * Aborted relay operations still own mapped GPU commands that
             * must reach a terminal generation before their slots and lane
             * leases can be destroyed.  Submit the setup-retained progress
             * graph from maintenance; this only enqueues work on the exact
             * per-device progress stream and never waits on inference.
             */
            std::string progress_error;
            if (owner->config_.physical_fabric &&
                !owner->config_.physical_fabric
                     ->submitOutstandingTransferProgress(&progress_error))
            {
                LOG_ERROR(
                    "[MoEOverlayDeviceController] Could not drain mapped "
                    "transfer progress during abort: " << progress_error);
                break;
            }
            bool complete = true;
            for (auto &migration : prepared.migrations)
            {
                for (auto &projection : migration.projections)
                {
                    if (!projection)
                        continue;
                    std::string ignored;
                    const auto progress = projection->pollAbort(&ignored);
                    complete = complete &&
                        progress == MoEOverlayResidencyWaveProgress::Ready;
                }
            }
            if (complete)
                break;
            pollPause();
        }
        if (owner->config_.physical_fabric && batch.valid())
        {
            std::string ignored;
            (void)owner->config_.physical_fabric->abortDeviceTransfers(
                batch, &ignored);
        }
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::
        observeAuthorityTransaction(
            std::uint64_t *transaction,
            MoEOverlayDeviceControllerTransactionKind *kind,
            MoEOverlayDeviceDemandPhase *phase,
            std::string *error) const noexcept
    {
        if (transaction)
            *transaction = 0u;
        if (kind)
            *kind = MoEOverlayDeviceControllerTransactionKind::Invalid;
        if (phase)
            *phase = MoEOverlayDeviceDemandPhase::Invalid;
        if (!transaction || !kind || !phase || protocols.empty())
        {
            return fail(
                error,
                "Dynamic authority-ticket observation has incomplete outputs or no local protocol");
        }

        std::uint64_t observed_transaction = 0u;
        MoEOverlayDeviceControllerTransactionKind observed_kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        MoEOverlayDeviceDemandPhase observed_phase =
            MoEOverlayDeviceDemandPhase::Invalid;
        for (const auto &protocol : protocols)
        {
            std::uint64_t candidate_transaction = 0u;
            MoEOverlayDeviceControllerTransactionKind candidate_kind =
                MoEOverlayDeviceControllerTransactionKind::Invalid;
            MoEOverlayDeviceDemandPhase candidate_phase =
                MoEOverlayDeviceDemandPhase::Invalid;
            if (!protocol->snapshotTransactionAfter(
                    last_transaction,
                    &candidate_transaction,
                    &candidate_kind,
                    &candidate_phase))
            {
                return true;
            }
            if (observed_transaction == 0u)
            {
                observed_transaction = candidate_transaction;
                observed_kind = candidate_kind;
                observed_phase = candidate_phase;
            }
            else if (candidate_transaction != observed_transaction ||
                     candidate_kind != observed_kind ||
                     candidate_phase != observed_phase)
            {
                return fail(
                    error,
                    "Dynamic local group roots observed divergent transaction or phase tickets");
            }
        }

        *transaction = observed_transaction;
        *kind = observed_kind;
        *phase = observed_phase;
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::runOne(
        TransactionObjective objective,
        TransactionResult *result,
        std::string *error) noexcept
    {
        if (result)
            *result = TransactionResult{};
        const bool prepared_context_restore =
            objective == TransactionObjective::PreparedContextRestore;
        const MoEOverlayInferencePhase phase =
            objective == TransactionObjective::DynamicDecode
                ? MoEOverlayInferencePhase::Decode
                : MoEOverlayInferencePhase::Prefill;
        const auto expected_kind = prepared_context_restore
            ? MoEOverlayDeviceControllerTransactionKind::
                  PreparedContextRestore
            : MoEOverlayDeviceControllerTransactionKind::DynamicPlacement;
        const auto expected_demand_phase = prepared_context_restore
            ? MoEOverlayDeviceDemandPhase::Invalid
            : phase == MoEOverlayInferencePhase::Prefill
            ? MoEOverlayDeviceDemandPhase::Prefill
            : MoEOverlayDeviceDemandPhase::Decode;
        const char *const inference_phase =
            prepared_context_restore
                ? "terminal_restore"
                : phase == MoEOverlayInferencePhase::Prefill
                ? "prefill"
                : phase == MoEOverlayInferencePhase::Decode
                ? "decode"
                : "invalid";
        PerfStatsCollector::ScopedTimer transaction_timer(
            "moe_overlay_controller",
            "dynamic_transaction_wall",
            "maintenance",
            owner ? owner->config_.perf_device : std::string{},
            {{"background", "true"},
             {"blocking_inference", "false"},
             {"inference_phase", inference_phase},
             {"policy_owner", "device"}});
        if (error)
            error->clear();
        if (!owner || !result || protocols.empty() ||
            !owner->config_.physical_fabric)
        {
            return false;
        }

        const auto wait_terminals = [&](const char *phase)
        {
            /* Measure the exact device-event delay separately from transaction
             * wall time. A retained phase may merely be queued behind live
             * inference; treating that queue delay as transfer cost would make
             * the economy model reject useful movement for the wrong reason. */
            PerfStatsCollector::ScopedTimer terminal_wait_timer(
                "moe_overlay_controller",
                "dynamic_graph_terminal_wait",
                "maintenance",
                owner->config_.perf_device,
                {{"controller_phase", phase},
                 {"inference_phase", inference_phase},
                 {"blocking_inference", "false"},
                 {"policy_owner", "device"}});
            const auto deadline = protocolDeadline();
            while (std::chrono::steady_clock::now() < deadline)
            {
                bool ready = false;
                std::string terminal_error;
                if (!owner->dynamicTerminalsReady(
                        &ready, &terminal_error))
                {
                    if (error)
                        *error = std::move(terminal_error);
                    return false;
                }
                if (ready)
                    return true;
                pollPause();
            }
            if (error)
                *error = std::string("device controller timed out in ") + phase;
            return false;
        };
        const auto wait_protocol = [&](auto &&predicate,
                                       const char *timeout_message)
        {
            /* Mapped lifecycle fan-in is a distinct queue. Keeping it separate
             * from GPU terminal waits makes cross-rank publication contention
             * visible without adding a synchronization edge to inference. */
            PerfStatsCollector::ScopedTimer protocol_wait_timer(
                "moe_overlay_controller",
                "dynamic_protocol_wait",
                "maintenance",
                owner->config_.perf_device,
                {{"protocol_edge", timeout_message},
                 {"inference_phase", inference_phase},
                 {"blocking_inference", "false"},
                 {"policy_owner", "device"}});
            const auto deadline = protocolDeadline();
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (predicate())
                    return true;
                pollPause();
            }
            if (error)
            {
                *error = timeout_message;
            }
            return false;
        };

        /*
         * A decision is four finite device epochs, not one cross-rank
         * resident wait chain:
         *
         *   1. the sole leader opens transaction N;
         *   2. each participant snapshots after its local inference boundary;
         *   3. group roots reduce only after mapped participant receipts prove
         *      every local snapshot is durable;
         *   4. the leader authors policy only after mapped group receipts prove
         *      that every snapshot is durable.
         *
         * This ordering lets a slow ROCm inference transaction drain while the
         * CUDA leader is idle on the host scheduler. No CUDA wait kernel can
         * occupy the resources needed by that ROCm transaction, and the host
         * still observes lifecycle identities only—not histograms or policy.
         */
        const auto begin_epoch = prepared_context_restore
            ? DynamicGraphEpoch::BeginPreparedContextRestore
            : phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::BeginPrefillDecision
                : DynamicGraphEpoch::BeginDecodeDecision;
        if (!owner->launchDynamicEpoch(begin_epoch, error) ||
            !wait_terminals(
                prepared_context_restore
                    ? "prepared-context restoration transaction-open epoch"
                    : phase == MoEOverlayInferencePhase::Prefill
                    ? "Dynamic bounded prefill transaction-open epoch"
                    : "Dynamic bounded decode transaction-open epoch"))
        {
            return false;
        }

        std::uint64_t decision_transaction = 0u;
        MoEOverlayDeviceControllerTransactionKind authority_kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        MoEOverlayDeviceDemandPhase authority_phase =
            MoEOverlayDeviceDemandPhase::Invalid;
        bool observation_failed = false;
        if (!wait_protocol(
                [&]
                {
                    std::string observation_error;
                        if (!observeAuthorityTransaction(
                            &decision_transaction,
                            &authority_kind,
                            &authority_phase,
                            &observation_error))
                    {
                        observation_failed = true;
                        if (error)
                            *error = std::move(observation_error);
                        return true;
                    }
                    return decision_transaction > last_transaction;
                },
                "Dynamic decision did not publish its transaction-open ticket") ||
            observation_failed || authority_kind != expected_kind ||
            authority_phase != expected_demand_phase)
        {
            return observation_failed || (error && !error->empty())
                ? false
                : fail(
                      error,
                      "device controller transaction-open ticket selected the wrong typed objective");
        }

        const auto snapshot_epoch = prepared_context_restore
            ? DynamicGraphEpoch::SnapshotPreparedContextRestore
            : phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::SnapshotPrefillDemand
                : DynamicGraphEpoch::SnapshotDecodeDemand;
        if (!owner->launchDynamicEpoch(snapshot_epoch, error) ||
            !wait_terminals(
                prepared_context_restore
                    ? "prepared-context restoration participant-snapshot epoch"
                    : phase == MoEOverlayInferencePhase::Prefill
                    ? "Dynamic bounded prefill participant-snapshot epoch"
                    : "Dynamic bounded decode participant-snapshot epoch"))
        {
            return false;
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(),
                        protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->localSnapshotsReady(
                                decision_transaction);
                        });
                },
                "Dynamic local participant snapshots did not reach their group roots"))
        {
            return false;
        }

        /* Group reduction has no inference edge and is admitted only after
         * all participant releases are visible. A fast root therefore never
         * occupies its device while waiting for a sibling whose controller
         * snapshot is queued behind sparse inference. */
        if (!owner->launchDynamicEpoch(
                DynamicGraphEpoch::PublishGroupSnapshot,
                error) ||
            !wait_terminals(
                prepared_context_restore
                    ? "prepared-context restoration group-snapshot publication epoch"
                    : phase == MoEOverlayInferencePhase::Prefill
                    ? "Dynamic bounded prefill group-snapshot publication epoch"
                    : "Dynamic bounded decode group-snapshot publication epoch") ||
            !wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(),
                        protocols.end(),
                        [&](const auto &protocol)
                        {
                    return protocol->allGroupsSnapshotted(
                                decision_transaction);
                        });
                },
                "Dynamic topology groups did not publish every snapshot receipt"))
        {
            return false;
        }

        const auto author_epoch = prepared_context_restore
            ? DynamicGraphEpoch::AuthorPreparedContextRestore
            : phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::AuthorPrefillDecision
                : DynamicGraphEpoch::AuthorDecodeDecision;
        if (!owner->launchDynamicEpoch(author_epoch, error) ||
            !wait_terminals(
                prepared_context_restore
                    ? "prepared-context restoration policy-author epoch"
                    : phase == MoEOverlayInferencePhase::Prefill
                    ? "Dynamic bounded prefill policy-author epoch"
                    : "Dynamic bounded decode policy-author epoch"))
        {
            return false;
        }

        std::vector<std::optional<MoEOverlayDeviceTransportCommandBatch>>
            acquired(protocols.size());
        const auto acquire_deadline = protocolDeadline();
        while (std::chrono::steady_clock::now() < acquire_deadline)
        {
            bool all_ready = true;
            for (std::size_t index = 0u; index < protocols.size(); ++index)
            {
                if (acquired[index])
                    continue;
                auto result = protocols[index]->tryAcquire(last_transaction);
                if (result.status ==
                    MoEOverlayDeviceTransportAcquireStatus::Failed)
                {
                    return fail(
                        error,
                        result.error.empty()
                            ? "device physical follower rejected an acquired command"
                            : std::move(result.error));
                }
                if (result.status ==
                    MoEOverlayDeviceTransportAcquireStatus::Ready)
                {
                    acquired[index] = std::move(result.batch);
                }
                else
                {
                    all_ready = false;
                }
            }
            if (all_ready && std::all_of(
                    acquired.begin(), acquired.end(),
                    [](const auto &batch) { return batch.has_value(); }))
            {
                break;
            }
            pollPause();
        }
        if (std::any_of(
                acquired.begin(), acquired.end(),
                [](const auto &batch) { return !batch.has_value(); }))
        {
            return fail(
                error,
                "device physical follower timed out acquiring every local group command");
        }

        const auto &command = *acquired.front();
        if (static_cast<MoEOverlayDeviceControllerTransactionKind>(
                command.header.kind) != expected_kind)
        {
            return fail(
                error,
                "device physical follower acquired a command for the wrong typed objective");
        }
        for (std::size_t index = 1u; index < acquired.size(); ++index)
        {
            if (!sameCommand(command, *acquired[index]))
            {
                return fail(
                    error,
                    "device physical follower observed divergent group command bytes");
            }
        }

        MoEOverlayDevicePhysicalMovementBatch batch;
        try
        {
            batch = makeMoEOverlayDevicePhysicalMovementBatch(
                command, *owner->config_.topology);
        }
        catch (const std::exception &exception)
        {
            return fail(
                error,
                std::string("device physical command projection failed: ") +
                    exception.what());
        }

        // Re-derive movement direction from immutable topology metadata. This
        // authenticates the device-authored evidence without reconstructing a
        // placement policy or reading a histogram on the host follower.
        std::uint64_t promotions = 0u;
        std::uint64_t demotions = 0u;
        std::uint64_t same_priority_moves = 0u;
        std::uint64_t cross_domain_moves = 0u;
        std::uint64_t cross_rank_moves = 0u;
        std::uint64_t cross_backend_moves = 0u;
        for (const auto &migration : batch.migrations)
        {
            switch (migration.direction)
            {
            case MoEOverlayTierMigrationDirection::Promotion:
                ++promotions;
                break;
            case MoEOverlayTierMigrationDirection::Demotion:
                ++demotions;
                break;
            case MoEOverlayTierMigrationDirection::SamePriority:
                ++same_priority_moves;
                break;
            }
            cross_domain_moves += migration.crossesDomain() ? 1u : 0u;
            cross_rank_moves += migration.crossesWorldRank() ? 1u : 0u;
            cross_backend_moves += migration.crossesBackend() ? 1u : 0u;
        }
        const auto capacity_evidence =
            analyzeMoEOverlayMigrationCapacity(batch.migrations);
        if (!capacity_evidence.capacityPreserved())
        {
            return fail(
                error,
                "device physical follower rejected non-conserving participant/tier slot flow");
        }
        if (promotions != command.header.promotions ||
            demotions != command.header.demotions ||
            same_priority_moves != command.header.same_priority_moves)
        {
            return fail(
                error,
                "device physical follower rejected inconsistent policy evidence");
        }

        if (!batch.movesWeights())
        {
            if (!wait_protocol(
                    [&]
                    {
                        return std::all_of(
                            protocols.begin(), protocols.end(),
                            [&](const auto &protocol)
                            {
                                return protocol->transactionComplete(command);
                            });
                    },
                    "zero-movement decision epoch did not publish direct completion"))
            {
                return false;
            }
            last_transaction = command.header.transaction_id;
            *result = {
                .kind = expected_kind,
                .transaction = last_transaction,
                .durable_epoch = command.header.candidate_epoch,
                .command_count = 0u,
                .snapshot_observations =
                    command.header.snapshot_observations,
            };
            if (prepared_context_restore)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "prepared_context_restore_certifications",
                    1.0,
                    "model_teardown",
                    owner->config_.perf_device,
                    {{"transaction", std::to_string(last_transaction)},
                     {"durable_epoch",
                      std::to_string(command.header.candidate_epoch)},
                     {"movement_commands", "0"},
                     {"policy_owner", "device"},
                     {"exact_initial_owner_table", "true"}});
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "dynamic_zero_movement_transactions",
                    1.0,
                    "maintenance",
                    owner->config_.perf_device,
                    {{"transaction", std::to_string(last_transaction)},
                     {"movement_commands", "0"},
                     {"physical_bytes", "0"},
                     {"snapshot_observations",
                      std::to_string(command.header.snapshot_observations)},
                     {"priority_cost_before",
                      std::to_string(command.header.priority_cost_before)},
                     {"priority_cost_after",
                      std::to_string(command.header.priority_cost_after)},
                     {"same_priority_makespan_before",
                      std::to_string(
                          command.header.same_priority_makespan_before)},
                     {"same_priority_makespan_after",
                      std::to_string(
                          command.header.same_priority_makespan_after)},
                     {"accepted_cycles",
                      std::to_string(command.header.accepted_cycles)},
                     {"rejected_cycles",
                      std::to_string(command.header.rejected_cycles)},
                     {"payoff_rejected_cycles",
                      std::to_string(
                          command.header.payoff_rejected_cycles)},
                     {"residency_rejected_cycles",
                      std::to_string(
                          command.header.residency_rejected_cycles)},
                     {"projected_service_gain_ns",
                      std::to_string(
                          command.header.projected_service_gain_ns)},
                     {"projected_transfer_and_repack_ns",
                      std::to_string(
                          command.header.projected_transfer_and_repack_ns)},
                     {"projected_inference_interference_ns",
                      std::to_string(
                          command.header.projected_inference_interference_ns)},
                     {"projected_net_benefit_ns",
                      std::to_string(
                          command.header.projected_net_benefit_ns)},
                     {"layer_scan_start",
                      std::to_string(command.header.layer_scan_start)},
                     {"layer_scan_next",
                      std::to_string(command.header.layer_scan_next)},
                     {"bounded_device_phases", "true"},
                     {"resident_external_waits", "0"},
                     {"prearmed_cross_device_fanin", "true"},
                     {"policy_owner", "device"}});
            }
            return true;
        }

        activity.store(
            MoEOptimizationActivityState::MovingWeights,
            std::memory_order_release);

        MoEOverlayParticipantPreparedTransfers prepared;
        const auto prepare_deadline = protocolDeadline();
        do
        {
            prepared = owner->config_.physical_fabric->prepareDeviceTransfers(
                batch,
                owner->config_.fabric->localParticipantIds());
            if (prepared.status !=
                MoEOverlayResidencyStageStartStatus::Deferred)
            {
                break;
            }
            pollPause();
        } while (std::chrono::steady_clock::now() < prepare_deadline);
        if (prepared.status != MoEOverlayResidencyStageStartStatus::Started)
        {
            return fail(
                error,
                prepared.error.empty()
                    ? "device physical follower could not reserve its complete movement wave"
                    : std::move(prepared.error));
        }

        bool graph_phase_in_flight = false;
        bool physical_published = false;
        const auto unwind = [&](std::string message)
        {
            for (auto &protocol : protocols)
                protocol->fail();
            if (!physical_published)
                abortPrepared(batch, prepared);
            if (graph_phase_in_flight)
                (void)wait_terminals("failed movement unwind");
            // A failed transaction cannot recycle its immutable descriptor
            // pages. Endpoint teardown retains them until captured readers
            // retire; there is no separate descriptor DMA to drain here.
            return fail(error, std::move(message));
        };
        const auto run_epoch = [&](DynamicGraphEpoch epoch,
                                   const char *description)
        {
            if (!owner->launchDynamicEpoch(epoch, error))
                return false;
            graph_phase_in_flight = true;
            if (!wait_terminals(description))
                return false;
            graph_phase_in_flight = false;
            return true;
        };

        const std::size_t operation_count =
            prepared.migrations.size() * kMoEOverlayExpertProjectionCount;
        std::vector<std::uint8_t> projection_ready(operation_count, 0u);
        const auto operationAt = [&](std::size_t operation)
            -> std::unique_ptr<IMoEOverlayTierTransferOperation> &
        {
            const std::size_t migration =
                operation / kMoEOverlayExpertProjectionCount;
            const std::size_t projection =
                operation % kMoEOverlayExpertProjectionCount;
            return prepared.migrations[migration].projections[projection];
        };
        const auto pollOperation = [&](std::size_t operation,
                                       std::string *poll_error)
        {
            auto &physical_operation = operationAt(operation);
            if (!physical_operation)
            {
                if (poll_error)
                {
                    *poll_error =
                        "device physical follower lost a projection operation";
                }
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            return physical_operation->poll(poll_error);
        };

        /*
         * Submit every projection exactly once before bounding host progress.
         * Device DMA and MPI requests therefore overlap across the complete
         * device-authored wave.  Only subsequent event/request queries are
         * rate-limited; this is a driver-fairness boundary, not serialized
         * movement.
         */
        for (std::size_t operation = 0u;
             operation < operation_count;
             ++operation)
        {
            std::string poll_error;
            const auto progress = pollOperation(operation, &poll_error);
            if (progress == MoEOverlayResidencyWaveProgress::Failed ||
                progress == MoEOverlayResidencyWaveProgress::Deferred)
            {
                return unwind(
                    poll_error.empty()
                        ? "device physical projection failed during parallel submission"
                        : std::move(poll_error));
            }
            projection_ready[operation] = static_cast<std::uint8_t>(
                progress == MoEOverlayResidencyWaveProgress::Ready);
        }

        /*
         * The device-authored command can be published by the maintenance
         * worker after the final request graph has already replayed.  Captured
         * inference branches remain the zero-host-tax steady-state path, but
         * they cannot be the only progress trigger: a tail movement would then
         * wait forever for inference that may never arrive.  Enqueue each
         * setup-retained per-device progress graph now.  Submission closes on
         * the GPU worker callback only; the DMA/repack graph remains fully
         * asynchronous and all device epochs execute concurrently.
         */
        std::string transfer_progress_error;
        if (!owner->config_.physical_fabric
                 ->submitOutstandingTransferProgress(
                     &transfer_progress_error))
        {
            return unwind(
                transfer_progress_error.empty()
                    ? "device physical follower could not enqueue mapped transfer progress"
                    : std::move(transfer_progress_error));
        }

        /*
         * One host poll budget per process-local participant scales naturally
         * with the number of independent device runtimes this worker services.
         * It avoids a topology-specific constant while bounding aggregate
         * CUDA/HIP driver pressure independently of cycles-per-wave.
         */
        const std::size_t maximum_polls_per_quantum =
            std::max<std::size_t>(
                1u,
                owner->config_.fabric->localParticipantIds().size());
        MoEOverlayPhysicalWavePollCursor poll_cursor(
            operation_count, maximum_polls_per_quantum);
        std::uint64_t poll_quanta = 0u;
        std::uint64_t operation_polls = operation_count;
        const auto transfer_deadline = protocolDeadline();
        while (std::chrono::steady_clock::now() < transfer_deadline)
        {
            if (std::all_of(
                    projection_ready.begin(),
                    projection_ready.end(),
                    [](std::uint8_t ready) { return ready != 0u; }))
            {
                break;
            }

            poll_cursor.beginQuantum();
            std::size_t quantum_polls = 0u;
            while (quantum_polls < poll_cursor.maximumPollsPerQuantum())
            {
                const auto operation = poll_cursor.nextPending(
                    std::span<const std::uint8_t>(projection_ready));
                if (!operation)
                    break;
                std::string poll_error;
                const auto progress = pollOperation(*operation, &poll_error);
                ++quantum_polls;
                ++operation_polls;
                if (progress == MoEOverlayResidencyWaveProgress::Failed ||
                    progress == MoEOverlayResidencyWaveProgress::Deferred)
                {
                    return unwind(
                        poll_error.empty()
                            ? "device physical projection failed after reservation"
                            : std::move(poll_error));
                }
                projection_ready[*operation] = static_cast<std::uint8_t>(
                    progress == MoEOverlayResidencyWaveProgress::Ready);
            }

            /*
             * Pipelined relays publish the next mapped generation only after
             * polling the previous chunk ready.  Re-enqueue after each bounded
             * host poll quantum so every newly published generation advances
             * even when no later inference boundary exists.
             */
            if (!owner->config_.physical_fabric
                     ->submitOutstandingTransferProgress(
                         &transfer_progress_error))
            {
                return unwind(
                    transfer_progress_error.empty()
                        ? "device physical follower lost mapped transfer progress"
                        : std::move(transfer_progress_error));
            }
            ++poll_quanta;
            pollPause();
        }
        const bool every_projection_ready = std::all_of(
            projection_ready.begin(), projection_ready.end(),
            [](std::uint8_t ready) { return ready != 0u; });
        if (!every_projection_ready)
        {
            // Capture physical receipts before unwind aborts operations. This
            // path adds no successful-wave polling, transfer, or event wait.
            return unwind(
                "device physical transfers exceeded the protocol deadline; " +
                batch.describeProjectionReadiness(projection_ready) +
                "; poll_quanta=" + std::to_string(poll_quanta) +
                "; operation_polls=" + std::to_string(operation_polls));
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "physical_wave_parallel_operations_started",
            static_cast<double>(operation_count),
            "maintenance",
            owner->config_.perf_device,
            {{"transaction", std::to_string(batch.transaction_id)},
             {"migrations", std::to_string(batch.migrations.size())},
             {"poll_quanta", std::to_string(poll_quanta)},
             {"maximum_polls_per_quantum",
              std::to_string(maximum_polls_per_quantum)},
             {"serialized_submission", "false"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "physical_wave_bounded_progress_polls",
            static_cast<double>(operation_polls),
            "maintenance",
            owner->config_.perf_device,
            {{"transaction", std::to_string(batch.transaction_id)},
             {"operations", std::to_string(operation_count)},
             {"poll_quanta", std::to_string(poll_quanta)},
             {"maximum_polls_per_quantum",
              std::to_string(maximum_polls_per_quantum)},
             {"driver_fair", "true"}});

        std::string stage_error;
        if (!owner->config_.physical_fabric->stageDevicePreparedTransfers(
                batch, prepared, &stage_error))
        {
            return unwind(
                stage_error.empty()
                    ? "device physical follower could not retain completed destinations"
                    : std::move(stage_error));
        }

        // Physical descriptors are tiny immutable host-authored metadata, not
        // live GPU placement state. Stage them on this transport worker before
        // its existing system-release receipt. The candidate graph acquires
        // that receipt and reads the mapped pages directly into the device bank.
        // No worker dispatch, copy-engine queue, or extra readiness event exists.
        for (auto &endpoint : owner->endpoints_)
        {
            std::string staging_error;
            if (!endpoint->arrival_inbox ||
                !endpoint->arrival_inbox->stage(batch, prepared, &staging_error))
            {
                return unwind(staging_error.empty()
                    ? "participant descriptor staging failed"
                    : std::move(staging_error));
            }
        }

        for (std::size_t index = 0u; index < protocols.size(); ++index)
        {
            std::string publication_error;
            if (!protocols[index]->publishPrepared(
                    *acquired[index], &publication_error))
            {
                return unwind(
                    publication_error.empty()
                        ? "device transport could not publish physical preparation"
                        : std::move(publication_error));
            }
        }

        /* Completed destination bytes and mapped descriptors are already retained.
         * Publish their allocation lifetimes and the immutable physical receipt
         * before the device epoch begins. Neither operation can select E+1 for
         * inference; the captured publication graph below remains the sole RCU
         * selector and admission authority. This precondition keeps every
         * action in that graph bounded. */
        std::string publication_error;
        if (!owner->config_.physical_fabric->publishDevicePreparedTransfers(
                batch, &publication_error))
        {
            return unwind(
                publication_error.empty()
                    ? "physical slot ledger could not publish the device epoch"
                    : std::move(publication_error));
        }
        physical_published = true;
        for (std::size_t index = 0u; index < protocols.size(); ++index)
        {
            if (!protocols[index]->publishPublished(
                    *acquired[index], &publication_error))
            {
                return unwind(
                    publication_error.empty()
                        ? "device transport could not publish runtime installation"
                        : std::move(publication_error));
            }
        }
        activity.store(
            MoEOptimizationActivityState::PublishingResidency,
            std::memory_order_release);

        /*
         * Every publication transition below is a separate finite graph.  The
         * host scheduler observes only monotonic, device-authored receipts and
         * uses them to select the next retained graph; it never authors policy
         * or placement.  Splitting at those receipts prevents a fast device's
         * peer-wait kernel from remaining resident while a sibling's next
         * controller graph is queued behind live sparse inference.
         */
        if (!run_epoch(
                DynamicGraphEpoch::PrepareRuntimeCandidate,
                "Dynamic bounded runtime-candidate preparation epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device runtime-candidate preparation epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->preparationReady(command);
                        });
                },
                "device controller timed out awaiting participant preparation receipts"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "participant preparation receipts were not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::AcknowledgePrepared,
                "Dynamic bounded group preparation-receipt epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device group preparation-receipt epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->allGroupsPrepared(command);
                        });
                },
                "device controller timed out awaiting topology preparation receipts"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "topology preparation receipts were not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::BeginCommit,
                "Dynamic bounded topology commit epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device topology commit epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->commitRequested(command);
                        });
                },
                "device controller timed out awaiting topology commit"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "topology commit was not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::PublishRuntimeCandidate,
                "Dynamic bounded runtime-candidate publication epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device runtime-candidate publication epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->publicationReady(command);
                        });
                },
                "device controller timed out awaiting participant publication receipts"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "participant publication receipts were not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::AcknowledgePublished,
                "Dynamic bounded group publication-receipt epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device group publication-receipt epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->allGroupsPublished(command);
                        });
                },
                "device controller timed out awaiting topology publication receipts"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "topology publication receipts were not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::PublishAdmission,
                "Dynamic bounded topology admission epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device topology admission epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->retirementOpen(command);
                        });
                },
                "device controller timed out awaiting retirement admission"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "device retirement admission failed");
        }
        activity.store(
            MoEOptimizationActivityState::MovingWeights,
            std::memory_order_release);

        if (!run_epoch(
                DynamicGraphEpoch::PublishRetirementReadiness,
                "Dynamic bounded runtime-retirement readiness epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device runtime-retirement readiness epoch did not terminate");
        }

        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->runtimeReadersReady(command);
                        });
                },
                "device controller timed out awaiting the topology-wide reader receipt"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "topology-wide reader receipt was not published");
        }
        if (!run_epoch(
                DynamicGraphEpoch::Retire,
                "Dynamic bounded local-retirement epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device retirement epoch did not terminate");
        }
        if (!std::all_of(
                protocols.begin(), protocols.end(),
                [&](const auto &protocol)
                {
                    return protocol->retirementRequested(command);
                }))
        {
            return unwind(
                "device retirement epoch omitted a local reclamation receipt");
        }

        std::string retirement_error;
        if (!owner->config_.physical_fabric->retireDevicePreviousSources(
                batch, &retirement_error))
        {
            return unwind(
                retirement_error.empty()
                    ? "physical slot ledger could not retire prior sources"
                    : std::move(retirement_error));
        }
        for (std::size_t index = 0u; index < protocols.size(); ++index)
        {
            if (!protocols[index]->publishRetired(
                    *acquired[index], &retirement_error))
            {
                return unwind(
                    retirement_error.empty()
                        ? "device transport could not publish source retirement"
                        : std::move(retirement_error));
            }
        }

        if (!std::all_of(
                protocols.begin(), protocols.end(),
                [&](const auto &protocol)
                {
                    return protocol->groupRetirementReady(command);
                }))
        {
            return unwind(
                "device physical retirement omitted a local lifecycle edge");
        }
        if (!run_epoch(
                DynamicGraphEpoch::Complete,
                "Dynamic bounded completion epoch"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "device completion epoch did not terminate");
        }
        if (!wait_protocol(
                [&]
                {
                    return std::all_of(
                        protocols.begin(), protocols.end(),
                        [&](const auto &protocol)
                        {
                            return protocol->transactionComplete(command);
                        });
                },
                "device controller timed out awaiting transaction completion"))
        {
            return unwind(error && !error->empty()
                              ? *error
                              : "device transaction completion was not published");
        }
        if (!owner->finishDynamicInboxes(*protocols.front(), command, error))
        {
            return false;
        }
        last_transaction = command.header.transaction_id;
        *result = {
            .kind = expected_kind,
            .transaction = last_transaction,
            .durable_epoch = batch.candidate_epoch,
            .command_count = batch.command_count,
            .snapshot_observations = command.header.snapshot_observations,
        };

        /* Terminal restoration is physical lifecycle work, not an optimization
         * decision. It deliberately bypasses Dynamic movement/economy ledgers
         * so production assertions cannot mistake cleanup for useful inferred
         * hotness movement. The next restoration transaction will snapshot the
         * newly published epoch and either continue or certify exact equality. */
        if (prepared_context_restore)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "prepared_context_restore_movement_waves",
                1.0,
                "model_teardown",
                owner->config_.perf_device,
                {{"transaction", std::to_string(last_transaction)},
                 {"base_epoch", std::to_string(batch.base_epoch)},
                 {"candidate_epoch",
                  std::to_string(batch.candidate_epoch)},
                 {"movement_commands",
                  std::to_string(batch.command_count)},
                 {"physical_bytes",
                  std::to_string(batch.packed_weight_bytes)},
                 {"policy_owner", "device"},
                 {"excluded_from_optimization_ledger", "true"}});
            return true;
        }

        std::vector<std::size_t> cycle_index_by_migration(
            batch.migrations.size(),
            std::numeric_limits<std::size_t>::max());
        std::vector<std::size_t> cycle_size_by_migration(
            batch.migrations.size(), 0u);
        for (std::size_t cycle_index = 0u;
             cycle_index < batch.migration_cycles.size(); ++cycle_index)
        {
            const auto &cycle = batch.migration_cycles[cycle_index];
            for (const std::size_t migration_index : cycle.migration_indices)
            {
                if (migration_index >= batch.migrations.size() ||
                    cycle_index_by_migration[migration_index] !=
                        std::numeric_limits<std::size_t>::max())
                {
                    return fail(
                        error,
                        "completed device movement has an invalid migration-cycle identity");
                }
                cycle_index_by_migration[migration_index] = cycle_index;
                cycle_size_by_migration[migration_index] =
                    cycle.migration_indices.size();
            }
        }

        std::vector<MoEOptimizationMovementEdge> completed_edges;
        completed_edges.reserve(batch.migrations.size());
        for (std::size_t migration_index = 0u;
             migration_index < batch.migrations.size(); ++migration_index)
        {
            const auto &migration = batch.migrations[migration_index];
            const auto *const source_group =
                owner->config_.topology->groupForParticipant(
                    migration.source.owner_participant);
            const auto *const destination_group =
                owner->config_.topology->groupForParticipant(
                    migration.destination.owner_participant);
            if (!source_group || !destination_group ||
                cycle_index_by_migration[migration_index] ==
                    std::numeric_limits<std::size_t>::max() ||
                cycle_size_by_migration[migration_index] == 0u)
            {
                return fail(
                    error,
                    "completed device movement lost its frozen topology or cycle identity");
            }

            MoEOptimizationMovementDirection direction =
                MoEOptimizationMovementDirection::SamePriority;
            if (migration.direction ==
                MoEOverlayTierMigrationDirection::Promotion)
            {
                direction = MoEOptimizationMovementDirection::Promotion;
            }
            else if (migration.direction ==
                     MoEOverlayTierMigrationDirection::Demotion)
            {
                direction = MoEOptimizationMovementDirection::Demotion;
            }
            completed_edges.push_back({
                .authority = MoEOptimizationAuthority::Device,
                .transaction = last_transaction,
                .candidate_epoch = batch.candidate_epoch,
                .layer = migration.layer_idx,
                .expert = migration.expert_id,
                .cycle_index = cycle_index_by_migration[migration_index],
                .cycle_size = cycle_size_by_migration[migration_index],
                .direction = direction,
                .axis = migration.axis,
                .source_participant =
                    migration.source.owner_participant,
                .destination_participant =
                    migration.destination.owner_participant,
                .source_priority = source_group->tier_priority,
                .destination_priority = destination_group->tier_priority,
                .source_device = migration.source.device,
                .destination_device = migration.destination.device,
                .source_world_rank = migration.source.owner_world_rank,
                .destination_world_rank =
                    migration.destination.owner_world_rank,
                .source_world_rank_known =
                    migration.source.owner_world_rank_known,
                .destination_world_rank_known =
                    migration.destination.owner_world_rank_known,
                .estimated_weight_bytes =
                    static_cast<std::uint64_t>(
                        migration.estimated_weight_bytes),
                .activation_count = migration.activation_count,
                .blocking_inference = false,
            });
        }
        std::optional<MoEOptimizationMovementEconomy> completed_economy;
        if (owner->ownsLeaderGraph())
        {
            completed_economy = MoEOptimizationMovementEconomy{
                .authority = MoEOptimizationAuthority::Device,
                .transaction = last_transaction,
                .candidate_epoch = batch.candidate_epoch,
                .command_count = batch.command_count,
                .cycle_count = static_cast<std::uint64_t>(
                    batch.migration_cycles.size()),
                .projected_service_gain_ns =
                    command.header.projected_service_gain_ns,
                .projected_transfer_and_repack_ns =
                    command.header.projected_transfer_and_repack_ns,
                .projected_inference_interference_ns =
                    command.header.projected_inference_interference_ns,
                .projected_net_benefit_ns =
                    command.header.projected_net_benefit_ns,
            };
            if (!completed_economy->valid())
            {
                return fail(
                    error,
                    "completed device movement lost its authoritative economy proof");
            }
        }
        {
            /* One lock publishes every edge plus the leader-only policy proof. */
            std::lock_guard<std::mutex> lock(movement_ledger_mutex);
            movement_ledger.insert(
                movement_ledger.end(),
                completed_edges.begin(),
                completed_edges.end());
            if (completed_economy)
                movement_economy.push_back(*completed_economy);
        }

        /*
         * Publish operational totals before optional telemetry. Benchmark and
         * correctness callers therefore observe completed physical work even
         * when PerfStats is disabled or filtered to an unrelated domain.
         */
        movement_publication_sequence.fetch_add(
            1u, std::memory_order_acq_rel);
        movement_commands.fetch_add(
            batch.command_count, std::memory_order_relaxed);
        movement_physical_bytes.fetch_add(
            batch.packed_weight_bytes, std::memory_order_relaxed);
        movement_promotions.fetch_add(promotions, std::memory_order_relaxed);
        movement_demotions.fetch_add(demotions, std::memory_order_relaxed);
        movement_same_priority.fetch_add(
            same_priority_moves, std::memory_order_relaxed);
        movement_transactions.fetch_add(1u, std::memory_order_relaxed);
        movement_publication_sequence.fetch_add(
            1u, std::memory_order_release);

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_movement_transactions",
            1.0,
            "maintenance",
            owner->config_.perf_device,
            {{"transaction", std::to_string(last_transaction)},
             {"base_epoch", std::to_string(batch.base_epoch)},
             {"candidate_epoch", std::to_string(batch.candidate_epoch)},
             {"movement_commands", std::to_string(batch.command_count)},
             {"physical_bytes", std::to_string(batch.packed_weight_bytes)},
             {"promotions", std::to_string(promotions)},
             {"demotions", std::to_string(demotions)},
             {"same_priority_moves",
              std::to_string(same_priority_moves)},
             {"cross_domain_moves",
              std::to_string(cross_domain_moves)},
             {"cross_rank_moves",
              std::to_string(cross_rank_moves)},
             {"cross_backend_moves",
              std::to_string(cross_backend_moves)},
             {"snapshot_observations",
              std::to_string(command.header.snapshot_observations)},
             {"priority_cost_before",
              std::to_string(command.header.priority_cost_before)},
             {"priority_cost_after",
              std::to_string(command.header.priority_cost_after)},
             {"same_priority_makespan_before",
              std::to_string(
                  command.header.same_priority_makespan_before)},
             {"same_priority_makespan_after",
              std::to_string(
                  command.header.same_priority_makespan_after)},
             {"accepted_cycles",
              std::to_string(command.header.accepted_cycles)},
             {"rejected_cycles",
              std::to_string(command.header.rejected_cycles)},
             {"payoff_rejected_cycles",
              std::to_string(command.header.payoff_rejected_cycles)},
             {"residency_rejected_cycles",
              std::to_string(command.header.residency_rejected_cycles)},
             {"projected_service_gain_ns",
              std::to_string(
                  command.header.projected_service_gain_ns)},
             {"projected_transfer_and_repack_ns",
              std::to_string(
                  command.header.projected_transfer_and_repack_ns)},
             {"projected_inference_interference_ns",
              std::to_string(
                  command.header.projected_inference_interference_ns)},
             {"projected_net_benefit_ns",
              std::to_string(command.header.projected_net_benefit_ns)},
             {"changed_layers",
              std::to_string(command.header.changed_layers)},
             {"layer_scan_start",
              std::to_string(command.header.layer_scan_start)},
             {"layer_scan_next",
              std::to_string(command.header.layer_scan_next)},
             {"parallel_submission", "true"},
             {"bounded_device_phases", "true"},
             {"resident_external_waits", "0"},
             {"prearmed_cross_device_fanin", "true"},
             {"retirement_attempts", "1"},
             {"retirement_busy_retries", "0"},
             {"blocking_inference", "false"},
             {"policy_owner", "device"}});
        const PerfStatsCollector::Tags movement_tags{
            {"transaction", std::to_string(last_transaction)},
            {"base_epoch", std::to_string(batch.base_epoch)},
            {"candidate_epoch", std::to_string(batch.candidate_epoch)},
            {"policy_owner", "device"},
            {"parallel_submission", "true"},
            {"bounded_device_phases", "true"},
            {"resident_external_waits", "0"},
            {"prearmed_cross_device_fanin", "true"},
            {"blocking_inference", "false"}};
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_movement_commands",
            static_cast<double>(batch.command_count),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_physical_bytes",
            static_cast<double>(batch.packed_weight_bytes),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_promotions",
            static_cast<double>(promotions),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_demotions",
            static_cast<double>(demotions),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_same_priority_moves",
            static_cast<double>(same_priority_moves),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_cross_domain_moves",
            static_cast<double>(cross_domain_moves),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_cross_rank_moves",
            static_cast<double>(cross_rank_moves),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_cross_backend_moves",
            static_cast<double>(cross_backend_moves),
            "maintenance",
            owner->config_.perf_device,
            movement_tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "dynamic_capacity_conservation_certifications",
            1.0,
            "maintenance",
            owner->config_.perf_device,
            {{"transaction", std::to_string(last_transaction)},
             {"candidate_epoch", std::to_string(batch.candidate_epoch)},
             {"edges_checked",
              std::to_string(capacity_evidence.edges_checked)},
             {"participant_coordinates_checked",
              std::to_string(
                  capacity_evidence.participant_coordinates_checked)},
             {"tier_coordinates_checked",
              std::to_string(capacity_evidence.tier_coordinates_checked)},
             {"malformed_edges",
              std::to_string(capacity_evidence.malformed_edges)},
             {"participant_flow_violations",
              std::to_string(
                  capacity_evidence.participant_flow_violations)},
             {"tier_flow_violations",
              std::to_string(capacity_evidence.tier_flow_violations)},
             {"direction_counts_are_capacity_proof", "false"},
             {"blocking_inference", "false"},
             {"policy_owner", "device"}});

        /* Keep one diagnostic row per authenticated physical edge. Aggregate
         * transaction counters prove economy, while these rows retain the
         * exact participant, priority, backend, and rank direction needed to
         * diagnose a parity campaign without reconstructing policy state. */
        for (const auto &migration : batch.migrations)
        {
            const auto *const source_group =
                owner->config_.topology->groupForParticipant(
                    migration.source.owner_participant);
            const auto *const destination_group =
                owner->config_.topology->groupForParticipant(
                    migration.destination.owner_participant);
            if (!source_group || !destination_group)
            {
                return fail(
                    error,
                    "completed device movement lost its frozen topology group");
            }
            const char *direction = "same_priority";
            if (migration.direction ==
                MoEOverlayTierMigrationDirection::Promotion)
            {
                direction = "promotion";
            }
            else if (migration.direction ==
                     MoEOverlayTierMigrationDirection::Demotion)
            {
                direction = "demotion";
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "dynamic_migration_edges",
                1.0,
                "maintenance",
                owner->config_.perf_device,
                {{"transaction", std::to_string(last_transaction)},
                 {"candidate_epoch", std::to_string(batch.candidate_epoch)},
                 {"layer", std::to_string(migration.layer_idx)},
                 {"expert", std::to_string(migration.expert_id)},
                 {"direction", direction},
                 {"movement_axis", movementAxisName(migration.axis)},
                 {"source_participant",
                  std::to_string(migration.source.owner_participant)},
                 {"destination_participant",
                  std::to_string(migration.destination.owner_participant)},
                 {"source_priority",
                  std::to_string(source_group->tier_priority)},
                 {"destination_priority",
                  std::to_string(destination_group->tier_priority)},
                 {"source_device", migration.source.device.to_string()},
                 {"destination_device",
                  migration.destination.device.to_string()},
                 {"source_world_rank",
                  std::to_string(migration.source.owner_world_rank)},
                 {"destination_world_rank",
                  std::to_string(migration.destination.owner_world_rank)},
                 {"estimated_weight_bytes",
                  std::to_string(migration.estimated_weight_bytes)},
                 {"blocking_inference", "false"},
                 {"policy_owner", "device"}});
        }
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::
        restorePreparedContext(std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!owner || !owner->config_.fabric || protocols.empty())
        {
            return fail(
                error,
                "prepared-context restoration lost its controller fabric or transport protocols");
        }

        /* Every non-terminal transaction changes at least one expert owner.
         * A flat owner table therefore bounds the number of movement waves
         * independently of topology, command capacity, and cycles-per-wave.
         * The additional transaction is the mandatory zero-command proof. */
        const std::uint64_t owner_words =
            owner->config_.fabric->layout().header
                .initial_owner_participants_words;
        if (owner_words == 0u)
        {
            return fail(
                error,
                "prepared-context restoration has no immutable initial owner table");
        }
        std::uint64_t movement_waves = 0u;
        for (std::uint64_t attempt = 0u; attempt <= owner_words; ++attempt)
        {
            TransactionResult transaction;
            if (!runOne(
                    TransactionObjective::PreparedContextRestore,
                    &transaction,
                    error))
            {
                return false;
            }
            if (transaction.kind !=
                    MoEOverlayDeviceControllerTransactionKind::
                        PreparedContextRestore ||
                transaction.transaction == 0u ||
                transaction.durable_epoch == 0u)
            {
                return fail(
                    error,
                    "prepared-context restoration returned an invalid typed transaction receipt");
            }
            if (!transaction.movedWeights())
            {
                restoration_movement_waves.store(
                    movement_waves, std::memory_order_release);
                restored_durable_epoch.store(
                    transaction.durable_epoch,
                    std::memory_order_release);
                prepared_context_certified.store(
                    true, std::memory_order_release);
                return true;
            }
            ++movement_waves;
        }
        return fail(
            error,
            "prepared-context restoration exceeded its owner-table progress bound without a zero-command proof");
    }

    std::size_t MoEOverlayDeviceControllerGraphService::localGraphCount()
        const noexcept
    {
        return endpoints_.size();
    }

    bool MoEOverlayDeviceControllerGraphService::ownsLeaderGraph() const
        noexcept
    {
        return std::any_of(
            endpoints_.begin(),
            endpoints_.end(),
            [](const auto &endpoint)
            {
                return endpoint && endpoint->binding.authority_leader;
            });
    }

    void MoEOverlayDeviceControllerGraphService::releaseEndpoint(
        Endpoint &endpoint) noexcept
    {
        if (!endpoint.worker)
            return;
        try
        {
            endpoint.worker->submitAndWait(
                [&endpoint]
                {
                    if (endpoint.in_flight && endpoint.terminal_event)
                    {
                        (void)endpoint.worker->synchronizeEventChecked(
                            endpoint.terminal_event);
                        endpoint.in_flight = false;
                    }
                    endpoint.static_graph.reset();
                    endpoint.dynamic_service_telemetry_snapshot_graph.reset();
                    endpoint.dynamic_histogram_rebase_graph.reset();
                    endpoint.dynamic_begin_prefill_graph.reset();
                    endpoint.dynamic_begin_decode_graph.reset();
                    endpoint.prepared_context_restore_begin_graph.reset();
                    endpoint.dynamic_snapshot_graph.reset();
                    endpoint.prepared_context_restore_snapshot_graph.reset();
                    endpoint.dynamic_group_snapshot_graph.reset();
                    endpoint.dynamic_author_prefill_graph.reset();
                    endpoint.dynamic_author_decode_graph.reset();
                    endpoint.prepared_context_restore_author_graph.reset();
                    endpoint.dynamic_prepare_candidate_graph.reset();
                    endpoint.dynamic_acknowledge_prepared_graph.reset();
                    endpoint.dynamic_begin_commit_graph.reset();
                    endpoint.dynamic_publish_candidate_graph.reset();
                    endpoint.dynamic_acknowledge_published_graph.reset();
                    endpoint.dynamic_publish_admission_graph.reset();
                    endpoint.dynamic_retirement_readiness_graph.reset();
                    endpoint.dynamic_retire_graph.reset();
                    endpoint.dynamic_complete_graph.reset();
                    endpoint.arrival_inbox.reset();
                    endpoint.kernel.reset();
                    if (endpoint.terminal_event)
                    {
                        endpoint.worker->destroyEvent(
                            endpoint.terminal_event);
                        endpoint.terminal_event = nullptr;
                    }
                    if (endpoint.policy_result && endpoint.backend)
                    {
                        endpoint.backend->free(
                            endpoint.policy_result,
                            endpoint.binding.device.gpu_ordinal());
                        endpoint.policy_result = nullptr;
                    }
                    if (endpoint.histogram_phase_baselines &&
                        endpoint.backend)
                    {
                        endpoint.backend->free(
                            endpoint.histogram_phase_baselines,
                            endpoint.binding.device.gpu_ordinal());
                        endpoint.histogram_phase_baselines = nullptr;
                        endpoint.histogram_baseline_words = 0u;
                    }
                });
        }
        catch (...)
        {
            // Destructors cannot recover a poisoned GPU context. Production
            // setup already reports the original failure through the runner.
        }
    }
} // namespace llaminar2
