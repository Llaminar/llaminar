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

    std::uint64_t MoEOverlayMaintenanceBoundaryGate::growRequiredTokens(
        std::uint64_t maximum_tokens,
        double growth_factor) noexcept
    {
        const std::uint64_t current =
            required_tokens_.load(std::memory_order_acquire);
        if (maximum_tokens == 0u || maximum_tokens <= current ||
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
        /** Participant-local demand publication with no peer dependency. */
        std::unique_ptr<IGPUGraphCapture> dynamic_snapshot_prefill_graph;
        std::unique_ptr<IGPUGraphCapture> dynamic_snapshot_decode_graph;
        /** Group-root reduction admitted after every participant receipt. */
        std::unique_ptr<IGPUGraphCapture> dynamic_group_snapshot_graph;
        /** Sole-leader policy nodes launched only after every group receipt. */
        std::unique_ptr<IGPUGraphCapture> dynamic_author_prefill_graph;
        std::unique_ptr<IGPUGraphCapture> dynamic_author_decode_graph;
        /** Physical-ready E -> E+1 publication and retirement admission. */
        std::unique_ptr<IGPUGraphCapture> dynamic_publish_graph;
        /** Reader-ticket-selected participant-local bank reclamation. */
        std::unique_ptr<IGPUGraphCapture> dynamic_retire_graph;
        /** Physical-retirement-selected topology terminal publication. */
        std::unique_ptr<IGPUGraphCapture> dynamic_complete_graph;
        void *stream = nullptr;
        void *terminal_event = nullptr;
        MoEOverlayDeviceControllerPolicyResult *policy_result = nullptr;
        /** Two disjoint `[layer][expert]` cumulative phase baselines. */
        std::uint64_t *histogram_phase_baselines = nullptr;
        std::size_t histogram_baseline_words = 0u;
        bool in_flight = false;
        bool terminal_ready = false;
    };

    /** Process-local follower for one device-owned physical transaction. */
    struct MoEOverlayDeviceControllerGraphService::DynamicWorker
    {
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
        /** Monotonic owner/worker proof that shutdown has no admitted work. */
        MoEOverlayWorkerDrainProtocol drain;
        mutable std::mutex failure_mutex;
        std::string failure;
        std::uint64_t last_transaction = 0u;
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

        /** Run coalesced complete device/physical transactions until stopped. */
        void run(std::stop_token stop_token) noexcept;

        /** Execute one complete device-authored transaction. */
        [[nodiscard]] bool runOne(
            MoEOverlayInferencePhase phase,
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
            MoEOverlayInferencePhase *phase,
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
            dynamic_worker_->thread = std::jthread(
                [worker = dynamic_worker_.get()](std::stop_token token)
                {
                    worker->run(token);
                });
        }
    }

    MoEOverlayDeviceControllerGraphService::
        ~MoEOverlayDeviceControllerGraphService()
    {
        if (dynamic_worker_)
        {
            /*
             * Every rank first closes ordinary inference. Background workers
             * remain live across this barrier, allowing a lagging group to
             * join a transaction that a faster peer already admitted. A
             * local jthread cancellation is not a distributed protocol edge.
             */
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
            dynamic_worker_.reset();
        }
        for (auto &endpoint : endpoints_)
        {
            if (endpoint)
                releaseEndpoint(*endpoint);
        }
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
                                (2u * sizeof(std::uint64_t)))
                    {
                        throw std::overflow_error(
                            "Device controller histogram baseline geometry overflowed size_t");
                    }
                    endpoint.histogram_phase_baselines = static_cast<
                        std::uint64_t *>(endpoint.backend->allocate(
                        2u * endpoint.histogram_baseline_words *
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
                         2u * endpoint.histogram_baseline_words *
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
                                               std::uint32_t source_mask,
                                               std::uint64_t *baseline)
                {
                    // Packing and publication share the participant's exact
                    // retained stream. The mapped participant record is the
                    // release edge consumed by the group root.
                    return endpoint.kernel->packDeviceRebalanceHistograms(
                        launch,
                        endpoint.runtime_binding.runtime_layers_device,
                        endpoint.binding.participant_collected_state,
                        endpoint.snapshot_config,
                        /*wave_state=*/nullptr,
                        /*controller_state=*/nullptr,
                        /*command_buffer_count=*/1u,
                        source_mask,
                        baseline);
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
                                    moe_runtime_abi::
                                        kAllHistogramSourcesMask,
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

                const std::uint32_t prefill_source_mask =
                    moe_runtime_abi::histogramSourceBit(
                        moe_runtime_abi::HistogramSource::Prefill);
                const std::uint32_t decode_source_mask =
                    moe_runtime_abi::histogramSourceBit(
                        moe_runtime_abi::HistogramSource::Decode) |
                    moe_runtime_abi::histogramSourceBit(
                        moe_runtime_abi::HistogramSource::GroupedVerifier);
                std::uint64_t *const prefill_baseline =
                    endpoint.histogram_phase_baselines;
                std::uint64_t *const decode_baseline =
                    endpoint.histogram_phase_baselines +
                    endpoint.histogram_baseline_words;
                const auto capture_decision_family = [&] (
                    std::unique_ptr<IGPUGraphCapture> &begin_graph,
                    std::unique_ptr<IGPUGraphCapture> &snapshot_graph,
                    std::unique_ptr<IGPUGraphCapture> &author_graph,
                    MoEOverlayDeviceDemandPhase demand_phase,
                    std::uint32_t source_mask,
                    std::uint64_t *baseline,
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
                    capture(
                        snapshot_graph,
                        [&]
                        {
                            return pack_snapshot(source_mask, baseline) &&
                                   enqueue(
                                       MoEOverlayDeviceControllerAction::
                                           PublishParticipantSnapshot);
                        },
                        phase_name);
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
                    endpoint.dynamic_histogram_rebase_graph,
                    [&]
                    {
                        /* Each pack advances its device-resident cumulative
                         * baseline. The packed scratch row is intentionally
                         * unpublished: certification traffic is valid economy
                         * evidence, but must not become placement demand. */
                        return pack_snapshot(
                                   prefill_source_mask,
                                   prefill_baseline) &&
                               pack_snapshot(
                                   decode_source_mask,
                                   decode_baseline);
                    },
                    "Dynamic calibration-demand histogram rebase");
                capture_decision_family(
                    endpoint.dynamic_begin_prefill_graph,
                    endpoint.dynamic_snapshot_prefill_graph,
                    endpoint.dynamic_author_prefill_graph,
                    MoEOverlayDeviceDemandPhase::Prefill,
                    prefill_source_mask,
                    prefill_baseline,
                    "Dynamic bounded prefill decision phase");
                capture_decision_family(
                    endpoint.dynamic_begin_decode_graph,
                    endpoint.dynamic_snapshot_decode_graph,
                    endpoint.dynamic_author_decode_graph,
                    MoEOverlayDeviceDemandPhase::Decode,
                    decode_source_mask,
                    decode_baseline,
                    "Dynamic bounded decode decision phase");
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

                capture(
                    endpoint.dynamic_publish_graph,
                    [&]
                    {
                        bool captured = enqueue(
                            MoEOverlayDeviceControllerAction::
                                ApplyRuntimeCandidate);
                        if (endpoint.binding.group_root)
                        {
                            captured = captured && enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgePrepared);
                        }
                        if (endpoint.binding.authority_leader)
                        {
                            captured = captured && enqueue(
                                MoEOverlayDeviceControllerAction::BeginCommit);
                        }
                        captured = captured && enqueue(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeCandidate);
                        if (endpoint.binding.group_root)
                        {
                            captured = captured && enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgePublished);
                        }
                        if (endpoint.binding.authority_leader)
                        {
                            captured = captured &&
                                enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        PublishAdmission) &&
                                enqueue(
                                    MoEOverlayDeviceControllerAction::
                                        BeginDynamicRetirement);
                        }
                        /* Unlike the inference-release epilogue, this probe is
                         * part of the already-prearmed Publish epoch. Join the
                         * leader's retirement-open edge before probing so a
                         * fast follower cannot no-op before Retiring and then
                         * wait forever for inference that may never recur. */
                        return captured &&
                            enqueue(
                                MoEOverlayDeviceControllerAction::
                                    AwaitRuntimeRetirement) &&
                            enqueue(
                                MoEOverlayDeviceControllerAction::
                                    PublishRuntimeRetirementReadiness);
                    },
                    "Dynamic bounded publication epoch");

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
            case DynamicGraphEpoch::SnapshotPrefillDemand:
                return "snapshot-prefill-demand";
            case DynamicGraphEpoch::SnapshotDecodeDemand:
                return "snapshot-decode-demand";
            case DynamicGraphEpoch::PublishGroupSnapshot:
                return "publish-group-snapshot";
            case DynamicGraphEpoch::AuthorPrefillDecision:
                return "author-prefill-decision";
            case DynamicGraphEpoch::AuthorDecodeDecision:
                return "author-decode-decision";
            case DynamicGraphEpoch::Publish:
                return "publish";
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
            case DynamicGraphEpoch::SnapshotPrefillDemand:
                graph = endpoint.dynamic_snapshot_prefill_graph.get();
                break;
            case DynamicGraphEpoch::SnapshotDecodeDemand:
                graph = endpoint.dynamic_snapshot_decode_graph.get();
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
            case DynamicGraphEpoch::Publish:
                graph = endpoint.dynamic_publish_graph.get();
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
                epoch == DynamicGraphEpoch::PublishGroupSnapshot ||
                epoch == DynamicGraphEpoch::AuthorPrefillDecision ||
                epoch == DynamicGraphEpoch::AuthorDecodeDecision;
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
                epoch == DynamicGraphEpoch::SnapshotDecodeDemand;
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
        std::string *error) noexcept
    {
        for (auto &endpoint : endpoints_)
        {
            if (!endpoint || !endpoint->arrival_inbox ||
                !endpoint->arrival_inbox->finishWave(error))
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
            std::lock_guard<std::mutex> lock(
                dynamic_worker_->failure_mutex);
            dynamic_worker_->failure = std::move(message);
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
                    drain.acknowledge(drain.currentRequest());
                    wake_cv.notify_all();
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
                MoEOverlayInferencePhase authority_phase =
                    MoEOverlayInferencePhase::Prefill;
                std::string ticket_error;
                if (!observeAuthorityTransaction(
                        &transaction, &authority_phase, &ticket_error))
                {
                    owner->failDynamic(
                        ticket_error.empty()
                            ? "device controller follower observed a divergent authority ticket"
                            : std::move(ticket_error));
                    break;
                }
                if (transaction != 0u)
                {
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
                    drain.acknowledge(drain.currentRequest());
                    wake_cv.notify_all();
                }
                continue;
            }
            std::string error;
            if (!runOne(window.phase, &error))
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
                boundary_gate.growRequiredTokens(
                    maximum_window_tokens, window_growth_factor);
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
                drain.acknowledge(drain.currentRequest());
                wake_cv.notify_all();
            }
        }
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
            MoEOverlayInferencePhase *phase,
            std::string *error) const noexcept
    {
        if (transaction)
            *transaction = 0u;
        if (phase)
            *phase = MoEOverlayInferencePhase::Prefill;
        if (!transaction || !phase || protocols.empty())
        {
            return fail(
                error,
                "Dynamic authority-ticket observation has incomplete outputs or no local protocol");
        }

        std::uint64_t observed_transaction = 0u;
        MoEOverlayDeviceDemandPhase observed_phase =
            MoEOverlayDeviceDemandPhase::Invalid;
        for (const auto &protocol : protocols)
        {
            std::uint64_t candidate_transaction = 0u;
            MoEOverlayDeviceDemandPhase candidate_phase =
                MoEOverlayDeviceDemandPhase::Invalid;
            if (!protocol->snapshotTransactionAfter(
                    last_transaction,
                    &candidate_transaction,
                    &candidate_phase))
            {
                return true;
            }
            if (observed_transaction == 0u)
            {
                observed_transaction = candidate_transaction;
                observed_phase = candidate_phase;
            }
            else if (candidate_transaction != observed_transaction ||
                     candidate_phase != observed_phase)
            {
                return fail(
                    error,
                    "Dynamic local group roots observed divergent transaction or phase tickets");
            }
        }

        *transaction = observed_transaction;
        *phase = observed_phase == MoEOverlayDeviceDemandPhase::Prefill
            ? MoEOverlayInferencePhase::Prefill
            : MoEOverlayInferencePhase::Decode;
        return true;
    }

    bool MoEOverlayDeviceControllerGraphService::DynamicWorker::runOne(
        MoEOverlayInferencePhase phase,
        std::string *error) noexcept
    {
        const char *const inference_phase =
            phase == MoEOverlayInferencePhase::Prefill
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
        if (!owner || protocols.empty() ||
            (phase != MoEOverlayInferencePhase::Prefill &&
             phase != MoEOverlayInferencePhase::Decode) ||
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
        const auto begin_epoch =
            phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::BeginPrefillDecision
                : DynamicGraphEpoch::BeginDecodeDecision;
        if (!owner->launchDynamicEpoch(begin_epoch, error) ||
            !wait_terminals(
                phase == MoEOverlayInferencePhase::Prefill
                    ? "Dynamic bounded prefill transaction-open epoch"
                    : "Dynamic bounded decode transaction-open epoch"))
        {
            return false;
        }

        std::uint64_t decision_transaction = 0u;
        MoEOverlayInferencePhase authority_phase = phase;
        bool observation_failed = false;
        if (!wait_protocol(
                [&]
                {
                    std::string observation_error;
                    if (!observeAuthorityTransaction(
                            &decision_transaction,
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
            observation_failed || authority_phase != phase)
        {
            return observation_failed || (error && !error->empty())
                ? false
                : fail(
                      error,
                      "Dynamic transaction-open ticket selected the wrong inference phase");
        }

        const auto snapshot_epoch =
            phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::SnapshotPrefillDemand
                : DynamicGraphEpoch::SnapshotDecodeDemand;
        if (!owner->launchDynamicEpoch(snapshot_epoch, error) ||
            !wait_terminals(
                phase == MoEOverlayInferencePhase::Prefill
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
                phase == MoEOverlayInferencePhase::Prefill
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

        const auto author_epoch =
            phase == MoEOverlayInferencePhase::Prefill
                ? DynamicGraphEpoch::AuthorPrefillDecision
                : DynamicGraphEpoch::AuthorDecodeDecision;
        if (!owner->launchDynamicEpoch(author_epoch, error) ||
            !wait_terminals(
                phase == MoEOverlayInferencePhase::Prefill
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
            return true;
        }

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
        bool inboxes_enqueued = false;
        bool physical_published = false;
        const auto unwind = [&](std::string message)
        {
            for (auto &protocol : protocols)
                protocol->fail();
            if (!physical_published)
                abortPrepared(batch, prepared);
            if (graph_phase_in_flight)
                (void)wait_terminals("failed movement unwind");
            if (inboxes_enqueued)
            {
                const auto deadline = protocolDeadline();
                bool inboxes_ready = false;
                while (std::chrono::steady_clock::now() < deadline)
                {
                    inboxes_ready = true;
                    for (auto &endpoint : owner->endpoints_)
                    {
                        bool ready = false;
                        std::string ignored;
                        if (!endpoint->arrival_inbox->queryReady(
                                &ready, &ignored))
                        {
                            inboxes_ready = false;
                            break;
                        }
                        inboxes_ready = inboxes_ready && ready;
                    }
                    if (inboxes_ready)
                        break;
                    pollPause();
                }
                if (inboxes_ready)
                {
                    std::string ignored;
                    (void)owner->finishDynamicInboxes(&ignored);
                }
            }
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
            return unwind(
                "device physical transfers exceeded the protocol deadline");
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

        // Submit every descriptor DMA and event edge before launching any
        // participant continuation. These worker futures cover API submission
        // only; the copies remain asynchronous on their dedicated streams.
        std::vector<std::future<void>> inbox_submissions;
        inbox_submissions.reserve(owner->endpoints_.size());
        try
        {
            for (auto &owned_endpoint : owner->endpoints_)
            {
                auto &endpoint = *owned_endpoint;
                inbox_submissions.push_back(endpoint.worker->submitAsync(
                    [&endpoint, &batch, &prepared]
                    {
                        std::string inbox_error;
                        if (!endpoint.arrival_inbox ||
                            !endpoint.arrival_inbox->enqueue(
                                batch, prepared, &inbox_error) ||
                            !endpoint.arrival_inbox->enqueueDependency(
                                &inbox_error))
                        {
                            throw std::runtime_error(
                                inbox_error.empty()
                                    ? "participant descriptor/event submission failed"
                                    : std::move(inbox_error));
                        }
                    }));
            }
            for (auto &submission : inbox_submissions)
                submission.get();
            inboxes_enqueued = true;
        }
        catch (const std::exception &exception)
        {
            return unwind(
                std::string("device prepared-arrival submission failed: ") +
                exception.what());
        }

        const auto descriptor_deadline = protocolDeadline();
        bool descriptors_ready = false;
        while (std::chrono::steady_clock::now() < descriptor_deadline)
        {
            descriptors_ready = true;
            for (auto &endpoint : owner->endpoints_)
            {
                bool ready = false;
                std::string query_error;
                if (!endpoint->arrival_inbox->queryReady(
                        &ready, &query_error))
                {
                    return unwind(
                        query_error.empty()
                            ? "descriptor event query failed"
                            : std::move(query_error));
                }
                descriptors_ready = descriptors_ready && ready;
            }
            if (descriptors_ready)
                break;
            pollPause();
        }
        if (!descriptors_ready)
            return unwind("prepared descriptor DMA exceeded the protocol deadline");

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

        /* Completed destination bytes and descriptor DMA are already retained.
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
        if (!run_epoch(
                DynamicGraphEpoch::Publish,
                "Dynamic bounded publication epoch"))
        {
            return unwind(
                error && !error->empty()
                    ? *error
                    : "device publication epoch did not terminate");
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
        if (!owner->finishDynamicInboxes(error))
        {
            return false;
        }
        last_transaction = command.header.transaction_id;

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
                    endpoint.dynamic_snapshot_prefill_graph.reset();
                    endpoint.dynamic_snapshot_decode_graph.reset();
                    endpoint.dynamic_group_snapshot_graph.reset();
                    endpoint.dynamic_author_prefill_graph.reset();
                    endpoint.dynamic_author_decode_graph.reset();
                    endpoint.dynamic_publish_graph.reset();
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
