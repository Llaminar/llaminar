/**
 * @file MoEOverlayHostAuthorityDeviceBankPublisher.cpp
 * @brief Persistent event-polled CUDA/ROCm inactive-bank publication.
 *
 * Each endpoint progresses through reserve, bounded bank DMA, ready marking,
 * selector publication, and old-bank retirement on its own exact background
 * stream.  Host polling only observes worker futures and terminal events.  It
 * never synchronizes a stream, copies a whole live runtime, or performs device
 * allocation after model setup.
 */

#include "MoEOverlayHostAuthorityDeviceBankPublisher.h"

#include "DeviceMoEExpertDescriptorBuilder.h"
#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayResidencyAuthority.h"
#include "MoERuntimeTable.h"

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Two scalar fields published without overwriting live runtime state. */
        struct RuntimePublicationSelector
        {
            std::uint32_t active_bank = 0u;
            std::uint32_t active_epoch = 0u;
        };

        static_assert(
            offsetof(DeviceMoELayerRuntime, active_bank) == 0u &&
                offsetof(DeviceMoELayerRuntime, active_epoch) ==
                    sizeof(std::uint32_t) &&
                sizeof(RuntimePublicationSelector) ==
                    2u * sizeof(std::uint32_t),
            "host publication must update only the leading runtime selector fields");

        /** Round @p value up to the next @p alignment boundary. */
        [[nodiscard]] constexpr std::size_t alignUp(
            std::size_t value,
            std::size_t alignment) noexcept
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        /** Stable offsets inside one endpoint's model-lifetime mapped page. */
        struct EndpointPageLayout
        {
            std::size_t banks_offset = 0u;
            std::size_t selectors_offset = 0u;
            std::size_t epoch_offset = 0u;
            std::size_t status_offset = 0u;
            std::size_t total_bytes = 0u;

            /** Build a naturally aligned page for the complete layer family. */
            [[nodiscard]] static EndpointPageLayout build(
                std::uint32_t layer_count)
            {
                if (layer_count == 0u)
                    throw std::invalid_argument(
                        "GPU bank publisher page requires at least one layer");
                EndpointPageLayout layout;
                const std::size_t bank_bytes =
                    static_cast<std::size_t>(layer_count) *
                    sizeof(DeviceMoEPlacementBank);
                const std::size_t selector_bytes =
                    static_cast<std::size_t>(layer_count) *
                    sizeof(RuntimePublicationSelector);
                layout.selectors_offset = alignUp(
                    layout.banks_offset + bank_bytes,
                    alignof(RuntimePublicationSelector));
                layout.epoch_offset = alignUp(
                    layout.selectors_offset + selector_bytes,
                    alignof(std::uint64_t));
                layout.status_offset = alignUp(
                    layout.epoch_offset + sizeof(std::uint64_t),
                    alignof(DeviceMoEOverlayEpochStatus));
                layout.total_bytes = layout.status_offset +
                                     sizeof(DeviceMoEOverlayEpochStatus);
                return layout;
            }

            /** Return one layer's staged bank offset. */
            [[nodiscard]] std::size_t bankOffset(
                std::uint32_t layer) const noexcept
            {
                return banks_offset +
                       static_cast<std::size_t>(layer) *
                           sizeof(DeviceMoEPlacementBank);
            }

            /** Return one layer's staged selector offset. */
            [[nodiscard]] std::size_t selectorOffset(
                std::uint32_t layer) const noexcept
            {
                return selectors_offset +
                       static_cast<std::size_t>(layer) *
                           sizeof(RuntimePublicationSelector);
            }
        };

        /** Typed progress of one independently scheduled GPU endpoint. */
        enum class EndpointPhase : std::uint8_t
        {
            HostRecipeReady,
            ReserveQueued,
            ReserveInFlight,
            BankPrepareQueued,
            BankPrepareInFlight,
            Prepared,
            PublicationQueued,
            PublicationInFlight,
            Published,
            AbortQueued,
            AbortInFlight,
            Aborted,
            RetirementQueued,
            RetirementInFlight,
            Retired,
            Failed,
        };

        /** Stable name for one participant's background publication stream. */
        [[nodiscard]] std::string publicationStreamName(
            const MoEOverlayDeviceControllerRuntimeBinding &binding)
        {
            return "moe_overlay_host_authority_bank_participant_" +
                   std::to_string(binding.overlay_participant_id);
        }

        /** Return whether a future's worker submission has completed. */
        [[nodiscard]] bool submissionReady(
            const std::future<void> &submission) noexcept
        {
            return submission.valid() &&
                   submission.wait_for(std::chrono::seconds(0)) ==
                       std::future_status::ready;
        }

        /** Format an exact endpoint diagnostic prefix. */
        [[nodiscard]] std::string endpointPrefix(
            int participant_id,
            DeviceId device)
        {
            return "ExpertOverlay host-authority GPU participant p" +
                   std::to_string(participant_id) + " (" +
                   device.toString() + ") ";
        }
    } // namespace

    /** Persistent model-lifetime resources shared by serial wave proxies. */
    struct MoEOverlayHostAuthorityDeviceBankPublisher::State
    {
        /** One exact GPU publication resource family. */
        struct Endpoint
        {
            MoEOverlayDeviceControllerRuntimeBinding binding;
            EndpointPageLayout layout;
            std::shared_ptr<MappedHostTransferRegion> page;
            IWorkerGPUContext *worker = nullptr;
            IBackend *backend = nullptr;
            std::unique_ptr<IMoEKernel> kernel;
            void *stream = nullptr;
            void *terminal_event = nullptr;
        };

        explicit State(Config config_value)
            : config(std::move(config_value))
        {
            if (!config.registry || config.runtime_bindings.empty())
            {
                throw std::invalid_argument(
                    "Host-authority GPU bank publisher requires a registry and at least one runtime binding");
            }
            if (config.perf_device.empty())
                config.perf_device = "expert_overlay";

            std::sort(
                config.runtime_bindings.begin(),
                config.runtime_bindings.end(),
                [](const auto &left, const auto &right)
                {
                    return left.overlay_participant_id <
                           right.overlay_participant_id;
                });

            std::unordered_set<int> participants;
            for (const auto &binding : config.runtime_bindings)
            {
                const auto endpoint_authority =
                    config.registry->endpoint(binding.overlay_participant_id);
                if (!binding.hostPublicationValid() ||
                    !endpoint_authority ||
                    endpoint_authority->device() != binding.device ||
                    endpoint_authority->numLayers() !=
                        static_cast<int>(binding.layer_count) ||
                    endpoint_authority->numExperts() !=
                        static_cast<int>(binding.expert_count) ||
                    !participants.insert(
                         binding.overlay_participant_id).second)
                {
                    throw std::invalid_argument(
                        "Host-authority GPU bank publisher received an incomplete, duplicate, or registry-inconsistent binding");
                }

                auto endpoint = std::make_unique<Endpoint>();
                endpoint->binding = binding;
                endpoint->layout = EndpointPageLayout::build(
                    binding.layer_count);
                materialize(*endpoint);
                endpoints.push_back(std::move(endpoint));
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "host_authority_device_bank_publishers_materialized",
                static_cast<double>(endpoints.size()),
                "model_setup",
                config.perf_device,
                {{"authority", "host"},
                 {"stream", "background_maintenance"},
                 {"blocking_inference", "false"},
                 {"runtime_copy", "inactive_bank_and_selector_only"}});
        }

        ~State()
        {
            if (transaction_active.load(std::memory_order_acquire))
            {
                LOG_ERROR(
                    "[ExpertOverlay] Destroying the GPU bank publisher with an active epoch transaction");
                std::terminate();
            }
            for (auto &endpoint : endpoints)
            {
                if (endpoint)
                    release(*endpoint);
            }
        }

        /** Allocate one endpoint's mapped page, stream, event, and kernel. */
        void materialize(Endpoint &endpoint)
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
                    endpointPrefix(
                        endpoint.binding.overlay_participant_id,
                        endpoint.binding.device) +
                    "could not resolve its worker, backend, or MoE kernel");
            }

            const std::array<DeviceId, 1u> devices{
                endpoint.binding.device};
            endpoint.page =
                TransferEngine::instance().allocateMappedHostRegion(
                    endpoint.layout.total_bytes,
                    devices);
            if (!endpoint.page || !endpoint.page->isBound())
            {
                throw std::runtime_error(
                    endpointPrefix(
                        endpoint.binding.overlay_participant_id,
                        endpoint.binding.device) +
                    "could not allocate its persistent mapped publication page");
            }
            std::memset(
                endpoint.page->mutableHostData(),
                0,
                endpoint.layout.total_bytes);

            endpoint.worker->submitAndWait(
                [&endpoint]
                {
                    endpoint.stream =
                        endpoint.worker->getOrCreateAuxiliaryStream(
                            publicationStreamName(endpoint.binding),
                            GPUAuxiliaryStreamSchedulingClass::
                                BackgroundMaintenance);
                    endpoint.terminal_event =
                        endpoint.worker->createEvent();
                    if (!endpoint.stream || !endpoint.terminal_event)
                    {
                        throw std::runtime_error(
                            "Host-authority GPU bank publisher could not retain an exact background stream/event");
                    }
                });
        }

        /** Destroy only setup resources after every transaction event is terminal. */
        void release(Endpoint &endpoint) noexcept
        {
            if (!endpoint.worker)
                return;
            try
            {
                endpoint.worker->submitAndWait(
                    [&endpoint]
                    {
                        endpoint.kernel.reset();
                        if (endpoint.terminal_event)
                        {
                            endpoint.worker->destroyEvent(
                                endpoint.terminal_event);
                            endpoint.terminal_event = nullptr;
                        }
                    });
            }
            catch (...)
            {
                /* Teardown cannot recover a poisoned backend context. */
            }
            endpoint.page.reset();
        }

        /** Claim the sole unpublished transaction slot. */
        [[nodiscard]] bool claimTransaction() noexcept
        {
            bool expected = false;
            return transaction_active.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        }

        /** Release the transaction slot only after abort or retirement terminal. */
        void releaseTransaction() noexcept
        {
            bool expected = true;
            if (!transaction_active.compare_exchange_strong(
                    expected,
                    false,
                    std::memory_order_release,
                    std::memory_order_acquire))
            {
                LOG_ERROR(
                    "[ExpertOverlay] GPU bank transaction slot was released twice");
                std::terminate();
            }
        }

        Config config;
        std::vector<std::unique_ptr<Endpoint>> endpoints;
        std::atomic<bool> transaction_active{false};
    };

    namespace
    {
        /** Per-wave state associated with one persistent endpoint. */
        struct DeviceBankEndpointWave
        {
            MoEOverlayHostAuthorityDeviceBankPublisher::State::Endpoint
                *endpoint = nullptr;
            std::vector<DeviceMoERuntimeBankPublicationRecipe> recipes;
            MoEOverlayParticipantBankLease candidate_bank;
            std::future<void> submission;
            EndpointPhase phase = EndpointPhase::HostRecipeReady;
        };

        /**
         * @brief One serial host-authority epoch transaction over all local GPUs.
         *
         * Endpoint loops never return early on Pending: every CUDA/ROCm device
         * gets a chance to advance during each maintenance poll.  This is what
         * makes publication parallel across devices while retaining one global
         * commit barrier.
         */
        class HostAuthorityDeviceBankTransaction final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /**
             * @brief One legal state for the process-local GPU transaction.
             *
             * Publication is the irreversible boundary.  Failures before it
             * remain abortable; failures after it are fatal to the enclosing
             * authority because some device selectors may already name E+1.
             */
            enum class Phase
            {
                Created,
                Preparing,
                Prepared,
                Publishing,
                Published,
                Retiring,
                RetirementReady,
                Aborting,
                Aborted,
                Retired,
                FailedBeforePublication,
                FailedAfterPublication,
            };

            /** Adopt the already-claimed publisher slot and immutable snapshots. */
            HostAuthorityDeviceBankTransaction(
                std::shared_ptr<
                    MoEOverlayHostAuthorityDeviceBankPublisher::State> state,
                std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
                std::shared_ptr<const MoEOverlayResidencySnapshot> candidate)
                : state_(std::move(state)),
                  previous_(std::move(previous)),
                  candidate_(std::move(candidate))
            {
                endpoints_.reserve(state_->endpoints.size());
                for (const auto &endpoint : state_->endpoints)
                {
                    endpoints_.push_back({.endpoint = endpoint.get()});
                }
            }

            /** Require an explicit terminal before releasing persistent pages. */
            ~HostAuthorityDeviceBankTransaction() override
            {
                if (phase_ != Phase::Aborted && phase_ != Phase::Retired)
                {
                    LOG_ERROR(
                        "[ExpertOverlay] GPU bank transaction was destroyed before abort or retirement completed");
                    std::terminate();
                }
            }

            /** Build every host recipe, then submit all reserve kernels. */
            bool beginPrepare(std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ != Phase::Created || !previous_ || !candidate_ ||
                    !candidate_->valid() ||
                    !previous_->valid() ||
                    candidate_->epoch != previous_->epoch + 1u ||
                    candidate_->epoch >
                        std::numeric_limits<std::uint32_t>::max())
                {
                    return fail(
                        "Host-authority GPU bank preparation has an invalid lifecycle or epoch geometry",
                        error);
                }
                phase_ = Phase::Preparing;

                try
                {
                    /*
                     * Finish every CPU-side recipe before the first GPU can
                     * reserve a bank. A validation failure therefore leaves no
                     * partially submitted device transaction to guess about.
                     */
                    for (auto &endpoint : endpoints_)
                        buildHostRecipes(endpoint);
                    for (auto &endpoint : endpoints_)
                        queueReserve(endpoint);
                    return true;
                }
                catch (const std::exception &exception)
                {
                    failure_ = exception.what();
                }
                catch (...)
                {
                    failure_ =
                        "Host-authority GPU bank preparation raised a non-standard exception";
                }
                phase_ = Phase::FailedBeforePublication;
                if (error)
                    *error = failure_;
                return false;
            }

            /** Advance reserve and bank-ready event chains on every endpoint. */
            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ == Phase::Prepared)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Preparing)
                {
                    return failedProgress(
                        "Host-authority GPU bank preparation was not started or was already aborted",
                        error);
                }
                if (!failure_.empty())
                {
                    phase_ = Phase::FailedBeforePublication;
                    return failedProgress(failure_, error);
                }

                bool pending = false;
                try
                {
                    for (auto &endpoint : endpoints_)
                    {
                        const auto progress = advancePreparation(endpoint);
                        pending = pending ||
                                  progress ==
                                      MoEOverlayResidencyWaveProgress::Pending;
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Failed)
                        {
                            phase_ = Phase::FailedBeforePublication;
                            return failedProgress(failure_, error);
                        }
                    }
                }
                catch (const std::exception &exception)
                {
                    phase_ = Phase::FailedBeforePublication;
                    return failedProgress(exception.what(), error);
                }
                catch (...)
                {
                    phase_ = Phase::FailedBeforePublication;
                    return failedProgress(
                        "Host-authority GPU bank preparation poll raised a non-standard exception",
                        error);
                }

                if (pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                phase_ = Phase::Prepared;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "host_authority_device_banks_prepared",
                    static_cast<double>(endpoints_.size()),
                    "maintenance",
                    state_->config.perf_device,
                    {{"previous_epoch", std::to_string(previous_->epoch)},
                     {"candidate_epoch", std::to_string(candidate_->epoch)},
                     {"parallel_endpoints", "true"}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** Submit the irreversible selector kernel on every ready endpoint. */
            bool beginPublication(std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ != Phase::Prepared)
                {
                    return fail(
                        "Host-authority GPU selector publication has an invalid lifecycle",
                        error);
                }

                /*
                 * Set the irreversible state before the first worker submit.
                 * If a later endpoint cannot be queued, the outer residency
                 * authority must terminate rather than roll back a selector
                 * that may already have changed.
                 */
                phase_ = Phase::Publishing;
                try
                {
                    for (auto &endpoint : endpoints_)
                        queuePublication(endpoint);
                    return true;
                }
                catch (const std::exception &exception)
                {
                    failure_ = exception.what();
                    phase_ = Phase::FailedAfterPublication;
                }
                catch (...)
                {
                    failure_ =
                        "Host-authority GPU selector publication raised a non-standard exception";
                    phase_ = Phase::FailedAfterPublication;
                }
                if (error)
                    *error = failure_;
                return false;
            }

            /** Poll all selector terminals and acknowledge their host recipes. */
            MoEOverlayResidencyWaveProgress pollPublication(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ == Phase::Published)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Publishing)
                {
                    return failedProgress(
                        "Host-authority GPU selector publication was not started",
                        error);
                }
                if (!failure_.empty())
                {
                    phase_ = Phase::FailedAfterPublication;
                    return failedProgress(failure_, error);
                }

                bool pending = false;
                try
                {
                    for (auto &endpoint : endpoints_)
                    {
                        const auto progress = advancePublication(endpoint);
                        pending = pending ||
                                  progress ==
                                      MoEOverlayResidencyWaveProgress::Pending;
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Failed)
                        {
                            phase_ = Phase::FailedAfterPublication;
                            return failedProgress(failure_, error);
                        }
                    }
                }
                catch (const std::exception &exception)
                {
                    phase_ = Phase::FailedAfterPublication;
                    return failedProgress(exception.what(), error);
                }
                catch (...)
                {
                    phase_ = Phase::FailedAfterPublication;
                    return failedProgress(
                        "Host-authority GPU publication poll raised a non-standard exception",
                        error);
                }
                if (pending)
                    return MoEOverlayResidencyWaveProgress::Pending;

                phase_ = Phase::Published;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "host_authority_device_selectors_published",
                    static_cast<double>(endpoints_.size()),
                    "maintenance",
                    state_->config.perf_device,
                    {{"candidate_epoch", std::to_string(candidate_->epoch)},
                     {"parallel_endpoints", "true"},
                     {"blocking_inference", "false"}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** Request asynchronous rollback before any selector is submitted. */
            void abort() noexcept override
            {
                if (phase_ == Phase::Aborting || phase_ == Phase::Aborted ||
                    phase_ == Phase::Retired)
                    return;
                if (phase_ == Phase::Publishing ||
                    phase_ == Phase::Published ||
                    phase_ == Phase::Retiring ||
                    phase_ == Phase::RetirementReady ||
                    phase_ == Phase::FailedAfterPublication)
                {
                    LOG_ERROR(
                        "[ExpertOverlay] Attempted to abort after GPU selector publication began");
                    std::terminate();
                }
                phase_ = Phase::Aborting;
            }

            /** Drain submitted preparation, then abort every reserved candidate. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ == Phase::Aborted)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Aborting)
                {
                    return failedProgress(
                        "Host-authority GPU bank abort has an invalid lifecycle",
                        error);
                }

                bool pending = false;
                try
                {
                    for (auto &endpoint : endpoints_)
                    {
                        const auto progress = advanceAbort(endpoint);
                        pending = pending ||
                                  progress ==
                                      MoEOverlayResidencyWaveProgress::Pending;
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Failed)
                        {
                            return failedProgress(failure_, error);
                        }
                    }
                }
                catch (const std::exception &exception)
                {
                    return failedProgress(exception.what(), error);
                }
                catch (...)
                {
                    return failedProgress(
                        "Host-authority GPU abort poll raised a non-standard exception",
                        error);
                }
                if (pending)
                    return MoEOverlayResidencyWaveProgress::Pending;

                completeTerminal(Phase::Aborted);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "host_authority_device_bank_aborts",
                    static_cast<double>(endpoints_.size()),
                    "maintenance",
                    state_->config.perf_device,
                    {{"candidate_epoch", std::to_string(candidate_->epoch)}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** Poll device RCU grace periods without delaying inference readers. */
            MoEOverlayResidencyWaveProgress pollRetirementFence(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ == Phase::RetirementReady)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Published && phase_ != Phase::Retiring)
                {
                    return failedProgress(
                        "Host-authority GPU retirement has an invalid publication lifecycle",
                        error);
                }
                phase_ = Phase::Retiring;

                bool pending = false;
                try
                {
                    for (auto &endpoint : endpoints_)
                    {
                        const auto progress = advanceRetirement(endpoint);
                        pending = pending ||
                                  progress ==
                                      MoEOverlayResidencyWaveProgress::Pending;
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Failed)
                        {
                            phase_ = Phase::FailedAfterPublication;
                            return failedProgress(failure_, error);
                        }
                    }
                }
                catch (const std::exception &exception)
                {
                    phase_ = Phase::FailedAfterPublication;
                    return failedProgress(exception.what(), error);
                }
                catch (...)
                {
                    phase_ = Phase::FailedAfterPublication;
                    return failedProgress(
                        "Host-authority GPU retirement poll raised a non-standard exception",
                        error);
                }
                if (pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                phase_ = Phase::RetirementReady;
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** Release candidate engine leases after every device bank retired. */
            void retirePrevious() noexcept override
            {
                if (phase_ == Phase::Retired)
                    return;
                if (phase_ != Phase::RetirementReady ||
                    std::any_of(
                        endpoints_.begin(),
                        endpoints_.end(),
                        [](const auto &endpoint)
                        {
                            return endpoint.phase != EndpointPhase::Retired;
                        }))
                {
                    LOG_ERROR(
                        "[ExpertOverlay] GPU bank retirement was finalized before every device grace period completed");
                    std::terminate();
                }

                completeTerminal(Phase::Retired);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "host_authority_device_banks_retired",
                    static_cast<double>(endpoints_.size()),
                    "maintenance",
                    state_->config.perf_device,
                    {{"retired_epoch", std::to_string(previous_->epoch)},
                     {"live_epoch", std::to_string(candidate_->epoch)},
                     {"blocking_inference", "false"}});
            }

        private:
            /** Build one complete placement update for one endpoint/layer. */
            [[nodiscard]] MoEPlacementUpdate buildPlacementUpdate(
                const DeviceBankEndpointWave &endpoint,
                const MoEExpertOwnerParticipant &participant,
                int layer) const
            {
                const auto &binding = endpoint.endpoint->binding;
                const auto &bank = *endpoint.candidate_bank;
                MoEPlacementUpdate update;
                update.epoch = static_cast<std::uint32_t>(candidate_->epoch);
                update.expert_count = binding.expert_count;
                update.participant_id = binding.domain_participant_id;
                update.participant_count =
                    binding.domain_participant_count;
                update.experts.resize(binding.expert_count);
                update.local_compute_mask.assign(binding.expert_count, 0u);
                update.replica_role.assign(
                    binding.expert_count,
                    static_cast<std::uint8_t>(
                        DeviceMoEReplicaRole::None));
                update.resident_participant_mask.assign(
                    binding.expert_count,
                    0u);
                update.overlay_route_participant.assign(
                    binding.expert_count,
                    -1);

                const auto &local_layer = bank.layers.at(
                    static_cast<std::size_t>(layer));
                for (std::uint32_t expert = 0u;
                     expert < binding.expert_count;
                     ++expert)
                {
                    const auto *const owner =
                        candidate_->owner_map.ownerFor(
                            layer,
                            static_cast<int>(expert));
                    if (!owner || owner->owner_participant < 0)
                    {
                        throw std::logic_error(
                            endpointPrefix(
                                binding.overlay_participant_id,
                                binding.device) +
                            "candidate owner map has an unowned expert");
                    }

                    auto &descriptor = update.experts[expert];
                    descriptor.logical_expert_id =
                        static_cast<std::int32_t>(expert);
                    update.overlay_route_participant[expert] =
                        owner->owner_participant;

                    /*
                     * Domain-local owner masks cannot use global overlay IDs.
                     * Only owners in this exact named execution domain receive
                     * a dense bit; another tier/domain remains an explicit
                     * sparse route with no local descriptor.
                     */
                    if (owner->domain_name == participant.domain_name)
                    {
                        if (owner->domain_participant_index < 0 ||
                            static_cast<std::uint32_t>(
                                owner->domain_participant_index) >=
                                update.participant_count)
                        {
                            throw std::logic_error(
                                endpointPrefix(
                                    binding.overlay_participant_id,
                                    binding.device) +
                                "candidate owner has an invalid domain-local identity");
                        }
                        descriptor.owner_participant =
                            owner->domain_participant_index;
                        update.resident_participant_mask[expert] =
                            1u << static_cast<std::uint32_t>(
                                owner->domain_participant_index);
                    }
                    else
                    {
                        descriptor.owner_participant = -1;
                    }

                    if (!local_layer.resident_mask[expert])
                        continue;
                    if (owner->owner_participant !=
                        binding.overlay_participant_id)
                    {
                        throw std::logic_error(
                            endpointPrefix(
                                binding.overlay_participant_id,
                                binding.device) +
                            "candidate bank residency disagrees with the global owner map");
                    }

                    const auto &triplet = local_layer.experts[expert];
                    if (!triplet.complete() ||
                        !exportDeviceMoEExpertWeightDescriptors(
                            triplet.gate.get(),
                            triplet.up.get(),
                            triplet.down.get(),
                            descriptor))
                    {
                        throw std::runtime_error(
                            endpointPrefix(
                                binding.overlay_participant_id,
                                binding.device) +
                            "could not export a complete NativeVNNI/FP16/BF16/FP32 expert descriptor");
                    }
                    descriptor.logical_expert_id =
                        static_cast<std::int32_t>(expert);
                    descriptor.owner_participant =
                        static_cast<std::int32_t>(
                            binding.domain_participant_id);
                    descriptor.local_slot =
                        static_cast<std::int32_t>(expert);
                    descriptor.flags = toMoEExpertFlags(
                        DeviceMoEExpertFlags::Valid |
                        DeviceMoEExpertFlags::Resident |
                        DeviceMoEExpertFlags::LocalCompute |
                        DeviceMoEExpertFlags::PreferredOwner);
                    update.local_compute_mask[expert] = 1u;
                    update.replica_role[expert] =
                        static_cast<std::uint8_t>(
                            DeviceMoEReplicaRole::Primary);
                    update.resident_participant_mask[expert] |=
                        1u << binding.domain_participant_id;
                }
                return update;
            }

            /** Resolve candidate leases and stage every immutable bank recipe. */
            void buildHostRecipes(DeviceBankEndpointWave &endpoint)
            {
                auto &resource = *endpoint.endpoint;
                const auto &binding = resource.binding;
                const auto *const participant =
                    candidate_->owner_map.participantForId(
                        binding.overlay_participant_id);
                const auto registry_endpoint = state_->config.registry->endpoint(
                    binding.overlay_participant_id);
                if (!participant || !registry_endpoint ||
                    participant->device != binding.device ||
                    participant->domain_participant_index < 0 ||
                    static_cast<std::uint32_t>(
                        participant->domain_participant_index) !=
                        binding.domain_participant_id)
                {
                    throw std::logic_error(
                        endpointPrefix(
                            binding.overlay_participant_id,
                            binding.device) +
                        "runtime identity disagrees with the candidate topology");
                }

                std::vector<std::uint8_t> domain_ids(
                    binding.domain_participant_count,
                    0u);
                for (const auto &member :
                     candidate_->owner_map.participants())
                {
                    if (member.domain_name != participant->domain_name)
                        continue;
                    if (member.domain_participant_index < 0 ||
                        static_cast<std::uint32_t>(
                            member.domain_participant_index) >=
                            binding.domain_participant_count ||
                        domain_ids[static_cast<std::size_t>(
                            member.domain_participant_index)] != 0u)
                    {
                        throw std::logic_error(
                            endpointPrefix(
                                binding.overlay_participant_id,
                                binding.device) +
                            "candidate domain lacks a dense unique participant namespace");
                    }
                    domain_ids[static_cast<std::size_t>(
                        member.domain_participant_index)] = 1u;
                }
                if (std::any_of(
                        domain_ids.begin(),
                        domain_ids.end(),
                        [](std::uint8_t present) { return present == 0u; }))
                {
                    throw std::logic_error(
                        endpointPrefix(
                            binding.overlay_participant_id,
                            binding.device) +
                        "candidate domain cardinality disagrees with the runtime binding");
                }

                endpoint.candidate_bank = registry_endpoint->acquire(
                    candidate_->epoch);
                if (!endpoint.candidate_bank ||
                    !endpoint.candidate_bank->valid(
                        binding.overlay_participant_id,
                        binding.device,
                        static_cast<int>(binding.layer_count),
                        static_cast<int>(binding.expert_count)))
                {
                    throw std::logic_error(
                        endpointPrefix(
                            binding.overlay_participant_id,
                            binding.device) +
                        "cannot acquire the installed candidate participant bank");
                }

                endpoint.recipes.clear();
                endpoint.recipes.reserve(binding.layer_count);
                for (std::uint32_t layer = 0u;
                     layer < binding.layer_count;
                     ++layer)
                {
                    auto update = buildPlacementUpdate(
                        endpoint,
                        *participant,
                        static_cast<int>(layer));
                    binding.runtime_table_host->prepareInactiveBank(
                        static_cast<int>(layer),
                        update);
                    auto recipe = binding.runtime_table_host
                                      ->preparedInactiveBankPublicationRecipe(
                                          static_cast<int>(layer),
                                          update.epoch);
                    if (!recipe.valid())
                    {
                        throw std::logic_error(
                            endpointPrefix(
                                binding.overlay_participant_id,
                                binding.device) +
                            "returned an invalid inactive-bank recipe");
                    }
                    endpoint.recipes.push_back(recipe);

                    std::memcpy(
                        resource.page->mutableHostData(
                            resource.layout.bankOffset(layer)),
                        recipe.host_bank,
                        sizeof(DeviceMoEPlacementBank));
                    const RuntimePublicationSelector selector{
                        .active_bank = recipe.bank,
                        .active_epoch = recipe.epoch,
                    };
                    std::memcpy(
                        resource.page->mutableHostData(
                            resource.layout.selectorOffset(layer)),
                        &selector,
                        sizeof(selector));
                }

                const std::uint64_t epoch = candidate_->epoch;
                std::memcpy(
                    resource.page->mutableHostData(
                        resource.layout.epoch_offset),
                    &epoch,
                    sizeof(epoch));
                std::memset(
                    resource.page->mutableHostData(
                        resource.layout.status_offset),
                    0,
                    sizeof(DeviceMoEOverlayEpochStatus));
            }

            /** Submit reserve plus semantic-status publication on one stream. */
            void queueReserve(DeviceBankEndpointWave &endpoint)
            {
                auto *const resource = endpoint.endpoint;
                endpoint.submission = resource->worker->submitAsync(
                    [resource]
                    {
                        auto &transfer = TransferEngine::instance();
                        transfer.enqueueMappedHostToPersistentDeviceRegion(
                            *resource->page,
                            resource->layout.epoch_offset,
                            resource->binding.maintenance_epoch,
                            sizeof(std::uint64_t),
                            0u,
                            sizeof(std::uint64_t),
                            resource->binding.device,
                            resource->stream);
                        const MoEKernelLaunchContext launch{
                            .stream = resource->stream};
                        if (!resource->kernel
                                 ->reserveMoEOverlayEpochCandidate(
                                     launch,
                                     resource->binding.epoch_control,
                                     resource->binding.maintenance_epoch,
                                     resource->binding.maintenance_status))
                        {
                            throw std::runtime_error(
                                "GPU candidate reservation kernel enqueue failed");
                        }
                        transfer.enqueuePersistentDeviceRegionToMappedHost(
                            resource->binding.maintenance_status,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            0u,
                            *resource->page,
                            resource->layout.status_offset,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            resource->binding.device,
                            resource->stream);
                        if (!resource->worker->recordEventChecked(
                                resource->terminal_event,
                                resource->stream))
                        {
                            throw std::runtime_error(
                                "GPU candidate reservation terminal event enqueue failed");
                        }
                    });
                endpoint.phase = EndpointPhase::ReserveQueued;
            }

            /** Upload only inactive banks/selectors, then mark the epoch ready. */
            void queueBankPrepare(DeviceBankEndpointWave &endpoint)
            {
                auto *const resource = endpoint.endpoint;
                auto *const wave = &endpoint;
                endpoint.submission = resource->worker->submitAsync(
                    [resource, wave]
                    {
                        auto &transfer = TransferEngine::instance();
                        for (std::uint32_t layer = 0u;
                             layer < resource->binding.layer_count;
                             ++layer)
                        {
                            const auto &recipe = wave->recipes.at(layer);
                            transfer.enqueueMappedHostToPersistentDeviceRegion(
                                *resource->page,
                                resource->layout.bankOffset(layer),
                                recipe.device_bank,
                                sizeof(DeviceMoEPlacementBank),
                                0u,
                                sizeof(DeviceMoEPlacementBank),
                                resource->binding.device,
                                resource->stream);
                        }
                        for (std::uint32_t layer = 0u;
                             layer < resource->binding.layer_count;
                             ++layer)
                        {
                            const auto &recipe = wave->recipes.at(layer);
                            transfer.enqueueMappedHostToPersistentDeviceRegion(
                                *resource->page,
                                resource->layout.selectorOffset(layer),
                                recipe.device_runtime,
                                sizeof(DeviceMoELayerRuntime),
                                0u,
                                sizeof(RuntimePublicationSelector),
                                resource->binding.device,
                                resource->stream);
                        }
                        const MoEKernelLaunchContext launch{
                            .stream = resource->stream};
                        if (!resource->kernel
                                 ->markMoEOverlayEpochCandidateReady(
                                     launch,
                                     resource->binding.epoch_control,
                                     resource->binding.maintenance_epoch,
                                     resource->binding.maintenance_status))
                        {
                            throw std::runtime_error(
                                "GPU candidate-ready kernel enqueue failed");
                        }
                        transfer.enqueuePersistentDeviceRegionToMappedHost(
                            resource->binding.maintenance_status,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            0u,
                            *resource->page,
                            resource->layout.status_offset,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            resource->binding.device,
                            resource->stream);
                        if (!resource->worker->recordEventChecked(
                                resource->terminal_event,
                                resource->stream))
                        {
                            throw std::runtime_error(
                                "GPU candidate-ready terminal event enqueue failed");
                        }
                    });
                endpoint.phase = EndpointPhase::BankPrepareQueued;
            }

            /** Submit the sole device ticket-admission linearization kernel. */
            void queuePublication(DeviceBankEndpointWave &endpoint)
            {
                auto *const resource = endpoint.endpoint;
                endpoint.submission = resource->worker->submitAsync(
                    [resource]
                    {
                        const MoEKernelLaunchContext launch{
                            .stream = resource->stream};
                        if (!resource->kernel
                                 ->publishMoEOverlayEpochCandidate(
                                     launch,
                                     resource->binding.epoch_control,
                                     resource->binding.maintenance_epoch,
                                     resource->binding.maintenance_status))
                        {
                            throw std::runtime_error(
                                "GPU candidate publication kernel enqueue failed");
                        }
                        TransferEngine::instance()
                            .enqueuePersistentDeviceRegionToMappedHost(
                                resource->binding.maintenance_status,
                                sizeof(DeviceMoEOverlayEpochStatus),
                                0u,
                                *resource->page,
                                resource->layout.status_offset,
                                sizeof(DeviceMoEOverlayEpochStatus),
                                resource->binding.device,
                                resource->stream);
                        if (!resource->worker->recordEventChecked(
                                resource->terminal_event,
                                resource->stream))
                        {
                            throw std::runtime_error(
                                "GPU candidate publication terminal event enqueue failed");
                        }
                    });
                endpoint.phase = EndpointPhase::PublicationQueued;
            }

            /** Submit rollback of one reserved but unpublished device bank. */
            void queueAbort(DeviceBankEndpointWave &endpoint)
            {
                auto *const resource = endpoint.endpoint;
                endpoint.submission = resource->worker->submitAsync(
                    [resource]
                    {
                        const MoEKernelLaunchContext launch{
                            .stream = resource->stream};
                        if (!resource->kernel
                                 ->abortMoEOverlayEpochCandidate(
                                     launch,
                                     resource->binding.epoch_control,
                                     resource->binding.maintenance_epoch,
                                     resource->binding.maintenance_status))
                        {
                            throw std::runtime_error(
                                "GPU candidate abort kernel enqueue failed");
                        }
                        TransferEngine::instance()
                            .enqueuePersistentDeviceRegionToMappedHost(
                                resource->binding.maintenance_status,
                                sizeof(DeviceMoEOverlayEpochStatus),
                                0u,
                                *resource->page,
                                resource->layout.status_offset,
                                sizeof(DeviceMoEOverlayEpochStatus),
                                resource->binding.device,
                                resource->stream);
                        if (!resource->worker->recordEventChecked(
                                resource->terminal_event,
                                resource->stream))
                        {
                            throw std::runtime_error(
                                "GPU candidate abort terminal event enqueue failed");
                        }
                    });
                endpoint.phase = EndpointPhase::AbortQueued;
            }

            /** Submit one nonblocking old-reader retirement attempt. */
            void queueRetirement(DeviceBankEndpointWave &endpoint)
            {
                auto *const resource = endpoint.endpoint;
                const std::uint64_t epoch = previous_->epoch;
                std::memcpy(
                    resource->page->mutableHostData(
                        resource->layout.epoch_offset),
                    &epoch,
                    sizeof(epoch));
                endpoint.submission = resource->worker->submitAsync(
                    [resource]
                    {
                        auto &transfer = TransferEngine::instance();
                        transfer.enqueueMappedHostToPersistentDeviceRegion(
                            *resource->page,
                            resource->layout.epoch_offset,
                            resource->binding.maintenance_epoch,
                            sizeof(std::uint64_t),
                            0u,
                            sizeof(std::uint64_t),
                            resource->binding.device,
                            resource->stream);
                        const MoEKernelLaunchContext launch{
                            .stream = resource->stream};
                        if (!resource->kernel->retireMoEOverlayEpoch(
                                launch,
                                resource->binding.epoch_control,
                                resource->binding.maintenance_epoch,
                                resource->binding.maintenance_status))
                        {
                            throw std::runtime_error(
                                "GPU retiring-bank kernel enqueue failed");
                        }
                        transfer.enqueuePersistentDeviceRegionToMappedHost(
                            resource->binding.maintenance_status,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            0u,
                            *resource->page,
                            resource->layout.status_offset,
                            sizeof(DeviceMoEOverlayEpochStatus),
                            resource->binding.device,
                            resource->stream);
                        if (!resource->worker->recordEventChecked(
                                resource->terminal_event,
                                resource->stream))
                        {
                            throw std::runtime_error(
                                "GPU retiring-bank terminal event enqueue failed");
                        }
                    });
                endpoint.phase = EndpointPhase::RetirementQueued;
            }

            /** Convert a queued worker future into an event-polled phase. */
            [[nodiscard]] bool finishSubmission(
                DeviceBankEndpointWave &endpoint,
                EndpointPhase queued,
                EndpointPhase in_flight)
            {
                if (endpoint.phase != queued)
                    return endpoint.phase == in_flight;
                if (!submissionReady(endpoint.submission))
                    return false;
                try
                {
                    endpoint.submission.get();
                }
                catch (...)
                {
                    endpoint.phase = EndpointPhase::Failed;
                    throw;
                }
                endpoint.phase = in_flight;
                return true;
            }

            /** Query the exact terminal event without synchronizing its stream. */
            [[nodiscard]] bool eventReady(
                const DeviceBankEndpointWave &endpoint) const
            {
                bool ready = false;
                const auto &resource = *endpoint.endpoint;
                if (!resource.backend->queryEvent(
                        resource.terminal_event,
                        resource.binding.device.gpu_ordinal(),
                        &ready))
                {
                    throw std::runtime_error(
                        endpointPrefix(
                            resource.binding.overlay_participant_id,
                            resource.binding.device) +
                        "terminal event query failed");
                }
                return ready;
            }

            /** Read status only after its D2H terminal event has completed. */
            [[nodiscard]] const DeviceMoEOverlayEpochStatus &status(
                const DeviceBankEndpointWave &endpoint) const
            {
                return *static_cast<const DeviceMoEOverlayEpochStatus *>(
                    endpoint.endpoint->page->mutableHostData(
                        endpoint.endpoint->layout.status_offset));
            }

            /** Validate one successful semantic operation and exact epoch/bank. */
            [[nodiscard]] bool requireSuccess(
                DeviceBankEndpointWave &endpoint,
                DeviceMoEOverlayEpochOperation operation,
                std::uint64_t epoch,
                std::uint32_t bank)
            {
                const auto &result = status(endpoint);
                if (result.succeeded() &&
                    result.typedOperation() == operation &&
                    result.epoch == epoch && result.bank == bank)
                {
                    return true;
                }
                const auto &binding = endpoint.endpoint->binding;
                failure_ = endpointPrefix(
                               binding.overlay_participant_id,
                               binding.device) +
                           "reported operation=" +
                           std::to_string(result.operation) + " code=" +
                           std::to_string(result.code) + " epoch=" +
                           std::to_string(result.epoch) + " bank=" +
                           std::to_string(result.bank) +
                           " while expecting operation=" +
                           std::to_string(static_cast<std::uint32_t>(
                               operation)) +
                           " epoch=" + std::to_string(epoch) + " bank=" +
                           std::to_string(bank);
                endpoint.phase = EndpointPhase::Failed;
                return false;
            }

            /** Advance one endpoint through reserve and ready marking. */
            MoEOverlayResidencyWaveProgress advancePreparation(
                DeviceBankEndpointWave &endpoint)
            {
                if (endpoint.phase == EndpointPhase::ReserveQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::ReserveQueued,
                        EndpointPhase::ReserveInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::ReserveInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const std::uint32_t bank = endpoint.recipes.front().bank;
                    if (!requireSuccess(
                            endpoint,
                            DeviceMoEOverlayEpochOperation::ReserveCandidate,
                            candidate_->epoch,
                            bank))
                    {
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    queueBankPrepare(endpoint);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::BankPrepareQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::BankPrepareQueued,
                        EndpointPhase::BankPrepareInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::BankPrepareInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const std::uint32_t bank = endpoint.recipes.front().bank;
                    if (!requireSuccess(
                            endpoint,
                            DeviceMoEOverlayEpochOperation::MarkCandidateReady,
                            candidate_->epoch,
                            bank))
                    {
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    endpoint.phase = EndpointPhase::Prepared;
                }
                return endpoint.phase == EndpointPhase::Prepared
                           ? MoEOverlayResidencyWaveProgress::Ready
                           : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** Advance one endpoint through selector publication. */
            MoEOverlayResidencyWaveProgress advancePublication(
                DeviceBankEndpointWave &endpoint)
            {
                if (endpoint.phase == EndpointPhase::PublicationQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::PublicationQueued,
                        EndpointPhase::PublicationInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::PublicationInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const std::uint32_t bank = endpoint.recipes.front().bank;
                    if (!requireSuccess(
                            endpoint,
                            DeviceMoEOverlayEpochOperation::PublishCandidate,
                            candidate_->epoch,
                            bank))
                    {
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    for (std::uint32_t layer = 0u;
                         layer < endpoint.endpoint->binding.layer_count;
                         ++layer)
                    {
                        endpoint.endpoint->binding.runtime_table_host
                            ->acknowledgeDevicePublishedBank(
                                static_cast<int>(layer),
                                static_cast<std::uint32_t>(
                                    candidate_->epoch),
                                bank);
                    }
                    endpoint.phase = EndpointPhase::Published;
                }
                return endpoint.phase == EndpointPhase::Published
                           ? MoEOverlayResidencyWaveProgress::Ready
                           : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** Progress in-flight preparation to a safe abort edge. */
            MoEOverlayResidencyWaveProgress advanceAbort(
                DeviceBankEndpointWave &endpoint)
            {
                if (endpoint.phase == EndpointPhase::HostRecipeReady)
                {
                    endpoint.phase = EndpointPhase::Aborted;
                }
                if (endpoint.phase == EndpointPhase::ReserveQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::ReserveQueued,
                        EndpointPhase::ReserveInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::ReserveInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const auto &result = status(endpoint);
                    if (result.succeeded() &&
                        result.typedOperation() ==
                            DeviceMoEOverlayEpochOperation::ReserveCandidate)
                    {
                        queueAbort(endpoint);
                        return MoEOverlayResidencyWaveProgress::Pending;
                    }
                    /* Busy/invalid reserve created no candidate to reclaim. */
                    endpoint.phase = EndpointPhase::Aborted;
                }
                if (endpoint.phase == EndpointPhase::BankPrepareQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::BankPrepareQueued,
                        EndpointPhase::BankPrepareInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::BankPrepareInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    queueAbort(endpoint);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::Prepared)
                {
                    queueAbort(endpoint);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::AbortQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::AbortQueued,
                        EndpointPhase::AbortInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::AbortInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const std::uint32_t bank = endpoint.recipes.front().bank;
                    if (!requireSuccess(
                            endpoint,
                            DeviceMoEOverlayEpochOperation::AbortCandidate,
                            candidate_->epoch,
                            bank))
                    {
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    endpoint.phase = EndpointPhase::Aborted;
                }
                return endpoint.phase == EndpointPhase::Aborted
                           ? MoEOverlayResidencyWaveProgress::Ready
                           : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** Advance one nonblocking old-reader retirement attempt. */
            MoEOverlayResidencyWaveProgress advanceRetirement(
                DeviceBankEndpointWave &endpoint)
            {
                if (endpoint.phase == EndpointPhase::Published)
                {
                    queueRetirement(endpoint);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::RetirementQueued &&
                    !finishSubmission(
                        endpoint,
                        EndpointPhase::RetirementQueued,
                        EndpointPhase::RetirementInFlight))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint.phase == EndpointPhase::RetirementInFlight)
                {
                    if (!eventReady(endpoint))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    const auto &result = status(endpoint);
                    if (result.typedOperation() !=
                            DeviceMoEOverlayEpochOperation::Retire ||
                        result.epoch != previous_->epoch)
                    {
                        failure_ = endpointPrefix(
                                       endpoint.endpoint->binding
                                           .overlay_participant_id,
                                       endpoint.endpoint->binding.device) +
                                   "reported malformed retirement status";
                        endpoint.phase = EndpointPhase::Failed;
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    if (result.typedCode() ==
                        DeviceMoEOverlayEpochStatusCode::Busy)
                    {
                        /* A live inference reader owns the old bank. Retry on
                         * a later maintenance poll without delaying it. */
                        endpoint.phase = EndpointPhase::Published;
                        return MoEOverlayResidencyWaveProgress::Pending;
                    }
                    const std::uint32_t retiring_bank =
                        1u - endpoint.recipes.front().bank;
                    if (!requireSuccess(
                            endpoint,
                            DeviceMoEOverlayEpochOperation::Retire,
                            previous_->epoch,
                            retiring_bank))
                    {
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    endpoint.phase = EndpointPhase::Retired;
                }
                return endpoint.phase == EndpointPhase::Retired
                           ? MoEOverlayResidencyWaveProgress::Ready
                           : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** Latch the first failure while preserving its exact diagnostic. */
            bool fail(const std::string &message, std::string *error) noexcept
            {
                if (failure_.empty())
                    failure_ = message;
                if (error)
                    *error = failure_;
                return false;
            }

            /** Return Failed and retain the first exact diagnostic. */
            MoEOverlayResidencyWaveProgress failedProgress(
                const std::string &message,
                std::string *error) noexcept
            {
                (void)fail(message, error);
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** Release exact bank leases and the publisher's serial slot once. */
            void completeTerminal(Phase terminal_phase) noexcept
            {
                if (phase_ == terminal_phase)
                    return;
                if (terminal_phase != Phase::Aborted &&
                    terminal_phase != Phase::Retired)
                    std::terminate();
                for (auto &endpoint : endpoints_)
                {
                    endpoint.candidate_bank.reset();
                    endpoint.recipes.clear();
                }
                phase_ = terminal_phase;
                state_->releaseTransaction();
            }

            std::shared_ptr<
                MoEOverlayHostAuthorityDeviceBankPublisher::State>
                state_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate_;
            std::vector<DeviceBankEndpointWave> endpoints_;
            std::string failure_;
            Phase phase_ = Phase::Created;
        };
    } // namespace

    MoEOverlayHostAuthorityDeviceBankPublisher::
        MoEOverlayHostAuthorityDeviceBankPublisher(Config config)
        : state_(std::make_shared<State>(std::move(config)))
    {
    }

    MoEOverlayHostAuthorityDeviceBankPublisher::
        ~MoEOverlayHostAuthorityDeviceBankPublisher() = default;

    std::unique_ptr<IMoEOverlayInactiveBankTransaction>
    MoEOverlayHostAuthorityDeviceBankPublisher::createTransaction(
        std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
        std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!state_ || !previous || !candidate || !previous->valid() ||
            !candidate->valid() || candidate->epoch != previous->epoch + 1u)
        {
            if (error)
            {
                *error =
                    "Host-authority GPU bank publisher requires adjacent valid snapshots";
            }
            return nullptr;
        }
        if (!state_->claimTransaction())
        {
            if (error)
            {
                *error =
                    "Host-authority GPU bank publisher already owns an unpublished transaction";
            }
            return nullptr;
        }

        try
        {
            return std::make_unique<HostAuthorityDeviceBankTransaction>(
                state_,
                std::move(previous),
                std::move(candidate));
        }
        catch (const std::exception &exception)
        {
            state_->releaseTransaction();
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            state_->releaseTransaction();
            if (error)
            {
                *error =
                    "Host-authority GPU bank transaction construction raised a non-standard exception";
            }
        }
        return nullptr;
    }

    std::size_t MoEOverlayHostAuthorityDeviceBankPublisher::endpointCount()
        const noexcept
    {
        return state_ ? state_->endpoints.size() : 0u;
    }
} // namespace llaminar2
