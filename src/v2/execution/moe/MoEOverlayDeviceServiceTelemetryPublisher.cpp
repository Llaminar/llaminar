/**
 * @file MoEOverlayDeviceServiceTelemetryPublisher.cpp
 * @brief Retained finite GPU-to-mapped-page economy observation graphs.
 */

#include "MoEOverlayDeviceServiceTelemetryPublisher.h"

#include "MoEOverlayServiceTelemetryPublication.h"

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <future>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Typed state of one independently progressing GPU publication. */
        enum class EndpointState : std::uint8_t
        {
            AwaitingBoundary,
            SubmissionQueued,
            PublicationInFlight,
            Stopped,
            Failed,
        };

        /** @return Stable per-participant auxiliary stream identity. */
        std::string streamName(
            const MoEOverlayDeviceControllerRuntimeBinding &binding)
        {
            return "moe_overlay_service_telemetry_participant_" +
                   std::to_string(binding.overlay_participant_id);
        }

        /** @return Stable name for one device service-economy phase plane. */
        const char *servicePhaseName(std::size_t phase) noexcept
        {
            constexpr std::array<const char *,
                                 kDeviceMoEOverlayServicePhaseCount>
                names{"decode", "prefill", "grouped_verifier"};
            return phase < names.size() ? names[phase] : "invalid";
        }
    } // namespace

    /** Model-lifetime resources and typed progress for one local GPU. */
    struct MoEOverlayDeviceServiceTelemetryPublisher::Endpoint
    {
        MoEOverlayDeviceControllerRuntimeBinding binding;
        std::shared_ptr<MappedHostTransferRegion> publication_region;
        MoEOverlayDeviceServiceTelemetryPublicationHeader *host_publication =
            nullptr;
        MoEOverlayDeviceServiceTelemetryPublicationHeader *device_publication =
            nullptr;
        IWorkerGPUContext *worker = nullptr;
        IBackend *backend = nullptr;
        std::unique_ptr<IMoEKernel> kernel;
        std::unique_ptr<IGPUGraphCapture> graph;
        void *stream = nullptr;
        void *terminal_event = nullptr;
        EndpointState state = EndpointState::AwaitingBoundary;
        std::future<void> submission;
        MoEOverlayInferenceBoundaryStatus submission_status =
            MoEOverlayInferenceBoundaryStatus::Deferred;
        bool launch_succeeded = false;
        std::uint64_t last_publication_generation = 0u;
        std::chrono::steady_clock::time_point next_snapshot{};
    };

    MoEOverlayDeviceServiceTelemetryPublisher::
        MoEOverlayDeviceServiceTelemetryPublisher(Config config)
        : config_(std::move(config))
    {
        if (config_.runtime_bindings.empty() ||
            config_.minimum_snapshot_interval.count() <= 0)
        {
            throw std::invalid_argument(
                "GPU service telemetry publisher requires at least one binding and a positive cadence");
        }

        std::sort(
            config_.runtime_bindings.begin(),
            config_.runtime_bindings.end(),
            [](const auto &left, const auto &right)
            {
                return left.overlay_participant_id <
                       right.overlay_participant_id;
            });
        std::unordered_set<int> participants;
        for (const auto &binding : config_.runtime_bindings)
        {
            if (!binding.serviceTelemetryValid() ||
                !binding.inference_boundary ||
                !participants.insert(binding.overlay_participant_id).second)
            {
                throw std::invalid_argument(
                    "GPU service telemetry publisher received an incomplete or duplicate participant binding");
            }
            auto endpoint = std::make_unique<Endpoint>();
            endpoint->binding = binding;
            materializeEndpoint(*endpoint);
            endpoints_.push_back(std::move(endpoint));
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_service_publishers_materialized",
            static_cast<double>(endpoints_.size()),
            "model_setup",
            config_.perf_device,
            {{"authority", "host"},
             {"publication", "mapped_system_scope"},
             {"stream", "dedicated_background"},
             {"blocking_inference", "false"}});
        for (const auto &binding : config_.runtime_bindings)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "device_service_publisher_bindings",
                1.0,
                "model_setup",
                binding.device.toString(),
                {{"participant",
                  std::to_string(binding.overlay_participant_id)},
                 {"telemetry_allocation",
                  std::to_string(reinterpret_cast<std::uintptr_t>(
                      binding.service_telemetry_device))},
                 {"sample_allocation",
                  std::to_string(reinterpret_cast<std::uintptr_t>(
                      binding.service_samples_device))}});
        }
    }

    MoEOverlayDeviceServiceTelemetryPublisher::
        ~MoEOverlayDeviceServiceTelemetryPublisher()
    {
        requestStop();
        for (auto &endpoint : endpoints_)
        {
            if (endpoint)
                releaseEndpoint(*endpoint);
        }
    }

    void MoEOverlayDeviceServiceTelemetryPublisher::materializeEndpoint(
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
                "GPU service telemetry publisher could not resolve its worker, backend, or MoE kernel");
        }

        const std::array<DeviceId, 1u> devices{endpoint.binding.device};
        endpoint.publication_region =
            TransferEngine::instance().allocateMappedHostRegion(
                deviceMoEOverlayServiceTelemetryPublicationBytes(
                    endpoint.binding.layer_count),
                devices);
        if (!endpoint.publication_region ||
            !endpoint.publication_region->isBound())
        {
            throw std::runtime_error(
                "GPU service telemetry publisher could not map its publication page");
        }
        endpoint.host_publication = static_cast<
            MoEOverlayDeviceServiceTelemetryPublicationHeader *>(
            endpoint.publication_region->mutableHostData());
        endpoint.device_publication = static_cast<
            MoEOverlayDeviceServiceTelemetryPublicationHeader *>(
            endpoint.publication_region->deviceAlias(
                endpoint.binding.device));

        endpoint.worker->submitAndWait(
            [&endpoint]
            {
                endpoint.stream =
                    endpoint.worker->getOrCreateAuxiliaryStream(
                        streamName(endpoint.binding));
                endpoint.terminal_event = endpoint.worker->createEvent();
                endpoint.graph =
                    endpoint.worker->createGraphCapture(endpoint.stream);
                const MoEKernelLaunchContext launch{
                    .stream = endpoint.stream};
                if (!endpoint.stream || !endpoint.terminal_event ||
                    !endpoint.graph || !endpoint.graph->beginCapture() ||
                    !endpoint.kernel->publishMoEOverlayServiceTelemetry(
                        launch,
                        endpoint.binding.service_telemetry_device,
                        endpoint.binding.service_samples_device,
                        endpoint.binding.layer_count,
                        endpoint.binding.overlay_participant_id,
                        endpoint.device_publication) ||
                    !endpoint.graph->endCapture() ||
                    !endpoint.graph->instantiate())
                {
                    throw std::runtime_error(
                        "GPU service telemetry publisher could not capture its finite publication graph");
                }
            });
    }

    bool MoEOverlayDeviceServiceTelemetryPublisher::poll(
        std::vector<MoEOverlayDeviceServiceTelemetrySnapshot> *output,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!output)
            return fail("GPU service telemetry publisher requires an output");
        output->clear();
        if (!healthy())
        {
            if (error)
                *error = failureMessage();
            return false;
        }

        try
        {
            const auto now = std::chrono::steady_clock::now();
            for (auto &owned : endpoints_)
            {
                Endpoint &endpoint = *owned;
                switch (endpoint.state)
                {
                case EndpointState::AwaitingBoundary:
                    if (stop_requested_.load(std::memory_order_acquire))
                    {
                        endpoint.state = EndpointState::Stopped;
                        break;
                    }
                    if (now < endpoint.next_snapshot)
                        break;
                    endpoint.submission_status =
                        MoEOverlayInferenceBoundaryStatus::Deferred;
                    endpoint.launch_succeeded = false;
                    endpoint.submission = endpoint.worker->submitAsync(
                        [&endpoint]
                        {
                            endpoint.submission_status = endpoint.binding
                                .inference_boundary
                                ->enqueueMoEOverlayDeviceInferenceBoundary(
                                    endpoint.stream,
                                    MoEOverlayInferenceBoundaryRequest{
                                        .purpose =
                                            MoEOverlayInferenceBoundaryPurpose::
                                                ServiceTelemetrySnapshot,
                                    });
                            if (endpoint.submission_status !=
                                MoEOverlayInferenceBoundaryStatus::Submitted)
                            {
                                return;
                            }
                            endpoint.launch_succeeded =
                                endpoint.graph->launchOnStream(
                                    endpoint.stream) &&
                                endpoint.worker->recordEventChecked(
                                    endpoint.terminal_event,
                                    endpoint.stream);
                            if (!endpoint.launch_succeeded)
                            {
                                endpoint.submission_status =
                                    MoEOverlayInferenceBoundaryStatus::Failed;
                            }
                        });
                    endpoint.state = EndpointState::SubmissionQueued;
                    boundary_submissions_.fetch_add(
                        1u, std::memory_order_relaxed);
                    break;

                case EndpointState::SubmissionQueued:
                    if (!endpoint.submission.valid() ||
                        endpoint.submission.wait_for(
                            std::chrono::seconds(0)) !=
                            std::future_status::ready)
                    {
                        break;
                    }
                    endpoint.submission.get();
                    if (endpoint.submission_status ==
                        MoEOverlayInferenceBoundaryStatus::Deferred)
                    {
                        boundary_deferrals_.fetch_add(
                            1u, std::memory_order_relaxed);
                        endpoint.next_snapshot = now +
                            std::min(
                                config_.minimum_snapshot_interval,
                                std::chrono::milliseconds(2));
                        endpoint.state = EndpointState::AwaitingBoundary;
                        break;
                    }
                    if (endpoint.submission_status !=
                            MoEOverlayInferenceBoundaryStatus::Submitted ||
                        !endpoint.launch_succeeded)
                    {
                        endpoint.state = EndpointState::Failed;
                        return fail(
                            "GPU service telemetry boundary or retained publication launch failed for participant " +
                            std::to_string(
                                endpoint.binding.overlay_participant_id));
                    }
                    publication_launches_.fetch_add(
                        1u, std::memory_order_relaxed);
                    endpoint.state = EndpointState::PublicationInFlight;
                    break;

                case EndpointState::PublicationInFlight:
                {
                    bool ready = false;
                    if (!endpoint.backend->queryEvent(
                            endpoint.terminal_event,
                            endpoint.binding.device.gpu_ordinal(),
                            &ready))
                    {
                        endpoint.state = EndpointState::Failed;
                        return fail(
                            "GPU service telemetry terminal query failed for participant " +
                            std::to_string(
                                endpoint.binding.overlay_participant_id));
                    }
                    if (!ready)
                        break;

                    MoEOverlayDeviceServiceTelemetrySnapshot snapshot;
                    snapshot.participant_id =
                        endpoint.binding.overlay_participant_id;
                    if (!trySnapshotMoEOverlayServiceTelemetryPublication(
                            endpoint.host_publication,
                            endpoint.binding.overlay_participant_id,
                            endpoint.binding.layer_count,
                            &snapshot.rows,
                            &snapshot.publication_generation) ||
                        snapshot.publication_generation <=
                            endpoint.last_publication_generation)
                    {
                        endpoint.state = EndpointState::Failed;
                        return fail(
                            "GPU service telemetry publication was incoherent or non-monotonic for participant " +
                            std::to_string(
                                endpoint.binding.overlay_participant_id));
                    }
                    endpoint.last_publication_generation =
                        snapshot.publication_generation;
                    snapshot.valid_sample_count =
                        endpoint.host_publication->valid_sample_count;
                    snapshot.armed_sample_count =
                        endpoint.host_publication->armed_sample_count;
                    snapshot.begun_sample_count =
                        endpoint.host_publication->begun_sample_count;

                    const std::map<std::string, std::string>
                        cursor_tags{
                            {"participant",
                             std::to_string(
                                 endpoint.binding.overlay_participant_id)},
                            {"publication_generation",
                             std::to_string(
                                 snapshot.publication_generation)},
                        };
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "device_service_snapshot_valid_samples",
                        static_cast<double>(
                            snapshot.valid_sample_count),
                        "maintenance",
                        config_.perf_device,
                        cursor_tags);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "device_service_snapshot_begun_samples",
                        static_cast<double>(
                            snapshot.begun_sample_count),
                        "maintenance",
                        config_.perf_device,
                        cursor_tags);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "device_service_snapshot_armed_samples",
                        static_cast<double>(
                            snapshot.armed_sample_count),
                        "maintenance",
                        config_.perf_device,
                        cursor_tags);

                    /*
                     * Preserve cumulative sample, activation, and rejected-
                     * sample evidence at the passive publication boundary.
                     * The mapped page cannot change again until this endpoint
                     * admits its next retained publication, so these reads are
                     * covered by the accepted seqlock generation above. This
                     * makes an idle participant distinguishable from a graph
                     * whose timing markers ran but rejected every observation,
                     * without downloading any inference-owned state.
                     */
                    const auto *const cells =
                        deviceMoEOverlayServiceTelemetryCells(
                            endpoint.host_publication);
                    for (std::size_t phase = 0u;
                         phase < kDeviceMoEOverlayServicePhaseCount;
                         ++phase)
                    {
                        std::uint64_t samples = 0u;
                        std::uint64_t activations = 0u;
                        std::uint64_t dropped = 0u;
                        for (std::uint32_t layer = 0u;
                             layer < endpoint.binding.layer_count;
                             ++layer)
                        {
                            const auto &cell = cells[
                                static_cast<std::size_t>(layer) *
                                    kDeviceMoEOverlayServicePhaseCount +
                                phase];
                            samples += cell.sample_count;
                            activations += cell.activation_count;
                            dropped += cell.dropped_samples;
                        }
                        const std::map<std::string, std::string> tags{
                            {"participant",
                             std::to_string(
                                 endpoint.binding.overlay_participant_id)},
                            {"publication_generation",
                             std::to_string(
                                 snapshot.publication_generation)},
                            {"source", servicePhaseName(phase)}};
                        PerfStatsCollector::addCounter(
                            "moe_overlay_residency",
                            "device_service_snapshot_samples",
                            static_cast<double>(samples),
                            "maintenance",
                            config_.perf_device,
                            tags);
                        PerfStatsCollector::addCounter(
                            "moe_overlay_residency",
                            "device_service_snapshot_activations",
                            static_cast<double>(activations),
                            "maintenance",
                            config_.perf_device,
                            tags);
                        PerfStatsCollector::addCounter(
                            "moe_overlay_residency",
                            "device_service_snapshot_dropped_samples",
                            static_cast<double>(dropped),
                            "maintenance",
                            config_.perf_device,
                            tags);
                    }
                    endpoint.next_snapshot =
                        now + config_.minimum_snapshot_interval;
                    endpoint.state = stop_requested_.load(
                                             std::memory_order_acquire)
                                         ? EndpointState::Stopped
                                         : EndpointState::AwaitingBoundary;
                    publication_completions_.fetch_add(
                        1u, std::memory_order_relaxed);
                    snapshots_delivered_.fetch_add(
                        1u, std::memory_order_relaxed);
                    output->push_back(std::move(snapshot));
                    break;
                }

                case EndpointState::Stopped:
                    break;

                case EndpointState::Failed:
                    return fail(
                        "GPU service telemetry publisher retained a failed endpoint");
                }
            }
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return fail(
                std::string("GPU service telemetry publication threw: ") +
                exception.what());
        }
        catch (...)
        {
            if (error)
                *error =
                    "GPU service telemetry publication threw an unknown exception";
            return fail(
                "GPU service telemetry publication threw an unknown exception");
        }
        return true;
    }

    void MoEOverlayDeviceServiceTelemetryPublisher::requestStop() noexcept
    {
        stop_requested_.store(true, std::memory_order_release);
    }

    bool MoEOverlayDeviceServiceTelemetryPublisher::healthy() const noexcept
    {
        return healthy_.load(std::memory_order_acquire);
    }

    std::string
    MoEOverlayDeviceServiceTelemetryPublisher::failureMessage() const
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        return failure_message_;
    }

    MoEOverlayDeviceServiceTelemetryPublisherStats
    MoEOverlayDeviceServiceTelemetryPublisher::stats() const noexcept
    {
        return {
            .boundary_submissions = boundary_submissions_.load(
                std::memory_order_relaxed),
            .boundary_deferrals = boundary_deferrals_.load(
                std::memory_order_relaxed),
            .publication_launches = publication_launches_.load(
                std::memory_order_relaxed),
            .publication_completions = publication_completions_.load(
                std::memory_order_relaxed),
            .snapshots_delivered = snapshots_delivered_.load(
                std::memory_order_relaxed),
            .fatal_failures = fatal_failures_.load(
                std::memory_order_relaxed),
        };
    }

    std::size_t
    MoEOverlayDeviceServiceTelemetryPublisher::endpointCount() const noexcept
    {
        return endpoints_.size();
    }

    void MoEOverlayDeviceServiceTelemetryPublisher::releaseEndpoint(
        Endpoint &endpoint) noexcept
    {
        if (!endpoint.worker)
            return;
        try
        {
            if (endpoint.submission.valid())
                endpoint.submission.get();
            const bool publication_submitted =
                endpoint.submission_status ==
                    MoEOverlayInferenceBoundaryStatus::Submitted &&
                endpoint.launch_succeeded;
            endpoint.worker->submitAndWait(
                [&endpoint, publication_submitted]
                {
                    if (publication_submitted &&
                        endpoint.terminal_event)
                    {
                        (void)endpoint.worker->synchronizeEventChecked(
                            endpoint.terminal_event);
                    }
                    endpoint.graph.reset();
                    endpoint.kernel.reset();
                    if (endpoint.terminal_event)
                    {
                        endpoint.worker->destroyEvent(
                            endpoint.terminal_event);
                        endpoint.terminal_event = nullptr;
                    }
                    endpoint.state = EndpointState::Stopped;
                });
        }
        catch (...)
        {
            /* Teardown cannot recover a poisoned device context. The first
             * runtime failure was already retained by fail(). */
        }
        endpoint.publication_region.reset();
        endpoint.host_publication = nullptr;
        endpoint.device_publication = nullptr;
    }

    bool MoEOverlayDeviceServiceTelemetryPublisher::fail(
        std::string message) noexcept
    {
        bool expected = true;
        if (healthy_.compare_exchange_strong(
                expected,
                false,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            {
                std::lock_guard<std::mutex> lock(failure_mutex_);
                failure_message_ = std::move(message);
            }
            fatal_failures_.fetch_add(1u, std::memory_order_relaxed);
            stop_requested_.store(true, std::memory_order_release);
            LOG_ERROR(
                "[ExpertOverlay][Economy] " << failureMessage());
        }
        return false;
    }
} // namespace llaminar2
