/**
 * @file MappedTransferProgressEpoch.cpp
 * @brief Event-polled asynchronous copy implementation for expert movement.
 *
 * Setup owns permanent topology command identities and a separately bounded
 * pool of background stream/event execution lanes. Maintenance
 * release-publishes a bounded device region, the exact GPU worker assigns it to
 * a free lane and enqueues a TransferEngine copy, and a later non-blocking event
 * query acquires completion. CUDA also owns graph-bounded copy branches whose
 * GPU cursor survives intervals: graph retirement never waits for a complete
 * maintenance command. ROCm retains native asynchronous SDMA progress.
 */

#include "MappedTransferProgressEpoch.h"

#include "TransferEngine.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /**
     * @brief One cache-owned CUDA interval with an explicit fork/close/join DAG.
     *
     * Only the primary graph opens/closes its private word. The worker holds no
     * host lifetime ticket and releases partial claims before its terminal edge.
     * Epoch storage outlives every captured pointer because the branch retains
     * the epoch; the graph cache retires executables before releasing the branch.
     */
    class MappedTransferProgressEpoch::CapturedBranch final
        : public IGraphCaptureAuxiliaryBranch
    {
    public:
        /** @brief Prepare graph-only Open, worker and Close on the resource owner. */
        explicit CapturedBranch(std::shared_ptr<MappedTransferProgressEpoch> epoch)
            : epoch_(std::move(epoch))
        {
            auto &context = *epoch_->context_;
            context.submitAndWait([&]
            {
                TransferEngine transfers;
                interval_ = transfers.allocateDeviceTransferBuffer(sizeof(std::uint32_t), device());
                // All source fragments are cold, graph-only recordings. They
                // share one setup stream and consume no runtime queue at replay.
                void *stream = context.getOrCreateAuxiliaryStream(
                    epoch_->config_.name + "_captured_progress",
                    GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance);
                if (!stream)
                    throw std::runtime_error("Mapped transfer branch has no setup stream");
                for (std::size_t index = 0u; index < fragments_.size(); ++index)
                {
                    fragments_[index] = context.createGraphCapture(stream);
                    if (!fragments_[index])
                        throw std::runtime_error("Mapped transfer branch fragment allocation failed");
                    ScopedBackendGraphCapture capture(
                        context, *fragments_[index], "mapped transfer branch fragment");
                    if (!capture.begin())
                        throw std::runtime_error("Mapped transfer branch fragment recording failed");
                    if (index == 1u)
                        transfers.enqueueMappedTransferService(*epoch_->service_inbox_,
                            *epoch_->service_cursors_, epoch_->maximumBytes(), interval_.get(),
                            MappedTransferServiceRun::CapturedInterval, stream);
                    else
                        transfers.enqueueMappedTransferInterval(*interval_,
                            index == 0u ? MappedTransferInterval::Open : MappedTransferInterval::Closed,
                            stream);
                    capture.finish();
                }
            });
        }

        /** @brief Retire sources only after the cache destroyed their native clones. */
        ~CapturedBranch() override
        {
            epoch_->context_->submitAndWait([&]
            {
                for (auto &fragment : fragments_) fragment.reset();
                interval_.reset();
            });
        }
        /** @copydoc IGraphCaptureAuxiliaryBranch::authorityIdentity */
        const void *authorityIdentity() const noexcept override { return epoch_.get(); }
        /** @copydoc IGraphCaptureAuxiliaryBranch::device */
        DeviceId device() const noexcept override { return epoch_->device(); }
        /** @copydoc IGraphCaptureAuxiliaryBranch::name */
        std::string_view name() const noexcept override { return epoch_->config_.name; }

        /** @brief Decorate the final native body once, including all external waits. */
        bool attach(IGPUGraphCapture &graph) noexcept override
        {
            if (state_ != AttachmentState::Prepared || isGraphCaptureActive())
                return false;
            try
            {
                if (!graph.appendParallelBranch({
                        .open = *fragments_[0],
                        .worker = *fragments_[1],
                        .close = *fragments_[2]}))
                    return false;
                state_ = AttachmentState::Attached;
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MappedTransferProgressEpoch] Parallel attachment failed: " << error.what());
                return false;
            }
        }

    private:
        /** One cold attachment replaces paired, cross-thread capture flags/events. */
        enum class AttachmentState { Prepared, Attached };
        std::shared_ptr<MappedTransferProgressEpoch> epoch_;
        std::shared_ptr<DeviceTransferBuffer> interval_;
        std::array<std::unique_ptr<IGPUGraphCapture>, 3> fragments_;
        AttachmentState state_ = AttachmentState::Prepared;
    };

    GraphCaptureAuxiliaryBranchFactory MappedTransferProgressEpoch::graphBranchFactory()
    {
        if (!config_.device.is_cuda()) return {};
        return {.authority_identity = this, .device = device(),
            .create = [epoch = shared_from_this()]
            { return std::make_unique<CapturedBranch>(epoch); }};
    }

    namespace
    {
        /** @return Stable human-readable completion error. */
        const char *progressErrorName(
            MappedTransferProgressError error) noexcept
        {
            switch (error)
            {
            case MappedTransferProgressError::None:
                return "none";
            case MappedTransferProgressError::InvalidIdentity:
                return "invalid command identity";
            case MappedTransferProgressError::InvalidAddress:
                return "invalid source or destination address";
            case MappedTransferProgressError::InvalidByteCount:
                return "invalid byte count";
            }
            return "unknown transfer error";
        }

        /** @return Stable diagnostic name for a permanent slot direction. */
        const char *directionName(MappedTransferDirection direction) noexcept
        {
            switch (direction)
            {
            case MappedTransferDirection::DeviceToHost:
                return "device_to_host";
            case MappedTransferDirection::HostToDevice:
                return "host_to_device";
            }
            return "invalid";
        }

        /** @return Positive monotonic nanoseconds for rare stall diagnostics. */
        std::uint64_t steadyNanoseconds() noexcept
        {
            const auto count = std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
            return static_cast<std::uint64_t>(
                std::max<std::int64_t>(1, count));
        }

        /** @return Whether every generation-bound command word is coherent. */
        bool commandIdentityIsValid(
            const MappedTransferProgressCommand &command,
            std::uint64_t generation,
            std::size_t maximum_bytes) noexcept
        {
            return generation != 0u &&
                   command.generation_magic ==
                       mappedTransferProgressGenerationMagic(generation) &&
                   command.generation_version ==
                       mappedTransferProgressGenerationVersion(generation) &&
                   command.source_address != 0u &&
                   command.destination_address != 0u &&
                   command.source_complement == ~command.source_address &&
                   command.destination_complement ==
                       ~command.destination_address &&
                   command.bytes != 0u &&
                   command.bytes <= maximum_bytes &&
                   command.bytes_complement == ~command.bytes;
        }
    } // namespace

    MappedTransferProgressSlot::MappedTransferProgressSlot(
        std::shared_ptr<MappedTransferProgressEpoch> epoch,
        std::size_t index) noexcept
        : epoch_(std::move(epoch)), index_(index)
    {
    }

    MappedTransferProgressSlot::~MappedTransferProgressSlot()
    {
        if (pending_)
        {
            LOG_ERROR(
                "[MappedTransferProgressSlot] Destroyed with an outstanding "
                "copy command at slot "
                << index_);
            std::terminate();
        }
    }

    MappedTransferProgressSlot::MappedTransferProgressSlot(
        MappedTransferProgressSlot &&other) noexcept
    {
        moveFrom(std::move(other));
    }

    MappedTransferProgressSlot &MappedTransferProgressSlot::operator=(
        MappedTransferProgressSlot &&other) noexcept
    {
        if (this == &other)
            return *this;
        if (pending_)
            std::terminate();
        epoch_.reset();
        moveFrom(std::move(other));
        return *this;
    }

    void MappedTransferProgressSlot::moveFrom(
        MappedTransferProgressSlot &&other) noexcept
    {
        epoch_ = std::move(other.epoch_);
        index_ = std::exchange(
            other.index_, static_cast<std::size_t>(-1));
        next_generation_ = std::exchange(other.next_generation_, 0u);
        pending_generation_ =
            std::exchange(other.pending_generation_, 0u);
        pending_ = std::exchange(other.pending_, false);
        completed_generation_ = std::exchange(other.completed_generation_, 0u);
        completed_device_nanoseconds_ = std::exchange(other.completed_device_nanoseconds_, std::nullopt);
    }

    bool MappedTransferProgressSlot::valid() const noexcept
    {
        return epoch_ && index_ < epoch_->slotCapacity();
    }

    std::uint64_t MappedTransferProgressSlot::publishDeviceToMappedHost(
        const void *source,
        std::size_t source_capacity,
        std::size_t source_offset,
        std::size_t bytes,
        TransferProducerDependency dependency)
    {
        if (!valid())
            throw std::logic_error(
                "Mapped transfer-progress publication has no reserved slot");
        if (pending_)
            throw std::logic_error(
                "Mapped transfer-progress slot cannot overwrite an outstanding generation");
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error(
                "Mapped transfer-progress slot generation overflowed");

        const std::uint64_t generation = ++next_generation_;
        pending_generation_ = epoch_->publish(
            index_, generation, MappedTransferDirection::DeviceToHost,
            source, source_capacity, source_offset, bytes, dependency);
        pending_ = true;
        return pending_generation_;
    }

    std::uint64_t MappedTransferProgressSlot::publishMappedHostToDevice(
        void *destination,
        std::size_t destination_capacity,
        std::size_t destination_offset,
        std::size_t bytes)
    {
        if (!valid())
            throw std::logic_error(
                "Mapped transfer-progress publication has no reserved slot");
        if (pending_)
            throw std::logic_error(
                "Mapped transfer-progress slot cannot overwrite an outstanding generation");
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error(
                "Mapped transfer-progress slot generation overflowed");

        const std::uint64_t generation = ++next_generation_;
        pending_generation_ = epoch_->publish(
            index_, generation, MappedTransferDirection::HostToDevice,
            destination, destination_capacity, destination_offset, bytes);
        pending_ = true;
        return pending_generation_;
    }

    MappedTransferProgress MappedTransferProgressSlot::poll(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!valid() || !pending_)
        {
            if (error)
                *error =
                    "Mapped transfer-progress poll has no outstanding command";
            return MappedTransferProgress::Failed;
        }
        const std::size_t expected_bytes = static_cast<std::size_t>(
            epoch_->command(index_).bytes);
        const MappedTransferProgress result = epoch_->poll(
            index_, pending_generation_, expected_bytes, error);
        if (result != MappedTransferProgress::Pending)
        {
            if (result == MappedTransferProgress::Ready)
            {
                completed_generation_ = pending_generation_;
                const auto measured = epoch_->completion(index_).device_active_nanoseconds;
                completed_device_nanoseconds_ = measured
                    ? std::optional<std::uint64_t>{measured} : std::nullopt;
            }
            else
            {
                completed_generation_ = 0u;
                completed_device_nanoseconds_.reset();
            }
            pending_ = false;
            pending_generation_ = 0u;
        }
        return result;
    }

    std::optional<std::uint64_t> MappedTransferProgressSlot::completedDeviceNanoseconds() const
    {
        if (!valid() || pending_ || !completed_generation_)
            throw std::logic_error("Transfer timing requires an acquired successful completion");
        return completed_device_nanoseconds_;
    }

    std::shared_ptr<MappedTransferProgressEpoch>
    MappedTransferProgressEpoch::create(Config config)
    {
        auto epoch = std::shared_ptr<MappedTransferProgressEpoch>(
            new MappedTransferProgressEpoch(std::move(config)));
        epoch->materialize();
        return epoch;
    }

    MappedTransferProgressEpoch::MappedTransferProgressEpoch(Config config)
        : config_(std::move(config))
    {
        if (!config_.device.is_gpu() || config_.slot_capacity == 0u ||
            config_.execution_lane_capacity == 0u ||
            config_.execution_lane_capacity > config_.slot_capacity ||
            config_.execution_streams.empty() ||
            config_.execution_streams.size() >
                config_.execution_lane_capacity ||
            config_.maximum_bytes == 0u || config_.name.empty())
        {
            throw std::invalid_argument(
                "Mapped transfer-progress epoch requires one GPU, positive command/lane/stream geometry, streams no greater than execution lanes, execution lanes no greater than command slots, and a stable name");
        }
        for (std::size_t index = 0u;
             index < config_.execution_streams.size(); ++index)
        {
            const auto &stream = config_.execution_streams[index];
            if (!stream.valid() || stream.device() != config_.device)
            {
                throw std::invalid_argument(
                    "Mapped transfer-progress stream pool contains an invalid or wrong-device execution lane");
            }
            for (std::size_t previous = 0u; previous < index; ++previous)
            {
                if (config_.execution_streams[previous].stream() ==
                    stream.stream())
                {
                    throw std::invalid_argument(
                        "Mapped transfer-progress stream pool contains a duplicate physical stream identity");
                }
            }
        }
        if (config_.perf_device.empty())
            config_.perf_device = config_.device.toString();
    }

    MappedTransferProgressEpoch::~MappedTransferProgressEpoch()
    {
        if (outstanding_commands_.load(std::memory_order_acquire) != 0u)
        {
            LOG_ERROR(
                "[MappedTransferProgressEpoch] Destroyed with outstanding "
                "commands on "
                << config_.device.toString());
            std::terminate();
        }

        try
        {
            if (context_)
            {
                context_->submitAndWait(
                    [this]
                    {
                        // Teardown retains cursors until the last finite idle
                        // pass retires, including a pass overtaken by a graph.
                        if (idle_service_ == IdleServiceSubmission::InFlight)
                        {
                            context_->synchronizeEvent(execution_lanes_.front().terminal_event);
                            idle_service_ = IdleServiceSubmission::Quiescent;
                        }
                        for (const SlotRuntime &runtime : slot_runtimes_)
                        {
                            if (runtime.lifecycle == SlotLifecycle::InFlight)
                            {
                                LOG_ERROR(
                                    "[MappedTransferProgressEpoch] Teardown reached a live copy on "
                                    << config_.device.toString());
                                std::terminate();
                            }
                            if (runtime.lifecycle != SlotLifecycle::Unreserved &&
                                runtime.lifecycle != SlotLifecycle::Idle)
                            {
                                LOG_ERROR(
                                    "[MappedTransferProgressEpoch] Teardown reached an unretired command slot on "
                                    << config_.device.toString());
                                std::terminate();
                            }
                        }
                        for (ExecutionLaneRuntime &lane : execution_lanes_)
                        {
                            if (lane.busy())
                            {
                                LOG_ERROR(
                                    "[MappedTransferProgressEpoch] Teardown reached a leased execution lane on "
                                    << config_.device.toString());
                                std::terminate();
                            }
                            if (lane.terminal_event)
                            {
                                context_->destroyEvent(lane.terminal_event);
                                lane.terminal_event = nullptr;
                            }
                            /* Named auxiliary streams belong to the worker
                             * context and may be reused by the next model
                             * instance; this epoch owns only their identity. */
                            lane.stream = nullptr;
                        }
                        service_cursors_.reset();
                        service_inbox_.reset();
                    });
            }
        }
        catch (...)
        {
            LOG_ERROR(
                "[MappedTransferProgressEpoch] Failed to release GPU events on "
                << config_.device.toString());
            std::terminate();
        }
    }

    void MappedTransferProgressEpoch::materialize()
    {
        backend_ = getBackendFor(config_.device);
        if (!backend_)
            throw std::runtime_error(
                "Mapped transfer-progress epoch has no backend for " +
                config_.device.toString());
        context_ = &GPUDeviceContextPool::instance().getContext(config_.device);

        commands_ = std::make_unique<MappedTransferProgressCommand[]>(
            config_.slot_capacity);
        completions_ = std::make_unique<MappedTransferProgressCompletion[]>(
            config_.slot_capacity);
        slot_runtimes_.resize(config_.slot_capacity);
        slot_labels_.resize(config_.slot_capacity);
        execution_lanes_.resize(config_.execution_lane_capacity);

        context_->submitAndWait(
            [this]
            {
                for (std::size_t index = 0u;
                     index < config_.execution_lane_capacity; ++index)
                {
                    ExecutionLaneRuntime &lane = execution_lanes_[index];
                    /* Every logical lane retains its own completion event, but
                     * compatible lanes share the setup-owned physical stream
                     * pool round-robin. All published lanes are therefore
                     * enqueued in one bounded callback; no later host replay is
                     * needed merely because runtime queue creation is bounded. */
                    lane.stream = config_
                                      .execution_streams[
                                          index %
                                          config_.execution_streams.size()]
                                      .stream();
                    lane.terminal_event = context_->createEvent();
                    if (!lane.stream || !lane.terminal_event)
                    {
                        throw std::runtime_error(
                            "Mapped transfer-progress epoch could not materialize its bounded background execution-lane pool");
                    }
                }
            });

        if (config_.device.is_cuda())
        {
            context_->submitAndWait([this]
            {
                // Map physical concurrency, never the permanent topology directory.
                TransferEngine transfers;
                const auto capacity = config_.execution_lane_capacity;
                constexpr auto record_bytes = sizeof(MappedTransferProgressCommand) +
                                              sizeof(MappedTransferProgressCompletion);
                if (capacity > std::numeric_limits<std::size_t>::max() / record_bytes)
                    throw std::overflow_error("Mapped transfer physical inbox extent overflowed");
                const DeviceId devices[] = {config_.device};
                service_inbox_ = transfers.allocateMappedHostRegion(capacity * record_bytes, devices);
                service_cursors_ = transfers.allocateMappedTransferServiceCursors(
                    capacity, config_.device, executionStream());
                auto *const setup_event = execution_lanes_.front().terminal_event;
                if (!context_->recordEventChecked(setup_event, executionStream()))
                    throw std::runtime_error("Mapped transfer service setup event publication failed");
                // CUDA capture cannot import an uncaptured event dependency.
                // Cold construction returns only a ready service: join this
                // exact initialization once, then reuse the event for idle
                // submissions. No setup event is embedded in captured graphs.
                context_->synchronizeEvent(setup_event);
            });
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "mapped_transfer_progress_epochs_materialized",
            1.0,
            "model_setup",
            config_.perf_device,
            {{"device", config_.device.toString()},
             {"slot_capacity", std::to_string(config_.slot_capacity)},
             {"execution_lane_capacity",
              std::to_string(config_.execution_lane_capacity)},
             {"execution_stream_capacity",
              std::to_string(config_.execution_streams.size())},
             {"maximum_bytes", std::to_string(config_.maximum_bytes)},
             {"stream_class", "background_maintenance"},
             {"copy_mechanism", config_.device.is_cuda() ? "graph_bounded_service" : "native_sdma"},
             {"scope", "node_local"},
             {"blocking", "false"}});
    }

    MappedTransferProgressSlot MappedTransferProgressEpoch::reserveSlot(
        MappedTransferDirection direction,
        std::shared_ptr<MappedHostTransferRegion> mapped_region,
        std::string diagnostic_label)
    {
        if (!mapped_region || !mapped_region->isBound() ||
            !mapped_region->hasDevice(config_.device) ||
            mapped_region->sizeBytes() == 0u)
        {
            throw std::invalid_argument(
                "Mapped transfer-progress slot requires a nonempty bounded region registered to its exact GPU");
        }

        std::lock_guard<std::mutex> lock(reservation_mutex_);
        if (reserved_slots_ >= config_.slot_capacity)
            throw std::runtime_error(
                "Mapped transfer-progress topology accounting exhausted its fixed slot capacity");
        const std::size_t index = reserved_slots_++;
        if (diagnostic_label.empty())
            diagnostic_label = "slot_" + std::to_string(index);

        SlotRuntime &runtime = slot_runtimes_[index];
        if (runtime.lifecycle != SlotLifecycle::Unreserved ||
            runtime.mapped_region)
            throw std::logic_error(
                "Mapped transfer-progress slot runtime lost setup identity");
        runtime.direction = direction;
        runtime.mapped_region = std::move(mapped_region);
        runtime.lifecycle = SlotLifecycle::Idle;
        slot_labels_[index] = std::move(diagnostic_label);
        return MappedTransferProgressSlot(shared_from_this(), index);
    }

    bool MappedTransferProgressEpoch::validateDeviceAddresses(
        std::span<const void *const> addresses,
        std::string_view role,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!context_ || addresses.empty() || role.empty() ||
            std::any_of(
                addresses.begin(), addresses.end(),
                [](const void *address) { return address == nullptr; }))
        {
            if (error)
                *error =
                    "Mapped transfer-progress address validation has incomplete identity";
            return false;
        }

        bool valid = true;
        std::string failure;
        const auto validate = [&]
        {
            for (std::size_t index = 0u; index < addresses.size(); ++index)
            {
                const PointerValidationResult result =
                    context_->validatePointerDevice(
                        addresses[index], config_.device.ordinal);
                if (result.valid)
                    continue;

                std::ostringstream message;
                message << "Mapped transfer-progress " << role
                        << " address " << index << " does not belong to "
                        << config_.device.toString()
                        << " pointer=" << addresses[index]
                        << " actual_device=" << result.actual_device;
                if (!result.details.empty())
                    message << " details={" << result.details << '}';
                failure = message.str();
                valid = false;
                break;
            }
        };

        try
        {
            if (context_->ownsCurrentThread())
                validate();
            else
                context_->submitAndWait(validate);
        }
        catch (const std::exception &exception)
        {
            valid = false;
            failure =
                "Mapped transfer-progress pointer validation threw for " +
                config_.device.toString() + ": " + exception.what();
        }
        catch (...)
        {
            valid = false;
            failure =
                "Mapped transfer-progress pointer validation threw a non-standard exception for " +
                config_.device.toString();
        }
        if (!valid && error)
            *error = std::move(failure);
        return valid;
    }

    std::uint64_t MappedTransferProgressEpoch::publish(
        std::size_t index,
        std::uint64_t generation,
        MappedTransferDirection direction,
        const void *device_region,
        std::size_t device_capacity,
        std::size_t device_offset,
        std::size_t bytes,
        TransferProducerDependency dependency)
    {
        if (generation == 0u || !device_region || bytes == 0u ||
            bytes > config_.maximum_bytes ||
            device_offset > device_capacity ||
            bytes > device_capacity - device_offset)
        {
            throw std::invalid_argument(
                "Mapped transfer-progress command has invalid slot, device bounds, generation, or bytes");
        }

        std::size_t previous_outstanding = 0u;
        {
            /* Publication and device-worker lane assignment share this lock.
             * The immutable command cache line is still release-published so
             * observation remains correct if submission later moves to a
             * retained device-side dispatcher. */
            std::lock_guard<std::mutex> lock(reservation_mutex_);
            if (index >= reserved_slots_)
                throw std::invalid_argument(
                    "Mapped transfer-progress command names an unreserved slot");

            SlotRuntime &runtime = slot_runtimes_[index];
            if (runtime.lifecycle != SlotLifecycle::Idle ||
                runtime.direction != direction || !runtime.mapped_region)
            {
                throw std::logic_error(
                    "Mapped transfer-progress publication contradicts its permanent slot direction or lifecycle");
            }
            if (!runtime.mapped_region->contains(0u, bytes))
                throw std::invalid_argument(
                    "Mapped transfer-progress command exceeds its exclusive mapped slot extent");

            MappedTransferProgressCommand &entry = command(index);
            const std::uint64_t completed =
                std::atomic_ref<std::uint64_t>(
                    completion(index).completed_generation)
                    .load(std::memory_order_acquire);
            const std::uint64_t prior_generation =
                std::atomic_ref<std::uint64_t>(entry.generation)
                    .load(std::memory_order_acquire);
            if (completed != prior_generation)
                throw std::logic_error(
                    "Mapped transfer-progress slot publication raced an unobserved generation");

            const auto *const device_bytes =
                static_cast<const unsigned char *>(device_region) +
                device_offset;
            const std::uint64_t device_address = static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(device_bytes));
            const std::uint64_t host_address = static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(
                    runtime.mapped_region->mutableHostData()));
            const std::uint64_t source_address =
                direction == MappedTransferDirection::DeviceToHost
                    ? device_address
                    : host_address;
            const std::uint64_t destination_address =
                direction == MappedTransferDirection::DeviceToHost
                    ? host_address
                    : device_address;
            const std::uint64_t byte_count =
                static_cast<std::uint64_t>(bytes);

            /* Publish generation last. The release/acquire pair keeps the
             * command body coherent until a free execution lane claims it. */
            entry.generation_magic =
                mappedTransferProgressGenerationMagic(generation);
            entry.generation_version =
                mappedTransferProgressGenerationVersion(generation);
            entry.source_address = source_address;
            entry.destination_address = destination_address;
            entry.bytes = byte_count;
            entry.source_complement = ~source_address;
            entry.destination_complement = ~destination_address;
            entry.bytes_complement = ~byte_count;
            std::atomic_ref<std::uint64_t>(entry.generation).store(
                generation, std::memory_order_release);

            runtime.launched_generation = 0u;
            runtime.execution_lane_index = static_cast<std::size_t>(-1);
            runtime.progress_watch.published(steadyNanoseconds());
            runtime.dependency = dependency;
            runtime.lifecycle = SlotLifecycle::Published;
            previous_outstanding = outstanding_commands_.fetch_add(
                1u, std::memory_order_acq_rel);
        }
        if (previous_outstanding == 0u)
        {
            /* A batch boundary occurs for every reusable transfer generation.
             * Keep this per-buffer traffic at TRACE so stress campaigns do
             * not turn observability into part of the copy critical path. */
            LOG_TRACE(
                "[MappedTransferProgressEpoch] Command batch became active"
                << " device=" << config_.device.toString()
                << " authority=" << config_.name
                << " first_slot=" << index
                << " direction=" << directionName(direction)
                << " generation=" << generation);
        }
        commands_published_.fetch_add(1u, std::memory_order_relaxed);
        return generation;
    }

    MappedTransferProgress MappedTransferProgressEpoch::poll(
        std::size_t index,
        std::uint64_t generation,
        std::size_t expected_bytes,
        std::string *error) noexcept
    {
        if (generation == 0u)
        {
            if (error)
                *error =
                    "Mapped transfer-progress completion identity is invalid";
            return MappedTransferProgress::Failed;
        }
        {
            std::lock_guard<std::mutex> lock(reservation_mutex_);
            if (index >= reserved_slots_ ||
                slot_runtimes_[index].lifecycle ==
                    SlotLifecycle::Unreserved)
            {
                if (error)
                    *error =
                        "Mapped transfer-progress completion names an unreserved slot";
                return MappedTransferProgress::Failed;
            }
        }
        MappedTransferProgressCompletion &result = completion(index);
        const std::uint64_t completed =
            std::atomic_ref<std::uint64_t>(result.completed_generation)
                .load(std::memory_order_acquire);
        if (completed < generation)
        {
            // Each slot owns its publication timestamp. A busy queue's lifetime
            // must not be reported as the latency of a newly submitted command.
            std::lock_guard<std::mutex> lock(reservation_mutex_);
            SlotRuntime &runtime = slot_runtimes_[index];
            const auto now_ns = steadyNanoseconds();
            if (const auto age = runtime.progress_watch.pending(now_ns))
            {
                const auto query_age =
                    runtime.progress_watch.incompleteEventQueryAge(now_ns);
                LOG_WARN(
                    "[MappedTransferProgressEpoch] copy command remains pending"
                    << " device=" << config_.device.toString()
                    << " authority=" << config_.name
                    << " slot=" << index
                    << " slot_label=" << slot_labels_[index]
                    << " direction=" << directionName(runtime.direction)
                    << " requested_generation=" << generation
                    << " completed_generation=" << completed
                    << " command_pending_ns=" << *age
                    << " native_not_ready_queries="
                    << runtime.progress_watch.incompleteEventQueries()
                    << " last_native_not_ready_age_ns="
                    << (query_age ? std::to_string(*query_age) : "never_queried")
                    << " bytes=" << command(index).bytes
                    << " lifecycle="
                    << static_cast<unsigned>(runtime.lifecycle)
                    << " execution_lane="
                    << runtime.execution_lane_index
                    << " outstanding="
                    << outstanding_commands_.load(std::memory_order_acquire)
                    << " published="
                    << commands_published_.load(std::memory_order_relaxed)
                    << " completed="
                    << commands_completed_.load(std::memory_order_relaxed)
                    << " copy_submissions="
                    << copy_submissions_.load(std::memory_order_relaxed));
            }
            return MappedTransferProgress::Pending;
        }

        bool lifecycle_valid = false;
        {
            /* The completion generation is release-published before the worker
             * releases this lock. Acquiring the same lock therefore observes
             * the matching typed state transition as one indivisible fact. */
            std::lock_guard<std::mutex> lock(reservation_mutex_);
            SlotRuntime &runtime = slot_runtimes_[index];
            lifecycle_valid =
                runtime.lifecycle == SlotLifecycle::CompletionPublished &&
                runtime.launched_generation == completed &&
                runtime.execution_lane_index ==
                    static_cast<std::size_t>(-1);
            runtime.lifecycle = SlotLifecycle::Idle;
            runtime.launched_generation = 0u;
        }

        const auto finish = [this]() noexcept
        {
            const std::size_t previous = outstanding_commands_.fetch_sub(
                1u, std::memory_order_acq_rel);
            if (previous == 0u)
                std::terminate();
            return previous == 1u;
        };
        if (!lifecycle_valid || completed != generation)
        {
            (void)finish();
            command_failures_.fetch_add(1u, std::memory_order_relaxed);
            if (error)
            {
                *error = !lifecycle_valid
                             ? "Mapped transfer-progress completion contradicted the typed slot lifecycle"
                             : "Mapped transfer-progress completion skipped the requested generation";
            }
            return MappedTransferProgress::Failed;
        }

        const auto transfer_error = static_cast<MappedTransferProgressError>(
            result.error);
        if (transfer_error != MappedTransferProgressError::None ||
            result.completed_bytes != expected_bytes)
        {
            (void)finish();
            command_failures_.fetch_add(1u, std::memory_order_relaxed);
            if (error)
            {
                *error = std::string(
                             "Mapped transfer-progress copy command failed: ") +
                         progressErrorName(transfer_error);
            }
            return MappedTransferProgress::Failed;
        }

        const bool batch_drained = finish();
        commands_completed_.fetch_add(1u, std::memory_order_relaxed);
        bytes_completed_.fetch_add(
            result.completed_bytes, std::memory_order_relaxed);
        if (batch_drained)
        {
            /* Completion is generation-frequency traffic, not a rare
             * lifecycle summary. TRACE preserves diagnosis without making a
             * 256-generation integration proof sensitive to log formatting. */
            LOG_TRACE(
                "[MappedTransferProgressEpoch] Command batch drained"
                << " device=" << config_.device.toString()
                << " authority=" << config_.name
                << " published="
                << commands_published_.load(std::memory_order_relaxed)
                << " completed="
                << commands_completed_.load(std::memory_order_relaxed)
                << " bytes="
                << bytes_completed_.load(std::memory_order_relaxed));
        }
        return MappedTransferProgress::Ready;
    }

    bool MappedTransferProgressEpoch::submitOutstandingProgress() noexcept
    {
        if (!hasOutstandingCommands())
            return true;
        if (!context_)
            return false;
        if (context_->ownsCurrentThread())
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "mapped_transfer_progress_epoch_submission_paths",
                1.0,
                "maintenance",
                config_.perf_device,
                {{"device", config_.device.toString()},
                 {"path", "worker_direct"},
                 {"wait_requested", "false"}});
            return launchOutstandingProgress();
        }

        bool submitted = false;
        using Clock = std::chrono::steady_clock;
        const bool timing_enabled = PerfStatsCollector::isDomainEnabled(
            "moe_overlay_residency");
        const auto submit_begin =
            timing_enabled ? Clock::now() : Clock::time_point{};
        try
        {
            /* This wait covers only the bounded CPU enqueue callback. It never
             * waits for copy or inference work on the device. */
            context_->submitAndWait(
                [this, &submitted]
                { submitted = launchOutstandingProgress(); });
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR(
                "[MappedTransferProgressEpoch] Worker submission failed on "
                << config_.device.toString() << ": " << exception.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR(
                "[MappedTransferProgressEpoch] Worker submission threw a non-standard exception on "
                << config_.device.toString());
            return false;
        }
        if (timing_enabled)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "mapped_transfer_progress_epoch_worker_handoff_host_time",
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - submit_begin)
                        .count()),
                "maintenance",
                config_.perf_device,
                {{"device", config_.device.toString()},
                 {"path", "worker_handoff"},
                 {"wait_scope", "host_callback_only"},
                 {"device_wait", "false"}});
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "mapped_transfer_progress_epoch_submission_paths",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"device", config_.device.toString()},
             {"path", "worker_handoff"},
             {"wait_requested", "host_callback_only"}});
        return submitted;
    }

    bool MappedTransferProgressEpoch::launchOutstandingProgress() noexcept
    {
        if (!context_ || !context_->ownsCurrentThread())
        {
            LOG_ERROR(
                "[MappedTransferProgressEpoch] copy submission is not on the exact device worker for "
                << config_.device.toString());
            return false;
        }

        bool all_ok = true;
        bool observed_work = false;
        std::uint64_t submissions = 0u;
        TransferEngine transfer_engine;
        const auto begin = std::chrono::steady_clock::now();
        // One timestamp per bounded worker pass is sufficient to distinguish
        // actual native not-ready results from delayed host service. Logging
        // consumes this observation without adding driver calls or GPU waits.
        const auto observation_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                begin.time_since_epoch()).count());

        constexpr std::size_t no_index = static_cast<std::size_t>(-1);
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        auto *service_commands = service_inbox_
            ? static_cast<MappedTransferProgressCommand *>(service_inbox_->mutableHostData()) : nullptr;
        auto *service_completions = service_inbox_
            ? reinterpret_cast<MappedTransferProgressCompletion *>(
                service_commands + execution_lanes_.size()) : nullptr;

        /* Completion belongs to the execution lane, not the permanent command
         * slot. Query every busy lane once, then publish the matching slot's
         * completion before making that lane available to another command. */
        for (std::size_t lane_index = 0u;
             lane_index < execution_lanes_.size(); ++lane_index)
        {
            ExecutionLaneRuntime &lane = execution_lanes_[lane_index];
            if (!lane.busy())
                continue;

            observed_work = true;
            if (lane.active_slot >= reserved_slots_)
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] Execution lane owns an invalid command slot"
                    << " device=" << config_.device.toString()
                    << " lane=" << lane_index
                    << " slot=" << lane.active_slot);
                all_ok = false;
                continue;
            }
            const std::size_t slot_index = lane.active_slot;
            SlotRuntime &runtime = slot_runtimes_[slot_index];
            if (runtime.lifecycle != SlotLifecycle::InFlight ||
                runtime.execution_lane_index != lane_index)
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] Execution lane and command slot disagree on ownership"
                    << " device=" << config_.device.toString()
                    << " lane=" << lane_index
                    << " slot=" << slot_index
                    << " slot_lane=" << runtime.execution_lane_index
                    << " lifecycle="
                    << static_cast<unsigned>(runtime.lifecycle));
                all_ok = false;
                continue;
            }

            bool ready = false;
            if (service_completions)
            {
                const auto completed = std::atomic_ref<std::uint64_t>(
                    service_completions[lane_index].completed_generation).load(std::memory_order_acquire);
                if (completed > lane.service_generation)
                {
                    LOG_ERROR("[MappedTransferProgressEpoch] Device service skipped an inbox generation");
                    return false;
                }
                ready = completed == lane.service_generation;
            }
            else if (!context_->queryEventChecked(lane.terminal_event, ready))
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] copy event query failed"
                    << " device=" << config_.device.toString()
                    << " lane=" << lane_index
                    << " slot=" << slot_index
                    << " label=" << slot_labels_[slot_index]);
                all_ok = false;
                continue;
            }
            if (!ready)
            {
                if (!service_completions)
                    runtime.progress_watch.observedIncompleteEvent(observation_ns);
                in_flight_observations_.fetch_add(
                    1u, std::memory_order_relaxed);
                continue;
            }

            MappedTransferProgressCompletion &result = completion(slot_index);
            result.completed_bytes = service_completions
                ? service_completions[lane_index].completed_bytes : command(slot_index).bytes;
            result.error = service_completions ? service_completions[lane_index].error
                : static_cast<std::uint32_t>(MappedTransferProgressError::None);
            result.device_active_nanoseconds = service_completions
                ? service_completions[lane_index].device_active_nanoseconds : 0u;
            std::atomic_ref<std::uint64_t>(result.completed_generation)
                .store(runtime.launched_generation, std::memory_order_release);
            runtime.execution_lane_index = no_index;
            runtime.lifecycle = SlotLifecycle::CompletionPublished;
            lane.active_slot = no_index;
        }

        std::size_t free_lane_cursor = 0u;
        for (std::size_t index = 0u; index < reserved_slots_; ++index)
        {
            SlotRuntime &runtime = slot_runtimes_[index];
            if (runtime.lifecycle != SlotLifecycle::Published)
                continue;

            // Source admission is independent of the service's native stream.
            // Never publish readable addresses to a resident worker until the
            // exact producer completed. Other independent slots still progress.
            if (auto *producer = runtime.dependency.event())
            {
                bool producer_ready = false;
                if (!context_->queryEventChecked(producer, producer_ready))
                {
                    LOG_ERROR("[MappedTransferProgressEpoch] Source producer event query failed"
                              << " device=" << config_.device.toString()
                              << " slot=" << index);
                    all_ok = false;
                    continue;
                }
                if (!producer_ready)
                    continue;
                runtime.dependency = TransferProducerDependency::published();
            }

            MappedTransferProgressCommand &entry = command(index);
            const std::uint64_t generation =
                std::atomic_ref<std::uint64_t>(entry.generation)
                    .load(std::memory_order_acquire);
            const std::uint64_t completed =
                std::atomic_ref<std::uint64_t>(
                    completion(index).completed_generation)
                    .load(std::memory_order_acquire);
            observed_work = true;
            if (generation == 0u || generation <= completed ||
                !runtime.mapped_region ||
                !commandIdentityIsValid(
                    entry, generation, config_.maximum_bytes))
            {
                MappedTransferProgressCompletion &result = completion(index);
                result.completed_bytes = 0u;
                result.error = static_cast<std::uint32_t>(
                    MappedTransferProgressError::InvalidIdentity);
                std::atomic_ref<std::uint64_t>(
                    result.completed_generation)
                    .store(generation, std::memory_order_release);
                runtime.launched_generation = generation;
                runtime.execution_lane_index = no_index;
                runtime.lifecycle = SlotLifecycle::CompletionPublished;
                command_failures_.fetch_add(1u, std::memory_order_relaxed);
                all_ok = false;
                continue;
            }

            while (free_lane_cursor < execution_lanes_.size() &&
                   execution_lanes_[free_lane_cursor].busy())
            {
                ++free_lane_cursor;
            }
            if (free_lane_cursor == execution_lanes_.size())
            {
                /* A queued command is normal bounded backpressure. Its slot and
                 * mapped bytes remain immutable until a later progress pass
                 * observes a free physical execution lane. */
                break;
            }

            const std::size_t lane_index = free_lane_cursor++;
            ExecutionLaneRuntime &lane = execution_lanes_[lane_index];
            if (!lane.stream || !lane.terminal_event || lane.busy())
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] Free execution lane lost its setup identity"
                    << " device=" << config_.device.toString()
                    << " lane=" << lane_index);
                all_ok = false;
                break;
            }

            /* Claim ownership before invoking backend code. If an enqueue or
             * event record fails after partially submitting work, the lane
             * remains non-reusable and teardown fails rather than guessing. */
            runtime.launched_generation = generation;
            runtime.execution_lane_index = lane_index;
            runtime.lifecycle = SlotLifecycle::InFlight;
            lane.active_slot = index;
            try
            {
                const std::size_t bytes =
                    static_cast<std::size_t>(entry.bytes);
                if (service_commands)
                {
                    if (lane.service_generation == std::numeric_limits<std::uint64_t>::max())
                        throw std::overflow_error("Mapped transfer physical inbox generation overflowed");
                    const auto inbox_generation = ++lane.service_generation;
                    auto &inbox = service_commands[lane_index];
                    const auto mapped = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(
                        runtime.mapped_region->deviceAlias(config_.device)));
                    inbox.generation_magic = mappedTransferProgressGenerationMagic(inbox_generation);
                    inbox.generation_version = mappedTransferProgressGenerationVersion(inbox_generation);
                    inbox.source_address = runtime.direction == MappedTransferDirection::HostToDevice
                        ? mapped : entry.source_address;
                    inbox.destination_address = runtime.direction == MappedTransferDirection::DeviceToHost
                        ? mapped : entry.destination_address;
                    inbox.bytes = entry.bytes;
                    inbox.source_complement = ~inbox.source_address;
                    inbox.destination_complement = ~inbox.destination_address;
                    inbox.bytes_complement = ~inbox.bytes;
                    // The GPU claims execution. CPU publication reserves only
                    // immutable physical IO storage, never a queued-worker lock.
                    std::atomic_ref<std::uint64_t>(inbox.generation)
                        .store(inbox_generation, std::memory_order_release);
                    ++submissions;
                    continue;
                }
                // The typed lease includes setup-time function preparation.
                // The backend avoids the native queue that can be occupied by
                // peer-held inference. Terminal events and slot lifecycle remain
                // exactly the same in both directions and on both backends.
                const auto address = runtime.direction == MappedTransferDirection::DeviceToHost
                    ? entry.source_address : entry.destination_address;
                transfer_engine.enqueueBackgroundMappedCopy(
                    config_.execution_streams[lane_index % config_.execution_streams.size()],
                    runtime.direction,
                    reinterpret_cast<void *>(static_cast<std::uintptr_t>(address)),
                    bytes, 0u, *runtime.mapped_region, 0u, bytes);
                if (!context_->recordEventChecked(
                        lane.terminal_event, lane.stream))
                {
                    throw std::runtime_error(
                        "could not record the exact copy completion event");
                }
                ++submissions;
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] copy enqueue failed"
                    << " device=" << config_.device.toString()
                    << " slot=" << index
                    << " label=" << slot_labels_[index]
                    << " direction=" << directionName(runtime.direction)
                    << " error={" << exception.what() << '}');
                command_failures_.fetch_add(1u, std::memory_order_relaxed);
                all_ok = false;
            }
            catch (...)
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] copy enqueue threw a non-standard exception"
                    << " device=" << config_.device.toString()
                    << " slot=" << index
                    << " label=" << slot_labels_[index]);
                command_failures_.fetch_add(1u, std::memory_order_relaxed);
                all_ok = false;
            }
        }

        if (service_commands && observed_work)
        {
            // An idle pass and captured branches share the same GPU claims. A
            // queued idle pass cannot exclude an already resident graph worker.
            try
            {
                if (idle_service_ == IdleServiceSubmission::InFlight)
                {
                    bool ready = false;
                    if (!context_->queryEventChecked(execution_lanes_.front().terminal_event, ready))
                        throw std::runtime_error("Mapped transfer idle completion query failed");
                    if (ready) idle_service_ = IdleServiceSubmission::Quiescent;
                }
                if (idle_service_ == IdleServiceSubmission::Quiescent &&
                    std::any_of(execution_lanes_.begin(), execution_lanes_.end(),
                        [](const ExecutionLaneRuntime &lane) { return lane.busy(); }))
                {
                    transfer_engine.enqueueMappedTransferService(*service_inbox_, *service_cursors_,
                        maximumBytes(), nullptr, MappedTransferServiceRun::PublishedPass, executionStream());
                    idle_service_ = IdleServiceSubmission::InFlight;
                    if (!context_->recordEventChecked(execution_lanes_.front().terminal_event, executionStream()))
                        throw std::runtime_error("Mapped transfer idle terminal publication failed");
                }
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MappedTransferProgressEpoch] Idle service submission failed: " << error.what());
                all_ok = false;
            }
        }

        if (submissions != 0u)
        {
            copy_submissions_.fetch_add(submissions, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "mapped_transfer_progress_epoch_copy_submissions",
                static_cast<double>(submissions),
                "maintenance",
                config_.perf_device,
                {{"device", config_.device.toString()},
                 {"execution_lane_capacity",
                  std::to_string(config_.execution_lane_capacity)},
                 {"execution_stream_capacity",
                  std::to_string(config_.execution_streams.size())},
                 {"stream_class", "background_maintenance"},
                 {"copy_mechanism", config_.device.is_cuda() ? "graph_bounded_service" : "native_sdma"},
                 {"inference_wait", "false"},
                 {"blocking", "false"}});
            if (PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_residency"))
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_residency",
                    "mapped_transfer_progress_epoch_copy_enqueue_host_time",
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(
                            1,
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - begin)
                                .count())),
                    "maintenance",
                    config_.perf_device,
                    {{"device", config_.device.toString()},
                     {"submission_count", std::to_string(submissions)},
                     {"device_wait", "false"}});
            }
        }
        else if (!observed_work)
        {
            idle_submission_skips_.fetch_add(
                1u, std::memory_order_relaxed);
        }
        return all_ok;
    }

    MappedTransferProgressCommand &MappedTransferProgressEpoch::command(
        std::size_t index) noexcept
    {
        return commands_[index];
    }

    MappedTransferProgressCompletion &
    MappedTransferProgressEpoch::completion(std::size_t index) noexcept
    {
        return completions_[index];
    }

    MappedTransferProgressEpochStats
    MappedTransferProgressEpoch::stats() const noexcept
    {
        std::lock_guard<std::mutex> lock(reservation_mutex_);
        return {
            .slots_reserved = static_cast<std::uint64_t>(reserved_slots_),
            .commands_published = commands_published_.load(
                std::memory_order_relaxed),
            .commands_completed = commands_completed_.load(
                std::memory_order_relaxed),
            .bytes_completed = bytes_completed_.load(
                std::memory_order_relaxed),
            .copy_submissions = copy_submissions_.load(
                std::memory_order_relaxed),
            .idle_submission_skips = idle_submission_skips_.load(
                std::memory_order_relaxed),
            .in_flight_observations = in_flight_observations_.load(
                std::memory_order_relaxed),
            .command_failures = command_failures_.load(
                std::memory_order_relaxed),
        };
    }
} // namespace llaminar2
