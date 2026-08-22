/**
 * @file MappedTransferProgressEpoch.cpp
 * @brief Event-polled asynchronous DMA implementation for expert movement.
 *
 * Setup owns one registered mapped-host region, background stream, and event
 * per permanent topology slot. Maintenance release-publishes a bounded device
 * region, the exact GPU worker enqueues a TransferEngine DMA, and a later
 * non-blocking event query release-publishes host completion. Inference graphs
 * never capture, launch, join, or wait for this maintenance scheduler.
 */

#include "MappedTransferProgressEpoch.h"

#include "TransferEngine.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
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
                "DMA command at slot "
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
    }

    bool MappedTransferProgressSlot::valid() const noexcept
    {
        return epoch_ && index_ < epoch_->slotCapacity();
    }

    std::uint64_t MappedTransferProgressSlot::publishDeviceToMappedHost(
        const void *source,
        std::size_t source_capacity,
        std::size_t source_offset,
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
            index_, generation, MappedTransferDirection::DeviceToHost,
            source, source_capacity, source_offset, bytes);
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
            pending_ = false;
            pending_generation_ = 0u;
        }
        return result;
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
            config_.maximum_bytes == 0u || config_.name.empty())
        {
            throw std::invalid_argument(
                "Mapped transfer-progress epoch requires one GPU, positive geometry, and a stable name");
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
                        for (SlotRuntime &runtime : slot_runtimes_)
                        {
                            if (runtime.in_flight)
                            {
                                LOG_ERROR(
                                    "[MappedTransferProgressEpoch] Teardown reached a live DMA on "
                                    << config_.device.toString());
                                std::terminate();
                            }
                            if (runtime.terminal_event)
                            {
                                context_->destroyEvent(runtime.terminal_event);
                                runtime.terminal_event = nullptr;
                            }
                            runtime.stream = nullptr;
                        }
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

        context_->submitAndWait(
            [this]
            {
                for (std::size_t index = 0u;
                     index < config_.slot_capacity; ++index)
                {
                    SlotRuntime &runtime = slot_runtimes_[index];
                    runtime.stream = context_->getOrCreateAuxiliaryStream(
                        "mapped_transfer_progress:" + config_.name + ":" +
                            std::to_string(index),
                        GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance);
                    runtime.terminal_event = context_->createEvent();
                    if (!runtime.stream || !runtime.terminal_event)
                    {
                        throw std::runtime_error(
                            "Mapped transfer-progress epoch could not allocate a background stream and event per permanent slot");
                    }
                }
            });

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "mapped_transfer_progress_epochs_materialized",
            1.0,
            "model_setup",
            config_.perf_device,
            {{"device", config_.device.toString()},
             {"slot_capacity", std::to_string(config_.slot_capacity)},
             {"maximum_bytes", std::to_string(config_.maximum_bytes)},
             {"stream_class", "background_maintenance"},
             {"copy_mechanism", "async_dma"},
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
            !mapped_region->contains(0u, config_.maximum_bytes))
        {
            throw std::invalid_argument(
                "Mapped transfer-progress slot requires a sufficiently large region registered to its exact GPU");
        }

        std::lock_guard<std::mutex> lock(reservation_mutex_);
        if (reserved_slots_ >= config_.slot_capacity)
            throw std::runtime_error(
                "Mapped transfer-progress topology accounting exhausted its fixed slot capacity");
        const std::size_t index = reserved_slots_++;
        if (diagnostic_label.empty())
            diagnostic_label = "slot_" + std::to_string(index);

        SlotRuntime &runtime = slot_runtimes_[index];
        if (runtime.reserved || !runtime.stream || !runtime.terminal_event)
            throw std::logic_error(
                "Mapped transfer-progress slot runtime lost setup identity");
        runtime.direction = direction;
        runtime.mapped_region = std::move(mapped_region);
        runtime.reserved = true;
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
        std::size_t bytes)
    {
        if (index >= reserved_slots_ || generation == 0u || !device_region ||
            bytes == 0u || bytes > config_.maximum_bytes ||
            device_offset > device_capacity ||
            bytes > device_capacity - device_offset)
        {
            throw std::invalid_argument(
                "Mapped transfer-progress command has invalid slot, device bounds, generation, or bytes");
        }

        const SlotRuntime &runtime = slot_runtimes_[index];
        if (!runtime.reserved || runtime.direction != direction ||
            !runtime.mapped_region)
        {
            throw std::logic_error(
                "Mapped transfer-progress publication contradicts its permanent slot direction");
        }

        MappedTransferProgressCommand &entry = command(index);
        const std::uint64_t completed =
            std::atomic_ref<std::uint64_t>(
                completion(index).completed_generation)
                .load(std::memory_order_acquire);
        if (completed != entry.generation || runtime.in_flight)
            throw std::logic_error(
                "Mapped transfer-progress slot publication raced an unobserved generation");

        const auto *const device_bytes =
            static_cast<const unsigned char *>(device_region) + device_offset;
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
        const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes);

        /* Publish generation last. The release/acquire pair keeps a future
         * independently queued maintenance submission safe without a shadow. */
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

        const std::size_t previous_outstanding =
            outstanding_commands_.fetch_add(1u, std::memory_order_acq_rel);
        if (previous_outstanding == 0u)
        {
            active_batch_started_ns_.store(
                steadyNanoseconds(), std::memory_order_release);
            last_pending_warning_ns_.store(0u, std::memory_order_release);
            LOG_DEBUG(
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
        if (index >= reserved_slots_ || generation == 0u)
        {
            if (error)
                *error =
                    "Mapped transfer-progress completion identity is invalid";
            return MappedTransferProgress::Failed;
        }
        MappedTransferProgressCompletion &result = completion(index);
        const std::uint64_t completed =
            std::atomic_ref<std::uint64_t>(result.completed_generation)
                .load(std::memory_order_acquire);
        if (completed < generation)
        {
            constexpr std::uint64_t kPendingWarningIntervalNs =
                5'000'000'000ull;
            const std::uint64_t started = active_batch_started_ns_.load(
                std::memory_order_acquire);
            const std::uint64_t now = steadyNanoseconds();
            std::uint64_t last = last_pending_warning_ns_.load(
                std::memory_order_acquire);
            if (started != 0u && now >= started + kPendingWarningIntervalNs &&
                (last == 0u || now >= last + kPendingWarningIntervalNs) &&
                last_pending_warning_ns_.compare_exchange_strong(
                    last, now, std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                const SlotRuntime &runtime = slot_runtimes_[index];
                LOG_WARN(
                    "[MappedTransferProgressEpoch] DMA command remains pending"
                    << " device=" << config_.device.toString()
                    << " authority=" << config_.name
                    << " slot=" << index
                    << " slot_label=" << slot_labels_[index]
                    << " direction=" << directionName(runtime.direction)
                    << " requested_generation=" << generation
                    << " completed_generation=" << completed
                    << " bytes=" << command(index).bytes
                    << " in_flight=" << runtime.in_flight
                    << " outstanding="
                    << outstanding_commands_.load(std::memory_order_acquire)
                    << " published="
                    << commands_published_.load(std::memory_order_relaxed)
                    << " completed="
                    << commands_completed_.load(std::memory_order_relaxed)
                    << " dma_submissions="
                    << dma_submissions_.load(std::memory_order_relaxed));
            }
            return MappedTransferProgress::Pending;
        }

        const auto finish = [this]()
        {
            const std::size_t previous = outstanding_commands_.fetch_sub(
                1u, std::memory_order_acq_rel);
            if (previous == 0u)
                std::terminate();
            return previous == 1u;
        };
        if (completed != generation)
        {
            const bool batch_drained = finish();
            if (batch_drained)
                active_batch_started_ns_.store(
                    0u, std::memory_order_release);
            command_failures_.fetch_add(1u, std::memory_order_relaxed);
            if (error)
                *error =
                    "Mapped transfer-progress completion skipped the requested generation";
            return MappedTransferProgress::Failed;
        }

        const auto transfer_error = static_cast<MappedTransferProgressError>(
            result.error);
        if (transfer_error != MappedTransferProgressError::None ||
            result.completed_bytes != expected_bytes)
        {
            const bool batch_drained = finish();
            if (batch_drained)
                active_batch_started_ns_.store(
                    0u, std::memory_order_release);
            command_failures_.fetch_add(1u, std::memory_order_relaxed);
            if (error)
            {
                *error = std::string(
                             "Mapped transfer-progress DMA command failed: ") +
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
            active_batch_started_ns_.store(0u, std::memory_order_release);
            LOG_DEBUG(
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
             * waits for DMA or inference work on the device. */
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
                "[MappedTransferProgressEpoch] DMA submission is not on the exact device worker for "
                << config_.device.toString());
            return false;
        }

        bool all_ok = true;
        bool observed_work = false;
        std::uint64_t submissions = 0u;
        TransferEngine transfer_engine;
        const auto begin = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(reservation_mutex_);
        for (std::size_t index = 0u; index < reserved_slots_; ++index)
        {
            SlotRuntime &runtime = slot_runtimes_[index];
            if (!runtime.in_flight)
                continue;

            observed_work = true;
            bool ready = false;
            if (!context_->queryEventChecked(runtime.terminal_event, ready))
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] DMA event query failed"
                    << " device=" << config_.device.toString()
                    << " slot=" << index
                    << " label=" << slot_labels_[index]);
                all_ok = false;
                continue;
            }
            if (!ready)
            {
                in_flight_observations_.fetch_add(
                    1u, std::memory_order_relaxed);
                continue;
            }

            MappedTransferProgressCompletion &result = completion(index);
            result.completed_bytes = command(index).bytes;
            result.error = static_cast<std::uint32_t>(
                MappedTransferProgressError::None);
            std::atomic_ref<std::uint64_t>(result.completed_generation)
                .store(runtime.launched_generation, std::memory_order_release);
            runtime.in_flight = false;
        }

        for (std::size_t index = 0u; index < reserved_slots_; ++index)
        {
            SlotRuntime &runtime = slot_runtimes_[index];
            if (runtime.in_flight)
                continue;

            MappedTransferProgressCommand &entry = command(index);
            const std::uint64_t generation =
                std::atomic_ref<std::uint64_t>(entry.generation)
                    .load(std::memory_order_acquire);
            const std::uint64_t completed =
                std::atomic_ref<std::uint64_t>(
                    completion(index).completed_generation)
                    .load(std::memory_order_acquire);
            if (generation == 0u || generation <= completed)
                continue;

            observed_work = true;
            if (!runtime.reserved || !runtime.mapped_region ||
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
                command_failures_.fetch_add(1u, std::memory_order_relaxed);
                all_ok = false;
                continue;
            }

            try
            {
                const std::size_t bytes =
                    static_cast<std::size_t>(entry.bytes);
                if (runtime.direction ==
                    MappedTransferDirection::DeviceToHost)
                {
                    transfer_engine.enqueuePersistentDeviceRegionToMappedHost(
                        reinterpret_cast<const void *>(
                            static_cast<std::uintptr_t>(entry.source_address)),
                        bytes, 0u, *runtime.mapped_region, 0u, bytes,
                        config_.device, runtime.stream);
                }
                else
                {
                    transfer_engine.enqueueMappedHostToPersistentDeviceRegion(
                        *runtime.mapped_region, 0u,
                        reinterpret_cast<void *>(
                            static_cast<std::uintptr_t>(
                                entry.destination_address)),
                        bytes, 0u, bytes, config_.device, runtime.stream);
                }
                if (!context_->recordEventChecked(
                        runtime.terminal_event, runtime.stream))
                {
                    throw std::runtime_error(
                        "could not record the exact DMA completion event");
                }
                runtime.launched_generation = generation;
                runtime.in_flight = true;
                ++submissions;
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR(
                    "[MappedTransferProgressEpoch] DMA enqueue failed"
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
                    "[MappedTransferProgressEpoch] DMA enqueue threw a non-standard exception"
                    << " device=" << config_.device.toString()
                    << " slot=" << index
                    << " label=" << slot_labels_[index]);
                command_failures_.fetch_add(1u, std::memory_order_relaxed);
                all_ok = false;
            }
        }

        if (submissions != 0u)
        {
            dma_submissions_.fetch_add(submissions, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "mapped_transfer_progress_epoch_dma_submissions",
                static_cast<double>(submissions),
                "maintenance",
                config_.perf_device,
                {{"device", config_.device.toString()},
                 {"stream_class", "background_maintenance"},
                 {"copy_mechanism", "async_dma"},
                 {"inference_wait", "false"},
                 {"blocking", "false"}});
            if (PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_residency"))
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_residency",
                    "mapped_transfer_progress_epoch_dma_enqueue_host_time",
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
            .dma_submissions = dma_submissions_.load(
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
