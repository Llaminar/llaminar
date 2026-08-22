/**
 * @file MappedTransferProgressEpoch.h
 * @brief Node-local asynchronous-DMA scheduler for background expert movement.
 *
 * A physical ExpertOverlay fabric owns one epoch per local GPU. Permanent lane
 * slots publish typed D2H or H2D commands. The maintenance service submits
 * those commands on setup-owned background streams and observes per-slot
 * events. Inference graphs never capture, join, or wait for this scheduler.
 */

#pragma once

#include "MappedTransferProgressABI.h"
#include "backends/DeviceId.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class IWorkerGPUContext;
    class MappedHostTransferRegion;

    /** Immutable direction owned by one permanent DMA slot. */
    enum class MappedTransferDirection : std::uint8_t
    {
        DeviceToHost = 0u, ///< Immutable residency bank into mapped staging.
        HostToDevice = 1u, ///< Mapped staging into an inactive residency bank.
    };

    /** Result of one non-blocking permanent-slot completion observation. */
    enum class MappedTransferProgress : std::uint8_t
    {
        Pending, ///< The command has not release-published completion yet.
        Ready,   ///< Every requested byte is complete and system-visible.
        Failed,  ///< The device rejected the immutable command geometry.
    };

    /** Model-lifetime evidence for one per-device retained progress epoch. */
    struct MappedTransferProgressEpochStats
    {
        std::uint64_t slots_reserved = 0u;
        std::uint64_t commands_published = 0u;
        std::uint64_t commands_completed = 0u;
        std::uint64_t bytes_completed = 0u;
        std::uint64_t dma_submissions = 0u;
        std::uint64_t idle_submission_skips = 0u;
        std::uint64_t in_flight_observations = 0u;
        std::uint64_t command_failures = 0u;
    };

    class MappedTransferProgressEpoch;
    /**
     * @brief Exclusive permanent command slot leased to one physical lane.
     *
     * The lease is move-only and retains the epoch. Exactly one command may be
     * outstanding at a time; publication before completion is rejected instead
     * of overwriting mapped bytes that a retained replay may still claim.
     */
    class MappedTransferProgressSlot final
    {
    public:
        /** Construct an empty, non-publishable slot for aggregate members. */
        MappedTransferProgressSlot() = default;

        /** Terminate if an owner drops a command whose device may still use it. */
        ~MappedTransferProgressSlot();

        MappedTransferProgressSlot(const MappedTransferProgressSlot &) = delete;
        MappedTransferProgressSlot &operator=(
            const MappedTransferProgressSlot &) = delete;
        MappedTransferProgressSlot(
            MappedTransferProgressSlot &&other) noexcept;
        MappedTransferProgressSlot &operator=(
            MappedTransferProgressSlot &&other) noexcept;

        /** @return Whether this lease names one reserved epoch slot. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Whether a published generation remains unobserved. */
        [[nodiscard]] bool pending() const noexcept { return pending_; }

        /** @return Permanent zero-based slot index, or SIZE_MAX when empty. */
        [[nodiscard]] std::size_t index() const noexcept { return index_; }

        /**
         * @brief Publish a bounded device-to-mapped-host DMA command.
         * @param source Stable base of an immutable device residency region.
         * @param source_capacity Complete byte capacity of @p source.
         * @param source_offset First source byte copied by this generation.
         * @param bytes Positive byte count within both fixed regions.
         * @return Positive monotonic slot generation.
         * @throws std::logic_error for the wrong slot direction or overlap.
         * @throws std::invalid_argument for invalid bounds or identity.
         */
        std::uint64_t publishDeviceToMappedHost(
            const void *source,
            std::size_t source_capacity,
            std::size_t source_offset,
            std::size_t bytes);

        /**
         * @brief Publish a mapped-host-to-device DMA command.
         * @param destination Stable base of an inactive device residency region.
         * @param destination_capacity Complete byte capacity of @p destination.
         * @param destination_offset First destination byte written by this generation.
         * @param bytes Positive byte count within both fixed regions.
         * @return Positive monotonic slot generation.
         * @throws std::logic_error for the wrong slot direction or overlap.
         * @throws std::invalid_argument for invalid bounds or identity.
         */
        std::uint64_t publishMappedHostToDevice(
            void *destination,
            std::size_t destination_capacity,
            std::size_t destination_offset,
            std::size_t bytes);

        /**
         * @brief Acquire the device completion without waiting.
         * @param error Optional exact device validation failure.
         * @return Pending, Ready, or Failed for the current generation.
         */
        MappedTransferProgress poll(std::string *error = nullptr) noexcept;

    private:
        friend class MappedTransferProgressEpoch;

        /** Retain one factory-validated permanent index. */
        MappedTransferProgressSlot(
            std::shared_ptr<MappedTransferProgressEpoch> epoch,
            std::size_t index) noexcept;

        /** Move state only after proving the destination has no live command. */
        void moveFrom(MappedTransferProgressSlot &&other) noexcept;

        std::shared_ptr<MappedTransferProgressEpoch> epoch_;
        std::size_t index_ = static_cast<std::size_t>(-1);
        std::uint64_t next_generation_ = 0u;
        std::uint64_t pending_generation_ = 0u;
        bool pending_ = false;
    };

    /**
     * @brief One finite asynchronous-DMA scheduler shared by GPU relay lanes.
     *
     * Construction allocates every stream and event. Maintenance publishes
     * fixed commands and invokes @ref submitOutstandingProgress; the exact GPU
     * worker then queries prior events and enqueues every newly visible command.
     * A stream per permanent slot preserves topology-declared parallelism and
     * lets backend copy engines overlap opposite directions without allowing
     * movement kernels to occupy inference compute units.
     */
    class MappedTransferProgressEpoch final
        : public std::enable_shared_from_this<
              MappedTransferProgressEpoch>
    {
    public:
        /** Immutable device, capacity, byte bound, and evidence identity. */
        struct Config
        {
            DeviceId device = DeviceId::invalid();
            std::size_t slot_capacity = 0u;
            std::size_t maximum_bytes = 0u;
            std::string name;
            std::string perf_device;
        };

        /**
         * @brief Materialize a complete DMA epoch during model setup.
         * @throws std::invalid_argument for incomplete geometry or identity.
         * @throws std::runtime_error for backend, allocation, or capture failure.
         */
        [[nodiscard]] static std::shared_ptr<MappedTransferProgressEpoch>
        create(Config config);

        /** Destroy only after every command and event is quiescent. */
        ~MappedTransferProgressEpoch();

        MappedTransferProgressEpoch(
            const MappedTransferProgressEpoch &) = delete;
        MappedTransferProgressEpoch &operator=(
            const MappedTransferProgressEpoch &) = delete;

        /**
         * @brief Permanently reserve one typed setup-time lane slot.
         * @param direction Immutable D2H/H2D role for every generation.
         * @param mapped_region Registered host pages owned by this slot.
         * @param diagnostic_label Stable topology identity used in failures.
         * @throws std::runtime_error when topology accounting under-sized capacity.
         */
        [[nodiscard]] MappedTransferProgressSlot reserveSlot(
            MappedTransferDirection direction,
            std::shared_ptr<MappedHostTransferRegion> mapped_region,
            std::string diagnostic_label = {});

        /**
         * @brief Prove that immutable transfer addresses belong to this GPU.
         * @param addresses Non-null device allocations consumed by a command.
         * @param role Stable lane/endpoint role included in failures.
         * @param error Optional exact runtime ownership diagnostic.
         * @return True only when every pointer belongs to @ref device().
         *
         * Validation is a movement-admission operation, never an inference-hot-
         * path operation. The query runs on the exact device worker so CUDA and
         * HIP pointer inspection cannot accidentally use another runtime
         * context. Mapped-host aliases are deliberately excluded: callers pass
         * only the device-allocation side of a mapped transfer command.
         */
        [[nodiscard]] bool validateDeviceAddresses(
            std::span<const void *const> addresses,
            std::string_view role,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Query and enqueue one DMA progress pass on the exact worker.
         *
         * This method must run on the exact device worker. It performs only an
         * event queries, DMA submissions, and event records; it never synchronizes or
         * makes the inference stream wait. Idle and already-running epochs are
         * successful no-ops with distinct evidence counters.
         *
         * @return True after a launch or safe bounded skip.
         */
        [[nodiscard]] bool launchOutstandingProgress() noexcept;

        /**
         * @brief Route outstanding maintenance progress through the owning worker.
         *
         * Callers outside the worker wait only for the bounded host callback
         * that enqueues the graph and terminal event; they never wait for GPU
         * work. When no command is outstanding this is an allocation-free
         * immediate success, avoiding a steady-state per-token worker hop.
         *
         * @return True after an ordered enqueue or an idle no-op.
         */
        [[nodiscard]] bool submitOutstandingProgress() noexcept;

        /** @return Whether at least one published command is not yet observed. */
        [[nodiscard]] bool hasOutstandingCommands() const noexcept
        {
            return outstanding_commands_.load(std::memory_order_acquire) != 0u;
        }

        /** @return Exact GPU whose address space owns every command. */
        [[nodiscard]] DeviceId device() const noexcept { return config_.device; }

        /** @return Immutable number of topology-accounted slots. */
        [[nodiscard]] std::size_t slotCapacity() const noexcept
        {
            return config_.slot_capacity;
        }

        /** @return Immutable positive payload limit for each slot. */
        [[nodiscard]] std::size_t maximumBytes() const noexcept
        {
            return config_.maximum_bytes;
        }

        /** @return First persistent background stream, for diagnostics only. */
        [[nodiscard]] void *executionStream() const noexcept
        {
            return slot_runtimes_.empty()
                       ? nullptr
                       : slot_runtimes_.front().stream;
        }

        /** @return Race-safe cumulative proof counters. */
        [[nodiscard]] MappedTransferProgressEpochStats stats() const noexcept;

    private:
        friend class MappedTransferProgressSlot;

        /** Store validated identity before the factory performs GPU setup. */
        explicit MappedTransferProgressEpoch(Config config);

        /** Allocate command arrays plus one background stream/event per slot. */
        void materialize();

        /** Publish one direction-checked command for a validated permanent slot. */
        std::uint64_t publish(
            std::size_t index,
            std::uint64_t generation,
            MappedTransferDirection direction,
            const void *device_region,
            std::size_t device_capacity,
            std::size_t device_offset,
            std::size_t bytes);

        /** Acquire one matching completion and retire its outstanding count. */
        MappedTransferProgress poll(
            std::size_t index,
            std::uint64_t generation,
            std::size_t expected_bytes,
            std::string *error) noexcept;

        /** @return Host-owned command cache line. */
        [[nodiscard]] MappedTransferProgressCommand &command(
            std::size_t index) noexcept;

        /** @return Host-owned completion cache line. */
        [[nodiscard]] MappedTransferProgressCompletion &completion(
            std::size_t index) noexcept;

        Config config_;
        IBackend *backend_ = nullptr;
        IWorkerGPUContext *context_ = nullptr;
        std::unique_ptr<MappedTransferProgressCommand[]> commands_;
        std::unique_ptr<MappedTransferProgressCompletion[]> completions_;
        /** Worker-owned runtime for one independently schedulable DMA slot. */
        struct SlotRuntime
        {
            MappedTransferDirection direction =
                MappedTransferDirection::DeviceToHost;
            std::shared_ptr<MappedHostTransferRegion> mapped_region;
            void *stream = nullptr;
            void *terminal_event = nullptr;
            std::uint64_t launched_generation = 0u;
            bool reserved = false;
            bool in_flight = false;
        };
        std::vector<SlotRuntime> slot_runtimes_;
        mutable std::mutex reservation_mutex_;
        std::size_t reserved_slots_ = 0u;
        /** Setup-only labels indexed by permanent command-slot identity. */
        std::vector<std::string> slot_labels_;
        std::atomic<std::size_t> outstanding_commands_{0u};
        /** Steady-clock origin of the current non-empty command batch. */
        std::atomic<std::uint64_t> active_batch_started_ns_{0u};
        /** Last throttled warning time for an unexpectedly retained command. */
        std::atomic<std::uint64_t> last_pending_warning_ns_{0u};
        std::atomic<std::uint64_t> commands_published_{0u};
        std::atomic<std::uint64_t> commands_completed_{0u};
        std::atomic<std::uint64_t> bytes_completed_{0u};
        std::atomic<std::uint64_t> dma_submissions_{0u};
        std::atomic<std::uint64_t> idle_submission_skips_{0u};
        std::atomic<std::uint64_t> in_flight_observations_{0u};
        std::atomic<std::uint64_t> command_failures_{0u};
    };
} // namespace llaminar2
