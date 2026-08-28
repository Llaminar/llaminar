/**
 * @file ReusableExecutionWorkspace.h
 * @brief Typed model-lifetime leases for graph workspace backing allocations.
 *
 * A production runner owns graph topology, streams, publications, and request
 * state, but a prepared model may outlive that runner. CUDA/HIP runtimes can
 * retain a freed graph-referenced slab without returning it to the public free
 * counter, so destroying a large workspace and immediately allocating an
 * identical replacement can charge the same process twice. This registry
 * keeps the slab under an explicit model-lifetime owner instead.
 *
 * The registry never shares live workspace pointers. One structural owner key
 * has exactly one exclusive lease. The old runner must destroy every captured
 * graph, then seal the allocator into backing-only state before the lease can
 * become reusable. A missing transition permanently invalidates that slot; a
 * later runner cannot allocate around the lifecycle defect.
 */

#pragma once

#include "WorkspaceAllocator.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Structural identity of one independently runnable graph workspace.
     *
     * MTP depth and request policy are intentionally absent: after the prior
     * graph family is destroyed, the retained block may be replanned for a new
     * family. Fields here distinguish owners that can coexist, such as two PP
     * stages on one physical device or two tensor-parallel participants.
     */
    struct ReusableExecutionWorkspaceKey
    {
        DeviceId device = DeviceId::invalid();
        int first_layer = 0;
        int last_layer = -1;
        int tensor_parallel_participant = 0;
        int tensor_parallel_degree = 1;
        bool owns_embedding = true;
        bool owns_terminal_head = true;

        /** @return Whether the key describes one real GPU graph owner. */
        [[nodiscard]] bool valid() const noexcept
        {
            return device.is_gpu() && first_layer >= 0 &&
                   last_layer >= first_layer &&
                   tensor_parallel_degree > 0 &&
                   tensor_parallel_participant >= 0 &&
                   tensor_parallel_participant < tensor_parallel_degree;
        }

        friend bool operator==(
            const ReusableExecutionWorkspaceKey &,
            const ReusableExecutionWorkspaceKey &) noexcept = default;

        /** @brief Deterministic ordering used by the small model-local map. */
        friend bool operator<(
            const ReusableExecutionWorkspaceKey &left,
            const ReusableExecutionWorkspaceKey &right) noexcept
        {
            if (left.device != right.device)
                return left.device < right.device;
            if (left.first_layer != right.first_layer)
                return left.first_layer < right.first_layer;
            if (left.last_layer != right.last_layer)
                return left.last_layer < right.last_layer;
            if (left.tensor_parallel_participant !=
                right.tensor_parallel_participant)
            {
                return left.tensor_parallel_participant <
                       right.tensor_parallel_participant;
            }
            if (left.tensor_parallel_degree != right.tensor_parallel_degree)
                return left.tensor_parallel_degree < right.tensor_parallel_degree;
            if (left.owns_embedding != right.owns_embedding)
                return left.owns_embedding < right.owns_embedding;
            return left.owns_terminal_head < right.owns_terminal_head;
        }
    };

    /**
     * @brief Model-lifetime authority for reusable workspace backing blocks.
     *
     * The registry is shared only through ModelContextReuseContract. Its lease
     * state is independent from diagnostics and PerfStats: typed transitions
     * are the sole authority for whether retained bytes may be consumed.
     */
    class ReusableExecutionWorkspaceRegistry final
        : public std::enable_shared_from_this<
              ReusableExecutionWorkspaceRegistry>
    {
    public:
        /** @brief Complete lifecycle of one structural workspace slot. */
        enum class SlotState : std::uint8_t
        {
            Leased,   ///< One live runner may publish graph-visible pointers.
            Reusable, ///< Graph topology is gone; only backing bytes remain.
            Invalid,  ///< A required seal was skipped or failed.
        };

        /**
         * @brief Move-only exclusive ownership of one workspace allocator.
         *
         * Destruction without @ref publishReusable invalidates the slot. This
         * turns exceptional graph construction/teardown paths into an explicit
         * failed lifecycle instead of exposing stale pointers to another runner.
         */
        class Lease final
        {
        public:
            ~Lease()
            {
                if (active_)
                    invalidate("workspace lease ended without a reusable seal");
            }

            Lease(const Lease &) = delete;
            Lease &operator=(const Lease &) = delete;
            Lease(Lease &&) = delete;
            Lease &operator=(Lease &&) = delete;

            /** @return Exclusive allocator owned by this live lease. */
            [[nodiscard]] std::shared_ptr<WorkspaceAllocator> allocator() const
            {
                return allocator_;
            }

            /** @return Structural owner identity carried by this lease. */
            [[nodiscard]] const ReusableExecutionWorkspaceKey &key() const noexcept
            {
                return key_;
            }

            /**
             * @brief Seal graph metadata and publish retained backing for reuse.
             *
             * The caller must have destroyed every graph executable before
             * invoking this method. WorkspaceAllocator performs the final
             * metadata retirement; the registry then makes the transition
             * visible atomically to the next runner.
             *
             * @param error Optional precise rejection diagnostic.
             * @return True only for `Leased -> Reusable`.
             */
            bool publishReusable(std::string *error = nullptr) noexcept
            {
                if (error)
                    error->clear();
                if (!active_ || !allocator_)
                {
                    if (error)
                        *error = "workspace lease is not active";
                    return false;
                }

                std::string seal_error;
                if (!allocator_->sealReusablePrimaryBlocks(&seal_error))
                {
                    const std::string diagnostic =
                        seal_error.empty()
                            ? "workspace allocator could not seal reusable backing"
                            : std::move(seal_error);
                    invalidate(diagnostic);
                    if (error)
                        *error = diagnostic;
                    return false;
                }

                const auto registry = registry_.lock();
                if (!registry ||
                    !registry->publishReusable(key_, generation_))
                {
                    const std::string diagnostic =
                        "workspace registry rejected a stale reusable publication";
                    invalidate(diagnostic);
                    if (error)
                        *error = diagnostic;
                    return false;
                }
                active_ = false;
                return true;
            }

            /**
             * @brief Permanently reject this slot after incomplete teardown.
             * @param diagnostic First retained failure reason.
             */
            void invalidate(std::string diagnostic) noexcept
            {
                if (!active_)
                    return;
                if (const auto registry = registry_.lock())
                {
                    registry->invalidate(
                        key_, generation_, std::move(diagnostic));
                }
                active_ = false;
            }

        private:
            friend class ReusableExecutionWorkspaceRegistry;

            Lease(
                std::weak_ptr<ReusableExecutionWorkspaceRegistry> registry,
                ReusableExecutionWorkspaceKey key,
                std::uint64_t generation,
                std::shared_ptr<WorkspaceAllocator> allocator)
                : registry_(std::move(registry)),
                  key_(std::move(key)),
                  generation_(generation),
                  allocator_(std::move(allocator))
            {
            }

            std::weak_ptr<ReusableExecutionWorkspaceRegistry> registry_;
            ReusableExecutionWorkspaceKey key_;
            std::uint64_t generation_ = 0;
            std::shared_ptr<WorkspaceAllocator> allocator_;
            bool active_ = true;
        };

        /**
         * @brief Acquire one structural workspace owner for a live runner.
         * @param key Exact device/PP/TP ownership identity.
         * @param error Optional lifecycle rejection diagnostic.
         * @return Exclusive lease, or null for invalid/concurrent ownership.
         */
        [[nodiscard]] std::unique_ptr<Lease> acquire(
            const ReusableExecutionWorkspaceKey &key,
            std::string *error = nullptr)
        {
            if (error)
                error->clear();
            if (!key.valid())
            {
                if (error)
                    *error = "reusable workspace key is invalid";
                return nullptr;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            auto [it, inserted] = slots_.try_emplace(key);
            Slot &slot = it->second;
            if (inserted)
            {
                slot.allocator = std::make_shared<WorkspaceAllocator>();
                slot.state = SlotState::Leased;
                slot.generation = 1u;
            }
            else
            {
                if (slot.state != SlotState::Reusable || !slot.allocator)
                {
                    if (error)
                    {
                        *error = slot.state == SlotState::Invalid
                                     ? (slot.diagnostic.empty()
                                            ? "reusable workspace slot is invalid"
                                            : slot.diagnostic)
                                     : "reusable workspace slot already has a live runner";
                    }
                    return nullptr;
                }
                slot.state = SlotState::Leased;
                slot.diagnostic.clear();
                ++slot.generation;
            }

            return std::unique_ptr<Lease>(new Lease(
                weak_from_this(), key, slot.generation, slot.allocator));
        }

        /**
         * @brief Sum sealed primary blocks for one physical device.
         * @param device GPU whose reusable bytes should be credited.
         * @return Checked sum; max size_t denotes arithmetic saturation.
         */
        [[nodiscard]] std::size_t retainedPrimaryBytes(
            DeviceId device) const noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::size_t total = 0;
            for (const auto &[key, slot] : slots_)
            {
                if (key.device != device ||
                    slot.state != SlotState::Reusable || !slot.allocator)
                {
                    continue;
                }
                const std::size_t bytes =
                    slot.allocator->retainedPrimaryBytes();
                if (bytes > std::numeric_limits<std::size_t>::max() - total)
                    return std::numeric_limits<std::size_t>::max();
                total += bytes;
            }
            return total;
        }

        /** @return Whether no slot has entered the permanent Invalid state. */
        [[nodiscard]] bool valid() const noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &[_, slot] : slots_)
            {
                if (slot.state == SlotState::Invalid)
                    return false;
            }
            return true;
        }

        /** @return First retained invalid-slot diagnostic, if any. */
        [[nodiscard]] std::string diagnostic() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &[_, slot] : slots_)
            {
                if (slot.state == SlotState::Invalid)
                    return slot.diagnostic;
            }
            return {};
        }

    private:
        struct Slot
        {
            SlotState state = SlotState::Invalid;
            std::uint64_t generation = 0;
            std::shared_ptr<WorkspaceAllocator> allocator;
            std::string diagnostic;
        };

        /** @brief Commit one exact lease generation as reusable. */
        bool publishReusable(
            const ReusableExecutionWorkspaceKey &key,
            std::uint64_t generation) noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = slots_.find(key);
            if (it == slots_.end() ||
                it->second.state != SlotState::Leased ||
                it->second.generation != generation ||
                !it->second.allocator)
            {
                return false;
            }
            it->second.state = SlotState::Reusable;
            it->second.diagnostic.clear();
            return true;
        }

        /** @brief Invalidate one exact live lease generation. */
        void invalidate(
            const ReusableExecutionWorkspaceKey &key,
            std::uint64_t generation,
            std::string diagnostic) noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = slots_.find(key);
            if (it == slots_.end() ||
                it->second.generation != generation)
            {
                return;
            }
            it->second.state = SlotState::Invalid;
            if (it->second.diagnostic.empty())
            {
                it->second.diagnostic = diagnostic.empty()
                    ? "reusable workspace lifecycle failed"
                    : std::move(diagnostic);
            }
        }

        mutable std::mutex mutex_;
        std::map<ReusableExecutionWorkspaceKey, Slot> slots_;
    };
} // namespace llaminar2
