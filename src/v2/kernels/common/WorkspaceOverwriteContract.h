/**
 * @file WorkspaceOverwriteContract.h
 * @brief Checked live extents and ordered writers for uninitialized GPU scratch.
 *
 * Persistent allocation provides storage, never an initial value. A complete
 * overwrite producer must define every element its consumer will read, including
 * empty arithmetic partitions and absent route contributions. Bytes outside
 * that consumed extent remain unspecified. Read-modify-write accumulators and
 * initialized recurrent/protocol state cannot use this contract.
 *
 * These non-owning values perform no allocation, transfer, clear, synchronization,
 * or memory accounting. Existing arenas and PhysicalMemoryAuthority retain all
 * ownership. Launch builders execute immediately on the host during capture or
 * enqueue; they are not GPU host callbacks. The read view becomes available only
 * after successful producer submission and retains that exact stream. Cross-
 * stream consumers still require the ordinary explicit event edge.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /** @brief Validated contiguous extent that a producer completely overwrites. */
    class WorkspaceOverwriteExtent final
    {
    public:
        /**
         * @brief Describe rows, columns and optional independent partial planes.
         * @param rows Consumed rows; zero describes an empty publication.
         * @param columns Positive physical row stride consumed by the reader.
         * @param planes Positive number of independently written partial planes.
         * @return Exact live extent, without allocation or tile padding.
         * @throws std::invalid_argument for empty strides or planes.
         * @throws std::overflow_error if the element product is unrepresentable.
         */
        [[nodiscard]] static WorkspaceOverwriteExtent matrix(
            std::size_t rows, std::size_t columns, std::size_t planes = 1u)
        {
            if (columns == 0u || planes == 0u)
                throw std::invalid_argument("complete workspace overwrite requires positive columns and planes");
            return WorkspaceOverwriteExtent(product(product(rows, columns), planes));
        }

        /** @return Exact number of consumed elements, excluding unused capacity. */
        [[nodiscard]] std::size_t elements() const noexcept { return elements_; }

        /**
         * @brief Prove an existing allocation contains the complete live extent.
         * @tparam T Element representation used by both producer and consumer.
         * @param capacity_bytes Available existing storage, never a clear extent.
         * @return Exact live bytes after overflow and capacity validation.
         * @throws std::overflow_error if live bytes are unrepresentable.
         * @throws std::length_error if existing storage cannot contain the extent.
         */
        template <class T>
        [[nodiscard]] std::size_t requireCapacity(std::size_t capacity_bytes) const
        {
            const auto live_bytes = product(elements_, sizeof(T));
            if (live_bytes > capacity_bytes)
                throw std::length_error("complete workspace overwrite exceeds bound storage capacity");
            return live_bytes;
        }

    private:
        /** @brief Restrict construction to checked extent factories. */
        explicit WorkspaceOverwriteExtent(std::size_t elements) : elements_(elements) {}

        /** @brief Reject overflow without constructing an invalid extent. */
        static std::size_t product(std::size_t left, std::size_t right)
        {
            if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)
                throw std::overflow_error("complete workspace overwrite extent overflow");
            return left * right;
        }

        std::size_t elements_; ///< Immutable live element extent.
    };

    template <class T> class WorkspaceOverwrite;

    /**
     * @brief Read-only live scratch whose full producer was successfully enqueued.
     * @tparam T Scratch element representation.
     *
     * This is a submission-order proof, not a host observation of GPU completion.
     * Only WorkspaceOverwrite can construct it, after the producer succeeds.
     */
    template <class T>
    class EnqueuedWorkspaceRead final
    {
    public:
        /** @return Only the consumed, completely overwritten element span. */
        [[nodiscard]] std::span<const T> values() const noexcept { return values_; }
        /** @return Exact producer stream on which the next reader must be ordered. */
        [[nodiscard]] void *stream() const noexcept { return stream_; }

    private:
        friend class WorkspaceOverwrite<T>;
        /** @brief Retain the validated extent after successful producer submission. */
        EnqueuedWorkspaceRead(std::span<const T> values, void *stream)
            : values_(values), stream_(stream) {}
        std::span<const T> values_; ///< Initialized live span; never allocation capacity.
        void *stream_; ///< Exact non-null producer stream.
    };

    /**
     * @brief Bind dirty scratch to one complete writer and its ordered reader.
     * @tparam T Scratch element representation.
     *
     * The binding exposes no read accessor before submission. A producer is
     * responsible for unique writes to every live element; focused poisoned-
     * replay and native graph economy gates prove that implementation promise.
     */
    template <class T>
    class WorkspaceOverwrite final
    {
    public:
        /**
         * @brief Bind existing storage without inspecting or initializing bytes.
         * @param storage Aligned persistent device storage owned by the caller.
         * @param capacity_bytes Exact accessible capacity at this pointer.
         * @param extent Complete live extent written and then consumed.
         * @param stream Exact non-null stream shared by the writer and reader.
         * @return Immutable non-owning launch binding.
         * @throws std::invalid_argument for missing/misaligned storage or stream.
         * @throws std::length_error for insufficient existing storage.
         * @throws std::overflow_error for an unrepresentable byte extent.
         */
        [[nodiscard]] static WorkspaceOverwrite bind(
            T *storage, std::size_t capacity_bytes,
            WorkspaceOverwriteExtent extent, void *stream)
        {
            if (!storage || !stream ||
                reinterpret_cast<std::uintptr_t>(storage) % alignof(T) != 0u)
                throw std::invalid_argument("complete workspace overwrite requires aligned storage and an exact non-null stream");
            (void)extent.template requireCapacity<T>(capacity_bytes);
            return WorkspaceOverwrite(storage, extent.elements(), stream);
        }

        /**
         * @brief Submit a complete writer before exposing its read-only live span.
         * @param producer Immediate launch builder receiving writable live span
         *        and exact stream; returns false if any required write failed.
         * @param consumer Immediate launch builder receiving the enqueued read
         *        view; it must order its read on that same stream or an event edge.
         * @return False if the writer fails, otherwise the consumer's result.
         * @note Failed/throwing producers never invoke the consumer. No host
         *       wait, clear, allocation or alternate execution occurs here.
         */
        template <class Producer, class Consumer>
        [[nodiscard]] bool overwriteThenRead(Producer &&producer, Consumer &&consumer) const
        {
            if (!std::forward<Producer>(producer)(values_, stream_))
                return false;
            const EnqueuedWorkspaceRead<T> read(values_, stream_);
            return std::forward<Consumer>(consumer)(read);
        }

    private:
        /** @brief Construct only after storage, extent and stream validation. */
        WorkspaceOverwrite(T *storage, std::size_t elements, void *stream)
            : values_(storage, elements), stream_(stream) {}
        std::span<T> values_; ///< Complete producer extent, excluding retained padding.
        void *stream_; ///< Exact stream used for both ordered submissions.
    };
} // namespace llaminar2
