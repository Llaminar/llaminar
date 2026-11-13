/**
 * @file NativeCollectiveBytes.h
 * @brief Owned enqueue outcomes and borrowed byte operands for original native broadcasts.
 *
 * Device metadata remains the communication authority. These immutable views
 * admit geometry and retain no count mirror, allocation or communicator. Their
 * buffers, metadata and optional receipt must remain alive and ordered on the
 * exact original stream through captured execution and terminal retirement.
 */
#pragma once

#include "transfer/CapturedTransferChannelProtocol.h"

#include <cstddef>
#include <cstdint>
#include "NativeCollectiveCompletion.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace llaminar2
{
    /** @brief Required origin-aware completed payload and abortable GROUP semantics, not receipt/work ABI. */
    inline constexpr std::uint32_t kNativeCompletedProtocolVersion = 3;

    /**
     * @brief One caller-owned enqueue outcome; concurrent participants share no diagnostic string.
     * Successful calls carry an empty error value. A failed call retains its
     * original runtime/transport diagnostic even after another participant fails.
     */
    class NativeCollectiveEnqueueResult final
    {
    public:
        /** @return Successful original enqueue without an error payload. */
        [[nodiscard]] static NativeCollectiveEnqueueResult success()
        { return NativeCollectiveEnqueueResult(Disposition::Enqueued, std::string{}); }
        /**
         * @param message Original nonempty failure, transferred to this call alone.
         * @return Failed enqueue with its own immutable diagnostic value.
         * @throws std::invalid_argument If a failed call has no diagnostic.
         */
        [[nodiscard]] static NativeCollectiveEnqueueResult failure(std::string message)
        {
            if (message.empty())
                throw std::invalid_argument("Native collective failure requires its original diagnostic");
            return NativeCollectiveEnqueueResult(Disposition::Failed, std::move(message));
        }
        /** @return True only when this call completed native enqueue successfully. */
        explicit operator bool() const noexcept { return disposition_ == Disposition::Enqueued; }
        /** @return An owned diagnostic copy, never a reference to another caller's mutable state. */
        [[nodiscard]] std::string message() const { return message_; }
    private:
        enum class Disposition : std::uint8_t { Enqueued, Failed };
        /** @brief Keep failure disposition after a move; only diagnostic ownership transfers. */
        NativeCollectiveEnqueueResult(Disposition disposition, std::string message)
            : disposition_(disposition), message_(std::move(message)) {}
        Disposition disposition_;
        std::string message_;
    };

    /** @brief One addressed edge in the original communicator; unrelated members never participate. */
    class NativeByteEdge final {
    public:
        /** @brief Freeze distinct original producer/consumer coordinates before capture.
         * @param producer Original source coordinate. @param consumer Actual consumer coordinate. */
        NativeByteEdge(int producer,int consumer):producer_(producer),consumer_(consumer) {
            if(producer<0 || consumer<0 || producer==consumer)
                throw std::invalid_argument("Native addressed byte edge requires distinct original endpoints");
        }
        /** @return Original producer coordinate, never physical device ordinal. */
        int producer()const noexcept{return producer_;}
        /** @return Original consumer coordinate, never physical device ordinal. */
        int consumer()const noexcept{return consumer_;}
        /** @brief Authenticate both original endpoints before entering native work.
         * @param degree Complete original communicator population.
         * @param participant Actual calling original coordinate.
         * @return Whether the caller is one of the two valid original endpoints. */
        bool contains(int degree,int participant)const noexcept {
            return degree>=2 && producer_<degree && consumer_<degree &&
                (participant==producer_ || participant==consumer_);
        }
        /** @brief Resolve the actual addressed peer without inferring a two-rank topology.
         * @param participant Actual original local endpoint coordinate.
         * @return Exact opposite original endpoint.
         * @throws std::invalid_argument For a caller unrelated to the edge. */
        int peer(int participant)const {
            if(participant!=producer_ && participant!=consumer_)
                throw std::invalid_argument("Native addressed byte caller is unrelated to the original edge");
            return participant==producer_?consumer_:producer_;
        }
    private:
        int producer_,consumer_;
    };

    /** @brief One native operation family; fixed traffic needs no dynamic header. */
    enum class NativeByteExtent : std::uint8_t { Fixed, DevicePrefix, DeviceSourceSlice };

    /** @brief Original root-relative byte descriptor, interpreted inside the admitted subspan origin. */
    using NativeByteSourceSlice = CapturedTransferSourceSlice;
    static_assert(sizeof(NativeByteSourceSlice) == 16 &&
        offsetof(NativeByteSourceSlice, byte_offset) == 0 &&
        offsetof(NativeByteSourceSlice, live_bytes) == 8 &&
        std::is_standard_layout_v<NativeByteSourceSlice>);

    /** @brief Checked original communication operands without reading GPU values. */
    class NativeCollectiveBytes final
    {
    public:
        /**
         * @brief Declare an exact known wire length; no header/count is transmitted.
         * @param bytes Exact positive useful wire bytes, not unused capacity.
         * @return Immutable ordinary native Broadcast operands.
         */
        [[nodiscard]] static NativeCollectiveBytes fixed(std::size_t bytes)
        { return NativeCollectiveBytes(NativeByteExtent::Fixed, bytes, nullptr, nullptr, 0, nullptr); }

        /**
         * @brief Borrow an original UINT64 prefix, written by root or acquired by receivers.
         * @param extent Original aligned device communication field, never host-read.
         * @param completion Required original frame/error/receipt binding; no host GPU reads.
         * @param capacity Positive admitted native progress/destination bound.
         * @return Immutable borrowed prefix operands.
         */
        [[nodiscard]] static NativeCollectiveBytes prefix(std::uint64_t* extent, std::size_t capacity, const ncclLlaminarCompletionOperand* completion)
        { return NativeCollectiveBytes(NativeByteExtent::DevicePrefix, capacity, extent, nullptr, 0, completion); }

        /**
         * @brief Bind the root's original absolute descriptor and exact admitted source subspan.
         * @param slice Original device offset/bytes publication, never a host snapshot.
         * @param source_bytes Exact admitted source subspan extent, at least native capacity.
         * @param source_offset_origin Immutable absolute origin of that subspan; device descriptors are unchanged.
         * @param completion Required original frame/error/receipt binding.
         * @param capacity Maximum live payload for native progress and destination storage.
         * @return Root-only borrowed original descriptor; no metadata bank is created.
         */
        [[nodiscard]] static NativeCollectiveBytes sourceSlice(
            NativeByteSourceSlice* slice, std::size_t source_bytes, std::size_t capacity, const ncclLlaminarCompletionOperand* completion,
            std::uint64_t source_offset_origin = 0)
        {
            if (!slice || reinterpret_cast<std::uintptr_t>(slice) % alignof(NativeByteSourceSlice) ||
                reinterpret_cast<std::uintptr_t>(slice) >
                    std::numeric_limits<std::uintptr_t>::max() - sizeof(NativeByteSourceSlice))
                throw std::invalid_argument("Native byte slice requires its original aligned descriptor");
            return NativeCollectiveBytes(NativeByteExtent::DeviceSourceSlice, capacity,
                &slice->live_bytes, slice, source_bytes, completion, source_offset_origin);
        }

        /**
         * @brief Bind a receiver's original extent field; receivers own no source slice.
         * @param extent Original aligned writable device communication field.
         * @param completion Required original frame/error/receipt binding; no host GPU reads.
         * @param capacity Positive admitted destination/progress bound.
         * @return Receiver-only borrowed extent, checked against the actual root at dispatch.
         */
        [[nodiscard]] static NativeCollectiveBytes receiveSlice(std::uint64_t* extent, std::size_t capacity, const ncclLlaminarCompletionOperand* completion)
        { return NativeCollectiveBytes(NativeByteExtent::DeviceSourceSlice, capacity, extent, nullptr, 0, completion); }

        /** @return Exact immutable wire extent interpretation. */
        [[nodiscard]] NativeByteExtent kind() const noexcept { return kind_; }
        /** @return Admitted capacity; only Fixed interprets it as live wire bytes. */
        [[nodiscard]] std::size_t capacityBytes() const noexcept { return capacity_; }
        /** @return Original root/receiver UINT64 communication field, never host-read. */
        [[nodiscard]] std::uint64_t* extent() const noexcept { return extent_; }
        /** @return Original root descriptor or null for a receiver/prefix/fixed operation. */
        [[nodiscard]] const NativeByteSourceSlice* sourceSlice() const noexcept { return slice_; }
        /** @return Original source bank bound, zero for participants without a source slice. */
        [[nodiscard]] std::size_t sourceCapacityBytes() const noexcept { return source_bytes_; }
        /** @return Immutable absolute lower bound of the root's exact admitted subspan. */
        [[nodiscard]] std::uint64_t sourceOffsetOrigin() const noexcept { return source_offset_origin_; }
        /** @return Original device completion/frame/error binding; absent only for fixed traffic. */
        [[nodiscard]] const ncclLlaminarCompletionOperand* completion() const noexcept { return completion_; }

        /**
         * @brief Authenticate participant/root roles without dereferencing device metadata.
         * @param degree Original communicator population.
         * @param participant Calling original communicator coordinate.
         * @param root Original rooted broadcast coordinate, not a device ordinal.
         * @return Whether source-only slice bindings belong to the actual root.
         */
        [[nodiscard]] bool participantValid(int degree, int participant, int root) const noexcept
        {
            if (degree < 2 || participant < 0 || participant >= degree || root < 0 || root >= degree)
                return false;
            return kind_ != NativeByteExtent::DeviceSourceSlice || (slice_ != nullptr) == (participant == root);
        }

    private:
        /**
         * @brief Check static representability only; device values remain unread.
         * @param kind Exact fixed/prefix/source-slice interpretation.
         * @param capacity Positive native wire progress/destination bound.
         * @param extent Original device field, absent only for fixed traffic.
         * @param slice Original root-only descriptor or null.
         * @param source_bytes Original source bank, at least capacity when a slice exists.
         * @param completion Original admitted device interval/error/receive evidence binding.
         * @param source_offset_origin Immutable original source subspan lower bound; zero without a slice.
         * @throws std::invalid_argument For invalid static geometry or alignment.
         */
        NativeCollectiveBytes(NativeByteExtent kind, std::size_t capacity, std::uint64_t* extent,
            const NativeByteSourceSlice* slice, std::size_t source_bytes, const ncclLlaminarCompletionOperand* completion,
            std::uint64_t source_offset_origin = 0)
            : kind_(kind), capacity_(capacity), extent_(extent), slice_(slice), source_bytes_(source_bytes), completion_(completion), source_offset_origin_(source_offset_origin)
        {
            const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
            if (!capacity || capacity > maximum || source_bytes > maximum ||
                (kind != NativeByteExtent::Fixed && (!completion ||
                    reinterpret_cast<std::uintptr_t>(completion) > std::numeric_limits<std::uintptr_t>::max() - sizeof(ncclLlaminarCompletionOperand) ||
                    reinterpret_cast<std::uintptr_t>(completion) % alignof(ncclLlaminarCompletionOperand) || !extent ||
                    reinterpret_cast<std::uintptr_t>(extent) % alignof(std::uint64_t))) ||
                (slice && (source_bytes < capacity || source_offset_origin > std::numeric_limits<std::uint64_t>::max() - source_bytes)) ||
                (!slice && source_offset_origin))
                throw std::invalid_argument("Native byte operation has invalid original extent/bank geometry");
        }

        NativeByteExtent kind_; ///< Original fixed/prefix/source-slice operation.
        std::size_t capacity_; ///< Admitted wire progress/destination capacity.
        std::uint64_t* extent_; ///< Borrowed original device communication metadata.
        const NativeByteSourceSlice* slice_; ///< Borrowed root-only descriptor, never copied to host.
        std::size_t source_bytes_; ///< Original source bank bound for device slice validation.
        const ncclLlaminarCompletionOperand* completion_; ///< Original admitted device metadata, never host-read.
        std::uint64_t source_offset_origin_; ///< Immutable lower bound; never a host mirror of the descriptor.
    };
}
