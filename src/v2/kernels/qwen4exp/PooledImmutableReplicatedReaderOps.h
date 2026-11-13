/**
 * @file PooledImmutableReplicatedReaderOps.h
 * @brief Device-owned reader handshakes around one original replicated PLE lookup.
 *
 * These records carry only protocol evidence. The original coordinator frame
 * owns the interval, the Rank-owned replicated query owns input equivalence,
 * and the original controller owns failure. No reader owns a source ticket,
 * physical pin, cache policy, or independent lookup sequence.
 */
#pragma once
#include "PooledImmutableCoordinatorState.h"
#include "kernels/common/SamplingMath.h"
#include "transfer/CapturedTransferChannelProtocol.h"
#include "collective/NativeCollectiveCompletion.h"
#include <initializer_list>
#include <stdexcept>

namespace llaminar2::qwen4exp::replicated_reader {
namespace p = adaptive_lookup::pooled;
using adaptive_lookup::Status;

/** @brief Exact live metadata prefix; message roles and epochs belong to the channel. */
struct Header {
    p::Interval interval;
    std::uint32_t version, original_ids, row_elements, coordinate, consumers;
    Status status;
};
static_assert(sizeof(Header) == 48);
static_assert(std::is_trivially_copyable_v<Header>);
/** @brief Graph-arena protocol bank; overwritten by original producers before every use. */
struct alignas(64) Control {
    Header outgoing, incoming;
    std::uint64_t header_bytes, payload_bytes, output_bytes;
};
static_assert(sizeof(Control) == 128);


/** @brief Immutable original reader coordinate and exact graph output envelope. */
struct Geometry {
    std::uint32_t coordinate, consumers, maximum_original, row_elements;
};

/** @return Whether immutable reader geometry is representable without empty population aliases. */
QFN_ADAPTIVE_HD inline bool valid(Geometry g)
{
    return g.consumers > 1 && g.coordinate < g.consumers &&
        g.maximum_original && g.maximum_original % 16 == 0 && g.row_elements &&
        static_cast<std::uint64_t>(g.maximum_original) * g.row_elements <=
            (~std::uint64_t{0}) / sizeof(float);
}

/** @brief Original fixed Header/result/ACK operation roles, not independent transport channels. */
enum class Wire : std::uint32_t { Header, Payload, Ack };
/** @return Native logical operation key shared across exclusive shape variants; frame/shape are validated separately. */
inline std::uint64_t nativeMessageKey(Wire wire, Geometry geometry, std::uint32_t version)
{
    if (!valid(geometry) || !version || static_cast<std::uint32_t>(wire) > static_cast<std::uint32_t>(Wire::Ack))
        throw std::invalid_argument("Native reader message requires its original declared role/geometry/version");
    std::uint64_t value = 0x51464e5245414445ULL;
    for (const auto word : {static_cast<std::uint64_t>(wire), std::uint64_t{geometry.coordinate},
            std::uint64_t{geometry.row_elements}, std::uint64_t{version}, std::uint64_t{0x4e434c5f52454144ULL}})
        value = (value ^ word) * 1099511628211ULL;
    return value ? value : 1;
}

/** @return Original first-error validity; controller Ok is deliberately not an admission surrogate. */
QFN_ADAPTIVE_HD inline bool forwardValid(const int* first_error)
{
    return first_error &&
        *first_error == static_cast<int>(sampling_math::DeviceGenerationError::None);
}

/**
 * @return Declared fixed Header extent only after its whole original native launch.
 * @param original Immutable authority of that exact native communicator.
 * @param aborted System-acquired original communicator abort, never a header-selected pointer.
 * @param first_error Same original sticky controller error.
 * This is not a counted completion receipt. Fixed Broadcast has a declared
 * extent; original native launch ordering and failure propagation establish its
 * delivery. An aborted or failed operation cannot reuse an old matching header.
 */
QFN_ADAPTIVE_HD inline std::uint64_t nativeFixedHeaderBytes(
    ncclLlaminarCompletionAuthority original, std::uint32_t aborted, std::int32_t first_error)
{
    return original.abortFlag && original.deviceCommunicator && !aborted && !first_error ? sizeof(Header) : 0;
}

/** @return Exact required output bytes, never the allocated maximum. */
QFN_ADAPTIVE_HD inline std::uint64_t outputBytes(Geometry g, std::uint32_t original_ids)
{
    return static_cast<std::uint64_t>(original_ids) * g.row_elements * sizeof(float);
}

/**
 * @brief Publish the source's actual completed-read interval for one declared reader.
 * @note This follows the optimized Read and precedes result publication. It neither
 * advances an interval nor changes the coordinator/pin authority.
 */
QFN_ADAPTIVE_HD inline bool prepare(const p::LookupFrame& frame, Geometry g,
    std::uint32_t version, const int* first_error, Header& output)
{
    const bool accepted = valid(g) && version && forwardValid(first_error) &&
        frame.interval.table && frame.interval.admission && frame.interval.sequence &&
        frame.original_ids <= g.maximum_original && frame.original_ids % 16 == 0 &&
        frame.phase == p::LookupPhase::Ready && frame.status == Status::Success;
    output = {frame.interval, version, frame.original_ids, g.row_elements,
        g.coordinate, g.consumers, accepted ? Status::Success : Status::Cancelled};
    return accepted;
}

/**
 * @brief Validate an actually received result header and the reader's original replicated count.
 * @param received_bytes Actual acquired publication, zero after terminal cancellation.
 * @param original_ids Reader's own canonical query extent, never copied from the source.
 * @note A cancelled receive never parses the prior invocation's stale header.
 */
QFN_ADAPTIVE_HD inline bool accept(const Header& input, std::uint64_t received_bytes,
    Geometry g, std::uint32_t version, std::uint64_t source_table, std::uint32_t original_ids,
    const int* first_error, Header& reply)
{
    reply = {{}, version, original_ids, g.row_elements, g.coordinate, g.consumers,
        Status::Cancelled};
    if (received_bytes != sizeof(Header) || !valid(g) || !version ||
        !source_table || !forwardValid(first_error) || original_ids > g.maximum_original || original_ids % 16 != 0)
        return false;
    const bool accepted = input.version == version && input.interval.table == source_table &&
        input.interval.admission && input.interval.sequence &&
        input.original_ids == original_ids && input.row_elements == g.row_elements &&
        input.coordinate == g.coordinate && input.consumers == g.consumers &&
        input.status == Status::Success;
    reply = input;
    if (!accepted) reply.status = Status::Cancelled;
    return accepted;
}

/** @brief Authenticate ACK against the source's actual current frame. */
QFN_ADAPTIVE_HD inline bool confirm(const Header& input, std::uint64_t received_bytes,
    const p::LookupFrame& frame, Geometry g, std::uint32_t version,
    const int* first_error)
{
    return frame.phase == p::LookupPhase::Ready && received_bytes == sizeof(Header) && valid(g) && version &&
        forwardValid(first_error) && input.version == version &&
        input.interval.table == frame.interval.table &&
        input.interval.admission == frame.interval.admission &&
        input.interval.sequence == frame.interval.sequence &&
        input.original_ids == frame.original_ids && frame.original_ids <= g.maximum_original &&
        input.row_elements == g.row_elements && input.coordinate == g.coordinate &&
        input.consumers == g.consumers && input.status == Status::Success &&
        frame.status == Status::Success;
}

/**
 * @brief Publish reader ACK only after its exact required output has arrived.
 * @note An empty valid query requires zero payload, but still has an actual
 * authenticated result header/reply. Cancellation never turns a failed query into ACK.
 */
QFN_ADAPTIVE_HD inline bool acknowledge(Header& reply, std::uint64_t received_bytes,
    Geometry g, const int* first_error, CapturedTransferStatus payload_status)
{
    const bool accepted = payload_status == CapturedTransferStatus::Ready && valid(g) && forwardValid(first_error) &&
        reply.status == Status::Success && reply.original_ids <= g.maximum_original &&
        reply.row_elements == g.row_elements && reply.coordinate == g.coordinate &&
        reply.consumers == g.consumers &&
        received_bytes == outputBytes(g, reply.original_ids);
    if (!accepted) reply.status = Status::Cancelled;
    return accepted;
}
} // namespace llaminar2::qwen4exp::replicated_reader
