/**
 * @file PooledImmutableReplicatedReaderGraph.h
 * @brief Explicit compact result and all-reader ACK edges for one shared PLE lookup.
 *
 * Existing per-device graphs retain their own model execution. Only the source
 * graph owns/coalesces the replicated query and launches storage-only owners.
 * The reader graph owns no source service or physical pin authority.
 */
#pragma once
#include "PooledImmutableDirectedTransport.h"
#include "PooledImmutableLocalLookupGraph.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"

namespace llaminar2 {
/** @brief Actual immutable source result/ACK stage inventory, not a retirement capability. */
struct PooledImmutableReplicatedReaderCompletion final {
    struct Reader {
        std::shared_ptr<const qwen4exp::PooledImmutableReplicatedReaderTransport> original;
        std::string header, headerSent, extent, payloadSent, ackReceived, ackValidated;
    };
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport;
    DeviceGenerationControlBinding control;
    DeviceRowRange rows;
    qwen4exp::PooledNativeTensorSpan output;
    std::optional<qwen4exp::PooledNativeTensorSpan> count;
    std::vector<Reader> readers;
    std::string sourceRead;
};
/** @brief Original foreign result/ACK node inventory, never native retirement authority. */
struct PooledImmutableReplicatedConsumerCompletion final {
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport;
    std::shared_ptr<const qwen4exp::PooledImmutableReplicatedReaderTransport> reader;
    DeviceGenerationControlBinding control;
    DeviceRowRange rows;
    qwen4exp::PooledNativeTensorSpan output;
    std::optional<qwen4exp::PooledNativeTensorSpan> count;
    std::string header, accepted, payload, acknowledged, sent;
};
/** @return Exact typed ACK inventory in this actual cold recipient graph; duplicate or foreign definitions reject. */
PooledImmutableReplicatedConsumerCompletion findPooledReplicatedConsumerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport>& transport, DeviceId device);
/** @brief Authenticate the exact original foreign reader phases, receive receipt and ACK dependency chain. */
void requirePooledReplicatedConsumerCompletion(const ComputeGraph& graph,
    const PooledImmutableReplicatedConsumerCompletion& completion);
/**
 * @brief Append source result sends and exact ACK validators after the original optimized Read.
 * @return Complete original node inventory; every ACK gates existing ReadersRetired.
 */
std::shared_ptr<const PooledImmutableReplicatedReaderCompletion> appendPooledReplicatedSourceReaders(
    ComputeGraph& graph, BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport,
    DeviceGenerationControlBinding control, DeviceRowRange originalRows,
    const qwen4exp::PooledNativeTensorSpan& output, const std::string& readOutput,
    const std::string& readersRetired, const std::string& prefix,
    std::optional<qwen4exp::PooledNativeTensorSpan> count = {});
/** @brief Append the original foreign reader's receive/validate/ACK program, without a CPU source boundary. */
PooledImmutableLookupGraphEndpoints appendPooledReplicatedReader(
    ComputeGraph& graph, BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableDirectedTransport> transport,
    DeviceGenerationControlBinding control, DeviceRowRange originalRows,
    const qwen4exp::PooledNativeTensorSpan& output, const std::string& prefix,
    const std::string& outputIdentity,
    std::optional<qwen4exp::PooledNativeTensorSpan> count = {},
    PooledImmutableReplicatedConsumerCompletion* completion = nullptr);
/** @brief Authenticate the exact original validators and every ACK→ReadersRetired dependency. */
void requirePooledReplicatedReaderCompletion(const ComputeGraph& graph,
    const PooledImmutableReplicatedReaderCompletion& completion, const std::string& readersRetired);
} // namespace llaminar2
