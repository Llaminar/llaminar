/**
 * @file PipelineTransferMemory.cpp
 * @brief Canonical channel BOM geometry, delegating allocation rounding to transport.
 *
 * Replayed graph families share these two non-overlapping directional slots.
 * Neither TP width, token count nor MTP depth variants multiply their physical
 * storage. The caller contributes this immutable BOM to PhysicalMemoryAuthority
 * before TransferEngine can materialize it.
 */
#include "PipelineTransferMemory.h"
#include "transfer/CapturedTransferChannel.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
PipelineTransferMemory PipelineTransferMemory::forRows(std::size_t width,
    std::size_t prefill_rows, std::size_t verifier_rows)
{
    if (!width || !prefill_rows || !verifier_rows)
        throw std::invalid_argument("Pipeline transfer memory requires positive physical dimensions");
    const auto multiply = [](size_t left, size_t right) {
        if (left > std::numeric_limits<size_t>::max() / right)
            throw std::overflow_error("Pipeline transfer capacity overflow");
        return left * right;
    };
    const auto add = [](size_t left, size_t right) {
        if (left > std::numeric_limits<size_t>::max() - right)
            throw std::overflow_error("Pipeline transfer allocation sum overflow");
        return left + right;
    };
    const auto activation_bytes = multiply(multiply(width, std::max(prefill_rows, verifier_rows)), sizeof(float));
    const auto metadata_bytes = multiply(verifier_rows, sizeof(int32_t));
    const auto activation = CapturedTransferChannel::memoryFor(activation_bytes);
    const auto metadata = CapturedTransferChannel::memoryFor(metadata_bytes);
    return {activation_bytes, metadata_bytes,
        add(activation.cursor_bytes_per_device, metadata.cursor_bytes_per_device),
        add(activation.mapped_host_bytes, metadata.mapped_host_bytes)};
}
}
