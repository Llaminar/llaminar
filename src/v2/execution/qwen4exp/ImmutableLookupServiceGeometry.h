/**
 * @file ImmutableLookupServiceGeometry.h
 * @brief Immutable disjoint channels for one original source and retained I/O worker.
 *
 * Every actual consumer owns one protocol frame. Source/cache backing and the
 * worker are shared; no consumer aliases another consumer's request, response,
 * or terminal words. The same checked offsets price and construct the single
 * mapped allocation through TransferEngine and PhysicalMemoryAuthority.
 */
#pragma once
#include "ImmutableLookupProtocol.h"
#include "backends/DeviceId.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::qwen4exp {
/** @brief Checked source-owned protocol geometry; this is not a live allocation or ledger. */
class ImmutableLookupServiceGeometry final {
public:
    /**
     * @brief Seal one disjoint frame per original consumer in authored order.
     * @param layout Exact common source response envelope, including protocol identity.
     * @param consumers Actual unique GPU consumers; storage-only owners are not channels.
     * @return Overflow-checked immutable mapped storage geometry.
     * @throws std::invalid_argument For empty, duplicate or non-GPU consumers.
     * @throws std::overflow_error For an unrepresentable disjoint block.
     */
    static ImmutableLookupServiceGeometry checked(ImmutableLookupLayout layout,
        std::span<const DeviceId> consumers)
    {
        if (consumers.empty()) throw std::invalid_argument("Lookup service requires actual channel consumers");
        std::vector<DeviceId> devices;
        for (const auto device : consumers) {
            if (!device.is_gpu() || std::find(devices.begin(), devices.end(), device) != devices.end())
                throw std::invalid_argument("Lookup service channel consumer is not a unique GPU");
            devices.push_back(device);
        }
        if (layout.bytes() > std::numeric_limits<std::size_t>::max() - 63)
            throw std::overflow_error("Lookup channel alignment overflow");
        const auto stride = (layout.bytes() + 63) / 64 * 64;
        if (devices.size() > std::numeric_limits<std::size_t>::max() / stride)
            throw std::overflow_error("Lookup service channel block overflow");
        const auto bytes = stride * devices.size();
        return {layout, std::move(devices), stride, bytes};
    }
    /** @return Exact number of independent original request/response channels. */
    std::size_t channelCount() const noexcept { return devices_.size(); }
    /** @return Complete original channel membership without allocation or another authority. */
    std::span<const DeviceId> devices() const noexcept { return devices_; }
    /** @return Actual GPU consumer of the checked channel ordinal. */
    DeviceId device(std::size_t ordinal) const { return devices_.at(ordinal); }
    /** @return Exact common channel layout; runtime live counts never change capacity. */
    ImmutableLookupLayout channelLayout(std::size_t ordinal) const
    {
        (void)devices_.at(ordinal);
        return layout_;
    }
    /** @return Disjoint cache-line-aligned frame offset in the one mapped allocation. */
    std::size_t offset(std::size_t ordinal) const
    {
        (void)devices_.at(ordinal);
        return ordinal * stride_;
    }
    /** @return Requested mapped bytes before the canonical allocation's page rounding. */
    std::size_t bytes() const noexcept { return bytes_; }
    /** @return Exact canonical physical storage contribution for all concurrent channels together. */
    std::size_t hostBytes() const { return TransferEngine::mappedHostRegionAllocationBytes(bytes_); }
private:
    /** @brief Construct only after unique endpoint and overflow validation. */
    ImmutableLookupServiceGeometry(ImmutableLookupLayout layout, std::vector<DeviceId> devices,
        std::size_t stride, std::size_t bytes)
        : layout_(layout), devices_(std::move(devices)), stride_(stride), bytes_(bytes) {}
    ImmutableLookupLayout layout_;
    std::vector<DeviceId> devices_;
    std::size_t stride_, bytes_;
};
}
