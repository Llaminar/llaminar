/**
 * @file CapturedVerifierExtent.cpp
 * @brief Authenticate and scope exact hosted verifier construction identities.
 *
 * All comparisons use immutable addresses and geometry. Device-owned count
 * contents remain invisible to the host and continue to guard every kernel.
 */
#include "CapturedVerifierExtent.h"
#include <exception>
#include <functional>
#include <stdexcept>

namespace llaminar2
{
CapturedVerifierExtent CapturedVerifierExtent::exactHostedRequest(
    DeviceId device, const DeviceRowRange &rows)
{
    const auto identity = rows.captureIdentity();
    if (!device.is_gpu() || !identity.count || identity.count_capacity < 2 ||
        identity.first_physical_row != 0 || identity.repeats_per_row != 1 ||
        identity.physical_rows != identity.count_capacity)
        throw std::invalid_argument("Exact hosted verifier requires original single-request GPU rows");
    CapturedVerifierExtent extent;
    extent.device_ = device;
    extent.count_ = identity.count;
    extent.rows_ = identity.physical_rows;
    return extent;
}

bool CapturedVerifierExtent::matches(DeviceId device, const DeviceRowRange &rows) const noexcept
{
    const auto identity = rows.captureIdentity();
    return isExact() && device_ == device && identity.count == count_ &&
        identity.count_capacity == rows_ && identity.physical_rows == rows_ &&
        identity.first_physical_row == 0 && identity.repeats_per_row == 1;
}

std::size_t CapturedVerifierExtent::captureHash() const noexcept
{
    std::size_t hash = std::hash<DeviceId>{}(device_);
    hash ^= std::hash<const void *>{}(count_) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(rows_) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

thread_local const ScopedCapturedVerifierExtent *ScopedCapturedVerifierExtent::active_ = nullptr;

ScopedCapturedVerifierExtent::ScopedCapturedVerifierExtent(CapturedVerifierExtent extent) noexcept
    : extent_(extent), previous_(active_)
{
    active_ = this;
}

ScopedCapturedVerifierExtent::~ScopedCapturedVerifierExtent()
{
    if (active_ != this) std::terminate();
    active_ = previous_;
}

CapturedVerifierExtent ScopedCapturedVerifierExtent::current() noexcept
{
    return active_ ? active_->extent_ : CapturedVerifierExtent{};
}
}
