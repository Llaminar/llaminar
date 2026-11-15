/**
 * @file CapturedVerifierExtent.h
 * @brief Immutable hosted verifier width admitted by a retained transaction.
 *
 * A hosted transaction names one exact request/depth graph. Its original device
 * count remains the arithmetic authority, including empty/error publications.
 * This fact only selects a kernel at construction; it never reads that count,
 * removes a row guard, or asserts that a generic physical bucket is full.
 */
#pragma once

#include "backends/DeviceId.h"
#include "kernels/common/DeviceRowRange.h"
#include <cstddef>

namespace llaminar2
{
/** @brief Capture identity of one exact hosted request, or ordinary guarded capacity. */
class CapturedVerifierExtent final
{
public:
    /** @brief Ordinary capacity carries no exact-width scheduling fact. */
    CapturedVerifierExtent() = default;

    /**
     * @brief Bind a width from an admitted exact hosted transaction family.
     * @param device Original participant owning the complete transaction.
     * @param rows Original, unsliced, single-request device count and width.
     * @throws std::invalid_argument For CPU, fixed, sliced, repeated or M1 rows.
     */
    [[nodiscard]] static CapturedVerifierExtent exactHostedRequest(
        DeviceId device, const DeviceRowRange &rows);

    /** @return Whether admission supplied an exact hosted transaction identity. */
    [[nodiscard]] bool isExact() const noexcept { return count_ != nullptr; }
    /** @return Whether this exact fact belongs to the named native participant. */
    [[nodiscard]] bool belongsTo(DeviceId device) const noexcept { return isExact() && device_ == device; }
    /** @return True only for the original participant and complete count binding. */
    [[nodiscard]] bool matches(DeviceId device, const DeviceRowRange &rows) const noexcept;
    /** @return Hash of immutable policy, participant, pointer and physical width. */
    [[nodiscard]] std::size_t captureHash() const noexcept;
    /** @return Whether every immutable authority and geometry field agrees. */
    bool operator==(const CapturedVerifierExtent &) const = default;

private:
    DeviceId device_ = DeviceId::invalid(); ///< Original device; never inferred from active TLS.
    const std::int32_t *count_ = nullptr; ///< Borrowed original device count, never dereferenced here.
    int rows_ = 0; ///< Exact retained transaction width, independent of count contents.
};

/**
 * @brief Install construction policy on the original GPU capture worker.
 *
 * Every capture installs a scope, including guarded capacity, so unrelated
 * nested graphs cannot inherit a parent's admission. Scope state contains only
 * immutable identity and is never queried during native graph replay.
 */
class ScopedCapturedVerifierExtent final
{
public:
    /** @param extent Immutable identity authenticated by the graph cache. */
    explicit ScopedCapturedVerifierExtent(CapturedVerifierExtent extent) noexcept;
    /** @brief Restore the enclosing worker scope, enforcing exact LIFO ownership. */
    ~ScopedCapturedVerifierExtent();
    ScopedCapturedVerifierExtent(const ScopedCapturedVerifierExtent &) = delete;
    ScopedCapturedVerifierExtent &operator=(const ScopedCapturedVerifierExtent &) = delete;
    /** @return This worker's construction policy; absent scopes mean guarded capacity. */
    [[nodiscard]] static CapturedVerifierExtent current() noexcept;

private:
    CapturedVerifierExtent extent_; ///< Scope-owned immutable value, no borrowed host lifecycle.
    const ScopedCapturedVerifierExtent *previous_; ///< Enclosing scope on this exact worker.
    static thread_local const ScopedCapturedVerifierExtent *active_; ///< Construction-only owner.
};
}
