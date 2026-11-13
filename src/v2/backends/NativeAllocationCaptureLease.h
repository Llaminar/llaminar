/**
 * @file NativeAllocationCaptureLease.h
 * @brief Borrow an existing backend allocation record without another allocation ledger.
 *
 * The allocator retains one immutable identity object per real allocation.
 * Captured consumers retain that same object. Allocation/free and borrow
 * acquisition use the backend lifecycle lock; free rejects live borrowers
 * before any native API. The token owns neither device bytes nor a PMA claim.
 */
#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
namespace llaminar2 {
/** @brief Unique shared identity belonging to one canonical live allocator record. */
struct NativeAllocationCaptureIdentity final {};
/** @brief Immutable bounded borrow; copies retain the same physical-record identity. */
class NativeAllocationCaptureLease final {
public:
    /** @brief Validate and borrow an actual live record under its allocator lock.
     * @param identity Original record token, never a pointer-derived identity.
     * @param base Original allocation base.
     * @param bytes Complete allocation extent.
     * @param owner Exact allocating GPU.
     * @param span First consumed byte, possibly an interior view.
     * @param extent Positive consumed span.
     * @param device Exact consumer GPU.
     * @return A bounded reference to that record.
     * @throws std::invalid_argument For missing, foreign or out-of-range storage.
     */
    static NativeAllocationCaptureLease acquire(
        const std::shared_ptr<const NativeAllocationCaptureIdentity>& identity,
        const void* base, size_t bytes, DeviceId owner,
        const void* span, size_t extent, DeviceId device) {
        const auto first=reinterpret_cast<std::uintptr_t>(base);
        const auto start=reinterpret_cast<std::uintptr_t>(span);
        if(!identity||!base||!span||!extent||!owner.is_gpu()||owner!=device||
           start<first||start-first>bytes||extent>bytes-(start-first))
            throw std::invalid_argument("Captured storage span has no exact native allocation owner");
        return NativeAllocationCaptureLease(identity,span,extent,device);
    }
    /** @brief Reject reclamation before the native free; caller holds its lifecycle lock.
     * @param identity Sole live record identity, plus any captured borrowers.
     * @throws std::logic_error When any capture still borrows this allocation.
     */
    static void requireUnborrowed(const std::shared_ptr<const NativeAllocationCaptureIdentity>& identity) {
        if(!identity||identity.use_count()!=1)
            throw std::logic_error("Native allocation is still borrowed by retained capture storage");
    }
    /** @return Whether another observation names the same allocation and exact consumed span. */
    bool sameStorage(const NativeAllocationCaptureLease& other) const noexcept {
        return identity_==other.identity_&&span_==other.span_&&extent_==other.extent_&&device_==other.device_;
    }
private:
    /** @brief Publish only after exact record/bounds admission above.
     * @param identity Original physical record token.
     * @param span First retained consumer byte.
     * @param extent Exact consumer extent.
     * @param device Exact GPU participant.
     */
    NativeAllocationCaptureLease(std::shared_ptr<const NativeAllocationCaptureIdentity> identity,
        const void* span,size_t extent,DeviceId device)
        :identity_(std::move(identity)),span_(span),extent_(extent),device_(device){}
    std::shared_ptr<const NativeAllocationCaptureIdentity> identity_;
    const void* span_;
    size_t extent_;
    DeviceId device_;
};
}
