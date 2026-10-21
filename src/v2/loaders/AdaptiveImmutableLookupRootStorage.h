/**
 * @file AdaptiveImmutableLookupRootStorage.h
 * @brief Source-owned canonical GPU root allocations and their actual PMA leases.
 *
 * One setup factory materializes the already-admitted adaptive cache/frame and
 * original protocol roots. Aliasing root borrows retain this owner, so native
 * graphs cannot outlive their claims after the model or boundary releases its
 * reference. This owner stores no GPU cache state, source-row directory, free
 * capacity balance or separately computed live-byte counters.
 */
#pragma once
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheNativePlan.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/Tensors.h"
#include <memory>
#include <mutex>
namespace llaminar2
{
namespace qwen4exp { class ImmutableLookupExecutionBoundary;class AdaptiveLookupPreparedStorage; }
/** @brief Ordinary setup stream for the actual participant; no full-source upload receipt is needed. */
struct AdaptiveImmutableLookupDevicePreparation final
{
    DeviceId device;
    void *stream;
};
/** @brief Original strong tensor handles, never copied payload or replacement roots. */
struct AdaptiveImmutableLookupRoots final
{
    std::shared_ptr<TensorBase> cache; ///< Absent exactly when no GPU slots were admitted.
    std::shared_ptr<TensorBase> frame; ///< Mandatory bounded sort/response/control work.
    std::shared_ptr<TensorBase> protocol; ///< Original canonical INT32/64B admission root.
};
/** @brief Read-only point-in-time actual lease observation, not allocation permission or a storage pin. */
class AdaptiveImmutableLookupOwnerObservation final
{
public:
    /** @return Exact canonical authority that issued the still-retained claims. */
    const std::shared_ptr<const PhysicalMemoryAuthority> &authority() const noexcept { return authority_; }
    /** @return Exact original physical GPU allocator, independent of mutable cache contents. */
    PhysicalMemoryAllocatorIdentity allocator() const noexcept { return allocator_; }
    /** @return Actual optional ImmutableTableStorage lease, zero at C0. */
    std::size_t tableStorageBytes() const noexcept { return storage_; }
    /** @return Checked sum of actual frame and protocol ImmutableLookupStaging leases. */
    std::size_t lookupStagingBytes() const noexcept { return staging_; }
private:
    friend class AdaptiveImmutableLookupRootStorage;
    /** @brief Only the physical owner can author this observation from live leases. */
    AdaptiveImmutableLookupOwnerObservation(std::shared_ptr<const PhysicalMemoryAuthority> authority,
        PhysicalMemoryAllocatorIdentity allocator,std::size_t storage,std::size_t staging)
        :authority_(std::move(authority)),allocator_(allocator),storage_(storage),staging_(staging) {}
    std::shared_ptr<const PhysicalMemoryAuthority> authority_;
    PhysicalMemoryAllocatorIdentity allocator_;
    std::size_t storage_,staging_;
};
/** @brief One physical root bank shared only by serialized graph families. */
class AdaptiveImmutableLookupRootStorage final : public std::enable_shared_from_this<AdaptiveImmutableLookupRootStorage>
{
public:
    /**
     * @brief Claim exact admitted owners before ordinary setup materialization.
     * @param plan Checked original source/policy/native root geometry.
     * @param memory Sole canonical PMA for the exact local rank and GPU.
     * @param setup_stream Exact non-null ordinary setup stream; capture is forbidden.
     * @return One retained owner; C0 still has frame/protocol and no optional cache allocation.
     * @throws std::exception For foreign authority, invalid stream, exhausted admitted owners or allocation failure.
     * @note This selects no capacity and initializes no cache/protocol metadata.
     */
    static std::shared_ptr<const AdaptiveImmutableLookupRootStorage> create(
        const qwen4exp::AdaptiveImmutableRowCacheNativePlan &plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void *setup_stream);
    /** @brief Retire actual tensor allocations before their sole live PMA claims. */
    ~AdaptiveImmutableLookupRootStorage() = default;
    /**
     * @return Original tensor aliases, each strongly retaining this actual lease owner.
     * @note No aliases are retained internally; this creates no reference cycle.
     */
    AdaptiveImmutableLookupRoots roots() const;
    /**
     * @brief Reject a substituted authority, allocator, extent or root before native seal/reuse.
     * @param plan Checked complete source-owned root geometry.
     * @param memory Exact original canonical authority, never an equivalent new ledger.
     * @throws std::logic_error For missing/live-claim mismatch or foreign source geometry.
     */
    void requireFor(const qwen4exp::AdaptiveImmutableRowCacheNativePlan &plan,
        const std::shared_ptr<const PhysicalMemoryAuthority> &memory) const;
    /** @return Actual live claims; borrowers must separately retain this owner or its aliases. */
    AdaptiveImmutableLookupOwnerObservation retainedOwnerObservation() const;
private:
    friend class qwen4exp::ImmutableLookupExecutionBoundary;
    /** @brief Bind exactly one actual source-issued prepared owner before native initialization; never reset after retirement. */
    void bindPreparedStorage(const std::shared_ptr<qwen4exp::AdaptiveLookupPreparedStorage> &storage) const;
    /** @brief Only create() constructs the exact setup ownership record. */
    AdaptiveImmutableLookupRootStorage(qwen4exp::AdaptiveImmutableRowCacheNativePlan plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory)
        :plan_(std::move(plan)),memory_(std::move(memory)),device_(plan_.parameters().device) {}
    qwen4exp::AdaptiveImmutableRowCacheNativePlan plan_; ///< Complete immutable declaration sealed to these physical roots.
    mutable std::mutex setup_mutex_; ///< Serializes physical source-owner association, not device state.
    mutable std::weak_ptr<qwen4exp::AdaptiveLookupPreparedStorage> prepared_;
    mutable bool ever_bound_=false; ///< Retired source ownership can never initialize the same physical bytes again.
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    DeviceId device_;
    // Reverse destruction frees native allocations before releasing the claims.
    PhysicalMemoryAllocationLease cache_lease_,frame_lease_,protocol_lease_;
    std::unique_ptr<TensorBase> cache_,frame_,protocol_;
};
} // namespace llaminar2
