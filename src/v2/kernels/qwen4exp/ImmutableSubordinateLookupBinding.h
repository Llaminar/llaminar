/**
 * @file ImmutableSubordinateLookupBinding.h
 * @brief Service-issued original source/channel lifetime for adaptive GPU lookup.
 *
 * The existing host service alone creates this binding from its PMA-owned mapped
 * backing and actual source provider. Ordinary protocol1 bindings cannot be
 * reinterpreted as intent/ACK channels. No row locator or execution state is
 * mirrored here; native kernels consume only the exact explicit protocol2 ABI.
 */
#pragma once
#include "ImmutableLookupKernels.h"
#include "AdaptiveImmutableRowCacheNativePlan.h"
#include <memory>
#include <utility>
namespace llaminar2::qwen4exp
{
class ImmutableLookupService;
class AdaptiveLookupPreparedStorage;
class PooledImmutableSourceGeometry;
class PooledImmutableSourcePreparedStorage;
class PooledImmutableOwnerPreparedStorage;
class PooledImmutableCoordinatorPreparedStorage;
} // namespace llaminar2::qwen4exp
namespace llaminar2
{
class TensorBase;
class PhysicalMemoryAuthority;
struct ImmutableRowTableSource;
}
namespace llaminar2::qwen4exp
{
/** @brief Exact device aliases, produced only with the complete service lifetime. */
struct ImmutableSubordinateLookupDeviceView
{
    ImmutableLookupDeviceView base;
    ImmutableRowSubordinateIntent *intents;
    ImmutableRowInvalidationAcknowledgement *acknowledgements;
};
/** @brief Strong setup binding, not a public arbitrary-pointer proof factory. */
class ImmutableSubordinateLookupBinding final
{
public:
    /** @return Frozen device aliases; their backing remains retained by this value. */
    const ImmutableSubordinateLookupDeviceView &deviceView() const noexcept
    {
        return view_;
    }
    /** @return Exact GPU requester, never an inferred storage endpoint. */
    DeviceId device() const noexcept
    {
        return device_;
    }
    /** @brief Reject changed source/layout/allocator aliases without reading mutable GPU values. */
    void requireFor(const AdaptiveNativeParameters &parameters) const;
    /**
     * @brief Declare/reuse this source channel's one original prepared storage owner.
     * @param plan Exact checked source/consumer/policy/frame geometry.
     * @param cache Original optional native cache root, absent exactly at C0.
     * @param frame Original mandatory frame root.
     * @param protocol Original64B sequence/admission root.
     * @return Same strong setup owner on an identical declaration; no root can be replaced.
     * @throws std::exception On foreign/replaced roots, plan, failed source or capture-time declaration.
     * @note The service retains this owner for its whole lifetime. The owner does not retain the service.
     */
    std::shared_ptr<AdaptiveLookupPreparedStorage>
    declareStorage(const AdaptiveImmutableRowCacheNativePlan &plan, std::shared_ptr<TensorBase> cache,
                   std::shared_ptr<TensorBase> frame, std::shared_ptr<TensorBase> protocol) const;
    /**
     * @brief Require this source's original owner to be physically sealed before native codebook setup.
     * @param plan Exact factory geometry; it cannot be changed after preparation.
     * @throws std::exception For absent, unprepared, replaced or stale original storage.
     * @note This prevents a codebook bundle from preceding the runtime generation its roots authenticate.
     */
    void requirePreparedStorage(const AdaptiveImmutableRowCacheNativePlan &plan)const;
    /**
     * @brief Authenticate the original source-domain setup owner, not merely equal byte geometry.
     * @param storage Actual owner returned by this binding's declareStorage factory.
     * @throws std::invalid_argument Before native work for a missing or foreign service-issued owner.
     */
    void requireStorage(const AdaptiveLookupPreparedStorage &storage) const;
    /** @return Actual retained immutable source metadata; the binding retains the provider through every reader. */
    const ImmutableRowTableSource& source()const;
    /** @return Original service-instance source generation, distinct from request admission and I/O sequences. */
    std::uint64_t storageGeneration()const noexcept{return view_.base.table;}
    /**
     * @brief Authenticate pooled source declarations against this exact existing provider/channel.
     * @param geometry Original source/accepted-owner/coordinator geometry.
     * @throws std::exception On changed source, table/generation/format/rows, rank, channel ABI or aliases.
     * @note Geometry equality never mints permission; only this service-created binding can authorize the native source.
     */
    void requireFor(const PooledImmutableSourceGeometry& geometry)const;
    /**
     * @brief Create/reuse this actual service's original pooled source staging before capture.
     * @param geometry Original source-native exact packet geometry, already in the PMA BOM.
     * @param owner Actual retained accepted-grant authority, never decoded grant integers.
     * @param coordinator Actual retained policy/coalescing authority.
     * @param memory The exact original service/provider PMA.
     * @param stream Exact setup allocation stream; no request or live metadata is read.
     * @return Same source registration for an identical declaration; competing roots/arms fail.
     */
    std::shared_ptr<PooledImmutableSourcePreparedStorage> preparePooled(const PooledImmutableSourceGeometry& geometry,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)const;
    /** @return Complete source/channel declaration identity for the native capture signature. */
    std::uint64_t captureIdentity() const noexcept;

private:
    friend class ImmutableLookupService;
    /** @brief Service retains original backing/provider and exact mapped aliases. */
    ImmutableSubordinateLookupBinding(std::shared_ptr<ImmutableLookupService> owner, DeviceId device,
                                      ImmutableSubordinateLookupDeviceView view)
        : owner_(std::move(owner)), device_(device), view_(view)
    {
    }
    std::shared_ptr<ImmutableLookupService> owner_;
    DeviceId device_;
    ImmutableSubordinateLookupDeviceView view_;
};
} // namespace llaminar2::qwen4exp
