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
#include <memory>
#include <utility>
namespace llaminar2::qwen4exp
{
class ImmutableLookupService;
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
     * @return Same source registration for an identical declaration; competing roots fail.
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
