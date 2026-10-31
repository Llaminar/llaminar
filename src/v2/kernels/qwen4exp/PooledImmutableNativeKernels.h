/**
 * @file PooledImmutableNativeKernels.h
 * @brief Backend-symmetric exact-stream phase submission over sealed persistent pooled roots.
 *
 * Public callers retain strong original roots and acquired producer operands.
 * Only checked enqueue reaches raw native arguments. No method creates a source
 * service, performs transport, reads a live count on the host, allocates during
 * capture or selects a serving mode.
 */
#pragma once
#include "PooledImmutableNativeTypes.h"
#include <memory>
namespace llaminar2::qwen4exp
{
class PooledImmutableOwnerPreparedStorage;
class PooledImmutableCoordinatorPreparedStorage;
class PooledImmutableOwnerOperands;
class PooledImmutableCoordinatorOperands;
/** @brief Stateless backend code bundle; persistent bytes remain with the one admitted storage owner. */
class IPooledImmutableNativeKernels
{
public:
    /** @brief Destroy only after retained native graphs and all stage readers have retired. */
    virtual ~IPooledImmutableNativeKernels()=default;
    /** @brief Initialize only virgin owner metadata once outside capture on the exact setup stream. */
    void initialize(PooledImmutableOwnerPreparedStorage& storage,void* exact_stream)const;
    /** @brief Initialize actual independent coordinator setup records once outside capture. */
    void initialize(PooledImmutableCoordinatorPreparedStorage& storage,void* exact_stream)const;
    /** @brief Enqueue one fixed participant-local owner phase; graph collectives remain explicit external nodes. */
    void enqueue(pooled_native::OwnerAction action,PooledImmutableOwnerPreparedStorage& storage,
        const PooledImmutableOwnerOperands& inputs,void* exact_stream)const;
    /** @brief Enqueue one complete coordinator phase, including the inseparable parallel Begin producer bundle. */
    void enqueue(pooled_native::CoordinatorAction action,PooledImmutableCoordinatorPreparedStorage& storage,
        const PooledImmutableCoordinatorOperands& inputs,void* exact_stream)const;
    /** @return Complete backend/launch version; stages also bind exact geometry, prepared roots and input producers. */
    virtual std::uint64_t codeIdentity()const noexcept=0;
    /** @return Actual selected native backend; factories do not query or activate hardware. */
    virtual DeviceType backend()const noexcept=0;
protected:
    /** @brief Receive only checked ordinary-prepared owner roots and exact setup stream. */
    virtual void initializeOwner(const pooled_native::OwnerArgs&,void*)const=0;
    /** @brief Copy only immutable owner geometry by exact setup launch scalars, then parallelize independent initialization. */
    virtual void initializeCoordinator(const pooled_native::CoordinatorSetupArgs&,
        std::span<const adaptive_lookup::pooled::CoordinatorOwnerGeometry>,void*)const=0;
    /** @brief Submit the fixed owner metadata/copy kernels without allocation, host decisions or hidden transport. */
    virtual void launch(pooled_native::OwnerAction,const pooled_native::OwnerArgs&,void*)const=0;
    /** @brief Submit a complete coordinator operation with device-owned live counts. */
    virtual void launch(pooled_native::CoordinatorAction,const pooled_native::CoordinatorArgs&,
        const pooled_native::CoordinatorQuery*,void*)const=0;
};
/** @return CUDA code bundle without runtime discovery, source binding or allocation. */
std::unique_ptr<IPooledImmutableNativeKernels> createCUDAPooledImmutableNativeKernels();
/** @return HIP code bundle with the same schemas/lifetime/arithmetic contracts. */
std::unique_ptr<IPooledImmutableNativeKernels> createROCmPooledImmutableNativeKernels();
}
