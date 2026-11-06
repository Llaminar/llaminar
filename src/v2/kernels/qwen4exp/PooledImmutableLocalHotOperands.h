/**
 * @file PooledImmutableLocalHotOperands.h
 * @brief Strong original hash/count/controller/output producers for checked local-cache phases.
 *
 * Host declaration/sealing is kept separate from GPU launch PODs. No factory
 * reads a live value or creates another source/storage admission authority.
 */
#pragma once
#include "PooledImmutableLocalHotKernels.h"
#include "PooledImmutableLocalHotPreparedStorage.h"
#include "PooledImmutableNativeOperands.h"
#include "execution/qwen4exp/PooledImmutableNativeTransfer.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include <optional>
namespace llaminar2::qwen4exp {
/** @brief Immutable prepared inputs; no factory reads a device count or creates execution authority. */
class PooledImmutableLocalHotOperands final {
public:
    /** @brief Bind original sixteen-head hashes/count plus canonical controller before Probe capture. */
    static PooledImmutableLocalHotOperands query(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,DeviceRowRange original_token_rows,PooledNativeTensorSpan hash_ids,
        std::optional<PooledNativeTensorSpan> original_count,DeviceGenerationControlBinding,PooledImmutableLocalHotAccessIntent);
    /** @brief Bind the sole model embedding output, with the SAME original native frame/controller. */
    static PooledImmutableLocalHotOperands read(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,PooledNativeTensorSpan output,DeviceGenerationControlBinding);
    /** @brief Bind the actual immutable own-request producer only for response acquisition. */
    static PooledImmutableLocalHotOperands result(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,PooledNativeTransferSpan own_request,DeviceGenerationControlBinding);
    /** @brief Bind the actual once-per-request local source publication for reset after retirement. */
    static PooledImmutableLocalHotOperands requestReset(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,PooledNativeTransferSpan source_admission,DeviceGenerationControlBinding);
    /** @brief Bind a fixed frame phase; actual graph builders retain the consumed/retired producer edges. */
    static PooledImmutableLocalHotOperands frame(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,DeviceGenerationControlBinding);
    /** @brief Seal the original already-prepared producers outside capture; allocate no model storage. */
    void seal();
    /** @return Strong original spans for ordinary declarative arena access contracts. */
    const std::vector<PooledNativeTensorSpan>& spans() const noexcept;
    /** @return Exact original request-row identity, not merely its whole controller tensor. */
    std::uint64_t controllerIdentity() const noexcept;
    /** @return Canonical controller owner whose original graph edge orders health/error publication. */
    TensorBase* controllerOwner() const noexcept;
    /** @return Immutable declarations usable before addresses are physically prepared. */
    std::uint64_t declarationIdentity() const;
    /** @brief Reject foreign roots, actions, row/count/output spans or controller before native enqueue. */
    void require(PooledImmutableLocalHotAction,const PooledImmutableLocalHotPreparedStorage&)const;
    /** @return Complete immutable launch inputs for graph identity; mutable contents are excluded. */
    std::uint64_t captureIdentity()const;
private:
    friend class IPooledImmutableLocalHotKernels;
    struct Data;
    /** @return Checked launch-only original frame/controller coordinates. */
    local_hot::NativeFrame nativeFrame(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @return Complete original hash/count coordinates after physical sealing. */
    local_hot::NativeQuery nativeQuery(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @return Sole original embedding output coordinates after physical sealing. */
    local_hot::NativeRead nativeRead(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @return Actual once-per-request original local admission after physical sealing. */
    local_hot::NativeRequestReset nativeRequestReset(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @return Actual original own-request producer after physical sealing. */
    local_hot::NativeResult nativeResult(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @brief Construct only after checking one complete native input schema. */
    explicit PooledImmutableLocalHotOperands(std::shared_ptr<Data> data):data_(std::move(data)){}
    std::shared_ptr<Data> data_;
};
} // namespace llaminar2::qwen4exp
