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
#include "execution/qwen4exp/Qwen4ExpPrefillNextWindowBinding.h"
#include <optional>
namespace llaminar2::qwen4exp {
/** @brief Immutable prepared inputs; no factory reads a device count or creates execution authority. */
class PooledImmutableLocalHotOperands final {
public:
    /** @brief Bind original sixteen-head hashes/count plus canonical controller before Probe capture. */
    static PooledImmutableLocalHotOperands query(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,DeviceRowRange original_token_rows,PooledNativeTensorSpan hash_ids,
        std::optional<PooledNativeTensorSpan> original_count,DeviceGenerationControlBinding,PooledImmutableLocalHotAccessIntent);
    /**
     * @brief Bind the same original next-token window used by Hash, without a copied count bank.
     * @param storage Same bounded local frame/cache owner.
     * @param consumer Actual original consumer geometry.
     * @param window Borrow from the admitted current/request publication.
     * @param hash_ids Exact next-bucket sixteen-head output from that same window Hash.
     * @param controller Original request controller, not a second execution authority.
     * @throws std::invalid_argument For foreign participants or insufficient admitted capacity.
     */
    static PooledImmutableLocalHotOperands nextWindowQuery(const PooledImmutableLocalHotPreparedStorage& storage,
        local_hot_transport::ConsumerGeometry consumer,Qwen4ExpPrefillNextWindowBinding window,
        PooledNativeTensorSpan hash_ids,DeviceGenerationControlBinding controller);
    /** @return Whether this immutable Probe uses the explicit next-window count specialization. */
    bool hasNextWindow()const noexcept;
    /** @return Exact immutable Hash/Probe window identity, or zero for unchanged contiguous queries. */
    std::uint64_t nextWindowIdentity()const noexcept;
    /**
     * @brief Bind actual complete ordered demand and original request publication.
     * @param storage Same original single-frame local cache owner.
     * @param consumer Exact original consumer envelope and storage identity.
     * @param original_token_rows Genuine token count/range; each row contributes sixteen original IDs.
     * @param hash_ids Complete actual original ordered hash producer, retained without copying.
     * @param original_count Exact original count owner when the range is device-counted.
     * @param source_admission Original locally published once-per-request admission, never a result echo.
     * @param controller Same actual request-row controller used by the consumer graph.
     * @return Cold checked adoption schema; a frame-only operand cannot authorize this action.
     */
    static PooledImmutableLocalHotOperands adoption(const PooledImmutableLocalHotPreparedStorage&,
        local_hot_transport::ConsumerGeometry,DeviceRowRange original_token_rows,PooledNativeTensorSpan hash_ids,
        std::optional<PooledNativeTensorSpan> original_count,PooledNativeTransferSpan source_admission,
        DeviceGenerationControlBinding);
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
    /**
     * @return Immutable original query purpose, never its later consumed/adopted state.
     * @throws std::logic_error If these operands do not declare an actual Probe query.
     */
    PooledImmutableLocalHotAccessIntent queryIntent() const;
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
    /** @return Same original query roots plus its checked immutable borrowed window. */
    local_hot::NativeNextWindowQuery nativeNextWindowQuery(const PooledImmutableLocalHotPreparedStorage&) const;
    /** @return Actual checked demand IDs/count and exact original local request publication. */
    local_hot::NativeAdoption nativeAdoption(const PooledImmutableLocalHotPreparedStorage&) const;
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
