/**
 * @file PooledImmutableSourcePhaseStage.h
 * @brief Reusable graph phase for the actual retained RAM/SSD pooled source boundary.
 *
 * Model graphs declare native source phases and explicit owner/coordinator
 * dependencies. This stage keeps actual provider/channel/roots alive through
 * graph terminal and final reader retirement, and never manufactures a source
 * result or substitutes test delivery. Live extents stay entirely native.
 */
#pragma once
#include "execution/compute_stages/IComputeStage.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "kernels/qwen4exp/PooledImmutableNativeOperands.h"
namespace llaminar2 {
/** @brief Bind only original source-root aliases; the arena acquires no second physical claim. */
void bindPooledSourceRoots(BufferArena& arena,const qwen4exp::ImmutableLookupExecutionBoundary& boundary);
/** @brief One fixed captured real-provider operation, with canonical control and explicit original output. */
class PooledImmutableSourcePhaseStage final:public IComputeStage {
public:
    /**
     * @brief Retain the existing source boundary and seal every embedded output/control address before graph insertion.
     * @param arena Exact participant arena, retaining all original source/owner/coordinator/controller roles.
     * @param boundary Original mutually exclusive pooled service arm; no alternate provider is accepted.
     * @param action Fixed source producer/reader operation.
     * @param control Existing canonical request row, independent of owner/coordinator membership.
     * @param output Exact original FP32 output span for Read only; other operations reject an output.
     */
    PooledImmutableSourcePhaseStage(BufferArena& arena,std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
        qwen4exp::PooledSourceAction action,DeviceGenerationControlBinding control,
        std::optional<qwen4exp::PooledNativeTensorSpan> output=std::nullopt);
    /** @brief Revalidate actual roots/runtime and join only exact original setup events on the supplied stream. */
    bool prepareGraphLaunch(IDeviceContext*,void*)override;
    /** @brief Enqueue one fixed source phase on the executor's exact stream and publish its original writers. */
    bool execute(IDeviceContext*)override;
    /** @return Setup event joins occur once for retained recording; request reset never changes topology. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Every service/root/address/runtime/action/controller/output/code configuration in one immutable identity. */
    std::uint64_t graphCaptureVariantSignature()const override;
    /** @return Semantic real-provider stage kind, with no serving/topology support advertisement. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_SOURCE_PHASE;}
    /** @return Only the declared actual GPU backend can execute this phase. */
    bool supportsBackend(ComputeBackendType)const override;
    /** @return All storage was already declared/materialized under its original PMA owner. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Original authority inputs, source inouts, canonical controller and actual output. */
    StageBufferContract bufferContract()const override;
    /** @return Static source/action metadata only; no live protocol fields are read. */
    StageDumpInfo buildDumpInfoImpl()const override;
private:
    /** @brief Reject source/role/root/controller/runtime drift before any enqueue. */
    void requireBindings()const;
    BufferArena* arena_;
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary_;
    qwen4exp::PooledSourceAction action_;
    DeviceGenerationControlBinding control_;
    std::optional<qwen4exp::PooledNativeTensorSpan> output_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::shared_ptr<const CapturedDeviceStorageLease> captured_; ///< Retires before the original strong output Tensor owner.
    std::uint64_t identity_=0;
};
}
