/**
 * @file PooledImmutableLocalHotPhaseStage.h
 * @brief Checked local native phases with one ordinary model-output producer.
 *
 * Private cache/frame visibility belongs to the original L1 ready/pin protocol,
 * not ordinary tensor publication. Only real hash/count/controller and LocalRead
 * output participate in the graph's ordinary arena dependency ledger.
 */
#pragma once
#include "execution/compute_stages/IComputeStage.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotOperands.h"
#include "execution/qwen4exp/PooledImmutableLocalHotPoolQuery.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
namespace llaminar2 {
/** @brief Typed private compact query into the unchanged original coordinator Begin bundle. */
class PooledImmutableLocalHotBeginStage final:public IComputeStage {
public:
    /** @brief Bind the original complete source/coordinator and closed actual query producers. */
    PooledImmutableLocalHotBeginStage(BufferArena&,std::shared_ptr<qwen4exp::PooledImmutableSourcePreparedStorage>,
        std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels>,qwen4exp::PooledImmutableLocalHotPoolQuery);
    /** @return Actual source identity for cold graph/retirement validation. */
    const auto& source()const noexcept{return source_;}
    /** @return Closed actual producer declaration, not a downloaded count. */
    const auto& query()const noexcept{return query_;}
    bool prepareGraphLaunch(IDeviceContext*,void*)override;
    bool execute(IDeviceContext*)override;
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    std::uint64_t graphCaptureVariantSignature()const override;
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType)const override;
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override;
    StageDumpInfo buildDumpInfoImpl()const override;
private:
    /** @brief Authenticate unchanged original arena roots and backend code. */
    void requireDeclarations()const;
    BufferArena* arena_;
    std::shared_ptr<qwen4exp::PooledImmutableSourcePreparedStorage> source_;
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code_;
    qwen4exp::PooledImmutableLocalHotPoolQuery query_;
    std::uint64_t identity_=0;
};
/** @brief One exact native local operation, retaining original storage and code through parent retirement. */
class PooledImmutableLocalHotPhaseStage final:public IComputeStage {
public:
    /** @brief Bind actual local roots and original producer/output declarations before capture. */
    PooledImmutableLocalHotPhaseStage(BufferArena&,std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>,
        std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels>,qwen4exp::PooledImmutableLocalHotAction,
        qwen4exp::PooledImmutableLocalHotOperands);
    /** @return Actual native phase, not a readiness or source capability. */
    qwen4exp::PooledImmutableLocalHotAction action()const noexcept{return action_;}
    /** @return Actual selected stateless code image, never factory object address equality. */
    std::uint64_t codeIdentity()const noexcept{return code_->codeIdentity();}
    /** @return Exact original producer declarations for cold topology authentication. */
    const auto& operands()const noexcept{return operands_;}
    /** @return Same original physical local owner used by typed graph validation. */
    const auto& storage()const noexcept{return storage_;}
    /** @brief Seal exact producers and join original local setup on the actual capture stream. */
    bool prepareGraphLaunch(IDeviceContext*,void*)override;
    /** @brief Enqueue the actual native phase; only LocalRead publishes ordinary model output. */
    bool execute(IDeviceContext*)override;
    /** @return Original declarations are sealed once per cold capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Complete immutable phase/code/storage/producer identity. */
    std::uint64_t graphCaptureVariantSignature()const override;
    /** @return Participant-local native operation classification. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    /** @return Actual arbitrary counted/empty row contract. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType)const override;
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Sole output writer or exact input/controller access, excluding private protocol roots. */
    StageBufferContract bufferContract()const override;
    StageDumpInfo buildDumpInfoImpl()const override;
private:
    /** @brief Reject changed arena/source/native owner or original producer roles. */
    void requireDeclarations()const;
    BufferArena* arena_;
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> storage_;
    std::shared_ptr<const qwen4exp::IPooledImmutableLocalHotKernels> code_;
    qwen4exp::PooledImmutableLocalHotAction action_;
    qwen4exp::PooledImmutableLocalHotOperands operands_;
    std::uint64_t identity_=0;
    std::optional<std::uint64_t> sealed_operands_; ///< Original pointer/runtime seal, never mutable device state.
};
/** @brief Original mapped GEQ timeline gates; they do not join an owner response or auxiliary terminal. */
class PooledImmutableLocalHotReadyStage final:public IComputeStage {
public:
    enum class Frontier {DataReady,LocalReadComplete};
    /** @brief Retain the actual original local ready region and nominal frontier. */
    PooledImmutableLocalHotReadyStage(std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>,Frontier);
    const auto& storage()const noexcept{return storage_;}
    Frontier frontier()const noexcept{return frontier_;}
    bool prepareGraphLaunch(IDeviceContext*,void*)override;
    bool execute(IDeviceContext*)override;
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    std::uint64_t graphCaptureVariantSignature()const override;
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType)const override;
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override{return StageBufferContract::build();}
    StageDumpInfo buildDumpInfoImpl()const override;
private:
    std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage> storage_;
    Frontier frontier_;
    std::uint64_t identity_;
};
} // namespace llaminar2
