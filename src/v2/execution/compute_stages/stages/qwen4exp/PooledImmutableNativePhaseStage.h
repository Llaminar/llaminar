/**
 * @file PooledImmutableNativePhaseStage.h
 * @brief Reusable participant-local graph binding for one exact pooled-native phase.
 *
 * A model declares these phases and explicit collective/source boundaries. This
 * stage retains the original storage, code and producer roots, and performs no
 * rank coordination or source I/O. Every physical address is sealed before the
 * stage enters a graph, so its cache identity is complete even before capture.
 */
#pragma once
#include "execution/compute_stages/IComputeStage.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include "kernels/qwen4exp/PooledImmutableNativeOperands.h"
#include "memory/BufferArena.h"
#include <variant>
namespace llaminar2 {
/** @brief Bind exact native storage once; existing arena roles cannot be replaced or inferred. */
void bindPooledNativeRoots(BufferArena& arena,const qwen4exp::PooledImmutableOwnerPreparedStorage& storage);
/** @brief Bind the independent coordinator's original roots once under distinct canonical roles. */
void bindPooledNativeRoots(BufferArena& arena,const qwen4exp::PooledImmutableCoordinatorPreparedStorage& storage);
/** @brief One actual checked native operation with a declarative dependency/publication contract. */
class PooledImmutableNativePhaseStage final:public IComputeStage {
public:
    /** @brief Actual owner phase; no local consumer/controller/source capability is invented. */
    struct Owner {
        qwen4exp::pooled_native::OwnerAction action;
        std::shared_ptr<qwen4exp::PooledImmutableOwnerPreparedStorage> storage;
        qwen4exp::PooledImmutableOwnerOperands operands;
    };
    /** @brief Actual independent coordinator phase; only Begin carries a query descriptor. */
    struct Coordinator {
        qwen4exp::pooled_native::CoordinatorAction action;
        std::shared_ptr<qwen4exp::PooledImmutableCoordinatorPreparedStorage> storage;
        qwen4exp::PooledImmutableCoordinatorOperands operands;
    };
    using Phase=std::variant<Owner,Coordinator>;
    /**
     * @brief Retain the original bound arena and seal every already-prepared input before graph insertion.
     * @param arena Original arena that retains all canonical roles through graph retirement.
     * @param code Backend code only, with no device discovery or mutable execution state.
     * @param phase Actual typed owner/coordinator operation and its exact producer inputs.
     * @throws std::exception On absent roots, wrong roles, backend mismatch or unprepared input addresses.
     */
    PooledImmutableNativePhaseStage(BufferArena& arena,std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,Phase phase);
    /** @brief Join only the original setup event on the exact capture stream and revalidate immutable seals. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override;
    /** @brief Record the checked phase and publish original mutable roots on the executor's exact stream. */
    bool execute(IDeviceContext* context)override;
    /** @return Setup event joins occur once for each retained capture, never each replay. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Every original pointer/runtime/geometry/action/code/input and arena role in one complete identity. */
    std::uint64_t graphCaptureVariantSignature()const override;
    /** @return Stable semantic native-phase type; this does not advertise a serving policy. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    /** @return Only its exact explicit GPU backend can execute this stage. */
    bool supportsBackend(ComputeBackendType backend)const override;
    /** @return All roots were already declared/materialized; no hidden workspace is requested. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Exact once-bound root inouts and acquired original producer inputs. */
    StageBufferContract bufferContract()const override;
    /** @return Metadata only; live protocol values are never inspected for a dump. */
    StageDumpInfo buildDumpInfoImpl()const override;
private:
    /** @return Original native roots, never replacement aliases. */
    qwen4exp::PooledImmutableNativeRoots roots()const;
    /** @return Actual immutable producer spans for this nominal operation. */
    const qwen4exp::PooledNativeOperandSpans& inputs()const;
    /** @brief Reject any role/address/type/runtime/code/root drift before capture or launch. */
    void requireBindings()const;
    BufferArena* arena_;
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code_;
    Phase phase_;
    std::uint64_t identity_=0;
};
}
