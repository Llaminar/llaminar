/**
 * @file Qwen4ExpPLERetainedStage.h
 * @brief Participant-local PLE immutable transaction and verifier-publication stages.
 *
 * Graph wiring must add SnapshotPrior before the FIRST verifier PLE history reader
 * or mutator, then make every verifier consumer depend on that snapshot node.
 * RetainCandidates depends on snapshot plus token and normalized-key producers.
 * Dedicated prior banks have snapshot as their sole writer and remain immutable
 * through publication/republication. Existing controller/DAG retirement orders the
 * next snapshot after every reader; accepted and ordinary next banks stay distinct.
 * A byte offset denotes a real arena subview; TensorBase row views are not assumed.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../qwen4exp/ImmutableVerifierPrior.h"
#include "../../../qwen4exp/Qwen4ExpPreparedStorage.h"
#include "../../../../kernels/qwen4exp/Qwen4ExpPLERetainedDispatch.h"
#include <array>
namespace llaminar2 {
/** @brief Explicit snapshot/retention stages joining the existing verifier publisher. */
class Qwen4ExpPLERetainedStage final:public IComputeStage,public ImmutableVerifierPriorProducer {
public:
 /** @brief Distinguish the pre-mutation edge from post-projection retention. */
 enum class Operation { SnapshotPrior,RetainCandidates };
 /** @brief Immutable captured topology and mandatory arena ownership for one request. */
 struct Params {
  STAGE_PARAMS_COMMON_FIELDS;
  Operation operation; int requests,requestIndex,rowCapacity,channels;
  std::optional<Qwen4ExpIntInput> liveCounts; ///< Canonical lengths; absent explicitly selects fully active requests.
  Qwen4ExpIntBinding transactionLiveCounts; ///< Immutable transaction lengths, written only by snapshot.
  Qwen4ExpIntBinding acceptedTokens,priorTokens,retainedTokens;
  Qwen4ExpIntInput tokens; ///< Arena or explicit execution-owned native token producer, without a copy.
  Qwen4ExpFloatBinding acceptedConvolution,priorConvolution,normalized,retainedNormalized;
  Qwen4ExpIntBinding copiedBytes; ///< Required aligned64 payload observation for this operation.
  RequestRowOrigins verifierOrigins; ///< Exact compact CPU or padded GPU first-row coordinates.
 };
 static_assert(StageParamsRequired<Params>);
 /** @brief Validate all types/capacities/aliases and select native submission at setup. */
 explicit Qwen4ExpPLERetainedStage(Params params);
 /** @brief Submit only declared prepared operands on the executor's exact stream. */
 bool execute(IDeviceContext *context) override;
 /** @brief Return the distinct snapshot or retained-publication semantic tag. */
 ComputeStageType type() const override;
 /** @brief No arithmetic is performed: every payload bit is copied exactly. */
 std::size_t estimatedFlops() const override{return 0;}
 /** @brief Return admitted operand byte extent without claiming it is live transfer volume. */
 std::size_t estimatedMemoryBytes() const override;
 /** @brief Report only this stage's sealed participant backend. */
 bool supportsBackend(ComputeBackendType backend) const override;
 /** @brief All persistent storage is explicit caller-admitted arena storage. */
 StageBufferRequirements getBufferRequirements() const override{return {};}
 /** @brief Contract includes accepted banks written later by the verifier publication hook. */
 StageBufferContract bufferContract() const override;
 /** @brief Publish exact byte subviews and transaction geometry without observing live state. */
 StageDumpInfo buildDumpInfoImpl() const override;
 /** @brief Both operations obey execution-owned live request prefixes. */
 bool supportsPaddedPrefillRealLengthContract() const override{return true;}
 /** @brief Only the retention stage participates in accepted-state publication. */
 bool hasVerifierStateCapture() const override{return params_.operation==Operation::RetainCandidates;}
 /** @brief Every retained PLE transaction needs explicit accepted-state publication. */
 bool requiresVerifierStateCaptureForPublication() const override{return hasVerifierStateCapture();}
 /** @brief Plan exact CPU history destinations without mutating accepted state. */
 CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row) override;
 /** @brief Validate the whole decision vector before appending this request's plan. */
 bool planCPUVerifierStateRestoreRows(const int *rows,int requests,std::vector<CPUVerifierStateRestorePlan>&plans) override;
 /** @brief Publish an explicit host-owned single-request decision. */
 bool restoreVerifierStateCaptureRow(int row,void *stream) override;
 /** @brief Publish this request after the publisher has validated all request plans. */
 bool restoreVerifierStateCaptureRows(const int *rows,int requests,void *stream) override;
 /** @brief Consume one device decision only for a single-request transaction. */
 bool restoreVerifierStateCaptureRowFromDeviceIndex(const int *row,void *stream) override;
 /** @brief Accepted PLE publication writes only token and convolution history banks. */
 std::span<const BufferBinding> verifierStatePublicationOutputs() const noexcept override;

 /** @brief Consume the complete independently-strided device decision vector. */
 bool restoreVerifierStateCaptureRowsFromDeviceIndices(const int *rows,int requests,int stride,void *stream) override;
 /** @brief Seal normally prepared storage once before native graph recording. */
 bool prepareGraphLaunch(IDeviceContext*context,void*stream)override;
 /** @brief Storage seals run only during initial capture preparation, never native replay. */
 GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
 /** @brief Stable declarations plus validated physical seals; excludes mutable contents. */
 std::uint64_t graphCaptureVariantSignature()const override;
 /** @brief Sealed immutable pointer/geometry policy, including borrowed native token identity. */
 const Params &getParams()const noexcept{return params_;}
private:
 /** @brief Constructor-owned native operand metadata shared by all stage contracts. */
 struct Operand{TensorBase*tensor;BufferId buffer;TensorType dtype;std::size_t offset,bytes;bool write;const char*name;};
 /** @brief Register one checked arena-native byte interval. */
 template<TensorType Type>void add(Qwen4ExpArenaBinding<Type> binding,std::size_t bytes,bool write,const char*name);
 /** @brief Resolve prepared CPU/GPU bytes without coherence repair or allocation. */
 template<class T,TensorType Type>T*pointer(Qwen4ExpArenaBinding<Type> binding)const;
 /** @brief Resolve an arena/native token input already ordered by its explicit producer edge. */
 const std::int32_t *tokenPointer()const;
 /** @brief Resolve borrowed arena/native INT32 metadata without transfer or readback. */
 const std::int32_t *intInput(const Qwen4ExpIntInput&binding)const;
 /** @brief Borrow the immutable complete transaction request geometry. */
 DeviceRequestRowRanges transactionRows()const;
 /** @brief Resolve complete native publication arguments without observing GPU values. */
 qwen4exp::PLEPublishAccepted publication(const int*rows,int stride)const;
 /** @brief Execute exact-stream validated publication and publish its output events. */
 bool publish(const int*rows,int requests,int stride,void*stream);
 std::array<BufferBinding, 2> acceptedPublicationOutputs_{}; ///< Immutable accepted-history declaration.
 Params params_;qwen4exp::PLETransactionOperations operations_;
 std::vector<Operand>operands_;
 std::optional<Qwen4ExpPreparedStorage>storage_;
 std::array<CPUVerifierStateRestoreDestination,2>writes_{}; ///< Fixed setup-owned CPU plan storage.
};
}
