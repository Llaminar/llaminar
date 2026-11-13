/**
 * @file Qwen4ExpFixedHeadLookupStage.h
 * @brief Participant-local original-native fixed-head operations between explicit collectives.
 *
 * These four operations share one immutable decoder bundle and retain the exact
 * admitted source range. They own no communicator, placement decision, host
 * service, live count, or payload allocation. The reusable PLE builder declares
 * the two actual collective nodes; no compute stage hides cross-device work.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include "loaders/ResidentImmutableRowTable.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
namespace llaminar2 {
/** @brief Distinct admitted fixed-shard storage; cannot be passed as a whole-table view. */
struct Qwen4ExpFixedHeadLookupStorage {
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort;
    Qwen4ExpReplicatedPLEQuery query;
    std::shared_ptr<const ResidentImmutableRowTable> table;
    DeviceGenerationControlBinding control;
    Qwen4ExpIntBinding localHeader,gatheredHeaders,transportRows,localNative,gatheredNative;
};
/** @brief One local operation in the explicit validation/exchange/pack/exchange/decode DAG. */
class Qwen4ExpFixedHeadLookupStage final : public IComputeStage {
public:
    /** @brief Immutable arithmetic operation; collective nodes are separate graph stages. */
    enum class Operation { ValidateLocal, ValidateCohort, Pack, Decode };
    /** @brief The same prepared kernel bundle is retained by all four local nodes. */
    struct Frame {
        Qwen4ExpFixedHeadLookupStorage storage;
        Qwen4ExpIntBinding ids;
        Qwen4ExpFloatBinding output;
        std::shared_ptr<const qwen4exp::IImmutableFixedHeadLookupKernels> kernels;
    };
    /**
     * @brief Authenticate the actual range/cohort/query and prepare one original decoder bundle.
     * @param storage Admitted table, DGO-issued query, original controller and distinct arena banks.
     * @param ids Original [M,16] hash IDs, shared without row rehashing or renumbering.
     * @param output Single original-head-order FP32 lookup result.
     * @return Shared immutable frame; no graph mutation or arena allocation occurs.
     * @throws std::exception For source/coordinate/count/control or binding geometry disagreement.
     */
    static std::shared_ptr<const Frame> frame(Qwen4ExpFixedHeadLookupStorage storage,
        Qwen4ExpIntBinding ids,Qwen4ExpFloatBinding output);
    /** @brief Retain a fully checked immutable frame and one semantic operation. */
    Qwen4ExpFixedHeadLookupStage(std::shared_ptr<const Frame> frame,Operation operation,std::string name);
    /** @return Actual immutable query/range/arena declaration retained by this graph node. */
    const Frame &declaration() const noexcept {return *frame_;}
    /** @brief Seal ordinary prepared arena roots and acquire the actual table upload on the exact stream. */
    bool prepareGraphLaunch(IDeviceContext *context,void *stream) override;
    /** @return Captured native replay needs no mutable host launch preparation. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override {return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @brief Validate retained identities, submit one local native operation and publish through the executor. */
    bool execute(IDeviceContext *context) override;
    /** @return Complete immutable operation/source/cohort/query/control/binding identity. */
    std::uint64_t graphCaptureVariantSignature() const override;
    /** @return Existing semantic native lookup type; only final decode publishes its snapshot. */
    ComputeStageType type() const override {return ComputeStageType::QWEN4EXP_NGRAM_LOOKUP;}
    /** @return Stable explicit graph node identity. */
    std::string name() const override {return name_;}
    /** @return Only the actual owning native backend is supported. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @return Counted primitive operations support partial/empty bucket replay. */
    bool supportsPaddedPrefillRealLengthContract() const override {return true;}
    /** @return Exact source-native producer/consumer arena roles for this operation. */
    StageBufferContract bufferContract() const override;
    /** @return Only final original-head output is a semantic model observation. */
    StageDumpInfo buildDumpInfoImpl() const override;
private:
    /** @brief Resolve only already prepared typed arena bytes; never materialize or rebind. */
    template<class T,TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding) const;
    /** @return Complete persistent frame with its original query count and compact packet extents. */
    qwen4exp::ImmutableFixedHeadLookupArgs arguments() const;
    std::shared_ptr<const Frame> frame_;
    Operation operation_;
    std::string name_;
    Qwen4ExpPreparedStorage prepared_;
    std::optional<ImmutableNativeDeviceRead> read_;
    std::optional<qwen4exp::ImmutableFixedHeadLookupCaptureIdentity> capture_;
};
} // namespace llaminar2
