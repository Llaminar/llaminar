/**
 * @file Qwen4ExpBranchConcatStage.h
 * @brief Dense0 branch concatenation borrowing the original token-row authority.
 *
 * One token row writes four [embedding|hidden_branch] rows without expanding
 * the live-count owner. All arena intervals and native policy are sealed before
 * capture. The following FC may repeat this SAME descriptor four times.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../../kernels/qwen4exp/Qwen4ExpBranchConcat.h"
#include "../../../qwen4exp/Qwen4ExpPreparedStorage.h"
namespace llaminar2 {
/** @brief Participant-local, bit-preserving Dense0 input concatenation. */
class Qwen4ExpBranchConcatStage final : public IComputeStage {
public:
    /** @brief Exact typed arena operands with original token-row geometry. */
    struct Params {
        STAGE_PARAMS_COMMON_FIELDS;
        DeviceRowRange rows; ///< Original token rows; repeats_per_row must be one.
        int hidden_dim; ///< Positive per-branch width H.
        Qwen4ExpFloatBinding hidden; ///< [M,4H], fixed branch order.
        Qwen4ExpFloatBinding embedding; ///< [M,H], shared across four branches.
        Qwen4ExpFloatBinding output; ///< [M,8H], laid out [M,4,2H].
    };
    static_assert(StageParamsRequired<Params>);
    /** @brief Reject wrong ownership, repeated counts, capacity, overflow and aliasing. */
    explicit Qwen4ExpBranchConcatStage(Params params);
    /** @brief Copy only live token rows on the exact prepared executor stream. */
    bool execute(IDeviceContext *context) override;
    /** @brief Return the dedicated branch-concatenation semantic tag. */
    ComputeStageType type() const override;
    /** @brief No floating point arithmetic is performed. */
    std::size_t estimatedFlops() const override { return 0; }
    /** @brief Return admitted operand bytes, never a live transfer claim. */
    std::size_t estimatedMemoryBytes() const override;
    /** @brief Admit only the constructor-selected participant backend. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @brief Caller arena owns all operands; no extra workspace is required. */
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @brief Declare the two exact arena inputs and output. */
    StageBufferContract bufferContract() const override;
    /** @brief Preserve exact byte subviews and original row descriptor identity. */
    StageDumpInfo buildDumpInfoImpl() const override;
    /** @brief Seal ordinarily prepared root storage once before capture. */
    bool prepareGraphLaunch(IDeviceContext *context,void *stream) override;
    /** @brief Native replay never invokes host storage preparation. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /** @brief Stable declaration identity authenticates every pointer through its physical seal. */
    std::uint64_t graphCaptureVariantSignature() const override;
    /** @brief Live row gating preserves dirty inactive rows without transfer. */
    bool supportsPaddedPrefillRealLengthContract() const override { return true; }
    /** @brief Expose immutable setup operands for declarative composition. */
    const Params &getParams() const noexcept { return params_; }
private:
    /** @brief Resolve an already prepared CPU or GPU arena subview. */
    float *pointer(Qwen4ExpFloatBinding binding) const;
    Params params_;
    qwen4exp::BranchConcatOperation operation_;
    std::optional<Qwen4ExpPreparedStorage> storage_;
};
}
