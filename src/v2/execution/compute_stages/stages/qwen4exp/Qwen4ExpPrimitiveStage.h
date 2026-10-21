/**
 * @file Qwen4ExpPrimitiveStage.h
 * @brief Declarative backend-neutral GR/PLE stages with mandatory arena bindings.
 *
 * Operation variants make each primitive's operands and geometry explicit.
 * Projection and norm stages remain independent reusable graph operations.
 * All tensor views, counts, policies and kernel ownership are sealed during
 * construction; execute only verifies prepared storage and submits arithmetic
 * on the exact executor stream. Histories are immutable candidate inputs until
 * an explicit accepted-prefix publication stage writes a distinct next bank.
 */
#pragma once
#include "../../IComputeStage.h"
#include "../RMSNormStage.h"
#include "../../StageParamsBase.h"
#include "../../../../kernels/qwen4exp/Qwen4ExpKernels.h"
#include "../../../../tensors/TensorType.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace llaminar2
{
/** @brief Required typed arena tensor view, including an explicit byte offset.
 * @tparam Type Native scalar type; no execution-time conversions are permitted.
 */
template <TensorType Type> struct Qwen4ExpArenaBinding
{
    TensorBase *tensor; ///< Borrowed arena tensor, retained by the graph.
    BufferId buffer; ///< Mandatory coherence/ownership identity.
    std::size_t byteOffset; ///< Admitted subview offset; part of capture identity.
    /** @brief Bind an existing arena tensor without taking ownership.
     * @param value Arena tensor whose native type is checked by the stage.
     * @param id Canonical graph buffer identity.
     * @param offset Explicit aligned byte offset into the admitted allocation.
     */
    Qwen4ExpArenaBinding(TensorBase &value, BufferId id, std::size_t offset = 0)
        : tensor(&value), buffer(id), byteOffset(offset) {}
};
/** @brief FP32 model-specific arithmetic binding. */
using Qwen4ExpFloatBinding = Qwen4ExpArenaBinding<TensorType::FP32>;
/** @brief INT32 tokens, control words and packed INT64 row-ID backing. */
using Qwen4ExpIntBinding = Qwen4ExpArenaBinding<TensorType::INT32>;

/**
 * @brief Borrow immutable INT32 input coordinates from the execution authority.
 *
 * Tokens and controller counts may live in the generation session rather than
 * the activation arena. The producer remains their sole owner. The enclosing
 * graph supplies its producer edge and retains this exact address/capacity in
 * capture identity; this view never uploads, mirrors or owns the values.
 */
struct Qwen4ExpNativeIntInput
{
    DeviceId owner; ///< Explicit CPU/CUDA/ROCm storage authority.
    const std::int32_t *data; ///< Native address on that authority, including CPU spans.
    std::size_t capacity; ///< Admitted scalar extent, independent of live contents.
};
/** @brief An arena input or an explicitly borrowed generation-session input. */
struct Qwen4ExpIntInput
{
    std::variant<Qwen4ExpIntBinding, Qwen4ExpNativeIntInput> value;
    /** @brief Retain an already checked arena subview. */
    Qwen4ExpIntInput(Qwen4ExpIntBinding binding) : value(binding) {}
    /** @brief Retain one exact native execution-owner input. */
    Qwen4ExpIntInput(Qwen4ExpNativeIntInput binding) : value(binding) {}
    /** @brief Bind an arena tensor directly with its canonical identity and offset. */
    Qwen4ExpIntInput(TensorBase &tensor, BufferId id, std::size_t offset = 0)
        : value(Qwen4ExpIntBinding(tensor, id, offset)) {}
};

/** @brief Immutable native FP32 model weight, owned by the frozen model weight set.
 * @details This is the same raw-weight lifetime contract used by RMSNormStage:
 * the executor prepares its exact source tensor once and the stage consumes
 * already-resident FP32 bytes. It has no activation-arena BufferId and performs
 * no packing, arena copy or gain conversion during execution.
 */
struct Qwen4ExpImmutableWeightBinding
{
    const TensorBase *tensor; ///< Frozen model-owned norm gain or convolution coefficients.
    /** @brief Bind the exact native model-weight tensor without taking ownership.
     * @param source Immutable FP32 source retained by the model weight authority.
     */
    explicit Qwen4ExpImmutableWeightBinding(const TensorBase &source) : tensor(&source) {}
};

/** @brief Sealed row capacity with an optional authoritative execution-owner count. */
struct Qwen4ExpStageRows
{
    int capacity; ///< Positive admitted maximum row count.
    std::optional<Qwen4ExpIntInput> liveRows; ///< One INT32 count on the execution device.
};
/** @brief Explicit candidate-prefix acceptance; never inferred from host mutable state. */
struct Qwen4ExpStageAcceptance
{
    std::optional<Qwen4ExpIntInput> acceptedRows; ///< One device-/CPU-owned INT32 control word.
    int fixedAcceptedRows = 0; ///< Fixed setup policy when acceptedRows is absent.
};

/** @brief One semantic GR/PLE compute operation selected declaratively by its variant. */
class Qwen4ExpPrimitiveStage final : public IComputeStage
{
public:
    /** @brief Reuse the existing normalization diagnostic policy; no second publication mode. */
    using DiagnosticInputPublication = RMSNormStage::DiagnosticInputPublication;
    /** @brief Mathematical normalization input role, independent of diagnostic snapshot policy. */
    enum class BranchNormInputRole : std::uint8_t
    {
        Ordinary, ///< Ordinary residual branches retain the uninitialized-zero diagnostic.
        PredictorHidden, ///< Published original predictor hidden may be the model-defined zero seed.
    };
    /** @brief Normalize all four branches with distinct actual gains. */
    struct BranchRMSNorm
    {
        int hidden; ///< Per-branch compute width H.
        float epsilon; ///< Positive finite checkpoint epsilon.
        Qwen4ExpFloatBinding input; ///< [M,4H] original branches.
        Qwen4ExpImmutableWeightBinding gain; ///< [4H] actual branch-specific model gain.
        Qwen4ExpFloatBinding output; ///< [M,4H], may alias input exactly.
        /** Optional read-only transaction input publication; requires disjoint original input/output. */
        DiagnosticInputPublication diagnostic_input_publication = DiagnosticInputPublication::None;
        BranchNormInputRole input_role = BranchNormInputRole::Ordinary; ///< Immutable zero-preserving semantic role.
        /**
         * @brief Declare the predictor's zero-preserving original hidden normalization.
         * @param hidden Per-branch width from the checked predictor geometry.
         * @param epsilon Original positive finite normalization epsilon.
         * @param input Exact original4H hidden arena input; its producer/lease remains authoritative.
         * @param gain Immutable original branch gain, not an activation owner.
         * @param output Canonical MTP_NORM_HIDDEN4H arena destination; draft publication stays disjoint.
         * @param diagnostic Optional readonly input snapshot; never permission to skip validation.
         * @return Typed normalization contract. The stage validates roles, geometry and aliases.
         */
        static BranchRMSNorm predictorHidden(int hidden, float epsilon, Qwen4ExpFloatBinding input,
            Qwen4ExpImmutableWeightBinding gain, Qwen4ExpFloatBinding output,
            DiagnosticInputPublication diagnostic = DiagnosticInputPublication::None)
        {
            return {hidden, epsilon, input, gain, output, diagnostic, BranchNormInputRole::PredictorHidden};
        }
    };
    /** @brief Expand H-wide embedding rows into four residual branches. */
    struct ResidualExpand
    {
        int hidden; ///< Compute width H.
        Qwen4ExpFloatBinding input, output; ///< [M,H] input and distinct [M,4H] output.
    };
    /** @brief Apply /4 and SiLU to the existing low-rank down projection. */
    struct ResidualDown
    {
        int lowRank; ///< GR bottleneck width.
        Qwen4ExpFloatBinding input, output; ///< [M,lowRank], optionally in place.
    };
    /** @brief Gated normalized branch read, also used for the final mixer. */
    struct ResidualRead
    {
        int hidden; ///< Compute width H.
        Qwen4ExpFloatBinding normalized, gateLogits, output; ///< [M,4H], [M,4H], [M,H].
    };
    /** @brief Learned scalar injection of a block result into four branches. */
    struct ResidualWrite
    {
        int hidden; ///< Compute width H.
        Qwen4ExpFloatBinding residual, block, gateLogits, output; ///< [M,4H], [M,H], [M,4], [M,4H].
    };
    /** @brief Exact suffix hash from INT32 token IDs into packed INT64 lookup IDs.
     * @details outputWords is INT32 [M,32] backing for INT64 [M,16], with an
     * eight-byte-aligned byte offset. It never converts the 64-bit values to
     * float or narrows their row-index range. Diagnostics expose the raw words.
     */
    struct NgramHash
    {
        Qwen4ExpIntInput tokens; ///< [M] exact tokens from the execution authority.
        Qwen4ExpIntBinding priorTokens, outputWords; ///< [2], [M,32] raw words.
        qwen4exp::NgramHashPolicy policy; ///< Exact authenticated checkpoint constants.
    };
    /** @brief Signed-square-root branch similarity and shared-value gating. */
    struct PLEGate
    {
        int hidden; ///< Compute width H.
        Qwen4ExpFloatBinding normalizedKey, normalizedQuery, value, gated; ///< [M,4H], [M,4H], [M,H], [M,4H].
    };
    /** @brief Dilated four-tap depthwise convolution and PLE residual addition. */
    struct PLEConv
    {
        int channels; ///< Trunk width 4H.
        Qwen4ExpFloatBinding normalized, priorHistory; ///< [M,4H], immutable [9,4H].
        Qwen4ExpImmutableWeightBinding weights; ///< [4H,4] frozen native model coefficients.
        Qwen4ExpFloatBinding gated, residual, output; ///< [M,4H] values and residual publication.
    };
    /** @brief Publish nine normalized convolution rows after explicit acceptance. */
    struct CommitConvHistory
    {
        int channels; ///< Trunk width 4H.
        Qwen4ExpStageAcceptance acceptance; ///< Device/CPU accepted-prefix authority.
        Qwen4ExpFloatBinding priorHistory, candidates, outputHistory; ///< [9,4H], [M,4H], distinct [9,4H].
    };
    /** @brief Publish the last two raw tokens after explicit acceptance. */
    struct CommitTokenHistory
    {
        Qwen4ExpStageAcceptance acceptance; ///< Device/CPU accepted-prefix authority.
        Qwen4ExpIntBinding priorTokens; ///< [2] immutable prior suffix.
        Qwen4ExpIntInput candidates; ///< [M] exact execution-owner candidate tokens.
        Qwen4ExpIntBinding outputTokens; ///< Distinct next [2] suffix bank.
    };
    /** @brief Closed operation set; absent operands cannot be represented. */
    using Operation = std::variant<BranchRMSNorm, ResidualExpand, ResidualDown, ResidualRead, ResidualWrite,
                                   NgramHash, PLEGate, PLEConv, CommitConvHistory, CommitTokenHistory>;
    /** @brief Immutable graph configuration; no backend-specific switch is exposed. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        Qwen4ExpStageRows rows; ///< Admitted geometry and optional count subview.
        Operation operation; ///< Fully typed semantic arithmetic/state operation.
    };
    static_assert(StageParamsRequired<Params>);

    /** @brief Validate and seal one operation and construct its backend bundle.
     * @param params Immutable geometry, arena views and execution-device policy.
     * @throws std::invalid_argument On type, geometry, capacity, alignment or alias violations.
     */
    explicit Qwen4ExpPrimitiveStage(Params params);
    /** @brief Run only the selected arithmetic on already-prepared storage.
     * @param context Matching graph executor context; never used to repair coherence.
     * @return True after successful CPU execution or GPU submission.
     * @throws std::logic_error If GPU preparation/stream ownership is missing.
     */
    bool execute(IDeviceContext *context) override;
    /** @brief Return the semantic operation tag without inspecting live device state. */
    ComputeStageType type() const override;
    /** @brief Distinguish an additional captured diagnostic input view from ordinary normalization. */
    uint64_t graphCaptureVariantSignature() const override;
    /**
     * @brief Permit mathematically valid zero only for the checked predictor-hidden norm role.
     * @details RMS(x=0,gain,epsilon>0)=0, including the visible-zero bootstrap.
     * Original hidden publication/lease remains an execution-boundary obligation.
     * This changes only the zero heuristic; null/nonfinite/shape validation stays active.
     */
    bool allowsZeroOutput() const override;
    /** @brief Return arithmetic at the admitted row bound; dynamic counts stay device-owned. */
    std::size_t estimatedFlops() const override;
    /** @brief Return exact bound operand bytes, excluding spare tensor capacity. */
    std::size_t estimatedMemoryBytes() const override;
    /** @brief Report support for the backend already selected by construction.
     * @param backend Backend being queried by placement validation.
     * @return True only for the constructed bundle's backend.
     */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @brief Describe precise bound views without reading dynamic device counts. */
    StageDumpInfo buildDumpInfoImpl() const override;
    /** @brief No scratch allocation is owned or requested by these primitives. */
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @brief Return complete mandatory read/write arena contracts, including count words. */
    StageBufferContract bufferContract() const override;
    /** @brief Inspect sealed configuration; no mutable runtime updates are supported. */
    const Params &getParams() const noexcept { return params_; }

private:
    /** @brief Access policy of one immutable byte view. */
    enum class Access { Input, Output };
    /** @brief Whether an input view permits an exact in-place output alias. */
    enum class AliasPolicy { Disjoint, ExactOutputAliasAllowed };
    /** @brief Prevalidated operand, retained outside the execution hot path. */
    struct Operand
    {
        TensorBase *tensor;
        BufferId buffer;
        TensorType dtype;
        std::size_t byteOffset, rows, columns;
        Access access;
        const char *name;
        AliasPolicy alias; ///< Typed operation-specific write-overlap permission.
        /** @brief Native extent in bytes, excluding unused capacity. */
        std::size_t bytes() const noexcept { return rows * columns * sizeof(std::int32_t); }
    };
    /** @brief Register one operand after native-type, alignment and byte-capacity checks. */
    template <TensorType Type> void add(Qwen4ExpArenaBinding<Type> binding,
        std::size_t rows, std::size_t columns, Access access, const char *name,
        AliasPolicy alias = AliasPolicy::Disjoint);
    /** @brief Validate an arena or native INT32 input against its declared owner and extent. */
    void addIntInput(const Qwen4ExpIntInput &binding, std::size_t count, const char *name);
    /** @brief Resolve native input storage without host access or allocation. */
    const std::int32_t *intInput(const Qwen4ExpIntInput &binding) const;
    /** @brief One immutable model-weight view, separate from activation arena identity. */
    struct WeightOperand
    {
        const TensorBase *tensor;
        std::size_t rows, columns;
        const char *name;
        /** @brief Exact native FP32 weight extent consumed by this operation. */
        std::size_t bytes() const noexcept { return rows * columns * sizeof(float); }
    };
    /** @brief Register native immutable FP32 model bytes without inventing an arena buffer. */
    void addWeight(Qwen4ExpImmutableWeightBinding binding, std::size_t rows, std::size_t columns, const char *name);
    /** @brief Resolve an already-prepared immutable model weight on this stage's device. */
    const float *weightPointer(Qwen4ExpImmutableWeightBinding binding) const;
    /** @brief Register optional acceptance control and reject invalid fixed policies. */
    void addAcceptance(const Qwen4ExpStageAcceptance &acceptance);
    /** @brief Reject overlapping input/output views unless exact in-place semantics allow them. */
    void validateAliases() const;
    /** @brief Resolve a prepared device pointer or authoritative CPU byte view. */
    template <typename T, TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const;
    /** @brief Resolve execution-owner live count without copying it across devices. */
    qwen4exp::RowExtent rowExtent() const;
    /** @brief Resolve execution-owner accepted prefix without host device-state access. */
    qwen4exp::AcceptedPrefix acceptedPrefix(const Qwen4ExpStageAcceptance &acceptance) const;

    Params params_; ///< Immutable operation identity, including pointers and policy scalars.
    std::vector<Operand> operands_; ///< Complete constructor-created contract/dump views.
    std::vector<WeightOperand> weights_; ///< Frozen model weights published through the raw-weight contract.
    /** @brief Borrowed generation inputs, retained for alias validation and byte accounting. */
    std::vector<Qwen4ExpNativeIntInput> native_inputs_;
    std::unique_ptr<qwen4exp::IQwen4ExpKernels> kernels_; ///< Stateless bundle constructed before capture.
};
} // namespace llaminar2
