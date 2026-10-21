/**
 * @file Qwen4ExpPredictorResidualBinding.h
 * @brief Checked Dense0 HC source and direct admitted wide-residual destinations.
 *
 * This immutable graph-construction input names its real frozen source, native
 * token-row authority and the existing arena's intermediate/terminal trunks.
 * It owns no allocation, execution state, event or independent memory ledger.
 */
#pragma once
#include "Qwen4ExpPredictorRows.h"
#include "Qwen4ExpPredictorGlobalBindings.h"
#include "Qwen4ExpPredictorProjectionLayout.h"
#include "Qwen4ExpHyperconnectionBuilder.h"
#include "../../../../models/qwen/LayerGraphIdentity.h"
#include <limits>
#include <span>
#include <stdexcept>
#include <variant>
namespace llaminar2
{
class BufferArena;
struct MTPDepthWeightBindings;
struct LayerWeights;
/** @brief Exact direct HC-write destination; terminal preserves the recursive4H value. */
enum class Qwen4ExpPredictorTrunk { Intermediate, Terminal };
/** @brief Metadata projection of one actual arena buffer, never an allocation request. */
struct Qwen4ExpDeclaredTrunk
{
    BufferId id;
    TensorType type;
    std::span<const size_t> shape;
    DeviceId device;
};
/** @brief Borrowed immutable write capability authenticated against one existing arena root. */
class Qwen4ExpPredictorTrunkTarget final
{
public:
    /**
     * @brief Admit exactly the predictor's existing FP32 intermediate or terminal root.
     * @param arena Canonical owner of already-declared activation tensors.
     * @param role Exact intermediate MTP_PROJECTED or terminal MTP_HIDDEN intent.
     * @param rows Positive current token-row capacity.
     * @param width Authenticated complete residual width4H.
     * @param device Exact participant.
     * @return Borrowed write target; GPU physical preparation is not required yet.
     * @throws std::exception For missing/external/view/storage/shape/participant mismatch.
     */
    static Qwen4ExpPredictorTrunkTarget fromArena(BufferArena &arena,
        Qwen4ExpPredictorTrunk role, int rows, int width, DeviceId device);
    /**
     * @brief Validate the same code-owned metadata contract without touching native storage.
     * @param role Required destination intent.
     * @param declared Actual arena descriptor.
     * @param rows Positive live-capacity envelope for this fragment.
     * @param width Authenticated residual width.
     * @param device Exact executing participant.
     * @throws std::exception For a foreign role/type/shape/device or byte-count overflow.
     */
    static void requireDeclared(Qwen4ExpPredictorTrunk role,
        const Qwen4ExpDeclaredTrunk &declared, int rows, int width, DeviceId device)
    {
        if ((role != Qwen4ExpPredictorTrunk::Intermediate && role != Qwen4ExpPredictorTrunk::Terminal) ||
            rows <= 0 || width <= 0 || !device.is_valid() || declared.device != device ||
            declared.id != bufferFor(role) || declared.type != TensorType::FP32 ||
            declared.shape.size() != 2 || declared.shape[0] < static_cast<size_t>(rows) ||
            declared.shape[1] != static_cast<size_t>(width))
            throw std::invalid_argument("Dense0 HC write requires its admitted exact FP32 residual role/device/shape");
        if (declared.shape[0] > std::numeric_limits<size_t>::max() / declared.shape[1] ||
            declared.shape[0] * declared.shape[1] > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::overflow_error("Dense0 HC write storage extent overflows");
    }
    /** @return Exact admitted existing arena operand; no pointer retargeting is exposed. */
    const Qwen4ExpFloatBinding &binding() const noexcept { return binding_; }
    /** @return The canonical graph arena, used to reject mixed-instance composition. */
    const BufferArena &arena() const noexcept { return *arena_; }
    /** @return Sealed intermediate versus recursive-terminal intent. */
    Qwen4ExpPredictorTrunk role() const noexcept { return role_; }
    /** @return Authenticated complete residual width supplied at canonical admission. */
    int width() const noexcept { return width_; }
    /** @return Physical BufferId selected only by its typed role. */
    static BufferId bufferFor(Qwen4ExpPredictorTrunk role)
    {
        if (role == Qwen4ExpPredictorTrunk::Intermediate) return BufferId::MTP_PROJECTED;
        if (role == Qwen4ExpPredictorTrunk::Terminal) return BufferId::MTP_HIDDEN;
        throw std::invalid_argument("Unknown Dense0 residual destination intent");
    }
private:
    /** @brief Only fromArena can issue a write capability over an admitted root. */
    Qwen4ExpPredictorTrunkTarget(BufferArena &arena, Qwen4ExpPredictorTrunk role, TensorBase &tensor, int width)
        : arena_(&arena), role_(role), width_(width), binding_(tensor, bufferFor(role)) {}
    BufferArena *arena_;
    Qwen4ExpPredictorTrunk role_;
    int width_;
    Qwen4ExpFloatBinding binding_;
};

/** @brief Explicit immutable predictor residual construction input for shared FA and FFN helpers. */
class Qwen4ExpPredictorResidualBinding final
{
public:
    /**
     * @brief Select the exact invocation capability used by public predictor construction.
     * @param invocation Draft writes or shifted-prefill reads, never an inferred capacity mode.
     * @param source Checked original predictor source/cache identity.
     * @param weights Immutable frozen original predictor depth.
     * @param rows Original token rows and their sole native count owner.
     * @param layout Authenticated frozen participant projection policy.
     * @param arena Existing admitted activation owner; only Draft borrows write roots.
     * @param width Exact residual width required for Draft write admission.
     * @return Read-only shifted-prefill or strict direct-write Draft capability.
     * @throws std::exception For an unknown invocation or invalid selected source/row/write contract.
     */
    static Qwen4ExpPredictorResidualBinding fromArena(Qwen4ExpPredictorInvocation invocation,
        const LayerGraphSourceIdentity &source, const MTPDepthWeightBindings &weights,
        Qwen4ExpPredictorRows rows, Qwen4ExpPredictorProjectionLayout layout,
        BufferArena &arena, int width);
    /**
     * @brief Admit frozen source48 HC weights, original counted rows and direct4H outputs.
     * @param source Checked paired predictor source/cache identity.
     * @param weights Existing frozen predictor depth, preserving native source ownership.
     * @param rows Original token rows, never an independently repeated scalar.
     * @param layout Authenticated frozen local projection/sharing policy.
     * @param attention_output Admitted MTP_PROJECTED receiving the attention HC write.
     * @param ffn_output Admitted MTP_HIDDEN receiving the final FFN HC write directly.
     * @return Construction-only binding with no live state shadow or allocation.
     * @throws std::invalid_argument For contradictory identity, QSA/PLE bindings, roles or owners.
     */
    static Qwen4ExpPredictorResidualBinding admit(const LayerGraphSourceIdentity &source,
        const MTPDepthWeightBindings &weights, Qwen4ExpPredictorRows rows,
        Qwen4ExpPredictorProjectionLayout layout, Qwen4ExpPredictorTrunkTarget attention_output, Qwen4ExpPredictorTrunkTarget ffn_output);
    /**
     * @brief Admit only HC read/projection operands for shifted KV-prefill.
     *
     * A bucket-wide K/V producer never writes attention/FFN residuals or the
     * recursive draft output. It therefore owns no write capability and cannot
     * require or enlarge the separately admitted compact draft destination.
     *
     * @param source Checked original predictor source and cache identity.
     * @param weights Existing immutable source48 frozen depth.
     * @param rows Exact original shifted-prefill rows/count owner.
     * @param layout Authenticated frozen participant projection policy.
     * @return Read-only construction binding with no residual destinations.
     * @throws std::invalid_argument For a foreign source, count device or frozen layout.
     */
    static Qwen4ExpPredictorResidualBinding forShiftedPrefill(const LayerGraphSourceIdentity &source,
        const MTPDepthWeightBindings &weights, Qwen4ExpPredictorRows rows,
        Qwen4ExpPredictorProjectionLayout layout);
    /** @return Explicit invocation derived from the owned read-only or write variant. */
    Qwen4ExpPredictorInvocation invocation() const noexcept
    {
        return std::holds_alternative<ShiftedPrefillRead>(use_)
            ? Qwen4ExpPredictorInvocation::ShiftedPrefill : Qwen4ExpPredictorInvocation::Draft;
    }
    /**
     * @brief Reject the wrong construction role before any graph stage is added.
     * @param required Exact public KV-only or draft body being entered.
     * @throws std::invalid_argument For unknown or contradictory invocation.
     */
    void requireInvocation(Qwen4ExpPredictorInvocation required) const
    {
        if ((required!=Qwen4ExpPredictorInvocation::Draft && required!=Qwen4ExpPredictorInvocation::ShiftedPrefill) ||
            required!=invocation())
            throw std::invalid_argument("Dense0 residual construction belongs to another predictor invocation");
    }
    /** @return Exact immutable frozen sidecar source. */
    const MTPDepthWeightBindings &weights() const noexcept { return *weights_; }
    /** @return Original query rows with their sole native count authority. */
    const Qwen4ExpPredictorRows &rows() const noexcept { return rows_; }
    /** @return Authenticated native local heads and dense/shared reduction policy. */
    const Qwen4ExpPredictorProjectionLayout &layout() const noexcept { return layout_; }
    /**
     * @return Direct admitted intermediate destination for draft attention injection.
     * @throws std::logic_error When a read-only shifted-prefill binding is misused as a writer.
     */
    const Qwen4ExpPredictorTrunkTarget &attentionOutput() const { return draftWrites().attention; }
    /**
     * @return Direct admitted recursive4H destination for draft FFN injection.
     * @throws std::logic_error When a read-only shifted-prefill binding is misused as a writer.
     */
    const Qwen4ExpPredictorTrunkTarget &ffnOutput() const { return draftWrites().ffn; }
    /**
     * @brief Reject substituting foreign native tensor pointers for the sealed source bundle.
     * @param layer Actual legacy projection view supplied to the common FA/FFN body.
     * @throws std::invalid_argument For any source matrix, QK gain or routed/shared mismatch.
     */
    void requireLayerSource(const LayerWeights &layer) const;
    /** @brief Reject any frozen source/policy before stage insertion, without materializing payloads. */
    static void requireFrozenSource(const LayerGraphSourceIdentity &source,
        const MTPDepthWeightBindings &weights);
private:
    /** @brief Shifted prefill owns no residual write target. */
    struct ShiftedPrefillRead {};
    /** @brief Draft owns both already authenticated direct write capabilities. */
    struct DraftWrites
    {
        Qwen4ExpPredictorTrunkTarget attention, ffn;
    };
    /**
     * @brief Store the named read-only variant without inventing a destination.
     * @param weights Checked immutable source depth whose owner outlives this binding.
     * @param rows Original token/count descriptor.
     * @param layout Checked projection/source identity.
     */
    Qwen4ExpPredictorResidualBinding(const MTPDepthWeightBindings &weights, Qwen4ExpPredictorRows rows,
        Qwen4ExpPredictorProjectionLayout layout)
        : weights_(&weights), rows_(std::move(rows)), layout_(std::move(layout)), use_(ShiftedPrefillRead{}) {}
    /**
     * @brief Store only checked immutable draft operands and both direct writes.
     * @param weights Checked immutable source depth.
     * @param rows Original token/count descriptor.
     * @param layout Checked projection/source identity.
     * @param attention_output Admitted intermediate write capability.
     * @param ffn_output Admitted recursive-terminal write capability.
     */
    Qwen4ExpPredictorResidualBinding(const MTPDepthWeightBindings &weights, Qwen4ExpPredictorRows rows,
        Qwen4ExpPredictorProjectionLayout layout, Qwen4ExpPredictorTrunkTarget attention_output, Qwen4ExpPredictorTrunkTarget ffn_output)
        : weights_(&weights), rows_(std::move(rows)), layout_(std::move(layout)),
          use_(DraftWrites{std::move(attention_output),std::move(ffn_output)}) {}
    const MTPDepthWeightBindings *weights_;
    Qwen4ExpPredictorRows rows_;
    Qwen4ExpPredictorProjectionLayout layout_;
    std::variant<ShiftedPrefillRead,DraftWrites> use_;
    /**
     * @return The sole authenticated draft-write alternative.
     * @throws std::logic_error For read-only shifted-prefill construction.
     */
    const DraftWrites &draftWrites() const
    {
        const auto *writes=std::get_if<DraftWrites>(&use_);
        if(!writes)throw std::logic_error("Dense0 shifted-prefill binding has no residual write destinations");
        return *writes;
    }
};
} // namespace llaminar2
