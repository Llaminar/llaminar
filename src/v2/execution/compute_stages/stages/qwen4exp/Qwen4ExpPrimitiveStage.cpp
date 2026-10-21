/**
 * @file Qwen4ExpPrimitiveStage.cpp
 * @brief Bind declarative GR/PLE variants to prepared participant-local storage.
 *
 * Constructor-created operand descriptors are the single extent authority for
 * type/capacity checks, buffer contracts, estimates and dumps. GPU execution
 * never touches a host count or repairs coherence. Candidate evaluation and
 * accepted-prefix publication remain distinct semantic stages and storage banks.
 */
#include "Qwen4ExpPrimitiveStage.h"
#include <string_view>
#include "../../../../kernels/KernelFactory.h"
#include "../../../../kernels/qwen4exp/Qwen4ExpValidation.h"
#include "../../../../tensors/Tensors.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
namespace
{
/** @brief Visit a closed operation set with distinct compile-time handlers. */
template <class... Functions> struct Overload : Functions... { using Functions::operator()...; };
template <class... Functions> Overload(Functions...) -> Overload<Functions...>;
/** @brief Reject malformed stage setup before any execution can begin. */
void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(message);
}
} // namespace

template <TensorType Type>
void Qwen4ExpPrimitiveStage::add(Qwen4ExpArenaBinding<Type> binding,
    std::size_t rows, std::size_t columns, Access access, const char *name, AliasPolicy alias)
{
    static_assert(Type == TensorType::FP32 || Type == TensorType::INT32);
    require(binding.tensor && binding.tensor->native_type() == Type,
            "Qwen4Exp stage arena operand has the wrong native scalar type");
    require(binding.buffer < BufferId::_COUNT, "Qwen4Exp stage requires a canonical arena BufferId");
    require(rows > 0 && columns > 0 && rows <= std::numeric_limits<std::size_t>::max() / columns / 4,
            "Qwen4Exp stage operand geometry is empty or overflows bytes");
    const auto bytes = rows * columns * 4;
    require(binding.byteOffset % 4 == 0 && binding.byteOffset <= binding.tensor->size_bytes() &&
                bytes <= binding.tensor->size_bytes() - binding.byteOffset,
            "Qwen4Exp stage arena view is misaligned or exceeds admitted tensor capacity");
    operands_.push_back({binding.tensor, binding.buffer, Type, binding.byteOffset, rows, columns, access, name, alias});
}

void Qwen4ExpPrimitiveStage::addIntInput(const Qwen4ExpIntInput &binding,
    std::size_t count, const char *name)
{
    std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { add(value, count, 1, Access::Input, name); },
        [&](Qwen4ExpNativeIntInput value)
        {
            require(value.owner == device() && value.data && count > 0 && count <= value.capacity &&
                        reinterpret_cast<std::uintptr_t>(value.data) % alignof(std::int32_t) == 0 &&
                        value.capacity <= (std::numeric_limits<std::uintptr_t>::max() -
                            reinterpret_cast<std::uintptr_t>(value.data)) / sizeof(std::int32_t),
                    "Qwen4Exp native input requires an exact matching owner, aligned address and admitted extent");
            // Only the consumed extent belongs to this stage's read contract.
            value.capacity = count;
            native_inputs_.push_back(value);
        }
    }, binding.value);
}

const std::int32_t *Qwen4ExpPrimitiveStage::intInput(const Qwen4ExpIntInput &binding) const
{
    return std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { return pointer<const std::int32_t>(value, Access::Input); },
        [](Qwen4ExpNativeIntInput value) { return value.data; }
    }, binding.value);
}

void Qwen4ExpPrimitiveStage::addWeight(Qwen4ExpImmutableWeightBinding binding,
    std::size_t rows, std::size_t columns, const char *name)
{
    require(binding.tensor && binding.tensor->native_type() == TensorType::FP32,
            "Qwen4Exp stage immutable model weight must use native FP32 storage");
    require(rows > 0 && columns > 0 && rows <= std::numeric_limits<std::size_t>::max() / columns / 4 &&
                rows * columns * 4 <= binding.tensor->size_bytes(),
            "Qwen4Exp stage immutable model-weight geometry exceeds its source tensor");
    weights_.push_back({binding.tensor,rows,columns,name});
}

const float *Qwen4ExpPrimitiveStage::weightPointer(Qwen4ExpImmutableWeightBinding binding) const
{
    const void *base = device().is_gpu() ? binding.tensor->gpu_data_ptr() : binding.tensor->raw_data();
    if (!base || reinterpret_cast<std::uintptr_t>(base) % alignof(float) != 0)
        throw std::logic_error("Qwen4Exp stage immutable model weight is not already bound and aligned");
    return static_cast<const float *>(base);
}

void Qwen4ExpPrimitiveStage::addAcceptance(const Qwen4ExpStageAcceptance &acceptance)
{
    if (acceptance.acceptedRows)
        addIntInput(*acceptance.acceptedRows, 1, "accepted_rows");
    else
        require(acceptance.fixedAcceptedRows >= 0 && acceptance.fixedAcceptedRows <= params_.rows.capacity,
                "Qwen4Exp stage fixed accepted prefix exceeds admitted candidate rows");
}

Qwen4ExpPrimitiveStage::Qwen4ExpPrimitiveStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
    require(params_.device_id.is_valid(), "Qwen4Exp stage requires an explicit valid participant device");
    qwen4exp::detail::validateExtent({params_.rows.capacity}, 1);
    const auto rows = static_cast<std::size_t>(params_.rows.capacity);
    if (params_.rows.liveRows) addIntInput(*params_.rows.liveRows, 1, "live_rows");
    const auto width = [&](int value) {
        qwen4exp::detail::validateExtent({params_.rows.capacity}, value);
        return static_cast<std::size_t>(value);
    };
    const auto channels = [&](int value) {
        require(value % 4 == 0, "Qwen4Exp stage trunk width must contain exactly four equal branches");
        return width(value);
    };
    std::visit(Overload{
        [&](const BranchRMSNorm &a) {
            const auto h = width(a.hidden);
            require(std::isfinite(a.epsilon) && a.epsilon > 0, "Qwen4Exp branch RMS epsilon must be finite and positive");
            require(a.diagnostic_input_publication == DiagnosticInputPublication::None ||
                    a.diagnostic_input_publication == DiagnosticInputPublication::MTPTerminalHidden,
                    "QFN branch norm has an unknown diagnostic publication policy");
            require(a.input_role == BranchNormInputRole::Ordinary ||
                    a.input_role == BranchNormInputRole::PredictorHidden,
                    "QFN branch norm has an unknown mathematical input role");
            const bool predictor = a.input_role == BranchNormInputRole::PredictorHidden;
            if (predictor)
            {
                const bool hidden_role = a.input.buffer == BufferId::HIDDEN_STATE ||
                    a.input.buffer == BufferId::PREFIX_TERMINAL_HIDDEN || a.input.buffer == BufferId::MTP_HIDDEN ||
                    a.input.buffer == BufferId::MTP_NORM_HIDDEN;
                require(hidden_role && a.input.tensor && a.output.tensor &&
                        a.output.buffer == BufferId::MTP_NORM_HIDDEN &&
                        a.input.byteOffset == 0 && a.output.byteOffset == 0 &&
                        a.input.tensor->shape().size() == 2 && a.output.tensor->shape().size() == 2 &&
                        a.input.tensor->cols() == 4*h && a.output.tensor->cols() == 4*h,
                        "QFN predictor norm requires exact original4H hidden and canonical normalized-hidden roots");
            }
            const bool publish = a.diagnostic_input_publication == DiagnosticInputPublication::MTPTerminalHidden;
            if (publish)
                require(a.output.buffer == BufferId::MTP_NORM_HIDDEN,
                        "QFN MTP input publication requires its canonical normalized-hidden output role");
            // A diagnostic is recorded after normalization. In-place storage
            // would already contain normalized bytes, not the consumed input.
            add(a.input, rows, 4*h, Access::Input, "input",
                publish ? AliasPolicy::Disjoint : AliasPolicy::ExactOutputAliasAllowed);
            addWeight(a.gain,4,h,"gain");
            add(a.output, rows, 4*h, Access::Output, "output");
        },
        [&](const ResidualExpand &a) {
            const auto h = width(a.hidden);
            add(a.input, rows, h, Access::Input, "input");
            add(a.output, rows, 4*h, Access::Output, "output");
        },
        [&](const ResidualDown &a) {
            const auto h = width(a.lowRank);
            add(a.input, rows, h, Access::Input, "input", AliasPolicy::ExactOutputAliasAllowed);
            add(a.output, rows, h, Access::Output, "output");
        },
        [&](const ResidualRead &a) {
            const auto h = width(a.hidden);
            add(a.normalized, rows, 4*h, Access::Input, "normalized");
            add(a.gateLogits, rows, 4*h, Access::Input, "gate_logits");
            add(a.output, rows, h, Access::Output, "branch_read");
        },
        [&](const ResidualWrite &a) {
            const auto h = width(a.hidden);
            add(a.residual, rows, 4*h, Access::Input, "residual", AliasPolicy::ExactOutputAliasAllowed);
            add(a.block, rows, h, Access::Input, "block");
            add(a.gateLogits, rows, 4, Access::Input, "gate_logits");
            add(a.output, rows, 4*h, Access::Output, "output");
        },
        [&](const NgramHash &a) {
            a.policy.validate();
            require(a.outputWords.byteOffset % 8 == 0, "Qwen4Exp packed INT64 row IDs require eight-byte alignment");
            addIntInput(a.tokens, rows, "tokens");
            add(a.priorTokens, 1, 2, Access::Input, "prior_tokens");
            add(a.outputWords, rows, 32, Access::Output, "row_id_words");
        },
        [&](const PLEGate &a) {
            const auto h = width(a.hidden);
            add(a.normalizedKey, rows, 4*h, Access::Input, "normalized_key");
            add(a.normalizedQuery, rows, 4*h, Access::Input, "normalized_query");
            add(a.value, rows, h, Access::Input, "value");
            add(a.gated, rows, 4*h, Access::Output, "gated");
        },
        [&](const PLEConv &a) {
            const auto c = channels(a.channels);
            add(a.normalized, rows, c, Access::Input, "normalized");
            add(a.priorHistory, 9, c, Access::Input, "prior_history");
            addWeight(a.weights,c,4,"weights");
            add(a.gated, rows, c, Access::Input, "gated", AliasPolicy::ExactOutputAliasAllowed);
            add(a.residual, rows, c, Access::Input, "residual", AliasPolicy::ExactOutputAliasAllowed);
            add(a.output, rows, c, Access::Output, "output");
        },
        [&](const CommitConvHistory &a) {
            const auto c = channels(a.channels);
            addAcceptance(a.acceptance);
            add(a.priorHistory, 9, c, Access::Input, "prior_history");
            add(a.candidates, rows, c, Access::Input, "candidates");
            add(a.outputHistory, 9, c, Access::Output, "output_history");
        },
        [&](const CommitTokenHistory &a) {
            addAcceptance(a.acceptance);
            add(a.priorTokens, 1, 2, Access::Input, "prior_tokens");
            addIntInput(a.candidates, rows, "candidates");
            add(a.outputTokens, 1, 2, Access::Output, "output_tokens");
        }
    }, params_.operation);
    validateAliases();
    kernels_ = llaminar::v2::kernels::KernelFactory::createQwen4Exp(params_.device_id);
}

void Qwen4ExpPrimitiveStage::validateAliases() const
{
    for (const auto &output : operands_)
        if (output.access == Access::Output)
            for (const auto &input : operands_)
                if (input.access == Access::Input &&
                    (input.tensor == output.tensor || input.buffer == output.buffer))
                {
                    const bool overlaps = output.byteOffset < input.byteOffset + input.bytes() &&
                                          input.byteOffset < output.byteOffset + output.bytes();
                    if (!overlaps) continue;
                    const bool exact = input.tensor == output.tensor && input.byteOffset == output.byteOffset &&
                                       input.bytes() == output.bytes();
                    const bool permitted = input.alias == AliasPolicy::ExactOutputAliasAllowed;
                    require(exact && permitted, "Qwen4Exp stage forbids overlapping output and immutable input views");
                }    for (const auto &output : operands_)
        if (output.access == Access::Output)
            for (const auto &weight : weights_)
                require(output.tensor != weight.tensor,
                        "Qwen4Exp stage output must never overwrite an immutable model weight");

}

template <typename T, TensorType Type>
T *Qwen4ExpPrimitiveStage::pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const
{
    void *base = nullptr;
    if (device().is_gpu()) base = binding.tensor->gpu_data_ptr();
    else if (access == Access::Input) base = const_cast<void *>(binding.tensor->raw_data());
    else base = binding.tensor->raw_mutable_data();
    if (!base) throw std::logic_error("Qwen4Exp stage requires already-bound authoritative operand storage");
    auto *result = static_cast<std::byte *>(base) + binding.byteOffset;
    if (reinterpret_cast<std::uintptr_t>(result) % alignof(T) != 0)
        throw std::logic_error("Qwen4Exp stage prepared operand address violates native scalar alignment");
    return reinterpret_cast<T *>(result);
}

qwen4exp::RowExtent Qwen4ExpPrimitiveStage::rowExtent() const
{
    return {params_.rows.capacity, params_.rows.liveRows ? intInput(*params_.rows.liveRows) : nullptr};
}

qwen4exp::AcceptedPrefix Qwen4ExpPrimitiveStage::acceptedPrefix(const Qwen4ExpStageAcceptance &acceptance) const
{
    return {rowExtent(), acceptance.acceptedRows ? intInput(*acceptance.acceptedRows) : nullptr,
            acceptance.fixedAcceptedRows};
}

bool Qwen4ExpPrimitiveStage::execute(IDeviceContext *context)
{
    if (!context || context->deviceId() != device())
        throw std::invalid_argument("Qwen4Exp stage requires the matching participant execution context");
    void *stream = nullptr;
    if (device().is_gpu())
    {
        const auto execution = gpuExecution();
        stream = gpuStream();
        for (const auto &operand : operands_)
            if (operand.access == Access::Input) execution.requirePreparedInput(operand.tensor);
            else execution.requirePreparedOutput(operand.tensor);
        for (const auto &weight : weights_)
            execution.requirePreparedInput(const_cast<TensorBase *>(weight.tensor));
    }
    // Native generation inputs remain owned by their producer. Validate their
    // final bound addresses without reading device values or manufacturing an
    // activation tensor/coherence authority for session memory.
    for (const auto &input : native_inputs_)
        for (const auto &output : operands_)
            if (output.access == Access::Output)
            {
                const auto begin = reinterpret_cast<std::uintptr_t>(input.data);
                const auto out = reinterpret_cast<std::uintptr_t>(device().is_gpu()
                    ? output.tensor->gpu_data_ptr() : output.tensor->raw_data()) + output.byteOffset;
                require(begin + input.capacity * sizeof(std::int32_t) <= out ||
                        out + output.bytes() <= begin,
                        "Qwen4Exp native controller/token inputs must not alias stage outputs");
            }
    const auto f = [&](Qwen4ExpFloatBinding binding) { return pointer<const float>(binding, Access::Input); };
    const auto out = [&](Qwen4ExpFloatBinding binding) { return pointer<float>(binding, Access::Output); };
    const auto i = [&](Qwen4ExpIntBinding binding) { return pointer<const std::int32_t>(binding, Access::Input); };
    const auto rows = rowExtent();
    std::visit(Overload{
        [&](const BranchRMSNorm &a) {
            const auto *input = f(a.input);
            if (a.diagnostic_input_publication == DiagnosticInputPublication::MTPTerminalHidden)
            {
                const auto begin = reinterpret_cast<std::uintptr_t>(input);
                const auto end = reinterpret_cast<std::uintptr_t>(pointer<const float>(a.output, Access::Input));
                const auto bytes = static_cast<std::size_t>(params_.rows.capacity) * 4 * a.hidden * sizeof(float);
                require(begin <= std::numeric_limits<std::uintptr_t>::max() - bytes &&
                        end <= std::numeric_limits<std::uintptr_t>::max() - bytes &&
                        (begin + bytes <= end || end + bytes <= begin),
                        "QFN MTP diagnostic input must retain disjoint original physical bytes");
            }
            kernels_->branchRMSNorm({rows,a.hidden,a.epsilon,input,weightPointer(a.gain),out(a.output)},stream);
        },
        [&](const ResidualExpand &a) { kernels_->residualExpand({rows,a.hidden,f(a.input),out(a.output)},stream); },
        [&](const ResidualDown &a) { kernels_->residualDown({rows,a.lowRank,f(a.input),out(a.output)},stream); },
        [&](const ResidualRead &a) { kernels_->residualRead({rows,a.hidden,f(a.normalized),f(a.gateLogits),out(a.output)},stream); },
        [&](const ResidualWrite &a) { kernels_->residualWrite({rows,a.hidden,f(a.residual),f(a.block),f(a.gateLogits),out(a.output)},stream); },
        [&](const NgramHash &a) { kernels_->ngramHash({rows,intInput(a.tokens),i(a.priorTokens),pointer<std::int64_t>(a.outputWords,Access::Output),a.policy},stream); },
        [&](const PLEGate &a) { kernels_->pleGate({rows,a.hidden,f(a.normalizedKey),f(a.normalizedQuery),f(a.value),out(a.gated)},stream); },
        [&](const PLEConv &a) { kernels_->pleConv({rows,a.channels,f(a.normalized),f(a.priorHistory),weightPointer(a.weights),f(a.gated),f(a.residual),out(a.output)},stream); },
        [&](const CommitConvHistory &a) { kernels_->commitConvHistory({acceptedPrefix(a.acceptance),a.channels,f(a.priorHistory),f(a.candidates),out(a.outputHistory)},stream); },
        [&](const CommitTokenHistory &a) { kernels_->commitTokenHistory({acceptedPrefix(a.acceptance),i(a.priorTokens),intInput(a.candidates),pointer<std::int32_t>(a.outputTokens,Access::Output)},stream); }
    }, params_.operation);
    // The executor publishes every declared output using bufferContract and
    // the exact stage stream; the arithmetic stage does not duplicate events.
    return true;
}

uint64_t Qwen4ExpPrimitiveStage::graphCaptureVariantSignature() const
{
    const auto *norm = std::get_if<BranchRMSNorm>(&params_.operation);
    if (!norm) return 0;
    return (norm->diagnostic_input_publication == DiagnosticInputPublication::MTPTerminalHidden ? uint64_t{1} : 0) |
        (norm->input_role == BranchNormInputRole::PredictorHidden ? uint64_t{2} : 0);
}

bool Qwen4ExpPrimitiveStage::allowsZeroOutput() const
{
    const auto *norm = std::get_if<BranchRMSNorm>(&params_.operation);
    return norm && norm->input_role == BranchNormInputRole::PredictorHidden;
}

ComputeStageType Qwen4ExpPrimitiveStage::type() const
{
    return std::visit(Overload{
        [](const BranchRMSNorm &) { return ComputeStageType::QWEN4EXP_BRANCH_RMS_NORM; },
        [](const ResidualExpand &) { return ComputeStageType::QWEN4EXP_RESIDUAL_EXPAND; },
        [](const ResidualDown &) { return ComputeStageType::QWEN4EXP_RESIDUAL_DOWN; },
        [](const ResidualRead &) { return ComputeStageType::QWEN4EXP_RESIDUAL_READ; },
        [](const ResidualWrite &) { return ComputeStageType::QWEN4EXP_RESIDUAL_WRITE; },
        [](const NgramHash &) { return ComputeStageType::QWEN4EXP_NGRAM_HASH; },
        [](const PLEGate &) { return ComputeStageType::QWEN4EXP_PLE_GATE; },
        [](const PLEConv &) { return ComputeStageType::QWEN4EXP_PLE_CONV; },
        [](const CommitConvHistory &) { return ComputeStageType::QWEN4EXP_COMMIT_CONV_HISTORY; },
        [](const CommitTokenHistory &) { return ComputeStageType::QWEN4EXP_COMMIT_TOKEN_HISTORY; }
    }, params_.operation);
}

std::size_t Qwen4ExpPrimitiveStage::estimatedFlops() const
{
    const auto rows = static_cast<std::size_t>(params_.rows.capacity);
    return std::visit(Overload{
        [=](const BranchRMSNorm &a) { return rows*4*a.hidden*5; },
        [=](const ResidualExpand &) { return std::size_t{0}; },
        [=](const ResidualDown &a) { return rows*a.lowRank*6; },
        [=](const ResidualRead &a) { return rows*a.hidden*4*7; },
        [=](const ResidualWrite &a) { return rows*4*(a.hidden*2+6); },
        [=](const NgramHash &) { return std::size_t{0}; },
        [=](const PLEGate &a) { return rows*4*(a.hidden*3+8); },
        [=](const PLEConv &a) { return rows*a.channels*15; },
        [=](const CommitConvHistory &) { return std::size_t{0}; },
        [=](const CommitTokenHistory &) { return std::size_t{0}; }
    }, params_.operation);
}

std::size_t Qwen4ExpPrimitiveStage::estimatedMemoryBytes() const
{
    std::size_t bytes = 0;
    for (const auto &operand : operands_) bytes += operand.bytes();
    for (const auto &weight : weights_) bytes += weight.bytes();
    for (const auto &input : native_inputs_) bytes += input.capacity * sizeof(std::int32_t);
    return bytes;
}

bool Qwen4ExpPrimitiveStage::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cpu() && backend == ComputeBackendType::CPU) ||
           (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
           (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
}

StageBufferContract Qwen4ExpPrimitiveStage::bufferContract() const
{
    StageBufferContract contract;
    for (const auto &operand : operands_)
    {
        const bool read = std::any_of(operands_.begin(), operands_.end(), [&](const auto &other) {
            return other.buffer == operand.buffer && other.access == Access::Input;
        });
        const bool write = std::any_of(operands_.begin(), operands_.end(), [&](const auto &other) {
            return other.buffer == operand.buffer && other.access == Access::Output;
        });
        const auto bound = [&](const auto &bindings) {
            return std::any_of(bindings.begin(), bindings.end(), [&](const auto &binding) { return binding.id == operand.buffer; });
        };
        const char *dtype = tensorTypeName(operand.dtype);
        if (read && write) { if (!bound(contract.inouts)) contract.addPreallocatedInOut(operand.buffer,dtype); }
        else if (read) { if (!bound(contract.inputs)) contract.addInput(operand.buffer,dtype); }
        // A borrowed arena declaration reserves bytes, but does not promise an
        // existing native address. Cold preparation owns write-only results.
        else { if (!bound(contract.outputs)) contract.addOutput(operand.buffer,dtype); }
    }
    for (const auto &weight : weights_) contract.addWeight(const_cast<TensorBase *>(weight.tensor));
    return contract;
}

StageDumpInfo Qwen4ExpPrimitiveStage::buildDumpInfoImpl() const
{
    StageDumpInfo dump;
    const auto *norm = std::get_if<BranchRMSNorm>(&params_.operation);
    for (const auto &operand : operands_)
    {
        // Diagnostic publication retains the exact tensor and byte subview;
        // building descriptors never materializes host copies of GPU state.
        const auto *base = static_cast<const std::byte *>(operand.tensor->raw_data());
        const void *data = base ? base+operand.byteOffset : nullptr;
        if (operand.access == Access::Input)
        {
            dump.inputs.push_back({operand.name,data,operand.rows,operand.columns,tensorTypeName(operand.dtype),4,
                                   operand.bytes(),operand.tensor,operand.byteOffset});
            if (norm && norm->diagnostic_input_publication == DiagnosticInputPublication::MTPTerminalHidden &&
                std::string_view(operand.name) == "input")
            {
                // Tensor-backed readonly view, on the same snapshot event as
                // the norm output. It creates no graph write/coherence owner.
                dump.outputs.push_back({"mtp_terminal_hidden_input", data, operand.rows, operand.columns,
                    "FP32", sizeof(float), operand.bytes(), operand.tensor, {}, operand.byteOffset});
            }
        }
        else
        {
            const bool hash_ids = std::holds_alternative<NgramHash>(params_.operation);
            // Hash backing is INT32 [M,32], but its mathematical publication is
            // INT64 [M,16]. Keep the exact arena byte subview and total extent.
            dump.outputs.push_back({operand.name, data, operand.rows,
                hash_ids ? operand.columns / 2 : operand.columns,
                hash_ids ? "INT64" : tensorTypeName(operand.dtype), hash_ids ? 8u : 4u,
                operand.bytes(), operand.tensor, {}, operand.byteOffset});
        }
    }
    for (const auto &weight : weights_)
        dump.addWeight(weight.name,weight.tensor,weight.rows,weight.columns,"FP32");
    dump.addScalarInt("admitted_rows",params_.rows.capacity);
    dump.addScalarBool("device_counted_rows",params_.rows.liveRows.has_value());
    return dump;
}
} // namespace llaminar2
