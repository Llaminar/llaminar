/**
 * @file DeepSeekV4PrimitiveStage.cpp
 * @brief Bind declarative DeepSeek-V4 variants to prepared participant-local storage.
 *
 * Constructor-created operand descriptors are the single extent authority for
 * type/capacity checks, buffer contracts, estimates and dumps (the Qwen4Exp
 * primitive-stage discipline). GPU execution never reads a host count or
 * repairs coherence; persistent state banks are ordinary admitted arena views.
 */
#include "DeepSeekV4PrimitiveStage.h"
#include "../../../../kernels/KernelFactory.h"
#include "../../../../tensors/Tensors.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
template <class... Functions> struct Overload : Functions... { using Functions::operator()...; };
template <class... Functions> Overload(Functions...) -> Overload<Functions...>;

void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(std::string("DeepSeekV4 stage: ") + message);
}

deepseek4::WeightFormat formatOf(const TensorBase &tensor)
{
    switch (tensor.native_type())
    {
    case TensorType::FP32: return deepseek4::WeightFormat::FP32;
    case TensorType::BF16: return deepseek4::WeightFormat::BF16;
    case TensorType::Q8_0: return deepseek4::WeightFormat::Q8_0;
    default: throw std::invalid_argument("DeepSeekV4 grouped projection supports FP32, BF16 and Q8_0 weights");
    }
}
} // namespace

template <TensorType Type>
void DeepSeekV4PrimitiveStage::add(Qwen4ExpArenaBinding<Type> binding, std::size_t rows, std::size_t columns,
                                   Access access, const char *name, AliasPolicy alias)
{
    static_assert(Type == TensorType::FP32 || Type == TensorType::INT32);
    require(binding.tensor && binding.tensor->native_type() == Type, "arena operand has the wrong native scalar type");
    require(binding.buffer < BufferId::_COUNT, "arena operand requires a canonical BufferId");
    require(rows > 0 && columns > 0 && rows <= std::numeric_limits<std::size_t>::max() / columns / 4,
            "operand geometry is empty or overflows bytes");
    const auto bytes = rows * columns * 4;
    require(binding.byteOffset % 4 == 0 && binding.byteOffset <= binding.tensor->size_bytes() &&
                bytes <= binding.tensor->size_bytes() - binding.byteOffset,
            "arena view is misaligned or exceeds admitted tensor capacity");
    operands_.push_back({binding.tensor, binding.buffer, Type, binding.byteOffset, rows, columns, access, name, alias});
}

void DeepSeekV4PrimitiveStage::addIntInput(const Qwen4ExpIntInput &binding, std::size_t count, const char *name)
{
    std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { add(value, count, 1, Access::Input, name); },
        [&](Qwen4ExpNativeIntInput value) {
            require(value.owner == device() && value.data && count > 0 && count <= value.capacity,
                    "native INT32 input requires the matching owner and admitted extent");
            value.capacity = count;
            native_inputs_.push_back(value);
        }}, binding.value);
}

const std::int32_t *DeepSeekV4PrimitiveStage::intInput(const Qwen4ExpIntInput &binding) const
{
    return std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { return pointer<const std::int32_t>(value, Access::Input); },
        [](Qwen4ExpNativeIntInput value) { return value.data; }}, binding.value);
}

void DeepSeekV4PrimitiveStage::addFP32Weight(Qwen4ExpImmutableWeightBinding binding, std::size_t elements,
                                             const char *name)
{
    require(binding.tensor && binding.tensor->native_type() == TensorType::FP32,
            "immutable operator weight must use native FP32 storage");
    require(elements > 0 && elements * 4 <= binding.tensor->size_bytes(), "immutable weight is smaller than its geometry");
    weights_.push_back({binding.tensor, name});
}

void DeepSeekV4PrimitiveStage::addNativeWeight(DeepSeekV4NativeWeightBinding binding, const char *name)
{
    require(binding.tensor != nullptr, "native weight binding is null");
    weights_.push_back({binding.tensor, name});
}

const void *DeepSeekV4PrimitiveStage::weightPointer(const TensorBase *tensor) const
{
    const void *base = device().is_gpu() ? tensor->gpu_data_ptr() : tensor->raw_data();
    if (!base) throw std::logic_error("DeepSeekV4 stage immutable weight is not bound on its participant");
    return base;
}

DeepSeekV4PrimitiveStage::DeepSeekV4PrimitiveStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
    require(params_.device_id.is_valid(), "requires an explicit valid participant device");
    if (params_.device_id.is_rocm())
        throw std::invalid_argument(deepseek4::kROCmUnimplemented);
    require(params_.rows.capacity > 0, "requires a positive admitted row capacity");
    const auto rows = static_cast<std::size_t>(params_.rows.capacity);
    if (params_.rows.liveRows) addIntInput(*params_.rows.liveRows, 1, "live_rows");
    const auto sz = [](int v) {
        require(v > 0, "geometry must be positive");
        return static_cast<std::size_t>(v);
    };
    std::visit(Overload{
        [&](const HCRead &a) {
            const std::size_t mix = a.head ? sz(a.hc) : sz((2 + a.hc) * a.hc);
            require(a.hc <= 16 && a.sinkhornIters > 0, "hyper-connection count must be in [1,16]");
            add(a.trunk, rows, sz(a.hc) * sz(a.hidden), Access::Input, "trunk");
            add(a.mixes, rows, mix, Access::Input, "mixes");
            addFP32Weight(a.scale, a.head ? 1 : 3, "scale");
            addFP32Weight(a.base, mix, "base");
            if (a.gain) addFP32Weight(*a.gain, sz(a.hidden), "gain");
            if (a.mixed) add(*a.mixed, rows, sz(a.hidden), Access::Output, "mixed");
            add(a.output, rows, sz(a.hidden), Access::Output, "output");
            require(a.head || (a.post && a.comb), "block hyper-connection read requires post and comb outputs");
            if (!a.head)
            {
                add(*a.post, rows, sz(a.hc), Access::Output, "post");
                add(*a.comb, rows, sz(a.hc * a.hc), Access::Output, "comb");
            }
        },
        [&](const HCWrite &a) {
            const auto w = sz(a.hc) * sz(a.hidden);
            add(a.block, rows, sz(a.hidden), Access::Input, "block");
            add(a.post, rows, sz(a.hc), Access::Input, "post");
            add(a.comb, rows, sz(a.hc * a.hc), Access::Input, "comb");
            add(a.trunk, rows, w, Access::Input, "trunk", AliasPolicy::ExactOutputAliasAllowed);
            add(a.output, rows, w, Access::Output, "output");
        },
        [&](const HCExpand &a) {
            add(a.input, rows, sz(a.hidden), Access::Input, "input");
            add(a.output, rows, sz(a.hc) * sz(a.hidden), Access::Output, "output");
        },
        [&](const HeadNormRope &a) {
            require(a.rope.ropeDim <= a.headDim && a.rope.ropeDim % 2 == 0, "rope width must fit the head");
            require(a.quant == deepseek4::ActivationQuant::None ||
                        (a.headDim <= deepseek4::kMaxHeadDim &&
                         (a.quant == deepseek4::ActivationQuant::FP8NopeBF16Rope
                              ? deepseek4::PackedKVEntryFormat{a.headDim, a.rope.ropeDim}.valid()
                              : deepseek4::PackedIndexerKeyFormat{a.headDim}.valid())),
                    "QAT post-operation geometry is not representable");
            if (a.rope.ropeDim > 0)
            {
                require(a.positions.has_value(), "rotation requires row positions");
                addIntInput(*a.positions, rows, "positions");
            }
            if (a.gain) addFP32Weight(*a.gain, sz(a.headDim), "gain");
            add(a.data, rows, sz(a.heads) * sz(a.headDim), Access::Input, "input", AliasPolicy::ExactOutputAliasAllowed);
            add(a.data, rows, sz(a.heads) * sz(a.headDim), Access::Output, "output");
        },
        [&](const CompressorStore &a) {
            const auto w = sz(a.geometry.projectionWidth());
            addIntInput(a.positions, rows, "positions");
            add(a.kv, rows, w, Access::Input, "kv");
            add(a.score, rows, w, Access::Input, "score");
            addFP32Weight(a.ape, sz(a.geometry.ratio) * w, "ape");
            add(a.stateKV, sz(a.geometry.stateRows()), w, Access::Output, "state_kv");
            add(a.stateScore, sz(a.geometry.stateRows()), w, Access::Output, "state_score");
        },
        [&](const CompressorPool &a) {
            const auto &g = a.geometry;
            const auto w = sz(g.projectionWidth());
            require(g.headDim <= deepseek4::kMaxHeadDim && g.entryBytes() % 4 == 0 &&
                        (g.stream == deepseek4::CompressedStream::Indexer
                             ? deepseek4::PackedIndexerKeyFormat{g.headDim}.valid()
                             : deepseek4::PackedKVEntryFormat{g.headDim, g.ropeDim}.valid() &&
                                   g.ropeDim == a.rope.ropeDim),
                    "compressor record geometry is not representable");
            addIntInput(a.positions, rows, "positions");
            add(a.kv, rows, w, Access::Input, "kv");
            add(a.score, rows, w, Access::Input, "score");
            addFP32Weight(a.ape, sz(g.ratio) * w, "ape");
            add(a.stateKV, sz(g.stateRows()), w, Access::Input, "state_kv");
            add(a.stateScore, sz(g.stateRows()), w, Access::Input, "state_score");
            addFP32Weight(a.gain, sz(g.headDim), "gain");
            add(a.pool, sz(g.poolRows), sz(g.entryBytes() / 4), Access::Output, "pool");
        },
        [&](const RawStore &a) {
            require(a.format.valid() && a.format.headDim <= deepseek4::kMaxHeadDim, "raw record geometry is invalid");
            addIntInput(a.positions, rows, "positions");
            add(a.kv, rows, sz(a.format.headDim), Access::Input, "kv");
            add(a.ring, sz(a.window), sz(a.format.entryWords()), Access::Output, "ring");
        },
        [&](const IndexerTopK &a) {
            require(a.format.valid(), "indexer key geometry is invalid");
            addIntInput(a.positions, rows, "positions");
            add(a.query, rows, sz(a.heads) * sz(a.format.headDim), Access::Input, "query");
            add(a.weights, rows, sz(a.heads), Access::Input, "weights");
            add(a.pool, sz(a.poolRows), sz(a.format.entryWords()), Access::Input, "pool");
            require(a.scores.has_value() || !device().is_gpu(), "GPU indexer selection requires score scratch");
            if (a.scores) add(*a.scores, sz(a.scoreRows), sz(a.poolRows), Access::Output, "scores");
            add(a.selected, rows, sz(a.topK), Access::Output, "selected");
        },
        [&](const SparseAttention &a) {
            require(a.format.valid() && a.format.headDim <= deepseek4::kMaxHeadDim &&
                        a.format.ropeDim == a.rope.ropeDim, "attention record geometry is invalid");
            const auto w = sz(a.heads) * sz(a.format.headDim);
            addIntInput(a.positions, rows, "positions");
            add(a.query, rows, w, Access::Input, "query");
            add(a.chunkKV, rows, sz(a.format.headDim), Access::Input, "chunk_kv");
            add(a.rawRing, sz(a.window), sz(a.format.entryWords()), Access::Input, "raw_ring");
            if (a.access != deepseek4::CompressedAccess::None)
            {
                require(a.pool.has_value() && a.ratio > 0, "compressed access requires its pool");
                add(*a.pool, sz(a.poolRows), sz(a.format.entryWords()), Access::Input, "pool");
            }
            if (a.access == deepseek4::CompressedAccess::Selected)
            {
                require(a.selected.has_value(), "selected access requires the indexer output");
                add(*a.selected, rows, sz(a.topK), Access::Input, "selected");
            }
            addFP32Weight(a.sinks, sz(a.heads), "sinks");
            add(a.output, rows, w, Access::Output, "output");
        },
        [&](const AdvanceSequenceFrontier &a) {
            require(device().is_gpu(), "sequence frontier publication is a GPU cache operation");
            require(a.cache != nullptr && a.capturedRows > 0, "sequence frontier requires its cache and rows");
            addIntInput(a.positions, rows, "positions");
        },
        [&](const GroupedProjection &a) {
            (void)formatOf(*a.weight.tensor);
            addNativeWeight(a.weight, "weight");
            add(a.input, rows, sz(a.groups) * sz(a.groupWidth), Access::Input, "input");
            add(a.output, rows, sz(a.groups) * sz(a.rank), Access::Output, "output");
        },
        [&](const Route &a) {
            add(a.logits, rows, sz(a.experts), Access::Input, "logits");
            require(a.hashTable.has_value() == a.tokens.has_value(), "hash routing requires both table and tokens");
            require(a.hashTable.has_value() != a.bias.has_value(), "exactly one of bias (score) or hash table");
            if (a.bias) addFP32Weight(*a.bias, sz(a.experts), "bias");
            if (a.hashTable)
            {
                require(a.hashTable->tensor->native_type() == TensorType::INT32, "hash table must be INT32");
                addNativeWeight(*a.hashTable, "hash_table");
                addIntInput(*a.tokens, rows, "tokens");
            }
            add(a.ids, rows, sz(a.topK), Access::Output, "ids");
            add(a.weights, rows, sz(a.topK), Access::Output, "weights");
        }}, params_.operation);
    validateAliases();
    kernels_ = llaminar::v2::kernels::KernelFactory::createDeepSeekV4(params_.device_id);
}

void DeepSeekV4PrimitiveStage::validateAliases() const
{
    for (const auto &output : operands_)
        if (output.access == Access::Output)
            for (const auto &input : operands_)
                if (input.access == Access::Input && input.tensor == output.tensor)
                {
                    const bool overlaps = output.byteOffset < input.byteOffset + input.bytes() &&
                                          input.byteOffset < output.byteOffset + output.bytes();
                    if (!overlaps) continue;
                    const bool exact = input.byteOffset == output.byteOffset && input.bytes() == output.bytes();
                    require(exact && input.alias == AliasPolicy::ExactOutputAliasAllowed,
                            "forbids overlapping output and immutable input views");
                }
}

template <typename T, TensorType Type>
T *DeepSeekV4PrimitiveStage::pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const
{
    void *base = nullptr;
    if (device().is_gpu()) base = binding.tensor->gpu_data_ptr();
    else if (access == Access::Input) base = const_cast<void *>(binding.tensor->raw_data());
    else base = binding.tensor->raw_mutable_data();
    if (!base) throw std::logic_error("DeepSeekV4 stage requires already-bound authoritative operand storage");
    return reinterpret_cast<T *>(static_cast<std::byte *>(base) + binding.byteOffset);
}

deepseek4::RowExtent DeepSeekV4PrimitiveStage::rowExtent() const
{
    return {params_.rows.capacity, params_.rows.liveRows ? intInput(*params_.rows.liveRows) : nullptr};
}

bool DeepSeekV4PrimitiveStage::execute(IDeviceContext *context)
{
    if (!context || context->deviceId() != device())
        throw std::invalid_argument("DeepSeekV4 stage requires the matching participant execution context");
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
    const auto f = [&](Qwen4ExpFloatBinding b) { return pointer<const float>(b, Access::Input); };
    const auto out = [&](Qwen4ExpFloatBinding b) { return pointer<float>(b, Access::Output); };
    const auto w32 = [&](Qwen4ExpImmutableWeightBinding b) { return static_cast<const float *>(weightPointer(b.tensor)); };
    // Packed records travel in INT32 arena words; kernels address their bytes.
    const auto bytesIn = [&](Qwen4ExpIntBinding b) {
        return reinterpret_cast<const std::uint8_t *>(pointer<const std::int32_t>(b, Access::Input));
    };
    const auto bytesOut = [&](Qwen4ExpIntBinding b) {
        return reinterpret_cast<std::uint8_t *>(pointer<std::int32_t>(b, Access::Output));
    };
    const auto rows = rowExtent();
    std::visit(Overload{
        [&](const HCRead &a) {
            kernels_->hcRead({rows, a.hc, a.hidden, a.head, a.normEps, a.hcEps, a.sinkhornIters, f(a.trunk), f(a.mixes),
                              w32(a.scale), w32(a.base), a.gain ? w32(*a.gain) : nullptr, a.outputEps,
                              a.mixed ? out(*a.mixed) : nullptr, out(a.output), a.post ? out(*a.post) : nullptr,
                              a.comb ? out(*a.comb) : nullptr}, stream);
        },
        [&](const HCWrite &a) {
            kernels_->hcWrite({rows, a.hc, a.hidden, f(a.block), f(a.post), f(a.comb), f(a.trunk), out(a.output)}, stream);
        },
        [&](const HCExpand &a) { kernels_->hcExpand({rows, a.hc, a.hidden, f(a.input), out(a.output)}, stream); },
        [&](const HeadNormRope &a) {
            kernels_->headNormRope({rows, a.heads, a.headDim, a.normalize, a.eps, a.gain ? w32(*a.gain) : nullptr,
                                    a.rope, a.inverse, a.quant, a.positions ? intInput(*a.positions) : nullptr,
                                    out(a.data)}, stream);
        },
        [&](const CompressorStore &a) {
            kernels_->compressorStore({rows, a.geometry, intInput(a.positions), f(a.kv), f(a.score), w32(a.ape),
                                       out(a.stateKV), out(a.stateScore)}, stream);
        },
        [&](const CompressorPool &a) {
            kernels_->compressorPool({rows, a.geometry, intInput(a.positions), f(a.kv), f(a.score), w32(a.ape),
                                      f(a.stateKV), f(a.stateScore), w32(a.gain), a.eps, a.rope,
                                      bytesOut(a.pool)}, stream);
        },
        [&](const RawStore &a) {
            kernels_->rawStore({rows, a.format, a.window, intInput(a.positions), f(a.kv), bytesOut(a.ring)}, stream);
        },
        [&](const IndexerTopK &a) {
            kernels_->indexerTopK({rows, a.heads, a.format, a.ratio, a.topK, a.poolRows, a.scoreRows,
                                   intInput(a.positions), f(a.query), f(a.weights), a.weightScale, bytesIn(a.pool),
                                   a.scores ? out(*a.scores) : nullptr,
                                   pointer<std::int32_t>(a.selected, Access::Output)}, stream);
        },
        [&](const SparseAttention &a) {
            kernels_->sparseAttention({rows, a.heads, a.format, a.window, a.ratio, a.topK, a.access, a.scale, a.rope,
                                       intInput(a.positions), f(a.query), f(a.chunkKV), bytesIn(a.rawRing),
                                       a.pool ? bytesIn(*a.pool) : nullptr,
                                       a.selected ? pointer<const std::int32_t>(*a.selected, Access::Input) : nullptr,
                                       w32(a.sinks), out(a.output)}, stream);
        },
        [&](const AdvanceSequenceFrontier &a) {
            // Exactly one device-resident frontier advance per forward, ordered
            // after the terminal layer on this stage's stream; no host count.
            if (!a.cache->advanceRecurrentSequenceState(a.layer, 1, a.requestRows, a.capturedRows, stream))
                throw std::runtime_error("DeepSeekV4 stage: cache rejected the sequence frontier publication");
        },
        [&](const GroupedProjection &a) {
            kernels_->groupedProjection({rows, a.groups, a.rank, a.groupWidth, formatOf(*a.weight.tensor),
                                         weightPointer(a.weight.tensor), f(a.input), out(a.output)}, stream);
        },
        [&](const Route &a) {
            const int vocab = a.hashTable ? static_cast<int>(a.hashTable->tensor->rows()) : 0;
            kernels_->route({rows, a.experts, a.topK, a.routeScale, f(a.logits), a.bias ? w32(*a.bias) : nullptr,
                             a.hashTable ? static_cast<const std::int32_t *>(weightPointer(a.hashTable->tensor)) : nullptr,
                             vocab, a.tokens ? intInput(*a.tokens) : nullptr,
                             pointer<std::int32_t>(a.ids, Access::Output), out(a.weights)}, stream);
        }}, params_.operation);
    return true;
}

bool DeepSeekV4PrimitiveStage::allowsZeroOutput() const
{
    return std::holds_alternative<CompressorStore>(params_.operation) ||
           std::holds_alternative<CompressorPool>(params_.operation) ||
           std::holds_alternative<RawStore>(params_.operation) ||
           std::holds_alternative<IndexerTopK>(params_.operation) ||
           std::holds_alternative<Route>(params_.operation) ||
           std::holds_alternative<AdvanceSequenceFrontier>(params_.operation);
}

ComputeStageType DeepSeekV4PrimitiveStage::type() const
{
    return std::visit(Overload{
        [](const HCRead &) { return ComputeStageType::DEEPSEEK_V4_HC_READ; },
        [](const HCWrite &) { return ComputeStageType::DEEPSEEK_V4_HC_WRITE; },
        [](const HCExpand &) { return ComputeStageType::DEEPSEEK_V4_HC_EXPAND; },
        [](const HeadNormRope &) { return ComputeStageType::DEEPSEEK_V4_HEAD_NORM_ROPE; },
        [](const CompressorStore &) { return ComputeStageType::DEEPSEEK_V4_COMPRESSOR_STORE; },
        [](const CompressorPool &) { return ComputeStageType::DEEPSEEK_V4_COMPRESSOR_POOL; },
        [](const RawStore &) { return ComputeStageType::DEEPSEEK_V4_RAW_STORE; },
        [](const IndexerTopK &) { return ComputeStageType::DEEPSEEK_V4_INDEXER_TOPK; },
        [](const SparseAttention &) { return ComputeStageType::DEEPSEEK_V4_SPARSE_ATTENTION; },
        [](const GroupedProjection &) { return ComputeStageType::DEEPSEEK_V4_GROUPED_PROJECTION; },
        [](const Route &) { return ComputeStageType::DEEPSEEK_V4_ROUTE; },
        [](const AdvanceSequenceFrontier &) { return ComputeStageType::DEEPSEEK_V4_SEQUENCE_FRONTIER; }}, params_.operation);
}

std::size_t DeepSeekV4PrimitiveStage::estimatedFlops() const
{
    const auto rows = static_cast<std::size_t>(params_.rows.capacity);
    return std::visit(Overload{
        [=](const HCRead &a) { return rows * a.hc * a.hidden * 4; },
        [=](const HCWrite &a) { return rows * a.hc * a.hidden * (2 * a.hc + 2); },
        [=](const HCExpand &) { return std::size_t{0}; },
        [=](const HeadNormRope &a) { return rows * a.heads * a.headDim * 4; },
        [=](const CompressorStore &a) { return rows * a.geometry.projectionWidth(); },
        [=](const CompressorPool &a) { return rows * a.geometry.headDim * 8; },
        [=](const RawStore &) { return std::size_t{0}; },
        [=](const IndexerTopK &a) { return rows * a.poolRows * a.heads * a.format.headDim * 2; },
        [=](const SparseAttention &a) { return rows * a.heads * (a.window + a.topK) * a.format.headDim * 4; },
        [=](const GroupedProjection &a) { return rows * a.groups * a.rank * a.groupWidth * 2; },
        [=](const Route &a) { return rows * a.experts * 8; },
        [=](const AdvanceSequenceFrontier &) { return std::size_t{0}; }}, params_.operation);
}

std::size_t DeepSeekV4PrimitiveStage::estimatedMemoryBytes() const
{
    std::size_t bytes = 0;
    for (const auto &operand : operands_) bytes += operand.bytes();
    for (const auto &weight : weights_) bytes += weight.tensor->size_bytes();
    for (const auto &input : native_inputs_) bytes += input.capacity * sizeof(std::int32_t);
    return bytes;
}

bool DeepSeekV4PrimitiveStage::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cpu() && backend == ComputeBackendType::CPU) ||
           (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA);
}

StageBufferContract DeepSeekV4PrimitiveStage::bufferContract() const
{
    StageBufferContract contract;
    for (const auto &operand : operands_)
    {
        const bool read = std::any_of(operands_.begin(), operands_.end(), [&](const auto &o) {
            return o.buffer == operand.buffer && o.access == Access::Input;
        });
        const bool write = std::any_of(operands_.begin(), operands_.end(), [&](const auto &o) {
            return o.buffer == operand.buffer && o.access == Access::Output;
        });
        const auto bound = [&](const auto &bindings) {
            return std::any_of(bindings.begin(), bindings.end(), [&](const auto &b) { return b.id == operand.buffer; });
        };
        const char *dtype = tensorTypeName(operand.dtype);
        if (read && write) { if (!bound(contract.inouts)) contract.addPreallocatedInOut(operand.buffer, dtype); }
        else if (read) { if (!bound(contract.inputs)) contract.addInput(operand.buffer, dtype); }
        else { if (!bound(contract.outputs)) contract.addOutput(operand.buffer, dtype); }
    }
    for (const auto &weight : weights_) contract.addWeight(const_cast<TensorBase *>(weight.tensor));
    return contract;
}

StageDumpInfo DeepSeekV4PrimitiveStage::buildDumpInfoImpl() const
{
    StageDumpInfo dump;
    for (const auto &operand : operands_)
    {
        const auto *base = static_cast<const std::byte *>(operand.tensor->raw_data());
        const void *data = base ? base + operand.byteOffset : nullptr;
        if (operand.access == Access::Input)
            dump.inputs.push_back({operand.name, data, operand.rows, operand.columns, tensorTypeName(operand.dtype), 4,
                                   operand.bytes(), operand.tensor, operand.byteOffset});
        else
            dump.outputs.push_back({operand.name, data, operand.rows, operand.columns, tensorTypeName(operand.dtype), 4,
                                    operand.bytes(), operand.tensor, {}, operand.byteOffset});
    }
    dump.addScalarInt("admitted_rows", params_.rows.capacity);
    dump.addScalarBool("device_counted_rows", params_.rows.liveRows.has_value());
    return dump;
}
} // namespace llaminar2
