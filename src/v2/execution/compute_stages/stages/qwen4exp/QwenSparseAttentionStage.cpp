/**
 * @file QwenSparseAttentionStage.cpp
 * @brief Bind sparse primitives to admitted arena state and native cache authority.
 *
 * Constructor-created views own extent validation and coherence declarations.
 * Execute resolves only already-prepared addresses. Native selected reads use
 * the exact canonical per-row mapping produced earlier on the executor DAG.
 */
#include "QwenSparseAttentionStage.h"
#include "../../../../kernels/KernelFactory.h"
#include "../../../../tensors/Tensors.h"
#include "../../../../transfer/TransferEngine.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include "utils/PerfStatsCollector.h"
#include "snapshots/Qwen4ExpSnapshotGeometry.h"

namespace llaminar2
{
namespace
{
/** @brief Exhaustive closed-variant visitor. */
template<class... F> struct Overload : F... { using F::operator()...; };
template<class... F> Overload(F...) -> Overload<F...>;
/** @brief Fail malformed static bindings before capture records any operation. */
void require(bool valid, const char *message) { if (!valid) throw std::invalid_argument(message); }
/** @brief Publish exact CPU-owned observations after actual operations, without a controller mirror. */
void observeHost(qwen4exp::QwenSparseCounterDelta bytes,DeviceId device) {
    const char* names[]{"verifier_prior_header_bytes","verifier_prior_payload_bytes",
        "verifier_candidate_payload_bytes","verifier_accepted_payload_bytes"};
    const std::uint64_t values[]{bytes.prior_header,bytes.prior_payload,bytes.candidate_payload,bytes.accepted_payload};
    for(int metric=0;metric<4;++metric)if(values[metric])
        PerfStatsCollector::addCounterWithSequence("qsa",names[metric],static_cast<double>(values[metric]),
            {values[metric]},"decode",device.toString(),{{"authority","host_execution_owner"}});
}
/** @brief Checked native FP32/INT32 element extent. */
std::size_t bytes(std::size_t rows, std::size_t width)
{
    require(rows && width && rows <= std::numeric_limits<std::size_t>::max()/width/4,
            "QSA stage extent is empty or overflows bytes");
    return rows*width*4;
}
}

template<TensorType Type>
void QwenSparseAttentionStage::add(Qwen4ExpArenaBinding<Type> binding, std::size_t extent,
    Access access, const char *name, bool exactAlias)
{
    require(binding.tensor && binding.tensor->native_type() == Type && binding.buffer < BufferId::_COUNT,
            "QSA stage requires native typed arena bindings with canonical buffer identity");
    require(extent && extent%4 == 0 && binding.byteOffset%4 == 0 &&
            binding.byteOffset <= binding.tensor->size_bytes() && extent <= binding.tensor->size_bytes()-binding.byteOffset,
            "QSA stage native byte view exceeds admitted capacity or alignment");
    operands_.push_back({binding.tensor,binding.buffer,Type,binding.byteOffset,extent,access,name,exactAlias});
}
void QwenSparseAttentionStage::addIntInput(const Qwen4ExpIntInput &binding,std::size_t count,const char *name)
{
    std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { add(value,bytes(count,1),Access::Input,name); },
        [&](Qwen4ExpNativeIntInput value) {
            require(value.owner==device() && value.data && count>0 && count<=value.capacity &&
                reinterpret_cast<std::uintptr_t>(value.data)%alignof(std::int32_t)==0,
                "QSA native input requires exact owner, aligned address and admitted extent");
            value.capacity=count;nativeInputs_.push_back(value);
        }
    },binding.value);
}
const std::int32_t *QwenSparseAttentionStage::intInput(const Qwen4ExpIntInput &binding) const
{
    return std::visit(Overload{
        [&](Qwen4ExpIntBinding value) { return pointer<const std::int32_t>(value,Access::Input); },
        [](Qwen4ExpNativeIntInput value) { return value.data; }
    },binding.value);
}

void QwenSparseAttentionStage::addWeight(Qwen4ExpImmutableWeightBinding binding, std::size_t elements, const char *name)
{
    require(binding.tensor && binding.tensor->native_type() == TensorType::FP32 && elements &&
            elements <= binding.tensor->size_bytes()/4,"QSA immutable FP32 weight extent exceeds native source");
    weights_.push_back({binding.tensor,elements,name});
}

QwenSparseAttentionStage::QwenSparseAttentionStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
    require(device().is_valid() && params_.rowCapacity > 0 && params_.tokenCapacity > 0,
            "QSA stage requires an explicit participant device and positive admitted geometry");
    const auto m = static_cast<std::size_t>(params_.rowCapacity);
    add(params_.replay,sizeof(qwen4exp::QwenSparseReplay),
        std::holds_alternative<PrepareReplay>(params_.operation) ? Access::Output : Access::Input,"replay");
    std::visit(Overload{
        [&](const PrepareReplay &a) {
            require(a.layer>=0 && a.sequence>=0 && a.sequence<a.cache.request_capacity() &&
                params_.tokenCapacity==a.cache.max_seq_len(),"QSA replay stage requires exact admitted native cache identity");
            if(device().is_gpu())require(a.cache.deviceRingHeadPtr(a.layer,a.sequence) &&
                a.cache.deviceCachedTokenCountPtr(a.layer,a.sequence),"QSA replay stage requires canonical GPU cursor/count pointers");
            if(a.liveRows)addIntInput(*a.liveRows,1,"live_rows");
            std::visit(Overload{
                [&](const Qwen4ExpIntInput &input) { addIntInput(input,m,"position_ids"); },
                [&](PrepareReplay::ContiguousPositions positions) { require(device().is_cpu() && positions.first>=0,
                    "QSA fixed contiguous positions require explicit nonnegative CPU policy"); }
            },a.positionIds);
            require(device().is_gpu()==a.rowParameters.has_value(),"QSA replay row-parameter storage must match the selected backend");
            if(a.rowParameters)add(*a.rowParameters,bytes(m,sizeof(attention::AttentionDeviceParams)/4),Access::Output,"row_parameters");
        },
        [&](const SnapshotPrior& a) {
            if(a.terminalCounters)a.terminalCounters->requireFor(device());
            require(a.traffic.buffer==BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC,
                "QSA prior snapshot requires its exact native traffic role");
            add(a.acceptedPartial,bytes(4,128),Access::Input,"accepted_partial");
            addTransactionPrior(a.output,Access::Output);
            add(a.traffic,sizeof(qwen4exp::QwenSparsePriorTraffic),Access::Output,"prior_traffic");
        },
        [&](const Indexer &a) {
            qwen4exp::validateQwenSparsePoolSelection(a.poolSelection, params_.rowCapacity);
            // The arena/cache contract owns one exact native bank per layer.
            // A larger containing allocation uses an exact byte subview; it
            // cannot silently enlarge the captured diagnostic state payload.
            const auto poolShape=qwen4ExpSparseSnapshotShape(Qwen4ExpSnapshotOutput::PooledKeys,
                params_.rowCapacity,params_.tokenCapacity,qwen4exp::QwenSparseSelected);
            require(a.blockCapacity == poolShape.geometry.rows,
                "QSA indexer pooled-state bank must exactly match admitted cache-context capacity");
            require(a.positionCapacity > 0 &&
                    a.rotaryDim > 0 && a.rotaryDim <= 128 && a.rotaryDim%2 == 0 &&
                    std::isfinite(a.epsilon) && a.epsilon > 0,"QSA indexer has invalid immutable geometry/policy");
            require(a.representation == qwen4exp::QwenSparseScalar::FP32 || a.representation == qwen4exp::QwenSparseScalar::FP16 ||
                    a.representation == qwen4exp::QwenSparseScalar::BF16,"QSA indexer has invalid representation");
            add(a.query,bytes(m,512),Access::Input,"query");
            add(a.rawKey,bytes(m,128),Access::Input,"raw_key");
            if(a.transactionPrior)addTransactionPrior(*a.transactionPrior,Access::Input);
            else add(a.priorPartial,bytes(4,128),Access::Input,"prior_partial");
            add(a.pooledKeys,bytes(a.blockCapacity,128),Access::Input,"pooled_keys",true);
            add(a.pooledKeys,bytes(a.blockCapacity,128),Access::Output,"pooled_keys");
            addWeight(a.queryNorm,128,"query_norm"); addWeight(a.keyNorm,128,"key_norm");
            if(a.rotarySource==qwen4exp::QwenSparseRotarySource::TablePlanes) {
                require(a.cosine && a.sine,"QSA table rotary source requires explicit immutable planes");
                const auto plane=bytes(a.positionCapacity,a.rotaryDim/2)/4;
                addWeight(*a.cosine,plane,"cosine");addWeight(*a.sine,plane,"sine");
            } else require(a.rotarySource==qwen4exp::QwenSparseRotarySource::AnalyticTheta && !a.cosine && !a.sine &&
                std::isfinite(a.ropeTheta) && a.ropeTheta>0,"QSA analytic rotary source requires finite positive theta without table aliases");
            if(a.rotarySource==qwen4exp::QwenSparseRotarySource::AnalyticTheta)
                indexerFrequencies_=qwen4exp::makeQwenSparsePowerFrequencies(a.ropeTheta,a.rotaryDim);
            add(a.preparedQuery,bytes(m,512),Access::Output,"prepared_query");
            add(a.scores,bytes(m,a.blockCapacity),Access::Output,"scores");
            add(a.selected,bytes(m,qwen4exp::QwenSparseSelected),Access::Output,"selected");
            add(a.selectedCount,bytes(m,1),Access::Output,"selected_count");
        },
        [&](const SelectedAttention &a) {
            qwen4exp::validateQwenSparseAttentionArithmetic(a.arithmetic,device().type);
            if(a.arithmetic==qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals)
                require(a.native.cache.k_precision()==ActivationPrecision::FP16 && a.native.cache.v_precision()==ActivationPrecision::FP16 &&
                    (!a.native.keyPolicy.transformsOnRead() || a.keyRotaryDim==64),
                    "QSA fp16_tile requires native FP16 K/V and the explicit 64-coordinate pre-RoPE policy");
            require(a.queryHeads > 0 && a.kvHeads > 0 && a.queryHeadStart >= 0 && a.kvHeadStart >= 0 &&
                a.queryHeadStart+a.queryHeads <= 24 && a.kvHeadStart+a.kvHeads <= 2 &&
                a.queryHeadStart/12 >= a.kvHeadStart && (a.queryHeadStart+a.queryHeads-1)/12 < a.kvHeadStart+a.kvHeads,
                "QSA selected stage violates participant-local GQA ownership");
            require(a.native.layer >= 0 && a.native.sequence >= 0,"QSA selected cache identity is invalid");
            require(device().is_gpu() == a.rowParameters.has_value() && device().is_gpu() == a.gpuPartials.has_value(),
                "QSA GPU stage requires canonical row params/exclusive partials; CPU has no GPU scratch");
            add(a.query,bytes(m,a.queryHeads*256),Access::Input,"query");
            add(a.selected,bytes(m,qwen4exp::QwenSparseSelected),Access::Input,"selected");
            add(a.selectedCount,bytes(m,1),Access::Input,"selected_count");
            add(a.output,bytes(m,a.queryHeads*256),Access::Output,"output");
            if (a.gate) add(*a.gate,bytes(m,a.queryHeads*256),Access::Input,"gate");
            if (a.rowParameters) add(*a.rowParameters,bytes(m,sizeof(attention::AttentionDeviceParams)/4),Access::Input,"row_params");
            if (a.gpuPartials) add(*a.gpuPartials,bytes(m,a.queryHeads*8*sizeof(qwen4exp::QwenSparseAttentionPartial)/4),Access::Output,"partials");
            if (a.observations) {
                require(a.observations->byteOffset%alignof(qwen4exp::QwenSparseAttentionObservation)==0,"QSA observations require native uint64 alignment");
                add(*a.observations,bytes(m,a.queryHeads*sizeof(qwen4exp::QwenSparseAttentionObservation)/4),Access::Output,"observations");
            }
            if (a.native.keyPolicy.transformsOnRead()) {
                require(device().is_gpu() && a.keyRotaryDim > 0 && a.keyRotaryDim <= 256 && a.keyRotaryDim%2 == 0 &&
                    int(256*a.native.keyPolicy.partial_rotary_factor)==a.keyRotaryDim,
                    "QSA pre-RoPE native read requires explicit immutable transform planes");
                if(a.keyRotarySource==qwen4exp::QwenSparseRotarySource::TablePlanes) {
                    require(a.keyCosine && a.keySine && a.keyPositionCapacity>0,"QSA selected-key table source requires exact immutable planes");
                    const auto plane=bytes(a.keyPositionCapacity,a.keyRotaryDim/2)/4;
                    addWeight(*a.keyCosine,plane,"key_cosine");addWeight(*a.keySine,plane,"key_sine");
                } else require(a.keyRotarySource==qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta && !a.keyCosine && !a.keySine &&
                    std::isfinite(a.native.keyPolicy.rope_theta) && a.native.keyPolicy.rope_theta>0,
                    "QSA selected-key analytic source requires finite positive theta without table aliases");
                if(a.keyRotarySource==qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta)
                    keyNegativeLogTheta_=-std::log(a.native.keyPolicy.rope_theta);
            } else require(!a.keyCosine && !a.keySine,"QSA post-RoPE cache must not declare unused transform planes");
        },
        [&](const RetainCandidates &a) {
            if(a.terminalCounters)a.terminalCounters->requireRequest(device(),a.requestIndex);
            require(a.rawKey.buffer==BufferId::QFN_QSA_RAW_KEY &&
                a.retainedRawKey.buffer==BufferId::QFN_QSA_CANDIDATE_RAW_KEYS &&
                a.copiedBytes.buffer==BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,
                "QSA candidate retention requires exact raw, retained and SparseMTP byte-counter roles");
            require(a.copiedBytes.byteOffset%alignof(std::uint64_t)==0,
                "QSA candidate retention byte observation requires UINT64 alignment");
            require(a.requestCount>0 && a.requestIndex>=0 && a.requestIndex<a.requestCount &&
                a.verifierOrigins.validForRequests(a.requestCount),"QSA retained publication has invalid canonical request/origin geometry");
            add(a.priorPartial,bytes(4,128),Access::Input,"accepted_partial_keys");
            addTransactionPrior(a.transactionPrior,Access::Input);
            if(a.verifierOrigins.words())addIntInput(Qwen4ExpNativeIntInput{device(),a.verifierOrigins.words(),
                std::size_t(a.requestCount-1)*a.verifierOrigins.wordStride()+1},
                std::size_t(a.requestCount-1)*a.verifierOrigins.wordStride()+1,"verifier_first_rows");
            add(a.rawKey,bytes(m,128),Access::Input,"raw_key");
            add(a.retainedRawKey,bytes(m,128),Access::Output,"raw_key");
            add(a.copiedBytes,sizeof(std::uint64_t),Access::Output,"copied_candidate_bytes");
        },
        [&](const CommitPartial &a) {
            add(a.priorPartial,bytes(4,128),Access::Input,"prior_partial",true);
            add(a.rawKey,bytes(m,128),Access::Input,"raw_key");
            addIntInput(a.acceptedRows,1,"accepted_rows");
            add(a.outputPartial,bytes(4,128),Access::Output,"output_partial");
        }
    },params_.operation);
    validateAliases();
    kernels_ = llaminar::v2::kernels::KernelFactory::createQwenSparseAttention(device());
    if(const auto *attention=std::get_if<SelectedAttention>(&params_.operation)) {
        // Setup observes the exact policy forwarded below. No replay counter,
        // synchronization, environment read or device-state read occurs here.
        PerfStatsCollector::addCounter("qsa","attention_arithmetic_policy",1.0,"setup",device().to_string(),
            {{"policy",std::string(qwen4exp::qwenSparseAttentionArithmeticName(attention->arithmetic))},
             {"rows",std::to_string(params_.rowCapacity)},{"query_heads",std::to_string(attention->queryHeads)},
             {"kv_heads",std::to_string(attention->kvHeads)},{"query_head_start",std::to_string(attention->queryHeadStart)},
             {"kv_head_start",std::to_string(attention->kvHeadStart)}});
    }
}

void QwenSparseAttentionStage::addTransactionPrior(QwenSparseTransactionPriorBinding binding,Access access)
{
    require(binding.storage.buffer==BufferId::QFN_QSA_TRANSACTION_PRIOR &&
        binding.storage.byteOffset%alignof(qwen4exp::QwenSparseTransactionPrior)==0,
        "QSA immutable transaction prior requires its exact aligned packed role");
    if(access==Access::Input)require(params_.replay.tensor==binding.storage.tensor &&
        params_.replay.buffer==binding.storage.buffer && params_.replay.byteOffset==binding.storage.byteOffset,
        "QSA immutable transaction prior must own the original replay header");
    add(binding.storage,sizeof(qwen4exp::QwenSparseTransactionPrior),access,"transaction_prior");
}
const qwen4exp::QwenSparseTransactionPrior* QwenSparseAttentionStage::transactionPrior(QwenSparseTransactionPriorBinding binding) const
{
    return pointer<const qwen4exp::QwenSparseTransactionPrior>(binding.storage,Access::Input);
}

void QwenSparseAttentionStage::validateAliases() const
{
    for (const auto &out : operands_) if (out.access == Access::Output) {
        for (const auto &in : operands_) if (in.access == Access::Input && (in.tensor==out.tensor || in.buffer==out.buffer)) {
            if (out.offset >= in.offset+in.bytes || in.offset >= out.offset+out.bytes) continue;
            require(in.exactAlias && in.tensor==out.tensor && in.offset==out.offset && in.bytes==out.bytes,
                    "QSA stage forbids overlapping immutable input and output native views");
        }
        for (const auto &other : operands_) if (&out != &other && other.access == Access::Output &&
                (other.tensor==out.tensor || other.buffer==out.buffer))
            require(out.offset >= other.offset+other.bytes || other.offset >= out.offset+out.bytes,
                    "QSA stage requires exclusive nonoverlapping output views");
        for (const auto &weight : weights_)
            require(weight.tensor != out.tensor,"QSA output must not overwrite immutable model bytes");
    }
}

template<typename T, TensorType Type>
T *QwenSparseAttentionStage::pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const
{
    void *base = device().is_gpu() ? binding.tensor->gpu_data_ptr() :
        access == Access::Input ? const_cast<void *>(binding.tensor->raw_data()) : binding.tensor->raw_mutable_data();
    require(base != nullptr,"QSA stage requires already-prepared authoritative native storage");
    auto *result = static_cast<std::byte *>(base)+binding.byteOffset;
    require(reinterpret_cast<std::uintptr_t>(result)%alignof(T)==0,"QSA prepared pointer violates native alignment");
    return reinterpret_cast<T *>(result);
}
const float *QwenSparseAttentionStage::weightPointer(Qwen4ExpImmutableWeightBinding binding) const
{
    const void *base = device().is_gpu() ? binding.tensor->gpu_data_ptr() : binding.tensor->raw_data();
    require(base && reinterpret_cast<std::uintptr_t>(base)%alignof(float)==0,"QSA model weight was not prepared");
    return static_cast<const float *>(base);
}

bool QwenSparseAttentionStage::execute(IDeviceContext *context)
{
    require(context && context->deviceId()==device(),"QSA stage requires its exact participant execution context");
    void *stream = nullptr;
    if (device().is_gpu()) {
        auto execution = gpuExecution(); stream = gpuStream();
        for (const auto &operand : operands_)
            if (operand.access==Access::Input) execution.requirePreparedInput(operand.tensor);
            else execution.requirePreparedOutput(operand.tensor);
        for (const auto &weight : weights_) execution.requirePreparedInput(const_cast<TensorBase *>(weight.tensor));
    }
    for(const auto &input:nativeInputs_)for(const auto &output:operands_)if(output.access==Access::Output) {
        const auto begin=reinterpret_cast<std::uintptr_t>(input.data);
        const auto destination=reinterpret_cast<std::uintptr_t>(device().is_gpu()?output.tensor->gpu_data_ptr():output.tensor->raw_data())+output.offset;
        require(begin+input.capacity*4<=destination || destination+output.bytes<=begin,
            "QSA native request/controller inputs must not alias stage outputs");
    }
    const auto f = [&](Qwen4ExpFloatBinding b) { return pointer<const float>(b,Access::Input); };
    const auto out = [&](Qwen4ExpFloatBinding b) { return pointer<float>(b,Access::Output); };
    const auto i = [&](Qwen4ExpIntBinding b) { return pointer<const std::int32_t>(b,Access::Input); };
    const auto *replay = pointer<const qwen4exp::QwenSparseReplay>(params_.replay,
        std::holds_alternative<PrepareReplay>(params_.operation) ? Access::Output : Access::Input);
    std::visit(Overload{
        [&](const PrepareReplay &a) {
            int cpuCursor=0,cpuCount=0;
            qwen4exp::QwenSparsePreAppendSequenceView sequence;
            sequence.capacity=params_.tokenCapacity;
            if(device().is_gpu()) {
                sequence.cursor=a.cache.deviceRingHeadPtr(a.layer,a.sequence);
                sequence.cached_tokens=a.cache.deviceCachedTokenCountPtr(a.layer,a.sequence);
            } else {
                cpuCursor=a.cache.ring_head(a.layer,a.sequence);cpuCount=a.cache.get_cached_tokens(a.layer,a.sequence);
                sequence.cursor=&cpuCursor;sequence.cached_tokens=&cpuCount;sequence.encoding=qwen4exp::QwenSparseRingCursor::OldestRow;
            }
            const auto *positions=std::get_if<Qwen4ExpIntInput>(&a.positionIds);
            const int firstPosition=positions ? 0 : std::get<PrepareReplay::ContiguousPositions>(a.positionIds).first;
            kernels_->prepareReplay({sequence,a.liveRows ? intInput(*a.liveRows) : nullptr,positions ? intInput(*positions) : nullptr,
                pointer<qwen4exp::QwenSparseReplay>(params_.replay,Access::Output),
                a.rowParameters ? pointer<attention::AttentionDeviceParams>(*a.rowParameters,Access::Output) : nullptr,params_.rowCapacity,firstPosition},stream);
        },
        [&](const SnapshotPrior& a) {
            if(a.terminalCounters)a.terminalCounters->requireFor(device());
            kernels_->snapshotPrior({replay,f(a.acceptedPartial),
                pointer<qwen4exp::QwenSparseTransactionPrior>(a.output.storage,Access::Output),
                pointer<qwen4exp::QwenSparsePriorTraffic>(a.traffic,Access::Output),
                params_.rowCapacity,params_.tokenCapacity,a.terminalCounters ? a.terminalCounters->native() : qwen4exp::QwenSparseWorkCounterRow{}},stream);
            if(device().is_cpu()) {
                const auto* actual=pointer<const qwen4exp::QwenSparsePriorTraffic>(a.traffic,Access::Input);
                observeHost({actual->metadata_bytes,actual->payload_bytes,0,0},device());
            }
        },
        [&](const Indexer &a) {
            kernels_->indexer({.replay=replay,.query=f(a.query),.raw_keys=f(a.rawKey),.partial_keys=a.transactionPrior ? transactionPrior(*a.transactionPrior)->partial : f(a.priorPartial),
                .pooled_keys=out(a.pooledKeys),.query_norm=weightPointer(a.queryNorm),.key_norm=weightPointer(a.keyNorm),
                .cosine=a.cosine ? weightPointer(*a.cosine) : nullptr,.sine=a.sine ? weightPointer(*a.sine) : nullptr,.prepared_query=out(a.preparedQuery),.scores=out(a.scores),
                .selected=pointer<std::int32_t>(a.selected,Access::Output),.selected_count=pointer<std::int32_t>(a.selectedCount,Access::Output),
                .row_capacity=params_.rowCapacity,.token_capacity=params_.tokenCapacity,.block_capacity=a.blockCapacity,
                .position_capacity=a.positionCapacity,.rotary_dim=a.rotaryDim,.norm_epsilon=a.epsilon,.representation=a.representation,.rotary_source=a.rotarySource,.rope_theta=a.ropeTheta,.inverse_frequencies=indexerFrequencies_,
                .transaction_prior=a.transactionPrior ? transactionPrior(*a.transactionPrior) : nullptr,
                .pool_selection=a.poolSelection},stream);
        },
        [&](const SelectedAttention &a) {
            auto kv = a.native.cache.acquireSparseAttentionNativeView(a.native.layer,a.native.sequence,
                a.rowParameters ? pointer<const attention::AttentionDeviceParams>(*a.rowParameters,Access::Input) : nullptr,
                a.native.keyPolicy,stream);
            kv.key_rotary_source=a.keyRotarySource;kv.key_rotary_dim=a.keyRotaryDim;kv.key_negative_log_theta=keyNegativeLogTheta_;
            if (a.keyCosine) {
                kv.key_cosine=weightPointer(*a.keyCosine); kv.key_sine=weightPointer(*a.keySine);
                kv.key_rotary_dim=a.keyRotaryDim; kv.key_position_capacity=a.keyPositionCapacity;
            }
            kernels_->attention({.replay=replay,.query=f(a.query),.kv=kv,.selected=i(a.selected),.selected_count=i(a.selectedCount),
                .gate=a.gate ? f(*a.gate) : nullptr,.output=out(a.output),
                .gpu_partials=a.gpuPartials ? pointer<qwen4exp::QwenSparseAttentionPartial>(*a.gpuPartials,Access::Output) : nullptr,
                .observations=a.observations ? pointer<qwen4exp::QwenSparseAttentionObservation>(*a.observations,Access::Output) : nullptr,
                .row_capacity=params_.rowCapacity,.token_capacity=params_.tokenCapacity,.query_heads=a.queryHeads,.kv_heads=a.kvHeads,
                .query_head_start=a.queryHeadStart,.kv_head_start=a.kvHeadStart,.arithmetic=a.arithmetic},stream);
        },
        [&](const RetainCandidates &a) {
            if(a.terminalCounters)a.terminalCounters->requireRequest(device(),a.requestIndex);
            kernels_->retainCandidates({f(a.rawKey),out(a.retainedRawKey),&replay->live_rows,
                pointer<std::uint64_t>(a.copiedBytes,Access::Output),params_.rowCapacity,
                a.terminalCounters ? a.terminalCounters->native() : qwen4exp::QwenSparseWorkCounterRow{}},stream);
            if(device().is_cpu())observeHost({0,0,*pointer<const std::uint64_t>(a.copiedBytes,Access::Input),0},device());
        },
        [&](const CommitPartial &a) {
            kernels_->commitPartial({replay,f(a.priorPartial),f(a.rawKey),
                AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,params_.rowCapacity,&replay->live_rows),
                    intInput(a.acceptedRows)),out(a.outputPartial)},stream);
        }
    },params_.operation);
    return true;
}

bool QwenSparseAttentionStage::hasVerifierStateCapture() const
{
    return std::holds_alternative<RetainCandidates>(params_.operation);
}
RequestRowOrigins QwenSparseAttentionStage::retainedOrigins() const
{
    const auto &a=std::get<RetainCandidates>(params_.operation);
    return a.verifierOrigins.words() ? RequestRowOrigins::nativeWords(
        a.verifierOrigins.words()+std::size_t(a.requestIndex)*a.verifierOrigins.wordStride()) :
        RequestRowOrigins::fixedStride(a.verifierOrigins.resolveNative(a.requestIndex),1);
}
CPUVerifierStateRestorePlan QwenSparseAttentionStage::planCPUVerifierStateRestoreRow(int row)
{
    if(device().is_gpu() || !hasVerifierStateCapture())return {};
    const auto &a=std::get<RetainCandidates>(params_.operation);
    const auto *replay=pointer<const qwen4exp::QwenSparseReplay>(params_.replay,Access::Input);
    const auto acceptance=AcceptedRequestPrefixes::fromFlatRows(
        DeviceRequestRowRanges::deviceCounted(1,params_.rowCapacity,&replay->live_rows),&row,retainedOrigins());
    const int accepted=acceptance.resolveNative(0);
    if(accepted<0 || replay->prefix_tokens<0 || replay->prefix_tokens>params_.tokenCapacity ||
        accepted>params_.tokenCapacity-replay->prefix_tokens)
        return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
    if(transactionPrior(a.transactionPrior)->row_capacity!=params_.rowCapacity)
        return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
    const auto written=replay->live_rows==0 ? 0u :
        std::size_t((replay->prefix_tokens+accepted)%4)*128*sizeof(float);
    cpuPublicationWrites_[0]={pointer<float>(a.priorPartial,Access::Output),written};
    return {.status=CPUVerifierStateRestorePlanStatus::Ready,
        .kind=CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit,
        .native_destinations=written ? std::span<const CPUVerifierStateRestoreDestination>{cpuPublicationWrites_} :
            std::span<const CPUVerifierStateRestoreDestination>{}};
}
bool QwenSparseAttentionStage::planCPUVerifierStateRestoreRows(const int *rows,int requests,
    std::vector<CPUVerifierStateRestorePlan> &plans)
{
    if(device().is_gpu() || !hasVerifierStateCapture() || !rows)return false;
    const auto &retained=std::get<RetainCandidates>(params_.operation);
    if(requests!=retained.requestCount)return false;
    const auto plan=planCPUVerifierStateRestoreRow(rows[retained.requestIndex]);
    if(!plan.ready())return false;
    plans.push_back(plan);return true;
}
bool QwenSparseAttentionStage::publishRetained(const int *flatRow,void *stream)
{
    if(!hasVerifierStateCapture() || !flatRow || (device().is_gpu() ? !stream : stream!=nullptr))return false;
    const auto &a=std::get<RetainCandidates>(params_.operation);
    if(a.terminalCounters)a.terminalCounters->requireRequest(device(),a.requestIndex);
    std::optional<StageGPUExecution> execution;
    if(device().is_gpu()) {
        // Accepted publication is a separate invocation on the canonical
        // publisher stream, not necessarily the verifier recording stream.
        execution=gpuExecutionForVerifierPublication(stream);stream=execution->nativeStream();
        execution->requirePreparedInput(params_.replay.tensor);
        execution->requirePreparedInput(a.retainedRawKey.tensor);
        execution->requirePreparedOutput(a.priorPartial.tensor);
    }
    const auto *replay=pointer<const qwen4exp::QwenSparseReplay>(params_.replay,Access::Input);
    const auto* original=transactionPrior(a.transactionPrior);
    kernels_->commitPartial({replay,original ? original->partial : pointer<const float>(a.priorPartial,Access::Input),
        pointer<const float>(a.retainedRawKey,Access::Input),
        AcceptedRequestPrefixes::fromFlatRows(DeviceRequestRowRanges::deviceCounted(1,params_.rowCapacity,&replay->live_rows),
            flatRow,retainedOrigins()),pointer<float>(a.priorPartial,Access::Output),original,params_.tokenCapacity,
        a.terminalCounters ? a.terminalCounters->native() : qwen4exp::QwenSparseWorkCounterRow{}},stream);
    if(execution)execution->publish(a.priorPartial.tensor);
    else if(replay->live_rows>0) {
        const auto accepted=AcceptedRequestPrefixes::fromFlatRows(
            DeviceRequestRowRanges::deviceCounted(1,params_.rowCapacity,&replay->live_rows),flatRow,retainedOrigins()).resolveNative(0);
        observeHost({0,0,0,static_cast<std::uint64_t>((replay->prefix_tokens+accepted)%4)*128*4},device());
    }
    return true;
}
bool QwenSparseAttentionStage::restoreVerifierStateCaptureRow(int row,void *stream)
{
    if(device().is_gpu() || !hasVerifierStateCapture())return false;
    return publishRetained(&row,stream);
}
bool QwenSparseAttentionStage::restoreVerifierStateCaptureRows(const int *rows,int requestCount,void *stream)
{
    if(device().is_gpu() || !hasVerifierStateCapture() || !rows)return false;
    const auto &a=std::get<RetainCandidates>(params_.operation);
    if(requestCount!=a.requestCount)return false;
    return publishRetained(rows+a.requestIndex,stream);
}
bool QwenSparseAttentionStage::restoreVerifierStateCaptureRowFromDeviceIndex(const int *row,void *stream)
{
    if(!device().is_gpu() || !hasVerifierStateCapture())return false;
    const auto &a=std::get<RetainCandidates>(params_.operation);
    return a.requestCount==1 && publishRetained(row,stream);
}
bool QwenSparseAttentionStage::restoreVerifierStateCaptureRowsFromDeviceIndices(
    const int *rows,int requestCount,int rowIndexStride,void *stream)
{
    if(!device().is_gpu() || !hasVerifierStateCapture() || !rows || rowIndexStride<=0)return false;
    const auto &a=std::get<RetainCandidates>(params_.operation);
    if(requestCount!=a.requestCount)return false;
    return publishRetained(rows+std::size_t(a.requestIndex)*rowIndexStride,stream);
}

uint64_t QwenSparseAttentionStage::graphCaptureVariantSignature() const
{
    const auto* retained=std::get_if<RetainCandidates>(&params_.operation);
    const auto* indexer=std::get_if<Indexer>(&params_.operation);
    const auto* attention=std::get_if<SelectedAttention>(&params_.operation);
    // Ordinary and verifier indexers both embed a native selector and static
    // arithmetic geometry. Live replay bytes never select or repair that policy.
    if(!retained && !std::holds_alternative<SnapshotPrior>(params_.operation) && !indexer && !attention)return 0;
    uint64_t signature=1469598103934665603ULL;
    const auto mix=[&](uint64_t word){signature^=word;signature*=1099511628211ULL;};
    mix(params_.operation.index());mix(params_.rowCapacity);mix(params_.tokenCapacity);
    if(const auto* snapshot=std::get_if<SnapshotPrior>(&params_.operation))
        mix(snapshot->terminalCounters ? snapshot->terminalCounters->captureIdentity() : 0);
    if(retained)mix(retained->terminalCounters ? retained->terminalCounters->captureIdentity() : 0);
    for(const auto& operand:operands_) {
        mix(reinterpret_cast<std::uintptr_t>(operand.tensor));mix(static_cast<uint64_t>(operand.buffer));
        mix(static_cast<uint64_t>(operand.dtype));mix(operand.offset);mix(operand.bytes);mix(static_cast<uint64_t>(operand.access));
    }
    for(const auto& input:nativeInputs_) {mix(reinterpret_cast<std::uintptr_t>(input.data));mix(input.capacity);}
    if(indexer) {
        mix(static_cast<uint64_t>(device().type));mix(device().ordinal);
        mix(static_cast<uint64_t>(indexer->poolSelection));mix(indexer->transactionPrior.has_value());
        mix(indexer->blockCapacity);mix(indexer->positionCapacity);mix(indexer->rotaryDim);
        mix(std::bit_cast<uint32_t>(indexer->epsilon));mix(std::bit_cast<uint32_t>(indexer->ropeTheta));
        mix(static_cast<uint64_t>(indexer->representation));mix(static_cast<uint64_t>(indexer->rotarySource));
        for(const auto& weight:weights_){mix(reinterpret_cast<std::uintptr_t>(weight.tensor));mix(weight.elements);}
    }
    if(attention) {
        // Native cache allocation generations remain under the canonical cache
        // and graph owner. This seals every immutable selected-stage binding;
        // it does not acquire a GPU view or read mutable ring/request state.
        mix(static_cast<uint64_t>(device().type));mix(device().ordinal);
        mix(static_cast<uint64_t>(attention->arithmetic));
        mix(reinterpret_cast<std::uintptr_t>(&attention->native.cache));
        mix(attention->native.layer);mix(attention->native.sequence);
        mix(static_cast<uint64_t>(attention->native.keyPolicy.encoding));
        mix(std::bit_cast<uint32_t>(attention->native.keyPolicy.rope_theta));
        mix(std::bit_cast<uint32_t>(attention->native.keyPolicy.partial_rotary_factor));
        mix(static_cast<uint64_t>(attention->native.cache.k_precision()));
        mix(static_cast<uint64_t>(attention->native.cache.v_precision()));
        mix(attention->queryHeads);mix(attention->kvHeads);
        mix(attention->queryHeadStart);mix(attention->kvHeadStart);
        mix(attention->keyPositionCapacity);mix(attention->keyRotaryDim);
        mix(static_cast<uint64_t>(attention->keyRotarySource));
        for(const auto& weight:weights_){mix(reinterpret_cast<std::uintptr_t>(weight.tensor));mix(weight.elements);}
    }
    if(retained) {
        mix(retained->requestIndex);mix(retained->requestCount);
        mix(reinterpret_cast<std::uintptr_t>(retained->verifierOrigins.words()));mix(retained->verifierOrigins.wordStride());
        if(!retained->verifierOrigins.words()) {
            mix(retained->verifierOrigins.resolveNative(0));
            mix(retained->verifierOrigins.resolveNative(retained->requestCount-1));
        }
    }
    return signature ? signature :1;
}

ComputeStageType QwenSparseAttentionStage::type() const
{
    return std::visit(Overload{
        [](const PrepareReplay &) { return ComputeStageType::QWEN4EXP_QSA_PREPARE_REPLAY; },
        [](const SnapshotPrior &) { return ComputeStageType::QWEN4EXP_QSA_SNAPSHOT_PRIOR; },
        [](const Indexer &) { return ComputeStageType::QWEN4EXP_QSA_INDEXER; },
        [](const SelectedAttention &) { return ComputeStageType::QWEN4EXP_QSA_ATTENTION; },
        [](const CommitPartial &) { return ComputeStageType::QWEN4EXP_QSA_COMMIT_PARTIAL; },
        [](const RetainCandidates &) { return ComputeStageType::QWEN4EXP_QSA_RETAIN_CANDIDATES; }
    },params_.operation);
}
std::size_t QwenSparseAttentionStage::estimatedFlops() const
{
    const auto m=static_cast<std::size_t>(params_.rowCapacity);
    return std::visit(Overload{
        [](const PrepareReplay &) { return std::size_t{0}; },
        [](const SnapshotPrior &) { return std::size_t{0}; },
        [=](const Indexer &a) { return m*512*std::size_t(a.blockCapacity)*2; },
        [this,m](const SelectedAttention &a) { return m*a.queryHeads*256*std::min(params_.tokenCapacity,2051)*4; },
        [](const CommitPartial &) { return std::size_t{0}; },
        [](const RetainCandidates &) { return std::size_t{0}; }
    },params_.operation);
}
std::size_t QwenSparseAttentionStage::estimatedMemoryBytes() const
{
    std::size_t total=0;
    for (const auto &operand : operands_) total+=operand.bytes;
    for (const auto &weight : weights_) total+=weight.elements*4;
    for (const auto &input : nativeInputs_) total+=input.capacity*4;
    return total;
}
bool QwenSparseAttentionStage::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cpu() && backend==ComputeBackendType::CPU) ||
        (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);
}
StageBufferContract QwenSparseAttentionStage::bufferContract() const
{
    StageBufferContract result;
    for (const auto &operand : operands_) {
        const bool read=std::any_of(operands_.begin(),operands_.end(),[&](const auto &a){return a.buffer==operand.buffer && a.access==Access::Input;});
        const bool write=std::any_of(operands_.begin(),operands_.end(),[&](const auto &a){return a.buffer==operand.buffer && a.access==Access::Output;});
        const auto contains=[&](const auto &views) { return std::any_of(views.begin(),views.end(),[&](const auto &a){return a.id==operand.buffer;}); };
        if (read && write) { if (!contains(result.inouts)) result.addPreallocatedInOut(operand.buffer,tensorTypeName(operand.dtype)); }
        else if (read) { if (!contains(result.inputs)) result.addInput(operand.buffer,tensorTypeName(operand.dtype)); }
        // Write-only scratch is admitted host storage until the executor's cold
        // allocation pass materializes this exact arena owner. Declaring it as
        // preallocated would skip storage for diagnostic-only outputs with no
        // later reader, including prepared_query and retained traffic records.
        else if (!contains(result.outputs)) result.addOutput(operand.buffer,tensorTypeName(operand.dtype));
    }
    for (const auto &weight : weights_) result.addWeight(const_cast<TensorBase *>(weight.tensor));
    return result;
}
bool QwenSparseAttentionStage::allowsZeroOutput() const
{
    // Commit publishes only (prefix + accepted) % 4 raw keys. A complete pool
    // writes no tail; the admitted zero state is valid even for nonzero inputs.
    // Numerical indexer/attention outputs retain the ordinary nonzero check.
    return std::holds_alternative<CommitPartial>(params_.operation);
}
StageDumpInfo QwenSparseAttentionStage::buildDumpInfoImpl() const
{
    StageDumpInfo result;
    for (const auto &a : operands_) {
        // Ranking is unnecessary when every visible block fits the selection.
        // The score bank is private workspace: neither its unwritten short-
        // context bytes nor its causal padding are semantic stage outputs.
        // Keep its arena/coherence contract, but never validate or snapshot it.
        if (std::string_view(a.name) == "scores") continue;
        const auto *base=static_cast<const std::byte *>(a.tensor->raw_data());
        const void *data=base ? base+a.offset : nullptr;
        std::size_t rows=1,columns=a.bytes/4;
        const std::string_view name(a.name);
        // Sequence diagnostics retain the same row axis used by canonical
        // chunk joins; cache banks retain their physical logical-block axis.
        if(name=="query" || name=="raw_key" || name=="prepared_query" || name=="scores" ||
           name=="selected" || name=="selected_count" || name=="output" || name=="gate" ||
           name=="partials" || name=="observations" || name=="row_parameters" || name=="row_params" || name=="position_ids") {
            rows=params_.rowCapacity;columns=a.bytes/(4*rows);
        } else if(name=="pooled_keys") {
            rows=a.bytes/(128*4);columns=128;
        } else if(name=="prior_partial" || name=="output_partial") {
            rows=4;columns=128;
        }
        if (a.access==Access::Input) result.inputs.push_back({a.name,data,rows,columns,tensorTypeName(a.dtype),4,a.bytes,a.tensor,a.offset});
        else result.outputs.push_back({a.name,data,rows,columns,tensorTypeName(a.dtype),4,a.bytes,a.tensor,{},a.offset});
    }
    if(const auto *indexer=std::get_if<Indexer>(&params_.operation)) {
        const SnapshotInt32Binding live{params_.replay.tensor,params_.replay.byteOffset+sizeof(std::int32_t)};
        for(auto &output:result.outputs) {
            const std::string_view name(output.name);
            const auto role = name=="prepared_query" ? Qwen4ExpSnapshotOutput::PreparedQuery :
                name=="pooled_keys" ? Qwen4ExpSnapshotOutput::PooledKeys :
                name=="selected" ? Qwen4ExpSnapshotOutput::SelectedIndices : Qwen4ExpSnapshotOutput::SelectedCount;
            const auto shape = qwen4ExpSparseSnapshotShape(role, params_.rowCapacity,
                params_.tokenCapacity, qwen4exp::QwenSparseSelected);
            requireQwen4ExpSnapshotShape(shape, output.rows, output.cols, output.element_size);
            if(name=="prepared_query" || name=="selected_count") output.semantic_extent=SnapshotTokenRows{live};
            else if(name=="pooled_keys") output.semantic_extent=SnapshotStatePrefixRows{
                {params_.replay.tensor,params_.replay.byteOffset},params_.tokenCapacity,4};
            else if(name=="selected") output.semantic_extent=SnapshotCountedSelectionRows{live,
                {indexer->selectedCount.tensor,indexer->selectedCount.byteOffset}};
        }
    }
    if(std::holds_alternative<RetainCandidates>(params_.operation))
        for(auto &output:result.outputs)if(std::string_view(output.name)=="raw_key")
            output.semantic_extent=SnapshotTokenRows{{params_.replay.tensor,params_.replay.byteOffset+sizeof(std::int32_t)}};
    for (const auto &weight : weights_) result.addWeight(weight.name,weight.tensor,1,weight.elements,"FP32");
    result.addScalarInt("admitted_rows",params_.rowCapacity);
    result.addScalarInt("native_token_capacity",params_.tokenCapacity);
    if(const auto *attention=std::get_if<SelectedAttention>(&params_.operation))
        result.addScalarInt("attention_arithmetic_policy",static_cast<int>(attention->arithmetic));
    return result;
}
} // namespace llaminar2
