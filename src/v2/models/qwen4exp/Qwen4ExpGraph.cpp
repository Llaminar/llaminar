/**
 * @file Qwen4ExpGraph.cpp
 * @brief Bind exact Flash Next model operands to reusable participant-local graph builders.
 *
 * Ordinary core blocks retain H-wide projections and explicit TP collectives.
 * Branch mixers retain the 4H residual; lexical table storage is rank-owned.
 * Sparse indexer pools and lexical histories attach to the canonical KV cache
 * so reset, rollback and prefix archive share one lifecycle and frontier.
 */
#include "Qwen4ExpGraph.h"
#include "Qwen4ExpConfig.h"
#include "Qwen4ExpBufferLayout.h"
#include "Qwen4ExpTrunkBinding.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "../../execution/compute_stages/stages/qwen4exp/Qwen4ExpSparseAttentionBuilder.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../loaders/ModelContext.h"
#include "../../kernels/IHybridKVCache.h"
#include "../../backends/ExplicitGPUStream.h"
#include "../../tensors/Tensors.h"
#include "../../transfer/TransferEngine.h"
#include "../../utils/DebugEnv.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
Qwen4ExpGraph::Qwen4ExpGraph(const GraphConfig &config,std::shared_ptr<IMPIContext> mpi)
    :Qwen35MoEGraph(config,std::move(mpi)),
     sparse_attention_arithmetic_(debugEnv().attention.qfn_sparse_attention)
{
    if(!config_.flash_next)
        throw std::invalid_argument("Flash Next graph requires an authenticated exact GGUF configuration");
}
GraphSchema Qwen4ExpGraph::getSchema() const
{
    auto schema=Qwen35MoEGraph::getSchema();schema.name=architectureName();
    for(int local=0;local<config_.n_layers;++local)
        schema.layer_template_names[local]=config_.layer_types.at(config_.pp_layer_offset+local);
    applyQwen4ExpBufferLayout(schema,config_);return schema;
}
GraphResolverConfig Qwen4ExpGraph::getResolverConfig(int seq_len) const
{
    auto resolver=Qwen35MoEGraph::getResolverConfig(seq_len);
    applyQwen4ExpBufferGeometry(resolver,config_);return resolver;
}
void Qwen4ExpGraph::setArena(BufferArena *arena)
{
    Qwen35MoEGraph::setArena(arena);
    // Arena declaration precedes ordinary weights and lowest-priority table
    // materialization in both runner factories. It must not acquire a runtime
    // source boundary. addModelLayerInput/retainedStorageBoundary authenticate
    // the exact prepared placement and register its original external roots
    // before actual PLE stages are constructed or captured.
    // Setup queries cache state before the first forward invocation publishes
    // ModelBuffers. Bind from the same declaration inventory used by planning
    // and allocation, including exact PP placement omissions.
    for(const auto &entry:qwen4ExpBufferDeclarations())
    {
        buffers_.layer_buffers.extensions.erase(entry.id);
        if(arena)
            if(auto *view=dynamic_cast<TensorBase *>(arena->getTensor(entry.id)))
                buffers_.layer_buffers.extensions.emplace(entry.id,view);
    }
}
void Qwen4ExpGraph::bindStorageMemoryAuthority(std::shared_ptr<PhysicalMemoryAuthority> authority)
{
    if(!authority) throw std::invalid_argument("Flash Next storage requires the participant's admitted physical-memory authority");
    if(memory_ && memory_!=authority)
        throw std::logic_error("Flash Next cannot replace a bound physical-memory authority");
    memory_=std::move(authority);
}
Qwen4ExpGraph::AuxiliaryContent Qwen4ExpGraph::auxiliaryContent() const
{
    const auto &mixers=config_.flash_next->mixers();
    const auto first=config_.pp_layer_offset;
    if(first<0 || config_.n_layers<=0 || std::int64_t(first)+config_.n_layers>std::int64_t(mixers.size()))
        throw std::invalid_argument("Flash Next auxiliary ownership requires an exact admitted model-layer interval");
    const bool sparse=std::any_of(mixers.begin()+first,mixers.begin()+first+config_.n_layers,
        [](auto mixer){return mixer==Qwen4ExpMixer::SparseAttention;});
    const bool ple=ownsPLE();
    return sparse ? (ple?AuxiliaryContent::PLEAndSparse:AuxiliaryContent::SparseOnly) :
        (ple?AuxiliaryContent::PLEOnly:AuxiliaryContent::Absent);
}
void Qwen4ExpGraph::initializeCacheAuxiliaryState(IKVCache &cache,void *publicationStream)
{
    if(auxiliaryContent()!=AuxiliaryContent::Absent) {(void)state(cache,publicationStream);return;}
    // A GDN-only PP participant has no QSA/PLE banks. Its recurrence remains
    // owned by the actual hybrid cache; no dummy extension/checkpoint/event or
    // hidden arena allocation may stand in for the absent model state.
    const auto *hybrid=dynamic_cast<const IHybridKVCache *>(&cache);
    if(!config_.default_device.is_valid() || config_.max_request_count<=0 ||
        config_.max_request_count>cache.request_capacity() || cache.first_layer_index()!=config_.pp_layer_offset ||
        cache.n_layers()!=config_.n_layers || cache.max_seq_len()!=config_.max_seq_len || !hybrid ||
        hybrid->kvLayerCount()!=0 || hybrid->gdnLayerCount()!=config_.n_layers || cache.auxiliaryState())
        throw std::invalid_argument("Flash Next absent auxiliary state requires its exact GDN-only cache interval/request/context owner");
    for(int layer=config_.pp_layer_offset;layer<config_.pp_layer_offset+config_.n_layers;++layer)
        if(!hybrid->isGDNLayer(layer))
            throw std::invalid_argument("Flash Next absent auxiliary scope contains a non-GDN cache layer");
    if(config_.default_device.is_gpu())
    {
        (void)ExplicitGPUStream(publicationStream);
        if(!cache.supportsDeviceResidentSequenceStatePublication())
            throw std::invalid_argument("Flash Next GPU auxiliary absence requires a native device-owned cache frontier");
    }
    else if(publicationStream || cache.supportsDeviceResidentSequenceStatePublication())
        throw std::invalid_argument("Flash Next CPU auxiliary absence requires its host-owned cache and no GPU stream");
}
bool Qwen4ExpGraph::ownsPLE() const
{
    const auto layer=static_cast<int>(config_.flash_next->ngrams().decoder_layer);
    return layer>=config_.pp_layer_offset && layer<config_.pp_layer_offset+config_.n_layers;
}
TensorBase &Qwen4ExpGraph::tensor(BufferId role) const
{
    auto *value=buffers_.layer_buffers.get(buffers_.layer_buffers.idFor(role));
    if(!value)throw std::invalid_argument("Flash Next missing admitted role "+std::string(bufferIdName(role)));
    return *value;
}
Qwen4ExpFloatBinding Qwen4ExpGraph::fp32(BufferId role,std::size_t offset) const
{return {tensor(role),buffers_.layer_buffers.idFor(role),offset};}
Qwen4ExpIntBinding Qwen4ExpGraph::integer(BufferId role,std::size_t offset) const
{return {tensor(role),buffers_.layer_buffers.idFor(role),offset};}
Qwen4ExpPreparedProjection Qwen4ExpGraph::projection(const WeightBinding *weight,DeviceId device) const
{
    if(!weight || !prepared_weight_store_)throw std::invalid_argument("Flash Next projection lacks its frozen prepared source");
    return {*weight,*prepared_weight_store_,device};
}
Qwen4ExpImmutableWeightBinding Qwen4ExpGraph::gain(const WeightBinding *weight) const
{
    if(!weight || !weight->immutable || !weight->tensor)
        throw std::invalid_argument("Flash Next norm/convolution requires its immutable native source");
    return Qwen4ExpImmutableWeightBinding(*weight->tensor);
}
sampling_math::DeviceGenerationForwardFailurePolicy Qwen4ExpGraph::deviceGenerationForwardFailurePolicy() const
{
    using Policy=sampling_math::DeviceGenerationForwardFailurePolicy;
    if (!config_.default_device.is_gpu() || !ownsPLE()) return Policy::ResetAtResponseAdmission;
    if (!config_.ngram_table_placement)
        throw std::logic_error("Native request failure lifetime requires admitted immutable-table placement");
    if (config_.ngram_table_placement->isGPUAdaptive())
        return Policy::PreserveUntilRequestReset;
    return immutableRowLookupUsesNativeDeviceStorage(config_.ngram_table_placement->lookupForConsumer(
        {config_.ngram_table_placement->worldRank(),config_.default_device}))
        ? Policy::PreserveUntilRequestReset : Policy::ResetAtResponseAdmission;
}
std::shared_ptr<IRetainedGraphStorageBoundary> Qwen4ExpGraph::retainedStorageBoundary(DeviceId device)
{
    if(device!=config_.default_device)
        throw std::invalid_argument("Flash Next storage boundary requested for a different participant");
    if(!device.is_gpu() || !ownsPLE()) return {};
    if (!config_.ngram_table_placement)
        throw std::logic_error("Flash Next GPU storage requires its resolved admission placement");
    const auto storage = config_.ngram_table_placement->lookupForConsumer(
        {config_.ngram_table_placement->worldRank(),device});
    if (storage == ImmutableRowLookupExecution::AdaptiveGPUCache)
    {
        if (!model_ctx_ || !config_.ngram_table_placement->isGPUAdaptive())
            throw std::logic_error(
                "Flash Next adaptive lookup requires its exact model-owned mixed placement");
        const auto sourcePlacement = model_ctx_->resolvedImmutableRowTablePlacement();
        if (!sourcePlacement || *sourcePlacement != *config_.ngram_table_placement)
            throw std::logic_error("Adaptive graph and original source have different admitted placements");
        auto boundary = model_ctx_->adaptiveImmutableLookupBoundary(device);
        if (!boundary || !boundary->isAdaptive() || boundary->device() != device ||
            (lookup_ && lookup_ != boundary))
            throw std::logic_error(
                "Flash Next adaptive lookup replaced or lost its model-owned source boundary");
        // Register original model-owned external roots only. The shared arena
        // inventory never allocates a second cache, frame or protocol tensor.
        if (arena_)
            boundary->bindAdaptiveArena(*arena_);
        lookup_ = std::move(boundary);
        return lookup_;
    }
    if (immutableRowLookupUsesNativeDeviceStorage(storage))
    {
        if (!model_ctx_ || !model_ctx_->residentImmutableRowTable(device) || lookup_)
            throw std::logic_error("Flash Next resident lookup lost its exact GPU owner or retained a host lane");
        return {};
    }
    if(lookup_) return lookup_;
    if(isGraphCaptureActive())
        throw std::logic_error("Flash Next immutable storage must bind before graph capture");
    if(!model_ctx_ || !memory_ || !model_ctx_->hostImmutableRowProvider())
        throw std::logic_error("Flash Next GPU lookup requires admitted model table and physical-memory authority");
    auto &protocol=tensor(BufferId::QFN_NGRAM_LOOKUP_STATE);
    // The storage lane retains its original protocol tensor before ordinary graph
    // scratch materialization. Its declared arena tensor remains the sole
    // physical owner; request admission publishes the actual initial bytes.
    TransferEngine::allocateDeviceStorage(&protocol,device);
    auto protocol_owner=arena_->getSharedTensor(BufferId::QFN_NGRAM_LOOKUP_STATE);
    if(!protocol_owner || protocol_owner.get()!=&protocol)
        throw std::logic_error("Flash Next immutable lookup protocol lost its original arena owner");
    const auto rows=tensor(BufferId::QFN_PLE_EMBEDDING).rows();
    lookup_=qwen4exp::ImmutableLookupExecutionBoundary::create(
        model_ctx_->hostImmutableRowProvider(),device,std::move(protocol_owner),rows,memory_);
    return lookup_;
}
std::optional<Qwen4ExpIntBinding> Qwen4ExpGraph::rowParameters(DeviceId device) const
{
    if(!device.is_gpu()) return {};
    return integer(BufferId::QFN_QSA_REPLAY,
        static_cast<std::size_t>(config_.max_request_count)*sizeof(qwen4exp::QwenSparseReplay));
}
int Qwen4ExpGraph::poolCapacity() const
{
    // BufferArena preserves axis zero and flattens trailing axes. A declared
    // [sparse_layers * requests, blocks, 128] bank is therefore [banks, blocks*128].
    const auto &pool=tensor(BufferId::QFN_QSA_POOLED_KEYS);
    if(pool.shape().size()!=2 || pool.cols()%qwen4exp::QwenSparseIndexDim!=0 ||
        pool.cols()/qwen4exp::QwenSparseIndexDim>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::logic_error("Flash Next pooled-key matrix does not preserve its declared block geometry");
    return static_cast<int>(pool.cols()/qwen4exp::QwenSparseIndexDim);
}
Qwen4ExpAuxiliaryState &Qwen4ExpGraph::state(IKVCache &cache,void *publicationStream)
{
    if(auxiliaryContent()==AuxiliaryContent::Absent)
        throw std::logic_error("Flash Next GDN-only participant has no QSA/PLE auxiliary owner");
    if(auto *installed=cache.auxiliaryState())
    {
        auto *qfn=dynamic_cast<Qwen4ExpAuxiliaryState *>(installed);
        if(!qfn || qfn->device()!=config_.default_device)
            throw std::logic_error("Flash Next cache already owns an incompatible auxiliary state");
        qfn->requireReady(publicationStream);return *qfn;
    }
    Qwen4ExpAuxiliaryState::Params params{.cache=cache,.device=config_.default_device,
        .requestCapacity=config_.max_request_count,.eosToken=static_cast<int>(config_.flash_next->ngrams().hash_reset_token)};
    std::size_t sparseIndex=0;
    for(int layer=config_.pp_layer_offset;layer<config_.pp_layer_offset+config_.n_layers;++layer)
    {
        // The hybrid cache public API names global model layers, including PP
        // participants whose interval does not begin at zero. Local/compressed
        // indices are an implementation detail owned by the cache itself.
        const int cacheLayer=layer;
        if(config_.flash_next->mixers().at(layer)==Qwen4ExpMixer::SparseAttention)
        {
            const auto poolBytes=static_cast<std::size_t>(poolCapacity())*qwen4exp::QwenSparseIndexDim*sizeof(float);
            for(int sequence=0;sequence<config_.max_request_count;++sequence)
            {
                const auto bank=sparseIndex*config_.max_request_count+sequence;
                std::optional<Qwen4ExpAuxiliaryState::SparseVerifierStorage> verifier;
                const int retainedRows=resolveMTPRetainedTargetQueryRows(config_.mtp);
                // This lane binds the canonical R1 slot. Other request slots
                // remain guarded until request-major model composition is installed.
                if(retainedRows>0 && sequence==0)
                    verifier=Qwen4ExpAuxiliaryState::SparseVerifierStorage{retainedRows,
                        fp32(BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,sparseIndex*retainedRows*128*sizeof(float)),
                        integer(BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,bank*sizeof(std::uint64_t)),
                        {integer(BufferId::QFN_QSA_TRANSACTION_PRIOR,bank*sizeof(qwen4exp::QwenSparseTransactionPrior))},
                        integer(BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC,bank*sizeof(qwen4exp::QwenSparsePriorTraffic))};
                params.sparseLayers.push_back({layer,cacheLayer,sequence,cache.max_seq_len(),
                    fp32(BufferId::QFN_QSA_POOLED_KEYS,bank*poolBytes),
                    fp32(BufferId::QFN_QSA_PARTIAL_KEYS,bank*qwen4exp::QwenSparsePool*qwen4exp::QwenSparseIndexDim*sizeof(float)),
                    std::move(verifier)});
            }
            ++sparseIndex;
        }
        if(layer==static_cast<int>(config_.flash_next->ngrams().decoder_layer))
            for(int sequence=0;sequence<config_.max_request_count;++sequence)
            {
                const std::size_t tokens=sequence*4*sizeof(std::int32_t);
                const std::size_t tail=9*config_.residualWidth()*sizeof(float),conv=sequence*2*tail;
                Qwen4ExpAuxiliaryState::PLEHistory history{layer,cacheLayer,sequence,config_.residualWidth(),
                    integer(BufferId::QFN_PLE_TOKEN_HISTORY,tokens),
                    integer(BufferId::QFN_PLE_TOKEN_HISTORY,tokens+2*sizeof(std::int32_t)),
                    fp32(BufferId::QFN_PLE_CONV_HISTORY,conv),fp32(BufferId::QFN_PLE_CONV_HISTORY,conv+tail)};
                // The retained quantized projection path is single-request until its
                // counted request-hole ABI is implemented. Dedicated prior banks cannot
                // alias ordinary accepted/next banks or reset/archive payloads.
                if(sequence==0&&retainsMTPGraphCapacity(config_.mtp)) {
                    const int retained=resolveMTPRetainedTargetQueryRows(config_.mtp);
                    history.verifier=Qwen4ExpAuxiliaryState::PLEVerifierStorage{retained,
                        integer(BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY),integer(BufferId::QFN_PLE_TRANSACTION_LIVE),
                        integer(BufferId::QFN_PLE_RETAINED_TOKENS),integer(BufferId::QFN_PLE_TRANSACTION_COPY_BYTES),
                        fp32(BufferId::QFN_PLE_PRIOR_CONV_HISTORY),fp32(BufferId::QFN_PLE_RETAINED_NORMALIZED)};
                }
                params.pleHistories.push_back(std::move(history));
            }
    }
    auto owner=std::make_unique<Qwen4ExpAuxiliaryState>(std::move(params));
    auto *result=owner.get();cache.attachAuxiliaryState(std::move(owner),publicationStream);return *result;
}
Qwen4ExpPLEContext Qwen4ExpGraph::pleContext(const ForwardInput &input,DeviceId device,const std::string &prefix) const
{
    if(config_.grouped_mtp_verifier && &modelForwardInput()!=&input)
        throw std::invalid_argument("PLE verifier context must retain its exact complete forward invocation");
    // The common projection helper also serves ordinary compact CPU inputs and
    // MTP conditions. PLE's retained verifier policy is narrower on every backend:
    // its accepted-state transaction must be the actual grouped decode invocation.
    if(config_.grouped_mtp_verifier &&
        (input.execution_role!=ForwardExecutionRole::GroupedMTPVerifier || input.execution_phase!=ForwardExecutionPhase::Decode))
        throw std::invalid_argument("PLE grouped projections require GroupedMTPVerifier Decode invocation");
    const auto &ngram=config_.flash_next->ngrams();
    qwen4exp::NgramHashPolicy hash;
    std::copy(ngram.multipliers.begin(),ngram.multipliers.end(),hash.multipliers.begin());
    std::copy(ngram.head_vocab_sizes.begin(),ngram.head_vocab_sizes.end(),hash.moduli.begin());
    std::copy(ngram.head_offsets.begin(),ngram.head_offsets.end(),hash.offsets.begin());
    hash.eosToken=ngram.hash_reset_token;
    // This is an explicit arithmetic choice, not a padding workaround. Ordinary
    // projections retain their existing policy; verifier key/value GEMMs borrow
    // the same original count as hash/lookup, even for a physical one-row graph.
    return {.device=device,.mpi=mpi_ctx_.get(),.prefix=prefix,.rows=modelExtensionRows(input.seq_len,device),
        .hidden=config_.d_model,.epsilon=config_.rms_norm_eps,.hash=hash,
        .nativeFormat=ggufToTensorType(ngram.table.type),.nativeRowElements=static_cast<int>(ngram.row_elements),
        .nativeRowBytes=static_cast<std::size_t>(ngram.row_bytes),
        .arithmetic=config_.grouped_mtp_verifier ? Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier :
            Qwen4ExpProjectionArithmetic::Ordinary,
        .projectionRows=config_.grouped_mtp_verifier ?
            std::optional{singleRequestProjectionQueryRows(input.seq_len,device)} : std::nullopt};
}
std::optional<Qwen4ExpIntBinding> Qwen4ExpGraph::adaptiveCountBinding(const ForwardInput &input,
                                                                      const DeviceRowRange &rows) const
{
    if (!rows.countOwner())
        return {};
    if (!arena_ || input.device != config_.default_device || !input.device.is_gpu())
        throw std::logic_error("Adaptive original query requires its actual participant arena");
    BufferId role;
    if (input.device_prefill_chunk)
        role = BufferId::PREFILL_CHUNK_GEOMETRY;
    else if (input.execution_phase == ForwardExecutionPhase::Decode &&
             input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
        role = BufferId::MTP_VERIFIER_REQUEST_LENGTHS;
    else if (input.device_request_query ||
             (input.graph_submission_intent == ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
              input.execution_phase == ForwardExecutionPhase::Prefill))
        role = BufferId::REQUEST_BATCH_GEOMETRY;
    else
        throw std::invalid_argument(
            "Adaptive original query has no implemented request/chunk/verifier count producer");
    const auto owner = arena_->getSharedTensor(role);
    if (!owner || arena_->getTensor(role) != owner.get() || owner->native_type() != TensorType::INT32 ||
        owner->is_view() || owner->transferStorageOwner() != owner.get() ||
        owner->current_device() != input.device || !owner->gpu_data_ptr())
        throw std::invalid_argument("Adaptive original count requires its canonical prepared arena root");
    const auto base = reinterpret_cast<std::uintptr_t>(owner->gpu_data_ptr());
    const auto count = reinterpret_cast<std::uintptr_t>(rows.countOwner());
    if (count < base || count - base > owner->size_bytes() ||
        sizeof(std::int32_t) > owner->size_bytes() - (count - base) || (count - base) % alignof(std::int32_t))
        throw std::invalid_argument("Adaptive query count is outside its actual producer arena root");
    return Qwen4ExpIntBinding(*owner, role, static_cast<std::size_t>(count - base));
}

std::string Qwen4ExpGraph::addModelLayerInput(ComputeGraph &graph,const ForwardInput &input,
    int layer,DeviceId device,const std::string &dependency)
{
    const auto &ngram=config_.flash_next->ngrams();
    if(layer!=static_cast<int>(ngram.decoder_layer))return dependency;
    const std::string prefix="layer"+std::to_string(layer)+"_ple_";
    const auto context=pleContext(input,device,prefix);
    auto *cache=input.getKVCacheForDevice(device);
    if(!cache)throw std::invalid_argument("Flash Next requires canonical KV and recurrent cache ownership");
    auto &owner=state(*cache,input.device_state_publication_stream);
    if(input.batch_size!=1)
        throw std::logic_error("Flash Next PLE quantized projections require a counted request-hole ABI before request batching");
    const auto rows=context.rows;
    const auto *tokens=static_cast<const std::int32_t *>(device.is_gpu()?input.token_ids_device:input.token_ids);
    if(!tokens)throw std::invalid_argument("Flash Next PLE requires exact native token rows on its execution authority");
    const Qwen4ExpIntInput tokenRows=Qwen4ExpNativeIntInput{device,tokens,static_cast<std::size_t>(input.seq_len)};
    const auto &history=owner.ple(0);
    const auto &weights=layerWeightBindingsForGraph(layer).flash_next.ple;
    if (!model_ctx_ || !config_.ngram_table_placement)
        throw std::logic_error("Flash Next PLE requires its admitted source and concrete placement");
    const auto sourcePlacement=model_ctx_->resolvedImmutableRowTablePlacement();
    if (!sourcePlacement || *sourcePlacement!=*config_.ngram_table_placement)
        throw std::logic_error("Flash Next graph and model source have different admitted ngram placements");
    const auto tier=config_.ngram_table_placement->lookupForConsumer(
        {config_.ngram_table_placement->worldRank(),device});
    const Qwen4ExpPLEBuilder::LookupStorage storage=[&]() -> Qwen4ExpPLEBuilder::LookupStorage {
        if(tier==ImmutableRowLookupExecution::FixedHeadShardGPU)
        {
            if(!config_.ngram_replicated_ple_cohort || !input.replicated_ple_query || !arena_)
                throw std::invalid_argument("Fixed-head PLE requires its actual Rank cohort and DGO query producer");
            const auto *native=rows.liveRows?std::get_if<Qwen4ExpNativeIntInput>(&rows.liveRows->value):nullptr;
            if(rows.liveRows && (!native || native->owner!=device || native->capacity!=1 || !native->data))
                throw std::invalid_argument("Fixed-head PLE has no original native primitive count");
            const auto original=native?DeviceRowRange::deviceCounted(rows.capacity,native->data):DeviceRowRange::fullyActive(rows.capacity);
            config_.ngram_replicated_ple_cohort->requireQuery(*input.replicated_ple_query,input,original,owner);
            auto table=model_ctx_->residentImmutableRowTable(device);
            if(!table)throw std::invalid_argument("Fixed-head PLE exact range owner is not materialized");
            return Qwen4ExpFixedHeadLookupStorage{config_.ngram_replicated_ple_cohort,*input.replicated_ple_query,std::move(table),
                DeviceGenerationControlBinding::admit(*arena_,device,0),
                integer(BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER),integer(BufferId::QFN_NGRAM_SHARD_HEADERS),
                integer(BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS),integer(BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE),
                integer(BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE)};
        }
        if (tier == ImmutableRowLookupExecution::AdaptiveGPUCache)
        {
            (void)retainedStorageBoundary(device);
            if (!lookup_ || !arena_)
                throw std::logic_error(
                    "Adaptive PLE requires its original source boundary and canonical arena");
            const auto original = singleRequestProjectionQueryRows(input.seq_len, device);
            const auto roots = lookup_->adaptiveRoots();
            return ImmutableLookupStage::Storage{ImmutableLookupStage::AdaptiveGPUStorage{
                lookup_,
                DeviceGenerationControlBinding::admit(*arena_, device, 0),
                roots.cache ? std::optional{Qwen4ExpArenaBinding<TensorType::INT8>(
                                  *roots.cache, BufferId::QFN_NGRAM_ADAPTIVE_CACHE)}
                            : std::nullopt,
                {*roots.frame, BufferId::QFN_NGRAM_ADAPTIVE_FRAME},
                {*roots.protocol, BufferId::QFN_NGRAM_LOOKUP_STATE},
                original,
                adaptiveCountBinding(input, original)}};
        }
        if (tier==ImmutableRowLookupExecution::ResidentGPU)
        {
            auto table=model_ctx_->residentImmutableRowTable(device);
            if (!device.is_gpu() || !table || !arena_)
                throw std::logic_error("Flash Next resident PLE requires its exact GPU table and canonical arena");
            return ImmutableLookupStage::Storage{ImmutableLookupStage::ResidentGPUStorage{std::move(table),
                DeviceGenerationControlBinding::admit(*arena_,device,0)}};
        }
        auto table=model_ctx_->hostImmutableRowProvider();
        if (!table || (config_.ngram_table_placement->isHostAdaptive()
            ? table->mode()!=HostImmutableRowMode::AdaptiveRAMSSD
            : table->wholeStorage()!=config_.ngram_table_placement->storage()))
            throw std::logic_error("Flash Next hosted PLE provider differs from its admitted actual mode");
        const auto native=integer(BufferId::QFN_NGRAM_NATIVE_ROWS);
        if (device.is_cpu()) return ImmutableLookupStage::Storage{ImmutableLookupStage::CPUStorage{std::move(table),native}};
        (void)retainedStorageBoundary(device);
        return ImmutableLookupStage::Storage{ImmutableLookupStage::GPUStorage{lookup_,integer(BufferId::QFN_NGRAM_LOOKUP_STATE),native}};
    }();
    const auto normalized=fp32(BufferId::QFN_PLE_NORMALIZED);
    std::vector<std::string> dependencies=dependency.empty()?std::vector<std::string>{}:std::vector<std::string>{dependency};
    std::optional<Qwen4ExpAuxiliaryState::PLEVerifierPrior> prior;
    if(config_.grouped_mtp_verifier) {
        prior.emplace(owner.appendVerifierPLEPrior(graph,prefix,0,rows,tokenRows,normalized,dependencies));
        dependencies={prior->snapshotProducer()};
    }
    const auto terminal=Qwen4ExpPLEBuilder::append(graph,
        context,
        {projection(weights.key,device),projection(weights.value,device),gain(weights.key_norm),gain(weights.query_norm),
            gain(weights.convolution_norm),gain(weights.convolution)},
        {tokenRows,prior?prior->tokens():history.acceptedTokens,integer(BufferId::QFN_NGRAM_ROW_IDS),
            fp32(BufferId::QFN_PLE_EMBEDDING),fp32(BufferId::QFN_PLE_KEY),fp32(BufferId::QFN_PLE_VALUE),
            fp32(BufferId::QFN_PLE_QUERY),fp32(BufferId::QFN_PLE_GATED),fp32(BufferId::QFN_PLE_NORMALIZED),
            prior?prior->convolution():history.acceptedConvolution,qwen4ExpTrunkBinding(buffers_.layer_buffers)},storage,
        dependencies);
    if(prior)
        return owner.appendVerifierPLERetain(graph,prefix,std::move(*prior),tokenRows,normalized,{terminal,prefix+"conv_norm"});
    return owner.appendAcceptedPLECommit(graph,prefix,0,rows,{rows.liveRows,rows.capacity},tokenRows,
        fp32(BufferId::QFN_PLE_NORMALIZED),{terminal});
}
std::string Qwen4ExpGraph::addModelAttentionPreparation(ComputeGraph &graph,const AttentionConsumerBinding &binding)
{
    binding.requireLayerIdentity();
    if (binding.layerIdentity.consumer() == GraphAttentionConsumer::Dense)
        return {};
    binding.layerIdentity.requireSelectedConsumer();
    const auto &input=modelForwardInput();
    if(!binding.cache || binding.requests!=1)
        throw std::invalid_argument("Flash Next sparse preparation requires an exact single-request native cache binding");
    (void)state(*binding.cache,input.device_state_publication_stream);
    using Stage=QwenSparseAttentionStage;
    std::variant<Qwen4ExpIntInput,Stage::PrepareReplay::ContiguousPositions> positions=Stage::PrepareReplay::ContiguousPositions{
        input.token_offset!=0?input.token_offset:input.position_offset};
    if(input.position_policy==ForwardPositionPolicy::ExplicitRows)
    {
        const auto *rows=static_cast<const std::int32_t *>(binding.device.is_gpu()?input.position_ids_device:input.position_ids);
        if(!rows)throw std::invalid_argument("Flash Next sparse preparation requires exact native absolute positions");
        positions=Qwen4ExpIntInput(Qwen4ExpNativeIntInput{binding.device,rows,static_cast<std::size_t>(binding.queryRows)});
    }
    const auto name=binding.prefix+"qsa_prepare_replay";
    graph.addNode(name,std::make_unique<Stage>(Stage::Params{
        .device_id=binding.device,.mpi_ctx=mpi_ctx_.get(),.rowCapacity=binding.queryRows,.tokenCapacity=binding.cache->max_seq_len(),
        .replay=integer(BufferId::QFN_QSA_REPLAY),.operation=Stage::PrepareReplay{*binding.cache,binding.cacheLayer,0,
            modelExtensionRows(binding.queryRows,binding.device).liveRows,positions,rowParameters(binding.device)}}),binding.device);
    graph.addDependency(name,binding.preparedQueryReady);return name;
}
std::optional<std::string> Qwen4ExpGraph::addModelAttentionConsumer(ComputeGraph &graph,const AttentionConsumerBinding &binding)
{
    binding.requireLayerIdentity();
    if (binding.layerIdentity.consumer() == GraphAttentionConsumer::Dense)
        return {};
    binding.layerIdentity.requireSelectedConsumer();
    const auto &input=modelForwardInput();
    if(!binding.cache || binding.requests!=1 || input.batch_size!=1)
        throw std::invalid_argument("Flash Next sparse consumer requires the supported canonical single-request lane");
    auto &owner=state(*binding.cache,input.device_state_publication_stream);
    const auto &pool=owner.sparse(binding.modelLayer,0);
    const auto &weights=layerWeightBindingsForGraph(binding.modelLayer).flash_next.indexer;
    const auto ready=binding.prefix+"qsa_prepare_replay";
    auto replay=integer(BufferId::QFN_QSA_REPLAY);
    std::string replayReady=ready;
    std::optional<ImmutableVerifierPrior> priorLease;
    std::optional<Qwen4ExpSparseAttentionRetention> retained;
    const bool verifier=config_.grouped_mtp_verifier;
    if(verifier) {
        if(!pool.verifier || binding.queryRows>pool.verifier->rowCapacity)
            throw std::invalid_argument("Flash Next grouped sparse rows exceed admitted retained candidate/prior storage");
        const auto* append=graph.getNode(binding.nativeCacheReady);
        if(!append || !append->stage || append->stage->type()!=ComputeStageType::KV_CACHE_APPEND ||
            append->stage->device()!=binding.device)
            throw std::invalid_argument("Flash Next prior snapshot must precede the actual canonical KV append node");
        if(binding.device.is_cpu() && input.sequence_lengths &&
            (input.sequence_lengths->size()!=1 || input.sequence_lengths->front()!=binding.queryRows))
            throw std::invalid_argument("Flash Next CPU grouped projections require actual full invocation rows, without padded capacity");
        const auto terminalCounters=binding.device.is_gpu() ?
            std::optional{QwenSparseCounterBinding::admit(*arena_,binding.device,0,config_.n_layers,pool.verifier->rowCapacity)} : std::nullopt;
        priorLease.emplace(owner.appendSparseVerifierPrior(graph,binding.prefix,binding.modelLayer,0,
            binding.queryRows,replay,{ready},terminalCounters));
        priorLease->orderBefore(graph,binding.nativeCacheReady);
        replay=pool.verifier->prior.storage;
        replayReady=priorLease->snapshotProducer();
        retained=Qwen4ExpSparseAttentionRetention{pool.verifier->rawKeys,pool.verifier->candidateCopyBytes,
            RequestRowOrigins::fixedStride(0,binding.queryRows),pool.verifier->prior,terminalCounters};
    }
    const auto terminal=Qwen4ExpSparseAttentionBuilder::append(graph,binding,
        {.hidden=config_.d_model,.tokenCapacity=binding.cache->max_seq_len(),
            .blockCapacity=poolCapacity(),
            .positionCapacity=static_cast<int>(config_.flash_next->contextLength()),.rotaryDimension=64,
            .epsilon=config_.rms_norm_eps,.projectionArithmetic=verifier ? Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier :
                Qwen4ExpProjectionArithmetic::Ordinary,
            .projectionRows=verifier ? Qwen4ExpSparseAttentionBuilder::verifierProjectionRows(binding.device,
                modelExtensionRows(binding.queryRows,binding.device)) : std::nullopt,.mpi=mpi_ctx_.get(),
            .rotarySource=qwen4exp::QwenSparseRotarySource::AnalyticTheta,.ropeTheta=config_.rope_theta,
            .publication=verifier ? Qwen4ExpSparsePublication::DeferredVerifierAcceptance : Qwen4ExpSparsePublication::AcceptAllLiveRows,
            .poolSelection=Qwen4ExpSparseAttentionBuilder::poolSelectionForInvocation(
                input.execution_phase,binding.device,binding.queryRows),
            .attentionArithmetic=sparse_attention_arithmetic_},
        {projection(weights.query,binding.device),projection(weights.key,binding.device),gain(weights.query_norm),gain(weights.key_norm),{},{}},
        {replay,fp32(BufferId::QFN_QSA_QUERY),fp32(BufferId::QFN_QSA_RAW_KEY),
            pool.partialKeys,pool.pooledKeys,fp32(BufferId::QFN_QSA_PREPARED_QUERY),fp32(BufferId::QFN_QSA_SCORES),
            integer(BufferId::QFN_QSA_SELECTED),integer(BufferId::QFN_QSA_SELECTED_COUNT),rowParameters(binding.device),
            binding.device.is_gpu()?std::optional{integer(BufferId::QFN_QSA_PARTIALS)}:std::nullopt,
            integer(BufferId::QFN_QSA_OBSERVATIONS),replayReady,ready,retained});
    if(priorLease)std::move(*priorLease).retainForPublication(graph,binding.prefix+"qsa_retain_candidates");
    return terminal;
}
} // namespace llaminar2
