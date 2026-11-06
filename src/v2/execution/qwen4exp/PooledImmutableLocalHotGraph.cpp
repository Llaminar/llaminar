/**
 * @file PooledImmutableLocalHotGraph.cpp
 * @brief Original compact request/native response DAG with independent local model readiness.
 *
 * Every movement is an explicit original addressed transport stage. The sole
 * model writer is LocalRead; all request, pool and policy work is captured in
 * the typed participant-local auxiliary range and retired by the same parent.
 * Each declaration borrows its original immutable execution descriptor so a
 * native mixed-GPU I/O parent cannot be mistaken for a CPU expert service.
 */
#include "PooledImmutableLocalHotGraph.h"
#include "PooledImmutableLocalHotSourceGraph.h"
#include "PooledImmutableRetainedParticipants.h"
#include "PooledImmutableDirectedLookupGraph.h"
#include "PooledImmutableTransferBinding.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableLocalHotPhaseStage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTransportKernels.h"
#include <algorithm>
#include <set>
namespace llaminar2 {
namespace {
namespace q=qwen4exp;
namespace t=q::local_hot_transport;
using Local=q::PooledImmutableLocalHotPreparedStorage;
using Transport=q::PooledImmutableDirectedTransport;
using Action=q::PooledImmutableLocalHotAction;
/** @return Actual canonical controller participant; malformed or retired owners reject before dereference. */
DeviceId controllerDevice(const DeviceGenerationControlBinding& control)
{const auto device=control.owner()->current_device();if(!device || !device->is_gpu())throw std::invalid_argument("Compact controller lost its actual GPU owner");control.requireFor(*device);return *device;}
/** @return Actual declared consumer coordinate, rejecting equal-looking foreign source owners. */
std::uint32_t localCoordinate(const Transport& transport,const std::shared_ptr<Local>& local)
{
    transport.requireCurrent();const auto& geometry=transport.localHotGeometry();
    const auto source=transport.sourceBoundary()->pooledStorage();
    if(!geometry || !local || local->originalSource()!=source || local->originalSource().owner_before(source) || source.owner_before(local->originalSource()))
        throw std::invalid_argument("Compact graph requires its actual original local/source association");
    local->requireAuthority(transport.sourceBoundary()->pooledStorage()->retainedOwnerObservation().authority());
    if(local->geometry().maximumOriginalIds()!=geometry->maximumPoolUniqueRows() || local->geometry().extent()!=geometry->sourceExtent())
        throw std::invalid_argument("Compact graph changed original query/native extent");
    for(std::uint32_t i=0;i<geometry->consumers().size();++i)
        if(geometry->consumers()[i]==local->geometry().allocator())return i;
    throw std::invalid_argument("Compact graph local allocator is not an original consumer");
}
/** @return Exact original foreign reader endpoint, not a synthesized self channel. */
std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> originalReader(const Transport& transport,std::uint32_t coordinate)
{
    for(const auto& reader:transport.readers())if(reader->coordinate()==coordinate){reader->requireCurrent();return reader;}
    throw std::invalid_argument("Compact operation requires its actual foreign reader channel");
}
/** @return One bounded private frame subrange; never an invented ordinary tensor role. */
q::PooledNativeTransferSpan localFrame(const Local& local,std::size_t offset,std::size_t bytes)
{return {local.roots().frame,offset,bytes};}
/** @return A source-local control subrange under the original transport owner. */
q::PooledNativeTransferSpan field(const Transport& transport,DeviceId device,std::uint32_t coordinate,std::size_t offset,std::size_t bytes)
{return transport.localHotControl(device,coordinate).subspan(offset,bytes);}
/** @return Checked exact native address without observing a live value. */
template<class T>T* address(const q::PooledNativeTransferSpan& span,DeviceId device)
{return static_cast<T*>(span.pointer(device,alignof(T)));}
/** @brief Mix only immutable producer/geometry identity. */
void mix(std::uint64_t& h,std::uint64_t x){h=(h^x)*1099511628211ULL;}
/** @brief Actual fixed/count operation on an original channel; no implicit cross-device work. */
enum class CompactWire:std::uint32_t{Admission,RequestHeader,MissKeys,ResultHeader,NativeRows,Ack};
/** @return Static original operation key, shared by both endpoints and excluding every live count/ordinal. */
std::uint64_t wireKey(const Transport& transport,const Local& local,std::uint32_t coordinate,CompactWire wire)
{
    std::uint64_t h=0x51464e4c31434f4dULL;const auto& g=*transport.localHotGeometry();
    mix(h,t::Version);mix(h,static_cast<unsigned>(wire));mix(h,coordinate);mix(h,local.consumerStorageIdentity());
    mix(h,local.plan().table());mix(h,local.plan().sourceStorageGeneration());mix(h,g.maximumPoolUniqueRows());
    mix(h,g.sourceExtent().rowBytes());mix(h,static_cast<unsigned>(g.sourceExtent().format()));return h?h:1;
}
/** @return True only for the source-to-reader original directions. */
bool sourceDirection(CompactWire wire)
{return wire==CompactWire::Admission || wire==CompactWire::ResultHeader || wire==CompactWire::NativeRows;}
/** @return Actual endpoint GPU for an explicit wire role. */
DeviceId wireDevice(const q::PooledImmutableReplicatedReaderTransport& reader,CompactWire wire,CapturedTransferEndpoint side)
{
    const bool source=sourceDirection(wire)==(side==CapturedTransferEndpoint::Producer);
    return source?reader.originalSource().device():reader.originalReader().device();
}
/** @brief An original fixed/count endpoint with the complete native receipt kept through aggregate retirement. */
class CompactTransfer final:public IComputeStage {
public:
    /** @brief Bind only an actual original consumer and its own native row/key span. */
    CompactTransfer(BufferArena& arena,std::shared_ptr<const Transport> transport,std::shared_ptr<Local> local,
        DeviceGenerationControlBinding control,CompactWire wire,CapturedTransferEndpoint side)
        :IComputeStage(wireDevice(*originalReader(*transport,localCoordinate(*transport,local)),wire,side)),arena_(&arena),
         transport_(std::move(transport)),local_(std::move(local)),coordinate_(localCoordinate(*transport_,local_)),
         reader_(originalReader(*transport_,coordinate_)),control_(control),wire_(wire),side_(side)
    {
        if(!reader_->nativeCollective())throw std::invalid_argument("Compact reader requires its declared original native route");
        control_.requireFor(device());identity_=graphCaptureVariantSignature();
    }
    /** @return Exact role observation for cold graph/retirement authentication. */
    bool matches(const Transport* transport,const Local* local,CompactWire wire,CapturedTransferEndpoint side)const
    {return transport_.get()==transport && local_.get()==local && wire_==wire && side_==side;}
    /** @return Original per-operation static key, never an observed payload extent. */
    std::uint64_t messageKey()const{return wireKey(*transport_,*local_,coordinate_,wire_);}
    /** @return Actual local whole-launch native receipt, retained without host observation. */
    NativeReceiveCompletionView completion()const
    {if(!binding_ || (wire_!=CompactWire::NativeRows && wire_!=CompactWire::MissKeys))throw std::logic_error("Compact receipt has no actual payload binding");return binding_->nativeCompletion();}
    /** @brief Resolve only admitted private roots and the actual original controller once before capture. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override
    {
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
            throw std::invalid_argument("Compact transfer changed original participant or exact stream");
        require();const bool source=device()==reader_->originalSource().device();
        if(!binding_){
            const auto& channel=sourceDirection(wire_)?reader_->nativeResult():reader_->nativeAcknowledgement();
            const auto& f=local_->geometry().frameOffsets();const auto capacity=transport_->localHotGeometry()->keyCapacityBytes();
            auto control=transport_->localHotControl(device(),coordinate_);
            const auto span=[&]()->q::PooledNativeTransferSpan{
                switch(wire_){
                case CompactWire::Admission:return transport_->localHotAdmission(device());
                case CompactWire::RequestHeader:return source?control.subspan(offsetof(t::Control,request_in),sizeof(t::RequestHeader)):transport_->localHotOriginalRequest(device());
                case CompactWire::MissKeys:{
                    if(!source)return localFrame(*local_,f.miss_keys,capacity);
                    std::size_t index=0;for(std::uint32_t i=0;i<coordinate_;++i)if(transport_->localHotGeometry()->consumers()[i]!=transport_->localHotGeometry()->coordinator())++index;
                    return transport_->localHotSourceScratch(Transport::LocalHotSourceScratch::IncomingKeys).subspan(index*capacity,capacity);
                }
                case CompactWire::ResultHeader:return source?transport_->localHotNativePack(local_).spans()[1]:localFrame(*local_,f.result_header,sizeof(t::ResultHeader));
                case CompactWire::NativeRows:return source?transport_->localHotNativePack(local_).spans()[2]:localFrame(*local_,f.response,transport_->localHotGeometry()->resultCapacityBytes());
                case CompactWire::Ack:return source?control.subspan(offsetof(t::Control,result_in),sizeof(t::ResultHeader)):localFrame(*local_,f.result_header,sizeof(t::ResultHeader));
                }throw std::logic_error("Invalid compact wire");
            }();
            auto bound=channel.bind(side_,{messageKey(),span.bytes()},span);
            if(wire_==CompactWire::MissKeys || wire_==CompactWire::NativeRows){
                auto owner=arena_->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);
                const auto base=reinterpret_cast<std::uintptr_t>(owner->gpu_data_ptr()),error=reinterpret_cast<std::uintptr_t>(control_.error());
                if(error<base || error-base>owner->size_bytes()-sizeof(std::int32_t))throw std::logic_error("Compact native error is not its original controller subrange");
                q::PooledNativeTransferSpan errorSpan(owner,error-base,sizeof(std::int32_t));
                if(wire_==CompactWire::MissKeys){
                    auto extent=control.subspan(source?offsetof(t::Control,request_receive_bytes):offsetof(t::Control,request_send_bytes),sizeof(std::uint64_t));
                    q::PooledNativeRequestCompletionFrame frame{control.subspan(offsetof(t::Control,request_lane),sizeof(t::RequestLaneIdentity)),std::move(errorSpan),
                        static_cast<std::int32_t>(sampling_math::DeviceGenerationError::ImmutableStorageFailure)};
                    binding_.emplace(bound.withNativeRequestCounted(std::move(extent),std::move(frame)));
                }else{
                    auto header=source?transport_->localHotNativePack(local_).spans()[1]:localFrame(*local_,f.result_header,sizeof(t::ResultHeader));
                    auto extent=source?transport_->localHotNativePack(local_).spans()[3]:localFrame(*local_,f.received_payload_bytes,sizeof(std::uint64_t));
                    q::PooledNativeCompletionFrame frame{header.subspan(offsetof(t::ResultHeader,pool_interval),sizeof(q::adaptive_lookup::pooled::Interval)),std::move(errorSpan),
                        static_cast<std::int32_t>(sampling_math::DeviceGenerationError::ImmutableStorageFailure)};
                    binding_.emplace(bound.withNativeCounted(std::move(extent),q::PooledNativeCountedMessage::ReaderResult,std::move(frame)));
                }
            }else binding_.emplace(std::move(bound));
        }
        binding_->prepare(stream);setGPUStream(stream);return true;
    }
    /** @brief Submit the exact retained native endpoint; no root allocation/count readback occurs. */
    bool execute(IDeviceContext* context)override
    {
        if(!context || context->deviceId()!=device() || !binding_)throw std::logic_error("Compact transfer has no prepared original endpoint");
        require();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Compact native capture identity changed");
        binding_->enqueue(gpuExecution().nativeStream());return true;
    }
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return CollectiveGraphCaptureAuthority::RuntimeBackend;}
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override{return StageBufferContract::build().addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");}
    std::uint64_t graphCaptureVariantSignature()const override
    {require();auto h=local_->captureIdentity();mix(h,reinterpret_cast<std::uintptr_t>(transport_.get()));mix(h,control_.captureIdentity());mix(h,static_cast<unsigned>(wire_));mix(h,static_cast<unsigned>(side_));mix(h,messageKey());return h?h:1;}
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo result;result.addScalarInt("compact_wire",static_cast<unsigned>(wire_));return result;}
private:
    /** @brief Validate complete original source/worker/PMA/controller/root ownership without reading bytes. */
    void require()const
    {transport_->requireCurrent();reader_->requireCurrent();control_.requireFor(device());if(arena_->getTensor(BufferId::MTP_GENERATION_CONTROL)!=control_.owner() || localCoordinate(*transport_,local_)!=coordinate_)throw std::logic_error("Compact transfer changed actual controller/consumer");}
    BufferArena* arena_;std::shared_ptr<const Transport> transport_;std::shared_ptr<Local> local_;
    std::uint32_t coordinate_;std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> reader_;
    DeviceGenerationControlBinding control_;CompactWire wire_;CapturedTransferEndpoint side_;
    std::optional<q::PooledImmutableTransferBinding> binding_;std::uint64_t identity_;
};
/** @brief Exact native fixed/header/receipt projections for one original consumer. */
class CompactMetadata final:public IComputeStage {
public:
    /** @brief Retain actual source/consumer/private producer owners; never infer an admission from an L1 ordinal. */
    CompactMetadata(BufferArena& arena,std::shared_ptr<const Transport> transport,std::shared_ptr<Local> local,
        std::shared_ptr<Local> sourceLocal,DeviceGenerationControlBinding control,t::MetadataAction action,
        const CompactTransfer* payload=nullptr,
        q::PooledImmutableLocalHotAccessIntent intent=q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)
        :IComputeStage(controllerDevice(control)),arena_(&arena),transport_(std::move(transport)),local_(std::move(local)),
         source_local_(std::move(sourceLocal)),control_(control),action_(action),payload_(payload),intent_(intent),coordinate_(localCoordinate(*transport_,local_))
    {
        control_.requireFor(device());
        if(!q::validLocalHotAccessIntent(intent_))throw std::invalid_argument("Compact metadata has unknown immutable purpose");
        const bool source=device()==transport_->localHotGeometry()->coordinator().device;
        const bool sourceAction=action_==t::MetadataAction::PublishSourceAdmission || action_==t::MetadataAction::AcceptRequestHeader ||
            action_==t::MetadataAction::ConfirmRequestKeys || action_==t::MetadataAction::ConfirmAck;
        if(sourceAction!=source && action_!=t::MetadataAction::PrepareRequest)
            throw std::invalid_argument("Compact metadata action changed its original source/reader role");
        if(action_==t::MetadataAction::PrepareRequest && local_->geometry().allocator().device!=device())
            throw std::invalid_argument("Compact request must be authored by its actual local probe");
        if(source && (!source_local_ || source_local_->geometry().allocator()!=transport_->localHotGeometry()->coordinator() || localCoordinate(*transport_,source_local_)>=transport_->localHotGeometry()->consumers().size()))
            throw std::invalid_argument("Compact source metadata lost its actual local canonical query");
        identity_=graphCaptureVariantSignature();
    }
    /** @return Exact original cold identity for dependency and native retirement validation. */
    bool matches(const Transport* transport,const Local* local,t::MetadataAction action)const
    {return transport_.get()==transport && local_.get()==local && action_==action;}
    const DeviceGenerationControlBinding& control()const noexcept{return control_;}
    const auto& sourceLocal()const noexcept{return source_local_;}
    const CompactTransfer* payload()const noexcept{return payload_;}
    auto intent()const noexcept{return intent_;}
    /** @brief Bind actual prepared private addresses and exact native authority after original producers are prepared. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override
    {
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
            throw std::invalid_argument("Compact metadata changed original participant/exact stream");
        require();
        if(local_->geometry().allocator().device==device())local_->acquireSetup(stream);
        if(source_local_ && source_local_!=local_ && source_local_->geometry().allocator().device==device())source_local_->acquireSetup(stream);
        args_=arguments();setGPUStream(stream);return true;
    }
    /** @brief Record only the checked fixed native projection on its exact original stream. */
    bool execute(IDeviceContext* context)override
    {
        if(!context || context->deviceId()!=device() || !args_)throw std::logic_error("Compact metadata has no original prepared arguments");
        require();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Compact metadata capture identity changed");
        auto execution=gpuExecution();execution.requirePreparedInput(control_.owner());execution.requirePreparedOutput(control_.owner());
        const auto args=arguments();
        if(device().is_cuda()){
#if defined(HAVE_CUDA)
            t::enqueueCUDAPooledLocalHotTransport(action_,args,execution.nativeStream());
#else
            throw std::logic_error("Compact metadata requires its compiled CUDA backend");
#endif
        }else{
#if defined(HAVE_ROCM)
            t::enqueueROCmPooledLocalHotTransport(action_,args,execution.nativeStream());
#else
            throw std::logic_error("Compact metadata requires its compiled HIP backend");
#endif
        }
        execution.publish(control_.owner());return true;
    }
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override{return StageBufferContract::build().addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");}
    std::uint64_t graphCaptureVariantSignature()const override
    {require();auto h=local_->captureIdentity();mix(h,reinterpret_cast<std::uintptr_t>(transport_.get()));mix(h,control_.captureIdentity());mix(h,static_cast<unsigned>(action_));if(intent_!=q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)mix(h,static_cast<unsigned>(intent_));if(source_local_)mix(h,source_local_->captureIdentity());if(payload_)mix(h,payload_->graphCaptureVariantSignature());return h?h:1;}
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo result;result.addScalarInt("compact_metadata",static_cast<unsigned>(action_));return result;}
private:
    /** @brief Authenticate unchanged original source/PMA/controller and exact peer role. */
    void require()const
    {transport_->requireCurrent();control_.requireFor(device());if(arena_->getTensor(BufferId::MTP_GENERATION_CONTROL)!=control_.owner() || localCoordinate(*transport_,local_)!=coordinate_)throw std::logic_error("Compact metadata changed actual controller/consumer");}
    /** @return Exact source-local or own-consumer addresses; never dereference a foreign GPU alias. */
    t::MetadataArgs arguments()const
    {
        t::MetadataArgs args{};args.request.intent=intent_;args.device=device();args.first_error=control_.error();args.healthy=control_.healthy();
        const auto& g=*transport_->localHotGeometry();const bool source=device()==g.coordinator().device;
        const auto reporting=source && source_local_?source_local_:local_;
        args.reporting=address<q::local_hot::Counters>(localFrame(*reporting,reporting->geometry().frameOffsets().counters,
            sizeof(q::local_hot::Counters)),device());
        args.geometry=g.consumer(coordinate_,local_->consumerStorageIdentity());
        const auto control=transport_->localHotControl(device(),coordinate_);
        const auto ownControl=[&](std::size_t offset,std::size_t bytes){return control.subspan(offset,bytes);};
        const auto& f=local_->geometry().frameOffsets();
        if(action_==t::MetadataAction::PrepareRequest){
            args.request.query=address<const t::QueryIdentity>(localFrame(*local_,f.identity+offsetof(q::local_hot::FrameIdentity,query),sizeof(t::QueryIdentity)),device());
            args.request.original_ids=address<const std::uint64_t>(localFrame(*local_,f.control+offsetof(q::adaptive_lookup::FrameControl,original_ids),8),device());
            args.request.miss_rows=address<const std::uint64_t>(localFrame(*local_,f.control+offsetof(q::adaptive_lookup::FrameControl,source_rows),8),device());
            args.request.source_admission=address<const t::SourceAdmissionHeader>(transport_->localHotAdmission(device()),device());
            args.request.received_admission_bytes=address<const std::uint64_t>(ownControl(offsetof(t::Control,admission_receive_bytes),8),device());
            args.request.geometry=args.geometry;
            args.request.output=address<t::RequestHeader>(transport_->localHotOriginalRequest(device()),device());
            args.request.lane=address<t::RequestLaneIdentity>(ownControl(offsetof(t::Control,request_lane),sizeof(t::RequestLaneIdentity)),device());
            args.request.key_bytes=address<std::uint64_t>(ownControl(offsetof(t::Control,request_send_bytes),8),device());
            args.fixed_header_bytes=address<std::uint64_t>(ownControl(offsetof(t::Control,header_receive_bytes),8),device());
            return args;
        }
        args.admission=address<const t::SourceAdmissionHeader>(transport_->localHotAdmission(source?g.coordinator().device:device()),device());
        if(source){
            const auto& sf=source_local_->geometry().frameOffsets();
            args.source_query=address<const t::QueryIdentity>(localFrame(*source_local_,sf.identity+offsetof(q::local_hot::FrameIdentity,query),sizeof(t::QueryIdentity)),device());
            args.original_ids=address<const std::uint64_t>(localFrame(*source_local_,sf.control+offsetof(q::adaptive_lookup::FrameControl,original_ids),8),device());
            args.received_request=address<const t::RequestHeader>(ownControl(offsetof(t::Control,request_in),sizeof(t::RequestHeader)),device());
        }else args.received_request=address<const t::RequestHeader>(transport_->localHotOriginalRequest(device()),device());
        if(action_==t::MetadataAction::AcceptRequestHeader || action_==t::MetadataAction::ConfirmRequestKeys){
            args.received_lane=address<t::RequestLaneIdentity>(ownControl(offsetof(t::Control,request_lane),sizeof(t::RequestLaneIdentity)),device());
            args.payload_bytes=address<std::uint64_t>(ownControl(offsetof(t::Control,request_receive_bytes),8),device());
            args.received_payload_bytes=args.payload_bytes;
            args.fixed_header_bytes=address<std::uint64_t>(ownControl(offsetof(t::Control,header_receive_bytes),8),device());
        }
        if(action_==t::MetadataAction::AcceptResultHeader || action_==t::MetadataAction::ConfirmResult){
            args.received_result=address<const t::ResultHeader>(localFrame(*local_,f.result_header,sizeof(t::ResultHeader)),device());
            args.fixed_header_bytes=address<std::uint64_t>(localFrame(*local_,f.received_header_bytes,8),device());
            args.received_payload_bytes=address<const std::uint64_t>(localFrame(*local_,f.received_payload_bytes,8),device());
        }
        if(action_==t::MetadataAction::ConfirmAck){
            args.received_result=address<const t::ResultHeader>(ownControl(offsetof(t::Control,result_in),sizeof(t::ResultHeader)),device());
            args.received_payload_bytes=address<const std::uint64_t>(transport_->localHotNativePack(local_).spans()[3],device());
        }
        const auto reader=originalReader(*transport_,coordinate_);
        args.native_fixed_header=reader->nativeResult().nativeStorage()->authority(device()).binding();
        if(payload_){args.native_payload_completion=payload_->completion();args.native_payload_message=payload_->messageKey();}
        return args;
    }
    BufferArena* arena_;std::shared_ptr<const Transport> transport_;std::shared_ptr<Local> local_,source_local_;
    DeviceGenerationControlBinding control_;t::MetadataAction action_;const CompactTransfer* payload_;
    q::PooledImmutableLocalHotAccessIntent intent_;
    std::uint32_t coordinate_;std::optional<t::MetadataArgs> args_;std::uint64_t identity_;
};
/** @brief Exact native canonical-subset check/concatenation before original pool Begin. */
class CompactUnion final:public IComputeStage {
public:
    /** @brief Retain the complete original root roster; no partial consumer list can seal. */
    CompactUnion(std::shared_ptr<const Transport> transport,std::vector<std::shared_ptr<Local>> locals,
        DeviceRowRange tokens,DeviceGenerationControlBinding control,
        q::PooledImmutableLocalHotAccessIntent intent=q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)
        :IComputeStage(transport->localHotGeometry()->coordinator().device),transport_(std::move(transport)),locals_(std::move(locals)),control_(control),intent_(intent)
    {
        if(locals_.size()!=transport_->localHotGeometry()->consumers().size())throw std::invalid_argument("Compact union omitted actual consumers");
        for(std::uint32_t i=0;i<locals_.size();++i){if(localCoordinate(*transport_,locals_[i])!=i)throw std::invalid_argument("Compact union changed actual consumer order");if(locals_[i]->geometry().allocator()==transport_->localHotGeometry()->coordinator())source_=locals_[i];}
        if(!source_)throw std::invalid_argument("Compact union lost actual source canonical query");
        if(!q::validLocalHotAccessIntent(intent_))throw std::invalid_argument("Compact union has unknown immutable purpose");
        std::array<t::ConsumerGeometry,8> consumerBounds{};
        for(std::uint32_t i=0;i<locals_.size();++i){
            consumerBounds[i]=transport_->localHotGeometry()->consumer(i,locals_[i]->consumerStorageIdentity());
            consumerBounds[i].maximum_misses=std::min(consumerBounds[i].maximum_misses,locals_[i]->geometry().maximumOriginalIds());
        }
        auto bound=q::PooledImmutableLocalHotQueryBound::forCanonicalQuery(tokens,
            std::span{consumerBounds.data(),locals_.size()},transport_->localHotGeometry()->maximumPoolOriginalRows());
        query_.emplace(transport_->localHotPoolQuery(source_,std::move(bound),intent_==q::PooledImmutableLocalHotAccessIntent::FutureReadAhead
            ?q::adaptive_lookup::pooled::LookupPurpose::FutureLookupOnly:q::adaptive_lookup::pooled::LookupPurpose::CurrentDemand));
        identity_=graphCaptureVariantSignature();
    }
    const q::PooledImmutableLocalHotPoolQuery& query()const{return *query_;}
    const auto& transport()const noexcept{return transport_;}
    const auto& source()const noexcept{return source_;}
    const auto& locals()const noexcept{return locals_;}
    const auto& control()const noexcept{return control_;}
    auto intent()const noexcept{return intent_;}
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override
    {
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Compact union changed exact original participant/stream");
        control_.requireFor(device());query_->seal();args_=arguments();setGPUStream(stream);return true;
    }
    bool execute(IDeviceContext* context)override
    {
        if(!context || context->deviceId()!=device() || !args_ || graphCaptureVariantSignature()!=identity_)throw std::logic_error("Compact union has stale/unprepared original producers");
        query_->require(transport_->sourceBoundary()->pooledStorage()->coordinator()->plan());auto execution=gpuExecution();
        execution.requirePreparedInput(control_.owner());execution.requirePreparedOutput(control_.owner());
        if(device().is_cuda()){
#if defined(HAVE_CUDA)
            t::enqueueCUDAPooledLocalHotUnion(device(),arguments(),control_.healthy(),control_.error(),execution.nativeStream());
#else
            throw std::logic_error("Compact union requires compiled CUDA");
#endif
        }else{
#if defined(HAVE_ROCM)
            t::enqueueROCmPooledLocalHotUnion(device(),arguments(),control_.healthy(),control_.error(),execution.nativeStream());
#else
            throw std::logic_error("Compact union requires compiled HIP");
#endif
        }
        execution.publish(control_.owner());return true;
    }
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override{return StageBufferContract::build().addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");}
    std::uint64_t graphCaptureVariantSignature()const override
    {transport_->requireCurrent();control_.requireFor(device());auto h=query_->declarationIdentity();mix(h,t::unionLaunchIdentity(query_->queryBound().maximumOriginal()));mix(h,control_.captureIdentity());for(const auto& local:locals_)mix(h,local->captureIdentity());return h?h:1;}
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo result;result.addScalarInt("compact_union_consumers",locals_.size());return result;}
private:
    /** @return Only source-local aliases; remote native roots are used solely as immutable cold identities. */
    t::UnionArgs arguments()const
    {
        t::UnionArgs args{};args.expected_intent=intent_;const auto& g=*transport_->localHotGeometry();const auto& f=source_->geometry().frameOffsets();
        args.expected=address<const t::QueryIdentity>(localFrame(*source_,f.identity+offsetof(q::local_hot::FrameIdentity,query),sizeof(t::QueryIdentity)),device());
        args.source_admission=address<const t::SourceAdmissionHeader>(transport_->localHotAdmission(device()),device());
        args.canonical={address<const q::adaptive_lookup::KeyIndex>(q::PooledNativeTransferSpan(source_->roots().native,source_->geometry().scratchOffsets().keys_a,
            std::size_t{g.maximumPoolUniqueRows()}*sizeof(q::adaptive_lookup::KeyIndex)),device()),
            address<const std::uint64_t>(localFrame(*source_,f.control+offsetof(q::adaptive_lookup::FrameControl,original_ids),8),device()),query_->queryBound().canonicalOriginal()};
        args.consumer_count=static_cast<std::uint32_t>(locals_.size());args.capacity=query_->queryBound().maximumOriginal();args.output=address<t::UnionControl>(transport_->localHotUnion(),device());
        args.merged_keys=locals_.size()==1?address<std::uint64_t>(localFrame(*source_,f.miss_keys,g.keyCapacityBytes()),device()):address<std::uint64_t>(transport_->localHotSourceScratch(Transport::LocalHotSourceScratch::MergedKeys),device());
        std::size_t remote=0;
        for(std::uint32_t i=0;i<locals_.size();++i){
            const bool self=locals_[i]==source_;const auto bank=transport_->localHotControl(device(),i);
            args.consumers[i]=g.consumer(i,locals_[i]->consumerStorageIdentity());
            args.consumers[i].maximum_misses=query_->queryBound().consumerMaximum(i);
            args.requests[i].header=address<const t::RequestHeader>(bank.subspan(self?offsetof(t::Control,request_out):offsetof(t::Control,request_in),sizeof(t::RequestHeader)),device());
            args.requests[i].received_header_bytes=address<const std::uint64_t>(bank.subspan(offsetof(t::Control,header_receive_bytes),8),device());
            args.requests[i].received_key_bytes=address<const std::uint64_t>(bank.subspan(self?offsetof(t::Control,request_send_bytes):offsetof(t::Control,request_receive_bytes),8),device());
            args.requests[i].keys=self?address<const std::uint64_t>(localFrame(*source_,f.miss_keys,g.keyCapacityBytes()),device()):
                address<const std::uint64_t>(transport_->localHotSourceScratch(Transport::LocalHotSourceScratch::IncomingKeys).subspan(remote++*g.keyCapacityBytes(),g.keyCapacityBytes()),device());
        }
        return args;
    }
    std::shared_ptr<const Transport> transport_;std::vector<std::shared_ptr<Local>> locals_;std::shared_ptr<Local> source_;
    DeviceGenerationControlBinding control_;q::PooledImmutableLocalHotAccessIntent intent_;std::optional<q::PooledImmutableLocalHotPoolQuery> query_;std::optional<t::UnionArgs> args_;std::uint64_t identity_;
};
/** @return Exact actual source native roots read by the original native-row pack. */
auto sourceRoots(const q::PooledImmutableSourcePreparedStorage& storage)
{
    q::PooledImmutableNativeRoots owner{};if(storage.geometry().parameters().local_owner!=q::adaptive_lookup::None)owner=storage.owner()->roots();
    const auto source=storage.roots(),coordinator=storage.coordinator()->roots();
    return std::array<std::pair<BufferId,std::shared_ptr<TensorBase>>,8>{{
        {BufferId::QFN_POOLED_SOURCE_FRAME,source.frame},{BufferId::QFN_POOLED_SOURCE_NATIVE,source.native},
        {BufferId::QFN_POOLED_OWNER_CACHE,owner.cache},{BufferId::QFN_POOLED_OWNER_FRAME,owner.frame},{BufferId::QFN_POOLED_OWNER_NATIVE,owner.native},
        {BufferId::QFN_POOLED_COORDINATOR_POLICY,coordinator.cache},{BufferId::QFN_POOLED_COORDINATOR_FRAME,coordinator.frame},{BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator.native}}};
}
/** @brief Native-only actual source reader; no expanded FP32 destination exists. */
class CompactPack final:public IComputeStage {
public:
    CompactPack(BufferArena& arena,std::shared_ptr<const Transport> transport,std::shared_ptr<Local> local,DeviceGenerationControlBinding control)
        :IComputeStage(transport->sourceBoundary()->device()),arena_(&arena),transport_(std::move(transport)),local_(std::move(local)),
         control_(control),pack_(transport_->localHotNativePack(local_))
    {require();identity_=graphCaptureVariantSignature();}
    bool matches(const Transport* transport,const Local* local)const{return transport_.get()==transport && local_.get()==local;}
    const auto& control()const noexcept{return control_;}
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override
    {if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Native pack changed original participant/stream");require();pack_.seal();setGPUStream(stream);return true;}
    bool execute(IDeviceContext* context)override
    {
        if(!context || context->deviceId()!=device() || graphCaptureVariantSignature()!=identity_)throw std::logic_error("Native pack changed captured original source");
        require();const auto& boundary=transport_->sourceBoundary();auto execution=gpuExecution();
        execution.requirePreparedInput(control_.owner());execution.requirePreparedOutput(control_.owner());
        for(const auto& [role,root]:sourceRoots(*boundary->pooledStorage()))if(root)execution.requirePreparedInput(root.get());
        boundary->pooledKernels().enqueueCompactNativePack(*boundary->pooledStorage(),control_,pack_,execution.nativeStream());
        execution.publish(control_.owner());return true;
    }
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override
    {require();auto contract=StageBufferContract::build();for(const auto& [role,root]:sourceRoots(*transport_->sourceBoundary()->pooledStorage()))if(root)contract.addInput(role,"INT8");contract.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");return contract;}
    std::uint64_t graphCaptureVariantSignature()const override
    {require();auto h=pack_.declarationIdentity();mix(h,control_.captureIdentity());mix(h,transport_->sourceBoundary()->pooledKernels().captureIdentity(*transport_->sourceBoundary()->pooledStorage()));return h?h:1;}
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo result;result.addScalarInt("compact_native_row_bytes",transport_->localHotGeometry()->sourceExtent().rowBytes());return result;}
private:
    void require()const
    {transport_->requireCurrent();control_.requireFor(device());if(arena_->getTensor(BufferId::MTP_GENERATION_CONTROL)!=control_.owner())throw std::logic_error("Native pack substituted controller");for(const auto& [role,root]:sourceRoots(*transport_->sourceBoundary()->pooledStorage()))if(root && arena_->getTensor(role)!=root.get())throw std::logic_error("Native pack substituted original source role");}
    BufferArena* arena_;std::shared_ptr<const Transport> transport_;std::shared_ptr<Local> local_;DeviceGenerationControlBinding control_;
    q::PooledImmutableLocalHotNativePack pack_;std::uint64_t identity_;
};
/** @return Actual node with exact original device identity. */
const ComputeNode& actualNode(const ComputeGraph& graph,const std::string& name,DeviceId device)
{const auto* node=graph.getNode(name);if(!node || node->device!=device)throw std::invalid_argument("Compact completion lost an actual participant-local node: "+name);return *node;}
/** @brief Require the exact protocol edge; ordinary trunk additions belong only to main ready-wait. */
void direct(const ComputeGraph& graph,const std::string& consumer,const std::string& producer,DeviceId device)
{const auto& node=actualNode(graph,consumer,device);actualNode(graph,producer,device);if(node.dependencies!=std::vector<std::string>{producer})throw std::invalid_argument("Compact completion changed an exact protocol dependency: "+consumer);}
/** @return Whether a real ancestor exists; cycles and missing nodes fail closed. */
bool ancestor(const ComputeGraph& graph,const std::string& consumer,const std::string& producer)
{
    std::vector<std::string> pending{consumer},seen;
    while(!pending.empty()){auto name=std::move(pending.back());pending.pop_back();if(name==producer)return true;
        if(std::find(seen.begin(),seen.end(),name)!=seen.end())continue;seen.push_back(name);
        const auto* node=graph.getNode(name);if(!node)throw std::invalid_argument("Compact dependency names an absent node");
        pending.insert(pending.end(),node->dependencies.begin(),node->dependencies.end());}
    return false;
}
/** @return Actual native endpoint, with no metadata/name-only substitute accepted. */
const CompactTransfer& wireNode(const ComputeGraph& graph,const std::string& name,const std::shared_ptr<const Transport>& transport,
    const std::shared_ptr<Local>& local,CompactWire wire,CapturedTransferEndpoint side)
{
    const auto reader=originalReader(*transport,localCoordinate(*transport,local));const auto& node=actualNode(graph,name,wireDevice(*reader,wire,side));
    const auto* stage=dynamic_cast<const CompactTransfer*>(node.stage.get());
    if(!stage || !stage->matches(transport.get(),local.get(),wire,side))throw std::invalid_argument("Compact completion substituted an original native endpoint");return *stage;
}
/** @return Actual checked native metadata projection. */
const CompactMetadata& metadataNode(const ComputeGraph& graph,const std::string& name,const std::shared_ptr<const Transport>& transport,
    const std::shared_ptr<Local>& local,t::MetadataAction action,DeviceId device,
    q::PooledImmutableLocalHotAccessIntent intent=q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)
{
    const auto& node=actualNode(graph,name,device);const auto* stage=dynamic_cast<const CompactMetadata*>(node.stage.get());
    if(!stage || !stage->matches(transport.get(),local.get(),action) || stage->intent()!=intent)throw std::invalid_argument("Compact completion substituted its actual metadata producer");return *stage;
}
/** @brief Add a stage and its exact predecessor, rejecting all collisions before replacement. */
void add(ComputeGraph& graph,const std::string& name,std::unique_ptr<IComputeStage> stage,DeviceId device,const std::string& predecessor)
{
    if(name.empty() || graph.getNode(name))throw std::invalid_argument("Compact graph would replace an actual stage");
    graph.addNode(name,std::move(stage),device);if(!predecessor.empty())graph.addDependency(name,predecessor);
}
} // namespace
void requirePooledCompactQueryProducer(const ComputeGraph& graph,const q::PooledImmutableLocalHotPoolQuery& query,const std::string& beginName)
{
    const auto* node=graph.getNode(beginName);const auto* begin=node?dynamic_cast<const PooledImmutableLocalHotBeginStage*>(node->stage.get()):nullptr;
    if(!begin || node->device!=begin->source()->geometry().allocator().device || begin->query().declarationIdentity()!=query.declarationIdentity() || node->dependencies.size()!=1)
        throw std::invalid_argument("Compact Begin lost its actual closed original query producer");
    const auto& producer=actualNode(graph,node->dependencies.front(),node->device);const auto* prepared=dynamic_cast<const CompactUnion*>(producer.stage.get());
    if(!prepared || prepared->query().declarationIdentity()!=query.declarationIdentity() ||
        prepared->transport()->sourceBoundary()->pooledStorage()!=begin->source())
        throw std::invalid_argument("Compact Begin does not follow its exact canonical union seal");
    const auto transport=prepared->transport();const auto device=node->device;
    for(const auto& local:prepared->locals()){
        const bool self=local==prepared->source();const auto action=self?t::MetadataAction::PrepareRequest:t::MetadataAction::ConfirmRequestKeys;
        const CompactMetadata* request=nullptr;std::string terminal;
        for(const auto& name:graph.getExecutionOrder()){
            const auto* candidate=graph.getNode(name);const auto* stage=candidate?dynamic_cast<const CompactMetadata*>(candidate->stage.get()):nullptr;
            if(stage && candidate->device==device && stage->matches(transport.get(),local.get(),action)){
                if(request)throw std::invalid_argument("Compact union has duplicate original request producers");request=stage;terminal=name;
            }
        }
        if(!request || request->intent()!=prepared->intent() || request->sourceLocal()!=prepared->source() || request->control().captureIdentity()!=prepared->control().captureIdentity() ||
            !ancestor(graph,node->dependencies.front(),terminal))throw std::invalid_argument("Compact union omitted an actual current request/native receipt");
        if(!self){
            const auto& confirmed=actualNode(graph,terminal,device);
            if(confirmed.dependencies.size()!=1)throw std::invalid_argument("Compact request receipt lost its actual native payload edge");
            const auto& payload=wireNode(graph,confirmed.dependencies.front(),transport,local,CompactWire::MissKeys,CapturedTransferEndpoint::Consumer);
            if(request->payload()!=&payload)throw std::invalid_argument("Compact request confirmation substituted its native completion");
            const auto& received=actualNode(graph,confirmed.dependencies.front(),device);
            if(received.dependencies.size()!=1)throw std::invalid_argument("Compact request keys precede fixed-header validation");
            const auto& header=metadataNode(graph,received.dependencies.front(),transport,local,t::MetadataAction::AcceptRequestHeader,device,prepared->intent());
            if(header.sourceLocal()!=prepared->source() || header.control().captureIdentity()!=prepared->control().captureIdentity())
                throw std::invalid_argument("Compact header changed canonical source/controller");
            const auto& accepted=actualNode(graph,received.dependencies.front(),device);
            if(accepted.dependencies.size()!=1)throw std::invalid_argument("Compact header validation omitted its actual native receive");
            wireNode(graph,accepted.dependencies.front(),transport,local,CompactWire::RequestHeader,CapturedTransferEndpoint::Consumer);
        }
    }
}
void requirePooledCompactConsumerCompletion(const ComputeGraph& graph,const PooledImmutableCompactConsumerCompletion& completion)
{
    const auto& transport=completion.transport;const auto& local=completion.local;
    if(!transport || !local)throw std::invalid_argument("Compact consumer completion lost original owners");
    const auto coordinate=localCoordinate(*transport,local);const auto device=local->geometry().allocator().device;
    if(device==transport->localHotGeometry()->coordinator().device)throw std::invalid_argument("Source-self does not forge a remote consumer ACK");
    wireNode(graph,completion.requested,transport,local,CompactWire::MissKeys,CapturedTransferEndpoint::Producer);
    wireNode(graph,completion.header_received,transport,local,CompactWire::ResultHeader,CapturedTransferEndpoint::Consumer);
    const auto& payload=wireNode(graph,completion.payload_received,transport,local,CompactWire::NativeRows,CapturedTransferEndpoint::Consumer);
    const auto& received=metadataNode(graph,completion.received,transport,local,t::MetadataAction::ConfirmResult,device,completion.intent);
    if(received.payload()!=&payload)throw std::invalid_argument("Compact consumer lost its exact whole native payload receipt");
    direct(graph,completion.received,completion.payload_received,device);
    const auto& payloadNode=actualNode(graph,completion.payload_received,device);
    if(payloadNode.dependencies.size()!=1)throw std::invalid_argument("Compact payload lost fixed-header admission");
    metadataNode(graph,payloadNode.dependencies.front(),transport,local,t::MetadataAction::AcceptResultHeader,device,completion.intent);
    direct(graph,payloadNode.dependencies.front(),completion.header_received,device);
    if(!ancestor(graph,completion.header_received,completion.requested))throw std::invalid_argument("Compact result precedes its original request");
    const auto& acceptedNode=actualNode(graph,completion.accepted,device);const auto* accepted=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(acceptedNode.stage.get());
    const auto expected=q::PooledImmutableLocalHotOperands::result(*local,transport->localHotGeometry()->consumer(coordinate,local->consumerStorageIdentity()),
        transport->localHotOriginalRequest(device),received.control());
    if(!accepted || accepted->storage()!=local || accepted->action()!=Action::AcceptResult ||
        accepted->operands().declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Compact consumer substituted actual own-request acquisition");
    direct(graph,completion.accepted,completion.received,device);
    wireNode(graph,completion.ack_sent,transport,local,CompactWire::Ack,CapturedTransferEndpoint::Producer);
    direct(graph,completion.ack_sent,completion.accepted,device);
}
PooledImmutableCompactConsumerCompletion findPooledCompactConsumerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const Transport>& transport,const std::shared_ptr<Local>& local)
{
    if(!transport || !local)throw std::invalid_argument("Compact consumer inspection needs its actual owners");
    PooledImmutableCompactConsumerCompletion result;result.transport=transport;result.local=local;
    const auto assign=[&](std::string& destination,const std::string& name){if(!destination.empty())throw std::invalid_argument("Compact graph has duplicate original consumer endpoints");destination=name;};
    for(const auto& name:graph.getExecutionOrder()){
        const auto* node=graph.getNode(name);if(!node)continue;
        if(const auto* wire=dynamic_cast<const CompactTransfer*>(node->stage.get())){
            if(wire->matches(transport.get(),local.get(),CompactWire::MissKeys,CapturedTransferEndpoint::Producer))assign(result.requested,name);
            if(wire->matches(transport.get(),local.get(),CompactWire::ResultHeader,CapturedTransferEndpoint::Consumer))assign(result.header_received,name);
            if(wire->matches(transport.get(),local.get(),CompactWire::NativeRows,CapturedTransferEndpoint::Consumer))assign(result.payload_received,name);
            if(wire->matches(transport.get(),local.get(),CompactWire::Ack,CapturedTransferEndpoint::Producer))assign(result.ack_sent,name);
        }
        if(const auto* metadata=dynamic_cast<const CompactMetadata*>(node->stage.get()))
            if(metadata->matches(transport.get(),local.get(),t::MetadataAction::ConfirmResult)){assign(result.received,name);result.intent=metadata->intent();}
        if(const auto* phase=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node->stage.get()))
            if(phase->storage()==local && phase->action()==Action::AcceptResult)assign(result.accepted,name);
    }
    requirePooledCompactConsumerCompletion(graph,result);return result;
}
std::shared_ptr<const PooledImmutableCompactReadCompletion> appendPooledCompactNativeDelivery(
    ComputeGraph& graph,BufferArena& arena,PooledImmutableCompactReadRequest request,DeviceGenerationControlBinding control,
    const std::string& nativeRowsReady,const std::string& identity)
{
    if(!request.transport || !request.source_code || identity.empty())throw std::invalid_argument("Compact delivery requires original transport/source code");
    const auto transport=request.transport;const auto& geometry=transport->localHotGeometry();
    if(!geometry || request.consumers.size()!=geometry->consumers().size())throw std::invalid_argument("Compact delivery omitted actual consumers");
    const auto device=geometry->coordinator().device;actualNode(graph,nativeRowsReady,device);control.requireFor(device);
    std::size_t self=request.consumers.size();
    for(std::uint32_t i=0;i<request.consumers.size();++i){if(localCoordinate(*transport,request.consumers[i])!=i)throw std::invalid_argument("Compact delivery reordered actual consumers");if(request.consumers[i]->geometry().allocator()==geometry->coordinator())self=i;}
    if(self==request.consumers.size() || request.source_code->backend()!=device.type)throw std::invalid_argument("Compact delivery lost original source consumer/code");
    if(!q::validLocalHotAccessIntent(request.intent))throw std::invalid_argument("Compact delivery has unknown immutable purpose");
    auto result=std::make_shared<PooledImmutableCompactReadCompletion>();result->original=std::move(request);result->native_rows_ready=nativeRowsReady;
    const auto count=result->original.consumers.size();for(auto* names:{&result->packed,&result->header_sent,&result->payload_sent,&result->ack_received,&result->ack_validated})names->resize(count);
    std::vector<const CompactTransfer*> payloads(count);std::string previous=nativeRowsReady;
    // Source-self publishes its private result first. Other consumers cannot
    // delay source model readiness merely by reaching their ACK later.
    std::vector<std::size_t> order{self};for(std::size_t i=0;i<count;++i)if(i!=self)order.push_back(i);
    for(const auto i:order){const auto& local=result->original.consumers[i];const auto name=identity+"/consumer"+std::to_string(i)+"/pack";result->packed[i]=name;
        add(graph,name,std::make_unique<CompactPack>(arena,transport,local,control),device,previous);previous=name;
        if(i==self){
            const auto accepted=identity+"/source_accept";
            const auto operands=q::PooledImmutableLocalHotOperands::result(*local,geometry->consumer(i,local->consumerStorageIdentity()),transport->localHotOriginalRequest(device),control);
            add(graph,accepted,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,result->original.source_code,Action::AcceptResult,operands),device,previous);previous=accepted;
        }
    }
    // Submit every payload before the first ACK receive, preserving one slow
    // reader's independence from all other needed native deliveries.
    for(const auto i:order)if(i!=self){const auto& local=result->original.consumers[i];const auto prefix=identity+"/consumer"+std::to_string(i);
        result->header_sent[i]=prefix+"/header_send";result->payload_sent[i]=prefix+"/rows_send";
        add(graph,result->header_sent[i],std::make_unique<CompactTransfer>(arena,transport,local,control,CompactWire::ResultHeader,CapturedTransferEndpoint::Producer),device,previous);
        auto payload=std::make_unique<CompactTransfer>(arena,transport,local,control,CompactWire::NativeRows,CapturedTransferEndpoint::Producer);payloads[i]=payload.get();
        add(graph,result->payload_sent[i],std::move(payload),device,result->header_sent[i]);previous=result->payload_sent[i];
    }
    for(const auto i:order)if(i!=self){const auto& local=result->original.consumers[i];const auto prefix=identity+"/consumer"+std::to_string(i);
        result->ack_received[i]=prefix+"/ack_receive";result->ack_validated[i]=prefix+"/ack_validate";
        add(graph,result->ack_received[i],std::make_unique<CompactTransfer>(arena,transport,local,control,CompactWire::Ack,CapturedTransferEndpoint::Consumer),device,previous);
        add(graph,result->ack_validated[i],std::make_unique<CompactMetadata>(arena,transport,local,result->original.consumers[self],control,t::MetadataAction::ConfirmAck,payloads[i],result->original.intent),device,result->ack_received[i]);
        previous=result->ack_validated[i];
    }
    result->terminal=previous;requirePooledCompactNativeDelivery(graph,*result);return result;
}
void requirePooledCompactNativeDelivery(const ComputeGraph& graph,const PooledImmutableCompactReadCompletion& completion)
{
    const auto& request=completion.original;const auto& transport=request.transport;
    if(!transport || !transport->localHotGeometry() || !request.source_code)throw std::invalid_argument("Native delivery lost actual original owners");
    const auto& geometry=*transport->localHotGeometry();const auto count=geometry.consumers().size();const auto device=geometry.coordinator().device;
    if(request.consumers.size()!=count || completion.packed.size()!=count || completion.header_sent.size()!=count ||
        completion.payload_sent.size()!=count || completion.ack_received.size()!=count || completion.ack_validated.size()!=count)
        throw std::invalid_argument("Native delivery lost exact consumer population");
    std::size_t self=count;for(std::uint32_t i=0;i<count;++i){if(localCoordinate(*transport,request.consumers[i])!=i)throw std::invalid_argument("Native delivery substituted original local roots");if(request.consumers[i]->geometry().allocator()==geometry.coordinator())self=i;}
    if(self==count)throw std::invalid_argument("Native delivery omitted source-self");
    const auto& ready=actualNode(graph,completion.native_rows_ready,device);const auto* sealed=dynamic_cast<const PooledImmutableSourcePhaseStage*>(ready.stage.get());
    const auto* phase=dynamic_cast<const PooledImmutableNativePhaseStage*>(ready.stage.get());const auto* original=phase?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&phase->phase()):nullptr;
    if(!(sealed && sealed->matchesDemand(transport->sourceBoundary().get(),q::PooledSourceAction::SealReads)) &&
        !(original && original->storage==transport->sourceBoundary()->pooledStorage()->coordinator() && original->action==q::pooled_native::CoordinatorAction::Publish &&
            transport->sourceBoundary()->pooledStorage()->geometry().parameters().coordinator.policy.participants()==1))
        throw std::invalid_argument("Native delivery has no actual original sealed read image");
    std::vector<std::size_t> order{self};for(std::size_t i=0;i<count;++i)if(i!=self)order.push_back(i);
    std::string previous=completion.native_rows_ready;
    for(const auto i:order){
        const auto& node=actualNode(graph,completion.packed[i],device);const auto* pack=dynamic_cast<const CompactPack*>(node.stage.get());
        if(!pack || !pack->matches(transport.get(),request.consumers[i].get()))throw std::invalid_argument("Native delivery substituted actual native source reader");
        direct(graph,completion.packed[i],previous,device);previous=completion.packed[i];
        if(i==self){
            std::string accepted;
            for(const auto& name:graph.getExecutionOrder()){const auto* candidate=graph.getNode(name);const auto* local=candidate?dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(candidate->stage.get()):nullptr;
                if(local && local->storage()==request.consumers[i] && local->action()==Action::AcceptResult){if(!accepted.empty())throw std::invalid_argument("Source-self has duplicate native result acquisitions");accepted=name;}}
            if(accepted.empty())throw std::invalid_argument("Source-self native result never becomes locally ready");
            const auto* acquisition=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(graph.getNode(accepted)->stage.get());
            const auto expected=q::PooledImmutableLocalHotOperands::result(*request.consumers[i],geometry.consumer(i,request.consumers[i]->consumerStorageIdentity()),
                transport->localHotOriginalRequest(device),pack->control());
            if(acquisition->operands().declarationIdentity()!=expected.declarationIdentity() || acquisition->codeIdentity()!=request.source_code->codeIdentity())
                throw std::invalid_argument("Source-self acquisition changed original own-request/code identity");
            direct(graph,accepted,previous,device);previous=accepted;
            if(!completion.header_sent[i].empty() || !completion.payload_sent[i].empty() || !completion.ack_received[i].empty() || !completion.ack_validated[i].empty())throw std::invalid_argument("Native source-self invented remote traffic");
        }
    }
    for(const auto i:order)if(i!=self){
        wireNode(graph,completion.header_sent[i],transport,request.consumers[i],CompactWire::ResultHeader,CapturedTransferEndpoint::Producer);
        wireNode(graph,completion.payload_sent[i],transport,request.consumers[i],CompactWire::NativeRows,CapturedTransferEndpoint::Producer);
        direct(graph,completion.header_sent[i],previous,device);direct(graph,completion.payload_sent[i],completion.header_sent[i],device);previous=completion.payload_sent[i];
    }
    for(const auto i:order)if(i!=self){
        wireNode(graph,completion.ack_received[i],transport,request.consumers[i],CompactWire::Ack,CapturedTransferEndpoint::Consumer);
        const auto& ack=metadataNode(graph,completion.ack_validated[i],transport,request.consumers[i],t::MetadataAction::ConfirmAck,device,request.intent);
        const auto& payload=wireNode(graph,completion.payload_sent[i],transport,request.consumers[i],CompactWire::NativeRows,CapturedTransferEndpoint::Producer);
        if(ack.payload()!=&payload)throw std::invalid_argument("Native ACK lost its actual source payload completion");
        direct(graph,completion.ack_received[i],previous,device);direct(graph,completion.ack_validated[i],completion.ack_received[i],device);previous=completion.ack_validated[i];
    }
    if(completion.terminal!=previous)throw std::invalid_argument("Native delivery omitted a real last reader/ACK");
}
void validatePooledLocalHotDemandFrontier(const ComputeGraph& graph,const PooledImmutableLocalHotDemandFrontier& frontier)
{
    const auto& probeNode=actualNode(graph,frontier.published,frontier.participant);
    const auto& waitNode=actualNode(graph,frontier.consume,frontier.participant);
    const auto& readNode=actualNode(graph,frontier.terminal,frontier.participant);
    const auto* probe=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(probeNode.stage.get());
    const auto* wait=dynamic_cast<const PooledImmutableLocalHotReadyStage*>(waitNode.stage.get());
    const auto* read=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(readNode.stage.get());
    if(frontier.published==frontier.consume || frontier.published==frontier.terminal || frontier.consume==frontier.terminal ||
        !probe || probe->action()!=Action::Probe || !wait || wait->frontier()!=PooledImmutableLocalHotReadyStage::Frontier::DataReady ||
        !read || read->action()!=Action::LocalRead || probe->storage()!=wait->storage() || probe->storage()!=read->storage() ||
        probe->operands().controllerIdentity()!=read->operands().controllerIdentity() ||
        std::find(waitNode.dependencies.begin(),waitNode.dependencies.end(),frontier.published)==waitNode.dependencies.end())
        throw std::invalid_argument("Local-hot demand frontier substituted actual Probe/ready/LocalRead roots");
    direct(graph,frontier.terminal,frontier.consume,frontier.participant);
}
void bindPooledLocalHotDemandConsumption(ComputeGraph& graph,const PooledImmutableLocalHotDemandFrontier& frontier,const std::vector<std::string>& trunk)
{
    validatePooledLocalHotDemandFrontier(graph,frontier);std::set<std::string> unique;
    for(const auto& name:trunk){actualNode(graph,name,frontier.participant);
        if(name==frontier.published || name==frontier.consume || name==frontier.terminal || !unique.insert(name).second ||
            !ancestor(graph,name,frontier.published) || ancestor(graph,name,frontier.consume))
            throw std::invalid_argument("Local-hot demand trunk is duplicate, foreign or creates a response cycle");}
    if(trunk.empty())throw std::invalid_argument("Local-hot demand consumption requires actual model trunk producers");
    for(const auto& name:trunk)graph.addDependency(frontier.consume,name);
}
PooledImmutableLocalHotDemandHandoff::PooledImmutableLocalHotDemandHandoff(ComputeGraph& graph,
    PooledImmutableLocalHotDemandFrontier frontier,std::uint64_t query,std::uint64_t hash)
    :graph_(&graph),boundary_(graph.retainedStorageBoundary()),domain_(graph.retainedParticipantDefinitionDomain()),
     frontier_(std::move(frontier)),query_(query),hash_(hash)
{
    if(!query_ || !hash_ || (!boundary_ && !domain_))throw std::invalid_argument("Local-hot demand handoff lost original source/reader/query");
    validatePooledLocalHotDemandFrontier(graph,frontier_);
}
void PooledImmutableLocalHotDemandHandoff::consume(ComputeGraph& graph,std::uint64_t query,std::uint64_t hash,const std::vector<std::string>& trunk)
{
    const auto boundary=graph.retainedStorageBoundary();
    if(consumed_ || &graph!=graph_ || boundary!=boundary_ || boundary.owner_before(boundary_) || boundary_.owner_before(boundary) ||
        graph.retainedParticipantDefinitionDomain()!=domain_ || query!=query_ || hash!=hash_)
        throw std::invalid_argument("Local-hot demand consumption changed original graph/source/query or was repeated");
    bindPooledLocalHotDemandConsumption(graph,frontier_,trunk);consumed_=true;
}
void PooledImmutableLocalHotDemandHandoff::requireConsumed()const
{if(!consumed_)throw std::logic_error("Local-hot current demand remains unconsumed");}
namespace {
/** @brief Append actual post-LocalRead policy/fill/retirement, independent of further model computation. */
std::pair<std::string,std::string> appendLocalMaintenance(ComputeGraph& graph,BufferArena& arena,
    const std::shared_ptr<Local>& local,const std::shared_ptr<const q::IPooledImmutableLocalHotKernels>& code,
    t::ConsumerGeometry geometry,DeviceGenerationControlBinding control,const std::string& predecessor,const std::string& identity)
{
    const auto device=local->geometry().allocator().device;const auto wait=identity+"/read_complete";
    add(graph,wait,std::make_unique<PooledImmutableLocalHotReadyStage>(local,PooledImmutableLocalHotReadyStage::Frontier::LocalReadComplete),device,predecessor);
    std::string previous=wait,consumed;
    for(const auto& [action,suffix]:std::array<std::pair<Action,const char*>,4>{{{Action::ConsumeDemand,"consume"},{Action::CopyFills,"copy_fills"},{Action::CompleteFills,"fills_complete"},{Action::Retire,"retired"}}}){
        const auto name=identity+"/"+suffix;add(graph,name,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,code,action,
            q::PooledImmutableLocalHotOperands::frame(*local,geometry,control)),device,previous);previous=name;if(action==Action::ConsumeDemand)consumed=name;
    }
    return {consumed,previous};
}
/** @brief Publish a complete participant-local range, excluding all model-ready stages. */
RetainedGraphParticipantAuxiliaryRange auxiliaryRange(const ComputeGraph& graph,const std::set<std::string>& before,
    const PooledImmutableLocalHotGraphEndpoints& endpoints,RetainedGraphParticipantAuxiliaryRole role,DeviceId device)
{
    RetainedGraphParticipantAuxiliaryRange range;range.role=role;range.participant=device;
    range.probe_ready=endpoints.probe;range.main_read=endpoints.read;range.ready_publication=endpoints.auxiliary.ready_publication;range.retirement=endpoints.retired;
    for(const auto& name:graph.getExecutionOrder())if(!before.contains(name) && name!=endpoints.probe && name!=endpoints.ready_wait && name!=endpoints.read)range.nodes.push_back(name);
    std::uint64_t h=0x51464e4c31415558ULL;mix(h,static_cast<unsigned>(role));mix(h,static_cast<unsigned>(device.type));mix(h,device.ordinal);
    for(const auto& name:range.nodes){for(const auto byte:name)mix(h,static_cast<unsigned char>(byte));const auto* node=graph.getNode(name);mix(h,node->stage->graphCaptureVariantSignature());}
    range.identity=h?h:1;return range;
}
/** @brief Declare independent Probe/main-ready/output before auxiliary owner work is assembled. */
PooledImmutableLocalHotGraphEndpoints appendLocalMain(ComputeGraph& graph,BufferArena& arena,
    const std::shared_ptr<Local>& local,const std::shared_ptr<const q::IPooledImmutableLocalHotKernels>& code,
    t::ConsumerGeometry geometry,DeviceGenerationControlBinding control,DeviceRowRange tokens,q::PooledNativeTensorSpan ids,
    std::optional<q::PooledNativeTensorSpan> count,q::PooledNativeTensorSpan output,const std::string& identity,const std::string& outputIdentity)
{
    if(!code || code->backend()!=local->geometry().allocator().device.type || identity.empty() || outputIdentity.empty())throw std::invalid_argument("Local-hot main graph lost original code/output identity");
    PooledImmutableLocalHotGraphEndpoints e;e.probe=identity+"/local_probe";e.ready_wait=identity+"/local_ready";e.read=outputIdentity;e.first=e.probe;e.last=e.read;
    const auto device=local->geometry().allocator().device;
    add(graph,e.probe,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,code,Action::Probe,
        q::PooledImmutableLocalHotOperands::query(*local,geometry,tokens,std::move(ids),std::move(count),control,q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)),device,{});
    add(graph,e.ready_wait,std::make_unique<PooledImmutableLocalHotReadyStage>(local,PooledImmutableLocalHotReadyStage::Frontier::DataReady),device,e.probe);
    add(graph,e.read,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,code,Action::LocalRead,
        q::PooledImmutableLocalHotOperands::read(*local,geometry,std::move(output),control)),device,e.ready_wait);
    e.demand={e.probe,e.ready_wait,e.read,device};return e;
}
} // namespace
/**
 * @brief Lower the same real source protocol with an immutable current or lookup-only purpose.
 * @param graph Original graph. @param arena Same prepared arena. @param actual Original transport.
 * @param originalLocals Complete local-root roster. @param poolCode Original pool code.
 * @param local Actual coordinator local root. @param localCode Same local code.
 * @param tokens Original token extent. @param ids Actual ordered hashes. @param count Optional original count.
 * @param control Same request controller. @param output Sole Current output; absent for Future.
 * @param identity Unique stage namespace. @param outputIdentity Current output name; empty for Future.
 * @param intent Immutable source/wire purpose, independent of later actual-demand adoption.
 * @return Actual stages and source retirement descriptor, without a model SourcePlan or history authority.
 */
static PooledImmutableLocalHotSourceBody appendLocalHotSourceBody(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const Transport> actual,std::span<const std::shared_ptr<Local>> originalLocals,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode,std::shared_ptr<Local> local,
    std::shared_ptr<const q::IPooledImmutableLocalHotKernels> localCode,DeviceRowRange tokens,q::PooledNativeTensorSpan ids,
    std::optional<q::PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,q::PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity,
    q::PooledImmutableLocalHotAccessIntent intent)
{
    if(!actual)throw std::invalid_argument("Combined PLE source lost its original transport");
    actual->requireCurrent();const auto& geometry=actual->localHotGeometry();
    if(!geometry || originalLocals.size()!=geometry->consumers().size())
        throw std::invalid_argument("Combined PLE source requires every original local root");
    for(std::uint32_t i=0;i<originalLocals.size();++i)
        if(localCoordinate(*actual,originalLocals[i])!=i)
            throw std::invalid_argument("Combined PLE source changed original local-root order");
    if(!geometry || localCoordinate(*actual,local)>=geometry->consumers().size() || local->geometry().allocator()!=geometry->coordinator())
        throw std::invalid_argument("Combined PLE source requires its actual source local roots");
    std::set<std::string> before;for(const auto& name:graph.getExecutionOrder())before.insert(name);
    graph.setRetainedParticipantDefinitionDomain({geometry->coordinator().device,actual.get()});
    const auto coordinate=localCoordinate(*actual,local);
    PooledImmutableLocalHotGraphEndpoints endpoints;
    if(intent==q::PooledImmutableLocalHotAccessIntent::CurrentInvocation)
        endpoints=appendLocalMain(graph,arena,local,localCode,
            geometry->consumer(coordinate,local->consumerStorageIdentity()),control,tokens,std::move(ids),std::move(count),std::move(output),identity,outputIdentity);
    else if(intent==q::PooledImmutableLocalHotAccessIntent::FutureReadAhead){
        endpoints.probe=identity+"/future_probe";endpoints.first=endpoints.probe;
        add(graph,endpoints.probe,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,localCode,Action::Probe,
            q::PooledImmutableLocalHotOperands::query(*local,geometry->consumer(coordinate,local->consumerStorageIdentity()),
                tokens,std::move(ids),std::move(count),control,intent)),geometry->coordinator().device,{});
    }else throw std::invalid_argument("Local source body has unknown immutable purpose");
    auto previous=identity+"/own_request";
    add(graph,previous,std::make_unique<CompactMetadata>(arena,actual,local,local,control,t::MetadataAction::PrepareRequest,nullptr,intent),local->geometry().allocator().device,endpoints.probe);
    std::vector<std::shared_ptr<Local>> locals(originalLocals.begin(),originalLocals.end());
    for(std::uint32_t i=0;i<locals.size();++i)if(locals[i]!=local){
        const auto prefix=identity+"/request"+std::to_string(i),header=prefix+"/header",accepted=prefix+"/accepted",keys=prefix+"/keys",received=prefix+"/received";
        add(graph,header,std::make_unique<CompactTransfer>(arena,actual,locals[i],control,CompactWire::RequestHeader,CapturedTransferEndpoint::Consumer),geometry->coordinator().device,previous);
        add(graph,accepted,std::make_unique<CompactMetadata>(arena,actual,locals[i],local,control,t::MetadataAction::AcceptRequestHeader,nullptr,intent),geometry->coordinator().device,header);
        auto payload=std::make_unique<CompactTransfer>(arena,actual,locals[i],control,CompactWire::MissKeys,CapturedTransferEndpoint::Consumer);const auto* receipt=payload.get();
        add(graph,keys,std::move(payload),geometry->coordinator().device,accepted);
        add(graph,received,std::make_unique<CompactMetadata>(arena,actual,locals[i],local,control,t::MetadataAction::ConfirmRequestKeys,receipt,intent),geometry->coordinator().device,keys);previous=received;
    }
    const auto unionReady=identity+"/compact_union";
    auto unionStage=std::make_unique<CompactUnion>(actual,locals,tokens,control,intent);auto query=unionStage->query();
    add(graph,unionReady,std::move(unionStage),geometry->coordinator().device,previous);
    const auto pool=appendPooledImmutableCompactLookupGraph(graph,arena,actual->sourceBoundary(),actual,poolCode,std::move(query),
        {actual,locals,localCode,intent},control,unionReady,identity+"/pool");
    for(const auto& name:graph.getExecutionOrder()){const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node->stage.get()):nullptr;
        if(stage && stage->storage()==local && stage->action()==Action::AcceptResult){if(!endpoints.auxiliary.ready_publication.empty())throw std::logic_error("Combined source has duplicate local result acquisitions");endpoints.auxiliary.ready_publication=name;}}
    if(intent==q::PooledImmutableLocalHotAccessIntent::CurrentInvocation){
        const auto maintenance=appendLocalMaintenance(graph,arena,local,localCode,geometry->consumer(coordinate,local->consumerStorageIdentity()),control,pool.source.last,identity+"/local");
        endpoints.consumed=maintenance.first;endpoints.retired=maintenance.second;
    }else endpoints.last=endpoints.retired=pool.source.last;
    endpoints.auxiliary=auxiliaryRange(graph,before,endpoints,RetainedGraphParticipantAuxiliaryRole::Source,geometry->coordinator().device);
    if(intent==q::PooledImmutableLocalHotAccessIntent::FutureReadAhead){
        endpoints.auxiliary.lifecycle=RetainedGraphParticipantAuxiliaryLifecycle::FutureProducer;
        mix(endpoints.auxiliary.identity,static_cast<unsigned>(endpoints.auxiliary.lifecycle));
    }
    return {std::move(endpoints),pool.retirement};
}
PooledImmutableLocalHotSourceBody appendPooledImmutableLocalHotSourceBody(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const Transport> transport,std::span<const std::shared_ptr<Local>> locals,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode,std::shared_ptr<Local> local,
    std::shared_ptr<const q::IPooledImmutableLocalHotKernels> localCode,DeviceRowRange tokens,q::PooledNativeTensorSpan ids,
    std::optional<q::PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,q::PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity)
{
    return appendLocalHotSourceBody(graph,arena,std::move(transport),locals,std::move(poolCode),std::move(local),
        std::move(localCode),tokens,std::move(ids),std::move(count),control,std::move(output),identity,outputIdentity,
        q::PooledImmutableLocalHotAccessIntent::CurrentInvocation);
}
PooledImmutableLocalHotFutureSourceBody appendPooledImmutableLocalHotFutureSourceBody(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const Transport> transport,std::span<const std::shared_ptr<Local>> locals,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode,std::shared_ptr<Local> local,
    std::shared_ptr<const q::IPooledImmutableLocalHotKernels> localCode,DeviceRowRange tokens,q::PooledNativeTensorSpan ids,
    std::optional<q::PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,const std::string& identity)
{
    auto body=appendLocalHotSourceBody(graph,arena,std::move(transport),locals,std::move(poolCode),std::move(local),
        std::move(localCode),tokens,std::move(ids),std::move(count),control,{},identity,{},
        q::PooledImmutableLocalHotAccessIntent::FutureReadAhead);
    return {body.local.probe,body.local.auxiliary.ready_publication,body.local.retired,
        std::move(body.pool_retirement),std::move(body.local.auxiliary)};
}

namespace qwen4exp {
PooledImmutableLocalHotGraphEndpoints PooledImmutableRetainedParticipants::appendLocalHotPLE(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const IPooledImmutableNativeKernels> poolCode,std::shared_ptr<Local> local,
    std::shared_ptr<const IPooledImmutableLocalHotKernels> localCode,DeviceRowRange tokens,PooledNativeTensorSpan ids,
    std::optional<PooledNativeTensorSpan> count,DeviceGenerationControlBinding control,PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity)
{
    requireCurrent();
    auto body=appendPooledImmutableLocalHotSourceBody(graph,arena,transport(),localHotParticipants(),std::move(poolCode),
        std::move(local),std::move(localCode),tokens,std::move(ids),std::move(count),control,std::move(output),identity,outputIdentity);
    bindSourcePlan(graph,arena,std::move(body.pool_retirement));
    body.local.auxiliary.original_execution=originalExecutionPolicy();
    graph.setRetainedParticipantAuxiliaryRange(body.local.auxiliary);
    return std::move(body.local);
}
PooledImmutableLocalHotGraphEndpoints PooledImmutableRetainedParticipants::appendLocalHotReaderBody(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<Local> local,std::shared_ptr<const IPooledImmutableLocalHotKernels> localCode,
    DeviceGenerationControlBinding control,DeviceRowRange tokens,PooledNativeTensorSpan ids,PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity,std::optional<PooledNativeTensorSpan> count,
    PooledImmutableLocalHotAccessIntent intent)
{
    requireCurrent();const auto actual=transport();const auto& geometry=localHotGeometry();const auto coordinate=localCoordinate(*actual,local);
    if(!geometry || local->geometry().allocator()==geometry->coordinator())throw std::invalid_argument("Combined reader cannot impersonate original source");
    const auto device=local->geometry().allocator().device;std::set<std::string> before;for(const auto& name:graph.getExecutionOrder())before.insert(name);
    graph.setRetainedParticipantDefinitionDomain({geometry->coordinator().device,actual.get()});
    PooledImmutableLocalHotGraphEndpoints endpoints;
    if(intent==PooledImmutableLocalHotAccessIntent::CurrentInvocation)
        endpoints=appendLocalMain(graph,arena,local,localCode,geometry->consumer(coordinate,local->consumerStorageIdentity()),control,tokens,
            std::move(ids),std::move(count),std::move(output),identity,outputIdentity);
    else if(intent==PooledImmutableLocalHotAccessIntent::FutureReadAhead){
        endpoints.probe=identity+"/future_probe";endpoints.first=endpoints.probe;
        add(graph,endpoints.probe,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,localCode,Action::Probe,
            PooledImmutableLocalHotOperands::query(*local,geometry->consumer(coordinate,local->consumerStorageIdentity()),
                tokens,std::move(ids),std::move(count),control,intent)),device,{});
    }else throw std::invalid_argument("Local reader has unknown immutable purpose");
    const auto request=identity+"/request",headerSent=identity+"/request_header_send",keysSent=identity+"/miss_keys_send";
    add(graph,request,std::make_unique<CompactMetadata>(arena,actual,local,std::shared_ptr<Local>{},control,t::MetadataAction::PrepareRequest,nullptr,intent),device,endpoints.probe);
    add(graph,headerSent,std::make_unique<CompactTransfer>(arena,actual,local,control,CompactWire::RequestHeader,CapturedTransferEndpoint::Producer),device,request);
    add(graph,keysSent,std::make_unique<CompactTransfer>(arena,actual,local,control,CompactWire::MissKeys,CapturedTransferEndpoint::Producer),device,headerSent);
    std::string previous=keysSent;const PooledImmutableRemoteTransport* physical=nullptr;
    const auto ownerCode=localHotInlineOwnerCode(device);
    if(ownerCode){
        const auto& owners=boundary()->pooledStorage()->originalOwners();
        for(std::uint32_t i=0;i<owners.size();++i)if(owners[i]->plan().allocator()==local->geometry().allocator())physical=&actual->remote(i);
        if(!physical)throw std::logic_error("Combined inline reader lost its actual physical storage coordinate");
        const auto owner=appendPooledImmutableInlineOwnerPrefix(graph,arena,*physical,ownerCode,identity+"/owner");
        graph.addDependency(owner.first,keysSent);previous=owner.last;
    }
    PooledImmutableCompactConsumerCompletion consumer;consumer.transport=actual;consumer.local=local;consumer.requested=keysSent;consumer.intent=intent;
    consumer.header_received=identity+"/result_header_receive";const auto headerAccepted=identity+"/result_header_accept";
    consumer.payload_received=identity+"/native_rows_receive";consumer.received=identity+"/native_rows_confirm";
    consumer.accepted=identity+"/local_accept";consumer.ack_sent=identity+"/ack_send";
    add(graph,consumer.header_received,std::make_unique<CompactTransfer>(arena,actual,local,control,CompactWire::ResultHeader,CapturedTransferEndpoint::Consumer),device,previous);
    add(graph,headerAccepted,std::make_unique<CompactMetadata>(arena,actual,local,std::shared_ptr<Local>{},control,t::MetadataAction::AcceptResultHeader,nullptr,intent),device,consumer.header_received);
    auto payload=std::make_unique<CompactTransfer>(arena,actual,local,control,CompactWire::NativeRows,CapturedTransferEndpoint::Consumer);const auto* receipt=payload.get();
    add(graph,consumer.payload_received,std::move(payload),device,headerAccepted);
    add(graph,consumer.received,std::make_unique<CompactMetadata>(arena,actual,local,std::shared_ptr<Local>{},control,t::MetadataAction::ConfirmResult,receipt,intent),device,consumer.payload_received);
    add(graph,consumer.accepted,std::make_unique<PooledImmutableLocalHotPhaseStage>(arena,local,localCode,Action::AcceptResult,
        PooledImmutableLocalHotOperands::result(*local,geometry->consumer(coordinate,local->consumerStorageIdentity()),actual->localHotOriginalRequest(device),control)),device,consumer.received);
    add(graph,consumer.ack_sent,std::make_unique<CompactTransfer>(arena,actual,local,control,CompactWire::Ack,CapturedTransferEndpoint::Producer),device,consumer.accepted);
    requirePooledCompactConsumerCompletion(graph,consumer);previous=consumer.ack_sent;
    if(physical)previous=appendPooledRetirementOwnerAfterCompactConsumer(graph,arena,physical->retirement,ownerCode,
        findPooledReadOwnerCompletion(graph,physical->read),identity+"/owner_retirement",consumer).last;
    if(intent==PooledImmutableLocalHotAccessIntent::CurrentInvocation){
        const auto maintenance=appendLocalMaintenance(graph,arena,local,localCode,geometry->consumer(coordinate,local->consumerStorageIdentity()),control,previous,identity+"/local");
        endpoints.consumed=maintenance.first;endpoints.retired=maintenance.second;
    }else endpoints.last=endpoints.retired=previous;
    endpoints.auxiliary.ready_publication=consumer.accepted;
    endpoints.auxiliary=auxiliaryRange(graph,before,endpoints,RetainedGraphParticipantAuxiliaryRole::Reader,device);
    if(intent==PooledImmutableLocalHotAccessIntent::FutureReadAhead){
        endpoints.auxiliary.lifecycle=RetainedGraphParticipantAuxiliaryLifecycle::FutureProducer;
        mix(endpoints.auxiliary.identity,static_cast<unsigned>(endpoints.auxiliary.lifecycle));
    }
    endpoints.auxiliary.original_execution = originalExecutionPolicy();
    graph.setRetainedParticipantAuxiliaryRange(endpoints.auxiliary);return endpoints;
}
PooledImmutableLocalHotGraphEndpoints PooledImmutableRetainedParticipants::appendLocalHotReader(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<Local> local,std::shared_ptr<const IPooledImmutableLocalHotKernels> code,
    DeviceGenerationControlBinding control,DeviceRowRange tokens,PooledNativeTensorSpan ids,PooledNativeTensorSpan output,
    const std::string& identity,const std::string& outputIdentity,std::optional<PooledNativeTensorSpan> count)
{
    return appendLocalHotReaderBody(graph,arena,std::move(local),std::move(code),control,tokens,std::move(ids),std::move(output),
        identity,outputIdentity,std::move(count),PooledImmutableLocalHotAccessIntent::CurrentInvocation);
}
PooledImmutableLocalHotFutureSourceBody PooledImmutableRetainedParticipants::appendLocalHotFuturePLE(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<const IPooledImmutableNativeKernels> poolCode,
    std::shared_ptr<Local> local, std::shared_ptr<const IPooledImmutableLocalHotKernels> code,
    DeviceRowRange tokens, PooledNativeTensorSpan ids, std::optional<PooledNativeTensorSpan> count,
    DeviceGenerationControlBinding control, const std::string& identity)
{
    requireCurrent();
    auto result = appendPooledImmutableLocalHotFutureSourceBody(graph, arena, transport(), localHotParticipants(),
        std::move(poolCode), std::move(local), std::move(code), tokens, std::move(ids), std::move(count), control, identity);
    bindSourcePlan(graph, arena, result.pool_retirement);
    result.auxiliary.original_execution = originalExecutionPolicy();
    graph.setRetainedParticipantAuxiliaryRange(result.auxiliary);
    return result;
}

PooledImmutableLocalHotFutureProducerEndpoints PooledImmutableRetainedParticipants::appendLocalHotFutureReader(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<Local> local,
    std::shared_ptr<const IPooledImmutableLocalHotKernels> code, DeviceGenerationControlBinding control,
    DeviceRowRange tokens, PooledNativeTensorSpan ids, const std::string& identity,
    std::optional<PooledNativeTensorSpan> count)
{
    auto result = appendLocalHotReaderBody(graph, arena, std::move(local), std::move(code), control, tokens,
        std::move(ids), {}, identity, {}, std::move(count), PooledImmutableLocalHotAccessIntent::FutureReadAhead);
    return {result.probe, result.auxiliary.ready_publication, result.retired, std::move(result.auxiliary)};
}

std::string PooledImmutableRetainedParticipants::appendLocalHotAdoptionCheck(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<Local> local,
    std::shared_ptr<const IPooledImmutableLocalHotKernels> code, DeviceGenerationControlBinding control,
    DeviceRowRange tokens, PooledNativeTensorSpan ids, std::optional<PooledNativeTensorSpan> count,
    const std::string& identity)
{
    requireCurrent();
    const auto actual = transport();
    const auto coordinate = localCoordinate(*actual, local);
    const auto device = local->geometry().allocator().device;
    add(graph, identity, std::make_unique<PooledImmutableLocalHotPhaseStage>(arena, local, code, Action::Adopt,
        PooledImmutableLocalHotOperands::adoption(*local,
            actual->localHotGeometry()->consumer(coordinate, local->consumerStorageIdentity()), tokens,
            std::move(ids), std::move(count), actual->localHotAdmission(device), control)), device, {});
    return identity;
}

std::string PooledImmutableRetainedParticipants::appendLocalHotPredictionMissRetirement(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<Local> local,
    std::shared_ptr<const IPooledImmutableLocalHotKernels> code, DeviceGenerationControlBinding control,
    const std::string& adoptionCheck, const std::string& identity)
{
    requireCurrent();
    const auto actual = transport();
    const auto coordinate = localCoordinate(*actual, local);
    const auto device = local->geometry().allocator().device;
    const auto& node = actualNode(graph, adoptionCheck, device);
    const auto* adoption = dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node.stage.get());
    if (!adoption || adoption->storage() != local || adoption->action() != Action::Adopt ||
        adoption->operands().controllerIdentity() != control.captureIdentity() || !code ||
        adoption->codeIdentity() != code->codeIdentity() || graph.retainedParticipantSourcePlan() ||
        graph.retainedParticipantAuxiliaryRange())
        throw std::invalid_argument("Prediction-miss retirement requires its actual selected complete-demand adoption");
    add(graph, identity, std::make_unique<PooledImmutableLocalHotPhaseStage>(arena, local, code, Action::Retire,
        PooledImmutableLocalHotOperands::frame(*local,
            actual->localHotGeometry()->consumer(coordinate, local->consumerStorageIdentity()), control)),
        device, adoptionCheck);
    return identity;
}

PooledImmutableLocalHotConsumerEndpoints PooledImmutableRetainedParticipants::appendLocalHotConsumer(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<Local> local,
    std::shared_ptr<const IPooledImmutableLocalHotKernels> code, DeviceGenerationControlBinding control,
    PooledImmutableLocalHotConsumerBranch branch, std::optional<PooledNativeTensorSpan> output,
    const std::string& identity, const std::string& outputIdentity)
{
    requireCurrent();
    const auto actual = transport();
    const auto coordinate = localCoordinate(*actual, local);
    const auto device = local->geometry().allocator().device;
    const auto geometry = actual->localHotGeometry()->consumer(coordinate, local->consumerStorageIdentity());
    graph.setRetainedParticipantDefinitionDomain({actual->localHotGeometry()->coordinator().device, actual.get()});
    std::set<std::string> before;
    for (const auto& name : graph.getExecutionOrder()) before.insert(name);
    PooledImmutableLocalHotGraphEndpoints e;
    if (branch == PooledImmutableLocalHotConsumerBranch::Accepted)
    {
        if (!output) throw std::invalid_argument("Accepted local consumer requires its actual sole output");
        e.ready_wait = identity + "/ready";
        e.read = outputIdentity;
        e.first = e.ready_wait;
        e.last = e.read;
        add(graph, e.ready_wait, std::make_unique<PooledImmutableLocalHotReadyStage>(
            local, PooledImmutableLocalHotReadyStage::Frontier::DataReady), device, {});
        add(graph, e.read, std::make_unique<PooledImmutableLocalHotPhaseStage>(arena, local, code, Action::LocalRead,
            PooledImmutableLocalHotOperands::read(*local, geometry, std::move(*output), control)), device, e.ready_wait);
        // Maintenance forks after the actual read fragment, never a fabricated new Probe.
        e.probe = e.read;
        const auto maintenance = appendLocalMaintenance(graph, arena, local, code, geometry, control, e.read, identity);
        e.consumed = maintenance.first;
        e.retired = maintenance.second;
    }
    else if (branch == PooledImmutableLocalHotConsumerBranch::Discard)
    {
        if (output) throw std::invalid_argument("Discard cannot declare a model output");
        e.first = e.last = e.probe = identity + "/discard";
        e.retired = identity + "/retired";
        add(graph, e.probe, std::make_unique<PooledImmutableLocalHotPhaseStage>(arena, local, code, Action::Discard,
            PooledImmutableLocalHotOperands::frame(*local, geometry, control)), device, {});
        add(graph, e.retired, std::make_unique<PooledImmutableLocalHotPhaseStage>(arena, local, code, Action::Retire,
            PooledImmutableLocalHotOperands::frame(*local, geometry, control)), device, e.probe);
    }
    else throw std::invalid_argument("Local consumer branch is unknown");
    auto range = auxiliaryRange(graph, before, e, device == actual->localHotGeometry()->coordinator().device
        ? RetainedGraphParticipantAuxiliaryRole::Source : RetainedGraphParticipantAuxiliaryRole::Reader, device);
    range.lifecycle = branch == PooledImmutableLocalHotConsumerBranch::Accepted
        ? RetainedGraphParticipantAuxiliaryLifecycle::AdoptedConsumer
        : RetainedGraphParticipantAuxiliaryLifecycle::DiscardedConsumer;
    mix(range.identity, static_cast<unsigned>(range.lifecycle));
    range.original_execution = originalExecutionPolicy();
    bindLocalConsumption(graph, arena, local, range);
    return {e.first, e.last, e.retired, *graph.retainedParticipantAuxiliaryRange()};
}

} // namespace qwen4exp
namespace {
/** @brief Authenticate the shared immutable Probe/acquisition/controller tuple for a real producer. */
void requireFutureProducer(const ComputeGraph& graph,const RetainedGraphParticipantAuxiliaryRange& range,
    const std::shared_ptr<const Transport>& transport,const std::shared_ptr<Local>& local)
{
    const auto device=local->geometry().allocator().device;
    const auto& probeNode=actualNode(graph,range.probe_ready,device);
    const auto* probe=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(probeNode.stage.get());
    const auto& acquiredNode=actualNode(graph,range.ready_publication,device);
    const auto* acquired=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(acquiredNode.stage.get());
    if(!probe || probe->storage()!=local || probe->action()!=Action::Probe ||
       probe->operands().queryIntent()!=q::PooledImmutableLocalHotAccessIntent::FutureReadAhead || !acquired || acquired->storage()!=local ||
       acquired->action()!=Action::AcceptResult || probe->operands().controllerIdentity()!=acquired->operands().controllerIdentity() ||
       !range.main_read.empty() || range.nodes.back()!=range.retirement)
        throw std::invalid_argument("Future producer changed its actual Probe/acquisition or invented model consumption");
    for(const auto& name:range.nodes){
        const auto& node=actualNode(graph,name,device);
        if(const auto* phase=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node.stage.get()))
            if(phase->storage()!=local || phase->action()!=Action::AcceptResult)
                throw std::invalid_argument("Future producer cannot consume/read/fill/retire the retained local frame");
        if(dynamic_cast<const PooledImmutableLocalHotReadyStage*>(node.stage.get()))
            throw std::invalid_argument("Future producer cannot wait for a later local model consumer");
        if(const auto* metadata=dynamic_cast<const CompactMetadata*>(node.stage.get()))
            if(metadata->intent()!=q::PooledImmutableLocalHotAccessIntent::FutureReadAhead)
                throw std::invalid_argument("Future producer metadata changed its immutable lookup-only purpose");
        if(const auto* joined=dynamic_cast<const CompactUnion*>(node.stage.get()))
            if(joined->intent()!=q::PooledImmutableLocalHotAccessIntent::FutureReadAhead)
                throw std::invalid_argument("Future producer union changed its immutable lookup-only purpose");
    }
    if(range.role==RetainedGraphParticipantAuxiliaryRole::Source){
        const auto* retired=dynamic_cast<const PooledImmutableSourcePhaseStage*>(actualNode(graph,range.retirement,device).stage.get());
        if(!retired || !retired->matchesDemand(transport->sourceBoundary().get(),q::PooledSourceAction::ObserveRetired))
            throw std::invalid_argument("Future source omitted its actual original pool/source retirement");
    }else{
        const auto consumer=findPooledCompactConsumerCompletion(graph,transport,local);
        if(consumer.intent!=q::PooledImmutableLocalHotAccessIntent::FutureReadAhead || consumer.accepted!=range.ready_publication)
            throw std::invalid_argument("Future reader substituted its original result/ACK purpose");
        auto expected=consumer.ack_sent;
        const auto& owners=transport->sourceBoundary()->pooledStorage()->originalOwners();
        for(std::uint32_t i=0;i<owners.size();++i)if(owners[i]->plan().allocator()==local->geometry().allocator())
            expected=findPooledRetirementOwnerAfterCompactConsumer(graph,transport->remote(i).retirement,consumer);
        if(expected!=range.retirement)throw std::invalid_argument("Future reader omitted actual ACK/inline-owner retirement");
    }
}
}
void requirePooledLocalHotLocalConsumption(const ComputeGraph& graph,const RetainedGraphParticipantAuxiliaryRange& range,
    const std::shared_ptr<qwen4exp::PooledImmutableLocalHotPreparedStorage>& local)
{
    using Lifecycle=RetainedGraphParticipantAuxiliaryLifecycle;
    if(!local || !localConsumptionOnly(range.lifecycle) || graph.retainedParticipantSourcePlan() ||
       !range.local_consumption || !range.ready_publication.empty() || range.nodes.empty())
        throw std::invalid_argument("Local consumption cannot claim new source execution or response acquisition");
    const auto device=local->geometry().allocator().device;
    const auto& anchorNode=actualNode(graph,range.probe_ready,device);
    const auto* anchor=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(anchorNode.stage.get());
    const bool accepted=range.lifecycle==Lifecycle::AdoptedConsumer;
    if(!anchor || anchor->storage()!=local || anchor->action()!=(accepted?Action::LocalRead:Action::Discard) ||
       (accepted ? range.main_read!=range.probe_ready : !range.main_read.empty()))
        throw std::invalid_argument("Local consumption substituted actual selected read/discard branch");
    const auto controller=anchor->operands().controllerIdentity();
    std::string previous=range.retirement;std::vector<std::string> reverse;
    const auto phase=[&](Action action){
        const auto& node=actualNode(graph,previous,device);
        const auto* actual=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node.stage.get());
        if(!actual || actual->storage()!=local || actual->action()!=action ||
           actual->codeIdentity()!=anchor->codeIdentity() || actual->operands().controllerIdentity()!=controller || node.dependencies.size()!=1)
            throw std::invalid_argument("Local consumption changed its actual policy/fill/retirement chain");
        reverse.push_back(previous);previous=node.dependencies.front();
    };
    phase(Action::Retire);
    if(accepted){
        phase(Action::CompleteFills);phase(Action::CopyFills);phase(Action::ConsumeDemand);
        const auto& waitNode=actualNode(graph,previous,device);
        const auto* wait=dynamic_cast<const PooledImmutableLocalHotReadyStage*>(waitNode.stage.get());
        if(!wait || wait->storage()!=local || wait->frontier()!=PooledImmutableLocalHotReadyStage::Frontier::LocalReadComplete ||
           waitNode.dependencies.size()!=1 || waitNode.dependencies.front()!=range.probe_ready)
            throw std::invalid_argument("Local policy must follow the actual local read-complete frontier");
        reverse.push_back(previous);
        const auto& read=actualNode(graph,range.main_read,device);
        if(read.dependencies.size()!=1)throw std::invalid_argument("Adopted read lost its actual data-ready wait");
        const auto* ready=dynamic_cast<const PooledImmutableLocalHotReadyStage*>(actualNode(graph,read.dependencies.front(),device).stage.get());
        if(!ready || ready->storage()!=local || ready->frontier()!=PooledImmutableLocalHotReadyStage::Frontier::DataReady)
            throw std::invalid_argument("Adopted local read changed its actual retained data-ready frontier");
    }else if(previous!=range.probe_ready)
        throw std::invalid_argument("Discard retirement must follow its actual unconsumed-frame discard");
    std::reverse(reverse.begin(),reverse.end());
    if(reverse!=range.nodes)throw std::invalid_argument("Local-only body omitted, duplicated or added a source/transport stage");
}
void requirePooledLocalHotAuxiliary(const ComputeGraph& graph,const RetainedGraphParticipantAuxiliaryRange& range,
    const std::shared_ptr<const Transport>& transport,const std::shared_ptr<Local>& local)
{
    if(!transport || !local)throw std::invalid_argument("Local-hot auxiliary lost original transport/local roots");
    const auto coordinate=localCoordinate(*transport,local);const auto& geometry=*transport->localHotGeometry();const auto device=local->geometry().allocator().device;
    const auto domain=graph.retainedParticipantDefinitionDomain();
    if(range.lifecycle==RetainedGraphParticipantAuxiliaryLifecycle::FutureProducer){
        if(!domain || *domain!=RetainedGraphParticipantDefinitionDomain{geometry.coordinator().device,transport.get()} ||
           range.participant!=device || (range.role==RetainedGraphParticipantAuxiliaryRole::Source)!=(device==geometry.coordinator().device))
            throw std::invalid_argument("Future producer changed actual source/domain/participant role");
        requireFutureProducer(graph,range,transport,local);return;
    }
    if(localConsumptionOnly(range.lifecycle)){
        requirePooledLocalHotLocalConsumption(graph,range,local);return;
    }
    if(range.lifecycle!=RetainedGraphParticipantAuxiliaryLifecycle::CurrentDemand)
        throw std::invalid_argument("Local auxiliary has unknown lifecycle");
    if(!domain || *domain!=RetainedGraphParticipantDefinitionDomain{geometry.coordinator().device,transport.get()} || range.participant!=device ||
        (range.role==RetainedGraphParticipantAuxiliaryRole::Source)!=(local->geometry().allocator()==geometry.coordinator()) ||
        range.nodes.empty() || range.nodes.back()!=range.retirement || !range.identity)
        throw std::invalid_argument("Local-hot auxiliary changed actual source/domain/role/retirement");
    const auto& mainRead=actualNode(graph,range.main_read,device);const auto* read=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(mainRead.stage.get());
    const auto& probeNode=actualNode(graph,range.probe_ready,device);const auto* probe=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(probeNode.stage.get());
    const auto& publicationNode=actualNode(graph,range.ready_publication,device);const auto* publication=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(publicationNode.stage.get());
    const auto& terminalNode=actualNode(graph,range.retirement,device);const auto* terminal=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(terminalNode.stage.get());
    if(!probe || probe->storage()!=local || probe->action()!=Action::Probe || !read || read->storage()!=local || read->action()!=Action::LocalRead ||
        !publication || publication->storage()!=local || publication->action()!=Action::AcceptResult || !terminal || terminal->storage()!=local || terminal->action()!=Action::Retire ||
        probe->operands().controllerIdentity()!=read->operands().controllerIdentity() || probe->operands().controllerIdentity()!=publication->operands().controllerIdentity() ||
        probe->operands().controllerIdentity()!=terminal->operands().controllerIdentity())
        throw std::invalid_argument("Local-hot auxiliary substituted actual Probe/read/acquisition/retirement roots");
    if(mainRead.dependencies.size()!=1)throw std::invalid_argument("LocalRead lost its actual main-ready frontier");
    validatePooledLocalHotDemandFrontier(graph,{range.probe_ready,mainRead.dependencies.front(),range.main_read,device});
    if(std::find(range.nodes.begin(),range.nodes.end(),range.probe_ready)!=range.nodes.end() ||
        std::find(range.nodes.begin(),range.nodes.end(),range.main_read)!=range.nodes.end() ||
        std::find(range.nodes.begin(),range.nodes.end(),range.ready_publication)==range.nodes.end())
        throw std::invalid_argument("Local-hot auxiliary swallowed a main producer or omitted data publication");
    std::set<std::string> names;bool completeWait=false;
    for(const auto& name:range.nodes){const auto& node=actualNode(graph,name,device);if(!names.insert(name).second)throw std::invalid_argument("Local-hot auxiliary repeats a stage");
        if(const auto* ready=dynamic_cast<const PooledImmutableLocalHotReadyStage*>(node.stage.get()))if(ready->storage()==local && ready->frontier()==PooledImmutableLocalHotReadyStage::Frontier::LocalReadComplete)completeWait=true;
    }
    if(!completeWait)throw std::invalid_argument("Local-hot auxiliary policy does not wait for actual model read completion");
    std::string policy=range.retirement;
    for(const auto action:{Action::Retire,Action::CompleteFills,Action::CopyFills,Action::ConsumeDemand}){
        const auto& node=actualNode(graph,policy,device);const auto* phase=dynamic_cast<const PooledImmutableLocalHotPhaseStage*>(node.stage.get());
        if(!phase || phase->storage()!=local || phase->action()!=action || phase->operands().controllerIdentity()!=probe->operands().controllerIdentity() || node.dependencies.size()!=1)
            throw std::invalid_argument("Local-hot auxiliary changed actual consumed/fill/retire chain");
        policy=node.dependencies.front();
    }
    const auto& waitNode=actualNode(graph,policy,device);const auto* waited=dynamic_cast<const PooledImmutableLocalHotReadyStage*>(waitNode.stage.get());
    if(!waited || waited->storage()!=local || waited->frontier()!=PooledImmutableLocalHotReadyStage::Frontier::LocalReadComplete || waitNode.dependencies.size()!=1)
        throw std::invalid_argument("Local-hot maintenance does not follow its real read-complete gate");
    if(range.role==RetainedGraphParticipantAuxiliaryRole::Source){
        const auto& retired=actualNode(graph,waitNode.dependencies.front(),device);const auto* observed=dynamic_cast<const PooledImmutableSourcePhaseStage*>(retired.stage.get());
        if(!observed || !observed->matchesDemand(transport->sourceBoundary().get(),q::PooledSourceAction::ObserveRetired))
            throw std::invalid_argument("Local-hot maintenance omitted original source physical retirement");
    }
    if(range.role==RetainedGraphParticipantAuxiliaryRole::Reader){
        const auto consumer=findPooledCompactConsumerCompletion(graph,transport,local);
        if(consumer.accepted!=range.ready_publication)throw std::invalid_argument("Reader auxiliary omitted its actual compact result frontier");
        std::string expectedBeforeWait=consumer.ack_sent;
        const auto& owners=transport->sourceBoundary()->pooledStorage()->originalOwners();
        for(std::uint32_t i=0;i<owners.size();++i)if(owners[i]->plan().allocator()==local->geometry().allocator()){
            const auto& original=transport->remote(i);
            const auto readSent=findPooledReadOwnerCompletion(graph,original.read);
            if(!ancestor(graph,consumer.header_received,readSent) || !ancestor(graph,readSent,consumer.requested))throw std::invalid_argument("Inline compact reader changed request-before-owner/read-before-result ordering");
            const auto retired=findPooledRetirementOwnerAfterCompactConsumer(graph,original.retirement,consumer);
            if(!names.contains(retired) || !ancestor(graph,range.retirement,retired))throw std::invalid_argument("Inline compact auxiliary omitted original physical owner retirement");
            expectedBeforeWait=retired;
        }
        if(waitNode.dependencies.front()!=expectedBeforeWait)throw std::invalid_argument("Reader local maintenance bypassed original ACK/physical retirement");
    }
    (void)coordinate;
}
} // namespace llaminar2
