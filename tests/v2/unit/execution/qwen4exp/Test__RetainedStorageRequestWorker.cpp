/**
 * @file Test__RetainedStorageRequestWorker.cpp
 * @brief Actual request admission, reset and hosted publisher bodies over original recorded workers.
 * Native APIs alone are doubled. The original source/pool/PMA, DGO methods,
 * capture scope and complete ticket builder remain production implementations.
 */
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
#include "execution/local_execution/graph/IRetainedGraphStorageBoundary.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "models/qwen/QwenStandardGraph.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "mocks/MockBackend.h"
#include "mocks/MockModelContext.h"
#include "mocks/MockLocalTPContext.h"
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/qwen4exp/ImmutableLookupServiceGeometry.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <vector>
namespace llaminar2 {
/** @brief Existing closed DGO fixture friendship; no production admission is manufactured. */
struct DeviceGraphOrchestratorLiveStateTestAccess {
    /** @brief Bind only original request/control/event identities to the actual DGO. */
    static void seed(DeviceGraphOrchestrator& runner,const PreparedGPUContextUsePermit& source,int* words){
        runner.state_.device_id=source.device();runner.physical_memory_authority_=source.memory();runner.session_epoch_=7;
        runner.device_generation_storage_.control_device=words;
        runner.device_generation_terminal_host_ready_event_=std::shared_ptr<void>(new int(1),[](void* p){delete static_cast<int*>(p);});
    }
    /** @brief Advance only the original host request epoch for a completed/reset interval. */
    static void epoch(DeviceGraphOrchestrator& runner,std::uint64_t value){runner.session_epoch_=value;}
    /** @brief Invoke the same original-stream binder used by ordinary and MTP parents. */
    static bool bindParent(DeviceGraphOrchestrator& runner,const std::shared_ptr<void>& stream,std::string* error){return runner.bindRetainedStorageSubmissionStream(stream,error);}
    /** @brief Invoke actual once-per-request admission. */
    static bool admit(DeviceGraphOrchestrator& runner,void* stream){return runner.beginRetainedStorageAdmission(stream);}
    /** @brief Execute the actual main-model forward prelude, before native replay. */
    static bool prelude(DeviceGraphOrchestrator& runner,const ForwardInput& input,void* stream){return runner.prepareLiveStateForForwardGraphExecution(input,stream,runner.state_.device_id);}
    /** @brief Invoke actual terminal reset sequencing. */
    static void reset(DeviceGraphOrchestrator& runner,void* stream){runner.retireRetainedStorageForRequestReset(stream);}
    /** @brief Drain the actual original terminal right after its exact native close completed. */
    static void awaitOriginal(DeviceGraphOrchestrator& runner){runner.awaitRetainedStorageTerminal();}

    /** @brief Publish/complete the actual original close while leaving service retirement for the complete TP group. */
    static void publishResetClose(DeviceGraphOrchestrator& runner,void* stream){
        runner.retireRetainedStorageForRequestReset(stream,DeviceGraphOrchestrator::RetainedStorageTerminalPhase::PublishOnly);
    }

    /** @brief Replay a foreign ownership block without replacing the source permit. */
    static void foreignMemory(DeviceGraphOrchestrator& runner,const std::shared_ptr<PhysicalMemoryAuthority>& memory){runner.physical_memory_authority_=memory;}
    /** @return Original already-owned terminal event; creates no event. */
    static void* terminalEvent(const DeviceGraphOrchestrator& runner){return runner.device_generation_terminal_host_ready_event_.get();}
    /** @brief Withdraw test-only borrowed words before normal DGO destruction. */
    static void clear(DeviceGraphOrchestrator& runner){runner.state_.device_id=DeviceId::cpu();runner.device_generation_storage_.control_device=nullptr;runner.device_generation_terminal_host_ready_event_.reset();}
};
namespace test {
/** @brief Invoke the actual Rank terminal fanout owner without constructing a model or native context. */
struct RankRetainedStorageRetirementReplayAccess {
    /** @brief Invoke the actual condition fan-out, including its original group preparation/attempt lifetime. */
    static bool condition(RankOrchestrator& rank,bool forward,const std::function<bool(IInferenceRunner&,size_t)>& operation){
        return rank.dispatchLocalTPMTPMainCondition("original-source-condition-replay",
            forward?RankOrchestrator::LocalTPConditionEffect::ForwardMainRow:RankOrchestrator::LocalTPConditionEffect::PublishInput,operation);
    }
    /** @return Opaque identity only; no private source state or device state is fabricated. */
    static const void* admission(const RankOrchestrator& rank){return rank.retained_source_admission_.get();}
    template<class Participants,class Retire,class Finish>
    static void retire(const Participants& participants,Retire retire,Finish finish){
        RankOrchestrator::retireOriginalStorageParticipants(participants,retire,finish);
    }
};
/** @brief Closed replay of the exact production source-membership operations; no native source is manufactured. */
struct ImmutableLookupSourceAdmissionReplayAccess {
    using Owner=qwen4exp::ImmutableLookupExecutionBoundary;
    using Identity=Owner::SourceAdmissionIdentity;
    static auto order(std::span<const DeviceId> channels,std::span<const Identity> members){return Owner::sourceAdmissionOrder(channels,members);}
    static bool same(const Identity& a,const Identity& b){return Owner::sameSourceAdmissionIdentity(a,b);}
    static bool reuses(std::span<const Identity> a,std::span<const Identity> b){
        return Owner::sourceAdmissionReuses<Identity>(a,b,[](const Identity& value){return value;});
    }
};
namespace {
using Access=NativeContextPreparationReplayAccess;
using Peer=DeviceGraphOrchestratorLiveStateTestAccess;
/** @brief Recorded original worker with actual mock worker-entry/exception scope. */
class Worker final:public testing::MockWorkerGPUContext {
public:
    Worker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
    int submissions=0,brokenEntries=0;bool brokenDispatch=false;
    void submitAndWait(std::function<void()> work)override{
        ++submissions;
        if(brokenDispatch){
            // Failure-only tripwire bounds the original missing-guard control;
            // it never makes a broken dispatch pass or enters worker identity.
            if(++brokenEntries>1)throw std::runtime_error("Broken dispatcher recursively reentered");
            work();return;
        }
        MockWorkerGPUContext::submitAndWait(std::move(work));
    }
};
/** @brief Observe one source interval; no model/controller values are read. */
class Boundary final:public IRetainedGraphStorageBoundary {
public:
    PreparedGPUContextUsePermit original;RetainedStorageGenerationControl control;void* stream;
    std::optional<RetainedGraphParticipantParentStream> parent;
    /** @brief Observe the actual DGO-minted original parent capability without creating resources. */
    void bindSourceSubmissionStream(const RetainedGraphParticipantParentStream& actual)override{
        EXPECT_EQ(actual.original().device(),original.device());original.requireWorker(actual.original().worker());
        EXPECT_EQ(actual.original().memory(),original.memory());parent.emplace(actual);
    }
    std::exception_ptr awaitFailure;std::function<void()> beforeAwait;mutable std::vector<int> trace;const int* expectedHealthy=nullptr;const int* expectedComplete=nullptr;int admissions=0;mutable int closes=0;bool active=false,failAdmission=false,failClose=false,failAwait=false;
    explicit Boundary(const PreparedGPUContextUsePermit& source):original(source),stream(source.worker()->defaultStream()){}
    DeviceId device()const noexcept override{return original.device();}
    std::uint64_t captureIdentity()const noexcept override{return 19;}
    std::optional<PreparedGPUContextUsePermit> retainedParticipantSource()const override{return original;}
    void bindGenerationControl(RetainedStorageGenerationControl c)override{control=c;}
    RetainedStorageGenerationControl generationControl()const override{return control;}
    void beginAdmission(std::uint64_t epoch,void* exact)override{EXPECT_TRUE(original.worker()->ownsCurrentThread());EXPECT_EQ(epoch,7u);EXPECT_EQ(exact,stream);++admissions;trace.push_back(1);if(failAdmission)throw std::runtime_error("original admission failure");active=true;}
    bool hasActiveInterval()const noexcept override{return active;}
    void requireArmedForLaunch()const override{if(!active)throw std::logic_error("inactive");}
    void recordTerminalClose(void* exact,const int* healthy=nullptr,const int* complete=nullptr)const override{
        EXPECT_TRUE(original.worker()->ownsCurrentThread());EXPECT_EQ(exact,stream);EXPECT_EQ(bool(healthy),bool(complete));if(healthy){EXPECT_EQ(healthy,expectedHealthy);EXPECT_EQ(complete,expectedComplete);}
        ++closes;trace.push_back(2);if(failClose)throw std::runtime_error("original close failure");
    }
    void awaitTerminal()override{if(beforeAwait)beforeAwait();EXPECT_FALSE(original.worker()->ownsCurrentThread());trace.push_back(5);active=false;if(awaitFailure)std::rethrow_exception(awaitFailure);if(failAwait)throw std::runtime_error("original terminal failure");}
    RetainedStorageTerminalObservation terminalObservation()const override{RetainedStorageTerminalObservation out;out.generation=1;return out;}
};
/** @brief Original graph builder keeps setup metadata off-worker. */
class Builder final:public QwenStandardGraph {
public:
    std::shared_ptr<IRetainedGraphStorageBoundary> boundary;
    static GraphConfig tinyConfig(){GraphConfig c;c.d_model=32;c.n_heads=1;c.n_kv_heads=1;c.head_dim=32;c.n_layers=1;c.d_ff=64;c.vocab_size=32;c.max_seq_len=16;return c;}
    /** @brief Declare only the real hosted-domain geometry required by the original binder. */
    static GraphConfig hostedConfig(DeviceId device){
        auto config=tinyConfig();auto execution=std::make_shared<MoEExpertOverlayExecutionPlan>();
        execution->current_rank.execution_kind=OverlayRankExecutionKind::ContinuationAuthority;
        MoEOverlayRuntimeDomain domain;
        domain.participants.push_back({.address=GlobalDeviceAddress::cpu(0)});
        domain.participants.push_back({.address=device.is_cuda()?GlobalDeviceAddress::cuda(device.ordinal):GlobalDeviceAddress::rocm(device.ordinal)});
        execution->domains.push_back(std::move(domain));config.moe.expert_overlay_execution_plan=std::move(execution);return config;
    }
    explicit Builder(std::shared_ptr<IRetainedGraphStorageBoundary> source,bool hosted=false):QwenStandardGraph(hosted?hostedConfig(source->device()):tinyConfig(),nullptr),boundary(std::move(source)){}
    std::shared_ptr<IRetainedGraphStorageBoundary> retainedStorageBoundary(DeviceId device)override{EXPECT_EQ(device,boundary->device());return boundary;}
};
/** @brief Native-shaped backend proves event publication/terminal wait remain on the caller. */
class Backend final:public MockBackend {
public:
    std::shared_ptr<Boundary> boundary;Worker* worker=nullptr;void* event=nullptr;void* words=nullptr;void* ticketRows=nullptr;const void* maintenance=nullptr;int tickets=0;bool failTicket=false;
    explicit Backend(DeviceType type):MockBackend(type){}
    bool recordEvent(void* original,int ordinal,void* stream)override{EXPECT_EQ(original,event);EXPECT_FALSE(worker->ownsCurrentThread());EXPECT_EQ(ordinal,boundary->device().ordinal);EXPECT_EQ(stream,boundary->stream);boundary->trace.push_back(3);return true;}
    bool waitForEvent(void* original,int)override{EXPECT_EQ(original,event);EXPECT_FALSE(worker->ownsCurrentThread());boundary->trace.push_back(4);return true;}
    bool enqueuePublishDeviceGenerationDispatchTickets(void* originalWords,int stride,int count,const void* due,void* rows,int ordinal,void* stream)override{
        EXPECT_EQ(originalWords,words);EXPECT_EQ(stride,sampling_math::kDeviceGenerationControlCount);EXPECT_EQ(count,1);EXPECT_EQ(due,maintenance);EXPECT_EQ(rows,ticketRows);
        EXPECT_TRUE(worker->ownsCurrentThread());EXPECT_TRUE(worker->isDeviceGraphCaptureActive());EXPECT_TRUE(isGraphCaptureActive());
        EXPECT_EQ(ordinal,boundary->device().ordinal);EXPECT_EQ(stream,boundary->stream);++tickets;
        if(failTicket)throw std::runtime_error("original ticket failure");return true;
    }
};
Backend* native=nullptr;
IBackend* groupNative=nullptr;
/** @brief The exact complete publisher must enter/leave/instantiate on one original worker. */
class Graph final:public IGPUGraphCapture {
public:
    Worker& worker;void* stream;int begins=0,ends=0,instances=0;bool ready=false;
    Graph(Worker& original,void* exact):worker(original),stream(exact){}
    bool beginCapture()override{EXPECT_TRUE(worker.ownsCurrentThread());++begins;return true;}
    bool endCapture()override{EXPECT_TRUE(worker.ownsCurrentThread());++ends;return true;}
    bool instantiate()override{EXPECT_TRUE(worker.ownsCurrentThread());++instances;return ready=true;}
    bool launch()override{ADD_FAILURE();return false;}
    void* executionStream()const noexcept override{return stream;}
    GraphUpdateResult tryUpdate()override{return GraphUpdateResult::Failed;}
    bool supportsExecutableUpdate()const noexcept override{return false;}
    bool hasExecutable()const override{return ready;}
    std::size_t residentMemoryBytes()const noexcept override{return 0;}
    std::size_t nodeCount()const override{return 1;}
    void reset()override{ready=false;}
    const char* backendName()const override{return "recorded-original-publisher";}
};
/** @brief Actual original source freeze/admission, without native creators. */
struct Source {
    std::shared_ptr<NativeContextPreparation> source;std::optional<PreparedGPUContextUsePermit> original;
    explicit Source(DeviceId device){Access::seed(device,[device](auto stack){return Access::construct<Worker>(std::move(stack),device.ordinal);});source=Access::source<Worker>(device);original.emplace(source->acquire(device));source->freezeForPlanning();source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM()));}
    Worker& worker(){return static_cast<Worker&>(*original->worker());}
    void retire(){const auto device=original->device();original.reset();auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
};
/** @brief Tiny actual DGO, with only already-owned controller/event identities injected. */
struct Runner {
    std::array<int,sampling_math::kDeviceGenerationControlCount> control{};std::shared_ptr<Boundary> boundary;std::shared_ptr<Builder> builder;std::unique_ptr<DeviceGraphOrchestrator> runner;
    explicit Runner(Source& source,bool hosted=false):boundary(std::make_shared<Boundary>(*source.original)),builder(std::make_shared<Builder>(boundary,hosted)),runner(std::make_unique<DeviceGraphOrchestrator>(builder)){
        Peer::seed(*runner,*source.original,control.data());}
    ~Runner(){boundary->active=false;Peer::clear(*runner);runner.reset();}
};

/** @brief Native APIs alone are doubled; PMA, mapped service, boundary and rank bodies remain original. */
class GroupBackend final:public MockBackend {
public:
    /** @brief Bind only the recorded native family, without discovery. */
    explicit GroupBackend(DeviceType type):MockBackend(type){setDeviceCount(2);}
    /** @copydoc IBackend::deviceRuntimeGeneration */
    std::uint64_t deviceRuntimeGeneration(int ordinal)const override{return ordinal>=0&&ordinal<2?1:0;}
    /** @brief Preserve the original protocol ABI alignment at the doubled native allocation boundary. */
    void* allocate(std::size_t bytes,int)override{return std::aligned_alloc(64,(bytes+63)/64*64);}
    /** @copydoc IBackend::free */
    void free(void* value,int)override{std::free(value);}
    /** @copydoc IBackend::supportsPortableMappedAllocation */
    bool supportsPortableMappedAllocation()const noexcept override{return true;}
    /** @copydoc IBackend::allocatePortableMapped */
    void* allocatePortableMapped(std::size_t bytes,int,void** alias)override{
        void* result=std::aligned_alloc(64,(bytes+63)/64*64);if(alias)*alias=result;return result;
    }
    /** @copydoc IBackend::freeMapped */
    void freeMapped(void* value,int)override{std::free(value);}
    /** @copydoc IBackend::externalMappedHostDevicePointer */
    bool externalMappedHostDevicePointer(void* host,int,void** alias)override{*alias=host;return true;}
    /** @copydoc IBackend::recordEvent */
    bool recordEvent(void* event,int,void* stream)override{return event&&stream;}
    /** @copydoc IBackend::waitForEvent */
    bool waitForEvent(void* event,int)override{return event!=nullptr;}
    /** @copydoc IBackend::queryEvent */
    bool queryEvent(void*,int,bool* ready)override{*ready=true;return true;}
};
/** @brief Emulate only the native admit/terminal ABI; no lookup, controller or rank operation is substituted. */
class AdmissionKernels final:public qwen4exp::IImmutableLookupKernels {
public:
    mutable int admissions=0;
    /** @copydoc qwen4exp::IImmutableLookupKernels::admit */
    void admit(const qwen4exp::ImmutableLookupAdmissionArgs& args,void*)const override{
        ++admissions;
        if(args.initialization==qwen4exp::ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce)
            std::memset(args.state,0,sizeof(*args.state));
        *args.deviceGeneration=args.generation;
    }
    /** @copydoc qwen4exp::IImmutableLookupKernels::publish */
    void publish(const qwen4exp::ImmutableLookupPublishArgs&,void*)const override{ADD_FAILURE()<<"Admission replay must not execute a lookup";}
    /** @copydoc qwen4exp::IImmutableLookupKernels::consume */
    void consume(const qwen4exp::ImmutableLookupConsumeArgs&,void*)const override{ADD_FAILURE()<<"Admission replay must not execute a lookup";}
    /** @copydoc qwen4exp::IImmutableLookupKernels::close */
    void close(const qwen4exp::ImmutableLookupCloseArgs& args,void*)const override{
        std::atomic_ref(args.channel.acknowledgement->closedGeneration).store(*args.generation,std::memory_order_release);
    }
};
/** @brief One original two-worker source/PMA and one real RAM provider for all isolated failure cohorts. */
class RankSourceProgram final {
public:
    /** @brief Admit the full three-case original program set before creating any source owner. */
    explicit RankSourceProgram(DeviceType type):backend(type),devices{{{type,0},{type,1}}}{
        groupNative=&backend;
        char path[]="/tmp/rank-original-source-XXXXXX";
        fd=::mkstemp(path);requireFile(fd>=0);filename=path;
        std::array<float,16> rows{};requireFile(::write(fd,rows.data(),sizeof(rows))==sizeof(rows));
        descriptor={.path=filename,.tensor_name="original-rank-admission",.rows=16,.row_elements=1,.format=GGUFTensorType::F32};
        for(const auto device:devices)Access::seed(device,[device](auto stack){return Access::construct<Worker>(std::move(stack),device.ordinal);});
        const std::array observations{PhysicalMemoryResource{0,DeviceId::cpu(),std::size_t{16}<<30,std::size_t{16}<<30},
            PhysicalMemoryResource{0,devices[0],std::size_t{2}<<30,std::size_t{2}<<30},
            PhysicalMemoryResource{0,devices[1],std::size_t{2}<<30,std::size_t{2}<<30}};
        source=Access::source<Worker>(devices,observations);
        for(std::size_t i=0;i<devices.size();++i)permits[i].emplace(source->acquire(devices[i]));
        source->freezeForPlanning();
        PhysicalMemoryPlanBuilder bom(source->mandatoryBOM());
        const auto host=source->originalResource(DeviceId::cpu());
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM);
        bom.add(host,PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(descriptor,policy));
        const auto channel=qwen4exp::ImmutableLookupLayout::create(16,descriptor.rowBytes());
        bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,qwen4exp::ImmutableLookupServiceGeometry::checked(channel,devices).hostBytes());
        for(std::size_t cohort=0;cohort<3;++cohort){
            auto io=RetainedHostServiceFixture::recorded(RetainedHostServiceRole::ImmutableLookupIO,{0,DeviceId::cpu()},cohort);
            const auto worker=io.bom(host);for(const auto& row:worker.resources())bom.add(row);
            for(const auto device:devices){
                const auto ordering=NativeOrderingEventGroup::bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,source->originalResource(device));
                for(const auto& row:ordering.resources())bom.add(row);
            }
        }
        const CPUTensorStorageDeclaration protocol(1,16,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        for(const auto device:devices){bom.add(host,PhysicalMemoryOwner::ActivationArena,protocol.physicalBytes());bom.add(source->originalResource(device),PhysicalMemoryOwner::ActivationArena,64);}
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bom.build()));
        provider=HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(descriptor,policy,DeviceId::cpu(),source->memory(),ImmutableRowTableHostPlacement::taskPolicy()));
    }
    /** @brief Prove original physical reset after all channel, tensor and worker readers have retired. */
    ~RankSourceProgram(){
        provider.reset();
        for(std::size_t i=0;i<devices.size();++i){permits[i].reset();auto retirement=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(devices[i]);Access::physicalReset(retirement);}
        source.reset();::close(fd);::unlink(filename.c_str());groupNative=nullptr;
    }
    GroupBackend backend;
    std::array<DeviceId,2> devices;
    std::shared_ptr<NativeContextPreparation> source;
    std::array<std::optional<PreparedGPUContextUsePermit>,2> permits;
    std::shared_ptr<HostImmutableRowProvider> provider;
    ImmutableRowTableSource descriptor;
private:
    /** @brief Fail setup rather than substituting another source or hiding an incomplete write. */
    static void requireFile(bool valid){if(!valid)throw std::runtime_error("Original rank source file setup failed");}
    int fd=-1;
    std::string filename;
};
/** @brief Actual ordinary boundaries over distinct channels of one real worker, consumed by actual DGOs and rank dispatch. */
struct RankSourceCohort final {
    using QBoundary=qwen4exp::ImmutableLookupExecutionBoundary;
    RankSourceProgram& program;
    std::shared_ptr<qwen4exp::ImmutableLookupService> service;
    std::array<std::shared_ptr<TensorBase>,2> protocols;
    std::array<PhysicalMemoryAllocationLease,2> claims;
    std::array<std::shared_ptr<QBoundary>,2> boundaries;
    std::array<AdmissionKernels*,2> kernels{};
    std::array<DeviceGraphOrchestrator*,2> children{};
    std::array<std::array<int,sampling_math::kDeviceGenerationControlCount>,2> controls{};
    std::unique_ptr<RankOrchestrator> rank;
    /** @brief Bind this declared isolated failure case to the same source/PMA and exact two native channels. */
    RankSourceCohort(RankSourceProgram& original,std::uint64_t ordinal):program(original){
        const auto memory=program.source->memory();
        auto io=RetainedHostServiceFixture::recorded(RetainedHostServiceRole::ImmutableLookupIO,{0,DeviceId::cpu()},ordinal);
        service=qwen4exp::ImmutableLookupService::create(program.provider,qwen4exp::ImmutableLookupLayout::create(16,4),program.devices,memory,DeviceId::cpu(),io.take(memory));
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        RankOrchestrator::Config config;
        for(std::size_t i=0;i<children.size();++i){
            const auto device=program.devices[i];
            auto& worker=*program.permits[i]->worker();
            claims[i]=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,64);
            worker.submitAndWait([&]{
                protocols[i]=std::shared_ptr<TensorBase>(CPUTensorStorageDeclaration(1,16,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()).materialize(memory));
                protocols[i]->setBackendForTesting(&program.backend);
                TransferEngine::prepareDeviceInput(protocols[i].get(),device,worker.defaultStream());
                TransferEngine::requireDeviceInput(protocols[i].get(),device,worker.defaultStream());
                auto code=std::make_unique<AdmissionKernels>();kernels[i]=code.get();
                boundaries[i]=std::make_shared<QBoundary>(service,device,std::move(code),protocols[i],memory);
            });
            auto child=std::make_unique<DeviceGraphOrchestrator>(std::make_shared<Builder>(boundaries[i]));
            Peer::seed(*child,*program.permits[i],controls[i].data());children[i]=child.get();
            runners.push_back(std::move(child));
            config.devices.push_back(device.is_cuda()?GlobalDeviceAddress::cuda(device.ordinal):GlobalDeviceAddress::rocm(device.ordinal));
            config.weights.push_back(0.5f);
        }
        auto tp=std::make_unique<MockLocalTPContext>();tp->setDevices(config.devices);
        rank=RankOrchestrator::createForTest(MockModelContext::createMinimal(),std::move(runners),std::move(tp),config);
        rank->setSkipLogitsGatherDecode(true);
    }
    /** @brief Drop rank/DGO borrows before protocol roots, leases and the shared source. */
    ~RankSourceCohort(){
        close(true);
        for(auto* child:children)Peer::clear(*child);
        rank.reset();boundaries={};protocols={};claims={};service.reset();
    }
    /** @brief Publish every real channel close before consuming any exact await right; failed groups may truthfully reject awaits. */
    void close(bool allowFailure=false){
        for(std::size_t i=0;i<boundaries.size();++i)if(boundaries[i]->hasActiveInterval())program.permits[i]->worker()->submitAndWait([&,i]{boundaries[i]->recordTerminalClose(program.permits[i]->worker()->defaultStream());});
        for(auto& boundary:boundaries)if(boundary->hasActiveInterval())try{boundary->awaitTerminal();}catch(...){if(!allowFailure)throw;}
    }
    /** @brief Execute actual rank fan-out and actual DGO admission; fault injection changes only the selected child result. */
    bool forward(int failure=-1,bool throws=false){
        return RankRetainedStorageRetirementReplayAccess::condition(*rank,true,[&](IInferenceRunner&,std::size_t i){
            if(static_cast<int>(i)==failure){if(throws)throw std::runtime_error("original condition participant failed");return false;}
            return Peer::admit(*children[i],program.permits[i]->worker()->defaultStream());
        });
    }
};
}
}
// Replace only the native backend boundary in this isolated model-free target.
IBackend* getBackendFor(DeviceId device){return device.is_gpu()?(test::groupNative?test::groupNative:test::native):getCPUBackend();}
namespace test {
TEST(RetainedStorageRequestWorker,SourceGroupMembershipPreservesAuthoredTPOrderAndOriginalOwnership){
    using Membership=ImmutableLookupSourceAdmissionReplayAccess;
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm}){
        const DeviceId first{type,0},second{type,1},foreign{type,2};
        const std::array channels{first,second};
        auto a=std::make_shared<int>(1),b=std::make_shared<int>(2);
        const std::array<Membership::Identity,2> authored{{{second,b,9},{first,a,7}}};
        EXPECT_EQ(Membership::order(channels,authored),(std::vector<std::size_t>{1,0}));
        const std::array<Membership::Identity,2> canonical{{authored[1],authored[0]}};
        EXPECT_TRUE(Membership::reuses(canonical,authored));
        EXPECT_EQ(authored[0].device,second); // Compute/TP coordinates are unchanged.
        EXPECT_TRUE(Membership::same({first,a,7},authored[1]));
        EXPECT_TRUE(Membership::same({second,b,9},authored[0]));
        auto duplicate=authored;duplicate[0]=authored[1];
        EXPECT_THROW(Membership::order(channels,duplicate),std::invalid_argument);
        EXPECT_FALSE(Membership::reuses(canonical,duplicate));
        EXPECT_FALSE(Membership::reuses(canonical,std::span(authored).first(1)));
        EXPECT_THROW(Membership::order(channels,std::span(authored).first(1)),std::invalid_argument);
        auto missing=authored;missing[0].device=foreign;
        EXPECT_THROW(Membership::order(channels,missing),std::invalid_argument);
        EXPECT_FALSE(Membership::reuses(canonical,missing));
        auto empty=authored;empty[0].owner.reset();
        EXPECT_THROW(Membership::order(channels,empty),std::invalid_argument);
        auto stale=authored;stale[1].epoch=8;
        EXPECT_FALSE(Membership::reuses(canonical,stale));
        EXPECT_FALSE(Membership::same({first,a,8},authored[1]));
        const std::shared_ptr<const void> alias(a.get(),[](const void*){});
        EXPECT_FALSE(Membership::same({first,alias,7},authored[1]));
        auto aliased=authored;aliased[1].owner=alias;
        EXPECT_FALSE(Membership::reuses(canonical,aliased));
        EXPECT_FALSE(Membership::same({foreign,a,7},authored[1]));
    }
}
TEST(RetainedStorageRequestWorker,EveryMainSourceRoleArmsBeforeLiveStateMutation){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);{Runner fixture(source);auto& boundary=*fixture.boundary;
            // An admission exception stops the actual prelude before mailbox/KV
            // mutation. No preceding prefill or complete generation admission
            // is available to hide a missing first-source arm.
            boundary.failAdmission=true;
            for(auto role:{ForwardExecutionRole::MainInference,ForwardExecutionRole::MTPCondition,ForwardExecutionRole::GroupedMTPVerifier}){
                ForwardInput input{};input.execution_role=role;
                const auto before=boundary.admissions;
                EXPECT_THROW(Peer::prelude(*fixture.runner,input,boundary.stream),std::runtime_error);
                EXPECT_EQ(boundary.admissions,before+1);
                EXPECT_FALSE(source.worker().ownsCurrentThread());
            }
        }source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedStorageRequestWorker,AdmissionDispatchesOriginalOffAndOnWorkerAndPropagatesFailures){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);{Runner fixture(source);auto& worker=source.worker();auto& runner=*fixture.runner;auto& boundary=*fixture.boundary;
            const auto before=worker.submissions;EXPECT_TRUE(Peer::admit(runner,boundary.stream));EXPECT_EQ(worker.submissions,before+1);
            // Active intervals still cross the host enqueue boundary on each
            // normal forward entry; only the real boundary owns epoch idempotence.
            EXPECT_TRUE(Peer::admit(runner,boundary.stream));EXPECT_EQ(worker.submissions,before+2);
            worker.submitAndWait([&]{const auto entered=worker.submissions;EXPECT_TRUE(Peer::admit(runner,boundary.stream));EXPECT_EQ(worker.submissions,entered);});
            boundary.failAdmission=true;EXPECT_THROW(Peer::admit(runner,boundary.stream),std::runtime_error);EXPECT_FALSE(worker.ownsCurrentThread());
            const std::shared_ptr<PhysicalMemoryAuthority> foreign(source.original->memory().get(),[](auto*){});Peer::foreignMemory(runner,foreign);
            const auto calls=boundary.admissions;EXPECT_THROW(Peer::admit(runner,boundary.stream),std::invalid_argument);EXPECT_EQ(boundary.admissions,calls);
            Peer::foreignMemory(runner,source.original->memory());
        }source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedStorageRequestWorker,ResetClosesOnWorkerAndAggregatesOnlyAfterOriginalTerminalEvent){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);{Runner fixture(source);Backend backend(device.type);backend.boundary=fixture.boundary;backend.worker=&source.worker();backend.event=Peer::terminalEvent(*fixture.runner);native=&backend;
            auto& boundary=*fixture.boundary;ASSERT_TRUE(Peer::admit(*fixture.runner,boundary.stream));boundary.trace.clear();Peer::reset(*fixture.runner,boundary.stream);EXPECT_EQ(boundary.trace,(std::vector<int>{2,3,4,5}));
            boundary.trace.clear();Peer::reset(*fixture.runner,boundary.stream);EXPECT_TRUE(boundary.trace.empty());
            boundary.active=true;boundary.failClose=true;EXPECT_THROW(Peer::reset(*fixture.runner,boundary.stream),std::runtime_error);EXPECT_EQ(boundary.trace,(std::vector<int>{2}));
            boundary.failClose=false;boundary.trace.clear();boundary.active=true;source.worker().submitAndWait([&]{EXPECT_THROW(Peer::reset(*fixture.runner,boundary.stream),std::logic_error);});EXPECT_TRUE(boundary.trace.empty());
            boundary.failAwait=true;boundary.trace.clear();boundary.active=true;
            try{Peer::reset(*fixture.runner,boundary.stream);FAIL()<<"Missing original terminal exception";}
            catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"original terminal failure");}
            EXPECT_EQ(boundary.trace,(std::vector<int>{2,3,4,5}));
            native=nullptr;
        }source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedStorageRequestWorker,GroupedResetPublishesEveryOriginalCloseBeforeFirstServiceAwait){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);
        {
            Runner first(source),delayed(source);
            Backend backend(device.type);native=&backend;backend.worker=&source.worker();
            auto select=[&](Runner& fixture){backend.boundary=fixture.boundary;backend.event=Peer::terminalEvent(*fixture.runner);};
            ASSERT_TRUE(Peer::admit(*first.runner,first.boundary->stream));
            ASSERT_TRUE(Peer::admit(*delayed.runner,delayed.boundary->stream));
            first.boundary->trace.clear();delayed.boundary->trace.clear();
            first.boundary->beforeAwait=[&]{EXPECT_EQ(delayed.boundary->closes,1);};
            select(first);Peer::publishResetClose(*first.runner,first.boundary->stream);
            EXPECT_EQ(first.boundary->trace,(std::vector<int>{2,3,4}));
            EXPECT_TRUE(delayed.boundary->trace.empty());
            EXPECT_TRUE(first.boundary->active); // Early completion owns no peer's await right.
            select(delayed);Peer::publishResetClose(*delayed.runner,delayed.boundary->stream);
            EXPECT_EQ(delayed.boundary->trace,(std::vector<int>{2,3,4}));
            select(first);Peer::reset(*first.runner,first.boundary->stream);
            select(delayed);Peer::reset(*delayed.runner,delayed.boundary->stream);
            EXPECT_EQ(first.boundary->trace,(std::vector<int>{2,3,4,5}));
            EXPECT_EQ(delayed.boundary->trace,(std::vector<int>{2,3,4,5}));
            EXPECT_EQ(first.boundary->closes,1);EXPECT_EQ(delayed.boundary->closes,1);
            native=nullptr;
        }
        source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedStorageRequestWorker,RankRetiresEveryOriginalPeerAndGroupWhilePreservingFirstAwaitFailure){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);
        {
            Runner first(source),peer(source);
            Backend backend(device.type);native=&backend;backend.worker=&source.worker();
            auto select=[&](Runner& fixture){backend.boundary=fixture.boundary;backend.event=Peer::terminalEvent(*fixture.runner);};
            ASSERT_TRUE(Peer::admit(*first.runner,first.boundary->stream));
            ASSERT_TRUE(Peer::admit(*peer.runner,peer.boundary->stream));
            select(first);Peer::publishResetClose(*first.runner,first.boundary->stream);
            select(peer);Peer::publishResetClose(*peer.runner,peer.boundary->stream);
            first.boundary->awaitFailure=std::make_exception_ptr(std::range_error("first original source await"));
            peer.boundary->awaitFailure=std::make_exception_ptr(std::logic_error("later peer source await"));
            int finalRetirements=0,dataResets=0;const std::array originals{&first,&peer};
            try{
                RankRetainedStorageRetirementReplayAccess::retire(originals,[&](Runner* fixture){
                    select(*fixture);Peer::awaitOriginal(*fixture->runner);
                },[&]{
                    ++finalRetirements;
                    EXPECT_FALSE(first.boundary->active);EXPECT_FALSE(peer.boundary->active);
                    throw std::runtime_error("later group retirement diagnostic");
                });
                ++dataResets; // Same phase boundary as actual Rank full data reset.
                FAIL()<<"Missing original first retirement failure";
            }catch(const std::range_error& error){EXPECT_STREQ(error.what(),"first original source await");}
            catch(...){FAIL()<<"Peer or group retirement masked the original failure";}
            EXPECT_EQ(first.boundary->trace.back(),5);EXPECT_EQ(peer.boundary->trace.back(),5);
            EXPECT_EQ(first.boundary->closes,1);EXPECT_EQ(peer.boundary->closes,1);
            EXPECT_EQ(finalRetirements,1);EXPECT_EQ(dataResets,0);native=nullptr;
        }
        source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedStorageRequestWorker,WholePublisherUsesOriginalWorkerAndRejectsForeignContextBeforeCapture){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);{auto boundary=std::make_shared<Boundary>(*source.original);Backend backend(device.type);backend.boundary=boundary;backend.worker=&source.worker();
            std::array<int,sampling_math::kDeviceGenerationControlCount> rows{};sampling_math::DeviceGenerationDispatchTicket ticket{};
            const DeviceGenerationGraphControl control{sampling_math::DeviceGenerationPolicy::ordinary(),rows.data(),sampling_math::kDeviceGenerationControlCount,1};std::string error;std::uint32_t due=0;backend.words=rows.data();backend.ticketRows=&ticket;backend.maintenance=&due;boundary->expectedHealthy=rows.data()+sampling_math::kDeviceGenerationControlOk;boundary->expectedComplete=rows.data()+sampling_math::kDeviceGenerationControlRequestComplete;
            auto build=[&](Graph& graph,IWorkerGPUContext& worker){return DeviceGenerationGraphProgram::ticketPublisher(graph,worker,backend,device,control,&due,&ticket,error,boundary.get());};
            Graph graph(source.worker(),boundary->stream);EXPECT_TRUE(build(graph,source.worker()));EXPECT_EQ(graph.begins,1);EXPECT_EQ(graph.ends,1);EXPECT_EQ(graph.instances,1);
            Graph on(source.worker(),boundary->stream);source.worker().submitAndWait([&]{const auto count=source.worker().submissions;EXPECT_TRUE(build(on,source.worker()));EXPECT_EQ(source.worker().submissions,count);});
            testing::MockWorkerGPUContext foreign(device.ordinal);Graph rejected(source.worker(),boundary->stream);EXPECT_THROW(build(rejected,foreign),std::invalid_argument);EXPECT_EQ(rejected.begins,0);
            Graph failed(source.worker(),boundary->stream);backend.failTicket=true;EXPECT_THROW(build(failed,source.worker()),std::runtime_error);EXPECT_EQ(failed.begins,1);EXPECT_EQ(failed.ends,1);EXPECT_EQ(failed.instances,0);EXPECT_FALSE(isGraphCaptureActive());EXPECT_FALSE(source.worker().isDeviceGraphCaptureActive());
            Graph nested(source.worker(),boundary->stream);{GraphCaptureGuard outer;EXPECT_FALSE(build(nested,source.worker()));EXPECT_EQ(nested.begins,0);}
        }source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test A broken dispatcher rejects before recursion or any native publisher operation. */
TEST(RetainedStorageRequestWorker,BrokenPublisherDispatchRejectsBeforeRecursiveCapture)
{
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device);{
            auto boundary=std::make_shared<Boundary>(*source.original);Backend backend(device.type);
            backend.boundary=boundary;backend.worker=&source.worker();
            std::array<int,sampling_math::kDeviceGenerationControlCount> rows{};sampling_math::DeviceGenerationDispatchTicket ticket{};
            const DeviceGenerationGraphControl control{sampling_math::DeviceGenerationPolicy::ordinary(),rows.data(),sampling_math::kDeviceGenerationControlCount,1};
            std::string error;Graph graph(source.worker(),boundary->stream);
            const auto before=source.worker().submissions;source.worker().brokenDispatch=true;
            EXPECT_THROW((void)DeviceGenerationGraphProgram::ticketPublisher(graph,source.worker(),backend,device,
                control,nullptr,&ticket,error,boundary.get()),std::logic_error);
            source.worker().brokenDispatch=false;
            EXPECT_EQ(source.worker().submissions,before+1);EXPECT_EQ(source.worker().brokenEntries,1);
            EXPECT_EQ(graph.begins,0);EXPECT_EQ(graph.ends,0);EXPECT_EQ(graph.instances,0);
            EXPECT_EQ(backend.tickets,0);EXPECT_EQ(boundary->closes,0);
            EXPECT_FALSE(source.worker().isDeviceGraphCaptureActive());EXPECT_FALSE(isGraphCaptureActive());
        }source.retire();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @test Fixed1/Fixed3/Dynamic reuse the same complete publisher and original terminal interval. */
TEST(RetainedStorageRequestWorker,ActivePoliciesReuseOriginalWholePublisherAndTerminalPredicates){
 GTEST_FLAG_SET(death_test_style,"threadsafe");
 ASSERT_EXIT(([]{
  auto dynamic=sampling_math::DeviceGenerationPolicy::fixed(1);dynamic.mode=sampling_math::DeviceGenerationPolicyMode::Dynamic;dynamic.maximum_depth=15;
  const std::array policies{sampling_math::DeviceGenerationPolicy::fixed(1),sampling_math::DeviceGenerationPolicy::fixed(3),dynamic};
  for(const auto device:{DeviceId::cuda(82),DeviceId::rocm(82)}){
   Source source(device);{
    auto boundary=std::make_shared<Boundary>(*source.original);Backend backend(device.type);backend.boundary=boundary;backend.worker=&source.worker();
    std::array<int,sampling_math::kDeviceGenerationControlCount> words{};sampling_math::DeviceGenerationDispatchTicket ticket{};std::uint32_t due=0;
    backend.words=words.data();backend.ticketRows=&ticket;backend.maintenance=&due;
    boundary->expectedHealthy=words.data()+sampling_math::kDeviceGenerationControlOk;
    boundary->expectedComplete=words.data()+sampling_math::kDeviceGenerationControlRequestComplete;
    for(const auto& policy:policies){
     ASSERT_TRUE(policy.valid());const DeviceGenerationGraphControl control{policy,words.data(),sampling_math::kDeviceGenerationControlCount,1};
     const auto closes=boundary->closes;Graph graph(source.worker(),boundary->stream);std::string error;
     ASSERT_TRUE(DeviceGenerationGraphProgram::ticketPublisher(graph,source.worker(),backend,device,control,&due,&ticket,error,boundary.get()));
     EXPECT_EQ(graph.begins,1);EXPECT_EQ(graph.ends,1);EXPECT_EQ(graph.instances,1);
     EXPECT_EQ(boundary->closes,closes+1);EXPECT_EQ(boundary->admissions,0);EXPECT_FALSE(boundary->active);
     EXPECT_FALSE(source.worker().isDeviceGraphCaptureActive());EXPECT_FALSE(isGraphCaptureActive());
    }
    EXPECT_EQ(backend.tickets,3);EXPECT_EQ(boundary->closes,3);
   }source.retire();
  }
  std::_Exit(::testing::Test::HasFailure()?2:0);
 }()),::testing::ExitedWithCode(0),"");
}

/** @test Both native backends bind the caller's original whole-parent owner, never a child/default stream or foreign PMA. */
TEST(RetainedStorageRequestWorker,ActiveParentBindingKeepsOriginalStreamAndRejectsForeignAuthority){
 GTEST_FLAG_SET(death_test_style,"threadsafe");ASSERT_EXIT(([]{
  for(const auto device:{DeviceId::cuda(83),DeviceId::rocm(83)}){
   Source source(device);{
    Runner r(source,true);auto parent=std::shared_ptr<void>(new int(17),[](void* p){delete static_cast<int*>(p);});
    auto child=std::shared_ptr<void>(new int(19),[](void* p){delete static_cast<int*>(p);});std::string error;
    ASSERT_TRUE(Peer::bindParent(*r.runner,parent,&error));ASSERT_TRUE(r.boundary->parent);
    EXPECT_EQ(r.boundary->parent->stream(),parent.get());EXPECT_NE(r.boundary->parent->stream(),child.get());
    EXPECT_FALSE(r.boundary->parent->owner().owner_before(parent));EXPECT_FALSE(parent.owner_before(r.boundary->parent->owner()));
    EXPECT_EQ(r.boundary->admissions,0);EXPECT_EQ(r.boundary->closes,0);EXPECT_FALSE(r.boundary->active);
    auto alias=std::shared_ptr<PhysicalMemoryAuthority>(source.original->memory().get(),[](auto*){});
    Peer::foreignMemory(*r.runner,alias);EXPECT_FALSE(Peer::bindParent(*r.runner,parent,&error));
    EXPECT_NE(error.find("same-PMA"),std::string::npos);Peer::foreignMemory(*r.runner,source.original->memory());
    EXPECT_THROW(Peer::bindParent(*r.runner,{},&error),std::invalid_argument);
    for(const auto value:{std::uintptr_t{1},std::uintptr_t{2}}){auto implicit=std::shared_ptr<void>(reinterpret_cast<void*>(value),[](void*){});
     EXPECT_THROW(Peer::bindParent(*r.runner,implicit,&error),std::invalid_argument);}
   }source.retire();
  }
  std::_Exit(::testing::Test::HasFailure()?2:0);
 }()),::testing::ExitedWithCode(0),"");
}

}
}

namespace llaminar2::test {
/** @brief Replay real rank preparation, source-worker arm/adoption/reuse/reset and failure cancellation without a native runtime. */
TEST(RetainedStorageRequestWorker,RankMainConditionArmsOriginalGroupBeforeFanoutAndAbortsFailures){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})ASSERT_EXIT(([&]{
        {
        RankSourceProgram original(type);
        {
            RankSourceCohort success(original,0);
            const auto access=RankRetainedStorageRetirementReplayAccess{};
            EXPECT_TRUE(access.condition(*success.rank,false,[](IInferenceRunner&,std::size_t){return true;}));
            EXPECT_EQ(access.admission(*success.rank),nullptr);
            ASSERT_TRUE(success.forward());
            const auto first=access.admission(*success.rank);ASSERT_NE(first,nullptr);
            ASSERT_TRUE(success.forward());EXPECT_EQ(access.admission(*success.rank),first);
            for(auto* code:success.kernels)EXPECT_EQ(code->admissions,1);
            success.close();
            for(auto* child:success.children)Peer::epoch(*child,8);
            ASSERT_TRUE(success.forward());
            for(auto* code:success.kernels)EXPECT_EQ(code->admissions,2);
            success.close();
        }
        for(const auto throws:{false,true}){
            RankSourceCohort failed(original,throws?2:1);
            if(throws)EXPECT_THROW(failed.forward(0,true),std::runtime_error);
            else EXPECT_FALSE(failed.forward(0));
            EXPECT_EQ(RankRetainedStorageRetirementReplayAccess::admission(*failed.rank),nullptr);
            failed.close(true);
            for(const auto& boundary:failed.boundaries)EXPECT_FALSE(boundary->hasActiveInterval());
        }
        }
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
}
