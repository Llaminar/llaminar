/**
 * @file Test__PooledImmutableCompleteCohort.cpp
 * @brief Actual complete-cohort admission and original-owner lifetime replay without vendor APIs.
 *
 * Only backend calls, capture handles and device-context APIs are doubled. The
 * original PMA/root factories, actual graph appenders/guards, boundary admission,
 * real retained service worker and submission-retirement bodies execute here.
 * Recorded terminal acknowledgement below stands in for a device kernel; it
 * does not certify captured native execution, traffic or a full model factory.
 */
#include "PooledImmutableReadGraphFixture.h"
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include "collective/OriginalNativeGraphCaptureGroup.h"
#include "mocks/MockLocalTPContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include <barrier>
#include <future>
#include <condition_variable>
#include <deque>
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "execution/qwen4exp/PooledImmutableDirectedLookupGraph.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
namespace llaminar2::test {
/** @brief Context identity only; every data/API operation remains unavailable to this host proof. */
class CohortContext final:public IGPUDeviceContext {
public:
    explicit CohortContext(DeviceId device):IGPUDeviceContext(device,device.ordinal,device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM){}
    std::string deviceName()const override{return "CohortApiDouble";}
    void synchronize()override{throw std::logic_error("Unit cannot synchronize a GPU");}
    void barrier()override{throw std::logic_error("Unit cannot synchronize a GPU");}
    void* allocate(size_t)override{throw std::logic_error("Unit cannot allocate a GPU");}
    void free(void*)override{std::terminate();}
    size_t availableMemory()const override{throw std::logic_error("Unit cannot query a GPU");}
    size_t totalMemory()const override{throw std::logic_error("Unit cannot query a GPU");}
    bool copyToDevice(void*,const void*,size_t,void*)override{return false;}
    bool copyToHost(void*,const void*,size_t,void*)override{return false;}
    bool copyFromDevice(void*,const void*,size_t,IDeviceContext*,void*,void*)override{return false;}
};
/** @brief Dedicated queued worker, preserving the actual native actor wait topology without vendor APIs. */
class QueuedCohortWorker final:public testing::MockWorkerGPUContext {
public:
    explicit QueuedCohortWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack):MockWorkerGPUContext(ordinal){
        auto started=ready_.get_future();
        worker_.start(std::move(stack),[](void* argument){static_cast<QueuedCohortWorker*>(argument)->run();},this);
        started.get();
    }
    ~QueuedCohortWorker()override{{std::lock_guard lock(mutex_);stopping_=true;}changed_.notify_one();worker_.join();}
    bool ownsCurrentThread()const noexcept override{return std::this_thread::get_id()==identity_;}
    void submitAndWait(std::function<void()> work)override{
        if(forbidCrossWait.load() && current_ && current_!=this){std::fputs("ADMISSION_ATTEMPTED_WORKER_HELD_PEER_WAIT\n",stderr);std::abort();}
        if(ownsCurrentThread())work();else submitAsync(std::move(work)).get();
    }
    inline static std::atomic<bool> forbidCrossWait=false;
    std::atomic<bool> failNextAsyncBeforeEntry=false; ///< Return a worker-wrapper failure without invoking its callback.
    std::atomic<bool> rejectNextAsync=false; ///< Inject only CPU queue admission failure, never native execution state.
    std::future<void> submitAsync(std::function<void()> work)override{
        if(rejectNextAsync.exchange(false))throw std::runtime_error("original queued worker rejected submission");
        if(failNextAsyncBeforeEntry.exchange(false)) {
            std::promise<void> failed;auto result=failed.get_future();
            failed.set_exception(std::make_exception_ptr(std::runtime_error("original worker wrapper failed before callback")));
            return result;
        }
        std::packaged_task<void()> job(std::move(work));auto result=job.get_future();
        {std::lock_guard lock(mutex_);if(stopping_)throw std::logic_error("Queued fixture worker retired");jobs_.push_back(std::move(job));}changed_.notify_one();return result;
    }
private:
    inline static thread_local QueuedCohortWorker* current_=nullptr;
    std::mutex mutex_;std::condition_variable changed_;std::deque<std::packaged_task<void()>> jobs_;bool stopping_=false;
    /** @brief Actual original mapped stack executes the existing queued actor protocol. */
    void run(){identity_=std::this_thread::get_id();ready_.set_value();
        for(;;){std::packaged_task<void()> job;{std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stopping_ || !jobs_.empty();});if(stopping_ && jobs_.empty())return;job=std::move(jobs_.front());jobs_.pop_front();}current_=this;job();current_=nullptr;}
    }
    std::promise<void> ready_;std::thread::id identity_;PreparedGPUWorkerThread worker_;
};
/** @brief Closed recorded recipe set; payload shares only between declared nonconcurrent groups. */
struct CohortRecipe {
    std::vector<uint32_t> capacities;
    uint32_t local=q::adaptive_lookup::None;
    DeviceType source=DeviceType::CUDA;
    unsigned group=0,definitions=1,retirementClaims=1;
};
/** @brief One original Preparing source and final PMA for all process-retained instances of one test. */
class CohortPrograms final {
public:
    explicit CohortPrograms(std::vector<CohortRecipe> recipes):backends(std::make_shared<OwnerChainBackendScope>()),recipes_(std::move(recipes)){
        if(active_ || recipes_.empty())throw std::logic_error("Cohort recipe scope must be original and nonempty");
        std::vector<PhysicalMemoryPlan> payloads;std::map<unsigned,PhysicalMemoryPlanBuilder> groups;
        PhysicalMemoryPlanBuilder retained;
        std::uint64_t ioOrdinal=0;
        for(const auto& recipe:recipes_){
            SourceGraphCase declaration(recipe.capacities,true,recipe.local,true,recipe.source,backends,nullptr,{},SourceGraphRecipePhase::DeclareOnly,ioOrdinal++);
            for(const auto& row:declaration.declaredProgram.resources()){
                for(unsigned i=0;i<static_cast<unsigned>(PhysicalMemoryOwner::Count);++i){const auto owner=static_cast<PhysicalMemoryOwner>(i);const auto charge=row.charge(owner);
                    auto& out=(owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent||owner==PhysicalMemoryOwner::RetainedHostServiceRuntime)?retained:groups[recipe.group];
                    out.add(row.resource(),owner,charge.planned_bytes,charge.already_resident_bytes);
                }
            }
            std::vector<DeviceId> members{DeviceId{recipe.source,0}};
            for(const auto& owner:declaration.physical.owners())if(std::find(members.begin(),members.end(),owner.allocator.device)==members.end())members.push_back(owner.allocator.device);
            for(const auto device:members){
                if(std::find(devices_.begin(),devices_.end(),device)==devices_.end())devices_.push_back(device);
                retained.add({0,device,1ULL<<30,1ULL<<30},PhysicalMemoryOwner::NativeGraphExecutable,recipe.definitions*GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            }
            for(unsigned claim=0;claim<recipe.retirementClaims;++claim)for(size_t i=0;i<declaration.physical.owners().size();++i)if(i!=recipe.local)
                PooledImmutableFactoryReplayAccess::addOrdering(retained,NativeOrderingEventProgram::lookupSubmissionRetirement(),{0,DeviceId::cpu(),1ULL<<30,1ULL<<30},{0,declaration.physical.owners()[i].allocator.device,1ULL<<30,1ULL<<30});
        }
        for(auto& [_,group]:groups)payloads.push_back(group.build());
        std::vector<PhysicalMemoryResource> observed{{0,DeviceId::cpu(),1ULL<<30,1ULL<<30}};
        for(const auto device:devices_){observed.push_back({0,device,1ULL<<30,1ULL<<30});NativeContextPreparationReplayAccess::seed(device,[device](auto stack){return NativeContextPreparationReplayAccess::construct<QueuedCohortWorker>(std::move(stack),device.ordinal);});}
        source_=NativeContextPreparationReplayAccess::source<QueuedCohortWorker>(devices_,observed);
        for(const auto device:devices_)permits_.push_back(source_->acquire(device));source_->freezeForPlanning();
        PhysicalMemoryPlanBuilder plan(source_->mandatoryBOM());plan.addMutuallyExclusive(payloads);const auto native=retained.build();for(const auto& row:native.resources())plan.add(row);
        source_->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        active_=this;
    }
    /** @brief Retire the exact source roster after graph/worker borrows are gone. */
    ~CohortPrograms(){
        active_=nullptr;permits_.clear();
        for(const auto device:devices_)PooledImmutableFactoryReplayAccess::retireFixtureGeneration(device);
    }
    /** @return The original recorded case token; later constructors cannot add an undeclared program. */
    std::shared_ptr<SourceGraphCase> make(const std::vector<uint32_t>& capacities,uint32_t local,DeviceType type){
        if(next_>=recipes_.size())throw std::logic_error("Undeclared original cohort case");
        const auto ordinal=next_++;const auto& expected=recipes_[ordinal];if(capacities!=expected.capacities || local!=expected.local || type!=expected.source)throw std::logic_error("Original cohort recipe order/geometry drift");
        return std::make_shared<SourceGraphCase>(capacities,true,local,true,type,backends,nullptr,source_->memory(),SourceGraphRecipePhase::Materialize,ordinal);
    }
    /** @return Immutable original prepared permit, never ordinary worker adoption. */
    const PreparedGPUContextUsePermit& permit(DeviceId device)const{for(const auto& p:permits_)if(p.device()==device)return p;throw std::logic_error("Missing original participant");}
    static CohortPrograms& active(){if(!active_)throw std::logic_error("Cohort case requires its declared original source set");return *active_;}
    std::shared_ptr<OwnerChainBackendScope> backends;
private:
    inline static thread_local CohortPrograms* active_=nullptr;
    std::vector<CohortRecipe> recipes_;size_t next_=0;
    std::vector<DeviceId> devices_; ///< Exact original source roster, including ordinals beyond the legacy mock sweep.
    std::shared_ptr<NativeContextPreparation> source_;std::vector<PreparedGPUContextUsePermit> permits_;
};

/** @brief Closed recorded groups enter the same validated production binding body. */
struct PooledImmutableCompleteCohortReplayAccess {
    static auto bind(q::PooledImmutableCohortParticipant source,PooledImmutableRetirementCompletion terminal,
        std::vector<q::PooledImmutableCohortRemote> remotes,std::vector<std::unique_ptr<NativeOrderingEventGroup>> groups){
        return q::PooledImmutableCompleteCohort::bindWithOrderingGroups(std::move(source),std::move(terminal),std::move(remotes),std::move(groups));
    }
};
/** @brief Actual entire original read and terminal graph, retaining the original fixture as each graph/arena owner. */
struct CohortCase {
    inline static std::atomic<unsigned> capturesCreated=0,capturesDestroyed=0,expiredAtCaptureDestruction=0;
    std::shared_ptr<SourceGraphCase> original;
    PooledImmutableRetirementCompletion terminal;
    q::PooledImmutableCohortParticipant source;
    std::vector<q::PooledImmutableCohortRemote> remotes;
    std::shared_ptr<q::PooledImmutableCompleteCohort> binding;
    explicit CohortCase(std::vector<uint32_t> capacities,uint32_t local=q::adaptive_lookup::None,DeviceType sourceType=DeviceType::CUDA):original(CohortPrograms::active().make(capacities,local,sourceType)) {
        auto& c=*original;
        c.coordCode->initialize(*c.coordinator,c.originalSourceStream);
        for(const auto& owner:c.owners)(owner->plan().allocator().device==c.boundary->device()?c.coordCode:c.ownerCode)->initialize(*owner,c.originalOwnerStreams[&owner-&c.owners[0]]);
        c.readOutputLease=c.memory->claimNewAllocation(c.boundary->device(),PhysicalMemoryOwner::ActivationArena,16*160*sizeof(float));
        TransferEngine::prepareDeviceOutput(c.readOutput.get(),c.boundary->device(),c.originalSourceStream);
        const auto fill=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);
        const auto ready=appendPooledPublicationCoordinator(c.graph,c.arena,fill,publicationMembers(c),c.coordCode,"publication");
        std::vector<q::PooledImmutableReadParticipant> readers;
        for(size_t i=0;i<c.owners.size();++i) {
            if(i==local)readers.emplace_back(c.localBinding);
            else readers.emplace_back(q::PooledImmutableCapturedReadWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(ready.owners[i]),c.memory));
        }
        const auto read=appendPooledReadCoordinator(c.graph,c.arena,ready,readers,{c.readOutput,BufferId::QFN_PLE_EMBEDDING,0,16*160*sizeof(float)},"read","actual_output");
        std::vector<q::PooledImmutableRetirementParticipant> members;
        for(size_t i=0;i<readers.size();++i) {
            if(i==local)members.emplace_back(c.localBinding);
            else members.emplace_back(q::PooledImmutableCapturedRetirementWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(readers[i]),c.memory));
        }
        terminal=appendPooledRetirementCoordinator(c.graph,c.arena,read,members,c.coordCode,"terminal");
        c.graph.setRetainedStorageBoundary(c.boundary);
        source=participant(std::shared_ptr<ComputeGraph>(original,&c.graph),std::shared_ptr<BufferArena>(original,&c.arena),c.boundary->device(),c.originalSourceStream);
        for(uint32_t i=0;i<c.owners.size();++i)if(i!=local) {
            auto& graph=*c.ownerGraphs[i];auto& arena=*c.ownerArenas[i];const auto holder=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(members[i]);
            const auto filled=appendPooledSourceOwnerFill(graph,arena,c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source");
            const auto published=appendPooledPublicationOwner(graph,arena,holder->read()->publication(),c.ownerCode,filled.last,"publication");
            const auto rows=appendPooledReadOwner(graph,arena,holder->read(),c.ownerCode,published.last,"read");
            const auto retired=appendPooledRetirementOwner(graph,arena,holder,c.ownerCode,rows.last,"terminal");
            const auto device=c.owners[i]->plan().allocator().device;
            remotes.push_back({i,participant(std::shared_ptr<ComputeGraph>(original,&graph),std::shared_ptr<BufferArena>(original,&arena),device,c.originalOwnerStreams[i]),retired.last});
        }
    }
    /** @brief Prepare actual leaves before first service.arm, then provide a cold capture API double. */
    static q::PooledImmutableCohortParticipant participant(std::shared_ptr<ComputeGraph> graph,std::shared_ptr<BufferArena> arena,DeviceId device,void* stream) {
        auto worker=CohortPrograms::active().permit(device).worker();
        auto context=std::make_shared<CohortContext>(device);
        const std::weak_ptr<ComputeGraph> retainedGraph=graph;const std::weak_ptr<IWorkerGPUContext> retainedWorker=worker;
        const auto mock=std::dynamic_pointer_cast<testing::MockWorkerGPUContext>(worker);if(!mock)throw std::logic_error("Device-free recorder needs its registered API-double worker");
        mock->capture_created=[] {++capturesCreated;};
        mock->capture_destroyed=[retainedGraph,retainedWorker]{++capturesDestroyed;if(retainedGraph.expired() || retainedWorker.expired())++expiredAtCaptureDestruction;};
        auto executable=RetainedParticipantGraphExecutableFixtureAccess::recorded(CohortPrograms::active().permit(device));
        return {worker,context,arena,graph,worker->defaultStream(),std::move(executable)};
    }
    /** @brief A retired definition is never reset: retry consumes a separately predeclared original instance. */
    void renewDefinitions(){
        source.executable=RetainedParticipantGraphExecutableFixtureAccess::recorded(CohortPrograms::active().permit(original->boundary->device()));
        for(auto& remote:remotes)remote.participant.executable=RetainedParticipantGraphExecutableFixtureAccess::recorded(CohortPrograms::active().permit(original->owners[remote.coordinate]->plan().allocator().device));
    }
    /** @brief Cohort records unchanged graphs through the supplied sole original recording owners. */
    std::shared_ptr<q::PooledImmutableCompleteCohort> bindOriginal(){
        std::vector<std::unique_ptr<NativeOrderingEventGroup>> groups;
        for(size_t i=0;i<original->owners.size();++i)if(i!=original->localCoordinate)
            groups.push_back(PooledImmutableFactoryReplayAccess::claim(original->memory,original->owners[i]->plan().allocator().device,NativeOrderingEventProgram::lookupSubmissionRetirement()));
        return PooledImmutableCompleteCohortReplayAccess::bind(source,terminal,remotes,std::move(groups));
    }
    void bind(){
        if(binding){binding=q::PooledImmutableCompleteCohort::bind(source,terminal,remotes);return;}
        binding=bindOriginal();
    }
    /** @brief Actual boundary admits under the retained original source worker. */
    void admit(uint64_t generation){source.worker->submitAndWait([&]{original->boundary->beginAdmission(generation,source.stream);});}
    /** @brief Recorded no-work terminal acknowledgement at the device API-double boundary, then actual worker join. */
    void terminalJoin() {
        const auto view=original->boundary->pooledKernels().source().deviceView();
        const auto* io=static_cast<const q::ImmutableLookupDeviceState*>(original->boundary->pooledStorage()->roots().native->gpu_data_ptr());
        std::atomic_ref(view.base.acknowledgement->closedGeneration).store(io->admissionGeneration,std::memory_order_release);
        original->boundary->awaitTerminal();
    }
};
/** @brief Replay one original cohort cell under the standard per-process timeout. */
void replayCohort(size_t count,bool local,bool empty) {
    std::vector<uint32_t> capacities(count);for(size_t i=0;i<count;++i)capacities[i]=empty?0:i%3==1?0:3;
    CohortCase c(capacities,local?0:q::adaptive_lookup::None);
    EXPECT_FALSE(c.original->boundary->hasActiveInterval());EXPECT_THROW(c.admit(1),std::logic_error);
    const auto identity=c.original->boundary->captureIdentity();const auto roots=c.original->boundary->pooledStorage()->roots();
    c.bind();EXPECT_NO_THROW(c.binding->requireCurrent());
    for(uint64_t generation:{1u,2u,3u}) {
        c.admit(generation);EXPECT_TRUE(c.original->boundary->hasActiveInterval());EXPECT_NO_THROW(c.admit(generation));
        EXPECT_NO_THROW(c.binding->launch());
        EXPECT_THROW(c.admit(generation+1),std::logic_error);c.terminalJoin();EXPECT_FALSE(c.original->boundary->hasActiveInterval());
    }
    EXPECT_EQ(c.original->boundary->captureIdentity(),identity);EXPECT_EQ(c.original->boundary->pooledStorage()->roots().native,roots.native);
    c.binding.reset();EXPECT_THROW(c.admit(4),std::logic_error);
}
TEST(PooledImmutableCompleteCohort, OriginalCohortP1RemotePositive){
    CohortPrograms programs({{{3},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(1,false,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP1RemoteEmpty){
    CohortPrograms programs({{{0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(1,false,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP2RemotePositive){
    CohortPrograms programs({{{3,0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(2,false,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP2RemoteEmpty){
    CohortPrograms programs({{{0,0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(2,false,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP2LocalPositive){
    CohortPrograms programs({{{3,0},0,DeviceType::CUDA,0,1}});replayCohort(2,true,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP2LocalEmpty){
    CohortPrograms programs({{{0,0},0,DeviceType::CUDA,0,1}});replayCohort(2,true,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP4RemotePositive){
    CohortPrograms programs({{{3,0,3,3},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(4,false,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP4RemoteEmpty){
    CohortPrograms programs({{{0,0,0,0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(4,false,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP4LocalPositive){
    CohortPrograms programs({{{3,0,3,3},0,DeviceType::CUDA,0,1}});replayCohort(4,true,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP4LocalEmpty){
    CohortPrograms programs({{{0,0,0,0},0,DeviceType::CUDA,0,1}});replayCohort(4,true,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP8RemotePositive){
    CohortPrograms programs({{{3,0,3,3,0,3,3,0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(8,false,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP8RemoteEmpty){
    CohortPrograms programs({{{0,0,0,0,0,0,0,0},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});replayCohort(8,false,true);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP8LocalPositive){
    CohortPrograms programs({{{3,0,3,3,0,3,3,0},0,DeviceType::CUDA,0,1}});replayCohort(8,true,false);}
TEST(PooledImmutableCompleteCohort, OriginalCohortP8LocalEmpty){
    CohortPrograms programs({{{0,0,0,0,0,0,0,0},0,DeviceType::CUDA,0,1}});replayCohort(8,true,true);}
TEST(PooledImmutableCompleteCohort, MissingWrongOrStaleOriginalParticipantsRejectBeforeArm) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,1,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,2,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,3,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,4,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,5,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,6,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,7,1},{{2,0},q::adaptive_lookup::None,DeviceType::CUDA,8,1}});
    for(unsigned defect=0;defect<8;++defect) {
        CohortCase c({2,0,3,1});
        if(defect==0)c.remotes.pop_back();
        if(defect==1)c.remotes[0].coordinate=1;
        if(defect==2)std::swap(c.remotes[0],c.remotes[1]);
        if(defect==3)c.remotes[0].participant.worker=c.source.worker;
        if(defect==4)c.remotes[0].participant.stream=nullptr;
        if(defect==5)c.remotes[0].retiredSent="terminal/retire";
        if(defect==6)c.source.graph->getNode(c.terminal.read.output)->dependencies.clear();
        if(defect==7)c.remotes[0].participant.graph=c.source.graph;
        EXPECT_ANY_THROW(c.bind());EXPECT_FALSE(c.original->boundary->hasActiveInterval());
    }
    CohortCase c({2,0});c.bind();c.remotes[0].participant.graph->getNode(c.remotes[0].retiredSent)->dependencies.clear();
    EXPECT_ANY_THROW(c.admit(1));EXPECT_FALSE(c.original->boundary->hasActiveInterval());
}
TEST(PooledImmutableCompleteCohort, ExternalLastBindingCannotReleaseAnActiveOriginalCapture) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    EXPECT_DEATH({CohortCase c({2});c.bind();c.admit(1);c.binding.reset();},"terminate");
    CohortCase c({2});c.bind();std::weak_ptr<SourceGraphCase> owner=c.original;auto alias=c.binding;
    c.binding.reset();c.admit(1);c.terminalJoin();c.original.reset();c.source={};c.remotes.clear();c.terminal={};
    EXPECT_FALSE(owner.expired());alias.reset();EXPECT_TRUE(owner.expired());
}
/** @brief Metadata failure joins the original worker but cannot certify native retirement or release the duty. */
TEST(PooledImmutableCompleteCohort, TerminalMetadataDriftStillJoinsTheOriginalWorker) {
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT(([&] {
        CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
        CohortCase c({2});c.bind();c.admit(1);
        c.source.graph->getNode(c.terminal.read.output)->dependencies.clear();
        EXPECT_THROW(c.terminalJoin(),std::logic_error);
        EXPECT_NO_THROW(c.original->service->stats());
        EXPECT_TRUE(c.original->boundary->hasActiveInterval());
        EXPECT_THROW(c.original->boundary->terminalObservation(),std::logic_error);
        // Intentionally unproven native duty cannot be normally destroyed.
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @brief A missing required distributed authority rejects only after joining the original CPU worker. */
TEST(PooledImmutableCompleteCohort, MissingRequiredTerminalCohortNeverClassifiesSafeRetirement) {
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT(([&] {
        CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
        CohortCase c({2});c.bind();c.admit(1);
        PooledImmutableFactoryReplayAccess::withdrawCohortForTerminalReplay(*c.original->boundary);
        // Missing close makes the original worker fail truthfully. A missing
        // required native authority must outrank safe-retired IO classification.
        EXPECT_THROW(c.original->boundary->awaitTerminal(),std::logic_error);
        EXPECT_NO_THROW(c.original->service->stats());
        EXPECT_TRUE(c.original->boundary->hasActiveInterval());
        EXPECT_THROW(c.original->boundary->terminalObservation(),std::logic_error);
        // The required case deliberately retains its unproven native duty.
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

TEST(PooledImmutableCompleteCohort, PartialRemoteSetupRetiresEverySubmittedFrontierBeforeCancel) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    CohortCase c({2,0,3,1});
    c.original->backends.cuda.resetEventRecords();c.original->backends.rocm.resetEventRecords();
    c.original->backends.rocm.failJoinOrdinal=c.original->owners[1]->plan().allocator().device.ordinal;
    EXPECT_ANY_THROW(c.bind());EXPECT_FALSE(c.original->boundary->hasActiveInterval());
    const auto events=c.original->backends.rocm.getEventRecords();
    for(const auto& remote:c.remotes){const auto device=c.original->owners[remote.coordinate]->plan().allocator().device;
        EXPECT_TRUE(std::any_of(events.begin(),events.end(),[&](const auto& e){return e.type==MockBackend::EventRecord::WAIT && e.device_id==device.ordinal && !e.stream;}));
    }
    EXPECT_THROW(c.admit(1),std::logic_error);
}
TEST(PooledImmutableCompleteCohort, TypedResetUsesOnlyTheOriginalCompleteSourceAndRetainedWorker) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    CohortCase c({2});c.bind();const auto& code=c.original->boundary->pooledKernels();
    const auto* entry=dynamic_cast<const SourceGuardCode*>(&code);ASSERT_NE(entry,nullptr);const auto before=entry->closes;
    EXPECT_THROW(code.closeForReset(*c.original->boundary->pooledStorage(),*c.original->control,c.source.stream),std::invalid_argument);
    EXPECT_EQ(entry->closes,before);
    // Actual host typed-reset body reaches only the original native entry.
    // This API double intentionally throws there; native close/drain is a
    // separate captured device qualification, never invented by this replay.
    c.source.worker->submitAndWait([&]{
        EXPECT_THROW(c.original->boundary->recordTerminalClose(c.source.stream),std::logic_error);
        GraphCaptureGuard capture;
        EXPECT_THROW(c.original->boundary->recordTerminalClose(c.source.stream),std::logic_error);
    });
    EXPECT_EQ(entry->closes,before+2);
    EXPECT_THROW(c.original->boundary->recordTerminalClose(c.source.stream),std::invalid_argument);
    EXPECT_EQ(entry->closes,before+2);
}

TEST(PooledImmutableCompleteCohort, ConstructionFailureDestroysCapturesBeforeOriginalRootsAndContexts) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    CohortCase c({2,0,3,1});const auto before=CohortCase::capturesDestroyed.load();const auto births=CohortCase::capturesCreated.load();
    c.remotes[0].coordinate=1;EXPECT_THROW(c.bind(),std::invalid_argument);
    EXPECT_EQ(CohortCase::capturesDestroyed.load(),before);
    EXPECT_EQ(CohortCase::expiredAtCaptureDestruction.load(),0u);
    EXPECT_FALSE(c.original->boundary->hasActiveInterval());
    EXPECT_EQ(CohortCase::capturesCreated.load(),births);
    // Native reset additionally selects its immutable device itself: actual
    // CUDA CUDAGraphCapture::reset and HIP reset/activateOwner are source-bound
    // audit prerequisites. This mock proves only the original C++ owner order.
}

TEST(PooledImmutableCompleteCohort, ActualRecorderVisitsEveryDeclaredStageAndRejectsReplacementBeforeNativeWork) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    CohortCase c({2,0,3,1});const auto births=CohortCase::capturesCreated.load();c.bind();
    EXPECT_EQ(CohortCase::capturesCreated.load(),births);
    const auto declaredCodeNodes = [](const ComputeGraph &graph) {
        unsigned count = 0;
        for (const auto &name : graph.getExecutionOrder()) {
            const auto *stage = graph.getNode(name)->stage.get();
            if (dynamic_cast<const PooledImmutableSourcePhaseStage *>(stage) ||
                dynamic_cast<const PooledImmutableNativePhaseStage *>(stage))
                ++count;
        }
        return count;
    };
    const auto sourceCodeNodes = declaredCodeNodes(c.original->graph);
    unsigned ownerCodeNodes = 0;
    std::size_t ownerNodes = 0;
    for (const auto &graph : c.original->ownerGraphs)
        if (graph) {
            ownerCodeNodes += declaredCodeNodes(*graph);
            ownerNodes += graph->size();
        }
    const auto *sourceCode = dynamic_cast<const SourceGuardCode *>(&c.original->boundary->pooledKernels());
    ASSERT_NE(sourceCode, nullptr);
    EXPECT_EQ(c.original->coordCode->recordedCount() + sourceCode->launches, sourceCodeNodes);
    EXPECT_EQ(c.original->ownerCode->recordedCount(), ownerCodeNodes);
    const auto sourceTransfers = c.original->backends.cuda.capturedTransferCount();
    const auto ownerTransfers = c.original->backends.rocm.capturedTransferCount();
    EXPECT_EQ(sourceTransfers, c.original->graph.size() - sourceCodeNodes);
    EXPECT_EQ(ownerTransfers, ownerNodes - ownerCodeNodes);
    const auto ownerEvents=c.original->backends.rocm.getEventRecords().size();const auto sourceEvents=c.original->backends.cuda.getEventRecords().size();
    const auto captured=CohortCase::capturesCreated.load();auto held=c.binding;
    EXPECT_THROW(c.bind(),std::logic_error);EXPECT_EQ(c.binding,held);
    EXPECT_EQ(CohortCase::capturesCreated.load(),captured);
    EXPECT_EQ(c.original->backends.cuda.capturedTransferCount(), sourceTransfers);
    EXPECT_EQ(c.original->backends.rocm.capturedTransferCount(), ownerTransfers);
    EXPECT_EQ(c.original->backends.rocm.getEventRecords().size(),ownerEvents);EXPECT_EQ(c.original->backends.cuda.getEventRecords().size(),sourceEvents);
    c.admit(1);c.binding->launch();c.terminalJoin();
}
static_assert(!std::is_constructible_v<q::PooledImmutableCohortParticipant,
    std::shared_ptr<IWorkerGPUContext>,std::shared_ptr<IDeviceContext>,std::shared_ptr<BufferArena>,std::shared_ptr<ComputeGraph>,std::unique_ptr<IGPUGraphCapture>,void*>,
    "Caller-provided empty/unrelated executables cannot enter the complete-cohort declaration");

TEST(PooledImmutableCompleteCohort, PartialColdConstructionAndRecordingRetireOriginalFrontiersWithoutArm) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1},{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,1,1}});
    for(bool eventFailure:{true,false}){
        CohortCase c({2,0,3,1});const auto births=CohortCase::capturesCreated.load();const auto deaths=CohortCase::capturesDestroyed.load();
        if(eventFailure)c.original->backends.rocm.failEventCreationOrdinal=c.original->owners[1]->plan().allocator().device.ordinal;
        else {const auto* code=dynamic_cast<const SourceGuardCode*>(&c.original->boundary->pooledKernels());ASSERT_NE(code,nullptr);code->failRecordedAction=q::PooledSourceAction::PlanReads;}
        c.original->backends.cuda.resetEventRecords();c.original->backends.rocm.resetEventRecords();
        EXPECT_ANY_THROW(c.bind());EXPECT_FALSE(c.original->boundary->hasActiveInterval());EXPECT_FALSE(isGraphCaptureActive());
        EXPECT_EQ(CohortCase::capturesCreated.load()-births,0u);
        EXPECT_EQ(CohortCase::capturesDestroyed.load()-deaths,5u);
        EXPECT_EQ(CohortCase::expiredAtCaptureDestruction.load(),0u);
        const auto events=c.original->backends.rocm.getEventRecords();
        EXPECT_TRUE(std::any_of(events.begin(),events.end(),[](const auto& event){return event.type==MockBackend::EventRecord::WAIT && !event.stream;}));
        EXPECT_THROW(c.admit(1),std::logic_error);
    }
}

TEST(PooledImmutableCompleteCohort, ReplacedPublicStageRejectsBeforeAnyCachedLeafInspection) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    CohortCase c({2});c.bind();const auto name=c.terminal.read.planned;
    auto* node=c.original->graph.getNode(name);const auto topology=c.original->graph.topologyGeneration();
    const auto* prior=node->stage.get();
    node->stage=std::make_unique<PooledImmutableSourcePhaseStage>(c.original->arena,c.original->boundary,q::PooledSourceAction::PlanReads,*c.original->control);
    ASSERT_NE(node->stage.get(),prior);ASSERT_EQ(c.original->graph.topologyGeneration(),topology);
    EXPECT_THROW(c.binding->requireCurrent(),std::logic_error);
    EXPECT_THROW(c.admit(1),std::logic_error);EXPECT_FALSE(c.original->boundary->hasActiveInterval());
}
TEST(PooledImmutableCompleteCohort, ConcurrentColdBindingPublishesOneOriginalOwnerBeforeAnyAdmission) {
    CohortPrograms programs({{{2,0,3,1},q::adaptive_lookup::None,DeviceType::CUDA,0,1,2}});
    CohortCase c({2,0,3,1});const auto births=CohortCase::capturesCreated.load();
    std::array<std::shared_ptr<q::PooledImmutableCompleteCohort>,2> result;
    std::array<std::exception_ptr,2> failure;
    std::barrier start(3);std::atomic<bool> finished=false;
    auto reader=std::async(std::launch::async,[&] {
        while(!finished.load(std::memory_order_acquire)){
            try {c.original->boundary->requireArmedForLaunch();ADD_FAILURE()<<"Cold publication armed a service interval";}
            catch(const std::logic_error&){}
        }
    });
    std::array<std::future<void>,2> writers;
    for(size_t i=0;i<2;++i)writers[i]=std::async(std::launch::async,[&,i]{start.arrive_and_wait();
        try{result[i]=c.bindOriginal();}catch(...){failure[i]=std::current_exception();}
    });
    start.arrive_and_wait();for(auto& writer:writers)writer.get();finished.store(true,std::memory_order_release);reader.get();
    EXPECT_EQ(bool(result[0])+bool(result[1]),1);EXPECT_EQ(bool(failure[0])+bool(failure[1]),1);
    EXPECT_EQ(CohortCase::capturesCreated.load(),births);
    c.binding=result[0]?result[0]:result[1];ASSERT_NE(c.binding,nullptr);c.admit(1);c.binding->launch();c.terminalJoin();
}

TEST(PooledImmutableCompleteCohort, CallerRuleAndOppositeSourceBindingsUseRealQueuedWorkerIdentity) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1},{{2},q::adaptive_lookup::None,DeviceType::ROCm,0,1}});
    {
        CohortCase a({2},q::adaptive_lookup::None,DeviceType::CUDA);
        CohortCase b({2},q::adaptive_lookup::None,DeviceType::ROCm);
        for(const auto& worker:{a.source.worker,b.source.worker}){
            auto mock=std::dynamic_pointer_cast<testing::MockWorkerGPUContext>(worker);
            mock->capture_destroyed=[] {++CohortCase::capturesDestroyed;};
        }
        const auto births=CohortCase::capturesCreated.load();
        ASSERT_NE(std::dynamic_pointer_cast<QueuedCohortWorker>(a.source.worker),nullptr);
        ASSERT_NE(std::dynamic_pointer_cast<QueuedCohortWorker>(b.source.worker),nullptr);
        a.source.worker->submitAndWait([&]{EXPECT_TRUE(a.source.worker->ownsCurrentThread());EXPECT_THROW(a.bind(),std::invalid_argument);});
        a.remotes[0].participant.worker->submitAndWait([&]{EXPECT_TRUE(a.remotes[0].participant.worker->ownsCurrentThread());EXPECT_THROW(a.bind(),std::invalid_argument);});
        EXPECT_EQ(CohortCase::capturesCreated.load(),births);
        std::barrier start(3);
        auto first=std::async(std::launch::async,[&]{start.arrive_and_wait();a.bind();});
        auto second=std::async(std::launch::async,[&]{start.arrive_and_wait();b.bind();});
        start.arrive_and_wait();first.get();second.get();
        EXPECT_EQ(CohortCase::capturesCreated.load(),births);
        a.admit(1);a.binding->launch();a.terminalJoin();b.admit(1);b.binding->launch();b.terminalJoin();
    }
}
TEST(PooledImmutableCompleteCohort, ExpiredWeakOwnerCannotReplaceOriginalRetirementInProgress) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,2,2}});
    CohortCase c({2});c.bind();const auto births=CohortCase::capturesCreated.load();
    std::promise<void> entered,release;auto permission=release.get_future().share();std::atomic<bool> hold=true;
    c.original->backends.rocm.beforeRetirementWait=[&]{if(hold.exchange(false)){entered.set_value();permission.wait();}};
    const std::weak_ptr<q::PooledImmutableCompleteCohort> weak=c.binding;
    auto original=std::move(c.binding);
    auto retiring=std::async(std::launch::async,[owner=std::move(original)]()mutable{owner.reset();});
    entered.get_future().wait();ASSERT_TRUE(weak.expired());
    const auto mock=std::dynamic_pointer_cast<testing::MockWorkerGPUContext>(c.remotes[0].participant.worker);
    mock->before_submit=[] {std::fputs("RETIRING_OWNER_ALLOWED_NEW_NATIVE_SUBMISSION\n",stderr);std::abort();};
    EXPECT_THROW(c.bind(),std::logic_error);EXPECT_EQ(CohortCase::capturesCreated.load(),births);
    mock->before_submit={};
    release.set_value();retiring.get();c.original->backends.rocm.beforeRetirementWait={};
    EXPECT_NO_THROW({c.renewDefinitions();c.bind();});EXPECT_EQ(CohortCase::capturesCreated.load(),births+2);
    c.admit(1);c.binding->launch();c.terminalJoin();
}

TEST(PooledImmutableCompleteCohort, FinalOwnerCannotBlockItsOriginalParticipatingWorker) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
    EXPECT_DEATH({CohortCase c({2});c.bind();auto worker=c.source.worker;worker->submitAndWait([&]{c.binding.reset();});},"terminate");
}

TEST(PooledImmutableCompleteCohort, QueuedOppositeSourceAdmissionsAndFailureNeverWaitForPeers) {
    CohortPrograms programs({{{2},q::adaptive_lookup::None,DeviceType::CUDA,0,1},{{2},q::adaptive_lookup::None,DeviceType::ROCm,0,1},{{2},q::adaptive_lookup::None,DeviceType::CUDA,1,1},{{2},q::adaptive_lookup::None,DeviceType::ROCm,1,1}});
    for(bool failure:{false,true}){
        CohortCase a({2},q::adaptive_lookup::None,DeviceType::CUDA),b({2},q::adaptive_lookup::None,DeviceType::ROCm);
        for(const auto& w:{a.source.worker,b.source.worker})std::dynamic_pointer_cast<testing::MockWorkerGPUContext>(w)->capture_destroyed=[] {++CohortCase::capturesDestroyed;};
        ASSERT_NE(std::dynamic_pointer_cast<QueuedCohortWorker>(a.source.worker),nullptr);
        ASSERT_NE(std::dynamic_pointer_cast<QueuedCohortWorker>(b.source.worker),nullptr);
        a.bind();b.bind();
        // Reject immutable remote worker submissions after cold bind. Admission
        // is source-only; opposite source jobs can therefore run concurrently.
        const auto* source=dynamic_cast<const SourceGuardCode*>(&a.original->boundary->pooledKernels());ASSERT_NE(source,nullptr);source->failAdmission=failure;
        std::barrier start(3);std::array<std::exception_ptr,2> errors;
        QueuedCohortWorker::forbidCrossWait.store(true);
        auto first=std::async(std::launch::async,[&]{try{a.source.worker->submitAndWait([&]{start.arrive_and_wait();a.original->boundary->beginAdmission(1,a.source.stream);});}catch(...){errors[0]=std::current_exception();}});
        auto second=std::async(std::launch::async,[&]{try{b.source.worker->submitAndWait([&]{start.arrive_and_wait();b.original->boundary->beginAdmission(1,b.source.stream);});}catch(...){errors[1]=std::current_exception();}});
        start.arrive_and_wait();first.get();second.get();QueuedCohortWorker::forbidCrossWait.store(false);EXPECT_EQ(bool(errors[0]),failure);EXPECT_FALSE(errors[1]);
        if(!failure){a.binding->launch();a.terminalJoin();}else{EXPECT_FALSE(a.original->boundary->hasActiveInterval());EXPECT_NO_THROW(a.original->service->stats());}
        b.binding->launch();b.terminalJoin();
        a.source.worker->submitAndWait([&]{
            EXPECT_THROW(a.binding->launch(),std::exception);
            EXPECT_THROW(a.binding->ownerKernelInventory(0),std::invalid_argument);
        });
    }
}

/** @brief Actual original native-role declarations seal against the pre-admission pure recipe. */
TEST(PooledImmutableCompleteCohort, DirectedOwnerProgramSealsActualRolesAndEdgesBeforeCapture) {
    EXPECT_EXIT(([]{
        {
            CohortPrograms programs({{{0,0},0,DeviceType::CUDA,0,0},{{0,3},0,DeviceType::CUDA,0,0},
                {{0,0},0,DeviceType::ROCm,0,0},{{0,3},0,DeviceType::ROCm,0,0}});
            for(const auto source:{DeviceType::CUDA,DeviceType::ROCm})for(const auto capacity:{0u,3u}){
                auto original=programs.make({0,capacity},0,source);auto& c=*original;
                const auto publication=q::PooledImmutableCapturedPublicationWave::bind(c.sources[1],c.memory);
                const auto read=q::PooledImmutableCapturedReadWave::bind(publication,c.memory);
                const auto retired=q::PooledImmutableCapturedRetirementWave::bind(read,c.memory);
                const q::PooledImmutableRemoteTransport remote{1,c.forwards[1],c.returns[1],c.frees[1],c.victims[1],c.sources[1],publication,read,retired};
                const auto program=q::PooledImmutableDirectedOwnerProgram::declare(c.physical,c.policy,1);
                ComputeGraph graph;BufferArena arena;
                const auto endpoints=appendPooledImmutableDirectedOwnerGraph(graph,arena,remote,c.ownerCode,program,"actual_owner");
                EXPECT_FALSE(endpoints.first.empty());EXPECT_FALSE(endpoints.last.empty());
                EXPECT_NO_THROW(q::PooledImmutableDirectedOwnerProgram::requireActual(graph,c.owners[1]->plan(),c.coordinator->plan(),program));
                EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::requireNativeActual(graph,c.owners[1]->plan(),c.coordinator->plan()),std::invalid_argument);
                const auto order=graph.getExecutionOrder();ASSERT_GT(order.size(),2u);
                const auto collectives=graph.collectiveNodeNames();ASSERT_FALSE(collectives.empty());
                for(const auto& name:collectives){const auto* stage=graph.getNode(name)->stage.get();EXPECT_TRUE(stage->isCollectiveStage());EXPECT_TRUE(stage->isParticipantLocalCapturedTransfer());}
                EXPECT_FALSE(graph.requiresCollectiveRuntimeCapture(&collectives));
                DeviceGraphExecutor::GraphSegmentCache nativePlan;
                EXPECT_NO_THROW(DeviceGraphCaptureController::buildCapturePlan(graph,nativePlan,&collectives,true,false));
                for(const auto& segment:nativePlan.segments)EXPECT_TRUE(segment.capturable);

                // The two native wire kernels can have identical counts while
                // carrying different ownership roles. Cardinality alone cannot seal them.
                auto* first=graph.getNode(order.front());auto* last=graph.getNode(order.back());
                std::swap(first->stage,last->stage);
                EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::requireActual(graph,c.owners[1]->plan(),c.coordinator->plan(),program),std::logic_error);
                std::swap(first->stage,last->stage);
                auto* second=graph.getNode(order[1]);const auto dependencies=second->dependencies;
                second->dependencies.push_back(order.back());
                EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::requireActual(graph,c.owners[1]->plan(),c.coordinator->plan(),program),std::logic_error);
                second->dependencies=dependencies;
                EXPECT_NO_THROW(q::PooledImmutableDirectedOwnerProgram::requireActual(graph,c.owners[1]->plan(),c.coordinator->plan(),program));
            }
        }
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @brief A losing distinct-boundary claim cannot retire the shared original winner's definition. */
TEST(PooledImmutableCompleteCohort, DistinctBoundariesRaceOneFreshOriginalRemoteOwner) {
    EXPECT_EXIT(([]{
        {
            CohortPrograms programs({{{3},q::adaptive_lookup::None,DeviceType::CUDA,0,1},{{3},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});
            CohortCase a({3}),b({3});b.remotes[0].participant.executable=a.remotes[0].participant.executable;
            std::array<CohortCase*,2> cases{&a,&b};
            std::array<std::shared_ptr<q::PooledImmutableCompleteCohort>,2> result;
            std::array<std::exception_ptr,2> failures;std::barrier start(3);
            std::array<std::future<void>,2> writers;
            for(std::size_t i=0;i<2;++i)writers[i]=std::async(std::launch::async,[&,i]{start.arrive_and_wait();try{result[i]=cases[i]->bindOriginal();}catch(...){failures[i]=std::current_exception();}});
            start.arrive_and_wait();for(auto& writer:writers)writer.get();
            EXPECT_EQ(bool(result[0])+bool(result[1]),1);EXPECT_EQ(bool(failures[0])+bool(failures[1]),1);
            const std::size_t index=result[0]?0:1;auto& winner=*cases[index];winner.binding=result[index];
            ASSERT_NE(winner.binding,nullptr);EXPECT_NO_THROW(winner.binding->requireCurrent());
            // Binding sealed the winner's original definition. Its mutable
            // recording facade must stay closed while requireCurrent/launch
            // below prove the losing claimant left that winner intact.
            EXPECT_THROW((void)RetainedParticipantGraphExecutableFixtureAccess::capture(winner.remotes[0].participant.executable),std::logic_error);
            winner.admit(1);EXPECT_NO_THROW(winner.binding->launch());winner.terminalJoin();
            EXPECT_NO_THROW(winner.binding->requireCurrent());winner.binding.reset();result[index].reset();
        }
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @brief Strong recording aliases cannot reopen, launch or reset a cohort-retired original definition. */
TEST(PooledImmutableCompleteCohort, RetainedFacadeRejectsMutationAfterCohortRetirement) {
    EXPECT_EXIT(([]{
        {
            CohortPrograms programs({{{3},q::adaptive_lookup::None,DeviceType::CUDA,0,1}});CohortCase c({3});
            const auto retained=c.source.executable;auto& recording=RetainedParticipantGraphExecutableFixtureAccess::capture(retained);c.bind();
            ASSERT_TRUE(recording.hasExecutable());c.binding.reset();
            c.source.worker->submitAndWait([&]{
                EXPECT_THROW(recording.beginCapture(),std::logic_error);
                EXPECT_THROW(recording.endCapture(),std::logic_error);
                EXPECT_THROW(recording.instantiate(),std::logic_error);
                EXPECT_THROW(recording.prepareRuntimeContextStorage(c.original->memory),std::logic_error);
                EXPECT_THROW(recording.reset(),std::logic_error);
                EXPECT_THROW(recording.launch(),std::logic_error);
                EXPECT_THROW(recording.launchOnStream(c.source.stream),std::logic_error);
            });
        }
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @brief SDK-free original LocalTP phase API; only native rendezvous/abort calls are doubled. */
class RecordingGroupContext final:public MockLocalTPContext {
public:
    /** @brief Wake every original actor before its failed future may be joined. */
    bool signalNativeCompletionAbort()override {
        {std::lock_guard lock(phaseMutex);aborted=true;}changed.notify_all();return true;
    }
    /** @brief Model the existing strict cyclic native capture phase boundary. */
    bool graphCaptureBoundaryRendezvous(const std::string& name,int member,int timeout)override {
        std::unique_lock lock(phaseMutex);
        if(aborted)return false;
        if(arrivals==0){phase=name;seen.assign(degree(),false);}
        if(phase!=name || member<0 || member>=degree() || seen[member])throw std::logic_error("capture replay phase/coordinate mismatch");
        seen[member]=true;const auto epoch=generation;
        if(++arrivals==degree()) {arrivals=0;++generation;changed.notify_all();return true;}
        if(!changed.wait_for(lock,std::chrono::milliseconds(timeout),[&]{return aborted || generation!=epoch;}))
            throw std::runtime_error("capture replay did not dispatch every original worker");
        return !aborted;
    }
    std::atomic<bool> aborted=false;
    unsigned generation=0;
private:
    std::mutex phaseMutex;std::condition_variable changed;std::string phase;
    std::vector<bool> seen;int arrivals=0;
};
/** @brief Exercise actual production capture grouping on original queued workers with no SDK. */
TEST(PooledImmutableCompleteCohort, NativeCaptureGroupMatchesAllPhasesAndRepeatedOriginalRosters) {
    constexpr std::size_t members=9;
    CohortPrograms programs({{std::vector<std::uint32_t>(members),q::adaptive_lookup::None,DeviceType::ROCm,0,1}});
    for(const std::size_t count:{2u,3u,9u})for(unsigned replay=0;replay<20;++replay) {
        RecordingGroupContext context;std::vector<GlobalDeviceAddress> devices;
        std::vector<std::shared_ptr<IWorkerGPUContext>> workers;
        std::vector<std::unique_ptr<IGPUGraphCapture>> captures;
        std::vector<OriginalNativeGraphRecording> jobs;
        std::atomic<unsigned> recorded=0;
        for(std::size_t i=0;i<count;++i)devices.push_back(GlobalDeviceAddress::cuda(static_cast<int>(i)));
        context.setDevices(devices);context.setBackend(CollectiveBackendType::NCCL);
        for(std::size_t i=0;i<count;++i) {
            const auto worker=programs.permit(DeviceId::cuda(static_cast<int>(i))).worker();workers.push_back(worker);
            worker->submitAndWait([&]{captures.push_back(worker->createGraphCapture(worker->defaultStream()));});
        }
        for(std::size_t i=0;i<count;++i)jobs.push_back({workers[i],DeviceId::cuda(static_cast<int>(i)),captures[i].get(),nullptr,[&] {
            for(const auto& worker:workers)EXPECT_TRUE(worker->isDeviceGraphCaptureActive());
            ++recorded;
        }});
        EXPECT_NO_THROW(OriginalNativeGraphCaptureGroup::record(jobs,context.borrowOriginalNativeContext(),"original native replay"));
        EXPECT_EQ(recorded,count);EXPECT_EQ(context.generation,3u);EXPECT_FALSE(context.aborted);
        for(const auto& worker:workers)EXPECT_FALSE(worker->isDeviceGraphCaptureActive());
        for(const auto& capture:captures)EXPECT_EQ(capture->nodeCount(),1u);
        auto missing=jobs;missing.pop_back();
        EXPECT_THROW(OriginalNativeGraphCaptureGroup::record(missing,context.borrowOriginalNativeContext(),"missing original"),std::invalid_argument);
        auto duplicate=jobs;duplicate.back()=duplicate.front();
        EXPECT_THROW(OriginalNativeGraphCaptureGroup::record(duplicate,context.borrowOriginalNativeContext(),"duplicate original"),std::invalid_argument);
    }
}
/** @brief Failure before a peer starts and failure during recording both abort before joining peers. */
TEST(PooledImmutableCompleteCohort, NativeCaptureGroupAbortsBeforePartialDispatchOrRecordingJoin) {
    CohortPrograms programs({{{0,0,0},q::adaptive_lookup::None,DeviceType::ROCm,0,1}});
    for(unsigned defect=0;defect<3;++defect) {
        RecordingGroupContext context;
        context.setDevices({GlobalDeviceAddress::cuda(0),GlobalDeviceAddress::cuda(1),GlobalDeviceAddress::cuda(2)});
        context.setBackend(CollectiveBackendType::NCCL);
        std::vector<std::shared_ptr<IWorkerGPUContext>> workers;
        std::vector<std::unique_ptr<IGPUGraphCapture>> captures;
        std::vector<OriginalNativeGraphRecording> jobs;
        for(int i=0;i<3;++i) {
            workers.push_back(programs.permit(DeviceId::cuda(i)).worker());
            workers.back()->submitAndWait([&]{captures.push_back(workers.back()->createGraphCapture(workers.back()->defaultStream()));});
        }
        for(int i=0;i<3;++i)jobs.push_back({workers[i],DeviceId::cuda(i),captures[i].get(),nullptr,[&,i] {
            if(defect==1 && i==1)throw std::runtime_error("original recording failure");
        }});
        if(defect!=1) {
            const auto queued=std::dynamic_pointer_cast<QueuedCohortWorker>(workers[1]);
            if(!queued)throw std::logic_error("Failure replay lost its original queued worker");
            if(defect==0)queued->rejectNextAsync=true;
            else queued->failNextAsyncBeforeEntry=true;
        }
        try {
            OriginalNativeGraphCaptureGroup::record(jobs,context.borrowOriginalNativeContext(),"failure replay");
            FAIL()<<"Original failure must propagate";
        }catch(const std::runtime_error& error) {
            EXPECT_STREQ(error.what(),defect==0?"original queued worker rejected submission":defect==1?"original recording failure":"original worker wrapper failed before callback");
        }
        EXPECT_TRUE(context.aborted);
        for(const auto& worker:workers)EXPECT_FALSE(worker->isDeviceGraphCaptureActive());
    }
}

}
