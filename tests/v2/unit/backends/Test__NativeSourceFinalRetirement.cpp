/** @file Test__NativeSourceFinalRetirement.cpp
 * @brief Actual source/pool/PMA and TransferEngine retirement of admitted failed and successful owners.
 * Only native backend results are doubled; source-owned mapped stack/object,
 * mixed storage endpoints, reset ordering and original physical lifetime execute.
 */
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "mocks/MockWorkerGPUContext.h"
#include "mocks/MockBackend.h"
#include "transfer/TransferEngine.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/ModelContextRetirement.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "tools/ModelKLDQFNOwner.h"
#include "integration/parity/qwen4exp/Qwen4ExpParityTestBase.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "backends/BackendManager.h"
#include "loaders/ModelContext.h"
#include "execution/local_execution/device/ReusableExecutionWorkspace.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionObservation.h"
#include "mocks/MockMPIContext.h"
#include <gtest/gtest.h>
namespace llaminar2 {
/** @brief Existing resolved-admission friend replays the exact admitted, not-inference-ready boundary without a model. */
class OrchestrationRunnerResolvedAdmissionTestAccess {
public:
 /** @brief Replay the complete post-init ownership shape with actual model/workspace/source objects. */
 static void exportedSource(OrchestrationRunner& runner,const std::shared_ptr<NativeContextPreparation>& source,
     const std::shared_ptr<ModelContext>& model,const std::shared_ptr<ReusableExecutionWorkspaceRegistry>& workspace,
     const std::shared_ptr<ModelContextReuseAuthority>& authority){
  runner.plan_.runtime.native_context_preparation=source;runner.physical_memory_authority_=source->memory();
  runner.model_ctx_=model;runner.reusable_execution_workspaces_=workspace;runner.model_context_reuse_authority_=authority;
  runner.initialization_lifecycle_.publish(OrchestrationInitializationLifecycle::Completion::Inference);
 }
 /** @brief Install original observed rank facts and an already-owned preparation without creating another source. */
 static void sourceObservation(OrchestrationRunner& runner,const std::shared_ptr<NativeContextPreparation>& source,
     ImmutableRowTablePolicy policy,int rank=0,std::span<const PhysicalMemoryResource> originalResources={}){
  runner.plan_.rank=rank;runner.plan_.runtime.ngram_table_policy=std::move(policy);
  runner.plan_.runtime.native_context_preparation=source;
  runner.cluster_inventory_.ranks.resize(static_cast<std::size_t>(rank+1));
  auto& observed=runner.cluster_inventory_.ranks[static_cast<std::size_t>(rank)];observed.rank=rank;
  const auto resource=[&](DeviceId device){
   if(originalResources.empty())return source->originalResource(device);
   for(const auto& value:originalResources)if(value.device==device)return value;
   throw std::logic_error("Original recorded observation omitted endpoint");
  };
  const auto cpu=resource(DeviceId::cpu());
  observed.cpu.memory_bytes=cpu.total_bytes;observed.cpu.free_memory_bytes=cpu.admission_available_bytes;
  observed.cpu_memory_bytes=cpu.total_bytes;
  for(const auto device:{DeviceId::cuda(231),DeviceId::rocm(231)}){
   const auto original=resource(device);DeviceInfo gpu;
   gpu.type=device.type;gpu.local_device_id=device.ordinal;gpu.memory_bytes=original.total_bytes;
   gpu.free_memory_bytes=original.admission_available_bytes;observed.gpus.push_back(std::move(gpu));
  }
 }
 /** @brief Substitute only the declared source, retaining its independently recorded original rank inventory. */
 static void replaceSource(OrchestrationRunner& runner,const std::shared_ptr<NativeContextPreparation>& source){
  runner.plan_.runtime.native_context_preparation=source;
 }
 /** @return Actual selected-context preparation; no factory/result override exists. */
 static bool prepareContexts(OrchestrationRunner& runner,const std::vector<DeviceId>& devices){
  return runner.prepareRankLocalGPUContexts(devices,"original-observation-replay");
 }
 /** @return Original retained source identity after the actual preparation. */
 static auto source(const OrchestrationRunner& runner){return runner.plan_.runtime.native_context_preparation;}
 /** @return Same untouched original device observation, never a newly sampled capacity. */
 static std::size_t observedFree(const OrchestrationRunner& runner,DeviceId device){
  const auto& gpus=runner.cluster_inventory_.ranks.at(static_cast<std::size_t>(runner.plan_.rank)).gpus;
  for(const auto& gpu:gpus)if(gpu.type==device.type&&gpu.local_device_id==device.ordinal)return gpu.free_memory_bytes;
  throw std::logic_error("Recorded original inventory omitted endpoint");
 }
 /** @brief Withdraw only a deliberately injected foreign authority after the negative invariant test. */
 static void discardForeignAuthority(OrchestrationRunner& runner){runner.model_context_reuse_authority_.reset();}
 static void failedSource(OrchestrationRunner& runner,const std::shared_ptr<NativeContextPreparation>& source,bool createdHere){
  runner.plan_.runtime.native_context_preparation=source;
  if(createdHere)runner.created_native_context_source_=source;
  runner.initialization_lifecycle_.publish(OrchestrationInitializationLifecycle::Completion::Admission);
 }
};
}
namespace llaminar2::test { namespace {
/** @brief Observations only; canonical byte accounting stays in original PMA. */
struct Trace {unsigned destroyed=0,resets=0,created=0,expectedDestroyedBeforeReset=2;bool allDestroyed=true;std::function<void()> beforeReset;};
Trace* active=nullptr;
/** @brief Actual mapped original worker storage and prepared thread, with no SDK call. */
class Worker final:public testing::MockWorkerGPUContext {
public:
 /** @brief Observe actual original worker birth before its prepared thread starts.
  * @param n Original device ordinal. @param root Original PMA-owned stack mapping. */
 Worker(int n,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(n){++active->created;thread_.start(std::move(root),[](void*){},nullptr);}
 ~Worker()override{thread_.join();++active->destroyed;}
private:PreparedGPUWorkerThread thread_;
};
/** @brief Native boundary double; actual TransferEngine authenticates the complete receipt. */
class Backend final:public MockBackend {
public:
 explicit Backend(DeviceType type):MockBackend(type){}
 std::size_t live=0;std::uint64_t generation=1;
 mutable unsigned freeQueries=0,totalQueries=0;
 DeviceAllocationAccounting deviceAllocationAccounting(int)const override{return {.supported=true,.active_allocations=live,.active_bytes=live*64};}
 std::size_t deviceMemoryFree(int)const override{++freeQueries;return 32u<<20;}
 std::size_t deviceMemoryTotal(int)const override{++totalQueries;return 64u<<20;}
 DeviceRuntimeGenerationRetirementResult retireExclusiveDeviceRuntimeGeneration(const DeviceRuntimeGenerationRetirementRequest&)override{
  if(active->beforeReset)active->beforeReset();
  active->allDestroyed &= active->destroyed==active->expectedDestroyedBeforeReset;++active->resets;
  DeviceRuntimeGenerationRetirementResult v;v.supported=v.success=v.reset_invoked=true;
  v.retired_generation=generation++;v.successor_generation=generation;v.post_reset_state=DeviceRuntimePostResetState::Quiescent;v.driver_free_bytes_before=32u<<20;return v;
 }
};
Backend *cuda=nullptr,*rocm=nullptr;
/** @return Only exact recorded native backends; no live SDK discovery. */
IBackend* resolve(DeviceId d){return d.is_cuda()?cuda:rocm;}
/** @return Actual original source admitted on its same PMA, with two declared native endpoints. */
std::shared_ptr<NativeContextPreparation> prepare(const std::array<DeviceId,2>& devices,NativePhysicalLifetime lifetime){
 using A=NativeContextPreparationReplayAccess;
 for(const auto d:devices)A::seed(d,[d](auto root){return A::construct<Worker>(std::move(root),d.ordinal);});
 const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),128u<<20,128u<<20},PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
 auto source=A::source<Worker>(devices,observed,lifetime);
 for(const auto d:devices){auto original=source->acquire(d);EXPECT_EQ(original.memory(),source->memory());}
 source->freezeForPlanning();source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM()));return source;
}
/** @return Original ledger commitments, never independently inferred capacity. */
std::size_t opaque(const std::shared_ptr<PhysicalMemoryAuthority>& m){std::size_t n=0;for(const auto& r:m->rankAttestation())n+=r.opaque_runtime_committed_bytes;return n;}
}
TEST(NativeSourceFinalRetirement, AdmittedMixedSourceRetiresStorageOnlyBirthAndRepeatsSafely){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(201),DeviceId::rocm(201)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);const auto m=source->memory();
 // This is the original process-teardown failure after an admitted, non-ready runner.
 EXPECT_THROW(GPUDeviceContextPool::instance().shutdown(),std::logic_error);
 EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
 NativeContextPreparationReplayAccess::retireReleased(source,transfers);
 EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);EXPECT_EQ(opaque(m),0u);
 for(const auto d:devices)EXPECT_THROW((void)GPUDeviceContextPool::instance().retainExistingPreparedContext(d,m),std::logic_error);
 NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
 EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}
TEST(NativeSourceFinalRetirement, LiveModelAllocationAndBorrowedWorkerRejectBeforeReset){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(202),DeviceId::rocm(202)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 c.live=1;EXPECT_THROW(NativeContextPreparationReplayAccess::retireReleased(source,transfers),std::runtime_error);
 EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);c.live=0;
 {auto borrowed=source->acquire(devices[0]);EXPECT_THROW(NativeContextPreparationReplayAccess::retireReleased(source,transfers),std::logic_error);
  EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);EXPECT_EQ(borrowed.memory(),source->memory());}
 NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
}
TEST(NativeSourceFinalRetirement, FailedAdmittedInitializationPreservesOriginalProcessObligation){
 GTEST_FLAG_SET(death_test_style,"threadsafe");
 ASSERT_EXIT(([]{
  Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
  const std::array devices{DeviceId::cuda(203),DeviceId::rocm(203)};auto source=prepare(devices,NativePhysicalLifetime::ProcessRuntime);const auto m=source->memory();const auto committed=opaque(m);EXPECT_GT(committed,0u);
  NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);EXPECT_EQ(t.destroyed,2u);EXPECT_TRUE(t.allDestroyed);EXPECT_EQ(opaque(m),committed);
  for(const auto d:devices)EXPECT_THROW((void)GPUDeviceContextPool::instance().retainExistingPreparedContext(d,m),std::logic_error);
  source.reset();GPUDeviceProcessLifetime::retireActive();EXPECT_EQ(opaque(m),committed);
  std::_Exit(::testing::Test::HasFailure()?2:0);
 }()),::testing::ExitedWithCode(0),"");
}
/** @test The actual runner shutdown must retire an admitted fresh failed initialization; direct source-only replay cannot replace this. */
TEST(NativeSourceFinalRetirement, ActualRunnerFailedInitializationRetiresItsOriginalMixedSource){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(204),DeviceId::rocm(204)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 {
  OrchestrationRunner runner(OrchestrationConfig{},nullptr);
  OrchestrationRunnerResolvedAdmissionTestAccess::failedSource(runner,source,true);
  EXPECT_NO_THROW(runner.shutdown());
  EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
  for(const auto d:devices)EXPECT_THROW((void)GPUDeviceContextPool::instance().retainExistingPreparedContext(d,source->memory()),std::logic_error);
  EXPECT_NO_THROW(runner.shutdown());EXPECT_EQ(t.resets,2u);
 }
}
/** @test A failed runner borrowing an existing admitted source cannot retire its external owner's workers. */
TEST(NativeSourceFinalRetirement, ActualRunnerBorrowedSourceRemainsWithItsOriginalOwner){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(205),DeviceId::rocm(205)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 {
  OrchestrationRunner runner(OrchestrationConfig{},nullptr);
  OrchestrationRunnerResolvedAdmissionTestAccess::failedSource(runner,source,false);
  EXPECT_NO_THROW(runner.shutdown());EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
  for(const auto d:devices){auto stillOwned=source->acquire(d);EXPECT_EQ(stillOwned.memory(),source->memory());}
 }
 EXPECT_EQ(t.resets,0u);NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
}

/** @brief Actual runner teardown from the recorded successful-init ownership shape. */
void exportedRunnerShutdown(bool observationFailure,bool successor)
{
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(207),DeviceId::rocm(207)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 auto model=ModelContext::createForTesting("recorded-exported-source.gguf",std::make_shared<MockMPIContext>(),1,true);
 model->bindNativeContextPreparation(source);
 auto workspace=std::make_shared<ReusableExecutionWorkspaceRegistry>();auto authority=std::make_shared<ModelContextReuseAuthority>();
 {
  auto runner=std::make_unique<OrchestrationRunner>(OrchestrationConfig{},nullptr);
  OrchestrationRunnerResolvedAdmissionTestAccess::exportedSource(*runner,source,model,workspace,authority);
  ASSERT_TRUE(runner->isInitialized());
  if(observationFailure){
   EXPECT_THROW(parity::qwen4exp::observeQwen4ExpAdmissionThenRetire(
    []{throw std::runtime_error("recorded post-initialization observation failure");},[&]{runner->shutdown();}),std::runtime_error);
  }else EXPECT_NO_THROW(runner->shutdown());
  EXPECT_FALSE(runner->isInitialized());EXPECT_EQ(authority->state(),ModelContextReuseAuthority::State::Reusable);
  EXPECT_EQ(runner->executionPlan().runtime.native_context_preparation,nullptr);
  EXPECT_EQ(model->nativeContextPreparation(),source);EXPECT_EQ(model->nativeContextPreparation()->memory(),source->memory());
  EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
  if(successor){std::string error;ASSERT_TRUE(authority->acquireRunnerExclusive(&error))<<error;}
  EXPECT_NO_THROW(runner->shutdown());runner.reset();
  EXPECT_EQ(authority->state(),successor?ModelContextReuseAuthority::State::RunnerExclusive:ModelContextReuseAuthority::State::Reusable);
  EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
 }
 model.reset();workspace.reset();NativeContextPreparationReplayAccess::retireReleased(source,transfers);
 EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
}
/** @test Final exported model retirement resets every original source participant, including storage-only workers. */
TEST(NativeSourceFinalRetirement,ExportedContractRetiresOriginalStorageOnlySourceBirths){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(209),DeviceId::rocm(209)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 ModelContextReuseContract contract;
 contract.context=ModelContext::createForTesting("recorded-final-source.gguf",std::make_shared<MockMPIContext>(),1,true);
 contract.context->bindNativeContextPreparation(source);
 contract.physical_memory_authority=source->memory();
 contract.reusable_execution_workspaces=std::make_shared<ReusableExecutionWorkspaceRegistry>();
 contract.reuse_authority=std::make_shared<ModelContextReuseAuthority>();std::string error;
 ASSERT_TRUE(contract.reuse_authority->beginSealing(&error));ASSERT_TRUE(contract.reuse_authority->publishReusable({},&error));
 // The original fixture's missing final-owner handoff must remain RED:
 // dropping reusable ownership is not the physical reset required by pool shutdown.
 try{GPUDeviceContextPool::instance().shutdown();FAIL()<<"Unretired exported source accepted process shutdown";}
 catch(const std::logic_error& error){EXPECT_EQ(std::string(error.what()),"Source-prepared native contexts require original physical reset before shutdown");}
 EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
 // No prepared-weight rows exist in this recorded final model. Both original
 // workers therefore expose the residual-birth path, not a fabricated weight BOM.
 // Same address with a foreign control block cannot authorize source cleanup.
 const auto originalMemory=contract.physical_memory_authority;
 contract.physical_memory_authority=std::shared_ptr<PhysicalMemoryAuthority>(originalMemory.get(),[](auto*){});
 EXPECT_THROW((void)retireExclusiveModelContextReuseContract(contract),std::logic_error);
 EXPECT_TRUE(contract.context);EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
 contract.physical_memory_authority=originalMemory;
 EXPECT_NO_THROW((void)retireExclusiveModelContextReuseContract(contract));
 EXPECT_THROW((void)retireExclusiveModelContextReuseContract(contract),std::invalid_argument);
 EXPECT_FALSE(contract.context);EXPECT_FALSE(contract.physical_memory_authority);
 EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
 for(const auto d:devices)EXPECT_THROW((void)GPUDeviceContextPool::instance().retainExistingPreparedContext(d,source->memory()),std::logic_error);
 EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}

/** @brief Only export readiness is recorded; observer, actual runner shutdown and final source reset execute production bodies. */
class RecordedObserverRunner final:public OrchestrationRunner {
public:
 RecordedObserverRunner():OrchestrationRunner(OrchestrationConfig{},nullptr){}
 ModelContextReuseContract exported;
 unsigned* exports=nullptr;
 bool failShutdown=false;
 bool failWorkerLoop=false;
 std::vector<std::string>* commands=nullptr;
 void setMPICoordinatedMode(bool value)override{if(commands)commands->push_back(value?"coordinated":"uncoordinated");}
 void runMPIWorkerLoop()override{if(commands)commands->push_back("worker_loop");if(failWorkerLoop)throw std::runtime_error("original follower loop failure");}
 void shutdownMPIWorkers()override{if(commands)commands->push_back("stop_workers");}
 void abortMPIWorkers(const std::string&)override{if(commands)commands->push_back("abort_workers");}
 void shutdown()override {if(commands)commands->push_back("retire");if(failShutdown)throw std::runtime_error("recorded terminal failure");OrchestrationRunner::shutdown();}
 std::optional<ModelContextReuseContract> modelContextReuseContract()const override {++*exports;return exported;}
};
/** @brief Exercise the actual QFN fixture observation and teardown edges without running SetUp/model work. */
class RecordedQFNFixture final : public parity::qwen4exp::Qwen4ExpConfigDrivenParityTest<RecordedQFNFixture> {
public:
 RecordedQFNFixture(){
  cell_=parity::expandModelParityDefinition(parity::qwen4exp::qwen4ExpProductionDefinition(
      parity::qwen4exp::Qwen4ExpParityTopology::CPU)).front();view_=cell_.toTestConfig();
 }
 const parity::TestConfig& getTestConfig()const{return view_;}
 const parity::ModelParityCase& modelParityCase()const{return cell_;}
 /** @brief Transfer the same original recorded runner into its real fixture owner. */
 void attach(std::unique_ptr<IOrchestrationRunner> runner){orch_runner_=std::move(runner);}
 /** @return The actual fixture's original, once-exported observation contract. */
 const ModelContextReuseContract& observe(){return qfnObservationOwner();}
 /** @brief Invoke the same virtual retirement edge called by parity TearDown. */
 void finish(){retireOwnedParityRunners();}
 /** @brief Keep an old-body negative control physically clean after its assertions fail. */
 void finishRemainingTestOwner(){accuracy::retireQFNFinalOwner(orch_runner_,qfn_observation_owner_);}
private:
 void TestBody()override{}
 parity::ModelParityCase cell_;parity::TestConfig view_;
};
/** @brief Replay actual QFN fixture export retention through success or observer failure. */
void replayQFNFixtureFinalOwner(bool failObservation){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(210),DeviceId::rocm(210)};
 auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 auto actual=std::make_unique<RecordedObserverRunner>();unsigned exports=0;actual->exports=&exports;
 auto model=ModelContext::createForTesting("recorded-qfn-fixture-source.gguf",std::make_shared<MockMPIContext>(),1,true);
 model->bindNativeContextPreparation(source);
 auto workspace=std::make_shared<ReusableExecutionWorkspaceRegistry>();auto authority=std::make_shared<ModelContextReuseAuthority>();
 OrchestrationRunnerResolvedAdmissionTestAccess::exportedSource(*actual,source,model,workspace,authority);
 actual->exported.context=model;actual->exported.reusable_execution_workspaces=workspace;
 actual->exported.reuse_authority=authority;actual->exported.physical_memory_authority=source->memory();
 model.reset();workspace.reset();authority.reset();
 RecordedQFNFixture fixture;fixture.attach(std::move(actual));
 try {
  const auto& first=fixture.observe();const auto& second=fixture.observe();
  EXPECT_EQ(&first,&second);EXPECT_EQ(first.context->nativeContextPreparation(),source);
  EXPECT_EQ(first.physical_memory_authority,source->memory());EXPECT_EQ(exports,1u);EXPECT_EQ(t.resets,0u);
  if(failObservation)throw std::runtime_error("original fixture observation failure");
 }catch(const std::runtime_error& error){EXPECT_TRUE(failObservation);EXPECT_STREQ(error.what(),"original fixture observation failure");}
 EXPECT_NO_THROW(fixture.finish());
 EXPECT_EQ(exports,1u);EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
 // If the old fixture only dropped its runner, this same genuine final export
 // remains available to clean the test's own source after the failed oracle.
 EXPECT_NO_THROW(fixture.finishRemainingTestOwner());
 EXPECT_EQ(t.resets,2u);EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}
/** @test The actual parity fixture retains one observation export until original model/source retirement. */
TEST(NativeSourceFinalRetirement,QFNParityFixtureSuccessRetiresOriginalObservationOwner){replayQFNFixtureFinalOwner(false);}
/** @test A failed observation still crosses actual fixture runner and same-source final retirement. */
TEST(NativeSourceFinalRetirement,QFNParityFixtureFailureRetiresOriginalObservationOwner){replayQFNFixtureFinalOwner(true);}

/** @brief Exact follower failure boundary, with no claimed native worker success. */
enum class FollowerObservationFailure { None, WorkerLoop, PassiveRecord };
/** @brief Exercise actual frontend ownership through observation and original source retirement. */
void replayObserverFinalOwner(bool failBeforeHistory,bool cleanupFailure=false,int coordinatedRole=0,
 FollowerObservationFailure followerFailure=FollowerObservationFailure::None){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(210),DeviceId::rocm(210)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 auto actual=std::make_unique<RecordedObserverRunner>();auto* recorded=actual.get();std::vector<std::string> commands;if(coordinatedRole)actual->commands=&commands;unsigned exports=0;actual->exports=&exports;unsigned expectedExports=1;actual->failWorkerLoop=followerFailure==FollowerObservationFailure::WorkerLoop;
 auto model=ModelContext::createForTesting("recorded-observer-source.gguf",std::make_shared<MockMPIContext>(),1,true);model->bindNativeContextPreparation(source);
 auto workspace=std::make_shared<ReusableExecutionWorkspaceRegistry>();auto authority=std::make_shared<ModelContextReuseAuthority>();
 OrchestrationRunnerResolvedAdmissionTestAccess::exportedSource(*actual,source,model,workspace,authority);
 actual->exported.context=model;actual->exported.reusable_execution_workspaces=workspace;
 actual->exported.reuse_authority=authority;actual->exported.physical_memory_authority=source->memory();
 model.reset();workspace.reset();authority.reset();
 std::unique_ptr<IOrchestrationRunner> runner=std::move(actual);unsigned histories=0;
 const auto observe=[&](const ModelContextReuseContract& owner){
  EXPECT_EQ(owner.context->nativeContextPreparation(),source);EXPECT_EQ(owner.physical_memory_authority,source->memory());
  EXPECT_EQ(t.resets,0u);EXPECT_EQ(exports,expectedExports);
  if(coordinatedRole)commands.push_back("observe");
  if(failBeforeHistory)throw std::runtime_error("original failure before first history");
  // The callback borrows this same model through each history/reset observation;
  // it cannot retire or replace the runner's model between observations.
  const auto* original=owner.context.get();
  for(unsigned i=0;i<3;++i){EXPECT_EQ(owner.context.get(),original);++histories;}
 };
 const auto invoke=[&] {
  if(coordinatedRole)accuracy::observeQFNCoordinatedWithFinalOwner(runner,coordinatedRole==1,observe,
    [&](const ModelContextReuseContract& owner){EXPECT_EQ(owner.context->nativeContextPreparation(),source);commands.push_back("follower_record");if(followerFailure==FollowerObservationFailure::PassiveRecord)throw std::runtime_error("original follower record failure");});
  else accuracy::observeQFNWithFinalOwner(runner,observe);
 };
 if(cleanupFailure){
  recorded->failShutdown=true;
  ::testing::internal::CaptureStderr();
  try{invoke();FAIL()<<"Missing terminal failure";}
  catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"original failure before first history; QFN final retirement failed: recorded terminal failure");}
  const auto diagnostic=::testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("QFN observation failed before final retirement: original failure before first history"),std::string::npos);
  EXPECT_TRUE(runner);EXPECT_EQ(t.resets,0u);recorded->failShutdown=false;expectedExports=2;
 }
 if(followerFailure!=FollowerObservationFailure::None) {
  try{invoke();FAIL()<<"Missing follower lifecycle failure";}
  catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),followerFailure==FollowerObservationFailure::WorkerLoop?
   "original follower loop failure":"original follower record failure");}
 }
 else if(failBeforeHistory){try{invoke();FAIL()<<"Missing observer failure";}
  catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"original failure before first history");}}
 else EXPECT_NO_THROW(invoke());
 EXPECT_EQ(histories,(failBeforeHistory || coordinatedRole==2)?0u:3u);
 if(coordinatedRole==1)EXPECT_EQ(commands,(failBeforeHistory?std::vector<std::string>{"coordinated","observe","abort_workers","retire"}:std::vector<std::string>{"coordinated","observe","stop_workers","retire"}));
 if(coordinatedRole==2)EXPECT_EQ(commands,(followerFailure==FollowerObservationFailure::WorkerLoop?
  std::vector<std::string>{"coordinated","worker_loop","abort_workers","retire"}:
  std::vector<std::string>{"coordinated","worker_loop","follower_record","retire"}));EXPECT_EQ(exports,expectedExports);EXPECT_FALSE(runner);
 EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
 EXPECT_THROW(accuracy::observeQFNWithFinalOwner(runner,observe),std::invalid_argument);EXPECT_EQ(t.resets,2u);
 EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}
/** @test A retained-capacity predictor remains materialized even while execution is disabled. */
TEST(NativeSourceFinalRetirement,QFNObserverOffCapacitySelectsOriginalPredictorMaterialization){
 MTPRuntimeConfig off;off.enabled=false;off.draft_tokens=0;off.graph_capacity_draft_tokens=0;
 EXPECT_EQ(accuracy::qfnPredictorMaterialization(off),SharedPredictorMaterialization::Inactive);
 off.graph_capacity_draft_tokens=15;
 EXPECT_EQ(accuracy::qfnPredictorMaterialization(off),SharedPredictorMaterialization::Active);
}
/** @test Successful repeated-history observation retains one export and resets both original native participants. */
TEST(NativeSourceFinalRetirement,QFNObserverSuccessRetiresSoleOriginalOwner){replayObserverFinalOwner(false);}
/** @test Pre-history failure preserves its primary exception and still retires the same complete original source. */
TEST(NativeSourceFinalRetirement,QFNObserverBeforeHistoryFailureRetiresSoleOriginalOwner){replayObserverFinalOwner(true);}
/** @test Cleanup failure cannot overwrite the primary observation diagnostic or claim successful retirement. */
TEST(NativeSourceFinalRetirement,QFNObserverPrimaryAndTerminalFailuresRemainExplicit){replayObserverFinalOwner(true,true);}

/** @test Explicit completed shutdown followed by destruction cannot re-enter reuse sealing. */
TEST(NativeSourceFinalRetirement,ActualRunnerExportedSourceShutdownThenDestructorIsIdempotent){exportedRunnerShutdown(false,false);}
/** @test The real source observer's post-init failure retires once, preserving external model ownership. */
TEST(NativeSourceFinalRetirement,ActualRunnerPostInitializationFailureThenDestructorRetiresOnce){exportedRunnerShutdown(true,false);}
/** @test A retired exporter must not seal the SAME authority after a successor acquired it. */
TEST(NativeSourceFinalRetirement,RetiredRunnerCannotSealSuccessorActiveReuseAuthority){exportedRunnerShutdown(false,true);}
/** @test A genuinely live owner with a foreign reusable authority still fails the existing invariant. */
TEST(NativeSourceFinalRetirement,LiveRunnerForeignReusableAuthorityStillRejectsShutdown){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(208),DeviceId::rocm(208)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 auto model=ModelContext::createForTesting("recorded-foreign-source.gguf",std::make_shared<MockMPIContext>(),1,true);model->bindNativeContextPreparation(source);
 auto workspace=std::make_shared<ReusableExecutionWorkspaceRegistry>();auto foreign=std::make_shared<ModelContextReuseAuthority>();std::string error;
 ASSERT_TRUE(foreign->beginSealing(&error));ASSERT_TRUE(foreign->publishReusable({},&error));
 {
  OrchestrationRunner runner(OrchestrationConfig{},nullptr);OrchestrationRunnerResolvedAdmissionTestAccess::exportedSource(runner,source,model,workspace,foreign);
  // The native retirement boundary is noexcept: this genuine ownership
  // violation must terminate the child, leaving the parent's owners intact.
  EXPECT_DEATH(runner.shutdown(),"Prepared context sealing does not own its reuse authority");
  EXPECT_EQ(foreign->state(),ModelContextReuseAuthority::State::Reusable);
  EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);
  // The invalid reference was injected only by this closed fixture. Remove it
  // for cleanup; production keeps rejecting that genuine ownership violation.
  OrchestrationRunnerResolvedAdmissionTestAccess::discardForeignAuthority(runner);EXPECT_NO_THROW(runner.shutdown());
 }
 model.reset();workspace.reset();NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
}

/** @test Every GPU storage policy reuses the same original prepared source without late free-memory sampling. */
TEST(NativeSourceFinalRetirement, EveryGPUStoragePolicyPreservesOriginalPrequotaSource){
 using Access=OrchestrationRunnerResolvedAdmissionTestAccess;
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(231),DeviceId::rocm(231)};
 auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 for(const auto storage:{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped,
     ImmutableRowStoragePreference::VRAM,ImmutableRowStoragePreference::Automatic,
     ImmutableRowStoragePreference::HostAdaptiveRAMSSD,ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD}){
  OrchestrationRunner runner(OrchestrationConfig{},nullptr);
  Access::sourceObservation(runner,source,ImmutableRowTablePolicy{.storage=storage});
  const auto freeQueries=c.freeQueries+r.freeQueries,totalQueries=c.totalQueries+r.totalQueries;
  auto before=source->acquire(devices[0]);
  EXPECT_TRUE(Access::prepareContexts(runner,{devices[0],DeviceId::cpu(),devices[0]}))<<runner.lastError();
  EXPECT_EQ(Access::source(runner),source);
  EXPECT_FALSE(Access::source(runner).owner_before(source));EXPECT_FALSE(source.owner_before(Access::source(runner)));
  EXPECT_EQ(c.freeQueries+r.freeQueries,freeQueries);EXPECT_EQ(c.totalQueries+r.totalQueries,totalQueries);
  EXPECT_EQ(Access::observedFree(runner,devices[0]),source->originalResource(devices[0]).admission_available_bytes);
  auto after=GPUDeviceContextPool::instance().retainExistingPreparedContext(devices[0],source->memory());
  EXPECT_EQ(before.generation(),after.generation());EXPECT_EQ(before.worker(),after.worker());
  EXPECT_FALSE(before.worker().owner_before(after.worker()));EXPECT_FALSE(after.worker().owner_before(before.worker()));
  EXPECT_EQ(after.memory(),source->memory());
 }
 NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
}
/** @test Authored storage membership/rank and foreign original PMA fail before adopting any worker. */
TEST(NativeSourceFinalRetirement, SelectedSourceMembershipAndRankRemainExactAcrossStoragePolicies){
 using Access=OrchestrationRunnerResolvedAdmissionTestAccess;
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;TransferEngine transfers(resolve);
 const std::array devices{DeviceId::cuda(231),DeviceId::rocm(231)};
 auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
 policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:rocm:231/0/0@0:cuda:231");
 {OrchestrationRunner runner(OrchestrationConfig{},nullptr);Access::sourceObservation(runner,source,policy);
  EXPECT_TRUE(Access::prepareContexts(runner,{devices[0]}))<<runner.lastError();}
 {OrchestrationRunner runner(OrchestrationConfig{},nullptr);Access::sourceObservation(runner,source,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM});
  EXPECT_FALSE(Access::prepareContexts(runner,{DeviceId::cuda(232)}));
  EXPECT_NE(runner.lastError().find("did not declare"),std::string::npos);}
 {OrchestrationRunner runner(OrchestrationConfig{},nullptr);Access::sourceObservation(runner,source,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM},1);
  EXPECT_FALSE(Access::prepareContexts(runner,{devices[0]}));
  EXPECT_NE(runner.lastError().find("crossed original physical ranks"),std::string::npos);}
 {auto foreign=NativeContextPreparationReplayAccess::source<Worker>(devices,std::array{
    source->originalResource(DeviceId::cpu()),source->originalResource(devices[0]),source->originalResource(devices[1])});
  OrchestrationRunner runner(OrchestrationConfig{},nullptr);Access::sourceObservation(runner,source,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM});
  Access::replaceSource(runner,foreign);
  EXPECT_FALSE(Access::prepareContexts(runner,{devices[0]}));
  EXPECT_NE(runner.lastError().find("foreign PMA control block"),std::string::npos);}
 NativeContextPreparationReplayAccess::retireReleased(source,transfers);EXPECT_EQ(t.resets,2u);
}

/** @test A source may not adopt/reset an ordinary legacy worker even on the same physical endpoint. */
TEST(NativeSourceFinalRetirement, SelectedSourceCannotAdoptLegacyOrdinaryWorker){
 GTEST_FLAG_SET(death_test_style,"threadsafe");
 ASSERT_EXIT(([]{
  using Access=OrchestrationRunnerResolvedAdmissionTestAccess;
  Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
  auto& pool=GPUDeviceContextPool::instance();
  pool.registerNvidiaFactory([](int ordinal){return std::make_unique<testing::MockWorkerGPUContext>(ordinal);},232);
  auto legacy=pool.retainNvidiaContext(231);
  const std::array devices{DeviceId::cuda(231),DeviceId::rocm(231)};
  const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),128u<<20,128u<<20},
      PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
  auto source=NativeContextPreparationReplayAccess::source<Worker>(devices,observed);
  {OrchestrationRunner runner(OrchestrationConfig{},nullptr);
   Access::sourceObservation(runner,source,ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM},0,observed);
   EXPECT_FALSE(Access::prepareContexts(runner,{devices[0]}));
   EXPECT_EQ(pool.retainNvidiaContext(231),legacy);
   EXPECT_EQ(t.resets,0u);
  }
  source.reset();legacy.reset();EXPECT_NO_THROW(pool.shutdown());
  std::_Exit(::testing::Test::HasFailure()?2:0);
 }()),::testing::ExitedWithCode(0),"");
}

/** @test CPU-only/empty preparation needs no GPU rank inventory, while a selected GPU still does. */
TEST(NativeSourceFinalRetirement, CPUOnlyPreparationDoesNotRequireGPUInventory){
 using Access=OrchestrationRunnerResolvedAdmissionTestAccess;
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 OrchestrationRunner runner(OrchestrationConfig{},nullptr);
 EXPECT_TRUE(Access::prepareContexts(runner,{}));
 EXPECT_TRUE(Access::prepareContexts(runner,{DeviceId::cpu(),DeviceId::cpu()}));
 EXPECT_EQ(Access::source(runner),nullptr);
 EXPECT_FALSE(Access::prepareContexts(runner,{DeviceId::cuda(231)}));
 EXPECT_EQ(Access::source(runner),nullptr);
 EXPECT_EQ(c.freeQueries+r.freeQueries,0u);EXPECT_EQ(c.totalQueries+r.totalQueries,0u);
}

}

// This isolated model-free target supplies only the native backend boundary;
// original runner/source/pool/TransferEngine implementations remain unchanged.
namespace llaminar2 {
IBackend* getBackendFor(DeviceId device){return device.is_cpu()?getCPUBackend():test::resolve(device);}
}

namespace llaminar2::test {
/** @test Only original authority observes histories, then stops workers before physical retirement. */
TEST(NativeSourceFinalRetirement,QFNDistributedAuthorityStopsBeforeOriginalRetirement){replayObserverFinalOwner(false,false,1);}
/** @test Original failure aborts worker command flow before final physical teardown. */
TEST(NativeSourceFinalRetirement,QFNDistributedAuthorityFailureAbortsBeforeOriginalRetirement){replayObserverFinalOwner(true,false,1);}
/** @test Follower records actual worker-loop completion without independent history/logit observation. */
TEST(NativeSourceFinalRetirement,QFNDistributedFollowerOnlyRunsOriginalWorkerLoop){replayObserverFinalOwner(false,false,2);}

}

namespace llaminar2::test {
/** @test Failure after the original follower loop returns retires locally without issuing a root command. */
TEST(NativeSourceFinalRetirement,QFNDistributedFollowerRecordFailureRetiresWithoutRootAbort) {
 replayObserverFinalOwner(false,false,2,FollowerObservationFailure::PassiveRecord);
}
/** @test A failure inside the original follower command loop still cancels before original retirement. */
TEST(NativeSourceFinalRetirement,QFNDistributedFollowerLoopFailureAbortsBeforeRetirement) {
 replayObserverFinalOwner(false,false,2,FollowerObservationFailure::WorkerLoop);
}
/** @test Missing original export fails before a worker loop and still invokes its existing cancellation owner. */
TEST(NativeSourceFinalRetirement,QFNDistributedFollowerMissingOwnerAbortsBeforeLoop) {
 auto actual=std::make_unique<RecordedObserverRunner>();std::vector<std::string> commands;unsigned exports=0;
 actual->commands=&commands;actual->exports=&exports;
 // No model/prepared/PMA owner exists here: this is the negative export boundary,
 // not an inference-ready or native-success fixture.
 std::unique_ptr<IOrchestrationRunner> runner=std::move(actual);
 EXPECT_THROW(accuracy::observeQFNCoordinatedWithFinalOwner(runner,false,
  [](const ModelContextReuseContract&){ADD_FAILURE()<<"Follower ran authority observation";},
  [](const ModelContextReuseContract&){ADD_FAILURE()<<"Missing owner reached follower record";}),std::invalid_argument);
 EXPECT_EQ(commands,(std::vector<std::string>{"coordinated","abort_workers","retire"}));
 EXPECT_EQ(exports,1u);EXPECT_FALSE(runner);
}
}

namespace llaminar2::test {
/** @test Terminal failed reuse seals discard original owners without forging Reusable or losing storage-only resets. */
TEST(NativeSourceFinalRetirement, InvalidFinalContractDiscardsOnlyAfterOriginalOwnersRetire){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(241),DeviceId::rocm(241)};auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 ModelContextReuseContract contract;
 contract.context=ModelContext::createForTesting("recorded-invalid-final-source.gguf",std::make_shared<MockMPIContext>(),1,true);
 contract.context->bindNativeContextPreparation(source);contract.physical_memory_authority=source->memory();
 contract.reusable_execution_workspaces=std::make_shared<ReusableExecutionWorkspaceRegistry>();
 contract.reuse_authority=std::make_shared<ModelContextReuseAuthority>();
 EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::logic_error);
 ASSERT_TRUE(contract.reuse_authority->beginSealing());
 EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::logic_error);
 ASSERT_TRUE(contract.reuse_authority->publishReusable({}));
 EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::logic_error);
 contract.reuse_authority->invalidate("recorded partial retention seal failure");
 // Original failed-retirement control stays strict: Invalid is never Reusable.
 EXPECT_THROW((void)retireExclusiveModelContextReuseContract(contract),std::logic_error);
 const auto rejectBorrowed=[&]{
  try{(void)discardExclusiveInvalidModelContextReuseContract(contract);FAIL()<<"Borrowed invalid owner was discarded";}
  catch(const std::logic_error& error){EXPECT_NE(std::string(error.what()).find("recorded partial retention seal failure"),std::string::npos);}
  EXPECT_TRUE(contract.context);EXPECT_EQ(t.resets,0u);
 };
 {auto borrowed=contract.context;rejectBorrowed();}
 {auto borrowed=contract.reusable_execution_workspaces;rejectBorrowed();}
 {auto borrowed=contract.reuse_authority;rejectBorrowed();EXPECT_EQ(borrowed->state(),ModelContextReuseAuthority::State::Invalid);}
 const auto originalMemory=contract.physical_memory_authority;
 contract.physical_memory_authority=std::shared_ptr<PhysicalMemoryAuthority>(originalMemory.get(),[](auto*){});
 EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::logic_error);
 contract.physical_memory_authority=originalMemory;
 {
  auto originalModel=std::move(contract.context);
  contract.context=ModelContext::createForTesting("recorded-absent-original-source.gguf",std::make_shared<MockMPIContext>(),1,true);
  EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::logic_error);
  EXPECT_TRUE(contract.context);EXPECT_EQ(t.resets,0u);
  contract.context=std::move(originalModel);
 }
 std::weak_ptr<ModelContext> modelLifetime=contract.context;
 std::weak_ptr<ReusableExecutionWorkspaceRegistry> workspaceLifetime=contract.reusable_execution_workspaces;
 t.beforeReset=[&]{EXPECT_TRUE(modelLifetime.expired());EXPECT_TRUE(workspaceLifetime.expired());};
 const auto receipt=discardExclusiveInvalidModelContextReuseContract(contract);
 EXPECT_EQ(receipt.retiredDeviceCount(),2u);EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);EXPECT_TRUE(t.allDestroyed);
 EXPECT_FALSE(contract.context);EXPECT_FALSE(contract.reuse_authority);EXPECT_FALSE(contract.reusable_execution_workspaces);
 EXPECT_THROW((void)discardExclusiveInvalidModelContextReuseContract(contract),std::invalid_argument);
 EXPECT_EQ(source->retireReleasedContexts(),0u);EXPECT_EQ(t.resets,2u);
 EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}
/** @test A terminal CPU-only failure releases owners without entering any native reset path. */
TEST(NativeSourceFinalRetirement, InvalidCpuOnlyContractDiscardsWithoutNativeSource){
 ModelContextReuseContract contract;contract.context=ModelContext::createForTesting("recorded-invalid-cpu-final.gguf");
 contract.reusable_execution_workspaces=std::make_shared<ReusableExecutionWorkspaceRegistry>();
 contract.reuse_authority=std::make_shared<ModelContextReuseAuthority>();contract.reuse_authority->invalidate("recorded CPU seal failure");
 PhysicalMemoryPlanBuilder builder;builder.add(PhysicalMemoryResource{0,DeviceId::cpu(),1024,1024},PhysicalMemoryOwner::ModelSourcePayload,1);
 contract.physical_memory_authority=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
 EXPECT_EQ(discardExclusiveInvalidModelContextReuseContract(contract).retiredDeviceCount(),0u);
 EXPECT_FALSE(contract.context);EXPECT_FALSE(contract.reuse_authority);
}
}

namespace llaminar2::test {
/** @test Admission handoff borrows the same actual source and rejects foreign authority before publishing metadata. */
TEST(NativeSourceFinalRetirement, RetainedAdmissionKeepsOriginalNativeSource){
 Trace t;active=&t;Backend c(DeviceType::CUDA),r(DeviceType::ROCm);cuda=&c;rocm=&r;
 const std::array devices{DeviceId::cuda(242),DeviceId::rocm(242)};
 const std::array foreignDevices{DeviceId::cuda(243),DeviceId::rocm(243)};
 auto source=prepare(devices,NativePhysicalLifetime::ContextGeneration);
 auto foreign=prepare(foreignDevices,NativePhysicalLifetime::ContextGeneration);
 ASSERT_EQ(t.created,4u);
 RuntimeConfig original;original.native_context_preparation=source;
 original.resident_graph_rows=512;original.moe_routed_prefill.overlay_segment_rows=64;
 const auto originalMemory=source->memory();const auto committed=opaque(originalMemory);
 std::array<const void*,2> workerIdentity{};
 for(size_t i=0;i<devices.size();++i)workerIdentity[i]=source->acquire(devices[i]).worker().get();
 for(int request=0;request<20;++request){
  RuntimeConfig current;current.resident_graph_rows=17;current.moe_routed_prefill.overlay_segment_rows=9;
  const auto rejectWithoutMutation=[&](int rank){
   const auto before=current.native_context_preparation;
   if(rank<0)EXPECT_THROW(restoreRetainedAdmittedRuntime(current,original,rank),std::invalid_argument);
   else EXPECT_THROW(restoreRetainedAdmittedRuntime(current,original,rank),std::logic_error);
   EXPECT_EQ(current.native_context_preparation,before);
   if(before){EXPECT_FALSE(current.native_context_preparation.owner_before(before));EXPECT_FALSE(before.owner_before(current.native_context_preparation));}
   EXPECT_EQ(current.resident_graph_rows,17);EXPECT_EQ(current.moe_routed_prefill.overlay_segment_rows,9);
   EXPECT_FALSE(current.ngram_table_placement);EXPECT_FALSE(current.attention_participants);
   EXPECT_FALSE(current.retained_host_services);EXPECT_TRUE(current.cpu_serial_expert_storage.empty());
   EXPECT_EQ(t.created,4u);EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);EXPECT_EQ(opaque(originalMemory),committed);
  };
  rejectWithoutMutation(-1);rejectWithoutMutation(1);
  current.native_context_preparation=foreign;rejectWithoutMutation(0);
  // Same address with independent shared ownership is not the original source.
  current.native_context_preparation=std::shared_ptr<NativeContextPreparation>(source.get(),[](auto*){});
  rejectWithoutMutation(0);
  current.native_context_preparation=source;
  RuntimeConfig absent;EXPECT_THROW(restoreRetainedAdmittedRuntime(current,absent,0),std::logic_error);
  EXPECT_EQ(current.native_context_preparation,source);EXPECT_EQ(current.resident_graph_rows,17);
  current.native_context_preparation.reset();
  restoreRetainedAdmittedRuntime(current,original,0);
  EXPECT_EQ(current.native_context_preparation,source);
  EXPECT_FALSE(current.native_context_preparation.owner_before(source));EXPECT_FALSE(source.owner_before(current.native_context_preparation));
  EXPECT_EQ(current.native_context_preparation->memory(),originalMemory);
  EXPECT_EQ(current.resident_graph_rows,512);EXPECT_EQ(current.moe_routed_prefill.overlay_segment_rows,64);
  restoreRetainedAdmittedRuntime(current,original,0);
  for(size_t i=0;i<devices.size();++i){const auto retained=source->acquire(devices[i]);EXPECT_EQ(retained.worker().get(),workerIdentity[i]);EXPECT_EQ(retained.memory(),originalMemory);}
  EXPECT_EQ(t.created,4u);EXPECT_EQ(t.destroyed,0u);EXPECT_EQ(t.resets,0u);EXPECT_EQ(opaque(originalMemory),committed);
 }
 // Handoff never acquired another worker or changed original PMA commitments.
 // The two real source populations retain their independent final retirement.
 EXPECT_EQ(foreign->retireReleasedContexts(),2u);EXPECT_EQ(t.destroyed,2u);EXPECT_EQ(t.resets,2u);
 t.expectedDestroyedBeforeReset=4;
 original.native_context_preparation.reset();
 EXPECT_EQ(source->retireReleasedContexts(),2u);EXPECT_EQ(t.destroyed,4u);EXPECT_EQ(t.resets,4u);EXPECT_TRUE(t.allDestroyed);
 EXPECT_NO_THROW(GPUDeviceContextPool::instance().shutdown());
}
}
