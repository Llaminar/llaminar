/** @file Test__PLELifecycle.cpp
 * @brief Dedicated original-tail ownership through actual CPU auxiliary and publication operations.
 * The graph/controller tests use real cache-owned accepted/next banks. Dedicated
 * transaction prior storage has no ordinary/reset/archive writer and is checked
 * byte-for-byte across repeated acceptance and subsequent transaction snapshots.
 */
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/OpenMPUtils.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen4exp/Qwen4ExpEarlyCurrentInput.h"
#include "models/GraphTypes.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
using namespace llaminar2;
namespace {
/** @brief Explicit already-materialized input/normalization producer in a device-free graph. */
class ReadyStage final:public IComputeStage {
public:
 /** @brief Keep a declared producer identity; foreign-device instances are negative controls only. */
 explicit ReadyStage(DeviceId device=DeviceId::cpu()):IComputeStage(device){}
 /** @brief Fixture inputs are initialized before this graph executes. */
 bool execute(IDeviceContext*)override{return true;}
 ComputeStageType type()const override{return ComputeStageType::COPY;}
 std::size_t estimatedFlops()const override{return 0;}
 std::size_t estimatedMemoryBytes()const override{return 0;}
 bool supportsBackend(ComputeBackendType b)const override{return b==ComputeBackendType::CPU;}
 StageBufferRequirements getBufferRequirements()const override{return {};}
 StageDumpInfo buildDumpInfoImpl()const override{return {}; }
};
/** @brief One real auxiliary cache owner and independently admitted verifier evidence. */
struct Fixture {
 static constexpr int C=8,M=16;
 MPIContext mpi{0,1,MPI_COMM_WORLD};
 std::shared_ptr<IKVCache>cache=createCPURingKVCache(ActivationPrecision::FP32,mpi,2,1,64,1,128,DeviceId::cpu());
 CPUDeviceContext context{DeviceId::cpu(),1};
 INT32Tensor ordinaryTokens{{4}},priorTokens{{2}},transaction{{1}},retainedTokens{{M}},copyBytes{{4}};
 FP32Tensor ordinaryConv{{18,C}},priorConv{{9,C}},retainedNorm{{M,C}},normalized{{M,C}};
 std::array<int,M>tokens{};int live=12;
 Qwen4ExpAuxiliaryState*owner=nullptr;
 /** @brief Attach the one real state owner; no model or GPU backend is initialized. */
 Fixture(){
  Qwen4ExpAuxiliaryState::Params p{.cache=*cache,.device=DeviceId::cpu(),.requestCapacity=1,.eosToken=42};
  p.pleHistories.push_back({1,1,0,C,{ordinaryTokens,BufferId::QFN_PLE_TOKEN_HISTORY},
   {ordinaryTokens,BufferId::QFN_PLE_TOKEN_HISTORY,8},{ordinaryConv,BufferId::QFN_PLE_CONV_HISTORY},
   {ordinaryConv,BufferId::QFN_PLE_CONV_HISTORY,9*C*4},
   Qwen4ExpAuxiliaryState::PLEVerifierStorage{M,{priorTokens,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY},
    {transaction,BufferId::QFN_PLE_TRANSACTION_LIVE},{retainedTokens,BufferId::QFN_PLE_RETAINED_TOKENS},
    {copyBytes,BufferId::QFN_PLE_TRANSACTION_COPY_BYTES},{priorConv,BufferId::QFN_PLE_PRIOR_CONV_HISTORY},
    {retainedNorm,BufferId::QFN_PLE_RETAINED_NORMALIZED}}});
  auto state=std::make_unique<Qwen4ExpAuxiliaryState>(std::move(p));owner=state.get();cache->attachAuxiliaryState(std::move(state),nullptr);
  auto*h=static_cast<int*>(ordinaryTokens.raw_mutable_data());h[0]=100;h[1]=101;
  for(int i=0;i<9*C;++i)ordinaryConv.mutable_data()[i]=float(-1000-i);
  for(int i=0;i<M;++i){tokens[i]=i+300;for(int c=0;c<C;++c)normalized.mutable_data()[i*C+c]=float(i*100+c);}
 }
 /** @brief Borrow the actual canonical CPU token/count authority; no tensor view is synthesized. */
 Qwen4ExpIntInput tokenInput(){return Qwen4ExpNativeIntInput{DeviceId::cpu(),tokens.data(),M};}
 /** @brief Original token-row live owner, never multiplied into a branch-row count. */
 Qwen4ExpStageRows rows(){return {M,Qwen4ExpIntInput{Qwen4ExpNativeIntInput{DeviceId::cpu(),&live,1}}};}
 /** @brief Execute the actual graph's topological stage order. */
 void execute(ComputeGraph&g){for(const auto&name:g.getExecutionOrder())ASSERT_TRUE(g.getNode(name)->stage->execute(&context));}
 /** @brief Existing canonical CPU publisher is the sole accepted-state mutation boundary. */
 MTPSpecStatePublicationResult publish(ComputeGraph&g,int accepted){MTPSpecStepPlanBatch p;p.ok=true;p.request_count=1;p.steps.push_back({.request_index=0,.draft_count=15,.accepted_count=accepted});const int row=accepted?accepted-1:-1;auto stages=g.getExecutionStages();return publishAcceptedMTPSpecStateFromVerifierRows(p,&row,stages,DeviceId::cpu(),nullptr,true);}
 /** @brief Distinguish an existing local producer from a participant's first model operation. */
 enum class Entry { LocalProducer, PipelineIngress };
 /** @brief Snapshot then retain through the real capability, preserving a pipeline fragment's root. */
 void compose(ComputeGraph &graph, Entry entry=Entry::LocalProducer)
 {
  std::vector<std::string> dependencies;
  if(entry==Entry::LocalProducer) {
   graph.addNode("tokens_ready",std::make_unique<ReadyStage>());
   dependencies={"tokens_ready"};
  }
  auto prior=owner->appendVerifierPLEPrior(graph,"ple_",0,rows(),tokenInput(),
   {normalized,BufferId::QFN_PLE_NORMALIZED},dependencies);
  EXPECT_EQ(prior.tokens().buffer,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY);
  graph.addNode("conv_norm",std::make_unique<ReadyStage>());
  graph.addDependency("conv_norm",prior.snapshotProducer());
  owner->appendVerifierPLERetain(graph,"ple_",std::move(prior),tokenInput(),
   {normalized,BufferId::QFN_PLE_NORMALIZED},{"conv_norm"});
  EXPECT_THROW((void)prior.snapshotProducer(),std::logic_error);
 }
};
}
/**
 * @test A PLE-only pipeline fragment starts at its actual snapshot and replays exact live rows.
 *
 * The pipeline owner encloses graph roots after model construction. The model
 * fragment must therefore preserve this real snapshot as its root, without a
 * fabricated readiness node. CPU replay checks the same snapshot/retention
 * implementation across reset, empty work and shrinking then growing batches.
 */
TEST(PLELifecycle, PipelineEntrySnapshotReplaysResetEmptyAndPartialRows)
{
 llaminar::v2::ThreadCountGuard threads(1);
 Fixture fixture;ComputeGraph graph;
 ASSERT_NO_THROW(fixture.compose(graph,Fixture::Entry::PipelineIngress));
 ASSERT_EQ(graph.getRootNodes(),std::vector<std::string>{"ple_transaction_prior"});
 ASSERT_EQ(graph.size(),3u);
 EXPECT_EQ(graph.getNode("conv_norm")->dependencies,std::vector<std::string>{"ple_transaction_prior"});
 for(int replay=0;replay<3;++replay)for(int live:{16,3,0,1,12}) {
  SCOPED_TRACE(::testing::Message()<<"replay="<<replay<<", live="<<live);
  fixture.live=live;
  const std::array<int,2> acceptedTokens{
   static_cast<const int*>(fixture.ordinaryTokens.raw_data())[0],
   static_cast<const int*>(fixture.ordinaryTokens.raw_data())[1]};
  const std::vector<float> acceptedConvolution(fixture.ordinaryConv.data(),
   fixture.ordinaryConv.data()+9*Fixture::C);
  std::fill_n(static_cast<int*>(fixture.retainedTokens.raw_mutable_data()),Fixture::M,-777);
  std::fill_n(fixture.retainedNorm.mutable_data(),Fixture::M*Fixture::C,-777.0f);
  fixture.execute(graph);
  if(live) {
   EXPECT_EQ(std::memcmp(fixture.priorTokens.raw_data(),acceptedTokens.data(),8),0);
   EXPECT_EQ(std::memcmp(fixture.priorConv.raw_data(),acceptedConvolution.data(),9*Fixture::C*4),0);
  }
  for(int row=0;row<Fixture::M;++row) {
   EXPECT_EQ(static_cast<const int*>(fixture.retainedTokens.raw_data())[row],
    row<live?fixture.tokens[row]:-777);
   for(int channel=0;channel<Fixture::C;++channel)
    EXPECT_EQ(fixture.retainedNorm.data()[row*Fixture::C+channel],
     row<live?fixture.normalized.data()[row*Fixture::C+channel]:-777.0f);
  }
  const auto *bytes=static_cast<const std::uint64_t*>(fixture.copyBytes.raw_data());
  EXPECT_EQ(bytes[0],live?8u+9u*Fixture::C*4u:0u);
  EXPECT_EQ(bytes[1],std::uint64_t(live)*(1+Fixture::C)*4u);
  const auto result=fixture.publish(graph,std::min(live,2));
  ASSERT_TRUE(result.ok)<<result.error;
  fixture.owner->reset({0,2,0,1},nullptr);
 }
}

/** @test An empty dependency list is valid only at graph entry; named producers remain exact. */
TEST(PLELifecycle, PriorRejectsMissingAndForeignProducerBeforeMutation)
{
 Fixture fixture;
 for(const auto foreign:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)}) {
  ComputeGraph graph;
  graph.addNode("producer",std::make_unique<ReadyStage>(foreign),foreign);
  const auto append=[&](const std::vector<std::string>&dependencies) {
   return fixture.owner->appendVerifierPLEPrior(graph,"ple_",0,fixture.rows(),fixture.tokenInput(),
    {fixture.normalized,BufferId::QFN_PLE_NORMALIZED},dependencies);
  };
  EXPECT_THROW(append({}),std::invalid_argument);
  EXPECT_THROW(append({"missing"}),std::invalid_argument);
  EXPECT_EQ(graph.size(),1u);
  if(foreign==DeviceId::cpu())EXPECT_NO_THROW(append({"producer"}));
  else {
   EXPECT_THROW(append({"producer"}),std::invalid_argument);
   EXPECT_EQ(graph.size(),1u);
  }
 }
}
/** @test Alias-isolation negative control: direct ordinary/reset writes cannot touch dedicated prior bytes.
 * This deliberately bypasses production transaction sequencing to probe physical
 * aliases. It does NOT authorize publishing an old transaction after real reset
 * or retirement, and the test performs no publication after that reset.
 */
TEST(PLELifecycle, DedicatedPriorSurvivesOrdinaryNextPromotionAndReset){llaminar::v2::ThreadCountGuard threads(1);Fixture f;ComputeGraph verifier;f.compose(verifier);f.execute(verifier);std::array<int,2>originalT;std::memcpy(originalT.data(),f.priorTokens.raw_data(),8);const auto originalC=std::vector<float>(f.priorConv.data(),f.priorConv.data()+9*Fixture::C);
 ComputeGraph ordinary;ordinary.addNode("ready",std::make_unique<ReadyStage>());f.owner->appendAcceptedPLECommit(ordinary,"ordinary_",0,{1,std::nullopt},{std::nullopt,1},f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});f.execute(ordinary);f.owner->reset({0,2,0,1},nullptr);
 EXPECT_EQ(std::memcmp(originalT.data(),f.priorTokens.raw_data(),8),0);EXPECT_EQ(std::memcmp(originalC.data(),f.priorConv.raw_data(),9*Fixture::C*4),0);
 EXPECT_NE(std::memcmp(originalT.data(),f.ordinaryTokens.raw_data(),8),0);
}
/** @test A still-live transaction can revise accepted12 to0 without any reset/retirement boundary. */
TEST(PLELifecycle, RevisedAllZeroRestoresOriginalWithinSameLiveTransaction){
 llaminar::v2::ThreadCountGuard threads(1);Fixture f;ComputeGraph verifier;f.compose(verifier);f.execute(verifier);
 std::array<int,2>originalT;std::memcpy(originalT.data(),f.priorTokens.raw_data(),8);
 const auto originalC=std::vector<float>(f.priorConv.data(),f.priorConv.data()+9*Fixture::C);
 auto result=f.publish(verifier,12);ASSERT_TRUE(result.ok)<<result.error;
 result=f.publish(verifier,0);ASSERT_TRUE(result.ok)<<result.error;
 EXPECT_EQ(std::memcmp(originalT.data(),f.ordinaryTokens.raw_data(),8),0);
 EXPECT_EQ(std::memcmp(originalC.data(),f.ordinaryConv.raw_data(),9*Fixture::C*4),0);
}
/** @test Revised publication and then a later snapshot use exactly their own transaction originals. */
TEST(PLELifecycle, RepeatAcceptanceAndNextTransactionUseDistinctImmutableOrigins){llaminar::v2::ThreadCountGuard threads(1);Fixture f;ComputeGraph g;f.compose(g);f.execute(g);const auto prior=std::vector<float>(f.priorConv.data(),f.priorConv.data()+9*Fixture::C);
 for(int n:{12,2,0,7}){auto r=f.publish(g,n);ASSERT_TRUE(r.ok)<<r.error;EXPECT_EQ(std::memcmp(prior.data(),f.priorConv.raw_data(),9*Fixture::C*4),0);}
 // CPU publication is synchronous: its return retires every prior reader. The
 // next graph execution is therefore the next transaction, not an overlapping
 // snapshot. GPU production supplies this same edge through native fragments/events.
 const auto accepted=std::vector<float>(f.ordinaryConv.data(),f.ordinaryConv.data()+9*Fixture::C);f.live=1;f.execute(g);EXPECT_EQ(std::memcmp(accepted.data(),f.priorConv.raw_data(),9*Fixture::C*4),0);auto r=f.publish(g,0);ASSERT_TRUE(r.ok)<<r.error;EXPECT_EQ(std::memcmp(accepted.data(),f.ordinaryConv.raw_data(),9*Fixture::C*4),0);
}
/** @test Empty observed transactions preserve all dedicated payload bytes and publish exact zero traffic. */
TEST(PLELifecycle, EmptyTransactionCopiesNoPriorOrCandidatePayload){llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.live=0;std::fill_n(f.priorConv.mutable_data(),9*Fixture::C,-777.0f);std::fill_n(static_cast<int*>(f.priorTokens.raw_mutable_data()),2,-777);ComputeGraph g;f.compose(g);f.execute(g);const auto*b=static_cast<const std::uint64_t*>(f.copyBytes.raw_data());EXPECT_EQ(b[0],0u);EXPECT_EQ(b[1],0u);auto r=f.publish(g,0);ASSERT_TRUE(r.ok)<<r.error;for(int i=0;i<9*Fixture::C;++i)EXPECT_EQ(f.priorConv.data()[i],-777.0f);}
/** @test A capability cannot be consumed by a foreign graph or reused after valid composition. */
TEST(PLELifecycle, CapabilityRejectsCrossGraphAndDoubleConsumption){Fixture f;ComputeGraph a,b;a.addNode("ready",std::make_unique<ReadyStage>());b.addNode("ready",std::make_unique<ReadyStage>());auto p=f.owner->appendVerifierPLEPrior(a,"ple_",0,f.rows(),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});
 EXPECT_THROW(f.owner->appendVerifierPLERetain(b,"ple_",std::move(p),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"}),std::invalid_argument);f.owner->appendVerifierPLERetain(a,"ple_",std::move(p),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});
 EXPECT_THROW(f.owner->appendVerifierPLERetain(a,"second_",std::move(p),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"}),std::invalid_argument);}
/** @test Sole declaration/PMA input prices exactly dedicated tails, live metadata and live-capacity candidates. */
TEST(PLELifecycle, DedicatedPLEMTPDeclarationsUseExactGeometryAndPlacement){
 Qwen4ExpMemoryGeometry geometry;geometry.residual_width=32;geometry.residual_branches=4;
 geometry.residual_gate_rank=8;geometry.compression_ratio=4;geometry.ple_heads=16;
 geometry.ple_table_row_bytes=90;geometry.ple_decoder_layer=1;geometry.sparse_decoder_layers={3};
 const std::array<const char*,6>names{"qfn_ple_prior_token_history","qfn_ple_prior_conv_history","qfn_ple_transaction_live","qfn_ple_retained_tokens","qfn_ple_retained_normalized","qfn_ple_transaction_copy_bytes"};
 for(auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})for(int requests:{1,3})for(int retained:{0,requests*8})for(bool owns:{false,true}){
  Qwen4ExpBufferDimensions d{.main_layers=4,.first_layer=owns?0:2,.layer_count=owns?4:2,
   .compute_width=8,.context_tokens=64,.request_capacity=requests,.flattened_rows=64,
   .local_query_heads=1,.retained_mtp_candidate_rows=retained,.device=device};
  GraphSchema schema;applyQwen4ExpBufferLayout(schema,geometry,d);GraphResolverConfig resolver;
  resolver.seq_len=64;resolver.d_model=8;resolver.local_n_heads=1;applyQwen4ExpBufferGeometry(resolver,geometry,d);
  std::size_t total=0;for(auto name:names){const auto it=std::find_if(schema.layer_buffers.begin(),schema.layer_buffers.end(),[&](const auto&b){return b.name==name;});EXPECT_EQ(it!=schema.layer_buffers.end(),owns&&retained>0);if(it!=schema.layer_buffers.end()){auto allocation=BufferAllocator::resolve(*it,resolver);std::size_t bytes=4;for(auto n:allocation.shape)bytes*=n;total+=bytes;EXPECT_EQ(it->semantic,BufferSemantic::InOut);}}
  const auto expected=owns&&retained>0?std::size_t(requests)*(8+9*32*4+4+16)+std::size_t(retained)*(1+32)*4:0;
  EXPECT_EQ(total,expected);
 }
}
/** @test Replacing the named producer invalidates its weak stage-owned proof before edges change. */
TEST(PLELifecycle, CapabilityRejectsProducerReplacementBeforeMutation){
 Fixture f;ComputeGraph graph;graph.addNode("ready",std::make_unique<ReadyStage>());
 auto prior=f.owner->appendVerifierPLEPrior(graph,"ple_",0,f.rows(),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});
 const auto producer=prior.snapshotProducer();graph.addNode(producer,std::make_unique<ReadyStage>());
 EXPECT_THROW(prior.snapshotProducer(),std::logic_error);
 EXPECT_THROW(f.owner->appendVerifierPLERetain(graph,"ple_",std::move(prior),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"}),std::invalid_argument);
 EXPECT_EQ(graph.getNode("ple_retain_candidates"),nullptr);
}
/** @test Clearing/rebuilding the same graph/name cannot revive an original capability. */
TEST(PLELifecycle, CapabilityRejectsClearRebuildBeforeMutation){
 Fixture f;ComputeGraph graph;graph.addNode("ready",std::make_unique<ReadyStage>());
 auto prior=f.owner->appendVerifierPLEPrior(graph,"ple_",0,f.rows(),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});
 graph.clear();graph.addNode("ready",std::make_unique<ReadyStage>());
 auto current=f.owner->appendVerifierPLEPrior(graph,"ple_",0,f.rows(),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"});
 EXPECT_THROW(f.owner->appendVerifierPLERetain(graph,"old_",std::move(prior),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"}),std::invalid_argument);
 EXPECT_EQ(graph.getNode("old_retain_candidates"),nullptr);
 EXPECT_NO_THROW(f.owner->appendVerifierPLERetain(graph,"new_",std::move(current),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{"ready"}));
}
/** @test Capture identity follows exact native ownership/policy while counts remain device/CPU state. */
TEST(PLELifecycle, RetainedStageCaptureIdentityUsesNativeOwnersAndGeometry){
 Fixture f;ComputeGraph graph;f.compose(graph);
 auto *stage=dynamic_cast<Qwen4ExpPLERetainedStage*>(graph.getNode("ple_transaction_prior")->stage.get());ASSERT_NE(stage,nullptr);
 const auto original=stage->graphCaptureVariantSignature();f.live=0;EXPECT_EQ(original,stage->graphCaptureVariantSignature());
 auto params=stage->getParams();int alternateLive=0;params.liveCounts=Qwen4ExpNativeIntInput{DeviceId::cpu(),&alternateLive,1};
 Qwen4ExpPLERetainedStage otherOwner(params);EXPECT_NE(original,otherOwner.graphCaptureVariantSignature());
 params=stage->getParams();params.rowCapacity=15;Qwen4ExpPLERetainedStage otherRows(params);EXPECT_NE(original,otherRows.graphCaptureVariantSignature());
 params=stage->getParams();params.copiedBytes.byteOffset=8;Qwen4ExpPLERetainedStage otherOffset(params);EXPECT_NE(original,otherOffset.graphCaptureVariantSignature());
 params=stage->getParams();int firstRow=0;params.verifierOrigins=RequestRowOrigins::nativeWords(&firstRow);
 Qwen4ExpPLERetainedStage nativeOrigins(params);EXPECT_NE(original,nativeOrigins.graphCaptureVariantSignature());
 const auto nativeIdentity=nativeOrigins.graphCaptureVariantSignature();firstRow=100;EXPECT_EQ(nativeIdentity,nativeOrigins.graphCaptureVariantSignature());
}
#ifdef QFN_PRIVATE_CPU_STANDALONE
/** @brief Initialize only MPI's host singleton metadata for the real CPU cache fixture. */
int main(int argc,char**argv){MPI_Init(&argc,&argv);::testing::InitGoogleTest(&argc,argv);const int result=RUN_ALL_TESTS();MPI_Finalize();return result;}

#endif

/** @test The real prior/hash/retain implementation replays one original transaction without accepted-state writes. */
TEST(PLELifecycle, EarlyCurrentPriorHashRetainReplay)
{
 llaminar::v2::ThreadCountGuard threads(1);
 for(const int capacity:{1,2,3,4,16}) {
  Fixture f;ComputeGraph graph;
  const Qwen4ExpStageRows rows{capacity,f.rows().liveRows};
  auto prior=f.owner->appendVerifierPLEPrior(graph,"early_",0,rows,f.tokenInput(),
   {f.normalized,BufferId::QFN_PLE_NORMALIZED},{});
  INT32Tensor ids({Fixture::M*16*2});
  qwen4exp::NgramHashPolicy policy;policy.multipliers={9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};policy.eosToken=42;
  for(int head=0;head<16;++head){policy.moduli[head]=1009+head;policy.offsets[head]=2048*head;}
  ASSERT_NO_THROW(prior.requireFor(graph,DeviceId::cpu(),rows,prior.tokens(),prior.convolution()));
  graph.addNode("hash",std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
   .device_id=DeviceId::cpu(),.rows=rows,.operation=Qwen4ExpPrimitiveStage::NgramHash{
    f.tokenInput(),prior.tokens(),{ids,BufferId::QFN_NGRAM_ROW_IDS},policy}}));
  prior.orderBefore(graph,"hash");
  graph.addNode("trunk",std::make_unique<ReadyStage>());graph.addDependency("trunk","hash");
  f.owner->appendVerifierPLERetain(graph,"early_",std::move(prior),f.tokenInput(),
   {f.normalized,BufferId::QFN_PLE_NORMALIZED},{"trunk"});
  EXPECT_THROW(prior.orderBefore(graph,"hash"),std::invalid_argument);
  const auto order=graph.getExecutionOrder();
  const auto at=[&](const char*name){return std::find(order.begin(),order.end(),name)-order.begin();};
  ASSERT_LT(at("early_transaction_prior"),at("hash"));ASSERT_LT(at("hash"),at("trunk"));
  ASSERT_LT(at("trunk"),at("early_retain_candidates"));
  for(int replay=0;replay<20;++replay)for(int live:{capacity,std::min(3,capacity),0,1}) {
   SCOPED_TRACE(::testing::Message()<<"capacity="<<capacity<<" replay="<<replay<<" live="<<live);
   f.live=live;graph.reset();ASSERT_EQ(graph.getExecutionOrder(),order);
   const std::array<int,2> history{static_cast<const int*>(f.ordinaryTokens.raw_data())[0],static_cast<const int*>(f.ordinaryTokens.raw_data())[1]};
   std::fill_n(static_cast<int*>(f.priorTokens.raw_mutable_data()),2,-987);
   std::fill_n(reinterpret_cast<std::int64_t*>(ids.raw_mutable_data()),Fixture::M*16,-123);
   f.execute(graph);
   ASSERT_EQ(std::memcmp(f.ordinaryTokens.raw_data(),history.data(),8),0);
   for(int row=0;row<capacity;++row)for(int head=0;head<16;++head){
    std::int64_t expected=-123;
    if(row<live){const auto previous=row?f.tokens[row-1]:history[1];auto second=row>1?f.tokens[row-2]:history[row];
     if(previous==policy.eosToken)second=policy.eosToken;
     auto hash=std::uint64_t(f.tokens[row])*policy.multipliers[0] ^ std::uint64_t(previous)*policy.multipliers[1];
     if(head>=8)hash^=std::uint64_t(second)*policy.multipliers[2];
     expected=hash%policy.moduli[head]+policy.offsets[head];}
    ASSERT_EQ(reinterpret_cast<const std::int64_t*>(ids.raw_data())[row*16+head],expected);
   }
   ASSERT_TRUE(f.publish(graph,std::min(live,2)).ok);
   f.owner->reset({0,2,0,1},nullptr);
  }
 }
}
/** @test Genuine prior capability rejects substituted count, bank, device and graph before mutation. */
TEST(PLELifecycle, EarlyCurrentPriorRejectsSubstitutedAuthority)
{
 Fixture f;ComputeGraph graph,foreign;
 auto prior=f.owner->appendVerifierPLEPrior(graph,"early_",0,f.rows(),f.tokenInput(),{f.normalized,BufferId::QFN_PLE_NORMALIZED},{});
 const auto before=graph.getExecutionOrder();int sameValue=f.live;
 auto wrongRows=f.rows();wrongRows.liveRows=Qwen4ExpNativeIntInput{DeviceId::cpu(),&sameValue,1};
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cpu(),wrongRows,prior.tokens(),prior.convolution()),std::invalid_argument);
 wrongRows=f.rows();--wrongRows.capacity;
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cpu(),wrongRows,prior.tokens(),prior.convolution()),std::invalid_argument);
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cuda(0),f.rows(),prior.tokens(),prior.convolution()),std::invalid_argument);
 EXPECT_THROW(prior.requireFor(foreign,DeviceId::cpu(),f.rows(),prior.tokens(),prior.convolution()),std::invalid_argument);
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cpu(),f.rows(),{f.ordinaryTokens,BufferId::QFN_PLE_TOKEN_HISTORY},prior.convolution()),std::invalid_argument);
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cpu(),f.rows(),prior.tokens(),{f.ordinaryConv,BufferId::QFN_PLE_CONV_HISTORY}),std::invalid_argument);
 ASSERT_EQ(graph.getExecutionOrder(),before);
 auto* snapshot=graph.getNode(prior.snapshotProducer());snapshot->device=DeviceId::cuda(0);
 EXPECT_THROW(prior.requireFor(graph,DeviceId::cpu(),f.rows(),prior.tokens(),prior.convolution()),std::invalid_argument);
 snapshot->device=DeviceId::cpu();
 EXPECT_NO_THROW(prior.requireFor(graph,DeviceId::cpu(),f.rows(),prior.tokens(),prior.convolution()));
}

/** @test The actual production role resolver keeps all scalar condition purposes on accepted history. */
TEST(PLELifecycle, EarlyCurrentConditionUsesOriginalScalarAcceptedHistory)
{
    for (int replay = 0; replay < 20; ++replay) {
        GraphConfig config;
        ForwardInput input;
        input.batch_size = 1;
        input.seq_len = 1;
        input.execution_phase = ForwardExecutionPhase::Decode;
        input.execution_role = ForwardExecutionRole::MTPCondition;
        config.live_mtp_request_batch_condition = true;
        // Original condition mailbox values are cached-token positions, not row
        // counts. Eligibility must never dereference/reinterpret that address.
        const std::int32_t cached_tokens = 128 + replay;
        input.sequence_lengths_device = &cached_tokens;
        for (const auto transaction : {ForwardStateTransaction::Ordinary,
                 ForwardStateTransaction::CommittedMTPCondition,
                 ForwardStateTransaction::RestoredPrefixMTPDecodeBridge}) {
            input.state_transaction = transaction;
            EXPECT_EQ(resolveQwen4ExpEarlyCurrentHistory(config, input),
                Qwen4ExpEarlyCurrentHistory::AcceptedState);
            EXPECT_EQ(cached_tokens, 128 + replay);
        }
        config.live_mtp_request_batch_condition = false;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
        config.live_mtp_request_batch_condition = true;
        input.seq_len = 2;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
        input.seq_len = 1;
        input.batch_size = 2;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
        input.batch_size = 1;
        input.execution_phase = ForwardExecutionPhase::Prefill;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
        input.execution_phase = ForwardExecutionPhase::Decode;
        config.grouped_mtp_verifier = true;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
        config.live_mtp_request_batch_condition = false;
        input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
        EXPECT_EQ(resolveQwen4ExpEarlyCurrentHistory(config, input),
            Qwen4ExpEarlyCurrentHistory::VerifierPrior);
        config.grouped_mtp_verifier = false;
        input.execution_role = ForwardExecutionRole::MainInference;
        EXPECT_EQ(resolveQwen4ExpEarlyCurrentHistory(config, input),
            Qwen4ExpEarlyCurrentHistory::AcceptedState);
        config.grouped_mtp_verifier = true;
        EXPECT_THROW(resolveQwen4ExpEarlyCurrentHistory(config, input), std::invalid_argument);
    }
}
