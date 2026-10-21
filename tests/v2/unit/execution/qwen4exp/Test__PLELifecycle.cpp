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
#include "execution/local_execution/graph/GraphResolver.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
using namespace llaminar2;
namespace {
/** @brief Explicit already-materialized input/normalization producer in a device-free graph. */
class ReadyStage final:public IComputeStage {
public:
 ReadyStage():IComputeStage(DeviceId::cpu()){}
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
 /** @brief Snapshot then retain through the typed owner capability and real graph dependencies. */
 void compose(ComputeGraph&g){g.addNode("tokens_ready",std::make_unique<ReadyStage>());auto prior=owner->appendVerifierPLEPrior(g,"ple_",0,rows(),tokenInput(),{normalized,BufferId::QFN_PLE_NORMALIZED},{"tokens_ready"});EXPECT_EQ(prior.tokens().buffer,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY);g.addNode("conv_norm",std::make_unique<ReadyStage>());g.addDependency("conv_norm",prior.snapshotProducer());owner->appendVerifierPLERetain(g,"ple_",std::move(prior),tokenInput(),{normalized,BufferId::QFN_PLE_NORMALIZED},{"conv_norm"});
 EXPECT_THROW((void)prior.snapshotProducer(),std::logic_error);}
};
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
