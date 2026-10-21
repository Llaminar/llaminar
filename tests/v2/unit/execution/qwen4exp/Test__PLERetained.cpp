/**
 * @file Test__PLERetained.cpp
 * @brief Actual private PLE stage acceptance, rollback and live-copy CPU proofs.
 *
 * The tests use the real arena stage and existing CPU publisher. Dirty capacity,
 * empty middle participants and late invalid decisions prove exact extents and
 * atomic publication rather than merely checking a happy-path convolution tail.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLERetainedStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
using namespace llaminar2;
namespace {
using Stage=Qwen4ExpPLERetainedStage;using Int=Qwen4ExpIntBinding;using Float=Qwen4ExpFloatBinding;
/** @brief Interleaved real accepted/next history banks plus independent retained candidates. */
struct Fixture {
 static constexpr int R=3,M=16,C=7;
 CPUDeviceContext context{DeviceId::cpu(),1};
 INT32Tensor live{{R}},transaction{{R}},tokens{{R*M}},retainedTokens{{R*M}},history{{R*4}},bytes{{R*4}};
 FP32Tensor normalized{{R*M,C}},retainedNormalized{{R*M,C}},convolution{{R*18,C}};
 std::array<int,R>origins{0,7,7};
 std::array<std::unique_ptr<Stage>,R>snapshots,retainers;
 std::vector<int>originalTokens;std::vector<float>originalConvolution;
 /** @brief Materialize all banks before any stage construction; no model/backend startup. */
 Fixture(){
  auto*length=static_cast<int*>(live.raw_mutable_data());length[0]=7;length[1]=0;length[2]=12;
  auto*h=static_cast<int*>(history.raw_mutable_data());auto*t=static_cast<int*>(tokens.raw_mutable_data());
  std::fill_n(static_cast<int*>(retainedTokens.raw_mutable_data()),R*M,-9999);std::fill_n(retainedNormalized.mutable_data(),R*M*C,-9999.0f);
  for(int r=0;r<R;++r){for(int i=0;i<4;++i)h[r*4+i]=-100-r*4-i;for(int i=0;i<M;++i){t[r*M+i]=r*100+i;for(int c=0;c<C;++c)normalized.mutable_data()[(r*M+i)*C+c]=float(r*1000+i*C+c);}for(int i=0;i<18*C;++i)convolution.mutable_data()[r*18*C+i]=-float(r*1000+i+1);}
  originalTokens.assign(h,h+R*4);originalConvolution.assign(convolution.data(),convolution.data()+R*18*C);
  for(int r=0;r<R;++r){snapshots[r]=std::make_unique<Stage>(params(r,Stage::Operation::SnapshotPrior));retainers[r]=std::make_unique<Stage>(params(r,Stage::Operation::RetainCandidates));}
 }
 /** @brief Exact subview identities match the production interleaved history geometry. */
 Stage::Params params(int r,Stage::Operation op){return {.device_id=DeviceId::cpu(),.operation=op,.requests=R,.requestIndex=r,.rowCapacity=M,.channels=C,
  .liveCounts=Int(live,BufferId::QFN_QSA_SELECTED_COUNT),.transactionLiveCounts=Int(transaction,BufferId::QFN_PLE_TRANSACTION_LIVE),
  .acceptedTokens=Int(history,BufferId::QFN_PLE_TOKEN_HISTORY,r*16),.priorTokens=Int(history,BufferId::QFN_PLE_TOKEN_HISTORY,r*16+8),
  .retainedTokens=Int(retainedTokens,BufferId::QFN_PLE_RETAINED_TOKENS,r*M*4),.tokens=Int(tokens,BufferId::REQUEST_TOKEN_IDS,r*M*4),
  .acceptedConvolution=Float(convolution,BufferId::QFN_PLE_CONV_HISTORY,r*18*C*4),.priorConvolution=Float(convolution,BufferId::QFN_PLE_CONV_HISTORY,(r*18+9)*C*4),
  .normalized=Float(normalized,BufferId::QFN_PLE_NORMALIZED,r*M*C*4),.retainedNormalized=Float(retainedNormalized,BufferId::QFN_PLE_RETAINED_NORMALIZED,r*M*C*4),
  .copiedBytes=Int(bytes,BufferId::QFN_PLE_TRANSACTION_COPY_BYTES,(r*2+(op==Stage::Operation::RetainCandidates))*8),.verifierOrigins=RequestRowOrigins::nativeWords(origins.data())};}
 /** @brief The snapshot edge is complete for all requests before any verifier work. */
 void begin(){for(auto&s:snapshots)ASSERT_TRUE(s->execute(&context));for(auto&s:retainers)ASSERT_TRUE(s->execute(&context));}
 /** @brief Plan every request before invoking any live mutation hook. */
 bool publish(const std::array<int,R>&rows){std::vector<CPUVerifierStateRestorePlan>plans;for(auto&s:retainers)if(!s->planCPUVerifierStateRestoreRows(rows.data(),R,plans))return false;for(auto&s:retainers)if(!s->restoreVerifierStateCaptureRows(rows.data(),R,nullptr))return false;return true;}
 /** @brief Compare both exact suffixes to one immutable transaction origin. */
 void check(int r,int accepted,int liveRows){
  auto*h=static_cast<const int*>(history.raw_data());auto*t=static_cast<const int*>(tokens.raw_data());
  for(int i=0;i<2;++i){const int source=accepted-2+i;EXPECT_EQ(h[r*4+i],!liveRows?originalTokens[r*4+i]:source<0?originalTokens[r*4+source+2]:t[r*M+source]);}
  for(int i=0;i<9;++i)for(int c=0;c<C;++c){const int source=accepted-9+i;EXPECT_EQ(convolution.data()[(r*18+i)*C+c],!liveRows?originalConvolution[(r*18+i)*C+c]:source<0?originalConvolution[(r*18+source+9)*C+c]:normalized.data()[(r*M+source)*C+c]);}
 }
};
}
/** @test Full/partial/zero repeated publications always use the same original two-token/nine-row tails. */
TEST(PLERetained, ExactSuffixesAndRepeatedRollbackAcrossHistoryBoundary){llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.begin();
 for(auto counts:std::array<std::array<int,Fixture::R>,5>{{{7,0,12},{1,0,2},{0,0,0},{6,0,10},{0,0,0}}}){std::array<int,3>rows;for(int r=0;r<3;++r)rows[r]=counts[r]?f.origins[r]+counts[r]-1:-1;ASSERT_TRUE(f.publish(rows));for(int r=0;r<3;++r)f.check(r,counts[r],r==1?0:1);}
 // Next banks remain the original prestate through every republication.
 auto*h=static_cast<const int*>(f.history.raw_data());for(int r:{0,2}){EXPECT_EQ(h[r*4+2],f.originalTokens[r*4]);EXPECT_EQ(h[r*4+3],f.originalTokens[r*4+1]);EXPECT_EQ(std::memcmp(f.convolution.data()+(r*18+9)*Fixture::C,f.originalConvolution.data()+r*18*Fixture::C,9*Fixture::C*4),0);}
}
/** @test Empty holes and large-to-small replays copy only live payload; stale candidates are unobserved. */
TEST(PLERetained, MixedEmptyAndLargeToSmallRetainOnlyLiveBytes){llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.begin();auto*copy=static_cast<const std::uint64_t*>(f.bytes.raw_data());for(int r=0;r<3;++r){int n=r==0?7:r==1?0:12;EXPECT_EQ(copy[r*2],n?8u+36u*Fixture::C:0u);EXPECT_EQ(copy[r*2+1],std::uint64_t(n)*(Fixture::C+1)*4);for(int row=n;row<Fixture::M;++row)EXPECT_EQ(static_cast<const int*>(f.retainedTokens.raw_data())[r*Fixture::M+row],-9999);}
 auto*length=static_cast<int*>(f.live.raw_mutable_data());length[0]=1;length[1]=0;length[2]=0;
 std::fill_n(static_cast<int*>(f.retainedTokens.raw_mutable_data()),3*Fixture::M,-7777);f.begin();EXPECT_EQ(copy[1],(Fixture::C+1)*4u);EXPECT_EQ(copy[2],0u);EXPECT_EQ(copy[3],0u);EXPECT_EQ(copy[4],0u);EXPECT_EQ(copy[5],0u);
 for(int i=1;i<3*Fixture::M;++i)EXPECT_EQ(static_cast<const int*>(f.retainedTokens.raw_data())[i],-7777);
 ASSERT_TRUE(f.publish({0,-1,-1}));
}
/** @test A corrupt later decision cannot modify any request, even through a direct native commit. */
TEST(PLERetained, AllRequestInvalidDecisionAtomicity){llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.begin();auto beforeT=std::vector<int>(static_cast<const int*>(f.history.raw_data()),static_cast<const int*>(f.history.raw_data())+12);auto beforeC=std::vector<float>(f.convolution.data(),f.convolution.data()+f.convolution.numel());EXPECT_FALSE(f.publish({0,-1,19}));EXPECT_EQ(std::memcmp(beforeT.data(),f.history.raw_data(),f.history.size_bytes()),0);EXPECT_EQ(std::memcmp(beforeC.data(),f.convolution.raw_data(),f.convolution.size_bytes()),0);
 const int invalid[]{0,-1,19};EXPECT_THROW(f.retainers[0]->restoreVerifierStateCaptureRows(invalid,3,nullptr),std::invalid_argument);EXPECT_EQ(std::memcmp(beforeT.data(),f.history.raw_data(),f.history.size_bytes()),0);
}
/** @test Actual existing publisher validates all stages/requests before any PLE native splice. */
TEST(PLERetained, ExistingPublisherPlansBeforeAnyNativeMutation){llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.begin();std::vector<IComputeStage*>stages;for(auto&s:f.retainers)stages.push_back(s.get());MTPSpecStepPlanBatch batch;batch.ok=true;batch.request_count=3;for(int r=0;r<3;++r)batch.steps.push_back({.request_index=r,.draft_count=15,.accepted_count=r==1?0:1});const int invalid[]{0,-1,19};auto result=publishAcceptedMTPSpecStateFromVerifierRows(batch,invalid,stages,DeviceId::cpu(),nullptr,true);EXPECT_FALSE(result.ok);f.check(0,0,1);f.check(2,0,1);const int valid[]{0,-1,7};result=publishAcceptedMTPSpecStateFromVerifierRows(batch,valid,stages,DeviceId::cpu(),nullptr,true);ASSERT_TRUE(result.ok)<<result.error;f.check(0,1,1);f.check(1,0,0);f.check(2,1,1);}
/** @test Constructor rejects snapshot/input aliasing and misaligned packed observations. */
TEST(PLERetained, SetupRejectsAliasingAndCapacityViolations){Fixture f;auto p=f.params(0,Stage::Operation::SnapshotPrior);p.priorTokens=p.acceptedTokens;EXPECT_THROW((void)Stage{p},std::invalid_argument);
p=f.params(0,Stage::Operation::RetainCandidates);p.copiedBytes.byteOffset=4;EXPECT_THROW((void)Stage{p},std::invalid_argument);
p=f.params(0,Stage::Operation::RetainCandidates);p.retainedTokens.byteOffset=f.retainedTokens.size_bytes()-4;EXPECT_THROW((void)Stage{p},std::invalid_argument);
}
/** @test An observed all-empty transaction preserves both histories and every unused candidate byte. */
TEST(PLERetained, AllEmptyTransactionHasNoPayloadWrites){
 llaminar::v2::ThreadCountGuard threads(1);Fixture f;
 std::fill_n(static_cast<int*>(f.live.raw_mutable_data()),3,0);
 const std::vector<int>tokensBefore(static_cast<const int*>(f.history.raw_data()),static_cast<const int*>(f.history.raw_data())+12);
 const std::vector<float>convBefore(f.convolution.data(),f.convolution.data()+f.convolution.numel());
 f.begin();ASSERT_TRUE(f.publish({-1,-1,-1}));
 EXPECT_EQ(std::memcmp(tokensBefore.data(),f.history.raw_data(),f.history.size_bytes()),0);
 EXPECT_EQ(std::memcmp(convBefore.data(),f.convolution.raw_data(),f.convolution.size_bytes()),0);
 const auto*copy=static_cast<const std::uint64_t*>(f.bytes.raw_data());
 for(int r=0;r<3;++r){EXPECT_EQ(copy[r*2],0u);EXPECT_EQ(copy[r*2+1],0u);}
 std::vector<CPUVerifierStateRestorePlan>plans;const int decisions[]{-1,-1,-1};
 for(auto&s:f.retainers)ASSERT_TRUE(s->planCPUVerifierStateRestoreRows(decisions,3,plans));
 for(const auto&p:plans)EXPECT_TRUE(p.native_destinations.empty());
}
/** @test Admission/frontier updates cannot replace the immutable transaction live authority. */
TEST(PLERetained, PublicationUsesCapturedLiveCountsRatherThanLaterAdmissionMetadata){
 llaminar::v2::ThreadCountGuard threads(1);Fixture f;f.begin();
 std::fill_n(static_cast<int*>(f.live.raw_mutable_data()),3,0);
 ASSERT_TRUE(f.publish({6,-1,18}));f.check(0,7,1);f.check(1,0,0);f.check(2,12,1);
 ASSERT_TRUE(f.publish({-1,-1,-1}));f.check(0,0,1);f.check(2,0,1);
}
/** @test Native generation token inputs retain their explicit owner and need no tensor row view/copy. */
TEST(PLERetained, NativeTokenProducerBindingHasExactOwnerAndExtent){
 llaminar::v2::ThreadCountGuard threads(1);Fixture f;
 for(auto&s:f.snapshots)ASSERT_TRUE(s->execute(&f.context));
 auto p=f.params(2,Stage::Operation::RetainCandidates);
 p.tokens=Qwen4ExpNativeIntInput{DeviceId::cpu(),static_cast<const int*>(f.tokens.raw_data())+2*Fixture::M,Fixture::M};
 Stage native(p);ASSERT_TRUE(native.execute(&f.context));
 EXPECT_EQ(std::memcmp(static_cast<const int*>(f.retainedTokens.raw_data())+2*Fixture::M,static_cast<const int*>(f.tokens.raw_data())+2*Fixture::M,12*4),0);
 p.tokens=Qwen4ExpNativeIntInput{DeviceId::cuda(0),static_cast<const int*>(f.tokens.raw_data()),Fixture::M};
 EXPECT_THROW((void)Stage{p},std::invalid_argument);
 p.tokens=Qwen4ExpNativeIntInput{DeviceId::cpu(),static_cast<const int*>(f.tokens.raw_data()),Fixture::M-1};
 EXPECT_THROW((void)Stage{p},std::invalid_argument);
}
/** @test Tail publication preserves subnormals, signed zeros and NaN payloads as native bits. */
TEST(PLERetained, ConvolutionTailsAreBitCopiesWithoutFloatArithmetic){
 llaminar::v2::ThreadCountGuard threads(1);Fixture f;
 const std::array<std::uint32_t,4>bits{0x80000000u,0x00000001u,0x7fc12345u,0xff812345u};
 for(int i=0;i<7*Fixture::C;++i)std::memcpy(f.normalized.mutable_data()+i,&bits[i%4],4);
 f.begin();ASSERT_TRUE(f.publish({6,-1,-1}));
 EXPECT_EQ(std::memcmp(f.convolution.data()+2*Fixture::C,f.normalized.data(),7*Fixture::C*4),0);
 ASSERT_TRUE(f.publish({-1,-1,-1}));
 EXPECT_EQ(std::memcmp(f.convolution.data(),f.originalConvolution.data(),9*Fixture::C*4),0);
}

/** @test New prior/live/candidate observations are cold outputs, not already materialized banks. */
TEST(PLERetained, ColdWriteOnlyRootsUseOrdinaryPreparation) {
 Fixture f;
 for(auto*stage:{f.snapshots[0].get(),f.retainers[0].get()}) {
  const auto c=stage->bufferContract();ASSERT_FALSE(c.outputs.empty());
  for(const auto&output:c.outputs)EXPECT_TRUE(output.prepare_write_storage)<<bufferIdName(output.id);
  EXPECT_EQ(c.writesRequiringPrepare().size(),c.outputs.size());
  for(const auto&inout:c.inouts)EXPECT_FALSE(inout.prepare_write_storage)<<bufferIdName(inout.id);
 }
}
