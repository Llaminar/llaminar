/**
 * @file Test__MTPShiftedCacheOrigin.cpp
 * @brief Device-free exact origin, reset, prefix,4H packing and native publication regressions.
 *
 * Literal coordinate tables exercise production host/device arithmetic without
 * a model, GPU, old-core ABI linkage or a second count/storage authority.
 */
#include "execution/mtp/MTPShiftedCacheOrigin.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "models/qwen4exp/Qwen4ExpPredictorPolicy.h"
#include "kernels/common/SamplingMath.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
using namespace llaminar2;
namespace {
constexpr auto Generic=MTPShiftedCacheOrigin::SuccessorPairsOnly;
constexpr auto QFN=MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero;
}
TEST(MTPShiftedCacheOrigin, MandatoryConventionIsDerivedFromAuthenticatedMetadata) {
 const Qwen4ExpPredictorMetadata source{48,Qwen4ExpPredictorAttention::Dense,Qwen4ExpPredictorHiddenNorm::PerBranch,Qwen4ExpPredictorSource::SharedGGUF};
 ASSERT_TRUE(source.validForMainLayers(48));EXPECT_EQ(source.cacheOrigin(),QFN);
 EXPECT_EQ(mtpShiftedPrefixOffset(QFN),0);EXPECT_EQ(mtpShiftedPrefixOffset(Generic),1);EXPECT_EQ(mtpShiftedPrefixOffset(QFN,1),-1);
}
TEST(MTPShiftedCacheOrigin, EveryCommittedPrefixPreservesBothConventions) {
 for(int n=0;n<=4096;++n) {EXPECT_EQ(mtpShiftedPrefixRows(n,Generic),n?n-1:0);EXPECT_EQ(mtpShiftedPrefixRows(n,QFN),n);}
 EXPECT_EQ(mtpShiftedPrefixRows(-1,QFN),-1);EXPECT_EQ(mtpShiftedPrefixOffset(Generic,INT_MAX),-1);
}
TEST(MTPShiftedCacheOrigin, SingleTokenPromptProducesOneVisibleZero4HSeed) {
 const auto q=mtpShiftedPrefillPlan(1,0,1,QFN);ASSERT_TRUE(q.valid);EXPECT_TRUE(q.bootstrap);EXPECT_FALSE(q.prior);
 EXPECT_EQ(q.append_rows,1);EXPECT_EQ(q.token_offset,0);EXPECT_EQ(q.hiddenRow(0),-2);
 const auto old=mtpShiftedPrefillPlan(1,0,1,Generic);ASSERT_TRUE(old.valid);EXPECT_FALSE(old.bootstrap);EXPECT_EQ(old.append_rows,0);
}
TEST(MTPShiftedCacheOrigin, InitialPackingPairsZeroAndOnlyPreviousRealRows) {
 const auto q=mtpShiftedPrefillPlan(4,0,4,QFN);ASSERT_TRUE(q.valid);const std::array expected{-2,0,1,2};
 for(int row=0;row<4;++row) {EXPECT_EQ(q.hiddenRow(row),expected[row]);EXPECT_EQ(row+q.token_offset,row);}
 const auto old=mtpShiftedPrefillPlan(4,0,4,Generic);ASSERT_TRUE(old.valid);EXPECT_EQ(old.append_rows,3);EXPECT_EQ(old.hiddenRow(0),0);EXPECT_EQ(old.token_offset,1);
}
TEST(MTPShiftedCacheOrigin, SegmentsAndPartialPrefixResumeUsePriorWithoutAnotherSeed) {
 for(int prefix:{1,2,7,64})for(int suffix:{1,2,9}) {
  const auto q=mtpShiftedPrefillPlan(prefix+suffix,prefix,suffix,QFN);ASSERT_TRUE(q.valid);
  EXPECT_FALSE(q.bootstrap);EXPECT_TRUE(q.prior);EXPECT_EQ(q.append_rows,suffix);EXPECT_EQ(q.hiddenRow(0),-1);EXPECT_EQ(q.token_offset,0);
  if(suffix>1)EXPECT_EQ(q.hiddenRow(1),0);
  const auto old=mtpShiftedPrefillPlan(prefix+suffix,prefix-1,suffix,Generic);ASSERT_TRUE(old.valid);EXPECT_TRUE(old.prior);EXPECT_FALSE(old.bootstrap);EXPECT_EQ(old.append_rows,suffix);
 }
}
TEST(MTPShiftedCacheOrigin, ResidentConditionSkipsDuplicateRows) {
 const auto q=mtpShiftedPrefillPlan(8,8,1,QFN);ASSERT_TRUE(q.valid);EXPECT_FALSE(q.bootstrap);EXPECT_FALSE(q.prior);EXPECT_EQ(q.append_rows,0);
 const auto rest=mtpShiftedPrefillPlan(10,8,3,QFN);ASSERT_TRUE(rest.valid);EXPECT_EQ(rest.append_rows,2);EXPECT_EQ(rest.token_offset,1);EXPECT_EQ(rest.hiddenRow(0),0);
}
TEST(MTPShiftedCacheOrigin, ResetUsesZeroSeedAndMalformedProgressFailsClosed) {
 const auto first=mtpShiftedPrefillPlan(3,0,3,QFN),reset=mtpShiftedPrefillPlan(3,0,3,QFN);ASSERT_TRUE(first.valid);ASSERT_TRUE(reset.valid);EXPECT_TRUE(reset.bootstrap);
 for(int row=0;row<3;++row)EXPECT_EQ(first.hiddenRow(row),reset.hiddenRow(row));
 for(const auto &bad:std::array<std::array<int,3>,5>{{{3,2,3},{8,5,1},{3,0,0},{1,0,2},{3,-1,3}}})EXPECT_FALSE(mtpShiftedPrefillPlan(bad[0],bad[1],bad[2],QFN).valid);
 EXPECT_FALSE(mtpShiftedPrefillPlan(1,0,1,static_cast<MTPShiftedCacheOrigin>(99)).valid);
}
TEST(MTPShiftedCacheOrigin, BlockResiduesProjectExactLiveRanges) {
 for(int first=0;first<65;++first)for(int count=0;count<=65;++count) {
  const auto q=mtpShiftedBlockRange(first,count,QFN);EXPECT_EQ(q.first,first);EXPECT_EQ(q.count,count);EXPECT_EQ(q.end,first+count);
  const auto old=mtpShiftedBlockRange(first,count,Generic);EXPECT_EQ(old.first,first?first-1:0);EXPECT_EQ(old.end,first+count?first+count-1:0);EXPECT_EQ(old.count,old.end-old.first);
 }
 EXPECT_THROW((void)mtpShiftedBlockRange(INT_MAX,1,QFN),std::overflow_error);
 EXPECT_THROW((void)mtpShiftedBlockRange(0,1,QFN,1),std::invalid_argument);
 EXPECT_THROW((void)mtpShiftedBlockRange(-1,1,QFN),std::invalid_argument);
}
TEST(MTPShiftedCacheOrigin, DevicePublicationUsesCanonicalPrimaryCounts) {
 for(int base:{0,1,9})for(int accepted:{0,1,5}) {
  int target=-1,count=-1,ok=-1;
  sampling_math::derive_shifted_speculative_publication_metadata_from_primary(base,base+accepted,1,0,&target,&count,&ok,QFN);
  EXPECT_EQ(ok,1);EXPECT_EQ(target,base+accepted);EXPECT_EQ(count,accepted);
  sampling_math::derive_shifted_speculative_publication_metadata_from_primary(base,base+accepted,1,0,&target,&count,&ok);
  EXPECT_EQ(ok,1);EXPECT_EQ(target,base+accepted?base+accepted-1:0);EXPECT_EQ(count,target-(base?base-1:0));
 }
 int target=9,count=9,ok=9;sampling_math::derive_shifted_speculative_publication_metadata_from_primary(1,2,1,1,&target,&count,&ok,QFN);EXPECT_EQ(ok,0);EXPECT_EQ(target,0);EXPECT_EQ(count,0);
}
TEST(MTPShiftedCacheOrigin, BootstrapFitsExisting4HCapacityAndNeverReadsStaleArchive) {
 constexpr int width=4*2560,rows=4;std::vector<float> input(rows*width),archive(width,73),packed(rows*width,-91);
 for(int row=0;row<rows;++row)for(int col=0;col<width;++col)input[row*width+col]=float(row+1);
 const auto plan=mtpShiftedPrefillPlan(rows,0,rows,QFN);ASSERT_TRUE(plan.valid);
 for(int row=0;row<plan.append_rows;++row)for(int col=0;col<width;++col)packed[row*width+col]=plan.hiddenRow(row)==-2?0:input[plan.hiddenRow(row)*width+col];
 EXPECT_EQ(packed.size()*sizeof(float),size_t(rows)*40960);for(int row=0;row<rows;++row) {EXPECT_EQ(packed[row*width],float(row));EXPECT_EQ(packed[(row+1)*width-1],float(row));}EXPECT_EQ(archive.front(),73);
}

TEST(MTPShiftedCacheOrigin, ArchiveAndSnapshotIdentityRejectsCrossModelProgressBeforeMutation) {
 int mutations=0;
 auto restore=[&](MTPShiftedCacheOrigin expected,MTPShiftedCacheOrigin archived) {
  requireMTPShiftedCacheOrigin(expected,archived);
  ++mutations;
 };
 EXPECT_THROW(restore(QFN,Generic),std::invalid_argument);
 EXPECT_THROW(restore(Generic,QFN),std::invalid_argument);
 EXPECT_THROW(restore(QFN,static_cast<MTPShiftedCacheOrigin>(99)),std::invalid_argument);
 EXPECT_EQ(mutations,0);
 EXPECT_NO_THROW(restore(QFN,QFN));
 EXPECT_NO_THROW(restore(Generic,Generic));
 EXPECT_EQ(mutations,2);
}

TEST(MTPShiftedCacheOrigin, CanonicalReplayAppendsAtQFNFrontierAndReusesOnlyExistingCondition) {
 for(int main:{0,1,9,4096}) {
  const auto append=planMTPShiftedReplayRow(main,main,QFN);
  ASSERT_TRUE(append.ok);EXPECT_EQ(append.action,MTPShiftedReplayRowAction::AppendRow);EXPECT_EQ(append.append_position_offset,main);
  const auto reuse=planMTPShiftedReplayRow(main,main+1,QFN);
  ASSERT_TRUE(reuse.ok);EXPECT_EQ(reuse.action,MTPShiftedReplayRowAction::ReuseResidentRow);
  const auto old=planMTPShiftedReplayRow(main,main);ASSERT_TRUE(old.ok);EXPECT_EQ(old.action,MTPShiftedReplayRowAction::ReuseResidentRow);
  const auto snapshot=makeLogicalMTPVerifierBaseSnapshot(main,QFN);ASSERT_TRUE(snapshot.valid);
  EXPECT_EQ(snapshot.mtp_shifted_origin,QFN);ASSERT_EQ(snapshot.mtp_cached_tokens.size(),1);EXPECT_EQ(snapshot.mtp_cached_tokens.front(),main);
  if(main>0) {
   EXPECT_FALSE(planMTPShiftedReplayRow(main,main-1,QFN).ok);
   const auto generic=planMTPShiftedReplayRow(main,main-1);ASSERT_TRUE(generic.ok);EXPECT_EQ(generic.action,MTPShiftedReplayRowAction::AppendRow);
  }
 }
 EXPECT_FALSE(planMTPShiftedReplayRow(-1,0,QFN).ok);
 EXPECT_EQ(expectedShiftedMTPTokens(1,QFN),1);
 EXPECT_EQ(expectedShiftedMTPTokens(1),0);
}

TEST(MTPShiftedCacheOrigin, CPUCompactBoundaryRejectsPaddingHolesAndOverflowBeforeMutation) {
 const std::array<int,1> length{3},padded{2},single{1};
 const std::array<int,3> positions{7,8,9},hole{7,9,10};
 EXPECT_NO_THROW(requireMTPShiftedHostPrefill(QFN,7,3,1,length,positions));
 EXPECT_NO_THROW(requireMTPShiftedHostPrefill(QFN,0,1,1,single));
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,7,3,1,padded,positions),std::invalid_argument);
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,7,3,1,length,hole),std::invalid_argument);
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,7,3,2,length,positions),std::invalid_argument);
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,INT_MAX,1,1,single),std::invalid_argument);
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,-1,1,1,single),std::invalid_argument);
 EXPECT_THROW(requireMTPShiftedHostPrefill(QFN,0,3,1,length,std::span<const int>{}),std::invalid_argument);
 EXPECT_NO_THROW(requireMTPShiftedHostPrefill(Generic,7,3,2,padded,hole));
}

TEST(MTPShiftedCacheOrigin, AcceptedAppendBoundariesPreserveSeedAndResidentRows) {
 for(int position:{0,1,9,4096})for(int resident:{0,1,7}) {
  EXPECT_EQ(mtpShiftedAppendBoundaryRows(position,resident,QFN),position+resident);
  EXPECT_EQ(mtpShiftedAppendBoundaryRows(position,resident,Generic),std::max(0,position-1+resident));
 }
 EXPECT_THROW(mtpShiftedAppendBoundaryRows(INT_MAX,1,QFN),std::overflow_error);
 EXPECT_THROW(mtpShiftedAppendBoundaryRows(INT_MAX,1,Generic),std::overflow_error);
 EXPECT_THROW(mtpShiftedAppendBoundaryRows(1,-1,QFN),std::invalid_argument);
 EXPECT_THROW(mtpShiftedAppendBoundaryRows(-1,0,QFN),std::invalid_argument);
}
TEST(MTPShiftedCacheOrigin, PrebuiltProgressSelectorUsesOriginalCountOwnerAndNeverInventsSeedOrPrior) {
 for(int base:{0,1,9})for(int rows:{1,4,16}) {
  EXPECT_EQ(mtpShiftedInputHiddenRow(base+rows,base,rows,QFN),-1);
  for(int row=0;row<rows;++row) {
   EXPECT_EQ(mtpShiftedInputHiddenRow(base+rows,base+row+1,rows,QFN),row);
   EXPECT_EQ(mtpShiftedInputHiddenRow(base+rows,base+row,rows,Generic),row);
  }
  EXPECT_EQ(mtpShiftedInputHiddenRow(base+rows,base+rows+1,rows,QFN),-1);
 }
 EXPECT_EQ(mtpShiftedInputHiddenRow(1,0,2,QFN),-1);
 EXPECT_EQ(mtpShiftedInputHiddenRow(1,-1,1,QFN),-1);
}

/** @test A direct emit appends at its saved model frontier after fresh/full/partial restore. */
TEST(MTPShiftedCacheOrigin, DirectEmitAnchorPreservesBothCheckpointOrigins) {
 for(const auto origin:{Generic,QFN})for(int rows:{0,1,9,11,64,2048,65536,INT_MAX-1}) {
  SCOPED_TRACE(rows);
  const int position=mtpShiftedAppendPosition(rows,origin);
  EXPECT_EQ(position,origin==QFN?rows:rows+1);
  EXPECT_EQ(mtpShiftedAppendBoundaryRows(position,0,origin),rows);
 }
 // Literal negative control for the real CPU Depth1 failure: eleven resident
 // QFN rows cannot satisfy the old unconditional position-twelve precondition.
 EXPECT_EQ(mtpShiftedAppendPosition(11,QFN),11);
 EXPECT_EQ(mtpShiftedAppendBoundaryRows(12,0,QFN),12);
 EXPECT_EQ(mtpShiftedAppendPosition(11,Generic),12);
 EXPECT_EQ(mtpShiftedAppendPosition(INT_MAX,QFN),INT_MAX);
 EXPECT_THROW(mtpShiftedAppendPosition(INT_MAX,Generic),std::overflow_error);
 EXPECT_THROW(mtpShiftedAppendPosition(-1,QFN),std::invalid_argument);
 EXPECT_THROW(mtpShiftedAppendPosition(11,static_cast<MTPShiftedCacheOrigin>(99)),std::invalid_argument);
}
