/**
 * @file Test__Qwen4ExpPredictorWorkspaceRows.cpp
 * @brief Device-free exact4M/shifted-prefix BOM arithmetic and unchanged ordinary/main controls.
 *
 * The tested resolver is called by all three production projection workspace
 * estimators and the Dense0 FC composer. It returns typed row inputs only;
 * PhysicalMemoryAuthority remains the sole allocation/admission ledger.
 */
#include "models/qwen4exp/Qwen4ExpPredictorProjectionGeometry.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace {
/** @return Exact source-authenticated separate Dense0 policy. */
Qwen4ExpPredictorMetadata policy() {return {48,Qwen4ExpPredictorAttention::Dense,Qwen4ExpPredictorHiddenNorm::PerBranch,Qwen4ExpPredictorSource::SharedGGUF};}
/** @return Invoke the exact production role-aware resolver for one source48 matrix. */
int rows(WeightRole role,std::string_view name,int ordinary,int shifted) {
    return Qwen4ExpPredictorProjectionGeometry::workspaceRows(ordinary,shifted,role,name,48,48,4,policy());
}
}
TEST(Qwen4ExpPredictorWorkspaceRows, FCAdmitsFourPhysicalRowsForM1AndLargeShiftedPrefill)
{
    for(const int m : {1,4,16,512,1024,8192})
        EXPECT_EQ(rows(WeightRole::PredictorProjection,"blk.48.nextn.eh_proj.weight",1,m),m*4);
    EXPECT_EQ(rows(WeightRole::PredictorProjection,"blk.48.nextn.eh_proj.weight",16,1),64);
}
TEST(Qwen4ExpPredictorWorkspaceRows, OnlyShiftedPrefillConsumersOwnThatEnvelope)
{
    EXPECT_EQ(rows(WeightRole::AttentionK,"blk.48.attn_k.weight",16,1024),1024);
    EXPECT_EQ(rows(WeightRole::AttentionV,"blk.48.attn_v.weight",16,1024),1024);
    EXPECT_EQ(rows(WeightRole::HyperConnectionProjection,"blk.48.hc_attn_down.weight",16,1024),1024);
    EXPECT_EQ(rows(WeightRole::HyperConnectionProjection,"blk.48.hc_attn_up.weight",16,1024),1024);
    for(const auto [role,name] : {std::pair{WeightRole::AttentionQ,"blk.48.attn_q.weight"},
        {WeightRole::AttentionWO,"blk.48.attn_output.weight"},
        {WeightRole::HyperConnectionProjection,"blk.48.hc_attn_inject.weight"},
        {WeightRole::HyperConnectionProjection,"blk.48.hc_ffn_down.weight"},
        {WeightRole::HyperConnectionProjection,"blk.48.nextn.hc_head_up.weight"},
        {WeightRole::SharedExpertDown,"blk.48.ffn_down_shexp.weight"}})
        EXPECT_EQ(rows(role,name,16,1024),16)<<name;
}
TEST(Qwen4ExpPredictorWorkspaceRows, MainAndOrdinaryFamiliesKeepTheirExactExistingRows)
{
    for(const int m : {1,16,1024}) {
        for(const auto role : {WeightRole::AttentionQ,WeightRole::AttentionK,WeightRole::PredictorProjection,
                               WeightRole::HyperConnectionProjection,WeightRole::MoEExpertDown}) {
            EXPECT_EQ(Qwen4ExpPredictorProjectionGeometry::workspaceRows(m,8192,role,"blk.24.weight",24,48,4,policy()),m);
            EXPECT_EQ(Qwen4ExpPredictorProjectionGeometry::workspaceRows(m,8192,role,"blk.48.weight",48,48,std::nullopt,std::nullopt),m);
        }
    }
}
TEST(Qwen4ExpPredictorWorkspaceRows, MissingOrUnknownQFNLayoutFailsBeforeBOMInputs)
{
    const WorkspaceMemoryGeometry unbound{.device=DeviceId::cpu()};
    EXPECT_FALSE(unbound.mtp_dense_layout);
    EXPECT_THROW((void)Qwen4ExpPredictorProjectionGeometry::requireLayout(unbound.mtp_dense_layout),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorProjectionGeometry::requireLayout(static_cast<MTPShiftedKVHeadLayout>(99)),std::invalid_argument);
    for(const auto layout : {MTPShiftedKVHeadLayout::PrimaryTensorParallelShard,MTPShiftedKVHeadLayout::FullModelPerParticipant}) {
        const WorkspaceMemoryGeometry declared{.device=DeviceId::cpu(),.mtp_target_query_rows=16,.mtp_dense_layout=layout};
        EXPECT_EQ(Qwen4ExpPredictorProjectionGeometry::requireLayout(declared.mtp_dense_layout),layout);
    }
}
TEST(Qwen4ExpPredictorWorkspaceRows, RejectsWrongSourceMissingPolicyAndNativeOverflow)
{
    using G=Qwen4ExpPredictorProjectionGeometry;
    EXPECT_THROW((void)G::workspaceRows(1,1024,WeightRole::PredictorProjection,"blk.48.nextn.eh_proj.weight",48,48,4,std::nullopt),std::invalid_argument);
    EXPECT_THROW((void)G::workspaceRows(1,1024,WeightRole::PredictorProjection,"blk.49.nextn.eh_proj.weight",49,48,4,policy()),std::invalid_argument);
    EXPECT_THROW((void)G::workspaceRows(1,1024,WeightRole::PredictorProjection,"blk.48.nextn.eh_proj.weight",48,48,3,policy()),std::invalid_argument);
    EXPECT_THROW((void)G::workspaceRows(1,1024,WeightRole::PredictorProjection,"blk.48.nextn.eh_proj.weight",48,48,std::nullopt,policy()),std::invalid_argument);
    EXPECT_THROW((void)G::perBranchRows(std::numeric_limits<int>::max()/4+1,4),std::overflow_error);
    EXPECT_EQ(G::perBranchRows(std::numeric_limits<int>::max()/4,4),(std::numeric_limits<int>::max()/4)*4);
    EXPECT_THROW((void)G::perBranchRows(0,4),std::invalid_argument);
    EXPECT_THROW((void)G::perBranchRows(1,0),std::invalid_argument);
}
