/**
 * @file Test__PredictorProjectionLayout.cpp
 * @brief Device-free full/sharded/replicated-KV geometry and collective policy negatives.
 *
 * Pure validation returns arithmetic decisions, not a frozen construction
 * capability. Only fromFrozen can issue a predictor layout consumable by graph
 * bindings; these fixtures never create native weight/storage lookalikes.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorProjectionLayout.h"
#include <gtest/gtest.h>
#include <array>
#include <type_traits>
using namespace llaminar2;
namespace {
struct Fixture {
    Qwen4ExpPredictorParticipantGeometry p{DeviceId::cuda(0),MTPShiftedKVHeadLayout::FullModelPerParticipant,
        2,0,true,true,2560,256,24,2,12,1,0,512,0};
    std::array<size_t,2> q{12288,2560},k{512,2560},v{k},wo{2560,6144};
    std::array<size_t,2> gate{512,2560},up{gate},down{2560,512};
    bool wo_shard=false,shared_shard=false;
    Qwen4ExpPredictorProjectionDecision validate() const {
        return Qwen4ExpPredictorProjectionLayout::validate(p.canonical_layout,p,
            {q,k,v,wo,gate,up,down,wo_shard,shared_shard});
    }
    void shard(bool replicated_kv=false) {
        p.canonical_layout=MTPShiftedKVHeadLayout::PrimaryTensorParallelShard;
        p.participant=1;p.query_head_start=12;p.kv_head_start=replicated_kv?0:1;p.local_kv_heads=replicated_kv?2:1;
        q={6144,2560};k={static_cast<size_t>(p.local_kv_heads)*256,2560};v=k;wo={2560,3072};
        gate={256,2560};up=gate;down={2560,256};wo_shard=shared_shard=true;
    }
};
}
TEST(PredictorProjectionLayout, FullModelReplicaNeverReducesDenseOrSharedOutputs) {
    static_assert(!std::is_default_constructible_v<Qwen4ExpPredictorProjectionLayout>);
    static_assert(!std::is_aggregate_v<Qwen4ExpPredictorProjectionLayout>);
    const Fixture f;const auto d=f.validate();
    EXPECT_EQ(d.query_heads,24);EXPECT_EQ(d.kv_heads,2);EXPECT_EQ(d.query_start,0);EXPECT_EQ(d.kv_start,0);
    EXPECT_FALSE(d.attention_reduction);EXPECT_FALSE(d.shared_reduction);
}
TEST(PredictorProjectionLayout, ExactTPShardsRequireTheirRealPartialsToBeReconstructed) {
    Fixture f;f.shard();const auto d=f.validate();
    EXPECT_EQ(d.query_heads,12);EXPECT_EQ(d.kv_heads,1);EXPECT_EQ(d.query_start,12);EXPECT_EQ(d.kv_start,1);
    EXPECT_TRUE(d.attention_reduction);EXPECT_TRUE(d.shared_reduction);
}
TEST(PredictorProjectionLayout, ReplicatedKVHeadsRemainValidWithShardedQueries) {
    Fixture f;f.shard(true);const auto d=f.validate();
    EXPECT_EQ(d.query_heads,12);EXPECT_EQ(d.kv_heads,2);EXPECT_EQ(d.kv_start,0);EXPECT_TRUE(d.attention_reduction);
}
TEST(PredictorProjectionLayout, RejectsMissingFusedGateAndEveryWrongQKVWoShape) {
    for(int field=0;field<4;++field) {
        Fixture f;if(field==0)f.q[0]/=2;if(field==1)++f.k[0];if(field==2)++f.v[1];if(field==3)++f.wo[1];
        EXPECT_THROW((void)f.validate(),std::invalid_argument)<<field;
    }
}
TEST(PredictorProjectionLayout, RejectsForeignPolicyAndParticipantOrGQARange) {
    Fixture f;EXPECT_THROW((void)Qwen4ExpPredictorProjectionLayout::validate(MTPShiftedKVHeadLayout::PrimaryTensorParallelShard,f.p,
        {f.q,f.k,f.v,f.wo,f.gate,f.up,f.down,false,false}),std::invalid_argument);
    for(int field=0;field<5;++field) {
        Fixture x;x.shard();if(field==0)x.p.participant=2;if(field==1)x.p.local_query_heads=-1;
        if(field==2)x.p.query_head_start=13;if(field==3)x.p.kv_head_start=0;if(field==4)x.p.global_kv_heads=5;
        EXPECT_THROW((void)x.validate(),std::invalid_argument)<<field;
    }
}
TEST(PredictorProjectionLayout, RejectsReplicaWithSlicesAndShardsWithoutRealRowParallelTags) {
    for(int field=0;field<4;++field) {
        Fixture f;if(field<2)f.shard();if(field==0)f.wo_shard=false;if(field==1)f.shared_shard=false;
        if(field==2)f.wo_shard=true;if(field==3)f.shared_shard=true;
        EXPECT_THROW((void)f.validate(),std::invalid_argument)<<field;
    }
    Fixture f;f.shard();f.p.canonical_layout=MTPShiftedKVHeadLayout::FullModelPerParticipant;
    EXPECT_THROW((void)f.validate(),std::invalid_argument);
}
TEST(PredictorProjectionLayout, RejectsPartialSharedReplicaAndMismatchedGateUpDown) {
    for(int field=0;field<4;++field) {
        Fixture f;if(field==0){f.gate[0]=256;f.up=f.gate;f.down[1]=256;f.shared_shard=true;}
        if(field==1)++f.up[0];if(field==2)++f.down[1];if(field==3)f.gate[0]=0;
        EXPECT_THROW((void)f.validate(),std::invalid_argument)<<field;
    }
}
TEST(PredictorProjectionLayout, SingleParticipantPrimaryRetainsFullHeadsAndZeroReduction) {
    Fixture f;f.p.canonical_layout=MTPShiftedKVHeadLayout::PrimaryTensorParallelShard;f.p.participants=1;
    const auto d=f.validate();EXPECT_EQ(d.query_heads,24);EXPECT_EQ(d.kv_heads,2);EXPECT_FALSE(d.attention_reduction);EXPECT_FALSE(d.shared_reduction);
}
