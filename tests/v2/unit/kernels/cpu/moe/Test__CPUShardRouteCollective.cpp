/**
 * @file Test__CPUShardRouteCollective.cpp
 * @brief Adversarial actual compact shard-record reduction, before original top-k weighting.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
using namespace llaminar2;
using namespace llaminar2::cpu::moe;
namespace {
struct Replay {
    static constexpr int H=7,K=3,M=3;
    std::array<float,M*K> indices{64,1,511,64,511,2,1,2,64};
    std::array<float,M*K> weights{.25f,.5f,.75f,.5f,.25f,.75f,.25f,.5f,0};
    std::array<int,2> experts{64,511};
    std::array<RootedFloatRecordRange,2> ranges;
    std::vector<float> records;
    std::array<int32_t,2*M*K> maps{};
    std::array<float,M*K> required{};
    std::array<float,M*K*H> combined;
    explicit Replay(bool reverse=false) {
        combined.fill(-991);
        // Root may arrive before or after the other member; source identity remains explicit.
        for(int block=0;block<2;++block) {
            const int participant=reverse?1-block:block;
            const size_t first=records.size()/(H+1);
            for(int slot:{0,2,3,4}) {
                const auto begin=records.size();records.resize(begin+H+1);
                for(int h=0;h<H;++h)records[begin+h]=float(participant*13+slot*3+h)*.125f;
                EXPECT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(records.data()+begin,H,slot));
            }
            ranges[participant]={participant,first,4};
        }
    }
    CanonicalShardRouteRecordArgs args() {
        return {records,ranges,indices,weights,experts,maps,required,combined,512,M,K,H};
    }
};
TEST(CPUShardRouteCollective, ArrivalOrderPartialAndRepeatedReset) {
    Replay a,b(true);
    for(int iteration=0;iteration<20;++iteration) {
        a.combined.fill(-991);b.combined.fill(-991);
        ASSERT_TRUE(combineCanonicalShardRouteRecords(a.args()));
        ASSERT_TRUE(combineCanonicalShardRouteRecords(b.args()));
        EXPECT_EQ(a.combined,b.combined);
        for(int slot=0;slot<Replay::M*Replay::K;++slot)for(int h=0;h<Replay::H;++h) {
            const bool live=slot==0 || slot==2 || slot==3 || slot==4;
            const float expected=live?float(slot*3+h)*.125f+float(13+slot*3+h)*.125f:-991;
            EXPECT_EQ(a.combined[slot*Replay::H+h],expected);
        }
    }
}
TEST(CPUShardRouteCollective, AllCohortWidthsUseSourceOrder) {
    for(size_t degree=2;degree<=8;++degree) {
        std::vector<float> records;
        std::vector<RootedFloatRecordRange> ranges(degree);
        float expected=0,arrival=0;
        const auto value=[](size_t participant) {
            return participant==0?1e20f:participant==1?-1e20f:float(participant+5);
        };
        for(size_t p=0;p<degree;++p)expected=p?expected+value(p):value(p);
        for(size_t block=0;block<degree;++block) {
            const size_t p=degree-1-block;
            const auto first=records.size()/2;
            records.push_back(value(p));records.push_back(std::bit_cast<float>(uint32_t{0}));
            ranges[p]={int(p),first,1};arrival=block?arrival+value(p):value(p);
        }
        const std::array<float,1> indices{64},weights{.5f};
        const std::array<int,1> selected{64};
        std::vector<int32_t> maps(degree);
        std::array<float,1> required{},output{-991};
        ASSERT_TRUE(combineCanonicalShardRouteRecords({records,ranges,indices,weights,selected,
            maps,required,output,512,1,1,1}));
        EXPECT_EQ(std::bit_cast<uint32_t>(output[0]),std::bit_cast<uint32_t>(expected));
        if(degree>=3)EXPECT_NE(output[0],arrival);
    }
}
TEST(CPUShardRouteCollective, InvalidCompletePrefixNeverPublishes) {
    for(int mutation=0;mutation<9;++mutation) {
        Replay a;const auto before=a.combined;
        switch(mutation) {
        case 0:a.ranges[1].participant=0;break;
        case 1:a.ranges[1].first_record=0;break;
        case 2:a.ranges[1].record_count=3;break;
        case 3:canonical_moe_route_record::writeFlatRouteSlot(a.records.data()+Replay::H+1,Replay::H,0);break;
        case 4:canonical_moe_route_record::writeFlatRouteSlot(a.records.data(),Replay::H,1);break;
        case 5:canonical_moe_route_record::writeFlatRouteSlot(a.records.data(),Replay::H,99);break;
        case 6:a.indices[0]=64.5f;break;
        case 7:a.experts={511,64};break;
        case 8:a.ranges[0].participant=-1;break;
        }
        EXPECT_FALSE(combineCanonicalShardRouteRecords(a.args()))<<mutation;
        EXPECT_EQ(a.combined,before)<<mutation;
    }
}
TEST(CPUShardRouteCollective, EmptyAndLargeToSmallPreserveLiveExtent) {
    Replay a;
    ASSERT_TRUE(combineCanonicalShardRouteRecords(a.args()));
    // Every source must explicitly contribute zero records when there is no selected route.
    a.records.clear();a.ranges={RootedFloatRecordRange{0,0,0},RootedFloatRecordRange{1,0,0}};
    a.weights.fill(0);a.combined.fill(117);
    ASSERT_TRUE(combineCanonicalShardRouteRecords(a.args()));
    EXPECT_TRUE(std::all_of(a.combined.begin(),a.combined.end(),[](float x){return x==117;}));
    auto empty=a.args();empty.rows=0;empty.route_indices={};empty.route_weights={};
    ASSERT_TRUE(combineCanonicalShardRouteRecords(empty));
    a.ranges[1].participant=-1;
    EXPECT_FALSE(combineCanonicalShardRouteRecords(a.args()));
}
TEST(CPUShardRouteCollective, FixedParticipantSumPrecedesOriginalWeighting) {
    std::array<float,2> weights{.5f,.75f};
    std::array<float,2> p0{1e20f,3},p1{-1e20f,5},p2{7,11};
    std::array<CanonicalRouteDownPartialRows,3> participants{{{p0},{p1},{p2}}};
    std::array<float,2> combined{};std::array<float,1> output{};
    ASSERT_TRUE(reduceCanonicalRouteDownPartialsOrderedFMA({participants,combined,weights.data(),output.data(),1,2,1}));
    EXPECT_EQ(combined[0],7);EXPECT_EQ(combined[1],19);
    float expected=0;expected=std::fma(weights[0],combined[0],expected);expected=std::fma(weights[1],combined[1],expected);
    EXPECT_EQ(output[0],expected);
}
} // namespace

/** @test Original consumed sparse entries produce the same fixed-order rows without synthetic router state. */
TEST(CPUShardRouteCollective, OriginalSparseRoutesMatchDenseAndRejectPartialIdentity) {
    const std::array<int32_t,4> slots{0,2,3,4},experts{64,511,64,511};
    const std::array<float,4> weights{.25f,.75f,.5f,.25f};
    Replay dense,sparse(true);
    ASSERT_TRUE(combineCanonicalShardRouteRecords(dense.args()));
    for(int repeat=0;repeat<20;++repeat) {
        sparse.combined.fill(-991);
        auto args=sparse.args();args.route_indices={};args.route_weights={};
        args.sparse_routes=CanonicalShardSparseRouteView{slots,experts,weights};
        ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
        EXPECT_EQ(sparse.combined,dense.combined);
        for(int mutation=0;mutation<6;++mutation) {
            auto badSlots=slots,badExperts=experts;auto badWeights=weights;
            if(mutation==0)badSlots[1]=badSlots[0];
            if(mutation==1)badSlots[1]=99;
            if(mutation==2)badExperts[1]=1;
            if(mutation==3)badWeights[1]=0;
            args.sparse_routes=CanonicalShardSparseRouteView{badSlots,badExperts,badWeights};
            if(mutation==4)args.sparse_routes->original_slots=args.sparse_routes->original_slots.first(3);
            if(mutation==5)args.route_indices=dense.indices;
            sparse.combined.fill(-991);
            EXPECT_FALSE(combineCanonicalShardRouteRecords(args))<<mutation;
            EXPECT_TRUE(std::all_of(sparse.combined.begin(),sparse.combined.end(),[](float v){return v==-991;}));
        }
    }
}

#include "CPUCompactGatherEconomyContract.inc"
