/**
 * @file Test__CPUShardRouteCollective.cpp
 * @brief Adversarial actual compact shard-record reduction, before original top-k weighting.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "utils/CPUFeatures.h"
#include <cstring>
#include <omp.h>
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


namespace {
/** @brief Restore the caller's existing team width after a bounded host replay. */
struct FirstPairTeamScope {
    int saved=omp_get_max_threads();
    explicit FirstPairTeamScope(int threads) { omp_set_num_threads(threads); }
    ~FirstPairTeamScope() { omp_set_num_threads(saved); }
};
}

/**
 * @test The production combine retains the original copy-then-add bytes while
 * removing its redundant first scratch copy. The oracle deliberately retains
 * that former sequence and uses the same ISA-dispatched vec_add helper.
 *
 * Dense and compact records cover live prefixes, inactive slots, vector tails,
 * participant cancellation, signed zeros, infinities and NaN payloads. Reusing
 * poisoned output across large/small/empty calls proves storage capacity is not
 * a write extent. Existing-team entry exercises the actual collective workshare.
 * This is numerical/lifecycle evidence, not a timing or NUMA certificate.
 */
TEST(CPUShardRouteCollective, FirstPairDirectAddMatchesOriginalBytesAndLiveExtents) {
    ::testing::Test::RecordProperty("runtime_isa",isaLevelName(activeISALevel()));
    constexpr int capacity=7,top_k=3;
    constexpr float poison=-913.25f;
    constexpr std::array<uint32_t,16> patterns{
        0x00000000u,0x80000000u,0x3f800000u,0xbf800000u,
        0x7f800000u,0xff800000u,0x7fc12345u,0xffc54321u,
        0x7f812345u,0x00000001u,0x80000001u,0x00800000u,
        0x60ad78ecu,0xe0ad78ecu,0x33800000u,0x3f800001u};
    for(int threads:{1,3}) {
        FirstPairTeamScope team(threads);
        for(size_t degree:{2u,4u,8u}) for(int columns:{1,2,15,16,17,31,32,33,63,64,65,513,2560}) {
            SCOPED_TRACE(::testing::Message()<<"workers="<<threads<<" degree="<<degree<<" columns="<<columns);
            const size_t capacity_slots=capacity*top_k;
            std::vector<std::vector<float>> dense(degree,std::vector<float>(capacity_slots*columns));
            for(size_t p=0;p<degree;++p) for(size_t i=0;i<dense[p].size();++i)
                dense[p][i]=std::bit_cast<float>(patterns[(i+p*5u)%patterns.size()]);
            std::vector<float> weights(capacity_slots);
            for(size_t slot=0;slot<capacity_slots;++slot) weights[slot]=slot%4u==1u?0.0f:1.0f;
            for(bool packed:{false,true}) {
              std::vector<float> actual(capacity_slots*columns,poison);
              size_t replay=0;
              for(int rows:{7,2,0,1,5,1}) {
                SCOPED_TRACE(::testing::Message()<<"packed="<<packed<<" rows="<<rows);
                const size_t slots=static_cast<size_t>(rows)*top_k;
                // Change the next source bytes while retaining the previous
                // destination. A capacity-wide write would corrupt dirty tails.
                ++replay;
                for(size_t p=0;p<degree;++p) for(size_t i=0;i<dense[p].size();++i)
                    dense[p][i]=std::bit_cast<float>(patterns[(i+p*5u+replay)%patterns.size()]);
                std::vector<std::vector<float>> records(degree);
                std::vector<std::vector<int32_t>> maps(degree,std::vector<int32_t>(slots,-1));
                std::vector<CanonicalRouteDownPartialRows> banks(degree);
                for(size_t p=0;p<degree;++p) {
                    if(!packed || slots==0) { banks[p]={dense[p]};continue; }
                    // Each participant may pack in a different arrival order.
                    for(size_t ordinal=0;ordinal<slots;++ordinal) {
                        const size_t slot=p%2u?slots-1u-ordinal:ordinal;
                        if(weights[slot]==0.0f)continue;
                        maps[p][slot]=static_cast<int32_t>(records[p].size()/(columns+1));
                        const size_t first=records[p].size();
                        records[p].resize(first+columns+1);
                        std::copy_n(dense[p].data()+slot*columns,columns,records[p].data()+first);
                        ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(records[p].data()+first,columns,slot));
                    }
                    banks[p]={records[p],maps[p]};
                }
                std::vector<float> expected=actual;
                for(size_t slot=0;slot<slots;++slot) {
                    if(weights[slot]==0.0f)continue;
                    float *out=expected.data()+slot*columns;
                    std::copy_n(dense[0].data()+slot*columns,columns,out);
                    for(size_t p=1;p<degree;++p)
                        primitives::vec_add(out,out,dense[p].data()+slot*columns,columns);
                }
                auto args=CanonicalRouteDownPartialCombineArgs{banks,actual,weights.data(),rows,top_k,columns};
                ASSERT_TRUE(combineCanonicalRouteDownPartialsOrdered(args));
                ASSERT_EQ(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)),0);
                if(threads==3 && rows==capacity && columns==65) {
                    std::fill(actual.begin(),actual.end(),poison);
                    std::array<int,3> accepted{};
#pragma omp parallel num_threads(3)
                    { accepted[static_cast<size_t>(omp_get_thread_num())]=combineCanonicalRouteDownPartialsOrdered(args); }
                    for(int ok:accepted)ASSERT_EQ(ok,1);
                    ASSERT_EQ(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)),0);
                }
                if(rows==capacity) {
                    // Validation must precede every write, including the first pair.
                    const auto before=actual;
                    args.ordered_participants=std::span(banks).first(1);
                    EXPECT_FALSE(combineCanonicalRouteDownPartialsOrdered(args));
                    EXPECT_EQ(std::memcmp(before.data(),actual.data(),before.size()*sizeof(float)),0);
                    args.ordered_participants=banks;
                    const auto saved=banks[1];banks[1]=banks[0];
                    EXPECT_FALSE(combineCanonicalRouteDownPartialsOrdered(args));
                    EXPECT_EQ(std::memcmp(before.data(),actual.data(),before.size()*sizeof(float)),0);
                    banks[1]=saved;
                    if(!packed) {
                        args.combined_route_rows={dense[0].data(),dense[0].size()};
                        const auto input_before=dense[0];
                        EXPECT_FALSE(combineCanonicalRouteDownPartialsOrdered(args));
                        EXPECT_EQ(std::memcmp(input_before.data(),dense[0].data(),input_before.size()*sizeof(float)),0);
                    }
                }
              }
            }
        }
    }
}
