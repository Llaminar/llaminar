/**
 * @file Test__CPUShardRouteCollective.cpp
 * @brief Adversarial actual compact shard-record reduction, before original top-k weighting.
 *
 * Raw nonfinite payloads exercise arithmetic byte preservation with publication
 * diagnostics explicitly scoped off. Separate cases retain finite-output
 * rejection and restore the caller's validation policy after every lifetime.
 */
#include <gtest/gtest.h>
#include <limits>
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "utils/CPUFeatures.h"
#include "utils/DebugEnv.h"
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
/** @brief Scope only the publication diagnostic for authored nonfinite payloads. */
class ScopedRouteFiniteValidation final {
public:
    /** @brief Save the caller's policy and select the diagnostic under test. */
    explicit ScopedRouteFiniteValidation(bool enabled)
        : previous_(mutableDebugEnv().validation.fail_on_nan) {
        mutableDebugEnv().validation.fail_on_nan=enabled;
    }
    /** @brief Restore the caller even when a fatal GTest assertion returns early. */
    ~ScopedRouteFiniteValidation() {mutableDebugEnv().validation.fail_on_nan=previous_;}
    ScopedRouteFiniteValidation(const ScopedRouteFiniteValidation&)=delete;
    ScopedRouteFiniteValidation& operator=(const ScopedRouteFiniteValidation&)=delete;
private:
    bool previous_;
};
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

/** @test Diagnostic selection cannot change finite arithmetic or admit an invalid route weight. */
TEST(CPUShardRouteCollective, FusedFinalAddPreweightFiniteValidationIsExplicit) {
    std::array<int32_t,3> slots{0,1,2};
    std::array<float,3> weights{.5f,.25f,-.5f};
    std::array<float,6> source{2,4,-6,8,-10,12},output{};
    const std::array<float,6> expected{1,2,-1.5f,2,5,-6};
    MoEOverlaySparseRows input;
    input.row_capacity=input.live_row_count=1;
    input.entry_capacity=input.live_entry_count=slots.size();
    input.d_model=2;input.top_k=3;
    input.original_route_slots_host=slots.data();
    input.compact_route_slots_host=slots.data();input.route_weights_host=weights.data();
    for(const bool diagnose:{false,true}) {
        const ScopedRouteFiniteValidation policy(diagnose);
        source[2]=-6.0f;output.fill(-991.0f);
        ASSERT_TRUE(preweightMoEOverlayCanonicalRouteRows(input,source,
            MoEOverlayCanonicalRouteSourceLayout::originalSlots(),output,
            MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot));
        EXPECT_EQ(output,expected);
        source[2]=std::bit_cast<float>(uint32_t(0x7fc12345u));
        EXPECT_EQ(preweightMoEOverlayCanonicalRouteRows(input,source,
            MoEOverlayCanonicalRouteSourceLayout::originalSlots(),output,
            MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot),!diagnose);
        EXPECT_TRUE(std::isnan(output[2]));
    }
    const ScopedRouteFiniteValidation arithmetic_only(false);
    weights[0]=std::bit_cast<float>(uint32_t(0x7fc12345u));
    output.fill(-993.0f);const auto untouched=output;
    EXPECT_FALSE(preweightMoEOverlayCanonicalRouteRows(input,source,
        MoEOverlayCanonicalRouteSourceLayout::originalSlots(),output,
        MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot));
    EXPECT_EQ(output,untouched);
}

/**
 * @test Authored compact shard records preserve both original outputs when the
 * worker preweights each completed final-add tile. This is numerical/lifecycle contract
 * coverage, not a recorded route trace, NUMA or captured-consumer certificate.
 * The deliberate NaN/Inf corpus uses the diagnostic nonfinite-preserving policy;
 * the independent policy test below proves normal publication still rejects it.
 */
TEST(CPUShardRouteCollective, FusedFinalAddPreservesRawAndWeightedBytes) {
    // The source corpus intentionally includes NaN/Inf. Compare its exact
    // arithmetic bytes independently of the optional publication diagnostic.
    const ScopedRouteFiniteValidation arithmetic_only(false);
    const int saved=omp_get_max_threads();
    struct Restore {int value;~Restore(){omp_set_num_threads(value);}} restore{saved};
    // Same source bit corpus as the first-pair control, now reaching the NEW
    // fused workshare and its scalar tails. Explicit participant-varying
    // NaN lanes make operand reversal visible even in a one-column route.
    constexpr std::array<uint32_t,16> patterns{
        0x00000000u,0x80000000u,0x3f800000u,0xbf800000u,
        0x7f800000u,0xff800000u,0x7fc12345u,0xffc54321u,
        0x7f812345u,0x00000001u,0x80000001u,0x00800000u,
        0x60ad78ecu,0xe0ad78ecu,0x33800000u,0x3f800001u};
    // Every remainder position matters: compiler vectorization can choose
    // different NaN operands even when the source expression is identical.
    std::vector<int> widths;
    for(int columns=1;columns<=65;++columns)widths.push_back(columns);
    widths.insert(widths.end(),{127,128,129,2560});
    for(const int threads:{1,3})for(const int columns:widths)
    for(size_t degree=2;degree<=8;++degree) {
        SCOPED_TRACE("workers="+std::to_string(threads)+" columns="+std::to_string(columns)+
            " degree="+std::to_string(degree));
        omp_set_num_threads(threads);
        constexpr int rows=3,top_k=3,slots=rows*top_k;
        std::array<int32_t,5> original_slots{0,2,3,5,8},compact_slots{0,1,3,5,7};
        const std::array<int32_t,5> experts{1,1,1,1,1};
        const std::array<int,1> selected{1};
        std::array<float,5> weights{.3f,-.75f,.5f,0x1.000002p-1f,-0x1p-120f};
        std::vector<float> records;
        std::vector<RootedFloatRecordRange> ranges(degree);
        for(size_t arrival=0;arrival<degree;++arrival) {
            const size_t participant=degree-1-arrival,first=records.size()/size_t(columns+1);
            for(const auto slot:original_slots) {
                const auto begin=records.size();records.resize(begin+size_t(columns+1));
                for(int column=0;column<columns;++column) {
                    const size_t lane=(size_t(column)+size_t(slot)*3u)%patterns.size();
                    uint32_t bits=patterns[(lane+participant*5u)%patterns.size()];
                    const uint32_t sign=participant%2u?0x80000000u:0u;
                    if(lane==6)bits=sign|uint32_t(0x7fc12345u+participant); // Distinct quiet NaNs.
                    if(lane==8)bits=sign|uint32_t(0x7f812345u+participant); // Distinct signaling NaNs.
                    if(lane==9)bits=uint32_t(1u+participant); // Actual positive subnormals.
                    if(lane==10)bits=0x80000000u|uint32_t(1u+participant); // Negative subnormals.
                    float value=std::bit_cast<float>(bits);
                    if(lane==0)value=participant==0?1e20f:participant==1?-1e20f:float(participant+3);
                    if(lane==1)value=participant==0?-0.0f:0.0f;
                    records[begin+size_t(column)]=value;
                }
                ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(records.data()+begin,columns,size_t(slot)));
            }
            ranges[participant]={int(participant),first,original_slots.size()};
        }
        std::vector<int32_t> maps(degree*slots);
        std::vector<float> required(slots),raw(size_t(slots)*columns,-991.0f),weighted(raw.size(),-993.0f);
        CanonicalShardRouteRecordArgs args{records,ranges,{},{},selected,maps,required,raw,512,rows,top_k,columns,
            CanonicalShardSparseRouteView{original_slots,experts,weights}};
        MoEOverlaySparseRows input;
        input.row_capacity=rows;input.entry_capacity=slots;input.live_row_count=rows;
        input.live_entry_count=original_slots.size();input.d_model=columns;input.top_k=top_k;
        input.original_route_slots_host=original_slots.data();input.compact_route_slots_host=compact_slots.data();
        input.route_weights_host=weights.data();
        ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
        ASSERT_TRUE(preweightMoEOverlayCanonicalRouteRows(input,raw,
            MoEOverlayCanonicalRouteSourceLayout::originalSlots(),weighted,
            MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot));
        const auto expected_raw=raw,expected_weighted=weighted;
        for(int replay=0;replay<3;++replay) {
            std::fill(raw.begin(),raw.end(),-991.0f);std::fill(weighted.begin(),weighted.end(),-993.0f);
            args.preweighted_output=CanonicalRoutePreweightedOutput{weighted,original_slots,compact_slots,weights};
            ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
            EXPECT_EQ(std::memcmp(raw.data(),expected_raw.data(),raw.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(weighted.data(),expected_weighted.data(),weighted.size()*sizeof(float)),0);
        }
        // An existing-team call consumes already-validated immutable maps. The
        // record validator itself remains a single caller, matching serving.
        std::array<CanonicalRouteDownPartialRows,8> banks;
        for(size_t p=0;p<degree;++p)banks[p]={
            std::span<const float>(records).subspan(ranges[p].first_record*size_t(columns+1),
                ranges[p].record_count*size_t(columns+1)),std::span<const int32_t>(maps).subspan(p*slots,slots)};
        std::fill(raw.begin(),raw.end(),-991.0f);std::fill(weighted.begin(),weighted.end(),-993.0f);
        int failures=0;
#pragma omp parallel reduction(+:failures)
        {
            if(!combineCanonicalRouteDownPartialsOrdered({std::span(banks).first(degree),raw,
                required.data(),rows,top_k,columns,args.preweighted_output}))++failures;
        }
        EXPECT_EQ(failures,0);
        EXPECT_EQ(std::memcmp(raw.data(),expected_raw.data(),raw.size()*sizeof(float)),0);
        EXPECT_EQ(std::memcmp(weighted.data(),expected_weighted.data(),weighted.size()*sizeof(float)),0);
    }
}

/** @test Nonfinite payload rejection is a validation policy, independent of workers or scalar width. */
TEST(CPUShardRouteCollective, FusedFinalAddPreweightHonorsNonFiniteValidationPolicy) {
    const int saved=omp_get_max_threads();
    struct Restore {int value;~Restore(){omp_set_num_threads(value);}} restore{saved};
    const bool original_policy=debugEnv().validation.fail_on_nan;
    constexpr size_t live=9,capacity=10;
    std::array<int32_t,live> slots{};
    std::iota(slots.begin(),slots.end(),0);
    std::array<float,live> weights;weights.fill(.5f);
    MoEOverlaySparseRows input;
    input.row_capacity=capacity;input.entry_capacity=capacity;input.d_model=1;input.top_k=1;
    input.original_route_slots_host=slots.data();input.compact_route_slots_host=slots.data();
    input.route_weights_host=weights.data();
    for(const int threads:{1,3}) {
        omp_set_num_threads(threads);
        for(const bool reject_nonfinite:{false,true}) {
            const ScopedRouteFiniteValidation policy(reject_nonfinite);
            input.live_row_count=live;input.live_entry_count=live;
            std::array<float,capacity> source,destination;
            source.fill(4.0f);destination.fill(-993.0f);
            const auto preweight=[&] {
                return preweightMoEOverlayCanonicalRouteRows(input,source,
                    MoEOverlayCanonicalRouteSourceLayout::originalSlots(),destination,
                    MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot);
            };
            ASSERT_TRUE(preweight()); // Finite one-column work is valid under either policy.
            EXPECT_EQ(destination[0],2.0f);
            EXPECT_EQ(destination.back(),-993.0f);
            for(const uint32_t bits:{0x7fc12345u,0x7f812345u,0x7f800000u,0xff800000u}) {
                SCOPED_TRACE(::testing::Message()<<threads<<'/'<<reject_nonfinite<<'/'<<bits);
                source[live-1]=std::bit_cast<float>(bits);
                const auto original_source=source;
                destination.fill(-993.0f);
                EXPECT_EQ(preweight(),!reject_nonfinite);
                EXPECT_EQ(destination[0],2.0f);
                EXPECT_FALSE(std::isfinite(destination[live-1]));
                EXPECT_EQ(destination.back(),-993.0f);
                EXPECT_EQ(std::memcmp(source.data(),original_source.data(),sizeof(source)),0);
            }
            input.live_row_count=0;input.live_entry_count=0;
            destination.fill(-993.0f);
            ASSERT_TRUE(preweight()); // Inactive nonfinite storage is not published live data.
            for(const auto value:destination)EXPECT_EQ(value,-993.0f);
        }
        EXPECT_EQ(debugEnv().validation.fail_on_nan,original_policy);
    }
}

/** @test Malformed weighted sinks fail before either payload changes; empty/reset work preserves holes. */
TEST(CPUShardRouteCollective, FusedFinalAddRejectsIncompleteSinksAndResetsLiveExtent) {
    Replay a;
    const std::array<int32_t,4> slots{0,2,3,4},destinations{0,1,3,6},experts{64,511,64,511};
    const std::array<float,4> weights{.25f,.75f,.5f,.25f};
    std::array<float,Replay::M*Replay::K*Replay::H> weighted;
    auto args=a.args();
    for(const bool sparse_input:{false,true}) {
      args=a.args();
      if(sparse_input) {
        args.route_indices={};args.route_weights={};
        args.sparse_routes=CanonicalShardSparseRouteView{slots,experts,weights};
      }
      for(int mutation=0;mutation<14;++mutation) {
        auto original=slots,compact=destinations;auto factors=weights;
        a.combined.fill(-991);weighted.fill(-993);
        auto sink=CanonicalRoutePreweightedOutput{weighted,original,compact,factors};
        if(mutation==0)sink.values=sink.values.first(3);
        if(mutation==1)sink.values=a.combined;
        if(mutation==2)sink.values=std::span(a.records).first(weighted.size());
        if(mutation==3)sink.original_slots=sink.original_slots.first(3);
        if(mutation==4)sink.weights=sink.weights.first(3);
        if(mutation==5)original[1]=original[0];
        if(mutation==6)original[3]=99;
        if(mutation==7)compact[1]=compact[0];
        if(mutation==8)compact[3]=99;
        if(mutation==9)factors[2]=NAN;
        if(mutation==10)factors[2]=0.0f;
        if(mutation==11)sink.destination_slots=sink.destination_slots.first(3);
        if(mutation==12)original[2]=1; // Does not name a required original route.
        if(mutation==13)factors[2]=.375f; // Finite but NOT the original router multiplier.
        args.preweighted_output=sink;
        EXPECT_FALSE(combineCanonicalShardRouteRecords(args))<<mutation;
        EXPECT_TRUE(std::all_of(a.combined.begin(),a.combined.end(),[](float x){return x==-991;}));
        EXPECT_TRUE(std::all_of(weighted.begin(),weighted.end(),[](float x){return x==-993;}));
      }
      // Both original dense and sparse router views admit the exact factors.
      args.preweighted_output=CanonicalRoutePreweightedOutput{weighted,slots,destinations,weights};
      ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
    }
    args.preweighted_output=CanonicalRoutePreweightedOutput{weighted,slots,destinations,weights};
    ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
    // Reuse the same capacity for one live record and then empty work. Neither
    // operation may clear or communicate the now-inactive payload capacity.
    for(size_t p=0;p<2;++p)a.ranges[p].record_count=1;
    args.sparse_routes=CanonicalShardSparseRouteView{std::span(slots).first(1),std::span(experts).first(1),std::span(weights).first(1)};
    args.preweighted_output=CanonicalRoutePreweightedOutput{weighted,std::span(slots).first(1),std::span(destinations).first(1),std::span(weights).first(1)};
    a.combined.fill(-991);weighted.fill(-993);
    // The ready record bank retains its original ranges; a one-record prefix
    // is not a new gather and must still form a complete nonoverlapping range.
    a.records.resize(2*(Replay::H+1));
    for(size_t p=0;p<2;++p) {
        a.ranges[p]={int(p),p,1};
        ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(a.records.data()+p*(Replay::H+1),Replay::H,0));
    }
    args.records=a.records;args.source_ranges=a.ranges;
    ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
    EXPECT_TRUE(std::all_of(a.combined.begin()+Replay::H,a.combined.end(),[](float x){return x==-991;}));
    EXPECT_TRUE(std::all_of(weighted.begin()+Replay::H,weighted.end(),[](float x){return x==-993;}));
    a.records.clear();a.ranges={RootedFloatRecordRange{0,0,0},RootedFloatRecordRange{1,0,0}};
    args.records={};args.source_ranges=a.ranges;args.sparse_routes=CanonicalShardSparseRouteView{};
    args.preweighted_output=CanonicalRoutePreweightedOutput{weighted,{},{},{}};
    const auto prior_raw=a.combined,prior_weighted=weighted;
    ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
    EXPECT_EQ(a.combined,prior_raw);EXPECT_EQ(weighted,prior_weighted);
}

/** @test An unarmed/foreign-epoch publication cannot touch either root payload. */
TEST(CPUShardRouteCollective, FusedTicketRejectsUnarmedLeaseBeforePayloadWrites) {
    Replay a;MoEOverlaySparseRows input;
    MoEOverlayCanonicalRouteReturnTicketStorage ticket;
    MoEOverlayCanonicalRouteReturnTicketStorage::Publication unarmed;
    EXPECT_FALSE(unarmed.authorizes(ticket,1));
    EXPECT_FALSE(combineAndPublishMoEOverlayCanonicalShardRouteRows(ticket,unarmed,input,a.args()));
    EXPECT_TRUE(std::all_of(a.combined.begin(),a.combined.end(),[](float x){return x==-991;}));
}


/** @test Original CPU host canonical return copies raw live rows and preserves one later weighting authority. */
TEST(CPUShardRouteCollective, CPUHostCanonicalReturnKeepsOriginalSlotsUnweightedAndCompact)
{
    constexpr std::size_t H=7,slots=9,capacity=5;
    std::array<float,slots*H+1> source{};
    std::array<float,capacity*H+1> returned;
    std::array<int32_t,capacity+1> ids;
    std::array<int32_t,3> original{0,2,6},compact{0,1,4};
    std::array<float,3> weights{.25F,-.5F,.75F};
    MoEOverlaySparseRows input;
    input.d_model=H;input.top_k=3;input.row_capacity=3;input.entry_capacity=3;
    input.original_route_slots_host=original.data();input.compact_route_slots_host=compact.data();
    input.route_weights_host=weights.data();input.source_participant=0;input.target_participant=2;
    MoEOverlayReturnRows output;
    output.d_model=H;output.layout=MoEOverlayReturnLayout::CanonicalExpertRoutes;
    output.row_capacity=capacity;output.row_ids_host=ids.data();output.output_rows_fp32=returned.data();
    for(int replay=0;replay<20;++replay) {
        const std::size_t count=std::array<std::size_t,5>{3,1,0,2,3}[replay%5];
        input.live_entry_count=count;input.live_row_count=count?3:0;input.residency_epoch=101+replay;
        input.key=makeMoEOverlayCollectiveKey(9,replay,3,0,0,2,MoEOverlayCollectiveDirection::Dispatch);
        for(std::size_t i=0;i<source.size();++i) source[i]=float(replay*97+i)*.125F;
        returned.fill(-719.F);ids.fill(-83);output.live_row_count=capacity;
        const auto original_source=source;
        ASSERT_TRUE(copyMoEOverlayCanonicalRouteRowsToHostReturn(input,
            std::span<const float>(source).first(slots*H),output));
        EXPECT_EQ(output.key,input.key);EXPECT_EQ(output.residency_epoch,input.residency_epoch);
        EXPECT_EQ(output.live_row_count,count);EXPECT_EQ(output.source_participant,input.target_participant);
        EXPECT_EQ(output.target_participant,input.source_participant);EXPECT_EQ(source,original_source);
        for(std::size_t entry=0;entry<count;++entry) {
            EXPECT_EQ(ids[entry],original[entry]);
            EXPECT_EQ(std::memcmp(returned.data()+entry*H,source.data()+std::size_t(original[entry])*H,H*sizeof(float)),0);
        }
        for(std::size_t i=count*H;i<returned.size();++i) EXPECT_EQ(returned[i],-719.F);
        for(std::size_t i=count;i<ids.size();++i) EXPECT_EQ(ids[i],-83);
    }
    input.live_row_count=3;input.live_entry_count=3;
    const auto untouched=returned;const auto untouched_ids=ids;
    original[1]=original[0];
    EXPECT_FALSE(copyMoEOverlayCanonicalRouteRowsToHostReturn(input,std::span<const float>(source).first(slots*H),output));
    EXPECT_EQ(returned,untouched);EXPECT_EQ(ids,untouched_ids);original[1]=2;
    weights[1]=std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(copyMoEOverlayCanonicalRouteRowsToHostReturn(input,std::span<const float>(source).first(slots*H),output));
    EXPECT_EQ(returned,untouched);weights[1]=-.5F;
    output.output_rows_fp32=source.data();
    EXPECT_FALSE(copyMoEOverlayCanonicalRouteRowsToHostReturn(input,std::span<const float>(source).first(slots*H),output));
    output.output_rows_fp32=returned.data();output.row_capacity=2;
    EXPECT_FALSE(copyMoEOverlayCanonicalRouteRowsToHostReturn(input,std::span<const float>(source).first(slots*H),output));
    EXPECT_EQ(returned,untouched);
}

/** @test Dynamic original cohort populations keep the identical fixed participant arithmetic order beyond eight. */
TEST(CPUShardRouteCollective, VariableOriginalCohortsKeepCanonicalAdditionOrderWithoutParticipantCap)
{
    for(int population=2;population<=29;++population) {
        SCOPED_TRACE(population);
        constexpr int H=7,slots=4;
        const auto width=canonical_moe_route_record::recordWidth(H);
        std::vector<float> records(std::size_t(population)*2*width);
        std::vector<RootedFloatRecordRange> ranges;
        std::array<float,slots*H+1> raw{},expected{};
        std::array<float,slots*H+1> weighted{},expected_weighted{};
        raw.fill(-733.F);expected.fill(-733.F);weighted.fill(-735.F);expected_weighted.fill(-735.F);
        std::array<int32_t,2> original_slots{0,3},destinations{1,3},experts{0,0};
        std::array<float,2> weights{.5F,-.75F};
        for(int p=0;p<population;++p) {
            for(int r=0;r<2;++r) {
                // Packed record arrival order differs from immutable route
                // order. Large cancelling terms expose addition reassociation.
                const int slot=original_slots[1-r];
                auto* record=records.data()+(std::size_t(p)*2+r)*width;
                for(int h=0;h<H;++h) {
                    const float value=p==0?1.0e20F:p==1?-1.0e20F:
                        float((p%3)-1)*float((slot+1)*(h+1))*.125F;
                    record[h]=value;
                    expected[slot*H+h]=p==0?value:expected[slot*H+h]+value;
                }
                ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(record,H,slot));
            }
            ranges.push_back({p,std::size_t(p)*2,2});
        }
        for(size_t entry=0;entry<original_slots.size();++entry)
            for(int h=0;h<H;++h)
                expected_weighted[destinations[entry]*H+h]=expected[original_slots[entry]*H+h]*weights[entry];
        std::vector<int32_t> maps(size_t(population)*slots);std::array<float,slots> required{};
        std::array<int,1> selected{0};
        CanonicalShardRouteRecordArgs args{records,ranges,{}, {},selected,maps,required,
            std::span<float>(raw).first(slots*H),1,2,2,H,
            CanonicalShardSparseRouteView{original_slots,experts,weights}};
        // Both publication shapes share the exact original common arithmetic.
        for(int replay=0;replay<4;++replay) {
            raw.fill(-733.F);weighted.fill(-735.F);
            args.preweighted_output=replay%2?std::optional{CanonicalRoutePreweightedOutput{
                std::span<float>(weighted).first(slots*H),original_slots,destinations,weights}}:std::nullopt;
            ASSERT_TRUE(combineCanonicalShardRouteRecords(args));
            EXPECT_EQ(std::memcmp(raw.data(),expected.data(),sizeof(raw)),0);
            if(args.preweighted_output)EXPECT_EQ(std::memcmp(weighted.data(),expected_weighted.data(),sizeof(weighted)),0);
            else for(float value:weighted)EXPECT_EQ(value,-735.F);
        }
        // A malformed last original member must fail before either live output
        // changes, even when all preceding participants are complete.
        auto* final=records.data()+(std::size_t(population)*2-1)*width;
        ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(final,H,3));
        const auto before_raw=raw,before_weighted=weighted;
        EXPECT_FALSE(combineCanonicalShardRouteRecords(args));
        EXPECT_EQ(raw,before_raw);EXPECT_EQ(weighted,before_weighted);
        ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(final,H,0));
        ASSERT_TRUE(combineCanonicalShardRouteRecords(args)); // Restore the validated caller-owned maps.
        // The descriptor API remains a valid first-class caller of the SAME
        // reducer, including variable cohorts and its empty-row contract.
        std::vector<CanonicalRouteDownPartialRows> banks;
        for(int p=0;p<population;++p)banks.push_back({
            std::span<const float>(records).subspan(size_t(p)*2*width,2*width),
            std::span<const int32_t>(maps).subspan(size_t(p)*slots,slots)});
        raw.fill(-733.F);
        ASSERT_TRUE(combineCanonicalRouteDownPartialsOrdered({banks,std::span<float>(raw).first(slots*H),
            required.data(),2,2,H}));
        EXPECT_EQ(std::memcmp(raw.data(),expected.data(),sizeof(raw)),0);
        EXPECT_TRUE(reduceCanonicalRouteDownPartialsOrderedFMA({banks,{},nullptr,nullptr,0,2,H}));
    }
}

/** @test Indexed original source boundaries cannot be overwritten by either numerical output. */
TEST(CPUShardRouteCollective, IndexedOriginalRangeMetadataCannotAliasEitherOutput)
{
    std::array<float,4> records{1.F,0.F,2.F,0.F};
    ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(records.data(),1,0));
    ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(records.data()+2,1,0));
    std::array<RootedFloatRecordRange,2> ranges{{{0,0,1},{1,1,1}}};
    std::array<int32_t,2> maps{0,0};
    std::array<float,1> required{1.F};
    std::array<float,2> raw{-711.F,-713.F},weighted{-715.F,-717.F};
    std::array<int32_t,1> original{0},destination{0};std::array<float,1> factor{.5F};
    const CanonicalRouteDownIndexedPartialRows indexed{records,ranges,maps};
    CanonicalRouteDownPartialCombineArgs args{{},std::span<float>(raw).first(1),required.data(),1,1,1,
        CanonicalRoutePreweightedOutput{std::span<float>(weighted).first(1),original,destination,factor}};
    ASSERT_TRUE(combineCanonicalRouteDownPartialsOrderedIndexed(args,indexed));
    EXPECT_EQ(raw,(std::array<float,2>{3.F,-713.F}));
    EXPECT_EQ(weighted,(std::array<float,2>{1.5F,-717.F}));
    std::array<std::byte,sizeof(ranges)> before{};
    std::memcpy(before.data(),ranges.data(),sizeof(ranges));
    // These are borrowed byte-address aliases only. The validator must reject
    // before dereferencing/storing through the incompatible floating view.
    auto* metadata=reinterpret_cast<float*>(&ranges[0].first_record);
    args.combined_route_rows={metadata,1};
    const auto saved_weighted=weighted;
    EXPECT_FALSE(combineCanonicalRouteDownPartialsOrderedIndexed(args,indexed));
    EXPECT_EQ(std::memcmp(ranges.data(),before.data(),sizeof(ranges)),0);
    EXPECT_EQ(weighted,saved_weighted);
    args.combined_route_rows=std::span<float>(raw).first(1);
    args.preweighted_output->values={metadata,1};
    const auto saved_raw=raw;
    EXPECT_FALSE(combineCanonicalRouteDownPartialsOrderedIndexed(args,indexed));
    EXPECT_EQ(std::memcmp(ranges.data(),before.data(),sizeof(ranges)),0);
    EXPECT_EQ(raw,saved_raw);
    // A second-source range alias rejects too; checking only the first
    // descriptor would leave later tile reads exposed to output mutation.
    args.preweighted_output.reset();
    args.combined_route_rows={reinterpret_cast<float*>(&ranges[1].record_count),1};
    EXPECT_FALSE(combineCanonicalRouteDownPartialsOrderedIndexed(args,indexed));
    EXPECT_EQ(std::memcmp(ranges.data(),before.data(),sizeof(ranges)),0);
}
