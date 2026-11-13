/**
 * @file Test__PooledNgramReplayTrace.cpp
 * @brief Device-free source-byte and immutable input-trace contracts for native lookup timing.
 *
 * Actual sixteen-head identities remain distinct across all declared token shapes;
 * mixed/churn inputs are fixed disjoint banks, not inferred from live cache state.
 */
#include "utils/PooledNgramReplayTrace.h"
#include "utils/PooledNgramReplaySchedule.h"
#include <gtest/gtest.h>
#include <array>
#include <set>
using namespace llaminar2::test::ngram_replay;
/** @brief Each live token has16distinct original source IDs; all bank bounds remain admitted. */
TEST(PooledNgramReplayTrace, DistinctHeadsAndBanksAcrossRealTokenShapes){
    for(const std::uint32_t tokens:{1,8,16,64,512}){
        const Shape shape{1048576,tokens,tokens};const auto a=ids(shape,Pattern::DistinctA),b=ids(shape,Pattern::DistinctB),c=ids(shape,Pattern::DistinctC);
        EXPECT_EQ(a.size(),tokens*Heads);EXPECT_EQ(uniqueCount(a),a.size());EXPECT_EQ(uniqueCount(b),b.size());EXPECT_EQ(uniqueCount(c),c.size());
        std::set<std::uint64_t> all(a.begin(),a.end());all.insert(b.begin(),b.end());all.insert(c.begin(),c.end());EXPECT_EQ(all.size(),3*a.size());
        for(std::size_t i=0;i<a.size();++i){EXPECT_LT(a[i],shape.table_rows);EXPECT_EQ(a[i]/(shape.table_rows/Heads),i%Heads);}
        const auto mixed=ids(shape,Pattern::MixedB);EXPECT_EQ(uniqueCount(mixed),mixed.size());
        std::size_t hot=0;for(std::size_t i=0;i<mixed.size();++i)hot+=mixed[i]==a[i];EXPECT_EQ(hot,mixed.size()*3/4);
        const auto duplicates=ids(shape,Pattern::DuplicateHeavy);EXPECT_EQ(uniqueCount(duplicates),Heads*std::min(tokens,2u));
        EXPECT_TRUE(ids(shape,Pattern::Empty).empty());
    }
}
/** @brief Original IQ4_NL blocks retain the existing explicit scale/codebook pattern and high source bits. */
TEST(PooledNgramReplayTrace, OriginalNativeBytesAndGeometryRejections){
    std::array<std::byte,RowBytes> row{},high{};fillRow(7,row);fillRow(0x100000007ULL,high);EXPECT_NE(row,high);
    for(std::size_t block=0;block<5;++block){EXPECT_EQ(row[block*18],std::byte{0});EXPECT_EQ(row[block*18+1],std::byte{0x3c});}
    EXPECT_EQ(row[2],std::byte{7});EXPECT_EQ(row[3],std::byte{0});
    EXPECT_THROW(fillRow(7,std::span<std::byte>{}),std::invalid_argument);
    EXPECT_THROW((void)ids({1048576,0,0},Pattern::DistinctA),std::invalid_argument);
    EXPECT_THROW((void)ids({1048576,1,2},Pattern::DistinctA),std::invalid_argument);
    EXPECT_THROW((void)ids({1048575,1,1},Pattern::DistinctA),std::invalid_argument);
    EXPECT_THROW((void)ids({1048576,513,1},Pattern::DistinctA),std::invalid_argument);
}

/** @brief Full source bits distinguish heads, banks, neighbors and high-bit aliases before byte validation. */
TEST(PooledNgramReplayTrace, NativeRowIdentityRejectsHeadBankNeighborAndHighBitSubstitution){
    const Shape shape{1048576,512,512,Locality::Contiguous};
    const auto a=ids(shape,Pattern::DistinctA),b=ids(shape,Pattern::DistinctB);
    const std::array<std::uint64_t,6> originals{a[0],a[1],b[0],a[0]+1,a[0]+(std::uint64_t{1}<<32),std::numeric_limits<std::uint64_t>::max()};
    std::vector<std::array<std::byte,RowBytes>> bytes(originals.size());
    for(std::size_t i=0;i<originals.size();++i){fillRow(originals[i],bytes[i]);
        std::uint64_t decoded=0;for(unsigned byte=0;byte<8;++byte)decoded|=std::uint64_t{std::to_integer<unsigned>(bytes[i][2+byte])}<<(8*byte);
        EXPECT_EQ(decoded,originals[i]);
        for(std::size_t j=0;j<i;++j)EXPECT_NE(bytes[i],bytes[j]);}
    // Actual old low-nibble witness could not detect this original head swap.
    EXPECT_EQ((originals[0]^(originals[0]>>32))%16,(originals[1]^(originals[1]>>32))%16);
    const auto expected=bytes[0];const auto substituted=bytes[1];EXPECT_NE(expected,substituted);
    const auto scattered=ids({1048576,512,512,Locality::Scattered},Pattern::DistinctA);
    EXPECT_NE(scattered,a);EXPECT_EQ(uniqueCount(scattered),a.size());
    for(std::size_t head=0;head<Heads;++head){EXPECT_NE(scattered[head],scattered[Heads+head]);
        EXPECT_GT(scattered[Heads+head]>scattered[head]?scattered[Heads+head]-scattered[head]:scattered[head]-scattered[Heads+head],1u);}
}

/** @brief Every actor shares exact prelude/sample banks at each A/B boundary. */
TEST(PooledNgramReplayTrace, FixedPreludeAndMeasuredTrafficSchedule){
    for(const auto scenario:{Scenario::WarmHits,Scenario::AlternatingSets,Scenario::Mixed,Scenario::Churn,Scenario::DuplicateControl,Scenario::Empty}){
        EXPECT_EQ(scheduledFrame(scenario,true,0).pattern,scenario==Scenario::WarmHits?Pattern::DistinctA:scenario==Scenario::DuplicateControl?Pattern::DuplicateHeavy:Pattern::DistinctB);
        const auto last=scheduledFrame(scenario,true,1).pattern;
        EXPECT_EQ(last,scenario==Scenario::Churn?Pattern::DistinctC:scenario==Scenario::DuplicateControl?Pattern::DuplicateHeavy:Pattern::DistinctA);
        EXPECT_THROW((void)scheduledFrame(scenario,true,2),std::invalid_argument);
    }
    for(std::uint32_t index=0;index<12;++index){
        EXPECT_EQ(scheduledFrame(Scenario::Churn,false,index).pattern,index%2?Pattern::DistinctB:Pattern::DistinctA);
        EXPECT_EQ(scheduledFrame(Scenario::AlternatingSets,false,index).pattern,index%2?Pattern::DistinctC:Pattern::DistinctB);
        EXPECT_EQ(scheduledFrame(Scenario::Mixed,false,index).pattern,index%2?Pattern::MixedC:Pattern::MixedB);
    }
    EXPECT_EQ(scheduledFrame(Scenario::Churn,false,0).phase,"hot_set_a");
    EXPECT_EQ(scheduledFrame(Scenario::Churn,false,1).phase,"hot_set_b");
    EXPECT_EQ(scheduledFrame(Scenario::Churn,false,2).phase,"hot_set_a_return");
    EXPECT_THROW((void)scheduledFrame(static_cast<Scenario>(99),false,0),std::invalid_argument);
    EXPECT_EQ(scheduledInvocation(Scenario::Churn,2,0).pattern,Pattern::DistinctB);
    EXPECT_EQ(scheduledInvocation(Scenario::Churn,2,1).pattern,Pattern::DistinctC);
    EXPECT_EQ(scheduledInvocation(Scenario::Churn,2,2).pattern,Pattern::DistinctA);
    EXPECT_EQ(scheduledInvocation(Scenario::Churn,2,3).pattern,Pattern::DistinctB);
    EXPECT_EQ(scheduledInvocation(Scenario::Churn,0,0).pattern,Pattern::DistinctA);
    EXPECT_THROW((void)scheduledInvocation(Scenario::Churn,1,0),std::invalid_argument);
}

/** @brief The actual remote schedule keeps duplicates at full capacity across reset/repeated inputs. */
TEST(PooledNgramReplayTrace,AuthoredRemoteDuplicateHeavyRetainsFullAdmittedCapacity){
    using namespace llaminar2::test::ngram_replay;
    for(const auto capacity:{1u,2u,512u})for(std::uint64_t epoch=0;epoch<20;++epoch){const Shape shape{1048576,capacity,1,Locality::Scattered};
        const auto large=authoredRemoteLifecycleIds(shape,epoch*4),small=authoredRemoteLifecycleIds(shape,epoch*4+1);
        const auto empty=authoredRemoteLifecycleIds(shape,epoch*4+2),duplicates=authoredRemoteLifecycleIds(shape,epoch*4+3);
        EXPECT_EQ(large.size(),capacity*Heads);EXPECT_EQ(uniqueCount(large),large.size());
        EXPECT_EQ(small.size(),Heads);EXPECT_EQ(uniqueCount(small),Heads);EXPECT_TRUE(empty.empty());
        EXPECT_EQ(duplicates.size(),capacity*Heads);EXPECT_EQ(uniqueCount(duplicates),Heads*std::min(capacity,2u));
        if(capacity==512){EXPECT_EQ(duplicates.size(),8192u);EXPECT_EQ(uniqueCount(duplicates),32u);}
    }
}
