/**
 * @file Test__ProjectionRowWork.cpp
 * @brief Device-free proof of original request intervals and independent work scopes.
 *
 * Host tensors below model only small native publication words. No model,
 * accelerator, native kernel or profiler is required. Original-row coverage,
 * publication ownership and arithmetic/work independence are tested directly.
 */
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/common/FloatingProjectionRows.h"
#include "kernels/qwen4exp/QwenSparseQueryRowWork.h"
#include "tensors/TensorKernels.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2 {
TEST(ProjectionRowWork, EveryRowAcrossThreeIndependentStridedRequests) {
    std::array<std::array<int,3>,3> header{};
    for(int capacity=1;capacity<=16;++capacity) {
        const auto work=ProjectionRowWork::causalStrided(3,capacity,&header[0][0],&header[0][1],3,8192,2051);
        for(int firstPrefix:{0,2047,2048,2049,2050,2051,2052})for(int firstLive=0;firstLive<=capacity;++firstLive) {
            header={{{firstPrefix,firstLive,0},{4096,capacity,0},{2048,capacity/2,77}}};
            for(int request=0;request<3;++request) {
                const auto span=work.resolveHost(request);
                ASSERT_TRUE(span.valid());
                for(int row=0;row<3*capacity;++row) {
                    const int owner=row/capacity,local=row%capacity;
                    EXPECT_EQ(span.contains(row),owner==request&&local<header[request][1]&&
                        std::int64_t(header[request][0])+local>=2051);
                }
            }
        }
    }
}
TEST(ProjectionRowWork, ArbitraryOwnersUseAnImmutableAddressTableWithoutCopiedCounts) {
    std::array<int,3> first{512,0,0},second{2048,6,0},third{4096,2,0};
    const std::array<ProjectionRequestPublication,3> addresses{{{&first[0],&first[1]},
        {&second[0],&second[1]},{&third[0],&third[1]}}};
    const auto work=ProjectionRowWork::causalIndirect(3,8,addresses.data(),8192,2051,91);
    const auto identity=work.captureIdentity();
    EXPECT_EQ(work.resolveHost(0).size(),0);
    EXPECT_EQ(work.resolveHost(1).first,11);EXPECT_EQ(work.resolveHost(1).end,14);
    EXPECT_EQ(work.resolveHost(2).first,16);EXPECT_EQ(work.resolveHost(2).end,18);
    second[0]=4096;second[1]=1;
    EXPECT_EQ(work.resolveHost(1).first,8);EXPECT_EQ(work.resolveHost(1).end,9);
    EXPECT_EQ(work.captureIdentity(),identity);
}
TEST(ProjectionRowWork, ObservationRetainsOriginalWordsAndAuthenticatesInheritedCountOnce) {
    std::array<std::array<int,3>,2> header{{{2048,6,73},{4096,2,0}}};
    std::array<int,2> inherited{6,2};
    const auto work=ProjectionRowWork::causalStrided(2,8,&header[0][0],&header[0][1],3,8192,2051)
        .withInheritedRequests(DeviceRequestRowRanges::deviceCounted(2,8,inherited.data()));
    const auto observed=work.observeHost(0);
    EXPECT_EQ(observed.frontier,2048);EXPECT_EQ(observed.live,6);
    EXPECT_EQ(observed.span.first,3);EXPECT_EQ(observed.span.end,6);
    header[0][0]=512;header[0][1]=1;inherited[0]=1;
    EXPECT_EQ(observed.frontier,2048);EXPECT_EQ(observed.live,6);
    EXPECT_EQ(observed.span.first,3);EXPECT_EQ(observed.span.end,6);
    const auto next=work.observeHost(0);
    EXPECT_EQ(next.frontier,512);EXPECT_EQ(next.live,1);EXPECT_EQ(next.span.size(),0);
    inherited[0]=2;
    EXPECT_THROW((void)work.observeHost(0),std::invalid_argument);
    const auto full=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(4)).observeHost();
    EXPECT_EQ(full.frontier,0);EXPECT_EQ(full.live,4);EXPECT_EQ(full.span.size(),4);
}
TEST(ProjectionRowWork, SliceRepeatAndDirtyReplayRetainOriginalRequestCoordinates) {
    std::array<std::array<int,3>,3> header{{{4096,8,0},{2048,6,0},{0,0,0}}};
    const auto original=ProjectionRowWork::causalStrided(3,8,&header[0][0],&header[0][1],3,8192,2051);
    const auto tile=original.repeatEachRow(4).slice(36,24);
    const auto sameTile=original.slice(9,6).repeatEachRow(4);
    EXPECT_EQ(tile.captureIdentity(),sameTile.captureIdentity());
    EXPECT_EQ(tile.physicalRows(),24);EXPECT_EQ(tile.rowsPerRequest(),32);
    EXPECT_EQ(tile.originalRow(0),36);EXPECT_EQ(tile.requestForRow(0),1);
    EXPECT_EQ(tile.resolveHost(1).first,8);EXPECT_EQ(tile.resolveHost(1).end,20);
    const auto identity=tile.captureIdentity();
    for(const auto values:{std::array{4096,8},std::array{512,1},std::array{512,0},std::array{2048,6}}) {
        header[1][0]=values[0];header[1][1]=values[1];
        const auto span=tile.resolveHost(1);
        for(int row=0;row<tile.physicalRows();++row) {
            const int originalRow=tile.originalRow(row),request=originalRow/32,logical=(originalRow%32)/4;
            EXPECT_EQ(span.contains(row),request==1&&logical<header[1][1]&&
                std::int64_t(header[1][0])+logical>=2051);
        }
        EXPECT_EQ(tile.captureIdentity(),identity);
    }
}
TEST(ProjectionRowWork, InheritedCountsAreCheckedRatherThanSilentlyChosen) {
    std::array<std::array<int,3>,2> header{{{2048,6,0},{4096,2,0}}};
    std::array<int,2> inherited{6,2};
    const auto work=ProjectionRowWork::causalStrided(2,8,&header[0][0],&header[0][1],3,8192,2051)
        .withInheritedRequests(DeviceRequestRowRanges::deviceCounted(2,8,inherited.data()));
    EXPECT_EQ(work.resolveHost(0).size(),3);EXPECT_EQ(work.resolveHost(1).size(),2);
    inherited[1]=1;
    EXPECT_THROW((void)work.resolveHost(1),std::invalid_argument);
    EXPECT_EQ(work.resolveHost(0).size(),3);
    EXPECT_THROW((void)work.withInheritedRequests(DeviceRequestRowRanges::fullyActiveRequests(3,8)),std::invalid_argument);
    const auto single=ProjectionRowWork::causalStrided(1,8,&header[0][0],&header[0][1],3,8192,2051)
        .withInheritedPrefix(DeviceRowRange::deviceCounted(8,inherited.data()));
    EXPECT_EQ(single.resolveHost().size(),3);
    inherited[0]=0;
    EXPECT_THROW((void)single.resolveHost(),std::invalid_argument);
}
TEST(ProjectionRowWork, CountFreeFlattenedPrefixSuppliesOnlyExactPhysicalGeometry) {
    std::array<std::array<int,3>,3> replay{{{4096,0,0},{2048,6,0},{4096,2,0}}};
    int flattenedCount=24;
    const auto work=ProjectionRowWork::causalStrided(3,8,&replay[0][0],&replay[0][1],3,8192,2051);
    const auto inherited=work.withInheritedPrefix(DeviceRowRange::fullyActive(24));
    EXPECT_EQ(inherited.captureIdentity(),work.captureIdentity());
    EXPECT_EQ(inherited.requests(),3);EXPECT_EQ(inherited.resolveHost(1).first,11);
    EXPECT_THROW((void)work.withInheritedPrefix(DeviceRowRange::deviceCounted(24,&flattenedCount)),std::invalid_argument);
    EXPECT_THROW((void)work.withInheritedPrefix(DeviceRowRange::fullyActive(23)),std::invalid_argument);
    const auto slice=work.slice(6,12);
    EXPECT_NO_THROW((void)slice.withInheritedPrefix(DeviceRowRange::fullyActive(24).slice(6,12)));
    EXPECT_THROW((void)slice.withInheritedPrefix(DeviceRowRange::fullyActive(12)),std::invalid_argument);
    EXPECT_THROW((void)slice.withInheritedPrefix(DeviceRowRange::fullyActive(24).slice(5,12)),std::invalid_argument);
    const auto repeated=work.repeatEachRow(2);
    EXPECT_NO_THROW((void)repeated.withInheritedPrefix(DeviceRowRange::fullyActive(24).repeatEachRow(2)));
    EXPECT_THROW((void)repeated.withInheritedPrefix(DeviceRowRange::fullyActive(48)),std::invalid_argument);
}
TEST(ProjectionRowWork, InvalidMetadataNeverTurnsIntoAnEmptySafeLookingInterval) {
    std::array<int,3> header{2048,6,0};
    const auto work=ProjectionRowWork::causalStrided(1,8,&header[0],&header[1],3,4096,2051);
    EXPECT_FALSE(work.resolveFor(-1,0,0).valid());EXPECT_FALSE(work.resolveFor(1,0,0).valid());
    EXPECT_FALSE(work.resolveFor(0,-1,0).valid());EXPECT_FALSE(work.resolveFor(0,9,0).valid());
    EXPECT_FALSE(work.resolveFor(0,0,-1).valid());EXPECT_FALSE(work.resolveFor(0,1,4096).valid());
    EXPECT_FALSE(work.resolveFor(0,8,std::numeric_limits<int>::max()).valid());
    EXPECT_TRUE(work.resolveFor(0,0,4096).valid());
    header[1]=9;
    EXPECT_THROW((void)work.resolveHost(),std::invalid_argument);
    EXPECT_THROW((void)work.slice(8,1),std::out_of_range);
    EXPECT_THROW((void)work.repeatEachRow(0),std::invalid_argument);
}
TEST(ProjectionRowWork, FullAndInheritedPrefixFactoriesKeepExistingMeanings) {
    int count=6;
    const auto work=ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(8,&count).slice(2,4));
    EXPECT_EQ(work.resolveHost().first,0);EXPECT_EQ(work.resolveHost().end,4);
    count=3;EXPECT_EQ(work.resolveHost().end,1);
    std::array<int,3> lengths{0,3,2};
    const auto requests=ProjectionRowWork::fromRequests(DeviceRequestRowRanges::deviceCounted(3,4,lengths.data()));
    EXPECT_EQ(requests.resolveHost(0).size(),0);EXPECT_EQ(requests.resolveHost(1).first,4);
    EXPECT_EQ(requests.resolveHost(1).end,7);EXPECT_EQ(requests.resolveHost(2).end,10);
    const auto full=ProjectionRowWork::fromRequests(DeviceRequestRowRanges::fullyActiveRequests(3,4));
    EXPECT_EQ(full.resolveHost(2).first,8);EXPECT_EQ(full.resolveHost(2).end,12);
}
TEST(ProjectionRowWork, FactoriesNeverReadOpaqueNativePublicationWords) {
    const auto *base=reinterpret_cast<const std::int32_t *>(std::uintptr_t{4096});
    const auto work=ProjectionRowWork::causalStrided(3,8,base,base+1,388,8192,2051);
    EXPECT_EQ(work.resolveFor(1,6,2048).first,11);
    const auto *table=reinterpret_cast<const ProjectionRequestPublication *>(std::uintptr_t{8192});
    const auto indirect=ProjectionRowWork::causalIndirect(3,8,table,8192,2051,99);
    EXPECT_EQ(indirect.resolveFor(2,2,4096).end,18);
    EXPECT_THROW((void)ProjectionRowWork::causalIndirect(3,8,table,8192,2051,0),std::invalid_argument);
    EXPECT_NE(indirect.captureIdentity(),ProjectionRowWork::causalIndirect(3,8,table,8192,2051,100).captureIdentity());
    EXPECT_NE(indirect.captureIdentity(),ProjectionRowWork::causalIndirect(3,8,table,8192,2052,99).captureIdentity());
}
TEST(ProjectionRowWork, RecorderScopeNeverEnablesOrDisablesVerifierArithmetic) {
    EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
    EXPECT_EQ(currentProjectionRowWork(8),nullptr);
    const auto full=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(8));
    {
        ITensorGemm::ProjectionWorkScope work(full);
        EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
        EXPECT_EQ(currentProjectionRowWork(8)->captureIdentity(),full.captureIdentity());
        EXPECT_THROW((void)currentProjectionRowWork(7),std::logic_error);
        {
            ITensorGemm::VerifierKernelModeScope arithmetic;
            EXPECT_TRUE(ITensorGemm::VerifierKernelModeScope::active());
            const auto smaller=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(4));
            EXPECT_THROW(([&]{ITensorGemm::ProjectionWorkScope nested(smaller);throw std::runtime_error("unwind");}()),std::runtime_error);
            EXPECT_EQ(currentProjectionRowWork(8)->captureIdentity(),full.captureIdentity());
            EXPECT_TRUE(ITensorGemm::VerifierKernelModeScope::active());
        }
        EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
    }
    EXPECT_EQ(currentProjectionRowWork(8),nullptr);
}
TEST(ProjectionRowWork, RecorderMetadataIsThreadLocal) {
    const auto full=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(8));
    ITensorGemm::ProjectionWorkScope work(full);
    bool childEmpty=false;
    std::thread child([&]{childEmpty=currentProjectionRowWork(8)==nullptr;});child.join();
    EXPECT_TRUE(childEmpty);EXPECT_NE(currentProjectionRowWork(8),nullptr);
}
TEST(FloatingProjectionRowWork, ExistingSinglePrefixRetainsOriginalAdmission) {
    int live=5;
    const auto prefix=DeviceRowRange::deviceCounted(8,&live);
    const auto selected=FloatingProjectionRows::bind(8,&prefix,nullptr);
    selected.validateHostPublications();
    EXPECT_EQ(selected.hostTileSpan(0,4).first,0);EXPECT_EQ(selected.hostTileSpan(0,4).end,4);
    EXPECT_EQ(selected.hostTileSpan(1,4).first,4);EXPECT_EQ(selected.hostTileSpan(1,4).end,5);
    for(int row=0;row<8;++row)EXPECT_EQ(selected.hostRowIsActive(row),row<live);
}
TEST(FloatingProjectionRowWork, RequiredSuffixBeginningInsideTileIsNeverLost) {
    std::array<int,3> replay{2048,6,0};
    const auto work=ProjectionRowWork::causalStrided(1,8,&replay[0],&replay[1],3,4096,2051);
    ITensorGemm::ProjectionWorkScope scope(work);
    const auto selected=FloatingProjectionRows::bind(8,nullptr,nullptr);
    selected.validateHostPublications();
    EXPECT_EQ(selected.hostTileSpan(0,4).first,3);EXPECT_EQ(selected.hostTileSpan(0,4).end,4);
    EXPECT_EQ(selected.hostTileFirst(0,4),3);EXPECT_EQ(selected.hostTileRows(0,4),1);
    EXPECT_EQ(selected.hostTileSpan(1,4).first,4);EXPECT_EQ(selected.hostTileSpan(1,4).end,6);
    for(int row=0;row<8;++row)EXPECT_EQ(selected.hostRowIsActive(row),row>=3&&row<6);
    EXPECT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
}
TEST(FloatingProjectionRowWork, EmptyRequestCannotSuppressLaterRequiredSuffixes) {
    std::array<std::array<int,3>,3> replay{{{4096,0,0},{2048,6,0},{4096,2,0}}};
    std::array<int,3> inherited{0,6,2};
    const auto work=ProjectionRowWork::causalStrided(3,8,&replay[0][0],&replay[0][1],3,8192,2051);
    const auto requests=DeviceRequestRowRanges::deviceCounted(3,8,inherited.data());
    ITensorGemm::ProjectionWorkScope scope(work);
    const auto selected=FloatingProjectionRows::bind(24,nullptr,&requests);
    selected.validateHostPublications();
    EXPECT_EQ(selected.hostTileRows(0,4),0);EXPECT_EQ(selected.hostTileRows(1,4),0);
    EXPECT_EQ(selected.hostTileSpan(2,4).first,11);EXPECT_EQ(selected.hostTileSpan(2,4).end,12);
    EXPECT_EQ(selected.hostTileSpan(3,4).first,12);EXPECT_EQ(selected.hostTileSpan(3,4).end,14);
    EXPECT_EQ(selected.hostTileSpan(4,4).first,16);EXPECT_EQ(selected.hostTileSpan(4,4).end,18);
    inherited[1]=5;
    EXPECT_THROW(selected.validateHostPublications(),std::invalid_argument);
}
TEST(FloatingProjectionRowWork, SlicedRepeatedTilesCoverOnlyTheirOriginalAuthoredRowsOnce) {
    std::array<std::array<int,3>,3> replay{{{4096,8,0},{2048,6,0},{512,0,0}}};
    const auto work=ProjectionRowWork::causalStrided(3,8,&replay[0][0],&replay[0][1],3,8192,2051)
        .repeatEachRow(4).slice(36,24);
    ITensorGemm::ProjectionWorkScope scope(work);
    const auto selected=FloatingProjectionRows::bind(24,nullptr,nullptr);
    std::array<int,24> visits{};
    for(int tile=0;tile<selected.hostTiles(4);++tile) {
        const auto span=selected.hostTileSpan(tile,4);ASSERT_TRUE(span.valid());
        ASSERT_GE(span.first,0);ASSERT_LE(span.end,24);
        for(int row=span.first;row<span.end;++row)++visits[row];
    }
    for(int row=0;row<24;++row) {
        const int logical=(work.originalRow(row)%32)/4;
        EXPECT_EQ(visits[row],int(logical>=3&&logical<6));
        EXPECT_EQ(selected.hostRowIsActive(row),visits[row]!=0);
    }
}
/** @brief Preparation inherits original demand without a host read or arithmetic change. */
TEST(QwenSparseQueryWork, PreparationDistinguishesAllLiveAndRankingOnlyAtOriginalBoundary) {
    qwen4exp::QwenSparseReplay replay{512,8,0};
    qwen4exp::QwenSparseIndexer p;
    p.replay=&replay;p.row_capacity=8;p.token_capacity=8192;
    const auto all=qwen4exp::qwenSparseIndexerQueryWork(p);
    p.query_demand=qwen4exp::QwenSparseQueryDemand::RankingOnly;
    const auto ranking=qwen4exp::qwenSparseIndexerQueryWork(p);
    EXPECT_EQ(all.resolveHost().size(),8);EXPECT_EQ(ranking.resolveHost().size(),0);
    replay.prefix_tokens=2048;
    EXPECT_EQ(ranking.resolveHost().first,3);EXPECT_EQ(ranking.resolveHost().end,8);
    const auto heads=ranking.repeatEachRow(4);
    EXPECT_EQ(heads.resolveHost().first,12);EXPECT_EQ(heads.resolveHost().end,32);
    replay.live_rows=0;
    EXPECT_EQ(all.resolveHost().size(),0);EXPECT_EQ(ranking.resolveHost().size(),0);
    p.query_demand=static_cast<qwen4exp::QwenSparseQueryDemand>(255);
    EXPECT_THROW((void)qwen4exp::qwenSparseIndexerQueryWork(p),std::invalid_argument);
}
/** @brief Empty request tiles retain physical origins while nonempty suffixes advance. */
TEST(FloatingProjectionRowWork, EmptyTilesRetainOriginalOriginsAcrossPrefixAndCausalDirtyReplay) {
    std::array<int,3> live{0,5,2};
    const auto requests=DeviceRequestRowRanges::deviceCounted(3,5,live.data());
    const auto prefix=FloatingProjectionRows::bind(15,nullptr,&requests);
    const std::array<int,6> origins{0,4,5,9,10,14},counts{0,0,4,1,2,0};
    for(int tile=0;tile<6;++tile) {
        EXPECT_EQ(prefix.hostTileFirst(tile,4),origins[tile]);
        EXPECT_EQ(prefix.hostTileRows(tile,4),counts[tile]);
        const auto span=prefix.hostTileSpan(tile,4);
        EXPECT_EQ(span.first,origins[tile]);EXPECT_EQ(span.end,origins[tile]+counts[tile]);
    }
    std::array<std::array<int,3>,3> replay{{{2048,3,0},{2050,5,0},{4096,2,0}}};
    const auto work=ProjectionRowWork::causalStrided(3,5,&replay[0][0],&replay[0][1],3,8192,2051);
    ITensorGemm::ProjectionWorkScope scope(work);
    const auto causal=FloatingProjectionRows::bind(15,nullptr,nullptr);
    const std::array<int,6> selectedFirst{0,4,6,9,10,14},selectedCount{0,0,3,1,2,0};
    for(int tile=0;tile<6;++tile) {
        const auto span=causal.hostTileSpan(tile,4);ASSERT_TRUE(span.valid());
        EXPECT_EQ(span.first,selectedFirst[tile]);EXPECT_EQ(span.size(),selectedCount[tile]);
    }
    replay[1][1]=0;
    EXPECT_EQ(causal.hostTileFirst(2,4),5);EXPECT_EQ(causal.hostTileRows(2,4),0);
    EXPECT_EQ(causal.hostTileSpan(2,4).end,5);
    EXPECT_EQ(causal.hostTileFirst(3,4),9);EXPECT_EQ(causal.hostTileRows(3,4),0);
    replay[1][1]=5;
    EXPECT_EQ(causal.hostTileFirst(2,4),6);EXPECT_EQ(causal.hostTileRows(2,4),3);
}
} // namespace llaminar2
