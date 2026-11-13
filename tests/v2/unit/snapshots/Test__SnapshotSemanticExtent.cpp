/**
 * @file Test__SnapshotSemanticExtent.cpp
 * @brief Device-free proof of authored diagnostic extents, packing and empty states.
 *
 * Poisoned unused capacity proves that native-prefix observations do not copy
 * padding. Required metadata authenticates every output row before diagnostic
 * reconstruction; corrupt counts cannot be hidden by truncation or sentinels.
 */
#include "kernels/common/DiagnosticSnapshotCopy.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>

using namespace llaminar2;

/** @test Complete state pools are a latest native frontier, including a valid zero-pool observation. */
TEST(SnapshotSemanticExtent, StatePrefixCopiesOnlyCompletedRowsIncludingEmpty)
{
    constexpr int capacity=17,columns=128,sourceCapacity=67;
    const DiagnosticSnapshotCopyGeometry geometry{DiagnosticSnapshotCopyKind::StatePrefixRows,capacity,columns,4,sourceCapacity,4};
    std::vector<std::uint32_t> source(capacity*columns),destination(source.size());
    std::iota(source.begin(),source.end(),1u);
    alignas(8) SnapshotObservedHeader metadata;
    for(const auto frontier:std::array{std::array{0,0},std::array{0,3},std::array{3,1},std::array{9,2},std::array{60,7}})
    {
        SCOPED_TRACE(::testing::Message()<<frontier[0]<<"+"<<frontier[1]);
        std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);
        copyDiagnosticSnapshotCPU({geometry,source.data(),destination.data(),&metadata,frontier.data()});
        const auto rows=static_cast<std::size_t>((frontier[0]+frontier[1])/4);
        EXPECT_EQ(metadata.priorRows,frontier[0]);EXPECT_EQ(metadata.liveRows,frontier[1]);
        EXPECT_EQ(metadata.payloadBytes,rows*columns*4);
        const auto observed=validateSnapshotObservedHeader(SnapshotStatePrefixRows{{},sourceCapacity,4},metadata,capacity,columns,4);
        EXPECT_EQ(observed.rows,rows);EXPECT_EQ(observed.payloadBytes,rows*columns*4);EXPECT_EQ(observed.offsetBytes,0u);
        EXPECT_TRUE(std::equal(source.begin(),source.begin()+rows*columns,destination.begin()));
        EXPECT_TRUE(std::all_of(destination.begin()+rows*columns,destination.end(),[](auto v){return v==0xa5a5a5a5u;}));
    }
}

/** @test Ragged selections transfer only authored IDs, preserving inactive rows and destination suffixes. */
TEST(SnapshotSemanticExtent, CountedSelectionsPackOnlyAuthoredValuesAtAllServingRows)
{
    for(const int capacity:{1,15,512,1024})for(const int elementBytes:{4,8})
    {
        SCOPED_TRACE(::testing::Message()<<"M="<<capacity<<" bytes="<<elementBytes);
        constexpr int columns=2051;
        const DiagnosticSnapshotCopyGeometry geometry{DiagnosticSnapshotCopyKind::CountedSelectionRows,capacity,columns,elementBytes};
        std::vector<std::uint32_t> source(static_cast<std::size_t>(capacity)*columns*(elementBytes/4)),destination(source.size());
        std::iota(source.begin(),source.end(),7u);
        const auto metadataBytes=diagnosticSnapshotMetadataCapacity(geometry);
        std::vector<std::uint64_t> metadata(metadataBytes/8+2);
        auto *header=reinterpret_cast<SnapshotObservedHeader *>(metadata.data());
        std::vector<std::int32_t> counts(capacity);
        for(const int live:{capacity,std::min(2,capacity),0,capacity})
        {
            std::fill(counts.begin(),counts.end(),-99);
            for(int row=0;row<live;++row)counts[row]=row%4==0?0:row%4==1?1:row%4==2?columns:17;
            std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);std::fill(metadata.begin(),metadata.end(),0xc3c3c3c3c3c3c3c3ull);
            copyDiagnosticSnapshotCPU({geometry,source.data(),destination.data(),header,nullptr,&live,counts.data()});
            const auto shape=validateSnapshotObservedHeader(SnapshotCountedSelectionRows{},*header,capacity,columns,elementBytes);
            const auto *offsets=reinterpret_cast<const std::uint64_t *>(header+1);
            validateSnapshotSelectionOffsets({offsets,static_cast<std::size_t>(live)+1},shape,columns,elementBytes);
            std::vector<std::uint32_t> expected;
            for(int row=0;row<live;++row)
            {
                const auto begin=source.begin()+static_cast<std::size_t>(row)*columns*(elementBytes/4);
                expected.insert(expected.end(),begin,begin+counts[row]*(elementBytes/4));
            }
            EXPECT_EQ(shape.rows,static_cast<std::size_t>(live));EXPECT_EQ(shape.payloadBytes,expected.size()*4);
            EXPECT_TRUE(std::equal(expected.begin(),expected.end(),destination.begin()));
            EXPECT_TRUE(std::all_of(destination.begin()+expected.size(),destination.end(),[](auto v){return v==0xa5a5a5a5u;}));
            const auto usedMetadata=sizeof(SnapshotObservedHeader)/8+static_cast<std::size_t>(live)+1;
            EXPECT_TRUE(std::all_of(metadata.begin()+usedMetadata,metadata.end(),[](auto v){return v==0xc3c3c3c3c3c3c3c3ull;}));
        }
    }
}

/** @test Corrupt native counts and fabricated payload extents fail before data can become a valid observation. */
TEST(SnapshotSemanticExtent, RejectsCorruptCountsAndNeverSilentlyTruncates)
{
    std::array<std::uint32_t,12> source{},destination;
    std::fill(destination.begin(),destination.end(),0xa5a5a5a5u);
    alignas(8) std::array<std::uint64_t,8> metadata{};
    auto *header=reinterpret_cast<SnapshotObservedHeader *>(metadata.data());
    int live=3;std::array<int,3> counts{1,5,0};
    const DiagnosticSnapshotCopyGeometry geometry{DiagnosticSnapshotCopyKind::CountedSelectionRows,3,4,4};
    EXPECT_THROW(copyDiagnosticSnapshotCPU({geometry,source.data(),destination.data(),header,nullptr,&live,counts.data()}),std::logic_error);
    EXPECT_TRUE(std::all_of(destination.begin(),destination.end(),[](auto v){return v==0xa5a5a5a5u;}));
    EXPECT_THROW(validateSnapshotObservedHeader(SnapshotTokenRows{},SnapshotObservedHeader{0,4,64},3,4,4),std::logic_error);
    EXPECT_THROW(validateSnapshotObservedHeader(SnapshotTokenRows{},SnapshotObservedHeader{0,2,16},3,4,4),std::logic_error);
    const auto shape=validateSnapshotObservedHeader(SnapshotCountedSelectionRows{},SnapshotObservedHeader{0,2,8},3,4,4);
    const std::array<std::uint64_t,3> descending{0,2,1},oversized{0,5,2},wrongTotal{0,0,1};
    EXPECT_THROW(validateSnapshotSelectionOffsets(descending,shape,4,4),std::logic_error);
    EXPECT_THROW(validateSnapshotSelectionOffsets(oversized,shape,4,4),std::logic_error);
    EXPECT_THROW(validateSnapshotSelectionOffsets(wrongTotal,shape,4,4),std::logic_error);
}
