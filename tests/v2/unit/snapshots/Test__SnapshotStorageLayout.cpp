/**
 * @file Test__SnapshotStorageLayout.cpp
 * @brief Device-free regression for absolute authored-header alignment and live copies.
 *
 * The real diagnostic validator/copy and descriptor alias mapper exercise the
 * extracted executor layout. Poisoned unused bytes prove exact live extents;
 * no graph, core DSO, device, model, allocation authority or host mirror is used.
 */
#include "snapshots/SnapshotStorageLayout.h"
#include "snapshots/SnapshotSemanticObservation.h"
#include "snapshots/SnapshotCapture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>
using namespace llaminar2;
namespace {
/** @return Exact maximum payload for small already-positive fixture geometry. */
std::size_t bytes(DiagnosticSnapshotCopyGeometry geometry)
{
    return static_cast<std::size_t>(geometry.rows)*geometry.columns*geometry.elementBytes;
}
/** @return Frozen maximum dimensions with no native pointer or live state access. */
DiagnosticSnapshotCopyGeometry geometry(DiagnosticSnapshotCopyKind kind,int elementBytes)
{
    return {kind,3,3,elementBytes,
        kind==DiagnosticSnapshotCopyKind::StatePrefixRows?12:0,
        kind==DiagnosticSnapshotCopyKind::StatePrefixRows?4:1};
}
/** @return Actual descriptor projection; semantic aliases are not another output. */
SnapshotStorageLayout layout(const StageDumpInfo::OutputBuffer &output)
{
    return snapshotStorageLayout({output.byte_size,snapshotCopyGeometry(output)});
}
}

/** @test Old4mod8 header fails the actual validator; per-slot authored alignment fixes every kind/width. */
TEST(SnapshotStorageLayout, OddLegacyPredecessorRejectsOldFourByteBaseForEveryAuthoredKind)
{
    const auto legacy=snapshotStorageLayout({sizeof(float),std::nullopt});
    const auto preceding=placeSnapshotStorageSlot(0,legacy);
    ASSERT_EQ(preceding.end_bytes,4u);
    for(const auto kind:{DiagnosticSnapshotCopyKind::TokenRows,
        DiagnosticSnapshotCopyKind::StatePrefixRows,DiagnosticSnapshotCopyKind::CountedSelectionRows})
    for(const int elementBytes:{4,8})
    {
        SCOPED_TRACE(::testing::Message()<<"kind="<<static_cast<int>(kind)<<" element="<<elementBytes);
        const auto g=geometry(kind,elementBytes);
        const auto authored=snapshotStorageLayout({bytes(g),g});
        EXPECT_EQ(authored.required_alignment,alignof(SnapshotObservedHeader));
        EXPECT_EQ(authored.metadata_bytes,diagnosticSnapshotMetadataCapacity(g));
        EXPECT_EQ(authored.metadata_offset_bytes%8,0u);
        std::array<std::uint64_t,256> arena{};
        std::vector<std::uint32_t> source(bytes(g)/4);std::iota(source.begin(),source.end(),1u);
        const std::array<int,2> frontier{3,2};const int live=3;const std::array<int,3> counts{1,0,3};
        auto *base=reinterpret_cast<std::byte *>(arena.data());
        DiagnosticSnapshotCopyArgs old{g,source.data(),base+preceding.end_bytes,
            reinterpret_cast<SnapshotObservedHeader *>(base+preceding.end_bytes+authored.metadata_offset_bytes),
            kind==DiagnosticSnapshotCopyKind::StatePrefixRows?frontier.data():nullptr,&live,
            kind==DiagnosticSnapshotCopyKind::CountedSelectionRows?counts.data():nullptr};
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(old.metadata)%8,4u);
        EXPECT_THROW(validateDiagnosticSnapshotCopy(old),std::invalid_argument);
        const auto slot=placeSnapshotStorageSlot(preceding.end_bytes,authored);
        ASSERT_EQ(slot.offset_bytes,8u);
        old.destination=base+slot.offset_bytes;
        old.metadata=reinterpret_cast<SnapshotObservedHeader *>(base+slot.offset_bytes+authored.metadata_offset_bytes);
        EXPECT_NO_THROW(validateDiagnosticSnapshotCopy(old));
        EXPECT_NO_THROW(copyDiagnosticSnapshotCPU(old));
        const auto expectedRows=kind==DiagnosticSnapshotCopyKind::StatePrefixRows?1u:3u;
        const auto expectedBytes=kind==DiagnosticSnapshotCopyKind::CountedSelectionRows
            ?4u*elementBytes:expectedRows*g.columns*elementBytes;
        EXPECT_EQ(old.metadata->payloadBytes,expectedBytes);
        EXPECT_LE(slot.offset_bytes+authored.metadata_offset_bytes+authored.metadata_bytes,slot.end_bytes);
    }
}

/** @test Large-small-empty copies retain exactly their metadata and leave inactive payload/padding poisoned. */
TEST(SnapshotStorageLayout, AuthoredReplayPreservesInactivePayloadAndBothAlignmentGaps)
{
    const auto legacy=placeSnapshotStorageSlot(0,snapshotStorageLayout({4,std::nullopt}));
    for(const auto kind:{DiagnosticSnapshotCopyKind::TokenRows,
        DiagnosticSnapshotCopyKind::StatePrefixRows,DiagnosticSnapshotCopyKind::CountedSelectionRows})
    for(const int elementBytes:{4,8})
    {
        const auto g=geometry(kind,elementBytes);const auto retained=snapshotStorageLayout({bytes(g),g});
        const auto slot=placeSnapshotStorageSlot(legacy.end_bytes,retained);
        std::array<std::uint64_t,256> arena;
        auto *base=reinterpret_cast<std::byte *>(arena.data());
        std::vector<std::uint32_t> source(bytes(g)/4);std::iota(source.begin(),source.end(),7u);
        const std::array<int,3> counts{1,0,3};
        for(const int live:{3,1,0,3})
        {
            SCOPED_TRACE(::testing::Message()<<"kind="<<static_cast<int>(kind)<<" element="<<elementBytes<<" live="<<live);
            std::memset(arena.data(),0xa5,sizeof(arena));
            const int nativeLive=kind==DiagnosticSnapshotCopyKind::StatePrefixRows?live*4:live;
            const std::array<int,2> frontier{0,nativeLive};
            auto *header=reinterpret_cast<SnapshotObservedHeader *>(base+slot.offset_bytes+retained.metadata_offset_bytes);
            auto *destination=base+slot.offset_bytes;
            copyDiagnosticSnapshotCPU({g,source.data(),destination,header,
                kind==DiagnosticSnapshotCopyKind::StatePrefixRows?frontier.data():nullptr,&live,
                kind==DiagnosticSnapshotCopyKind::CountedSelectionRows?counts.data():nullptr});
            std::vector<std::uint32_t> expected;
            if(kind==DiagnosticSnapshotCopyKind::CountedSelectionRows) {
                for(int row=0;row<live;++row) {
                    const auto begin=source.begin()+row*g.columns*(elementBytes/4);
                    expected.insert(expected.end(),begin,begin+counts[row]*(elementBytes/4));
                }
                const auto *offsets=reinterpret_cast<const std::uint64_t *>(header+1);
                EXPECT_EQ(offsets[0],0u);EXPECT_EQ(offsets[live]*elementBytes,header->payloadBytes);
            } else {
                expected.assign(source.begin(),source.begin()+live*g.columns*(elementBytes/4));
            }
            EXPECT_EQ(header->liveRows,nativeLive);EXPECT_EQ(header->payloadBytes,expected.size()*4);
            const SnapshotSemanticExtent extent=kind==DiagnosticSnapshotCopyKind::StatePrefixRows
                ? SnapshotSemanticExtent(SnapshotStatePrefixRows{{},12,4})
                : kind==DiagnosticSnapshotCopyKind::CountedSelectionRows
                    ? SnapshotSemanticExtent(SnapshotCountedSelectionRows{})
                    : SnapshotSemanticExtent(SnapshotTokenRows{});
            const auto observed=validateSnapshotObservedHeader(extent,*header,g.rows,g.columns,elementBytes);
            EXPECT_EQ(observed.rows,static_cast<std::size_t>(live));
            if(kind==DiagnosticSnapshotCopyKind::CountedSelectionRows)
                EXPECT_NO_THROW(validateSnapshotSelectionOffsets(
                    {reinterpret_cast<const std::uint64_t *>(header+1),static_cast<std::size_t>(live)+1},
                    observed,g.columns,elementBytes));
            if (!expected.empty()) EXPECT_EQ(std::memcmp(destination,expected.data(),expected.size()*4),0);
            const auto poisoned=[](std::byte value){return value==std::byte{0xa5};};
            EXPECT_TRUE(std::all_of(base+legacy.end_bytes,base+slot.offset_bytes,poisoned));
            EXPECT_TRUE(std::all_of(destination+header->payloadBytes,destination+retained.payload_storage_bytes,poisoned));
            EXPECT_TRUE(std::all_of(destination+retained.payload_storage_bytes,destination+retained.metadata_offset_bytes,poisoned));
            EXPECT_TRUE(std::all_of(base+slot.end_bytes,base+sizeof(arena),poisoned));
        }
    }
}

/** @test The actual alias mapper yields two names for ONE descriptor and storage is priced once. */
TEST(SnapshotStorageLayout, SemanticAliasesShareOneOutputButTemporalProducersRemainSeparate)
{
    std::array<float,3> source{1,2,3};
    StageDumpInfo dump;
    dump.outputs.push_back({.name="branch_read",.data=source.data(),.rows=1,.cols=3,
        .dtype="FP32",.element_size=4,.byte_size=12,.semantic_extent=SnapshotTokenRows{}});
    const auto keys=SnapshotCapture::possibleKeysForStage("layer0_attn_norm",dump);
    ASSERT_EQ(keys.size(),2u);
    EXPECT_NE(std::find(keys.begin(),keys.end(),"layer0_ATTENTION_NORM"),keys.end());
    EXPECT_NE(std::find(keys.begin(),keys.end(),"layer0_GR_READ"),keys.end());
    ASSERT_EQ(dump.outputs.size(),1u);
    const auto firstLayout=layout(dump.outputs[0]);const auto first=placeSnapshotStorageSlot(0,firstLayout);
    EXPECT_EQ(first.end_bytes,firstLayout.storage_bytes);
    // A later producer reuses the same physical source owner. It is a second
    // point-in-time observation, regardless of tensor identity or semantic keys.
    const auto secondLayout=layout(dump.outputs[0]);const auto second=placeSnapshotStorageSlot(first.end_bytes,secondLayout);
    EXPECT_EQ(second.offset_bytes,first.end_bytes);EXPECT_EQ(second.end_bytes,2*first.end_bytes);
    std::array<std::uint64_t,32> arena{};auto *base=reinterpret_cast<std::byte *>(arena.data());
    auto copyAt=[&](SnapshotStorageSlot slot,const SnapshotStorageLayout &shape) {
        auto *header=reinterpret_cast<SnapshotObservedHeader *>(base+slot.offset_bytes+shape.metadata_offset_bytes);
        copyDiagnosticSnapshotCPU({snapshotCopyGeometry(dump.outputs[0]),source.data(),base+slot.offset_bytes,header});
    };
    copyAt(first,firstLayout);source={4,5,6};copyAt(second,secondLayout);
    const std::array<float,3> before{1,2,3},after{4,5,6};
    EXPECT_EQ(std::memcmp(base+first.offset_bytes,before.data(),sizeof(before)),0);
    EXPECT_EQ(std::memcmp(base+second.offset_bytes,after.data(),sizeof(after)),0);
}

/** @test Existing four-byte legacy payload rounding and placement remain unchanged. */
TEST(SnapshotStorageLayout, LegacyRoundingAndMixedOrderUseOnlyNecessaryPadding)
{
    for(std::size_t bytes=1;bytes<=33;++bytes) {
        const auto shape=snapshotStorageLayout({bytes,std::nullopt});
        EXPECT_EQ(shape.payload_storage_bytes,((bytes+3)/4)*4);
        EXPECT_EQ(shape.storage_bytes,shape.payload_storage_bytes);
        EXPECT_EQ(shape.metadata_bytes,0u);EXPECT_EQ(shape.metadata_offset_bytes,0u);
        EXPECT_EQ(shape.required_alignment,4u);
        const auto slot=placeSnapshotStorageSlot(4,shape);EXPECT_EQ(slot.offset_bytes,4u);
    }
    const auto odd=snapshotStorageLayout({4,std::nullopt});const auto even=snapshotStorageLayout({8,std::nullopt});
    const auto authored=snapshotStorageLayout({4,DiagnosticSnapshotCopyGeometry{DiagnosticSnapshotCopyKind::TokenRows,1,1,4}});
    auto next=placeSnapshotStorageSlot(0,even);EXPECT_EQ(placeSnapshotStorageSlot(next.end_bytes,authored).offset_bytes,8u);
    next=placeSnapshotStorageSlot(0,odd);const auto current=placeSnapshotStorageSlot(next.end_bytes,authored);
    EXPECT_EQ(current.offset_bytes,8u);EXPECT_EQ(current.end_bytes,32u);
    EXPECT_EQ(placeSnapshotStorageSlot(current.end_bytes,odd).offset_bytes,32u);
}

/** @test Zero retained rows are invalid while an admitted slot with zero live rows remains valid. */
TEST(SnapshotStorageLayout, RejectsInvalidGeometryAndUndersizedNativePayload)
{
    EXPECT_THROW((void)snapshotStorageLayout({0,std::nullopt}),std::invalid_argument);
    for(const auto kind:{DiagnosticSnapshotCopyKind::TokenRows,
        DiagnosticSnapshotCopyKind::StatePrefixRows,DiagnosticSnapshotCopyKind::CountedSelectionRows}) {
        auto g=geometry(kind,4);g.rows=0;EXPECT_THROW((void)snapshotStorageLayout({4,g}),std::invalid_argument);
        g=geometry(kind,4);g.columns=0;EXPECT_THROW((void)snapshotStorageLayout({4,g}),std::invalid_argument);
        g=geometry(kind,4);g.elementBytes=2;EXPECT_THROW((void)snapshotStorageLayout({4,g}),std::invalid_argument);
        g=geometry(kind,4);EXPECT_THROW((void)snapshotStorageLayout({bytes(g)-1,g}),std::invalid_argument);
    }
    auto state=geometry(DiagnosticSnapshotCopyKind::StatePrefixRows,4);state.sourceCapacity=13;
    EXPECT_THROW((void)snapshotStorageLayout({bytes(state),state}),std::invalid_argument);
    state=geometry(DiagnosticSnapshotCopyKind::StatePrefixRows,4);state.sourceRowsPerStateRow=0;
    EXPECT_THROW((void)snapshotStorageLayout({bytes(state),state}),std::invalid_argument);
    auto unknown=geometry(DiagnosticSnapshotCopyKind::TokenRows,4);unknown.kind=static_cast<DiagnosticSnapshotCopyKind>(255);
    EXPECT_THROW((void)snapshotStorageLayout({bytes(unknown),unknown}),std::invalid_argument);
}

/** @test Payload rounding, internal/leading alignment, metadata and complete ends cannot wrap. */
TEST(SnapshotStorageLayout, CheckedOverflowAtEveryLayoutBoundary)
{
    constexpr auto maximum=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((void)snapshotStorageLayout({maximum,std::nullopt}),std::overflow_error);
    const auto largestAligned=snapshotStorageLayout({maximum-3,std::nullopt});
    EXPECT_EQ(largestAligned.storage_bytes,maximum-3);
    EXPECT_EQ(placeSnapshotStorageSlot(0,largestAligned).end_bytes,maximum-3);
    const DiagnosticSnapshotCopyGeometry token{DiagnosticSnapshotCopyKind::TokenRows,1,1,4};
    EXPECT_THROW((void)snapshotStorageLayout({maximum-3,token}),std::overflow_error);
    EXPECT_THROW((void)snapshotStorageLayout({maximum-7,token}),std::overflow_error);
    const auto legacy=snapshotStorageLayout({4,std::nullopt});
    EXPECT_THROW((void)placeSnapshotStorageSlot(maximum,legacy),std::overflow_error);
    EXPECT_THROW((void)placeSnapshotStorageSlot(maximum-3,legacy),std::overflow_error);
    const auto authored=snapshotStorageLayout({4,token});
    EXPECT_THROW((void)placeSnapshotStorageSlot(maximum-3,authored),std::overflow_error);
}
