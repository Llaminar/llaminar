/**
 * @file Test__DeviceRequestRowRanges.cpp
 * @brief Device-free admission and indexing proof for captured request rows.
 *
 * A length pointer is an identity only: deliberately invalid addresses prove
 * factories and pure validation never dereference device metadata on the host.
 * Empty requests preserve following requests' physical coordinates.
 * Merged recurrence admission also proves disjoint source/output byte ranges
 * while preserving the separate, intentional in-place state alias.
 */
#include <gtest/gtest.h>
#include "kernels/common/DeviceRequestRowRanges.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "kernels/gdn/GDNPreparedQK.h"
#include <array>
#include <limits>

using namespace llaminar2;

TEST(DeviceRequestRowRanges, FixedRowsHaveNoMutableAuthority)
{
    const auto rows = DeviceRequestRowRanges::fullyActive(16);
    EXPECT_EQ(rows.requests(), 1);
    EXPECT_EQ(rows.rowsPerRequest(), 16);
    EXPECT_EQ(rows.physicalRows(), 16);
    EXPECT_EQ(rows.lengthOwner(), nullptr);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(0), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::fullyActive(-1), std::invalid_argument);
}

TEST(DeviceRequestRowRanges, RequestPrefixesRemainDisjointAndChecked)
{
    const auto *identity = reinterpret_cast<const std::int32_t *>(std::uintptr_t{0x12340});
    const auto rows = DeviceRequestRowRanges::deviceCounted(3, 16, identity);
    EXPECT_EQ(rows.lengthOwner(), identity);
    EXPECT_EQ(rows.physicalRows(), 48);
    for (int request = 0; request < 3; ++request)
        for (int length = 0; length <= 16; ++length)
            EXPECT_EQ(rows.activeRowsFor(request, length), length);
    EXPECT_EQ(rows.activeRowsFor(0, 0), 0);
    EXPECT_EQ(rows.activeRowsFor(2, 16), 16);
    EXPECT_EQ(rows.activeRowsFor(-1, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(3, 1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, -1), -1);
    EXPECT_EQ(rows.activeRowsFor(0, 17), -1);
}

TEST(DeviceRequestRowRanges, InvalidAdmissionCannotOverflowOrOmitAuthority)
{
    const std::int32_t owner = 0;
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(1, 16, nullptr), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(0, 16, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, 0, &owner), std::invalid_argument);
    EXPECT_THROW((void)DeviceRequestRowRanges::deviceCounted(2, std::numeric_limits<int>::max(), &owner),
        std::invalid_argument);
    const auto largest = DeviceRequestRowRanges::deviceCounted(1, std::numeric_limits<int>::max(), &owner);
    EXPECT_EQ(largest.physicalRows(), std::numeric_limits<int>::max());
}

TEST(DeviceRequestRowRanges, DeinterleaveGeometryRejectsUnrepresentableMatrices)
{
    const std::int32_t owner = 0;
    const auto rows = DeviceRequestRowRanges::deviceCounted(2, 16, &owner);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, 128), 16 * 40 * 128 * 3);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 0, 40, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, -1, 128, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 0, 128), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, 16, 40, 128, -1), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(rows, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()), 0);
    EXPECT_EQ(gdnDeinterleaveElementsPerRequest(DeviceRequestRowRanges::fullyActive(
        std::numeric_limits<int>::max()), 1, 1, 1, 1), 0);
}

/** @test Half-open source/output ranges cover every physical request row. */
TEST(DeviceRequestRowRanges, MergedSourceOutputIntervalsPreserveOnlyStateAlias)
{
    constexpr int physical_rows = 6;
    constexpr int source_elements = physical_rows * (2 * 64 + 2 * 65);
    constexpr int output_elements = physical_rows * 2 * 65;
    std::array<float, source_elements + output_elements> storage{};
    const auto *lengths = reinterpret_cast<const int *>(std::uintptr_t{0x12340});
    gdn::MergedRecurrenceInput input{
        .rows = DeviceRequestRowRanges::deviceCounted(2, 3, lengths),
        .key_heads = 1, .value_heads = 2, .key_width = 64, .value_width = 65,
        .merged = storage.data(),
        .output = storage.data() + source_elements,
        .initial_state = storage.data(), .updated_state = storage.data(),
    };
    EXPECT_TRUE(input.sourceOutputDisjoint());
    --input.output;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = storage.data();
    EXPECT_FALSE(input.sourceOutputDisjoint());

    input.merged = storage.data() + output_elements;
    EXPECT_TRUE(input.sourceOutputDisjoint());
    --input.merged;
    EXPECT_FALSE(input.sourceOutputDisjoint());
}

/** @test Invalid shape and pointer-end overflow fail before any payload access. */
TEST(DeviceRequestRowRanges, MergedSourceOutputRejectsUnrepresentableRanges)
{
    const auto source = std::uintptr_t{0x10000};
    const auto destination = std::uintptr_t{0x100000};
    gdn::MergedRecurrenceInput input{
        .rows = DeviceRequestRowRanges::fullyActive(1),
        .key_heads = 1, .value_heads = 2, .key_width = 64, .value_width = 65,
        .merged = reinterpret_cast<const float *>(source),
        .output = reinterpret_cast<float *>(destination),
    };
    ASSERT_TRUE(input.sourceOutputDisjoint());
    input.merged = nullptr;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.merged = reinterpret_cast<const float *>(source);
    input.output = nullptr;
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = reinterpret_cast<float *>(destination);
    const auto near_end = std::numeric_limits<std::uintptr_t>::max() - sizeof(float);
    input.merged = reinterpret_cast<const float *>(near_end);
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.merged = reinterpret_cast<const float *>(source);
    input.output = reinterpret_cast<float *>(near_end);
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.output = reinterpret_cast<float *>(destination);
    input.key_width = std::numeric_limits<int>::max();
    EXPECT_FALSE(input.sourceOutputDisjoint());
    input.key_width = 64;
    input.value_heads = 0;
    EXPECT_FALSE(input.sourceOutputDisjoint());
}


/** @test Canonical planar metadata preserves float alignment and exact state alias only. */
TEST(DeviceRequestRowRanges, PlanarRecurrenceLayoutsProtectEveryNativeSpan)
{
    const auto read=[](std::uintptr_t p){return reinterpret_cast<const float*>(p);};
    const auto write=[](std::uintptr_t p){return reinterpret_cast<float*>(p);};
    gdn::PlanarRecurrenceInput p{
        DeviceRequestRowRanges::deviceCounted(3,16,reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x200000000})),
        3,64,65,true,read(0x100000000),read(0x110000000),read(0x120000000),read(0x130000000),read(0x140000000),
        read(0x150000000),read(0x160000000),write(0x170000000),read(0x180000000),write(0x190000000),write(0x1a0000000),
        3*64*65+13,48,GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    ASSERT_TRUE(gdn::validPlanarRecurrenceStorage(p));
    // Equal capacity is not a format detector: both explicit known formats are
    // valid metadata; owner-bound typed workspace wrappers prove compatibility.
    p.recurrence_layout=GDNRecurrenceStateLayout::KeyMajorValueContiguous;
    EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(p));
    p.recurrence_layout=static_cast<GDNRecurrenceStateLayout>(255);
    EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));
    p.recurrence_layout=GDNRecurrenceStateLayout::ValueMajorKeyContiguous;
    const auto initial=p.initial_state;const auto updated=p.updated_state;
    p.updated_state=const_cast<float*>(initial);EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(p));
    p.updated_state=write(reinterpret_cast<std::uintptr_t>(initial)+4);EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));
    p.updated_state=updated;
    p.query=read(reinterpret_cast<std::uintptr_t>(p.query)+4);
    p.initial_state=read(reinterpret_cast<std::uintptr_t>(initial)+4);
    p.updated_state=write(reinterpret_cast<std::uintptr_t>(updated)+4);
    EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(p)); // First-class scalar-aligned endpoints.
    p.initial_state=read(reinterpret_cast<std::uintptr_t>(initial)+1);EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));
    p.initial_state=initial;p.updated_state=updated;
    const auto output=p.output;p.output=const_cast<float*>(p.query);EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));p.output=output;
    const auto shots=p.snapshots;p.snapshots=p.output;EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));p.snapshots=shots;
    p.max_snapshot_rows=17;EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(p)); // Partial long-prefill envelope.
    p.snapshot_stride_floats=0;EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));p.snapshot_stride_floats=3*64*65+13;
    p.query=nullptr;EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));p.query=read(0x100000000);
    p.rows=DeviceRequestRowRanges::fullyActiveRequests(3,16);EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(p));
    p.heads=std::numeric_limits<int>::max();EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));p.heads=3;
    p.output=write(std::numeric_limits<std::uintptr_t>::max()-3);EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(p));
}

/** @test Compact merged spans use original compact bytes, reject overflow and preserve both layout tags. */
TEST(DeviceRequestRowRanges, MergedRecurrenceLayoutsProtectCompactSourcesAndState)
{
    const auto read=[](std::uintptr_t p){return reinterpret_cast<const float*>(p);};
    const auto write=[](std::uintptr_t p){return reinterpret_cast<float*>(p);};
    constexpr std::uintptr_t source=0x100000000;
    constexpr int source_words=3*16*(2*64+3*65);
    gdn::MergedRecurrenceInput p{
        DeviceRequestRowRanges::fullyActiveRequests(3,16),1,3,64,65,-5,true,read(source),read(0x130000000),read(0x140000000),
        read(0x150000000),read(0x160000000),write(source+source_words*sizeof(float)),read(0x180000000),write(0x190000000),
        write(0x1a0000000),3*64*65+13,48,GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    ASSERT_TRUE(gdn::validMergedRecurrenceStorage(p));
    p.recurrence_layout=GDNRecurrenceStateLayout::KeyMajorValueContiguous;EXPECT_TRUE(gdn::validMergedRecurrenceStorage(p));
    p.recurrence_layout=static_cast<GDNRecurrenceStateLayout>(255);EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));
    p.recurrence_layout=GDNRecurrenceStateLayout::ValueMajorKeyContiguous;
    const auto updated=p.updated_state;p.updated_state=const_cast<float*>(p.initial_state);EXPECT_TRUE(gdn::validMergedRecurrenceStorage(p));
    p.updated_state=write(reinterpret_cast<std::uintptr_t>(p.initial_state)+4);EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));p.updated_state=updated;
    p.merged=read(source+4);p.output=write(source+(source_words+1)*sizeof(float));EXPECT_TRUE(gdn::validMergedRecurrenceStorage(p));
    p.merged=read(source+1);EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));p.merged=read(source);
    const auto alpha=p.alpha;p.alpha=p.output;EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));p.alpha=alpha;
    p.merged=nullptr;EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));p.merged=read(source);
    p.key_heads=std::numeric_limits<int>::max();EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));p.key_heads=1;
    p.initial_state=read(std::numeric_limits<std::uintptr_t>::max()-3);EXPECT_FALSE(gdn::validMergedRecurrenceStorage(p));
}

/** @test Valid request-state spans exceeding INT32 elements retain checked size_t addressing. */
TEST(DeviceRequestRowRanges, RecurrenceStateExtentsUseWideRequestOffsets)
{
    static_assert(sizeof(std::uintptr_t)>=8,"Supported native targets use 64-bit addresses");
    constexpr std::uintptr_t tb=std::uintptr_t{1}<<40;
    const auto read=[](std::uintptr_t p){return reinterpret_cast<const float*>(p);};
    const auto write=[](std::uintptr_t p){return reinterpret_cast<float*>(p);};
    constexpr int requests=131072,heads=1,dk=128,dv=128;
    static_assert(std::uint64_t(requests)*heads*dk*dv>std::uint64_t(std::numeric_limits<int>::max()));
    gdn::PlanarRecurrenceInput planar{
        DeviceRequestRowRanges::fullyActiveRequests(requests,1),heads,dk,dv,true,
        read(tb),read(2*tb),read(3*tb),read(4*tb),read(5*tb),read(6*tb),read(7*tb),
        write(20*tb),read(30*tb),write(40*tb),nullptr,0,0,GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    EXPECT_TRUE(gdn::validPlanarRecurrenceStorage(planar));
    gdn::MergedRecurrenceInput merged{
        DeviceRequestRowRanges::fullyActiveRequests(requests,1),1,heads,dk,dv,0,true,
        read(tb),read(4*tb),read(5*tb),read(6*tb),read(7*tb),write(20*tb),read(30*tb),write(40*tb),nullptr,0,0,
        GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
    EXPECT_TRUE(gdn::validMergedRecurrenceStorage(merged));
    planar.updated_state=write(std::numeric_limits<std::uintptr_t>::max()-3);
    merged.updated_state=planar.updated_state;
    EXPECT_FALSE(gdn::validPlanarRecurrenceStorage(planar));
    EXPECT_FALSE(gdn::validMergedRecurrenceStorage(merged));
}


/** @test One scalar activation formula agrees across original projection/head geometry. */
TEST(DeviceRequestRowRanges, ScalarPreparedQKCapacityUsesOriginalProjectionWidths)
{
    for(const int dk:{64,128})
        for(const int keys:{1,4,16})
            for(const int heads:{1,3,12})
                for(const int dv:{1,65,128,129})
                {
                    const auto from_heads=gdn::ScalarPreparedQKLayout::fromHeads(keys,heads,dk,dv);
                    const auto from_weights=gdn::ScalarPreparedQKLayout::fromProjectionWidths(
                        2*std::size_t(keys)*dk+std::size_t(heads)*dv,std::size_t(heads)*dv);
                    EXPECT_EQ(from_heads.plane_elements,from_weights.plane_elements);
                    EXPECT_EQ(from_heads.elements,from_weights.elements);
                    EXPECT_EQ(from_heads.bytes,from_weights.bytes);
                }
    EXPECT_EQ(gdn::ScalarPreparedQKLayout::fromHeads(4,12,128,128).bytes,4096u);
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromProjectionWidths(128,128),std::invalid_argument);
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromProjectionWidths(130,129),std::invalid_argument);
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromHeads(1,1,32,128),std::invalid_argument);
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromHeads(std::numeric_limits<int>::max(),1,128,128),std::invalid_argument);
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromProjectionWidths(
        std::numeric_limits<std::size_t>::max(),1),std::invalid_argument);
}

/** @test Compact producer/consumer admission preserves all original owners and exact state alias only. */
TEST(DeviceRequestRowRanges, ScalarPreparedQKStorageProtectsRawStateAndSnapshots)
{
    const auto read=[](std::uintptr_t p){return reinterpret_cast<const float*>(p);};
    const auto write=[](std::uintptr_t p){return reinterpret_cast<float*>(p);};
    // Deliberately non-dereferenceable authored ranges prove device data stays opaque.
    gdn::ScalarPreparedQKConvolutionInput producer{
        {4,12,128,128,true,read(0x100000000),read(0x110000000),read(0x120000000),
         write(0x130000000),write(0x140000000)},
        read(0x150000000),write(0x160000000),write(0x170000000),3*2560+13,2};
    ASSERT_TRUE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    const auto original=producer;
    producer.updated_state=const_cast<float*>(producer.initial_state);
    EXPECT_TRUE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;
    producer.updated_state=write(reinterpret_cast<std::uintptr_t>(producer.initial_state)+4);
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.raw_output=const_cast<float*>(producer.operands.input);
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.prepared=producer.operands.raw_output;
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.prepared=write(reinterpret_cast<std::uintptr_t>(producer.operands.weight)+4);
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.snapshots=const_cast<float*>(producer.initial_state);
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.prepared=write(std::numeric_limits<std::uintptr_t>::max()-3);
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.value_heads=std::numeric_limits<int>::max();
    EXPECT_FALSE(gdn::validScalarPreparedQKConvolutionStorage(producer));
    producer=original;producer.operands.prepared=write(0x140000004);
    EXPECT_TRUE(gdn::validScalarPreparedQKConvolutionStorage(producer)); // Scalar-aligned native endpoint.

    gdn::ScalarPreparedQKRecurrenceInput consumer{{
        DeviceRequestRowRanges::deviceCounted(1,1,reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x210000000})),
        4,12,128,128,-5,true,read(0x100000000),read(0x110000000),read(0x120000000),
        read(0x130000000),read(0x140000000),write(0x150000000),read(0x160000000),
        write(0x170000000),write(0x180000000),12*128*128+13,2,
        GDNRecurrenceStateLayout::ValueMajorKeyContiguous},read(0x190000000)};
    ASSERT_TRUE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    const auto sealed=consumer;
    consumer.original.updated_state=const_cast<float*>(consumer.original.initial_state);
    EXPECT_TRUE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=consumer.original.merged;
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=consumer.original.alpha;
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=consumer.original.output;
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=consumer.original.initial_state;
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=read(reinterpret_cast<std::uintptr_t>(consumer.original.snapshots)+
        std::size_t(consumer.original.snapshot_stride_floats)*sizeof(float));
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer)); // Protect full retained envelope.
    consumer=sealed;consumer.prepared=reinterpret_cast<const float*>(consumer.original.rows.lengthOwner());
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.original.recurrence_layout=GDNRecurrenceStateLayout::KeyMajorValueContiguous;
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.original.rows=DeviceRequestRowRanges::fullyActive(2);
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.original.rows=DeviceRequestRowRanges::fullyActiveRequests(2,1);
    EXPECT_FALSE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
    consumer=sealed;consumer.prepared=read(0x190000004);
    EXPECT_TRUE(gdn::validScalarPreparedQKRecurrenceStorage(consumer));
}
