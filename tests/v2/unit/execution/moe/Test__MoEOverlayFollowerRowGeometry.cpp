/**
 * @file Test__MoEOverlayFollowerRowGeometry.cpp
 * @brief Exact dynamic follower row identities over unchanged admitted packet roots.
 *
 * Actual CPU storage declarations and production transaction protocol execute
 * without model/native readiness. Exact execution rows never become backing
 * capacities, and source-declared prefill ladders remain independently owned.
 */
#include "execution/moe/MoEOverlayFollowerRowGeometry.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include <gtest/gtest.h>
#include <array>
#include <numeric>

namespace llaminar2::test {
/** @test All live decode rows exist while original typed physical root quotes remain unchanged. */
TEST(MoEOverlayFollowerRowGeometry, EveryLiveDecodeRowUsesOriginalAdmittedBacking)
{
    const std::vector<int> prefill{16,32,64,128};
    for(const int capacity:{1,4,16,64})
    {
        CPUSerialExpertStorageEnvelope envelope(DeviceId::cpu(),128,capacity,8,2,prefill,16,8,false,
            CPUSerialExpertFamilyPolicy::PublishedSchedule);
        const auto& storage=envelope.admitted();const auto original_bytes=storage->physicalBytes();
        std::vector<std::size_t> buckets;
        for(const auto& family:storage->families())buckets.push_back(family.rows);
        const auto original_buckets=buckets;
        const auto geometry=MoEOverlayFollowerRowGeometry::declare(128,capacity,prefill,buckets);
        for(int row=1;row<=capacity;++row)
        {
            const auto* shape=geometry.forRows(row);ASSERT_NE(shape,nullptr);
            EXPECT_EQ(shape->physical_rows,row);EXPECT_GE(shape->backing_rows,std::size_t(row));
            EXPECT_EQ(shape->backing_rows,*std::lower_bound(buckets.begin(),buckets.end(),std::size_t(row)));
        }
        if(capacity>=4)
        {
            ASSERT_NE(geometry.forRows(3),nullptr);
            EXPECT_EQ(geometry.forRows(3)->physical_rows,3);
            EXPECT_EQ(geometry.forRows(3)->backing_rows,4u);
        }
        for(int row:prefill)ASSERT_NE(geometry.forRows(row),nullptr);
        EXPECT_EQ(storage->physicalBytes(),original_bytes);EXPECT_EQ(buckets,original_buckets);
        EXPECT_EQ(storage.get(),envelope.admitted().get());
        EXPECT_EQ(geometry.forRows(129),nullptr);EXPECT_EQ(geometry.forRows(0),nullptr);
    }
}
/** @test An odd source prefill bucket survives verbatim; undeclared prompt shapes are not invented. */
TEST(MoEOverlayFollowerRowGeometry, ExactDecodeAndOriginalPrefillLadderAreIndependent)
{
    const std::array<int,2> prefill{16,600};
    const std::array<std::size_t,5> storage{1,2,4,16,600};
    const auto main=MoEOverlayFollowerRowGeometry::declare(600,4,prefill,storage);
    const std::vector<MoEOverlayFollowerRowGeometry::Shape> expected{{1,1},{2,2},{3,4},{4,4},{16,16},{600,600}};
    EXPECT_EQ(main.shapes(),expected);EXPECT_EQ(main.forRows(8),nullptr);
    const auto sidecar=MoEOverlayFollowerRowGeometry::declare(1,4,prefill,storage);
    ASSERT_EQ(sidecar.shapes().size(),1u);EXPECT_EQ(sidecar.shapes().front().physical_rows,1);
}
/** @test Retained4->3->1 and partial/reset inputs traverse the real ticket protocol without rounding execution. */
TEST(MoEOverlayFollowerRowGeometry, LargeSmallPartialAndResetTicketsKeepExactRows)
{
    const std::array<std::size_t,3> storage{1,2,4};
    const auto geometry=MoEOverlayFollowerRowGeometry::declare(4,4,{},storage);
    const MoEOverlayInferenceTopologyIdentity topology{71,0x1234,0x5678,0,1};
    MoEOverlayInferenceTransactionProtocol protocol({topology,4,1,4,3});
    for(std::uint64_t generation=1;generation<=20;++generation)
    {
        const MoEOverlayInferenceCommandIdentity command{generation,generation,1};
        ASSERT_TRUE(protocol.beginCommand(command));std::uint64_t ordinal=1;
        for(const int rows:{4,3,1})
        {
            const auto* shape=geometry.forRows(rows);ASSERT_NE(shape,nullptr);
            const int logical_rows=rows==3?2:rows;
            const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology,command,ordinal,100+ordinal,1,
                rows==1?MoEOverlayInferenceGraphRole::MainDecode:MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                1,logical_rows,shape->physical_rows,rows==1?-1:logical_rows-1);
            EXPECT_EQ(ticket.physical_rows_per_request,rows);
            EXPECT_EQ(ticket.logical_rows_per_request,logical_rows);
            const auto admitted=protocol.accept(ticket);ASSERT_TRUE(admitted.accepted())<<admitted.error;
            ASSERT_TRUE(protocol.markSubmitted(admitted.slot_index,ticket));
            ASSERT_TRUE(protocol.markReturnReady(admitted.slot_index,ticket));
            ASSERT_TRUE(protocol.retire(admitted.slot_index,ticket));++ordinal;
        }
        const auto terminal=makeMoEOverlayInferenceTerminalTicket(topology,command,ordinal,1,
            MoEOverlayInferenceTransactionAction::Complete);
        EXPECT_EQ(protocol.accept(terminal).status,MoEOverlayInferenceAdmissionStatus::Complete);
        EXPECT_EQ(protocol.inFlightSlotCount(),0u);
    }
}
/** @test Invalid admission and unknown execution identities remain explicit errors with retained-shape diagnostics. */
TEST(MoEOverlayFollowerRowGeometry, MissingShapeAndInvalidBackingFailPrecisely)
{
    const std::array<int,1> prefill{64};const std::array<std::size_t,4> storage{1,2,4,64};
    EXPECT_THROW(MoEOverlayFollowerRowGeometry::declare(64,0,prefill,storage),std::invalid_argument);
    EXPECT_THROW(MoEOverlayFollowerRowGeometry::declare(65,4,prefill,storage),std::invalid_argument);
    const std::array<std::size_t,3> duplicate{1,4,4};
    EXPECT_THROW(MoEOverlayFollowerRowGeometry::declare(4,4,{},duplicate),std::invalid_argument);
    const std::array<int,2> foreign_ladder{64,16};
    EXPECT_THROW(MoEOverlayFollowerRowGeometry::declare(64,4,foreign_ladder,storage),std::invalid_argument);
    const std::array<int,3> retained{1,2,4};
    const auto diagnostic=MoEOverlayFollowerRowGeometry::missingShapeDiagnostic(3,retained);
    EXPECT_NE(diagnostic.find("physical_rows=3"),std::string::npos);
    EXPECT_NE(diagnostic.find("retained_rows=[1,2,4]"),std::string::npos);
}
}
