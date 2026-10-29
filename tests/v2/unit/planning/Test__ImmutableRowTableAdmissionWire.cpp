/**
 * @file Test__ImmutableRowTableAdmissionWire.cpp
 * @brief Device-free source/rank/eligibility and exact-mode BOM startup-wire regressions.
 *
 * No model, backend, allocation or engine DSO is needed. Invalid sender frames
 * fail before a final placement/certificate can exist.
 */
#include "planning/ImmutableRowTableAdmissionWire.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::test
{
namespace
{
/** @return One rank's exact original source and unequal PLE-consumer resources. */
ImmutableRowTableAdmissionRequest request()
{
    const PhysicalMemoryResource host{2,DeviceId::cpu(),1u<<30,1u<<29};
    const std::array consumers{PhysicalMemoryResource{2,DeviceId::cuda(0),1u<<30,1u<<29},
        PhysicalMemoryResource{2,DeviceId::rocm(1),1u<<29,1u<<28}};
    auto input=ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),1024,4096),host,consumers,ImmutableRowTableHostPlacement::onNode(3));
    PhysicalMemoryPlanBuilder hosted;
    hosted.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,128u)
        .add(consumers[0],PhysicalMemoryOwner::ActivationArena,256u)
        .add(consumers[1],PhysicalMemoryOwner::NativeGraphExecutable,512u);
    return {{ImmutableRowStoragePreference::Automatic,65536,3},std::move(input),
        {PhysicalMemoryPlan{},hosted.build(),hosted.build()}};
}
}
/** @test Native source extent, exact host/consumers and eligible-empty VRAM survive setup agreement. */
TEST(ImmutableRowTableAdmissionWire, RoundTripPreservesSourceConsumersAndExactModeBOM)
{
    const std::array requests{request()};
    const auto bytes=serializeImmutableRowTableAdmissionRequests(requests);
    const auto decoded=parseImmutableRowTableAdmissionRequests(bytes,2);
    ASSERT_EQ(decoded.size(),1u);
    EXPECT_EQ(decoded.front().requested,requests.front().requested);
    EXPECT_EQ(decoded.front().input.extent(),requests.front().input.extent());
    ASSERT_EQ(decoded.front().input.consumers().size(),2u);
    EXPECT_EQ(decoded.front().input.host().device,DeviceId::cpu());
    EXPECT_EQ(decoded.front().input.hostPlacement(),ImmutableRowTableHostPlacement::onNode(3));
    ASSERT_TRUE(decoded.front().contributions.vram);
    EXPECT_TRUE(decoded.front().contributions.vram->resources().empty());
    EXPECT_EQ(decoded.front().contributions.ram.summary(),requests.front().contributions.ram.summary());
    EXPECT_EQ(decoded.front().contributions.mmap.summary(),requests.front().contributions.mmap.summary());
}
/** @test Missing original upload geometry remains ineligible instead of becoming an empty eligible plan. */
TEST(ImmutableRowTableAdmissionWire, MissingOriginalSlotCannotBecomeEligibleAcrossWire)
{
    auto missing=request();missing.contributions.vram.reset();
    const std::array requests{missing};
    const auto decoded=parseImmutableRowTableAdmissionRequests(serializeImmutableRowTableAdmissionRequests(requests),2);
    ASSERT_EQ(decoded.size(),1u);EXPECT_FALSE(decoded.front().contributions.vram);
}
/** @test Schema/rank/version/truncation failures reject exact frames, including integral enum truncation attacks. */
TEST(ImmutableRowTableAdmissionWire, ForeignVersionTruncationTrailingAndEnumAliasesAreRejected)
{
    const std::array requests{request()};
    const auto valid=serializeImmutableRowTableAdmissionRequests(requests);
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(valid,3),std::invalid_argument);
    for(std::size_t n=0;n<valid.size();++n)
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(std::span(valid).first(n),2),std::invalid_argument) << n;
    auto bad=valid;bad.push_back(0);
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(bad,2),std::invalid_argument);
    bad=valid;bad[4]=99;
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(bad,2),std::invalid_argument);
    bad=valid;bad[16]=0;bad[17]=1; // preference256 must not truncate to Automatic0.
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(bad,2),std::invalid_argument);
    bad=valid;bad[160]=255;bad[161]=255; // invalid physical backend must not truncate into a supported type.
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(bad,2),std::invalid_argument);
}
/** @test A rank without PLE contributes a real empty frame; duplicate rank authority is never merged. */
TEST(ImmutableRowTableAdmissionWire, EmptyAndDuplicateSenderContractsAreExplicit)
{
    const auto empty=serializeImmutableRowTableAdmissionRequests({});
    EXPECT_TRUE(parseImmutableRowTableAdmissionRequests(empty,2).empty());
    const std::array duplicate{request(),request()};
    EXPECT_THROW(serializeImmutableRowTableAdmissionRequests(duplicate),std::invalid_argument);
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(empty,-1),std::invalid_argument);
}
}

namespace llaminar2::test
{
/** @test Whole-host affinity is explicit task policy; invalid sentinels never become node0. */
TEST(ImmutableRowTableAdmissionWire, HostPlacementSeparatesAffinityFromAllocator)
{
    const auto task=ImmutableRowTableHostPlacement::fromObservedNUMANode(-1);
    EXPECT_TRUE(task.usesTaskPolicy()); EXPECT_FALSE(task.numaNode());
    EXPECT_EQ(ImmutableRowTableHostPlacement::fromObservedNUMANode(3).numaNode(),3);
    EXPECT_NE(task,ImmutableRowTableHostPlacement::onNode(0));
    EXPECT_THROW(ImmutableRowTableHostPlacement::fromObservedNUMANode(-2),std::invalid_argument);
    EXPECT_THROW(ImmutableRowTableHostPlacement::onNode(-1),std::invalid_argument);
    auto original=request();
    original.input=ImmutableRowTablePlacementInput::forConsumers(original.input.extent(),
        original.input.host(),original.input.consumers(),task);
    const std::array values{original};
    const auto decoded=parseImmutableRowTableAdmissionRequests(serializeImmutableRowTableAdmissionRequests(values),2);
    ASSERT_EQ(decoded.size(),1u); EXPECT_EQ(decoded.front().input.hostPlacement(),task);
    EXPECT_EQ(decoded.front().input.host().device,DeviceId::cpu());
}

/** @test Locality wire arms are canonical and version1 cannot silently discard affinity. */
TEST(ImmutableRowTableAdmissionWire, LocalityWireRejectsInvalidOrDiscardedPolicy)
{
    const std::array values{request()};
    const auto valid=serializeImmutableRowTableAdmissionRequests(values);
    ASSERT_GT(valid.size(),124u);
    for (const auto mutation : {0,1,2,3})
    {
        auto bytes=valid;
        if(mutation==0)bytes[4]=1; // Previous schema has no locality authority.
        if(mutation==1)bytes[184]=2; // Unknown locality arm.
        if(mutation==2)bytes[184]=0; // Task arm cannot retain a bound-node payload.
        if(mutation==3)for(int i=0;i<4;++i)bytes[188+i]=255; // Unrepresentable node.
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(bytes,2),std::invalid_argument)<<mutation;
    }
}
}

namespace llaminar2::test
{
/** @test Sender readiness freezes exact live bytes and rejects local invalid/foreign contributions. */
TEST(ImmutableRowTableAdmissionWire, PreparedSenderFrameAuthenticatesBeforeCollective)
{
    const std::array values{request()};
    const auto frame = ImmutableRowTableAdmissionSendFrame::prepare(values, 2);
    const auto encoded = serializeImmutableRowTableAdmissionRequests(values);
    EXPECT_EQ(frame.senderRank(), 2);
    EXPECT_EQ(frame.byteCount(), static_cast<int>(encoded.size()));
    EXPECT_EQ(std::vector<std::uint8_t>(frame.bytes().begin(), frame.bytes().end()), encoded);
    ASSERT_EQ(parseImmutableRowTableAdmissionRequests(frame.bytes(), 2).size(), 1u);
    const std::array malformed{request(), request()};
    EXPECT_THROW((void)ImmutableRowTableAdmissionSendFrame::prepare(malformed, 2), std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableAdmissionSendFrame::prepare(values, 3), std::invalid_argument);
    EXPECT_THROW((void)ImmutableRowTableAdmissionSendFrame::prepare({}, -1), std::invalid_argument);
    const auto empty = ImmutableRowTableAdmissionSendFrame::prepare({}, 4);
    EXPECT_GT(empty.byteCount(), 0);
    EXPECT_TRUE(parseImmutableRowTableAdmissionRequests(empty.bytes(), 4).empty());
}
} // namespace llaminar2::test
