/**
 * @file Test__ImmutableRowTableAdmissionWire.cpp
 * @brief Device-free source/rank/eligibility and exact-mode BOM startup-wire regressions.
 *
 * No model, backend, allocation or engine DSO is needed. Invalid sender frames
 * fail before a final placement/certificate can exist.
 */
#include "planning/ImmutableRowTableAdmissionWire.h"
#include <algorithm>
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
    bad=valid;bad[176]=255;bad[177]=255; // v16 adds explicit backing; physical backend must not truncate.
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
    // Discover the locality words from two actual serializer outputs, rather
    // than pin an offset that changes when preceding optional arms evolve.
    auto task = values.front();
    task.input = ImmutableRowTablePlacementInput::forConsumers(task.input.extent(),
        task.input.host(), task.input.consumers(), ImmutableRowTableHostPlacement::taskPolicy());
    const auto task_bytes = serializeImmutableRowTableAdmissionRequests(std::array{task});
    ASSERT_EQ(valid.size(), task_bytes.size());
    std::vector<std::size_t> changed;
    for (std::size_t i = 0; i < valid.size(); ++i)
        if (valid[i] != task_bytes[i]) changed.push_back(i);
    ASSERT_EQ(changed.size(), 2u);
    const auto arm = changed[0], node = changed[1];
    ASSERT_GE(valid.size(), node + sizeof(std::uint32_t));
    EXPECT_EQ(parseImmutableRowTableAdmissionRequests(task_bytes, 2).front().input.hostPlacement(),
              ImmutableRowTableHostPlacement::taskPolicy());
    for (const auto mutation : {0,1,2,3})
    {
        auto bytes=valid;
        if(mutation==0)bytes[4]=1; // Previous schema has no locality authority.
        if(mutation==1)bytes[arm]=2; // Unknown locality arm.
        if(mutation==2)bytes[arm]=0; // Task arm cannot retain a bound-node payload.
        if(mutation==3)for(int i=0;i<4;++i)bytes[node+i]=255; // Unrepresentable node.
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

namespace llaminar2::test {
/** @test Exact host backing survives canonical source agreement; unknown backing cannot cross its schema. */
TEST(ImmutableRowTableAdmissionWire, AdaptiveFullRAMBackingRoundTripsWithoutHostCacheAlias){
    auto original=request();original.requested.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    original.requested.gpu_adaptive.source_backing=AdaptiveImmutableRowBacking::RAM;
    const std::array requests{original};const auto decoded=parseImmutableRowTableAdmissionRequests(serializeImmutableRowTableAdmissionRequests(requests),2);
    ASSERT_EQ(decoded.size(),1u);EXPECT_EQ(decoded.front().requested,original.requested);
    auto old=serializeImmutableRowTableAdmissionRequests(requests);old[4]=15;
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(old,2),std::invalid_argument);
    for(const std::uint32_t raw:{2u,99u,256u,257u,UINT32_MAX}){
        auto foreign=serializeImmutableRowTableAdmissionRequests(requests);
        // The source-backing field follows the exact v16 header, window and
        // host-policy words. Mutate all four bytes, not an already-narrowed enum.
        for(unsigned byte=0;byte<4;++byte)foreign[72+byte]=static_cast<std::uint8_t>(raw>>(8*byte));
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(foreign,2),std::invalid_argument)<<raw;
    }
    original.requested.gpu_adaptive.source_backing=static_cast<AdaptiveImmutableRowBacking>(99);
    EXPECT_THROW(serializeImmutableRowTableAdmissionRequests(std::array{original}),std::invalid_argument);
}
}

namespace llaminar2::test {
/** @test Hostile full-width placement values cannot alias the uint8 ConsumerLocal arm. */
TEST(ImmutableRowTableAdmissionWire, PlacementModeWireRejectsFullWidthAliases){
    const std::array requests{request()};const auto valid=serializeImmutableRowTableAdmissionRequests(requests);
    for(const std::uint32_t raw:{2u,99u,256u,257u,UINT32_MAX}){
        auto foreign=valid;
        for(unsigned byte=0;byte<4;++byte)foreign[140+byte]=static_cast<std::uint8_t>(raw>>(8*byte));
        EXPECT_THROW(parseImmutableRowTableAdmissionRequests(foreign,2),std::invalid_argument)<<raw;
    }
}
}

namespace llaminar2::test {
/** @test New wire keeps explicit local C0 distinct, and rejects prior schema and foreign reader overrides. */
TEST(ImmutableRowTableAdmissionWire, NgramLocalHotCacheIntentAndConsumerAuthority)
{
    auto original = request();
    original.requested.storage = ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    original.requested.gpu_adaptive.placement = parseAdaptiveGPUCachePlacement("2:rocm:3/128/1048576@2:cuda:0");
    const auto host = original.input.host();
    const std::array consumers{PhysicalMemoryResource{2,DeviceId::cuda(0),1u<<30,1u<<29},
        PhysicalMemoryResource{2,DeviceId::cuda(1),1u<<30,1u<<29}};
    const std::array resources{consumers[0], consumers[1], PhysicalMemoryResource{2,DeviceId::rocm(3),1u<<30,1u<<29}};
    original.requested.gpu_adaptive.local_hot_cache.emplace();
    original.requested.gpu_adaptive.local_hot_cache->consumer_overrides = {{{2,DeviceType::CUDA,1},17,4096}};
    original.input = ImmutableRowTablePlacementInput::forConsumers(original.input.extent(),host,consumers)
        .withGPUCacheResources(original.requested,resources);
    const auto encoded = serializeImmutableRowTableAdmissionRequests(std::array{original});
    const auto decoded = parseImmutableRowTableAdmissionRequests(encoded,2);
    ASSERT_EQ(decoded.size(),1u);
    EXPECT_EQ(decoded.front().requested,original.requested);
    ASSERT_TRUE(decoded.front().requested.gpu_adaptive.local_hot_cache);
    EXPECT_EQ(decoded.front().requested.gpu_adaptive.local_hot_cache->maximum_cache_rows,0u);
    auto old = encoded; old[4]=16;
    EXPECT_THROW(parseImmutableRowTableAdmissionRequests(old,2),std::invalid_argument);
    original.requested.gpu_adaptive.local_hot_cache->consumer_overrides.front().participant.ordinal=7;
    EXPECT_THROW(original.input.withGPUCacheResources(original.requested,resources),std::invalid_argument);
}
}
