/**
 * @file Test__MoEGroupedIntermediateExchange.cpp
 * @brief Device-free geometry and binding validation for lossless MoE exchange.
 *
 * The packet is the existing consumer representation, not a new quantizer.
 * Admission must reject truncated blocks, unknown encodings, missing planes and
 * receive-bank overflow before a graph embeds any of these pointers or sizes.
 */
#include <gtest/gtest.h>
#include "execution/moe/MoEGroupedIntermediateExchangeABI.h"

using namespace llaminar2;

/** @brief Quantized and floating geometry account for exactly their native bytes. */
TEST(MoEGroupedIntermediateExchange, GeometryPreservesExistingRepresentation)
{
    MoEGroupedIntermediateLayout layout{MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 4096, 2};
    ASSERT_TRUE(layout.valid());
    EXPECT_EQ(layout.valueWords(), 128u);
    EXPECT_EQ(layout.scaleWords(), 16u);
    EXPECT_EQ(layout.packetBytes(), 4096u * 576u);
    ASSERT_TRUE(layout.compactValid());
    EXPECT_EQ(layout.compactRecordWords(), 145u);
    EXPECT_EQ(layout.compactCapacityBytes(), 4096u * 580u);
    layout.columns = 513;
    EXPECT_FALSE(layout.valid());
    layout.encoding = MoEGroupedIntermediateEncoding::FP32;
    EXPECT_TRUE(layout.valid());
    EXPECT_EQ(layout.valueWords(), 513u);
    EXPECT_EQ(layout.scaleWords(), 0u);
    EXPECT_EQ(layout.packetBytes(), 4096u * 513u * sizeof(float));
    EXPECT_EQ(layout.compactCapacityBytes(), 4096u * 514u * sizeof(float));
}

/** @brief No invalid enum, empty plane or overflow reaches a launch descriptor. */
TEST(MoEGroupedIntermediateExchange, InvalidOrOverflowingGeometryIsRejected)
{
    const MoEGroupedIntermediateLayout valid{MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 512, 4096, 2};
    auto layout = valid;
    layout.encoding = static_cast<MoEGroupedIntermediateEncoding>(3);
    EXPECT_FALSE(layout.valid());
    layout = valid; layout.columns = 0; EXPECT_FALSE(layout.valid());
    layout = valid; layout.route_capacity = 0; EXPECT_FALSE(layout.valid());
    layout = valid; layout.participants = 0; EXPECT_FALSE(layout.valid());
    layout = valid; layout.route_capacity = 0x80000000u; EXPECT_FALSE(layout.valid());
    layout = valid; layout.participants = 0x80000000u; EXPECT_FALSE(layout.valid());
    layout = {MoEGroupedIntermediateEncoding::FP32, 0xffffffffu, 0x7fffffffu, 0x7fffffffu};
    EXPECT_FALSE(layout.valid());
    EXPECT_FALSE(layout.compactValid());
}

/** @brief Compact bindings require the existing group-end fields and an explicit device count. */
TEST(MoEGroupedIntermediateExchange, CompactBindingsRequireAuthoritativeCountsAndOwner)
{
    std::int32_t index = 0;
    std::uint32_t word = 0;
    std::uint64_t bytes = 0;
    MoEGroupedIntermediatePackLaunch payload{.layout = {MoEGroupedIntermediateEncoding::FP32, 3, 7, 2},
        .route_owners = &index, .original_to_grouped = &index, .grouped_values = &word, .packet = &word, .participant = 1};
    MoECompactIntermediatePackLaunch pack{payload, &index, &index, &bytes};
    ASSERT_TRUE(pack.valid());
    pack.packet_bytes = nullptr; EXPECT_FALSE(pack.valid());
    pack.packet_bytes = &bytes; pack.last_group_count = nullptr; EXPECT_FALSE(pack.valid());
    pack.last_group_count = &index; pack.last_group_offset = nullptr; EXPECT_FALSE(pack.valid());
    MoECompactIntermediateConsumeLaunch consumer{.layout = payload.layout, .route_owners = &index,
        .original_to_grouped = &index, .packet = &word, .packet_bytes = &bytes, .participant = 0, .grouped_values = &word};
    ASSERT_TRUE(consumer.valid());
    consumer.participant = 2; EXPECT_FALSE(consumer.valid());
    consumer.participant = 0; consumer.packet_bytes = nullptr; EXPECT_FALSE(consumer.valid());
}

/** @brief Native scales are mandatory only for the existing block-Q8 encoding. */
TEST(MoEGroupedIntermediateExchange, BindingsRejectMissingOrConflictingPlanes)
{
    std::int32_t index = 0;
    std::uint32_t word = 0;
    float scale = 1;
    MoEGroupedIntermediatePackLaunch pack{.layout = {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, 32, 1, 2},
        .route_owners = &index, .original_to_grouped = &index, .grouped_values = &word,
        .grouped_scales = &scale, .packet = &word, .participant = 0};
    ASSERT_TRUE(pack.valid());
    pack.participant = -1; EXPECT_FALSE(pack.valid());
    pack.participant = 2; EXPECT_FALSE(pack.valid());
    pack.participant = 1; EXPECT_TRUE(pack.valid());
    pack.grouped_scales = nullptr; EXPECT_FALSE(pack.valid());
    pack.layout.encoding = MoEGroupedIntermediateEncoding::FP32; EXPECT_TRUE(pack.valid());
    pack.grouped_scales = &scale; EXPECT_FALSE(pack.valid());
    pack.grouped_scales = nullptr; pack.route_owners = nullptr; EXPECT_FALSE(pack.valid());
    MoEGroupedIntermediateConsumeLaunch consume{.layout = pack.layout, .route_owners = &index,
        .original_to_grouped = &index, .participant_packets = &word, .grouped_values = &word};
    ASSERT_TRUE(consume.valid());
    consume.grouped_scales = &scale; EXPECT_FALSE(consume.valid());
    consume.layout.encoding = MoEGroupedIntermediateEncoding::BlockQ8FP32Scales; EXPECT_TRUE(consume.valid());
    consume.grouped_scales = nullptr; EXPECT_FALSE(consume.valid());
    consume.grouped_scales = &scale; consume.participant_packets = nullptr; EXPECT_FALSE(consume.valid());
}
