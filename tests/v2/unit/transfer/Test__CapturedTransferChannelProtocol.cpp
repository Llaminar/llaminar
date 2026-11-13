/**
 * @file Test__CapturedTransferChannelProtocol.cpp
 * @brief Adversarial device-free specification for retained transfer-channel replay.
 *
 * Tests interleave acquire, byte work and release independently. They prove
 * backpressure, exact message identity and failure absorption without using a
 * model, stream, accelerator or a host mirror in production. GPU integrations
 * must separately prove system-scope visibility and captured byte transport.
 */
#include "transfer/CapturedTransferChannelProtocol.h"
#include <gtest/gtest.h>
#include <array>
#include <random>

using namespace llaminar2;

namespace
{
    using Protocol = CapturedTransferChannelProtocol;
    using Role = CapturedTransferEndpoint;
    using Status = CapturedTransferStatus;
    constexpr CapturedTransferChannelIdentity identity{19, 4096};
    constexpr CapturedTransferMessage message{7, 128};

    /** @brief Independent owners plus their two system-publication records. */
    struct Lane
    {
        CapturedTransferCursor producer, consumer;
        CapturedTransferPublication produced, consumed;

        /** @return One producer acquisition; the peer record remains read-only. */
        CapturedTransferDecision acquireProducer(CapturedTransferMessage input = message)
        { return Protocol::acquire(Role::Producer, identity, identity, producer, consumed, input); }
        /** @return One consumer acquisition after the modeled system acquire. */
        CapturedTransferDecision acquireConsumer(CapturedTransferMessage input = message)
        { return Protocol::acquire(Role::Consumer, identity, identity, consumer, produced, input); }
        /** @return One producer release after the modeled payload copy. */
        CapturedTransferDecision publishProducer()
        { return Protocol::publish(producer, producer.completed_epoch + 1, produced); }
        /** @return One consumer acknowledgement after its modeled payload copy. */
        CapturedTransferDecision publishConsumer()
        { return Protocol::publish(consumer, consumer.completed_epoch + 1, consumed); }
    };
}

TEST(CapturedTransferChannelProtocol, ConsumerCannotReadBeforePublication)
{
    Lane lane;
    EXPECT_EQ(lane.acquireConsumer().status, Status::WaitForPeer);
    ASSERT_TRUE(lane.acquireProducer().ready());
    for (int retry = 0; retry < 64; ++retry)
    {
        EXPECT_EQ(lane.acquireConsumer().status, Status::WaitForPeer);
        EXPECT_EQ(lane.consumer.phase, CapturedTransferPhase::Idle);
    }
    ASSERT_TRUE(lane.publishProducer().ready());
    ASSERT_TRUE(lane.acquireConsumer().ready());
    EXPECT_EQ(lane.acquireProducer().status, Status::WaitForPeer);
    ASSERT_TRUE(lane.publishConsumer().ready());
    EXPECT_TRUE(lane.acquireProducer().ready());
}

TEST(CapturedTransferChannelProtocol, ReplayAndRequestBoundariesNeverResetAnEpoch)
{
    Lane lane;
    for (std::uint64_t epoch = 1; epoch <= 4096; ++epoch)
    {
        // Different retained bucket/metadata messages share the same lifetime
        // lane. No reset or host-side generation is necessary at a new request.
        const CapturedTransferMessage input{1 + epoch % 4, 1 + epoch % identity.capacity};
        ASSERT_EQ(lane.acquireProducer(input).epoch, epoch);
        ASSERT_EQ(lane.publishProducer().epoch, epoch);
        ASSERT_EQ(lane.acquireConsumer(input).epoch, epoch);
        ASSERT_EQ(lane.publishConsumer().epoch, epoch);
        EXPECT_EQ(lane.produced.epoch, epoch);
        EXPECT_EQ(lane.consumed.epoch, epoch);
    }
}

TEST(CapturedTransferChannelProtocol, RandomizedPayloadInterleavingsKeepOneOutstandingMessage)
{
    for (unsigned seed = 0; seed < 32; ++seed)
    {
        Lane lane;
        std::mt19937 random(seed);
        std::uint64_t slot = 0, read = 0;
        enum class Step { Acquire, Payload, Publish };
        Step producer = Step::Acquire, consumer = Step::Acquire;
        for (int iteration = 0; lane.consumer.completed_epoch < 384; ++iteration)
        {
            ASSERT_LT(iteration, 100000);
            if (random() % 2 == 0 && lane.producer.completed_epoch < 384)
            {
                if (producer == Step::Acquire)
                {
                    const auto decision = lane.acquireProducer();
                    ASSERT_TRUE(decision.ready() || decision.status == Status::WaitForPeer);
                    if (decision.ready()) producer = Step::Payload;
                }
                else if (producer == Step::Payload)
                {
                    slot = lane.producer.completed_epoch + 1;
                    producer = Step::Publish;
                }
                else
                {
                    ASSERT_TRUE(lane.publishProducer().ready());
                    producer = Step::Acquire;
                }
            }
            else if (consumer == Step::Acquire)
            {
                const auto decision = lane.acquireConsumer();
                ASSERT_TRUE(decision.ready() || decision.status == Status::WaitForPeer);
                if (decision.ready()) consumer = Step::Payload;
            }
            else if (consumer == Step::Payload)
            {
                read = slot;
                ASSERT_EQ(read, lane.consumer.completed_epoch + 1);
                consumer = Step::Publish;
            }
            else
            {
                // A delayed acknowledgement must still see the same payload;
                // an eager producer overwrite would make this fail immediately.
                ASSERT_EQ(slot, read);
                ASSERT_TRUE(lane.publishConsumer().ready());
                consumer = Step::Acquire;
            }
            ASSERT_GE(lane.producer.completed_epoch, lane.consumer.completed_epoch);
            ASSERT_LE(lane.producer.completed_epoch - lane.consumer.completed_epoch, 1);
        }
    }
}

TEST(CapturedTransferChannelProtocol, RejectsStaleAndFuturePeerEpochs)
{
    for (const auto role : {Role::Producer, Role::Consumer})
        for (const auto epoch : {0u, 8u})
        {
            CapturedTransferCursor cursor{.completed_epoch = 5};
            EXPECT_EQ(Protocol::acquire(role, identity, identity, cursor,
                          CapturedTransferPublication{.epoch = epoch, .message = message}, message).status,
                      Status::EpochMismatch);
            EXPECT_EQ(cursor.phase, CapturedTransferPhase::Failed);
        }
}

TEST(CapturedTransferChannelProtocol, RejectsForeignBindingAndChangedMessageBeforePayload)
{
    for (const auto actual : {CapturedTransferChannelIdentity{20, 4096},
                              CapturedTransferChannelIdentity{19, 4095},
                              CapturedTransferChannelIdentity{}})
    {
        CapturedTransferCursor cursor;
        EXPECT_EQ(Protocol::acquire(Role::Producer, identity, actual, cursor, {}, message).status,
                  Status::InvalidBinding);
    }
    for (const auto input : {CapturedTransferMessage{0, 128}, CapturedTransferMessage{7, 0},
                             CapturedTransferMessage{7, 4097}})
    {
        Lane lane;
        EXPECT_EQ(lane.acquireProducer(input).status, Status::InvalidBinding);
    }
    for (const auto input : {CapturedTransferMessage{8, 128}, CapturedTransferMessage{7, 127},
                             CapturedTransferMessage{7, 129}})
    {
        Lane lane;
        ASSERT_TRUE(lane.acquireProducer().ready());
        ASSERT_TRUE(lane.publishProducer().ready());
        EXPECT_EQ(lane.acquireConsumer(input).status, Status::MessageMismatch);
        EXPECT_EQ(lane.consumer.completed_epoch, 0);
        EXPECT_EQ(lane.consumed.epoch, 0);
    }
}

TEST(CapturedTransferChannelProtocol, DoubleAcquireDoublePublishAndForeignLeaseAreFatal)
{
    Lane twice;
    ASSERT_TRUE(twice.acquireProducer().ready());
    EXPECT_EQ(twice.acquireProducer().status, Status::InvalidPhase);
    EXPECT_EQ(twice.publishProducer().status, Status::InvalidPhase);
    EXPECT_EQ(twice.produced.epoch, 0);
    Lane released;
    ASSERT_TRUE(released.acquireProducer().ready());
    ASSERT_TRUE(released.publishProducer().ready());
    EXPECT_EQ(released.publishProducer().status, Status::InvalidPhase);
    for (const std::uint64_t epoch : {0ull, 2ull, ~0ull})
    {
        Lane wrong;
        ASSERT_TRUE(wrong.acquireProducer().ready());
        EXPECT_EQ(Protocol::publish(wrong.producer, epoch, wrong.produced).status, Status::EpochMismatch);
        EXPECT_EQ(wrong.produced.epoch, 0);
    }
}

TEST(CapturedTransferChannelProtocol, AbortAndExhaustionAreAbsorbingInEitherDirection)
{
    for (const auto role : {Role::Producer, Role::Consumer})
    {
        CapturedTransferCursor cursor;
        CapturedTransferPublication own;
        EXPECT_EQ(Protocol::acquire(role, identity, identity, cursor,
                      CapturedTransferPublication{.epoch = kCapturedTransferAbortEpoch}, message).status,
                  Status::PeerAborted);
        Protocol::abort(cursor, own);
        EXPECT_EQ(own.epoch, kCapturedTransferAbortEpoch);
        EXPECT_EQ(Protocol::acquire(role, identity, identity, cursor, {}, message).status, Status::InvalidPhase);
        EXPECT_EQ(Protocol::publish(cursor, 1, own).status, Status::InvalidPhase);
        EXPECT_EQ(own.epoch, kCapturedTransferAbortEpoch);
        cursor = {.completed_epoch = kCapturedTransferAbortEpoch - 1};
        EXPECT_EQ(Protocol::acquire(role, identity, identity, cursor,
                      CapturedTransferPublication{.epoch = cursor.completed_epoch}, message).status,
                  Status::EpochExhausted);
    }
}

TEST(CapturedTransferChannelProtocol, CountedReplayUsesOnePublishedExtentIncludingEmptyEpochs)
{
    using Source = CapturedTransferExtentSource;
    for (unsigned seed = 0; seed < 32; ++seed)
    {
        Lane lane;
        std::mt19937 random(seed);
        const CapturedTransferMessage capacity{41, identity.capacity};
        std::array<std::uint8_t, identity.capacity> slot{}, received{};
        for (std::uint64_t epoch = 1; epoch <= 384; ++epoch)
        {
            const auto bytes = epoch % 3 == 0 ? 0 : epoch % 3 == 1 ? capacity.bytes : random() % capacity.bytes;
            const auto acquire_consumer = [&] {
                return Protocol::acquire(Role::Consumer, identity, identity, lane.consumer, lane.produced,
                    capacity, Source::ProducerPublication);
            };
            EXPECT_EQ(acquire_consumer().status, Status::WaitForPeer);
            ASSERT_TRUE(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
                capacity, Source::ProducerDevice, bytes).ready());
            ASSERT_EQ(lane.producer.acquired_message.bytes, bytes);
            for (size_t i = 0; i < bytes; ++i) slot[i] = static_cast<std::uint8_t>(epoch + i);
            ASSERT_TRUE(lane.publishProducer().ready());
            ASSERT_TRUE(acquire_consumer().ready());
            ASSERT_EQ(lane.consumer.acquired_message.bytes, bytes);
            received.fill(0xa5);
            for (size_t i = 0; i < lane.consumer.acquired_message.bytes; ++i) received[i] = slot[i];
            // A next producer, even an empty one, cannot invalidate this lease
            // before the receiver has finished consuming the current message.
            EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
                capacity, Source::ProducerDevice, 0).status, Status::WaitForPeer);
            ASSERT_TRUE(lane.publishConsumer().ready());
            for (size_t i = 0; i < received.size(); ++i)
                ASSERT_EQ(received[i], i < bytes ? static_cast<std::uint8_t>(epoch + i) : 0xa5);
            EXPECT_EQ(lane.consumed.message.bytes, bytes);
            EXPECT_EQ(lane.consumed.epoch, epoch);
        }
    }
}

TEST(CapturedTransferChannelProtocol, CountedExtentRejectsWrongAuthorityAndOverflowBeforePayload)
{
    using Source = CapturedTransferExtentSource;
    for (const auto role : {Role::Producer, Role::Consumer})
    {
        CapturedTransferCursor cursor;
        EXPECT_EQ(Protocol::acquire(role, identity, identity, cursor, {}, message,
            role == Role::Producer ? Source::ProducerPublication : Source::ProducerDevice).status, Status::InvalidBinding);
    }
    for (const auto bytes : {message.bytes + 1, kCapturedTransferAbortEpoch})
    {
        CapturedTransferCursor producer, consumer;
        EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, producer, {}, message,
            Source::ProducerDevice, bytes).status, Status::InvalidBinding);
        EXPECT_EQ(Protocol::acquire(Role::Consumer, identity, identity, consumer,
            {.epoch = 1, .message = {message.key, bytes}}, message, Source::ProducerPublication).status, Status::MessageMismatch);
        EXPECT_EQ(consumer.completed_epoch, 0u);
        EXPECT_EQ(consumer.phase, CapturedTransferPhase::Failed);
    }
    CapturedTransferCursor wrong_key;
    EXPECT_EQ(Protocol::acquire(Role::Consumer, identity, identity, wrong_key,
        {.epoch = 1, .message = {message.key + 1, 0}}, message, Source::ProducerPublication).status, Status::MessageMismatch);
}

/** @test Offsets and lengths are one acquired snapshot; channel epochs survive request data resets. */
TEST(CapturedTransferChannelProtocol, SourceSliceReplayBoundsAndEpochs)
{
    Lane lane;
    constexpr std::uint64_t span = 1024;
    for (std::uint64_t replay = 0; replay < 256; ++replay)
    {
        const auto local = replay % 16; // New request data; the channel epoch never resets.
        CapturedTransferSourceSlice slice{local * 31, local % 3 ? 1 + local * 7 : 0};
        if (!slice.live_bytes) slice.byte_offset = span; // Legal empty end-of-region.
        const auto expected = slice;
        const auto lease = Protocol::acquire(Role::Producer, identity, identity, lane.producer,
            lane.consumed, message, CapturedTransferExtentSource::ProducerDevice,
            slice.live_bytes, slice.byte_offset, span);
        ASSERT_TRUE(lease.ready()); EXPECT_EQ(lease.epoch, replay + 1);
        slice = {~std::uint64_t{0}, ~std::uint64_t{0}}; // A later writer cannot change acquired work.
        EXPECT_EQ(lane.producer.acquired_source_offset, expected.byte_offset);
        EXPECT_EQ(lane.producer.acquired_message.bytes, expected.live_bytes);
        ASSERT_TRUE(lane.publishProducer().ready());
        ASSERT_TRUE(Protocol::acquire(Role::Consumer, identity, identity, lane.consumer,
            lane.produced, message, CapturedTransferExtentSource::ProducerPublication).ready());
        EXPECT_EQ(lane.consumer.acquired_source_offset, 0u);
        EXPECT_EQ(lane.consumer.acquired_message.bytes, expected.live_bytes);
        ASSERT_TRUE(lane.publishConsumer().ready());
        EXPECT_EQ(lane.produced.epoch, replay + 1); EXPECT_EQ(lane.consumed.epoch, replay + 1);
        EXPECT_EQ(lane.producer.acquired_source_offset, 0u);
    }
}

/** @test Region overflow, foreign roles and stale peer generations fail before any payload access. */
TEST(CapturedTransferChannelProtocol, SourceSliceRejectsInvalidBoundsAndStaleEpoch)
{
    for (const auto slice : {CapturedTransferSourceSlice{1025, 0}, {1024, 1}, {1000, 25},
             {0, 129}, {~std::uint64_t{0}, 1}})
    {
        Lane lane;
        EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
            message, CapturedTransferExtentSource::ProducerDevice, slice.live_bytes, slice.byte_offset, 1024).status,
            Status::InvalidBinding);
        EXPECT_EQ(lane.producer.phase, CapturedTransferPhase::Failed);
    }
    Lane lane;
    EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
        message, CapturedTransferExtentSource::ProducerDevice, 1, 1, 127).status, Status::InvalidBinding);
    lane = {};
    EXPECT_EQ(Protocol::acquire(Role::Consumer, identity, identity, lane.consumer, lane.produced,
        message, CapturedTransferExtentSource::ProducerPublication, 0, 0, 1024).status, Status::InvalidBinding);
    lane = {}; lane.consumed.epoch = 2;
    EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
        message, CapturedTransferExtentSource::ProducerDevice, 1, 31, 1024).status, Status::EpochMismatch);
}

/** @test Original absolute bank coordinates normalize once without rewriting their source descriptor. */
TEST(CapturedTransferChannelProtocol, SourceSliceOriginalCoordinateOrigin)
{
    Lane lane;
    ASSERT_TRUE(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
        message, CapturedTransferExtentSource::ProducerDevice, 17, 4007, 1024, 4000).ready());
    EXPECT_EQ(lane.producer.acquired_source_offset, 7u);
    lane = {};
    EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
        message, CapturedTransferExtentSource::ProducerDevice, 17, 3999, 1024, 4000).status, Status::InvalidBinding);
    lane = {};
    EXPECT_EQ(Protocol::acquire(Role::Producer, identity, identity, lane.producer, lane.consumed,
        message, CapturedTransferExtentSource::ProducerDevice, 1, ~std::uint64_t{0}, 1024,
        ~std::uint64_t{0}).status, Status::InvalidBinding);
}

/** @test The native lowering cannot attach a slice to a foreign role, scalar or overflowing region. */
TEST(CapturedTransferChannelProtocol, SourceSliceBindingRejectsForeignRoleAndCountAddress)
{
    CapturedTransferChannelControl control{};
    CapturedTransferCursor cursor{};
    CapturedTransferSourceSlice slice{};
    CapturedTransferChannelDeviceBinding binding{
        .control = &control, .cursor = &cursor, .expected = identity, .message = message,
        .role = Role::Producer, .timeout_ticks = 100,
        .extent_source = CapturedTransferExtentSource::ProducerDevice,
        .device_extent = &slice.live_bytes, .source_slice = &slice, .source_span_bytes = 1024};
    EXPECT_TRUE(binding.valid());
    binding.source_span_bytes = 0; EXPECT_FALSE(binding.valid()); binding.source_span_bytes = 1024;
    binding.device_extent = &slice.byte_offset; EXPECT_FALSE(binding.valid());
    binding.device_extent = &slice.live_bytes; binding.source_offset_origin = ~std::uint64_t{0};
    EXPECT_FALSE(binding.valid()); binding.source_offset_origin = 0;
    binding.role = Role::Consumer; binding.extent_source = CapturedTransferExtentSource::ProducerPublication;
    EXPECT_FALSE(binding.valid());
}

/** @test Parent cancellation is terminal and cannot authenticate another physical owner. */
TEST(CapturedTransferChannelProtocol, OriginalCancellationNeverMasksForeignPhysicalIdentity)
{
    CapturedTransferCursor cursor{};
    const auto acquire = [&](CapturedTransferChannelIdentity expected, std::uint64_t cancelled)
    {
        return Protocol::acquire(Role::Consumer, expected, identity, cursor, {}, message,
            CapturedTransferExtentSource::FixedMessage, 0, 0, 0, 0, cancelled);
    };
    EXPECT_EQ(acquire(identity, identity.nonce).status, Status::Cancelled);
    EXPECT_EQ(cursor.phase, CapturedTransferPhase::Cancelled);
    EXPECT_EQ(cursor.completed_epoch, 0u);
    EXPECT_EQ(acquire(identity, identity.nonce).status, Status::Cancelled);
    EXPECT_EQ(acquire(identity, 0).status, Status::InvalidPhase);
    cursor = {};
    EXPECT_EQ(acquire({identity.nonce + 1, identity.capacity}, identity.nonce).status, Status::InvalidBinding);
    cursor = {};
    EXPECT_EQ(acquire(identity, identity.nonce + 1).status, Status::InvalidBinding);
}

/** @test Original faults win every cancellation interleaving through the actual native decision owner. */
TEST(CapturedTransferChannelProtocol, OriginalFaultsNeverDrainAsCancellation)
{
    for (const auto error : {Status::TimedOut, Status::InvalidPhase, Status::MessageMismatch,
             Status::InvalidBinding, Status::PeerAborted, Status::EpochMismatch})
    {
        for (const auto phase : {CapturedTransferPhase::Idle, CapturedTransferPhase::Acquired,
                 CapturedTransferPhase::Publishing, CapturedTransferPhase::Failed, CapturedTransferPhase::Cancelled})
            EXPECT_FALSE(Protocol::drainsCancellation(phase, error, true));
        CapturedTransferCursor cursor{};
        CapturedTransferPublication publication{};
        Protocol::abort(cursor, publication);
        EXPECT_EQ(Protocol::acquire(Role::Consumer, identity, identity, cursor, {}, message,
            CapturedTransferExtentSource::FixedMessage, 0, 0, 0, 0, identity.nonce).status, Status::InvalidPhase);
        EXPECT_EQ(cursor.phase, CapturedTransferPhase::Failed);
    }
    EXPECT_FALSE(Protocol::drainsCancellation(CapturedTransferPhase::Failed, Status::Cancelled, true));
    EXPECT_FALSE(Protocol::drainsCancellation(CapturedTransferPhase::Idle, Status::Cancelled, false));
    EXPECT_TRUE(Protocol::drainsCancellation(CapturedTransferPhase::Acquired, Status::Cancelled, true));
}

/** @test Actual native publisher validates original phase/message/epoch before cancellation. */
TEST(CapturedTransferChannelProtocol, PublisherFaultsPrecedeCancellationAndNeverAdvancePublication)
{
    for(const auto source:{CapturedTransferExtentSource::FixedMessage,CapturedTransferExtentSource::ProducerDevice}){
        for(int fault=0;fault<4;++fault){
            CapturedTransferCursor cursor{};cursor.phase=CapturedTransferPhase::Acquired;
            cursor.completed_epoch=5;cursor.acquired_message=message;
            CapturedTransferPublication publication{};publication.epoch=5;
            Status expected=Status::Cancelled;
            if(fault==1){cursor.phase=CapturedTransferPhase::Failed;expected=Status::InvalidPhase;}
            if(fault==2){++cursor.acquired_message.key;expected=Status::MessageMismatch;}
            if(fault==3){publication.epoch=4;expected=Status::EpochMismatch;}
            const auto epoch=publication.epoch;
            const auto result=Protocol::preparePublication(cursor,publication,message,source,identity.nonce,identity.nonce);
            EXPECT_EQ(result.status,expected);
            EXPECT_EQ(publication.epoch,epoch);EXPECT_EQ(cursor.completed_epoch,5u);
            EXPECT_EQ(Protocol::drainsCancellation(cursor.phase,result.status,true),fault==0);
        }
    }
    CapturedTransferCursor cursor{};cursor.phase=CapturedTransferPhase::Acquired;cursor.acquired_message=message;
    CapturedTransferPublication publication{};
    EXPECT_TRUE(Protocol::preparePublication(cursor,publication,message,
        CapturedTransferExtentSource::FixedMessage,0,identity.nonce).ready());
    EXPECT_EQ(publication.epoch,1u);
}
