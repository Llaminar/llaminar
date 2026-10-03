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
