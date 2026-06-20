#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"

namespace llaminar2
{

TEST(Test__PinnedRingBuffer, ConstructorStoresParams)
{
    PinnedRingBuffer buf(4096, 3);
    EXPECT_EQ(buf.slotSize(), 4096u);
    EXPECT_EQ(buf.numSlots(), 3);
    EXPECT_FALSE(buf.isAllocated());
}

TEST(Test__PinnedRingBuffer, AcquireReturnsValidPointer)
{
    PinnedRingBuffer buf(1024, 2);
    ASSERT_TRUE(buf.allocate());
    EXPECT_TRUE(buf.isAllocated());

    int slot_idx = -1;
    void* ptr = buf.acquireSlot(slot_idx);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(slot_idx, 0);
}

TEST(Test__PinnedRingBuffer, AdvanceWrapsAround)
{
    PinnedRingBuffer buf(512, 3);
    ASSERT_TRUE(buf.allocate());

    int idx;
    // Slot 0
    void* p0 = buf.acquireSlot(idx);
    EXPECT_EQ(idx, 0);

    buf.advance();
    // Slot 1
    void* p1 = buf.acquireSlot(idx);
    EXPECT_EQ(idx, 1);
    EXPECT_NE(p0, p1);

    buf.advance();
    // Slot 2
    void* p2 = buf.acquireSlot(idx);
    EXPECT_EQ(idx, 2);
    EXPECT_NE(p1, p2);

    buf.advance();
    // Wrap back to slot 0
    void* p3 = buf.acquireSlot(idx);
    EXPECT_EQ(idx, 0);
    EXPECT_EQ(p0, p3);
}

TEST(Test__PinnedRingBuffer, SlotIndexIsCorrect)
{
    PinnedRingBuffer buf(256, 4);
    ASSERT_TRUE(buf.allocate());

    for (int expected = 0; expected < 4; ++expected)
    {
        int idx = -1;
        buf.acquireSlot(idx);
        EXPECT_EQ(idx, expected);
        buf.advance();
    }
}

TEST(Test__PinnedRingBuffer, GetSlotByIndex)
{
    PinnedRingBuffer buf(1024, 3);
    ASSERT_TRUE(buf.allocate());

    void* s0 = buf.getSlot(0);
    void* s1 = buf.getSlot(1);
    void* s2 = buf.getSlot(2);

    EXPECT_NE(s0, nullptr);
    EXPECT_NE(s1, nullptr);
    EXPECT_NE(s2, nullptr);

    // Slots should be offset by slot_size
    auto diff01 = static_cast<uint8_t*>(s1) - static_cast<uint8_t*>(s0);
    auto diff12 = static_cast<uint8_t*>(s2) - static_cast<uint8_t*>(s1);
    EXPECT_EQ(diff01, 1024);
    EXPECT_EQ(diff12, 1024);

    // Out of bounds returns nullptr
    EXPECT_EQ(buf.getSlot(-1), nullptr);
    EXPECT_EQ(buf.getSlot(3), nullptr);
}

TEST(Test__PinnedRingBuffer, NotAllocatedReturnsNull)
{
    PinnedRingBuffer buf(1024, 2);

    int idx = -1;
    void* ptr = buf.acquireSlot(idx);
    EXPECT_EQ(ptr, nullptr);

    EXPECT_EQ(buf.getSlot(0), nullptr);
}

TEST(Test__PinnedRingBuffer, MoveConstructor)
{
    PinnedRingBuffer buf(2048, 3);
    ASSERT_TRUE(buf.allocate());

    int idx;
    void* original_ptr = buf.acquireSlot(idx);
    EXPECT_NE(original_ptr, nullptr);

    // Move construct
    PinnedRingBuffer moved(std::move(buf));
    EXPECT_TRUE(moved.isAllocated());
    EXPECT_EQ(moved.slotSize(), 2048u);
    EXPECT_EQ(moved.numSlots(), 3);

    void* moved_ptr = moved.acquireSlot(idx);
    EXPECT_EQ(moved_ptr, original_ptr);

    // Original should be invalidated
    EXPECT_FALSE(buf.isAllocated());
}

} // namespace llaminar2
