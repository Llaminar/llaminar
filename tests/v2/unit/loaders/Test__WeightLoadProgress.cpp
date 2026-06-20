/**
 * @file Test__WeightLoadProgress.cpp
 * @brief Unit tests for WeightLoadProgress and IProgressPublisher integration
 *
 * Tests cover:
 * 1. WeightLoadProgress construction and enabled state
 * 2. Device label formatting (single-rank vs multi-rank)
 * 3. Device registration with and without aggregator
 * 4. Progress updates routed to aggregator
 * 5. Finish events routed to aggregator
 * 6. makeCallback() logic (enabled/disabled/aggregator combos)
 * 7. Thread safety of concurrent updates
 * 8. Edge cases: max devices, zero-byte devices, OOB indices
 * 9. Failing aggregator handling
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

#include "loaders/WeightLoadProgress.h"
#include "loaders/IProgressPublisher.h"
#include "../../mocks/MockProgressPublisher.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightLoadProgress : public ::testing::Test
{
protected:
    // Suppress stderr output during tests (progress renders to stderr)
    void SetUp() override
    {
        // Redirect stderr to /dev/null during tests
        saved_stderr_ = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
    }

    void TearDown() override
    {
        // Restore stderr
        if (saved_stderr_ >= 0)
        {
            dup2(saved_stderr_, STDERR_FILENO);
            close(saved_stderr_);
        }
    }

    int saved_stderr_ = -1;
};

// ============================================================================
// Construction and State Tests
// ============================================================================

TEST_F(Test__WeightLoadProgress, Rank0IsEnabled)
{
    WeightLoadProgress progress(/*rank=*/0, /*world_size=*/2);
    EXPECT_TRUE(progress.isEnabled());
}

TEST_F(Test__WeightLoadProgress, NonRank0IsDisabled)
{
    WeightLoadProgress progress(/*rank=*/1, /*world_size=*/2);
    EXPECT_FALSE(progress.isEnabled());
}

TEST_F(Test__WeightLoadProgress, SingleRankIsEnabled)
{
    WeightLoadProgress progress(/*rank=*/0, /*world_size=*/1);
    EXPECT_TRUE(progress.isEnabled());
}

TEST_F(Test__WeightLoadProgress, DefaultConstructorIsRank0SingleRank)
{
    WeightLoadProgress progress;
    EXPECT_TRUE(progress.isEnabled());
}

// ============================================================================
// Device Label Formatting
// ============================================================================

TEST_F(Test__WeightLoadProgress, SingleRankOmitsRankPrefix)
{
    WeightLoadProgress progress(/*rank=*/0, /*world_size=*/1);
    EXPECT_EQ(progress.makeDeviceLabel("rocm:0"), "rocm:0");
    EXPECT_EQ(progress.makeDeviceLabel("cuda:1"), "cuda:1");
    EXPECT_EQ(progress.makeDeviceLabel("cpu"), "cpu");
}

TEST_F(Test__WeightLoadProgress, MultiRankAddsRankPrefix)
{
    WeightLoadProgress progress0(/*rank=*/0, /*world_size=*/2);
    EXPECT_EQ(progress0.makeDeviceLabel("rocm:0"), "0:rocm:0");
    EXPECT_EQ(progress0.makeDeviceLabel("cpu"), "0:cpu");

    WeightLoadProgress progress1(/*rank=*/1, /*world_size=*/2);
    EXPECT_EQ(progress1.makeDeviceLabel("rocm:1"), "1:rocm:1");
    EXPECT_EQ(progress1.makeDeviceLabel("cuda:0"), "1:cuda:0");
}

TEST_F(Test__WeightLoadProgress, FourRanksLabelFormatting)
{
    WeightLoadProgress progress3(/*rank=*/3, /*world_size=*/4);
    EXPECT_EQ(progress3.makeDeviceLabel("rocm:2"), "3:rocm:2");
}

// ============================================================================
// Device Registration (No Aggregator)
// ============================================================================

TEST_F(Test__WeightLoadProgress, RegisterDeviceReturnsSequentialIndices)
{
    WeightLoadProgress progress(0, 1);
    int idx0 = progress.registerDevice("cuda:0", 1024);
    int idx1 = progress.registerDevice("cuda:1", 2048);
    int idx2 = progress.registerDevice("cpu", 512);
    EXPECT_EQ(idx0, 0);
    EXPECT_EQ(idx1, 1);
    EXPECT_EQ(idx2, 2);
}

TEST_F(Test__WeightLoadProgress, RegisterDeviceOnDisabledRank)
{
    WeightLoadProgress progress(1, 2); // Not rank 0
    int idx = progress.registerDevice("rocm:0", 4096);
    EXPECT_EQ(idx, 0); // Still returns valid index
}

// ============================================================================
// Device Registration (With Aggregator)
// ============================================================================

TEST_F(Test__WeightLoadProgress, RegisterDevicePublishesToAggregator)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(1, 2);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("1:rocm:0", 8192);
    EXPECT_EQ(idx, 0);

    auto regs = mock->registrations();
    ASSERT_EQ(regs.size(), 1u);
    EXPECT_EQ(regs[0].label, "1:rocm:0");
    EXPECT_EQ(regs[0].total_bytes, 8192u);
    EXPECT_EQ(regs[0].returned_idx, 0);
}

TEST_F(Test__WeightLoadProgress, MultipleDevicesPublishAllToAggregator)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    progress.registerDevice("0:cuda:0", 1000);
    progress.registerDevice("0:cuda:1", 2000);
    progress.registerDevice("0:cpu", 3000);

    auto regs = mock->registrations();
    ASSERT_EQ(regs.size(), 3u);
    EXPECT_EQ(regs[0].total_bytes, 1000u);
    EXPECT_EQ(regs[1].total_bytes, 2000u);
    EXPECT_EQ(regs[2].total_bytes, 3000u);
}

// ============================================================================
// Progress Updates
// ============================================================================

TEST_F(Test__WeightLoadProgress, UpdatePublishesToAggregator)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(1, 2);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("1:rocm:0", 10000);
    progress.update(idx, 5000);

    auto updates = mock->progressUpdates();
    ASSERT_GE(updates.size(), 1u);
    EXPECT_EQ(updates.back().local_device_idx, 0); // aggregator idx, not local idx
    EXPECT_EQ(updates.back().bytes_loaded, 5000u);
}

TEST_F(Test__WeightLoadProgress, UpdateWithInvalidIndexIsNoOp)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(mock);

    // No devices registered
    progress.update(-1, 1000);
    progress.update(0, 1000);
    progress.update(99, 1000);

    EXPECT_EQ(mock->progressUpdateCount(), 0u);
}

TEST_F(Test__WeightLoadProgress, ProgressUpdatesAreMonotonic)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("cuda:0", 10000);

    // Simulate loading
    for (size_t bytes = 1000; bytes <= 10000; bytes += 1000)
    {
        progress.update(idx, bytes);
    }

    auto updates = mock->progressUpdates();
    ASSERT_GE(updates.size(), 1u);

    // Verify monotonically increasing
    size_t prev = 0;
    for (const auto &u : updates)
    {
        EXPECT_GE(u.bytes_loaded, prev);
        prev = u.bytes_loaded;
    }
}

// ============================================================================
// Finish Events
// ============================================================================

TEST_F(Test__WeightLoadProgress, FinishPublishesToAggregator)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(1, 2);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("1:cpu", 4096);
    progress.finish(idx);

    auto events = mock->finishEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].local_device_idx, 0); // aggregator idx
}

TEST_F(Test__WeightLoadProgress, FinishWithInvalidIndexIsNoOp)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(mock);

    progress.finish(-1);
    progress.finish(0); // No device registered
    progress.finish(99);

    EXPECT_EQ(mock->finishEventCount(), 0u);
}

TEST_F(Test__WeightLoadProgress, FinishMultipleDevices)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    int idx0 = progress.registerDevice("0:cuda:0", 1000);
    int idx1 = progress.registerDevice("0:cuda:1", 2000);

    progress.finish(idx0);
    progress.finish(idx1);

    auto events = mock->finishEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].local_device_idx, 0);
    EXPECT_EQ(events[1].local_device_idx, 1);
}

// ============================================================================
// makeCallback() Logic
// ============================================================================

TEST_F(Test__WeightLoadProgress, MakeCallbackReturnsNullWhenDisabledNoAggregator)
{
    WeightLoadProgress progress(1, 2); // Not rank 0, no aggregator
    int idx = progress.registerDevice("1:cpu", 1000);
    auto cb = progress.makeCallback(idx);
    EXPECT_EQ(cb, nullptr);
}

TEST_F(Test__WeightLoadProgress, MakeCallbackReturnsNonNullWhenEnabled)
{
    WeightLoadProgress progress(0, 1); // Rank 0
    int idx = progress.registerDevice("cuda:0", 1000);
    auto cb = progress.makeCallback(idx);
    EXPECT_NE(cb, nullptr);
}

TEST_F(Test__WeightLoadProgress, MakeCallbackReturnsNonNullWhenAggregatorAttached)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(1, 2); // Not rank 0
    progress.setAggregator(mock);

    int idx = progress.registerDevice("1:rocm:0", 5000);
    auto cb = progress.makeCallback(idx);
    EXPECT_NE(cb, nullptr);
}

TEST_F(Test__WeightLoadProgress, MakeCallbackReturnsNullForInvalidIndex)
{
    WeightLoadProgress progress(0, 1);
    auto cb = progress.makeCallback(-1);
    EXPECT_EQ(cb, nullptr);
}

TEST_F(Test__WeightLoadProgress, CallbackInvokesUpdateOnProgress)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("0:cuda:0", 10000);
    auto cb = progress.makeCallback(idx);
    ASSERT_NE(cb, nullptr);

    cb(5000, 10000);

    auto updates = mock->progressUpdates();
    ASSERT_GE(updates.size(), 1u);
    EXPECT_EQ(updates.back().bytes_loaded, 5000u);
}

// ============================================================================
// Aggregator Attachment
// ============================================================================

TEST_F(Test__WeightLoadProgress, SetAggregatorIsRecoverable)
{
    auto mock1 = std::make_shared<MockProgressPublisher>();
    auto mock2 = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);

    progress.setAggregator(mock1);
    EXPECT_EQ(progress.aggregator(), mock1);

    progress.setAggregator(mock2);
    EXPECT_EQ(progress.aggregator(), mock2);
}

TEST_F(Test__WeightLoadProgress, NullAggregatorIsValid)
{
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(nullptr);
    EXPECT_EQ(progress.aggregator(), nullptr);

    // Should still work without crashing
    int idx = progress.registerDevice("cpu", 1000);
    progress.update(idx, 500);
    progress.finish(idx);
}

// ============================================================================
// Thread Safety
// ============================================================================

TEST_F(Test__WeightLoadProgress, ConcurrentUpdatesAreThreadSafe)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("0:cuda:0", 100000);

    constexpr int NUM_THREADS = 8;
    constexpr int UPDATES_PER_THREAD = 1000;
    std::atomic<int> started{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&, t]()
                             {
            started.fetch_add(1);
            while (started.load() < NUM_THREADS)
                ; // Spin until all threads ready

            for (int i = 0; i < UPDATES_PER_THREAD; ++i)
            {
                size_t bytes = static_cast<size_t>((t * UPDATES_PER_THREAD + i) * 10);
                progress.update(idx, bytes);
            } });
    }

    for (auto &th : threads)
        th.join();

    // Verify no crash and updates were recorded
    EXPECT_GE(mock->progressUpdateCount(), 1u);
}

TEST_F(Test__WeightLoadProgress, ConcurrentRegisterAndUpdateAreThreadSafe)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    constexpr int NUM_DEVICES = 4;
    std::vector<std::thread> threads;
    std::atomic<int> indices[NUM_DEVICES];
    for (auto &i : indices)
        i.store(-1);

    // Register devices from separate threads
    for (int d = 0; d < NUM_DEVICES; ++d)
    {
        threads.emplace_back([&, d]()
                             {
            std::string label = "dev:" + std::to_string(d);
            int idx = progress.registerDevice(label, 1000 * (d + 1));
            indices[d].store(idx);

            // Immediately start updating
            for (int i = 0; i < 100; ++i)
            {
                progress.update(indices[d].load(), i * 10);
            }
            progress.finish(indices[d].load()); });
    }

    for (auto &th : threads)
        th.join();

    // All devices registered
    EXPECT_EQ(mock->registrationCount(), NUM_DEVICES);
    // All devices finished
    EXPECT_EQ(mock->finishEventCount(), NUM_DEVICES);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__WeightLoadProgress, ZeroBytesDeviceHandled)
{
    WeightLoadProgress progress(0, 1);
    int idx = progress.registerDevice("empty", 0);
    EXPECT_EQ(idx, 0);

    // Should not crash on zero-byte device
    progress.update(idx, 0);
    progress.finish(idx);
}

TEST_F(Test__WeightLoadProgress, VeryLargeByteCountsHandled)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(mock);

    size_t huge = 100ULL * 1024 * 1024 * 1024; // 100 GB
    int idx = progress.registerDevice("cuda:0", huge);
    progress.update(idx, huge / 2);
    progress.finish(idx);

    auto updates = mock->progressUpdates();
    ASSERT_GE(updates.size(), 1u);
    EXPECT_EQ(updates.back().bytes_loaded, huge / 2);
}

TEST_F(Test__WeightLoadProgress, FailingAggregatorDeviceRegistration)
{
    auto failing = std::make_shared<FailingProgressPublisher>();
    WeightLoadProgress progress(1, 2);
    progress.setAggregator(failing);

    // publishDevice returns -1, aggregator_idx will be -1
    int idx = progress.registerDevice("1:rocm:0", 5000);
    EXPECT_EQ(idx, 0); // Local index is still valid

    // Updates should be no-ops since aggregator_idx is -1
    progress.update(idx, 2500);
    progress.finish(idx);
    // No crash — graceful handling of failed aggregator slot
}

// ============================================================================
// IProgressPublisher Interface Contract Tests
// ============================================================================

class Test__IProgressPublisher : public ::testing::Test
{
};

TEST_F(Test__IProgressPublisher, MockImplementsInterface)
{
    std::unique_ptr<IProgressPublisher> publisher = std::make_unique<MockProgressPublisher>();
    int idx = publisher->publishDevice("test", 1024);
    EXPECT_EQ(idx, 0);

    publisher->publishProgress(idx, 512);
    publisher->publishFinished(idx);
}

TEST_F(Test__IProgressPublisher, SequentialDeviceIndices)
{
    MockProgressPublisher mock;
    EXPECT_EQ(mock.publishDevice("dev0", 100), 0);
    EXPECT_EQ(mock.publishDevice("dev1", 200), 1);
    EXPECT_EQ(mock.publishDevice("dev2", 300), 2);
    EXPECT_EQ(mock.publishDevice("dev3", 400), 3);
}

TEST_F(Test__IProgressPublisher, ProgressUpdatesRecorded)
{
    MockProgressPublisher mock;
    mock.publishDevice("dev0", 1000);
    mock.publishProgress(0, 100);
    mock.publishProgress(0, 500);
    mock.publishProgress(0, 999);

    auto updates = mock.progressUpdates();
    ASSERT_EQ(updates.size(), 3u);
    EXPECT_EQ(updates[0].bytes_loaded, 100u);
    EXPECT_EQ(updates[1].bytes_loaded, 500u);
    EXPECT_EQ(updates[2].bytes_loaded, 999u);
}

TEST_F(Test__IProgressPublisher, FinishEventsRecorded)
{
    MockProgressPublisher mock;
    mock.publishDevice("dev0", 1000);
    mock.publishDevice("dev1", 2000);
    mock.publishFinished(0);
    mock.publishFinished(1);

    auto events = mock.finishEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].local_device_idx, 0);
    EXPECT_EQ(events[1].local_device_idx, 1);
}

TEST_F(Test__IProgressPublisher, ResetClearsAllState)
{
    MockProgressPublisher mock;
    mock.publishDevice("dev0", 1000);
    mock.publishProgress(0, 500);
    mock.publishFinished(0);

    mock.reset();
    EXPECT_EQ(mock.registrationCount(), 0u);
    EXPECT_EQ(mock.progressUpdateCount(), 0u);
    EXPECT_EQ(mock.finishEventCount(), 0u);
}

// ============================================================================
// Aggregator DeviceSlot Layout Tests (verifies memory layout for MPI transfer)
// ============================================================================

#include "loaders/WeightLoadProgressAggregator.h"

class Test__AggregatorSlotLayout : public ::testing::Test
{
};

TEST_F(Test__AggregatorSlotLayout, DeviceSlotSizeIs72Bytes)
{
    EXPECT_EQ(sizeof(WeightLoadProgressAggregator::DeviceSlot), 72u);
}

TEST_F(Test__AggregatorSlotLayout, RankWindowSizeIsMaxDevicesTimesSlotSize)
{
    constexpr size_t expected = WeightLoadProgressAggregator::MAX_DEVICES_PER_RANK *
                                sizeof(WeightLoadProgressAggregator::DeviceSlot);
    EXPECT_EQ(sizeof(WeightLoadProgressAggregator::RankWindow), expected);
}

TEST_F(Test__AggregatorSlotLayout, DeviceSlotIsTriviallyCopyable)
{
    EXPECT_TRUE(std::is_trivially_copyable_v<WeightLoadProgressAggregator::DeviceSlot>);
}

TEST_F(Test__AggregatorSlotLayout, RankWindowIsTriviallyCopyable)
{
    EXPECT_TRUE(std::is_trivially_copyable_v<WeightLoadProgressAggregator::RankWindow>);
}

TEST_F(Test__AggregatorSlotLayout, DeviceSlotAlignment)
{
    EXPECT_EQ(alignof(WeightLoadProgressAggregator::DeviceSlot), 8u);
}

TEST_F(Test__AggregatorSlotLayout, LabelCapacityIs48)
{
    EXPECT_EQ(WeightLoadProgressAggregator::LABEL_LEN, 48);
}

TEST_F(Test__AggregatorSlotLayout, MaxDevicesPerRankIs4)
{
    EXPECT_EQ(WeightLoadProgressAggregator::MAX_DEVICES_PER_RANK, 4);
}

TEST_F(Test__AggregatorSlotLayout, SlotFieldOffsets)
{
    // Verify field ordering matches MPI transfer expectations
    WeightLoadProgressAggregator::DeviceSlot slot{};
    const auto base = reinterpret_cast<uintptr_t>(&slot);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot.bytes_loaded) - base, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot.total_bytes) - base, 8u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot.active) - base, 16u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot.finished) - base, 20u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot.label) - base, 24u);
}

// ============================================================================
// Simulated Multi-Rank Aggregator Test (without real MPI)
// ============================================================================

/**
 * @brief Simulates the aggregator polling logic using in-memory RankWindows.
 *
 * This tests the data interpretation logic that pollOnce() performs,
 * without requiring an actual MPI runtime.
 */
class Test__AggregatorPollingLogic : public ::testing::Test
{
protected:
    using DeviceSlot = WeightLoadProgressAggregator::DeviceSlot;
    using RankWindow = WeightLoadProgressAggregator::RankWindow;

    /// Simulate what a remote rank writes to its window
    void simulateRemoteDevice(RankWindow &window, int device_idx,
                              const std::string &label, size_t total, size_t loaded,
                              bool finished)
    {
        auto &slot = window.devices[device_idx];
        slot.total_bytes = total;
        slot.bytes_loaded = loaded;
        slot.active = 1;
        slot.finished = finished ? 1 : 0;
        std::strncpy(slot.label, label.c_str(), WeightLoadProgressAggregator::LABEL_LEN - 1);
        slot.label[WeightLoadProgressAggregator::LABEL_LEN - 1] = '\0';
    }
};

TEST_F(Test__AggregatorPollingLogic, EmptyRemoteWindowHasNoActiveDevices)
{
    RankWindow window{};
    std::memset(&window, 0, sizeof(window));

    for (int d = 0; d < WeightLoadProgressAggregator::MAX_DEVICES_PER_RANK; ++d)
    {
        EXPECT_EQ(window.devices[d].active, 0u);
    }
}

TEST_F(Test__AggregatorPollingLogic, SimulatedRemoteDeviceIsDiscoverable)
{
    RankWindow window{};
    std::memset(&window, 0, sizeof(window));

    simulateRemoteDevice(window, 0, "1:rocm:0", 20000000000ULL, 10000000000ULL, false);

    EXPECT_EQ(window.devices[0].active, 1u);
    EXPECT_EQ(window.devices[0].total_bytes, 20000000000ULL);
    EXPECT_EQ(window.devices[0].bytes_loaded, 10000000000ULL);
    EXPECT_EQ(window.devices[0].finished, 0u);
    EXPECT_STREQ(window.devices[0].label, "1:rocm:0");
}

TEST_F(Test__AggregatorPollingLogic, FinishedDeviceHasFinishedFlag)
{
    RankWindow window{};
    std::memset(&window, 0, sizeof(window));

    simulateRemoteDevice(window, 0, "1:cuda:0", 5000, 5000, true);

    EXPECT_EQ(window.devices[0].finished, 1u);
    EXPECT_EQ(window.devices[0].bytes_loaded, 5000u);
}

TEST_F(Test__AggregatorPollingLogic, MultipleDevicesPerRank)
{
    RankWindow window{};
    std::memset(&window, 0, sizeof(window));

    simulateRemoteDevice(window, 0, "1:cuda:0", 10000, 5000, false);
    simulateRemoteDevice(window, 1, "1:cuda:1", 10000, 7000, false);
    simulateRemoteDevice(window, 2, "1:cpu", 2000, 2000, true);

    EXPECT_EQ(window.devices[0].active, 1u);
    EXPECT_EQ(window.devices[1].active, 1u);
    EXPECT_EQ(window.devices[2].active, 1u);
    EXPECT_EQ(window.devices[3].active, 0u); // Unused slot
}

TEST_F(Test__AggregatorPollingLogic, LabelTruncationAt48Bytes)
{
    RankWindow window{};
    std::memset(&window, 0, sizeof(window));

    // Label longer than LABEL_LEN-1 characters
    std::string long_label(100, 'X');
    simulateRemoteDevice(window, 0, long_label, 1000, 500, false);

    // Should be truncated, not overflow
    EXPECT_EQ(std::strlen(window.devices[0].label),
              static_cast<size_t>(WeightLoadProgressAggregator::LABEL_LEN - 1));
}

// ============================================================================
// Integration: WeightLoadProgress + MockAggregator Full Flow
// ============================================================================

class Test__WeightLoadProgressIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        saved_stderr_ = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0)
        {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
    }

    void TearDown() override
    {
        if (saved_stderr_ >= 0)
        {
            dup2(saved_stderr_, STDERR_FILENO);
            close(saved_stderr_);
        }
    }

    int saved_stderr_ = -1;
};

TEST_F(Test__WeightLoadProgressIntegration, FullLoadingSequenceRank0)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 2);
    progress.setAggregator(mock);

    // Simulate loading a model on rank 0
    int idx = progress.registerDevice(progress.makeDeviceLabel("rocm:0"), 20ULL * 1024 * 1024 * 1024);

    // Simulate 10 progress updates
    for (int i = 1; i <= 10; ++i)
    {
        size_t bytes = (20ULL * 1024 * 1024 * 1024) * i / 10;
        progress.update(idx, bytes);
    }

    progress.finish(idx);
    progress.finalize();

    // Verify aggregator received the full sequence
    EXPECT_EQ(mock->registrationCount(), 1u);
    auto regs = mock->registrations();
    EXPECT_EQ(regs[0].label, "0:rocm:0");
    EXPECT_EQ(regs[0].total_bytes, 20ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(mock->finishEventCount(), 1u);
}

TEST_F(Test__WeightLoadProgressIntegration, FullLoadingSequenceRank1NoRender)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(1, 2); // Non-rendering rank
    progress.setAggregator(mock);

    int idx = progress.registerDevice(progress.makeDeviceLabel("rocm:1"), 5000);

    // Simulate loading
    for (size_t b = 500; b <= 5000; b += 500)
    {
        progress.update(idx, b);
    }

    progress.finish(idx);

    // Aggregator still receives everything even though rendering is disabled
    EXPECT_EQ(mock->registrationCount(), 1u);
    EXPECT_GE(mock->progressUpdateCount(), 1u);
    EXPECT_EQ(mock->finishEventCount(), 1u);
}

TEST_F(Test__WeightLoadProgressIntegration, MultiDeviceMultiRankSimulation)
{
    // Simulates what happens in a real TP=2 scenario:
    // Rank 0 has cuda:0, Rank 1 has cuda:1
    // Both publish to their local aggregator window

    auto mock0 = std::make_shared<MockProgressPublisher>();
    auto mock1 = std::make_shared<MockProgressPublisher>();

    WeightLoadProgress progress0(0, 2);
    WeightLoadProgress progress1(1, 2);
    progress0.setAggregator(mock0);
    progress1.setAggregator(mock1);

    // Rank 0 registers and loads
    int idx0 = progress0.registerDevice("0:cuda:0", 10000);
    auto cb0 = progress0.makeCallback(idx0);
    ASSERT_NE(cb0, nullptr);

    // Rank 1 registers and loads
    int idx1 = progress1.registerDevice("1:cuda:1", 10000);
    auto cb1 = progress1.makeCallback(idx1);
    ASSERT_NE(cb1, nullptr); // Non-null because aggregator is attached

    // Simulate concurrent loading
    cb0(5000, 10000);
    cb1(7000, 10000);

    // Both aggregators got their updates
    EXPECT_GE(mock0->progressUpdateCount(), 1u);
    EXPECT_GE(mock1->progressUpdateCount(), 1u);

    progress0.finish(idx0);
    progress1.finish(idx1);

    EXPECT_EQ(mock0->finishEventCount(), 1u);
    EXPECT_EQ(mock1->finishEventCount(), 1u);
}

TEST_F(Test__WeightLoadProgressIntegration, CallbackStressTest)
{
    auto mock = std::make_shared<MockProgressPublisher>();
    WeightLoadProgress progress(0, 1);
    progress.setAggregator(mock);

    int idx = progress.registerDevice("cuda:0", 1000000);
    auto cb = progress.makeCallback(idx);
    ASSERT_NE(cb, nullptr);

    // Simulate rapid callback from GPU pipeline (thousands of weight chunks)
    constexpr int NUM_CHUNKS = 10000;
    size_t chunk_size = 1000000 / NUM_CHUNKS;
    for (int i = 0; i < NUM_CHUNKS; ++i)
    {
        cb(chunk_size * (i + 1), 1000000);
    }

    progress.finish(idx);

    // All publishes went to aggregator (throttling is for rendering only)
    EXPECT_GE(mock->progressUpdateCount(), static_cast<size_t>(NUM_CHUNKS));
    EXPECT_EQ(mock->finishEventCount(), 1u);
}
