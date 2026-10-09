/**
 * @file Test__HIPEventProducerLifetime.hip
 * @brief Native event completion survives retirement and reuse of its producer.
 *
 * A tensor can retain an upload event after a captured stream owner retires.
 * The event owns its completion; an obsolete producer pointer cannot be used
 * to inspect capture state. These model-free ABI probes use public HIP calls,
 * real transfers, both retirement orders, concurrent queries and actual stream
 * address reuse. They deliberately keep events alive across allocator churn.
 * Native allocations and terminal waits belong to this dependency fixture;
 * no production inference synchronization or alternate execution mode is used.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/** @brief Reject a native operation at the boundary that failed. */
void requireHip(hipError_t status, const char* operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/** @brief One exact native stream, retired before the retained event by tests. */
class Stream final {
public:
    /** @brief Create an explicit nondefault producer or consumer. */
    Stream() { requireHip(hipStreamCreateWithFlags(&value_, hipStreamNonBlocking), "create stream"); }
    /** @brief Complete native owner retirement at the fixture boundary. */
    ~Stream() { if (value_) requireHip(hipStreamDestroy(value_), "destroy stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    /** @brief Access the currently owned stream handle. */
    hipStream_t get() const { return value_; }
    /** @brief Retire one generation while preserving its old identity for comparison. */
    hipStream_t retire()
    {
        const auto previous = value_;
        requireHip(hipStreamDestroy(value_), "retire producer before event");
        value_ = nullptr;
        return previous;
    }
private:
    hipStream_t value_ = nullptr;
};

/** @brief Own a reusable event independently of any producer stream. */
class Event final {
public:
    /** @brief Admit either timing or ordering completion semantics. */
    explicit Event(unsigned flags)
    { requireHip(hipEventCreateWithFlags(&value_, flags), "create retained event"); }
    /** @brief Retire the completion only after all consumers have finished. */
    ~Event() { requireHip(hipEventDestroy(value_), "destroy retained event"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    /** @brief Access the live event handle, including after producer retirement. */
    hipEvent_t get() const { return value_; }
private:
    hipEvent_t value_ = nullptr;
};

/** @brief Persistent source/result storage for exact native transfer evidence. */
class TransferReceipt final {
public:
    /** @brief Allocate a small native ABI fixture, outside any captured work. */
    TransferReceipt()
    {
        requireHip(hipMalloc(reinterpret_cast<void**>(&device_), kBytes), "allocate receipt");
        requireHip(hipHostMalloc(reinterpret_cast<void**>(&host_), kBytes), "allocate terminal receipt");
    }
    /** @brief Release the fixture after its explicit terminal observation. */
    ~TransferReceipt()
    {
        requireHip(hipHostFree(host_), "free terminal receipt");
        requireHip(hipFree(device_), "free device receipt");
    }
    /** @brief Publish distinguishable data and its event on the exact producer. */
    void publish(hipStream_t stream, hipEvent_t event, unsigned char epoch)
    {
        requireHip(hipMemsetAsync(device_, epoch, kBytes, stream), "write producer receipt");
        requireHip(hipEventRecord(event, stream), "publish exact producer event");
    }
    /** @brief Consume an event, then inspect only the terminal copy. */
    void verify(hipStream_t stream, hipEvent_t event, unsigned char epoch)
    {
        requireHip(hipStreamWaitEvent(stream, event, 0), "join retained event");
        requireHip(hipMemcpyAsync(host_, device_, kBytes, hipMemcpyDeviceToHost, stream), "copy receipt");
        requireHip(hipStreamSynchronize(stream), "observe terminal fixture receipt");
        for (std::size_t index = 0; index < kBytes; ++index)
            ASSERT_EQ(host_[index], epoch) << "receipt byte " << index;
    }
private:
    static constexpr std::size_t kBytes = 64;
    unsigned char* device_ = nullptr;
    unsigned char* host_ = nullptr;
};

/** @brief Cover timing, ordering and blocking event flags with the same lifecycle. */
class HIPEventProducerLifetime : public ::testing::TestWithParam<unsigned> {
protected:
    /** @brief Select the explicitly admitted test device. */
    void SetUp() override { requireHip(hipSetDevice(0), "select admitted device"); }
};

TEST_P(HIPEventProducerLifetime, CompletionSurvivesProducerRetirementAndRerecord)
{
    TransferReceipt receipt;
    Stream consumer;
    Event completion(GetParam());
    for (unsigned epoch = 1; epoch <= 20; ++epoch) {
        SCOPED_TRACE(epoch);
        Stream producer;
        receipt.publish(producer.get(), completion.get(), static_cast<unsigned char>(epoch));
        producer.retire();
        // Other valid host owners may occupy the old stream's allocation.
        const std::vector<std::array<std::uint64_t, 96>> churn(32);
        const auto ready = hipEventQuery(completion.get());
        ASSERT_TRUE(ready == hipSuccess || ready == hipErrorNotReady) << hipGetErrorString(ready);
        requireHip(hipEventSynchronize(completion.get()), "wait after producer retirement");
        receipt.verify(consumer.get(), completion.get(), static_cast<unsigned char>(epoch));
    }
}

TEST_P(HIPEventProducerLifetime, QueriesRaceProducerRetirementWithoutBorrowingFreedState)
{
    TransferReceipt receipt;
    Stream consumer;
    Event completion(GetParam());
    for (unsigned epoch = 1; epoch <= 20; ++epoch) {
        SCOPED_TRACE(epoch);
        Stream producer;
        receipt.publish(producer.get(), completion.get(), static_cast<unsigned char>(epoch));
        std::latch ready(1);
        std::atomic<int> failure{hipSuccess};
        std::thread observer([&] {
            const auto selected = hipSetDevice(0);
            if (selected != hipSuccess) failure.store(selected);
            ready.count_down();
            if (selected != hipSuccess) return;
            for (unsigned read = 0; read < 512; ++read) {
                const auto status = hipEventQuery(completion.get());
                if (status != hipSuccess && status != hipErrorNotReady) {
                    failure.store(status);
                    break;
                }
            }
        });
        ready.wait();
        producer.retire();
        observer.join();
        ASSERT_EQ(failure.load(), hipSuccess);
        receipt.verify(consumer.get(), completion.get(), static_cast<unsigned char>(epoch));
    }
}

TEST_P(HIPEventProducerLifetime, ReusedAddressDoesNotJoinAnUnrelatedCapture)
{
    TransferReceipt receipt;
    Stream consumer;
    Event completion(GetParam());
    unsigned reused = 0;
    for (unsigned epoch = 1; epoch <= 20; ++epoch) {
        SCOPED_TRACE(epoch);
        Stream producer;
        receipt.publish(producer.get(), completion.get(), static_cast<unsigned char>(epoch));
        const auto retired = producer.retire();
        Stream replacement;
        reused += replacement.get() == retired;
        requireHip(hipStreamBeginCapture(replacement.get(), hipStreamCaptureModeRelaxed),
                   "capture unrelated producer generation");
        const auto wait_status = hipStreamWaitEvent(consumer.get(), completion.get(), 0);
        hipGraph_t graph = nullptr;
        const auto end_status = hipStreamEndCapture(replacement.get(), &graph);
        if (graph) requireHip(hipGraphDestroy(graph), "retire unrelated capture");
        ASSERT_EQ(end_status, hipSuccess);
        ASSERT_EQ(wait_status, hipSuccess) << "old event adopted a new producer generation";
        receipt.verify(consumer.get(), completion.get(), static_cast<unsigned char>(epoch));
        // Rerecord deliberately adopts the new identity after the old event
        // remained independent of that stream's preceding capture.
        receipt.publish(replacement.get(), completion.get(), static_cast<unsigned char>(epoch + 20));
        receipt.verify(consumer.get(), completion.get(), static_cast<unsigned char>(epoch + 20));
    }
    ASSERT_GT(reused, 0U) << "native allocator did not exercise stream address reuse";
}

TEST_P(HIPEventProducerLifetime, LiveCapturedEventStillRejectsHostSynchronization)
{
    for (const bool query : {false, true}) {
        Stream producer;
        Event completion(GetParam());
        requireHip(hipStreamBeginCapture(producer.get(), hipStreamCaptureModeRelaxed), "begin capture");
        requireHip(hipEventRecord(completion.get(), producer.get()), "record captured dependency");
        const auto rejected = query ? hipEventQuery(completion.get()) : hipEventSynchronize(completion.get());
        hipGraph_t graph = nullptr;
        const auto ended = hipStreamEndCapture(producer.get(), &graph);
        if (graph) requireHip(hipGraphDestroy(graph), "retire invalidated capture");
        EXPECT_EQ(rejected, hipErrorCapturedEvent);
        EXPECT_EQ(ended, hipErrorStreamCaptureInvalidated);
    }
}

INSTANTIATE_TEST_SUITE_P(NativeEventFlags, HIPEventProducerLifetime,
    ::testing::Values(static_cast<unsigned>(hipEventDefault),
                      static_cast<unsigned>(hipEventDisableTiming),
                      static_cast<unsigned>(hipEventBlockingSync)));
} // namespace
