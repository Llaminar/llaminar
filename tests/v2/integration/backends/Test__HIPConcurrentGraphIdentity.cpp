/**
 * @file Test__HIPConcurrentGraphIdentity.cpp
 * @brief Prove that independently built HIP graphs retain every operation.
 *
 * Production TP participants construct different graphs concurrently. ROCm
 * 7.2.4 allocated diagnostic node IDs with a racy process-global increment and
 * used those IDs as the scheduler's visited set. A repeated ID could silently
 * truncate an otherwise valid graph. Collectives then waited for omitted work.
 * Event queries can report success for a marker never recorded, even though
 * attempting to measure that marker's elapsed time reports an invalid handle.
 *
 * This native-runtime regression deliberately has no model or collective: each
 * node writes a distinct result word, and every timing marker must actually be
 * recorded. Concurrent builders own disjoint graphs, streams, and test-only
 * allocations. It is not concurrent mutation of one graph. Native HIP fixtures
 * are the infrastructure boundary under test, not production allocation APIs.
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/**
 * @brief Throw an attributable native failure without stranding a test barrier.
 * @param status Native result from the immediately preceding HIP operation.
 * @param operation Stable diagnostic naming the failed lifecycle boundary.
 * @throws std::runtime_error when HIP did not report success.
 */
void requireHip(hipError_t status, const char *operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/** @brief Own one independent native transaction and all its diagnostic markers. */
class NativeGraphProof final {
public:
    /**
     * @brief Allocate setup-only native resources on the caller's exact device.
     * @param device Native ordinal owned by this independent graph builder.
     * @throws std::runtime_error on failure, after retiring partial construction.
     */
    explicit NativeGraphProof(int device) : device_(device)
    {
        try
        {
            requireHip(hipSetDevice(device_), "hipSetDevice");
            requireHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "create stream");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming), "create terminal");
            requireHip(hipMalloc(reinterpret_cast<void **>(&words_), kWords * sizeof(*words_)), "allocate words");
        }
        catch (...)
        {
            releaseResources();
            throw;
        }
    }

    /** @brief Release completed graph owners before their referenced resources. */
    ~NativeGraphProof()
    {
        releaseResources();
    }

    NativeGraphProof(const NativeGraphProof &) = delete;
    NativeGraphProof &operator=(const NativeGraphProof &) = delete;

    /**
     * @brief Build a new chain while other participants construct their graphs.
     *
     * Distinct destination words prove every node, not just the terminal one.
     * Marker pairs also cover the original invalid elapsed-time symptom.
     * @throws std::runtime_error on construction or instantiation failure.
     */
    void construct()
    {
        clearGraph();
        requireHip(hipGraphCreate(&graph_, 0), "create graph");
        hipGraphNode_t tail = nullptr;
        for (std::size_t word = 0; word < kWords; ++word)
        {
            if (word % kMarkerStride == 0) addMarker(tail);
            hipMemsetParams parameters{};
            parameters.dst = words_ + word;
            parameters.value = static_cast<unsigned>(word + 1);
            parameters.elementSize = sizeof(*words_);
            parameters.width = 1;
            parameters.height = 1;
            hipGraphNode_t node = nullptr;
            requireHip(hipGraphAddMemsetNode(&node, graph_, &tail, 1, &parameters), "add result word");
            tail = node;
        }
        addMarker(tail);
        requireHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0), "instantiate graph");
    }

    /**
     * @brief Replay the unchanged executable and verify every result and marker.
     * @throws std::runtime_error if a native call fails or any node was omitted.
     */
    void verify()
    {
        std::vector<std::uint32_t> result(kWords);
        requireHip(hipMemsetAsync(words_, 0, result.size() * sizeof(*words_), stream_), "reset words");
        requireHip(hipGraphLaunch(executable_, stream_), "launch graph");
        requireHip(hipMemcpyAsync(result.data(), words_, result.size() * sizeof(*words_),
                                  hipMemcpyDeviceToHost, stream_), "read results");
        requireHip(hipEventRecord(terminal_, stream_), "record terminal");
        // Only the terminal test readback waits. No node has a host completion
        // edge, and stream/device synchronization is unnecessary.
        requireHip(hipEventSynchronize(terminal_), "retire test readback");
        for (std::size_t word = 0; word < result.size(); ++word)
            if (result[word] != word + 1)
                throw std::runtime_error("HIP graph omitted word " + std::to_string(word) +
                                         " on device " + std::to_string(device_));
        for (hipEvent_t marker : markers_)
        {
            float elapsed = -1.0f;
            // Query alone is insufficient: HIP returns success for an event
            // never recorded. A self-interval proves it was actually executed.
            requireHip(hipEventElapsedTime(&elapsed, marker, marker), "marker must be recorded");
            if (elapsed != 0.0f) throw std::runtime_error("invalid self-interval");
        }
    }

private:
    /** @brief Retire partial construction and normal teardown through one owner. */
    void releaseResources() noexcept
    {
        (void)hipSetDevice(device_);
        clearGraph();
        if (terminal_) (void)hipEventDestroy(terminal_);
        if (words_) (void)hipFree(words_);
        if (stream_) (void)hipStreamDestroy(stream_);
        terminal_ = nullptr;
        words_ = nullptr;
        stream_ = nullptr;
    }

    /**
     * @brief Add an owned event node to the current chain without executing it.
     * @param tail Current chain end, replaced by the new marker on success.
     * @throws std::runtime_error on native event or node construction failure.
     */
    void addMarker(hipGraphNode_t &tail)
    {
        hipEvent_t event = nullptr;
        requireHip(hipEventCreate(&event), "create marker");
        markers_.push_back(event);
        hipGraphNode_t node = nullptr;
        requireHip(hipGraphAddEventRecordNode(&node, graph_, tail ? &tail : nullptr,
                                             tail ? 1 : 0, event), "add marker");
        tail = node;
    }

    /** @brief Retire a completed native graph before rebuilding the next generation. */
    void clearGraph() noexcept
    {
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (graph_) (void)hipGraphDestroy(graph_);
        executable_ = nullptr;
        graph_ = nullptr;
        for (hipEvent_t event : markers_) (void)hipEventDestroy(event);
        markers_.clear();
    }

    static constexpr std::size_t kWords = 2048;
    static constexpr std::size_t kMarkerStride = 64;
    int device_;
    hipStream_t stream_ = nullptr;
    hipEvent_t terminal_ = nullptr;
    std::uint32_t *words_ = nullptr;
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    std::vector<hipEvent_t> markers_;
};

/** @test Independent native graphs remain complete under concurrent construction. */
TEST(Test__HIPConcurrentGraphIdentity, EveryNodeAndMarkerExecutesAcrossConcurrentBuilders)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    if (devices == 0) GTEST_SKIP() << "Requires a ROCm device";

    // Several independent owners on each available device also cover one-card
    // hosts. The model's TP graph builders exercise the same process-global ID.
    constexpr std::size_t kBuilders = 16;
    constexpr std::size_t kGenerations = 16;
    std::barrier boundary(static_cast<std::ptrdiff_t>(kBuilders));
    std::array<std::string, kBuilders> failures;
    std::vector<std::thread> workers;
    for (std::size_t builder = 0; builder < kBuilders; ++builder)
    {
        workers.emplace_back([&, builder] {
            std::unique_ptr<NativeGraphProof> proof;
            try { proof = std::make_unique<NativeGraphProof>(builder % std::min(devices, 4)); }
            catch (const std::exception &error) { failures[builder] = error.what(); }
            for (std::size_t generation = 0; generation < kGenerations; ++generation)
            {
                // Failed owners still participate in both barriers, so one
                // reported omission cannot deadlock the remaining builders.
                boundary.arrive_and_wait();
                if (failures[builder].empty())
                {
                    try
                    {
                        proof->construct();
                        proof->verify();
                        proof->verify();
                    }
                    catch (const std::exception &error) { failures[builder] = error.what(); }
                }
                boundary.arrive_and_wait();
            }
        });
    }
    for (auto &worker : workers) worker.join();
    for (std::size_t builder = 0; builder < kBuilders; ++builder)
        EXPECT_TRUE(failures[builder].empty()) << "builder=" << builder << ": " << failures[builder];
}
} // namespace
