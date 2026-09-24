/**
 * @file Test__HIPMultiStreamGraphReplayLifetime.hip
 * @brief Guards the host lifetime of repeated wide HIP graph launches.
 *
 * ROCm releases before 7.2.4 appended every launch's internal parallel
 * streams to a graph-owned vector without clearing its logical size.  A wide
 * retained graph therefore consumed host memory on every replay even though
 * the graph topology and executable were unchanged.  Production decode uses
 * exactly this retained, multistream shape, so a small request could leak
 * hundreds of KiB without creating or recapturing a graph.
 *
 * This test constructs independent graph branches so HIP must assign internal
 * streams, warms every lazy runtime path, and then proves that thousands of
 * replays have bounded live host allocation.  It also verifies the device
 * result so a runtime cannot satisfy the lifetime assertion by dropping graph
 * work.
 */

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <malloc.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

/** @brief Number of independent branches used to force HIP parallel streams. */
constexpr std::size_t kBranchCount = 8;

/** @brief Nodes in each branch, representative of a wide retained subgraph. */
constexpr std::size_t kNodesPerBranch = 64;

/** @brief Replays used only to settle one-time HIP allocator state. */
constexpr std::size_t kWarmupReplays = 128;

/** @brief Replays over which per-launch retained state would become obvious. */
constexpr std::size_t kMeasuredReplays = 4096;

/**
 * @brief Maximum tolerated post-warmup growth in glibc live allocations.
 *
 * This is deliberately larger than ordinary allocator bookkeeping noise and
 * much smaller than the stream-table growth produced by the defective HIP
 * runtime for this graph and replay count.
 */
constexpr std::size_t kMaximumLiveGrowthBytes = 128U * 1024U;

/** @brief Increment one device counter for each graph node that really ran. */
__global__ void incrementReplayCounter(std::uint32_t *counter)
{
    atomicAdd(counter, 1U);
}

/** @brief Fail the test immediately when a HIP operation is unsuccessful. */
void requireHipSuccess(hipError_t status, const char *operation)
{
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 hipGetErrorString(status));
    }
}

/**
 * @brief Own every HIP object used by the graph-replay lifetime proof.
 *
 * Destruction is best-effort because GoogleTest assertions may return early.
 * Keeping cleanup in one RAII owner prevents a failed regression from
 * poisoning the next preflight test on the same worker.
 */
class ReplayFixture final {
public:
    /** @brief Release graph objects in reverse dependency order. */
    ~ReplayFixture()
    {
        if (terminal_event_ != nullptr)
            (void)hipEventDestroy(terminal_event_);
        if (executable_ != nullptr)
            (void)hipGraphExecDestroy(executable_);
        if (graph_ != nullptr)
            (void)hipGraphDestroy(graph_);
        if (device_counter_ != nullptr)
            (void)hipFree(device_counter_);
        if (launch_stream_ != nullptr)
            (void)hipStreamDestroy(launch_stream_);
    }

    ReplayFixture(const ReplayFixture &) = delete;
    ReplayFixture &operator=(const ReplayFixture &) = delete;

    /** @brief Create a wide graph whose branches require internal HIP streams. */
    ReplayFixture()
    {
        requireHipSuccess(hipSetDevice(0), "hipSetDevice");
        requireHipSuccess(
            hipStreamCreateWithFlags(&launch_stream_, hipStreamNonBlocking),
            "hipStreamCreateWithFlags");
        requireHipSuccess(
            hipMalloc(reinterpret_cast<void **>(&device_counter_),
                      sizeof(*device_counter_)),
            "hipMalloc");
        requireHipSuccess(
            hipMemsetAsync(device_counter_, 0, sizeof(*device_counter_),
                           launch_stream_),
            "hipMemsetAsync");
        requireHipSuccess(hipGraphCreate(&graph_, 0), "hipGraphCreate");

        void *kernel_arguments[] = {&device_counter_};
        hipKernelNodeParams parameters{};
        parameters.func = reinterpret_cast<void *>(incrementReplayCounter);
        parameters.gridDim = dim3(1);
        parameters.blockDim = dim3(1);
        parameters.kernelParams = kernel_arguments;

        // Each branch is serial internally but independent of every other
        // branch.  HIP must therefore allocate and reuse parallel streams.
        std::array<hipGraphNode_t, kBranchCount> tails{};
        for (std::size_t branch = 0; branch < kBranchCount; ++branch) {
            for (std::size_t node = 0; node < kNodesPerBranch; ++node) {
                hipGraphNode_t next = nullptr;
                const hipGraphNode_t *dependencies =
                    tails[branch] != nullptr ? &tails[branch] : nullptr;
                const std::size_t dependency_count =
                    tails[branch] != nullptr ? 1U : 0U;
                requireHipSuccess(
                    hipGraphAddKernelNode(&next, graph_, dependencies,
                                          dependency_count, &parameters),
                    "hipGraphAddKernelNode");
                tails[branch] = next;
            }
        }

        requireHipSuccess(
            hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
            "hipGraphInstantiate");
        requireHipSuccess(
            hipEventCreateWithFlags(&terminal_event_, hipEventDisableTiming),
            "hipEventCreateWithFlags");
    }

    /** @brief Replay and retire the retained transaction exactly @p count times. */
    void replay(std::size_t count)
    {
        for (std::size_t iteration = 0; iteration < count; ++iteration) {
            requireHipSuccess(hipGraphLaunch(executable_, launch_stream_),
                             "hipGraphLaunch");
            requireHipSuccess(hipEventRecord(terminal_event_, launch_stream_),
                             "hipEventRecord");
            // This is an isolated lifetime test, not production inference.
            // Retirement before sampling the host allocator makes the
            // ownership assertion deterministic without changing the graph.
            requireHipSuccess(hipEventSynchronize(terminal_event_),
                             "hipEventSynchronize");
        }
    }

    /** @return The completed device counter copied after an explicit event edge. */
    std::uint32_t readCounter()
    {
        std::uint32_t host_counter = 0;
        requireHipSuccess(
            hipMemcpyAsync(&host_counter, device_counter_, sizeof(host_counter),
                           hipMemcpyDeviceToHost, launch_stream_),
            "hipMemcpyAsync");
        requireHipSuccess(hipEventRecord(terminal_event_, launch_stream_),
                         "hipEventRecord(readCounter)");
        requireHipSuccess(hipEventSynchronize(terminal_event_),
                         "hipEventSynchronize(readCounter)");
        return host_counter;
    }

private:
    hipStream_t launch_stream_ = nullptr;
    std::uint32_t *device_counter_ = nullptr;
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    hipEvent_t terminal_event_ = nullptr;
};

/** @return Bytes currently owned by live glibc heap allocations. */
std::size_t liveHostAllocationBytes()
{
    return mallinfo2().uordblks;
}

/**
 * @test A retained multistream HIP graph has bounded host state across replay.
 */
TEST(Test__HIPMultiStreamGraphReplayLifetime,
     RepeatedLaunchDoesNotRetainPerLaunchStreamAssignments)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count == 0)
        GTEST_SKIP() << "No ROCm device is available";

    ReplayFixture fixture;
    fixture.replay(kWarmupReplays);
    const std::size_t before = liveHostAllocationBytes();

    fixture.replay(kMeasuredReplays);
    const std::size_t after = liveHostAllocationBytes();
    const std::size_t growth = after > before ? after - before : 0U;

    EXPECT_LE(growth, kMaximumLiveGrowthBytes)
        << "HIP retained host state across identical graph launches; before="
        << before << " after=" << after;

    constexpr std::uint32_t nodes_per_replay =
        static_cast<std::uint32_t>(kBranchCount * kNodesPerBranch);
    constexpr std::uint32_t expected = nodes_per_replay *
        static_cast<std::uint32_t>(kWarmupReplays + kMeasuredReplays);
    EXPECT_EQ(fixture.readCounter(), expected);
}

} // namespace
