/**
 * @file Test__HIPGraphQueuePlacement.hip
 * @brief Prove retained HIP fork/join graphs do not lose available physical queues.
 *
 * HIP 7.2.4 subtracts the future launch stream when creating a graph's auxiliary
 * streams. That future stream can share a hardware queue with an auxiliary:
 * three independent branches then execute on two physical FIFOs even though
 * the bounded device pool contains four. A logical stream count is not proof
 * of concurrency. These kernels record their real queue identities, avoiding
 * wall-clock thresholds, spinning rendezvous and profiler dependencies.
 *
 * Repeated short forks exercise the pinned runtime's classic scheduler, the
 * same scheduler selected for production nested collective/FFN graphs. Native
 * allocations belong only to this runtime-infrastructure fixture. Production
 * still uses PhysicalMemoryAuthority and exact-stream TransferEngine APIs.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** @brief Shape of unrelated live stream owners before graph instantiation. */
enum class QueueReferenceDistribution { Uniform, Skewed };

/**
 * @brief Publish the executing dispatch's HSA queue identity into one owned slot.
 * @param queues Device result storage, never dereferenced as a host pointer.
 * @param slot Unique output word for this independent graph branch.
 *
 * Classic graph scheduling prepares the dispatch arguments on its assigned
 * native queue. The AMD queue-pointer builtin identifies that physical queue,
 * not the HIP logical stream. No branches communicate with each other.
 */
__global__ void recordPhysicalQueue(std::uint64_t* queues, unsigned slot)
{
    queues[slot] = reinterpret_cast<std::uint64_t>(__builtin_amdgcn_queue_ptr());
}

/**
 * @brief Fail a native setup/replay boundary with its exact HIP diagnostic.
 * @param result Result of the immediately preceding native operation.
 * @param operation Stable label identifying the failed lifecycle boundary.
 * @throws std::runtime_error when the operation did not succeed.
 */
void requireHip(hipError_t result, const char* operation)
{
    if (result != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}

/** @brief Own the graph, launch-stream cohort and terminal readback boundary. */
class QueuePlacementProof final {
public:
    /**
     * @brief Materialize a bounded hardware pool and unrelated logical owners.
     * @param distribution Whether to unbalance reference counts before capture.
     *
     * All twenty streams execute before graph construction. Destroying selected
     * owners in the skewed case leaves at least one owner of every physical
     * queue, but makes a least-reference-count-only allocation insufficient.
     * @throws std::runtime_error after retiring partial setup on native failure.
     */
    explicit QueuePlacementProof(QueueReferenceDistribution distribution)
    {
        try {
            requireHip(hipSetDevice(0), "select fixture device");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming), "create terminal");
            requireHip(hipMalloc(reinterpret_cast<void**>(&device_queues_), kResultWords * sizeof(std::uint64_t)),
                       "allocate queue identities");
            for (unsigned i = 0; i != kReplayCount; ++i) {
                hipStream_t stream = nullptr;
                requireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create unrelated stream");
                streams_.push_back(stream);
                hipLaunchKernelGGL(recordPhysicalQueue, dim3(1), dim3(1), 0, stream, device_queues_, 0);
                requireHip(hipGetLastError(), "materialize hardware queue");
                readResult(stream);
                launch_queues_.push_back(host_queues_[0]);
            }
            available_queues_ = std::set<std::uint64_t>(launch_queues_.begin(), launch_queues_.end()).size();
            if (distribution == QueueReferenceDistribution::Skewed) {
                std::set<std::uint64_t> retained;
                for (unsigned i = 0; i != streams_.size(); ++i) {
                    const bool first_owner = retained.insert(launch_queues_[i]).second;
                    if (!first_owner && launch_queues_[i] != launch_queues_.front()) {
                        requireHip(hipStreamDestroy(streams_[i]), "retire surplus unrelated owner");
                        streams_[i] = nullptr;
                    }
                }
                std::erase(streams_, nullptr);
            }
        } catch (...) {
            release();
            throw;
        }
    }

    /** @brief Retire native objects after their last terminal observation. */
    ~QueuePlacementProof() { release(); }
    QueuePlacementProof(const QueuePlacementProof&) = delete;
    QueuePlacementProof& operator=(const QueuePlacementProof&) = delete;

    /** @return Distinct materialized physical queues, not logical stream count. */
    std::size_t availableQueues() const { return available_queues_; }

    /**
     * @brief Build one retained graph with successive independent branch waves.
     * @param width Number of independent branches, between two and four.
     * @throws std::runtime_error on invalid shape or native construction failure.
     */
    void construct(unsigned width)
    {
        if (width < 2 || width > kMaximumWidth)
            throw std::runtime_error("Queue placement fixture width must be 2..4");
        retireGraph();
        width_ = width;
        requireHip(hipGraphCreate(&graph_, 0), "create retained graph");
        hipGraphNode_t predecessor = nullptr;
        requireHip(hipGraphAddEmptyNode(&predecessor, graph_, nullptr, 0), "create root");
        for (unsigned wave = 0; wave != kWaves; ++wave) {
            std::array<hipGraphNode_t, kMaximumWidth> branches{};
            for (unsigned branch = 0; branch != width_; ++branch) {
                unsigned slot = wave * width_ + branch;
                void* arguments[]{&device_queues_, &slot};
                hipKernelNodeParams params{};
                params.func = reinterpret_cast<void*>(recordPhysicalQueue);
                params.gridDim = dim3(1);
                params.blockDim = dim3(1);
                params.kernelParams = arguments;
                requireHip(hipGraphAddKernelNode(&branches[branch], graph_, &predecessor, 1, &params),
                           "add independent branch");
            }
            // Only the next wave waits. Nothing orders siblings within a wave.
            requireHip(hipGraphAddEmptyNode(&predecessor, graph_, branches.data(), width_), "join wave");
        }
        requireHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0), "instantiate retained graph");
    }

    /**
     * @brief Replay without rebuilding, including changes of the launch stream.
     *
     * Verify every branch on every wave, not just a terminal counter. The graph
     * may not satisfy the test by omitting work or reusing a prior result. Queue
     * comparisons are functional identities, not a noisy speedup gate.
     * @throws std::runtime_error on a native replay/readback failure.
     */
    void verifyTwentyLaunches()
    {
        for (unsigned replay = 0; replay != kReplayCount; ++replay) {
            const auto stream = streams_[replay % streams_.size()];
            requireHip(hipMemsetAsync(device_queues_, 0, kResultWords * sizeof(std::uint64_t), stream),
                       "poison previous identities");
            requireHip(hipGraphLaunch(executable_, stream), "replay unchanged graph");
            readResult(stream);
            for (unsigned wave = 0; wave != kWaves; ++wave) {
                SCOPED_TRACE("replay=" + std::to_string(replay) + " wave=" + std::to_string(wave));
                const auto begin = host_queues_.begin() + wave * width_;
                ASSERT_EQ(std::find(begin, begin + width_, 0), begin + width_) << "A graph branch was omitted";
                const std::set<std::uint64_t> physical(begin, begin + width_);
                ASSERT_EQ(physical.size(), width_)
                    << "Independent HIP branches share a physical FIFO despite available queues";
            }
        }
    }

private:
    /** @brief Observe one completed result using only its exact producer stream. */
    void readResult(hipStream_t stream)
    {
        requireHip(hipMemcpyAsync(host_queues_.data(), device_queues_, kResultWords * sizeof(std::uint64_t),
                                  hipMemcpyDeviceToHost, stream), "read queue identities");
        requireHip(hipEventRecord(terminal_, stream), "record terminal readback");
        // The terminal test observation is the sole host completion boundary.
        requireHip(hipEventSynchronize(terminal_), "observe terminal readback");
    }

    /** @brief Retire captured pointer/stream owners before a new graph generation. */
    void retireGraph() noexcept
    {
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (graph_) (void)hipGraphDestroy(graph_);
        executable_ = nullptr;
        graph_ = nullptr;
    }

    /** @brief Release partial setup and normal ownership through one path. */
    void release() noexcept
    {
        (void)hipSetDevice(0);
        retireGraph();
        if (terminal_) (void)hipEventDestroy(terminal_);
        if (device_queues_) (void)hipFree(device_queues_);
        terminal_ = nullptr;
        device_queues_ = nullptr;
        for (auto stream : streams_) if (stream) (void)hipStreamDestroy(stream);
        streams_.clear();
    }

    static constexpr unsigned kMaximumWidth = 4;
    static constexpr unsigned kWaves = 8;
    static constexpr unsigned kReplayCount = 20;
    static constexpr unsigned kResultWords = kMaximumWidth * kWaves;
    std::vector<hipStream_t> streams_;
    std::vector<std::uint64_t> launch_queues_;
    std::array<std::uint64_t, kResultWords> host_queues_{};
    std::size_t available_queues_ = 0;
    unsigned width_ = 0;
    hipEvent_t terminal_ = nullptr;
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    std::uint64_t* device_queues_ = nullptr;
};

/** @test Full available branch width survives launch-stream changes and skewed owners. */
TEST(Test__HIPGraphQueuePlacement, IndependentBranchesRetainAvailablePhysicalQueueWidth)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    if (devices == 0) GTEST_SKIP() << "Requires a ROCm device";
    for (const auto distribution : {QueueReferenceDistribution::Uniform, QueueReferenceDistribution::Skewed}) {
        SCOPED_TRACE(distribution == QueueReferenceDistribution::Uniform ? "uniform" : "skewed");
        QueuePlacementProof proof(distribution);
        ASSERT_GE(proof.availableQueues(), 4u) << "Requires the production four-queue HIP pool";
        for (unsigned width : {2u, 3u, 4u}) {
            SCOPED_TRACE("width=" + std::to_string(width));
            proof.construct(width);
            ASSERT_NO_FATAL_FAILURE(proof.verifyTwentyLaunches());
        }
    }
}

} // namespace
