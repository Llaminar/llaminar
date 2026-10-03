/**
 * @file Test__HIPGraphLaunchOrdering.hip
 * @brief Prove exact entry and exit frontiers of retained multi-root HIP graphs.
 *
 * A graph launch is ordered after earlier work on its exact launch stream.
 * That contract applies to every independent root, including a child's roots,
 * not only the root assigned to the launch stream's hardware queue. A missing
 * fan-out can silently bypass a producer event and race model state or another
 * collective using the same communicator.
 *
 * The delayed producer makes that edge observable without a profiler, host
 * callback, rendezvous or per-replay host wait. Every root has single-writer
 * sticky receipts, so a later good replay cannot conceal an earlier violation.
 * A successor also checks the complete exit frontier after uneven roots reuse
 * auxiliary FIFOs. Dependency depth is not a queue's last submitted command.
 * Native allocations are limited to this runtime-infrastructure fixture;
 * production storage remains owned by PhysicalMemoryAuthority/TransferEngine.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned kLanes = 64;
constexpr unsigned kReplays = 20;
constexpr unsigned long long kProducerDelayTicks = 5000000;

/** @brief One graph root's private, persistent ordering receipt. */
struct EntryReceipt {
    std::uint32_t observed_epoch;
    std::uint32_t visits;
    std::uint32_t errors;
};

/**
 * @brief Publish a GPU-owned epoch after bounded unrelated producer work.
 * @param epoch Persistent single-writer producer state.
 *
 * This diagnostic delay does not poll another participant. The graph must
 * consume its existing stream/event dependency; root kernels remain independent
 * of each other. Only the sole producer changes the epoch between launches.
 */
__global__ void advanceEpoch(std::uint32_t* epoch)
{
    if (threadIdx.x == 0) {
        const auto begin = clock64();
        while (clock64() - begin < kProducerDelayTicks) {}
        ++*epoch;
    }
}

/**
 * @brief Verify the producer frontier independently on each graph root.
 * @param epoch GPU-owned epoch published by the preceding producer.
 * @param receipts Private result rows for the actual graph width.
 * @param branch Exact root-owned result row.
 */
__global__ void observeEpoch(const std::uint32_t* epoch, EntryReceipt* receipts,
                            unsigned branch)
{
    auto& receipt = receipts[branch * kLanes + threadIdx.x];
    const auto observed = *epoch;
    receipt.errors |= observed != receipt.visits + 1;
    receipt.observed_epoch = observed;
    ++receipt.visits;
}

/**
 * @brief Finish one independent root after bounded diagnostic work.
 * @param receipts One single-writer completion word per actual root.
 * @param branch Exact root-owned completion slot.
 * @param delayed_branch The one root whose tail ordering is being checked.
 *
 * No root waits on another root. The delay makes an incorrectly joined FIFO
 * observable without timing thresholds or a deadlocking GPU rendezvous.
 */
__global__ void finishIndependentRoot(std::uint32_t* receipts, unsigned branch,
                                     unsigned delayed_branch)
{
    if (threadIdx.x == 0) {
        if (branch == delayed_branch) {
            const auto begin = clock64();
            while (clock64() - begin < 60000000ull) {}
        }
        receipts[branch] = 1;
    }
}

/**
 * @brief Record the exit violation before any terminal host readback.
 * @param receipts Root-owned completion words.
 * @param missing One successor-owned sticky result per lane.
 * @param width Exact live root count, not allocation capacity.
 */
__global__ void observeGraphExit(const std::uint32_t* receipts,
                                std::uint32_t* missing, unsigned width)
{
    unsigned unfinished = 0;
    for (unsigned branch = 0; branch != width; ++branch)
        unfinished += receipts[branch] != 1;
    missing[threadIdx.x] = unfinished;
}

/**
 * @brief Attribute each native error to its exact setup or replay operation.
 * @param result Native result of the immediately preceding operation.
 * @param operation Stable diagnostic naming that boundary.
 * @throws std::runtime_error if the native operation failed.
 */
void requireHip(hipError_t result, const char* operation)
{
    if (result != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}

/** @brief Three public entry contracts, with no implicit/default stream. */
enum class EntryFrontier { LaunchStreamProducer, ExternalProducerEvent, ParentProducerNode };

/** @brief Own one immutable graph and all setup-only stream/event/storage. */
class EntryOrderingProof final {
public:
    /**
     * @brief Construct a retained graph whose roots share only the entry frontier.
     * @param width Actual independent root count, including physical queue reuse.
     * @param frontier Exact producer relationship being verified.
     * @throws std::runtime_error after retiring partial setup on native failure.
     */
    EntryOrderingProof(unsigned width, EntryFrontier frontier)
        : width_(width), frontier_(frontier), host_receipts_(width * kLanes)
    {
        try {
            requireHip(hipSetDevice(0), "select ordering-proof device");
            for (auto& stream : streams_)
                requireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create exact stream");
            for (auto& event : events_)
                requireHip(hipEventCreateWithFlags(&event, hipEventDisableTiming), "create exact event");
            requireHip(hipMalloc(reinterpret_cast<void**>(&epoch_), sizeof(*epoch_)), "allocate epoch");
            requireHip(hipMalloc(reinterpret_cast<void**>(&receipts_), receiptBytes()), "allocate root receipts");
            requireHip(hipMemsetAsync(epoch_, 0, sizeof(*epoch_), streams_[0]), "initialize epoch");
            requireHip(hipMemsetAsync(receipts_, 0, receiptBytes(), streams_[0]), "initialize root receipts");
            requireHip(hipEventRecord(events_[2], streams_[0]), "publish fixture initialization");
            observeTerminal();
            construct();
        } catch (...) {
            release();
            throw;
        }
    }

    /** @brief Retire native ownership after the terminal observation. */
    ~EntryOrderingProof() { release(); }
    EntryOrderingProof(const EntryOrderingProof&) = delete;
    EntryOrderingProof& operator=(const EntryOrderingProof&) = delete;

    /**
     * @brief Replay without host observations, buffer rebinding or recapture.
     * @throws std::runtime_error on native failure or an entry-order violation.
     */
    void verify()
    {
        for (unsigned replay = 0; replay != kReplays; ++replay) {
            if (frontier_ == EntryFrontier::ExternalProducerEvent) {
                if (replay != 0)
                    requireHip(hipStreamWaitEvent(streams_[1], events_[1], 0), "order the next producer");
                hipLaunchKernelGGL(advanceEpoch, dim3(1), dim3(kLanes), 0, streams_[1], epoch_);
                requireHip(hipGetLastError(), "launch external producer");
                requireHip(hipEventRecord(events_[0], streams_[1]), "publish external producer");
                requireHip(hipStreamWaitEvent(streams_[0], events_[0], 0), "consume exact producer event");
            } else if (frontier_ == EntryFrontier::LaunchStreamProducer) {
                hipLaunchKernelGGL(advanceEpoch, dim3(1), dim3(kLanes), 0, streams_[0], epoch_);
                requireHip(hipGetLastError(), "launch same-stream producer");
            }
            requireHip(hipGraphLaunch(executable_, streams_[0]), "launch retained multi-root graph");
            if (frontier_ == EntryFrontier::ExternalProducerEvent)
                requireHip(hipEventRecord(events_[1], streams_[0]), "publish preceding graph completion");
        }
        requireHip(hipMemcpyAsync(host_receipts_.data(), receipts_, receiptBytes(),
                                  hipMemcpyDeviceToHost, streams_[0]), "read terminal root receipts");
        requireHip(hipEventRecord(events_[2], streams_[0]), "publish terminal readback");
        observeTerminal();
        for (unsigned branch = 0; branch != width_; ++branch) {
            for (unsigned lane = 0; lane != kLanes; ++lane) {
                const auto& receipt = host_receipts_[branch * kLanes + lane];
                if (receipt.visits != kReplays || receipt.observed_epoch != kReplays || receipt.errors != 0)
                    throw std::runtime_error("Graph root bypassed its entry frontier: width=" +
                                             std::to_string(width_) + " branch=" + std::to_string(branch) +
                                             " lane=" + std::to_string(lane) + " visits=" +
                                             std::to_string(receipt.visits) + " observed_epoch=" +
                                             std::to_string(receipt.observed_epoch) + " errors=" +
                                             std::to_string(receipt.errors));
            }
        }
    }

private:
    /** @return Exact live result extent, with no maximum-width padding. */
    std::size_t receiptBytes() const { return host_receipts_.size() * sizeof(EntryReceipt); }

    /**
     * @brief Add a native kernel with already-owned immutable argument storage.
     * @param graph Exact graph owning the node.
     * @param function Exact kernel entry.
     * @param arguments Values copied by native node construction.
     * @return The newly owned graph node.
     */
    hipGraphNode_t addKernel(hipGraph_t graph, void* function, void** arguments)
    {
        hipKernelNodeParams parameters{};
        parameters.func = function;
        parameters.gridDim = dim3(1);
        parameters.blockDim = dim3(kLanes);
        parameters.kernelParams = arguments;
        hipGraphNode_t node = nullptr;
        requireHip(hipGraphAddKernelNode(&node, graph, nullptr, 0, &parameters), "add retained root kernel");
        return node;
    }

    /** @brief Build independent roots, optionally nested after a parent producer. */
    void construct()
    {
        requireHip(hipGraphCreate(&graph_, 0), "create retained root graph");
        for (unsigned branch = 0; branch != width_; ++branch) {
            void* arguments[]{&epoch_, &receipts_, &branch};
            (void)addKernel(graph_, reinterpret_cast<void*>(observeEpoch), arguments);
        }
        hipGraph_t executable_graph = graph_;
        if (frontier_ == EntryFrontier::ParentProducerNode) {
            requireHip(hipGraphCreate(&parent_, 0), "create retained parent graph");
            void* arguments[]{&epoch_};
            const auto producer = addKernel(parent_, reinterpret_cast<void*>(advanceEpoch), arguments);
            hipGraphNode_t child = nullptr;
            requireHip(hipGraphAddChildGraphNode(&child, parent_, &producer, 1, graph_), "bind exact child predecessor");
            executable_graph = parent_;
        }
        requireHip(hipGraphInstantiate(&executable_, executable_graph, nullptr, nullptr, 0), "instantiate immutable graph");
    }

    /** @brief Observe only setup/final completion, with the standard bounded allowance. */
    void observeTerminal()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            const auto result = hipEventQuery(events_[2]);
            if (result == hipSuccess) return;
            if (result != hipErrorNotReady) requireHip(result, "observe terminal event");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Graph entry-order proof stalled before terminal publication");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /** @brief Release setup-only owners in graph-before-storage order. */
    void release() noexcept
    {
        (void)hipSetDevice(0);
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (parent_) (void)hipGraphDestroy(parent_);
        if (graph_) (void)hipGraphDestroy(graph_);
        for (auto event : events_) if (event) (void)hipEventDestroy(event);
        if (receipts_) (void)hipFree(receipts_);
        if (epoch_) (void)hipFree(epoch_);
        for (auto stream : streams_) if (stream) (void)hipStreamDestroy(stream);
    }

    unsigned width_;
    EntryFrontier frontier_;
    std::vector<EntryReceipt> host_receipts_;
    std::array<hipStream_t, 2> streams_{};
    std::array<hipEvent_t, 3> events_{};
    hipGraph_t graph_ = nullptr;
    hipGraph_t parent_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    std::uint32_t* epoch_ = nullptr;
    EntryReceipt* receipts_ = nullptr;
};

/** @brief One typed test definition, independent of previous case outcomes. */
struct EntrySpecification {
    unsigned width;
    EntryFrontier frontier;
    const char* name;
};

/** @brief Expand every entry contract over actual and reused physical queues. */
std::vector<EntrySpecification> entrySpecifications()
{
    std::vector<EntrySpecification> specifications;
    for (unsigned width : {2u, 3u, 4u, 8u}) {
        specifications.push_back({width, EntryFrontier::LaunchStreamProducer, "LaunchStreamProducer"});
        specifications.push_back({width, EntryFrontier::ExternalProducerEvent, "ExternalProducerEvent"});
        specifications.push_back({width, EntryFrontier::ParentProducerNode, "ParentProducerNode"});
    }
    return specifications;
}

/** @brief Run every independently owned native ordering case even after a failure. */
class Test__HIPGraphLaunchOrdering : public testing::TestWithParam<EntrySpecification> {};

/** @test Every root and child consumes the launch stream's exact producer frontier. */
TEST_P(Test__HIPGraphLaunchOrdering, IndependentRootsConsumeTheExactEntryFrontier)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    if (devices == 0) GTEST_SKIP() << "Requires a ROCm device";
    const auto& specification = GetParam();
    EntryOrderingProof proof(specification.width, specification.frontier);
    ASSERT_NO_THROW(proof.verify());
}

INSTANTIATE_TEST_SUITE_P(EntryFrontiers, Test__HIPGraphLaunchOrdering,
    testing::ValuesIn(entrySpecifications()),
    [](const testing::TestParamInfo<EntrySpecification>& information) {
        return "Width" + std::to_string(information.param.width) + "_" + information.param.name;
    });

/** @brief Whether a retained root graph is launched directly or as a child. */
enum class GraphEnvelope { TopLevel, NestedChild };

/** @brief Exact root/FIFO completion contract for one independent test. */
struct ExitSpecification {
    unsigned width;
    unsigned delayed_branch;
    GraphEnvelope envelope;
};

/**
 * @brief Own a retained graph and its device-written successor proof.
 *
 * Width fourteen crosses the native command map's first rehash boundary.
 * Delaying every root in turn avoids assumptions about node enumeration or
 * which branch the scheduler assigns to a reused physical queue.
 */
class ExitOrderingProof final {
public:
    /**
     * @brief Construct one exact native root graph and stable successor storage.
     * @param specification Immutable root count, delayed root and graph envelope.
     * @throws std::runtime_error after retiring partial native setup on failure.
     */
    explicit ExitOrderingProof(ExitSpecification specification)
        : specification_(specification)
    {
        try {
            if (specification_.width < 2 || specification_.delayed_branch >= specification_.width)
                throw std::runtime_error("Invalid native exit-frontier specification");
            requireHip(hipSetDevice(0), "select exit-proof device");
            requireHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "create exact exit stream");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming), "create exact exit event");
            requireHip(hipMalloc(reinterpret_cast<void**>(&receipts_), receiptBytes()), "allocate live root receipts");
            requireHip(hipMalloc(reinterpret_cast<void**>(&missing_), sizeof(host_missing_)), "allocate successor receipts");
            requireHip(hipMemsetAsync(receipts_, 0, receiptBytes(), stream_), "initialize root receipts");
            requireHip(hipEventRecord(terminal_, stream_), "publish exit-proof initialization");
            observeTerminal();
            construct();
        } catch (...) {
            release();
            throw;
        }
    }

    /** @brief Retire infrastructure ownership after its immutable result exists. */
    ~ExitOrderingProof() { release(); }
    ExitOrderingProof(const ExitOrderingProof&) = delete;
    ExitOrderingProof& operator=(const ExitOrderingProof&) = delete;

    /**
     * @brief Verify the successor consumes every submitted root's completion.
     * @throws std::runtime_error on native failure or any prematurely observed root.
     */
    void verify()
    {
        requireHip(hipGraphLaunch(executable_, stream_), "launch retained exit-proof graph");
        hipLaunchKernelGGL(observeGraphExit, dim3(1), dim3(kLanes), 0, stream_,
                           receipts_, missing_, specification_.width);
        requireHip(hipGetLastError(), "launch exact-stream successor");
        requireHip(hipMemcpyAsync(host_missing_.data(), missing_, sizeof(host_missing_),
                                  hipMemcpyDeviceToHost, stream_), "read immutable successor result");
        requireHip(hipEventRecord(terminal_, stream_), "publish successor result");
        observeTerminal();
        for (unsigned lane = 0; lane != kLanes; ++lane) {
            if (host_missing_[lane] != 0)
                throw std::runtime_error("Graph successor bypassed a submitted queue tail: width=" +
                    std::to_string(specification_.width) + " delayed_branch=" +
                    std::to_string(specification_.delayed_branch) + " lane=" +
                    std::to_string(lane) + " unfinished_roots=" + std::to_string(host_missing_[lane]));
        }
    }

private:
    /** @return Exact live root extent, without maximum-width padding. */
    std::size_t receiptBytes() const { return specification_.width * sizeof(*receipts_); }

    /** @brief Bind independent roots once, optionally inside a retained parent. */
    void construct()
    {
        requireHip(hipGraphCreate(&graph_, 0), "create exact root graph");
        for (unsigned branch = 0; branch != specification_.width; ++branch) {
            void* arguments[]{&receipts_, &branch, &specification_.delayed_branch};
            hipKernelNodeParams parameters{};
            parameters.func = reinterpret_cast<void*>(finishIndependentRoot);
            parameters.gridDim = dim3(1);
            parameters.blockDim = dim3(kLanes);
            parameters.kernelParams = arguments;
            hipGraphNode_t node = nullptr;
            requireHip(hipGraphAddKernelNode(&node, graph_, nullptr, 0, &parameters), "bind exact independent root");
        }
        hipGraph_t executable_graph = graph_;
        if (specification_.envelope == GraphEnvelope::NestedChild) {
            requireHip(hipGraphCreate(&parent_, 0), "create retained exit-proof parent");
            hipGraphNode_t child = nullptr;
            requireHip(hipGraphAddChildGraphNode(&child, parent_, nullptr, 0, graph_), "bind retained child root graph");
            executable_graph = parent_;
        }
        requireHip(hipGraphInstantiate(&executable_, executable_graph, nullptr, nullptr, 0), "instantiate exact exit graph");
    }

    /** @brief Observe only fixture initialization and the immutable terminal result. */
    void observeTerminal()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            const auto result = hipEventQuery(terminal_);
            if (result == hipSuccess) return;
            if (result != hipErrorNotReady) requireHip(result, "observe native exit frontier");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Native exit-frontier proof stalled before terminal publication");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
     * @brief Retire an infrastructure fixture safely even on the defective runtime.
     *
     * The negative control deliberately proves that its terminal event can
     * precede an unjoined root. Only teardown therefore joins the native device
     * before destroying graph/storage owners. The successor's sticky result
     * already exists: this cannot repair the tested order and is never a
     * production synchronization or a per-replay observation.
     */
    void release() noexcept
    {
        (void)hipSetDevice(0);
        if (executable_) (void)hipDeviceSynchronize();
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (parent_) (void)hipGraphDestroy(parent_);
        if (graph_) (void)hipGraphDestroy(graph_);
        if (terminal_) (void)hipEventDestroy(terminal_);
        if (missing_) (void)hipFree(missing_);
        if (receipts_) (void)hipFree(receipts_);
        if (stream_) (void)hipStreamDestroy(stream_);
    }

    ExitSpecification specification_;
    std::array<std::uint32_t, kLanes> host_missing_{};
    hipStream_t stream_ = nullptr;
    hipEvent_t terminal_ = nullptr;
    hipGraph_t graph_ = nullptr;
    hipGraph_t parent_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    std::uint32_t* receipts_ = nullptr;
    std::uint32_t* missing_ = nullptr;
};

/** @brief Cover queue reuse and native map boundaries without privileged assumptions. */
std::vector<ExitSpecification> exitSpecifications()
{
    std::vector<ExitSpecification> specifications;
    for (unsigned width : {4u, 8u, 14u, 16u, 30u, 32u})
        for (unsigned branch = 0; branch != width; ++branch)
            for (auto envelope : {GraphEnvelope::TopLevel, GraphEnvelope::NestedChild})
                specifications.push_back({width, branch, envelope});
    return specifications;
}

/** @brief Every exit contract has an independent outcome, including negative controls. */
class Test__HIPGraphExitOrdering : public testing::TestWithParam<ExitSpecification> {};

/** @test A graph successor observes every exact submitted FIFO tail, including children. */
TEST_P(Test__HIPGraphExitOrdering, SuccessorConsumesEverySubmittedQueueTail)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    if (devices == 0) GTEST_SKIP() << "Requires a ROCm device";
    ExitOrderingProof proof(GetParam());
    ASSERT_NO_THROW(proof.verify());
}

INSTANTIATE_TEST_SUITE_P(ExitFrontiers, Test__HIPGraphExitOrdering,
    testing::ValuesIn(exitSpecifications()),
    [](const testing::TestParamInfo<ExitSpecification>& information) {
        return "Width" + std::to_string(information.param.width) + "_Root" +
            std::to_string(information.param.delayed_branch) + "_" +
            (information.param.envelope == GraphEnvelope::TopLevel ? "TopLevel" : "NestedChild");
    });

} // namespace
