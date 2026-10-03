/**
 * @file Test__HIPGraphBatchPublication.hip
 * @brief Prove complete AQL batch publication across retained graph queue wraps.
 *
 * A command processor can prefetch interior dispatches while the producer is
 * still copying their bodies. A valid header is therefore a publication, not
 * part of an ordinary structure copy. The canonical HIP repair copies every
 * packet with an invalid header before release-publishing its original header.
 *
 * These all-kernel graphs exercise native packet capture, unlike an empty-node
 * or event-node chain that can select a different scheduler. Long independent
 * branches and repeated unchanged launches wrap and saturate the bounded native
 * queues. Saturation must preserve the native writers' vacant-slot admission
 * boundary for the entire batch, not only its first reserved packet.
 * Every node retains a per-lane visit count and sticky ordering error; checking
 * only the final result would hide an omitted dispatch on an earlier replay.
 *
 * Native allocations are confined to this runtime-infrastructure fixture.
 * Production allocation and transfer remain PhysicalMemoryAuthority and
 * TransferEngine responsibilities. Host observation occurs only at the final
 * test readback, never between graph nodes or launches.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned kLanes = 64;
constexpr unsigned kChainLength = 512;
constexpr unsigned kReplays = 512;
constexpr std::uint32_t kSaltMultiplier = 2654435761u;

/** @brief One single-writer receipt; errors survive subsequent correct replays. */
struct DispatchReceipt {
    std::uint32_t value;
    std::uint32_t visits;
    std::uint32_t errors;
};

/**
 * @brief Mix one dispatch's immutable argument into its predecessor's value.
 * @param value Value read from the graph-ordered predecessor.
 * @param salt Distinct by-value argument for this graph node.
 * @return Defined modulo-2^32 arithmetic shared by the host and GPU oracle.
 */
__host__ __device__ std::uint32_t mix(std::uint32_t value, std::uint32_t salt)
{
    return ((value << 7) | (value >> 25)) * 1664525u + salt + 1013904223u;
}

/**
 * @brief Derive a branch prefix independently of any stored predecessor value.
 * @param slot Last included node, or zero for the graph root.
 * @return Closed-form sum of this branch's immutable salts, modulo 2^32.
 *
 * The arithmetic sequence needs only a few integer operations, not a replay of
 * the branch. It lets every dispatch detect earlier value corruption even when
 * a subsequent replay overwrites that receipt with an apparently good result.
 */
__host__ __device__ std::uint32_t prefixSalt(unsigned slot)
{
    if (slot == 0) return 0;
    const auto count = (slot - 1) % kChainLength + 1;
    const auto first = slot - count + 1;
    return (count * (first + slot) / 2) * kSaltMultiplier;
}

/**
 * @brief Publish a new GPU-owned replay epoch and initialize its graph root.
 * @param receipts Persistent, setup-owned result storage for every graph node.
 * @param epoch Replay counter; no host uploads occur between graph launches.
 */
__global__ void beginReplay(DispatchReceipt* receipts, std::uint32_t* epoch)
{
    if (threadIdx.x == 0) ++*epoch;
    __syncthreads();
    const auto replay = *epoch;
    auto& receipt = receipts[threadIdx.x];
    receipt.errors |= receipt.visits != replay - 1;
    receipt.value = mix(replay, threadIdx.x);
    ++receipt.visits;
}

/**
 * @brief Verify and extend one retained dispatch's serial branch lineage.
 * @param receipts Per-node, per-lane single-writer receipts.
 * @param epoch Current GPU-owned replay epoch.
 * @param slot Unique receipt row written by this kernel node.
 * @param predecessor Receipt row of this node's explicit graph dependency.
 * @param salt Unique immutable argument, exposing partially published packets.
 */
__global__ void extendLineage(DispatchReceipt* receipts, const std::uint32_t* epoch,
                              unsigned slot, unsigned predecessor, std::uint32_t salt)
{
    const auto replay = *epoch;
    const auto source = receipts[predecessor * kLanes + threadIdx.x];
    auto& receipt = receipts[slot * kLanes + threadIdx.x];
    const auto expected_predecessor = (slot - 1) % kChainLength == 0 ? 0 : slot - 1;
    receipt.errors |= source.errors | (source.visits != replay) |
                      (receipt.visits != replay - 1) |
                      (predecessor != expected_predecessor) |
                      (salt != slot * kSaltMultiplier) |
                      (source.value != mix(replay, threadIdx.x) + prefixSalt(expected_predecessor));
    receipt.value = source.value + salt;
    ++receipt.visits;
}

/**
 * @brief Join every actual branch, not unused maximum-width storage.
 * @param receipts Persistent receipts including the terminal row.
 * @param epoch Current GPU-owned replay epoch.
 * @param width Actual branch count; also determines the terminal row.
 */
__global__ void joinLineages(DispatchReceipt* receipts, const std::uint32_t* epoch, unsigned width)
{
    const auto replay = *epoch;
    auto& receipt = receipts[(1 + width * kChainLength) * kLanes + threadIdx.x];
    receipt.errors |= receipt.visits != replay - 1;
    auto value = mix(replay, threadIdx.x);
    for (unsigned branch = 0; branch != width; ++branch) {
        const auto source = receipts[((branch + 1) * kChainLength) * kLanes + threadIdx.x];
        receipt.errors |= source.errors | (source.visits != replay) |
                          (source.value != mix(replay, threadIdx.x) + prefixSalt((branch + 1) * kChainLength));
        value = mix(value, source.value);
    }
    receipt.value = value;
    ++receipt.visits;
}

/**
 * @brief Attribute every native setup or replay failure to its exact boundary.
 * @param status Native result of the immediately preceding operation.
 * @param operation Stable diagnostic naming that operation.
 * @throws std::runtime_error on a native HIP failure.
 */
void requireHip(hipError_t status, const char* operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/** @brief Exact launch-stream policy; neither option permits concurrent writes. */
enum class LaunchPlacement { SameStream, OrderedStreamChanges };

/** @brief Own one unchanged retained graph, its receipts and terminal boundary. */
class PacketPublicationProof final {
public:
    /**
     * @brief Allocate persistent native fixture storage and build an all-kernel DAG.
     * @param width Actual independent branch count in the bounded HIP queue pool.
     * @throws std::runtime_error on setup failure, after retiring partial ownership.
     */
    explicit PacketPublicationProof(unsigned width) : width_(width),
        host_receipts_((2 + width * kChainLength) * kLanes)
    {
        try {
            if (width_ < 1 || width_ > 4)
                throw std::runtime_error("Packet publication fixture width must be 1..4");
            requireHip(hipSetDevice(0), "select fixture device");
            for (auto& stream : streams_)
                requireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create exact stream");
            for (auto& event : ordered_)
                requireHip(hipEventCreateWithFlags(&event, hipEventDisableTiming), "create ordering event");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming), "create terminal");
            requireHip(hipMalloc(reinterpret_cast<void**>(&device_receipts_), receiptBytes()), "allocate receipts");
            requireHip(hipMalloc(reinterpret_cast<void**>(&epoch_), sizeof(*epoch_)), "allocate replay epoch");
            construct();
        } catch (...) {
            release();
            throw;
        }
    }

    /** @brief Retire native objects only after their terminal observation. */
    ~PacketPublicationProof() { release(); }
    PacketPublicationProof(const PacketPublicationProof&) = delete;
    PacketPublicationProof& operator=(const PacketPublicationProof&) = delete;

    /**
     * @brief Replay enough retained packets to wrap the native queue many times.
     * @param placement Whether ordered launches change their exact stream owner.
     *
     * All resets precede the first replay. Later replays update only GPU-owned
     * epochs. Stream changes wait on the previous launch's exact recorded event;
     * there is no host wait, allocation, argument rebinding or graph recapture.
     * @throws std::runtime_error on a replay, terminal or receipt failure.
     */
    void verify(LaunchPlacement placement)
    {
        auto stream = streams_.front();
        requireHip(hipMemsetAsync(device_receipts_, 0, receiptBytes(), stream), "initialize receipts");
        requireHip(hipMemsetAsync(epoch_, 0, sizeof(*epoch_), stream), "initialize replay epoch");
        for (unsigned replay = 0; replay != kReplays; ++replay) {
            if (placement == LaunchPlacement::OrderedStreamChanges && replay != 0) {
                stream = streams_[replay % streams_.size()];
                requireHip(hipStreamWaitEvent(stream, ordered_[(replay - 1) % ordered_.size()], 0),
                           "join preceding exact launch");
            }
            requireHip(hipGraphLaunch(executable_, stream), "launch retained packet graph");
            if (placement == LaunchPlacement::OrderedStreamChanges)
                requireHip(hipEventRecord(ordered_[replay % ordered_.size()], stream), "publish exact launch");
        }
        requireHip(hipMemcpyAsync(host_receipts_.data(), device_receipts_, receiptBytes(),
                                  hipMemcpyDeviceToHost, stream), "read terminal receipts");
        requireHip(hipEventRecord(terminal_, stream), "publish terminal readback");
        observeTerminal();
        verifyReceipts();
    }

private:
    /** @return The exact live receipt extent for this graph width. */
    std::size_t receiptBytes() const { return host_receipts_.size() * sizeof(DispatchReceipt); }

    /**
     * @brief Add one native kernel node with already-owned, copied parameters.
     * @param function Exact kernel entry.
     * @param dependencies Explicit predecessors; siblings remain independent.
     * @param count Actual predecessor count.
     * @param arguments Native kernel argument addresses copied during this call.
     * @return Newly owned graph node.
     * @throws std::runtime_error on native construction failure.
     */
    hipGraphNode_t addKernel(void* function, const hipGraphNode_t* dependencies,
                            std::size_t count, void** arguments)
    {
        hipKernelNodeParams parameters{};
        parameters.func = function;
        parameters.gridDim = dim3(1);
        parameters.blockDim = dim3(kLanes);
        parameters.kernelParams = arguments;
        hipGraphNode_t node = nullptr;
        requireHip(hipGraphAddKernelNode(&node, graph_, dependencies, count, &parameters), "add retained kernel");
        return node;
    }

    /** @brief Construct only kernel nodes so native captured packet batches stay eligible. */
    void construct()
    {
        requireHip(hipGraphCreate(&graph_, 0), "create packet graph");
        void* root_arguments[]{&device_receipts_, &epoch_};
        const auto root = addKernel(reinterpret_cast<void*>(beginReplay), nullptr, 0, root_arguments);
        std::vector<hipGraphNode_t> tails;
        for (unsigned branch = 0; branch != width_; ++branch) {
            auto predecessor = root;
            unsigned predecessor_slot = 0;
            for (unsigned step = 0; step != kChainLength; ++step) {
                unsigned slot = 1 + branch * kChainLength + step;
                std::uint32_t salt = slot * kSaltMultiplier;
                void* arguments[]{&device_receipts_, &epoch_, &slot, &predecessor_slot, &salt};
                predecessor = addKernel(reinterpret_cast<void*>(extendLineage), &predecessor, 1, arguments);
                predecessor_slot = slot;
            }
            tails.push_back(predecessor);
        }
        void* join_arguments[]{&device_receipts_, &epoch_, &width_};
        (void)addKernel(reinterpret_cast<void*>(joinLineages), tails.data(), tails.size(), join_arguments);
        requireHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0), "instantiate packet graph");
    }

    /** @brief Observe only the terminal result, with the standard bounded progress allowance. */
    void observeTerminal()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
            const auto status = hipEventQuery(terminal_);
            if (status == hipSuccess) return;
            if (status != hipErrorNotReady) requireHip(status, "observe terminal");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Retained packet graph stalled before its terminal receipt");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /** @brief Verify every replay's sticky invariants and every final node's exact arithmetic. */
    void verifyReceipts() const
    {
        for (unsigned lane = 0; lane != kLanes; ++lane) {
            const auto root_value = mix(kReplays, lane);
            checkReceipt(0, lane, root_value);
            auto joined = root_value;
            for (unsigned branch = 0; branch != width_; ++branch) {
                auto value = root_value;
                for (unsigned step = 0; step != kChainLength; ++step) {
                    const auto slot = 1 + branch * kChainLength + step;
                    value += slot * kSaltMultiplier;
                    checkReceipt(slot, lane, value);
                }
                joined = mix(joined, value);
            }
            checkReceipt(1 + width_ * kChainLength, lane, joined);
        }
    }

    /**
     * @brief Reject any omitted, duplicated, reordered or argument-corrupted dispatch.
     * @param slot Exact owned receipt row.
     * @param lane Exact scalar lane.
     * @param expected Independent host arithmetic for the last replay.
     * @throws std::runtime_error with the first attributable receipt mismatch.
     */
    void checkReceipt(unsigned slot, unsigned lane, std::uint32_t expected) const
    {
        const auto& receipt = host_receipts_[slot * kLanes + lane];
        if (receipt.visits != kReplays || receipt.errors != 0 || receipt.value != expected)
            throw std::runtime_error("Invalid captured packet receipt: width=" + std::to_string(width_) +
                                     " slot=" + std::to_string(slot) + " lane=" + std::to_string(lane) +
                                     " visits=" + std::to_string(receipt.visits) +
                                     " errors=" + std::to_string(receipt.errors));
    }

    /** @brief Release setup-only native ownership in graph-before-buffer order. */
    void release() noexcept
    {
        (void)hipSetDevice(0);
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (graph_) (void)hipGraphDestroy(graph_);
        if (terminal_) (void)hipEventDestroy(terminal_);
        for (auto event : ordered_) if (event) (void)hipEventDestroy(event);
        if (device_receipts_) (void)hipFree(device_receipts_);
        if (epoch_) (void)hipFree(epoch_);
        for (auto stream : streams_) if (stream) (void)hipStreamDestroy(stream);
    }

    unsigned width_;
    std::vector<DispatchReceipt> host_receipts_;
    std::array<hipStream_t, 2> streams_{};
    std::array<hipEvent_t, 2> ordered_{};
    hipEvent_t terminal_ = nullptr;
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    DispatchReceipt* device_receipts_ = nullptr;
    std::uint32_t* epoch_ = nullptr;
};

/** @test Every retained packet survives deep batches, queue wrap and ordered stream changes. */
TEST(Test__HIPGraphBatchPublication, EveryDispatchPublishesItsCompleteBodyAcrossQueueWrap)
{
    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    if (devices == 0) GTEST_SKIP() << "Requires a ROCm device";
    for (unsigned width : {1u, 2u, 3u, 4u}) {
        SCOPED_TRACE("width=" + std::to_string(width));
        for (auto placement : {LaunchPlacement::SameStream, LaunchPlacement::OrderedStreamChanges}) {
            SCOPED_TRACE(placement == LaunchPlacement::SameStream ? "same-stream" : "ordered-stream-changes");
            std::cout << "Packet publication proof: width=" << width << " placement="
                      << (placement == LaunchPlacement::SameStream ? "same-stream" : "ordered-stream-changes")
                      << std::endl;
            PacketPublicationProof proof(width);
            ASSERT_NO_THROW(proof.verify(placement));
        }
    }
}

} // namespace
