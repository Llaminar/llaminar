/**
 * @file Test__HIPEmptySegmentPublication.hip
 * @brief Prove zero-packet captured segments publish every consumed completion.
 *
 * Empty graph nodes carry dependencies even though they carry no model payload.
 * ROCm 10 initially reserved signals for empty-only segments without submitting
 * a packet to complete them, deadlocking a cross-stream successor. Exercise
 * default native packet capture, retained replay, changing launch streams,
 * nested children, empty-only branches and widths beyond the hardware pool.
 * Native allocations are confined to this model-free runtime boundary fixture;
 * production allocations remain PhysicalMemoryAuthority/TransferEngine owned.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kLanes = 64;
constexpr unsigned kWaves = 8;
constexpr unsigned kReplays = 20;

/** @brief Whether branch segments contain payload or only dependency nodes. */
enum class BranchContents { Receipts, Empty };
/** @brief Whether this graph is top-level or a retained parent's child. */
enum class Envelope { Direct, Child };
/** @brief One uniquely written lane's sticky ordering and execution witness. */
struct Receipt { std::uint32_t visits; std::uint32_t errors; };

/** @brief Publish the next GPU-owned epoch on the exact graph launch stream. */
__global__ void publishEpoch(std::uint32_t* epoch) {
    if (threadIdx.x == 0) ++*epoch;
}

/**
 * @brief Record one branch-owned visit and preserve any earlier ordering error.
 * @param epoch Device-only serial producer epoch, never mirrored to the host.
 * @param receipts Persistent single-writer lane storage.
 * @param slot Branch identity within the retained graph.
 */
__global__ void observeEpoch(const std::uint32_t* epoch, Receipt* receipts, unsigned slot) {
    auto& receipt = receipts[slot * kLanes + threadIdx.x];
    if (*epoch != receipt.visits + 1) ++receipt.errors;
    ++receipt.visits;
}

/** @brief Surface the exact failed native lifecycle operation. */
void requireHip(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/** @brief Own one native graph lifetime and all persistent replay bindings. */
class EmptySegmentProof final {
public:
    /**
     * @brief Allocate before capture and declare the fork/join dependency DAG.
     * @param width Independent branches, including oversubscribed queue widths.
     * @param contents Whether branch nodes execute kernels or are all empty.
     * @param envelope Whether to embed this graph in a retained parent.
     * @throws std::runtime_error after releasing partial native setup.
     */
    EmptySegmentProof(unsigned width, BranchContents contents, Envelope envelope)
        : slots_(contents == BranchContents::Receipts ? width * kWaves : 1),
          host_(slots_ * kLanes) {
        try {
            requireHip(hipSetDevice(0), "select fixture GPU");
            for (auto& stream : streams_)
                requireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create explicit stream");
            requireHip(hipEventCreateWithFlags(&completion_, hipEventDisableTiming), "create publication event");
            requireHip(hipMalloc(&epoch_, sizeof(*epoch_)), "allocate persistent epoch");
            requireHip(hipMalloc(&receipts_, host_.size() * sizeof(Receipt)), "allocate sticky receipts");
            requireHip(hipMemsetAsync(epoch_, 0, sizeof(*epoch_), streams_[0]), "initialize epoch");
            requireHip(hipMemsetAsync(receipts_, 0, host_.size() * sizeof(Receipt), streams_[0]), "initialize receipts");
            requireHip(hipGraphCreate(&graph_, 0), "create graph");
            hipGraphNode_t predecessor{};
            requireHip(hipGraphAddEmptyNode(&predecessor, graph_, nullptr, 0), "declare empty root");
            for (unsigned wave = 0; wave < kWaves; ++wave) {
                std::vector<hipGraphNode_t> branches(width);
                for (unsigned branch = 0; branch < width; ++branch) {
                    if (contents == BranchContents::Receipts)
                        branches[branch] = addReceipt(predecessor, wave * width + branch);
                    else
                        requireHip(hipGraphAddEmptyNode(&branches[branch], graph_, &predecessor, 1), "declare empty branch");
                }
                // Empty joins must transmit their incoming frontier at later forks.
                requireHip(hipGraphAddEmptyNode(&predecessor, graph_, branches.data(), width), "declare empty join");
            }
            if (contents == BranchContents::Empty) (void)addReceipt(predecessor, 0);
            hipGraph_t captured = graph_;
            if (envelope == Envelope::Child) {
                requireHip(hipGraphCreate(&parent_, 0), "create parent graph");
                hipGraphNode_t child{};
                requireHip(hipGraphAddChildGraphNode(&child, parent_, nullptr, 0, graph_), "declare child graph");
                captured = parent_;
            }
            requireHip(hipGraphInstantiate(&executable_, captured, nullptr, nullptr, 0), "instantiate retained graph");
        } catch (...) {
            release();
            throw;
        }
    }

    /** @brief Retire graph owners before storage and explicit streams. */
    ~EmptySegmentProof() { release(); }
    EmptySegmentProof(const EmptySegmentProof&) = delete;
    EmptySegmentProof& operator=(const EmptySegmentProof&) = delete;

    /**
     * @brief Replay twenty times with device events, then check every sticky lane.
     *
     * There is no per-replay host wait or readback. Each new launch stream waits
     * on its predecessor's exact event before advancing the epoch. Only the
     * terminal result reaches the host, proving complete graph publications.
     */
    void verify() {
        for (unsigned replay = 0; replay < kReplays; ++replay) {
            const auto stream = streams_[replay % streams_.size()];
            if (replay != 0)
                requireHip(hipStreamWaitEvent(stream, completion_, 0), "consume previous graph publication");
            hipLaunchKernelGGL(publishEpoch, dim3(1), dim3(kLanes), 0, stream, epoch_);
            requireHip(hipGetLastError(), "submit epoch producer");
            requireHip(hipGraphLaunch(executable_, stream), "submit retained graph");
            requireHip(hipEventRecord(completion_, stream), "publish graph completion");
        }
        const auto terminal_stream = streams_[(kReplays - 1) % streams_.size()];
        requireHip(hipMemcpyAsync(host_.data(), receipts_, host_.size() * sizeof(Receipt),
                                 hipMemcpyDeviceToHost, terminal_stream), "read terminal receipts");
        requireHip(hipEventRecord(completion_, terminal_stream), "publish terminal readback");
        requireHip(hipEventSynchronize(completion_), "observe terminal result");
        for (unsigned slot = 0; slot < slots_; ++slot) {
            SCOPED_TRACE("slot=" + std::to_string(slot));
            for (unsigned lane = 0; lane < kLanes; ++lane) {
                SCOPED_TRACE("lane=" + std::to_string(lane));
                const auto& receipt = host_[slot * kLanes + lane];
                ASSERT_EQ(receipt.visits, kReplays);
                ASSERT_EQ(receipt.errors, 0u);
            }
        }
    }

private:
    /** @brief Bind one uniquely owned kernel row after its exact predecessor. */
    hipGraphNode_t addReceipt(hipGraphNode_t predecessor, unsigned slot) {
        void* arguments[]{&epoch_, &receipts_, &slot};
        hipKernelNodeParams parameters{};
        parameters.func = reinterpret_cast<void*>(observeEpoch);
        parameters.gridDim = dim3(1);
        parameters.blockDim = dim3(kLanes);
        parameters.kernelParams = arguments;
        hipGraphNode_t node{};
        requireHip(hipGraphAddKernelNode(&node, graph_, &predecessor, 1, &parameters), "declare receipt kernel");
        return node;
    }

    /** @brief Release completed native objects, including partial constructor setup. */
    void release() noexcept {
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (parent_) (void)hipGraphDestroy(parent_);
        if (graph_) (void)hipGraphDestroy(graph_);
        if (receipts_) (void)hipFree(receipts_);
        if (epoch_) (void)hipFree(epoch_);
        if (completion_) (void)hipEventDestroy(completion_);
        for (auto stream : streams_) if (stream) (void)hipStreamDestroy(stream);
    }

    unsigned slots_;
    std::vector<Receipt> host_;
    std::array<hipStream_t, 2> streams_{};
    hipEvent_t completion_{};
    hipGraph_t graph_{};
    hipGraph_t parent_{};
    hipGraphExec_t executable_{};
    std::uint32_t* epoch_{};
    Receipt* receipts_{};
};

/** @brief Typed native graph geometry, independent of names and prior outcomes. */
struct Specification { unsigned width; BranchContents contents; Envelope envelope; };
/** @brief Include no-cross-stream, physical-pool and oversubscribed widths. */
std::vector<Specification> specifications() {
    std::vector<Specification> result;
    for (unsigned width : {1u, 2u, 4u, 8u})
        for (auto contents : {BranchContents::Receipts, BranchContents::Empty})
            for (auto envelope : {Envelope::Direct, Envelope::Child})
                result.push_back({width, contents, envelope});
    return result;
}
/** @brief Each parameter owns a complete independent runtime lifecycle. */
class Test__HIPEmptySegmentPublication : public testing::TestWithParam<Specification> {};
/** @test Every zero-payload frontier remains live through native captured replay. */
TEST_P(Test__HIPEmptySegmentPublication, EveryConsumedCompletionHasAProducer) {
    int devices{};
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);
    ASSERT_GT(devices, 0) << "This explicit ROCm gate requires a real device";
    const auto& specification = GetParam();
    EmptySegmentProof proof(specification.width, specification.contents, specification.envelope);
    ASSERT_NO_THROW(proof.verify());
}
INSTANTIATE_TEST_SUITE_P(EmptyFrontiers, Test__HIPEmptySegmentPublication,
    testing::ValuesIn(specifications()),
    [](const testing::TestParamInfo<Specification>& information) {
        const auto& value = information.param;
        return "Width" + std::to_string(value.width) +
               (value.contents == BranchContents::Receipts ? "_Receipts" : "_Empty") +
               (value.envelope == Envelope::Direct ? "_Direct" : "_Child");
    });
} // namespace
