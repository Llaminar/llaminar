/**
 * @file Test__HIPGraphWaitConcurrency.hip
 * @brief Prove graph ROI scheduling cannot serialize a waiter before its producer.
 *
 * The native graph declares Open -> {Body, Worker}, Body -> Close, then joins
 * Close and Worker. Worker waits for Close using HIP's typed memory operation.
 * Treating that wait, or an opaque child containing it, as one finite cost unit
 * can collapse both branches onto one FIFO and deadlock. This model-free ABI
 * fixture owns its native allocations and bounded diagnostic cancellation;
 * production graph capture must make progress without either host intervention
 * or an environment override. Both timeline widths and flat/child forms matter.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

namespace {
/** @brief Preserve either an explicit typed memory node or an opaque child. */
enum class GraphShape { Flat, Children };
/** @brief The immutable phases of the native parallel publication protocol. */
enum class Phase { Open, Body, Worker, Close };

/**
 * @brief Fail at a rejected native operation, without changing execution modes.
 * @param status Native HIP return value.
 * @param operation Lifecycle operation retained in the diagnostic.
 */
void requireHip(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

/**
 * @brief Own one complete retained parent and its mapped terminal evidence.
 * @tparam Word Exact native timeline width, uint32_t or uint64_t.
 *
 * No child is separately launched. The host's only progress observation is the
 * terminal receipt; a test-only cancellation releases an already failed graph
 * so negative controls cannot strand the accelerator during teardown.
 */
template<class Word>
class WaitConcurrencyProof final {
public:
    /**
     * @brief Bind the immutable publication DAG and instantiate it once.
     * @param shape Flat memory-operation nodes or equivalent retained children.
     * @throws std::runtime_error After retiring a rejected partial setup.
     */
    explicit WaitConcurrencyProof(GraphShape shape) : shape_(shape) {
        static_assert(std::is_same_v<Word, std::uint32_t> ||
                      std::is_same_v<Word, std::uint64_t>);
        try {
            requireHip(hipSetDevice(0), "select real graph device");
            requireHip(hipHostMalloc(reinterpret_cast<void**>(&words_), 4096,
                                     hipHostMallocMapped), "allocate persistent publication words");
            requireHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&alias_), words_, 0),
                       "bind exact publication mapping");
            requireHip(hipCtxGetCurrent(&context_), "retain current native context identity");
            if (!context_) throw std::runtime_error("Batch memory graph requires a real context");
            requireHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                       "create exact graph launch stream");
            requireHip(hipEventCreateWithFlags(&terminal_, hipEventDisableTiming),
                       "create terminal completion frontier");
            requireHip(hipGraphCreate(&graph_, 0), "create retained parent");
            const auto open = attach(Phase::Open, nullptr, 0);
            const auto body = attach(Phase::Body, &open, 1);
            const auto worker = attach(Phase::Worker, &open, 1);
            const auto close = attach(Phase::Close, &body, 1);
            const hipGraphNode_t frontier[]{close, worker};
            hipGraphNode_t joined{};
            requireHip(hipGraphAddEmptyNode(&joined, graph_, frontier, 2),
                       "join both actual terminal branches");
            requireHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
                       "instantiate complete parallel publication DAG");
        } catch (...) {
            retire();
            throw;
        }
    }

    /** @brief Retire the completed parent before the children and mappings. */
    ~WaitConcurrencyProof() { retire(); }
    WaitConcurrencyProof(const WaitConcurrencyProof&) = delete;
    WaitConcurrencyProof& operator=(const WaitConcurrencyProof&) = delete;

    /**
     * @brief Prove twenty data-only resets progress through the retained parent.
     *
     * Cancellation is diagnostic only, after a failed terminal observation.
     * It precedes assertion so a red control cannot leave an infinite GPU wait.
     * Successful graphs require no cancellation, recapture or child submission.
     */
    void verify() {
        for (unsigned replay = 0; replay < 20; ++replay) {
            SCOPED_TRACE(replay);
            for (std::size_t word = 0; word < 4; ++word)
                std::atomic_ref<Word>(words_[word]).store(0, std::memory_order_release);
            requireHip(hipGraphLaunch(executable_, stream_), "replay complete retained parent");
            requireHip(hipEventRecord(terminal_, stream_), "publish terminal join");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::atomic_ref<Word>(words_[3]).load(std::memory_order_acquire) == 0 &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            const bool complete = std::atomic_ref<Word>(words_[3]).load(std::memory_order_acquire) == 1;
            if (!complete)
                for (std::size_t word = 0; word < 4; ++word)
                    std::atomic_ref<Word>(words_[word]).store(1, std::memory_order_release);
            requireHip(hipEventSynchronize(terminal_), "observe exact terminal completion");
            ASSERT_TRUE(complete) << "ROI scheduling serialized Worker before its Close producer";
            for (std::size_t word = 0; word < 4; ++word)
                EXPECT_EQ(words_[word], static_cast<Word>(1));
        }
    }

private:
    /** @brief Build a typed system-visible publication for one persistent word. */
    hipStreamBatchMemOpParams write(std::size_t word) const {
        hipStreamBatchMemOpParams operation{};
        operation.operation = std::is_same_v<Word, std::uint32_t>
            ? hipStreamMemOpWriteValue32 : hipStreamMemOpWriteValue64;
        operation.writeValue.address = reinterpret_cast<hipDeviceptr_t>(alias_ + word);
        if constexpr (std::is_same_v<Word, std::uint32_t>) operation.writeValue.value = 1;
        else operation.writeValue.value64 = 1;
        return operation;
    }
    /** @brief Build the explicit native wait for the independent Close producer. */
    hipStreamBatchMemOpParams waitForClose() const {
        hipStreamBatchMemOpParams operation{};
        operation.operation = std::is_same_v<Word, std::uint32_t>
            ? hipStreamMemOpWaitValue32 : hipStreamMemOpWaitValue64;
        operation.waitValue.address = reinterpret_cast<hipDeviceptr_t>(alias_ + 2);
        operation.waitValue.flags = hipStreamWaitValueEq;
        if constexpr (std::is_same_v<Word, std::uint32_t>) operation.waitValue.value = 1;
        else operation.waitValue.value64 = 1;
        return operation;
    }
    /**
     * @brief Add the exact native memory operations for a protocol phase.
     * @param graph Immutable owner under construction.
     * @param phase Required wait/publication semantics.
     * @param dependencies Complete declared incoming frontier.
     * @param count Number of actual producer dependencies, never capacity.
     * @return Native node bound to the exact persistent words.
     */
    hipGraphNode_t memoryNode(hipGraph_t graph, Phase phase,
                              const hipGraphNode_t* dependencies, std::size_t count) {
        std::array<hipStreamBatchMemOpParams, 2> operations{};
        unsigned int operationCount = 1;
        switch (phase) {
        case Phase::Open: operations[0] = write(0); break;
        case Phase::Body: operations[0] = write(1); break;
        case Phase::Close: operations[0] = write(2); break;
        case Phase::Worker:
            operations[0] = waitForClose();
            operations[1] = write(3);
            operationCount = 2;
            break;
        }
        const hipBatchMemOpNodeParams params{
            .ctx = context_, .count = operationCount, .paramArray = operations.data(), .flags = 0};
        hipGraphNode_t node{};
        requireHip(hipGraphAddBatchMemOpNode(&node, graph, dependencies, count, &params),
                   "add exact typed memory-operation phase");
        // HIP owns a deep copy of the operation array, not this temporary view.
        return node;
    }
    /**
     * @brief Attach a direct phase or its retained child with identical edges.
     * @param phase Protocol phase to preserve across both representations.
     * @param dependencies Complete declared incoming frontier.
     * @param count Number of actual dependencies.
     * @return Parent node, never an independently submitted executable.
     */
    hipGraphNode_t attach(Phase phase, const hipGraphNode_t* dependencies, std::size_t count) {
        if (shape_ == GraphShape::Flat) return memoryNode(graph_, phase, dependencies, count);
        auto& child = children_[static_cast<std::size_t>(phase)];
        requireHip(hipGraphCreate(&child, 0), "create retained phase child");
        (void)memoryNode(child, phase, nullptr, 0);
        hipGraphNode_t node{};
        requireHip(hipGraphAddChildGraphNode(&node, graph_, dependencies, count, child),
                   "attach child with exact parallel frontier");
        return node;
    }
    /** @brief Release infrastructure, including a rejected partial setup. */
    void retire() noexcept {
        if (executable_) (void)hipGraphExecDestroy(executable_);
        if (graph_) (void)hipGraphDestroy(graph_);
        for (auto child : children_) if (child) (void)hipGraphDestroy(child);
        if (terminal_) (void)hipEventDestroy(terminal_);
        if (stream_) (void)hipStreamDestroy(stream_);
        if (words_) (void)hipHostFree(words_);
    }

    GraphShape shape_;
    hipCtx_t context_{};
    hipStream_t stream_{};
    hipEvent_t terminal_{};
    hipGraph_t graph_{};
    hipGraphExec_t executable_{};
    std::array<hipGraph_t, 4> children_{};
    Word* words_{};
    Word* alias_{};
};

/** @test Typed 32-bit waits prevent unsafe collapse even without child nodes. */
TEST(HIPGraphWaitConcurrency, Flat32BitWaitPreservesProducerProgress) {
    WaitConcurrencyProof<std::uint32_t> proof(GraphShape::Flat);
    proof.verify();
}
/** @test Flat 64-bit timeline waits require the same finite-work admission. */
TEST(HIPGraphWaitConcurrency, Flat64BitWaitPreservesProducerProgress) {
    WaitConcurrencyProof<std::uint64_t> proof(GraphShape::Flat);
    proof.verify();
}
/** @test An opaque retained child is not one finite unit of estimated work. */
TEST(HIPGraphWaitConcurrency, Child32BitWaitPreservesProducerProgress) {
    WaitConcurrencyProof<std::uint32_t> proof(GraphShape::Children);
    proof.verify();
}
/** @test Retained 64-bit children preserve the identical publication DAG. */
TEST(HIPGraphWaitConcurrency, Child64BitWaitPreservesProducerProgress) {
    WaitConcurrencyProof<std::uint64_t> proof(GraphShape::Children);
    proof.verify();
}
} // namespace
