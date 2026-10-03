/**
 * @file hip_graph_wait_progress.hip
 * @brief Standalone native HIP child-graph collapse/progress reproducer.
 *
 * Build with the selected SDK, for example:
 * hipcc -O3 --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" \
 *   --offload-arch=gfx906 hip_graph_wait_progress.hip -o repro
 *
 * No model, RCCL, MPI, profiler, GTest, application kernel or scheduling override
 * is required. Worker waits for an independent Close producer. After a failed
 * two-second terminal observation, diagnostic cancellation releases the wait
 * before teardown; it never changes the result into a successful certificate.
 */
#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

/** @brief Exit at a rejected native call, retaining its exact API diagnostic. */
void check(hipError_t status, const char* operation) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    std::_Exit(2);
}
#define HIP(call) check((call), #call)

/**
 * @brief Capture one immutable native memory-wait/publication fragment.
 * @param stream Exact non-default recording stream, never separately launched.
 * @param words Persistent device-visible publication mapping.
 * @param waitWord Word to await, or -1 for finite write-only work.
 * @param publishWord Word to publish after the optional wait.
 * @return Retained child graph; its caller owns destruction.
 */
hipGraph_t record(hipStream_t stream, std::uint64_t* words, int waitWord, unsigned publishWord) {
    hipGraph_t graph{};
    HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    if (waitWord >= 0)
        HIP(hipStreamWaitValue64(stream, words + waitWord, 1, hipStreamWaitValueEq));
    HIP(hipStreamWriteValue64(stream, words + publishWord, 1, 0));
    HIP(hipStreamEndCapture(stream, &graph));
    return graph;
}

/** @brief Prove one complete retained parallel parent makes independent progress. */
int main() {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    HIP(hipSetDevice(0));
    std::uint64_t* words{};
    std::uint64_t* alias{};
    hipStream_t stream{};
    hipEvent_t terminal{};
    HIP(hipHostMalloc(&words, 4096, hipHostMallocMapped));
    HIP(hipHostGetDevicePointer(reinterpret_cast<void**>(&alias), words, 0));
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreateWithFlags(&terminal, hipEventDisableTiming));
    for (unsigned word = 0; word < 5; ++word) words[word] = 0;
    // Open, Body, Close, Worker receipt and the joined terminal receipt.
    // A write-only child performs the join: there are no empty graph segments,
    // keeping this reproducer independent of the separate completion defect.
    const std::array<hipGraph_t, 5> children{
        record(stream, alias, -1, 0), record(stream, alias, -1, 1),
        record(stream, alias, 2, 3), record(stream, alias, -1, 2),
        record(stream, alias, -1, 4)};
    hipGraph_t parent{};
    hipGraphExec_t executable{};
    HIP(hipGraphCreate(&parent, 0));
    std::array<hipGraphNode_t, 5> nodes{};
    HIP(hipGraphAddChildGraphNode(&nodes[0], parent, nullptr, 0, children[0]));
    HIP(hipGraphAddChildGraphNode(&nodes[1], parent, &nodes[0], 1, children[1]));
    HIP(hipGraphAddChildGraphNode(&nodes[2], parent, &nodes[0], 1, children[2]));
    HIP(hipGraphAddChildGraphNode(&nodes[3], parent, &nodes[1], 1, children[3]));
    const hipGraphNode_t frontier[]{nodes[2], nodes[3]};
    HIP(hipGraphAddChildGraphNode(&nodes[4], parent, frontier, 2, children[4]));
    HIP(hipGraphInstantiate(&executable, parent, nullptr, nullptr, 0));
    HIP(hipGraphLaunch(executable, stream));
    HIP(hipEventRecord(terminal, stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::atomic_ref<std::uint64_t>(words[4]).load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool complete = std::atomic_ref<std::uint64_t>(words[4]).load(std::memory_order_acquire) == 1;
    std::printf("parallel Close/Worker progress: %s; words=%llu,%llu,%llu,%llu,%llu\n",
                complete ? "PASS" : "FAIL",
                static_cast<unsigned long long>(words[0]), static_cast<unsigned long long>(words[1]),
                static_cast<unsigned long long>(words[2]), static_cast<unsigned long long>(words[3]),
                static_cast<unsigned long long>(words[4]));
    if (!complete)
        for (unsigned word = 0; word < 5; ++word)
            std::atomic_ref<std::uint64_t>(words[word]).store(1, std::memory_order_release);
    HIP(hipEventSynchronize(terminal));
    HIP(hipGraphExecDestroy(executable));
    HIP(hipGraphDestroy(parent));
    for (auto child : children) HIP(hipGraphDestroy(child));
    HIP(hipEventDestroy(terminal));
    HIP(hipStreamDestroy(stream));
    HIP(hipHostFree(words));
    return complete ? 0 : 1;
}
