/**
 * @file hip_sdma_event_progress.hip
 * @brief Model-free reproducer for SDMA events blocked by unrelated compute.
 *
 * Build with the selected SDK's hipcc, C++20 and the native GPU architecture.
 * Run with an optional held-graph count (0 through 64). --wait-frontier and
 * --compute-frontier are negative controls: their terminals really must wait
 * for the held producer. No model, custom kernel, MPI, RCCL, profiler, default
 * stream, queue-limit override or alternative copy implementation is involved.
 * This infrastructure test alone owns direct native storage; production uses
 * PhysicalMemoryAuthority and TransferEngine. Failures are decided before the
 * bounded cancellation cleanup, so releasing a hold cannot manufacture a pass.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

/** @brief Reject the exact native error without trying another execution path. */
void requireHip(hipError_t result, const char* operation) {
    if (result == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(result));
    std::_Exit(2);
}
#define HIP_REQUIRE(call) requireHip((call), #call)

/** @brief State whether a copy's terminal has a real peer dependency. */
enum class Frontier { Independent, ExternalEvent, Compute };

/** @brief Retain one independently held graph and its exact launch FIFO. */
struct HeldGraph {
    hipStream_t stream{};
    hipGraph_t graph{};
    hipGraphExec_t executable{};
};

/** @brief Retain one independent DMA FIFO and its reusable ordering receipt. */
struct CopyLane {
    hipStream_t stream{};
    hipEvent_t terminal{};
};

/**
 * @brief Authenticate progress, exact bytes, event reuse and real dependencies.
 * @param argc Optional count and one explicit negative-control scenario.
 * @param argv Counts are test geometry, never native scheduler overrides.
 * @return Zero only if completion semantics and every payload byte are correct.
 */
int main(int argc, char** argv) {
    const unsigned held_count = argc >= 2 ? std::strtoul(argv[1], nullptr, 10) : 1;
    Frontier frontier = Frontier::Independent;
    if (argc == 3 && std::strcmp(argv[2], "--wait-frontier") == 0)
        frontier = Frontier::ExternalEvent;
    else if (argc == 3 && std::strcmp(argv[2], "--compute-frontier") == 0)
        frontier = Frontier::Compute;
    else if (argc > 2) return 2;
    if (held_count > 64 || (held_count == 0 && frontier != Frontier::Independent)) return 2;

    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    HIP_REQUIRE(hipSetDevice(0));
    constexpr unsigned lane_count = 32;
    constexpr unsigned reuse_generations = 3;
    constexpr std::size_t payload_bytes = 4 * 1024 * 1024 + 13;
    constexpr std::size_t guard_bytes = 64;
    constexpr std::size_t stride = payload_bytes + 2 * guard_bytes;
    constexpr std::uint8_t guard_value = 0xc7;
    std::uint64_t* control{};
    std::uint64_t* control_alias{};
    std::uint8_t* output{};
    std::uint8_t* output_alias{};
    std::uint8_t* source{};
    hipStream_t setup{};
    hipEvent_t prepared{};
    hipEvent_t held_frontier{};
    HIP_REQUIRE(hipHostMalloc(&control, 4096, hipHostMallocMapped));
    HIP_REQUIRE(hipHostGetDevicePointer(reinterpret_cast<void**>(&control_alias), control, 0));
    HIP_REQUIRE(hipHostMalloc(&output, lane_count * stride, hipHostMallocMapped));
    HIP_REQUIRE(hipHostGetDevicePointer(reinterpret_cast<void**>(&output_alias), output, 0));
    HIP_REQUIRE(hipMalloc(&source, payload_bytes));
    HIP_REQUIRE(hipStreamCreateWithFlags(&setup, hipStreamNonBlocking));
    HIP_REQUIRE(hipEventCreateWithFlags(&prepared, hipEventDisableTiming));
    HIP_REQUIRE(hipEventCreateWithFlags(&held_frontier, hipEventDisableTiming));
    std::memset(control, 0, 4096);
    std::memset(output, guard_value, lane_count * stride);
    HIP_REQUIRE(hipMemsetAsync(source, 0x53, payload_bytes, setup));
    HIP_REQUIRE(hipEventRecord(prepared, setup));
    HIP_REQUIRE(hipEventSynchronize(prepared)); // Fixture setup, not an inference edge.

    std::vector<HeldGraph> graphs(held_count);
    std::vector<CopyLane> copies(lane_count);
    for (auto& graph : graphs) {
        HIP_REQUIRE(hipStreamCreateWithFlags(&graph.stream, hipStreamNonBlocking));
        HIP_REQUIRE(hipStreamBeginCapture(graph.stream, hipStreamCaptureModeGlobal));
        HIP_REQUIRE(hipStreamWaitValue64(graph.stream, control_alias, 1, hipStreamWaitValueEq));
        HIP_REQUIRE(hipStreamWriteValue64(graph.stream, control_alias + 1, 1, 0));
        HIP_REQUIRE(hipStreamEndCapture(graph.stream, &graph.graph));
        HIP_REQUIRE(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    }
    for (auto& copy : copies) {
        HIP_REQUIRE(hipStreamCreateWithFlags(&copy.stream, hipStreamNonBlocking));
        HIP_REQUIRE(hipEventCreateWithFlags(&copy.terminal, hipEventDisableTiming));
    }

    // Cover submission itself becoming stuck, not only a later event query.
    // Cancellation is a negative-test cleanup boundary and never a pass path.
    std::atomic<bool> cancelled{false};
    std::jthread cancellation([&](std::stop_token stop) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!stop.stop_requested() && std::chrono::steady_clock::now() < limit)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!stop.stop_requested()) {
            cancelled.store(true, std::memory_order_release);
            std::atomic_ref<std::uint64_t>(*control).store(1, std::memory_order_release);
        }
    });
    for (auto& graph : graphs) HIP_REQUIRE(hipGraphLaunch(graph.executable, graph.stream));
    if (frontier == Frontier::ExternalEvent)
        HIP_REQUIRE(hipEventRecord(held_frontier, graphs.front().stream));

    /** @brief Verify each live extent and both guards after its exact receipt. */
    const auto verify_bytes = [&] {
        for (unsigned lane = 0; lane < lane_count; ++lane) {
            const auto* begin = output + lane * stride;
            if (!std::all_of(begin, begin + guard_bytes,
                             [](std::uint8_t b) { return b == guard_value; }) ||
                !std::all_of(begin + guard_bytes, begin + guard_bytes + payload_bytes,
                             [](std::uint8_t b) { return b == 0x53; }) ||
                !std::all_of(begin + guard_bytes + payload_bytes, begin + stride,
                             [](std::uint8_t b) { return b == guard_value; })) return false;
        }
        return true;
    };

    bool contract_passed = true;
    const unsigned generations = frontier == Frontier::Independent ? reuse_generations : 1;
    for (unsigned generation = 0; generation < generations; ++generation) {
        // Previous terminals certify CPU access before overwriting destinations.
        // Poisoning the entire live extent catches prematurely reused signals.
        std::memset(output, guard_value, lane_count * stride);
        for (unsigned lane = 0; lane < lane_count; ++lane) {
            auto& copy = copies[(lane + generation) % lane_count];
            const auto destination = output_alias + ((lane + generation) % lane_count) * stride + guard_bytes;
            HIP_REQUIRE(hipMemcpyAsync(destination, source, payload_bytes,
                                       hipMemcpyDeviceToDeviceNoCU, copy.stream));
            if (frontier == Frontier::ExternalEvent)
                HIP_REQUIRE(hipStreamWaitEvent(copy.stream, held_frontier, 0));
            else if (frontier == Frontier::Compute)
                HIP_REQUIRE(hipStreamWaitValue64(copy.stream, control_alias, 1, hipStreamWaitValueEq));
            HIP_REQUIRE(hipEventRecord(copy.terminal, copy.stream));
            HIP_REQUIRE(hipEventRecord(copy.terminal, copy.stream)); // Same-frontier coalescing.
        }
        unsigned complete = 0;
        const auto deadline = std::chrono::steady_clock::now() +
            (frontier == Frontier::Independent ? std::chrono::milliseconds(2000) : std::chrono::milliseconds(10));
        do {
            complete = 0;
            for (const auto& copy : copies) {
                const auto status = hipEventQuery(copy.terminal);
                if (status == hipSuccess) ++complete;
                else if (status != hipErrorNotReady) requireHip(status, "query exact DMA terminal");
            }
            if (frontier == Frontier::Independent && complete == lane_count) break;
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        const bool expected_completion = frontier == Frontier::Independent ? complete == lane_count : complete == 0;
        const bool independent_bytes = frontier != Frontier::Independent || (expected_completion && verify_bytes());
        contract_passed &= expected_completion && independent_bytes && !cancelled.load(std::memory_order_acquire);
        std::printf("held_graphs=%u generation=%u native_terminals=%u/%u contract=%s\n",
                    held_count, generation, complete, lane_count, contract_passed ? "PASS" : "FAIL");
        std::fflush(stdout);
        if (!contract_passed) break;
    }

    // Decide the result above before releasing the peer-owned frontier. This
    // final join merely makes cleanup safe, even with the unpatched runtime.
    std::atomic_ref<std::uint64_t>(*control).store(1, std::memory_order_release);
    cancellation.request_stop();
    cancellation.join();
    for (auto& graph : graphs) HIP_REQUIRE(hipStreamSynchronize(graph.stream));
    for (auto& copy : copies) HIP_REQUIRE(hipEventSynchronize(copy.terminal));
    const bool final_bytes = verify_bytes();
    std::printf("terminal_payload_and_guards=%s\n", final_bytes ? "PASS" : "FAIL");
    for (auto& graph : graphs) {
        HIP_REQUIRE(hipGraphExecDestroy(graph.executable));
        HIP_REQUIRE(hipGraphDestroy(graph.graph));
        HIP_REQUIRE(hipStreamDestroy(graph.stream));
    }
    for (auto& copy : copies) {
        HIP_REQUIRE(hipEventDestroy(copy.terminal));
        HIP_REQUIRE(hipStreamDestroy(copy.stream));
    }
    HIP_REQUIRE(hipEventDestroy(held_frontier));
    HIP_REQUIRE(hipEventDestroy(prepared));
    HIP_REQUIRE(hipStreamDestroy(setup));
    HIP_REQUIRE(hipFree(source));
    HIP_REQUIRE(hipHostFree(output));
    HIP_REQUIRE(hipHostFree(control));
    return contract_passed && final_bytes ? 0 : 1;
}
