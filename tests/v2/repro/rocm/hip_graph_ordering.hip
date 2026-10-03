/**
 * @file hip_graph_ordering.hip
 * @brief Standalone HIP-only entry/exit ordering reproducer for upstream reports.
 *
 * Build with hipcc -O3 -std=c++20 --offload-arch=gfx906. Run with entry,
 * exit, or child-exit. No model, collective, profiler, or Llaminar is linked.
 * Setup and terminal observations may wait; replay itself has no host waits.
 */
#include <hip/hip_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <thread>
#include <vector>

/** @brief Fail at the exact native call instead of continuing after an error. */
void check(hipError_t result, const char* call) {
    if (result == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", call, hipGetErrorString(result));
    std::fflush(stderr);
    std::_Exit(2);
}
#define HIP(call) check((call), #call)

/** @brief Observe only setup/final completion with a bounded diagnostic wait. */
void wait(hipEvent_t event) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) return;
        if (status != hipErrorNotReady) check(status, "hipEventQuery");
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "terminal event did not complete in 30 seconds\n");
            std::fflush(stderr);
            std::_Exit(124);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/** @brief Retain sticky per-root evidence, so a good replay cannot hide a bad one. */
struct Receipt { unsigned epoch, visits, errors; };

/** @brief GPU-owned producer; the bounded delay makes bypass observable. */
__global__ void advance(unsigned* epoch) {
    if (threadIdx.x == 0) {
        const auto start = clock64();
        while (clock64() - start < 5000000ull) {}
        ++*epoch;
    }
}

/** @brief Every root independently checks the launch stream's producer epoch. */
__global__ void observe(const unsigned* epoch, Receipt* receipts, unsigned root) {
    auto& r = receipts[root * 64 + threadIdx.x];
    const auto value = *epoch;
    r.errors |= value != r.visits + 1;
    r.epoch = value;
    ++r.visits;
}

/** @brief Independent roots never wait for one another; one is deliberately slower. */
__global__ void finish(unsigned* done, unsigned root, unsigned delayed) {
    if (threadIdx.x == 0) {
        if (root == delayed) {
            const auto start = clock64();
            while (clock64() - start < 60000000ull) {}
        }
        done[root] = 1;
    }
}

/** @brief Record premature completion on device, before any host synchronization. */
__global__ void successor(const unsigned* done, unsigned* missing, unsigned width) {
    unsigned count = 0;
    for (unsigned root = 0; root < width; ++root) count += done[root] != 1;
    missing[threadIdx.x] = count;
}

/** @brief Add one kernel with argument values copied by HIP node construction. */
hipGraphNode_t kernel(hipGraph_t graph, void* function, void** args) {
    hipKernelNodeParams params{};
    params.func = function;
    params.gridDim = dim3(1);
    params.blockDim = dim3(64);
    params.kernelParams = args;
    hipGraphNode_t node{};
    HIP(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    return node;
}

/** @brief Check each independent root after twenty same-stream producer/graph pairs. */
bool entry(unsigned width) {
    hipStream_t stream{};
    hipEvent_t event{};
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    unsigned* epoch{};
    Receipt* receipts{};
    std::vector<Receipt> host(width * 64);
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    HIP(hipMalloc(&epoch, sizeof(*epoch)));
    HIP(hipMalloc(&receipts, host.size() * sizeof(Receipt)));
    HIP(hipMemsetAsync(epoch, 0, sizeof(*epoch), stream));
    HIP(hipMemsetAsync(receipts, 0, host.size() * sizeof(Receipt), stream));
    HIP(hipEventRecord(event, stream));
    wait(event); // Initialization is complete before graph construction.
    HIP(hipGraphCreate(&graph, 0));
    for (unsigned root = 0; root < width; ++root) {
        void* args[]{&epoch, &receipts, &root};
        kernel(graph, reinterpret_cast<void*>(observe), args);
    }
    HIP(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < 20; ++replay) {
        hipLaunchKernelGGL(advance, dim3(1), dim3(64), 0, stream, epoch);
        HIP(hipGetLastError());
        HIP(hipGraphLaunch(exec, stream));
    }
    HIP(hipMemcpyAsync(host.data(), receipts, host.size() * sizeof(Receipt), hipMemcpyDeviceToHost, stream));
    HIP(hipEventRecord(event, stream));
    wait(event);
    bool okay = true;
    for (unsigned root = 0; root < width; ++root) {
        const auto& r = host[root * 64];
        std::printf("entry width=%u root=%u epoch=%u visits=%u errors=%u\n", width, root, r.epoch, r.visits, r.errors);
        for (unsigned lane = 0; lane < 64; ++lane) {
            const auto& value = host[root * 64 + lane];
            okay &= value.epoch == 20 && value.visits == 20 && value.errors == 0;
        }
    }
    HIP(hipDeviceSynchronize()); // Diagnostic teardown, after immutable results.
    HIP(hipGraphExecDestroy(exec));
    HIP(hipGraphDestroy(graph));
    HIP(hipFree(receipts));
    HIP(hipFree(epoch));
    HIP(hipEventDestroy(event));
    HIP(hipStreamDestroy(stream));
    return okay;
}

/** @brief Delay each root in turn, checking the following same-stream operation. */
bool exitFrontier(unsigned width, unsigned delayed, bool child) {
    hipStream_t stream{};
    hipEvent_t event{};
    hipGraph_t graph{}, parent{};
    hipGraphExec_t exec{};
    unsigned *done{}, *missing{};
    std::array<unsigned, 64> host{};
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    HIP(hipMalloc(&done, width * sizeof(unsigned)));
    HIP(hipMalloc(&missing, sizeof(host)));
    HIP(hipMemsetAsync(done, 0, width * sizeof(unsigned), stream));
    HIP(hipEventRecord(event, stream));
    wait(event); // This case isolates exit ordering, not an initialization race.
    HIP(hipGraphCreate(&graph, 0));
    for (unsigned root = 0; root < width; ++root) {
        void* args[]{&done, &root, &delayed};
        kernel(graph, reinterpret_cast<void*>(finish), args);
    }
    auto executableGraph = graph;
    if (child) {
        HIP(hipGraphCreate(&parent, 0));
        hipGraphNode_t node{};
        HIP(hipGraphAddChildGraphNode(&node, parent, nullptr, 0, graph));
        executableGraph = parent;
    }
    HIP(hipGraphInstantiate(&exec, executableGraph, nullptr, nullptr, 0));
    HIP(hipGraphLaunch(exec, stream));
    hipLaunchKernelGGL(successor, dim3(1), dim3(64), 0, stream, done, missing, width);
    HIP(hipGetLastError());
    HIP(hipMemcpyAsync(host.data(), missing, sizeof(host), hipMemcpyDeviceToHost, stream));
    HIP(hipEventRecord(event, stream));
    wait(event);
    bool okay = true;
    for (auto value : host) okay &= value == 0;
    std::printf("exit child=%u width=%u delayed=%u unfinished=%u\n", child, width, delayed, host[0]);
    // A defective runtime can leave a root running after the terminal event.
    // Joining only here protects teardown; it cannot change the captured result.
    HIP(hipDeviceSynchronize());
    HIP(hipGraphExecDestroy(exec));
    if (parent) HIP(hipGraphDestroy(parent));
    HIP(hipGraphDestroy(graph));
    HIP(hipFree(missing));
    HIP(hipFree(done));
    HIP(hipEventDestroy(event));
    HIP(hipStreamDestroy(stream));
    return okay;
}

/** @brief Select one independent ordering contract using the stock public HIP API. */
int main(int argc, char** argv) {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    HIP(hipSetDevice(0));
    const std::string_view mode = argc > 1 ? argv[1] : "entry";
    bool okay = true;
    if (mode == "entry") {
        for (unsigned width : {2u, 3u, 4u, 8u}) okay &= entry(width);
    } else if (mode == "exit" || mode == "child-exit") {
        for (unsigned delayed = 0; delayed < 14; ++delayed)
            okay &= exitFrontier(14, delayed, mode == "child-exit");
    } else {
        std::fprintf(stderr, "usage: %s entry|exit|child-exit\n", argv[0]);
        return 2;
    }
    return okay ? 0 : 1;
}
