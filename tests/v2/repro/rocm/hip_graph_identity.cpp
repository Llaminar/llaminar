/**
 * @file hip_graph_identity.cpp
 * @brief HIP-only concurrent, independently owned graph construction reproducer.
 *
 * Build with hipcc -O3 -std=c++20 -pthread. There are no custom kernels, models,
 * collectives, shared graphs, profilers or Llaminar dependencies. Every memset
 * node writes its own word; event self-intervals prove marker execution as well.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/** @brief Attribute a native error without abandoning the other thread barriers. */
void check(hipError_t result, const char* call) {
    if (result != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(result));
}
#define HIP(call) check((call), #call)

/** @brief Each worker exclusively owns its graph, stream, events and result bytes. */
struct Owner {
    hipStream_t stream{};
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    hipEvent_t terminal{};
    std::vector<hipEvent_t> markers;
    std::uint32_t* words{};

    /** @brief Retire one graph generation without rebinding its persistent buffers. */
    void clearGraph() {
        if (exec) HIP(hipGraphExecDestroy(exec));
        if (graph) HIP(hipGraphDestroy(graph));
        exec = nullptr;
        graph = nullptr;
        for (auto marker : markers) HIP(hipEventDestroy(marker));
        markers.clear();
    }

    /** @brief Retire diagnostic ownership on the same thread/device that created it. */
    ~Owner() {
        if (exec) (void)hipGraphExecDestroy(exec);
        if (graph) (void)hipGraphDestroy(graph);
        for (auto marker : markers) (void)hipEventDestroy(marker);
        if (terminal) (void)hipEventDestroy(terminal);
        if (words) (void)hipFree(words);
        if (stream) (void)hipStreamDestroy(stream);
    }
};

/** @brief Concurrent calls mutate only different graphs, not one shared HIP object. */
int main() {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    constexpr unsigned builders = 16, generations = 16, count = 2048;
    int devices = 0;
    HIP(hipGetDeviceCount(&devices));
    if (devices < 1) return 2;
    std::barrier boundary(static_cast<std::ptrdiff_t>(builders));
    std::array<std::string, builders> errors;
    std::vector<std::thread> threads;
    for (unsigned builder = 0; builder < builders; ++builder) threads.emplace_back([&, builder] {
        Owner owner;
        std::vector<std::uint32_t> host(count);
        try {
            HIP(hipSetDevice(builder % std::min(devices, 4)));
            HIP(hipStreamCreateWithFlags(&owner.stream, hipStreamNonBlocking));
            HIP(hipEventCreateWithFlags(&owner.terminal, hipEventDisableTiming));
            HIP(hipMalloc(&owner.words, count * sizeof(std::uint32_t)));
        } catch (const std::exception& error) { errors[builder] = error.what(); }
        for (unsigned generation = 0; generation < generations; ++generation) {
            // Failed workers still enter both barriers: no stranded peers.
            boundary.arrive_and_wait();
            if (errors[builder].empty()) try {
                owner.clearGraph();
                HIP(hipGraphCreate(&owner.graph, 0));
                hipGraphNode_t tail{};
                for (unsigned word = 0; word < count; ++word) {
                    if (word % 64 == 0 || word == count - 1) {
                        hipEvent_t marker{};
                        HIP(hipEventCreate(&marker));
                        owner.markers.push_back(marker);
                        hipGraphNode_t node{};
                        HIP(hipGraphAddEventRecordNode(&node, owner.graph,
                            tail ? &tail : nullptr, tail ? 1 : 0, marker));
                        tail = node;
                    }
                    hipMemsetParams params{};
                    params.dst = owner.words + word;
                    params.value = word + 1;
                    params.elementSize = sizeof(std::uint32_t);
                    params.width = params.height = 1;
                    hipGraphNode_t node{};
                    HIP(hipGraphAddMemsetNode(&node, owner.graph, &tail, 1, &params));
                    tail = node;
                }
                HIP(hipGraphInstantiate(&owner.exec, owner.graph, nullptr, nullptr, 0));
                for (unsigned replay = 0; replay < 2; ++replay) {
                    HIP(hipMemsetAsync(owner.words, 0, count * sizeof(std::uint32_t), owner.stream));
                    HIP(hipGraphLaunch(owner.exec, owner.stream));
                    HIP(hipMemcpyAsync(host.data(), owner.words, count * sizeof(std::uint32_t),
                                       hipMemcpyDeviceToHost, owner.stream));
                    HIP(hipEventRecord(owner.terminal, owner.stream));
                    HIP(hipEventSynchronize(owner.terminal));
                    for (unsigned word = 0; word < count; ++word) if (host[word] != word + 1)
                        throw std::runtime_error("omitted word " + std::to_string(word) +
                            " generation " + std::to_string(generation));
                    for (auto marker : owner.markers) {
                        float elapsed = -1;
                        HIP(hipEventElapsedTime(&elapsed, marker, marker));
                        if (elapsed != 0) throw std::runtime_error("marker did not execute");
                    }
                }
            } catch (const std::exception& error) { errors[builder] = error.what(); }
            boundary.arrive_and_wait();
        }
    });
    for (auto& thread : threads) thread.join();
    bool okay = true;
    for (unsigned builder = 0; builder < builders; ++builder) {
        if (!errors[builder].empty()) {
            std::fprintf(stderr, "builder %u: %s\n", builder, errors[builder].c_str());
            okay = false;
        }
    }
    std::printf("independent builders: %s\n", okay ? "PASS" : "FAIL");
    return okay ? 0 : 1;
}
