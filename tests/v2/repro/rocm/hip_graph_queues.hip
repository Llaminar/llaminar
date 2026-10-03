/**
 * @file hip_graph_queues.hip
 * @brief HIP-only physical queue placement reproducer, not a timing assertion.
 *
 * Three independent graph branches can collapse onto two physical queues while
 * four queues are available. An AMD queue-pointer builtin observes the actual
 * executing queue; empty fork/join nodes deliberately use classic scheduling.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

/** @brief Stop immediately after a failing native operation. */
void check(hipError_t result, const char* call) {
    if (result == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", call, hipGetErrorString(result));
    std::fflush(stderr);
    std::_Exit(2);
}
#define HIP(call) check((call), #call)

/**
 * @brief Print an optional host lifecycle marker for a stalled native replay.
 * @param operation Exact API boundary about to be entered or just completed.
 * @param width Number of independent graph branches.
 * @param replay Retained replay index; construction uses zero.
 *
 * Leave HIP_REPRO_TRACE unset for qualification. Synchronous logging changes
 * host submission timing, so traced runs isolate a failure but never certify
 * performance or stability.
 */
void trace(const char* operation, unsigned width, unsigned replay) {
    if (std::getenv("HIP_REPRO_TRACE") == nullptr) return;
    std::fprintf(stderr, "%s width=%u replay=%u\n", operation, width, replay);
    std::fflush(stderr);
}

/** @brief Record physical queue identity without cross-branch communication. */
__global__ void recordQueue(std::uint64_t* result, unsigned slot) {
    result[slot] = reinterpret_cast<std::uint64_t>(__builtin_amdgcn_queue_ptr());
}

/** @brief Materialize ordinary stream owners, then test one retained graph. */
bool run(bool skewed) {
    constexpr unsigned waves = 8;
    std::uint64_t* result{};
    std::array<std::uint64_t, 32> host{};
    hipEvent_t terminal{};
    std::vector<hipStream_t> streams;
    std::vector<std::uint64_t> physical;
    HIP(hipMalloc(&result, sizeof(host)));
    HIP(hipEventCreateWithFlags(&terminal, hipEventDisableTiming));
    const auto read = [&](hipStream_t stream, unsigned words) {
        HIP(hipMemcpyAsync(host.data(), result, words * sizeof(std::uint64_t), hipMemcpyDeviceToHost, stream));
        HIP(hipEventRecord(terminal, stream));
        HIP(hipEventSynchronize(terminal));
    };
    for (unsigned i = 0; i < 20; ++i) {
        hipStream_t stream{};
        HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        streams.push_back(stream);
        hipLaunchKernelGGL(recordQueue, dim3(1), dim3(1), 0, stream, result, 0);
        HIP(hipGetLastError());
        read(stream, 1);
        physical.push_back(host[0]);
    }
    const auto available = std::set<std::uint64_t>(physical.begin(), physical.end()).size();
    std::printf("available=%zu distribution=%s\n", available, skewed ? "skewed" : "uniform");
    if (available < 4) {
        std::fprintf(stderr, "This shape requires four materialized native queues.\n");
        std::_Exit(2);
    }
    if (skewed) {
        std::set<std::uint64_t> kept;
        for (unsigned i = 0; i < streams.size(); ++i) {
            const bool first = kept.insert(physical[i]).second;
            if (!first && physical[i] != physical.front()) {
                HIP(hipStreamDestroy(streams[i]));
                streams[i] = nullptr;
            }
        }
        std::erase(streams, nullptr); // One queue has many owners; each other has one.
    }
    bool okay = true;
    for (unsigned width : {2u, 3u, 4u}) {
        hipGraph_t graph{};
        hipGraphExec_t exec{};
        HIP(hipGraphCreate(&graph, 0));
        hipGraphNode_t previous{};
        HIP(hipGraphAddEmptyNode(&previous, graph, nullptr, 0));
        for (unsigned wave = 0; wave < waves; ++wave) {
            std::array<hipGraphNode_t, 4> branches{};
            for (unsigned branch = 0; branch < width; ++branch) {
                unsigned slot = wave * width + branch;
                void* args[]{&result, &slot};
                hipKernelNodeParams params{};
                params.func = reinterpret_cast<void*>(recordQueue);
                params.gridDim = params.blockDim = dim3(1);
                params.kernelParams = args;
                HIP(hipGraphAddKernelNode(&branches[branch], graph, &previous, 1, &params));
            }
            HIP(hipGraphAddEmptyNode(&previous, graph, branches.data(), width));
        }
        trace("instantiate begin", width, 0);
        HIP(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        trace("instantiate end", width, 0);
        for (unsigned replay = 0; replay < 20; ++replay) {
            const auto stream = streams[replay % streams.size()];
            HIP(hipMemsetAsync(result, 0, width * waves * sizeof(std::uint64_t), stream));
            trace("launch begin", width, replay);
            HIP(hipGraphLaunch(exec, stream));
            trace("launch end", width, replay);
            read(stream, width * waves);
            trace("terminal read complete", width, replay);
            for (unsigned wave = 0; wave < waves; ++wave) {
                const auto first = host.begin() + wave * width;
                const auto count = std::set<std::uint64_t>(first, first + width).size();
                if (count != width || std::find(first, first + width, 0) != first + width) {
                    std::printf("FAIL width=%u replay=%u wave=%u distinct_queues=%zu\n", width, replay, wave, count);
                    okay = false;
                }
            }
        }
        HIP(hipGraphExecDestroy(exec));
        HIP(hipGraphDestroy(graph));
    }
    HIP(hipFree(result));
    HIP(hipEventDestroy(terminal));
    for (auto stream : streams) HIP(hipStreamDestroy(stream));
    return okay;
}

/** @brief Check both ordinary and uneven owner counts without changing queue limits. */
int main() {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    HIP(hipSetDevice(0));
    const bool uniform = run(false);
    const bool skewed = run(true);
    return uniform && skewed ? 0 : 1;
}
