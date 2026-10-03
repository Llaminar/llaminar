/**
 * @file hip_graph_empty_segment.hip
 * @brief Standalone ROCm 10 empty-segment completion-signal reproducer.
 *
 * Build with hipcc -O3 --offload-arch=gfx906 hip_graph_empty_segment.hip -o repro.
 * Run under an external 30-second watchdog: the unpatched runtime blocks at the
 * terminal observation of the first retained graph launch. No RCCL, MPI, model,
 * profiler, spin rendezvous, or scheduler override is involved.
 */
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/** @brief Fail immediately on a native API error, preserving its exact call. */
void check(hipError_t status, const char* operation) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    std::fflush(stderr);
    std::_Exit(2);
}
#define HIP(call) check((call), #call)

/** @brief Record one visit in a branch-owned slot without cross-branch sharing. */
__global__ void visit(std::uint32_t* receipts, unsigned slot) {
    ++receipts[slot];
}

/** @brief Exercise one captured graph through twenty ordered retained launches. */
int main() {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    constexpr unsigned waves = 8, width = 2, replays = 20;
    std::array<std::uint32_t, waves * width> host{};
    std::uint32_t* device{};
    hipStream_t stream{};
    hipEvent_t terminal{};
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    HIP(hipSetDevice(0));
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreateWithFlags(&terminal, hipEventDisableTiming));
    HIP(hipMalloc(&device, sizeof(host)));
    HIP(hipMemsetAsync(device, 0, sizeof(host), stream));
    HIP(hipGraphCreate(&graph, 0));
    hipGraphNode_t predecessor{};
    HIP(hipGraphAddEmptyNode(&predecessor, graph, nullptr, 0));
    for (unsigned wave = 0; wave < waves; ++wave) {
        std::array<hipGraphNode_t, width> branches{};
        for (unsigned branch = 0; branch < width; ++branch) {
            unsigned slot = wave * width + branch;
            void* arguments[]{&device, &slot};
            hipKernelNodeParams parameters{};
            parameters.func = reinterpret_cast<void*>(visit);
            parameters.gridDim = parameters.blockDim = dim3(1);
            parameters.kernelParams = arguments;
            HIP(hipGraphAddKernelNode(&branches[branch], graph, &predecessor, 1, &parameters));
        }
        // Empty nodes remain real dependency points even without GPU payload.
        HIP(hipGraphAddEmptyNode(&predecessor, graph, branches.data(), width));
    }
    HIP(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < replays; ++replay)
        HIP(hipGraphLaunch(executable, stream));
    // Only the terminal readback waits on the host; it is not a replay repair.
    HIP(hipMemcpyAsync(host.data(), device, sizeof(host), hipMemcpyDeviceToHost, stream));
    HIP(hipEventRecord(terminal, stream));
    HIP(hipEventSynchronize(terminal));
    unsigned errors = 0;
    for (unsigned slot = 0; slot < host.size(); ++slot) {
        if (host[slot] != replays) ++errors;
    }
    std::printf("empty-segment receipts: %s (%u errors)\n", errors ? "FAIL" : "PASS", errors);
    HIP(hipGraphExecDestroy(executable));
    HIP(hipGraphDestroy(graph));
    HIP(hipFree(device));
    HIP(hipEventDestroy(terminal));
    HIP(hipStreamDestroy(stream));
    return errors ? 1 : 0;
}
