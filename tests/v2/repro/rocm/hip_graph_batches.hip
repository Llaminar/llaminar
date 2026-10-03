/**
 * @file hip_graph_batches.hip
 * @brief HIP-only all-kernel graph reproducer for AQL batch publication/admission.
 *
 * Four independent 512-node branches and 512 ordered retained replays repeatedly
 * wrap the native queue. Every node writes a sticky single-writer receipt, not
 * just a final checksum. No model, RCCL, MPI, profiler or Llaminar is required.
 */
#include <hip/hip_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

constexpr unsigned lanes = 64, length = 512, width = 4, replays = 512;
constexpr std::uint32_t saltMultiplier = 2654435761u;

/** @brief One node/lane owns its value and sticky replay/order evidence. */
struct Receipt { std::uint32_t value, visits, errors; };

/** @brief Shared modulo-32-bit arithmetic, independent of floating-point state. */
__host__ __device__ std::uint32_t mix(std::uint32_t value, std::uint32_t salt) {
    return ((value << 7) | (value >> 25)) * 1664525u + salt + 1013904223u;
}

/** @brief Closed-form branch prefix; no host replay oracle is needed. */
__host__ __device__ std::uint32_t prefix(unsigned slot) {
    if (!slot) return 0;
    const auto count = (slot - 1) % length + 1;
    return (count * (2 * slot - count + 1) / 2) * saltMultiplier;
}

/** @brief Publish one new GPU-owned epoch on the root's exact graph dependency. */
__global__ void begin(Receipt* receipts, unsigned* epoch) {
    if (threadIdx.x == 0) ++*epoch;
    __syncthreads();
    auto& r = receipts[threadIdx.x];
    r.errors |= r.visits != *epoch - 1;
    r.value = mix(*epoch, threadIdx.x);
    ++r.visits;
}

/** @brief Validate every immutable argument and predecessor before extending it. */
__global__ void extend(Receipt* receipts, const unsigned* epoch,
                       unsigned slot, unsigned predecessor, std::uint32_t salt) {
    const auto e = *epoch;
    const auto previous = receipts[predecessor * lanes + threadIdx.x];
    auto& r = receipts[slot * lanes + threadIdx.x];
    const auto expected = (slot - 1) % length == 0 ? 0 : slot - 1;
    r.errors |= previous.errors | (previous.visits != e) | (r.visits != e - 1) |
        (predecessor != expected) | (salt != slot * saltMultiplier) |
        (previous.value != mix(e, threadIdx.x) + prefix(expected));
    r.value = previous.value + salt;
    ++r.visits;
}

/** @brief Join all four actual branch receipts through explicit graph edges. */
__global__ void join(Receipt* receipts, const unsigned* epoch) {
    auto& r = receipts[(1 + width * length) * lanes + threadIdx.x];
    const auto e = *epoch;
    r.errors |= r.visits != e - 1;
    auto value = mix(e, threadIdx.x);
    for (unsigned branch = 0; branch < width; ++branch) {
        const auto previous = receipts[((branch + 1) * length) * lanes + threadIdx.x];
        r.errors |= previous.errors | (previous.visits != e) |
            (previous.value != mix(e, threadIdx.x) + prefix((branch + 1) * length));
        value = mix(value, previous.value);
    }
    r.value = value;
    ++r.visits;
}

/** @brief A native API failure is fatal and attributed to its call. */
void check(hipError_t result, const char* call) {
    if (result == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", call, hipGetErrorString(result));
    std::fflush(stderr);
    std::_Exit(2);
}
#define HIP(call) check((call), #call)

/** @brief The sole setup/terminal observer; never a per-node or per-replay wait. */
void wait(hipEvent_t event) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) return;
        if (status != hipErrorNotReady) check(status, "hipEventQuery");
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "packet graph terminal stalled\n");
            std::fflush(stderr);
            std::_Exit(124);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/** @brief Submit one all-kernel graph without allocations or recapture during replay. */
int main() {
    setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
    setenv("HSA_USE_SVM", "0", 0);
    HIP(hipSetDevice(0));
    std::array<hipStream_t, 2> streams{};
    std::array<hipEvent_t, 2> ordered{};
    hipEvent_t terminal{};
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    Receipt* receipts{};
    unsigned* epoch{};
    std::vector<Receipt> host((2 + width * length) * lanes);
    const auto bytes = host.size() * sizeof(Receipt);
    for (auto& stream : streams) HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (auto& event : ordered) HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    HIP(hipEventCreateWithFlags(&terminal, hipEventDisableTiming));
    HIP(hipMalloc(&receipts, bytes));
    HIP(hipMalloc(&epoch, sizeof(unsigned)));
    HIP(hipMemsetAsync(receipts, 0, bytes, streams[0]));
    HIP(hipMemsetAsync(epoch, 0, sizeof(unsigned), streams[0]));
    HIP(hipEventRecord(terminal, streams[0]));
    wait(terminal);
    HIP(hipGraphCreate(&graph, 0));
    const auto add = [&](void* function, const hipGraphNode_t* deps, std::size_t count, void** args) {
        hipKernelNodeParams params{};
        params.func = function;
        params.gridDim = dim3(1);
        params.blockDim = dim3(lanes);
        params.kernelParams = args;
        hipGraphNode_t node{};
        HIP(hipGraphAddKernelNode(&node, graph, deps, count, &params));
        return node;
    };
    void* rootArgs[]{&receipts, &epoch};
    const auto root = add(reinterpret_cast<void*>(begin), nullptr, 0, rootArgs);
    std::array<hipGraphNode_t, width> tails{};
    for (unsigned branch = 0; branch < width; ++branch) {
        auto tail = root;
        unsigned predecessor = 0;
        for (unsigned step = 0; step < length; ++step) {
            unsigned slot = 1 + branch * length + step;
            std::uint32_t salt = slot * saltMultiplier;
            void* args[]{&receipts, &epoch, &slot, &predecessor, &salt};
            tail = add(reinterpret_cast<void*>(extend), &tail, 1, args);
            predecessor = slot;
        }
        tails[branch] = tail;
    }
    add(reinterpret_cast<void*>(join), tails.data(), tails.size(), rootArgs);
    HIP(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < replays; ++replay) {
        const auto stream = streams[replay % streams.size()];
        if (replay) HIP(hipStreamWaitEvent(stream, ordered[(replay - 1) % ordered.size()], 0));
        HIP(hipGraphLaunch(exec, stream));
        HIP(hipEventRecord(ordered[replay % ordered.size()], stream));
    }
    const auto last = streams[(replays - 1) % streams.size()];
    HIP(hipMemcpyAsync(host.data(), receipts, bytes, hipMemcpyDeviceToHost, last));
    HIP(hipEventRecord(terminal, last));
    wait(terminal);
    bool okay = true;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        const auto base = mix(replays, lane);
        auto joined = base;
        for (unsigned slot = 0; slot <= width * length; ++slot) {
            const auto& r = host[slot * lanes + lane];
            if (r.visits != replays || r.errors || r.value != base + prefix(slot)) {
                std::printf("FAIL slot=%u lane=%u visits=%u errors=%u\n", slot, lane, r.visits, r.errors);
                okay = false;
            }
        }
        for (unsigned branch = 0; branch < width; ++branch)
            joined = mix(joined, base + prefix((branch + 1) * length));
        const auto& r = host[(1 + width * length) * lanes + lane];
        okay &= r.visits == replays && r.errors == 0 && r.value == joined;
    }
    HIP(hipGraphExecDestroy(exec));
    HIP(hipGraphDestroy(graph));
    HIP(hipFree(epoch));
    HIP(hipFree(receipts));
    HIP(hipEventDestroy(terminal));
    for (auto event : ordered) HIP(hipEventDestroy(event));
    for (auto stream : streams) HIP(hipStreamDestroy(stream));
    std::printf("packet receipts: %s\n", okay ? "PASS" : "FAIL");
    return okay ? 0 : 1;
}
