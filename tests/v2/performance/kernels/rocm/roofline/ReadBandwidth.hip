/**
 * @file ReadBandwidth.hip
 * @brief Event-timed and byte-checked gfx906 aligned read calibration.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP(call) do { const auto status = (call); if (status != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(status)); std::exit(1); } } while (0)

using Word = uint32_t;
using Vector = Word __attribute__((ext_vector_type(4)));
static_assert(sizeof(Vector) == 16);

/** Fill each physical vector with four distinct, deterministic integer words. */
__host__ __device__ Vector dataAt(Word i, Word seed) {
    Word x = i + seed;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return {x, (x << 11) | (x >> 21), i ^ (x >> 2), x + 0x9e3779b9U};
}

__global__ void initialize(Vector* output, Word words, Word seed) {
    for (Word i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += blockDim.x * gridDim.x) {
        output[i] = dataAt(i, seed);
    }
}

/** Stripe through an aligned read-only input. Each issued vector contributes to checked output.
 * Four accumulators expose independent memory operations on wide spans. With one group,
 * the same kernel structure can cover the 512 KiB set while activating 128 CTAs.
 */
template<unsigned Groups>
__global__ void readOnly(const Vector* __restrict__ input, Vector* __restrict__ output,
                         Word vectors, Word passes) {
    const Word id = blockIdx.x * blockDim.x + threadIdx.x;
    const Word stride = blockDim.x * gridDim.x;
    Vector partial[Groups]{};
    for (Word p = 0; p < passes; ++p) {
        // Prevent CSE of a repeated resident sweep without changing the load's cache flags.
        asm volatile("" ::: "memory");
        for (Word i = id; i < vectors; i += Groups * stride) {
            #pragma unroll
            for (unsigned group = 0; group < Groups; ++group) {
                partial[group] += input[i + group * stride];
            }
        }
    }
    Vector sum{};
    #pragma unroll
    for (unsigned group = 0; group < Groups; ++group) {
        sum += partial[group];
    }
    output[id] = sum;
}

struct Case {
    size_t bytes;
    Word passes;
    unsigned blocks;
    unsigned groups;
};

void run(Case c, hipStream_t stream, unsigned round) {
    constexpr unsigned lanes = 256;
    const auto vectors = static_cast<Word>(c.bytes / sizeof(Vector));
    const unsigned threads = c.blocks * lanes;
    if (vectors % (threads * c.groups)) std::abort();
    const size_t outputBytes = threads * sizeof(Vector);
    Vector* input;
    Vector* output;
    hipEvent_t start, end;
    HIP(hipMalloc(&input, c.bytes));
    HIP(hipMalloc(&output, outputBytes));
    HIP(hipEventCreate(&start));
    HIP(hipEventCreate(&end));
    const Word seed = 137U * (round + 1);
    initialize<<<256, lanes, 0, stream>>>(input, vectors, seed);
    HIP(hipGetLastError());
    auto launch = [&] {
        if (c.groups == 4) {
            readOnly<4><<<c.blocks, lanes, 0, stream>>>(input, output, vectors, c.passes);
        } else {
            readOnly<1><<<c.blocks, lanes, 0, stream>>>(input, output, vectors, c.passes);
        }
        HIP(hipGetLastError());
    };
    for (int warm = 0; warm < 4; ++warm) launch();
    HIP(hipStreamSynchronize(stream));

    constexpr unsigned samples = 13;
    constexpr unsigned batch = 12;
    for (unsigned sample = 0; sample < samples; ++sample) {
        HIP(hipEventRecord(start, stream));
        for (unsigned j = 0; j < batch; ++j) launch();
        HIP(hipEventRecord(end, stream));
        HIP(hipEventSynchronize(end));
        float elapsed = 0.0f;
        HIP(hipEventElapsedTime(&elapsed, start, end));
        const double us = static_cast<double>(elapsed) * 1000 / batch;
        const double gbps = static_cast<double>(c.bytes) * c.passes / (us * 1000);
        std::printf("READ,%u,%zu,%u,%u,%u,%u,%.6f,%.4f\n",
                    round, c.bytes, c.passes, c.blocks, c.groups, sample, us, gbps);
    }

    std::vector<Vector> actual(threads);
    HIP(hipMemcpyAsync(actual.data(), output, outputBytes, hipMemcpyDeviceToHost, stream));
    HIP(hipStreamSynchronize(stream));
    for (unsigned id = 0; id < threads; ++id) {
        Vector expected{};
        for (Word v = id; v < vectors; v += threads) expected += dataAt(v, seed);
        expected *= c.passes;
        for (unsigned k = 0; k < 4; ++k) {
            if (actual[id][k] != expected[k]) {
                std::fprintf(stderr, "bad checksum %u %u: %08x != %08x\n", id, k, actual[id][k], expected[k]);
                std::exit(2);
            }
        }
    }
    std::printf("CHECK,%u,%zu,%u,%u,%u,%u\n", round, c.bytes, c.passes, c.blocks, c.groups, threads * 4);
    HIP(hipEventDestroy(end));
    HIP(hipEventDestroy(start));
    HIP(hipFree(output));
    HIP(hipFree(input));
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const unsigned round = std::strtoul(argv[1], nullptr, 10);
    int count;
    HIP(hipGetDeviceCount(&count));
    if(count!=1)return 2;
    int chosen = -1;
    for (int id = 0; id < count; ++id) {
        char pci[32]{};
        HIP(hipDeviceGetPCIBusId(pci, sizeof(pci), id));
        if (std::strcmp(pci, argv[2]) == 0) chosen = id;
    }
    if (chosen < 0) return 3;
    HIP(hipSetDevice(chosen));
    hipDeviceProp_t p{};
    HIP(hipGetDeviceProperties(&p, chosen));
    if (std::strncmp(p.gcnArchName, "gfx906", 6) != 0) return 4;
    std::printf("DEVICE,%s,%s,%s,%d,%d,%d\n",argv[2],
                p.name, p.gcnArchName, p.multiProcessorCount, p.l2CacheSize, p.memoryClockRate);
    constexpr size_t MiB = 1U << 20;
    std::vector<Case> cases{
        {512U << 10, 128, 128, 1}, {1 * MiB, 96, 128, 1},
        {2 * MiB, 64, 128, 4}, {4 * MiB, 32, 256, 4},
        {32 * MiB, 4, 256, 4}, {64 * MiB, 4, 256, 4},
        {128 * MiB, 2, 256, 4}, {256 * MiB, 1, 256, 4},
        {256 * MiB, 1, 128, 4}, {256 * MiB, 1, 512, 4},
    };
    if (round & 1) std::reverse(cases.begin(), cases.end());
    hipStream_t stream;
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (const Case c : cases) run(c, stream, round);
    HIP(hipStreamDestroy(stream));
    return 0;
}
