/**
 * @file CUDADecodeRoofline.cu
 * @brief Calibrate same-device read and arithmetic issue rates for CUDA decode.
 *
 * The probe owns one stream and captures its timed launches before sampling.
 * Host verification observes every output lane outside the measured interval.
 * Counter collection selects the same graph and one physical kernel launch.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    /** @brief Emit the exact local HC/FFN work coordinates without opening a CUDA device. */
    int cudaHCFFNRooflineWork(int argc, char** argv);
}

namespace
{
    /** @brief Abort a sample on the first failed CUDA operation. */
    void check(cudaError_t status, const char* operation)
    {
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
            std::exit(EXIT_FAILURE);
        }
    }

    #define CUDA_CHECK(operation) check((operation), #operation)

    constexpr unsigned threadsPerBlock = 256;
    constexpr unsigned launchesPerGraph = 16;
    constexpr unsigned samples = 21;
    constexpr unsigned dotIterations = 4096;
    constexpr unsigned dotChains = 8;
    constexpr unsigned permuteIterations = 4097;
    constexpr unsigned permuteChains = 8;
    constexpr unsigned mmaIterations = 2048;
    constexpr unsigned mmaChains = 8;
    constexpr unsigned issueIterations = 4097;
    constexpr unsigned issueChains = 8;
    constexpr unsigned fmaIterations = 4096;
    constexpr unsigned fmaChains = 8;

    enum class Kernel { Read, Dp4a, BytePermute, Imma, Logic, MixedInteger, Fma };

    /** @brief Generate distinct deterministic words for each input vector. */
    __host__ __device__ uint4 inputAt(unsigned index)
    {
        uint32_t word = index + 0x9e3779b9U;
        word = (word ^ (word >> 16)) * 0x7feb352dU;
        word = (word ^ (word >> 15)) * 0x846ca68bU;
        return {word, (word << 13) | (word >> 19), index ^ (word >> 5), word + index};
    }

    /** @brief Initialize the probe allocation before graph capture. */
    __global__ void prepareReadInput(uint4* input, unsigned vectors)
    {
        const unsigned stride = gridDim.x * blockDim.x;
        for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
             index < vectors; index += stride)
            input[index] = inputAt(index);
    }

    /** @brief Add modulo 2^32, independently in each vector component. */
    __host__ __device__ uint4 add(uint4 left, uint4 right)
    {
        return {left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w};
    }

    /**
     * @brief Issue aligned, coalesced 16-byte reads over a cache-sized or DRAM-sized set.
     * @param input Initialized vectors, rounded to complete stripes.
     * @param output One checked checksum per launch thread.
     * @param vectors Number of vectors per traversal.
     * @param passes Number of traversals per dispatch.
     */
    __global__ void decodeReadCeiling(const uint4* input, uint4* output,
                                     unsigned vectors, unsigned passes)
    {
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned stride = gridDim.x * blockDim.x;
        uint4 sums[4]{};
        for (unsigned pass = 0; pass < passes; ++pass)
        {
            for (unsigned index = thread; index < vectors; index += 4 * stride)
            {
                #pragma unroll
                for (unsigned group = 0; group < 4; ++group)
                {
                    const auto* address = input + index + group * stride;
                    const auto global = __cvta_generic_to_global(address);
                    uint4 value;
                    // Keep independent vector loads and ensure repeated L2 passes issue again.
                    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                        : "l"(global) : "memory");
                    sums[group] = add(sums[group], value);
                }
            }
        }
        output[thread] = add(add(sums[0], sums[1]), add(sums[2], sums[3]));
    }

    /**
     * @brief Drive independent signed DP4A chains using byte values 0 through 127.
     * @param output One checked sum of the chains per launch thread.
     */
    __global__ void decodeDp4aCeiling(uint32_t* output)
    {
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        const int left = static_cast<int>(thread & 127U) * 0x01010101;
        uint32_t accumulators[dotChains]{};
        #pragma unroll 1
        for (unsigned iteration = 0; iteration < dotIterations; ++iteration)
        {
            #pragma unroll
            for (unsigned chain = 0; chain < dotChains; ++chain)
            {
                const int right = static_cast<int>(chain + 1) * 0x01010101;
                asm volatile("dp4a.s32.s32 %0, %1, %2, %0;"
                    : "+r"(accumulators[chain]) : "r"(left), "r"(right));
            }
        }
        uint32_t sum = 0;
        #pragma unroll
        for (unsigned chain = 0; chain < dotChains; ++chain)
            sum += accumulators[chain];
        output[thread] = sum;
    }

    /**
     * @brief Measure independent scalar FP32 issue at the requested CTA geometry.
     * @param output One checksum per lane encoded as its exact FP32 bits.
     * @details Eight independent accumulators keep dependency latency out of
     * the issue reference. Integer-valued operands keep every accumulator and
     * the final checksum within the exact binary32 integer range.
     */
    __global__ void decodeFmaCeiling(uint32_t* output)
    {
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        const float left = static_cast<float>(thread & 63u);
        float accumulators[fmaChains]{};
        #pragma unroll 1
        for (unsigned iteration = 0; iteration < fmaIterations; ++iteration)
        {
            #pragma unroll
            for (unsigned chain = 0; chain < fmaChains; ++chain)
            {
                const float right = static_cast<float>(chain + 1);
                asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                    : "+f"(accumulators[chain]) : "f"(left), "f"(right));
            }
        }
        float sum = 0.0f;
        #pragma unroll
        for (unsigned chain = 0; chain < fmaChains; ++chain)
            sum += accumulators[chain];
        output[thread] = __float_as_uint(sum);
    }

    /**
     * @brief Drive independent PRMT chains with the byte selection used by packed IQ readers.
     * @param output One checked modular checksum per launch thread.
     * @details The selector rotates the four bytes right by one position.
     * Distinct seeds exercise each byte and keep the independent chains live.
     */
    __global__ void decodeBytePermuteCeiling(uint32_t* output)
    {
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t words[permuteChains];
        #pragma unroll
        for (unsigned chain = 0; chain < permuteChains; ++chain)
            words[chain] = inputAt(thread * permuteChains + chain).x;

        #pragma unroll 1
        for (unsigned iteration = 0; iteration < permuteIterations; ++iteration)
        {
            #pragma unroll
            for (unsigned chain = 0; chain < permuteChains; ++chain)
                asm volatile("prmt.b32 %0, %0, 0, 0x0321;" : "+r"(words[chain]));
        }

        uint32_t sum = 0;
        #pragma unroll
        for (unsigned chain = 0; chain < permuteChains; ++chain)
            sum += words[chain];
        output[thread] = sum;
    }

    /**
     * @brief Drive independent dense SM80+ INT8 tensor-core accumulator chains.
     * @param output Four S32 accumulator elements, summed across chains per lane.
     *
     * Uniform operand values within a warp give every result in the tile the
     * same independent closed-form value despite the native fragment layout.
     */
    __global__ void decodeImmaCeiling(uint4* output)
    {
#if __CUDA_ARCH__ >= 800
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        const int left = static_cast<int>(thread / 32 % 127 + 1) * 0x01010101;
        uint4 accumulators[mmaChains]{};
        #pragma unroll 1
        for (unsigned iteration = 0; iteration < mmaIterations; ++iteration)
        {
            #pragma unroll
            for (unsigned chain = 0; chain < mmaChains; ++chain)
            {
                const int right = static_cast<int>(chain + 1) * 0x01010101;
                auto& value = accumulators[chain];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
                    : "+r"(value.x), "+r"(value.y), "+r"(value.z), "+r"(value.w)
                    : "r"(left), "r"(right));
            }
        }
        uint4 sum{};
        #pragma unroll
        for (unsigned chain = 0; chain < mmaChains; ++chain)
            sum = add(sum, accumulators[chain]);
        output[thread] = sum;
#else
        // Fat binaries can also contain pre-Ampere images for the other modes.
        (void)output;
        __trap();
#endif
    }


    /**
     * @brief Measure independent Ampere integer arithmetic issue.
     * @tparam Mixed Issue two independent LOP3, a signed DP4A and an
     *         integer multiply-add for each chain.
     * @param output Checked modular sum of the results for each lane.
     * @details Independent chains exercise sustained issue. The odd iteration
     *          count leaves one XOR of each input mask in the final checksum.
     */
    template <bool Mixed>
    __global__ void decodeIntegerIssueCeiling(uint32_t* output)
    {
        const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
        const uint4 seed = inputAt(thread);
        uint32_t logic[issueChains];
        uint32_t second[issueChains];
        uint32_t dot[issueChains]{};
        uint32_t product[issueChains]{};
        const int packed = static_cast<int>(thread & 127U) * 0x01010101;
        #pragma unroll
        for (unsigned chain = 0; chain < issueChains; ++chain)
        {
            logic[chain] = seed.x + chain;
            if constexpr (Mixed)
                second[chain] = seed.y + chain;
        }

        #pragma unroll 1
        for (unsigned iteration = 0; iteration < issueIterations; ++iteration)
        {
            #pragma unroll
            for (unsigned chain = 0; chain < issueChains; ++chain)
            {
                const uint32_t c = seed.z + chain;
                asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                             : "+r"(logic[chain]) : "r"(seed.y), "r"(c));
                if constexpr (Mixed)
                {
                    const int coefficient = static_cast<int>(chain + 1) * 0x01010101;
                    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                                 : "+r"(second[chain]) : "r"(seed.x), "r"(c));
                    asm volatile("dp4a.s32.s32 %0, %1, %2, %0;"
                                 : "+r"(dot[chain]) : "r"(packed), "r"(coefficient));
                    asm volatile("mad.lo.u32 %0, %1, %2, %0;"
                                 : "+r"(product[chain]) : "r"(seed.w), "r"(c));
                }
            }
        }
        uint32_t sum = 0;
        #pragma unroll
        for (unsigned chain = 0; chain < issueChains; ++chain)
        {
            sum += logic[chain];
            if constexpr (Mixed)
                sum += second[chain] + dot[chain] + product[chain];
        }
        output[thread] = sum;
    }

    /// @brief All geometry and useful work for a single probe.
    struct Probe
    {
        std::string_view name;
        unsigned blocks;
        unsigned vectors;
        unsigned passes;
        Kernel kernel;

        /** @return Requested bytes, scalar-lane instruction count, or warp IMMA count per launch. */
        uint64_t count() const
        {
            if (kernel == Kernel::Logic || kernel == Kernel::MixedInteger)
                return uint64_t(blocks) * threadsPerBlock * issueIterations * issueChains *
                    (kernel == Kernel::MixedInteger ? 4u : 1u);
            if (kernel == Kernel::Dp4a)
                return uint64_t(blocks) * threadsPerBlock * dotIterations * dotChains;
            if (kernel == Kernel::Fma)
                return uint64_t(blocks) * threadsPerBlock * fmaIterations * fmaChains;
            if (kernel == Kernel::BytePermute)
                return uint64_t(blocks) * threadsPerBlock * permuteIterations * permuteChains;
            if (kernel == Kernel::Imma)
                return uint64_t(blocks) * (threadsPerBlock / 32) * mmaIterations * mmaChains;
            return uint64_t(vectors) * passes * sizeof(uint4);
        }
    };

    /**
     * @brief Capture, time and verify one probe with fixed addresses and work.
     * @param probe Benchmark mode and geometry.
     * @param stream The selected device's private nonblocking stream.
     */
    bool measure(const Probe& probe, cudaStream_t stream)
    {
        const unsigned threads = probe.blocks * threadsPerBlock;
        uint4* input = nullptr;
        uint4* output = nullptr;
        const bool scalarOutput = probe.kernel != Kernel::Read && probe.kernel != Kernel::Imma;
        const auto outputBytes = threads * (scalarOutput ? sizeof(uint32_t) : sizeof(uint4));
        CUDA_CHECK(cudaMalloc(&output, outputBytes));
        if (probe.kernel == Kernel::Read)
        {
            CUDA_CHECK(cudaMalloc(&input, size_t(probe.vectors) * sizeof(uint4)));
            prepareReadInput<<<probe.blocks, threadsPerBlock, 0, stream>>>(input, probe.vectors);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (unsigned launch = 0; launch < launchesPerGraph; ++launch)
        {
            if (probe.kernel == Kernel::Dp4a)
                decodeDp4aCeiling<<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<uint32_t*>(output));
            else if (probe.kernel == Kernel::Fma)
                decodeFmaCeiling<<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<uint32_t*>(output));
            else if (probe.kernel == Kernel::BytePermute)
                decodeBytePermuteCeiling<<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<uint32_t*>(output));
            else if (probe.kernel == Kernel::Logic)
                decodeIntegerIssueCeiling<false><<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<uint32_t*>(output));
            else if (probe.kernel == Kernel::MixedInteger)
                decodeIntegerIssueCeiling<true><<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<uint32_t*>(output));
            else if (probe.kernel == Kernel::Imma)
                decodeImmaCeiling<<<probe.blocks, threadsPerBlock, 0, stream>>>(output);
            else
                decodeReadCeiling<<<probe.blocks, threadsPerBlock, 0, stream>>>(
                    input, output, probe.vectors, probe.passes);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));

        cudaEvent_t start{}, end{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        for (unsigned warmup = 0; warmup < 5; ++warmup)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (unsigned sample = 0; sample < samples; ++sample)
        {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
            std::printf("%.*s,%u,%u,%u,%u,%u,%llu,%.6f\n",
                        int(probe.name.size()), probe.name.data(), sample, probe.blocks, threadsPerBlock,
                        probe.vectors, probe.passes, static_cast<unsigned long long>(probe.count()),
                        double(milliseconds) * 1000 / launchesPerGraph);
        }

        bool valid = true;
        if (scalarOutput)
        {
            std::vector<uint32_t> actual(threads);
            CUDA_CHECK(cudaMemcpyAsync(actual.data(), output, outputBytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (unsigned thread = 0; thread < threads; ++thread)
            {
                uint32_t expected = 0;
                if (probe.kernel == Kernel::Dp4a)
                {
                    constexpr unsigned sumOfCoefficients = dotChains * (dotChains + 1) / 2;
                    expected = (thread & 127U) * 4 * dotIterations * sumOfCoefficients;
                }
                else if (probe.kernel == Kernel::Fma)
                {
                    constexpr unsigned sumOfCoefficients = fmaChains * (fmaChains + 1) / 2;
                    const float value = static_cast<float>((thread & 63u) * fmaIterations * sumOfCoefficients);
                    expected = std::bit_cast<uint32_t>(value);
                }
                else if (probe.kernel == Kernel::Logic || probe.kernel == Kernel::MixedInteger)
                {
                    const auto seed = inputAt(thread);
                    static_assert(issueIterations % 2 == 1);
                    for (unsigned chain = 0; chain < issueChains; ++chain)
                    {
                        const uint32_t c = seed.z + chain;
                        expected += (seed.x + chain) ^ seed.y ^ c;
                        if (probe.kernel == Kernel::MixedInteger)
                        {
                            expected += (seed.y + chain) ^ seed.x ^ c;
                            expected += (thread & 127U) * 4 * (chain + 1) * issueIterations;
                            expected += seed.w * c * issueIterations;
                        }
                    }
                }
                else
                {
                    constexpr int shift = 8 * (permuteIterations % 4);
                    for (unsigned chain = 0; chain < permuteChains; ++chain)
                        expected += std::rotr(inputAt(thread * permuteChains + chain).x, shift);
                }
                valid &= actual[thread] == expected;
            }
        }
        else
        {
            std::vector<uint4> actual(threads);
            CUDA_CHECK(cudaMemcpyAsync(actual.data(), output, outputBytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (unsigned thread = 0; thread < threads; ++thread)
            {
                uint4 expected{};
                if (probe.kernel == Kernel::Read)
                {
                    for (unsigned index = thread; index < probe.vectors; index += threads)
                        expected = add(expected, inputAt(index));
                }
                else
                {
                    constexpr unsigned sumOfCoefficients = mmaChains * (mmaChains + 1) / 2;
                    const uint32_t value = (thread / 32 % 127 + 1) * 32 * mmaIterations * sumOfCoefficients;
                    expected = {value, value, value, value};
                }
                const auto result = actual[thread];
                valid &= result.x == expected.x * probe.passes && result.y == expected.y * probe.passes &&
                         result.z == expected.z * probe.passes && result.w == expected.w * probe.passes;
            }
        }
        std::printf("CHECK,%s,%u\n", valid ? "pass" : "fail", threads);
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        if (input)
            CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
        return valid;
    }
}

/** @brief Admit one explicit device and run exactly one reference regime. */
int main(int argc, char** argv)
{
    if (argc >= 2 && std::string_view(argv[1]) == "hc-ffn-work")
        return llaminar2::test::cudaHCFFNRooflineWork(argc, argv);
    int device = -1;
    if (argc != 3 && argc != 4)
    {
        std::fprintf(stderr, "Usage: %s <cuda-device> <dram|l2|dp4a|prmt|imma|lop3|int-mix|ffma> [blocks]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string_view ordinal(argv[1]);
    const auto parsed = std::from_chars(ordinal.data(), ordinal.data() + ordinal.size(), device);
    const std::string_view mode(argv[2]);
    if (parsed.ec != std::errc{} || parsed.ptr != ordinal.data() + ordinal.size() ||
        device < 0 || (mode != "dram" && mode != "l2" && mode != "dp4a" &&
                        mode != "prmt" && mode != "imma" && mode != "lop3" && mode != "int-mix" &&
                        mode != "ffma"))
        return EXIT_FAILURE;

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    char pciBusId[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));
    std::printf("DEVICE,%s,%s,sm_%d%d,%d,%d\n", pciBusId, properties.name,
                properties.major, properties.minor, properties.multiProcessorCount, properties.l2CacheSize);

    const auto sms = static_cast<unsigned>(properties.multiProcessorCount);
    const auto kernel = mode == "dp4a" ? Kernel::Dp4a : mode == "prmt" ? Kernel::BytePermute :
        mode == "imma" ? Kernel::Imma : mode == "lop3" ? Kernel::Logic :
        mode == "int-mix" ? Kernel::MixedInteger : mode == "ffma" ? Kernel::Fma : Kernel::Read;
    if (kernel == Kernel::Imma && properties.major < 8)
    {
        std::fprintf(stderr, "The dense INT8 m16n8k32 reference requires SM80 or newer\n");
        return EXIT_FAILURE;
    }
    Probe probe{.name = mode, .blocks = 4 * sms, .vectors = 0, .passes = 1, .kernel = kernel};
    const bool explicitBlocks = argc == 4;
    if (explicitBlocks)
    {
        const std::string_view argument(argv[3]);
        const auto count = std::from_chars(argument.data(), argument.data() + argument.size(), probe.blocks);
        if (count.ec != std::errc{} || count.ptr != argument.data() + argument.size() || !probe.blocks ||
            probe.blocks > static_cast<unsigned>(properties.maxGridSize[0]) ||
            probe.blocks > std::numeric_limits<unsigned>::max() / (4 * threadsPerBlock))
        {
            std::fprintf(stderr, "Invalid native CTA count\n");
            return EXIT_FAILURE;
        }
    }
    if (kernel == Kernel::Read)
    {
        const size_t l2Bytes = properties.l2CacheSize;
        const unsigned bytesPerBlock = 4 * threadsPerBlock * sizeof(uint4);
        const size_t minimumDramBytes = 8 * l2Bytes + probe.blocks * bytesPerBlock;
        const size_t targetBytes = mode == "dram"
            ? std::max(size_t{256} << 20, minimumDramBytes)
            : l2Bytes / 2;
        if (mode == "l2")
        {
            // Four interleaved loads per thread still have to fit in half L2.
            if (explicitBlocks && probe.blocks > targetBytes / bytesPerBlock)
            {
                std::fprintf(stderr, "Requested CTA count cannot address a complete stripe inside half L2\n");
                return EXIT_FAILURE;
            }
            if (!explicitBlocks)
            {
                probe.blocks = std::min<size_t>(probe.blocks, targetBytes / bytesPerBlock);
                probe.blocks = probe.blocks / sms * sms;
            }
        }
        if (!probe.blocks)
        {
            std::fprintf(stderr, "Device L2 cannot fit a complete launch wave\n");
            return EXIT_FAILURE;
        }
        const unsigned stripe = 4 * threadsPerBlock * probe.blocks;
        probe.vectors = targetBytes / sizeof(uint4) / stripe * stripe;
        probe.passes = mode == "dram" ? 1 : 128;
    }
    std::puts("probe,sample,blocks,threads,vectors,passes,count,us_per_launch");
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const bool valid = measure(probe, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
