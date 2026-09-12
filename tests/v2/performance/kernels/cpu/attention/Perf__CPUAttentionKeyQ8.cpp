/**
 * @file Perf__CPUAttentionKeyQ8.cpp
 * @brief Isolated native AQ8 encode/decode latency and profiler attachment point.
 *
 * Each process measures one head width, ISA and operation. Allocation, fixture
 * construction, byte authentication and warmup precede the timed interval.
 * Output is compact CSV; perf/ISA evidence belongs to a separate invocation.
 * This harness measures kernel economics, not whole-model throughput or parity.
 */
#include "kernels/cpu/attention/CPUAttentionKeyQ8.h"
#include "../../../../integration/kernels/AttentionKeyQ8DeviceTestCommon.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string_view>
#include <vector>

namespace
{
using namespace llaminar2;
using cpu::attention_key_q8::ISA;

/** @brief Exactly one production operation per timing/profiler launch. */
enum class Operation { Encode, Decode };

/** @brief Validate a complete positive integer argument without partial parsing. */
int positive(std::string_view text)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value <= 0)
        throw std::invalid_argument("expected a positive integer argument");
    return value;
}

/**
 * @brief Repeated production kernel work with a stable, named profiling symbol.
 * @param source Immutable original keys, one contiguous head per block.
 * @param anchors Immutable bases with matching geometry.
 * @param blocks Persistent compressed outputs/inputs.
 * @param output Persistent reconstruction workspace.
 * @param iterations Number of complete passes over the selected heads.
 * @param workers Physical workers with disjoint contiguous head ranges.
 */
template <int D, ISA Isa, Operation Op>
__attribute__((noinline)) void kernelLoop(
    const std::vector<float> &source, const std::vector<float> &anchors,
    std::vector<AttentionKeyQ8Block<D>> &blocks, std::vector<float> &output, int iterations, int workers)
{
    // The harness creates one team, not one parallel region per head or pass.
    // Production callers own their own workshare; the primitive is deliberately
    // independent of OpenMP so it works inside an existing cache-append team.
#pragma omp parallel num_threads(workers) if(workers > 1)
    {
        const auto worker = static_cast<std::size_t>(omp_get_thread_num());
        const auto team = static_cast<std::size_t>(omp_get_num_threads());
        const auto begin = blocks.size() * worker / team;
        const auto end = blocks.size() * (worker + 1) / team;
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            for (std::size_t head = begin; head < end; ++head)
            {
                const auto anchor = std::span<const float, D>(anchors.data() + head * D, D);
                if constexpr (Op == Operation::Encode)
                    cpu::attention_key_q8::quantize<D>(
                        std::span<const float, D>(source.data() + head * D, D), anchor, blocks[head], Isa);
                else
                    cpu::attention_key_q8::dequantize<D>(
                        blocks[head], anchor, std::span<float, D>(output.data() + head * D, D), Isa);
            }
            // Keep repeated writes observable without adding timed checksum work.
            asm volatile("" ::: "memory");
        }
    }
}

/** @brief Authenticate exact bytes, warm the selected kernel, then time only it. */
template <int D, ISA Isa, Operation Op>
void measure(int heads, int iterations, int workers)
{
    const auto source = test::makeAttentionKeyQ8DeviceInput<D>(heads);
    std::vector<float> anchors(source.size()), residual(source.size()), output(source.size());
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        anchors[index] = source[(index % D)] * 0.875f;
        residual[index] = source[index] - anchors[index];
    }
    const auto expected = test::makeAttentionKeyQ8ReferenceBlocks<D>(residual);
    auto decoded = test::makeAttentionKeyQ8ReferenceDecoded<D>(expected);
    for (std::size_t index = 0; index < decoded.size(); ++index) decoded[index] += anchors[index];
    auto blocks = expected;
    kernelLoop<D, Isa, Op>(source, anchors, blocks, output, 1, workers);
    if constexpr (Op == Operation::Encode)
    {
        if (std::memcmp(blocks.data(), expected.data(), blocks.size() * sizeof(blocks[0])) != 0)
            throw std::runtime_error("AQ8 encode bytes disagree with scalar authority");
    }
    else if (std::memcmp(output.data(), decoded.data(), output.size() * sizeof(float)) != 0)
        throw std::runtime_error("AQ8 decode bytes disagree with scalar authority");
    kernelLoop<D, Isa, Op>(source, anchors, blocks, output, 256, workers);
    std::array<double, 7> elapsed{};
    for (double &sample : elapsed)
    {
        const auto start = std::chrono::steady_clock::now();
        kernelLoop<D, Isa, Op>(source, anchors, blocks, output, iterations, workers);
        sample = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(elapsed.begin(), elapsed.end());
    const double ns_per_head = elapsed[elapsed.size() / 2] * 1e9 / iterations / heads;
    const double logical_gib_s = (2.0 * D * sizeof(float) + sizeof(blocks[0])) / ns_per_head * 1e9 / (1024.0 * 1024.0 * 1024.0);
    std::cout << "codegen_isa,runtime_isa,operation,head_dim,heads,iterations,samples,workers,ns_per_head,logical_gib_s,byte_exact\n"
#if LLAMINAR_COMPILED_WITH_AVX512
              << "AVX512,"
#else
              << "AVX2,"
#endif
              << (Isa == ISA::AVX512 ? "AVX512" : "AVX2") << ','
              << (Op == Operation::Encode ? "encode" : "decode") << ','
              << D << ',' << heads << ',' << iterations << ',' << elapsed.size() << ',' << workers << ','
              << ns_per_head << ',' << logical_gib_s << ",true\n";
}

/** @brief Dispatch once outside timing, preserving a single-operation process. */
template <int D>
void dispatch(ISA isa, Operation operation, int heads, int iterations, int workers)
{
#if defined(__AVX512F__)
    if (isa == ISA::AVX512)
    {
        if (operation == Operation::Encode) measure<D, ISA::AVX512, Operation::Encode>(heads, iterations, workers);
        else measure<D, ISA::AVX512, Operation::Decode>(heads, iterations, workers);
        return;
    }
#endif
    if (operation == Operation::Encode) measure<D, ISA::AVX2, Operation::Encode>(heads, iterations, workers);
    else measure<D, ISA::AVX2, Operation::Decode>(heads, iterations, workers);
}
} // namespace

/** @brief Fail closed on incomplete or unsupported benchmark configuration. */
int main(int argc, char **argv)
{
    try
    {
        ISA isa = ISA::Automatic;
        Operation operation = Operation::Encode;
        int dimensions = 64, heads = 128, iterations = 1000, workers = 1;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view option(argv[index]);
            if (++index >= argc) throw std::invalid_argument("missing option value");
            const std::string_view value(argv[index]);
            if (option == "--head-dim") dimensions = positive(value);
            else if (option == "--heads") heads = positive(value);
            else if (option == "--iterations") iterations = positive(value);
            else if (option == "--workers") workers = positive(value);
            else if (option == "--isa")
            {
                if (value == "avx2") isa = ISA::AVX2;
                else if (value == "avx512") isa = ISA::AVX512;
                else if (value != "auto") throw std::invalid_argument("unsupported ISA");
            }
            else if (option == "--operation")
            {
                if (value == "encode") operation = Operation::Encode;
                else if (value == "decode") operation = Operation::Decode;
                else throw std::invalid_argument("unsupported operation");
            }
            else throw std::invalid_argument("unknown benchmark option");
        }
        // A reduced OpenMP team would make the reported worker count false.
        // Authenticate it before fixture construction and before any timing.
        omp_set_dynamic(0);
        int admitted_workers = 0;
#pragma omp parallel num_threads(workers) if(workers > 1)
        {
#pragma omp single
            admitted_workers = omp_get_num_threads();
        }
        if (admitted_workers != workers)
            throw std::runtime_error("OpenMP did not admit the requested benchmark worker count");
        isa = cpu::attention_key_q8::resolveISA(isa);
        switch (dimensions)
        {
        case 64: dispatch<64>(isa, operation, heads, iterations, workers); break;
        case 128: dispatch<128>(isa, operation, heads, iterations, workers); break;
        case 256: dispatch<256>(isa, operation, heads, iterations, workers); break;
        default: throw std::invalid_argument("AQ8 head dimension must be 64, 128 or 256");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CPU AQ8 benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
