/**
 * @file Perf__CPUNativeNEONPanelTournament.cpp
 * @brief One-candidate-per-launch timing of native NEON decode panel widths.
 *
 * Times CPUNativeNEONProjectionBatch::execute<Instruction, Strips>() — the
 * production batch path with an explicit panel width — on the persistent
 * OpenMP team, for production-sized shapes. Each launch measures exactly one
 * Strips candidate (LLAMINAR_PANEL_STRIPS) so profiler evidence and timings
 * stay attributable; every timed output is compared byte for byte with the
 * Strips=1 strip program before its sample is accepted. Microbenchmark wins
 * are candidates only: a width ships only after engine E2E confirmation.
 *
 * Environment (all optional):
 * - LLAMINAR_PANEL_STRIPS: 1, 2 or 4 (default 1); panels tile a 64-column chunk
 * - LLAMINAR_PANEL_ISA: asimd | dotprod | i8mm (default dotprod when compiled)
 * - LLAMINAR_PANEL_FORMATS: CSV of source labels (default Q4_K,Q6_K,IQ3_S,IQ4_XS,Q8_0)
 * - LLAMINAR_PANEL_SHAPES: CSV of NxK (default 2048x4096,17408x5120,512x2048)
 * - LLAMINAR_PANEL_M: CSV of rows (default 1,2,4)
 * - LLAMINAR_PANEL_WARMUP / LLAMINAR_PANEL_ITERS (default 5 / 31)
 *
 * Output: one CSV line per cell:
 * format,n,k,m,threads,isa,strips,median_us,p95_us,min_us,iters
 */
#include "kernels/cpu/gemm/CPUNativeNEONProjectionBatch.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::cpu::native_quantised;
using Dot = multi_scale::NEONByteDot;

/** @return Environment value or @p fallback. */
std::string env(const char *name, const char *fallback)
{
    const char *raw = std::getenv(name);
    return raw && *raw ? raw : fallback;
}

/** @return CSV fields of @p text. */
std::vector<std::string> fields(const std::string &text)
{
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

/** @return Registered source metadata for a label. */
const NativeQuantisedFormatInfo &format(const std::string &label)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        if (label == source.quant_type) return *source.metadata;
    throw std::invalid_argument("Unknown panel tournament format " + label);
}

/** @brief Deterministic Q8 activation rows with exact sums. */
std::vector<Q8_1Block> activation(int rows, int blocks)
{
    std::vector<Q8_1Block> bank(std::size_t(rows) * blocks);
    for (std::size_t index = 0; index < bank.size(); ++index)
    {
        auto &a = bank[index];
        a.d = static_cast<std::uint16_t>(0x2c00u + index % 512u);
        int sum = 0;
        for (int k = 0; k < 32; ++k)
        {
            a.qs[k] = static_cast<std::int8_t>((int(index) * 29 + k * 53) % 255 - 127);
            sum += a.qs[k];
        }
        a.sum_qs = static_cast<std::int16_t>(sum);
    }
    return bank;
}

/** @brief Run one batch execution with the selected family and width. */
template <Dot Instruction, int Strips> void run(const CPUNativeNEONProjectionBatch &batch)
{ batch.template execute<Instruction, Strips>(); }

/** @brief Dispatch a runtime width to its instantiation. */
template <Dot Instruction> void runStrips(const CPUNativeNEONProjectionBatch &batch, int strips)
{
    switch (strips)
    {
    case 1: return run<Instruction, 1>(batch);
    case 2: return run<Instruction, 2>(batch);
    case 4: return run<Instruction, 4>(batch);
    }
    throw std::invalid_argument("LLAMINAR_PANEL_STRIPS must be 1, 2 or 4");
}

/** @brief Dispatch a runtime family name to its compiled leaf. */
void execute(const CPUNativeNEONProjectionBatch &batch, const std::string &isa, int strips)
{
    if (isa == "asimd") return runStrips<Dot::ASIMD>(batch, strips);
#if defined(__ARM_FEATURE_DOTPROD)
    if (isa == "dotprod") return runStrips<Dot::DotProd>(batch, strips);
#endif
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
    if (isa == "i8mm") return runStrips<Dot::I8MM>(batch, strips);
#endif
    throw std::invalid_argument("LLAMINAR_PANEL_ISA names a family this binary does not contain: " + isa);
}
} // namespace

int main()
{
    const int strips = std::stoi(env("LLAMINAR_PANEL_STRIPS", "1"));
#if defined(__ARM_FEATURE_DOTPROD)
    const std::string isa = env("LLAMINAR_PANEL_ISA", "dotprod");
#else
    const std::string isa = env("LLAMINAR_PANEL_ISA", "asimd");
#endif
    const int warmup = std::stoi(env("LLAMINAR_PANEL_WARMUP", "5"));
    const int iters = std::stoi(env("LLAMINAR_PANEL_ITERS", "31"));
    const int threads = omp_get_max_threads();
    for (const auto &label : fields(env("LLAMINAR_PANEL_FORMATS", "Q4_K,Q6_K,IQ3_S,IQ4_XS,Q8_0")))
        for (const auto &shape : fields(env("LLAMINAR_PANEL_SHAPES", "2048x4096,17408x5120,512x2048")))
        {
            const auto x = shape.find('x');
            const int n = std::stoi(shape.substr(0, x)), k = std::stoi(shape.substr(x + 1));
            test::NativeQuantisedPreparedByteFixture fixture(format(label), n, k, std::size_t(1) << 31);
            fixture.descriptor.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
            fixture.seedPayloadAndMetadata();
            const auto weights = fixture.view();
            for (const auto &m_text : fields(env("LLAMINAR_PANEL_M", "1,2,4")))
            {
                const int m = std::stoi(m_text);
                const auto input = activation(m, weights.blocks_per_row);
                std::vector<float> output(std::size_t(m) * n), reference(output.size());
                const auto bind = [&](std::vector<float> &target) {
                    return CPUNativeNEONProjectionBatch::bind(weights, input, target, m, n, NEONProjectionRowTile::Four);
                };
                execute(bind(reference), isa, 1);
                const auto batch = bind(output);
                std::vector<double> samples;
                for (int iteration = 0; iteration < warmup + iters; ++iteration)
                {
                    const auto start = std::chrono::steady_clock::now();
                    execute(batch, isa, strips);
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    if (std::memcmp(output.data(), reference.data(), output.size() * sizeof(float)) != 0)
                    {
                        std::fprintf(stderr, "panel bytes differ from the strip program: %s %s m=%d strips=%d\n",
                            label.c_str(), shape.c_str(), m, strips);
                        return 2;
                    }
                    if (iteration >= warmup) samples.push_back(us);
                }
                std::sort(samples.begin(), samples.end());
                const auto at = [&](double q) { return samples[std::min(samples.size() - 1, std::size_t(q * samples.size()))]; };
                std::printf("%s,%d,%d,%d,%d,%s,%d,%.2f,%.2f,%.2f,%d\n", label.c_str(), n, k, m, threads, isa.c_str(),
                    strips, at(0.5), at(0.95), samples.front(), iters);
                std::fflush(stdout);
            }
        }
    return 0;
}
