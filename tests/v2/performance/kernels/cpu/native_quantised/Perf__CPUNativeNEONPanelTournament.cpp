/**
 * @file Perf__CPUNativeNEONPanelTournament.cpp
 * @brief One-candidate-per-launch timing of native NEON decode panel widths.
 *
 * Times CPUNativeNEONProjectionBatch::execute<Instruction, Strips>() — the
 * production batch path with an explicit panel width — for production-sized
 * shapes. Each launch measures exactly one Strips candidate so profiler
 * evidence and timings stay attributable. Microbenchmark wins are candidates
 * only: a width ships only after engine E2E confirmation.
 *
 * Team: all iterations of a cell run inside one retained OpenMP parallel
 * region, so execute() workshares on that team exactly as it does inside the
 * engine's persistent team; no region is forked per call. The team width the
 * region actually observed is reported, not the requested capacity. The timed
 * interval is one team barrier, the workshare and its closing barrier.
 *
 * Candidates: a cell runs only when every row tile executes the requested
 * width. Compact multi-scale encodings (own column walk) and I8MM cells whose
 * row pairs run the SMMLA strip walk (M >= 2 with Strips > 1) are rejected
 * rather than mislabelled.
 *
 * Publication: the Strips = 1 reference runs twice over two distinct dirty
 * patterns and must agree byte for byte, which proves it wrote every live
 * element. Before every timed invocation (outside the timed interval) the
 * output is dirtied with a pattern that alternates between iterations, and
 * afterwards it must equal the reference, so an invocation that omits an
 * element or publishes nothing cannot pass on stale bytes. Guard words around
 * the output span, the weights and the activations are checked after each
 * cell. LLAMINAR_PANEL_SELF_TEST=1 runs negative controls of these checks and
 * of the input admission instead of timing.
 *
 * Environment (all optional; every value is admitted before any allocation):
 * - LLAMINAR_PANEL_STRIPS: 1, 2 or 4 (default 1); panels tile a 64-column chunk
 * - LLAMINAR_PANEL_ISA: asimd | dotprod | i8mm (default dotprod when compiled)
 * - LLAMINAR_PANEL_FORMATS: unique source labels (default Q4_K,Q6_K,IQ3_S,IQ4_XS,Q8_0)
 * - LLAMINAR_PANEL_SHAPES: unique NxK, 1 <= N, K <= 65536 (default 2048x4096,17408x5120,512x2048)
 * - LLAMINAR_PANEL_M: unique rows in 1..64 (default 1,2,4)
 * - LLAMINAR_PANEL_WARMUP: 0..10000 (default 5); LLAMINAR_PANEL_ITERS: 1..100000 (default 31)
 *
 * Output: one CSV line per cell:
 * format,n,k,m,threads,isa,strips,median_us,p95_us,min_us,iters
 * where threads is the observed width of the timing team.
 */
#include "kernels/cpu/gemm/CPUNativeNEONProjectionBatch.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

#include <omp.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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
    return raw ? raw : fallback;
}

/**
 * @brief Parse a complete decimal integer inside [low, high].
 * @throws std::invalid_argument For empty text, trailing characters or a value out of range.
 */
int integer(std::string_view text, int low, int high, const char *what)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size() || value < low || value > high)
        throw std::invalid_argument(std::string(what) + " must be an integer in [" + std::to_string(low) + ", " +
            std::to_string(high) + "]: '" + std::string(text) + "'");
    return value;
}

/**
 * @brief Split a nonempty CSV list of unique, nonempty fields.
 * @throws std::invalid_argument For an empty list, an empty field or a duplicate.
 */
std::vector<std::string> fields(const std::string &text, const char *what)
{
    std::vector<std::string> out;
    std::set<std::string> seen;
    std::size_t begin = 0;
    while (true)
    {
        const auto end = text.find(',', begin);
        const auto item = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty() || !seen.insert(item).second)
            throw std::invalid_argument(std::string(what) + " needs unique, nonempty comma-separated entries: '" +
                text + "'");
        out.push_back(item);
        if (end == std::string::npos) return out;
        begin = end + 1;
    }
}

/** @brief Admitted N x K geometry. */
struct Shape
{
    int n = 0, k = 0;
};

/** @brief Parse exactly one NxK field with both extents in [1, 65536]. */
Shape shape(const std::string &text)
{
    const auto x = text.find('x');
    if (x == std::string::npos || text.find('x', x + 1) != std::string::npos)
        throw std::invalid_argument("LLAMINAR_PANEL_SHAPES entries must be NxK: '" + text + "'");
    return {integer(std::string_view(text).substr(0, x), 1, 1 << 16, "shape N"),
            integer(std::string_view(text).substr(x + 1), 1, 1 << 16, "shape K")};
}

/** @return Registered source metadata for a label. */
const NativeQuantisedFormatInfo &format(const std::string &label)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        if (label == source.quant_type) return *source.metadata;
    throw std::invalid_argument("Unknown panel tournament format " + label);
}

/** @brief Complete admitted workload of one launch. */
struct Workload
{
    int strips = 1;
    std::string isa;
    std::vector<std::string> formats;
    std::vector<Shape> shapes;
    std::vector<int> rows;
    int warmup = 5, iters = 31;
};

/**
 * @brief Admit every selection before any allocation or timing.
 * @throws std::invalid_argument For malformed, empty, duplicate or out-of-range input,
 * unknown or compact formats, uncompiled families and candidates a tile would not run.
 */
Workload admit(const std::function<std::string(const char *, const char *)> &get)
{
    Workload w;
    w.strips = integer(get("LLAMINAR_PANEL_STRIPS", "1"), 1, 4, "LLAMINAR_PANEL_STRIPS");
    if (w.strips == 3)
        throw std::invalid_argument("LLAMINAR_PANEL_STRIPS must be 1, 2 or 4: a panel never crosses its chunk");
#if defined(__ARM_FEATURE_DOTPROD)
    w.isa = get("LLAMINAR_PANEL_ISA", "dotprod");
#else
    w.isa = get("LLAMINAR_PANEL_ISA", "asimd");
#endif
    bool compiled = w.isa == "asimd";
#if defined(__ARM_FEATURE_DOTPROD)
    compiled = compiled || w.isa == "dotprod";
#endif
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
    compiled = compiled || w.isa == "i8mm";
#endif
    if (!compiled)
        throw std::invalid_argument("LLAMINAR_PANEL_ISA names a family this binary does not contain: " + w.isa);
    w.formats = fields(get("LLAMINAR_PANEL_FORMATS", "Q4_K,Q6_K,IQ3_S,IQ4_XS,Q8_0"), "LLAMINAR_PANEL_FORMATS");
    for (const auto &label : w.formats)
        if (preparedEncodingForSource(format(label)) == CPUNativeQuantisedEncoding::CompactMultiScale)
            throw std::invalid_argument("Compact multi-scale format " + label +
                " keeps its own column walk; it has no panel candidate");
    for (const auto &text : fields(get("LLAMINAR_PANEL_SHAPES", "2048x4096,17408x5120,512x2048"), "LLAMINAR_PANEL_SHAPES"))
        w.shapes.push_back(shape(text));
    for (const auto &text : fields(get("LLAMINAR_PANEL_M", "1,2,4"), "LLAMINAR_PANEL_M"))
        w.rows.push_back(integer(text, 1, 64, "LLAMINAR_PANEL_M"));
    for (const int m : w.rows)
        if (w.isa == "i8mm" && m >= 2 && w.strips > 1)
            throw std::invalid_argument("I8MM row pairs run the SMMLA strip walk: M >= 2 has no Strips > 1 candidate");
    w.warmup = integer(get("LLAMINAR_PANEL_WARMUP", "5"), 0, 10000, "LLAMINAR_PANEL_WARMUP");
    w.iters = integer(get("LLAMINAR_PANEL_ITERS", "31"), 1, 100000, "LLAMINAR_PANEL_ITERS");
    return w;
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

/** @brief Dispatch an admitted width to its instantiation. */
template <Dot Instruction> void runStrips(const CPUNativeNEONProjectionBatch &batch, int strips)
{
    switch (strips)
    {
    case 1: return run<Instruction, 1>(batch);
    case 2: return run<Instruction, 2>(batch);
    case 4: return run<Instruction, 4>(batch);
    }
    std::abort(); // admit() accepts no other width.
}

/** @brief Dispatch an admitted family name to its compiled leaf. */
void execute(const CPUNativeNEONProjectionBatch &batch, const std::string &isa, int strips)
{
    if (isa == "asimd") return runStrips<Dot::ASIMD>(batch, strips);
#if defined(__ARM_FEATURE_DOTPROD)
    if (isa == "dotprod") return runStrips<Dot::DotProd>(batch, strips);
#endif
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
    if (isa == "i8mm") return runStrips<Dot::I8MM>(batch, strips);
#endif
    std::abort(); // admit() accepts no other family.
}

/**
 * @brief Output bank with guard words on both sides of the exact live span.
 * @details The batch binds only the inner span; the guards prove no write
 * escapes it. Dirty patterns are quiet-NaN payloads no kernel produces.
 */
class GuardedOutput
{
public:
    static constexpr std::size_t kGuard = 16;                  ///< Guard words per side.
    static constexpr std::uint32_t kGuardWord = 0x7fc0dead;    ///< Guard payload.
    static constexpr std::uint32_t kDirty[2] = {0x7fc0a11a, 0x7fc0b22b}; ///< Alternating dirt.

    explicit GuardedOutput(std::size_t live)
        : words_(live + 2 * kGuard, std::bit_cast<float>(kGuardWord)), live_(live) {}

    /** @return The exact live span a batch binds. */
    std::span<float> live() { return {words_.data() + kGuard, live_}; }
    /** @brief Overwrite every live word with dirty pattern @p which (0 or 1); copies keep NaN payloads. */
    void dirty(int which)
    { std::fill(words_.begin() + kGuard, words_.begin() + kGuard + live_, std::bit_cast<float>(kDirty[which])); }
    /** @return Whether every guard word is intact. */
    [[nodiscard]] bool guardsIntact() const
    {
        for (std::size_t i = 0; i < kGuard; ++i)
            if (bits(i) != kGuardWord || bits(kGuard + live_ + i) != kGuardWord) return false;
        return true;
    }
    /** @return Copy of the live words' bit patterns. */
    [[nodiscard]] std::vector<std::uint32_t> snapshot() const
    {
        std::vector<std::uint32_t> out(live_);
        std::memcpy(out.data(), words_.data() + kGuard, live_ * sizeof(float));
        return out;
    }
    /** @return Whether the live words equal @p reference byte for byte. */
    [[nodiscard]] bool equals(const std::vector<std::uint32_t> &reference) const
    {
        return reference.size() == live_ &&
            std::memcmp(reference.data(), words_.data() + kGuard, live_ * sizeof(float)) == 0;
    }
    /** @brief Overwrite one word by index from the live start, for negative controls only. */
    void poke(std::size_t index, std::uint32_t value) { words_[kGuard + index] = std::bit_cast<float>(value); }

private:
    [[nodiscard]] std::uint32_t bits(std::size_t index) const { return std::bit_cast<std::uint32_t>(words_[index]); }

    std::vector<float> words_;
    std::size_t live_;
};

/**
 * @brief Prove a reference wrote every live word: two runs over distinct dirt must agree.
 * @return The reference words, or empty when an element kept its dirt.
 */
std::vector<std::uint32_t> completeReference(GuardedOutput &output, const std::function<void()> &invoke)
{
    output.dirty(0);
    invoke();
    const auto first = output.snapshot();
    output.dirty(1);
    invoke();
    if (!output.equals(first) || !output.guardsIntact()) return {};
    return first;
}

/** @brief Timing summary of one cell. */
struct Cell
{
    std::vector<double> samples;
    int observed_threads = 0;
    bool published = true;
};

/**
 * @brief Time one cell on one retained team, checking publication after every invocation.
 * @param invoke One production execute(); called by every team member.
 */
Cell timeCell(GuardedOutput &output, const std::vector<std::uint32_t> &reference, int warmup, int iters,
    const std::function<void()> &invoke)
{
    Cell cell;
    cell.samples.reserve(std::size_t(iters));
    std::chrono::steady_clock::time_point start;
#pragma omp parallel
    {
#pragma omp single
        cell.observed_threads = omp_get_num_threads();
        for (int iteration = 0; iteration < warmup + iters; ++iteration)
        {
#pragma omp single
            output.dirty(iteration & 1); // Outside the timed interval; implicit barrier follows.
#pragma omp master
            start = std::chrono::steady_clock::now();
#pragma omp barrier
            invoke(); // Workshares on this retained team; its omp for closes with a barrier.
#pragma omp master
            {
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
                if (!output.equals(reference)) cell.published = false;
                if (iteration >= warmup) cell.samples.push_back(us);
            }
#pragma omp barrier
        }
    }
    return cell;
}

/** @brief Negative controls of input admission and publication checks; returns failures. */
int selfTest()
{
    int failures = 0;
    const auto expect = [&](bool ok, const char *what) {
        if (!ok) { std::fprintf(stderr, "self-test control failed: %s\n", what); ++failures; }
    };
    const auto rejects = [&](std::vector<std::pair<std::string, std::string>> values, const char *what) {
        const auto get = [&](const char *name, const char *fallback) -> std::string {
            for (const auto &[key, value] : values)
                if (key == name) return value;
            return fallback;
        };
        try { (void)admit(get); expect(false, what); }
        catch (const std::invalid_argument &) {}
    };
    rejects({{"LLAMINAR_PANEL_ITERS", "0"}}, "zero iterations");
    rejects({{"LLAMINAR_PANEL_ITERS", "2147483647"}}, "iterations beyond the bound");
    rejects({{"LLAMINAR_PANEL_WARMUP", "-1"}}, "negative warmup");
    rejects({{"LLAMINAR_PANEL_FORMATS", ","}}, "empty format list");
    rejects({{"LLAMINAR_PANEL_FORMATS", ""}}, "empty format value");
    rejects({{"LLAMINAR_PANEL_FORMATS", "Q8_0,Q8_0"}}, "duplicate format");
    rejects({{"LLAMINAR_PANEL_SHAPES", "128"}}, "shape without K");
    rejects({{"LLAMINAR_PANEL_SHAPES", "128x64junk"}}, "shape with trailing characters");
    rejects({{"LLAMINAR_PANEL_SHAPES", "0x64"}}, "zero N");
    rejects({{"LLAMINAR_PANEL_SHAPES", "64x64x64"}}, "three extents");
    rejects({{"LLAMINAR_PANEL_M", "0"}}, "zero rows");
    rejects({{"LLAMINAR_PANEL_M", "1,1"}}, "duplicate rows");
    rejects({{"LLAMINAR_PANEL_STRIPS", "3"}}, "three strips");
    rejects({{"LLAMINAR_PANEL_STRIPS", "2x"}}, "strips with trailing characters");
    rejects({{"LLAMINAR_PANEL_ISA", "sve"}}, "uncompiled family");
    bool compact_found = false;
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        if (!compact_found && preparedEncodingForSource(*source.metadata) == CPUNativeQuantisedEncoding::CompactMultiScale)
        {
            rejects({{"LLAMINAR_PANEL_FORMATS", std::string(source.quant_type)}}, "compact format");
            compact_found = true;
        }
    expect(compact_found, "a registered compact format exists for the control");
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
    rejects({{"LLAMINAR_PANEL_ISA", "i8mm"}, {"LLAMINAR_PANEL_STRIPS", "4"}, {"LLAMINAR_PANEL_M", "1,2"}},
        "I8MM row pairs with a forced panel");
#endif

    // Publication controls over 64 live words with a known complete writer.
    constexpr std::size_t live = 64;
    GuardedOutput output(live);
    const auto write_all = [&] { for (std::size_t i = 0; i < live; ++i) output.live()[i] = float(i); };
    const auto reference = completeReference(output, write_all);
    expect(reference.size() == live, "complete reference accepted");
    const auto skip_one = [&] { for (std::size_t i = 0; i < live; ++i) if (i != 5) output.live()[i] = float(i); };
    expect(completeReference(output, skip_one).empty(), "omitted reference element rejected");
    output.dirty(0);
    expect(!output.equals(reference), "omitted result rejected");
    write_all();
    expect(output.equals(reference), "stale bytes would pass without dirtying");
    output.dirty(1);
    expect(!output.equals(reference), "stale result rejected after dirtying");
    output.poke(live, 0); // One word past the live span.
    expect(!output.guardsIntact(), "escaped write rejected");
    return failures;
}
} // namespace

int main()
{
    try
    {
        if (env("LLAMINAR_PANEL_SELF_TEST", "0") == "1")
        {
            const int failures = selfTest();
            std::printf("panel tournament self-test: %d failed control(s)\n", failures);
            return failures == 0 ? 0 : 1;
        }
        const auto w = admit([](const char *name, const char *fallback) { return env(name, fallback); });
        for (const auto &label : w.formats)
            for (const auto &geometry : w.shapes)
            {
                test::NativeQuantisedPreparedByteFixture fixture(format(label), geometry.n, geometry.k,
                    std::size_t(1) << 31);
                fixture.descriptor.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
                fixture.seedPayloadAndMetadata();
                const auto weights = fixture.view();
                const std::vector<std::uint8_t> weights_before(weights.bytes().begin(), weights.bytes().end());
                for (const int m : w.rows)
                {
                    const auto input = activation(m, weights.blocks_per_row);
                    const auto input_before = input;
                    GuardedOutput output(std::size_t(m) * geometry.n);
                    const auto batch = CPUNativeNEONProjectionBatch::bind(weights, input, output.live(), m,
                        geometry.n, NEONProjectionRowTile::Four);
                    const auto reference = completeReference(output, [&] { execute(batch, w.isa, 1); });
                    if (reference.empty())
                        throw std::runtime_error("Strips=1 reference did not publish every live element");
                    const auto cell = timeCell(output, reference, w.warmup, w.iters,
                        [&] { execute(batch, w.isa, w.strips); });
                    if (!cell.published || !output.guardsIntact() ||
                        std::memcmp(input.data(), input_before.data(), input.size() * sizeof(Q8_1Block)) != 0 ||
                        !std::equal(weights_before.begin(), weights_before.end(), weights.bytes().begin()))
                        throw std::runtime_error("panel bytes differ from the strip program, a write escaped, or an "
                            "input changed: " + label + " m=" + std::to_string(m));
                    auto samples = cell.samples;
                    std::sort(samples.begin(), samples.end());
                    const auto at = [&](double q) {
                        return samples[std::min(samples.size() - 1, std::size_t(q * samples.size()))];
                    };
                    std::printf("%s,%d,%d,%d,%d,%s,%d,%.2f,%.2f,%.2f,%d\n", label.c_str(), geometry.n, geometry.k, m,
                        cell.observed_threads, w.isa.c_str(), w.strips, at(0.5), at(0.95), samples.front(), w.iters);
                    std::fflush(stdout);
                }
            }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "panel tournament: %s\n", error.what());
        return 2;
    }
}
