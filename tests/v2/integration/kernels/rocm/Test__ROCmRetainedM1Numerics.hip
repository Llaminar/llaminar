/**
 * @file Test__ROCmRetainedM1Numerics.hip
 * @brief Retained MI50 half-scale, activation and packed router arithmetic replay.
 */
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/rocm/gemm/retained_hc_ffn/CanonicalFFNQ8.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1QuietMagnitude.h"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2::test {
namespace {
namespace router = rocm::router_native;
namespace q8 = device_q8_activation_contract;
namespace hc = hc_down_publication;
using qfn_castagna_native_impl::CanonicalFFNQ8Block;
using Word = std::uint32_t;

/** @brief Fail replay immediately on a native API failure. */
void checked(hipError_t status)
{
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

/** @brief Select one actual visible gfx906 participant before any fixture allocation. */
int nativeMI50()
{
    initializeROCmHardwareMetadata();
    const auto &inventory = rocmHardwareMetadata();
    for (int ordinal = 0; ordinal < static_cast<int>(inventory.size()); ++ordinal)
        if (inventory.at(ordinal).isa_target.starts_with("gfx906"))
            return ordinal;
    return -1;
}

/** @brief Own all test allocations and one exact non-null stream through replay retirement. */
class Replay {
public:
    explicit Replay(int ordinal)
    {
        checked(hipSetDevice(ordinal));
        std::array<char, 32> pci{};
        checked(hipDeviceGetPCIBusId(pci.data(), int(pci.size()), ordinal));
        std::printf("Retained MI50 replay device %s\n", pci.data());
        checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    Replay(const Replay &) = delete;
    Replay &operator=(const Replay &) = delete;

    ~Replay()
    {
        if (recording && hipStreamEndCapture(stream, &graph_) != hipSuccess)
            std::terminate();
        if (hipStreamSynchronize(stream) != hipSuccess)
            std::terminate();
        if ((exec_ && hipGraphExecDestroy(exec_) != hipSuccess) ||
            (graph_ && hipGraphDestroy(graph_) != hipSuccess))
            std::terminate();
        for (void *ptr : allocations_)
            if (hipFree(ptr) != hipSuccess)
                std::terminate();
        if (hipStreamDestroy(stream) != hipSuccess)
            std::terminate();
    }

    /** @param count Exact number of elements. @return Test-owned device storage. */
    template<class T> T *allocate(std::size_t count)
    {
        allocations_.push_back(nullptr);
        checked(hipMalloc(&allocations_.back(), count * sizeof(T)));
        return static_cast<T *>(allocations_.back());
    }

    /** @brief Record once and check each native kernel launch before instantiation. */
    template<class Record> void capture(Record record)
    {
        checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        recording = true;
        record();
        checked(hipGetLastError());
        const auto ended = hipStreamEndCapture(stream, &graph_);
        recording = false;
        checked(ended);
        checked(hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    }

    /** @brief Replay the same native executable. All setup and observations are outside capture. */
    void enqueue() { checked(hipGraphLaunch(exec_, stream)); }

    hipStream_t stream = nullptr;
private:
    std::vector<void *> allocations_;
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t exec_ = nullptr;
    bool recording = false;
};

struct ScaleCase { float precursor; Word expected; };
struct Q8Input { float values[32]; };
struct NormHC {
    float scale;
    signed char codes[32];
};
static_assert(sizeof(NormHC) == 36);
struct PairedHC {
    float activated[32];
    signed char primary[32], residual[32];
    float primary_scale, residual_scale;
};
static_assert(sizeof(PairedHC) == 200);

/** @brief Exercise the exact production half publishers and four-value-lane Q8 body. */
template<bool Native>
__global__ void replayScale(const ScaleCase *cases, const Q8Input *input,
    Word *width, NormHC *norm, CanonicalFFNQ8Block *rows, unsigned count)
{
    const unsigned quad = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = quad / 8;
    if (row >= count)
        return;
    const unsigned lane = quad % 8;
    if (lane == 0) {
        const float value = cases[row].precursor;
        width[row] = __float_as_uint(Native ? q8::publishPositiveFiniteHalfScale(value)
                                           : canonicalPreparedHalfValue(value));
    }
    float4 values{};
    __builtin_memcpy(&values, input[row].values + 4 * lane, sizeof(values));
    qfn_retained_m1_hc_ffn_device::current_hc_norm_q8_four<Native>(
        values, norm[row].codes, &norm[row].scale, lane);
    qfn_castagna_native_impl::canonicalFFNQuantizeFour<Native>(values, quad * 4, rows);
}

/** @brief Compare the real paired HC output with an unchanged 32-lane work assignment. */
template<bool Native>
__global__ void replayPairedHC(const Q8Input *input, PairedHC *out, unsigned count)
{
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    if (row >= count)
        return;
    struct Output {
        float *activated;
        signed char *primary, *residual;
        float *primaryScales, *residualScales;
    };
    auto &actual = out[row];
    using Scale = std::conditional_t<Native, hc::NativePositiveHalfScalePublication, hc::OriginalScalePublication>;
    hc::publish(input[row].values[lane],
        Output{actual.activated, actual.primary, actual.residual,
            &actual.primary_scale, &actual.residual_scale}, lane, Scale{});
}

/** @brief Unpack one nonnegative half exactly, independently of the device converter. */
float halfValue(unsigned word)
{
    const auto exponent = word >> 10, mantissa = word & 1023u;
    return exponent ? std::ldexp(float(1024 + mantissa), int(exponent) - 25)
                    : std::ldexp(float(mantissa), -24);
}

/** @brief Exhaust the positive finite half domain and its binary32 rounding boundaries. */
std::vector<ScaleCase> scaleCases()
{
    constexpr unsigned last = 0x7bff;
    constexpr float inf = std::numeric_limits<float>::infinity();
    std::vector<ScaleCase> cases;
    const auto add = [&](float value, unsigned expected) {
        cases.push_back({std::clamp(value, 0x1p-24f, 65504.0f),
            std::bit_cast<Word>(halfValue(expected))});
    };
    for (unsigned word = 1; word <= last; ++word) {
        const float exact = halfValue(word);
        add(std::nextafter(exact, 0.0f), word);
        add(exact, word);
        add(std::nextafter(exact, inf), word);
        if (word < last) {
            const float midpoint = (exact + halfValue(word + 1)) * 0.5f;
            add(std::nextafter(midpoint, 0.0f), word);
            add(midpoint, (word & 1u) ? word + 1 : word);
            add(std::nextafter(midpoint, inf), word + 1);
        }
    }
    for (float edge : {0.0f, 0x1p-25f, 65520.0f}) {
        const unsigned expected = edge < 1.0f ? 1 : last;
        add(std::nextafter(edge, -inf), expected);
        add(edge, expected);
        add(std::nextafter(edge, inf), expected);
    }
    return cases;
}

/** @brief Original signed quantization boundaries and exceptional rows on every lane. */
std::vector<Q8Input> activationRows(std::vector<ScaleCase> &cases, unsigned &finite)
{
    std::vector<Q8Input> rows(cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const float scale = cases[i].precursor;
        auto &values = rows[i].values;
        values[0] = 127.0f * scale;
        values[1] = -values[0];
        for (unsigned j = 2; j < 32; ++j) {
            const float midpoint = (float(int((i + j * 17) % 251) - 125) + 0.5f) * scale;
            constexpr float inf = std::numeric_limits<float>::infinity();
            values[j] = j % 3 == 0 ? std::nextafter(midpoint, -inf)
                      : j % 3 == 1 ? midpoint : std::nextafter(midpoint, inf);
        }
    }
    const ScaleCase floor{0x1p-24f, std::bit_cast<Word>(0x1p-24f)};
    rows.push_back({});
    cases.push_back(floor);
    for (Word word : {0x80000000u, 0x00000001u, 0x7f7fffffu}) {
        Q8Input row{};
        row.values[0] = std::bit_cast<float>(word);
        row.values[13] = std::bit_cast<float>(word ^ 0x80000000u);
        rows.push_back(row);
        cases.push_back(floor);
    }
    finite = static_cast<unsigned>(rows.size());
    const auto append = [&](const Q8Input &row) {
        rows.push_back(row);
        cases.push_back(floor);
    };
    const auto filled = [](Word word) {
        Q8Input row{};
        std::fill(std::begin(row.values), std::end(row.values), std::bit_cast<float>(word));
        return row;
    };
    constexpr Word signaling = 0x7f800001u;
    auto sole_zero = filled(signaling);
    sole_zero.values[0] = -0.0f;
    append(sole_zero);

    // The prior reassociated tree lost 96 known signaling-NaN rows.
    // Keep the exact surrounding lane inputs, including sole zero/numeric roots.
    constexpr Word exceptional[]{0x7f800000u, 0xff800000u, 0x7fc00001u,
        0xffc12345u, signaling, 0xff800001u};
    for (Word word : exceptional) {
        const float exceptional_value = std::bit_cast<float>(word);
        const auto all = filled(word);
        append(all);
        for (unsigned lane = 0; lane < 32; ++lane) {
            Q8Input single{};
            single.values[lane] = exceptional_value;
            append(single);
            single.values[(lane + 1) % 32] = 1.0f;
            append(single);
            single.values[(lane + 4) % 32] = -0x1p-24f;
            append(single);
            auto mixed = all;
            mixed.values[lane] = -0.0f;
            append(mixed);
            mixed.values[lane] = 1.0f;
            append(mixed);
        }
    }

    // Sweep each ordered pair of distinct positions over lane and component
    // boundaries. Exercise opposite NaN signs/payloads, denormals and infinities.
    constexpr Word numeric[]{0x00000000u, 0x80000000u, 0x00000001u, 0x807fffffu,
        0xbf800000u, 0x7f7fffffu, 0x7f800000u, 0xff800000u};
    constexpr Word unordered[]{signaling, 0xff800001u, 0x7fa1aa11u, 0x7fc31234u, 0xffcaff00u};
    for (Word background : unordered) {
        const auto all = filled(background);
        for (unsigned first = 0; first < 32; ++first)
            for (unsigned second = 0; second < 32; ++second) {
                if (first == second)
                    continue;
                const auto other = first + second;
                for (Word value : numeric) {
                    auto row = all;
                    row.values[first] = std::bit_cast<float>(value);
                    row.values[second] = std::bit_cast<float>(unordered[other % std::size(unordered)]);
                    append(row);
                    row.values[second] = std::bit_cast<float>(numeric[other % std::size(numeric)]);
                    append(row);
                }
            }
    }
    return rows;
}

constexpr unsigned Experts = 512, MaxTopK = 16, KeyWords = 13;
struct RouteInput {
    float values[Experts];
    int experts, top_k;
    bool normalize, direct;
};
struct RouteOutput {
    float probabilities[Experts], weights[MaxTopK];
    int ids[MaxTopK];
    Word guard[3];
};

/** @brief Run the actual original or packed queue, with or without the original softmax. */
template<bool Packed>
__global__ void replayRoutes(const RouteInput *recorded, RouteOutput *out)
{
    const auto &input = recorded[blockIdx.x];
    auto &output = out[blockIdx.x];
    __shared__ router::RouterSelectionShared scratch;
    constexpr auto transfer = router::TopKTransferPolicy::RowNativeRoot;
    if (input.direct) {
        router::moe_select_topk_probabilities_wave64_register_queue<8, false, transfer, true, Packed>(
            input.values, input.experts, input.top_k, scratch.selected_ids, scratch.selected_weights);
        __syncthreads();
    } else {
        router::selectProbabilities<false, 256, false, rocm::WorkgroupJoinMemory::LDSOnly,
            transfer, router::SelectorIntermediateStorage::ThreadPair, true, Packed>(
                input.values, nullptr, input.experts, input.top_k, input.normalize, scratch);
        if (threadIdx.x < input.experts)
            output.probabilities[threadIdx.x] = scratch.expert_probs[threadIdx.x];
    }
    if (threadIdx.x < input.top_k) {
        output.ids[threadIdx.x] = scratch.selected_ids[threadIdx.x];
        output.weights[threadIdx.x] = scratch.selected_weights[threadIdx.x];
    }
}

/** @brief Test monotone packed float and ID ordering against the exact original comparison. */
__global__ void replayRouteKeys(Word *failures)
{
    constexpr Word words[KeyWords]{0xff800000u, 0xbf800000u, 0x80000001u,
        0x80000000u, 0x00000000u, 0x00000001u, 0x33800000u, 0x3f000000u,
        0x3f000001u, 0x3f800000u, 0x7f800000u, 0x7fc12345u, 0xff800001u};
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned ids = Experts + 1;
    if (i >= KeyWords * KeyWords * ids)
        return;
    const float a = __uint_as_float(words[i / ids % KeyWords]);
    const float b = __uint_as_float(words[i / (ids * KeyWords)]);
    const unsigned id = i % ids, next = (id + 1) % ids;
    const auto key = router::routeKey(a, id);
    const auto compare = [&](unsigned other_id) {
        return router::keyBetter(key, router::routeKey(b, other_id)) ==
            router::moe_topk_pair_better(a, int(id), b, int(other_id));
    };
    if (!compare(id) || !compare(next) || !compare(Experts - id))
        atomicAdd(failures, 1u);
}

/** @brief Capture route ties, cutoff neighbours and unordered inputs across all prefix widths. */
std::vector<RouteInput> routeCases()
{
    std::vector<RouteInput> cases;
    constexpr Word exceptional[]{0x7f800000, 0xff800000, 0x7fc12345, 0xff800001};
    for (int experts : {257, 383, 511, 512})
        for (int top_k = 1; top_k <= int(MaxTopK); ++top_k)
            for (bool direct : {false, true})
                for (bool normalize : {false, true})
                    for (unsigned shape = 0; shape < 8; ++shape) {
                        RouteInput row{.experts=experts, .top_k=top_k, .normalize=normalize, .direct=direct};
                        for (int i = 0; i < experts; ++i) {
                            const float value = float((i * 47 + 11) % 503 - 251) * 0x1p-6f;
                            row.values[i] = shape == 0 ? (i % 2 ? -0.0f : 0.0f) :
                                shape == 1 ? std::nextafter(0.5f, i % 3 ? 0.0f : 1.0f) : value;
                            if (shape >= 3 && (i == 0 || i == 64 || (shape == 7 && i % 3)))
                                row.values[i] = std::bit_cast<float>(exceptional[(shape - 3) % 4]);
                        }
                        cases.push_back(row);
                    }
    return cases;
}
} // namespace

/** @test Bounded positive native half storage preserves exact Q8 scale and code bytes. */
TEST(ROCmRetainedM1Numerics, CapturedPositiveHalfScaleMatchesCanonical)
{
    auto lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
    const int ordinal = nativeMI50();
    if (ordinal < 0)
        GTEST_SKIP() << "UNQUALIFIED: no visible gfx906 participant";
    Replay replay(ordinal);
    auto cases = scaleCases();
    unsigned finite = 0;
    auto rows = activationRows(cases, finite);
    const unsigned n = static_cast<unsigned>(rows.size());
    // Paired HC's original nonfinite inputs are fatal. Exercise its complete
    // finite prefix here; the ordinary FFN helper also handles exceptional rows.
    auto *d_input = replay.allocate<Q8Input>(n);
    auto *d_cases = replay.allocate<ScaleCase>(n);
    std::array<Word *, 2> widths{replay.allocate<Word>(n), replay.allocate<Word>(n)};
    std::array<NormHC *, 2> norms{replay.allocate<NormHC>(n), replay.allocate<NormHC>(n)};
    std::array<CanonicalFFNQ8Block *, 2> blocks{
        replay.allocate<CanonicalFFNQ8Block>(n), replay.allocate<CanonicalFFNQ8Block>(n)};
    std::array<PairedHC *, 2> pairs{replay.allocate<PairedHC>(finite), replay.allocate<PairedHC>(finite)};
    replay.capture([&] {
        replayScale<false><<<(n + 31) / 32, 256, 0, replay.stream>>>(d_cases, d_input, widths[0], norms[0], blocks[0], n);
        replayScale<true><<<(n + 31) / 32, 256, 0, replay.stream>>>(d_cases, d_input, widths[1], norms[1], blocks[1], n);
        replayPairedHC<false><<<finite, 32, 0, replay.stream>>>(d_input, pairs[0], finite);
        replayPairedHC<true><<<finite, 32, 0, replay.stream>>>(d_input, pairs[1], finite);
    });
    std::array<std::vector<Word>, 2> actual_widths{std::vector<Word>(n), std::vector<Word>(n)};
    std::array<std::vector<NormHC>, 2> actual_norms{std::vector<NormHC>(n), std::vector<NormHC>(n)};
    std::array<std::vector<CanonicalFFNQ8Block>, 2> actual_blocks{
        std::vector<CanonicalFFNQ8Block>(n), std::vector<CanonicalFFNQ8Block>(n)};
    std::array<std::vector<PairedHC>, 2> actual_pairs{
        std::vector<PairedHC>(finite), std::vector<PairedHC>(finite)};
    checked(hipMemcpyAsync(d_cases, cases.data(), cases.size() * sizeof(ScaleCase), hipMemcpyHostToDevice, replay.stream));
    for (unsigned pass = 0; pass < 3; ++pass) {
        if (pass) {
            std::reverse(rows.begin(), rows.begin() + finite);
            std::reverse(rows.begin() + finite, rows.end());
        }
        checked(hipMemcpyAsync(d_input, rows.data(), rows.size() * sizeof(Q8Input), hipMemcpyHostToDevice, replay.stream));
        for (unsigned arm = 0; arm < 2; ++arm) {
            const int dirty = 0x42 + int(arm) * 23 + int(pass);
            checked(hipMemsetAsync(widths[arm], dirty, n * sizeof(Word), replay.stream));
            checked(hipMemsetAsync(norms[arm], dirty, n * sizeof(NormHC), replay.stream));
            checked(hipMemsetAsync(blocks[arm], dirty, n * sizeof(CanonicalFFNQ8Block), replay.stream));
            checked(hipMemsetAsync(pairs[arm], dirty, finite * sizeof(PairedHC), replay.stream));
        }
        replay.enqueue();
        for (unsigned arm = 0; arm < 2; ++arm) {
            checked(hipMemcpyAsync(actual_widths[arm].data(), widths[arm], n * sizeof(Word), hipMemcpyDeviceToHost, replay.stream));
            checked(hipMemcpyAsync(actual_norms[arm].data(), norms[arm], n * sizeof(NormHC), hipMemcpyDeviceToHost, replay.stream));
            checked(hipMemcpyAsync(actual_blocks[arm].data(), blocks[arm], n * sizeof(CanonicalFFNQ8Block), hipMemcpyDeviceToHost, replay.stream));
            checked(hipMemcpyAsync(actual_pairs[arm].data(), pairs[arm], finite * sizeof(PairedHC), hipMemcpyDeviceToHost, replay.stream));
        }
        checked(hipStreamSynchronize(replay.stream));
        const unsigned sole_zero = pass % 2 ? n - 1 : finite;
        ASSERT_EQ(std::bit_cast<Word>(actual_norms[1][sole_zero].scale), std::bit_cast<Word>(1.0f));
        ASSERT_EQ(std::bit_cast<Word>(actual_blocks[1][sole_zero].scale), std::bit_cast<Word>(1.0f));
        for (unsigned i = 0; i < n; ++i) {
            ASSERT_EQ(actual_widths[0][i], cases[i].expected) << i;
            ASSERT_EQ(actual_widths[1][i], cases[i].expected) << i;
            ASSERT_EQ(std::memcmp(&actual_norms[0][i], &actual_norms[1][i], sizeof(NormHC)), 0)
                << "HC norm row " << i;
            const auto &block = actual_blocks[1][i];
            ASSERT_EQ(std::memcmp(&actual_blocks[0][i], &block, sizeof(block)), 0) << i;
            int sum = 0;
            for (int code : block.qs)
                for (unsigned j = 0; j < 4; ++j)
                    sum += std::bit_cast<std::int8_t>(std::uint8_t(Word(code) >> (8 * j)));
            ASSERT_EQ(block.sum, sum) << i;
            if (i < finite)
                ASSERT_EQ(std::memcmp(&actual_pairs[0][i], &actual_pairs[1][i], sizeof(PairedHC)), 0) << i;
        }
    }
}

/** @test Exact packed keys and the real capped queue preserve ordered IDs and probability bits. */
TEST(ROCmRetainedM1Numerics, CapturedPackedQueueMatchesOriginal)
{
    auto lifetime = GPUDeviceProcessLifetime::acquireNativeContext();
    const int ordinal = nativeMI50();
    if (ordinal < 0)
        GTEST_SKIP() << "UNQUALIFIED: no visible gfx906 participant";
    Replay replay(ordinal);
    auto cases = routeCases();
    auto *input = replay.allocate<RouteInput>(cases.size());
    std::array<RouteOutput *, 2> output{
        replay.allocate<RouteOutput>(cases.size()), replay.allocate<RouteOutput>(cases.size())};
    auto *failures = replay.allocate<Word>(1);
    replay.capture([&] {
        replayRouteKeys<<<(KeyWords * KeyWords * (Experts + 1) + 255) / 256, 256, 0, replay.stream>>>(failures);
        replayRoutes<false><<<cases.size(), 1024, 0, replay.stream>>>(input, output[0]);
        replayRoutes<true><<<cases.size(), 1024, 0, replay.stream>>>(input, output[1]);
    });
    std::array<std::vector<RouteOutput>, 2> actual{
        std::vector<RouteOutput>(cases.size()), std::vector<RouteOutput>(cases.size())};
    const auto in_bytes = cases.size() * sizeof(RouteInput), out_bytes = cases.size() * sizeof(RouteOutput);
    for (unsigned pass = 0; pass < 3; ++pass) {
        if (pass)
            std::reverse(cases.begin(), cases.end());
        checked(hipMemcpyAsync(input, cases.data(), in_bytes, hipMemcpyHostToDevice, replay.stream));
        checked(hipMemsetAsync(failures, 0, sizeof(Word), replay.stream));
        const int dirty = 0x73 + int(pass);
        for (auto *arm : output)
            checked(hipMemsetAsync(arm, dirty, out_bytes, replay.stream));
        replay.enqueue();
        Word mismatches = 0;
        checked(hipMemcpyAsync(&mismatches, failures, sizeof(Word), hipMemcpyDeviceToHost, replay.stream));
        for (unsigned arm = 0; arm < 2; ++arm)
            checked(hipMemcpyAsync(actual[arm].data(), output[arm], out_bytes, hipMemcpyDeviceToHost, replay.stream));
        checked(hipStreamSynchronize(replay.stream));
        ASSERT_EQ(mismatches, 0u);
        const Word sentinel = Word(dirty) * 0x01010101u;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            ASSERT_EQ(std::memcmp(&actual[0][i], &actual[1][i], sizeof(RouteOutput)), 0) << i;
            for (Word guard : actual[1][i].guard)
                ASSERT_EQ(guard, sentinel) << i;
        }
    }
}
} // namespace llaminar2::test
