/**
 * @file Test__ROCmGroupedHCDownLongK.cpp
 * @brief Dirty retained native Q8 HC-down partial and result replay on gfx906.
 *
 * The production grouped admission and serial GEMV are the only numeric
 * implementations in the fixture. Model-free CI uses distinguishable N and
 * K axes. A private release replay can use the original GGUF block bytes.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "../../../utils/ROCmQFNReplayDevice.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void rocmGemv_native_quantised_set_decode_equivalent_m1_config(int);
extern "C" bool rocmGemv_native_quantised_fp32_with_policy(
    const int8_t *, const uint8_t *, const void *, const void *, const void *, float *,
    const float *, float *, int, int, uint8_t, uint8_t, int, void *, const float *);

namespace {
using llaminar2::DeviceRowRange;
constexpr int n = 320, k = 10240, groups = k / 32, partitions = 16, maxRows = 8, guard = 16;
constexpr int scratchPartitions = llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
constexpr uint8_t q8 = 19;
constexpr float poison = -190.5f;

void checked(hipError_t status)
{
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

/** @brief Own test buffers before graph capture and collect only after terminal events. */
template<class T>
struct Bank
{
    explicit Bank(std::size_t size) : size(size)
    {
        checked(hipMalloc(&data, size * sizeof(T)));
    }
    ~Bank() { (void)hipFree(data); }
    Bank(const Bank&) = delete;
    Bank& operator=(const Bank&) = delete;

    /** @brief Upload a fixture publication ordered on the exact replay stream. */
    void input(const std::vector<T>& host, hipStream_t stream)
    {
        if (host.size() != size) throw std::invalid_argument("Fixture extent changed");
        checked(hipMemcpyAsync(data, host.data(), size * sizeof(T), hipMemcpyHostToDevice, stream));
    }

    /** @brief Read an already completed graph result for byte comparison. */
    std::vector<T> output(hipStream_t stream)
    {
        std::vector<T> host(size);
        checked(hipMemcpyAsync(host.data(), data, size * sizeof(T), hipMemcpyDeviceToHost, stream));
        checked(hipStreamSynchronize(stream));
        return host;
    }

    T *data = nullptr;
    std::size_t size;
};

struct Stream
{
    Stream() { checked(hipStreamCreateWithFlags(&native, hipStreamNonBlocking)); }
    ~Stream() { (void)hipStreamDestroy(native); }
    hipStream_t native = nullptr;
};

struct Events
{
    Events() { checked(hipEventCreate(&start)); checked(hipEventCreate(&end)); }
    ~Events() { (void)hipEventDestroy(end); (void)hipEventDestroy(start); }
    hipEvent_t start = nullptr, end = nullptr;
};

/** @brief Scope the production thread-local serial arithmetic policy during native recording. */
struct DecodeEquivalent
{
    DecodeEquivalent() { rocmGemv_native_quantised_set_decode_equivalent_m1_config(1); }
    ~DecodeEquivalent() { rocmGemv_native_quantised_set_decode_equivalent_m1_config(0); }
};

/** @brief Retain native producer and reducer identities across dirty replays. */
struct Capture
{
    ~Capture()
    {
        if (exec) (void)hipGraphExecDestroy(exec);
        if (graph) (void)hipGraphDestroy(graph);
    }

    template<class Record>
    void record(hipStream_t stream, Record&& callback)
    {
        checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        const bool ok = callback();
        checked(hipStreamEndCapture(stream, &graph));
        if (!ok) throw std::runtime_error("Native grouped entry rejected fixture");
        checked(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }

    void submit(hipStream_t stream) const { checked(hipGraphLaunch(exec, stream)); }

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
};

/** @brief Check the exact producer grid and the original reducer native symbol privately. */
void expectGraph(const Capture& capture, hipStream_t stream, int m, bool optedIn)
{
    std::size_t count = 0;
    checked(hipGraphGetNodes(capture.graph, nullptr, &count));
    ASSERT_EQ(count, 2u);
    std::array<hipGraphNode_t, 2> nodes{};
    checked(hipGraphGetNodes(capture.graph, nodes.data(), &count));
    int producers = 0, reducers = 0;
    for (auto node : nodes)
    {
        hipKernelNodeParams launch{};
        checked(hipGraphKernelNodeGetParams(node, &launch));
        const auto *symbol = hipKernelNameRefByPtr(launch.func, stream);
        ASSERT_NE(symbol, nullptr);
        const std::string_view name(symbol);
        if (name.find("gemv_native_quantised_reduce_small_m_kernel_t") != name.npos)
        {
            EXPECT_EQ(launch.gridDim.x, 5u);
            EXPECT_EQ(launch.blockDim.x, 64u);
            ++reducers;
        }
        else
        {
            const auto leaf = optedIn ? "qfn_hc_down_long_k_contributions"
                                      : "gemv_native_quantised_small_m_kernel_t";
            EXPECT_NE(name.find(leaf), name.npos);
            EXPECT_EQ(launch.gridDim.x, optedIn ? 10u : 5u);
            EXPECT_EQ(launch.gridDim.y, 16u);
            EXPECT_EQ(launch.gridDim.z, unsigned((m + 3) / 4));
            EXPECT_EQ(launch.blockDim.x, optedIn ? 640u : 64u);
            ++producers;
        }
    }
    EXPECT_EQ(producers, 1);
    EXPECT_EQ(reducers, 1);
}

/** @brief Prepare original Q8_0 row-major blocks into existing physical K-block-major axes. */
void sourceWeights(std::vector<uint8_t>& payload, std::vector<uint16_t>& scales)
{
    const auto *source = std::getenv("LLAMINAR_HCDOWN_ORIGINAL_WINDOW");
    std::vector<uint8_t> raw;
    if (source)
    {
        raw.resize(n * groups * 34);
        std::ifstream file(source, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != std::streamsize(raw.size()))
            throw std::runtime_error("Original HC-down Q8_0 window disagrees with shape");
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(raw.data()), raw.size()))
            throw std::runtime_error("Unable to read original HC-down Q8_0 window");
    }
    for (int col = 0; col < n; ++col)
        for (int group = 0; group < groups; ++group)
        {
            const int native = group * n + col;
            if (source)
            {
                const int stored = (col * groups + group) * 34;
                std::memcpy(&scales[native], raw.data() + stored, 2);
                std::memcpy(payload.data() + native * 32, raw.data() + stored + 2, 32);
            }
            else
            {
                scales[native] = uint16_t(0x2000u + ((col * 379 + group * 17) & 2047));
                for (int lane = 0; lane < 32; ++lane)
                    payload[native * 32 + lane] = uint8_t((col * 71 + group * 19 + lane * 13) % 255 - 127);
            }
        }
}

/** @brief Assert all output, every split-K partial and the full inactive/sentinel extent. */
void matchReplay(const std::vector<float>& output, const std::vector<float>& partials,
    const std::vector<float>& reference, const std::vector<float>& referencePartials, int m, int live)
{
    ASSERT_EQ(0, std::memcmp(output.data(), reference.data(), live * n * sizeof(float)));
    for (int part = 0; part < partitions; ++part)
        for (int row = 0; row < live; ++row)
        {
            const int groupedOffset = (part * m + row) * n;
            const int serialOffset = (row * scratchPartitions + part) * n;
            ASSERT_EQ(0, std::memcmp(partials.data() + groupedOffset,
                referencePartials.data() + serialOffset, n * sizeof(float))) << "row=" << row << " part=" << part;
        }
    const auto clean = [](float value) { return std::bit_cast<uint32_t>(value) == std::bit_cast<uint32_t>(poison); };
    for (int part = 0; part < partitions; ++part)
        ASSERT_TRUE(std::all_of(partials.begin() + (part * m + live) * n,
            partials.begin() + (part + 1) * m * n, clean));
    ASSERT_TRUE(std::all_of(partials.begin() + partitions * m * n, partials.end(), clean));
    ASSERT_TRUE(std::all_of(output.begin() + live * n, output.end(), clean));
}

} // namespace

/** @test Every enabled capacity replays empty, partial, full and zero-after-full without recapture. */
TEST(ROCmGroupedHCDownLongK, DecoderPartialsAndFinalMatchSerialWithDirtyCapacity)
{
    const int ordinal = llaminar2::test::findQFNMI50ReplayDevice();
    if (ordinal < 0)
        GTEST_SKIP() << "HC-down long-K opt-in is scoped to 60-CU gfx906";
    checked(hipSetDevice(ordinal));
    const Stream streamOwner;
    const auto stream = streamOwner.native;
    const auto *mode = std::getenv("LLAMINAR_QFN_ROCM_GROUPED_HC_DOWN_LONG_K");
    const bool optedIn = mode && std::string_view(mode) == "1";
    Bank<uint8_t> packed(n * k);
    Bank<uint16_t> weightScales(n * groups);
    std::vector<uint8_t> source(n * k);
    std::vector<uint16_t> scales(n * groups);
    sourceWeights(source, scales);
    packed.input(source, stream);
    weightScales.input(scales, stream);

    Bank<int8_t> input(maxRows * k);
    Bank<float> activationScales(maxRows * groups);
    Bank<int> count(1);
    Bank<float> result(maxRows * n + guard), partials(scratchPartitions * maxRows * n + guard);
    Bank<float> reference(maxRows * n + guard), referencePartials(scratchPartitions * maxRows * n + guard);
    const std::vector<float> resultPoison(result.size, poison), partialPoison(partials.size, poison);

    for (int m = 2; m <= maxRows; ++m)
        for (int origin : {0, 1})
        {
            SCOPED_TRACE("M=" + std::to_string(m) + ", origin=" + std::to_string(origin));
            Capture grouped;
            const auto rows = DeviceRowRange::deviceCounted(m + origin, count.data).slice(origin, m);
            const auto enqueue = [&] {
                return rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
                    input.data, packed.data, weightScales.data, nullptr, nullptr,
                    result.data, activationScales.data, nullptr, partials.data,
                    m, n, k, q8, q8, ordinal, stream, &rows);
            };
            {
                const DecodeEquivalent policy;
                grouped.record(stream, enqueue);
            }
            expectGraph(grouped, stream, m, optedIn);
            if (m == 4 && origin == 0 && optedIn)
            {
                Capture alternate;
                const DecodeEquivalent policy;
                llaminar2::ProjectionRowWorkScope explicitWork(llaminar2::ProjectionRowWork::fromPrefix(rows));
                alternate.record(stream, enqueue);
                expectGraph(alternate, stream, m, false);
            }

            std::vector<int8_t> activation(input.size);
            std::vector<float> activationScale(activationScales.size);
            for (int history = 0; history < 4; ++history)
            {
                for (int i = 0; i < int(activation.size()); ++i)
                    activation[i] = int8_t((i * 7919 + 17 * history + i / 23) % 255 - 127);
                for (int i = 0; i < int(activationScale.size()); ++i)
                    activationScale[i] = float((i * 31 + 17 * history) % 4093 + 1) / 65536.0f;
                input.input(activation, stream);
                activationScales.input(activationScale, stream);
                reference.input(resultPoison, stream);
                referencePartials.input(partialPoison, stream);
                for (int row = 0; row < m; ++row)
                    ASSERT_TRUE(rocmGemv_native_quantised_fp32_with_policy(
                        input.data + row * k, packed.data, weightScales.data, nullptr, nullptr,
                        reference.data + row * n, nullptr, referencePartials.data + row * scratchPartitions * n,
                        n, k, q8, q8, ordinal, stream, activationScales.data + row * groups));
                const auto serialResults = reference.output(stream);
                const auto serialPartials = referencePartials.output(stream);
                ASSERT_TRUE(std::any_of(serialResults.begin(), serialResults.begin() + m * n,
                    [](float value) { return value != 0.0f && value != poison; }));

                // Replay high→empty first, then every live prefix from low to
                // high, and finally zero again with poisoned mutable scratch.
                std::vector<int> sequence{m, 0};
                for (int live = 1; live <= m; ++live) sequence.push_back(live);
                sequence.push_back(0);
                for (int live : sequence)
                {
                    const int published = live ? origin + live : 0;
                    checked(hipMemcpyAsync(count.data, &published, sizeof(published), hipMemcpyHostToDevice, stream));
                    result.input(resultPoison, stream);
                    partials.input(partialPoison, stream);
                    grouped.submit(stream);
                    grouped.submit(stream);
                    matchReplay(result.output(stream), partials.output(stream), serialResults, serialPartials, m, live);
                    ASSERT_FALSE(HasFatalFailure());
                }
            }
            if (origin == 0 && std::getenv("LLAMINAR_HCDOWN_NATIVE_EVENTS"))
            {
                checked(hipMemcpyAsync(count.data, &m, sizeof(m), hipMemcpyHostToDevice, stream));
                for (int rep = 0; rep < 8; ++rep) grouped.submit(stream);
                const Events event;
                for (int trial = 0; trial < 32; ++trial)
                {
                    checked(hipEventRecord(event.start, stream));
                    for (int rep = 0; rep < 48; ++rep) grouped.submit(stream);
                    checked(hipEventRecord(event.end, stream));
                    checked(hipEventSynchronize(event.end));
                    float elapsed = 0;
                    checked(hipEventElapsedTime(&elapsed, event.start, event.end));
                    std::cout << "HCDOWN_PRODUCTION_AB," << optedIn << ',' << m << ',' << trial
                              << ',' << elapsed * 1000.0f / 48 << '\n';
                }
            }
        }
}
