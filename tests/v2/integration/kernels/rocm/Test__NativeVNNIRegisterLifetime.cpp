/**
 * @file Test__NativeVNNIRegisterLifetime.cpp
 * @brief Captured all-format proof for register-bounded ROCm GEMM schedules.
 *
 * This deliberately small fixture links the actual public GEMM/GEMV dispatchers
 * and their production shards, not a reimplemented kernel or a model runner.
 * It can therefore diagnose a kernel change without rebuilding unrelated GPU
 * subsystems. Serial row replay is an oracle only; the candidate is one grouped
 * production launch. Every retained replay poisons its output and guard tail.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "backends/rocm/ROCmRuntimeStartup.h"
#include "kernels/rocm/gemm/ROCmNativeVNNIGemmShard.h"
#include "tensors/IQQuantTables.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

extern "C" {
bool rocmInitIQGridTables(int, const void*, const void*, const void*,
                         const void*, const void*, const void*);
bool rocmInitIQGridTables_gemm(int, const void*, const void*, const void*,
                              const void*, const void*, const void*);
bool rocmGemm_native_vnni_fp32(const int8_t*, const uint8_t*, const void*,
    const void*, const void*, float*, const float*, const float*,
    int, int, int, uint8_t, int, void*);
bool rocmGemv_native_vnni_fp32(const int8_t*, const uint8_t*, const void*,
    const void*, const void*, float*, const float*, float*, int, int,
    uint8_t, int, void*, const float*);
bool rocmGemv_native_vnni_query_serial_m1_config_with_policy(
    uint8_t, int, int, int*, int*);
}

namespace {
/** @brief Surface the originating HIP failure before inspecting any result. */
void checked(hipError_t status)
{
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

/** @brief Test-only persistent storage; never allocated within capture. */
template<class T> class Storage final
{
public:
    /** @brief Allocate the exact fixture extent before graph recording. */
    explicit Storage(size_t count) : size(count) { checked(hipMalloc(&data, count * sizeof(T))); }
    /** @brief Release only after the later-declared graph owner has retired. */
    ~Storage() { (void)hipFree(data); }
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    /** @brief Publish fixture bytes on the exact test stream. */
    void upload(const std::vector<T>& source, hipStream_t stream)
    {
        if (!stream || source.size() != size) throw std::logic_error("invalid fixture upload");
        checked(hipMemcpyAsync(data, source.data(), size * sizeof(T), hipMemcpyHostToDevice, stream));
    }
    T* data = nullptr;
    size_t size;
};

/** @brief Exact stream and graph owner, declared after every retained buffer. */
class Execution final
{
public:
    /** @brief Initialize the non-default stream and terminal observation event. */
    Execution()
    {
        checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        checked(hipEventCreateWithFlags(&terminal, hipEventDisableTiming));
    }
    /** @brief Join all consumers before destroying graphs and storage. */
    ~Execution()
    {
        (void)hipEventRecord(terminal, stream);
        (void)hipEventSynchronize(terminal);
        for (auto graph : graphs) if (graph) (void)hipGraphExecDestroy(graph);
        (void)hipEventDestroy(terminal);
        (void)hipStreamDestroy(stream);
    }
    /** @brief Observe test results without a device-wide synchronization. */
    void observe()
    {
        checked(hipEventRecord(terminal, stream));
        checked(hipEventSynchronize(terminal));
    }
    hipStream_t stream = nullptr;
    hipEvent_t terminal = nullptr;
    std::array<hipGraphExec_t, 2> graphs{};
};

/** @brief Reproducible signed, nontrivial inputs without model-loading overhead. */
uint32_t mixed(uint32_t v)
{
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU;
    return v ^ (v >> 16);
}

/** @brief Prove grouped bytes and guard tails across twenty captured replays. */
void verify(int codebook, int payload_bytes, int m, int n, int k)
{
    const int blocks = k / 32;
    const size_t weight_blocks = static_cast<size_t>(blocks) * n;
    int partitions = 0, waves = 0;
    const auto policy = static_cast<uint8_t>(
        codebook == llaminar2::kNativeVnniExpandedInt8MinCodebook ? 19 : codebook);
    ASSERT_TRUE(rocmGemv_native_vnni_query_serial_m1_config_with_policy(
        policy, n, k, &partitions, &waves));
    ASSERT_GT(partitions, 0);
    std::vector<uint8_t> payload(weight_blocks * payload_bytes);
    std::vector<uint16_t> scales(weight_blocks), secondary(weight_blocks);
    std::vector<uint32_t> minima(weight_blocks);
    std::vector<int8_t> activation(m * k);
    std::vector<float> a_scales(m * blocks), row_scales(m, 1.0f);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = mixed(i + 203);
    for (size_t i = 0; i < weight_blocks; ++i)
    {
        scales[i] = static_cast<uint16_t>(0x1000U + (mixed(i) & 0x1fffU));
        secondary[i] = static_cast<uint16_t>(0x1000U + (mixed(i + 77) & 0x1fffU));
        minima[i] = scales[i] | (static_cast<uint32_t>(secondary[i]) << 16);
    }
    for (size_t i = 0; i < activation.size(); ++i)
        activation[i] = static_cast<int>(mixed(i + 389) % 255) - 127;
    for (size_t i = 0; i < a_scales.size(); ++i)
        a_scales[i] = static_cast<float>(1 + mixed(i + 7) % 31) / 256;

    Storage<uint8_t> p(payload.size());
    Storage<uint16_t> s(weight_blocks), secondary_s(weight_blocks);
    Storage<uint32_t> em(weight_blocks);
    Storage<int8_t> a(activation.size());
    const size_t output_size = static_cast<size_t>(m) * n + 8;
    Storage<float> sa(a_scales.size()), sr(m), partials(static_cast<size_t>(n) * partitions);
    Storage<float> expected(output_size), actual(output_size);
    Execution execution;
    const auto stream = execution.stream;
    p.upload(payload, stream); s.upload(scales, stream); secondary_s.upload(secondary, stream);
    em.upload(minima, stream); a.upload(activation, stream);
    sa.upload(a_scales, stream); sr.upload(row_scales, stream);
    execution.observe();

    for (int path = 0; path < 2; ++path)
    {
        checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        float* destination = path == 0 ? expected.data : actual.data;
        checked(hipMemsetAsync(destination, 0x5a, output_size * sizeof(float), stream));
        bool launched = true;
        if (path == 0)
            for (int row = 0; row < m; ++row)
                launched &= rocmGemv_native_vnni_fp32(
                    a.data + row * k, p.data, s.data, secondary_s.data, em.data,
                    destination + row * n, sr.data + row, partials.data,
                    n, k, static_cast<uint8_t>(codebook), 0, stream, sa.data + row * blocks);
        else
            launched = rocmGemm_native_vnni_fp32(
                a.data, p.data, s.data, secondary_s.data, em.data, destination,
                sr.data, sa.data, m, n, k, static_cast<uint8_t>(codebook), 0, stream);
        hipGraph_t graph = nullptr;
        checked(hipStreamEndCapture(stream, &graph));
        const auto status = hipGraphInstantiate(&execution.graphs[path], graph, nullptr, nullptr, 0);
        checked(hipGraphDestroy(graph));
        checked(status);
        ASSERT_TRUE(launched);
    }
    for (int replay = 0; replay < 20; ++replay)
        for (auto graph : execution.graphs) checked(hipGraphLaunch(graph, stream));

    std::vector<float> oracle(output_size), candidate(output_size);
    checked(hipMemcpyAsync(oracle.data(), expected.data, output_size * sizeof(float), hipMemcpyDeviceToHost, stream));
    checked(hipMemcpyAsync(candidate.data(), actual.data, output_size * sizeof(float), hipMemcpyDeviceToHost, stream));
    execution.observe();
    for (size_t i = 0; i < output_size - 8; ++i)
    {
        ASSERT_TRUE(std::isfinite(oracle[i])) << "index=" << i;
        ASSERT_EQ(std::bit_cast<uint32_t>(oracle[i]), std::bit_cast<uint32_t>(candidate[i]))
            << "index=" << i << " oracle=" << oracle[i] << " candidate=" << candidate[i];
    }
    for (size_t i = output_size - 8; i < output_size; ++i)
        ASSERT_EQ(std::bit_cast<uint32_t>(candidate[i]), 0x5a5a5a5aU);
}
} // namespace

/** @brief Cover every physical format, streaming/cooperative shapes, and row tails. */
TEST(ROCmNativeVNNIRegisterLifetime, CapturedAllFormatsMatchSerialRows)
{
    llaminar2::requireROCmRuntimeStartup();
    int devices = 0;
    checked(hipGetDeviceCount(&devices));
    ASSERT_GT(devices, 0);
    checked(hipSetDevice(0));
    using namespace llaminar2;
    for (auto initialize : {rocmInitIQGridTables, rocmInitIQGridTables_gemm})
        ASSERT_TRUE(initialize(0, iq3s_grid, iq3xxs_grid, iq2s_grid,
                               iq2xs_grid, iq2xxs_grid, iq1s_grid));
    std::map<int, int> formats;
    for (const auto& source : native_vnni_formats::kAllSourceFormats)
    {
        const auto& format = *source.metadata;
        formats.emplace(canonicalDeviceVnniCodebookId(format.codebook_id), format.payload_bytes);
        const auto promoted = migrationStableDeviceVnniFormat(format);
        formats.emplace(promoted.codebook_id, promoted.payload_bytes_per_block);
    }
    // Exercise exact tiles as well as tails: aligned Q6_K selects its
    // full-tile specialization, while large-N formats exercise streaming.
    // Both sides of every row-tile boundary must preserve serial bytes.
    for (const auto& [codebook, payload_bytes] : formats)
        for (const auto& [n, k] : std::array{
                 std::pair{1024, 256}, std::pair{1025, 160},
                 std::pair{16384, 256}, std::pair{16385, 256}})
            for (int m : {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65})
            {
                SCOPED_TRACE(::testing::Message() << "codebook=" << codebook
                    << " M=" << m << " N=" << n << " K=" << k);
                verify(codebook, payload_bytes, m, n, k);
                if (::testing::Test::HasFatalFailure()) return;
            }
}
