/**
 * @file Test__CUDANativeVNNIWide.cpp
 * @brief Captured WIDE decode totality against the canonical single-K-part oracle.
 *
 * This model-free fixture calls the production launch bridge for both paths.
 * KPAR with one partition retains WIDE's ascending block-contribution tree,
 * but has independent activation loading and output publication. Every replay
 * poisons the outputs before computing them; guard bytes detect tail writes.
 * The format catalog supplies both ordinary and CPU-promotion representations.
 * This functional gate deliberately contains no performance threshold.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C"
{
    bool cudaNativeVNNIInitIQGridTables_tuned();
    void cudaNativeVNNIGemvSweep_setConfig(int, int, int, int, int, int, int, int);
    void cudaNativeVNNIGemvSweep_clearConfig();
    bool cudaNativeVNNIGemvTuned_fp32(
        const int8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
        const uint32_t *, float *, const float *, int, int, float, float,
        const float *, const float *, uint8_t, int, void *, CUDAGemvContext *,
        CUDARowMajorWeights **);
}

namespace
{
/** @brief Fail at the originating CUDA operation rather than comparing stale data. */
void checked(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

/** @brief Test-only storage, allocated before recording and freed after retirement. */
template<class T> class Storage final
{
public:
    /** @brief Allocate exactly the requested native element count. */
    explicit Storage(size_t count) : count_(count)
    {
        checked(cudaMalloc(&data, count * sizeof(T)));
    }
    /** @brief Release after the execution owner has joined the terminal stream. */
    ~Storage() { (void)cudaFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;

    /** @brief Stage complete fixture data on its exact non-default stream. */
    void upload(const std::vector<T> &values, cudaStream_t stream)
    {
        if (!stream || values.size() != count_)
            throw std::logic_error("invalid WIDE fixture publication");
        checked(cudaMemcpyAsync(data, values.data(), count_ * sizeof(T),
                                cudaMemcpyHostToDevice, stream));
    }
    T *data = nullptr;
private:
    size_t count_;
};

/** @brief One execution owner, declared after the storage its graphs retain. */
class Execution final
{
public:
    /** @brief Create an exact stream and a context with externally bound partials. */
    Execution() : context(cudaGemvContext_create(0), cudaGemvContext_destroy)
    {
        if (!context) throw std::runtime_error("missing native GEMV context");
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    /** @brief Join before graph destruction, stream destruction, and buffer release. */
    ~Execution()
    {
        (void)cudaStreamSynchronize(stream);
        graphs = {};
        (void)cudaStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    std::unique_ptr<CUDAGemvContext, decltype(&cudaGemvContext_destroy)> context;
    cudaStream_t stream = nullptr;
    std::array<std::unique_ptr<llaminar2::CUDAGraphCapture>, 2> graphs;
};

/** @brief Named production families used by this arithmetic-equivalence proof. */
enum class Family : int { Wide = 0, SinglePartition = 1 };

/** @brief Own the existing diagnostic override only during native recording. */
class RecordingPolicy final
{
public:
    /** @brief Select a physical WIDE tile or the exact one-partition KPAR oracle. */
    RecordingPolicy(Family family, int tile, int columns_per_thread)
    {
        cudaNativeVNNIGemvSweep_setConfig(
            static_cast<int>(family), tile, columns_per_thread,
            1, 1, 1, 1, family == Family::SinglePartition ? 1 : 0);
    }
    /** @brief Restore ordinary production dispatch even when recording throws. */
    ~RecordingPolicy() { cudaNativeVNNIGemvSweep_clearConfig(); }
    RecordingPolicy(const RecordingPolicy &) = delete;
    RecordingPolicy &operator=(const RecordingPolicy &) = delete;
};

/** @brief Stable signed, nonzero fixture values without model loading. */
uint32_t mixed(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

/**
 * @brief Byte-certify one codebook/tile/tail through twenty retained replays.
 * @param codebook Canonical physical execution codebook, not a source alias.
 * @param payload_bytes Exact packed bytes per 32-element weight block.
 * @param tile Output columns assigned to a WIDE CTA.
 * @param cpt Consecutive output columns assigned to each thread.
 * @param n Logical output width straddling a physical tile boundary.
 * @param k Input width including single-block and odd-block-count witnesses.
 */
void verifyWide(int codebook, int payload_bytes, int tile, int cpt, int n, int k)
{
    const size_t blocks = static_cast<size_t>(n) * (k / 32);
    std::vector<uint8_t> payload(blocks * payload_bytes);
    std::vector<uint16_t> scales(blocks), secondary(blocks);
    std::vector<uint32_t> minima(blocks);
    std::vector<int8_t> activations(k);
    std::vector<float> activation_scales(k / 32);
    for (size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<uint8_t>(mixed(i + 203));
    for (size_t i = 0; i < blocks; ++i)
    {
        scales[i] = static_cast<uint16_t>(0x1000U + (mixed(i) & 0x1fffU));
        secondary[i] = static_cast<uint16_t>(0x1000U + (mixed(i + 77) & 0x1fffU));
        minima[i] = scales[i] | (static_cast<uint32_t>(secondary[i]) << 16);
    }
    for (int i = 0; i < k; ++i)
        activations[i] = static_cast<int>(mixed(i + 389) % 255) - 127;
    for (int i = 0; i < k / 32; ++i)
        activation_scales[i] = static_cast<float>(1 + mixed(i + 7) % 31) / 256;

    Storage<uint8_t> p(payload.size());
    Storage<uint16_t> s(blocks), secondary_s(blocks);
    Storage<uint32_t> em(blocks);
    Storage<int8_t> a(k);
    Storage<float> sa(k / 32), partials(n), oracle(n + 8), output(n + 8);
    Execution execution;
    const auto stream = execution.stream;
    p.upload(payload, stream); s.upload(scales, stream);
    secondary_s.upload(secondary, stream); em.upload(minima, stream);
    a.upload(activations, stream); sa.upload(activation_scales, stream);
    cudaGemvContext_bindWorkspace(execution.context.get(), partials.data, n * sizeof(float));
    checked(cudaStreamSynchronize(stream));

    for (int path = 0; path < 2; ++path)
    {
        auto &graph = execution.graphs[path];
        graph = std::make_unique<llaminar2::CUDAGraphCapture>(stream, 0);
        RecordingPolicy policy(path == 0 ? Family::SinglePartition : Family::Wide,
                               path == 0 ? 64 : tile, path == 0 ? 1 : cpt);
        llaminar2::ScopedBackendGraphCapture recording(*graph, "native WIDE tail proof");
        ASSERT_TRUE(recording.begin());
        float *destination = path == 0 ? oracle.data : output.data;
        // This poison is part of the replay, not just setup: stale outputs
        // cannot masquerade as complete publication on a later launch.
        checked(cudaMemsetAsync(destination, 0x5a, (n + 8) * sizeof(float), stream));
        ASSERT_TRUE(cudaNativeVNNIGemvTuned_fp32(
            a.data, p.data, s.data, secondary_s.data, em.data, destination,
            sa.data, n, k, 1.0f, 0.0f, nullptr, nullptr,
            static_cast<uint8_t>(codebook), 0, stream,
            execution.context.get(), nullptr));
        recording.finish();
        ASSERT_TRUE(graph->instantiate());
    }
    std::vector<float> expected(n + 8), actual(n + 8);
    for (int replay = 0; replay < 20; ++replay)
    {
        ASSERT_TRUE(execution.graphs[0]->launch());
        ASSERT_TRUE(execution.graphs[1]->launch());
        checked(cudaMemcpyAsync(expected.data(), oracle.data, (n + 8) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(actual.data(), output.data, (n + 8) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
        checked(cudaStreamSynchronize(stream));
        ASSERT_EQ(std::memcmp(expected.data(), actual.data(), n * sizeof(float)), 0)
            << "replay=" << replay;
        for (int i = n; i < n + 8; ++i)
        {
            ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), 0x5a5a5a5aU);
            ASSERT_EQ(std::bit_cast<uint32_t>(expected[i]), 0x5a5a5a5aU);
        }
    }
}
} // namespace

/** @brief Sweep every physical codebook and every compiled WIDE output geometry. */
TEST(CUDANativeVNNIWide, CapturedAllCodebooksAndTileTailsMatchSinglePartition)
{
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0) << "explicit CUDA integration requires a device";
    checked(cudaSetDevice(0));
    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned());
    std::map<int, int> formats;
    for (const auto &source : llaminar2::native_vnni_formats::kAllSourceFormats)
    {
        const auto &format = *source.metadata;
        formats.emplace(llaminar2::canonicalDeviceVnniCodebookId(format.codebook_id),
                        format.payload_bytes);
        const auto promoted = llaminar2::migrationStableDeviceVnniFormat(format);
        formats.emplace(promoted.codebook_id, promoted.payload_bytes_per_block);
    }
    for (const auto &[codebook, payload_bytes] : formats)
        for (const auto &[tile, cpt] : std::array{
                 std::pair{32, 1}, std::pair{64, 1}, std::pair{64, 2},
                 std::pair{128, 1}, std::pair{128, 2}, std::pair{256, 2},
                 std::pair{256, 4}, std::pair{512, 4}})
            for (const auto &[delta, k] : std::array{
                     std::pair{-1, 32}, std::pair{0, 96}, std::pair{1, 640}})
            {
                SCOPED_TRACE(::testing::Message()
                    << "codebook=" << codebook << " tile=" << tile
                    << " cpt=" << cpt << " N=" << tile + delta << " K=" << k);
                verifyWide(codebook, payload_bytes, tile, cpt, tile + delta, k);
                if (::testing::Test::HasFatalFailure()) return;
            }
}
