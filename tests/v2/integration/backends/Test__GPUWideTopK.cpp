/**
 * @file Test__GPUWideTopK.cpp
 * @brief Captured wide Top-K distributions: CPU ordering, serial bytes and live rows.
 *
 * Exercise the public backend operation with the unchanged admitted stochastic
 * scratch budget. Native CUDA/HIP captures survive large-to-small and empty
 * replays. Exact token ranks and serial-row probability bytes authenticate the
 * same law for every MTP row count; a CPU oracle independently checks Top-P.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/common/SamplingMath.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using namespace llaminar2;

/** @brief Own backend allocations even when an assertion exits the fixture. */
class Storage
{
public:
    /** @brief Allocate an already bounded test buffer on the selected device. */
    Storage(IBackend *backend, size_t bytes) : backend_(backend), pointer_(backend->allocate(bytes, 0)) {}
    /** @brief Return the allocation through its exact backend. */
    ~Storage() { if (pointer_) backend_->free(pointer_, 0); }
    /** @return Stable typed address embedded in the retained test graph. */
    template <class T> T *as() const { return static_cast<T *>(pointer_); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
private:
    IBackend *backend_;
    void *pointer_;
};

/** @brief One identical production sampling contract for each GPU backend. */
class WideTopK : public ::testing::TestWithParam<std::string> {};

/** @brief Capture all supported row counts, compare oracle/ranks and serial bytes. */
TEST_P(WideTopK, CapturedFullVocabularyAndEveryMTPRowCount)
{
    constexpr int rows = 16, stride = 256, scratch_entries = rows * 8192;
    IBackend *backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &worker = GetParam() == "CUDA"
        ? GPUDeviceContextPool::instance().getNvidiaContext(0)
        : GPUDeviceContextPool::instance().getAMDContext(0);
    worker.submitAndWait([&]
    {
        void *const stream = worker.defaultStream();
        ASSERT_NE(stream, nullptr);
        auto upload = [&](void *destination, const void *source, size_t bytes)
        { ASSERT_TRUE(backend->hostToDevice(destination, source, bytes, 0, stream)); };
        auto download = [&](void *destination, const void *source, size_t bytes)
        { ASSERT_TRUE(backend->deviceToHost(destination, source, bytes, 0, stream)); };
        Storage live(backend, sizeof(int));
        Storage values(backend, scratch_entries * sizeof(float));
        Storage indices(backend, scratch_entries * sizeof(int));
        Storage ids(backend, rows * stride * sizeof(int));
        Storage probs(backend, rows * stride * sizeof(float));
        Storage serial_ids(backend, rows * stride * sizeof(int));
        Storage serial_probs(backend, rows * stride * sizeof(float));
        ASSERT_NE(values.as<float>(), nullptr);
        ASSERT_NE(indices.as<int>(), nullptr);
        ASSERT_NE(live.as<int>(), nullptr);

        for (int vocabulary : {257, 248320, 300001})
        {
            SCOPED_TRACE(::testing::Message() << GetParam() << " vocabulary=" << vocabulary);
            std::vector<float> logits(static_cast<size_t>(rows) * vocabulary);
            std::vector<std::array<int, stride>> ranks(rows);
            for (int row = 0; row < rows; ++row)
            {
                float *input = logits.data() + static_cast<size_t>(row) * vocabulary;
                for (int token = 0; token < vocabulary; ++token)
                    // Many equal logits force independent exact token-id ties.
                    input[token] = static_cast<float>((token * 37LL + row * 101) % 8191) / 1024.0f;
                std::vector<int> order(vocabulary);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + stride, order.end(),
                    [&](int a, int b) { return input[a] > input[b] || (input[a] == input[b] && a < b); });
                std::copy_n(order.begin(), stride, ranks[row].begin());
            }
            Storage input(backend, logits.size() * sizeof(float));
            ASSERT_NE(input.as<float>(), nullptr);
            upload(input.as<float>(), logits.data(), logits.size() * sizeof(float));
            for (int k : {65, 96, 128, 255, 256})
            {
                SCOPED_TRACE(::testing::Message() << "top_k=" << k);
                const float top_p = k % 2 ? 0.9f : 1.0f;
                const float temperature = k % 2 ? 0.7f : 1.3f;
                int active = rows;
                upload(live.as<int>(), &active, sizeof(active));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                auto graph = worker.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                ASSERT_TRUE(graph->beginCapture());
                ASSERT_TRUE(backend->enqueueBuildTopKTopPDistributionsF32Device(
                    input.as<float>(), rows, vocabulary, vocabulary, k, top_p, temperature,
                    0, stream, ids.as<int>(), stride, probs.as<float>(),
                    values.as<float>(), indices.as<int>(), scratch_entries, live.as<int>()));
                ASSERT_TRUE(graph->endCapture());
                ASSERT_TRUE(graph->instantiate());
                std::vector<int> output_ids(rows * stride), reference_ids(rows * stride);
                std::vector<float> output_probs(rows * stride), reference_probs(rows * stride);
                for (int replay = 0; replay < 20; ++replay)
                {
                    active = replay < 16 ? replay + 1 : std::array<int, 4>{16, 1, 0, 16}[replay - 16];
                    SCOPED_TRACE(::testing::Message() << "replay=" << replay << " active=" << active);
                    std::fill(output_ids.begin(), output_ids.end(), -777);
                    std::fill(output_probs.begin(), output_probs.end(), -777.0f);
                    upload(ids.as<int>(), output_ids.data(), output_ids.size() * sizeof(int));
                    upload(probs.as<float>(), output_probs.data(), output_probs.size() * sizeof(float));
                    upload(live.as<int>(), &active, sizeof(active));
                    ASSERT_TRUE(graph->launch());
                    for (int row = 0; row < active; ++row)
                        ASSERT_TRUE(backend->enqueueBuildTopKTopPDistributionsF32Device(
                            input.as<float>() + static_cast<size_t>(row) * vocabulary,
                            1, vocabulary, vocabulary, k, top_p, temperature,
                            0, stream, serial_ids.as<int>() + row * stride, stride,
                            serial_probs.as<float>() + row * stride,
                            values.as<float>(), indices.as<int>(), scratch_entries));
                    download(output_ids.data(), ids.as<int>(), output_ids.size() * sizeof(int));
                    download(output_probs.data(), probs.as<float>(), output_probs.size() * sizeof(float));
                    download(reference_ids.data(), serial_ids.as<int>(), reference_ids.size() * sizeof(int));
                    download(reference_probs.data(), serial_probs.as<float>(), reference_probs.size() * sizeof(float));
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                    for (int row = 0; row < rows; ++row)
                    {
                        if (row < active)
                        {
                            std::array<float, stride> sorted{}, oracle_probs{}, weights{};
                            std::array<int, stride> oracle_ids{};
                            for (int rank = 0; rank < k; ++rank)
                                sorted[rank] = logits[static_cast<size_t>(row) * vocabulary + ranks[row][rank]];
                            sampling_math::build_topk_topp_distribution_from_sorted(
                                sorted.data(), ranks[row].data(), k, top_p, temperature,
                                oracle_ids.data(), oracle_probs.data(), weights.data());
                            for (int rank = 0; rank < k; ++rank)
                            {
                                EXPECT_EQ(output_ids[row * stride + rank], oracle_ids[rank]);
                                EXPECT_NEAR(output_probs[row * stride + rank], oracle_probs[rank], 1e-6f);
                            }
                            EXPECT_EQ(std::memcmp(output_ids.data() + row * stride,
                                reference_ids.data() + row * stride, k * sizeof(int)), 0);
                            EXPECT_EQ(std::memcmp(output_probs.data() + row * stride,
                                reference_probs.data() + row * stride, k * sizeof(float)), 0);
                        }
                        for (int rank = row < active ? k : 0; rank < stride; ++rank)
                        {
                            EXPECT_EQ(output_ids[row * stride + rank], -777);
                            EXPECT_EQ(output_probs[row * stride + rank], -777.0f);
                        }
                    }
                }
            }
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, WideTopK, ::testing::Values("CUDA", "ROCm"),
                        [](const auto &info) { return info.param; });
} // namespace
