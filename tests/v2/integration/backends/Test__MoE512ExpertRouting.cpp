/**
 * @file Test__MoE512ExpertRouting.cpp
 * @brief Captured serial/grouped byte equivalence through the 512th routed expert.
 *
 * Production softmax/top-k bridges consume identical retained buffers. Replays
 * alternate ties and skew, require expert 511 to win, and compare every weight
 * bit plus the authoritative runtime histogram. No model weights are needed.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoERuntimeTable.h"
#include "../../utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
extern "C" bool cudaMoE_softmax_topk_decode_unique_owner_runtime(
    float *, void *, float *, float *, int, int, bool, bool, bool, int, void *);
extern "C" bool cudaMoE_softmax_topk(
    float *, float *, float *, int, int, int, bool, int, void *, const int *, void *);
#endif
#ifdef HAVE_ROCM
extern "C" bool hipMoE_softmax_topk_decode_unique_owner_wave64(
    float *, void *, float *, float *, int, int, bool, bool, bool, int, void *);
extern "C" bool hipMoE_softmax_topk_decode_equivalent_rows(
    float *, float *, float *, int, int, int, bool, int, void *, const int *, void *);
#endif

namespace llaminar2::test
{
    /** @brief Both vendors execute the same retained routing geometry and assertions. */
    class MoE512ExpertRouting : public ::testing::TestWithParam<std::string> {};

    /** @test Twenty retained serial/grouped replays preserve all route IDs and exact weight bits. */
    TEST_P(MoE512ExpertRouting, CapturedSerialGroupedWeightsAndLastExpertAreExact)
    {
        const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            ASSERT_NE(backend, nullptr);
            ASSERT_NE(stream, nullptr);
            constexpr int rows = 16;
            for (const int experts : {1, 63, 64, 65, 255, 256, 257, 319, 383, 447, 511, 512})
            {
                SCOPED_TRACE(experts);
                const int top_k = std::min(experts, 10);
                const size_t logits_count = rows * experts;
                const size_t route_count = rows * top_k;
                // Tensor fixtures supply RAII storage; this opaque bank is not
                // an FP32 activation and is copied using its exact typed ABI.
                auto serial_logits = TestTensorFactory::createFP32({logits_count});
                auto grouped_logits = TestTensorFactory::createFP32({logits_count});
                auto serial_ids = TestTensorFactory::createFP32({route_count});
                auto grouped_ids = TestTensorFactory::createFP32({route_count});
                auto serial_weights = TestTensorFactory::createFP32({route_count});
                auto grouped_weights = TestTensorFactory::createFP32({route_count});
                auto runtime_storage = TestTensorFactory::createFP32({(sizeof(DeviceMoELayerRuntime) + 3) / 4});
                auto participant_storage = TestTensorFactory::createFP32({static_cast<size_t>(top_k)});
                for (auto *storage : {serial_logits.get(), grouped_logits.get(), serial_ids.get(),
                                     grouped_ids.get(), serial_weights.get(), grouped_weights.get(),
                                     runtime_storage.get(), participant_storage.get()})
                    ASSERT_TRUE(storage->ensureOnDevice(device, stream));
                auto runtime = std::make_unique<DeviceMoELayerRuntime>();
                runtime->expert_count = experts;
                runtime->top_k = top_k;
                runtime->active_epoch = 1;
                runtime->banks[0].epoch = 1;
                runtime->banks[0].expert_count = experts;
                runtime->route_participant_ids = static_cast<int32_t *>(participant_storage->gpu_data_ptr());
                runtime->prefill_route_capacity = top_k;
                for (int expert = 0; expert < experts; ++expert)
                {
                    auto &bank = runtime->banks[0];
                    bank.local_compute_mask[expert] = 1;
                    bank.resident_participant_mask[expert] = 1;
                    bank.experts[expert].logical_expert_id = expert;
                    bank.experts[expert].owner_participant = 0;
                }
                const auto enqueue = [&] {
                    bool ok = true;
                    for (int row = 0; row < rows; ++row)
                    {
                        auto *logits = static_cast<float *>(serial_logits->gpu_data_ptr()) + row * experts;
                        auto *ids = static_cast<float *>(serial_ids->gpu_data_ptr()) + row * top_k;
                        auto *weights = static_cast<float *>(serial_weights->gpu_data_ptr()) + row * top_k;
#ifdef HAVE_CUDA
                        if (device.is_cuda())
                            ok = ok && cudaMoE_softmax_topk_decode_unique_owner_runtime(
                                logits, runtime_storage->gpu_data_ptr(), ids, weights,
                                experts, top_k, true, true, true, device.ordinal, stream);
#endif
#ifdef HAVE_ROCM
                        if (device.is_rocm())
                            ok = ok && hipMoE_softmax_topk_decode_unique_owner_wave64(
                                logits, runtime_storage->gpu_data_ptr(), ids, weights,
                                experts, top_k, true, true, true, device.ordinal, stream);
#endif
                    }
#ifdef HAVE_CUDA
                    if (device.is_cuda())
                        ok = ok && cudaMoE_softmax_topk(
                            static_cast<float *>(grouped_logits->gpu_data_ptr()),
                            static_cast<float *>(grouped_ids->gpu_data_ptr()),
                            static_cast<float *>(grouped_weights->gpu_data_ptr()),
                            rows, experts, top_k, true, device.ordinal, stream, nullptr, nullptr);
#endif
#ifdef HAVE_ROCM
                    if (device.is_rocm())
                        ok = ok && hipMoE_softmax_topk_decode_equivalent_rows(
                            static_cast<float *>(grouped_logits->gpu_data_ptr()),
                            static_cast<float *>(grouped_ids->gpu_data_ptr()),
                            static_cast<float *>(grouped_weights->gpu_data_ptr()),
                            rows, experts, top_k, true, device.ordinal, stream, nullptr, nullptr);
#endif
                    return ok;
                };
                auto graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                {
                    GraphCaptureGuard guard;
                    ASSERT_TRUE(graph->beginCapture());
                    ASSERT_TRUE(enqueue());
                    ASSERT_TRUE(graph->endCapture());
                }
                ASSERT_TRUE(graph->instantiate());
                for (int replay = 0; replay < 20; ++replay)
                {
                    SCOPED_TRACE(replay);
                    std::vector<float> input(logits_count);
                    for (int row = 0; row < rows; ++row)
                        for (int expert = 0; expert < experts; ++expert)
                            input[row * experts + expert] = replay % 2 == 0 ? 0.f :
                                std::sin(float(expert * 17 + row * 31 + replay));
                    // Never leave the upper half unexercised, even for a tied
                    // row: the final expert wins each transaction and histogram.
                    for (int row = 0; row < rows; ++row)
                        input[row * experts + experts - 1] = 8.f;
                    std::fill_n(runtime->decode_histogram, kDeviceMoEMaxExperts, uint64_t{0});
                    std::fill_n(runtime->decode_local_histogram, kDeviceMoEMaxExperts, uint64_t{0});
                    ASSERT_TRUE(backend->hostToDeviceOnStream(serial_logits->gpu_data_ptr(), input.data(), input.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->hostToDeviceOnStream(grouped_logits->gpu_data_ptr(), input.data(), input.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->hostToDeviceOnStream(runtime_storage->gpu_data_ptr(), runtime.get(), sizeof(*runtime), device.ordinal, stream));
                    ASSERT_TRUE(graph->launch());
                    std::vector<float> ids(route_count), grouped(route_count), weights(route_count), grouped_w(route_count);
                    auto after = std::make_unique<DeviceMoELayerRuntime>();
                    ASSERT_TRUE(backend->deviceToHostOnStream(ids.data(), serial_ids->gpu_data_ptr(), ids.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->deviceToHostOnStream(grouped.data(), grouped_ids->gpu_data_ptr(), grouped.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->deviceToHostOnStream(weights.data(), serial_weights->gpu_data_ptr(), weights.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->deviceToHostOnStream(grouped_w.data(), grouped_weights->gpu_data_ptr(), grouped_w.size() * sizeof(float), device.ordinal, stream));
                    ASSERT_TRUE(backend->deviceToHostOnStream(after.get(), runtime_storage->gpu_data_ptr(), sizeof(*after), device.ordinal, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                    EXPECT_EQ(ids, grouped);
                    for (size_t route = 0; route < route_count; ++route)
                        EXPECT_EQ(std::bit_cast<uint32_t>(weights[route]), std::bit_cast<uint32_t>(grouped_w[route])) << route;
                    for (int row = 0; row < rows; ++row)
                        EXPECT_EQ(ids[row * top_k], experts - 1);
                    EXPECT_EQ(after->decode_histogram[experts - 1], rows);
                    EXPECT_EQ(after->decode_local_histogram[experts - 1], rows);
                    EXPECT_EQ(std::accumulate(std::begin(after->decode_histogram), std::end(after->decode_histogram), uint64_t{0}), route_count);
                }
            }
        });
    }

    /** @brief Enumerate only backends compiled into this executable. */
    std::vector<std::string> routingBackends()
    {
        std::vector<std::string> result;
#ifdef HAVE_CUDA
        result.emplace_back("CUDA");
#endif
#ifdef HAVE_ROCM
        result.emplace_back("ROCm");
#endif
        return result;
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MoE512ExpertRouting,
        ::testing::ValuesIn(routingBackends()),
        [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
}
