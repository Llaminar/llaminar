/**
 * @file MoEGroupedFixedFP16RouterReplay.inc
 * @brief Captured small-row native FP16 router against serial M1 and dirty tails.
 */
#include <cstring>

extern "C" bool hipMoE_gate_logits_single_token_fp16_weights(
    const float*, const void*, float*, int, int, int, void*);

namespace llaminar2::test
{
namespace
{
    /** @brief Check counted fixed-row native projection and original M1 bytes. */
    void proveFixedFP16Router(int rows)
    {
        constexpr int hidden = 2560, experts = 512;
        constexpr float dirty = -91.5f;
        const DeviceId device = DeviceId::rocm(0);
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            auto *stream = worker.defaultStream();
            requireRouterLogits(backend && stream, "fixed FP16 router test stream");
            auto input = TestTensorFactory::createFP32({size_t(rows) * hidden});
            auto weights = TestTensorFactory::createFP16({size_t(experts) * hidden});
            auto logits = TestTensorFactory::createFP32({size_t(rows + 1) * experts});
            auto oracle = TestTensorFactory::createFP32({size_t(rows + 1) * experts});
            auto count = TestTensorFactory::createINT32({1});
            auto graph = worker.createGraphCapture(stream);
            RouterObservationJoin retire{*backend, device, stream};
            auto *half_weights = weights->mutable_typed_data();
            for (size_t i = 0; i < weights->numel(); ++i)
                half_weights[i] = fp32_to_fp16(float(int((i * 17 + i / hidden) % 251) - 125) / 4093.f);
            requireRouterLogits(weights->ensureOnDevice(device, stream) &&
                input->allocateOnDevice(device, stream) && logits->allocateOnDevice(device, stream) &&
                oracle->allocateOnDevice(device, stream) && count->allocateOnDevice(device, stream),
                "admit all fixture storage before capture");
            const auto *source = static_cast<const float *>(input->gpu_data_ptr());
            auto *output = static_cast<float *>(logits->gpu_data_ptr());
            auto *expected_output = static_cast<float *>(oracle->gpu_data_ptr());
            const auto *half = weights->gpu_data_ptr();
            const auto *live_count = static_cast<const int *>(count->gpu_data_ptr());
            {
                ScopedBackendGraphCapture recording(worker, *graph, "counted FP16 router replay");
                requireRouterLogits(recording.begin() &&
                    hipMoE_gate_logits_fp16_decode_equivalent_rows(source, half, output, rows, hidden, experts,
                        device.ordinal, stream, live_count), "record original grouped projection bridge");
                recording.finish();
            }
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graph->inspectKernelNodes(nodes));
            ASSERT_EQ(nodes.size(), 1u);
            ASSERT_TRUE(nodes.front().name_resolved);
            EXPECT_NE(nodes.front().name.find("qfn_grouped_fp16_router_fixed_rows"), std::string::npos);
            EXPECT_EQ(nodes.front().grid_x, unsigned(experts));
            EXPECT_EQ(nodes.front().block_x, 128u);
            EXPECT_EQ(nodes.front().local_memory_bytes_per_thread, 0u);
            requireRouterLogits(graph->instantiate(), "instantiate fixed-row projection");

            std::vector<float> source_values(input->numel());
            std::vector<float> expected(oracle->numel()), observed(logits->numel()), poison(logits->numel(), dirty);
            const auto upload = [&](void *target, const auto *data, size_t elements) {
                requireRouterLogits(backend->hostToDevice(target, data, elements * sizeof(*data), device.ordinal, stream),
                    "publish counted fixture input");
            };
            for (int history = 0; history < 3; ++history)
            {
                for (size_t i = 0; i < source_values.size(); ++i)
                {
                    float value = float(int((i * 31 + i / hidden * 7) % 229) - 114) / 259.f;
                    if (history == 1) value = i % 2 ? -0.f : 0.f;
                    if (history == 2) value = float(int(i % 7) - 3) * 0x1p-25f;
                    source_values[i] = value;
                }
                upload(input->gpu_data_ptr(), source_values.data(), source_values.size());
                for (int live = rows; live >= 0; --live)
                {
                    SCOPED_TRACE(::testing::Message() << "rows=" << rows << " live=" << live << " history=" << history);
                    upload(count->gpu_data_ptr(), &live, 1);
                    upload(output, poison.data(), poison.size());
                    upload(expected_output, poison.data(), poison.size());
                    for (int row = 0; row < live; ++row)
                        requireRouterLogits(hipMoE_gate_logits_single_token_fp16_weights(source + size_t(row) * hidden,
                            half, expected_output + size_t(row) * experts, hidden, experts, device.ordinal, stream),
                            "enqueue independent original M1 projection");
                    requireRouterLogits(graph->launch() &&
                        backend->deviceToHost(expected.data(), expected_output, expected.size() * sizeof(float), device.ordinal, stream) &&
                        backend->deviceToHost(observed.data(), output, observed.size() * sizeof(float), device.ordinal, stream) &&
                        backend->synchronizeStream(stream, device.ordinal), "observe retained grouped projection");
                    EXPECT_EQ(std::memcmp(expected.data(), observed.data(), observed.size() * sizeof(float)), 0);
                }
            }
        });
    }
} // namespace

/** @test All admitted physical small-row classes preserve serial M1 projection and inactive guards. */
TEST(MoERouterLogits, ROCmGroupedFixedFP16RouterCapturedByteReplay)
{
    ASSERT_TRUE(qfnGroupedFP16RouterFixedRowsOptIn());
    for (int rows : {4, 7, 8})
        proveFixedFP16Router(rows);
}
} // namespace llaminar2::test
