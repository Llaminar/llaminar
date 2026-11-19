/**
 * @file MoEGroupedRegisterTopKReplay.inc
 * @brief Replay the original counted Top-K dispatch against independent serial rows.
 *
 * The selector's numerical input is FP32 logits for every prepared router
 * policy. This fixture drives the shared production bridge with finite rows,
 * many equal winners, nonfinite rows and retained dirty storage. The serial
 * diagnostics use the real M1 path and independent deferred ledgers.
 */
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/moe/MoERuntimeTable.h"
#include "utils/DebugEnv.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"

#include <array>
#include <limits>

extern "C" bool hipMoE_softmax_topk_decode_equivalent_rows(
    float *logits, float *indices, float *weights, int rows, int experts, int top_k,
    bool normalize, int device, void *stream, const int *live_rows, void *deferred_ledger);

extern "C" bool hipMoE_gate_logits_fp32_decode_equivalent_rows(
    const float*, const float*, float*, int, int, int, int, void*, const int*);
extern "C" bool hipMoE_gate_logits_fp16_decode_equivalent_rows(
    const float*, const void*, float*, int, int, int, int, void*, const int*);
extern "C" bool hipMoE_gate_logits_bf16_decode_equivalent_rows(
    const float*, const void*, float*, int, int, int, int, void*, const int*);

namespace llaminar2::test
{
namespace
{
    constexpr int replay_experts = 512, replay_rows = 9;
    using RouterPolicyLogits = std::array<std::vector<float>, 4>;

    /**
     * @brief Record FP32 logits from all four original router arithmetic bridges.
     * @return Native FP32, FP16, BF16 and signed-Q8 results from one bounded QFN geometry.
     *
     * The fixture uses generated finite native operands. It runs the original
     * projection once per policy, outside Top-K capture. Compare selection
     * from the exact same resulting FP32 bytes without toggling producer policy.
     */
    RouterPolicyLogits recordOriginalRouterPolicyLogits()
    {
        constexpr int hidden = 2560;
        const DeviceId device = DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        RouterPolicyLogits observations;
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            requireRouterLogits(backend && stream, "policy logits exact context stream");
            auto activations = TestTensorFactory::createFP32({replay_rows, hidden});
            auto fp32_gate = TestTensorFactory::createFP32({replay_experts, hidden});
            auto fp16_gate = TestTensorFactory::createFP16({replay_experts, hidden});
            auto bf16_gate = TestTensorFactory::createBF16({replay_experts, hidden});
            auto q8_gate = TestTensorFactory::createINT32({replay_experts * hidden / sizeof(int32_t)});
            auto gate_scales = TestTensorFactory::createFP32({replay_experts, hidden / 32});
            auto hidden_codes = TestTensorFactory::createINT32({replay_rows * hidden / sizeof(int32_t)});
            auto hidden_scales = TestTensorFactory::createFP32({replay_rows, hidden / 32});
            auto logits = TestTensorFactory::createFP32({replay_rows, replay_experts});
            RouterObservationJoin join{*backend, device, stream};

            for (size_t i = 0; i < activations->numel(); ++i)
                activations->mutable_data()[i] = float(int((i * 71 + i / hidden) % 229) - 114) / 257.f;
            auto *gate_codes = reinterpret_cast<int8_t *>(q8_gate->mutable_int32_data());
            auto *gate_weights = fp32_gate->mutable_data();
            auto *half_weights = fp16_gate->mutable_typed_data();
            auto *bfloat_weights = bf16_gate->mutable_typed_data();
            for (size_t i = 0; i < fp32_gate->numel(); ++i)
            {
                const int code = int((i * 11 + i / hidden * 43 + i / 32) % 249) - 124;
                const float value = float(code) / 4093.f + float(int(i % 11) - 5) / 65521.f;
                gate_codes[i] = static_cast<int8_t>(code);
                gate_weights[i] = value;
                half_weights[i] = fp32_to_fp16(value);
                bfloat_weights[i] = simd::fp32_to_bf16(value);
            }
            std::fill_n(gate_scales->mutable_data(), gate_scales->numel(), 1.0f / 4096.f);
            for (auto *source : std::array<TensorBase *, 6>{activations.get(), fp32_gate.get(), fp16_gate.get(),
                                                       bf16_gate.get(), q8_gate.get(), gate_scales.get()})
                requireRouterLogits(source->ensureOnDevice(device, stream), "admit original router policy fixture operands");
            requireRouterLogits(hidden_codes->allocateOnDevice(device, stream) &&
                hidden_scales->allocateOnDevice(device, stream) && logits->allocateOnDevice(device, stream),
                "admit original router policy fixture workspace");
            const auto *input = static_cast<const float *>(activations->gpu_data_ptr());
            auto *output = static_cast<float *>(logits->gpu_data_ptr());
            for (size_t policy = 0; policy < observations.size(); ++policy)
            {
                bool enqueued = false;
                switch (policy)
                {
                    case 0: enqueued = hipMoE_gate_logits_fp32_decode_equivalent_rows(input,
                        static_cast<const float *>(fp32_gate->gpu_data_ptr()), output, replay_rows, hidden, replay_experts,
                        device.ordinal, stream, nullptr); break;
                    case 1: enqueued = hipMoE_gate_logits_fp16_decode_equivalent_rows(input,
                        fp16_gate->gpu_data_ptr(), output, replay_rows, hidden, replay_experts,
                        device.ordinal, stream, nullptr); break;
                    case 2: enqueued = hipMoE_gate_logits_bf16_decode_equivalent_rows(input,
                        bf16_gate->gpu_data_ptr(), output, replay_rows, hidden, replay_experts,
                        device.ordinal, stream, nullptr); break;
                    case 3: enqueued = hipMoE_gate_logits_q8_weights_decode_equivalent_rows(input,
                        static_cast<int8_t *>(hidden_codes->gpu_data_ptr()),
                        static_cast<float *>(hidden_scales->gpu_data_ptr()),
                        static_cast<const int8_t *>(q8_gate->gpu_data_ptr()),
                        static_cast<const float *>(gate_scales->gpu_data_ptr()), output, replay_rows, hidden, replay_experts,
                        device.ordinal, stream); break;
                }
                auto &observation = observations[policy];
                observation.resize(logits->numel());
                requireRouterLogits(enqueued && backend->deviceToHost(observation.data(), output,
                    observation.size() * sizeof(float), device.ordinal, stream) &&
                    backend->synchronizeStream(stream, device.ordinal), "observe production router policy logits");
                requireRouterLogits(std::all_of(observation.begin(), observation.end(),
                    [](float value) { return std::isfinite(value); }), "expected finite recorded policy logits");
            }
        });
        return observations;
    }

    /** @brief Original logits and both original routed publications for one oracle or candidate. */
    struct TopKReplayPlanes
    {
        std::unique_ptr<FP32Tensor> logits, ids, weights;
        std::vector<float> received_logits, received_ids, received_weights;
        /** @brief Allocate exactly one extra guard row, which must never be published. */
        TopKReplayPlanes(int rows, int experts, int top_k)
            : logits(TestTensorFactory::createFP32({size_t(rows + 1) * experts})),
              ids(TestTensorFactory::createFP32({size_t(rows + 1) * top_k})),
              weights(TestTensorFactory::createFP32({size_t(rows + 1) * top_k})),
              received_logits(logits->numel()), received_ids(ids->numel()), received_weights(weights->numel())
        {}
    };

    /**
     * @brief Capture the production counted selection and a separate diagnostic serial oracle.
     * @param rows Immutable graph capacity, including out-of-scope M1 and M9 controls.
     * @param counted Supply device-owned capacity changes, or exercise an original uncounted graph.
     * @param recorded Exact FP32 results from the native router policies; generated inputs cover adversarial Top-K.
     */
    void proveGroupedRegisterTopK(int rows, bool counted, const RouterPolicyLogits &recorded)
    {
        constexpr int experts = replay_experts, top_k = 10;
        constexpr uint32_t dirty_bits = 0x7fc01234u;
        constexpr int dirty_route = -93;
        const float dirty_value = std::bit_cast<float>(dirty_bits);
        const DeviceId device = DeviceId::rocm(0);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            requireRouterLogits(backend && stream, "Top-K replay exact context stream");
            const auto &hardware = rocmHardwareMetadata().at(device.ordinal);
            const bool candidate = qfnGroupedTopKRegisterOptIn() && counted && rows >= 2 && rows <= 8 &&
                hardware.compute_units == 60 && hardware.native_wave_width == 64 &&
                (hardware.isa_target == "gfx906" || hardware.isa_target.starts_with("gfx906:"));
            TopKReplayPlanes grouped(rows, experts, top_k), serial(rows, experts, top_k);
            for (auto *plane : {grouped.logits.get(), grouped.ids.get(), grouped.weights.get(),
                                serial.logits.get(), serial.ids.get(), serial.weights.get()})
                requireRouterLogits(plane->allocateOnDevice(device, stream), "allocate independent Top-K planes");
            auto grouped_count = TestTensorFactory::createINT32({1});
            auto serial_counts = TestTensorFactory::createINT32({size_t(rows)});
            requireRouterLogits(grouped_count->allocateOnDevice(device, stream) &&
                serial_counts->allocateOnDevice(device, stream), "allocate exact device count owners");

            const DeviceMoERuntimeTable::Config config{.device_id = device, .num_layers = 1, .num_experts = experts,
                .top_k = top_k, .mirror_to_device = true, .deferred_verifier_token_capacity = rows + 1};
            DeviceMoERuntimeTable grouped_ledger(config);
            auto serial_config = config;
            serial_config.num_layers = rows;
            serial_config.deferred_verifier_token_capacity = 1;
            DeviceMoERuntimeTable serial_ledger(serial_config);

            const auto getFloat = [](const auto &tensor) { return static_cast<float *>(tensor->gpu_data_ptr()); };
            const auto getInt = [](const auto &tensor) { return static_cast<int *>(tensor->gpu_data_ptr()); };
            const auto launch = [&](TopKReplayPlanes &out, int capacity, int first, bool normalize,
                                    const int *live, DeviceMoELayerRuntime *ledger) {
                return hipMoE_softmax_topk_decode_equivalent_rows(
                    getFloat(out.logits) + size_t(first) * experts, getFloat(out.ids) + size_t(first) * top_k,
                    getFloat(out.weights) + size_t(first) * top_k, capacity, experts, top_k, normalize,
                    device.ordinal, stream, live, ledger);
            };
            std::vector<float> input(grouped.logits->numel()), dirty_output(grouped.ids->numel(), dirty_value);
            std::vector<int> dirty_ledger(grouped.ids->numel(), dirty_route), row_counts(size_t(rows), 0);
            std::array<std::vector<int>, 2> received_group{dirty_ledger, dirty_ledger};
            std::array<std::vector<int>, 2> received_serial{dirty_ledger, dirty_ledger};

            for (bool deferred : {false, true})
                for (bool normalize : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << "M=" << rows << " counted=" << counted
                        << " normalize=" << normalize << " deferred=" << deferred);
                    auto grouped_graph = context.createGraphCapture(stream);
                    auto serial_graph = context.createGraphCapture(stream);
                    requireRouterLogits(grouped_graph && serial_graph, "allocate original captured Top-K executables");
                    RouterObservationJoin join{*backend, device, stream};
                    {
                        ScopedBackendGraphCapture recording(context, *grouped_graph, "counted QFN Top-K replay");
                        requireRouterLogits(recording.begin() &&
                            launch(grouped, rows, 0, normalize, counted ? getInt(grouped_count) : nullptr,
                                   deferred ? grouped_ledger.deviceLayerState(0) : nullptr),
                            "record grouped production selector");
                        recording.finish();
                    }
                    {
                        ScopedBackendGraphCapture recording(context, *serial_graph, "serial QFN Top-K oracle");
                        requireRouterLogits(recording.begin(), "record serial Top-K oracle");
                        for (int row = 0; row < rows; ++row)
                            requireRouterLogits(launch(serial, 1, row, normalize, counted ? getInt(serial_counts) + row : nullptr,
                                deferred ? serial_ledger.deviceLayerState(row) : nullptr), "record original serial row");
                        recording.finish();
                    }
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    ASSERT_TRUE(grouped_graph->inspectKernelNodes(nodes));
                    ASSERT_EQ(grouped_graph->nodeCount(), 1u);
                    ASSERT_EQ(serial_graph->nodeCount(), size_t(rows));
                    ASSERT_EQ(nodes.size(), 1u);
                    ASSERT_TRUE(nodes[0].name_resolved);
                    EXPECT_NE(nodes[0].name.find(candidate ? "qfn_grouped_softmax_register_selector_kernel" :
                        "rocm_moe_softmax_topk_decode_equivalent_rows_kernel"), std::string::npos);
                    EXPECT_EQ(nodes[0].grid_x, static_cast<unsigned>(rows));
                    EXPECT_EQ(nodes[0].block_x, 256u);
                    if (candidate) EXPECT_EQ(nodes[0].local_memory_bytes_per_thread, 0u);
                    requireRouterLogits(grouped_graph->instantiate() && serial_graph->instantiate(), "instantiate exact Top-K captures");

                    const auto upload = [&](void *to, const auto *from, size_t count) {
                        requireRouterLogits(backend->hostToDevice(to, from, count * sizeof(*from), device.ordinal, stream),
                            "upload original Top-K replay input");
                    };
                    const auto download = [&](auto *to, const void *from, size_t count) {
                        requireRouterLogits(backend->deviceToHost(to, from, count * sizeof(*to), device.ordinal, stream),
                            "observe original Top-K replay output");
                    };
                    const auto ledgerCopies = [&](const DeviceMoERuntimeTable &table, bool is_grouped, auto &&copy) {
                        const int layer_count = is_grouped ? 1 : rows;
                        const size_t count = is_grouped ? dirty_ledger.size() : top_k;
                        for (int layer = 0; layer < layer_count; ++layer)
                        {
                            const auto &state = table.hostLayerState(layer);
                            const std::array pointers{state.deferred_verifier_route_expert_ids,
                                state.deferred_verifier_route_participant_ids};
                            for (size_t plane = 0; plane < pointers.size(); ++plane)
                                copy(pointers[plane], plane, size_t(layer) * top_k, count);
                        }
                    };
                    const auto restore = [&](TopKReplayPlanes &planes, const auto &ledger, bool is_grouped) {
                        upload(getFloat(planes.logits), input.data(), input.size());
                        upload(getFloat(planes.ids), dirty_output.data(), dirty_output.size());
                        upload(getFloat(planes.weights), dirty_output.data(), dirty_output.size());
                        ledgerCopies(ledger, is_grouped, [&](int *pointer, size_t, size_t offset, size_t count) {
                            upload(pointer, dirty_ledger.data() + offset, count);
                        });
                    };
                    const auto observe = [&](TopKReplayPlanes &planes, const auto &ledger, bool is_grouped, auto &metadata) {
                        download(planes.received_logits.data(), getFloat(planes.logits), planes.received_logits.size());
                        download(planes.received_ids.data(), getFloat(planes.ids), planes.received_ids.size());
                        download(planes.received_weights.data(), getFloat(planes.weights), planes.received_weights.size());
                        ledgerCopies(ledger, is_grouped, [&](int *pointer, size_t plane, size_t offset, size_t count) {
                            download(metadata[plane].data() + offset, pointer, count);
                        });
                    };
                    // The tested donor's adversarial cases and native policy
                    // recordings both retain their exact input bytes over replay.
                    constexpr int adversarial_patterns = 10;
                    for (int pattern = 0; pattern < adversarial_patterns + int(recorded.size()); ++pattern)
                    {
                        if (pattern >= adversarial_patterns)
                            std::copy_n(recorded[pattern - adversarial_patterns].begin(), size_t(rows) * experts, input.begin());
                        else for (int row = 0; row < rows; ++row)
                            for (int expert = 0; expert < experts; ++expert)
                            {
                                float value = float((expert * 321 + row * 173) % 557 - 278) / 64.f;
                                const float nan = std::numeric_limits<float>::quiet_NaN();
                                switch (pattern)
                                {
                                    case 1: value = float(expert % 7); break;
                                    case 2: value = expert == 129 ? 500.f : -500.f; break;
                                    case 3: if (expert % 4 == 0) value = nan; break;
                                    case 4: value = nan; break;
                                    case 5: if (expert == 1) value = std::numeric_limits<float>::infinity(); break;
                                    case 6: if (expert > 1) value = nan; break;
                                    case 7: value = expert % 2 ? -0.f : 0.f; break;
                                    // Every 64-expert wave has candidates tied at the global cutoff.
                                    case 8: value = ((expert + row) % 64 < 3) ? 1.0f : -1.0f; break;
                                    // Put the ten winners in one 16-entry local frontier; the
                                    // next group has adjacent FP32 logits and competes at rank ten.
                                    case 9: value = expert >= 448 && expert < 458 ? 4.0f :
                                                    (expert >= 256 && expert < 274 ?
                                                        std::nextafter(4.0f, 0.0f) : 0.0f); break;
                                }
                                input[size_t(row) * experts + expert] = value;
                            }
                        std::fill_n(input.end() - experts, experts, dirty_value);
                        const std::vector<int> live_counts = counted
                            ? std::vector<int>{rows, 0, 1, rows - 1, rows + 2, -3, rows}
                            : std::vector<int>{rows, rows};
                        for (int live : live_counts)
                        {
                            SCOPED_TRACE(::testing::Message() << "pattern=" << pattern << " live=" << live);
                            for (int row = 0; row < rows; ++row)
                                row_counts[row] = row < std::clamp(live, 0, rows);
                            // M1 accepts a single-device Boolean count; oversize and
                            // negative cardinalities are exercised only on grouped M.
                            if (rows == 1 && (live < 0 || live > 1)) continue;
                            upload(getInt(grouped_count), &live, 1);
                            upload(getInt(serial_counts), row_counts.data(), row_counts.size());
                            restore(grouped, grouped_ledger, true);
                            restore(serial, serial_ledger, false);
                            requireRouterLogits(grouped_graph->launch() && serial_graph->launch(), "launch retained Top-K replays");
                            observe(grouped, grouped_ledger, true, received_group);
                            observe(serial, serial_ledger, false, received_serial);
                            requireRouterLogits(backend->synchronizeStream(stream, device.ordinal), "join test observation");
                            const auto byteEqual = [](const auto &a, const auto &b) {
                                return std::equal(a.begin(), a.end(), b.begin(),
                                    [](float x, float y) { return std::bit_cast<uint32_t>(x) == std::bit_cast<uint32_t>(y); });
                            };
                            ASSERT_TRUE(byteEqual(grouped.received_logits, serial.received_logits));
                            ASSERT_TRUE(byteEqual(grouped.received_ids, serial.received_ids));
                            ASSERT_TRUE(byteEqual(grouped.received_weights, serial.received_weights));
                            ASSERT_EQ(received_group, received_serial);
                            ASSERT_EQ(std::bit_cast<uint32_t>(grouped.received_logits.back()), dirty_bits);
                            ASSERT_EQ(std::bit_cast<uint32_t>(grouped.received_ids.back()), dirty_bits);
                            ASSERT_EQ(std::bit_cast<uint32_t>(grouped.received_weights.back()), dirty_bits);
                            ASSERT_EQ(received_group[0].back(), dirty_route);
                            ASSERT_EQ(received_group[1].back(), dirty_route);
                            if (!deferred)
                                ASSERT_EQ(received_group, (std::array{dirty_ledger, dirty_ledger}));
                            if (live > 0 && pattern == 7)
                                for (int slot = 0; slot < top_k; ++slot)
                                    EXPECT_EQ(grouped.received_ids[slot], float(slot));
                        }
                    }
                }
        });
    }
}

/** @test The installed opt-in records genuine MI50 selector nodes and matches the original serial bytes. */
TEST(MoERouterLogits, ROCmGroupedRegisterTopKCapturedByteReplay)
{
    ASSERT_TRUE(qfnGroupedTopKRegisterOptIn()) << "Focused replay requires the registered native selector";
    const auto recorded = recordOriginalRouterPolicyLogits();
    for (int rows = 1; rows <= replay_rows; ++rows)
        proveGroupedRegisterTopK(rows, true, recorded);
    proveGroupedRegisterTopK(4, false, recorded);
}
} // namespace llaminar2::test
