/**
 * @file Test__MoEProjectionPhases.cpp
 * @brief Captured phase/column equivalence through the public GPU MoE interface.
 *
 * The ordinary complete pipeline is the oracle already covered by serial-row
 * gates. This test prepares real native weights, including independently packed
 * source-column slices, then compares complete and split transactions byte for
 * byte. No host row replay, format conversion, transport shortcut or alternate
 * device arithmetic occurs in the candidate graph. Host copies and joins are
 * fixture admission/observation only, outside capture.
 * Compact-packet cases deliberately give private and runtime grouping different
 * live counts, then destroy intermediate compute storage before import. They
 * therefore prove count authority and reconstruction, not an accidental no-op
 * round trip. Full/empty/sparse replay also checks exact byte extents and guards.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** @brief Select one real backend, never skip an advertised execution path. */
    class MoEProjectionPhases : public ::testing::TestWithParam<std::string> {};

    /** @brief Choose direct phase equivalence or the compact production-kernel boundary. */
    enum class IntermediateProof { DirectWorkspace, CompactPacket };

    /** @brief Source factory and exact weight descriptor family. */
    struct Format
    {
        std::string name;
        DeviceMoEWeightFormat family;
        QuantizedVerifierWeightCreator create;
    };

    /** @return The canonical codebook inventory plus each floating weight type. */
    std::vector<Format> formats()
    {
        std::vector<Format> result;
        for (const auto &format : quantizedMoEVerifierFormats())
            result.push_back({format.label, DeviceMoEWeightFormat::NativeVNNI, format.create});
        result.push_back({"FP16", DeviceMoEWeightFormat::FP16,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createFP16Random(shape, -0.125f, 0.125f, seed); }});
        result.push_back({"BF16", DeviceMoEWeightFormat::BF16,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createBF16Random(shape, -0.125f, 0.125f, seed); }});
        result.push_back({"FP32", DeviceMoEWeightFormat::FP32,
            [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
            { return TestTensorFactory::createFP32Random(shape, -0.125f, 0.125f, seed); }});
        return result;
    }

    /** @brief Reject a fixture operation; the enclosing join retains in-flight storage. */
    void require(bool value, const char *operation)
    {
        if (!value) throw std::runtime_error(operation);
    }

    /** @brief Join fixture work before the graph, workspace and prepared weights retire. */
    struct ObservationJoin
    {
        IBackend *backend;
        DeviceId device;
        void *stream;
        ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
    };

    /** @brief Retain physical slices and original source intervals until graph retirement. */
    struct Slice
    {
        int first;
        int width;
        int descriptor_table;
        std::unique_ptr<FP32Tensor> output;
        std::unique_ptr<FP32Tensor> canonical_output;
        std::unique_ptr<FP32Tensor> contributions;
    };

    /**
     * @brief Verify all grouped row widths and TP degrees on one prepared format.
     * @param device Actual endpoint; all launches use its context's exact stream.
     * @param format Native source format, never substituted or dequantized.
     * @param tails Exercise non-block-aligned floating N/K and uneven intervals.
     * @param proof Compact cases additionally reconstruct destroyed intermediates from an exact live packet.
     */
    void prove(DeviceId device, const Format &format, bool tails = false,
        IntermediateProof proof = IntermediateProof::DirectWorkspace)
    {
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&]
        {
            constexpr int capacity = 65, experts = 3, top_k = 2, guard = 16;
            const int d_model = tails ? 271 : 256;
            const int intermediate = tails ? 259 : 256;
            const bool floating = deviceMoEWeightFormatIsFloating(format.family);
            const bool compact = proof == IntermediateProof::CompactPacket;
            auto *backend = getBackendFor(device);
            auto *stream = context.defaultStream();
            require(backend && stream, "projection fixture needs a real explicit-stream backend");

            std::vector<std::shared_ptr<TensorBase>> sources;
            std::vector<GpuPreparedGemm> prepared;
            const auto requirements = device.is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(capacity, d_model, intermediate, experts, top_k)
                : MoEWorkspaceBuffers::rocmMoE(capacity, d_model, intermediate, experts, top_k);
            DeviceWorkspaceManager workspace(device,
                requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
            require(workspace.allocate(requirements), "projection workspace admission");
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            require(kernel != nullptr, "projection kernel factory");
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            require(consumer != nullptr, "projection workspace consumer");
            consumer->bindWorkspace(&workspace);

            auto tensor = [&](size_t count)
            {
                auto value = TestTensorFactory::createFP32({count});
                std::fill_n(value->mutable_data(), count, 0.0f);
                require(value->ensureOnDevice(device, stream), "projection tensor admission");
                return value;
            };
            auto publish = [&](ITensor *target, const std::vector<float> &values)
            {
                require(backend->hostToDevice(target->gpu_data_ptr(), values.data(),
                    values.size() * sizeof(float), device.ordinal, stream), "projection fixture publication");
            };
            auto observe = [&](ITensor *target)
            {
                std::vector<float> values(target->numel());
                require(backend->deviceToHost(values.data(), target->gpu_data_ptr(),
                    values.size() * sizeof(float), device.ordinal, stream) &&
                    backend->synchronizeStream(stream, device.ordinal), "projection fixture observation");
                return values;
            };

            std::vector<DeviceNativeVNNIMatrixDesc> gates(experts), ups(experts), downs(experts);
            std::vector<DeviceMoEFloatingMatrixDesc> fp_gates(experts), fp_ups(experts), fp_downs(experts);
            auto prepare = [&](std::shared_ptr<TensorBase> source,
                               DeviceNativeVNNIMatrixDesc &native, DeviceMoEFloatingMatrixDesc &fp)
            {
                const std::string name = "test.projection_phases." + format.name + "." + std::to_string(prepared.size());
                sources.push_back(std::move(source));
                prepared.push_back(floating
                    ? makeGpuPreparedFloatingPointGemm(sources.back().get(), device, name)
                    : makeGpuPreparedGemm(sources.back().get(), device, name));
                if (floating)
                {
                    ContiguousFloatingPointWeightDescriptor source_descriptor;
                    DeviceMoEWeightFormat exported_family;
                    require(prepared.back().kernel->exportContiguousFloatingPointWeights(source_descriptor) &&
                        exportDeviceMoEFloatingMatrixDescriptor(source_descriptor, fp, exported_family) &&
                        exported_family == format.family, "floating projection descriptor export");
                }
                else
                    require(prepared.back().kernel->exportNativeVNNIMatrixDesc(native), "projection descriptor export");
            };
            std::vector<std::shared_ptr<TensorBase>> down_sources;
            for (int expert = 0; expert < experts; ++expert)
            {
                prepare(format.create({size_t(intermediate), size_t(d_model)}, 18200 + expert), gates[expert], fp_gates[expert]);
                prepare(format.create({size_t(intermediate), size_t(d_model)}, 18300 + expert), ups[expert], fp_ups[expert]);
                down_sources.push_back(format.create({size_t(d_model), size_t(intermediate)}, 18400 + expert));
                prepare(down_sources.back(), downs[expert], fp_downs[expert]);
            }
            const int gate_table = floating
                ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(fp_gates.data(), fp_ups.data(),
                    format.family, experts, d_model, intermediate)
                : kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), experts, d_model, intermediate);
            const int down_table = floating
                ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(fp_downs.data(), format.family, experts, d_model, intermediate)
                : kernel->uploadGroupedExpertDownDescriptorTable(downs.data(), experts, d_model, intermediate);
            require(gate_table >= 0 && down_table >= 0, "projection descriptor publication");
            DeviceMoERuntimeTable::Config runtime_config;
            runtime_config.device_id = device;
            runtime_config.num_layers = 1;
            runtime_config.num_experts = experts;
            runtime_config.top_k = top_k;
            runtime_config.mirror_to_device = true;
            runtime_config.prefill_token_capacity = capacity;
            DeviceMoERuntimeTable runtime(runtime_config);
            MoEPlacementUpdate placement;
            placement.epoch = 1;
            placement.expert_count = experts;
            placement.participant_id = 0;
            placement.participant_count = 1;
            placement.experts.resize(experts);
            placement.local_compute_mask.assign(experts, 1u);
            placement.replica_role.assign(experts, static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
            for (int expert = 0; expert < experts; ++expert)
            {
                auto &descriptor = placement.experts[expert];
                descriptor.logical_expert_id = expert;
                descriptor.owner_participant = 0;
                descriptor.local_slot = expert;
                descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
                descriptor.weight_format = format.family;
                descriptor.gate = gates[expert];
                descriptor.up = ups[expert];
                descriptor.down = downs[expert];
                descriptor.floating_gate = fp_gates[expert];
                descriptor.floating_up = fp_ups[expert];
                descriptor.floating_down = fp_downs[expert];
            }
            require(runtime.prepareInactiveBank(0, placement) &&
                runtime.flipActiveBank(0, placement.epoch, stream), "projection runtime publication");
            const auto runtime_addresses = runtime.hostLayerState(0);
            auto hidden = tensor(capacity * d_model);
            auto indices = tensor(capacity * top_k);
            auto weights = tensor(capacity * top_k);
            auto complete = tensor(capacity * d_model + guard);
            auto runtime_complete = tensor(capacity * d_model + guard);
            auto runtime_split = tensor(capacity * d_model + guard);
            const MoEGroupedIntermediateLayout packet_capacity{
                floating ? MoEGroupedIntermediateEncoding::FP32 : MoEGroupedIntermediateEncoding::BlockQ8FP32Scales,
                static_cast<std::uint32_t>(intermediate), capacity * top_k, 1};
            auto compact_packet = compact ? tensor(packet_capacity.compactCapacityBytes() / sizeof(float) + guard) : nullptr;
            auto compact_count = compact ? tensor(2 + guard) : nullptr;
            auto empty_indices = compact ? tensor(capacity * top_k) : nullptr;
            auto empty_weights = compact ? tensor(capacity * top_k) : nullptr;
            auto zero_hidden = compact ? tensor(capacity * d_model) : nullptr;
            if (compact)
            {
                const std::vector<float> empty(capacity * top_k, -1.f);
                publish(empty_indices.get(), empty);
                // Keep the setup upload alive until its exact stream consumes it.
                require(backend->synchronizeStream(stream, device.ordinal), "compact empty-input preparation");
            }
            std::vector<Slice> slices;
            std::vector<std::pair<int, int>> intervals;
            if (compact)
                intervals = {{0, d_model}};
            else if (tails)
                intervals = {{0, 1}, {1, 17}, {18, 117}, {135, 136}};
            else
                for (const int degree : {1, 2, 4, 8})
                    for (int participant = 0; participant < degree; ++participant)
                        intervals.emplace_back(participant * (d_model / degree), d_model / degree);
            for (const auto &[first, width] : intervals)
            {
                for (int expert = 0; expert < experts; ++expert)
                    prepare(down_sources[expert]->create_view({size_t(width), size_t(intermediate)},
                        size_t(first) * intermediate), downs[expert], fp_downs[expert]);
                const int table = floating
                    ? kernel->uploadGroupedExpertFloatingDownDescriptorTable(fp_downs.data(), format.family, experts, width, intermediate)
                    : kernel->uploadGroupedExpertDownDescriptorTable(downs.data(), experts, width, intermediate);
                require(table >= 0, "projection slice descriptor publication");
                slices.push_back({first, width, table, tensor(capacity * width + guard),
                    tensor(capacity * width + guard), tensor(capacity * top_k * width + guard)});
            }

            // Reject contradictory source geometry before any device work.
            // These checks must not treat a local down width as a new model.
            EXPECT_FALSE(kernel->executeGroupedPrefillPipeline(hidden.get(), complete.get(), gate_table,
                down_table, 1, 0, intermediate, experts, top_k));
            EXPECT_FALSE(kernel->executeGroupedPrefillProjection(hidden.get(), complete.get(), gate_table,
                down_table, std::numeric_limits<int>::max(), d_model, intermediate, experts, top_k,
                MoEPrefillProjectionExecution::complete(d_model)));
            EXPECT_FALSE(kernel->executeGroupedPrefillProjectionFromPublishedRuntimePlan(
                runtime.deviceLayerState(0), runtime_addresses, nullptr, complete.get(), gate_table,
                down_table, 1, d_model, intermediate, experts, top_k,
                MoEPrefillProjectionExecution::down(d_model, 1, d_model - 1)));

            // The complete and all sliced graphs reuse one admitted workspace,
            // strictly sequentially. They never own concurrent writes to it.
            for (const int rows : {1, 2, 3, 15, 16, 17, 33, capacity})
            {
                SCOPED_TRACE(::testing::Message() << "format=" << format.name << " rows=" << rows);
                auto layout = packet_capacity;
                layout.route_capacity = rows * top_k;
                if (compact)
                {
                    // A packet is a bounded subrange of one arena/coherence
                    // owner, not a new tensor. Reject misalignment, a missing
                    // tail and overflow before either backend launches work.
                    for (const auto offset : {std::size_t{1}, compact_packet->size_bytes(),
                            compact_packet->size_bytes() + sizeof(std::uint32_t),
                            std::numeric_limits<std::size_t>::max() - 3})
                        EXPECT_FALSE(kernel->importCompactGroupedPrefillIntermediates(layout,
                            runtime_addresses.route_participant_ids, compact_packet.get(), offset,
                            static_cast<const std::uint64_t *>(compact_count->gpu_data_ptr()), 0));
                }
                const auto compact_round_trip = [&]
                {
                    if (!compact) return true;
                    auto *count = static_cast<std::uint64_t *>(compact_count->gpu_data_ptr());
                    // No count is downloaded to choose an operation. The same
                    // captured pack/import nodes handle every live row count.
                    return kernel->exportCompactGroupedPrefillIntermediates(layout, runtime_addresses,
                            compact_packet.get(), count) &&
                        kernel->prepareExpertGroupsAsync(indices.get(), weights.get(), rows, experts, top_k) &&
                        kernel->executeGroupedPrefillProjection(zero_hidden.get(), nullptr, gate_table, down_table,
                            rows, d_model, intermediate, experts, top_k, MoEPrefillProjectionExecution::gateUp(d_model)) &&
                        kernel->importCompactGroupedPrefillIntermediates(layout, runtime_addresses.route_participant_ids,
                            compact_packet.get(), 0, count, 0);
                };
                const auto split_down = [&]
                {
                    if (compact)
                    {
                        // Import targets the consumer's private grouping, as
                        // MoEProjectionPipeline::ImportDown does in production.
                        // Its grouped weights/order need not equal the runtime
                        // producer's order (ROCm diverges above one route tile).
                        // Never pair the private inverse map with runtime weights.
                        return kernel->executeGroupedPrefillProjection(nullptr, runtime_split.get(), gate_table,
                            down_table, rows, d_model, intermediate, experts, top_k,
                            MoEPrefillProjectionExecution::down(d_model, 0, d_model));
                    }
                    return kernel->executeGroupedPrefillProjectionFromPublishedRuntimePlan(runtime.deviceLayerState(0),
                        runtime_addresses, nullptr, runtime_split.get(), gate_table, down_table,
                        rows, d_model, intermediate, experts, top_k, MoEPrefillProjectionExecution::down(d_model, 0, d_model));
                };
                const auto enqueue = [&]
                {
                    if (!kernel->prepareExpertGroupsAsync(indices.get(), weights.get(), rows, experts, top_k) ||
                        !kernel->executeGroupedPrefillPipeline(hidden.get(), complete.get(), gate_table, down_table,
                            rows, d_model, intermediate, experts, top_k) ||
                        // The stale private counts are deliberately all zero;
                        // published-runtime execution must use runtime counts.
                        (compact && !kernel->prepareExpertGroupsAsync(empty_indices.get(), empty_weights.get(),
                            rows, experts, top_k)) ||
                        !kernel->publishCompleteGroupedPrefillPlanFromRouter(runtime.deviceLayerState(0),
                            indices.get(), weights.get(), rows, rows, experts, top_k, gate_table, down_table, false) ||
                        !kernel->executeGroupedPrefillPipelineFromPublishedRuntimePlan(runtime.deviceLayerState(0),
                            runtime_addresses, hidden.get(), runtime_complete.get(), gate_table, down_table,
                            rows, d_model, intermediate, experts, top_k) ||
                        !kernel->executeGroupedPrefillProjectionFromPublishedRuntimePlan(runtime.deviceLayerState(0),
                            runtime_addresses, hidden.get(), nullptr, gate_table, down_table,
                            rows, d_model, intermediate, experts, top_k, MoEPrefillProjectionExecution::gateUp(d_model)) ||
                        !compact_round_trip() ||
                        !split_down() ||
                        !kernel->prepareExpertGroupsAsync(indices.get(), weights.get(), rows, experts, top_k) ||
                        !kernel->executeGroupedPrefillProjection(hidden.get(), nullptr, gate_table, down_table,
                            rows, d_model, intermediate, experts, top_k, MoEPrefillProjectionExecution::gateUp(d_model)))
                        return false;
                    for (const auto &slice : slices)
                        if (!kernel->executeGroupedPrefillProjection(nullptr, slice.output.get(), gate_table,
                            slice.descriptor_table, rows, d_model, intermediate, experts, top_k,
                            MoEPrefillProjectionExecution::down(d_model, slice.first, slice.width)) ||
                            !kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table,
                            slice.descriptor_table, rows, d_model, intermediate, experts, top_k,
                            MoEPrefillProjectionExecution::down(d_model, slice.first, slice.width), slice.contributions.get()) ||
                            !kernel->reduceCanonicalRouteContributions(slice.contributions.get(), slice.canonical_output.get(),
                                rows, top_k, slice.width))
                            return false;
                    return true;
                };
                std::vector<float> h(capacity * d_model, 0.125f), ids(capacity * top_k, 0.0f), probabilities(capacity * top_k, 0.5f);
                for (size_t slot = 0; slot < ids.size(); ++slot)
                    ids[slot] = static_cast<float>(slot % top_k);
                // Immutable upload storage outlives the join even if a replay
                // fails. Reusing it also avoids per-replay host allocation.
                const std::vector<float> poison(std::max<std::size_t>(capacity * top_k * d_model + guard,
                    compact ? compact_packet->numel() : 0), -1234.5f);
                auto graph = context.createGraphCapture(stream);
                require(graph != nullptr, "projection capture owner");
                // Even a failed warmup must retire submitted work before its
                // inputs or executable. This join is observation-only, never
                // an edge inside the captured production transaction.
                ObservationJoin join{backend, device, stream};
                publish(hidden.get(), h); publish(indices.get(), ids); publish(weights.get(), probabilities);
                require(enqueue() && backend->synchronizeStream(stream, device.ordinal), "projection workspace preparation");
                {
                    GraphCaptureGuard recording;
                    require(graph->beginCapture(), "projection begin capture");
                    require(enqueue(), "projection record phases");
                    require(graph->endCapture(), "projection end capture");
                }
                require(graph->instantiate(), "projection instantiate");
                for (int replay = 0; replay < 20; ++replay)
                {
                    SCOPED_TRACE(::testing::Message() << "replay=" << replay);
                    std::size_t live_routes = 0;
                    for (size_t i = 0; i < h.size(); ++i)
                        h[i] = 0.0071f * (int((i * 19 + replay * 13) % 101) - 50);
                    for (int row = 0; row < rows; ++row)
                        for (int route = 0; route < top_k; ++route)
                        {
                            const int slot = row * top_k + route;
                            // Full -> empty -> sparse -> full, with high physical
                            // row indices and changing expert/route identities.
                            const bool active = replay % 5 != 1 && (replay % 5 != 2 || row % 3 == 2);
                            ids[slot] = active ? float((row + route + replay) % experts) : -1.0f;
                            probabilities[slot] = active ? (route == 0 ? 0.625f : 0.375f) : 0.0f;
                            live_routes += active;
                        }
                    publish(hidden.get(), h); publish(indices.get(), ids); publish(weights.get(), probabilities);
                    const auto poison_tensor = [&](ITensor *target)
                    {
                        require(backend->hostToDevice(target->gpu_data_ptr(), poison.data(),
                            target->numel() * sizeof(float), device.ordinal, stream), "projection guard publication");
                    };
                    poison_tensor(complete.get());
                    poison_tensor(runtime_complete.get());
                    poison_tensor(runtime_split.get());
                    if (compact)
                    {
                        poison_tensor(compact_packet.get());
                        poison_tensor(compact_count.get());
                    }
                    for (const auto &slice : slices)
                    {
                        poison_tensor(slice.output.get());
                        poison_tensor(slice.canonical_output.get());
                        poison_tensor(slice.contributions.get());
                    }
                    require(graph->launch(), "projection retained replay");
                    if (compact)
                    {
                        const auto count_words = observe(compact_count.get());
                        std::uint64_t packet_bytes = 0;
                        std::memcpy(&packet_bytes, count_words.data(), sizeof(packet_bytes));
                        ASSERT_EQ(packet_bytes, live_routes * layout.compactRecordWords() * sizeof(std::uint32_t));
                        for (std::size_t i = 2; i < count_words.size(); ++i)
                            ASSERT_EQ(count_words[i], -1234.5f) << "compact count overrun at " << i;
                        const auto packet_words = observe(compact_packet.get());
                        for (std::size_t i = packet_bytes / sizeof(float); i < packet_words.size(); ++i)
                            ASSERT_EQ(packet_words[i], -1234.5f) << "compact export wrote unused capacity at " << i;
                    }
                    const auto expected = observe(complete.get());
                    for (ITensor *runtime_output : {runtime_complete.get(), runtime_split.get()})
                    {
                        const auto actual = observe(runtime_output);
                        for (size_t i = 0; i < actual.size(); ++i)
                            ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i]))
                                << (runtime_output == runtime_complete.get() ? "runtime complete" : "split/imported")
                                << " output offset=" << i;
                    }
                    for (size_t i = size_t(rows) * d_model; i < expected.size(); ++i)
                        ASSERT_EQ(expected[i], -1234.5f) << "complete pipeline wrote unused capacity at " << i;
                    for (const auto &slice : slices)
                    {
                        const auto actual = observe(slice.output.get());
                        const auto canonical = observe(slice.canonical_output.get());
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < slice.width; ++column)
                            {
                                const float ref = expected[row * d_model + slice.first + column];
                                ASSERT_TRUE(std::isfinite(ref));
                                ASSERT_EQ(std::bit_cast<uint32_t>(actual[row * slice.width + column]),
                                    std::bit_cast<uint32_t>(ref)) << "slice=" << slice.first << '+' << slice.width
                                    << " row=" << row << " column=" << column;
                                ASSERT_EQ(std::bit_cast<uint32_t>(canonical[row * slice.width + column]),
                                    std::bit_cast<uint32_t>(ref)) << "canonical slice=" << slice.first << '+' << slice.width;
                            }
                        for (size_t i = size_t(rows) * slice.width; i < actual.size(); ++i)
                        {
                            ASSERT_EQ(actual[i], -1234.5f) << "projection wrote unused capacity at " << i;
                            ASSERT_EQ(canonical[i], -1234.5f) << "canonical projection wrote unused capacity at " << i;
                        }
                        const auto routes = observe(slice.contributions.get());
                        for (size_t i = size_t(rows) * top_k * slice.width; i < routes.size(); ++i)
                            ASSERT_EQ(routes[i], -1234.5f) << "route projection wrote unused capacity at " << i;
                    }
                }
            }
        });
    }

    /** @brief Every quantized codebook plus FP16/BF16/FP32, P1/P2/P4/P8 and M1..65. */
    TEST_P(MoEProjectionPhases, CapturedAllFormats)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        for (const auto &format : formats()) prove(device, format);
    }

    /** @brief Floating intervals are not restricted to quantized block geometry. */
    TEST_P(MoEProjectionPhases, FloatingColumnTails)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        for (const auto &format : formats())
            if (deviceMoEWeightFormatIsFloating(format.family)) prove(device, format, true);
    }

    /** @brief Every codebook and FP16/BF16/FP32 keeps exact outputs and live extents after destructive regrouping. */
    TEST_P(MoEProjectionPhases, CapturedCompactAllFormats)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        for (const auto &format : formats()) prove(device, format, false, IntermediateProof::CompactPacket);
    }

    /** @brief Compact FP payloads preserve odd widths without adding a quantization alignment requirement. */
    TEST_P(MoEProjectionPhases, CompactFloatingColumnTails)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        for (const auto &format : formats())
            if (deviceMoEWeightFormatIsFloating(format.family)) prove(device, format, true, IntermediateProof::CompactPacket);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MoEProjectionPhases, ::testing::ValuesIn(std::vector<std::string>{
#ifdef HAVE_CUDA
        "CUDA",
#endif
#ifdef HAVE_ROCM
        "ROCm",
#endif
        }), [](const auto &info) { return info.param; });
} // namespace
