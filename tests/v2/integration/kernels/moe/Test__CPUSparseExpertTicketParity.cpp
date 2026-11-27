/**
 * @file Test__CPUSparseExpertTicketParity.cpp
 * @brief Numerical proof of retained CPU expert execution and captured GPU ingress.
 *
 * Moving experts changes the CPU endpoint's local route width even when the
 * model's top-k is unchanged. This test reuses the production stage, arena and
 * canonical return ticket across narrow/wide packets. Independent serial CPU
 * expert rows certify the result after the real captured CUDA/ROCm consumer;
 * synthetic ticket payloads alone cannot catch a compute-to-ticket stride bug.
 * The endpoint starts with an empty prepared bank. Its metadata declaration is
 * admitted and bound through the production non-graph workspace allocator,
 * then real prepared experts arrive without rebinding or growing that storage.
 * The fixture authors the same complete packet key as the actual sparse
 * dispatch producer; workspace views provide storage and never infer identity.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/CPUProjectionTestWorkspace.h"
#include "../../../utils/CPUSerialArenaTestAdmission.h"

#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

namespace llaminar2::test
{
namespace
{
/**
 * @brief Create floating expert weights with one shared storage rounding.
 * @param type Native floating storage format, independent of FP32 activations.
 * @param shape Exact projection matrix geometry.
 * @param seed Distinguishes experts and projections without unbounded values.
 * @return Owning tensor read by both the stage and serial numerical oracle.
 */
std::unique_ptr<TensorBase> floatingWeight(
    TensorType type, const std::vector<size_t> &shape, uint32_t seed)
{
    std::vector<float> values(shape[0] * shape[1]);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = 0.00213f * static_cast<float>(
            static_cast<int>((i * (2u * seed + 1u) + seed) % 97u) - 48);
    if (type == TensorType::FP16)
    {
        auto result = TestTensorFactory::createFP16(shape);
        result->from_fp32(values.data(), values.size());
        return result;
    }
    if (type == TensorType::BF16)
    {
        auto result = TestTensorFactory::createBF16(shape);
        result->from_fp32(values.data(), values.size());
        return result;
    }
    auto result = TestTensorFactory::createFP32(shape);
    std::copy(values.begin(), values.end(), result->mutable_data());
    return result;
}

/**
 * @brief Exercise all expert formats through one retained sparse endpoint.
 * @param device Exact CUDA or ROCm continuation consuming the CPU return ticket.
 *
 * Graph launches intentionally precede CPU publication. A guard publishes an
 * abort on a failed assertion, so the test cannot strand a device wait while
 * destroying its mapped storage. Only terminal test observation synchronizes.
 */
void runSparseExpertTicketParity(DeviceId device)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    constexpr int experts = 8;
    constexpr int max_rows = 15;
    constexpr int top_k = 8;
    constexpr size_t capacity = max_rows * top_k;
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), device.ordinal);
    void *stream = backend->createStream(device.ordinal);
    ASSERT_NE(stream, nullptr);
    auto release_stream = [&](void *value) { backend->destroyStream(value, device.ordinal); };
    std::unique_ptr<void, decltype(release_stream)> stream_owner(stream, release_stream);
    auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    ASSERT_NE(kernel, nullptr);
    CPUDeviceContext cpu(DeviceId::cpu());

    auto formats = quantizedMoEVerifierFormats();
    for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        formats.push_back({
            .label = type == TensorType::FP16 ? "FP16" :
                     type == TensorType::BF16 ? "BF16" : "FP32",
            .tensor_type = type,
            .source_codebook_id = 0,
            .source_is_superblock = false,
            .device_execution_codebook_id = 0,
            .create = [type](const std::vector<size_t> &shape, uint32_t seed) {
                return floatingWeight(type, shape, seed);
            }});

    for (const auto &format : formats)
    {
        SCOPED_TRACE(device.to_string() + " " + format.label);
        // Include the failing real model's two expert-storage geometries.
        // Other formats still exercise multiple logical reduction partitions.
        const bool model_geometry = format.tensor_type == TensorType::Q8_0 ||
                                    format.tensor_type == TensorType::BF16;
        const int width = model_geometry ? 3072 : 512;
        const int intermediate = model_geometry ? 1024 : 256;
        const bool floating = format.tensor_type == TensorType::FP16 ||
                              format.tensor_type == TensorType::BF16 ||
                              format.tensor_type == TensorType::FP32;
        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<std::shared_ptr<ITensorGemm>> engines;
        std::array<std::vector<ITensorGemm *>, 3> projections;
        for (int expert = 0; expert < experts; ++expert)
            for (int projection = 0; projection < 3; ++projection)
            {
                weights.push_back(format.create(
                    projection == 2 ? std::vector<size_t>{size_t(width), size_t(intermediate)}
                                    : std::vector<size_t>{size_t(intermediate), size_t(width)},
                    131u + expert * 17u + projection * 7u));
                if (floating)
                    engines.push_back(std::make_unique<gemm::FloatingPointGemmKernel>(
                        weights.back().get(),
                        gemm::FloatingPointGemmKernel::NumericalPolicy::GPUAlignedExpert));
                else
                    engines.push_back(std::make_unique<cpu::native_quantised::CPUNativeQuantisedGemmKernel>(
                        weights.back().get(), 0, -1,
                        CPUProjectionNumericalPolicy::GPUAlignedExpert));
                projections[projection].push_back(engines.back().get());
            }

        MoEOverlayCollectiveWorkspace packets({
            .max_rows = max_rows, .max_entries = capacity,
            .d_model = width, .top_k = top_k, .device = DeviceId::cpu(),
            .return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes});
        auto input = packets.localExpertInput(0, 0);
        auto output = packets.localExpertOutput(0, 0);
        // Workspace views are passive storage. The original dispatch caller
        // owns packet identity; the bound canonical ticket must see layer0,
        // never the default unassigned layer (-1). Keep the original main
        // decode arithmetic phase while advancing this fixture's exact steps.
        input.key = {.generation_id = 1, .step_id = 1,
            .key_namespace = MoEOverlayCollectiveNamespace::Main,
            .histogram_source = ExpertHistogramSource::DecodeToken,
            .mtp_depth = -1, .layer_idx = 0, .tier_idx = 0,
            .domain_id = 0, .participant_id = 1,
            .direction = MoEOverlayCollectiveDirection::Dispatch, .sequence = 1};
        ASSERT_TRUE(input.key.isValid());
        // Preserve the authored coherent families and grouped/canonical roots;
        // freeze their exact common layout before any backing is materialized.
        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            admitCPUSerialArenaForTest(MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(), .row_capacity = max_rows,
                .row_capacity_buckets = {1, 3, max_rows},
                .d_model = width, .routing_top_k = top_k,
                .cpu_canonical_route_storage = MoELocalExpertSerialBufferArena::
                    CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
                .cpu_canonical_route_gpu_consumer = device,
                .cpu_grouped_scratch_storage = MoELocalExpertSerialBufferArena::
                    CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
                .num_experts = experts, .expert_intermediate = intermediate,
                .logical_participant_id = 1,
                .debug_name = "integration.cpu_sparse_ticket"}));
        const std::array<DeviceId, 1> endpoints{device};
        auto metadata = TransferEngine::instance().createMappedHostArena(endpoints);
        auto ticket = std::make_shared<MoEOverlayCanonicalRouteReturnTicketStorage>();
        ticket->bindFixedCapacity(0, capacity, width, device, 1,
            arena->mappedCPUCanonicalRoutes(device), metadata);
        // A real bound ticket still refuses a foreign original packet layer
        // before touching payload or release-publishing any success. This
        // producer lease cancels cold; no captured consumer has been launched.
        {
            auto publication = ticket->arm(1u);
            ASSERT_TRUE(publication);
            auto foreign = input;
            foreign.residency_epoch = 1u;
            foreign.key.layer_idx = ticket->layerIndex() + 1;
            ticket->originalRouteSlotsHost()[0] = -91;
            ticket->compactRouteSlotsHost()[0] = -92;
            ticket->contributionRowsHost()[0] = -93.0f;
            EXPECT_FALSE(publishMoEOverlayCanonicalRouteRows(*ticket, publication,
                foreign, {}, MoEOverlayCanonicalRouteSourceLayout::compactSlots()));
            EXPECT_EQ(ticket->originalRouteSlotsHost()[0], -91);
            EXPECT_EQ(ticket->compactRouteSlotsHost()[0], -92);
            EXPECT_EQ(ticket->contributionRowsHost()[0], -93.0f);
            EXPECT_FALSE(ticket->publicationSucceededFor(1u));
        }
        EXPECT_FALSE(ticket->payloadReady());
        MoELocalExpertStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input_rows = &input;
        params.output_rows = &output;
        params.serial_compact_buffer_arena = arena;
        params.graph_row_capacity = max_rows;
        params.cpu_canonical_route_ticket_return =
            MoELocalExpertStage::CPUCanonicalRouteTicketReturnBinding{ticket};
        params.num_experts = experts;
        params.top_k = top_k;
        params.d_model = width;
        params.expert_intermediate = intermediate;
        params.layer_idx = 0;
        params.runtime_participant_index = 1;
        params.expert_mask.assign(experts, false);
        params.expert_weight_resolution_policy = MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
        params.cpu_workspace_source = MoELocalExpertStage::CPUExpertWorkspaceSource{
            .gate = format.tensor_type, .up = format.tensor_type, .down = format.tensor_type,
            .workers = cpu::native_quantised::nativeQuantisedInvocationWorkerCount(),
            .execution = CPUExecutionGeometry::local()};
        auto residency = std::make_shared<MoEOverlayParticipantResidency>(
            MoEOverlayParticipantResidency::Config{.participant_id = 1, .device = DeviceId::cpu(),
                .num_layers = 1, .num_experts = experts});
        MoEOverlayParticipantResidencyBank initial{.epoch = 1, .participant_id = 1,
            .device = DeviceId::cpu(), .layers = {{std::vector<bool>(experts, false),
                std::vector<MoEOverlayPreparedExpertPayload>(experts)}}};
        auto ready = residency->prepareReadyBank(std::move(initial));
        ASSERT_TRUE(ready.has_value());
        ASSERT_EQ(residency->installReadyBank(std::move(*ready)),
            MoEOverlayParticipantBankInstallStatus::Installed);
        params.overlay_participant_residency = residency;
        MoELocalExpertStage stage(params);

        // Do not lend the stage any currently prepared engine requirements:
        // the empty tier's source declaration alone must cover future work.
        const auto declared = stage.getWorkspaceRequirements(max_rows);
        const size_t bytes = declared.total_bytes_with_alignment();
        PhysicalMemoryPlanBuilder memory_plan;
        memory_plan.add({.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = bytes, .admission_available_bytes = bytes},
            PhysicalMemoryOwner::ExecutionWorkspace, bytes);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(memory_plan.build()), 0);
        WorkspaceAllocator allocator(memory);
        ComputeGraph host_boundary;
        WorkspaceSizingHints hints;
        hints.max_seq_len = hints.serial_family_max_rows = max_rows;
        hints.d_model = width;
        hints.batch_size = 1;
        hints.graph_family_policy = WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
        ASSERT_TRUE(allocator.allocateForGraph(host_boundary, hints,
            {{.consumer = &stage, .device = DeviceId::cpu(), .m = max_rows,
              .shape_policy = WorkspaceConsumerShapePolicy::FixedDeclaredShape}}));
        ASSERT_EQ(stage.getWorkspace(), allocator.getDeviceWorkspace(DeviceId::cpu()));
        const auto workspace_generation = allocator.deviceGeneration(DeviceId::cpu());
        std::vector<void *> workspace_addresses;
        for (const auto &buffer : declared.buffers)
            workspace_addresses.push_back(stage.getWorkspace()->getBuffer(buffer.name));
        // Revoke borrowed pointers before the allocator retires its storage,
        // including assertion-unwind paths below.
        auto unbind = [&](MoELocalExpertStage *value) { value->unbindWorkspace(); };
        std::unique_ptr<MoELocalExpertStage, decltype(unbind)> binding(&stage, unbind);
        auto result = TestTensorFactory::createFP32({capacity, size_t(width)});
        ASSERT_TRUE(result->ensureOnDevice(device, stream));
        TransferEngine::requireDeviceInput(result.get(), device, stream);
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch{
            .control = ticket->controlDeviceAlias(),
            .original_route_slots = ticket->originalRouteSlotsDeviceAlias(),
            .compact_route_slots = ticket->compactRouteSlotsDeviceAlias(),
            .compact_preweighted_contributions_fp32 = ticket->contributionRowsDeviceAlias(),
            .canonical_route_contributions_fp32 = static_cast<float *>(result->gpu_data_ptr()),
            .route_capacity = capacity, .d_model = width};
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        auto graph = context.createGraphCapture(stream);
        ASSERT_NE(graph, nullptr);
        ScopedBackendGraphCapture capture(context, *graph, "CPU sparse expert ticket ingress");
        ASSERT_TRUE(capture.begin());
        const bool captured = kernel->consumeMoEOverlayCanonicalRouteTicket(
            MoEKernelLaunchContext{.stream = stream}, launch);
        capture.finish();
        ASSERT_TRUE(captured);
        ASSERT_TRUE(graph->instantiate());
        bool pending = false;
        auto drain = [&](void *) {
            if (pending)
            {
                (void)ticket->publishAbort();
                (void)backend->synchronizeStream(stream, device.ordinal);
            }
        };
        std::unique_ptr<void, decltype(drain)> pending_guard(ticket.get(), drain);
        // The oracle is an independent serial invocation, not a borrower of
        // the endpoint's scratch. Include each engine's exact K-tree envelope
        // even at one row, where wider CPU teams can require ordered partials.
        auto oracle_requirements = cpuSwiGLUWorkspaceRequirements(1, intermediate);
        for (const auto &engine : engines)
            oracle_requirements.merge(
                dynamic_cast<const IWorkspaceConsumer &>(*engine).getWorkspaceRequirements(1));
        CPUProjectionTestWorkspace oracle_workspace(oracle_requirements);
        auto oracle_input = TestTensorFactory::createFP32({1u, size_t(width)});
        auto gate = TestTensorFactory::createFP32({1u, size_t(intermediate)});
        auto up = TestTensorFactory::createFP32({1u, size_t(intermediate)});
        auto down = TestTensorFactory::createFP32({1u, size_t(width)});
        struct PacketShape { int rows; int routes; };
        constexpr std::array<PacketShape, 9> shapes{{{1, 1}, {1, 8}, {3, 2}, {1, 7},
            {3, 3}, {1, 4}, {15, 2}, {1, 1}, {1, 8}}};
        for (size_t replay = 0; replay < shapes.size(); ++replay)
        {
            if (replay % 3 == 0)
            {
                const uint64_t previous = 1 + replay / 3;
                auto candidate = residency->cloneCandidate(previous, previous + 1);
                for (int expert = 0; expert < experts; ++expert)
                    candidate.layers[0].setResidentExpert(expert,
                        {engines[expert * 3], engines[expert * 3 + 1], engines[expert * 3 + 2]});
                auto prepared = residency->prepareReadyBank(std::move(candidate));
                ASSERT_TRUE(prepared.has_value());
                ASSERT_EQ(residency->installReadyBank(std::move(*prepared)),
                    MoEOverlayParticipantBankInstallStatus::Installed);
                ASSERT_TRUE(residency->retire(previous));
            }
            const auto [rows, routes] = shapes[replay];
            SCOPED_TRACE("replay=" + std::to_string(replay) +
                         " rows=" + std::to_string(rows) + " routes=" + std::to_string(routes));
            input.key.step_id = replay + 1u;
            input.key.sequence = replay + 1u;
            ASSERT_TRUE(input.key.isValid());
            ASSERT_EQ(input.key.layer_idx, ticket->layerIndex());
            ASSERT_EQ(input.key.participant_id, params.runtime_participant_index);
            input.residency_epoch = 2u + replay / 3u;
            input.live_row_count = rows;
            input.live_entry_count = rows * routes;
            std::vector<float> expected(capacity * width, 0.0f);
            for (int row = 0; row < rows; ++row)
            {
                input.row_ids_host[row] = row;
                input.entry_offsets_host[row] = row * routes;
                input.entry_offsets_host[row + 1] = (row + 1) * routes;
                for (int col = 0; col < width; ++col)
                {
                    const float value = 0.0173f * static_cast<float>(
                        static_cast<int>((col * 31u + row * 13u + replay * 7u) % 97u) - 48);
                    input.hidden_rows_fp32[size_t(row) * width + col] = value;
                    oracle_input->mutable_data()[col] = value;
                }
                for (int route = 0; route < routes; ++route)
                {
                    const int entry = row * routes + route;
                    const int expert = (route * 3 + row + replay) % experts;
                    const int original_slot = row * top_k + route;
                    const float weight = 0.1137f * (route + 1);
                    input.expert_ids_host[entry] = expert;
                    input.original_route_slots_host[entry] = original_slot;
                    input.compact_route_slots_host[entry] = row * top_k + route;
                    input.route_weights_host[entry] = weight;
                    std::vector<ITensorGemm::TensorProjectionDesc> pair{
                        {projections[0][expert], gate.get(), intermediate, nullptr, "gate"},
                        {projections[1][expert], up.get(), intermediate, nullptr, "up"}};
                    ASSERT_TRUE(projections[0][expert]->multiply_fused_tensor(
                        oracle_input.get(), pair, 1, width, nullptr, oracle_workspace.get()));
                    ASSERT_TRUE(projections[2][expert]->multiply_tensor_with_fused_swiglu(
                        gate.get(), up.get(), down.get(), 1, width, intermediate,
                        1.0f, 0.0f, oracle_workspace.get()));
                    for (int col = 0; col < width; ++col)
                    {
                        ASSERT_TRUE(std::isfinite(down->data()[col]));
                        expected[size_t(original_slot) * width + col] = down->data()[col] * weight;
                    }
                }
            }
            ASSERT_TRUE(backend->memset(launch.canonical_route_contributions_fp32,
                0, expected.size() * sizeof(float), device.ordinal, stream));
            ASSERT_TRUE(graph->launch());
            pending = true;
            ASSERT_TRUE(stage.execute(&cpu));
            ASSERT_EQ(allocator.deviceGeneration(DeviceId::cpu()), workspace_generation);
            for (size_t buffer = 0; buffer < declared.buffers.size(); ++buffer)
            {
                const auto &descriptor = declared.buffers[buffer];
                ASSERT_EQ(stage.getWorkspace()->getBuffer(descriptor.name), workspace_addresses[buffer]);
                ASSERT_EQ(stage.getWorkspace()->getBufferSize(descriptor.name), descriptor.size_bytes);
            }
            TransferEngine::publishDeviceWrite(result.get(), device, stream);
            ASSERT_TRUE(result->ensureOnHost(stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            pending = false;
            ASSERT_TRUE(ticket->publicationSucceededFor(input.residency_epoch));
            for (size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(std::memcmp(result->data() + i, expected.data() + i, sizeof(float)), 0)
                    << "word=" << i << " actual=" << result->data()[i]
                    << " expected=" << expected[i];
        }
    }
}

/** @brief Replay MainPrefill per-token FixedPoint32Adaptive16 CPU canonical ticket
 *         publication (both 1xCPU unsharded and odd-degree 3xCPU sharded) through
 *         captured CUDA/ROCm ticket consumption for narrow int16, wide int32,
 *         zero-sentinel duplicate routes, empty replay, and both int32-bit and FP32 outputs.
 */
void runPrefillFixedPointCanonicalTicketReplay(DeviceId device)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() <= device.ordinal)
        GTEST_SKIP() << device.toString() << " unavailable";
    auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    ASSERT_NE(kernel, nullptr);

    // Certify peer-coherent VRAM allocation contract (hipDeviceMallocUncached on ROCm, coherent on CUDA).
    void *peer_coherent_buf = backend->allocatePeerCoherent(4096u, device.ordinal);
    ASSERT_NE(peer_coherent_buf, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(peer_coherent_buf) & 255u, 0u);
    backend->free(peer_coherent_buf, device.ordinal);

    void *stream = backend->createStream(device.ordinal);
    ASSERT_NE(stream, nullptr);
    auto release_stream = [&](void *value) { backend->destroyStream(value, device.ordinal); };
    std::unique_ptr<void, decltype(release_stream)> stream_owner(stream, release_stream);

    constexpr int kRows = 4;
    constexpr int kTopK = 4;
    constexpr int kWidth = 64;
    constexpr size_t kCapacity = static_cast<size_t>(kRows * kTopK);
    constexpr float kScale = 4096.0f;
    constexpr float kInvScale = 1.0f / 4096.0f;
    constexpr float kClamp = 16000.0f;
    auto quantize = [=](float v) -> int32_t {
        const float cl = std::max(-kClamp, std::min(kClamp, v));
        return static_cast<int32_t>(std::nearbyintf(cl * kScale));
    };

    const std::array<DeviceId, 1> endpoints{device};
    auto metadata_arena = TransferEngine::instance().createMappedHostArena(endpoints);
    auto contrib_arena = TransferEngine::instance().createMappedHostArena(endpoints);
    auto contrib_region = contrib_arena->allocate(kCapacity * kWidth * sizeof(float), 64u);
    auto ticket = std::make_shared<MoEOverlayCanonicalRouteReturnTicketStorage>();
    ticket->bindFixedCapacity(0, kCapacity, kWidth, device, 1u, contrib_region, metadata_arena);
    ticket->enablePrefillFixedPoint();
    ASSERT_TRUE(ticket->prefillFixedPointEnabled());

    // 3 tokens active:
    //   tok 0 (slots 0, 2): narrow int16 sum fits in [-32768, 32767]
    //   tok 1 (slot 5): wide int32 outlier (> 9.0 -> > 36864 > 32767)
    //   tok 3 (slots 12, 13, 15): narrow 3-route pre-fold
    const std::array<int32_t, 6> orig_slots{0, 2, 5, 12, 13, 15};
    const std::array<int32_t, 6> comp_slots{0, 1, 4, 8, 9, 10};
    const std::array<int32_t, 6> expert_ids{0, 1, 2, 0, 1, 2};
    const std::array<float, 6> weights{0.5f, 0.25f, 1.0f, 0.3f, -0.4f, 0.6f};
    const std::array<int, 3> selected_experts{0, 1, 2};

    // Build 3-participant (odd degree) shard records + single-source rows with matching sums.
    constexpr size_t kPop = 3u;
    constexpr size_t kRecWidth = kWidth + 1u;
    std::vector<float> records(kPop * orig_slots.size() * kRecWidth, 0.0f);
    std::vector<float> single_source(kCapacity * kWidth, 0.0f);
    std::array<RootedFloatRecordRange, kPop> ranges{};
    for (size_t p = 0; p < kPop; ++p)
        ranges[p] = RootedFloatRecordRange{static_cast<int>(p), p * orig_slots.size(), orig_slots.size()};

    for (size_t e = 0; e < orig_slots.size(); ++e)
    {
        const bool is_wide_entry = (e == 2u);
        for (int col = 0; col < kWidth; ++col)
        {
            const float p0 = is_wide_entry ? (4.5f + 0.01f * col) : (0.15f * ((col % 7) - 3));
            const float p1 = is_wide_entry ? (3.5f - 0.005f * col) : (-0.10f * ((col % 5) - 2));
            const float p2 = is_wide_entry ? (2.25f) : (0.08f * ((col % 3) - 1));
            const float sum = (p0 + p1) + p2;
            single_source[static_cast<size_t>(comp_slots[e]) * kWidth + col] = sum;
            records[(ranges[0].first_record + e) * kRecWidth + col] = p0;
            records[(ranges[1].first_record + e) * kRecWidth + col] = p1;
            records[(ranges[2].first_record + e) * kRecWidth + col] = p2;
        }
        for (size_t p = 0; p < kPop; ++p)
        {
            ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(
                records.data() + (ranges[p].first_record + e) * kRecWidth, kWidth,
                static_cast<size_t>(orig_slots[e])));
        }
    }

    // Compute exact expected per-token pre-folded int32 sums at first slot of each token, 0 on duplicate slots.
    std::vector<int32_t> expected_i32(kCapacity * kWidth, 0);
    const std::array<std::pair<size_t, size_t>, 3> runs{{{0, 2}, {2, 3}, {3, 6}}};
    for (const auto &[begin, end] : runs)
    {
        const size_t first_slot = static_cast<size_t>(orig_slots[begin]);
        for (size_t e = begin; e < end; ++e)
        {
            for (int col = 0; col < kWidth; ++col)
            {
                const float val = single_source[static_cast<size_t>(comp_slots[e]) * kWidth + col] * weights[e];
                expected_i32[first_slot * kWidth + col] += quantize(val);
            }
        }
    }

    auto result = TestTensorFactory::createFP32({kCapacity, static_cast<size_t>(kWidth)});
    ASSERT_TRUE(result->ensureOnDevice(device, stream));
    TransferEngine::requireDeviceInput(result.get(), device, stream);
    auto &context = GPUDeviceContextPool::instance().getContext(device);

    for (uint32_t out_int32_bits : {1u, 0u})
    {
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch{
            .control = ticket->controlDeviceAlias(),
            .original_route_slots = ticket->originalRouteSlotsDeviceAlias(),
            .compact_route_slots = ticket->compactRouteSlotsDeviceAlias(),
            .compact_preweighted_contributions_fp32 = ticket->contributionRowsDeviceAlias(),
            .canonical_route_contributions_fp32 = static_cast<float *>(result->gpu_data_ptr()),
            .route_capacity = kCapacity,
            .d_model = kWidth,
            .output_fixed_point_int32_bits = out_int32_bits};
        auto graph = context.createGraphCapture(stream);
        ASSERT_NE(graph, nullptr);
        ScopedBackendGraphCapture capture(context, *graph, "CPU fixed-point prefill ticket replay");
        ASSERT_TRUE(capture.begin());
        ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket(
            MoEKernelLaunchContext{.stream = stream}, launch));
        capture.finish();
        ASSERT_TRUE(graph->instantiate());

        // Replay 0: 3xCPU sharded combine + publish; Replay 1: 1xCPU unsharded publish; Replay 2: empty 0-entry replay.
        for (int mode = 0; mode < 3; ++mode)
        {
            const uint64_t epoch = static_cast<uint64_t>(out_int32_bits * 10u + mode + 1u);
            const size_t live_entries = (mode == 2) ? 0u : orig_slots.size();
            MoEOverlaySparseRows input{};
            input.key = makeMoEOverlayCollectiveKey(1, epoch, 0, 0, 0, 1, MoEOverlayCollectiveDirection::Dispatch);
            input.key.histogram_source = ExpertHistogramSource::PrefillChunk;
            input.residency_epoch = epoch;
            input.d_model = kWidth;
            input.top_k = kTopK;
            input.row_capacity = kRows;
            input.entry_capacity = kCapacity;
            input.live_row_count = (mode == 2) ? 0u : 3u;
            input.live_entry_count = live_entries;
            input.original_route_slots_host = const_cast<int32_t *>(orig_slots.data());
            input.compact_route_slots_host = const_cast<int32_t *>(comp_slots.data());
            input.expert_ids_host = const_cast<int32_t *>(expert_ids.data());
            input.route_weights_host = const_cast<float *>(weights.data());

            ASSERT_TRUE(backend->memset(launch.canonical_route_contributions_fp32,
                0, kCapacity * kWidth * sizeof(float), device.ordinal, stream));
            ASSERT_TRUE(graph->launch());

            auto pub = ticket->arm(epoch);
            ASSERT_TRUE(pub);
            if (mode == 0)
            {
                std::vector<int32_t> maps(kCapacity * kPop, -1);
                std::vector<float> req_w(kCapacity, 0.0f);
                std::vector<float> combined(kCapacity * kWidth, 0.0f);
                cpu::moe::CanonicalShardRouteRecordArgs shard_args{
                    records, ranges, {}, {}, selected_experts, maps, req_w, combined,
                    /*original_expert_count=*/4, kRows, kTopK, kWidth,
                    cpu::moe::CanonicalShardSparseRouteView{orig_slots, expert_ids, weights},
                    std::nullopt};
                ASSERT_TRUE(combineAndPublishMoEOverlayCanonicalShardRouteRows(*ticket, pub, input, shard_args));
                // Verify narrow/wide/zero slot encoding produced by CPU pre-folder.
                EXPECT_EQ(ticket->compactRouteSlotsHost()[0] & kMoEOverlayCanonicalTicketCompactWideFlag, 0);
                EXPECT_EQ(ticket->compactRouteSlotsHost()[1], kMoEOverlayCanonicalTicketZeroSlot);
                EXPECT_NE(ticket->compactRouteSlotsHost()[2] & kMoEOverlayCanonicalTicketCompactWideFlag, 0);
                EXPECT_EQ(ticket->compactRouteSlotsHost()[3] & kMoEOverlayCanonicalTicketCompactWideFlag, 0);
                EXPECT_EQ(ticket->compactRouteSlotsHost()[4], kMoEOverlayCanonicalTicketZeroSlot);
                EXPECT_EQ(ticket->compactRouteSlotsHost()[5], kMoEOverlayCanonicalTicketZeroSlot);
            }
            else
            {
                ASSERT_TRUE(publishMoEOverlayCanonicalRouteRows(*ticket, pub, input,
                    single_source, MoEOverlayCanonicalRouteSourceLayout::compactSlots()));
            }

            TransferEngine::publishDeviceWrite(result.get(), device, stream);
            ASSERT_TRUE(result->ensureOnHost(stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            ASSERT_TRUE(ticket->publicationSucceededFor(epoch));

            for (size_t i = 0; i < kCapacity * kWidth; ++i)
            {
                const int32_t exp_q = (mode == 2) ? 0 : expected_i32[i];
                if (out_int32_bits != 0u)
                {
                    int32_t got_q = 0;
                    std::memcpy(&got_q, result->data() + i, sizeof(got_q));
                    ASSERT_EQ(got_q, exp_q) << "mode=" << mode << " idx=" << i;
                }
                else
                {
                    const float exp_f = static_cast<float>(exp_q) * kInvScale;
                    ASSERT_EQ(std::memcmp(result->data() + i, &exp_f, sizeof(float)), 0)
                        << "mode=" << mode << " idx=" << i
                        << " got=" << result->data()[i] << " exp=" << exp_f;
                }
            }
        }
    }
}
} // namespace

#ifdef HAVE_CUDA
/** @brief Every CPU format retains exact sparse route words through captured CUDA ingress. */
TEST(Test__CrossTierExpertArithmeticParity, CPUCanonicalSparseEndpointAllFormatsCUDA)
{
    runSparseExpertTicketParity(DeviceId::cuda(0));
}
/** @brief MainPrefill FixedPoint32Adaptive16 CPU canonical ticket replay on CUDA. */
TEST(Test__CrossTierExpertArithmeticParity, CPUPrefillFixedPointCanonicalTicketReplayCUDA)
{
    runPrefillFixedPointCanonicalTicketReplay(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @brief Symmetric captured ROCm consumer certifies the same retained CPU publication. */
TEST(Test__CrossTierExpertArithmeticParity, CPUCanonicalSparseEndpointAllFormatsROCm)
{
    runSparseExpertTicketParity(DeviceId::rocm(0));
}
/** @brief MainPrefill FixedPoint32Adaptive16 CPU canonical ticket replay on ROCm. */
TEST(Test__CrossTierExpertArithmeticParity, CPUPrefillFixedPointCanonicalTicketReplayROCm)
{
    runPrefillFixedPointCanonicalTicketReplay(DeviceId::rocm(0));
}
#endif
} // namespace llaminar2::test

#include "CPUJointPaddedPrefillContract.inc"
