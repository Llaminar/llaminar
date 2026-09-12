/**
 * @file Test__CPUSparseExpertTicketParity.cpp
 * @brief Numerical proof of retained CPU expert execution and captured GPU ingress.
 *
 * Moving experts changes the CPU endpoint's local route width even when the
 * model's top-k is unchanged. This test reuses the production stage, arena and
 * canonical return ticket across narrow/wide packets. Independent serial CPU
 * expert rows certify the result after the real captured CUDA/ROCm consumer;
 * synthetic ticket payloads alone cannot catch a compute-to-ticket stride bug.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/QuantizedVerifierFormats.h"

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
        std::vector<std::unique_ptr<ITensorGemm>> engines;
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
                    engines.push_back(std::make_unique<cpu::native_vnni::CPUNativeVNNIGemmKernel>(
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
        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            MoELocalExpertSerialBufferArena::Config{
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
                .debug_name = "integration.cpu_sparse_ticket"});
        const std::array<DeviceId, 1> endpoints{device};
        auto metadata = TransferEngine::instance().createMappedHostArena(endpoints);
        auto ticket = std::make_shared<MoEOverlayCanonicalRouteReturnTicketStorage>();
        ticket->bindFixedCapacity(0, capacity, width, device, 1,
            arena->mappedCPUCanonicalRoutes(device), metadata);
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
        params.expert_mask.assign(experts, true);
        params.prepared_gate_gemm = projections[0];
        params.prepared_up_gemm = projections[1];
        params.prepared_down_gemm = projections[2];
        MoELocalExpertStage stage(params);
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
        auto oracle_input = TestTensorFactory::createFP32({1u, size_t(width)});
        auto gate = TestTensorFactory::createFP32({1u, size_t(intermediate)});
        auto up = TestTensorFactory::createFP32({1u, size_t(intermediate)});
        auto down = TestTensorFactory::createFP32({1u, size_t(width)});
        struct PacketShape { int rows; int routes; };
        constexpr std::array<PacketShape, 9> shapes{{{1, 1}, {1, 8}, {3, 2}, {1, 7},
            {3, 3}, {1, 4}, {15, 2}, {1, 1}, {1, 8}}};
        for (size_t replay = 0; replay < shapes.size(); ++replay)
        {
            const auto [rows, routes] = shapes[replay];
            SCOPED_TRACE("replay=" + std::to_string(replay) +
                         " rows=" + std::to_string(rows) + " routes=" + std::to_string(routes));
            input.residency_epoch = 1u + replay / 3u;
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
                        oracle_input.get(), pair, 1, width));
                    ASSERT_TRUE(projections[2][expert]->multiply_tensor_with_fused_swiglu(
                        gate.get(), up.get(), down.get(), 1, width, intermediate));
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
} // namespace

#ifdef HAVE_CUDA
/** @brief Every CPU format retains exact sparse route words through captured CUDA ingress. */
TEST(Test__CrossTierExpertArithmeticParity, CPUCanonicalSparseEndpointAllFormatsCUDA)
{
    runSparseExpertTicketParity(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @brief Symmetric captured ROCm consumer certifies the same retained CPU publication. */
TEST(Test__CrossTierExpertArithmeticParity, CPUCanonicalSparseEndpointAllFormatsROCm)
{
    runSparseExpertTicketParity(DeviceId::rocm(0));
}
#endif
} // namespace llaminar2::test
