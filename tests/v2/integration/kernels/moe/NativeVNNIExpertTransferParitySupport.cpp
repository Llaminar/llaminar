/**
 * @file NativeVNNIExpertTransferParitySupport.cpp
 * @brief Linked cross-tier demotion and retained CPU packet regression support.
 *
 * CUDA, ROCm and cross-tier arithmetic executables call this same support
 * object. Keeping the implementation outside any GTest registration unit
 * prevents missing definitions or accidental registration of another suite.
 */
#include "NativeVNNIExpertTransferParityTest.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/local_execution/device/DeviceContext.h"

namespace llaminar2::test::native_vnni_transfer_parity_detail
{
    /** @copydoc crossTierArithmeticFormats */
    std::vector<QuantizedVerifierFormatCase> crossTierArithmeticFormats()
    {
        auto formats = quantizedMoEVerifierFormats();
        auto boundary = *std::find_if(formats.begin(), formats.end(),
            [](const auto &format) { return format.tensor_type == TensorType::Q8_0; });
        boundary.label = "Q8_0_SmallScales";
        boundary.create = [](const std::vector<size_t> &shape, uint32_t seed)
            -> std::unique_ptr<TensorBase>
        {
            auto original = TestTensorFactory::createQ8_0Random(shape, seed);
            const size_t blocks = shape[0] * shape[1] / Q8_0Block::BLOCK_SIZE;
            std::vector<uint8_t> bytes(blocks * sizeof(Q8_0Block));
            std::memcpy(bytes.data(), original->typed_data(), bytes.size());
            // Early real-model layers contain zero and subnormal half scales;
            // normally distributed synthetic weights mostly miss this range.
            // Preserve signed payload diversity while varying only metadata.
            constexpr std::array<uint16_t, 12> scales{
                0x0000, 0x0001, 0x0003, 0x0101, 0x0203, 0x03ff,
                0x0400, 0x0401, 0x0603, 0x0801, 0x0c03, 0x1001};
            for (size_t block = 0; block < blocks; ++block)
            {
                const auto scale = scales[(block * 7u + seed) % scales.size()];
                std::memcpy(bytes.data() + block * sizeof(Q8_0Block),
                            &scale, sizeof(scale));
            }
            return std::make_unique<Q8_0Tensor>(shape, bytes);
        };
        formats.push_back(std::move(boundary));
        return formats;
    }

    /** @copydoc expectCPUExpertPacketParity */
    void expectCPUExpertPacketParity(
        const std::array<ITensorGemm *, 3> &engines,
        const float *hidden, const float *expected,
        int rows, int d_model, int intermediate)
    {
        constexpr int routes = 8;
        const auto count = static_cast<size_t>(rows);
        MoEOverlayCollectiveWorkspace workspace({
            .max_rows = count, .max_entries = count * routes,
            .d_model = d_model, .top_k = routes, .device = DeviceId::cpu(),
            .return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes,
        });
        auto input = workspace.localExpertInput(0, 0);
        auto output = workspace.localExpertOutput(0, 0);
        input.residency_epoch = 1;
        input.live_row_count = count;
        input.live_entry_count = count * routes;
        input.key.histogram_source = rows == 1
            ? ExpertHistogramSource::DecodeToken
            : ExpertHistogramSource::PrefillChunk;

        // Odd physical rows carry the real inputs. Even rows are poison:
        // accidentally treating a physical-row packet as compact is visible.
        std::vector<float> physical((count * 2u + 1u) * d_model, -17.0f);
        input.hidden_rows_fp32 = physical.data();
        input.hidden_row_capacity = count * 2u + 1u;
        input.hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::SharedPhysicalRows;
        for (int row = 0; row < rows; ++row)
        {
            input.row_ids_host[row] = row * 2 + 1;
            std::copy_n(hidden + static_cast<size_t>(row) * d_model,
                        d_model, physical.data() + (row * 2u + 1u) * d_model);
        }

        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(), .row_capacity = count,
                .d_model = d_model, .routing_top_k = routes,
                .cpu_canonical_route_storage = MoELocalExpertSerialBufferArena::
                    CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
                .cpu_grouped_scratch_storage = MoELocalExpertSerialBufferArena::
                    CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
                .num_experts = routes, .expert_intermediate = intermediate,
            });
        MoELocalExpertStage::Params params;
        params.device_id = DeviceId::cpu();
        params.return_layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
        params.input_rows = &input;
        params.output_rows = &output;
        params.serial_compact_buffer_arena = arena;
        params.graph_row_capacity = count;
        params.num_experts = routes;
        params.top_k = routes;
        params.d_model = d_model;
        params.expert_intermediate = intermediate;
        params.layer_idx = 0;
        params.expert_mask.assign(routes, true);
        params.prepared_gate_gemm.assign(routes, engines[0]);
        params.prepared_up_gemm.assign(routes, engines[1]);
        params.prepared_down_gemm.assign(routes, engines[2]);
        MoELocalExpertStage stage(params);
        CPUDeviceContext context(DeviceId::cpu());

        // Movement changes local route count without rebuilding the executor.
        // Cross every compact width bucket, including an empty packet followed
        // by reuse of the full width. Multi-row packets additionally contain
        // unequal/empty CSR rows, so physical and compact row indices differ.
        constexpr std::array<int, 9> widths{8, 3, 1, 5, 7, 2, 4, 0, 8};
        const int invocations = rows == 1 ? static_cast<int>(widths.size()) : 2;
        for (int invocation = 0; invocation < invocations; ++invocation)
        {
            SCOPED_TRACE("packet invocation=" + std::to_string(invocation));
            size_t entries = 0;
            for (int row = 0; row < rows; ++row)
            {
                input.entry_offsets_host[row] = entries;
                const int live_routes = rows == 1 ? widths[invocation]
                    : widths[(row + invocation) % widths.size()];
                for (int route = 0; route < live_routes; ++route)
                {
                    input.expert_ids_host[entries] = routes - 1 - route;
                    input.route_weights_host[entries] = 0.13f + 0.07f * route;
                    input.original_route_slots_host[entries] = (row * 2 + 1) * routes + route;
                    input.compact_route_slots_host[entries] = entries;
                    ++entries;
                }
            }
            input.entry_offsets_host[rows] = entries;
            input.live_entry_count = entries;
            std::fill_n(output.output_rows_fp32, count * routes * d_model, -29.0f);
            ASSERT_TRUE(stage.execute(&context));
            ASSERT_EQ(output.live_row_count, entries);
            for (int row = 0; row < rows; ++row)
                for (auto entry = input.entry_offsets_host[row];
                     entry < input.entry_offsets_host[row + 1]; ++entry)
                {
                    ASSERT_EQ(output.row_ids_host[entry], input.original_route_slots_host[entry]);
                    const float *actual = output.output_rows_fp32 + entry * d_model;
                    const float *oracle = expected + static_cast<size_t>(row) * d_model;
                    expectByteEqual("retained CPU sparse packet", {actual, actual + d_model},
                                    {oracle, oracle + d_model}, d_model);
                    if (::testing::Test::HasFailure())
                        return;
                }
        }
    }

    /**
     * @brief Exercise the production demotion lane and final CPU engine owner.
     *
     * Promotion-only arithmetic cannot detect a demotion layout mismatch. The
     * destination here is the same preallocated slot/engine pair installed by
     * residency publication, not a byte vector interpreted by the test.
     *
     * @param device Exact GPU source.
     * @param stream Source preparation stream whose event orders the lane read.
     * @param format Catalogued quantized source identity.
     * @param descriptors Complete immutable gate/up/down source triplet.
     * @return Alias lease preserving the destination engines and their pool.
     * @throws std::runtime_error on invalid preparation, transfer or completion.
     */
    CpuExpertSlotPool::Lease demoteGPUExpert(
        DeviceId device, void *stream,
        const QuantizedVerifierFormatCase &format,
        const std::array<DeviceNativeVNNIMatrixDesc, 3> &descriptors)
    {
        constexpr std::array roles = {
            ExpertTierWeightProjection::Gate, ExpertTierWeightProjection::Up,
            ExpertTierWeightProjection::Down};
        const auto *source_format = native_vnni_formats::forSourceIdentity(
            format.source_codebook_id, format.source_is_superblock);
        if (!source_format || !stream)
            throw std::runtime_error("Demotion proof requires a source format and stream");
        const auto expert_format = ExpertWeightFormat::nativeVnni({
            .codebook_id = format.source_codebook_id,
            .is_superblock = format.source_is_superblock,
            .present = true,
        });
        std::vector<CpuExpertSlotPool::ProjectionSpec> specs;
        for (size_t i = 0; i < roles.size(); ++i)
            specs.push_back({roles[i], descriptors[i].n, descriptors[i].k, expert_format});
        auto pool = CpuExpertSlotPool::createForTest({
            .participant_id = 1, .layer_idx = 0, .capacity = 1,
            .projections = std::move(specs),
            .memory_placement = CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
            .perf_device = "cross-tier-demotion-cpu",
        });
        auto lease = pool->acquire(/*expert_id=*/0, /*residency_epoch=*/2);
        if (!lease)
            throw std::runtime_error("Demotion proof could not acquire its inactive CPU slot");
        auto *backend = getBackendFor(device);
        auto destroy_event = [&](void *event) { backend->destroyEvent(event, device.ordinal); };
        std::unique_ptr<void, decltype(destroy_event)> ready(
            backend->createEvent(device.ordinal), destroy_event);
        if (!ready || !backend->recordEvent(ready.get(), device.ordinal, stream))
            throw std::runtime_error("Demotion proof could not publish source readiness");

        for (size_t i = 0; i < roles.size(); ++i)
        {
            const auto &descriptor = descriptors[i];
            const auto layout = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format, descriptor.n, descriptor.k,
                /*residency_epoch=*/1, /*layer_idx=*/0, /*expert_id=*/0,
                roles[i], /*maximum_units_per_chunk=*/16).deviceLayout();
            const size_t blocks = static_cast<size_t>(layout.N) * layout.blocks_per_row;
            const ExpertTierGpuConstProjectionView source{
                .payload = descriptor.payload,
                .scales = static_cast<const uint16_t *>(descriptor.scales),
                .mins = static_cast<const uint16_t *>(descriptor.mins),
                .emins = static_cast<const uint32_t *>(descriptor.emins),
                .payload_bytes = blocks * layout.gpu_payload_bytes_per_block,
                .scales_bytes = blocks * sizeof(uint16_t),
                .mins_bytes = layout.gpu_is_asymmetric ? blocks * sizeof(uint16_t) : 0u,
                .emins_bytes = layout.gpu_has_emins ? blocks * sizeof(uint32_t) : 0u,
            };
            auto &transfer = TransferEngine::instance();
            ExpertTierWeightTransferLane lane({
                .device = device,
                .staging = transfer.allocatePersistentTransferStagingSlices(
                    layout.chunkBytes(layout.maximum_units_per_chunk), 1u, device).front(),
                .execution = transfer.allocatePersistentTransferExecutionLanes(
                    1u, device, "cross-tier-demotion").front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cross-tier-demotion",
                .perf_device = device.to_string(),
            });
            std::string error;
            if (!lane.materialize(&error) ||
                !lane.startGpuToCpu(layout, source, lease->projections[i].destination_bytes,
                                    ExpertTierSourceReadiness::producerEvent(ready.get()), &error))
                throw std::runtime_error("Demotion proof could not start: " + error);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            auto progress = lane.progress();
            while (progress == ExpertTierWeightTransferProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                progress = lane.poll(&error);
                std::this_thread::yield();
            }
            if (progress != ExpertTierWeightTransferProgress::Ready)
                throw std::runtime_error("Demotion proof did not complete: " + error);
            const auto stats = lane.stats();
            if (stats.transfers_completed != 1u || stats.blocking_synchronizations != 0u ||
                stats.inference_stream_waits != 0u)
                throw std::runtime_error("Demotion proof violated its asynchronous transfer contract");
        }
        // Engine aliases retain the pool when this local shared owner leaves.
        return std::move(*lease);
    }
} // namespace llaminar2::test::native_vnni_transfer_parity_detail
