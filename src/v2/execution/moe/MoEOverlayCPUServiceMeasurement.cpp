/**
 * @file MoEOverlayCPUServiceMeasurement.cpp
 * @brief Measures complete CPU expert FFNs without changing serving state.
 *
 * Preparation owns a small bounded input and the normal grouped-CPU workspace.
 * Every physical payload is claimed before construction and released after
 * the last stage has retired. Different phases/classes reuse that capacity
 * serially. Only the measured service rows survive setup; no graph, weight
 * copy, inference counter, or alternate placement lifecycle is retained.
 */
#include "MoEOverlayCPUServiceMeasurement.h"

#include "MoEOverlayPreparedWeightSource.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        using llaminar::v2::kernels::KernelFactory;
        // Small real-kernel observations, not full-model synthetic inference.
        constexpr int kPrefillRows = 8;
        constexpr int kVerifierRows = 4;
        constexpr int kMeasuredSamples = 3;

        /** @return Production phase and bounded representative batch geometry. */
        std::pair<MoEOverlayServicePhaseHint, int> phaseGeometry(ExpertHistogramSource source)
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken: return {MoEOverlayServicePhaseHint::Decode, 1};
            case ExpertHistogramSource::PrefillChunk: return {MoEOverlayServicePhaseHint::Prefill, kPrefillRows};
            case ExpertHistogramSource::GroupedVerifier: return {MoEOverlayServicePhaseHint::GroupedVerifier, kVerifierRows};
            default: throw std::invalid_argument("CPU service measurement requires a priced production phase");
            }
        }

        /** @return The exact CPU workspace constructor geometry used by a probe. */
        CPUGroupedMoESerialWorkspace::Config workspaceConfig(
            MoEOverlayCPUServiceMeasurement::Geometry geometry, int rows)
        {
            return {.row_capacity = static_cast<size_t>(rows),
                    .d_model = geometry.d_model, .expert_intermediate = geometry.intermediate,
                    .num_experts = 1, .routing_top_k = 1,
                    .debug_name = "overlay_prepared_service"};
        }

        /** @return Exact input/output/route tensor and grouped-workspace payload bytes. */
        size_t payloadBytes(MoEOverlayCPUServiceMeasurement::Geometry geometry, int rows)
        {
            const size_t workspace = CPUGroupedMoESerialWorkspace::plannedAllocationBytes(
                workspaceConfig(geometry, rows));
            const size_t tensor_elements = static_cast<size_t>(rows) *
                (2u * static_cast<size_t>(geometry.d_model) + 2u);
            if (tensor_elements > (std::numeric_limits<size_t>::max() - workspace) / sizeof(float))
                throw std::overflow_error("CPU prepared-service tensor payload overflow");
            return workspace + tensor_elements * sizeof(float);
        }

        /**
         * @brief Reject geometry mismatches before constructing any scratch or stage.
         * @param engine Exact prepared source retained by the bank lease.
         * @param device CPU physical owner.
         * @param n Expected full projection output width.
         * @param k Expected full projection input width.
         */
        void requireProjection(const std::shared_ptr<ITensorGemm> &engine, DeviceId device, int n, int k)
        {
            MoEOverlayPreparedWeightSource source;
            std::string error;
            if (!resolveMoEOverlayPreparedWeightSource(engine, device, source, &error))
                throw std::invalid_argument("CPU prepared-service source: " + error);
            const auto actual_n = source.cpu_packed ? source.cpu_packed->N : source.floating.n;
            const auto actual_k = source.cpu_packed ? source.cpu_packed->K : source.floating.k;
            if (actual_n != n || actual_k != k)
                throw std::invalid_argument("CPU prepared-service projection differs from admitted geometry");
        }

        /**
         * @brief Run one complete FFN with exclusively owned invocation storage.
         * @return Sum of physical wall-clock nanoseconds across measured samples.
         *
         * The first invocation touches code/weights and starts the existing
         * OpenMP team outside the timed samples. Inputs are nonzero bounded FP32;
         * the sole selected expert is privately indexed at zero, without
         * modifying its global bank identity or any routing histogram.
         */
        uint64_t measureExpert(const MoEOverlayPreparedExpertTriplet &source,
                              DeviceId device, int layer, ExpertHistogramSource source_phase,
                              MoEOverlayCPUServiceMeasurement::Geometry geometry,
                              const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        {
            const auto [phase, rows] = phaseGeometry(source_phase);
            requireProjection(source.gate, device, geometry.intermediate, geometry.d_model);
            requireProjection(source.up, device, geometry.intermediate, geometry.d_model);
            requireProjection(source.down, device, geometry.d_model, geometry.intermediate);
            // Declaration order retires stages/engines/tensors before the claim.
            auto claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace,
                                                    payloadBytes(geometry, rows));
            FP32Tensor input({static_cast<size_t>(rows), static_cast<size_t>(geometry.d_model)});
            FP32Tensor indices({static_cast<size_t>(rows), 1});
            FP32Tensor weights({static_cast<size_t>(rows), 1});
            FP32Tensor output({static_cast<size_t>(rows), static_cast<size_t>(geometry.d_model)});
            for (int row = 0; row < rows; ++row)
            {
                indices.mutable_typed_data()[row] = 0.0f;
                weights.mutable_typed_data()[row] = 1.0f;
                for (int col = 0; col < geometry.d_model; ++col)
                    input.mutable_typed_data()[static_cast<size_t>(row) * geometry.d_model + col] =
                        0.0001f * static_cast<float>((col % 31 * 13 + row * 7) % 31 - 15);
            }
            const MoEOverlayPreparedExpertTriplet engines{
                KernelFactory::createExpertServiceExecutionView(source.gate, device),
                KernelFactory::createExpertServiceExecutionView(source.up, device),
                KernelFactory::createExpertServiceExecutionView(source.down, device)};
            auto workspace = std::make_shared<CPUGroupedMoESerialWorkspace>(workspaceConfig(geometry, rows));
            MoEExpertComputeStage::Params p;
            p.device_id = device;
            p.input = &input;
            p.output = &output;
            p.output_registered_in_arena = false;
            p.routing_indices = &indices;
            p.routing_weights = &weights;
            p.seq_len = rows;
            p.d_model = geometry.d_model;
            p.expert_intermediate = geometry.intermediate;
            p.num_experts = 1;
            p.top_k = 1;
            p.layer_idx = layer;
            p.expert_mask = {true};
            p.prepared_gate_gemm = {engines.gate.get()};
            p.prepared_up_gemm = {engines.up.get()};
            p.prepared_down_gemm = {engines.down.get()};
            p.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
            p.cpu_router_q8_input_publication = CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
            p.cpu_grouped_serial_workspace = std::move(workspace);
            p.service_phase = phase;
            p.force_decode_equivalent_verifier_prefill = phase == MoEOverlayServicePhaseHint::GroupedVerifier;
            MoEExpertComputeStage stage(std::move(p));
            auto context = IDeviceContext::create(device);
            if (!stage.execute(context.get()))
                throw std::runtime_error("CPU prepared-service warmup failed");
            uint64_t total = 0;
            for (int sample = 0; sample < kMeasuredSamples; ++sample)
            {
                const auto start = std::chrono::steady_clock::now();
                if (!stage.execute(context.get()))
                    throw std::runtime_error("CPU prepared-service FFN failed");
                const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (nanoseconds <= 0 || static_cast<uint64_t>(nanoseconds) > UINT64_MAX - total)
                    throw std::runtime_error("CPU prepared-service timing is not positive and representable");
                total += static_cast<uint64_t>(nanoseconds);
            }
            return total;
        }
    }

    size_t MoEOverlayCPUServiceMeasurement::allocationBytes(Geometry geometry)
    {
        return payloadBytes(geometry, std::max(kPrefillRows, kVerifierRows));
    }

    std::vector<MoEOverlayParticipantLayerServiceTotals> MoEOverlayCPUServiceMeasurement::measure(
        const MoEOverlayParticipantResidencyRegistry &registry, uint64_t epoch,
        const MoEOverlayEconomyCalibrationLayerCatalog &catalog,
        const ExpertHistogramProductionTopology &topology, Geometry geometry,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!memory) throw std::invalid_argument("CPU prepared-service measurement requires physical admission");
        if (epoch != registry.initialEpoch())
            throw std::logic_error("CPU prepared-service measurement may only lease the published initial epoch");
        (void)allocationBytes(geometry); // Validate before indexing the immutable banks.
        const auto started = std::chrono::steady_clock::now();
        const auto ids = registry.localParticipantIds();
        std::vector<MoEOverlayParticipantLayerServiceTotals> live;
        std::vector<MoEOverlayParticipantLayerServiceTotals> result;
        for (int id : ids)
        {
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            const auto endpoint = registry.endpoint(id);
            if (!endpoint || !endpoint->trySnapshotServiceMeasurements(&rows))
                throw std::logic_error("CPU prepared-service setup raced a service writer");
            live.insert(live.end(), rows.begin(), rows.end());
            for (size_t layer = 0; layer < catalog.layerCount(); ++layer)
                result.push_back({.participant_id = id, .layer = static_cast<int>(layer)});
        }
        size_t observations = 0;
        for (const auto &gap : catalog.serviceEvidenceGaps(live, ids, topology))
        {
            const auto endpoint = registry.endpoint(gap.participant_id);
            if (!endpoint->device().is_cpu()) continue; // GPU prices remain device-produced.
            const auto bank = endpoint->acquire(epoch);
            if (!bank) throw std::logic_error("CPU prepared-service lost its published bank lease");
            int selected_layer = -1;
            int selected_expert = -1;
            for (int layer : gap.eligible_layers)
            {
                const auto &entries = bank->layers.at(static_cast<size_t>(layer));
                for (size_t expert = 0; expert < entries.experts.size(); ++expert)
                    if (entries.resident_mask.at(expert) && entries.experts[expert].complete())
                    {
                        selected_layer = layer;
                        selected_expert = static_cast<int>(expert);
                        break;
                    }
                if (selected_expert >= 0) break;
            }
            if (selected_expert < 0)
                throw std::logic_error("CPU prepared-service class has no exact resident source");
            const auto duration = measureExpert(bank->layers[selected_layer].experts[selected_expert],
                bank->device, selected_layer, gap.source, geometry, memory);
            const auto participant_index = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), gap.participant_id) - ids.begin());
            auto &row = result.at(participant_index * catalog.layerCount() + selected_layer);
            const auto phase = expertHistogramProductionSourceIndex(gap.source);
            row.total_nanoseconds[phase] = duration;
            row.activation_count[phase] = static_cast<uint64_t>(phaseGeometry(gap.source).second) * kMeasuredSamples;
            row.sample_count[phase] = kMeasuredSamples;
            ++observations;
            LOG_DEBUG("[ExpertOverlay] prepared CPU service participant=" << gap.participant_id
                << " layer=" << selected_layer << " expert=" << selected_expert
                << " phase=" << phase << " rows=" << phaseGeometry(gap.source).second
                << " samples=" << kMeasuredSamples << " nanoseconds=" << duration);
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        LOG_INFO("[ExpertOverlay] prepared CPU service observations=" << observations
            << " elapsed_ms=" << elapsed_ms);
        return result;
    }
}
