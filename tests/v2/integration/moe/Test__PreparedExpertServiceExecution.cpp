/**
 * @file Test__PreparedExpertServiceExecution.cpp
 * @brief Real FFN execution proves private service probes cannot rebind serving.
 *
 * A retained serving stage executes before and after a short-lived probe over
 * the same prepared expert bytes. The probe owns independent execution handles,
 * routing, activations and workspace. GPU observations come from captured
 * complete expert FFNs on distinct explicit streams, never three stand-alone
 * GEMM estimates. CPU uses the production transported-row grouped workspace.
 * These are model-free ownership/capture regressions, not economy benchmarks.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "kernels/KernelFactory.h"
#include "transfer/TransferEngine.h"
#include "../../utils/QuantizedVerifierFormats.h"

#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/cuda/gemm/CUDAWeightPacker.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using llaminar::v2::kernels::KernelFactory;
        constexpr int kWidth = 256; // Valid K block geometry for every codebook.
        constexpr int kExperts = 4;
        constexpr int kExpert = 2; // A sparse, nonzero global expert identity.

        /** @brief Turn a failed infrastructure operation into a fatal test error. */
        void require(bool success, const char *operation)
        {
            if (!success) throw std::runtime_error(operation);
        }

        /** @brief Prepare one exact source format using the real backend engine. */
        std::shared_ptr<ITensorGemm> prepare(
            const std::shared_ptr<TensorBase> &weight, DeviceId device, void *stream)
        {
            if (device.is_cpu())
                return KernelFactory::prepareExpertGemmLocal(weight, device);
            const auto type = weight->native_type();
            const bool floating = type == TensorType::FP16 ||
                                  type == TensorType::BF16 || type == TensorType::FP32;
            if (floating) TransferEngine::prepareDeviceInput(weight.get(), device, stream);
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                if (!floating)
                {
                    auto packed = std::make_shared<cuda::CUDAPackedWeights>();
                    require(cuda::packWeightsToCUDA(weight.get(), *packed), "CUDA prepared weights");
                    std::shared_ptr<ITensorGemm> engine(
                        new cuda::CUDAQuantisedGemmKernel(packed.get(), device.gpu_ordinal()),
                        [packed](ITensorGemm *value) { delete value; });
                    engine->prepareWeights();
                    return engine;
                }
                using Precision = cuda::CUDAFloatingPointGemmKernel::Precision;
                return std::make_shared<cuda::CUDAFloatingPointGemmKernel>(
                    weight.get(), device.gpu_ordinal(),
                    type == TensorType::FP16 ? Precision::FP16 :
                    type == TensorType::BF16 ? Precision::BF16 : Precision::FP32);
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                if (!floating)
                {
                    auto packed = std::make_shared<rocm::ROCmPackedWeights>();
                    // Overlay uses separated native blocks for every source,
                    // including Q8. The generic legacy GEMM packer selects a
                    // different INT8 row-scale representation for those inputs.
                    packed->N = static_cast<int>(weight->rows());
                    packed->K = static_cast<int>(weight->cols());
                    require(rocm::packNativeVNNI(weight.get(), *packed), "ROCm prepared native blocks");
                    std::shared_ptr<ITensorGemm> engine(
                        new rocm::ROCmQuantisedGemmKernel(packed.get(), device.gpu_ordinal()),
                        [packed](ITensorGemm *value) { delete value; });
                    engine->prepareWeights();
                    return engine;
                }
                using Precision = rocm::ROCmFloatingPointGemmKernel::Precision;
                return std::make_shared<rocm::ROCmFloatingPointGemmKernel>(
                    weight.get(), device.gpu_ordinal(),
                    type == TensorType::FP16 ? Precision::FP16 :
                    type == TensorType::BF16 ? Precision::BF16 : Precision::FP32);
            }
#endif
            throw std::invalid_argument("Prepared expert test requires a compiled backend");
        }

        /**
         * @brief Private stage, buffers and captured executable for one geometry.
         *
         * Member order retires the native graph before its stage, execution
         * handles, workspace and tensors. Every replay joins its exact terminal
         * event before reading output; these waits are test/setup
         * observations and do not add synchronization to serving inference.
         */
        class ExpertExecution final
        {
        public:
            /** @brief Bind one complete prepared expert without a live router or histogram. */
            ExpertExecution(DeviceId device, void *stream, int rows,
                            MoEOverlayServicePhaseHint phase,
                            MoEOverlayPreparedExpertTriplet engines)
                : device_(device), stream_(stream), rows_(rows),
                  input_({static_cast<size_t>(rows), kWidth}),
                  indices_({static_cast<size_t>(rows), 1}),
                  weights_({static_cast<size_t>(rows), 1}),
                  output_({static_cast<size_t>(rows), kWidth}),
                  context_(IDeviceContext::create(device)), engines_(std::move(engines))
            {
                input_.setDebugName("prepared_service_hidden");
                indices_.setDebugName("prepared_service_routes");
                weights_.setDebugName("prepared_service_route_weights");
                output_.setDebugName("prepared_service_output");
                for (int row = 0; row < rows; ++row)
                {
                    indices_.mutable_typed_data()[row] = kExpert;
                    weights_.mutable_typed_data()[row] = 1.0f;
                    for (int col = 0; col < kWidth; ++col)
                        input_.mutable_typed_data()[row * kWidth + col] =
                            0.0001f * static_cast<float>((col * 13 + row * 7) % 31 - 15);
                }
                MoEExpertComputeStage::Params p;
                p.device_id = device;
                p.input = &input_;
                p.seq_len = rows;
                p.d_model = kWidth;
                p.num_experts = kExperts;
                p.top_k = 1;
                p.expert_intermediate = kWidth;
                p.layer_idx = 10;
                p.routing_indices = &indices_;
                p.routing_weights = &weights_;
                p.output = &output_;
                p.output_registered_in_arena = false;
                p.service_phase = phase;
                p.force_decode_equivalent_verifier_prefill =
                    phase == MoEOverlayServicePhaseHint::GroupedVerifier;
                p.expert_mask.assign(kExperts, false);
                p.expert_mask[kExpert] = true;
                p.prepared_gate_gemm.assign(kExperts, nullptr);
                p.prepared_up_gemm.assign(kExperts, nullptr);
                p.prepared_down_gemm.assign(kExperts, nullptr);
                p.prepared_gate_gemm[kExpert] = engines_.gate.get();
                p.prepared_up_gemm[kExpert] = engines_.up.get();
                p.prepared_down_gemm[kExpert] = engines_.down.get();
                p.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
                if (device.is_cpu())
                {
                    p.cpu_router_q8_input_publication = CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
                    p.cpu_grouped_serial_workspace = std::make_shared<CPUGroupedMoESerialWorkspace>(
                        CPUGroupedMoESerialWorkspace::Config{
                            .row_capacity = static_cast<size_t>(rows), .d_model = kWidth,
                            .expert_intermediate = kWidth, .num_experts = kExperts,
                            .routing_top_k = 1, .debug_name = "prepared_service_execution_test"});
                }
                else
                {
                    p.require_device_routing_tensor_decode = true;
                    for (auto *tensor : {&input_, &indices_, &weights_})
                        TransferEngine::prepareDeviceInput(tensor, device, stream);
                    TransferEngine::prepareDeviceOutput(&output_, device, stream);
                }
                stage_ = std::make_unique<MoEExpertComputeStage>(std::move(p));
                if (!device.is_gpu()) return;

                const auto requirements = stage_->getWorkspaceRequirements(rows);
                workspace_ = std::make_unique<DeviceWorkspaceManager>(
                    device, requirements.total_bytes_with_alignment());
                require(workspace_->allocate(requirements), "private stage workspace allocation");
                stage_->bindWorkspace(workspace_.get());
                require(stage_->prepareGraphLaunch(context_.get(), stream), "private stage capture preparation");
                // Preparation publishes H2D events. Admit those external
                // producer edges before recording this graph, as the ordinary
                // graph executor does; capture cannot create them retroactively.
                for (auto *tensor : {&input_, &indices_, &weights_})
                    TransferEngine::requireDeviceInput(tensor, device, stream);
                auto &worker = GPUDeviceContextPool::instance().getContext(device);
                graph_ = worker.createGraphCapture(stream);
                ScopedBackendGraphCapture capture(worker, *graph_, "prepared expert service execution proof");
                require(capture.begin(), "private stage capture begin");
                require(stage_->execute(context_.get()), "private stage captured execute");
                capture.finish();
                require(graph_->nodeCount() > 0, "nonempty captured expert FFN");
                require(graph_->instantiate(), "private stage instantiate");
            }

            /** @brief Replay the exact stage, returning positive physical service time. */
            std::uint64_t replay()
            {
                if (device_.is_cpu())
                {
                    const auto start = std::chrono::steady_clock::now();
                    require(stage_->execute(context_.get()), "CPU prepared expert execute");
                    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count();
                }
                auto *backend = getBackendFor(device_);
                const int ordinal = device_.gpu_ordinal();
                // Events are owned by this observation and the same explicit
                // stream as the captured transaction, not a serving stream.
                auto deleter = [backend, ordinal](void *event)
                { if (event) backend->destroyEvent(event, ordinal); };
                std::unique_ptr<void, decltype(deleter)> start(backend->createTimingEvent(ordinal), deleter);
                std::unique_ptr<void, decltype(deleter)> stop(backend->createTimingEvent(ordinal), deleter);
                require(start && stop, "prepared expert timing events");
                require(backend->recordEvent(start.get(), ordinal, stream_), "prepared expert start event");
                require(graph_->launch(), "prepared expert replay");
                require(backend->recordEvent(stop.get(), ordinal, stream_), "prepared expert stop event");
                require(backend->waitForEvent(stop.get(), ordinal), "prepared expert terminal event");
                float milliseconds = 0;
                require(backend->eventElapsedTimeMs(start.get(), stop.get(), ordinal, &milliseconds),
                        "prepared expert elapsed time");
                require(std::isfinite(milliseconds) && milliseconds > 0, "positive measured GPU service time");
                TransferEngine::publishDeviceWrite(&output_, device_, stream_);
                return static_cast<std::uint64_t>(milliseconds * 1000000.0);
            }

            /** @brief Observe complete numerical output outside timed graph execution. */
            std::vector<float> output()
            {
                if (device_.is_gpu()) TransferEngine::prepareHostInput(&output_);
                return {output_.typed_data(), output_.typed_data() + rows_ * kWidth};
            }

        private:
            DeviceId device_;
            void *stream_;
            int rows_;
            FP32Tensor input_, indices_, weights_, output_;
            std::unique_ptr<IDeviceContext> context_;
            std::unique_ptr<DeviceWorkspaceManager> workspace_;
            MoEOverlayPreparedExpertTriplet engines_;
            std::unique_ptr<MoEExpertComputeStage> stage_;
            std::unique_ptr<IGPUGraphCapture> graph_;
        };

        /**
         * @brief Prove the production CPU producer prices a completely unrouted bank.
         *
         * The same caller-owned source engines remain bound to a serving stage.
         * PMA admits exactly the producer's declared scratch, and must show no
         * live probe payload after completion or a rejected invocation.
         */
        void proveProductionCPUMeasurement(const MoEOverlayPreparedExpertTriplet &source)
        {
            RoutedExpertDomain domain;
            domain.name = "cpu_service";
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::HOST;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {0};
            domain.owner_rank = 0;
            RoutedExpertTier tier;
            tier.name = "priority_4";
            tier.domain = domain.name;
            tier.priority = 4;
            tier.fallback = true;
            MoERoutedExpertPlacementPlan placement;
            placement.enabled = true;
            placement.topology = RoutedExpertPlacementTopology::TieredOverlay;
            placement.continuation_domain = domain.name;
            placement.shared_expert_domain = domain.name;
            placement.domains = {domain};
            placement.routed_tiers = {tier};
            placement.placements = {{.layer = 0, .routed_expert_tier = {0, 0, 0, 0}}};
            MoEOverlayParticipantResidencyRegistry registry({
                .owner_map = MoEExpertOwnerMap::build(placement), .local_participant_ids = {0},
                .num_layers = 1, .num_experts = kExperts, .initial_epoch = 1,
                .collect_economy_service_measurements = true});
            ASSERT_TRUE(registry.registerInitialLayer(0, 0,
                std::vector<bool>(kExperts, true), std::vector<MoEOverlayPreparedExpertTriplet>(kExperts, source)));
            MoEOverlayPreparedWeightSource descriptor;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(source.gate, DeviceId::cpu(), descriptor));
            const MoEOverlayEconomyCalibrationLayerCatalog catalog({{
                .layer_idx = 0,
                .projections = {{
                    {.projection = ExpertTierWeightProjection::Gate, .N = kWidth, .K = kWidth, .format = descriptor.format},
                    {.projection = ExpertTierWeightProjection::Up, .N = kWidth, .K = kWidth, .format = descriptor.format},
                    {.projection = ExpertTierWeightProjection::Down, .N = kWidth, .K = kWidth, .format = descriptor.format}}}}});
            const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
            const MoEOverlayCPUServiceMeasurement::Geometry geometry{kWidth, kWidth};
            const size_t bytes = MoEOverlayCPUServiceMeasurement::allocationBytes(geometry);
            const auto make_memory = [](size_t admitted)
            {
                PhysicalMemoryBOMBuilder bom({.world_rank = 0, .device = DeviceId::cpu(),
                    .total_bytes = admitted, .admission_available_bytes = admitted});
                bom.add(PhysicalMemoryOwner::ExecutionWorkspace, admitted);
                PhysicalMemoryPlanBuilder plan;
                return std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.add(bom.build()).build()), 0);
            };
            const auto memory = make_memory(bytes);
            const auto measured = MoEOverlayCPUServiceMeasurement::measure(
                registry, 1, catalog, topology, geometry, memory);
            ASSERT_EQ(measured.size(), 1u);
            for (size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
            {
                EXPECT_GT(measured[0].total_nanoseconds[phase], 0u);
                EXPECT_GT(measured[0].activation_count[phase], 0u);
                EXPECT_GT(measured[0].sample_count[phase], 0u);
            }
            std::vector<MoEOverlayParticipantLayerServiceTotals> live;
            ASSERT_TRUE(registry.endpoint(0)->trySnapshotServiceMeasurements(&live));
            ASSERT_EQ(live.size(), 1u);
            EXPECT_EQ(live[0].total_nanoseconds, (std::array<uint64_t, 3>{}));
            EXPECT_EQ(live[0].activation_count, (std::array<uint64_t, 3>{}));
            EXPECT_EQ(live[0].sample_count, (std::array<uint64_t, 3>{}));
            EXPECT_TRUE(registry.endpoint(0)->acquire(1));
            EXPECT_FALSE(registry.endpoint(0)->acquire(2));
            EXPECT_TRUE(catalog.serviceEvidenceGaps(
                catalog.withPreparedServiceEvidence(live, measured, {0}, topology), {0}, topology).empty());
            EXPECT_EQ(memory->remainingAdmittedNewAllocationBytes(
                DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace), bytes);
            EXPECT_THROW((void)MoEOverlayCPUServiceMeasurement::measure(
                registry, 2, catalog, topology, geometry, memory), std::logic_error);
            // One byte less than even decode scratch rejects before any FFN.
            const auto insufficient = make_memory(1);
            EXPECT_THROW((void)MoEOverlayCPUServiceMeasurement::measure(
                registry, 1, catalog, topology, geometry, insufficient), std::exception);
            EXPECT_EQ(insufficient->remainingAdmittedNewAllocationBytes(
                DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace), 1u);
        }

        /** @brief Sweep exact formats and phases, retaining the original serving graph. */
        void prove(DeviceId device)
        {
            const auto body = [&]
            {
                void *serving_stream = nullptr;
                void *probe_stream = nullptr;
                if (device.is_gpu())
                {
                    auto &worker = GPUDeviceContextPool::instance().getContext(device);
                    serving_stream = worker.getOrCreateAuxiliaryStream("service_test_serving");
                    probe_stream = worker.getOrCreateAuxiliaryStream("service_test_probe");
                    ASSERT_NE(serving_stream, probe_stream);
                }
                std::vector<QuantizedVerifierWeightCreator> creators;
                for (const auto &format : quantizedMoEVerifierFormats()) creators.push_back(format.create);
                creators.push_back([](const auto &shape, auto seed) { return TestTensorFactory::createFP16Random(shape, -0.02f, 0.02f, seed); });
                creators.push_back([](const auto &shape, auto seed) { return TestTensorFactory::createBF16Random(shape, -0.02f, 0.02f, seed); });
                creators.push_back([](const auto &shape, auto seed) { return TestTensorFactory::createFP32Random(shape, -0.02f, 0.02f, seed); });
                for (const auto &create : creators)
                {
                    std::array<std::shared_ptr<TensorBase>, 3> weights;
                    for (size_t i = 0; i < weights.size(); ++i) weights[i] = create({kWidth, kWidth}, 71 + i);
                    SCOPED_TRACE(static_cast<int>(weights[0]->native_type()));
                    MoEOverlayPreparedExpertTriplet source{
                        prepare(weights[0], device, serving_stream),
                        prepare(weights[1], device, serving_stream),
                        prepare(weights[2], device, serving_stream)};
                    ASSERT_TRUE(source.complete());
                    for (const auto [phase, rows] : {
                             std::pair{MoEOverlayServicePhaseHint::Decode, 1},
                             std::pair{MoEOverlayServicePhaseHint::Prefill, 8},
                             std::pair{MoEOverlayServicePhaseHint::GroupedVerifier, 4}})
                    {
                        SCOPED_TRACE(rows);
                        ExpertExecution serving(device, serving_stream, rows, phase, source);
                        EXPECT_GT(serving.replay(), 0u);
                        const auto expected = serving.output();
                        for (float value : expected) ASSERT_TRUE(std::isfinite(value));
                        if (device.is_cpu()) proveProductionCPUMeasurement(source);
                        {
                            MoEOverlayPreparedExpertTriplet private_engines{
                                KernelFactory::createExpertServiceExecutionView(source.gate, device),
                                KernelFactory::createExpertServiceExecutionView(source.up, device),
                                KernelFactory::createExpertServiceExecutionView(source.down, device)};
                            ExpertExecution probe(device, probe_stream, rows, phase, std::move(private_engines));
                            EXPECT_GT(probe.replay(), 0u);
                            const auto observed = probe.output();
                            ASSERT_EQ(expected.size(), observed.size());
                            EXPECT_EQ(std::memcmp(expected.data(), observed.data(), expected.size() * sizeof(float)), 0);
                        }
                        // The probe's handles, graph and workspace are already
                        // gone. Replaying the retained graph proves they did not
                        // clear serving publications or leave dangling bindings.
                        EXPECT_GT(serving.replay(), 0u);
                        const auto after = serving.output();
                        EXPECT_EQ(std::memcmp(expected.data(), after.data(), expected.size() * sizeof(float)), 0);
                    }
                }
            };
            if (device.is_gpu()) GPUDeviceContextPool::instance().getContext(device).submitAndWait(body);
            else body();
        }
    }

    TEST(PreparedExpertServiceExecution, CPUAllFormats) { prove(DeviceId::cpu()); }
#ifdef HAVE_CUDA
    TEST(PreparedExpertServiceExecution, CUDAAllFormats) { prove(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
    TEST(PreparedExpertServiceExecution, ROCmAllFormats) { prove(DeviceId::rocm(0)); }
#endif
}
