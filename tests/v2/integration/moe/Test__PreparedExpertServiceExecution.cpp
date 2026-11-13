/**
 * @file Test__PreparedExpertServiceExecution.cpp
 * @brief Real FFN execution proves private service probes cannot rebind serving.
 *
 * A retained serving stage executes before and after a short-lived probe over
 * the same prepared expert bytes. The probe owns independent execution handles,
 * routing, activations and workspace. GPU observations come from captured
 * complete expert FFNs or declared gate/up projections, never three stand-alone
 * GEMM estimates. CPU uses the production transported-row grouped workspace.
 * Each native workspace is admitted from its real descriptor; the exact backing
 * and live PMA claim must agree. These are model-free ownership/capture
 * regressions, not economy benchmarks.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/NativeContextPreparation.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "planning/PlanningGPUExpertMeasurement.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/HostMemoryCapacity.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "loaders/ExpertGemmRegistry.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "tensors/TensorFactory.h"
#include "collective/LocalTPContext.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <thread>

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

        /**
         * @brief Prove setup/test work retired on its exact original worker stream.
         * @param worker Original device worker, currently executing this scope.
         * @param stream Exact nondefault original stream containing all preceding work.
         * @throws std::exception On invalid ownership, asynchronous native failure or 30-second timeout.
         */
        void requireTerminal(IWorkerGPUContext& worker,void* stream)
        {
            require(worker.ownsCurrentThread(),"prepared fixture requires original worker");
            const ExplicitGPUStream exact(stream);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(worker.queryStreamExecutionState(exact.get(),"original prepared service fixture terminal")!=GPUStreamExecutionState::Complete){
                if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Original prepared fixture terminal timeout");
                std::this_thread::yield();
            }
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
                    require(rocm::packNativeQuantised(weight.get(), *packed), "ROCm prepared native blocks");
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
            /**
             * @brief Bind one complete prepared expert without a live router or histogram.
             * @param device Exact participant device.
             * @param stream Exact original explicit stream, or null only for CPU.
             * @param rows Live input rows retained by this fixture.
             * @param phase Actual expert service phase.
             * @param engines Original complete prepared gate/up/down engines.
             * @param original_memory Optional original native source PMA; absent preserves legacy fixture admission.
             * @param original_graph Optional same-PMA General owner, retired off-worker before fixture destruction.
             * @throws std::exception For invalid roots, insufficient admission or failed preparation/capture.
             */
            ExpertExecution(DeviceId device, void *stream, int rows,
                            MoEOverlayServicePhaseHint phase,
                            MoEOverlayPreparedExpertPayload engines,
                            std::shared_ptr<PhysicalMemoryAuthority> original_memory = {},
                            std::shared_ptr<RetainedParticipantGraphExecutable> original_graph = {})
                : device_(device), stream_(stream), rows_(rows),
                  input_({static_cast<size_t>(rows), kWidth},
                      original_memory ? MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},rows).input.allocate(original_memory)
                                      : AlignedVector<float>(size_t(rows)*kWidth), DeviceId::cpu()),
                  indices_({static_cast<size_t>(rows), 1},
                      original_memory ? MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},rows).indices.allocate(original_memory)
                                      : AlignedVector<float>(size_t(rows)*1), DeviceId::cpu()),
                  weights_({static_cast<size_t>(rows), 1},
                      original_memory ? MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},rows).weights.allocate(original_memory)
                                      : AlignedVector<float>(size_t(rows)*1), DeviceId::cpu()),
                  output_({static_cast<size_t>(rows), kWidth},
                      original_memory ? MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},rows).output.allocate(original_memory)
                                      : AlignedVector<float>(size_t(rows)*kWidth), DeviceId::cpu()),
                  context_(IDeviceContext::create(device)), engines_(std::move(engines)), original_graph_(std::move(original_graph))
            {
                // A failed constructor unwinds member tensors before its
                // caller can intervene. Prove original input copies retired
                // first; the nested native-capture scope unwinds before this.
                struct ConstructionRetirement final {
                    std::function<void()> failed;
                    bool complete=false;
                    /** @brief Retire original partial setup before member storage can disappear. */
                    ~ConstructionRetirement()noexcept{if(!complete)try{failed();}catch(...){std::terminate();}}
                } construction{[&]{
                    if(original_memory && device.is_gpu()){
                        requireTerminal(GPUDeviceContextPool::instance().getContext(device),stream);
                        if(stage_)stage_->unbindWorkspace();
                    }
                }};
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
                p.prepared_gate_gemm[kExpert] = engines_.gate().get();
                p.prepared_up_gemm[kExpert] = engines_.up().get();
                p.prepared_down_gemm[kExpert] = engines_.down().get();
                p.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
                std::shared_ptr<PhysicalMemoryAuthority> workspace_memory=original_memory;
                if (device.is_cpu())
                {
                    p.cpu_router_q8_input_publication = CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
                    // Freeze the complete real stage declaration before either
                    // its supplied serial bundle or named invocation workspace
                    // is materialized. Both retain this one exact PMA issuer.
                    MoEExpertComputeStage declaration(p);
                    const auto requirements=declaration.getWorkspaceRequirements(rows);
                    const size_t bytes=requirements.total_bytes_with_alignment();
                    PhysicalMemoryPlanBuilder plan;
                    plan.add({0,device,bytes,bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
                    workspace_memory=std::make_shared<PhysicalMemoryAuthority>(
                        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
                    p.cpu_grouped_serial_workspace = std::make_shared<CPUGroupedMoESerialWorkspace>(
                        CPUGroupedMoESerialWorkspace::Config{
                            .row_capacity = static_cast<size_t>(rows), .d_model = kWidth,
                            .expert_intermediate = kWidth, .num_experts = kExperts,
                            .routing_top_k = 1, .debug_name = "prepared_service_execution_test",
                            .device_id=device,.physical_memory_authority=workspace_memory});
                }
                else
                {
                    p.require_device_routing_tensor_decode = true;
                    if(original_memory)gpu_tensor_claim_=original_memory->claimNewAllocation(device,
                        PhysicalMemoryOwner::ExecutionWorkspace,
                        MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},rows).deviceBytes());
                    for (auto *tensor : {&input_, &indices_, &weights_})
                        TransferEngine::prepareDeviceInput(tensor, device, stream);
                    TransferEngine::prepareDeviceOutput(&output_, device, stream);
                }
                stage_ = std::make_unique<MoEExpertComputeStage>(std::move(p));
                const auto requirements = stage_->getWorkspaceRequirements(rows);
                const size_t workspace_bytes=requirements.total_bytes_with_alignment();
                if (device.is_gpu())
                {
                    // Prepared engines author this backend's scratch capacity.
                    // Admit that named capacity before captured workspace
                    // construction, without borrowing another format's budget.
                    PhysicalMemoryPlanBuilder plan;
                    const size_t capacity=std::max(size_t{1},workspace_bytes);
                    plan.add({0,device,capacity,capacity},PhysicalMemoryOwner::ExecutionWorkspace,workspace_bytes);
                    if(!workspace_memory)workspace_memory=std::make_shared<PhysicalMemoryAuthority>(
                        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
                }
                const auto prior_workspace_live=workspace_memory->claimedBytes(device,
                    PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);
                workspace_ = std::make_unique<DeviceWorkspaceManager>(
                    device, requirements.total_bytes_with_alignment(), workspace_memory);
                require(workspace_->allocate(requirements), "private stage workspace allocation");
                if (device.is_gpu())
                {
                    // Descriptor capacity includes per-buffer tail alignment;
                    // the contiguous allocation need not retain unused tail
                    // padding. The live claim must match its actual backing.
                    const size_t physical_bytes=workspace_->allocatedPhysicalBytes();
                    require(physical_bytes<=workspace_bytes &&
                            (workspace_bytes==0 || physical_bytes>0),
                        "prepared GPU workspace fits its source descriptor capacity");
                    require(workspace_memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,
                        PhysicalMemoryMaterializationKind::NewAllocation)==prior_workspace_live+physical_bytes,
                        "prepared GPU workspace live claim matches its physical allocation");
                }
                stage_->bindWorkspace(workspace_.get());
                if (!device.is_gpu()){construction.complete=true;return;}
                require(stage_->prepareGraphLaunch(context_.get(), stream), "private stage capture preparation");
                // Preparation publishes H2D events. Admit those external
                // producer edges before recording this graph, as the ordinary
                // graph executor does; capture cannot create them retroactively.
                for (auto *tensor : {&input_, &indices_, &weights_})
                    TransferEngine::requireDeviceInput(tensor, device, stream);
                auto &worker = GPUDeviceContextPool::instance().getContext(device);
                if(!original_graph_)graph_ = worker.createGraphCapture(stream);
                auto& definition=original_graph_ ? RetainedParticipantGraphExecutableFixtureAccess::capture(original_graph_) : *graph_;
                ScopedBackendGraphCapture capture(worker, definition, "prepared expert service execution proof");
                require(capture.begin(), "private stage capture begin");
                require(stage_->execute(context_.get()), "private stage captured execute");
                capture.finish();
                require(definition.nodeCount() > 0, "nonempty captured expert FFN");
                if(original_graph_)require(definition.prepareRuntimeContextStorage(original_memory), "private stage native storage");
                require(definition.instantiate(), "private stage instantiate");
                construction.complete=true;
            }

            /** @brief Unbind original execution handles before their workspace member retires. */
            ~ExpertExecution(){if(stage_)stage_->unbindWorkspace();}

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

            /** @brief Execute the original retained serving stage without inventing a service timing. */
            void replayOriginal()
            {
                require(bool(original_graph_),"missing original serving graph");
                require(RetainedParticipantGraphExecutableFixtureAccess::submit(original_graph_,++submission_),"original serving replay");
                auto& worker=GPUDeviceContextPool::instance().getContext(device_);
                requireTerminal(worker,stream_);
                TransferEngine::publishDeviceWrite(&output_,device_,stream_);
            }

            /** @brief Retire the original graph off-worker before worker-owned stage/storage destruction. */
            void retireOriginalGraph()
            {
                if(!original_graph_)return;
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(original_graph_)){
                    if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Original serving owner retirement timeout");
                    std::this_thread::yield();
                }
                original_graph_.reset();
            }

        private:
            DeviceId device_;
            void *stream_;
            int rows_;
            PhysicalMemoryAllocationLease gpu_tensor_claim_;
            FP32Tensor input_, indices_, weights_, output_;
            std::unique_ptr<IDeviceContext> context_;
            std::unique_ptr<DeviceWorkspaceManager> workspace_;
            MoEOverlayPreparedExpertPayload engines_;
            std::unique_ptr<MoEExpertComputeStage> stage_;
            std::unique_ptr<IGPUGraphCapture> graph_;
            std::shared_ptr<RetainedParticipantGraphExecutable> original_graph_;
            uint64_t submission_=0;
        };

        /**
         * @brief Prove the production CPU producer prices a completely unrouted bank.
         *
         * The same caller-owned source engines remain bound to a serving stage.
         * PMA admits exactly the producer's declared scratch, and must show no
         * live probe payload after completion or a rejected invocation.
         */
        void proveProductionCPUMeasurement(const MoEOverlayPreparedExpertPayload &source)
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
                std::vector<bool>(kExperts, true), std::vector<MoEOverlayPreparedExpertPayload>(kExperts, source)));
            MoEOverlayPreparedWeightSource descriptor;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(source.gate(), DeviceId::cpu(), descriptor));
            const MoEOverlayEconomyCalibrationLayerCatalog catalog({{
                .layer_idx = 0,
                .projections = {{
                    {.projection = ExpertTierWeightProjection::Gate, .N = kWidth, .K = kWidth, .format = descriptor.format},
                    {.projection = ExpertTierWeightProjection::Up, .N = kWidth, .K = kWidth, .format = descriptor.format},
                    {.projection = ExpertTierWeightProjection::Down, .N = kWidth, .K = kWidth, .format = descriptor.format}}}}});
            const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
            const MoEOverlayCPUServiceMeasurement::Geometry geometry{kWidth, kWidth};
            constexpr int sample_rows = MoEOverlayCPUServiceMeasurement::kMaximumRows;
            auto invocation = cpuSwiGLUWorkspaceRequirements(sample_rows, kWidth);
            for (const auto &engine : {source.gate(), source.up(), source.down()})
                if (const auto *consumer = dynamic_cast<const IWorkspaceConsumer *>(engine.get()))
                    invocation.merge(consumer->getWorkspaceRequirements(sample_rows, kWidth, kWidth));
            const size_t bytes = MoEOverlayCPUServiceMeasurement::allocationBytes(
                geometry, invocation.total_bytes_with_alignment());
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

        /**
         * @brief Exercise actual original-PMA GPU startup calibration before any routed observation.
         * @param device Explicit CUDA/ROCm device selected by this native registration.
         * @param gate_up Also replay live gate/up calibration with authentic fixed down columns.
         * @param native_gate_up Use differing original native codebooks for gate/up and fixed down.
         * @throws std::exception For admission, capture, measurement or retirement failure.
         * The original resident FP32 weights and retained serving bindings stay
         * alive while all three private measurement phases execute. Adjacent
         * all-format tests retain their independent codebook/binding coverage.
         */
        void proveProductionGPUMeasurement(DeviceId device, bool gate_up = false, bool native_gate_up = false)
        {
            // Deterministic immutable fixture inputs are formed before native
            // source admission; final runtime source storage is admitted below.
            std::vector<std::unique_ptr<TensorBase>> native_fixture;
            std::vector<CPUStorageRoot<uint8_t>> native_roots;
            size_t native_host_bytes=0,native_device_bytes=0;
            if(native_gate_up) {
                // The current QFN first-layer pair is Q4_K/Q4_K/Q5_1.
                // Keep native source bytes; no conversion aliases the mixed down codebook.
                native_fixture.push_back(TestTensorFactory::createQ4_KRandom({kWidth,kWidth},171));
                native_fixture.push_back(TestTensorFactory::createQ4_KRandom({kWidth,kWidth},172));
                native_fixture.push_back(TestTensorFactory::createQ5_1Random({kWidth/2,kWidth},173));
                for(const auto& tensor:native_fixture) {
                    native_roots.emplace_back(tensor->size_bytes(),PhysicalMemoryOwner::ExecutionWorkspace,
                        DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized);
                    native_host_bytes+=native_roots.back().physicalBytes();
                    const auto* quant=dynamic_cast<const IINT8Unpackable*>(tensor.get());
                    require(quant && quant->vnniFormatInfo(),"native fixture source format");
                    const auto format=canonicalDeviceVnniFormat(*quant->vnniFormatInfo());
                    native_device_bytes+=size_t(tensor->rows())*(tensor->cols()/32)*
                        (format.payload_bytes_per_block+sizeof(uint16_t)+
                         (format.is_asymmetric?sizeof(uint16_t):0)+(format.has_emins?sizeof(uint32_t):0));
                }
            }
            const auto additional_device_bytes=native_gate_up?native_device_bytes:
                (gate_up?size_t(kWidth)*kWidth/2*sizeof(float):0);
            auto* backend=getBackendFor(device);
            require(backend!=nullptr,"prepared GPU native backend");
            const auto host=observeSystemMemoryCapacity();
            require(host.valid() && host.admission_available_bytes>0,"original host capacity observation");
            const auto total=backend->deviceMemoryTotal(device.ordinal),free=backend->deviceMemoryFree(device.ordinal);
            require(total>0 && free>0 && free<=total,"original GPU capacity observation");
            const std::array observations{PhysicalMemoryResource{0,DeviceId::cpu(),host.total_bytes,host.admission_available_bytes},
                PhysicalMemoryResource{0,device,total,free}};
            const std::array selected{device};
            auto source=NativeContextPreparation::create(observations,0,selected);
            std::optional<PreparedGPUContextUsePermit> original(source->acquire(device));
            bool source_retired=false;
            // Covers declaration/admission failures before any model allocation;
            // the main cleanup below retires all later native/storage owners first.
            struct OriginalSourceGuard {
                std::shared_ptr<NativeContextPreparation>& source;
                std::optional<PreparedGPUContextUsePermit>& original;
                bool& retired;
                /** @brief Preserve original physical retirement even when declaration/admission throws. */
                ~OriginalSourceGuard()noexcept{if(!retired)try{original.reset();source->retireReleasedContexts();}catch(...){std::terminate();}}
            } source_guard{source,original,source_retired};
            const auto memory=source->memory();
            const auto generation=original->generation();
            RankInventory inventory;inventory.rank=0;
            inventory.gpus.push_back({.type=device.type,.local_device_id=device.ordinal,
                .memory_bytes=total,.free_memory_bytes=free,
                .compute_units=int(original->provenance().originalBirthKey()->compute_units)});
            ModelMemoryProfile profile;
            profile.architecture="qwen35moe";profile.n_layers=1;profile.d_model=kWidth;
            profile.d_ff=kWidth;profile.n_heads=4;profile.n_kv_heads=2;profile.head_dim=64;
            profile.vocab_size=kWidth;profile.max_seq_len=8;
            profile.expert_count=kExperts;profile.expert_used_count=1;profile.expert_feed_forward_length=kWidth;
            for(const auto name:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"})
                profile.tensors.push_back({.name=std::string("blk.0.")+name,.native_bytes=size_t(kExperts)*kWidth*kWidth*sizeof(float),
                    .quant_type="F32",.elements=size_t(kExperts)*kWidth*kWidth,.K=kWidth,.layer_index=0});
            const WorkspaceMemoryGeometry geometry{.device=device,.device_compute_units=inventory.gpus.front().compute_units,
                .resident_graph_rows=8,.first_layer=0,.last_layer=1,.apportioned_routed_experts=true};
            auto probe_workspace=WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry);
            if(gate_up) {
                const auto requirements=device.is_cuda()?MoEWorkspaceBuffers::cudaMoE(8,kWidth,kWidth,1,1):
                    MoEWorkspaceBuffers::rocmMoE(8,kWidth,kWidth,1,1);
                probe_workspace=std::max(probe_workspace,requirements.total_bytes_with_alignment());
            }
            auto serving_geometry=geometry;serving_geometry.resident_graph_rows=1;
            const auto serving_workspace=WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,serving_geometry);
            auto probe=MoEOverlayCPUServiceMeasurement::gpuAllocationBytes({kWidth,kWidth},probe_workspace);
            if(gate_up) {
                // This admission has no private route/output GPU storage. A
                // complete-FFN allocation cannot borrow the serving graph's
                // already materialized bytes to masquerade as movable work.
                const auto layout=MoEOverlayCPUServiceMeasurement::gpuStorageLayout({kWidth,kWidth},8);
                probe.device_bytes=layout.input.capacity()*sizeof(float)+probe_workspace;
            }
            const auto serving=MoEOverlayCPUServiceMeasurement::gpuAllocationBytes({kWidth,kWidth},serving_workspace);
            const auto weight_root=CPUStorageRoot<float>(size_t(kWidth)*kWidth,PhysicalMemoryOwner::ExecutionWorkspace,DeviceId::cpu(),
                CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized);
            source->freezeForPlanning();
            PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
            const auto graphs=RetainedParticipantGraphExecutable::generalBOM(observations[0],observations[1]);
            const auto calibration=RetainedParticipantGraphExecutable::calibrationBOM(observations[0],observations[1]);
            for(const auto& row:graphs.resources())plan.add(row);
            for(const auto& row:calibration.resources())plan.add(row);
            plan.add(observations[0],PhysicalMemoryOwner::ExecutionWorkspace,
                3*weight_root.physicalBytes()+native_host_bytes+probe.host_bytes+serving.host_bytes);
            plan.add(observations[1],PhysicalMemoryOwner::ExecutionWorkspace,
                (3*size_t(kWidth)*kWidth*sizeof(float)+additional_device_bytes)+probe.device_bytes+serving.device_bytes);
            source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
            std::array<std::shared_ptr<TensorBase>,3> weights;
            std::array<std::vector<float>,3> original_bytes;
            std::array<const void*,3> original_device_pointers{};
            std::array<ITensorGemm*,3> original_engine_handles{};
            MoEOverlayPreparedExpertPayload resident;
            std::shared_ptr<TensorBase> fixed_weight;
            std::vector<std::shared_ptr<TensorBase>> native_weights;
            MoEOverlayPreparedExpertPayload movable;
            std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixed_bank;
            std::unique_ptr<ExpertExecution> serving_execution;
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
            PhysicalMemoryAllocationLease weight_gpu_claim;
            const auto cleanup=[&]{
                if(source_retired)return;
                original->worker()->submitAndWait([&]{requireTerminal(*original->worker(),original->worker()->defaultStream());});
                if(serving_execution)serving_execution->retireOriginalGraph();
                original->worker()->submitAndWait([&]{
                    serving_execution.reset();registry.reset();fixed_bank.reset();fixed_weight.reset();movable={};native_weights.clear();resident={};
                    for(auto& weight:weights)weight.reset();
                    weight_gpu_claim={};
                });
                original.reset();
                try{source->retireReleasedContexts();source_retired=true;}catch(...){std::terminate();}
            };
            try{
                original->worker()->submitAndWait([&]{
                    weight_gpu_claim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ExecutionWorkspace,
                        (3*size_t(kWidth)*kWidth*sizeof(float)+additional_device_bytes));
                    for(size_t projection=0;projection<weights.size();++projection){
                        auto tensor=std::make_shared<FP32Tensor>(std::vector<size_t>{kWidth,kWidth},weight_root.allocate(memory),DeviceId::cpu());
                        for(size_t i=0;i<size_t(kWidth)*kWidth;++i)tensor->mutable_typed_data()[i]=
                            0.0001f*float(int((i*13+projection*7)%31)-15);
                        original_bytes[projection].assign(tensor->typed_data(),tensor->typed_data()+size_t(kWidth)*kWidth);
                        weights[projection]=tensor;
                    }
                    auto* stream=original->worker()->defaultStream();
                    resident={prepare(weights[0],device,stream),prepare(weights[1],device,stream),prepare(weights[2],device,stream)};
                    if (gate_up) {
                        // The real down slice has local N=H/2 and full K=F;
                        // substituting the complete expert descriptor is invalid.
                        std::shared_ptr<ITensorGemm> down;
                        if(native_gate_up) {
                            for(size_t index=0;index<native_fixture.size();++index) {
                                auto storage=native_roots[index].allocate(memory);
                                std::memcpy(storage.data(),native_fixture[index]->raw_data(),native_fixture[index]->size_bytes());
                                native_weights.emplace_back(TensorFactory::adoptQuantizedFinalStorage(
                                    native_fixture[index]->native_type(),native_fixture[index]->shape(),std::move(storage),DeviceId::cpu()));
                            }
                            movable=MoEOverlayPreparedExpertPayload::gateUp(prepare(native_weights[0],device,stream),
                                prepare(native_weights[1],device,stream));
                            down=prepare(native_weights[2],device,stream);
                        } else {
                            fixed_weight=weights[2]->create_view({kWidth/2,kWidth},0);
                            require(bool(fixed_weight),"original fixed down column view");
                            down=prepare(fixed_weight,device,stream);
                            movable=MoEOverlayPreparedExpertPayload::gateUp(resident.gate(),resident.up());
                        }
                        const auto ownership=MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                            {1,kWidth,kWidth},0,2);
                        const MoEExpertOwnerParticipant participant{.participant_id=0,.domain_name="original_gpu_service",
                            .domain_participant_index=0,.device=device,.world_rank=0,.world_rank_known=true};
                        ExpertGemmRegistry bank_registry;
                        for(int expert=0;expert<1;++expert)
                            bank_registry.registerEngineForParticipant(participant.domain_name,device,0,0,0,expert,
                                ExpertGemmRegistry::WeightRole::DOWN,down.get(),down,ownership);
                        fixed_bank=MoEOverlayFixedDownProjectionBank::resolve(bank_registry,participant,0,ownership);
                        (void)fixed_bank->exportDescriptorTable();
                    }
                    const std::array engines{resident.gate(),resident.up(),resident.down()};
                    for(size_t i=0;i<engines.size();++i){
                        MoEOverlayPreparedWeightSource physical;
                        require(resolveMoEOverlayPreparedWeightSource(engines[i],device,physical),"prepared physical source before measurement");
                        original_device_pointers[i]=physical.floating.data;
                        original_engine_handles[i]=physical.engine.get();
                    }
                });
                auto graph=RetainedParticipantGraphExecutable::create(*original,original->worker()->defaultStream());
                original->worker()->submitAndWait([&]{
                    serving_execution=std::make_unique<ExpertExecution>(device,original->worker()->defaultStream(),1,
                        MoEOverlayServicePhaseHint::Decode,resident,memory,graph);
                    serving_execution->replayOriginal();
                });
                graph.reset(); // The retained serving fixture is now its sole graph owner.
                std::vector<float> expected;
                original->worker()->submitAndWait([&]{expected=serving_execution->output();});
                RoutedExpertDomain domain;domain.name="original_gpu_service";
                domain.scope=ExecutionDomainScope::SINGLE;
                domain.backend=device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
                domain.participants={device.is_cuda()?GlobalDeviceAddress::cuda(device.ordinal):GlobalDeviceAddress::rocm(device.ordinal)};
                domain.world_ranks={0};domain.owner_rank=0;
                RoutedExpertTier tier;tier.name="resident";tier.domain=domain.name;tier.fallback=true;
                MoERoutedExpertPlacementPlan placement;placement.enabled=true;
                placement.topology=RoutedExpertPlacementTopology::TieredOverlay;
                placement.continuation_domain=domain.name;placement.shared_expert_domain=domain.name;
                placement.domains={domain};placement.routed_tiers={tier};
                placement.placements={{.layer=0,.routed_expert_tier={0,0,0,0}}};
                registry=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map=MoEExpertOwnerMap::build(placement),.local_participant_ids={0},.num_layers=1,
                    .num_experts=kExperts,.initial_epoch=1,.collect_economy_service_measurements=true});
                require(registry->registerInitialLayer(0,0,std::vector<bool>(kExperts,true),
                    std::vector<MoEOverlayPreparedExpertPayload>(kExperts,resident)),"original resident bank publication");
                MoEOverlayPreparedWeightSource descriptor;
                require(resolveMoEOverlayPreparedWeightSource(resident.gate(),device,descriptor),"original GPU source view");
                const MoEOverlayEconomyCalibrationLayerCatalog catalog({{.layer_idx=0,.projections={{
                    {.projection=ExpertTierWeightProjection::Gate,.N=kWidth,.K=kWidth,.format=descriptor.format},
                    {.projection=ExpertTierWeightProjection::Up,.N=kWidth,.K=kWidth,.format=descriptor.format},
                    {.projection=ExpertTierWeightProjection::Down,.N=kWidth,.K=kWidth,.format=descriptor.format}}}}});
                const auto topology=ExpertHistogramProductionTopology::uniform(1,kAllExpertHistogramProductionSources);
                std::vector<MoEOverlayParticipantLayerServiceTotals> live;
                require(registry->endpoint(0)->trySnapshotServiceMeasurements(&live),"initial live service snapshot");
                require(live.size()==1,"exact initial live service row");
                EXPECT_EQ(live.front().sample_count,(std::array<uint64_t,3>{}));
                require(catalog.serviceEvidenceGaps(live,{0},topology).size()==3,"three initially missing priced phases");
                const auto host_live=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);
                const auto gpu_live=memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);
                const auto host_graph=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable);
                const auto gpu_graph=memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable);
                EXPECT_EQ(gpu_graph,GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
                EXPECT_THROW((void)memory->reserveNewAllocations(device,PhysicalMemoryOwner::NativeGraphExecutable,1),std::logic_error);
                const auto preparation_host=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable);
                const auto preparation_gpu=memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable);
                if (gate_up) {
                    const auto pair=movable;
                    // This component owns one original GPU and one real
                    // prepared expert. It does not advertise a measured TP4
                    // domain or duplicate sources to enter a model policy.
                    LocalTPContext owner_domain({GlobalDeviceAddress::fromLocalDeviceId(device)},
                        {1.0f},device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
                    const std::array original_sources{pair};
                    const PlanningPreparedOwnerDispatch dispatch{.domain=&owner_domain,.top_k=1,
                        .resident_sources=original_sources,.has_live_rows=false};
                    auto calibration=PlanningGPUExpertMeasurement::beginPreparedCalibration(device,memory);
                    const auto invoke=[&](const MoEOverlayPreparedExpertPayload& payload,
                        std::shared_ptr<const MoEOverlayFixedDownProjectionBank> bank,int expert,
                        int layer=0,int width=kWidth,int intermediate=kWidth) {
                        return PlanningGPUExpertMeasurement::measurePrepared(payload,device,layer,
                            ExpertHistogramSource::DecodeToken,width,intermediate,memory,probe_workspace,calibration,
                            std::move(bank),expert,dispatch);
                    };
                    EXPECT_THROW((void)invoke(pair,{},0),std::invalid_argument);
                    const auto foreign=device.is_cuda()?DeviceId::rocm(device.ordinal):DeviceId::cuda(device.ordinal);
                    EXPECT_THROW((void)PlanningGPUExpertMeasurement::measurePrepared(pair,foreign,0,
                        ExpertHistogramSource::DecodeToken,kWidth,kWidth,memory,probe_workspace,calibration,
                        fixed_bank,0,dispatch),std::invalid_argument);
                    EXPECT_THROW((void)invoke(pair,fixed_bank,-1),std::invalid_argument);
                    EXPECT_THROW((void)invoke(pair,fixed_bank,1),std::invalid_argument);
                    EXPECT_THROW((void)invoke(pair,fixed_bank,0,1),std::invalid_argument);
                    EXPECT_THROW((void)invoke(pair,fixed_bank,0,0,kWidth/2),std::invalid_argument);
                    EXPECT_THROW((void)invoke(pair,fixed_bank,0,0,kWidth,kWidth/2),std::invalid_argument);
                    EXPECT_THROW((void)invoke(resident,fixed_bank,0),std::invalid_argument);
                    // Same retained calibration definition is reset and reused,
                    // including a large-to-small replay, without changing weights.
                    for(int replay=0;replay<2;++replay)
                        for(const auto phase:{ExpertHistogramSource::PrefillChunk,
                            ExpertHistogramSource::GroupedVerifier,ExpertHistogramSource::DecodeToken})
                            EXPECT_GT(PlanningGPUExpertMeasurement::measurePrepared(pair,device,0,phase,
                                kWidth,kWidth,memory,probe_workspace,calibration,fixed_bank,0,dispatch),0u);
                    calibration.reset();
                }
                if(!gate_up) {
                    const auto measured=MoEOverlayCPUServiceMeasurement::measure(*registry,1,catalog,topology,{kWidth,kWidth},memory,&profile,&inventory);
                    require(measured.size()==1,"exact completed prepared service row");
                    for(size_t phase=0;phase<3;++phase){
                        EXPECT_GT(measured.front().total_nanoseconds[phase],0u);
                        EXPECT_EQ(measured.front().sample_count[phase],3u);
                        const auto source_phase=phase==expertHistogramProductionSourceIndex(ExpertHistogramSource::DecodeToken)?ExpertHistogramSource::DecodeToken:
                            phase==expertHistogramProductionSourceIndex(ExpertHistogramSource::PrefillChunk)?ExpertHistogramSource::PrefillChunk:ExpertHistogramSource::GroupedVerifier;
                        EXPECT_EQ(measured.front().activation_count[phase],uint64_t(MoEOverlayCPUServiceMeasurement::rowsForPhase(source_phase))*3);
                    }
                    EXPECT_TRUE(catalog.serviceEvidenceGaps(catalog.withPreparedServiceEvidence(live,measured,{0},topology),{0},topology).empty());
                }
                std::vector<MoEOverlayParticipantLayerServiceTotals> after_live;
                require(registry->endpoint(0)->trySnapshotServiceMeasurements(&after_live),"unchanged live service snapshot");
                require(after_live.size()==1,"unchanged live service geometry");
                EXPECT_EQ(after_live.front().sample_count,(std::array<uint64_t,3>{}));
                EXPECT_EQ(after_live.front().total_nanoseconds,(std::array<uint64_t,3>{}));
                EXPECT_EQ(after_live.front().activation_count,(std::array<uint64_t,3>{}));
                EXPECT_TRUE(registry->endpoint(0)->acquire(1));
                EXPECT_FALSE(registry->endpoint(0)->acquire(2));
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),host_live);
                EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),gpu_live);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),host_graph);
                EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable),gpu_graph);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable)-preparation_host,
                    GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(device));
                EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable)-preparation_gpu,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
                original->worker()->submitAndWait([&]{
                    serving_execution->replayOriginal();
                    const auto after=serving_execution->output();
                    require(expected.size()==after.size(),"retained serving output extent changed");
                    EXPECT_EQ(std::memcmp(expected.data(),after.data(),expected.size()*sizeof(float)),0);
                    MoEOverlayPreparedWeightSource after_view;
                    require(resolveMoEOverlayPreparedWeightSource(resident.gate(),device,after_view),"original GPU source after calibration");
                    EXPECT_EQ(after_view.engine,descriptor.engine);EXPECT_EQ(after_view.floating.data,descriptor.floating.data);
                    const std::array engines{resident.gate(),resident.up(),resident.down()};
                    for(size_t i=0;i<weights.size();++i){
                        MoEOverlayPreparedWeightSource physical;
                        require(resolveMoEOverlayPreparedWeightSource(engines[i],device,physical),"original physical weight source");
                        EXPECT_EQ(physical.engine.get(),original_engine_handles[i]);
                        EXPECT_EQ(physical.floating.data,original_device_pointers[i]);
                        EXPECT_EQ(physical.floating.bytes,original_bytes[i].size()*sizeof(float));
                        std::vector<float> downloaded(original_bytes[i].size());
                        // Test-only observation of the original physical GPU
                        // weights, independent of their still-valid CPU copy.
                        require(backend->deviceToHostOnStream(downloaded.data(),physical.floating.data,
                            downloaded.size()*sizeof(float),device.ordinal,original->worker()->defaultStream()),
                            "original GPU weight byte observation");
                        requireTerminal(*original->worker(),original->worker()->defaultStream());
                        EXPECT_EQ(std::memcmp(original_bytes[i].data(),downloaded.data(),downloaded.size()*sizeof(float)),0);
                    }
                });
                descriptor={}; // Borrowed view must not outlive the original physical cleanup.
                EXPECT_EQ(original->generation(),generation);
                cleanup();
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),0u);
                EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),0u);
            }catch(...){
                const auto first=std::current_exception();
                try{cleanup();}catch(...){std::terminate();}
                std::rethrow_exception(first);
            }
        }

        /** @brief Sweep exact formats and phases, retaining the original serving graph. */
        void prove(DeviceId device)
        {
            if (device.is_cpu() && !hasCPUBackend()) initCPUBackend(-1);
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
                    MoEOverlayPreparedExpertPayload source{
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
                            MoEOverlayPreparedExpertPayload private_engines{
                                KernelFactory::createExpertServiceExecutionView(source.gate(), device),
                                KernelFactory::createExpertServiceExecutionView(source.up(), device),
                                KernelFactory::createExpertServiceExecutionView(source.down(), device)};
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
    TEST(PreparedExpertServiceExecution, CUDAOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::cuda(0)); }
    TEST(PreparedExpertServiceExecution, CUDAGateUpOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::cuda(0),true); }
    TEST(PreparedExpertServiceExecution, CUDANativeGateUpOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::cuda(0),true,true); }
    TEST(PreparedExpertServiceExecution, CUDAAllFormats) { prove(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
    TEST(PreparedExpertServiceExecution, ROCmOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::rocm(0)); }
    TEST(PreparedExpertServiceExecution, ROCmGateUpOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::rocm(0),true); }
    TEST(PreparedExpertServiceExecution, ROCmNativeGateUpOriginalMeasurement) { proveProductionGPUMeasurement(DeviceId::rocm(0),true,true); }
    TEST(PreparedExpertServiceExecution, ROCmAllFormats) { prove(DeviceId::rocm(0)); }
#endif
}
