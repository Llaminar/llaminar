/**
 * @file PlanningGPUExpertMeasurement.cpp
 * @brief Canonical GPU weight preparation and captured full-expert sampling.
 *
 * A single source triplet is prepared in the production GPU pool. The ordinary
 * MoE stage supplies the graph and exact runtime workspace; metadata admission
 * uses the same routed-participant workspace authority. Phase graphs execute
 * serially and release their private tensors/workspace before the next phase.
 * Pool-family admission spans all phases; native graph growth is attested as
 * an aggregate, never attributed to an individual graph from a free-byte delta.
 */
#include "PlanningGPUExpertMeasurement.h"
#include "PlanningExecutionMeasurement.h"
#include "WorkspaceMemoryEstimator.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "kernels/KernelFactory.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmWorkspaceContract.h"
#endif
#include "loaders/GPUVramPreflight.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        constexpr auto kWeightOwner = PhysicalMemoryOwner::RoutedExpertWeights;
        constexpr std::array kPhases{ExpertHistogramSource::DecodeToken,
            ExpertHistogramSource::PrefillChunk, ExpertHistogramSource::GroupedVerifier};

        /** @brief Fail closed at preparation boundaries; never substitute eager work. */
        void require(bool value, const char *operation)
        {
            if (!value) throw std::runtime_error(std::string("GPU expert sample: ") + operation);
        }

        /** @return Shared phase geometry, identical to the CPU expert witness. */
        int rowsFor(ExpertHistogramSource phase)
        {
            return MoEOverlayCPUServiceMeasurement::rowsForPhase(phase);
        }

        /** @return Largest serial-phase tensor envelope. */
        int maximumRows()
        {
            int rows = 0;
            for (auto phase : kPhases) rows = std::max(rows, rowsFor(phase));
            return rows;
        }

        /** @return Exact host/device FP32 input, routes, weights and output payload. */
        size_t tensorBytes(const PlanningExpertSampleDescription &description, int rows)
        {
            const size_t width = description.matrices[0].k;
            if (width > (std::numeric_limits<size_t>::max() / sizeof(float) / size_t(rows) - 2) / 2)
                throw std::overflow_error("GPU expert sample tensor extent overflow");
            return sizeof(float) * size_t(rows) * (2 * width + 2);
        }

        /** @brief Populate a logical pool using its own exact aligned allocation arithmetic. */
        void planWeights(LoadOrchestrator &pool, DeviceId device,
            const PlanningExpertSampleRequest &request, const PlanningExpertSampleDescription &description)
        {
            pool.addDevice(device.gpu_ordinal());
            const auto inputs = request.projections();
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                const auto &matrix = description.matrices[i];
                const auto *format = native_vnni_formats::forQuantType(description.formats[i]);
                if (format)
                    pool.planWeightForOwner(device.gpu_ordinal(), inputs[i]->tensor_name,
                        matrix.n, matrix.k, format->payload_bytes, format->is_asymmetric,
                        format->has_emins, matrix.source_bytes, kWeightOwner);
                else if (description.formats[i] == "F32" || description.formats[i] == "F16" ||
                         description.formats[i] == "BF16")
                    pool.planRawWeightForOwner(device.gpu_ordinal(), inputs[i]->tensor_name,
                        matrix.n, matrix.k, matrix.source_bytes, kWeightOwner);
                else throw std::invalid_argument("GPU expert sample has an unsupported native source format");
            }
        }

        /**
         * @return Canonical routed workspace contribution for this one expert.
         *
         * Strip unrelated layers/attention/terminal geometry. The retained
         * participant variants coincide for one route, so this is the same
         * production scratch ABI rather than a full-model workspace proxy.
         */
        size_t workspaceBytes(const PlanningExpertSamplePlan &plan,
            const PlanningExpertSampleRequest &request, const PlanningExpertSampleDescription &description,
            DeviceId device)
        {
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                WorkspaceRequirements requirements;
                const int n = description.matrices[0].n, k = description.matrices[0].k;
                // Each projection has its own source format and physical K.
                // Down cannot borrow gate's smaller quantized input bank.
                const auto projection = [&](size_t index, int rows)
                {
                    const auto &matrix = description.matrices[index];
                    const auto &format = description.formats[index];
                    if (const auto *native = native_vnni_formats::forQuantType(format))
                    {
                        const uint8_t codebook = canonicalDeviceVnniCodebookId(native->codebook_id);
                        return cuda::quantized_gemm_workspace::projectionRequirements(
                            rows, matrix.n, matrix.k, device.gpu_ordinal(),
                            cuda::quantized_gemm_workspace::NativeCodebooks{codebook, codebook});
                    }
                    if (format == "F32" || format == "F16" || format == "BF16")
                        return cuda::floating_gemm_workspace::projectionRequirements(rows, matrix.n);
                    throw std::invalid_argument("GPU expert sample has an unsupported projection format");
                };
                for (auto phase : kPhases)
                {
                    const int rows = rowsFor(phase);
                    auto current = MoEWorkspaceBuffers::cudaMoE(rows, k, n, 1, 1);
                    current.merge(projection(0, rows));
                    if (native_vnni_formats::forQuantType(description.formats[0]))
                    {
                        const std::array columns{n, n};
                        cuda::quantized_gemm_workspace::appendFusedProjectionRequirements(current, rows, columns, k);
                    }
                    // The side-stream declaration reads the already merged
                    // gate partial descriptor. An empty temporary loses it.
                    addCudaConcurrentDecodeGemvSideStreamWorkspace(current, device, rows, 2);
                    current.merge(projection(2, rows));
                    requirements.merge(current);
                }
                return requirements.total_bytes_with_alignment();
            }
#endif
            ModelMemoryProfile profile;
            profile.architecture = plan.architecture();
            profile.n_layers = description.layer + 1;
            profile.d_model = description.matrices[0].k;
            profile.expert_feed_forward_length = description.matrices[0].n;
            profile.expert_count = profile.expert_used_count = 1;
            const auto inputs = request.projections();
            for (size_t i = 0; i < inputs.size(); ++i)
                profile.tensors.push_back({inputs[i]->tensor_name, description.matrices[i].source_bytes,
                    description.formats[i], description.matrices[i].n * description.matrices[i].k,
                    description.matrices[i].k, description.layer});
            return WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,
                {.device = device, .resident_graph_rows = maximumRows(),
                 .first_layer = description.layer, .last_layer = description.layer,
                 .apportioned_routed_experts = true});
        }

        /** @brief End all borrowed workspace bindings before their arena retires. */
        class StageBinding final
        {
        public:
            /** @brief Construct before binding, so partial binding failure also cleans up. */
            explicit StageBinding(MoEExpertComputeStage &stage) : stage_(stage) {}
            /** @brief Called after the retained graph is destroyed. */
            ~StageBinding() { stage_.unbindWorkspace(); }
        private:
            MoEExpertComputeStage &stage_;
        };

        /**
         * @brief Capture and observe one FFN phase with private, admitted storage.
         *
         * Declaration order is the lifecycle: graph, binding, stage, workspace,
         * tensors, leases retire in that order. Loaded/prepared weights remain
         * owned by the enclosing sample transaction throughout.
         */
        PlanningGPUExpertPhaseObservation measurePhase(DeviceId device, IWorkerGPUContext &worker,
            const PlanningExpertSampleDescription &description, ExpertHistogramSource phase,
            const MoEOverlayPreparedExpertTriplet &triplet, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const std::string &identity, size_t &pool_growth)
        {
            const int rows = rowsFor(phase);
            const int width = description.matrices[0].k, intermediate = description.matrices[0].n;
            auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                tensorBytes(description, rows));
            auto gpu_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace,
                tensorBytes(description, rows));
            FP32Tensor input({size_t(rows), size_t(width)}), indices({size_t(rows), 1}),
                weights({size_t(rows), 1}), output({size_t(rows), size_t(width)});
            input.setDebugName("planning_expert_input");
            indices.setDebugName("planning_expert_routes");
            weights.setDebugName("planning_expert_route_weights");
            output.setDebugName("planning_expert_output");
            for (int row = 0; row < rows; ++row)
            {
                // A private one-expert routing bank maps the requested source
                // expert to slot zero. The observation still names its source ID.
                indices.mutable_typed_data()[row] = 0;
                weights.mutable_typed_data()[row] = 1;
                for (int col = 0; col < width; ++col)
                    input.mutable_typed_data()[size_t(row) * width + col] =
                        0.0001f * float((col % 31 * 13 + row * 7) % 31 - 15);
            }
            void *stream = worker.defaultStream();
            const ExplicitGPUStream exact_stream(stream);
            for (auto *tensor : {&input, &indices, &weights})
                TransferEngine::prepareDeviceInput(tensor, device, exact_stream.get());
            TransferEngine::prepareDeviceOutput(&output, device, exact_stream.get());
            auto context = IDeviceContext::create(device);
            std::unique_ptr<DeviceWorkspaceManager> workspace;
            MoEExpertComputeStage::Params params;
            params.device_id = device;
            params.input = &input;
            params.output = &output;
            params.seq_len = rows;
            params.d_model = width;
            params.expert_intermediate = intermediate;
            params.num_experts = params.top_k = 1;
            params.layer_idx = description.layer;
            params.routing_indices = &indices;
            params.routing_weights = &weights;
            params.output_registered_in_arena = false;
            params.service_phase = phase == ExpertHistogramSource::DecodeToken ? MoEOverlayServicePhaseHint::Decode :
                phase == ExpertHistogramSource::PrefillChunk ? MoEOverlayServicePhaseHint::Prefill :
                MoEOverlayServicePhaseHint::GroupedVerifier;
            params.force_decode_equivalent_verifier_prefill = phase == ExpertHistogramSource::GroupedVerifier;
            params.require_device_routing_tensor_decode = true;
            params.expert_mask = {true};
            params.prepared_gate_gemm = {triplet.gate.get()};
            params.prepared_up_gemm = {triplet.up.get()};
            params.prepared_down_gemm = {triplet.down.get()};
            params.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
            MoEExpertComputeStage stage(std::move(params));
            StageBinding binding(stage);
            const auto requirements = stage.getWorkspaceRequirements(rows);
            workspace = std::make_unique<DeviceWorkspaceManager>(device,
                requirements.total_bytes_with_alignment(), memory);
            require(workspace->allocate(requirements), "workspace allocation failed");
            stage.bindWorkspace(workspace.get());
            require(stage.prepareGraphLaunch(context.get(), stream), "capture preparation failed");
            for (auto *tensor : {&input, &indices, &weights})
                TransferEngine::requireDeviceInput(tensor, device, stream);
            auto graph = worker.createGraphCapture(stream);
            require(bool(graph), "graph creation failed");
            ScopedBackendGraphCapture capture(worker, *graph, "planning source-backed expert sample");
            require(capture.begin(), "capture begin failed");
            require(stage.execute(context.get()), "captured FFN failed");
            capture.finish();
            require(graph->nodeCount() > 0 && graph->instantiate(), "nonempty graph instantiation failed");
            if (graph->residentMemoryBytes() > std::numeric_limits<size_t>::max() - pool_growth)
                throw std::overflow_error("GPU expert graph pool growth overflow");
            pool_growth += graph->residentMemoryBytes();
            auto *backend = getBackendFor(device);
            require(backend != nullptr, "backend missing");
            const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                6.0 * rows * width * intermediate, identity + "; rows=" + std::to_string(rows));
            return {phase, rows, graph->nodeCount(),
                PlanningExecutionMeasurement::gpu(work, *backend, device, *graph)};
        }
    }

    void PlanningGPUExpertMeasurement::contributeMemory(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, PhysicalMemoryResource host,
        PhysicalMemoryResource gpu, PhysicalMemoryPlanBuilder &builder)
    {
        contributeMemory(PlanningExpertSamplePlan::resolve(source, request), PlanningSampleOrigin::LocalGGUF,
            host, gpu, builder);
    }

    void PlanningGPUExpertMeasurement::contributeMemory(const PlanningExpertSamplePlan &sample_plan,
        PlanningSampleOrigin origin, PhysicalMemoryResource host, PhysicalMemoryResource gpu, PhysicalMemoryPlanBuilder &builder)
    {
        if (origin != PlanningSampleOrigin::LocalGGUF && origin != PlanningSampleOrigin::PublishedPayload)
            throw std::invalid_argument("Unknown GPU sample source materialization contract");
        if (host.device != DeviceId::cpu() || !gpu.device.is_gpu() || host.world_rank != gpu.world_rank)
            throw std::invalid_argument("GPU expert sample requires same-rank CPU staging and exact GPU resources");
        if (!GPUDeviceContextPool::instance().getContext(gpu.device).ownsCurrentThread())
            throw std::invalid_argument("GPU expert sample BOM must query launch policy on its owning device worker");
        const auto &description = sample_plan.description();
        const auto &request = sample_plan.request();
        LoadOrchestrator plan;
        planWeights(plan, gpu.device, request, description);
        const auto staging = resolveGPUWeightLoadMemoryGeometry(description.largest_source_bytes,
            {.staging_stream_count = 3});
        builder.add(host, PhysicalMemoryOwner::ModelSourcePayload, description.source_bytes);
        // Reader and upload staging do not overlap: load all source matrices
        // first, then allocate the upload ring. Tensor publication occurs after
        // finalize has retired that ring.
        builder.add(host, PhysicalMemoryOwner::WeightLoadStaging,
            std::max(origin == PlanningSampleOrigin::LocalGGUF ? description.largest_source_bytes : size_t{0},
                staging.host_staging_bytes));
        builder.add(gpu, kWeightOwner, plan.plannedPersistentBytes(gpu.device.gpu_ordinal(), kWeightOwner));
        builder.add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes);
        builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(description, maximumRows()));
        builder.add(gpu, PhysicalMemoryOwner::ExecutionWorkspace,
            workspaceBytes(sample_plan, request, description, gpu.device));
        builder.add(gpu, PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(description, maximumRows()));
        builder.add(gpu, PhysicalMemoryOwner::NativeGraphExecutable,
            kPhases.size() * GPUGraphMemoryContract::reservationBytesPerExecutable(gpu.device));
    }

    PlanningGPUExpertObservations PlanningGPUExpertMeasurement::measure(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_gpu() || !memory)
            throw std::invalid_argument("GPU expert sampling requires an admitted GPU device");
        if (!GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
            throw std::invalid_argument("GPU expert sampling must run on the exact device's owning worker");
        const auto sample = PlanningExpertSamplePlan::resolve(source, request).load(source, memory, DeviceId::cpu());
        return measure(sample, device, memory);
    }

    PlanningGPUExpertObservations PlanningGPUExpertMeasurement::measure(const PlanningLoadedExpertSample &sample,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_gpu() || !memory)
            throw std::invalid_argument("GPU expert sampling requires an admitted GPU device");
        const auto &description = sample.plan().description();
        const auto &request = sample.plan().request();
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        if (!worker.ownsCurrentThread())
            throw std::invalid_argument("GPU expert sampling must run on the exact device's owning worker");
        auto *backend = getBackendFor(device);
        require(backend != nullptr, "backend missing");
        const auto inputs = request.projections();
        auto pool = std::make_shared<LoadOrchestrator>(backend, memory, kWeightOwner);
        planWeights(*pool, device, request, description);
        pool->allocate(description.largest_source_bytes, 3);
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto *format = native_vnni_formats::forQuantType(description.formats[i]);
            const auto repack = format ? codebookIdToRepackFormat(format->codebook_id, format->is_superblock) :
                std::optional<RepackFormat>(RepackFormat::RAW_FP);
            require(repack.has_value(), "source format has no native GPU repacker");
            pool->addWeightJob(device.gpu_ordinal(), {.name = inputs[i]->tensor_name,
                .host_raw_data = sample.tensor(i).raw_data(), .raw_bytes = description.matrices[i].source_bytes,
                .format = *repack, .N = int(description.matrices[i].n), .K = int(description.matrices[i].k),
                .is_asymmetric = format && format->is_asymmetric});
        }
        pool->load();
        pool->finalize();
        std::array<std::shared_ptr<ITensorGemm>, 3> engines;
        for (size_t i = 0; i < engines.size(); ++i)
            engines[i] = llaminar::v2::kernels::KernelFactory::createGemmFromGPUWeightPool(
                sample.tensor(i), device, pool, inputs[i]->tensor_name);
        const MoEOverlayPreparedExpertTriplet triplet{engines[0], engines[1], engines[2]};
        const size_t family_bytes = kPhases.size() * GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        auto graph_reservation = memory->reserveNewAllocations(device,
            PhysicalMemoryOwner::NativeGraphExecutable, family_bytes);
        PlanningGPUExpertObservations result{device, description.matrices[0].k, description.matrices[0].n,
            pool->plannedPersistentBytes(device.gpu_ordinal(), kWeightOwner), description.formats, request, {}};
        const std::string identity = "prepared GPU expert; model=" + sample.plan().modelPath() + "; source=" +
            request.gate.tensor_name + "/" + request.up.tensor_name + "/" + request.down.tensor_name +
            "; expert=" + std::to_string(std::get<PlanningExpertMatrix>(request.gate.selection).index) +
            "; device=" + device.toString() + "; gate/up/down=" + description.formats[0] + "/" +
            description.formats[1] + "/" + description.formats[2] + "; input=" +
            std::to_string(result.input_width) + "; intermediate=" + std::to_string(result.intermediate_width) +
            "; cache-regime=repeated-same-prepared-expert";
        size_t pool_growth = 0;
        for (auto phase : kPhases)
            result.phases.push_back(measurePhase(device, worker, description, phase, triplet, memory, identity, pool_growth));
        if (!GPUGraphMemoryContract::acceptsFamilyObservation(device, pool_growth, family_bytes))
            throw std::runtime_error("GPU expert sample native graph family exceeded certified physical admission: observed=" +
                std::to_string(pool_growth) + " admitted=" + std::to_string(family_bytes));
        return result;
    }
}
