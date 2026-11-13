/**
 * @file PlanningGPUExpertMeasurement.cpp
 * @brief Canonical GPU weight preparation and captured movable-expert sampling.
 *
 * A single source payload is prepared in the production GPU pool. The ordinary
 * MoE stage supplies the graph and exact runtime workspace; metadata admission
 * uses the same routed-participant workspace authority. Phase graphs execute
 * serially and release their private tensors/workspace before the next phase.
 * Pool-family admission spans all phases; native graph growth is attested as
 * an aggregate, never attributed to an individual graph from a free-byte delta.
 * Startup samples bracket three complete expert invocations with captured
 * device clocks, excluding host event submission from the compute service.
 * Live GateUp maintenance measures only movable owner-local projections. Its
 * original fixed-down lease supplies policy metadata without down arithmetic.
 */
#include "PlanningGPUExpertMeasurement.h"
#include "PlanningExecutionMeasurement.h"
#include "WorkspaceMemoryEstimator.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "kernels/IMoEKernel.h"
#include "collective/ILocalTPContext.h"
#include "backends/HardwareInventory.h"
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmMoERouteOwnedTuning.h"
#endif
#include <cmath>
#include <chrono>
#include <thread>
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/moe/GPUExpertProjectionContract.h"
#include "kernels/KernelFactory.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmWorkspaceContract.h"
#endif
#include "loaders/GPUVramPreflight.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/Logger.h"
#include <algorithm>
#include <limits>
#include <type_traits>
#include <stdexcept>

namespace llaminar2
{
    using llaminar::v2::kernels::KernelFactory;
    namespace
    {
        constexpr auto kWeightOwner = PhysicalMemoryOwner::RoutedExpertWeights;
        constexpr auto kPhases=kPlanningExpertStartupPhases;

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
                const auto *format = native_quantised_formats::forQuantType(description.formats[i]);
                if (format)
                    pool.planWeightForOwner(device.gpu_ordinal(), inputs[i]->tensor_name,
                        matrix.n, matrix.k, format->payload_bytes, canonicalDeviceVnniFormat(*format).is_asymmetric,
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
                    if (const auto *native = native_quantised_formats::forQuantType(format))
                    {
                        const uint8_t codebook = canonicalDeviceVnniCodebookId(*native);
                        return cuda::quantized_gemm_workspace::projectionRequirements(
                            rows, matrix.n, matrix.k, device.gpu_ordinal(),
                            cuda::quantized_gemm_workspace::NativeCodebooks{codebook, canonicalDeviceVnniCodebookId(native->codebook_id)});
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
                    if (native_quantised_formats::forQuantType(description.formats[0]))
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
            return WorkspaceMemoryEstimator::estimateStartupExpertService(profile,
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
         * @brief One source-owned production stage binding for planning and resident observations.
         * @param device Original GPU. @param layer Original model layer.
         * @param phase Original semantic phase. @param rows Actual bounded phase rows.
         * @param width Full expert width. @param intermediate Full intermediate width.
         * @param input Private input. @param indices Private one-expert route indices.
         * @param weights Private route weights. @param output Private FFN result.
         * @param payload Original prepared execution views, with exclusive mutable workspace bindings.
         * @return Same complete production stage parameters for both setup callers.
         */
        MoEExpertComputeStage::Params preparedStageParams(DeviceId device, int layer,
            ExpertHistogramSource phase, int rows, int width, int intermediate,
            FP32Tensor &input, FP32Tensor &indices, FP32Tensor &weights, FP32Tensor &output,
            const MoEOverlayPreparedExpertPayload &payload)
        {
            MoEExpertComputeStage::Params params;
            params.device_id = device;
            params.input = &input;
            params.output = &output;
            params.seq_len = rows;
            params.d_model = width;
            params.expert_intermediate = intermediate;
            params.num_experts = params.top_k = 1;
            params.layer_idx = layer;
            params.routing_indices = &indices;
            params.routing_weights = &weights;
            params.output_registered_in_arena = false;
            params.service_phase = phase == ExpertHistogramSource::DecodeToken ? MoEOverlayServicePhaseHint::Decode :
                phase == ExpertHistogramSource::PrefillChunk ? MoEOverlayServicePhaseHint::Prefill :
                MoEOverlayServicePhaseHint::GroupedVerifier;
            params.force_decode_equivalent_verifier_prefill = phase == ExpertHistogramSource::GroupedVerifier;
            params.require_device_routing_tensor_decode = true;
            params.expert_mask = {true};
            params.prepared_gate_gemm = {payload.gate().get()};
            params.prepared_up_gemm = {payload.up().get()};
            params.prepared_down_gemm = {payload.down().get()};
            params.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
            return params;
        }

        /**
         * @brief Record the identical original FFN body through the supplied owner's facade.
         * @param stage Prepared production expert stage. @param context Exact original device context.
         * @param worker Original worker. @param device Original physical GPU.
         * @param stream Exact original stream.
         * @param graph Actual raw planning capture or original-owned recording facade.
         * @param inputs Same three prepared device inputs.
         * @return Actual ended recording's kernel/node population, before materialization.
         * @throws std::exception On missing inputs or capture failure; never eager replay.
         */
        size_t recordPreparedPhase(MoEExpertComputeStage &stage, IDeviceContext *context,
            IWorkerGPUContext &worker, DeviceId device, void *stream, IGPUGraphCapture &graph,
            std::array<FP32Tensor *, 3> inputs)
        {
            const auto preparation_begin=std::chrono::steady_clock::now();
            const size_t rows=inputs[0]->rows();
            require(stage.prepareGraphLaunch(context, stream), "capture preparation failed");
            for (auto *tensor : inputs) TransferEngine::requireDeviceInput(tensor, device, stream);
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " rows=" << rows
                << " phase=stage_capture_prepare seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-preparation_begin).count());
            const auto record_begin=std::chrono::steady_clock::now();
            ScopedBackendGraphCapture capture(worker, graph, "prepared original expert service");
            require(capture.begin(), "capture begin failed");
            require(stage.execute(context), "captured FFN failed");
            capture.finish();
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " rows=" << rows
                << " phase=graph_record seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-record_begin).count());
            require(graph.nodeCount() > 0, "expert recording is empty");
            return graph.nodeCount();
        }

        /**
         * @brief Materialize a resident service's original recording through its owning facade.
         * @param graph Exact original resident graph, with its native lifecycle authority.
         * @param memory Original authority admitting the resident service.
         * @param device Physical GPU retained by that authority.
         * @param rows Actual service row count for lifecycle diagnostics.
         * @note Startup measurements instead materialize their complete clocked transaction.
         */
        void materializePreparedPhase(IGPUGraphCapture &graph,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId device, size_t rows)
        {
            const auto instantiate_begin=std::chrono::steady_clock::now();
            require(graph.prepareRuntimeContextStorage(memory), "expert context storage preparation failed");
            const auto nodes = graph.nodeCount();
            require(nodes > 0 && graph.instantiate(), "nonempty graph instantiation failed");
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " rows=" << rows
                << " phase=graph_storage_instantiate seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-instantiate_begin).count());
        }

        /**
         * @brief Capture and observe one FFN phase with private, admitted storage.
         *
         * Declaration order is the lifecycle: graph, binding, stage, workspace,
         * tensors, leases retire in that order. Loaded/prepared weights remain
         * owned by the enclosing sample transaction throughout.
         */
        PlanningGPUExpertPhaseObservation measurePhase(DeviceId device, IWorkerGPUContext &worker,
            const PlanningExpertSampleDescription &description, ExpertHistogramSource phase,
            const MoEOverlayPreparedExpertPayload &payload, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const std::string &identity, size_t &pool_growth)
        {
            const auto phase_begin=std::chrono::steady_clock::now();
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
            auto params = preparedStageParams(device, description.layer, phase, rows, width, intermediate,
                input, indices, weights, output, payload);
            MoEExpertComputeStage stage(std::move(params));
            StageBinding binding(stage);
            const auto requirements = stage.getWorkspaceRequirements(rows);
            workspace = std::make_unique<DeviceWorkspaceManager>(device,
                requirements.total_bytes_with_alignment(), memory);
            require(workspace->allocate(requirements), "workspace allocation failed");
            stage.bindWorkspace(workspace.get());
            auto graph = worker.createGraphCapture(stream);
            require(bool(graph), "graph creation failed");
            auto clock = std::make_unique<PlanningNativeGPUClockBuffer>(worker, device, exact_stream, 2, memory);
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << description.layer << " rows=" << rows
                << " phase=phase_storage_stage_prepare seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_begin).count());
            (void)recordPreparedPhase(stage, context.get(), worker, device, stream, *graph,
                {&input, &indices, &weights});
            const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                6.0 * rows * width * intermediate, identity + "; rows=" + std::to_string(rows));
            auto timing = PlanningExecutionMeasurement::recordGPU(work, worker, std::move(clock), std::move(graph));
            const auto measurement_begin=std::chrono::steady_clock::now();
            timing.prepare(memory);
            if (timing.residentMemoryBytes() > std::numeric_limits<size_t>::max() - pool_growth)
                throw std::overflow_error("GPU expert graph pool growth overflow");
            pool_growth += timing.residentMemoryBytes();
            auto observed = timing.observe();
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << description.layer << " rows=" << rows
                << " phase=warmup_and_isolated_measure seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-measurement_begin).count()
                << " native_timed_seconds=" << observed.elapsedSeconds()
                << " before_retirement_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_begin).count());
            return {phase,rows,timing.graphNodeCount(),std::move(observed)};
        }
    }

    std::shared_ptr<RetainedParticipantGraphExecutable> PlanningGPUExpertMeasurement::beginPreparedCalibration(
        DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
        if(original.worker()->ownsCurrentThread())
            throw std::invalid_argument("Prepared calibration must retain its off-worker owner");
        void* stream=nullptr;
        original.worker()->submitAndWait([&]{stream=ExplicitGPUStream(original.worker()->defaultStream()).get();});
        return original.borrowPreparedCalibration(stream);
    }

    uint64_t PlanningGPUExpertMeasurement::measurePrepared(const MoEOverlayPreparedExpertPayload &payload,
        DeviceId device, int layer, ExpertHistogramSource phase, int width, int intermediate,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, size_t declared_workspace_bytes,
        const std::shared_ptr<RetainedParticipantGraphExecutable> &graph,
        std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixed_down, int source_expert,
        PlanningPreparedOwnerDispatch owner_dispatch)
    {
        if (!device.is_gpu() || !memory || !payload.ready() || layer < 0 || width <= 0 || intermediate <= 0)
            throw std::invalid_argument("Prepared GPU service requires original resident movable projection geometry");
        if (payload.complete() ? bool(fixed_down) || source_expert != -1
            : !fixed_down || fixed_down->device() != device || fixed_down->layer() != layer ||
              fixed_down->ownership().geometry().model_columns != width ||
              fixed_down->ownership().geometry().intermediate_columns != intermediate ||
              source_expert < 0 || size_t(source_expert) >= fixed_down->engines().size())
            throw std::invalid_argument("Prepared service fixed-down metadata does not match original movable family");
        const auto original = GPUDeviceContextPool::instance().retainExistingPreparedContext(device, memory);
        const auto worker = original.worker();
        if (worker->ownsCurrentThread())
            throw std::invalid_argument("Prepared GPU service requires its off-worker startup owner");
        static_assert(PlanningExecutionMeasurement::kTimedInvocations == MoEOverlayCPUServiceMeasurement::kMeasuredSamples);
        const int rows = rowsFor(phase);
        const auto layout = MoEOverlayCPUServiceMeasurement::gpuStorageLayout(
            {.d_model = width, .intermediate = intermediate}, rows);
        // Gate/up publishes intermediate workspace only; no route/output
        // device tensors are materialized for the singleton service probe.
        const size_t device_bytes = payload.complete() ? layout.deviceBytes()
            : layout.input.capacity() * sizeof(float);
        auto gpu_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace, device_bytes);
        auto input_storage=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),size_t(width)},layout.input.allocate(memory),DeviceId::cpu());
        auto indices_storage=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),1},layout.indices.allocate(memory),DeviceId::cpu());
        auto weights_storage=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),1},layout.weights.allocate(memory),DeviceId::cpu());
        auto output_storage=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),size_t(width)},layout.output.allocate(memory),DeviceId::cpu());
        auto& input=*input_storage;auto& indices=*indices_storage;auto& weights=*weights_storage;auto& output=*output_storage;
        for (int row = 0; row < rows; ++row)
        {
            indices.mutable_typed_data()[row] = 0;
            weights.mutable_typed_data()[row] = 1;
            for (int col = 0; col < width; ++col)
                input.mutable_typed_data()[size_t(row) * width + col] =
                    0.0001f * float((col % 31 * 13 + row * 7) % 31 - 15);
        }
        void *stream = nullptr;
        worker->submitAndWait([&] { stream = ExplicitGPUStream(worker->defaultStream()).get(); });
        if(!graph || graph->purpose_!=RetainedParticipantGraphExecutable::DefinitionPurpose::PreparedServiceCalibration ||
           graph->device_!=device || graph->stream_!=stream || graph->memory_.get()!=memory.get() ||
           graph->memory_.owner_before(memory) || memory.owner_before(graph->memory_) ||
           graph->phase_!=RetainedParticipantGraphExecutable::Phase::Fresh)
            throw std::invalid_argument("Prepared service requires its same fresh original cold calibration owner");
        graph->requireLiveOriginal();
        MoEOverlayPreparedExpertPayload engines;
        std::unique_ptr<IDeviceContext> context;
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::unique_ptr<MoEExpertComputeStage> stage;
        std::unique_ptr<IMoEKernel> gate_up_kernel;
        std::unique_ptr<StageBinding> binding;
        // Both prelaunch input copies and possible partial launches must retire
        // on the exact original stream before any physical/view owner is freed.
        const auto retire = [&](bool completed) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!graph->observeTerminal())
            {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Prepared expert original terminal exceeded 30s");
                std::this_thread::yield();
            }
            if(completed)graph->resetPreparedCalibration();
            else graph->retireDefinition();
            worker->submitAndWait([&] {
                binding.reset(); stage.reset(); gate_up_kernel.reset(); workspace.reset(); context.reset(); engines = {};
                // Tensor destruction retires its original completion event and
                // device copies on the same worker after the graph is gone.
                output_storage.reset();weights_storage.reset();indices_storage.reset();input_storage.reset();
            });
        };
        try
        {
            worker->submitAndWait([&] {
                const auto validate = [&](const std::shared_ptr<ITensorGemm> &engine, int n, int k) {
                    MoEOverlayPreparedWeightSource source;
                    std::string error;
                    if (!resolveMoEOverlayPreparedWeightSource(engine, device, source, &error))
                        throw std::invalid_argument("Prepared GPU service original source: " + error);
                    const bool packed = source.kind == MoEOverlayPreparedWeightSourceKind::GpuSeparatedNativeQuantised;
                    if ((packed ? source.gpu_packed.n : source.floating.n) != n ||
                        (packed ? source.gpu_packed.k : source.floating.k) != k)
                        throw std::invalid_argument("Prepared GPU service source projection geometry changed");
                };
                validate(payload.gate(), intermediate, width);
                validate(payload.up(), intermediate, width);
                if (payload.complete()) validate(payload.down(), width, intermediate);
                if (payload.complete()) engines = {KernelFactory::createExpertServiceExecutionView(payload.gate(), device),
                    KernelFactory::createExpertServiceExecutionView(payload.up(), device),
                    KernelFactory::createExpertServiceExecutionView(payload.down(), device)};
                TransferEngine::prepareDeviceInput(&input, device, stream);
                if (payload.complete())
                {
                    TransferEngine::prepareDeviceInput(&indices, device, stream);
                    TransferEngine::prepareDeviceInput(&weights, device, stream);
                    TransferEngine::prepareDeviceOutput(&output, device, stream);
                }
                context = IDeviceContext::create(device);
                if (!payload.complete())
                {
                    // The movable family owns gate/up only. Fixed down columns
                    // and exchanges do not change with reassignment, so their
                    // service must never enter the migration payoff.
                    DeviceMoEExpertDescriptor descriptor{};
                    require(exportDeviceMoEPreparedPayload(payload, descriptor), "resident gate/up source export failed");
                    gate_up_kernel = KernelFactory::createMoEKernel(device);
                    require(bool(gate_up_kernel), "gate/up kernel missing");
                    gate_up_kernel->setGPUStream(stream);
                    const auto requirements = device.is_cuda()
                        ? MoEWorkspaceBuffers::cudaMoE(rows, width, intermediate, 1, 1)
                        : MoEWorkspaceBuffers::rocmMoE(rows, width, intermediate, 1, 1);
                    if (requirements.total_bytes_with_alignment() > declared_workspace_bytes)
                        throw std::logic_error("Prepared gate/up service exceeds original declared source workspace");
                    workspace = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment(), memory);
                    require(workspace->allocate(requirements), "prepared gate/up workspace admission failed");
                    auto* consumer = dynamic_cast<IWorkspaceConsumer*>(gate_up_kernel.get());
                    require(consumer != nullptr, "gate/up kernel workspace contract missing");
                    consumer->bindWorkspace(workspace.get());
                    const int table = deviceMoEWeightFormatIsFloating(descriptor.weight_format)
                        ? gate_up_kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                            &descriptor.floating_gate, &descriptor.floating_up, descriptor.weight_format, 1, width, intermediate)
                        : gate_up_kernel->uploadGroupedExpertGateUpDescriptorTables(
                            &descriptor.gate, &descriptor.up, 1, width, intermediate,
                            MoEDecodeDescriptorSource::RuntimePlacementTable);
                    require(table >= 0, "prepared gate/up descriptor publication failed");
                    // Retain the genuine fixed column descriptor for the existing
                    // pair-policy key. No down source is copied or executed.
                    const auto down_table = fixed_down->exportDescriptorTable();
                    const int down_columns = fixed_down->ownership().projection(WeightRole::MoEExpertDown).rows;
                    const int down_id = std::visit([&](const auto& original_table) {
                        if constexpr (std::is_same_v<std::decay_t<decltype(original_table)>,
                            MoEOverlayFixedDownProjectionBank::FloatingDescriptorTable>)
                        {
                            if (original_table.format != descriptor.weight_format)
                                throw std::invalid_argument("Prepared gate/up and fixed down arithmetic family differ");
                            return gate_up_kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                                &original_table.experts.at(source_expert), original_table.format, 1, down_columns, intermediate);
                        }
                        else return gate_up_kernel->uploadGroupedExpertDownDescriptorTable(
                            &original_table.experts.at(source_expert), 1, down_columns, intermediate);
                    }, down_table);
                    require(down_id >= 0, "original fixed down policy metadata publication failed");
                    std::uint64_t owner_intent = 0;
#ifdef HAVE_ROCM
                    if (device.is_rocm() && rows == 1 && descriptor.weight_format == DeviceMoEWeightFormat::NativeQuantised)
                    {
                        if (!owner_dispatch.domain || owner_dispatch.top_k <= 0 || owner_dispatch.resident_sources.empty())
                            throw std::invalid_argument("Owner gate/up calibration requires original domain/operation/source facts");
                        const auto policy = rocm::compiledMoERouteOwnedModelPartitionPolicy();
                        const rocm::MoERouteOwnedModelOperation operation{
                            .shape = {.output_columns=intermediate,.reduction_columns=width},
                            .rows=rows,.experts=fixed_down->ownership().geometry().experts,.top_k=owner_dispatch.top_k,
                            .phase=phase == ExpertHistogramSource::DecodeToken ? MoEOverlayServicePhaseHint::Decode :
                                phase == ExpertHistogramSource::PrefillChunk ? MoEOverlayServicePhaseHint::Prefill : MoEOverlayServicePhaseHint::GroupedVerifier,
                            .has_live_rows=owner_dispatch.has_live_rows};
                        const auto* hardware = DeviceManager::instance().hardware();
                        const auto observed = hardware ? std::span<const ComputeDevice>{hardware->rocm_devices}
                            : std::span<const ComputeDevice>{};
                        NativeQuantisedExecutionFormatEnvelope envelope;
                        std::vector<NativeQuantisedSourceIdentity> sources;
                        const bool inspect = policy == rocm::MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes ||
                            (policy == rocm::MoERouteOwnedModelPartitionPolicy::Automatic &&
                             rocm::moeRouteOwnedModelMeasuredDomain(operation, owner_dispatch.domain->devices(),
                                owner_dispatch.domain->weights(),owner_dispatch.domain->backend(),observed));
                        if (inspect) for (const auto& resident : owner_dispatch.resident_sources)
                        {
                            if (resident.empty()) continue;
                            for (const auto& engine : {resident.gate(),resident.up()})
                            {
                                DeviceNativeQuantisedMatrixDesc source{};
                                if (!engine || !engine->inspectPreparedNativeQuantisedMatrixDesc(source) ||
                                    source.n != intermediate || source.k != width)
                                    throw std::invalid_argument("Owner service original native source geometry changed");
                                const NativeQuantisedSourceIdentity identity{source.source_codebook_id,
                                    source.source_is_superblock != 0,source.source_identity_present != 0};
                                if (!addNativeQuantisedExecutionFormat(envelope,source.codebook_id,identity,true))
                                    throw std::invalid_argument("Owner service native source envelope changed");
                                sources.push_back(identity);
                            }
                        }
                        const auto execution = rocm::selectMoERouteOwnedModelPartitionExecution(policy,operation,
                            owner_dispatch.domain->devices(),owner_dispatch.domain->weights(),owner_dispatch.domain->backend(),observed,sources);
                        if (execution == rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
                        {
                            const auto plan = rocm::planMoERouteOwnedGateUp(rocm::MoERouteOwnedGateUpGeometry::TN64,
                                envelope,device.rocm_ordinal(),execution,operation.shape);
                            if (policy == rocm::MoERouteOwnedModelPartitionPolicy::Automatic && plan.maximumKPartitions()!=16)
                                throw std::logic_error("Automatic owner service source partitions changed");
                            owner_intent=plan.identity();
                        }
                    }
#endif
                    gate_up_kernel->prepareOwnerGateUpCaptureBinding(table, rows, owner_intent);
                    TransferEngine::requireDeviceInput(&input,device,stream);
                    ScopedBackendGraphCapture capture(*worker, graph->capture(), "original resident gate/up service");
                    require(capture.begin(), "gate/up capture begin failed");
                    require(gate_up_kernel->prepareSharedExpertPrefillGroup(DeviceRowRange::fullyActive(rows)),
                        "gate/up grouping failed");
                    require(gate_up_kernel->executeGroupedPrefillProjection(&input, nullptr, table, down_id,
                        rows, width, intermediate, 1, 1, MoEPrefillProjectionExecution::gateUp(width)),
                        "captured resident gate/up failed");
                    capture.finish();
                    require(graph->capture().nodeCount() > 0, "gate/up recording is empty");
                    materializePreparedPhase(graph->capture(), memory, device, rows);
                    return;
                }
                stage = std::make_unique<MoEExpertComputeStage>(preparedStageParams(device, layer, phase, rows,
                    width, intermediate, input, indices, weights, output, engines));
                binding = std::make_unique<StageBinding>(*stage);
                const auto requirements = stage->getWorkspaceRequirements(rows);
                if (requirements.total_bytes_with_alignment() > declared_workspace_bytes)
                    throw std::logic_error("Prepared GPU service exceeds original declared source workspace");
                workspace = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment(), memory);
                require(workspace->allocate(requirements), "prepared resident workspace admission failed");
                stage->bindWorkspace(workspace.get());
                (void)recordPreparedPhase(*stage, context.get(), *worker, device, stream, graph->capture(),
                    {&input, &indices, &weights});
                materializePreparedPhase(graph->capture(), memory, device, rows);
            });
            auto *backend = getBackendFor(device);
            require(backend != nullptr, "prepared resident backend missing");
            const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                (payload.complete() ? 6.0 : 4.0) * rows * width * intermediate,
                std::string(payload.complete() ? "original resident complete expert; layer=" :
                    "original resident owner-local gate/up; fixed down excluded; layer=") + std::to_string(layer));
            const auto observation = PlanningExecutionMeasurement::gpu(work, *backend, device, *graph);
            const double nanoseconds = observation.elapsedSeconds() * 1.0e9;
            if (!std::isfinite(nanoseconds) || nanoseconds < 1.0 || nanoseconds >= double(UINT64_MAX))
                throw std::runtime_error("Prepared GPU service has no positive representable native timing");
            retire(true);
            return static_cast<uint64_t>(nanoseconds);
        }
        catch (...)
        {
            const auto first = std::current_exception();
            try { retire(false); } catch (...) { std::terminate(); } // Missing native proof is never recoverable teardown.
            std::rethrow_exception(first);
        }
    }

    void PlanningGPUExpertMeasurement::contributeMemory(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, PhysicalMemoryResource host,
        PhysicalMemoryResource gpu, PhysicalMemoryPlanBuilder &builder)
    {
        contributeMemory(PlanningExpertSamplePlan::resolve(source, request), PlanningSampleOrigin::LocalGGUF,
            host, gpu, builder);
    }

    void PlanningGPUExpertMeasurement::validateSourceArithmetic(
        const PlanningExpertSampleRequest &request, const PlanningExpertSampleDescription &description)
    {
        const std::array labels{"gate", "up", "down"};
        const auto sources = request.projections();
        std::vector<GpuExpertSlotPool::ProjectionSpec> projections;
        projections.reserve(labels.size());
        for (std::size_t index = 0; index < labels.size(); ++index)
        {
            const auto &matrix = description.matrices[index];
            if (matrix.n > std::numeric_limits<int>::max() || matrix.k > std::numeric_limits<int>::max())
                throw std::invalid_argument("GPU expert sample source geometry exceeds native indexing: " +
                    sources[index]->tensor_name);
            GpuExpertSlotPool::ProjectionSpec projection{
                .label = labels[index], .N = static_cast<int>(matrix.n), .K = static_cast<int>(matrix.k)};
            const auto &format = description.formats[index];
            if (const auto *native = native_quantised_formats::forQuantType(format))
            {
                projection.format = ExpertWeightFormat::nativeQuantised({native->codebook_id, native->is_superblock, true});
                projection.payload_bytes_per_block = native->payload_bytes;
                projection.codebook_id = canonicalDeviceVnniCodebookId(*native);
                projection.is_asymmetric = canonicalDeviceVnniFormat(*native).is_asymmetric;
                projection.has_emins = native->has_emins;
            }
            else if (format == "F32") projection.format = ExpertWeightFormat::floating(TensorType::FP32);
            else if (format == "F16") projection.format = ExpertWeightFormat::floating(TensorType::FP16);
            else if (format == "BF16") projection.format = ExpertWeightFormat::floating(TensorType::BF16);
            else throw std::invalid_argument("GPU expert sample has an uncatalogued source format: " +
                sources[index]->tensor_name + '=' + format);
            projections.push_back(std::move(projection));
        }
        validateGPUExpertProjectionContract(
            projections, DeviceMoEProjectionSet::CompleteExpert, description.layer);
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
        validateSourceArithmetic(request, description);
        LoadOrchestrator plan;
        planWeights(plan, gpu.device, request, description);
        const auto staging = resolveGPUWeightLoadMemoryGeometry(description.largest_source_bytes,
            {.staging_stream_count = 3});
        for (const auto &matrix : description.matrices)
            builder.add(host, PhysicalMemoryOwner::ModelSourcePayload, matrix.physicalBytes());
        // Native reads populate final sample storage directly. The existing
        // event-ordered upload ring remains the only host staging allocation.
        builder.add(host, PhysicalMemoryOwner::WeightLoadStaging,
            staging.host_staging_bytes);
        builder.add(gpu, kWeightOwner, plan.plannedPersistentBytes(gpu.device.gpu_ordinal(), kWeightOwner));
        builder.add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes);
        builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(description, maximumRows()));
        builder.add(gpu, PhysicalMemoryOwner::ExecutionWorkspace,
            workspaceBytes(sample_plan, request, description, gpu.device));
        builder.add(gpu, PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(description, maximumRows()));
        PlanningExecutionMeasurement::contributeGPUObservationMemory(host, gpu, builder);
    }

    PlanningGPUExpertObservations PlanningGPUExpertMeasurement::measure(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_gpu() || !memory)
            throw std::invalid_argument("GPU expert sampling requires an admitted GPU device");
        if (!GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
            throw std::invalid_argument("GPU expert sampling must run on the exact device's owning worker");
        const auto source_begin=std::chrono::steady_clock::now();
        const auto plan = PlanningExpertSamplePlan::resolve(source, request);
        validateSourceArithmetic(plan.request(), plan.description());
        const auto sample = plan.load(source, memory, DeviceId::cpu());
        LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << plan.description().layer
            << " phase=original_source_load seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-source_begin).count());
        return measure(sample,device,memory);
    }

    PlanningGPUExpertObservations PlanningGPUExpertMeasurement::measure(const PlanningLoadedExpertSample &sample,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_gpu() || !memory)
            throw std::invalid_argument("GPU expert sampling requires an admitted GPU device");
        const auto weight_begin=std::chrono::steady_clock::now();
        const auto &description = sample.plan().description();
        const auto &request = sample.plan().request();
        validateSourceArithmetic(request, description);
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
            const auto *format = native_quantised_formats::forQuantType(description.formats[i]);
            const auto repack = format ? codebookIdToRepackFormat(format->codebook_id, format->is_superblock) :
                std::optional<RepackFormat>(RepackFormat::RAW_FP);
            require(repack.has_value(), "source format has no native GPU repacker");
            pool->addWeightJob(device.gpu_ordinal(), {.name = inputs[i]->tensor_name,
                .host_raw_data = sample.tensor(i).raw_data(), .raw_bytes = description.matrices[i].source_bytes,
                .format = *repack, .N = int(description.matrices[i].n), .K = int(description.matrices[i].k),
                .is_asymmetric = format && canonicalDeviceVnniFormat(*format).is_asymmetric});
        }
        pool->load();
        pool->finalize();
        std::array<std::shared_ptr<ITensorGemm>, 3> engines;
        for (size_t i = 0; i < engines.size(); ++i)
            engines[i] = llaminar::v2::kernels::KernelFactory::createGemmFromGPUWeightPool(
                sample.tensor(i), device, pool, inputs[i]->tensor_name);
        const MoEOverlayPreparedExpertPayload payload{engines[0], engines[1], engines[2]};
        // One phase owns the body, two clock fragments and the timed parent.
        // The phases retire serially; family growth proof still sums both.
        const size_t observation_bytes = PlanningExecutionMeasurement::kPreparedGraphCount *
            GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        const size_t family_bytes = kPhases.size() * observation_bytes;
        auto graph_reservation = memory->reserveNewAllocations(device,
            PhysicalMemoryOwner::NativeGraphExecutable, observation_bytes);
        PlanningGPUExpertObservations result{device, description.matrices[0].k, description.matrices[0].n,
            pool->plannedPersistentBytes(device.gpu_ordinal(), kWeightOwner), description.formats, request, {}};
        const std::string identity = "prepared GPU expert; model=" + sample.plan().modelPath() + "; source=" +
            request.gate.tensor_name + "/" + request.up.tensor_name + "/" + request.down.tensor_name +
            "; expert=" + std::to_string(std::get<PlanningExpertMatrix>(request.gate.selection).index) +
            "; device=" + device.toString() + "; gate/up/down=" + description.formats[0] + "/" +
            description.formats[1] + "/" + description.formats[2] + "; input=" +
            std::to_string(result.input_width) + "; intermediate=" + std::to_string(result.intermediate_width) +
            "; cache-regime=repeated-same-prepared-expert";
        LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << description.layer
            << " formats=" << description.formats[0] << "/" << description.formats[1] << "/" << description.formats[2]
            << " phase=weight_pool_engine_prepare seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-weight_begin).count());
        size_t pool_growth = 0;
        for (auto phase : kPhases) {
            const auto phase_begin=std::chrono::steady_clock::now();
            result.phases.push_back(measurePhase(device,worker,description,phase,payload,memory,identity,pool_growth));
            // The call has returned: graph, binding, stage, workspace, tensors
            // and leases have all retired, unlike the preceding inner timer.
            LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << description.layer << " rows=" << rowsFor(phase)
                << " phase=phase_complete_after_retirement seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_begin).count());
        }
        if (!GPUGraphMemoryContract::acceptsFamilyObservation(device, pool_growth, family_bytes))
            throw std::runtime_error("GPU expert sample native graph family exceeded certified physical admission: observed=" +
                std::to_string(pool_growth) + " admitted=" + std::to_string(family_bytes));
        LOG_INFO("[PlanningExpertWall] device=" << device.toString() << " layer=" << description.layer
            << " phase=sample_complete_before_weight_retirement seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-weight_begin).count());
        return result;
    }
}
