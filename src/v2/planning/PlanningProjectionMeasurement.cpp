/**
 * @file PlanningProjectionMeasurement.cpp
 * @brief Ordinary production projections with exact native preparation and capture.
 *
 * The source owner encloses prepared kernels; kernels enclose serial phase
 * observations. Each phase retires its graph before unbinding its workspace,
 * and all physical payloads retain their canonical authority leases. Sampling
 * has no global kernel registry, model runner, routing policy or inference state.
 */
#include "PlanningProjectionMeasurement.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "kernels/rocm/gemm/ROCmFloatingPointGemmWorkspaceContract.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAFloatingPointGemmWorkspaceContract.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#endif
#include "loaders/GPUVramPreflight.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <omp.h>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        constexpr auto kWeightOwner = PhysicalMemoryOwner::PrimaryModelWeights;
        using llaminar::v2::kernels::KernelFactory;
#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr ISALevel kCompiledISA = ISALevel::AVX512;
#else
        constexpr ISALevel kCompiledISA = ISALevel::AVX2;
#endif

        /** @brief Reject failures without assigning a fabricated service rate. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::runtime_error(std::string("Planning projection: ") + detail);
        }

        /** @brief Prepared projection ABI geometry, deliberately independent of source provenance. */
        struct ProjectionGeometry final
        {
            size_t n, k, payload_bytes;
            TensorType type;
            std::string format, name;

            /** @return Canonical model-loader representation without changing the source contract. */
            static ProjectionGeometry model(const PlanningMatrixSamplePlan &plan)
            {
                return {plan.geometry().n, plan.geometry().k, plan.executionPayloadBytes(),
                    plan.executionType(), plan.executionFormat(), plan.request().tensor_name};
            }
            /** @return Source-free arithmetic geometry; this has no GGUF path or tensor identity. */
            static ProjectionGeometry arithmetic()
            {
                return {PlanningFP32ArithmeticPlan::kN, PlanningFP32ArithmeticPlan::kK,
                    size_t(PlanningFP32ArithmeticPlan::kN) * PlanningFP32ArithmeticPlan::kK * sizeof(float),
                    TensorType::FP32, "F32", "planning-fp32-arithmetic-basis"};
            }
        };

        /** @return Validated FP32 input/output bytes; never prepared-weight demand. */
        size_t tensorBytes(const ProjectionGeometry &shape, int rows)
        {
            if (rows <= 0 || !shape.n || !shape.k || shape.n > INT_MAX || shape.k > INT_MAX)
                throw std::invalid_argument("Planning projection requires positive representable M/N/K");
            const size_t columns = shape.n + shape.k;
            if (size_t(rows) > std::numeric_limits<size_t>::max() / sizeof(float) / columns)
                throw std::overflow_error("Planning projection activation extent overflow");
            return size_t(rows) * columns * sizeof(float);
        }

        /** @return True for the executed representation, including canonical source promotion. */
        bool floating(const ProjectionGeometry &plan)
        {
            return plan.type == TensorType::FP32 || plan.type == TensorType::FP16 || plan.type == TensorType::BF16;
        }

        /** @return Native CPU retained/temporary demand; ordinary floating kernels borrow source. */
        CPUWeightPreparationMemory cpuPreparation(const ProjectionGeometry &plan)
        {
            auto result = CPUWeightPreparationMemory::sourceNative(plan.format, plan.n, plan.k);
            if (floating(plan)) result.persistent_bytes = 0;
            return result;
        }

        /** @return The prepared engine's own named workspace ABI, before preparation. */
        WorkspaceRequirements workspaceRequirements(const ProjectionGeometry &plan, DeviceId device,
            int rows, const std::optional<PlanningProjectionCPUObservation> &cpu)
        {
            (void)tensorBytes(plan, rows);
            const auto &shape = plan;
            if (device.is_cpu())
            {
                if (!cpu || cpu->workers <= 0)
                    throw std::invalid_argument("CPU projection admission requires the observing workshare");
                return CPUProjectionWorkspaceContract::sourceNative(plan.format,
                    {.rows = rows, .n = int(shape.n), .k = int(shape.k), .workers = cpu->workers,
                     .execution = cpu->execution, .numerical_policy = CPUProjectionNumericalPolicy::BackendNative});
            }
            if (!device.is_gpu() || cpu)
                throw std::invalid_argument("GPU projection cannot inherit a CPU observer or unknown backend");
            if (!GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
                throw std::invalid_argument("GPU projection declaration requires its exact owning worker");
            if (device.is_rocm())
                return floating(plan) ? rocm::floating_gemm_workspace::projectionRequirements() :
                    rocm::quantized_gemm_workspace::projectionRequirements(rows, shape.n, shape.k);
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                if (floating(plan))
                    return cuda::floating_gemm_workspace::projectionRequirements(rows, shape.n);
                const auto *format = native_vnni_formats::forQuantType(plan.format);
                if (!format) throw std::invalid_argument("Unsupported CUDA projection source format");
                const uint8_t codebook = canonicalDeviceVnniCodebookId(format->codebook_id);
                return cuda::quantized_gemm_workspace::projectionRequirements(rows, shape.n, shape.k,
                    device.gpu_ordinal(), cuda::quantized_gemm_workspace::NativeCodebooks{codebook, codebook});
            }
#endif
            throw std::invalid_argument("Planning projection backend is not compiled");
        }

        /** @brief Let the production weight pool own aligned layout/accounting on GPU. */
        void planGPUWeight(LoadOrchestrator &pool, const ProjectionGeometry &plan, DeviceId device)
        {
            pool.addDevice(device.gpu_ordinal());
            if (const auto *format = native_vnni_formats::forQuantType(plan.format))
                pool.planWeightForOwner(device.gpu_ordinal(), plan.name, plan.n, plan.k,
                    format->payload_bytes, format->is_asymmetric, format->has_emins, plan.payload_bytes, kWeightOwner);
            else if (floating(plan))
                pool.planRawWeightForOwner(device.gpu_ordinal(), plan.name,
                    plan.n, plan.k, plan.payload_bytes, kWeightOwner);
            else throw std::invalid_argument("Unsupported native projection source format");
        }

        /** @brief Clear engine borrows after graph retirement and before workspace destruction. */
        class KernelBinding final
        {
        public:
            /** @brief Construct before binding so partial initialization is also unwound. */
            KernelBinding(ITensorGemm &kernel, IWorkspaceConsumer &consumer)
                : kernel_(kernel), consumer_(consumer) {}
            /** @brief No execution or synchronization is performed while retiring a borrow. */
            ~KernelBinding() { consumer_.unbindWorkspace(); kernel_.clearGPUStreamBinding(); }
        private:
            ITensorGemm &kernel_;
            IWorkspaceConsumer &consumer_;
        };

        /**
         * @brief Observe one ordinary projection, with graph lifetime inside its workspace.
         * @return Completed M-specific service; CPU and GPU use the same public GEMM ABI.
         */
        PlanningProjectionPhaseObservation measurePhase(const ProjectionGeometry &plan, DeviceId device,
            ITensorGemm &kernel, int rows, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const std::string &identity, size_t &graph_growth)
        {
            const auto &shape = plan;
            auto host_claim = memory->claimNewAllocation(DeviceId::cpu(),
                PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(plan, rows));
            std::optional<PhysicalMemoryAllocationLease> gpu_claim;
            if (device.is_gpu()) gpu_claim.emplace(memory->claimNewAllocation(device,
                PhysicalMemoryOwner::ExecutionWorkspace, tensorBytes(plan, rows)));
            FP32Tensor input({size_t(rows), shape.k}), output({size_t(rows), shape.n});
            for (size_t row = 0; row < size_t(rows); ++row)
                for (size_t col = 0; col < shape.k; ++col)
                    input.mutable_typed_data()[row * shape.k + col] =
                        0.0001f * float(int((col % 31 * 13 + row * 7) % 31) - 15);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&kernel);
            require(consumer != nullptr, "prepared projection has no workspace contract");
            const auto requirements = consumer->getWorkspaceRequirements(rows, shape.n, shape.k);
            DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment(), memory);
            require(workspace.allocate(requirements), "workspace materialization failed");
            KernelBinding binding(kernel, *consumer);
            consumer->bindWorkspace(&workspace);
            const auto execute = [&] {
                return kernel.multiply_tensor(&input, &output, rows, int(shape.n), int(shape.k),
                    floating(plan), 1.0f, 0.0f, nullptr, nullptr, device.toKernelDeviceIndex(), &workspace);
            };
            const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                2.0 * rows * shape.n * shape.k, identity + "; rows=" + std::to_string(rows));
            if (device.is_cpu()) return {rows, 0, PlanningExecutionMeasurement::cpu(work, execute)};

            auto &worker = GPUDeviceContextPool::instance().getContext(device);
            const ExplicitGPUStream stream(worker.defaultStream());
            kernel.bindGPUStream(stream);
            TransferEngine::prepareDeviceInput(&input, device, stream.get());
            TransferEngine::prepareDeviceOutput(&output, device, stream.get());
            TransferEngine::requireDeviceInput(&input, device, stream.get());
            auto graph = worker.createGraphCapture(stream.get());
            require(bool(graph), "graph creation failed");
            ScopedBackendGraphCapture capture(worker, *graph, "planning ordinary projection");
            require(capture.begin(), "graph begin failed");
            require(execute(), "captured projection failed");
            capture.finish();
            require(graph->nodeCount() && graph->instantiate(), "nonempty graph instantiation failed");
            if (graph->residentMemoryBytes() > std::numeric_limits<size_t>::max() - graph_growth)
                throw std::overflow_error("Planning projection graph-family growth overflow");
            graph_growth += graph->residentMemoryBytes();
            auto *backend = getBackendFor(device);
            require(backend != nullptr, "prepared graph backend missing");
            return {rows, graph->nodeCount(), PlanningExecutionMeasurement::gpu(work, *backend, device, *graph)};
        }
        /** @brief Contribute one prepared engine's demand; source ownership is supplied by its caller. */
        void contributePreparedMemory(const ProjectionGeometry &plan,
            size_t reader_bytes, PhysicalMemoryResource host, PhysicalMemoryResource execution,
            PhysicalMemoryPlanBuilder &builder, int prefill_rows,
            const std::optional<PlanningProjectionCPUObservation> &cpu)
        {
            const size_t tensors = tensorBytes(plan, prefill_rows);
            if (host.device != DeviceId::cpu() || host.world_rank != execution.world_rank ||
                (execution.device.is_cpu() && execution.device != host.device))
                throw std::invalid_argument("Projection sample requires same-rank staging");
            auto workspace = workspaceRequirements(plan, execution.device, 1, cpu);
            workspace.merge(workspaceRequirements(plan, execution.device, prefill_rows, cpu));
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, tensors);
            builder.add(execution, PhysicalMemoryOwner::ExecutionWorkspace, workspace.total_bytes_with_alignment());
            if (execution.device.is_cpu())
            {
                const auto packing = cpuPreparation(plan);
                builder.add(execution, kWeightOwner, packing.persistent_bytes);
                builder.add(host, PhysicalMemoryOwner::WeightLoadStaging, std::max(reader_bytes, packing.temporary_bytes));
                return;
            }
            LoadOrchestrator pool;
            planGPUWeight(pool, plan, execution.device);
            const auto staging = resolveGPUWeightLoadMemoryGeometry(plan.payload_bytes, {.staging_stream_count = 3});
            builder.add(execution, kWeightOwner, pool.plannedPersistentBytes(execution.device.gpu_ordinal(), kWeightOwner));
            builder.add(host, PhysicalMemoryOwner::WeightLoadStaging, std::max(reader_bytes, staging.host_staging_bytes));
            builder.add(execution, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes);
            builder.add(execution, PhysicalMemoryOwner::ExecutionWorkspace, tensors);
            builder.add(execution, PhysicalMemoryOwner::NativeGraphExecutable,
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(execution.device));
        }

        /** @brief Private result shared by source-backed and source-free preparation; no source claim is implied. */
        struct PreparedProjectionResult final
        {
            DeviceId device;
            size_t prepared_bytes;
            std::optional<PlanningProjectionCPUObservation> cpu;
            std::vector<PlanningProjectionPhaseObservation> phases;
        };

        /** @return Complete observations after the engine, retained graphs and every borrow retire. */
        PreparedProjectionResult measurePrepared(const ProjectionGeometry &plan, const TensorBase &execution,
            DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory, int prefill_rows, std::string identity)
        {
            (void)tensorBytes(plan, prefill_rows);
            if (!memory || (!device.is_cpu() && !device.is_gpu()))
                throw std::invalid_argument("Projection sampling requires admitted CPU/CUDA/ROCm execution");
            std::optional<PlanningProjectionCPUObservation> cpu;
            if (device.is_cpu())
            {
                if (device != DeviceId::cpu() || omp_in_parallel() || omp_get_dynamic() ||
                    omp_get_max_threads() <= 0 || omp_get_thread_limit() < omp_get_max_threads())
                    throw std::invalid_argument("CPU projection sampling requires its fixed, non-nested rank workshare");
                cpu = PlanningProjectionCPUObservation{
                    omp_get_max_threads(), activeISALevel(), kCompiledISA, CPUExecutionGeometry::local()};
            }
            else if (!GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
                throw std::invalid_argument("GPU projection sampling requires its exact owning worker");

            // Floating CPU engines borrow the runtime source; quantized CPU
            // engines own a separately admitted pack.
            std::optional<PhysicalMemoryAllocationLease> prepared_claim;
            std::shared_ptr<KernelFactory::PreparedGemmHandle> cpu_handle;
            std::shared_ptr<LoadOrchestrator> gpu_pool;
            std::shared_ptr<ITensorGemm> gpu_engine;
            ITensorGemm *engine = nullptr;
            size_t prepared_bytes = 0;
            if (device.is_cpu())
            {
                const auto packing = cpuPreparation(plan);
                prepared_bytes = packing.persistent_bytes;
                if (prepared_bytes) prepared_claim.emplace(memory->claimNewAllocation(device, kWeightOwner, prepared_bytes));
                std::optional<PhysicalMemoryAllocationLease> staging;
                if (packing.temporary_bytes) staging.emplace(memory->claimNewAllocation(device,
                    PhysicalMemoryOwner::WeightLoadStaging, packing.temporary_bytes));
                cpu_handle = KernelFactory::prepareGemmHandleLocal(&execution, device);
                engine = KernelFactory::getOrCreateGemmEngine(cpu_handle.get());
                require(engine && engine->hasWeights() && engine->packedWeightBytes() == prepared_bytes,
                    "CPU prepared payload differs from its admitted native representation");
            }
            else
            {
                auto *backend = getBackendFor(device);
                require(backend != nullptr, "GPU backend missing");
                gpu_pool = std::make_shared<LoadOrchestrator>(backend, memory, kWeightOwner);
                planGPUWeight(*gpu_pool, plan, device);
                gpu_pool->allocate(plan.payload_bytes, 3);
                const auto *format = native_vnni_formats::forQuantType(plan.format);
                const auto repack = format ? codebookIdToRepackFormat(format->codebook_id, format->is_superblock) :
                    std::optional<RepackFormat>(RepackFormat::RAW_FP);
                require(repack.has_value(), "source format has no native GPU repacker");
                gpu_pool->addWeightJob(device.gpu_ordinal(), {.name = plan.name,
                    .host_raw_data = execution.raw_data(), .raw_bytes = plan.payload_bytes,
                    .format = *repack, .N = int(plan.n), .K = int(plan.k),
                    .is_asymmetric = format && format->is_asymmetric});
                gpu_pool->load();
                gpu_pool->finalize();
                gpu_engine = KernelFactory::createGemmFromGPUWeightPool(execution, device, gpu_pool, plan.name);
                engine = gpu_engine.get();
                require(engine && engine->hasWeights(), "GPU prepared projection missing");
                prepared_bytes = gpu_pool->plannedPersistentBytes(device.gpu_ordinal(), kWeightOwner);
            }

            PreparedProjectionResult result{device, prepared_bytes, cpu, {}};
            identity += "; execution-format=" + plan.format +
                "; N=" + std::to_string(plan.n) + "; K=" + std::to_string(plan.k) +
                "; device=" + device.toString() + "; cache-regime=repeated-same-prepared-matrix; activations=FP32";
            if (cpu) identity += "; workers=" + std::to_string(cpu->workers) + "; ISA=" + isaLevelName(cpu->isa) +
                "; compiled-ISA=" + isaLevelName(cpu->compiled_isa) + "; numerical-policy=BackendNative";
            const size_t graph_bytes = device.is_gpu() ?
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device) : 0;
            std::optional<PhysicalMemoryOwnerReservation> graph_reservation;
            if (graph_bytes) graph_reservation.emplace(memory->reserveNewAllocations(
                device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes));
            size_t graph_growth = 0;
            for (int rows : {1, prefill_rows})
                result.phases.push_back(measurePhase(plan, device, *engine, rows, memory, identity, graph_growth));
            if (device.is_gpu() && !GPUGraphMemoryContract::acceptsFamilyObservation(device, graph_growth, graph_bytes))
                throw std::runtime_error("Projection graph family exceeded its certified physical admission");
            return result;
        }
    } // namespace

    void PlanningProjectionMeasurement::contributeMemory(const PlanningMatrixSamplePlan &plan,
        PlanningSampleOrigin origin, PhysicalMemoryResource host, PhysicalMemoryResource execution,
        PhysicalMemoryPlanBuilder &builder, int prefill_rows,
        const std::optional<PlanningProjectionCPUObservation> &cpu)
    {
        if (origin != PlanningSampleOrigin::LocalGGUF && origin != PlanningSampleOrigin::PublishedPayload)
            throw std::invalid_argument("Projection sample requires a known source origin");
        const auto geometry = ProjectionGeometry::model(plan);
        contributePreparedMemory(geometry, origin == PlanningSampleOrigin::LocalGGUF ? plan.geometry().source_bytes : 0,
            host, execution, builder, prefill_rows, cpu);
        builder.add(host, PhysicalMemoryOwner::ModelSourcePayload, plan.geometry().source_bytes);
        if (plan.executionType() != plan.tensorType()) builder.add(host, kWeightOwner, geometry.payload_bytes);
    }

    PlanningProjectionObservations PlanningProjectionMeasurement::measure(const PlanningLoadedMatrixSample &sample,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory, int prefill_rows)
    {
        if (!memory) throw std::invalid_argument("Projection sampling requires admitted physical memory");
        const auto &plan = sample.plan();
        const auto geometry = ProjectionGeometry::model(plan);
        // The model loader alone decides runtime promotion. Retain the native
        // source independently and retire the converted owner after all engine
        // borrowers return; neither conversion nor capture is timed arithmetic.
        std::optional<PhysicalMemoryAllocationLease> converted_claim;
        std::unique_ptr<FP32Tensor> converted;
        const TensorBase *execution = &sample.tensor();
        if (plan.executionType() != plan.tensorType())
        {
            converted_claim.emplace(memory->claimNewAllocation(DeviceId::cpu(), kWeightOwner, geometry.payload_bytes));
            converted = std::make_unique<FP32Tensor>(sample.tensor().shape(), DeviceId::cpu());
            sample.tensor().to_fp32(converted->mutable_typed_data());
            execution = converted.get();
        }
        auto observed = measurePrepared(geometry, *execution, device, memory, prefill_rows,
            "ordinary prepared projection; model=" + plan.modelPath() + "; source=" + plan.request().tensor_name +
            "; source-format=" + plan.format());
        return {plan, observed.device, observed.prepared_bytes, observed.cpu, std::move(observed.phases)};
    }

    void PlanningProjectionMeasurement::contributeMemory(PlanningFP32ArithmeticPlan,
        PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder,
        const std::optional<PlanningProjectionCPUObservation> &cpu)
    {
        const auto geometry = ProjectionGeometry::arithmetic();
        contributePreparedMemory(geometry, 0, host, execution, builder, PlanningFP32ArithmeticPlan::kPrefillRows, cpu);
        // Synthetic operands are setup workspace, never fictitious model bytes.
        builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, geometry.payload_bytes);
    }

    PlanningFP32ArithmeticObservations PlanningProjectionMeasurement::measure(PlanningFP32ArithmeticPlan,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!memory) throw std::invalid_argument("Arithmetic sampling requires admitted physical memory");
        const auto geometry = ProjectionGeometry::arithmetic();
        auto claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace, geometry.payload_bytes);
        FP32Tensor operand({geometry.n, geometry.k});
        for (size_t index = 0; index < geometry.n * geometry.k; ++index)
            operand.mutable_typed_data()[index] = 0.0001f * float(int((index * 13) % 31) - 15);
        auto observed = measurePrepared(geometry, operand, device, memory, PlanningFP32ArithmeticPlan::kPrefillRows,
            "source-free FP32 arithmetic proxy; not-model-weights; not-attention-or-GDN-timing");
        return {observed.device, observed.prepared_bytes, observed.cpu, std::move(observed.phases)};
    }
}
