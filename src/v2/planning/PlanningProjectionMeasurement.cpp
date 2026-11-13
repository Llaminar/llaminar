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
#include "loaders/PreparedHalfWeights.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/CPUFeatures.h"
#include "utils/Logger.h"
#include <chrono>
#include <thread>
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
            FloatingActivationOperandPrecision operand_precision = FloatingActivationOperandPrecision::Native;

            /** @return Canonical model-loader representation without changing the source contract. */
            static ProjectionGeometry model(const PlanningMatrixSamplePlan &plan)
            {
                return {plan.geometry().n, plan.geometry().k, plan.executionPayloadBytes(),
                    plan.executionType(), plan.executionFormat(), plan.request().tensor_name, plan.operandPrecision()};
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
            if (floating(plan))
            {
                result.persistent_bytes = 0;
                result.persistent_payload_bytes = 0;
            }
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
                return floating(plan) ? rocm::floating_gemm_workspace::projectionRequirements(rows, shape.k, plan.operand_precision) :
                    rocm::quantized_gemm_workspace::projectionRequirements(rows, shape.n, shape.k);
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                if (floating(plan))
                    return cuda::floating_gemm_workspace::projectionRequirements(rows, shape.n, shape.k, plan.operand_precision);
                const auto *format = native_quantised_formats::forQuantType(plan.format);
                if (!format) throw std::invalid_argument("Unsupported CUDA projection source format");
                const uint8_t codebook = canonicalDeviceVnniCodebookId(*format);
                return cuda::quantized_gemm_workspace::projectionRequirements(rows, shape.n, shape.k,
                    device.gpu_ordinal(), cuda::quantized_gemm_workspace::NativeCodebooks{codebook, canonicalDeviceVnniCodebookId(format->codebook_id)});
            }
#endif
            throw std::invalid_argument("Planning projection backend is not compiled");
        }

        /** @brief Let the production weight pool own aligned layout/accounting on GPU. */
        void planGPUWeight(LoadOrchestrator &pool, const ProjectionGeometry &plan, DeviceId device)
        {
            pool.addDevice(device.gpu_ordinal());
            if (const auto *format = native_quantised_formats::forQuantType(plan.format))
                pool.planWeightForOwner(device.gpu_ordinal(), plan.name, plan.n, plan.k,
                    format->payload_bytes, canonicalDeviceVnniFormat(*format).is_asymmetric, format->has_emins, plan.payload_bytes, kWeightOwner);
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
         * @brief Observe an ordinary CPU projection using its canonical workspace and workshare.
         * @return Completed CPU M-specific service; GPU observations belong to the staged owner.
         */
        PlanningProjectionPhaseObservation measureCPUPhase(const ProjectionGeometry &plan, DeviceId device,
            ITensorGemm &kernel, int rows, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const std::string &identity)
        {
            const auto &shape = plan;
            const auto allocate = [&](size_t count) {
                return CPUStorageRoot<float>(count, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu(),
                    CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized).allocate(memory);
            };
            FP32Tensor input({size_t(rows), shape.k}, allocate(size_t(rows) * shape.k), DeviceId::cpu());
            FP32Tensor output({size_t(rows), shape.n}, allocate(size_t(rows) * shape.n), DeviceId::cpu());
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
            return {rows,0,PlanningExecutionMeasurement::cpu(work,execute)};
        }
        /** @brief Contribute one prepared engine's demand; source ownership is supplied by its caller. */
        void contributePreparedMemory(const ProjectionGeometry &plan,
            PhysicalMemoryResource host, PhysicalMemoryResource execution,
            PhysicalMemoryPlanBuilder &builder, int prefill_rows,
            const std::optional<PlanningProjectionCPUObservation> &cpu)
        {
            const size_t tensors = tensorBytes(plan, prefill_rows);
            if (host.device != DeviceId::cpu() || host.world_rank != execution.world_rank ||
                (execution.device.is_cpu() && execution.device != host.device))
                throw std::invalid_argument("Projection sample requires same-rank staging");
            auto workspace = workspaceRequirements(plan, execution.device, 1, cpu);
            workspace.merge(workspaceRequirements(plan, execution.device, prefill_rows, cpu));
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace,
                CPUStoragePlacement::local().physicalBytes<float>(size_t(prefill_rows) * plan.k));
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace,
                CPUStoragePlacement::local().physicalBytes<float>(size_t(prefill_rows) * plan.n));
            builder.add(execution, PhysicalMemoryOwner::ExecutionWorkspace, workspace.total_bytes_with_alignment());
            if (execution.device.is_cpu())
            {
                const auto packing = cpuPreparation(plan);
                builder.add(execution, kWeightOwner, packing.persistent_bytes);
                builder.add(host, PhysicalMemoryOwner::WeightLoadStaging, packing.temporary_bytes);
                return;
            }
            LoadOrchestrator pool;
            planGPUWeight(pool, plan, execution.device);
            const auto staging = resolveGPUWeightLoadMemoryGeometry(plan.payload_bytes, {.staging_stream_count = 3});
            builder.add(execution, kWeightOwner, pool.plannedPersistentBytes(execution.device.gpu_ordinal(), kWeightOwner));
            builder.add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes);
            builder.add(execution, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes);
            builder.add(execution, PhysicalMemoryOwner::ExecutionWorkspace, tensors);
            PlanningExecutionMeasurement::contributeGPUObservationMemory(host,execution,builder);
        }

        /** @brief CPU-only preparation result; all GPU owners use the single staged producer. */
        struct PreparedCPUProjectionResult final
        {
            DeviceId device;
            size_t prepared_bytes;
            std::optional<PlanningProjectionCPUObservation> cpu;
            std::vector<PlanningProjectionPhaseObservation> phases;
        };

        /** @return Complete observations after the engine, retained graphs and every borrow retire. */
        PreparedCPUProjectionResult measureCPUPrepared(const ProjectionGeometry &plan,const TensorBase &execution,
            DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority> &memory,int prefill_rows,std::string identity)
        {
            const auto preparation_begin=std::chrono::steady_clock::now();
            (void)tensorBytes(plan,prefill_rows);
            if(!memory || device!=DeviceId::cpu() || omp_in_parallel() || omp_get_dynamic() ||
                omp_get_max_threads()<=0 || omp_get_thread_limit()<omp_get_max_threads())
                throw std::invalid_argument("CPU projection sampling requires its fixed nonnested rank workshare");
            const PlanningProjectionCPUObservation cpu{omp_get_max_threads(),activeISALevel(),kCompiledISA,CPUExecutionGeometry::local()};
            const auto packing=cpuPreparation(plan);
            auto handle=KernelFactory::prepareGemmHandleLocal(&execution,device,KernelFactory::GemmPreparationKind::AUTO,
                plan.operand_precision,CPUWeightStoragePlacement::accounted(memory,device,kWeightOwner,CPUWeightStoragePlacement::local()));
            auto *engine=KernelFactory::getOrCreateGemmEngine(handle.get());
            require(engine && engine->hasWeights() && engine->packedWeightBytes()==packing.persistent_payload_bytes,
                "CPU prepared payload differs from its admitted native representation");
            LOG_INFO("[PlanningProjectionWall] device=" << device.toString() << " source=" << plan.name
                << " format=" << plan.format << " N=" << plan.n << " K=" << plan.k << " phase=payload_engine_prepare seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-preparation_begin).count());
            PreparedCPUProjectionResult result{device,packing.persistent_bytes,cpu,{}};
            identity+="; execution-format="+plan.format+"; N="+std::to_string(plan.n)+"; K="+std::to_string(plan.k)+
                "; device="+device.toString()+"; cache-regime=repeated-same-prepared-matrix; activations=FP32; activation-operand-policy="+
                std::to_string(static_cast<int>(plan.operand_precision))+"; workers="+std::to_string(cpu.workers)+"; ISA="+
                isaLevelName(cpu.isa)+"; compiled-ISA="+isaLevelName(cpu.compiled_isa)+"; numerical-policy=BackendNative";
            for(int rows:{1,prefill_rows}){
                const auto begin=std::chrono::steady_clock::now();
                result.phases.push_back(measureCPUPhase(plan,device,*engine,rows,memory,identity));
                LOG_INFO("[PlanningProjectionWall] device=" << device.toString() << " source=" << plan.name
                    << " format=" << plan.format << " rows=" << rows << " phase=capture_measure_retire seconds="
                    << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
                    << " native_timed_seconds=" << result.phases.back().service.elapsedSeconds());
            }
            return result;
        }
        /** @brief Materialize only the source-selected execution conversion; the native loaded owner remains independent. */
        std::shared_ptr<TensorBase> convertProjectionSource(const PlanningLoadedMatrixSample &sample,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        {
            const auto &plan=sample.plan();
            if(plan.executionType()==plan.tensorType())return {};
            const auto placement=CPUWeightStoragePlacement::accounted(memory,sample.tensor().home_device(),kWeightOwner,
                CPUWeightStoragePlacement::local());
            if(plan.executionType()==TensorType::FP16)return materializePreparedHalfWeights(sample.tensor(),placement);
            auto fp32=std::make_shared<FP32Tensor>(sample.tensor().shape(),placement.allocate<float>(sample.tensor().numel()),placement.allocatorDevice());
            sample.tensor().to_fp32(fp32->mutable_typed_data());return fp32;
        }
        /** @brief A prepared producer may retain only its already-created original physical worker. */
        void requirePreparedProjectionWorker(DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        {
            if(!device.is_gpu() || !memory)throw std::invalid_argument("Prepared projection requires an admitted GPU endpoint");
            const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);
            auto &worker=*original.worker();
            if(!worker.ownsCurrentThread() || isGraphCaptureActive() || worker.isDeviceGraphCaptureActive())
                throw std::invalid_argument("Prepared projection construction requires its original worker outside capture");
        }

    } // namespace

    /** @brief Payload/engine lifetime spans both phases; phase storage and clock transaction tokens are worker-local. */
    struct PlanningPreparedGPUProjection::Impl
    {
        enum class PhaseState { StorageReady, Captured, Prepared, Failed };
        struct Phase {
            PhaseState state=PhaseState::StorageReady;
            PhysicalMemoryAllocationLease gpu_claim;
            std::unique_ptr<FP32Tensor> input,output;
            std::unique_ptr<DeviceWorkspaceManager> workspace;
            std::unique_ptr<PlanningGPUClockBuffer> clock;
            std::unique_ptr<IGPUGraphCapture> graph;
            std::optional<PlanningPreparedGPUObservation> timing;
            size_t graph_nodes=0;
            explicit Phase(PhysicalMemoryAllocationLease claim):gpu_claim(std::move(claim)){}
        };
        DeviceId device; std::shared_ptr<PhysicalMemoryAuthority> memory;
        GPUContextProvenance original;
        std::optional<PlanningLoadedMatrixSample> model_source;
        ProjectionGeometry shape;
        int prefill_rows;
        std::string identity;
        PhysicalMemoryAllocationLease source_claim;
        std::shared_ptr<TensorBase> execution_owner;
        std::shared_ptr<LoadOrchestrator> pool;
        std::unique_ptr<ITensorGemm> engine;
        PhysicalMemoryOwnerReservation graph_claim;
        std::unique_ptr<Phase> phase;
        std::vector<PlanningProjectionPhaseObservation> observations;
        std::size_t graph_growth=0; int next=0;
        void requireOwner() const {
            require(original.worker() && original.worker()->ownsCurrentThread() && original.worker()->isInitialized(),
                "Prepared projection requires its original owning GPU worker");
            require(!isGraphCaptureActive() && !original.worker()->isDeviceGraphCaptureActive(),
                "Prepared projection lifecycle must be outside capture");
        }
        Impl(DeviceId d,const std::shared_ptr<PhysicalMemoryAuthority> &m,
            std::optional<PlanningLoadedMatrixSample> source,int rows)
            :device(d),memory(m),original(GPUDeviceContextPool::instance().retainExistingContextProvenance(d)),
             model_source(std::move(source)),shape(model_source?ProjectionGeometry::model(model_source->plan()):ProjectionGeometry::arithmetic()),
             prefill_rows(rows),graph_claim(m->reserveNewAllocations(d,PhysicalMemoryOwner::NativeGraphExecutable,
                PlanningExecutionMeasurement::kPreparedGraphCount*GPUGraphMemoryContract::reservationBytesPerExecutable(d)))
        {
            requireOwner();(void)tensorBytes(shape,prefill_rows);
            const auto source_begin=std::chrono::steady_clock::now();
            const TensorBase *execution=nullptr;
            if(model_source) {
                execution_owner=convertProjectionSource(*model_source,memory);
                execution=execution_owner?execution_owner.get():&model_source->tensor();
                identity="ordinary prepared projection; model="+model_source->plan().modelPath()+"; source="+
                    model_source->plan().request().tensor_name+"; source-format="+model_source->plan().format();
            }else {
                source_claim=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,shape.payload_bytes);
                auto operand=std::make_shared<FP32Tensor>(std::vector<size_t>{shape.n,shape.k});
                for(size_t index=0;index<shape.n*shape.k;++index)
                    operand->mutable_typed_data()[index]=0.0001f*float(int((index*13)%31)-15);
                execution_owner=std::move(operand);execution=execution_owner.get();
                identity="source-free FP32 arithmetic proxy; not-model-weights; not-attention-or-GDN-timing";
            }
            LOG_INFO("[PlanningProjectionWall] device=" << device.toString() << " source=" << shape.name
                << " phase=" << (model_source?"source_execution_conversion":"host_operand_allocate_fill") << " seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-source_begin).count());
            const auto prepare_begin=std::chrono::steady_clock::now();
            auto *backend=getBackendFor(device);require(backend,"Prepared projection backend missing");
            pool=std::make_shared<LoadOrchestrator>(backend,memory,kWeightOwner);
            planGPUWeight(*pool,shape,device);pool->allocate(shape.payload_bytes,3);
            const auto *format=native_quantised_formats::forQuantType(shape.format);
            const auto repack=format?codebookIdToRepackFormat(format->codebook_id,format->is_superblock):std::optional<RepackFormat>(RepackFormat::RAW_FP);
            require(repack.has_value(),"Prepared source format has no native GPU repacker");
            pool->addWeightJob(device.ordinal,{.name=shape.name,.host_raw_data=execution->raw_data(),
                .raw_bytes=shape.payload_bytes,.format=*repack,.N=int(shape.n),.K=int(shape.k),
                .is_asymmetric=format && canonicalDeviceVnniFormat(*format).is_asymmetric});
            pool->load();pool->finalize();
            engine=KernelFactory::createGemmFromGPUWeightPool(*execution,device,pool,shape.name,0,
                GPUPreparedWeightPoolLayout::SourceNative,shape.operand_precision);
            require(engine && engine->hasWeights(),"Prepared projection engine missing");
            observations.reserve(2);
            identity+="; execution-format="+shape.format+"; N="+std::to_string(shape.n)+"; K="+std::to_string(shape.k)+
                "; device="+device.toString()+"; cache-regime=repeated-same-prepared-matrix; activations=FP32; activation-operand-policy="+
                std::to_string(static_cast<int>(shape.operand_precision));
            LOG_INFO("[PlanningProjectionWall] device=" << device.toString() << " source=" << shape.name
                << " format=" << shape.format << " N=" << shape.n << " K=" << shape.k << " phase=payload_engine_prepare seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now()-prepare_begin).count());
        }

    };
    PlanningPreparedGPUProjection::PlanningPreparedGPUProjection(PlanningFP32ArithmeticPlan,DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        requirePreparedProjectionWorker(device,memory);
        impl_=std::make_unique<Impl>(device,memory,std::nullopt,PlanningFP32ArithmeticPlan::kPrefillRows);
    }
    PlanningPreparedGPUProjection::PlanningPreparedGPUProjection(const PlanningLoadedMatrixSample &sample,DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory,int prefill_rows)
    {
        requirePreparedProjectionWorker(device,memory);
        (void)tensorBytes(ProjectionGeometry::model(sample.plan()),prefill_rows);
        impl_=std::make_unique<Impl>(device,memory,sample,prefill_rows);
    }
    PlanningPreparedGPUProjection::~PlanningPreparedGPUProjection()
    {
        if(impl_ && !impl_->original.worker()->ownsCurrentThread())std::terminate();
    }
    void PlanningPreparedGPUProjection::prepareNextPhase()
    {
        prepareNextPhaseStorage();capturePreparedPhase();completePreparedPhase();
    }
    void PlanningPreparedGPUProjection::prepareNextPhaseStorage()
    {
        auto &owner=*impl_;owner.requireOwner();
        if(owner.phase || owner.next>=2)throw std::logic_error("Prepared projection phase is active or complete");
        const int rows=owner.next==0?1:owner.prefill_rows;
        const auto begin=std::chrono::steady_clock::now();
        const auto allocate=[&](size_t count){return CPUStorageRoot<float>(count,PhysicalMemoryOwner::ExecutionWorkspace,
            DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(owner.memory);};
        auto phase=std::make_unique<Impl::Phase>(owner.memory->claimNewAllocation(owner.device,
            PhysicalMemoryOwner::ExecutionWorkspace,tensorBytes(owner.shape,rows)));
        phase->input=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),owner.shape.k},allocate(size_t(rows)*owner.shape.k),DeviceId::cpu());
        phase->output=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(rows),owner.shape.n},allocate(size_t(rows)*owner.shape.n),DeviceId::cpu());
        for(size_t row=0;row<size_t(rows);++row)for(size_t col=0;col<owner.shape.k;++col)
            phase->input->mutable_typed_data()[row*owner.shape.k+col]=0.0001f*float(int((col%31*13+row*7)%31)-15);
        auto *consumer=dynamic_cast<IWorkspaceConsumer*>(owner.engine.get());require(consumer,"Prepared projection workspace consumer missing");
        const auto requirements=consumer->getWorkspaceRequirements(rows,owner.shape.n,owner.shape.k);
        phase->workspace=std::make_unique<DeviceWorkspaceManager>(owner.device,requirements.total_bytes_with_alignment(),owner.memory);
        require(phase->workspace->allocate(requirements),"Prepared projection workspace allocation failed");
        auto &worker=*owner.original.worker();const ExplicitGPUStream stream(worker.defaultStream());
        TransferEngine::prepareDeviceInput(phase->input.get(),owner.device,stream.get());
        TransferEngine::prepareDeviceOutput(phase->output.get(),owner.device,stream.get());
        TransferEngine::requireDeviceInput(phase->input.get(),owner.device,stream.get());
        phase->clock=std::make_unique<PlanningNativeGPUClockBuffer>(worker,owner.device,stream,2,owner.memory);
        phase->graph=worker.createGraphCapture(stream.get());require(bool(phase->graph),"Prepared projection graph missing");
        // No other worker may begin capture until these ordered uploads and
        // output initialization really retire. Poll the exact original stream
        // outside capture, never a device-wide blocking synchronization.
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(worker.queryStreamExecutionState(stream.get(),"planning projection storage cohort")!=GPUStreamExecutionState::Complete) {
            if(std::chrono::steady_clock::now()>=deadline)
                throw std::runtime_error("Prepared projection storage did not retire within native setup deadline");
            std::this_thread::yield();
        }
        owner.phase=std::move(phase);
        LOG_INFO("[PlanningProjectionWall] device=" << owner.device.toString() << " source=" << owner.shape.name
            << " format=" << owner.shape.format << " rows=" << rows << " phase=phase_storage_prepare seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    }
    void PlanningPreparedGPUProjection::capturePreparedPhase()
    {
        auto &owner=*impl_;owner.requireOwner();
        if(!owner.phase || owner.phase->state!=Impl::PhaseState::StorageReady)
            throw std::logic_error("Prepared projection capture requires completed storage preparation");
        auto &phase=*owner.phase;phase.state=Impl::PhaseState::Failed;
        const int rows=owner.next==0?1:owner.prefill_rows;
        const auto begin=std::chrono::steady_clock::now();
        auto &worker=*owner.original.worker();const ExplicitGPUStream stream(worker.defaultStream());
        auto *consumer=dynamic_cast<IWorkspaceConsumer*>(owner.engine.get());require(consumer,"Prepared projection workspace consumer missing");
        KernelBinding binding(*owner.engine,*consumer);consumer->bindWorkspace(phase.workspace.get());owner.engine->bindGPUStream(stream);
        ScopedBackendGraphCapture capture(worker,*phase.graph,"planning prepared projection");require(capture.begin(),"Prepared projection capture begin failed");
        require(owner.engine->multiply_tensor(phase.input.get(),phase.output.get(),rows,int(owner.shape.n),int(owner.shape.k),
            floating(owner.shape),1.0f,0.0f,nullptr,nullptr,owner.device.toKernelDeviceIndex(),phase.workspace.get()),"Prepared projection projection failed");
        capture.finish();
        const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,2.0*rows*owner.shape.n*owner.shape.k,
            owner.identity+"; rows="+std::to_string(rows));
        phase.timing.emplace(PlanningExecutionMeasurement::recordGPU(work,worker,std::move(phase.clock),std::move(phase.graph)));
        phase.state=Impl::PhaseState::Captured;
        LOG_INFO("[PlanningProjectionWall] device=" << owner.device.toString() << " source=" << owner.shape.name
            << " format=" << owner.shape.format << " rows=" << rows << " phase=phase_capture_record seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    }
    void PlanningPreparedGPUProjection::completePreparedPhase()
    {
        auto &owner=*impl_;owner.requireOwner();
        if(!owner.phase || owner.phase->state!=Impl::PhaseState::Captured)
            throw std::logic_error("Prepared projection completion requires ended capture");
        auto &phase=*owner.phase;phase.state=Impl::PhaseState::Failed;
        const int rows=owner.next==0?1:owner.prefill_rows;
        const auto begin=std::chrono::steady_clock::now();
        phase.timing->prepare(owner.memory);
        const auto growth=phase.timing->residentMemoryBytes();
        if(growth>std::numeric_limits<size_t>::max()-owner.graph_growth)throw std::overflow_error("Prepared projection graph-family growth overflow");
        owner.graph_growth+=growth;
        phase.graph_nodes=phase.timing->graphNodeCount();
        phase.state=Impl::PhaseState::Prepared;
        LOG_INFO("[PlanningProjectionWall] device=" << owner.device.toString() << " source=" << owner.shape.name
            << " format=" << owner.shape.format << " rows=" << rows << " phase=phase_instantiate seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    }
    void PlanningPreparedGPUProjection::observePreparedPhase()
    {
        auto &owner=*impl_;owner.requireOwner();
        if(!owner.phase || owner.phase->state!=Impl::PhaseState::Prepared)
            throw std::logic_error("Prepared projection observation requires its sealed executable family");
        owner.phase->state=Impl::PhaseState::Failed;
        const auto begin=std::chrono::steady_clock::now();
        auto service=owner.phase->timing->observe();
        const double native_seconds=service.elapsedSeconds();
        owner.observations.push_back({owner.next==0?1:owner.prefill_rows,owner.phase->graph_nodes,std::move(service)});
        owner.phase.reset();++owner.next;
        LOG_INFO("[PlanningProjectionWall] device=" << owner.device.toString() << " source=" << owner.shape.name
            << " format=" << owner.shape.format << " phase=isolated_measure_retire seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count() << " native_timed_seconds=" << native_seconds);
    }
    PlanningPreparedGPUProjectionObservations PlanningPreparedGPUProjection::result() const
    {
        const auto &owner=*impl_;owner.requireOwner();
        if(owner.next!=2 || owner.phase)throw std::logic_error("Prepared projection observations are incomplete");
        require(GPUGraphMemoryContract::acceptsFamilyObservation(owner.device,owner.graph_growth,
            2*PlanningExecutionMeasurement::kPreparedGraphCount*GPUGraphMemoryContract::reservationBytesPerExecutable(owner.device)),"Prepared projection graph family exceeded admission");
        const auto bytes=owner.pool->plannedPersistentBytes(owner.device.ordinal,kWeightOwner);
        if(owner.model_source)return PlanningProjectionObservations{owner.model_source->plan(),owner.device,bytes,{},owner.observations};
        return PlanningFP32ArithmeticObservations{owner.device,bytes,{},owner.observations};
    }

    void PlanningProjectionMeasurement::contributeSourceMemory(const PlanningMatrixSamplePlan &plan,
        PlanningSampleOrigin origin,PhysicalMemoryResource host,PhysicalMemoryPlanBuilder &builder)
    {
        if((origin!=PlanningSampleOrigin::LocalGGUF && origin!=PlanningSampleOrigin::PublishedPayload) || host.device!=DeviceId::cpu())
            throw std::invalid_argument("Projection source requires a known original publication and CPU allocator");
        builder.add(host,PhysicalMemoryOwner::ModelSourcePayload,plan.geometry().physicalBytes());
    }
    void PlanningProjectionMeasurement::contributeGPUParticipantMemory(const PlanningMatrixSamplePlan &plan,
        PhysicalMemoryResource host,PhysicalMemoryResource execution,PhysicalMemoryPlanBuilder &builder,int prefill_rows)
    {
        if(!execution.device.is_gpu())throw std::invalid_argument("GPU projection private BOM requires a GPU endpoint");
        const auto geometry=ProjectionGeometry::model(plan);
        contributePreparedMemory(geometry,host,execution,builder,prefill_rows,{});
        if(plan.executionType()!=plan.tensorType())
            builder.add(host,kWeightOwner,CPUStoragePlacement::local().physicalBytes<uint8_t>(geometry.payload_bytes));
    }
    void PlanningProjectionMeasurement::contributeMemory(const PlanningMatrixSamplePlan &plan,
        PlanningSampleOrigin origin,PhysicalMemoryResource host,PhysicalMemoryResource execution,
        PhysicalMemoryPlanBuilder &builder,int prefill_rows,const std::optional<PlanningProjectionCPUObservation> &cpu)
    {
        contributeSourceMemory(plan,origin,host,builder);
        if(execution.device.is_gpu()) {
            if(cpu)throw std::invalid_argument("GPU projection cannot inherit a CPU observer");
            contributeGPUParticipantMemory(plan,host,execution,builder,prefill_rows);
        }else {
            const auto geometry=ProjectionGeometry::model(plan);
            contributePreparedMemory(geometry,host,execution,builder,prefill_rows,cpu);
            if(plan.executionType()!=plan.tensorType())
                builder.add(host,kWeightOwner,CPUStoragePlacement::local().physicalBytes<uint8_t>(geometry.payload_bytes));
        }
    }

    PlanningProjectionObservations PlanningProjectionMeasurement::measure(const PlanningLoadedMatrixSample &sample,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory, int prefill_rows)
    {
        if (!memory) throw std::invalid_argument("Projection sampling requires admitted physical memory");
        if(device.is_gpu()) {
            PlanningPreparedGPUProjection prepared(sample,device,memory,prefill_rows);
            for(int phase=0;phase<2;++phase){prepared.prepareNextPhase();prepared.observePreparedPhase();}
            return std::get<PlanningProjectionObservations>(prepared.result());
        }
        const auto conversion_begin=std::chrono::steady_clock::now();
        const auto &plan = sample.plan();
        const auto geometry = ProjectionGeometry::model(plan);
        // The model loader alone decides runtime promotion. Retain the native
        // source independently and retire the converted owner after all engine
        // borrowers return; neither conversion nor capture is timed arithmetic.
        const auto converted=convertProjectionSource(sample,memory);
        const TensorBase *execution=converted?converted.get():&sample.tensor();
        LOG_INFO("[PlanningProjectionWall] device=" << device.toString() << " source=" << geometry.name
            << " format=" << geometry.format << " phase=source_execution_conversion seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-conversion_begin).count());
        auto observed = measureCPUPrepared(geometry, *execution, device, memory, prefill_rows,
            "ordinary prepared projection; model=" + plan.modelPath() + "; source=" + plan.request().tensor_name +
            "; source-format=" + plan.format());
        return {plan, observed.device, observed.prepared_bytes, observed.cpu, std::move(observed.phases)};
    }

    void PlanningProjectionMeasurement::contributeMemory(PlanningFP32ArithmeticPlan,
        PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder,
        const std::optional<PlanningProjectionCPUObservation> &cpu)
    {
        const auto geometry = ProjectionGeometry::arithmetic();
        contributePreparedMemory(geometry, host, execution, builder, PlanningFP32ArithmeticPlan::kPrefillRows, cpu);
        // Synthetic operands are setup workspace, never fictitious model bytes.
        builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, geometry.payload_bytes);
    }

    PlanningFP32ArithmeticObservations PlanningProjectionMeasurement::measure(PlanningFP32ArithmeticPlan,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!memory) throw std::invalid_argument("Arithmetic sampling requires admitted physical memory");
        if(device.is_gpu()) {
            PlanningPreparedGPUProjection prepared(PlanningFP32ArithmeticPlan{},device,memory);
            for(int phase=0;phase<2;++phase){prepared.prepareNextPhase();prepared.observePreparedPhase();}
            return std::get<PlanningFP32ArithmeticObservations>(prepared.result());
        }
        const auto geometry = ProjectionGeometry::arithmetic();
        auto claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace, geometry.payload_bytes);
        const auto fill_begin=std::chrono::steady_clock::now();
        FP32Tensor operand({geometry.n, geometry.k});
        for (size_t index = 0; index < geometry.n * geometry.k; ++index)
            operand.mutable_typed_data()[index] = 0.0001f * float(int((index * 13) % 31) - 15);
        LOG_INFO("[PlanningProjectionWall] device=" << device.toString()
            << " source=planning-fp32-arithmetic-basis phase=host_operand_allocate_fill seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_begin).count());
        auto observed = measureCPUPrepared(geometry, operand, device, memory, PlanningFP32ArithmeticPlan::kPrefillRows,
            "source-free FP32 arithmetic proxy; not-model-weights; not-attention-or-GDN-timing");
        return {observed.device, observed.prepared_bytes, observed.cpu, std::move(observed.phases)};
    }
}
