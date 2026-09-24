/**
 * @file Test__PlanningExecutionMeasurement.cpp
 * @brief Real CPU and captured CUDA/ROCm proof of startup service observations.
 *
 * The same production prepared-weight kernels serve every format. Preparation,
 * capture and diagnostic downloads stay outside timing. Two observations reuse
 * one executable and must leave byte-identical results, proving that the timing
 * boundary neither changes execution policy nor retires storage prematurely.
 * These small, cache-resident witnesses certify the measurement lifecycle, not
 * a model throughput prediction. No speed threshold belongs in preflight.
 * Coalesced binding witnesses compare every independently addressable expert
 * against its standalone captured execution, including migration-reserved pools.
 */
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "transfer/TransferEngine.h"
#include "../../utils/GpuPreparedGemmHarness.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Reuse the canonical quantized inventory, with all floating types. */
    std::vector<QuantizedVerifierFormatCase> formats()
    {
        auto result = quantizedVerifierFormats();
        result.push_back({"FP32", TensorType::FP32, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, seed); }});
        result.push_back({"FP16", TensorType::FP16, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, seed); }});
        result.push_back({"BF16", TensorType::BF16, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, seed); }});
        return result;
    }

    /** @brief Restore the caller's worker budget after a completed CPU sample. */
    struct WorkerBudget final
    {
        int previous = omp_get_max_threads();
        /** @brief Exercise both serial and a non-power-of-two real workshare. */
        explicit WorkerBudget(int threads) { omp_set_num_threads(threads); }
        /** @brief No work remains active when the measurement returns. */
        ~WorkerBudget() { omp_set_num_threads(previous); }
    };

    /** @brief Retire a borrowed workspace binding before its owning manager. */
    struct WorkspaceBinding final
    {
        IWorkspaceConsumer &consumer;
        /** @brief Install the exact preallocated workspace before capture. */
        WorkspaceBinding(IWorkspaceConsumer &value, DeviceWorkspaceManager &workspace)
            : consumer(value) { consumer.bindWorkspace(&workspace); }
        /** @brief Remove the borrowed pointer after the retained graph retires. */
        ~WorkspaceBinding() { consumer.unbindWorkspace(); }
    };

    /** @brief Check work/time units without treating a microbenchmark as a speed gate. */
    void expectObservation(const PlanningServiceObservation &observation, int rows)
    {
        EXPECT_EQ(observation.unit(), PlanningWorkUnit::ArithmeticOperations);
        EXPECT_EQ(observation.completedWork(), 3.0 * 2 * rows * 256 * 256);
        EXPECT_GT(observation.elapsedSeconds(), 0);
        EXPECT_TRUE(std::isfinite(observation.unitsPerSecond()));
        std::cout << "MEASUREMENT " << observation.provenance()
                  << " seconds=" << observation.elapsedSeconds() << '\n';
    }

    /**
     * @brief Execute one independently bound matrix through a retained graph.
     * @param kernel Production engine retaining its prepared pool ownership.
     * @param device Exact worker/backend on which the graph is recorded.
     * @param context Worker owning capture and the non-default stream.
     * @param rows Input rows; input bytes are identical for standalone and coalesced views.
     * @param n Exact output width of the prepared matrix.
     * @param k Exact input width of the prepared matrix.
     * @return Diagnostic output bytes, downloaded only after ordered publication.
     */
    std::vector<float> capturedResult(ITensorGemm &kernel, DeviceId device,
        IWorkerGPUContext &context, int rows, int n = 256, int k = 256)
    {
        const auto stream = context.defaultStream();
        kernel.setGPUStream(stream);
        auto &consumer = dynamic_cast<IWorkspaceConsumer &>(kernel);
        const auto requirements = consumer.getWorkspaceRequirements(rows, n, k);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
        if (!workspace.allocate(requirements)) throw std::runtime_error("Cannot bind coalesced proof workspace");
        WorkspaceBinding binding(consumer, workspace);
        auto input = TestTensorFactory::createFP32Random({size_t(rows), size_t(k)}, 937);
        auto output = TestTensorFactory::createFP32({size_t(rows), size_t(n)});
        if (!input->ensureOnDevice(device, stream) || !output->ensureOnDevice(device, stream))
            throw std::runtime_error("Cannot prepare coalesced proof tensors");
        TransferEngine::requireDeviceInput(input.get(), device, stream);
        TransferEngine::requireDeviceInput(output.get(), device, stream);
        auto graph = context.createGraphCapture(stream);
        {
            ScopedBackendGraphCapture capture(context, *graph, "coalesced pool binding proof");
            if (!capture.begin()) throw std::runtime_error("Cannot capture coalesced proof");
            if (!kernel.multiply_tensor(input.get(), output.get(), rows, n, k))
                throw std::runtime_error("Coalesced proof GEMM failed");
            capture.finish();
        }
        if (!graph->instantiate() || !graph->launch())
            throw std::runtime_error("Cannot launch coalesced proof");
        TransferEngine::publishDeviceWrite(output.get(), device, stream);
        if (!output->ensureOnHost(stream)) throw std::runtime_error("Cannot read coalesced proof output");
        return {output->data(), output->data() + output->numel()};
    }

    /**
     * @brief Prove source-sampled expert matrices through actual backend preparation.
     *
     * The sparse fixture contains one nonzero final expert. Its source owner is
     * destroyed before kernel preparation, then the exact sampled bytes must
     * match standalone GEMM for M=1/32. CUDA/HIP run captured production kernels.
     * Q8_1 retains its existing runtime-format test; it has no GGUF source type.
     */
    void proveSourceSamples(DeviceId device, IWorkerGPUContext *context = nullptr)
    {
        for (const auto &format : formats())
        {
            if (format.tensor_type == TensorType::Q8_1) continue;
            SCOPED_TRACE(format.label);
            auto standalone = format.create({512, 256}, 731);
            PlanningGGUFFixture fixture(true, false, PlanningGGUFFixture::sourceType(format.tensor_type), 512);
            const std::string name = "blk.0.ffn_gate_exps.weight";
            const size_t bytes = standalone->size_bytes();
            fixture.writePayload(name, 7 * bytes, {static_cast<const uint8_t *>(standalone->raw_data()), bytes});
            const PhysicalMemoryResource resource{.world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = 2 * bytes, .admission_available_bytes = 2 * bytes};
            PhysicalMemoryPlanBuilder bom;
            bom.add(resource, PhysicalMemoryOwner::ModelSourcePayload, bytes);
            bom.add(resource, PhysicalMemoryOwner::WeightLoadStaging, bytes);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bom.build()), 0);
            std::optional<PlanningLoadedModelSample> sample;
            {
                PlanningModelSource source(fixture.path());
                sample = source.loadSample({name, PlanningExpertMatrix{7}}, memory, DeviceId::cpu());
            }
            // The test store's registration helper publishes prepared-coherence
            // metadata through a mutable pointer. Weight bytes stay unchanged;
            // the typed sample remains alive until both prepared stores retire.
            auto *sampled = const_cast<TensorBase *>(&sample->tensor());
            ASSERT_EQ(sampled->native_type(), format.tensor_type);
            ASSERT_EQ(std::memcmp(sampled->raw_data(), standalone->raw_data(), bytes), 0);
            if (device.is_cpu())
            {
                auto reference = makePreparedGemmFixture(standalone.get(), device, "source.reference");
                auto actual = makePreparedGemmFixture(sampled, device, "source.sampled");
                auto *reference_kernel = reference.store->gemmKernel(reference.ref);
                auto *actual_kernel = actual.store->gemmKernel(actual.ref);
                auto requirements = dynamic_cast<IWorkspaceConsumer &>(*reference_kernel).getWorkspaceRequirements(32, 512, 256);
                requirements.merge(dynamic_cast<IWorkspaceConsumer &>(*actual_kernel).getWorkspaceRequirements(32, 512, 256));
                DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
                ASSERT_TRUE(workspace.allocate(requirements));
                for (int rows : {1, 32})
                {
                    auto input = TestTensorFactory::createFP32Random({size_t(rows), 256}, 937);
                    auto expected = TestTensorFactory::createFP32({size_t(rows), 512});
                    auto output = TestTensorFactory::createFP32({size_t(rows), 512});
                    // These projections are serial consumers of one admitted
                    // invocation bank, never bindings stored on the weights.
                    ASSERT_TRUE(reference_kernel->multiply_tensor(input.get(), expected.get(), rows, 512, 256,
                        true, 1.f, 0.f, nullptr, nullptr, -1, &workspace));
                    ASSERT_TRUE(actual_kernel->multiply_tensor(input.get(), output.get(), rows, 512, 256,
                        true, 1.f, 0.f, nullptr, nullptr, -1, &workspace));
                    ASSERT_EQ(std::memcmp(output->data(), expected->data(), output->numel() * sizeof(float)), 0);
                }
            }
            else
            {
                ASSERT_NE(context, nullptr);
                const bool floating = format.tensor_type == TensorType::FP32 ||
                    format.tensor_type == TensorType::FP16 || format.tensor_type == TensorType::BF16;
                auto reference = floating ? makeGpuPreparedFloatingPointGemm(standalone.get(), device, "source.reference")
                    : makeGpuPreparedGemm(standalone.get(), device, "source.reference");
                auto actual = floating ? makeGpuPreparedFloatingPointGemm(sampled, device, "source.sampled")
                    : makeGpuPreparedGemm(sampled, device, "source.sampled");
                for (int rows : {1, 32})
                {
                    const auto expected = capturedResult(*reference.kernel, device, *context, rows, 512, 256);
                    const auto result = capturedResult(*actual.kernel, device, *context, rows, 512, 256);
                    ASSERT_EQ(result.size(), expected.size());
                    ASSERT_EQ(std::memcmp(result.data(), expected.data(), result.size() * sizeof(float)), 0);
                }
            }
            sample.reset();
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ModelSourcePayload,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    }

    /**
     * @brief Prove group offsets, physical stride and borrowed-owner lifetimes.
     *
     * Three different experts share one upload/repack. The middle and last
     * expert catch code which accidentally binds the first row or uses compact
     * source width to stride through migration-reserved group storage. Dropping
     * the caller's owner before replay proves the engines retain that storage.
     */
    void proveCoalescedGPU(DeviceId device)
    {
        namespace kf = llaminar::v2::kernels;
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        context.submitAndWait([&] {
            for (const auto &format : formats())
            {
                SCOPED_TRACE(format.label);
                std::vector<std::unique_ptr<TensorBase>> weights;
                std::vector<uint8_t> source_bytes;
                std::vector<std::vector<float>> expected;
                const bool floating = format.tensor_type == TensorType::FP32 ||
                    format.tensor_type == TensorType::FP16 || format.tensor_type == TensorType::BF16;
                for (int group = 0; group != 3; ++group)
                {
                    weights.push_back(format.create({256, 256}, 731 + group));
                    const auto &tensor = weights.back();
                    const auto *begin = static_cast<const uint8_t *>(tensor->raw_data());
                    source_bytes.insert(source_bytes.end(), begin, begin + tensor->size_bytes());
                    auto prepared = floating
                        ? makeGpuPreparedFloatingPointGemm(tensor.get(), device, "binding.standalone")
                        : makeGpuPreparedGemm(tensor.get(), device, "binding.standalone");
                    for (int rows : {1, 32})
                        expected.push_back(capturedResult(*prepared.kernel, device, context, rows));
                }
                for (auto layout : {GPUPreparedWeightPoolLayout::SourceNative, GPUPreparedWeightPoolLayout::MigrationReusable})
                {
                    SCOPED_TRACE(static_cast<int>(layout));
                    auto owner = std::make_shared<LoadOrchestrator>(backend, kTestOnlyUnadmittedGPUAllocation);
                    owner->addDevice(device.ordinal);
                    WeightJob job{};
                    job.name = "binding.coalesced";
                    job.host_raw_data = source_bytes.data();
                    job.raw_bytes = source_bytes.size();
                    job.N = 3 * 256;
                    job.K = 256;
                    job.full_N = job.N;
                    job.full_K = job.K;
                    job.packed_group_rows = 256;
                    if (floating)
                    {
                        job.format = RepackFormat::RAW_FP;
                        owner->planRawWeight(device.ordinal, job.name, job.N, job.K, job.raw_bytes);
                    }
                    else
                    {
                        const auto &source = *dynamic_cast<const IINT8Unpackable &>(*weights.front()).vnniFormatInfo();
                        const auto allocation = layout == GPUPreparedWeightPoolLayout::MigrationReusable
                            ? reusableDeviceVnniAllocationFormat(source)
                            : NativeVnniReusableDeviceAllocationFormat{static_cast<uint8_t>(source.payload_bytes),
                                source.is_asymmetric, source.has_emins};
                        job.format = codebookIdToRepackFormat(source.codebook_id, source.is_superblock).value();
                        job.is_asymmetric = source.is_asymmetric;
                        job.packed_payload_capacity_bytes_per_block = allocation.payload_bytes_per_block;
                        owner->planWeight(device.ordinal, job.name, job.N, job.K,
                            allocation.payload_bytes_per_block, allocation.has_mins, allocation.has_emins, job.raw_bytes);
                    }
                    // Chunk boundaries deliberately cross expert boundaries;
                    // the loader must preserve each group's standalone layout.
                    owner->allocate(weights.front()->size_bytes() * 3 / 2, 2);
                    owner->addWeightJob(device.ordinal, job);
                    owner->load();
                    owner->finalize();
                    std::vector<std::unique_ptr<ITensorGemm>> kernels;
                    for (int group = 0; group != 3; ++group)
                        kernels.push_back(kf::KernelFactory::createGemmFromGPUWeightPool(
                            *weights[group], device, owner, job.name, group * 256, layout));
                    std::weak_ptr<LoadOrchestrator> lifetime = owner;
                    owner.reset();
                    EXPECT_FALSE(lifetime.expired());
                    size_t index = 0;
                    for (auto &kernel : kernels)
                        for (int rows : {1, 32})
                        {
                            SCOPED_TRACE(index);
                            const auto actual = capturedResult(*kernel, device, context, rows);
                            const auto &reference = expected.at(index++);
                            ASSERT_EQ(actual.size(), reference.size());
                            EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }));
                            EXPECT_EQ(std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)), 0);
                        }
                    kernels.clear();
                    EXPECT_TRUE(lifetime.expired());
                }
            }
        });
    }

    /** @brief Exercise the installed backend on its owning worker, never an eager substitute. */
    void proveGPU(DeviceId device)
    {
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        context.submitAndWait([&] {
            const auto stream = context.defaultStream();
            ASSERT_NE(stream, nullptr);
            for (const auto &format : formats())
            {
                SCOPED_TRACE(format.label);
                auto weights = format.create({256, 256}, 731);
                const bool floating = format.tensor_type == TensorType::FP32 ||
                    format.tensor_type == TensorType::FP16 || format.tensor_type == TensorType::BF16;
                auto prepared = floating
                    ? makeGpuPreparedFloatingPointGemm(weights.get(), device, "planning.sample")
                    : makeGpuPreparedGemm(weights.get(), device, "planning.sample");
                auto *kernel = prepared.kernel;
                ASSERT_NE(kernel, nullptr);
                kernel->setGPUStream(stream);
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
                ASSERT_NE(consumer, nullptr);
                const auto requirements = consumer->getWorkspaceRequirements(32, 256, 256);
                DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
                ASSERT_TRUE(workspace.allocate(requirements));
                WorkspaceBinding binding(*consumer, workspace);
                for (const int rows : {1, 32})
                {
                    SCOPED_TRACE(rows);
                    auto input = TestTensorFactory::createFP32Random({size_t(rows), 256}, 937);
                    auto output = TestTensorFactory::createFP32({size_t(rows), 256});
                    ASSERT_TRUE(input->ensureOnDevice(device, stream));
                    ASSERT_TRUE(output->ensureOnDevice(device, stream));
                    TransferEngine::requireDeviceInput(input.get(), device, stream);
                    TransferEngine::requireDeviceInput(output.get(), device, stream);
                    auto graph = context.createGraphCapture(stream);
                    ASSERT_NE(graph, nullptr);
                    {
                        ScopedBackendGraphCapture capture(context, *graph, "planning measurement proof");
                        ASSERT_TRUE(capture.begin());
                        ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(), rows, 256, 256));
                        capture.finish();
                    }
                    ASSERT_TRUE(graph->instantiate());
                    const auto nodes = graph->nodeCount();
                    ASSERT_GT(nodes, 0);
                    const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                        2.0 * rows * 256 * 256, device.to_string() + "; " + format.label +
                        "; M=" + std::to_string(rows) + "; N=256; K=256; warm resident fixture");
                    std::vector<float> first;
                    for (int repeat = 0; repeat != 2; ++repeat)
                    {
                        expectObservation(PlanningExecutionMeasurement::gpu(work, *backend, device, *graph), rows);
                        EXPECT_EQ(graph->nodeCount(), nodes);
                        // Diagnostic I/O still carries the exact producer
                        // stream; observing timing is not a coherence token.
                        TransferEngine::publishDeviceWrite(output.get(), device, stream);
                        ASSERT_TRUE(output->ensureOnHost(stream));
                        const float *data = output->data();
                        ASSERT_TRUE(std::all_of(data, data + output->numel(),
                            [](float value) { return std::isfinite(value); }));
                        if (repeat == 0) first.assign(data, data + output->numel());
                        else EXPECT_EQ(std::memcmp(first.data(), data, first.size() * sizeof(float)), 0);
                    }
                }
            }
        });
    }
}

/** @test Production CPU workshares complete before the observation is returned. */
TEST(PlanningExecutionMeasurementIntegration, CPU)
{
    for (int workers : {1, 3})
    {
        WorkerBudget budget(workers);
        for (const auto &format : formats())
        {
            SCOPED_TRACE(format.label);
            auto weights = format.create({256, 256}, 731);
            auto prepared = makePreparedGemmFixture(weights.get(), DeviceId::cpu(), "planning.sample");
            auto *kernel = prepared.store->gemmKernel(prepared.ref);
            ASSERT_NE(kernel, nullptr);
            const auto requirements = dynamic_cast<IWorkspaceConsumer &>(*kernel).getWorkspaceRequirements(32, 256, 256);
            DeviceWorkspaceManager workspace(DeviceId::cpu(), requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace.allocate(requirements));
            for (int rows : {1, 32})
            {
                auto input = TestTensorFactory::createFP32Random({size_t(rows), 256}, 937);
                auto output = TestTensorFactory::createFP32({size_t(rows), 256});
                const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,
                    2.0 * rows * 256 * 256, std::string("CPU; ") + format.label +
                    "; M=" + std::to_string(rows) + "; N=256; K=256; workers=" +
                    std::to_string(workers) + "; warm resident fixture");
                // Both observations reuse the same preallocated invocation
                // storage; no allocation or binding belongs in the timed work.
                const auto execute = [&] { return kernel->multiply_tensor(input.get(), output.get(), rows, 256, 256,
                    true, 1.f, 0.f, nullptr, nullptr, -1, &workspace); };
                expectObservation(PlanningExecutionMeasurement::cpu(work, execute), rows);
                std::vector<float> first(output->data(), output->data() + output->numel());
                expectObservation(PlanningExecutionMeasurement::cpu(work, execute), rows);
                ASSERT_TRUE(std::all_of(first.begin(), first.end(), [](float value) { return std::isfinite(value); }));
                EXPECT_EQ(std::memcmp(first.data(), output->data(), first.size() * sizeof(float)), 0);
            }
        }
    }
}

/** @test Source sampling composes with CPU preparation for every GGUF codebook. */
TEST(PlanningExecutionMeasurementIntegration, CPU_SourceSamples)
{
    WorkerBudget budget(3);
    proveSourceSamples(DeviceId::cpu());
}

#ifdef HAVE_CUDA
/** @test Every weight format uses retained CUDA execution and native event time. */
TEST(PlanningExecutionMeasurementIntegration, CUDA)
{
    ensureNvidiaFactoryRegistered();
    ASSERT_TRUE(GPUDeviceContextPool::instance().hasNvidiaSupport());
    proveGPU(DeviceId::cuda(0));
}
/** @test Every GGUF codebook, including 3-D Q3_K, reaches captured CUDA execution. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_SourceSamples)
{
    ensureNvidiaFactoryRegistered();
    auto &context = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    context.submitAndWait([&] { proveSourceSamples(DeviceId::cuda(0), &context); });
}
/** @test CUDA group views retain every format's captured standalone bytes. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_CoalescedPool)
{
    ensureNvidiaFactoryRegistered();
    ASSERT_TRUE(GPUDeviceContextPool::instance().hasNvidiaSupport());
    proveCoalescedGPU(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @test Every weight format uses retained HIP execution and native event time. */
TEST(PlanningExecutionMeasurementIntegration, ROCm)
{
    ensureAMDFactoryRegistered();
    ASSERT_TRUE(GPUDeviceContextPool::instance().hasAMDSupport());
    proveGPU(DeviceId::rocm(0));
}
/** @test HIP proves the same expert-source lifetime and byte-equivalent GEMM path. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_SourceSamples)
{
    ensureAMDFactoryRegistered();
    auto &context = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    context.submitAndWait([&] { proveSourceSamples(DeviceId::rocm(0), &context); });
}
/** @test HIP group views retain every format's captured standalone bytes. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_CoalescedPool)
{
    ensureAMDFactoryRegistered();
    ASSERT_TRUE(GPUDeviceContextPool::instance().hasAMDSupport());
    proveCoalescedGPU(DeviceId::rocm(0));
}
#endif
