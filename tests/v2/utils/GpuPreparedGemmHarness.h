#pragma once

/**
 * @file GpuPreparedGemmHarness.h
 * @brief Shared test helper that builds a *real* GPU quantized GEMM kernel from
 *        a host weight tensor by driving the production GPU weight load pipeline.
 *
 * Why this exists: `KernelFactory::prepareGemmHandleLocal()` (and therefore
 * `PreparedWeightStore::prepareGemm()` / the `makePreparedGemmFixture` helpers)
 * intentionally REJECT GPU INT8-packed weights — those device GEMM kernels must
 * be constructed from VRAM-resident, VNNI-repacked payloads owned by a
 * `WeightVRAMPool`. The legacy lazy tensor constructor
 * (`CUDAQuantisedGemmKernel(weights, id)`) does not set this up correctly and
 * triggers illegal-memory-access faults under the modern workspace contract.
 *
 * This helper mirrors the real production path in `WeightManager.cpp`:
 *   1. Plan the weight in a per-device `WeightVRAMPool` (via `LoadOrchestrator`).
 *   2. Upload + GPU-repack the raw GGUF blocks into native-VNNI layout.
 *   3. Construct the quantized GEMM kernel from the pool slot device pointers,
 *      retaining the orchestrator as a lifetime owner (keeps the VRAM alive).
 *   4. Register the kernel handle in a `PreparedWeightStore` via
 *      `registerPreparedGemmHandle()` (the only un-guarded registration path).
 *
 * Lifetime/ownership: the returned `GpuPreparedGemm` owns BOTH the
 * `LoadOrchestrator` (which owns the VRAM pool) and the `PreparedWeightStore`
 * (which owns the kernel handle). Keep the struct alive for as long as the
 * returned `kernel` pointer is used.
 *
 * Thread-safety: not thread-safe. Intended for single-threaded test setup.
 */

#include "backends/BackendManager.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "kernels/KernelFactory.h"
#include "tensors/TensorClasses.h"

#include "PreparedWeightTestHarness.h"

#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
    /**
     * @brief Owns all lifetime objects backing a GPU-prepared GEMM kernel.
     *
     * The `kernel` pointer is borrowed from `store`; both `store` and
     * `orchestrator` must outlive any use of `kernel`.
     */
    struct GpuPreparedGemm
    {
        std::shared_ptr<LoadOrchestrator> orchestrator; ///< Owns the VRAM pool (kernel lifetime owner).
        std::unique_ptr<PreparedWeightStore> store;     ///< Owns the prepared GEMM handle.
        WeightBinding binding;                           ///< Binding used for registration.
        PreparedWeightRef ref;                           ///< Ref to fetch the kernel from the store.
        ITensorGemm *kernel = nullptr;                   ///< Borrowed kernel (not owned).
    };

    /**
     * @brief Build a real GPU quantized GEMM kernel for @p weights on @p device.
     *
     * Uploads and VNNI-repacks the weight into a dedicated `WeightVRAMPool`, then
     * constructs and registers the device GEMM kernel exactly as the production
     * load pipeline does. Only quantized weights (implementing `IINT8Unpackable`
     * with a valid `vnniFormatInfo()`) are supported.
     *
     * @param weights         Host-resident quantized weight tensor (raw GGUF blocks).
     * @param device          Target GPU device (must be CUDA or ROCm).
     * @param canonical_name  Logical weight name used for the pool slot + binding.
     * @param model_id        Model context id for the throwaway PreparedWeightStore.
     * @return Fully wired `GpuPreparedGemm` with a ready-to-run `kernel`.
     *
     * @throws std::runtime_error if no GPU backend is available, the weight is not
     *         quantized, the format is unsupported, or the pipeline/registration fails.
     */
    inline GpuPreparedGemm makeGpuPreparedGemm(
        TensorBase *weights,
        DeviceId device,
        const std::string &canonical_name = "test.gpu_prepared_gemm.weight",
        ModelContextId model_id = ModelContextId{9900})
    {
        namespace kf = llaminar::v2::kernels;

        if (!weights)
            throw std::runtime_error("makeGpuPreparedGemm: null weight tensor");
        if (!device.is_cuda() && !device.is_rocm())
            throw std::runtime_error("makeGpuPreparedGemm: device must be CUDA or ROCm");

        // The pipeline reads raw GGUF block data straight from host and uploads it
        // itself, so the weight must expose intrinsic native-VNNI format metadata.
        auto *unpackable = dynamic_cast<IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!vnni)
            throw std::runtime_error("makeGpuPreparedGemm: weight is not a quantized native-VNNI tensor");

        const int N = static_cast<int>(weights->rows());
        const int K = static_cast<int>(weights->cols());
        const size_t raw_bytes = weights->size_bytes();

        // Resolve the GPU repack-kernel dispatch format from the codebook id.
        auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
        if (!repack_fmt)
            throw std::runtime_error(
                "makeGpuPreparedGemm: unsupported format (codebook=" +
                std::to_string(static_cast<int>(vnni->codebook_id)) +
                ", superblock=" + std::to_string(vnni->is_superblock) + ")");

        // Step 1: Acquire the backend for the target device.
        IBackend *backend = getBackendFor(device);
        if (!backend)
            throw std::runtime_error("makeGpuPreparedGemm: no GPU backend available for device");

        GpuPreparedGemm out;

        // Step 2: Create the orchestrator + plan the single weight on its device pool.
        // The orchestrator owns the WeightVRAMPool; we retain it as the kernel's
        // lifetime owner so the VRAM payload outlives the kernel's execution.
        out.orchestrator = std::make_shared<LoadOrchestrator>(backend);
        out.orchestrator->addDevice(device.ordinal);
        out.orchestrator->planWeight(
            device.ordinal, canonical_name, N, K,
            vnni->payload_bytes, vnni->is_asymmetric, vnni->has_emins, raw_bytes);

        // Allocate pool + staging. Pinned slot size must cover the largest raw weight.
        constexpr int kNumStreams = 3;
        out.orchestrator->allocate(raw_bytes, kNumStreams);

        // Step 3: Submit the upload + GPU-repack job and run the pipeline.
        WeightJob job;
        job.name = canonical_name;
        job.host_raw_data = weights->raw_data();
        job.raw_bytes = raw_bytes;
        job.format = *repack_fmt;
        job.N = N;
        job.K = K;
        job.is_asymmetric = vnni->is_asymmetric;
        out.orchestrator->addWeightJob(device.ordinal, job);
        out.orchestrator->load();

        // Step 4: Fetch the VRAM slot and build the device GEMM kernel from its pointers.
        WeightVRAMPool *pool = out.orchestrator->getPool(device.ordinal);
        if (!pool)
            throw std::runtime_error("makeGpuPreparedGemm: pool not found after load");
        auto slot = pool->getSlot(canonical_name);
        if (!slot)
            throw std::runtime_error("makeGpuPreparedGemm: weight slot missing after load");

        const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);
        std::unique_ptr<ITensorGemm> kernel;
        kf::KernelFactory::GemmPreparationKind prep_kind =
            kf::KernelFactory::GemmPreparationKind::CPU_PACKED;

#ifdef HAVE_CUDA
        if (device.is_cuda())
        {
            kernel = std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                N, K, device.ordinal,
                slot->d_native_vnni_payload,
                static_cast<uint16_t *>(slot->d_native_vnni_scales),
                static_cast<uint16_t *>(slot->d_native_vnni_mins),
                static_cast<uint32_t *>(slot->d_native_vnni_emins),
                vnni->codebook_id, blocks_per_row,
                out.orchestrator); // lifetime owner: keeps VRAM pool alive
            prep_kind = kf::KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED;
        }
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm())
        {
            kernel = std::make_unique<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                N, K, device.ordinal,
                slot->d_native_vnni_payload,
                slot->d_native_vnni_scales,
                slot->d_native_vnni_mins,
                slot->d_native_vnni_emins,
                vnni->codebook_id, blocks_per_row,
                out.orchestrator); // lifetime owner: keeps VRAM pool alive
            prep_kind = kf::KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED;
        }
#endif
        if (!kernel)
            throw std::runtime_error("makeGpuPreparedGemm: failed to construct GPU GEMM kernel");

        // Step 5: Wrap the kernel in a prepared handle and register it in the store.
        auto owned_kernel = std::shared_ptr<ITensorGemm>(std::move(kernel));
        auto prepared_weights = std::make_shared<kf::KernelFactory::PreparedGemmWeights>();
        prepared_weights->kind = prep_kind;
        prepared_weights->owned_kernel = owned_kernel;
        prepared_weights->kernel = owned_kernel.get();

        auto handle = std::make_shared<kf::KernelFactory::PreparedGemmHandle>();
        handle->tensor = weights;
        handle->device_id = device;
        handle->kind = prep_kind;
        handle->variant = static_cast<int>(weights->native_type());
        handle->prepared_weights = std::move(prepared_weights);

        out.store = std::make_unique<PreparedWeightStore>(model_id);
        out.binding = makePreparedWeightTestBinding(weights, device, canonical_name, model_id);
        out.ref = out.store->registerPreparedGemmHandle(
            out.binding,
            preparedKindForDevice(device),
            device,
            std::move(handle));

        out.kernel = out.store->gemmKernel(out.ref);
        if (!out.kernel)
            throw std::runtime_error("makeGpuPreparedGemm: gemmKernel returned null after registration");

        return out;
    }

} // namespace llaminar2::test
