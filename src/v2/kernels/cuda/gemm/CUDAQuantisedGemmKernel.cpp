/**
 * @file CUDAQuantisedGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for CUTLASS INT8 quantized GEMM
 *
 * This is the C++ adapter that wraps the CUTLASS INT8 GEMM kernel. It implements
 * the full ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not nvcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**:
 * 1. Implements ITensorGemm (includes IMPIContext, TensorBase, etc.)
 * 2. Delegates CUDA work to CUDAQuantisedGemmKernel_Impl (in .cu file)
 * 3. Handles tensor type introspection in multiply_tensor()
 * 4. Manages lazy weight conversion to INT8 + scales
 *
 * **Weight Conversion Pipeline**:
 * 1. Dequantize original quantized weights to FP32
 * 2. Per-column symmetric quantization to INT8
 * 3. Store INT8 weights in ColumnMajor layout (CUTLASS requirement)
 * 4. Store per-column scales for output rescaling
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CUDAQuantisedGemmKernel.h"
#include "CUDADeviceWorkspace.h"
#include "backends/ComputeBackend.h" // DeviceManager
#include "backends/DeviceId.h"       // DeviceId
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/TensorSlice.h"     // TensorSlice - for unwrapping sliced biases
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/KernelSnapshotInfo.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h" // isGraphCaptureActive()
#include "utils/Logger.h"
#include "utils/CUDAKernelProfiler.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{
    namespace cuda
    {

        // =====================================================================
        // Forward declarations for CUDA implementation (defined in .cu file)
        // =====================================================================

        // These functions are implemented in CUDAQuantisedGemmKernel_CUTLASS.cu
        extern "C"
        {
            void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
            bool cudaNativeVNNIPrefill_getDeterministicMode();

            // Upload work buffers for activation quantization
            bool cudaQuantGemm_ensureWorkBuffers(
                int8_t **d_A_int8,   // [M x K] quantized activations
                float **d_scales_A,  // [M] per-row scales
                int32_t **d_C_int32, // [M x N] INT32 accumulator
                int *work_buffer_M,  // Current capacity
                int M, int K, int N,
                int cuda_device_id);

            // Quantize FP32 activations to INT8 with per-block-of-32 scales
            bool cudaQuantGemm_quantizeActivationsBlockwise(
                const float *d_A_fp32,       // [M x K]
                int8_t *d_A_int8,            // [M x K] output
                float *d_scales_A_blockwise, // [M x (K/32)] output
                int M, int K,
                int cuda_device_id,
                void *stream = nullptr);

            bool cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
                const float *d_A_fp32,       // [M x K]
                int8_t *d_A_int8,            // [M x K] output
                float *d_scales_A_blockwise, // [M x (K/32)] output
                int32_t *d_sums_A_blockwise, // [M x (K/32)] output
                int M, int K,
                int cuda_device_id,
                void *stream = nullptr);

            // Free device memory
            void cudaQuantGemm_freeDevice(void *d_ptr);

            // Fused SwiGLU + blockwise quantization (from CUDAFusedOpsKernels.cu)
            bool cudaOps_fused_swiglu_quantize_blockwise(
                const float *gate,
                const float *up,
                int8_t *A_int8,
                float *scales_A_blockwise,
                int M, int K,
                int device_idx,
                void *stream);

            bool cudaOps_fused_swiglu_quantize_blockwise_with_sums(
                const float *gate,
                const float *up,
                int8_t *A_int8,
                float *scales_A_blockwise,
                int32_t *sums_A_blockwise,
                int M, int K,
                int device_idx,
                void *stream);

            // Concurrent prefill stream/event helpers (from CUDAQuantisedGemmKernel_CUTLASS.cu)
            bool cudaQuantGemm_createStream(void **out_stream, int cuda_device_id);
            void cudaQuantGemm_destroyStream(void *stream);
            bool cudaQuantGemm_createEvent(void **out_event, int cuda_device_id);
            void cudaQuantGemm_destroyEvent(void *event);
            bool cudaQuantGemm_recordEvent(void *event, void *stream);
            bool cudaQuantGemm_streamWaitEvent(void *stream, void *event);

            // Upload raw bytes from host to device (nvcc-compiled for CUDA runtime consistency)
            bool cudaQuantGemm_uploadRawBytes(const void *h_src, void **d_dst, size_t bytes, int cuda_device_id);

            // Memory management helpers for fused tensor projections
            bool cudaQuantGemm_allocFloat(float **d_ptr, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyDeviceToDeviceAsync(float *d_dst, const float *d_src, size_t count, int cuda_device_id, void *stream);
            bool cudaQuantGemm_setDevice(int cuda_device_id);
            bool cudaQuantGemm_streamSync(int cuda_device_id, void *stream);

            // -----------------------------------------------------------------
            // Per-shape tensor-core GEMM kernel family (decomposed, fallback)
            // (CUDADecomposedTCGemm.cu)
            // -----------------------------------------------------------------

            bool cudaNativeVNNIGemvTuned_supportsCodebook(uint8_t codebook_id);

            bool cudaNativeVNNIInitIQGridTables_tuned();

            bool cudaNativeVNNIGemvSweep_isActive();

            bool cudaNativeVNNIGemvTuned_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream,
                CUDAGemvContext *gemv_ctx,
                CUDARowMajorWeights **rm_slot);

            bool cudaNativeVNNIGemvTuned_m2_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream,
                CUDAGemvContext *gemv_ctx,
                CUDARowMajorWeights **rm_slot);

            bool cudaNativeVNNIGemvTuned_small_m_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                int M,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream,
                CUDAGemvContext *gemv_ctx,
                CUDARowMajorWeights **rm_slot);

            void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int enabled);
            int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config();

            bool cudaNativeVNNIInitIQGridTables_tuned();

            // Unified prefill dispatch for all formats
            bool cudaNativeVNNIPrefill_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                const int32_t *d_sums_A_block,
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream,
                CUDAPrefillContext *prefill_ctx);

            void cudaPrefillContext_bindWorkspace(
                CUDAPrefillContext *ctx,
                float *splitk_partials,
                size_t splitk_partials_bytes,
                float *streamk_fixup,
                size_t streamk_fixup_bytes);

            bool cudaNativeVNNIPrefill_getWorkspacePlan(
                uint8_t codebook_id,
                int M,
                int N,
                int K,
                int cuda_device_id,
                size_t *splitk_partials_bytes,
                size_t *streamk_fixup_bytes,
                int *planned_split_k,
                int *planned_streamk);

            void cudaNativeVNNIPrefill_getLastLaunchSelection(
                int *tile_id,
                int *split_k,
                int *used_bk256,
                int *used_streamk);

            // cuBLAS FP16 GEMM for Q4_0 native VNNI weights (CUDAcuBLASQuantGemm.cu)
            bool cudaCuBLAS_fp16_gemm_q40(
                const uint8_t *d_payload,
                const uint16_t *d_scales_B,
                const float *d_A_fp32,
                float *d_C_fp32,
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                int cuda_device_id,
                void *stream,
                CUDACuBLASContext *cublas_ctx);
        }

        // =====================================================================
        // Concurrent prefill stream pool (per-device stream/event handles only)
        // =====================================================================

        // Current graph builders can form up to four quantized projection groups
        // that benefit from concurrent prefill dispatch: fused QKV (3), gate/up
        // (2), and GDN q/k/v/z (4). Slot 0 reuses the normal ACC_INT32
        // workspace; the remaining slots live in
        // CUDA_CONCURRENT_PREFILL_ACC_INT32.
        constexpr int kCudaConcurrentPrefillWorkspaceSlots = 4;
        constexpr int kCudaConcurrentPrefillExtraAccumulatorSlots =
            kCudaConcurrentPrefillWorkspaceSlots - 1;
        // Decode projection fan-out can exceed the stream count, especially
        // for MoE top-k gate/up decode where 8 experts become 16 projections.
        // Only this many stream slots can be active at once; later projections
        // reuse a slot after that slot's completion event is observed.
        constexpr int kCudaConcurrentDecodeWorkspaceSlots = 8;

        /**
         * @brief Select serial-decode-shaped CUDA NativeVNNI dispatch inside verifier hooks.
         *
         * The normal CUDA generated table optimizes M=2..4 grouped GEMV rows
         * independently from the M=1 decode route.  That is fine for throughput
         * microbenchmarks, but MTP verifier rows can be published into live KV
         * and GDN state.  While this scope is active, small-M grouped kernels
         * keep batching rows but choose the generated M=1 dispatch policy.  The
         * scope also disables CUDA prefill/concurrent decode paths that would
         * otherwise need extra verifier-only workspace and reorder reductions.
         * Together these choices give each codebook the same reduction contract
         * as serial decode without turning on process-wide deterministic mode.
         */
        class ScopedNativeVNNIDecodeEquivalentDispatch final : public ITensorGemm::VerifierKernelModeScope
        {
        public:
            ScopedNativeVNNIDecodeEquivalentDispatch()
                : previous_gemv_(cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config()),
                  previous_prefill_(cudaNativeVNNIPrefill_getDeterministicMode())
            {
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
                cudaNativeVNNIPrefill_setDeterministicMode(true);
            }

            ~ScopedNativeVNNIDecodeEquivalentDispatch()
            {
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(previous_gemv_);
                cudaNativeVNNIPrefill_setDeterministicMode(previous_prefill_);
            }

            ScopedNativeVNNIDecodeEquivalentDispatch(const ScopedNativeVNNIDecodeEquivalentDispatch &) = delete;
            ScopedNativeVNNIDecodeEquivalentDispatch &operator=(const ScopedNativeVNNIDecodeEquivalentDispatch &) = delete;

        private:
            int previous_gemv_ = 0;
            bool previous_prefill_ = false;
        };

        bool useCanonicalM1SmallMDecode()
        {
            return debugEnv().gemm.deterministic ||
                   cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config() != 0;
        }

        struct CUDAConcurrentPrefillPool
        {
            static constexpr int MAX_STREAMS = 8;

            void *streams[MAX_STREAMS] = {};
            void *completion[MAX_STREAMS] = {};
            void *quant_ready = nullptr;
            int count = 0;
            int device_id = -1;
            bool initialized = false;

            void init(int dev_id, int num_streams)
            {
                if (initialized)
                    return;
                (void)num_streams;
                device_id = dev_id;
                // Create the full stream/event set during eager setup so a
                // later graph capture never has to grow the pool.
                count = MAX_STREAMS;
                for (int i = 0; i < count; ++i)
                {
                    cudaQuantGemm_createStream(&streams[i], dev_id);
                    cudaQuantGemm_createEvent(&completion[i], dev_id);
                }
                cudaQuantGemm_createEvent(&quant_ready, dev_id);
                initialized = true;
                LOG_DEBUG("[CUDAConcurrentPrefillPool] Initialized " << count
                                                                     << " streams on device " << dev_id);
            }

            void destroy()
            {
                if (!initialized)
                    return;
                for (int i = 0; i < count; ++i)
                {
                    cudaQuantGemm_destroyStream(streams[i]);
                    streams[i] = nullptr;
                    cudaQuantGemm_destroyEvent(completion[i]);
                    completion[i] = nullptr;
                }
                cudaQuantGemm_destroyEvent(quant_ready);
                quant_ready = nullptr;
                initialized = false;
                count = 0;
            }

            ~CUDAConcurrentPrefillPool() { destroy(); }
        };

        // =====================================================================
        // PIMPL implementation struct
        // =====================================================================

        struct CUDAQuantisedGemmKernel::Impl
        {
            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            uint8_t *d_weights_native_vnni = nullptr;
            uint16_t *d_weights_native_scales = nullptr;
            uint16_t *d_weights_native_mins = nullptr;
            uint32_t *d_weights_native_emins = nullptr;
            uint8_t native_codebook_id = 0;
            uint32_t native_blocks_per_row = 0;

            // Per-device contexts (replaces process-global static state)
            mutable CUDAGemvContext *gemv_ctx = nullptr;
            mutable CUDAPrefillContext *prefill_ctx = nullptr;
            mutable CUDACuBLASContext *cublas_ctx = nullptr;

            // Work buffers - ALWAYS from workspace (never owned by kernel)
            // These pointers are set from workspace in validateWorkspace()
            int8_t *d_A_int8 = nullptr;   // [M x K] - from workspace
            float *d_scales_A = nullptr;  // [M] - from workspace
            int32_t *d_C_int32 = nullptr; // [M x N] - from workspace

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            ~Impl()
            {
                // Only free weight memory if we own it (not from CUDAPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_native_vnni)
                        cudaQuantGemm_freeDevice(d_weights_native_vnni);
                    if (d_weights_native_scales)
                        cudaQuantGemm_freeDevice(d_weights_native_scales);
                    if (d_weights_native_mins)
                        cudaQuantGemm_freeDevice(d_weights_native_mins);
                    if (d_weights_native_emins)
                        cudaQuantGemm_freeDevice(d_weights_native_emins);
                }
                // Per-device contexts
                if (gemv_ctx)
                {
                    cudaGemvContext_destroy(gemv_ctx);
                    gemv_ctx = nullptr;
                }
                if (prefill_ctx)
                {
                    cudaPrefillContext_destroy(prefill_ctx);
                    prefill_ctx = nullptr;
                }
                if (cublas_ctx)
                {
                    cudaCuBLASContext_destroy(cublas_ctx);
                    cublas_ctx = nullptr;
                }
                // Work buffers are NEVER owned by kernel - they come from workspace
            }
        };

        namespace
        {
            size_t paddedNativePrefillM(int m)
            {
                return (m > 1) ? static_cast<size_t>((m + 127) & ~127) : static_cast<size_t>(m);
            }

            size_t concurrentPrefillScratchSlotsForM(int m)
            {
                return (m > 16 && debugEnv().gemm.cuda_concurrent_prefill)
                           ? static_cast<size_t>(kCudaConcurrentPrefillWorkspaceSlots)
                           : 1ULL;
            }

            size_t paddedSplitKPartialBytes(int m, int n, int split_k)
            {
                if (m <= 0 || n <= 0 || split_k <= 1)
                    return 0;
                return static_cast<size_t>(split_k) *
                       paddedNativePrefillM(m) *
                       static_cast<size_t>(n) *
                       sizeof(float);
            }

            template <typename T>
            bool uploadHostArrayToDevice(
                const std::vector<T> &host,
                T **device,
                int cuda_device_id)
            {
                *device = nullptr;
                if (host.empty())
                {
                    return true;
                }

                const size_t bytes = host.size() * sizeof(T);
                void *d_ptr = nullptr;
                if (!cudaQuantGemm_uploadRawBytes(host.data(), &d_ptr, bytes, cuda_device_id))
                {
                    return false;
                }
                *device = reinterpret_cast<T *>(d_ptr);
                return true;
            }

            void freeDeviceUploadNativeBuffers(CUDAPackedWeights::DeviceUpload &upload)
            {
                if (upload.d_native_vnni)
                    cudaQuantGemm_freeDevice(upload.d_native_vnni);
                if (upload.d_native_scales)
                    cudaQuantGemm_freeDevice(upload.d_native_scales);
                if (upload.d_native_mins)
                    cudaQuantGemm_freeDevice(upload.d_native_mins);
                if (upload.d_native_emins)
                    cudaQuantGemm_freeDevice(upload.d_native_emins);

                upload.d_native_vnni = nullptr;
                upload.d_native_scales = nullptr;
                upload.d_native_mins = nullptr;
                upload.d_native_emins = nullptr;
            }

            bool uploadNativePackedWeights(
                const CUDAPackedWeights &packed,
                CUDAPackedWeights::DeviceUpload &upload,
                int cuda_device_id)
            {
                if (!uploadHostArrayToDevice(packed.native_vnni, &upload.d_native_vnni, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_scales, &upload.d_native_scales, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_mins, &upload.d_native_mins, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_emins, &upload.d_native_emins, cuda_device_id))
                {
                    freeDeviceUploadNativeBuffers(upload);
                    return false;
                }

                return true;
            }

            bool runSelectedBlockwiseBackend(
                const int8_t * /*d_A_int8*/,
                const int8_t * /*d_weights_int8*/,
                const int8_t * /*d_weights_int8_tc_blocked*/,
                int32_t * /*d_partial_int32*/,
                float * /*d_C_fp32*/,
                const float * /*d_scales_A_blockwise*/,
                const float * /*d_scales_B*/,
                int /*m*/, int /*n*/, int /*k*/,
                float /*alpha*/, float /*beta*/,
                const float * /*d_C_existing*/,
                const float * /*d_bias*/,
                int /*cuda_device_id*/,
                void * /*stream*/)
            {
                // TC/CUTLASS fallback paths have been removed.
                // NativeVNNI is the only CUDA GEMM path.
                LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise TC/CUTLASS fallback called but has been removed. "
                          "NativeVNNI is now the only CUDA GEMM path.");
                return false;
            }

            // Returns true if a codebook has native VNNI prefill support.
            bool nativeVNNIPrefillSupportsCodebook(uint8_t cb)
            {
                switch (cb)
                {
                case 0:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 16:
                case 17:
                case 19:
                    return true;
                default:
                    return false;
                }
            }

            bool nativeVNNIUsesAsymmetricCorrection(uint8_t cb)
            {
                switch (cb)
                {
                case 5:  // Q4_1 / Q4_K
                case 7:  // Q5_1 / Q5_K
                case 16: // IQ1_S
                    return true;
                default:
                    return false;
                }
            }

            template <typename ImplT>
            bool canUseNativeVNNIBlockwise(const ImplT *impl, int m, int k)
            {
                return impl &&
                       m > 0 &&
                       k > 0 &&
                       (k % 32) == 0 &&
                       impl->d_weights_native_vnni &&
                       impl->d_weights_native_scales &&
                       (m == 1
                            ? cudaNativeVNNIGemvTuned_supportsCodebook(impl->native_codebook_id)
                            : nativeVNNIPrefillSupportsCodebook(impl->native_codebook_id));
            }

            bool ensureNativeVNNIIQGridTablesInitialized(uint8_t codebook_id, int cuda_device_id, const char *context)
            {
                const bool needs_iq_tables = codebook_id >= 11 && codebook_id <= 17;
                if (!needs_iq_tables)
                {
                    return true;
                }

                static std::mutex iq_table_mutex;
                static std::unordered_set<int> iq_init_devices;

                std::lock_guard<std::mutex> lock(iq_table_mutex);
                if (iq_init_devices.count(cuda_device_id))
                {
                    return true;
                }

                if (!cudaQuantGemm_setDevice(cuda_device_id))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] " << context
                                                           << " failed to set CUDA device " << cuda_device_id
                                                           << " before IQ grid table initialization");
                    return false;
                }
                if (!cudaNativeVNNIInitIQGridTables_tuned())
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] " << context
                                                           << " failed to initialize IQ grid tables for CUDA device "
                                                           << cuda_device_id);
                    return false;
                }

                iq_init_devices.insert(cuda_device_id);
                return true;
            }

            template <typename ImplT>
            bool runNativeVNNIBlockwiseIfSupported(
                const ImplT *impl,
                const int8_t *d_A_int8,
                int32_t *d_partial_int32,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int m, int n, int k,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream,
                CUDARowMajorWeights **rm_slot = nullptr,
                const int32_t *d_sums_A_blockwise = nullptr)
            {
                if (!impl || m <= 0 || k <= 0 || (k % 32) != 0)
                {
                    return false;
                }

                if (!ensureNativeVNNIIQGridTablesInitialized(
                        impl->native_codebook_id,
                        cuda_device_id,
                        "NativeVNNI blockwise route"))
                {
                    return false;
                }

                if (impl->d_weights_native_vnni &&
                    impl->d_weights_native_scales &&
                    m == 1)
                {
                    static std::once_flag native_vnni_decode_once;
                    std::call_once(native_vnni_decode_once, [&]()
                                   { LOG_DEBUG("[CUDAQuantisedGemmKernel] NativeVNNI tuned GEMV decode enabled for supported CUDA codebooks"); });

                    // Lazy-create per-device GEMV context (SM count, kpar partials)
                    if (!impl->gemv_ctx)
                        impl->gemv_ctx = cudaGemvContext_create(cuda_device_id);

                    const bool gemv_ok = cudaNativeVNNIGemvTuned_fp32(
                        d_A_int8,
                        impl->d_weights_native_vnni,
                        impl->d_weights_native_scales,
                        impl->d_weights_native_mins,
                        impl->d_weights_native_emins,
                        d_C_fp32,
                        d_scales_A_blockwise,
                        n, k,
                        alpha, beta,
                        d_C_existing,
                        d_bias,
                        impl->native_codebook_id,
                        cuda_device_id,
                        stream,
                        impl->gemv_ctx,
                        rm_slot);
                    if (!gemv_ok)
                    {
                        cudaError_t le = cudaPeekAtLastError();
                        LOG_ERROR("[CUDAQuantisedGemmKernel] NativeVNNI GEMV dispatch failed"
                                  << " codebook=" << static_cast<int>(impl->native_codebook_id)
                                  << " M=" << m << " N=" << n << " K=" << k
                                  << " stream=" << stream
                                  << " cuda_error=" << cudaGetErrorName(le)
                                  << " (" << cudaGetErrorString(le) << ")"
                                  << " rowmajor_slot=" << (rm_slot && *rm_slot ? "present" : "absent"));
                    }
                    return gemv_ok;
                }

                // Unified native VNNI prefill for all supported codebooks
                if (impl->d_weights_native_vnni &&
                    impl->d_weights_native_scales &&
                    nativeVNNIPrefillSupportsCodebook(impl->native_codebook_id))
                {
                    static std::once_flag native_vnni_prefill_once;
                    std::call_once(native_vnni_prefill_once, [&]()
                                   { LOG_DEBUG("[CUDAQuantisedGemmKernel] NativeVNNI prefill kernel active (codebook " << static_cast<int>(impl->native_codebook_id) << ")"); });

                    // Lazy-create per-device prefill context (stream-K fixup buffer + SM count)
                    if (!impl->prefill_ctx)
                        impl->prefill_ctx = cudaPrefillContext_create(cuda_device_id);

                    if (cudaNativeVNNIPrefill_fp32(
                            d_A_int8,
                            impl->d_weights_native_vnni,
                            impl->d_weights_native_scales,
                            impl->d_weights_native_mins,
                            impl->d_weights_native_emins,
                            d_C_fp32,
                            d_scales_A_blockwise,
                            d_sums_A_blockwise,
                            m, n, k,
                            alpha, beta,
                            d_C_existing,
                            d_bias,
                            impl->native_codebook_id,
                            cuda_device_id,
                            stream,
                            impl->prefill_ctx))
                    {
                        return true;
                    }

                    int tile_id = -99;
                    int split_k = -1;
                    int used_bk256 = 0;
                    int used_streamk = 0;
                    cudaNativeVNNIPrefill_getLastLaunchSelection(
                        &tile_id, &split_k, &used_bk256, &used_streamk);
                    LOG_ERROR("[CUDAQuantisedGemmKernel] NativeVNNI prefill kernel failed for codebook "
                              << static_cast<int>(impl->native_codebook_id)
                              << " M=" << m << " N=" << n << " K=" << k
                              << " tile_id=" << tile_id
                              << " split_k=" << split_k
                              << " bk256=" << used_bk256
                              << " streamk=" << used_streamk
                              << " (no fallback available — TC/CUTLASS paths have been removed)");
                }

                return false;
            }
        }

        // Static method stubs kept for ABI compatibility but are no-ops.
        // NativeVNNI is now always enabled; CUTLASS fallback no longer exists.
        void CUDAQuantisedGemmKernel::setNativeVNNIEnabled(bool /*enabled*/) {}
        bool CUDAQuantisedGemmKernel::isNativeVNNIEnabled() { return true; }
        void CUDAQuantisedGemmKernel::setForceCutlassFallback(bool /*enabled*/) {}
        bool CUDAQuantisedGemmKernel::isForceCutlassFallback() { return false; }

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(const TensorBase *weights, int cuda_device_id)
            : weights_(weights),
              packed_(nullptr),
              cuda_device_id_(cuda_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),
              owns_weight_memory_(true), // Legacy path owns weight memory
              impl_(std::make_unique<Impl>())
        {
            if (!weights)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Null weight tensor");
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Validate it's a quantized type
            TensorType wt = weights->native_type();
            bool is_quantized = (wt == TensorType::IQ4_NL ||
                                 wt == TensorType::Q8_0 ||
                                 wt == TensorType::Q4_0 ||
                                 wt == TensorType::Q4_1 ||
                                 wt == TensorType::Q5_0 ||
                                 wt == TensorType::Q5_1 ||
                                 wt == TensorType::Q4_K ||
                                 wt == TensorType::Q5_K ||
                                 wt == TensorType::Q6_K ||
                                 wt == TensorType::Q8_K ||
                                 wt == TensorType::Q2_K ||
                                 wt == TensorType::Q3_K ||
                                 wt == TensorType::Q8_1 ||
                                 wt == TensorType::IQ4_XS ||
                                 wt == TensorType::IQ2_XXS ||
                                 wt == TensorType::IQ2_XS ||
                                 wt == TensorType::IQ3_XXS ||
                                 wt == TensorType::IQ2_S ||
                                 wt == TensorType::IQ3_S ||
                                 wt == TensorType::IQ1_S ||
                                 wt == TensorType::IQ1_M);

            if (!is_quantized)
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Weight tensor must be quantized type, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            impl_->owns_weight_memory = true; // Legacy constructor owns weight memory

            LOG_DEBUG("[CUDAQuantisedGemmKernel] Created (legacy) for " << N_ << "x" << K_
                                                                        << " quantized weights (type=" << static_cast<int>(wt)
                                                                        << ") on CUDA device " << cuda_device_id_);
        }

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(CUDAPackedWeights *packed, int cuda_device_id)
            : weights_(nullptr),
              packed_(packed),
              cuda_device_id_(cuda_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),  // Not yet uploaded to device
              owns_weight_memory_(false), // CUDAPackedWeights owns the memory
              impl_(std::make_unique<Impl>())
        {
            if (!packed)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Null packed weights");
            }

            N_ = static_cast<size_t>(packed->N);
            K_ = static_cast<size_t>(packed->K);

            impl_->owns_weight_memory = false; // Packed cache owns weight memory

            LOG_DEBUG("[CUDAQuantisedGemmKernel] Created (pre-packed) for " << N_ << "x" << K_
                                                                            << " INT8 weights on CUDA device " << cuda_device_id_);
        }

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(
            int N, int K, int cuda_device_id,
            uint8_t *d_vnni, uint16_t *d_scales, uint16_t *d_mins, uint32_t *d_emins,
            uint8_t codebook_id, uint32_t blocks_per_row,
            std::shared_ptr<void> lifetime_owner)
            : weights_(nullptr),
              packed_(nullptr),
              lifetime_owner_(std::move(lifetime_owner)),
              cuda_device_id_(cuda_device_id),
              N_(static_cast<size_t>(N)),
              K_(static_cast<size_t>(K)),
              weights_converted_(true),   // Already on device
              owns_weight_memory_(false), // Shared batch allocation owns it
              impl_(std::make_unique<Impl>())
        {
            impl_->d_weights_native_vnni = d_vnni;
            impl_->d_weights_native_scales = d_scales;
            impl_->d_weights_native_mins = d_mins;
            impl_->d_weights_native_emins = d_emins;
            impl_->native_codebook_id = codebook_id;
            impl_->native_blocks_per_row = blocks_per_row;
            impl_->owns_weight_memory = false;

            LOG_DEBUG("[CUDAQuantisedGemmKernel] Created (MoE batch) for " << N_ << "x" << K_
                                                                           << " on CUDA device " << cuda_device_id_);
        }

        CUDAQuantisedGemmKernel::~CUDAQuantisedGemmKernel() = default;

        // ---------------------------------------------------------------------
        // Shared per-device prefill stream pool.
        //
        // Multiple kernels on the same CUDA device share a single pool so that
        // we don't allocate duplicate scratch buffers / streams per kernel
        // instance. KernelFactory::clearCache() invokes
        // clearSharedPrefillPools() to release these between test runs.
        // ---------------------------------------------------------------------
        namespace
        {
            std::mutex &sharedPrefillPoolsMutex()
            {
                static std::mutex m;
                return m;
            }

            std::unordered_map<int, std::unique_ptr<CUDAConcurrentPrefillPool>> &sharedPrefillPools()
            {
                static std::unordered_map<int, std::unique_ptr<CUDAConcurrentPrefillPool>> pools;
                return pools;
            }

            CUDAConcurrentPrefillPool &getSharedCUDAPrefillPool(int cuda_device_id)
            {
                std::lock_guard<std::mutex> lk(sharedPrefillPoolsMutex());
                auto &pools = sharedPrefillPools();
                auto it = pools.find(cuda_device_id);
                if (it == pools.end())
                {
                    it = pools.emplace(cuda_device_id, std::make_unique<CUDAConcurrentPrefillPool>()).first;
                }
                return *it->second;
            }
        } // namespace

        void CUDAQuantisedGemmKernel::clearSharedPrefillPools()
        {
            std::lock_guard<std::mutex> lk(sharedPrefillPoolsMutex());
            sharedPrefillPools().clear();
        }

        void CUDAQuantisedGemmKernel::resetDynamicState()
        {
            if (!impl_)
                return;

            /*
             * Request/session reset must not destroy GEMM contexts. CUDA graph
             * executables capture kernel parameters by value, including pointers
             * to context-owned handles or workspace bindings used by the native
             * GEMV/prefill/cuBLAS paths. Those contexts are request-independent
             * resources and are released by Impl destruction or full KernelFactory
             * cache teardown. Keeping them alive lets preserved prefill graphs
             * replay correctly after clearCache(); workspace rebinding still
             * invalidates executable graph captures at the ForwardGraphCache layer.
             */
            gpu_stream_ = nullptr;
        }

        bool CUDAQuantisedGemmKernel::hasDynamicStateActive() const
        {
            return gpu_stream_ != nullptr;
        }

        bool CUDAQuantisedGemmKernel::weights_converted() const
        {
            /*
             * Prepared-weight CUDA kernels can be constructed directly from
             * model-lifetime WeightVRAMPool native-VNNI descriptors.  The
             * resident descriptor pointers are the execution contract for
             * grouped MoE verifier kernels; weights_converted_ remains the
             * legacy lazy-upload flag for host-weight constructors.
             */
            return weights_converted_ ||
                   (impl_ &&
                    impl_->d_weights_native_vnni &&
                    impl_->d_weights_native_scales &&
                    N_ > 0 &&
                    K_ > 0 &&
                    impl_->native_blocks_per_row > 0 &&
                    nativeVNNIPrefillSupportsCodebook(impl_->native_codebook_id));
        }

        bool CUDAQuantisedGemmKernel::exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out)
        {
            out = {};
            try
            {
                ensureWeightsConverted();
            }
            catch (const std::exception &ex)
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel] Cannot export native-VNNI descriptor: " << ex.what());
                return false;
            }

            if (!impl_ || !impl_->d_weights_native_vnni || !impl_->d_weights_native_scales ||
                N_ == 0 || K_ == 0 || impl_->native_blocks_per_row == 0 ||
                !nativeVNNIPrefillSupportsCodebook(impl_->native_codebook_id))
            {
                return false;
            }

            out.payload = impl_->d_weights_native_vnni;
            out.scales = impl_->d_weights_native_scales;
            out.mins = impl_->d_weights_native_mins;
            out.emins = impl_->d_weights_native_emins;
            out.n = static_cast<int>(N_);
            out.k = static_cast<int>(K_);
            out.blocks_per_row = impl_->native_blocks_per_row;
            out.codebook_id = impl_->native_codebook_id;
            return out.valid();
        }

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(CUDAQuantisedGemmKernel &&other) noexcept
            : weights_(other.weights_),
              packed_(other.packed_),
              lifetime_owner_(std::move(other.lifetime_owner_)),
              cuda_device_id_(other.cuda_device_id_),
              N_(other.N_),
              K_(other.K_),
              weights_converted_(other.weights_converted_),
              owns_weight_memory_(other.owns_weight_memory_),
              impl_(std::move(other.impl_))
        {
            other.weights_ = nullptr;
            other.packed_ = nullptr;
            other.weights_converted_ = false;
            other.owns_weight_memory_ = false;
        }

        CUDAQuantisedGemmKernel &CUDAQuantisedGemmKernel::operator=(CUDAQuantisedGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                weights_ = other.weights_;
                packed_ = other.packed_;
                lifetime_owner_ = std::move(other.lifetime_owner_);
                cuda_device_id_ = other.cuda_device_id_;
                N_ = other.N_;
                K_ = other.K_;
                weights_converted_ = other.weights_converted_;
                owns_weight_memory_ = other.owns_weight_memory_;
                impl_ = std::move(other.impl_);

                other.weights_ = nullptr;
                other.packed_ = nullptr;
                other.weights_converted_ = false;
                other.owns_weight_memory_ = false;
            }
            return *this;
        }

        std::unique_ptr<ITensorGemm::VerifierKernelModeScope>
        CUDAQuantisedGemmKernel::beginVerifierDecodeEquivalentScope()
        {
            return std::make_unique<ScopedNativeVNNIDecodeEquivalentDispatch>();
        }

        // =====================================================================
        // Weight conversion: Any quantized format → INT8 + scales
        // =====================================================================

        void CUDAQuantisedGemmKernel::ensureWeightsConverted()
        {
            LOG_DEBUG("[CUDAQuantisedGemmKernel::ensureWeightsConverted] Entry: N_=" << N_ << " K_=" << K_
                                                                                     << " weights_converted_=" << weights_converted_
                                                                                     << " d_native_vnni=" << (impl_ ? (void *)impl_->d_weights_native_vnni : nullptr)
                                                                                     << " d_native_scales=" << (impl_ ? (void *)impl_->d_weights_native_scales : nullptr));
            if (weights_converted_)
            {
                return;
            }

            // Pre-packed path: upload from CUDAPackedWeights
            if (packed_)
            {
                std::lock_guard<std::mutex> lock(packed_->upload_mutex);

                auto upload_it = packed_->device_uploads.find(cuda_device_id_);
                if (upload_it == packed_->device_uploads.end())
                {
                    CUDAPackedWeights::DeviceUpload upload;

                    // NativeVNNI is the only CUDA GEMM path.
                    // No Int8Expanded or TC-blocked weights are uploaded.
                    static std::once_flag vnni_only_once;
                    std::call_once(vnni_only_once, [&]()
                                   { LOG_DEBUG("[CUDAQuantisedGemmKernel] NativeVNNI-only mode (codebook "
                                               << static_cast<int>(packed_->native_codebook_id) << ")"); });

                    if (!uploadNativePackedWeights(*packed_, upload, cuda_device_id_))
                    {
                        throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload pre-packed native buffers");
                    }

                    auto emplaced = packed_->device_uploads.emplace(cuda_device_id_, upload);
                    upload_it = emplaced.first;
                }

                const auto &upload = upload_it->second;
                packed_->d_native_vnni = upload.d_native_vnni;
                packed_->d_native_scales = upload.d_native_scales;
                packed_->d_native_mins = upload.d_native_mins;
                packed_->d_native_emins = upload.d_native_emins;
                packed_->cuda_device_id = cuda_device_id_;
                packed_->uploaded = true;

                impl_->d_weights_native_vnni = upload.d_native_vnni;
                impl_->d_weights_native_scales = upload.d_native_scales;
                impl_->d_weights_native_mins = upload.d_native_mins;
                impl_->d_weights_native_emins = upload.d_native_emins;
                impl_->native_codebook_id = packed_->native_codebook_id;
                impl_->native_blocks_per_row = packed_->native_blocks_per_row;
                weights_converted_ = true;

                // Release host-side packing buffers — data is now on GPU.
                // This saves ~2× the quantized weight size of host memory.
                // Only safe when this packed_ won't be uploaded to additional devices;
                // we check device_uploads.size() == 1 as a proxy (TP shards have
                // separate packed_ per shard, so this is typically the only device).
                if (packed_->device_uploads.size() <= 1)
                {
                    const size_t freed_bytes =
                        packed_->native_vnni.capacity() +
                        packed_->native_scales.capacity() * sizeof(uint16_t) +
                        packed_->native_mins.capacity() * sizeof(uint16_t) +
                        packed_->native_emins.capacity() * sizeof(uint32_t);
                    packed_->native_vnni.clear();
                    packed_->native_vnni.shrink_to_fit();
                    packed_->native_scales.clear();
                    packed_->native_scales.shrink_to_fit();
                    packed_->native_mins.clear();
                    packed_->native_mins.shrink_to_fit();
                    packed_->native_emins.clear();
                    packed_->native_emins.shrink_to_fit();
                    if (freed_bytes > 0)
                    {
                        LOG_DEBUG("[CUDAQuantisedGemmKernel] Released host packing buffers: "
                                  << (freed_bytes / (1024 * 1024)) << " MB");
                    }
                }

                LOG_DEBUG("[CUDAQuantisedGemmKernel] Using cached pre-packed weights on CUDA:" << cuda_device_id_);
                return;
            }

            // Legacy path: convert from raw tensor
            LOG_DEBUG("[CUDAQuantisedGemmKernel] Converting weights to INT8 (legacy path)...");

            if (!weights_)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] No weights or packed data available");
            }

            // Get dequantized FP32 from tensor
            const float *h_weights_fp32 = weights_->data();
            if (!h_weights_fp32)
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Failed to get FP32 data from weight tensor");
            }

            // DEBUG: Print first few weight values to verify slicing
            if (N_ == 64 && K_ == 896) // This is the K weight shape for LOCAL TP
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel DEBUG] K weight N=" << N_ << " K=" << K_
                                                                        << " device=" << cuda_device_id_
                                                                        << " first 5 weights[0]: " << h_weights_fp32[0] << ", "
                                                                        << h_weights_fp32[1] << ", " << h_weights_fp32[2] << ", "
                                                                        << h_weights_fp32[3] << ", " << h_weights_fp32[4]);
            }

            CUDAPackedWeights legacy_packed;
            if (!packWeightsToCUDA(weights_, legacy_packed))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Failed to pack converted weights");
            }

            // Upload native VNNI weights to device (NativeVNNI-only, no Int8Expanded upload)
            if (!uploadNativePackedWeights(legacy_packed, legacy_packed.device_uploads[cuda_device_id_], cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload converted native buffers");
            }

            impl_->d_weights_native_vnni = legacy_packed.device_uploads[cuda_device_id_].d_native_vnni;
            impl_->d_weights_native_scales = legacy_packed.device_uploads[cuda_device_id_].d_native_scales;
            impl_->d_weights_native_mins = legacy_packed.device_uploads[cuda_device_id_].d_native_mins;
            impl_->d_weights_native_emins = legacy_packed.device_uploads[cuda_device_id_].d_native_emins;
            impl_->native_codebook_id = legacy_packed.native_codebook_id;
            impl_->native_blocks_per_row = legacy_packed.native_blocks_per_row;

            weights_converted_ = true;
            LOG_DEBUG("[CUDAQuantisedGemmKernel] Weight conversion complete (legacy)");
        }

        void CUDAQuantisedGemmKernel::validateWorkspace() const
        {
            // Kernels REQUIRE workspace - no internal buffer allocation
            if (!hasWorkspace())
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace not bound. Kernels require pre-allocated "
                    "workspace buffers via bindWorkspace(). Call bindWorkspace() with a "
                    "DeviceWorkspaceManager that has allocated the buffers from getWorkspaceRequirements().");
            }

            // Validate required buffers exist
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::QUANT_A))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::QUANT_A));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::SCALES_A));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ACC_INT32))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::ACC_INT32));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE));
            }

            LOG_TRACE("[CUDAQuantisedGemmKernel::validateWorkspace] Workspace validated"
                      << " A_int8=" << workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A)
                      << " scales_A=" << workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A)
                      << " scales_A_blockwise=" << workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)
                      << " sums_A_blockwise=" << workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE)
                      << " C_int32=" << workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

            if (!impl_->prefill_ctx)
            {
                impl_->prefill_ctx = cudaPrefillContext_create(cuda_device_id_);
            }
            if (!impl_->gemv_ctx)
            {
                impl_->gemv_ctx = cudaGemvContext_create(cuda_device_id_);
            }
            if (impl_->gemv_ctx)
            {
                float *kpar_partials = nullptr;
                size_t kpar_partials_bytes = 0;
                if (workspace_->hasBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS))
                {
                    kpar_partials = static_cast<float *>(
                        workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS));
                    kpar_partials_bytes =
                        workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                }

                cudaGemvContext_bindWorkspace(
                    impl_->gemv_ctx,
                    kpar_partials,
                    kpar_partials_bytes);
            }
            if (impl_->prefill_ctx)
            {
                float *splitk_partials = nullptr;
                size_t splitk_partials_bytes = 0;
                if (workspace_->hasBuffer(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS))
                {
                    splitk_partials = static_cast<float *>(
                        workspace_->getBuffer(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS));
                    splitk_partials_bytes =
                        workspace_->getBufferSize(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS);
                }

                float *streamk_fixup = nullptr;
                size_t streamk_fixup_bytes = 0;
                if (workspace_->hasBuffer(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP))
                {
                    streamk_fixup = static_cast<float *>(
                        workspace_->getBuffer(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP));
                    streamk_fixup_bytes =
                        workspace_->getBufferSize(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP);
                }

                cudaPrefillContext_bindWorkspace(
                    impl_->prefill_ctx,
                    splitk_partials,
                    splitk_partials_bytes,
                    streamk_fixup,
                    streamk_fixup_bytes);
            }
        }

        void CUDAQuantisedGemmKernel::bindConcurrentNativePrefillScratch(
            int m,
            int n,
            int k,
            int stream_idx) const
        {
            if (!impl_ || !impl_->prefill_ctx || !workspace_ || m <= 1)
                return;

            size_t splitk_bytes = 0;
            size_t streamk_bytes = 0;
            int planned_split_k = 1;
            int planned_streamk = 0;
            if (!cudaNativeVNNIPrefill_getWorkspacePlan(
                    impl_->native_codebook_id,
                    m,
                    n,
                    k,
                    cuda_device_id_,
                    &splitk_bytes,
                    &streamk_bytes,
                    &planned_split_k,
                    &planned_streamk))
            {
                return;
            }

            splitk_bytes = std::max(splitk_bytes, paddedSplitKPartialBytes(m, n, planned_split_k));
            if (splitk_bytes == 0 && streamk_bytes == 0)
                return;

            if (stream_idx < 0 || stream_idx >= kCudaConcurrentPrefillWorkspaceSlots)
            {
                throw std::runtime_error(
                    "[ConcurrentPrefill] NativeVNNI scratch stream slot " +
                    std::to_string(stream_idx) + " is outside the declared workspace slot range");
            }

            float *splitk_ptr = nullptr;
            size_t splitk_slot_bytes = 0;
            if (splitk_bytes > 0)
            {
                void *buffer = workspace_->getBuffer(
                    GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS);
                const size_t total_bytes = workspace_->getBufferSize(
                    GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS);
                splitk_slot_bytes = total_bytes / static_cast<size_t>(kCudaConcurrentPrefillWorkspaceSlots);
                const size_t required_total =
                    splitk_bytes * static_cast<size_t>(kCudaConcurrentPrefillWorkspaceSlots);
                if (!buffer || total_bytes < required_total || splitk_slot_bytes < splitk_bytes)
                {
                    throw std::runtime_error(
                        "[ConcurrentPrefill] " +
                        std::string(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS) +
                        " workspace is missing or undersized for concurrent split-K projection: need total " +
                        std::to_string(required_total) + " bytes (" +
                        std::to_string(splitk_bytes) + " per slot), have " +
                        std::to_string(total_bytes));
                }
                auto *base = static_cast<unsigned char *>(buffer);
                splitk_ptr = reinterpret_cast<float *>(
                    base + static_cast<size_t>(stream_idx) * splitk_slot_bytes);
            }

            float *streamk_ptr = nullptr;
            size_t streamk_slot_bytes = 0;
            if (streamk_bytes > 0)
            {
                void *buffer = workspace_->getBuffer(
                    GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP);
                const size_t total_bytes = workspace_->getBufferSize(
                    GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP);
                streamk_slot_bytes = total_bytes / static_cast<size_t>(kCudaConcurrentPrefillWorkspaceSlots);
                const size_t required_total =
                    streamk_bytes * static_cast<size_t>(kCudaConcurrentPrefillWorkspaceSlots);
                if (!buffer || total_bytes < required_total || streamk_slot_bytes < streamk_bytes)
                {
                    throw std::runtime_error(
                        "[ConcurrentPrefill] " +
                        std::string(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP) +
                        " workspace is missing or undersized for concurrent stream-K projection: need total " +
                        std::to_string(required_total) + " bytes (" +
                        std::to_string(streamk_bytes) + " per slot), have " +
                        std::to_string(total_bytes));
                }
                auto *base = static_cast<unsigned char *>(buffer);
                streamk_ptr = reinterpret_cast<float *>(
                    base + static_cast<size_t>(stream_idx) * streamk_slot_bytes);
            }

            cudaPrefillContext_bindWorkspace(
                impl_->prefill_ctx,
                splitk_ptr,
                splitk_slot_bytes,
                streamk_ptr,
                streamk_slot_bytes);
        }

        void CUDAQuantisedGemmKernel::bindConcurrentNativeDecodeScratch(
            int m,
            int n,
            int k,
            int stream_idx,
            int projection_count) const
        {
            if (!impl_ || !impl_->gemv_ctx || !workspace_ || m <= 0)
                return;

            if (projection_count < 2)
            {
                throw std::runtime_error(
                    "[ConcurrentDecode] NativeVNNI GEMV projection count " +
                    std::to_string(projection_count) + " is outside the declared workspace slot range");
            }

            const int active_slots =
                std::min(projection_count, kCudaConcurrentDecodeWorkspaceSlots);

            if (stream_idx < 0 || stream_idx >= active_slots)
            {
                throw std::runtime_error(
                    "[ConcurrentDecode] NativeVNNI GEMV stream slot " +
                    std::to_string(stream_idx) + " is outside the declared workspace slot range");
            }

            const int rows = std::max(1, std::min(m, 4));
            const int k_groups = (k + 31) / 32;
            const size_t required_bytes =
                static_cast<size_t>(k_groups) * static_cast<size_t>(rows) *
                static_cast<size_t>(n) * sizeof(float);
            if (required_bytes == 0)
                return;

            float *partials = nullptr;
            size_t slot_bytes = 0;
            if (stream_idx == 0)
            {
                void *buffer = workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                slot_bytes = workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                if (!buffer || slot_bytes < required_bytes)
                {
                    throw std::runtime_error(
                        "[ConcurrentDecode] " +
                        std::string(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS) +
                        " workspace is missing or undersized for stream slot 0: need " +
                        std::to_string(required_bytes) + " bytes, have " +
                        std::to_string(slot_bytes));
                }
                partials = static_cast<float *>(buffer);
            }
            else
            {
                void *buffer = workspace_->getBuffer(
                    GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
                const size_t total_bytes = workspace_->getBufferSize(
                    GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
                const int extra_slots = active_slots - 1;
                slot_bytes = total_bytes / static_cast<size_t>(extra_slots);
                if (!buffer || slot_bytes < required_bytes ||
                    stream_idx > extra_slots)
                {
                    throw std::runtime_error(
                        "[ConcurrentDecode] " +
                        std::string(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS) +
                        " workspace is missing or undersized for stream slot " +
                        std::to_string(stream_idx) + ": need " +
                        std::to_string(required_bytes) + " bytes, have " +
                        std::to_string(slot_bytes));
                }

                auto *base = static_cast<unsigned char *>(buffer);
                partials = reinterpret_cast<float *>(
                    base + static_cast<size_t>(stream_idx - 1) * slot_bytes);
            }

            cudaGemvContext_bindWorkspace(
                impl_->gemv_ctx,
                partials,
                slot_bytes);
        }

        // =====================================================================
        // ITensorGemm interface - multiply_tensor() PRIMARY ENTRY POINT
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            (void)bias; // TODO: Implement bias support
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            int m = static_cast<int>(A->rows());
            int n = static_cast<int>(N_);
            int k = static_cast<int>(K_);

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace, activation_row_offset);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool /*transpose_B*/,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_tensor_impl(A, C, m, n, k, alpha, beta, bias, activation_row_offset);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_tensor_impl(A, C, m, n, k, alpha, beta, bias, activation_row_offset);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor_impl(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            float alpha, float beta,
            const TensorBase *bias,
            int activation_row_offset)
        {
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // Coherence handled automatically by DeviceGraphExecutor

            // Ensure weights are converted
            ensureWeightsConverted();

            // Type dispatch based on A and C types
            TensorType a_type = A->native_type();
            TensorType c_type = C->native_type();

            // =================================================================
            // MAPPED OUTPUT REDIRECT: Detect host-mapped FP32 output memory.
            // Mapped memory (used for logits) causes PCIe-speed scattered writes
            // (~12 GB/s) instead of HBM-speed writes (~900 GB/s on RTX 3090).
            // Fix: redirect kernel output to HBM workspace, then bulk DMA.
            // =================================================================
            const bool output_is_mapped = (c_type == TensorType::FP32) && C->isMapped();
            float *d_mapped_output = nullptr; // original mapped pointer for DMA copy

            if (a_type == TensorType::Q8_1 && c_type == TensorType::FP32)
            {
                // Q8_1 → FP32: Use Q8_1 blocks directly
                auto *q8_tensor = dynamic_cast<const Q8_1Tensor *>(A);
                if (!q8_tensor)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast A to Q8_1Tensor");
                    return false;
                }

                auto *fp32_tensor = dynamic_cast<FP32Tensor *>(C);
                if (!fp32_tensor)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast C to FP32Tensor");
                    return false;
                }

                // Get device pointers
                const Q8_1Block *d_A_q8 = static_cast<const Q8_1Block *>(q8_tensor->gpu_data_ptr());
                float *d_C = static_cast<float *>(fp32_tensor->gpu_data_ptr());

                if (!d_A_q8 || !d_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] A and C must be on GPU");
                    return false;
                }

                // Redirect mapped output to HBM workspace
                if (output_is_mapped)
                {
                    validateWorkspace();
                    d_mapped_output = d_C;
                    d_C = static_cast<float *>(workspace_->getBuffer(tempCFp32BufferName()));
                    static std::once_flag q8_mapped_once;
                    std::call_once(q8_mapped_once, [&]()
                                   { LOG_WARN("[CUDAQuantisedGemmKernel] Q8→FP32 MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                                      << " mapped_ptr=" << d_mapped_output << " -> workspace=" << d_C
                                                                                                      << " (" << (static_cast<size_t>(m) * n * 4 / 1024) << " KB)"); });
                }

                bool success = multiply_q8_to_fp32(d_A_q8, d_C, m, n, k, alpha, beta);

                // Bulk DMA from HBM workspace to mapped output
                if (success && output_is_mapped)
                {
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
            else if (a_type == TensorType::FP32 && c_type == TensorType::FP32)
            {
                // FP32 → FP32: Quantize activations on-the-fly
                const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
                float *d_C = static_cast<float *>(C->gpu_data_ptr());

                if (!d_A || !d_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] A and C must be on GPU");
                    return false;
                }

                // Redirect mapped output to HBM workspace
                if (output_is_mapped)
                {
                    validateWorkspace();
                    d_mapped_output = d_C;
                    d_C = static_cast<float *>(workspace_->getBuffer(tempCFp32BufferName()));
                    static std::once_flag fp32_mapped_once;
                    std::call_once(fp32_mapped_once, [&]()
                                   { LOG_WARN("[CUDAQuantisedGemmKernel] FP32→FP32 MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                                        << " mapped_ptr=" << d_mapped_output << " -> workspace=" << d_C
                                                                                                        << " (" << (static_cast<size_t>(m) * n * 4 / 1024) << " KB)"); });
                }

                // Apply activation row offset
                if (activation_row_offset > 0)
                {
                    d_A += static_cast<size_t>(activation_row_offset) * k;
                }

                // Extract bias pointer if present
                const float *d_bias = bias ? static_cast<const float *>(bias->gpu_data_ptr()) : nullptr;

                bool success;
                if (d_bias)
                {
                    success = multiply_fp32_to_fp32_with_bias(d_A, d_C, d_bias, m, n, k, alpha, beta);
                }
                else
                {
                    success = multiply_fp32_to_fp32(d_A, d_C, m, n, k, alpha, beta);
                }

                // Bulk DMA from HBM workspace to mapped output
                if (success && output_is_mapped)
                {
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
            else if (a_type == TensorType::Q8_1 && c_type == TensorType::Q8_1)
            {
                // Q8_1 → Q8_1: Fused requantization
                auto *q8_A = dynamic_cast<const Q8_1Tensor *>(A);
                auto *q8_C = dynamic_cast<Q8_1Tensor *>(C);
                if (!q8_A || !q8_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast tensors");
                    return false;
                }

                const Q8_1Block *d_A_q8 = static_cast<const Q8_1Block *>(q8_A->gpu_data_ptr());
                Q8_1Block *d_C_q8 = static_cast<Q8_1Block *>(q8_C->gpu_data_ptr());

                bool success = multiply_q8_to_q8(d_A_q8, d_C_q8, m, n, k);
                return success;
            }
            else if (a_type == TensorType::FP32 && c_type == TensorType::Q8_1)
            {
                // FP32 → Q8_1: Quantize input, fused requant output
                const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
                auto *q8_C = dynamic_cast<Q8_1Tensor *>(C);
                if (!q8_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast C to Q8_1Tensor");
                    return false;
                }

                Q8_1Block *d_C_q8 = static_cast<Q8_1Block *>(q8_C->gpu_data_ptr());

                bool success = multiply_fp32_to_q8(d_A, d_C_q8, m, n, k);
                return success;
            }
            else
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Unsupported type combination: A="
                          << static_cast<int>(a_type) << ", C=" << static_cast<int>(c_type));
                return false;
            }
        }

        // =====================================================================
        // ITensorGemm interface - multiply_fused_tensor() for TensorBase API
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext * /*mpi_ctx*/,
            DeviceWorkspaceManager *workspace)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_fused_tensor_impl(input, projections, m, k);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_fused_tensor_impl(input, projections, m, k);
        }

        bool CUDAQuantisedGemmKernel::multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            if (m <= 1 || m > 4)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] grouped verifier projection requires M=2..4, got M="
                          << m);
                return false;
            }
            auto decode_equivalent_dispatch = beginVerifierDecodeEquivalentScope();
            return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
        }

        bool CUDAQuantisedGemmKernel::multiply_fused_tensor_impl(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k)
        {
            if (!input || projections.empty())
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }

            if (m <= 0 || k <= 0)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Invalid dimensions: m=" << m << " k=" << k);
                return false;
            }

            if (!gpu_stream_)
                cudaQuantGemm_setDevice(cuda_device_id_);
            DeviceId target_device = DeviceId::cuda(cuda_device_id_);

            // Step 1: Ensure input is on the GPU
            const float *d_input = nullptr;
            if (input->native_type() == TensorType::FP32)
            {
                auto *fp32_input = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(input));
                if (!fp32_input)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast input to FP32Tensor");
                    return false;
                }
                // Coherence handled automatically by DeviceGraphExecutor
                d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
                // NOTE: Don't log fp32_input->data() here - it triggers D2H transfer!
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Input GPU ptr=" << d_input);
            }
            else
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Unsupported input type: "
                          << static_cast<int>(input->native_type()));
                return false;
            }

            if (!d_input)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Input has no GPU data");
                return false;
            }

            if (m > 1 && m <= 4)
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M verifier GEMV path M="
                          << m << " projections=" << projections.size());

                if (!gpu_stream_)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M verifier GEMV path requires an explicit CUDA stream");
                    return false;
                }

                struct SmallMProjectionBinding
                {
                    CUDAQuantisedGemmKernel *kernel = nullptr;
                    float *output = nullptr;
                    const float *bias = nullptr;
                    int n = 0;
                    const char *name = nullptr;
                };

                std::vector<SmallMProjectionBinding> bindings;
                bindings.reserve(projections.size());

                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    if (!proj.kernel || !proj.output)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                  << i << " has null kernel or output");
                        return false;
                    }

                    auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                    if (!cuda_kernel)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                  << i << " kernel is not CUDAQuantisedGemmKernel");
                        return false;
                    }

                    auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_output)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                  << i << " output is not FP32Tensor");
                        return false;
                    }

                    float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                    if (!d_output)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                  << i << " output has no GPU data");
                        return false;
                    }

                    const float *d_bias = nullptr;
                    if (proj.bias)
                    {
                        const TensorBase *bias_tensor = proj.bias;
                        if (auto *slice = dynamic_cast<const TensorSlice *>(proj.bias))
                            bias_tensor = slice->inner();

                        auto *fp32_bias = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(bias_tensor));
                        if (!fp32_bias)
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                      << i << " bias is not FP32Tensor");
                            return false;
                        }

                        auto current_dev = fp32_bias->current_device();
                        if (current_dev.has_value() && current_dev.value() == target_device)
                        {
                            d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                        }
                        else if (current_dev.has_value() && current_dev->is_gpu())
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                      << i << " bias is on " << current_dev->to_string()
                                      << " but CUDA:" << cuda_device_id_ << " is required");
                            return false;
                        }
                        else
                        {
                            if (!fp32_bias->ensureOnDevice(target_device, gpu_stream_))
                            {
                                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                          << i << " failed to upload bias to CUDA:" << cuda_device_id_);
                                return false;
                            }
                            d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                        }
                    }

                    cuda_kernel->setGPUStream(gpu_stream_);
                    cuda_kernel->validateWorkspace();
                    cuda_kernel->ensureWeightsConverted();
                    if (!canUseNativeVNNIBlockwise(cuda_kernel->impl_.get(), 1, k))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M projection "
                                  << i << " does not support native-VNNI decode-equivalent GEMV"
                                  << " codebook="
                                  << (cuda_kernel->impl_
                                          ? std::to_string(static_cast<int>(cuda_kernel->impl_->native_codebook_id))
                                          : std::string("unknown"))
                                  << " N=" << proj.n << " K=" << k);
                        return false;
                    }

                    bindings.push_back(SmallMProjectionBinding{
                        cuda_kernel,
                        d_output,
                        d_bias,
                        proj.n,
                        proj.name});
                }

                // All small-M verifier projections share the same activation rows.
                // Quantize those rows once, then feed the same quantized block into
                // each decode-equivalent GEMV. This removes repeated hot-path work
                // while preserving the exact per-projection GEMV kernels and
                // accumulator order used by the serial verifier path.
                validateWorkspace();
                int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
                float *d_scales_A_blockwise =
                    static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                if (!d_A_int8 || !d_scales_A_blockwise)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M verifier workspace missing quantized activation buffers");
                    return false;
                }

                if (!cudaQuantGemm_quantizeActivationsBlockwise(
                        d_input,
                        d_A_int8,
                        d_scales_A_blockwise,
                        m,
                        k,
                        cuda_device_id_,
                        gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M verifier shared activation quantization failed");
                    return false;
                }

                for (size_t i = 0; i < bindings.size(); ++i)
                {
                    const auto &binding = bindings[i];
                    binding.kernel->setGPUStream(gpu_stream_);
                    if (!binding.kernel->multiply_quantized_small_m_gemv(
                            d_A_int8,
                            d_scales_A_blockwise,
                            binding.output,
                            binding.bias,
                            m,
                            binding.n,
                            k,
                            1.0f,
                            0.0f))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Small-M verifier GEMV failed for projection "
                                  << i << " (" << (binding.name ? binding.name : "unnamed") << ")");
                        return false;
                    }
                }

                if (PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cuda_native_vnni_small_m_fused_projection_calls",
                        1.0,
                        "gemm",
                        "cuda:" + std::to_string(cuda_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())},
                            {"route", "shared_quantized_activation"}});
                }

                return true;
            }

            // Step 2: Validate workspace and get buffer pointers
            // Use this kernel's workspace for quantized activations (shared across all projections)
            validateWorkspace();
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));

            // Ensure the LEADING kernel's weights are converted before checking
            // blockwise eligibility. canUseNativeVNNIBlockwise() reads impl_->
            // d_weights_native_vnni which is only populated after ensureWeightsConverted().
            ensureWeightsConverted();

            bool needs_block_sums = (m > 1) && nativeVNNIUsesAsymmetricCorrection(impl_->native_codebook_id);
            if (m > 1)
            {
                for (const auto &proj : projections)
                {
                    auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                    if (!cuda_kernel)
                        continue;
                    cuda_kernel->ensureWeightsConverted();
                    if (cuda_kernel->impl_ &&
                        nativeVNNIUsesAsymmetricCorrection(cuda_kernel->impl_->native_codebook_id))
                    {
                        needs_block_sums = true;
                    }
                }
            }

            // Use blockwise quantization for prefill and for decode when a native
            // payload GEMV path is available.
            const bool use_blockwise = (k % 32 == 0) && (m > 1 || canUseNativeVNNIBlockwise(impl_.get(), m, k));
            float *d_scales_A_blockwise = nullptr;
            int32_t *d_sums_A_blockwise = nullptr;

            if (!use_blockwise)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] K=" << k
                                                                                << " not divisible by 32; NativeVNNI requires K%32==0");
                return false;
            }

            {
                d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                d_sums_A_blockwise = needs_block_sums
                                         ? static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
                                         : nullptr;
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Blockwise quantizing activations once, m=" << m << " k=" << k);

                // Blockwise quantize activations ONCE (shared across all projections)
                const bool quantized = d_sums_A_blockwise
                                           ? cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
                                                 d_input, d_A_int8, d_scales_A_blockwise, d_sums_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_)
                                           : cudaQuantGemm_quantizeActivationsBlockwise(
                                                 d_input, d_A_int8, d_scales_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_);
                if (!quantized)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Blockwise activation quantization failed");
                    return false;
                }
            }

            // Step 4: Try concurrent multi-stream dispatch for prefill
            // Deterministic parity mode intentionally disables this path. We
            // also avoid it for very small prefill M, which is the regime where
            // the multi-stream fused path still proved unstable in local-PP
            // parity runs. Keep the fast path for larger prompt lengths where
            // concurrent projection dispatch is most useful.
            const bool deterministic_prefill = cudaNativeVNNIPrefill_getDeterministicMode() ||
                                               debugEnv().gemm.deterministic;
            const bool small_m_stage_stream = (gpu_stream_ != nullptr) && (m <= 16);

            // Prefill concurrency: larger prompt M, multi-stream fused projections.
            // The small-M regime (m <= 16) is intentionally excluded here because the
            // multi-stream fused path proved unstable in local-PP parity runs.
            const bool prefill_concurrent_eligible = use_blockwise && m > 16 &&
                                                     projections.size() >= 2 &&
                                                     debugEnv().gemm.cuda_concurrent_prefill &&
                                                     !deterministic_prefill &&
                                                     !small_m_stage_stream;

            // Decode concurrency: m == 1 GEMV projections (e.g. the GDN q/k/v/z and the
            // tiny alpha/beta gates) dispatched on separate streams so the small,
            // latency-bound gate GEMVs overlap the larger qkv read instead of running
            // strictly serially. Gated separately from prefill (LLAMINAR_CUDA_CONCURRENT_DECODE)
            // because this is single-device decode only. The GEMV path ignores the INT32
            // accumulator scratch, but KPAR reductions still need per-stream FP32
            // partial arenas declared by the fused stage that knows projection fan-out.
            const bool decode_concurrent_eligible = use_blockwise && m == 1 &&
                                                    projections.size() >= 2 &&
                                                    debugEnv().gemm.cuda_concurrent_decode &&
                                                    !deterministic_prefill;

            const bool concurrent_eligible = prefill_concurrent_eligible || decode_concurrent_eligible;
            const bool concurrent_decode = decode_concurrent_eligible;

            // CUDA stream/event creation (pool.init) is illegal while a graph capture is
            // active. The pool is initialized during the eager warmup step (step 0) that
            // precedes capture, so by the time the captured decode runs it is already
            // initialized. Guard defensively: if capture is active and the pool has not
            // yet been initialized, fall back to the sequential path rather than issue an
            // illegal allocation inside the capture.
            bool concurrent_safe = concurrent_eligible;
            if (concurrent_eligible && isGraphCaptureActive())
            {
                auto &pool_check = getSharedCUDAPrefillPool(cuda_device_id_);
                if (!pool_check.initialized)
                {
                    LOG_DEBUG("[ConcurrentGemm] Graph capture active but stream pool not "
                              "initialized; using sequential fallback");
                    concurrent_safe = false;
                }
            }

            if (concurrent_safe)
            {
                const int num_proj = static_cast<int>(projections.size());
                if (!concurrent_decode && num_proj > kCudaConcurrentPrefillWorkspaceSlots)
                {
                    throw std::runtime_error(
                        "[ConcurrentPrefill] Quantized prefill supports at most " +
                        std::to_string(kCudaConcurrentPrefillWorkspaceSlots) +
                        " concurrent projections with workspace-backed INT32 accumulators; got " +
                        std::to_string(num_proj));
                }
                auto &pool = getSharedCUDAPrefillPool(cuda_device_id_);
                pool.init(cuda_device_id_, num_proj);

                // Record event after quantization completes on main stream
                cudaQuantGemm_recordEvent(pool.quant_ready, gpu_stream_);

                for (int pi = 0; pi < num_proj; ++pi)
                {
                    const auto &proj = projections[pi];
                    auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                    if (!cuda_kernel || !proj.output)
                    {
                        throw std::runtime_error(
                            "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                            " has null kernel or output — cannot continue inference");
                    }

                    const int n = proj.n;
                    cuda_kernel->ensureWeightsConverted();

                    // Use separate workspace accumulator slots instead of hidden per-stream
                    // cudaMalloc scratch. Slot 0 reuses the normal ACC_INT32 buffer; slots
                    // 1..N come from CUDA_CONCURRENT_PREFILL_ACC_INT32.
                    int stream_idx = pi % pool.count;
                    size_t acc_elements = static_cast<size_t>(m) * static_cast<size_t>(n);
                    // The decode (m == 1) GEMV path ignores the INT32 accumulator entirely
                    // (it reduces directly into FP32 via the per-kernel GEMV context), so we
                    // skip accumulator selection there.
                    int32_t *proj_d_C_int32 = nullptr;
                    if (concurrent_decode)
                    {
                        if (!cuda_kernel->workspace_)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                " has no bound workspace for GEMV context binding");
                        }
                        cuda_kernel->validateWorkspace();
                        cuda_kernel->bindConcurrentNativeDecodeScratch(m, n, k, stream_idx, num_proj);
                    }
                    else
                    {
                        if (!cuda_kernel->workspace_)
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                                " has no bound workspace for concurrent accumulator selection");
                        }

                        if (stream_idx == 0)
                        {
                            proj_d_C_int32 = static_cast<int32_t *>(
                                cuda_kernel->workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
                            const size_t acc_bytes =
                                cuda_kernel->workspace_->getBufferSize(GemmWorkspaceBuffers::ACC_INT32);
                            const size_t needed_bytes = acc_elements * sizeof(int32_t);
                            if (!proj_d_C_int32 || acc_bytes < needed_bytes)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentPrefill] ACC_INT32 workspace is missing or undersized for projection " +
                                    std::to_string(pi) + ": need " + std::to_string(needed_bytes) +
                                    " bytes, have " + std::to_string(acc_bytes));
                            }
                        }
                        else
                        {
                            void *extra_buffer = cuda_kernel->workspace_->getBuffer(
                                GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);
                            const size_t extra_bytes = cuda_kernel->workspace_->getBufferSize(
                                GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);
                            const size_t extra_slot_bytes =
                                extra_bytes / static_cast<size_t>(kCudaConcurrentPrefillExtraAccumulatorSlots);
                            const size_t needed_bytes = acc_elements * sizeof(int32_t);
                            if (!extra_buffer || extra_slot_bytes < needed_bytes ||
                                stream_idx > kCudaConcurrentPrefillExtraAccumulatorSlots)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentPrefill] " +
                                    std::string(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32) +
                                    " workspace is missing or undersized for projection " +
                                    std::to_string(pi) + ": need slot " +
                                    std::to_string(needed_bytes) + " bytes, have " +
                                    std::to_string(extra_slot_bytes) + " bytes");
                            }

                            auto *extra_bytes_ptr = static_cast<unsigned char *>(extra_buffer);
                            proj_d_C_int32 = reinterpret_cast<int32_t *>(
                                extra_bytes_ptr + static_cast<size_t>(stream_idx - 1) * extra_slot_bytes);
                        }

                        cuda_kernel->validateWorkspace();
                        cuda_kernel->bindConcurrentNativePrefillScratch(m, n, k, stream_idx);
                    }

                    auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_output)
                    {
                        throw std::runtime_error(
                            "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                            " output is not FP32Tensor — cannot continue inference");
                    }
                    float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                    if (!d_output)
                    {
                        throw std::runtime_error(
                            "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                            " output has no GPU data — cannot continue inference");
                    }

                    const float *d_bias = nullptr;
                    if (proj.bias)
                    {
                        const TensorBase *bias_tensor = proj.bias;
                        if (auto *slice = dynamic_cast<const TensorSlice *>(proj.bias))
                            bias_tensor = slice->inner();

                        auto *fp32_bias = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(bias_tensor));
                        if (fp32_bias)
                        {
                            auto current_dev = fp32_bias->current_device();
                            if (current_dev.has_value() && current_dev.value() == target_device)
                                d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                            else if (!current_dev.has_value() || !current_dev->is_gpu())
                            {
                                fp32_bias->ensureOnDevice(target_device);
                                d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                            }
                        }
                    }

                    // stream_idx already computed above for scratch allocation

                    // This stream waits for quantization to complete
                    cudaQuantGemm_streamWaitEvent(pool.streams[stream_idx], pool.quant_ready);

                    // If reusing a stream, wait for its previous work
                    if (pi >= pool.count)
                        cudaQuantGemm_streamWaitEvent(pool.streams[stream_idx], pool.completion[stream_idx]);

                    LOG_DEBUG("[ConcurrentPrefill] Projection " << pi
                                                                << " (" << (proj.name ? proj.name : "?")
                                                                << ") M=" << m << " N=" << n << " K=" << k
                                                                << " on stream " << stream_idx);

                    bool proj_ok = runNativeVNNIBlockwiseIfSupported(
                        cuda_kernel->impl_.get(),
                        d_A_int8, proj_d_C_int32, d_output, d_scales_A_blockwise,
                        m, n, k, 1.0f, 0.0f, nullptr, d_bias,
                        cuda_device_id_, pool.streams[stream_idx],
                        nullptr,
                        nativeVNNIUsesAsymmetricCorrection(cuda_kernel->impl_->native_codebook_id)
                            ? d_sums_A_blockwise
                            : nullptr);

                    if (!proj_ok)
                    {
                        throw std::runtime_error(
                            "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                            " (" + std::string(proj.name ? proj.name : "?") +
                            ") kernel launch failed on stream " + std::to_string(stream_idx) +
                            " — cannot continue inference");
                    }

                    cudaQuantGemm_recordEvent(pool.completion[stream_idx], pool.streams[stream_idx]);
                }

                // All projections dispatched — main stream waits for completion
                for (int si = 0; si < std::min(num_proj, pool.count); ++si)
                {
                    cudaQuantGemm_streamWaitEvent(gpu_stream_, pool.completion[si]);
                }
                LOG_DEBUG("[ConcurrentPrefill] All " << num_proj << " projections dispatched concurrently");
                return true;
            }

            // Step 5: Execute each projection using the SHARED quantized activations (sequential fallback)
            bool all_success = true;
            for (size_t i = 0; i < projections.size() && all_success; ++i)
            {
                const auto &proj = projections[i];
                if (!proj.kernel || !proj.output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " has null kernel or output");
                    all_success = false;
                    break;
                }

                // Get the CUDA kernel for this projection
                auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                if (!cuda_kernel)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " kernel is not a CUDAQuantisedGemmKernel");
                    all_success = false;
                    break;
                }

                const int n = proj.n;
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " (" << (proj.name ? proj.name : "unnamed") << "): m=" << m << " n=" << n << " k=" << k);

                // Ensure the projection's weights are converted
                cuda_kernel->ensureWeightsConverted();

                // Validate this projection's workspace is bound and get its d_C_int32 buffer
                cuda_kernel->validateWorkspace();
                int32_t *proj_d_C_int32 = static_cast<int32_t *>(
                    cuda_kernel->workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

                // Ensure output tensor is on device
                if (proj.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " output must be FP32, got " << static_cast<int>(proj.output->native_type()));
                    all_success = false;
                    break;
                }

                auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                if (!fp32_output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast output to FP32Tensor");
                    all_success = false;
                    break;
                }

                // Coherence handled automatically by DeviceGraphExecutor
                float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                if (!d_output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Output has no GPU data for projection " << i);
                    all_success = false;
                    break;
                }

                // Get bias pointer if present (needed for both CUTLASS and blockwise paths)
                const float *d_bias = nullptr;
                if (proj.bias)
                {
                    if (proj.bias->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " bias must be FP32, got " << static_cast<int>(proj.bias->native_type()));
                        all_success = false;
                        break;
                    }

                    // Handle TensorSlice - unwrap to get inner FP32Tensor
                    const TensorBase *bias_tensor = proj.bias;
                    bool was_slice = false;
                    const void *slice_ptr = nullptr;
                    if (auto *slice = dynamic_cast<const TensorSlice *>(proj.bias))
                    {
                        slice_ptr = slice;
                        bias_tensor = slice->inner();
                        was_slice = true;
                    }

                    auto *fp32_bias = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(bias_tensor));
                    if (!fp32_bias)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast bias to FP32Tensor"
                                  << " | was_slice=" << was_slice
                                  << " | bias_tensor=" << bias_tensor
                                  << " | native_type=" << static_cast<int>(bias_tensor->native_type()));
                        all_success = false;
                        break;
                    }

                    // Check if bias is already on the correct CUDA device
                    // In multi-GPU scenarios, each device should have its own bias tensor clone
                    // (created during weight preloading via WeightPreloader::uploadNonGemmWeights)
                    DeviceId target_device = DeviceId::cuda(cuda_device_id_);
                    auto current_dev = fp32_bias->current_device();

                    if (current_dev.has_value() && current_dev.value() == target_device)
                    {
                        // Already on correct device - use directly
                        d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                    }
                    else if (current_dev.has_value() && current_dev->is_gpu())
                    {
                        // Tensor is on a DIFFERENT GPU - this is a multi-GPU race condition!
                        // Do NOT call ensureOnDevice() as it would free the other GPU's memory.
                        // The correct fix is to ensure each device has its own bias tensor clone.
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] MULTI-GPU CONFLICT: Bias tensor is on "
                                  << current_dev->to_string() << " but we need CUDA:" << cuda_device_id_
                                  << ". Ensure WeightPreloader::uploadNonGemmWeights() was called for this device.");
                        all_success = false;
                        break;
                    }
                    else
                    {
                        // Tensor is on CPU or not uploaded yet - safe to upload to this device
                        if (!fp32_bias->ensureOnDevice(target_device))
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to upload bias to CUDA:" << cuda_device_id_);
                            all_success = false;
                            break;
                        }
                        d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                    }

                    if (!d_bias)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Bias has no GPU data for projection " << i
                                                                                                                          << " | was_slice=" << was_slice
                                                                                                                          << " | slice_ptr=" << slice_ptr
                                                                                                                          << " | fp32_bias=" << fp32_bias
                                                                                                                          << " | numel=" << fp32_bias->numel()
                                                                                                                          << " | host_data=" << fp32_bias->data()
                                                                                                                          << " | device_valid=" << fp32_bias->deviceValid()
                                                                                                                          << " | device=" << (fp32_bias->current_device().has_value() ? fp32_bias->current_device()->to_string() : "none"));
                        all_success = false;
                        break;
                    }
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " using bias ptr=" << static_cast<const void *>(d_bias));
                }

                if (use_blockwise)
                {
                    const bool trace_fused = debugEnv().gemm.cuda_fused_gemm_trace;

                    // Diagnostic: checksum weight data to detect weight corruption
                    if (trace_fused && cuda_kernel->impl_)
                    {
                        cudaStreamSynchronize(static_cast<cudaStream_t>(gpu_stream_));
                        const void *wt_ptr = cuda_kernel->impl_->d_weights_native_vnni;
                        if (wt_ptr)
                        {
                            const int wt_bytes = 1024;
                            std::vector<uint8_t> wt_host(wt_bytes);
                            cudaMemcpy(wt_host.data(), wt_ptr,
                                       wt_bytes, cudaMemcpyDeviceToHost);
                            uint64_t wt_hash = 0;
                            for (int wi = 0; wi < wt_bytes; ++wi)
                                wt_hash = wt_hash * 31 + wt_host[wi];
                            LOG_WARN("[GEMM_WEIGHTS] proj=" << i
                                                            << " name=" << (proj.name ? proj.name : "?")
                                                            << " wt_hash=" << wt_hash
                                                            << " d_output=" << static_cast<void *>(d_output));
                        }
                    }

                    if (m == 1 && useCanonicalM1SmallMDecode())
                    {
                        const bool canonical_ok = cuda_kernel->multiply_quantized_m1_via_small_m_gemv(
                            d_A_int8,
                            d_scales_A_blockwise,
                            d_output,
                            d_bias,
                            n,
                            k,
                            1.0f,
                            0.0f);
                        if (!canonical_ok)
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Deterministic canonical M=1 small-M GEMV failed for projection "
                                      << i << " (" << (proj.name ? proj.name : "unnamed") << ")");
                            all_success = false;
                            break;
                        }

                        if (trace_fused)
                        {
                            LOG_WARN("[GEMM_PATH] proj=" << i
                                                         << " name=" << (proj.name ? proj.name : "?")
                                                         << " backend=canonical_m1_small_m"
                                                         << " m=" << m << " n=" << n << " k=" << k
                                                         << " output=" << static_cast<void *>(d_output));
                        }
                        continue;
                    }

                    const bool used_native = runNativeVNNIBlockwiseIfSupported(
                        cuda_kernel->impl_.get(),
                        d_A_int8,
                        proj_d_C_int32,
                        d_output,
                        d_scales_A_blockwise,
                        m, n, k,
                        1.0f, 0.0f,
                        nullptr,
                        d_bias,
                        cuda_device_id_, gpu_stream_,
                        nullptr,
                        nativeVNNIUsesAsymmetricCorrection(cuda_kernel->impl_->native_codebook_id)
                            ? d_sums_A_blockwise
                            : nullptr);

                    if (trace_fused)
                    {
                        LOG_WARN("[GEMM_PATH] proj=" << i
                                                     << " name=" << (proj.name ? proj.name : "?")
                                                     << " backend=" << (used_native ? "native_vnni" : "fallback_blockwise")
                                                     << " m=" << m << " n=" << n << " k=" << k
                                                     << " output=" << static_cast<void *>(d_output)
                                                     << " acc=" << static_cast<void *>(proj_d_C_int32));
                    }

                    if (used_native)
                    {
                        // Diagnostic: checksum output after GEMM to detect corruption source.
                        // NOTE: cudaStreamSynchronize is ILLEGAL during stream capture and
                        // would leave the capture stream in an error state, failing the
                        // next projection's GEMV with a sticky cudaErrorStreamCaptureUnsupported.
                        // Only sync when the trace is explicitly requested — callers using
                        // LLAMINAR_CUDA_FUSED_GEMM_TRACE must not be running under capture.
                        if (trace_fused)
                        {
                            cudaStreamSynchronize(static_cast<cudaStream_t>(gpu_stream_));
                            const size_t total = static_cast<size_t>(m) * n;
                            std::vector<float> host_all(total);
                            cudaMemcpy(host_all.data(), d_output, total * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            double sum = 0;
                            float abs_max = 0;
                            for (size_t ci = 0; ci < total; ++ci)
                            {
                                sum += static_cast<double>(host_all[ci]);
                                float a = std::fabs(host_all[ci]);
                                if (a > abs_max)
                                    abs_max = a;
                            }
                            LOG_WARN("[GEMM_DIAG] proj=" << i
                                                         << " name=" << (proj.name ? proj.name : "?")
                                                         << " m=" << m << " n=" << n << " k=" << k
                                                         << " total=" << total
                                                         << " fullsum=" << std::fixed << std::setprecision(6) << sum
                                                         << " absmax=" << abs_max
                                                         << " first4=[" << host_all[0] << "," << host_all[1]
                                                         << "," << host_all[2] << "," << host_all[3] << "]");
                        }
                        continue;
                    }

                    // NativeVNNI is the only path — no TC/CUTLASS fallback.
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] NativeVNNI GEMM failed for projection " << i
                                                                                                                        << " (no fallback available)");
                    all_success = false;
                    break;
                }

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
                // Debug: Log output values after scaling - EXPENSIVE, guarded by compile flag
                // CRITICAL: cudaMemcpy(D2H) on default stream is illegal during graph capture
                if (d_output && m > 0 && n > 0)
                {
                    size_t copy_count = std::min(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<size_t>(8));
                    std::vector<float> h_output(copy_count);
                    cudaQuantGemm_copyDeviceToHost(h_output.data(), d_output, h_output.size(), cuda_device_id_);
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] " << (proj.name ? proj.name : "unnamed")
                                                                                  << " output[0:4]=" << h_output[0] << "," << (h_output.size() > 1 ? h_output[1] : 0.f) << ","
                                                                                  << (h_output.size() > 2 ? h_output[2] : 0.f) << "," << (h_output.size() > 3 ? h_output[3] : 0.f));
                }
#endif
            }

            return all_success;
        }

        // =====================================================================
        // Internal dispatch methods
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_q8_to_fp32(
            const Q8_1Block * /*d_A_q8*/, float * /*d_C*/,
            int /*m*/, int /*n*/, int /*k*/,
            float /*alpha*/, float /*beta*/)
        {
            // TODO: Implement Q8_1 direct path
            // For now, this would need to extract int8 data from Q8_1 blocks
            LOG_ERROR("[CUDAQuantisedGemmKernel] Q8_1→FP32 path not yet implemented");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_q8_to_q8(
            const Q8_1Block * /*d_A_q8*/, Q8_1Block * /*d_C_q8*/,
            int /*m*/, int /*n*/, int /*k*/)
        {
            // TODO: Implement Q8_1→Q8_1 fused requant path
            LOG_ERROR("[CUDAQuantisedGemmKernel] Q8_1→Q8_1 path not yet implemented");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_with_fused_swiglu(
            const float *d_gate, const float *d_up,
            float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_with_fused_swiglu] m=" << m << " n=" << n << " k=" << k);

            validateWorkspace();

            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));

            ensureWeightsConverted();

            const bool verifier_small_m =
                (m > 1 && m <= 4) &&
                (k % 32 == 0) &&
                canUseNativeVNNIBlockwise(impl_.get(), 1, k);
            const bool use_blockwise =
                (k % 32 == 0) &&
                (verifier_small_m || m > 1 || canUseNativeVNNIBlockwise(impl_.get(), m, k));

            if (use_blockwise)
            {
                float *d_scales_A_blockwise = static_cast<float *>(
                    workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                int32_t *d_sums_A_blockwise =
                    (!verifier_small_m && m > 1 && nativeVNNIUsesAsymmetricCorrection(impl_->native_codebook_id))
                        ? static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
                        : nullptr;

                // Fused SwiGLU + blockwise quantization: replaces separate SwiGLU + quant kernels
                const bool quantized = d_sums_A_blockwise
                                           ? cudaOps_fused_swiglu_quantize_blockwise_with_sums(
                                                 d_gate, d_up, d_A_int8, d_scales_A_blockwise, d_sums_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_)
                                           : cudaOps_fused_swiglu_quantize_blockwise(
                                                 d_gate, d_up, d_A_int8, d_scales_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_);
                if (!quantized)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Fused SwiGLU+quantize failed");
                    return false;
                }

                if (verifier_small_m)
                {
                    if (!multiply_quantized_small_m_gemv(
                            d_A_int8,
                            d_scales_A_blockwise,
                            d_C,
                            nullptr,
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            true))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel] Fused SwiGLU small-M NativeVNNI GEMV failed");
                        return false;
                    }

                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_with_fused_swiglu] Complete (small-M native GEMV)");
                    return true;
                }

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;

                // Try native VNNI blockwise path (GEMV for decode)
                if (runNativeVNNIBlockwiseIfSupported(
                        impl_.get(),
                        d_A_int8, nullptr, d_C, d_scales_A_blockwise,
                        m, n, k, alpha, beta, d_C_existing, nullptr,
                        cuda_device_id_, gpu_stream_,
                        packed_ ? &packed_->rowmajor_ : nullptr,
                        d_sums_A_blockwise))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_with_fused_swiglu] Complete (native GEMV)");

                    // Diagnostic: checksum FFN_DOWN output (NativeVNNI path)
                    if (debugEnv().gemm.cuda_fused_gemm_trace)
                    {
                        cudaStreamSynchronize(static_cast<cudaStream_t>(gpu_stream_));
                        const size_t total = static_cast<size_t>(m) * n;
                        std::vector<float> host_all(total);
                        cudaMemcpy(host_all.data(), d_C, total * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        double sum = 0;
                        float abs_max = 0;
                        for (size_t ci = 0; ci < total; ++ci)
                        {
                            sum += static_cast<double>(host_all[ci]);
                            float a = std::fabs(host_all[ci]);
                            if (a > abs_max)
                                abs_max = a;
                        }
                        LOG_WARN("[GEMM_SWIGLU] m=" << m << " n=" << n << " k=" << k
                                                    << " total=" << total
                                                    << " fullsum=" << std::fixed << std::setprecision(6) << sum
                                                    << " absmax=" << abs_max
                                                    << " first4=[" << host_all[0] << "," << host_all[1]
                                                    << "," << host_all[2] << "," << host_all[3] << "]");
                    }
                    return true;
                }

                // NativeVNNI is the only path — no TC/CUTLASS fallback.
                LOG_ERROR("[CUDAQuantisedGemmKernel] Fused SwiGLU NativeVNNI GEMM failed (no fallback available)");
                return false;
            }

            // Fallback: row-wise path (K not divisible by 32).
            // Compute SwiGLU into a temp buffer, then use standard quantize + GEMM.
            // This is rare (Qwen K=18944 is divisible by 32).
            LOG_WARN("[CUDAQuantisedGemmKernel::multiply_with_fused_swiglu] "
                     "Falling back to non-fused path (K="
                     << k << " not divisible by 32)");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor_with_fused_swiglu(
            const TensorBase *gate, const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace)
        {
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                const bool result = multiply_tensor_with_fused_swiglu(
                    gate, up, output, m, n, k, alpha, beta, nullptr);
                workspace_ = saved_ws;
                return result;
            }

            // Get device pointers (tensors must already be on GPU via DeviceGraphExecutor coherence)
            const float *d_gate = static_cast<const float *>(gate->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up->gpu_data_ptr());
            float *d_C = static_cast<float *>(output->gpu_data_ptr());

            if (!d_gate || !d_up || !d_C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "Null GPU pointer: gate="
                          << (void *)d_gate
                          << " up=" << (void *)d_up << " C=" << (void *)d_C);
                return false;
            }

            return multiply_with_fused_swiglu(d_gate, d_up, d_C, m, n, k, alpha, beta);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate, const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace)
        {
            if (m <= 1 || m > 4)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] grouped verifier SwiGLU requires M=2..4, got M="
                          << m);
                return false;
            }
            auto decode_equivalent_dispatch = beginVerifierDecodeEquivalentScope();
            return multiply_tensor_with_fused_swiglu(
                gate, up, output, m, n, k, alpha, beta, workspace);
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_small_m_gemv(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            if (!d_A || !d_C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_small_m_gemv] Null input or output");
                return false;
            }
            if (m <= 1 || m > 4 || n <= 0 || k <= 0)
            {
                return false;
            }
            if ((k % 32) != 0)
            {
                return false;
            }

            validateWorkspace();
            ensureWeightsConverted();

            if (!canUseNativeVNNIBlockwise(impl_.get(), 1, k))
            {
                return false;
            }

            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            float *d_scales_A_blockwise =
                static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));

            if (!cudaQuantGemm_quantizeActivationsBlockwise(
                    d_A,
                    d_A_int8,
                    d_scales_A_blockwise,
                    m,
                    k,
                    cuda_device_id_,
                    gpu_stream_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_small_m_gemv] Blockwise activation quantization failed");
                return false;
            }

            return multiply_quantized_small_m_gemv(
                d_A_int8,
                d_scales_A_blockwise,
                d_C,
                d_bias,
                m,
                n,
                k,
                alpha,
                beta);
        }

        bool CUDAQuantisedGemmKernel::multiply_quantized_small_m_gemv(
            const int8_t *d_A_int8,
            const float *d_scales_A_blockwise,
            float *d_C,
            const float *d_bias,
            int m, int n, int k,
            float alpha, float beta,
            bool use_specialized_small_m_kernel)
        {
            if (!d_A_int8 || !d_scales_A_blockwise || !d_C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_small_m_gemv] Null input, scale, or output");
                return false;
            }
            if (m <= 1 || m > 4 || n <= 0 || k <= 0 || (k % 32) != 0)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_small_m_gemv] Invalid dimensions: M="
                          << m << " N=" << n << " K=" << k);
                return false;
            }

            ensureWeightsConverted();
            if (!canUseNativeVNNIBlockwise(impl_.get(), 1, k))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_small_m_gemv] NativeVNNI GEMV unsupported for codebook "
                          << static_cast<int>(impl_->native_codebook_id)
                          << " N=" << n << " K=" << k);
                return false;
            }
            if (!ensureNativeVNNIIQGridTablesInitialized(
                    impl_->native_codebook_id,
                    cuda_device_id_,
                    "NativeVNNI small-M route"))
            {
                return false;
            }

            const int blocks_per_row = k / 32;
            auto record_small_m_route = [&](const char *route)
            {
                if (!PerfStatsCollector::isEnabled())
                    return;

                PerfStatsCollector::addCounter(
                    "kernel",
                    "cuda_native_vnni_small_m_calls",
                    1.0,
                    "gemm",
                    "cuda:" + std::to_string(cuda_device_id_),
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"codebook", std::to_string(static_cast<int>(impl_->native_codebook_id))},
                        {"n", std::to_string(n)},
                        {"k", std::to_string(k)},
                        {"route", route}});

                if (m == 2)
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cuda_native_vnni_m2_calls",
                        1.0,
                        "gemm",
                        "cuda:" + std::to_string(cuda_device_id_),
                        PerfStatsCollector::Tags{
                            {"codebook", std::to_string(static_cast<int>(impl_->native_codebook_id))},
                            {"n", std::to_string(n)},
                            {"k", std::to_string(k)},
                            {"route", route}});
                }
            };

            if (use_specialized_small_m_kernel)
            {
                if (!impl_->gemv_ctx)
                    impl_->gemv_ctx = cudaGemvContext_create(cuda_device_id_);

                const bool ok = cudaNativeVNNIGemvTuned_small_m_fp32(
                    d_A_int8,
                    impl_->d_weights_native_vnni,
                    impl_->d_weights_native_scales,
                    impl_->d_weights_native_mins,
                    impl_->d_weights_native_emins,
                    d_C,
                    d_scales_A_blockwise,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta != 0.0f ? d_C : nullptr,
                    d_bias,
                    impl_->native_codebook_id,
                    cuda_device_id_,
                    gpu_stream_,
                    impl_->gemv_ctx,
                    packed_ ? &packed_->rowmajor_ : nullptr);
                if (!ok)
                {
                    if (cudaNativeVNNIGemvSweep_isActive())
                    {
                        return false;
                    }

                    cudaError_t le = cudaPeekAtLastError();
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_small_m_gemv] NativeVNNI small-M GEMV failed"
                              << " M=" << m
                              << " codebook=" << static_cast<int>(impl_->native_codebook_id)
                              << " N=" << n << " K=" << k
                              << " stream=" << gpu_stream_
                              << " cuda_error=" << cudaGetErrorName(le)
                              << " (" << cudaGetErrorString(le) << ")"
                              << " rowmajor_slot=" << (packed_ && packed_->rowmajor_ ? "present" : "absent"));
                    return false;
                }

                record_small_m_route("specialized");
                return true;
            }

            for (int row = 0; row < m; ++row)
            {
                const int8_t *row_A = d_A_int8 + static_cast<size_t>(row) * static_cast<size_t>(k);
                const float *row_scales =
                    d_scales_A_blockwise + static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
                float *row_C = d_C + static_cast<size_t>(row) * static_cast<size_t>(n);
                const float *row_existing = (beta != 0.0f) ? row_C : nullptr;

                if (!runNativeVNNIBlockwiseIfSupported(
                        impl_.get(),
                        row_A,
                        nullptr,
                        row_C,
                        row_scales,
                        1,
                        n,
                        k,
                        alpha,
                        beta,
                        row_existing,
                        d_bias,
                        cuda_device_id_,
                        gpu_stream_,
                        packed_ ? &packed_->rowmajor_ : nullptr))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_small_m_gemv] Row "
                              << row << " native-VNNI GEMV failed");
                    return false;
                }
            }

            record_small_m_route("rowwise");

            return true;
        }

        bool CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv(
            const int8_t *d_A_int8,
            const float *d_scales_A_blockwise,
            float *d_C,
            const float *d_bias,
            int n, int k,
            float alpha, float beta)
        {
            if (!useCanonicalM1SmallMDecode())
                return false;
            if (!d_A_int8 || !d_scales_A_blockwise || !d_C || n <= 0 || k <= 0 || (k % 32) != 0)
                return false;
            if (beta != 0.0f)
                return false;
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv] "
                          "Deterministic canonical M=1 GEMV requires an explicit CUDA stream");
                return false;
            }

            validateWorkspace();
            ensureWeightsConverted();
            if (!canUseNativeVNNIBlockwise(impl_.get(), 1, k))
                return false;

            auto *scratch = static_cast<float *>(workspace_->getBuffer(tempCFp32BufferName()));
            const size_t scratch_bytes = workspace_->getBufferSize(tempCFp32BufferName());
            const size_t required_scratch_bytes = static_cast<size_t>(2) * static_cast<size_t>(n) * sizeof(float);
            if (!scratch || scratch_bytes < required_scratch_bytes)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv] "
                          << tempCFp32BufferName() << " workspace is missing or undersized: need "
                          << required_scratch_bytes << " bytes, have " << scratch_bytes);
                return false;
            }

            const int blocks_per_row = k / 32;
            auto *mutable_A = const_cast<int8_t *>(d_A_int8);
            auto *mutable_scales = const_cast<float *>(d_scales_A_blockwise);
            cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream_);
            cudaError_t err = cudaMemsetAsync(
                mutable_A + static_cast<size_t>(k),
                0,
                static_cast<size_t>(k) * sizeof(int8_t),
                stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv] Failed to clear padded A row: "
                          << cudaGetErrorString(err));
                return false;
            }

            err = cudaMemsetAsync(
                mutable_scales + static_cast<size_t>(blocks_per_row),
                0,
                static_cast<size_t>(blocks_per_row) * sizeof(float),
                stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv] Failed to clear padded scale row: "
                          << cudaGetErrorString(err));
                return false;
            }

            if (!multiply_quantized_small_m_gemv(
                    d_A_int8,
                    d_scales_A_blockwise,
                    scratch,
                    d_bias,
                    2,
                    n,
                    k,
                    alpha,
                    0.0f,
                    true))
            {
                return false;
            }

            err = cudaMemcpyAsync(
                d_C,
                scratch,
                static_cast<size_t>(n) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_quantized_m1_via_small_m_gemv] Failed to copy canonical row output: "
                          << cudaGetErrorString(err));
                return false;
            }

            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cuda_native_vnni_m1_canonical_small_m_calls",
                    1.0,
                    "gemm",
                    "cuda:" + std::to_string(cuda_device_id_),
                    PerfStatsCollector::Tags{
                        {"codebook", std::to_string(static_cast<int>(impl_->native_codebook_id))},
                        {"n", std::to_string(n)},
                        {"k", std::to_string(k)}});
            }

            return true;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] m=" << m << " n=" << n << " k=" << k
                                                                            << " alpha=" << alpha << " beta=" << beta
                                                                            << " d_A=" << static_cast<const void *>(d_A)
                                                                            << " d_C=" << static_cast<void *>(d_C));

            // Validate workspace is bound (REQUIRED - kernels don't do internal allocation)
            validateWorkspace();

            // Get workspace buffer pointers
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            int32_t *d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

            // Ensure weights converted
            ensureWeightsConverted();

            // ──── cuBLAS FP16 GEMM path (LLAMINAR_CUBLAS_GEMM=1) ────────────
            // Dequant Q4_0 native VNNI weights → FP16 per-call,
            // convert FP32 activations → FP16,
            // then cuBLAS FP16 tensor-core GEMM with FP32 accumulation.
            static const bool use_cublas_gemm = []()
            {
                return debugEnv().gemm.cuda_cublas_gemm;
            }();

            if (use_cublas_gemm && m > 1 && impl_->d_weights_native_vnni && impl_->d_weights_native_scales)
            {
                static std::once_flag cublas_gemm_once;
                std::call_once(cublas_gemm_once, []()
                               { LOG_DEBUG("[CUDAQuantisedGemmKernel] cuBLAS FP16 GEMM path active (Q4_0 native dequant)"); });

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM, gpu_stream_);
                // Lazy-create per-device cuBLAS context
                if (!impl_->cublas_ctx)
                    impl_->cublas_ctx = cudaCuBLASContext_create(cuda_device_id_);

                bool ok = cudaCuBLAS_fp16_gemm_q40(
                    impl_->d_weights_native_vnni,
                    impl_->d_weights_native_scales,
                    d_A, d_C,
                    m, n, k,
                    alpha, beta,
                    d_C_existing,
                    cuda_device_id_, gpu_stream_,
                    impl_->cublas_ctx);
                if (ok)
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (cuBLAS FP16 Q4_0)");
                    return true;
                }
                LOG_WARN("[CUDAQuantisedGemmKernel] cuBLAS FP16 GEMM failed, falling back");
            }

            if (m > 1 && m <= 4)
            {
                if (multiply_fp32_to_fp32_small_m_gemv(
                        d_A, d_C, nullptr, m, n, k, alpha, beta))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (small-M row-wise native payload GEMV)");
                    return true;
                }

                if (cudaNativeVNNIGemvSweep_isActive())
                {
                    // Training sweeps must fail closed for M=2..4. Falling
                    // through to the generic M>1 GEMM path would time a
                    // different kernel family while labelling the CSV row as
                    // the requested small-M candidate.
                    return false;
                }
            }

            // Use blockwise quantization for prefill and for decode when a native
            // payload GEMV path is available.
            const bool use_blockwise =
                (k % 32 == 0) &&
                (m > 1 || canUseNativeVNNIBlockwise(impl_.get(), m, k));

            if (use_blockwise)
            {
                float *d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                int32_t *d_sums_A_blockwise =
                    (m > 1 && nativeVNNIUsesAsymmetricCorrection(impl_->native_codebook_id))
                        ? static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
                        : nullptr;

                // Step 1: Blockwise quantize activations
                const bool quantized = d_sums_A_blockwise
                                           ? cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
                                                 d_A, d_A_int8, d_scales_A_blockwise, d_sums_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_)
                                           : cudaQuantGemm_quantizeActivationsBlockwise(
                                                 d_A, d_A_int8, d_scales_A_blockwise,
                                                 m, k, cuda_device_id_, gpu_stream_);
                if (!quantized)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise activation quantization failed");
                    return false;
                }

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
                if (m == 1 && useCanonicalM1SmallMDecode())
                {
                    if (multiply_quantized_m1_via_small_m_gemv(
                            d_A_int8,
                            d_scales_A_blockwise,
                            d_C,
                            nullptr,
                            n,
                            k,
                            alpha,
                            beta))
                    {
                        LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (deterministic canonical M=1 small-M GEMV)");
                        return true;
                    }

                    LOG_ERROR("[CUDAQuantisedGemmKernel] Deterministic M=1 native-VNNI canonical small-M GEMV failed");
                    return false;
                }

                if (runNativeVNNIBlockwiseIfSupported(
                        impl_.get(),
                        d_A_int8,
                        d_C_int32,
                        d_C,
                        d_scales_A_blockwise,
                        m, n, k,
                        alpha, beta,
                        d_C_existing,
                        nullptr,
                        cuda_device_id_, gpu_stream_,
                        packed_ ? &packed_->rowmajor_ : nullptr,
                        d_sums_A_blockwise))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (native payload GEMV)");
                    return true;
                }

                // NativeVNNI-only mode: no INT8 expanded fallback.
                LOG_ERROR("[CUDAQuantisedGemmKernel] NativeVNNI GEMM failed (no fallback available)");
                return false;
            }

            // K not divisible by 32 — no blockwise path available.
            LOG_ERROR("[CUDAQuantisedGemmKernel] K=" << k << " not divisible by 32; "
                                                             "NativeVNNI requires K%32==0 and row-wise CUTLASS has been removed");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Validate workspace is bound (REQUIRED - kernels don't do internal allocation)
            validateWorkspace();

            // Get workspace buffer pointers
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            int32_t *d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

            // Ensure weights converted
            ensureWeightsConverted();

            if (m > 1 && m <= 4 &&
                multiply_fp32_to_fp32_small_m_gemv(
                    d_A, d_C, d_bias, m, n, k, alpha, beta))
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Complete (small-M row-wise native payload GEMV)");
                return true;
            }

            // Use blockwise quantization whenever K is block-aligned.
            const bool use_blockwise =
                (k % 32 == 0);

            if (use_blockwise)
            {
                float *d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                int32_t *d_sums_A_blockwise =
                    (m > 1 && nativeVNNIUsesAsymmetricCorrection(impl_->native_codebook_id))
                        ? static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
                        : nullptr;

                // Step 1: Blockwise quantize activations
                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::QUANTIZE_ACTIVATIONS, gpu_stream_);
                    const bool quantized = d_sums_A_blockwise
                                               ? cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
                                                     d_A, d_A_int8, d_scales_A_blockwise, d_sums_A_blockwise,
                                                     m, k, cuda_device_id_, gpu_stream_)
                                               : cudaQuantGemm_quantizeActivationsBlockwise(
                                                     d_A, d_A_int8, d_scales_A_blockwise,
                                                     m, k, cuda_device_id_, gpu_stream_);
                    if (!quantized)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise activation quantization failed");
                        return false;
                    }
                }

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
                if (m == 1 && useCanonicalM1SmallMDecode())
                {
                    if (multiply_quantized_m1_via_small_m_gemv(
                            d_A_int8,
                            d_scales_A_blockwise,
                            d_C,
                            d_bias,
                            n,
                            k,
                            alpha,
                            beta))
                    {
                        LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Complete (deterministic canonical M=1 small-M GEMV)");
                        return true;
                    }

                    LOG_ERROR("[CUDAQuantisedGemmKernel] Deterministic M=1 native-VNNI canonical small-M GEMV with bias failed");
                    return false;
                }

                if (runNativeVNNIBlockwiseIfSupported(
                        impl_.get(),
                        d_A_int8,
                        d_C_int32,
                        d_C,
                        d_scales_A_blockwise,
                        m, n, k,
                        alpha, beta,
                        d_C_existing,
                        d_bias,
                        cuda_device_id_, gpu_stream_,
                        packed_ ? &packed_->rowmajor_ : nullptr,
                        d_sums_A_blockwise))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Complete (native payload GEMV)");
                    return true;
                }

                // NativeVNNI is the only path — no TC/CUTLASS fallback.
                LOG_ERROR("[CUDAQuantisedGemmKernel] NativeVNNI GEMM with bias failed (no fallback available)");
                return false;
            }

            // K not divisible by 32 — no blockwise path available.
            LOG_ERROR("[CUDAQuantisedGemmKernel] K=" << k << " not divisible by 32; "
                                                             "NativeVNNI requires K%32==0 and row-wise CUTLASS has been removed");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_q8(
            const float * /*d_A*/, Q8_1Block * /*d_C_q8*/,
            int /*m*/, int /*n*/, int /*k*/)
        {
            // TODO: Implement FP32→Q8_1 fused requant path
            LOG_ERROR("[CUDAQuantisedGemmKernel] FP32→Q8_1 path not yet implemented");
            return false;
        }

        // =====================================================================
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAQuantisedGemmKernel] multiply_activations not supported - use dedicated attention kernel");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_activations_strided(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            int /*lda*/, int /*ldb*/, int /*ldc*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAQuantisedGemmKernel] multiply_activations_strided not supported - use dedicated attention kernel");
            return false;
        }

        // =====================================================================
        // ITensorKernel interface
        // =====================================================================

        bool CUDAQuantisedGemmKernel::supports_device(int device_idx) const
        {
            if (device_idx < 0)
            {
                return false; // CPU not supported
            }

            const auto &dm = DeviceManager::instance();
            if (static_cast<size_t>(device_idx) >= dm.devices().size())
            {
                return false;
            }

            const auto &dev = dm.devices()[device_idx];
            return (dev.type == ComputeBackendType::GPU_CUDA && dev.device_id == cuda_device_id_);
        }

        // =====================================================================
        // IKernelSnapshotCapable interface
        // =====================================================================

        KernelSnapshotInfo CUDAQuantisedGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "quantized weight matrix [n, k] (converted to INT8)", KernelBufferDtype::INT8)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("cuda_device_id", "CUDA device ID", KernelBufferDtype::INT32)
                .withScalar("weights_converted", "whether weights are converted to INT8", KernelBufferDtype::INT32);
        }

        // =====================================================================
        // IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAQuantisedGemmKernel::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            // Use internal dimensions if not specified
            if (n == 0)
                n = static_cast<int>(N_);
            if (k == 0)
                k = static_cast<int>(K_);

            // Native prefill kernels execute row tiles up to 128 rows. Edge
            // tiles guard logical output writes, but their scratch paths share
            // the same buffers as full tiles. Size by tile-padded M so a short
            // prompt cannot leave the CUDA workspace under-provisioned while
            // keeping active/bucket sizing for the rest of the graph.
            const bool deterministic_m1 = (m == 1 && useCanonicalM1SmallMDecode());
            const int workspace_m = deterministic_m1 ? 2 : ((m > 1) ? ((m + 127) & ~127) : m);

            // INT8 path needs quantization + accumulator buffers
            size_t quant_a_bytes = static_cast<size_t>(workspace_m) * k * sizeof(int8_t);
            size_t scales_a_bytes = static_cast<size_t>(workspace_m) * sizeof(float);
            // NativeVNNI doesn't use INT32 accumulator split-K, so 1 chunk is sufficient.
            constexpr size_t partial_chunk_blocks = 1;
            size_t acc_int32_bytes = static_cast<size_t>(workspace_m) * n * partial_chunk_blocks * sizeof(int32_t);
            size_t concurrent_prefill_acc_bytes =
                static_cast<size_t>(kCudaConcurrentPrefillExtraAccumulatorSlots) * acc_int32_bytes;

            reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, quant_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, scales_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32, acc_int32_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32,
                                    concurrent_prefill_acc_bytes, 256, true});

            // Blockwise activation quantization scales: one float per 32-element block
            size_t num_blocks_per_row = static_cast<size_t>((k + 31) / 32);
            size_t scales_a_blockwise_bytes = static_cast<size_t>(workspace_m) * num_blocks_per_row * sizeof(float);
            size_t sums_a_blockwise_bytes = static_cast<size_t>(workspace_m) * num_blocks_per_row * sizeof(int32_t);
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A_BLOCKWISE, scales_a_blockwise_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SUMS_A_BLOCKWISE, sums_a_blockwise_bytes, 256, true});

            // FP32 output workspace for mapped memory redirect
            // When output is host-mapped (e.g., logits), scattered GPU writes go over PCIe.
            // This buffer provides an HBM target; we bulk-DMA to mapped memory after.
            size_t temp_c_fp32_bytes = static_cast<size_t>(workspace_m) * n * sizeof(float);
            reqs.buffers.push_back({tempCFp32BufferName(), temp_c_fp32_bytes, 256, true});

            bool has_native_codebook = false;
            uint8_t native_codebook_id = 0;
            if (packed_)
            {
                native_codebook_id = packed_->native_codebook_id;
                has_native_codebook = true;
            }
            else if (weights_converted_ && impl_ && impl_->d_weights_native_vnni)
            {
                native_codebook_id = impl_->native_codebook_id;
                has_native_codebook = true;
            }
            else if (weights_)
            {
                if (const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights_))
                {
                    if (const auto *info = unpackable->vnniFormatInfo())
                    {
                        native_codebook_id = info->codebook_id;
                        has_native_codebook = true;
                    }
                }
            }

            if (has_native_codebook && workspace_m > 1)
            {
                size_t splitk_partials_bytes = 0;
                size_t streamk_fixup_bytes = 0;
                int planned_split_k = 1;
                int planned_streamk = 0;
                if (cudaNativeVNNIPrefill_getWorkspacePlan(
                        native_codebook_id,
                        m,
                        n,
                        k,
                        cuda_device_id_,
                        &splitk_partials_bytes,
                        &streamk_fixup_bytes,
                        &planned_split_k,
                        &planned_streamk))
                {
                    if (planned_split_k > 1)
                    {
                        splitk_partials_bytes = std::max(
                            splitk_partials_bytes,
                            paddedSplitKPartialBytes(m, n, planned_split_k));
                    }

                    const size_t prefill_scratch_slots = concurrentPrefillScratchSlotsForM(m);

                    if (splitk_partials_bytes > 0)
                    {
                        reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS,
                                                splitk_partials_bytes * prefill_scratch_slots, 256, true});
                    }
                    if (streamk_fixup_bytes > 0)
                    {
                        reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_STREAMK_FIXUP,
                                                streamk_fixup_bytes * prefill_scratch_slots, 256, true});
                    }
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::getWorkspaceRequirements] NativeVNNI prefill plan: codebook="
                              << static_cast<int>(native_codebook_id)
                              << " split_k=" << planned_split_k
                              << " streamk=" << planned_streamk
                              << " splitk_partials=" << (splitk_partials_bytes / 1024) << "KB"
                              << " streamk_fixup=" << (streamk_fixup_bytes / 1024) << "KB");
                }
            }

            // GEMV KPAR partials for decode/verifier reductions. The caller's
            // declared graph shape owns the bound; dynamic-depth graph users
            // must request their max verifier M explicitly.
            const int gemv_workspace_m = (m == 1 && useCanonicalM1SmallMDecode()) ? 2 : m;
            if (gemv_workspace_m > 0 && gemv_workspace_m <= 4)
            {
                const int k_groups = (k + 31) / 32;
                const size_t kpar_bytes =
                    static_cast<size_t>(k_groups) * static_cast<size_t>(gemv_workspace_m) * n * sizeof(float);
                reqs.buffers.push_back({GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS, kpar_bytes, 256, true});
            }

            LOG_DEBUG("[CUDAQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "scales_a_blockwise=" << (scales_a_blockwise_bytes) << "B, "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB"
                      << ", concurrent_acc=" << (concurrent_prefill_acc_bytes / 1024) << "KB"
                      << " (chunk_blocks=" << partial_chunk_blocks << ")"
                      << ", temp_c_fp32=" << (temp_c_fp32_bytes / 1024) << "KB");

            return reqs;
        }

        void CUDAQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            if (workspace_ && workspace_ != workspace)
            {
                resetDynamicState();
            }
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel] Bound workspace manager at " << (void *)workspace
                                                                                  << ", entering managed mode");
            }
            else
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel] Unbound workspace, returning to legacy mode");
            }
        }

        bool CUDAQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *CUDAQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }

    } // namespace cuda
} // namespace llaminar2
