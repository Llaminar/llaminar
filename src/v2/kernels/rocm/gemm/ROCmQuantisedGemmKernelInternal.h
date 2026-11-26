/**
 * @file ROCmQuantisedGemmKernelInternal.h
 * @brief Shared HIP extern "C" declarations, Impl struct, and internal helpers for ROCmQuantisedGemmKernel shards.
 */
#pragma once
/**
 * @file ROCmQuantisedGemmKernel.cpp
 * @note Const prepared-descriptor inspection observes only existing resident
 * metadata. Active export retains the original preparation contract and reuses
 * that projection; declaration must never materialize GPU weights implicitly.
 * @brief Device-resident prepared quantized GEMM and captured launch admission.
 *
 * Prepared weight handles and arena-owned activation/scratch surfaces survive
 * graph recording and replay. Explicit producer policies select lossless
 * NativeQuantised low-bit or blockwise INT8 kernels; execution quantizes resident
 * activations on the exact caller stream and publishes resident FP32 output.
 * The immutable dense-prefill overlay owns only its measured physical keys.
 * Unmeasured geometry remains owned by the complete generic launch policy,
 * while explicit diagnostic controls take precedence during construction.
 * Workspace sizing and launch identity share the producer's typed contract;
 * captured execution neither allocates scratch nor stages activations through
 * host storage. TransferEngine owns setup weight publication and event edges.
 * Raw resident IQ descriptors publish their runtime-generation-owned lookup
 * tables during construction, before a retained graph can consume them.
 *
 * Diagnostic native-Q8 serial geometry scopes admit only a real prepared M1
 * native19 producer; an unrelated format, grouped operation or epilogue cannot
 * silently consume an experiment's requested geometry. Production without a
 * diagnostic scope retains its existing producer and arithmetic selection.
 *
 * @see ROCmQuantisedGemmWorkspaceContract.h
 * @see ROCmDenseProductionPrefillOverlay.h
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 *
 * Projection work admission is copied independently of arithmetic dispatch.
 * Borrowed original request intervals guard native operand/partial/output
 * accesses; physical matrix strides and each row's reduction order stay fixed.
 * Context-owned projection resources are borrowed for a complete eager fork/join
 * or the enclosing whole HIP capture; no global ordinal resource pool survives.
 */

#include "kernels/common/HCDownOutputPublication.h"
#include "kernels/common/ProjectionWorkLaunch.h"
#include "ROCmQuantisedActivationProducers.h"
#include "kernels/rocm/gdn/ROCmGatedNormSingleQ8.h"
#include "ROCmQuantisedEpilogues.h"
#include "ROCmQuantisedGemmKernel.h"
#include "ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/common/VerifierProjectionBundle.h"
#include "kernels/common/VerifierGDNBundle.h"
#include "ROCmNativeQuantisedGDNVerifier.h"
#include "ROCmFloatingPointGemmKernel.h"
#include "kernels/common/HCDownProjectionInput.h"
#include "kernels/common/HCBranchNormSingleQ8Publication.h"
#include "kernels/common/GPUQ8ResidualNative.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/rocm/AMDDeviceContext.h"
#include "ROCmDenseProductionPrefillOverlay.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"
#include "ROCmQuantisedGemmWorkspaceContract.h"
#include "ROCmNativeQuantisedSerialGeometry.h"
#include "ROCmNativeM1Reduction.h"
#include "transfer/TransferEngine.h"
#include "../ROCmKernelBase.h"
#include "ROCmWeightPacker.h"     // packWeightsToROCm, packNativeQuantised
#include "backends/ComputeBackend.h" // DeviceManager
#include "backends/DeviceId.h"       // DeviceId
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/TensorSlice.h"     // TensorSlice - unwrap TP-sharded biases
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h" // iq3s_grid, iq2xs_grid, ksigns_iq2xs etc.
#include "tensors/TensorKernels.h"
#include "tensors/KernelSnapshotInfo.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/SlabGemmConfig.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"
#include "utils/DebugEnv.h"
#include "utils/Assertions.h"
#include "utils/PerfStatsCollector.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "utils/WeightLoadingProfiler.h"

#include <stdexcept>
#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <string>
#include <array>
#include <mutex>
#include <set>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
#include <cstring>
#include <chrono>


namespace llaminar2
{
    namespace rocm
    {
        namespace rqg_detail
        {
            extern std::atomic<uint32_t> g_rocm_gemm_workspace_slice_counter;
            extern thread_local bool g_rocm_native_quantised_decode_equivalent_scope;
        }
        using rqg_detail::g_rocm_gemm_workspace_slice_counter;
        using rqg_detail::g_rocm_native_quantised_decode_equivalent_scope;
        namespace
        {
            constexpr int ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS =
                static_cast<int>(
                    quantized_gemm_workspace::kFusedProjectionStreamCapacity);
            constexpr int ROCM_NATIVE_SMALL_M_GRAPH_SAFE_KB_CAP =
                quantized_gemm_workspace::kScatterKBlockCapacity;

            /**
             * @brief Apply the original epilogue using the enclosing row authority.
             * @param output Exact persistent projection destination [m,n].
             * @param bias Optional original FP32 bias/view native address [n].
             * @param m Captured physical row count.
             * @param n Captured output width.
             * @param alpha Original output multiplier.
             * @param device Exact owning ROCm ordinal.
             * @param stream Exact projection stream, already ordered after its producer.
             * @return Result of the selected count-owned or count-free native specialization.
             */
            bool applyFp32EpilogueScoped(float* output,const float* bias,int m,int n,
                float alpha,int device,void* stream)
            {
                const auto* rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
                if (rows && rows->countOwner())
                    return rocmQuantGemm_applyFp32Epilogue_rows(output,bias,m,n,alpha,device,stream,rows);
                return rocmQuantGemm_applyFp32Epilogue(output,bias,m,n,alpha,device,stream);
            }



        /** @brief Submit quantization with the enclosing immutable prefix authority.
         * Ordinary work retains its original ABI and count-free native kernel.
         * Counted work owns every input read and every scratch write by live row.
         */
        bool rocmQuantGemm_quantizeActivationsBlockwiseScoped(const float *input, int8_t *quant, float *scales, int m, int k, int device, void *stream, int block_size = 32)
        {
            const auto *rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
            if (rows && rows->countOwner())
                return rocmQuantGemm_quantizeActivationsBlockwise_rows(input, quant, scales, m, k, device, stream, block_size, rows);
            return rocmQuantGemm_quantizeActivationsBlockwise(input, quant, scales, m, k, device, stream, block_size);
        }

        /** @brief Submit quantization with the enclosing immutable prefix authority.
         * Ordinary work retains its original ABI and count-free native kernel.
         * Counted work owns every input read and every scratch write by live row.
         */
        bool rocmQuantGemm_quantizeActivationsBlockwiseWithSumsScoped(const float *input, int8_t *quant, float *scales, int32_t *sums, int m, int k, int device, void *stream, int block_size = 32)
        {
            const auto *rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
            if (rows && rows->countOwner())
                return rocmQuantGemm_quantizeActivationsBlockwiseWithSums_rows(input, quant, scales, sums, m, k, device, stream, block_size, rows);
            return rocmQuantGemm_quantizeActivationsBlockwiseWithSums(input, quant, scales, sums, m, k, device, stream, block_size);
        }

        /** @brief Submit quantization with the enclosing immutable prefix authority.
         * Ordinary work retains its original ABI and count-free native kernel.
         * Counted work owns every input read and every scratch write by live row.
         */
        bool hipOps_fused_swiglu_quantize_blockwiseScoped(const float *gate, const float *up, int8_t *quant, float *scales, int m, int k, int device, void *stream)
        {
            const auto *rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
            if (rows && rows->countOwner())
                return hipOps_fused_swiglu_quantize_blockwise_rows(gate, up, quant, scales, m, k, device, stream, rows);
            return hipOps_fused_swiglu_quantize_blockwise(gate, up, quant, scales, m, k, device, stream);
        }

        /** @brief Submit quantization with the enclosing immutable prefix authority.
         * Ordinary work retains its original ABI and count-free native kernel.
         * Counted work owns every input read and every scratch write by live row.
         */
        bool hipOps_fused_swiglu_quantize_blockwise_with_sumsScoped(const float *gate, const float *up, int8_t *quant, float *scales, int32_t *sums, int m, int k, int device, void *stream)
        {
            const auto *rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
            if (rows && rows->countOwner())
                return hipOps_fused_swiglu_quantize_blockwise_with_sums_rows(gate, up, quant, scales, sums, m, k, device, stream, rows);
            return hipOps_fused_swiglu_quantize_blockwise_with_sums(gate, up, quant, scales, sums, m, k, device, stream);
        }

        /**
         * @brief Select counted M1 without changing the ordinary decode route.
         * The descriptor is immutable recording metadata; its device count is
         * consumed only by submitted kernels. Geometry validation remains owned
         * by VerifierKernelModeScope, including rejection of request-hole scopes.
         */
        bool countedVerifierRows(int m)
        {
            const auto *rows = ITensorGemm::VerifierKernelModeScope::rowsFor(m);
            return rows && rows->countOwner();
        }

            int computeNativeQuantisedSmallMBatchedKBForValidation(
                const std::array<int, 8> &Ns,
                int num_projections,
                int M,
                int K)
            {
                (void)Ns;
                (void)num_projections;
                (void)M;
                (void)K;

                // The batched small-M kernel may choose different split-K
                // counts as generated dispatch tables, verifier row layout,
                // or tuning environment variables change.  The workspace
                // contract is therefore cap-based: every projection receives
                // enough storage for any graph-safe split-K launch instead of
                // mirroring launcher heuristics in host-side slicing code.
                return ROCM_NATIVE_SMALL_M_GRAPH_SAFE_KB_CAP;
            }

            /**
             * @brief Publish a completed ROCm tensor write to the coherence system.
             *
             * ROCm GEMM entry points enqueue work directly into storage owned by a
             * TensorBase. Merely returning success after the enqueue is insufficient:
             * a tensor that remains marked SYNCED still advertises stale host bytes as
             * current, and a consumer on another stream has no producer event to wait
             * upon. This helper makes the write-completion contract identical for
             * decode, grouped-verifier, and prefill paths.
             *
             * Mapped tensors retain their mapped ownership state while receiving the
             * same event publication. Ordinary device allocations become
             * DEVICE_AUTHORITATIVE. transitionToWithEvent records the event on the
             * exact stream that enqueued the write, preserving asynchronous execution
             * without a stream or device synchronization.
             *
             * @param output Tensor whose device allocation was written.
             * @param device_id ROCm device ordinal that owns the allocation.
             * @param stream Producer stream containing all writes to @p output.
             */
            void publishROCmTensorWrite(
                TensorBase &output,
                int device_id,
                void *stream)
            {
                TransferEngine::publishDeviceWrite(
                    &output,
                    DeviceId::rocm(device_id),
                    stream);
            }
        } // namespace

        // =====================================================================
        // Forward declarations for HIP implementation
        // =====================================================================
        //
        // Functions are defined across three .hip files:
        //   - ROCmQuantisedGemmKernel.hip      : Common kernels + utilities (quantization, scaling, memory)
        //   - ROCmQuantisedGemmKernel_CK.hip   : CK INT8 GEMM dispatch + kernel context
        //   - ROCmQuantisedGemmKernel_INT8_VNNI.hip : VNNI prefill/decode kernels
        //

        // These functions are implemented in ROCmQuantisedGemmKernel_*.hip
        extern "C"
        {
            // Blockwise quantize FP32 activations to INT8 with per-block scales (common .hip)

            // Allocate INT8 buffer (common .hip)
            bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id);

            // Free device memory (common .hip)
            void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id);

            // Memory management helpers (all common .hip)
            bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id);
            bool rocmQuantGemm_setDevice(int rocm_device_id);

            // Apply scaling with full epilogue (common .hip) - matches CUDA's cudaQuantGemm_applyScaling
            bool rocmQuantGemm_applyScaling(
                const int32_t *d_C_int32, // [M×N] INT32 GEMM output
                float *d_C_fp32,          // [M×N] FP32 output
                const float *d_scales_A,  // [M] per-row scales
                const float *d_scales_B,  // [N] per-column scales
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing, // For beta != 0 (nullable)
                const float *d_bias,       // [N] optional bias (nullable)
                int rocm_device_id, void *stream);

            // Apply scale-A only (for grouped ratio-VNNI path where B-scales
            // are already applied per-group in the GEMV kernel).
            bool rocmQuantGemm_applyScaleA_fp32(
                const float *d_C_fp32_grouped, // [M×N] FP32 with B-scales applied
                float *d_output,               // [M×N] FP32 output
                const float *d_scales_A,       // [M] per-row activation scales
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int rocm_device_id, void *stream);

            // In-place bias addition: output[m,n] += bias[n] (common .hip)
            bool rocmQuantGemm_biasAdd(
                float *d_output,     // [M × N] FP32 output (modified in-place)
                const float *d_bias, // [N] bias vector
                int M, int N,
                int rocm_device_id, void *stream);

            // =========================================================================
            // GEMV kernel for decode (M=1) - bypasses CK INT8 GEMM entirely
            // Defined in ROCmGemvKernel.hip
            // =========================================================================

            // INT8×INT8 GEMV with VNNI-packed weights: B is [K/4 × N × 4]
            bool rocmGemv_int8_int8_int32_vnni(
                const int8_t *d_A_int8,      // [K] INT8 activation vector
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed weights
                int32_t *d_C_int32,          // [N] INT32 output vector
                int N, int K,
                int device_id, void *stream);

            bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scale_B,
                int N, int K,
                float alpha,
                float beta,
                const float *d_C_existing,
                const float *d_bias,
                int device_id, void *stream);

            // Get current activation block size for blockwise quantization (32, 64, or 128)
            int rocmGemv_int8_vnni_get_act_block_k();

            // NativeQuantised GEMM: lossless decode with FP16 per-block scales (M>1 prefill).
            // Supports all native-quantised formats via codebook_id.
            // Output is FP32 with scale_A applied inline — no separate epilogue needed.
            // Halved HBM bandwidth vs INT8 GEMM (4.5 bpw vs 8 bpw for Q4_0/IQ4_NL).
            // Public dispatcher for the physically sharded NativeQuantised GEMM TUs.
            bool rocmGemm_native_quantised_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* (FP16 m, nullable)
                const void *d_block_emins,  // uint32_t* (packed FP16 emins, Q2_K only, else NULL)
                float *d_output,
                const float *d_scales_A,
                const float *d_scales_A_blockwise, // [M × blocks_per_row] per-block scales (nullptr = row-wise)
                int M, int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);

            /** Execute physical NativeQuantised bytes with source math policy. */
            bool rocmGemm_native_quantised_fp32_with_policy(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales,
                const void *d_block_mins,
                const void *d_block_emins,
                float *d_output,
                const float *d_scales_A,
                const float *d_scales_A_blockwise,
                int M, int N, int K,
                uint8_t codebook_id,
                uint8_t arithmetic_policy_codebook_id,
                int device_id, void *stream);

            // NativeQuantised GEMV: lossless decode with FP16 per-block scales.
            // Supports all ≤6-bit quantized formats via codebook_id:
            //   Q-quants: Q4_0, Q4_1, Q5_0, Q5_1
            //   K-quants: Q2_K, Q3_K, Q4_K (via Q4_1 codebook), Q5_K (via Q5_1 codebook), Q6_K
            //   IQ-quants: IQ4_NL, IQ4_XS (physical24, source4), IQ3_S, IQ3_XXS,
            //              IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M
            // Q8_0/Q8_1/Q8_K use the INT8-VNNI path instead (already 1 byte/element).
            // Output is FP32 with scale_A applied inline — no epilogue kernel needed.
            // Uses scatter+reduce pattern: KB=1 → direct, KB>1 → scatter+reduce.
            bool rocmGemv_native_quantised_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* (FP16 m), NULL for symmetric formats
                const void *d_block_emins,  // uint32_t* (packed FP16 emins, Q2_K only, else NULL)
                float *d_C_fp32,
                const float *d_scale_A,
                float *d_partial_fp32, // [KB_MAX × N] partial buffer (nullable when KB=1)
                int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream,
                const float *d_scale_A_blockwise = nullptr);

            /** Execute serial GEMV with distinct decoder and math policy. */
            bool rocmGemv_native_quantised_fp32_with_policy(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales,
                const void *d_block_mins,
                const void *d_block_emins,
                float *d_C_fp32,
                const float *d_scale_A,
                float *d_partial_fp32,
                int N, int K,
                uint8_t codebook_id,
                uint8_t arithmetic_policy_codebook_id,
                int device_id, void *stream,
                const float *d_scale_A_blockwise = nullptr);

            bool rocmGemv_native_quantised_small_m_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* auxiliary mins/scales, nullable for symmetric formats
                const void *d_block_emins,  // uint32_t* packed emins, Q2_K only
                float *d_C_fp32,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                float *d_partial_fp32,            // [KB_MAX × M × N]
                int M, int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);

            bool rocmGemv_native_quantised_small_m_fp32_with_sums(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* auxiliary mins/scales, nullable for symmetric formats
                const void *d_block_emins,  // uint32_t* packed emins, Q2_K only
                float *d_C_fp32,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                const int32_t *d_sum_A_blockwise, // [M × blocks_per_row], nullable
                float *d_partial_fp32,            // [KB_MAX × M × N]
                int M, int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);


            bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums(
                const int8_t *d_A_int8,
                const uint8_t *const *d_payloads,
                const uint16_t *const *d_block_scales,
                const uint16_t *const *d_block_mins,
                const uint32_t *const *d_block_emins,
                const float *const *d_biases,
                float *const *d_outputs,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                const int32_t *d_sum_A_blockwise, // [M × blocks_per_row], nullable
                float *const *d_partials,         // per-projection [KB_MAX × M × N]
                const int *Ns,
                int num_projections,
                int M, int K,
                uint8_t codebook_id,
                int device_id, void *stream);



            bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums(
                const int8_t *d_A_int8,
                const uint8_t *const *d_payloads,
                const uint16_t *const *d_block_scales,
                const uint16_t *const *d_block_mins,
                const uint32_t *const *d_block_emins,
                const float *const *d_biases,
                float *const *d_outputs,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                const int32_t *d_sum_A_blockwise, // [M × blocks_per_row], nullable
                float *const *d_partials,         // per-projection [KB_MAX × M × N]
                const int *Ns,
                const uint8_t *codebook_ids,
                int num_projections,
                int M, int K,
                int device_id, void *stream);


            void rocmGemv_native_quantised_set_tuning_overrides(int kb, int target_waves_per_cu);
            void rocmGemv_native_quantised_set_decode_equivalent_m1_config(int enabled);
            void rocmGemv_native_quantised_set_serial_partition_n(int n);
            int rocmGemv_native_quantised_get_serial_partition_n();
            bool rocmGemv_native_quantised_query_serial_m1_config(
                uint8_t codebook_id,
                int N,
                int K,
                int *kb,
                int *target_waves_per_cu);
            /** @brief Query exact original source arithmetic without physical-codebook inference. */
            bool rocmGemv_native_quantised_query_serial_m1_config_with_policy(
                uint8_t policy_codebook_id, int N, int K, int *kb, int *target_waves_per_cu);
            bool rocmGemv_native_quantised_query_generated_config(
                uint8_t codebook_id,
                int M,
                int N,
                int K,
                int *kb,
                int *target_waves_per_cu);

            // =========================================================================
            // Fused FP32→INT8 quantize + GEMV + scale kernel for decode (M=1)
            // Eliminates 3-kernel pipeline: quantize → GEMV → applyScaling
            // Single kernel does per-tile quantization in shared memory, INT8 dot
            // products via v_dot4_i32_i8, and FP32 atomicAdd with scale application.
            // Bias is applied separately via rocmQuantGemm_biasAdd (atomicAdd race).
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_fused_fp32_int8_vnni(
                const float *d_A_fp32,       // [K] FP32 activation vector (unquantized)
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed INT8 weights
                float *d_C_fp32,             // [N] FP32 output vector (zeroed by kernel)
                const float *d_scales_B,     // [N] per-column weight scales
                const float *d_bias,         // [N] optional bias (nullptr if none; applied via biasAdd)
                int N, int K,
                int device_id, void *stream);

            // =========================================================================
            // Fused scatter+reduce GEMV: 2-kernel pipeline replacing the 3-kernel
            // (quantize → GEMV → scale) pipeline. Scatter kernel fuses FP32→INT8
            // quantization + INT8 GEMV and writes unscaled FP32 partials.
            // Reduce kernel sums partials and applies scale_B + bias.
            // Eliminates memset, separate quant launch, and one kernel launch.
            // For KB=1, falls back to single-kernel fused path internally.
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_fused_scatter_fp32_int8_vnni(
                const float *d_A_fp32,       // [K] FP32 activations
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed weights
                float *d_C_fp32,             // [N] FP32 output
                const float *d_scales_B,     // [N] per-column weight scales
                const float *d_bias,         // [N] optional bias (nullable)
                float *d_partial_buf,        // [KB_MAX × N] pre-allocated partial buffer
                int *d_counter,              // [ceil(N/128)] pre-allocated self-reduce counters
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing, // [N] used when beta != 0
                int device_id, void *stream);

            // =========================================================================
            bool rocmGemv_int8_scatter_vnni_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scales_B,
                const float *d_bias,
                float *d_partial_buf,
                int *d_counter,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                int device_id, void *stream);

            // =========================================================================
            // Batched INT8 scatter GEMV: multiple projections in 2 kernel launches.
            // QKV: 6→2 launches, Gate/Up: 4→2 launches.
            // Better reduce occupancy from batched grid.
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_int8_scatter_batched_vnni(
                const int8_t *d_A_int8,              // [K] pre-quantized INT8 activations
                const float *d_scale_A,              // [1] activation scale (device pointer)
                float *d_partial_buf,                // [KB_MAX × max_N] partial buffer
                int num_projections,                 // Number of projections (1..8)
                const int8_t *const *d_B_ptrs,       // [num_proj] VNNI weight pointers
                float *const *d_C_ptrs,              // [num_proj] FP32 output pointers
                const float *const *d_scales_B_ptrs, // [num_proj] weight scale pointers
                const float *const *d_bias_ptrs,     // [num_proj] bias pointers
                const int *N_per_proj,               // [num_proj] output dimensions
                int K,
                float alpha, float beta,
                int device_id, void *stream);

            bool rocmGemv_int8_scatter_batched_vnni_blockwise(
                const int8_t *d_A_int8,
                const float *d_scales_A_blockwise,
                float *d_partial_buf,
                int num_projections,
                const int8_t *const *d_B_ptrs,
                float *const *d_C_ptrs,
                const float *const *d_scales_B_ptrs,
                const float *const *d_bias_ptrs,
                const int *N_per_proj,
                int K,
                float alpha, float beta,
                int device_id, void *stream);

            bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                const int8_t *d_A_int8,
                const float *d_scales_A_blockwise,
                int num_projections,
                const int8_t *const *d_B_ptrs,
                float *const *d_C_ptrs,
                const float *const *d_scales_B_ptrs,
                const float *const *d_bias_ptrs,
                const int *N_per_proj,
                int K,
                float alpha, float beta,
                int device_id, void *stream);

            // =========================================================================
            // PREFILL GEMM scaffold entrypoints (M>1)
            //
            // These functions are intentionally scaffold-first in early phases. They
            // let C++ dispatch code route into a stable C API now, while HIP kernel
            // internals are implemented incrementally behind the same ABI.
            // =========================================================================

            // INT8 VNNI native prefill GEMM scaffold: INT8 x INT8 -> INT32
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill(
                const int8_t *d_A_int8,      // [M x K] row-major INT8 activations
                const int8_t *d_B_int8_vnni, // [K/4 x N x 4] VNNI-packed INT8 weights
                int32_t *d_C_int32,          // [M x N] row-major INT32 accumulators
                int M, int N, int K,
                int tile_variant,
                int cpt,
                int device_id, void *stream);

            // INT8 VNNI native prefill GEMM split-K/grid-kpar variant.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int split_k_slices,
                int tile_variant,
                int cpt,
                int kernel_body_variant,
                int grid_swizzle_variant,
                int device_id, void *stream);

            // INT8 VNNI wide-tile V3 prefill GEMM (LDS double-buffered pipeline).
            // kt_select: 8 or 16.  Auto-dispatched for K-heavy shapes (k >= n).
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int kt_select,
                int device_id, void *stream);

            // INT8 VNNI wide-tile V7 prefill GEMM (N128 safe-tile split, branchless interior).
            // kt_select: 8 or 16.  Auto-dispatched for N-heavy shapes (n > k).
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int kt_select,
                int device_id, void *stream);

            // INT8 VNNI wide-tile V3 blockwise GEMM (activation scales baked in, FP32 output).
            // KT=8 forced (1 block per tile). Caller applies weight-only epilogue.
            // m_tile_override: -1=auto, or 16/32 to force specific M_TILE.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v3_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int M, int N, int K,
                int device_id, void *stream,
                int m_tile_override = -1,
                int unroll_kk = -1);

            // INT8 VNNI wide-tile V7 blockwise GEMM (activation scales baked in, FP32 output).
            // KT=8 forced (1 block per tile). Caller applies weight-only epilogue.
            // m_tile_override: -1=auto, or 16/32/64 to force specific M_TILE.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v7_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int M, int N, int K,
                int device_id, void *stream,
                int m_tile_override = -1,
                int unroll_kk = -1);

            // Weight-only scaling epilogue for blockwise GEMM output.
            // Input is FP32 (activation scales already applied by blockwise kernel).
            // Applies: output = alpha * input * scale_B[n] + beta*existing + bias
            // Defined in ROCmQuantisedGemmKernel.hip.
            bool rocmQuantGemm_applyScaling_weightOnly(
                const float *d_C_fp32_in,
                float *d_output,
                const float *d_scales_B,
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int rocm_device_id, void *stream);
        }

        // =====================================================================
        // 3-Way Dispatch - Optimized kernel selection based on M
        // =====================================================================
        //
        // The .hip file implements 3-way dispatch using GemmMNPadding-enabled kernels:
        //
        //   ┌─────────────────────────────────────────────────────────────────┐
        //   │ M ≤ 32?      ───YES───> 32×32 kernel (handles M=1 natively!)   │
        //   │         │                                                       │
        //   │         NO                                                      │
        //   │         │                                                       │
        //   │ 32 < M < 128? ─YES───> 64×64 kernel (good for small batches)   │
        //   │         │                                                       │
        //   │         NO                                                      │
        //   │         │                                                       │
        //   │ M >= 128  ───────────> 128×128 kernel (peak throughput)        │
        //   └─────────────────────────────────────────────────────────────────┘
        //
        // Performance (FFN Up: K=3584, N=18944, MI50 gfx906):
        //   M=1:   0.46ms, ~2.4 TFLOPS  (32×32 kernel, 2.6x faster than 128-pad)
        //   M=32:  0.53ms, ~8.2 TFLOPS  (32×32 kernel)
        //   M=64:  0.51ms, ~17 TFLOPS   (64×64 kernel, 1.3x faster than 128-pad)
        //   M=128: 0.82ms, ~21 TFLOPS   (128×128 kernel)
        //
        // =====================================================================

        // Minimum M that the 32×32 kernel can handle without explicit padding
        // With GemmMNPadding, any M >= 1 works, but we keep 8 as the threshold
        // for explicit padding to match the old code path (for safety)
        constexpr int CK_MIN_M_FOR_EXPLICIT_PADDING = 8;
        constexpr int CK_MIN_N = 32; // All kernels: minimum N
        constexpr int CK_MIN_K = 32; // All kernels: minimum K

        // Returns the padded M value needed (only for M < 8)
        // The 32×32 kernel with GemmMNPadding handles most M values natively
        inline int getPaddedM(int M)
        {
            return (M < CK_MIN_M_FOR_EXPLICIT_PADDING) ? CK_MIN_M_FOR_EXPLICIT_PADDING : M;
        }

        // Check if M needs explicit padding (rare - only M < 8)
        inline bool needsMPadding(int M)
        {
            return M < CK_MIN_M_FOR_EXPLICIT_PADDING;
        }

        // =====================================================================
        // PIMPL implementation struct
        // =====================================================================

        struct ROCmQuantisedGemmKernel::Impl
        {
            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            // VNNI-only on device: native-quantised (≤6-bit) or INT8-VNNI (8-bit).
            int8_t *d_weights_int8_vnni = nullptr;    // [K/4 x N x 4] VNNI layout (sole INT8 device copy)
            float *d_scales_B = nullptr;              // [N] per-column scales
            uint8_t *d_weights_native_quantised = nullptr; // [blocks_per_row × N × payload_bytes]
            void *d_weights_native_scales = nullptr;  // [blocks_per_row × N] __half*
            void *d_weights_native_mins = nullptr;    // [blocks_per_row × N] __half* (asymmetric only, else NULL)
            void *d_weights_native_emins = nullptr;   // [blocks_per_row × N] uint32_t* (Q2_K only, packed {lo,hi} FP16 emins)
            uint8_t native_quantised_codebook_id = 0;
            uint32_t native_quantised_blocks_per_row = 0;
            NativeQuantisedSourceIdentity native_source_identity;
            // Borrowed immutable IQ view is bound cold once and never replaced.
            // A backend reset invalidates it together with the prepared weights.
            NativeEmbeddingIQTables residual_iq_tables;
            std::atomic<std::uint64_t> residual_iq_generation{0};
            std::once_flag residual_iq_once;
            NativeQuantisedReusableDeviceAllocationFormat native_allocation_format;
            bool has_native_quantised = false;
            void *startup_h2d_pinned_scales = nullptr;
            void *startup_h2d_pinned_vnni = nullptr;
            void *startup_h2d_pinned_native_quantised = nullptr;
            void *startup_h2d_pinned_native_scales = nullptr;
            void *startup_h2d_pinned_native_mins = nullptr;
            void *startup_h2d_pinned_native_emins = nullptr;

            // Work buffer pointers - obtained from workspace at execution time
            // These are NOT owned by the kernel - they point into workspace-managed memory
            int8_t *d_A_int8 = nullptr;            // [M x K] quantized activations
            float *d_scales_A = nullptr;           // [M] per-row scales (row-wise mode)
            float *d_scales_A_blockwise = nullptr; // [M x blocks_per_row] per-block scales (blockwise mode)
            int32_t *d_sums_A_blockwise = nullptr; // [M x blocks_per_row] per-block quantized activation sums
            int32_t *d_C_int32 = nullptr;          // [M x N] INT32 accumulator

            // Transfer work buffers - also from workspace
            float *d_A_fp32 = nullptr; // [M x K] input FP32
            float *d_C_fp32 = nullptr; // [M x N] output FP32

            // Scatter+reduce partial buffer (from workspace)
            float *d_scatter_partial = nullptr; // [KB_MAX × N] FP32 scatter partials
            float *d_scatter_partial_batched = nullptr; // [batch × KB_MAX × N] FP32 scatter partials
            int *d_selfreduce_counters = nullptr; // [ceil(N / 128)] INT32 self-reduce counters

            // Capacity tracking for workspace buffers (set during validateWorkspace)
            size_t d_A_fp32_capacity = 0;
            size_t d_C_fp32_capacity = 0;

            // Workspace validation cache: skip redundant hash map lookups when
            // the same workspace is re-validated across GEMM calls.
            const void *validated_workspace = nullptr;

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            // ROCm device ID for proper cleanup
            int rocm_device_id = 0;

            ~Impl()
            {
                // Only free weight memory if we own it (not from ROCmPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_int8_vnni)
                        rocmQuantGemm_freeDevice(d_weights_int8_vnni, rocm_device_id);
                    if (d_scales_B)
                        rocmQuantGemm_freeDevice(d_scales_B, rocm_device_id);
                    if (d_weights_native_quantised)
                        rocmQuantGemm_freeDevice(d_weights_native_quantised, rocm_device_id);
                    if (d_weights_native_scales)
                        rocmQuantGemm_freeDevice(d_weights_native_scales, rocm_device_id);
                    if (d_weights_native_mins)
                        rocmQuantGemm_freeDevice(d_weights_native_mins, rocm_device_id);
                    if (d_weights_native_emins)
                        rocmQuantGemm_freeDevice(d_weights_native_emins, rocm_device_id);
#ifdef HAVE_ROCM
                    auto free_pinned = [](void *&ptr)
                    {
                        if (ptr)
                        {
                            (void)hipHostFree(ptr);
                            ptr = nullptr;
                        }
                    };
                    free_pinned(startup_h2d_pinned_scales);
                    free_pinned(startup_h2d_pinned_vnni);
                    free_pinned(startup_h2d_pinned_native_quantised);
                    free_pinned(startup_h2d_pinned_native_scales);
                    free_pinned(startup_h2d_pinned_native_mins);
                    free_pinned(startup_h2d_pinned_native_emins);
#endif
                }
                // Work buffers are NOT freed - they are owned by workspace
            }
        };

        namespace
        {
            /**
             * @brief Resolve the serial arithmetic authority for prepared bytes.
             *
             * Device execution may normalize source formats (notably a CPU
             * promotion into codebook 23), but launch geometry must remain
             * owned by the original source codebook. Constructors establish
             * provenance once; reaching inference without it is a fatal
             * lifecycle violation rather than permission to guess.
             */
            inline uint8_t requireNativeQuantisedArithmeticPolicyCodebook(
                uint8_t execution_codebook_id,
                const NativeQuantisedSourceIdentity &source_identity)
            {
                if (!source_identity.present)
                {
                    throw std::logic_error(
                        "[ROCmQuantisedGemmKernel] NativeQuantised execution lost "
                        "source arithmetic identity for codebook " +
                        std::to_string(
                            static_cast<int>(execution_codebook_id)));
                }
                return canonicalDeviceVnniCodebookId(
                    source_identity.codebook_id);
            }

            inline bool validatePointerDeviceOrLog(
                const void *ptr,
                int expected_device,
                const char *pointer_name,
                const char *scope)
            {
#if LLAMINAR_ASSERTIONS_ACTIVE
                if (!ptr)
                {
                    LOG_ERROR("[" << scope << "] " << pointer_name << " is null");
                    return false;
                }

                hipPointerAttribute_t attr{};
                hipError_t err = hipPointerGetAttributes(&attr, ptr);
                if (err == hipSuccess)
                {
                    if (attr.device != expected_device)
                    {
                        LOG_ERROR("[" << scope << "] " << pointer_name
                                      << " on wrong device: ptr=" << ptr
                                      << " attr.device=" << attr.device
                                      << " expected=" << expected_device);
                        return false;
                    }
                    return true;
                }

                LOG_WARN("[" << scope << "] hipPointerGetAttributes failed for " << pointer_name
                             << " ptr=" << ptr << ": " << hipGetErrorString(err));
                return true;
#else
                // Release builds: skip HIP runtime validation for performance.
                // Each hipPointerGetAttributes() call costs ~1-5μs, and with ~8 calls
                // per GEMM × ~280 GEMMs per decode token, this adds ~2-10ms per token.
                (void)ptr;
                (void)expected_device;
                (void)pointer_name;
                (void)scope;
                return true;
#endif
            }

            template <typename UploadT>
            inline void freeStartupPinnedStaging(UploadT &upload)
            {
#ifdef HAVE_ROCM
                auto free_if_set = [](void *&ptr)
                {
                    if (ptr)
                    {
                        const hipError_t err = hipHostFree(ptr);
                        if (err != hipSuccess)
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] hipHostFree failed for startup staging buffer: "
                                     << hipGetErrorString(err));
                        }
                        ptr = nullptr;
                    }
                };

                free_if_set(upload.startup_h2d_pinned_scales);
                free_if_set(upload.startup_h2d_pinned_vnni);
                free_if_set(upload.startup_h2d_pinned_native_quantised);
                free_if_set(upload.startup_h2d_pinned_native_scales);
                free_if_set(upload.startup_h2d_pinned_native_mins);
#else
                (void)upload;
#endif
            }

            inline hipError_t startupMemcpyAsyncOrSync(
                void *dst,
                const void *src,
                size_t bytes,
                bool async_enabled,
                void *stream,
                void **pinned_staging_slot,
                const char *log_scope,
                const char *tensor_name)
            {
#ifdef HAVE_ROCM
                if (pinned_staging_slot)
                {
                    *pinned_staging_slot = nullptr;
                }

                if (async_enabled)
                {
                    // Async path: stage through pinned host memory to avoid
                    // ROCm hipMemcpy failures on pageable brk-heap addresses,
                    // and to enable DMA overlap with other GPU work.
                    const void *copy_src = src;
                    if (bytes > 0 && pinned_staging_slot)
                    {
                        void *pinned_buf = nullptr;
                        const hipError_t alloc_err = hipHostMalloc(&pinned_buf, bytes, hipHostMallocDefault);
                        if (alloc_err == hipSuccess && pinned_buf)
                        {
                            std::memcpy(pinned_buf, src, bytes);
                            *pinned_staging_slot = pinned_buf;
                            copy_src = pinned_buf;
                        }
                        else
                        {
                            LOG_ERROR("[" << log_scope << "] hipHostMalloc failed for startup staging (" << tensor_name
                                         << ", bytes=" << bytes << "): " << hipGetErrorString(alloc_err)
                                         << ". Refusing pageable-host fallback.");
                            return alloc_err;
                        }
                    }
                    return hipMemcpyAsync(dst, copy_src, bytes, hipMemcpyHostToDevice, reinterpret_cast<hipStream_t>(stream));
                }

                LOG_ERROR("[" << log_scope << "] explicit H2D stream is required for startup upload ("
                              << tensor_name << ", bytes=" << bytes << ")");
                return hipErrorInvalidValue;
#else
                (void)dst;
                (void)src;
                (void)bytes;
                (void)async_enabled;
                (void)stream;
                (void)pinned_staging_slot;
                (void)log_scope;
                (void)tensor_name;
                return hipSuccess;
#endif
            }

        }

        /** @brief Borrow typed HIP handles without adding another native resource owner. */
        struct HIPProjectionResourceView
        {
            static_assert(static_cast<int>(GPUProjectionStreamCount::Four) == ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS);
            std::array<hipStream_t, ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS> streams{};
            std::array<hipEvent_t, ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS> completion{};
            hipEvent_t quant_ready = nullptr;
            const int count = ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS;
        };

        /**
         * @brief Project only a live shared lease's exact handles into HIP native types.
         * @param resources Complete resource view borrowed from a live submission.
         * @return Non-owning typed handles with the unchanged fixed stream population.
         */
        static HIPProjectionResourceView hipProjectionView(const GPUProjectionResourceView &resources)
        {
            if (resources.count != ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS)
                throw std::logic_error("HIP projection resource population disagrees with the admitted workspace contract");
            HIPProjectionResourceView native;
            for (int index = 0; index < resources.count; ++index)
            {
                native.streams[index] = static_cast<hipStream_t>(resources.streams[index]);
                native.completion[index] = static_cast<hipEvent_t>(resources.completion[index]);
            }
            native.quant_ready = static_cast<hipEvent_t>(resources.quant_ready);
            return native;
        }
        /**
         * @brief Own the complete eager fork/join or borrow its enclosing exact native capture.
         * @param ordinal Exact worker-context device. @param root Non-null parent stream.
         * @return Move-only interval lease retaining exclusion for the complete caller scope.
         * @throws For failed stream observation, missing preparation or incompatible capture ownership.
         */
        static GPUProjectionSubmission acquireHIPProjectionSubmission(int ordinal, void *root)
        {
            hipStreamCaptureStatus status{};
            if (!root || hipStreamIsCapturing(static_cast<hipStream_t>(root), &status) != hipSuccess)
                throw std::runtime_error("HIP projection submission requires an authenticated exact stream");
            auto context = GPUDeviceContextPool::instance().retainAMDContext(ordinal);
            return context->acquireProjectionSubmission(
                root, GPUProjectionSubmissionKind::Eager, status != hipStreamCaptureStatusNone);
        }
    } // namespace rocm
} // namespace llaminar2
