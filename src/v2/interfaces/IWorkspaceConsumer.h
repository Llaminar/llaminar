/**
 * @file IWorkspaceConsumer.h
 * @brief Interface for kernels that consume centralized workspace buffers
 *
 * This interface enables kernels to declare their workspace requirements and
 * receive pre-allocated buffers from DeviceWorkspaceManager instead of managing
 * their own ad-hoc allocations.
 *
 * Works with any device type: CPU, CUDA, ROCm.
 * (Formerly IGpuWorkspaceConsumer.h)
 *
 * ## Design Goals
 *
 * 1. **Memory Budgeting**: Centralized allocation allows global VRAM budget control
 * 2. **Hot-Path Efficiency**: No allocations during GEMM execution
 * 3. **Backward Compatibility**: Kernels work in both legacy and managed modes
 *
 * ## Usage Pattern
 *
 * ```cpp
 * // At graph construction time:
 * auto requirements = kernel->getWorkspaceRequirements(max_m, max_n, max_k);
 * workspaceManager->allocate(requirements);
 *
 * // Bind workspace to kernel:
 * kernel->bindWorkspace(workspaceManager);
 *
 * // During execution (no allocations):
 * kernel->multiply_tensor(A, C, m, n, k);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class DeviceWorkspaceManager;
    struct WorkspaceRequirements;

    /**
     * @brief Interface for kernels that consume centralized workspace buffers
     *
     * Kernels implementing this interface:
     * 1. Declare workspace requirements via getWorkspaceRequirements()
     * 2. Receive pre-allocated buffers via bindWorkspace()
     * 3. Use bound workspace instead of internal allocations when available
     * 4. Fall back to legacy internal buffers when workspace not bound
     *
     * (Formerly IGpuWorkspaceConsumer)
     */
    class IWorkspaceConsumer
    {
    public:
        virtual ~IWorkspaceConsumer() = default;

        // =========================================================================
        // Workspace Requirements Declaration
        // =========================================================================

        /**
         * @brief Get workspace buffer requirements for this kernel
         *
         * Returns the set of buffers this kernel needs for execution at the given
         * dimensions. The caller allocates these buffers in DeviceWorkspaceManager
         * and then calls bindWorkspace().
         *
         * @param m Number of rows (batch size / sequence length)
         * @param n Number of output features (may be 0 if kernel-specific)
         * @param k Number of input features (may be 0 if kernel-specific)
         * @return WorkspaceRequirements describing all needed buffers
         *
         * @note Dimensions are typically the MAXIMUM expected values to avoid
         *       re-allocation during inference. For variable-length sequences,
         *       pass the maximum sequence length used in KV cache.
         *
         * @note If n and k are 0, kernel uses its internal N_ and K_ dimensions.
         */
        virtual WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const = 0;

        // =========================================================================
        // Workspace Binding
        // =========================================================================

        /**
         * @brief Bind a workspace manager to this kernel
         *
         * After binding, the kernel uses buffers from the workspace manager
         * instead of its internal ad-hoc allocations. The workspace manager
         * must have allocated all buffers returned by getWorkspaceRequirements()
         * for the maximum expected dimensions.
         *
         * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
         *                  Pass nullptr to unbind and return to legacy mode.
         *
         * @note Thread Safety: This method should only be called during setup,
         *       not during concurrent kernel execution.
         */
        virtual void bindWorkspace(DeviceWorkspaceManager *workspace) = 0;

        /**
         * @brief Unbind workspace and return to legacy mode
         *
         * Equivalent to bindWorkspace(nullptr). Kernel will use internal
         * buffer management after this call.
         */
        virtual void unbindWorkspace()
        {
            bindWorkspace(nullptr);
        }

        // =========================================================================
        // Workspace Query
        // =========================================================================

        /**
         * @brief Check if a workspace is currently bound
         *
         * @return true if bindWorkspace() was called with non-null workspace
         */
        virtual bool hasWorkspace() const = 0;

        /**
         * @brief Get the currently bound workspace manager
         *
         * @return Pointer to bound workspace manager, or nullptr if not bound
         */
        virtual DeviceWorkspaceManager *getWorkspace() const = 0;
    };

    // =============================================================================
    // Standard Buffer Names for GEMM Kernels
    // =============================================================================

    /**
     * Standard buffer names used by GEMM workspace consumers.
     *
     * Using consistent names across CUDA/ROCm kernels enables the workspace
     * manager to share buffers between different kernel types on the same device.
     */
    namespace GemmWorkspaceBuffers
    {
        // INT8 quantization buffers
        constexpr const char *QUANT_A = "gemm_quant_a";     ///< [M × K] INT8 quantized activations
        constexpr const char *SCALES_A = "gemm_scales_a";   ///< [M] FP32 per-row activation scales
        constexpr const char *ACC_INT32 = "gemm_acc_int32"; ///< [M × N] INT32 accumulator

        // FP32 temporary buffers
        constexpr const char *TEMP_A_FP32 = "gemm_temp_a_fp32"; ///< [M × K] FP32 activation copy
        constexpr const char *TEMP_C_FP32 = "gemm_temp_c_fp32"; ///< [M × N] FP32 output copy

        // FP16 slab buffers (for slab-based FP16 GEMM)
        constexpr const char *SLAB_A_FP16 = "gemm_slab_a_fp16"; ///< [slab_m × slab_k] FP16 A slab
        constexpr const char *SLAB_B_FP16 = "gemm_slab_b_fp16"; ///< [slab_k × slab_n] FP16 B slab
        constexpr const char *SLAB_C_FP16 = "gemm_slab_c_fp16"; ///< [slab_m × slab_n] FP16 C slab

        // Full-matrix FP16 buffers (legacy path, deprecated in managed mode)
        constexpr const char *FULL_A_FP16 = "gemm_full_a_fp16"; ///< [M × K] FP16 full matrix
        constexpr const char *FULL_B_FP16 = "gemm_full_b_fp16"; ///< [K × N] FP16 full matrix
        constexpr const char *FULL_C_FP16 = "gemm_full_c_fp16"; ///< [M × N] FP16 full matrix

        // ROCm M-padding buffers (for CK when M < 8)
        constexpr const char *ROCM_A_PADDED = "rocm_a_padded";             ///< [padded_m × K] INT8 padded activations
        constexpr const char *ROCM_SCALE_A_PADDED = "rocm_scale_a_padded"; ///< [padded_m] FP32 padded scales
        constexpr const char *ROCM_E_PADDED = "rocm_e_padded";             ///< [padded_m × N] FP32 padded output
        constexpr const char *ROCM_CK_INT32 = "rocm_ck_int32";             ///< [padded_m × N] INT32 CK accumulator
    }

} // namespace llaminar2
