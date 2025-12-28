/**
 * @file KernelSnapshotInfo.h
 * @brief Kernel-level snapshot capability interface
 *
 * Provides compile-time enforcement of snapshot support for all kernels.
 * Every kernel MUST implement getKernelSnapshotInfo() to declare its
 * snapshot-compatible buffers and parameters.
 *
 * Design Goals:
 * 1. **Compile-time enforcement**: Pure virtual interface = compiler error if not implemented
 * 2. **Self-documenting**: Kernels declare their I/O buffers explicitly
 * 3. **Predictable wiring**: Stages can query kernel snapshot info for consistent capture
 * 4. **Type-safe**: Buffers declare their dtype for proper dequantization
 *
 * Usage:
 * @code
 * // In kernel implementation:
 * KernelSnapshotInfo MyGemmKernel::getKernelSnapshotInfo() const override {
 *     return KernelSnapshotInfo::gemm()
 *         .withInput("A", "activations", KernelBufferDtype::FP32)
 *         .withWeight("B", "weights", KernelBufferDtype::Q4_0)
 *         .withOutput("C", "output", KernelBufferDtype::FP32);
 * }
 *
 * // In stage (predictable wiring):
 * auto kernel_info = gemm_kernel->getKernelSnapshotInfo();
 * for (const auto& output : kernel_info.outputs) {
 *     dump_info.addOutput(output.name, ...);
 * }
 * @endcode
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Buffer Data Type Enum
    // =========================================================================

    /**
     * @brief Data types for kernel snapshot buffers
     *
     * Used to properly dequantize/convert buffers for snapshot comparison
     * against PyTorch FP32 reference.
     */
    enum class KernelBufferDtype
    {
        // Floating point
        FP32,
        FP16,
        BF16,

        // Quantized formats
        Q8_0,
        Q8_1,
        Q16_1,
        Q4_0,
        Q4_1,
        Q5_0,
        Q5_1,
        Q6_K,
        Q8_K,
        Q2_K,
        Q3_K,
        Q4_K,
        Q5_K,
        IQ4_NL,
        IQ4_XS,
        IQ3_S,
        IQ3_XXS,
        IQ2_S,
        IQ2_XS,
        IQ2_XXS,
        IQ1_S,
        IQ1_M,

        // Integer types (for internal intermediates)
        INT8,
        INT16,
        INT32,

        // Special
        Unknown
    };

    /**
     * @brief Convert KernelBufferDtype to string representation
     */
    inline const char *to_string(KernelBufferDtype dtype)
    {
        switch (dtype)
        {
        case KernelBufferDtype::FP32:
            return "FP32";
        case KernelBufferDtype::FP16:
            return "FP16";
        case KernelBufferDtype::BF16:
            return "BF16";
        case KernelBufferDtype::Q8_0:
            return "Q8_0";
        case KernelBufferDtype::Q8_1:
            return "Q8_1";
        case KernelBufferDtype::Q16_1:
            return "Q16_1";
        case KernelBufferDtype::Q4_0:
            return "Q4_0";
        case KernelBufferDtype::Q4_1:
            return "Q4_1";
        case KernelBufferDtype::Q5_0:
            return "Q5_0";
        case KernelBufferDtype::Q5_1:
            return "Q5_1";
        case KernelBufferDtype::Q6_K:
            return "Q6_K";
        case KernelBufferDtype::Q8_K:
            return "Q8_K";
        case KernelBufferDtype::Q2_K:
            return "Q2_K";
        case KernelBufferDtype::Q3_K:
            return "Q3_K";
        case KernelBufferDtype::Q4_K:
            return "Q4_K";
        case KernelBufferDtype::Q5_K:
            return "Q5_K";
        case KernelBufferDtype::IQ4_NL:
            return "IQ4_NL";
        case KernelBufferDtype::IQ4_XS:
            return "IQ4_XS";
        case KernelBufferDtype::IQ3_S:
            return "IQ3_S";
        case KernelBufferDtype::IQ3_XXS:
            return "IQ3_XXS";
        case KernelBufferDtype::IQ2_S:
            return "IQ2_S";
        case KernelBufferDtype::IQ2_XS:
            return "IQ2_XS";
        case KernelBufferDtype::IQ2_XXS:
            return "IQ2_XXS";
        case KernelBufferDtype::IQ1_S:
            return "IQ1_S";
        case KernelBufferDtype::IQ1_M:
            return "IQ1_M";
        case KernelBufferDtype::INT8:
            return "INT8";
        case KernelBufferDtype::INT16:
            return "INT16";
        case KernelBufferDtype::INT32:
            return "INT32";
        case KernelBufferDtype::Unknown:
        default:
            return "Unknown";
        }
    }

    // =========================================================================
    // Kernel Buffer Descriptors
    // =========================================================================

    /**
     * @brief Describes a kernel input buffer for snapshot capture
     */
    struct KernelInputDesc
    {
        const char *name;        ///< Semantic name (e.g., "A", "Q", "input")
        const char *description; ///< Human-readable description
        KernelBufferDtype dtype; ///< Data type for proper dequantization
        bool required;           ///< Whether this input is always present

        KernelInputDesc(const char *n, const char *desc, KernelBufferDtype d, bool req = true)
            : name(n), description(desc), dtype(d), required(req)
        {
        }
    };

    /**
     * @brief Describes a kernel output buffer for snapshot capture
     */
    struct KernelOutputDesc
    {
        const char *name;        ///< Semantic name (e.g., "C", "output", "context")
        const char *description; ///< Human-readable description
        KernelBufferDtype dtype; ///< Data type for proper dequantization
        bool is_intermediate;    ///< True if intermediate (may need explicit capture)

        KernelOutputDesc(const char *n, const char *desc, KernelBufferDtype d, bool intermediate = false)
            : name(n), description(desc), dtype(d), is_intermediate(intermediate)
        {
        }
    };

    /**
     * @brief Describes a kernel weight tensor for snapshot capture
     */
    struct KernelWeightDesc
    {
        const char *name;        ///< Semantic name (e.g., "B", "Wq", "gamma")
        const char *description; ///< Human-readable description
        KernelBufferDtype dtype; ///< Native weight dtype (for dequant)

        KernelWeightDesc(const char *n, const char *desc, KernelBufferDtype d)
            : name(n), description(desc), dtype(d)
        {
        }
    };

    /**
     * @brief Describes a scalar parameter for snapshot capture
     */
    struct KernelScalarDesc
    {
        const char *name;        ///< Parameter name (e.g., "epsilon", "scale")
        const char *description; ///< Human-readable description
        KernelBufferDtype dtype; ///< Type (usually FP32 or INT32)

        KernelScalarDesc(const char *n, const char *desc, KernelBufferDtype d = KernelBufferDtype::FP32)
            : name(n), description(desc), dtype(d)
        {
        }
    };

    // =========================================================================
    // KernelSnapshotInfo - Main Snapshot Metadata Structure
    // =========================================================================

    /**
     * @brief Kernel snapshot capability metadata
     *
     * Declares all buffers and parameters a kernel can emit for snapshots.
     * This is returned by getKernelSnapshotInfo() and allows stages to
     * consistently wire up snapshot capture.
     *
     * Use the builder pattern for clean construction:
     * @code
     * return KernelSnapshotInfo::gemm()
     *     .withInput("A", "activations", KernelBufferDtype::FP32)
     *     .withWeight("B", "weights", KernelBufferDtype::Q4_0)
     *     .withOutput("C", "output", KernelBufferDtype::FP32);
     * @endcode
     */
    struct KernelSnapshotInfo
    {
        const char *kernel_name;        ///< Kernel type name (e.g., "GEMM", "Attention")
        const char *kernel_description; ///< Human-readable description

        std::vector<KernelInputDesc> inputs;
        std::vector<KernelOutputDesc> outputs;
        std::vector<KernelWeightDesc> weights;
        std::vector<KernelScalarDesc> scalars;

        /// True if kernel has nothing to snapshot (e.g., pure pass-through)
        bool is_passthrough = false;

        // =====================================================================
        // Builder Methods
        // =====================================================================

        /**
         * @brief Add an input buffer descriptor
         */
        KernelSnapshotInfo &withInput(const char *name, const char *desc,
                                      KernelBufferDtype dtype, bool required = true)
        {
            inputs.emplace_back(name, desc, dtype, required);
            return *this;
        }

        /**
         * @brief Add an output buffer descriptor
         */
        KernelSnapshotInfo &withOutput(const char *name, const char *desc,
                                       KernelBufferDtype dtype, bool intermediate = false)
        {
            outputs.emplace_back(name, desc, dtype, intermediate);
            return *this;
        }

        /**
         * @brief Add a weight tensor descriptor
         */
        KernelSnapshotInfo &withWeight(const char *name, const char *desc,
                                       KernelBufferDtype dtype)
        {
            weights.emplace_back(name, desc, dtype);
            return *this;
        }

        /**
         * @brief Add a scalar parameter descriptor
         */
        KernelSnapshotInfo &withScalar(const char *name, const char *desc,
                                       KernelBufferDtype dtype = KernelBufferDtype::FP32)
        {
            scalars.emplace_back(name, desc, dtype);
            return *this;
        }

        /**
         * @brief Mark as passthrough (nothing to snapshot)
         */
        KernelSnapshotInfo &asPassthrough()
        {
            is_passthrough = true;
            return *this;
        }

        // =====================================================================
        // Factory Methods for Common Kernel Types
        // =====================================================================

        /**
         * @brief Create info for GEMM kernel (C = A @ B)
         */
        static KernelSnapshotInfo gemm()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "GEMM";
            info.kernel_description = "Matrix multiplication: C = alpha * A @ B + beta * C";
            return info;
        }

        /**
         * @brief Create info for Attention kernel
         */
        static KernelSnapshotInfo attention()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "Attention";
            info.kernel_description = "Multi-head attention: softmax(Q @ K^T / sqrt(d)) @ V";
            return info;
        }

        /**
         * @brief Create info for RoPE kernel
         */
        static KernelSnapshotInfo rope()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "RoPE";
            info.kernel_description = "Rotary Position Embedding";
            return info;
        }

        /**
         * @brief Create info for SwiGLU kernel
         */
        static KernelSnapshotInfo swiglu()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "SwiGLU";
            info.kernel_description = "SwiGLU activation: output = up * swish(gate)";
            return info;
        }

        /**
         * @brief Create info for RMSNorm kernel
         */
        static KernelSnapshotInfo rmsnorm()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "RMSNorm";
            info.kernel_description = "Root Mean Square Layer Normalization";
            return info;
        }

        /**
         * @brief Create info for Softmax kernel
         */
        static KernelSnapshotInfo softmax()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "Softmax";
            info.kernel_description = "Row-wise softmax normalization";
            return info;
        }

        /**
         * @brief Create info for Embedding kernel
         */
        static KernelSnapshotInfo embedding()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "Embedding";
            info.kernel_description = "Token embedding lookup";
            return info;
        }

        /**
         * @brief Create info for Fused Attention + Wo + Residual kernel
         *
         * Used by integer-domain fused attention kernels like Q16_1 that combine:
         * - Attention computation (Q×K^T → softmax → ×V)
         * - Wo projection (context × Wo weights)
         * - Residual add
         * All in a single fused kernel to maximize register utilization.
         */
        static KernelSnapshotInfo fusedAttentionWo()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "FusedAttentionWo";
            info.kernel_description = "Fused attention + Wo projection + residual add (integer-domain)";
            return info;
        }

        /**
         * @brief Create info for passthrough/no-op kernel
         */
        static KernelSnapshotInfo passthrough()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "Passthrough";
            info.kernel_description = "No computation, buffers unchanged";
            info.is_passthrough = true;
            return info;
        }

        /**
         * @brief Create info for unspecified/not-yet-implemented kernel snapshot
         * 
         * Used as default return value for kernels that haven't had their
         * snapshot info explicitly defined yet. Allows gradual rollout.
         */
        static KernelSnapshotInfo unspecified()
        {
            KernelSnapshotInfo info;
            info.kernel_name = "Unspecified";
            info.kernel_description = "Kernel snapshot info not yet defined";
            info.is_passthrough = false;
            return info;
        }
    };

    // =========================================================================
    // IKernelSnapshotCapable - Pure Virtual Interface
    // =========================================================================

    /**
     * @brief Interface that all kernels MUST implement for snapshot support
     *
     * This is the core mechanism for compile-time enforcement of snapshot
     * capability in kernels. Any class inheriting from ITensorKernel will
     * also inherit from this interface and MUST implement getKernelSnapshotInfo().
     *
     * **Compile-time enforcement**: This is a pure virtual method. If a kernel
     * implementation fails to override it, the code won't compile.
     *
     * **Design rationale**: By making snapshot support a first-class requirement
     * at the kernel level, we ensure:
     * 1. New kernels cannot be added without considering snapshot support
     * 2. Stages can reliably query kernel capabilities
     * 3. Documentation of kernel I/O is built into the code
     */
    class IKernelSnapshotCapable
    {
    public:
        virtual ~IKernelSnapshotCapable() = default;

        /**
         * @brief Get snapshot capability metadata for this kernel
         *
         * MUST be implemented by all kernel implementations. Returns metadata
         * describing all input, output, weight, and scalar buffers that can
         * be captured for debugging/validation.
         *
         * @return KernelSnapshotInfo describing kernel's snapshot capabilities
         *
         * @note Kernels with no data to snapshot should return
         *       KernelSnapshotInfo::passthrough() to explicitly opt out.
         *
         * @example
         * @code
         * KernelSnapshotInfo CPUGemmKernel::getKernelSnapshotInfo() const override {
         *     return KernelSnapshotInfo::gemm()
         *         .withInput("A", "activation matrix [m, k]", KernelBufferDtype::FP32)
         *         .withWeight("B", "weight matrix [k, n]", weight_dtype_)
         *         .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
         *         .withScalar("alpha", "scale factor", KernelBufferDtype::FP32)
         *         .withScalar("beta", "residual factor", KernelBufferDtype::FP32);
         * }
         * @endcode
         */
        virtual KernelSnapshotInfo getKernelSnapshotInfo() const
        {
            // Default implementation returns unspecified - derived classes should override
            return KernelSnapshotInfo::unspecified();
        }
    };

} // namespace llaminar2
