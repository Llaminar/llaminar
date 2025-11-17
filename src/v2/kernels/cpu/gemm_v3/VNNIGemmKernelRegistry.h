/**
 * @file VNNIGemmKernelRegistry.h
 * @brief Runtime registry for VNNI GEMM kernel template instantiations (gemm_v3)
 * @author David Sanftenberg
 *
 * This registry enables runtime dispatch to compile-time specialized VNNI GEMM kernels
 * without virtual function overhead. Follows the same pattern as Q8_1GemmKernelRegistry.
 *
 * Design Philosophy:
 * - Pre-packed panel layout (A: 4x4-grouped, B: column-major K-contiguous)
 * - Maximize time in _mm512_dpbusd_epi32 inner loop
 * - Minimize on-the-fly format conversion overhead
 *
 * Configuration Space:
 * - M_R: 8, 16, 32, 64 (micro-kernel M dimension, must be multiple of 4)
 * - N_R: 16, 32, 64, 128 (micro-kernel N dimension, must be multiple of 16)
 * - K_BLK: 32, 64, 128 (K block size, must be multiple of 4)
 * - UNROLL_K: 1, 2, 4 (K-loop unroll factor)
 * - PREFETCH_B_L1: 0, 64, 128 (L1 prefetch distance in bytes)
 *
 * Total: ~200 valid configurations (after constraint filtering)
 */

#pragma once

#include <functional>
#include <map>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace llaminar2
{
    class IActivationTensor;
    class Q8_0Tensor;
}

namespace llaminar2
{

    /**
     * @brief Function signature for VNNI GEMM kernel with pre-packed panels
     *
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A / rows in B
     * @param A Activation tensor (will be packed internally to 4x4-grouped layout)
     * @param B Q8_0 weight tensor (will be packed internally to column-major K-contiguous)
     * @param C Output FP32 matrix
     * @param ldc Leading dimension of C
     */
    using VNNIGemmFunc = std::function<void(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        float *C, int ldc)>;

    /**
     * @brief Registry key: 5-tuple uniquely identifying a VNNI GEMM configuration
     *
     * Tuple elements:
     * 0: M_R (micro-kernel M dimension, multiple of 4)
     * 1: N_R (micro-kernel N dimension, multiple of 16)
     * 2: K_BLK (K block size, multiple of 4)
     * 3: UNROLL_K (K-loop unroll factor)
     * 4: PREFETCH_B_L1 (L1 prefetch distance in bytes)
     */
    using VNNIGemmKey = std::tuple<int, int, int, int, int>;

    /**
     * @brief Singleton registry for VNNI GEMM kernel instantiations
     *
     * Thread-safe singleton pattern with lazy initialization.
     * Auto-populated by __attribute__((constructor)) functions in generated files.
     */
    class VNNIGemmKernelRegistry
    {
    public:
        /**
         * @brief Get singleton instance
         * @return Reference to global registry
         */
        static VNNIGemmKernelRegistry &instance()
        {
            static VNNIGemmKernelRegistry registry;
            registry.ensureInitialized();
            return registry;
        }

        /**
         * @brief Register a kernel template instantiation
         *
         * Called by __attribute__((constructor)) functions in generated files.
         *
         * @param m_r Micro-kernel M dimension (multiple of 4)
         * @param n_r Micro-kernel N dimension (multiple of 16)
         * @param k_blk K block size (multiple of 4)
         * @param unroll_k K-loop unroll factor
         * @param prefetch_b_l1 L1 prefetch distance in bytes
         * @param func Kernel function pointer
         */
        void register_kernel(
            int m_r, int n_r, int k_blk, int unroll_k, int prefetch_b_l1,
            VNNIGemmFunc func)
        {
            VNNIGemmKey key = std::make_tuple(m_r, n_r, k_blk, unroll_k, prefetch_b_l1);
            kernels_[key] = func;
            initialized_ = true;
        }

        /**
         * @brief Lookup kernel by configuration parameters
         *
         * @param m_r Micro-kernel M dimension
         * @param n_r Micro-kernel N dimension
         * @param k_blk K block size
         * @param unroll_k K-loop unroll factor
         * @param prefetch_b_l1 L1 prefetch distance
         * @return VNNIGemmFunc Kernel function, or nullptr if not found
         */
        VNNIGemmFunc get_kernel(int m_r, int n_r, int k_blk, int unroll_k, int prefetch_b_l1) const
        {
            VNNIGemmKey key = std::make_tuple(m_r, n_r, k_blk, unroll_k, prefetch_b_l1);
            auto it = kernels_.find(key);
            if (it != kernels_.end())
            {
                return it->second;
            }
            return nullptr;
        }

        /**
         * @brief Get number of registered kernels
         * @return size_t Number of configurations in registry
         */
        size_t size() const
        {
            return kernels_.size();
        }

        /**
         * @brief Check if registry has been initialized
         * @return true if at least one kernel is registered
         */
        bool is_initialized() const
        {
            return initialized_;
        }

        /**
         * @brief Print all registered kernels (for debugging)
         */
        void print_registered_kernels() const
        {
            std::cout << "VNNI GEMM Kernel Registry (" << size() << " configurations):\n";
            for (const auto &[key, func] : kernels_)
            {
                auto [m_r, n_r, k_blk, unroll_k, prefetch] = key;
                std::cout << "  M_R=" << m_r << ", N_R=" << n_r
                          << ", K_BLK=" << k_blk << ", UNROLL_K=" << unroll_k
                          << ", PREFETCH_B_L1=" << prefetch << "\n";
            }
        }

    private:
        VNNIGemmKernelRegistry() = default;
        ~VNNIGemmKernelRegistry() = default;
        VNNIGemmKernelRegistry(const VNNIGemmKernelRegistry &) = delete;
        VNNIGemmKernelRegistry &operator=(const VNNIGemmKernelRegistry &) = delete;

        void ensureInitialized()
        {
            if (!initialized_)
            {
                extern void forceLink_VNNIGemmKernelRegistry();
                forceLink_VNNIGemmKernelRegistry();
                initialized_ = true;
            }
        }

        std::map<VNNIGemmKey, VNNIGemmFunc> kernels_;
        bool initialized_ = false;
    };

} // namespace llaminar2
