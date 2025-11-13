/**
 * @file IntegerGemmKernelRegistry.h
 * @brief Runtime dispatch to template-instantiated Integer GEMM kernels
 *
 * This registry provides runtime selection of the optimal template instantiation
 * based on (ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC) parameters.
 *
 * Similar to GemmMicroKernelRegistry but for integer GEMM kernels.
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#pragma once

#include <functional>
#include <map>
#include <tuple>
#include <string>

// Forward declarations from main namespace
namespace llaminar2
{
    struct Q8_0Block;
}

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            // Forward declarations
            class Q8_0BlockProvider;

            // Use Q8_0Block from main namespace
            using ::llaminar2::Q8_0Block;

            /**
             * @brief Integer GEMM kernel function signature
             *
             * Matches IntegerGemmKernel::multiply()
             */
            using IntegerGemmFunc = std::function<bool(
                const Q8_0Block *A_blocks,
                Q8_0BlockProvider &B_provider,
                Q8_0Block *C_blocks,
                int m,
                int n,
                int k)>;

            /**
             * @brief Configuration key for registry lookup
             *
             * (ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC)
             */
            using IntegerGemmKey = std::tuple<std::string, int, int, int, int, int, int, int>;

            /**
             * @brief Integer GEMM kernel registry singleton
             *
             * Provides runtime dispatch to template instantiations.
             */
            class IntegerGemmKernelRegistry
            {
            public:
                /**
                 * @brief Get singleton instance
                 */
                static IntegerGemmKernelRegistry &instance()
                {
                    static IntegerGemmKernelRegistry registry;
                    registry.ensureInitialized();
                    return registry;
                }

                /**
                 * @brief Register an integer GEMM kernel template instantiation
                 *
                 * Called by explicit instantiation files during static initialization.
                 */
                void register_kernel(
                    const std::string &isa,
                    int mr, int nr,
                    int unroll_k,
                    int prefetch_dist,
                    int mc, int kc, int nc,
                    IntegerGemmFunc kernel)
                {
                    IntegerGemmKey key{isa, mr, nr, unroll_k, prefetch_dist, mc, kc, nc};
                    registry_[key] = kernel;
                }

                /**
                 * @brief Get integer GEMM kernel for given configuration
                 *
                 * Returns nullptr if no exact match found.
                 */
                IntegerGemmFunc get_kernel(
                    const std::string &isa,
                    int mr, int nr,
                    int unroll_k,
                    int prefetch_dist,
                    int mc, int kc, int nc) const
                {
                    IntegerGemmKey key{isa, mr, nr, unroll_k, prefetch_dist, mc, kc, nc};
                    auto it = registry_.find(key);
                    if (it != registry_.end())
                    {
                        return it->second;
                    }

                    // No exact match found
                    return nullptr;
                }

                /**
                 * @brief Check if kernel exists for given configuration
                 */
                bool has_kernel(
                    const std::string &isa,
                    int mr, int nr,
                    int unroll_k,
                    int prefetch_dist,
                    int mc, int kc, int nc) const
                {
                    IntegerGemmKey key{isa, mr, nr, unroll_k, prefetch_dist, mc, kc, nc};
                    return registry_.find(key) != registry_.end();
                }

                /**
                 * @brief Get number of registered kernels
                 */
                size_t size() const
                {
                    return registry_.size();
                }

            private:
                IntegerGemmKernelRegistry() = default;

                void ensureInitialized()
                {
                    if (!initialized_)
                    {
                        extern void ensureIntegerGemmKernelsRegistered();
                        ensureIntegerGemmKernelsRegistered();
                        initialized_ = true;
                    }
                }

                std::map<IntegerGemmKey, IntegerGemmFunc> registry_;
                bool initialized_ = false;
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
