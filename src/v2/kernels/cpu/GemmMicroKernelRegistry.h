/**
 * @file GemmMicroKernelRegistry.h
 * @brief Runtime dispatch to template-instantiated micro-kernels
 *
 * This registry provides runtime selection of the optimal template instantiation
 * based on (ISA, MR, NR, UNROLL_K, PREFETCH_DIST) parameters.
 *
 * The auto-tuner calls get_micro_kernel() with desired parameters, and the
 * registry returns a function pointer to the pre-compiled template instantiation.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <functional>
#include <map>
#include <tuple>
#include <string>

namespace llaminar2 {
namespace kernels {
namespace gemm {

/**
 * @brief Micro-kernel function signature
 *
 * Parameters match MicroKernelTemplate::micro_kernel()
 */
using MicroKernelFunc = std::function<void(
    const float* A_panel,    // MR × k_panel
    const float* B_panel,    // NR × k_panel
    float* C,                // MR × NR with ldc stride
    int ldc,                 // Leading dimension of C
    int k_panel,             // K-dimension of panels
    float alpha,             // A*B scaling
    float beta,              // C scaling
    int mr,                  // Actual rows (≤ MR)
    int nr                   // Actual cols (≤ NR)
)>;

/**
 * @brief Pack A panel function signature
 */
using PackAPanelFunc = std::function<void(
    const float* A,
    float* A_packed,
    int m_panel,
    int k_panel,
    int lda
)>;

/**
 * @brief Pack B panel function signature
 */
using PackBPanelFunc = std::function<void(
    const float* B_decoded,
    float* B_packed,
    int k_panel,
    int n_panel,
    int ldb
)>;

/**
 * @brief Configuration key for registry lookup
 *
 * (ISA, MR, NR, UNROLL_K, PREFETCH_DIST)
 */
using MicroKernelKey = std::tuple<std::string, int, int, int, int>;

/**
 * @brief Micro-kernel bundle (compute + packing)
 */
struct MicroKernelBundle {
    MicroKernelFunc micro_kernel;
    PackAPanelFunc pack_A;
    PackBPanelFunc pack_B;
    
    MicroKernelBundle() = default;
    
    MicroKernelBundle(MicroKernelFunc mk, PackAPanelFunc pa, PackBPanelFunc pb)
        : micro_kernel(mk), pack_A(pa), pack_B(pb) {}
};

/**
 * @brief Micro-kernel registry singleton
 *
 * Provides runtime dispatch to template instantiations.
 */
class MicroKernelRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static MicroKernelRegistry& instance() {
        static MicroKernelRegistry registry;
        registry.ensureInitialized();
        return registry;
    }

    /**
     * @brief Register a micro-kernel template instantiation
     *
     * Called by explicit instantiation files during static initialization.
     */
    void register_kernel(
        const std::string& isa,
        int mr, int nr,
        int unroll_k,
        int prefetch_dist,
        MicroKernelBundle bundle)
    {
        MicroKernelKey key{isa, mr, nr, unroll_k, prefetch_dist};
        registry_[key] = bundle;
    }

    /**
     * @brief Get micro-kernel bundle for given configuration
     *
     * Returns nullptr bundle if no exact match found.
     */
    MicroKernelBundle get_kernel(
        const std::string& isa,
        int mr, int nr,
        int unroll_k,
        int prefetch_dist) const
    {
        MicroKernelKey key{isa, mr, nr, unroll_k, prefetch_dist};
        auto it = registry_.find(key);
        if (it != registry_.end()) {
            return it->second;
        }
        
        // No exact match found - return empty bundle
        return MicroKernelBundle{};
    }

    /**
     * @brief Check if kernel exists for given configuration
     */
    bool has_kernel(
        const std::string& isa,
        int mr, int nr,
        int unroll_k,
        int prefetch_dist) const
    {
        MicroKernelKey key{isa, mr, nr, unroll_k, prefetch_dist};
        return registry_.find(key) != registry_.end();
    }

    /**
     * @brief Get number of registered kernels
     */
    size_t size() const {
        return registry_.size();
    }

private:
    MicroKernelRegistry() = default;
    
    void ensureInitialized() {
        if (!initialized_) {
            extern void ensureMicroKernelsRegistered();
            ensureMicroKernelsRegistered();
            initialized_ = true;
        }
    }
    
    std::map<MicroKernelKey, MicroKernelBundle> registry_;
    bool initialized_ = false;
};

/**
 * @brief Helper macro for registering template instantiations
 *
 * Usage in generated files:
 *   REGISTER_MICROKERNEL(simd::AVX512Tag, 8, 6, 4, 2);
 */
#define REGISTER_MICROKERNEL(ISA, MR, NR, UNROLL_K, PREFETCH_DIST) \
    namespace { \
        struct Registrar_##ISA##_##MR##_##NR##_##UNROLL_K##_##PREFETCH_DIST { \
            Registrar_##ISA##_##MR##_##NR##_##UNROLL_K##_##PREFETCH_DIST() { \
                using KernelType = MicroKernelTemplate<ISA, MR, NR, UNROLL_K, PREFETCH_DIST>; \
                MicroKernelRegistry::instance().register_kernel( \
                    #ISA, MR, NR, UNROLL_K, PREFETCH_DIST, \
                    MicroKernelBundle{ \
                        &KernelType::micro_kernel, \
                        &KernelType::pack_A_panel, \
                        &KernelType::pack_B_panel \
                    } \
                ); \
            } \
        }; \
        static Registrar_##ISA##_##MR##_##NR##_##UNROLL_K##_##PREFETCH_DIST \
            registrar_##ISA##_##MR##_##NR##_##UNROLL_K##_##PREFETCH_DIST; \
    }

} // namespace gemm
} // namespace kernels
} // namespace llaminar2
