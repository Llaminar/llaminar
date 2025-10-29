/**
 * @file GemmMicroKernelAdapter.h
 * @brief Adapter integrating micro-kernel with auto-tuner via cache blocking
 *
 * Architecture:
 * - GemmAutoTuner: Benchmarks 1,225 variants, selects best
 * - GemmMicroKernelAdapter: Maps GemmAutoTuner selection → actual micro-kernel
 * - MicroKernelRegistry: Stores all instantiated micro-kernels
 *
 * @author David Sanftenberg
 */

#pragma once

#include "GemmMicroKernelRegistry.h"
#include "GemmAutoTuner.h"
#include "../../tensors/TensorKernels.h"
#include <string>
#include <sstream>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Adapter that wraps a MicroKernelBundle as IQuantizedGemmVariant
             *
             * This allows the auto-tuner to use pre-compiled template instantiations
             * from the MicroKernelRegistry instead of generating code at runtime.
             *
             * Performance Optimizations:
             * - Cache-level blocking: Large tiles (up to 256×256) for L1/L2/L3 cache locality
             * - OpenMP parallelization: Multi-threaded execution across cache tiles
             * - Cached packing buffers (thread-local, avoid repeated allocation)
             * - Fused decode + packing for B matrix (single pass)
             *
             * Blocking Strategy:
             * - Outer loop: Cache tiles (MC × NC) - parallelized with OpenMP
             * - Inner loop: Micro-tiles (MR × NR) - register blocking
             */
            class MicroKernelVariantAdapter : public llaminar::v2::kernels::IQuantizedGemmVariant
            {
            public:
                /**
                 * @brief Construct adapter for a specific microkernel configuration
                 *
                 * @param isa_name ISA tag name (e.g., "simd::AVX512Tag")
                 * @param mr M register blocking (rows per micro-tile)
                 * @param nr N register blocking (cols per micro-tile)
                 * @param unroll_k K loop unroll factor
                 * @param prefetch_dist Prefetch distance in iterations
                 * @param bundle Microkernel function pointers
                 * @param decoder IBlockDecoder for quantized weight access
                 */
                MicroKernelVariantAdapter(
                    const std::string &isa_name,
                    int mr, int nr, int unroll_k, int prefetch_dist,
                    const MicroKernelBundle &bundle,
                    const llaminar2::IBlockDecoder *decoder)
                    : isa_name_(isa_name),
                      mr_(mr),
                      nr_(nr),
                      unroll_k_(unroll_k),
                      prefetch_dist_(prefetch_dist),
                      bundle_(bundle),
                      decoder_(decoder)
                {
                    // Build human-readable name
                    std::ostringstream oss;
                    oss << "MicroKernel_" << isa_name << "_"
                        << mr << "x" << nr << "_u" << unroll_k << "_p" << prefetch_dist;
                    name_ = oss.str();

                    // Select cache tile sizes based on micro-kernel dimensions
                    // Larger micro-kernels can use larger cache tiles
                    if (mr * nr >= 32)
                    {
                        // Large micro-kernel (e.g., 8×6): Use large cache tiles
                        mc_ = 256;
                        nc_ = 256;
                    }
                    else if (mr * nr >= 16)
                    {
                        // Medium micro-kernel (e.g., 4×4, 8×2): Use medium cache tiles
                        mc_ = 128;
                        nc_ = 128;
                    }
                    else
                    {
                        // Small micro-kernel (e.g., 1×4, 2×2): Use small cache tiles
                        mc_ = 64;
                        nc_ = 64;
                    }
                }

                const char *name() const override
                {
                    return name_.c_str();
                }

                bool multiply(
                    const float *A,
                    float *C,
                    int m, int n, int k,
                    const llaminar2::IBlockDecoder *decoder,
                    float alpha = 1.0f,
                    float beta = 0.0f) override
                {
                    const size_t block_size = decoder->block_size();
                    const size_t num_k_blocks = (k + block_size - 1) / block_size;

// Parallel loop over N dimension (columns) with cache blocking
#pragma omp parallel
                    {
                        // Thread-local buffers (allocated once per thread)
                        // NOTE: Add generous padding for SIMD over-reads in the micro-kernel
                        // The micro-kernel accesses B_panel[j * k + offset] where j ∈ [0, NR)
                        // With packed layout, we need padding at the end of EACH micro-panel row
                        // Conservative approach: Add enough for worst-case SIMD over-reads
                        constexpr size_t CACHE_LINE = 64; // 64 bytes = 16 floats
                        const size_t NR = nr_;            // Number of columns in micro-panel
                        const size_t MR = mr_;            // Number of rows in micro-panel

                        const size_t a_size = mc_ * k;
                        const size_t b_size = k * nc_;

                        // Very conservative padding: nc_ cache lines for B (one per potential row),
                        // mc_ cache lines for A, plus 10% margin
                        // Minimum: 2 cache lines (32 floats) for SIMD over-read safety
                        const size_t min_padding = 2 * CACHE_LINE / sizeof(float);
                        const size_t a_padding = std::max(min_padding, mc_ * CACHE_LINE / sizeof(float) + (a_size / 10));
                        const size_t b_padding = std::max(min_padding, nc_ * CACHE_LINE / sizeof(float) + (b_size / 10));

                        std::vector<float> A_packed_local(a_size + a_padding);
                        std::vector<float> B_decoded_local(b_size + b_padding);
                        std::vector<float> B_packed_local(b_size + b_padding);

#pragma omp for schedule(dynamic)
                        for (int jc = 0; jc < n; jc += nc_)
                        {
                            int nc = std::min(nc_, n - jc);

                            // Step 1: Decode B matrix for this NC panel
                            // Decode all columns [jc, jc+nc) across all K blocks
                            for (size_t kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * block_size;
                                size_t k_end = std::min(k_start + block_size, static_cast<size_t>(k));

                                for (int jj = 0; jj < nc; ++jj)
                                {
                                    float block_data[256]; // Max block size
                                    decoder->decode_block_at(jc + jj, kb, block_data);

                                    // Store decoded values in row-major layout
                                    for (size_t kk = k_start; kk < k_end; ++kk)
                                    {
                                        B_decoded_local[jj * k + kk] = block_data[kk - k_start];
                                    }
                                }
                            }

                            // Step 2: Pack B panel using micro-kernel packing function
                            // This ensures the layout matches what micro_kernel expects
                            for (int j = 0; j < nc; j += nr_)
                            {
                                int nb = std::min(nr_, nc - j);
                                // Pack B: note ldb=k because B_decoded is row-major with stride k
                                bundle_.pack_B(B_decoded_local.data() + j * k,
                                               B_packed_local.data() + j * k,
                                               k, nb, k);
                            }

                            // Loop over M dimension with cache blocking
                            for (int ic = 0; ic < m; ic += mc_)
                            {
                                int mc = std::min(mc_, m - ic);

                                // Pack A panel: MC × K (using micro-kernel packing)
                                for (int i = 0; i < mc; i += mr_)
                                {
                                    int mb = std::min(mr_, mc - i);
                                    bundle_.pack_A(A + (ic + i) * k, A_packed_local.data() + i * k, mb, k, k);
                                }

                                // Process MC × NC cache tile with MR × NR micro-tiles
                                for (int ir = 0; ir < mc; ir += mr_)
                                {
                                    int mb = std::min(mr_, mc - ir);

                                    for (int jr = 0; jr < nc; jr += nr_)
                                    {
                                        int nb = std::min(nr_, nc - jr);

                                        // Call microkernel to compute tile
                                        bundle_.micro_kernel(
                                            A_packed_local.data() + ir * k, // Packed A panel
                                            B_packed_local.data() + jr * k, // Packed B panel
                                            C + (ic + ir) * n + (jc + jr),  // Output tile location
                                            n,                              // Leading dimension of C
                                            k,                              // K dimension
                                            alpha,                          // A*B scaling
                                            beta,                           // C scaling
                                            mb,                             // Actual rows
                                            nb);                            // Actual cols
                                    }
                                }
                            }
                        }
                    }

                    return true;
                }

            private:
                llaminar::v2::kernels::GemmKernelConfig config() const override
                {
                    llaminar::v2::kernels::GemmKernelConfig cfg;
                    cfg.unroll_factor = unroll_k_;
                    cfg.prefetch_blocks = prefetch_dist_;
                    cfg.tile_m = mr_;
                    cfg.tile_n = nr_;
                    return cfg;
                }

                std::string isa_name_;
                int mr_;
                int nr_;
                int unroll_k_;
                int prefetch_dist_;
                MicroKernelBundle bundle_;
                const llaminar2::IBlockDecoder *decoder_;
                std::string name_;

                // Cache blocking parameters (match L1Opt for optimal performance)
                int mc_; // M dimension cache tile size
                int nc_; // N dimension cache tile size
            };

            /**
             * @brief Register all microkernels from registry with auto-tuner
             *
             * Queries MicroKernelRegistry for all available pre-compiled
             * instantiations and creates adapter wrappers for auto-tuner use.
             *
             * @param decoder IBlockDecoder for quantized weight access
             * @return Vector of variant adapters (up to 1,225)
             */
            inline std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
            registerMicroKernelVariants(const llaminar2::IBlockDecoder *decoder)
            {
                std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>> variants;

                auto &registry = MicroKernelRegistry::instance();

                // Enumerate all registered kernels
                // Parameter ranges from generate_microkernel_instantiations.py:
                // ISA: {AVX512Tag, AVX2Tag}
                // MR: {1, 2, 4, 8, 16, 32, 64}
                // NR: {1, 2, 4, 6, 8, 16, 32, 64}
                // UNROLL_K: {1, 2, 4, 8, 16}
                // PREFETCH_DIST: {0, 1, 2, 3, 5}
                // Filtered by register constraints (MR*NR ≤ 48 for AVX512, ≤32 for AVX2)

                const std::vector<std::string> isa_names = {"simd::AVX512Tag", "simd::AVX2Tag"};
                const std::vector<int> mr_values = {1, 2, 4, 8, 16, 32, 64};
                const std::vector<int> nr_values = {1, 2, 4, 6, 8, 16, 32, 64};
                const std::vector<int> unroll_values = {1, 2, 4, 8, 16};
                const std::vector<int> prefetch_values = {0, 1, 2, 3, 5};

                // Probe all combinations (registry will filter invalid ones)
                for (const auto &isa_name : isa_names)
                {
                    for (int mr : mr_values)
                    {
                        for (int nr : nr_values)
                        {
                            // Apply same register constraints as generator
                            if (isa_name == "simd::AVX512Tag" && mr * nr > 48)
                                continue;
                            if (isa_name == "simd::AVX2Tag" && mr * nr > 32)
                                continue;

                            for (int unroll_k : unroll_values)
                            {
                                for (int prefetch : prefetch_values)
                                {
                                    if (registry.has_kernel(isa_name, mr, nr, unroll_k, prefetch))
                                    {
                                        auto bundle = registry.get_kernel(isa_name, mr, nr, unroll_k, prefetch);

                                        variants.push_back(
                                            std::make_unique<MicroKernelVariantAdapter>(
                                                isa_name, mr, nr, unroll_k, prefetch,
                                                bundle, decoder));
                                    }
                                }
                            }
                        }
                    }
                }

                return variants;
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
