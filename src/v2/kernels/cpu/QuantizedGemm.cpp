/**
 * @file QuantizedGemm.cpp
 * @brief Factory function for creating auto-tuned quantized GEMM kernels
 *
 * This file provides a simple factory that creates ITensorGemm instances
 * using the GemmAutoTuner infrastructure with the MicroKernelRegistry system.
 *
 * @author David Sanftenberg
 */

#include "QuantizedGemm.h"
#include "GemmAutoTuner.h"
#include "GemmMicroKernelAdapter.h"
#include <mutex>

namespace llaminar2
{
    /**
     * @brief Wrapper kernel that delegates to auto-tuner
     */
    class AutoTunedQuantizedGemm : public ITensorGemm
    {
    public:
        explicit AutoTunedQuantizedGemm(const IBlockDecoder *decoder)
            : decoder_(decoder)
        {
            ensureVariantsRegistered();
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!decoder_)
            {
                return false;
            }

            // Validate dimensions
            int expected_cols = transpose_B ? k : n;
            if (static_cast<int>(decoder_->decoder_cols()) != expected_cols)
            {
                return false;
            }

            // Use auto-tuner to select optimal variant
            auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
            auto *optimal = tuner.getOptimalKernel(m, n, k);

            if (!optimal)
            {
                return false;
            }

            // Delegate to auto-selected variant
            return optimal->multiply(A, C, m, n, k, decoder_, alpha, beta);
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

    private:
        const IBlockDecoder *decoder_;

        void ensureVariantsRegistered()
        {
            static bool registered = false;
            static std::mutex registration_mutex;

            if (!registered)
            {
                std::lock_guard<std::mutex> lock(registration_mutex);
                if (!registered)
                {
                    auto variants = kernels::gemm::registerMicroKernelVariants(decoder_);
                    auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();

                    for (auto &variant : variants)
                    {
                        tuner.registerVariant(std::move(variant));
                    }

                    registered = true;
                }
            }
        }
    };

    std::unique_ptr<ITensorGemm> createQuantizedGemm(const IBlockDecoder *decoder)
    {
        return std::make_unique<AutoTunedQuantizedGemm>(decoder);
    }

} // namespace llaminar2
