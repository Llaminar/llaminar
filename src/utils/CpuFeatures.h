// CpuFeatures.h - CPU instruction set detection for optimal backend selection
#pragma once

#include <string>

namespace llaminar
{

    /**
     * @brief Detects CPU instruction set features relevant to quantized operations
     *
     * Uses CPUID instruction to detect available SIMD extensions. Results are
     * cached after first call for performance.
     */
    class CpuFeatures
    {
    public:
        /**
         * @brief Get singleton instance with cached feature detection
         */
        static const CpuFeatures &instance();

        /**
         * @brief Check if CPU supports AVX512_BF16 instructions
         *
         * AVX512_BF16 provides native BF16 arithmetic (vdpbf16ps instruction).
         * Required for efficient cblas_sbgemm on large matrices.
         *
         * Present on: Cooper Lake (2020), Ice Lake (2021), Sapphire Rapids (2023)
         * Missing on: Cascade Lake, earlier generations
         */
        bool has_avx512_bf16() const { return avx512_bf16_; }

        /**
         * @brief Check if CPU supports AVX512_FP16 instructions
         *
         * AVX512_FP16 provides native FP16 arithmetic (half-precision float).
         *
         * Present on: Sapphire Rapids (2023+)
         */
        bool has_avx512_fp16() const { return avx512_fp16_; }

        /**
         * @brief Check if CPU supports AVX512_VNNI instructions
         *
         * AVX512_VNNI provides INT8/INT16 dot product acceleration.
         * Useful for INT8 quantization, not for FP16/BF16.
         *
         * Present on: Cascade Lake (2019+), Ice Lake, later generations
         */
        bool has_avx512_vnni() const { return avx512_vnni_; }

        /**
         * @brief Check if CPU supports F16C instructions
         *
         * F16C provides FP32↔FP16 conversion (vcvtph2ps, vcvtps2ph).
         * Does NOT provide FP16 arithmetic, only conversion.
         *
         * Present on: Ivy Bridge (2012+), all modern CPUs
         */
        bool has_f16c() const { return f16c_; }

        /**
         * @brief Check if CPU supports AVX512F (foundation)
         */
        bool has_avx512f() const { return avx512f_; }

        /**
         * @brief Check if CPU supports AMX-BF16 instructions
         *
         * AMX (Advanced Matrix Extensions) provides tile-based BF16 matrix multiplication.
         * Much faster than AVX512_BF16 for large matrices (hardware accelerated).
         *
         * Present on: Ice Lake server (2021+), Sapphire Rapids (2023+)
         * Missing on: Cascade Lake, Cooper Lake
         */
        bool has_amx_bf16() const { return amx_bf16_; }

        /**
         * @brief Get human-readable CPU feature summary
         */
        std::string summary() const;

    private:
        CpuFeatures(); // Private constructor for singleton

        void detect_features();

        bool avx512_bf16_ = false;
        bool avx512_fp16_ = false;
        bool avx512_vnni_ = false;
        bool f16c_ = false;
        bool avx512f_ = false;
        bool amx_bf16_ = false;
    };

    /**
     * @brief Convenience function to check if native BF16 GEMM is safe to use
     *
     * Returns true only if CPU has hardware BF16 support (avx512_bf16).
     * If false, caller should use BF16→FP32 expansion instead of cblas_sbgemm.
     */
    inline bool can_use_native_bf16_gemm()
    {
        return CpuFeatures::instance().has_avx512_bf16();
    }

} // namespace llaminar
