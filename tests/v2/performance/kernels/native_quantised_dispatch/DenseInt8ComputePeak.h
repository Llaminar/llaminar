/**
 * @file DenseInt8ComputePeak.h
 * @brief Dense INT8 matrix-compute denominators for matched GPU measurements.
 *
 * Useful GEMM work counts a multiply and an add as two operations. NVIDIA's
 * sparse headline rates are not dense rates, and DP4A capacity is not tensor
 * capacity. The caller supplies the native runtime's peak clock and actual
 * enabled SM/CU count; instantaneous clocks remain separate telemetry.
 */
#pragma once

#include <cstdint>

namespace llaminar2::test::compute_peak
{
/**
 * @brief Return dense INT8 tensor operations per SM clock for admitted CUDA ISAs.
 * @param capability Decimal compute capability, for example 86 for sm_86.
 * @return Dense operations per SM clock, or zero for an unqualified architecture.
 *
 * NVIDIA's Turing/GA102, A100, Ada and H100 architecture whitepapers own these
 * rates. GA102/Ada have 2048; A100 has 4096 and Hopper doubles A100 to 8192.
 * Sparsity doubles their headline throughput without changing dense capacity.
 */
[[nodiscard]] constexpr std::uint32_t cudaDenseInt8OpsPerSmClock(int capability) noexcept
{
    switch (capability)
    {
    case 75:
    case 86:
    case 89: return 2048;
    case 80: return 4096;
    case 90: return 8192;
    default: return 0;
    }
}

/**
 * @brief Convert authenticated compute-unit capacity and native peak clock to TOPS.
 * @param units Actual enabled compute units (SMs or CUs).
 * @param clock_khz Native runtime peak multiprocessor frequency in kilohertz.
 * @param dense_ops_per_unit_clock Qualified dense operations per unit per clock.
 * @return Dense tera-operations/second, or zero for incomplete/unqualified inputs.
 */
[[nodiscard]] constexpr double denseInt8Tops(
    int units, int clock_khz, std::uint32_t dense_ops_per_unit_clock) noexcept
{
    return units > 0 && clock_khz > 0 && dense_ops_per_unit_clock > 0
        ? static_cast<double>(units) * clock_khz * dense_ops_per_unit_clock / 1.0e9 : 0.0;
}
}
