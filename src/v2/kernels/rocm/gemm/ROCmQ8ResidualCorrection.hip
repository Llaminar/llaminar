/**
 * @file ROCmQ8ResidualCorrection.hip
 * @brief Exact-stream native boundary for prepared two-term Q8 projection.
 *
 * The shared implementation owns fixed per-row paired-dot arithmetic and
 * admitted activation scratch. This translation unit selects the native
 * backend quantizer without repacking or replacing immutable source weights.
 */
#include "kernels/common/GPUQ8ResidualCorrection.cuh"
