/**
 * @file CUDAQ8ResidualCorrection.cu
 * @brief Exact-stream native boundary for prepared two-term Q8 projection.
 *
 * The shared implementation owns fixed per-row paired-dot arithmetic and
 * admitted activation scratch. This translation unit selects the native
 * backend quantizer without repacking or replacing immutable source weights.
 */
#define Q8_RESIDUAL_CUDA 1
#include "kernels/common/GPUQ8ResidualCorrection.cuh"
