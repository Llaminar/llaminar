/**
 * @file Perf__FloatingReduction.hip
 * @brief Release-native timing and isolated profiling of fixed-tree reductions.
 *
 * The shared fixture retains literal historical arithmetic and the production
 * launch bridges. Its opt-in event benchmark isolates one format and geometry;
 * functional certification remains registered separately in preflight.
 */
#include "../../../integration/kernels/rocm/ROCmFloatingReductionRegression.h"
