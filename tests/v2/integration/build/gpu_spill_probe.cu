/**
 * @file gpu_spill_probe.cu
 * @brief Compile the shared spill witnesses through the real CUDA toolchain.
 *
 * This object is never linked or launched. Its native ptxas outcome is the
 * contract being tested; the common header keeps both backends symmetric.
 */
#include "GpuSpillProbe.h"
