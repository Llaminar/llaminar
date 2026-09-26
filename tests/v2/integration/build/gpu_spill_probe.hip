/**
 * @file gpu_spill_probe.hip
 * @brief Compile the shared spill witnesses through the real HIP toolchain.
 *
 * This object is never linked or launched. Its compiler remarks and final code
 * metadata exercise the same mandatory guard as production HIP objects.
 */
#include "GpuSpillProbe.h"
