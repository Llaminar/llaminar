/**
 * @file GemmMicroKernelInit.h
 * @brief Initialization function for MicroKernel registry
 * @author David Sanftenberg
 */

#pragma once

namespace llaminar2 {
namespace kernels {
namespace gemm {

/**
 * @brief Ensure all MicroKernel template instantiations are registered
 *
 * Call this function before using MicroKernelRegistry::instance() to ensure
 * all constructor-attributed registration functions have executed.
 *
 * This is necessary because instantiations are in a static library (.a),
 * and their constructors don't run unless symbols from those objects are
 * explicitly referenced by the final executable.
 */
void ensureMicroKernelsRegistered();

} // namespace gemm
} // namespace kernels
} // namespace llaminar2
