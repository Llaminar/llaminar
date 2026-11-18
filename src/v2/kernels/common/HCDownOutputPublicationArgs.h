/**
 * @file HCDownOutputPublicationArgs.h
 * @brief Original HC reducer epilogue destinations, with independent publisher rows.
 *
 * The native projection keeps its original row work. This additional row owner
 * gates only the existing /4-SiLU and two-term Q8 publication, including count0.
 * Every pointer borrows an already admitted tensor or named up workspace bank.
 */
#pragma once
#include "DeviceRowRange.h"
namespace llaminar2 {
/** @brief Explicit original HC-down producer program; only its named publication purpose may opt in. */
enum class HCDownProducerProgram : unsigned char { Original, TwoK32Lookahead };
/** @brief Immutable device bindings for one original HC bottleneck publication. */
struct GPUHCDownOutputView final {
 DeviceRowRange rows; ///< Original publisher authority, independent of projection work.
 int columns; ///< Exact bottleneck width, made of complete32-column blocks.
 float* activated; ///< Original public activated output; exact raw alias is allowed.
 signed char *primary,*residual; ///< Existing original up activation banks.
 float *primaryScales,*residualScales; ///< Existing original scale banks.
};
}
