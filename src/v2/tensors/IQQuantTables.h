#pragma once

#include <cstdint>

/**
 * @file IQQuantTables.h
 * @brief Lookup tables for IQ4_NL quantization
 * @author David Sanftenberg
 *
 * IQ4_NL uses a 16-entry non-linear lookup table (kvalues_iq4nl).
 * Each 4-bit index maps to an optimized float value, providing
 * better accuracy than uniform 4-bit quantization.
 */

namespace llaminar2 {

/**
 * @brief IQ4_NL 16-value lookup table
 *
 * Optimized via k-means clustering on typical LLM weight distributions.
 * Provides ~7.1× compression vs FP32 with minimal accuracy loss.
 *
 * Source: GGML's block_iq4_nl implementation
 */
static constexpr float kvalues_iq4nl[16] = {
    -127.0f / 127.0f,  // Index 0
     -97.0f / 127.0f,  // Index 1
     -67.0f / 127.0f,  // Index 2
     -38.0f / 127.0f,  // Index 3
      -9.0f / 127.0f,  // Index 4
      20.0f / 127.0f,  // Index 5
      49.0f / 127.0f,  // Index 6
      78.0f / 127.0f,  // Index 7
     106.0f / 127.0f,  // Index 8 (note: gap in sequence)
     -65.0f / 127.0f,  // Index 9
     -42.0f / 127.0f,  // Index 10
     -15.0f / 127.0f,  // Index 11
      11.0f / 127.0f,  // Index 12
      34.0f / 127.0f,  // Index 13
      63.0f / 127.0f,  // Index 14
      92.0f / 127.0f   // Index 15
};

} // namespace llaminar2
