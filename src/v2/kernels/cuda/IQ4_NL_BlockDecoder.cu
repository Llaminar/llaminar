/**
 * @file IQ4_NL_BlockDecoder.cu
 * @brief IQ4_NL constant memory definitions
 *
 * Contains constant memory lookup tables used by IQ4_NL decoder.
 * This file must be compiled with NVCC.
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#include "IQ4_NL_BlockDecoder.h"

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief IQ4_NL lookup table in constant memory
         *
         * Maps 4-bit indices (0-15) to quantized values (-127 to 113).
         * Provides non-linear quantization for improved accuracy.
         *
         * Constant memory is:
         * - Cached per SM (fast repeated access)
         * - Broadcast to all threads in a warp
         * - Ideal for read-only lookup tables
         */
        __constant__ int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};

    } // namespace cuda
} // namespace llaminar2
