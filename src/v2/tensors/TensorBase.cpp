/**
 * @file TensorBase.cpp
 * @brief TensorBase class implementation (helper methods)
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "../utils/Logger.h"
#include <stdexcept>

namespace llaminar2
{

    void TensorBase::to_fp32_via_blocks(float *dst) const
    {
        // This helper is for quantized tensors that implement IBlockDecoder
        const IBlockDecoder *decoder = dynamic_cast<const IBlockDecoder *>(this);
        if (!decoder)
        {
            throw std::runtime_error("to_fp32_via_blocks() called on non-IBlockDecoder tensor");
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_via_blocks() requires 2D tensor");
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        const size_t block_sz = decoder->block_size();
        const size_t blocks_per_row = (cols + block_sz - 1) / block_sz;

        // Decode each block to the output buffer
        for (size_t row = 0; row < rows; ++row)
        {
            float *row_dst = dst + row * cols;
            for (size_t kb = 0; kb < blocks_per_row; ++kb)
            {
                const size_t offset = kb * block_sz;
                const size_t count = std::min(block_sz, cols - offset);

                // Decode block to temporary buffer
                float block_buffer[256]; // Max block size supported
                decoder->decode_block_at(row, kb, block_buffer);

                // Copy decoded values to output
                for (size_t i = 0; i < count; ++i)
                {
                    row_dst[offset + i] = block_buffer[i];
                }
            }
        }
    }

} // namespace llaminar2
