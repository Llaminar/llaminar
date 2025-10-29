/**
 * @file Q8_0Tensor.cpp
 * @brief Q8_0 quantized tensor implementation (8-bit per weight, 4.0x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/GemmAutoTuner.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    Q8_0Tensor::Q8_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_0Tensor: shape cannot be empty");
        }

        // Calculate total elements
        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        // Each Q8_0Block contains 32 elements
        size_t n_blocks = (n_elems + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q8_0Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q8_0Tensor: insufficient raw data (" +
                                        std::to_string(raw_data_.size()) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    Q8_0Tensor::~Q8_0Tensor()
    {
        // Destructor
    }

    bool Q8_0Tensor::set_device(int device_idx)
    {
        // TODO: Implement device transfer
        device_idx_ = device_idx;
        return true;
    }

    const float *Q8_0Tensor::data() const
    {
        // Dequantize to temp cache
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // Decode all blocks (parallelized for large tensors)
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

#pragma omp parallel for collapse(2) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const Q8_0Block &block = blocks[r * blocks_per_row + b];
                    decodeBlock(block, &dequant_cache_[r * shape_[1] + b * Q8_0Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q8_0Tensor::mutable_data()
    {
        throw std::runtime_error("Q8_0Tensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<ITensorRoPE> Q8_0Tensor::createRoPE()
    {
        throw std::runtime_error("Q8_0Tensor: RoPE not supported on quantized tensors");
    }

    std::unique_ptr<ITensorSwiGLU> Q8_0Tensor::createSwiGLU()
    {
        throw std::runtime_error("Q8_0Tensor: SwiGLU not supported on quantized tensors");
    }

    std::unique_ptr<ITensorSoftmax> Q8_0Tensor::createSoftmax()
    {
        throw std::runtime_error("Q8_0Tensor: Softmax not supported on quantized tensors");
    }

    std::unique_ptr<ITensorRMSNorm> Q8_0Tensor::createRMSNorm()
    {
        throw std::runtime_error("Q8_0Tensor: RMSNorm not supported on quantized tensors");
    }

    std::unique_ptr<ITensorGemm> Q8_0Tensor::createGemm()
    {
        // Use generic QuantizedGemmKernel with IBlockDecoder interface
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void Q8_0Tensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(raw_data_.data());
        const Q8_0Block &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q8_0Tensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q8_0Tensor::decodeBlock(const Q8_0Block &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback
        const float scale = fp16_to_fp32(block.d);
        for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
        {
            output[i] = scale * static_cast<float>(block.qs[i]);
        }
#endif
    }

#if defined(__AVX512F__)
    void Q8_0Tensor::decodeBlockAVX512(const Q8_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m512 vscale = _mm512_set1_ps(scale);

        // Process 16 int8 values at a time (2 iterations for 32 elements)
        for (size_t i = 0; i < 2; ++i)
        {
            // Load 16 int8 values and convert to int32
            __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block.qs[i * 16]));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);

            // Convert to float and scale
            __m512 vf = _mm512_cvtepi32_ps(vi32);
            vf = _mm512_mul_ps(vf, vscale);

            // Store result
            _mm512_storeu_ps(&output[i * 16], vf);
        }
    }
#endif

#if defined(__AVX2__)
    void Q8_0Tensor::decodeBlockAVX2(const Q8_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(scale);

        // Process 8 int8 values at a time (4 iterations for 32 elements)
        for (size_t i = 0; i < 4; ++i)
        {
            // Load 8 int8 values from memory
            __m128i vi8_half = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&block.qs[i * 8]));

            // Sign-extend int8 → int32 (8 elements)
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8_half);

            // Convert int32 → float
            __m256 vf = _mm256_cvtepi32_ps(vi32);

            // Multiply by scale
            vf = _mm256_mul_ps(vf, vscale);

            // Store result
            _mm256_storeu_ps(&output[i * 8], vf);
        }
    }
#endif

    bool Q8_0Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q8_0Tensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
