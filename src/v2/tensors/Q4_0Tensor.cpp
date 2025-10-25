/**
 * @file Q4_0Tensor.cpp
 * @brief Q4_0 quantized tensor implementation (4-bit per weight, 8.0x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "Tensors.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    Q4_0Tensor::Q4_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q4_0Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_0Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_0Tensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> Q4_0Tensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void Q4_0Tensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(raw_data_.data());
        const Q4_0Block &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *Q4_0Tensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void Q4_0Tensor::decodeBlock(const Q4_0Block &block, float *output)
    {
#if defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback: decode formula is scale * (nibble - 8)
        const float scale = fp16_to_fp32(block.d);

        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t byte = block.qs[i];

            // Extract two 4-bit nibbles (low and high)
            const int8_t v0 = static_cast<int8_t>((byte & 0x0F) - 8);
            const int8_t v1 = static_cast<int8_t>((byte >> 4) - 8);

            output[i * 2 + 0] = scale * static_cast<float>(v0);
            output[i * 2 + 1] = scale * static_cast<float>(v1);
        }
#endif
    }

#if defined(__AVX2__)
    void Q4_0Tensor::decodeBlockAVX2(const Q4_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256i vminus8 = _mm256_set1_epi8(-8);

        // Process 16 bytes → 32 nibbles → 32 float outputs
        // Load 16 bytes (contains 32 4-bit values)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Broadcast to 256-bit for parallel processing
        __m256i v4bit_256 = _mm256_set_m128i(v4bit, v4bit);

        // Extract low nibbles (mask 0x0F)
        __m256i vlow = _mm256_and_si256(v4bit_256, _mm256_set1_epi8(0x0F));

        // Extract high nibbles (shift right 4)
        __m256i vhigh = _mm256_and_si256(_mm256_srli_epi16(v4bit_256, 4), _mm256_set1_epi8(0x0F));

        // Subtract 8 from both (Q4_0 bias)
        vlow = _mm256_add_epi8(vlow, vminus8);
        vhigh = _mm256_add_epi8(vhigh, vminus8);

        // Unpack and convert to float (4 chunks of 8 elements each)
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            __m128i vlow_chunk = (chunk == 0) ? _mm256_castsi256_si128(vlow) : _mm256_extracti128_si256(vlow, 1);
            __m128i vhigh_chunk = (chunk == 0) ? _mm256_castsi256_si128(vhigh) : _mm256_extracti128_si256(vhigh, 1);

            // Convert first 8 low nibbles
            __m256i vi32_low = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(&vlow_chunk)));
            __m256 vf_low = _mm256_cvtepi32_ps(vi32_low);
            vf_low = _mm256_mul_ps(vf_low, vscale);
            _mm256_storeu_ps(&output[chunk * 16], vf_low);

            // Convert first 8 high nibbles
            __m256i vi32_high = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(&vhigh_chunk)));
            __m256 vf_high = _mm256_cvtepi32_ps(vi32_high);
            vf_high = _mm256_mul_ps(vf_high, vscale);
            _mm256_storeu_ps(&output[chunk * 16 + 8], vf_high);
        }
    }
#endif

    Q4_0Tensor::~Q4_0Tensor() {}

    bool Q4_0Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_0Tensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_0Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_0Tensor::mutable_data()
    {
        throw std::runtime_error("Q4_0Tensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> Q4_0Tensor::createRoPE()
    {
        throw std::runtime_error("Q4_0Tensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> Q4_0Tensor::createSwiGLU()
    {
        throw std::runtime_error("Q4_0Tensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> Q4_0Tensor::createSoftmax()
    {
        throw std::runtime_error("Q4_0Tensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> Q4_0Tensor::createRMSNorm()
    {
        throw std::runtime_error("Q4_0Tensor: RMSNorm not supported");
    }




    bool Q4_0Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[Q4_0Tensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
