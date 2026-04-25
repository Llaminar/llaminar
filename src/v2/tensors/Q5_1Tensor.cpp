#include "../utils/Logger.h"
/**
 * @file Q5_1Tensor.cpp
 * @brief Q5_1 quantized tensor implementation (5-bit with min offset)
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
#include "../utils/DebugEnv.h"
#include "../utils/CPUFeatures.h"
#include "FP16Utils.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#include "SIMDHelpers.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    Q5_1Tensor::Q5_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape),
          is_view_(false),
          raw_data_(raw_data),
          raw_data_ptr_(nullptr),
          view_byte_offset_(0),
          parent_(nullptr),
          device_(DeviceId::cpu()),
          device_blocks_(nullptr)
    {
        if (shape_.size() != 2)
        {
            throw std::invalid_argument("Q5_1Tensor requires 2D shape");
        }

        // Validate block alignment
        const size_t num_elements = shape_[0] * shape_[1];
        const size_t num_blocks = (num_elements + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        const size_t expected_bytes = num_blocks * sizeof(Q5_1Block);

        if (raw_data_.size() != expected_bytes)
        {
            throw std::invalid_argument("Q5_1Tensor: raw_data size mismatch (expected " +
                                        std::to_string(expected_bytes) + ", got " +
                                        std::to_string(raw_data_.size()) + ")");
        }
    }

    Q5_1Tensor::Q5_1Tensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<TensorBase> parent)
        : shape_(shape),
          is_view_(true),
          raw_data_(),
          raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset),
          parent_(parent),
          device_(DeviceId::cpu()),
          device_blocks_(nullptr)
    {
        if (shape_.size() != 2)
        {
            throw std::invalid_argument("Q5_1Tensor view requires 2D shape");
        }
    }

    // Zero-copy constructor for mmap-backed data
    Q5_1Tensor::Q5_1Tensor(const std::vector<size_t> &shape,
                           const uint8_t *mmap_data,
                           size_t byte_size,
                           std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q5_1Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q5_1Block);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("Q5_1Tensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    Q5_1Tensor::~Q5_1Tensor()
    {
        // Pre-destroy heap vectors to avoid glibc free(): invalid pointer crash
        // during implicit member destruction of large 3D MoE expert weight tensors.
        // See Q4_KTensor teardown investigation for details.
        { std::vector<uint8_t>().swap(raw_data_); }
        { std::vector<size_t>().swap(shape_); }
    }

    const float *Q5_1Tensor::data() const
    {
        assertValid("Q5_1Tensor::data");
        // Check if raw data was released after GEMM packing
        if (raw_data_released_)
        {
            LOG_DEBUG("Q5_1Tensor::data() called but raw data was released after GEMM packing");
            return nullptr;
        }

        // Dequantize to cache
        const size_t num_elements = shape_[0] * shape_[1];
        dequant_cache_.resize(num_elements);

        const size_t num_blocks = (num_elements + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);

#pragma omp parallel for schedule(static) if (num_elements > 10000)
        for (size_t i = 0; i < num_blocks; ++i)
        {
            decodeBlock(blocks[i], &dequant_cache_[i * Q5_1Block::BLOCK_SIZE]);
        }

        return dequant_cache_.data();
    }

    float *Q5_1Tensor::mutable_data()
    {
        throw std::runtime_error("Q5_1Tensor is read-only (quantized format)");
    }

    bool Q5_1Tensor::copyFrom(const TensorBase *src)
    {
        throw std::runtime_error("Q5_1Tensor::copyFrom not implemented (read-only)");
    }

    std::unique_ptr<ITensorGemm> Q5_1Tensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    std::shared_ptr<TensorBase> Q5_1Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q5_1Tensor::create_view: only 2D views supported");
        }

        // Compute effective 2D layout (supports both 2D and 3D parents).
        // GGUF 3D: shape=[ne0, ne1, ne2] where ne0=cols (fastest), ne2=outermost.
        // Flattened to 2D [ne1*ne2, ne0] = [total_rows, K].
        size_t K, total_rows;
        if (shape_.size() == 2)
        {
            K = shape_[1];
            total_rows = shape_[0];
        }
        else if (shape_.size() == 3)
        {
            // GGUF 3D: shape = [ne[0], ne[1], ne[2]], ne[0] is fastest-varying (cols/K)
            K = shape_[0];
            total_rows = shape_[1] * shape_[2];
        }
        else
        {
            throw std::invalid_argument("Q5_1Tensor::create_view: parent must be 2D or 3D");
        }

        // Validate: K dimension must match
        if (new_shape[1] != K)
        {
            throw std::invalid_argument("Q5_1Tensor::create_view: K dimension must match parent");
        }

        // Row-slice only: offset must be row-aligned
        if (offset % K != 0)
        {
            throw std::invalid_argument("Q5_1Tensor::create_view: offset must be row-aligned");
        }

        // Validate: view must fit within parent bounds
        const size_t start_row = offset / K;
        const size_t end_row = start_row + new_shape[0];
        if (end_row > total_rows)
        {
            throw std::out_of_range("Q5_1Tensor::create_view: view exceeds parent bounds");
        }

        const size_t blocks_per_row = (K + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        size_t byte_offset = start_row * blocks_per_row * sizeof(Q5_1Block);

        const uint8_t *base_ptr;
        std::shared_ptr<TensorBase> root_parent;

        if (is_view_)
        {
            // Chain views: add offsets
            byte_offset += view_byte_offset_;
            base_ptr = raw_data_ptr_;
            root_parent = parent_;
        }
        else
        {
            base_ptr = raw_data_.data();
            root_parent = std::static_pointer_cast<TensorBase>(shared_from_this());
        }

        return std::shared_ptr<Q5_1Tensor>(new Q5_1Tensor(new_shape, base_ptr, byte_offset, root_parent));
    }

    void Q5_1Tensor::decodeBlock(const Q5_1Block &block, float *output)
    {
#if defined(__AVX512F__)
        if (cpu_supports_avx512())
        {
            decodeBlockAVX512(block, output);
            return;
        }
#endif

#if defined(__AVX2__)
        if (cpu_supports_avx2())
        {
            decodeBlockAVX2(block, output);
            return;
        }
#endif

        // Final fallback to scalar
        decodeBlockScalar(block, output);
    }

    void Q5_1Tensor::decodeBlockScalar(const Q5_1Block &block, float *output)
    {
        // Reference: llama.cpp ggml-quants.c dequantize_row_q5_1()
        // Block: 32 elements, FP16 scale + FP16 min, 4-byte qh (high bits), 16-byte qs (low 4 bits)

        const float d = fp16_to_fp32(block.d);
        const float m = fp16_to_fp32(block.m);

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Decode 32 elements (2 per iteration)
        for (int j = 0; j < 16; ++j)
        {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10; // Extract bit j from qh
            const uint8_t xh_1 = ((qh >> (j + 12)) & 0x10);     // Extract bit j+12 from qh

            const int x0 = (block.qs[j] & 0x0F) | xh_0; // Lower nibble + high bit (0-31)
            const int x1 = (block.qs[j] >> 4) | xh_1;   // Upper nibble + high bit (0-31)

            output[j + 0] = x0 * d + m;
            output[j + 16] = x1 * d + m;
        }
    }

#if defined(__AVX512F__)
    void Q5_1Tensor::decodeBlockAVX512(const Q5_1Block &block, float *output)
    {
        // AVX512 implementation for Q5_1: combines Q5_0's high bit extraction with Q4_1's FMA pattern
        const float d = fp16_to_fp32(block.d);
        const float m = fp16_to_fp32(block.m);
        const __m512 vscale = _mm512_set1_ps(d);
        const __m512 vmin = _mm512_set1_ps(m);

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Load 16 bytes containing 32 4-bit low values
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Process first 16 elements: bits 0-15 of qh
        __m512i vi32_low = _mm512_cvtepu8_epi32(vlow);
        uint16_t qh_low = qh & 0xFFFF;
        __m512i vhigh_bits_low = _mm512_set_epi32(
            (qh_low >> 15) & 1, (qh_low >> 14) & 1, (qh_low >> 13) & 1, (qh_low >> 12) & 1,
            (qh_low >> 11) & 1, (qh_low >> 10) & 1, (qh_low >> 9) & 1, (qh_low >> 8) & 1,
            (qh_low >> 7) & 1, (qh_low >> 6) & 1, (qh_low >> 5) & 1, (qh_low >> 4) & 1,
            (qh_low >> 3) & 1, (qh_low >> 2) & 1, (qh_low >> 1) & 1, (qh_low >> 0) & 1);
        vhigh_bits_low = _mm512_slli_epi32(vhigh_bits_low, 4); // Shift to bit 4 position
        vi32_low = _mm512_or_si512(vi32_low, vhigh_bits_low);
        __m512 vf_low = _mm512_cvtepi32_ps(vi32_low);
        vf_low = _mm512_fmadd_ps(vf_low, vscale, vmin); // scale * value + min
        _mm512_storeu_ps(&output[0], vf_low);

        // Process second 16 elements: bits 16-31 of qh
        __m512i vi32_high = _mm512_cvtepu8_epi32(vhigh);
        uint16_t qh_high = (qh >> 16) & 0xFFFF;
        __m512i vhigh_bits_high = _mm512_set_epi32(
            (qh_high >> 15) & 1, (qh_high >> 14) & 1, (qh_high >> 13) & 1, (qh_high >> 12) & 1,
            (qh_high >> 11) & 1, (qh_high >> 10) & 1, (qh_high >> 9) & 1, (qh_high >> 8) & 1,
            (qh_high >> 7) & 1, (qh_high >> 6) & 1, (qh_high >> 5) & 1, (qh_high >> 4) & 1,
            (qh_high >> 3) & 1, (qh_high >> 2) & 1, (qh_high >> 1) & 1, (qh_high >> 0) & 1);
        vhigh_bits_high = _mm512_slli_epi32(vhigh_bits_high, 4);
        vi32_high = _mm512_or_si512(vi32_high, vhigh_bits_high);
        __m512 vf_high = _mm512_cvtepi32_ps(vi32_high);
        vf_high = _mm512_fmadd_ps(vf_high, vscale, vmin);
        _mm512_storeu_ps(&output[16], vf_high);
    }
#endif

#if defined(__AVX2__)
    void Q5_1Tensor::decodeBlockAVX2(const Q5_1Block &block, float *output)
    {
        // AVX2 implementation for Q5_1: process in 8-element chunks
        const float d = fp16_to_fp32(block.d);
        const float m = fp16_to_fp32(block.m);
        const __m256 vscale = _mm256_set1_ps(d);
        const __m256 vmin = _mm256_set1_ps(m);

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Load 16 bytes containing 32 4-bit low values
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Process first 8 low nibbles (output[0-7])
        __m256i vi32_low0 = _mm256_cvtepu8_epi32(vlow);
        uint8_t qh_bits_low0 = qh & 0xFF;
        __m256i vhigh_bits_low0 = _mm256_set_epi32(
            (qh_bits_low0 >> 7) & 1, (qh_bits_low0 >> 6) & 1,
            (qh_bits_low0 >> 5) & 1, (qh_bits_low0 >> 4) & 1,
            (qh_bits_low0 >> 3) & 1, (qh_bits_low0 >> 2) & 1,
            (qh_bits_low0 >> 1) & 1, (qh_bits_low0 >> 0) & 1);
        vhigh_bits_low0 = _mm256_slli_epi32(vhigh_bits_low0, 4);
        vi32_low0 = _mm256_or_si256(vi32_low0, vhigh_bits_low0);
        __m256 vf_low0 = _mm256_cvtepi32_ps(vi32_low0);
        vf_low0 = _mm256_fmadd_ps(vf_low0, vscale, vmin); // scale * value + min
        _mm256_storeu_ps(&output[0], vf_low0);

        // Process second 8 low nibbles (output[8-15])
        __m256i vi32_low1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vlow, 8));
        uint8_t qh_bits_low1 = (qh >> 8) & 0xFF;
        __m256i vhigh_bits_low1 = _mm256_set_epi32(
            (qh_bits_low1 >> 7) & 1, (qh_bits_low1 >> 6) & 1,
            (qh_bits_low1 >> 5) & 1, (qh_bits_low1 >> 4) & 1,
            (qh_bits_low1 >> 3) & 1, (qh_bits_low1 >> 2) & 1,
            (qh_bits_low1 >> 1) & 1, (qh_bits_low1 >> 0) & 1);
        vhigh_bits_low1 = _mm256_slli_epi32(vhigh_bits_low1, 4);
        vi32_low1 = _mm256_or_si256(vi32_low1, vhigh_bits_low1);
        __m256 vf_low1 = _mm256_cvtepi32_ps(vi32_low1);
        vf_low1 = _mm256_fmadd_ps(vf_low1, vscale, vmin);
        _mm256_storeu_ps(&output[8], vf_low1);

        // Process first 8 high nibbles (output[16-23])
        __m256i vi32_high0 = _mm256_cvtepu8_epi32(vhigh);
        uint8_t qh_bits_high0 = (qh >> 16) & 0xFF;
        __m256i vhigh_bits_high0 = _mm256_set_epi32(
            (qh_bits_high0 >> 7) & 1, (qh_bits_high0 >> 6) & 1,
            (qh_bits_high0 >> 5) & 1, (qh_bits_high0 >> 4) & 1,
            (qh_bits_high0 >> 3) & 1, (qh_bits_high0 >> 2) & 1,
            (qh_bits_high0 >> 1) & 1, (qh_bits_high0 >> 0) & 1);
        vhigh_bits_high0 = _mm256_slli_epi32(vhigh_bits_high0, 4);
        vi32_high0 = _mm256_or_si256(vi32_high0, vhigh_bits_high0);
        __m256 vf_high0 = _mm256_cvtepi32_ps(vi32_high0);
        vf_high0 = _mm256_fmadd_ps(vf_high0, vscale, vmin);
        _mm256_storeu_ps(&output[16], vf_high0);

        // Process second 8 high nibbles (output[24-31])
        __m256i vi32_high1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vhigh, 8));
        uint8_t qh_bits_high1 = (qh >> 24) & 0xFF;
        __m256i vhigh_bits_high1 = _mm256_set_epi32(
            (qh_bits_high1 >> 7) & 1, (qh_bits_high1 >> 6) & 1,
            (qh_bits_high1 >> 5) & 1, (qh_bits_high1 >> 4) & 1,
            (qh_bits_high1 >> 3) & 1, (qh_bits_high1 >> 2) & 1,
            (qh_bits_high1 >> 1) & 1, (qh_bits_high1 >> 0) & 1);
        vhigh_bits_high1 = _mm256_slli_epi32(vhigh_bits_high1, 4);
        vi32_high1 = _mm256_or_si256(vi32_high1, vhigh_bits_high1);
        __m256 vf_high1 = _mm256_cvtepi32_ps(vi32_high1);
        vf_high1 = _mm256_fmadd_ps(vf_high1, vscale, vmin);
        _mm256_storeu_ps(&output[24], vf_high1);
    }
#endif

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q5_1Tensor::to_bf16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to BF16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(temp_fp32[i]);
        }
    }

    void Q5_1Tensor::to_fp16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to FP16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(temp_fp32[i]);
        }
    }

    void Q5_1Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Decode to FP32 first, then quantize to int8
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = temp_fp32[offset + i] * scale;
                const float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst_int8[offset + i] = static_cast<int8_t>(std::round(clamped));
            }

            // Zero-fill partial block tail (if any)
            for (size_t i = count; i < block_size; ++i)
            {
                dst_int8[offset + i] = 0;
            }
        }
    }

    void Q5_1Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_row() requires 2D tensor");
        }
        if (row_idx >= shp[0])
        {
            throw std::out_of_range("Row index out of bounds");
        }

        const size_t cols = shp[1];
        const size_t blocks_per_row = (cols + block_size() - 1) / block_size();

        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            const size_t offset = kb * block_size();
            const size_t count = std::min(block_size(), cols - offset);

            float temp[256]; // Max block size
            decode_block_at(row_idx, kb, temp);

            for (size_t i = 0; i < count; ++i)
            {
                buffer[offset + i] = temp[i];
            }
        }
    }

    void Q5_1Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        // Decode full tensor (inefficient but simple)
        std::vector<float> temp_fp32(element_count());
        to_fp32(temp_fp32.data());
        std::memcpy(buffer, temp_fp32.data() + offset, count * sizeof(float));
    }

    void Q5_1Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q5_1Tensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q5_1Tensor::decode_to_q8_0: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q5_1Tensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q5_1Tensor::decode_to_q8_0: block offset out of bounds");
        }

        // Get Q5_1 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
        const Q5_1Block &q5_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Use SIMD helper to decode Q5_1 → Q8_0
        uint16_t q8_scale_fp16;
        simd::decode_q5_1_to_q8_0(
            q5_block.qs,     // Q5 lower 4 bits
            q5_block.qh,     // Q5 high bits
            q5_block.d,      // Q5 scale (FP16)
            q5_block.m,      // Q5 min (FP16)
            output->qs,      // Q8 output values
            &q8_scale_fp16); // Q8 output scale (FP16)

        output->d = q8_scale_fp16;
    }

    void Q5_1Tensor::unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
        const Q5_1Block *q5_block = &blocks[row_idx * blocks_per_row + k_block_offset];

        simd::unpack_q5_1_to_int8(*q5_block, output);
    }

    float Q5_1Tensor::get_block_scale(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
        return fp16_to_fp32(blocks[row_idx * blocks_per_row + k_block_offset].d);
    }

    float Q5_1Tensor::get_block_min(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
        return fp16_to_fp32(blocks[row_idx * blocks_per_row + k_block_offset].m);
    }

    void Q5_1Tensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const auto *blk = &typed_data()[static_cast<size_t>(n) * ctx.blocks_per_row + b];
        uint8_t *dst = vnniPayloadDst(ctx, linear);
        std::memcpy(dst, blk->qs, 16);
        std::memcpy(dst + 16, blk->qh, 4);
        ctx.scales_array[linear] = blk->d;
        ctx.mins_array[linear] = blk->m;
    }

} // namespace llaminar2
