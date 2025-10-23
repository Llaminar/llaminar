/**
 * @file IQ2_XSTensor.h
 * @brief IQ2_XS quantized tensor (2.3125 bits per weight, 256 elements/block, codebook-based)
 * 
 * IQ2_XS (Importance Quantization 2-bit XS) achieves strong compression (13.84×) using
 * codebook-based quantization with a larger grid (512 entries) and explicit scales array.
 * Better quality than IQ2_XXS at the cost of 8 extra bytes per block.
 * 
 * Block Structure (74 bytes for 256 elements):
 * - d (2 bytes): FP16 scale factor
 * - qs[32] (64 bytes): Packed grid indices (9 bits) + sign bits (7 bits) per uint16_t
 * - scales[8] (8 bytes): Explicit scales, 2 per sub-block (4-bit nibbles)
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Split into 8 sub-blocks of 32 elements
 * 2. For each sub-block:
 *    a. Extract TWO 4-bit scales from scales[ib32]:
 *       - Low nibble: scales[ib32] & 0xf → db[0]
 *       - High nibble: scales[ib32] >> 4 → db[1]
 *    b. Compute db[0] = d * (0.5 + low_scale) * 0.25
 *    c. Compute db[1] = d * (0.5 + high_scale) * 0.25
 *    d. Decode 4 groups of 8 elements:
 *       - Grid index (9 bits): qs[4*ib32 + l] & 511 → iq2xs_grid[512]
 *       - Sign index (7 bits): qs[4*ib32 + l] >> 9 → ksigns_iq2xs[128]
 *       - Use db[l/2]: alternates between db[0] (groups 0-1) and db[1] (groups 2-3)
 *       - Apply: value = db[l/2] * grid_value * sign
 * 
 * Key Differences from IQ2_XXS:
 * - Larger grid: 512 entries vs 256 entries (9-bit vs 8-bit indices)
 * - Explicit scales: 8 bytes of scales[8] in block structure
 * - Dual scales: Two 4-bit scales per sub-block (low/high nibbles)
 * - Simpler packing: grid index (9 bits) + sign index (7 bits) in uint16_t
 * - Better quality: More quantization options and finer scale granularity
 * 
 * Reference: ggml-quants.c line 2303 (dequantize_row_iq2_xs)
 * 
 * @author David Sanftenberg
 * @date 2025-10-21
 */

#pragma once

#include "QuantizedTensorBase.h"
#include "TensorFactory.h"
#include "IQQuantTables.h"
#include "../utils/SIMDHelpers.h"
#include "../utils/BFloat16.h"
#include <vector>
#include <cstring>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar {

/**
 * @brief IQ2_XS block structure (74 bytes, 256 elements)
 * 
 * Matches GGML block_iq2_xs from Vulkan shaders (types.comp).
 */
struct IQ2_XSBlock {
    uint16_t d;        ///< FP16 scale factor
    uint16_t qs[32];   ///< Packed 9-bit grid indices + 7-bit sign indices (QK_K/8 = 256/8 = 32)
    uint8_t scales[8]; ///< Explicit scales: 2 per sub-block (4-bit nibbles, QK_K/32 = 256/32 = 8)
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block
};

static_assert(sizeof(IQ2_XSBlock) == 74, "IQ2_XSBlock must be 74 bytes");

/**
 * @brief IQ2_XS quantized tensor (2.3125 bpw, 13.84× compression)
 * 
 * Implements codebook-based quantization with larger grid and dual scales per sub-block.
 */
class IQ2_XSTensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ2_XS tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ2_XS blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ2_XSTensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ2_XSTensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ2_XSBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ2_XS raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ2_XS; }
    float compression_ratio() const override { return 13.84f; }
    
    size_t element_count() const override {
        size_t count = 1;
        for (int dim : shape_) count *= dim;
        return count;
    }
    
    // ========== Decode API ==========
    
    void decode_to_fp32(float* dst) const override {
        int rows = shape_[0];
        int cols = shape_[1];
        #pragma omp parallel for if(rows > 4)
        for (int row = 0; row < rows; ++row) {
            decodeRow(row, dst + row * cols);
        }
    }
    
    void decode_to_bf16(void* dst) const override {
        int rows = shape_[0];
        int cols = shape_[1];
        bfloat16* bf16_dst = static_cast<bfloat16*>(dst);
        #pragma omp parallel for if(rows > 4)
        for (int row = 0; row < rows; ++row) {
            decodeRowToBF16(row, bf16_dst + row * cols);
        }
    }
    
    std::shared_ptr<TensorBase> copy() const override {
        return std::make_shared<IQ2_XSTensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ2_XSTensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ2_XSBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ2_XSBlock::BLOCK_SIZE;
        
        const IQ2_XSBlock* blocks = reinterpret_cast<const IQ2_XSBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ2_XSBlock::BLOCK_SIZE;
            float temp[IQ2_XSBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ2_XSBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ2_XSBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ2_XSBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ2_XSTensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ2_XSBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ2_XSBlock::BLOCK_SIZE;
        
        const IQ2_XSBlock* blocks = reinterpret_cast<const IQ2_XSBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ2_XSBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ2_XSBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ2_XSBlock::BLOCK_SIZE) - block_start;
            size_t copy_count = copy_end - copy_start;
            
            std::memcpy(buffer + buffer_offset, temp + copy_start, copy_count * sizeof(float));
            buffer_offset += copy_count;
        }
    }
    
    void decodeRowToBF16(size_t row_idx, bfloat16* buffer) const {
        int cols = shape_[1];
        float temp_buffer[cols];
        decodeRow(row_idx, temp_buffer);
        
        for (int i = 0; i < cols; ++i) {
            buffer[i] = bfloat16::from_float(temp_buffer[i]);
        }
    }
    
    // ========== Raw Block Access ==========
    
    const uint8_t* raw_data() const override {
        return raw_data_.data();
    }
    
    size_t raw_size() const override {
        return raw_data_.size();
    }
    
    const QuantBlockDescriptor& block_descriptor() const override {
        static QuantBlockDescriptor desc{
            256,  // elements_per_block
            74,   // bytes_per_block
            16,   // scale_count (8 sub-blocks × 2 scales per sub-block = 16)
            2,    // bits_per_value (2.3125 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
#if defined(__AVX2__)
    /**
     * @brief AVX2-optimized IQ2_XS block decode
     * 
     * Vectorizes the 8-element grid processing using AVX2 intrinsics.
     * Achieves ~2× speedup over scalar code on AVX2-capable CPUs.
     */
    static void decodeBlockAVX2(const IQ2_XSBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            float db[2];
            db[0] = d * (0.5f + static_cast<float>(block.scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + static_cast<float>(block.scales[ib32] >> 4)) * 0.25f;
            
            for (int l = 0; l < 4; ++l) {
                const uint16_t packed = block.qs[4 * ib32 + l];
                const uint16_t grid_idx = packed & 511;
                const uint8_t signs = ksigns_iq2xs[packed >> 9];
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xs_grid[grid_idx]);
                const float scale = db[l / 2];
                
                // SIMD: Load 8 uint8 grid values and convert to float
                __m128i grid_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(grid));
                __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                __m256 grid_f32 = _mm256_cvtepi32_ps(grid_i32);
                
                // Apply scale
                const __m256 scale_vec = _mm256_set1_ps(scale);
                __m256 result = _mm256_mul_ps(scale_vec, grid_f32);
                
                // Apply signs (conditional negation requires scalar)
                alignas(32) float result_arr[8];
                _mm256_store_ps(result_arr, result);
                for (int j = 0; j < 8; ++j) {
                    if (signs & kmask_iq2xs[j]) result_arr[j] = -result_arr[j];
                }
                _mm256_storeu_ps(output, _mm256_load_ps(result_arr));
                output += 8;
            }
        }
    }
#endif
    
    /**
     * @brief Decode one IQ2_XS block (256 elements) to FP32
     * 
     * Implements GGML dequantize_row_iq2_xs algorithm (ggml-quants.c line 2303).
     * Dispatches to AVX2 SIMD if available, otherwise scalar fallback.
     * 
     * Algorithm:
     * 1. Extract FP16 scale d
     * 2. Process 8 sub-blocks of 32 elements:
     *    a. Extract TWO 4-bit scales from scales[ib32]:
     *       - db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
     *       - db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
     *    b. Process 4 groups of 8 elements:
     *       - Grid index (9 bits): qs[4*ib32 + l] & 511
     *       - Sign index (7 bits): qs[4*ib32 + l] >> 9
     *       - Lookup grid values: iq2xs_grid[grid_idx]
     *       - Lookup signs: ksigns_iq2xs[sign_idx]
     *       - Use db[l/2]: alternates between db[0] (l=0,1) and db[1] (l=2,3)
     *       - Apply: y[j] = db[l/2] * grid[j] * (signs & kmask ? -1 : 1)
     * 
     * @param block Input IQ2_XS block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ2_XSBlock& block, float* output) {
#if defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback
        const float d = simd::fp16_to_fp32(block.d);
        
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // Extract TWO 4-bit scales from scales[ib32] (low and high nibbles)
            float db[2];
            db[0] = d * (0.5f + static_cast<float>(block.scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + static_cast<float>(block.scales[ib32] >> 4)) * 0.25f;
            
            // Process 4 groups of 8 elements
            for (int l = 0; l < 4; ++l) {
                // Extract 9-bit grid index and 7-bit sign index from uint16_t
                const uint16_t packed = block.qs[4 * ib32 + l];
                const uint16_t grid_idx = packed & 511;  // 9 bits: 0x1FF
                const uint8_t signs = ksigns_iq2xs[packed >> 9];  // 7 bits: top 7 bits
                
                // Lookup grid values (8 bytes packed in uint64_t)
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xs_grid[grid_idx]);
                
                // Alternate between db[0] (l=0,1) and db[1] (l=2,3)
                const float scale = db[l / 2];
                
                for (int j = 0; j < 8; ++j) {
                    output[j] = scale * static_cast<float>(grid[j]) *
                               ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                output += 8;
            }
        }
#endif
    }
};

} // namespace llaminar
