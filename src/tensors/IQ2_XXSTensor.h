/**
 * @file IQ2_XXSTensor.h
 * @brief IQ2_XXS quantized tensor (2.0625 bits per weight, 256 elements/block, codebook-based)
 * 
 * IQ2_XXS (Importance Quantization 2-bit XXS) achieves extreme compression (15.52×) using
 * codebook-based quantization with grid lookups and sign patterns. Designed for importance-
 * weighted quantization (imatrix).
 * 
 * Block Structure (66 bytes for 256 elements):
 * - d (2 bytes): FP16 scale factor
 * - qs[32] (64 bytes): Packed grid indices + sign bits + sub-block scales
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Split into 8 sub-blocks of 32 elements
 * 2. For each sub-block:
 *    a. Read 8 bytes (2 uint32_t) from qs[]
 *    b. Extract 4-bit scale: (aux32[1] >> 28)
 *    c. Compute: db = d * (0.5 + scale) * 0.25
 *    d. Decode 4 groups of 8 elements:
 *       - Grid index (8 bits) → lookup iq2xxs_grid[256]
 *       - Sign index (7 bits) → lookup ksigns_iq2xs[128]
 *       - Apply: value = db * grid_value * sign
 * 
 * Reference: ggml-quants.c line 2275 (dequantize_row_iq2_xxs)
 * 
 * @author David Sanftenberg
 * @date 2025-10-21
 */

#pragma once

#include "QuantizedTensorBase.h"
#include "TensorFactory.h"
#include "IQQuantTables.h"
#include "../utils/SIMDHelpers.h"
#include <vector>
#include <cstring>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar {

/**
 * @brief IQ2_XXS block structure (66 bytes, 256 elements)
 * 
 * Matches GGML block_iq2_xxs from ggml-common.h line 347.
 */
struct IQ2_XXSBlock {
    uint16_t d;      ///< FP16 scale factor
    uint16_t qs[32]; ///< Packed grid indices + signs + scales (QK_K/8 = 256/8 = 32)
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block
};

static_assert(sizeof(IQ2_XXSBlock) == 66, "IQ2_XXSBlock must be 66 bytes");

/**
 * @brief IQ2_XXS quantized tensor (2.0625 bpw, 15.52× compression)
 * 
 * Implements codebook-based quantization with grid lookups and sign patterns.
 */
class IQ2_XXSTensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ2_XXS tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ2_XXS blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ2_XXSTensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ2_XXSTensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ2_XXSBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ2_XXS raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ2_XXS; }
    float compression_ratio() const override { return 15.52f; }
    
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
        return std::make_shared<IQ2_XXSTensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ2_XXSTensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ2_XXSBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        
        const IQ2_XXSBlock* blocks = reinterpret_cast<const IQ2_XXSBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ2_XXSBlock::BLOCK_SIZE;
            float temp[IQ2_XXSBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ2_XXSBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ2_XXSBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ2_XXSBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ2_XXSTensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ2_XXSBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        
        const IQ2_XXSBlock* blocks = reinterpret_cast<const IQ2_XXSBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ2_XXSBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ2_XXSBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ2_XXSBlock::BLOCK_SIZE) - block_start;
            size_t copy_count = copy_end - copy_start;
            
            std::memcpy(buffer + buffer_offset, temp + copy_start, copy_count * sizeof(float));
            buffer_offset += copy_count;
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
            66,   // bytes_per_block
            8,    // scale_count (8 sub-block scales)
            2,    // bits_per_value (2.0625 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
#if defined(__AVX2__)
    /**
     * @brief AVX2-optimized IQ2_XXS block decode
     * 
     * Vectorizes the 8-element grid value processing using AVX2.
     */
    static void decodeBlockAVX2(const IQ2_XXSBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32[2];
            std::memcpy(aux32, &block.qs[4 * ib32], 8);
            const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
            
            const float db = d * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
            const __m256 scale_vec = _mm256_set1_ps(db);
            
            for (int l = 0; l < 4; ++l) {
                const uint8_t grid_idx = aux8[l];
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xxs_grid[grid_idx]);
                
                // Load 8 uint8 grid values and convert to float
                __m128i grid_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(grid));
                __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                __m256 grid_f32 = _mm256_cvtepi32_ps(grid_i32);
                
                // Apply scale
                __m256 result = _mm256_mul_ps(scale_vec, grid_f32);
                
                // Apply signs (store, flip, reload)
                alignas(32) float result_arr[8];
                _mm256_store_ps(result_arr, result);
                for (int j = 0; j < 8; ++j) {
                    if (signs & kmask_iq2xs[j]) {
                        result_arr[j] = -result_arr[j];
                    }
                }
                _mm256_storeu_ps(output, _mm256_load_ps(result_arr));
                output += 8;
            }
        }
    }
#endif
    
    /**
     * @brief Decode one IQ2_XXS block (256 elements) to FP32
     * 
     * Implements GGML dequantize_row_iq2_xxs algorithm (ggml-quants.c line 2275).
     * Dispatches to SIMD version if available, otherwise uses scalar fallback.
     * 
     * Algorithm:
     * 1. Extract FP16 scale d
     * 2. Process 8 sub-blocks of 32 elements:
     *    a. Read 8 bytes (2 uint32_t) from qs[]
     *    b. Extract 4-bit scale: (aux32[1] >> 28) → 0-15
     *    c. Compute db = d * (0.5 + scale) * 0.25
     *    d. Process 4 groups of 8 elements:
     *       - Grid index: aux8[l] (8 bits)
     *       - Sign index: (aux32[1] >> 7*l) & 127 (7 bits)
     *       - Lookup grid values: iq2xxs_grid[grid_idx]
     *       - Lookup signs: ksigns_iq2xs[sign_idx]
     *       - Apply: y[j] = db * grid[j] * (signs & kmask ? -1 : 1)
     * 
     * @param block Input IQ2_XXS block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ2_XXSBlock& block, float* output) {
#if defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback
        const float d = simd::fp16_to_fp32(block.d);
        
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // Read 8 bytes (2 uint32_t) from qs[]
            uint32_t aux32[2];
            std::memcpy(aux32, &block.qs[4 * ib32], 8);
            const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
            
            // Extract 4-bit sub-block scale from top 4 bits of aux32[1]
            const float db = d * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
            
            // Process 4 groups of 8 elements
            for (int l = 0; l < 4; ++l) {
                const uint8_t grid_idx = aux8[l];
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&iq2xxs_grid[grid_idx]);
                
                for (int j = 0; j < 8; ++j) {
                    output[j] = db * static_cast<float>(grid[j]) *
                               ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                output += 8;
            }
        }
#endif
    }
};

} // namespace llaminar
