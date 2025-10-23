/**
 * @file IQ3_XXSTensor.h
 * @brief IQ3_XXS quantized tensor (3.0625 bits per weight, 256 elements/block, codebook-based)
 * 
 * IQ3_XXS (Importance Quantization 3-bit XXS) achieves high compression (10.44×) using
 * codebook-based quantization with 3-bit grid lookups, sign patterns, and sub-block scales.
 * Designed for importance-weighted quantization (imatrix).
 * 
 * Block Structure (98 bytes for 256 elements):
 * - d (2 bytes): FP16 scale factor
 * - qs[96] (96 bytes): Packed grid indices (256 elements * 3 bits / 8 = 96 bytes)
 *   Layout: First QK_K/4 (64) bytes = grid indices, next bytes contain scales+signs
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Split into 8 sub-blocks of 32 elements
 * 2. For each sub-block:
 *    a. Read 4 bytes from scales_and_signs section
 *    b. Extract 4-bit scale: (aux32 >> 28)
 *    c. Compute: db = d * (0.5 + scale) * 0.5
 *    d. Decode 4 groups of 8 elements:
 *       - Grid index (8 bits) from qs[]
 *       - Sign index (7 bits) from aux32
 *       - Lookup: iq3xxs_grid[grid_idx] → 4 uint8 values
 *       - Lookup: ksigns_iq2xs[sign_idx] → sign pattern
 *       - Apply: value = db * grid_value * sign
 * 
 * Reference: ggml-quants.c (dequantize_row_iq3_xxs)
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
 * @brief IQ3_XXS block structure (98 bytes, 256 elements)
 * 
 * Matches GGML block_iq3_xxs from ggml-common.h.
 */
struct IQ3_XXSBlock {
    uint16_t d;      ///< FP16 scale factor
    uint8_t qs[96];  ///< Packed grid indices + scales + signs (3*QK_K/8 = 3*256/8 = 96)
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block
};

static_assert(sizeof(IQ3_XXSBlock) == 98, "IQ3_XXSBlock must be 98 bytes");

/**
 * @brief IQ3_XXS quantized tensor (3.0625 bpw, 10.44× compression)
 * 
 * Implements 3-bit codebook-based quantization with grid lookups and sign patterns.
 */
class IQ3_XXSTensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ3_XXS tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ3_XXS blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ3_XXSTensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ3_XXSTensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ3_XXSBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ3_XXS raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ3_XXS; }
    float compression_ratio() const override { return 10.44f; }
    
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
            std::vector<float> temp(cols);
            decodeRow(row, temp.data());
            // Convert FP32 to BF16 manually
            for (int col = 0; col < cols; ++col) {
                bf16_dst[row * cols + col] = bfloat16::from_float(temp[col]);
            }
        }
    }
    
    std::shared_ptr<TensorBase> copy() const override {
        return std::make_shared<IQ3_XXSTensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ3_XXSTensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ3_XXSBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        
        const IQ3_XXSBlock* blocks = reinterpret_cast<const IQ3_XXSBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ3_XXSBlock::BLOCK_SIZE;
            float temp[IQ3_XXSBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ3_XXSBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ3_XXSBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ3_XXSBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ3_XXSTensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ3_XXSBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        
        const IQ3_XXSBlock* blocks = reinterpret_cast<const IQ3_XXSBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ3_XXSBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ3_XXSBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ3_XXSBlock::BLOCK_SIZE) - block_start;
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
            98,   // bytes_per_block
            8,    // scale_count (8 sub-block scales)
            3,    // bits_per_value (3.0625 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
#if defined(__AVX2__)
    /**
     * @brief Decode one IQ3_XXS block (256 elements) to FP32 using AVX2 SIMD
     * 
     * Vectorized implementation processing 8 elements at a time.
     * Expected speedup: ~2× over scalar on modern CPUs.
     * 
     * @param block Input IQ3_XXS block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlockAVX2(const IQ3_XXSBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        const uint8_t* qs = block.qs;
        const uint8_t* scales_and_signs = qs + 64;
        
        float* y = output;
        
        // Process 8 sub-blocks of 32 elements
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            const __m256 db_vec = _mm256_set1_ps(db);
            
            // Process 4 groups of 8 elements
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                
                // Load 8 grid values (4 from each grid entry)
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+0]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+1]]);
                
                // Load and convert grid values to float (8 values total)
                __m128i grid1_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid1));
                __m128i grid2_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid2));
                __m128i grid_u8 = _mm_unpacklo_epi32(grid1_u8, grid2_u8);  // Interleave
                __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                __m256 grid_f = _mm256_cvtepi32_ps(grid_i32);
                
                // Create sign mask: convert bits to -1.0f or +1.0f
                __m256 sign_vec;
                alignas(32) float sign_vals[8];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                sign_vec = _mm256_load_ps(sign_vals);
                
                // Compute: y = db * grid * sign
                __m256 result = _mm256_mul_ps(db_vec, grid_f);
                result = _mm256_mul_ps(result, sign_vec);
                
                _mm256_storeu_ps(y, result);
                y += 8;
            }
            qs += 8;
        }
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Decode one IQ3_XXS block (256 elements) to FP32 using AVX512 SIMD
     * 
     * Vectorized implementation processing 16 elements at a time.
     * Expected speedup: ~3-4× over scalar on AVX512-capable CPUs.
     * 
     * @param block Input IQ3_XXS block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlockAVX512(const IQ3_XXSBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        const uint8_t* qs = block.qs;
        const uint8_t* scales_and_signs = qs + 64;
        
        float* y = output;
        
        // Process 8 sub-blocks of 32 elements
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            const __m512 db_vec = _mm512_set1_ps(db);
            
            // Process 2 groups at a time (16 elements)
            for (int l = 0; l < 4; l += 2) {
                // Load grid values for 2 groups (16 values total)
                const uint8_t* grid1_a = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+0]]);
                const uint8_t* grid2_a = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+1]]);
                const uint8_t* grid1_b = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*(l+1)+0]]);
                const uint8_t* grid2_b = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*(l+1)+1]]);
                
                // Pack 16 grid values
                alignas(64) uint8_t grid_packed[16];
                std::memcpy(grid_packed + 0, grid1_a, 4);
                std::memcpy(grid_packed + 4, grid2_a, 4);
                std::memcpy(grid_packed + 8, grid1_b, 4);
                std::memcpy(grid_packed + 12, grid2_b, 4);
                
                __m128i grid_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(grid_packed));
                __m512i grid_i32 = _mm512_cvtepu8_epi32(grid_u8);
                __m512 grid_f = _mm512_cvtepi32_ps(grid_i32);
                
                // Create sign masks for both groups
                const uint8_t signs_a = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                const uint8_t signs_b = ksigns_iq2xs[(aux32 >> (7*(l+1))) & 127];
                
                alignas(64) float sign_vals[16];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs_a & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    sign_vals[j+8] = (signs_b & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                __m512 sign_vec = _mm512_load_ps(sign_vals);
                
                // Compute: y = db * grid * sign
                __m512 result = _mm512_mul_ps(db_vec, grid_f);
                result = _mm512_mul_ps(result, sign_vec);
                
                _mm512_storeu_ps(y, result);
                y += 16;
            }
            qs += 8;
        }
    }
#endif
    
    /**
     * @brief Decode one IQ3_XXS block (256 elements) to FP32
     * 
     * Own implementation using iq3xxs_grid from IQQuantTables.h.
     * Based on GGML dequantize_row_iq3_xxs from ggml-quants.c.
     * 
     * Algorithm:
     * 1. Extract FP16 scale d
     * 2. Process 8 sub-blocks of 32 elements:
     *    a. Read 4 bytes from scales_and_signs section
     *    b. Extract 4-bit scale: (aux32 >> 28)
     *    c. Compute db = d * (0.5 + scale) * 0.5
     *    d. Process 4 groups of 8 elements:
     *       - Grid indices from qs[]: 2 bytes → 2 lookups in iq3xxs_grid
     *       - Sign index: (aux32 >> 7*l) & 127
     *       - Lookup grid: iq3xxs_grid[idx] → cast to 4 uint8 values
     *       - Lookup signs: ksigns_iq2xs[idx]
     *       - Apply: y[j] = db * grid[j] * sign
     * 
     * @param block Input IQ3_XXS block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ3_XXSBlock& block, float* output) {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback
        // Extract FP16 scale
        const float d = simd::fp16_to_fp32(block.d);
        
        // Split qs into grid indices (64 bytes) and scales_and_signs (32 bytes)
        const uint8_t* qs = block.qs;  // First 64 bytes are grid indices (256 elements / 4 per uint32 = 64 indices)
        const uint8_t* scales_and_signs = qs + 64;  // Last 32 bytes (8 sub-blocks × 4 bytes)
        
        float* y = output;
        
        // Process 8 sub-blocks of 32 elements
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // Read 4 bytes containing scale and sign info
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            
            // Extract 4-bit scale from top 4 bits
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            
            // Process 4 groups of 8 elements (32 elements total in sub-block)
            for (int l = 0; l < 4; ++l) {
                // Extract 7-bit sign index from aux32
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                
                // Get grid values (each iq3xxs_grid entry contains 4 uint8 values)
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+0]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2*l+1]]);
                
                // Decode 8 elements (4 from grid1, 4 from grid2)
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * static_cast<float>(grid1[j]) * 
                             ((signs & kmask_iq2xs[j+0]) ? -1.0f : 1.0f);
                    y[j+4] = db * static_cast<float>(grid2[j]) * 
                             ((signs & kmask_iq2xs[j+4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;  // Advance to next 8 grid indices
        }
#endif
    }
};

} // namespace llaminar
