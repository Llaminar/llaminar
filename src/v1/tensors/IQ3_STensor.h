/**
 * @file IQ3_STensor.h
 * @brief IQ3_S quantized tensor (3.4375 bits per weight, 256 elements/block, codebook-based)
 * 
 * IQ3_S (Importance Quantization 3-bit S) achieves good compression (9.29×) using advanced
 * codebook-based quantization with 9-bit grid lookups (8-bit qs + 1-bit qh), dedicated sign
 * arrays, and sub-block scales for improved accuracy.
 * 
 * Block Structure (110 bytes for 256 elements):
 * - d (2 bytes): FP16 scale factor
 * - qs[64] (64 bytes): Main quantized data (low 8 bits of grid indices)
 * - qh[8] (8 bytes): High bits for grid indices (extends qs to 9 bits)
 * - signs[32] (32 bytes): Sign bits for decoded values
 * - scales[4] (4 bytes): Sub-block scales (IQ3S_N_SCALE = QK_K/64 = 4)
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Split into 8 sub-blocks of 32 elements (processed in pairs)
 * 2. For each pair of sub-blocks (64 elements):
 *    a. Read 2 scales: (scales[ib32/2] & 0xf), (scales[ib32/2] >> 4)
 *    b. Compute: db1 = d * (1 + 2*scale1), db2 = d * (1 + 2*scale2)
 *    c. For each sub-block (4 groups of 8 elements):
 *       - Combine qs + qh to form 9-bit grid index
 *       - Lookup: iq3s_grid[qs | ((qh << shift) & 256)] → 4 uint8 values
 *       - Lookup signs: signs[l] with kmask_iq2xs
 *       - Apply: y[j] = db * grid[j] * sign
 * 
 * Reference: ggml-quants.c (dequantize_row_iq3_s)
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
 * @brief IQ3_S block structure (110 bytes, 256 elements)
 * 
 * Matches GGML block_iq3_s from ggml-common.h.
 */
struct IQ3_SBlock {
    static constexpr size_t N_SCALE = 4; ///< QK_K/64 = 256/64 = 4
    
    uint16_t d;            ///< FP16 scale factor
    uint8_t qs[64];        ///< Main quantized data (QK_K/4)
    uint8_t qh[8];         ///< High bits (QK_K/32)
    uint8_t signs[32];     ///< Sign bits (QK_K/8)
    uint8_t scales[N_SCALE]; ///< Sub-block scales
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block
};

static_assert(sizeof(IQ3_SBlock) == 110, "IQ3_SBlock must be 110 bytes");

/**
 * @brief IQ3_S quantized tensor (3.4375 bpw, 9.29× compression)
 * 
 * Implements advanced 3-bit codebook quantization with 9-bit grid indices,
 * dedicated sign arrays, and sub-block scales.
 */
class IQ3_STensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ3_S tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ3_S blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ3_STensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ3_STensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ3_SBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ3_S raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ3_S; }
    float compression_ratio() const override { return 9.29f; }
    
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
        return std::make_shared<IQ3_STensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ3_STensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ3_SBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ3_SBlock::BLOCK_SIZE;
        
        const IQ3_SBlock* blocks = reinterpret_cast<const IQ3_SBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ3_SBlock::BLOCK_SIZE;
            float temp[IQ3_SBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ3_SBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ3_SBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ3_SBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ3_STensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ3_SBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ3_SBlock::BLOCK_SIZE;
        
        const IQ3_SBlock* blocks = reinterpret_cast<const IQ3_SBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ3_SBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ3_SBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ3_SBlock::BLOCK_SIZE) - block_start;
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
            110,  // bytes_per_block
            4,    // scale_count (4 sub-block scales)
            3,    // bits_per_value (3.4375 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
#if defined(__AVX2__)
    /**
     * @brief Decode one IQ3_S block (256 elements) to FP32 using AVX2 SIMD
     * 
     * Hybrid implementation: Scalar 9-bit grid indexing + SIMD computation.
     * The 9-bit indexing (qs[8 bits] | qh[1 bit]) is complex to vectorize,
     * so we keep it scalar but vectorize the multiplication/sign application.
     * Expected speedup: ~1.5-2× over pure scalar.
     * 
     * @param block Input IQ3_S block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlockAVX2(const IQ3_SBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        
        const uint8_t* qs = block.qs;
        const uint8_t* qh = block.qh;
        const uint8_t* signs = block.signs;
        const uint8_t* scales_u8 = block.scales;
        
        float* y = output;
        
        // Process 8 sub-blocks in pairs (each pair = 64 elements)
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            // Extract two 4-bit scales from one byte
            const float db1 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] & 0xf));
            const float db2 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] >> 4));
            const __m256 db1_vec = _mm256_set1_ps(db1);
            const __m256 db2_vec = _mm256_set1_ps(db2);
            
            // First sub-block (32 elements, 4 groups of 8)
            for (int l = 0; l < 4; ++l) {
                // Scalar 9-bit indexing (complex bit manipulation)
                const uint16_t idx1 = qs[2*l+0] | ((qh[0] << (8-2*l)) & 256);
                const uint16_t idx2 = qs[2*l+1] | ((qh[0] << (7-2*l)) & 256);
                
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2]);
                
                // Load 8 grid values (4 from each)
                __m128i grid1_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid1));
                __m128i grid2_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid2));
                __m128i grid_u8 = _mm_unpacklo_epi32(grid1_u8, grid2_u8);
                __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                __m256 grid_f = _mm256_cvtepi32_ps(grid_i32);
                
                // Create sign mask
                alignas(32) float sign_vals[8];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                __m256 sign_vec = _mm256_load_ps(sign_vals);
                
                // Compute: y = db1 * grid * sign
                __m256 result = _mm256_mul_ps(db1_vec, grid_f);
                result = _mm256_mul_ps(result, sign_vec);
                _mm256_storeu_ps(y, result);
                y += 8;
            }
            
            qs += 8;
            signs += 4;
            
            // Second sub-block (32 elements, 4 groups of 8)
            for (int l = 0; l < 4; ++l) {
                // Scalar 9-bit indexing
                const uint16_t idx1 = qs[2*l+0] | ((qh[1] << (8-2*l)) & 256);
                const uint16_t idx2 = qs[2*l+1] | ((qh[1] << (7-2*l)) & 256);
                
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2]);
                
                // Load 8 grid values
                __m128i grid1_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid1));
                __m128i grid2_u8 = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(grid2));
                __m128i grid_u8 = _mm_unpacklo_epi32(grid1_u8, grid2_u8);
                __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                __m256 grid_f = _mm256_cvtepi32_ps(grid_i32);
                
                // Create sign mask
                alignas(32) float sign_vals[8];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                __m256 sign_vec = _mm256_load_ps(sign_vals);
                
                // Compute: y = db2 * grid * sign
                __m256 result = _mm256_mul_ps(db2_vec, grid_f);
                result = _mm256_mul_ps(result, sign_vec);
                _mm256_storeu_ps(y, result);
                y += 8;
            }
            
            qs += 8;
            signs += 4;
            qh += 2;
        }
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Decode one IQ3_S block (256 elements) to FP32 using AVX512 SIMD
     * 
     * Hybrid implementation: Scalar 9-bit indexing + AVX512 computation.
     * Processes 16 elements at a time for better throughput.
     * Expected speedup: ~2-3× over pure scalar.
     * 
     * @param block Input IQ3_S block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlockAVX512(const IQ3_SBlock& block, float* output) {
        const float d = simd::fp16_to_fp32(block.d);
        
        const uint8_t* qs = block.qs;
        const uint8_t* qh = block.qh;
        const uint8_t* signs = block.signs;
        const uint8_t* scales_u8 = block.scales;
        
        float* y = output;
        
        // Process 8 sub-blocks in pairs (each pair = 64 elements)
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] & 0xf));
            const float db2 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] >> 4));
            const __m512 db1_vec = _mm512_set1_ps(db1);
            const __m512 db2_vec = _mm512_set1_ps(db2);
            
            // First sub-block (32 elements) - process 2 groups at a time (16 elements)
            for (int l = 0; l < 4; l += 2) {
                // Scalar 9-bit indexing for 4 lookups (16 values total)
                const uint16_t idx1_a = qs[2*l+0] | ((qh[0] << (8-2*l)) & 256);
                const uint16_t idx2_a = qs[2*l+1] | ((qh[0] << (7-2*l)) & 256);
                const uint16_t idx1_b = qs[2*(l+1)+0] | ((qh[0] << (8-2*(l+1))) & 256);
                const uint16_t idx2_b = qs[2*(l+1)+1] | ((qh[0] << (7-2*(l+1))) & 256);
                
                const uint8_t* grid1_a = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1_a]);
                const uint8_t* grid2_a = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2_a]);
                const uint8_t* grid1_b = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1_b]);
                const uint8_t* grid2_b = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2_b]);
                
                // Pack 16 grid values
                alignas(64) uint8_t grid_packed[16];
                std::memcpy(grid_packed + 0, grid1_a, 4);
                std::memcpy(grid_packed + 4, grid2_a, 4);
                std::memcpy(grid_packed + 8, grid1_b, 4);
                std::memcpy(grid_packed + 12, grid2_b, 4);
                
                __m128i grid_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(grid_packed));
                __m512i grid_i32 = _mm512_cvtepu8_epi32(grid_u8);
                __m512 grid_f = _mm512_cvtepi32_ps(grid_i32);
                
                // Create sign masks
                alignas(64) float sign_vals[16];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    sign_vals[j+8] = (signs[l+1] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                __m512 sign_vec = _mm512_load_ps(sign_vals);
                
                // Compute: y = db1 * grid * sign
                __m512 result = _mm512_mul_ps(db1_vec, grid_f);
                result = _mm512_mul_ps(result, sign_vec);
                _mm512_storeu_ps(y, result);
                y += 16;
            }
            
            qs += 8;
            signs += 4;
            
            // Second sub-block (32 elements) - process 2 groups at a time (16 elements)
            for (int l = 0; l < 4; l += 2) {
                // Scalar 9-bit indexing
                const uint16_t idx1_a = qs[2*l+0] | ((qh[1] << (8-2*l)) & 256);
                const uint16_t idx2_a = qs[2*l+1] | ((qh[1] << (7-2*l)) & 256);
                const uint16_t idx1_b = qs[2*(l+1)+0] | ((qh[1] << (8-2*(l+1))) & 256);
                const uint16_t idx2_b = qs[2*(l+1)+1] | ((qh[1] << (7-2*(l+1))) & 256);
                
                const uint8_t* grid1_a = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1_a]);
                const uint8_t* grid2_a = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2_a]);
                const uint8_t* grid1_b = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx1_b]);
                const uint8_t* grid2_b = reinterpret_cast<const uint8_t*>(&iq3s_grid[idx2_b]);
                
                // Pack 16 grid values
                alignas(64) uint8_t grid_packed[16];
                std::memcpy(grid_packed + 0, grid1_a, 4);
                std::memcpy(grid_packed + 4, grid2_a, 4);
                std::memcpy(grid_packed + 8, grid1_b, 4);
                std::memcpy(grid_packed + 12, grid2_b, 4);
                
                __m128i grid_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(grid_packed));
                __m512i grid_i32 = _mm512_cvtepu8_epi32(grid_u8);
                __m512 grid_f = _mm512_cvtepi32_ps(grid_i32);
                
                // Create sign masks
                alignas(64) float sign_vals[16];
                for (int j = 0; j < 8; ++j) {
                    sign_vals[j] = (signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    sign_vals[j+8] = (signs[l+1] & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                }
                __m512 sign_vec = _mm512_load_ps(sign_vals);
                
                // Compute: y = db2 * grid * sign
                __m512 result = _mm512_mul_ps(db2_vec, grid_f);
                result = _mm512_mul_ps(result, sign_vec);
                _mm512_storeu_ps(y, result);
                y += 16;
            }
            
            qs += 8;
            signs += 4;
            qh += 2;
        }
    }
#endif
    
    /**
     * @brief Decode one IQ3_S block (256 elements) to FP32
     * 
     * Own implementation using iq3s_grid from IQQuantTables.h.
     * Based on GGML dequantize_row_iq3_s from ggml-quants.c.
     * 
     * Algorithm:
     * 1. Extract FP16 scale d
     * 2. Process 8 sub-blocks in pairs (64 elements per pair):
     *    a. Extract scales: scale1 = scales[ib32/2] & 0xf, scale2 = scales[ib32/2] >> 4
     *    b. Compute: db1 = d * (1 + 2*scale1), db2 = d * (1 + 2*scale2)
     *    c. For first sub-block (4 groups of 8):
     *       - Combine: grid_idx = qs[2*l+i] | ((qh[0] << shift) & 256)
     *       - Lookup: iq3s_grid[grid_idx] → cast to 4 uint8 values
     *       - Apply signs: signs[l] with kmask_iq2xs
     *       - Compute: y[j] = db1 * grid[j] * sign
     *    d. For second sub-block (4 groups of 8):
     *       - Same pattern with qh[1] and db2
     * 
     * @param block Input IQ3_S block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ3_SBlock& block, float* output) {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        // Scalar fallback
        // Extract FP16 scale
        const float d = simd::fp16_to_fp32(block.d);
        
        const uint8_t* qs = block.qs;
        const uint8_t* qh = block.qh;
        const uint8_t* signs = block.signs;
        const uint8_t* scales_u8 = block.scales;
        
        float* y = output;
        
        // Process 8 sub-blocks in pairs (each pair = 64 elements)
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            // Extract two 4-bit scales from one byte
            const float db1 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] & 0xf));
            const float db2 = d * static_cast<float>(1 + 2*(scales_u8[ib32/2] >> 4));
            
            // First sub-block (32 elements, 4 groups of 8)
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    &iq3s_grid[qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    &iq3s_grid[qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)]);
                
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * static_cast<float>(grid1[j]) * 
                             ((signs[l] & kmask_iq2xs[j+0]) ? -1.0f : 1.0f);
                    y[j+4] = db1 * static_cast<float>(grid2[j]) * 
                             ((signs[l] & kmask_iq2xs[j+4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            
            qs += 8;
            signs += 4;
            
            // Second sub-block (32 elements, 4 groups of 8)
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    &iq3s_grid[qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    &iq3s_grid[qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)]);
                
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * static_cast<float>(grid1[j]) * 
                             ((signs[l] & kmask_iq2xs[j+0]) ? -1.0f : 1.0f);
                    y[j+4] = db2 * static_cast<float>(grid2[j]) * 
                             ((signs[l] & kmask_iq2xs[j+4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            
            qs += 8;
            signs += 4;
            qh += 2;
        }
#endif
    }
};

} // namespace llaminar
