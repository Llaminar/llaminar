/**
 * @file IQ1_STensor.h
 * @brief IQ1_S quantized tensor (1.5625 bits per weight, extreme compression)
 * 
 * IQ1_S (Importance Quantization 1-bit S) achieves extreme compression (20.48×) using
 * 11-bit grid lookups with delta shifting for fine-grained value representation.
 * 
 * Block Structure (50 bytes for 256 elements):
 * - d (2 bytes): FP16 scale factor
 * - qs[32] (32 bytes): Grid indices (low 8 bits)
 * - qh[8] (16 bytes): High 3 bits of grid indices + delta sign bits (uint16)
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Split into 8 sub-blocks of 32 elements
 * 2. For each sub-block:
 *    a. Extract 3-bit scale: (qh[ib] >> 12) & 7
 *    b. Compute: dl = d * (2*scale + 1)
 *    c. Extract delta sign: (qh[ib] & 0x8000) ? -0.125 : +0.125
 *    d. For each of 4 groups of 8 elements:
 *       - Combine qs + qh to form 11-bit grid index
 *       - Lookup: iq1s_grid[index] → 8 int8 values
 *       - Apply: y[j] = dl * (grid[j] + delta)
 * 
 * Reference: ggml-quants.c (dequantize_row_iq1_s)
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
 * @brief IQ1_S block structure (50 bytes, 256 elements)
 * 
 * Matches GGML block_iq1_s from ggml-common.h.
 */
struct IQ1_SBlock {
    uint16_t d;            ///< FP16 scale factor
    uint8_t qs[32];        ///< Grid indices (low 8 bits), QK_K/8
    uint16_t qh[8];        ///< High 3 bits + delta sign + 3-bit scale, QK_K/32
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block (QK_K)
};

static_assert(sizeof(IQ1_SBlock) == 50, "IQ1_SBlock must be 50 bytes");

/**
 * @brief IQ1_S quantized tensor (1.5625 bpw, 20.48× compression)
 * 
 * Implements extreme 1-bit codebook quantization with 11-bit grid indices
 * and delta shifting for improved accuracy.
 */
class IQ1_STensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ1_S tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ1_S blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ1_STensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ1_STensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ1_SBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ1_S raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ1_S; }
    float compression_ratio() const override { return 20.48f; }
    
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
            // Convert FP32 to BF16
            for (int col = 0; col < cols; ++col) {
                bf16_dst[row * cols + col] = bfloat16::from_float(temp[col]);
            }
        }
    }
    
    std::shared_ptr<TensorBase> copy() const override {
        return std::make_shared<IQ1_STensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ1_STensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        if (row_idx >= static_cast<size_t>(shape_[0])) {
            throw std::out_of_range("IQ1_STensor: Row index out of bounds");
        }
        
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ1_SBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ1_SBlock::BLOCK_SIZE;
        
        const IQ1_SBlock* blocks = reinterpret_cast<const IQ1_SBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ1_SBlock::BLOCK_SIZE;
            float temp[IQ1_SBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ1_SBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ1_SBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ1_SBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ1_STensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ1_SBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ1_SBlock::BLOCK_SIZE;
        
        const IQ1_SBlock* blocks = reinterpret_cast<const IQ1_SBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ1_SBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ1_SBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ1_SBlock::BLOCK_SIZE) - block_start;
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
            50,   // bytes_per_block
            1,    // scale_count (one main scale + 8 sub-block scales in qh)
            2,    // bits_per_value (1.5625 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
    static constexpr float IQ1S_DELTA = 0.125f; ///< Delta shift value
    
    /**
     * @brief Decode one IQ1_S block (256 elements) to FP32
     * 
     * Own implementation using iq1s_grid from IQQuantTables.h.
     * Based on GGML dequantize_row_iq1_s from ggml-quants.c.
     * 
     * Algorithm:
     * 1. Extract FP16 scale d
     * 2. Process 8 sub-blocks of 32 elements (QK_K/32):
     *    a. Extract 3-bit scale: (qh[ib] >> 12) & 7
     *    b. Compute: dl = d * (2*scale + 1)
     *    c. Extract delta sign: (qh[ib] & 0x8000) ? -IQ1S_DELTA : +IQ1S_DELTA
     *    d. For each of 4 groups of 8 elements:
     *       - Combine: grid_idx = qs[l] | (((qh[ib] >> 3*l) & 7) << 8)  [11-bit index]
     *       - Lookup: iq1s_grid[grid_idx] → cast to 8 int8 values
     *       - Apply: y[j] = dl * (grid[j] + delta)
     * 
     * @param block Input IQ1_S block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ1_SBlock& block, float* output) {
        // Extract FP16 scale
        const float d = simd::fp16_to_fp32(block.d);
        
        const uint8_t* qs = block.qs;
        const uint16_t* qh = block.qh;
        
        float* y = output;
        
        // Process 8 sub-blocks of 32 elements
        for (int ib = 0; ib < 8; ++ib) {  // QK_K/32 = 256/32 = 8
            // Extract 3-bit sub-block scale (bits [12:14])
            const float dl = d * (2.0f * static_cast<float>((qh[ib] >> 12) & 7) + 1.0f);
            
            // Extract delta sign (bit 15)
            const float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            
            // Process 4 groups of 8 elements
            for (int l = 0; l < 4; ++l) {
                // Combine qs (8 bits) + qh (3 bits) to form 11-bit grid index
                const uint16_t grid_idx = qs[l] | (((qh[ib] >> (3*l)) & 7) << 8);
                
                // Lookup grid entry (uint64_t contains 8 int8 values)
                const int8_t* grid = reinterpret_cast<const int8_t*>(&iq1s_grid[grid_idx]);
                
                // Decode 8 elements
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (static_cast<float>(grid[j]) + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
};

} // namespace llaminar
