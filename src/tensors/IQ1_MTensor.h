/**
 * @file IQ1_MTensor.h
 * @brief IQ1_M quantized tensor (1.75 bits per weight, extreme compression with dual-scale)
 * 
 * IQ1_M (Importance Quantization 1-bit M) achieves extreme compression (18.29×) using
 * 11-bit grid lookups with dual sub-block scales for improved accuracy.
 * 
 * Block Structure (56 bytes for 256 elements):
 * - qs[32] (32 bytes): Grid indices (low 8 bits)
 * - qh[16] (16 bytes): High 3 bits of grid indices + delta sign bits
 * - scales[8] (8 bytes): Packed 3-bit sub-block scales + global FP16 scale
 * 
 * Decoding Algorithm (per 256-element block):
 * 1. Extract global FP16 scale from packed scales[] array
 * 2. Split into 8 sub-blocks of 32 elements
 * 3. For each sub-block:
 *    a. Extract two 3-bit scales (dl1, dl2) from scales[]
 *    b. Extract 4 delta signs from qh
 *    c. Extract 4 grid indices (11-bit each)
 *    d. Groups 0-1 use dl1, groups 2-3 use dl2
 *    e. Apply: y[j] = dl * (grid[j] + delta)
 * 
 * Reference: ggml-quants.c (dequantize_row_iq1_m)
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
 * @brief IQ1_M block structure (56 bytes, 256 elements)
 * 
 * Matches GGML block_iq1_m from ggml-common.h.
 */
struct IQ1_MBlock {
    uint8_t qs[32];        ///< Grid indices (low 8 bits), QK_K/8
    uint8_t qh[16];        ///< High 3 bits + delta signs, QK_K/16
    uint8_t scales[8];     ///< Packed 3-bit scales + global FP16 scale, QK_K/32
    
    static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block (QK_K)
};

static_assert(sizeof(IQ1_MBlock) == 56, "IQ1_MBlock must be 56 bytes");

/**
 * @brief IQ1_M quantized tensor (1.75 bpw, 18.29× compression)
 * 
 * Implements extreme 1-bit codebook quantization with 11-bit grid indices
 * and dual sub-block scales for improved accuracy over IQ1_S.
 */
class IQ1_MTensor : public QuantizedTensorBase {
public:
    /**
     * @brief Construct IQ1_M tensor from shape and raw data
     * 
     * @param shape Tensor dimensions (2D: [rows, cols])
     * @param raw_data Raw bytes (IQ1_M blocks)
     * @throws std::invalid_argument if shape not 2D or size mismatch
     */
    IQ1_MTensor(const std::vector<int>& shape, const std::vector<uint8_t>& raw_data)
        : shape_(shape), raw_data_(raw_data) {
        
        if (shape_.size() != 2) {
            throw std::invalid_argument("IQ1_MTensor only supports 2D tensors");
        }
        
        size_t num_elements = shape_[0] * shape_[1];
        size_t num_blocks = (num_elements + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        size_t expected_size = num_blocks * sizeof(IQ1_MBlock);
        
        if (raw_data_.size() != expected_size) {
            throw std::invalid_argument(
                "IQ1_M raw data size mismatch: expected " + std::to_string(expected_size) +
                " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
        }
    }
    
    // ========== Shape and Metadata ==========
    
    const std::vector<int>& shape() const override { return shape_; }
    int size() const override { return shape_[0] * shape_[1]; }
    int ndim() const override { return 2; }
    
    QuantType quant_type() const override { return QuantType::IQ1_M; }
    float compression_ratio() const override { return 18.29f; }
    
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
        return std::make_shared<IQ1_MTensor>(shape_, raw_data_);
    }
    
    void copy_from(const TensorBase&) override {
        throw std::runtime_error("IQ1_MTensor::copy_from not supported - quantization is lossy");
    }
    
    // ========== Streaming Decode API ==========
    
    void decodeRow(size_t row_idx, float* buffer) const override {
        if (row_idx >= static_cast<size_t>(shape_[0])) {
            throw std::out_of_range("IQ1_MTensor: Row index out of bounds");
        }
        
        int cols = shape_[1];
        size_t global_start = row_idx * cols;
        size_t global_end = global_start + cols;
        
        size_t start_block = global_start / IQ1_MBlock::BLOCK_SIZE;
        size_t end_block = (global_end - 1) / IQ1_MBlock::BLOCK_SIZE;
        
        const IQ1_MBlock* blocks = reinterpret_cast<const IQ1_MBlock*>(raw_data_.data());
        
        if (start_block == end_block) {
            // Single block case
            size_t offset_in_block = global_start % IQ1_MBlock::BLOCK_SIZE;
            float temp[IQ1_MBlock::BLOCK_SIZE];
            decodeBlock(blocks[start_block], temp);
            std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
        } else {
            // Multi-block case
            size_t offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
                float temp[IQ1_MBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);
                
                size_t block_start = block_idx * IQ1_MBlock::BLOCK_SIZE;
                size_t copy_start = std::max(global_start, block_start) - block_start;
                size_t copy_end = std::min(global_end, block_start + IQ1_MBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;
                
                std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                offset += copy_count;
            }
        }
    }
    
    void decodeSpan(size_t offset, size_t count, float* buffer) const override {
        if (offset + count > element_count()) {
            throw std::out_of_range("IQ1_MTensor::decodeSpan: range exceeds tensor bounds");
        }
        
        size_t start_block = offset / IQ1_MBlock::BLOCK_SIZE;
        size_t end_block = (offset + count - 1) / IQ1_MBlock::BLOCK_SIZE;
        
        const IQ1_MBlock* blocks = reinterpret_cast<const IQ1_MBlock*>(raw_data_.data());
        
        size_t buffer_offset = 0;
        for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx) {
            float temp[IQ1_MBlock::BLOCK_SIZE];
            decodeBlock(blocks[block_idx], temp);
            
            size_t block_start = block_idx * IQ1_MBlock::BLOCK_SIZE;
            size_t copy_start = std::max(offset, block_start) - block_start;
            size_t copy_end = std::min(offset + count, block_start + IQ1_MBlock::BLOCK_SIZE) - block_start;
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
            56,   // bytes_per_block
            2,    // scale_count (global + 16 sub-block scales)
            2,    // bits_per_value (1.75 bpw)
            false // is_k_quant
        };
        return desc;
    }
    
private:
    std::vector<int> shape_;              ///< Tensor dimensions
    std::vector<uint8_t> raw_data_;       ///< Raw quantized data
    
    static constexpr float IQ1S_DELTA = 0.125f; ///< Delta shift value (same as IQ1_S)
    
    /**
     * @brief Decode one IQ1_M block (256 elements) to FP32
     * 
     * Own implementation using iq1s_grid from IQQuantTables.h.
     * Based on GGML dequantize_row_iq1_m from ggml-quants.c.
     * 
     * Algorithm:
     * 1. Extract global FP16 scale from packed scales[] array:
     *    scale_u16 = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000)
     * 2. Process 8 sub-blocks of 32 elements:
     *    a. Extract two 3-bit scales (dl1, dl2) from scales[]
     *    b. Extract 4 grid indices (11-bit each)
     *    c. Extract 4 delta signs from qh
     *    d. Groups 0-1 use dl1, groups 2-3 use dl2
     *    e. Apply: y[j] = dl * (grid[j] + delta)
     * 
     * @param block Input IQ1_M block
     * @param output Output buffer (must have space for 256 floats)
     */
    static void decodeBlock(const IQ1_MBlock& block, float* output) {
        // Extract global FP16 scale from packed scales array
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(block.scales);
        uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                             ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = simd::fp16_to_fp32(scale_u16);
        
        const uint8_t* qs = block.qs;
        const uint8_t* qh = block.qh;
        
        float* y = output;
        
        // Process 8 sub-blocks of 32 elements
        for (int ib = 0; ib < 8; ++ib) {  // QK_K/32 = 256/32 = 8
            // Extract two 3-bit scales per sub-block (6 bits total packed in uint16)
            // dl1 for groups 0-1, dl2 for groups 2-3
            const float dl1 = d * (2.0f * static_cast<float>((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1.0f);
            const float dl2 = d * (2.0f * static_cast<float>((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1.0f);
            
            // Extract 4 grid indices (11-bit each)
            uint16_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            
            // Extract 4 delta signs
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            
            // Process first 2 groups (use dl1)
            for (int l = 0; l < 2; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(&iq1s_grid[idx[l]]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (static_cast<float>(grid[j]) + delta[l]);
                }
                y += 8;
            }
            
            // Process last 2 groups (use dl2)
            for (int l = 2; l < 4; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(&iq1s_grid[idx[l]]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (static_cast<float>(grid[j]) + delta[l]);
                }
                y += 8;
            }
            
            qs += 4;
            qh += 2;
        }
    }
};

} // namespace llaminar
