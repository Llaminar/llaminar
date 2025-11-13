/**
 * @file Q8_0WeightAccessor.h
 * @brief Backward-compatible Q8_0 accessor & factory used by legacy tests.
 *
 * Tests expect:
 *   - createQ8_0Accessor(tensor [, warmup]) -> shared_ptr accessor
 *   - accessor->is_zero_copy()
 *   - accessor->get_q8_block(row, k_block_idx)
 *
 * For native Q8_0 tensors we expose zero-copy access directly to the
 * underlying blocks. For other quantized tensors (currently IQ4_NL in tests)
 * we lazily decode blocks on first access and cache them. An optional warmup
 * count triggers eager decode of the first N blocks to simulate previous
 * behavior where a micro-kernel primed its cache.
 *
 * This is intentionally lightweight; production kernels use the newer
 * WeightAccessor abstractions. We isolate test-only legacy behaviors here to
 * avoid polluting the primary kernel path.
 */
#pragma once
#include "WeightAccessor.h"
#include "../../../tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstddef>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            class Q8_0Accessor
            {
            public:
                Q8_0Accessor(TensorBase *tensor, size_t warmup_blocks = 0)
                    : tensor_(tensor)
                {
                    // Determine zero-copy eligibility
                    q8_tensor_ = dynamic_cast<Q8_0Tensor *>(tensor_);
                    if (q8_tensor_)
                    {
                        zero_copy_ = true;
                        blocks_per_row_ = q8_tensor_->shape()[1] / Q8_0Block::BLOCK_SIZE;
                        return; // Nothing else needed
                    }

                    // Currently only IQ4_NL used in tests for non-zero-copy path
                    iq4_tensor_ = dynamic_cast<IQ4_NLTensor *>(tensor_);
                    if (iq4_tensor_)
                    {
                        zero_copy_ = false;
                        blocks_per_row_ = iq4_tensor_->shape()[1] / Q8_0Block::BLOCK_SIZE;
                    }
                    else
                    {
                        // Fallback: treat as unsupported, zero blocks
                        zero_copy_ = false;
                        blocks_per_row_ = 0;
                    }

                    const size_t rows = tensor_->shape()[0];
                    const size_t total_blocks = rows * blocks_per_row_;
                    cache_.resize(total_blocks); // Uninitialized until decode
                    decoded_.assign(total_blocks, false);

                    // Optional warmup
                    if (warmup_blocks > 0)
                    {
                        size_t to_warm = std::min(warmup_blocks, total_blocks);
                        for (size_t idx = 0; idx < to_warm; ++idx)
                        {
                            size_t row = idx / blocks_per_row_;
                            size_t kb = idx % blocks_per_row_;
                            decode_block(row, kb);
                        }
                    }
                }

                bool is_zero_copy() const { return zero_copy_; }

                const Q8_0Block *get_q8_block(size_t row, size_t k_block_idx)
                {
                    if (zero_copy_)
                    {
                        // Use public raw block accessor (computes pointer internally)
                        return reinterpret_cast<const Q8_0Block *>(q8_tensor_->get_raw_block_at(row, k_block_idx));
                    }
                    // Lazy decode
                    decode_block(row, k_block_idx);
                    return &cache_[row * blocks_per_row_ + k_block_idx];
                }

                // Legacy test warmup helper: decode a range eagerly (no-op for zero-copy)
                void warmup_cache(size_t start_row, size_t end_row, size_t start_kb, size_t blocks_per_row)
                {
                    if (zero_copy_)
                        return;
                    for (size_t r = start_row; r < end_row; ++r)
                    {
                        for (size_t kb = start_kb; kb < blocks_per_row; ++kb)
                        {
                            decode_block(r, kb);
                        }
                    }
                }

            private:
                void decode_block(size_t row, size_t kb)
                {
                    if (zero_copy_)
                        return; // Should not happen
                    size_t idx = row * blocks_per_row_ + kb;
                    if (idx >= cache_.size() || decoded_[idx])
                        return;
                    // Use tensor's decode_to_q8_0 API
                    iq4_tensor_->decode_to_q8_0(row, kb, &cache_[idx]);
                    decoded_[idx] = true;
                }

                TensorBase *tensor_ = nullptr;
                IQ4_NLTensor *iq4_tensor_ = nullptr;
                Q8_0Tensor *q8_tensor_ = nullptr;
                bool zero_copy_ = false;
                size_t blocks_per_row_ = 0;
                std::vector<Q8_0Block> cache_;
                std::vector<bool> decoded_;
            };

            inline std::shared_ptr<Q8_0Accessor> createQ8_0Accessor(TensorBase *tensor, size_t warmup_blocks = 0)
            {
                return std::make_shared<Q8_0Accessor>(tensor, warmup_blocks);
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
