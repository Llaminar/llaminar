/**
 * @file Q16_1Tensor.cpp
 * @brief Q16_1 quantized tensor implementation (16-bit with pre-computed sum, high-precision residual format)
 * @author David Sanftenberg
 *
 * Like Q8_1 but with 256× more precision (int16 vs int8). Designed for residual stream
 * where quantization error accumulation is critical.
 */

// Suppress deprecation warnings for internal usage of mutable_q16_1_blocks()
// This is the implementation file - we know what we're doing here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "CPUTensors.h"
#include "../../utils/Logger.h"
#include "../../utils/Assertions.h"
#include "../../kernels/cpu/ops/CPURoPEKernelT.h"
#include "../SIMDHelpers.h"
#include "../FP16Utils.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <atomic>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    // ===== Constructors =====

    Q16_1Tensor::Q16_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr),
          is_mutable_(false), cache_dirty_(false)
    {
        LOG_TRACE("[Q16_1Tensor] Creating IMMUTABLE tensor from raw_data, shape=[" << shape[0] << "," << shape[1]
                                                                                   << "], raw_data.size()=" << raw_data.size());

        if (shape.empty())
        {
            throw std::invalid_argument("Q16_1Tensor: shape cannot be empty");
        }

        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q16_1Tensor: expected 2D shape for Q16_1 storage");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const size_t n_blocks = rows * blocks_per_row;
        const size_t expected_bytes = n_blocks * sizeof(Q16_1Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q16_1Tensor: insufficient raw data (" +
                                        std::to_string(raw_data_.size()) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    Q16_1Tensor::Q16_1Tensor(const std::vector<size_t> &shape, const Q16_1Block *blocks, size_t num_blocks, DeviceId device)
        : shape_(shape), is_view_(false), raw_data_(), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(device), device_blocks_(nullptr),
          is_mutable_(true), cache_dirty_(false)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q16_1Tensor: shape cannot be empty");
        }
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q16_1Tensor: expected 2D shape for Q16_1 storage");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row_ = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const size_t expected_blocks = rows * blocks_per_row_;

        if (num_blocks < expected_blocks)
        {
            throw std::invalid_argument("Q16_1Tensor: insufficient blocks (" +
                                        std::to_string(num_blocks) + ", expected " +
                                        std::to_string(expected_blocks) + ")");
        }

        const size_t required_bytes = expected_blocks * sizeof(Q16_1Block);
        raw_data_.resize(required_bytes);
        std::memcpy(raw_data_.data(), blocks, required_bytes);

        LOG_TRACE("[Q16_1Tensor] Created from Q16_1Block* shape=[" << shape[0] << "," << shape[1]
                                                                   << "], blocks=" << expected_blocks);
    }

    Q16_1Tensor::Q16_1Tensor(const Q16_1Tensor &other)
        : shape_(other.shape_), is_view_(false), raw_data_(other.raw_data_), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(other.device_), device_blocks_(nullptr),
          is_mutable_(other.is_mutable_), cache_dirty_(false)
    {
        if (other.is_view_)
        {
            const size_t rows = shape_[0];
            const size_t cols = shape_[1];
            const size_t blocks_per_row_ = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
            const size_t n_blocks = rows * blocks_per_row_;
            const size_t required_bytes = n_blocks * sizeof(Q16_1Block);

            raw_data_.resize(required_bytes);
            std::memcpy(raw_data_.data(), other.raw_data_ptr_ + other.view_byte_offset_, required_bytes);
        }

        if (!other.dequant_cache_.empty())
        {
            dequant_cache_ = other.dequant_cache_;
        }

        LOG_TRACE("[Q16_1Tensor] Copy constructed shape=[" << shape_[0] << "," << shape_[1]
                                                           << "], is_mutable=" << is_mutable_);
    }

    Q16_1Tensor::Q16_1Tensor(const std::vector<size_t> &shape, DeviceId device)
        : shape_(shape), is_view_(false), raw_data_(), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(device), device_blocks_(nullptr),
          is_mutable_(true), cache_dirty_(false)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q16_1Tensor: shape cannot be empty");
        }
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q16_1Tensor: mutable activation buffer requires 2D shape");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row_ = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const size_t n_blocks = rows * blocks_per_row_;
        const size_t required_bytes = n_blocks * sizeof(Q16_1Block);

        raw_data_.resize(required_bytes, 0);

        LOG_TRACE("[Q16_1Tensor] Created MUTABLE tensor shape=[" << shape[0] << "," << shape[1]
                                                                 << "], blocks=" << n_blocks << ", bytes=" << required_bytes);
    }

    Q16_1Tensor::Q16_1Tensor(const std::vector<size_t> &shape, Q16BlockSize block_size, DeviceId device)
        : shape_(shape), is_view_(false), raw_data_(), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(device), device_blocks_(nullptr),
          is_mutable_(true), cache_dirty_(false), block_size_(block_size)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q16_1Tensor: shape cannot be empty");
        }
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q16_1Tensor: mutable activation buffer requires 2D shape");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t bs = static_cast<size_t>(block_size);
        const size_t blocks_per_row_ = (cols + bs - 1) / bs;
        const size_t n_blocks = rows * blocks_per_row_;

        // Calculate bytes based on block size
        size_t block_bytes = 0;
        switch (block_size)
        {
        case Q16BlockSize::BLOCK_32:
            block_bytes = sizeof(Q16_1Block);
            break;
        case Q16BlockSize::BLOCK_64:
            block_bytes = sizeof(Q16_1Block_64);
            break;
        case Q16BlockSize::BLOCK_128:
            block_bytes = sizeof(Q16_1Block_128);
            break;
        }
        const size_t required_bytes = n_blocks * block_bytes;

        raw_data_.resize(required_bytes, 0);

        LOG_TRACE("[Q16_1Tensor] Created MUTABLE tensor shape=[" << shape[0] << "," << shape[1]
                                                                 << "], block_size=" << static_cast<int>(block_size)
                                                                 << ", blocks=" << n_blocks << ", bytes=" << required_bytes);
    }

    Q16_1Tensor::Q16_1Tensor(const std::vector<size_t> &shape,
                             const uint8_t *parent_raw_data,
                             size_t byte_offset,
                             std::shared_ptr<CPUTensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr),
          is_mutable_(false), cache_dirty_(false)
    {
        if (auto parent_q16_1 = std::dynamic_pointer_cast<Q16_1Tensor>(parent))
        {
            is_mutable_ = parent_q16_1->is_mutable_;
        }

        LOG_TRACE("[Q16_1Tensor] Creating VIEW tensor shape=[" << shape[0] << "," << shape[1]
                                                               << "], is_mutable=" << is_mutable_ << " (inherited from parent)");
    }

    Q16_1Tensor::~Q16_1Tensor()
    {
        // Destructor
    }

    // ===== TensorBase Interface =====

    const float *Q16_1Tensor::data() const
    {
        // Q16_1 doesn't throw like Q8_1 - it's a high-precision format where FP32 access is expected
        return fp32_data();
    }

    const float *Q16_1Tensor::fp32_data() const
    {
        assertValid("Q16_1Tensor::fp32_data");

        if (is_view_ && is_mutable_ && parent_)
        {
            auto parent_q16_1 = std::dynamic_pointer_cast<Q16_1Tensor>(parent_);
            if (parent_q16_1)
            {
                size_t K = shape_[1];
                size_t blocks_per_row_ = (K + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
                size_t block_offset = view_byte_offset_ / sizeof(Q16_1Block);
                size_t row_offset = block_offset / blocks_per_row_;
                size_t element_offset = row_offset * K;

                return parent_q16_1->fp32_data() + element_offset;
            }
        }

        const size_t rows = shape_.size() > 0 ? shape_[0] : 0;
        const size_t cols = shape_.size() > 1 ? shape_[1] : 0;
        const size_t total_elements = rows * cols;

        if (total_elements == 0)
        {
            return nullptr;
        }

        if (!is_mutable_ && !dequant_cache_.empty())
        {
            return dequant_cache_.data();
        }

        if (dequant_cache_.size() != total_elements)
        {
            dequant_cache_.assign(total_elements, 0.0f);
        }

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t blocks_per_row = (cols + bs - 1) / bs;

        // Dispatch based on actual block size
        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
        {
            const Q16_1Block *blocks = reinterpret_cast<const Q16_1Block *>(data_ptr);
#pragma omp parallel for collapse(2) if (total_elements > 10000)
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t col0 = b * Q16_1Block::BLOCK_SIZE;
                    if (col0 >= cols)
                        continue;

                    const Q16_1Block &block = blocks[r * blocks_per_row + b];

                    alignas(64) float tmp[Q16_1Block::BLOCK_SIZE];
                    simd::decode_q16_block_to_fp32<Q16_1Block>(block, tmp);

                    const size_t copy = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), cols - col0);
                    std::memcpy(&dequant_cache_[r * cols + col0], tmp, copy * sizeof(float));
                }
            }
            break;
        }
        case Q16BlockSize::BLOCK_64:
        {
            const Q16_1Block_64 *blocks = reinterpret_cast<const Q16_1Block_64 *>(data_ptr);
#pragma omp parallel for collapse(2) if (total_elements > 10000)
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t col0 = b * Q16_1Block_64::BLOCK_SIZE;
                    if (col0 >= cols)
                        continue;

                    const Q16_1Block_64 &block = blocks[r * blocks_per_row + b];

                    alignas(64) float tmp[Q16_1Block_64::BLOCK_SIZE];
                    simd::decode_q16_block_to_fp32<Q16_1Block_64>(block, tmp);

                    const size_t copy = std::min(static_cast<size_t>(Q16_1Block_64::BLOCK_SIZE), cols - col0);
                    std::memcpy(&dequant_cache_[r * cols + col0], tmp, copy * sizeof(float));
                }
            }
            break;
        }
        case Q16BlockSize::BLOCK_128:
        {
            const Q16_1Block_128 *blocks = reinterpret_cast<const Q16_1Block_128 *>(data_ptr);
#pragma omp parallel for collapse(2) if (total_elements > 10000)
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t col0 = b * Q16_1Block_128::BLOCK_SIZE;
                    if (col0 >= cols)
                        continue;

                    const Q16_1Block_128 &block = blocks[r * blocks_per_row + b];

                    alignas(64) float tmp[Q16_1Block_128::BLOCK_SIZE];
                    simd::decode_q16_block_to_fp32<Q16_1Block_128>(block, tmp);

                    const size_t copy = std::min(static_cast<size_t>(Q16_1Block_128::BLOCK_SIZE), cols - col0);
                    std::memcpy(&dequant_cache_[r * cols + col0], tmp, copy * sizeof(float));
                }
            }
            break;
        }
        }

        cache_dirty_ = false;
        return dequant_cache_.data();
    }

    float *Q16_1Tensor::mutable_data()
    {
        // Unlike Q8_1, Q16_1 allows mutable FP32 access since it's a high-precision format
        // and the conversion overhead is acceptable for this use case
        if (!is_mutable_)
        {
            throw std::runtime_error("Q16_1Tensor::mutable_data() called on immutable tensor");
        }

        const size_t total_elements = element_count();
        if (dequant_cache_.size() != total_elements)
        {
            dequant_cache_.resize(total_elements);
            // Initialize from current Q16_1 blocks
            fp32_data();
        }

        cache_dirty_ = true;
        return dequant_cache_.data();
    }

    void Q16_1Tensor::release_raw_data()
    {
        if (!is_view_ && !raw_data_released_)
        {
            raw_data_.clear();
            raw_data_.shrink_to_fit();
            raw_data_released_ = true;
        }
    }

    // ============================================================================
    // ITensorGemmTileDataProvider interface - Variable block size support
    // ============================================================================

    namespace
    {
        /**
         * @brief Templated decode helper for variable Q16 block sizes
         */
        template <typename BlockType>
        void decode_q16_block(const void *block_ptr, float *output)
        {
            const BlockType &block = *static_cast<const BlockType *>(block_ptr);
            const float scale = block.d;
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                output[i] = static_cast<float>(block.qs[i]) * scale;
            }
        }
    } // anonymous namespace

    void Q16_1Tensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t blocks_per_row_ = (shape_[1] + bs - 1) / bs;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();

        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
        {
            const Q16_1Block *blocks = reinterpret_cast<const Q16_1Block *>(data_ptr);
            decode_q16_block<Q16_1Block>(&blocks[row_idx * blocks_per_row_ + k_block_offset], output);
            break;
        }
        case Q16BlockSize::BLOCK_64:
        {
            const Q16_1Block_64 *blocks = reinterpret_cast<const Q16_1Block_64 *>(data_ptr);
            decode_q16_block<Q16_1Block_64>(&blocks[row_idx * blocks_per_row_ + k_block_offset], output);
            break;
        }
        case Q16BlockSize::BLOCK_128:
        {
            const Q16_1Block_128 *blocks = reinterpret_cast<const Q16_1Block_128 *>(data_ptr);
            decode_q16_block<Q16_1Block_128>(&blocks[row_idx * blocks_per_row_ + k_block_offset], output);
            break;
        }
        }
    }

    const void *Q16_1Tensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t blocks_per_row_ = (shape_[1] + bs - 1) / bs;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();

        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
        {
            const Q16_1Block *blocks = reinterpret_cast<const Q16_1Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row_ + k_block_offset];
        }
        case Q16BlockSize::BLOCK_64:
        {
            const Q16_1Block_64 *blocks = reinterpret_cast<const Q16_1Block_64 *>(data_ptr);
            return &blocks[row_idx * blocks_per_row_ + k_block_offset];
        }
        case Q16BlockSize::BLOCK_128:
        {
            const Q16_1Block_128 *blocks = reinterpret_cast<const Q16_1Block_128 *>(data_ptr);
            return &blocks[row_idx * blocks_per_row_ + k_block_offset];
        }
        default:
            return nullptr;
        }
    }

    bool Q16_1Tensor::copyFrom(const CPUTensorBase *src)
    {
        if (!src)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom] Source is null");
            return false;
        }

        if (!is_mutable_)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom] Immutable Q16_1 tensors do not support copyFrom");
            return false;
        }

        const auto &src_shape = src->shape();
        if (src_shape.size() != shape_.size())
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom] Shape dimension mismatch");
            return false;
        }
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            if (src_shape[i] != shape_[i])
            {
                LOG_ERROR("[Q16_1Tensor::copyFrom] Shape mismatch at dim " << i);
                return false;
            }
        }

        const float *src_data = src->data();
        if (!src_data)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom] Source data is null");
            return false;
        }

        // Quantize FP32 source directly to Q16_1 blocks
        const size_t total_elements = element_count();
        const size_t n_blocks = (total_elements + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;

        Q16_1Block *blocks = mutable_q16_1_blocks();

#pragma omp parallel for if (n_blocks > 4)
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q16_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), total_elements - offset);

            Q16_1Block &block = blocks[block_idx];

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src_data[offset + i]));
            }

            // Compute scale factor (d = max_abs / 32767 for int16 range)
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d; // FP32 scale, no conversion needed

            // Quantize AND compute sum simultaneously
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float val = src_data[offset + i];
                    const float scaled = val * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::round(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            // Zero-fill partial block tail
            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            // Store sum (int32 range: 32 × int16 values sum to [-1048544, 1048544], safe for INT32)
            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

        return true;
    }

    std::shared_ptr<CPUTensorBase> Q16_1Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        if (new_shape.size() != 2)
        {
            LOG_ERROR("[Q16_1Tensor::create_view] ERROR: View must be 2D (got "
                      << new_shape.size() << "D)");
            return nullptr;
        }

        if (new_shape[1] != shape_[1])
        {
            LOG_ERROR("[Q16_1Tensor::create_view] ERROR: View must preserve K dimension");
            return nullptr;
        }

        size_t K = shape_[1];

        if (offset % K != 0)
        {
            LOG_ERROR("[Q16_1Tensor::create_view] ERROR: Offset must be row-aligned");
            return nullptr;
        }

        size_t start_row = offset / K;
        size_t view_rows = new_shape[0];
        if (start_row + view_rows > shape_[0])
        {
            LOG_ERROR("[Q16_1Tensor::create_view] ERROR: View exceeds parent bounds");
            return nullptr;
        }

        size_t blocks_per_row_ = (K + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        size_t block_offset = start_row * blocks_per_row_;
        size_t byte_offset = block_offset * sizeof(Q16_1Block);

        std::shared_ptr<CPUTensorBase> root_parent;
        const uint8_t *root_data_ptr;
        size_t root_byte_offset;

        if (is_view_)
        {
            root_parent = parent_;
            root_data_ptr = raw_data_ptr_;
            root_byte_offset = view_byte_offset_ + byte_offset;
        }
        else
        {
            try
            {
                root_parent = shared_from_this();
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[Q16_1Tensor::create_view] ERROR: shared_from_this() failed");
                return nullptr;
            }
            root_data_ptr = raw_data_.data();
            root_byte_offset = byte_offset;
        }

        return std::shared_ptr<Q16_1Tensor>(new Q16_1Tensor(
            new_shape, root_data_ptr, root_byte_offset, root_parent));
    }

    // ===== Kernel Factory Methods =====
    // Q16_1 is a high-precision format primarily used for residual stream storage.
    // For most operations, it dequantizes to FP32 internally.
    // Native Q16_1 kernels can be added later as needed.

    std::unique_ptr<ITensorGemm> Q16_1Tensor::createGemm()
    {
        // Q16_1 GEMM: Dequantize to FP32 and use FP32 GEMM kernel
        // For now, return nullptr - caller should use fp32_data() and FP32 kernel
        LOG_WARN("[Q16_1Tensor::createGemm] No native Q16_1 GEMM kernel. Use fp32_data() with FP32 GEMM.");
        return nullptr;
    }

    std::unique_ptr<ITensorRoPE> Q16_1Tensor::createRoPE()
    {
        // Use native Q16_1 in-place RoPE kernel
        return std::make_unique<CPURoPEKernelT<ActivationPrecision::Q16_1>>();
    }

    std::unique_ptr<ITensorSwiGLU> Q16_1Tensor::createSwiGLU()
    {
        LOG_WARN("[Q16_1Tensor::createSwiGLU] No native Q16_1 SwiGLU kernel. Use FP32 path.");
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> Q16_1Tensor::createSoftmax()
    {
        LOG_WARN("[Q16_1Tensor::createSoftmax] No native Q16_1 Softmax kernel. Use FP32 path.");
        return nullptr;
    }

    std::unique_ptr<ITensorRMSNorm> Q16_1Tensor::createRMSNorm()
    {
        LOG_WARN("[Q16_1Tensor::createRMSNorm] No native Q16_1 RMSNorm kernel. Use FP32 path.");
        return nullptr;
    }

    std::unique_ptr<ITensorAttention> Q16_1Tensor::createAttention()
    {
        LOG_WARN("[Q16_1Tensor::createAttention] No native Q16_1 Attention kernel. Use FP32 path.");
        return nullptr;
    }

    std::unique_ptr<ITensorEmbedding> Q16_1Tensor::createEmbedding()
    {
        LOG_WARN("[Q16_1Tensor::createEmbedding] No native Q16_1 Embedding kernel. Use FP32 path.");
        return nullptr;
    }

    // ===== IActivationTensor Interface =====

    bool Q16_1Tensor::applyRoPE(
        float *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)use_bf16;
        (void)mpi_ctx;

        // Q16_1 In-Place RoPE: Uses native Q16_1 kernel with dequant→rotate→requant internally.
        // K is passed as float* but is actually Q16_1Block* (same pattern as Q8_1Tensor).
        auto kernel = createRoPE();
        if (!kernel)
        {
            LOG_ERROR("[Q16_1Tensor::applyRoPE] Failed to create RoPE kernel");
            return false;
        }

        // Get Q pointer (this tensor) - use mutable_q16_1_blocks() to invalidate cache
        void *Q_ptr = mutable_q16_1_blocks();

        // K is passed as float* but is actually Q16_1 block data
        // Note: K's cache should have been invalidated by the caller via mutable_q16_1_blocks()
        void *K_ptr = (void *)K;

        bool success = kernel->apply_q16_1(
            Q_ptr, K_ptr,
            position_ids,
            seq_len, n_heads, n_kv_heads, head_dim,
            rope_theta,
            device_idx);

        // Cache was already cleared by mutable_q16_1_blocks() call above
        return success;
    }

    bool Q16_1Tensor::applyRMSNorm(
        const float *gamma,
        int seq_len,
        int d_model,
        float eps,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;
        (void)gamma;
        (void)seq_len;
        (void)d_model;
        (void)eps;

        // Q16_1 doesn't support in-place RMSNorm - use createRMSNorm() instead
        LOG_ERROR("[Q16_1Tensor::applyRMSNorm] Not supported for Q16_1 tensors");
        return false;
    }

    bool Q16_1Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        if (!accum)
        {
            LOG_ERROR("[Q16_1Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[Q16_1Tensor::from_int32_with_scales] tensor must be 2D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[Q16_1Tensor::from_int32_with_scales] shape mismatch");
            return false;
        }

        const size_t total_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t total_blocks = (total_elements + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;

        // First convert INT32 accumulator to FP32
        std::vector<float> temp_fp32(total_elements);
        simd::requantize_int32_matrix_to_fp32(
            accum,
            temp_fp32.data(),
            rows,
            cols,
            row_scales,
            col_scales,
            bias);

        // Then quantize FP32 to Q16_1
        Q16_1Block *blocks = mutable_q16_1_blocks();

#pragma omp parallel for if (total_blocks > 4)
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q16_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), total_elements - offset);

            Q16_1Block &block = blocks[block_idx];

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d; // FP32 scale, no conversion needed

            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = temp_fp32[offset + i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

        dequant_cache_.clear();
        return true;
    }

    ActivationPack Q16_1Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        // Dequant to FP32, then use base class method
        return pack_activation_rows_to_int8(rows, cols);
    }

    bool Q16_1Tensor::quantize_to_q8_1(void *q8_1_buffer, int m, int k) const
    {
        if (!q8_1_buffer)
        {
            LOG_ERROR("[Q16_1Tensor::quantize_to_q8_1] Output buffer is null");
            return false;
        }

        const size_t total_elements = static_cast<size_t>(m) * static_cast<size_t>(k);
        const size_t n_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

        // First dequantize Q16_1 to FP32
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        // Then quantize FP32 to Q8_1
        Q8_1Block *out_blocks = static_cast<Q8_1Block *>(q8_1_buffer);

#pragma omp parallel for if (n_blocks > 4)
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q8_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q8_1Block::BLOCK_SIZE), total_elements - offset);

            Q8_1Block &block = out_blocks[block_idx];

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            const float d = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 0.0f;
            block.d = fp32_to_fp16(d);

            int32_t sum_i32 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = temp_fp32[offset + i] * inv_d;
                    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
                    const int8_t quantized = static_cast<int8_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i32 += static_cast<int32_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            for (size_t i = count; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int16_t>(sum_i32);
        }

        return true;
    }

    bool Q16_1Tensor::quantize_from_cache()
    {
        if (!cache_dirty_ || dequant_cache_.empty())
        {
            return true; // Nothing to do
        }

        const size_t total_elements = element_count();
        const size_t n_blocks = (total_elements + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;

        Q16_1Block *blocks = mutable_q16_1_blocks();

#pragma omp parallel for if (n_blocks > 4)
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q16_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), total_elements - offset);

            Q16_1Block &block = blocks[block_idx];

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(dequant_cache_[offset + i]));
            }

            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d; // FP32 scale, no conversion needed

            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = dequant_cache_[offset + i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

        cache_dirty_ = false;
        return true;
    }

    // ===== Format Conversion Methods =====

    void Q16_1Tensor::to_bf16(uint16_t *dst) const
    {
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(temp_fp32[i]);
        }
    }

    void Q16_1Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(temp_fp32[i]);
        }
    }

    void Q16_1Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (size_t i = 0; i < count; ++i)
            {
                const float val = temp_fp32[offset + i] * scale;
                const float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst_int8[offset + i] = static_cast<int8_t>(std::round(clamped));
            }

            for (size_t i = count; i < block_size; ++i)
            {
                dst_int8[offset + i] = 0;
            }
        }
    }

    void Q16_1Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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
        const size_t blocks_per_row_ = (cols + block_size() - 1) / block_size();

        for (size_t kb = 0; kb < blocks_per_row_; ++kb)
        {
            const size_t offset = kb * block_size();
            const size_t count = std::min(block_size(), cols - offset);

            float temp[256];
            decode_block_at(row_idx, kb, temp);

            for (size_t i = 0; i < count; ++i)
            {
                buffer[offset + i] = temp[i];
            }
        }
    }

    void Q16_1Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        std::vector<float> temp_fp32(element_count());
        to_fp32(temp_fp32.data());
        std::memcpy(buffer, temp_fp32.data() + offset, count * sizeof(float));
    }

    // ===== Q16_1-Specific Methods =====

    std::shared_ptr<Q8_1Tensor> Q16_1Tensor::to_q8_1() const
    {
        const size_t rows = shape_[0];
        const size_t cols = shape_[1];
        const size_t total_elements = rows * cols;
        const size_t n_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

        // Allocate Q8_1 tensor
        auto q8_1_tensor = std::make_shared<Q8_1Tensor>(shape_, device_);

        // Dequantize Q16_1 to FP32
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        // Quantize FP32 to Q8_1
        Q8_1Block *out_blocks = q8_1_tensor->mutable_typed_data();

#pragma omp parallel for if (n_blocks > 4)
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q8_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q8_1Block::BLOCK_SIZE), total_elements - offset);

            Q8_1Block &block = out_blocks[block_idx];

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            const float d = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 0.0f;
            block.d = fp32_to_fp16(d);

            int32_t sum_i32 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = temp_fp32[offset + i] * inv_d;
                    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
                    const int8_t quantized = static_cast<int8_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i32 += static_cast<int32_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            for (size_t i = count; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int16_t>(sum_i32);
        }

        return q8_1_tensor;
    }

    // ===== Block Decode Methods =====

    void Q16_1Tensor::decodeBlockScalar(const Q16_1Block &block, float *output)
    {
        const float scale = block.d; // FP32 scale, no conversion needed
        for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
        {
            output[i] = scale * static_cast<float>(block.qs[i]);
        }
    }

    void Q16_1Tensor::decodeBlock(const Q16_1Block &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        decodeBlockScalar(block, output);
#endif
    }

#if defined(__AVX512F__)
    void Q16_1Tensor::decodeBlockAVX512(const Q16_1Block &block, float *output)
    {
        const float scale = block.d; // FP32 scale, no conversion needed
        const __m512 vscale = _mm512_set1_ps(scale);

        // Process 16 int16 values at a time (2 iterations for 32 elements)
        for (size_t i = 0; i < 2; ++i)
        {
            // Load 16 int16 values and convert to int32
            __m256i vi16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&block.qs[i * 16]));
            __m512i vi32 = _mm512_cvtepi16_epi32(vi16);

            // Convert to float and scale
            __m512 vf = _mm512_cvtepi32_ps(vi32);
            vf = _mm512_mul_ps(vf, vscale);

            // Store result
            _mm512_storeu_ps(&output[i * 16], vf);
        }
    }
#endif

#if defined(__AVX2__)
    void Q16_1Tensor::decodeBlockAVX2(const Q16_1Block &block, float *output)
    {
        const float scale = block.d; // FP32 scale, no conversion needed
        const __m256 vscale = _mm256_set1_ps(scale);

        // Process 8 int16 values at a time (4 iterations for 32 elements)
        for (size_t i = 0; i < 4; ++i)
        {
            // Load 8 int16 values
            __m128i vi16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block.qs[i * 8]));

            // Sign-extend int16 → int32 (8 elements)
            __m256i vi32 = _mm256_cvtepi16_epi32(vi16);

            // Convert int32 → float
            __m256 vf = _mm256_cvtepi32_ps(vi32);

            // Multiply by scale
            vf = _mm256_mul_ps(vf, vscale);

            // Store result
            _mm256_storeu_ps(&output[i * 8], vf);
        }
    }
#endif

    // ===== Static Quantization Method =====

    std::shared_ptr<Q16_1Tensor> Q16_1Tensor::quantize_from_fp32(
        const float *src,
        const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q16_1Tensor::quantize_from_fp32: shape cannot be empty");
        }

        size_t total_elements = 1;
        for (auto dim : shape)
        {
            total_elements *= dim;
        }

        const size_t n_blocks = (total_elements + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const size_t total_bytes = n_blocks * sizeof(Q16_1Block);

        std::vector<uint8_t> raw_data(total_bytes);
        Q16_1Block *blocks = reinterpret_cast<Q16_1Block *>(raw_data.data());

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q16_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), total_elements - offset);

            Q16_1Block &block = blocks[block_idx];

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[offset + i]));
            }

            // Compute scale factor (d = max_abs / 32767 for int16 range)
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d; // FP32 scale, no conversion needed

            // Quantize AND compute sum simultaneously
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float val = src[offset + i];
                    const float scaled = val * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    block.qs[i] = static_cast<int16_t>(std::round(clamped));
                    sum_i64 += static_cast<int64_t>(block.qs[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            // Zero-fill partial block tail
            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            // Store pre-computed integer sum (INT32 for wider int16 value range)
            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

        return std::make_shared<Q16_1Tensor>(shape, raw_data);
    }

    // ===== Private Helper Methods =====

    bool Q16_1Tensor::copyFrom_fp32_rows(const float *src_data, size_t num_rows)
    {
        if (!src_data)
        {
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows] Requires 2D tensor");
            return false;
        }

        const size_t max_rows = shape_[0];
        const size_t cols = shape_[1];
        const size_t actual_rows = std::min(num_rows, max_rows);
        const size_t blocks_per_row = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const size_t n_blocks = actual_rows * blocks_per_row;
        const size_t elements_per_row = cols;

        Q16_1Block *blocks = mutable_q16_1_blocks();

#pragma omp parallel for if (n_blocks > 4)
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t row_idx = block_idx / blocks_per_row;
            const size_t col_block = block_idx % blocks_per_row;
            const size_t col_offset = col_block * Q16_1Block::BLOCK_SIZE;
            const size_t count = std::min(static_cast<size_t>(Q16_1Block::BLOCK_SIZE), cols - col_offset);

            Q16_1Block &block = blocks[block_idx];

            // Source data is row-major: src_data[row * cols + col]
            const float *row_data = src_data + row_idx * elements_per_row + col_offset;

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(row_data[i]));
            }

            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;

            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = row_data[i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            for (size_t i = count; i < Q16_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            block.sum_qs = static_cast<int32_t>(sum_i64);
        }

        dequant_cache_.clear();
        return true;
    }

    namespace
    {
        /**
         * @brief Templated quantization helper for variable Q16 block sizes
         */
        template <typename BlockType>
        void quantize_fp32_to_q16_blocks(const float *src_data, void *dst_blocks,
                                         size_t total_elements, size_t n_blocks)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            BlockType *blocks = static_cast<BlockType *>(dst_blocks);

#pragma omp parallel for if (n_blocks > 4)
            for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
            {
                const size_t offset = block_idx * BLOCK_SIZE;
                const size_t count = std::min(BLOCK_SIZE, total_elements - offset);

                BlockType &block = blocks[block_idx];

                float max_abs = 0.0f;
                for (size_t i = 0; i < count; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(src_data[offset + i]));
                }

                const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
                block.d = d;

                int64_t sum_i64 = 0;
                if (d > 1e-10f)
                {
                    const float inv_d = 1.0f / d;
                    for (size_t i = 0; i < count; ++i)
                    {
                        const float scaled = src_data[offset + i] * inv_d;
                        const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                        const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                        block.qs[i] = quantized;
                        sum_i64 += static_cast<int64_t>(quantized);
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        block.qs[i] = 0;
                    }
                }

                // Zero-fill any remaining elements in the block
                for (size_t i = count; i < BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }

                block.sum_qs = static_cast<int32_t>(sum_i64);
            }
        }
    } // anonymous namespace

    bool Q16_1Tensor::copyFrom_fp32(const float *src_data)
    {
        if (!src_data)
        {
            return false;
        }

        const size_t total_elements = element_count();
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t n_blocks = (total_elements + bs - 1) / bs;

        void *raw_blocks = raw_mutable_data();

        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
            quantize_fp32_to_q16_blocks<Q16_1Block>(src_data, raw_blocks, total_elements, n_blocks);
            break;
        case Q16BlockSize::BLOCK_64:
            quantize_fp32_to_q16_blocks<Q16_1Block_64>(src_data, raw_blocks, total_elements, n_blocks);
            break;
        case Q16BlockSize::BLOCK_128:
            quantize_fp32_to_q16_blocks<Q16_1Block_128>(src_data, raw_blocks, total_elements, n_blocks);
            break;
        default:
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32] Unknown block size");
            return false;
        }

        dequant_cache_.clear();
        return true;
    }

    // ============================================================================
    // Fixed-Scale Quantization (VNNI-Safe)
    // ============================================================================

    namespace
    {
        /**
         * @brief Templated FIXED-SCALE quantization for VNNI-safe Q16 blocks
         *
         * CRITICAL: Unlike adaptive quantization, this uses a FIXED scale and clips
         * INT16 values to prevent VNNI INT32 overflow during dot-product accumulation.
         *
         * @tparam BlockType Q16_1Block, Q16_1Block_64, or Q16_1Block_128
         * @param src_data Source FP32 data
         * @param dst_blocks Destination Q16 blocks
         * @param total_elements Total number of elements to quantize
         * @param n_blocks Number of blocks
         * @param kv_cache_scale Fixed scale factor (e.g., 8.0 for ±8.0 FP32 range)
         * @param max_safe_int16 Maximum safe INT16 magnitude for VNNI (from head_dim)
         */
        template <typename BlockType>
        void quantize_fp32_to_q16_blocks_fixed_scale(const float *src_data, void *dst_blocks,
                                                     size_t total_elements, size_t n_blocks,
                                                     float kv_cache_scale, int16_t max_safe_int16)
        {
            constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
            BlockType *blocks = static_cast<BlockType *>(dst_blocks);

            // Fixed scale: d = kv_cache_scale / 32767
            // Quantization: int16 = fp32 / d = fp32 * 32767 / kv_cache_scale
            const float d = kv_cache_scale / 32767.0f;
            const float inv_d = 32767.0f / kv_cache_scale;

#if LLAMINAR_ASSERTIONS_ACTIVE
            // Debug-build overflow detection: track clipping statistics
            std::atomic<size_t> total_clipped{0};
            std::atomic<size_t> total_processed{0};
#endif

#pragma omp parallel for if (n_blocks > 4)
            for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
            {
                const size_t offset = block_idx * BLOCK_SIZE;
                const size_t count = std::min(BLOCK_SIZE, total_elements - offset);

                BlockType &block = blocks[block_idx];

                // Fixed scale for ALL blocks (not adaptive!)
                block.d = d;

#if LLAMINAR_ASSERTIONS_ACTIVE
                size_t local_clipped = 0;
#endif

                int64_t sum_i64 = 0;
                for (size_t i = 0; i < count; ++i)
                {
                    // Quantize with fixed scale
                    const float scaled = src_data[offset + i] * inv_d;

#if LLAMINAR_ASSERTIONS_ACTIVE
                    // Track if this value required clipping
                    if (std::abs(scaled) > static_cast<float>(max_safe_int16))
                    {
                        ++local_clipped;
                    }
#endif

                    // CRITICAL: Clip to VNNI-safe range, NOT ±32767!
                    const float clamped = std::max(static_cast<float>(-max_safe_int16),
                                                   std::min(scaled, static_cast<float>(max_safe_int16)));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }

                // Zero-fill any remaining elements in the block
                for (size_t i = count; i < BLOCK_SIZE; ++i)
                {
                    block.qs[i] = 0;
                }

                block.sum_qs = static_cast<int32_t>(sum_i64);

#if LLAMINAR_ASSERTIONS_ACTIVE
                total_clipped.fetch_add(local_clipped, std::memory_order_relaxed);
                total_processed.fetch_add(count, std::memory_order_relaxed);
#endif
            }

#if LLAMINAR_ASSERTIONS_ACTIVE
            // Log warning if clipping occurs frequently (>1% of values)
            const size_t clipped_count = total_clipped.load(std::memory_order_relaxed);
            const size_t processed_count = total_processed.load(std::memory_order_relaxed);
            if (processed_count > 0 && clipped_count > 0)
            {
                const float clip_rate = 100.0f * static_cast<float>(clipped_count) / static_cast<float>(processed_count);
                if (clip_rate > 1.0f)
                {
                    LOG_WARN("[Q16_1Tensor] High clipping rate during fixed-scale quantization: "
                             << clipped_count << "/" << processed_count << " values ("
                             << clip_rate << "%) exceeded max_safe_int16=" << max_safe_int16
                             << " (kv_cache_scale=" << kv_cache_scale << ")");
                }
                else
                {
                    LOG_DEBUG("[Q16_1Tensor] Fixed-scale quantization clipped "
                              << clipped_count << "/" << processed_count << " values ("
                              << clip_rate << "%)");
                }
            }
#endif
        }

        /**
         * @brief Get MAX_SAFE_INT16 for a given head_dim
         *
         * See VNNISafetyConstants.h for derivation. These are the maximum INT16
         * magnitudes that prevent INT32 overflow during VNNI dot-product accumulation.
         */
        inline int16_t get_max_safe_int16_for_head_dim(int head_dim)
        {
            // Formula: MAX_SAFE_INT16 = floor(sqrt(INT32_MAX / (head_dim / 16)))
            // Pre-computed values for common head dimensions:
            switch (head_dim)
            {
            case 64:
                return 23170; // sqrt(2^31 / 4)
            case 96:
                return 18918; // sqrt(2^31 / 6)
            case 128:
                return 16383; // sqrt(2^31 / 8)
            case 192:
                return 13377; // sqrt(2^31 / 12) - MLA
            case 256:
                return 11585; // sqrt(2^31 / 16)
            default:
                // Conservative fallback: safe for head_dim ≤ 192
                if (head_dim <= 64)
                    return 23170;
                if (head_dim <= 96)
                    return 18918;
                if (head_dim <= 128)
                    return 16383;
                if (head_dim <= 192)
                    return 13377;
                return 11585;
            }
        }
    } // anonymous namespace

    bool Q16_1Tensor::copyFrom_fp32_fixed_scale(const float *src_data, float kv_cache_scale, int head_dim)
    {
        if (!src_data)
        {
            return false;
        }

        if (kv_cache_scale <= 0.0f)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_fixed_scale] Invalid kv_cache_scale: " << kv_cache_scale);
            return false;
        }

        if (head_dim <= 0)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_fixed_scale] Invalid head_dim: " << head_dim);
            return false;
        }

        const int16_t max_safe_int16 = get_max_safe_int16_for_head_dim(head_dim);
        const size_t total_elements = element_count();
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t n_blocks = (total_elements + bs - 1) / bs;

        void *raw_blocks = raw_mutable_data();

        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block>(
                src_data, raw_blocks, total_elements, n_blocks, kv_cache_scale, max_safe_int16);
            break;
        case Q16BlockSize::BLOCK_64:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block_64>(
                src_data, raw_blocks, total_elements, n_blocks, kv_cache_scale, max_safe_int16);
            break;
        case Q16BlockSize::BLOCK_128:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block_128>(
                src_data, raw_blocks, total_elements, n_blocks, kv_cache_scale, max_safe_int16);
            break;
        default:
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_fixed_scale] Unknown block size");
            return false;
        }

        dequant_cache_.clear();
        return true;
    }

    bool Q16_1Tensor::copyFrom_fp32_rows_fixed_scale(const float *src_data, size_t num_rows,
                                                     float kv_cache_scale, int head_dim)
    {
        if (!src_data)
        {
            return false;
        }

        if (shape_.size() < 2)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows_fixed_scale] Tensor must be at least 2D");
            return false;
        }

        if (num_rows > shape_[0])
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows_fixed_scale] num_rows (" << num_rows
                                                                                 << ") exceeds tensor rows (" << shape_[0] << ")");
            return false;
        }

        if (kv_cache_scale <= 0.0f)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows_fixed_scale] Invalid kv_cache_scale: "
                      << kv_cache_scale);
            return false;
        }

        if (head_dim <= 0)
        {
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows_fixed_scale] Invalid head_dim: " << head_dim);
            return false;
        }

        const int16_t max_safe_int16 = get_max_safe_int16_for_head_dim(head_dim);
        const size_t cols = shape_[1];
        const size_t elements_to_copy = num_rows * cols;
        const size_t bs = static_cast<size_t>(block_size_);
        const size_t blocks_to_fill = (elements_to_copy + bs - 1) / bs;

        void *raw_blocks = raw_mutable_data();

        switch (block_size_)
        {
        case Q16BlockSize::BLOCK_32:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block>(
                src_data, raw_blocks, elements_to_copy, blocks_to_fill, kv_cache_scale, max_safe_int16);
            break;
        case Q16BlockSize::BLOCK_64:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block_64>(
                src_data, raw_blocks, elements_to_copy, blocks_to_fill, kv_cache_scale, max_safe_int16);
            break;
        case Q16BlockSize::BLOCK_128:
            quantize_fp32_to_q16_blocks_fixed_scale<Q16_1Block_128>(
                src_data, raw_blocks, elements_to_copy, blocks_to_fill, kv_cache_scale, max_safe_int16);
            break;
        default:
            LOG_ERROR("[Q16_1Tensor::copyFrom_fp32_rows_fixed_scale] Unknown block size");
            return false;
        }

        dequant_cache_.clear();
        return true;
    }

} // namespace llaminar2

#pragma GCC diagnostic pop