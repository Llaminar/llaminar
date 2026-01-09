/**
 * @file Q8_1Tensor.cpp
 * @brief Q8_1 quantized tensor implementation (8-bit with pre-computed sum, intermediate activation format)
 * @author David Sanftenberg
 */

#include "CPUTensors.h"
#include "../../kernels/KernelFactory.h"
#include "../../kernels/cpu/ops/CPURMSNormKernelT.h"

#include "../../kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "../../kernels/cpu/attention/CPUAttentionKernelT.h"
#include "../../utils/Logger.h"
#include "../SIMDHelpers.h"
#include "../FP16Utils.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    Q8_1Tensor::Q8_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr),
          is_mutable_(false), cache_dirty_(false)
    {
        LOG_TRACE("[Q8_1Tensor] Creating IMMUTABLE tensor from raw_data, shape=[" << shape[0] << "," << shape[1]
                                                                                  << "], raw_data.size()=" << raw_data.size());

        if (shape.empty())
        {
            throw std::invalid_argument("Q8_1Tensor: shape cannot be empty");
        }

        // Q8_1 tensors are treated as row-major blocks with per-row padding:
        // blocks_per_row = ceil(cols/32), total_blocks = rows * blocks_per_row.
        // This must match q8_1_blocks() / decode_block_at() indexing.
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q8_1Tensor: expected 2D shape for Q8_1 storage");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t n_blocks = rows * blocks_per_row;
        const size_t expected_bytes = n_blocks * sizeof(Q8_1Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q8_1Tensor: insufficient raw data (" +
                                        std::to_string(raw_data_.size()) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    // Construct from Q8_1Block array (for testing and direct block construction)
    Q8_1Tensor::Q8_1Tensor(const std::vector<size_t> &shape, const Q8_1Block *blocks, size_t num_blocks, int device_idx)
        : shape_(shape), is_view_(false), raw_data_(), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::fromLegacyIndex(device_idx)), device_blocks_(nullptr),
          is_mutable_(true), cache_dirty_(false)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_1Tensor: shape cannot be empty");
        }
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q8_1Tensor: expected 2D shape for Q8_1 storage");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t expected_blocks = rows * blocks_per_row;

        if (num_blocks < expected_blocks)
        {
            throw std::invalid_argument("Q8_1Tensor: insufficient blocks (" +
                                        std::to_string(num_blocks) + ", expected " +
                                        std::to_string(expected_blocks) + ")");
        }

        // Copy blocks to internal storage
        const size_t required_bytes = expected_blocks * sizeof(Q8_1Block);
        raw_data_.resize(required_bytes);
        std::memcpy(raw_data_.data(), blocks, required_bytes);

        LOG_TRACE("[Q8_1Tensor] Created from Q8_1Block* shape=[" << shape[0] << "," << shape[1]
                                                                 << "], blocks=" << expected_blocks);
    }

    // Copy constructor - deep copy of all data
    Q8_1Tensor::Q8_1Tensor(const Q8_1Tensor &other)
        : shape_(other.shape_), is_view_(false), raw_data_(other.raw_data_), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(other.device_), device_blocks_(nullptr),
          is_mutable_(other.is_mutable_), cache_dirty_(false)
    {
        // If the source is a view, we need to copy the actual viewed data
        if (other.is_view_)
        {
            const size_t rows = shape_[0];
            const size_t cols = shape_[1];
            const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const size_t n_blocks = rows * blocks_per_row;
            const size_t required_bytes = n_blocks * sizeof(Q8_1Block);

            raw_data_.resize(required_bytes);
            std::memcpy(raw_data_.data(), other.raw_data_ptr_ + other.view_byte_offset_, required_bytes);
        }

        // Copy dequant cache if present
        if (!other.dequant_cache_.empty())
        {
            dequant_cache_ = other.dequant_cache_;
        }

        LOG_TRACE("[Q8_1Tensor] Copy constructed shape=[" << shape_[0] << "," << shape_[1]
                                                          << "], is_mutable=" << is_mutable_);
    }

    // Mutable activation buffer constructor - allocates uninitialized Q8_1 storage
    //
    // IMPORTANT: Q8_1 is an activation-only format designed for memory bandwidth savings.
    // Mutable Q8_1 tensors have mutable Q8_1 BLOCKS, not an FP32 cache.
    // Kernels must write directly to Q8_1 blocks via mutable_q8_1_blocks().
    // Do NOT use mutable_data() - it throws for Q8_1 tensors.
    Q8_1Tensor::Q8_1Tensor(const std::vector<size_t> &shape, int device_idx)
        : shape_(shape), is_view_(false), raw_data_(), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::fromLegacyIndex(device_idx)), device_blocks_(nullptr),
          is_mutable_(true), cache_dirty_(false)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_1Tensor: shape cannot be empty");
        }
        if (shape.size() != 2)
        {
            throw std::invalid_argument("Q8_1Tensor: mutable activation buffer requires 2D shape");
        }

        // Allocate row-major blocks with per-row padding (see q8_1_blocks()/decode_block_at()).
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t n_blocks = rows * blocks_per_row;
        const size_t required_bytes = n_blocks * sizeof(Q8_1Block);

        // Allocate and zero-initialize the raw block storage
        // NOTE: We do NOT allocate an FP32 dequant cache here.
        // Q8_1 tensors should be accessed via q8_1_blocks()/mutable_q8_1_blocks(),
        // not via data()/mutable_data(). The dequant_cache_ is only for lazy
        // read-only dequantization (debugging, snapshot capture, etc.).
        raw_data_.resize(required_bytes, 0);

        LOG_TRACE("[Q8_1Tensor] Created MUTABLE tensor shape=[" << shape[0] << "," << shape[1]
                                                                << "], blocks=" << n_blocks << ", bytes=" << required_bytes);
    }

    // Private view constructor
    Q8_1Tensor::Q8_1Tensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<CPUTensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr),
          is_mutable_(false), cache_dirty_(false)
    {
        // Check if parent is a mutable Q8_1Tensor - if so, inherit mutability
        if (auto parent_q8_1 = std::dynamic_pointer_cast<Q8_1Tensor>(parent))
        {
            is_mutable_ = parent_q8_1->is_mutable_;
            if (is_mutable_)
            {
                // For mutable views, we need to set up access to the parent's dequant cache
                // Calculate the FP32 element offset for this view
                size_t K = shape_[1];
                size_t blocks_per_row = (K + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                size_t block_offset_from_root = byte_offset / sizeof(Q8_1Block);
                size_t row_offset = block_offset_from_root / blocks_per_row;

                // Pre-allocate a view of the parent's dequant cache
                // The view's mutable_data() will return a pointer into the parent's cache
                // Note: we store view_byte_offset_ to calculate the FP32 offset later
            }
        }

        LOG_TRACE("[Q8_1Tensor] Creating VIEW tensor shape=[" << shape[0] << "," << shape[1]
                                                              << "], is_mutable=" << is_mutable_ << " (inherited from parent)");
    }

    Q8_1Tensor::~Q8_1Tensor()
    {
        // Destructor
    }

    std::shared_ptr<CPUTensorBase> Q8_1Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate 2D shape
        if (new_shape.size() != 2)
        {
            LOG_ERROR("[Q8_1Tensor::create_view] ERROR: View must be 2D (got "
                      << new_shape.size() << "D)");
            return nullptr;
        }

        // Validate K dimension matches (row-slice only)
        if (new_shape[1] != shape_[1])
        {
            LOG_ERROR("[Q8_1Tensor::create_view] ERROR: View must preserve K dimension\n"
                      << "  Parent K: " << shape_[1] << ", View K: " << new_shape[1]);
            return nullptr;
        }

        size_t K = shape_[1];

        // Validate offset is row-aligned
        if (offset % K != 0)
        {
            LOG_ERROR("[Q8_1Tensor::create_view] ERROR: Offset must be row-aligned (multiple of K="
                      << K << ")\n"
                      << "  Got offset: " << offset);
            return nullptr;
        }

        // Validate bounds
        size_t start_row = offset / K;
        size_t view_rows = new_shape[0];
        if (start_row + view_rows > shape_[0])
        {
            LOG_ERROR("[Q8_1Tensor::create_view] ERROR: View exceeds parent bounds\n"
                      << "  Parent rows: " << shape_[0] << ", Start row: " << start_row
                      << ", View rows: " << view_rows);
            return nullptr;
        }

        // Calculate byte offset
        size_t blocks_per_row = (K + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        size_t block_offset = start_row * blocks_per_row;
        size_t byte_offset = block_offset * sizeof(Q8_1Block);

        // Determine root parent
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
                LOG_ERROR("[Q8_1Tensor::create_view] ERROR: shared_from_this() failed");
                return nullptr;
            }
            root_data_ptr = raw_data_.data();
            root_byte_offset = byte_offset;
        }

        return std::shared_ptr<Q8_1Tensor>(new Q8_1Tensor(
            new_shape, root_data_ptr, root_byte_offset, root_parent));
    }

    bool Q8_1Tensor::set_device(int device_idx)
    {
        // TODO: Implement device transfer
        device_ = DeviceId::fromLegacyIndex(device_idx);
        return true;
    }

    const float *Q8_1Tensor::data() const
    {
        // Q8_1Tensor::data() is DEPRECATED and now throws to catch accidental FP32 dequantization.
        //
        // The whole point of Q8_1 is memory bandwidth optimization - silently converting to FP32
        // defeats this purpose and hides precision issues from developers.
        //
        // If you need FP32 data:
        //   1. Use fp32_data() - explicit acknowledgment of dequantization
        //   2. Prefer q8_1_blocks() for kernels that can operate on Q8_1 natively
        //   3. Use typed kernels (e.g., CPUAttentionKernel_Q8_1) for best performance
        //
        // This change surfaces places where Q8_1 precision is being silently degraded.

        throw std::runtime_error(
            "Q8_1Tensor::data() is deprecated - use fp32_data() if you explicitly need FP32 values, "
            "or q8_1_blocks() for native Q8_1 access. data() throws to catch accidental dequantization "
            "that defeats the purpose of Q8_1 (memory bandwidth optimization).");
    }

    const float *Q8_1Tensor::fp32_data() const
    {
        assertValid("Q8_1Tensor::fp32_data");
        // Explicit FP32 dequantization - the caller acknowledges the precision conversion.

        // For mutable views, return data from parent's cache
        if (is_view_ && is_mutable_ && parent_)
        {
            auto parent_q8_1 = std::dynamic_pointer_cast<Q8_1Tensor>(parent_);
            if (parent_q8_1)
            {
                // Calculate the row offset for this view
                size_t K = shape_[1];
                size_t blocks_per_row = (K + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                size_t block_offset = view_byte_offset_ / sizeof(Q8_1Block);
                size_t row_offset = block_offset / blocks_per_row;
                size_t element_offset = row_offset * K;

                // Return pointer into parent's cache
                return parent_q8_1->fp32_data() + element_offset;
            }
        }

        // IMPORTANT:
        // - For immutable tensors, caching is safe because blocks never change.
        // - For mutable tensors, blocks are frequently updated by kernels writing directly into
        //   `mutable_q8_1_blocks()`. Any previously-populated FP32 cache would become stale.
        //   To ensure correctness (snapshots/debugging + any callers that legitimately require FP32),
        //   we always dequantize from the current Q8_1 blocks for mutable tensors.

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
        const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);
        const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

#pragma omp parallel for collapse(2) if (total_elements > 10000)
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const size_t col0 = b * Q8_1Block::BLOCK_SIZE;
                if (col0 >= cols)
                    continue;

                const Q8_1Block &block = blocks[r * blocks_per_row + b];

                alignas(64) float tmp[Q8_1Block::BLOCK_SIZE];
                decodeBlock(block, tmp);

                const size_t copy = std::min(static_cast<size_t>(Q8_1Block::BLOCK_SIZE), cols - col0);
                std::memcpy(&dequant_cache_[r * cols + col0], tmp, copy * sizeof(float));
            }
        }

        // Cache is now consistent with current blocks.
        cache_dirty_ = false;
        return dequant_cache_.data();
    }

    float *Q8_1Tensor::mutable_data()
    {
        // Q8_1 tensors should NEVER be accessed via mutable_data().
        //
        // Q8_1 is a memory bandwidth optimization format. The whole point is to avoid
        // writing/reading FP32 data to/from DRAM. If you need to write to a Q8_1 tensor:
        //   1. Use mutable_q8_1_blocks() to get a Q8_1Block* pointer
        //   2. Use typed kernels that write Q8_1 directly (e.g., CPURMSNormKernelT<Q8_1>)
        //   3. Use SIMD helpers like simd::quantize_fp32_to_q8_1_blocks() if converting from FP32
        //
        // If your code path requires mutable FP32 access, use FP32Tensor instead of Q8_1Tensor.
        //
        // Common patterns:
        //   - RMSNorm output: kernel->apply_q8_1(input_blocks, gamma, output->mutable_q8_1_blocks(), ...)
        //   - Residual add: simd::q8_1_add_q8_1(residual_blocks, input_blocks, output->mutable_q8_1_blocks(), n)
        //   - GEMM output: quantize result directly to output->mutable_q8_1_blocks()

        throw std::runtime_error(
            "Q8_1Tensor::mutable_data() is not supported. Q8_1 is a memory bandwidth optimization format - "
            "writing FP32 to a cache defeats its purpose. Use mutable_q8_1_blocks() and typed kernels instead. "
            "See the Q8_1 typed kernel pattern in CPURMSNormKernelT<ActivationPrecision::Q8_1>.");
    }

    bool Q8_1Tensor::quantize_from_cache()
    {
        // This function is DEPRECATED and should not be used.
        // The FP32 cache pattern defeats the purpose of Q8_1 (memory bandwidth optimization).
        //
        // Instead, use typed kernels that write directly to Q8_1 blocks:
        //   kernel->apply_q8_1(input_blocks, weights, output->mutable_q8_1_blocks(), ...)
        //
        // Or use SIMD helpers for explicit conversion:
        //   simd::quantize_fp32_to_q8_1_blocks(fp32_src, output->mutable_q8_1_blocks(), n_elements)
        LOG_WARN("[Q8_1Tensor::quantize_from_cache] DEPRECATED: This function should not be used. "
                 "Use typed kernels with mutable_q8_1_blocks() instead.");

        // Return true for backward compatibility but log warning
        return true;
    }

    std::unique_ptr<ITensorGemm> Q8_1Tensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    std::unique_ptr<ITensorRoPE> Q8_1Tensor::createRoPE()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createRoPE(this, dev_type);
    }

    std::unique_ptr<ITensorSwiGLU> Q8_1Tensor::createSwiGLU()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createSwiGLU(this, dev_type);
    }

    std::unique_ptr<ITensorSoftmax> Q8_1Tensor::createSoftmax()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createSoftmax(this, dev_type);
    }

    std::unique_ptr<ITensorRMSNorm> Q8_1Tensor::createRMSNorm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createRMSNorm(this, dev_type);
    }

    std::unique_ptr<ITensorAttention> Q8_1Tensor::createAttention()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createAttention(this, dev_type);
    }

    std::unique_ptr<ITensorEmbedding> Q8_1Tensor::createEmbedding()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createEmbedding(this, dev_type);
    }

    bool Q8_1Tensor::applyRoPE(
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
        (void)use_bf16; // Q8_1 uses native integer path
        (void)mpi_ctx;  // Not used for Q8_1 RoPE

        // Use centralized KernelFactory for device-aware dispatch
        auto kernel = createRoPE();
        if (!kernel)
        {
            LOG_ERROR("[Q8_1Tensor::applyRoPE] Failed to create RoPE kernel");
            return false;
        }

        // Get Q pointer (this tensor) - use mutable_q8_1_blocks() to invalidate cache
        void *Q_ptr = mutable_q8_1_blocks();

        // Get K pointer - K is passed as float* but is actually Q8_1 block data
        // Note: K's cache should have been invalidated by the caller via mutable_q8_1_blocks()
        void *K_ptr = (void *)K;

        bool success = kernel->apply_q8_1(
            Q_ptr, K_ptr,
            position_ids,
            seq_len, n_heads, n_kv_heads, head_dim,
            rope_theta,
            device_idx);

        // Cache was already cleared by mutable_q8_1_blocks() call above
        return success;
    }

    bool Q8_1Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        if (!accum)
        {
            LOG_ERROR("[Q8_1Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[Q8_1Tensor::from_int32_with_scales] tensor must be 2D, got " << shape_.size() << "D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[Q8_1Tensor::from_int32_with_scales] shape mismatch: tensor=[" << shape_[0]
                                                                                      << ", " << shape_[1] << "] input=[" << rows << ", " << cols << "]");
            return false;
        }

        const size_t total_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t total_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

        uint8_t *dst_bytes = nullptr;
        if (is_view_)
        {
            if (!raw_data_ptr_)
            {
                LOG_ERROR("[Q8_1Tensor::from_int32_with_scales] view has no parent data pointer");
                return false;
            }
            dst_bytes = const_cast<uint8_t *>(raw_data_ptr_) + view_byte_offset_;
        }
        else
        {
            const size_t required_bytes = total_blocks * sizeof(Q8_1Block);
            if (raw_data_.size() < required_bytes)
            {
                raw_data_.resize(required_bytes);
            }
            dst_bytes = raw_data_.data();
        }

        Q8_1Block *blocks = reinterpret_cast<Q8_1Block *>(dst_bytes);

        thread_local std::vector<float> temp_fp32;
        temp_fp32.resize(total_elements);
        simd::requantize_int32_matrix_to_fp32(
            accum,
            temp_fp32.data(),
            rows,
            cols,
            row_scales,
            col_scales,
            bias);

#pragma omp parallel for if (total_blocks > 4)
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q8_1Block::BLOCK_SIZE;
            const size_t count = std::min(Q8_1Block::BLOCK_SIZE, total_elements - offset);

            Q8_1Block &block = blocks[block_idx];

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

        dequant_cache_.clear();
        return true;
    }

    void Q8_1Tensor::decodeBlockScalar(const Q8_1Block &block, float *output)
    {
        // Scalar implementation for Q8_1: scale * int8 value
        const float scale = fp16_to_fp32(block.d);
        for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
        {
            output[i] = scale * static_cast<float>(block.qs[i]);
        }
    }

    void Q8_1Tensor::decodeBlock(const Q8_1Block &block, float *output)
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
    void Q8_1Tensor::decodeBlockAVX512(const Q8_1Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m512 vscale = _mm512_set1_ps(scale);

        // Process 16 int8 values at a time (2 iterations for 32 elements)
        for (size_t i = 0; i < 2; ++i)
        {
            // Load 16 int8 values and convert to int32
            __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block.qs[i * 16]));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);

            // Convert to float and scale
            __m512 vf = _mm512_cvtepi32_ps(vi32);
            vf = _mm512_mul_ps(vf, vscale);

            // Store result
            _mm512_storeu_ps(&output[i * 16], vf);
        }
    }
#endif

#if defined(__AVX2__)
    void Q8_1Tensor::decodeBlockAVX2(const Q8_1Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(scale);

        // Process 8 int8 values at a time (4 iterations for 32 elements)
        for (size_t i = 0; i < 4; ++i)
        {
            // Load 8 int8 values from memory
            __m128i vi8_half = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&block.qs[i * 8]));

            // Sign-extend int8 → int32 (8 elements)
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8_half);

            // Convert int32 → float
            __m256 vf = _mm256_cvtepi32_ps(vi32);

            // Multiply by scale
            vf = _mm256_mul_ps(vf, vscale);

            // Store result
            _mm256_storeu_ps(&output[i * 8], vf);
        }
    }
#endif

    bool Q8_1Tensor::copyFrom(const CPUTensorBase *src)
    {
        if (!src)
        {
            LOG_ERROR("[Q8_1Tensor::copyFrom] Source is null");
            return false;
        }

        // For mutable tensors, we can copy from FP32 source
        if (is_mutable_)
        {
            // Verify shape compatibility
            const auto &src_shape = src->shape();
            if (src_shape.size() != shape_.size())
            {
                LOG_ERROR("[Q8_1Tensor::copyFrom] Shape dimension mismatch");
                return false;
            }
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                if (src_shape[i] != shape_[i])
                {
                    LOG_ERROR("[Q8_1Tensor::copyFrom] Shape mismatch at dim " << i);
                    return false;
                }
            }

            // Copy FP32 data to cache
            const float *src_data = src->data();
            if (!src_data)
            {
                LOG_ERROR("[Q8_1Tensor::copyFrom] Source data is null");
                return false;
            }

            const size_t total_elements = element_count();
            if (dequant_cache_.size() != total_elements)
            {
                dequant_cache_.resize(total_elements);
            }

            std::memcpy(dequant_cache_.data(), src_data, total_elements * sizeof(float));
            cache_dirty_ = true;

            // Optionally quantize immediately
            return quantize_from_cache();
        }

        // Immutable tensors don't support copyFrom
        LOG_ERROR("[Q8_1Tensor::copyFrom] Immutable Q8_1 tensors do not support copyFrom");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q8_1Tensor::to_bf16(uint16_t *dst) const
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

    void Q8_1Tensor::to_fp16(uint16_t *dst) const
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

    void Q8_1Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q8_1Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q8_1Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    ActivationPack Q8_1Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        return pack_activation_rows_to_int8(rows, cols);
    }

    const Q8_1Block *Q8_1Tensor::decode_to_q8_1(size_t row_idx, size_t k_block_offset) const
    {
        // Q8_1 is already in Q8_1 format - return direct pointer (zero-copy!)
        const size_t blocks_per_row = (shape_[1] + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q8_1Tensor::decode_to_q8_1: row_idx out of range");
        }
        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q8_1Tensor::decode_to_q8_1: k_block_offset exceeds blocks per row");
        }

        // Get pointer to source Q8_1 block
        const uint8_t *data_ptr = is_view_ ? raw_data_ptr_ + view_byte_offset_ : raw_data_.data();
        const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);

        // Return direct pointer - no copy needed!
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    } // ===== Static Quantization Method (Key Method!) =====

    std::shared_ptr<Q8_1Tensor> Q8_1Tensor::quantize_from_fp32(
        const float *src,
        const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_1Tensor::quantize_from_fp32: shape cannot be empty");
        }

        // Calculate total elements
        size_t total_elements = 1;
        for (auto dim : shape)
        {
            total_elements *= dim;
        }

        // Calculate number of blocks
        const size_t n_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t total_bytes = n_blocks * sizeof(Q8_1Block);

        // Allocate raw data buffer
        std::vector<uint8_t> raw_data(total_bytes);
        Q8_1Block *blocks = reinterpret_cast<Q8_1Block *>(raw_data.data());

        // Quantize each block
#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * Q8_1Block::BLOCK_SIZE;
            const size_t count = std::min(Q8_1Block::BLOCK_SIZE, total_elements - offset);

            Q8_1Block &block = blocks[block_idx];

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[offset + i]));
            }

            // Compute scale factor (d = max_abs / 127)
            const float d = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 0.0f;
            block.d = fp32_to_fp16(d);

            // Quantize AND compute sum simultaneously (CUDA pattern!)
            // CRITICAL (Nov 2024): Store RAW integer sum, not d × sum!
            int32_t sum_i32 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float val = src[offset + i];
                    const float scaled = val * inv_d;
                    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
                    block.qs[i] = static_cast<int8_t>(std::round(clamped));
                    sum_i32 += static_cast<int32_t>(block.qs[i]); // Sum QUANTIZED values (CRITICAL!)
                }
            }
            else
            {
                // Zero block
                for (size_t i = 0; i < count; ++i)
                {
                    block.qs[i] = 0;
                }
            }

            // Zero-fill partial block tail (if any)
            for (size_t i = count; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                block.qs[i] = 0;
            }

            // Store pre-computed integer sum directly (INT16)
            // Range check: 32 int8 values sum to [-4064, 4064], safe for INT16 [-32768, 32767]
            block.sum_qs = static_cast<int16_t>(sum_i32);
        }

        // Create and return Q8_1Tensor
        return std::make_shared<Q8_1Tensor>(shape, raw_data);
    }

    bool Q8_1Tensor::applyRMSNorm(
        const float *gamma,
        int seq_len,
        int d_model,
        float eps,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)gamma;
        (void)seq_len;
        (void)d_model;
        (void)eps;
        (void)mpi_ctx;
        (void)device_idx;
        // Q8_1 tensor doesn't support in-place RMSNorm - use createRMSNorm() instead
        LOG_ERROR("[Q8_1Tensor::applyRMSNorm] Not supported for Q8_1 tensors");
        return false;
    }

    // ===== Bulk Q8_1 Quantization (Q8_1 → Q8_1 direct copy) =====

    bool Q8_1Tensor::quantize_to_q8_1(void *q8_1_buffer, int m, int k) const
    {
        if (!q8_1_buffer || m <= 0 || k <= 0)
        {
            return false;
        }

        // Validate dimensions against tensor shape
        const size_t cols = shape_[1];
        const size_t rows = shape_[0];
        if (static_cast<size_t>(m) > rows || static_cast<size_t>(k) > cols)
        {
            return false;
        }

        const int k_blocks = (k + 31) / 32;
        const size_t blocks_per_row = (cols + 31) / 32;

        // Get source data pointer (handle views)
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q8_1Block *src_blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);
        Q8_1Block *dst_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);

        // Direct copy - Q8_1 is already in the right format
        // Just need to handle potential row stride differences if k < cols
        if (k == static_cast<int>(cols))
        {
            // Fast path: contiguous copy
            std::memcpy(dst_blocks, src_blocks, static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block));
        }
        else
        {
            // Copy row by row (handle different k vs cols)
#pragma omp parallel for schedule(static)
            for (int i = 0; i < m; ++i)
            {
                const Q8_1Block *src_row = src_blocks + i * blocks_per_row;
                Q8_1Block *dst_row = dst_blocks + i * k_blocks;
                std::memcpy(dst_row, src_row, k_blocks * sizeof(Q8_1Block));
            }
        }

        return true;
    }

    // ===== Q8_1 Block-Aligned K-Slicing (Phase 6.4) =====

    std::shared_ptr<Q8_1Tensor> Q8_1Tensor::slice_k_blocks(size_t k_start, size_t k_size) const
    {
        // Validate k_start alignment (must be multiple of BLOCK_SIZE=32)
        if (!is_k_aligned(k_start))
        {
            LOG_ERROR("[Q8_1Tensor::slice_k_blocks] k_start=" << k_start
                                                              << " is not aligned to block boundary (must be multiple of "
                                                              << Q8_1Block::BLOCK_SIZE << ")");
            return nullptr;
        }

        // Validate k_size alignment (must be multiple of BLOCK_SIZE=32, or extend to end)
        const size_t cols = shape_[1];
        const size_t k_end = k_start + k_size;
        const bool extends_to_end = (k_end == cols);
        const bool is_tail_aligned = (k_size % Q8_1Block::BLOCK_SIZE == 0) || extends_to_end;

        if (!is_tail_aligned)
        {
            LOG_ERROR("[Q8_1Tensor::slice_k_blocks] k_size=" << k_size
                                                             << " is not aligned (must be multiple of "
                                                             << Q8_1Block::BLOCK_SIZE << " or extend to end of tensor)");
            return nullptr;
        }

        // Validate bounds
        if (k_end > cols)
        {
            LOG_ERROR("[Q8_1Tensor::slice_k_blocks] slice exceeds bounds: k_start=" << k_start
                                                                                    << ", k_size=" << k_size
                                                                                    << ", cols=" << cols);
            return nullptr;
        }

        // Calculate block counts
        const size_t rows = shape_[0];
        const size_t src_blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t dst_blocks_per_row = (k_size + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t start_block_idx = k_start / Q8_1Block::BLOCK_SIZE;

        // Create new Q8_1Tensor with sliced shape
        std::vector<size_t> new_shape = {rows, k_size};
        auto result = std::make_shared<Q8_1Tensor>(new_shape, device_.toLegacyIndex());

        // Get source and destination block pointers
        const uint8_t *src_data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q8_1Block *src_blocks = reinterpret_cast<const Q8_1Block *>(src_data_ptr);
        Q8_1Block *dst_blocks = result->mutable_typed_data();

        // Copy blocks row by row (cannot be a zero-copy view since we're slicing columns)
#pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row)
        {
            const Q8_1Block *src_row = src_blocks + row * src_blocks_per_row + start_block_idx;
            Q8_1Block *dst_row = dst_blocks + row * dst_blocks_per_row;
            std::memcpy(dst_row, src_row, dst_blocks_per_row * sizeof(Q8_1Block));
        }

        LOG_TRACE("[Q8_1Tensor::slice_k_blocks] Created k-slice: [" << rows << ", " << k_size
                                                                    << "] from [" << rows << ", " << cols
                                                                    << "] at k_start=" << k_start
                                                                    << " (" << dst_blocks_per_row << " blocks/row)");

        return result;
    }

} // namespace llaminar2
