/**
 * @file TensorSlice.h
 * @brief Tensor slice wrapper for tensor parallelism weight sharding
 *
 * TensorSlice wraps any TensorBase and provides metadata about how it was
 * sliced from a larger tensor. The underlying tensor contains the data in
 * its original quantized format (no dequantization).
 *
 * Key benefits:
 * 1. Preserves quantized format - no dequantization needed for slicing
 * 2. Stores sharding metadata (mode, original dims, slice range)
 * 3. GEMM kernels can query slice info for correct dimension handling
 *
 * Usage:
 * @code
 * // Create a row-sliced tensor via TensorFactory
 * auto slice = TensorFactory::createRowParallelSlice(
 *     std::move(full_tensor), rank, world_size);
 *
 * // Slice knows its sharding context
 * assert(slice->mode() == SliceMode::ROW_PARALLEL);
 * assert(slice->original_rows() == 4096);
 * assert(slice->slice_rows() == 2048);  // Half the rows
 *
 * // GEMM kernel works transparently
 * auto gemm = slice->createGemm();  // Creates kernel for slice
 * @endcode
 *
 * @author David Sanftenberg
 * @date 2025-12-03
 */

#pragma once

#include "Tensors.h"
#include "../backends/DeviceId.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "../utils/Logger.h"
#include <memory>
#include <cstring>

namespace llaminar2
{

    /**
     * @brief Slice mode indicating how tensor was partitioned
     */
    enum class SliceMode
    {
        FULL,           ///< Not sliced (full tensor, row_start=0, row_end=N)
        ROW_PARALLEL,   ///< Rows sliced: this tensor has rows [row_start, row_end) of original
        COLUMN_PARALLEL ///< Columns sliced: this tensor has cols [col_start, col_end) of original
    };

    /**
     * @brief Metadata about a tensor slice for tensor parallelism
     */
    struct SliceMetadata
    {
        SliceMode mode = SliceMode::FULL;

        // Original full tensor dimensions before slicing
        size_t original_rows = 0; ///< Full N (output features) before slicing
        size_t original_cols = 0; ///< Full K (input features) before slicing

        // Slice range (in original tensor coordinates)
        size_t slice_start = 0; ///< First row/col index in original tensor
        size_t slice_end = 0;   ///< One past last row/col index

        // MPI context
        int rank = 0;       ///< MPI rank that owns this slice
        int world_size = 1; ///< Total number of MPI ranks

        // Storage mode: does inner tensor have full data or just slice?
        // - inner_is_presliced=false: Inner has full [N, K] data, slice during kernel creation
        // - inner_is_presliced=true:  Inner has only [slice_rows, K] data, no further slicing
        bool inner_is_presliced = false;

        /**
         * @brief Get the number of rows/cols in this slice
         */
        size_t slice_size() const { return slice_end - slice_start; }

        /**
         * @brief Check if this is a full (unsliced) tensor
         */
        bool is_full() const { return mode == SliceMode::FULL; }

        /**
         * @brief Create metadata for a row-parallel slice
         */
        static SliceMetadata forRowParallel(
            size_t original_rows, size_t original_cols,
            int rank, int world_size,
            bool inner_is_presliced = false)
        {
            SliceMetadata meta;
            meta.mode = SliceMode::ROW_PARALLEL;
            meta.original_rows = original_rows;
            meta.original_cols = original_cols;

            // Compute this rank's slice range
            size_t rows_per_rank = original_rows / world_size;
            meta.slice_start = rows_per_rank * rank;

            // Last rank gets remainder
            if (rank == world_size - 1)
            {
                meta.slice_end = original_rows;
            }
            else
            {
                meta.slice_end = meta.slice_start + rows_per_rank;
            }

            meta.rank = rank;
            meta.world_size = world_size;
            meta.inner_is_presliced = inner_is_presliced;
            return meta;
        }

        /**
         * @brief Create metadata for a column-parallel slice
         */
        static SliceMetadata forColumnParallel(
            size_t original_rows, size_t original_cols,
            int rank, int world_size,
            bool inner_is_presliced = false)
        {
            SliceMetadata meta;
            meta.mode = SliceMode::COLUMN_PARALLEL;
            meta.original_rows = original_rows;
            meta.original_cols = original_cols;

            // Compute this rank's slice range
            size_t cols_per_rank = original_cols / world_size;
            meta.slice_start = cols_per_rank * rank;

            // Last rank gets remainder
            if (rank == world_size - 1)
            {
                meta.slice_end = original_cols;
            }
            else
            {
                meta.slice_end = meta.slice_start + cols_per_rank;
            }

            meta.rank = rank;
            meta.world_size = world_size;
            meta.inner_is_presliced = inner_is_presliced;
            return meta;
        }
    };

    /**
     * @brief Tensor slice wrapper for tensor parallelism
     *
     * Wraps any TensorBase-derived type and adds slice metadata.
     * The inner tensor can contain either:
     * - Full tensor data (inner_is_presliced=false): slicing happens during kernel creation
     * - Only slice data (inner_is_presliced=true): memory efficient, requires sliced loading
     *
     * Also implements IINT8Unpackable by forwarding to inner tensor if it supports it.
     * This enables QuantisedGemmKernel to work with sliced quantized tensors.
     */
    class TensorSlice : public TensorBase, public IINT8Unpackable
    {
    public:
        /**
         * @brief Construct from unique_ptr (takes ownership)
         */
        TensorSlice(std::unique_ptr<TensorBase> inner, SliceMetadata metadata)
            : inner_(std::move(inner)), metadata_(std::move(metadata))
        {
            if (!inner_)
            {
                throw std::invalid_argument("TensorSlice: inner tensor cannot be null");
            }
        }

        /**
         * @brief Construct from shared_ptr (shared ownership)
         */
        TensorSlice(std::shared_ptr<TensorBase> inner, SliceMetadata metadata)
            : inner_shared_(std::move(inner)), metadata_(std::move(metadata))
        {
            if (!inner_shared_)
            {
                throw std::invalid_argument("TensorSlice: inner tensor cannot be null");
            }
        }

        ~TensorSlice() override = default;

        // =======================================================================
        // Slice Metadata Access
        // =======================================================================

        SliceMode mode() const { return metadata_.mode; }
        const SliceMetadata &metadata() const { return metadata_; }
        size_t original_rows() const { return metadata_.original_rows; }
        size_t original_cols() const { return metadata_.original_cols; }
        size_t slice_start() const { return metadata_.slice_start; }
        size_t slice_end() const { return metadata_.slice_end; }

        size_t slice_rows() const
        {
            return (metadata_.mode == SliceMode::ROW_PARALLEL)
                       ? metadata_.slice_size()
                       : inner()->shape()[0];
        }

        size_t slice_cols() const
        {
            return (metadata_.mode == SliceMode::COLUMN_PARALLEL)
                       ? metadata_.slice_size()
                       : inner()->shape()[1];
        }

        bool is_full() const { return metadata_.is_full(); }
        bool is_row_parallel() const { return metadata_.mode == SliceMode::ROW_PARALLEL; }
        bool is_column_parallel() const { return metadata_.mode == SliceMode::COLUMN_PARALLEL; }

        // =======================================================================
        // Inner Tensor Access
        // =======================================================================

        TensorBase *inner()
        {
            return inner_ ? inner_.get() : inner_shared_.get();
        }

        const TensorBase *inner() const
        {
            return inner_ ? inner_.get() : inner_shared_.get();
        }

        // =======================================================================
        // TensorBase Interface (delegate to inner)
        // =======================================================================

        TensorType native_type() const override { return inner()->native_type(); }
        const std::vector<size_t> &shape() const override { return inner()->shape(); }
        DeviceId home_device() const override { return inner()->home_device(); }
        bool is_on_device(DeviceId device) const override { return inner()->is_on_device(device); }
        const float *data() const override { return inner()->data(); }
        float *mutable_data() override { return inner()->mutable_data(); }
        bool copyFrom(const TensorBase *src) override { return inner()->copyFrom(src); }

        // =======================================================================
        // GPU Transfer Support (delegate to inner) - Required for ensureOnDevice()
        // =======================================================================
        // Note: TensorSlice is a friend of CPUTensorBase, so it can access
        // protected methods like byte_size() and raw_host_data_ptr() on inner().

        size_t size_bytes() const override { return inner()->byte_size(); }

    protected:
        /**
         * @brief Get raw host data pointer for GPU transfer
         *
         * TensorSlice delegates directly to inner tensor's protected raw_host_data_ptr().
         * This works because TensorSlice is a friend of CPUTensorBase.
         */
        void *raw_host_data_ptr() override
        {
            return inner()->raw_host_data_ptr();
        }

        const void *raw_host_data_ptr() const override
        {
            return inner()->raw_host_data_ptr();
        }

        /**
         * @brief Return byte size by delegating to inner's protected byte_size()
         */
        size_t byte_size() const override { return inner()->byte_size(); }

    public:
        // =======================================================================
        // Unified FP32 Access Interface (Phase 1 Infrastructure - delegate to inner)
        // =======================================================================
        float *mutable_fp32_data() override { return inner()->mutable_fp32_data(); }
        bool is_fp32_backed() const override { return inner()->is_fp32_backed(); }
        bool is_view() const override { return true; } // TensorSlice is always a view/wrapper

        // =======================================================================
        // Type Conversion (delegate to inner)
        // =======================================================================

        void to_fp32(float *dst) const override { inner()->to_fp32(dst); }
        void to_bf16(uint16_t *dst) const override { inner()->to_bf16(dst); }
        void to_fp16(uint16_t *dst) const override { inner()->to_fp16(dst); }

        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override
        {
            inner()->to_int8_blocked(dst_int8, dst_scales, block_size);
        }

        bool to_int8_perchannel(int8_t *dst_int8, float *row_scales, float *col_scales) const override
        {
            return inner()->to_int8_perchannel(dst_int8, row_scales, col_scales);
        }

        void to_fp32_row(size_t row_idx, float *buffer) const override
        {
            inner()->to_fp32_row(row_idx, buffer);
        }

        void to_fp32_span(size_t offset, size_t count, float *buffer) const override
        {
            inner()->to_fp32_span(offset, count, buffer);
        }

        std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &new_shape, size_t offset = 0) override
        {
            return inner()->create_view(new_shape, offset);
        }

        // =======================================================================
        // Kernel Creation
        // =======================================================================

        /**
         * @brief Create a GEMM kernel for this slice
         *
         * For row-parallel slices with full inner data, creates a QuantisedGemmKernel
         * that only packs the slice rows. Otherwise delegates to inner tensor.
         *
         * IMPORTANT: When inner tensor's raw data has been released (after first
         * GEMM packing), we can't re-pack. In that case, we throw an error.
         * Callers should use KernelFactory::getOrCreateGemm() to get the cached kernel.
         */
        std::unique_ptr<ITensorGemm> createGemm() override
        {
            if (metadata_.mode == SliceMode::ROW_PARALLEL && !metadata_.inner_is_presliced)
            {
                // Inner has full data - create a row-sliced kernel directly
                // Check if raw data has been released
                if (inner()->is_raw_data_released())
                {
                    LOG_ERROR("TensorSlice: Cannot create row-sliced kernel - inner tensor's raw data was released. "
                              "Use KernelFactory::getOrCreateGemm() instead of createGemm().");
                    throw std::runtime_error("Cannot create GEMM kernel: raw weight data was released. Use KernelFactory::getOrCreateGemm().");
                }
                LOG_DEBUG("TensorSlice: Creating row-sliced kernel ["
                          << metadata_.slice_start << ", " << metadata_.slice_end
                          << ") from full inner tensor");
                return std::make_unique<gemm_v4::QuantisedGemmKernel>(
                    inner(),
                    static_cast<int>(metadata_.slice_start),
                    static_cast<int>(metadata_.slice_end));
            }

            // Inner is already presliced (or FULL/COLUMN_PARALLEL) - use full kernel
            if (metadata_.inner_is_presliced)
            {
                // Check if raw data has been released
                if (inner()->is_raw_data_released())
                {
                    LOG_ERROR("TensorSlice: Cannot create kernel - inner tensor's raw data was released. "
                              "Use KernelFactory::getOrCreateGemm() instead of createGemm().");
                    throw std::runtime_error("Cannot create GEMM kernel: raw weight data was released. Use KernelFactory::getOrCreateGemm().");
                }
                LOG_DEBUG("TensorSlice: Inner already presliced, using full kernel for "
                          << metadata_.slice_size() << " rows");
            }
            return inner()->createGemm();
        }

        // =======================================================================
        // IINT8Unpackable Interface (forward to inner tensor)
        // =======================================================================

        /**
         * @brief Check if inner tensor supports INT8 unpacking
         */
        bool supports_int8_unpack() const
        {
            return dynamic_cast<const IINT8Unpackable *>(inner()) != nullptr;
        }

        /**
         * @brief Unpack a single block to INT8 values
         */
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            auto *unpackable = dynamic_cast<const IINT8Unpackable *>(inner());
            if (unpackable)
            {
                unpackable->unpack_block_to_int8(row_idx, k_block_offset, output);
            }
            else
            {
                // Fallback: zero-fill if inner doesn't support INT8 unpacking
                LOG_WARN("TensorSlice: inner tensor (type " << static_cast<int>(inner()->native_type())
                                                            << ") does not implement IINT8Unpackable");
                std::memset(output, 0, 32 * sizeof(int8_t)); // block_size = 32
            }
        }

        /**
         * @brief Get the scale factor for a block
         */
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            auto *unpackable = dynamic_cast<const IINT8Unpackable *>(inner());
            if (unpackable)
            {
                return unpackable->get_block_scale(row_idx, k_block_offset);
            }
            return 1.0f; // Default scale
        }

        /**
         * @brief Get the minimum value for a block (for asymmetric quant)
         */
        float get_block_min(size_t row_idx, size_t k_block_offset) const override
        {
            auto *unpackable = dynamic_cast<const IINT8Unpackable *>(inner());
            if (unpackable)
            {
                return unpackable->get_block_min(row_idx, k_block_offset);
            }
            return 0.0f; // Default min (symmetric)
        }

        /**
         * @brief Get the superblock size
         */
        size_t superblock_size() const override
        {
            auto *unpackable = dynamic_cast<const IINT8Unpackable *>(inner());
            if (unpackable)
            {
                return unpackable->superblock_size();
            }
            return 32; // Default to block size
        }

        /**
         * @brief Unpack a full superblock to INT8 with scales and mins
         */
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx,
                                       int8_t *output, float *scales, float *mins) const override
        {
            auto *unpackable = dynamic_cast<const IINT8Unpackable *>(inner());
            if (unpackable)
            {
                unpackable->unpack_superblock_to_int8(row_idx, superblock_idx, output, scales, mins);
            }
            else
            {
                // Fallback: use default implementation from base
                IINT8Unpackable::unpack_superblock_to_int8(row_idx, superblock_idx, output, scales, mins);
            }
        }

        // =======================================================================
        // Memory Management (forward to inner tensor)
        // =======================================================================

        /**
         * @brief Release raw data from inner tensor after GEMM packing
         *
         * Forwards to inner tensor's release_raw_data() to free the original
         * quantized weight data after the GEMM kernel has repacked it.
         */
        void release_raw_data() override { inner()->release_raw_data(); }
        bool is_raw_data_released() const override { return inner()->is_raw_data_released(); }

    private:
        std::unique_ptr<TensorBase> inner_;
        std::shared_ptr<TensorBase> inner_shared_;
        SliceMetadata metadata_;
    };

} // namespace llaminar2
