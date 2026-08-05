/**
 * @file GpuTensorView.h
 * @brief Lightweight tensor view for external GPU memory
 *
 * This class provides an ITensor interface around existing GPU memory,
 * without owning or managing the memory lifetime. It is used by CUDA and
 * ROCm cache adapters to expose persistent device storage to graph stages.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "ITensor.h"
#include "TensorType.h"
#include "BlockStructures.h"
#include "../backends/DeviceId.h"
#include <vector>
#include <stdexcept>

namespace llaminar2
{
    class KVCacheAppendStage;

    /**
     * @brief Tensor view wrapping external GPU memory
     *
     * Does NOT own the GPU memory - the caller is responsible for ensuring
     * the memory remains valid for the lifetime of this view.
     *
     * Host operations (data(), mutable_data()) return nullptr since
     * this is a pure GPU view - data stays on device.
     */
    class GpuTensorView : public ITensor
    {
    public:
        /**
         * @brief Construct a backend-qualified pure-device tensor view.
         *
         * A bare integer ordinal is deliberately not accepted. CUDA and ROCm
         * ordinals overlap, so accepting an integer lets a valid HIP pointer be
         * mislabeled as CUDA memory. Requiring DeviceId makes backend ownership
         * part of the view's type-level construction contract.
         *
         * @param gpu_ptr Pointer to persistent GPU memory owned by the caller.
         * @param rows Number of logical rows exposed by the view.
         * @param cols Number of logical columns exposed by the view.
         * @param tensor_type Tensor element or block type.
         * @param device Backend-qualified device that owns @p gpu_ptr.
         */
        GpuTensorView(void *gpu_ptr, size_t rows, size_t cols,
                      TensorType tensor_type, DeviceId device)
            : gpu_ptr_(gpu_ptr),
              rows_(rows),
              cols_(cols),
              tensor_type_(tensor_type),
              device_(device),
              shape_({rows, cols})
        {
            if (!gpu_ptr || !device.is_gpu())
            {
                throw std::invalid_argument(
                    "GpuTensorView requires non-null device storage and an "
                    "explicit CUDA or ROCm DeviceId");
            }
        }

        /**
         * @brief Reject backend-ambiguous device ordinals at compile time.
         *
         * Callers must spell DeviceId::cuda(ordinal) or
         * DeviceId::rocm(ordinal), making it impossible to silently attach the
         * wrong backend identity to otherwise valid device storage.
         */
        GpuTensorView(void *, size_t, size_t, TensorType, int) = delete;

        ~GpuTensorView() override = default;

        // Non-copyable but movable (view semantics)
        GpuTensorView(const GpuTensorView &) = delete;
        GpuTensorView &operator=(const GpuTensorView &) = delete;
        GpuTensorView(GpuTensorView &&) = default;
        GpuTensorView &operator=(GpuTensorView &&) = default;

        /**
         * @brief Update the view's GPU pointer and row count in-place.
         *
         * This keeps the same object identity (stable address) while updating
         * the GPU pointer and dimensions. Essential for KV cache views where
         * the graph stores raw ITensor* pointers and the underlying data may
         * change between iterations (e.g., after KV cache append).
         */
        void update_view(void *new_gpu_ptr, size_t new_rows)
        {
            gpu_ptr_ = new_gpu_ptr;
            rows_ = new_rows;
            shape_[0] = new_rows;
        }

        // =========================================================================
        // ITensor interface implementation
        // =========================================================================

        // Type info
        int native_type_id() const override { return static_cast<int>(tensor_type_); }

        // Shape
        const std::vector<size_t> &shape() const override { return shape_; }
        size_t numel() const override { return rows_ * cols_; }
        size_t size_bytes() const override { return numel() * element_size_for_type(tensor_type_); }

        // Device
        DeviceId home_device() const override { return device_; }
        std::optional<DeviceId> current_device() const override { return device_; }
        bool is_on_cpu() const override { return false; }
        bool is_on_gpu() const override { return true; }

        // GPU data access - returns the wrapped pointer
        void *gpu_data_ptr() override { return gpu_ptr_; }
        const void *gpu_data_ptr() const override { return gpu_ptr_; }

        // Host data access - not available for GPU-only view
        const float *data() const override { return nullptr; }
        float *mutable_data() override { return nullptr; }
        const void *raw_data() const override { return nullptr; }
        void *raw_mutable_data() override { return nullptr; }

        // Device-aware access returns GPU pointer
        const void *active_data_ptr() const override { return gpu_ptr_; }
        void *active_mutable_data_ptr() override { return gpu_ptr_; }

        // Coherence - GPU view is always device-valid
        bool isDeviceValid() const override { return true; }
        bool isHostValid() const override { return false; }

        // Conversion - not supported for GPU-only view
        void to_fp32(float *dst) const override
        {
            (void)dst;
            // Cannot dequantize GPU data from a pure view
            // Caller should use GPU kernels instead
            throw std::runtime_error("GpuTensorView::to_fp32() not supported - data is on GPU");
        }

    private:
        void *gpu_ptr_;
        size_t rows_;
        size_t cols_;
        TensorType tensor_type_;
        DeviceId device_;
        std::vector<size_t> shape_;

        static size_t element_size_for_type(TensorType t)
        {
            switch (t)
            {
            case TensorType::FP32:
                return 4;
            case TensorType::FP16:
                return 2;
            case TensorType::BF16:
                return 2;
            case TensorType::Q8_1:
                return sizeof(Q8_1Block);
            default:
                return 4; // Fallback
            }
        }
    };

    /**
     * @brief Non-owning view whose producer has already been joined to one stream.
     *
     * This type exists for the narrow case where a compute stage first orders a
     * coherence-aware parent tensor on its executor stream and then passes a
     * pointer-offset slice to a device kernel. The slice itself cannot own a
     * second coherence record, so it carries the exact backend-qualified device
     * and stream on which the parent was prepared.
     *
     * Only KVCacheAppendStage may construct this view. Cache adapters must call
     * isPreparedFor() before consuming it and must never route it back through
     * TransferEngine as though the non-owning wrapper were a TensorBase owner.
     * Ordinary GpuTensorView instances do not carry this authority.
     */
    class PreparedGpuTensorView final : public GpuTensorView
    {
    public:
        /**
         * @brief Verify the consumer is the exact producer-ordered boundary.
         *
         * @param device Backend-qualified device that will consume the slice.
         * @param stream Exact consumer stream used by the preparing stage.
         */
        bool isPreparedFor(DeviceId device, void *stream) const noexcept
        {
            return device == prepared_device_ &&
                   stream != nullptr &&
                   stream == prepared_stream_;
        }

    private:
        friend class KVCacheAppendStage;

        PreparedGpuTensorView(
            void *gpu_ptr,
            size_t rows,
            size_t cols,
            TensorType tensor_type,
            DeviceId device,
            void *prepared_stream)
            : GpuTensorView(gpu_ptr, rows, cols, tensor_type, device),
              prepared_device_(device),
              prepared_stream_(prepared_stream)
        {
            if (!gpu_ptr || !device.is_gpu() || !prepared_stream)
            {
                throw std::invalid_argument(
                    "PreparedGpuTensorView requires device storage, a GPU device, "
                    "and the exact non-null producer stream");
            }
        }

        DeviceId prepared_device_;
        void *prepared_stream_ = nullptr;
    };

} // namespace llaminar2
