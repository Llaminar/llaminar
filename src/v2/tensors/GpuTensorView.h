/**
 * @file GpuTensorView.h
 * @brief Lightweight tensor view for external GPU memory
 *
 * This class provides an ITensor interface around existing GPU memory,
 * without owning or managing the memory lifetime. Used primarily by
 * CUDA KV cache to return tensor pointers for attention computation.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "ITensor.h"
#include "TensorType.h"
#include "../backends/DeviceId.h"
#include <vector>
#include <stdexcept>

namespace llaminar2
{

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
         * @brief Construct a GPU tensor view
         *
         * @param gpu_ptr Pointer to GPU memory (must remain valid!)
         * @param rows Number of rows (e.g., seq_len for KV cache)
         * @param cols Number of columns (e.g., kv_dim for KV cache)
         * @param tensor_type Data type (TensorType::FP32, TensorType::FP16, TensorType::BF16)
         * @param device_id CUDA device ID where the memory resides
         */
        GpuTensorView(void *gpu_ptr, size_t rows, size_t cols,
                      TensorType tensor_type, int device_id = 0)
            : gpu_ptr_(gpu_ptr), rows_(rows), cols_(cols), tensor_type_(tensor_type), device_id_(device_id), shape_({rows, cols})
        {
        }

        ~GpuTensorView() override = default;

        // Non-copyable but movable (view semantics)
        GpuTensorView(const GpuTensorView &) = delete;
        GpuTensorView &operator=(const GpuTensorView &) = delete;
        GpuTensorView(GpuTensorView &&) = default;
        GpuTensorView &operator=(GpuTensorView &&) = default;

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
        DeviceId home_device() const override { return DeviceId::cuda(device_id_); }
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
        int device_id_;
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
            default:
                return 4; // Fallback
            }
        }
    };

} // namespace llaminar2
