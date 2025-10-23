#pragma once

#include <vector>
#include <memory>
#include "kernels/TensorKernels.h"
#include "MPIContext.h"
#include "ComputeBackend.h"

namespace llaminar
{

/**
 * @brief Data type enumeration
 */
enum class TensorDataType
{
    FP32,
    FP16,
    BF16,
    INT8,
    QUANTIZED  // IQ4_NL, Q8_0, etc.
};

/**
 * @brief Base tensor interface for inference
 * 
 * Minimal interface focused on:
 * 1. Metadata access (shape, type)
 * 2. Kernel creation (fused operations)
 * 3. Optional data access (for non-fused ops)
 */
class TensorBase
{
public:
    virtual ~TensorBase() = default;

    // Metadata
    virtual const std::vector<size_t>& shape() const = 0;
    virtual TensorDataType native_type() const = 0;
    
    // Kernel creation (cached instances returned)
    // Returns nullptr if kernel not supported for this tensor type
    virtual ITensorGemm* createGemm() = 0;
    virtual ITensorRoPE* createRoPE() = 0;
    virtual ITensorSwiGLU* createSwiGLU() = 0;
    virtual ITensorSoftmax* createSoftmax() = 0;
    virtual ITensorRMSNorm* createRMSNorm() = 0;
    
    /**
     * @brief Get FP32 data pointer (for non-fused operations)
     * 
     * For SimpleTensor: Returns direct pointer (zero-copy)
     * For QuantizedTensor: Returns cached decoded buffer (expensive!)
     * 
     * Prefer using kernels over data() access for performance.
     */
    virtual const float* data() = 0;

    /**
     * @brief Get device affinity for this tensor
     * 
     * Returns device index (from DeviceManager) where this tensor resides.
     * -1 = host memory (CPU)
     * ≥0 = device memory (GPU)
     * 
     * Enables multi-GPU support:
     * - Different tensors can be on different GPUs
     * - Kernels execute on tensor's resident device
     * - Pipeline orchestrates cross-device transfers if needed
     */
    virtual int device_index() const = 0;

    /**
     * @brief Set device affinity (upload tensor to device)
     * 
     * @param device_idx Target device index (-1 = host)
     * @return true on success, false on error
     */
    virtual bool set_device(int device_idx) = 0;

    /**
     * @brief Check if tensor is resident on device
     * 
     * @param device_idx Device index to check
     * @return true if tensor data is on device, false otherwise
     */
    virtual bool is_on_device(int device_idx) const = 0;
};

/**
 * @brief Simple FP32 tensor (host or device)
 * 
 * Supports:
 * - Host memory (CPU inference)
 * - Device memory (GPU inference, any backend)
 * - Lazy migration between host/device
 */
class SimpleTensor : public TensorBase
{
public:
    /**
     * @brief Create tensor on host (CPU)
     */
    SimpleTensor(const std::vector<size_t>& shape);
    
    /**
     * @brief Create tensor on specific device
     * @param shape Tensor dimensions
     * @param device_idx Device index from DeviceManager (-1 = host)
     */
    SimpleTensor(const std::vector<size_t>& shape, int device_idx);
    
    ~SimpleTensor();
    
    const std::vector<size_t>& shape() const override { return shape_; }
    TensorDataType native_type() const override { return TensorDataType::FP32; }
    
    // Kernels (SimpleTensor uses standard BLAS kernels, no quantization)
    ITensorGemm* createGemm() override;
    ITensorRoPE* createRoPE() override;
    ITensorSwiGLU* createSwiGLU() override;
    ITensorSoftmax* createSoftmax() override;
    ITensorRMSNorm* createRMSNorm() override;
    
    /**
     * @brief Get host data pointer (copies from device if needed)
     * 
     * For CPU tensors: Direct pointer (zero-copy)
     * For GPU tensors: Downloads to staging buffer, returns staging pointer
     */
    const float* data() override;
    
    /**
     * @brief Get mutable host data pointer
     * 
     * Marks device data as stale (requires upload on next device operation)
     */
    float* mutable_data();
    
    /**
     * @brief Get device pointer (for direct kernel access)
     * 
     * @return Device pointer if on GPU, nullptr if on host
     */
    void* device_data();
    
    int device_index() const override { return device_idx_; }
    bool set_device(int device_idx) override;
    bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

private:
    std::vector<size_t> shape_;
    int device_idx_ = -1;  // -1 = host, ≥0 = device index
    
    // Host storage (always allocated for CPU tensors, staging buffer for GPU tensors)
    std::vector<float> host_data_;
    
    // Device storage (only allocated if device_idx ≥ 0)
    void* device_data_ = nullptr;
    
    // Synchronization flags
    bool host_dirty_ = false;    // Host modified, needs upload
    bool device_dirty_ = false;  // Device modified, needs download
    
    // Cached kernel instances (per device)
    std::map<int, std::unique_ptr<ITensorGemm>> gemm_kernels_;
    std::map<int, std::unique_ptr<ITensorRoPE>> rope_kernels_;
    std::map<int, std::unique_ptr<ITensorSwiGLU>> swiglu_kernels_;
    std::map<int, std::unique_ptr<ITensorSoftmax>> softmax_kernels_;
    std::map<int, std::unique_ptr<ITensorRMSNorm>> rmsnorm_kernels_;
    
    void sync_to_device();
    void sync_from_device();
};

/**
 * @brief Quantized tensor with fused streaming dequantization
 * 
 * Example: IQ4_NLTensor, Q8_0Tensor, etc.
 * Never stores decoded FP32 data - always streams during kernel execution.
 */
class IQ4_NLTensor : public TensorBase
{
public:
    IQ4_NLTensor(const std::vector<size_t>& shape, const uint8_t* raw_blocks);
    
    const std::vector<size_t>& shape() const override { return shape_; }
    TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
    
    // Kernels with fused streaming dequantization
    ITensorGemm* createGemm() override;          // IQ4_NLQuantizedGemm (already exists!)
    ITensorRoPE* createRoPE() override;          // Future: IQ4_NLRoPEKernel
    ITensorSwiGLU* createSwiGLU() override;      // Future: IQ4_NLSwiGLUKernel
    ITensorSoftmax* createSoftmax() override;    // nullptr (activations not quantized)
    ITensorRMSNorm* createRMSNorm() override;    // nullptr (activations not quantized)
    
    /**
     * @brief Fallback FP32 data access (expensive - avoid!)
     * 
     * Fully decodes tensor on first access and caches result.
     * Only use for debugging or non-performance-critical paths.
     */
    const float* data() override;
    
    int device_index() const override { return device_idx_; }
    bool set_device(int device_idx) override;
    bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

private:
    std::vector<size_t> shape_;
    std::vector<uint8_t> raw_blocks_;  // Quantized data (host)
    int device_idx_ = -1;  // -1 = host only, ≥0 = uploaded to device
    
    // Device storage (if uploaded to GPU)
    void* device_blocks_ = nullptr;  // Quantized blocks on device
    
    // Cached kernel instances (per device)
    std::map<int, std::unique_ptr<ITensorGemm>> gemm_kernels_;
    
    // Lazy decode cache (avoid using!)
    mutable std::vector<float> decoded_cache_;
    mutable bool decoded_ = false;
};

} // namespace llaminar
