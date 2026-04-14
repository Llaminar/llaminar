/**
 * @file CUDABackend.h
 * @brief CUDA backend public API (no CUDA headers exposed)
 *
 * **Purpose**: Public interface for CUDA backend. Implementation lives in .cu file
 * to avoid exposing cuda_runtime.h to other compilation units.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../IBackend.h"
#include <future>
#include <vector>

namespace llaminar2
{

    /**
     * @class CUDABackend
     * @brief CUDA compute backend implementation
     *
     * **Implementation**: See CUDABackend.cu
     * **Requirements**: NVIDIA GPU with CUDA Toolkit 11.0+
     * **Compilation**: Requires nvcc compiler, -DHAVE_CUDA=ON
     */
    class CUDABackend : public IBackend
    {
    public:
        CUDABackend();
        ~CUDABackend() override;

        // Memory transfer operations (see IBackend documentation)
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) override;
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) override;
        bool synchronize(int device_id) override;
        bool streamSynchronize(int device_id) override;
        bool setDevice(int device_id) override;

        // Async variants of transfer/sync operations
        std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> synchronizeAsync(int device_id) override;

        // Event operations (fine-grained synchronization)
        void *createEvent(int device_id) override;
        void destroyEvent(void *event, int device_id) override;
        bool recordEvent(void *event, int device_id, void *stream = nullptr) override;
        bool waitForEvent(void *event, int device_id) override;

        // Memory allocation operations
        void *allocate(size_t bytes, int device_id) override;
        void free(void *ptr, int device_id) override;
        bool memset(void *ptr, int value, size_t bytes, int device_id) override;

        // Async variants of allocation operations
        std::future<void *> allocateAsync(size_t bytes, int device_id) override;
        std::future<void> freeAsync(void *ptr, int device_id) override;
        std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) override;

        // Zero-copy mapped memory operations
        void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override;
        void freeMapped(void *host_ptr, int device_id) override;

        // Device query operations
        int deviceCount() const override;
        std::string backendName() const override;
        std::string deviceName(int device_id) const override;
        size_t deviceMemoryTotal(int device_id) const override;
        size_t deviceMemoryFree(int device_id) const override;

        // Host memory pinning for async DMA
        bool pinHostMemory(void *ptr, size_t bytes) override;
        bool unpinHostMemory(void *ptr) override;

        // GPU-side argmax for greedy sampling
        bool argmaxF32(const void *data_device, int n, int device_id,
                       float *out_value, int *out_index) override;

        // GPU-side top-k selection for sampling
        bool topKF32(const void *data_device, int n, int k, int device_id,
                     float *out_values, int *out_indices) override;

        // Capability queries
        bool supportsBF16(int device_id) const override;
        bool supportsFP16(int device_id) const override;
        bool supportsINT8(int device_id) const override;

        // Compute operations
        bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) override;

        // Stream management
        void *createStream(int device_id) override;
        void destroyStream(void *stream, int device_id) override;
        bool synchronizeStream(void *stream, int device_id) override;
        bool streamWaitEvent(void *stream, void *event, int device_id) override;

        // Stream-aware memory operations
        bool deviceCopyAsync(void *dst, const void *src, size_t bytes,
                             int device_id, void *stream = nullptr) override;

        // Collective reduction primitives
        bool vectorAddInplace(void *output, const void *input, size_t count,
                              int element_size, int device_id, void *stream = nullptr) override;

        // Backend identity
        DeviceType backendDeviceType() const override { return DeviceType::CUDA; }

    private:
        int device_count_;

        // Per-device argmax result buffers (lazily allocated)
        struct ArgmaxDeviceBuffers
        {
            void *value_ptr = nullptr;
            void *index_ptr = nullptr;
        };
        std::vector<ArgmaxDeviceBuffers> argmax_buffers_;

        // Per-device top-k result buffers (lazily allocated)
        struct TopKDeviceBuffers
        {
            void *values_ptr = nullptr;
            void *indices_ptr = nullptr;
            int allocated_k = 0;
        };
        std::vector<TopKDeviceBuffers> topk_buffers_;
    };

} // namespace llaminar2
