/**
 * @file ROCmBackend.h
 * @brief ROCm/HIP backend public API (no HIP headers exposed)
 *
 * **Purpose**: Public interface for ROCm backend. Implementation lives in .cpp file
 * to avoid exposing hip_runtime.h to other compilation units.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../IBackend.h"

namespace llaminar2
{

    /**
     * @class ROCmBackend
     * @brief ROCm/HIP compute backend implementation
     *
     * **Implementation**: See ROCmBackend.cpp
     * **Requirements**: AMD GPU with ROCm 5.0+
     * **Compilation**: Requires hipcc compiler, -DHAVE_ROCM=ON
     */
    class ROCmBackend : public IBackend
    {
    public:
        ROCmBackend();
        ~ROCmBackend() override;

        // Memory transfer operations (see IBackend documentation)
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) override;
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) override;
        bool synchronize(int device_id) override;
        bool setDevice(int device_id) override;

        // Memory allocation operations
        void *allocate(size_t bytes, int device_id) override;
        void free(void *ptr, int device_id) override;

        // Device query operations
        int deviceCount() const override;
        std::string backendName() const override;
        std::string deviceName(int device_id) const override;
        size_t deviceMemoryTotal(int device_id) const override;
        size_t deviceMemoryFree(int device_id) const override;

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

    private:
        int device_count_;
    };

} // namespace llaminar2
